# Standard library
import json
import random
import time
from argparse import ArgumentParser
from pathlib import Path

# Third-party
# for logging the model:
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities import seed
from loguru import logger

# Local
from . import utils
from .config import load_config_and_datastores
from .models import GraphLAM, HiLAM, HiLAMParallel
from .weather_dataset import WeatherDataModule

# Custom callback to stop training at specific epoch
class StopAtEpochCallback(pl.callbacks.Callback):
    def __init__(self, stop_epoch):
        self.stop_epoch = stop_epoch

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.stop_epoch:
            trainer.should_stop = True


class ModelSwitchCallback(pl.callbacks.Callback):
    def __init__(self, models):
        self.models = models

    def on_train_epoch_end(self, trainer, pl_module):
        next_epoch = trainer.current_epoch + 1
        if next_epoch < trainer.max_epochs:
            next_idx = next_epoch % len(self.models)
            import ipdb; ipdb.set_trace()
            graph_name = "triangular2D_split13_DOM0{next_idx}".format(next_idx=next_idx)
            pl_module.update(self.models[next_idx], graph_name, pl_module.args, datastore=self.models[next_idx]._datastore, config=self.models[next_idx]._config)


class CyclingWeatherDataModule(pl.LightningDataModule):
    def __init__(self, data_modules):
        super().__init__()
        self.data_modules = data_modules
        self.trainer = None

    def train_dataloader(self):
        epoch = 0
        if self.trainer:
            epoch = self.trainer.current_epoch
        idx = epoch % len(self.data_modules)
        self.data_modules[idx].setup("fit")
        return self.data_modules[idx].train_dataloader()

    def val_dataloader(self):
        epoch = 0
        if self.trainer:
            epoch = self.trainer.current_epoch
        idx = epoch % len(self.data_modules)
        return self.data_modules[idx].val_dataloader()

    def test_dataloader(self):
        epoch = 0
        if self.trainer:
            epoch = self.trainer.current_epoch
        idx = epoch % len(self.data_modules)
        return self.data_modules[idx].test_dataloader()


MODELS = {
    "graph_lam": GraphLAM,
    "hi_lam": HiLAM,
    "hi_lam_parallel": HiLAMParallel,
}


@logger.catch
def main(input_args=None):
    """Main function for training and evaluating models."""
    parser = ArgumentParser(
        description="Train or evaluate NeurWP models for LAM"
    )
    parser.add_argument(
        "--config_paths",
        nargs="+",
        type=str,
        help="Paths to the configurations for neural-lam",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="graph_lam",
        help="Model architecture to train/evaluate (default: graph_lam)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers in data loader (default: 4)",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes to use in DDP (default: 1)",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        type=str,
        default=["auto"],
        help="Devices to use for training. Can be the string 'auto' or a list "
        "of integer id's corresponding to the desired devices, e.g. "
        "'--devices 0 1'. Note that this cannot be used with SLURM, instead "
        "set 'ntasks-per-node' in the slurm setup (default: auto)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="upper epoch limit (default: 200)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="batch size (default: 4)"
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Path to load model parameters from (default: None)",
    )
    parser.add_argument(
        "--restore_opt",
        action="store_true",
        help="If full training state should be restored with model "
        "(default: false)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=32,
        help="Numerical precision to use for model (32/16/bf16) (default: 32)",
    )
    parser.add_argument(
        "--num_sanity_steps",
        type=int,
        default=2,
        help="Number of sanity checking validation steps to run before starting"
        " training (default: 2)",
    )

    # Model architecture
    parser.add_argument(
        "--graph_name",
        type=str,
        default="multiscale",
        help="Graph to load and use in graph-based model (default: multiscale)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Dimensionality of hidden representations (default: 64)",
    )
    parser.add_argument(
        "--hidden_dim_grid",
        type=int,
        help=(
            "Dimensionality of hidden representations related to grid nodes "
            "(grid encodings and in grid-level MLPs)"
            "(default: None, use same as hidden_dim)"
        ),
    )
    parser.add_argument(
        "--hidden_layers",
        type=int,
        default=1,
        help="Number of hidden layers in all MLPs (default: 1)",
    )
    parser.add_argument(
        "--processor_layers",
        type=int,
        default=4,
        help="Number of GNN layers in processor GNN (default: 4)",
    )
    parser.add_argument(
        "--mesh_aggr",
        type=str,
        default="sum",
        help="Aggregation to use for m2m processor GNN layers (sum/mean) "
        "(default: sum)",
    )
    parser.add_argument(
        "--output_std",
        action="store_true",
        help="If models should additionally output std.-dev. per "
        "output dimensions "
        "(default: False (no))",
    )
    parser.add_argument(
        "--shared_grid_embedder",
        action="store_true",  # Default to separate embedders
        help="If the same embedder MLP should be used for interior and boundary"
        " grid nodes. Note that this requires the same dimensionality for "
        "both kinds of grid inputs. (default: False (no))",
    )
    parser.add_argument(
        "--time_delta_enc_dim",
        type=int,
        help="Dimensionality of positional encoding for time deltas of boundary"
        " forcing. If None, same as hidden_dim. If given, must be even "
        "(default: None)",
    )
    parser.add_argument(
        "--dynamic_time_deltas",
        action="store_true",
        help="If models should use dynamically computed time-deltas between"
        "interior and boundary time steps (default: False (no))",
    )

    # Training options
    parser.add_argument(
        "--ar_steps_train",
        type=int,
        default=1,
        help="Number of steps to unroll prediction for in loss function "
        "(default: 1)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="wmse",
        help="Loss function to use, see metric.py (default: wmse)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-4,
        help="Minimum learning rate for cosine annealing (default: 1e-4)",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Number of epochs training between each validation run "
        "(default: 1)",
    )
    parser.add_argument(
        "--grad_checkpointing",
        action="store_true",
        help="If gradient checkpointing should be used in-between each "
        "unrolling step (default: false)",
    )

    # Evaluation options
    parser.add_argument(
        "--eval",
        type=str,
        help="Eval model on given data split (val/test) "
        "(default: None (train model))",
    )
    parser.add_argument(
        "--ar_steps_eval",
        type=int,
        default=10,
        help="Number of steps to unroll prediction for during evaluation "
        "(default: 10)",
    )
    parser.add_argument(
        "--n_example_pred",
        type=int,
        default=1,
        help="Number of example predictions to plot during evaluation "
        "(default: 1)",
    )
    parser.add_argument(
        "--eval_init_times",
        nargs="*",
        default=[0, 12],
        help="List of init times (UTC) where validation and evaluation "
        "forecasts should be started from (default: 0, 12)",
    )
    parser.add_argument(
        "--save_eval_to_zarr_path",
        type=str,
        help="Save evaluation results to zarr dataset at given path ",
    )
    parser.add_argument(
        "--plot_vars",
        nargs="+",
        type=str,
        default=["t2m"],
        help="List of variables to plot during eval (default: t2m)",
    )

    # Logger Settings
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices=["wandb", "mlflow"],
        help="Logger to use for training (wandb/mlflow) (default: wandb)",
    )
    parser.add_argument(
        "--logger-project",
        type=str,
        default="neural_lam",
        help="Logger project name, for eg. Wandb (default: neural_lam)",
    )
    parser.add_argument(
        "--logger_run_name",
        type=str,
        default=None,
        help="Logger run name, for e.g. MLFlow (with default value `None` neural-lam default format string is used)",
    )
    parser.add_argument(
        "--val_steps_to_log",
        nargs="+",
        type=int,
        default=[1, 2, 3, 5, 10, 15, 19],
        help="Steps to log val loss for (default: 1 2 3 5 10 15 19)",
    )
    parser.add_argument(
        "--metrics_watch",
        nargs="+",
        default=[],
        help="List of metrics to watch, including any prefix (e.g. val_rmse)",
    )
    parser.add_argument(
        "--var_leads_metrics_watch",
        type=str,
        default="{}",
        help="""JSON string with variable-IDs and lead times to log watched
             metrics (e.g. '{"1": [1, 2], "3": [3, 4]}')""",
    )
    parser.add_argument(
        "--num_past_forcing_steps",
        type=int,
        default=1,
        help="Number of past time steps to use as forcing input (default: 1)",
    )
    parser.add_argument(
        "--num_future_forcing_steps",
        type=int,
        default=1,
        help="Number of future time steps to use as forcing input (default: 1)",
    )
    parser.add_argument(
        "--num_past_boundary_steps",
        type=int,
        default=1,
        help="Number of past time steps to use as boundary input (default: 1)",
    )
    parser.add_argument(
        "--num_future_boundary_steps",
        type=int,
        default=1,
        help="Number of future time steps to use as boundary input "
        "(default: 1)",
    )
    args = parser.parse_args(input_args)
    args.var_leads_metrics_watch = {
        int(k): v for k, v in json.loads(args.var_leads_metrics_watch).items()
    }

    # Asserts for arguments
    assert (
        len(args.config_paths) > 0
    ), "Specify at least one config with --config_paths"
    assert args.model in MODELS, f"Unknown model: {args.model}"
    assert args.eval in (
        None,
        "val",
        "test",
    ), f"Unknown eval setting: {args.eval}"
    for step in args.val_steps_to_log:
        assert step <= args.ar_steps_eval, (
            f"Can not log validation step {step} when validation is "
            f"only unrolled {args.ar_steps_eval} steps."
        )
    assert (
        args.load or not args.restore_opt
    ), "Can not restore opt state when not loading a checkpoint"

    # Get an (actual) random run id as a unique identifier
    random_run_id = random.randint(0, 9999)

    # Set seed
    seed.seed_everything(args.seed)

    # Load all neural-lam configurations and datastores
    configs = []
    datastores = []
    datastore_boundaries = []
    for config_path in args.config_paths:
        c, ds, dsb = load_config_and_datastores(config_path)
        configs.append(c)
        datastores.append(ds)
        datastore_boundaries.append(dsb)

    # Create datamodules and models for each config
    data_modules = []
    models = []
    ModelClass = MODELS[args.model]
    for i in range(len(configs)):
        config = configs[i]
        datastore = datastores[i]
        datastore_boundary = datastore_boundaries[i]
        data_module = WeatherDataModule(
            datastore=datastore,
            datastore_boundary=datastore_boundary,
            ar_steps_train=args.ar_steps_train,
            ar_steps_eval=args.ar_steps_eval,
            standardize=True,
            num_past_forcing_steps=args.num_past_forcing_steps,
            num_future_forcing_steps=args.num_future_forcing_steps,
            num_past_boundary_steps=args.num_past_boundary_steps,
            num_future_boundary_steps=args.num_future_boundary_steps,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            # Make sure that dataset provided for eval contains correct split
            eval_split=args.eval if args.eval is not None else "test",
            eval_init_times=args.eval_init_times,
            dynamic_time_deltas=args.dynamic_time_deltas,
            excluded_intervals=config.training.excluded_intervals,
        )
        data_modules.append(data_module)

        if args.load and not args.restore_opt:
            if i == 0:  # load for first model
                model = ModelClass.load_from_checkpoint(
                    args.load,
                    args=args,
                    config=config,
                    datastore=datastore,
                    datastore_boundary=datastore_boundary,
                )
            else:
                model = ModelClass(
                    args,
                    config=config,
                    datastore=datastore,
                    datastore_boundary=datastore_boundary,
                )
        else:
            model = ModelClass(
                args,
                config=config,
                datastore=datastore,
                datastore_boundary=datastore_boundary,
            )
        models.append(model)

    # Instantiate trainer
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.set_float32_matmul_precision(
            "high"
        )  # Allows using Tensor Cores on A100s
    else:
        device_name = "cpu"

    # Set devices to use
    if args.devices == ["auto"]:
        devices = "auto"
    else:
        try:
            devices = [int(i) for i in args.devices]
        except ValueError:
            raise ValueError("devices should be 'auto' or a list of integers")

    if args.eval:
        prefix = f"eval-{args.eval}-"
    else:
        prefix = "train-"

    if args.logger_run_name:
        run_name = args.logger_run_name
    elif args.load:
        last_ckpt = torch.load(args.load, weights_only=False)
        path_last_ckpt = Path(list(last_ckpt['callbacks'].values())[0]['last_model_path'])
        run_name = path_last_ckpt.parts[-2]
        if args.eval:
            run_name = run_name.replace("train-","eval-")
    else:
        run_name = (
            f"{prefix}{args.model}-{args.processor_layers}x{args.hidden_dim}-"
            f"{time.strftime('%m_%d_%H')}-{random_run_id:04d}"
        )

    # Checkpoint each 2 epochs
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"saved_models/{run_name}",
        filename="{epoch:03d}-{step:07d}",
        save_top_k=-1,
        every_n_epochs=2,
        save_last=True,
    )

    training_logger = utils.setup_training_logger(
        datastore=datastores[0], args=args, run_name=run_name
    )

    cycling_data_module = CyclingWeatherDataModule(data_modules)
    cycling_data_module.trainer = None  # will be set by trainer later

    callbacks = [checkpoint_callback, ModelSwitchCallback(models)]
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        deterministic=True,
        strategy="ddp",
        accelerator=device_name,
        num_nodes=args.num_nodes,
        devices=devices,
        logger=training_logger,
        log_every_n_steps=1,
        callbacks=callbacks,
        check_val_every_n_epoch=args.val_interval,
        precision=args.precision,
        num_sanity_val_steps=0,
        reload_dataloaders_every_n_epochs=1,
    )

    # Only init once, on rank 0 only
    if trainer.global_rank == 0:
        utils.init_training_logger_metrics(
            training_logger, val_steps=args.val_steps_to_log
        )  # Do after initializing logger
    if args.eval:
        # For evaluation, use cycling data module for train, but since eval, use first
        trainer.test(
            model=models[0],
            datamodule=data_modules[0],
            ckpt_path=args.load,
        )
    else:
        # Train with cycling data and model switching per epoch
        ckpt_for_fit = args.load if args.restore_opt else None
        trainer.fit(model=models[0], datamodule=cycling_data_module, ckpt_path=ckpt_for_fit)


if __name__ == "__main__":
    main()
