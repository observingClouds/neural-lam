"""CLI entry point for training Neural-LAM models."""

# Standard library
import dataclasses
import faulthandler
import json
import os
import random
import shutil
import signal
import sys
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from typing import Any, cast

# Third-party
# for logging the model:
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities import seed
from loguru import logger

# Local
from . import utils
from .config import (
    ComputeConfig,
    DataConfig,
    DatastoreSelection,
    LoggingConfig,
    ManualStateFeatureWeighting,
    ModelConfig,
    NeuralLAMConfig,
    OutputClamping,
    QuantileMatching,
    TrainingConfig,
    TrainRunConfig,
    UniformFeatureWeighting,
    load_config_and_datastore,
)
from .datastore.base import BaseDatastore
from .gnn_layers import GNN_TYPES
from .models import (
    MODELS,
    BaseGraphModel,
    BaseHiGraphModel,
    DeterministicARForecaster,
    DeterministicForecastingModule,
)
from .weather_dataset import WeatherDataModule

# Models taking graph-specific options (e.g. boundary forcing)
GRAPH_MODELS = ("graph_lam", "hi_lam", "hi_lam_parallel")

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals(
        [
            Namespace,
            DatastoreSelection,
            ManualStateFeatureWeighting,
            NeuralLAMConfig,
            OutputClamping,
            QuantileMatching,
            TrainingConfig,
            UniformFeatureWeighting,
        ]
    )


@dataclasses.dataclass(frozen=True)
class Run:
    """Handle containing output paths for a Neural-LAM run."""

    run_dir: Path
    checkpoint_path: Path | None = None

    @property
    def checkpoint_dir(self) -> Path:
        """Directory holding ``min_val_loss.ckpt`` and ``last.ckpt``."""
        return self.run_dir / "checkpoints"

    @property
    def plot_dir(self) -> Path:
        """Directory evaluation plots are written to (logger ``save_dir``)."""
        return self.run_dir

    @property
    def rmse_plot(self) -> Path:
        """Lead time x variable RMSE heatmap written by test evaluation."""
        return self.plot_dir / "test_rmse.pdf"

    def example_plots(self, example_i: int) -> Path:
        """Frames and GIF of example ``example_i`` (1-based, create_gif)."""
        return self.plot_dir / f"example_plots_{example_i}"


def build_predictor(
    predictor_class: type,
    args: Namespace,
    config: NeuralLAMConfig,
    datastore: BaseDatastore,
    datastore_boundary: BaseDatastore | None = None,
    additional_domains: (
        list[tuple[BaseDatastore, BaseDatastore | None, str]] | None
    ) = None,
) -> Any:
    """
    Instantiate a step predictor with the GNN kwargs its family accepts.

    Hierarchical GNN kwargs are only passed to ``BaseHiGraphModel``
    subclasses, gating on the class hierarchy so that future hierarchical
    models are covered without maintaining a model-name list. GNN type
    arguments fall back to ``InteractionNet`` for checkpoints saved before
    those CLI flags existed. Graph models are boundary-forced when a
    ``datastore_boundary`` is given.
    """
    kwargs = dict(
        datastore=datastore,
        graph_name=args.graph,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        processor_layers=args.processor_layers,
        mesh_aggr=args.mesh_aggr,
        num_past_forcing_steps=args.num_past_forcing_steps,
        num_future_forcing_steps=args.num_future_forcing_steps,
        output_std=args.output_std,
        output_clamping_lower=config.training.output_clamping.lower,
        output_clamping_upper=config.training.output_clamping.upper,
        g2m_gnn_type=getattr(args, "g2m_gnn_type", "InteractionNet"),
        m2g_gnn_type=getattr(args, "m2g_gnn_type", "InteractionNet"),
    )
    if issubclass(predictor_class, BaseHiGraphModel):
        kwargs["mesh_up_gnn_type"] = getattr(
            args, "mesh_up_gnn_type", "InteractionNet"
        )
        kwargs["mesh_down_gnn_type"] = getattr(
            args, "mesh_down_gnn_type", "InteractionNet"
        )
    if issubclass(predictor_class, BaseGraphModel):
        output_clamping = config.training.output_clamping
        kwargs.update(
            datastore_boundary=datastore_boundary,
            num_past_boundary_steps=getattr(args, "num_past_boundary_steps", 1),
            num_future_boundary_steps=getattr(
                args, "num_future_boundary_steps", 1
            ),
            hidden_dim_grid=getattr(args, "hidden_dim_grid", None),
            time_delta_enc_dim=getattr(args, "time_delta_enc_dim", None),
            shared_grid_embedder=getattr(args, "shared_grid_embedder", False),
            num_quantiles=getattr(args, "num_quantiles", 0),
            gated_features=getattr(output_clamping, "gated_features", None),
            gate_thresholds=getattr(output_clamping, "gate_thresholds", None),
            graph_dir=getattr(args, "graph_dir", None),
            additional_domains=additional_domains,
        )
    return predictor_class(**kwargs)


class AdaptiveHelpFormatter(ArgumentDefaultsHelpFormatter):
    """``--help`` formatter that scales the column width to the terminal."""

    def __init__(self, prog: str) -> None:
        """Pick a help-column width based on the current terminal size."""
        terminal_width = shutil.get_terminal_size(fallback=(100, 20)).columns
        width = max(80, min(terminal_width, 120))
        help_position = min(44, width // 3)
        super().__init__(
            prog,
            max_help_position=help_position,
            width=width,
        )


def load_forecasting_module_from_checkpoint(
    ckpt_path: str,
    config: NeuralLAMConfig,
    datastore: BaseDatastore,
    datastore_boundary: BaseDatastore | None = None,
) -> DeterministicForecastingModule:
    """
    Reconstruct a DeterministicForecastingModule from a checkpoint without
    requiring the caller to know the original architecture kwargs.

    The checkpoint must have been saved with args in hyper_parameters (i.e.
    created via train_model.main), so that model class and architecture kwargs
    can be recovered automatically.

    Parameters
    ----------
    ckpt_path : str
        Path to the checkpoint.
    config : NeuralLAMConfig
        Neural-LAM configuration.
    datastore : BaseDatastore
        Interior datastore.
    datastore_boundary : BaseDatastore, optional
        Boundary datastore, needed to restore the boundary standardization
        statistics of a checkpoint trained with one.

    Returns
    -------
    DeterministicForecastingModule
        The restored module.
    """
    ckpt = torch.load(ckpt_path, weights_only=False)
    args = ckpt["hyper_parameters"]["args"]
    predictor_class = MODELS[args.model]
    predictor = build_predictor(
        predictor_class, args, config, datastore, datastore_boundary
    )
    forecaster = DeterministicARForecaster(
        predictor,
        datastore,
        config=config,
        loss=args.loss,
        grad_checkpointing=getattr(args, "grad_checkpointing", False),
    )
    return DeterministicForecastingModule.load_from_checkpoint(
        ckpt_path,
        forecaster=forecaster,
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        weights_only=False,
    )


def build_parser() -> ArgumentParser:
    """Build the argument parser for training and evaluating models.

    Returns
    -------
    ArgumentParser
        The parser.
    """

    parser = ArgumentParser(
        description="Train or evaluate MLWP models for LAM",
        formatter_class=AdaptiveHelpFormatter,
    )

    # Core Configuration
    core_group = parser.add_argument_group("Core Configuration")
    core_group.add_argument(
        "--config_path",
        type=str,
        help="Path to the configuration for neural-lam (required unless "
        "--config_paths is given)",
    )
    core_group.add_argument(
        "--config_paths",
        nargs="+",
        type=str,
        default=None,
        help="Paths to the configurations of several domains to train on "
        "together, paired with --graph_names. The first one is used for "
        "validation and testing.",
    )
    core_group.add_argument(
        "--model",
        type=str,
        default="graph_lam",
        help="Model architecture to train/evaluate",
        choices=MODELS.keys(),
    )
    core_group.add_argument("--seed", type=int, default=42, help="random seed")

    # Runtime & Device Settings
    runtime_group = parser.add_argument_group("Runtime & Device Settings")
    runtime_group.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers in data loader",
    )
    runtime_group.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes to use in DDP",
    )
    runtime_group.add_argument(
        "--devices",
        nargs="+",
        type=str,
        default=["auto"],
        help="Devices to use for training. Can be the string 'auto' or a list "
        "of integer id's corresponding to the desired devices, e.g. "
        "'--devices 0 1'. Note that this cannot be used with SLURM, instead "
        "set 'ntasks-per-node' in the slurm setup",
    )
    runtime_group.add_argument(
        "--precision",
        type=str,
        default=32,
        help="Numerical precision to use for model (32/16/bf16)",
    )
    runtime_group.add_argument(
        "--load",
        type=str,
        help="Path to load model parameters from",
    )
    runtime_group.add_argument(
        "--restore_opt",
        action="store_true",
        help="If optimizer state should be restored with model",
    )

    # Model architecture
    arch_group = parser.add_argument_group("Model Architecture")
    arch_group.add_argument(
        "--graph",
        type=str,
        default="multiscale",
        help="Graph to load and use in graph-based model",
    )
    arch_group.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Dimensionality of all hidden representations",
    )
    arch_group.add_argument(
        "--hidden_layers",
        type=int,
        default=1,
        help="Number of hidden layers in all MLPs",
    )
    arch_group.add_argument(
        "--processor_layers",
        type=int,
        default=4,
        help="Number of GNN layers in processor GNN",
    )
    arch_group.add_argument(
        "--mesh_aggr",
        type=str,
        default="sum",
        help="Aggregation to use for m2m processor GNN layers (sum/mean)",
    )
    arch_group.add_argument(
        "--output_std",
        action="store_true",
        help="If models should additionally output std.-dev. per "
        "output dimensions",
    )
    arch_group.add_argument(
        "--g2m_gnn_type",
        type=str,
        default="InteractionNet",
        choices=list(GNN_TYPES.keys()),
        help="GNN type for grid-to-mesh encoding. Applies to all models, "
        "including the probabilistic Graph-EFM model",
    )
    arch_group.add_argument(
        "--m2g_gnn_type",
        type=str,
        default="InteractionNet",
        choices=list(GNN_TYPES.keys()),
        help="GNN type for mesh-to-grid decoding. Applies to all models, "
        "including the probabilistic Graph-EFM model",
    )
    arch_group.add_argument(
        "--mesh_up_gnn_type",
        type=str,
        default="InteractionNet",
        choices=list(GNN_TYPES.keys()),
        help="GNN type for upward mesh message passing in hierarchical "
        "models. Only affects Hi-LAM; the probabilistic Graph-EFM model "
        "hard-codes its mesh-up GNN types",
    )
    arch_group.add_argument(
        "--mesh_down_gnn_type",
        type=str,
        default="InteractionNet",
        choices=list(GNN_TYPES.keys()),
        help="GNN type for downward mesh message passing in hierarchical "
        "models. Only affects Hi-LAM; the probabilistic Graph-EFM model "
        "hard-codes its mesh-down GNN type",
    )

    # Training options
    train_group = parser.add_argument_group("Training Options")
    train_group.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="upper epoch limit",
    )
    train_group.add_argument(
        "--batch_size", type=int, default=4, help="batch size"
    )

    train_group.add_argument(
        "--ar_steps_train",
        type=int,
        default=1,
        help="Number of steps to unroll prediction for in loss function",
    )
    train_group.add_argument(
        "--loss",
        type=str,
        default="wmse",
        help="Loss function to use, see metric.py",
    )
    train_group.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate"
    )
    train_group.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Number of epochs training between each validation run",
    )

    train_group.add_argument(
        "--num_sanity_val_steps",
        type=int,
        default=2,
        help="Number of sanity validation steps to run before training",
    )

    # Evaluation options
    eval_group = parser.add_argument_group("Evaluation Options")
    eval_group.add_argument(
        "--eval",
        type=str,
        help="Eval model on given data split (val/test). If not given, "
        "train model.",
        choices=["val", "test"],
    )
    eval_group.add_argument(
        "--ar_steps_eval",
        type=int,
        default=10,
        help="Number of steps to unroll prediction for during evaluation",
    )
    eval_group.add_argument(
        "--n_example_pred",
        type=int,
        default=1,
        help="Number of example predictions to plot during evaluation",
    )
    eval_group.add_argument(
        "--create_gif",
        action="store_true",
        help="If set, create GIF animations from prediction PNG frames and "
        "save to disk. PNGs are always created and logged to wandb/mlflow.",
    )

    # Logger Settings
    logger_group = parser.add_argument_group("Logger Settings")
    logger_group.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices=["wandb", "mlflow"],
        help="Logger to use for training (wandb/mlflow)",
    )
    logger_group.add_argument(
        "--logger-project",
        type=str,
        default="neural_lam",
        help="Logger project name, for eg. Wandb",
    )
    logger_group.add_argument(
        "--logger_run_name",
        type=str,
        default=None,
        help="""Logger run name, for e.g. MLFlow (with default value `None`
          neural-lam's default format string is used)""",
    )
    parser.add_argument(
        "--runs_root",
        type=str,
        default="runs",
        help="Root directory under which per-run output dirs (checkpoints, "
        "logger files, plots) are written as `<runs_root>/<run_name>/`",
    )

    logger_group.add_argument(
        "--wandb_id",
        type=str,
        default=None,
        help="Wandb run ID to use. If the run ID already exists in the "
        "project, W&B resumes that run. If it does not exist, W&B creates "
        "a new run with that ID. Useful on HPC systems with limited job "
        "runtimes or that may crash, allowing training to be continued "
        "across multiple job submissions.",
    )

    # Metrics & Monitoring (logger-agnostic: applies to both wandb and mlflow)
    metrics_group = parser.add_argument_group("Metrics & Monitoring")
    metrics_group.add_argument(
        "--val_steps_to_log",
        nargs="+",
        type=int,
        default=[1, 2, 3, 5, 10],
        help="Steps to log val loss for",
    )
    metrics_group.add_argument(
        "--train_steps_to_log",
        nargs="+",
        type=int,
        default=[],
        help="Steps to log train loss for during training (optional)",
    )
    metrics_group.add_argument(
        "--metrics_watch",
        nargs="+",
        default=[],
        help="List of metrics to watch, including any prefix (e.g. val_rmse)",
    )
    metrics_group.add_argument(
        "--var_leads_metrics_watch",
        type=str,
        default="{}",
        help="""JSON string with variable-IDs and lead times to log watched
             metrics (e.g. '{"1": [1, 2], "3": [3, 4]}')""",
    )

    # Data Loading & Forcing
    data_group = parser.add_argument_group("Data Loading & Forcing")
    data_group.add_argument(
        "--num_past_forcing_steps",
        type=int,
        default=1,
        help="Number of past time steps to use as input for forcing data",
    )
    data_group.add_argument(
        "--num_future_forcing_steps",
        type=int,
        default=1,
        help="Number of future time steps to use as input for forcing data",
    )
    data_group.add_argument(
        "--num_past_boundary_steps",
        type=int,
        default=1,
        help="Number of past time steps to use as input for boundary forcing, "
        "when present",
    )
    data_group.add_argument(
        "--num_future_boundary_steps",
        type=int,
        default=1,
        help="Number of future time steps to use as input for boundary "
        "forcing, when present",
    )
    data_group.add_argument(
        "--load_single_member",
        action="store_true",
        help=(
            "If set, only use ensemble member 0 instead of treating all "
            "ensemble members as independent samples."
        ),
    )
    # Options from the boundary-forcing research branch
    arch_group.add_argument(
        "--graph_names",
        nargs="+",
        type=str,
        default=None,
        help="Graphs of the domains given by --config_paths, in the same order",
    )
    arch_group.add_argument(
        "--graph_dir",
        type=str,
        default=None,
        help="Root directory containing graph directories "
        "(default: <datastore root>/graph)",
    )
    arch_group.add_argument(
        "--hidden_dim_grid",
        type=int,
        default=None,
        help="Dimensionality of hidden representations related to grid "
        "nodes (grid encodings and grid-level MLPs). Default: hidden_dim",
    )
    arch_group.add_argument(
        "--num_quantiles",
        type=int,
        default=0,
        help="Number of quantiles to predict per output dimension, use "
        "with --loss crps_quantile (0: no quantiles)",
    )
    arch_group.add_argument(
        "--shared_grid_embedder",
        action="store_true",
        help="Use the same embedder MLP for interior and boundary grid "
        "nodes, requiring the same input dimensionality for both",
    )
    arch_group.add_argument(
        "--time_delta_enc_dim",
        type=int,
        default=None,
        help="Dimensionality of the positional encoding of boundary forcing "
        "time deltas, must be even. Default: hidden_dim",
    )
    data_group.add_argument(
        "--dynamic_time_deltas",
        action="store_true",
        help="Compute the boundary time deltas relative to the interior "
        "state times instead of the matched boundary times",
    )
    train_group.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["constant", "linear_cosine"],
        help="Learning rate schedule. linear_cosine chains a linear warm-up "
        "with cosine annealing over --epochs scheduler steps",
    )
    train_group.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        help="Minimum learning rate of the cosine annealing",
    )
    train_group.add_argument(
        "--lr_warmup_iters",
        type=int,
        default=10,
        help="Number of scheduler steps of linear warm-up (linear_cosine)",
    )
    train_group.add_argument(
        "--lr_scheduler_frequency",
        type=int,
        default=700,
        help="Number of optimizer steps between learning rate scheduler "
        "steps (linear_cosine)",
    )
    train_group.add_argument(
        "--grad_checkpointing",
        action="store_true",
        help="Use gradient checkpointing in-between each unrolling step",
    )
    train_group.add_argument(
        "--log_variable_rmse",
        action="store_true",
        help="Log the first-step RMSE of each state variable and the learning "
        "rate during training and validation",
    )
    train_group.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=None,
        help="Additionally keep a checkpoint every this many epochs",
    )
    eval_group.add_argument(
        "--eval_init_times",
        nargs="*",
        type=int,
        default=[],
        help="UTC hours that validation and evaluation forecasts are "
        "started from, e.g. '0 12' (default: all)",
    )
    eval_group.add_argument(
        "--save_eval_to_zarr_path",
        type=str,
        default=None,
        help="Save evaluation predictions to a zarr dataset at this path",
    )
    eval_group.add_argument(
        "--plot_vars",
        nargs="+",
        type=str,
        default=[],
        help="State variables to plot example predictions for (default: all)",
    )
    logger_group.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level of loguru",
    )
    return parser


def fit(
    model_config: ModelConfig,
    train_config: TrainRunConfig,
    data_config: DataConfig,
    compute_config: ComputeConfig,
    logging_config: LoggingConfig,
    *,
    config: NeuralLAMConfig | None = None,
    datastore: BaseDatastore | None = None,
    datastore_boundary: BaseDatastore | None = None,
    config_path: str | None = None,
) -> Run:
    """
    Run training or evaluation using strongly-typed configuration dataclasses.

    Parameters
    ----------
    model_config : ModelConfig
        Model architecture and graph configuration.
    train_config : TrainRunConfig
        Training hyperparameters and evaluation settings.
    data_config : DataConfig
        Data loading and forcing parameters.
    compute_config : ComputeConfig
        Compute devices, seed, and precision settings.
    logging_config : LoggingConfig
        Experiment tracking and logging settings.
    config : NeuralLAMConfig or None, optional
        Loaded Neural-LAM configuration.
    datastore : BaseDatastore or None, optional
        Initialized datastore instance.
    datastore_boundary : BaseDatastore or None, optional
        Initialized boundary datastore, used together with ``config`` and
        ``datastore``. Loaded from ``config_path`` otherwise.
    config_path : str or None, optional
        Path to the configuration file, used if config or datastore is None.

    Returns
    -------
    Run
        Object containing paths to run outputs and saved checkpoints.
    """
    for phase, max_steps in [
        ("train", train_config.ar_steps_train),
        ("val", train_config.ar_steps_eval),
    ]:
        steps = (
            train_config.train_steps_to_log
            if phase == "train"
            else train_config.val_steps_to_log
        )
        for step in steps:
            if not 1 <= step <= max_steps:
                raise ValueError(
                    f"Can not log {phase} step {step}: must be between 1 "
                    f"and {max_steps}, the number of unrolled steps during "
                    f"{phase} phase. Adjust --{phase}_steps_to_log."
                )

    for var_i, leads in train_config.var_leads_metrics_watch.items():
        for step in leads:
            if not 1 <= step <= train_config.ar_steps_eval:
                raise ValueError(
                    f"Can not log validation step {step} for variable "
                    f"{var_i}: must be between 1 and "
                    f"{train_config.ar_steps_eval}, the number of unrolled "
                    "validation steps. Adjust --var_leads_metrics_watch."
                )
    # Watched metrics are only logged when both are set
    if bool(train_config.metrics_watch) != bool(
        train_config.var_leads_metrics_watch
    ):
        missing = (
            "var_leads_metrics_watch"
            if train_config.metrics_watch
            else "metrics_watch"
        )
        raise ValueError(
            "metrics_watch and var_leads_metrics_watch must be set together, "
            f"but {missing} is not set."
        )

    if train_config.eval and not train_config.load:
        logger.warning(
            "Evaluation (--eval) without --load: no checkpoint will be loaded.",
        )

    random_run_id = random.randint(0, 9999)
    seed.seed_everything(compute_config.seed, workers=True)

    if config is None and datastore is None:
        if config_path is None:
            raise ValueError(
                "Either (config and datastore) or config_path must be provided."
            )
        config, datastore, datastore_boundary = load_config_and_datastore(
            config_path=config_path
        )
    elif config is None or datastore is None:
        raise ValueError("config and datastore must be provided together.")

    # Check --var_leads_metrics_watch variable indices against the datastore
    # so users get an immediate error instead of an IndexError deep in the
    # first validation epoch.
    state_var_names = datastore.get_vars_names(category="state")
    for var_i in train_config.var_leads_metrics_watch:
        if not 0 <= var_i < len(state_var_names):
            raise ValueError(
                f"Invalid state variable index {var_i} in "
                f"--var_leads_metrics_watch. Index must be between 0 and "
                f"{len(state_var_names) - 1} (datastore has "
                f"{len(state_var_names)} state variables)."
            )

    # Further domains to train on (multi-domain training)
    if len(data_config.additional_config_paths) != len(
        model_config.additional_graphs
    ):
        raise ValueError(
            "One graph name is needed per domain: got "
            f"{len(data_config.additional_config_paths)} additional configs "
            f"but {len(model_config.additional_graphs)} additional graphs."
        )
    additional_domains = []
    for extra_config_path, extra_graph in zip(
        data_config.additional_config_paths, model_config.additional_graphs
    ):
        _, extra_datastore, extra_datastore_boundary = (
            load_config_and_datastore(config_path=extra_config_path)
        )
        additional_domains.append(
            (extra_datastore, extra_datastore_boundary, extra_graph)
        )

    data_module = WeatherDataModule(
        datastore=datastore,
        ar_steps_train=train_config.ar_steps_train,
        ar_steps_eval=train_config.ar_steps_eval,
        num_past_forcing_steps=data_config.num_past_forcing_steps,
        num_future_forcing_steps=data_config.num_future_forcing_steps,
        num_past_boundary_steps=data_config.num_past_boundary_steps,
        num_future_boundary_steps=data_config.num_future_boundary_steps,
        datastore_boundary=datastore_boundary,
        load_single_member=data_config.load_single_member,
        batch_size=train_config.batch_size,
        num_workers=data_config.num_workers,
        eval_split=train_config.eval or "test",
        add_boundary_time_deltas=datastore_boundary is not None
        and model_config.model in GRAPH_MODELS,
        dynamic_time_deltas=data_config.dynamic_time_deltas,
        eval_init_times=train_config.eval_init_times,
        excluded_intervals=config.training.excluded_intervals,
        additional_domains=[
            (extra_ds, extra_dsb)
            for extra_ds, extra_dsb, _ in additional_domains
        ],
    )

    # Instantiate model + trainer
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.set_float32_matmul_precision(
            "high"
        )  # Allows using Tensor Cores on A100s
    else:
        device_name = "cpu"

    devices: int | str | list[int]
    if compute_config.devices == "auto" or compute_config.devices == ["auto"]:
        devices = "auto"
    elif isinstance(compute_config.devices, list):
        try:
            devices = [int(i) for i in compute_config.devices]
        except ValueError:
            raise ValueError("devices should be 'auto' or a list of integers")
    else:
        devices = compute_config.devices

    hparams = Namespace(
        **dataclasses.asdict(model_config),
        **dataclasses.asdict(train_config),
        **dataclasses.asdict(data_config),
        **dataclasses.asdict(compute_config),
        **dataclasses.asdict(logging_config),
    )

    # Build predictor and forecaster externally, then inject into
    # DeterministicForecastingModule
    predictor_class = MODELS[model_config.model]
    predictor = build_predictor(
        predictor_class,
        hparams,
        config,
        datastore,
        datastore_boundary=datastore_boundary,
        additional_domains=additional_domains or None,
    )

    if not train_config.eval and not predictor.trainable:
        raise ValueError(
            f"The {model_config.model} model cannot be trained. Run with "
            "--eval <val/test> to evaluate a baseline."
        )
    forecaster = DeterministicARForecaster(
        predictor,
        datastore,
        config=config,
        loss=train_config.loss,
        grad_checkpointing=train_config.grad_checkpointing,
    )

    model = DeterministicForecastingModule(
        forecaster=forecaster,
        config=config,
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        lr=train_config.lr,
        restore_opt=train_config.restore_opt,
        n_example_pred=train_config.n_example_pred,
        create_gif=train_config.create_gif,
        val_steps_to_log=train_config.val_steps_to_log,
        train_steps_to_log=train_config.train_steps_to_log,
        metrics_watch=train_config.metrics_watch,
        var_leads_metrics_watch=train_config.var_leads_metrics_watch,
        args=hparams,
    )

    prefix = f"eval-{train_config.eval}-" if train_config.eval else "train-"
    if logging_config.logger_run_name:
        run_name = logging_config.logger_run_name
    elif train_config.load and _run_name_from_checkpoint(train_config.load):
        # Continue in the run directory of the loaded checkpoint
        run_name = cast("str", _run_name_from_checkpoint(train_config.load))
        if train_config.eval:
            run_name = f"{run_name}-eval"
    else:
        run_name = (
            f"{prefix}{model_config.model}-"
            f"{model_config.processor_layers}x{model_config.hidden_dim}-"
            f"{time.strftime('%m_%d_%H')}-{random_run_id:04d}"
        )

    run_dir = os.path.join(logging_config.runs_root, run_name)

    training_logger = utils.setup_training_logger(
        datastore=datastore, args=hparams, run_name=run_name, run_dir=run_dir
    )

    # Two separate callbacks decouple "best validated model" from
    # "rescue / resume" snapshots: long HPC jobs that crash or time out
    # between validations can still resume from a recent train-epoch
    # checkpoint instead of losing all progress since the last validation.
    val_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(run_dir, "checkpoints"),
        filename="min_val_loss",
        monitor="val_mean_loss",
        mode="min",
        save_top_k=1,
        save_on_train_epoch_end=False,
    )
    latest_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(run_dir, "checkpoints"),
        filename="last",
        monitor=None,
        save_top_k=1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        enable_version_counter=False,
    )
    callbacks: list[pl.Callback] = [val_checkpoint, latest_checkpoint]
    if train_config.save_every_n_epochs:
        callbacks.append(
            PeriodicModelCheckpoint(
                dirpath=os.path.join(run_dir, "checkpoints"),
                filename="{epoch:03d}-{step:07d}",
                monitor=None,
                save_top_k=-1,
                every_n_epochs=train_config.save_every_n_epochs,
                save_on_train_epoch_end=True,
            )
        )

    trainer = pl.Trainer(
        max_epochs=train_config.epochs,
        deterministic=True,
        default_root_dir=run_dir,
        strategy="auto",
        accelerator=device_name,
        num_nodes=compute_config.num_nodes,
        devices=devices,
        logger=training_logger,
        log_every_n_steps=1,
        callbacks=callbacks,
        check_val_every_n_epoch=train_config.val_interval,
        precision=cast("Any", compute_config.precision),
        num_sanity_val_steps=train_config.num_sanity_val_steps,
    )

    # Only init once, on rank 0 only
    if trainer.global_rank == 0:
        utils.init_training_logger_metrics(
            training_logger, val_steps=train_config.val_steps_to_log
        )  # Do after initializing logger

    if train_config.eval:
        trainer.test(
            model=model,
            datamodule=data_module,
            ckpt_path=train_config.load,
        )
        checkpoint_path = train_config.load
    else:
        trainer.fit(
            model=model,
            datamodule=data_module,
            ckpt_path=train_config.load,
        )
        checkpoint_path = (
            val_checkpoint.best_model_path or latest_checkpoint.best_model_path
        )

    # Close the W&B run so a later train/evaluate call in the same Python
    # session (e.g. a notebook) starts a new run instead of reusing this one.
    # setup_training_logger is rank_zero_only, so other ranks have no logger.
    if isinstance(training_logger, pl.loggers.WandbLogger):
        training_logger.experiment.finish()

    return Run(
        run_dir=Path(run_dir),
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
    )


class PeriodicModelCheckpoint(pl.callbacks.ModelCheckpoint):
    """Checkpoint callback keeping all checkpoints of every n-th epoch, with
    a state key distinct from the other checkpoint callbacks."""

    @property
    def state_key(self) -> str:
        """Unique key of the callback state in checkpoints.

        Returns
        -------
        str
            The state key.
        """
        return f"periodic-{super().state_key}"


def _run_name_from_checkpoint(ckpt_path: str) -> str | None:
    """Run name of a checkpoint stored as ``<runs_root>/<run>/checkpoints``.

    Parameters
    ----------
    ckpt_path : str
        Path to the checkpoint file.

    Returns
    -------
    str or None
        The run name, or ``None`` if the path has another layout.
    """
    path = Path(ckpt_path).resolve()
    if path.parent.name == "checkpoints":
        return path.parent.parent.name
    return None


def run(
    args: Namespace,
    config: NeuralLAMConfig | None = None,
    datastore: BaseDatastore | None = None,
    datastore_boundary: BaseDatastore | None = None,
) -> Run:
    """Run the training or evaluation loop from parsed CLI arguments.

    Parameters
    ----------
    args : Namespace
        Parsed arguments from :func:`build_parser`.
    config : NeuralLAMConfig or None, optional
        Loaded configuration, loaded from ``args.config_path`` if ``None``.
    datastore : BaseDatastore or None, optional
        Interior datastore, used together with ``config``.
    datastore_boundary : BaseDatastore or None, optional
        Boundary datastore, used together with ``config``.

    Returns
    -------
    Run
        Paths to the run outputs and saved checkpoints.
    """
    # Several domains given with --config_paths/--graph_names, the first one
    # is the main domain
    config_paths = getattr(args, "config_paths", None)
    graph_names = getattr(args, "graph_names", None)
    if config_paths:
        if args.config_path and args.config_path != config_paths[0]:
            raise ValueError("Give either --config_path or --config_paths.")
        args.config_path = config_paths[0]
        args.additional_config_paths = list(config_paths[1:])
    if graph_names:
        args.graph = graph_names[0]
        args.additional_graphs = list(graph_names[1:])
    if not getattr(args, "config_path", None) and config is None:
        raise ValueError("A config must be given with --config_path.")

    var_leads = getattr(args, "var_leads_metrics_watch", "{}")
    if isinstance(var_leads, str):
        args.var_leads_metrics_watch = {
            int(k): v for k, v in json.loads(var_leads).items()
        }

    def _from_args(cls: Any) -> Any:
        """Instantiate a configuration dataclass from matching CLI arguments."""
        return cls(
            **{
                f.name: getattr(args, f.name)
                for f in dataclasses.fields(cls)
                if hasattr(args, f.name)
            }
        )

    model_config = _from_args(ModelConfig)
    train_config = _from_args(TrainRunConfig)
    data_config = _from_args(DataConfig)
    compute_config = _from_args(ComputeConfig)
    logging_config = _from_args(LoggingConfig)

    return fit(
        model_config=model_config,
        train_config=train_config,
        data_config=data_config,
        compute_config=compute_config,
        logging_config=logging_config,
        config=config,
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        config_path=args.config_path,
    )


@logger.catch(reraise=True)
def main(input_args: list[str] | None = None) -> None:
    """Main function for training and evaluating models."""
    parser = build_parser()
    args = parser.parse_args(input_args)
    log_level = getattr(args, "log_level", None)
    if isinstance(log_level, str):
        logger.remove()
        logger.add(sys.stderr, level=log_level)
    run(args)


if __name__ == "__main__":
    # `scancel -s USR1 <jobid>` dumps all thread stacks into the job log
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    main()
    # Jobs have been seen to hang for hours in interpreter shutdown hooks
    # after finishing, so exit hard. Errors are re-raised by main and give a
    # non-zero exit code (for Slurm afterok dependencies).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
