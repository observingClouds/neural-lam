"""Multi-domain evaluation utility for neural-lam.

This script orchestrates a sequence of one-step evaluations of a single
trained model over a collection of spatial domains.  Each domain is defined by
its own configuration file.  After each step the predictions from one domain
are used to construct boundary forcing for its neighbours; the boundary arrays
are kept in-memory but the model forecasts are always written to disk.  The
behaviour mirrors the manual workflow described in the issue/feature request
and eliminates the need to create intermediate zarr boundary datasets by hand.

The adjacency between domains is provided by a small yaml/json file that maps
integer domain indices to lists of neighbour indices; if no adjacency file is
specified the domains are assumed to be laid out sequentially with neighbours
(i-1,i+1).

Example invocation::

    python -m neural_lam.eval_multi_domain \
        --config_paths dom0.yaml dom1.yaml ... dom10.yaml \
        --checkpoint saved_models/run/last.ckpt \
        --steps 20 \
        --adjacency_path neighbours.yml \
        --output_dir eval_out

The script writes predictions for every domain and step to
``<output_dir>/domain{n}/step{s}.zarr`` and retains all arrays in the
``predictions`` dictionary (indexed by step,domain) for optional post-processing.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr
import yaml

from . import utils
from .config import load_config_and_datastores
from .datastore.memory import ArrayDatastore
# `MODELS` dict lives in train_model; replicate here to avoid importing it
from .models import GraphLAM, HiLAM, HiLAMParallel
from .models.base_graph_model import BaseGraphModel
from .weather_dataset import WeatherDataModule

MODELS = {
    "graph_lam": GraphLAM,
    "hi_lam": HiLAM,
    "hi_lam_parallel": HiLAMParallel,
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_adjacency(path: Optional[str], n_domains: int) -> Dict[int, List[int]]:
    """Read adjacency mapping from YAML/JSON file or return default.

    The file should map integer keys to lists of integers, e.g.::

        0: [1]
        1: [0,2]
        2: [1]

    If ``path`` is None we assume a linear chain (i<i+1).
    """
    if path is None:
        return {i: [i - 1, i + 1] for i in range(n_domains)}
    with open(path) as f:
        data = yaml.safe_load(f)
    # allow json as well
    if isinstance(data, str):
        data = json.loads(data)
    return {int(k): [int(x) for x in v] for k, v in data.items()}


def extract_overlap(
    prediction: xr.DataArray,
    existing_boundary: Optional[xr.DataArray],
) -> xr.DataArray:
    """Return boundary forcing for ``target_idx`` after a step of ``source_idx``.

    The caller passes the full ``prediction`` array produced by the model on
    the ``source_idx`` domain together with the *current* boundary array for
    the ``target_idx`` domain (``existing_boundary``).  The overlap between
    the two is computed and merged on top of any previous boundary values.
    This mirrors the notebook pattern::

        ds_merged = xr.merge([ds_forecast_sel, ds_sel], compat="override")

    which you already used for the interior dataset.  Here, ``ds_forecast_sel``
    corresponds to the slice of ``prediction`` that lies inside the target
    boundary region; ``ds_sel`` is the previous boundary array.

    The selection itself is application-specific: in the simplest case one can
    take the grid indices of ``existing_boundary`` and select those from
    ``prediction``.  A more complex geometry (e.g. north/south/east/west
    edges) can also be handled by inspecting ``source_idx``/``target_idx``
    and using precomputed index lists.

    Parameters
    ----------
    prediction : xr.DataArray
        Model output for the entire source domain.
    existing_boundary : xr.DataArray or None
        Current boundary forcing for the target domain, or ``None`` if this is
        the first step and no boundary exists yet.

    Returns
    -------
    xr.DataArray
        New boundary forcing array to use for ``target_idx`` on the next
        evaluation step.
    """
    # early exit if there is nothing to update
    if existing_boundary is None:
        # without an existing boundary we cannot infer which grid points
        # belong to the target.  fall back to returning zeros with the same
        # shape as one time slice of prediction.
        # users will normally have an initial boundary datastore, so this case
        # should only occur if the domain has no boundary at all.
        return xr.zeros_like(prediction.isel(time=0))

    # attempt to select the overlap using the grid_index values of the
    # existing boundary; this covers the common case where the boundary array
    # already contains exactly the cells we care about
    try:
        overlap = prediction.sel(grid_index=existing_boundary.grid_index)
    except Exception:
        # if the selection fails (e.g. different coordinate name) just use the
        # full prediction; the merge below will still function correctly but
        # could be wasteful
        overlap = prediction

    # merge the old and new arrays, giving precedence to the freshly
    # computed overlap (compat="override" behaves like ds_merged above)
    merged = xr.merge([existing_boundary, overlap], compat="override")
    return merged


def build_trainer(args: argparse.Namespace) -> pl.Trainer:
    """Construct a PyTorch Lightning trainer using the same logic as
    :mod:`train_model`.
    """
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.set_float32_matmul_precision("high")
    else:
        device_name = "cpu"
    if args.devices == ["auto"]:
        devices = "auto"
    else:
        try:
            devices = [int(i) for i in args.devices]
        except ValueError:
            raise ValueError("devices should be 'auto' or a list of integers")
    
    class datastore_mock:
        def __init__(self, *args, **kwargs):
            pass
        
        @property
        def _config(self):
            return {}
    
    datastore = datastore_mock()

    training_logger = utils.setup_training_logger(
        datastore=datastore, args=args, run_name="neural-lam-eval-multi-domain"
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        deterministic=True,
        strategy="ddp",
        accelerator=device_name,
        num_nodes=args.num_nodes,
        devices=devices,
        logger=training_logger,
        enable_progress_bar=False,
        log_every_n_steps=1,
        callbacks=[],
        check_val_every_n_epoch=args.val_interval,
        precision=args.precision,
        num_sanity_val_steps=args.num_sanity_steps,
    )
    return trainer


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(input_args=None):
    parser = argparse.ArgumentParser(
        description="Multi-domain evaluation driver for neural-lam"
    )
    # reuse many arguments from train_model for convenience
    parser.add_argument(
        "--config_paths",
        nargs="+",
        type=str,
        help="Paths to the configurations for neural-lam",
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS.keys()),
        default="hi_lam",
        help="Model architecture to use for evaluation (must match the configuration used during training)",
    )
    # graph settings similar to train_model
    parser.add_argument(
        "--graph_names",
        nargs="+",
        type=str,
        default=["triangular2D_split13_domain03_hierarchical"],
        help="Graphs to load/use in graph models (default: triangular2D_split13_domain03_hierarchical)",
    )
    parser.add_argument(
        "--graph_dir",
        type=str,
        default=None,
        help="Root directory containing graph artifacts; defaults to datastore.root_path/graph",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained checkpoint to use for evaluation",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of one-step evaluation iterations to run",
    )
    parser.add_argument(
        "--adjacency_path",
        type=str,
        help="YAML/JSON file specifying domain adjacency (optional)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_multi_domain_out",
        help="Base directory where per-domain/step zarrs will be written",
    )
    # copy a small subset of options that matter for evaluation
    parser.add_argument(
        "--batch_size", type=int, default=1, help="batch size (default: 1)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers in data loader (default: 1)",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        type=str,
        default=["auto"],
        help="Devices to use for training/eval",
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="mlflow",
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
        default="neural-lam-eval-multi-domain",
        help="Logger run name, for e.g. MLFlow (with default value `None` neural-lam default format string is used)",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes to use in DDP (default: 1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Upper epoch limit; ignored but required for trainer builder",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        help="Numerical precision to use (32/16/bf16)",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Validation interval (ignored)",
    )
    parser.add_argument(
        "--num_sanity_steps",
        type=int,
        default=2,
        help="Number of sanity checking validation steps (ignored)",
    )
    parser.add_argument(
        "--ar_steps_eval",
        type=int,
        default=1,
        help="Number of unroll steps performed by each test datamodule",
    )
    parser.add_argument(
        "--init_time",
        type=str,
        help=(
            "UTC datetime of the single initialization time to evaluate, "
            "e.g. '2020-02-01T05:00'. The string will be parsed with "
            "numpy.datetime64."
        ),
    )
    parser.add_argument(
        "--time_delta_enc_dim",
        type=int,
        help="Dimensionality of positional encoding for time deltas of boundary"
        " forcing. If None, same as hidden_dim. If given, must be even "
        "(default: None)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=200,
        help="Dimensionality of hidden representations (default: 200)",
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
        "--grad_checkpointing",
        action="store_true",
        help="If gradient checkpointing should be used in-between each "
        "unrolling step (default: false)",
    )
    parser.add_argument(
        "--n_example_pred",
        type=int,
        default=1,
        help="Number of example predictions to plot during evaluation "
        "(default: 1)",
    )
    parser.add_argument(
        "--processor_layers",
        type=int,
        default=2,
        help="Number of GNN layers in processor GNN (default: 4)",
    )
    parser.add_argument(
        "--dynamic_time_deltas",
        action="store_true",
        help="If models should use dynamically computed time-deltas between"
        "interior and boundary time steps (default: False (no))",
    )
    parser.add_argument(
        "--output_std",
        action="store_true",
        help="If models should additionally output std.-dev. per "
        "output dimensions "
        "(default: False (no))",
    )
    parser.add_argument(
        "--val_steps_to_log",
        nargs="+",
        type=int,
        default=[1],
        help="Steps to log val loss for (default: 1)",
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
        "--eval_init_times",
        nargs="*",
        default=None,
        help="List of init times for evaluation forecasts",
    )
    parser.add_argument(
        "--save_eval_to_zarr_path",
        type=str,
        help="(ignored) placeholder for compatibility",
    )
    parser.add_argument(
        "--plot_vars",
        nargs="+",
        type=str,
        default=["t_2m"],
        help="List of variables to plot (ignored)",
    )
    parser.add_argument(
        "--shared_grid_embedder",
        action="store_true",  # Default to separate embedders
        help="If the same embedder MLP should be used for interior and boundary"
        " grid nodes. Note that this requires the same dimensionality for "
        "both kinds of grid inputs. (default: False (no))",
    )
    parser.add_argument(
        "--num_past_forcing_steps",
        type=int,
        default=1,
        help="Number of past forcing steps",
    )
    parser.add_argument(
        "--num_future_forcing_steps",
        type=int,
        default=1,
        help="Number of future forcing steps",
    )
    parser.add_argument(
        "--num_past_boundary_steps",
        type=int,
        default=2,
        help="Number of past boundary steps",
    )
    parser.add_argument(
        "--num_future_boundary_steps",
        type=int,
        default=0,
        help="Number of future boundary steps",
    )

    args = parser.parse_args(input_args)

    # load configurations and datastores
    domain_triples = []  # list of (config, datastore, datastore_boundary)
    for cfg_path in args.config_paths:
        cfg, ds, dsb = load_config_and_datastores(cfg_path)
        domain_triples.append((cfg, ds, dsb))

    n_domains = len(domain_triples)
    adjacency = load_adjacency(args.adjacency_path, n_domains)

    # load model
    # determine if graph-sized metadata is required
    ModelClass = MODELS[args.model]
    graph_names = args.graph_names if issubclass(ModelClass, BaseGraphModel) else None
    graph_sizes = None
    if graph_names is not None:
        graph_root = Path(args.graph_dir) if args.graph_dir else Path(domain_triples[0][1].root_path) / "graph"
        graph_dir_path = graph_root / graph_names[0]
        from .graph_data import load_graph, build_graph_sizes

        # use first datastore to load graph metadata
        graph_features_and_edges = load_graph(graph_dir_path, domain_triples[0][1])
        graph_sizes = build_graph_sizes(graph_features_and_edges)

    model_cls = ModelClass
    model = model_cls.load_from_checkpoint(
        args.checkpoint,
        args=args,
        config=domain_triples[0][0],
        datastore=domain_triples[0][1],
        datastore_boundary=domain_triples[0][2],
        graph_sizes=graph_sizes if graph_names is not None else None,
        weights_only=False,
    )

    trainer = build_trainer(args)

    # prepare initial boundary forcings
    current_boundaries: List[Optional[xr.DataArray]] = []
    reference_stores: List[Optional[Any]] = []
    for _, _, dsb in domain_triples:
        if dsb is not None:
            arr = dsb.get_dataarray(
                category="forcing",
                split="val",  # evaluation always uses val split
            )
        else:
            arr = None
        current_boundaries.append(arr)
        reference_stores.append(dsb)

    predictions: Dict[int, Dict[int, xr.DataArray]] = {}

    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    for step in range(args.steps):
        predictions[step] = {}
        for i, (cfg, ds, _) in enumerate(domain_triples):
            ds._ds.splits[2,1] = "2020-02-12T00:30"
            boundary_arr = current_boundaries[i]
            boundary_ds = (
                ArrayDatastore(boundary_arr, reference=reference_stores[i])
                if boundary_arr is not None
                else None
            )

            import ipdb; ipdb.set_trace()
            dm = WeatherDataModule(
                datastores=[ds],
                datastores_boundary=[boundary_ds],
                ar_steps_train=1,  # irrelevant
                ar_steps_eval=args.ar_steps_eval,
                standardize=True,
                num_past_forcing_steps=args.num_past_forcing_steps,
                num_future_forcing_steps=args.num_future_forcing_steps,
                num_past_boundary_steps=args.num_past_boundary_steps,
                num_future_boundary_steps=args.num_future_boundary_steps,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                eval_split="val",
                eval_init_times=args.eval_init_times or None,
                dynamic_time_deltas=False,
                excluded_intervals=cfg.training.excluded_intervals,
                graph_names=graph_names,
                graph_dir=args.graph_dir,
            )
            dm.setup(stage="test")

            # destination path for this domain/step
            out_path = out_base / f"domain{i}" / f"step{step}.zarr"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # make sure the model writes to the correct file
            model.args.save_eval_to_zarr_path = str(out_path)
            # optionally restrict the datamodule to a single init time
            if args.init_time is not None:
                # parse the provided string once
                target_dt = np.datetime64(args.init_time)

                class SingleInitWrapper(torch.utils.data.Dataset):
                    def __init__(self, ds, target_dt):
                        self.ds = ds
                        self.target = target_dt
                        self.idx = None
                        ind = np.argwhere(ds.weather_dataset.da_state.time.values == target_dt)[0]
                        if len(ind) == 0:
                            raise ValueError(f"no sample with init time {target_dt}")
                        else:
                            self.idx = ind[0].item() - 2
                        # assert ds[self.idx]["batch_times"].numpy().astype("datetime64[ns]") == target_dt, (
                        #     f"expected init time {target_dt} but found {ds[self.idx]['batch_times'].numpy().astype('datetime64[ns]')}"
                        # )

                    def __len__(self):
                        return 1

                    def __getitem__(self, ii):
                        return self.ds[self.idx]

                dm.test_dataset = SingleInitWrapper(dm.test_dataset, target_dt)

            trainer.test(
                model=model,
                datamodule=dm,
                ckpt_path=args.checkpoint,
                weights_only=False,
                verbose=False,
            )

            # read zarr back into memory for boundary extraction
            pred_arr = xr.open_zarr(out_path)
            predictions[step][i] = pred_arr

        # update boundary arrays for next iteration
        for i in range(n_domains):
            for j in adjacency.get(i, []):
                predictions_i = predictions[step][i]
                current_boundaries[j] = extract_overlap(
                    predictions_i,
                    current_boundaries[j],
                    source_idx=i,
                    target_idx=j,
                )

    # optionally the full `predictions` dict can be saved/pickled here
    # but we leave that to caller
    return predictions


if __name__ == "__main__":
    main()
