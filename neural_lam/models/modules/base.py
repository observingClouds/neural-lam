"""Abstract Lightning module shared by deterministic and probabilistic
forecasting modules."""

# Standard library
import os
import pickle
import warnings
from abc import ABC, abstractmethod
from typing import Any

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import xarray as xr
from loguru import logger
from PIL import Image

# First-party
from neural_lam.utils import get_integer_time

# Local
from ... import vis
from ...config import NeuralLAMConfig
from ...datastore.base import BaseDatastore
from ...weather_dataset import WeatherDataset
from ..forecasters.base import BaseForecaster

# (init_states, target_states, forcing, boundary, target_times)
Batch = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


class BaseForecastingModule(pl.LightningModule, ABC):
    """
    Abstract Lightning module wrapping a ``BaseForecaster``.

    Owns everything that does not depend on whether the wrapped forecaster
    produces a single deterministic forecast or samples an ensemble:
    batch standardization, the training loop, optimizer configuration,
    checkpoint compatibility, and the plotting/aggregation helpers used by
    validation and testing. Producing one forecast (``common_step``) belongs
    here too, since every forecaster shares the ``forward`` contract. What
    differs is what a subclass evaluates: one forecast, or an ensemble
    sampled from repeated calls. ``validation_step``, ``test_step`` and
    ``on_test_epoch_end`` therefore stay abstract; concrete subclasses
    implement them independently (see ``DeterministicForecastingModule`` and
    ``ProbabilisticForecastingModule``) rather than overriding one another.
    """

    # pylint: disable=arguments-differ

    val_metrics: dict[str, list[torch.Tensor]]
    test_metrics: dict[str, list[torch.Tensor]]

    interior_mask_bool: torch.Tensor
    per_var_std: torch.Tensor | None
    state_mean: torch.Tensor
    state_std: torch.Tensor
    forcing_mean: torch.Tensor | None
    forcing_std: torch.Tensor | None
    forcing_mean_tiled: torch.Tensor | None
    forcing_std_tiled: torch.Tensor | None
    boundary_mean: torch.Tensor | None
    boundary_std: torch.Tensor | None
    boundary_mean_tiled: torch.Tensor | None
    boundary_std_tiled: torch.Tensor | None

    def __init__(
        self,
        forecaster: BaseForecaster,
        config: NeuralLAMConfig,
        datastore: BaseDatastore,
        datastore_boundary: BaseDatastore | None = None,
        lr: float = 1e-3,
        restore_opt: bool = False,
        n_example_pred: int = 1,
        create_gif: bool = False,
        val_steps_to_log: list[int] | None = None,
        train_steps_to_log: list[int] | None = None,
        metrics_watch: list[str] | None = None,
        var_leads_metrics_watch: dict[int, list[int]] | None = None,
        args: Any | None = None,
    ) -> None:
        """
        Initialize the BaseForecastingModule.

        Parameters
        ----------
        forecaster : BaseForecaster
            The forecaster model to use for predictions. Owns the training
            objective (``compute_training_loss``); this module and its
            subclasses never compute a loss themselves, only inject shared
            inputs (e.g. the interior mask) and log what the forecaster
            returns.
        config : NeuralLAMConfig
            Configuration object for the neural LAM model.
        datastore : BaseDatastore
            Datastore providing grid metadata and data access.
        datastore_boundary : BaseDatastore, optional
            Boundary forcing datastore, used for its standardization
            statistics.
        lr : float, default 1e-3
            Learning rate for the optimizer.
        restore_opt : bool, default False
            Whether to restore optimizer state from checkpoint.
        n_example_pred : int, default 1
            Number of example predictions to plot during testing.
        create_gif : bool, default False
            Whether to create GIFs of example predictions.
        val_steps_to_log : list of int, optional
            Specific rollout steps to log during validation/testing.
        train_steps_to_log : list of int, optional
            Specific predicted steps to log during training. Only has an
            effect for a forecaster whose objective decomposes per step;
            this class's ``training_step`` logs the aggregate ``train_loss``
            alone (see its docstring for why).
        metrics_watch : list of str, optional
            List of metrics to watch and log specifically.
        var_leads_metrics_watch : dict of {int: list of int}, optional
            Mapping from variable index to a list of rollout steps to log
            individually for the configured metrics.
        args : argparse.Namespace, optional
            Pre-refactor ``ARModel`` checkpoint hyperparameters. When
            provided, attributes on ``args`` take precedence over the
            corresponding explicit kwargs (``lr``, ``restore_opt``,
            ``n_example_pred``, ``create_gif``, ``val_steps_to_log``,
            ``train_steps_to_log``, ``metrics_watch``,
            ``var_leads_metrics_watch``) so legacy checkpoints round-trip
            through ``load_from_checkpoint`` correctly.
        """
        super().__init__()
        # Pre-refactor ``ARModel`` checkpoints saved every hyperparameter nested
        # inside an argparse Namespace under the single key 'args'. When
        # Lightning calls __init__ during load_from_checkpoint it would
        # otherwise drop 'args' (not in the new signature) and silently fall
        # back to defaults for lr/create_gif/etc. Unpack the namespace here
        # so legacy checkpoints round-trip correctly.
        if args is not None:
            lr = getattr(args, "lr", lr)
            restore_opt = getattr(args, "restore_opt", restore_opt)
            n_example_pred = getattr(args, "n_example_pred", n_example_pred)
            create_gif = getattr(args, "create_gif", create_gif)
            val_steps_to_log = getattr(
                args, "val_steps_to_log", val_steps_to_log
            )
            train_steps_to_log = getattr(
                args, "train_steps_to_log", train_steps_to_log
            )
            metrics_watch = getattr(args, "metrics_watch", metrics_watch)
            var_leads_metrics_watch = getattr(
                args, "var_leads_metrics_watch", var_leads_metrics_watch
            )

        # Resolve mutable defaults
        if val_steps_to_log is None:
            val_steps_to_log = [
                1,
            ]
        if train_steps_to_log is None:
            train_steps_to_log = []
        if metrics_watch is None:
            metrics_watch = []
        if var_leads_metrics_watch is None:
            var_leads_metrics_watch = {}

        # Hyperparameters are named explicitly rather than left to
        # save_hyperparameters' default of inspecting the constructor chain.
        # That inspection records the arguments of the most derived
        # __init__, i.e. as a subclass received them, whereas the values
        # this module runs on are the ones resolved just above (the args
        # namespace, then the mutable defaults). Passing them keeps what is
        # recorded equal to what is used, whatever a subclass's signature
        # looks like. Subclasses add their own with a further
        # save_hyperparameters call, which merges into these.
        #
        # datastore and forecaster are deliberately absent and must be
        # provided explicitly when calling load_from_checkpoint. Saving args
        # makes the checkpoint self-describing: it carries model,
        # graph_name, hidden_dim, etc. so the caller can reconstruct the
        # exact forecaster architecture from the checkpoint alone.
        self.save_hyperparameters(
            {
                "config": config,
                "lr": lr,
                "restore_opt": restore_opt,
                "n_example_pred": n_example_pred,
                "create_gif": create_gif,
                "val_steps_to_log": val_steps_to_log,
                "train_steps_to_log": train_steps_to_log,
                "metrics_watch": metrics_watch,
                "var_leads_metrics_watch": var_leads_metrics_watch,
                "args": args,
            }
        )

        self.datastore = datastore
        self.datastore_boundary = datastore_boundary
        self.forecaster = forecaster
        self.matched_metrics: set = set()

        # Compute interior_mask_bool directly from datastore
        boundary_mask = (
            torch.tensor(datastore.boundary_mask.values, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(-1)
        )  # (1, num_grid_nodes, 1)
        interior_mask = 1.0 - boundary_mask
        self.register_buffer(
            "interior_mask_bool",
            interior_mask[0, :, 0].to(torch.bool),
            persistent=False,
        )

        # Standardization statistics used to normalize each batch on-device in
        # `on_after_batch_transfer`. WeatherDataset returns unstandardized
        # data, so state and forcing are normalized here rather than on CPU.
        eps = torch.finfo(torch.float32).eps
        da_state_stats = datastore.get_standardization_dataarray(
            category="state"
        )
        self.register_buffer(
            "state_mean",
            torch.tensor(da_state_stats.state_mean.values, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "state_std",
            self._safe_std(da_state_stats.state_std.values, eps, "state"),
            persistent=False,
        )

        if datastore.get_num_data_vars(category="forcing") > 0:
            da_forcing_stats = datastore.get_standardization_dataarray(
                category="forcing"
            )
            self.register_buffer(
                "forcing_mean",
                torch.tensor(
                    da_forcing_stats.forcing_mean.values, dtype=torch.float32
                ),
                persistent=False,
            )
            self.register_buffer(
                "forcing_std",
                self._safe_std(
                    da_forcing_stats.forcing_std.values, eps, "forcing"
                ),
                persistent=False,
            )
            # The windowed forcing width is only known once the first batch
            # arrives, so the tiled mean/std are built and cached then (in
            # on_after_batch_transfer) instead of recomputed every batch.
            self.register_buffer("forcing_mean_tiled", None, persistent=False)
            self.register_buffer("forcing_std_tiled", None, persistent=False)
        else:
            self.forcing_mean = None
            self.forcing_std = None

        # Boundary tiling matches forcing, see `on_after_batch_transfer`.
        if (
            datastore_boundary is not None
            and datastore_boundary.get_num_data_vars(category="forcing") > 0
        ):
            da_boundary_stats = (
                datastore_boundary.get_standardization_dataarray(
                    category="forcing"
                )
            )
            self.register_buffer(
                "boundary_mean",
                torch.tensor(
                    da_boundary_stats.forcing_mean.values, dtype=torch.float32
                ),
                persistent=False,
            )
            self.register_buffer(
                "boundary_std",
                self._safe_std(
                    da_boundary_stats.forcing_std.values, eps, "boundary"
                ),
                persistent=False,
            )
            self.register_buffer("boundary_mean_tiled", None, persistent=False)
            self.register_buffer("boundary_std_tiled", None, persistent=False)
        else:
            self.boundary_mean = None
            self.boundary_std = None

        # For making restoring of optimizer state optional
        self.restore_opt = restore_opt

        # For example plotting
        self.n_example_pred = n_example_pred
        self.create_gif = create_gif
        self.plotted_examples = 0

        # Warn once per phase if steps_to_log exceeds the actual rollout
        self._steps_warn_issued = {
            "val": False,
            "test": False,
            "train": False,
        }

        self.time_step_int, self.time_step_unit = get_integer_time(
            self.datastore.step_length
        )

    def _create_dataarray_from_tensor(
        self,
        tensor: torch.Tensor,
        time: torch.Tensor,
        split: str,
        category: str,
    ) -> xr.DataArray:
        """
        Create an xarray DataArray from a torch tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor to convert.
        time : torch.Tensor
            The time coordinates for the data.
        split : str
            The data split (e.g., 'train', 'val', 'test').
        category : str
            The category of data (e.g., 'state', 'forcing').

        Returns
        -------
        xr.DataArray
            The resulting xarray DataArray.
        """
        weather_dataset = WeatherDataset(datastore=self.datastore, split=split)
        time_np = np.array(time.cpu(), dtype="datetime64[ns]")
        da = weather_dataset.create_dataarray_from_tensor(
            tensor=tensor, time=time_np, category=category
        )
        return da

    def _run_arg(self, name: str, default: Any) -> Any:
        """Read a run option from the ``args`` hyperparameters.

        Parameters
        ----------
        name : str
            Name of the option.
        default : Any
            Value used when ``args`` is missing or lacks the option.

        Returns
        -------
        Any
            The option value.
        """
        args = self.hparams.get("args")  # ty: ignore[unresolved-attribute]
        return getattr(args, name, default) if args is not None else default

    def configure_optimizers(self) -> Any:
        """
        Configure the optimizers and learning rate schedulers.

        With ``args.lr_scheduler == "linear_cosine"`` a linear warm-up
        (``lr_warmup_iters`` scheduler steps) is chained with cosine annealing
        down to ``min_lr`` over ``epochs`` scheduler steps, stepping every
        ``lr_scheduler_frequency`` optimizer steps. Otherwise the learning
        rate is constant.

        Returns
        -------
        torch.optim.Optimizer or dict
            The optimizer, or a Lightning optimizer/scheduler configuration.
        """
        lr = self.hparams.lr  # ty: ignore[unresolved-attribute]
        opt = torch.optim.AdamW(self.parameters(), lr=lr, betas=(0.9, 0.95))
        if self._run_arg("lr_scheduler", "constant") != "linear_cosine":
            return opt

        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, total_iters=self._run_arg("lr_warmup_iters", 10)
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self._run_arg("epochs", 200),
            eta_min=self._run_arg("min_lr", 0.0),
        )
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [warmup, cosine], optimizer=opt
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": self._run_arg("lr_scheduler_frequency", 700),
            },
        }

    def on_train_start(self) -> None:
        """
        When resuming, the restored cosine schedule keeps the ``T_max`` of
        the original run; extend it to the requested number of epochs.
        """
        epochs = self._run_arg("epochs", None)
        if epochs is None:
            return
        for config in self.trainer.lr_scheduler_configs:
            schedulers = getattr(
                config.scheduler, "_schedulers", [config.scheduler]
            )
            for sched in schedulers:
                if (
                    isinstance(
                        sched, torch.optim.lr_scheduler.CosineAnnealingLR
                    )
                    and sched.T_max != epochs
                ):
                    logger.info(
                        f"Setting cosine T_max {sched.T_max} -> {epochs}"
                    )
                    sched.T_max = epochs

    def _current_lr(self) -> float | None:
        """Learning rate of the first optimizer, if attached to a trainer.

        Returns
        -------
        float or None
            The current learning rate.
        """
        try:
            return self.trainer.optimizers[0].param_groups[0]["lr"]
        except (RuntimeError, IndexError, AttributeError):
            return None

    @staticmethod
    def _safe_std(
        std_values: np.ndarray | torch.Tensor, eps: float, category: str
    ) -> torch.Tensor:
        """Build a float32 std tensor, clamping near-zero values to `eps`.

        Mirrors the previous CPU-side WeatherDataset behavior: features with
        near-zero std are clamped to machine epsilon to avoid division-by-zero
        NaNs, and a single warning is emitted naming the affected category.
        """
        std = torch.tensor(std_values, dtype=torch.float32)
        if bool((std <= eps).any()):
            warnings.warn(
                f"Some {category} features have near-zero std and will be "
                "standardized using machine epsilon to avoid NaN.",
                UserWarning,
                stacklevel=2,
            )
        return torch.clamp(std, min=eps)

    def on_after_batch_transfer(
        self,
        batch: Batch,
        dataloader_idx: int,
    ) -> Batch:
        """Standardize a batch on-device after transfer to the accelerator.

        Lightning calls this for every train/val/test/predict batch.
        WeatherDataset returns unstandardized state, forcing and boundary;
        all three are normalized here so the work runs on the accelerator.
        Forcing and boundary share the same feature-major
        ``(feature, window)`` stacking, so per-feature mean/std are tiled
        once and cached.

        Parameters
        ----------
        batch : Batch
            ``(init_states, target_states, forcing, boundary, target_times)``
            as returned by the dataloader.
        dataloader_idx : int
            Index of the dataloader the batch comes from.

        Returns
        -------
        Batch
            The batch with state, forcing and boundary standardized.
        """
        init_states, target_states, forcing, boundary, batch_times = batch[:5]
        # Multi-domain training batches also carry the domain index
        extra = tuple(batch[5:])

        init_states = (init_states - self.state_mean) / self.state_std
        target_states = (target_states - self.state_mean) / self.state_std

        if forcing.shape[-1] > 0:
            assert self.forcing_mean is not None
            assert self.forcing_std is not None
            if self.forcing_mean_tiled is None:
                # Forcing is (..., num_forcing_vars * window_size).
                # WeatherDataset stacks (forcing_feature, window)
                # feature-major, so each per-feature mean/std is repeated
                # window_size times. Tile once and cache as buffers rather
                # than recomputing every batch.
                window_size = forcing.shape[-1] // self.forcing_mean.shape[-1]
                self.forcing_mean_tiled = self.forcing_mean.repeat_interleave(
                    window_size
                )
                self.forcing_std_tiled = self.forcing_std.repeat_interleave(
                    window_size
                )
            assert self.forcing_mean_tiled is not None
            assert self.forcing_std_tiled is not None
            forcing = (
                forcing - self.forcing_mean_tiled
            ) / self.forcing_std_tiled

        if boundary.shape[-1] > 0:
            if self.boundary_mean is None:
                raise ValueError(
                    "Batch has non-empty boundary forcing but no boundary "
                    "standardization statistics are registered. Pass "
                    "`datastore_boundary` when constructing or loading "
                    "BaseForecastingModule."
                )
            assert self.boundary_std is not None
            # Trailing time delta features (if any) are not standardized
            num_time_delta_dims = self._num_boundary_time_delta_dims()
            num_forcing_dims = boundary.shape[-1] - num_time_delta_dims
            boundary_time_deltas = boundary[..., num_forcing_dims:]
            boundary = boundary[..., :num_forcing_dims]
            if self.boundary_mean_tiled is None:
                window_size = boundary.shape[-1] // self.boundary_mean.shape[-1]
                self.boundary_mean_tiled = self.boundary_mean.repeat_interleave(
                    window_size
                )
                self.boundary_std_tiled = self.boundary_std.repeat_interleave(
                    window_size
                )
            assert self.boundary_mean_tiled is not None
            assert self.boundary_std_tiled is not None
            boundary = (
                boundary - self.boundary_mean_tiled
            ) / self.boundary_std_tiled
            boundary = torch.cat((boundary, boundary_time_deltas), dim=-1)

        return (
            init_states,
            target_states,
            forcing,
            boundary,
            batch_times,
            *extra,
        )  # ty: ignore[invalid-return-type]

    def _num_boundary_time_delta_dims(self) -> int:
        """Number of trailing boundary features holding time deltas.

        Returns
        -------
        int
            ``boundary_time_delta_dims`` of a boundary-forced predictor,
            else 0.
        """
        predictor = getattr(self.forecaster, "predictor", None)
        if getattr(predictor, "boundary_forced", False):
            return int(predictor.boundary_time_delta_dims)
        return 0

    def _select_domain(self, batch: Batch) -> None:
        """Switch the predictor to the domain of a multi-domain batch.

        Parameters
        ----------
        batch : tuple
            Batch, with the domain indices as optional sixth entry.
        """
        predictor = getattr(self.forecaster, "predictor", None)
        if not hasattr(predictor, "set_domain"):
            return
        if len(batch) > 5:
            domain_idx = batch[5]
            assert bool((domain_idx == domain_idx[0]).all()), (
                "Batches must hold samples of a single domain"
            )
            predictor.set_domain(int(domain_idx[0]))
        else:
            predictor.set_domain(0)

    def common_step(
        self,
        batch: Batch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """
        Produce one forecast for the batch.

        Every forecaster shares the ``forward`` contract, so this works for
        a probabilistic forecaster too; there it returns a single sample
        rather than a whole ensemble (see
        ``BaseProbabilisticForecaster.sample_ensemble``).

        Parameters
        ----------
        batch : tuple
            The batch of data containing initial states, target states,
            forcing features, boundary forcing features, and batch times.

        Returns
        -------
        tuple
            A tuple containing prediction, target states, predicted standard
            deviation, and batch times.
        """
        (
            init_states,
            target_states,
            forcing_features,
            boundary_features,
            batch_times,
        ) = batch[:5]
        self._select_domain(batch)
        predictor = getattr(self.forecaster, "predictor", None)
        if getattr(predictor, "boundary_forced", False):
            prediction, pred_std = self.forecaster(
                init_states,
                forcing_features,
                target_states,
                boundary_forcing=boundary_features,
            )
        else:
            # Boundary forcing is only used by boundary-forced predictors
            prediction, pred_std = self.forecaster(
                init_states, forcing_features, target_states
            )
        return prediction, target_states, pred_std, batch_times

    def training_step(
        self,
        batch: Batch,
    ) -> torch.Tensor:
        """
        Perform a single training step.

        The training objective is fully assembled by the wrapped forecaster,
        which owns its own scoring rule; this method injects the interior
        mask, then logs the loss and any loss components the forecaster
        returns.

        This general implementation logs only the scalar objective, since
        ``compute_training_loss`` returns only a scalar: a forecaster's
        objective (e.g. an ELBO) need not decompose over predicted steps, so
        there is no per-step tensor to select from with
        ``train_steps_to_log``. Subclasses whose forecaster does expose such
        a decomposition override this to log the breakdown as well; see
        ``DeterministicForecastingModule.training_step``.

        Parameters
        ----------
        batch : tuple
            The batch of data.

        Returns
        -------
        torch.Tensor
            The computed loss for the training step.
        """
        init_states, target_states, forcing_features, _boundary, _ = batch
        batch_loss, loss_components = self.forecaster.compute_training_loss(
            init_states,
            forcing_features,
            target_states,
            interior_mask_bool=self.interior_mask_bool,
        )

        log_dict = {
            f"train_{name}": value for name, value in loss_components.items()
        }
        log_dict["train_loss"] = batch_loss
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )
        return batch_loss

    def all_gather_cat(self, tensor_to_gather: torch.Tensor) -> torch.Tensor:
        """
        Gather tensors from all GPUs and concatenate them.

        Parameters
        ----------
        tensor_to_gather : torch.Tensor
            The tensor to gather from all processes.

        Returns
        -------
        torch.Tensor
            The concatenated tensor from all processes.
        """
        gathered = self.all_gather(tensor_to_gather)
        assert isinstance(gathered, torch.Tensor)
        # all_gather adds dim 0 only on multi-device; on single
        # device it returns the same tensor unchanged.
        if gathered.dim() > tensor_to_gather.dim():
            gathered = gathered.flatten(0, 1)
        return gathered

    # pylint: disable-next=unused-argument
    def _warn_skipped_steps(self, pred_steps: int, phase: str) -> None:
        """Warn once per phase if any log steps exceed pred_steps."""
        if self._steps_warn_issued[phase]:
            return
        hparam_name = (
            "train_steps_to_log" if phase == "train" else "val_steps_to_log"
        )
        steps_to_log = getattr(self.hparams, hparam_name, [])

        invalid = [s for s in steps_to_log if s > pred_steps]
        if invalid:
            warnings.warn(
                f"{hparam_name} contains steps {invalid} that exceed "
                f"the {phase} rollout length ({pred_steps}). "
                "These steps will be skipped from logging. "
                f"Adjust --{hparam_name} or the ar_steps.",
                UserWarning,
                stacklevel=2,
            )
        self._steps_warn_issued[phase] = True

    def _log_step_loss(
        self,
        time_step_loss: torch.Tensor,
        mean_loss: torch.Tensor,
        phase: str,
        batch_size: int,
    ) -> None:
        """Log mean and step-wise losses for a given phase."""
        self._warn_skipped_steps(len(time_step_loss), phase)

        loss_key = "train_loss" if phase == "train" else f"{phase}_mean_loss"
        log_dict = {loss_key: mean_loss}

        hparam_name = (
            "train_steps_to_log" if phase == "train" else "val_steps_to_log"
        )
        steps_to_log = getattr(self.hparams, hparam_name, [])

        for step in steps_to_log:
            if step <= len(time_step_loss):
                log_dict[f"{phase}_loss_unroll{step}"] = time_step_loss[
                    step - 1
                ]

        is_train = phase == "train"
        self.log_dict(
            log_dict,
            prog_bar=is_train,
            on_step=is_train,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )

    @abstractmethod
    def validation_step(
        self,
        batch: Batch,
        batch_idx: int,
    ) -> None:
        """
        Perform a single validation step.

        Concrete subclasses must both score the batch and populate the
        ``val_metrics`` they created, which ``on_validation_epoch_end``
        below aggregates.

        Kept abstract even though ``LightningModule`` defines this method:
        that definition is a no-op stub, not an abstract method, so without
        this declaration a subclass that omitted it would instantiate
        happily and silently skip validation. Declaring it abstract turns
        that into a ``TypeError`` at construction.

        Parameters
        ----------
        batch : tuple
            The batch of data.
        batch_idx : int
            The index of the batch.
        """

    def on_validation_epoch_end(self) -> None:
        """
        Perform actions at the end of the validation epoch.
        Aggregates and plots validation metrics.
        """
        self.aggregate_and_plot_metrics(self.val_metrics, prefix="val")

        if (
            self.trainer.is_global_zero and self.hparams.metrics_watch  # ty: ignore[unresolved-attribute]
        ):
            metrics_watch = (
                self.hparams.metrics_watch  # ty: ignore[unresolved-attribute]
            )
            unmatched = set(metrics_watch) - self.matched_metrics
            if unmatched:
                warnings.warn(
                    "The following metrics in --metrics_watch "
                    "were not found during validation phase: "
                    f"{sorted(unmatched)}. Ensure the metric prefix "
                    "matches the evaluation mode (expected 'val_')."
                )

        self.matched_metrics = set()

        for metric_list in self.val_metrics.values():
            metric_list.clear()

    @abstractmethod
    def test_step(
        self,
        batch: Batch,
        batch_idx: int,
    ) -> None:
        """
        Perform a single test step.

        Concrete subclasses must both score the batch and populate the
        ``test_metrics`` they created, which their ``on_test_epoch_end``
        aggregates.

        Kept abstract for the same reason as ``validation_step``: the
        ``LightningModule`` definition is a no-op stub rather than an
        abstract method, so omitting it would silently skip testing instead
        of failing at construction.

        Parameters
        ----------
        batch : tuple
            The batch of data.
        batch_idx : int
            The index of the batch.
        """

    def plot_examples(
        self,
        batch: Batch,
        n_examples: int,
        split: str,
        prediction: torch.Tensor,
    ) -> None:
        """
        Plot example predictions.

        Parameters
        ----------
        batch : tuple
            The batch of data.
        n_examples : int
            Number of examples to plot.
        split : str
            The data split.
        prediction : torch.Tensor
            The model predictions.
        """

        target = batch[1]
        time = batch[4]

        da_state_stats = self.datastore.get_standardization_dataarray("state")
        state_std = torch.tensor(
            da_state_stats.state_std.values,
            dtype=torch.float32,
            device=prediction.device,
        )
        state_mean = torch.tensor(
            da_state_stats.state_mean.values,
            dtype=torch.float32,
            device=prediction.device,
        )

        prediction_rescaled = prediction * state_std + state_mean
        target_rescaled = target * state_std + state_mean

        for pred_slice, target_slice, time_slice in zip(
            prediction_rescaled[:n_examples],
            target_rescaled[:n_examples],
            time[:n_examples],
        ):
            self.plotted_examples += 1

            da_prediction = self._create_dataarray_from_tensor(
                tensor=pred_slice,
                time=time_slice,
                split=split,
                category="state",
            )
            da_target = self._create_dataarray_from_tensor(
                tensor=target_slice,
                time=time_slice,
                split=split,
                category="state",
            )
            if self.datastore.is_on_rectilinear_grid:
                da_prediction = da_prediction.unstack("grid_index")
                da_target = da_target.unstack("grid_index")

            var_vmin = (
                torch.minimum(
                    pred_slice.flatten(0, 1).min(dim=0)[0],
                    target_slice.flatten(0, 1).min(dim=0)[0],
                )
                .cpu()
                .numpy()
            )
            var_vmax = (
                torch.maximum(
                    pred_slice.flatten(0, 1).max(dim=0)[0],
                    target_slice.flatten(0, 1).max(dim=0)[0],
                )
                .cpu()
                .numpy()
            )
            var_vranges = list(zip(var_vmin, var_vmax))

            # Optionally only plot a subset of the variables
            plot_vars = self._run_arg("plot_vars", None)
            plot_var_names = [
                var_name
                for var_name in self.datastore.get_vars_names("state")
                if not plot_vars or var_name in plot_vars
            ]

            example_i = self.plotted_examples

            if self.create_gif:
                plot_dir_path = os.path.join(
                    self.logger.save_dir,  # ty: ignore[unresolved-attribute]
                    f"example_plots_{example_i}",
                )
                os.makedirs(plot_dir_path, exist_ok=True)
                png_frames: dict[str, list[str]] = {
                    var_name: []
                    for var_name in self.datastore.get_vars_names("state")
                }

            for t_i, _ in enumerate(zip(pred_slice, target_slice), start=1):
                var_figs = [
                    vis.plot_prediction(
                        datastore=self.datastore,
                        title=f"{var_name} ({var_unit}), "
                        f"t={t_i} ({(self.time_step_int * t_i)}"
                        f"{self.time_step_unit})",
                        vrange=var_vrange,
                        da_prediction=da_prediction.isel(
                            state_feature=var_i, time=t_i - 1
                        ).squeeze(),
                        da_target=da_target.isel(
                            state_feature=var_i, time=t_i - 1
                        ).squeeze(),
                    )
                    for var_i, (var_name, var_unit, var_vrange) in enumerate(
                        zip(
                            self.datastore.get_vars_names("state"),
                            self.datastore.get_vars_units("state"),
                            var_vranges,
                        )
                    )
                    if var_name in plot_var_names
                ]

                for var_name, fig in zip(plot_var_names, var_figs):
                    if isinstance(self.logger, pl.loggers.WandbLogger):
                        key = f"{var_name}_example"
                    else:
                        key = f"{var_name}_example_{example_i}"

                    if hasattr(self.logger, "log_image"):
                        self.logger.log_image(  # ty: ignore[call-non-callable]
                            key=key, images=[fig], step=t_i
                        )
                    else:
                        warnings.warn(
                            f"{self.logger} does not support image logging."
                        )

                    if self.create_gif:
                        png_path = os.path.join(
                            plot_dir_path,
                            f"{var_name}_example_{example_i}"
                            f"_prediction_t_{t_i:02d}.png",
                        )
                        fig.savefig(png_path, dpi=100, bbox_inches="tight")
                        png_frames[var_name].append(png_path)

                plt.close("all")

            if self.create_gif:
                for var_name, frames_for_var in png_frames.items():
                    if frames_for_var:
                        gif_path = os.path.join(
                            plot_dir_path,
                            f"{var_name}_example_{example_i}_prediction.gif",
                        )
                        frames = [Image.open(f) for f in frames_for_var]
                        try:
                            frames[0].save(
                                gif_path,
                                save_all=True,
                                append_images=frames[1:],
                                loop=0,
                                duration=1000,
                            )
                        finally:
                            for frame in frames:
                                frame.close()

            os.makedirs(
                self.logger.save_dir,  # ty: ignore[unresolved-attribute]
                exist_ok=True,
            )
            torch.save(
                pred_slice.cpu(),
                os.path.join(
                    self.logger.save_dir,  # ty: ignore[unresolved-attribute]
                    f"example_pred_{self.plotted_examples}.pt",
                ),
            )
            torch.save(
                target_slice.cpu(),
                os.path.join(
                    self.logger.save_dir,  # ty: ignore[unresolved-attribute]
                    f"example_target_{self.plotted_examples}.pt",
                ),
            )

    def create_metric_log_dict(
        self, metric_tensor: torch.Tensor, prefix: str, metric_name: str
    ) -> dict[str, Any]:
        """
        Create a dictionary of metrics to log.

        Parameters
        ----------
        metric_tensor : torch.Tensor
            The metric values.
        prefix : str
            Prefix for the log names (e.g., 'val', 'test').
        metric_name : str
            The name of the metric.

        Returns
        -------
        dict
            Dictionary of logged metrics and figures.
        """
        log_dict: dict[str, Any] = {}
        metric_fig = vis.plot_error_heatmap(
            errors=metric_tensor,
            datastore=self.datastore,
        )
        full_log_name = f"{prefix}_{metric_name}"
        log_dict[full_log_name] = metric_fig

        if prefix == "test":
            metric_fig.savefig(
                os.path.join(
                    self.logger.save_dir,  # ty: ignore[unresolved-attribute]
                    f"{full_log_name}.pdf",
                )
            )
            np.savetxt(
                os.path.join(
                    self.logger.save_dir,  # ty: ignore[unresolved-attribute]
                    f"{full_log_name}.csv",
                ),
                metric_tensor.cpu().numpy(),
                delimiter=",",
            )

        var_names = self.datastore.get_vars_names(category="state")
        hparams = self.hparams
        metrics_watch = (
            hparams.metrics_watch  # ty: ignore[unresolved-attribute]
        )
        var_leads_metrics_watch = (
            hparams.var_leads_metrics_watch  # ty: ignore[unresolved-attribute]
        )
        if full_log_name in metrics_watch:
            self.matched_metrics.add(full_log_name)
            for var_i, timesteps in var_leads_metrics_watch.items():
                var_name = var_names[var_i]
                for step in timesteps:
                    key = f"{full_log_name}_{var_name}_step_{step}"
                    log_dict[key] = metric_tensor[step - 1, var_i]

        return log_dict

    def aggregate_and_plot_metrics(
        self, metrics_dict: dict[str, list[torch.Tensor]], prefix: str
    ) -> None:
        """
        Aggregate metrics from all GPUs and plot them.

        Parameters
        ----------
        metrics_dict : dict
            Dictionary of metrics lists.
        prefix : str
            Prefix for the log names.
        """
        log_dict = {}
        xr_data_vars: dict[str, Any] = {}
        for metric_name, metric_val_list in metrics_dict.items():
            metric_tensor = self.all_gather_cat(
                torch.cat(metric_val_list, dim=0)
            )

            if self.trainer.is_global_zero:
                metric_tensor_averaged = torch.mean(metric_tensor, dim=0)

                if "mse" in metric_name:
                    metric_tensor_averaged = torch.sqrt(metric_tensor_averaged)
                    metric_name = metric_name.replace("mse", "rmse")

                da_state_stats = self.datastore.get_standardization_dataarray(
                    "state"
                )
                state_std = torch.tensor(
                    da_state_stats.state_std.values,
                    dtype=torch.float32,
                    device=metric_tensor_averaged.device,
                )
                metric_rescaled = metric_tensor_averaged * state_std

                log_dict.update(
                    self.create_metric_log_dict(
                        metric_rescaled, prefix, metric_name
                    )
                )
                xr_data_vars[metric_name] = (
                    ["variable", "lead_time"],
                    metric_rescaled.cpu().numpy().T,
                )

        figure_dict = {
            k: v for k, v in log_dict.items() if isinstance(v, plt.Figure)
        }
        scalar_dict = {
            k: v for k, v in log_dict.items() if not isinstance(v, plt.Figure)
        }

        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            # Log scalars via Lightning's built-in mechanism. No sync_dist:
            # we are inside `if is_global_zero`, so only rank 0 reaches this
            # call. sync_dist=True would launch a distributed collective
            # from one rank and hang DDP. The values are already aggregated
            # via all_gather_cat above, so no further sync is needed.
            if scalar_dict:
                self.log_dict(scalar_dict, rank_zero_only=True)

            current_epoch = self.trainer.current_epoch

            for key, figure in figure_dict.items():
                if not isinstance(self.logger, pl.loggers.WandbLogger):
                    key = f"{key}-{current_epoch}"

                if hasattr(self.logger, "log_image"):
                    self.logger.log_image(  # ty: ignore[call-non-callable]
                        key=key, images=[figure]
                    )

            plt.close("all")

            if xr_data_vars:
                self._save_metrics_dataset(xr_data_vars, prefix)

    def _save_predictions_to_zarr(
        self,
        batch_times: torch.Tensor,
        batch_predictions: torch.Tensor,
        batch_idx: int,
        zarr_output_path: str,
    ) -> None:
        """
        Save state predictions for a single batch to a zarr dataset,
        appending for ``batch_idx > 0``. The dataset holds a variable
        ``state`` with coordinates ``(start_time, elapsed_forecast_duration,
        grid_index, state_feature)``.

        Parameters
        ----------
        batch_times : torch.Tensor
            Shape ``(B, pred_steps)``. Target times as epoch nanoseconds.
        batch_predictions : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, num_state_vars)``.
            Standardized predictions.
        batch_idx : int
            Index of the batch in the epoch.
        zarr_output_path : str
            Path of the zarr dataset to write.
        """
        # Third-party
        import numcodecs

        batch_predictions_rescaled = (
            batch_predictions * self.state_std + self.state_mean
        )
        split = self._run_arg("eval", None) or "test"
        das_pred = []
        for i in range(len(batch_times)):
            da_pred = self._create_dataarray_from_tensor(
                tensor=batch_predictions_rescaled[i].float(),
                time=batch_times[i],
                split=split,
                category="state",
            )
            # The forecast is initialized one step before the first target
            t0 = da_pred.coords["time"].values[0] - np.timedelta64(
                self.datastore.step_length
            )
            da_pred.coords["start_time"] = t0
            da_pred.coords["elapsed_forecast_duration"] = da_pred.time - t0
            da_pred = da_pred.swap_dims({"time": "elapsed_forecast_duration"})
            da_pred.name = "state"
            # MultiIndex coordinates can not be stored in zarr, keep the
            # spatial coordinates as plain coordinates along grid_index
            grid_index = da_pred.indexes.get("grid_index")
            spatial_coords = {}
            if isinstance(grid_index, pd.MultiIndex):
                spatial_coords = {
                    name: ("grid_index", grid_index.get_level_values(name))
                    for name in grid_index.names
                }
            da_pred = xr.DataArray(
                da_pred.transpose(
                    "elapsed_forecast_duration", "grid_index", "state_feature"
                ).values,
                dims=(
                    "elapsed_forecast_duration",
                    "grid_index",
                    "state_feature",
                ),
                coords={
                    "elapsed_forecast_duration": da_pred[
                        "elapsed_forecast_duration"
                    ].values,
                    "start_time": t0,
                    "grid_index": np.arange(da_pred.sizes["grid_index"]),
                    "state_feature": self.datastore.get_vars_names("state"),
                    **spatial_coords,
                },
                name="state",
            )
            das_pred.append(da_pred)

        da_pred_batch = xr.concat(das_pred, dim="start_time").chunk(
            {"start_time": 1, "elapsed_forecast_duration": 1}
        )

        if batch_idx == 0:
            logger.info(f"Saving predictions to {zarr_output_path}")
            compressor = numcodecs.Blosc(
                cname="zstd", clevel=9, shuffle=numcodecs.Blosc.SHUFFLE
            )
            da_pred_batch.to_zarr(
                zarr_output_path,
                mode="w",
                consolidated=True,
                zarr_format=2,
                encoding={
                    "start_time": {
                        "units": "minutes since 2020-01-01T00:00:00Z",
                        "dtype": "int64",
                    },
                    "state": {"compressor": compressor},
                },
            )
        else:
            da_pred_batch.to_zarr(
                zarr_output_path, mode="a", append_dim="start_time"
            )

    def _save_metrics_dataset(
        self, xr_data_vars: dict[str, Any], prefix: str
    ) -> None:
        """Save aggregated metrics as a pickled ``xr.Dataset`` with
        dimensions ``(variable, lead_time)`` to ``<prefix>_metrics.pkl`` in
        the logger directory.

        Parameters
        ----------
        xr_data_vars : dict
            Data variables of the dataset, metric name to
            ``(dims, values)``.
        prefix : str
            Split prefix (``val``/``test``).
        """
        num_steps = next(iter(xr_data_vars.values()))[1].shape[1]
        lead_time = (
            np.arange(1, num_steps + 1)
            * np.timedelta64(self.datastore.step_length)
        ).astype("timedelta64[ns]")
        metric_ds = xr.Dataset(
            data_vars=xr_data_vars,
            coords={
                "variable": self.datastore.get_vars_names(category="state"),
                "variable_units": (
                    "variable",
                    self.datastore.get_vars_units(category="state"),
                ),
                "variable_long_name": (
                    "variable",
                    self.datastore.get_vars_long_names(category="state"),
                ),
                "lead_time": lead_time,
            },
        )
        save_dir = self.logger.save_dir  # ty: ignore[unresolved-attribute]
        if save_dir is None:
            return
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{prefix}_metrics.pkl"), "wb") as f:
            pickle.dump(metric_ds, f)

    @abstractmethod
    def on_test_epoch_end(self) -> None:
        """
        Perform actions at the end of the test epoch.

        Concrete subclasses must at least aggregate and plot
        ``self.test_metrics`` (typically via ``aggregate_and_plot_metrics``)
        and reset any epoch-scoped state they accumulate during
        ``test_step``.
        """

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """
        Perform actions when loading a checkpoint.
        Handles backward compatibility for older checkpoints.

        Parameters
        ----------
        checkpoint : dict
            The loaded checkpoint dictionary.
        """
        loaded_state_dict = checkpoint["state_dict"]

        # 1. Broad namespace remap: for pre-refactor checkpoints
        # The old ``ARModel`` was a flat LightningModule. Everything that
        # belonged to the predictor needs to be moved to
        # 'forecaster.predictor.', while 'per_var_std' (now owned by the
        # forecaster itself) moves to 'forecaster.per_var_std' and
        # 'interior_mask_bool' (still owned by the module) stays as-is.
        old_keys = list(loaded_state_dict.keys())
        for key in old_keys:
            if key.startswith("forecaster.") or key == "interior_mask_bool":
                continue
            if key == "per_var_std":
                loaded_state_dict["forecaster.per_var_std"] = (
                    loaded_state_dict.pop(key)
                )
                continue
            new_key = f"forecaster.predictor.{key}"
            loaded_state_dict[new_key] = loaded_state_dict.pop(key)

        # 2. Specific rename from g2m_gnn.grid_mlp -> encoding_grid_mlp
        # Will be under forecaster.predictor due to the remap above, or
        # already there if from a recent checkpoint before this rename.
        if (
            "forecaster.predictor.g2m_gnn.grid_mlp.0.weight"
            in loaded_state_dict
        ):
            replace_keys = list(
                filter(
                    lambda key: key.startswith(
                        "forecaster.predictor.g2m_gnn.grid_mlp"
                    ),
                    loaded_state_dict.keys(),
                )
            )
            for old_key in replace_keys:
                new_key = old_key.replace(
                    "forecaster.predictor.g2m_gnn.grid_mlp",
                    "forecaster.predictor.encoding_grid_mlp",
                )
                loaded_state_dict[new_key] = loaded_state_dict[old_key]
                del loaded_state_dict[old_key]

        # Checkpoints of the boundary-forcing research models named the grid
        # embedder `interior_embedder`
        for old_key in [
            k
            for k in loaded_state_dict
            if k.startswith("forecaster.predictor.interior_embedder.")
        ]:
            new_key = old_key.replace(
                "forecaster.predictor.interior_embedder.",
                "forecaster.predictor.grid_embedder.",
            )
            loaded_state_dict[new_key] = loaded_state_dict.pop(old_key)

        if not self.restore_opt:
            opt_config = self.configure_optimizers()
            if isinstance(opt_config, dict):
                opt = opt_config["optimizer"]
                checkpoint["lr_schedulers"] = [
                    opt_config["lr_scheduler"]["scheduler"].state_dict()
                ]
            else:
                opt = opt_config
                checkpoint["lr_schedulers"] = []
            checkpoint["optimizer_states"] = [opt.state_dict()]
