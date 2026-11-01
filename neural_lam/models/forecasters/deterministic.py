"""Forecasters trained by scoring a single deterministic forecast."""

# Standard library

# Third-party
import torch

# Local
from ... import metrics
from ...config import NeuralLAMConfig
from ...datastore import BaseDatastore
from ...loss_weighting import get_per_var_std
from ..step_predictors.base import StepPredictor
from .autoregressive import unroll_forecast
from .base import BaseForecaster


class BaseDeterministicForecaster(BaseForecaster):
    """
    Forecaster whose ``forward`` is deterministic, trained by scoring it.

    Repeating a call returns the same forecast, so one call is the whole
    forecast; contrast ``BaseProbabilisticForecaster``, which samples a fresh
    one each call. How that forecast is produced stays open, ``forward``
    being inherited abstract; see ``DeterministicARForecaster`` for the
    auto-regressive one. ``compute_training_loss`` produces one forecast
    and scores it with ``self.loss``, and ``compute_loss_from_forecast``
    applies the same scoring rule to an already-produced forecast for
    reporting.
    """

    per_var_std: torch.Tensor | None

    def __init__(
        self,
        datastore: BaseDatastore,
        config: NeuralLAMConfig | None = None,
        loss: str = "wmse",
    ) -> None:
        """
        Set up the scoring rule and the constant ``pred_std`` fallback.

        Parameters
        ----------
        datastore : BaseDatastore
            The datastore providing the state standardization statistics
            used to compute ``per_var_std``.
        config : NeuralLAMConfig or None, optional
            Configuration used to compute the constant per-variable std
            substituted for ``pred_std`` when the forecast carries no std of
            its own. Needed only when ``loss`` is a scoring rule that uses a
            std; without it in that case ``compute_loss_from_forecast`` and
            ``compute_training_loss`` raise ``ValueError`` via
            ``_resolve_pred_std``. Forecasters used purely for inference can
            always omit it. Default ``None``.
        loss : str, optional
            The scoring rule (from ``neural_lam.metrics``) applied by
            ``compute_training_loss``, stored as ``self.loss``. Default
            ``"wmse"``.
        """
        super().__init__(datastore=datastore)
        self.loss = metrics.get_metric(loss)
        self.loss_name = loss

        # Whether the fallback is needed is settled per call in
        # _resolve_pred_std
        per_var_std = (
            get_per_var_std(config=config, datastore=datastore)
            if config is not None
            else None
        )
        self.register_buffer("per_var_std", per_var_std, persistent=False)
        self._init_quantile_matching(config, datastore)

    def _init_quantile_matching(
        self, config: NeuralLAMConfig | None, datastore: BaseDatastore
    ) -> None:
        """Set up the optional quantile matching loss term.

        Parameters
        ----------
        config : NeuralLAMConfig or None
            Configuration holding ``training.quantile_matching``.
        datastore : BaseDatastore
            Datastore providing the state feature names.

        Raises
        ------
        ValueError
            If weights are given for features that are not state features.
        """
        qm_weights = (
            dict(config.training.quantile_matching.weights)
            if config is not None
            else {}
        )
        state_feature_names = datastore.get_vars_names(category="state")
        unknown = set(qm_weights) - set(state_feature_names)
        if unknown:
            raise ValueError(
                f"Quantile matching weights given for {unknown}, which are "
                f"not state features ({state_feature_names})"
            )
        self.qm_feature_names = list(qm_weights)
        qm_indices = torch.tensor(
            [state_feature_names.index(n) for n in qm_weights],
            dtype=torch.long,
        )
        self.register_buffer("qm_indices", qm_indices, persistent=False)
        self.register_buffer(
            "qm_weights",
            torch.tensor(list(qm_weights.values()), dtype=torch.float32),
            persistent=False,
        )
        # Scale the term like the wmse term of the same feature, so that the
        # weight is relative to that feature's own loss term
        if self.loss_name == "wmse" and self.per_var_std is not None:
            qm_scale = 1.0 / self.per_var_std[qm_indices] ** 2
        else:
            qm_scale = torch.ones(len(qm_weights))
        self.register_buffer("qm_scale", qm_scale, persistent=False)

    def compute_auxiliary_losses(
        self,
        prediction: torch.Tensor,
        target_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        """
        Additional loss terms on top of the scoring rule: the quantile
        matching term (``training.quantile_matching``) and the binary
        cross-entropy of gated features (``output_clamping.gated_features``)
        from the last rollout.

        Parameters
        ----------
        prediction : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, num_state_vars)``.
        target_states : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, num_state_vars)``.
        mask : torch.Tensor or None, optional
            Shape ``(num_grid_nodes,)``, boolean mask of nodes to use.

        Returns
        -------
        step_loss : torch.Tensor or None
            Shape ``(B, pred_steps)``, summed additional terms, or ``None``
            if there are none.
        log_components : dict of {str: torch.Tensor}
            Scalar values to log (without split prefix).
        """
        step_loss = None
        log_components: dict[str, torch.Tensor] = {}
        if mask is not None and mask.shape[0] != prediction.shape[-2]:
            # Mask of another domain, only all-interior masks carry over
            mask = None

        if self.qm_indices.numel() > 0:
            qm_per_feature = metrics.sorted_mse(
                prediction[..., self.qm_indices],
                target_states[..., self.qm_indices],
                mask=mask,
            )  # (B, pred_steps, n_qm)
            step_loss = torch.sum(
                qm_per_feature * self.qm_scale * self.qm_weights, dim=-1
            )
            for i, name in enumerate(self.qm_feature_names):
                log_components[f"qm_{name}"] = qm_per_feature[..., i].mean()

        predictor = getattr(self, "predictor", None)
        rollout_aux = getattr(predictor, "rollout_aux", {}) or {}
        gate_logits = rollout_aux.get("gate_logits")
        if gate_logits is not None:
            assert predictor is not None
            gated_targets = target_states[..., predictor.gated_indices]
            binary_targets = (gated_targets > predictor.gate_thresholds).to(
                gate_logits.dtype
            )
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                gate_logits, binary_targets, reduction="none"
            )  # (B, pred_steps, num_grid_nodes, num_gated)
            if mask is not None:
                bce = bce[..., mask, :]
            bce_loss = bce.sum(dim=-1).mean(dim=-1)  # (B, pred_steps)
            step_loss = bce_loss if step_loss is None else step_loss + bce_loss
            log_components["gate_bce"] = bce_loss.mean()

        return step_loss, log_components

    def compute_training_loss(
        self,
        init_states: torch.Tensor,
        forcing_features: torch.Tensor,
        target_states: torch.Tensor,
        interior_mask_bool: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Score a single forecast with ``self.loss``.

        Produces one forecast over every predicted step, scores it against
        the target states on interior nodes and averages over batch and
        time. Callers that already hold a forecast should score that one
        with ``compute_loss_from_forecast`` rather than producing another
        here.

        Parameters
        ----------
        init_states : torch.Tensor
            Shape ``(B, 2, num_grid_nodes, num_state_vars)``. The two initial
            states ``[X_{t-1}, X_t]`` used to start the forecast from. Dims:
            ``B`` is batch size, ``2`` is the time index (``[X_{t-1}, X_t]``),
            ``num_grid_nodes`` is the number of spatial nodes, and
            ``num_state_vars`` is the state feature dimension.
        forcing_features : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, num_forcing_vars)``.
            External forcings provided at each predicted step. Dims: ``B``
            is batch size, ``pred_steps`` is the number of predicted steps,
            ``num_grid_nodes`` is the number of spatial nodes, and
            ``num_forcing_vars`` is the forcing feature dimension (already
            concatenated past/current/future windows).
        target_states : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, num_state_vars)``. True
            states at each predicted step, used both as the prediction
            targets and to overwrite boundary nodes while forecasting.
            Dims: same as the prediction.
        interior_mask_bool : torch.Tensor
            Shape ``(num_grid_nodes,)``, boolean. ``True`` for interior
            nodes; passed as ``mask`` to ``self.loss`` so that only interior
            nodes are scored.

        Returns
        -------
        batch_loss : torch.Tensor
            Scalar. The scoring rule applied to the forecast, averaged over
            batch and time.
        loss_components : dict of {str: torch.Tensor}
            Empty; the deterministic objective has no separate components.

        Raises
        ------
        ValueError
            If ``self.loss`` needs a std, the forecast carries none of its
            own and no ``per_var_std`` fallback is available; see
            ``_resolve_pred_std``.

        Notes
        -----
        Customize the deterministic objective by overriding
        ``compute_loss_from_forecast``. ``DeterministicForecastingModule``
        calls that method directly for its per-step training, validation
        and test losses. Overriding only ``compute_training_loss`` does not
        change those losses.
        """
        prediction, pred_std = self(
            init_states, forcing_features, target_states
        )
        step_losses = self.compute_loss_from_forecast(
            prediction,
            target_states,
            pred_std,
            mask=interior_mask_bool,
        )
        aux_loss, aux_components = self.compute_auxiliary_losses(
            prediction, target_states, mask=interior_mask_bool
        )
        if aux_loss is not None:
            step_losses = step_losses + aux_loss
        return torch.mean(step_losses), aux_components

    def _resolve_pred_std(
        self, pred_std: torch.Tensor | None
    ) -> torch.Tensor | None:
        """
        Return the std ``self.loss`` should be applied with.

        Parameters
        ----------
        pred_std : torch.Tensor or None
            Predicted standard deviation as returned by ``forward``,
            possibly ``None``.

        Returns
        -------
        torch.Tensor or None
            ``pred_std`` unchanged when given; otherwise
            ``self.per_var_std``, or ``None`` if ``self.loss`` ignores the
            std anyway.

        Raises
        ------
        ValueError
            If ``pred_std`` is ``None``, ``self.loss`` needs one, and no
            ``per_var_std`` fallback is available (this forecaster was
            constructed without ``config``).
        """
        if pred_std is not None:
            return pred_std
        if not metrics.requires_pred_std(self.loss):
            return None
        if self.per_var_std is None:
            raise ValueError(
                "No pred_std available for scoring: this forecaster's "
                "scoring rule needs one, the forecast carries no std and "
                "there is no per_var_std fallback (it was constructed "
                "without config). Pass config to the constructor, use a "
                "predictor that outputs its own std, or score with an "
                "unweighted metric."
            )
        return self.per_var_std

    def compute_loss_from_forecast(
        self,
        prediction: torch.Tensor,
        target_states: torch.Tensor,
        pred_std: torch.Tensor | None,
        mask: torch.Tensor | None = None,
        average_grid: bool = True,
        sum_vars: bool = True,
    ) -> torch.Tensor:
        """
        Apply this forecaster's scoring rule to an already-produced forecast.

        Resolves ``pred_std`` via ``_resolve_pred_std`` (substituting
        ``self.per_var_std`` when ``None`` and ``self.loss`` needs one), then
        applies ``self.loss``. Used to report the loss on a forecast the
        caller already has, at a reduction of its choosing, without
        recomputing it.

        Only the loss goes through here. Metrics unrelated to the training
        objective depend on nothing but the shapes of the three tensors
        below, so callers compute those directly from
        ``neural_lam.metrics``.

        Parameters
        ----------
        prediction : torch.Tensor
            Shape ``(..., num_grid_nodes, num_state_vars)``. Forecast to
            score.
        target_states : torch.Tensor
            Shape ``(..., num_grid_nodes, num_state_vars)``. True states to
            score against. Dims: same as ``prediction``.
        pred_std : torch.Tensor or None
            Shape ``(..., num_grid_nodes, num_state_vars)``, or ``None``.
            Predicted standard deviation for ``prediction``; ``None`` when
            the forecast carries no std, in which case ``self.per_var_std``
            is substituted if ``self.loss`` needs a std at all (see
            ``_resolve_pred_std`` for when this raises instead).
        mask : torch.Tensor or None, optional
            Shape ``(num_grid_nodes,)``, boolean. Forwarded to ``self.loss``.
        average_grid : bool, optional
            Forwarded to ``self.loss``. Default ``True``.
        sum_vars : bool, optional
            Forwarded to ``self.loss``. Default ``True``.

        Returns
        -------
        torch.Tensor
            The scoring rule's output; shape depends on ``average_grid`` and
            ``sum_vars`` (see ``neural_lam.metrics``).

        Raises
        ------
        ValueError
            If ``pred_std`` is ``None``, ``self.loss`` needs one and no
            ``per_var_std`` fallback is available; see ``_resolve_pred_std``.
        """
        pred_std = self._resolve_pred_std(pred_std)
        return self.loss(
            prediction,
            target_states,
            pred_std,
            mask=mask,
            average_grid=average_grid,
            sum_vars=sum_vars,
        )


class DeterministicARForecaster(BaseDeterministicForecaster):
    """
    Auto-regressive forecaster trained by scoring its single rollout.

    Produces its forecast by unrolling a ``StepPredictor``; the training
    objective is the one ``BaseDeterministicForecaster`` supplies.
    """

    def __init__(
        self,
        predictor: StepPredictor,
        datastore: BaseDatastore,
        config: NeuralLAMConfig | None = None,
        loss: str = "wmse",
        grad_checkpointing: bool = False,
    ) -> None:
        """
        Initialize the DeterministicARForecaster.

        Parameters
        ----------
        predictor : StepPredictor
            The predictor to use for each AR step.
        datastore : BaseDatastore
            The datastore providing grid metadata and boundary masks.
        config : NeuralLAMConfig or None
            Configuration used to compute the constant per-variable std
            substituted for ``pred_std`` when ``predictor`` does not output
            its own. Needed only when ``loss`` is a scoring rule that uses a
            std; without it in that case ``compute_loss_from_forecast`` and
            ``compute_training_loss`` raise ``ValueError``. Forecasters used
            purely for inference (``forward``) can always omit it.
        loss : str, default "wmse"
            The scoring rule (from ``neural_lam.metrics``) applied by
            ``compute_training_loss``.
        grad_checkpointing : bool, default False
            Use gradient checkpointing in-between each unrolled step.
        """
        super().__init__(datastore=datastore, config=config, loss=loss)
        self.predictor = predictor
        self.grad_checkpointing = grad_checkpointing

    @property
    def predicts_std(self) -> bool:
        """
        Whether the forecaster predicts standard deviation.

        Returns
        -------
        bool
            ``True`` if the wrapped predictor predicts standard deviation,
            ``False`` otherwise.
        """
        return self.predictor.predicts_std

    def forward(
        self,
        init_states: torch.Tensor,
        forcing_features: torch.Tensor,
        boundary_states: torch.Tensor,
        boundary_forcing: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Unroll one auto-regressive forecast.

        Parameters
        ----------
        init_states : torch.Tensor
            Shape ``(B, 2, num_grid_nodes, num_state_vars)``. The two initial
            states ``[X_{t-1}, X_t]`` used to start the rollout from.
        forcing_features : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, num_forcing_vars)``.
            Forcing features for each predicted step; ``pred_steps`` defines
            the rollout length.
        boundary_states : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, num_state_vars)``. True
            state values used ONLY to overwrite boundary nodes at each AR
            step.
        boundary_forcing : torch.Tensor, optional
            Shape ``(B, pred_steps, num_boundary_nodes, num_boundary_feats)``.
            Forcing from a separate boundary datastore.

        Returns
        -------
        prediction : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, num_state_vars)``.
            Stacked per-step forecasts. Dims: same as ``boundary_states``.
        pred_std : torch.Tensor or None
            Shape ``(B, pred_steps, num_grid_nodes, num_state_vars)`` when
            the wrapped predictor outputs an std, otherwise ``None``. Dims:
            same as ``prediction``.
        """
        return unroll_forecast(
            self.predictor,
            init_states,
            forcing_features,
            boundary_states,
            self.boundary_mask,
            self.interior_mask,
            boundary_forcing=boundary_forcing,
            grad_checkpointing=self.grad_checkpointing,
        )
