"""Auto-regressive unrolling of a step predictor into a full forecast."""

# Third-party
import torch
import torch.utils.checkpoint

# Local
from ..step_predictors.base import StepPredictor


def unroll_forecast(
    predictor: StepPredictor,
    init_states: torch.Tensor,
    forcing_features: torch.Tensor,
    boundary_states: torch.Tensor,
    boundary_mask: torch.Tensor,
    interior_mask: torch.Tensor,
    boundary_forcing: torch.Tensor | None = None,
    grad_checkpointing: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Unroll one forecast auto-regressively.

    At each step ``i`` calls ``predictor`` to produce the next state, then
    overwrites boundary nodes with the true value from
    ``boundary_states[:, i]``.

    This is a function rather than a base class or a mix-in so that both
    forecaster families can reuse it without either inheriting from the
    other; see ``DeterministicARForecaster.forward`` and
    ``BaseProbabilisticARForecaster.forward`` for the two call sites.

    Parameters
    ----------
    predictor : StepPredictor
        The predictor advancing the state one time step, applied once per
        predicted step. A predictor that samples its output makes each call
        to this function an independent trajectory.
    init_states : torch.Tensor
        Shape ``(B, 2, num_grid_nodes, num_state_vars)``. The two initial
        states ``[X_{t-1}, X_t]`` used to start the rollout from. Dims: ``B``
        is batch size, ``2`` initial time steps (``X_{t-1}, X_t``),
        ``num_grid_nodes`` is the number of spatial nodes, and
        ``num_state_vars`` is the number of state variables.
    forcing_features : torch.Tensor
        Shape ``(B, pred_steps, num_grid_nodes, num_forcing_vars)``. Forcing
        features for each predicted step; ``pred_steps`` defines the rollout
        length. Dims: ``B`` is batch size, ``pred_steps`` is the number of
        predicted steps, ``num_grid_nodes`` is the number of spatial nodes,
        and ``num_forcing_vars`` is the number of forcing variables (already
        concatenated past/current/future windows).
    boundary_states : torch.Tensor
        Shape ``(B, pred_steps, num_grid_nodes, num_state_vars)``. True state
        values used ONLY to overwrite boundary nodes at each AR step. The
        interior prediction at step ``i`` must not depend on
        ``boundary_states[:, i]`` in any other way. Dims: same as the
        prediction.
    boundary_mask : torch.Tensor
        Shape ``(1, num_grid_nodes, 1)``. ``1`` on boundary nodes, ``0``
        elsewhere.
    interior_mask : torch.Tensor
        Shape ``(1, num_grid_nodes, 1)``. The complement of
        ``boundary_mask``, passed in rather than derived so the buffer the
        forecaster already holds is reused.
    boundary_forcing : torch.Tensor, optional
        Shape ``(B, pred_steps, num_boundary_nodes, num_boundary_features)``.
        Boundary forcing from a separate boundary datastore, passed on to
        boundary-forced predictors.
    grad_checkpointing : bool, optional
        If True, use gradient checkpointing for each unrolled step, trading
        compute for memory. Default False.

    Returns
    -------
    prediction : torch.Tensor
        Shape ``(B, pred_steps, num_grid_nodes, num_state_vars)``. Stacked
        per-step forecasts (with boundary overwritten by the true value).
        Dims: same as ``boundary_states``.
    pred_std : torch.Tensor or None
        Shape ``(B, pred_steps, num_grid_nodes, num_state_vars)`` when
        ``predictor`` outputs an std, otherwise ``None`` (in which case
        substituting a fallback std is left to whatever consumes the
        forecast). Dims: same as ``prediction``.
    """
    prev_prev_state = init_states[:, 0]
    prev_state = init_states[:, 1]
    prediction_list = []
    pred_std_list = []
    aux_lists: dict[str, list[torch.Tensor]] = {}
    pred_steps = forcing_features.shape[1]
    boundary_forced = getattr(predictor, "boundary_forced", False)

    def step(
        prev_state: torch.Tensor,
        prev_prev_state: torch.Tensor,
        forcing: torch.Tensor,
        boundary_forcing_step: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """One predictor step, also returning its gate logits (if any).

        Parameters
        ----------
        prev_state, prev_prev_state, forcing : torch.Tensor
            Predictor inputs.
        boundary_forcing_step : torch.Tensor or None
            Boundary forcing for boundary-forced predictors.

        Returns
        -------
        tuple
            ``(pred_state, pred_std, gate_logits)``.
        """
        if boundary_forced:
            pred_state, pred_std = predictor(
                prev_state,
                prev_prev_state,
                forcing,
                boundary_forcing=boundary_forcing_step,
            )
        else:
            pred_state, pred_std = predictor(
                prev_state, prev_prev_state, forcing
            )
        step_aux = getattr(predictor, "step_aux", {}) or {}
        return pred_state, pred_std, step_aux.get("gate_logits")

    for i in range(pred_steps):
        forcing = forcing_features[:, i]
        boundary_state = boundary_states[:, i]
        boundary_forcing_step = (
            boundary_forcing[:, i]
            if boundary_forcing is not None and boundary_forcing.shape[-1] > 0
            else None
        )

        if grad_checkpointing:
            pred_state, pred_std, gate_logits = (
                torch.utils.checkpoint.checkpoint(
                    step,
                    prev_state,
                    prev_prev_state,
                    forcing,
                    boundary_forcing_step,
                    use_reentrant=False,
                )
            )
        else:
            pred_state, pred_std, gate_logits = step(
                prev_state, prev_prev_state, forcing, boundary_forcing_step
            )

        if boundary_mask.shape[-2] == pred_state.shape[-2]:
            new_state = (
                boundary_mask * boundary_state + interior_mask * pred_state
            )
        elif not bool(boundary_mask.any()):
            # Mask of another domain without boundary nodes, nothing to
            # overwrite (multi-domain training with boundary datastores)
            new_state = pred_state
        else:
            raise ValueError(
                "The boundary mask does not match the number of grid nodes."
            )

        prediction_list.append(new_state)
        if pred_std is not None:
            pred_std_list.append(pred_std)
        if gate_logits is not None:
            aux_lists.setdefault("gate_logits", []).append(gate_logits)

        # Update conditioning states
        prev_prev_state = prev_state
        prev_state = new_state

    prediction = torch.stack(prediction_list, dim=1)
    stacked_std = torch.stack(pred_std_list, dim=1) if pred_std_list else None
    # Stacked auxiliary outputs, (B, pred_steps, ...), read by forecasters
    predictor.rollout_aux = {
        name: torch.stack(values, dim=1) for name, values in aux_lists.items()
    }

    return prediction, stacked_std
