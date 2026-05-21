# Standard library
from typing import Any, Dict, Union

# Third-party
import numpy as np
import torch
from torch import nn

# Local
from .. import utils
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore
from ..graph_data import GraphSizes
from ..interaction_net import InteractionNet
from .ar_model import ARModel


class BaseGraphModel(ARModel):
    """
    Base (abstract) class for graph-based models building on
    the encode-process-decode idea.
    """

    def __init__(
        self,
        args,
        config: NeuralLAMConfig,
        datastore: BaseDatastore,
        datastore_boundary: Union[BaseDatastore, None],
        graph_sizes: GraphSizes,
    ):
        super().__init__(
            args,
            config=config,
            datastore=datastore,
            datastore_boundary=datastore_boundary,
        )

        self.graph_sizes = graph_sizes
        self.hierarchical = graph_sizes.hierarchical
        self.current_graph: Union[dict[str, Any], None] = None

        # Specify dimensions of data
        utils.rank_zero_print(
            f"Loaded graph with {self.num_total_grid_nodes + self.num_mesh_nodes} "
            f"nodes ({self.num_total_grid_nodes} grid, {self.num_mesh_nodes} mesh)"
        )

        # Determine grid hidden dim
        if args.hidden_dim_grid is None:
            # Same as hidden_dim
            hidden_dim_grid = args.hidden_dim
        else:
            hidden_dim_grid = args.hidden_dim_grid

        # interior_dim from data + static
        g2m_dim = graph_sizes.g2m_dim
        m2g_dim = graph_sizes.m2g_dim

        # Define sub-models
        # Feature embedders for interior
        self.mlp_blueprint_end = [args.hidden_dim] * (args.hidden_layers + 1)
        # For grid hidden dim
        self.grid_mlp_blueprint_end = [hidden_dim_grid] * (
            args.hidden_layers + 1
        )
        self.interior_embedder = utils.make_mlp(
            [self.interior_dim] + self.grid_mlp_blueprint_end
        )

        if self.boundary_forced:
            # Define embedder for boundary nodes
            # Optional separate embedder for boundary nodes
            if args.shared_grid_embedder:
                assert self.interior_dim == self.boundary_dim, (
                    "Grid and boundary input dimension must "
                    "be the same when using "
                    f"the same embedder, got interior_dim={self.interior_dim}, "
                    f"boundary_dim={self.boundary_dim}"
                )
                self.boundary_embedder = self.interior_embedder
            else:
                self.boundary_embedder = utils.make_mlp(
                    [self.boundary_dim] + self.grid_mlp_blueprint_end
                )

        # Projections between grid dim and hidden dim before and after processor
        self.pre_mesh_proj = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim_grid, args.hidden_dim)
        )
        self.post_mesh_proj = nn.Sequential(
            nn.SiLU(), nn.Linear(args.hidden_dim, hidden_dim_grid)
        )

        self.g2m_embedder = utils.make_mlp(
            [g2m_dim] + self.grid_mlp_blueprint_end
        )
        self.m2g_embedder = utils.make_mlp(
            [m2g_dim] + self.grid_mlp_blueprint_end
        )

        # GNNs
        # encoder
        self.g2m_gnn = InteractionNet(
            hidden_dim_grid,
            hidden_layers=args.hidden_layers,
            update_edges=False,
            num_rec=self.num_grid_connected_mesh_nodes,
        )
        self.encoding_grid_mlp = utils.make_mlp(
            [hidden_dim_grid] + self.grid_mlp_blueprint_end
        )

        # decoder
        self.m2g_gnn = InteractionNet(
            hidden_dim_grid,
            hidden_layers=args.hidden_layers,
            update_edges=False,
            num_rec=self.num_interior_nodes,
        )

        # Output mapping (hidden_dim -> output_dim)
        self.output_map = utils.make_mlp(
            [hidden_dim_grid]
            + [hidden_dim_grid] * args.hidden_layers
            + [self.grid_output_dim],
            layer_norm=False,
        )  # No layer norm on this one

        # Compute constants for use in time_delta encoding
        if self.boundary_forced:
            step_length_ratio = (
                datastore_boundary.step_length / datastore.step_length
            )
            min_time_delta = (
                -(args.num_past_boundary_steps + 1) * step_length_ratio
            )
            max_time_delta = args.num_future_boundary_steps * step_length_ratio
            time_delta_magnitude = max(max_time_delta, abs(min_time_delta))
            freq_indices = 1.0 + torch.arange(
                self.time_delta_enc_dim // 2,
                dtype=torch.float,
            )
            self.register_buffer(
                "enc_freq_denom",
                (2 * time_delta_magnitude)
                ** (2 * freq_indices / self.time_delta_enc_dim),
                persistent=False,
            )

        # Compute indices and define clamping functions
        self.prepare_clamping_params(config, datastore)

        # Identify gated features and define gate projection
        gated_feature_names = config.training.output_clamping.gated_features
        state_feature_names = datastore.get_vars_names(category="state")
        gated_indices = [
            state_feature_names.index(name)
            for name in gated_feature_names
            if name in state_feature_names
        ]
        self.register_buffer(
            "gated_indices",
            torch.tensor(gated_indices, dtype=torch.long),
            persistent=False,
        )

        if len(gated_indices) > 0:
            self.gate_map = utils.make_mlp(
                [hidden_dim_grid]
                + [hidden_dim_grid] * args.hidden_layers
                + [len(gated_indices)],
                layer_norm=False,
            )
        else:
            self.gate_map = None

    @property
    def num_mesh_nodes(self):
        """
        Get the total number of mesh nodes in the used mesh graph
        """
        raise NotImplementedError("num_mesh_nodes not implemented")

    def set_graph(self, graph: Dict[str, Any]):
        """
        Store graph tensors for the current batch on the correct device.
        """
        device = self.interior_static_features.device

        def move_to_device(value):
            if isinstance(value, torch.Tensor):
                return value.to(device=device)
            if isinstance(value, list):
                return [move_to_device(v) for v in value]
            return value

        self.current_graph = {
            key: move_to_device(val)
            for key, val in graph.items()
            if key not in ["hierarchical", "boundary_static_features"]
        }
        self.current_graph["hierarchical"] = graph["hierarchical"]
        if self.current_graph["hierarchical"] != self.hierarchical:
            raise ValueError(
                "Graph hierarchy level changed between batches, "
                "which is not supported."
            )

        # Set boundary_static_features for multi-domain support
        if "boundary_static_features" in graph:
            self.current_boundary_static_features = graph["boundary_static_features"].to(device)
        else:
            self.current_boundary_static_features = None

    def prepare_clamping_params(
        self, config: NeuralLAMConfig, datastore: BaseDatastore
    ):
        """
        Prepare parameters for clamping predicted values to valid range
        """

        # Read configs
        state_feature_names = datastore.get_vars_names(category="state")
        lower_lims = config.training.output_clamping.lower
        upper_lims = config.training.output_clamping.upper

        # Check that limits in config are for valid features
        unknown_features_lower = set(lower_lims.keys()) - set(
            state_feature_names
        )
        unknown_features_upper = set(upper_lims.keys()) - set(
            state_feature_names
        )
        if unknown_features_lower or unknown_features_upper:
            raise ValueError(
                "State feature limits were provided for unknown features: "
                f"{unknown_features_lower.union(unknown_features_upper)}"
            )

        # Constant parameters for clamping
        sigmoid_sharpness = 1
        softplus_sharpness = 1
        sigmoid_center = 0
        softplus_center = 0

        normalize_clamping_lim = (
            lambda x, feature_idx: (x - self.state_mean[feature_idx])
            / self.state_std[feature_idx]
        )

        # Check which clamping functions to use for each feature
        sigmoid_lower_upper_idx = []
        sigmoid_lower_lims = []
        sigmoid_upper_lims = []

        softplus_lower_idx = []
        softplus_lower_lims = []

        softplus_upper_idx = []
        softplus_upper_lims = []

        for feature_idx, feature in enumerate(state_feature_names):
            if feature in lower_lims and feature in upper_lims:
                assert (
                    lower_lims[feature] < upper_lims[feature]
                ), f'Invalid clamping limits for feature "{feature}",\
                     lower: {lower_lims[feature]}, larger than\
                     upper: {upper_lims[feature]}'
                sigmoid_lower_upper_idx.append(feature_idx)
                sigmoid_lower_lims.append(
                    normalize_clamping_lim(lower_lims[feature], feature_idx)
                )
                sigmoid_upper_lims.append(
                    normalize_clamping_lim(upper_lims[feature], feature_idx)
                )
            elif feature in lower_lims and feature not in upper_lims:
                softplus_lower_idx.append(feature_idx)
                softplus_lower_lims.append(
                    normalize_clamping_lim(lower_lims[feature], feature_idx)
                )
            elif feature not in lower_lims and feature in upper_lims:
                softplus_upper_idx.append(feature_idx)
                softplus_upper_lims.append(
                    normalize_clamping_lim(upper_lims[feature], feature_idx)
                )

        self.register_buffer(
            "sigmoid_lower_lims", torch.tensor(sigmoid_lower_lims)
        )
        self.register_buffer(
            "sigmoid_upper_lims", torch.tensor(sigmoid_upper_lims)
        )
        self.register_buffer(
            "softplus_lower_lims", torch.tensor(softplus_lower_lims)
        )
        self.register_buffer(
            "softplus_upper_lims", torch.tensor(softplus_upper_lims)
        )

        self.register_buffer(
            "clamp_lower_upper_idx", torch.tensor(sigmoid_lower_upper_idx)
        )
        self.register_buffer(
            "clamp_lower_idx", torch.tensor(softplus_lower_idx)
        )
        self.register_buffer(
            "clamp_upper_idx", torch.tensor(softplus_upper_idx)
        )

        # Define clamping functions
        self.clamp_lower_upper = lambda x: (
            self.sigmoid_lower_lims
            + (self.sigmoid_upper_lims - self.sigmoid_lower_lims)
            * torch.sigmoid(sigmoid_sharpness * (x - sigmoid_center))
        )
        self.clamp_lower = lambda x: (
            self.softplus_lower_lims
            + torch.nn.functional.softplus(
                x - softplus_center, beta=softplus_sharpness
            )
        )
        self.clamp_upper = lambda x: (
            self.softplus_upper_lims
            - torch.nn.functional.softplus(
                softplus_center - x, beta=softplus_sharpness
            )
        )

        self.inverse_clamp_lower_upper = lambda x: (
            sigmoid_center
            + utils.inverse_sigmoid(
                (x - self.sigmoid_lower_lims)
                / (self.sigmoid_upper_lims - self.sigmoid_lower_lims)
            )
            / sigmoid_sharpness
        )
        self.inverse_clamp_lower = lambda x: (
            utils.inverse_softplus(
                x - self.softplus_lower_lims, beta=softplus_sharpness
            )
            + softplus_center
        )
        self.inverse_clamp_upper = lambda x: (
            -utils.inverse_softplus(
                self.softplus_upper_lims - x, beta=softplus_sharpness
            )
            + softplus_center
        )

    def get_clamped_new_state(self, state_delta, prev_state):
        """
        Clamp prediction to valid range supplied in config
        Returns the clamped new state after adding delta to original state

        Instead of the new state being computed as
        $X_{t+1} = X_t + \\delta = X_t + model(\\{X_t,X_{t-1},...\\}, forcing)$
        The clamped values will be
        $f(f^{-1}(X_t) + model(\\{X_t, X_{t-1},... \\}, forcing))$
        Which means the model will learn to output values in the range of the
        inverse clamping function

        state_delta: (B, num_grid_nodes, feature_dim)
        prev_state: (B, num_grid_nodes, feature_dim)
        """

        # Assign new state, but overwrite clamped values of each type later
        new_state = prev_state + state_delta

        # Sigmoid/logistic clamps between ]a,b[
        if self.clamp_lower_upper_idx.numel() > 0:
            idx = self.clamp_lower_upper_idx

            new_state[:, :, idx] = self.clamp_lower_upper(
                self.inverse_clamp_lower_upper(prev_state[:, :, idx])
                + state_delta[:, :, idx]
            )

        # Softplus clamps between ]a,infty[
        if self.clamp_lower_idx.numel() > 0:
            idx = self.clamp_lower_idx

            new_state[:, :, idx] = self.clamp_lower(
                self.inverse_clamp_lower(prev_state[:, :, idx])
                + state_delta[:, :, idx]
            )

        # Softplus clamps between ]-infty,b[
        if self.clamp_upper_idx.numel() > 0:
            idx = self.clamp_upper_idx

            new_state[:, :, idx] = self.clamp_upper(
                self.inverse_clamp_upper(prev_state[:, :, idx])
                + state_delta[:, :, idx]
            )

        return new_state

    @property
    def num_grid_connected_mesh_nodes(self):
        """
        Get the total number of mesh nodes that have a connection to
        the grid (e.g. bottom level in a hierarchy)
        """
        raise NotImplementedError(
            "num_grid_connected_mesh_nodes not implemented"
        )

    def embedd_mesh_nodes(self):
        """
        Embed static mesh features
        Returns tensor of shape (num_mesh_nodes, d_h)
        """
        raise NotImplementedError("embedd_mesh_nodes not implemented")

    def process_step(self, mesh_rep):
        """
        Process step of embedd-process-decode framework
        Processes the representation on the mesh, possible in multiple steps

        mesh_rep: has shape (B, num_mesh_nodes, d_h)
        Returns mesh_rep: (B, num_mesh_nodes, d_h)
        """
        raise NotImplementedError("process_step not implemented")

    def predict_step(
        self, prev_state, prev_prev_state, forcing, boundary_forcing
    ):
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        prev_state: (B, num_interior_nodes, feature_dim), X_t
        prev_prev_state: (B, num_interior_nodes, feature_dim), X_{t-1}
        forcing: (B, num_interior_nodes, forcing_dim)
        boundary_forcing: (B, num_boundary_nodes, boundary_forcing_dim)
        """
        if self.current_graph is None:
            raise RuntimeError(
                "Graph data has not been set for the current batch. "
                "Ensure the dataloader provides graph information via "
                "WeatherDatasetWithGraph."
            )
        batch_size = prev_state.shape[0]
        graph = self.current_graph

        # Create full interior node features of shape
        # (B, num_interior_nodes, interior_dim)
        interior_features = torch.cat(
            (
                prev_state,
                prev_prev_state,
                forcing,
                self.expand_to_batch(self.interior_static_features, batch_size),
            ),
            dim=-1,
        )

        if self.boundary_forced:
            # sin-encode time deltas for boundary forcing
            boundary_forcing = self.encode_forcing_time_deltas(boundary_forcing)

            # Use per-domain static features if available, else model's
            current_boundary_static = (
                self.current_boundary_static_features
                if self.current_boundary_static_features is not None
                else self.boundary_static_features
            )

            # Create full boundary node features of shape
            # (B, num_boundary_nodes, boundary_dim)
            boundary_features = torch.cat(
                (
                    boundary_forcing,
                    self.expand_to_batch(
                        current_boundary_static,
                        batch_size
                    ),
                ),
                dim=-1,
            )

            # Embed boundary features
            boundary_emb = self.boundary_embedder(boundary_features)
            # (B, num_boundary_nodes, d_h)

        # Embed all features
        interior_emb = self.interior_embedder(
            interior_features
        )  # (B, num_interior_nodes, d_h)
        g2m_emb = self.g2m_embedder(graph["g2m_features"])  # (M_g2m, d_h)
        m2g_emb = self.m2g_embedder(graph["m2g_features"])  # (M_m2g, d_h)
        mesh_emb = self.embedd_mesh_nodes()

        if self.boundary_forced:
            # Merge interior and boundary emb into input embedding
            # We enforce ordering (interior, boundary) of nodes
            full_grid_emb = torch.cat((interior_emb, boundary_emb), dim=1)
        else:
            # Only maps from interior to mesh
            full_grid_emb = interior_emb

        # Map from grid to mesh
        mesh_emb_expanded = self.expand_to_batch(
            mesh_emb, batch_size
        )  # (B, num_mesh_nodes, d_h)
        g2m_emb_expanded = self.expand_to_batch(g2m_emb, batch_size)

        # Encode to mesh
        mesh_rep = self.g2m_gnn(
            full_grid_emb,
            mesh_emb_expanded,
            g2m_emb_expanded,
            graph["g2m_edge_index"],
        )  # (B, num_mesh_nodes, d_h)
        # Also MLP with residual for grid representation
        grid_rep = interior_emb + self.encoding_grid_mlp(
            interior_emb
        )  # (B, num_interior_nodes, d_h)

        # Project up mesh rep to hidden dim of graph
        mesh_rep = self.pre_mesh_proj(mesh_rep)

        # Run processor step
        mesh_rep = self.process_step(mesh_rep)

        # Project down mesh rep to hidden dim of grid
        mesh_rep = self.post_mesh_proj(mesh_rep)

        # Map back from mesh to grid
        m2g_emb_expanded = self.expand_to_batch(m2g_emb, batch_size)
        grid_rep = self.m2g_gnn(
            mesh_rep,
            grid_rep,
            m2g_emb_expanded,
            graph["m2g_edge_index"],
        )  # (B, num_interior_nodes, d_h)

        # Map to output dimension, only for grid
        net_output = self.output_map(
            grid_rep
        )  # (B, num_interior_nodes, d_grid_out)

        if self.output_std:
            pred_delta_mean, pred_std_raw = net_output.chunk(
                2, dim=-1
            )  # both (B, num_interior_nodes, d_f)
            # NOTE: The predicted std. is not scaled in any way here
            # linter for some reason does not think softplus is callable
            # pylint: disable-next=not-callable
            pred_std = torch.nn.functional.softplus(pred_std_raw)
        elif self.num_quantiles > 0:
            # Net output is (B, num_interior_nodes, num_quantiles * d_f)
            # Treat the net_output as (B, num_interior_nodes, d_f, n_q)
            pred_quantiles = net_output.view(
                batch_size, -1, self.num_quantiles
            )  # (B*N, d_f, n_q)
            pred_quantiles = pred_quantiles.view(
                batch_size, -1, pred_quantiles.shape[1], self.num_quantiles
            )  # (B, N, d_f, n_q)

            # Use median as point prediction
            mid_idx = self.num_quantiles // 2
            if self.num_quantiles % 2 == 1:
                pred_delta_mean = pred_quantiles[:, :, :, mid_idx]
            else:
                pred_delta_mean = (
                    pred_quantiles[:, :, :, mid_idx - 1]
                    + pred_quantiles[:, :, :, mid_idx]
                ) / 2
            pred_std = pred_quantiles  # (B, N, d_f, n_q)
        else:
            pred_delta_mean = net_output
            pred_std = None

        # Rescale with one-step difference statistics
        rescaled_delta_mean = pred_delta_mean * self.diff_std + self.diff_mean

        if self.gate_map is not None:
            # Predict gate logits
            gate_logits = self.gate_map(grid_rep)  # (B, N, num_gated)
            gate_probs = torch.sigmoid(gate_logits)

            # Apply gate to rescaled deltas
            # Note: rescaled_delta_mean is (B, N, d_f)
            # gate_probs is (B, N, num_gated)
            rescaled_delta_mean[:, :, self.gated_indices] = (
                rescaled_delta_mean[:, :, self.gated_indices] * gate_probs
            )

            # We also need to return the gate probabilities (or logits) for loss
            # But predict_step only returns (new_state, pred_std)
            # We can pack gate_probs into pred_std if it's otherwise None,
            # but pred_std is used for uncertainty.
            # A better way might be to attach it to the model or return it.
            # Given ARModel.unroll_prediction, we should probably return it.
            # Let's see how unroll_prediction handles pred_std.
            gate_info = gate_logits
        else:
            gate_info = None

        # Clamp values to valid range (also add the delta to the previous state)
        new_state = self.get_clamped_new_state(rescaled_delta_mean, prev_state)

        if gate_info is not None:
            if pred_std is None:
                pred_std = gate_info
            elif isinstance(pred_std, torch.Tensor):
                # Check if it has the same shape as gate_info (except last dim)
                # This is a bit hacky, but we need to return both.
                # If pred_std is (B, N, d_f), and gate_info is (B, N, num_gated)
                # we could concatenate them or return a tuple/dict.
                # But unroll_prediction expects a tensor that it can stack.
                # Actually, unroll_prediction in ARModel stacks them:
                # pred_std_list.append(step_pred_std)
                # return ..., torch.stack(pred_std_list, dim=1)
                # So it MUST be a tensor.

                if pred_std.dim() == 4:  # (B, N, d_f, n_q)
                    # For quantiles, gate_info (B, N, num_gated) needs to be
                    # expanded to (B, N, num_gated, n_q) or similar
                    # BCE loss in ARModel expects (..., num_gated).
                    # We can't easily concatenate across different dimensions.
                    # For now, we append it as a new "quantile-like" dimension
                    # but only if n_q=1 or we just expand it.
                    # To keep it simple and compatible with ARModel extraction:
                    # we expand gate_info to (B, N, num_gated, 1) and pad with
                    # zeros to match n_q, then concatenate on d_f dim? No.
                    
                    # Better: concatenate on d_f dimension, with n_q = 1 for gate
                    # but we need n_q to match.
                    gate_info_unsqueezed = gate_info.unsqueeze(-1) # (B, N, num_gated, 1)
                    gate_info_expanded = gate_info_unsqueezed.expand(
                        -1, -1, -1, self.num_quantiles
                    )
                    pred_std = torch.cat([pred_std, gate_info_expanded], dim=-2)
                else:
                    # If we have both, we concatenate them along the last dimension.
                    # We'll need to handle this in training_step.
                    pred_std = torch.cat([pred_std, gate_info], dim=-1)

        return new_state, pred_std

    def encode_forcing_time_deltas(self, boundary_forcing):
        """
        Build sinusoidal encodings of time deltas in boundary forcing. Removes
        original time delta features and replaces these with encoded sinusoidal
        features, returning the full new forcing tensor.

        Parameters
        ----------
        boundary_forcing : torch.Tensor
            Tensor of shape (B, num_nodes, num_forcing_dims) containing boundary
            forcing features. Time delta features are the last
            self.boundary_time_delta_dims dimensions of the num_forcing_dims
            feature dimensions.


        Returns
        -------
        encoded_forcing : torch.Tensor
            Tensor of shape (B, num_nodes, num_forcing_dims'), where the
            time delta features have been removed and encoded versions added.
            Note that this might change the number of feature dimensions.
        """
        # Extract time delta dimensions
        time_deltas = boundary_forcing[..., -self.boundary_time_delta_dims :]
        # (B, num_boundary_nodes, num_time_deltas)

        # Compute sinusoidal encodings
        frequencies = time_deltas.unsqueeze(-1) / self.enc_freq_denom
        # (B, num_boundary_nodes, num_time_deltas, num_freq)
        encodings_stacked = torch.cat(
            (
                torch.sin(frequencies),
                torch.cos(frequencies),
            ),
            dim=-1,
        )
        # (B, num_boundary_nodes, num_time_deltas, 2*num_freq)

        encoded_time_deltas = encodings_stacked.flatten(-2, -1)
        # (B, num_boundary_nodes, num_encoding_dims)

        # Put together encoded time deltas with rest of boundary_forcing
        return torch.cat(
            (
                boundary_forcing[..., : -self.boundary_time_delta_dims],
                encoded_time_deltas,
            ),
            dim=-1,
        )
