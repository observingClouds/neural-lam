"""Base class for graph-based step predictors."""

# Standard library
from pathlib import Path

# Third-party
import torch
from torch import nn

# Local
from .... import utils
from ....datastore import BaseDatastore
from ....gnn_layers import InteractionNet, get_gnn_class
from ..base import StepPredictor

# Graph tensors that are swapped when changing domain
GRAPH_TENSOR_NAMES = (
    "g2m_edge_index",
    "m2g_edge_index",
    "m2m_edge_index",
    "mesh_up_edge_index",
    "mesh_down_edge_index",
    "g2m_features",
    "m2g_features",
    "m2m_features",
    "mesh_up_features",
    "mesh_down_features",
    "mesh_static_features",
)
EDGE_INDEX_NAMES = (
    "g2m_edge_index",
    "m2g_edge_index",
    "m2m_edge_index",
    "mesh_up_edge_index",
    "mesh_down_edge_index",
)


def _load_static_features(
    datastore: BaseDatastore, category: str
) -> torch.Tensor:
    """Standardized static features of a datastore, as ``(N, d)`` tensor.

    Parameters
    ----------
    datastore : BaseDatastore
        Datastore to read static features from.
    category : str
        Category whose grid size to use when there are no static features
        (``"state"`` for interior, ``"forcing"`` for boundary datastores).

    Returns
    -------
    torch.Tensor
        Shape ``(num_grid_points, num_static_vars)``, possibly with zero
        features.
    """
    da_static = datastore.get_dataarray(
        category="static", split=None, standardize=True
    )
    if da_static is None:
        return torch.empty((datastore.num_grid_points, 0), dtype=torch.float32)
    return torch.tensor(da_static.values, dtype=torch.float32)


class DomainGraph(nn.Module):
    """Container holding the graph and static grid features of one domain
    as non-persistent buffers, so that they follow the model's device."""


class BaseGraphModel(StepPredictor):
    """
    Base (abstract) class for graph-based models building on
    the encode-process-decode idea.

    Optionally the model is forced by boundary data from a separate
    boundary datastore. The boundary grid nodes are then embedded
    separately and are senders (but not receivers) in the grid-to-mesh
    graph, which indexes grid nodes as ``[interior..., boundary...]``.
    """

    diff_mean: torch.Tensor
    diff_std: torch.Tensor
    g2m_features: torch.Tensor
    m2g_features: torch.Tensor
    g2m_edge_index: torch.Tensor
    m2g_edge_index: torch.Tensor
    mesh_static_features: torch.Tensor | list[torch.Tensor]
    m2m_features: torch.Tensor | list[torch.Tensor]
    m2m_edge_index: torch.Tensor | list[torch.Tensor]
    boundary_static_features: torch.Tensor
    enc_freq_denom: torch.Tensor
    gated_indices: torch.Tensor
    gate_thresholds: torch.Tensor

    def __init__(
        self,
        datastore: BaseDatastore,
        graph_name: str = "multiscale",
        hidden_dim: int = 64,
        hidden_layers: int = 1,
        processor_layers: int = 4,
        mesh_aggr: str = "sum",
        num_past_forcing_steps: int = 1,
        num_future_forcing_steps: int = 1,
        output_std: bool = False,
        output_clamping_lower: dict[str, float] | None = None,
        output_clamping_upper: dict[str, float] | None = None,
        g2m_gnn_type: str = "InteractionNet",
        m2g_gnn_type: str = "InteractionNet",
        datastore_boundary: BaseDatastore | None = None,
        num_past_boundary_steps: int = 1,
        num_future_boundary_steps: int = 1,
        hidden_dim_grid: int | None = None,
        time_delta_enc_dim: int | None = None,
        shared_grid_embedder: bool = False,
        num_quantiles: int = 0,
        gated_features: list[str] | None = None,
        gate_thresholds: dict[str, float] | None = None,
        graph_dir: str | Path | None = None,
        additional_domains: (
            list[tuple[BaseDatastore, BaseDatastore | None, str]] | None
        ) = None,
    ) -> None:
        """
        Initialize the BaseGraphModel.

        Parameters
        ----------
        datastore : BaseDatastore
            Datastore supplying data and information about dataset and
            forecast region.
        graph_name : str, default "multiscale"
            The name of the graph to load.
        hidden_dim : int, default 64
            The dimension of the hidden representations.
        hidden_layers : int, default 1
            The number of hidden layers in the MLPs.
        processor_layers : int, default 4
            The number of processor layers in the GNN.
        mesh_aggr : str, default "sum"
            The aggregation method for mesh nodes.
        num_past_forcing_steps : int, default 1
            The number of past forcing steps to include.
        num_future_forcing_steps : int, default 1
            The number of future forcing steps to include.
        output_std : bool, default False
            Whether to output a predicted standard deviation.
        output_clamping_lower : dict, optional
            Lower clamping limits for state variables.
        output_clamping_upper : dict, optional
            Upper clamping limits for state variables.
        g2m_gnn_type, m2g_gnn_type : str, default "InteractionNet"
            GNN layer types used for encoding and decoding.
        datastore_boundary : BaseDatastore, optional
            Datastore providing boundary forcing. If given, the model takes
            boundary forcing (with time delta features appended last) as
            input, and the graph must connect the boundary grid nodes.
        num_past_boundary_steps, num_future_boundary_steps : int, default 1
            Boundary forcing window used by the dataset.
        hidden_dim_grid : int, optional
            Dimensionality of the hidden representations on the grid (grid
            embeddings, g2m/m2g GNNs, output map). If given, linear
            projections map between grid and mesh hidden dimensions.
            Defaults to ``hidden_dim``.
        time_delta_enc_dim : int, optional
            Dimensionality of the sinusoidal encoding of each boundary time
            delta, must be even. Defaults to ``hidden_dim``.
        shared_grid_embedder : bool, default False
            Use the same embedder for interior and boundary grid nodes,
            which requires equal input dimensionalities.
        num_quantiles : int, default 0
            If > 0, predict this many equally spaced quantiles of each
            state variable (returned in place of the predicted std).
        gated_features : list of str, optional
            State features whose predicted increment is gated by a learned
            probability of exceeding a threshold.
        gate_thresholds : dict of str to float, optional
            Physical thresholds of gated features (default physical zero).
        graph_dir : str or Path, optional
            Directory holding the graph directories, instead of
            ``<datastore.root_path>/graph``.
        additional_domains : list of (datastore, datastore_boundary, graph)
            Further domains the model is trained on, see
            :meth:`set_domain`. They must have the same variables as
            ``datastore``.
        """
        super().__init__(
            datastore=datastore,
            output_std=output_std,
            output_clamping_lower=output_clamping_lower,
            output_clamping_upper=output_clamping_upper,
        )
        self.g2m_gnn_type = g2m_gnn_type
        self.m2g_gnn_type = m2g_gnn_type

        # Retrieve difference statistics for rescaling in forward pass
        da_state_stats = datastore.get_standardization_dataarray("state")
        self.register_buffer(
            "diff_mean",
            torch.tensor(
                da_state_stats.state_diff_mean_standardized.values,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "diff_std",
            torch.tensor(
                da_state_stats.state_diff_std_standardized.values,
                dtype=torch.float32,
            ),
            persistent=False,
        )

        # Store architecture hyperparameters for subclass use
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.processor_layers = processor_layers
        self.mesh_aggr = mesh_aggr
        self.hidden_dim_grid = (
            hidden_dim if hidden_dim_grid is None else hidden_dim_grid
        )
        self.graph_dir = graph_dir

        # Quantile outputs
        self.num_quantiles = int(num_quantiles)
        if self.num_quantiles > 0:
            if self.output_std:
                raise ValueError(
                    "output_std and num_quantiles can not be combined"
                )
            self.grid_output_dim = (
                self.num_quantiles
                * datastore.get_num_data_vars(category="state")
            )

        # Load graph with static features
        # NOTE: (IMPORTANT!) mesh nodes MUST have the first
        # num_mesh_nodes indices,
        self.hierarchical = self._load_graph(self, datastore, graph_name)

        # Specify dimensions of data
        self.num_mesh_nodes, _ = self.get_num_mesh()
        utils.log_on_rank_zero(
            f"Loaded graph with {self.num_grid_nodes + self.num_mesh_nodes} "
            f"nodes ({self.num_grid_nodes} grid, {self.num_mesh_nodes} mesh)"
        )

        # Compute grid_input_dim: total input dimensionality on the grid
        self.grid_input_dim = utils.compute_grid_input_dim(
            datastore,
            num_past_forcing_steps,
            num_future_forcing_steps,
        )

        self.g2m_edges, g2m_dim = self.g2m_features.shape
        self.m2g_edges, m2g_dim = self.m2g_features.shape

        # Define sub-models
        # Feature embedders for grid
        self.mlp_blueprint_end = [hidden_dim] * (hidden_layers + 1)
        # Blueprint for grid-level MLPs, same as above unless hidden_dim_grid
        self.grid_mlp_blueprint_end = [self.hidden_dim_grid] * (
            hidden_layers + 1
        )
        self.grid_embedder = utils.make_mlp(
            [self.grid_input_dim] + self.grid_mlp_blueprint_end
        )

        # Boundary forcing
        self.boundary_forced = datastore_boundary is not None
        if datastore_boundary is not None:
            self._init_boundary(
                datastore,
                datastore_boundary,
                num_past_boundary_steps,
                num_future_boundary_steps,
                time_delta_enc_dim,
                shared_grid_embedder,
            )

        self.g2m_embedder = utils.make_mlp(
            [g2m_dim] + self.grid_mlp_blueprint_end
        )
        self.m2g_embedder = utils.make_mlp(
            [m2g_dim] + self.grid_mlp_blueprint_end
        )

        # Projections between grid and mesh hidden dims, only created when
        # a separate grid dimensionality was requested
        if hidden_dim_grid is not None:
            self.pre_mesh_proj: nn.Module = nn.Sequential(
                nn.SiLU(), nn.Linear(self.hidden_dim_grid, hidden_dim)
            )
            self.post_mesh_proj: nn.Module = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_dim, self.hidden_dim_grid)
            )
        else:
            self.pre_mesh_proj = nn.Identity()
            self.post_mesh_proj = nn.Identity()

        # GNNs
        # encoder
        self.g2m_gnn = get_gnn_class(g2m_gnn_type)(
            self.g2m_edge_index,
            self.hidden_dim_grid,
            hidden_layers=hidden_layers,
            update_edges=False,
            num_rec=self._num_grid_connected_mesh_nodes(),
        )
        self.encoding_grid_mlp = utils.make_mlp(
            [self.hidden_dim_grid] + self.grid_mlp_blueprint_end
        )

        # decoder, only decoding to the interior grid nodes
        self.m2g_gnn = get_gnn_class(m2g_gnn_type)(
            self.m2g_edge_index,
            self.hidden_dim_grid,
            hidden_layers=hidden_layers,
            update_edges=False,
            num_rec=self.num_grid_nodes,
        )

        # Output mapping (hidden_dim -> output_dim)
        self.output_map = utils.make_mlp(
            [self.hidden_dim_grid] * (hidden_layers + 1)
            + [self.grid_output_dim],
            layer_norm=False,
        )  # No layer norm on this one

        # Compute indices and define clamping functions
        self.prepare_clamping_params(datastore)

        # Gating of selected features
        self._init_gating(datastore, gated_features, gate_thresholds)

        # Additional domains (multi-domain training)
        self._active_domain = 0
        self._gnn_graph_map: list[tuple[InteractionNet, str, int | None]] = []
        self.domain_graphs = nn.ModuleList()
        if additional_domains:
            self.domain_graphs.append(self._domain_graph_from_self())
            for (
                domain_datastore,
                domain_datastore_boundary,
                domain_graph_name,
            ) in additional_domains:
                self.domain_graphs.append(
                    self._load_domain_graph(
                        domain_datastore,
                        domain_datastore_boundary,
                        domain_graph_name,
                    )
                )

    def _num_grid_connected_mesh_nodes(self) -> int:
        """Number of mesh nodes the grid encodes to (the bottom level of a
        hierarchical mesh, else all mesh nodes), which need not all have an
        incoming grid edge.

        Returns
        -------
        int
            Number of g2m receiver nodes.
        """
        mesh_static_features = self.mesh_static_features
        if isinstance(mesh_static_features, torch.Tensor):
            return mesh_static_features.shape[0]
        return mesh_static_features[0].shape[0]

    @property
    def predicts_std(self) -> bool:
        """
        Whether this predictor outputs a predicted standard deviation (or
        quantiles, which are returned in its place).

        Returns
        -------
        bool
            ``True`` if a std. or quantiles are predicted.
        """
        return self.output_std or self.num_quantiles > 0

    @property
    def predicts_quantiles(self) -> bool:
        """
        Whether this predictor outputs quantiles instead of a std.

        Returns
        -------
        bool
            ``True`` if quantiles are predicted.
        """
        return self.num_quantiles > 0

    def _load_graph(
        self, module: nn.Module, datastore: BaseDatastore, graph_name: str
    ) -> bool:
        """Load graph ``graph_name`` for ``datastore`` onto ``module``.

        Parameters
        ----------
        module : nn.Module
            Module to register the graph tensors on.
        datastore : BaseDatastore
            Interior datastore of the domain.
        graph_name : str
            Name of the graph.

        Returns
        -------
        bool
            Whether the graph is hierarchical.
        """
        grid_xy_extent = datastore.get_xy_extent(category="state")
        grid_xy_max_span = max(
            grid_xy_extent[1] - grid_xy_extent[0],
            grid_xy_extent[3] - grid_xy_extent[2],
        )
        return utils.load_and_register_graph(
            module,
            datastore,
            graph_name,
            mesh_node_features_scaling=grid_xy_max_span,
            graph_dir=self.graph_dir,
        )

    def _init_boundary(
        self,
        datastore: BaseDatastore,
        datastore_boundary: BaseDatastore,
        num_past_boundary_steps: int,
        num_future_boundary_steps: int,
        time_delta_enc_dim: int | None,
        shared_grid_embedder: bool,
    ) -> None:
        """Set up embedding of boundary forcing and its time deltas.

        Parameters
        ----------
        datastore : BaseDatastore
            Interior datastore.
        datastore_boundary : BaseDatastore
            Boundary datastore.
        num_past_boundary_steps, num_future_boundary_steps : int
            Boundary forcing window.
        time_delta_enc_dim : int or None
            Encoding dimensionality per time delta.
        shared_grid_embedder : bool
            Whether to share the interior grid embedder.
        """
        self.register_buffer(
            "boundary_static_features",
            _load_static_features(datastore_boundary, "forcing"),
            persistent=False,
        )
        self.num_boundary_nodes, boundary_static_dim = (
            self.boundary_static_features.shape
        )
        num_boundary_forcing_vars = datastore_boundary.get_num_data_vars(
            category="forcing"
        )
        self.time_delta_enc_dim = (
            self.hidden_dim
            if time_delta_enc_dim is None
            else int(time_delta_enc_dim)
        )
        if self.time_delta_enc_dim % 2 != 0:
            raise ValueError(
                "Number of dimensions to use for time delta encoding must be "
                "even (sin and cos)"
            )
        # Number of trailing boundary features holding time deltas
        self.boundary_time_delta_dims = (
            num_past_boundary_steps + num_future_boundary_steps + 1
        )
        self.boundary_dim = (
            boundary_static_dim
            + (num_boundary_forcing_vars + self.time_delta_enc_dim)
            * self.boundary_time_delta_dims
        )

        if shared_grid_embedder:
            if self.grid_input_dim != self.boundary_dim:
                raise ValueError(
                    "Grid and boundary input dimension must be the same when "
                    "using the same embedder, got "
                    f"grid_input_dim={self.grid_input_dim}, "
                    f"boundary_dim={self.boundary_dim}"
                )
            self.boundary_embedder = self.grid_embedder
        else:
            self.boundary_embedder = utils.make_mlp(
                [self.boundary_dim] + self.grid_mlp_blueprint_end
            )

        # Constants for the sinusoidal time delta encoding
        step_length_ratio = (
            datastore_boundary.step_length / datastore.step_length
        )
        min_time_delta = -(num_past_boundary_steps + 1) * step_length_ratio
        max_time_delta = num_future_boundary_steps * step_length_ratio
        time_delta_magnitude = max(max_time_delta, abs(min_time_delta))
        freq_indices = 1.0 + torch.arange(
            self.time_delta_enc_dim // 2, dtype=torch.float
        )
        self.register_buffer(
            "enc_freq_denom",
            (2 * time_delta_magnitude)
            ** (2 * freq_indices / self.time_delta_enc_dim),
            persistent=False,
        )

    def _init_gating(
        self,
        datastore: BaseDatastore,
        gated_features: list[str] | None,
        gate_thresholds: dict[str, float] | None,
    ) -> None:
        """Set up the gate for gated features.

        Parameters
        ----------
        datastore : BaseDatastore
            Interior datastore.
        gated_features : list of str or None
            Names of gated state features.
        gate_thresholds : dict or None
            Physical thresholds per gated feature (default physical zero).
        """
        state_feature_names = datastore.get_vars_names(category="state")
        gated_features = list(gated_features or [])
        gate_thresholds = dict(gate_thresholds or {})
        unknown = set(gated_features) - set(state_feature_names)
        if unknown:
            raise ValueError(
                f"Gated features {unknown} are not state features "
                f"({state_feature_names})"
            )
        gated_indices = [state_feature_names.index(n) for n in gated_features]
        thresholds = [
            (
                gate_thresholds.get(state_feature_names[idx], 0.0)
                - self.state_mean[idx]
            )
            / self.state_std[idx]
            for idx in gated_indices
        ]
        self.register_buffer(
            "gated_indices",
            torch.tensor(gated_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "gate_thresholds",
            torch.tensor(thresholds, dtype=torch.float32),
            persistent=False,
        )
        if gated_indices:
            self.gate_map: nn.Module | None = utils.make_mlp(
                [self.hidden_dim_grid] * (self.hidden_layers + 1)
                + [len(gated_indices)],
                layer_norm=False,
            )
        else:
            self.gate_map = None

    def _domain_graph_from_self(self) -> DomainGraph:
        """Container referencing this model's (domain 0) graph tensors.

        Returns
        -------
        DomainGraph
            Container with the graph and static features of domain 0.
        """
        domain_graph = DomainGraph()
        for name in GRAPH_TENSOR_NAMES:
            value = getattr(self, name)
            if isinstance(value, torch.Tensor):
                domain_graph.register_buffer(name, value, persistent=False)
            else:
                setattr(domain_graph, name, value)
        domain_graph.register_buffer(
            "grid_static_features", self.grid_static_features, persistent=False
        )
        if self.boundary_forced:
            domain_graph.register_buffer(
                "boundary_static_features",
                self.boundary_static_features,
                persistent=False,
            )
        return domain_graph

    def _load_domain_graph(
        self,
        datastore: BaseDatastore,
        datastore_boundary: BaseDatastore | None,
        graph_name: str,
    ) -> DomainGraph:
        """Load the graph and static features of an additional domain.

        Parameters
        ----------
        datastore : BaseDatastore
            Interior datastore of the domain.
        datastore_boundary : BaseDatastore or None
            Boundary datastore of the domain.
        graph_name : str
            Name of the graph of the domain.

        Returns
        -------
        DomainGraph
            Container with the graph and static features of the domain.

        Raises
        ------
        ValueError
            If the domain does not match the setup of the first domain.
        """
        domain_graph = DomainGraph()
        hierarchical = self._load_graph(domain_graph, datastore, graph_name)
        if hierarchical != self.hierarchical:
            raise ValueError(
                f"Graph {graph_name} differs in hierarchy from the graph of "
                "the first domain."
            )
        domain_graph.register_buffer(
            "grid_static_features",
            _load_static_features(datastore, "state"),
            persistent=False,
        )
        if self.boundary_forced:
            if datastore_boundary is None:
                raise ValueError(
                    "All domains need a boundary datastore when the first "
                    "one has one."
                )
            domain_graph.register_buffer(
                "boundary_static_features",
                _load_static_features(datastore_boundary, "forcing"),
                persistent=False,
            )
        return domain_graph

    def _map_gnns_to_graph(self) -> None:
        """Find which graph edge index each GNN layer uses, by comparing
        its edge index with those of the current (domain 0) graph."""
        candidates: list[tuple[str, int | None, torch.Tensor]] = []
        for name in EDGE_INDEX_NAMES:
            value = getattr(self, name)
            if isinstance(value, torch.Tensor):
                candidates.append((name, None, value))
            else:
                candidates += [(name, i, ei) for i, ei in enumerate(value)]
        self._gnn_graph_map = []
        for module in self.modules():
            if not isinstance(module, InteractionNet):
                continue
            for name, level, edge_index in candidates:
                shifted = torch.stack(
                    (edge_index[0] + module.num_rec, edge_index[1]), dim=0
                )
                if shifted.shape == module.edge_index.shape and torch.equal(
                    shifted, module.edge_index
                ):
                    self._gnn_graph_map.append((module, name, level))
                    break
            else:
                raise RuntimeError(
                    "Could not determine which graph a GNN layer uses."
                )

    def set_domain(self, domain_idx: int) -> None:
        """
        Switch to the graph and static features of domain ``domain_idx``
        (0 is the domain of ``datastore``).

        Parameters
        ----------
        domain_idx : int
            Index of the domain, as tagged by the data module.

        Raises
        ------
        IndexError
            If no such domain was set up.
        """
        if domain_idx == self._active_domain:
            return
        if domain_idx >= len(self.domain_graphs):
            raise IndexError(
                f"Domain {domain_idx} requested but only "
                f"{max(len(self.domain_graphs), 1)} domain(s) set up."
            )
        if not self._gnn_graph_map:
            self._map_gnns_to_graph()

        domain_graph = self.domain_graphs[domain_idx]
        for name in GRAPH_TENSOR_NAMES + (
            "grid_static_features",
            "boundary_static_features",
        ):
            if hasattr(domain_graph, name):
                setattr(self, name, getattr(domain_graph, name))
        self.num_grid_nodes = self.grid_static_features.shape[0]
        for module, name, level in self._gnn_graph_map:
            value = getattr(self, name)
            edge_index = value if level is None else value[level]
            if name == "m2g_edge_index":
                num_rec: int | None = self.num_grid_nodes
            elif name == "g2m_edge_index":
                num_rec = self._num_grid_connected_mesh_nodes()
            else:
                num_rec = None
            module.set_edge_index(edge_index, num_rec=num_rec)
        self._active_domain = domain_idx

    def get_num_mesh(self) -> tuple[int, int]:
        """
        Compute number of mesh nodes from loaded features,
        and number of mesh nodes that should be ignored in encoding/decoding.

        Returns
        -------
        num_mesh_nodes : int
            The number of mesh nodes.
        num_ignore_mesh_nodes : int
            The number of mesh nodes to ignore.
        """
        raise NotImplementedError("get_num_mesh not implemented")

    def embedd_mesh_nodes(self) -> torch.Tensor:
        """
        Embed static mesh node features.

        Returns
        -------
        torch.Tensor
            Shape ``(num_mesh_nodes, hidden_dim)``. Embedded mesh node
            representations. Dims: ``num_mesh_nodes`` is the number of
            mesh nodes and ``hidden_dim`` is the hidden dimension.
        """
        raise NotImplementedError("embedd_mesh_nodes not implemented")

    def process_step(self, mesh_rep: torch.Tensor) -> torch.Tensor:
        """
        Process the mesh representation in the encode-process-decode
        framework, running one or more message-passing steps.

        Parameters
        ----------
        mesh_rep : torch.Tensor
            Shape ``(B, num_mesh_nodes, hidden_dim)``. Current mesh node
            representations. Dims: ``B`` is batch size,
            ``num_mesh_nodes`` is the number of mesh nodes, and ``hidden_dim``
            is the hidden dimension.

        Returns
        -------
        torch.Tensor
            Shape ``(B, num_mesh_nodes, hidden_dim)``. Updated mesh node
            representations. Dims: same as ``mesh_rep``.
        """
        raise NotImplementedError("process_step not implemented")

    def encode_forcing_time_deltas(
        self, boundary_forcing: torch.Tensor
    ) -> torch.Tensor:
        """
        Replace the time delta features of the boundary forcing with their
        sinusoidal encodings.

        Parameters
        ----------
        boundary_forcing : torch.Tensor
            Shape ``(B, num_boundary_nodes, num_boundary_forcing_dims)``,
            where the last ``boundary_time_delta_dims`` features are time
            deltas.

        Returns
        -------
        torch.Tensor
            Shape ``(B, num_boundary_nodes, num_encoded_dims)``, the forcing
            features followed by the encoded time deltas.
        """
        time_deltas = boundary_forcing[..., -self.boundary_time_delta_dims :]
        frequencies = time_deltas.unsqueeze(-1) / self.enc_freq_denom
        encodings_stacked = torch.cat(
            (torch.sin(frequencies), torch.cos(frequencies)), dim=-1
        )
        encoded_time_deltas = encodings_stacked.flatten(-2, -1)
        return torch.cat(
            (
                boundary_forcing[..., : -self.boundary_time_delta_dims],
                encoded_time_deltas,
            ),
            dim=-1,
        )

    def forward(
        self,
        prev_state: torch.Tensor,
        prev_prev_state: torch.Tensor,
        forcing: torch.Tensor,
        boundary_forcing: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Advance the state by one step using the encode-process-decode
        graph pipeline: embed grid (and boundary) + edge features, map
        grid -> mesh via the g2m GNN, run the (subclass-defined) processor
        on the mesh, decode mesh -> interior grid via the m2g GNN, project
        to a state delta, rescale with difference statistics, and add to
        ``prev_state`` (with optional gating and clamping).

        Parameters
        ----------
        prev_state : torch.Tensor
            Shape ``(B, num_grid_nodes, num_state_vars)``. The current
            state ``X_t``.
        prev_prev_state : torch.Tensor
            Shape ``(B, num_grid_nodes, num_state_vars)``. The previous state
            ``X_{t-1}``.
        forcing : torch.Tensor
            Shape ``(B, num_grid_nodes, num_forcing_vars)``. Windowed forcing
            for this step.
        boundary_forcing : torch.Tensor, optional
            Shape ``(B, num_boundary_nodes, num_boundary_forcing_dims)``.
            Windowed (standardized) boundary forcing with time delta
            features last. Required for boundary-forced models.

        Returns
        -------
        new_state : torch.Tensor
            Shape ``(B, num_grid_nodes, num_state_vars)``. The predicted
            next state ``X_{t+1}``.
        pred_std : torch.Tensor or None
            Shape ``(B, num_grid_nodes, num_state_vars)`` when ``output_std``,
            ``(B, num_grid_nodes, num_state_vars, num_quantiles)`` holding
            predicted quantiles (in state space) when ``num_quantiles > 0``,
            otherwise ``None``.
        """
        batch_size = prev_state.shape[0]
        self.step_aux = {}

        # Create full grid node features of shape (B, num_grid_nodes, grid_dim)
        grid_features = torch.cat(
            (
                prev_state,
                prev_prev_state,
                forcing,
                self.expand_to_batch(self.grid_static_features, batch_size),
            ),
            dim=-1,
        )

        # Embed all features
        grid_emb = self.grid_embedder(
            grid_features
        )  # (B, num_grid_nodes, hidden_dim_grid)

        if self.boundary_forced:
            if boundary_forcing is None or boundary_forcing.shape[-1] == 0:
                raise ValueError(
                    "This model is boundary-forced but no boundary forcing "
                    "was given."
                )
            boundary_features = torch.cat(
                (
                    self.encode_forcing_time_deltas(boundary_forcing),
                    self.expand_to_batch(
                        self.boundary_static_features, batch_size
                    ),
                ),
                dim=-1,
            )
            boundary_emb = self.boundary_embedder(boundary_features)
            # Enforce ordering (interior, boundary) of grid nodes
            full_grid_emb = torch.cat((grid_emb, boundary_emb), dim=1)
        else:
            full_grid_emb = grid_emb

        g2m_emb = self.g2m_embedder(self.g2m_features)
        m2g_emb = self.m2g_embedder(self.m2g_features)
        mesh_emb = self.embedd_mesh_nodes()

        # Map from grid to mesh
        mesh_emb_expanded = self.expand_to_batch(mesh_emb, batch_size)
        g2m_emb_expanded = self.expand_to_batch(g2m_emb, batch_size)
        mesh_rep = self.g2m_gnn(
            full_grid_emb, mesh_emb_expanded, g2m_emb_expanded
        )  # (B, num_mesh_nodes, hidden_dim_grid)
        # Also MLP with residual for grid representation
        grid_rep = grid_emb + self.encoding_grid_mlp(
            grid_emb
        )  # (B, num_grid_nodes, hidden_dim_grid)

        # Run processor step on the mesh in hidden_dim
        mesh_rep = self.pre_mesh_proj(mesh_rep)
        mesh_rep = self.process_step(mesh_rep)
        mesh_rep = self.post_mesh_proj(mesh_rep)

        # Map back from mesh to grid
        m2g_emb_expanded = self.expand_to_batch(m2g_emb, batch_size)
        grid_rep = self.m2g_gnn(
            mesh_rep, grid_rep, m2g_emb_expanded
        )  # (B, num_grid_nodes, hidden_dim_grid)

        # Map to output dimension, only for grid
        net_output = self.output_map(
            grid_rep
        )  # (B, num_grid_nodes, grid_output_dim)

        pred_quantile_deltas = None
        if self.output_std:
            pred_delta_mean, pred_std_raw = net_output.chunk(
                2, dim=-1
            )  # both (B, num_grid_nodes, num_state_vars)
            # NOTE: The predicted std. is not scaled in any way here
            # linter for some reason does not think softplus is callable
            # pylint: disable-next=not-callable
            pred_std = torch.nn.functional.softplus(pred_std_raw)
        elif self.num_quantiles > 0:
            pred_quantile_deltas = net_output.view(
                batch_size, net_output.shape[1], -1, self.num_quantiles
            )  # (B, num_grid_nodes, num_state_vars, num_quantiles)
            # Median as point prediction
            mid_idx = self.num_quantiles // 2
            if self.num_quantiles % 2 == 1:
                pred_delta_mean = pred_quantile_deltas[..., mid_idx]
            else:
                pred_delta_mean = (
                    pred_quantile_deltas[..., mid_idx - 1]
                    + pred_quantile_deltas[..., mid_idx]
                ) / 2
            pred_std = None
        else:
            pred_delta_mean = net_output
            pred_std = None

        # Rescale with one-step difference statistics
        rescaled_delta_mean = pred_delta_mean * self.diff_std + self.diff_mean

        if self.gate_map is not None:
            gate_logits = self.gate_map(grid_rep)  # (B, N, num_gated)
            gate_probs = torch.sigmoid(gate_logits)
            # When the gate is closed the increment becomes the threshold
            rescaled_delta_mean = rescaled_delta_mean.clone()
            rescaled_delta_mean[:, :, self.gated_indices] = (
                1.0 - gate_probs
            ) * self.gate_thresholds + gate_probs * rescaled_delta_mean[
                :, :, self.gated_indices
            ]
            self.step_aux["gate_logits"] = gate_logits

        # Clamp values to valid range (also add the delta to the previous state)
        new_state = self.get_clamped_new_state(rescaled_delta_mean, prev_state)

        if pred_quantile_deltas is not None:
            rescaled_quantile_deltas = (
                pred_quantile_deltas * self.diff_std.unsqueeze(-1)
                + self.diff_mean.unsqueeze(-1)
            )
            pred_std = torch.stack(
                [
                    self.get_clamped_new_state(
                        rescaled_quantile_deltas[..., q_i], prev_state
                    )
                    for q_i in range(self.num_quantiles)
                ],
                dim=-1,
            )

        return new_state, pred_std
