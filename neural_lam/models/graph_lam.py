# Standard library
from typing import Union

# Third-party
from torch import nn

# Local
from .. import utils
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore
from ..graph_data import GraphSizes
from ..interaction_net import InteractionNet
from .base_graph_model import BaseGraphModel


class GraphLAM(BaseGraphModel):
    """
    Full graph-based LAM model that can be used with different
    (non-hierarchical )graphs. Mainly based on GraphCast, but the model from
    Keisler (2022) is almost identical. Used for GC-LAM and L1-LAM in
    Oskarsson et al. (2023).
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
            graph_sizes=graph_sizes,
        )

        assert (
            not self.hierarchical
        ), "GraphLAM does not use a hierarchical mesh graph"

        mesh_dim = self.graph_sizes.mesh_feature_dims[0]
        m2m_dim = self.graph_sizes.m2m_feature_dims[0]

        # Define sub-models
        # Feature embedders for mesh
        # Bottom mesh level is first embedded to hidden dim of grid
        self.mesh_embedder = utils.make_mlp(
            [mesh_dim] + self.grid_mlp_blueprint_end
        )
        self.m2m_embedder = utils.make_mlp([m2m_dim] + self.mlp_blueprint_end)

        # GNNs
        # processor
        if args.processor_layers == 0:
            self.processor_nets = None
        else:
            self.processor_nets = nn.ModuleList(
                [
                    InteractionNet(
                        args.hidden_dim,
                        hidden_layers=args.hidden_layers,
                        aggr=args.mesh_aggr,
                    )
                    for _ in range(args.processor_layers)
                ]
            )

    @property
    def num_mesh_nodes(self):
        """
        Get the total number of mesh nodes in the used mesh graph
        """
        return self.graph_sizes.num_mesh_nodes

    @property
    def num_grid_connected_mesh_nodes(self):
        """
        Get the total number of mesh nodes that have a connection to
        the grid (e.g. bottom level in a hierarchy)
        """
        return (
            self.graph_sizes.num_mesh_nodes
        )  # All nodes ?????? This is now the same as num_mesh_nodes

    def embedd_mesh_nodes(self):
        """
        Embed static mesh features
        Returns tensor of shape (N_mesh, d_h)
        """
        if self.current_graph is None:
            raise RuntimeError(
                "Graph data not set before embedding mesh nodes."
            )
        return self.mesh_embedder(
            self.current_graph["mesh_static_features"]
        )  # (N_mesh, d_h)

    def process_step(self, mesh_rep):
        """
        Process step of embedd-process-decode framework
        Processes the representation on the mesh, possible in multiple steps

        mesh_rep: has shape (B, N_mesh, d_h)
        Returns mesh_rep: (B, N_mesh, d_h)
        """
        # Embed m2m here first
        batch_size = mesh_rep.shape[0]
        if self.current_graph is None:
            raise RuntimeError("Graph data not set before processing mesh.")
        m2m_emb = self.m2m_embedder(
            self.current_graph["m2m_features"]
        )  # (M_mesh, d_h)
        m2m_emb_expanded = self.expand_to_batch(
            m2m_emb, batch_size
        )  # (B, M_mesh, d_h)

        if self.processor_nets is not None:
            for net in self.processor_nets:
                mesh_rep, m2m_emb_expanded = net(
                    mesh_rep,
                    mesh_rep,
                    m2m_emb_expanded,
                    self.current_graph["m2m_edge_index"],
                )
        # (B, N_mesh, d_h)
        return mesh_rep
