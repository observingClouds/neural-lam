"""Build rectangular graphs with weather-model-graphs for interior and
(optional) boundary grid nodes.

Grid nodes are ordered with the interior nodes first and the boundary nodes
(from a separate boundary datastore) last. All grid nodes send to the mesh,
but the mesh only decodes to the interior nodes. Graphs are saved in the
neural-lam graph storage format (see ``docs/graph_storage_spec.md``).
"""

# Standard library
import argparse
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Third-party
import cartopy.crs as ccrs
import weather_model_graphs as wmg
from loguru import logger

# Local
from . import utils
from .config import load_config_and_datastore
from .datastore.base import BaseDatastore

WMG_ARCHETYPES = {
    "keisler": wmg.create.archetype.create_keisler_graph,
    "graphcast": wmg.create.archetype.create_graphcast_graph,
    "hierarchical": wmg.create.archetype.create_oskarsson_hierarchical_graph,
}


def main(input_args: list[str] | None = None) -> None:
    """
    Build rectangular graph from archetype, using cmd-line arguments.

    Parameters
    ----------
    input_args : list of str, optional
        Arguments to parse instead of ``sys.argv``.
    """
    parser = argparse.ArgumentParser(
        description="Rectangular graph generation using weather-models-graph",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Inputs and outputs
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the configuration for neural-lam",
    )
    parser.add_argument(
        "--graph_name",
        type=str,
        required=True,
        help="Name to save graph as",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save graph to (default: <datastore root>/graph)",
    )

    # Graph structure
    parser.add_argument(
        "--archetype",
        type=str,
        default="keisler",
        choices=list(WMG_ARCHETYPES),
        help="Archetype to use to create graph",
    )
    parser.add_argument(
        "--mesh_node_distance",
        type=float,
        default=3.0,
        help="Distance between created mesh nodes, in the coordinates of "
        "the projection of the interior datastore",
    )
    parser.add_argument(
        "--level_refinement_factor",
        type=int,
        default=3,
        help="Refinement factor between grid points and bottom level of "
        "mesh hierarchy",
    )
    parser.add_argument(
        "--max_num_levels",
        type=int,
        help="Limit multi-scale mesh to given number of levels, from bottom up",
    )
    args = parser.parse_args(input_args)

    _, datastore, datastore_boundary = load_config_and_datastore(
        config_path=args.config_path
    )

    create_kwargs: dict[str, Any] = {
        "mesh_node_distance": args.mesh_node_distance,
    }
    if args.archetype != "keisler":
        # Add additional multi-level kwargs
        create_kwargs.update(
            {
                "level_refinement_factor": args.level_refinement_factor,
                "max_num_levels": args.max_num_levels,
            }
        )

    build_graph_from_archetype(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        graph_name=args.graph_name,
        archetype=args.archetype,
        dir_save_path=args.output_dir,
        **create_kwargs,
    )


def _build_wmg_graph(
    datastore: BaseDatastore,
    datastore_boundary: BaseDatastore | None,
    graph_build_func: Callable[..., Any],
    kwargs: dict[str, Any],
    graph_name: str,
    dir_save_path: str | Path | None = None,
    hierarchical: bool = False,
) -> None:
    """
    Build a graph using WMG in a way that's compatible with neural-lam.
    Given datastores are used for coordinates and decode masking.

    Parameters
    ----------
    datastore : BaseDatastore
        Datastore representing interior region of grid.
    datastore_boundary : BaseDatastore or None
        Datastore representing boundary region, or None if no boundary
        forcing.
    graph_build_func : callable
        Function from WMG to use to build graph.
    kwargs : dict
        Keyword arguments to feed to ``graph_build_func``. Should not include
        coords, coords_crs, graph_crs, return_components or decode_mask, as
        these are here derived in a consistent way from the datastores.
    graph_name : str
        Name to save the graph as.
    dir_save_path : str or Path, optional
        Directory in which the graph directory ``graph_name`` is created.
        If None, ``<datastore.root_path>/graph`` is used.
    hierarchical : bool, optional
        Whether the built graph is hierarchical.

    Raises
    ------
    ValueError
        If any of the derived arguments is given in ``kwargs``.
    """
    for derived_kwarg in (
        "coords",
        "coords_crs",
        "graph_crs",
        "return_components",
        "decode_mask",
    ):
        if derived_kwarg in kwargs:
            raise ValueError(
                f"Argument {derived_kwarg} should not be manually given when "
                "building rectangular graph."
            )

    # Lon-lat positions of all grid nodes, (num_nodes_full, 2)
    coords = utils.get_stacked_lat_lons(datastore, datastore_boundary)

    # Only decode to the interior if there is a boundary
    decode_mask = (
        None
        if datastore_boundary is None
        else utils.get_interior_mask(datastore, datastore_boundary)
    )

    create_kwargs = {
        "coords": coords,
        "decode_mask": decode_mask,
        # Project to the crs of the interior datastore for graph building
        "graph_crs": datastore.coords_projection,
        "coords_crs": ccrs.PlateCarree(),
        "return_components": True,
    }
    create_kwargs.update(kwargs)

    graph_comp = graph_build_func(**create_kwargs)

    logger.info("Created graph:")
    for name, subgraph in graph_comp.items():
        logger.info(f"{name}: {subgraph}")

    graph_root = (
        Path(dir_save_path)
        if dir_save_path is not None
        else Path(datastore.root_path) / "graph"
    )
    graph_dir_path = graph_root / graph_name
    os.makedirs(graph_dir_path, exist_ok=True)
    wmg.save.to_torch_tensors_on_disk(
        graph_components=graph_comp,
        output_directory=str(graph_dir_path),
        hierarchical=hierarchical,
    )
    logger.info(f"Saved graph to {graph_dir_path}")


def build_graph_from_archetype(
    datastore: BaseDatastore,
    datastore_boundary: BaseDatastore | None,
    graph_name: str,
    archetype: str,
    dir_save_path: str | Path | None = None,
    **kwargs: Any,
) -> None:
    """
    Build a graph using a wmg archetype. Keyword arguments are passed on
    directly to the archetype function.

    Parameters
    ----------
    datastore : BaseDatastore
        Datastore representing interior region of grid.
    datastore_boundary : BaseDatastore or None
        Datastore representing boundary region, or None if no boundary
        forcing.
    graph_name : str
        Name to save the graph as.
    archetype : str
        Archetype to build, one of "keisler", "graphcast" or "hierarchical".
    dir_save_path : str or Path, optional
        Directory in which the graph directory ``graph_name`` is created.
        If None, ``<datastore.root_path>/graph`` is used.
    **kwargs
        Keyword arguments passed on to the wmg archetype function.

    Raises
    ------
    ValueError
        If the archetype is unknown.
    """
    if archetype not in WMG_ARCHETYPES:
        raise ValueError(f"Unknown archetype: {archetype}")

    _build_wmg_graph(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        graph_build_func=WMG_ARCHETYPES[archetype],
        graph_name=graph_name,
        dir_save_path=dir_save_path,
        kwargs=kwargs,
        hierarchical=archetype == "hierarchical",
    )


def build_graph(
    datastore: BaseDatastore,
    datastore_boundary: BaseDatastore | None,
    graph_name: str,
    dir_save_path: str | Path | None = None,
    **kwargs: Any,
) -> None:
    """
    Build a graph with fine-grained control, directly using
    ``wmg.create.base.create_all_graph_components`` with the given kwargs.

    Parameters
    ----------
    datastore : BaseDatastore
        Datastore representing interior region of grid.
    datastore_boundary : BaseDatastore or None
        Datastore representing boundary region, or None if no boundary
        forcing.
    graph_name : str
        Name to save the graph as.
    dir_save_path : str or Path, optional
        Directory in which the graph directory ``graph_name`` is created.
        If None, ``<datastore.root_path>/graph`` is used.
    **kwargs
        Keyword arguments passed on to
        ``wmg.create.base.create_all_graph_components``.
    """
    _build_wmg_graph(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        graph_build_func=wmg.create.base.create_all_graph_components,
        graph_name=graph_name,
        dir_save_path=dir_save_path,
        kwargs=kwargs,
        hierarchical=kwargs.get("m2m_connectivity") == "hierarchical",
    )


if __name__ == "__main__":
    main()
