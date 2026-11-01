"""Helpers for the combined interior + boundary grid node set."""

# Third-party
import cartopy.crs as ccrs
import numpy as np

# Local
from ..datastore.base import BaseDatastore


def get_stacked_lat_lons(
    datastore: BaseDatastore, datastore_boundary: BaseDatastore | None = None
) -> np.ndarray:
    """
    Stack the lon-lat coordinates of all grid nodes in the ordering used by
    the models: interior nodes first, then boundary nodes.

    Parameters
    ----------
    datastore : BaseDatastore
        The datastore containing data for the interior region of the grid.
    datastore_boundary : BaseDatastore or None
        The datastore containing data for boundary forcing, if any.

    Returns
    -------
    np.ndarray
        Array of all coordinates, shape ``(num_total_grid_nodes, 2)``.
    """
    grid_coords = datastore.get_lat_lon(category="state")
    if datastore_boundary is None:
        return grid_coords
    boundary_coords = datastore_boundary.get_lat_lon(category="forcing")
    return np.concatenate((grid_coords, boundary_coords), axis=0)


def project_lat_lons(
    lat_lons: np.ndarray, datastore: BaseDatastore
) -> np.ndarray:
    """
    Project an ``(N, 2)`` array of lon-lat coordinates to the CRS of the
    given datastore.

    Parameters
    ----------
    lat_lons : np.ndarray
        Lon-lat coordinates, shape ``(N, 2)``.
    datastore : BaseDatastore
        Datastore whose ``coords_projection`` is used.

    Returns
    -------
    np.ndarray
        Projected coordinates, shape ``(N, 2)``.
    """
    xyz = datastore.coords_projection.transform_points(
        ccrs.PlateCarree(), lat_lons[:, 0], lat_lons[:, 1]
    )
    return xyz[:, :2]


def get_stacked_xy(
    datastore: BaseDatastore, datastore_boundary: BaseDatastore | None = None
) -> np.ndarray:
    """
    Stack the coordinates of all grid nodes (interior first, then boundary)
    in the CRS of the interior datastore.

    Parameters
    ----------
    datastore : BaseDatastore
        The datastore containing data for the interior region of the grid.
    datastore_boundary : BaseDatastore or None
        The datastore containing data for boundary forcing, if any.

    Returns
    -------
    np.ndarray
        Array of all coordinates, shape ``(num_total_grid_nodes, 2)``.
    """
    return project_lat_lons(
        get_stacked_lat_lons(datastore, datastore_boundary), datastore
    )


def get_interior_mask(
    datastore: BaseDatastore, datastore_boundary: BaseDatastore
) -> np.ndarray:
    """
    Boolean mask over the stacked grid nodes, True for interior nodes.

    Parameters
    ----------
    datastore : BaseDatastore
        The datastore containing data for the interior region of the grid.
    datastore_boundary : BaseDatastore
        The datastore containing data for boundary forcing.

    Returns
    -------
    np.ndarray
        Boolean array, shape ``(num_total_grid_nodes,)``.
    """
    return np.concatenate(
        (
            np.ones(datastore.num_grid_points, dtype=bool),
            np.zeros(datastore_boundary.num_grid_points, dtype=bool),
        ),
        axis=0,
    )
