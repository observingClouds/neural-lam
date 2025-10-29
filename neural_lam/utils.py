# Standard library
import os
import shutil
import warnings

# Third-party
import cartopy.crs as ccrs
import numpy as np
import pytorch_lightning as pl
import torch
import urllib3
from pytorch_lightning.loggers import MLFlowLogger, WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import nn
from tueplots import bundles, figsizes

# Local
from .custom_loggers import CustomMLFlowLogger


class BufferList(nn.Module):
    """
    A list of torch buffer tensors that sit together as a Module with no
    parameters and only buffers.

    This should be replaced by a native torch BufferList once implemented.
    See: https://github.com/pytorch/pytorch/issues/37386
    """

    def __init__(self, buffer_tensors, persistent=True):
        super().__init__()
        self.n_buffers = len(buffer_tensors)
        for buffer_i, tensor in enumerate(buffer_tensors):
            self.register_buffer(f"b{buffer_i}", tensor, persistent=persistent)

    def __getitem__(self, key):
        return getattr(self, f"b{key}")

    def __setitem__(self, key, value):
        return setattr(self, f"b{key}", value)

    def __len__(self):
        return self.n_buffers

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __itruediv__(self, other):
        """Divide each element in list with other"""
        return self.__imul__(1.0 / other)

    def __imul__(self, other):
        """Multiply each element in list with other"""
        for buffer_tensor in self:
            buffer_tensor *= other

        return self


def zero_index_edge_index(edge_index):
    """
    Make both sender and receiver indices of edge_index start at 0
    """
    return edge_index - edge_index.min(dim=1, keepdim=True)[0]


def make_mlp(blueprint, layer_norm=True):
    """
    Create MLP from list blueprint, with
    input dimensionality: blueprint[0]
    output dimensionality: blueprint[-1] and
    hidden layers of dimensions: blueprint[1], ..., blueprint[-2]

    if layer_norm is True, includes a LayerNorm layer at
    the output (as used in GraphCast)
    """
    hidden_layers = len(blueprint) - 2
    assert hidden_layers >= 0, "Invalid MLP blueprint"

    layers = []
    for layer_i, (dim1, dim2) in enumerate(zip(blueprint[:-1], blueprint[1:])):
        layers.append(nn.Linear(dim1, dim2))
        if layer_i != hidden_layers:
            layers.append(nn.SiLU())  # Swish activation

    # Optionally add layer norm to output
    if layer_norm:
        layers.append(nn.LayerNorm(blueprint[-1]))

    return nn.Sequential(*layers)


def fractional_plot_bundle(fraction):
    """
    Get the tueplots bundle, but with figure width as a fraction of
    the page width.
    """
    # If latex is not available, some visualizations might not render
    # correctly, but will at least not raise an error. Alternatively, use
    # unicode raised numbers.
    usetex = True if shutil.which("latex") else False
    bundle = bundles.neurips2023(usetex=usetex, family="serif")
    bundle.update(figsizes.neurips2023())
    original_figsize = bundle["figure.figsize"]
    bundle["figure.figsize"] = (
        original_figsize[0] / fraction,
        original_figsize[1],
    )
    return bundle


@rank_zero_only
def rank_zero_print(*args, **kwargs):
    """Print only from rank 0 process"""
    print(*args, **kwargs)


def get_stacked_lat_lons(datastore, datastore_boundary=None):
    """
    Stack the lat-lon coordinates of all grid nodes in the correct ordering

    Parameters
    ----------
    datastore : BaseDatastore
        The datastore containing data for the interior region of the grid
    datastore_boundary : BaseDatastore or None
        (Optional) The datastore containing data for boundary forcing

    Returns
    -------
    stacked_coords : np.ndarray
        Array of all coordinates, shaped (num_total_grid_nodes, 2)
    """
    grid_coords = datastore.get_lat_lon(category="state")

    if datastore_boundary is None:
        return grid_coords

    # Append boundary forcing positions last
    boundary_coords = datastore_boundary.get_lat_lon(category="forcing")
    return np.concatenate((grid_coords, boundary_coords), axis=0)


def get_stacked_xy(datastore, datastore_boundary=None):
    """
    Stack the xy coordinates of all grid nodes in the correct ordering,
    with xy coordinates being in the CRS of the datastore

    Parameters
    ----------
    datastore : BaseDatastore
        The datastore containing data for the interior region of the grid
    datastore_boundary : BaseDatastore or None
        (Optional) The datastore containing data for boundary forcing

    Returns
    -------
    stacked_coords : np.ndarray
        Array of all coordinates, shaped (num_total_grid_nodes, 2)
    """
    lat_lons = get_stacked_lat_lons(datastore, datastore_boundary)

    # transform to datastore CRS
    xyz = datastore.coords_projection.transform_points(
        ccrs.PlateCarree(), lat_lons[:, 0], lat_lons[:, 1]
    )
    return xyz[:, :2]


def project_lat_lons(lat_lons, datastore):
    """
    Project given (N, 2) array of latlon coordinates to CRS of given datastore.
    """
    xyz = datastore.coords_projection.transform_points(
        ccrs.PlateCarree(), lat_lons[:, 0], lat_lons[:, 1]
    )
    return xyz[:, :2]


def get_interior_mask(datastore, datastore_boundary):
    """
    Get a binary mask of same length as stacked xy or lat_lons, where a True
    entry means that the coordinate is part of the interior.

    Parameters
    ----------
    datastore : BaseDatastore
        The datastore containing data for the interior region of the grid
    datastore_boundary : BaseDatastore
        The datastore containing data for boundary forcing

    Returns
    -------
    interior_mask : np.ndarray[bool]
        Array of boolean values , shaped (num_total_grid_nodes,)
    """
    # Construct mask to decode only to interior
    num_interior = datastore.num_grid_points
    num_boundary = datastore_boundary.num_grid_points
    return np.concatenate(
        (
            np.ones(num_interior, dtype=bool),
            np.zeros(num_boundary, dtype=bool),
        ),
        axis=0,
    )


def get_time_step(times):
    """Calculate the time step from a time dataarray.

    Parameters
    ----------
    times : xr.DataArray
        The time dataarray to calculate the time step from.

    Returns
    -------
    time_step : float
        The time step in the the datetime-format of the times dataarray.
    """
    time_diffs = np.diff(times)
    if not np.all(time_diffs == time_diffs[0]):
        raise ValueError(
            "Inconsistent time steps in data. "
            f"Found different time steps: {np.unique(time_diffs)}"
        )
    return time_diffs[0]


def check_time_overlap(
    da1,
    da2,
    da1_is_forecast=False,
    da2_is_forecast=False,
    num_past_steps=1,
    num_future_steps=1,
):
    """Check that the time coverage of two dataarrays overlap.

    Parameters
    ----------
    da1 : xr.DataArray
        The first dataarray to check.
    da2 : xr.DataArray
        The second dataarray to check.
    da1_is_forecast : bool, optional
        Whether the first dataarray is forecast data.
    da2_is_forecast : bool, optional
        Whether the second dataarray is forecast data.
    num_past_steps : int, optional
        Number of past forcing steps.
    num_future_steps : int, optional
        Number of future forcing steps.

    Raises
    ------
    ValueError
        If the time coverage of the dataarrays does not overlap.
    """

    if da1_is_forecast:
        times_da1 = da1.analysis_time
    else:
        times_da1 = da1.time
    time_min_da1 = times_da1.min().values
    time_max_da1 = times_da1.max().values

    if da2_is_forecast:
        times_da2 = da2.analysis_time
        time_min_da2 = times_da2.min().values
        time_max_da2 = times_da2.max().values

        time_step_da2 = get_time_step(times_da2.values)
        time_step_da1 = get_time_step(times_da1.values)

        analysis_offset = max(time_step_da1, num_past_steps * time_step_da2)
        da2_required_time_min = time_min_da1 - analysis_offset
        da2_required_time_max = time_max_da1 - analysis_offset
    else:
        times_da2 = da2.time
        time_min_da2 = times_da2.min().values
        time_max_da2 = times_da2.max().values
        time_step_da2 = get_time_step(times_da2.values)

        # Calculate required bounds for da2 using its time step
        da2_required_time_min = time_min_da1 - num_past_steps * time_step_da2
        da2_required_time_max = time_max_da1 + num_future_steps * time_step_da2

    if time_min_da2 > da2_required_time_min:
        raise ValueError(
            f"The second DataArray (e.g. 'boundary forcing') starts too late."
            f"Required start: {da2_required_time_min}, "
            f"but DataArray starts at {time_min_da2}."
        )

    if time_max_da2 < da2_required_time_max:
        raise ValueError(
            f"The second DataArray (e.g. 'boundary forcing') ends too early."
            f"Required end: {da2_required_time_max}, "
            f"but DataArray ends at {time_max_da2}."
        )


def crop_time_if_needed(
    da1,
    da2,
    da1_is_forecast=False,
    da2_is_forecast=False,
    num_past_steps=1,
    num_future_steps=1,
):
    """
    Slice away the first few timesteps from the first DataArray (e.g. 'state')
    if the second DataArray (e.g. boundary forcing) does not cover that range
    (including num_past_steps).

    Parameters
    ----------
    da1 : xr.DataArray
        The first DataArray to crop.
    da2 : xr.DataArray
        The second DataArray to compare against.
    da1_is_forecast : bool, optional
        Whether the first dataarray is forecast data.
    da2_is_forecast : bool, optional
        Whether the second dataarray is forecast data.
    num_past_steps : int
        Number of past time steps to consider.
    num_future_steps : int
        Number of future time steps to consider.

    Return
    ------
    da1 : xr.DataArray
        The cropped first DataArray and print a warning if any steps are
        removed.
    """
    # NOTE: Now this does not consider the ar_step at the end,
    # or the 2 init steps
    if da1 is None or da2 is None:
        return da1

    try:
        check_time_overlap(
            da1,
            da2,
            da1_is_forecast,
            da2_is_forecast,
            num_past_steps,
            num_future_steps,
        )
        return da1
    except ValueError:
        # If da2 coverage is insufficient, remove earliest da1 times
        # until coverage is possible. Figure out how many steps to remove.
        if da1_is_forecast:
            da1_tvals = da1.analysis_time.values
        else:
            da1_tvals = da1.time.values
        if da2_is_forecast:
            da2_tvals = da2.analysis_time.values
        else:
            da2_tvals = da2.time.values

        # Calculate how many steps we would have to remove
        da2_dt = get_time_step(da2_tvals)
        if da2_is_forecast:
            da1_dt = get_time_step(da1_tvals)
            # analysis time of boundary forecast must start this much earlier
            # than da1 timestep
            analysis_offset = max(da1_dt, num_past_steps * da2_dt)
            required_min = da2_tvals[0] + analysis_offset
            required_max = da2_tvals[-1] + analysis_offset
        else:
            required_min = da2_tvals[0] + num_past_steps * da2_dt
            required_max = da2_tvals[-1] - num_future_steps * da2_dt

        # Calculate how many steps to remove at beginning and end
        first_valid_idx = (da1_tvals >= required_min).argmax()
        n_removed_begin = first_valid_idx
        if da1_tvals[-1] > required_max:
            # Do cropping at the end
            last_valid_idx_plus_one = (
                da1_tvals > required_max
            ).argmax()  # To use for slice
            n_removed_end = len(da1_tvals) - last_valid_idx_plus_one
        else:
            # da1 ends before required_max
            last_valid_idx_plus_one = None  # slice without endpoint
            n_removed_end = 0

        if n_removed_begin > 0 or n_removed_end > 0:
            print(
                f"Warning: cropping da1 (e.g. 'state') to align with da2 "
                f"(e.g. 'boundary forcing'). Removed {n_removed_begin} steps "
                f"at start of data interval and {n_removed_end} at the end."
            )
            da1 = da1.isel(time=slice(first_valid_idx, last_valid_idx_plus_one))
        return da1


def inverse_softplus(x, beta=1, threshold=20):
    """
    Inverse of torch.nn.functional.softplus

    Input is clamped to approximately positive values of x, and the function is
    linear for inputs above x*beta for numerical stability.

    Input is clamped to x > ln(1+1e-6)/beta which is approximately positive
    values of x.
    Note that this torch.clamp_min will make gradients 0, but this is not a
    problem as values of x that are this close to 0 have gradients of 0 anyhow.
    """
    x_clamped = torch.clamp(
        x, min=torch.log(torch.tensor(1e-6 + 1)) / beta, max=threshold / beta
    )
    non_linear_part = torch.log(torch.expm1(x_clamped * beta)) / beta
    below_threshold = x * beta <= threshold
    x = torch.where(condition=below_threshold, input=non_linear_part, other=x)

    return x


def inverse_sigmoid(x):
    """
    Inverse of torch.sigmoid

    Sigmoid output takes values in [0,1], this makes sure input is just within
    this interval.
    Note that this torch.clamp will make gradients 0, but this is not a problem
    as values of x that are this close to 0 or 1 have gradients of 0 anyhow.
    """
    x_clamped = torch.clamp(x, min=1e-6, max=1 - 1e-6)
    return torch.log(x_clamped / (1 - x_clamped))


def init_training_logger_metrics(training_logger, val_steps):
    """
    Set up logger metrics to track
    """
    experiment = training_logger.experiment
    if isinstance(training_logger, WandbLogger):
        experiment.define_metric("val_mean_loss", summary="min")
        for step in val_steps:
            experiment.define_metric(f"val_loss_unroll{step}", summary="min")
    elif isinstance(training_logger, MLFlowLogger):
        pass
    else:
        warnings.warn(
            "Only WandbLogger & MLFlowLogger is supported for tracking metrics.\
             Experiment results will only go to stdout."
        )


@rank_zero_only
def setup_training_logger(datastore, args, run_name):
    """

    Parameters
    ----------
    datastore : Datastore
        Datastore object.

    args : argparse.Namespace
        Arguments from command line.

    run_name : str
        Name of the run.

    Returns
    -------
    logger : pytorch_lightning.loggers.base
        Logger object.
    """

    if args.logger == "wandb":
        logger = pl.loggers.WandbLogger(
            project=args.logger_project,
            name=run_name,
            config=dict(training=vars(args), datastore=datastore._config),
        )
    elif args.logger == "mlflow":
        url = os.getenv("MLFLOW_TRACKING_URI")
        if url is None:
            raise ValueError(
                "MLFlow logger requires setting MLFLOW_TRACKING_URI in env."
            )
        # suppress warnings about insecure requests so that we avoid warnings in
        # the logs when tracking on the MLflow tracking server
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        logger = CustomMLFlowLogger(
            experiment_name=args.logger_project,
            tracking_uri=url,
            run_name=run_name,
        )
        logger.log_hyperparams(
            dict(training=vars(args), datastore=datastore._config)
        )

    return logger
