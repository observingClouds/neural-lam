import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr
import yaml

import functools
import os

from .config import Config
from ..base import BaseDatastore


def convert_stats_to_torch(stats):
    """Convert the normalization statistics to torch tensors.

    Args:
        stats (xr.Dataset): The normalization statistics.

    Returns:
        dict(tensor): The normalization statistics as torch tensors."""
    return {
        var: torch.tensor(stats[var].values, dtype=torch.float32)
        for var in stats.data_vars
    }

class MultiZarrDatastore(BaseDatastore):
    DIMS_TO_KEEP = {"time", "grid_index", "variable"}

    def __init__(self, config_path):
        with open(config_path, encoding="utf-8", mode="r") as file:
            self._config = yaml.safe_load(file)

    def open_zarrs(self, category):
        """Open the zarr dataset for the given category.

        Args:
            category (str): The category of the dataset (state/forcing/static).

            Returns:
                xr.Dataset: The xarray Dataset object."""
        zarr_configs = self._config[category]["zarrs"]

        datasets = []
        for config in zarr_configs:
            dataset_path = config["path"]
            try:
                dataset = xr.open_zarr(dataset_path, consolidated=True)
            except Exception as e:
                raise Exception("Error opening dataset:", dataset_path) from e
            datasets.append(dataset)
        merged_dataset = xr.merge(datasets)
        merged_dataset.attrs["category"] = category
        return merged_dataset

    @functools.cached_property
    def coords_projection(self):
        """Return the projection object for the coordinates.

        The projection object is used to plot the coordinates on a map.

        Returns:
            cartopy.crs.Projection: The projection object."""
        proj_config = self._config["projection"]
        proj_class_name = proj_config["class"]
        proj_class = getattr(ccrs, proj_class_name)
        proj_params = proj_config.get("kwargs", {})
        return proj_class(**proj_params)

    @functools.cached_property
    def step_length(self):
        """Return the step length of the dataset in hours.

        Returns:
            int: The step length in hours."""
        dataset = self.open_zarrs("state")
        time = dataset.time.isel(time=slice(0, 2)).values
        step_length_ns = time[1] - time[0]
        step_length_hours = step_length_ns / np.timedelta64(1, "h")
        return int(step_length_hours)

    @functools.lru_cache()
    def get_vars_names(self, category):
        """Return the names of the variables in the dataset.

        Args:
            category (str): The category of the dataset (state/forcing/static).

        Returns:
            list: The names of the variables in the dataset."""
        surface_vars_names = self._config[category].get("surface_vars") or []
        atmosphere_vars_names = [
            f"{var}_{level}"
            for var in (self._config[category].get("atmosphere_vars") or [])
            for level in (self._config[category].get("levels") or [])
        ]
        return surface_vars_names + atmosphere_vars_names

    @functools.lru_cache()
    def get_vars_units(self, category):
        """Return the units of the variables in the dataset.

        Args:
            category (str): The category of the dataset (state/forcing/static).

            Returns:
                list: The units of the variables in the dataset."""
        surface_vars_units = self._config[category].get("surface_units") or []
        atmosphere_vars_units = [
            unit
            for unit in (self._config[category].get("atmosphere_units") or [])
            for _ in (self._config[category].get("levels") or [])
        ]
        return surface_vars_units + atmosphere_vars_units

    @functools.lru_cache()
    def get_num_data_vars(self, category):
        """Return the number of data variables in the dataset.

        Args:
            category (str): The category of the dataset (state/forcing/static).

        Returns:
            int: The number of data variables in the dataset."""
        surface_vars = self._config[category].get("surface_vars", [])
        atmosphere_vars = self._config[category].get("atmosphere_vars", [])
        levels = self._config[category].get("levels", [])

        surface_vars_count = (
            len(surface_vars) if surface_vars is not None else 0
        )
        atmosphere_vars_count = (
            len(atmosphere_vars) if atmosphere_vars is not None else 0
        )
        levels_count = len(levels) if levels is not None else 0

        return surface_vars_count + atmosphere_vars_count * levels_count

    def _stack_grid(self, ds):
        """Stack the grid dimensions of the dataset.

        Args:
            ds (xr.Dataset): The xarray Dataset object.

        Returns:
            xr.Dataset: The xarray Dataset object with stacked grid dimensions."""
        if "grid_index" in ds.dims:
            raise ValueError("Grid dimensions already stacked.")
        else:
            if "x" not in ds.dims or "y" not in ds.dims:
                self._rename_dataset_dims_and_vars(dataset=ds)
            ds = ds.stack(grid_index=("y", "x")).reset_index("grid_index")
            # reset the grid_index coordinates to have integer values, otherwise
            # the serialisation to zarr will fail
            ds["grid_index"] = np.arange(len(ds["grid_index"]))
        return ds

    def _convert_dataset_to_dataarray(self, dataset):
        """Convert the Dataset to a Dataarray.

        Args:
            dataset (xr.Dataset): The xarray Dataset object.

        Returns:
            xr.DataArray: The xarray DataArray object."""
        if isinstance(dataset, xr.Dataset):
            dataset = dataset.to_array()
        return dataset

    def _filter_dimensions(self, dataset, transpose_array=True):
        """Drop the dimensions and filter the data_vars of the dataset.

        Args:
            dataset (xr.Dataset): The xarray Dataset object.
            transpose_array (bool): Whether to transpose the array.

        Returns:
            xr.Dataset: The xarray Dataset object with filtered dimensions.
            OR xr.DataArray: The xarray DataArray object with filtered dimensions."""
        dims_to_keep = self.DIMS_TO_KEEP
        dataset_dims = set(list(dataset.dims) + ["variable"])
        min_req_dims = dims_to_keep.copy()
        min_req_dims.discard("time")
        if not min_req_dims.issubset(dataset_dims):
            missing_dims = min_req_dims - dataset_dims
            print(
                f"\033[91mMissing required dimensions in dataset: "
                f"{missing_dims}\033[0m"
            )
            print(
                "\033[91mAttempting to update dims and "
                "vars based on zarr config...\033[0m"
            )
            dataset = self._rename_dataset_dims_and_vars(
                dataset.attrs["category"], dataset=dataset
            )
            dataset = self._stack_grid(dataset)
            dataset_dims = set(list(dataset.dims) + ["variable"])
            if min_req_dims.issubset(dataset_dims):
                print(
                    "\033[92mSuccessfully updated dims and "
                    "vars based on zarr config.\033[0m"
                )
            else:
                print(
                    "\033[91mFailed to update dims and "
                    "vars based on zarr config.\033[0m"
                )
                return None

        dataset_dims = set(list(dataset.dims) + ["variable"])
        dims_to_drop = dataset_dims - dims_to_keep
        dataset = dataset.drop_dims(dims_to_drop)
        if dims_to_drop:
            print(
                "\033[91mDropped dimensions: --",
                dims_to_drop,
                "-- from dataset.\033[0m",
            )
            print(
                "\033[91mAny data vars dependent "
                "on these variables were dropped!\033[0m"
            )

        if transpose_array:
            dataset = self._convert_dataset_to_dataarray(dataset)

            if "time" in dataset.dims:
                dataset = dataset.transpose("time", "grid_index", "variable")
            else:
                dataset = dataset.transpose("grid_index", "variable")
        dataset_vars = (
            list(dataset.data_vars)
            if isinstance(dataset, xr.Dataset)
            else dataset["variable"].values.tolist()
        )

        print(  # noqa
            f"\033[94mYour {dataset.attrs['category']} xr.Dataarray has the "
            f"following variables: {dataset_vars} \033[0m",
        )

        return dataset

    def _reshape_grid_to_2d(self, dataset, grid_shape=None):
        """Reshape the grid to 2D for stacked data without multi-index.

        Args:
            dataset (xr.Dataset): The xarray Dataset object.
            grid_shape (dict): The shape of the grid.

        Returns:
            xr.Dataset: The xarray Dataset object with reshaped grid dimensions."""
        if grid_shape is None:
            grid_shape = dict(self.grid_shape_state.values.items())
        x_dim, y_dim = (grid_shape["x"], grid_shape["y"])

        x_coords = np.arange(x_dim)
        y_coords = np.arange(y_dim)
        multi_index = pd.MultiIndex.from_product(
            [y_coords, x_coords], names=["y", "x"]
        )

        mindex_coords = xr.Coordinates.from_pandas_multiindex(
            multi_index, "grid"
        )
        dataset = dataset.drop_vars(["grid", "x", "y"], errors="ignore")
        dataset = dataset.assign_coords(mindex_coords)
        reshaped_data = dataset.unstack("grid")

        return reshaped_data

    @functools.lru_cache()
    def get_xy(self, category, stacked=True):
        """Return the x, y coordinates of the dataset.

        Args:
            category (str): The category of the dataset (state/forcing/static).
            stacked (bool): Whether to stack the x, y coordinates.

        Returns:
            np.ndarray: The x, y coordinates of the dataset (if stacked) (2, N_y, N_x)

            OR tuple(np.ndarray, np.ndarray): The x, y coordinates of the dataset
            (if not stacked) ((N_y, N_x), (N_y, N_x))"""
        dataset = self.open_zarrs(category)
        x, y = dataset.x.values, dataset.y.values
        if x.ndim == 1:
            x, y = np.meshgrid(x, y)
        if stacked:
            xy = np.stack((x, y), axis=0)  # (2, N_y, N_x)
            return xy
        return x, y

    def get_xy_extent(self, category):
        """Return the extent of the x, y coordinates. This should be a list
        of 4 floats with `[xmin, xmax, ymin, ymax]`

        Args:
            category (str): The category of the dataset (state/forcing/static).

        Returns:
            list(float): The extent of the x, y coordinates."""
        x, y = self.get_xy(category, stacked=False)
        if self.projection.inverted:
            extent = [x.max(), x.min(), y.max(), y.min()]
        else:
            extent = [x.min(), x.max(), y.min(), y.max()]

        return extent

    @functools.lru_cache()
    def get_normalization_stats(self, category):
        """Load the normalization statistics for the dataset.

        Args:
            category (str): The category of the dataset (state/forcing/static).

            Returns:
                OR xr.Dataset: The normalization statistics for the dataset.
        """
        combined_stats = self._load_and_merge_stats()
        if combined_stats is None:
            return None

        combined_stats = self._rename_data_vars(combined_stats)

        stats = self._select_stats_by_category(combined_stats, category)
        if stats is None:
            return None

        return stats

    def _load_and_merge_stats(self):
        """Load and merge the normalization statistics for the dataset.

        Returns:
            xr.Dataset: The merged normalization statistics for the dataset."""
        combined_stats = None
        for i, zarr_config in enumerate(
            self._config["utilities"]["normalization"]["zarrs"]
        ):
            stats_path = zarr_config["path"]
            if not os.path.exists(stats_path):
                raise FileNotFoundError(
                    f"Normalization statistics not found at path: {stats_path}"
                )
            stats = xr.open_zarr(stats_path, consolidated=True)
            if i == 0:
                combined_stats = stats
            else:
                combined_stats = xr.merge([stats, combined_stats])
        return combined_stats

    def _rename_data_vars(self, combined_stats):
        """Rename the data variables of the normalization statistics.

        Args:
            combined_stats (xr.Dataset): The combined normalization statistics.

        Returns:
            xr.Dataset: The combined normalization statistics with renamed data
            variables."""
        vars_mapping = {}
        for zarr_config in self._config["utilities"]["normalization"]["zarrs"]:
            vars_mapping.update(zarr_config["stats_vars"])

        return combined_stats.rename_vars(
            {
                v: k
                for k, v in vars_mapping.items()
                if v in list(combined_stats.data_vars)
            }
        )

    def _select_stats_by_category(self, combined_stats, category):
        """Select the normalization statistics for the given category.

        Args:
            combined_stats (xr.Dataset): The combined normalization statistics.
            category (str): The category of the dataset (state/forcing/static).

        Returns:
            xr.Dataset: The normalization statistics for the dataset."""
        if category == "state":
            stats = combined_stats.loc[dict(variable=self.get_vars_names(category=category))]
            stats = stats.drop_vars(["forcing_mean", "forcing_std"])
            return stats
        elif category == "forcing":
            non_normalized_vars = (
                self.utilities.normalization.non_normalized_vars
            )
            if non_normalized_vars is None:
                non_normalized_vars = []
            vars = self.vars_names(category)
            window = self["forcing"]["window"]
            forcing_vars = [f"{var}_{i}" for var in vars for i in range(window)]
            normalized_vars = [
                var for var in forcing_vars if var not in non_normalized_vars
            ]
            non_normalized_vars = [
                var for var in forcing_vars if var in non_normalized_vars
            ]
            stats_normalized = combined_stats.loc[
                dict(forcing_variable=normalized_vars)
            ]
            if non_normalized_vars:
                stats_non_normalized = combined_stats.loc[
                    dict(forcing_variable=non_normalized_vars)
                ]
                stats = xr.merge([stats_normalized, stats_non_normalized])
            else:
                stats = stats_normalized
            stats_normalized = stats_normalized[["forcing_mean", "forcing_std"]]

            return stats
        else:
            print(f"Invalid category: {category}")
            return None

    def _extract_vars(self, category, ds=None):
        """Extract (select) the data variables from the dataset.

        Args:
            category (str): The category of the dataset (state/forcing/static).
            dataset (xr.Dataset): The xarray Dataset object.

        Returns:
            xr.Dataset: The xarray Dataset object with extracted variables.
        """
        if ds is None:
            ds = self.open_zarrs(category)
        surface_vars = self._config[category].get("surface_vars")
        atmoshere_vars = self._config[category].get("atmosphere_vars")
        
        ds_surface = None
        if surface_vars is not None:
            ds_surface = ds[surface_vars]

        ds_atmosphere = None
        if atmoshere_vars is not None:
            ds_atmosphere = self._extract_atmosphere_vars(category=category, ds=ds)

        if ds_surface and ds_atmosphere:
            return xr.merge([ds_surface, ds_atmosphere])
        elif ds_surface:
            return ds_surface
        elif ds_atmosphere:
            return ds_atmosphere
        else:
            raise ValueError(f"No variables found in dataset {category}")

    def _extract_atmosphere_vars(self, category, ds):
        """Extract the atmosphere variables from the dataset.

        Args:
            category (str): The category of the dataset (state/forcing/static).
            ds (xr.Dataset): The xarray Dataset object.

        Returns:
            xr.Dataset: The xarray Dataset object with atmosphere variables."""

        if "level" not in list(ds.dims) and self._config[category]["atmosphere_vars"]:
            ds = self._rename_dataset_dims_and_vars(
                ds.attrs["category"], dataset=ds
            )

        data_arrays = [
            ds[var].sel(level=level, drop=True).rename(f"{var}_{level}")
            for var in self._config[category]["atmosphere_vars"]
            for level in self._config[category]["levels"]
        ]

        if self._config[category]["atmosphere_vars"]:
            return xr.merge(data_arrays)
        else:
            return xr.Dataset()

    def _rename_dataset_dims_and_vars(self, category, dataset=None):
        """Rename the dimensions and variables of the dataset.

        Args:
            category (str): The category of the dataset (state/forcing/static).
            dataset (xr.Dataset): The xarray Dataset object. OR xr.DataArray:
            The xarray DataArray object.

        Returns:
            xr.Dataset: The xarray Dataset object with renamed dimensions and
            variables.
            OR xr.DataArray: The xarray DataArray object with renamed
            dimensions and variables."""
        convert = False
        if dataset is None:
            dataset = self.open_zarrs(category)
        elif isinstance(dataset, xr.DataArray):
            convert = True
            dataset = dataset.to_dataset("variable")
        dims_mapping = {}
        zarr_configs = self._config[category]["zarrs"]
        for zarr_config in zarr_configs:
            dims_mapping.update(zarr_config["dims"])

        dataset = dataset.rename_dims(
            {
                v: k
                for k, v in dims_mapping.items()
                if k not in dataset.dims and v in dataset.dims
            }
        )
        dataset = dataset.rename_vars(
            {v: k for k, v in dims_mapping.items() if v in dataset.coords}
        )
        if convert:
            dataset = dataset.to_array()
        return dataset

    def _apply_time_split(self, dataset, split="train"):
        """Filter the dataset by the time split.

        Args:
            dataset (xr.Dataset): The xarray Dataset object.
            split (str): The time split to filter the dataset.

        Returns:["window"]
            xr.Dataset: The xarray Dataset object filtered by the time split."""
        start, end = (
            self._config["splits"][split]["start"],
            self._config["splits"][split]["end"],
        )
        dataset = dataset.sel(time=slice(start, end))
        dataset.attrs["split"] = split
        return dataset

    def apply_window(self, category, dataset=None):
        """Apply the forcing window to the forcing dataset.

        Args:
            category (str): The category of the dataset (state/forcing/static).
            dataset (xr.Dataset): The xarray Dataset object.

        Returns:
            xr.Dataset: The xarray Dataset object with the window applied."""
        if dataset is None:
            dataset = self.open_zarrs(category)
        if isinstance(dataset, xr.Dataset):
            dataset = self._convert_dataset_to_dataarray(dataset)
        state = self.open_zarrs("state")
        state = self._apply_time_split(state, dataset.attrs["split"])
        state_time = state.time.values
        window = self._config[category]["window"]
        dataset = (
            dataset.sel(time=state_time, method="nearest")
            .pad(time=(window // 2, window // 2), mode="edge")
            .rolling(time=window, center=True)
            .construct("window")
            .stack(variable_window=("variable", "window"))
        )
        dataset = dataset.isel(time=slice(window // 2, -window // 2 + 1))
        return dataset

    @property
    def boundary_mask(self):
        """
        Load the boundary mask for the dataset, with spatial dimensions stacked.

        Returns
        -------
        xr.DataArray
            The boundary mask for the dataset, with dimensions `('grid_index',)`.
        """
        ds_boundary_mask = xr.open_zarr(self._config["boundary"]["mask"]["path"])
        return ds_boundary_mask.mask.stack(grid_index=("y", "x")).reset_index("grid_index")


    def get_dataarray(self, category, split="train", apply_windowing=True):
        """Process the dataset for the given category.

        Args:
            category (str): The category of the dataset (state/forcing/static).
            split (str): The time split to filter the dataset (train/val/test).
            apply_windowing (bool): Whether to apply windowing to the forcing dataset.

        Returns:
            xr.DataArray: The xarray DataArray object with processed dataset."""
        dataset = self.open_zarrs(category)
        dataset = self._extract_vars(category, dataset)
        if category != "static":
            dataset = self._apply_time_split(dataset, split)
        dataset = self._stack_grid(dataset)
        dataset = self._rename_dataset_dims_and_vars(category, dataset)
        dataset = self._filter_dimensions(dataset)
        dataset = self._convert_dataset_to_dataarray(dataset)
        if "window" in self._config[category] and apply_windowing:
            dataset = self.apply_window(category, dataset)
        if category == "static" and "time" in dataset.dims:
            dataset = dataset.isel(time=0, drop=True)

        return dataset
