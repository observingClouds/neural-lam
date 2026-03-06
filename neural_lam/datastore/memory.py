"""Lightweight in-memory datastore used for dynamic boundary forcing."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Union

import numpy as np
import xarray as xr

from .base import BaseDatastore


class ArrayDatastore(BaseDatastore):
    """Minimal `BaseDatastore` implementation wrapping a single DataArray.

    This class is intended for use during evaluation when boundary forcing is
    computed on the fly and therefore does *not* live on disk.  Only the
    ``forcing`` category is supported; all other categories return ``None``.

    Parameters
    ----------
    arr : xr.DataArray
        DataArray containing the boundary forcing.  The array is assumed to
        already have the correct dimensions (``time`` or ``analysis_time`` etc.)
        and the ``grid_index`` dimension.  ``arr`` is copied on construction
        so that subsequent modifications by the caller will not affect the
        datastore.
    reference : BaseDatastore
        A reference datastore from which metadata (``coords_projection``,
        ``get_standardization_dataarray`` etc.) will be delegated.  This makes
        it easy to reuse the same statistics and coordinate information that
        the original boundary datastore provided.
    """

    def __init__(self, arr: xr.DataArray, reference: BaseDatastore):
        self._ds = arr.copy()
        self._reference = reference

    @property
    def root_path(self) -> Path:
        # not meaningful for an in-memory object; delegate to reference
        return Path(self._reference.root_path)

    @property
    def num_grid_points(self) -> int:
        """Number of spatial grid points contained in ``self._ds``."""
        # assume a dimension called 'grid_index' is present
        if "grid_index" in self._ds.dims:
            return int(self._ds.sizes["grid_index"])
        # fall back to reference datastore if unknown
        return self._reference.num_grid_points

    @property
    def config(self) -> Any:
        return self._reference.config

    @property
    def step_length(self) -> int:
        # boundary forcing is always treated as "forcing" in the reference
        return self._reference.step_length

    def get_vars_units(self, category: str) -> List[str]:
        # delegate to reference for anything we don't store explicitly
        if category in ("forcing", "static"):
            return self._reference.get_vars_units(category)
        return []

    def get_vars_names(self, category: str) -> List[str]:
        if category in ("forcing", "static"):
            return self._reference.get_vars_names(category)
        return []

    def get_vars_long_names(self, category: str) -> List[str]:
        if category in ("forcing", "static"):
            return self._reference.get_vars_long_names(category)
        return []

    def get_num_data_vars(self, category: str) -> int:
        if category == "forcing":
            return int(self._ds.sizes.get("forcing_feature", 0))
        # ask reference for static
        if category == "static":
            return self._reference.get_num_data_vars(category)
        return 0

    def get_standardization_dataarray(self, category: str) -> xr.Dataset:
        # delegate entirely to the reference store so we use the same
        # statistics
        return self._reference.get_standardization_dataarray(category)

    def get_dataarray(
        self, category: str, split: str, standardize: bool = False
    ) -> Union[xr.DataArray, None]:
        if category == "forcing" or category == "state":
            da = self._ds
            if standardize:
                stats = self.get_standardization_dataarray(category=category)
                mean = stats.forcing_mean
                std = stats.forcing_std
                da = (da - mean) / std
            return da
        elif category == "static":
            # just proxy the reference static array (should be small)
            return self._reference.get_dataarray(category, split, standardize)
        else:
            raise ValueError(f"Unknown category: {category}")
        return None

    def get_xy(self, category: str) -> Any:
        # boundary forcing does not expose a grid; delegate to reference
        return self._reference.get_xy(category)

    @property
    def coords_projection(self):
        return self._reference.coords_projection
