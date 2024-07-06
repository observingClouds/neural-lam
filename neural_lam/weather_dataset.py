import dataclasses
import warnings

# Third-party
import pytorch_lightning as pl
import torch
import xarray as xr
import numpy as np

# First-party
from neural_lam.datastore.multizarr import config
from neural_lam.datastore.base import BaseDatastore


@dataclasses.dataclass
class TrainingSample:
    """
    A dataclass to hold a single training sample of `ar_steps` autoregressive steps,
    which consists of the initial states, target states, forcing and batch times. The
    inititial and target states should have `d_features` features, and the forcing should
    have `d_windowed_forcing` features.

    Parameters
    ----------
    init_states : torch.Tensor
        The initial states of the training sample, shape (2, N_grid, d_features).
    target_states : torch.Tensor
        The target states of the training sample, shape (ar_steps, N_grid, d_features).
    forcing : torch.Tensor
        The forcing of the training sample, shape (ar_steps, N_grid, d_windowed_forcing).
    batch_times : np.ndarray
        The times of the batch, shape (ar_steps,).
    """
    init_states: torch.Tensor
    target_states: torch.Tensor
    forcing: torch.Tensor
    batch_times: np.ndarray
    
    def __post_init__(self):
        """
        Validate the shapes of the tensors match between the different components of the training sample.
        
        # init_states: (2, N_grid, d_features)
        # target_states: (ar_steps, N_grid, d_features)
        # forcing: (ar_steps, N_grid, d_windowed_forcing)
        # batch_times: (ar_steps,)
        """
        assert self.init_states.shape[0] == 2
        _, N_grid, d_features = self.init_states.shape
        N_pred_steps = self.target_states.shape[0]

        # check number of grid points
        if not (self.target_states.shape[1] == self.target_states.shape[1] == N_grid):
            raise Exception(f"Number of grid points do not match, got {self.target_states.shape[1]=} and {self.target_states.shape[2]=}, expected {N_grid=}")

        # check number of features for init and target states
        assert self.target_states.shape[2] == d_features
        
        # check that target, forcing and batch times have the same number of prediction steps
        if not (self.target_states.shape[0] == self.forcing.shape[0] == self.batch_times.shape[0] == N_pred_steps):
            raise Exception(f"Number of prediction steps do not match, got {self.target_states.shape[0]=}, {self.forcing.shape[0]=} and {self.batch_times.shape[0]=}, expected {N_pred_steps=}")


class WeatherDataset(torch.utils.data.Dataset):
    """
    Dataset class for weather data.

    This class loads and processes weather data from a given datastore.
    """

    def __init__(
        self,
        datastore: BaseDatastore,
        split="train",
        ar_steps=3,
        forcing_window_size=3,
        batch_size=4,
        standardize=True,
    ):
        super().__init__()

        self.split = split
        self.batch_size = batch_size
        self.ar_steps = ar_steps
        self.datastore = datastore

        self.da_state = self.datastore.get_dataarray(category="state", split=self.split)
        self.da_forcing = self.datastore.get_dataarray(category="forcing", split=self.split)
        self.forcing_window_size = forcing_window_size

        # Set up for standardization
        # TODO: This will become part of ar_model.py soon!
        self.standardize = standardize
        if standardize:
            self.ds_state_stats = self.datastore.get_normalization_dataarray(category="state")

            self.da_state_mean = self.ds_state_stats.state_mean
            self.da_state_std = self.ds_state_stats.state_std

            if self.da_forcing is not None:
                self.ds_forcing_stats = self.datastore.get_normalization_dataarray(category="forcing")
                self.da_forcing_mean = self.ds_forcing_stats.forcing_mean
                self.da_forcing_std = self.ds_forcing_stats.forcing_std

    def __len__(self):
        if self.datastore.is_forecast:
            # for now we simply create a single sample for each analysis time
            # and then the next ar_steps forecast times
            if self.datastore.is_ensemble:
                warnings.warn(
                    "only using first ensemble member, so dataset size is effectively"
                    f" reduced by the number of ensemble members ({self.da_state.ensemble_member.size})", UserWarning
                )
                return self.da_state.analysis_time.size * self.da_state.ensemble_member.size
            return self.da_state.analysis_time.size
        else:
            # Skip first and last time step
            return len(self.da_state.time) - self.ar_steps
            
    def _sample_time(self, da, idx, n_steps:int, n_timesteps_offset:int=0):
        """
        Produce a time slice of the given dataarray `da` (state or forcing) starting at `idx` and
        with `n_steps` steps. The `n_timesteps_offset` parameter is used to offset the start of the
        sample, for example to exclude the first two steps when sampling the forcing data (and to 
        produce the windowing samples of forcing data by increasing the offset for each window).
        
        Parameters
        ----------
        da : xr.DataArray
            The dataarray to sample from. This is expected to have a `time` dimension if the datastore
            is providing analysis only data, and a `analysis_time` and `elapsed_forecast_time` dimensions
            if the datastore is providing forecast data.
        idx : int
            The index of the time step to start the sample from.
        n_steps : int
            The number of time steps to include in the sample.
        
        """
        # selecting the time slice
        if self.datastore.is_forecast:
            # this implies that the data will have both `analysis_time` and `elapsed_forecast_time` dimensions
            # for forecasts we for now simply select a analysis time and then
            # the next ar_steps forecast times
            da = da.isel(analysis_time=idx, elapsed_forecast_time=slice(n_timesteps_offset, n_steps + n_timesteps_offset))
            # create a new time dimension so that the produced sample has a `time` dimension, similarly 
            # to the analysis only data
            da["time"] = da.analysis_time + da.elapsed_forecast_time
            da = da.swap_dims({"elapsed_forecast_time": "time"})
        else:
            # only `time` dimension for analysis only data
            da = da.isel(time=slice(idx + n_timesteps_offset, idx + n_steps + n_timesteps_offset))
        return da

    def __getitem__(self, idx):
        """
        Return a single training sample, which consists of the initial states,
        target states, forcing and batch times. 
        
        The implementation currently uses xarray.DataArray objects for the normalisation
        so that we can make us of xarray's broadcasting capabilities. This makes it possible
        to normalise with both global means, but also for example where a grid-point mean
        has been computed. This code will have to be replace if normalisation is to be done
        on the GPU to handle different shapes of the normalisation.
        
        Parameters
        ----------
        idx : int
            The index of the sample to return, this will refer to the time of the initial state.

        Returns
        -------
        init_states : TrainingSample
            A training sample object containing the initial states, target states, forcing and batch times.
            The batch times are the times of the target steps.
        """
        # handling ensemble data
        if self.datastore.is_ensemble:
            # for the now the strategy is to simply select a random ensemble member
            # XXX: this could be changed to include all ensemble members by splitting `idx` into
            # two parts, one for the analysis time and one for the ensemble member and then increasing
            # self.__len__ to include all ensemble members
            i_ensemble = np.random.randint(self.da_state.ensemble_member.size)
            da_state = self.da_state.isel(ensemble_member=i_ensemble)
        else:
            da_state = self.da_state
            
        if self.da_forcing is not None:
            if "ensemble_member" in self.da_forcing.dims:
                raise NotImplementedError("Ensemble member not yet supported for forcing data")
            da_forcing = self.da_forcing
        else:
            da_forcing = xr.DataArray()
            
        # handle time sampling in a way that is compatible with both analysis and forecast data
        da_state = self._sample_time(da=da_state, idx=idx, n_steps=2+self.ar_steps)

        das_forcing = []
        for n in range(self.forcing_window_size):
            da_ = self._sample_time(da=da_forcing, idx=idx, n_steps=self.ar_steps, n_timesteps_offset=2+n)
            if n > 0:
                da_ = da_.drop_vars("time")
            das_forcing.append(da_)
        da_forcing_windowed = xr.concat(das_forcing, dim="window_sample")
            
        # ensure the dimensions are in the correct order
        da_state = da_state.transpose("time", "grid_index", "state_feature")
        da_forcing_windowed = da_forcing_windowed.transpose("time", "grid_index", "forcing_feature", "window_sample")

        da_init_states = da_state.isel(time=slice(None, 2))
        da_target_states = da_state.isel(time=slice(2, None))
        
        batch_times = da_forcing_windowed.time

        if self.standardize:
            da_init_states = (da_init_states - self.da_state_mean) / self.da_state_std
            da_target_states = (da_target_states - self.da_state_mean) / self.da_state_std

            if self.da_forcing is not None:
                da_forcing_windowed = (da_forcing_windowed - self.da_forcing_mean) / self.da_forcing_std
                
        # stack the `forcing_feature` and `window_sample` dimensions into a single `forcing_feature` dimension
        da_forcing_windowed = da_forcing_windowed.stack(forcing_feature_windowed=("forcing_feature", "window_sample"))
                
        init_states = torch.tensor(da_init_states.values, dtype=torch.float32)
        target_states = torch.tensor(da_target_states.values, dtype=torch.float32)
        forcing = torch.tensor(da_forcing_windowed.values, dtype=torch.float32)

        # init_states: (2, N_grid, d_features)
        # target_states: (ar_steps, N_grid, d_features)
        # forcing: (ar_steps, N_grid, d_windowed_forcing)
        # batch_times: (ar_steps,)

        return TrainingSample(
            init_states=init_states,
            target_states=target_states,
            forcing=forcing,
            batch_times=batch_times,
        )


class WeatherDataModule(pl.LightningDataModule):
    """DataModule for weather data."""

    def __init__(
        self,
        ar_steps_train=3,
        ar_steps_eval=25,
        standardize=True,
        batch_size=4,
        num_workers=16,
    ):
        super().__init__()
        self.ar_steps_train = ar_steps_train
        self.ar_steps_eval = ar_steps_eval
        self.standardize = standardize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = WeatherDataset(
                split="train",
                ar_steps=self.ar_steps_train,
                standardize=self.standardize,
                batch_size=self.batch_size,
            )
            self.val_dataset = WeatherDataset(
                split="val",
                ar_steps=self.ar_steps_eval,
                standardize=self.standardize,
                batch_size=self.batch_size,
            )

        if stage == "test" or stage is None:
            self.test_dataset = WeatherDataset(
                split="test",
                ar_steps=self.ar_steps_eval,
                standardize=self.standardize,
                batch_size=self.batch_size,
            )

    def train_dataloader(self):
        """Load train dataset."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def val_dataloader(self):
        """Load validation dataset."""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        """Load test dataset."""
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
