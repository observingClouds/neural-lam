# Third-party
import numpy as np
import pytest
import xarray as xr

# First-party
from neural_lam import config as nlconfig
from neural_lam.datastore import init_datastore
from neural_lam.datastore.mdp import move_state_to_forcing


def _make_ds():
    rng = np.random.default_rng(0)
    state_features = ["u_10m", "t_2m", "tqc_dia"]
    forcing_features = ["toa_radiation", "hour_of_day_cos"]
    n_time, n_grid = 5, 7

    def feature_coords(category, features):
        return {
            f"{category}_feature": features,
            f"{category}_feature_units": (
                f"{category}_feature",
                [f"unit_{f}" for f in features],
            ),
            f"{category}_feature_long_name": (
                f"{category}_feature",
                [f"long_{f}" for f in features],
            ),
        }

    ds = xr.Dataset(
        {
            "state": (
                ("state_feature", "time", "grid_index"),
                rng.normal(size=(3, n_time, n_grid)).astype(np.float32),
            ),
            "forcing": (
                ("forcing_feature", "time", "grid_index"),
                rng.normal(size=(2, n_time, n_grid)),
            ),
        },
        coords={
            "time": np.arange(n_time),
            "grid_index": np.arange(n_grid),
            **feature_coords("state", state_features),
            **feature_coords("forcing", forcing_features),
        },
    )
    for category, features in [
        ("state", state_features),
        ("forcing", forcing_features),
    ]:
        for op in ["mean", "std", "diff_mean", "diff_std"]:
            ds[f"{category}__train__{op}"] = (
                f"{category}_feature",
                rng.uniform(1, 2, size=len(features)).astype(np.float32),
            )
    return ds


def test_move_state_to_forcing():
    ds = _make_ds()
    ds_new = move_state_to_forcing(ds, ["tqc_dia"])

    assert ds_new.state_feature.values.tolist() == ["u_10m", "t_2m"]
    assert ds_new.forcing_feature.values.tolist() == [
        "toa_radiation",
        "hour_of_day_cos",
        "tqc_dia",
    ]
    assert ds_new.forcing.dtype == ds.forcing.dtype
    assert ds_new.forcing_feature_units.values[-1] == "unit_tqc_dia"
    assert ds_new.forcing_feature_long_name.values[-1] == "long_tqc_dia"

    # data and statistics of the moved feature are carried over
    np.testing.assert_allclose(
        ds_new.forcing.sel(forcing_feature="tqc_dia").values,
        ds.state.sel(state_feature="tqc_dia").values,
    )
    for op in ["mean", "std", "diff_mean", "diff_std"]:
        assert (
            ds_new[f"forcing__train__{op}"].sel(forcing_feature="tqc_dia")
            == ds[f"state__train__{op}"].sel(state_feature="tqc_dia")
        )

    # remaining features are unchanged
    xr.testing.assert_identical(
        ds_new.state, ds.state.sel(state_feature=["u_10m", "t_2m"])
    )
    xr.testing.assert_identical(
        ds_new.forcing.sel(forcing_feature=["toa_radiation", "hour_of_day_cos"]),
        ds.forcing,
    )


def test_move_state_to_forcing_unknown_feature():
    with pytest.raises(ValueError, match="not state features"):
        move_state_to_forcing(_make_ds(), ["qv_2m"])


def test_state_to_forcing_config():
    config = nlconfig.NeuralLAMConfig.from_yaml(
        """
datastores:
  interior:
    kind: mdp
    config_path: ""
    state_to_forcing:
    - tqc_dia
"""
    )
    assert config.datastores["interior"].state_to_forcing == ["tqc_dia"]


def test_state_to_forcing_only_for_mdp():
    with pytest.raises(NotImplementedError, match="state_to_forcing"):
        init_datastore(
            datastore_kind="npyfilesmeps",
            config_path="unused.yaml",
            state_to_forcing=["t2m"],
        )
