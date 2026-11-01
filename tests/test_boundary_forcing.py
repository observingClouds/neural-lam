"""Tests for boundary-forced graph models and the related data handling."""

# Standard library
from pathlib import Path

# Third-party
import numpy as np
import pytest
import torch

# First-party
from neural_lam import metrics
from neural_lam.build_rectangular_graph import build_graph_from_archetype
from neural_lam.config import load_config_and_datastore
from neural_lam.models import DeterministicARForecaster, GraphLAM, HiLAM
from neural_lam.weather_dataset import WeatherDataModule, WeatherDataset

CONFIG_PATH = (
    Path(__file__).parent
    / "datastore_examples"
    / "mdp"
    / "era5_1000hPa_danra_100m_winds"
    / "config.yaml"
)


@pytest.fixture(scope="module")
def datastores():
    """Interior and boundary example datastores."""
    _, datastore, datastore_boundary = load_config_and_datastore(
        str(CONFIG_PATH)
    )
    return datastore, datastore_boundary


@pytest.fixture(scope="module")
def graph_dir(tmp_path_factory, datastores):
    """Directory with graphs connecting interior and boundary nodes."""
    datastore, datastore_boundary = datastores
    graph_dir = tmp_path_factory.mktemp("graphs")
    build_graph_from_archetype(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        graph_name="keisler",
        archetype="keisler",
        dir_save_path=graph_dir,
        mesh_node_distance=50000,
    )
    build_graph_from_archetype(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        graph_name="keisler_fine",
        archetype="keisler",
        dir_save_path=graph_dir,
        mesh_node_distance=40000,
    )
    build_graph_from_archetype(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        graph_name="hierarchical",
        archetype="hierarchical",
        dir_save_path=graph_dir,
        mesh_node_distance=30000,
        max_num_levels=2,
    )
    return graph_dir


@pytest.mark.requires_real_data
def test_interior_has_no_boundary_mask_with_boundary_datastore(datastores):
    datastore, _ = datastores
    assert int(datastore.boundary_mask.sum()) == 0


@pytest.mark.requires_real_data
@pytest.mark.parametrize("dynamic", [False, True])
def test_boundary_time_delta_features(datastores, dynamic):
    datastore, datastore_boundary = datastores
    kwargs = dict(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        split="train",
        ar_steps=3,
    )
    plain = WeatherDataset(**kwargs)
    with_deltas = WeatherDataset(
        **kwargs, add_boundary_time_deltas=True, dynamic_time_deltas=dynamic
    )
    boundary = plain[0][3]
    boundary_deltas = with_deltas[0][3]
    window = 3
    assert boundary_deltas.shape[-1] == boundary.shape[-1] + window
    torch.testing.assert_close(boundary_deltas[..., :-window], boundary)

    time_deltas = boundary_deltas[..., -window:]
    # Same deltas for all boundary nodes
    assert torch.all(time_deltas == time_deltas[:, :1])
    # Boundary steps (6h) are two interior steps (3h)
    ratio = datastore_boundary.step_length / datastore.step_length
    if not dynamic:
        expected = torch.tensor([-ratio, 0.0, ratio])
        torch.testing.assert_close(
            time_deltas[:, 0], expected.expand(time_deltas.shape[0], -1)
        )
    else:
        # Relative to the state time the window is shifted by whole
        # interior steps
        np.testing.assert_allclose(
            np.diff(time_deltas[:, 0].numpy(), axis=-1), ratio
        )


@pytest.mark.requires_real_data
def test_eval_init_times_and_excluded_intervals(datastores):
    datastore, datastore_boundary = datastores
    data_module = WeatherDataModule(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        ar_steps_train=2,
        ar_steps_eval=2,
        num_workers=0,
        eval_init_times=[0, 12],
        excluded_intervals=[["2022-04-01T00", "2022-04-02T00"]],
    )
    data_module.setup()
    full_train = WeatherDataset(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        split="train",
        ar_steps=2,
    )
    assert 0 < len(data_module.train_dataset) < len(full_train)
    exc_start = np.datetime64("2022-04-01T00")
    exc_end = np.datetime64("2022-04-02T00")
    for sample in data_module.train_dataset:
        times = sample[4].numpy().astype("datetime64[ns]")
        assert not ((times >= exc_start) & (times <= exc_end)).any()

    for sample in data_module.val_dataset:
        init_time = sample[4].numpy().astype("datetime64[ns]")[0] - (
            np.timedelta64(datastore.step_length)
        )
        assert init_time.astype("datetime64[h]").astype(int) % 24 in (0, 12)


@pytest.mark.requires_real_data
@pytest.mark.parametrize(
    "model_class,graph_name", [(GraphLAM, "keisler"), (HiLAM, "hierarchical")]
)
def test_boundary_forced_predictor(
    datastores, graph_dir, model_class, graph_name
):
    datastore, datastore_boundary = datastores
    predictor = model_class(
        datastore=datastore,
        graph_name=graph_name,
        graph_dir=graph_dir,
        hidden_dim=8,
        hidden_dim_grid=4,
        processor_layers=1,
        datastore_boundary=datastore_boundary,
        time_delta_enc_dim=4,
        num_quantiles=3,
        gated_features=["r2m"],
        gate_thresholds={"r2m": 0.5},
    )
    assert predictor.boundary_forced
    dataset = WeatherDataset(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        split="train",
        ar_steps=2,
        add_boundary_time_deltas=True,
    )
    init_states, target_states, forcing, boundary, _ = (
        t.unsqueeze(0) for t in dataset[0]
    )
    forecaster = DeterministicARForecaster(
        predictor, datastore, loss="crps_quantile", grad_checkpointing=True
    )
    prediction, pred_quantiles = forecaster(
        init_states, forcing, target_states, boundary_forcing=boundary
    )
    assert prediction.shape == target_states.shape
    assert pred_quantiles.shape == (*target_states.shape, 3)
    gate_logits = predictor.rollout_aux["gate_logits"]
    assert gate_logits.shape == (*target_states.shape[:-1], 1)

    aux_loss, components = forecaster.compute_auxiliary_losses(
        prediction, target_states
    )
    assert aux_loss.shape == target_states.shape[:2]
    assert "gate_bce" in components
    loss = forecaster.compute_loss_from_forecast(
        prediction, target_states, pred_quantiles
    )
    (loss.mean() + aux_loss.mean()).backward()
    assert predictor.boundary_embedder[0].weight.grad is not None
    assert predictor.gate_map[0].weight.grad is not None


@pytest.mark.requires_real_data
def test_set_domain_switches_graph(datastores, graph_dir):
    datastore, datastore_boundary = datastores
    predictor = GraphLAM(
        datastore=datastore,
        graph_name="keisler",
        graph_dir=graph_dir,
        hidden_dim=8,
        processor_layers=1,
        datastore_boundary=datastore_boundary,
        additional_domains=[(datastore, datastore_boundary, "keisler_fine")],
    )
    dataset = WeatherDataset(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        split="train",
        ar_steps=1,
        add_boundary_time_deltas=True,
    )
    init_states, _, forcing, boundary, _ = (t.unsqueeze(0) for t in dataset[0])

    def predict():
        return predictor(
            init_states[:, 1], init_states[:, 0], forcing[:, 0], boundary[:, 0]
        )[0]

    g2m_edges_0 = predictor.g2m_gnn.edge_index.clone()
    num_mesh_0 = predictor.mesh_static_features.shape[0]
    pred_0 = predict()
    predictor.set_domain(1)
    assert predictor.mesh_static_features.shape[0] > num_mesh_0
    assert not torch.equal(predictor.g2m_gnn.edge_index, g2m_edges_0)
    pred_1 = predict()
    assert pred_1.shape == pred_0.shape
    assert not torch.allclose(pred_0, pred_1)
    predictor.set_domain(0)
    assert torch.equal(predictor.g2m_gnn.edge_index, g2m_edges_0)
    torch.testing.assert_close(predict(), pred_0)


def test_crps_quantile_metric():
    target = torch.zeros(2, 5, 3)
    quantiles = torch.stack(
        [torch.full_like(target, v) for v in (-1.0, 0.0, 1.0)], dim=-1
    )
    crps = metrics.crps_quantile(None, target, quantiles, sum_vars=False)
    assert crps.shape == (2, 3)
    # Pinball losses 0.25, 0, 0.25 averaged and doubled
    torch.testing.assert_close(crps, torch.full((2, 3), 1 / 3))
    assert metrics.requires_pred_std(metrics.crps_quantile)
