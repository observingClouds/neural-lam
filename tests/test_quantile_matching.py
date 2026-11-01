# Third-party
import torch

# First-party
from neural_lam import config as nlconfig
from neural_lam import metrics


def test_sorted_mse_ignores_spatial_permutation():
    target = torch.randn(2, 3, 100, 4)
    perm = torch.randperm(100)
    pred = target[..., perm, :]

    assert torch.allclose(
        metrics.sorted_mse(pred, target), torch.zeros(2, 3, 4), atol=1e-6
    )


def test_sorted_mse_bounded_by_mse():
    pred = torch.randn(2, 3, 100, 4)
    target = torch.randn(2, 3, 100, 4)

    qm = metrics.sorted_mse(pred, target)
    mse = metrics.mse(pred, target, None, sum_vars=False)

    assert qm.shape == (2, 3, 4)
    assert torch.all(qm <= mse + 1e-6)


def test_sorted_mse_penalizes_missing_tail():
    target = torch.randn(1, 1, 1000, 1)
    # Smoothed prediction: same mean, but shrunk towards it
    pred = 0.5 * target

    assert metrics.sorted_mse(pred, target).item() > 0.1


def test_sorted_mse_gradient():
    pred = torch.randn(1, 1, 50, 2, requires_grad=True)
    target = torch.randn(1, 1, 50, 2)

    metrics.sorted_mse(pred, target).sum().backward()

    assert pred.grad is not None
    assert torch.all(torch.isfinite(pred.grad))


def test_quantile_matching_config():
    config = nlconfig.NeuralLAMConfig.from_yaml(
        """
datastores:
  interior:
    kind: mdp
    config_path: ""
training:
  quantile_matching:
    weights:
      t_2m: 1.0
"""
    )
    assert config.training.quantile_matching.weights == {"t_2m": 1.0}

    default_config = nlconfig.NeuralLAMConfig.from_yaml(
        """
datastores:
  interior:
    kind: mdp
    config_path: ""
"""
    )
    assert default_config.training.quantile_matching.weights == {}
