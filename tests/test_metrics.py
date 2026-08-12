import numpy as np

from metrics import (
    RISK_METRICS,
    effective_sample_size,
    risk_metrics_plain,
    risk_metrics_weighted,
    weighted_quantile,
)


def test_weighted_quantile_uses_first_cdf_crossing() -> None:
    values = np.array([30.0, 10.0, 20.0])
    weights = np.array([0.2, 0.5, 0.3])
    assert weighted_quantile(values, weights, 0.5) == 10.0
    assert weighted_quantile(values, weights, 0.51) == 20.0


def test_ess_equal_and_unequal_weights() -> None:
    equal = np.ones(10)
    unequal = np.array([9.0] + [1.0] * 9)
    assert effective_sample_size(equal) == 10.0
    assert effective_sample_size(unequal) < 10.0


def test_risk_metric_functions_return_required_fields() -> None:
    losses = np.arange(100, dtype=float)
    plain = risk_metrics_plain(losses)
    weighted = risk_metrics_weighted(losses, np.ones(100))
    assert set(RISK_METRICS) <= plain.keys()
    assert set(RISK_METRICS) | {"ESS"} <= weighted.keys()
    assert all(np.isfinite(value) and value >= 0 for value in plain.values())
