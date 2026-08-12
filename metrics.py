"""Plain and self-normalized weighted risk metrics."""

from __future__ import annotations

import numpy as np

RISK_METRICS = ("EL", "VaR95", "ES95", "VaR99", "ES99")


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Select the smallest observed value whose normalized weighted CDF reaches q."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or weights.ndim != 1 or len(values) != len(weights) or not len(values):
        raise ValueError("Values and weights must be nonempty one-dimensional arrays of equal length.")
    if not 0 <= quantile <= 1:
        raise ValueError("Quantile must be between 0 and 1.")
    if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("Weights must be finite, nonnegative, and sum to a positive value.")
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    index = min(int(np.searchsorted(cumulative, quantile, side="left")), len(values) - 1)
    return float(values[order][index])


def effective_sample_size(weights: np.ndarray) -> float:
    """Calculate the Kish effective sample size."""
    weights = np.asarray(weights, dtype=float)
    return float(weights.sum() ** 2 / np.sum(weights**2))


def risk_metrics_plain(losses: np.ndarray) -> dict[str, float]:
    """Estimate EL, NumPy empirical VaR, and inclusive-tail ES."""
    losses = np.asarray(losses, dtype=float)
    result = {"EL": float(losses.mean())}
    for alpha in (0.95, 0.99):
        percentile = int(alpha * 100)
        var = float(np.quantile(losses, alpha))
        result[f"VaR{percentile}"] = var
        result[f"ES{percentile}"] = float(losses[losses >= var].mean())
    return result


def risk_metrics_weighted(losses: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Estimate self-normalized EL, weighted VaR/ES, and ESS."""
    losses = np.asarray(losses, dtype=float)
    weights = np.asarray(weights, dtype=float)
    result = {
        "EL": float(np.sum(weights * losses) / weights.sum()),
        "ESS": effective_sample_size(weights),
    }
    for alpha in (0.95, 0.99):
        percentile = int(alpha * 100)
        var = weighted_quantile(losses, weights, alpha)
        tail = losses >= var
        result[f"VaR{percentile}"] = var
        result[f"ES{percentile}"] = float(
            np.sum(weights[tail] * losses[tail]) / np.sum(weights[tail])
        )
    return result
