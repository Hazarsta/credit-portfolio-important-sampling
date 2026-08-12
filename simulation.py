"""Vectorized Gaussian-copula loss simulation."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm


def default_thresholds(pd_values: np.ndarray) -> np.ndarray:
    """Map marginal default probabilities to standard-normal thresholds."""
    values = np.asarray(pd_values, dtype=float)
    if np.any((values <= 0) | (values >= 1)):
        raise ValueError("Default probabilities must satisfy 0 < PD < 1.")
    return norm.ppf(values)


def simulate_plain(
    paths: int,
    thresholds: np.ndarray,
    loss_if_default: np.ndarray,
    correlation: float,
    rng: Any = np.random,
) -> np.ndarray:
    """Draw systemic shocks first, then all idiosyncratic shocks, as in the reference."""
    systemic = rng.normal(0.0, 1.0, size=paths)
    idiosyncratic = rng.normal(0.0, 1.0, size=(paths, len(thresholds)))
    assets = (
        np.sqrt(correlation) * systemic[:, None]
        + np.sqrt(1.0 - correlation) * idiosyncratic
    )
    return (assets < thresholds) @ loss_if_default


def likelihood_ratio(systemic: np.ndarray, mu_is: float) -> np.ndarray:
    """Return p(Z)/q(Z) for q=N(mu_is, 1), evaluated stably in log form."""
    systemic = np.asarray(systemic, dtype=float)
    log_weights = norm.logpdf(systemic, loc=0.0, scale=1.0) - norm.logpdf(
        systemic, loc=mu_is, scale=1.0
    )
    return np.exp(log_weights)


def simulate_importance_sampling(
    paths: int,
    thresholds: np.ndarray,
    loss_if_default: np.ndarray,
    correlation: float,
    mu_is: float,
    rng: Any = np.random,
) -> tuple[np.ndarray, np.ndarray]:
    """Shift only the systemic factor and return losses with likelihood ratios."""
    systemic = rng.normal(mu_is, 1.0, size=paths)
    idiosyncratic = rng.normal(0.0, 1.0, size=(paths, len(thresholds)))
    assets = (
        np.sqrt(correlation) * systemic[:, None]
        + np.sqrt(1.0 - correlation) * idiosyncratic
    )
    losses = (assets < thresholds) @ loss_if_default
    return losses, likelihood_ratio(systemic, mu_is)
