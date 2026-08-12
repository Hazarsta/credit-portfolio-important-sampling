from pathlib import Path

import numpy as np
import pandas as pd

from config import AnalysisConfig
from data import load_portfolio
from experiment import run_analysis
from metrics import risk_metrics_plain
from simulation import (
    default_thresholds,
    likelihood_ratio,
    simulate_importance_sampling,
    simulate_plain,
)


def test_default_thresholds_are_correct() -> None:
    np.testing.assert_allclose(default_thresholds(np.array([0.5])), np.array([0.0]), atol=1e-15)


def test_mu_zero_matches_plain_with_same_legacy_stream() -> None:
    thresholds = default_thresholds(np.array([0.01, 0.05]))
    losses_if_default = np.array([4.0, 8.0])
    plain_rng = np.random.RandomState(17)
    is_rng = np.random.RandomState(17)
    plain = simulate_plain(2_000, thresholds, losses_if_default, 0.2, plain_rng)
    importance, weights = simulate_importance_sampling(
        2_000, thresholds, losses_if_default, 0.2, 0.0, is_rng
    )
    np.testing.assert_array_equal(plain, importance)
    np.testing.assert_array_equal(weights, np.ones(2_000))


def test_likelihood_ratio_has_mean_near_one() -> None:
    rng = np.random.RandomState(123)
    systemic = rng.normal(-0.8, 1.0, size=300_000)
    assert abs(likelihood_ratio(systemic, -0.8).mean() - 1.0) < 0.01


def test_small_simulation_is_finite_and_reproducible(valid_workbook: Path) -> None:
    config = AnalysisConfig(paths_per_replication=500)
    portfolio = load_portfolio(valid_workbook, "Sheet1", "test", config)
    first = simulate_plain(
        500, portfolio.thresholds, portfolio.loss_if_default, 0.2, np.random.RandomState(4)
    )
    second = simulate_plain(
        500, portfolio.thresholds, portfolio.loss_if_default, 0.2, np.random.RandomState(4)
    )
    np.testing.assert_array_equal(first, second)
    assert all(np.isfinite(value) and value >= 0 for value in risk_metrics_plain(first).values())


def test_quick_pipeline_writes_expected_files(valid_workbook: Path, tmp_path: Path) -> None:
    output = tmp_path / "outputs"
    config = AnalysisConfig(
        paths_per_replication=500,
        replications=3,
        selection_replications=3,
        mu_grid=(0.0, -0.5, -1.0),
    )
    first = run_analysis(valid_workbook, output, config)
    second = run_analysis(valid_workbook, tmp_path / "repeat", config, make_figures=False)
    pd.testing.assert_frame_equal(first["summary"], second["summary"])

    table_names = {
        "selected_mu.csv",
        "all_plain_mc_replications.csv",
        "all_importance_sampling_replications.csv",
        "all_portfolios_summary.csv",
    }
    for label in ("30_70", "50_50", "70_30"):
        table_names |= {
            f"{label}_mu_selection.csv",
            f"{label}_plain_mc_replications.csv",
            f"{label}_importance_sampling_replications.csv",
            f"{label}_summary.csv",
        }
        for suffix in ("mu_selection", "es99_scatter", "es99_boxplot", "es99_sd", "is_weights"):
            assert (output / "figures" / f"{label}_{suffix}.png").is_file()
    assert table_names <= {path.name for path in (output / "tables").glob("*.csv")}
