"""Replication studies, shift selection, and output orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import AnalysisConfig
from data import Portfolio, load_portfolios
from metrics import RISK_METRICS, risk_metrics_plain, risk_metrics_weighted
from plotting import plot_mu_selection, plot_replication_comparison
from simulation import simulate_importance_sampling, simulate_plain


def _plain_replications(
    portfolio: Portfolio, config: AnalysisConfig, count: int, rng: Any
) -> pd.DataFrame:
    rows = []
    for replication in range(1, count + 1):
        losses = simulate_plain(
            config.paths_per_replication,
            portfolio.thresholds,
            portfolio.loss_if_default,
            config.asset_correlation,
            rng,
        )
        rows.append(
            {"Portfolio": portfolio.label, "Replication": replication, **risk_metrics_plain(losses)}
        )
    return pd.DataFrame(rows)


def _importance_replications(
    portfolio: Portfolio,
    config: AnalysisConfig,
    count: int,
    mu_is: float,
    rng: Any,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    last_weights = np.empty(0)
    for replication in range(1, count + 1):
        losses, weights = simulate_importance_sampling(
            config.paths_per_replication,
            portfolio.thresholds,
            portfolio.loss_if_default,
            config.asset_correlation,
            mu_is,
            rng,
        )
        rows.append(
            {
                "Portfolio": portfolio.label,
                "Replication": replication,
                "mu_IS": mu_is,
                **risk_metrics_weighted(losses, weights),
            }
        )
        last_weights = weights
    return pd.DataFrame(rows), last_weights


def select_shift(
    portfolio: Portfolio, config: AnalysisConfig, rng: Any = np.random
) -> tuple[pd.DataFrame, float]:
    """Select the nonzero grid shift with the largest empirical ES99 variance reduction."""
    plain = _plain_replications(portfolio, config, config.selection_replications, rng)
    plain_sd = float(plain["ES99"].std(ddof=1))
    rows = []
    for mu_is in config.mu_grid:
        importance, _ = _importance_replications(
            portfolio, config, config.selection_replications, mu_is, rng
        )
        es99_sd = float(importance["ES99"].std(ddof=1))
        rows.append(
            {
                "Portfolio": portfolio.label,
                "mu_IS": mu_is,
                "ES99_SD": es99_sd,
                "ES99_SE": es99_sd / np.sqrt(config.selection_replications),
                "Variance_Reduction_Ratio": plain_sd**2 / es99_sd**2,
                "Average_ESS": float(importance["ESS"].mean()),
                "ESS_Ratio": float(importance["ESS"].mean()) / config.paths_per_replication,
                "Plain_ES99_SD": plain_sd,
                "Plain_ES99_SE": plain_sd / np.sqrt(config.selection_replications),
            }
        )
    comparison = pd.DataFrame(rows)
    candidates = comparison.loc[comparison["mu_IS"] != 0.0]
    if candidates.empty:
        raise ValueError("mu_grid must contain at least one nonzero candidate shift.")
    selected = float(candidates.loc[candidates["Variance_Reduction_Ratio"].idxmax(), "mu_IS"])
    return comparison, selected


def summarize_replications(
    plain: pd.DataFrame,
    importance: pd.DataFrame,
    selected_mu: float,
    variance_reduction_ratio: float,
) -> pd.DataFrame:
    """Summarize point estimates and across-replication uncertainty in stable columns."""
    rows = []
    for method, frame in (("Plain MC", plain), ("Importance Sampling", importance)):
        average_ess = float(frame["ESS"].mean()) if "ESS" in frame else np.nan
        for metric in RISK_METRICS:
            sd = float(frame[metric].std(ddof=1))
            rows.append(
                {
                    "Portfolio": str(frame["Portfolio"].iloc[0]),
                    "Method": method,
                    "Metric": metric,
                    "Mean": float(frame[metric].mean()),
                    "SD": sd,
                    "SE": sd / np.sqrt(len(frame)),
                    "Average_ESS": average_ess,
                    "Selected_mu_IS": selected_mu,
                    "Variance_Reduction_Ratio": variance_reduction_ratio,
                }
            )
    return pd.DataFrame(rows)


def run_analysis(
    input_path: Path,
    output_dir: Path,
    config: AnalysisConfig = AnalysisConfig(),
    *,
    make_figures: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run selection then final comparisons using one reference-compatible RNG stream."""
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    portfolios = load_portfolios(input_path, config)

    np.random.seed(config.seed)
    all_plain: list[pd.DataFrame] = []
    all_importance: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    selection_rows = []

    for label, portfolio in portfolios.items():
        selection, selected_mu = select_shift(portfolio, config, np.random)
        selection.to_csv(table_dir / f"{label}_mu_selection.csv", index=False)
        selected_row = selection.loc[selection["mu_IS"] == selected_mu].iloc[0]
        selection_rows.append(
            {
                "Portfolio": label,
                "Selected_mu_IS": selected_mu,
                "Variance_Reduction_Ratio": selected_row["Variance_Reduction_Ratio"],
                "ES99_SE": selected_row["ES99_SE"],
                "Average_ESS": selected_row["Average_ESS"],
                "ESS_Ratio": selected_row["ESS_Ratio"],
            }
        )

        plain = _plain_replications(portfolio, config, config.replications, np.random)
        importance, last_weights = _importance_replications(
            portfolio, config, config.replications, selected_mu, np.random
        )
        summary = summarize_replications(
            plain,
            importance,
            selected_mu,
            float(selected_row["Variance_Reduction_Ratio"]),
        )
        plain.to_csv(table_dir / f"{label}_plain_mc_replications.csv", index=False)
        importance.to_csv(
            table_dir / f"{label}_importance_sampling_replications.csv", index=False
        )
        summary.to_csv(table_dir / f"{label}_summary.csv", index=False)
        if make_figures:
            plot_mu_selection(label, selection, figure_dir)
            plot_replication_comparison(label, plain, importance, last_weights, figure_dir)
        all_plain.append(plain)
        all_importance.append(importance)
        all_summaries.append(summary)

    selected = pd.DataFrame(selection_rows)
    combined_plain = pd.concat(all_plain, ignore_index=True)
    combined_importance = pd.concat(all_importance, ignore_index=True)
    combined_summary = pd.concat(all_summaries, ignore_index=True)
    selected.to_csv(table_dir / "selected_mu.csv", index=False)
    combined_plain.to_csv(table_dir / "all_plain_mc_replications.csv", index=False)
    combined_importance.to_csv(
        table_dir / "all_importance_sampling_replications.csv", index=False
    )
    combined_summary.to_csv(table_dir / "all_portfolios_summary.csv", index=False)
    return {
        "selected_mu": selected,
        "plain": combined_plain,
        "importance_sampling": combined_importance,
        "summary": combined_summary,
    }
