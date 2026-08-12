"""Consistent batch figures for the replication study."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import pandas as pd

PLAIN_COLOR = "#4C78A8"
IS_COLOR = "#E45756"
ESS_COLOR = "#54A24B"


def _finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mu_selection(label: str, table: pd.DataFrame, output_dir: Path) -> None:
    """Plot ES99 standard error and ESS ratio across candidate shifts."""
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(table["mu_IS"], table["ES99_SE"], "o-", color=IS_COLOR, label="ES99 SE")
    axis.set_xlabel(r"Systemic-factor shift $\mu_{IS}$")
    axis.set_ylabel("ES99 standard error (USD)", color=IS_COLOR)
    axis.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    axis.tick_params(axis="y", labelcolor=IS_COLOR)
    second = axis.twinx()
    second.plot(
        table["mu_IS"], table["ESS_Ratio"], "s--", color=ESS_COLOR, label="ESS ratio"
    )
    second.set_ylabel("Effective sample size / paths", color=ESS_COLOR)
    second.tick_params(axis="y", labelcolor=ESS_COLOR)
    second.set_ylim(bottom=0)
    axis.set_title(f"{label}: importance-sampling shift selection")
    _finish(fig, output_dir / f"{label}_mu_selection.png")


def plot_replication_comparison(
    label: str,
    plain: pd.DataFrame,
    importance: pd.DataFrame,
    last_weights,
    output_dir: Path,
) -> None:
    """Save scatter, boxplot, SD bars, and final-weight histogram."""
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(plain["Replication"], plain["ES99"], color=PLAIN_COLOR, label="Plain MC")
    axis.scatter(
        importance["Replication"], importance["ES99"], marker="s", color=IS_COLOR,
        label="Importance sampling",
    )
    axis.axhline(plain["ES99"].mean(), color=PLAIN_COLOR, linestyle="--")
    axis.axhline(importance["ES99"].mean(), color=IS_COLOR, linestyle="--")
    axis.set(xlabel="Replication", ylabel="ES99 (USD)", title=f"{label}: ES99 by replication")
    axis.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    axis.legend()
    _finish(fig, output_dir / f"{label}_es99_scatter.png")

    fig, axis = plt.subplots(figsize=(7, 5))
    box = axis.boxplot(
        [plain["ES99"], importance["ES99"]],
        tick_labels=["Plain MC", "Importance sampling"],
        patch_artist=True,
        showmeans=True,
    )
    for patch, color in zip(box["boxes"], (PLAIN_COLOR, IS_COLOR)):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    axis.set(ylabel="ES99 (USD)", title=f"{label}: ES99 sampling variability")
    axis.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    _finish(fig, output_dir / f"{label}_es99_boxplot.png")

    sds = [plain["ES99"].std(ddof=1), importance["ES99"].std(ddof=1)]
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.bar(["Plain MC", "Importance sampling"], sds, color=[PLAIN_COLOR, IS_COLOR])
    axis.set(ylabel="ES99 standard deviation (USD)", title=f"{label}: ES99 variance reduction")
    axis.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    _finish(fig, output_dir / f"{label}_es99_sd.png")

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.hist(last_weights, bins=80, color=IS_COLOR, edgecolor="white", alpha=0.8)
    axis.set(
        xlabel="Likelihood-ratio weight",
        ylabel="Frequency",
        title=f"{label}: final importance-sampling weights",
    )
    _finish(fig, output_dir / f"{label}_is_weights.png")
