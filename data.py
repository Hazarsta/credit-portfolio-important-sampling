"""Workbook loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import AnalysisConfig, PORTFOLIO_SHEETS

REQUIRED_COLUMNS = ("Name", "PortfolioWeight", "1_year_default_probability")
EXCEL_ERRORS = (
    "#NAME?", "#VALUE!", "#DIV/0!", "#N/A", "#N/A N/A", "#REF!", "#NUM!", "#NULL!"
)


@dataclass(frozen=True)
class Portfolio:
    """Validated issuer inputs and simulation-ready arrays."""

    label: str
    frame: pd.DataFrame
    thresholds: np.ndarray
    loss_if_default: np.ndarray


def validate_workbook_sheets(path: Path, required_sheets: tuple[str, ...]) -> None:
    """Require every named simulation sheet before reading portfolio data."""
    if not path.is_file():
        raise FileNotFoundError(f"Input workbook not found: {path}")
    available = set(pd.ExcelFile(path).sheet_names)
    missing = [sheet for sheet in required_sheets if sheet not in available]
    if missing:
        raise ValueError(
            f"Workbook {path} is missing required sheets {missing}; "
            f"available sheets are {sorted(available)}."
        )


def load_portfolio(
    path: Path,
    sheet_name: str,
    label: str,
    config: AnalysisConfig,
    *,
    expected_issuers: int = 50,
    weight_tolerance: float = 1e-6,
) -> Portfolio:
    """Load one sheet without renormalizing its supplied exposure weights."""
    frame = pd.read_excel(path, sheet_name=sheet_name, header=0)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Sheet {sheet_name!r} is missing required columns {missing}; "
            f"available columns are {list(frame.columns)}."
        )

    frame = frame.loc[:, REQUIRED_COLUMNS].replace(list(EXCEL_ERRORS), np.nan).copy()
    frame = frame.rename(columns={"1_year_default_probability": "PD"})
    frame["PortfolioWeight"] = pd.to_numeric(frame["PortfolioWeight"], errors="coerce")
    frame["PD"] = pd.to_numeric(frame["PD"], errors="coerce")
    before = len(frame)
    frame = frame.dropna(subset=["Name", "PortfolioWeight", "PD"]).copy()
    dropped = before - len(frame)
    if dropped:
        warnings.warn(
            f"Dropped {dropped} row(s) from {sheet_name!r} with missing or invalid "
            "Name, PortfolioWeight, or PD.",
            stacklevel=2,
        )

    invalid_pd = frame.loc[~frame["PD"].between(0, 1, inclusive="neither"), "PD"]
    if not invalid_pd.empty:
        raise ValueError(f"Sheet {sheet_name!r} requires 0 < PD < 1; found {invalid_pd.tolist()}.")
    invalid_weights = frame.loc[frame["PortfolioWeight"] <= 0, "PortfolioWeight"]
    if not invalid_weights.empty:
        raise ValueError(
            f"Sheet {sheet_name!r} requires positive portfolio weights; "
            f"found {invalid_weights.tolist()}."
        )
    if len(frame) != expected_issuers:
        raise ValueError(
            f"Sheet {sheet_name!r} contains {len(frame)} valid issuers; expected {expected_issuers}."
        )
    weight_sum = float(frame["PortfolioWeight"].sum())
    if not np.isclose(weight_sum, 1.0, atol=weight_tolerance, rtol=0.0):
        raise ValueError(
            f"Sheet {sheet_name!r} weights sum to {weight_sum:.10f}, not approximately 1. "
            "Correct the workbook; weights are never silently renormalized."
        )

    frame["EAD"] = frame["PortfolioWeight"] * config.portfolio_value
    thresholds = norm.ppf(frame["PD"].to_numpy(dtype=float))
    loss_if_default = frame["EAD"].to_numpy(dtype=float) * config.lgd
    return Portfolio(label, frame, thresholds, loss_if_default)


def load_portfolios(
    path: Path,
    config: AnalysisConfig,
    sheet_mapping: dict[str, str] = PORTFOLIO_SHEETS,
) -> dict[str, Portfolio]:
    """Validate and load the three portfolios in deterministic processing order."""
    validate_workbook_sheets(path, tuple(sheet_mapping.values()))
    return {
        label: load_portfolio(path, sheet, label, config)
        for label, sheet in sheet_mapping.items()
    }
