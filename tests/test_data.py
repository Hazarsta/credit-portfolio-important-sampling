from pathlib import Path

import pandas as pd
import pytest

from config import AnalysisConfig
from data import load_portfolio, validate_workbook_sheets


def test_workbook_sheet_validation_rejects_missing(tmp_path: Path, valid_frame: pd.DataFrame) -> None:
    path = tmp_path / "missing.xlsx"
    valid_frame.to_excel(path, sheet_name="Sheet1", index=False)
    with pytest.raises(ValueError, match="missing required sheets"):
        validate_workbook_sheets(path, ("Sheet1", "Sheet2"))


def test_required_column_validation(tmp_path: Path, valid_frame: pd.DataFrame) -> None:
    path = tmp_path / "columns.xlsx"
    valid_frame.drop(columns="Name").to_excel(path, sheet_name="Sheet1", index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_portfolio(path, "Sheet1", "test", AnalysisConfig())


def test_invalid_pd_is_rejected(tmp_path: Path, valid_frame: pd.DataFrame) -> None:
    valid_frame.loc[0, "1_year_default_probability"] = 1.0
    path = tmp_path / "pd.xlsx"
    valid_frame.to_excel(path, sheet_name="Sheet1", index=False)
    with pytest.raises(ValueError, match="0 < PD < 1"):
        load_portfolio(path, "Sheet1", "test", AnalysisConfig())


@pytest.mark.parametrize("replacement", [0.0, -0.1])
def test_nonpositive_weight_is_rejected(
    tmp_path: Path, valid_frame: pd.DataFrame, replacement: float
) -> None:
    valid_frame.loc[0, "PortfolioWeight"] = replacement
    valid_frame.loc[1, "PortfolioWeight"] += 1.0 / 50 - replacement
    path = tmp_path / "weights.xlsx"
    valid_frame.to_excel(path, sheet_name="Sheet1", index=False)
    with pytest.raises(ValueError, match="positive portfolio weights"):
        load_portfolio(path, "Sheet1", "test", AnalysisConfig())


def test_material_weight_sum_error_is_not_renormalized(
    tmp_path: Path, valid_frame: pd.DataFrame
) -> None:
    valid_frame["PortfolioWeight"] *= 0.9
    path = tmp_path / "sum.xlsx"
    valid_frame.to_excel(path, sheet_name="Sheet1", index=False)
    with pytest.raises(ValueError, match="never silently renormalized"):
        load_portfolio(path, "Sheet1", "test", AnalysisConfig())
