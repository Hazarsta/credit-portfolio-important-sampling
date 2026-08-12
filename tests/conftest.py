from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

@pytest.fixture
def valid_frame() -> pd.DataFrame:
    count = 50
    return pd.DataFrame(
        {
            "Name": [f"Issuer {i:02d}" for i in range(count)],
            "PortfolioWeight": np.full(count, 1.0 / count),
            "1_year_default_probability": np.linspace(0.002, 0.08, count),
        }
    )


@pytest.fixture
def valid_workbook(tmp_path: Path, valid_frame: pd.DataFrame) -> Path:
    path = tmp_path / "portfolios.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet in ("Sheet1", "Sheet2", "Sheet3"):
            valid_frame.to_excel(writer, sheet_name=sheet, index=False)
    return path
