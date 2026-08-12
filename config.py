"""Central analysis settings for the flat repository layout."""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class AnalysisConfig:
    """Immutable model, simulation, and reproducibility settings."""

    portfolio_value: float = 100_000_000.0
    lgd: float = 0.40
    asset_correlation: float = 0.20
    paths_per_replication: int = 20_000
    replications: int = 50
    selection_replications: int = 50
    seed: int = 42
    mu_grid: tuple[float, ...] = tuple(round(-0.1 * i, 1) for i in range(15))

    def quick(self) -> "AnalysisConfig":
        """Return settings suitable for a complete smoke run."""
        return replace(
            self,
            paths_per_replication=min(self.paths_per_replication, 2_000),
            replications=min(self.replications, 5),
            selection_replications=min(self.selection_replications, 5),
        )


PORTFOLIO_SHEETS: dict[str, str] = {
    "30_70": "Sheet1",
    "50_50": "Sheet2",
    "70_30": "Sheet3",
}
