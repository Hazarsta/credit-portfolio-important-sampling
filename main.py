#!/usr/bin/env python3
"""Run the full credit-tail-risk analysis from the repository root."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

from config import AnalysisConfig
from experiment import run_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/issuer_portfolios_lqd_hyg.xlsx")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paths", type=int, default=20_000)
    parser.add_argument("--replications", type=int, default=50)
    parser.add_argument("--selection-replications", type=int, default=50)
    parser.add_argument("--quick", action="store_true", help="Run a much smaller end-to-end smoke analysis.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AnalysisConfig(
        seed=args.seed,
        paths_per_replication=args.paths,
        replications=args.replications,
        selection_replications=args.selection_replications,
    )
    if args.quick:
        config = config.quick()
    if min(config.paths_per_replication, config.replications, config.selection_replications) < 2:
        print("error: paths and replication counts must all be at least 2", file=sys.stderr)
        return 2
    try:
        results = run_analysis(args.input, ROOT / "outputs", config)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("\nSelected systemic-factor shifts")
    print(results["selected_mu"].to_string(index=False))
    es99 = results["summary"].query("Metric == 'ES99'")
    print("\nES99 comparison")
    print(es99.to_string(index=False, formatters={"Mean": "${:,.0f}".format, "SD": "${:,.0f}".format}))
    print(f"\nOutputs written to {ROOT / 'outputs'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
