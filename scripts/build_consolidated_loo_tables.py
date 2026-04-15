#!/usr/bin/env python3
"""Build consolidated summary tables for the LOO evaluation runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.consolidated_loo_eval import PRIMARY_M, build_consolidated_tables, save_consolidated_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Build consolidated LOO evaluation tables.")
    parser.add_argument(
        "--output-dir",
        default="analysis/consolidated_loo_tables",
        help="Directory for CSV and JSON outputs.",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=10000,
        help="Bootstrap repetitions for confidence intervals and paired tests.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Alpha level for confidence intervals and significance.",
    )
    parser.add_argument(
        "--primary-m",
        type=int,
        default=PRIMARY_M,
        help="Primary exemplar count used for Tables 1-3 and 5.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Base random seed for bootstrap sampling.",
    )
    parser.add_argument(
        "--skip-regex-recalc",
        action="store_true",
        help="Reuse existing regex artifacts instead of recomputing them.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any expected primary-table condition is missing or incomplete.",
    )
    args = parser.parse_args()

    artifacts = build_consolidated_tables(
        primary_m=int(args.primary_m),
        bootstrap_reps=int(args.bootstrap_reps),
        alpha=float(args.alpha),
        random_state=int(args.random_state),
        force_recalc_regex=not args.skip_regex_recalc,
        strict=bool(args.strict),
    )
    saved = save_consolidated_tables(artifacts, Path(args.output_dir))

    for key, path in saved.items():
        print(f"{key}: {path}")
    print(f"warnings: {len(artifacts['warnings'])}")


if __name__ == "__main__":
    main()
