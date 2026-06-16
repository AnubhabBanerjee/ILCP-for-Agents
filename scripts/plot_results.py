#!/usr/bin/env python3
"""
Plot benchmark CSVs produced by scripts/benchmark_campaign.py (copied from the .example template).

Matplotlib defaults to Agg so headless GTX 1080 rigs without a desktop session still emit PNG receipts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    """
    Read all_trials.csv, emit latency strip plots, and never fabricate summary statistics beyond the file.

    Keeping aggregation strictly pandas-driven prevents hand-typed p50/p95 tables drifting from raw CSVs.
    """
    parser = argparse.ArgumentParser(description="Plot ILCP benchmark receipts.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=_ROOT / "examples" / "example-run-results" / "all_trials.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=_ROOT / "examples" / "example-run-results" / "plots")
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"missing csv: {args.csv} (run benchmark_campaign first)")

    df = pd.read_csv(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for branch in sorted(df["branch"].unique().tolist()):
        sub = df[df["branch"] == branch]
        plt.figure(figsize=(8, 4))
        plt.title(f"handoff latency seconds — {branch}")
        plt.scatter(sub["trial"], sub["latency_s"], alpha=0.85)
        plt.xlabel("trial index")
        plt.ylabel("latency_s")
        out = args.out_dir / f"latency_{branch}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        plt.close()
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
