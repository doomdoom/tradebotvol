"""Research / Accuracy Lab report from the LIVE prediction log.

Reads resolved predictions out of the SQLite log and summarises real, realised
accuracy (overall, by symbol / timeframe / market regime / confidence bucket),
plus confidence calibration and the majority-class floor. If a backtest summary
exists (reports/backtest_summary.json) its headline is folded in for context.

Outputs:
    reports/research_report.json
    reports/research_report.md

Read-only. No trading, no network, no private data.

Usage:
    py tools/research_report.py
    py tools/research_report.py --config config.json --outdir reports
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor.backtest_metrics import (  # noqa: E402
    accuracy_by,
    classification_metrics,
    confidence_calibration,
    direction_distribution,
    majority_baseline,
)
from predictor.config import load_config  # noqa: E402
from predictor.logger import setup_logging  # noqa: E402
from predictor.storage import PredictionStorage  # noqa: E402


def _fmt(x, suffix="%"):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1f}{suffix}"


def _load_resolved(config) -> pd.DataFrame:
    storage = PredictionStorage(
        config.db_path, config.csv_path, sqlite_enabled=True, csv_enabled=False
    )
    try:
        df = storage.load_dataframe()
    finally:
        storage.close()
    if df.empty or "actual_direction" not in df.columns:
        return pd.DataFrame()
    return df[df["actual_direction"].notna()].copy()


def build_summary(config) -> dict:
    df = _load_resolved(config)
    summary: dict = {"resolved": int(len(df))}
    if df.empty:
        summary["message"] = (
            "No resolved predictions yet. Let run_predictor.py run, or seed the "
            "log with 'py backtest_predictions.py --save'."
        )
        return summary

    # Normalise a couple of optional columns that older rows may lack.
    if "regime" not in df.columns and "market_regime" in df.columns:
        df["regime"] = df["market_regime"]
    if "regime" not in df.columns:
        df["regime"] = "UNKNOWN"
    df["regime"] = df["regime"].fillna("UNKNOWN")

    m = classification_metrics(df["actual_direction"], df["predicted_direction"])
    summary.update(
        {
            "overall_accuracy_pct": m["accuracy_pct"],
            "macro_f1_pct": m["macro_f1_pct"],
            "majority_floor": majority_baseline(df["actual_direction"]),
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "predicted_distribution": direction_distribution(df["predicted_direction"]),
            "actual_distribution": direction_distribution(df["actual_direction"]),
            "model_versions": sorted(df.get("model_version", pd.Series(dtype=str))
                                     .dropna().unique().tolist()),
            "accuracy_by_symbol": accuracy_by(df, "symbol", min_n=1),
            "accuracy_by_timeframe": accuracy_by(df, "timeframe", min_n=1),
            "accuracy_by_regime": accuracy_by(df, "regime", min_n=1),
            "accuracy_by_confidence_label": accuracy_by(df, "confidence_label", min_n=1),
            "calibration": confidence_calibration(df),
        }
    )
    return summary


def _read_backtest(outdir: Path) -> dict | None:
    path = outdir / "backtest_summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def to_markdown(summary: dict, backtest: dict | None) -> str:
    L: list[str] = ["# TradeBotVol — Research / Accuracy Lab\n"]
    if summary.get("resolved", 0) == 0:
        L.append(summary.get("message", "No data."))
        return "\n".join(L)

    maj = summary["majority_floor"]
    L.append(f"**Live resolved predictions:** {summary['resolved']}  ")
    L.append(f"**Overall accuracy:** {_fmt(summary['overall_accuracy_pct'])} "
             f"(macro-F1 {_fmt(summary['macro_f1_pct'])})  ")
    L.append(f"**Majority-class floor:** {_fmt(maj['accuracy_pct'])} "
             f"(always predict {maj['class']})  ")
    L.append(f"**Model versions in log:** {', '.join(summary['model_versions']) or 'n/a'}\n")

    if backtest:
        b = backtest.get("baseline", {}).get("metrics", {})
        e = backtest.get("enhanced", {}).get("metrics", {})
        L.append("## Backtest headline (out-of-sample)\n")
        L.append(f"- Baseline (rule): {_fmt(b.get('accuracy_pct'))}, "
                 f"Enhanced: {_fmt(e.get('accuracy_pct'))}, "
                 f"majority floor: {_fmt(backtest.get('enhanced',{}).get('majority_baseline',{}).get('accuracy_pct'))}")
        L.append(f"- {backtest.get('verdict','')}\n")

    def _table(title, mapping, cols=("key", "n", "Accuracy")):
        L.append(f"## {title}\n")
        L.append(f"| {cols[0]} | {cols[1]} | {cols[2]} |")
        L.append("|---|---|---|")
        for k, s in mapping.items():
            L.append(f"| {k} | {s['n']} | {_fmt(s['accuracy_pct'])} |")
        L.append("")

    _table("Accuracy by symbol", summary["accuracy_by_symbol"])
    _table("Accuracy by timeframe", summary["accuracy_by_timeframe"])
    _table("Accuracy by market regime", summary["accuracy_by_regime"])
    _table("Accuracy by confidence label", summary["accuracy_by_confidence_label"])

    L.append("## Confidence calibration (live)\n")
    L.append("| Bucket | n | Mean conf | Accuracy | Gap |")
    L.append("|---|---|---|---|---|")
    for c in summary["calibration"]:
        L.append(f"| {c['bucket']} | {c['n']} | {_fmt(c['mean_confidence_pct'])} | "
                 f"{_fmt(c['accuracy_pct'])} | {_fmt(c['gap_pct'])} |")

    L.append("\n_Live sample; small counts are noisy. Prediction/research only — "
             "no trades are placed and accuracy is not a profit guarantee._")
    return "\n".join(L)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--outdir", default="reports")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.log_level)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(config)
    backtest = _read_backtest(outdir)
    md = to_markdown(summary, backtest)

    (outdir / "research_report.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    (outdir / "research_report.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\nSaved: {outdir/'research_report.json'}, {outdir/'research_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
