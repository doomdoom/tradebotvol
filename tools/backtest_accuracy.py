"""Out-of-sample backtest: baseline (rule) vs enhanced model vs majority class.

Replays real historical candles for each (symbol, timeframe). The enhanced
model is trained **walk-forward** (expanding window, no shuffle) and only ever
predicts unseen rows; the rule baseline is scored on the exact same rows so the
comparison is apples-to-apples. The majority-class accuracy is reported as the
floor any real model must clear.

Outputs (under reports/):
    backtest_summary.json      full machine-readable metrics
    backtest_summary.md        human-readable report
    confidence_calibration.json calibration buckets per model

Read-only public data only. No trading, no orders, no private keys.

Usage:
    py tools/backtest_accuracy.py
    py tools/backtest_accuracy.py --symbols BTCUSDT ETHUSDT --timeframes 5m 15m 1h
    py tools/backtest_accuracy.py --candles 5000 --folds 6
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
    WAIT,
    accuracy_by,
    classification_metrics,
    confidence_calibration,
    coverage,
    direction_distribution,
    majority_baseline,
)
from predictor.binance_data import BinanceDataClient  # noqa: E402
from predictor.config import Config, load_config  # noqa: E402
from predictor.feature_engineering import add_labels, build_feature_frame  # noqa: E402
from predictor.historical_data import download_history  # noqa: E402
from predictor.logger import get_logger, setup_logging  # noqa: E402
from predictor.regime import add_regime  # noqa: E402
from predictor.research_model import EnhancedModel, config_to_enhanced  # noqa: E402
from predictor.rule_based_predictor import RuleBasedPredictor  # noqa: E402
from predictor.utils import confidence_label, higher_timeframes  # noqa: E402

log = get_logger("backtest_accuracy")

_SESSIONS = [(0, 8, "Asia"), (8, 16, "Europe"), (16, 24, "US")]


def _session(hour: float) -> str:
    for lo, hi, name in _SESSIONS:
        if lo <= hour < hi:
            return name
    return "?"


def _prepare(config: Config, client: BinanceDataClient, symbol: str, timeframe: str,
             candles: int) -> tuple[pd.DataFrame, list[str]]:
    df = download_history(client, symbol, timeframe, candles, cache_dir=config.data_dir)
    htf_frames: dict[str, pd.DataFrame] = {}
    if config.use_higher_timeframe_context:
        for htf in higher_timeframes(timeframe, config.higher_timeframe_count):
            htf_frames[htf] = download_history(
                client, symbol, htf, max(300, candles // 3), cache_dir=config.data_dir
            )
    frame, feature_cols = build_feature_frame(df, htf_frames or None)
    frame = add_labels(frame, config.neutral_threshold_pct)
    frame = add_regime(frame)
    frame = frame.dropna(subset=["label", *feature_cols]).reset_index(drop=True)
    return frame, feature_cols


def _walk_forward_enhanced(
    frame: pd.DataFrame, feature_cols: list[str], cfg, folds: int,
    initial_train_pct: float = 0.5,
) -> pd.DataFrame:
    """Expanding-window OOS predictions from the enhanced model.

    Returns the tail rows (all fold test sets) with enhanced-model columns
    attached. These same rows are used to score the baseline.
    """
    n = len(frame)
    start = int(n * initial_train_pct)
    fold_size = max(1, (n - start) // folds)
    pieces: list[pd.DataFrame] = []
    for fold in range(folds):
        train_end = start + fold * fold_size
        test_end = min(train_end + fold_size, n)
        if train_end >= n or train_end == test_end:
            break
        train = frame.iloc[:train_end]
        test = frame.iloc[train_end:test_end]
        model = EnhancedModel(feature_cols, cfg).fit(train, train["label"])
        preds = model.predict_frame(test)
        merged = test.reset_index(drop=True).join(preds.reset_index(drop=True), rsuffix="_pred")
        pieces.append(merged)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def _records_from_enhanced(oos: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "symbol": symbol,
            "timeframe": timeframe,
            "predicted_direction": oos["predicted_direction"].values,
            "actual_direction": oos["label"].values,
            "confidence": oos["confidence"].values,
            "edge": oos["edge"].values,
            "signal_strength": oos["signal_strength"].values,
            "regime": oos["market_regime"].values,
            "hour_of_day": oos["hour_of_day"].values,
        }
    )
    rows["session"] = rows["hour_of_day"].map(_session)
    rows["confidence_label"] = rows["confidence"].map(confidence_label)
    rows["prediction_correct"] = np.where(
        rows["predicted_direction"] == WAIT, np.nan,
        (rows["predicted_direction"] == rows["actual_direction"]).astype(float),
    )
    rows["model"] = "enhanced"
    return rows


def _records_from_rule(oos: pd.DataFrame, config: Config, symbol: str,
                       timeframe: str) -> pd.DataFrame:
    predictor = RuleBasedPredictor(config.neutral_threshold_pct)
    dirs, confs = [], []
    for row in oos.itertuples(index=False):
        pred = predictor.predict(pd.Series(row._asdict()))
        dirs.append(pred.direction)
        confs.append(pred.confidence_score)
    rows = pd.DataFrame(
        {
            "symbol": symbol,
            "timeframe": timeframe,
            "predicted_direction": dirs,
            "actual_direction": oos["label"].values,
            "confidence": confs,
            "regime": oos["regime"].values,
            "hour_of_day": oos["hour_of_day"].values,
        }
    )
    rows["signal_strength"] = ""
    rows["session"] = rows["hour_of_day"].map(_session)
    rows["confidence_label"] = rows["confidence"].map(confidence_label)
    rows["prediction_correct"] = (
        rows["predicted_direction"] == rows["actual_direction"]
    ).astype(float)
    rows["model"] = "baseline"
    return rows


def _model_block(df: pd.DataFrame) -> dict:
    """All metrics for one model across all its records."""
    return {
        "metrics": classification_metrics(df["actual_direction"], df["predicted_direction"]),
        "coverage": coverage(df["predicted_direction"]),
        "majority_baseline": majority_baseline(df["actual_direction"]),
        "predicted_distribution": direction_distribution(df["predicted_direction"]),
        "actual_distribution": direction_distribution(df["actual_direction"]),
        "accuracy_by_symbol": accuracy_by(df, "symbol", min_n=20),
        "accuracy_by_timeframe": accuracy_by(df, "timeframe", min_n=20),
        "accuracy_by_regime": accuracy_by(df, "regime", min_n=15),
        "accuracy_by_session": accuracy_by(df, "session", min_n=20),
        "accuracy_by_confidence_label": accuracy_by(df, "confidence_label", min_n=10),
        "calibration": confidence_calibration(df),
    }


def _selectivity_curve(df: pd.DataFrame, thresholds=(0.50, 0.55, 0.60, 0.65, 0.70)) -> list[dict]:
    """Accuracy vs coverage as the confidence gate tightens.

    Shows the statistical-edge trade-off: if the bot only commits when its
    (calibrated) confidence >= threshold, how often does it act (coverage) and
    how accurate is it on those calls, vs the majority-class floor on that same
    subset? This is where selective signalling + NO-SIGNAL mode earn their keep.
    """
    total = int(len(df))
    conf = pd.to_numeric(df["confidence"], errors="coerce")
    out: list[dict] = []
    for thr in thresholds:
        sel = df[conf >= thr]
        n = int(len(sel))
        if n == 0:
            out.append({"min_confidence": thr, "coverage_pct": 0.0, "n": 0,
                        "accuracy_pct": float("nan"), "majority_pct": float("nan")})
            continue
        acc = float((sel["predicted_direction"] == sel["actual_direction"]).mean() * 100.0)
        maj = majority_baseline(sel["actual_direction"])["accuracy_pct"]
        out.append({
            "min_confidence": thr,
            "coverage_pct": n / total * 100.0 if total else float("nan"),
            "n": n,
            "accuracy_pct": acc,
            "majority_pct": maj,
            "beats_floor": bool(acc > maj + 0.5),
        })
    return out


def _regime_edge(df: pd.DataFrame, min_n: int = 15) -> dict[str, dict]:
    """Per-regime accuracy vs the majority-class floor *within that regime*.

    A regime shows genuine edge only when its accuracy clears its own local
    floor (not just the global one) — breakouts, for instance, are inherently
    directional so their floor differs from choppy conditions.
    """
    out: dict[str, dict] = {}
    for reg, g in df[df["predicted_direction"] != WAIT].groupby("regime"):
        n = int(len(g))
        if n < min_n:
            continue
        acc = float((g["predicted_direction"] == g["actual_direction"]).mean() * 100.0)
        floor = majority_baseline(g["actual_direction"])["accuracy_pct"]
        out[str(reg)] = {
            "n": n,
            "accuracy_pct": acc,
            "floor_pct": floor,
            "edge_pct": acc - floor,
            "beats_floor": bool(acc > floor + 0.5),
        }
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["edge_pct"]))


def _pair_summary(df: pd.DataFrame) -> dict:
    out = {}
    for (sym, tf), g in df.groupby(["symbol", "timeframe"]):
        m = classification_metrics(g["actual_direction"], g["predicted_direction"])
        out[f"{sym} {tf}"] = {
            "n": m["n"],
            "accuracy_pct": m["accuracy_pct"],
            "macro_f1_pct": m["macro_f1_pct"],
            "majority_pct": majority_baseline(g["actual_direction"])["accuracy_pct"],
            "coverage_pct": coverage(g["predicted_direction"])["coverage_pct"],
        }
    return out


def _fmt(x, suffix="%"):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1f}{suffix}"


def _markdown(summary: dict) -> str:
    L: list[str] = []
    L.append("# TradeBotVol — Backtest Summary\n")
    L.append(f"_Generated: {summary['generated_at']}_  ")
    L.append(f"_Config: neutral_threshold={summary['neutral_threshold_pct']}%, "
             f"candles/pair={summary['candles']}, walk-forward folds={summary['folds']}, "
             f"market={summary['market_type']}_\n")
    L.append(f"**Out-of-sample rows evaluated:** {summary['n_oos_rows']} "
             f"across {len(summary['pairs'])} pairs.\n")

    b, e = summary["baseline"], summary["enhanced"]
    L.append("## Headline: baseline vs enhanced vs majority\n")
    L.append("| Model | OOS acc | Macro-F1 | Majority floor | Coverage |")
    L.append("|-------|---------|----------|----------------|----------|")
    L.append(f"| Baseline (rule) | {_fmt(b['metrics']['accuracy_pct'])} | "
             f"{_fmt(b['metrics']['macro_f1_pct'])} | "
             f"{_fmt(b['majority_baseline']['accuracy_pct'])} | "
             f"{_fmt(b['coverage']['coverage_pct'])} |")
    L.append(f"| **Enhanced (logreg)** | {_fmt(e['metrics']['accuracy_pct'])} | "
             f"{_fmt(e['metrics']['macro_f1_pct'])} | "
             f"{_fmt(e['majority_baseline']['accuracy_pct'])} | "
             f"{_fmt(e['coverage']['coverage_pct'])} |")
    L.append(f"\n**Verdict:** {summary['verdict']}\n")

    L.append("## Per-pair (enhanced, out-of-sample)\n")
    L.append("| Pair | n | Acc | Macro-F1 | Majority | Coverage |")
    L.append("|------|---|-----|----------|----------|----------|")
    for pair, s in summary["enhanced_pairs"].items():
        L.append(f"| {pair} | {s['n']} | {_fmt(s['accuracy_pct'])} | "
                 f"{_fmt(s['macro_f1_pct'])} | {_fmt(s['majority_pct'])} | "
                 f"{_fmt(s['coverage_pct'])} |")

    for name, block in (("Baseline", b), ("Enhanced", e)):
        L.append(f"\n## {name}: confidence calibration\n")
        L.append("| Bucket | n | Mean conf | Accuracy | Gap |")
        L.append("|--------|---|-----------|----------|-----|")
        for c in block["calibration"]:
            L.append(f"| {c['bucket']} | {c['n']} | {_fmt(c['mean_confidence_pct'])} | "
                     f"{_fmt(c['accuracy_pct'])} | {_fmt(c['gap_pct'])} |")

    L.append("\n## Enhanced: selectivity curve (accuracy vs coverage)\n")
    L.append("_Only commit when calibrated confidence >= threshold._\n")
    L.append("| Min confidence | Coverage | n | Accuracy | Majority floor | Beats floor |")
    L.append("|----------------|----------|---|----------|----------------|-------------|")
    for s in summary["enhanced_selectivity"]:
        beats = "yes" if s.get("beats_floor") else "no"
        L.append(f"| {int(s['min_confidence']*100)}% | {_fmt(s['coverage_pct'])} | {s['n']} | "
                 f"{_fmt(s['accuracy_pct'])} | {_fmt(s['majority_pct'])} | {beats} |")

    L.append("\n## Enhanced: edge by market regime (accuracy vs local floor)\n")
    L.append("| Regime | n | Accuracy | Local floor | Edge | Beats floor |")
    L.append("|--------|---|----------|-------------|------|-------------|")
    for reg, s in summary["enhanced_regime_edge"].items():
        beats = "yes" if s["beats_floor"] else "no"
        L.append(f"| {reg} | {s['n']} | {_fmt(s['accuracy_pct'])} | {_fmt(s['floor_pct'])} | "
                 f"{_fmt(s['edge_pct'], 'pp')} | {beats} |")

    L.append("\n_Prediction/research only. No trades are placed. Accuracy is not a "
             "profit guarantee; short timeframes are close to a coin flip._\n")
    return "\n".join(L)


def _verdict(baseline: dict, enhanced: dict) -> str:
    ba = baseline["metrics"]["accuracy_pct"]
    ea = enhanced["metrics"]["accuracy_pct"]
    maj = enhanced["majority_baseline"]["accuracy_pct"]
    if np.isnan(ea) or np.isnan(ba):
        return "Not enough data to compare."
    beats_base = ea > ba + 0.5
    beats_maj = ea > maj + 0.5
    parts = [f"Enhanced {ea:.1f}% vs baseline {ba:.1f}% vs majority-floor {maj:.1f}%."]
    if beats_base and beats_maj:
        parts.append("Enhanced beats BOTH baseline and the majority floor out-of-sample "
                     "-> candidate for enabling (per pair where it wins).")
    elif beats_base:
        parts.append("Enhanced beats the baseline but not the majority floor -> keep OFF; "
                     "a class-imbalance-aware approach is needed.")
    else:
        parts.append("Enhanced does NOT beat the baseline out-of-sample -> keep enhanced OFF.")
    return " ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--symbols", nargs="*", help="override symbols")
    parser.add_argument("--timeframes", nargs="*", help="override timeframes")
    parser.add_argument("--candles", type=int, default=3000, help="history per pair")
    parser.add_argument("--folds", type=int, default=5, help="walk-forward folds")
    parser.add_argument("--outdir", default="reports")
    parser.add_argument("--generated-at", default="", help="timestamp label (optional)")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.log_level)
    client = BinanceDataClient(config.market_type)
    cfg = config_to_enhanced(config)

    symbols = [s.upper() for s in (args.symbols or config.symbols)]
    timeframes = args.timeframes or config.timeframes

    base_all: list[pd.DataFrame] = []
    enh_all: list[pd.DataFrame] = []
    pairs_done: list[str] = []
    for symbol in symbols:
        for tf in timeframes:
            try:
                log.info("Backtesting %s %s (%d candles, %d folds)...",
                         symbol, tf, args.candles, args.folds)
                frame, feats = _prepare(config, client, symbol, tf, args.candles)
                if len(frame) < 400:
                    log.warning("%s %s: only %d usable rows - skipping", symbol, tf, len(frame))
                    continue
                oos = _walk_forward_enhanced(frame, feats, cfg, args.folds)
                if oos.empty:
                    log.warning("%s %s: no OOS rows produced - skipping", symbol, tf)
                    continue
                enh_all.append(_records_from_enhanced(oos, symbol, tf))
                base_all.append(_records_from_rule(oos, config, symbol, tf))
                pairs_done.append(f"{symbol} {tf}")
            except Exception as exc:
                log.error("%s %s: backtest failed: %s", symbol, tf, exc)

    if not enh_all:
        print("No backtest results produced (no pairs had enough data).")
        return 1

    enh_df = pd.concat(enh_all, ignore_index=True)
    base_df = pd.concat(base_all, ignore_index=True)

    enhanced_block = _model_block(enh_df)
    baseline_block = _model_block(base_df)
    summary = {
        "generated_at": args.generated_at or "(unstamped)",
        "market_type": config.market_type,
        "neutral_threshold_pct": config.neutral_threshold_pct,
        "candles": args.candles,
        "folds": args.folds,
        "pairs": pairs_done,
        "n_oos_rows": int(len(enh_df)),
        "baseline": baseline_block,
        "enhanced": enhanced_block,
        "enhanced_pairs": _pair_summary(enh_df),
        "baseline_pairs": _pair_summary(base_df),
        "enhanced_selectivity": _selectivity_curve(enh_df),
        "enhanced_regime_edge": _regime_edge(enh_df),
        "verdict": _verdict(baseline_block, enhanced_block),
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "backtest_summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    (outdir / "confidence_calibration.json").write_text(
        json.dumps(
            {"baseline": baseline_block["calibration"],
             "enhanced": enhanced_block["calibration"]},
            indent=2, default=float,
        ),
        encoding="utf-8",
    )
    md = _markdown(summary)
    (outdir / "backtest_summary.md").write_text(md, encoding="utf-8")

    print(md)
    print(f"\nSaved: {outdir/'backtest_summary.json'}, {outdir/'backtest_summary.md'}, "
          f"{outdir/'confidence_calibration.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
