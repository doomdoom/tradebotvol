"""Prediction performance reports: accuracy, per-class precision/recall,
confusion matrices, confidence-vs-accuracy and best/worst rankings."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .logger import get_logger
from .storage import PredictionStorage
from .utils import BEARISH, BULLISH, DIRECTIONS

log = get_logger("evaluator")

#: Minimum resolved predictions before a pair enters best/worst rankings.
MIN_SAMPLE_FOR_RANKING = 20


@dataclass
class GroupReport:
    symbol: str
    timeframe: str
    total_predictions: int
    resolved: int
    accuracy_pct: float
    precision: dict[str, float]
    recall: dict[str, float]
    confusion: pd.DataFrame
    avg_move_after_bullish_pct: float | None
    avg_move_after_bearish_pct: float | None
    confidence_vs_accuracy: dict[str, dict[str, float]]
    model_types: list[str] = field(default_factory=list)


def load_resolved(storage: PredictionStorage) -> pd.DataFrame:
    df = storage.load_dataframe()
    return df[df["actual_direction"].notna()].copy()


# ---------------------------------------------------------------------- #
# Metrics for one (symbol, timeframe) group
# ---------------------------------------------------------------------- #


def evaluate_group(df: pd.DataFrame, symbol: str, timeframe: str) -> GroupReport:
    """Compute all metrics for one pair. ``df`` must be resolved rows only."""
    total = len(df)
    correct = int(df["prediction_correct"].fillna(0).sum())
    accuracy = correct / total * 100.0 if total else 0.0

    predicted = df["predicted_direction"]
    actual = df["actual_direction"]

    precision: dict[str, float] = {}
    recall: dict[str, float] = {}
    for cls in DIRECTIONS:
        pred_mask = predicted == cls
        act_mask = actual == cls
        tp = int((pred_mask & act_mask).sum())
        precision[cls] = tp / int(pred_mask.sum()) * 100.0 if pred_mask.any() else float("nan")
        recall[cls] = tp / int(act_mask.sum()) * 100.0 if act_mask.any() else float("nan")

    confusion = pd.crosstab(
        predicted, actual, rownames=["predicted"], colnames=["actual"], dropna=False
    ).reindex(index=DIRECTIONS, columns=DIRECTIONS, fill_value=0)

    bull_moves = df.loc[predicted == BULLISH, "actual_return_pct"].dropna()
    bear_moves = df.loc[predicted == BEARISH, "actual_return_pct"].dropna()

    conf_bins: dict[str, dict[str, float]] = {}
    for label, group in df.groupby("confidence_label"):
        conf_bins[str(label)] = {
            "count": float(len(group)),
            "accuracy_pct": float(group["prediction_correct"].fillna(0).mean() * 100.0),
            "avg_confidence": float(group["confidence"].mean() * 100.0),
        }

    return GroupReport(
        symbol=symbol,
        timeframe=timeframe,
        total_predictions=total,
        resolved=total,
        accuracy_pct=accuracy,
        precision=precision,
        recall=recall,
        confusion=confusion,
        avg_move_after_bullish_pct=float(bull_moves.mean()) if len(bull_moves) else None,
        avg_move_after_bearish_pct=float(bear_moves.mean()) if len(bear_moves) else None,
        confidence_vs_accuracy=conf_bins,
        model_types=sorted(df["model_type"].dropna().unique().tolist()),
    )


# ---------------------------------------------------------------------- #
# Full report
# ---------------------------------------------------------------------- #


def build_report(resolved: pd.DataFrame) -> dict:
    """Per-pair reports plus best/worst symbol and timeframe rankings."""
    groups: list[GroupReport] = []
    for (symbol, timeframe), group in resolved.groupby(["symbol", "timeframe"]):
        groups.append(evaluate_group(group, str(symbol), str(timeframe)))

    def _ranked(by: str) -> list[tuple[str, float, int]]:
        agg = (
            resolved.groupby(by)["prediction_correct"]
            .agg(["mean", "count"])
            .query(f"count >= {MIN_SAMPLE_FOR_RANKING}")
        )
        return [
            (str(idx), float(row["mean"] * 100.0), int(row["count"]))
            for idx, row in agg.sort_values("mean", ascending=False).iterrows()
        ]

    tf_ranking = _ranked("timeframe")
    sym_ranking = _ranked("symbol")
    return {
        "groups": groups,
        "overall_accuracy_pct": (
            float(resolved["prediction_correct"].fillna(0).mean() * 100.0)
            if len(resolved)
            else 0.0
        ),
        "total_resolved": len(resolved),
        "best_timeframe": tf_ranking[0] if tf_ranking else None,
        "worst_timeframe": tf_ranking[-1] if tf_ranking else None,
        "best_symbol": sym_ranking[0] if sym_ranking else None,
        "worst_symbol": sym_ranking[-1] if sym_ranking else None,
    }


def format_report(report: dict) -> str:
    """Human-readable multi-section performance report."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("PREDICTION PERFORMANCE REPORT")
    lines.append("=" * 72)
    lines.append(
        f"Resolved predictions: {report['total_resolved']} | "
        f"Overall accuracy: {report['overall_accuracy_pct']:.1f}%"
    )

    for g in report["groups"]:
        lines.append("")
        lines.append(f"--- {g.symbol} {g.timeframe} "
                     f"(models: {', '.join(g.model_types) or 'n/a'}) ---")
        lines.append(
            f"predictions: {g.total_predictions} | accuracy: {g.accuracy_pct:.1f}%"
        )
        prec = " | ".join(
            f"{cls}: {g.precision[cls]:.1f}%"
            if not np.isnan(g.precision[cls])
            else f"{cls}: n/a"
            for cls in DIRECTIONS
        )
        rec = " | ".join(
            f"{cls}: {g.recall[cls]:.1f}%"
            if not np.isnan(g.recall[cls])
            else f"{cls}: n/a"
            for cls in DIRECTIONS
        )
        lines.append(f"precision -> {prec}")
        lines.append(f"recall    -> {rec}")
        if g.avg_move_after_bullish_pct is not None:
            lines.append(
                f"avg actual move after bullish prediction: "
                f"{g.avg_move_after_bullish_pct:+.3f}%"
            )
        if g.avg_move_after_bearish_pct is not None:
            lines.append(
                f"avg actual move after bearish prediction: "
                f"{g.avg_move_after_bearish_pct:+.3f}%"
            )
        lines.append("confusion matrix (rows=predicted, cols=actual):")
        lines.append(g.confusion.to_string())
        if g.confidence_vs_accuracy:
            lines.append("confidence vs accuracy:")
            for label in ("low", "medium", "high"):
                stats = g.confidence_vs_accuracy.get(label)
                if stats:
                    lines.append(
                        f"  {label:<6} n={int(stats['count']):>4} "
                        f"accuracy={stats['accuracy_pct']:.1f}% "
                        f"(avg confidence {stats['avg_confidence']:.0f}%)"
                    )

    lines.append("")
    lines.append("--- RANKINGS (min %d resolved predictions) ---" % MIN_SAMPLE_FOR_RANKING)
    for label, entry in (
        ("best timeframe", report["best_timeframe"]),
        ("worst timeframe", report["worst_timeframe"]),
        ("best symbol", report["best_symbol"]),
        ("worst symbol", report["worst_symbol"]),
    ):
        if entry:
            name, acc, count = entry
            lines.append(f"{label:<16}: {name} ({acc:.1f}% over {count})")
        else:
            lines.append(f"{label:<16}: not enough data")
    lines.append("=" * 72)
    return "\n".join(lines)


def report_from_storage(storage: PredictionStorage) -> str:
    resolved = load_resolved(storage)
    if resolved.empty:
        return "No resolved predictions yet - run the live predictor or a backtest first."
    return format_report(build_report(resolved))
