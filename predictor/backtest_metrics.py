"""Reusable evaluation metrics for the research/backtest pipeline.

Everything here is pure pandas/numpy so it can be unit-tested without network
or sklearn. Metrics cover the multi-class direction problem (bearish / neutral
/ bullish) plus optional WAIT (no-signal) rows, which are reported as reduced
*coverage* rather than counted as wrong.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from .utils import BEARISH, BULLISH, DIRECTIONS, NEUTRAL

WAIT = "wait"

# Confidence buckets used for calibration (lower-inclusive, upper-exclusive,
# except the final bucket which is closed).
CONFIDENCE_BINS: list[tuple[float, float]] = [
    (0.50, 0.55),
    (0.55, 0.60),
    (0.60, 0.65),
    (0.65, 0.70),
    (0.70, 1.01),
]


def _safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def classification_metrics(
    y_true: Iterable[str], y_pred: Iterable[str], classes: list[str] | None = None
) -> dict:
    """Accuracy, per-class precision/recall/F1, macro-F1 and confusion matrix.

    Rows whose prediction is WAIT are ignored (they are not a directional call);
    report coverage separately with :func:`coverage`.
    """
    classes = classes or DIRECTIONS
    yt = pd.Series(list(y_true), dtype="object").reset_index(drop=True)
    yp = pd.Series(list(y_pred), dtype="object").reset_index(drop=True)
    mask = yp != WAIT
    yt, yp = yt[mask], yp[mask]
    n = int(len(yt))
    if n == 0:
        return {
            "n": 0,
            "accuracy_pct": float("nan"),
            "precision": {c: float("nan") for c in classes},
            "recall": {c: float("nan") for c in classes},
            "f1": {c: float("nan") for c in classes},
            "macro_f1_pct": float("nan"),
            "confusion": {c: {c2: 0 for c2 in classes} for c in classes},
            "support": {c: 0 for c in classes},
        }

    correct = int((yt.values == yp.values).sum())
    precision: dict[str, float] = {}
    recall: dict[str, float] = {}
    f1: dict[str, float] = {}
    support: dict[str, int] = {}
    for c in classes:
        tp = int(((yp == c) & (yt == c)).sum())
        pred_c = int((yp == c).sum())
        act_c = int((yt == c).sum())
        p = _safe_div(tp, pred_c)
        r = _safe_div(tp, act_c)
        precision[c] = p * 100.0 if not math.isnan(p) else float("nan")
        recall[c] = r * 100.0 if not math.isnan(r) else float("nan")
        if not math.isnan(p) and not math.isnan(r) and (p + r) > 0:
            f1[c] = 2 * p * r / (p + r) * 100.0
        else:
            f1[c] = float("nan")
        support[c] = act_c

    valid_f1 = [f1[c] for c in classes if not math.isnan(f1[c])]
    macro_f1 = float(np.mean(valid_f1)) if valid_f1 else float("nan")

    confusion = {c: {c2: 0 for c2 in classes} for c in classes}
    for a, p in zip(yt.values, yp.values):
        if a in confusion and p in confusion[a]:
            confusion[a][p] += 1  # confusion[actual][predicted]

    return {
        "n": n,
        "accuracy_pct": correct / n * 100.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1_pct": macro_f1,
        "confusion": confusion,
        "support": support,
    }


def coverage(y_pred: Iterable[str]) -> dict:
    """Fraction of rows that produced a directional signal (not WAIT)."""
    yp = pd.Series(list(y_pred), dtype="object")
    total = int(len(yp))
    signalled = int((yp != WAIT).sum())
    return {
        "total": total,
        "signalled": signalled,
        "coverage_pct": _safe_div(signalled, total) * 100.0 if total else float("nan"),
    }


def majority_baseline(y_true: Iterable[str]) -> dict:
    """Accuracy of always predicting the most common actual class.

    This is the number a real model must beat: on next-candle data the majority
    class (often NEUTRAL) can dominate.
    """
    yt = pd.Series(list(y_true), dtype="object")
    if yt.empty:
        return {"class": None, "accuracy_pct": float("nan"), "n": 0}
    counts = yt.value_counts()
    top = str(counts.index[0])
    return {
        "class": top,
        "accuracy_pct": float(counts.iloc[0] / len(yt) * 100.0),
        "n": int(len(yt)),
    }


def accuracy_by(
    df: pd.DataFrame,
    by: str,
    correct_col: str = "prediction_correct",
    pred_col: str = "predicted_direction",
    min_n: int = 1,
) -> dict[str, dict]:
    """Accuracy grouped by a column (symbol, timeframe, regime, hour, ...).

    WAIT predictions are excluded from the accuracy denominator.
    """
    if by not in df.columns or df.empty:
        return {}
    work = df[df[pred_col] != WAIT] if pred_col in df.columns else df
    out: dict[str, dict] = {}
    for key, g in work.groupby(by):
        n = int(len(g))
        if n < min_n:
            continue
        out[str(key)] = {
            "n": n,
            "accuracy_pct": float(g[correct_col].fillna(0).mean() * 100.0),
        }
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def confidence_calibration(
    df: pd.DataFrame,
    conf_col: str = "confidence",
    correct_col: str = "prediction_correct",
    pred_col: str = "predicted_direction",
    bins: list[tuple[float, float]] | None = None,
) -> list[dict]:
    """Per-bucket confidence vs realized accuracy, with the calibration gap.

    ``gap_pct`` = mean confidence - realized accuracy. A well-calibrated model
    has gap ~ 0; a large positive gap means the model is over-confident.
    """
    bins = bins or CONFIDENCE_BINS
    work = df[df[pred_col] != WAIT] if pred_col in df.columns else df
    conf = pd.to_numeric(work[conf_col], errors="coerce")
    out: list[dict] = []
    for lo, hi in bins:
        sel = work[(conf >= lo) & (conf < hi)]
        n = int(len(sel))
        if n == 0:
            out.append(
                {
                    "bucket": f"{int(lo * 100)}-{min(int(hi * 100), 100)}%",
                    "n": 0,
                    "mean_confidence_pct": float("nan"),
                    "accuracy_pct": float("nan"),
                    "gap_pct": float("nan"),
                }
            )
            continue
        mean_conf = float(pd.to_numeric(sel[conf_col], errors="coerce").mean() * 100.0)
        acc = float(sel[correct_col].fillna(0).mean() * 100.0)
        out.append(
            {
                "bucket": f"{int(lo * 100)}-{min(int(hi * 100), 100)}%",
                "n": n,
                "mean_confidence_pct": mean_conf,
                "accuracy_pct": acc,
                "gap_pct": mean_conf - acc,
            }
        )
    return out


def direction_distribution(series: Iterable[str]) -> dict[str, int]:
    """Count of each label (handy for spotting a biased predictor)."""
    counts = pd.Series(list(series), dtype="object").value_counts()
    order = [BEARISH, NEUTRAL, BULLISH, WAIT]
    dist = {k: int(counts.get(k, 0)) for k in order if k in counts.index}
    for k, v in counts.items():  # any unexpected labels
        dist.setdefault(str(k), int(v))
    return dist
