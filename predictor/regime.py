"""Market-regime detection from leakage-safe candle features.

A *regime* is a coarse label for current market conditions (trending, choppy,
high/low volatility, breakout, ...). It is derived entirely from features that
``feature_engineering.compute_features`` already computes on **closed** candles,
so it introduces no look-ahead: the regime at row *i* uses only candles 0..i.

Regimes are used three ways:
* as a bucket in backtests ("accuracy by market regime"),
* as a confidence gate in the enhanced model (choppy/high-vol -> lower
  confidence, optionally WAIT),
* as a plain-English tag on the dashboard.

Nothing here places trades or fetches private data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Canonical regime labels.
TREND_UP = "TREND_UP"
TREND_DOWN = "TREND_DOWN"
SIDEWAYS = "SIDEWAYS"
CHOPPY = "CHOPPY"
HIGH_VOLATILITY = "HIGH_VOLATILITY"
LOW_VOLATILITY = "LOW_VOLATILITY"
BREAKOUT = "BREAKOUT"
UNCLEAR = "UNCLEAR"

REGIMES: list[str] = [
    TREND_UP,
    TREND_DOWN,
    SIDEWAYS,
    CHOPPY,
    HIGH_VOLATILITY,
    LOW_VOLATILITY,
    BREAKOUT,
    UNCLEAR,
]


@dataclass(frozen=True)
class RegimeThresholds:
    """Tunable cut points for regime classification (all overridable via config)."""

    high_vol_mult: float = 1.6   # vol_regime >= this -> abnormally high volatility
    low_vol_mult: float = 0.6    # vol_regime <= this -> abnormally low volatility
    breakout_vol_mult: float = 1.1  # min volatility expansion to call a breakout
    min_trend_strength: float = 0.30  # below this, a directionless market is choppy


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def compute_trend_strength(df: pd.DataFrame) -> pd.Series:
    """A 0..1 measure of how separated the fast/slow EMAs are vs candle noise.

    Uses ``ema9_21_spread`` (percent of price) relative to ``atr_pct`` (typical
    per-candle range). When the EMA stack is not aligned the score is damped, so
    a large spread in a non-trending stack does not read as a strong trend.
    """
    spread = _num(df.get("ema9_21_spread", pd.Series(index=df.index, dtype=float))).abs()
    atr = _num(df.get("atr_pct", pd.Series(index=df.index, dtype=float)))
    noise = atr.clip(lower=1e-6)
    rel = (spread / noise).clip(lower=0.0, upper=1.0)
    stack = _num(df.get("trend_regime", pd.Series(0.0, index=df.index))).fillna(0.0)
    # Damp the score when EMA9/21/50 are not stacked in one direction.
    return np.where(stack != 0.0, rel, rel * 0.3)


def add_regime(
    df: pd.DataFrame, thresholds: RegimeThresholds | None = None
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``regime``, ``trend_strength``,
    ``is_choppy`` and ``is_high_vol`` columns added.

    Expects the feature columns from ``compute_features`` to be present
    (``trend_regime``, ``vol_regime``, ``atr_pct``, ``ema9_21_spread``,
    ``ema50_slope``, ``roll_high_dist``, ``roll_low_dist``).
    """
    t = thresholds or RegimeThresholds()
    out = df.copy()

    stack = _num(out.get("trend_regime", pd.Series(0.0, index=out.index))).fillna(0.0)
    vol = _num(out.get("vol_regime", pd.Series(1.0, index=out.index))).fillna(1.0)
    slope50 = _num(out.get("ema50_slope", pd.Series(0.0, index=out.index))).fillna(0.0)
    high_dist = _num(out.get("roll_high_dist", pd.Series(np.nan, index=out.index)))
    low_dist = _num(out.get("roll_low_dist", pd.Series(np.nan, index=out.index)))

    strength = pd.Series(compute_trend_strength(out), index=out.index)
    out["trend_strength"] = strength

    is_high_vol = vol >= t.high_vol_mult
    is_low_vol = vol <= t.low_vol_mult
    broke_high = (high_dist > 0) & (vol >= t.breakout_vol_mult)
    broke_low = (low_dist < 0) & (vol >= t.breakout_vol_mult)
    trend_up = (stack == 1.0) & (slope50 > 0)
    trend_down = (stack == -1.0) & (slope50 < 0)
    # Directionless but not calm, with a weak trend score -> whipsaw / chop.
    is_choppy = (stack == 0.0) & (~is_low_vol) & (strength < t.min_trend_strength)

    # Priority order: breakout > extreme volatility > clear trend > calm/chop.
    conditions = [
        broke_high | broke_low,
        is_high_vol,
        trend_up,
        trend_down,
        is_low_vol,
        is_choppy,
        stack == 0.0,  # directionless, moderate vol, but not flagged choppy
    ]
    choices = [
        BREAKOUT,
        HIGH_VOLATILITY,
        TREND_UP,
        TREND_DOWN,
        LOW_VOLATILITY,
        CHOPPY,
        SIDEWAYS,
    ]
    out["regime"] = np.select(conditions, choices, default=UNCLEAR)
    out["is_choppy"] = is_choppy | (out["regime"] == CHOPPY)
    out["is_high_vol"] = is_high_vol
    return out


def classify_row(row: pd.Series, thresholds: RegimeThresholds | None = None) -> dict:
    """Classify a single feature row (used live). Returns a small dict with
    ``regime``, ``trend_strength``, ``is_choppy``, ``is_high_vol``."""
    frame = row.to_frame().T
    result = add_regime(frame, thresholds).iloc[0]
    return {
        "regime": str(result["regime"]),
        "trend_strength": float(result["trend_strength"]),
        "is_choppy": bool(result["is_choppy"]),
        "is_high_vol": bool(result["is_high_vol"]),
    }


def regime_label(regime: str) -> str:
    """Human-friendly phrase for a regime code (for the dashboard)."""
    return {
        TREND_UP: "up-trend",
        TREND_DOWN: "down-trend",
        SIDEWAYS: "sideways",
        CHOPPY: "choppy",
        HIGH_VOLATILITY: "high volatility",
        LOW_VOLATILITY: "low volatility",
        BREAKOUT: "breakout",
        UNCLEAR: "unclear",
    }.get(regime, regime.lower())
