"""Feature engineering from OHLCV candles.

Look-ahead safety: every feature at row *i* is computed only from candles
0..i. Higher-timeframe context is merged on candle *close* times, so a
higher-TF candle is only visible after it has fully closed. Labels (which
do look one candle ahead by definition) live in separate columns that are
added only for training/backtesting and are never part of the feature set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ta
from .utils import BEARISH, BULLISH, NEUTRAL

#: Features computed on the prediction timeframe itself.
BASE_FEATURES: list[str] = [
    # returns
    "ret_1", "log_ret_1", "ret_3", "ret_5", "ret_10",
    # candle anatomy
    "body_pct", "upper_wick_pct", "lower_wick_pct", "range_pct", "close_position",
    # volume
    "vol_change_pct", "vol_spike_ratio",
    # EMAs
    "ema9_dist", "ema21_dist", "ema50_dist",
    "ema9_slope", "ema21_slope", "ema50_slope", "ema9_21_spread",
    # oscillators
    "rsi", "rsi_slope",
    "macd_norm", "macd_hist_norm", "macd_hist_change",
    # bands / volatility
    "bb_width", "bb_pct", "atr_pct",
    # VWAP and structure
    "vwap_dist", "roll_high_dist", "roll_low_dist",
    # streaks and regimes
    "consec_bull", "consec_bear", "vol_regime", "trend_regime",
    # calendar
    "hour_of_day", "day_of_week", "hour_sin", "hour_cos",
]

#: Per higher timeframe, these context columns are added with an
#: ``htf_{tf}_`` prefix.
HTF_CONTEXT_FEATURES: list[str] = ["trend", "rsi", "macd_hist_sign", "ret_1", "vwap_dist"]

LABEL_COLUMNS = ["next_return_pct", "label"]


def compute_features(candles: pd.DataFrame) -> pd.DataFrame:
    """Add all base features to a candle DataFrame (copy is returned).

    Expects columns: open, high, low, close, volume, open_time, close_time.
    """
    df = candles.copy()
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # --- returns (percent units) ---
    df["ret_1"] = c.pct_change(fill_method=None) * 100.0
    df["log_ret_1"] = np.log(c / c.shift(1)) * 100.0
    df["ret_3"] = c.pct_change(3, fill_method=None) * 100.0
    df["ret_5"] = c.pct_change(5, fill_method=None) * 100.0
    df["ret_10"] = c.pct_change(10, fill_method=None) * 100.0

    # --- candle anatomy ---
    candle_range = (h - l).replace(0.0, np.nan)
    df["body_pct"] = (c - o).abs() / candle_range * 100.0
    df["upper_wick_pct"] = (h - np.maximum(o, c)) / candle_range * 100.0
    df["lower_wick_pct"] = (np.minimum(o, c) - l) / candle_range * 100.0
    df["range_pct"] = (h - l) / c * 100.0
    df["close_position"] = (c - l) / candle_range  # 0 = at low, 1 = at high

    # --- volume ---
    df["vol_change_pct"] = v.pct_change(fill_method=None).replace(
        [np.inf, -np.inf], np.nan
    ) * 100.0
    vol_ma = v.rolling(20, min_periods=20).mean().replace(0.0, np.nan)
    df["vol_spike_ratio"] = v / vol_ma

    # --- EMAs ---
    ema9, ema21, ema50 = ta.ema(c, 9), ta.ema(c, 21), ta.ema(c, 50)
    df["ema9_dist"] = (c - ema9) / c * 100.0
    df["ema21_dist"] = (c - ema21) / c * 100.0
    df["ema50_dist"] = (c - ema50) / c * 100.0
    df["ema9_slope"] = ta.slope(ema9, 3) / c * 100.0
    df["ema21_slope"] = ta.slope(ema21, 3) / c * 100.0
    df["ema50_slope"] = ta.slope(ema50, 5) / c * 100.0
    df["ema9_21_spread"] = (ema9 - ema21) / c * 100.0

    # --- RSI ---
    rsi = ta.rsi(c, 14)
    df["rsi"] = rsi
    df["rsi_slope"] = ta.slope(rsi, 3)

    # --- MACD (normalized by price so features transfer across symbols) ---
    macd = ta.macd(c)
    df["macd_norm"] = macd["macd"] / c * 100.0
    df["macd_hist_norm"] = macd["macd_hist"] / c * 100.0
    df["macd_hist_change"] = macd["macd_hist"].diff() / c * 100.0

    # --- Bollinger Bands ---
    bb = ta.bollinger(c, 20, 2.0)
    df["bb_width"] = bb["bb_width"]
    df["bb_pct"] = bb["bb_pct"]

    # --- ATR ---
    atr = ta.atr(h, l, c, 14)
    df["atr_pct"] = atr / c * 100.0

    # --- VWAP ---
    vwap = ta.rolling_vwap(h, l, c, v, 20)
    df["vwap_dist"] = (c - vwap) / c * 100.0

    # --- structure: distance to prior 20-bar extremes (excluding current bar,
    # so a positive roll_high_dist means a genuine breakout) ---
    prior_high = ta.rolling_high(h, 20).shift(1)
    prior_low = ta.rolling_low(l, 20).shift(1)
    df["roll_high_dist"] = (c - prior_high) / c * 100.0
    df["roll_low_dist"] = (c - prior_low) / c * 100.0

    # --- consecutive candle streaks ---
    bull = (c > o).astype(int)
    bear = (c < o).astype(int)
    df["consec_bull"] = _streak(bull)
    df["consec_bear"] = _streak(bear)

    # --- regimes ---
    atr_median = df["atr_pct"].rolling(100, min_periods=30).median()
    df["vol_regime"] = df["atr_pct"] / atr_median.replace(0.0, np.nan)
    df["trend_regime"] = np.select(
        [(ema9 > ema21) & (ema21 > ema50), (ema9 < ema21) & (ema21 < ema50)],
        [1.0, -1.0],
        default=0.0,
    )

    # --- calendar ---
    hours = df["open_time"].dt.hour.astype(float)
    df["hour_of_day"] = hours
    df["day_of_week"] = df["open_time"].dt.dayofweek.astype(float)
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

    return df


def _streak(flags: pd.Series) -> pd.Series:
    """Length of the run of consecutive 1s ending at each row."""
    groups = (flags != flags.shift()).cumsum()
    run = flags.groupby(groups).cumcount() + 1
    return run.where(flags == 1, 0).astype(float)


# ---------------------------------------------------------------------- #
# Higher-timeframe context
# ---------------------------------------------------------------------- #


def compute_htf_context(htf_candles: pd.DataFrame) -> pd.DataFrame:
    """Compact context frame from higher-timeframe candles.

    Returns one row per closed higher-TF candle, keyed by ``close_time_ms``.
    """
    c = htf_candles["close"]
    h, l, v = htf_candles["high"], htf_candles["low"], htf_candles["volume"]
    ema9, ema21 = ta.ema(c, 9), ta.ema(c, 21)
    macd = ta.macd(c)
    vwap = ta.rolling_vwap(h, l, c, v, 20)
    return pd.DataFrame(
        {
            "close_time_ms": htf_candles["close_time_ms"],
            "trend": np.sign(ema9 - ema21),
            "rsi": ta.rsi(c, 14),
            "macd_hist_sign": np.sign(macd["macd_hist"]),
            "ret_1": c.pct_change(fill_method=None) * 100.0,
            "vwap_dist": (c - vwap) / c * 100.0,
        }
    )


def add_htf_context(
    df: pd.DataFrame, htf_frames: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, list[str]]:
    """Merge higher-TF context into a lower-TF feature frame.

    A higher-TF candle is only matched to lower-TF rows whose own close time
    is >= the higher candle's close time (merge_asof backward on close
    times), which guarantees no look-ahead.
    """
    out = df.sort_values("close_time_ms").reset_index(drop=True)
    added: list[str] = []
    for tf, htf_candles in htf_frames.items():
        context = (
            compute_htf_context(htf_candles)
            .dropna(subset=["trend"])
            .sort_values("close_time_ms")
        )
        renamed = context.rename(
            columns={name: f"htf_{tf}_{name}" for name in HTF_CONTEXT_FEATURES}
        )
        out = pd.merge_asof(
            out,
            renamed,
            on="close_time_ms",
            direction="backward",
            allow_exact_matches=True,
        )
        added.extend(f"htf_{tf}_{name}" for name in HTF_CONTEXT_FEATURES)
    return out, added


def build_feature_frame(
    candles: pd.DataFrame, htf_frames: dict[str, pd.DataFrame] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Full feature pipeline; returns (frame, ordered feature column list)."""
    df = compute_features(candles)
    feature_cols = list(BASE_FEATURES)
    if htf_frames:
        df, htf_cols = add_htf_context(df, htf_frames)
        feature_cols += htf_cols
    return df, feature_cols


# ---------------------------------------------------------------------- #
# Labels (training / backtesting only — never used as features)
# ---------------------------------------------------------------------- #


def add_labels(df: pd.DataFrame, neutral_threshold_pct: float) -> pd.DataFrame:
    """Label each row with the direction of the *next* candle's close.

    The last row gets NaN labels (its future is unknown) and must be dropped
    before training.
    """
    out = df.copy()
    next_ret = (out["close"].shift(-1) / out["close"] - 1.0) * 100.0
    out["next_return_pct"] = next_ret
    out["label"] = np.select(
        [next_ret > neutral_threshold_pct, next_ret < -neutral_threshold_pct],
        [BULLISH, BEARISH],
        default=NEUTRAL,
    )
    out.loc[next_ret.isna(), "label"] = np.nan
    return out


def training_rows(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """Drop rows unusable for training (indicator warmup, missing label)."""
    usable = df.dropna(subset=[*feature_cols, "label"])
    return usable[feature_cols], usable["label"]
