"""Rule-based (no-ML) next-candle direction predictor.

Scores a weighted set of technical signals on the latest feature row,
converts bullish/bearish/neutral scores into probabilities via softmax and
produces a human-readable explanation of the strongest signals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .utils import BEARISH, BULLISH, NEUTRAL

MODEL_TYPE = "rule_based"


@dataclass
class RulePrediction:
    direction: str
    bullish_probability: float
    bearish_probability: float
    neutral_probability: float
    confidence_score: float
    explanation: str


def _get(row: pd.Series, name: str) -> float | None:
    value = row.get(name)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value


class RuleBasedPredictor:
    """Weighted-signal voting model. Stateless; safe to share across pairs."""

    def __init__(
        self, neutral_threshold_pct: float = 0.03, temperature: float = 2.2
    ) -> None:
        self.neutral_threshold_pct = neutral_threshold_pct
        self.temperature = temperature

    # ------------------------------------------------------------------ #

    def predict(self, row: pd.Series) -> RulePrediction:
        bull: list[tuple[float, str]] = []  # (weight, reason)
        bear: list[tuple[float, str]] = []
        neutral: list[tuple[float, str]] = []

        self._trend_rules(row, bull, bear)
        self._momentum_rules(row, bull, bear)
        self._vwap_volume_rules(row, bull, bear)
        self._candle_structure_rules(row, bull, bear)
        self._breakout_rules(row, bull, bear)
        self._mean_reversion_rules(row, bull, bear)
        self._neutral_rules(row, neutral)

        bull_score = sum(w for w, _ in bull)
        bear_score = sum(w for w, _ in bear)
        neutral_score = 1.0 + sum(w for w, _ in neutral)
        # Conflicting evidence on both sides raises the odds of chop.
        neutral_score += 0.35 * min(bull_score, bear_score)

        p_bull, p_bear, p_neutral = self._softmax(bull_score, bear_score, neutral_score)
        probs = {BULLISH: p_bull, BEARISH: p_bear, NEUTRAL: p_neutral}
        direction = max(probs, key=probs.get)
        confidence = probs[direction]

        explanation = self._explain(direction, probs, bull, bear, neutral)
        return RulePrediction(
            direction=direction,
            bullish_probability=p_bull,
            bearish_probability=p_bear,
            neutral_probability=p_neutral,
            confidence_score=confidence,
            explanation=explanation,
        )

    # ------------------------------------------------------------------ #
    # Rule groups
    # ------------------------------------------------------------------ #

    def _trend_rules(self, row, bull, bear) -> None:
        trend = _get(row, "trend_regime")
        if trend == 1.0:
            bull.append((1.5, "EMA stack is bullish (EMA9 > EMA21 > EMA50)"))
        elif trend == -1.0:
            bear.append((1.5, "EMA stack is bearish (EMA9 < EMA21 < EMA50)"))
        else:
            spread = _get(row, "ema9_21_spread")
            if spread is not None and spread > 0.01:
                bull.append((0.5, "EMA9 is above EMA21"))
            elif spread is not None and spread < -0.01:
                bear.append((0.5, "EMA9 is below EMA21"))

        slope50 = _get(row, "ema50_slope")
        if slope50 is not None and slope50 > 0:
            bull.append((0.5, "EMA50 slope is rising"))
        elif slope50 is not None and slope50 < 0:
            bear.append((0.5, "EMA50 slope is falling"))

    def _momentum_rules(self, row, bull, bear) -> None:
        rsi = _get(row, "rsi")
        rsi_slope = _get(row, "rsi_slope")
        if rsi is not None and rsi_slope is not None:
            if rsi >= 70:
                bear.append((0.75, f"RSI is overbought at {rsi:.0f}"))
            elif rsi <= 30:
                bull.append((0.75, f"RSI is oversold at {rsi:.0f}"))
            elif rsi > 55 and rsi_slope > 0:
                bull.append((1.0, "RSI is rising above 55"))
            elif rsi < 45 and rsi_slope < 0:
                bear.append((1.0, "RSI is falling below 45"))

        hist = _get(row, "macd_hist_norm")
        hist_change = _get(row, "macd_hist_change")
        if hist is not None and hist_change is not None:
            if hist > 0 and hist_change > 0:
                bull.append((1.0, "MACD histogram is positive and expanding"))
            elif hist < 0 and hist_change < 0:
                bear.append((1.0, "MACD histogram is negative and expanding"))

        ret5 = _get(row, "ret_5")
        atr = _get(row, "atr_pct") or 0.0
        if ret5 is not None and atr > 0:
            if ret5 > 1.5 * atr:
                bull.append((0.4, "strong 5-candle upward momentum"))
            elif ret5 < -1.5 * atr:
                bear.append((0.4, "strong 5-candle downward momentum"))

    def _vwap_volume_rules(self, row, bull, bear) -> None:
        vwap_dist = _get(row, "vwap_dist")
        if vwap_dist is not None:
            if vwap_dist > 0.05:
                bull.append((0.6, "price is above VWAP"))
            elif vwap_dist < -0.05:
                bear.append((0.6, "price is below VWAP"))

        spike = _get(row, "vol_spike_ratio")
        last_ret = _get(row, "ret_1")
        if spike is not None and last_ret is not None and spike >= 2.0:
            if last_ret > 0:
                bull.append((0.9, "volume spike on a bullish candle"))
            elif last_ret < 0:
                bear.append((0.9, "volume spike on a bearish candle"))

    def _candle_structure_rules(self, row, bull, bear) -> None:
        close_pos = _get(row, "close_position")
        if close_pos is not None:
            if close_pos >= 0.75:
                bull.append((0.5, "the last candle closed near its high"))
            elif close_pos <= 0.25:
                bear.append((0.5, "the last candle closed near its low"))

        body = _get(row, "body_pct")
        last_ret = _get(row, "ret_1")
        if body is not None and last_ret is not None and body >= 60:
            if last_ret > 0:
                bull.append((0.4, "strong-bodied bullish candle"))
            elif last_ret < 0:
                bear.append((0.4, "strong-bodied bearish candle"))

        lower_wick = _get(row, "lower_wick_pct")
        upper_wick = _get(row, "upper_wick_pct")
        if lower_wick is not None and lower_wick >= 60:
            bull.append((0.4, "long lower wick shows buyers defending the low"))
        if upper_wick is not None and upper_wick >= 60:
            bear.append((0.4, "long upper wick shows sellers capping the high"))

    def _breakout_rules(self, row, bull, bear) -> None:
        high_dist = _get(row, "roll_high_dist")
        low_dist = _get(row, "roll_low_dist")
        if high_dist is not None and high_dist > 0:
            bull.append((1.0, "close broke above the recent 20-candle high"))
        if low_dist is not None and low_dist < 0:
            bear.append((1.0, "close broke below the recent 20-candle low"))

    def _mean_reversion_rules(self, row, bull, bear) -> None:
        bb_pct = _get(row, "bb_pct")
        if bb_pct is not None:
            if bb_pct >= 0.98:
                bear.append((0.5, "price is stretched above the upper Bollinger Band"))
            elif bb_pct <= 0.02:
                bull.append((0.5, "price is stretched below the lower Bollinger Band"))

        consec_bull = _get(row, "consec_bull") or 0
        consec_bear = _get(row, "consec_bear") or 0
        if consec_bull >= 5:
            bear.append((0.5, f"{int(consec_bull)} consecutive bullish candles look extended"))
        elif 3 <= consec_bull <= 4:
            bull.append((0.25, f"{int(consec_bull)} consecutive bullish candles in a row"))
        if consec_bear >= 5:
            bull.append((0.5, f"{int(consec_bear)} consecutive bearish candles look extended"))
        elif 3 <= consec_bear <= 4:
            bear.append((0.25, f"{int(consec_bear)} consecutive bearish candles in a row"))

        # Higher-timeframe context (present only when enabled in config).
        for name in row.index:
            if name.startswith("htf_") and name.endswith("_trend"):
                tf = name.split("_")[1]
                value = _get(row, name)
                if value == 1.0:
                    bull.append((0.6, f"{tf} higher-timeframe trend is bullish"))
                elif value == -1.0:
                    bear.append((0.6, f"{tf} higher-timeframe trend is bearish"))

    def _neutral_rules(self, row, neutral) -> None:
        vol_regime = _get(row, "vol_regime")
        if vol_regime is not None and vol_regime < 0.75:
            neutral.append((0.8, "volatility is well below its recent norm"))
        atr = _get(row, "atr_pct")
        if atr is not None and atr < 2.0 * self.neutral_threshold_pct:
            neutral.append(
                (0.6, "the average candle range barely exceeds the neutral threshold")
            )
        bb_width = _get(row, "bb_width")
        if bb_width is not None and atr is not None and bb_width < 3.0 * atr:
            neutral.append((0.3, "Bollinger Bands are squeezed"))

    # ------------------------------------------------------------------ #

    def _softmax(self, bull: float, bear: float, neutral: float) -> tuple[float, float, float]:
        t = self.temperature
        exps = [math.exp(bull / t), math.exp(bear / t), math.exp(neutral / t)]
        total = sum(exps)
        return exps[0] / total, exps[1] / total, exps[2] / total

    @staticmethod
    def _explain(direction, probs, bull, bear, neutral) -> str:
        by_side = {BULLISH: bull, BEARISH: bear, NEUTRAL: neutral}
        winners = sorted(by_side[direction], key=lambda x: -x[0])[:5]
        if direction == NEUTRAL and not winners:
            return (
                "Neutral: bullish and bearish signals are balanced and no strong "
                "directional evidence is present."
            )
        reasons = ", ".join(reason for _, reason in winners) or "weak mixed signals"
        opposite = BEARISH if direction == BULLISH else BULLISH
        counter = sorted(by_side.get(opposite, []), key=lambda x: -x[0])[:1]
        text = (
            f"{direction.capitalize()} probability is highest "
            f"({probs[direction] * 100:.0f}%) because {reasons}."
        )
        if counter:
            text += f" Main counter-signal: {counter[0][1]}."
        return text
