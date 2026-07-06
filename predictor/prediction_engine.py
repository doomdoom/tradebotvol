"""Prediction orchestration: data -> features -> predictor -> PredictionResult."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime

import pandas as pd

from .binance_data import BinanceDataClient
from .config import Config
from .feature_engineering import build_feature_frame
from .logger import get_logger
from .ml_predictor import MLPredictor
from .regime import classify_row
from .research_model import EnhancedModel, config_to_enhanced
from .rule_based_predictor import RuleBasedPredictor
from .utils import (
    BEARISH,
    BULLISH,
    confidence_label,
    higher_timeframes,
    ms_to_datetime,
    timeframe_ms,
)


def _strength_from_confidence(conf: float) -> str:
    if conf >= 0.65:
        return "strong"
    if conf >= 0.55:
        return "medium"
    return "weak"

log = get_logger("prediction_engine")

#: How many higher-TF candles to fetch for context features.
_HTF_FETCH_LIMIT = 300


@dataclass
class PredictionResult:
    """One next-candle prediction, ready for display and storage."""

    symbol: str
    timeframe: str
    prediction_time: datetime  # close time of the last observed candle
    target_candle_time: datetime  # open time of the candle being predicted
    target_close_time_ms: int  # when the predicted candle will close
    predicted_direction: str
    bullish_probability: float
    bearish_probability: float
    neutral_probability: float
    confidence: float
    confidence_label: str
    expected_move_min_pct: float
    expected_move_max_pct: float
    reference_close: float  # close used as the return baseline
    model_type: str
    explanation: str
    meets_min_confidence: bool = True
    # Research annotations (Phase 5/10) — present for every mode; the enhanced
    # model sets them directly, other modes derive regime from the feature row.
    market_regime: str = ""
    signal_strength: str = ""
    model_version: str = "baseline"
    trend_strength: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class PredictionEngine:
    """Builds features from live candles and runs the configured predictor."""

    def __init__(self, config: Config, client: BinanceDataClient) -> None:
        self.config = config
        self.client = client
        self.rule_predictor = RuleBasedPredictor(config.neutral_threshold_pct)
        self._ml_predictors: dict[tuple[str, str], MLPredictor] = {}
        self._enhanced: dict[tuple[str, str], EnhancedModel] = {}

    # ------------------------------------------------------------------ #

    def load_ml_models(self, pairs: list[tuple[str, str]] | None = None) -> None:
        """Load model bundles for all pairs; raises with a clear message if missing."""
        for symbol, timeframe in pairs or self.config.pairs:
            path = self.config.model_path(symbol, timeframe)
            self._ml_predictors[(symbol, timeframe)] = MLPredictor.from_file(path)
            log.info("Loaded ML model for %s %s from %s", symbol, timeframe, path)

    def set_ml_predictor(self, symbol: str, timeframe: str, predictor: MLPredictor) -> None:
        self._ml_predictors[(symbol, timeframe)] = predictor

    def ensure_enhanced_models(self, pairs: list[tuple[str, str]] | None = None) -> None:
        """Train the enhanced (fitted, calibrated) model for each pair in memory.

        Downloads historical candles once at startup and fits a per-pair model.
        Only called when the enhanced model is selected; other modes are
        untouched. Trains on closed candles only (no leakage).
        """
        from .model_training import prepare_training_frame

        ecfg = config_to_enhanced(self.config)
        for symbol, timeframe in pairs or self.config.pairs:
            try:
                frame, feature_cols = prepare_training_frame(
                    self.config, self.client, symbol, timeframe,
                    self.config.enhanced_train_candles,
                )
                usable = frame.dropna(subset=[*feature_cols, "label"])
                if len(usable) < 300:
                    log.warning("%s %s: too few rows (%d) to fit enhanced model",
                                symbol, timeframe, len(usable))
                    continue
                model = EnhancedModel(feature_cols, ecfg).fit(usable, usable["label"])
                self._enhanced[(symbol, timeframe)] = model
                log.info("%s %s: enhanced model fitted on %d rows",
                         symbol, timeframe, len(usable))
            except Exception as exc:
                log.error("%s %s: enhanced model training failed: %s",
                          symbol, timeframe, exc)

    # ------------------------------------------------------------------ #

    def fetch_candles(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Latest closed candles for a pair."""
        return self.client.get_klines(
            symbol, timeframe, limit=self.config.candle_limit, only_closed=True
        )

    def build_features(
        self, symbol: str, timeframe: str, candles: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        htf_frames: dict[str, pd.DataFrame] | None = None
        if self.config.use_higher_timeframe_context:
            htf_frames = {}
            for htf in higher_timeframes(timeframe, self.config.higher_timeframe_count):
                htf_frames[htf] = self.client.get_klines(
                    symbol, htf, limit=_HTF_FETCH_LIMIT, only_closed=True
                )
        return build_feature_frame(candles, htf_frames)

    # ------------------------------------------------------------------ #

    def predict(
        self, symbol: str, timeframe: str, candles: pd.DataFrame | None = None
    ) -> PredictionResult:
        """Predict the next candle from the most recent closed candle."""
        if candles is None:
            candles = self.fetch_candles(symbol, timeframe)
        if len(candles) < 120:
            raise RuntimeError(
                f"{symbol} {timeframe}: got only {len(candles)} candles - "
                "not enough for indicator warmup."
            )
        frame, _ = self.build_features(symbol, timeframe, candles)
        row = frame.iloc[-1]

        if self.config.use_enhanced:
            model = self._enhanced.get((symbol, timeframe))
            if model is None:
                self.ensure_enhanced_models([(symbol, timeframe)])
                model = self._enhanced.get((symbol, timeframe))
            if model is None:
                raise RuntimeError(
                    f"{symbol} {timeframe}: enhanced model unavailable "
                    "(training failed - check logs)."
                )
            raw = model.predict_row(row)
            model_type = "enhanced"
        elif self.config.prediction_mode == "ml":
            predictor = self._ml_predictors.get((symbol, timeframe))
            if predictor is None:
                self.load_ml_models([(symbol, timeframe)])
                predictor = self._ml_predictors[(symbol, timeframe)]
            raw = predictor.predict_row(row)
            model_type = predictor.model_type
        else:
            rule = self.rule_predictor.predict(row)
            raw = {
                "direction": rule.direction,
                "bullish_probability": rule.bullish_probability,
                "bearish_probability": rule.bearish_probability,
                "neutral_probability": rule.neutral_probability,
                "confidence_score": rule.confidence_score,
                "explanation": rule.explanation,
            }
            model_type = "rule_based"

        return self._to_result(symbol, timeframe, row, raw, model_type)

    # ------------------------------------------------------------------ #

    def _to_result(
        self,
        symbol: str,
        timeframe: str,
        row: pd.Series,
        raw: dict,
        model_type: str,
    ) -> PredictionResult:
        step = timeframe_ms(timeframe)
        last_open_ms = int(row["open_time_ms"])
        target_open_ms = last_open_ms + step
        direction = str(raw["direction"])
        confidence = float(raw["confidence_score"])
        meets = confidence >= self.config.min_confidence
        explanation = str(raw["explanation"])
        if not meets:
            explanation += (
                f" (Confidence {confidence * 100:.0f}% is below the configured "
                f"minimum of {self.config.min_confidence * 100:.0f}% - treat as low conviction.)"
            )

        # Regime annotation for every mode; enhanced supplies its own, others
        # derive it from the same feature row (display-only, no effect on the
        # predicted direction or confidence).
        regime = str(raw.get("market_regime") or "")
        trend_strength = raw.get("trend_strength")
        if not regime or trend_strength is None:
            info = classify_row(row)
            regime = regime or info["regime"]
            trend_strength = info["trend_strength"] if trend_strength is None else trend_strength
        signal_strength = str(raw.get("signal_strength") or _strength_from_confidence(confidence))
        model_version = str(raw.get("model_version") or model_type)

        move_min, move_max = self._expected_range(direction, row)
        return PredictionResult(
            symbol=symbol,
            timeframe=timeframe,
            prediction_time=ms_to_datetime(int(row["close_time_ms"])),
            target_candle_time=ms_to_datetime(target_open_ms),
            target_close_time_ms=target_open_ms + step - 1,
            predicted_direction=direction,
            bullish_probability=float(raw["bullish_probability"]),
            bearish_probability=float(raw["bearish_probability"]),
            neutral_probability=float(raw["neutral_probability"]),
            confidence=confidence,
            confidence_label=confidence_label(confidence),
            expected_move_min_pct=move_min,
            expected_move_max_pct=move_max,
            reference_close=float(row["close"]),
            model_type=model_type,
            explanation=explanation,
            meets_min_confidence=meets,
            market_regime=regime,
            signal_strength=signal_strength,
            model_version=model_version,
            trend_strength=float(trend_strength or 0.0),
        )

    @staticmethod
    def _expected_range(direction: str, row: pd.Series) -> tuple[float, float]:
        """ATR-scaled expected close-to-close move for the predicted candle."""
        atr_pct = row.get("atr_pct")
        atr_pct = float(atr_pct) if atr_pct is not None and not pd.isna(atr_pct) else 0.1
        if direction == BULLISH:
            return round(0.30 * atr_pct, 4), round(0.85 * atr_pct, 4)
        if direction == BEARISH:
            return round(-0.85 * atr_pct, 4), round(-0.30 * atr_pct, 4)
        return round(-0.25 * atr_pct, 4), round(0.25 * atr_pct, 4)
