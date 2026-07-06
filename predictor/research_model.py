"""Enhanced, *fittable* next-candle model used by the research pipeline.

Design goals (see the audit): the baseline rule model is hand-weighted and its
softmax "confidence" is anti-calibrated. This model instead:

* learns weights from data with a balanced **logistic regression** (already a
  project dependency; light and naturally better-calibrated than a softmax over
  hand-picked votes),
* is trained/evaluated **chronologically** (no shuffle, no leakage) — the
  backtest fits it walk-forward and only ever predicts unseen rows,
* is **regime-aware**: in choppy / abnormally volatile conditions confidence is
  penalised,
* supports an optional **NO-SIGNAL / WAIT** mode: when the edge or confidence is
  too small it declines to make a directional call instead of forcing a weak one.

It is prediction-only. It never trades and uses only the leakage-safe features
from ``feature_engineering`` plus the ``regime`` label.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest_metrics import WAIT
from .logger import get_logger
from .model_training import build_estimator
from .regime import RegimeThresholds, add_regime
from .utils import BEARISH, BULLISH, NEUTRAL

log = get_logger("research_model")

MODEL_VERSION = "enhanced-logreg-v1"
_CLASSES = [BEARISH, NEUTRAL, BULLISH]


@dataclass
class EnhancedConfig:
    """Behaviour knobs for the enhanced model (populated from Config)."""

    enhanced_model_type: str = "logistic_regression"
    # Probability calibration (fixes the anti-calibrated softmax confidence).
    calibrate: bool = True
    calibration_method: str = "isotonic"  # or "sigmoid"
    # Regime filtering
    regime_filter_enabled: bool = True
    choppy_confidence_penalty: float = 0.10   # subtracted from confidence in chop
    high_vol_confidence_penalty: float = 0.08
    min_trend_strength_for_strong_signal: float = 0.35
    # NO-SIGNAL / WAIT mode
    no_signal_mode: bool = False
    min_confidence_for_signal: float = 0.55
    min_signal_edge: float = 0.05             # |p_up - p_down| floor
    # Signal-strength thresholds (on adjusted confidence)
    medium_confidence: float = 0.55
    strong_confidence: float = 0.65
    regime_thresholds: RegimeThresholds = field(default_factory=RegimeThresholds)


def _signal_strength(conf: float, cfg: EnhancedConfig) -> str:
    if conf >= cfg.strong_confidence:
        return "strong"
    if conf >= cfg.medium_confidence:
        return "medium"
    return "weak"


class EnhancedModel:
    """A fitted per-(symbol,timeframe) model. Use :meth:`fit` then
    :meth:`predict_frame` (backtest) or :meth:`predict_row` (live)."""

    def __init__(self, feature_cols: list[str], cfg: EnhancedConfig | None = None) -> None:
        self.feature_cols = list(feature_cols)
        self.cfg = cfg or EnhancedConfig()
        self._estimator = None
        self._classes: list[str] = list(_CLASSES)

    # ------------------------------------------------------------------ #

    def fit(self, frame: pd.DataFrame, labels: pd.Series) -> "EnhancedModel":
        """Fit on already-prepared feature rows (labels must align with frame).

        When calibration is enabled and there is enough data, the base estimator
        is fit on the first 80% of the (chronological) training window and its
        probabilities are calibrated on the last 20% — a held-out slice that is
        still strictly before any test row, so no leakage is introduced.
        """
        x = frame.reindex(columns=self.feature_cols).fillna(0.0).to_numpy(dtype=float)
        y = labels.to_numpy()
        self._classes = sorted(set(y))
        base = build_estimator(self.cfg.enhanced_model_type)

        if self.cfg.calibrate and self._can_calibrate(y):
            from sklearn.calibration import CalibratedClassifierCV

            cut = int(len(x) * 0.8)
            base.fit(x[:cut], y[:cut])
            try:
                try:  # sklearn >= 1.6: prefit via FrozenEstimator
                    from sklearn.frozen import FrozenEstimator

                    calibrated = CalibratedClassifierCV(
                        FrozenEstimator(base), method=self.cfg.calibration_method
                    )
                except ImportError:  # sklearn < 1.6: legacy cv="prefit"
                    calibrated = CalibratedClassifierCV(
                        base, method=self.cfg.calibration_method, cv="prefit"
                    )
                calibrated.fit(x[cut:], y[cut:])
                self._estimator = calibrated
                return self
            except Exception as exc:  # fall back to the uncalibrated fit
                log.warning("calibration failed (%s); using uncalibrated model", exc)

        base.fit(x, y)
        self._estimator = base
        return self

    def _can_calibrate(self, y: np.ndarray) -> bool:
        """Need a big enough calibration slice with every class represented."""
        cut = int(len(y) * 0.8)
        calib = y[cut:]
        if len(calib) < 200:
            return False
        counts = pd.Series(calib).value_counts()
        return all(counts.get(c, 0) >= 10 for c in set(y))

    # ------------------------------------------------------------------ #

    def _proba_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = frame.reindex(columns=self.feature_cols).fillna(0.0).to_numpy(dtype=float)
        raw = self._estimator.predict_proba(x)
        present = list(getattr(self._estimator, "classes_", self._classes))
        out = pd.DataFrame(0.0, index=frame.index, columns=_CLASSES)
        for i, cls in enumerate(present):
            if str(cls) in out.columns:
                out[str(cls)] = raw[:, i]
        return out

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Vectorised prediction for a feature frame. Returns columns:
        predicted_direction, bullish/bearish/neutral_probability, confidence
        (regime-adjusted), signal_strength, market_regime, meets_min_confidence.
        """
        if self._estimator is None:
            raise RuntimeError("EnhancedModel.predict_frame called before fit()")
        withreg = add_regime(frame, self.cfg.regime_thresholds)
        proba = self._proba_frame(frame)
        p_bear = proba[BEARISH].to_numpy()
        p_neut = proba[NEUTRAL].to_numpy()
        p_bull = proba[BULLISH].to_numpy()

        raw_dir = proba[_CLASSES].to_numpy().argmax(axis=1)
        direction = np.array(_CLASSES)[raw_dir]
        raw_conf = proba[_CLASSES].to_numpy().max(axis=1)
        edge = np.abs(p_bull - p_bear)

        cfg = self.cfg
        penalty = np.zeros(len(frame))
        if cfg.regime_filter_enabled:
            penalty += withreg["is_choppy"].to_numpy() * cfg.choppy_confidence_penalty
            penalty += withreg["is_high_vol"].to_numpy() * cfg.high_vol_confidence_penalty
        adj_conf = np.clip(raw_conf - penalty, 0.0, 1.0)

        meets = adj_conf >= cfg.min_confidence_for_signal
        final_dir = direction.copy()
        if cfg.no_signal_mode:
            weak = (adj_conf < cfg.min_confidence_for_signal) | (edge < cfg.min_signal_edge)
            final_dir = np.where(weak, WAIT, direction)

        strengths = [_signal_strength(float(c), cfg) for c in adj_conf]
        out = pd.DataFrame(index=frame.index)
        out["predicted_direction"] = final_dir
        out["bullish_probability"] = p_bull
        out["bearish_probability"] = p_bear
        out["neutral_probability"] = p_neut
        out["confidence"] = adj_conf
        out["raw_confidence"] = raw_conf
        out["edge"] = edge
        out["signal_strength"] = strengths
        out["market_regime"] = withreg["regime"].to_numpy()
        out["trend_strength"] = withreg["trend_strength"].to_numpy()
        out["meets_min_confidence"] = meets
        return out

    def predict_row(self, row: pd.Series) -> dict:
        """Single-row prediction for live use. Returns a dict shaped like the
        rule/ML predictors plus regime/strength fields and an explanation."""
        res = self.predict_frame(row.to_frame().T).iloc[0]
        direction = str(res["predicted_direction"])
        regime = str(res["market_regime"])
        strength = str(res["signal_strength"])
        conf = float(res["confidence"])
        edge = float(res["edge"])
        if direction == WAIT:
            explanation = (
                f"No clear edge: {strength} signal in a {regime.lower()} market "
                f"(confidence {conf * 100:.0f}%, edge {edge * 100:.1f}%). "
                "Waiting for a stronger setup."
            )
        else:
            explanation = (
                f"{MODEL_VERSION} predicts {direction} at {conf * 100:.0f}% "
                f"({strength} signal, {regime.lower()} regime, "
                f"edge {edge * 100:.1f}%)."
            )
        return {
            "direction": direction,
            "bullish_probability": float(res["bullish_probability"]),
            "bearish_probability": float(res["bearish_probability"]),
            "neutral_probability": float(res["neutral_probability"]),
            "confidence_score": conf,
            "signal_strength": strength,
            "market_regime": regime,
            "trend_strength": float(res["trend_strength"]),
            "model_version": MODEL_VERSION,
            "explanation": explanation,
        }


def config_to_enhanced(config) -> EnhancedConfig:
    """Build an EnhancedConfig from the main Config (safe attribute lookups)."""
    def g(name, default):
        return getattr(config, name, default)

    return EnhancedConfig(
        enhanced_model_type=g("enhanced_model_type", "logistic_regression"),
        regime_filter_enabled=g("market_regime_filter_enabled", True),
        choppy_confidence_penalty=g("choppy_market_confidence_penalty", 0.10),
        high_vol_confidence_penalty=g("high_volatility_confidence_penalty", 0.08),
        min_trend_strength_for_strong_signal=g("min_trend_strength_for_strong_signal", 0.35),
        no_signal_mode=g("enable_no_signal_mode", False),
        min_confidence_for_signal=g("min_confidence_for_signal", 0.55),
        min_signal_edge=g("min_signal_edge", 0.05),
        regime_thresholds=RegimeThresholds(
            high_vol_mult=g("regime_high_vol_mult", 1.6),
            low_vol_mult=g("regime_low_vol_mult", 0.6),
        ),
    )
