"""ML inference: load a trained model bundle and predict class probabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .logger import get_logger
from .utils import BEARISH, BULLISH, NEUTRAL

log = get_logger("ml_predictor")


@dataclass
class ModelBundle:
    """Everything needed to reproduce a model's predictions at inference time."""

    model: Any
    feature_cols: list[str]
    classes: list[str]
    model_type: str
    symbol: str
    timeframe: str
    market_type: str
    neutral_threshold_pct: float
    trained_at: str
    train_end_time_ms: int
    n_samples: int
    holdout_accuracy: float
    walk_forward_scores: list[float] = field(default_factory=list)


def save_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    log.info("Saved model bundle to %s", path)
    return path


def load_bundle(path: str | Path) -> ModelBundle:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. Train it first with "
            f"'py train_model.py' (see README)."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise TypeError(f"{path} does not contain a ModelBundle")
    return bundle


class MLPredictor:
    """Predicts bullish/bearish/neutral probabilities from a feature row."""

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle

    @classmethod
    def from_file(cls, path: str | Path) -> "MLPredictor":
        return cls(load_bundle(path))

    @property
    def model_type(self) -> str:
        return self.bundle.model_type

    # ------------------------------------------------------------------ #

    def predict_proba_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        """Class probabilities for each row; columns are bearish/neutral/bullish."""
        x = features.reindex(columns=self.bundle.feature_cols)
        missing = [c for c in self.bundle.feature_cols if c not in features.columns]
        if missing:
            log.warning(
                "Missing %d model features at inference (filled with 0): %s",
                len(missing),
                missing[:5],
            )
        x = x.fillna(0.0).to_numpy(dtype=float)
        raw = self.bundle.model.predict_proba(x)
        out = pd.DataFrame(0.0, index=features.index, columns=[BEARISH, NEUTRAL, BULLISH])
        for i, cls in enumerate(self.bundle.classes):
            if cls in out.columns:
                out[cls] = raw[:, i]
        return out

    def predict_row(self, row: pd.Series) -> dict[str, float | str]:
        probs = self.predict_proba_frame(row.to_frame().T).iloc[0]
        direction = str(probs.idxmax())
        return {
            "direction": direction,
            "bullish_probability": float(probs[BULLISH]),
            "bearish_probability": float(probs[BEARISH]),
            "neutral_probability": float(probs[NEUTRAL]),
            "confidence_score": float(probs.max()),
            "explanation": self._explain(row, direction, probs),
        }

    # ------------------------------------------------------------------ #

    def _explain(self, row: pd.Series, direction: str, probs: pd.Series) -> str:
        parts = [
            f"{self.bundle.model_type} model "
            f"(holdout accuracy {self.bundle.holdout_accuracy * 100:.1f}%) "
            f"predicts {direction} at {probs.max() * 100:.0f}%"
        ]
        drivers = self._top_features()
        if drivers:
            described = []
            for name in drivers:
                value = row.get(name)
                if value is not None and not pd.isna(value):
                    described.append(f"{name}={float(value):.3f}")
            if described:
                parts.append("key inputs: " + ", ".join(described))
        return "; ".join(parts) + "."

    def _top_features(self, n: int = 4) -> list[str]:
        model = self.bundle.model
        estimator = model
        # unwrap sklearn Pipeline
        if hasattr(model, "named_steps"):
            estimator = list(model.named_steps.values())[-1]
        importances = getattr(estimator, "feature_importances_", None)
        if importances is None:
            coef = getattr(estimator, "coef_", None)
            if coef is None:
                return []
            importances = np.abs(np.asarray(coef)).mean(axis=0)
        order = np.argsort(importances)[::-1][:n]
        return [self.bundle.feature_cols[i] for i in order]
