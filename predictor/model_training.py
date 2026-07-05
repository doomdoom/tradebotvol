"""Model training: time-ordered splits, walk-forward validation, persistence.

Time-series hygiene:
* rows are never shuffled;
* the train/test split is strictly chronological;
* walk-forward validation trains on an expanding window and tests on the
  following unseen fold;
* the saved bundle records ``train_end_time_ms`` so backtests can exclude
  in-sample candles.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .binance_data import BinanceDataClient
from .config import Config
from .feature_engineering import add_labels, build_feature_frame, training_rows
from .historical_data import download_history
from .logger import get_logger
from .ml_predictor import ModelBundle, save_bundle
from .utils import higher_timeframes

log = get_logger("model_training")

try:
    from xgboost import XGBClassifier
except ImportError:  # optional
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # optional
    LGBMClassifier = None


def build_estimator(model_type: str):
    """Instantiate an estimator by name. Optional libraries are guarded."""
    if model_type == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000, C=0.5, class_weight="balanced"
                    ),
                ),
            ]
        )
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
    if model_type == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError(
                "model_type 'xgboost' requires the optional 'xgboost' package "
                "(pip install xgboost)."
            )
        return XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42,
        )
    if model_type == "lightgbm":
        if LGBMClassifier is None:
            raise RuntimeError(
                "model_type 'lightgbm' requires the optional 'lightgbm' package "
                "(pip install lightgbm)."
            )
        return LGBMClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            verbose=-1,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


class _LabelWrapper:
    """Maps string labels to ints for estimators that require numeric classes
    (XGBoost) while exposing a uniform predict_proba/classes_ interface."""

    def __init__(self, estimator, classes: list[str]) -> None:
        self._estimator = estimator
        self._classes = classes
        self._index = {c: i for i, c in enumerate(classes)}

    def fit(self, x, y):
        encoded = np.asarray([self._index[v] for v in y])
        self._estimator.fit(x, encoded)
        return self

    def predict_proba(self, x):
        raw = self._estimator.predict_proba(x)
        present = list(getattr(self._estimator, "classes_", range(raw.shape[1])))
        out = np.zeros((raw.shape[0], len(self._classes)))
        for col, class_id in enumerate(present):
            out[:, int(class_id)] = raw[:, col]
        return out

    def predict(self, x):
        return np.asarray(self._classes)[np.argmax(self.predict_proba(x), axis=1)]

    @property
    def classes_(self):
        return np.asarray(self._classes)

    def __getattr__(self, item):  # expose feature_importances_ etc.
        return getattr(self._estimator, item)


def _fit(model_type: str, x_train: np.ndarray, y_train: np.ndarray):
    classes = sorted(set(y_train))
    estimator = build_estimator(model_type)
    if model_type == "xgboost":
        estimator = _LabelWrapper(estimator, classes)
    estimator.fit(x_train, y_train)
    return estimator


# ---------------------------------------------------------------------- #
# Validation
# ---------------------------------------------------------------------- #


def time_split(
    x: pd.DataFrame, y: pd.Series, split_pct: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronological split — no shuffling."""
    cut = int(len(x) * split_pct)
    xv = x.to_numpy(dtype=float)
    yv = y.to_numpy()
    return xv[:cut], xv[cut:], yv[:cut], yv[cut:]


def walk_forward_validate(
    x: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    folds: int = 5,
    initial_train_pct: float = 0.5,
) -> list[float]:
    """Expanding-window walk-forward accuracy per fold."""
    xv = x.to_numpy(dtype=float)
    yv = y.to_numpy()
    n = len(xv)
    start = int(n * initial_train_pct)
    fold_size = max(1, (n - start) // folds)
    scores: list[float] = []
    for fold in range(folds):
        train_end = start + fold * fold_size
        test_end = min(train_end + fold_size, n)
        if train_end >= n or train_end == test_end:
            break
        model = _fit(model_type, xv[:train_end], yv[:train_end])
        preds = model.predict(xv[train_end:test_end])
        score = accuracy_score(yv[train_end:test_end], preds)
        scores.append(float(score))
        log.info(
            "walk-forward fold %d/%d: train=%d test=%d accuracy=%.2f%%",
            fold + 1,
            folds,
            train_end,
            test_end - train_end,
            score * 100,
        )
    return scores


# ---------------------------------------------------------------------- #
# End-to-end training for one (symbol, timeframe)
# ---------------------------------------------------------------------- #


def prepare_training_frame(
    config: Config,
    client: BinanceDataClient,
    symbol: str,
    timeframe: str,
    candles: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Download history, build features (+ optional HTF context) and labels."""
    candles = candles or config.train_candles
    df = download_history(client, symbol, timeframe, candles, cache_dir=config.data_dir)
    if len(df) < 300:
        raise RuntimeError(
            f"{symbol} {timeframe}: only {len(df)} candles available - too few to train."
        )

    htf_frames: dict[str, pd.DataFrame] = {}
    if config.use_higher_timeframe_context:
        for htf in higher_timeframes(timeframe, config.higher_timeframe_count):
            # Enough higher-TF candles to cover the base history + warmup.
            htf_candles = max(300, candles // 3)
            htf_frames[htf] = download_history(
                client, symbol, htf, htf_candles, cache_dir=config.data_dir
            )

    frame, feature_cols = build_feature_frame(df, htf_frames or None)
    frame = add_labels(frame, config.neutral_threshold_pct)
    return frame, feature_cols


def train_for_pair(
    config: Config,
    client: BinanceDataClient,
    symbol: str,
    timeframe: str,
    model_type: str | None = None,
    candles: int | None = None,
) -> tuple[ModelBundle, dict]:
    """Train, validate and save one model. Returns (bundle, metrics)."""
    model_type = model_type or config.model_type
    frame, feature_cols = prepare_training_frame(config, client, symbol, timeframe, candles)
    x, y = training_rows(frame, feature_cols)
    if len(x) < 300:
        raise RuntimeError(
            f"{symbol} {timeframe}: only {len(x)} usable training rows after warmup."
        )
    log.info(
        "%s %s: training %s on %d rows (%d features); label distribution: %s",
        symbol,
        timeframe,
        model_type,
        len(x),
        len(feature_cols),
        y.value_counts(normalize=True).round(3).to_dict(),
    )

    # Holdout evaluation on the chronological tail.
    x_train, x_test, y_train, y_test = time_split(x, y, config.train_test_split_pct)
    holdout_model = _fit(model_type, x_train, y_train)
    holdout_pred = holdout_model.predict(x_test)
    holdout_accuracy = float(accuracy_score(y_test, holdout_pred))
    log.info(
        "%s %s: holdout accuracy %.2f%% on %d unseen rows",
        symbol,
        timeframe,
        holdout_accuracy * 100,
        len(y_test),
    )

    wf_scores: list[float] = []
    if config.walk_forward_enabled:
        wf_scores = walk_forward_validate(x, y, model_type, config.walk_forward_folds)
        if wf_scores:
            log.info(
                "%s %s: walk-forward mean accuracy %.2f%% over %d folds",
                symbol,
                timeframe,
                float(np.mean(wf_scores)) * 100,
                len(wf_scores),
            )

    # Final model refit on ALL rows so live predictions use the freshest data.
    final_model = _fit(model_type, x.to_numpy(dtype=float), y.to_numpy())
    classes = list(getattr(final_model, "classes_", sorted(set(y))))

    usable = frame.dropna(subset=[*feature_cols, "label"])
    bundle = ModelBundle(
        model=final_model,
        feature_cols=feature_cols,
        classes=[str(c) for c in classes],
        model_type=model_type,
        symbol=symbol,
        timeframe=timeframe,
        market_type=config.market_type,
        neutral_threshold_pct=config.neutral_threshold_pct,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        train_end_time_ms=int(usable["close_time_ms"].iloc[-1]),
        n_samples=len(x),
        holdout_accuracy=holdout_accuracy,
        walk_forward_scores=wf_scores,
    )
    path = save_bundle(bundle, config.model_path(symbol, timeframe, model_type))
    metrics = {
        "model_path": str(path),
        "n_samples": len(x),
        "holdout_accuracy": holdout_accuracy,
        "walk_forward_scores": wf_scores,
        "walk_forward_mean": float(np.mean(wf_scores)) if wf_scores else None,
    }
    return bundle, metrics
