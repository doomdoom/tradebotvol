"""Configuration loading: config.json + optional .env overrides."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

from .utils import validate_timeframe

try:  # optional dependency; .env support degrades gracefully
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

VALID_PREDICTION_MODES = ("rule_based", "ml")
VALID_MARKET_TYPES = ("spot", "futures")
VALID_MODEL_TYPES = (
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "lightgbm",
)


@dataclass
class Config:
    """Runtime configuration (see config.json)."""

    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT"])
    timeframes: list[str] = field(default_factory=lambda: ["1m", "5m", "15m"])
    prediction_mode: str = "rule_based"
    market_type: str = "futures"
    candle_limit: int = 1000
    neutral_threshold_pct: float = 0.03
    min_confidence: float = 0.55
    use_higher_timeframe_context: bool = True
    higher_timeframe_count: int = 2
    model_type: str = "random_forest"
    train_test_split_pct: float = 0.8
    walk_forward_enabled: bool = True
    walk_forward_folds: int = 5
    train_candles: int = 5000
    retrain_on_start: bool = False
    save_predictions: bool = True
    sqlite_enabled: bool = True
    csv_enabled: bool = True
    log_level: str = "INFO"
    display_utc_offset_hours: float = 0.0  # e.g. 5 shows times as UTC+5
    dashboard_controls_enabled: bool = False  # pause/clear/shutdown buttons
    use_websocket: bool = False
    data_dir: str = "data"
    models_dir: str = "models"
    db_path: str = "data/predictions.db"
    csv_path: str = "data/predictions.csv"
    # Optional API credentials (NOT required for public market data).
    api_key: str = ""
    api_secret: str = ""

    def __post_init__(self) -> None:
        self.symbols = [s.upper().strip() for s in self.symbols if s.strip()]
        if not self.symbols:
            raise ValueError("config: 'symbols' must contain at least one symbol")
        for tf in self.timeframes:
            validate_timeframe(tf)
        if self.prediction_mode not in VALID_PREDICTION_MODES:
            raise ValueError(
                f"config: prediction_mode must be one of {VALID_PREDICTION_MODES}"
            )
        if self.market_type not in VALID_MARKET_TYPES:
            raise ValueError(f"config: market_type must be one of {VALID_MARKET_TYPES}")
        if self.model_type not in VALID_MODEL_TYPES:
            raise ValueError(f"config: model_type must be one of {VALID_MODEL_TYPES}")
        if not 0.5 <= self.train_test_split_pct < 1.0:
            raise ValueError("config: train_test_split_pct must be in [0.5, 1.0)")
        if self.neutral_threshold_pct < 0:
            raise ValueError("config: neutral_threshold_pct must be >= 0")
        if self.candle_limit < 200:
            raise ValueError("config: candle_limit must be >= 200 (indicator warmup)")

    @property
    def pairs(self) -> list[tuple[str, str]]:
        """All (symbol, timeframe) combinations to predict."""
        return [(s, tf) for s in self.symbols for tf in self.timeframes]

    def model_path(self, symbol: str, timeframe: str, model_type: str | None = None) -> Path:
        model_type = model_type or self.model_type
        name = f"{self.market_type}_{symbol}_{timeframe}_{model_type}.joblib"
        return Path(self.models_dir) / name


def load_config(path: str | Path = "config.json") -> Config:
    """Load config.json, then apply .env / environment overrides."""
    if load_dotenv is not None:
        load_dotenv()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path.resolve()}. "
            "Copy the provided config.json or pass --config."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))

    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"config: unknown keys {sorted(unknown)}")

    cfg = Config(**raw)
    cfg.api_key = os.getenv("BINANCE_API_KEY", cfg.api_key)
    cfg.api_secret = os.getenv("BINANCE_API_SECRET", cfg.api_secret)
    env_level = os.getenv("LOG_LEVEL")
    if env_level:
        cfg.log_level = env_level
    return cfg
