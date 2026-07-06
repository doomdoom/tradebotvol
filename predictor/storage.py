"""Prediction persistence: SQLite (canonical) + CSV snapshot."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .logger import get_logger
from .utils import utc_now

if TYPE_CHECKING:
    from .prediction_engine import PredictionResult

log = get_logger("storage")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    prediction_time TEXT NOT NULL,
    target_candle_time TEXT NOT NULL,
    predicted_direction TEXT NOT NULL,
    bullish_probability REAL NOT NULL,
    bearish_probability REAL NOT NULL,
    neutral_probability REAL NOT NULL,
    confidence REAL NOT NULL,
    confidence_label TEXT NOT NULL,
    expected_move_min_pct REAL,
    expected_move_max_pct REAL,
    reference_close REAL NOT NULL,
    actual_direction TEXT,
    actual_return_pct REAL,
    prediction_correct INTEGER,
    model_type TEXT NOT NULL,
    explanation TEXT,
    market_regime TEXT,
    signal_strength TEXT,
    model_version TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_pair
    ON predictions (symbol, timeframe);
"""

#: Columns added after the original schema shipped. Added via ALTER TABLE on
#: existing databases so old prediction logs keep working (values are NULL for
#: rows written before the upgrade).
_MIGRATION_COLUMNS = {
    "market_regime": "TEXT",
    "signal_strength": "TEXT",
    "model_version": "TEXT",
}


class PredictionStorage:
    """Thread-safe prediction log backed by SQLite with optional CSV export."""

    def __init__(
        self,
        db_path: str | Path = "data/predictions.db",
        csv_path: str | Path = "data/predictions.csv",
        sqlite_enabled: bool = True,
        csv_enabled: bool = True,
    ) -> None:
        if not sqlite_enabled:
            # SQLite stays the canonical store; an in-memory DB keeps the API
            # identical while writing nothing to disk.
            db_path = ":memory:"
        else:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.csv_path = Path(csv_path)
        self.csv_enabled = csv_enabled
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Add any newer columns missing from a pre-existing predictions table."""
        existing = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(predictions)").fetchall()
        }
        for column, coltype in _MIGRATION_COLUMNS.items():
            if column not in existing:
                self._conn.execute(
                    f"ALTER TABLE predictions ADD COLUMN {column} {coltype}"
                )
                log.info("storage: migrated predictions table (+%s)", column)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ #

    def save_prediction(self, result: "PredictionResult") -> int:
        """Insert a new (unresolved) prediction; returns the row id."""
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO predictions (
                    timestamp, symbol, timeframe, prediction_time,
                    target_candle_time, predicted_direction,
                    bullish_probability, bearish_probability, neutral_probability,
                    confidence, confidence_label,
                    expected_move_min_pct, expected_move_max_pct,
                    reference_close, model_type, explanation,
                    market_regime, signal_strength, model_version
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    utc_now().isoformat(timespec="seconds"),
                    result.symbol,
                    result.timeframe,
                    result.prediction_time.isoformat(timespec="seconds"),
                    result.target_candle_time.isoformat(timespec="seconds"),
                    result.predicted_direction,
                    result.bullish_probability,
                    result.bearish_probability,
                    result.neutral_probability,
                    result.confidence,
                    result.confidence_label,
                    result.expected_move_min_pct,
                    result.expected_move_max_pct,
                    result.reference_close,
                    result.model_type,
                    result.explanation,
                    getattr(result, "market_regime", "") or None,
                    getattr(result, "signal_strength", "") or None,
                    getattr(result, "model_version", "") or None,
                ),
            )
            self._conn.commit()
            row_id = int(cursor.lastrowid)
        self._export_csv()
        return row_id

    def resolve_prediction(
        self,
        row_id: int,
        actual_direction: str,
        actual_return_pct: float,
    ) -> bool:
        """Fill in the actual outcome for a stored prediction."""
        with self._lock:
            row = self._conn.execute(
                "SELECT predicted_direction FROM predictions WHERE id = ?", (row_id,)
            ).fetchone()
            if row is None:
                log.warning("resolve_prediction: row %d not found", row_id)
                return False
            correct = int(row["predicted_direction"] == actual_direction)
            self._conn.execute(
                """
                UPDATE predictions
                SET actual_direction = ?, actual_return_pct = ?, prediction_correct = ?
                WHERE id = ?
                """,
                (actual_direction, actual_return_pct, correct, row_id),
            )
            self._conn.commit()
        self._export_csv()
        return bool(correct)

    def save_resolved_records(self, records: list[dict]) -> int:
        """Bulk-insert already-resolved predictions (used by backtests)."""
        rows = [
            (
                utc_now().isoformat(timespec="seconds"),
                r["symbol"],
                r["timeframe"],
                str(r["prediction_time"]),
                str(r.get("target_candle_time", r["prediction_time"])),
                r["predicted_direction"],
                r["bullish_probability"],
                r["bearish_probability"],
                r["neutral_probability"],
                r["confidence"],
                r["confidence_label"],
                r.get("expected_move_min_pct"),
                r.get("expected_move_max_pct"),
                r.get("reference_close", 0.0),
                r["actual_direction"],
                r["actual_return_pct"],
                int(r["prediction_correct"]),
                r["model_type"],
                r.get("explanation", ""),
                r.get("market_regime") or r.get("regime"),
                r.get("signal_strength"),
                r.get("model_version"),
            )
            for r in records
        ]
        with self._lock:
            self._conn.executemany(
                """
                INSERT INTO predictions (
                    timestamp, symbol, timeframe, prediction_time,
                    target_candle_time, predicted_direction,
                    bullish_probability, bearish_probability, neutral_probability,
                    confidence, confidence_label,
                    expected_move_min_pct, expected_move_max_pct,
                    reference_close, actual_direction, actual_return_pct,
                    prediction_correct, model_type, explanation,
                    market_regime, signal_strength, model_version
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
            self._conn.commit()
        self._export_csv()
        return len(rows)

    # ------------------------------------------------------------------ #

    def get_unresolved(
        self, symbol: str | None = None, timeframe: str | None = None
    ) -> list[dict]:
        query = "SELECT * FROM predictions WHERE actual_direction IS NULL"
        params: list[str] = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if timeframe:
            query += " AND timeframe = ?"
            params.append(timeframe)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_accuracy(self, symbol: str, timeframe: str) -> tuple[int, int, float]:
        """(correct, resolved_total, accuracy_pct) for one pair."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT COALESCE(SUM(prediction_correct), 0) AS correct,
                       COUNT(*) AS total
                FROM predictions
                WHERE symbol = ? AND timeframe = ? AND actual_direction IS NOT NULL
                """,
                (symbol, timeframe),
            ).fetchone()
        correct, total = int(row["correct"]), int(row["total"])
        accuracy = (correct / total * 100.0) if total else 0.0
        return correct, total, accuracy

    def load_dataframe(self) -> pd.DataFrame:
        with self._lock:
            return pd.read_sql_query(
                "SELECT * FROM predictions ORDER BY id", self._conn
            )

    # ------------------------------------------------------------------ #

    def _export_csv(self) -> None:
        if not self.csv_enabled:
            return
        try:
            df = self.load_dataframe()
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.csv_path, index=False)
        except Exception as exc:  # CSV export must never break the live loop
            log.warning("CSV export failed: %s", exc)
