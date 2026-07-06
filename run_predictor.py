"""Live next-candle prediction loop.

For every configured (symbol, timeframe): each time a candle closes, the
previous prediction is scored against the actual result, a new prediction
is made for the next candle, and everything is logged to SQLite/CSV.

This program only reads public market data. It never places orders.

Usage:
    py run_predictor.py [--config config.json] [--once]
"""

from __future__ import annotations

import argparse
import queue
import signal
import sys
import time

import pandas as pd

from predictor.binance_data import BinanceDataClient
from predictor.config import Config, load_config
from predictor.logger import get_logger, setup_logging
from predictor.model_training import train_for_pair
from predictor.prediction_engine import PredictionEngine, PredictionResult
from predictor.storage import PredictionStorage
from predictor.utils import (
    current_candle_close_ms,
    display_tz_label,
    expected_price_range,
    format_price,
    format_range_pct,
    label_return,
    to_display_time,
    utc_now,
    utc_now_ms,
)

log = get_logger("run")

#: Grace period after a candle's close before fetching it over REST.
CLOSE_BUFFER_MS = 2500


class LiveRunner:
    """Schedules per-pair prediction cycles on candle closes."""

    def __init__(
        self,
        config: Config,
        engine: PredictionEngine,
        storage: PredictionStorage | None,
    ) -> None:
        self.config = config
        self.engine = engine
        self.storage = storage
        self.pairs = config.pairs
        now = utc_now_ms()
        self._next_close = {
            pair: current_candle_close_ms(now, pair[1]) for pair in self.pairs
        }
        # pair -> (storage row id | None, PredictionResult)
        self._pending: dict[tuple[str, str], tuple[int | None, PredictionResult]] = {}
        self._stop = False
        self._ws_events: queue.Queue[tuple[str, str]] = queue.Queue()
        self._stream = None

    # ------------------------------------------------------------------ #

    def stop(self, *_args) -> None:
        self._stop = True

    def run(self, once: bool = False) -> None:
        if self.config.use_websocket:
            self._start_stream()
        log.info(
            "Live predictor started: %d pair(s), mode=%s, market=%s",
            len(self.pairs),
            self.config.prediction_mode,
            self.config.market_type,
        )
        if once:
            for pair in self.pairs:
                self._cycle(pair)
            return
        try:
            while not self._stop:
                if self.config.use_websocket:
                    self._drain_ws_events()
                else:
                    self._poll_due_pairs()
                time.sleep(0.5)
        finally:
            if self._stream is not None:
                self._stream.stop()
            if self.storage is not None:
                self.storage.close()
            log.info("Live predictor stopped")

    # ------------------------------------------------------------------ #

    def _start_stream(self) -> None:
        from predictor.websocket_data import KlineStream

        self._stream = KlineStream(
            self.config.symbols,
            self.config.timeframes,
            self.config.market_type,
            on_closed_candle=lambda s, tf, _k: self._ws_events.put((s, tf)),
        )
        self._stream.start()

    def _drain_ws_events(self) -> None:
        while True:
            try:
                symbol, timeframe = self._ws_events.get_nowait()
            except queue.Empty:
                return
            if (symbol, timeframe) in self._next_close:
                self._cycle((symbol, timeframe))

    def _poll_due_pairs(self) -> None:
        now = utc_now_ms()
        for pair in self.pairs:
            if now >= self._next_close[pair] + CLOSE_BUFFER_MS:
                self._cycle(pair)
                self._next_close[pair] = current_candle_close_ms(utc_now_ms(), pair[1])

    # ------------------------------------------------------------------ #

    def _cycle(self, pair: tuple[str, str]) -> None:
        symbol, timeframe = pair
        try:
            candles = self.engine.fetch_candles(symbol, timeframe)
            if candles.empty:
                log.warning("%s %s: no candles returned", symbol, timeframe)
                return
            self._resolve_pending(pair, candles)
            result = self.engine.predict(symbol, timeframe, candles)
            row_id = (
                self.storage.save_prediction(result)
                if (self.storage is not None and self.config.save_predictions)
                else None
            )
            self._pending[pair] = (row_id, result)
            self._print_prediction(result)
        except Exception as exc:
            log.error("%s %s: prediction cycle failed: %s", symbol, timeframe, exc)

    def _resolve_pending(self, pair: tuple[str, str], candles: pd.DataFrame) -> None:
        pending = self._pending.get(pair)
        if pending is None:
            return
        row_id, prediction = pending
        target_open_ms = int(prediction.target_candle_time.timestamp() * 1000)
        match = candles[candles["open_time_ms"] == target_open_ms]
        if match.empty:
            # Target candle not closed yet (or missed) — keep waiting unless stale.
            if utc_now_ms() > prediction.target_close_time_ms + 3 * 60_000:
                log.warning(
                    "%s %s: dropping stale unresolved prediction for %s",
                    *pair,
                    prediction.target_candle_time,
                )
                self._pending.pop(pair, None)
            return

        actual_close = float(match["close"].iloc[0])
        actual_return_pct = (actual_close / prediction.reference_close - 1.0) * 100.0
        actual_direction = label_return(
            actual_return_pct, self.config.neutral_threshold_pct
        )
        correct = prediction.predicted_direction == actual_direction
        if row_id is not None and self.storage is not None:
            self.storage.resolve_prediction(row_id, actual_direction, actual_return_pct)
        self._pending.pop(pair, None)
        self._print_resolution(pair, prediction, actual_direction, actual_return_pct, correct)

    # ------------------------------------------------------------------ #

    def _print_prediction(self, r: PredictionResult) -> None:
        off = self.config.display_utc_offset_hours
        tz = display_tz_label(off)
        stamp = to_display_time(utc_now(), off)
        target = to_display_time(r.target_candle_time, off, with_date=False)
        price_low, price_high = expected_price_range(
            r.reference_close, r.expected_move_min_pct, r.expected_move_max_pct
        )
        block = (
            f"\n[{stamp} {tz}] {r.symbol} {r.timeframe}\n"
            f"Prediction for next candle (opens {target} {tz}):\n"
            f"Direction: {r.predicted_direction}\n"
            f"Bullish probability: {r.bullish_probability * 100:.1f}%\n"
            f"Bearish probability: {r.bearish_probability * 100:.1f}%\n"
            f"Neutral probability: {r.neutral_probability * 100:.1f}%\n"
            f"Confidence: {r.confidence_label} ({r.confidence * 100:.1f}%)\n"
            f"Expected move: "
            f"{format_range_pct(r.expected_move_min_pct, r.expected_move_max_pct)}\n"
            f"Expected price: from {format_price(r.reference_close)} "
            f"to {format_price(price_low)}-{format_price(price_high)} "
            f"(estimate, next close)\n"
            f"Reason: {r.explanation}"
        )
        print(block, flush=True)

    def _print_resolution(
        self,
        pair: tuple[str, str],
        prediction: PredictionResult,
        actual_direction: str,
        actual_return_pct: float,
        correct: bool,
    ) -> None:
        symbol, timeframe = pair
        off = self.config.display_utc_offset_hours
        tz = display_tz_label(off)
        lines = [
            f"\n[{to_display_time(utc_now(), off)} {tz}] {symbol} {timeframe} - candle "
            f"{to_display_time(prediction.target_candle_time, off, with_date=False)} "
            f"{tz} closed",
            f"Actual result: {actual_direction} ({actual_return_pct:+.3f}%)",
            f"Prediction result: {'correct' if correct else 'wrong'} "
            f"(predicted {prediction.predicted_direction})",
        ]
        if self.storage is not None:
            _, total, accuracy = self.storage.get_accuracy(symbol, timeframe)
            lines.append(
                f"Updated accuracy for {symbol} {timeframe}: "
                f"{accuracy:.1f}% over {total} predictions"
            )
        print("\n".join(lines), flush=True)


# ---------------------------------------------------------------------- #


def _ensure_ml_models(config: Config, client: BinanceDataClient, engine: PredictionEngine) -> None:
    """Load models for all pairs, training missing ones when allowed."""
    for symbol, timeframe in config.pairs:
        path = config.model_path(symbol, timeframe)
        if not path.exists() or config.retrain_on_start:
            if not config.retrain_on_start and not path.exists():
                log.warning("%s %s: model missing - training it now", symbol, timeframe)
            _, metrics = train_for_pair(config, client, symbol, timeframe)
            log.info("%s %s: trained (%s)", symbol, timeframe, metrics)
    engine.load_ml_models()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json", help="path to config.json")
    parser.add_argument(
        "--once",
        action="store_true",
        help="make one prediction per pair and exit (no live loop)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.log_level)
    print(
        "NOTE: prediction-only research tool. It does not trade and its "
        "predictions do not guarantee accuracy or profit.\n",
        flush=True,
    )

    client = BinanceDataClient(config.market_type)
    storage = (
        PredictionStorage(
            config.db_path,
            config.csv_path,
            sqlite_enabled=config.sqlite_enabled,
            csv_enabled=config.csv_enabled,
        )
        if config.save_predictions
        else None
    )
    engine = PredictionEngine(config, client)
    if config.use_enhanced:
        log.info("Enhanced model selected - training per-pair models on startup...")
        engine.ensure_enhanced_models()
    elif config.prediction_mode == "ml":
        _ensure_ml_models(config, client, engine)

    runner = LiveRunner(config, engine, storage)
    signal.signal(signal.SIGINT, runner.stop)
    try:
        signal.signal(signal.SIGTERM, runner.stop)
    except (AttributeError, ValueError):
        pass  # not available on all platforms

    runner.run(once=args.once)
    return 0


if __name__ == "__main__":
    sys.exit(main())
