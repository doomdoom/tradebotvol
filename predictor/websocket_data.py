"""Binance kline websocket listener (market data only).

Used purely as a candle-close trigger for live prediction: when a kline
closes (``k.x == true``), the registered callback fires. Feature
computation still pulls a consistent candle window over REST, so the
websocket payload itself is only passed along for reference.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Callable

from .logger import get_logger
from .utils import validate_timeframe

log = get_logger("websocket_data")

try:
    import websocket  # websocket-client
except ImportError:  # pragma: no cover
    websocket = None

#: callback(symbol, timeframe, closed_kline_dict)
ClosedCandleCallback = Callable[[str, str, dict], None]

_STREAM_HOSTS = {
    "spot": "wss://stream.binance.com:9443",
    "futures": "wss://fstream.binance.com",
}


class KlineStream:
    """Subscribes to kline streams and reports closed candles.

    Runs in a daemon thread and reconnects automatically with backoff.
    """

    def __init__(
        self,
        symbols: list[str],
        timeframes: list[str],
        market_type: str,
        on_closed_candle: ClosedCandleCallback,
    ) -> None:
        if websocket is None:
            raise RuntimeError(
                "websocket-client is not installed. Run 'pip install websocket-client' "
                "or set use_websocket=false in config.json to use REST polling."
            )
        if market_type not in _STREAM_HOSTS:
            raise ValueError(f"market_type must be one of {list(_STREAM_HOSTS)}")
        for tf in timeframes:
            validate_timeframe(tf)

        streams = "/".join(
            f"{symbol.lower()}@kline_{tf}" for symbol in symbols for tf in timeframes
        )
        self._url = f"{_STREAM_HOSTS[market_type]}/stream?streams={streams}"
        self._on_closed_candle = on_closed_candle
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._app: "websocket.WebSocketApp | None" = None

    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_forever, name="kline-stream", daemon=True
        )
        self._thread.start()
        log.info("Websocket stream started: %s", self._url)

    def stop(self) -> None:
        self._stop.set()
        if self._app is not None:
            try:
                self._app.close()
            except Exception:  # pragma: no cover - best effort shutdown
                pass
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        log.info("Websocket stream stopped")

    # ------------------------------------------------------------------ #

    def _run_forever(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            self._app = websocket.WebSocketApp(
                self._url,
                on_message=self._on_message,
                on_error=lambda _ws, err: log.warning("Websocket error: %s", err),
            )
            try:
                self._app.run_forever(ping_interval=180, ping_timeout=10)
            except Exception as exc:  # pragma: no cover - network dependent
                log.warning("Websocket crashed: %s", exc)
            if self._stop.is_set():
                break
            log.info("Websocket disconnected, reconnecting in %.0fs", backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

    def _on_message(self, _ws: object, message: str) -> None:
        try:
            payload = json.loads(message)
            data = payload.get("data", payload)
            if data.get("e") != "kline":
                return
            kline = data["k"]
            if not kline.get("x"):  # candle not closed yet
                return
            symbol = str(data["s"]).upper()
            timeframe = str(kline["i"])
            self._on_closed_candle(symbol, timeframe, kline)
        except Exception as exc:
            log.warning("Failed to handle websocket message: %s", exc)
