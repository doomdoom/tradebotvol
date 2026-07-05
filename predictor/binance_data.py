"""Binance public market-data REST client (spot and USDT-M futures).

Read-only: only the public klines endpoints are used. No API key is
required and no trading endpoint is ever touched.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

from .logger import get_logger
from .utils import utc_now_ms, validate_timeframe

log = get_logger("binance_data")

KLINE_COLUMNS = [
    "open_time_ms",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time_ms",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "_ignore",
]

_FLOAT_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "taker_buy_base",
    "taker_buy_quote",
]


class BinanceAPIError(RuntimeError):
    """Raised when the Binance REST API cannot be reached or rejects a request."""


class BinanceDataClient:
    """Thin klines client with retries and rate-limit backoff."""

    _ENDPOINTS = {
        "spot": ("https://api.binance.com", "/api/v3/klines", 1000),
        "futures": ("https://fapi.binance.com", "/fapi/v1/klines", 1500),
    }

    def __init__(
        self,
        market_type: str = "spot",
        timeout: float = 10.0,
        max_retries: int = 4,
    ) -> None:
        if market_type not in self._ENDPOINTS:
            raise ValueError(f"market_type must be one of {list(self._ENDPOINTS)}")
        self.market_type = market_type
        base, path, max_limit = self._ENDPOINTS[market_type]
        self._url = base + path
        self.max_limit = max_limit
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "next-candle-predictor/1.0 (research)"

    # ------------------------------------------------------------------ #

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: int | None = None,
        end_time: int | None = None,
        only_closed: bool = True,
    ) -> pd.DataFrame:
        """Fetch candles as a DataFrame sorted by open time (ascending).

        Args:
            symbol: e.g. ``BTCUSDT``.
            interval: e.g. ``1m`` (validated against supported timeframes).
            limit: number of candles (capped at the endpoint maximum).
            start_time / end_time: optional epoch-ms bounds.
            only_closed: drop the still-forming candle at the end.
        """
        validate_timeframe(interval)
        params: dict[str, object] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(int(limit), self.max_limit),
        }
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)

        raw = self._request(params)
        df = self._to_dataframe(raw)
        if only_closed and not df.empty:
            df = df[df["close_time_ms"] <= utc_now_ms()].reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ #

    def _request(self, params: dict[str, object]) -> list[list]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.get(self._url, params=params, timeout=self.timeout)
                if resp.status_code in (429, 418):  # rate limited / banned
                    wait = float(resp.headers.get("Retry-After", 2 * attempt))
                    log.warning(
                        "Rate limited by Binance (HTTP %s), sleeping %.1fs",
                        resp.status_code,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                if resp.status_code >= 500:
                    raise BinanceAPIError(f"Binance server error HTTP {resp.status_code}")
                if resp.status_code != 200:
                    raise BinanceAPIError(
                        f"Binance rejected request (HTTP {resp.status_code}): "
                        f"{resp.text[:300]}"
                    )
                data = resp.json()
                if not isinstance(data, list):
                    raise BinanceAPIError(f"Unexpected klines payload: {data!r:.200}")
                return data
            except (requests.RequestException, BinanceAPIError) as exc:
                last_error = exc
                if isinstance(exc, BinanceAPIError) and "rejected" in str(exc):
                    raise  # 4xx client errors are not retryable
                backoff = min(2.0**attempt, 15.0)
                log.warning(
                    "Klines request failed (attempt %d/%d): %s - retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    exc,
                    backoff,
                )
                time.sleep(backoff)
        raise BinanceAPIError(f"Klines request failed after retries: {last_error}")

    @staticmethod
    def _to_dataframe(raw: list[list]) -> pd.DataFrame:
        if not raw:
            return pd.DataFrame(columns=[c for c in KLINE_COLUMNS if c != "_ignore"])
        df = pd.DataFrame(raw, columns=KLINE_COLUMNS).drop(columns="_ignore")
        for col in _FLOAT_COLUMNS:
            df[col] = df[col].astype(float)
        df["trades"] = df["trades"].astype(int)
        df["open_time_ms"] = df["open_time_ms"].astype("int64")
        df["close_time_ms"] = df["close_time_ms"].astype("int64")
        df["open_time"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time_ms"], unit="ms", utc=True)
        return df.sort_values("open_time_ms").reset_index(drop=True)
