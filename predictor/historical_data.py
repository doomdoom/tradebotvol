"""Bulk historical candle download with local CSV caching."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .binance_data import BinanceDataClient
from .logger import get_logger

log = get_logger("historical_data")


def cache_path(cache_dir: str | Path, market_type: str, symbol: str, timeframe: str) -> Path:
    return Path(cache_dir) / "history" / f"{market_type}_{symbol}_{timeframe}.csv"


def load_cached(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
        return df.sort_values("open_time_ms").reset_index(drop=True)
    except Exception as exc:  # corrupted cache -> redownload
        log.warning("Ignoring unreadable cache %s: %s", path, exc)
        return None


def download_history(
    client: BinanceDataClient,
    symbol: str,
    timeframe: str,
    candles: int,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return the most recent ``candles`` closed candles.

    Pages backwards through the klines endpoint until enough history is
    collected. If a cache directory is given, previously downloaded candles
    are reused and only missing ranges are fetched.
    """
    path = (
        cache_path(cache_dir, client.market_type, symbol, timeframe)
        if cache_dir
        else None
    )
    df = load_cached(path) if (path and use_cache) else None
    if df is None:
        df = pd.DataFrame()

    # 1) Extend forward: fetch candles newer than the cache.
    if not df.empty:
        start = int(df["open_time_ms"].iloc[-1]) + 1
        newer = _page_forward(client, symbol, timeframe, start)
        if not newer.empty:
            df = pd.concat([df, newer], ignore_index=True)

    # 2) Backfill: fetch candles older than what we have until we reach the target.
    while len(df) < candles:
        end_time = int(df["open_time_ms"].iloc[0]) - 1 if not df.empty else None
        batch = client.get_klines(
            symbol, timeframe, limit=client.max_limit, end_time=end_time
        )
        if batch.empty:
            log.info(
                "%s %s: exchange history exhausted at %d candles (requested %d)",
                symbol,
                timeframe,
                len(df),
                candles,
            )
            break
        got = len(batch)
        df = pd.concat([batch, df], ignore_index=True)
        log.debug("%s %s: backfilled %d candles (total %d)", symbol, timeframe, got, len(df))
        # A short batch means the start of the listing was reached — but only
        # when paging with an explicit end_time. The initial request (no
        # end_time) loses the still-forming candle to the only_closed filter,
        # so a full page legitimately arrives one candle short.
        if end_time is not None and got < client.max_limit:
            break

    df = (
        df.drop_duplicates(subset="open_time_ms", keep="last")
        .sort_values("open_time_ms")
        .reset_index(drop=True)
        .tail(candles)
        .reset_index(drop=True)
    )

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        log.info("%s %s: cached %d candles at %s", symbol, timeframe, len(df), path)
    return df


def _page_forward(
    client: BinanceDataClient, symbol: str, timeframe: str, start_time: int
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    cursor = start_time
    while True:
        batch = client.get_klines(
            symbol, timeframe, limit=client.max_limit, start_time=cursor
        )
        if batch.empty:
            break
        frames.append(batch)
        cursor = int(batch["open_time_ms"].iloc[-1]) + 1
        if len(batch) < client.max_limit:
            break
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
