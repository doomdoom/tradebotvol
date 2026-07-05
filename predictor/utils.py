"""Shared helpers: timeframe math, direction labels, confidence buckets."""

from __future__ import annotations

import math
from datetime import datetime, timezone

BULLISH = "bullish"
BEARISH = "bearish"
NEUTRAL = "neutral"

#: Canonical class order used everywhere (storage, models, reports).
DIRECTIONS: list[str] = [BEARISH, NEUTRAL, BULLISH]

#: Supported Binance kline intervals, in ascending duration order.
TIMEFRAME_MINUTES: dict[str, int] = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "1d": 1440,
}


def validate_timeframe(timeframe: str) -> str:
    if timeframe not in TIMEFRAME_MINUTES:
        supported = ", ".join(TIMEFRAME_MINUTES)
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Supported: {supported}")
    return timeframe


def timeframe_ms(timeframe: str) -> int:
    """Duration of one candle in milliseconds."""
    return TIMEFRAME_MINUTES[validate_timeframe(timeframe)] * 60_000


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_ms() -> int:
    return int(utc_now().timestamp() * 1000)


def ms_to_datetime(ms: int | float) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def floor_time_ms(ts_ms: int, timeframe: str) -> int:
    """Open time of the candle containing ``ts_ms``."""
    step = timeframe_ms(timeframe)
    return ts_ms - (ts_ms % step)


def current_candle_close_ms(now_ms: int, timeframe: str) -> int:
    """Timestamp at which the currently-forming candle will close."""
    return floor_time_ms(now_ms, timeframe) + timeframe_ms(timeframe)


def higher_timeframes(timeframe: str, count: int = 3) -> list[str]:
    """Higher timeframes usable as context for ``timeframe``.

    Picks the nearest larger intervals whose duration is between 3x and 240x
    the base interval (e.g. 1m -> ['5m', '15m', '30m'] with count=3).
    """
    base = TIMEFRAME_MINUTES[validate_timeframe(timeframe)]
    out = [
        tf
        for tf, minutes in TIMEFRAME_MINUTES.items()
        if 3 <= minutes / base <= 240
    ]
    return out[:count]


def confidence_label(score: float) -> str:
    """Bucket a winning-class probability into low / medium / high."""
    if score >= 0.70:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"


def label_return(return_pct: float, neutral_threshold_pct: float) -> str:
    """Classify a next-candle percent return into a direction label."""
    if return_pct is None or (isinstance(return_pct, float) and math.isnan(return_pct)):
        raise ValueError("return_pct must be a finite number")
    if return_pct > neutral_threshold_pct:
        return BULLISH
    if return_pct < -neutral_threshold_pct:
        return BEARISH
    return NEUTRAL


def format_signed_pct(value: float) -> str:
    return f"{value:+.2f}%"


def format_range_pct(low: float, high: float) -> str:
    return f"{format_signed_pct(low)} to {format_signed_pct(high)}"


def display_tz_label(offset_hours: float) -> str:
    """Short label for a UTC offset, e.g. 0 -> 'UTC', 5 -> 'UTC+5'."""
    if not offset_hours:
        return "UTC"
    whole = int(offset_hours)
    minutes = int(round(abs(offset_hours - whole) * 60))
    sign = "+" if offset_hours >= 0 else "-"
    if minutes:
        return f"UTC{sign}{abs(whole)}:{minutes:02d}"
    return f"UTC{sign}{abs(whole)}"


def to_display_time(
    utc_value: object, offset_hours: float = 0.0, with_date: bool = True
) -> str:
    """Render a UTC datetime / ISO string / epoch-ms shifted by ``offset_hours``.

    Storage always stays in UTC; this is purely for human-facing display.
    """
    from datetime import timedelta

    if utc_value is None or utc_value == "":
        return ""
    if isinstance(utc_value, (int, float)):
        dt = ms_to_datetime(int(utc_value))
    elif isinstance(utc_value, datetime):
        dt = utc_value if utc_value.tzinfo else utc_value.replace(tzinfo=timezone.utc)
    else:  # ISO string
        text = str(utc_value).replace("T", " ")[:19]
        dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    shifted = dt + timedelta(hours=offset_hours)
    fmt = "%Y-%m-%d %H:%M:%S" if with_date else "%H:%M:%S"
    return shifted.strftime(fmt)


def format_price(price: float) -> str:
    """Format a price with a sensible number of decimals for its magnitude."""
    price = float(price)
    if price >= 100:
        return f"{price:,.2f}"
    if price >= 1:
        return f"{price:,.4f}"
    return f"{price:.6f}"


def expected_price_range(
    reference_close: float, move_min_pct: float, move_max_pct: float
) -> tuple[float, float]:
    """Convert a percent move range into absolute price levels off a base price."""
    low = reference_close * (1.0 + move_min_pct / 100.0)
    high = reference_close * (1.0 + move_max_pct / 100.0)
    return low, high
