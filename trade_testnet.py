"""TESTNET trading bot - trades the model's signals on Binance Futures TESTNET.

FAKE MONEY. Connects only to the Binance testnet (see testnet_client.py), so no
real funds are ever at risk. Reads the enhanced signals and, when a directional
signal fires, opens a position with take-profit + stop-loss attached, then lets
the exchange manage the exit. Writes its status to reports/testnet_bot.json so
the dashboard can display it.

Modes:
  (default)      DRY-RUN - prints/records what it WOULD trade. No keys needed.
  --live-testnet places real orders on the TESTNET (needs free testnet keys).
                 If keys are missing it safely falls back to dry-run.

Testnet keys (fake money): https://testnet.binancefuture.com -> API Key, then
put BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET in .env.

Usage:
  py trade_testnet.py                          # dry-run, one pass
  py trade_testnet.py --loop                   # dry-run, every candle close
  py trade_testnet.py --live-testnet --loop    # trade on TESTNET

Never touches a real account; never guarantees profit. Learning tool.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from predictor.binance_data import BinanceDataClient
from predictor.config import load_config
from predictor.logger import get_logger, setup_logging
from predictor.prediction_engine import PredictionEngine
from predictor.utils import (
    BEARISH,
    BULLISH,
    current_candle_close_ms,
    display_tz_label,
    timeframe_ms,
    to_display_time,
    utc_now,
    utc_now_ms,
)

log = get_logger("trade_testnet")

BANNER = r"""
============================================================
  TESTNET TRADING BOT  -  FAKE MONEY, NO REAL FUNDS AT RISK
  Signals are ~coin-flip; this is for learning, not profit.
============================================================
"""

_MAX_RECENT = 40


def _round_step(qty: float, precision: int, step: float | None) -> float:
    if step:
        qty = math.floor(qty / step) * step
    return round(qty, precision)


def _tp_sl_prices(entry: float, side: str, tp_pct: float, sl_pct: float,
                  price_precision: int) -> tuple[float, float]:
    if side == "BUY":
        return (round(entry * (1 + tp_pct / 100.0), price_precision),
                round(entry * (1 - sl_pct / 100.0), price_precision))
    return (round(entry * (1 - tp_pct / 100.0), price_precision),
            round(entry * (1 + sl_pct / 100.0), price_precision))


def _decide(engine: PredictionEngine, symbol: str, timeframe: str):
    result = engine.predict(symbol, timeframe)
    d = result.predicted_direction
    if d == BULLISH and result.meets_min_confidence:
        return "BUY", result
    if d == BEARISH and result.meets_min_confidence:
        return "SELL", result
    return None, result


def _record(recent: list, symbol: str, event: str, text: str, off: float) -> None:
    recent.append({
        "time": to_display_time(utc_now(), off, with_date=False),
        "symbol": symbol, "event": event, "text": text,
    })
    del recent[:-_MAX_RECENT]


def _trade_symbol(args, engine, client, filters, symbol: str, recent: list, off: float) -> None:
    side, result = _decide(engine, symbol, args.timeframe)
    tag = f"{symbol} {args.timeframe}"
    if side is None:
        msg = (f"{result.predicted_direction} · conf {result.confidence*100:.0f}% · "
               f"{result.market_regime.lower() or 'n/a'} — below gate, no trade")
        log.info("%s: %s", tag, msg)
        _record(recent, symbol, "no_trade", msg, off)
        return

    entry = float(result.reference_close)
    close_side = "SELL" if side == "BUY" else "BUY"

    if client is None:  # dry-run
        qty = args.notional / entry if entry else 0.0
        tp, sl = _tp_sl_prices(entry, side, args.tp_pct, args.sl_pct, 2)
        msg = (f"would {side} ~{qty:.6f} (~${args.notional:.0f}) @ ~{entry:.2f} · "
               f"TP {tp:.2f} SL {sl:.2f} · {result.market_regime.lower()} · "
               f"conf {result.confidence*100:.0f}%")
        log.info("[DRY] %s: %s", tag, msg)
        _record(recent, symbol, "would_trade", msg, off)
        return

    pos = client.position(symbol)
    if not pos["flat"]:
        msg = f"already in position (amt {pos['amount']:.6f}, uPnL {pos['unrealized']:.2f}) — holding"
        log.info("%s: %s", tag, msg)
        _record(recent, symbol, "hold", msg, off)
        return

    f = filters.get(symbol, {"qty_precision": 3, "price_precision": 2, "step": None})
    price = client.price(symbol)
    qty = _round_step(args.notional / price, f["qty_precision"], f["step"])
    if qty <= 0:
        _record(recent, symbol, "skip", "notional too small for min qty", off)
        return
    tp, sl = _tp_sl_prices(price, side, args.tp_pct, args.sl_pct, f["price_precision"])
    try:
        client.set_leverage(symbol, args.leverage)
        client.market_order(symbol, side, qty)
        client.close_trigger(symbol, close_side, tp, "take_profit")
        client.close_trigger(symbol, close_side, sl, "stop")
        msg = (f"opened {side} {qty:.6f} @ ~{price:.2f} · TP {tp:.2f} SL {sl:.2f} · "
               f"{result.market_regime.lower()} · conf {result.confidence*100:.0f}%")
        log.info("TESTNET %s: %s", tag, msg)
        _record(recent, symbol, "open", msg, off)
    except Exception as exc:
        log.error("%s: order failed on testnet: %s", tag, exc)
        _record(recent, symbol, "error", f"order failed: {exc}", off)


def _write_state(path: Path, args, client, config, recent: list, off: float) -> None:
    positions = []
    balance = None
    if client is not None:
        try:
            balance = round(client.usdt_balance(), 2)
            for s in config.symbols:
                p = client.position(s)
                if not p["flat"]:
                    positions.append({
                        "symbol": s,
                        "side": "LONG" if p["amount"] > 0 else "SHORT",
                        "amount": round(p["amount"], 6),
                        "entry": round(p["entry"], 2),
                        "unrealized": round(p["unrealized"], 2),
                    })
        except Exception as exc:
            log.warning("state: could not read testnet account: %s", exc)
    state = {
        "updated_display": to_display_time(utc_now(), off),
        "tz": display_tz_label(off),
        "mode": "live-testnet" if client is not None else "dry-run",
        "timeframe": args.timeframe, "notional": args.notional,
        "leverage": args.leverage, "tp_pct": args.tp_pct, "sl_pct": args.sl_pct,
        "symbols": config.symbols, "balance": balance,
        "positions": positions, "recent": list(reversed(recent)),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("state: could not write %s: %s", path, exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--notional", type=float, default=100.0)
    parser.add_argument("--leverage", type=int, default=5)
    parser.add_argument("--tp-pct", type=float, default=0.6)
    parser.add_argument("--sl-pct", type=float, default=0.4)
    parser.add_argument("--live-testnet", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--state-file", default="reports/testnet_bot.json")
    args = parser.parse_args()

    print(BANNER, flush=True)
    config = load_config(args.config)
    setup_logging(config.log_level)
    off = config.display_utc_offset_hours
    state_path = Path(args.state_file)

    client = None
    filters: dict = {}
    if args.live_testnet:
        key = os.getenv("BINANCE_TESTNET_API_KEY", "")
        secret = os.getenv("BINANCE_TESTNET_API_SECRET", "")
        if key and secret:
            try:
                from predictor.testnet_client import BinanceTestnetClient
                client = BinanceTestnetClient(key, secret)
                filters = client.exchange_filters()
                log.info("Connected to Binance TESTNET. Fake USDT balance: %.2f",
                         client.usdt_balance())
            except Exception as exc:
                log.error("Could not connect to testnet (%s) - falling back to DRY-RUN", exc)
                client = None
        else:
            log.warning("--live-testnet set but no testnet keys in .env - running DRY-RUN. "
                        "Add BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET to trade.")
    else:
        log.info("DRY-RUN mode: showing intended trades only (add --live-testnet + keys to trade).")

    data_client = BinanceDataClient(config.market_type)
    engine = PredictionEngine(config, data_client)
    if config.use_enhanced:
        log.info("Training enhanced models for %s...", ", ".join(config.symbols))
        engine.ensure_enhanced_models([(s, args.timeframe) for s in config.symbols])

    recent: list = []

    def one_pass():
        for symbol in config.symbols:
            try:
                _trade_symbol(args, engine, client, filters, symbol, recent, off)
            except Exception as exc:
                log.error("%s: cycle failed: %s", symbol, exc)
                _record(recent, symbol, "error", str(exc)[:120], off)
        _write_state(state_path, args, client, config, recent, off)

    if not args.loop:
        one_pass()
        return 0

    log.info("Looping on %s candle closes. Ctrl-C to stop.", args.timeframe)
    next_close = current_candle_close_ms(utc_now_ms(), args.timeframe)
    one_pass()
    try:
        while True:
            if utc_now_ms() >= next_close + 2500:
                one_pass()
                next_close = current_candle_close_ms(utc_now_ms(), args.timeframe)
            time.sleep(1.0)
    except KeyboardInterrupt:
        log.info("Stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
