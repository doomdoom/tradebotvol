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
from predictor.regime import regime_label
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


def _fmt_qty(q: float) -> str:
    return f"{q:g}"


def _fmt_px(v: float) -> str:
    return f"{v:,.2f}" if v >= 1 else f"{v:.6f}"


def _base(symbol: str) -> str:
    return symbol.replace("USDT", "").replace("BUSD", "")

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


def _activation_move(args) -> float:
    """Price-move fraction at which the trailing stop activates. If
    --trail-activation-roi is set, it is derived from a target ROI on margin
    (roi% / leverage); otherwise the fixed --trail-activation-pct is used."""
    roi = getattr(args, "trail_activation_roi", 0.0) or 0.0
    if roi > 0 and getattr(args, "leverage", 0):
        return (roi / 100.0) / args.leverage
    return args.trail_activation_pct / 100.0


def _exit_prices(entry: float, side: str, sl_pct: float, act_pct: float,
                 price_precision: int) -> tuple[float, float]:
    """Return (stop_loss_price, trailing_activation_price). For a long the stop
    is below entry and trailing activates above it (once in profit); mirrored
    for a short."""
    if side == "BUY":
        return (round(entry * (1 - sl_pct / 100.0), price_precision),
                round(entry * (1 + act_pct / 100.0), price_precision))
    return (round(entry * (1 + sl_pct / 100.0), price_precision),
            round(entry * (1 - act_pct / 100.0), price_precision))


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


def _trade_symbol(args, engine, client, filters, symbol: str, recent: list,
                  off: float, trails: dict) -> None:
    side, result = _decide(engine, symbol, args.timeframe)
    tag = f"{symbol} {args.timeframe}"
    conf = result.confidence * 100
    regime = regime_label(result.market_regime)
    if side is None:
        msg = (f"No trade — signal too weak ({conf:.0f}% confidence) "
               f"in {regime} conditions")
        log.info("%s: %s", tag, msg)
        _record(recent, symbol, "no_trade", msg, off)
        return

    entry = float(result.reference_close)
    close_side = "SELL" if side == "BUY" else "BUY"
    direction = "LONG" if side == "BUY" else "SHORT"

    if client is None:  # dry-run
        qty = args.notional / entry if entry else 0.0
        msg = (f"Would open {direction} {_fmt_qty(round(qty, 6))} {_base(symbol)} "
               f"(~${args.notional:.0f}) at ${_fmt_px(entry)} — {regime}, "
               f"{conf:.0f}% confidence")
        log.info("[DRY] %s: %s", tag, msg)
        _record(recent, symbol, "would_trade", msg, off)
        return

    pos = client.position(symbol)
    if not pos["flat"]:
        upnl = pos["unrealized"]
        msg = f"Holding open trade (P&L {'+' if upnl >= 0 else ''}${upnl:.2f})"
        log.info("%s: %s", tag, msg)
        _record(recent, symbol, "hold", msg, off)
        return

    f = filters.get(symbol, {"qty_precision": 3, "price_precision": 2, "step": None})
    price = client.price(symbol)
    qty = _round_step(args.notional / price, f["qty_precision"], f["step"])
    if qty <= 0:
        _record(recent, symbol, "skip", "notional too small for min qty", off)
        return
    pp = f["price_precision"]
    try:
        client.cancel_all(symbol)  # clear any orphaned orders from a prior close
        client.set_leverage(symbol, args.leverage)
        client.market_order(symbol, side, qty)
        # Use the ACTUAL fill price (after slippage), not the pre-order price, so
        # the stop / break-even are correct. Stop-loss + trailing are handled
        # client-side by _manage_positions (this testnet rejects conditional
        # order types), so we only remember the position + its stop level here.
        opened = client.position(symbol)
        entry = float(opened["entry"]) if opened.get("entry") else price
        sl = round(entry * (1 - args.sl_pct / 100.0) if side == "BUY"
                   else entry * (1 + args.sl_pct / 100.0), pp)
        trails[symbol] = {"close_side": close_side, "entry": entry, "stop": sl,
                          "peak": entry, "pp": pp, "qty": qty}
        bet = "price to rise" if side == "BUY" else "price to fall"
        act_desc = (f"+{args.trail_activation_roi:.0f}% ROI"
                    if getattr(args, "trail_activation_roi", 0) > 0
                    else f"+{args.trail_activation_pct}%")
        msg = (f"Opened {direction} {_fmt_qty(qty)} {_base(symbol)} at ${_fmt_px(entry)} "
               f"(betting {bet}) — stop-loss ${_fmt_px(sl)}, trailing stop kicks in at "
               f"{act_desc} to lock profit. {regime.capitalize()}, {conf:.0f}% confidence.")
        log.info("TESTNET %s: %s", tag, msg)
        _record(recent, symbol, "open", msg, off)
    except Exception as exc:
        log.error("%s: order failed on testnet: %s", tag, exc)
        _record(recent, symbol, "error", f"order failed: {exc}", off)


def _manage_positions(config, client, filters, trails: dict, args, recent: list, off: float) -> None:
    """Fully client-side stop-loss + trailing stop. This testnet rejects
    conditional order types (STOP/TP/TRAILING, error -4120), so the bot watches
    the price each tick and CLOSES with a plain market order when the stop or
    trailed level is hit. As price moves in favour, the stop ratchets up (long)
    / down (short); once in profit it is floored at break-even+ so a winning
    trade cannot turn into a loss.
    """
    act = _activation_move(args)  # activate trailing at this price move (may be ROI-based)
    trail = args.trail_pct / 100.0
    for symbol in config.symbols:
        try:
            pos = client.position(symbol)
        except Exception:
            continue
        if pos["flat"]:
            trails.pop(symbol, None)
            continue
        try:
            price = client.price(symbol)
        except Exception:
            continue
        long = pos["amount"] > 0
        close_side = "SELL" if long else "BUY"
        fdict = filters.get(symbol, {"qty_precision": 3, "price_precision": 2})
        pp = fdict.get("price_precision", 2)
        # Round to the symbol's quantity precision so the reduce-only market
        # close is accepted (a raw float can trip Binance precision error -1111).
        qty = round(abs(pos["amount"]), fdict.get("qty_precision", 3))
        if qty <= 0:
            continue
        t = trails.get(symbol)
        if t is None:  # adopt a position opened before this run (e.g. after restart)
            entry0 = pos["entry"]
            stop0 = round(entry0 * (1 - args.sl_pct / 100.0) if long
                          else entry0 * (1 + args.sl_pct / 100.0), pp)
            t = {"close_side": close_side, "entry": entry0, "stop": stop0,
                 "peak": entry0, "pp": pp, "qty": qty}
            trails[symbol] = t
            try:
                client.cancel_all(symbol)  # drop any stale/rejected exchange orders
            except Exception:
                pass
        entry = t["entry"]

        # Track the best price and ratchet the trailing stop once in profit.
        if long:
            t["peak"] = max(t.get("peak", entry), price)
            if price >= entry * (1 + act):
                floor = entry * (1 + act)                      # break-even+ lock
                cand = round(max(t["peak"] * (1 - trail), floor), pp)
                if cand > t["stop"]:
                    t["stop"] = cand
                    _record(recent, symbol, "trail",
                            f"Raised stop to ${_fmt_px(cand)} to lock in profit", off)
            hit = price <= t["stop"]
        else:
            t["peak"] = min(t.get("peak", entry), price)
            if price <= entry * (1 - act):
                floor = entry * (1 - act)
                cand = round(min(t["peak"] * (1 + trail), floor), pp)
                if cand < t["stop"]:
                    t["stop"] = cand
                    _record(recent, symbol, "trail",
                            f"Lowered stop to ${_fmt_px(cand)} to lock in profit", off)
            hit = price >= t["stop"]

        if hit:
            try:
                client.market_close(symbol, close_side, qty)
                gain = (price - entry) if long else (entry - price)
                outcome = "profit locked in ✓" if gain >= 0 else "loss cut"
                _record(recent, symbol, "closed",
                        f"Closed at ${_fmt_px(price)} — {outcome}", off)
                log.info("%s: closed at %.4f (stop %.4f), booking %s",
                         symbol, price, t["stop"], "profit" if gain >= 0 else "loss")
                trails.pop(symbol, None)
                try:
                    client.cancel_all(symbol)
                except Exception:
                    pass
            except Exception as exc:
                log.warning("%s: market close failed: %s", symbol, exc)


def _write_state(path: Path, args, client, config, recent: list, off: float) -> None:
    from datetime import timedelta

    positions = []
    history: list = []
    realized_total = None
    balance = None
    today_profit = today_loss = today_fees = 0.0
    total_fees = None
    equity = None
    today = (utc_now() + timedelta(hours=off)).strftime("%Y-%m-%d")
    if client is not None:
        try:
            balance = round(client.usdt_balance(), 2)
            for s in config.symbols:
                p = client.position(s)
                if not p["flat"]:
                    value = round(abs(p["amount"]) * p["entry"], 2)
                    lev = max(p.get("leverage") or args.leverage, 1)
                    margin = round(value / lev, 2)
                    # Live ROI on margin (leveraged) — what tracks the +12% trigger.
                    roi = round(p["unrealized"] / margin * 100, 1) if margin else 0.0
                    positions.append({
                        "symbol": s,
                        "side": "LONG" if p["amount"] > 0 else "SHORT",
                        "amount": round(p["amount"], 6),
                        "value_usdt": value,      # leveraged notional (USDT)
                        "margin_usdt": margin,    # capital actually put up
                        "entry": round(p["entry"], 2),
                        "unrealized": round(p["unrealized"], 2),
                        "roi_pct": roi,
                    })
        except Exception as exc:
            log.warning("state: could not read testnet account: %s", exc)
        # Commission (fees) — the bot only sends MARKET orders, so every fill is
        # a TAKER fill. Map each fill's fee by tradeId to attach it to the trade.
        comm_by_trade: dict = {}
        try:
            comm_rows = client.income(income_type="COMMISSION", limit=300)
            total_fees = round(sum(abs(float(r.get("income", 0))) for r in comm_rows), 4)
            for r in comm_rows:
                fee = abs(float(r.get("income", 0)))
                if to_display_time(int(r.get("time", 0)), off).startswith(today):
                    today_fees += fee
                tid = r.get("tradeId")
                if tid is not None:
                    comm_by_trade[tid] = comm_by_trade.get(tid, 0.0) + fee
            today_fees = round(today_fees, 4)
        except Exception as exc:
            log.warning("state: could not read fees: %s", exc)
        try:
            rows = client.income(income_type="REALIZED_PNL", limit=100)
            rows.sort(key=lambda r: int(r.get("time", 0)), reverse=True)
            realized_total = round(sum(float(r.get("income", 0)) for r in rows), 2)
            for r in rows:
                ts = to_display_time(int(r.get("time", 0)), off)
                pnl = round(float(r.get("income", 0)), 2)
                if ts.startswith(today):
                    if pnl >= 0:
                        today_profit += pnl
                    else:
                        today_loss += pnl
                if len(history) < 15:
                    fee = comm_by_trade.get(r.get("tradeId"))
                    history.append({"time": ts, "symbol": str(r.get("symbol", "")),
                                    "pnl": pnl, "win": pnl >= 0,
                                    "fee": round(fee, 4) if fee else None,
                                    "fee_type": "taker"})
            today_profit = round(today_profit, 2)
            today_loss = round(today_loss, 2)
        except Exception as exc:
            log.warning("state: could not read trade history: %s", exc)
        # Total asset = wallet balance + unrealized P&L of open positions.
        if balance is not None:
            equity = round(balance + sum(p["unrealized"] for p in positions), 2)

    # Live status so the user can see the bot is working even when it's idle.
    next_check = to_display_time(
        current_candle_close_ms(utc_now_ms(), args.timeframe), off, with_date=False)
    trading = len(positions) > 0
    if trading:
        status_text = (f"In {len(positions)} open trade"
                       f"{'s' if len(positions) != 1 else ''} — trailing to lock profit")
    else:
        status_text = (f"Waiting for a signal — scanning {', '.join(config.symbols)} "
                       f"live, deciding on each {args.timeframe} candle close")
    state = {
        "updated_display": to_display_time(utc_now(), off),
        "tz": display_tz_label(off),
        "mode": "live-testnet" if client is not None else "dry-run",
        "timeframe": args.timeframe, "notional": args.notional,
        "leverage": args.leverage, "sl_pct": args.sl_pct,
        "trail_pct": args.trail_pct, "trail_activation_pct": args.trail_activation_pct,
        "trail_activation_roi": getattr(args, "trail_activation_roi", 0.0),
        "symbols": config.symbols, "balance": balance, "equity": equity,
        "status": "trading" if len(positions) > 0 else "waiting",
        "status_text": status_text, "next_check": next_check,
        "today_profit": today_profit, "today_loss": today_loss,
        "today_net": round(today_profit + today_loss, 2),
        "today_fees": today_fees, "total_fees": total_fees,
        "positions": positions, "recent": list(reversed(recent)),
        "history": history, "realized_total": realized_total,
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
    parser.add_argument("--sl-pct", type=float, default=0.5, help="protective stop-loss %%")
    parser.add_argument("--trail-pct", type=float, default=0.2,
                        help="trailing-stop callback %% (smaller = locks profit faster; min 0.1)")
    parser.add_argument("--trail-activation-pct", type=float, default=0.15,
                        help="price-move %% at which the stop locks + trailing starts")
    parser.add_argument("--trail-activation-roi", type=float, default=0.0,
                        help="if >0, activate trailing at this ROI%% on margin instead "
                             "(price move = roi/leverage). e.g. 12 = trail from +12%% ROI")
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
    trails: dict = {}

    def one_pass():
        for symbol in config.symbols:
            try:
                _trade_symbol(args, engine, client, filters, symbol, recent, off, trails)
            except Exception as exc:
                log.error("%s: cycle failed: %s", symbol, exc)
                _record(recent, symbol, "error", str(exc)[:120], off)
        _write_state(state_path, args, client, config, recent, off)

    if not args.loop:
        one_pass()
        return 0

    log.info("Looping on %s candle closes; trailing stops managed every 20s. Ctrl-C to stop.",
             args.timeframe)
    next_close = current_candle_close_ms(utc_now_ms(), args.timeframe)
    one_pass()
    last_manage = utc_now_ms()
    try:
        while True:
            now = utc_now_ms()
            if now >= next_close + 2500:
                one_pass()
                next_close = current_candle_close_ms(utc_now_ms(), args.timeframe)
            # Watch stops / trail + refresh the dashboard panel every ~5s (live).
            if client is not None and now - last_manage >= 5000:
                try:
                    _manage_positions(config, client, filters, trails, args, recent, off)
                    _write_state(state_path, args, client, config, recent, off)
                except Exception as exc:
                    log.warning("position management tick failed: %s", exc)
                last_manage = now
            time.sleep(1.0)
    except KeyboardInterrupt:
        log.info("Stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
