"""Paper-trading simulator: what WOULD the signals have done, after costs?

This is a SIMULATION only. It never places an order, never connects to a
trading endpoint, and never needs a private API key. It replays the enhanced
model's out-of-sample signals over historical candles and simulates:

    * go long  when the signal is bullish, short when bearish
    * enter at the candle close where the signal fires, exit one candle later
    * subtract realistic taker fees + slippage on entry and exit

...then reports the P&L you would have seen. The point is to find out, WITHOUT
risking a cent, whether trading these signals is actually profitable after the
costs that quietly eat near-coin-flip strategies alive.

Usage:
    py tools/paper_trade.py
    py tools/paper_trade.py --fee-bps 4 --slippage-bps 2 --notional 100
    py tools/paper_trade.py --symbols BTCUSDT ETHUSDT --timeframes 15m 1h

NOT FINANCIAL ADVICE. Past simulated performance does not predict real results;
real trading also suffers latency, partial fills, funding and liquidations that
this simulator does not model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor.backtest_metrics import WAIT
from predictor.binance_data import BinanceDataClient
from predictor.config import load_config
from predictor.logger import get_logger, setup_logging
from predictor.regime import BREAKOUT
from predictor.research_model import config_to_enhanced
from predictor.utils import BEARISH, BULLISH
from tools.backtest_accuracy import _prepare, _walk_forward_enhanced

log = get_logger("paper_trade")


def _signals_for_pair(config, client, cfg, symbol, timeframe, candles, folds):
    frame, feats = _prepare(config, client, symbol, timeframe, candles)
    if len(frame) < 400:
        return pd.DataFrame()
    oos = _walk_forward_enhanced(frame, feats, cfg, folds)
    if oos.empty:
        return pd.DataFrame()
    oos = oos.copy()
    oos["symbol"] = symbol
    oos["timeframe"] = timeframe
    return oos


def _simulate(trades: pd.DataFrame, cost_frac: float, notional: float) -> dict:
    """Turn signalled rows into simulated P&L. trades must have columns
    predicted_direction, next_return_pct, market_regime, close_time_ms."""
    # Only directional calls become trades (skip neutral and WAIT).
    live = trades[trades["predicted_direction"].isin([BULLISH, BEARISH])].copy()
    if live.empty:
        return {"n_trades": 0}
    sign = np.where(live["predicted_direction"] == BULLISH, 1.0, -1.0)
    gross = sign * live["next_return_pct"].to_numpy() / 100.0  # fraction
    net = gross - cost_frac                                    # round-trip cost
    live["gross_ret"] = gross
    live["net_ret"] = net
    live["pnl"] = net * notional
    live = live.sort_values("close_time_ms")

    wins = live[live["net_ret"] > 0]["net_ret"]
    losses = live[live["net_ret"] <= 0]["net_ret"]
    equity = notional + live["pnl"].cumsum()
    peak = equity.cummax()
    drawdown = (equity - peak)
    n = int(len(live))

    return {
        "n_trades": n,
        "win_rate_pct": float((live["net_ret"] > 0).mean() * 100.0),
        "gross_pnl": float(live["gross_ret"].sum() * notional),
        "net_pnl": float(live["pnl"].sum()),
        "fees_paid": float(cost_frac * notional * n),
        "avg_net_ret_pct": float(live["net_ret"].mean() * 100.0),
        "avg_win_pct": float(wins.mean() * 100.0) if len(wins) else 0.0,
        "avg_loss_pct": float(losses.mean() * 100.0) if len(losses) else 0.0,
        "profit_factor": (float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else float("inf")),
        "max_drawdown": float(drawdown.min()),
        "return_on_notional_pct": float(live["pnl"].sum() / notional * 100.0),
        "final_equity": float(equity.iloc[-1]),
    }


def _fmt_money(x):
    return f"${x:,.2f}" if x is not None and not (isinstance(x, float) and np.isnan(x)) else "n/a"


def _fmt_pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.2f}%"


def _markdown(summary: dict) -> str:
    L = ["# TradeBotVol - Paper-Trading Simulation\n"]
    L.append("> **Simulation only. No real orders, no API keys, not financial advice.**\n")
    p = summary["params"]
    L.append(f"_Signals: enhanced model (WAIT/regime filter as live) · "
             f"{summary['n_signals_evaluated']:,} OOS candles · {len(p['pairs'])} pairs_  ")
    L.append(f"_Costs: {p['fee_bps']}bps fee + {p['slippage_bps']}bps slippage per side "
             f"= {p['round_trip_cost_pct']:.3f}% round trip · ${p['notional']:.0f}/trade · no leverage_\n")

    def block(title, s):
        L.append(f"## {title}\n")
        if s.get("n_trades", 0) == 0:
            L.append("_No trades taken._\n"); return
        verdict = "PROFITABLE" if s["net_pnl"] > 0 else "LOSS-MAKING"
        L.append(f"**{verdict}** after costs.\n")
        L.append("| Metric | Value |")
        L.append("|--------|-------|")
        L.append(f"| Trades taken | {s['n_trades']:,} |")
        L.append(f"| Win rate | {s['win_rate_pct']:.1f}% |")
        L.append(f"| Gross P&L (no fees) | {_fmt_money(s['gross_pnl'])} |")
        L.append(f"| **Net P&L (after fees)** | **{_fmt_money(s['net_pnl'])}** |")
        L.append(f"| Fees paid | {_fmt_money(s['fees_paid'])} |")
        L.append(f"| Return on ${p['notional']:.0f}/trade | {_fmt_pct(s['return_on_notional_pct'])} |")
        L.append(f"| Avg net return / trade | {_fmt_pct(s['avg_net_ret_pct'])} |")
        L.append(f"| Avg win / avg loss | {_fmt_pct(s['avg_win_pct'])} / {_fmt_pct(s['avg_loss_pct'])} |")
        L.append(f"| Profit factor | {s['profit_factor']:.2f} |")
        L.append(f"| Max drawdown | {_fmt_money(s['max_drawdown'])} |")
        L.append("")

    block("All signals", summary["all"])
    block("Breakouts only", summary["breakouts_only"])
    block("No-fee fantasy (shows how much fees cost you)", summary["all_no_fees"])

    L.append("## What this means\n")
    L.append(summary["verdict"])
    L.append("\n\n_Real trading is harder than this sim: add latency, slippage on "
             "bigger size, funding costs, and the emotional pressure of a drawdown. "
             "Treat a simulated profit as necessary-but-not-sufficient._")
    return "\n".join(L)


def _verdict(all_s: dict, brk_s: dict, nofee_s: dict) -> str:
    parts = []
    if all_s.get("n_trades", 0) == 0:
        return "No directional signals were produced, so there is nothing to trade."
    net = all_s["net_pnl"]
    gross = nofee_s["net_pnl"]  # all_no_fees has cost 0 -> equals gross
    if net > 0:
        parts.append(f"Trading every signal nets {_fmt_money(net)} after fees - a small "
                     "positive edge, but verify it holds before risking real money.")
    else:
        parts.append(f"Trading every signal LOSES {_fmt_money(net)} after fees "
                     f"(it would have made {_fmt_money(gross)} with zero fees - so fees/near-"
                     "coin-flip accuracy are the killer).")
    if brk_s.get("n_trades", 0) >= 20:
        b = brk_s["net_pnl"]
        if b > 0:
            parts.append(f"Breakouts-only nets {_fmt_money(b)} over {brk_s['n_trades']} trades "
                         f"({brk_s['win_rate_pct']:.0f}% win) - the most promising subset, but "
                         "the sample is small; treat as a hypothesis, not a guarantee.")
        else:
            parts.append(f"Even breakouts-only loses {_fmt_money(b)} after costs.")
    parts.append("Bottom line: this is a research signal, not a proven money-maker. If you "
                 "trade it, do so manually, tiny size, money you can afford to lose.")
    return " ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--timeframes", nargs="*")
    parser.add_argument("--candles", type=int, default=3000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fee-bps", type=float, default=4.0, help="taker fee per side (bps)")
    parser.add_argument("--slippage-bps", type=float, default=2.0, help="slippage per side (bps)")
    parser.add_argument("--notional", type=float, default=100.0, help="$ per trade (no leverage)")
    parser.add_argument("--outdir", default="reports")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.log_level)
    client = BinanceDataClient(config.market_type)
    cfg = config_to_enhanced(config)
    # Trade the same signals the live bot commits to (WAIT/regime filter on).
    cfg.no_signal_mode = True

    symbols = [s.upper() for s in (args.symbols or config.symbols)]
    timeframes = args.timeframes or config.timeframes
    cost_frac = (args.fee_bps + args.slippage_bps) * 2 / 10000.0

    frames = []
    pairs_done = []
    for s in symbols:
        for tf in timeframes:
            try:
                log.info("Simulating %s %s...", s, tf)
                oos = _signals_for_pair(config, client, cfg, s, tf, args.candles, args.folds)
                if not oos.empty:
                    frames.append(oos); pairs_done.append(f"{s} {tf}")
            except Exception as exc:
                log.error("%s %s: %s", s, tf, exc)

    if not frames:
        print("No signals produced - nothing to simulate.")
        return 1
    allsig = pd.concat(frames, ignore_index=True)

    all_s = _simulate(allsig, cost_frac, args.notional)
    brk_s = _simulate(allsig[allsig["market_regime"] == BREAKOUT], cost_frac, args.notional)
    nofee_s = _simulate(allsig, 0.0, args.notional)

    summary = {
        "params": {
            "pairs": pairs_done, "candles": args.candles, "folds": args.folds,
            "fee_bps": args.fee_bps, "slippage_bps": args.slippage_bps,
            "round_trip_cost_pct": cost_frac * 100.0, "notional": args.notional,
            "market_type": config.market_type,
        },
        "n_signals_evaluated": int(len(allsig)),
        "all": all_s,
        "breakouts_only": brk_s,
        "all_no_fees": nofee_s,
    }
    summary["verdict"] = _verdict(all_s, brk_s, nofee_s)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "paper_trade.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    md = _markdown(summary)
    (outdir / "paper_trade.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\nSaved: {outdir/'paper_trade.json'}, {outdir/'paper_trade.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
