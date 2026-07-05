"""Train ML models for next-candle prediction.

Downloads historical candles (cached under data/history/), engineers
features, validates with a chronological holdout plus optional walk-forward
folds, then saves one model bundle per (symbol, timeframe) to models/.

Usage:
    py train_model.py                              # all pairs from config.json
    py train_model.py --symbol BTCUSDT --timeframe 15m
    py train_model.py --model gradient_boosting --candles 8000
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from predictor.binance_data import BinanceDataClient
from predictor.config import load_config
from predictor.logger import get_logger, setup_logging
from predictor.model_training import train_for_pair

log = get_logger("train")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json", help="path to config.json")
    parser.add_argument("--symbol", help="train only this symbol (default: all from config)")
    parser.add_argument("--timeframe", help="train only this timeframe (default: all from config)")
    parser.add_argument(
        "--model",
        help="model type override (logistic_regression, random_forest, "
        "gradient_boosting, xgboost, lightgbm)",
    )
    parser.add_argument(
        "--candles", type=int, help="number of historical candles to train on"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.log_level)
    client = BinanceDataClient(config.market_type)

    symbols = [args.symbol.upper()] if args.symbol else config.symbols
    timeframes = [args.timeframe] if args.timeframe else config.timeframes
    model_type = args.model or config.model_type

    results: list[tuple[str, str, dict]] = []
    failures: list[tuple[str, str, str]] = []
    for symbol in symbols:
        for timeframe in timeframes:
            log.info("=== Training %s %s (%s) ===", symbol, timeframe, model_type)
            try:
                _, metrics = train_for_pair(
                    config, client, symbol, timeframe, model_type, args.candles
                )
                results.append((symbol, timeframe, metrics))
            except Exception as exc:
                log.error("%s %s: training failed: %s", symbol, timeframe, exc)
                failures.append((symbol, timeframe, str(exc)))

    print("\n" + "=" * 68)
    print(f"TRAINING SUMMARY ({model_type})")
    print("=" * 68)
    for symbol, timeframe, metrics in results:
        wf = metrics.get("walk_forward_scores") or []
        wf_text = (
            f", walk-forward mean {np.mean(wf) * 100:.1f}% over {len(wf)} folds"
            if wf
            else ""
        )
        print(
            f"{symbol:<10} {timeframe:<4} -> holdout accuracy "
            f"{metrics['holdout_accuracy'] * 100:.1f}% "
            f"({metrics['n_samples']} rows{wf_text})"
        )
        print(f"{'':16}saved: {metrics['model_path']}")
    for symbol, timeframe, error in failures:
        print(f"{symbol:<10} {timeframe:<4} -> FAILED: {error}")
    print("=" * 68)
    return 1 if failures and not results else 0


if __name__ == "__main__":
    sys.exit(main())
