"""Backtest the predictor over historical candles.

Replays history candle by candle: for each row, a prediction is made using
only past data, then compared with the actual next candle. Produces the
same evaluation report as the live tool.

Rule-based mode evaluates every usable candle. ML mode evaluates only
candles *after* the model's training cutoff (out-of-sample); if none exist
it falls back to the chronological test split and warns.

Usage:
    py backtest_predictions.py                          # config defaults
    py backtest_predictions.py --mode ml --symbol BTCUSDT --timeframe 15m
    py backtest_predictions.py --candles 3000 --save
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from predictor.binance_data import BinanceDataClient
from predictor.config import Config, load_config
from predictor.evaluator import build_report, format_report
from predictor.feature_engineering import add_labels, build_feature_frame
from predictor.historical_data import download_history
from predictor.logger import get_logger, setup_logging
from predictor.ml_predictor import MLPredictor
from predictor.rule_based_predictor import RuleBasedPredictor
from predictor.storage import PredictionStorage
from predictor.utils import BEARISH, BULLISH, NEUTRAL, confidence_label, higher_timeframes

log = get_logger("backtest")


def _prepare_frame(
    config: Config, client: BinanceDataClient, symbol: str, timeframe: str, candles: int
) -> tuple[pd.DataFrame, list[str]]:
    df = download_history(client, symbol, timeframe, candles, cache_dir=config.data_dir)
    htf_frames: dict[str, pd.DataFrame] = {}
    if config.use_higher_timeframe_context:
        for htf in higher_timeframes(timeframe, config.higher_timeframe_count):
            htf_frames[htf] = download_history(
                client, symbol, htf, max(300, candles // 3), cache_dir=config.data_dir
            )
    frame, feature_cols = build_feature_frame(df, htf_frames or None)
    frame = add_labels(frame, config.neutral_threshold_pct)
    return frame.dropna(subset=["label"]), feature_cols


def _backtest_rule_based(
    config: Config, frame: pd.DataFrame, symbol: str, timeframe: str
) -> pd.DataFrame:
    predictor = RuleBasedPredictor(config.neutral_threshold_pct)
    records: list[dict] = []
    usable = frame.dropna(subset=["rsi", "atr_pct", "vol_regime"])
    for row in usable.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        pred = predictor.predict(row_series)
        records.append(
            _record(
                symbol,
                timeframe,
                row_series,
                pred.direction,
                pred.bullish_probability,
                pred.bearish_probability,
                pred.neutral_probability,
                pred.confidence_score,
                pred.explanation,
                "rule_based_backtest",
            )
        )
    return pd.DataFrame(records)


def _backtest_ml(
    config: Config, frame: pd.DataFrame, symbol: str, timeframe: str
) -> pd.DataFrame:
    predictor = MLPredictor.from_file(config.model_path(symbol, timeframe))
    bundle = predictor.bundle

    oos = frame[frame["close_time_ms"] > bundle.train_end_time_ms]
    n_oos = len(oos)
    if n_oos < 50:
        cut = int(len(frame) * config.train_test_split_pct)
        oos = frame.iloc[cut:]
        log.warning(
            "%s %s: only %d candles are newer than the model's training cutoff "
            "- falling back to the last %d%% of history. These rows may overlap "
            "the model's training data, so treat results as optimistic; the "
            "trustworthy out-of-sample numbers are the holdout/walk-forward "
            "scores printed by train_model.py.",
            symbol,
            timeframe,
            n_oos,
            round((1 - config.train_test_split_pct) * 100),
        )
    oos = oos.dropna(subset=[c for c in bundle.feature_cols if c in oos.columns])
    if oos.empty:
        raise RuntimeError(f"{symbol} {timeframe}: no usable out-of-sample rows")

    probs = predictor.predict_proba_frame(oos)
    records: list[dict] = []
    for (idx, row), (_, p) in zip(oos.iterrows(), probs.iterrows()):
        direction = str(p.idxmax())
        records.append(
            _record(
                symbol,
                timeframe,
                row,
                direction,
                float(p[BULLISH]),
                float(p[BEARISH]),
                float(p[NEUTRAL]),
                float(p.max()),
                f"{bundle.model_type} backtest prediction",
                f"{bundle.model_type}_backtest",
            )
        )
    return pd.DataFrame(records)


def _record(
    symbol: str,
    timeframe: str,
    row: pd.Series,
    direction: str,
    p_bull: float,
    p_bear: float,
    p_neutral: float,
    confidence: float,
    explanation: str,
    model_type: str,
) -> dict:
    actual_return = float(row["next_return_pct"])
    actual_direction = str(row["label"])
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "prediction_time": row["close_time"],
        "predicted_direction": direction,
        "bullish_probability": p_bull,
        "bearish_probability": p_bear,
        "neutral_probability": p_neutral,
        "confidence": confidence,
        "confidence_label": confidence_label(confidence),
        "actual_direction": actual_direction,
        "actual_return_pct": actual_return,
        "prediction_correct": int(direction == actual_direction),
        "model_type": model_type,
        "explanation": explanation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json", help="path to config.json")
    parser.add_argument("--mode", choices=["rule_based", "ml"], help="override prediction mode")
    parser.add_argument("--symbol", help="backtest only this symbol")
    parser.add_argument("--timeframe", help="backtest only this timeframe")
    parser.add_argument("--candles", type=int, default=3000, help="history length per pair")
    parser.add_argument(
        "--save",
        action="store_true",
        help="also store backtest predictions in SQLite/CSV",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.log_level)
    mode = args.mode or config.prediction_mode
    client = BinanceDataClient(config.market_type)

    symbols = [args.symbol.upper()] if args.symbol else config.symbols
    timeframes = [args.timeframe] if args.timeframe else config.timeframes

    all_records: list[pd.DataFrame] = []
    for symbol in symbols:
        for timeframe in timeframes:
            log.info("Backtesting %s %s (%s, %d candles)...", symbol, timeframe, mode, args.candles)
            try:
                frame, _ = _prepare_frame(config, client, symbol, timeframe, args.candles)
                if mode == "ml":
                    records = _backtest_ml(config, frame, symbol, timeframe)
                else:
                    records = _backtest_rule_based(config, frame, symbol, timeframe)
                log.info("%s %s: %d backtest predictions", symbol, timeframe, len(records))
                all_records.append(records)
            except Exception as exc:
                log.error("%s %s: backtest failed: %s", symbol, timeframe, exc)

    if not all_records:
        print("No backtest results produced.")
        return 1

    results = pd.concat(all_records, ignore_index=True)
    print(format_report(build_report(results)))

    if args.save:
        storage = PredictionStorage(
            config.db_path,
            config.csv_path,
            sqlite_enabled=config.sqlite_enabled,
            csv_enabled=config.csv_enabled,
        )
        saved = storage.save_resolved_records(results.to_dict("records"))
        storage.close()
        print(f"Saved {saved} backtest predictions to {config.db_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
