# Binance Next-Candle Prediction Bot

A research tool that predicts the **direction of the next candle** (bullish /
bearish / neutral) for multiple symbols and timeframes on Binance spot or
USDⓈ-M futures, using either a transparent **rule-based** engine or trained
**machine-learning** models. Every prediction is logged, then scored against
the actual candle once it closes, so accuracy is measured continuously.

> ## ⚠️ Safety note
> **This bot only predicts market direction. It does not guarantee accuracy
> or profit.** It places no orders, uses no leverage, and never touches
> trading endpoints — it reads public market data only. It should not be
> connected to live trading until its prediction accuracy has been tested
> over a large sample.

---

## Features

- **Two prediction modes**
  - `rule_based` — weighted technical signals (EMA stack, RSI, MACD, VWAP,
    Bollinger Bands, ATR, volume spikes, breakouts, candle anatomy,
    streaks) with a plain-English explanation for every prediction.
  - `ml` — logistic regression, random forest, gradient boosting, plus
    XGBoost / LightGBM if installed, predicting three classes.
- **Multi-timeframe**: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 1d — each pair gets
  its own data, features, model, and accuracy statistics.
- **Higher-timeframe context** (optional): e.g. 1m predictions can see the
  5m/15m trend — merged on candle *close* times, so no look-ahead.
- **Look-ahead-safe training**: chronological train/test split, no
  shuffling, expanding-window walk-forward validation, and the model bundle
  records its training cutoff so backtests stay out-of-sample.
- **Live loop**: predicts at every candle close, resolves the previous
  prediction, prints and stores the outcome, and keeps a running accuracy.
- **Storage**: SQLite (canonical) + CSV snapshot of the full prediction log.
- **Evaluation**: accuracy, per-class precision/recall, confusion matrices,
  average move after bullish/bearish calls, confidence-vs-accuracy, and
  best/worst symbol & timeframe rankings.

## Project layout

```
run_predictor.py          # live prediction loop
train_model.py            # download data + train ML models
backtest_predictions.py   # replay history and score predictions
config.json               # all runtime settings
predictor/
  config.py               # config loading & validation
  binance_data.py         # public klines REST client (spot + futures)
  websocket_data.py       # kline websocket (candle-close trigger)
  historical_data.py      # bulk download with CSV caching
  indicators.py           # EMA, RSI, MACD, Bollinger, ATR, VWAP, ...
  feature_engineering.py  # features, labels, higher-TF context
  rule_based_predictor.py # weighted-signal predictor with explanations
  ml_predictor.py         # model bundle loading + inference
  model_training.py       # time-split training + walk-forward validation
  prediction_engine.py    # orchestration: data -> features -> prediction
  evaluator.py            # performance reports
  storage.py              # SQLite + CSV prediction log
  logger.py, utils.py
```

## Installation

Requires **Python 3.11+**.

```bash
cd "preduct bot"
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env        # optional — no API key is needed
```

Optional extras: `pip install xgboost lightgbm` to unlock those model types.

## How to run rule-based prediction

`config.json` ships with `"prediction_mode": "rule_based"`, so just:

```bash
python run_predictor.py
```

You'll see one block per (symbol, timeframe) at each candle close:

```
[2026-07-04 14:30:00] BTCUSDT 1m
Prediction for next candle (opens 14:30:00 UTC):
Direction: bullish
Bullish probability: 64.2%
Bearish probability: 22.7%
Neutral probability: 13.1%
Confidence: medium (64.2%)
Expected move: +0.06% to +0.14%
Reason: Bullish probability is highest (64%) because EMA stack is bullish
(EMA9 > EMA21 > EMA50), RSI is rising above 55, price is above VWAP, ...

[2026-07-04 14:31:02] BTCUSDT 1m — candle 14:30:00 UTC closed
Actual result: bullish (+0.081%)
Prediction result: correct (predicted bullish)
Updated accuracy for BTCUSDT 1m: 57.8% over 45 predictions
```

Useful flags:

- `python run_predictor.py --once` — one prediction per pair, then exit.
- `python run_predictor.py --config my_config.json` — alternate config.
- Set `"use_websocket": true` to trigger cycles from the Binance kline
  websocket instead of REST polling (features are still computed from a
  consistent REST candle window either way).

## How to download historical data

History is downloaded automatically (and cached under `data/history/`)
whenever you train or backtest. To pre-fetch explicitly:

```bash
python train_model.py --symbol BTCUSDT --timeframe 15m --candles 8000
```

or from Python:

```python
from predictor.binance_data import BinanceDataClient
from predictor.historical_data import download_history

client = BinanceDataClient("futures")
df = download_history(client, "BTCUSDT", "15m", candles=8000, cache_dir="data")
```

## How to train the ML model

```bash
# everything in config.json (symbols x timeframes, config model_type):
python train_model.py

# one pair, specific model, more history:
python train_model.py --symbol BTCUSDT --timeframe 15m --model gradient_boosting --candles 10000
```

Training prints a chronological **holdout accuracy** plus **walk-forward**
fold scores, then saves one bundle per pair to
`models/{market}_{symbol}_{tf}_{model}.joblib`.

Labeling: with `neutral_threshold_pct = 0.03`, the next candle is *bullish*
if its close-to-close return is above +0.03%, *bearish* below −0.03%, else
*neutral*.

## How to run live ML prediction

```bash
# 1. set "prediction_mode": "ml" in config.json
# 2. (optional) set "retrain_on_start": true to refresh models at launch
python run_predictor.py
```

Missing models are trained automatically at startup.

## How to evaluate prediction accuracy

Backtest over history (no waiting for live candles):

```bash
python backtest_predictions.py --mode rule_based --candles 3000
python backtest_predictions.py --mode ml --symbol BTCUSDT --timeframe 15m
```

Report live results collected so far:

```python
from predictor.storage import PredictionStorage
from predictor.evaluator import report_from_storage

storage = PredictionStorage("data/predictions.db", "data/predictions.csv")
print(report_from_storage(storage))
```

Both produce accuracy, per-class precision/recall, confusion matrices,
average actual move after bullish/bearish predictions, confidence-vs-accuracy
buckets, and best/worst symbol & timeframe rankings. The raw log lives in
`data/predictions.db` and `data/predictions.csv`.

### Visual dashboard

For a browser view of the same data:

```bash
python dashboard.py                 # write dashboard.html and open it
python dashboard.py --no-open       # just write the file
python dashboard.py --output x.html # custom output path
```

`dashboard.py` renders a single self-contained `dashboard.html` (inline CSS,
inline SVG, inline JS, no external assets — works offline): a clean light-theme
fintech layout with an icon rail, KPI cards, a plain-English status banner, a
validation-health panel, model rankings, and **coin-grouped accordions**
(BTC/ETH/SOL) — expand a coin to see its per-timeframe cards, recent predictions
and confusion detail. A filterable recent-predictions table (by coin/timeframe/
result), probability bars, and confirmation modals for destructive actions round
it out. Fully responsive (desktop, tablet, mobile). Coin grouping is display-only
(BTCUSDT → BTC); the raw prediction log is never changed.

#### Live (auto-updating) dashboard

To watch it update on its own, run the predictor in one terminal and the
dashboard **server** in another — no manual refresh, no extra dependencies
(uses only the Python standard library):

```bash
# terminal 1 — generate predictions
python run_predictor.py

# terminal 2 — live web dashboard
python dashboard.py --serve                 # http://127.0.0.1:8787
python dashboard.py --serve --refresh 2     # refresh every 2 seconds
python dashboard.py --serve --port 9000     # custom port
```

The server regenerates the page from `data/predictions.db` on every request
and the page auto-refreshes on the `--refresh` interval, so it tracks new
predictions as fast as they are actually written. Note the *data itself* only
changes when a candle closes (once a minute at the fastest timeframe) or when
a prediction resolves — there is nothing new to show between those events, so
a refresh interval of a few seconds already captures every update.

#### Run it 24/7 in the cloud (remote access)

To keep the bot and dashboard running around the clock and reachable from
anywhere, see [deploy/DEPLOY.md](deploy/DEPLOY.md). It provisions a Google
Cloud VM, installs two auto-restarting `systemd` services (predictor +
dashboard), and exposes the live dashboard at `http://<VM_IP>:8080/`.

> **Binance blocks US IPs (HTTP 451).** Deploy in a non-US region (the scripts
> default to Tokyo). Google Cloud's US-only free tier will not work.

## How to add new symbols

Add any Binance symbol to `config.json`:

```json
"symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
```

For ML mode, train its models once: `python train_model.py --symbol BNBUSDT`
(or just start `run_predictor.py`, which trains missing models).

## How to add new timeframes

Add any supported interval (`1m 3m 5m 15m 30m 1h 2h 4h 1d`) to:

```json
"timeframes": ["1m", "5m", "15m", "4h"]
```

Each new timeframe automatically gets its own history, features, model and
accuracy report. Higher-timeframe context picks suitable larger intervals
automatically (`higher_timeframe_count` controls how many).

## Configuration reference (`config.json`)

| Key | Meaning |
| --- | --- |
| `symbols` / `timeframes` | pairs to predict (cartesian product) |
| `prediction_mode` | `rule_based` or `ml` |
| `market_type` | `spot` or `futures` (USDⓈ-M) — market **data** only |
| `candle_limit` | candles fetched per live prediction (≥ 200) |
| `neutral_threshold_pct` | % band that labels a candle neutral |
| `min_confidence` | predictions below this are flagged low-conviction |
| `use_higher_timeframe_context` | merge higher-TF trend features |
| `higher_timeframe_count` | how many higher TFs to use as context |
| `model_type` | `logistic_regression`, `random_forest`, `gradient_boosting`, `xgboost`, `lightgbm` |
| `train_test_split_pct` | chronological train fraction (e.g. 0.8) |
| `walk_forward_enabled` / `walk_forward_folds` | walk-forward validation |
| `train_candles` | history length used for training |
| `retrain_on_start` | retrain all models when the live loop starts |
| `save_predictions`, `sqlite_enabled`, `csv_enabled` | logging switches |
| `use_websocket` | candle-close trigger via websocket instead of polling |
| `display_utc_offset_hours` | shift displayed times off UTC (e.g. `5` = UTC+5, `5.5` = UTC+5:30). Storage stays in UTC; only display changes |
| `log_level` | console verbosity |

## Notes on look-ahead bias

- Every feature at candle *i* uses only candles ≤ *i*; labels
  (`next_return_pct`, `label`) look one candle ahead by definition and are
  kept in separate columns that are never fed to a model as features.
- Higher-TF context joins on candle **close** times with a backward
  `merge_asof`, so a 15m candle only influences 1m rows after it has closed.
- Train/test splits are strictly chronological; nothing is shuffled.
- Walk-forward validation retrains on an expanding window and tests on the
  following unseen fold.
- Model bundles store `train_end_time_ms`; ML backtests evaluate only
  candles after that cutoff (and warn loudly if they can't).

## LSTM / deep learning

Not bundled, to keep the dependency footprint small and the code auditable.
The training layer is a registry (`predictor/model_training.py:build_estimator`)
— any estimator exposing `fit` / `predict_proba` / `classes_` (e.g. a Keras
or PyTorch wrapper) can be added as a new `model_type` without touching the
rest of the pipeline.
