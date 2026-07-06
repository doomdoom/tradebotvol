# TradeBotVol — Backtest Summary

_Generated: 2026-07-06_  
_Config: neutral_threshold=0.015%, candles/pair=3000, walk-forward folds=5, market=futures_

**Out-of-sample rows evaluated:** 21990 across 15 pairs.

## Headline: baseline vs enhanced vs majority

| Model | OOS acc | Macro-F1 | Majority floor | Coverage |
|-------|---------|----------|----------------|----------|
| Baseline (rule) | 39.2% | 30.2% | 42.8% | 100.0% |
| **Enhanced (logreg)** | 43.6% | 38.5% | 42.8% | 100.0% |

**Verdict:** Enhanced 43.6% vs baseline 39.2% vs majority-floor 42.8%. Enhanced beats BOTH baseline and the majority floor out-of-sample -> candidate for enabling (per pair where it wins).

## Per-pair (enhanced, out-of-sample)

| Pair | n | Acc | Macro-F1 | Majority | Coverage |
|------|---|-----|----------|----------|----------|
| BTCUSDT 15m | 1470 | 47.8% | 47.1% | 47.0% | 100.0% |
| BTCUSDT 1h | 1470 | 48.0% | 49.3% | 48.6% | 100.0% |
| BTCUSDT 1m | 1460 | 37.5% | 33.1% | 40.8% | 100.0% |
| BTCUSDT 3m | 1470 | 38.3% | 42.0% | 39.1% | 100.0% |
| BTCUSDT 5m | 1460 | 39.0% | 31.2% | 40.3% | 100.0% |
| ETHUSDT 15m | 1470 | 48.4% | 49.8% | 47.6% | 100.0% |
| ETHUSDT 1h | 1470 | 48.6% | 48.7% | 49.1% | 100.0% |
| ETHUSDT 1m | 1460 | 37.1% | 36.6% | 36.0% | 100.0% |
| ETHUSDT 3m | 1470 | 41.9% | 30.3% | 42.8% | 100.0% |
| ETHUSDT 5m | 1460 | 46.2% | 48.4% | 45.6% | 100.0% |
| SOLUSDT 15m | 1470 | 46.8% | 47.6% | 47.8% | 100.0% |
| SOLUSDT 1h | 1470 | 49.0% | 49.6% | 48.9% | 100.0% |
| SOLUSDT 1m | 1460 | 35.0% | 32.2% | 36.2% | 100.0% |
| SOLUSDT 3m | 1470 | 42.9% | 43.4% | 43.3% | 100.0% |
| SOLUSDT 5m | 1460 | 47.5% | 34.6% | 47.1% | 100.0% |

## Baseline: confidence calibration

| Bucket | n | Mean conf | Accuracy | Gap |
|--------|---|-----------|----------|-----|
| 50-55% | 1889 | 52.5% | 42.0% | 10.5% |
| 55-60% | 2054 | 57.4% | 40.3% | 17.1% |
| 60-65% | 2108 | 62.4% | 40.0% | 22.4% |
| 65-70% | 2189 | 67.5% | 41.4% | 26.1% |
| 70-100% | 7085 | 80.7% | 38.1% | 42.6% |

## Enhanced: confidence calibration

| Bucket | n | Mean conf | Accuracy | Gap |
|--------|---|-----------|----------|-----|
| 50-55% | 4410 | 51.9% | 46.8% | 5.1% |
| 55-60% | 1182 | 57.0% | 47.0% | 10.0% |
| 60-65% | 499 | 62.2% | 44.3% | 17.9% |
| 65-70% | 449 | 67.4% | 42.3% | 25.0% |
| 70-100% | 584 | 78.2% | 42.5% | 35.7% |

## Enhanced: selectivity curve (accuracy vs coverage)

_Only commit when calibrated confidence >= threshold._

| Min confidence | Coverage | n | Accuracy | Majority floor | Beats floor |
|----------------|----------|---|----------|----------------|-------------|
| 50% | 32.4% | 7124 | 46.0% | 44.3% | yes |
| 55% | 12.3% | 2714 | 44.8% | 43.4% | yes |
| 60% | 7.0% | 1532 | 43.0% | 42.1% | yes |
| 65% | 4.7% | 1033 | 42.4% | 42.2% | no |
| 70% | 2.7% | 584 | 42.5% | 41.1% | yes |

## Enhanced: edge by market regime (accuracy vs local floor)

| Regime | n | Accuracy | Local floor | Edge | Beats floor |
|--------|---|----------|-------------|------|-------------|
| UNCLEAR | 87 | 46.0% | 37.9% | 8.0pp | yes |
| BREAKOUT | 657 | 52.2% | 46.7% | 5.5pp | yes |
| LOW_VOLATILITY | 108 | 46.3% | 45.4% | 0.9pp | yes |
| TREND_UP | 8150 | 43.4% | 43.8% | -0.3pp | no |
| HIGH_VOLATILITY | 1392 | 45.5% | 45.9% | -0.4pp | no |
| TREND_DOWN | 6963 | 43.6% | 44.2% | -0.6pp | no |
| CHOPPY | 4612 | 42.1% | 42.9% | -0.8pp | no |
| SIDEWAYS | 21 | 33.3% | 42.9% | -9.5pp | no |

_Prediction/research only. No trades are placed. Accuracy is not a profit guarantee; short timeframes are close to a coin flip._
