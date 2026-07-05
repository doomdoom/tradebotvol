"""Binance next-candle prediction bot.

Prediction-only research tool: collects market data, engineers features,
predicts the direction of the next candle (rule-based or ML), logs every
prediction and scores it against the actual outcome.

It never places orders and never talks to trading endpoints.
"""

__version__ = "1.0.0"
