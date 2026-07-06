"""Binance USDⓈ-M Futures TESTNET client - fake money, real API.

SAFETY (read this):
* This client talks ONLY to the Binance *testnet* (https://testnet.binancefuture.com).
  The base URL is hard-checked; any attempt to point it at the real exchange
  (api.binance.com / fapi.binance.com) raises immediately. It therefore cannot
  place an order against a real, funded account.
* Testnet API keys carry NO real money. Create them (free) at
  https://testnet.binancefuture.com after logging in with GitHub/Google.
* Never put your real Binance account keys here. This module will refuse the
  real endpoint even if you tried.

Only the authenticated trading endpoints needed for a simple bot are wrapped:
account balance, position, leverage, market orders and stop/take-profit closes.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from urllib.parse import urlencode

import requests

from .logger import get_logger

log = get_logger("testnet_client")

#: The ONLY base URL this client will use. Do not change to a real endpoint.
TESTNET_BASE = "https://testnet.binancefuture.com"

#: Substrings that indicate a real (non-testnet) endpoint - hard-blocked.
_FORBIDDEN = ("fapi.binance.com", "api.binance.com", "//binance.com")


class TestnetError(RuntimeError):
    pass


class BinanceTestnetClient:
    """Minimal signed REST client for Binance Futures testnet."""

    def __init__(self, api_key: str, api_secret: str, base_url: str = TESTNET_BASE,
                 timeout: float = 10.0) -> None:
        if not api_key or not api_secret:
            raise TestnetError(
                "Testnet API key/secret missing. Create free testnet keys at "
                "https://testnet.binancefuture.com and set BINANCE_TESTNET_API_KEY "
                "and BINANCE_TESTNET_API_SECRET in your .env."
            )
        if "testnet.binancefuture.com" not in base_url or any(f in base_url for f in _FORBIDDEN):
            raise TestnetError(
                f"Refusing base_url {base_url!r}: this client only trades on the "
                "Binance TESTNET. It will not touch a real account."
            )
        self._base = base_url.rstrip("/")
        self._key = api_key
        self._secret = api_secret.encode("utf-8")
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers["X-MBX-APIKEY"] = api_key
        self._time_offset = 0
        self._sync_time()

    # ------------------------------------------------------------------ #
    # low-level

    def _sync_time(self) -> None:
        try:
            server = self._public("GET", "/fapi/v1/time")["serverTime"]
            self._time_offset = int(server) - int(time.time() * 1000)
        except Exception as exc:  # non-fatal; requests may still work
            log.warning("could not sync testnet server time: %s", exc)

    def _now_ms(self) -> int:
        return int(time.time() * 1000) + self._time_offset

    def _public(self, method: str, path: str, params: dict | None = None) -> dict | list:
        resp = self._session.request(
            method, self._base + path, params=params or {}, timeout=self._timeout
        )
        return self._handle(resp)

    def _signed(self, method: str, path: str, params: dict | None = None) -> dict | list:
        p = dict(params or {})
        p["timestamp"] = self._now_ms()
        p.setdefault("recvWindow", 5000)
        query = urlencode(p)
        sig = hmac.new(self._secret, query.encode("utf-8"), hashlib.sha256).hexdigest()
        url = f"{self._base}{path}?{query}&signature={sig}"
        resp = self._session.request(method, url, timeout=self._timeout)
        return self._handle(resp)

    def _handle(self, resp: requests.Response):
        if resp.status_code != 200:
            raise TestnetError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        if isinstance(data, dict) and data.get("code") not in (None, 200):
            # Binance returns {"code":-XXXX,"msg":"..."} on errors.
            raise TestnetError(f"Binance error {data.get('code')}: {data.get('msg')}")
        return data

    # ------------------------------------------------------------------ #
    # market data / account

    def exchange_filters(self) -> dict[str, dict]:
        """symbol -> {qty_precision, price_precision, step, tick} from exchangeInfo."""
        info = self._public("GET", "/fapi/v1/exchangeInfo")
        out: dict[str, dict] = {}
        for s in info.get("symbols", []):
            step = tick = None
            for f in s.get("filters", []):
                if f["filterType"] == "LOT_SIZE":
                    step = float(f["stepSize"])
                elif f["filterType"] == "PRICE_FILTER":
                    tick = float(f["tickSize"])
            out[s["symbol"]] = {
                "qty_precision": int(s.get("quantityPrecision", 3)),
                "price_precision": int(s.get("pricePrecision", 2)),
                "step": step, "tick": tick,
            }
        return out

    def price(self, symbol: str) -> float:
        d = self._public("GET", "/fapi/v1/ticker/price", {"symbol": symbol.upper()})
        return float(d["price"])

    def usdt_balance(self) -> float:
        for a in self._signed("GET", "/fapi/v2/balance"):
            if a.get("asset") == "USDT":
                return float(a["balance"])
        return 0.0

    def position(self, symbol: str) -> dict:
        rows = self._signed("GET", "/fapi/v2/positionRisk", {"symbol": symbol.upper()})
        row = rows[0] if rows else {}
        amt = float(row.get("positionAmt", 0) or 0)
        return {
            "amount": amt,
            "flat": abs(amt) < 1e-12,
            "entry": float(row.get("entryPrice", 0) or 0),
            "unrealized": float(row.get("unRealizedProfit", 0) or 0),
            "leverage": float(row.get("leverage", 0) or 0),
        }

    # ------------------------------------------------------------------ #
    # trading

    def set_leverage(self, symbol: str, leverage: int) -> dict:
        return self._signed("POST", "/fapi/v1/leverage",
                            {"symbol": symbol.upper(), "leverage": int(leverage)})

    def market_order(self, symbol: str, side: str, quantity: float) -> dict:
        return self._signed("POST", "/fapi/v1/order", {
            "symbol": symbol.upper(), "side": side.upper(),
            "type": "MARKET", "quantity": quantity,
        })

    def market_close(self, symbol: str, side: str, quantity: float) -> dict:
        """Close (reduce-only) a position with a plain MARKET order. Used for
        client-side stop-loss / trailing / take-profit, because this testnet
        rejects conditional STOP/TP/TRAILING order types (error -4120)."""
        return self._signed("POST", "/fapi/v1/order", {
            "symbol": symbol.upper(), "side": side.upper(),
            "type": "MARKET", "quantity": quantity, "reduceOnly": "true",
        })

    def close_trigger(self, symbol: str, side: str, stop_price: float,
                      kind: str) -> dict:
        """Place a STOP_MARKET (stop-loss) or TAKE_PROFIT_MARKET that closes the
        whole position when stop_price is reached. ``side`` is the closing side
        (SELL to close a long, BUY to close a short)."""
        order_type = "STOP_MARKET" if kind == "stop" else "TAKE_PROFIT_MARKET"
        return self._signed("POST", "/fapi/v1/order", {
            "symbol": symbol.upper(), "side": side.upper(), "type": order_type,
            "stopPrice": stop_price, "closePosition": "true", "workingType": "MARK_PRICE",
        })

    def income(self, income_type: str | None = None, symbol: str | None = None,
               limit: int = 100) -> list:
        """Account income history (realized P&L, commission, funding). Used to
        build the closed-trade history: filter income_type='REALIZED_PNL'."""
        params: dict = {"limit": min(int(limit), 1000)}
        if income_type:
            params["incomeType"] = income_type
        if symbol:
            params["symbol"] = symbol.upper()
        result = self._signed("GET", "/fapi/v1/income", params)
        return result if isinstance(result, list) else []

    def trailing_stop(self, symbol: str, side: str, quantity: float,
                      callback_rate: float, activation_price: float | None = None) -> dict:
        """Trailing-stop that books profit: it follows the price in your favour
        and closes when price retraces by ``callback_rate`` percent from its best.
        ``side`` is the closing side (SELL to close a long, BUY to close a short).
        ``activation_price`` (optional) is where trailing starts."""
        params: dict = {
            "symbol": symbol.upper(), "side": side.upper(),
            "type": "TRAILING_STOP_MARKET", "quantity": quantity,
            "callbackRate": round(max(0.1, min(callback_rate, 5.0)), 1),
            "reduceOnly": "true", "workingType": "MARK_PRICE",
        }
        if activation_price is not None:
            params["activationPrice"] = activation_price
        return self._signed("POST", "/fapi/v1/order", params)

    def cancel_all(self, symbol: str) -> dict:
        return self._signed("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol.upper()})
