import os
import time
import json
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
import ccxt
from typing import Optional, Tuple
from decimal import Decimal, ROUND_DOWN, ROUND_UP

STATE_FILE = "state.json"


# ---------------------------
# Utils / Indicators
# ---------------------------
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    tr = true_range(df["high"], df["low"], df["close"])
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger(close: pd.Series, period: int, std_mult: float):
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    upper = ma + std_mult * sd
    lower = ma - std_mult * sd
    return ma, upper, lower


def adx(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(high, low, close)

    atr_w = pd.Series(tr, index=df.index).ewm(alpha=1 / period, adjust=False).mean()
    plus_dm_w = pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean()
    minus_dm_w = pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * (plus_dm_w / atr_w.replace(0, np.nan))
    minus_di = 100 * (minus_dm_w / atr_w.replace(0, np.nan))

    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(alpha=1 / period, adjust=False).mean()


# ---------------------------
# Config / State
# ---------------------------
def load_config(path="bot_config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {
            "last_candle_ts": None,
            "position": {
                "side": "flat",          # long | short | flat
                "amount": 0.0,           # BTC amount
                "entry_price": None,
                "stop_price": None,
                "take_profit": None,
                "mode": None,            # trend | range
            },
            "account": {
                "equity_usdt": 10000.0,
                "daily_start_equity": 10000.0,
                "daily_date": None,
            },
            "logs": [],
        }

    with open(STATE_FILE, "r", encoding="utf-8") as f:
        s = json.load(f)

    if "account" not in s and "paper" in s:
        s["account"] = s.pop("paper")

    if "take_profit" not in s.get("position", {}):
        s["position"]["take_profit"] = None

    return s


def save_state(state: dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def log(state: dict, msg: str):
    line = f"[{now_utc_iso()}] {msg}"
    print(line)
    state["logs"].append(line)
    state["logs"] = state["logs"][-600:]


# ---------------------------
# Exchange (Bybit via CCXT) + Demo Trading
# ---------------------------
def make_exchange(cfg: dict):
    ex_cls = getattr(ccxt, cfg["exchange_id"])

    opts = {
        "enableRateLimit": True,
        "options": {"defaultType": cfg.get("market_type", "swap")},
    }

    if cfg["mode"] == "live":
        load_dotenv()
        key = os.getenv("BYBIT_API_KEY", "")
        secret = os.getenv("BYBIT_API_SECRET", "")
        if not key or not secret:
            raise RuntimeError("Missing BYBIT_API_KEY / BYBIT_API_SECRET in .env.")
        opts["apiKey"] = key
        opts["secret"] = secret

    ex = ex_cls(opts)

    # Demo Trading domain (NOT testnet)
    if cfg.get("demo_trading", False):
        if hasattr(ex, "enable_demo_trading"):
            try:
                ex.enable_demo_trading(True)
            except Exception:
                pass
        try:
            ex.set_sandbox_mode(True)
        except Exception:
            pass

        ex.urls["api"] = {
            "public": "https://api.bybit.com",
            "private": "https://api.bybit.com",
        }

    # Testnet (don’t mix with demo_trading)
    if cfg.get("testnet", False) and not cfg.get("demo_trading", False):
        ex.set_sandbox_mode(True)

    ex.load_markets()

    # Auto-fix symbol if needed
    sym = cfg["symbol"]
    if sym not in ex.markets:
        alt = f"{sym}:USDT" if ":" not in sym else None
        if alt and alt in ex.markets:
            cfg["symbol"] = alt

    return ex


def apply_min_order_from_market(exchange, cfg: dict, state: dict):
    try:
        m = exchange.market(cfg["symbol"])
    except Exception as e:
        log(state, f"Min order fetch failed (continuing): {repr(e)}")
        return

    limits = m.get("limits") or {}
    amt_min = (limits.get("amount") or {}).get("min")
    if amt_min is None:
        return

    try:
        cfg["min_order_btc"] = float(amt_min)
        log(state, f"Min order size set from market: {cfg['min_order_btc']}")
    except Exception:
        return


def _perm_has_trade(perms_list) -> bool:
    if not isinstance(perms_list, list):
        return False
    for p in perms_list:
        if not isinstance(p, str):
            continue
        lp = p.lower()
        if "trade" in lp or "order" in lp or "place" in lp:
            return True
    return False


def check_trade_permissions(exchange, cfg: dict, state: dict):
    if cfg.get("mode") != "live" or cfg.get("demo_trading", False):
        return

    method = None
    for name in ("privateGetV5UserQueryApi", "private_get_v5_user_query_api"):
        if hasattr(exchange, name):
            method = getattr(exchange, name)
            break

    if method is None:
        log(state, "Permission check skipped (query-api not available in ccxt).")
        return

    try:
        resp = method({})
    except Exception as e:
        log(state, f"Permission check failed (continuing): {repr(e)}")
        return

    result = resp.get("result", {}) or {}
    read_only = result.get("readOnly")
    if str(read_only) == "1" or read_only is True:
        raise RuntimeError("API key is read-only. Enable Trade permission in Bybit API settings.")

    perms = result.get("permissions", {}) or {}
    contract_perms = perms.get("ContractTrade")
    deriv_perms = perms.get("Derivatives")

    if contract_perms is not None and not _perm_has_trade(contract_perms):
        raise RuntimeError("API key missing ContractTrade trade permission in Bybit API settings.")
    if deriv_perms is not None and not _perm_has_trade(deriv_perms):
        raise RuntimeError("API key missing Derivatives trade permission in Bybit API settings.")

    log(state, "API key permissions OK (trade enabled).")


def round_to_precision(exchange, symbol: str, price=None, amount=None):
    if price is not None:
        price = float(exchange.price_to_precision(symbol, price))
    if amount is not None:
        amount = float(exchange.amount_to_precision(symbol, amount))
    return price, amount


def position_idx_for_side(cfg: dict, side: str) -> int:
    if not cfg.get("hedge_mode", False):
        return int(cfg.get("position_idx", 0))
    if side == "long":
        return int(cfg.get("long_position_idx", 1))
    if side == "short":
        return int(cfg.get("short_position_idx", 2))
    return int(cfg.get("position_idx", 0))


def fetch_ohlcv_df(exchange, symbol: str, timeframe: str, limit=650) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def get_usdt_equity(exchange, cfg: dict, state: dict) -> float:
    if cfg["mode"] != "live":
        return float(state["account"]["equity_usdt"])

    try:
        bal = exchange.fetch_balance({"type": cfg.get("market_type", "swap")})

        eq = safe_float(bal.get("total", {}).get("USDT", np.nan))
        if not np.isnan(eq):
            return float(eq)

        eq = safe_float(bal.get("free", {}).get("USDT", np.nan))
        if not np.isnan(eq):
            return float(eq)

        info = bal.get("info", {}) or {}
        eq = safe_float(info.get("totalEquity", np.nan))
        if not np.isnan(eq):
            return float(eq)

    except Exception as e:
        log(state, f"fetch_balance failed (fallback equity). err={repr(e)}")

    return float(cfg.get("demo_equity_usdt", state["account"].get("equity_usdt", 10000.0)))


# ---------------------------
# Tick size (robust): prefer Bybit instruments-info
# ---------------------------
def bybit_symbol_id(exchange, symbol: str) -> str:
    m = exchange.market(symbol)
    return m.get("id", symbol.replace("/", "").replace(":USDT", ""))


def _try_tick_from_market(exchange, symbol: str) -> Optional[Decimal]:
    m = exchange.market(symbol)
    info = m.get("info", {}) or {}

    tick = None
    pf = info.get("priceFilter") or info.get("price_filter") or {}
    tick = pf.get("tickSize") or pf.get("tick_size") or tick
    tick = info.get("tickSize") or info.get("tick_size") or tick

    if tick is not None:
        try:
            return Decimal(str(tick))
        except Exception:
            return None

    prec = (m.get("precision") or {}).get("price")
    if isinstance(prec, int) and prec >= 0:
        return Decimal("1") / (Decimal(10) ** prec)

    return None


def _try_tick_from_instruments(exchange, cfg: dict, symbol: str) -> Optional[Decimal]:
    """
    Calls Bybit V5 instruments info to get tickSize.
    Works even when CCXT market precision is wrong.
    """
    category = cfg.get("bybit_category", "linear")
    sym_id = bybit_symbol_id(exchange, symbol)

    for method_name in ("publicGetV5MarketInstrumentsInfo", "public_get_v5_market_instruments_info"):
        if hasattr(exchange, method_name):
            try:
                resp = getattr(exchange, method_name)({"category": category, "symbol": sym_id})
                result = resp.get("result", {}) or {}
                lst = result.get("list", []) or []
                if lst:
                    price_filter = lst[0].get("priceFilter", {}) or {}
                    ts = price_filter.get("tickSize")
                    if ts is not None:
                        return Decimal(str(ts))
            except Exception:
                return None
    return None


def get_tick_size(exchange, cfg: dict, symbol: str) -> Decimal:
    t = _try_tick_from_instruments(exchange, cfg, symbol)
    if t is not None and t > 0:
        return t

    t = _try_tick_from_market(exchange, symbol)
    if t is not None and t > 0:
        return t

    # Final fallback for BTCUSDT perps
    return Decimal("0.1")


def quantize_to_tick(price: float, tick: Decimal, round_up: bool) -> str:
    p = Decimal(str(price))
    if tick <= 0:
        return format(p, "f")
    units = (p / tick).to_integral_value(rounding=ROUND_UP if round_up else ROUND_DOWN)
    q = units * tick
    return format(q, "f")


# ---------------------------
# Strategy: regime + signals
# ---------------------------
def detect_regime(df: pd.DataFrame, ema_trend_period: int) -> str:
    df = df.copy()
    df["ema_trend"] = ema(df["close"], ema_trend_period)
    N = 10
    if len(df) < ema_trend_period + N + 5:
        return "range"
    slope = (df["ema_trend"].iloc[-1] - df["ema_trend"].iloc[-N]) / df["ema_trend"].iloc[-N]
    return "trend" if abs(slope) > 0.003 else "range"


def is_clean_range(df: pd.DataFrame, cfg: dict) -> bool:
    df = df.copy()
    df["ema_trend"] = ema(df["close"], cfg["ema_trend"])
    df["atr"] = atr(df, cfg["atr_period"])
    df["adx"] = adx(df, int(cfg.get("adx_period", 14)))

    mid, upper, lower = bollinger(df["close"], cfg["bb_period"], cfg["bb_std"])
    df["bb_mid"] = mid
    df["bb_upper"] = upper
    df["bb_lower"] = lower
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)

    lookback = int(cfg.get("bb_width_lookback", 200))
    q = float(cfg.get("bb_width_quantile", 0.45))
    if len(df) < lookback + 50:
        return False

    last = df.iloc[-1]
    close = float(last["close"])
    ema200 = float(last["ema_trend"])
    atr_v = float(last["atr"])
    adx_v = float(last["adx"])
    width = float(last["bb_width"])
    width_q = float(df["bb_width"].rolling(lookback).quantile(q).iloc[-1])

    adx_ok = adx_v <= float(cfg.get("adx_max_for_range", 16))
    width_ok = width <= width_q
    ema_dist_ok = abs(close - ema200) / ema200 <= float(cfg.get("ema_trend_distance_max", 0.010))
    atr_ok = (atr_v / close) <= float(cfg.get("atr_pct_max_for_range", 0.030))

    return bool(adx_ok and width_ok and ema_dist_ok and atr_ok)


def trend_signal(df: pd.DataFrame, cfg: dict):
    df = df.copy()
    df["ema_fast"] = ema(df["close"], cfg["ema_fast"])
    df["ema_slow"] = ema(df["close"], cfg["ema_slow"])
    df["ema_trend"] = ema(df["close"], cfg["ema_trend"])
    df["atr"] = atr(df, cfg["atr_period"])
    last = df.iloc[-1]

    close = float(last["close"])
    atr_v = float(last["atr"])

    long_ok = (close > last["ema_trend"]) and (last["ema_fast"] > last["ema_slow"])
    short_ok = (close < last["ema_trend"]) and (last["ema_fast"] < last["ema_slow"])

    if long_ok:
        return {"side": "long", "atr": atr_v, "price": close}
    if short_ok:
        return {"side": "short", "atr": atr_v, "price": close}
    return {"side": "flat", "atr": atr_v, "price": close}


def _recent_swing_low(df: pd.DataFrame, lookback: int) -> Optional[float]:
    lows = df["low"].to_numpy()
    n = len(lows)
    if n < 3:
        return None
    start = max(0, n - lookback)
    for i in range(n - 2, start, -1):
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            return float(lows[i])
    if start < n:
        return float(np.min(lows[start:]))
    return None


def _recent_swing_high(df: pd.DataFrame, lookback: int) -> Optional[float]:
    highs = df["high"].to_numpy()
    n = len(highs)
    if n < 3:
        return None
    start = max(0, n - lookback)
    for i in range(n - 2, start, -1):
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            return float(highs[i])
    if start < n:
        return float(np.max(highs[start:]))
    return None


def structure_stop(df: pd.DataFrame, cfg: dict, side: str, atr_v: float, price: float) -> Optional[float]:
    lookback = int(cfg.get("structure_lookback", 50))
    atr_mult = float(cfg.get("structure_buffer_atr_mult", 0.7))
    pct = float(cfg.get("structure_buffer_pct", 0.002))
    buffer_amt = max(atr_mult * float(atr_v), pct * float(price))

    if side == "long":
        swing = _recent_swing_low(df, lookback)
        if swing is None:
            return None
        stop = swing - buffer_amt
    elif side == "short":
        swing = _recent_swing_high(df, lookback)
        if swing is None:
            return None
        stop = swing + buffer_amt
    else:
        return None

    return float(stop) if stop and stop > 0 else None


def range_signal(df: pd.DataFrame, cfg: dict):
    df = df.copy()
    df["atr"] = atr(df, cfg["atr_period"])
    df["rsi"] = rsi(df["close"], cfg["rsi_period"])
    mid, upper, lower = bollinger(df["close"], cfg["bb_period"], cfg["bb_std"])
    df["bb_mid"] = mid
    df["bb_upper"] = upper
    df["bb_lower"] = lower

    last = df.iloc[-1]
    close = float(last["close"])
    atr_v = float(last["atr"])
    r = float(last["rsi"])
    bb_upper = float(last["bb_upper"])
    bb_lower = float(last["bb_lower"])
    bb_mid = float(last["bb_mid"])

    if close < bb_lower and r < cfg["rsi_buy"]:
        sl = close - cfg["sl_atr_mult"] * atr_v
        tp = close + cfg["tp_rr"] * (close - sl)
        tp = min(tp, bb_mid)
        return {"side": "long", "atr": atr_v, "price": close, "sl": sl, "tp": tp}

    if close > bb_upper and r > cfg["rsi_sell"]:
        sl = close + cfg["sl_atr_mult"] * atr_v
        tp = close - cfg["tp_rr"] * (sl - close)
        tp = max(tp, bb_mid)
        return {"side": "short", "atr": atr_v, "price": close, "sl": sl, "tp": tp}

    return {"side": "flat", "atr": atr_v, "price": close, "sl": None, "tp": None}


# ---------------------------
# Risk sizing
# ---------------------------
def compute_position_size_btc(equity_usdt: float, entry_price: float, stop_price: float, cfg: dict):
    risk_usdt = equity_usdt * float(cfg["risk_per_trade"])
    stop_dist = abs(entry_price - stop_price)
    if stop_dist <= 0:
        return 0.0, False, 0.0, 0.0

    base_btc = risk_usdt / stop_dist
    btc_amt = base_btc
    min_order_btc = float(cfg.get("min_order_btc", 0.0))
    fallback_pct = float(cfg.get("fallback_min_order_pct", 0.05))
    used_fallback = False
    if min_order_btc > 0 and btc_amt < min_order_btc:
        fallback_notional = equity_usdt * fallback_pct
        btc_amt = max(min_order_btc, fallback_notional / entry_price)
        used_fallback = True

    notional = btc_amt * entry_price
    if notional > float(cfg["max_position_usdt"]):
        btc_amt = float(cfg["max_position_usdt"]) / entry_price

    return max(0.0, btc_amt), used_fallback, base_btc, btc_amt


# ---------------------------
# Live execution helpers
# ---------------------------
def set_isolated_best_effort(exchange, symbol: str, position_idx: int):
    try:
        exchange.set_margin_mode("isolated", symbol, {"positionIdx": position_idx})
    except Exception:
        try:
            exchange.set_margin_mode("isolated", symbol)
        except Exception:
            pass


def set_leverage_best_effort(exchange, symbol: str, leverage: int, position_idx: int):
    try:
        exchange.set_leverage(leverage, symbol, {"positionIdx": position_idx})
    except Exception:
        try:
            exchange.set_leverage(leverage, symbol)
        except Exception:
            pass


def live_place_market(exchange, symbol: str, side: str, amount: float, reduce_only: bool, position_idx: int):
    order_side = "buy" if side == "long" else "sell"
    params = {"reduceOnly": reduce_only, "positionIdx": int(position_idx)}
    return exchange.create_order(symbol, "market", order_side, amount, None, params)


# ---------------------------
# Bybit exchange-side TP/SL (V5 trading-stop)
# ---------------------------
def ccxt_bybit_set_trading_stop(
    exchange,
    cfg: dict,
    symbol: str,
    stop_loss: Optional[float],
    take_profit: Optional[float],
    position_idx: int,
    position_side: str,  # "long" or "short"
    state: dict,
):
    category = cfg.get("bybit_category", "linear")
    tpsl_mode = cfg.get("tpsl_mode", "Full")
    tp_trigger_by = cfg.get("tp_trigger_by", "MarkPrice")
    sl_trigger_by = cfg.get("sl_trigger_by", "MarkPrice")

    sym_id = bybit_symbol_id(exchange, symbol)
    tick = get_tick_size(exchange, cfg, symbol)

    payload = {
        "category": category,
        "symbol": sym_id,
        "tpslMode": tpsl_mode,
        "positionIdx": int(position_idx),
        "tpTriggerBy": tp_trigger_by,
        "slTriggerBy": sl_trigger_by,
    }

    # SL rounding: long SL down, short SL up
    if stop_loss is not None:
        if float(stop_loss) == 0.0:
            payload["stopLoss"] = "0"
        else:
            sl_round_up = (position_side == "short")
            payload["stopLoss"] = quantize_to_tick(float(stop_loss), tick, round_up=sl_round_up)

    # TP rounding: long TP up, short TP down
    if take_profit is not None:
        if float(take_profit) == 0.0:
            payload["takeProfit"] = "0"
        else:
            tp_round_up = (position_side == "long")
            payload["takeProfit"] = quantize_to_tick(float(take_profit), tick, round_up=tp_round_up)

    if cfg.get("debug_tpsl", False):
        log(state, f"trading-stop payload tick={tick} -> {payload}")

    for method_name in ("privatePostV5PositionTradingStop", "private_post_v5_position_trading_stop"):
        if hasattr(exchange, method_name):
            return getattr(exchange, method_name)(payload)

    return exchange.request("v5/position/trading-stop", "private", "POST", payload)


def sync_exchange_side_tpsl(exchange, cfg: dict, state: dict):
    if cfg["mode"] != "live":
        return

    pos = state["position"]
    if pos["side"] == "flat":
        return

    position_idx = position_idx_for_side(cfg, pos["side"])
    sl = pos.get("stop_price")

    if pos.get("mode") == "range":
        tp = pos.get("take_profit")
        if sl is None or tp is None:
            return
        ccxt_bybit_set_trading_stop(exchange, cfg, cfg["symbol"], float(sl), float(tp), position_idx, pos["side"], state)
    else:
        if sl is None:
            return
        # trend: cancel TP by sending 0
        ccxt_bybit_set_trading_stop(exchange, cfg, cfg["symbol"], float(sl), 0.0, position_idx, pos["side"], state)


# ---------------------------
# Live position fetch + reconcile on startup
# ---------------------------
def _position_to_side(p: dict) -> str:
    side = (p.get("side") or "").lower()
    if side in ("long", "short"):
        return side
    info = p.get("info", {}) or {}
    raw = (info.get("side") or info.get("positionSide") or "").lower()
    if raw in ("buy", "long"):
        return "long"
    if raw in ("sell", "short"):
        return "short"
    return "flat"


def _position_amount_btc(exchange, symbol: str, p: dict) -> float:
    market = exchange.market(symbol)
    contract_size = safe_float(p.get("contractSize"), np.nan)
    if np.isnan(contract_size):
        contract_size = safe_float(market.get("contractSize"), 1.0)

    contracts = p.get("contracts")
    if contracts is not None:
        return abs(float(contracts)) * float(contract_size)

    size = p.get("size")
    if size is not None:
        return abs(float(size))

    info = p.get("info", {}) or {}
    for k in ("size", "positionAmt", "qty"):
        if k in info and info[k] is not None:
            return abs(float(info[k]))

    return 0.0


def _position_entry_price(p: dict) -> float:
    for k in ("entryPrice", "average", "avgPrice", "entry_price"):
        v = p.get(k)
        if v is not None:
            fv = safe_float(v, np.nan)
            if not np.isnan(fv) and fv > 0:
                return float(fv)
    info = p.get("info", {}) or {}
    for k in ("avgPrice", "entryPrice", "entry_price"):
        if k in info and info[k] is not None:
            fv = safe_float(info[k], np.nan)
            if not np.isnan(fv) and fv > 0:
                return float(fv)
    return np.nan


def fetch_live_position(exchange, cfg: dict) -> Optional[dict]:
    symbol = cfg["symbol"]
    market_type = cfg.get("market_type", "swap")

    try:
        positions = exchange.fetch_positions([symbol], {"type": market_type})
    except Exception:
        positions = exchange.fetch_positions()

    for p in positions:
        if p.get("symbol") != symbol:
            continue
        side = _position_to_side(p)
        amt = _position_amount_btc(exchange, symbol, p)
        if side in ("long", "short") and amt > 0:
            return p
    return None


def reconcile_live_position_on_start(exchange, cfg: dict, state: dict):
    if cfg["mode"] != "live":
        return

    log(state, "Reconciling live position from Bybit...")
    df = fetch_ohlcv_df(exchange, cfg["symbol"], cfg["timeframe"], limit=650)

    p = fetch_live_position(exchange, cfg)
    if p is None:
        if state["position"]["side"] != "flat":
            log(state, "No live position found. Clearing local state -> flat.")
            state["position"] = {"side": "flat", "amount": 0.0, "entry_price": None, "stop_price": None, "take_profit": None, "mode": None}
            save_state(state)
        else:
            log(state, "No live position found. State already flat.")
        return

    live_side = _position_to_side(p)
    live_amt = _position_amount_btc(exchange, cfg["symbol"], p)
    live_entry = _position_entry_price(p)
    if np.isnan(live_entry):
        live_entry = float(df["close"].iloc[-1])

    # rebuild a trend stop using structure + buffer (fallback to ATR if needed)
    ts = trend_signal(df, cfg)
    atr_v = float(ts["atr"])
    last_price = float(df["close"].iloc[-1])
    sl = structure_stop(df, cfg, live_side, atr_v, last_price)
    if sl is None:
        mult = float(cfg["atr_stop_mult"])
        sl = last_price - mult * atr_v if live_side == "long" else last_price + mult * atr_v

    state["position"] = {
        "side": live_side,
        "amount": float(live_amt),
        "entry_price": float(live_entry),
        "stop_price": float(sl),
        "take_profit": None,
        "mode": "trend",
    }

    log(state, f"Reconciled LIVE -> side={live_side} amt={live_amt:.6f} entry={live_entry:.2f} sl={sl} tp=None mode=trend")
    save_state(state)

    try:
        sync_exchange_side_tpsl(exchange, cfg, state)
        log(state, "Exchange-side TP/SL synced on startup.")
    except Exception as e:
        log(state, f"Exchange-side TP/SL sync failed on startup (continuing): {repr(e)}")


def reconcile_live_position_each_candle(exchange, cfg: dict, state: dict):
    if cfg["mode"] != "live":
        return
    p = fetch_live_position(exchange, cfg)
    if p is None and state["position"]["side"] != "flat":
        log(state, "Position closed on exchange. Clearing local state -> flat.")
        state["position"] = {"side": "flat", "amount": 0.0, "entry_price": None, "stop_price": None, "take_profit": None, "mode": None}
        save_state(state)


# ---------------------------
# Risk controls
# ---------------------------
def reset_daily_if_needed(state: dict, equity_now: float):
    today = datetime.now(timezone.utc).date().isoformat()
    if state["account"]["daily_date"] != today:
        state["account"]["daily_date"] = today
        state["account"]["daily_start_equity"] = float(equity_now)
        log(state, f"Daily reset. daily_start_equity={state['account']['daily_start_equity']:.2f}")


def daily_loss_hit(state: dict, cfg: dict, equity_now: float) -> bool:
    start = float(state["account"]["daily_start_equity"])
    if start <= 0:
        return True
    dd = (start - float(equity_now)) / start
    return dd >= float(cfg["daily_loss_limit_pct"])


# ---------------------------
# Stops & trailing
# ---------------------------
def update_trailing_stop_trend(pos: dict, df: pd.DataFrame, last_price: float, atr_v: float, cfg: dict) -> Optional[float]:
    new_stop = structure_stop(df, cfg, pos.get("side", ""), atr_v, last_price)
    if new_stop is None:
        return None
    if pos["side"] == "long":
        return new_stop if pos["stop_price"] is None else max(float(pos["stop_price"]), new_stop)
    if pos["side"] == "short":
        return new_stop if pos["stop_price"] is None else min(float(pos["stop_price"]), new_stop)
    return None


def should_stop_out(pos: dict, last_price: float) -> bool:
    if pos["stop_price"] is None:
        return False
    stop = float(pos["stop_price"])
    if pos["side"] == "long":
        return last_price <= stop
    if pos["side"] == "short":
        return last_price >= stop
    return False


# ---------------------------
# Main candle step
# ---------------------------
def run_once(exchange, cfg: dict, state: dict):
    symbol = cfg["symbol"]
    timeframe = cfg["timeframe"]

    df = fetch_ohlcv_df(exchange, symbol, timeframe=timeframe, limit=650)
    if len(df) < 300:
        log(state, "Not enough candles yet.")
        return

    last_candle_ts = df["ts"].iloc[-1]
    last_candle_ts_ms = int(last_candle_ts.value / 1_000_000)

    if state["last_candle_ts"] == last_candle_ts_ms:
        return
    state["last_candle_ts"] = last_candle_ts_ms

    last_price = float(df["close"].iloc[-1])

    equity_now = get_usdt_equity(exchange, cfg, state)
    reset_daily_if_needed(state, equity_now)
    kill_new_entries = daily_loss_hit(state, cfg, equity_now)

    reconcile_live_position_each_candle(exchange, cfg, state)

    regime = detect_regime(df, cfg["ema_trend"])
    clean_range = is_clean_range(df, cfg)
    pos = state["position"]

    log(state, f"New candle. price={last_price:.2f} regime={regime} clean_range={clean_range} pos={pos['side']} mode={pos.get('mode')} equity={equity_now:.2f}")

    # daily loss limit: close any open position and stop trading
    if kill_new_entries and pos["side"] != "flat":
        amount = float(pos["amount"])
        close_side = "short" if pos["side"] == "long" else "long"
        try:
            position_idx = position_idx_for_side(cfg, pos["side"])
            live_place_market(exchange, symbol, close_side, amount, reduce_only=True, position_idx=position_idx)
            log(state, f"DAILY LOSS LIMIT CLOSE {pos['side']} amt={amount:.6f}")
        except Exception as e:
            log(state, f"Failed to close on daily loss limit (continuing): {repr(e)}")
        state["position"] = {"side": "flat", "amount": 0.0, "entry_price": None, "stop_price": None, "take_profit": None, "mode": None}
        save_state(state)
        return

    # manage open position
    if pos["side"] != "flat":
        if pos.get("mode") == "trend":
            tsig = trend_signal(df, cfg)
            new_stop = update_trailing_stop_trend(pos, df, last_price, tsig["atr"], cfg)
            if new_stop is not None:
                pos["stop_price"] = float(new_stop)
                try:
                    sync_exchange_side_tpsl(exchange, cfg, state)
                    log(state, f"Updated exchange-side trailing SL to {pos['stop_price']}")
                except Exception as e:
                    log(state, f"Failed to update exchange-side SL (continuing): {repr(e)}")

        # bot-side safety stop
        if should_stop_out(pos, last_price):
            amount = float(pos["amount"])
            close_side = "short" if pos["side"] == "long" else "long"
            position_idx = position_idx_for_side(cfg, pos["side"])
            live_place_market(exchange, symbol, close_side, amount, reduce_only=True, position_idx=position_idx)
            log(state, f"LIVE STOP CLOSE {pos['side']} amt={amount:.6f}")
            state["position"] = {"side": "flat", "amount": 0.0, "entry_price": None, "stop_price": None, "take_profit": None, "mode": None}
            save_state(state)
            return

    # entries only if flat
    if pos["side"] == "flat" and not kill_new_entries:
        if regime == "trend":
            tsig = trend_signal(df, cfg)
            if tsig["side"] in ("long", "short"):
                stop_price = structure_stop(df, cfg, tsig["side"], tsig["atr"], tsig["price"])
                if stop_price is None:
                    stop_price = (
                        tsig["price"] - float(cfg["atr_stop_mult"]) * tsig["atr"]
                        if tsig["side"] == "long"
                        else tsig["price"] + float(cfg["atr_stop_mult"]) * tsig["atr"]
                    )

                amount, used_fallback, base_btc, final_btc = compute_position_size_btc(
                    equity_now, tsig["price"], stop_price, cfg
                )
                if amount <= 0:
                    return
                position_idx = position_idx_for_side(cfg, tsig["side"])

                trailing_amt = abs(float(tsig["price"]) - float(stop_price))
                log(
                    state,
                    f"SETUP {symbol} {tsig['side']} entry={tsig['price']:.2f} SL={stop_price:.2f} "
                    f"TRAILING_STOP={trailing_amt:.2f} (trend, no TP)",
                )

                set_isolated_best_effort(exchange, symbol, position_idx)
                set_leverage_best_effort(exchange, symbol, int(cfg["max_leverage"]), position_idx)

                _, amount_r = round_to_precision(exchange, symbol, amount=amount)
                if amount_r <= 0:
                    log(
                        state,
                        f"Size rounded to 0. base_btc={base_btc:.8f} final_btc={final_btc:.8f} "
                        f"min_order_btc={cfg.get('min_order_btc')} fallback_used={used_fallback}",
                    )
                    return
                if used_fallback:
                    log(
                        state,
                        f"Fallback sizing used. base_btc={base_btc:.8f} final_btc={final_btc:.8f} "
                        f"min_order_btc={cfg.get('min_order_btc')} fallback_pct={cfg.get('fallback_min_order_pct')}",
                    )
                try:
                    live_place_market(exchange, symbol, tsig["side"], amount_r, reduce_only=False, position_idx=position_idx)
                except Exception as e:
                    log(state, f"Order failed (trend entry). err={repr(e)}")
                    return

                state["position"] = {
                    "side": tsig["side"],
                    "amount": float(amount_r),
                    "entry_price": float(tsig["price"]),
                    "stop_price": float(stop_price),
                    "take_profit": None,
                    "mode": "trend",
                }
                log(state, f"LIVE OPEN {tsig['side']} amt={amount_r:.6f} entry={tsig['price']:.2f} stop={stop_price:.2f} mode=trend")

                try:
                    sync_exchange_side_tpsl(exchange, cfg, state)
                    log(state, f"Exchange-side SL set (TP cancelled). SL={state['position']['stop_price']}")
                except Exception as e:
                    log(state, f"Failed to set exchange-side SL (continuing): {repr(e)}")

                save_state(state)
                return
        if regime == "range" and clean_range:
            rsig = range_signal(df, cfg)
            if rsig["side"] in ("long", "short"):
                stop_price = float(rsig["sl"])
                take_profit = float(rsig["tp"])

                amount, used_fallback, base_btc, final_btc = compute_position_size_btc(
                    equity_now, rsig["price"], stop_price, cfg
                )
                if amount <= 0:
                    return
                position_idx = position_idx_for_side(cfg, rsig["side"])

                log(
                    state,
                    f"SETUP {symbol} {rsig['side']} entry={rsig['price']:.2f} TP={take_profit:.2f} SL={stop_price:.2f} (range)",
                )

                set_isolated_best_effort(exchange, symbol, position_idx)
                set_leverage_best_effort(exchange, symbol, int(cfg["max_leverage"]), position_idx)

                _, amount_r = round_to_precision(exchange, symbol, amount=amount)
                if amount_r <= 0:
                    log(
                        state,
                        f"Size rounded to 0. base_btc={base_btc:.8f} final_btc={final_btc:.8f} "
                        f"min_order_btc={cfg.get('min_order_btc')} fallback_used={used_fallback}",
                    )
                    return
                if used_fallback:
                    log(
                        state,
                        f"Fallback sizing used. base_btc={base_btc:.8f} final_btc={final_btc:.8f} "
                        f"min_order_btc={cfg.get('min_order_btc')} fallback_pct={cfg.get('fallback_min_order_pct')}",
                    )
                try:
                    live_place_market(exchange, symbol, rsig["side"], amount_r, reduce_only=False, position_idx=position_idx)
                except Exception as e:
                    log(state, f"Order failed (range entry). err={repr(e)}")
                    return

                state["position"] = {
                    "side": rsig["side"],
                    "amount": float(amount_r),
                    "entry_price": float(rsig["price"]),
                    "stop_price": float(stop_price),
                    "take_profit": float(take_profit),
                    "mode": "range",
                }
                log(state, f"LIVE OPEN {rsig['side']} amt={amount_r:.6f} entry={rsig['price']:.2f} stop={stop_price:.2f} tp={take_profit:.2f} mode=range")

                try:
                    sync_exchange_side_tpsl(exchange, cfg, state)
                    log(state, f"Exchange-side TP/SL set. SL={state['position']['stop_price']} TP={state['position']['take_profit']}")
                except Exception as e:
                    log(state, f"Failed to set exchange-side TP/SL (continuing): {repr(e)}")

                save_state(state)
                return

    save_state(state)


# ---------------------------
# Main
# ---------------------------
def main():
    cfg = load_config()
    state = load_state()
    exchange = make_exchange(cfg)

    check_trade_permissions(exchange, cfg, state)
    apply_min_order_from_market(exchange, cfg, state)

    log(state, f"Bot starting. mode={cfg['mode']} demo_trading={cfg.get('demo_trading', False)} symbol={cfg['symbol']} timeframe={cfg['timeframe']}")
    save_state(state)

    try:
        reconcile_live_position_on_start(exchange, cfg, state)
    except Exception as e:
        log(state, f"Startup reconcile error (continuing): {repr(e)}")
        save_state(state)

    while True:
        try:
            run_once(exchange, cfg, state)
        except ccxt.RateLimitExceeded:
            log(state, "Rate limit hit. Sleeping...")
            save_state(state)
            time.sleep(10)
        except Exception as e:
            log(state, f"Error: {repr(e)}")
            save_state(state)
            time.sleep(10)

        time.sleep(int(cfg.get("poll_seconds", 30)))


if __name__ == "__main__":
    main()
