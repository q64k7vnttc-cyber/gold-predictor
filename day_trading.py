"""
day_trading.py
--------------
Intraday technical analysis and day-trading signals for gold futures (GC=F).

Algorithm sources & research backing:
  • RSI + MACD combined → ~73% win rate on gold futures (235-trade backtest,
    ACM BDEIM 2024, commissions included)
  • Multi-timeframe EMA confirmation raises accuracy to 90%+ (backtested)
  • Bollinger Bands: 30-period / 2.2σ for 1-min; 20-period / 2.0σ for 1H+ (research consensus)
  • EMA stack 9/21/50: triple-confirmation regime detection (day-trader consensus)
  • ATR-based sizing: 1.5× stop, 2.5× target → R:R = 1.67:1 (statistically optimal for gold)
  • VWAP: session-reset daily anchor — price > VWAP = institutional bullish bias
  • Candlestick patterns: 20+ proven patterns (Nison 1991; Bulkowski 2021)
    Morning/Evening Star ~74%/72%, 3 Soldiers/Crows ~78%/75%, Engulfing ~63%/60%
"""

import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

from adaptive_learning import load_weights
from candlestick_patterns import score_candlestick, detect_patterns
from smc_features import compute_smc_score

warnings.filterwarnings("ignore")

TICKER          = "GC=F"        # Gold futures — yfinance spot (XAUUSD=X) is unavailable; GC=F is 99.99% correlated
ATR_STOP_MULT   = 1.5    # stop-loss distance = 1.5 × ATR
ATR_TARGET_MULT = 2.5    # take-profit distance = 2.5 × ATR  (R:R ≈ 1.67)

_CONTEXT_CACHE: dict = {}   # simple in-process TTL cache for macro context


# ─── Data ─────────────────────────────────────────────────────────────────────

def _fetch_macro_context() -> dict:
    """
    Fetch live GVZ (Gold Volatility Index) and 10-year real yield from FRED.

    Cached in memory for 1 hour to avoid repeated network hits during the
    3-second Streamlit autorefresh cycle.

    Returns dict: { "gvz": float|None, "real10y": float|None }
    """
    now = datetime.utcnow()
    if (_CONTEXT_CACHE.get("ts") and
            (now - _CONTEXT_CACHE["ts"]).total_seconds() < 3600):
        return _CONTEXT_CACHE.get("data", {})
    ctx: dict = {}
    # GVZ
    try:
        tk  = yf.Ticker("^GVZ")
        hist = tk.history(period="5d", auto_adjust=True)
        if not hist.empty:
            ctx["gvz"] = float(hist["Close"].dropna().iloc[-1])
            ctx["gvz_5d_ago"] = float(hist["Close"].dropna().iloc[0]) if len(hist) > 1 else ctx["gvz"]
    except Exception:
        pass
    # 10-year TIPS real yield — read from cached model raw data (populated by scheduler)
    try:
        import pickle as _pk
        from pathlib import Path as _P
        _raw_cache = _P(__file__).parent / "data_cache" / "raw_data.pkl"
        if _raw_cache.exists():
            _obj = _pk.loads(_raw_cache.read_bytes())
            _rdf = _obj.get("data")
            if _rdf is not None and "real10y" in _rdf.columns:
                _ry_series = _rdf["real10y"].dropna()
                if not _ry_series.empty:
                    ctx["real10y"] = float(_ry_series.iloc[-1])
    except Exception:
        pass
    # DXY 21-day trend via yfinance
    try:
        dxy = yf.download("DX-Y.NYB", period="30d", progress=False, auto_adjust=True)
        if dxy is not None and not dxy.empty:
            c = dxy["Close"].squeeze().dropna()
            ctx["dxy_trend_pct"] = float((c.iloc[-1] - c.iloc[0]) / c.iloc[0] * 100)
    except Exception:
        pass
    _CONTEXT_CACHE["data"] = ctx
    _CONTEXT_CACHE["ts"]   = now
    return ctx


def fetch_intraday(period: str = "60d", interval: str = "1h") -> pd.DataFrame | None:
    """Download OHLCV via yfinance (gold futures GC=F). Used for HTF context and daily data."""
    try:
        df = yf.download(TICKER, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None


# ── Twelve Data interval mapping ────────────────────────────────────────────
_TD_INTERVAL_MAP = {
    "15m": "15min",
    "30m": "30min",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1day",
}
# How many bars to request for each interval so we always get ≥ 120 usable bars
_TD_BARS_MAP = {"15m": 250, "30m": 200, "1h": 220, "4h": 150, "1d": 500}


def fetch_intraday_td(interval: str = "15m") -> "pd.DataFrame | None":
    """
    Fetch XAU/USD OHLCV bars from Twelve Data (true spot price, zero delay).

    Requires environment variable TWELVE_DATA_KEY.
    Falls back to None if the key is missing or the request fails — caller
    should then fall back to fetch_intraday() (yfinance).

    Advantages over yfinance GC=F:
      • XAU/USD is the exact same reference price as IG CFD Gold
      • No 15-minute data delay — bars are current
      • Native 4H interval (yfinance 4H data is patchy)
    """
    import os, requests as _req
    api_key = os.environ.get("TWELVE_DATA_KEY", "").strip()
    if not api_key:
        return None
    td_interval = _TD_INTERVAL_MAP.get(interval, interval)
    bars        = _TD_BARS_MAP.get(interval, 200)
    try:
        r = _req.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol":     "XAU/USD",
                "interval":   td_interval,
                "outputsize": bars,
                "apikey":     api_key,
                "format":     "JSON",
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "error" or "values" not in data:
            return None
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        df = df.rename(columns={
            "open": "Open", "high": "High",
            "low":  "Low",  "close": "Close", "volume": "Volume",
        })
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # Gold spot has no exchange volume — fill with 0 so downstream code works
        df["Volume"] = pd.to_numeric(df.get("Volume", 0), errors="coerce").fillna(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df if len(df) >= 30 else None
    except Exception:
        return None


# ─── Indicators ───────────────────────────────────────────────────────────────

def _session_vwap(df: pd.DataFrame) -> pd.Series:
    """Session-reset VWAP (resets each calendar day)."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    tp_vol  = typical * df["Volume"]
    vwap    = (
        tp_vol.groupby(df.index.date).transform("cumsum")
        / df["Volume"].groupby(df.index.date).transform("cumsum")
    )
    return vwap


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicator columns to df and drop NaN rows."""
    df = df.copy()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    # RSI(14)
    df["rsi"] = RSIIndicator(close, window=14).rsi()

    # MACD(12,26,9)
    _macd       = MACD(close, window_fast=12, window_slow=26, window_sign=9)
    df["macd"]      = _macd.macd()
    df["macd_sig"]  = _macd.macd_signal()
    df["macd_hist"] = _macd.macd_diff()

    # Bollinger Bands(20, 2)
    _bb            = BollingerBands(close, window=20, window_dev=2.0)
    df["bb_upper"] = _bb.bollinger_hband()
    df["bb_mid"]   = _bb.bollinger_mavg()
    df["bb_lower"] = _bb.bollinger_lband()
    df["bb_pct"]   = _bb.bollinger_pband()   # 0 = at lower band, 1 = at upper band

    # EMA stack: 9 / 21 / 50
    df["ema9"]  = EMAIndicator(close, window=9).ema_indicator()
    df["ema21"] = EMAIndicator(close, window=21).ema_indicator()
    df["ema50"] = EMAIndicator(close, window=50).ema_indicator()

    # ATR(14)
    df["atr"] = AverageTrueRange(high, low, close, window=14).average_true_range()

    # ADX(14) + DI+/DI- — trend strength and direction
    _adx = ADXIndicator(high, low, close, window=14)
    df["adx"]      = _adx.adx()
    df["di_plus"]  = _adx.adx_pos()
    df["di_minus"] = _adx.adx_neg()

    # Stochastic(14, 3) — second momentum oscillator, independent of RSI
    _stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["stoch_k"] = _stoch.stoch()
    df["stoch_d"] = _stoch.stoch_signal()

    # OBV — volume-weighted price direction
    df["obv"] = OnBalanceVolumeIndicator(close, df["Volume"]).on_balance_volume()

    # Extended EMA stack for daily timeframe analysis
    df["ema20"]  = EMAIndicator(close, window=20).ema_indicator()
    df["ema100"] = EMAIndicator(close, window=100).ema_indicator()
    df["ema200"] = EMAIndicator(close, window=200).ema_indicator()

    # SMA(20) — used as daily fair-value anchor (replaces VWAP on daily bars)
    df["sma20"] = close.rolling(window=20).mean()

    # VWAP (session-reset, meaningful for intraday intervals only)
    try:
        df["vwap"] = _session_vwap(df)
    except Exception:
        df["vwap"] = df["bb_mid"]

    return df.dropna()


# ─── Signal Scoring ───────────────────────────────────────────────────────────

def _macd_crossed(df: pd.DataFrame, direction: str = "up", within: int = 5) -> bool:
    """True if MACD crossed the signal line within the last `within` bars."""
    diff = df["macd"] - df["macd_sig"]
    for i in range(min(within, len(diff) - 1)):
        prev = diff.iloc[-(i + 2)]
        curr = diff.iloc[-(i + 1)]
        if direction == "up"   and prev < 0 and curr >= 0:
            return True
        if direction == "down" and prev > 0 and curr <= 0:
            return True
    return False


def score_signals(df: pd.DataFrame, interval: str = "1h", htf_context: dict | None = None):
    """
    Score each indicator on a -2 … +2 scale, return (scores_dict, total, max_possible).

    `interval` switches between intraday ("1h") and daily ("1d") calibrations:
    - Thresholds, EMA stack, and anchor indicator all differ between timeframes.
    - Daily uses 20/50/200 EMA stack, SMA(20) anchor, and wider RSI/BB/Stoch bands.
    - Intraday uses 9/21/50 EMA stack, session VWAP anchor, and tighter bands.

    `htf_context` (optional): dict with higher-timeframe data for intraday signals.
    - "ema200_dist_pct": daily price distance from 200-day EMA (%).
      When present, contrary oscillators on 1H are neutralised in strong daily trends.
    """
    if df is None or len(df) < 60:
        return None

    _daily = (interval == "1d")

    row   = df.iloc[-1]
    close = float(row["Close"])
    rsi   = float(row["rsi"])
    bb    = float(row["bb_pct"])
    scores: dict[str, int] = {}

    # ── RSI ──────────────────────────────────────────────────────
    # Daily bars: RSI trends more slowly and sits near 50 for weeks.
    # Use wider bands so the oscillator actually contributes a signal.
    # Intraday: gold spikes quickly so tighter bands catch real extremes.
    if _daily:
        if   rsi < 35:  scores["RSI"] =  2   # deeply oversold on daily
        elif rsi < 47:  scores["RSI"] =  1   # approaching oversold
        elif rsi > 65:  scores["RSI"] = -2   # deeply overbought on daily
        elif rsi > 53:  scores["RSI"] = -1   # approaching overbought
        else:           scores["RSI"] =  0
    else:
        if   rsi < 30:  scores["RSI"] =  2
        elif rsi < 45:  scores["RSI"] =  1
        elif rsi > 70:  scores["RSI"] = -2
        elif rsi > 55:  scores["RSI"] = -1
        else:           scores["RSI"] =  0

    # ── MACD ─────────────────────────────────────────────────────
    if   _macd_crossed(df, "up"):
        scores["MACD"] =  2   # fresh bullish crossover
    elif float(row["macd"]) > float(row["macd_sig"]):
        scores["MACD"] =  1   # MACD above signal (bullish bias)
    elif _macd_crossed(df, "down"):
        scores["MACD"] = -2   # fresh bearish crossover
    else:
        scores["MACD"] = -1   # MACD below signal (bearish bias)

    # ── Bollinger Bands ───────────────────────────────────────────
    # Daily bands are wider so price naturally stays near the middle.
    # Widen daily triggers so the band contributes a real signal.
    if _daily:
        if   bb < 0.15:  scores["Bollinger"] =  2   # daily: near lower band
        elif bb < 0.35:  scores["Bollinger"] =  1
        elif bb > 0.85:  scores["Bollinger"] = -2   # daily: near upper band
        elif bb > 0.65:  scores["Bollinger"] = -1
        else:            scores["Bollinger"] =  0
    else:
        if   bb < 0.10:  scores["Bollinger"] =  2
        elif bb < 0.30:  scores["Bollinger"] =  1
        elif bb > 0.90:  scores["Bollinger"] = -2
        elif bb > 0.70:  scores["Bollinger"] = -1
        else:            scores["Bollinger"] =  0

    # ── EMA Trend ─────────────────────────────────────────────────
    # Daily timeframe uses 20/50/200 EMA stack — the standard daily trend
    # stack watched by institutions and algos worldwide.
    # Intraday uses 9/21/50 — short-term momentum stack.
    if _daily and "ema200" in df.columns:
        e20  = float(row.get("ema20",  row["ema9"]))
        e50  = float(row["ema50"])
        e200 = float(row.get("ema200", row["ema50"]))
        if   e20 > e50 > e200:  scores["EMA Trend"] =  2   # daily bull: price > 20 > 50 > 200
        elif e20 > e50:         scores["EMA Trend"] =  1   # short-term bullish
        elif e20 < e50 < e200:  scores["EMA Trend"] = -2   # daily bear: 20 < 50 < 200
        else:                   scores["EMA Trend"] = -1   # mixed
        # EMA200 distance — 3 tiers:
        # >10% above = strong confirmed bull trend (gold can run far from EMA200)
        # >0.5%-10% above = healthy position above long-term trend line
        # mirrored for below
        _e200_dist = (close - e200) / e200 * 100
        if   _e200_dist >  10.0:  scores["EMA200"] =  2   # strong confirmed bull trend
        elif _e200_dist >   0.5:  scores["EMA200"] =  1   # above 200-day EMA
        elif _e200_dist <  -0.5:  scores["EMA200"] = -1   # below 200-day EMA
        elif _e200_dist < -10.0:  scores["EMA200"] = -2   # strong confirmed bear trend

        # EMA200 slope — is the long-term trend itself rising?
        # A rising 200-day EMA means the secular trend is intact and accelerating.
        # Compare EMA200 now vs 20 bars ago (≈ 1 month).
        if len(df) >= 22 and "ema200" in df.columns:
            _e200_prev = float(df["ema200"].iloc[-21])
            _e200_slope_pct = (e200 - _e200_prev) / (_e200_prev + 1e-9) * 100
            if   _e200_slope_pct >  0.5:  scores["EMA200 Slope"] =  2   # strongly rising 200d
            elif _e200_slope_pct >  0.1:  scores["EMA200 Slope"] =  1   # gently rising 200d
            elif _e200_slope_pct < -0.5:  scores["EMA200 Slope"] = -2   # strongly falling 200d
            elif _e200_slope_pct < -0.1:  scores["EMA200 Slope"] = -1   # gently falling 200d

        # Price vs EMA50 — is price on the right side of the medium-term trend?
        # Capturing a fresh close above/below EMA50 independent of EMA ordering.
        _e50_dist = (close - e50) / e50 * 100
        if   _e50_dist >  0.3:  scores["vs EMA50"] =  1   # price above 50-day EMA
        elif _e50_dist < -0.3:  scores["vs EMA50"] = -1   # price below 50-day EMA
    else:
        e9, e21, e50 = float(row["ema9"]), float(row["ema21"]), float(row["ema50"])
        if   e9 > e21 > e50:  scores["EMA Trend"] =  2
        elif e9 > e21:        scores["EMA Trend"] =  1
        elif e9 < e21 < e50:  scores["EMA Trend"] = -2
        else:                 scores["EMA Trend"] = -1

    # ── Price Anchor ─────────────────────────────────────────────
    # Intraday: VWAP (institutional session fair-value anchor).
    # Daily: SMA(20) — the most-watched daily moving average for swing traders.
    #   Session VWAP on daily bars always equals that day's typical price ≈ close,
    #   so the distance is always ~0 — it never fires. Replace with SMA(20).
    if _daily and "sma20" in df.columns:
        sma20 = float(row["sma20"])
        sma_dist = (close - sma20) / sma20 * 100   # % from SMA(20)
        if   sma_dist >  1.5:  scores["SMA20"] = -1   # extended above SMA → mean-reversion risk
        elif sma_dist >  0.4:  scores["SMA20"] =  1   # healthy bull breakout above SMA
        elif sma_dist < -1.5:  scores["SMA20"] =  1   # extended below SMA → mean-reversion bounce
        elif sma_dist < -0.4:  scores["SMA20"] = -1   # price falling below SMA → bearish
        else:                  scores["SMA20"] =  0
    else:
        vwap    = float(row["vwap"])
        dist    = (close - vwap) / vwap * 100
        if   dist >  0.20:  scores["VWAP"] =  1
        elif dist < -0.20:  scores["VWAP"] = -1
        else:               scores["VWAP"] =  0

    # ── MACD Histogram momentum ───────────────────────────────────
    h_now  = float(row["macd_hist"])
    h_prev = float(df["macd_hist"].iloc[-2]) if len(df) >= 2 else 0.0
    h_prev2 = float(df["macd_hist"].iloc[-3]) if len(df) >= 3 else h_prev
    _expanding_bull = h_now > 0 and h_now > h_prev
    _expanding_bear = h_now < 0 and h_now < h_prev
    _recent_zero_cross = (h_prev2 <= 0 < h_now) or (h_prev2 >= 0 > h_now)
    if   _expanding_bull and _recent_zero_cross:  scores["Momentum"] =  2
    elif _expanding_bull:                          scores["Momentum"] =  1
    elif _expanding_bear and _recent_zero_cross:   scores["Momentum"] = -2
    elif _expanding_bear:                          scores["Momentum"] = -1
    else:                                          scores["Momentum"] =  0

    # ── Rate of Change ────────────────────────────────────────────
    # Daily ROC uses 10-bar (two-week) window and wider thresholds since
    # a meaningful daily price move is ±1%, not ±0.15% like intraday.
    _roc_bars = 10 if _daily else 5
    _roc_hi   = (1.20, 0.50) if _daily else (0.40, 0.15)   # (+2, +1) thresholds
    if len(df) >= _roc_bars + 1:
        _roc = (close - float(df["Close"].iloc[-(_roc_bars+1)])) / float(df["Close"].iloc[-(_roc_bars+1)]) * 100
        if   _roc >  _roc_hi[0]:  scores["ROC"] =  2
        elif _roc >  _roc_hi[1]:  scores["ROC"] =  1
        elif _roc < -_roc_hi[0]:  scores["ROC"] = -2
        elif _roc < -_roc_hi[1]:  scores["ROC"] = -1
        else:                     scores["ROC"] =  0

    # ── Candlestick patterns ──────────────────────────────────────────────
    cs_score, cs_patterns = score_candlestick(df)
    if cs_score != 0.0:
        scores["Candlestick"] = cs_score

    # ── Smart Money Concepts (BOS · FVG · Order Blocks · Liquidity Sweeps) ──
    # SMC detects institutional price-action: break of structure signals when
    # a trend has genuinely flipped; fair value gaps mark imbalances that get
    # re-tested; order blocks are institutional accumulation/distribution zones;
    # liquidity sweeps expose stop-hunts that often precede sharp reversals.
    # The ATR-quality pattern score from candlestick_pro is also folded in here.
    _smc_sub: dict = {}
    try:
        _smc_score, _smc_sub = compute_smc_score(df)
        if _smc_score != 0:
            scores["SMC"] = max(-2, min(2, _smc_score))
    except Exception:
        pass

    # ── GVZ (Gold Volatility Index) ───────────────────────────────────────
    # GVZ is the CBOE implied-vol index for gold (≈ "Gold VIX").
    # Strategy: elevated + falling GVZ signals fear unwind → contrarian buy;
    # spiking GVZ during a downmove signals capitulation / volatility crush ahead.
    # Research: post-GVZ-spike reversals occur within 5 days 64% of the time.
    ctx = _fetch_macro_context()
    gvz      = ctx.get("gvz")
    gvz_prev = ctx.get("gvz_5d_ago", gvz)
    if gvz is not None and gvz_prev is not None:
        gvz_chg_pct = (gvz - gvz_prev) / (gvz_prev + 1e-9)
        if gvz > 35 and gvz_chg_pct < -0.05:
            scores["GVZ"] = 2       # extreme fear subsiding → strong reversal buy signal
        elif gvz > 25 and gvz_chg_pct < -0.05:
            scores["GVZ"] = 1       # elevated vol falling → mild bullish
        elif gvz > 40:
            scores["GVZ"] = 1       # extreme panic → mean-reversion opportunity
        elif gvz < 14:
            scores["GVZ"] = 0       # very low vol, trending — directionally neutral
        elif gvz_chg_pct > 0.10:
            scores["GVZ"] = -1      # vol spiking → hedging demand rising, caution

    # ── Macro Regime ──────────────────────────────────────────────────────
    # Composite macro regime signal: real yields + DXY + risk appetite.
    # Gold is highly regime-dependent: identical technicals have very different
    # win-rates depending on whether macro is supportive or hostile.
    #   +2: ideal regime (negative real yields, weak DXY, risk-off)
    #   +1: one supportive factor
    #   -1: headwinds present
    #   -2: hostile regime (rising real yields, strong DXY)
    regime_score = 0
    real10y = ctx.get("real10y")
    if real10y is not None:
        if real10y < 0:
            regime_score += 1       # negative real yields → gold bullish
        elif real10y > 1.5:
            regime_score -= 1       # high positive real yield → gold headwind
    dxy_trend = ctx.get("dxy_trend_pct")
    if dxy_trend is not None:
        if dxy_trend < -1.0:
            regime_score += 1       # weakening dollar over 21 days → gold bullish
        elif dxy_trend > 1.5:
            regime_score -= 1       # strengthening dollar → gold headwind
    if gvz is not None:
        if gvz > 28:
            regime_score += 1       # risk-off / fear → flight to gold
    if regime_score != 0:
        scores["Regime"] = max(-2, min(2, regime_score))

    # ── ML Ensemble Forecast ───────────────────────────────────────────────────
    # Bridge the ML model (XGBoost/RF/MLP trained on 700+ macro features) with
    # the technical scoring system.  When both agree → high conviction.
    # When they disagree → the technical signal is tempered.
    # The ML forecast is read from the cached multi-horizon predictions file
    # produced by the scheduler every 15 min.  Low-confidence ML calls (<30%)
    # are ignored — only meaningful directional consensus fires.
    try:
        import json as _json; from pathlib import Path as _Path
        _mh_file = _Path(__file__).parent / "data_cache" / "multi_horizon_predictions.json"
        if _mh_file.exists():
            _mh = _json.loads(_mh_file.read_text())
            # raw_proba = P(UP).  < 0.5 means model leans DOWN.
            # Use raw_proba rather than the discretised 'direction' field so
            # that cases where the model is bearish-leaning but not at the
            # hard threshold still register as a headwind.
            _ml_probas = [_mh.get(str(h), {}).get("raw_proba", 0.5) for h in [1, 2, 5]]
            _avg_prob  = sum(_ml_probas) / max(1, len(_ml_probas))
            _n_down    = sum(1 for p in _ml_probas if p < 0.45)   # clearly bearish
            _n_up      = sum(1 for p in _ml_probas if p > 0.55)   # clearly bullish
            _n_str_dn  = sum(1 for p in _ml_probas if p < 0.38)   # strongly bearish
            _n_str_up  = sum(1 for p in _ml_probas if p > 0.62)   # strongly bullish
            if   _n_str_dn == 3 or (_n_down == 3 and _avg_prob < 0.38):
                scores["ML Forecast"] = -2   # all 3 strongly DOWN
            elif _n_down >= 2 and _avg_prob < 0.45:
                scores["ML Forecast"] = -1   # majority DOWN
            elif _n_str_up == 3 or (_n_up == 3 and _avg_prob > 0.62):
                scores["ML Forecast"] =  2   # all 3 strongly UP
            elif _n_up >= 2 and _avg_prob > 0.55:
                scores["ML Forecast"] =  1   # majority UP
    except Exception:
        pass

    # ── ADX — Trend Strength + Direction (DI+/DI-) ───────────────────────
    # ADX measures how strong the prevailing trend is; DI+ vs DI- gives
    # direction. A high ADX reading means the current EMA/RSI/MACD signals
    # are operating inside a confirmed trend — highest-quality setup.
    #   ADX > 30 = strong trend  |  ADX > 20 = emerging trend  |  < 15 = chop
    if "adx" in df.columns:
        _adx_val  = float(row["adx"])  if not np.isnan(float(row["adx"]))  else 0.0
        _di_p     = float(row["di_plus"])  if not np.isnan(float(row["di_plus"]))  else 0.0
        _di_m     = float(row["di_minus"]) if not np.isnan(float(row["di_minus"])) else 0.0
        if _adx_val > 30:
            if   _di_p > _di_m:  scores["ADX"] =  2   # strong confirmed uptrend
            elif _di_m > _di_p:  scores["ADX"] = -2   # strong confirmed downtrend
        elif _adx_val > 20:
            if   _di_p > _di_m:  scores["ADX"] =  1   # emerging uptrend
            elif _di_m > _di_p:  scores["ADX"] = -1   # emerging downtrend
        # ADX < 15: choppy, no score — noisy signals in trendless markets

    # ── Stochastic (14, 3) ────────────────────────────────────────────────
    # Daily: wider bands — stochastic on daily gold is less reactive.
    # Intraday: tighter bands for faster oscillation detection.
    if "stoch_k" in df.columns:
        _sk = float(row["stoch_k"]) if not np.isnan(float(row["stoch_k"])) else 50.0
        _sd = float(row["stoch_d"]) if not np.isnan(float(row["stoch_d"])) else 50.0
        if _daily:
            # Daily stochastic thresholds: more conservative on "overbought" —
            # gold can stay at K>80 for months in a strong bull market.
            # Only flag as strongly overbought at K>87 to avoid false sells.
            if   _sk < 25:  scores["Stochastic"] =  2   # strongly oversold (daily)
            elif _sk < 40:  scores["Stochastic"] =  1   # approaching oversold
            elif _sk > 87:  scores["Stochastic"] = -2   # extreme overbought (daily)
            elif _sk > 70:  scores["Stochastic"] = -1   # approaching overbought
            else:           scores["Stochastic"] =  0
        else:
            if   _sk < 20:  scores["Stochastic"] =  2
            elif _sk < 35:  scores["Stochastic"] =  1
            elif _sk > 80:  scores["Stochastic"] = -2
            elif _sk > 65:  scores["Stochastic"] = -1
            else:           scores["Stochastic"] =  0

    # ── OBV Slope — Volume Direction Confirmation ─────────────────────────
    # On-Balance Volume trending up = institutional buying behind the move;
    # trending down = distribution. Measured as the sign of the 10-bar slope.
    if "obv" in df.columns and len(df) >= 12:
        _obv_series = df["obv"].iloc[-11:].values.astype(float)
        _obv_slope  = np.polyfit(range(len(_obv_series)), _obv_series, 1)[0]
        _obv_norm   = _obv_slope / (abs(_obv_series).mean() + 1e-9) * 100
        if   _obv_norm >  0.5:  scores["OBV"] =  1   # volume confirming uptrend
        elif _obv_norm < -0.5:  scores["OBV"] = -1   # volume confirming downtrend
        else:                   scores["OBV"] =  0

    # ── 52-Week Range Position (daily only) ──────────────────────────────
    # Where price sits in its 52-week range is a primary momentum signal
    # for gold daily trading. Near a 52-week high = breakout momentum;
    # near a 52-week low = potential recovery setup.
    if _daily and len(df) >= 100:
        _lookback = min(252, len(df))
        _hi252 = float(df["High"].iloc[-_lookback:].max())
        _lo252 = float(df["Low"].iloc[-_lookback:].min())
        _range  = _hi252 - _lo252
        if _range > 0:
            _pos = (close - _lo252) / _range   # 0 = 52w low, 1 = 52w high
            if   _pos > 0.90:  scores["52W Range"] =  2   # near 52-week high — breakout momentum
            elif _pos > 0.70:  scores["52W Range"] =  1   # upper half of range
            elif _pos < 0.10:  scores["52W Range"] = -2   # near 52-week low
            elif _pos < 0.30:  scores["52W Range"] = -1   # lower half of range

    # ── Trend-Context Oscillator Adjustment (daily only) ──────────────────
    # In a confirmed strong trend oscillators like Stochastic and RSI CAN
    # remain "overbought/oversold" for months — this is momentum persistence,
    # NOT a reversal signal.
    #
    # Two tiers of adjustment:
    #   EMA200 = ±1 (>0.5% away): soften contrary oscillator by 1 step
    #   EMA200 = ±2 (>10% away):  fully neutralise contrary oscillator → 0
    #
    # This ensures a 13%+ bull trend doesn't get dragged down by a Stochastic
    # reading of K=93 that is simply "staying elevated" in a trending market.
    if _daily and "EMA200" in scores:
        _trend_str = scores.get("EMA200", 0)
        if _trend_str >= 2:    # strongly above EMA200 → fully neutralise bear oscillators
            if scores.get("Stochastic", 0) < 0:
                scores["Stochastic"] = 0
            if scores.get("RSI", 0) < 0:
                scores["RSI"] = 0
        elif _trend_str == 1:  # moderately above → soften by 1
            if scores.get("Stochastic", 0) < 0:
                scores["Stochastic"] = min(0, scores["Stochastic"] + 1)
            if scores.get("RSI", 0) < 0:
                scores["RSI"] = min(0, scores["RSI"] + 1)
        elif _trend_str <= -2: # strongly below EMA200 → fully neutralise bull oscillators
            if scores.get("Stochastic", 0) > 0:
                scores["Stochastic"] = 0
            if scores.get("RSI", 0) > 0:
                scores["RSI"] = 0
        elif _trend_str == -1: # moderately below → soften by 1
            if scores.get("Stochastic", 0) > 0:
                scores["Stochastic"] = max(0, scores["Stochastic"] - 1)
            if scores.get("RSI", 0) > 0:
                scores["RSI"] = max(0, scores["RSI"] - 1)

    # ── RSI / Price Divergence ─────────────────────────────────────────────────
    # Divergence is the most reliable early-warning reversal signal.
    # Bearish: price makes higher high but RSI makes lower high → distribution
    # Bullish: price makes lower low but RSI makes higher low  → accumulation
    # Look back 8 bars to find the prior swing high/low.
    if len(df) >= 9:
        _div_bars = 8
        _c_now  = float(df["Close"].iloc[-1])
        _r_now  = float(df["rsi"].iloc[-1])
        _c_prev = float(df["Close"].iloc[-1 - _div_bars])
        _r_prev = float(df["rsi"].iloc[-1 - _div_bars])
        _c_chg  = (_c_now - _c_prev) / (_c_prev + 1e-9)
        _r_chg  = _r_now - _r_prev
        _threshold = 0.003  # 0.3% price move required to be meaningful
        if _c_chg > _threshold and _r_chg < -3:
            scores["Divergence"] = -2   # bearish divergence: price up, RSI down
        elif _c_chg > _threshold and _r_chg < -1:
            scores["Divergence"] = -1
        elif _c_chg < -_threshold and _r_chg > 3:
            scores["Divergence"] =  2   # bullish divergence: price down, RSI up
        elif _c_chg < -_threshold and _r_chg > 1:
            scores["Divergence"] =  1

    # ── 4H Intermediate Trend ─────────────────────────────────────────────────
    # The 4-hour chart sits between the daily macro view and the 1H entry signal.
    # When all three timeframes agree (daily BUY → 4H uptrend → 1H BUY) the
    # trade has multi-timeframe confluence — the highest quality setup.
    # When they disagree (daily BUY but 4H downtrend) the 1H signal is weaker.
    if not _daily and htf_context is not None and "4h_ema_trend" in htf_context:
        _4h_tr   = int(htf_context.get("4h_ema_trend", 0))
        _4h_macd = int(htf_context.get("4h_macd_bullish", 0))
        _4h_rsi  = float(htf_context.get("4h_rsi", 50))
        # Score: EMA alignment + MACD agreement, penalise extreme RSI
        _4h_base = _4h_tr  # +1 bullish / -1 bearish / 0 mixed
        if _4h_tr != 0 and _4h_macd == _4h_tr:
            _4h_base = 2 * _4h_tr   # strong: EMA + MACD aligned
        # If 4H RSI is extreme in the OPPOSITE direction, reduce score
        if _4h_base > 0 and _4h_rsi > 76:    _4h_base = max(0, _4h_base - 1)
        if _4h_base < 0 and _4h_rsi < 24:    _4h_base = min(0, _4h_base + 1)
        if _4h_base != 0:
            scores["4H Trend"] = max(-2, min(2, _4h_base))

    # ── Higher Timeframe (HTF) Context — intraday only ────────────────────────
    # The 1H chart does not know if the daily trend is bullish or bearish.
    # In a strong daily bull trend (price >10% above 200-day EMA) an RSI of 72
    # or Stochastic K of 80 on the 1H is MOMENTUM PERSISTENCE, not a reversal.
    # IMPORTANT: only neutralise MILD contrary signals (score = ±1).
    # Extreme readings (score = ±2) are genuine overbought/oversold signals
    # that should fire even within a strong trend — they indicate exhaustion.
    if not _daily and htf_context is not None:
        _htf_e200 = float(htf_context.get("ema200_dist_pct", 0.0))
        # Score the daily trend as an explicit 1H indicator
        if   _htf_e200 >  10.0:  scores["HTF Trend"] =  2
        elif _htf_e200 >   0.5:  scores["HTF Trend"] =  1
        elif _htf_e200 < -10.0:  scores["HTF Trend"] = -2
        elif _htf_e200 <  -0.5:  scores["HTF Trend"] = -1
        # Tier 2 (>10% above EMA200): neutralise MILD bearish oscillators only.
        # Extreme bearish (-2) stays — that is genuine exhaustion, not noise.
        if _htf_e200 > 10.0:
            for _k in ("RSI", "Bollinger", "Stochastic"):
                if scores.get(_k, 0) == -1:   # mild bearish → neutralise
                    scores[_k] = 0
                # -2 (extreme overbought) intentionally preserved
        elif _htf_e200 > 0.5:
            for _k in ("RSI", "Bollinger", "Stochastic"):
                if scores.get(_k, 0) < 0:
                    scores[_k] = min(0, scores[_k] + 1)
        # Mirrored for bear trend: neutralise mild bullish only
        elif _htf_e200 < -10.0:
            for _k in ("RSI", "Bollinger", "Stochastic"):
                if scores.get(_k, 0) == 1:    # mild bullish → neutralise
                    scores[_k] = 0
                # +2 (extreme oversold) intentionally preserved
        elif _htf_e200 < -0.5:
            for _k in ("RSI", "Bollinger", "Stochastic"):
                if scores.get(_k, 0) > 0:
                    scores[_k] = max(0, scores[_k] - 1)

    # ── Directional Confluence Bonus (group-based) ─────────────────────────────
    # Groups prevent correlated oscillators from artificially dominating.
    # RSI + Stochastic + Bollinger all reading "overbought" is ONE data point
    # (price is extended), not three independent bearish signals.
    # Six groups: trend, momentum, oscillators (capped at 1), volume, macro, anchor.
    # Confluence is awarded based on how many GROUPS agree, not raw indicator count.
    _GROUPS = {
        "trend":      ["EMA Trend", "ADX", "EMA200", "EMA200 Slope", "vs EMA50", "52W Range"],
        "momentum":   ["MACD", "Momentum", "ROC"],
        "oscillator": ["RSI", "Bollinger", "Stochastic", "Divergence"],
        "volume":     ["OBV"],
        "macro":      ["Regime", "GVZ", "ML Forecast"],
        "anchor":     ["VWAP", "SMA20", "Candlestick", "HTF Trend", "4H Trend"],
        "structure":  ["SMC"],   # Smart Money Concepts — institutional price action
    }
    _bull_grps = 0
    _bear_grps = 0
    for _members in _GROUPS.values():
        _grp_sum = sum(scores.get(m, 0) for m in _members)
        if   _grp_sum > 0: _bull_grps += 1
        elif _grp_sum < 0: _bear_grps += 1

    if _bull_grps > _bear_grps:
        _gdiff = _bull_grps - _bear_grps
        scores["Confluence"] =  min(3, _gdiff)
    elif _bear_grps > _bull_grps:
        _gdiff = _bear_grps - _bull_grps
        scores["Confluence"] = -min(3, _gdiff)

    # ── Apply adaptive weights ─────────────────────────────────────────
    weights       = load_weights()
    weighted_total = sum(v * weights.get(k, 1.0) for k, v in scores.items())
    total     = weighted_total
    max_score = sum(abs(v) * weights.get(k, 1.0) for k, v in scores.items())
    return scores, total, max_score, cs_patterns, _bull_grps, _bear_grps, _smc_sub


# ─── Trading Recommendation ────────────────────────────────────────────────────

def build_signal(df: pd.DataFrame, interval: str = "1h",
                 htf_context: dict | None = None) -> dict | None:
    """
    Combine all indicator scores into an actionable trading recommendation.

    Returns a dict with:
      action, direction, entry, target, stop_loss, risk_reward,
      potential_gain, potential_loss, confidence, atr, rsi, indicators, …
    """
    if df is None or len(df) < 60:
        return None
    result = score_signals(df, interval=interval, htf_context=htf_context)
    if result is None:
        return None

    scores, total, max_score, cs_patterns, bull_grps, bear_grps, smc_sub = result
    row   = df.iloc[-1]
    close = float(row["Close"])
    atr   = float(row["atr"])

    confidence = abs(total) / max_score if max_score > 0 else 0.0
    confluence = scores.get("Confluence", 0)

    # ── Volume quality factor ─────────────────────────────────────────────────
    # Low-volume moves are unreliable (stop-hunts, thin market).
    # Amplify confidence when volume confirms; dampen it when volume is weak.
    vol_now = float(df["Volume"].iloc[-1])
    vol_avg = float(df["Volume"].iloc[-20:].mean()) if len(df) >= 20 else vol_now
    vol_ratio = vol_now / (vol_avg + 1e-9)
    vol_factor = min(1.25, max(0.65, vol_ratio))   # clamp 0.65 – 1.25
    eff_conf   = confidence * vol_factor

    # ── Session-based confidence modifier ────────────────────────────────────
    # Gold intraday signals have very different reliability depending on which
    # trading session is active. London/NY overlap is the highest-liquidity
    # window — the same technical setup has a materially better win rate there
    # than during the thin Asian session (02:00–07:00 UTC).
    # Applies to both 1H and 15-min timeframes (15-min candles have the same
    # session liquidity profile as 1H — just more granular).
    if interval in ("1h", "15m", "30m"):
        import datetime as _dt
        _utc_h = _dt.datetime.utcnow().hour
        if 13 <= _utc_h < 17:    session_mult = 1.12   # London/NY overlap — peak
        elif 8  <= _utc_h < 13:  session_mult = 1.06   # London open
        elif 17 <= _utc_h < 21:  session_mult = 1.00   # NY afternoon
        elif 21 <= _utc_h < 23:  session_mult = 0.92   # NY/early Asian
        else:                     session_mult = 0.82   # deep Asian (02–07 UTC)
        eff_conf *= session_mult

    # ── ATR volatility gate ───────────────────────────────────────────────────
    # During extreme volatility price action becomes random (momentum traders,
    # stop cascades). Raise the confidence bar in extreme vol conditions.
    # 15-min, 30-min, and 1H are all intraday — they share the 1.5% ATR gate.
    atr_pct      = atr / close * 100
    _intraday    = (interval != "1d")
    extreme_vol  = (_intraday and atr_pct > 1.5) or (not _intraday and atr_pct > 3.5)

    # ── Map to action using confidence + confluence gate ──────────────────────
    # Confidence = |weighted_total| / max_weighted_score  (scale-invariant 0-1).
    # Confluence = how many indicator groups agree (max ±3).
    # Two-gate system: both must pass for a signal to fire.
    # During extreme volatility the confidence bar is raised by 0.10.
    _vol_adj     = 0.10 if extreme_vol else 0.0
    _sb_conf     = 0.58 + _vol_adj   # STRONG BUY/SELL confidence threshold
    _b_conf      = 0.38 + _vol_adj   # BUY/SELL confidence threshold

    if   eff_conf >= _sb_conf and total > 0 and confluence >= 2:
        action, color_key = "STRONG BUY",  "strong_buy"
    elif eff_conf >= _b_conf  and total > 0 and confluence >= 1:
        action, color_key = "BUY",          "buy"
    elif eff_conf >= _sb_conf and total < 0 and confluence <= -2:
        action, color_key = "STRONG SELL",  "strong_sell"
    elif eff_conf >= _b_conf  and total < 0 and confluence <= -1:
        action, color_key = "SELL",         "sell"
    else:
        action, color_key = "NEUTRAL",      "neutral"

    is_long    = total >= 0
    entry      = close
    stop_loss  = entry - ATR_STOP_MULT   * atr if is_long else entry + ATR_STOP_MULT   * atr
    target     = entry + ATR_TARGET_MULT * atr if is_long else entry - ATR_TARGET_MULT * atr
    risk       = abs(entry - stop_loss)
    reward     = abs(target - entry)
    rr_ratio   = reward / risk if risk > 0 else 0.0

    # ── Fix 1: Short-term pullback detection (last 5 bars) ───────────────────
    # Even when the macro signal is BUY, a sharp intraday drop means price is
    # actively falling — entering now risks buying into continued momentum down.
    _pullback_warning  = False
    _pullback_severity = "none"
    if len(df) >= 5:
        _pc = df["Close"].iloc[-5:].values.astype(float)
        _5bar_ret = (_pc[-1] - _pc[0]) / (_pc[0] + 1e-9) * 100
        if _5bar_ret < -0.8:
            _pullback_warning  = True
            _pullback_severity = "strong"
        elif _5bar_ret < -0.4:
            _pullback_warning  = True
            _pullback_severity = "mild"

    # ── Fix 3: Stop-zone detection (dropped >1.3 ATR from recent 10-bar high) ─
    # If price has already fallen more than a normal stop distance from its
    # recent peak, a trade opened now is statistically in a losing position
    # from the start — the stop may already have been hit.
    _stop_zone_warning    = False
    _drop_from_high_atrs  = 0.0
    if len(df) >= 10 and atr > 0:
        _recent_high         = float(df["High"].iloc[-10:].max())
        _drop_from_high_atrs = (_recent_high - close) / atr
        _stop_zone_warning   = _drop_from_high_atrs > 1.3

    return {
        "action":         action,
        "color_key":      color_key,
        "total_score":    total,
        "max_score":      max_score,
        "confidence":     round(confidence, 3),
        "direction":      "LONG" if is_long else "SHORT",
        "entry":          round(entry, 2),
        "target":         round(target, 2),
        "stop_loss":      round(stop_loss, 2),
        "risk_reward":    round(rr_ratio, 2),
        "potential_gain": round(reward, 2),
        "potential_loss": round(risk, 2),
        "atr":            round(atr, 2),
        "rsi":            round(float(row["rsi"]), 2),
        "bb_pct":         round(float(row["bb_pct"]) * 100, 1),
        "vwap":           round(float(row["vwap"]), 2),
        "ema9":           round(float(row["ema9"]), 2),
        "ema21":          round(float(row["ema21"]), 2),
        "ema50":          round(float(row["ema50"]), 2),
        "macd":                  round(float(row["macd"]), 4),
        "macd_sig":              round(float(row["macd_sig"]), 4),
        "indicators":            scores,
        "candlestick_patterns":  cs_patterns,
        "smc_breakdown":         smc_sub,
        "timestamp":             df.index[-1].isoformat(),
        "pullback_warning":      _pullback_warning,
        "pullback_severity":     _pullback_severity,
        "stop_zone_warning":     _stop_zone_warning,
        "drop_from_high_atrs":   round(_drop_from_high_atrs, 2),
    }


# ─── Intraday Chart ────────────────────────────────────────────────────────────

_COLORS = {
    "strong_buy":  "#4caf50",
    "buy":         "#66bb6a",
    "strong_sell": "#ef5350",
    "sell":        "#ff7043",
    "neutral":     "#90a4ae",
}


def build_intraday_chart(df: pd.DataFrame, signal: dict | None,
                         show_bars: int = 120) -> plt.Figure:
    """
    Three-panel chart:
      Panel 1 (large)  — price + EMA 9/21/50 + Bollinger Bands + VWAP
      Panel 2 (medium) — RSI(14) with 30/42/58/70 threshold lines
      Panel 3 (medium) — MACD line / signal / histogram
    """
    dfc = df.tail(show_bars).copy()
    idx = dfc.index

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#0e1117")

    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1.1, 1.1], hspace=0.06)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    dark_bg = "#0e1117"
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(dark_bg)
        ax.tick_params(colors="#aaa", labelsize=7.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#333")
        ax.spines["bottom"].set_color("#333")
        ax.grid(alpha=0.08, color="#444")

    closes = dfc["Close"].values

    # ── Panel 1 : Price ──────────────────────────────────────────────────────
    # Bollinger Band fill
    ax1.fill_between(idx, dfc["bb_upper"], dfc["bb_lower"],
                     alpha=0.07, color="steelblue", zorder=1)
    ax1.plot(idx, dfc["bb_upper"], lw=0.7, color="#5c9bd6", alpha=0.55,
             linestyle="--", zorder=2)
    ax1.plot(idx, dfc["bb_lower"], lw=0.7, color="#5c9bd6", alpha=0.55,
             linestyle="--", zorder=2)
    ax1.plot(idx, dfc["bb_mid"],   lw=0.9, color="#546e7a", alpha=0.5,
             linestyle=":", label="BB(20)", zorder=2)

    # VWAP
    ax1.plot(idx, dfc["vwap"], lw=1.3, color="#FFD700", alpha=0.75,
             linestyle="--", label="VWAP", zorder=3)

    # EMA stack
    ax1.plot(idx, dfc["ema50"], lw=1.0, color="#ab47bc", alpha=0.65,
             linestyle="--", label="EMA 50", zorder=4)
    ax1.plot(idx, dfc["ema21"], lw=1.4, color="#ffa726", alpha=0.85,
             label="EMA 21", zorder=4)
    ax1.plot(idx, dfc["ema9"],  lw=1.6, color="#29b6f6", alpha=0.95,
             label="EMA 9",  zorder=4)

    # Close price line
    ax1.plot(idx, closes, lw=2.2, color="white", alpha=0.95,
             label="XAU/USD 1H", zorder=5)

    # Signal marker on latest bar
    if signal:
        sig_col = _COLORS[signal["color_key"]]
        marker  = "^" if signal["total_score"] >= 0 else "v"
        ax1.scatter([idx[-1]], [closes[-1]], color=sig_col, s=220,
                    marker=marker, zorder=10,
                    edgecolors="white", linewidths=0.8)
        ax1.annotate(
            f"  {signal['action']}",
            xy=(idx[-1], closes[-1]),
            xytext=(10, 0), textcoords="offset points",
            color=sig_col, fontsize=9, fontweight="bold", va="center",
        )

    ax1.set_ylabel("USD / oz", color="#aaa", fontsize=8.5)
    ax1.legend(loc="upper left", framealpha=0.18, labelcolor="#ccc",
               facecolor="#111", edgecolor="#333", fontsize=7.5, ncol=5)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── Panel 2 : RSI ────────────────────────────────────────────────────────
    rsi = dfc["rsi"]
    ax2.plot(idx, rsi, lw=1.6, color="#e91e63", label="RSI(14)", zorder=3)
    for lvl, col, ls in [(70, "#ef5350", "--"), (58, "#ff7043", ":"),
                          (50, "#666",    "-"),  (42, "#66bb6a", ":"),
                          (30, "#4caf50", "--")]:
        ax2.axhline(lvl, color=col, lw=0.75, linestyle=ls, alpha=0.7, zorder=1)
    ax2.fill_between(idx, rsi, 70, where=(rsi > 70),
                     alpha=0.22, color="#ef5350", zorder=2)
    ax2.fill_between(idx, rsi, 30, where=(rsi < 30),
                     alpha=0.22, color="#4caf50", zorder=2)
    ax2.set_ylim(0, 100)
    ax2.set_yticks([30, 50, 70])
    ax2.set_ylabel("RSI", color="#aaa", fontsize=8)
    ax2.legend(loc="upper left", framealpha=0.18, labelcolor="#ccc",
               facecolor="#111", edgecolor="#333", fontsize=7.5)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # ── Panel 3 : MACD ───────────────────────────────────────────────────────
    macd_line = dfc["macd"]
    sig_line  = dfc["macd_sig"]
    hist      = dfc["macd_hist"]
    bar_colors = ["#4caf50" if h >= 0 else "#ef5350" for h in hist]
    bar_w      = (idx[1] - idx[0]).total_seconds() / 86400 * 0.6 if len(idx) > 1 else 0.025
    ax3.bar(idx, hist, color=bar_colors, alpha=0.55, width=bar_w, zorder=2,
            label="Histogram")
    ax3.plot(idx, macd_line, lw=1.5, color="#29b6f6", label="MACD",   zorder=3)
    ax3.plot(idx, sig_line,  lw=1.3, color="#ffa726", linestyle="--",
             label="Signal", zorder=3)
    ax3.axhline(0, color="#555", lw=0.6, zorder=1)
    ax3.set_ylabel("MACD", color="#aaa", fontsize=8)
    ax3.legend(loc="upper left", framealpha=0.18, labelcolor="#ccc",
               facecolor="#111", edgecolor="#333", fontsize=7.5, ncol=3)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))
    ax3.tick_params(axis="x", labelsize=7)

    fig.suptitle("Spot Gold (XAU/USD) — 1-Hour Intraday Technical Chart",
                 color="#ddd", fontsize=11.5, y=0.995)
    plt.subplots_adjust(left=0.055, right=0.975, top=0.975, bottom=0.055)
    return fig


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def get_day_trading_signal(period: str = "60d",
                           interval: str = "1h") -> tuple[pd.DataFrame | None, dict | None]:
    """Download OHLCV data, compute indicators, return (df_with_indicators, signal_dict).

    Data source priority:
      1. Twelve Data (TWELVE_DATA_KEY env var) — true XAU/USD spot, zero delay
      2. yfinance GC=F (fallback) — gold futures, ~15-min delay, futures premium

    HTF context is always fetched via yfinance (daily/4H history) to conserve
    Twelve Data API credits for the primary signal bars.
    """
    # ── Primary OHLCV fetch (Twelve Data → yfinance fallback) ────────────────
    df_raw = None
    _data_source = "yfinance"
    if interval in _TD_INTERVAL_MAP:          # only intervals TD supports
        df_raw = fetch_intraday_td(interval=interval)
        if df_raw is not None:
            _data_source = "twelvedata"
    if df_raw is None:
        df_raw = fetch_intraday(period=period, interval=interval)
    if df_raw is None:
        return None, None
    df = compute_indicators(df_raw)

    # Build higher-timeframe context for intraday signals ─────────────────────
    # 1H  signals: confirmed against daily EMA200 + 4H trend.
    # 15M signals: confirmed against 1H trend + daily EMA200.
    #   A 15-min BUY that aligns with a 1H uptrend is far more reliable than
    #   one that trades against the hourly flow.
    htf_context = None
    if interval == "1h":
        try:
            _d_raw = fetch_intraday(period="2y", interval="1d")
            if _d_raw is not None:
                _d_df = compute_indicators(_d_raw)
                if "ema200" in _d_df.columns:
                    _cl  = float(_d_df.iloc[-1]["Close"])
                    _e2  = float(_d_df.iloc[-1]["ema200"])
                    _e2s = float(_d_df["ema200"].iloc[-21]) if len(_d_df) >= 21 else _e2
                    htf_context = {
                        "ema200_dist_pct":  (_cl - _e2) / (_e2 + 1e-9) * 100,
                        "ema200_slope_pct": (_e2 - _e2s) / (_e2s + 1e-9) * 100,
                    }
        except Exception:
            pass

        # 4H intermediate timeframe — EMA stack + MACD + RSI on 4-hour bars.
        # Adds a confirmation layer between the daily macro trend and 1H signals.
        # A 1H BUY aligned with a 4H uptrend is significantly more reliable.
        try:
            _4h_raw = fetch_intraday(period="60d", interval="4h")
            if _4h_raw is not None and len(_4h_raw) >= 20:
                _4h_df  = compute_indicators(_4h_raw)
                _4h_row = _4h_df.iloc[-1]
                _4h_e9  = float(_4h_row.get("ema9",  _4h_row["Close"]))
                _4h_e21 = float(_4h_row.get("ema21", _4h_row["Close"]))
                _4h_e50 = float(_4h_row.get("ema50", _4h_row["Close"]))
                _4h_rsi = float(_4h_row.get("rsi", 50))
                _4h_mac = float(_4h_row.get("macd", 0))
                _4h_sig = float(_4h_row.get("macd_sig", 0))
                if htf_context is None:
                    htf_context = {}
                htf_context["4h_ema_trend"]    = (1 if _4h_e9 > _4h_e21 > _4h_e50
                                                   else (-1 if _4h_e9 < _4h_e21 < _4h_e50
                                                         else 0))
                htf_context["4h_macd_bullish"] = 1 if _4h_mac > _4h_sig else -1
                htf_context["4h_rsi"]          = _4h_rsi
        except Exception:
            pass

    elif interval in ("15m", "30m"):
        # ── 15/30-min higher-timeframe: 1H trend + daily EMA200 ─────────────
        # Step 1: 1H bars → EMA 9/21/50 stack and MACD for trend filter.
        try:
            _1h_raw = fetch_intraday(period="60d", interval="1h")
            if _1h_raw is not None and len(_1h_raw) >= 30:
                _1h_df  = compute_indicators(_1h_raw)
                _1h_row = _1h_df.iloc[-1]
                _1h_e9  = float(_1h_row.get("ema9",  _1h_row["Close"]))
                _1h_e21 = float(_1h_row.get("ema21", _1h_row["Close"]))
                _1h_e50 = float(_1h_row.get("ema50", _1h_row["Close"]))
                _1h_rsi = float(_1h_row.get("rsi", 50))
                _1h_mac = float(_1h_row.get("macd", 0))
                _1h_sig_v = float(_1h_row.get("macd_sig", 0))
                htf_context = {
                    "4h_ema_trend":    (1 if _1h_e9 > _1h_e21 > _1h_e50
                                        else (-1 if _1h_e9 < _1h_e21 < _1h_e50
                                              else 0)),
                    "4h_macd_bullish": 1 if _1h_mac > _1h_sig_v else -1,
                    "4h_rsi":          _1h_rsi,
                }
        except Exception:
            pass

        # Step 2: Daily EMA200 distance so oscillators aren't contrarian in
        # a strong macro trend (same logic as 1H uses).
        try:
            _d_raw = fetch_intraday(period="2y", interval="1d")
            if _d_raw is not None:
                _d_df = compute_indicators(_d_raw)
                if "ema200" in _d_df.columns:
                    _cl  = float(_d_df.iloc[-1]["Close"])
                    _e2  = float(_d_df.iloc[-1]["ema200"])
                    _e2s = float(_d_df["ema200"].iloc[-21]) if len(_d_df) >= 21 else _e2
                    if htf_context is None:
                        htf_context = {}
                    htf_context["ema200_dist_pct"]  = (_cl - _e2) / (_e2 + 1e-9) * 100
                    htf_context["ema200_slope_pct"] = (_e2 - _e2s) / (_e2s + 1e-9) * 100
        except Exception:
            pass

    elif interval == "4h":
        # ── 4H higher-timeframe: daily EMA200 + weekly trend ──────────────────
        # The 4H chart's natural HTF is the daily chart.  A 4H BUY that aligns
        # with the daily uptrend (price above EMA200, rising EMA slope) is far
        # more reliable than one that fights the macro trend.
        try:
            _d_raw = fetch_intraday(period="2y", interval="1d")
            if _d_raw is not None:
                _d_df = compute_indicators(_d_raw)
                if "ema200" in _d_df.columns:
                    _cl  = float(_d_df.iloc[-1]["Close"])
                    _e2  = float(_d_df.iloc[-1]["ema200"])
                    _e2s = float(_d_df["ema200"].iloc[-21]) if len(_d_df) >= 21 else _e2
                    # Also look at daily EMA9/21 stack for shorter-term daily trend
                    _d_e9  = float(_d_df.iloc[-1].get("ema9",  _cl))
                    _d_e21 = float(_d_df.iloc[-1].get("ema21", _cl))
                    _d_mac = float(_d_df.iloc[-1].get("macd", 0))
                    _d_sig_v = float(_d_df.iloc[-1].get("macd_sig", 0))
                    htf_context = {
                        "ema200_dist_pct":  (_cl - _e2) / (_e2 + 1e-9) * 100,
                        "ema200_slope_pct": (_e2 - _e2s) / (_e2s + 1e-9) * 100,
                        # Reuse 4h_ema_trend key (same scoring logic in score_signals)
                        "4h_ema_trend":     (1 if _d_e9 > _d_e21
                                             else (-1 if _d_e9 < _d_e21 else 0)),
                        "4h_macd_bullish":  1 if _d_mac > _d_sig_v else -1,
                        "4h_rsi":           float(_d_df.iloc[-1].get("rsi", 50)),
                    }
        except Exception:
            pass

    # Weekly bars use the same scoring thresholds as daily (position-trade style)
    _score_interval = "1d" if interval == "1wk" else interval
    signal = build_signal(df, interval=_score_interval, htf_context=htf_context)
    if signal is not None:
        signal["data_source"] = _data_source

    # ── Fix 2: 15-min momentum confirmation for 1-hour signals ───────────────
    # The 1H signal reflects the macro/indicator direction, but if the 15-min
    # EMA stack and recent price action are moving the *opposite* way, you are
    # trying to buy into a falling knife.  Only enter a 1H BUY when the 15-min
    # is also ticking up (EMA9 > EMA21 and last 3 bars rising).
    if interval == "1h" and signal is not None:
        try:
            _15m_raw = None
            if "15m" in _TD_INTERVAL_MAP:
                _15m_raw = fetch_intraday_td(interval="15m")
            if _15m_raw is None:
                _15m_raw = fetch_intraday(period="5d", interval="15m")
            if _15m_raw is not None and len(_15m_raw) >= 20:
                _15m_df  = compute_indicators(_15m_raw)
                _15m_row = _15m_df.iloc[-1]
                _15m_e9  = float(_15m_row.get("ema9",  _15m_row["Close"]))
                _15m_e21 = float(_15m_row.get("ema21", _15m_row["Close"]))
                _15m_mac = float(_15m_row.get("macd",     0))
                _15m_sig = float(_15m_row.get("macd_sig", 0))
                _15m_cls = _15m_df["Close"].iloc[-4:].values.astype(float)
                _15m_ema_bull = _15m_e9 > _15m_e21
                _15m_mom_bull = _15m_cls[-1] > _15m_cls[0]   # 3-bar direction
                _15m_mac_bull = _15m_mac > _15m_sig
                _1h_bull = signal.get("total_score", 0) > 0
                # Both EMA stack AND short-term momentum must agree
                _15m_agrees   = (_15m_ema_bull == _1h_bull) and (_15m_mom_bull == _1h_bull)
                _15m_opposes  = (_15m_ema_bull != _1h_bull) and (_15m_mom_bull != _1h_bull)
                if _15m_agrees:
                    signal["m15_confirmation"] = "confirms"
                elif _15m_opposes:
                    signal["m15_confirmation"] = "conflicts"
                else:
                    signal["m15_confirmation"] = "mixed"
                signal["m15_ema_bull"] = _15m_ema_bull
                signal["m15_mac_bull"] = _15m_mac_bull
            else:
                signal["m15_confirmation"] = "unknown"
        except Exception:
            signal["m15_confirmation"] = "unknown"

    return df, signal
