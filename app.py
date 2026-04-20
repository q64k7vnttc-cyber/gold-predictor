"""
app.py - Gold Price Predictor  |  streamlit run app.py --server.port 8000
Results are loaded from disk (written by the background scheduler) so the
page shows instantly without waiting for the model to run.
"""
import json
import pickle
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
import yfinance as yf

warnings.filterwarnings("ignore")

from day_trading import get_day_trading_signal, build_intraday_chart
from adaptive_learning import (
    analyze_and_adjust, load_weights, load_analysis_log, summary_stats,
)
from gold_model import (
    CACHE_DIR, HORIZONS,
    load_all_data, make_features, walk_forward,
    load_live_predictions, save_live_prediction,
    resolve_live_predictions, live_accuracy_stats,
    load_multi_horizon_predictions,
)

RESULTS_FILE        = CACHE_DIR / "latest_results.pkl"
STATE_FILE          = CACHE_DIR / "scheduler_state.json"
PROGRESS_FILE       = CACHE_DIR / "scheduler_progress.json"
INTRADAY_PRED_FILE  = CACHE_DIR / "intraday_predictions.json"

st.set_page_config(
    page_title="Gold Price Predictor",
    page_icon="🥇",
    layout="wide",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": """
# Gold Price Predictor · How It Works

## Overview
This app predicts the **direction** of XAU/USD (gold spot) over multiple time horizons —
1 hour, 1 day, 1 week, and 1 month — using an ensemble of three machine-learning models
trained on 600 + global market and macro variables.  Every prediction comes with a
**conviction score** (0 = coin-flip, 100% = certain) and a **live accuracy dashboard**
that tracks how past calls resolved.

---

## Price Feed
The live spot price is sourced from three independent feeds and combined into a
**median-of-3** to eliminate outliers:

| Source | Type |
|---|---|
| Swissquote interbank XAU/USD | Real-time bid/ask mid |
| Stooq XAU/USD | Spot closing price |
| Yahoo Finance GC=F | COMEX front-month futures |

A WebSocket stream (OKX XAUT/USDT) provides **tick-level updates** between REST polls.
Because tokenised gold (XAUT) can trade $30–50 below interbank spot during risk-off,
the WS price is immediately calibrated on the first tick using the server-computed
seed price — so the ticker always reflects real spot, not the crypto-market discount.

---

## Machine-Learning Model
**Ensemble of 3 algorithms**, each trained independently and combined by adaptive voting:

- **XGBoost** — gradient-boosted trees; excels at non-linear interactions between macro variables
- **Random Forest** — bagged trees; robust to noisy features and outliers
- **MLP (Neural Net)** — two-hidden-layer perceptron; captures complex cross-asset patterns

### Walk-Forward Backtesting
The model uses **expanding-window walk-forward validation** — it is never shown future data:

1. Train on the oldest 70% of history
2. Predict the next bar (out-of-sample)
3. Roll the window forward by one bar and retrain
4. Repeat until the full dataset is covered

This gives a realistic estimate of live accuracy without look-ahead bias.

### Feature Engineering (~600 variables)
The 600 + raw market series are transformed into predictive features:

- **Returns & velocities** — 1 d, 2 d, 3 d, 5 d, 10 d, 21 d log-returns for every asset
- **Z-scores** — rolling 20-day and 60-day normalisation
- **Cross-asset spreads** — gold vs DXY, gold vs US10Y real yield, gold vs VIX
- **COT positioning** — commercial net short % rank, speculator long/short velocity
- **Real-yield signals** — US 10-year TIPS yield velocity (strongest 1-day gold driver)
- **VIX regime flags** — spike detection, rolling percentile
- **Momentum & mean-reversion** — RSI, Bollinger band position, distance from 200-day MA

The top 80 features are selected each walk-forward fold by XGBoost feature importance.

---

## Signals & Day Trading
Two timeframe signals are generated live:

| Signal | Bars used | Horizon |
|---|---|---|
| 1-Hour Trade | Hourly OHLCV | Next 1–4 hours |
| Day Trade | Daily OHLCV | Next 1–3 days |

Each signal shows a **direction** (STRONG BUY / BUY / HOLD / SELL / STRONG SELL),
a **target price** (TP) derived from ATR-scaled momentum, a **stop-loss** (SL), and a
**confidence** bar (0–100%).

When both timeframes agree the **ALIGNED** badge lights up — historically, aligned
signals have higher resolution accuracy than single-timeframe calls.

---

## Candlestick Pattern Engine
The app recognises 20 + classic candlestick patterns (hammer, engulfing, doji, morning
star, shooting star, etc.) and converts them into a **pattern score** that feeds into the
macro intelligence layer and the Day Trading signal.

---

## Adaptive Self-Learning
After each resolved prediction (price crosses the target or the time window expires) the
app adjusts the **voting weights** of the three models.  If XGBoost has been outperforming
Random Forest recently, its weight is increased automatically.  Weights are stored in
`data_cache/adaptive_weights.json` and updated without retraining.

---

## Macro Intelligence Layer
A rule-based **macro intelligence** module scores the current environment across five
dimensions — dollar strength, real yields, risk appetite, commodity momentum, and
geopolitical stress — and overlays a qualitative context panel on the Live tab.

---

## Background Scheduler
A separate Python process (`scheduler.py`) runs a full walk-forward backtest every
**4 hours** while the app is running.  Each cycle:

1. Downloads fresh data for all 600 + series from Yahoo Finance & FRED
2. Retrains all three models on the expanded history
3. Generates new multi-horizon predictions and saves them to `data_cache/`
4. The Streamlit frontend picks up the new results automatically on the next page refresh

---

*Built with Python · Streamlit · XGBoost · scikit-learn · Plotly · yfinance · pandas*
""",
    },
)

# ─────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────
for key, default in [
    ("results",       None),
    ("last_run",      None),
    ("next_refresh",  None),
    ("horizon_label", "Next day"),
    ("source",        None),        # "scheduler" | "manual"
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────
# Load results from disk (scheduler output)
# ─────────────────────────────────────────────
def load_disk_results():
    if not RESULTS_FILE.exists():
        return None
    try:
        return pickle.loads(RESULTS_FILE.read_bytes())
    except Exception:
        return None


def load_scheduler_state():
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return None


def load_progress():
    if not PROGRESS_FILE.exists():
        return None
    try:
        return json.loads(PROGRESS_FILE.read_text())
    except Exception:
        return None


# ── Intraday prediction persistence & accuracy ────────────────────────────────
def load_intraday_preds():
    if not INTRADAY_PRED_FILE.exists():
        return []
    try:
        return json.loads(INTRADAY_PRED_FILE.read_text())
    except Exception:
        return []


def save_intraday_preds(preds):
    try:
        INTRADAY_PRED_FILE.write_text(json.dumps(preds, indent=2))
    except Exception:
        pass


def resolve_intraday_preds(preds, price_series):
    """Fill in actual_price + correct for any prediction whose target time has passed.
    When a prediction is wrong, triggers adaptive weight adjustment in the background."""
    changed = False
    now_utc = datetime.utcnow()
    for p in preds:
        if p.get("correct") is not None:
            continue
        target_dt = datetime.fromisoformat(p["target_timestamp"])
        if now_utc < target_dt:
            continue
        target_ts = pd.Timestamp(target_dt).tz_localize(None)
        series_idx = price_series.index
        if hasattr(series_idx, "tz") and series_idx.tz is not None:
            series_idx_naive = series_idx.tz_localize(None)
        else:
            series_idx_naive = series_idx
        after  = price_series[series_idx_naive >= target_ts]
        before = price_series[series_idx_naive <= target_ts]
        if not after.empty:
            actual = float(after.iloc[0])
        elif not before.empty:
            actual = float(before.iloc[-1])
        else:
            continue
        base   = p["price_at_prediction"]
        p_dir  = p["predicted_direction"]
        a_dir  = (1 if actual > base else (-1 if actual < base else 0))
        p["actual_price"] = actual
        p["correct"]      = bool(p_dir == a_dir) if p_dir != 0 else bool(a_dir == 0)
        changed = True
        # ── Background self-learning: adjust weights on wrong predictions ──
        if not p["correct"]:
            try:
                analyze_and_adjust(p)
            except Exception:
                pass
    if changed:
        save_intraday_preds(preds)
    return preds


def intraday_accuracy_by_horizon(preds):
    """Returns {horizon_label: (accuracy_float, n_resolved)}."""
    buckets: dict = {}
    for p in preds:
        lbl = p["horizon_label"]
        if lbl not in buckets:
            buckets[lbl] = {"ok": 0, "total": 0}
        if p.get("correct") is not None:
            buckets[lbl]["total"] += 1
            if p["correct"]:
                buckets[lbl]["ok"] += 1
    return {
        k: (v["ok"] / v["total"], v["total"]) if v["total"] > 0 else (None, 0)
        for k, v in buckets.items()
    }


# On first load pull results from disk so the page isn't blank
if st.session_state.results is None:
    disk = load_disk_results()
    if disk is not None:
        st.session_state.results       = disk
        st.session_state.horizon_label = disk.get("horizon_label", "Next day")
        run_at = disk.get("run_at")
        if run_at:
            st.session_state.last_run = datetime.fromisoformat(run_at).timestamp()
        st.session_state.source = "scheduler"


# ─────────────────────────────────────────────
# Live gold price (fast, always loads)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=1800)
def fetch_gold_price(period="2y"):
    try:
        df = yf.download("GC=F", period=period, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        close = df["Close"]
        if hasattr(close, "squeeze"):
            close = close.squeeze()
        return close.dropna()
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=3600)
def cached_load(start_str):
    return load_all_data(start=start_str)


@st.cache_data(show_spinner=False, ttl=2)
def fetch_live_price():
    """Multi-source parallel gold price with outlier rejection and median blending.

    Sources used (all return actual XAU/USD spot or near-spot):
      1. Swissquote — live interbank XAU/USD bid/ask (most accurate, ±$1 vs IG)
      2. Stooq     — XAU/USD OTC spot feed (within ±$3 of interbank)
      3. Yahoo GC=F — COMEX near-month futures (~$30-40 futures basis over spot)

    Previous sources (PAXG/Coinbase, XAUT/OKX, XAUT/Gate.io) were crypto-market
    tokenized gold that can trade $30–50 below actual spot during risk-off events
    (crypto selling pressure), causing systematic underpricing vs IG CFD.
    """
    import urllib.request, csv, io, statistics
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _hdrs = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

    def _get(url):
        req = urllib.request.Request(url, headers=_hdrs)
        with urllib.request.urlopen(req, timeout=4) as r:
            return json.loads(r.read())

    def _swissquote():
        """Swissquote interbank XAU/USD — real-time forex spot, free, no API key."""
        d = _get("https://forex-data-feed.swissquote.com/public-quotes/bboquotes/instrument/XAU/USD")
        # d is a list of venue dicts; use the first profile's bid/ask mid
        for venue in d:
            for profile in venue.get("spreadProfilePrices", []):
                bid = profile.get("bid")
                ask = profile.get("ask")
                if bid and ask:
                    mid = (float(bid) + float(ask)) / 2
                    return "Swissquote·XAU/USD", mid, None
        raise ValueError("no prices in Swissquote response")

    def _stooq():
        req = urllib.request.Request(
            "https://stooq.com/q/l/?s=xauusd&f=sd2t2ohlcv&h&e=csv", headers=_hdrs)
        with urllib.request.urlopen(req, timeout=5) as r:
            reader = csv.DictReader(io.StringIO(r.read().decode()))
            row = next(reader)
        return "stooq·XAU/USD", float(row["Close"]), float(row["Open"])

    def _yahoo_gcf():
        d = _get("https://query2.finance.yahoo.com/v8/finance/chart/GC%3DF?interval=1m&range=1d")
        m = d["chart"]["result"][0]["meta"]
        return "Yahoo·GC=F", float(m["regularMarketPrice"]), float(m.get("chartPreviousClose", 0))

    fetchers = [_swissquote, _stooq, _yahoo_gcf]
    results  = {}   # name -> (price, prev_close_or_None)

    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(fn): fn.__name__ for fn in fetchers}
        for fut in as_completed(futs, timeout=4):
            try:
                name, price, prev = fut.result()
                if price and price > 100:
                    results[name] = (price, prev)
            except Exception:
                pass

    if not results:
        return None

    vals = sorted(p for p, _ in results.values())
    if len(vals) == 1:
        med_raw = vals[0]
    else:
        mid = len(vals) // 2
        med_raw = (vals[mid - 1] + vals[mid]) / 2 if len(vals) % 2 == 0 else vals[mid]

    # Keep only prices within 1.5% of the raw median (removes clear outliers)
    consensus = {n: (p, pc) for n, (p, pc) in results.items()
                 if abs(p - med_raw) / med_raw <= 0.015}
    if not consensus:
        consensus = results

    c_prices = sorted(p for p, _ in consensus.values())
    mid2 = len(c_prices) // 2
    final_price = (
        (c_prices[mid2 - 1] + c_prices[mid2]) / 2
        if len(c_prices) % 2 == 0
        else c_prices[mid2]
    )

    # Best prev_close: prefer stooq Open (daily anchor), else first available
    prev_close = final_price
    for name in ("stooq·XAU/USD", "Yahoo·GC=F", "Swissquote·XAU/USD"):
        if name in consensus and consensus[name][1]:
            prev_close = consensus[name][1]
            break

    src_str = " · ".join(sorted(consensus.keys()))
    n_total = len(results)
    n_used  = len(consensus)

    return {
        "price":      round(final_price, 2),
        "prev_close": round(prev_close,  2),
        "source":     f"Median of {n_used}/{n_total} sources · {src_str}",
        "ts":         datetime.utcnow().strftime("%H:%M:%S UTC"),
        "_all":       {n: p for n, (p, _) in results.items()},
    }


@st.cache_data(show_spinner=False, ttl=15)
def fetch_intraday_5m():
    """5-minute bars for the current session (last 1 day available)."""
    try:
        df = yf.download("GC=F", period="1d", interval="5m",
                         progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        close = df["Close"]
        if hasattr(close, "squeeze"):
            close = close.squeeze()
        return close.dropna()
    except Exception:
        return None


# ─────────────────────────────────────────────
# Resolve live predictions
# ─────────────────────────────────────────────
gold_2y = fetch_gold_price("2y")
live_preds = (resolve_live_predictions(gold_2y)
              if gold_2y is not None else load_live_predictions())
live_acc, live_n, _ = live_accuracy_stats(live_preds)
pending = [p for p in live_preds if p["outcome"] is None]
mh_preds = load_multi_horizon_predictions()   # {1d, 2d, 5d} forecasts

# ── Accuracy for topbar badges ─────────────────────────────────────────────
_top_ipreds = load_intraday_preds()
_top_iacc   = intraday_accuracy_by_horizon(_top_ipreds)
_1h_acc_pct, _1h_acc_n = _top_iacc.get("1 hour", (None, 0))
_day_acc_pct, _day_acc_n = (live_acc if live_n > 0 else None), live_n

# ─────────────────────────────────────────────
# Scheduler state
# ─────────────────────────────────────────────
sched    = load_scheduler_state()
progress = load_progress()
scheduler_running = (sched is not None and
                     sched.get("status") in ("running", "refreshing"))

# ─────────────────────────────────────────────
# Signal caches — two timeframes
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=30)    # refreshes every 30 s
def _load_signal_1h():
    """1-hour bars → signal for short intraday trades (~1 h hold)."""
    return get_day_trading_signal(period="60d", interval="1h")


@st.cache_data(show_spinner=False, ttl=120)   # refreshes every 2 min
def _load_signal_daily():
    """Daily bars → signal for full-day trades (hold through the session)."""
    return get_day_trading_signal(period="2y", interval="1d")


@st.cache_data(show_spinner=False, ttl=30)    # refreshes every 30 s
def _load_signal_15m():
    """15-minute bars → signal for scalp trades (~15-30 min hold)."""
    return get_day_trading_signal(period="5d", interval="15m")


@st.cache_data(show_spinner=False, ttl=45)    # refreshes every 45 s
def _load_signal_30m():
    """30-minute bars → signal for short intraday trades (~30-60 min hold)."""
    return get_day_trading_signal(period="10d", interval="30m")


@st.cache_data(show_spinner=False, ttl=60)    # refreshes every 60 s
def _load_signal_4h():
    """4-hour bars → signal for swing/position trades (~4-24h hold)."""
    return get_day_trading_signal(period="60d", interval="4h")


@st.cache_data(show_spinner=False, ttl=900)  # refreshes every 15 min (weekly bars move slowly)
def _load_signal_weekly():
    """Weekly bars → signal for position trades (multi-day / weekly hold)."""
    return get_day_trading_signal(period="5y", interval="1wk")


# ── Load all signals ─────────────────────────────────────────────────────────
_top_df, _top_sig  = _load_signal_1h()      # 1-hour bars  (intraday / 1-h trades)
_day_df, _day_sig  = _load_signal_daily()   # daily bars   (full-day / day trades)
_15m_df, _15m_sig  = _load_signal_15m()     # 15-min bars  (scalp trades)
_30m_df, _30m_sig  = _load_signal_30m()     # 30-min bars  (short intraday trades)
_4h_df,  _4h_sig   = _load_signal_4h()      # 4-hour bars  (swing trades)
_wk_df,  _wk_sig   = _load_signal_weekly()  # weekly bars  (position / week+ trades)


def _sig_style(sig):
    """Return (label, bg, fg) for a signal dict or None.
    Uses the action field from build_signal() for full consistency with signal cards.
    Falls back to total_score only for directional lean when action is NEUTRAL.
    """
    if sig is None:
        return "WAIT", "#1a1f2e", "#607d8b"
    action = sig.get("action", "NEUTRAL")
    if   action == "STRONG BUY":  return "STRONG BUY",  "#0a2e1a", "#00e676"
    elif action == "BUY":         return "BUY",         "#0d3320", "#4caf50"
    elif action == "STRONG SELL": return "STRONG SELL", "#2e0a0a", "#ff1744"
    elif action == "SELL":        return "SELL",        "#3b0d0d", "#ef5350"
    else:
        ts = sig.get("total_score", 0)
        if   ts >  0.5: return "LEAN BUY",  "#161e16", "#78909c"
        elif ts < -0.5: return "LEAN SELL", "#1e1616", "#78909c"
        else:           return "HOLD",      "#1a1f2e", "#607d8b"


_1h_label,  _1h_bg,  _1h_fg  = _sig_style(_top_sig)
_day_label, _day_bg, _day_fg = _sig_style(_day_sig)
_15m_label, _15m_bg, _15m_fg = _sig_style(_15m_sig)
_30m_label, _30m_bg, _30m_fg = _sig_style(_30m_sig)
_4h_label,  _4h_bg,  _4h_fg  = _sig_style(_4h_sig)
_wk_label,  _wk_bg,  _wk_fg  = _sig_style(_wk_sig)

# Fetch live price now so the top bar shows the most current value
_top_live      = fetch_live_price()
_top_live_price = _top_live["price"] if _top_live else None


def _acc_badge_html(acc_pct, acc_n, label):
    """Compact vertical accuracy badge for the topbar."""
    if acc_pct is None or acc_n == 0:
        pct_str  = "—"
        pct_color = "#888"
        n_str    = "no data"
    else:
        pct_str   = f"{acc_pct:.0%}"
        pct_color = "#4caf50" if acc_pct >= 0.55 else ("#ef5350" if acc_pct < 0.45 else "#ffc107")
        n_str     = f"{acc_n} resolved"
    return (
        f'<div style="display:flex;flex-direction:column;align-items:center;'
        f'background:#12161f;border:1px solid #2a2a2a;border-radius:8px;'
        f'padding:3px 10px;min-width:90px;">'
        f'<div style="font-size:9px;color:#9ba8bc;letter-spacing:1px;text-transform:uppercase;'
        f'margin-bottom:1px;">{label}</div>'
        f'<div style="font-size:18px;font-weight:900;color:{pct_color};font-family:monospace;'
        f'line-height:1.1;">{pct_str}</div>'
        f'<div style="font-size:9px;color:#8a9ab5;margin-top:1px;">{n_str}</div>'
        f'</div>'
    )


def _panel_html(sig, label, bg, fg, title, acc_pct=None, acc_n=0):
    is_actionable = label in ("BUY", "SELL", "STRONG BUY", "STRONG SELL")

    if sig is None:
        pred_line  = ""
        now_line   = ""
        detail     = "No data"
        strength_bar = ""
    else:
        _now_p = _top_live_price if _top_live_price else sig["entry"]
        _adj   = (_top_live_price - sig["entry"]) if _top_live_price else 0.0
        _tp    = sig["target"]    + _adj
        _sl    = sig["stop_loss"] + _adj

        now_line = (
            f"<div style='font-size:10px;color:#888;font-family:monospace;margin:1px 0;'>"
            f"Now&nbsp;<b class='tb-now-price' style='color:#e2e8f0'>${_now_p:,.2f}</b></div>"
        )

        # Dim TP/SL target line when signal is sub-threshold
        _target_color = fg if is_actionable else "#555"
        pred_line = (
            f"<div style='font-size:13px;font-weight:700;color:{_target_color};font-family:monospace;"
            f"letter-spacing:1px;margin:1px 0;'>⟶ ${_tp:,.2f}</div>"
        )

        _tp_color = "#4caf50" if is_actionable else "#445"
        _sl_color = "#ef5350" if is_actionable else "#445"
        _tp_pct   = abs(_tp - _now_p) / _now_p if _now_p else 0
        _tp_warn  = (
            f"<div style='font-size:8px;color:#ff9800;margin-top:2px;'>"
            f"⚠ High-ATR environment — TP is {_tp_pct:.1%} from entry. "
            f"Consider a partial target at 50%.</div>"
            if _tp_pct > 0.03 and is_actionable else ""
        )
        detail = (
            f"TP&nbsp;<b style='color:{_tp_color}'>${_tp:,.2f}</b>"
            f"&nbsp;&nbsp;SL&nbsp;<b style='color:{_sl_color}'>${_sl:,.2f}</b>"
            + _tp_warn
        )

        # Confidence bar
        _conf     = sig.get("confidence", 0.0)
        _bar_w    = max(2, int(_conf * 100))
        _bar_col  = fg if is_actionable else "#444"
        _conf_txt = f"{_conf:.0%}"
        strength_bar = (
            f'<div style="width:100%;margin-top:3px;">'
            f'<div style="display:flex;justify-content:space-between;font-size:8px;color:#8a9ab5;margin-bottom:1px;">'
            f'<span>Confidence</span><span style="color:{_bar_col}">{_conf_txt}</span></div>'
            f'<div style="width:100%;height:3px;background:#1e2330;border-radius:2px;">'
            f'<div style="width:{_bar_w}%;height:3px;background:{_bar_col};border-radius:2px;'
            f'transition:width 0.4s;"></div></div></div>'
        )

    # Accuracy line
    if acc_pct is not None and acc_n > 0:
        _ac = "#4caf50" if acc_pct >= 0.55 else ("#ef5350" if acc_pct < 0.45 else "#ffc107")
        acc_line = (
            f'<div style="font-size:9px;color:#8a9ab5;margin-top:2px;border-top:1px solid #ffffff12;'
            f'padding-top:2px;width:100%;text-align:center;">'
            f'Accuracy&nbsp;<b style="color:{_ac}">{acc_pct:.0%}</b>'
            f'&nbsp;<span style="color:#3a3a3a">·&nbsp;{acc_n} resolved</span></div>'
        )
    else:
        acc_line = (
            f'<div style="font-size:9px;color:#6a7a94;margin-top:2px;border-top:1px solid #ffffff12;'
            f'padding-top:2px;width:100%;text-align:center;">Accuracy&nbsp;—</div>'
        )

    _inner = "".join([
        f'<div style="font-size:9px;color:#9ba8bc;letter-spacing:1px;text-transform:uppercase;margin-bottom:1px;">{title}</div>',
        f'<div style="font-size:15px;font-weight:900;color:{fg};letter-spacing:2px;font-family:monospace;">{label}</div>',
        now_line,
        pred_line,
        f'<div style="font-size:10px;color:#888;margin-top:1px;">{detail}</div>',
        strength_bar,
        acc_line,
    ])
    return (f'<div class="tb-panel" style="display:flex;flex-direction:column;align-items:center;'
            f'background:{bg};border:1px solid {fg}44;border-radius:8px;'
            f'padding:4px 14px;min-width:210px;">{_inner}</div>')


# ── Confluence indicator ─────────────────────────────────────────────────────
def _confluence_html(lbl1, lbl2):
    actionable = {"BUY", "SELL", "STRONG BUY", "STRONG SELL"}
    a1 = lbl1 in actionable
    a2 = lbl2 in actionable
    bull = {"BUY", "STRONG BUY"}
    bear = {"SELL", "STRONG SELL"}
    if a1 and a2:
        if (lbl1 in bull and lbl2 in bull) or (lbl1 in bear and lbl2 in bear):
            icon, txt, col, bg = "✓", "ALIGNED", "#4caf50", "#0d2010"
            sub = "1h & daily agree"
        else:
            icon, txt, col, bg = "⚠", "CONFLICT", "#ff9800", "#2a1800"
            sub = "signals oppose each other"
    elif not a1 and not a2:
        icon, txt, col, bg = "·", "NO SIGNAL", "#607d8b", "#12161f"
        sub = "no clear direction"
    else:
        icon, txt, col, bg = "~", "MIXED", "#78909c", "#161a20"
        sub = "only one timeframe active"
    return (
        f'<div class="tb-conf" style="display:flex;flex-direction:column;align-items:center;justify-content:center;'
        f'background:{bg};border:1px solid {col}44;border-radius:6px;padding:4px 10px;min-width:72px;">'
        f'<div style="font-size:14px;color:{col};">{icon}</div>'
        f'<div style="font-size:8px;color:{col};letter-spacing:1px;font-weight:700;">{txt}</div>'
        f'<div style="font-size:7px;color:{col}99;text-align:center;margin-top:2px;line-height:1.2;">{sub}</div>'
        f'</div>'
    )

# ── COMEX Gold market hours indicator ────────────────────────────────────────
def _market_status_html():
    """
    COMEX Gold (GC=F) trading hours (ET):
      Sun 6 PM → Fri 5 PM  continuous, with a 60-min break (5–6 PM) each weekday.
      Sat all day + Fri 5 PM → Sun 6 PM = CLOSED.
    """
    import pytz
    ET   = pytz.timezone("America/New_York")
    AEST = pytz.timezone("Australia/Sydney")
    now  = datetime.now(ET)
    wd   = now.weekday()   # 0=Mon … 4=Fri, 5=Sat, 6=Sun
    tod  = now.hour * 60 + now.minute  # minutes since midnight ET

    BREAK_START = 17 * 60   # 5:00 PM ET
    BREAK_END   = 18 * 60   # 6:00 PM ET

    DAY_ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    def _fmt(dt):
        secs = max(0, int((dt - now).total_seconds()))
        h, m = divmod(secs // 60, 60)
        return f"{h}h {m:02d}m" if h else f"{m}m"

    def _aest(dt):
        """Return 'HH:MM AM/PM AEST/AEDT [Day]' for the given ET datetime."""
        local = dt.astimezone(AEST)
        tz_abbr = local.strftime("%Z")   # AEST or AEDT
        time_str = local.strftime("%-I:%M %p").lower().replace("am", "AM").replace("pm", "PM")
        # Only show day name when it differs from the current AEST day
        now_aest = now.astimezone(AEST)
        day_str = f" {DAY_ABBR[local.weekday()]}" if local.date() != now_aest.date() else ""
        return f"{time_str}{day_str} {tz_abbr}"

    def _next_weekday(day_from, target_wd, h, mi=0):
        """Return next datetime on target_wd at h:mi ET, from day_from."""
        days_ahead = (target_wd - day_from.weekday()) % 7 or 7
        d = day_from.replace(hour=h, minute=mi, second=0, microsecond=0)
        return d + timedelta(days=days_ahead)

    # Saturday: fully closed → opens Sunday 6 PM ET
    if wd == 5:
        open_dt = now.replace(hour=18, minute=0, second=0, microsecond=0) + timedelta(days=1)
        label, sub, dot, col, bg = "CLOSED", f"opens in {_fmt(open_dt)}", "🔴", "#ef5350", "#2e0d0d"
        aest_line = f"opens {_aest(open_dt)}"

    # Sunday: closed before 6 PM ET, open at/after 6 PM ET
    elif wd == 6:
        if tod < BREAK_END:
            open_dt = now.replace(hour=18, minute=0, second=0, microsecond=0)
            label, sub, dot, col, bg = "CLOSED", f"opens in {_fmt(open_dt)}", "🔴", "#ef5350", "#2e0d0d"
            aest_line = f"opens {_aest(open_dt)}"
        else:
            close_dt = (now + timedelta(days=1)).replace(hour=17, minute=0, second=0, microsecond=0)
            label, sub, dot, col, bg = "OPEN", f"closes in {_fmt(close_dt)}", "🟢", "#4caf50", "#0d2010"
            aest_line = f"closes {_aest(close_dt)}"

    # Friday: open before 5 PM ET, closed from 5 PM ET
    elif wd == 4:
        if tod < BREAK_START:
            close_dt = now.replace(hour=17, minute=0, second=0, microsecond=0)
            label, sub, dot, col, bg = "OPEN", f"closes in {_fmt(close_dt)}", "🟢", "#4caf50", "#0d2010"
            aest_line = f"closes {_aest(close_dt)}"
        else:
            open_dt = _next_weekday(now, 6, 18)
            label, sub, dot, col, bg = "CLOSED", f"opens in {_fmt(open_dt)}", "🔴", "#ef5350", "#2e0d0d"
            aest_line = f"opens {_aest(open_dt)}"

    # Monday–Thursday
    else:
        if BREAK_START <= tod < BREAK_END:
            resume_dt = now.replace(hour=18, minute=0, second=0, microsecond=0)
            label, sub, dot, col, bg = "BREAK", f"resumes in {_fmt(resume_dt)}", "🟡", "#ffc107", "#2a1800"
            aest_line = f"resumes {_aest(resume_dt)}"
        elif tod < BREAK_START:
            close_dt = now.replace(hour=17, minute=0, second=0, microsecond=0)
            label, sub, dot, col, bg = "OPEN", f"closes in {_fmt(close_dt)}", "🟢", "#4caf50", "#0d2010"
            aest_line = f"closes {_aest(close_dt)}"
        else:  # after 6 PM
            close_dt = (now + timedelta(days=1)).replace(hour=17, minute=0, second=0, microsecond=0)
            label, sub, dot, col, bg = "OPEN", f"closes in {_fmt(close_dt)}", "🟢", "#4caf50", "#0d2010"
            aest_line = f"closes {_aest(close_dt)}"

    return (
        f'<div class="tb-market" style="display:flex;flex-direction:column;align-items:center;justify-content:center;'
        f'background:{bg};border:2px solid {col}66;border-radius:8px;padding:6px 16px;min-width:120px;'
        f'box-shadow:0 0 10px {col}22;">'
        f'<div style="font-size:10px;color:#888;letter-spacing:1.5px;text-transform:uppercase;font-weight:600;margin-bottom:3px;">COMEX Gold</div>'
        f'<div style="font-size:18px;font-weight:900;color:{col};letter-spacing:2px;line-height:1;">{dot} {label}</div>'
        f'<div style="font-size:11px;color:{col}cc;margin-top:3px;white-space:nowrap;font-weight:500;">{sub}</div>'
        f'<div style="font-size:10px;color:#aaa;margin-top:2px;white-space:nowrap;">{aest_line}</div>'
        f'</div>'
    )


# Padding-top: desktop needs offset for fixed topbar; mobile topbar is in flow
st.markdown(
    '<style>'
    '.main .block-container{padding-top:155px!important;}'
    '@media(max-width:650px){.main .block-container{padding-top:8px!important;}}'
    '[data-testid="stApp"]{transition:none!important;opacity:1!important;}'
    '[data-testid="stApp"] *{transition:none!important;}'
    '.stApp{opacity:1!important;}'
    '</style>',
    unsafe_allow_html=True,
)

# Build the inner HTML for the topbar boxes
_topbar_inner = (
    _market_status_html()
    + '<div class="tb-divider" style="width:1px;height:48px;background:#222;flex-shrink:0;"></div>'
    + _panel_html(_top_sig, _1h_label, _1h_bg, _1h_fg, "1-Hour Trade · Hourly bars",
                  acc_pct=_1h_acc_pct, acc_n=_1h_acc_n)
    + _confluence_html(_1h_label, _day_label)
    + _panel_html(_day_sig, _day_label, _day_bg, _day_fg, "Day Trade · Daily bars",
                  acc_pct=_day_acc_pct, acc_n=_day_acc_n)
)

# CSS for the topbar (will be injected into the parent document head)
_topbar_css = (
    # Desktop: fixed overlay above Streamlit header area
    '#sig-topbar{'
    'position:fixed;top:52px;left:0;right:0;z-index:1000;'
    'background:#1a2235;border-bottom:1px solid #2a2a2a;'
    'display:flex;align-items:center;justify-content:center;'
    'gap:8px;padding:4px 16px;box-shadow:0 2px 8px rgba(0,0,0,0.5);'
    'flex-wrap:nowrap;}'
    # Mobile: sticky INSIDE the content flow — tabs appear below it naturally
    '@media(max-width:650px){'
    '#sig-topbar{'
    'position:sticky!important;top:52px!important;'
    'left:auto!important;right:auto!important;'
    'margin:-8px -1rem 8px -1rem;'  # bleed edge-to-edge inside block-container
    'width:calc(100% + 2rem);'
    'flex-wrap:wrap;gap:5px;padding:5px 6px;z-index:100;}'
    '.tb-market{order:1;flex:1 1 calc(50% - 5px);min-width:0!important;}'
    '.tb-conf{order:2;flex:1 1 calc(50% - 5px);min-width:0!important;}'
    '.tb-panel{order:3;flex:1 1 calc(50% - 5px);min-width:0!important;}'
    '.tb-divider{display:none!important;}}'
)

# Inject the topbar directly into window.parent.document.body via a hidden
# zero-height iframe. This bypasses Streamlit's CSS transform containers that
# break position:fixed when using st.markdown.
_tb_inner_js = json.dumps(_topbar_inner)
_tb_css_js   = json.dumps(_topbar_css)
components.html(
    f"""<script>
(function(){{
  try {{
    var doc = window.parent.document;
    // Inject/refresh topbar CSS
    var oldSt = doc.getElementById('sig-topbar-style');
    if (oldSt) oldSt.remove();
    var styleEl = doc.createElement('style');
    styleEl.id = 'sig-topbar-style';
    styleEl.textContent = {_tb_css_js};
    doc.head.appendChild(styleEl);

    var isMobile = window.parent.innerWidth <= 650;

    // Build (or reuse) the topbar element
    var tb = doc.getElementById('sig-topbar');
    if (!tb) {{
      tb = doc.createElement('div');
      tb.id = 'sig-topbar';
    }}
    tb.innerHTML = {_tb_inner_js};

    function _findBlockContainer() {{
      return doc.querySelector('.main .block-container') ||
             doc.querySelector('[data-testid="stMainBlockContainer"]') ||
             doc.querySelector('[data-testid="stAppViewBlockContainer"]') ||
             doc.querySelector('.block-container');
    }}

    function _insertMobile() {{
      // Insert topbar as first child of block-container so tabs flow below it
      var bc = _findBlockContainer();
      if (!bc) return;
      if (bc.firstChild !== tb) {{
        bc.insertBefore(tb, bc.firstChild);
      }}
    }}

    function _insertDesktop() {{
      if (tb.parentElement !== doc.body) {{
        doc.body.appendChild(tb);
      }}
      // Fix desktop layout: push content + sticky-pin tab bar below topbar
      var bottom = Math.ceil(tb.getBoundingClientRect().bottom);
      if (bottom < 10) return;
      var fixedTop = bottom + 2;
      var padTop = Math.max(130, bottom - 52 + 8);
      var old2 = doc.getElementById('sig-layout-fix');
      if (old2) old2.remove();
      var s = doc.createElement('style');
      s.id = 'sig-layout-fix';
      s.textContent =
        '.main .block-container{{padding-top:' + padTop + 'px!important;}}' +
        '[data-testid="stTabBar"]{{position:sticky!important;top:' + fixedTop + 'px!important;z-index:999!important;background:#1a2235!important;}}';
      doc.head.appendChild(s);
      var tabBar = doc.querySelector('[data-testid="stTabBar"]');
      if (tabBar) {{
        tabBar.style.setProperty('top', fixedTop + 'px', 'important');
      }}
    }}

    if (isMobile) {{
      _insertMobile();
    }} else {{
      _insertDesktop();
      setTimeout(_insertDesktop, 300);
      setTimeout(_insertDesktop, 800);
    }}

    // Re-apply after every Streamlit DOM update
    var _obs = new MutationObserver(function(muts) {{
      var changed = false;
      for (var i = 0; i < muts.length; i++) {{
        if (muts[i].addedNodes.length || muts[i].removedNodes.length) {{
          changed = true; break;
        }}
      }}
      if (!changed) return;
      if (isMobile) _insertMobile();
      else _insertDesktop();
    }});
    _obs.observe(doc.body, {{childList: true, subtree: true}});

    // ── Live price sync: keep topbar "Now" badges matching the ticker ──────
    // Reads the calibrated price from the ticker iframe and writes it to every
    // .tb-now-price badge.  Skips update when ticker price is >0.8% from the
    // Python seed price — this blocks uncalibrated WS prices (e.g. XAUT
    // tokenized gold at a crypto-market discount) from overwriting the accurate
    // server-rendered price before the Swissquote calibration has had a chance
    // to fire (which happens ~5-6 s after the WS connects).
    const _seedPrice = {_top_live_price or 0};
    function _syncNowPrices() {{
      try {{
        var frames = doc.querySelectorAll('iframe');
        for (var fi = 0; fi < frames.length; fi++) {{
          try {{
            var gtEl = frames[fi].contentDocument &&
                       frames[fi].contentDocument.getElementById('gt-price');
            if (gtEl) {{
              var raw = gtEl.textContent.replace(/[^0-9,.]/g, '').trim();
              if (raw) {{
                var tickerPrice = parseFloat(raw.replace(/,/g, ''));
                // Sanity check: skip if ticker price is >0.3% from the server seed.
                // Tightened from 0.8% → 0.3%: allows legitimate $15-18 intraday moves
                // through while still blocking the large XAUT crypto-discount (~0.5-1%
                // below interbank spot) before the Swissquote calibration fires.
                if (_seedPrice > 0 && Math.abs(tickerPrice - _seedPrice) / _seedPrice > 0.003) {{
                  break;  // ticker not yet calibrated — do not overwrite
                }}
                var badges = doc.querySelectorAll('.tb-now-price');
                for (var bi = 0; bi < badges.length; bi++) {{
                  badges[bi].textContent = '$' + raw;
                }}
              }}
              break;
            }}
          }} catch(ex) {{ /* cross-origin frame, skip */ }}
        }}
      }} catch(ex) {{}}
    }}
    setInterval(_syncNowPrices, 4000);
    // Delay first sync to 8 s, giving the WS basis calibration time to fire
    // before we try to read the ticker price (calibration fires ~5-6 s after
    // WS connects, which itself takes ~1-2 s after page load).
    setTimeout(_syncNowPrices, 8000);

  }} catch(e) {{ console.warn('topbar inject:', e); }}
}})();
</script>""",
    height=0,
    scrolling=False,
)

@st.fragment(run_every=5)
def _scheduler_progress_fragment():
    _sched    = load_scheduler_state()
    _progress = load_progress()
    _running  = _sched.get("status") in ("running", "refreshing") if _sched else False

    _status  = _sched.get("status", "unknown") if _sched else "no state file"
    _pct_txt = f" · {_progress.get('percent', 0):.0f}%" if (_progress and _running) else ""
    st.caption(f"🔄 Scheduler: **{_status}**{_pct_txt}")

    if _running and _progress is not None:
        pct    = float(_progress.get("percent", 0))
        msg    = _progress.get("message", "Working…")
        phase  = _progress.get("phase", "")
        updated = _progress.get("updated_at", "")

        phase_label = {
            "data":     "📥 Downloading data",
            "features": "⚙️  Building features",
            "training": "🧠 Training model",
            "saving":   "💾 Saving results",
            "done":     "✅ Complete",
            "init":     "🚀 Starting up",
            "quick":    "⚡ Quick refresh",
        }.get(phase, "⏳ Running")

        st.markdown(
            f"<div style='background:#1a1a2e;border-radius:12px;padding:20px 24px;margin-bottom:12px'>"
            f"<div style='font-size:14px;color:#aaa;margin-bottom:4px'>{phase_label}</div>"
            f"<div style='font-size:56px;font-weight:800;color:#FFD700;line-height:1'>{pct:.0f}%</div>"
            f"<div style='font-size:13px;color:#ccc;margin-top:8px'>{msg}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.progress(pct / 100)

        if phase not in ("done", "quick"):
            phases      = ["data", "features", "training", "saving", "done"]
            phase_names = ["Download (0–55%)", "Features (55–65%)",
                           "Train (65–98%)", "Save (98–99%)", "Complete"]
            phase_done  = ["✅" if phases.index(p) < phases.index(phase) else
                           ("🔄" if p == phase else "⬜")
                           for p in phases] if phase in phases else ["⬜"] * 5
            pcols = st.columns(5)
            for col, icon, name in zip(pcols, phase_done, phase_names):
                col.markdown(f"{icon} {name}")

        if updated:
            st.caption(f"Updated: {updated[11:19]} UTC")

        st.divider()

    elif _running:
        st.info("⏳ Scheduler running — loading progress…")
        st.divider()


_scheduler_progress_fragment()

# ── Auto-refresh every 30 s — keeps signal cards and confluence banner current
# st_autorefresh uses a JS counter so it does NOT create the infinite-loop
# that st.rerun() inside a fragment causes.
st_autorefresh(interval=30_000, key="global_refresh")

# ─────────────────────────────────────────────
# SECTION TABS
# ─────────────────────────────────────────────
_tab_signals, _tab_live, _tab_analysis, _tab_tools = st.tabs([
    "⚡  Trade Signals", "📊  Live", "📈  Signals & Analysis", "🛠  Tools"
])
_tab_live.__enter__()

# ── Compact top-of-page alert strip ──────────────────────────────────────────
# Shows immediately when bull/bear signals converge strongly — before anything else.
try:
    import json as _tp_j
    _tp_mhp    = _tp_j.loads(Path("data_cache/multi_horizon_predictions.json").read_text())
    _tp_probs  = [_tp_mhp.get(str(h), {}).get("raw_proba", 0.5) for h in [1, 2, 5]]
    _tp_up     = sum(1 for p in _tp_probs if p > 0.55)
    _tp_down   = sum(1 for p in _tp_probs if p < 0.45)
    _tp_bull   = (2 if _day_label == "STRONG BUY"  else (1 if _day_label == "BUY"  else 0))
    _tp_bull  += (1 if _1h_label  in ("BUY", "STRONG BUY")  else 0)
    _tp_bull  += (2 if _tp_up == 3 else (1 if _tp_up >= 2 else 0))
    _tp_bear   = (2 if _day_label == "STRONG SELL" else (1 if _day_label == "SELL" else 0))
    _tp_bear  += (1 if _1h_label  in ("SELL", "STRONG SELL") else 0)
    _tp_bear  += (2 if _tp_down == 3 else (1 if _tp_down >= 2 else 0))
    _tp_dir    = "BULL" if _tp_bull > _tp_bear else ("BEAR" if _tp_bear > _tp_bull else None)
    _tp_sc     = _tp_bull if _tp_dir == "BULL" else (_tp_bear if _tp_dir else 0)
    _tp_tier   = ("ALERT" if _tp_sc >= 5 else ("BUILDING" if _tp_sc >= 3 else None))
    if _tp_tier and _tp_dir:
        _tp_col  = "#4caf50" if _tp_dir == "BULL" else "#ef5350"
        _tp_bg   = "#001a08" if _tp_dir == "BULL" else "#1a0204"
        _tp_icon     = "🚀" if _tp_dir == "BULL" else "🔴"
        _tp_move     = "GOING UP ▲" if _tp_dir == "BULL" else "GOING DOWN ▼"
        _tp_run_lbl  = "Bull Run" if _tp_dir == "BULL" else "Bear Run"
        _tp_lbl      = f"{_tp_icon}  GOLD {_tp_move} — {_tp_run_lbl} {_tp_tier}  ·  {_tp_sc} factors aligned"
        st.markdown(
            f'<div style="background:{_tp_bg};border:1px solid {_tp_col}66;border-radius:8px;'
            f'padding:8px 16px;margin-bottom:10px;display:flex;align-items:center;gap:10px;">'
            f'<span style="font-size:13px;font-weight:800;color:{_tp_col};">{_tp_lbl}</span>'
            f'<span style="font-size:11px;color:{_tp_col}66;margin-left:auto;">'
            f'↓ Full breakdown below charts</span></div>',
            unsafe_allow_html=True,
        )
except Exception:
    pass

# ═══════════════════════════════════════════════════════════════════════════
# THE ONLY THING THAT MATTERS — TWO SIGNALS, BUY OR SELL
# Left:  1-Hour Trade   Right: Daily Trade
# ═══════════════════════════════════════════════════════════════════════════

# ── IG CFD constants (used in both the Live signal cards and IG Trade tab) ───
_IG_POINT_AUD_C  = 10.0    # AUD profit per $1/oz move per 1 contract
_IG_SPREAD_C     = 0.5     # typical bid-ask spread in $/oz
_IG_MARGIN_PCT_C = 0.05    # 5 % margin requirement
_IG_MIN_STOP_C   = 1.0     # minimum stop distance in $/oz

# ── Helper: get last-refresh age ─────────────────────────────────────────
try:
    _sched_data = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    _lq_ts      = _sched_data.get("last_quick")
    _sig_age    = f"{int((time.time()-_lq_ts)/60)} min ago" if _lq_ts else "—"
except Exception:
    _sig_age = "—"

# ── Helper: render one BUY/SELL card ────────────────────────────────────
def _signal_card(label: str, sig: dict | None, timeframe: str, current_price: float) -> str:
    """
    Render a large BUY / SELL signal card as HTML.
    label      : "BUY", "SELL", "STRONG BUY", etc.
    sig        : raw signal dict from get_day_trading_signal()
    timeframe  : display string, e.g. "⚡ 1-Hour Trade"
    """
    BULL = {"BUY", "STRONG BUY"}
    BEAR = {"SELL", "STRONG SELL"}

    is_buy       = label in BULL
    is_sell      = label in BEAR
    is_lean_buy  = label == "LEAN BUY"
    is_lean_sell = label == "LEAN SELL"
    is_hold      = not (is_buy or is_sell)

    if is_buy:
        col, bg, border = "#00e676", "#020d06", "#00e676"
        arrow, verb     = "▲", "BUY"
        action_word     = "STRONG BUY" if "STRONG" in label else "BUY"
    elif is_sell:
        col, bg, border = "#ff3d57", "#0d0203", "#ff3d57"
        arrow, verb     = "▼", "SELL"
        action_word     = "STRONG SELL" if "STRONG" in label else "SELL"
    elif label == "LEAN BUY":
        col, bg, border = "#37474f", "#060a0d", "#455a64"
        arrow, verb     = "~", "LEAN BUY"
        action_word     = "LEAN BUY"
    elif label == "LEAN SELL":
        col, bg, border = "#37474f", "#0d0a06", "#455a64"
        arrow, verb     = "~", "LEAN SELL"
        action_word     = "LEAN SELL"
    else:
        col, bg, border = "#455a64", "#0a0c10", "#2a3040"
        arrow, verb     = "—", "WAIT"
        action_word     = "HOLD / WAIT"

    conf     = float(sig.get("confidence", 0.5)) if sig else 0.5
    conf_pct = f"{conf:.0%}"
    bar_w    = max(4, int(conf * 100))

    # Entry / TP / SL — use live spot price as entry; scale distances from signal
    entry = current_price
    _is_long = is_buy or is_lean_buy    # treat lean as potential long for TP/SL display
    _is_short = is_sell or is_lean_sell
    if sig:
        atr = float(sig.get("atr", entry * 0.008))
        sig_entry = float(sig.get("entry", entry))
        sig_tp    = float(sig.get("target",   sig_entry + atr * 2.5))
        sig_sl    = float(sig.get("stop_loss", sig_entry - atr * 1.5))
        tp_dist   = abs(sig_tp - sig_entry)
        sl_dist   = abs(sig_entry - sig_sl)
        tp = entry + tp_dist if _is_long else entry - tp_dist
        sl = entry - sl_dist if _is_long else entry + sl_dist
    else:
        atr     = entry * 0.008
        tp_dist = atr * 2.5
        sl_dist = atr * 1.5
        tp = entry + tp_dist if _is_long else entry - tp_dist
        sl = entry - sl_dist if _is_long else entry + sl_dist

    tp_dist = abs(tp - entry)
    sl_dist = abs(entry - sl)
    rr      = tp_dist / sl_dist if sl_dist > 0 else 0

    # ── "What to do" recommendation line ──────────────────────────────────────
    _is_weekly = "Weekly" in timeframe or "1wk" in timeframe.lower()
    if is_buy and "STRONG" in label:
        if _is_weekly:
            _rec = f"→ Weekly uptrend confirmed · favour LONG on all lower timeframes"
        else:
            _rec = f"→ Enter LONG now · SL {sl:,.0f} · TP {tp:,.0f}"
        _rec_col = "#00e676"
    elif is_buy:
        if _is_weekly:
            _rec = "→ Weekly bias is LONG · wait for 4H/Daily to confirm entry"
        else:
            _rec = "→ Consider LONG · confirm Daily + 4H agree"
        _rec_col = "#4caf50"
    elif is_lean_buy:
        _rec = "→ Wait · need RSI > 50 + momentum confirmation"
        _rec_col = "#78909c"
    elif is_lean_sell:
        _rec = "→ Wait · need RSI < 50 + breakdown confirmation"
        _rec_col = "#78909c"
    elif is_sell and "STRONG" in label:
        if _is_weekly:
            _rec = f"→ Weekly downtrend confirmed · favour SHORT on all lower timeframes"
        else:
            _rec = f"→ Enter SHORT now · SL {sl:,.0f} · TP {tp:,.0f}"
        _rec_col = "#ff3d57"
    elif is_sell:
        if _is_weekly:
            _rec = "→ Weekly bias is SHORT · wait for 4H/Daily to confirm entry"
        else:
            _rec = "→ Consider SHORT · confirm Daily + 4H agree"
        _rec_col = "#ef5350"
    else:
        _rec = "→ No trade · signals mixed or flat"
        _rec_col = "#546e7a"

    # ── Market regime detection ───────────────────────────────────────────────
    _regime_label = "⟷ Ranging"
    _regime_col   = "#546e7a"
    _regime_bg    = "#12151a"
    if sig:
        _ema9_r  = sig.get("ema9",  entry)
        _ema21_r = sig.get("ema21", entry)
        _ema50_r = sig.get("ema50", entry)
        _bb_r    = sig.get("bb_pct", 50)
        if entry > _ema9_r and _ema9_r > _ema21_r and _ema21_r > _ema50_r:
            _regime_label = "📈 Uptrend"
            _regime_col   = "#00e676"
            _regime_bg    = "#001a0a"
        elif entry < _ema9_r and _ema9_r < _ema21_r and _ema21_r < _ema50_r:
            _regime_label = "📉 Downtrend"
            _regime_col   = "#ff3d57"
            _regime_bg    = "#1a0005"
        elif _bb_r > 90 or _bb_r < 10:
            _regime_label = "💥 Breakout"
            _regime_col   = "#f5a623"
            _regime_bg    = "#1a0f00"
    _regime_html = (
        f'<span style="background:{_regime_bg};color:{_regime_col};'
        f'border:1px solid {_regime_col}55;border-radius:4px;'
        f'padding:2px 8px;font-size:9px;font-weight:800;letter-spacing:0.5px;">'
        f'{_regime_label}</span>'
    )

    # ── RSI visual gauge ──────────────────────────────────────────────────────
    _rsi_gauge_html = ""
    _rsi_val2 = sig.get("rsi") if sig else None
    if _rsi_val2 is not None:
        _rsi_p = max(0, min(100, int(_rsi_val2)))
        _rsi_c2 = "#ef5350" if _rsi_p >= 70 else ("#4caf50" if _rsi_p <= 30 else "#78909c")
        _rsi_gauge_html = (
            f'<div style="margin:6px 0 10px;">'
            f'<div style="display:flex;justify-content:space-between;font-size:7px;color:#37474f;margin-bottom:2px;">'
            f'<span>0</span><span>30</span><span>70</span><span>100</span></div>'
            f'<div style="height:6px;background:#0d1117;border-radius:3px;overflow:visible;position:relative;">'
            f'<div style="position:absolute;left:0;width:30%;height:100%;background:#4caf5015;border-right:1px solid #4caf5033;"></div>'
            f'<div style="position:absolute;left:70%;width:30%;height:100%;background:#ef535015;border-left:1px solid #ef535033;"></div>'
            f'<div style="position:absolute;top:-2px;left:{_rsi_p}%;transform:translateX(-50%);'
            f'width:4px;height:10px;background:{_rsi_c2};border-radius:2px;'
            f'box-shadow:0 0 6px {_rsi_c2}88;"></div>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;font-size:8px;margin-top:4px;">'
            f'<span style="color:#4caf5099;">Oversold</span>'
            f'<span style="color:{_rsi_c2};font-weight:800;">RSI {_rsi_val2:.0f}</span>'
            f'<span style="color:#ef535099;">Overbought</span></div>'
            f'</div>'
        )

    # ── MACD / BB / VWAP mini-pill row ────────────────────────────────────────
    _tech_row_html = ""
    if sig:
        _macd_v  = sig.get("macd",     0)
        _macd_sv = sig.get("macd_sig", 0)
        _vwap_v  = sig.get("vwap",     entry)
        _bb_v    = sig.get("bb_pct",   50)
        _macd_c  = "#4caf50" if _macd_v > _macd_sv else "#ef5350"
        _macd_l  = "▲ Bull"  if _macd_v > _macd_sv else "▼ Bear"
        _vwap_c  = "#4caf50" if entry >= _vwap_v    else "#ef5350"
        _vwap_l  = "▲ Above" if entry >= _vwap_v    else "▼ Below"
        _bb_c    = ("#ef5350" if _bb_v > 70 else ("#4caf50" if _bb_v < 30 else "#78909c"))
        _bb_l    = ("Upper" if _bb_v > 70 else ("Lower" if _bb_v < 30 else "Middle"))
        def _pill(lbl, val, vc):
            return (
                f'<div style="background:#0d1117;border:1px solid {vc}44;border-radius:5px;'
                f'padding:3px 8px;font-size:9px;white-space:nowrap;">'
                f'<span style="color:#546e7a;">{lbl}</span> '
                f'<span style="color:{vc};font-weight:700;">{val}</span></div>'
            )
        _tech_row_html = (
            f'<div style="display:flex;gap:5px;flex-wrap:wrap;margin:0 0 8px;">'
            + _pill("MACD", _macd_l, _macd_c)
            + _pill("VWAP", _vwap_l, _vwap_c)
            + _pill("BB", f"{_bb_v:.0f}% {_bb_l}", _bb_c)
            + f'</div>'
        )

    # ── Indicator agreement count ─────────────────────────────────────────────
    _agree_n = _agree_tot = 0
    if sig:
        for _av in list((sig.get("indicators") or {}).values()):
            _agree_tot += 1
            if (_is_long and _av > 0) or (_is_short and _av < 0):
                _agree_n += 1
        for _av in list((sig.get("smc_breakdown") or {}).values()):
            _agree_tot += 1
            if (_is_long and _av > 0) or (_is_short and _av < 0):
                _agree_n += 1
    _agree_pct = int(_agree_n / max(_agree_tot, 1) * 100)
    _agree_col = "#4caf50" if _agree_pct >= 70 else ("#ffa726" if _agree_pct >= 50 else "#ef5350")
    _agree_html = (
        f'<div style="font-size:9px;color:#6a7a94;margin:2px 0 8px;">'
        f'<span style="color:{_agree_col};font-weight:800;">{_agree_n}/{_agree_tot}</span>'
        f' indicators align &nbsp;·&nbsp; '
        f'<span style="color:{_agree_col};font-weight:800;">{_agree_pct}%</span> confluence'
        f'</div>'
    ) if _agree_tot > 0 else ""

    # ── Candle countdown ──────────────────────────────────────────────────────
    import datetime as _cdt
    _now_u = _cdt.datetime.utcnow()
    _tf_min_map = {
        "15-Min": 15, "30-Min": 30, "1-Hour": 60,
        "4-Hour": 240, "Daily": 1440, "Weekly": 10080,
    }
    _cd_mins = None
    for _cd_k, _cd_v in _tf_min_map.items():
        if _cd_k in timeframe:
            _elapsed = (_now_u.hour * 60 + _now_u.minute) % _cd_v
            _cd_mins = _cd_v - _elapsed
            break
    _countdown_html = ""
    if _cd_mins is not None:
        _cd_c = "#ef5350" if _cd_mins <= 3 else ("#ffa726" if _cd_mins <= 10 else "#546e7a")
        _countdown_html = (
            f'<span style="color:{_cd_c};font-size:9px;">'
            f'⏱ {_cd_mins}m to candle close</span>'
        )

    # ── Indicators reorganized by category ────────────────────────────────────
    _TREND_KEYS = {"EMA Trend", "EMA200", "HTF EMA200", "SMA20", "VWAP"}
    _MOM_KEYS   = {"RSI", "MACD", "Bollinger", "Momentum", "ROC"}
    ind_rows = ""
    if sig:
        _all_inds    = sig.get("indicators", {})
        _trend_inds  = {k: v for k, v in _all_inds.items() if k in _TREND_KEYS}
        _mom_inds    = {k: v for k, v in _all_inds.items() if k in _MOM_KEYS}
        _other_inds  = {k: v for k, v in _all_inds.items()
                        if k not in _TREND_KEYS and k not in _MOM_KEYS}
        def _ind_row(name, score):
            ic = "#4caf50" if score > 0 else ("#ef5350" if score < 0 else "#455a64")
            ia = "▲▲" if score >= 2 else ("▲" if score > 0 else
                 ("▼▼" if score <= -2 else ("▼" if score < 0 else "—")))
            return (
                f'<div style="display:flex;justify-content:space-between;padding:2px 0;">'
                f'<span style="font-size:10px;color:#8a9ab5;">{name}</span>'
                f'<span style="font-size:10px;color:{ic};font-weight:700;">{ia}</span>'
                f'</div>'
            )
        def _cat_hdr(lbl):
            return (
                f'<div style="font-size:8px;color:#455a64;text-transform:uppercase;'
                f'letter-spacing:1px;margin:6px 0 2px;font-weight:700;">{lbl}</div>'
            )
        if _trend_inds:
            ind_rows += _cat_hdr("📊 Trend")
            for k, v in _trend_inds.items():
                ind_rows += _ind_row(k, v)
        if _mom_inds:
            ind_rows += _cat_hdr("📈 Momentum")
            for k, v in _mom_inds.items():
                ind_rows += _ind_row(k, v)
        if _other_inds:
            ind_rows += _cat_hdr("◈ Other")
            for k, v in _other_inds.items():
                ind_rows += _ind_row(k, v)
        _smc_sub = sig.get("smc_breakdown", {})
        if _smc_sub:
            ind_rows += _cat_hdr("📐 Smart Money")
            for sk, sv in _smc_sub.items():
                ind_rows += _ind_row(sk.strip(), sv)

    tp_col = "#4caf50" if _is_long  else "#ef5350"
    sl_col = "#ef5350" if _is_long  else "#4caf50"

    # ── IG Order Ticket values ─────────────────────────────────────────────
    # IG shows BUY at the ask price (spot + half spread)
    # and SELL at the bid price (spot − half spread)
    half_spread   = _IG_SPREAD_C / 2
    ig_deal_price = entry + half_spread if _is_long else entry - half_spread
    ig_limit      = round(tp, 2)   # type directly into IG Limit field
    ig_stop       = round(sl, 2)   # type directly into IG Stop field

    pnl_limit  = tp_dist * _IG_POINT_AUD_C          # AUD profit if limit hit (1 contract)
    pnl_stop   = sl_dist * _IG_POINT_AUD_C           # AUD loss  if stop  hit (1 contract)
    margin_aud = entry   * _IG_POINT_AUD_C * _IG_MARGIN_PCT_C  # AUD margin per contract
    rr_col     = "#4caf50" if rr >= 1.5 else ("#ffa726" if rr >= 1.0 else "#ef5350")

    def _row(lbl, val, vc="#e0e0e0", small=""):
        sm = f'<span style="font-size:9px;color:#8a9ab5;margin-left:3px;">{small}</span>' if small else ""
        return (f'<div style="display:flex;justify-content:space-between;'
                f'align-items:baseline;padding:3px 0;border-bottom:1px solid #ffffff07;">'
                f'<span style="font-size:10px;color:#8a9ab5;">{lbl}</span>'
                f'<span style="font-size:11px;font-weight:800;color:{vc};">{val}{sm}</span>'
                f'</div>')

    _dir_label = (
        "▲ BUY"       if is_buy       else
        "▼ SELL"      if is_sell      else
        "~ LEAN LONG" if is_lean_buy  else
        "~ LEAN SHORT" if is_lean_sell else
        "— NO POSITION"
    )
    ticket_rows = "".join([
        _row("Direction", _dir_label, col),
        _row("Deal price",       f"${ig_deal_price:,.2f}",            "#f5c518",
             "(incl. IG spread)"),
        _row("Limit (take profit)", f"${ig_limit:,.2f}",             tp_col,
             f"+${tp_dist:.0f}"),
        _row("Stop (stop loss)", f"${ig_stop:,.2f}",                 sl_col,
             f"−${sl_dist:.0f}"),
        _row("Size",             "1 contract",                        "#aaa",
             "adjust to your risk"),
        _row("Risk : Reward",    f"1 : {rr:.1f}",                    rr_col),
        _row("If limit hit",     f"+A${pnl_limit:,.0f}",             "#4caf50",
             "per contract"),
        _row("If stop hit",      f"−A${pnl_stop:,.0f}",             "#ef5350",
             "per contract"),
        _row("Margin required",  f"A${margin_aud:,.0f}",             "#888",
             "per contract"),
    ])

    # ── Session context ───────────────────────────────────────────────────────
    import datetime as _dt2
    _utc_h = _dt2.datetime.utcnow().hour
    if   13 <= _utc_h < 17:
        _sess_label = "London/NY Overlap"; _sess_col = "#4caf50"; _sess_dot = "🟢"
        _sess_quality = "peak"
    elif  8 <= _utc_h < 13:
        _sess_label = "London Open";       _sess_col = "#8bc34a"; _sess_dot = "🟡"
        _sess_quality = "good"
    elif 17 <= _utc_h < 21:
        _sess_label = "NY Afternoon";      _sess_col = "#ffa726"; _sess_dot = "🟡"
        _sess_quality = "moderate"
    elif 21 <= _utc_h < 23:
        _sess_label = "NY Close / Sydney"; _sess_col = "#ff7043"; _sess_dot = "🔴"
        _sess_quality = "low"
    else:
        _sess_label = "Asian Session";     _sess_col = "#ef5350"; _sess_dot = "🔴"
        _sess_quality = "low"

    _session_html = (
        f'<div style="font-size:9px;color:{_sess_col};font-weight:700;'
        f'letter-spacing:0.5px;margin-bottom:6px;">'
        f'{_sess_dot} {_sess_label}'
        + (" — low liquidity, wider spreads"
           if _sess_quality == "low" else
           " — active market" if _sess_quality in ("peak", "good") else "")
        + '</div>'
    )

    # ── RSI + ATR info row ────────────────────────────────────────────────────
    _rsi_val = sig.get("rsi") if sig else None
    _atr_val = sig.get("atr") if sig else None
    if _rsi_val is not None:
        if   _rsi_val >= 70: _rsi_col = "#ef5350"   # overbought
        elif _rsi_val <= 30: _rsi_col = "#4caf50"   # oversold
        else:                _rsi_col = "#8a9ab5"
    _stats_html = ""
    if _rsi_val is not None or _atr_val is not None:
        _parts = []
        if _rsi_val is not None:
            _parts.append(
                f'<span>RSI&nbsp;<b style="color:{_rsi_col}">{_rsi_val:.0f}</b></span>'
            )
        if _atr_val is not None:
            _parts.append(
                f'<span>ATR&nbsp;<b style="color:#8a9ab5">${_atr_val:.1f}</b></span>'
            )
        _stats_html = (
            f'<div style="display:flex;gap:12px;font-size:10px;color:#8a9ab5;'
            f'margin:4px 0 8px;">' + " · ".join(_parts) + '</div>'
        )

    # ── News sentiment badge ──────────────────────────────────────────────────
    _news_html = ""
    try:
        from pathlib import Path as _P2
        import json as _j2
        _nf = _P2("data_cache/news_sentiment.json")
        if _nf.exists():
            _nd = _j2.loads(_nf.read_text())
            _ns = _nd.get("score", 0.0)
            _nb = _nd.get("bullish_n", 0)
            _nbr = _nd.get("bearish_n", 0)
            _nt = _nd.get("total_n", 0)
            if _nt > 0:
                if   _ns >  0.25: _nc, _ni, _nl = "#4caf50", "📰 Bullish", f"+{_ns:.0%}"
                elif _ns < -0.25: _nc, _ni, _nl = "#ef5350", "📰 Bearish", f"{_ns:.0%}"
                else:             _nc, _ni, _nl = "#8a9ab5", "📰 Neutral",  "neutral"
                _news_html = (
                    f'<div style="display:flex;align-items:center;gap:6px;'
                    f'background:#0d1117;border:1px solid {_nc}33;border-radius:6px;'
                    f'padding:4px 8px;margin:4px 0;font-size:10px;">'
                    f'<span style="color:{_nc};font-weight:700;">{_ni}</span>'
                    f'<span style="color:#6a7a94;">{_nb}↑ {_nbr}↓ of {_nt} headlines</span>'
                    f'<span style="color:{_nc};font-weight:700;margin-left:auto;">{_nl}</span>'
                    f'</div>'
                )
    except Exception:
        pass

    # ── Entry-quality warning badges ─────────────────────────────────────────
    # These only matter when the signal says BUY/SELL — they warn the trader
    # that even though the macro direction is correct, the current moment may
    # be a poor entry: price is actively falling (pullback), has already fallen
    # a full stop distance from its recent peak (stop zone), or the 15-minute
    # chart's momentum opposes the 1H direction (15m conflict).
    def _badge(text, bg_col, border_col, text_col):
        return (
            f'<div style="background:{bg_col};border:1px solid {border_col};'
            f'border-radius:6px;padding:4px 8px;margin:3px 0;font-size:10px;'
            f'color:{text_col};font-weight:600;line-height:1.4;">{text}</div>'
        )

    warning_html = ""
    if sig and (is_buy or is_sell):
        _pw  = sig.get("pullback_warning",  False)
        _ps  = sig.get("pullback_severity", "none")
        _sz  = sig.get("stop_zone_warning", False)
        _dha = sig.get("drop_from_high_atrs", 0.0)
        _m15 = sig.get("m15_confirmation",  "unknown")

        if is_buy:
            if _ps == "strong":
                warning_html += _badge(
                    "⚠️ Strong pullback — price fell sharply (last 5 bars). Wait for candle to close up.",
                    "#2a0e00", "#cc440066", "#ff7744")
            elif _ps == "mild":
                warning_html += _badge(
                    "⚠️ Mild pullback in progress — let price stabilise before entering.",
                    "#1e1000", "#cc880044", "#ffaa44")
            if _sz:
                warning_html += _badge(
                    f"🛑 Stop zone — price is {_dha:.1f}× ATR below recent high. A stop may already be triggered.",
                    "#1a0000", "#ff000055", "#ff5555")
            if _m15 == "conflicts":
                warning_html += _badge(
                    "⚡ 15-min EMA & momentum point DOWN — 1H BUY conflicts with short-term trend. Wait for 15m to turn up.",
                    "#1a001a", "#cc00cc55", "#cc88ff")
            elif _m15 == "mixed":
                warning_html += _badge(
                    "〰 15-min signals mixed — EMA or momentum diverges. Entry timing uncertain.",
                    "#12121e", "#8888cc44", "#8899cc")
            elif _m15 == "confirms":
                warning_html += _badge(
                    "✅ 15-min EMA & momentum confirm BUY direction — good entry timing.",
                    "#001a00", "#00cc4455", "#44cc88")
        elif is_sell:
            if _ps == "strong":
                warning_html += _badge(
                    "⚠️ Sharp move down already underway — chasing a SELL here increases slippage risk.",
                    "#2a0e00", "#cc440066", "#ff7744")
            if _m15 == "conflicts":
                warning_html += _badge(
                    "⚡ 15-min EMA & momentum point UP — 1H SELL conflicts with short-term bounce.",
                    "#1a001a", "#cc00cc55", "#cc88ff")

    return f"""
    <div style="background:{bg};border:2px solid {border}88;border-radius:16px;
        padding:16px 16px;flex:1;min-width:0;
        box-shadow:0 0 40px {border}18;position:relative;overflow:hidden;">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;flex-wrap:wrap;">
        <span style="font-size:10px;color:{col}88;letter-spacing:1px;text-transform:uppercase;
            font-weight:700;">{timeframe} · IG CFD Gold</span>
        {_regime_html}
      </div>
      {_session_html}
      <div style="font-size:20px;font-weight:900;color:{col};letter-spacing:0.5px;
          font-family:monospace;line-height:1.15;margin-bottom:2px;white-space:nowrap;">
          {arrow}&nbsp;{action_word}
      </div>
      <div style="font-size:10px;color:{_rec_col};font-weight:600;margin-bottom:6px;
          letter-spacing:0.2px;">{_rec}</div>
      {_rsi_gauge_html}{_tech_row_html}{warning_html}<div style="margin:8px 0 4px;">
        <div style="display:flex;justify-content:space-between;
            font-size:10px;color:#8a9ab5;margin-bottom:4px;">
          <span>Confidence</span>
          <span style="color:{col};font-weight:700;">{conf_pct}</span>
        </div>
        <div style="height:8px;background:#0d1117;border-radius:4px;overflow:hidden;">
          <div style="width:{bar_w}%;height:8px;background:{col};border-radius:4px;"></div>
        </div>
      </div>
      {_agree_html}
      <div style="background:#080a0f;border:1px solid {border}33;border-radius:10px;
          padding:8px 10px;margin-bottom:10px;">
        <div style="font-size:9px;color:#6a7a94;text-transform:uppercase;letter-spacing:1px;
            margin-bottom:6px;font-weight:700;">📋 IG Deal Ticket</div>
        {ticket_rows}
      </div>{_news_html}{"" if not ind_rows else
        f'<div style="border-top:1px solid {border}22;padding-top:10px;">'
        f'<div style="font-size:9px;color:#6a7a94;text-transform:uppercase;'
        f'letter-spacing:1px;margin-bottom:4px;">Signal breakdown</div>'
        + ind_rows + "</div>"}
      <div style="margin-top:12px;font-size:9px;color:#6a7a94;display:flex;
          justify-content:space-between;align-items:center;flex-wrap:wrap;gap:4px;">
        <span>🤖 auto-refreshes · last {_sig_age} &nbsp; {_countdown_html}</span>
        {"<span style='background:#1a3a1a;color:#4caf50;border:1px solid #4caf5044;" +
         "border-radius:4px;padding:1px 6px;font-size:8px;font-weight:700;letter-spacing:0.5px'>" +
         "✦ TWELVE DATA · XAU/USD SPOT</span>"
         if sig and sig.get("data_source") == "twelvedata" else
         "<span style='background:#2a2a1a;color:#f5a623;border:1px solid #f5a62344;" +
         "border-radius:4px;padding:1px 6px;font-size:8px;font-weight:700;letter-spacing:0.5px'>" +
         "● yFinance · GC=F FUTURES</span>"}
      </div>
    </div>"""

# ── Current spot price for entry level ───────────────────────────────────
_card_price = _top_live_price if _top_live_price else 3000.0

# ── Build card HTML (rendered in Trade Signals tab) ──────────────────────
_card_15m = _signal_card(_15m_label, _15m_sig, "⚡ 15-Min Scalp",      _card_price)
_card_30m = _signal_card(_30m_label, _30m_sig, "⏱ 30-Min Trade",      _card_price)
_card_1h  = _signal_card(_1h_label,  _top_sig, "🕐 1-Hour Trade",      _card_price)
_card_4h  = _signal_card(_4h_label,  _4h_sig,  "📊 4-Hour Swing",      _card_price)
_card_day = _signal_card(_day_label, _day_sig,  "📅 Daily Trade",       _card_price)
_card_wk  = _signal_card(_wk_label,  _wk_sig,  "📆 Weekly Position",   _card_price)

# ── ML vs Technical conflict banner ──────────────────────────────────────
# When the ML ensemble forecast disagrees with either technical signal, show
# an explicit warning so the user knows the two systems are not aligned.
try:
    import json as _cj, datetime as _cdt
    _mhf = Path("data_cache/multi_horizon_predictions.json")
    if _mhf.exists():
        _mhp   = _cj.loads(_mhf.read_text())
        _ml_probas = [_mhp.get(str(h), {}).get("raw_proba", 0.5) for h in [1, 2, 5]]
        _avg_prob  = sum(_ml_probas) / max(1, len(_ml_probas))
        _nd = sum(1 for p in _ml_probas if p < 0.45)
        _nu = sum(1 for p in _ml_probas if p > 0.55)

        # ML consensus direction — use raw_proba for sensitivity
        _ml_consensus = None
        if _nd >= 2 and _avg_prob < 0.45:   _ml_consensus = "DOWN"
        elif _nu >= 2 and _avg_prob > 0.55: _ml_consensus = "UP"

        # Technical consensus (1H + daily)
        _bull_labels = {"BUY", "STRONG BUY"}
        _bear_labels = {"SELL", "STRONG SELL"}
        _n_bull_tech = sum(1 for lbl in [_1h_label, _day_label] if lbl in _bull_labels)
        _n_bear_tech = sum(1 for lbl in [_1h_label, _day_label] if lbl in _bear_labels)
        _tech_cons   = "UP" if _n_bull_tech >= 1 else ("DOWN" if _n_bear_tech >= 1 else None)

        # Horizons for display — show raw probability as a direction
        _hmap = {1: "1-day", 2: "2-day", 5: "5-day"}
        _hr_labels = []
        for _h, _p in zip([1,2,5], _ml_probas):
            _dstr = f"▲ UP ({_p:.0%})" if _p > 0.55 else (f"▼ DOWN ({1-_p:.0%})" if _p < 0.45 else f"— FLAT")
            _hr_labels.append(f"{_hmap[_h]}: {_dstr}")

        if _ml_consensus and _tech_cons and _ml_consensus != _tech_cons:
            _conf_dir_word = "DOWN ▼" if _ml_consensus == "DOWN" else "UP ▲"
            _tech_dir_word = "BULLISH ▲" if _tech_cons == "UP" else "BEARISH ▼"
            _fc_str  = "  ·  ".join(_hr_labels) if _hr_labels else "mixed"
            st.markdown(f"""
<div style="background:#1a1000;border:1px solid #f9a82588;border-radius:12px;
    padding:14px 18px;margin-bottom:20px;display:flex;align-items:flex-start;gap:12px;">
  <div style="font-size:22px;padding-top:2px;">⚠️</div>
  <div>
    <div style="font-size:13px;font-weight:800;color:#f9a825;margin-bottom:4px;">
      CONFLICTING SIGNALS — Use Caution
    </div>
    <div style="font-size:12px;color:#b0a070;line-height:1.6;">
      The <b style="color:#f0e68c;">ML ensemble forecast</b> predicts&nbsp;
      <b style="color:#ef5350;">{_conf_dir_word}</b>&nbsp;
      while the <b style="color:#f0e68c;">technical indicators</b> are&nbsp;
      <b style="color:#00e676;">{_tech_dir_word}</b>.
      This divergence reduces signal reliability — consider waiting for alignment
      before entering a position.
    </div>
    <div style="font-size:11px;color:#9ba8bc;margin-top:6px;">
      ML forecast: {_fc_str}
    </div>
  </div>
</div>""", unsafe_allow_html=True)
        elif _ml_consensus and _tech_cons and _ml_consensus == _tech_cons:
            _dir_word = "BULLISH ▲" if _ml_consensus == "UP" else "BEARISH ▼"
            st.markdown(f"""
<div style="background:#001a08;border:1px solid #00e67644;border-radius:12px;
    padding:10px 16px;margin-bottom:20px;display:flex;align-items:center;gap:10px;">
  <div style="font-size:16px;">✅</div>
  <div style="font-size:12px;color:#4caf50;font-weight:700;">
    ML forecast &amp; technical signals aligned — both {_dir_word}
  </div>
</div>""", unsafe_allow_html=True)
except Exception:
    pass

# ─────────────────────────────────────────────
# Top metrics row
# ─────────────────────────────────────────────
col_acc1, col_acc2, col_acc3, col_acc4 = st.columns(4)

if live_acc is not None:
    acc_icon = "🟢" if live_acc >= 0.55 else ("🟡" if live_acc >= 0.50 else "🔴")
    col_acc1.metric("Live Prediction Accuracy",
                    f"{acc_icon}  {live_acc:.1%}",
                    delta=f"{live_n} resolved predictions")
else:
    col_acc1.metric("Live Prediction Accuracy", "—",
                    delta="No resolved predictions yet")

col_acc2.metric("Pending Predictions", str(len(pending)),
                delta="awaiting target date")
col_acc3.metric("Total Live Predictions", str(len(live_preds)))

# Last backtest time
if st.session_state.last_run:
    mins_ago = int((time.time() - st.session_state.last_run) / 60)
    src = " (scheduler)" if st.session_state.source == "scheduler" else " (manual)"
    if mins_ago < 60:
        col_acc4.metric("Last Backtest", f"{mins_ago} min ago", delta=src)
    else:
        hrs = mins_ago // 60
        col_acc4.metric("Last Backtest", f"{hrs}h {mins_ago % 60}m ago", delta=src)
else:
    col_acc4.metric("Last Backtest", "Not run yet")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# BULL / BEAR RUN ALERT SYSTEM
# Scores 10 independent factors across ML, technicals, and macro.
# Fires a tiered alert (WATCH → BUILDING → ALERT) when signals converge.
# ═══════════════════════════════════════════════════════════════════════════
try:
    _bull_score = 0
    _bear_score = 0
    _bull_reasons: list[str] = []
    _bear_reasons: list[str] = []

    # ── 1. Daily signal ─────────────────────────────────────────────────
    if _day_label in ("STRONG BUY",):
        _bull_score += 2; _bull_reasons.append("Daily STRONG BUY")
    elif _day_label == "BUY":
        _bull_score += 1; _bull_reasons.append("Daily BUY")
    if _day_label in ("STRONG SELL",):
        _bear_score += 2; _bear_reasons.append("Daily STRONG SELL")
    elif _day_label == "SELL":
        _bear_score += 1; _bear_reasons.append("Daily SELL")

    # ── 2. 1H signal ─────────────────────────────────────────────────────
    if _1h_label in ("BUY", "STRONG BUY"):
        _bull_score += 1; _bull_reasons.append("1H BUY")
    if _1h_label in ("SELL", "STRONG SELL"):
        _bear_score += 1; _bear_reasons.append("1H SELL")

    # ── 3. ML forecast (raw_proba) ───────────────────────────────────────
    try:
        import json as _ra_j
        _ra_mhp = _ra_j.loads(Path("data_cache/multi_horizon_predictions.json").read_text())
        _ra_probs = [_ra_mhp.get(str(h), {}).get("raw_proba", 0.5) for h in [1, 2, 5]]
        _n_ml_up   = sum(1 for p in _ra_probs if p > 0.55)
        _n_ml_down = sum(1 for p in _ra_probs if p < 0.45)
        _avg_ml    = sum(_ra_probs) / 3
        if _n_ml_up == 3:
            _bull_score += 2; _bull_reasons.append("ML: all 3 horizons UP")
        elif _n_ml_up >= 2:
            _bull_score += 1; _bull_reasons.append("ML: 2/3 horizons UP")
        if _n_ml_down == 3:
            _bear_score += 2; _bear_reasons.append("ML: all 3 horizons DOWN")
        elif _n_ml_down >= 2:
            _bear_score += 1; _bear_reasons.append("ML: 2/3 horizons DOWN")
    except Exception:
        pass

    # ── 4. Daily indicator signals ───────────────────────────────────────
    _ra_ind = (_day_sig or {}).get("indicators", {})

    # EMA200: price above (bullish structural support)
    if _ra_ind.get("EMA200", 0) > 0:
        _bull_score += 1; _bull_reasons.append("Price above 200EMA")
    elif _ra_ind.get("EMA200", 0) < 0:
        _bear_score += 1; _bear_reasons.append("Price below 200EMA")

    # EMA Trend (short-term EMA 9/21/50 stack)
    if _ra_ind.get("EMA Trend", 0) > 0:
        _bull_score += 1; _bull_reasons.append("EMA stack bullish")
    elif _ra_ind.get("EMA Trend", 0) < 0:
        _bear_score += 1; _bear_reasons.append("EMA stack bearish")

    # MACD momentum
    if _ra_ind.get("MACD", 0) > 0:
        _bull_score += 1; _bull_reasons.append("MACD bullish")
    elif _ra_ind.get("MACD", 0) < 0:
        _bear_score += 1; _bear_reasons.append("MACD bearish")

    # Macro regime (real yields + DXY + risk appetite)
    if _ra_ind.get("Regime", 0) > 0:
        _bull_score += 1; _bull_reasons.append("Macro regime bullish")
    elif _ra_ind.get("Regime", 0) < 0:
        _bear_score += 1; _bear_reasons.append("Macro regime bearish")

    # GVZ: Gold Volatility — elevated = fear = flight to gold (bullish)
    if _ra_ind.get("GVZ", 0) > 0:
        _bull_score += 1; _bull_reasons.append("GVZ elevated (fear → gold)")
    elif _ra_ind.get("GVZ", 0) < 0:
        _bear_score += 1; _bear_reasons.append("GVZ low (complacency)")

    # OBV: Volume confirms direction
    if _ra_ind.get("OBV", 0) > 0:
        _bull_score += 1; _bull_reasons.append("Volume supporting move")
    elif _ra_ind.get("OBV", 0) < 0:
        _bear_score += 1; _bear_reasons.append("Volume declining")

    # 52-Week Range (near highs = breakout, near lows = breakdown)
    if _ra_ind.get("52W Range", 0) > 0:
        _bull_score += 1; _bull_reasons.append("Near 52-week highs")
    elif _ra_ind.get("52W Range", 0) < 0:
        _bear_score += 1; _bear_reasons.append("Near 52-week lows")

    # ── 5. Determine alert level ─────────────────────────────────────────
    _MAX_SCORE = 12   # max possible (STRONG BUY=2 + 1H=1 + ML=2 + 7 indicators)
    _alert_dir    = "BULL" if _bull_score > _bear_score else ("BEAR" if _bear_score > _bull_score else None)
    _alert_active_score = _bull_score if _alert_dir == "BULL" else (_bear_score if _alert_dir == "BEAR" else 0)
    _alert_reasons = _bull_reasons if _alert_dir == "BULL" else _bear_reasons

    # Tiered thresholds
    if   _alert_active_score >= 8:   _alert_tier = "ALERT"
    elif _alert_active_score >= 5:   _alert_tier = "BUILDING"
    elif _alert_active_score >= 3:   _alert_tier = "WATCH"
    else:                            _alert_tier = "QUIET"

    # ── 6. Render ─────────────────────────────────────────────────────────
    if _alert_tier in ("ALERT", "BUILDING"):
        _is_bull_alert = (_alert_dir == "BULL")
        _acol   = "#00e676" if _is_bull_alert else "#ef5350"
        _abg    = "#001a08" if _is_bull_alert else "#1a0204"
        _adir      = "GOING UP ▲" if _is_bull_alert else "GOING DOWN ▼"
        _arun      = "Bull Run" if _is_bull_alert else "Bear Run"
        _aicon     = "🚀" if _is_bull_alert else "🔴"
        _alevel    = "ALERT" if _alert_tier == "ALERT" else "BUILDING"
        _abar      = int(_alert_active_score / _MAX_SCORE * 100)
        _reasons_html = "".join(
            f'<span style="background:{_acol}22;border:1px solid {_acol}44;border-radius:4px;'
            f'padding:2px 8px;font-size:11px;color:{_acol};margin:2px 3px;display:inline-block;">'
            f'✓ {r}</span>'
            for r in _alert_reasons
        )
        st.markdown(f"""
<div style="background:{_abg};border:2px solid {_acol}88;border-radius:16px;
    padding:18px 24px;margin-bottom:20px;position:relative;overflow:hidden;">
  <div style="position:absolute;top:0;left:0;right:0;height:3px;background:{_acol};
      opacity:0.6;border-radius:16px 16px 0 0;"></div>
  <div style="display:flex;align-items:center;gap:14px;margin-bottom:12px;">
    <div style="font-size:34px;line-height:1;">{"▲" if _is_bull_alert else "▼"}</div>
    <div>
      <div style="font-size:11px;color:{_acol}99;text-transform:uppercase;letter-spacing:2px;
          font-weight:700;">{_arun} {_alevel} · {_alert_active_score}/{_MAX_SCORE} factors</div>
      <div style="font-size:26px;font-weight:900;color:{_acol};letter-spacing:1px;line-height:1.2;">
        GOLD {_adir}
      </div>
    </div>
    <div style="margin-left:auto;text-align:right;">
      <div style="font-size:10px;color:{_acol}66;margin-bottom:4px;">Confidence</div>
      <div style="width:80px;height:8px;background:#1a1a1a;border-radius:4px;">
        <div style="width:{_abar}%;height:8px;background:{_acol};border-radius:4px;"></div>
      </div>
    </div>
  </div>
  <div style="margin-bottom:10px;line-height:1.8;">{_reasons_html}</div>
  <div style="font-size:11px;color:{_acol}66;">
    {'All major timeframes and macro factors are aligned. Highest-conviction setup.' if _alert_tier == 'ALERT'
     else 'Signals are building. Monitor for additional confirmation before entering.'}
  </div>
</div>""", unsafe_allow_html=True)

    elif _alert_tier == "WATCH":
        _is_bull_watch = (_alert_dir == "BULL")
        _wcol = "#4caf50" if _is_bull_watch else "#ef5350"
        _wdir = "GOING UP ▲" if _is_bull_watch else "GOING DOWN ▼"
        _wrun = "Bull" if _is_bull_watch else "Bear"
        _wreason_str = " · ".join(_alert_reasons[:3])
        st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid #3a4a60;border-radius:10px;
    padding:10px 16px;margin-bottom:16px;display:flex;align-items:center;gap:12px;">
  <div style="font-size:18px;">👁</div>
  <div>
    <div style="font-size:11px;font-weight:700;color:{_wcol};">
      WATCH — Gold {_wdir} · {_wrun} signals building ({_alert_active_score}/{_MAX_SCORE})
    </div>
    <div style="font-size:11px;color:#8a9ab5;">{_wreason_str}</div>
  </div>
</div>""", unsafe_allow_html=True)
except Exception:
    pass

# ─────────────────────────────────────────────
# Live gold price chart
# ─────────────────────────────────────────────
# LIVE CHART FRAGMENT — refreshes every 30 s without touching the rest of the page
# ─────────────────────────────────────────────
@st.fragment(run_every=10)
def _live_chart_fragment():
    """Fetch fresh intraday data, render the short-term forecast metrics and the
    unified live chart.  Refreshes every 10 seconds."""
    # ── Fetch the real-time live price (same source as the top ticker boxes) ──
    _frag_live = fetch_live_price()
    _frag_live_price = _frag_live["price"] if _frag_live else None

    _i5m = fetch_intraday_5m()
    if _i5m is not None and len(_i5m) >= 4:
        _i2h = _i5m.tail(24).copy()
        _AEST_SHIFT = pd.Timedelta(hours=10)
        if hasattr(_i2h.index, "tzinfo") and _i2h.index.tzinfo is not None:
            _i2h.index = _i2h.index.tz_convert("UTC").tz_localize(None) + _AEST_SHIFT
        elif hasattr(_i2h.index, "tz") and _i2h.index.tz is not None:
            _i2h.index = _i2h.index.tz_convert("UTC").tz_localize(None) + _AEST_SHIFT
        else:
            _i2h.index = _i2h.index + _AEST_SHIFT
        _i_last_ts    = _i2h.index[-1]
        _i_last_price = float(_i2h.iloc[-1])

        if _top_sig is not None:
            _i_action = _top_sig.get("action", "NEUTRAL")
            _i_score  = _top_sig.get("total_score", 0)
            _i_atr    = _top_sig.get("atr", _i_last_price * 0.008)
            if _i_action in ("BUY", "STRONG BUY"):
                _i_dir, _i_conf, _i_lean = 1, _top_sig.get("confidence", 0.5), False
            elif _i_action in ("SELL", "STRONG SELL"):
                _i_dir, _i_conf, _i_lean = -1, _top_sig.get("confidence", 0.5), False
            elif _i_score > 0.5:
                _i_dir, _i_conf, _i_lean = 1,  0.20 + min(_i_score / 14.0, 0.12), True
            elif _i_score < -0.5:
                _i_dir, _i_conf, _i_lean = -1, 0.20 + min(abs(_i_score) / 14.0, 0.12), True
            else:
                _i_dir, _i_conf, _i_lean = 0, 0.15, False
        else:
            _i_dir, _i_conf, _i_atr, _i_lean = 0, 0.15, _i_last_price * 0.008, False

        _i_atr_per_min = _i_atr / (390 ** 0.5)
        _now_utc  = datetime.utcnow()
        _eod_utc  = _now_utc.replace(hour=21, minute=0, second=0, microsecond=0)
        if _eod_utc <= _now_utc:
            _eod_utc = _eod_utc + pd.Timedelta(days=1)
        _mins_to_eod = max(int((_eod_utc - _now_utc).total_seconds() / 60), 5)

        _fhorizons = [
            ("10 min",      10),
            ("30 min",      30),
            ("1 hour",      60),
            ("2 hours",    120),
            ("5 hours",    300),
            ("End of Day", _mins_to_eod),
        ]
        _fi_prices = {_fl: _i_last_price + _i_dir * _i_atr_per_min * (_fm ** 0.5) * (_i_conf / 0.5)
                      for _fl, _fm in _fhorizons}
        _fi_ts     = {_fl: _i_last_ts + pd.Timedelta(minutes=_fm) for _fl, _fm in _fhorizons}

        _ipreds = load_intraday_preds()
        _ipreds = resolve_intraday_preds(_ipreds, _i5m)
        _iacc   = intraday_accuracy_by_horizon(_ipreds)
        _last_10m_ts = next(
            (p["made_at"] for p in reversed(_ipreds) if p["horizon_label"] == "10 min"), None)
        _should_save_pred = (
            _last_10m_ts is None
            or (datetime.utcnow() - datetime.fromisoformat(_last_10m_ts)).total_seconds() >= 600
        )
        if _should_save_pred and _i_dir != 0:
            _now_str = datetime.utcnow().isoformat()
            _ind_scores_snap = dict(_top_sig.get("indicators", {})) if _top_sig else {}
            _new_recs = []
            for _fl2, _fm2 in _fhorizons:
                _new_recs.append({
                    "made_at": _now_str, "horizon_label": _fl2, "horizon_min": _fm2,
                    "price_at_prediction": _i_last_price, "predicted_price": _fi_prices[_fl2],
                    "predicted_direction": _i_dir,
                    "target_timestamp": (datetime.utcnow() + pd.Timedelta(minutes=_fm2)).isoformat(),
                    "actual_price": None, "correct": None,
                    "indicator_scores": _ind_scores_snap,
                })
            _ipreds.extend(_new_recs)
            _ipreds = _ipreds[-500:]
            save_intraday_preds(_ipreds)

        if _i_dir > 0:
            _fi_col, _fi_mkr, _fi_arrow = ("#78909c" if _i_lean else "#4caf50"), "^", "▲"
        elif _i_dir < 0:
            _fi_col, _fi_mkr, _fi_arrow = ("#78909c" if _i_lean else "#ef5350"), "v", "▼"
        else:
            _fi_col, _fi_mkr, _fi_arrow = "#555", "o", "◆"

        # (Intraday details live on the chart — no separate metric cards needed)
    else:
        _i_dir, _i_atr, _i_conf, _i_lean = 0, 0.0, 0.0, False
        _fi_col, _fi_mkr, _fi_arrow = "#555", "o", "◆"
        _fi_ts, _fi_prices, _fhorizons = {}, {}, []
        _i_last_ts, _i_last_price = None, None
        st.info("Intraday data unavailable — market may be closed or outside session hours.")

    # ── Build daily-horizon data ──────────────────────────────────────────
    _hist5       = gold_2y.tail(5).copy()
    _last_date   = _hist5.index[-1]
    _last_price  = float(_hist5.iloc[-1])
    _avg_move    = float(gold_2y.pct_change().dropna().tail(20).abs().mean())

    def _nth_td(base, n):
        d, count = base, 0
        while count < n:
            d += pd.Timedelta(days=1)
            if d.dayofweek < 5:
                count += 1
        return d

    _max_horizon    = 5
    _target_hz      = [1, 2, 5]
    _all_future     = [_nth_td(_last_date, k) for k in range(1, _max_horizon + 1)]
    _horizon_dates  = {k: _nth_td(_last_date, k) for k in _target_hz}

    _fb_dir  = (sorted(live_preds, key=lambda p: p["target_date"])[-1]["direction"]   if live_preds else 1)
    _fb_conf = (sorted(live_preds, key=lambda p: p["target_date"])[-1]["confidence"] if live_preds else 0.55)
    _horizon_info = {}
    for _hn in _target_hz:
        _mh = mh_preds.get(str(_hn)) or mh_preds.get(_hn)
        if _mh:
            _hd, _hc = _mh["direction"], _mh["confidence"]
        else:
            _hd, _hc = _fb_dir, _fb_conf
        _hd_sign = 1 if _hd == 1 else -1
        _horizon_info[_hn] = {
            "price":      _last_price * (1 + _hd_sign * _avg_move * (_hn ** 0.5) * (_hc / 0.5)),
            "direction":  _hd, "confidence": _hc,
            "color":      "#4caf50" if _hd == 1 else "#ef5350",
            "marker":     "^"       if _hd == 1 else "v",
            "arrow":      "▲ UP"    if _hd == 1 else "▼ DOWN",
        }

    # ── Blended consensus: technical intraday + ML multi-day ─────────────
    _1d_ml   = _horizon_info.get(1, {})
    _ml_sign = 1 if _1d_ml.get("direction", 0) == 1 else (-1 if _1d_ml.get("direction", 0) == -1 else 0)
    _ml_c    = _1d_ml.get("confidence", 0.0)
    # Weight: 45% intraday technical, 55% ML ensemble
    _cons_score   = (_i_dir * _i_conf * 0.45) + (_ml_sign * _ml_c * 0.55)
    _cons_dir     = 1 if _cons_score > 0.03 else (-1 if _cons_score < -0.03 else 0)
    _cons_conv    = min(abs(_cons_score) * 1.8, 0.99)
    _cons_aligned = (_i_dir == _ml_sign) and _i_dir != 0
    if _cons_dir == 1:
        _cons_label = "▲ BULLISH"
        _cons_col   = "#4caf50"
        _cons_bg    = "#0a2010"
        _cons_border= "#4caf5066"
    elif _cons_dir == -1:
        _cons_label = "▼ BEARISH"
        _cons_col   = "#ef5350"
        _cons_bg    = "#200a0a"
        _cons_border= "#ef535066"
    else:
        _cons_label = "◆ NEUTRAL"
        _cons_col   = "#78909c"
        _cons_bg    = "#111827"
        _cons_border= "#3a4a60"
    _align_txt  = "✓ Systems aligned" if _cons_aligned else "⚡ Systems split"
    _align_col  = "#4caf50" if _cons_aligned else "#f5a623"
    _tech_word  = {1: "BUY", -1: "SELL", 0: "NEUTRAL"}.get(_i_dir, "—")
    _ml_word    = {1: "UP", -1: "DOWN", 0: "FLAT"}.get(_ml_sign, "—")
    # Override chart line colour to consensus
    if _cons_dir == 1:
        _fi_col = "#4caf50"
    elif _cons_dir == -1:
        _fi_col = "#ef5350"

    st.markdown(f"""
<div style="background:{_cons_bg};border:1px solid {_cons_border};border-radius:10px;
     padding:14px 20px;margin-bottom:12px;display:flex;align-items:center;gap:20px;flex-wrap:wrap;">
  <div>
    <div style="font-size:10px;color:#6a7a94;text-transform:uppercase;letter-spacing:1px;
         margin-bottom:4px">Combined Signal</div>
    <div style="font-size:22px;font-weight:900;color:{_cons_col};letter-spacing:0.5px">
      {_cons_label}</div>
    <div style="font-size:12px;color:#aaa;margin-top:3px">{_cons_conv:.0%} conviction</div>
  </div>
  <div style="width:1px;height:50px;background:#3a4a60;"></div>
  <div style="flex:1;min-width:180px">
    <div style="font-size:11px;color:{_align_col};font-weight:700;margin-bottom:6px">{_align_txt}</div>
    <div style="font-size:11px;color:#aaa">
      Short-term (technical): <b style="color:#ddd">{_tech_word}</b> · {_i_conf:.0%} conf
    </div>
    <div style="font-size:11px;color:#aaa;margin-top:3px">
      Multi-day (ML model): <b style="color:#ddd">{_ml_word}</b> · {_ml_c:.0%} conf
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # UNIFIED LIVE CHART
    # ═══════════════════════════════════════════════════════════════════════
    st.subheader("📈 Gold — Live Chart with Forecast Overlay")
    _has_intraday = (_i5m is not None and len(_i5m) >= 4)

    # Taller figure so labels have room and don't overlap
    _fig0, _ax0 = plt.subplots(figsize=(15, 10))
    _fig0.patch.set_facecolor("#111827")
    _ax0.set_facecolor("#111827")

    # ── Live price line ────────────────────────────────────────────────────
    if _has_intraday:
        _i5m_full = _i5m.copy()
        _AEST = pd.Timedelta(hours=10)
        if hasattr(_i5m_full.index, "tzinfo") and _i5m_full.index.tzinfo is not None:
            _i5m_full.index = _i5m_full.index.tz_convert("UTC").tz_localize(None) + _AEST
        elif hasattr(_i5m_full.index, "tz") and _i5m_full.index.tz is not None:
            _i5m_full.index = _i5m_full.index.tz_convert("UTC").tz_localize(None) + _AEST
        else:
            _i5m_full.index = _i5m_full.index + _AEST
        _ax0.plot(_i5m_full.index, _i5m_full.values,
                  color="#ffd700", lw=2.2, label="Live price (5-min bars)",
                  solid_capstyle="round", zorder=4)
        # Use the real-time live price for the NOW dot so it matches the top boxes
        _chart_live_price = _frag_live_price if _frag_live_price else _i_last_price
        # Rebase all short-term forecast prices to start from the live price
        # (they were calculated from _i_last_price; shift by the difference)
        _price_offset = _chart_live_price - _i_last_price
        _fi_prices = {fl: fp + _price_offset for fl, fp in _fi_prices.items()}
        # The NOW time anchor is the real current time (AEST-adjusted)
        _now_ts    = pd.Timestamp(datetime.utcnow()) + _AEST_SHIFT
        _now_price = _chart_live_price
        _ax0.scatter([_now_ts], [_chart_live_price],
                     color="#ffd700", s=180, zorder=9, edgecolors="white", linewidths=1.4)
        _ax0.annotate(f"  ${_chart_live_price:,.2f}",
                      xy=(_now_ts, _chart_live_price),
                      color="#ffd700", fontsize=11, fontweight="bold",
                      xytext=(6, 0), textcoords="offset points", va="center")
        _vis_base  = float(_i5m_full.min())
        _i_atr_val = _i_atr if _i_atr else _i_last_price * 0.008
    else:
        _now_ts    = pd.Timestamp(_last_date)
        _now_price = _last_price
        _vis_base  = _last_price
        _i_atr_val = _last_price * 0.008
        _ax0.scatter([_now_ts], [_now_price], color="#ffd700", s=120, zorder=7,
                     edgecolors="white", linewidths=1.0)
        _ax0.annotate(f"  ${_now_price:,.0f}  (market closed)",
                      xy=(_now_ts, _now_price), color="#aaa", fontsize=10,
                      xytext=(6, 0), textcoords="offset points", va="center")

    if _all_future:
        _ax0.axvspan(_now_ts, pd.Timestamp(_all_future[-1]), alpha=0.05, color="steelblue", zorder=0)
    _ax0.axvline(x=_now_ts, color="#4a5a70", linestyle=":", lw=1.5, zorder=1)
    _ax0.text(_now_ts + pd.Timedelta(minutes=15), _vis_base * 0.9993,
              "NOW", color="#6a7a94", fontsize=8, va="bottom")

    # ── Short-term forecast line + staggered labels ────────────────────────
    if _has_intraday and _i_dir != 0:
        _st_xs = [_now_ts] + [_fi_ts[fl] for fl, _ in _fhorizons
                               if fl not in ("End of Day", "10 min")]
        _st_ys = [_now_price] + [_fi_prices[fl] for fl, _ in _fhorizons
                                  if fl not in ("End of Day", "10 min")]
        _ax0.plot(_st_xs, _st_ys,
                  color=_fi_col, lw=2.0, linestyle="--", alpha=0.9, zorder=5,
                  label=f"Short-term forecast ({_fi_arrow})")
        # Stagger labels: alternate up/down offsets so boxes never overlap
        _st_label_fhorizons = [(fl, fm) for fl, fm in _fhorizons
                               if fl not in ("End of Day", "10 min")]
        for _si, (_fl, _fm) in enumerate(_st_label_fhorizons):
            # Base direction + alternating shift keeps boxes well-separated
            _base = 26 if _i_dir >= 0 else -30
            _alt  = 28 if _si % 2 == 0 else -28
            _yoff = _base + _alt
            _va_l = "bottom" if _yoff > 0 else "top"
            _ax0.scatter([_fi_ts[_fl]], [_fi_prices[_fl]],
                         color=_fi_col, s=100, marker=_fi_mkr, zorder=8,
                         edgecolors="white", linewidths=0.7)
            _ax0.annotate(f"{_fl}\n${_fi_prices[_fl]:,.0f}",
                          xy=(_fi_ts[_fl], _fi_prices[_fl]),
                          xytext=(0, _yoff), textcoords="offset points",
                          fontsize=8, color=_fi_col, fontweight="bold",
                          ha="center", va=_va_l,
                          bbox=dict(boxstyle="round,pad=0.3", fc="#111827",
                                    ec=_fi_col, alpha=0.9, lw=0.8))
        _bridge_x = _fi_ts.get("5 hours", _now_ts)
        _bridge_y = _fi_prices.get("5 hours", _now_price)
    else:
        _bridge_x = _now_ts
        _bridge_y = _now_price

    # ── Multi-day forecasts: right-margin text labels (no chart clutter) ──
    # 20% right margin reserved for the labels; set via tight_layout rect below
    _margin_y_positions = [0.78, 0.60, 0.42]  # stacked down the right margin
    for _di, _hn in enumerate(_target_hz):
        _info = _horizon_info.get(_hn)
        if not _info:
            continue
        _ypos  = _margin_y_positions[_di % len(_margin_y_positions)]
        _arrow_ch = "▲" if _info["direction"] == 1 else "▼"
        # Day label line
        _ax0.text(1.04, _ypos + 0.05,
                  f"+{_hn}{'d' if _hn == 1 else 'd'}",
                  transform=_ax0.transAxes,
                  color="#6a7a94", fontsize=9, fontweight="bold",
                  va="bottom", ha="left")
        # Price + direction line
        _ax0.text(1.04, _ypos,
                  f"{_arrow_ch} ${_info['price']:,.0f}",
                  transform=_ax0.transAxes,
                  color=_info["color"], fontsize=11, fontweight="bold",
                  va="top", ha="left")
        # Confidence line
        _ax0.text(1.04, _ypos - 0.06,
                  f"{_info['confidence']:.0%} conf",
                  transform=_ax0.transAxes,
                  color=_info["color"] + "99", fontsize=8,
                  va="top", ha="left")

    # Thin dotted horizontal line at each forecast price (right portion only)
    for _hn in _target_hz:
        _info = _horizon_info.get(_hn)
        if _info:
            _ax0.axhline(_info["price"], color=_info["color"],
                         lw=0.7, linestyle=":", alpha=0.35, zorder=1)

    # ── Axis limits: centre NOW on the chart (4h history | 4h forecast) ───
    _xmin = _now_ts - pd.Timedelta(hours=4)
    _xmax = _now_ts + pd.Timedelta(hours=4)
    _ax0.set_xlim(_xmin, _xmax)

    _all_vis_y = [_now_price]
    if _has_intraday:
        _all_vis_y += list(_i5m_full.values.astype(float))
    if _has_intraday and _i_dir != 0:
        _all_vis_y += [_fi_prices[fl] for fl, _ in _fhorizons if fl != "End of Day"]
    if _all_vis_y:
        _ypad = max(float(_i_atr_val * 1.5), 20.0)
        _ax0.set_ylim(min(_all_vis_y) - _ypad, max(_all_vis_y) + _ypad)

    _ax0.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    _ax0.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    _ax0.tick_params(colors="#aaa", labelsize=8)
    _ax0.set_ylabel("USD / oz", color="#aaa", fontsize=9)
    _ax0.spines[["top", "right", "left", "bottom"]].set_color("#3a4a60")
    _ax0.grid(alpha=0.10, color="#3a4a60")
    _ax0.legend(loc="upper left", framealpha=0.25, labelcolor="#ddd",
                facecolor="#1a2235", edgecolor="#3a4a60", fontsize=8.5)

    # "Multi-day forecasts →" header text in right margin
    _ax0.text(1.04, 0.96, "Multi-day\nforecasts →",
              transform=_ax0.transAxes, color="#6a7a94",
              fontsize=8, va="top", ha="left",
              style="italic")

    _fig0.tight_layout(pad=1.8, rect=[0, 0, 0.80, 1])
    st.pyplot(_fig0, width="stretch")
    plt.close(_fig0)

    # (Multi-day forecasts shown in chart right margin — no separate cards needed)


# ─────────────────────────────────────────────
if gold_2y is not None and len(gold_2y) > 5:
    latest  = float(gold_2y.iloc[-1])
    prev    = float(gold_2y.iloc[-2])
    chg     = latest - prev
    chg_pct = chg / prev * 100
    hi52    = float(gold_2y.tail(252).max())
    lo52    = float(gold_2y.tail(252).min())
    chg30   = (latest / float(gold_2y.iloc[-22]) - 1) * 100 if len(gold_2y) > 22 else 0

    # ── Live streaming price ticker ──────────────────────────────────────────
    # Fetch the freshest 1-min bar — TTL=4s so it re-fetches on each 5s page refresh
    _live = fetch_live_price()
    _live_price = _live["price"]      if _live else latest
    _live_prev  = _live["prev_close"] if _live else prev
    _live_ts    = _live["ts"]         if _live else "—"

    _lchg     = _live_price - _live_prev
    _lchg_pct = _lchg / _live_prev * 100 if _live_prev else 0
    _lchg_col   = "#4caf50" if _lchg >= 0 else "#ef5350"
    _larrow     = "▲" if _lchg >= 0 else "▼"
    _flash_cls  = "flash-up" if _lchg >= 0 else "flash-dn"

    _live_source = _live.get("source", "Gold Spot · USD/oz") if _live else "Gold Spot · USD/oz"

    _ticker_html = f"""
    <style>
      @keyframes flash-up {{ 0%,100%{{background:transparent}} 40%{{background:#0d3320}} }}
      @keyframes flash-dn {{ 0%,100%{{background:transparent}} 40%{{background:#3b0d0d}} }}
      @keyframes pulse    {{ 0%,100%{{opacity:1}} 50%{{opacity:.25}} }}
      #gt-wrap {{
        display:flex; align-items:center; gap:28px; flex-wrap:wrap;
        background:#1a2235; border:1px solid #2a2a2a;
        border-radius:10px; padding:12px 22px; margin-bottom:8px;
        font-family:monospace;
      }}
      #gt-label  {{ font-size:10px; color:#8a9ab5; text-transform:uppercase;
                    letter-spacing:2px; margin-bottom:2px; }}
      #gt-price  {{ font-size:36px; font-weight:900; color:#f5c518;
                    letter-spacing:1px; border-radius:6px; padding:0 4px; }}
      #gt-chg    {{ font-size:15px; font-weight:700; }}
      #gt-right  {{ margin-left:auto; text-align:right; }}
      #gt-status {{ font-size:11px; font-weight:700; }}
      #gt-ts     {{ font-size:9px; color:#8a9ab5; margin-top:3px; }}
      #gt-src    {{ font-size:9px; color:#8a9ab5; }}
      .flash-up  {{ animation: flash-up .5s ease; }}
      .flash-dn  {{ animation: flash-dn .5s ease; }}
      .dot-live  {{ color:#4caf50; animation: pulse 1.5s infinite; }}
      .dot-conn  {{ color:#f5c518; }}
      .dot-off   {{ color:#ef5350; }}
    </style>
    <div id="gt-wrap">
      <div>
        <div id="gt-label" id="gt-src">{_live_source}</div>
        <div id="gt-price">${_live_price:,.2f}</div>
      </div>
      <div id="gt-chg" style="color:{_lchg_col}">
        {_larrow} {_lchg:+.2f} &nbsp; ({_lchg_pct:+.2f}%)
      </div>
      <div id="gt-right">
        <div id="gt-status" class="dot-conn">⬤ Connecting…</div>
        <div id="gt-ts">Seed: {_live_ts}</div>
        <div id="gt-src2" style="font-size:9px;color:#8a9ab5;">{_live_source}</div>
      </div>
    </div>
    <script>
    (function() {{
      const prevClose  = {_live_prev};
      let   prevPrice  = {_live_price};
      let   ticks      = 0;
      let   wsMode     = false;
      let   lastServer = Date.now();
      // Server-rendered seed price: Swissquote/Stooq/GC=F median, accurate to ~$5 of spot.
      // Used to instantly calibrate the WS basis on the first tick so the ticker never
      // shows the raw XAUT crypto-discount price (which can be $40+ below IG spot).
      const _seedPrice = {_live_price};
      // WS spot calibration: XAUT/PAXG on crypto exchanges can trade $30-50 below
      // actual XAU/USD spot during risk-off.  Every 30 s we fetch Swissquote (free
      // interbank feed) and compute the basis so we can correct WS prices.
      let   wsBasisAdj = 0;     // add to raw WS price to get estimated spot
      let   lastWsRaw  = null;  // last raw WS price (before basis adj)

      function fmt(n) {{
        return '$' + n.toFixed(2).replace(/\\B(?=(\\d{{3}})+(?!\\d))/g, ',');
      }}
      function fmtChg(price) {{
        const c   = price - prevClose;
        const pct = c / prevClose * 100;
        const pos = c >= 0;
        return {{
          text : (pos ? '▲ +' : '▼ ') + c.toFixed(2) + '\u00a0(' + (pos?'+':'') + pct.toFixed(2) + '%)',
          color: pos ? '#4caf50' : '#ef5350'
        }};
      }}
      function flash(price) {{
        const el  = document.getElementById('gt-price');
        const cls = price >= prevPrice ? 'flash-up' : 'flash-dn';
        el.classList.remove('flash-up','flash-dn');
        void el.offsetWidth;
        el.classList.add(cls);
      }}
      // ── Swissquote basis calibration ──────────────────────────────────────
      // Periodically fetches true XAU/USD interbank spot from Swissquote and
      // computes how far the WS source (XAUT tokenized gold) is from real spot.
      // Allows live WS ticks while still being accurate vs IG/OTC prices.
      function calibrateWsBasis() {{
        if (!wsMode || lastWsRaw === null) return;
        fetch('https://forex-data-feed.swissquote.com/public-quotes/bboquotes/instrument/XAU/USD')
          .then(r => r.json())
          .then(d => {{
            let spot = null;
            for (const v of d) {{
              for (const p of (v.spreadProfilePrices || [])) {{
                if (p.bid && p.ask) {{ spot = (parseFloat(p.bid) + parseFloat(p.ask)) / 2; break; }}
              }}
              if (spot) break;
            }}
            if (spot && spot > 1000) {{
              const newAdj = spot - lastWsRaw;
              // Sanity check: basis must be <5% of spot to be applied
              if (Math.abs(newAdj) < spot * 0.05) {{
                wsBasisAdj = newAdj;
                const adjEl = document.getElementById('gt-src2');
                if (adjEl && newAdj !== 0) {{
                  adjEl.textContent = 'Interbank calibrated · basis ' +
                    (newAdj >= 0 ? '+' : '') + newAdj.toFixed(1);
                }}
              }}
            }}
          }}).catch(() => {{}});
      }}
      // Run calibration immediately and then every 8 s (reduced from 30s)
      // Faster recalibration tightens the gap between our price and IG's quote.
      setInterval(calibrateWsBasis, 8000);

      function update(price, label) {{
        // Apply spot-calibration basis when WS is active.
        // wsBasisAdj compensates for XAUT/PAXG crypto-market discount vs interbank spot.
        const displayPrice = wsMode ? price + wsBasisAdj : price;
        if (wsMode) lastWsRaw = price;
        if (displayPrice !== prevPrice) {{
          flash(displayPrice);
          const chg = fmtChg(displayPrice);
          document.getElementById('gt-price').textContent = fmt(displayPrice);
          document.getElementById('gt-chg').textContent   = chg.text;
          document.getElementById('gt-chg').style.color   = chg.color;
          ticks++;
          prevPrice = displayPrice;
        }}
        if (label) {{
          document.getElementById('gt-status').textContent = label;
          document.getElementById('gt-status').className   = 'dot-live';
        }}
      }}

      // ── Try WebSocket sources in order ──────────────────────────────────
      // OKX first — most reliably reachable from cloud/server environments.
      // Binance global second — highly liquid but sometimes blocked by region.
      // Binance US as last resort.
      const WS_SOURCES = [
        // OKX — XAUT-USDT spot (subscription-based, very reliable from server IPs)
        {{ url: 'wss://ws.okx.com:8443/ws/v5/public',
           sub: JSON.stringify({{"op":"subscribe","args":[{{"channel":"tickers","instId":"XAUT-USDT"}}]}}),
           parse: (d) => (d.data && d.data[0]) ? parseFloat(d.data[0].last) : null,
           label: '⬤ WS · OKX · XAUT/USDT' }},
        // Binance — XAUT/USDT Gold Spot
        {{ url: 'wss://stream.binance.com:9443/ws/xautusdt@miniTicker',
           sub: null,
           parse: (d) => parseFloat(d.c),
           label: '⬤ WS · Binance · XAUT/USDT' }},
        // Binance US fallback
        {{ url: 'wss://stream.binance.us:9443/ws/xautusd@miniTicker',
           sub: null,
           parse: (d) => parseFloat(d.c),
           label: '⬤ WS · Binance.US · XAUT' }},
      ];

      let wsIdx = 0;
      function tryNextWS() {{
        if (wsIdx >= WS_SOURCES.length) {{
          // All WebSockets failed — switch to REST polling
          wsMode = false;
          startRestPolling();
          return;
        }}
        const src = WS_SOURCES[wsIdx++];
        const ws  = new WebSocket(src.url);
        let opened = false;

        ws.onopen = () => {{
          opened = true;
          wsMode = true;
          if (src.sub) ws.send(src.sub);
          document.getElementById('gt-status').textContent = src.label;
          document.getElementById('gt-status').className   = 'dot-live';
          document.getElementById('gt-src2').textContent   = 'Real-time WebSocket stream';
        }};
        let _wsFirstTick = true;
        ws.onmessage = (e) => {{
          const d = JSON.parse(e.data);
          const p = src.parse(d);
          if (p && p > 100) {{
            // ── First-tick: seed WS basis from Python server price ────────────
            // _seedPrice is the Swissquote/Stooq/GC=F median computed server-side
            // and is accurate to within ~$5 of IG spot.  Setting wsBasisAdj here
            // (BEFORE calling update()) means the very first WS display is already
            // corrected — we never flash the raw XAUT crypto-discount price.
            // The browser-side Swissquote CORS fetch (calibrateWsBasis) will try
            // to refine this after 3 s, but its silent failure is not a problem.
            if (_wsFirstTick) {{
              _wsFirstTick = false;
              if (_seedPrice > 0) {{
                wsBasisAdj = _seedPrice - p;
              }}
              setTimeout(calibrateWsBasis, 3000);
            }}
            update(p, src.label + ' · Tick #' + (++ticks));
            document.getElementById('gt-ts').textContent =
              new Date().toLocaleTimeString('en-GB') + ' local';
          }}
        }};
        ws.onerror = () => {{ if (!opened) tryNextWS(); }};
        ws.onclose = () => {{
          if (wsMode) {{
            wsMode = false;
            document.getElementById('gt-status').textContent = '⬤ Reconnecting…';
            document.getElementById('gt-status').className   = 'dot-conn';
            setTimeout(() => {{ wsIdx = 0; tryNextWS(); }}, 2000);
          }}
        }};
        // If no open within 2.5 s, move to next source
        setTimeout(() => {{ if (!opened) {{ ws.close(); tryNextWS(); }} }}, 2500);
      }}

      // ── REST polling fallback when all WebSockets unavailable ────────────
      // Sources: Swissquote interbank XAU/USD spot (primary, most accurate),
      // Yahoo GC=F COMEX futures (within $35 of spot), OKX XAUT as backup.
      // Previous PAXG/XAUT-only sources caused $30-50 underpricing vs IG during
      // risk-off sessions when crypto-tokenized gold trades at a discount.
      const REST_SOURCES = [
        {{ url: 'https://forex-data-feed.swissquote.com/public-quotes/bboquotes/instrument/XAU/USD',
           parse: (d) => {{
             for (const v of d) {{
               for (const p of (v.spreadProfilePrices || [])) {{
                 if (p.bid && p.ask) return (parseFloat(p.bid) + parseFloat(p.ask)) / 2;
               }}
             }}
             return null;
           }},
           name: 'Swissquote·Spot' }},
        {{ url: 'https://query2.finance.yahoo.com/v8/finance/chart/GC%3DF?interval=1m&range=1d',
           parse: (d) => d.chart && d.chart.result && d.chart.result[0]
                         ? parseFloat(d.chart.result[0].meta.regularMarketPrice) : null,
           name: 'Yahoo·GC=F' }},
        {{ url: 'https://www.okx.com/api/v5/market/ticker?instId=XAUT-USDT',
           parse: (d) => d.data && d.data[0] ? parseFloat(d.data[0].last) : null,
           name: 'OKX·XAUT' }},
      ];
      let restTimer = null;

      function _median(arr) {{
        const s = arr.slice().sort((a,b) => a-b);
        const m = Math.floor(s.length/2);
        return s.length % 2 === 0 ? (s[m-1]+s[m])/2 : s[m];
      }}

      function pollREST() {{
        if (wsMode) return;
        const promises = REST_SOURCES.map(src =>
          fetch(src.url).then(r => r.json()).then(d => src.parse(d)).catch(() => null)
        );
        Promise.all(promises).then(values => {{
          const valid = values.filter(v => v && v > 100);
          if (valid.length === 0) return;
          const med = _median(valid);
          // Drop prices >1.5% from median (outlier rejection)
          const consensus = valid.filter(v => Math.abs(v - med) / med <= 0.015);
          const price = consensus.length > 0 ? _median(consensus) : med;
          const n = consensus.length || valid.length;
          const label = '⬤ REST · ' + n + ' sources';
          update(price, label);
          document.getElementById('gt-ts').textContent   = new Date().toLocaleTimeString('en-GB') + ' local';
          document.getElementById('gt-src2').textContent = 'Multi-source median · ' + n + '/' + REST_SOURCES.length + ' live';
          document.getElementById('gt-status').textContent = label;
          document.getElementById('gt-status').className   = 'dot-live';
        }}).finally(() => {{
          if (!wsMode) restTimer = setTimeout(pollREST, 1500);
        }});
      }}

      function startRestPolling() {{
        document.getElementById('gt-src2').textContent = 'Multi-source REST · starting…';
        if (restTimer) clearTimeout(restTimer);
        pollREST();
      }}

      // Flash seed price on load
      flash(prevPrice + 0.01);

      tryNextWS();
    }})();
    </script>
    """
    st.components.v1.html(_ticker_html, height=90)

    # ── Live chart with forecast overlay ─────────────────────────────────────
    _live_chart_fragment()

    # ── Reference stats ───────────────────────────────────────────────────────
    m2, m3, m4 = st.columns(3)
    m2.metric("30-day change", f"{chg30:+.2f}%")
    m3.metric("52-week high",  f"${hi52:,.2f}")
    m4.metric("52-week low",   f"${lo52:,.2f}")

    # ── News Sentiment + Economic Calendar ───────────────────────────────────
    try:
        from gold_model import load_cached_sentiment
        from economic_calendar import get_upcoming_events, get_current_event_flags
        _sent     = load_cached_sentiment()
        _cal_flags = get_current_event_flags()
        _upcoming  = get_upcoming_events(n_days=21)

        _sc1, _sc2 = st.columns([1, 1])

        # News Sentiment card
        with _sc1:
            _ss = float(_sent.get("score", 0.0))
            _sb = _sent.get("bullish_n", 0)
            _sr = _sent.get("bearish_n", 0)
            _sn = _sent.get("total_n", 0)
            _sc = "#4caf50" if _ss > 0.1 else ("#ef5350" if _ss < -0.1 else "#888")
            _sl = "BULLISH" if _ss > 0.1 else ("BEARISH" if _ss < -0.1 else "NEUTRAL")
            _sa = "▲" if _ss > 0.1 else ("▼" if _ss < -0.1 else "—")
            _age_h = _sent.get("age_h", 0)
            if _sent.get("stale") and _sn > 0:
                _stale_note = f" · ⏳ {_age_h:.0f}h old"
            elif _sent.get("stale"):
                _stale_note = " · ⏳ refreshing…"
            else:
                _stale_note = ""
            _border_color = "#2a2a2a" if not _sent.get("stale") else "#3a2a00"
            st.markdown(
                f"""<div style="background:#1a2235;border:1px solid {_border_color};border-radius:10px;
                    padding:14px 18px;margin-bottom:4px;">
                    <div style="font-size:10px;color:#8a9ab5;text-transform:uppercase;margin-bottom:6px;">
                        📰 News Sentiment · {_sn} headlines{_stale_note}
                    </div>
                    <div style="display:flex;align-items:center;gap:14px;">
                        <div style="font-size:24px;font-weight:900;color:{_sc};">
                            {_sa} {_sl}
                        </div>
                        <div style="font-size:12px;color:#888;">
                            score <b style="color:{_sc};">{_ss:+.2f}</b>
                        </div>
                    </div>
                    <div style="font-size:11px;color:#9ba8bc;margin-top:4px;">
                        {_sb} bullish · {_sr} bearish · 
                        {_sn - _sb - _sr} neutral headlines
                    </div>
                    {"".join(f'<div style="font-size:10px;color:#8a9ab5;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">· {h[0][:80]}…</div>' for h in _sent.get("headlines", [])[:3])}
                </div>""",
                unsafe_allow_html=True,
            )

        # Macro Events card
        with _sc2:
            _today_flags = []
            if _cal_flags.get("fomc_in_3d"):
                _today_flags.append(("🏦 FOMC", _cal_flags["days_to_fomc"], "#ef5350"))
            if _cal_flags.get("cpi_in_3d"):
                _today_flags.append(("📊 CPI", _cal_flags["days_to_cpi"], "#ff9800"))
            if _cal_flags.get("nfp_in_3d"):
                _today_flags.append(("💼 NFP", _cal_flags["days_to_nfp"], "#ffa726"))

            _alert_html = ""
            for _ename, _edays, _ecol in _today_flags:
                _etxt = "TODAY" if _edays == 0 else f"in {_edays}d"
                _alert_html += (
                    f'<div style="background:{_ecol}22;border:1px solid {_ecol};border-radius:6px;'
                    f'padding:4px 10px;margin-bottom:4px;font-size:11px;font-weight:700;color:{_ecol};">'
                    f'{_ename} — {_etxt} ⚠️ High-impact event</div>'
                )

            _upcoming_html = ""
            for _ev in _upcoming[:4]:
                _dc  = "#ef5350" if _ev["impact"] == "HIGH" else "#f5c518"
                _dtx = "TODAY" if _ev["days_away"] == 0 else f"in {_ev['days_away']}d"
                _upcoming_html += (
                    f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'padding:4px 0;border-bottom:1px solid #1a1a1a;">'
                    f'<span style="font-size:11px;color:#aaa;">{_ev["label"]}</span>'
                    f'<span style="font-size:11px;font-weight:700;color:{_dc};">{_dtx}</span>'
                    f'</div>'
                )

            st.markdown(
                f"""<div style="background:#1a2235;border:1px solid #2a2a2a;border-radius:10px;
                    padding:14px 18px;margin-bottom:4px;">
                    <div style="font-size:10px;color:#8a9ab5;text-transform:uppercase;margin-bottom:8px;">
                        📅 Upcoming Macro Events (21-day window)
                    </div>
                    {_alert_html if _alert_html else '<div style="font-size:11px;color:#8a9ab5;">No high-impact events within 3 days</div>'}
                    <div style="margin-top:8px;">{_upcoming_html}</div>
                </div>""",
                unsafe_allow_html=True,
            )
    except Exception as _cal_ex:
        pass   # silently skip if calendar module not yet available

    # ── Self-Audit: System Intelligence Panel ──────────────────────────────────
    try:
        from adaptive_learning import summary_stats as _al_stats, load_analysis_log as _al_log

        _als  = _al_stats()
        _alw  = {k: v for k, v in _als.get("weights", {}).items() if isinstance(v, float)}
        _alt  = _als.get("total_analyses", 0)
        _allog = _al_log()

        # Recent accuracy from last 20 resolved log entries
        _recent_correct = sum(1 for e in _allog[-20:] if e.get("actual") == e.get("predicted"))
        _recent_n       = min(20, len(_allog))
        _recent_acc     = _recent_correct / _recent_n if _recent_n > 0 else None

        # Top 3 strongest and weakest indicators
        _sorted_w = sorted(_alw.items(), key=lambda x: x[1], reverse=True)
        _top3     = _sorted_w[:3]
        _bot3     = _sorted_w[-3:]

        def _w_bar(w, max_w=2.8, min_w=0.3):
            frac = (w - min_w) / (max_w - min_w) if max_w > min_w else 0.5
            frac = max(0.0, min(1.0, frac))
            col  = f"hsl({int(frac * 120)}, 70%, 45%)"
            return f'<div style="width:{max(4, int(frac*80))}px;height:6px;background:{col};border-radius:3px;display:inline-block;margin-left:6px;vertical-align:middle;"></div>'

        _ind_html = "".join(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:3px 0;">'
            f'<span style="font-size:11px;color:#888;">{nm}</span>'
            f'<span style="font-size:11px;color:#aaa;font-weight:700;">{wt:.2f}'
            f'{_w_bar(wt)}</span></div>'
            for nm, wt in _sorted_w
        )

        _acc_col  = "#4caf50" if (_recent_acc or 0) >= 0.55 else (
                    "#ef5350" if (_recent_acc or 0) < 0.45 else "#ffc107")
        _acc_html = (f'<span style="color:{_acc_col};font-weight:900;">'
                     f'{_recent_acc:.0%}</span> ({_recent_n} recent trades)') \
                    if _recent_acc is not None else "—"

        _last_upd = _als.get("last_updated", "")[:16].replace("T", " ") if _als.get("last_updated") else "—"

        st.markdown(
            f"""<div style="background:#0c0e14;border:1px solid #1e2330;border-radius:12px;
                padding:16px 20px;margin-bottom:16px;">
                <div style="font-size:10px;color:#8a9ab5;text-transform:uppercase;letter-spacing:1.5px;
                    margin-bottom:12px;">🧠 System Self-Audit · Autonomous Learning Status</div>
                <div style="display:flex;gap:20px;flex-wrap:wrap;margin-bottom:14px;">
                    <div>
                        <div style="font-size:10px;color:#8a9ab5;">Total analyses run</div>
                        <div style="font-size:20px;font-weight:900;color:#f5c518;">{_alt}</div>
                    </div>
                    <div>
                        <div style="font-size:10px;color:#8a9ab5;">Recent accuracy (last {_recent_n})</div>
                        <div style="font-size:20px;font-weight:900;">{_acc_html}</div>
                    </div>
                    <div>
                        <div style="font-size:10px;color:#8a9ab5;">Last weight update</div>
                        <div style="font-size:13px;font-weight:700;color:#888;">{_last_upd}</div>
                    </div>
                </div>
                <div style="font-size:10px;color:#8a9ab5;margin-bottom:6px;text-transform:uppercase;">
                    Indicator Weights (auto-adjusted · penalised when wrong · rewarded when right)
                </div>
                {_ind_html}
                <div style="font-size:9px;color:#2a2a2a;margin-top:8px;text-align:right;">
                    High weight = reliable indicator · Low weight = recently wrong · Range: 0.30 – 2.80
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    except Exception:
        pass   # skip silently if not available yet

    # ── Market Context Panel ──────────────────────────────────────
    st.subheader("🌐 Market Context")
    st.caption(
        "Live cross-asset signals — the same patterns that drove gold's "
        "−17.7% correction from its Jan 2026 peak."
    )
    try:
        import yfinance as _yf
        _ctx_tickers = {"SPY": "SPY", "TLT": "TLT", "DXY": "DX-Y.NYB",
                        "VIX": "^VIX", "OIL": "CL=F"}
        _ctx_data = {}
        for _cn, _ct in _ctx_tickers.items():
            try:
                _s = _yf.download(_ct, period="5d", auto_adjust=True, progress=False)["Close"]
                if hasattr(_s, "columns"): _s = _s.iloc[:, 0]
                _s = _s.squeeze().dropna()
                if len(_s) >= 2:
                    _ctx_data[_cn] = float(_s.iloc[-1])
                    _ctx_data[f"{_cn}_prev"] = float(_s.iloc[-2])
            except Exception:
                pass

        def _ctx_chg(key):
            v, p = _ctx_data.get(key), _ctx_data.get(f"{key}_prev")
            return (v / p - 1) if (v and p and p) else None

        _dxy_chg = _ctx_chg("DXY")
        _vix_chg = _ctx_chg("VIX")
        _oil_chg = _ctx_chg("OIL")
        _spy_chg = _ctx_chg("SPY")
        _tlt_chg = _ctx_chg("TLT")

        # Score each shock pattern (same thresholds as model features)
        _tariff_score = sum([
            (_dxy_chg or 0) > 0.003,
            (_vix_chg or 0) > 0.05,
            (_oil_chg or 0) > 0.02,
            (_spy_chg or 0) < 0,
        ])
        _risk_on_score = sum([
            (_spy_chg or 0) > 0.005,
            (_tlt_chg or 0) > 0,
            (_vix_chg or 0) < -0.03,
        ])
        _force_liq_score = sum([
            (_spy_chg or 0) < -0.01,
            (_vix_chg or 0) > 0.10,
            (_tlt_chg or 0) < -0.005,
        ])

        # Each condition shown with its actual live value so the user
        # can see exactly why a pattern is active or not.
        def _cond_row(label, value_str, met):
            c = "#4caf50" if met else "#555"
            icon = "✓" if met else "✗"
            return (
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:center;padding:3px 0;border-bottom:1px solid #111;">'
                f'<span style="font-size:11px;color:#888;">{label}</span>'
                f'<span style="font-size:11px;font-weight:700;color:{c};">'
                f'{icon} {value_str}</span></div>'
            )

        def _pattern_card(title, score, max_score, color, conditions_html,
                          why_inactive="", consequence=""):
            active = score >= max_score - 1   # active when all or all-but-one conditions met
            full   = score >= max_score        # ALL conditions met
            pct = int(score / max_score * 100)
            border = color if active else "#333"
            status = ("🔴 FULLY ACTIVE" if full else "● ACTIVE") if active else f"○ {score}/{max_score} conditions"
            status_c = color if active else "#555"
            inactive_note = (f'<div style="font-size:10px;color:#9ba8bc;margin-top:6px;'
                             f'font-style:italic;">{why_inactive}</div>') if (not active and why_inactive) else ""
            consequence_note = (f'<div style="font-size:10px;color:{color};margin-top:8px;'
                                f'padding:5px 8px;background:{color}18;border-left:2px solid {color};'
                                f'border-radius:0 4px 4px 0;">{consequence}</div>') if (active and consequence) else ""
            return (
                f'<div style="flex:1;min-width:200px;background:#1a2235;'
                f'border:1px solid {border};border-radius:10px;padding:12px 14px;">'
                f'<div style="font-size:10px;color:#888;text-transform:uppercase;'
                f'letter-spacing:1px;font-weight:700;margin-bottom:6px;">{title}</div>'
                f'<div style="font-size:12px;font-weight:800;color:{status_c};'
                f'margin-bottom:8px;">{status}</div>'
                f'<div style="height:4px;background:#1a1a1a;border-radius:2px;margin-bottom:10px;">'
                f'<div style="width:{pct}%;height:4px;background:{color};border-radius:2px;"></div></div>'
                f'{conditions_html}'
                f'{inactive_note}'
                f'{consequence_note}'
                f'</div>'
            )

        _ts_conds = (
            _cond_row("DXY ↑ >0.3%",  f"{(_dxy_chg or 0):+.2%}", (_dxy_chg or 0) > 0.003) +
            _cond_row("VIX ↑ >5%",    f"{(_vix_chg or 0):+.2%}", (_vix_chg or 0) > 0.05)  +
            _cond_row("OIL ↑ >2%",    f"{(_oil_chg or 0):+.2%}", (_oil_chg or 0) > 0.02)  +
            _cond_row("SPY ↓",        f"{(_spy_chg or 0):+.2%}", (_spy_chg or 0) < 0)
        )
        _ro_conds = (
            _cond_row("SPY ↑ >0.5%",  f"{(_spy_chg or 0):+.2%}", (_spy_chg or 0) > 0.005) +
            _cond_row("TLT ↑",        f"{(_tlt_chg or 0):+.2%}", (_tlt_chg or 0) > 0)     +
            _cond_row("VIX ↓ >3%",    f"{(_vix_chg or 0):+.2%}", (_vix_chg or 0) < -0.03)
        )
        _fl_conds = (
            _cond_row("SPY ↓ >1%",    f"{(_spy_chg or 0):+.2%}", (_spy_chg or 0) < -0.01) +
            _cond_row("VIX ↑ >10%",   f"{(_vix_chg or 0):+.2%}", (_vix_chg or 0) > 0.10)  +
            _cond_row("TLT ↓ >0.5%",  f"{(_tlt_chg or 0):+.2%}", (_tlt_chg or 0) < -0.005)
        )

        # Explain inactive patterns so user understands why
        _ts_why = ("Today: VIX falling (fear easing) and SPY rising — market is risk-on, "
                   "not a tariff shock day.") if _tariff_score < 3 else ""
        _fl_why = ("Today: SPY rising and VIX falling — no forced selling detected.") if _force_liq_score < 2 else ""

        def _asset_pill(name, chg):
            if chg is None: return ""
            color = "#4caf50" if chg > 0 else "#ef5350"
            return (f'<span style="background:#1a1a1a;border:1px solid #222;border-radius:4px;'
                    f'padding:2px 8px;font-size:11px;color:{color};margin:2px;font-weight:700;">'
                    f'{name} {chg:+.1%}</span>')

        _ts_consequence = ("Safe-haven bid rising — DXY and gold can rally together. Watch for BUY signal on a pullback.")
        _ro_consequence = ("Money rotating OUT of gold into equities. Gold historically −0.5% to −1.5% intraday. Avoid longs — wait for rotation to exhaust.")
        _fl_consequence = ("Everything sold for cash — gold drops WITH equities initially, then recovers sharply as hedge demand returns. Avoid new longs until VIX peaks.")

        _ctx_html = (
            '<div style="margin-bottom:12px;display:flex;flex-wrap:wrap;gap:4px;">'
            + "".join(_asset_pill(k, _ctx_chg(k)) for k in ["SPY", "TLT", "DXY", "VIX", "OIL"]
                      if _ctx_chg(k) is not None)
            + '</div>'
            '<div style="display:flex;gap:12px;flex-wrap:wrap;">'
            + _pattern_card("🛡 Tariff / Policy Shock", _tariff_score, 4, "#ff9800", _ts_conds, _ts_why, _ts_consequence)
            + _pattern_card("📈 Risk-On Rotation",      _risk_on_score, 3, "#2196f3", _ro_conds, "",       _ro_consequence)
            + _pattern_card("🔥 Forced Liquidation",    _force_liq_score, 3, "#ef5350", _fl_conds, _fl_why, _fl_consequence)
            + '</div>'
        )
        st.markdown(_ctx_html, unsafe_allow_html=True)

        # ── Full-activation impact banner ────────────────────────────────
        if _risk_on_score >= 3:
            st.markdown("""
<div style="background:#0a1428;border:2px solid #2196f3;border-radius:12px;
    padding:14px 20px;margin-top:14px;display:flex;align-items:flex-start;gap:14px;">
  <div style="font-size:26px;line-height:1;">📉</div>
  <div>
    <div style="font-size:11px;font-weight:800;color:#2196f3;letter-spacing:1.5px;
        text-transform:uppercase;margin-bottom:4px;">Gold Headwind — Risk-On Rotation Fully Active</div>
    <div style="font-size:12px;color:#aaa;line-height:1.6;">
      All 3 conditions met: <b style="color:#eee;">equities rallying, bonds rising, and fear collapsing</b>.
      Institutional money is actively rotating OUT of gold into risk assets.
      Historically this suppresses gold <b style="color:#ef5350;">−0.5% to −2%</b> over the next 24 hours.
      The day forecast model is already pricing this in. <b style="color:#fff;">Do not fight the rotation —
      wait for it to exhaust (VIX base forms, SPY stalls) before re-entering long.</b>
    </div>
  </div>
</div>""", unsafe_allow_html=True)
        elif _tariff_score >= 4:
            st.markdown("""
<div style="background:#1a1000;border:2px solid #ff9800;border-radius:12px;
    padding:14px 20px;margin-top:14px;display:flex;align-items:flex-start;gap:14px;">
  <div style="font-size:26px;line-height:1;">🛡</div>
  <div>
    <div style="font-size:11px;font-weight:800;color:#ff9800;letter-spacing:1.5px;
        text-transform:uppercase;margin-bottom:4px;">Safe-Haven Bid — Tariff Shock Fully Active</div>
    <div style="font-size:12px;color:#aaa;line-height:1.6;">
      All 4 conditions met: <b style="color:#eee;">DXY rising, VIX spiking, oil up, equities falling</b>.
      Gold and the dollar are rallying together — a classic policy shock pattern.
      Historically gold gains <b style="color:#4caf50;">+0.5% to +2%</b> in the next 24 hours.
      <b style="color:#fff;">BUY signals on dips are higher conviction in this environment.</b>
    </div>
  </div>
</div>""", unsafe_allow_html=True)
        elif _force_liq_score >= 3:
            st.markdown("""
<div style="background:#1a0204;border:2px solid #ef5350;border-radius:12px;
    padding:14px 20px;margin-top:14px;display:flex;align-items:flex-start;gap:14px;">
  <div style="font-size:26px;line-height:1;">🔥</div>
  <div>
    <div style="font-size:11px;font-weight:800;color:#ef5350;letter-spacing:1.5px;
        text-transform:uppercase;margin-bottom:4px;">Forced Liquidation Fully Active</div>
    <div style="font-size:12px;color:#aaa;line-height:1.6;">
      All 3 conditions met: <b style="color:#eee;">equities crashing, VIX spiking, bonds selling off</b>.
      This is a "sell everything for cash" event — gold falls WITH equities initially.
      <b style="color:#fff;">Avoid new longs.</b> Wait for VIX to peak and form a base,
      then gold typically recovers sharply as hedge demand floods back in.
    </div>
  </div>
</div>""", unsafe_allow_html=True)
    except Exception as _ctx_err:
        import traceback as _ctx_tb
        st.markdown(
            f'<div style="background:#1a0a0a;border:1px solid #3a1a1a;border-radius:8px;'
            f'padding:12px 16px;font-size:11px;color:#888;">'
            f'⚠️ Market context unavailable — {_ctx_err}</div>',
            unsafe_allow_html=True,
        )

    # ── 2-year overview (collapsible) ─────────────────────────────
    with st.expander("Show 2-year gold price chart"):
        fig2y, ax2y = plt.subplots(figsize=(12, 3))
        ax2y.plot(gold_2y.index, gold_2y.values, color="goldenrod", lw=1.6)
        ax2y.fill_between(gold_2y.index, gold_2y.values, gold_2y.values.min() * 0.97,
                          alpha=0.08, color="goldenrod")
        ax2y.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax2y.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=25, ha="right")
        ax2y.set_ylabel("USD / oz")
        ax2y.grid(alpha=0.2)
        st.pyplot(fig2y)
        plt.close(fig2y)
else:
    st.warning("Live gold data unavailable. Check connection.")

_tab_live.__exit__(None, None, None)

# ═══════════════════════════════════════════════════════════════════════════
# ⚡ TRADE SIGNALS TAB
# ═══════════════════════════════════════════════════════════════════════════
_tab_signals.__enter__()

# ═══════════════════════════════════════════════════════════════════════════════
# ⚡ TRADE SIGNALS — select a timeframe and get a single decisive recommendation
# ═══════════════════════════════════════════════════════════════════════════════

# ── Timeframe options ──────────────────────────────────────────────────────────
_TF_MAP = {
    "⚡  15 Min":  ("⚡ 15-Min Scalp",    _15m_label, _15m_sig,  "~15–30 min hold"),
    "⏱  30 Min":  ("⏱ 30-Min Trade",    _30m_label, _30m_sig,  "~30–60 min hold"),
    "🕐  1 Hour":  ("🕐 1-Hour Trade",    _1h_label,  _top_sig,  "~1–4 hour hold"),
    "📊  4 Hours": ("📊 4-Hour Swing",    _4h_label,  _4h_sig,   "~4–24 hour hold"),
    "📅  Daily":   ("📅 Daily Trade",     _day_label, _day_sig,  "1–3 day hold"),
    "📆  Weekly":  ("📆 Weekly Position", _wk_label,  _wk_sig,   "1–4 week hold"),
}
_tf_keys = list(_TF_MAP.keys())

# ── Score each timeframe to find the best trade right now ─────────────────────
def _tf_score(lbl, sig):
    """Return a numeric score: higher = stronger, more reliable trade signal."""
    _label_pts = {"STRONG BUY": 3, "STRONG SELL": 3,
                  "BUY": 2,        "SELL": 2,
                  "LEAN BUY": 1,   "LEAN SELL": 1}.get(lbl, 0)
    if _label_pts == 0:
        return 0.0
    _conf  = float((sig or {}).get("confidence", 0.4))
    _atr   = float((sig or {}).get("atr", 1))
    _score = (sig or {}).get("total_score", 0)
    _rr    = abs(_score) / max(_atr * 0.5, 0.01)          # crude R:R proxy
    return _label_pts * (_conf + 0.1) * (1 + min(_rr, 2) * 0.1)

_tf_scores = {
    "⚡  15 Min":  _tf_score(_15m_label, _15m_sig),
    "⏱  30 Min":  _tf_score(_30m_label, _30m_sig),
    "🕐  1 Hour":  _tf_score(_1h_label,  _top_sig),
    "📊  4 Hours": _tf_score(_4h_label,  _4h_sig),
    "📅  Daily":   _tf_score(_day_label, _day_sig),
    "📆  Weekly":  _tf_score(_wk_label,  _wk_sig),
}
_best_tf_key  = max(_tf_scores, key=_tf_scores.get)
_best_tf_score = _tf_scores[_best_tf_key]
_best_tf_data  = _TF_MAP[_best_tf_key]

# Pre-select the best timeframe on first load (before user touches the radio)
if "signals_tf_pick" not in st.session_state and _best_tf_score > 0:
    st.session_state["signals_tf_pick"] = _best_tf_key

# ── Compute "why now" supporting reasons for the best trade ───────────────────
import datetime as _rdt
_btf_lbl   = _best_tf_data[1] if _best_tf_score > 0 else "HOLD"
_btf_title = _best_tf_data[0] if _best_tf_score > 0 else "—"
_btf_hold  = _best_tf_data[3] if _best_tf_score > 0 else "—"
_btf_sig   = _best_tf_data[2] if _best_tf_score > 0 else None
_btf_conf  = int(float((_btf_sig or {}).get("confidence", 0)) * 100)
_btf_is_buy  = _btf_lbl in ("BUY", "STRONG BUY", "LEAN BUY")
_btf_is_sell = _btf_lbl in ("SELL", "STRONG SELL", "LEAN SELL")
_btf_is_strong = "STRONG" in _btf_lbl

# Count how many TFs agree with the best signal direction
_btf_agree_labels = ("BUY", "STRONG BUY", "LEAN BUY") if _btf_is_buy else (("SELL", "STRONG SELL", "LEAN SELL") if _btf_is_sell else ())
_btf_agree_n = sum(1 for l in [_15m_label,_30m_label,_1h_label,_4h_label,_day_label,_wk_label] if l in _btf_agree_labels)

# Session quality
_rdt_h = _rdt.datetime.utcnow().hour
if   13 <= _rdt_h < 17: _rec_sess = "London/NY overlap — peak liquidity"
elif  8 <= _rdt_h < 13: _rec_sess = "London open — active market"
elif 17 <= _rdt_h < 21: _rec_sess = "NY afternoon — moderate liquidity"
else:                    _rec_sess = "Asian / off-peak hours"
_peak_session = 8 <= _rdt_h < 21

# RSI check vs direction
_btf_rsi   = float((_btf_sig or {}).get("rsi", 50))
_btf_macd  = float((_btf_sig or {}).get("macd", 0))
_btf_macdS = float((_btf_sig or {}).get("macd_sig", 0))
_btf_vwap  = float((_btf_sig or {}).get("vwap", _card_price))
_rsi_ok    = (_btf_is_buy  and _btf_rsi < 70) or (_btf_is_sell and _btf_rsi > 30)
_macd_ok   = (_btf_is_buy  and _btf_macd > _btf_macdS) or (_btf_is_sell and _btf_macd < _btf_macdS)
_vwap_ok   = (_btf_is_buy  and _card_price >= _btf_vwap) or (_btf_is_sell and _card_price < _btf_vwap)

# Entry / SL / TP for the banner
_btf_atr  = float((_btf_sig or {}).get("atr", _card_price * 0.008))
_btf_ent  = _card_price
_btf_sl   = float((_btf_sig or {}).get("stop_loss", _btf_ent - _btf_atr * 1.5))
_btf_tp   = float((_btf_sig or {}).get("target",    _btf_ent + _btf_atr * 2.5))
if _btf_is_sell:
    _btf_sl, _btf_tp = (
        float((_btf_sig or {}).get("stop_loss", _btf_ent + _btf_atr * 1.5)),
        float((_btf_sig or {}).get("target",    _btf_ent - _btf_atr * 2.5)),
    )
_btf_sl_dist = abs(_btf_ent - _btf_sl)
_btf_tp_dist = abs(_btf_tp  - _btf_ent)
_btf_rr      = _btf_tp_dist / max(_btf_sl_dist, 0.01)

# Conviction grade
_conf_pts  = sum([_btf_is_strong, _btf_agree_n >= 4, _rsi_ok, _macd_ok, _vwap_ok, _peak_session])
if _best_tf_score == 0 or not (_btf_is_buy or _btf_is_sell):
    _action_grade = "WAIT"
elif _conf_pts >= 4 and _btf_is_strong:
    _action_grade = "ENTER NOW"
elif _conf_pts >= 3:
    _action_grade = "GOOD SETUP"
elif _conf_pts >= 2:
    _action_grade = "CONSIDER"
else:
    _action_grade = "WAIT"

# Reason bullets
_why_bullets = []
if _btf_agree_n >= 5: _why_bullets.append(f"{_btf_agree_n}/6 timeframes confirm direction")
elif _btf_agree_n >= 4: _why_bullets.append(f"{_btf_agree_n}/6 timeframes agree")
elif _btf_agree_n >= 3: _why_bullets.append(f"{_btf_agree_n}/6 timeframes lean same way")
if _btf_is_strong:  _why_bullets.append("strong signal label")
if _macd_ok:        _why_bullets.append("MACD confirms")
if _rsi_ok:         _why_bullets.append(f"RSI {_btf_rsi:.0f} — room to run")
if _vwap_ok:        _why_bullets.append("price on correct side of VWAP")
if _peak_session:   _why_bullets.append(_rec_sess)
_why_str = " · ".join(_why_bullets[:4]) or "signals mixed — low conviction"

# ── POSITION HEALTH PANEL — shown above everything when a trade is open ────────
_pos_cfg   = {}
try:
    import json as _jpos
    _pos_cfg = _jpos.loads(Path("data_cache/alert_config.json").read_text())
except Exception:
    pass

_pos_dir   = _pos_cfg.get("position_direction")   # "LONG" | "SHORT" | None
_pos_entry = _pos_cfg.get("entry_price")
_pos_stop  = _pos_cfg.get("stop_level")
_pos_tp    = _pos_cfg.get("limit_level")

if _pos_dir in ("LONG", "SHORT") and _pos_entry:
    _px_is_long   = _pos_dir == "LONG"
    _pos_pnl_pts  = round((_card_price - _pos_entry) * (1 if _px_is_long else -1), 1)
    _pos_pnl_aud  = _pos_pnl_pts * _IG_POINT_AUD_C
    _pnl_col      = "#00e676" if _pos_pnl_pts >= 0 else "#ff3d57"
    _pnl_sign     = "+" if _pos_pnl_pts >= 0 else ""

    _px_stop_dist = round(abs(_pos_stop   - _card_price), 1) if _pos_stop else None
    _px_tp_dist   = round(abs(_pos_tp     - _card_price), 1) if _pos_tp   else None

    _all_lbl   = [_15m_label, _30m_label, _1h_label, _4h_label, _day_label, _wk_label]
    _bull_set  = {"BUY", "STRONG BUY", "LEAN BUY"}
    _bear_set  = {"SELL", "STRONG SELL", "LEAN SELL"}
    _agree_set = _bull_set if _px_is_long else _bear_set
    _oppos_set = _bear_set if _px_is_long else _bull_set
    _n_agree   = sum(1 for l in _all_lbl if l in _agree_set)
    _n_oppose  = sum(1 for l in _all_lbl if l in _oppos_set)

    _px_stop_crit = _px_stop_dist is not None and _px_stop_dist < 15
    _px_reversed  = _n_oppose >= 4

    if _px_reversed or _px_stop_crit:
        _px_rec = "CLOSE NOW"
        _px_icon = "⛔"
        _px_col = "#ff3d57"
        _px_bg  = "#1a0005"
        if _px_reversed and _px_stop_crit:
            _px_why = f"{_n_oppose}/6 timeframes have reversed AND stop is only {_px_stop_dist:.0f} pts away"
        elif _px_reversed:
            _px_why = f"{_n_oppose}/6 timeframes now oppose your {_pos_dir} — the market has turned"
        else:
            _px_why = f"Stop is only {_px_stop_dist:.0f} pts away — the market is moving against you"
        _px_steps = "IG → Positions tab → find Gold Spot → tap Close → confirm"
    elif _n_oppose >= 2 and _pos_pnl_pts > 0:
        _px_rec = "TRAIL YOUR STOP"
        _px_icon = "⚠️"
        _px_col = "#ffa726"
        _px_bg  = "#1a0f00"
        _px_why = f"In profit but {_n_oppose} timeframes are turning against you — protect what you have"
        _px_steps = "IG → Positions → Edit (pencil) → move Stop level closer to price → Confirm"
    elif _n_agree >= 3:
        _px_rec = "HOLD STEADY"
        _px_icon = "✅"
        _px_col = "#00e676"
        _px_bg  = "#011a0a"
        _px_why = f"{_n_agree}/6 timeframes support your {_pos_dir} — trend is intact, let it run"
        _px_steps = "Keep your position open — target is still valid"
    else:
        _px_rec = "WATCH CLOSELY"
        _px_icon = "👁"
        _px_col = "#90a4ae"
        _px_bg  = "#0d1117"
        _px_why = "Mixed signals — no clear edge either way"
        _px_steps = "Monitor price action — don't add risk until signals clarify"

    _px_dir_arrow = "▲" if _px_is_long else "▼"
    _stop_row = (f'<span style="color:#ef5350;font-weight:700;">'
                 f'Stop {_pos_stop:,.0f} · {_px_stop_dist:.0f} pts away</span>  ') if _px_stop_dist else ""
    _tp_row   = (f'<span style="color:#4caf50;font-weight:700;">'
                 f'TP {_pos_tp:,.0f} · {_px_tp_dist:.0f} pts to go</span>') if _px_tp_dist else ""

    st.markdown(
        f'<div style="background:{_px_bg};border:2px solid {_px_col}cc;'
        f'border-radius:14px;padding:20px 24px;margin-bottom:14px;'
        f'box-shadow:0 0 40px {_px_col}22;">'

        f'<div style="font-size:9px;color:{_px_col}99;text-transform:uppercase;'
        f'letter-spacing:2px;font-weight:700;margin-bottom:10px;">'
        f'📍 Your Open Position</div>'

        f'<div style="display:flex;justify-content:space-between;'
        f'align-items:flex-start;flex-wrap:wrap;gap:16px;">'

        f'<div style="flex:2;min-width:200px;">'
        f'<div style="font-size:26px;font-weight:900;color:{_px_col};'
        f'letter-spacing:0.5px;margin-bottom:6px;">'
        f'{_px_icon} {_px_rec}</div>'
        f'<div style="font-size:13px;color:{_px_col}cc;margin-bottom:10px;'
        f'line-height:1.5;">{_px_why}</div>'
        f'<div style="font-size:11px;color:#b0bec5;background:#ffffff0a;'
        f'border-radius:8px;padding:8px 12px;">'
        f'<b>How:</b> {_px_steps}</div>'
        f'</div>'

        f'<div style="flex:1;min-width:160px;text-align:right;">'
        f'<div style="font-size:13px;color:#cfd8dc;margin-bottom:6px;">'
        f'{_px_dir_arrow} <b>{_pos_dir}</b> entered at <b>{_pos_entry:,.1f}</b></div>'
        f'<div style="font-size:24px;font-weight:900;color:{_pnl_col};margin-bottom:4px;">'
        f'{_pnl_sign}{_pos_pnl_pts} pts</div>'
        f'<div style="font-size:16px;font-weight:700;color:{_pnl_col};margin-bottom:8px;">'
        f'{_pnl_sign}A${abs(_pos_pnl_aud):,.0f}</div>'
        f'<div style="font-size:11px;color:#90a4ae;line-height:1.8;">'
        f'{_stop_row}<br>{_tp_row}</div>'
        f'<div style="font-size:10px;color:#546e7a;margin-top:6px;">'
        f'{_n_agree}/6 TFs agree · {_n_oppose}/6 oppose</div>'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

# ── Recommendation banner ──────────────────────────────────────────────────────
if _action_grade == "WAIT":
    st.markdown(
        f'<div style="background:#0d1117;border:2px solid #37474f;border-radius:14px;'
        f'padding:18px 22px;margin-bottom:14px;">'
        f'<div style="font-size:9px;color:#546e7a;text-transform:uppercase;'
        f'letter-spacing:2px;font-weight:700;margin-bottom:6px;">⏸ Recommendation</div>'
        f'<div style="font-size:22px;font-weight:900;color:#546e7a;margin-bottom:6px;">'
        f'NO TRADE — WAIT</div>'
        f'<div style="font-size:11px;color:#546e7a;">'
        f'Signals are mixed or weak across timeframes. '
        f'Stay flat and wait for a clearer setup. '
        f'Check back in 15 minutes.</div>'
        f'</div>',
        unsafe_allow_html=True
    )
else:
    _dir_word  = "LONG (BUY)" if _btf_is_buy else "SHORT (SELL)"
    _dir_col   = "#00e676"    if _btf_is_buy else "#ff3d57"
    _dir_bg    = "#011a0a"    if _btf_is_buy else "#1a0005"
    _dir_arrow = "▲"          if _btf_is_buy else "▼"
    _tp_col    = "#4caf50"    if _btf_is_buy else "#ef5350"
    _sl_col    = "#ef5350"    if _btf_is_buy else "#4caf50"
    _grade_col = {"ENTER NOW": "#00e676", "GOOD SETUP": "#8bc34a",
                  "CONSIDER": "#ffa726"}.get(_action_grade, "#546e7a")
    _pnl_tp  = _btf_tp_dist * _IG_POINT_AUD_C
    _pnl_sl  = _btf_sl_dist * _IG_POINT_AUD_C
    st.markdown(
        f'<div style="background:{_dir_bg};border:2px solid {_dir_col}cc;border-radius:14px;'
        f'padding:18px 22px;margin-bottom:14px;'
        f'box-shadow:0 0 30px {_dir_col}18;">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;'
        f'flex-wrap:wrap;gap:12px;">'
        f'<div style="flex:1;min-width:200px;">'
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
        f'<span style="font-size:9px;color:{_dir_col}99;text-transform:uppercase;'
        f'letter-spacing:2px;font-weight:700;">⭐ Recommended Trade</span>'
        f'<span style="background:{_grade_col}22;color:{_grade_col};border:1px solid {_grade_col}55;'
        f'border-radius:4px;padding:2px 8px;font-size:9px;font-weight:900;'
        f'letter-spacing:1px;">{_action_grade}</span>'
        f'</div>'
        f'<div style="font-size:28px;font-weight:900;color:{_dir_col};'
        f'letter-spacing:0.5px;line-height:1;margin-bottom:4px;">'
        f'{_dir_arrow} {_dir_word}</div>'
        f'<div style="font-size:12px;font-weight:700;color:{_dir_col}cc;margin-bottom:8px;">'
        f'{_btf_title} &nbsp;·&nbsp; hold {_btf_hold}</div>'
        f'<div style="font-size:10px;color:#6a7a94;">{_why_str}</div>'
        f'</div>'
        f'<div style="background:#00000033;border:1px solid {_dir_col}33;border-radius:10px;'
        f'padding:12px 16px;min-width:180px;">'
        f'<div style="font-size:8px;color:#546e7a;text-transform:uppercase;'
        f'letter-spacing:1px;font-weight:700;margin-bottom:8px;">Enter on IG as:</div>'
        f'<div style="display:flex;justify-content:space-between;padding:3px 0;">'
        f'<span style="font-size:10px;color:#8a9ab5;">Entry</span>'
        f'<span style="font-size:11px;font-weight:800;color:#f5c518;">${_btf_ent:,.0f}</span></div>'
        f'<div style="display:flex;justify-content:space-between;padding:3px 0;">'
        f'<span style="font-size:10px;color:#8a9ab5;">Stop loss</span>'
        f'<span style="font-size:11px;font-weight:800;color:{_sl_col};">${_btf_sl:,.0f}</span></div>'
        f'<div style="display:flex;justify-content:space-between;padding:3px 0;">'
        f'<span style="font-size:10px;color:#8a9ab5;">Target</span>'
        f'<span style="font-size:11px;font-weight:800;color:{_tp_col};">${_btf_tp:,.0f}</span></div>'
        f'<div style="border-top:1px solid {_dir_col}22;margin-top:6px;padding-top:6px;'
        f'display:flex;justify-content:space-between;">'
        f'<span style="font-size:9px;color:#6a7a94;">R:R &nbsp;'
        f'<b style="color:{"#4caf50" if _btf_rr>=1.5 else "#ffa726"};">1:{_btf_rr:.1f}</b></span>'
        f'<span style="font-size:9px;color:#4caf50;">+A${_pnl_tp:,.0f}</span>'
        f'<span style="font-size:9px;color:#ef5350;">−A${_pnl_sl:,.0f}</span>'
        f'</div>'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

# ── MTF Confluence Meter ───────────────────────────────────────────────────────
_all_labels = [_15m_label, _30m_label, _1h_label, _4h_label, _day_label, _wk_label]
_bull_n  = sum(1 for l in _all_labels if l in ("BUY", "STRONG BUY", "LEAN BUY"))
_bear_n  = sum(1 for l in _all_labels if l in ("SELL", "STRONG SELL", "LEAN SELL"))
_neut_n  = 6 - _bull_n - _bear_n
_strong_bull = sum(1 for l in _all_labels if l == "STRONG BUY")
_strong_bear = sum(1 for l in _all_labels if l == "STRONG SELL")
_conf_net = _bull_n - _bear_n   # -6 to +6
if _conf_net > 0:
    _conf_verdict = f"{'STRONGLY ' if _strong_bull >= 2 else ''}BULLISH"
    _conf_col     = "#00e676"
    _conf_bg      = "#011a0a"
elif _conf_net < 0:
    _conf_verdict = f"{'STRONGLY ' if _strong_bear >= 2 else ''}BEARISH"
    _conf_col     = "#ff3d57"
    _conf_bg      = "#1a0005"
else:
    _conf_verdict = "MIXED"
    _conf_col     = "#546e7a"
    _conf_bg      = "#0a0d10"
_bull_bar_w = int(_bull_n / 6 * 100)
_bear_bar_w = int(_bear_n / 6 * 100)
_neut_bar_w = 100 - _bull_bar_w - _bear_bar_w
_conf_cell = ""
for _cl, _cw, _cc in [
    (f"↑ {_bull_n} Bull", _bull_bar_w, "#00e676"),
    (f"— {_neut_n} Neut", _neut_bar_w, "#37474f"),
    (f"↓ {_bear_n} Bear", _bear_bar_w, "#ff3d57"),
]:
    if _cw > 0:
        _conf_cell += (
            f'<div style="flex:{_cw};background:{_cc}22;text-align:center;'
            f'font-size:9px;color:{_cc};font-weight:700;padding:4px 2px;">{_cl}</div>'
        )
st.markdown(
    f'<div style="background:{_conf_bg};border:1px solid {_conf_col}44;'
    f'border-radius:10px;padding:12px 16px;margin-bottom:12px;">'
    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
    f'<div style="font-size:9px;color:#6a7a94;text-transform:uppercase;'
    f'letter-spacing:2px;font-weight:700;">MTF Confluence · All 6 Timeframes</div>'
    f'<div style="font-size:13px;font-weight:900;color:{_conf_col};'
    f'letter-spacing:0.5px;">{_conf_verdict}</div>'
    f'</div>'
    f'<div style="display:flex;border-radius:5px;overflow:hidden;gap:2px;">'
    f'{_conf_cell}'
    f'</div>'
    f'<div style="font-size:8px;color:#37474f;margin-top:6px;">'
    f'{_strong_bull} STRONG BUY · {_strong_bear} STRONG SELL &nbsp;·&nbsp; '
    f'When 5–6 TFs agree: highest-conviction entries</div>'
    f'</div>',
    unsafe_allow_html=True
)

# ── Custom selector row ────────────────────────────────────────────────────────
_sel_hint = (
    "⭐ Recommended timeframe pre-selected — change only if you prefer a different hold time"
    if _action_grade != "WAIT" else
    "No clear signal — select a timeframe to see current conditions"
)
st.markdown(
    f'<div style="font-size:9px;color:#546e7a;margin-bottom:6px;">{_sel_hint}</div>',
    unsafe_allow_html=True
)
_sel_tf = st.radio(
    "tf",
    _tf_keys,
    horizontal=True,
    key="signals_tf_pick",
    label_visibility="collapsed",
)
_tf_title, _tf_label, _tf_sig, _tf_hold = _TF_MAP[_sel_tf]

# ── Hold-time hint ─────────────────────────────────────────────────────────────
_is_best = (_sel_tf == _best_tf_key and _best_tf_score > 0)
_hint_extra = ' &nbsp;<span style="color:#f5a623;font-weight:800;">⭐ Best trade right now</span>' if _is_best else ""
st.markdown(
    f'<div style="font-size:10px;color:#546e7a;margin-bottom:14px;">'
    f'Typical hold: <b style="color:#78909c">{_tf_hold}</b>{_hint_extra}</div>',
    unsafe_allow_html=True
)

# ── Single full-width signal card ──────────────────────────────────────────────
_active_card = _signal_card(_tf_label, _tf_sig, _tf_title, _card_price)
st.markdown(
    f'<div style="display:flex;margin-bottom:16px;">{_active_card}</div>',
    unsafe_allow_html=True
)

# ── Send Trade to Telegram ─────────────────────────────────────────────────────
_tg_label_is_trade = _tf_label in ("BUY", "STRONG BUY", "LEAN BUY",
                                    "SELL", "STRONG SELL", "LEAN SELL")
if _tg_label_is_trade:
    _tg_is_long  = _tf_label in ("BUY", "STRONG BUY", "LEAN BUY")
    _tg_dir_word = "LONG (BUY)"  if _tg_is_long else "SHORT (SELL)"
    _tg_arrow    = "▲"           if _tg_is_long else "▼"
    _tg_sig      = _tf_sig or {}
    _tg_atr      = float(_tg_sig.get("atr",      _card_price * 0.008))
    _tg_ent      = _card_price
    _tg_sl       = float(_tg_sig.get("stop_loss", _tg_ent - _tg_atr * 1.5 if _tg_is_long else _tg_ent + _tg_atr * 1.5))
    _tg_tp       = float(_tg_sig.get("target",    _tg_ent + _tg_atr * 2.5 if _tg_is_long else _tg_ent - _tg_atr * 2.5))
    _tg_sl_dist  = abs(_tg_ent - _tg_sl)
    _tg_tp_dist  = abs(_tg_tp  - _tg_ent)
    _tg_rr       = _tg_tp_dist / max(_tg_sl_dist, 0.01)
    _tg_conf     = int(_tg_sig.get("confidence", 0) * 100)
    _tg_rsi      = _tg_sig.get("rsi")
    _tg_pnl_tp   = _tg_tp_dist * _IG_POINT_AUD_C
    _tg_pnl_sl   = _tg_sl_dist * _IG_POINT_AUD_C
    _tg_spread   = _IG_SPREAD_C / 2
    _tg_deal_px  = _tg_ent + _tg_spread if _tg_is_long else _tg_ent - _tg_spread
    _tg_direction_ig = "BUY" if _tg_is_long else "SELL"

    def _build_tg_message():
        _rsi_str   = f"RSI {_tg_rsi:.0f}" if _tg_rsi else ""
        _dir_emoji = "🟢" if _tg_is_long else "🔴"
        _time_utc  = __import__("datetime").datetime.utcnow().strftime("%H:%M UTC")
        return (
            f"{_dir_emoji} <b>NEW TRADE — {_tf_title}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Signal: <b>{_tf_label}</b>  |  Confidence: <b>{_tg_conf}%</b>  |  {_time_utc}\n\n"

            f"📋 <b>IG DEAL TICKET — fill in exactly:</b>\n\n"
            f"  <b>Step 1.</b>  Open IG app → tap <b>Trade</b>\n"
            f"  <b>Step 2.</b>  Search: <b>Gold Spot</b> (or XAU/USD)\n"
            f"  <b>Step 3.</b>  Direction: tap <b>{_tg_direction_ig}</b>\n"
            f"  <b>Step 4.</b>  Size: <b>1</b>  (A$10 Spot Gold contract)\n"
            f"  <b>Step 5.</b>  Stop type: <b>Normal</b>\n"
            f"  <b>Step 6.</b>  Stop level: <b>{_tg_sl:,.0f}</b>"
            f"  ← type this  (−${_tg_sl_dist:.0f} pts · risk A${_tg_pnl_sl:,.0f})\n"
            f"  <b>Step 7.</b>  Limit level: <b>{_tg_tp:,.0f}</b>"
            f"  ← type this  (+${_tg_tp_dist:.0f} pts · profit A${_tg_pnl_tp:,.0f})\n"
            f"  <b>Step 8.</b>  Deal price: ~<b>{_tg_deal_px:,.0f}</b>  (current {'ask' if _tg_is_long else 'bid'})\n"
            f"  <b>Step 9.</b>  Tap <b>Place trade</b> — confirm on next screen\n\n"

            f"📊 <b>Trade summary:</b>\n"
            f"  • Timeframe: {_tf_title}  |  Hold: {_tf_hold}\n"
            f"  • R:R: 1 : {_tg_rr:.1f}"
            + (f"  |  {_rsi_str}" if _rsi_str else "") + "\n\n"

            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"⚠️ <b>If you receive an AMENDMENT message after this:</b>\n"
            f"  🔄 <i>AMEND STOP</i> → IG Positions tab → Edit → update <b>Stop</b> field\n"
            f"  ⛔ <i>CLOSE TRADE</i> → IG Positions tab → tap your trade → <b>Close</b>\n"
            f"  🔁 <i>FLIP DIRECTION</i> → close this {_tg_direction_ig} first, then open opposite\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Signal only. Not financial advice."
        )

    if st.button(
        f"📱 Send {_tg_direction_ig} trade instructions to Telegram",
        type="primary",
        use_container_width=True,
        key="tg_send_btn",
    ):
        try:
            from telegram_alerts import send_message as _tg_send
            _msg_ok = _tg_send(_build_tg_message())
            if _msg_ok:
                st.success("✅ Trade instructions sent to Telegram!")
            else:
                st.error("❌ Send failed — check your TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID secrets.")
        except Exception as _tg_err:
            st.error(f"❌ Error: {_tg_err}")
else:
    st.markdown(
        '<div style="font-size:10px;color:#455a64;margin-bottom:8px;padding:6px 0;">'
        '📵 No active signal on this timeframe — switch to one with a BUY or SELL to send trade instructions.</div>',
        unsafe_allow_html=True
    )

# ── Position Size Calculator ───────────────────────────────────────────────────
with st.expander("🧮 Position Size Calculator", expanded=False):
    _ps_col1, _ps_col2 = st.columns([1, 1])
    with _ps_col1:
        _ps_balance = st.number_input(
            "Account balance (A$)", min_value=100.0, max_value=500000.0,
            value=10000.0, step=500.0, format="%.0f", key="ps_balance"
        )
        _ps_risk_pct = st.select_slider(
            "Risk per trade (%)", options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            value=1.0, key="ps_risk"
        )
    with _ps_col2:
        _ps_sig_for_calc = _tf_sig
        _ps_entry = _card_price
        _ps_atr   = float((_ps_sig_for_calc or {}).get("atr", _ps_entry * 0.008))
        _ps_sl    = float((_ps_sig_for_calc or {}).get("stop_loss", _ps_entry - _ps_atr * 1.5))
        _ps_tp    = float((_ps_sig_for_calc or {}).get("target",    _ps_entry + _ps_atr * 2.5))
        _ps_sl_pts = abs(_ps_entry - _ps_sl)
        _ps_tp_pts = abs(_ps_tp   - _ps_entry)
        _ps_dollar_per_pt = _IG_POINT_AUD_C       # A$10 per point per contract
        _ps_risk_dollars  = _ps_balance * (_ps_risk_pct / 100)
        _ps_contracts = _ps_risk_dollars / max(_ps_sl_pts * _ps_dollar_per_pt, 0.01)
        _ps_contracts_whole = max(1, round(_ps_contracts, 1))
        _ps_actual_risk   = _ps_contracts_whole * _ps_sl_pts * _ps_dollar_per_pt
        _ps_potential_pnl = _ps_contracts_whole * _ps_tp_pts * _ps_dollar_per_pt
        _ps_margin = _ps_contracts_whole * _ps_entry * _ps_dollar_per_pt * _IG_MARGIN_PCT_C
        _ps_rr = _ps_tp_pts / max(_ps_sl_pts, 0.01)
        st.markdown(
            f'<div style="background:#0d1117;border:1px solid #1e2a3a;border-radius:10px;'
            f'padding:12px 14px;font-size:10px;">'
            f'<div style="color:#6a7a94;font-size:9px;text-transform:uppercase;'
            f'letter-spacing:1px;font-weight:700;margin-bottom:8px;">Recommended Position</div>'
            f'<div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #ffffff07;">'
            f'<span style="color:#8a9ab5;">Contracts to trade</span>'
            f'<span style="color:#f5c518;font-weight:800;font-size:13px;">'
            f'{_ps_contracts_whole}</span></div>'
            f'<div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #ffffff07;">'
            f'<span style="color:#8a9ab5;">Risk at {_ps_risk_pct}%</span>'
            f'<span style="color:#ef5350;font-weight:700;">−A${_ps_actual_risk:,.0f}</span></div>'
            f'<div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #ffffff07;">'
            f'<span style="color:#8a9ab5;">If target hit</span>'
            f'<span style="color:#4caf50;font-weight:700;">+A${_ps_potential_pnl:,.0f}</span></div>'
            f'<div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #ffffff07;">'
            f'<span style="color:#8a9ab5;">R:R</span>'
            f'<span style="color:{"#4caf50" if _ps_rr >= 1.5 else "#ffa726"};font-weight:700;">'
            f'1 : {_ps_rr:.1f}</span></div>'
            f'<div style="display:flex;justify-content:space-between;padding:3px 0;">'
            f'<span style="color:#8a9ab5;">Margin needed</span>'
            f'<span style="color:#78909c;font-weight:700;">A${_ps_margin:,.0f}</span></div>'
            f'</div>',
            unsafe_allow_html=True
        )

# ── Quick overview strip: all timeframes at a glance ──────────────────────────
_ov_items = [
    ("15m",  _15m_label, "⚡  15 Min"),
    ("30m",  _30m_label, "⏱  30 Min"),
    ("1H",   _1h_label,  "🕐  1 Hour"),
    ("4H",   _4h_label,  "📊  4 Hours"),
    ("Day",  _day_label, "📅  Daily"),
    ("Week", _wk_label,  "📆  Weekly"),
]
_ov_cells = ""
for _ov_tf, _ov_lbl, _ov_key in _ov_items:
    _ov_is_best = (_ov_key == _best_tf_key and _best_tf_score > 0)
    if _ov_lbl in ("BUY", "STRONG BUY"):
        _ov_c, _ov_bg2 = "#00e676", "#011a0a"
    elif _ov_lbl in ("SELL", "STRONG SELL"):
        _ov_c, _ov_bg2 = "#ff3d57", "#1a0005"
    elif _ov_lbl == "LEAN BUY":
        _ov_c, _ov_bg2 = "#4caf50", "#050f07"
    elif _ov_lbl == "LEAN SELL":
        _ov_c, _ov_bg2 = "#ef5350", "#0f0506"
    else:
        _ov_c, _ov_bg2 = "#546e7a", "#0a0d10"
    _ov_border = f"border:2px solid {_ov_c};" if _ov_is_best else "border:1px solid transparent;"
    _ov_star   = '<div style="font-size:8px;color:#f5a623;font-weight:800;letter-spacing:0px;">⭐ BEST</div>' if _ov_is_best else ""
    _ov_cells += (
        f'<div style="flex:1;text-align:center;background:{_ov_bg2};'
        f'border-radius:8px;padding:8px 4px;{_ov_border}">'
        f'{_ov_star}'
        f'<div style="font-size:9px;color:#6a7a94;font-weight:700;'
        f'letter-spacing:0.5px;">{_ov_tf}</div>'
        f'<div style="font-size:10px;font-weight:800;color:{_ov_c};'
        f'margin-top:3px;white-space:nowrap;">{_ov_lbl}</div>'
        f'</div>'
    )
st.markdown(
    f'<div style="display:flex;gap:6px;margin-bottom:16px;">{_ov_cells}</div>',
    unsafe_allow_html=True
)

# ── Trade monitoring panel ─────────────────────────────────────────────────────
import json as _ms_j, datetime as _ms_dt
from pathlib import Path as _ms_p

_ms_cfg = {}
_ms_cfg_file = _ms_p("data_cache/alert_config.json")
if _ms_cfg_file.exists():
    try: _ms_cfg = _ms_j.loads(_ms_cfg_file.read_text())
    except Exception: pass

_ms_enabled  = _ms_cfg.get("enabled", False)
_ms_stop     = _ms_cfg.get("stop_level")
_ms_limit    = _ms_cfg.get("limit_level")
_ms_entry    = _ms_cfg.get("entry_price")
_ms_dir      = _ms_cfg.get("position_direction")
_ms_tg_on    = _ms_cfg.get("enabled", False)
_ms_sig_on   = _ms_cfg.get("alert_signal_change", True)
_ms_price_on = _ms_cfg.get("alert_price_levels", True)

# Scheduler last-run
_ms_prog = {}
_ms_prog_file = _ms_p("data_cache/scheduler_progress.json")
if _ms_prog_file.exists():
    try: _ms_prog = _ms_j.loads(_ms_prog_file.read_text())
    except Exception: pass
_ms_ts_raw = _ms_prog.get("last_update", "")
_ms_ts_str = "—"
if _ms_ts_raw:
    try:
        _ms_ago = int((_ms_dt.datetime.utcnow() -
                       _ms_dt.datetime.fromisoformat(str(_ms_ts_raw).replace("Z","")
                       )).total_seconds() // 60)
        _ms_ts_str = f"{_ms_ago} min ago"
    except Exception:
        _ms_ts_str = str(_ms_ts_raw)[:16]

# Build position row
if _ms_dir and _ms_entry:
    _ms_pcol   = "#00e676" if _ms_dir == "LONG" else "#ff3d57"
    _ms_sstr   = f"${_ms_stop:,.0f}" if _ms_stop else "not set"
    _ms_lstr   = f"${_ms_limit:,.0f}" if _ms_limit else "not set"
    _ms_pnl_s  = ""
    _ms_pnl_l  = ""
    if _ms_stop and _ms_entry:
        _ms_pnl_s = f"  (−A${abs(_ms_stop - _ms_entry) * 10:.0f} if hit)"
    if _ms_limit and _ms_entry:
        _ms_pnl_l = f"  (+A${abs(_ms_limit - _ms_entry) * 10:.0f} if hit)"
    _ms_pos_html = f"""
<div style="background:#0a0e14;border:1px solid {_ms_pcol}33;border-radius:10px;
     padding:12px 16px;margin-top:10px;">
  <div style="font-size:9px;color:#6a7a94;text-transform:uppercase;letter-spacing:1px;
       font-weight:700;margin-bottom:8px;">Open Position Being Monitored</div>
  <div style="display:flex;gap:20px;flex-wrap:wrap;align-items:baseline;">
    <div><span style="font-size:13px;font-weight:900;color:{_ms_pcol};">{_ms_dir}</span></div>
    <div style="font-size:11px;color:#8a9ab5;">Entry&nbsp;<b style="color:#e0e0e0;">${_ms_entry:,.0f}</b></div>
    <div style="font-size:11px;color:#8a9ab5;">Stop&nbsp;<b style="color:#ef5350;">{_ms_sstr}</b><span style="color:#546e7a;font-size:10px;">{_ms_pnl_s}</span></div>
    <div style="font-size:11px;color:#8a9ab5;">Target&nbsp;<b style="color:#4caf50;">{_ms_lstr}</b><span style="color:#546e7a;font-size:10px;">{_ms_pnl_l}</span></div>
  </div>
  <div style="font-size:10px;color:#455a64;margin-top:8px;">
    Telegram fires when price is within 5 pts of stop or target · revision alerts fire if signal flips
  </div>
</div>"""
else:
    _ms_pos_html = """
<div style="background:#0a0e14;border:1px solid #1e2a38;border-radius:10px;
     padding:12px 16px;margin-top:10px;">
  <div style="font-size:10px;color:#546e7a;">
    <b style="color:#78909c;">No position tracked.</b>
    Open the <b style="color:#8a9ab5;">Tools → Alerts</b> tab, set your entry price,
    stop loss and take profit — the app will then monitor the trade and send Telegram
    messages if it needs to be adjusted.
  </div>
</div>"""

_ms_tg_col = "#4caf50" if _ms_tg_on else "#ef5350"
_ms_tg_txt = "ON" if _ms_tg_on else "OFF"
st.markdown(f"""
<div style="background:#0d1117;border:1px solid #1e2a38;border-radius:12px;
     padding:16px 20px;margin-bottom:8px;">
  <div style="display:flex;align-items:center;justify-content:space-between;
       flex-wrap:wrap;gap:10px;margin-bottom:2px;">
    <div style="font-size:11px;font-weight:700;color:#e0e0e0;display:flex;
         align-items:center;gap:6px;">
      <span style="font-size:8px;color:{_ms_tg_col};">⬤</span>
      TRADE MONITORING &amp; ALERTS
    </div>
    <div style="display:flex;gap:16px;flex-wrap:wrap;">
      <span style="font-size:10px;color:#6a7a94;">
        Telegram: <b style="color:{_ms_tg_col};">{_ms_tg_txt}</b>
      </span>
      <span style="font-size:10px;color:#6a7a94;">
        Entry alerts: <b style="color:#8a9ab5;">{'ON' if _ms_sig_on else 'OFF'}</b>
      </span>
      <span style="font-size:10px;color:#6a7a94;">
        Price alerts: <b style="color:#8a9ab5;">{'ON' if _ms_price_on else 'OFF'}</b>
      </span>
      <span style="font-size:10px;color:#6a7a94;">
        Last check: <b style="color:#8a9ab5;">{_ms_ts_str}</b>
      </span>
    </div>
  </div>
  {_ms_pos_html}
</div>
""", unsafe_allow_html=True)

_tab_signals.__exit__(None, None, None)
_tab_analysis.__enter__()
_sub_pred, _sub_dt = st.tabs(["🔮  Predictions", "📈  Day Trading"])
_sub_pred.__enter__()

# ── Forecast vs Signal explainer ─────────────────────────────────────────
# This sits right at the top of the Predictions tab so the user immediately
# understands the relationship between the two systems before reading any charts.
try:
    import json as _pe_j
    _pe_mhf = Path("data_cache/multi_horizon_predictions.json")
    _pe_sday = _day_sig   # already computed above
    _pe_mhp  = _pe_j.loads(_pe_mhf.read_text()) if _pe_mhf.exists() else {}
    _pe_probs = {h: _pe_mhp.get(str(h), {}).get("raw_proba", 0.5) for h in [1, 2, 5]}
    _pe_action = _pe_sday.get("action", "—") if _pe_sday else "—"
    _pe_conf   = _pe_sday.get("confidence", 0.0) if _pe_sday else 0.0

    def _pe_dir(p):
        return ("▲ UP", "#4caf50") if p > 0.55 else (("▼ DOWN", "#ef5350") if p < 0.45 else ("— FLAT", "#888"))

    _pe_rows = ""
    _pe_names = {1: "Tomorrow", 2: "2 days", 5: "5 days"}
    for _h, _p in _pe_probs.items():
        _d, _c = _pe_dir(_p)
        _pe_rows += (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:5px 0;border-bottom:1px solid #1a1a1a;">'
            f'<span style="font-size:12px;color:#888;">{_pe_names[_h]}</span>'
            f'<span style="font-size:12px;font-weight:700;color:{_c};">{_d}</span>'
            f'<span style="font-size:11px;color:#8a9ab5;">P(DOWN)={(1-_p):.0%}</span>'
            f'</div>'
        )

    # ── Synthesise all signals into ONE verdict ───────────────────────────
    _syn_1h_bull  = _1h_label  in {"BUY", "STRONG BUY"}
    _syn_1h_bear  = _1h_label  in {"SELL", "STRONG SELL"}
    _syn_day_bull = _day_label in {"BUY", "STRONG BUY"}
    _syn_day_bear = _day_label in {"SELL", "STRONG SELL"}
    _syn_ml_bull  = _pe_probs.get(1, 0.5) > 0.55
    _syn_ml_bear  = _pe_probs.get(1, 0.5) < 0.45

    _syn_bulls = sum([_syn_1h_bull, _syn_day_bull, _syn_ml_bull])
    _syn_bears = sum([_syn_1h_bear, _syn_day_bear, _syn_ml_bear])

    if _syn_bulls > _syn_bears:
        _syn_verdict = "BUY"
        _syn_arrow   = "▲"
        _syn_col     = "#00e676"
        _syn_bg      = "#001408"
        _syn_border  = "#00e67688"
        _syn_plain   = (
            "The weight of all signals points <b>UP</b>. "
            "Both short-term momentum and macro factors are bullish. "
            "Look for an entry on a dip toward support."
        )
    elif _syn_bears > _syn_bulls:
        _syn_verdict = "SELL"
        _syn_arrow   = "▼"
        _syn_col     = "#ef5350"
        _syn_bg      = "#140204"
        _syn_border  = "#ef535088"
        _syn_plain   = (
            "The weight of all signals points <b>DOWN</b>. "
            "Even if there is a short-term bounce intraday, the bigger picture "
            "is bearish — avoid new long positions and look for short entries on rallies."
        )
    else:
        _syn_verdict = "WAIT"
        _syn_arrow   = "—"
        _syn_col     = "#f9a825"
        _syn_bg      = "#131000"
        _syn_border  = "#f9a82588"
        _syn_plain   = (
            "Signals are <b>split</b> — short-term and macro are pointing in opposite "
            "directions. No clear edge right now. Wait for the models to align "
            "before opening a new position."
        )

    def _syn_row(lbl, timeframe, is_bull, is_bear):
        if is_bull:
            ic, txt = "#00e676", "▲ BUY / UP"
        elif is_bear:
            ic, txt = "#ef5350", "▼ SELL / DOWN"
        else:
            ic, txt = "#6a7a94", "— NEUTRAL"
        return (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:7px 0;border-bottom:1px solid #1e2840;">'
            f'<span style="font-size:12px;color:#c0cfe0;font-weight:600;">{lbl}</span>'
            f'<span style="font-size:10px;color:#6a7a94;padding:0 12px;">{timeframe}</span>'
            f'<span style="font-size:13px;font-weight:800;color:{ic};">{txt}</span>'
            f'</div>'
        )

    _syn_rows = (
        _syn_row("⚡ 1-Hour Technical Signal", "next 1–2 hrs",    _syn_1h_bull,  _syn_1h_bear)  +
        _syn_row("📅 Daily Technical Signal",  "today's session", _syn_day_bull, _syn_day_bear) +
        _syn_row("🤖 ML Macro Model",          "1-day outlook",   _syn_ml_bull,  _syn_ml_bear)
    )

    _syn_score_txt = f"{_syn_bulls}/3 bullish · {_syn_bears}/3 bearish"

    st.markdown(f"""
<div style="background:{_syn_bg};border:2px solid {_syn_border};border-radius:18px;
    padding:22px 26px;margin-bottom:26px;position:relative;overflow:hidden;">
  <div style="position:absolute;top:0;left:0;right:0;height:4px;background:{_syn_col};
      opacity:0.75;border-radius:18px 18px 0 0;"></div>
  <div style="font-size:10px;color:{_syn_col}99;text-transform:uppercase;
      letter-spacing:2px;font-weight:700;margin-bottom:8px;">
    Combined Signal — All Models · {_syn_score_txt}
  </div>
  <div style="display:flex;align-items:center;gap:20px;margin-bottom:18px;flex-wrap:wrap;">
    <div style="font-size:54px;font-weight:900;color:{_syn_col};font-family:monospace;
        line-height:1;letter-spacing:2px;min-width:160px;">
      {_syn_arrow}&nbsp;{_syn_verdict}
    </div>
    <div style="font-size:13px;color:#b0bdd0;line-height:1.7;max-width:420px;">
      {_syn_plain}
    </div>
  </div>
  <div style="background:#080c14;border-radius:10px;padding:10px 14px;">
    {_syn_rows}
    <div style="padding-top:10px;font-size:10px;color:#6a7a94;font-style:italic;">
      ↓ The charts below show the supporting evidence for each signal above.
      A short-term bounce can coexist with a bearish daily trend — they are
      answering different questions, not contradicting each other.
    </div>
  </div>
</div>""", unsafe_allow_html=True)
except Exception:
    pass

# ─────────────────────────────────────────────
# MULTI-TIMEFRAME FORECAST — 3 STATIC CHARTS
# ─────────────────────────────────────────────
st.subheader("📅 Price Forecast — 1 Hour · 1 Day · 1 Week")

if gold_2y is not None and len(gold_2y) >= 5:
    import matplotlib.pyplot as _fc_plt
    import matplotlib.dates as _fc_dates
    import matplotlib.ticker as _fc_ticker

    _BG      = "#111827"
    _GRID    = "#1e1e1e"
    _GOLD    = "#f5c518"
    _C_UP    = "#4caf50"
    _C_DN    = "#ef5350"
    _C_FLAT  = "#90a4ae"

    _cur  = float(gold_2y.iloc[-1])
    _now  = pd.Timestamp.utcnow().tz_localize(None)

    # Per-timeframe historical volatility
    _rets_d = gold_2y.pct_change().dropna()
    _vol_1d = float(_rets_d.tail(30).std())
    _vol_5d = (float(_rets_d.tail(30).rolling(5).std().dropna().iloc[-1])
               if len(_rets_d) >= 35 else _vol_1d * 2.2)
    _vol_1h = _vol_1d / (24 ** 0.5)

    # ── 1-hour signal (intraday model) ──────────────────────────────────────
    _ipreds_1h = [p for p in load_intraday_preds()
                  if p.get("horizon_label") == "1 hour"]
    _1h_latest = _ipreds_1h[-1] if _ipreds_1h else None
    if _1h_latest:
        # predicted_direction is binary (0=DOWN, 1=UP) — convert to signed ±1
        _1h_dir_raw = int(_1h_latest.get("predicted_direction", -1))
        _1h_dir     = 1 if _1h_dir_raw == 1 else -1 if _1h_dir_raw == 0 else 0
        _1h_price   = float(_1h_latest.get("predicted_price", _cur))
        _1h_conf    = min(max(abs(_1h_price/_cur - 1) / max(_vol_1h, 1e-6), 0.3), 1.0)
    else:
        _1h_dir, _1h_price, _1h_conf = 0, _cur, 0.5

    # ── 1-day signal (ensemble model) ──────────────────────────────────────
    # Model returns binary direction: 0=DOWN, 1=UP.
    # Must convert to signed ±1 before using as a price multiplier.
    # Without this: direction=0 (DOWN) gives price * (1 + ... * 0) = flat — no forecast shown.
    _mh_1d    = mh_preds.get("1", {})
    _1d_raw   = _mh_1d.get("direction")
    _1d_dir   = (1 if _1d_raw == 1 else -1) if _1d_raw is not None else 0
    _1d_conf  = float(_mh_1d.get("confidence", 0.5) or 0.5)
    _1d_price = _cur * (1 + _vol_1d * (0.4 + 0.6 * _1d_conf) * _1d_dir)

    # ── 1-week signal (ensemble model) ─────────────────────────────────────
    _mh_5d    = mh_preds.get("5", {})
    _5d_raw   = _mh_5d.get("direction")
    _5d_dir   = (1 if _5d_raw == 1 else -1) if _5d_raw is not None else 0
    _5d_conf  = float(_mh_5d.get("confidence", 0.5) or 0.5)
    _5d_price = _cur * (1 + _vol_5d * (0.4 + 0.6 * _5d_conf) * _5d_dir)

    def _sig_color(d): return _C_UP if d > 0 else (_C_DN if d < 0 else _C_FLAT)
    def _sig_label(d): return "▲  UP" if d > 0 else ("▼  DOWN" if d < 0 else "◆  MIXED")

    def _draw_forecast(ax, hist_x, hist_y, target_x, target_y, conf, sig_dir,
                       x_fmt, title):
        """Render one forecast panel onto ax."""
        color = _sig_color(sig_dir)
        band  = abs(target_y - hist_y[-1]) * (1.4 - 0.9 * conf) + _cur * _vol_1d * 0.3

        ax.set_facecolor(_BG)

        # ── Historical line ────────────────────────────────────────────────
        ax.plot(hist_x, hist_y, color=_GOLD, linewidth=2.2, solid_capstyle="round", zorder=3)
        ax.plot(hist_x[-1], hist_y[-1], "o", color=_GOLD, markersize=6, zorder=4)

        # Vertical "Now" separator
        ax.axvline(hist_x[-1], color="#3a4a60", linewidth=1, linestyle="--", zorder=2)

        # ── Confidence band ────────────────────────────────────────────────
        ax.fill_between([hist_x[-1], target_x],
                        [hist_y[-1], target_y - band],
                        [hist_y[-1], target_y + band],
                        color=color, alpha=0.13, zorder=1)

        # ── Forecast line ──────────────────────────────────────────────────
        ax.plot([hist_x[-1], target_x], [hist_y[-1], target_y],
                color=color, linewidth=2.2, linestyle="--",
                solid_capstyle="round", dash_capstyle="round", zorder=3)
        ax.plot(target_x, target_y, "D", color=color,
                markersize=10, markeredgecolor="#111827",
                markeredgewidth=1.5, zorder=5)

        # ── Price labels ───────────────────────────────────────────────────
        pct = (target_y / hist_y[-1] - 1) * 100
        ax.annotate(
            f"  ${target_y:,.0f}\n  {pct:+.2f}%",
            xy=(target_x, target_y),
            fontsize=9, color=color, fontweight="bold",
            va="center", ha="left",
        )

        # ── Signal badge (top-left of axes) ───────────────────────────────
        ax.text(0.03, 0.97, _sig_label(sig_dir),
                transform=ax.transAxes, ha="left", va="top",
                fontsize=16, fontweight="black", color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#111827",
                          edgecolor=color + "55", linewidth=1))
        ax.text(0.03, 0.72, f"conf  {conf:.0%}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8, color="#666")

        # ── Current price (bottom-left) ────────────────────────────────────
        ax.text(0.03, 0.05, f"now  ${hist_y[-1]:,.2f}",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8, color="#888")

        # ── Axes styling ───────────────────────────────────────────────────
        ax.set_title(title, color="#ccc", fontsize=13, fontweight="bold",
                     pad=10, loc="left")
        ax.tick_params(colors="#aaa", labelsize=8, length=3)
        for sp in ax.spines.values():
            sp.set_color("#3a4a60")
        ax.yaxis.set_major_formatter(
            _fc_ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax.xaxis.set_major_formatter(_fc_dates.DateFormatter(x_fmt))
        ax.xaxis.set_tick_params(rotation=30)
        ax.set_xlim(hist_x[0], target_x + (target_x - hist_x[-1]) * 0.5)
        ax.grid(True, color=_GRID, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)

    # ── Build 1-hour history from hourly bars ──────────────────────────────
    try:
        _hdf = _top_df[["Close"]].copy() if (_top_df is not None and "Close" in _top_df.columns) else None
        if _hdf is None and _top_df is not None:
            _hdf = _top_df.iloc[:, :1].copy()
            _hdf.columns = ["Close"]
        if _hdf is not None:
            _hdf.index = pd.to_datetime(_hdf.index).tz_localize(None)
            _h1_x = list(_hdf.tail(10).index)
            _h1_y = list(_hdf.tail(10)["Close"].values.astype(float))
        else:
            raise ValueError("no hourly df")
    except Exception:
        _daily_idx = pd.to_datetime(gold_2y.tail(10).index).tz_localize(None)
        _h1_x = [_now - pd.Timedelta(hours=i) for i in range(9, -1, -1)]
        _h1_y = list(gold_2y.tail(10).values.astype(float))

    _t1h_x = _h1_x[-1] + pd.Timedelta(hours=1)

    # ── Build 1-day history from daily bars ────────────────────────────────
    _d7 = gold_2y.tail(7).copy()
    _d7.index = pd.to_datetime(_d7.index).tz_localize(None)
    _h1d_x  = list(_d7.index)
    _h1d_y  = list(_d7.values.astype(float))
    _t1d_x  = _h1d_x[-1] + pd.Timedelta(days=1)

    # ── Build 1-week history from daily bars ───────────────────────────────
    _d21 = gold_2y.tail(21).copy()
    _d21.index = pd.to_datetime(_d21.index).tz_localize(None)
    _h1w_x  = list(_d21.index)
    _h1w_y  = list(_d21.values.astype(float))
    _t1w_x  = _h1w_x[-1] + pd.Timedelta(days=5)

    # ── Render 3 columns ───────────────────────────────────────────────────
    _fc_col1, _fc_col2, _fc_col3 = st.columns(3, gap="medium")

    for _col, _hist_x, _hist_y, _t_x, _t_y, _conf, _dir, _fmt, _title, _acc_src in [
        (_fc_col1, _h1_x,  _h1_y,  _t1h_x, _1h_price, _1h_conf, _1h_dir, "%H:%M",
         "1 Hour", _top_iacc.get("1 hour", (None, 0))),
        (_fc_col2, _h1d_x, _h1d_y, _t1d_x, _1d_price, _1d_conf, _1d_dir, "%d %b",
         "1 Day",  (live_acc, live_n)),
        (_fc_col3, _h1w_x, _h1w_y, _t1w_x, _5d_price, _5d_conf, _5d_dir, "%d %b",
         "1 Week", (None, 0)),
    ]:
        _fig_s, _ax_s = _fc_plt.subplots(figsize=(4.8, 3.6))
        _fig_s.patch.set_facecolor(_BG)
        _draw_forecast(_ax_s, _hist_x, _hist_y, _t_x, _t_y, _conf, _dir, _fmt, _title)
        _fig_s.tight_layout(pad=0.8)
        _col.pyplot(_fig_s, width="stretch")
        _fc_plt.close(_fig_s)

        # Accuracy footnote
        _acc_v, _acc_n2 = _acc_src
        if _acc_v:
            _col.caption(f"Historical accuracy: **{_acc_v:.0%}** · {_acc_n2} resolved")
        else:
            _col.caption("Historical accuracy: building…")

else:
    st.info("Gold price data not yet loaded — the scheduler populates this automatically.")

st.divider()

# ─────────────────────────────────────────────
# PREDICTION vs REALITY CHART
# ─────────────────────────────────────────────
st.subheader("🔮 Prediction vs Reality")
st.caption(
    "Each dot shows whether that day's predicted direction matched the actual move. "
    "The lower panel tracks rolling 10-day accuracy."
)

_pvr_results = st.session_state.get("results")
if _pvr_results is not None and gold_2y is not None:
    import matplotlib.pyplot as _plt
    import matplotlib.dates as _mdates
    import matplotlib.patches as _mpatches
    from adaptive_learning import apply_regime_shift_penalty as _apply_regime

    _gold_s = gold_2y.copy()
    _gold_s.index = pd.to_datetime(_gold_s.index).tz_localize(None)
    _gold_s.name = "gold_price"
    _pvr = (
        pd.DataFrame({
            "predicted":  _pvr_results["predictions"],
            "actual_dir": _pvr_results["actual"],
            "ret":        _pvr_results["returns"],
        })
        .join(_gold_s, how="inner")
        .dropna()
        .tail(60)
    )

    if len(_pvr) >= 5:
        _pvr = _pvr.copy()
        _pvr["correct"] = (_pvr["predicted"] == _pvr["actual_dir"])
        _pvr["roll10"]  = _pvr["correct"].rolling(10, min_periods=5).mean()

        _correct_n  = int(_pvr["correct"].sum())
        _acc        = _correct_n / len(_pvr)
        _acc_color  = "#4caf50" if _acc >= 0.55 else ("#ef5350" if _acc < 0.45 else "#ffc107")

        # Regime bias: model UP-call rate vs actual UP rate (last 60 days)
        _pred_up_rate   = float((_pvr["predicted"] == 1).mean())
        _actual_up_rate = float((_pvr["actual_dir"] == 1).mean())
        _up_bias        = _pred_up_rate - _actual_up_rate

        # Auto-apply regime recalibration if bias > 10 %
        _regime_changes = _apply_regime(_up_bias, _acc)

        # ── Figure: 2 rows — price+markers top, rolling accuracy bottom ────────
        _fig, (_ax1, _ax2) = _plt.subplots(
            2, 1, figsize=(12, 4.8),
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        )
        for _ax in (_ax1, _ax2):
            _ax.set_facecolor("#111827")
            for _sp in _ax.spines.values():
                _sp.set_edgecolor("#222")
            _ax.grid(color="#1a1f2e", linewidth=0.6)
        _fig.patch.set_facecolor("#111827")

        # ── Top panel: price + correct/wrong dots + shaded day bands ───────────
        _ok  = _pvr[_pvr["correct"]]
        _bad = _pvr[~_pvr["correct"]]

        # Faint column shading for each wrong day
        for _d in _bad.index:
            _ax1.axvspan(_d - pd.Timedelta(hours=10), _d + pd.Timedelta(hours=10),
                         color="#ef5350", alpha=0.10, linewidth=0)

        _ax1.plot(_pvr.index, _pvr["gold_price"],
                  color="#f5a623", linewidth=2.0, zorder=2)
        _ax1.scatter(_ok.index,  _ok["gold_price"],
                     color="#4caf50", s=50, zorder=4, label="Correct call")
        _ax1.scatter(_bad.index, _bad["gold_price"],
                     color="#ef5350", s=50, zorder=4, marker="x",
                     linewidths=1.6, label="Wrong call")

        _ax1.yaxis.set_major_formatter(_plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        _ax1.tick_params(axis="both", colors="#999", labelsize=9)
        _ax1.tick_params(axis="x", labelbottom=False)
        _ax1.set_title(
            f"Last 60 trading days  ·  {_acc:.0%} accuracy  ·  "
            f"{_correct_n} correct / {len(_pvr)} calls  ·  "
            f"UP bias {_up_bias:+.0%}",
            color=_acc_color, fontsize=11, pad=8,
        )
        _lgd_ok  = _mpatches.Patch(color="#4caf50", label=f"Correct ({_correct_n})")
        _lgd_bad = _mpatches.Patch(color="#ef5350", label=f"Wrong ({len(_pvr)-_correct_n})")
        _ax1.legend(handles=[_lgd_ok, _lgd_bad],
                    loc="upper left", framealpha=0, fontsize=9, labelcolor="white")

        # ── Bottom panel: rolling 10-day accuracy ──────────────────────────────
        _ax2.axhline(0.5, color="#3a4a60", linewidth=0.8, linestyle="--")
        _ax2.axhline(0.55, color="#4caf5044", linewidth=0.8, linestyle=":")
        _ax2.fill_between(_pvr.index, 0.5, _pvr["roll10"],
                          where=(_pvr["roll10"] >= 0.5),
                          color="#4caf50", alpha=0.35, linewidth=0)
        _ax2.fill_between(_pvr.index, _pvr["roll10"], 0.5,
                          where=(_pvr["roll10"] < 0.5),
                          color="#ef5350", alpha=0.35, linewidth=0)
        _ax2.plot(_pvr.index, _pvr["roll10"],
                  color="#40e0d0", linewidth=1.5, label="10-day accuracy")
        _ax2.yaxis.set_major_formatter(_plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        _ax2.set_ylim(0.25, 0.85)
        _ax2.tick_params(axis="both", colors="#999", labelsize=8)
        _ax2.xaxis.set_major_formatter(_mdates.DateFormatter("%b %d"))
        _ax2.xaxis.set_major_locator(_mdates.WeekdayLocator(interval=2))
        _plt.setp(_ax2.xaxis.get_majorticklabels(), rotation=30, ha="right",
                  color="#999", fontsize=8)
        _ax2.set_ylabel("10-day acc.", color="#666", fontsize=8)

        _plt.tight_layout()
        st.pyplot(_fig, width="stretch")
        _plt.close(_fig)

        # Show regime recalibration notice if it fired
        if _regime_changes:
            _bias_dir = "over-bullish" if _up_bias > 0 else "over-bearish"
            st.info(
                f"⚖️ **Regime recalibration applied** — model was {_bias_dir} "
                f"({_up_bias:+.0%} UP-call bias over 60 days, {_acc:.0%} accuracy). "
                f"Adjusted {len(_regime_changes)} indicator weights to counteract bias: "
                + ", ".join(f"{k} {v['old']}→{v['new']}" for k, v in _regime_changes.items())
            )
    else:
        st.info("Not enough overlapping data to display the chart.")
else:
    st.info("Run a backtest to see the prediction vs reality chart — the scheduler will populate this automatically.")

_sub_pred.__exit__(None, None, None)
_sub_dt.__enter__()

# ─────────────────────────────────────────────
# DAY TRADING SECTION  — two timeframe tabs
# ─────────────────────────────────────────────
st.subheader("📈 Day Trading — Live Signals")
st.caption(
    "RSI + MACD + Bollinger Bands + EMA(9/21/50) + VWAP + ATR · "
    "RSI+MACD combined ~73% win rate on gold futures (235-trade backtest) · "
    "1.5× ATR stop-loss · 2.5× ATR take-profit · R:R = 1.67"
)


def _render_signal_panel(dt_df, dt_signal, chart_bars: int, refresh_note: str):
    """Render the full BUY/SELL panel for one timeframe."""
    if dt_signal is None:
        if dt_df is None:
            st.warning("⚠️ Could not load data from Yahoo Finance.")
        else:
            st.warning("⚠️ Not enough data to compute signals yet.")
        return

    _sc   = dt_signal["total_score"]
    _atr  = dt_signal["atr"]
    # Spot-adjust: GC=F futures → XAU/USD spot using live stooq price
    _fut_ent = dt_signal["entry"]
    _sadj    = (_top_live_price - _fut_ent) if _top_live_price else 0.0
    _ent  = _top_live_price if _top_live_price else _fut_ent
    _tgt  = dt_signal["target"]    + _sadj
    _stp  = dt_signal["stop_loss"] + _sadj

    if _sc > 0:
        _lbl, _bg, _fg, _icon = "BUY",  "#0d3320", "#4caf50", "▲"
    elif _sc < 0:
        _lbl, _bg, _fg, _icon = "SELL", "#3b0d0d", "#ef5350", "▼"
    else:
        _lbl, _bg, _fg, _icon = "HOLD", "#1a1f2e", "#90a4ae", "◆"

    # Big sign
    st.markdown(
        f"""
        <div style="background:{_bg};border:2px solid {_fg};border-radius:14px;
                    padding:18px 28px;margin-bottom:14px;text-align:center">
          <div style="font-size:48px;font-weight:900;color:{_fg};
                      letter-spacing:4px;line-height:1">{_icon} {_lbl}</div>
          <div style="font-size:13px;color:#aaa;margin-top:8px">
            Spot Gold (XAU/USD) &nbsp;·&nbsp;
            Confidence: <b style="color:{_fg}">{int(_sc):+d}</b> / {int(dt_signal["max_score"])}
            &nbsp;·&nbsp; As of <b>{dt_signal["timestamp"][11:16]} UTC</b>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Entry / TP / SL cards
    _gp = (_tgt - _ent) / _ent * 100
    _lp = (_stp - _ent) / _ent * 100
    st.markdown(
        f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:18px">
          <div style="background:#1a2235;border:1px solid #334155;border-radius:12px;
                      padding:18px 20px;text-align:center">
            <div style="font-size:11px;color:#aaa;text-transform:uppercase;
                        letter-spacing:1px;margin-bottom:6px">Entry Price</div>
            <div style="font-size:28px;font-weight:800;color:#e2e8f0">${_ent:,.2f}</div>
            <div style="font-size:11px;color:#9ba8bc;margin-top:4px">Current market price</div>
          </div>
          <div style="background:#0d3320;border:2px solid #4caf50;border-radius:12px;
                      padding:18px 20px;text-align:center">
            <div style="font-size:11px;color:#aaa;text-transform:uppercase;
                        letter-spacing:1px;margin-bottom:6px">Take-Profit</div>
            <div style="font-size:28px;font-weight:800;color:#4caf50">${_tgt:,.2f}</div>
            <div style="font-size:12px;color:#66bb6a;margin-top:4px">
              {_gp:+.2f}% &nbsp;·&nbsp; +${abs(_tgt-_ent):,.2f} &nbsp;·&nbsp; 2.5 × ATR
            </div>
            {"<div style='font-size:9px;color:#ff9800;margin-top:4px;'>⚠ High-ATR — consider partial at 50%</div>" if abs(_gp) > 3.0 else ""}
          </div>
          <div style="background:#3b0d0d;border:2px solid #ef5350;border-radius:12px;
                      padding:18px 20px;text-align:center">
            <div style="font-size:11px;color:#aaa;text-transform:uppercase;
                        letter-spacing:1px;margin-bottom:6px">Stop-Loss</div>
            <div style="font-size:28px;font-weight:800;color:#ef5350">${_stp:,.2f}</div>
            <div style="font-size:12px;color:#ff7043;margin-top:4px">
              {_lp:+.2f}% &nbsp;·&nbsp; -${abs(_stp-_ent):,.2f} &nbsp;·&nbsp; 1.5 × ATR
            </div>
          </div>
        </div>
        <div style="background:#1a2235;border:1px solid #334155;border-radius:10px;
                    padding:10px 18px;text-align:center;margin-bottom:14px;
                    font-size:13px;color:#94a3b8">
          R:R &nbsp;=&nbsp; <b style="color:#e2e8f0">1 : {dt_signal["risk_reward"]:.2f}</b>
          &nbsp;|&nbsp; ATR(14) &nbsp;=&nbsp; <b style="color:#e2e8f0">${_atr:,.2f}</b>
          &nbsp;|&nbsp; RSI &nbsp;=&nbsp; <b style="color:#e2e8f0">{dt_signal["rsi"]:.1f}</b>
          &nbsp;|&nbsp; VWAP &nbsp;=&nbsp; <b style="color:#e2e8f0">${dt_signal["vwap"]:,.2f}</b>
        </div>""",
        unsafe_allow_html=True,
    )

    # Indicator breakdown
    def _score_label(v):
        if v >= 1.8:   return "🟢 Strong Buy"
        if v >= 0.8:   return "🟢 Buy"
        if v <= -1.8:  return "🔴 Strong Sell"
        if v <= -0.8:  return "🔴 Sell"
        return "⚪ Neutral"

    # ── Macro Context panel ───────────────────────────────────────────────
    from day_trading import _fetch_macro_context
    _ctx = _fetch_macro_context()
    _gvz     = _ctx.get("gvz")
    _r10y    = _ctx.get("real10y")
    _dxy_tr  = _ctx.get("dxy_trend_pct")
    _month_now = pd.Timestamp.utcnow().month
    _season_label = (
        "🇮🇳 Diwali Season — elevated Indian demand" if _month_now in [10, 11]
        else "🪷 Akshaya Tritiya / Spring Demand" if _month_now in [4, 5]
        else "🎊 Chinese New Year / Wedding Season — elevated Asian demand" if _month_now in [1, 2]
        else "💒 Indian Wedding Season" if _month_now in [11, 12]
        else "☀️ Summer Lull — seasonally weak demand" if _month_now in [6, 7]
        else "📅 Neutral season"
    )
    _regime_val = dt_signal["indicators"].get("Regime")
    _regime_txt = (
        "🟢 Gold-Favorable (risk-off + macro tailwinds)" if _regime_val and _regime_val >= 1
        else "🔴 Gold-Headwinds (risk-on / strong dollar)" if _regime_val and _regime_val <= -1
        else "⚪ Neutral macro backdrop"
    )
    _macro_items = []
    if _gvz is not None:
        _gvz_interp = (
            "Extreme fear — potential reversal zone" if _gvz > 35
            else "Elevated fear / risk-off" if _gvz > 25
            else "Low vol — trending regime" if _gvz < 15
            else "Normal"
        )
        _gvz_color = "#ef5350" if _gvz > 35 else "#ff9800" if _gvz > 25 else "#4caf50"
        _macro_items.append(
            f"<div style='background:#1a2235;border:1px solid #334155;border-radius:10px;"
            f"padding:12px 16px;flex:1;min-width:160px'>"
            f"<div style='font-size:10px;color:#888;text-transform:uppercase;letter-spacing:1px;"
            f"margin-bottom:4px'>GVZ · Gold VIX</div>"
            f"<div style='font-size:22px;font-weight:800;color:{_gvz_color}'>{_gvz:.1f}</div>"
            f"<div style='font-size:11px;color:#aaa;margin-top:2px'>{_gvz_interp}</div></div>"
        )
    if _r10y is not None:
        _ry_color = "#4caf50" if _r10y < 0 else "#ef5350" if _r10y > 1.5 else "#90a4ae"
        _ry_interp = "Negative → gold tailwind" if _r10y < 0 else "High → gold headwind" if _r10y > 1.5 else "Moderate"
        _macro_items.append(
            f"<div style='background:#1a2235;border:1px solid #334155;border-radius:10px;"
            f"padding:12px 16px;flex:1;min-width:160px'>"
            f"<div style='font-size:10px;color:#888;text-transform:uppercase;letter-spacing:1px;"
            f"margin-bottom:4px'>10Y Real Yield (TIPS)</div>"
            f"<div style='font-size:22px;font-weight:800;color:{_ry_color}'>{_r10y:+.2f}%</div>"
            f"<div style='font-size:11px;color:#aaa;margin-top:2px'>{_ry_interp}</div></div>"
        )
    if _dxy_tr is not None:
        _dxy_color = "#4caf50" if _dxy_tr < -1 else "#ef5350" if _dxy_tr > 1.5 else "#90a4ae"
        _dxy_interp = "Weakening $ → gold tailwind" if _dxy_tr < -1 else "Strong $ → gold headwind" if _dxy_tr > 1.5 else "Stable"
        _macro_items.append(
            f"<div style='background:#1a2235;border:1px solid #334155;border-radius:10px;"
            f"padding:12px 16px;flex:1;min-width:160px'>"
            f"<div style='font-size:10px;color:#888;text-transform:uppercase;letter-spacing:1px;"
            f"margin-bottom:4px'>DXY 21-day trend</div>"
            f"<div style='font-size:22px;font-weight:800;color:{_dxy_color}'>{_dxy_tr:+.2f}%</div>"
            f"<div style='font-size:11px;color:#aaa;margin-top:2px'>{_dxy_interp}</div></div>"
        )
    _macro_items.append(
        f"<div style='background:#1a2235;border:1px solid #334155;border-radius:10px;"
        f"padding:12px 16px;flex:1;min-width:160px'>"
        f"<div style='font-size:10px;color:#888;text-transform:uppercase;letter-spacing:1px;"
        f"margin-bottom:4px'>Seasonal Demand</div>"
        f"<div style='font-size:12px;font-weight:600;color:#e2e8f0;margin-top:4px'>{_season_label}</div></div>"
    )
    if _macro_items:
        st.markdown(
            "<div style='display:flex;flex-wrap:wrap;gap:10px;margin-bottom:14px'>"
            + "".join(_macro_items) + "</div>",
            unsafe_allow_html=True,
        )
    st.caption(f"Regime: {_regime_txt}")

    # ── Indicator breakdown ────────────────────────────────────────────────
    _cs_pats = dt_signal.get("candlestick_patterns", {})
    _cs_names = ", ".join(_cs_pats.keys()) if _cs_pats else "None detected"
    _gvz_score = dt_signal["indicators"].get("GVZ")
    _gvz_label = (
        f"GVZ = {_gvz:.1f} — {('falling ↓' if _ctx.get('gvz_5d_ago', _gvz) > _gvz else 'rising ↑')}"
        if _gvz is not None else "GVZ"
    )
    _lmap = {
        "RSI":         f"RSI(14)  =  {dt_signal['rsi']:.1f}",
        "MACD":        "MACD (12, 26, 9)",
        "Bollinger":   f"Bollinger Band position  =  {dt_signal['bb_pct']:.1f}%",
        "EMA Trend":   "EMA 9 / 21 / 50  alignment",
        "VWAP":        f"VWAP  =  ${dt_signal['vwap']:,.2f}",
        "Momentum":    "MACD histogram momentum",
        "Candlestick": f"Patterns: {_cs_names}",
        "GVZ":         _gvz_label,
        "Regime":      f"Macro regime: {_regime_txt}",
    }
    _rows = [{"Indicator": _lmap.get(k, k),
              "Reading":   _score_label(float(v)),
              "Score":     f"{float(v):+.1f}"}
             for k, v in dt_signal["indicators"].items()]
    with st.expander("📊 Indicator breakdown", expanded=False):
        st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
        if _cs_pats:
            st.markdown("**🕯️ Candlestick patterns detected:**")
            _cp_cols = st.columns(min(len(_cs_pats), 4))
            for _ci, (_cpn, _cpv) in enumerate(_cs_pats.items()):
                _cp_col_colour = "#4caf50" if _cpv > 0 else "#ef5350"
                _cp_cols[_ci % len(_cp_cols)].markdown(
                    f"<div style='background:#1a1f2e;border:1px solid {_cp_col_colour};"
                    f"border-radius:6px;padding:6px 10px;margin:3px 0;font-size:11px;"
                    f"color:{_cp_col_colour};text-align:center'>"
                    f"<b>{_cpn}</b><br/><span style='font-size:10px;color:#aaa'>"
                    f"Score {_cpv:+.1f}</span></div>",
                    unsafe_allow_html=True,
                )

    # Chart
    _fig = build_intraday_chart(dt_df, dt_signal, show_bars=chart_bars)
    st.pyplot(_fig)
    plt.close(_fig)

    st.caption(f"_{refresh_note} · Always apply your own risk management. Not financial advice._")


_tab_day, _tab_1h = st.tabs(["📅 Day Trade  (Daily bars)", "⏱️ 1-Hour Trade  (Hourly bars)"])

with _tab_day:
    st.caption(
        "Signal derived from **daily** gold price bars — "
        "suited for trades held over a full trading session (hours to end of day). "
        "ATR is the average daily range."
    )
    _render_signal_panel(
        _day_df, _day_sig,
        chart_bars=90,
        refresh_note="Signal refreshes every 5 minutes from live daily bars",
    )

with _tab_1h:
    st.caption(
        "Signal derived from **1-hour** gold price bars — "
        "suited for intraday trades targeting a move within the next 1–4 hours. "
        "ATR is the average hourly range."
    )
    _render_signal_panel(
        _top_df, _top_sig,
        chart_bars=120,
        refresh_note="Signal refreshes every 60 seconds from live 1-hour bars",
    )

# ── Adaptive learning panel ──────────────────────────────────────────────────
with st.expander("🧠 Adaptive Learning — Indicator Weights & Analysis Log", expanded=False):
    _al_stats = summary_stats()
    _al_weights = _al_stats["weights"]
    _al_log     = load_analysis_log()

    st.caption(
        f"Self-tuning system · {_al_stats['total_analyses']} analyses run · "
        f"Last updated: {_al_stats['last_updated'][:19].replace('T', ' ')} UTC"
    )

    # Weight bars
    _wt_cols = st.columns(len(_al_weights))
    _w_min, _w_max = 0.30, 2.80
    for _wc, (_wk, _wv) in zip(_wt_cols, _al_weights.items()):
        _w_pct = (_wv - _w_min) / (_w_max - _w_min)
        _w_col = "#4caf50" if _wv >= 1.0 else "#ef5350"
        _wc.markdown(
            f"<div style='text-align:center'>"
            f"<div style='font-size:10px;color:#888;text-transform:uppercase'>{_wk}</div>"
            f"<div style='font-size:18px;font-weight:800;color:{_w_col}'>{_wv:.3f}</div>"
            f"<div style='background:#222;border-radius:4px;height:6px;margin:4px 0'>"
            f"<div style='background:{_w_col};border-radius:4px;height:6px;"
            f"width:{_w_pct*100:.0f}%'></div></div>"
            f"<div style='font-size:9px;color:#9ba8bc'>{'▲ boosted' if _wv > 1.05 else ('▼ penalised' if _wv < 0.95 else '● neutral')}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    if _al_log:
        st.markdown(f"**Most recent analyses ({min(len(_al_log), 10)} of {len(_al_log)})**")
        for _entry in reversed(_al_log[-10:]):
            _ts_str  = _entry.get("timestamp", "")[:19].replace("T", " ")
            _hor     = _entry.get("horizon_label", "?")
            _pred    = _entry.get("predicted", "?")
            _actual  = _entry.get("actual", "?")
            _pred_col = "#4caf50" if _pred == "UP" else "#ef5350"
            _act_col  = "#4caf50" if _actual == "UP" else "#ef5350"
            with st.container():
                st.markdown(
                    f"<div style='background:#1a2235;border:1px solid #1e293b;"
                    f"border-radius:8px;padding:10px 14px;margin-bottom:8px;font-size:12px'>"
                    f"<b style='color:#aaa'>{_ts_str} UTC</b> &nbsp;·&nbsp; "
                    f"Horizon: <b>{_hor}</b> &nbsp;·&nbsp; "
                    f"Predicted <b style='color:{_pred_col}'>{_pred}</b> · "
                    f"Actual <b style='color:{_act_col}'>{_actual}</b><br>"
                    + ("".join(f"<span style='color:#ef5350'>⚠ {e}</span><br>"
                               for e in _entry.get("error_sources", [])))
                    + ("".join(f"<span style='color:#4caf50'>✓ {c}</span><br>"
                               for c in _entry.get("correct_sources", [])))
                    + "</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("No analyses yet — the system will start learning automatically once "
                "the first predictions resolve (10+ minutes after first load).")

st.divider()

# ─────────────────────────────────────────────
# ALGORITHM RESEARCH SUMMARY
# ─────────────────────────────────────────────
with st.expander("🔬 Algorithm Research — What Powers This System"):
    st.markdown("""
### Algorithms Used & Research Backing

| Component | Algorithm | Research Accuracy | Source |
|---|---|---|---|
| **Main predictor** | XGBoost + RandomForest + MLP (3-member ensemble) | Diversity reduces error 10–20% | ScienceDirect 2023-2025 |
| **Multi-horizon** | Ensemble (1d / 2d / 5d models) | +5–15% vs single model | Multiple studies |
| **Day trading** | RSI(14) + MACD(12,26,9) | ~73% win rate (235 trades) | ACM BDEIM 2024 |
| **Day trading** | Bollinger Bands(20, 2σ) | Mean-reversion confirmation | Research consensus |
| **Day trading** | EMA(9/21/50) triple alignment | ~90%+ with multi-timeframe | Backtested |
| **Position sizing** | ATR(14): 1.5× stop / 2.5× target | R:R = 1.67 (statistically optimal) | Gold futures studies |
| **Institutional bias** | Session-reset VWAP | Fair-value anchor | Market microstructure |

### Why This Ensemble Works
- **XGBoost** (55% weight): level-wise tree growth, strong regularization, best standalone classical ML
  for gold price prediction across 11-year datasets (MDPI Int'l Finance, 2025)
- **RandomForest** (30% weight): bagged trees reduce variance, complementary to XGBoost's bias profile,
  provides diversity that reduces ensemble error by 5–15%
- **MLP / Neural Network** (15% weight): two-layer (64→32 neurons), Adam optimizer with L2 regularisation.
  Captures non-linear feature interactions and momentum patterns that tree splits miss, particularly
  in trend-transition regimes. Inputs are z-score scaled before the MLP to ensure stable gradient flow.
- **Calibrated confidence**: after each training window, isotonic regression maps raw ensemble scores
  to true-frequency probabilities — so "70% confidence" actually resolves correctly ~70% of the time.

### What's Next (Roadmap)
- **CNN-BiLSTM** — R² ≈ 0.92 on gold (Amini & Kalantari, PMC 2024). Highest deep-learning accuracy
- **LSTM-Autoencoder** — Best at handling gold volatility regimes (Springer Nature 2025)
- **News sentiment** — Real-time NLP scoring on gold-related headlines for macro event capture
- **GARCH volatility model** — For dynamic ATR-based stop-loss adaptation
""")

st.divider()
_sub_dt.__exit__(None, None, None)
_tab_analysis.__exit__(None, None, None)

# ─────────────────────────────────────────────
# Sidebar — settings & scheduler status
# ─────────────────────────────────────────────
# ── All settings are fixed — the system is fully autonomous ──────────────────
# No user configuration needed. The scheduler handles all data refresh,
# retraining, and self-learning automatically.
horizon_label = "Next day"
train_years   = 3
retrain_days  = 63
start_date    = pd.to_datetime("2010-01-01")
run_button    = False   # manual backtest trigger disabled

st.sidebar.markdown(
    """<div style="background:#0d1f12;border:1px solid #2a4a2a;border-radius:10px;
        padding:14px 16px;margin-bottom:12px;">
        <div style="font-size:13px;font-weight:800;color:#4caf50;margin-bottom:6px;">
            🤖 Fully Autonomous
        </div>
        <div style="font-size:11px;color:#888;line-height:1.5;">
            No setup required.<br>
            The AI monitors gold 24/7, retrains every 24h, refreshes predictions every 15 min,
            and self-adjusts indicator weights after every resolved call.
        </div>
    </div>""",
    unsafe_allow_html=True,
)

# ── Scheduler status box ─────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**📡 Background Scheduler**")
if sched:
    status_label = sched.get("status", "?")
    status_icon  = {"idle": "✅", "running": "⏳", "refreshing": "🔄", "error": "❌"}.get(
        status_label, "❓")
    st.sidebar.markdown(f"{status_icon} Status: **{status_label}**")

    # Last quick refresh
    if sched.get("last_quick"):
        last_q = datetime.fromtimestamp(sched["last_quick"])
        mins_ago = int((datetime.utcnow() - last_q).total_seconds() / 60)
        st.sidebar.markdown(f"Last refresh: **{mins_ago} min ago**")

    # Next quick refresh
    if sched.get("next_quick"):
        next_q = datetime.fromtimestamp(sched["next_quick"])
        secs_left = max(0, int((next_q - datetime.utcnow()).total_seconds()))
        if secs_left < 90:
            st.sidebar.markdown(f"Next refresh: **{secs_left}s**")
        else:
            st.sidebar.markdown(f"Next refresh: **{secs_left // 60} min**")

    # Next full retrain
    if sched.get("next_run"):
        next_dt = datetime.fromtimestamp(sched["next_run"])
        hrs_left = max(0, (next_dt - datetime.utcnow()).total_seconds() / 3600)
        st.sidebar.markdown(f"Next full retrain: **{hrs_left:.1f}h**")

    if status_label == "error" and sched.get("error"):
        st.sidebar.error(f"Error: {sched['error'][:120]}")
    st.sidebar.markdown(f"Horizon: **{sched.get('horizon', '?')}**")
else:
    st.sidebar.markdown("⏳ Starting up…")


# ── Self-Audit status box ─────────────────────
_audit_log_path = Path(__file__).parent / "data_cache" / "self_audit_log.json"
if _audit_log_path.exists():
    try:
        _audit_entries = json.loads(_audit_log_path.read_text())
        _last_audit    = _audit_entries[-1] if _audit_entries else None
        if _last_audit:
            _acc    = _last_audit.get("rolling_accuracy", 0)
            _bias   = _last_audit.get("bias", "?")
            _streak = _last_audit.get("wrong_streak", 0)
            _acts   = _last_audit.get("actions", [])
            _acc_color  = "#4caf50" if _acc >= 0.55 else ("#ffa726" if _acc >= 0.42 else "#ef5350")
            _bias_color = "#4caf50" if _bias == "balanced" else "#ffa726"
            _streak_color = "#ef5350" if _streak >= 3 else "#4caf50"
            _act_html = "".join(
                f'<div style="color:#aaa;font-size:10px;margin-top:2px;">→ {a}</div>'
                for a in _acts
            )
            st.sidebar.markdown("---")
            st.sidebar.markdown(
                f"""<div style="background:#0a1a2a;border:1px solid #1a3a5a;border-radius:8px;
                    padding:10px 12px;margin-bottom:8px;">
                    <div style="font-size:12px;font-weight:700;color:#29b6f6;margin-bottom:5px;">
                        🔍 Auto Self-Audit
                    </div>
                    <div style="font-size:11px;margin-bottom:2px;">
                        Accuracy:
                        <span style="color:{_acc_color};font-weight:700;">{_acc*100:.0f}%</span>
                        &nbsp;|&nbsp;Streak:
                        <span style="color:{_streak_color};font-weight:700;">{_streak} wrong</span>
                    </div>
                    <div style="font-size:11px;margin-bottom:4px;">
                        Bias: <span style="color:{_bias_color};">{_bias}</span>
                    </div>
                    {_act_html}
                </div>""",
                unsafe_allow_html=True,
            )
    except Exception:
        pass

st.sidebar.markdown("---")
st.sidebar.caption(
    "Sources: Yahoo Finance · FRED · GPR · EPU\n"
    "~160 global variables\n"
    "Refreshes every 15 min · Full retrain every 24h\n"
    "Self-audits & auto-corrects every 15 min"
)

_tab_tools.__enter__()
_sub_bt, _sub_hist, _sub_ig, _sub_alerts, _sub_code = st.tabs([
    "🧠  Backtest", "📋  History", "🏦  IG Trade", "🔔  Alerts", "💻  Code"
])
_sub_bt.__enter__()

# ─────────────────────────────────────────────
# Manual backtest trigger
# ─────────────────────────────────────────────
if run_button:
    target_col, return_col, horizon_days = HORIZONS[horizon_label]
    st.session_state.horizon_label = horizon_label

    st.subheader("⏳ Running backtest…")
    st.caption(
        f"Horizon: **{horizon_label}** · Training window: **{train_years}y** · "
        f"Retrain every: **{retrain_days}d**  \n"
        "First run downloads ~160 data series — allow **3–5 minutes**. "
        "Subsequent runs use cache (~30 sec)."
    )

    progress = st.progress(0.0)
    status   = st.empty()

    def cb(msg, frac):
        status.text(msg)
        progress.progress(min(frac, 1.0))

    status.text("Downloading market data…")
    try:
        raw = cached_load("2010-01-01")
    except Exception as e:
        st.error(f"Data loading failed: {e}")
        st.stop()

    status.text(f"Loaded {raw.shape[1]} series. Building features…")
    progress.progress(0.6)
    try:
        features = make_features(raw)
    except Exception as e:
        st.error(f"Feature engineering failed: {e}")
        st.stop()

    n_feat = len([c for c in features.columns
                  if not c.startswith("target_") and not c.startswith("next_return_")])
    status.text(f"Built {n_feat} features. Walk-forward training…")
    progress.progress(0.75)
    try:
        results = walk_forward(
            features,
            target_col=target_col,
            return_col=return_col,
            train_years=train_years,
            retrain_every=retrain_days,
            progress_callback=cb,
        )
    except ValueError as e:
        st.error(str(e))
        st.stop()

    progress.empty()
    status.empty()

    results["horizon_label"] = horizon_label
    results["run_at"]        = datetime.utcnow().isoformat()

    st.session_state.results       = results
    st.session_state.last_run      = time.time()
    st.session_state.horizon_label = horizon_label
    st.session_state.source        = "manual"

    # Save live prediction
    latest_pred  = int(results["predictions"].iloc[-1])
    latest_proba = float(results["probas"].iloc[-1])
    latest_date  = results["predictions"].index[-1].to_pydatetime()
    save_live_prediction(latest_date, latest_pred, latest_proba,
                         horizon_label, horizon_days)

# ─────────────────────────────────────────────
# Display results
# ─────────────────────────────────────────────
results = st.session_state.results

if results is not None:
    hl = results.get("horizon_label", st.session_state.horizon_label)
    run_at = results.get("run_at")
    if run_at:
        run_dt = datetime.fromisoformat(run_at)
        st.subheader(f"Results — {hl}")
        st.caption(f"Model run: {run_dt.strftime('%d %b %Y %H:%M')} UTC · "
                   f"{'background scheduler' if st.session_state.source == 'scheduler' else 'manual run'}")
    else:
        st.subheader(f"Results — {hl}")

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Backtest accuracy", f"{results['accuracy']:.2%}",
              delta=f"{(results['accuracy'] - 0.5)*100:+.2f} pp vs coin flip")
    c2.metric("Sharpe (annualised)", f"{results['sharpe']:.2f}")
    strat_total = results["strategy_curve"].iloc[-1] - 1
    bh_total    = results["buyhold_curve"].iloc[-1] - 1
    c3.metric("Strategy total return", f"{strat_total:+.1%}",
              delta=f"{(strat_total - bh_total)*100:+.1f} pp vs B&H")
    c4.metric("Buy & hold gold", f"{bh_total:+.1%}")
    st.caption(f"{results['n_predictions']:,} trading days · {results['n_features']} features")

    # Latest signal
    latest_pred  = int(results["predictions"].iloc[-1])
    latest_proba = float(results["probas"].iloc[-1])
    latest_date  = results["predictions"].index[-1].date()
    conf_pct     = latest_proba * 100 if latest_pred == 1 else (1 - latest_proba) * 100

    if latest_pred == 1:
        st.success(f"**Current signal ({latest_date}):** 📈 UP  —  Confidence {conf_pct:.1f}%  ({hl})")
    else:
        st.error(f"**Current signal ({latest_date}):** 📉 DOWN  —  Confidence {conf_pct:.1f}%  ({hl})")

    st.divider()

    # Gold price + signals chart
    st.subheader("Gold Price with Model Signals")
    st.caption("🟢 Predicted UP · 🔴 Predicted DOWN")
    preds      = results["predictions"]
    one_rets   = results["returns"]
    gold_levels = (1 + one_rets).cumprod()
    if gold_2y is not None:
        overlap = gold_2y.index.intersection(gold_levels.index)
        if len(overlap) > 0:
            first = overlap[0]
            scale = float(gold_2y.loc[first]) / float(gold_levels.loc[first])
            gold_levels = gold_levels * scale
    up_mask = (preds == 1).values
    dn_mask = (preds == 0).values
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(gold_levels.index, gold_levels.values,
             color="goldenrod", lw=1.6, label="Gold price", zorder=2)
    ax1.scatter(gold_levels.index[up_mask], gold_levels.values[up_mask],
                color="green", s=5, alpha=0.45, label="Predicted UP", zorder=3)
    ax1.scatter(gold_levels.index[dn_mask], gold_levels.values[dn_mask],
                color="red",   s=5, alpha=0.45, label="Predicted DOWN", zorder=3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.set_ylabel("Price (USD / oz)")
    ax1.legend(markerscale=4)
    ax1.grid(alpha=0.2)
    st.pyplot(fig1)
    plt.close(fig1)

    # Rolling 90-day accuracy
    st.subheader("Rolling 90-Day Prediction Accuracy")
    ra = results["rolling_accuracy"].dropna()
    fig_ra, ax_ra = plt.subplots(figsize=(12, 3))
    ax_ra.plot(ra.index, ra.values * 100, color="steelblue", lw=1.5, label="90-day accuracy")
    ax_ra.axhline(50, color="grey",  lw=1,   ls="--", label="50% (coin flip)")
    ax_ra.axhline(55, color="green", lw=0.8, ls=":",  alpha=0.6, label="55% threshold")
    ax_ra.set_ylabel("Accuracy (%)")
    ax_ra.set_ylim(35, 70)
    ax_ra.legend(fontsize=8)
    ax_ra.grid(alpha=0.2)
    ax_ra.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    st.pyplot(fig_ra)
    plt.close(fig_ra)

    # Equity curve
    st.subheader("Strategy vs Buy & Hold")
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    results["strategy_curve"].plot(ax=ax2, label="Model long/short", color="darkgreen", lw=1.8)
    results["buyhold_curve"].plot(ax=ax2,  label="Buy & hold gold",  color="goldenrod", lw=1.8)
    ax2.set_ylabel("Growth of $1")
    ax2.legend()
    ax2.grid(alpha=0.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    st.pyplot(fig2)
    plt.close(fig2)

    st.divider()

    # Feature importance
    st.subheader("Top 30 Features by Importance")
    top = results["feature_importance"].head(30)
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    top.iloc[::-1].plot.barh(ax=ax3, color="steelblue")
    ax3.set_xlabel("Avg importance across walk-forward folds")
    ax3.grid(alpha=0.15, axis="x")
    st.pyplot(fig3)
    plt.close(fig3)
    with st.expander("All features (full table)"):
        st.dataframe(results["feature_importance"].rename("importance").to_frame(),
                     width="stretch")

_sub_bt.__exit__(None, None, None)
_sub_hist.__enter__()

# ─────────────────────────────────────────────
# Live prediction history
# ─────────────────────────────────────────────
st.subheader("📋 Live Prediction History")

# ── Intraday accuracy by horizon ──────────────────────────────────────────
_hist_ipreds = load_intraday_preds()
_hist_iacc   = intraday_accuracy_by_horizon(_hist_ipreds)

_intra_order = ["10 min", "30 min", "1 hour", "2 hours", "5 hours", "End of Day"]

st.markdown("##### Intraday Horizons")
_ih_cols = st.columns(len(_intra_order))
for _ihc, _ihl in zip(_ih_cols, _intra_order):
    _iacc_v, _iacc_n = _hist_iacc.get(_ihl, (None, 0))
    if _iacc_n > 0:
        _iacc_icon = "🟢" if _iacc_v >= 0.60 else ("🟡" if _iacc_v >= 0.45 else "🔴")
        _ihc.metric(_ihl, f"{_iacc_icon} {_iacc_v:.0%}", delta=f"{_iacc_n} resolved")
    else:
        _ihc.metric(_ihl, "—", delta="⏳ pending")

# ── Daily scheduler accuracy ───────────────────────────────────────────────
st.markdown("##### Daily & Multi-Day Horizons")
_daily_horizons = sorted(set(p["horizon"] for p in live_preds)) if live_preds else []
_dh_cols = st.columns(max(len(_daily_horizons), 1))
for _dhc, _dhl in zip(_dh_cols, _daily_horizons):
    _dh_preds  = [p for p in live_preds if p["horizon"] == _dhl]
    _dh_res    = [p for p in _dh_preds if p["outcome"] is not None]
    _dh_n      = len(_dh_res)
    if _dh_n > 0:
        _dh_acc  = sum(1 for p in _dh_res if p["direction"] == p["outcome"]) / _dh_n
        _dh_icon = "🟢" if _dh_acc >= 0.60 else ("🟡" if _dh_acc >= 0.45 else "🔴")
        _dhc.metric(_dhl, f"{_dh_icon} {_dh_acc:.0%}", delta=f"{_dh_n} resolved")
    else:
        _dhc.metric(_dhl, "—", delta="⏳ pending")

if not _daily_horizons:
    st.caption("_Daily predictions will appear after the first scheduler run._")

# ── Intraday prediction history table ─────────────────────────────────────
if _hist_ipreds:
    with st.expander(f"Intraday prediction log ({len(_hist_ipreds)} entries)"):
        _irows = []
        for _ip in reversed(_hist_ipreds[-50:]):
            _outcome = ("✅ Correct" if _ip.get("correct") is True
                        else ("❌ Wrong" if _ip.get("correct") is False else "⏳ Pending"))
            _irows.append({
                "Made at (UTC)":  _ip.get("made_at", "")[:16].replace("T", " "),
                "Horizon":        _ip.get("horizon_label", "?"),
                "Predicted $":    f"${_ip.get('predicted_price', 0):,.2f}",
                "Actual $":       (f"${_ip['actual_price']:,.2f}"
                                   if _ip.get("actual_price") else "—"),
                "Outcome":        _outcome,
            })
        st.dataframe(pd.DataFrame(_irows), use_container_width=True, hide_index=True)

# ── Daily prediction history table ────────────────────────────────────────
if len(live_preds) > 0:
    with st.expander(f"Daily prediction log ({len(live_preds)} entries)"):
        rows = []
        for p in reversed(live_preds):
            rows.append({
                "Made on":    p["made_on"],
                "Horizon":    p["horizon"],
                "Signal":     "📈 UP" if p["direction"] == 1 else "📉 DOWN",
                "Confidence": f"{p['confidence']*100:.1f}%",
                "Target date":p["target_date"],
                "Outcome":    ("✅ Correct" if p["direction"] == p["outcome"]
                               else ("❌ Wrong" if p["outcome"] is not None else "⏳ Pending")),
                "Gold move":  (f"{p.get('actual_return', '')}%"
                               if p["outcome"] is not None else "—"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if live_acc is not None:
            st.caption(
                f"Overall daily accuracy: **{live_acc:.1%}** across **{live_n}** "
                f"resolved predictions — benchmark (coin flip) is 50%"
            )

_sub_hist.__exit__(None, None, None)

# ─────────────────────────────────────────────
# IG TRADE TAB
# ─────────────────────────────────────────────
_sub_ig.__enter__()

# ── IG contract constants (Spot Gold A$10 Contract) ──────────────────────────
_IG_POINT_AUD  = 10.0   # AUD per 1 $/oz move per contract (confirmed via P&L formula)
_IG_MARGIN_PCT = 0.05   # 5 % margin (Tiers 1-3)
_IG_SPREAD     = 0.5    # points (0.5 $/oz spread)
_IG_MIN_SIZE   = 0.5    # minimum contract size
_IG_MIN_STOP   = 1      # minimum stop distance in points

# ── Pull signals ──────────────────────────────────────────────────────────────
_ig_price  = _top_live_price or latest          # live spot price
_ig_buy    = _ig_price + _IG_SPREAD / 2         # effective BUY entry (ask)
_ig_sell   = _ig_price - _IG_SPREAD / 2         # effective SELL entry (bid)

# Multi-horizon votes
_ig_mh_votes = []
for _ig_h in ["1", "2", "5"]:
    _ig_m = mh_preds.get(_ig_h, {})
    if _ig_m:
        _ig_mh_votes.append((_ig_h, int(_ig_m.get("direction", 0)),
                             float(_ig_m.get("confidence", 0.0))))

# Re-use the already-computed signals from the top of the file
# _day_sig / _1h_label / _day_label are set during the Live tab section above
_ig_daily = _day_sig or {}
_ig_hour  = _top_sig  or {}

def _ig_dir_int(label_str):
    """Convert signal label string → 1 (bullish) / 0 (bearish) / None."""
    lbl = str(label_str).upper()
    if "BUY"  in lbl: return 1
    if "SELL" in lbl: return 0
    return None

_ig_daily_dir = _ig_dir_int(_day_label)
_ig_hour_dir  = _ig_dir_int(_1h_label)

# Overall consensus (simple majority across all signals)
_ig_all_dirs = [d for _, d, _ in _ig_mh_votes]
if _ig_daily_dir is not None: _ig_all_dirs.append(_ig_daily_dir)
if _ig_hour_dir  is not None: _ig_all_dirs.append(_ig_hour_dir)
_ig_bull_count = sum(_ig_all_dirs)
_ig_bear_count = len(_ig_all_dirs) - _ig_bull_count
_ig_consensus  = "BUY" if _ig_bull_count > _ig_bear_count else "SELL"
_ig_cons_col   = "#4caf50" if _ig_consensus == "BUY" else "#ef5350"
_ig_cons_pct   = (_ig_bull_count / len(_ig_all_dirs) * 100) if _ig_all_dirs else 50

# ── ATR-based stop distance (use actual SL distance from daily signal) ────────
_ig_sig_entry = float(_ig_daily.get("entry", _ig_price))
_ig_sig_sl    = float(_ig_daily.get("stop_loss", _ig_sig_entry * 0.988))
_ig_atr       = abs(_ig_sig_entry - _ig_sig_sl) or (_ig_price * 0.012)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🏦 IG CFD Trade Planner")
st.caption("Spot Gold (A$10 Contract) · 1 point = 1 $/oz = AUD $10 per contract · Spread 0.5 pts · Margin 5%")

# ── Section 1: Consensus Signal ───────────────────────────────────────────────
_sig_arrow = "▲" if _ig_consensus == "BUY" else "▼"
_sig_pct   = f"{_ig_cons_pct:.0f}% of signals bullish"
st.markdown(
    f"""<div style="background:#1a2235;border:2px solid {_ig_cons_col};border-radius:12px;
        padding:18px 22px;margin-bottom:18px;display:flex;align-items:center;gap:24px;">
        <div style="font-size:40px;font-weight:900;color:{_ig_cons_col};">
            {_sig_arrow} {_ig_consensus}
        </div>
        <div>
            <div style="font-size:13px;color:#aaa;">Model Consensus · Spot Gold</div>
            <div style="font-size:16px;font-weight:700;color:#e2e8f0;">{_sig_pct}</div>
            <div style="font-size:12px;color:#9ba8bc;">
                {_ig_bull_count} bullish · {_ig_bear_count} bearish signal(s)
            </div>
        </div>
        <div style="margin-left:auto;text-align:right;">
            <div style="font-size:12px;color:#9ba8bc;">Live spot</div>
            <div style="font-size:22px;font-weight:900;color:#f5c518;">${_ig_price:,.2f}</div>
            <div style="font-size:11px;color:#8a9ab5;">Buy {_ig_buy:.2f} · Sell {_ig_sell:.2f}</div>
        </div>
    </div>""",
    unsafe_allow_html=True,
)

# ── Section 2: Signal Breakdown ───────────────────────────────────────────────
st.markdown("### Signal Breakdown")
_ig_sc = st.columns(len(_ig_mh_votes) + 2)

for _ci, (_ig_h, _ig_d, _ig_c) in enumerate(_ig_mh_votes):
    _lbl   = {"1":"1-Day","2":"2-Day","5":"5-Day"}.get(_ig_h, f"{_ig_h}d")
    _col   = "#4caf50" if _ig_d == 1 else "#ef5350"
    _arrow = "▲ UP" if _ig_d == 1 else "▼ DOWN"
    _ig_sc[_ci].markdown(
        f"""<div style="background:#1a2235;border:1px solid #2a2a2a;border-radius:8px;
            padding:12px;text-align:center;">
            <div style="font-size:10px;color:#8a9ab5;text-transform:uppercase;">{_lbl} Forecast</div>
            <div style="font-size:16px;font-weight:800;color:{_col};">{_arrow}</div>
            <div style="font-size:11px;color:#888;">{_ig_c:.0%} conviction</div>
        </div>""",
        unsafe_allow_html=True,
    )

_daily_lbl = _day_label
_daily_col = "#4caf50" if _ig_daily_dir == 1 else ("#ef5350" if _ig_daily_dir == 0 else "#888")
_daily_conf = int(float(_ig_daily.get("confidence", 0)) * 100)
_ig_sc[-2].markdown(
    f"""<div style="background:#1a2235;border:1px solid #2a2a2a;border-radius:8px;
        padding:12px;text-align:center;">
        <div style="font-size:10px;color:#8a9ab5;text-transform:uppercase;">Day Signal</div>
        <div style="font-size:14px;font-weight:800;color:{_daily_col};">{_daily_lbl}</div>
        <div style="font-size:11px;color:#888;">{_daily_conf}% confidence</div>
    </div>""",
    unsafe_allow_html=True,
)
_hour_lbl = _1h_label
_hour_col = "#4caf50" if _ig_hour_dir == 1 else ("#ef5350" if _ig_hour_dir == 0 else "#888")
_hour_conf = int(float(_ig_hour.get("confidence", 0)) * 100)
_ig_sc[-1].markdown(
    f"""<div style="background:#1a2235;border:1px solid #2a2a2a;border-radius:8px;
        padding:12px;text-align:center;">
        <div style="font-size:10px;color:#8a9ab5;text-transform:uppercase;">1-Hour Signal</div>
        <div style="font-size:14px;font-weight:800;color:{_hour_col};">{_hour_lbl}</div>
        <div style="font-size:11px;color:#888;">{_hour_conf}% confidence</div>
    </div>""",
    unsafe_allow_html=True,
)

st.divider()

# ── Section 3: AI-Generated Trade Setup ──────────────────────────────────────
st.markdown("### 🤖 AI Trade Setup")
st.caption(
    "The model generates a complete trade setup automatically — no input required. "
    "Review the parameters below before placing your order on IG."
)

# ── Auto-compute all trade parameters from the model ─────────────────────────
_ig_is_buy   = (_ig_consensus == "BUY")
_ig_contracts = 1.0   # default starting size; user adjusts on IG platform
_ig_entry_px  = float(round(_ig_buy if _ig_is_buy else _ig_sell, 2))

# Stop and Limit from ATR-based risk management
_ig_atr_stop  = round(_ig_entry_px - _ig_atr * 1.5, 0) if _ig_is_buy else round(_ig_entry_px + _ig_atr * 1.5, 0)
_ig_atr_limit = (
    float(_ig_daily.get("target", round(_ig_entry_px + _ig_atr * 2.5, 0)))
    if _ig_is_buy else
    float(_ig_daily.get("target", round(_ig_entry_px - _ig_atr * 2.5, 0)))
)
_ig_stop  = float(max(1.0, _ig_atr_stop))
_ig_limit = float(max(1.0, _ig_atr_limit))

# ── Read-only trade setup card ────────────────────────────────────────────────
_ts_dir_col  = "#4caf50" if _ig_is_buy else "#ef5350"
_ts_dir_lbl  = "▲ BUY (Long)"  if _ig_is_buy else "▼ SELL (Short)"
_ts_stop_dist  = abs(_ig_entry_px - _ig_stop)
_ts_limit_dist = abs(_ig_limit   - _ig_entry_px)
_ts_rr   = _ts_limit_dist / _ts_stop_dist if _ts_stop_dist > 0 else 0
_ts_profit = _ts_limit_dist * _ig_contracts * _IG_POINT_AUD
_ts_loss   = _ts_stop_dist  * _ig_contracts * _IG_POINT_AUD
_ts_margin = _ig_entry_px   * _ig_contracts * _IG_POINT_AUD * _IG_MARGIN_PCT

def _ts_row(label, value, colour="#e0e0e0", sublabel=""):
    sub = f'<div style="font-size:10px;color:#8a9ab5;margin-top:2px;">{sublabel}</div>' if sublabel else ""
    return (
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:8px 0;border-bottom:1px solid #ffffff08;">'
        f'<span style="font-size:12px;color:#9ba8bc;">{label}</span>'
        f'<div style="text-align:right;">'
        f'<span style="font-size:14px;font-weight:700;color:{colour};">{value}</span>'
        f'{sub}</div></div>'
    )

_ts_rows = "".join([
    _ts_row("Direction",   _ts_dir_lbl,            _ts_dir_col),
    _ts_row("Entry",       f"${_ig_entry_px:,.2f}", "#f5c518",
            "at current model bid/ask mid"),
    _ts_row("Stop Loss",   f"${_ig_stop:,.0f}",     "#ef5350",
            f"−${_ts_stop_dist:,.0f} per oz · {_ts_stop_dist/_ig_entry_px*100:.2f}% from entry"),
    _ts_row("Take Profit", f"${_ig_limit:,.0f}",    "#4caf50",
            f"+${_ts_limit_dist:,.0f} per oz · {_ts_limit_dist/_ig_entry_px*100:.2f}% from entry"),
    _ts_row("Size",        "1 contract",            "#e0e0e0",
            "adjust on IG to match your account risk tolerance"),
    _ts_row("Risk : Reward", f"1 : {_ts_rr:.1f}",
            "#4caf50" if _ts_rr >= 1.5 else "#ffa726"),
    _ts_row("Potential profit (1 lot)", f"A${_ts_profit:,.0f}", "#4caf50"),
    _ts_row("Max loss (1 lot)",         f"A${_ts_loss:,.0f}",   "#ef5350"),
    _ts_row("Margin required (1 lot)",  f"A${_ts_margin:,.0f}", "#888"),
])

st.markdown(
    f"""<div style="background:#1a2235;border:1px solid {_ts_dir_col}44;border-radius:12px;
        padding:18px 22px;margin-bottom:16px;">
        <div style="font-size:10px;color:#8a9ab5;letter-spacing:1.5px;text-transform:uppercase;
            margin-bottom:12px;">
            Model-Generated Setup · IG CFD · Gold (XAU/USD)
        </div>
        {_ts_rows}
        <div style="margin-top:12px;font-size:9px;color:#2a2a2a;text-align:right;">
            AI-generated setup — not financial advice. Always apply your own risk management.
        </div>
    </div>""",
    unsafe_allow_html=True,
)

# ── Calculations ─────────────────────────────────────────────────────────────
_ig_sign        = 1 if _ig_is_buy else -1
_ig_stop_dist   = abs(_ig_entry_px - _ig_stop)
_ig_limit_dist  = abs(_ig_limit   - _ig_entry_px)
_ig_stop_dist   = max(_ig_stop_dist, _IG_MIN_STOP)   # enforce min
_ig_rr          = (_ig_limit_dist / _ig_stop_dist) if _ig_stop_dist > 0 else 0

_ig_profit_aud  = _ig_limit_dist * _ig_contracts * _IG_POINT_AUD
_ig_loss_aud    = _ig_stop_dist  * _ig_contracts * _IG_POINT_AUD
_ig_margin_aud  = _ig_entry_px   * _ig_contracts * _IG_POINT_AUD * _IG_MARGIN_PCT
_ig_profit_pp   = _IG_POINT_AUD  * _ig_contracts   # AUD per 1-point move

_rr_col = "#4caf50" if _ig_rr >= 2.0 else ("#f5c518" if _ig_rr >= 1.0 else "#ef5350")

# ── Results display ───────────────────────────────────────────────────────────
st.markdown("#### Position Summary")
_igr1, _igr2, _igr3, _igr4 = st.columns(4)

_igr1.markdown(
    f"""<div style="background:#1a2235;border:1px solid #2a2a2a;border-radius:8px;padding:14px;">
        <div style="font-size:10px;color:#8a9ab5;text-transform:uppercase;">Profit if Limit Hit</div>
        <div style="font-size:22px;font-weight:900;color:#4caf50;">+A${_ig_profit_aud:,.0f}</div>
        <div style="font-size:11px;color:#9ba8bc;">+{_ig_limit_dist:.1f} pts × {_ig_contracts}c × $10</div>
    </div>""", unsafe_allow_html=True)

_igr2.markdown(
    f"""<div style="background:#1a2235;border:1px solid #2a2a2a;border-radius:8px;padding:14px;">
        <div style="font-size:10px;color:#8a9ab5;text-transform:uppercase;">Loss if Stop Hit</div>
        <div style="font-size:22px;font-weight:900;color:#ef5350;">−A${_ig_loss_aud:,.0f}</div>
        <div style="font-size:11px;color:#9ba8bc;">−{_ig_stop_dist:.1f} pts × {_ig_contracts}c × $10</div>
    </div>""", unsafe_allow_html=True)

_igr3.markdown(
    f"""<div style="background:#1a2235;border:1px solid {_rr_col};border-radius:8px;padding:14px;">
        <div style="font-size:10px;color:#8a9ab5;text-transform:uppercase;">Risk : Reward</div>
        <div style="font-size:22px;font-weight:900;color:{_rr_col};">1 : {_ig_rr:.1f}</div>
        <div style="font-size:11px;color:#9ba8bc;">{"✓ Favourable" if _ig_rr >= 2 else ("↗ Acceptable" if _ig_rr >= 1 else "✗ Poor — widen limit")}</div>
    </div>""", unsafe_allow_html=True)

_igr4.markdown(
    f"""<div style="background:#1a2235;border:1px solid #2a2a2a;border-radius:8px;padding:14px;">
        <div style="font-size:10px;color:#8a9ab5;text-transform:uppercase;">Margin Required</div>
        <div style="font-size:22px;font-weight:900;color:#e2e8f0;">A${_ig_margin_aud:,.0f}</div>
        <div style="font-size:11px;color:#9ba8bc;">5% of notional · {_ig_contracts}c</div>
    </div>""", unsafe_allow_html=True)

st.markdown(
    f"""<div style="background:#1a2235;border:1px solid #2a2a2a;border-radius:8px;
        padding:12px 18px;margin-top:10px;display:flex;gap:32px;flex-wrap:wrap;">
        <span style="font-size:12px;color:#888;">
            <b style="color:#e2e8f0;">AUD $10</b> per point per contract
        </span>
        <span style="font-size:12px;color:#888;">
            P&amp;L per full point move: <b style="color:#f5c518;">A${_ig_profit_pp:,.0f}</b>
        </span>
        <span style="font-size:12px;color:#888;">
            Stop distance: <b style="color:#ef5350;">{_ig_stop_dist:.1f} pts</b>
        </span>
        <span style="font-size:12px;color:#888;">
            Limit distance: <b style="color:#4caf50;">{_ig_limit_dist:.1f} pts</b>
        </span>
    </div>""",
    unsafe_allow_html=True,
)

st.divider()

# ── Section 4: Open Position Monitor ─────────────────────────────────────────
st.markdown("### Open Position Monitor")
st.caption("Enter your current IG position to see real-time P&L and model guidance.")

with st.expander("Enter open position", expanded=False):
    _igp1, _igp2, _igp3 = st.columns(3)
    _ig_pos_dir   = _igp1.selectbox("Position", ["BUY (Long)", "SELL (Short)"], key="ig_pos_dir")
    _ig_pos_open  = _igp2.number_input("Open price", min_value=100.0, max_value=99999.0,
                                        value=float(round(_ig_price, 2)), step=0.5, key="ig_pos_open")
    _ig_pos_size  = _igp3.number_input("Size (contracts)", min_value=0.5, max_value=100.0,
                                        value=1.0, step=0.5, key="ig_pos_size")
    _igp4, _igp5 = st.columns(2)
    _ig_pos_stop  = _igp4.number_input("Stop Level", min_value=1.0, max_value=99999.0,
                                        value=float(max(1.0, round(_ig_price * 0.97, 0))),
                                        step=1.0, key="ig_pos_stop")
    _ig_pos_limit = _igp5.number_input("Limit Level", min_value=1.0, max_value=99999.0,
                                        value=float(round(_ig_price * 1.02, 0)),
                                        step=1.0, key="ig_pos_limit")

    _ig_pos_sign  = 1 if "BUY" in _ig_pos_dir else -1

    # ── Correct P&L: (current_mid - open) × sign × size × $10 ──────────────
    _ig_pnl = (_ig_price - _ig_pos_open) * _ig_pos_sign * _ig_pos_size * _IG_POINT_AUD

    # ── Distance from CURRENT price to stop/limit (in points) ───────────────
    _ig_pts_to_stop  = abs(_ig_price - _ig_pos_stop)
    _ig_pts_to_limit = abs(_ig_pos_limit - _ig_price)
    _ig_aud_at_stop  = _ig_pts_to_stop  * _ig_pos_size * _IG_POINT_AUD   # AUD at risk if stop hit now
    _ig_aud_at_limit = _ig_pts_to_limit * _ig_pos_size * _IG_POINT_AUD   # AUD gained if limit hit now

    # ── Margin: IG calculates on CURRENT price, not open price ──────────────
    _ig_pos_margin = _ig_price * _ig_pos_size * _IG_POINT_AUD * _IG_MARGIN_PCT

    # ── Breakeven (what price you need to get to zero) ───────────────────────
    _ig_breakeven = _ig_pos_open  # exactly at open (before spread)

    _pnl_col = "#4caf50" if _ig_pnl >= 0 else "#ef5350"

    # ── Model alignment ───────────────────────────────────────────────────────
    _pos_aligned = ((_ig_pos_sign == 1 and _ig_consensus == "BUY") or
                    (_ig_pos_sign == -1 and _ig_consensus == "SELL"))
    _align_txt  = "Model AGREES — hold" if _pos_aligned else "Model DISAGREES — consider closing"
    _align_col  = "#4caf50" if _pos_aligned else "#ef5350"
    _align_icon = "✓" if _pos_aligned else "✗"

    _igm1, _igm2, _igm3, _igm4 = st.columns(4)
    _igm1.markdown(
        f"""<div style="background:#1a2235;border:1px solid #2a2a2a;border-radius:8px;padding:12px;text-align:center;">
            <div style="font-size:10px;color:#8a9ab5;">Current P&amp;L</div>
            <div style="font-size:20px;font-weight:900;color:{_pnl_col};">
                {"+" if _ig_pnl>=0 else ""}A${_ig_pnl:,.0f}
            </div>
            <div style="font-size:11px;color:#9ba8bc;">vs open {_ig_pos_open:.2f}</div>
        </div>""", unsafe_allow_html=True)
    _igm2.markdown(
        f"""<div style="background:#1a2235;border:1px solid #2a2a2a;border-radius:8px;padding:12px;text-align:center;">
            <div style="font-size:10px;color:#8a9ab5;">Pts to Stop</div>
            <div style="font-size:20px;font-weight:900;color:#ef5350;">{_ig_pts_to_stop:.0f} pts</div>
            <div style="font-size:11px;color:#9ba8bc;">A${_ig_aud_at_stop:,.0f} at risk</div>
        </div>""", unsafe_allow_html=True)
    _igm3.markdown(
        f"""<div style="background:#1a2235;border:1px solid #2a2a2a;border-radius:8px;padding:12px;text-align:center;">
            <div style="font-size:10px;color:#8a9ab5;">Pts to Limit</div>
            <div style="font-size:20px;font-weight:900;color:#4caf50;">{_ig_pts_to_limit:.0f} pts</div>
            <div style="font-size:11px;color:#9ba8bc;">A${_ig_aud_at_limit:,.0f} potential</div>
        </div>""", unsafe_allow_html=True)
    _igm4.markdown(
        f"""<div style="background:#1a2235;border:1px solid {_align_col};border-radius:8px;padding:12px;text-align:center;">
            <div style="font-size:10px;color:#8a9ab5;">Model View</div>
            <div style="font-size:18px;font-weight:900;color:{_align_col};">{_align_icon}</div>
            <div style="font-size:11px;font-weight:700;color:{_align_col};">{_align_txt}</div>
        </div>""", unsafe_allow_html=True)

    # ── Margin & detailed analysis row ────────────────────────────────────────
    st.markdown(
        f"""<div style="background:#1a2235;border:1px solid #2a2a2a;border-radius:8px;
            padding:12px 18px;margin-top:10px;display:flex;gap:28px;flex-wrap:wrap;align-items:center;">
            <span style="font-size:12px;color:#888;">
                Margin (current price):
                <b style="color:#e2e8f0;">A${_ig_pos_margin:,.0f}</b>
            </span>
            <span style="font-size:12px;color:#888;">
                Breakeven price:
                <b style="color:#f5c518;">${_ig_breakeven:,.2f}</b>
            </span>
            <span style="font-size:12px;color:#888;">
                Need
                <b style="color:{'#4caf50' if _ig_pos_sign==1 else '#ef5350'}">
                    {'+' if _ig_pos_sign==1 else '-'}{abs(_ig_price - _ig_breakeven):.1f} pts
                </b>
                to breakeven
            </span>
        </div>""", unsafe_allow_html=True)

    # ── Model vs position analysis ────────────────────────────────────────────
    _ig_1d      = mh_preds.get("1", {})
    _ig_1d_tp   = float(_ig_1d.get("target_price", 0)) if _ig_1d else 0
    _ig_5d      = mh_preds.get("5", {})
    _ig_5d_tp   = float(_ig_5d.get("target_price", 0)) if _ig_5d else 0

    # Pre-compute 1d & 5d P&L for the open position
    _ig_1d_pnl  = (_ig_1d_tp - _ig_pos_open) * _ig_pos_sign * _ig_pos_size * _IG_POINT_AUD if _ig_1d_tp else 0
    _ig_5d_pnl  = (_ig_5d_tp - _ig_pos_open) * _ig_pos_sign * _ig_pos_size * _IG_POINT_AUD if _ig_5d_tp else 0
    _tp1_col    = "#4caf50" if _ig_1d_pnl >= 0 else "#ef5350"
    _tp5_col    = "#4caf50" if _ig_5d_pnl >= 0 else "#ef5350"
    _tp1_sign   = "+" if _ig_1d_pnl >= 0 else ""
    _tp5_sign   = "+" if _ig_5d_pnl >= 0 else ""
    _ig_5d_html = (
        f'<span style="font-size:12px;color:#888;">5-day target: '
        f'<b style="color:#f5c518;">${_ig_5d_tp:,.2f}</b> '
        f'&rarr; P&amp;L: <b style="color:{_tp5_col};">{_tp5_sign}A${_ig_5d_pnl:,.0f}</b></span>'
    ) if _ig_5d_tp > 0 else ""

    if _ig_1d_tp > 0:
        st.markdown(
            f"""<div style="background:#1a2235;border:1px solid #333;border-radius:8px;
                padding:14px 18px;margin-top:12px;">
                <div style="font-size:11px;color:#9ba8bc;margin-bottom:8px;text-transform:uppercase;">
                    Model Forecast vs Your Position
                </div>
                <div style="display:flex;gap:32px;flex-wrap:wrap;">
                    <span style="font-size:12px;color:#888;">
                        1-day target: <b style="color:#f5c518;">${_ig_1d_tp:,.2f}</b>
                        &rarr; P&amp;L: <b style="color:{_tp1_col};">{_tp1_sign}A${_ig_1d_pnl:,.0f}</b>
                    </span>
                    {_ig_5d_html}
                    <span style="font-size:12px;color:#aaa;">
                        Price feed gap: our mid <b style="color:#f5c518;">${_ig_price:,.2f}</b>
                        vs IG bid/ask midpoint
                        <b style="color:#ccc;">(typically ±$2–5 due to different data sources &amp; timing)</b>
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)

st.divider()

# ── Section 5: IG Contract Reference ─────────────────────────────────────────
with st.expander("IG Contract Specifications — Spot Gold (A$10)", expanded=False):
    st.markdown("""
| Specification | Value |
|---|---|
| Instrument | Spot Gold (A$10 Contract) |
| Chart Code | GOLD |
| News Code | GOL |
| Contract Size | GOL 10 (10 troy oz) |
| **Value of One Point** | **AUD $10 per contract** |
| One Point Means | 1 $/Troy Ounce |
| Minimum Size | 0.50 contracts |
| Minimum Stop Distance | 1 point |
| Min Guaranteed Stop Distance | 2 points |
| Spread (typical) | 0.5 points |
| **Margin (Tiers 1–3, 0–690 USD)** | **5%** |
| Margin (Tier 4, 690+ USD) | 7.5% |
| Slippage Factor | 50% |
| Normal Market Size | 80 |

**P&L formula:** `(exit − entry) × contracts × AUD $10` (positive for BUY, negative for SELL when going against you)

**Margin formula:** `entry_price × contracts × AUD $10 × 5%`
""")

_sub_ig.__exit__(None, None, None)

# ─────────────────────────────────────────────
# CODE TAB — browse all source files
# ─────────────────────────────────────────────
_sub_code.__enter__()

st.markdown("## 💻 Source Code")
st.caption(
    "Browse every file that powers this app. "
    "Expand a file to view its full source."
)

_CODE_FILES = [
    ("app.py",                    "Main Streamlit app — UI, tabs, ticker, topbar"),
    ("gold_model.py",             "ML model — XGBoost · Random Forest · MLP ensemble + walk-forward backtest"),
    ("day_trading.py",            "Day trading signals — 1-hour and daily timeframe signal engine"),
    ("candlestick_patterns.py",   "Candlestick pattern recognition — 20+ classic patterns"),
    ("adaptive_learning.py",      "Self-learning adaptive weights — updates model voting weights after resolved predictions"),
    ("scheduler.py",              "Background scheduler — runs full retrain + prediction cycle every 4 hours"),
]

for _fname, _desc in _CODE_FILES:
    _fpath = Path(__file__).parent / _fname
    with st.expander(f"**{_fname}** · {_desc}", expanded=False):
        try:
            _src = _fpath.read_text(encoding="utf-8")
            _lines = _src.count("\n") + 1
            st.caption(f"{_lines:,} lines")
            st.code(_src, language="python", line_numbers=True)
        except Exception as _e:
            st.error(f"Could not read {_fname}: {_e}")

_sub_code.__exit__(None, None, None)

# ─────────────────────────────────────────────
# ALERTS TAB — Telegram configuration & controls
# ─────────────────────────────────────────────
_sub_alerts.__enter__()

from telegram_alerts import load_config as _tg_load, save_config as _tg_save, test_connection as _tg_test

_tg_cfg = _tg_load()

st.markdown("## 🔔 Telegram Alerts")
st.caption(
    "Receive instant Telegram messages when price approaches your stop/limit, "
    "signals change, or margin gets low."
)

# ── Status banner ──────────────────────────────────────────────────────────────
import os as _os
_tg_token   = _os.environ.get("TELEGRAM_BOT_TOKEN", "")
_tg_chat_id = _os.environ.get("TELEGRAM_CHAT_ID", "")
if _tg_token and _tg_chat_id:
    st.success("✅ Telegram is connected — bot token and chat ID are set.")
else:
    missing = []
    if not _tg_token:   missing.append("`TELEGRAM_BOT_TOKEN`")
    if not _tg_chat_id: missing.append("`TELEGRAM_CHAT_ID`")
    st.warning(
        f"⚠️ Not connected yet. Missing secrets: {', '.join(missing)}\n\n"
        "Follow the setup steps below, then add your token and chat ID as environment variables in your hosting platform."
    )

st.divider()

# ── Setup guide ────────────────────────────────────────────────────────────────
with st.expander("📱 How to set up your Telegram bot (2 min)", expanded=not (_tg_token and _tg_chat_id)):
    st.markdown("""
**Step 1 — Create a bot**
1. Open Telegram and search for **@BotFather**
2. Send the message: `/newbot`
3. Follow the prompts — give it any name (e.g. *Gold Alerts*)
4. BotFather will give you a **Bot Token** like `123456789:ABCdef...`

**Step 2 — Find your Chat ID**
1. Search for **@userinfobot** in Telegram
2. Send `/start`
3. It will reply with your **Chat ID** (a number like `987654321`)

**Step 3 — Add secrets to this app**

*If deployed on Streamlit Community Cloud:*
1. Open your app dashboard, click ⋮ → **Settings** → **Secrets**
2. Add these two lines:
   ```
   TELEGRAM_BOT_TOKEN = "your_bot_token_here"
   TELEGRAM_CHAT_ID = "your_chat_id_here"
   ```
3. Save — the app restarts automatically and picks them up

*If deployed on Railway / Render / Fly.io:*
1. Open your project's **Variables** (or **Environment**) tab
2. Add `TELEGRAM_BOT_TOKEN` and paste your bot token
3. Add `TELEGRAM_CHAT_ID` and paste your chat ID
4. Redeploy — the app picks them up on next start

**Step 4 — Start your bot**
Send any message (e.g. `/start`) to your new bot in Telegram so it can message you back.
""")

st.divider()

# ── Alert master switch ────────────────────────────────────────────────────────
_tg_enabled = st.toggle("Enable all alerts", value=_tg_cfg.get("enabled", True))

st.divider()

# ── Price level settings ───────────────────────────────────────────────────────
st.markdown("### 📍 Price Level Alerts")
st.caption("Set your active stop and limit levels. Alerts fire when price gets within the warning buffer.")

# ── Stale config warning ───────────────────────────────────────────────────
_cfg_stop_raw  = _tg_cfg.get("stop_level")
_cfg_limit_raw = _tg_cfg.get("limit_level")
_ref_price     = _top_live_price or 3200.0
if _cfg_stop_raw or _cfg_limit_raw:
    _stale = []
    if _cfg_stop_raw  and abs(_cfg_stop_raw  - _ref_price) > _ref_price * 0.10:
        _stale.append(f"Stop {_cfg_stop_raw:.0f}")
    if _cfg_limit_raw and abs(_cfg_limit_raw - _ref_price) > _ref_price * 0.10:
        _stale.append(f"Limit {_cfg_limit_raw:.0f}")
    if _stale:
        st.warning(
            f"⚠️ **Stale alert levels detected:** {' & '.join(_stale)} are more than 10% "
            f"from the current price (${_ref_price:,.0f}). These look like values from a "
            f"previous trade. Price-level alerts are **automatically skipped** until you "
            f"update these to match your current open position — or set them to 0 to clear.",
            icon=None
        )

_col_sl, _col_ll, _col_buf = st.columns(3)
with _col_sl:
    _stop_val = st.number_input(
        "Stop Level",
        value=float(_tg_cfg.get("stop_level") or 0.0),
        step=1.0, format="%.2f",
        help="Your stop loss price on IG. Leave 0 to disable."
    )
with _col_ll:
    _limit_val = st.number_input(
        "Limit Level",
        value=float(_tg_cfg.get("limit_level") or 0.0),
        step=1.0, format="%.2f",
        help="Your take-profit price on IG. Leave 0 to disable."
    )
with _col_buf:
    _buf_val = st.number_input(
        "Warning Buffer (pts)",
        value=int(_tg_cfg.get("warning_buffer", 5)),
        min_value=1, max_value=50, step=1,
        help="How many points from stop/limit to trigger an early warning."
    )

_alert_price  = st.checkbox("Price level alerts",   value=_tg_cfg.get("alert_price_levels",   True))

st.divider()

# ── Open position tracking ─────────────────────────────────────────────────────
st.markdown("### 📌 Open Position (Revision Alerts)")
st.caption(
    "Log your current IG position so the system can automatically alert you when "
    "your stop needs widening, the ML flips against your trade, or a bounce risk develops."
)
_col_dir, _col_entry = st.columns(2)
with _col_dir:
    _pos_dir_opts = ["None (no open position)", "LONG", "SHORT"]
    _pos_dir_saved = _tg_cfg.get("position_direction") or "None (no open position)"
    if _pos_dir_saved not in _pos_dir_opts:
        _pos_dir_saved = "None (no open position)"
    _pos_dir = st.selectbox(
        "Direction",
        options=_pos_dir_opts,
        index=_pos_dir_opts.index(_pos_dir_saved),
        help="Set to LONG or SHORT to enable revision alerts for your open trade."
    )
with _col_entry:
    _entry_price = st.number_input(
        "Avg Entry Price",
        value=float(_tg_cfg.get("entry_price") or 0.0),
        step=0.5, format="%.2f",
        help="Your average open price from IG. Used to calculate live P&L in revision alerts."
    )

_alert_revision = st.checkbox(
    "Trade revision alerts (stop too tight / ML flip / stop-zone risk)",
    value=_tg_cfg.get("alert_revision", True),
    help="Fires automatically every cycle when your position needs attention."
)

st.divider()

# ── Signal & margin alerts ─────────────────────────────────────────────────────
st.markdown("### 📊 Other Alerts")
_alert_signal  = st.checkbox("Strong signal change (15m/1H flip)", value=_tg_cfg.get("alert_signal_change",  True))
_alert_margin  = st.checkbox("Low margin warning",                  value=_tg_cfg.get("alert_margin_warning", True))
_alert_retrain = st.checkbox("Model retrain complete",              value=_tg_cfg.get("alert_retrain_done",   False))

_col_margin, _col_cool = st.columns(2)
with _col_margin:
    _margin_thresh = st.number_input(
        "Margin warning below (A$)",
        value=float(_tg_cfg.get("margin_threshold", 300)),
        min_value=50.0, max_value=2000.0, step=50.0
    )
with _col_cool:
    _cooldown = st.number_input(
        "Alert cooldown (min)",
        value=int(_tg_cfg.get("cooldown_minutes", 15)),
        min_value=1, max_value=120, step=5,
        help="Minimum minutes between repeat alerts of the same type."
    )

st.divider()

# ── Save + Test buttons ────────────────────────────────────────────────────────
_btn_save, _btn_test = st.columns(2)
with _btn_save:
    if st.button("💾  Save Settings", use_container_width=True, type="primary"):
        _pos_dir_save = None if _pos_dir.startswith("None") else _pos_dir
        _new_cfg = {
            "enabled":              _tg_enabled,
            "stop_level":          _stop_val   if _stop_val  > 0 else None,
            "limit_level":         _limit_val  if _limit_val > 0 else None,
            "warning_buffer":      int(_buf_val),
            "alert_price_levels":  _alert_price,
            "alert_signal_change": _alert_signal,
            "alert_margin_warning":_alert_margin,
            "alert_retrain_done":  _alert_retrain,
            "margin_threshold":    float(_margin_thresh),
            "cooldown_minutes":    int(_cooldown),
            "position_direction":  _pos_dir_save,
            "entry_price":         _entry_price if _entry_price > 0 else None,
            "alert_revision":      _alert_revision,
        }
        _tg_save(_new_cfg)
        st.success("✅ Settings saved.")

with _btn_test:
    if st.button("📨  Send Test Message", use_container_width=True):
        _ok, _msg = _tg_test()
        if _ok:
            st.success(f"✅ {_msg}")
        else:
            st.error(f"❌ {_msg}")

_sub_alerts.__exit__(None, None, None)

_tab_tools.__exit__(None, None, None)

# ─────────────────────────────────────────────
# Empty state
# ─────────────────────────────────────────────
if results is None:
    st.info(
        "⏳ The background scheduler is running its first backtest — this takes "
        "about 15–20 minutes on first run (downloading ~160 data series + training). "
        "Check the **Background Scheduler** status in the sidebar. "
        "Once complete, refresh the page or click **Reload from scheduler** and results "
        "will appear instantly every time after that.\n\n"
        "You can also click **Run backtest now** to run a manual backtest in this window."
    )

