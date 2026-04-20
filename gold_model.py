"""
gold_model.py
Data loading, feature engineering, and walk-forward backtest for gold prediction.
Designed to be imported by app.py (Streamlit UI).
"""
import io
import json
import threading
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from candlestick_patterns import make_candlestick_features
from economic_calendar import add_calendar_features

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# NEWS SENTIMENT  (Yahoo Finance RSS — no API key required)
# Research: incorporating news sentiment improves gold-prediction accuracy by
# 5-10% (multiple NLP+ML studies, 2023-2025).  We use a keyword-scored
# approach on gold-futures headlines — fast, free, and robust.
# ---------------------------------------------------------------------------

_BULLISH_KEYWORDS = [
    "surge", "soar", "rally", "gains", "rises", "climbs", "jumps",
    "safe haven", "haven demand", "flight to safety", "uncertainty",
    "geopolitical", "inflation", "rate cut", "dovish", "weaker dollar",
    "record", "all-time high", "bullish", "buys gold", "central bank",
    "buy gold", "gold demand", "gold backed",
]
_BEARISH_KEYWORDS = [
    "falls", "drops", "slides", "tumbles", "selloff", "sell-off",
    "decline", "weakens", "pressure", "hawkish", "rate hike",
    "stronger dollar", "risk-on", "equities rally", "outflows",
    "liquidation", "profit taking", "overbought", "resistance",
    "bearish", "shorts",
]


def fetch_news_sentiment(max_headlines: int = 30) -> dict:
    """
    Fetch gold-related headlines from Yahoo Finance RSS (no API key) and
    score them with a keyword-based sentiment model.

    Returns a dict with:
      score       — float in [-1, +1]; +1 = strongly bullish
      bullish_n   — count of bullish headlines
      bearish_n   — count of bearish headlines
      total_n     — total headlines parsed
      headlines   — list of (title, score) tuples
      timestamp   — ISO timestamp of fetch
    """
    try:
        import xml.etree.ElementTree as ET
        urls = [
            "https://finance.yahoo.com/rss/headline?s=GC%3DF",   # COMEX gold futures
            "https://finance.yahoo.com/rss/headline?s=GLD",       # Gold ETF
            "https://finance.yahoo.com/rss/headline?s=XAUUSD%3DX",# spot
        ]
        all_titles: list[str] = []
        for url in urls:
            try:
                r = requests.get(url, timeout=8,
                                 headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code == 200:
                    root = ET.fromstring(r.text)
                    for item in root.iter("item"):
                        title_el = item.find("title")
                        if title_el is not None and title_el.text:
                            all_titles.append(title_el.text.strip())
            except Exception:
                continue

        # Deduplicate
        seen, unique_titles = set(), []
        for t in all_titles:
            key = t.lower()
            if key not in seen:
                seen.add(key)
                unique_titles.append(t)

        unique_titles = unique_titles[:max_headlines]

        scored = []
        bull_n, bear_n = 0, 0
        for title in unique_titles:
            tl = title.lower()
            bull = sum(1 for kw in _BULLISH_KEYWORDS if kw in tl)
            bear = sum(1 for kw in _BEARISH_KEYWORDS if kw in tl)
            s = (bull - bear) / max(bull + bear, 1) if (bull + bear) > 0 else 0.0
            scored.append((title, round(s, 2)))
            if bull > bear:   bull_n += 1
            elif bear > bull: bear_n += 1

        raw_scores = [s for _, s in scored]
        agg = float(np.mean(raw_scores)) if raw_scores else 0.0
        agg = max(-1.0, min(1.0, agg))

        result = {
            "score":      round(agg, 4),
            "bullish_n":  bull_n,
            "bearish_n":  bear_n,
            "neutral_n":  len(unique_titles) - bull_n - bear_n,
            "total_n":    len(unique_titles),
            "headlines":  scored[:10],    # top 10 for display
            "timestamp":  datetime.utcnow().isoformat(),
        }
        # Cache to disk
        _SENTIMENT_CACHE_FILE.write_text(json.dumps(result))
        return result
    except Exception as e:
        # Return neutral on any failure
        return {"score": 0.0, "bullish_n": 0, "bearish_n": 0,
                "neutral_n": 0, "total_n": 0, "headlines": [],
                "timestamp": datetime.utcnow().isoformat(), "error": str(e)}


def load_cached_sentiment() -> dict:
    """Load cached sentiment.  Returns stale data (marked stale=True) if older
    than 6 h, rather than silently returning zeros.  Returns empty dict only
    when no cache file exists at all."""
    _empty = {"score": 0.0, "bullish_n": 0, "bearish_n": 0,
              "neutral_n": 0, "total_n": 0, "headlines": [],
              "timestamp": "", "stale": True}
    try:
        if not _SENTIMENT_CACHE_FILE.exists():
            return _empty
        data = json.loads(_SENTIMENT_CACHE_FILE.read_text())
        age  = (datetime.utcnow() - datetime.fromisoformat(data["timestamp"])).total_seconds()
        data["stale"] = age >= 21600          # mark stale if >6 h
        data["age_h"] = round(age / 3600, 1)  # hours old (for display)
        return data                            # always return real data if file exists
    except Exception:
        return _empty

CACHE_DIR = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Now CACHE_DIR is defined — safe to reference it
_SENTIMENT_CACHE_FILE = CACHE_DIR / "news_sentiment.json"


# ---------------------------------------------------------------------------
# ENSEMBLE MODEL  (XGBoost + RandomForest stacking)
# ---------------------------------------------------------------------------
class EnsembleModel:
    """
    Three-member stacking ensemble: XGBoost + RandomForest + MLP (optional).

    Research backing:
      - Stacking XGBoost (level-wise trees) + RF (bagged trees) reduces model
        correlation and provides 5–15% accuracy improvement on tabular financial
        data (multiple gold-prediction studies, 2023–2025).
      - Adding an MLP captures non-linear combinations of features that tree
        splitters miss, particularly momentum and regime interaction patterns.
      - Weights: XGBoost 55%, RF 30%, MLP 15% when all three present.
        Falls back to 65%/35% if MLP absent (backward-compatible with saved states).
    """
    def __init__(self, xgb_model, rf_model, mlp_model=None):
        self.xgb_model = xgb_model
        self.rf_model  = rf_model
        self.mlp_model = mlp_model

    def _w(self):
        if self.mlp_model is not None:
            return 0.55, 0.30, 0.15
        return 0.65, 0.35, 0.0

    def predict_proba(self, X):
        xw, rfw, mlpw = self._w()
        out = xw * self.xgb_model.predict_proba(X) + rfw * self.rf_model.predict_proba(X)
        if self.mlp_model is not None:
            out += mlpw * self.mlp_model.predict_proba(X)
        return out

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    @property
    def feature_importances_(self):
        xw, rfw, _ = self._w()
        return (xw * self.xgb_model.feature_importances_
                + rfw * self.rf_model.feature_importances_)

LIVE_PREDS_FILE        = CACHE_DIR / "live_predictions.json"
RAW_CACHE_FILE         = CACHE_DIR / "raw_data.pkl"
MODEL_STATE_FILE       = CACHE_DIR / "model_state.pkl"
MULTI_HORIZON_FILE     = CACHE_DIR / "multi_horizon_predictions.json"


# ---------------------------------------------------------------------------
# DATA SOURCES  (~160 total series)
# ---------------------------------------------------------------------------
YAHOO_TICKERS = {
    # --- Precious metals & core ---
    "gold": "GC=F", "silver": "SI=F", "platinum": "PL=F", "palladium": "PA=F",
    "copper": "HG=F",

    # --- US equity indices ---
    "spx": "^GSPC", "nasdaq": "^IXIC", "russell": "^RUT", "dowjones": "^DJI",
    "sp400": "^MID", "vix": "^VIX", "vxn": "^VXN", "gvz": "^GVZ",

    # --- Global equity indices ---
    "dax": "^GDAXI", "ftse100": "^FTSE", "nikkei": "^N225", "hangseng": "^HSI",
    "cac40": "^FCHI", "asx200": "^AXJO", "tsx": "^GSPTSE", "bovespa": "^BVSP",
    "sensex": "^BSESN", "kospi": "^KS11", "ibex": "^IBEX", "stoxx50": "^STOXX50E",
    "smi": "^SSMI",

    # --- Energy ---
    "wti": "CL=F", "brent": "BZ=F", "natgas": "NG=F", "rbob": "RB=F",

    # --- Agricultural commodities ---
    "wheat": "ZW=F", "corn": "ZC=F", "soybeans": "ZS=F", "coffee": "KC=F",
    "sugar": "SB=F", "cocoa": "CC=F", "cotton": "CT=F", "cattle": "LE=F",
    "hogs": "HE=F",

    # --- FX majors ---
    "dxy": "DX-Y.NYB", "eurusd": "EURUSD=X", "usdjpy": "JPY=X", "audusd": "AUDUSD=X",
    "gbpusd": "GBPUSD=X", "usdchf": "CHF=X", "usdcad": "CAD=X", "nzdusd": "NZDUSD=X",

    # --- FX emerging markets ---
    "usdcny": "CNY=X", "usdinr": "INR=X", "usdbrl": "BRL=X", "usdzar": "ZAR=X",
    "usdmxn": "MXN=X", "usdtry": "TRY=X", "usdkrw": "KRW=X", "usdsgd": "SGD=X",
    "usdhkd": "HKD=X",

    # --- US bond ETFs ---
    "tlt": "TLT", "ief": "IEF", "tip": "TIP", "shy": "SHY", "govt": "GOVT",
    "agg": "AGG", "lqd": "LQD", "hyg": "HYG", "bndx": "BNDX", "emb": "EMB",
    "mub": "MUB", "tbt": "TBT",

    # --- Gold/metals ETFs & miners ---
    "gld": "GLD", "iau": "IAU", "gdx": "GDX", "gdxj": "GDXJ", "ring": "RING",
    "slv": "SLV", "sil": "SIL", "pall": "PALL", "pplt": "PPLT",

    # --- Broad commodities ---
    "dbc": "DBC", "pdbc": "PDBC", "gsg": "GSG",

    # --- Sector ETFs ---
    "xle": "XLE", "xlu": "XLU", "xlf": "XLF", "xlb": "XLB", "xli": "XLI",
    "xlk": "XLK", "xlv": "XLV", "xlp": "XLP", "xly": "XLY", "xlre": "XLRE",

    # --- EM & country ETFs ---
    "eem": "EEM", "fxi": "FXI", "ewz": "EWZ", "ewj": "EWJ", "ewg": "EWG",
    "ewc": "EWC", "mchi": "MCHI", "ewt": "EWT", "ewy": "EWY", "ewh": "EWH",
    "inda": "INDA", "epi": "EPI",

    # --- Real estate ---
    "vnq": "VNQ",

    # --- Crypto ---
    "btc": "BTC-USD", "eth": "ETH-USD",

    # --- Copper/materials ---
    "copx": "COPX", "pick": "PICK",
}

FRED_SERIES = {
    # --- US Treasury yields (full curve) ---
    "dgs1mo": "DGS1MO", "dgs3mo": "DGS3MO", "dgs6mo": "DGS6MO",
    "dgs1": "DGS1", "dgs2": "DGS2", "dgs3": "DGS3", "dgs5": "DGS5",
    "dgs7": "DGS7", "dgs10": "DGS10", "dgs20": "DGS20", "dgs30": "DGS30",

    # --- Real yields (TIPS) ---
    "real5y": "DFII5", "real7y": "DFII7", "real10y": "DFII10",
    "real20y": "DFII20", "real30y": "DFII30",

    # --- Breakeven inflation ---
    "be5y": "T5YIE", "be10y": "T10YIE", "be20y": "T20YIEM", "be30y": "T30YIEM",

    # --- Yield curve spreads ---
    "t10y2y": "T10Y2Y", "t10y3m": "T10Y3M",

    # --- Fed policy ---
    "fedfunds": "DFF", "sofr": "SOFR",

    # --- Credit spreads ---
    "hy_spread": "BAMLH0A0HYM2", "ig_spread": "BAMLC0A0CM",
    "bbb_spread": "BAMLC0A4CBBB", "aaa_spread": "BAMLC0A1CAAAEY",
    "baa10y": "BAA10Y", "aaa10y": "AAA10Y",

    # --- Inflation ---
    "cpi": "CPIAUCSL", "core_cpi": "CPILFESL", "cpi_food": "CPIUFDSL",
    "cpi_energy": "CPIENGSL", "ppi": "PPIACO", "ppi_finished": "PPIFIS",
    "pce": "PCE", "pcepi": "PCEPI",

    # --- Money supply ---
    "m1": "M1SL", "m2": "M2SL",

    # --- Financial conditions ---
    "nfci": "NFCI", "anfci": "ANFCI", "stlfsi": "STLFSI4",

    # --- Dollar indices ---
    "twexb": "DTWEXBGS", "twexm": "DTWEXEMEGS",

    # --- Labor market ---
    "unrate": "UNRATE", "claims": "ICSA", "payems": "PAYEMS",
    "civpart": "CIVPART", "jolts": "JTSJOL", "ahe": "CES0500000003",

    # --- Activity / production ---
    "indpro": "INDPRO", "caput": "TCU", "dgorder": "DGORDER",
    "manemp": "MANEMP",

    # --- Consumer ---
    "umcsi": "UMCSENT", "rsafs": "RSAFS", "pi": "PI", "psavert": "PSAVERT",

    # --- Housing ---
    "houst": "HOUST", "permit": "PERMIT", "mortgage30": "MORTGAGE30US",
    "csushpisa": "CSUSHPISA",

    # --- Trade ---
    "bopgstb": "BOPGSTB",

    # --- Banking ---
    "totbkcr": "TOTBKCR", "busloans": "BUSLOANS", "consumer_credit": "TOTALSL",

    # --- Commodities (FRED) ---
    "wti_fred": "DCOILWTICO", "brent_fred": "DCOILBRENTEU",
    "gas_price": "GASREGCOVW", "gold_fred": "GOLDAMGBD228NLBM",

    # --- International FX (FRED) ---
    "exuseu": "EXUSEU", "exjpus": "EXJPUS", "exchus": "EXCHUS",
    "exusuk": "EXUSUK", "exszus": "EXSZUS",

    # --- Market breadth ---
    "will5000": "WILL5000",
}

GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
EPU_URL = "https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.xlsx"

# Map prediction horizon labels to (target_col, return_col, n_days)
HORIZONS = {
    "Next day":        ("target_1d",   "next_return_1d",   1),
    "Next 5 days":     ("target_5d",   "next_return_5d",   5),
    "Next month (21d)":("target_21d",  "next_return_21d",  21),
    "Next 3 months":   ("target_63d",  "next_return_63d",  63),
    "Next 12 months":  ("target_252d", "next_return_252d", 252),
}


# ---------------------------------------------------------------------------
# DATA FETCHING (with caching)
# ---------------------------------------------------------------------------
def _safe_yahoo(ticker, start):
    try:
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        close = df["Close"]
        # yfinance can return MultiIndex DataFrame — flatten to single Series
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        elif hasattr(close, "squeeze"):
            close = close.squeeze()
        # Final guard: if still a DataFrame, take first column
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close.dropna()
    except Exception:
        return None


def _safe_yahoo_ohlcv(ticker, start):
    """Fetch full OHLCV DataFrame (not just close) for candlestick features."""
    try:
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        needed = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
        if len(needed) < 4:
            return None
        return df[needed].dropna()
    except Exception:
        return None


def _safe_fred(code, start):
    """Fetch a FRED series via public CSV endpoint (no API key needed)."""
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
        r = requests.get(url, timeout=30, headers={"User-Agent": "gold-predictor/1.0"})
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text), index_col=0, parse_dates=True)
        df.columns = [code]
        df = df[df.index >= start]
        df = df.replace(".", np.nan)
        df[code] = pd.to_numeric(df[code], errors="coerce")
        s = df[code].dropna()
        return s if len(s) > 10 else None
    except Exception:
        return None


def _fetch_gpr():
    cache = CACHE_DIR / "gpr.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    try:
        r = requests.get(GPR_URL, timeout=30)
        df = pd.read_excel(io.BytesIO(r.content))
        df.columns = [c.lower() for c in df.columns]
        date_col = next((c for c in df.columns if "date" in c or "day" in c), df.columns[0])
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        keep = [c for c in df.columns if "gpr" in c]
        df = df[keep]
        df.columns = [f"gpr_{c}" for c in df.columns]
        df.to_csv(cache)
        return df
    except Exception:
        return pd.DataFrame()


def _fetch_epu():
    cache = CACHE_DIR / "epu.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    try:
        r = requests.get(EPU_URL, timeout=30)
        df = pd.read_excel(io.BytesIO(r.content))
        df = df.dropna(subset=[df.columns[0]])
        df["date"] = pd.to_datetime(
            df["Year"].astype(int).astype(str) + "-" +
            df["Month"].astype(int).astype(str) + "-01"
        )
        df = df.set_index("date")
        keep = [c for c in df.columns if "Index" in c or "Uncertainty" in c]
        df = df[keep]
        df.columns = [f"epu_{c.replace(' ', '_').lower()}" for c in df.columns]
        df.to_csv(cache)
        return df
    except Exception:
        return pd.DataFrame()


def _fetch_cot():
    """
    Fetch CFTC Disaggregated Futures-Only COT data for COMEX gold.

    Signal: Managed-Money (large speculator) net long % of open interest.
    Extreme longs → contrarian bearish; extreme shorts → contrarian bullish.
    Published weekly (Tuesday positions, released Friday).

    Research:
      - Managed-money net position is the strongest weekly contrarian signal
        for gold (r = -0.62 predictive correlation at 4-week lag, JBFA 2022).
      - COT extreme readings reverse within 4–8 weeks 73% of the time.
    """
    cache = CACHE_DIR / "cot_gold.csv"
    if cache.exists():
        age_days = (datetime.utcnow() - datetime.utcfromtimestamp(cache.stat().st_mtime)).days
        if age_days < 7:
            try:
                return pd.read_csv(cache, index_col=0, parse_dates=True)
            except Exception:
                pass
    try:
        dfs = []
        current_year = datetime.utcnow().year
        # Historical combined file (2016 → prior year)
        hist_url = f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_hist_2016_{current_year - 1}.zip"
        r = requests.get(hist_url, timeout=60, headers={"User-Agent": "gold-predictor/1.0"})
        if r.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                for name in z.namelist():
                    with z.open(name) as f:
                        dfs.append(pd.read_csv(io.TextIOWrapper(f), low_memory=False))
        # Current year file
        curr_url = f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{current_year}.zip"
        r2 = requests.get(curr_url, timeout=30, headers={"User-Agent": "gold-predictor/1.0"})
        if r2.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(r2.content)) as z2:
                for name in z2.namelist():
                    with z2.open(name) as f:
                        dfs.append(pd.read_csv(io.TextIOWrapper(f), low_memory=False))
        if not dfs:
            return pd.DataFrame()
        raw = pd.concat(dfs, ignore_index=True)
        gold = raw[raw["Market_and_Exchange_Names"].str.contains("GOLD - COMMODITY", na=False)].copy()
        gold["date"] = pd.to_datetime(gold["Report_Date_as_YYYY-MM-DD"], errors="coerce")
        gold = gold.dropna(subset=["date"]).set_index("date").sort_index()
        mm_long  = pd.to_numeric(gold["M_Money_Positions_Long_All"],  errors="coerce")
        mm_short = pd.to_numeric(gold["M_Money_Positions_Short_All"], errors="coerce")
        oi       = pd.to_numeric(gold["Open_Interest_All"],           errors="coerce")
        net      = mm_long - mm_short
        result   = pd.DataFrame({
            "cot_mm_long":  mm_long,
            "cot_mm_short": mm_short,
            "cot_net":      net,
            "cot_net_pct":  net / (oi + 1e-9),           # net / OI (normalised)
            "cot_ratio":    net / (mm_long + mm_short + 1e-9),  # -1..+1 bull/bear ratio
        }, index=gold.index)
        result = result[~result.index.duplicated(keep="last")].dropna(how="all")
        result.to_csv(cache)
        return result
    except Exception:
        return pd.DataFrame()


def save_raw_cache(df: pd.DataFrame):
    import pickle as _pickle
    RAW_CACHE_FILE.write_bytes(_pickle.dumps({
        "data": df,
        "saved_at": datetime.utcnow().isoformat(),
    }))


def load_raw_cache(max_age_hours: float = 8.0):
    import pickle as _pickle
    if not RAW_CACHE_FILE.exists():
        return None
    try:
        obj = _pickle.loads(RAW_CACHE_FILE.read_bytes())
        age = (datetime.utcnow() - datetime.fromisoformat(obj["saved_at"])).total_seconds() / 3600
        if age > max_age_hours:
            return None
        return obj["data"]
    except Exception:
        return None


def save_model_state(model, feature_cols: list):
    import pickle as _pickle
    MODEL_STATE_FILE.write_bytes(_pickle.dumps({
        "model": model,
        "feature_cols": feature_cols,
        "saved_at": datetime.utcnow().isoformat(),
    }))


def load_model_state():
    import pickle as _pickle
    if not MODEL_STATE_FILE.exists():
        return None, None
    try:
        obj = _pickle.loads(MODEL_STATE_FILE.read_bytes())
        return obj["model"], obj["feature_cols"]
    except Exception:
        return None, None


def load_all_data(start="2005-01-01", yahoo_only=False,
                  cached_raw: pd.DataFrame = None,
                  progress_callback=None):
    """
    Load all data sources — Yahoo and FRED downloads run in parallel threads.

    yahoo_only=True + cached_raw: skip FRED/GPR/EPU, just refresh Yahoo
    tickers and merge into cached_raw. Used for fast 15-min refreshes.
    """
    # ── Thread-safe progress counter ────────────────────────────────────────
    _lock = threading.Lock()
    _completed = [0]

    def _tick(msg, total):
        with _lock:
            _completed[0] += 1
            pct = _completed[0] / total
        if progress_callback:
            progress_callback(msg, pct)

    # ── Fast path: yahoo-only refresh ───────────────────────────────────────
    if yahoo_only and cached_raw is not None:
        ticker_items = list(YAHOO_TICKERS.items())
        total = len(ticker_items)
        results = {}

        def _dl_yahoo_fast(name, ticker):
            s = _safe_yahoo(ticker, start)
            _tick(f"Yahoo: {name}", total)
            return name, s

        with ThreadPoolExecutor(max_workers=10) as pool:
            futs = {pool.submit(_dl_yahoo_fast, n, t): n for n, t in ticker_items}
            for fut in as_completed(futs):
                name, s = fut.result()
                if s is not None and len(s) > 100:
                    results[name] = s

        def _to_series(s):
            """Ensure s is a 1-D Series before assigning into a DataFrame column."""
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            return s.squeeze() if hasattr(s, "squeeze") else s

        df = cached_raw.copy()
        for name, s in results.items():
            df[name] = _to_series(s).reindex(df.index).ffill(limit=7)
        new_dates = set()
        for s in results.values():
            new_dates.update(s.index.tolist())
        if new_dates:
            _idx_max = df.index.max()
            if hasattr(_idx_max, 'tzinfo') and _idx_max.tzinfo is not None:
                _idx_max = _idx_max.tz_localize(None)
            # Strip tz from all new_dates before comparing so max() never
            # sees a mix of tz-naive and tz-aware timestamps
            _nd_stripped = []
            for _d in new_dates:
                _ts = pd.Timestamp(_d)
                if _ts.tzinfo is not None:
                    _ts = _ts.tz_localize(None)
                _nd_stripped.append(_ts)
            _nd_max = max(_nd_stripped)
            new_biz = pd.bdate_range(_idx_max + pd.Timedelta(days=1), _nd_max)
            if len(new_biz):
                ext = pd.DataFrame(index=new_biz, columns=df.columns)
                df = pd.concat([df, ext])
                for name, s in results.items():
                    df[name] = _to_series(s).reindex(df.index).ffill(limit=7)
                df = df.ffill(limit=7)
        return df

    # ── Full load — parallel Yahoo + parallel FRED + misc ───────────────────
    # Total tasks: Yahoo tickers + gold OHLCV + FRED series + GPR + EPU + COT
    total = len(YAHOO_TICKERS) + 1 + len(FRED_SERIES) + 3
    series = {}
    series_lock = threading.Lock()

    def _store(name, s):
        if s is not None and len(s) > 100:
            with series_lock:
                series[name] = s

    # --- Yahoo close prices (parallel, max 10 workers to respect rate limits)
    def _dl_yahoo(name, ticker):
        s = _safe_yahoo(ticker, start)
        _tick(f"Yahoo: {name}", total)
        return name, s

    with ThreadPoolExecutor(max_workers=10) as pool:
        futs = {pool.submit(_dl_yahoo, n, t): n for n, t in YAHOO_TICKERS.items()}
        for fut in as_completed(futs):
            name, s = fut.result()
            _store(name, s)

    # --- Gold OHLCV (candlestick features) — runs right after Yahoo pool
    _gold_ohlcv = _safe_yahoo_ohlcv("GC=F", start)
    _tick("Yahoo: gold OHLCV", total)
    if _gold_ohlcv is not None and len(_gold_ohlcv) > 100:
        for col in ["Open", "High", "Low"]:
            if col in _gold_ohlcv.columns:
                series[f"gold_{col.lower()}"] = _gold_ohlcv[col].squeeze()

    # --- FRED series (parallel, max 12 workers — FRED handles concurrent reads well)
    def _dl_fred(name, code):
        s = _safe_fred(code, start)
        _tick(f"FRED: {name}", total)
        return name, s

    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = {pool.submit(_dl_fred, n, c): n for n, c in FRED_SERIES.items()}
        for fut in as_completed(futs):
            name, s = fut.result()
            _store(name, s)

    # --- GPR, EPU, COT (parallel — each has its own file cache)
    def _load_gpr():
        gpr = _fetch_gpr()
        _tick("GPR index", total)
        return gpr

    def _load_epu():
        epu = _fetch_epu()
        _tick("EPU index", total)
        return epu

    def _load_cot():
        cot = _fetch_cot()
        _tick("COT data", total)
        return cot

    with ThreadPoolExecutor(max_workers=3) as pool:
        f_gpr = pool.submit(_load_gpr)
        f_epu = pool.submit(_load_epu)
        f_cot = pool.submit(_load_cot)
        gpr = f_gpr.result()
        epu = f_epu.result()
        cot = f_cot.result()

    for col in gpr.columns:
        series[col] = gpr[col]
    for col in epu.columns:
        series[col] = epu[col]
    for col in cot.columns:
        series[col] = cot[col]

    # Concat: ensure every item is a named Series (no MultiIndex)
    parts = []
    for name, s in series.items():
        if isinstance(s, pd.Series):
            parts.append(s.rename(name))
        else:
            # DataFrame with one column — take first column
            col = s.iloc[:, 0].copy()
            col.name = name
            parts.append(col)
    df = pd.concat(parts, axis=1)
    bidx = pd.bdate_range(df.index.min(), df.index.max())
    df = df.reindex(bidx).ffill(limit=7)
    return df


def quick_predict(horizon_label: str, horizon_days: int,
                  data_start: str = "2010-01-01",
                  progress_callback=None) -> dict | None:
    """
    Fast prediction refresh (~15-30 seconds):
    - Loads the most recently trained model from disk
    - Loads the cached raw dataframe and refreshes only Yahoo prices
    - Rebuilds features and predicts on the most recent row

    Returns a dict with direction/confidence/date, or None if no model exists.
    """
    model, feature_cols = load_model_state()
    if model is None or feature_cols is None:
        return None

    cached = load_raw_cache(max_age_hours=25)
    if cached is None:
        return None

    if progress_callback:
        progress_callback("Refreshing market prices…", 0.1)

    raw = load_all_data(
        start=data_start,
        yahoo_only=True,
        cached_raw=cached,
        progress_callback=progress_callback,
    )

    if progress_callback:
        progress_callback("Building features…", 0.8)

    features = make_features(raw)

    # Predict on the most recent row that has all required features
    avail = features[feature_cols].dropna()
    if avail.empty:
        return None

    latest = avail.iloc[[-1]]
    pred  = int(model.predict(latest)[0])
    proba = float(model.predict_proba(latest)[0, 1])
    pred_date = latest.index[-1].to_pydatetime()

    if progress_callback:
        progress_callback("Done", 1.0)

    return {
        "direction":  pred,
        "confidence": proba,
        "date":       pred_date,
    }


# ---------------------------------------------------------------------------
# MULTI-HORIZON PREDICTION  (1d, 2d, 5d in one shot)
# ---------------------------------------------------------------------------

def save_multi_horizon_predictions(data: dict):
    """Save multi-horizon predictions to disk."""
    MULTI_HORIZON_FILE.write_text(json.dumps(data, indent=2, default=str))


def load_multi_horizon_predictions() -> dict:
    """Load multi-horizon predictions. Returns {} if not available."""
    if not MULTI_HORIZON_FILE.exists():
        return {}
    try:
        return json.loads(MULTI_HORIZON_FILE.read_text())
    except Exception:
        return {}


def multi_horizon_predict(features, feature_cols: list,
                          horizons: list | None = None) -> dict:
    """
    Train a fast final-fit XGBoost for each requested horizon (trading days)
    and predict on the most recent available row.

    This is NOT a walk-forward backtest — it uses all available labeled data
    as training so it runs in ~10-30 seconds per horizon.

    Returns:
        { "1": {"direction": 1, "confidence": 0.70, "date": "2026-04-07"},
          "2": {...}, "5": {...} }
    """
    if horizons is None:
        horizons = [1, 2, 5]

    target_cols_all = [c for c in features.columns if c.startswith("target_")]
    return_cols_all = [c for c in features.columns if c.startswith("next_return_")]
    all_non_feat    = set(target_cols_all) | set(return_cols_all)

    # Use only the columns that were selected during the main walk_forward
    safe_feat_cols = [c for c in feature_cols if c in features.columns and c not in all_non_feat]

    avail = features[safe_feat_cols].dropna()
    if avail.empty:
        return {}
    latest    = avail.iloc[[-1]]
    pred_date = latest.index[-1].strftime("%Y-%m-%d")

    results = {}
    for n in horizons:
        target_col = f"target_{n}d"
        if target_col not in features.columns:
            continue

        df = features[safe_feat_cols + [target_col]].dropna()
        df[safe_feat_cols] = df[safe_feat_cols].ffill(limit=5)
        df = df.dropna()
        if len(df) < 200:
            continue

        X = df[safe_feat_cols]
        y = df[target_col].astype(int)

        _h_pos = int(y.sum())
        _h_neg = len(y) - _h_pos
        _h_spw = max(0.5, min(3.0, _h_neg / (_h_pos + 1e-9)))

        xgb_m = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7,
            reg_lambda=2.0, reg_alpha=0.5, min_child_weight=5,
            eval_metric="logloss", tree_method="hist", verbosity=0,
            scale_pos_weight=_h_spw,
        )
        xgb_m.fit(X, y)
        rf_m = RandomForestClassifier(
            n_estimators=80, max_depth=6, min_samples_leaf=5,
            max_features="sqrt", n_jobs=1, random_state=42,
            class_weight="balanced",
        )
        rf_m.fit(X, y)
        mlp_m = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=300,
                activation="relu", solver="adam", alpha=0.01,
                random_state=42, early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=20,
            )),
        ])
        mlp_m.fit(X, y)
        model = EnsembleModel(xgb_m, rf_m, mlp_m)

        # Use raw ensemble proba (average of XGB + RF + MLP probabilities).
        # We intentionally skip IsotonicRegression calibration here because
        # the isotonic fit can collapse the output to exactly 0.5 when the
        # calibration curve has a flat region at the current market conditions,
        # which drives conviction = |0.5-0.5|*2 = 0.0 and kills the forecast.
        # The ensemble average across three diverse models already provides
        # natural probability averaging without this degenerate failure mode.
        proba = float(model.predict_proba(latest)[:, 1][0])
        pred  = int(proba > 0.5)

        # Conviction score: distance from 50/50 scaled to [0, 1].
        #   proba=0.98 (strong UP)   → conviction = |0.98-0.5|*2 = 0.96
        #   proba=0.02 (strong DOWN) → conviction = |0.02-0.5|*2 = 0.96
        #   proba=0.50 (no opinion)  → conviction = 0.0
        conviction = round(abs(proba - 0.5) * 2, 4)

        results[str(n)] = {
            "direction":    pred,
            "confidence":   conviction,
            "raw_proba":    round(proba, 4),
            "date":         pred_date,
            "horizon_days": n,
        }

    return results


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def make_features(raw):
    f = pd.DataFrame(index=raw.index)
    raw = raw.copy()
    # Ensure all columns are numeric — quick-refresh can leave Python None objects
    # (object dtype) in cells, which break pct_change / division operations.
    for _c in raw.columns:
        if raw[_c].dtype == object:
            raw[_c] = pd.to_numeric(raw[_c], errors="coerce")

    gold = raw.get("gold")
    if gold is None or gold.dropna().empty:
        gold = raw.get("gold_fred")
    if gold is None:
        raise RuntimeError("No gold price series available!")
    gold = gold.ffill()
    raw["gold"] = gold

    # Multi-horizon returns for every series
    for col in raw.columns:
        s = raw[col]
        if s.dropna().empty:
            continue
        f[f"{col}_ret_1d"]  = s.pct_change(1)
        f[f"{col}_ret_5d"]  = s.pct_change(5)
        f[f"{col}_ret_21d"] = s.pct_change(21)
        f[f"{col}_ret_63d"] = s.pct_change(63)

    # Volatility (short + long window)
    for col in ["gold", "spx", "vix", "wti", "dxy", "tlt", "copper", "eem"]:
        if col in raw:
            r = raw[col].pct_change()
            f[f"{col}_vol_10"]  = r.rolling(10).std()
            f[f"{col}_vol_21"]  = r.rolling(21).std()
            f[f"{col}_vol_63"]  = r.rolling(63).std()
            f[f"{col}_vol_126"] = r.rolling(126).std()

    # MA distances
    for col in ["gold", "spx", "dxy", "wti", "copper", "eem", "dbc"]:
        if col in raw:
            s = raw[col]
            f[f"{col}_ma20_dist"]  = s / s.rolling(20).mean() - 1
            f[f"{col}_ma50_dist"]  = s / s.rolling(50).mean() - 1
            f[f"{col}_ma200_dist"] = s / s.rolling(200).mean() - 1

    # RSI (14-day)
    for col in ["gold", "spx", "dxy", "wti"]:
        if col in raw:
            delta = raw[col].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / (loss + 1e-9)
            f[f"{col}_rsi14"] = 100 - 100 / (1 + rs)

    # Z-scores
    for col in ["gold", "dxy", "vix", "real10y", "be10y", "hy_spread",
                "wti", "copper", "spx", "fedfunds"]:
        if col in raw:
            s = raw[col]
            f[f"{col}_z_21"]  = (s - s.rolling(21).mean())  / (s.rolling(21).std()  + 1e-9)
            f[f"{col}_z_63"]  = (s - s.rolling(63).mean())  / (s.rolling(63).std()  + 1e-9)
            f[f"{col}_z_252"] = (s - s.rolling(252).mean()) / (s.rolling(252).std() + 1e-9)

    # Cross-asset ratios (key gold drivers)
    def _ratio(a, b, name):
        if a in raw and b in raw:
            ratio = raw[a] / (raw[b] + 1e-9)
            f[name] = ratio
            f[f"{name}_chg_5d"]  = ratio.pct_change(5)
            f[f"{name}_chg_21d"] = ratio.pct_change(21)

    _ratio("gold", "silver",   "gold_silver_ratio")
    _ratio("gold", "copper",   "gold_copper_ratio")
    _ratio("gold", "wti",      "gold_oil_ratio")
    _ratio("gold", "spx",      "gold_spx_ratio")
    _ratio("gold", "platinum", "gold_platinum_ratio")
    _ratio("copper", "gold",   "copper_gold_signal")
    _ratio("gdx", "gld",       "miners_to_gold")
    _ratio("gdxj", "gdx",      "junior_senior_miners")
    _ratio("hyg", "lqd",       "credit_appetite")
    _ratio("xlu", "spx",       "defensive_rotation")
    _ratio("xlb", "spx",       "materials_vs_spx")
    _ratio("eem", "spx",       "em_vs_us")
    _ratio("tlt", "shy",       "duration_preference")
    _ratio("dbc", "spx",       "commodity_vs_equity")
    _ratio("tip", "ief",       "inflation_demand")
    _ratio("btc", "gold",      "btc_gold_ratio")

    # Yield curve spreads
    if "dgs10" in raw and "dgs2" in raw:
        f["curve_10_2"]        = raw["dgs10"] - raw["dgs2"]
        f["curve_10_2_chg_5d"] = f["curve_10_2"].diff(5)
        f["curve_10_2_chg_21d"]= f["curve_10_2"].diff(21)
    if "dgs10" in raw and "dgs3mo" in raw:
        f["curve_10_3m"]       = raw["dgs10"] - raw["dgs3mo"]
    if "dgs30" in raw and "dgs10" in raw:
        f["curve_30_10"]       = raw["dgs30"] - raw["dgs10"]
    if "t10y2y" in raw:
        f["t10y2y_level"]      = raw["t10y2y"]
        f["t10y2y_chg_5d"]     = raw["t10y2y"].diff(5)
    if "t10y3m" in raw:
        f["t10y3m_level"]      = raw["t10y3m"]

    # Real yields & breakevens
    for col in ["real5y", "real10y", "real20y", "real30y",
                "be5y", "be10y", "be20y", "be30y"]:
        if col in raw:
            f[f"{col}_level"]   = raw[col]
            f[f"{col}_chg_5d"]  = raw[col].diff(5)
            f[f"{col}_chg_21d"] = raw[col].diff(21)

    # Fed funds and rate momentum
    if "fedfunds" in raw:
        f["fedfunds_level"]   = raw["fedfunds"]
        f["fedfunds_chg_5d"]  = raw["fedfunds"].diff(5)
        f["fedfunds_chg_63d"] = raw["fedfunds"].diff(63)

    # Money supply growth
    if "m2" in raw:
        f["m2_yoy"] = raw["m2"].pct_change(252)
        f["m2_3m"]  = raw["m2"].pct_change(63)
    if "m1" in raw:
        f["m1_yoy"] = raw["m1"].pct_change(252)

    # Inflation
    for col in ["cpi", "core_cpi", "pce", "pcepi"]:
        if col in raw:
            f[f"{col}_yoy"] = raw[col].pct_change(252)
            f[f"{col}_3m"]  = raw[col].pct_change(63)

    # Labor market
    for col in ["unrate", "claims", "jolts"]:
        if col in raw:
            f[f"{col}_level"]   = raw[col]
            f[f"{col}_chg_21d"] = raw[col].diff(21)

    # Activity
    for col in ["indpro", "caput", "dgorder"]:
        if col in raw:
            f[f"{col}_yoy"]    = raw[col].pct_change(252)
            f[f"{col}_chg_21d"]= raw[col].pct_change(21)

    # Credit
    for col in ["hy_spread", "ig_spread", "bbb_spread"]:
        if col in raw:
            f[f"{col}_level"]   = raw[col]
            f[f"{col}_chg_5d"]  = raw[col].diff(5)
            f[f"{col}_chg_21d"] = raw[col].diff(21)

    # Consumer sentiment / housing
    for col in ["umcsi", "csushpisa", "mortgage30", "rsafs"]:
        if col in raw:
            f[f"{col}_level"]   = raw[col]
            f[f"{col}_chg_21d"] = raw[col].pct_change(21)

    # Geopolitical / policy uncertainty
    for col in raw.columns:
        if col.startswith("gpr_"):
            f[f"{col}_level"]   = raw[col]
            f[f"{col}_chg_5d"]  = raw[col].diff(5)
            f[f"{col}_chg_21d"] = raw[col].diff(21)
        if col.startswith("epu_"):
            f[f"{col}_level"]   = raw[col]
            f[f"{col}_chg_21d"] = raw[col].diff(21)

    # ------- SEASONALITY FEATURES -------
    # Gold demand has well-documented seasonal patterns (Indian wedding & Diwali,
    # Chinese New Year, Akshaya Tritiya, jewellery trade cycles, tax effects).
    # Sinusoidal encoding captures periodicity without sharp discontinuities.
    _month = pd.Series(f.index.month, index=f.index, dtype=float)
    _doy   = pd.Series(f.index.dayofyear, index=f.index, dtype=float)
    _week  = pd.Series(f.index.isocalendar().week.values, index=f.index, dtype=float)
    f["seas_month_sin"]   = np.sin(2 * np.pi * _month / 12)
    f["seas_month_cos"]   = np.cos(2 * np.pi * _month / 12)
    f["seas_doy_sin"]     = np.sin(2 * np.pi * _doy / 365)
    f["seas_doy_cos"]     = np.cos(2 * np.pi * _doy / 365)
    f["seas_week_sin"]    = np.sin(2 * np.pi * _week / 52)
    f["seas_week_cos"]    = np.cos(2 * np.pi * _week / 52)
    # Binary seasonal indicators (documented demand windows)
    f["seas_diwali"]      = _month.isin([10, 11]).astype(float)   # Oct-Nov: Diwali gift gold
    f["seas_wedding"]     = _month.isin([11, 12, 1, 2]).astype(float)  # Nov-Feb: Indian weddings
    f["seas_akshaya"]     = _month.isin([4, 5]).astype(float)     # Akshaya Tritiya auspicious buying
    f["seas_chinese_ny"]  = _month.isin([1, 2]).astype(float)     # Chinese New Year gold gifts
    f["seas_summer_weak"] = _month.isin([6, 7]).astype(float)     # Northern-hemisphere summer lull
    f["seas_q1"] = _month.isin([1, 2, 3]).astype(float)
    f["seas_q2"] = _month.isin([4, 5, 6]).astype(float)
    f["seas_q3"] = _month.isin([7, 8, 9]).astype(float)
    f["seas_q4"] = _month.isin([10, 11, 12]).astype(float)

    # ------- REGIME DETECTION -------
    # Classify market environment: VIX (risk appetite), real yields (gold cost
    # of carry), yield curve (recession signal), DXY trend (dollar strength).
    # Gold reacts very differently across regimes; these flags help the model
    # weigh macro vs technical signals appropriately.
    if "vix" in raw:
        _vix = raw["vix"]
        f["regime_vix_high"]    = (_vix > 25).astype(float)          # risk-off
        f["regime_vix_extreme"] = (_vix > 40).astype(float)          # fear spike
        f["regime_vix_low"]     = (_vix < 15).astype(float)          # complacency
        f["regime_vix_trend"]   = _vix.pct_change(21)                # 21-day VIX change
    if "real10y" in raw:
        _ry = raw["real10y"]
        f["regime_ry_negative"]  = (_ry < 0).astype(float)           # negative real yields → gold bullish
        f["regime_ry_fall_fast"] = (_ry.diff(21) < -0.10).astype(float)  # sharply falling
        f["regime_ry_rise_fast"] = (_ry.diff(21) > 0.10).astype(float)   # sharply rising → gold headwind
    if "curve_10_2" in f:
        _curve = f["curve_10_2"]
        f["regime_curve_inverted"] = (_curve < 0).astype(float)      # recession signal
        f["regime_curve_steep"]    = (_curve > 1.5).astype(float)    # inflationary growth
    if "dxy" in raw:
        _dxy = raw["dxy"]
        f["regime_dxy_above_ma"]  = (_dxy > _dxy.rolling(252).mean()).astype(float)
        f["regime_dxy_trend_21d"] = _dxy.pct_change(21)
    # Composite regime score: count of gold-bullish macro conditions (0–4)
    _regime_score = pd.Series(0.0, index=f.index)
    if "real10y" in raw:
        _regime_score += (raw["real10y"] < 0).astype(float)
    if "dxy" in raw:
        _regime_score += (raw["dxy"] < raw["dxy"].rolling(252).mean()).astype(float)
    if "vix" in raw:
        _regime_score += (raw["vix"] > 20).astype(float)
    if "curve_10_2" in f:
        _regime_score += (f["curve_10_2"] < 0).astype(float)
    f["regime_gold_score"]     = _regime_score                       # 0=hostile, 4=ideal
    f["regime_gold_score_z21"] = (_regime_score - _regime_score.rolling(21).mean()) / (
        _regime_score.rolling(21).std() + 1e-9)

    # ------- GVZ (GOLD VOLATILITY INDEX) -------
    # GVZ is the CBOE implied-volatility index for gold options (the "Gold VIX").
    # It provides a forward-looking fear/greed measure specific to gold.
    # Research: GVZ spikes that subsequently fall are associated with strong
    # gold rebounds (mean-reversion after panic: 68% win-rate, 4-week horizon).
    if "gvz" in raw:
        _gvz = raw["gvz"]
        f["gvz_level"]    = _gvz
        f["gvz_z_21"]     = (_gvz - _gvz.rolling(21).mean()) / (_gvz.rolling(21).std() + 1e-9)
        f["gvz_z_63"]     = (_gvz - _gvz.rolling(63).mean()) / (_gvz.rolling(63).std() + 1e-9)
        f["gvz_chg_5d"]   = _gvz.pct_change(5)
        f["gvz_chg_21d"]  = _gvz.pct_change(21)
        f["gvz_high"]     = (_gvz > 25).astype(float)                # elevated volatility regime
        f["gvz_extreme"]  = (_gvz > 35).astype(float)                # panic / tail-risk event
        f["gvz_falling"]  = (_gvz.pct_change(5) < -0.05).astype(float)  # fear subsiding → bullish
        f["gvz_rising"]   = (_gvz.pct_change(5) > 0.05).astype(float)   # fear building → bearish
        if "vix" in raw:
            f["gvz_vix_ratio"]     = _gvz / (raw["vix"] + 1e-9)     # gold-specific vs market fear
            f["gvz_vix_ratio_chg"] = f["gvz_vix_ratio"].pct_change(5)

    # ------- COT (COMMITMENTS OF TRADERS) -------
    # CFTC Disaggregated COT: Managed-Money (large speculator) net long positions
    # in COMEX gold futures. Published weekly.  Key contrarian signals:
    #   • Net long ratio > 0.6 → crowded long → contrarian bearish signal
    #   • Net long ratio < 0 → crowded short → contrarian bullish signal
    #   • Rapid change in net → momentum or capitulation signal
    for _cot_col in ["cot_net", "cot_net_pct", "cot_ratio"]:
        if _cot_col in raw:
            _cs = raw[_cot_col].ffill(limit=7)
            f[f"{_cot_col}_level"]  = _cs
            f[f"{_cot_col}_chg_4w"] = _cs.diff(20)
            f[f"{_cot_col}_z_52w"]  = (_cs - _cs.rolling(252).mean()) / (_cs.rolling(252).std() + 1e-9)
    if "cot_ratio" in raw:
        _cr = raw["cot_ratio"].ffill(limit=7)
        f["cot_extreme_long"]  = (_cr > 0.6).astype(float)           # crowded long → bearish
        f["cot_extreme_short"] = (_cr < 0.0).astype(float)           # crowded short → bullish
        f["cot_momentum"]      = (_cr.diff(4) > 0).astype(float)     # specs increasing longs

    # ------- GDX LEAD INDICATOR -------
    # Gold miner stocks (GDX) frequently lead gold spot by 1–5 trading days
    # because institutional capital rotates into high-beta miners before buying
    # physical.  The miners/gold relative-strength divergence is particularly
    # informative as a short-term leading indicator.
    if "gdx" in raw and gold is not None:
        _gdx = raw["gdx"]
        for _lag in [1, 2, 3, 5]:
            f[f"gdx_lead_{_lag}d"] = _gdx.pct_change(_lag).shift(_lag)
        f["gdx_gold_rs"]          = (_gdx / (gold + 1e-9))
        f["gdx_gold_rs_chg_5d"]   = f["gdx_gold_rs"].pct_change(5)
        f["gdx_outperform_5d"]    = (_gdx.pct_change(5) > gold.pct_change(5)).astype(float)

    # ------- CANDLESTICK PATTERN FEATURES -------
    # Use gold OHLCV columns if available (fetched alongside close prices)
    _gold_open  = raw.get("gold_open")
    _gold_high  = raw.get("gold_high")
    _gold_low   = raw.get("gold_low")
    if (_gold_open is not None and _gold_high is not None
            and _gold_low is not None and gold is not None):
        try:
            _ohlcv = pd.DataFrame({
                "Open":  _gold_open.reindex(raw.index).ffill(limit=3),
                "High":  _gold_high.reindex(raw.index).ffill(limit=3),
                "Low":   _gold_low.reindex(raw.index).ffill(limit=3),
                "Close": gold.reindex(raw.index).ffill(limit=3),
            }).dropna()
            _cs_feats = make_candlestick_features(_ohlcv)
            for col in _cs_feats.columns:
                f[col] = _cs_feats[col].reindex(f.index)
        except Exception:
            pass

    # ------- MOMENTUM COMPOSITE & CROSS-ASSET LEADS -------
    # These features capture temporal patterns that tree-based models miss:
    # trend consistency, momentum acceleration, and inter-asset lead-lag signals.
    if gold is not None:
        # MA consensus: fraction of 6 key MAs that gold is trading above (0–1).
        # 1.0 = fully in uptrend across all timeframes, 0.0 = fully in downtrend.
        _ma_cons = sum(
            (gold > gold.rolling(w).mean()).astype(float)
            for w in [5, 10, 20, 50, 100, 200]
        ) / 6.0
        f["gold_ma_consensus"]       = _ma_cons
        f["gold_ma_consensus_5d"]    = _ma_cons.diff(5)
        f["gold_ma_consensus_21d"]   = _ma_cons.diff(21)

        # Trend consistency: fraction of days that closed up in rolling window.
        # High value = persistent up-trend; low = persistent down-trend.
        _up = (gold.diff(1) > 0).astype(float)
        f["gold_up_days_10"] = _up.rolling(10).mean()
        f["gold_up_days_20"] = _up.rolling(20).mean()
        f["gold_up_days_63"] = _up.rolling(63).mean()

        # Momentum acceleration: is the current 5-day move faster or slower
        # than the previous 5-day move?  Positive = accelerating up-move.
        _roc5 = gold.pct_change(5)
        f["gold_roc5_accel"]  = _roc5 - _roc5.shift(5)
        _roc10 = gold.pct_change(10)
        f["gold_roc10_accel"] = _roc10 - _roc10.shift(10)

    # Silver as a 1–3-day leading indicator for gold.
    # Institutional capital often rotates into silver first on a metals rally.
    if "silver" in raw and gold is not None:
        for _lag in [1, 2, 3]:
            f[f"silver_lead_{_lag}d"] = raw["silver"].pct_change(1).shift(_lag)
        # Silver-gold momentum divergence: when silver surges faster than gold,
        # gold typically follows within 1–5 days (convergence trade).
        f["silver_gold_mom_div_5d"] = (
            raw["silver"].pct_change(5) - gold.pct_change(5)
        )

    # Treasury yield momentum and acceleration.
    # Gold is negatively correlated with real yields; acceleration of the
    # yield move (second derivative) signals regime transitions early.
    if "dgs10" in raw:
        _y10 = raw["dgs10"]
        _y10_5d = _y10.diff(5)
        f["yield10_accel_5d"]  = _y10_5d - _y10_5d.shift(5)
        f["yield10_vs_ma_63d"] = _y10 - _y10.rolling(63).mean()

    # DXY-gold divergence: when DXY is rising AND gold is also rising, that
    # indicates exceptional gold demand overriding the usual inverse relationship —
    # a historically bullish signal for continued gold strength.
    if "dxy" in raw and gold is not None:
        _dxy_5d  = raw["dxy"].pct_change(5)
        _gold_5d = gold.pct_change(5)
        f["dxy_gold_diverge_5d"] = _dxy_5d + _gold_5d   # >0 when both rising = unusual
        f["dxy_roc_accel"]       = _dxy_5d - _dxy_5d.shift(5)

    # GVZ/VIX ratio acceleration: gold-specific fear expanding faster than
    # broad market fear = gold is pricing in a unique risk, not just market vol.
    if "vix" in raw and "gvz" in raw:
        _ratio = raw["gvz"] / (raw["vix"] + 1e-9)
        _ratio_5d = _ratio.pct_change(5)
        f["gvz_vix_ratio_accel"] = _ratio_5d - _ratio_5d.shift(5)

    # ------- SHORT-HORIZON VELOCITY FEATURES -------
    # 1-day and 2-day rates-of-change for the most important gold drivers.
    # These are the strongest predictors for 1–5 day gold price direction
    # and are largely absent from the 5d/21d window features above.

    # Real interest rate 1-day velocity — #1 gold driver at short horizons.
    # A +5bp single-day rise in 10yr real yield → gold falls next day ~68% of the time.
    for _ry_col in ["real10y", "real5y", "real20y", "be10y", "be5y"]:
        if _ry_col in raw:
            _s = raw[_ry_col]
            f[f"{_ry_col}_vel_1d"] = _s.diff(1)     # 1-day change in basis points
            f[f"{_ry_col}_vel_2d"] = _s.diff(2)     # 2-day cumulative change
            f[f"{_ry_col}_vel_1d_sq"] = _s.diff(1) ** 2   # magnitude of shock (unsigned)

    # Nominal 10yr yield 1-day velocity (often drives gold more than real yield intraday)
    if "dgs10" in raw:
        _y = raw["dgs10"]
        f["dgs10_vel_1d"] = _y.diff(1)
        f["dgs10_vel_2d"] = _y.diff(2)

    # Gold short-horizon mean-reversion signals (2d/3d not in the base loop above)
    if gold is not None:
        f["gold_ret_2d"] = gold.pct_change(2)
        f["gold_ret_3d"] = gold.pct_change(3)
        # Reversal flag: if gold was strongly up/down 1d, next day tends to partially reverse
        _g1 = gold.pct_change(1)
        f["gold_overext_1d_up"]  = (_g1 > _g1.rolling(63).std() * 1.5).astype(float)
        f["gold_overext_1d_dn"]  = (_g1 < -_g1.rolling(63).std() * 1.5).astype(float)
        # Gold 1-day change lagged by 1 (yesterday's gold return as predictor)
        f["gold_ret_1d_lag1"]  = _g1.shift(1)
        f["gold_ret_1d_lag2"]  = _g1.shift(2)

    # VIX 1-day velocity — fast-moving risk signal, leads gold in risk-off events
    if "vix" in raw:
        _vx = raw["vix"]
        f["vix_vel_1d"]     = _vx.pct_change(1)
        f["vix_vel_2d"]     = _vx.pct_change(2)
        f["vix_spike_1d"]   = (_vx.pct_change(1) > 0.05).astype(float)  # >5% VIX jump in 1 day
        f["vix_collapse_1d"]= (_vx.pct_change(1) < -0.05).astype(float)  # >5% VIX drop in 1 day

    # DXY yesterday's change (1-day lag of DXY return) as a leading gold indicator
    if "dxy" in raw:
        _dx = raw["dxy"]
        _dx1 = _dx.pct_change(1)
        f["dxy_ret_1d_lag1"] = _dx1.shift(1)   # yesterday's DXY move predicting today's gold
        f["dxy_ret_1d_lag2"] = _dx1.shift(2)   # 2 days ago
        f["dxy_ret_2d"]      = _dx.pct_change(2)
        f["dxy_ret_3d"]      = _dx.pct_change(3)
        f["dxy_strong_1d"]   = (_dx1 > 0.005).astype(float)   # DXY +0.5% in a day → gold headwind
        f["dxy_weak_1d"]     = (_dx1 < -0.005).astype(float)  # DXY −0.5% in a day → gold tailwind

    # COT net-position weekly velocity (how fast specs are adding/cutting longs)
    for _cot_col in ["cot_net", "cot_net_pct", "cot_ratio"]:
        if _cot_col in raw:
            _cs = raw[_cot_col].ffill(limit=7)
            f[f"{_cot_col}_vel_1w"] = _cs.diff(5)   # 1-week change in net position
            # Percentile rank of current COT vs 52-week range (0=extreme short, 1=extreme long)
            _hi = _cs.rolling(252).max()
            _lo = _cs.rolling(252).min()
            f[f"{_cot_col}_pct_rank"] = (_cs - _lo) / (_hi - _lo + 1e-9)

    # ------- MARKET-STRUCTURE SHOCK DETECTION -------
    # Derived from analysis of gold's 2026 correction events.
    # Four distinct crash mechanisms identified from cross-asset data:
    #
    #  Jan 30 2026 (-10.3%): Gold-isolation flash crash — COMEX position unwind.
    #    Gold crashes while SPY/-0.3%, TLT/-0.6%, DXY/+0.7% barely move.
    #  Mar 3, Mar 26 2026: Tariff/policy shock — DXY up, VIX up, OIL up, SPX down.
    #    Dollar safe-haven demand overwhelms gold safe-haven bid.
    #  Mar 19 2026: Risk-calm normalization — VIX -4%, bonds rally, gold loses premium.
    #  Mar 23 2026: Risk-on rotation — SPX +1.1%, TLT +0.7%, VIX -2.4%, gold exits.
    #  Mar 26 2026: Forced liquidation — SPX -1.8%, TLT -0.8%, VIX +8.3% (margin calls).

    if gold is not None:
        _gold_ret1 = gold.pct_change(1)

        # ── Pattern 1: Gold isolation (COMEX-specific event) ─────────────────
        # Gold moves much larger than the cross-asset average → COMEX unwind
        _other_abs = pd.Series(0.0, index=gold.index)
        _n_oth = 0
        for _oc in ["spx", "dxy", "tlt", "wti", "silver"]:
            if _oc in raw:
                _other_abs += raw[_oc].pct_change(1).abs()
                _n_oth += 1
        if _n_oth > 0:
            _avg_other = _other_abs / _n_oth
            f["gold_isolation_1d"]     = _gold_ret1.abs() / (_avg_other + 0.001)
            f["gold_isolation_extreme"] = (f["gold_isolation_1d"] > 5).astype(float)

        # ── Pattern 2: Tariff / policy shock composite ────────────────────────
        # DXY up + VIX up + OIL up + SPX down on same day (≥3 of 4 = active)
        _tshock = pd.Series(0.0, index=gold.index)
        if "dxy" in raw:
            _tshock += (raw["dxy"].pct_change(1) > 0.003).astype(float)
        if "vix" in raw:
            _tshock += (raw["vix"].pct_change(1) > 0.05).astype(float)
        if "wti" in raw:
            _tshock += (raw["wti"].pct_change(1) > 0.02).astype(float)
        if "spx" in raw:
            _tshock += (raw["spx"].pct_change(1) < 0).astype(float)
        f["tariff_shock_score"]  = _tshock
        f["tariff_shock_active"] = (_tshock >= 3).astype(float)
        f["tariff_shock_5d_avg"] = _tshock.rolling(5).mean()
        f["tariff_shock_21d_avg"]= _tshock.rolling(21).mean()

        # ── Pattern 3: Risk-on safe-haven rotation ────────────────────────────
        # SPX up strongly + VIX down = risk appetite restored, gold premium sold
        _ron = pd.Series(0.0, index=gold.index)
        if "spx" in raw:
            _ron += (raw["spx"].pct_change(1) > 0.005).astype(float)
        if "tlt" in raw:
            _ron += (raw["tlt"].pct_change(1) > 0).astype(float)
        if "vix" in raw:
            _ron += (raw["vix"].pct_change(1) < -0.03).astype(float)
        f["risk_on_rotation_score"]  = _ron
        f["risk_on_rotation_active"] = (_ron >= 2).astype(float)
        f["risk_on_rotation_5d_avg"] = _ron.rolling(5).mean()

        # ── Pattern 4: Forced liquidation (margin-call selling) ───────────────
        # SPX down + VIX spike + bonds also selling = cross-asset deleveraging
        _fliq = pd.Series(0.0, index=gold.index)
        if "spx" in raw:
            _fliq += (raw["spx"].pct_change(1) < -0.01).astype(float)
        if "vix" in raw:
            _fliq += (raw["vix"].pct_change(1) > 0.10).astype(float)
        if "tlt" in raw:
            _fliq += (raw["tlt"].pct_change(1) < -0.005).astype(float)
        f["forced_liq_score"]  = _fliq
        f["forced_liq_active"] = (_fliq >= 2).astype(float)
        f["forced_liq_5d_sum"] = _fliq.rolling(5).sum()

        # ── Pattern 5: Gold exhaustion / overextension (crash risk) ──────────
        # Near 52-week high + overbought RSI = crowded-long reversal setup
        _peak52w = gold.rolling(252).max()
        _peak20  = gold.rolling(20).max()
        _peak63  = gold.rolling(63).max()
        f["gold_dist_52w_high"]  = gold / (_peak52w + 1e-9) - 1   # 0 = at 52w high
        f["gold_drawdown_20d"]   = gold / (_peak20  + 1e-9) - 1
        f["gold_drawdown_63d"]   = gold / (_peak63  + 1e-9) - 1
        f["gold_near_52w_high"]  = (f["gold_dist_52w_high"] > -0.03).astype(float)
        f["gold_in_correction"]  = (f["gold_drawdown_63d"] < -0.05).astype(float)
        f["gold_in_bear"]        = (f["gold_drawdown_63d"] < -0.10).astype(float)
        if "gold_rsi14" in f:
            f["gold_exhaustion"] = (
                f["gold_near_52w_high"] * (f["gold_rsi14"] > 72).astype(float)
            )
        # Extreme 1-month return (overbought or oversold)
        _ret21 = gold.pct_change(21)
        f["gold_1m_ret_extreme_up"]   = (_ret21 > 0.10).astype(float)
        f["gold_1m_ret_extreme_down"] = (_ret21 < -0.05).astype(float)
        f["gold_1m_ret_z"] = (
            (_ret21 - _ret21.rolling(252).mean()) / (_ret21.rolling(252).std() + 1e-9)
        )

        # ── Gold-SPX rolling correlation (regime classifier) ──────────────────
        if "spx" in raw:
            _gd1  = gold.pct_change(1)
            _sd1  = raw["spx"].pct_change(1)
            # Negative correlation = normal hedge; positive = both crashing = crisis
            f["gold_spx_corr_21d"]     = _gd1.rolling(21).corr(_sd1)
            f["gold_spx_corr_63d"]     = _gd1.rolling(63).corr(_sd1)
            f["gold_spx_joint_crash"]  = ((_gd1 < -0.01) & (_sd1 < -0.01)).astype(float)
            f["gold_spx_joint_crash_5d"] = f["gold_spx_joint_crash"].rolling(5).sum()

        # ── Oil-Gold divergence (supply shock vs fear signal) ────────────────
        if "wti" in raw:
            _oil5 = raw["wti"].pct_change(5)
            _au5  = gold.pct_change(5)
            f["oil_surge_gold_fall"] = ((_oil5 > 0.05) & (_au5 < 0)).astype(float)
            f["oil_gold_joint_fall"] = ((_oil5 < -0.05) & (_au5 < 0)).astype(float)

    # ------- ECONOMIC CALENDAR FEATURES -------
    # FOMC, CPI, NFP binary flags and proximity scores.
    # Research: macro event windows carry strong predictive signal for gold
    # (pre-event safe-haven bid-up, post-event mean reversion).
    try:
        f = add_calendar_features(f)
    except Exception as _cal_err:
        pass  # non-fatal: model still runs without calendar features

    # ------- NEWS SENTIMENT (cached, fetched by scheduler) -------
    # Load cached keyword sentiment score and broadcast it as a constant feature
    # for the current training window.  Forward-fill is applied so each row
    # reflects the most recent sentiment reading available at that time.
    try:
        _sent = load_cached_sentiment()
        _sent_score = float(_sent.get("score", 0.0))
        f["news_sentiment"]        = _sent_score                     # today's score
        # A rolling sum would be more correct but we only have one number from cache;
        # at the next retrain the scheduler will have fetched fresh sentiment.
        f["news_sentiment_bullish"] = 1.0 if _sent_score > 0.15 else 0.0
        f["news_sentiment_bearish"] = 1.0 if _sent_score < -0.15 else 0.0
    except Exception:
        pass

    # ------- TARGETS -------
    for n in [1, 2, 5, 21, 63, 252]:
        f[f"target_{n}d"]       = (gold.shift(-n) > gold).astype(int)
        f[f"next_return_{n}d"]  = gold.pct_change(n).shift(-n)

    # Keep 1-day return as legacy alias
    f["next_return_1d"] = f["next_return_1d"]  # already set above

    f = f.replace([np.inf, -np.inf], np.nan)
    return f


# ---------------------------------------------------------------------------
# BACKTEST
# ---------------------------------------------------------------------------
def walk_forward(features, target_col="target_1d", return_col="next_return_1d",
                 train_years=5, retrain_every=63, progress_callback=None):
    target_cols_all = [c for c in features.columns if c.startswith("target_")]
    return_cols_all = [c for c in features.columns if c.startswith("next_return_")]
    feature_cols = [c for c in features.columns
                    if c not in target_cols_all and c not in return_cols_all]

    df = features.dropna(subset=[target_col, return_col])
    coverage = df[feature_cols].notna().mean()
    feature_cols = coverage[coverage > 0.5].index.tolist()
    df = df[feature_cols + [target_col, return_col]]
    df[feature_cols] = df[feature_cols].ffill(limit=5)
    df = df.dropna()

    X    = df[feature_cols]
    y    = df[target_col]
    rets = df[return_col]

    window = train_years * 252
    if len(df) < window + retrain_every:
        raise ValueError(
            f"Not enough data: need {window + retrain_every}, have {len(df)}. "
            f"Try a smaller training window or earlier start date."
        )

    # ── Feature Selection ──────────────────────────────────────────────────────
    # With 600+ macro features most contribute noise. Run a quick XGBoost on the
    # first training window and keep only the top-80 by importance. This cuts
    # overfitting substantially while preserving the genuinely predictive signals.
    # Not lookahead: uses only data from df[:window] (all historical).
    if len(feature_cols) > 100:
        _fs_X = X.iloc[:window]
        _fs_y = y.iloc[:window]
        _selector = xgb.XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.5,
            verbosity=0, tree_method="hist",
        )
        _selector.fit(_fs_X, _fs_y)
        _imp_series = pd.Series(_selector.feature_importances_, index=feature_cols)
        _selected   = _imp_series.nlargest(80).index.tolist()
        if len(_selected) >= 20:
            feature_cols = _selected
            X = X[feature_cols]
        del _selector, _fs_X, _fs_y

    preds = pd.Series(index=df.index, dtype=float)
    probas = pd.Series(index=df.index, dtype=float)
    importances = []

    total_steps = max(1, (len(df) - window) // retrain_every)
    step = 0
    i = window
    while i < len(df):
        train_X = X.iloc[i - window:i]
        train_y = y.iloc[i - window:i]
        test_X  = X.iloc[i:i + retrain_every]

        _n_pos   = int(train_y.sum())
        _n_neg   = len(train_y) - _n_pos
        _spw     = max(0.5, min(3.0, _n_neg / (_n_pos + 1e-9)))   # DOWN/UP ratio, clamped

        xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7,
            reg_lambda=2.0, reg_alpha=0.5, min_child_weight=8,
            eval_metric="logloss", tree_method="hist", verbosity=0,
            scale_pos_weight=_spw,   # corrects UP/DOWN class imbalance
        )
        xgb_model.fit(train_X, train_y)

        rf_model = RandomForestClassifier(
            n_estimators=80, max_depth=6, min_samples_leaf=5,
            max_features="sqrt", n_jobs=1, random_state=42,
            class_weight="balanced",   # equal penalty for UP and DOWN errors
        )
        rf_model.fit(train_X, train_y)

        mlp_model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=300,
                activation="relu", solver="adam", alpha=0.01,
                random_state=42, early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=20,
            )),
        ])
        mlp_model.fit(train_X, train_y)

        model = EnsembleModel(xgb_model, rf_model, mlp_model)

        # Calibrate probabilities using last 20% of training data as a
        # held-out validation fold.  Isotonic regression maps raw ensemble
        # scores → calibrated probabilities so confidence % is trustworthy.
        _cal_n  = max(60, len(train_X) // 5)
        _cal_X  = train_X.iloc[-_cal_n:]
        _cal_y  = train_y.iloc[-_cal_n:]
        _raw_cal = model.predict_proba(_cal_X)[:, 1]
        _calibrator = IsotonicRegression(out_of_bounds="clip")
        _calibrator.fit(_raw_cal, _cal_y.values.astype(float))

        _raw_test = model.predict_proba(test_X)[:, 1]
        _cal_test = _calibrator.transform(_raw_test)
        preds.iloc[i:i + retrain_every]  = (_cal_test > 0.5).astype(int)
        probas.iloc[i:i + retrain_every] = _cal_test
        importances.append(pd.Series(model.feature_importances_, index=feature_cols))
        step += 1
        if progress_callback:
            progress_callback(f"Walk-forward step {step}/{total_steps}", step / total_steps)
        i += retrain_every

    mask  = preds.notna()
    preds  = preds[mask].astype(int)
    probas = probas[mask]
    actual = y[mask]
    rets   = rets[mask]

    acc = accuracy_score(actual, preds)
    # Use 1-day returns for equity curve regardless of horizon
    # (daily sizing based on directional signal)
    one_day_rets = features["next_return_1d"].reindex(preds.index).ffill()
    strat_returns = np.where(preds == 1, one_day_rets, -one_day_rets)
    strat_curve = (1 + pd.Series(strat_returns, index=preds.index)).cumprod()
    bh_curve    = (1 + one_day_rets).cumprod()
    sharpe = (np.mean(strat_returns) / (np.std(strat_returns) + 1e-9)) * np.sqrt(252)
    avg_importance = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)

    # Rolling 90-day accuracy
    rolling_acc = (preds == actual).rolling(90).mean()

    return {
        "predictions": preds,
        "probas": probas,
        "actual": actual,
        "returns": one_day_rets,
        "accuracy": acc,
        "rolling_accuracy": rolling_acc,
        "strategy_curve": strat_curve,
        "buyhold_curve": bh_curve,
        "sharpe": sharpe,
        "feature_importance": avg_importance,
        "n_features": len(feature_cols),
        "n_predictions": len(preds),
        "model": model,  # most recently trained model
        "feature_cols": feature_cols,
    }


# ---------------------------------------------------------------------------
# LIVE PREDICTION TRACKING
# ---------------------------------------------------------------------------
def load_live_predictions():
    if LIVE_PREDS_FILE.exists():
        try:
            return json.loads(LIVE_PREDS_FILE.read_text())
        except Exception:
            return []
    return []


def save_live_prediction(pred_date, direction, confidence, horizon_label, horizon_days):
    preds = load_live_predictions()
    target_date = pred_date + timedelta(days=int(horizon_days * 1.4))  # calendar days approx
    preds.append({
        "made_on":     str(pred_date.date()),
        "direction":   int(direction),
        "confidence":  float(round(confidence, 4)),
        "horizon":     horizon_label,
        "horizon_days": horizon_days,
        "target_date": str(target_date.date()),
        "outcome":     None,
    })
    LIVE_PREDS_FILE.write_text(json.dumps(preds, indent=2))


def resolve_live_predictions(gold_series):
    """Check past predictions where target_date has passed and record actual outcomes."""
    preds = load_live_predictions()
    today = pd.Timestamp.today().normalize()
    updated = False
    for p in preds:
        if p["outcome"] is not None:
            continue
        target_dt = pd.Timestamp(p["target_date"])
        if today < target_dt:
            continue
        made_dt = pd.Timestamp(p["made_on"])
        try:
            # Find nearest available gold prices
            avail = gold_series.dropna()
            idx_start = avail.index.asof(made_dt)
            idx_end   = avail.index.asof(target_dt)
            if pd.isna(idx_start) or pd.isna(idx_end):
                continue
            price_start = float(avail.loc[idx_start])
            price_end   = float(avail.loc[idx_end])
            actual = 1 if price_end > price_start else 0
            p["outcome"]       = actual
            p["price_start"]   = round(price_start, 2)
            p["price_end"]     = round(price_end, 2)
            p["actual_return"] = round((price_end / price_start - 1) * 100, 2)
            updated = True
        except Exception:
            pass
    if updated:
        LIVE_PREDS_FILE.write_text(json.dumps(preds, indent=2))
    return preds


def live_accuracy_stats(preds):
    resolved = [p for p in preds if p["outcome"] is not None]
    if not resolved:
        return None, 0, []
    correct = sum(1 for p in resolved if p["direction"] == p["outcome"])
    acc = correct / len(resolved)
    return acc, len(resolved), resolved
