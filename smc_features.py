"""
Smart Money Concepts (SMC) feature engine.

Adapted from candlestick_pro (provided by user). Detects institutional
price-action concepts that classical TA misses:

  Break of Structure (BOS)   — confirms when a trend has genuinely flipped
  Fair Value Gaps (FVG)       — price imbalances that often get re-tested
  Order Blocks (OB)           — zones of institutional accumulation / distribution
  Liquidity Sweeps            — stop-hunts that frequently precede sharp reversals
  ATR-quality pattern score  — candlestick patterns weighted by ATR size and context

All functions are causal (no look-ahead into the future).
They accept a DataFrame whose OHLCV columns are titled ('Open','High','Low',
'Close','Volume') — matching our yfinance / Twelve Data feeds.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ── Helpers ─────────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["Close"].shift()
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - pc).abs(),
        (df["Low"]  - pc).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ── Smart Money Concepts ──────────────────────────────────────────────────────

def swing_points(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """
    Pivot highs / lows: a bar is a swing high if it is the highest in a
    window of 2k+1 bars centred on it.
    """
    h, l = df["High"], df["Low"]
    sh = h == h.rolling(2 * k + 1, center=True).max()
    sl = l == l.rolling(2 * k + 1, center=True).min()
    return pd.DataFrame({"swing_high": sh, "swing_low": sl}, index=df.index)


def break_of_structure(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """
    Bullish BOS  — close breaks above the most recent confirmed swing high.
    Bearish BOS  — close breaks below the most recent confirmed swing low.
    Only the bar where the break actually happens is flagged.
    """
    sw = swing_points(df, k)
    last_sh = df["High"].where(sw["swing_high"]).ffill()
    last_sl = df["Low"].where(sw["swing_low"]).ffill()
    bos_up   = (df["Close"] > last_sh.shift()) & (df["Close"].shift() <= last_sh.shift())
    bos_down = (df["Close"] < last_sl.shift()) & (df["Close"].shift() >= last_sl.shift())
    return pd.DataFrame({
        "last_swing_high": last_sh,
        "last_swing_low":  last_sl,
        "bos_up":   bos_up.astype(int),
        "bos_down": bos_down.astype(int),
    }, index=df.index)


def fair_value_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bullish FVG: gap between bar[i-1].high and bar[i+1].low that the middle
    bar (i) does not fill — price left an imbalance to the upside.
    Bearish FVG: bar[i-1].low > bar[i+1].high.

    Lagged by 1 bar after detection so there is zero look-ahead.
    """
    h_prev = df["High"].shift(2)
    l_next = df["Low"]
    l_prev = df["Low"].shift(2)
    h_next = df["High"]
    raw_up   = (h_prev < l_next).astype(int)
    raw_down = (l_prev > h_next).astype(int)
    return pd.DataFrame({
        "fvg_up":   raw_up.shift(1).fillna(0).astype(int),
        "fvg_down": raw_down.shift(1).fillna(0).astype(int),
    }, index=df.index)


def order_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bullish OB: the last bearish candle immediately before a strong upward
    impulse (> 1.5 × ATR over the next 3 bars).  Price returning to this
    level is a high-probability long entry zone.

    Bearish OB: last bullish candle before a strong downward impulse.
    Lagged by 3 bars — fully causal.
    """
    a = _atr(df)
    fwd_move     = df["Close"].shift(-3) - df["Close"]
    impulse_up   = fwd_move >  1.5 * a
    impulse_down = fwd_move < -1.5 * a
    is_bear = df["Close"] < df["Open"]
    is_bull = df["Close"] > df["Open"]
    bull_ob = (is_bear & impulse_up).astype(int).shift(3).fillna(0).astype(int)
    bear_ob = (is_bull & impulse_down).astype(int).shift(3).fillna(0).astype(int)
    return pd.DataFrame({"bull_ob": bull_ob, "bear_ob": bear_ob}, index=df.index)


def liquidity_sweeps(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Sweep high: wick above the prior N-bar high then price closes back below
    it — a stop-hunt above resistance, often preceding a bearish reversal.

    Sweep low: wick below prior N-bar low, close back above — bullish reversal.
    """
    prior_high = df["High"].rolling(lookback).max().shift()
    prior_low  = df["Low"].rolling(lookback).min().shift()
    sweep_high = ((df["High"] > prior_high) & (df["Close"] < prior_high)).astype(int)
    sweep_low  = ((df["Low"]  < prior_low)  & (df["Close"] > prior_low)).astype(int)
    return pd.DataFrame({"sweep_high": sweep_high, "sweep_low": sweep_low}, index=df.index)


# ── ATR-quality candlestick pattern scorer ────────────────────────────────────

def _pattern_score(df: pd.DataFrame) -> float:
    """
    Score the most recent bar on known candlestick patterns.
    Returns a value in [-1, +1] normalised by ATR and boosted at swing levels.
    Adapted from candlestick_pro/patterns/detector.py.
    """
    if len(df) < 10:
        return 0.0

    o  = df["Open"];  h  = df["High"]
    l  = df["Low"];   c  = df["Close"]
    a  = _atr(df, 14)
    body  = (c - o).abs()
    rng   = (h - l).replace(0, np.nan)
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    body_atr = body / a

    score = pd.Series(0.0, index=df.index)

    # Hammer — bullish reversal
    hammer = (lower > 2 * body) & (upper < body) & (body_atr > 0.1)
    score += hammer.astype(float) * 0.6

    # Shooting star — bearish
    shooting_star = (upper > 2 * body) & (lower < body) & (body_atr > 0.1)
    score -= shooting_star.astype(float) * 0.6

    # Marubozu — full-body momentum
    full = (body / rng) > 0.90
    score += (full & (c > o) & (body_atr > 0.5)).astype(float) * 0.5
    score -= (full & (c < o) & (body_atr > 0.5)).astype(float) * 0.5

    # Engulfing — strongest single-bar reversal signal
    o1, c1 = o.shift(), c.shift()
    bull_engulf = (c1 < o1) & (c > o) & (c >= o1) & (o <= c1) & (body_atr > 0.5)
    bear_engulf = (c1 > o1) & (c < o) & (c <= o1) & (o >= c1) & (body_atr > 0.5)
    score += bull_engulf.astype(float) * 0.8
    score -= bear_engulf.astype(float) * 0.8

    # Morning / Evening star (3-bar reversal)
    o2, c2 = o.shift(2), c.shift(2)
    body2   = (c2 - o2).abs()
    small_m = body.shift() < body2 * 0.5
    score += ((c2 < o2) & small_m & (c > o) & (c > (o2 + c2) / 2)).astype(float) * 0.9
    score -= ((c2 > o2) & small_m & (c < o) & (c < (o2 + c2) / 2)).astype(float) * 0.9

    # Tweezer tops / bottoms
    eq_high = (h - h.shift()).abs() < 0.05 * a
    eq_low  = (l - l.shift()).abs() < 0.05 * a
    score += (eq_low  & (c1 < o1) & (c > o)).astype(float) * 0.55
    score -= (eq_high & (c1 > o1) & (c < o)).astype(float) * 0.55

    # Volume boost: pattern on above-avg volume carries more weight
    vol_boost = (df["Volume"] > 1.5 * df["Volume"].rolling(20).mean()).astype(int)
    score *= 1 + 0.30 * vol_boost

    # Context boost: pattern at a recent swing point is higher quality
    sw = swing_points(df, k=3)
    near_swing = (sw["swing_high"] | sw["swing_low"]).astype(int)
    score *= 1 + 0.50 * near_swing

    raw = float(score.clip(-2, 2).iloc[-1]) / 2.0   # normalise to -1..+1
    return raw


# ── Master score function ─────────────────────────────────────────────────────

def compute_smc_score(df: pd.DataFrame) -> tuple[int, dict]:
    """
    Compute a combined SMC confluence score for the current bar.

    Returns:
        score      int in [-2, +2] (clipped total of sub-scores)
        sub_scores dict of label → score for display in the signal card
    """
    if df is None or len(df) < 30:
        return 0, {}

    sub: dict[str, int] = {}
    raw = 0

    # ── BOS ──────────────────────────────────────────────────────────────────
    try:
        bos_df = break_of_structure(df)
        if int(bos_df["bos_up"].iloc[-1]):
            sub["BOS ▲"] = 1
            raw += 1
        elif int(bos_df["bos_down"].iloc[-1]):
            sub["BOS ▼"] = -1
            raw -= 1
    except Exception:
        pass

    # ── FVG (any active gap in last 5 bars) ──────────────────────────────────
    try:
        fvg_df = fair_value_gaps(df)
        recent_up   = bool(fvg_df["fvg_up"].iloc[-5:].any())
        recent_down = bool(fvg_df["fvg_down"].iloc[-5:].any())
        if recent_up and not recent_down:
            sub["FVG ▲"] = 1
            raw += 1
        elif recent_down and not recent_up:
            sub["FVG ▼"] = -1
            raw -= 1
        elif recent_up and recent_down:
            pass   # mixed FVGs — neutral
    except Exception:
        pass

    # ── Order Blocks (price retesting a zone within 2 × ATR) ─────────────────
    try:
        ob_df  = order_blocks(df)
        atr_v  = float(_atr(df).iloc[-1])
        price  = float(df["Close"].iloc[-1])

        bull_bars = ob_df["bull_ob"].iloc[-25:]
        if bull_bars.any():
            ob_idx   = bull_bars[bull_bars > 0].index[-1]
            ob_level = float(df["Low"].loc[ob_idx])
            if abs(price - ob_level) < 2.0 * atr_v:
                sub["Bull OB"] = 2
                raw += 2

        bear_bars = ob_df["bear_ob"].iloc[-25:]
        if bear_bars.any():
            ob_idx   = bear_bars[bear_bars > 0].index[-1]
            ob_level = float(df["High"].loc[ob_idx])
            if abs(price - ob_level) < 2.0 * atr_v:
                sub["Bear OB"] = -2
                raw -= 2
    except Exception:
        pass

    # ── Liquidity Sweeps (last 3 bars) ───────────────────────────────────────
    try:
        sw_df = liquidity_sweeps(df)
        if sw_df["sweep_low"].iloc[-3:].any():
            sub["Sweep ▲"] = 1    # stop-hunt below lows → bullish reversal
            raw += 1
        if sw_df["sweep_high"].iloc[-3:].any():
            sub["Sweep ▼"] = -1   # stop-hunt above highs → bearish reversal
            raw -= 1
    except Exception:
        pass

    # ── ATR-quality candlestick pattern ──────────────────────────────────────
    try:
        ps = _pattern_score(df)
        if ps >= 0.40:
            sub["Pattern ▲"] = 1
            raw += 1
        elif ps <= -0.40:
            sub["Pattern ▼"] = -1
            raw -= 1
    except Exception:
        pass

    score = max(-2, min(2, raw))
    return score, sub
