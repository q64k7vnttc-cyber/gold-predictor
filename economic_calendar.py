"""
economic_calendar.py
Macro event calendar for gold price prediction.

Research basis:
  - FOMC, CPI (US CPI-U), and Non-Farm Payrolls (NFP) are the three highest-
    impact USD-denominated macro releases for gold.
  - Incorporating 'within N days of event' binary flags as model features
    captures the well-documented pre-event bid-up and post-event mean-reversion
    patterns (multiple studies, 2022-2025).
  - Pre-event window (3 days before): uncertainty → gold safe-haven demand rises.
  - Event day: sharp directional move.
  - Post-event window (1 day after): mean reversion / continuation.

Usage:
    from economic_calendar import add_calendar_features, get_upcoming_events
    df_with_flags = add_calendar_features(df)          # for model features
    events        = get_upcoming_events(n_days=14)      # for dashboard display
"""
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# 2025-2026 MACRO EVENT DATES
# Source: US Federal Reserve, BLS, FRED
# ---------------------------------------------------------------------------

FOMC_DATES = [
    # 2025
    date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7),
    date(2025, 6, 18), date(2025, 7, 30), date(2025, 9, 17),
    date(2025, 10, 29), date(2025, 12, 10),
    # 2026
    date(2026, 1, 29), date(2026, 3, 18), date(2026, 4, 29),
    date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
    date(2026, 11, 4), date(2026, 12, 16),
]

CPI_DATES = [
    # 2025
    date(2025, 1, 15), date(2025, 2, 12), date(2025, 3, 12),
    date(2025, 4, 10), date(2025, 5, 13), date(2025, 6, 11),
    date(2025, 7, 11), date(2025, 8, 12), date(2025, 9, 10),
    date(2025, 10, 15), date(2025, 11, 13), date(2025, 12, 10),
    # 2026
    date(2026, 1, 14), date(2026, 2, 11), date(2026, 3, 11),
    date(2026, 4, 10), date(2026, 5, 12), date(2026, 6, 10),
    date(2026, 7, 14), date(2026, 8, 12), date(2026, 9, 10),
    date(2026, 10, 14), date(2026, 11, 12), date(2026, 12, 9),
]

NFP_DATES = [
    # 2025 (first Friday of each month)
    date(2025, 1, 10), date(2025, 2, 7), date(2025, 3, 7),
    date(2025, 4, 4), date(2025, 5, 2), date(2025, 6, 6),
    date(2025, 7, 3), date(2025, 8, 1), date(2025, 9, 5),
    date(2025, 10, 3), date(2025, 11, 7), date(2025, 12, 5),
    # 2026
    date(2026, 1, 9), date(2026, 2, 6), date(2026, 3, 6),
    date(2026, 4, 3), date(2026, 5, 1), date(2026, 6, 5),
    date(2026, 7, 10), date(2026, 8, 7), date(2026, 9, 4),
    date(2026, 10, 2), date(2026, 11, 6), date(2026, 12, 4),
]

# US Treasury auctions (7-year and 10-year) tend to move gold via real-yield
# channel. Approximate schedule (3rd week of each month for 10Y).
TREASURY_AUCTION_DATES = [
    # 2026 10-year auction (~3rd Wed of each month)
    date(2026, 1, 14), date(2026, 2, 11), date(2026, 3, 11),
    date(2026, 4, 8), date(2026, 5, 13), date(2026, 6, 10),
    date(2026, 7, 8), date(2026, 8, 12), date(2026, 9, 9),
    date(2026, 10, 14), date(2026, 11, 11), date(2026, 12, 9),
]

EVENT_META = {
    "FOMC":     {"label": "FOMC Rate Decision", "color": "#ef5350", "impact": "HIGH",
                 "gold_bias": "↑ Dovish = gold up · ↓ Hawkish = gold down"},
    "CPI":      {"label": "US CPI Release",     "color": "#ff9800", "impact": "HIGH",
                 "gold_bias": "↑ Low CPI = gold up (lower rate expectation)"},
    "NFP":      {"label": "Non-Farm Payrolls",  "color": "#ffa726", "impact": "HIGH",
                 "gold_bias": "↑ Weak jobs = gold up (safe-haven)"},
    "AUCTION":  {"label": "10Y Treasury Auction","color": "#42a5f5","impact": "MEDIUM",
                 "gold_bias": "↑ Weak demand = yields rise = gold under pressure"},
}


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING FOR ML MODEL
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add macro event binary flags to a DataFrame with a DatetimeIndex.

    Features added:
      fomc_in_3d   — 1 if FOMC is within the next 3 calendar days
      fomc_day     — 1 on the FOMC decision day
      fomc_post_1d — 1 the day after FOMC (often continuation / reversal)
      cpi_in_3d    — 1 if CPI is within the next 3 calendar days
      cpi_day      — 1 on CPI release day
      nfp_in_3d    — 1 if NFP is within the next 3 calendar days
      nfp_day      — 1 on NFP release day
      macro_event_week — 1 if ANY high-impact event in the next 7 days
      days_to_fomc — calendar days until the next FOMC (0 = same day)
      days_to_cpi  — calendar days until the next CPI
      days_to_nfp  — calendar days until the next NFP
    """
    f = df.copy()

    all_fomc    = pd.DatetimeIndex([pd.Timestamp(d) for d in FOMC_DATES])
    all_cpi     = pd.DatetimeIndex([pd.Timestamp(d) for d in CPI_DATES])
    all_nfp     = pd.DatetimeIndex([pd.Timestamp(d) for d in NFP_DATES])
    all_auction = pd.DatetimeIndex([pd.Timestamp(d) for d in TREASURY_AUCTION_DATES])

    idx = f.index.normalize()

    def _days_until(idx_date, event_dates):
        """Minimum calendar days until any upcoming event (0 if today, -1 if past all)."""
        out = []
        for d in idx_date:
            future = event_dates[event_dates >= d]
            out.append(float((future[0] - d).days) if len(future) > 0 else 999.0)
        return np.array(out, dtype=float)

    def _window_flag(idx_date, event_dates, pre_days=3, post_days=1):
        """1 if within pre_days before or post_days after any event date."""
        out = np.zeros(len(idx_date), dtype=float)
        for ed in event_dates:
            window_start = ed - pd.Timedelta(days=pre_days)
            window_end   = ed + pd.Timedelta(days=post_days)
            out += ((idx_date >= window_start) & (idx_date <= window_end)).astype(float)
        return np.clip(out, 0, 1)

    def _exact_day(idx_date, event_dates):
        """1 on the exact event day."""
        out = np.zeros(len(idx_date), dtype=float)
        for ed in event_dates:
            out += (idx_date == ed).astype(float)
        return np.clip(out, 0, 1)

    def _next_day(idx_date, event_dates):
        """1 on the day AFTER an event (post-event drift)."""
        shifted = pd.DatetimeIndex([ed + pd.Timedelta(days=1) for ed in event_dates])
        return _exact_day(idx_date, shifted)

    f["fomc_in_3d"]        = _window_flag(idx, all_fomc, pre_days=3, post_days=0)
    f["fomc_day"]          = _exact_day(idx, all_fomc)
    f["fomc_post_1d"]      = _next_day(idx, all_fomc)
    f["cpi_in_3d"]         = _window_flag(idx, all_cpi, pre_days=3, post_days=0)
    f["cpi_day"]           = _exact_day(idx, all_cpi)
    f["nfp_in_3d"]         = _window_flag(idx, all_nfp, pre_days=3, post_days=0)
    f["nfp_day"]           = _exact_day(idx, all_nfp)
    f["auction_in_2d"]     = _window_flag(idx, all_auction, pre_days=2, post_days=0)
    f["macro_event_week"]  = np.clip(
        f["fomc_in_3d"] + f["cpi_in_3d"] + f["nfp_in_3d"] + f["auction_in_2d"], 0, 1
    )

    f["days_to_fomc"] = _days_until(idx, all_fomc)
    f["days_to_cpi"]  = _days_until(idx, all_cpi)
    f["days_to_nfp"]  = _days_until(idx, all_nfp)

    # Inverse-day weighting (higher signal closer to event)
    f["fomc_proximity_score"] = np.exp(-f["days_to_fomc"] / 5.0)
    f["cpi_proximity_score"]  = np.exp(-f["days_to_cpi"] / 3.0)
    f["nfp_proximity_score"]  = np.exp(-f["days_to_nfp"] / 3.0)

    return f


# ---------------------------------------------------------------------------
# DASHBOARD HELPER
# ---------------------------------------------------------------------------

def get_upcoming_events(n_days: int = 30, reference_date: date | None = None) -> list[dict]:
    """
    Return a sorted list of upcoming macro events within n_days.
    Each dict has: date, event_type, label, color, impact, gold_bias, days_away.
    """
    if reference_date is None:
        reference_date = date.today()

    cutoff = reference_date + timedelta(days=n_days)
    events = []

    for event_type, dates in [("FOMC", FOMC_DATES), ("CPI", CPI_DATES),
                               ("NFP", NFP_DATES), ("AUCTION", TREASURY_AUCTION_DATES)]:
        meta = EVENT_META[event_type]
        for d in dates:
            if reference_date <= d <= cutoff:
                events.append({
                    "date":      d,
                    "event_type": event_type,
                    "label":     meta["label"],
                    "color":     meta["color"],
                    "impact":    meta["impact"],
                    "gold_bias": meta["gold_bias"],
                    "days_away": (d - reference_date).days,
                })

    events.sort(key=lambda x: x["date"])
    return events


def get_current_event_flags() -> dict:
    """
    Return a dict of today's binary event flags for use in the dashboard.
    Also returns the proximity score for each event type.
    """
    today = date.today()

    def _days_to(dates):
        future = [d for d in dates if d >= today]
        return (future[0] - today).days if future else 999

    def _within(dates, pre=3, post=1):
        for d in dates:
            if (d - timedelta(days=pre)) <= today <= (d + timedelta(days=post)):
                return True
        return False

    dtf = _days_to(FOMC_DATES)
    dtc = _days_to(CPI_DATES)
    dtn = _days_to(NFP_DATES)

    return {
        "fomc_in_3d":     _within(FOMC_DATES, pre=3),
        "cpi_in_3d":      _within(CPI_DATES,  pre=3),
        "nfp_in_3d":      _within(NFP_DATES,  pre=3),
        "days_to_fomc":   dtf,
        "days_to_cpi":    dtc,
        "days_to_nfp":    dtn,
        "fomc_proximity": round(float(np.exp(-dtf / 5.0)), 4),
        "cpi_proximity":  round(float(np.exp(-dtc / 3.0)), 4),
        "nfp_proximity":  round(float(np.exp(-dtn / 3.0)), 4),
        "any_event_week": _within(FOMC_DATES, 7) or _within(CPI_DATES, 7) or _within(NFP_DATES, 7),
    }
