"""
adaptive_learning.py
--------------------
Background self-learning system for the Gold Price Predictor.

When a prediction is wrong this module:
  1. Identifies which indicators voted for the wrong direction.
  2. Penalises their weights and rewards those that were correct.
  3. Persists adjusted weights to disk (adaptive_weights.json).
  4. Appends a human-readable analysis entry to weight_analysis_log.json.

The score_signals() function in day_trading.py loads these weights on every
call, so future predictions automatically reflect the learned corrections.
"""

import json
from datetime import datetime
from pathlib import Path

CACHE_DIR        = Path(__file__).parent / "data_cache"
WEIGHTS_FILE     = CACHE_DIR / "adaptive_weights.json"
ANALYSIS_LOG     = CACHE_DIR / "weight_analysis_log.json"

# ── Default weights (all 1.0 = equal contribution) ────────────────────────────
DEFAULT_WEIGHTS: dict[str, float] = {
    "RSI":         1.0,
    "MACD":        1.0,
    "Bollinger":   1.0,
    "EMA Trend":   1.0,
    "VWAP":        1.0,
    "Momentum":    1.0,
    "Candlestick": 1.2,   # slightly upweighted — patterns have documented 60-78% accuracy
    "GVZ":         1.1,   # Gold Volatility Index — contrarian fear/greed signal for gold
    "Regime":      1.3,   # Macro regime (real yields + DXY + risk-off) — context multiplier
    "4H Trend":    1.2,   # 4-hour intermediate timeframe — multi-TF confluence
    "Divergence":  1.1,   # RSI/price divergence — leading reversal indicator
    "ML Forecast": 1.0,   # ML ensemble — weight starts equal, self-adjusts as track record builds
}

# ── Tuning knobs ──────────────────────────────────────────────────────────────
PENALTY_RATE = 0.06   # wrong indicator weight reduced by 6 %
REWARD_RATE  = 0.03   # correct indicator weight increased by 3 %
MIN_WEIGHT   = 0.30   # floor — never fully silence an indicator
MAX_WEIGHT   = 2.80   # ceiling — cap amplification
MAX_LOG      = 300    # keep last N analysis entries


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_weights() -> dict[str, float]:
    """Load current adaptive weights from disk, filling missing keys with defaults."""
    if not WEIGHTS_FILE.exists():
        return DEFAULT_WEIGHTS.copy()
    try:
        raw = json.loads(WEIGHTS_FILE.read_text())
        out = {}
        for k, default_v in DEFAULT_WEIGHTS.items():
            out[k] = float(raw.get(k, default_v))
        return out
    except Exception:
        return DEFAULT_WEIGHTS.copy()


def save_weights(weights: dict[str, float], extra: dict | None = None) -> None:
    data: dict = {k: round(v, 5) for k, v in weights.items()}
    data["last_updated"] = datetime.utcnow().isoformat()
    if extra:
        data.update(extra)
    try:
        WEIGHTS_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def load_analysis_log() -> list:
    if not ANALYSIS_LOG.exists():
        return []
    try:
        return json.loads(ANALYSIS_LOG.read_text())
    except Exception:
        return []


def _append_log(entry: dict) -> None:
    log = load_analysis_log()
    log.append(entry)
    log = log[-MAX_LOG:]
    try:
        ANALYSIS_LOG.write_text(json.dumps(log, indent=2))
    except Exception:
        pass


# ── Core analysis & adjustment ────────────────────────────────────────────────

def analyze_and_adjust(prediction: dict) -> dict | None:
    """
    Analyse a *wrong* resolved prediction, adjust indicator weights, and log
    the reasoning.  Returns a log-entry dict, or None if nothing was done.

    Expected keys in `prediction`:
        correct           bool  — must be False to trigger adjustment
        actual_price      float
        price_at_prediction float
        predicted_direction int (+1 / -1)
        indicator_scores  dict  {indicator_name: raw_score_int}
        horizon_label     str
        made_at           str  (ISO timestamp)
    """
    if prediction.get("correct") is not False:
        return None
    if "indicator_scores" not in prediction:
        return None

    actual_price = prediction.get("actual_price")
    base_price   = prediction.get("price_at_prediction")
    if actual_price is None or base_price is None:
        return None

    actual_dir    = 1 if actual_price > base_price else -1
    predicted_dir = prediction.get("predicted_direction", 0)

    ind_scores = prediction["indicator_scores"]
    weights    = load_weights()

    adjustments: dict = {}
    error_sources: list[str] = []
    correct_sources: list[str] = []

    for indicator, score in ind_scores.items():
        if score == 0 or indicator not in weights:
            continue

        ind_voted = 1 if score > 0 else -1
        old_w     = weights[indicator]

        if ind_voted != actual_dir:
            # Wrong — penalise
            new_w = max(round(old_w * (1 - PENALTY_RATE), 5), MIN_WEIGHT)
            direction_str = "UP" if ind_voted > 0 else "DOWN"
            actual_str    = "UP" if actual_dir > 0 else "DOWN"
            error_sources.append(
                f"{indicator} signalled {direction_str} (score {score:+d}) "
                f"but price moved {actual_str} — weight {old_w:.3f} → {new_w:.3f}"
            )
            adjustments[indicator] = {"direction": "penalised", "old": old_w, "new": new_w}
        else:
            # Correct — reward
            new_w = min(round(old_w * (1 + REWARD_RATE), 5), MAX_WEIGHT)
            correct_sources.append(
                f"{indicator} correctly called {'UP' if actual_dir > 0 else 'DOWN'} "
                f"— weight {old_w:.3f} → {new_w:.3f}"
            )
            adjustments[indicator] = {"direction": "rewarded", "old": old_w, "new": new_w}

        weights[indicator] = new_w

    # Save updated weights
    total_analyses = json.loads(WEIGHTS_FILE.read_text()).get("total_analyses", 0) + 1 \
        if WEIGHTS_FILE.exists() else 1
    save_weights(weights, {"total_analyses": total_analyses})

    # Build log entry
    entry = {
        "timestamp":       datetime.utcnow().isoformat(),
        "horizon_label":   prediction.get("horizon_label", "?"),
        "made_at":         prediction.get("made_at", "?"),
        "predicted":       "UP" if predicted_dir > 0 else "DOWN",
        "actual":          "UP" if actual_dir > 0 else "DOWN",
        "error_sources":   error_sources,
        "correct_sources": correct_sources,
        "adjustments":     adjustments,
        "weights_after":   {k: round(v, 4) for k, v in weights.items()},
        "analysis_number": total_analyses,
    }
    _append_log(entry)
    return entry


def apply_regime_shift_penalty(up_bias: float, recent_accuracy: float) -> dict:
    """
    When the backtest shows the model is systematically over-bullish or
    over-bearish, adjust day-trading indicator weights to counteract the bias.

    up_bias  = (model UP-call rate) - (actual UP rate)
               positive → over-bullish, negative → over-bearish
    Returns a dict of changes made (empty if bias is within tolerance).
    """
    if abs(up_bias) <= 0.10:
        return {}

    weights = load_weights()
    changes: dict = {}

    TREND_INDICATORS      = ["EMA Trend", "Momentum", "MACD", "Regime"]
    CONTRARIAN_INDICATORS = ["GVZ", "RSI", "Bollinger"]

    penalty_rate = min(0.22, abs(up_bias) * 1.5)
    reward_rate  = min(0.11, abs(up_bias) * 0.8)

    # Penalize indicators aligned with the bias direction
    for ind in TREND_INDICATORS:
        if ind not in weights:
            continue
        old = weights[ind]
        new = max(round(old * (1 - penalty_rate), 5), MIN_WEIGHT)
        if new != old:
            weights[ind] = new
            changes[ind] = {"old": round(old, 4), "new": round(new, 4), "change": "penalised"}

    # Reward indicators that push against the bias
    for ind in CONTRARIAN_INDICATORS:
        if ind not in weights:
            continue
        old = weights[ind]
        new = min(round(old * (1 + reward_rate), 5), MAX_WEIGHT)
        if new != old:
            weights[ind] = new
            changes[ind] = {"old": round(old, 4), "new": round(new, 4), "change": "rewarded"}

    if changes:
        total = json.loads(WEIGHTS_FILE.read_text()).get("total_analyses", 0) \
            if WEIGHTS_FILE.exists() else 0
        save_weights(weights, {
            "total_analyses": total,
            "last_regime_calibration": datetime.utcnow().isoformat(),
            "regime_up_bias": round(up_bias, 4),
            "regime_accuracy": round(recent_accuracy, 4),
        })
        _append_log({
            "timestamp":       datetime.utcnow().isoformat(),
            "type":            "regime_shift_calibration",
            "up_bias":         round(up_bias, 4),
            "recent_accuracy": round(recent_accuracy, 4),
            "bias_direction":  "over-bullish" if up_bias > 0 else "over-bearish",
            "changes":         changes,
            "weights_after":   {k: round(v, 4) for k, v in weights.items()},
        })

    return changes


def summary_stats() -> dict:
    """Return a quick summary: current weights + recent analysis count."""
    weights = load_weights()
    log     = load_analysis_log()
    total   = json.loads(WEIGHTS_FILE.read_text()).get("total_analyses", 0) \
              if WEIGHTS_FILE.exists() else 0
    recent  = [e for e in log if e.get("timestamp", "") >= ""]  # all entries
    return {
        "weights":         weights,
        "total_analyses":  total,
        "log_entries":     len(log),
        "last_updated":    json.loads(WEIGHTS_FILE.read_text()).get("last_updated", "never")
                           if WEIGHTS_FILE.exists() else "never",
    }
