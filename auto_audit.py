"""
auto_audit.py
-------------
Autonomous self-audit system that runs every 15 minutes inside the scheduler.

Checks performed every cycle:
  1. Direction bias — if >65% of recent signals share the same direction,
     the system is stuck. Calls apply_regime_shift_penalty() automatically.
  2. Accuracy degradation — if rolling-20 accuracy drops below 40%,
     weights are partially reset toward defaults and a recalibration is logged.
  3. Consecutive wrong streak — if 5+ predictions in a row are wrong,
     triggers an immediate weight reset regardless of overall accuracy.
  4. Wrong-prediction weight adjustment — calls analyze_and_adjust() for
     every resolved-wrong prediction that hasn't been analysed yet.
  5. Weights file sanity — coerces all keys/values to correct types,
     clamps to [MIN_WEIGHT, MAX_WEIGHT], adds any missing indicator keys.
  6. Self-audit log — writes a human-readable audit report every cycle to
     data_cache/self_audit_log.json so the app can surface it to the user.

All corrections are automatic. No user intervention required.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from adaptive_learning import (
    load_weights, save_weights, load_analysis_log, _append_log,
    analyze_and_adjust, apply_regime_shift_penalty,
    DEFAULT_WEIGHTS, MIN_WEIGHT, MAX_WEIGHT, WEIGHTS_FILE, CACHE_DIR,
)

log = logging.getLogger(__name__)

PREDICTIONS_FILE = CACHE_DIR / "live_predictions.json"
AUDIT_LOG_FILE   = CACHE_DIR / "self_audit_log.json"

# How many recent predictions to examine for bias / accuracy
ROLLING_WINDOW   = 20
BIAS_THRESHOLD   = 0.65   # >65% same direction = biased
ACCURACY_FLOOR   = 0.40   # <40% rolling accuracy = degrade alert
STREAK_LIMIT     = 5      # N consecutive wrongs = immediate reset


def _load_predictions() -> list:
    if not PREDICTIONS_FILE.exists():
        return []
    try:
        return json.loads(PREDICTIONS_FILE.read_text())
    except Exception:
        return []


def _save_audit_log(entry: dict) -> None:
    try:
        log_data: list = []
        if AUDIT_LOG_FILE.exists():
            try:
                log_data = json.loads(AUDIT_LOG_FILE.read_text())
            except Exception:
                log_data = []
        log_data.append(entry)
        log_data = log_data[-200:]  # keep last 200 audit cycles
        AUDIT_LOG_FILE.write_text(json.dumps(log_data, indent=2))
    except Exception:
        pass


# ── 1. Weights file sanity ────────────────────────────────────────────────────

def _sanitise_weights() -> dict:
    """Coerce all weights to float, clamp to [MIN, MAX], add missing keys."""
    try:
        raw = json.loads(WEIGHTS_FILE.read_text()) if WEIGHTS_FILE.exists() else {}
    except Exception:
        raw = {}

    weights: dict[str, float] = {}
    for k, default_v in DEFAULT_WEIGHTS.items():
        try:
            v = float(raw.get(k, default_v))
        except (TypeError, ValueError):
            v = float(default_v)
        weights[k] = max(MIN_WEIGHT, min(MAX_WEIGHT, v))

    # Also preserve non-weight metadata keys
    meta = {k: v for k, v in raw.items() if k not in DEFAULT_WEIGHTS}
    return weights, meta


# ── 2. Direction bias check ───────────────────────────────────────────────────

def _check_bias(resolved: list) -> tuple[float, float, str]:
    """
    Returns (up_rate, recent_accuracy, bias_label).
    up_rate = fraction of UP predictions in the rolling window.
    """
    window = resolved[-ROLLING_WINDOW:]
    if not window:
        return 0.5, 0.5, "insufficient data"

    total    = len(window)
    up_count = sum(1 for p in window if p.get("direction", 1) == 1)
    # correct = prediction matched actual direction (direction==outcome)
    correct  = sum(1 for p in window
                   if p.get("outcome") is not None
                   and p.get("direction") == p.get("outcome"))

    up_rate  = up_count / total
    accuracy = correct / total

    if up_rate > BIAS_THRESHOLD:
        bias = f"over-bullish ({up_rate:.0%} UP calls)"
    elif up_rate < (1 - BIAS_THRESHOLD):
        bias = f"over-bearish ({(1-up_rate):.0%} DOWN calls)"
    else:
        bias = "balanced"

    return up_rate, accuracy, bias


# ── 3. Consecutive wrong streak ───────────────────────────────────────────────

def _wrong_streak(resolved: list) -> int:
    streak = 0
    for p in reversed(resolved):
        outcome = p.get("outcome")
        if outcome is None:
            break
        # wrong = prediction did NOT match actual outcome
        if p.get("direction") != outcome:
            streak += 1
        else:
            break
    return streak


# ── 4. Analyse unaudited wrong predictions ────────────────────────────────────

def _audit_wrong_predictions(all_preds: list) -> int:
    """
    Call analyze_and_adjust() for any resolved-wrong prediction that has
    not yet been analysed (lacks 'audited' flag).  Returns count processed.
    """
    count = 0
    updated = []
    for p in all_preds:
        outcome = p.get("outcome")
        # trigger on any resolved wrong prediction (direction != outcome)
        was_wrong = (outcome is not None and
                     p.get("direction") is not None and
                     p.get("direction") != outcome)
        if (was_wrong and
                not p.get("audited") and
                p.get("indicator_scores")):
            result = analyze_and_adjust({
                "correct":               False,
                "actual_price":          p.get("actual_price"),
                "price_at_prediction":   p.get("price_at_signal", p.get("price")),
                "predicted_direction":   p.get("direction", 1),
                "indicator_scores":      p.get("indicator_scores", {}),
                "horizon_label":         p.get("horizon", "1H"),
                "made_at":               p.get("made_on", ""),
            })
            if result:
                p["audited"] = True
                count += 1
        updated.append(p)

    # Persist audited flags back
    if count > 0:
        try:
            PREDICTIONS_FILE.write_text(json.dumps(updated, indent=2))
        except Exception:
            pass
    return count


# ── 5. Partial weight reset ───────────────────────────────────────────────────

def _partial_reset(weights: dict, reason: str, blend: float = 0.40) -> dict:
    """
    Blend current weights toward defaults by `blend` factor (0 = no change,
    1 = full reset).  Preserves relative learning but prevents runaway drift.
    """
    new_weights = {}
    changes     = {}
    for k, default_v in DEFAULT_WEIGHTS.items():
        old = weights.get(k, default_v)
        new = round(old * (1 - blend) + default_v * blend, 5)
        new = max(MIN_WEIGHT, min(MAX_WEIGHT, new))
        new_weights[k] = new
        if abs(new - old) > 0.001:
            changes[k] = {"old": round(old, 4), "new": round(new, 4)}

    _append_log({
        "timestamp": datetime.utcnow().isoformat(),
        "type":      "auto_partial_reset",
        "reason":    reason,
        "blend":     blend,
        "changes":   changes,
    })
    log.warning("AUTO RESET (blend=%.0f%%): %s. Changes: %s", blend * 100, reason, changes)
    return new_weights


# ── ML accuracy calibration ───────────────────────────────────────────────────

def _calibrate_ml_weight(resolved: list, weights: dict) -> dict:
    """
    Directly calibrate the ML Forecast weight using the live_predictions track record.
    The live predictions ARE the ML model's output, so this measures ML accuracy directly.
    If ML has been consistently wrong (< 45%), penalise its weight.
    If consistently right (> 60%), reward it.
    """
    if len(resolved) < 10:
        return {}

    window      = resolved[-20:]   # rolling 20-day window
    ml_correct  = sum(1 for p in window
                      if p.get("outcome") is not None
                      and p.get("direction") == p.get("outcome"))
    ml_accuracy = ml_correct / len(window)
    old_w       = weights.get("ML Forecast", DEFAULT_WEIGHTS.get("ML Forecast", 1.0))

    if ml_accuracy > 0.60:
        new_w = min(round(old_w * 1.04, 5), MAX_WEIGHT)
        direction = "rewarded"
    elif ml_accuracy < 0.45:
        new_w = max(round(old_w * 0.94, 5), MIN_WEIGHT)
        direction = "penalised"
    else:
        return {}   # within acceptable range

    if abs(new_w - old_w) < 0.001:
        return {}

    changes = {"ML Forecast": {
        "old": round(old_w, 4), "new": round(new_w, 4),
        "direction": direction, "ml_accuracy": round(ml_accuracy, 3),
        "window_n": len(window),
    }}
    weights["ML Forecast"] = new_w
    _append_log({
        "timestamp":    datetime.utcnow().isoformat(),
        "type":         "ml_weight_calibration",
        "ml_accuracy":  round(ml_accuracy, 3),
        "direction":    direction,
        "changes":      changes,
    })
    log.info("ML calibration: acc=%.1f%%  weight %.3f → %.3f (%s)",
             ml_accuracy * 100, old_w, new_w, direction)
    return changes


# ── Main audit entry point ────────────────────────────────────────────────────

def run_audit() -> dict:
    """
    Run the full self-audit cycle. Called by the scheduler after every
    quick refresh. Returns a summary dict for the audit log.
    """
    audit = {
        "timestamp":       datetime.utcnow().isoformat(),
        "actions":         [],
        "warnings":        [],
        "weights_before":  {},
        "weights_after":   {},
    }

    # ── Step 1: sanitise weights file ────────────────────────────────────────
    weights, meta = _sanitise_weights()
    audit["weights_before"] = {k: round(v, 4) for k, v in weights.items()}

    # ── Step 2: load all predictions ─────────────────────────────────────────
    all_preds = _load_predictions()
    resolved  = [p for p in all_preds if p.get("outcome") is not None]

    if not resolved:
        audit["warnings"].append("No resolved predictions yet — skipping bias/accuracy checks.")
        _save_audit_log(audit)
        return audit

    # ── Step 3: direction bias + accuracy check ───────────────────────────────
    up_rate, accuracy, bias_label = _check_bias(resolved)
    audit["rolling_accuracy"]   = round(accuracy, 3)
    audit["direction_up_rate"]  = round(up_rate, 3)
    audit["bias"]               = bias_label

    log.info("AUDIT: accuracy=%.1f%%  up_rate=%.1f%%  bias=%s",
             accuracy * 100, up_rate * 100, bias_label)

    # Apply regime penalty if biased
    if bias_label != "balanced" and bias_label != "insufficient data":
        up_bias = up_rate - 0.50   # positive = over-bullish
        changes = apply_regime_shift_penalty(up_bias, accuracy)
        if changes:
            msg = f"Regime bias auto-corrected: {bias_label}"
            audit["actions"].append(msg)
            log.info("AUDIT: %s", msg)

    # ── Step 4: accuracy floor check ─────────────────────────────────────────
    if accuracy < ACCURACY_FLOOR:
        reason = f"Rolling-{ROLLING_WINDOW} accuracy {accuracy:.1%} below {ACCURACY_FLOOR:.0%} floor"
        audit["warnings"].append(reason)
        weights = _partial_reset(weights, reason, blend=0.35)
        audit["actions"].append(f"Partial weight reset: {reason}")

    # ── Step 5: consecutive wrong streak ─────────────────────────────────────
    streak = _wrong_streak(resolved)
    audit["wrong_streak"] = streak
    if streak >= STREAK_LIMIT:
        reason = f"{streak} consecutive wrong predictions"
        weights = _partial_reset(weights, reason, blend=0.50)
        audit["actions"].append(f"Aggressive weight reset: {reason}")
        log.warning("AUDIT: %s", reason)

    # ── Step 6a: ML-specific weight calibration ───────────────────────────────
    ml_changes = _calibrate_ml_weight(resolved, weights)
    if ml_changes:
        ml_acc = list(ml_changes.values())[0].get("ml_accuracy", 0)
        ml_dir = list(ml_changes.values())[0].get("direction", "")
        audit["actions"].append(
            f"ML Forecast weight {ml_dir} (rolling accuracy {ml_acc:.1%})"
        )

    # ── Step 6b: save sanitised weights ──────────────────────────────────────
    save_weights(weights, {
        k: v for k, v in meta.items()
        if k not in ("last_updated",)
    })
    audit["weights_after"] = {k: round(weights[k], 4) for k in DEFAULT_WEIGHTS}

    # ── Step 7: analyse unaudited wrong predictions ───────────────────────────
    audited_count = _audit_wrong_predictions(all_preds)
    if audited_count:
        msg = f"Auto-adjusted weights for {audited_count} new wrong prediction(s)"
        audit["actions"].append(msg)
        log.info("AUDIT: %s", msg)

    # ── Step 8: final summary ─────────────────────────────────────────────────
    if not audit["actions"]:
        audit["actions"].append("All checks passed — no corrections needed.")

    _save_audit_log(audit)
    log.info("AUDIT complete. Actions: %s", "; ".join(audit["actions"]))
    return audit
