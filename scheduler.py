"""
scheduler.py
Two-speed gold prediction scheduler:
  • Quick refresh every QUICK_MINUTES  — re-downloads only Yahoo prices (~15-30s),
    uses the saved model to emit a fresh prediction.
  • Full retrain  every RETRAIN_HOURS  — re-downloads all 160+ series, re-runs the
    full walk-forward backtest, saves a new model.

Run via its own workflow:
    python scheduler.py
"""
import json
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path

from gold_model import (
    load_all_data, make_features, walk_forward,
    save_live_prediction, resolve_live_predictions,
    save_raw_cache, load_raw_cache,
    save_model_state, load_model_state,
    quick_predict,
    multi_horizon_predict, save_multi_horizon_predictions,
    fetch_news_sentiment,
    CACHE_DIR, HORIZONS,
)
from auto_audit import run_audit
from telegram_alerts import check_price_alerts, alert_retrain_done

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Settings ──────────────────────────────────────────────────────────────────
HORIZON_LABEL   = "Next day"   # which horizon to predict
TRAIN_YEARS     = 3            # training window in years
RETRAIN_EVERY   = 63           # walk-forward retrain cadence (trading days)
DATA_START      = "2010-01-01" # data start date
RETRAIN_HOURS   = 24           # full backtest every N hours
QUICK_MINUTES   = 15           # quick prediction refresh every N minutes
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_FILE  = CACHE_DIR / "latest_results.pkl"
STATE_FILE    = CACHE_DIR / "scheduler_state.json"
PROGRESS_FILE = CACHE_DIR / "scheduler_progress.json"


def save_state(status: str, last_full_run: float | None = None,
               last_quick: float | None = None, error: str | None = None):
    now = time.time()
    state = {
        "status":          status,
        "last_run":        last_full_run,
        "next_run":        (last_full_run + RETRAIN_HOURS * 3600) if last_full_run else None,
        "last_quick":      last_quick or now,
        "next_quick":      (last_quick or now) + QUICK_MINUTES * 60,
        "horizon":         HORIZON_LABEL,
        "error":           error,
        "updated_at":      datetime.utcnow().isoformat(),
    }
    STATE_FILE.write_text(json.dumps(state, indent=2))


def save_progress(percent: float, message: str, phase: str = ""):
    PROGRESS_FILE.write_text(json.dumps({
        "percent":    round(percent, 1),
        "message":    message,
        "phase":      phase,
        "updated_at": datetime.utcnow().isoformat(),
    }, indent=2))


# ── Full backtest ─────────────────────────────────────────────────────────────

def run_full():
    log.info("─── Full backtest ───────────────────────────────────────────")
    save_state("running", error=None)
    save_progress(0.0, "Starting full backtest…", "init")

    target_col, return_col, horizon_days = HORIZONS[HORIZON_LABEL]

    # Phase 1 – data (0 → 55%)
    # 107 Yahoo + 1 OHLCV + 77 FRED + 3 (GPR/EPU/COT) = 188
    TOTAL_SERIES = 188
    downloaded = [0]

    def data_cb(msg, frac):
        downloaded[0] += 1
        pct = min(downloaded[0] / TOTAL_SERIES * 55, 55)
        save_progress(pct, msg, "data")

    log.info("Loading full data (start=%s)…", DATA_START)
    save_progress(1.0, "Downloading market data (~160 series)…", "data")
    raw = load_all_data(start=DATA_START, progress_callback=data_cb)
    log.info("Loaded %d series. Caching raw data…", raw.shape[1])
    save_raw_cache(raw)

    # Phase 2 – features (55 → 65%)
    save_progress(55.0, f"Building features from {raw.shape[1]} series…", "features")
    features = make_features(raw)
    n_feat = len([c for c in features.columns
                  if not c.startswith("target_") and not c.startswith("next_return_")])
    log.info("Built %d features. Running walk-forward…", n_feat)

    # Phase 3 – walk-forward (65 → 98%)
    save_progress(65.0, f"Walk-forward training ({n_feat} features)…", "training")

    def wf_cb(msg, frac):
        save_progress(65.0 + frac * 33.0, msg, "training")

    results = walk_forward(
        features,
        target_col=target_col,
        return_col=return_col,
        train_years=TRAIN_YEARS,
        retrain_every=RETRAIN_EVERY,
        progress_callback=wf_cb,
    )

    log.info("Done. Accuracy=%.2f%%  Sharpe=%.2f  N=%d",
             results["accuracy"] * 100, results["sharpe"], results["n_predictions"])
    save_progress(98.0, "Saving results…", "saving")

    # Save model state for quick refreshes
    save_model_state(results["model"], results["feature_cols"])
    log.info("Model state saved.")

    # Save display results (strip model object)
    results_to_save = {k: v for k, v in results.items() if k != "model"}
    results_to_save["horizon_label"] = HORIZON_LABEL
    results_to_save["run_at"] = datetime.utcnow().isoformat()
    RESULTS_FILE.write_bytes(pickle.dumps(results_to_save))
    log.info("Results saved.")

    # Multi-horizon forecast (1d, 2d, 5d) — fast final-fit models
    save_progress(98.5, "Multi-horizon forecast (1d, 2d, 5d)…", "saving")
    try:
        mh = multi_horizon_predict(features, results["feature_cols"])
        save_multi_horizon_predictions(mh)
        log.info("Multi-horizon: %s",
                 {k: ("UP" if v["direction"] == 1 else "DOWN") for k, v in mh.items()})
    except Exception as e:
        log.warning("Multi-horizon predict failed: %s", e)

    # Save live prediction
    latest_pred  = int(results["predictions"].iloc[-1])
    latest_proba = float(results["probas"].iloc[-1])
    latest_date  = results["predictions"].index[-1].to_pydatetime()
    save_live_prediction(latest_date, latest_pred, latest_proba,
                         HORIZON_LABEL, horizon_days)
    log.info("Live prediction: %s  %.1f%%",
             "UP" if latest_pred == 1 else "DOWN", latest_proba * 100)

    # Resolve past predictions
    import yfinance as yf
    try:
        gold = yf.download("GC=F", period="1y", progress=False, auto_adjust=True)["Close"]
        if hasattr(gold, "squeeze"):
            gold = gold.squeeze()
        resolve_live_predictions(gold.dropna())
        log.info("Live predictions resolved.")
    except Exception as e:
        log.warning("Could not resolve predictions: %s", e)

    # Self-audit — detect bias / accuracy degradation and auto-correct weights
    try:
        audit = run_audit()
        log.info("Self-audit: accuracy=%.1f%%  bias=%s  actions=%s",
                 audit.get("rolling_accuracy", 0) * 100,
                 audit.get("bias", "?"),
                 "; ".join(audit.get("actions", [])))
    except Exception as e:
        log.warning("Self-audit failed: %s", e)

    # Telegram: notify retrain complete
    try:
        cfg_retrain = __import__("telegram_alerts").load_config()
        if cfg_retrain.get("alert_retrain_done"):
            alert_retrain_done(
                results["accuracy"],
                "; ".join(audit.get("actions", [])) if "audit" in dir() else "",
            )
    except Exception as e:
        log.warning("Retrain alert failed: %s", e)

    now = time.time()
    save_state("idle", last_full_run=now, last_quick=now)
    save_progress(100.0, "Complete ✓", "done")
    log.info("Full backtest complete.")
    return now


# ── Quick prediction refresh ──────────────────────────────────────────────────

def run_quick(last_full_run: float):
    log.info("─── Quick refresh ───────────────────────────────────────────")
    save_state("refreshing", last_full_run=last_full_run)
    save_progress(10.0, "Refreshing market prices…", "quick")

    _, _, horizon_days = HORIZONS[HORIZON_LABEL]

    # Load model + refresh Yahoo prices + rebuild features
    model, feature_cols = load_model_state()
    cached = load_raw_cache(max_age_hours=25)

    if model is None or cached is None:
        log.warning("No model or no cache. Skipping quick refresh.")
        save_state("idle", last_full_run=last_full_run, last_quick=time.time())
        save_progress(100.0, "Complete ✓", "done")
        return

    save_progress(30.0, "Rebuilding features…", "quick")
    raw = load_all_data(start=DATA_START, yahoo_only=True, cached_raw=cached)
    features = make_features(raw)

    # 1-day quick prediction
    avail = features[feature_cols].dropna()
    if avail.empty:
        log.warning("No complete feature rows. Skipping.")
        save_state("idle", last_full_run=last_full_run, last_quick=time.time())
        save_progress(100.0, "Complete ✓", "done")
        return

    latest    = avail.iloc[[-1]]
    pred      = int(model.predict(latest)[0])
    proba     = float(model.predict_proba(latest)[0, 1])
    pred_date = latest.index[-1].to_pydatetime()

    save_live_prediction(pred_date, pred, proba, HORIZON_LABEL, horizon_days)
    log.info("Quick prediction: %s  %.1f%%",
             "UP" if pred == 1 else "DOWN", proba * 100)

    # Multi-horizon forecast (1d, 2d, 5d)
    save_progress(70.0, "Multi-horizon forecast (1d, 2d, 5d)…", "quick")
    try:
        mh = multi_horizon_predict(features, feature_cols)
        save_multi_horizon_predictions(mh)
        log.info("Multi-horizon: %s",
                 {k: ("UP" if v["direction"] == 1 else "DOWN") for k, v in mh.items()})
    except Exception as e:
        log.warning("Multi-horizon predict failed: %s", e)

    # News sentiment fetch (Yahoo Finance RSS, no API key)
    save_progress(85.0, "Fetching news sentiment…", "quick")
    try:
        sent = fetch_news_sentiment()
        log.info("News sentiment: score=%.2f  bull=%d  bear=%d  total=%d",
                 sent.get("score", 0), sent.get("bullish_n", 0),
                 sent.get("bearish_n", 0), sent.get("total_n", 0))
    except Exception as e:
        log.warning("News sentiment fetch failed: %s", e)

    # Resolve past predictions + grab live price for alert checks
    import yfinance as yf
    live_price = None
    try:
        gold = yf.download("GC=F", period="30d", progress=False, auto_adjust=True)["Close"]
        if hasattr(gold, "squeeze"):
            gold = gold.squeeze()
        gold = gold.dropna()
        resolve_live_predictions(gold)
        if not gold.empty:
            live_price = float(gold.iloc[-1])
    except Exception:
        pass

    # Self-audit — runs every quick cycle (every 15 min), auto-corrects bias
    save_progress(92.0, "Running self-audit…", "audit")
    try:
        audit = run_audit()
        log.info("Self-audit: accuracy=%.1f%%  bias=%s  actions=%s",
                 audit.get("rolling_accuracy", 0) * 100,
                 audit.get("bias", "?"),
                 "; ".join(audit.get("actions", [])))
    except Exception as e:
        log.warning("Self-audit failed: %s", e)

    # Telegram price-level alerts
    if live_price is not None:
        try:
            check_price_alerts(live_price)
            log.info("Alert check: live price=%.2f", live_price)
        except Exception as e:
            log.warning("Alert check failed: %s", e)

    now = time.time()
    save_state("idle", last_full_run=last_full_run, last_quick=now)
    save_progress(100.0, "Complete ✓", "done")
    log.info("Quick refresh complete.")
    return now


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    log.info("Gold Predictor Scheduler starting up.")
    log.info("Horizon=%s  FullRetrain=%dh  QuickRefresh=%dmin",
             HORIZON_LABEL, RETRAIN_HOURS, QUICK_MINUTES)
    CACHE_DIR.mkdir(exist_ok=True)

    last_full_run: float | None = None
    last_quick: float = 0.0

    while True:
        now = time.time()
        need_full  = (last_full_run is None or
                      now - last_full_run >= RETRAIN_HOURS * 3600)
        need_quick = (now - last_quick >= QUICK_MINUTES * 60)

        try:
            if need_full:
                last_full_run = run_full()
                last_quick    = last_full_run
            elif need_quick:
                run_quick(last_full_run)
                last_quick = time.time()
            else:
                # Wait until next event
                next_full  = (last_full_run or 0) + RETRAIN_HOURS * 3600
                next_quick = last_quick + QUICK_MINUTES * 60
                sleep_secs = max(10, min(next_full, next_quick) - time.time())
                log.info("Sleeping %.0fs (next quick in %.0fs)…",
                         sleep_secs, next_quick - time.time())
                time.sleep(sleep_secs)

        except Exception as e:
            log.error("Run failed: %s", e, exc_info=True)
            save_state("error", last_full_run=last_full_run, error=str(e))
            log.info("Retrying in 5 minutes…")
            time.sleep(300)


if __name__ == "__main__":
    main()
