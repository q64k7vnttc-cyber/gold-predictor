"""
telegram_alerts.py
Sends Telegram messages for gold price alerts.

Reads bot token + chat ID from environment variables:
    TELEGRAM_BOT_TOKEN   — from BotFather
    TELEGRAM_CHAT_ID     — your personal chat ID (@userinfobot)

Config (stop/limit levels, which alerts are on) is stored in
data_cache/alert_config.json so the app UI can read/write it.
"""
import json
import os
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

CACHE_DIR   = Path("data_cache")
CONFIG_FILE = CACHE_DIR / "alert_config.json"
SENT_FILE   = CACHE_DIR / "alert_sent_log.json"

# ── Default config ────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "enabled":              True,
    "stop_level":           None,   # e.g. 4758.0
    "limit_level":          None,   # e.g. 4795.0
    "warning_buffer":       5,      # points from stop/limit to warn early
    "alert_price_levels":   True,   # warn near stop / limit
    "alert_signal_change":  True,   # when 15m signal flips strongly
    "alert_margin_warning": True,   # when available margin < threshold (A$)
    "margin_threshold":     300,    # A$ available margin warning level
    "alert_retrain_done":   False,  # when full model retrain completes
    "cooldown_minutes":     15,     # minimum minutes between same alert type
    # ── Open position tracking (used by revision alerts) ──────────────────────
    "position_direction":   None,   # "LONG" | "SHORT" | null
    "entry_price":          None,   # float — avg open price
    "alert_revision":       True,   # send revision / stop-adjust recommendations
    # ── Custom watch levels (trade-specific intermediate alerts) ──────────────
    # Each entry: {"level": float, "direction": "above"|"below",
    #              "label": str, "action": str, "cooldown_minutes": int}
    "watch_levels":         [],
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            cfg = json.loads(CONFIG_FILE.read_text())
            return {**DEFAULT_CONFIG, **cfg}
        except Exception:
            pass
    return dict(DEFAULT_CONFIG)


def save_config(cfg: dict):
    CACHE_DIR.mkdir(exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def _load_sent_log() -> dict:
    if SENT_FILE.exists():
        try:
            return json.loads(SENT_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_sent_log(log: dict):
    CACHE_DIR.mkdir(exist_ok=True)
    SENT_FILE.write_text(json.dumps(log, indent=2))


def _cooldown_ok(alert_key: str, cooldown_minutes: int) -> bool:
    """Return True if enough time has passed since this alert was last sent."""
    log = _load_sent_log()
    last = log.get(alert_key)
    if last is None:
        return True
    elapsed = time.time() - last
    return elapsed >= cooldown_minutes * 60


def _mark_sent(alert_key: str):
    log = _load_sent_log()
    log[alert_key] = time.time()
    _save_sent_log(log)


# ── Core sender ───────────────────────────────────────────────────────────────

def send_message(text: str) -> bool:
    """
    Send a Telegram message.  Returns True on success.
    Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment.
    """
    # Remove ALL whitespace (spaces, newlines, tabs) — tokens sometimes wrap
    # when copied from BotFather messages
    token   = "".join(os.environ.get("TELEGRAM_BOT_TOKEN", "").split())
    chat_id = "".join(os.environ.get("TELEGRAM_CHAT_ID", "").split())

    if not token or not chat_id:
        return False

    url     = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({
        "chat_id":    chat_id,
        "text":       text,
        "parse_mode": "HTML",
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except urllib.error.URLError:
        return False


def test_connection() -> tuple[bool, str]:
    """Send a test message. Returns (success, message)."""
    ok = send_message(
        "✅ <b>Gold Predictor connected!</b>\n"
        "Telegram alerts are working. You'll receive notifications for:\n"
        "• Price approaching stop/limit levels\n"
        "• Strong signal changes\n"
        "• Margin warnings"
    )
    if ok:
        return True, "Test message sent successfully!"
    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token:
        return False, "TELEGRAM_BOT_TOKEN secret is not set."
    if not chat_id:
        return False, "TELEGRAM_CHAT_ID secret is not set."
    return False, "Could not reach Telegram API. Check your token and chat ID."


# ── Typed alert senders ───────────────────────────────────────────────────────

def _pnl_line(cfg: dict, exit_price: float, per_point: float = 10.0) -> str:
    """Return a P&L summary line if entry_price is known, else empty string."""
    entry = cfg.get("entry_price")
    dirn  = cfg.get("position_direction")
    if not entry or not dirn:
        return ""
    pts = (entry - exit_price) if dirn == "SHORT" else (exit_price - entry)
    aud = pts * per_point
    sign = "+" if pts >= 0 else ""
    return f"\nResult: <b>{sign}{pts:.1f} pts  /  {sign}A${aud:.0f}</b>"


def alert_price_near_limit(current_price: float, limit_level: float, cfg: dict) -> bool:
    key = "near_limit"
    if not _cooldown_ok(key, cfg.get("cooldown_minutes", 15)):
        return False
    dirn     = cfg.get("position_direction", "LONG")
    pts_away = abs(limit_level - current_price)
    entry    = cfg.get("entry_price")
    pot_pts  = abs(level - entry) if (level := limit_level) and entry else None
    pot_str  = f"\nPotential profit: <b>+A${pot_pts*10:.0f}  (+{pot_pts:.0f} pts)</b>" if pot_pts else ""
    ok = send_message(
        f"🎯 <b>TP almost there</b> — {pts_away:.0f} pts to go\n"
        f"Target: <b>{limit_level:,.0f}</b>  ·  Now: <b>{current_price:,.0f}</b>\n"
        f"Watch IG — position may fill soon."
    )
    if ok:
        _mark_sent(key)
    return ok


def alert_price_near_stop(current_price: float, stop_level: float, cfg: dict) -> bool:
    key = "near_stop"
    if not _cooldown_ok(key, cfg.get("cooldown_minutes", 15)):
        return False
    dirn     = cfg.get("position_direction", "LONG")
    pts_away = round(abs(stop_level - current_price), 1)
    entry    = cfg.get("entry_price")
    loss_pts = abs(stop_level - entry) if entry else None
    loss_str = f"\nLoss if stopped: <b>−A${loss_pts*10:.0f}  (−{loss_pts:.0f} pts)</b>" if loss_pts else ""
    time_utc = datetime.utcnow().strftime("%H:%M UTC")
    ok = send_message(
        f"⚠️ <b>Stop {pts_away} pts away</b> — {dirn}\n"
        f"Stop: <b>{stop_level:,.0f}</b>  ·  Now: <b>{current_price:,.0f}</b>\n"
        f"Hold, close (IG Positions → Close), or widen stop (Edit)."
    )
    if ok:
        _mark_sent(key)
    return ok


def _auto_clear_position():
    """Wipe position fields from alert_config after a TP or stop is hit."""
    try:
        cfg = load_config()
        cfg["position_direction"] = None
        cfg["entry_price"]        = None
        cfg["stop_level"]         = None
        cfg["limit_level"]        = None
        cfg["watch_levels"]       = []
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass


def alert_limit_hit(current_price: float, limit_level: float) -> bool:
    key = "limit_hit"
    if not _cooldown_ok(key, 60):
        return False
    cfg      = load_config()
    pnl_line = _pnl_line(cfg, limit_level)
    ok = send_message(
        f"✅ <b>TP hit</b> near <b>{limit_level:,.0f}</b>{pnl_line}\n"
        f"Check IG — confirm closed."
    )
    if ok:
        _mark_sent(key)
        _auto_clear_position()   # ← position auto-cleared, no manual action needed
    return ok


def alert_stop_hit(current_price: float, stop_level: float) -> bool:
    key = "stop_hit"
    if not _cooldown_ok(key, 60):
        return False
    cfg      = load_config()
    pos_dir  = cfg.get("position_direction", "LONG")
    pnl_line = _pnl_line(cfg, stop_level)
    ok = send_message(
        f"🔴 <b>Stop hit</b> — {pos_dir} at <b>{stop_level:,.0f}</b>{pnl_line}\n"
        f"Open IG Positions. If still open → Close manually."
    )
    if ok:
        _mark_sent(key)
        _auto_clear_position()   # ← position auto-cleared, no manual action needed
    return ok


def alert_signal_change(timeframe: str, direction: str, strength: str) -> bool:
    key = f"signal_{timeframe}"
    if not _cooldown_ok(key, 30):
        return False

    cfg       = load_config()
    pos_dir   = cfg.get("position_direction")   # "LONG" | "SHORT" | None
    emoji     = "🟢" if direction == "BUY" else "🔴"
    is_long   = direction == "BUY"
    ig_action = "BUY" if is_long else "SELL"
    opposite  = "SHORT" if is_long else "LONG"
    time_utc  = datetime.utcnow().strftime("%H:%M UTC")

    # Does the user have a position that OPPOSES this new signal?
    has_opposing = pos_dir and pos_dir == opposite

    if has_opposing:
        ok = send_message(
            f"🔁 <b>{timeframe} flipped {direction} — against your {pos_dir}</b>\n"
            f"Now: <b>{current_price:,.0f}</b>  ·  {time_utc}\n"
            f"Hold if stop is wide. Close: IG Positions → Close."
        )
    else:
        ok = send_message(
            f"{emoji} <b>{timeframe}: {strength} {direction}</b>\n"
            f"Now: <b>{current_price:,.0f}</b>  ·  {time_utc}\n"
            f"Check app for levels, then IG → Trade → <b>{ig_action}</b> Gold Spot."
        )

    if ok:
        _mark_sent(key)
    return ok


def alert_margin_warning(available: float, threshold: float) -> bool:
    key = "margin_warning"
    if not _cooldown_ok(key, 30):
        return False
    ok = send_message(
        f"⚠️ <b>Low margin</b> — A${available:.0f} left\n"
        f"Reduce size or add funds in IG before next move."
    )
    if ok:
        _mark_sent(key)
    return ok


def alert_retrain_done(accuracy: float, actions: str) -> bool:
    key = "retrain_done"
    if not _cooldown_ok(key, 60):
        return False
    ok = send_message(
        f"🧠 <b>Model retrain complete</b>\n\n"
        f"Accuracy: <b>{accuracy:.1%}</b>\n"
        f"Actions:  {actions or 'none'}\n\n"
        f"⏱ {datetime.utcnow().strftime('%H:%M UTC')}"
    )
    if ok:
        _mark_sent(key)
    return ok


# ── Entry signal detector ─────────────────────────────────────────────────────

def check_entry_signal(current_price: float):
    """
    Evaluates current model outputs and sends a Telegram alert when a
    high-quality entry opportunity is detected.

    Criteria for a STRONG signal:
    - All 3 multi-horizon forecasts agree on direction
    - 1-day confidence >= 20%
    - Average confidence across all horizons >= 15%
    - News sentiment aligns with direction (or is neutral)
    - Cooldown: no repeat within 60 min
    """
    cfg = load_config()
    if not cfg.get("enabled") or not cfg.get("alert_signal_change", True):
        return

    try:
        mh_path   = Path("data_cache/multi_horizon_predictions.json")
        news_path = Path("data_cache/news_sentiment.json")
        intra_path = Path("data_cache/intraday_predictions.json")

        if not mh_path.exists():
            return

        mh   = json.loads(mh_path.read_text())
        news = json.loads(news_path.read_text()) if news_path.exists() else {}

        directions  = [v["direction"] for v in mh.values()]
        confidences = [v["confidence"] for v in mh.values()]
        one_day_conf = mh.get("1", {}).get("confidence", 0)
        avg_conf     = sum(confidences) / len(confidences) if confidences else 0

        # At least 2 of 3 horizons must agree (was: all 3)
        up_count   = directions.count(1)
        down_count = directions.count(0)
        if up_count < 2 and down_count < 2:
            return
        direction   = 1 if up_count >= 2 else 0   # majority vote
        dir_label   = "BUY 📈" if direction == 1 else "SELL 📉"
        dir_word    = "LONG"   if direction == 1 else "SHORT"
        news_score  = news.get("score", 0)

        # ── Session gate — don't alert during thin Asian market ───────────────
        # Signals fired at 03:00 UTC have poor execution quality: wide spreads,
        # low volume, unreliable momentum. Only alert during active sessions.
        _now_h = datetime.utcnow().hour
        _in_active_session = (8 <= _now_h < 21)   # London open → NY close
        if not _in_active_session:
            return   # suppress during Asian session

        # ── Quality gate ──────────────────────────────────────────────────────
        # 1-day confidence ≥ 22% or avg ≥ 18% — enough ML conviction to alert.
        # Technical STRONG BUY/SELL alerts are the primary trade signal;
        # this ML gate is a secondary cross-check.
        if one_day_conf < 0.22 and avg_conf < 0.22:
            return
        # News must not strongly oppose the direction
        if direction == 1 and news_score < -0.3:
            return
        if direction == 0 and news_score > 0.3:
            return

        # ── 1H technical signal cross-check ───────────────────────────────────
        # Check the live 1H signal for pullback/stop-zone warnings.
        # If the technical signal has quality warnings, suppress or downgrade.
        tech_warning = ""
        try:
            from day_trading import get_day_trading_signal as _gds
            _, _sig1h = _gds(period="60d", interval="1h")
            if _sig1h:
                if _sig1h.get("stop_zone_warning"):
                    dha = _sig1h.get("drop_from_high_atrs", 0)
                    tech_warning += f"\n⚠️ Price is {dha:.1f}× ATR below recent high — entering here is risky"
                if _sig1h.get("pullback_severity") == "strong":
                    tech_warning += "\n⚠️ Strong short-term pullback in progress — wait for stabilisation"
                if _sig1h.get("m15_confirmation") == "conflicts":
                    tech_warning += "\n⚡ 15-min momentum opposes signal direction"
        except Exception:
            pass

        # ── Intraday alignment bonus check ─────────────────────────────────────
        intra_agrees = False
        if intra_path.exists():
            intra = json.loads(intra_path.read_text())
            recent = intra[-6:] if len(intra) >= 6 else intra
            intra_dirs = [p.get("predicted_direction", -1) for p in recent]
            intra_matches = sum(1 for d in intra_dirs if d == direction)
            intra_agrees = intra_matches >= 4  # 4 of 6 agree

        # Build quality label
        if one_day_conf >= 0.35 and intra_agrees:
            quality = "HIGH CONVICTION"
            star    = "⭐⭐⭐"
        elif one_day_conf >= 0.30:
            quality = "MODERATE"
            star    = "⭐⭐"
        else:
            quality = "BUILDING"
            star    = "⭐"

        key = f"entry_{dir_word.lower()}"
        if not _cooldown_ok(key, 60):
            return

        # ── ATR-based risk sizing ──────────────────────────────────────────────
        # Use actual ATR from the 1H signal if available; fall back to a
        # reasonable default (0.8% of price ≈ $26 at $3300 gold).
        try:
            from day_trading import get_day_trading_signal as _gds2
            _, _s = _gds2(period="60d", interval="1h")
            _atr = _s.get("atr", current_price * 0.008) if _s else current_price * 0.008
        except Exception:
            _atr = current_price * 0.008
        stop_dist  = max(10, round(_atr * 1.5))    # 1.5 × ATR (as per day_trading.py)
        limit_dist = max(15, round(_atr * 2.5))    # 2.5 × ATR → R:R ≈ 1.67:1
        available  = 7000
        risk_amt   = available * 0.02   # A$140 (2% risk)

        # A$10 contract for MODERATE or HIGH CONVICTION
        # A$1  contract for BUILDING signals (lower risk exposure)
        if quality in ("HIGH CONVICTION", "MODERATE"):
            contract  = "A$10"
            per_point = 10   # A$ per point per contract
        else:
            contract  = "A$1"
            per_point = 1

        size = max(1, round(risk_amt / (stop_dist * per_point)))
        max_loss   = size * stop_dist   * per_point
        max_profit = size * limit_dist  * per_point

        # Build reason bullets
        reasons = []
        reasons.append(f"All 3 model horizons agree: <b>{dir_word}</b>")
        reasons.append(f"1-day confidence: <b>{mh['1']['confidence']:.0%}</b>")
        reasons.append(f"5-day confidence: <b>{mh['5']['confidence']:.0%}</b>")
        news_label = "bullish 📰" if news_score > 0.1 else ("bearish 📰" if news_score < -0.1 else "neutral")
        reasons.append(f"News sentiment: {news_label}")
        if intra_agrees:
            reasons.append("Intraday signals also aligned ✓")
        reasons_str = "\n".join(f"  • {r}" for r in reasons)

        _dir_emoji  = "🟢" if direction == 1 else "🔴"
        _ig_dir     = "BUY" if direction == 1 else "SELL"
        _stop_level = round(current_price - stop_dist  if direction == 1 else current_price + stop_dist)
        _lim_level  = round(current_price + limit_dist if direction == 1 else current_price - limit_dist)
        _spread_px  = round(current_price + 0.3 if direction == 1 else current_price - 0.3, 2)
        _time_utc   = datetime.utcnow().strftime("%H:%M UTC")
        ok = send_message(
            f"{_dir_emoji} <b>{dir_label}</b>  {star}  {_time_utc}\n"
            f"IG → Trade → Gold Spot → <b>{_ig_dir}</b>\n"
            f"Stop: <b>{_stop_level:,}</b>  ·  Limit: <b>{_lim_level:,}</b>  ·  Size: <b>{size}</b> ({contract})"
            + (f"\n⚠️ {tech_warning.strip()}" if tech_warning else "")
        )
        if ok:
            _mark_sent(key)

    except Exception:
        pass


# ── Revision / position management alerts ────────────────────────────────────

def check_revision_signal(current_price: float):
    """
    Fires a Telegram revision recommendation when an open position needs attention:

    Trigger 1 — Stop too tight:
        Remaining buffer to stop < 1× ATR → noise could stop you out.
        Recommends widening to 1.5× ATR.

    Trigger 2 — Stop zone active with configured stop:
        Technical stop-zone warning fires while stop < 1.5× ATR buffer.
        Bounce risk in support/resistance area.

    Trigger 3 — ML consensus flips against your trade:
        All three ML horizons now point opposite to your open position.
        Recommends reviewing the trade.

    Requires config: position_direction ("LONG"/"SHORT") and entry_price to be set.
    Cooldown: 2 hours between same-key alerts.
    """
    cfg = load_config()
    if not cfg.get("enabled") or not cfg.get("alert_revision", True):
        return

    direction = cfg.get("position_direction")   # "LONG" | "SHORT" | None
    entry     = cfg.get("entry_price")
    stop      = cfg.get("stop_level")

    if not direction or direction not in ("LONG", "SHORT"):
        return   # no open position logged

    is_long = direction == "LONG"

    # ── Get live ATR + technical state ────────────────────────────────────────
    try:
        from day_trading import get_day_trading_signal as _gds
        _, _sig1h = _gds(period="60d", interval="1h")
        if not _sig1h:
            return
        atr       = float(_sig1h.get("atr", current_price * 0.005))
        stop_zone = bool(_sig1h.get("stop_zone_warning", False))
        rsi       = float(_sig1h.get("rsi", 50))
    except Exception:
        return

    # ── Get ML consensus ──────────────────────────────────────────────────────
    try:
        mh_path = Path("data_cache/multi_horizon_predictions.json")
        if mh_path.exists():
            mh = json.loads(mh_path.read_text())
            ml_dirs = [v["direction"] for v in mh.values()]
            ml_all_agree = len(set(ml_dirs)) == 1
            ml_direction = ml_dirs[0] if ml_all_agree else None  # 1=UP 0=DOWN
        else:
            ml_all_agree, ml_direction = False, None
    except Exception:
        ml_all_agree, ml_direction = False, None

    # ── Evaluate triggers ─────────────────────────────────────────────────────
    triggers = []

    # Trigger 1: stop too tight (< 1 ATR remaining buffer)
    if stop is not None:
        _max_dist = current_price * 0.10
        if abs(stop - current_price) <= _max_dist:          # only act on fresh stops
            buffer_remaining = abs(stop - current_price)
            if buffer_remaining < atr * 1.0:
                triggers.append(
                    f"🔴 Stop only <b>{buffer_remaining:.1f} pts</b> away "
                    f"— less than 1× ATR ({atr:.1f}). Noise can stop you out."
                )

    # Trigger 2: stop zone active + stop configured and relatively close
    if stop_zone and stop is not None:
        _max_dist = current_price * 0.10
        if abs(stop - current_price) <= _max_dist:
            buffer_remaining = abs(stop - current_price)
            if buffer_remaining < atr * 1.5:
                triggers.append(
                    f"⚠️ Technical <b>stop-zone warning</b> active "
                    f"— price in a support/resistance area, bounce risk with only "
                    f"{buffer_remaining:.1f} pts to your stop."
                )

    # Trigger 3: ML consensus fully flips against open position
    if ml_all_agree and ml_direction is not None:
        ml_is_long  = (ml_direction == 1)
        ml_is_short = (ml_direction == 0)
        if (is_long and ml_is_short) or (not is_long and ml_is_long):
            ml_word = "BEARISH" if ml_is_short else "BULLISH"
            triggers.append(
                f"🔄 ML consensus has <b>flipped {ml_word}</b> — "
                f"all 3 horizons now oppose your {direction} position. "
                f"Consider reviewing."
            )

    if not triggers:
        return

    # ── Cooldown — combine all triggers under one key ─────────────────────────
    key = "revision_alert"
    if not _cooldown_ok(key, 120):   # 2-hour cooldown
        return

    # ── Build suggested stop using ATR ────────────────────────────────────────
    suggested_stop = round(current_price - atr * 1.5) if is_long else round(current_price + atr * 1.5)
    pnl_pts = (current_price - entry) * (1 if is_long else -1) if entry else None
    pnl_str = ""
    if pnl_pts is not None:
        pnl_aud = pnl_pts * 10   # A$10 contract
        sign    = "+" if pnl_pts >= 0 else ""
        pnl_str = f"\nP&L now: <b>{sign}{pnl_pts:.1f} pts / {sign}A${pnl_aud:.0f}</b>"

    # ── Determine action type (amend stop vs close/flip) ─────────────────────
    ml_flip = any("flipped" in t for t in triggers)
    stop_triggers = [t for t in triggers if "flipped" not in t]
    time_utc = datetime.utcnow().strftime("%H:%M UTC")
    ig_dir_word = "BUY" if is_long else "SELL"
    ig_close_word = "SELL" if is_long else "BUY"   # opposite to close

    if ml_flip and not stop_triggers:
        ok = send_message(
            f"⛔ <b>Close your {direction}</b>{pnl_str}\n"
            f"ML flipped against you. Now: <b>{current_price:,.0f}</b>\n"
            f"IG Positions → tap trade → <b>Close</b>"
        )
    else:
        ok = send_message(
            f"🔄 <b>Move stop to {suggested_stop:,}</b> — {direction}{pnl_str}\n"
            f"Stop too tight. IG Positions → Edit → Stop level → <b>{suggested_stop:,}</b>"
        )
    if ok:
        _mark_sent(key)


# ── Technical signal watcher (primary trade alert source) ────────────────────

_TECH_STATE_FILE = CACHE_DIR / "last_technical_signals.json"
_WATCH_TFS = [
    ("15-min",  "5d",   "15m",  45),   # name, period, interval, cooldown_min
    ("1-Hour",  "60d",  "1h",   60),
    ("4-Hour",  "120d", "4h",   90),
    ("Daily",   "2y",   "1d",   120),
]


def check_technical_signal_alerts(current_price: float):
    """
    Watches live technical signals (day_trading.py) for 15m/1H/4H/Daily.
    Fires a full IG deal-ticket Telegram alert whenever a timeframe flips to
    STRONG BUY or STRONG SELL — the same signals shown in the app.
    Tracks previous signal state so alerts only fire on genuine direction changes.
    """
    cfg = load_config()
    if not cfg.get("enabled") or not cfg.get("alert_signal_change", True):
        return

    _now_h = datetime.utcnow().hour
    if not (7 <= _now_h < 22):
        return   # suppress outside London+NY window

    try:
        prev_state = json.loads(_TECH_STATE_FILE.read_text()) if _TECH_STATE_FILE.exists() else {}
    except Exception:
        prev_state = {}

    try:
        from day_trading import get_day_trading_signal as _gds
    except Exception:
        return

    new_state = dict(prev_state)
    pos_dir   = cfg.get("position_direction")   # "LONG" | "SHORT" | None

    for tf_name, period, interval, cooldown in _WATCH_TFS:
        key = f"tech_{tf_name.replace('-','').replace(' ','_').lower()}"
        if not _cooldown_ok(key, cooldown):
            continue

        try:
            _, sig = _gds(period=period, interval=interval)
            if not sig:
                continue

            action = sig.get("action", "NEUTRAL").upper()

            # Track all signals so we can detect flips correctly
            new_state[tf_name] = action

            # Only alert on STRONG signals
            if "STRONG" not in action:
                continue

            prev_action = prev_state.get(tf_name, "")
            if action == prev_action:
                continue   # same strong signal already alerted

            # ── New STRONG BUY or STRONG SELL — build IG deal ticket ───────
            is_long   = "BUY" in action
            ig_dir    = "BUY"  if is_long else "SELL"
            opposite  = "SHORT" if is_long else "LONG"
            dir_emoji = "🟢"   if is_long else "🔴"
            conf      = sig.get("confidence", 0)
            entry     = float(sig.get("entry",     current_price))
            sl        = float(sig.get("stop_loss", entry - 30 if is_long else entry + 30))
            tp        = float(sig.get("target",    entry + 50 if is_long else entry - 50))
            rsi       = sig.get("rsi",  50)
            rr        = sig.get("risk_reward", 0)
            sl_pts    = abs(entry - sl)
            tp_pts    = abs(tp   - entry)
            sl_aud    = round(sl_pts * 10)
            tp_aud    = round(tp_pts * 10)
            deal_px   = round(entry + 0.30 if is_long else entry - 0.30, 1)
            time_utc  = datetime.utcnow().strftime("%H:%M UTC")

            # Warn if this signal opposes an open position
            has_opp   = pos_dir == opposite
            opp_note  = f"\n⚠️ Opposes your open <b>{pos_dir}</b> — close it first." if has_opp else ""

            ok = send_message(
                f"{dir_emoji} <b>{action} — {tf_name}</b>  {time_utc}"
                f"{opp_note}\n"
                f"IG → Trade → Gold Spot → <b>{ig_dir}</b>\n"
                f"Stop: <b>{sl:,.0f}</b>  ·  Limit: <b>{tp:,.0f}</b>  ·  R:R 1:{rr:.1f}"
            )
            if ok:
                _mark_sent(key)

        except Exception:
            continue

    try:
        _TECH_STATE_FILE.write_text(json.dumps(new_state, indent=2))
    except Exception:
        pass


# ── Position health — decisive close alert when signals flip against trade ─────

def check_position_health_alert(current_price: float):
    """
    Fires a decisive Telegram alert when the user has an open position AND
    4 or more timeframes have flipped against it, OR the stop is < 15 pts away.

    This is the "CLOSE NOW" alert — it's more aggressive than check_revision_signal
    because it uses the live technical signal engine (same as the app UI).

    Cooldown: 60 minutes to avoid repeat spam.
    """
    cfg       = load_config()
    if not cfg.get("enabled") or not cfg.get("alert_revision", True):
        return

    pos_dir   = cfg.get("position_direction")
    entry     = cfg.get("entry_price")
    stop      = cfg.get("stop_level")
    tp        = cfg.get("limit_level")

    if pos_dir not in ("LONG", "SHORT") or not entry:
        return

    key = "position_health"
    if not _cooldown_ok(key, 60):
        return

    is_long    = pos_dir == "LONG"
    time_utc   = datetime.utcnow().strftime("%H:%M UTC")
    pnl_pts    = round((current_price - entry) * (1 if is_long else -1), 1)
    pnl_aud    = pnl_pts * 10.0
    pnl_sign   = "+" if pnl_pts >= 0 else ""

    stop_dist  = round(abs(stop  - current_price), 1) if stop else None
    tp_dist    = round(abs(tp    - current_price), 1) if tp   else None
    ig_dir     = "BUY" if is_long else "SELL"
    ig_close   = "SELL" if is_long else "BUY"

    # ── Count TF agreement via cached last_technical_signals.json ─────────────
    n_oppose = 0
    n_agree  = 0
    try:
        if _TECH_STATE_FILE.exists():
            state  = json.loads(_TECH_STATE_FILE.read_text())
            bull_s = {"BUY", "STRONG BUY", "LEAN BUY"}
            bear_s = {"SELL", "STRONG SELL", "LEAN SELL"}
            for v in state.values():
                v = v.upper()
                if is_long:
                    if v in bull_s: n_agree  += 1
                    if v in bear_s: n_oppose += 1
                else:
                    if v in bear_s: n_agree  += 1
                    if v in bull_s: n_oppose += 1
    except Exception:
        pass

    # ── Also pull ATR from day_trading for a better stop suggestion ───────────
    atr = current_price * 0.005
    try:
        from day_trading import get_day_trading_signal as _gds
        _, _s = _gds(period="60d", interval="1h")
        if _s:
            atr = float(_s.get("atr", atr))
    except Exception:
        pass

    stop_critical  = stop_dist is not None and stop_dist < 15
    signals_flipped = n_oppose >= 4

    if not stop_critical and not signals_flipped:
        return   # situation not critical enough

    # ── Build alert ───────────────────────────────────────────────────────────
    if signals_flipped and stop_critical:
        headline  = f"⛔ CLOSE YOUR {pos_dir} NOW"
        reason    = (f"{n_oppose} timeframes oppose your {pos_dir} AND "
                     f"stop is only {stop_dist:.0f} pts away — exit before you're forced out")
    elif signals_flipped:
        headline  = f"⛔ CLOSE YOUR {pos_dir} — MARKET REVERSED"
        reason    = (f"{n_oppose} timeframes (of 4 watched) now point against your {pos_dir}. "
                     f"The trend has turned — cutting this trade protects your account.")
    else:
        headline  = f"⚠️ STOP ALERT — {pos_dir} POSITION"
        reason    = f"Stop is only {stop_dist:.0f} pts away. Decide: close manually or let stop trigger."

    pnl_str  = f"{pnl_sign}{pnl_pts} pts / {pnl_sign}A${abs(pnl_aud):.0f}"
    stop_str = f"Stop: {stop:,.0f} ({stop_dist:.0f} pts away)" if stop_dist else ""
    tp_str   = f"TP: {tp:,.0f} ({tp_dist:.0f} pts to go)"     if tp_dist   else ""

    ok = send_message(
        f"{headline}\n"
        f"P&amp;L: <b>{pnl_str}</b>  ·  Now: <b>{current_price:,.0f}</b>\n"
        f"IG Positions → tap trade → <b>Close</b>"
    )
    if ok:
        _mark_sent(key)


# ── Custom watch-level alerts (trade-specific intermediate price targets) ──────

def check_watch_levels(current_price: float, cfg: dict):
    """
    Fires a Telegram alert when price crosses a user-defined watch level.
    Levels are set in alert_config.json under "watch_levels" as a list of:
        {"level": float, "direction": "above"|"below",
         "label": str, "action": str, "cooldown_minutes": int}

    direction="above"  → alert fires when current_price >= level
    direction="below"  → alert fires when current_price <= level
    """
    if not cfg.get("enabled"):
        return

    watch_levels = cfg.get("watch_levels", [])
    if not watch_levels:
        return

    pos_dir  = cfg.get("position_direction")
    entry    = cfg.get("entry_price")
    time_utc = datetime.utcnow().strftime("%H:%M UTC")

    for i, wl in enumerate(watch_levels):
        level     = wl.get("level")
        direction = wl.get("direction", "above")   # "above" | "below"
        label     = wl.get("label", "Watch level reached")
        action    = wl.get("action", "Review your position in IG.")
        cooldown  = wl.get("cooldown_minutes", 60)

        if not level:
            continue

        key = f"watch_{i}_{int(level)}"
        if not _cooldown_ok(key, cooldown):
            continue

        triggered = (
            (direction == "above" and current_price >= level) or
            (direction == "below" and current_price <= level)
        )
        if not triggered:
            continue

        # Build context line
        pnl_str = ""
        if pos_dir and entry:
            is_long = pos_dir == "LONG"
            pnl_pts = round((current_price - entry) * (1 if is_long else -1), 1)
            pnl_aud = pnl_pts * 10.0
            sign    = "+" if pnl_pts >= 0 else ""
            pnl_str = f"\nPosition P&amp;L: <b>{sign}{pnl_pts} pts / {sign}A${abs(pnl_aud):.0f}</b>"

        dir_arrow = "▲" if direction == "above" else "▼"
        dir_word  = "risen above" if direction == "above" else "dropped below"

        ok = send_message(
            f"📍 <b>{label}</b>\n"
            f"Now: <b>{current_price:,.0f}</b>  {dir_arrow}  {level:,.0f}{pnl_str}\n"
            f"{action}"
        )
        if ok:
            _mark_sent(key)


# ── High-conviction alert — fires only when everything lines up ───────────────

def check_high_conviction_signal(current_price: float):
    """
    Fires ONLY when 3+ timeframes AND 2+ ML horizons agree on the same direction.
    This is the 'don't miss this' alert — highest confidence setup available.
    Cooldown: 3 hours to avoid repeat alerts on the same move.
    """
    cfg = load_config()
    if not cfg.get("enabled"):
        return

    # Session gate — active market hours only
    hour = datetime.utcnow().hour
    if not (7 <= hour <= 21):
        return

    # Load technical signals
    tech_file = CACHE_DIR / "last_technical_signals.json"
    try:
        sigs = json.loads(tech_file.read_text()) if tech_file.exists() else {}
    except Exception:
        return

    # Count timeframe agreement
    buy_tfs  = [k for k, v in sigs.items() if "BUY"  in v.upper()]
    sell_tfs = [k for k, v in sigs.items() if "SELL" in v.upper()]
    n_buy    = len(buy_tfs)
    n_sell   = len(sell_tfs)

    if n_buy < 3 and n_sell < 3:
        return   # not enough agreement

    direction = "BUY" if n_buy >= n_sell else "SELL"
    n_agree   = n_buy if direction == "BUY" else n_sell

    # Load ML forecasts — must agree
    mh_file = CACHE_DIR / "multi_horizon_predictions.json"
    try:
        mh = json.loads(mh_file.read_text()) if mh_file.exists() else {}
    except Exception:
        return

    ml_dirs  = [v.get("direction", -1) for v in mh.values()]
    ml_bulls = ml_dirs.count(1)
    ml_bears = ml_dirs.count(0)
    ml_agree = (direction == "BUY" and ml_bulls >= 2) or (direction == "SELL" and ml_bears >= 2)

    if not ml_agree:
        return   # ML disagrees — not high conviction

    # News check — must not strongly oppose
    try:
        news_score = json.loads((CACHE_DIR / "news_sentiment.json").read_text()).get("score", 0)
        if direction == "BUY"  and news_score < -0.4:
            return
        if direction == "SELL" and news_score >  0.4:
            return
    except Exception:
        pass

    key = "high_conviction"
    if not _cooldown_ok(key, 180):   # 3-hour cooldown
        return

    # Build the alert
    is_long   = direction == "BUY"
    ig_dir    = "BUY" if is_long else "SELL"
    emoji     = "🟢" if is_long else "🔴"
    time_utc  = datetime.utcnow().strftime("%H:%M UTC")

    # ATR-based levels
    try:
        from day_trading import get_day_trading_signal as _gds
        _, _s = _gds(period="60d", interval="1h")
        atr = float(_s.get("atr", current_price * 0.008)) if _s else current_price * 0.008
    except Exception:
        atr = current_price * 0.008

    stop_dist  = max(15, round(atr * 1.5))
    limit_dist = max(25, round(atr * 2.5))
    stop_lvl   = round(current_price - stop_dist  if is_long else current_price + stop_dist)
    limit_lvl  = round(current_price + limit_dist if is_long else current_price - limit_dist)
    rr         = round(limit_dist / stop_dist, 1)
    pnl_risk   = stop_dist  * 10
    pnl_target = limit_dist * 10

    # Position note
    pos_dir = cfg.get("position_direction")
    opp_note = f"\n⚠️ You have an open {pos_dir} — close it first." if pos_dir else ""

    ok = send_message(
        f"{emoji} <b>HIGH CONVICTION {ig_dir}</b>  {time_utc}\n"
        f"{n_agree}/4 timeframes + ML aligned{opp_note}\n\n"
        f"IG → Trade → Gold Spot → <b>{ig_dir}</b>\n"
        f"Stop: <b>{stop_lvl:,}</b>  Limit: <b>{limit_lvl:,}</b>\n"
        f"Risk A${pnl_risk}  →  Target A${pnl_target}  (R:R 1:{rr})"
    )
    if ok:
        _mark_sent(key)


# ── Main price-level monitor (called from scheduler) ─────────────────────────

def check_price_alerts(current_price: float):
    """
    Called every scheduler quick-refresh cycle with the latest gold price.
    Sends Telegram alerts based on stop/limit config.
    Also checks for high-quality entry signals.
    """
    cfg = load_config()
    if not cfg.get("enabled"):
        return

    stop  = cfg.get("stop_level")
    limit = cfg.get("limit_level")
    buf   = cfg.get("warning_buffer", 5)

    # ── Position direction — determines which side stop/limit sit on ──────────
    # LONG : limit ABOVE entry (hit when price rises), stop BELOW (hit when falls)
    # SHORT: limit BELOW entry (hit when price falls), stop ABOVE (hit when rises)
    _dir     = cfg.get("position_direction")   # "LONG" | "SHORT" | None
    _is_long = (_dir != "SHORT")               # default to long-side logic if unset

    # ── Stale config guard ────────────────────────────────────────────────────
    _max_dist = current_price * 0.10
    if stop is not None and abs(stop - current_price) > _max_dist:
        stop = None
    if limit is not None and abs(limit - current_price) > _max_dist:
        limit = None

    if cfg.get("alert_price_levels"):
        if limit is not None:
            if _is_long:
                # LONG: TP is above — fires when price reaches or exceeds limit
                if current_price >= limit:
                    alert_limit_hit(current_price, limit)
                elif current_price >= limit - buf:
                    alert_price_near_limit(current_price, limit, cfg)
            else:
                # SHORT: TP is below — fires when price drops to or below limit
                if current_price <= limit:
                    alert_limit_hit(current_price, limit)
                elif current_price <= limit + buf:
                    alert_price_near_limit(current_price, limit, cfg)

        if stop is not None:
            # ── Confirmation buffer: require price to be THROUGH the stop by
            # 1.5 pts before declaring it hit. This prevents false positives from
            # the mid-price feed momentarily grazing the stop level (IG only fills
            # when the bid/ask price crosses it, not just the mid). ──────────────
            _STOP_CONFIRM_PTS = 1.5

            # Once stop_hit has fired, don't also send near_stop — it's redundant
            # and creates confusing "alternating" alerts. ─────────────────────────
            _stop_hit_recent = not _cooldown_ok("stop_hit", 180)

            if _is_long:
                # LONG: stop below entry — triggered when price falls through stop
                if current_price <= stop - _STOP_CONFIRM_PTS:
                    alert_stop_hit(current_price, stop)
                elif current_price <= stop + buf and not _stop_hit_recent:
                    alert_price_near_stop(current_price, stop, cfg)
            else:
                # SHORT: stop above entry — triggered when price rises through stop
                if current_price >= stop + _STOP_CONFIRM_PTS:
                    alert_stop_hit(current_price, stop)
                elif current_price >= stop - buf and not _stop_hit_recent:
                    alert_price_near_stop(current_price, stop, cfg)

    # High-conviction alert — 3+ TFs + ML all agree (fires first, highest priority)
    check_high_conviction_signal(current_price)
    # Custom trade-specific watch levels
    check_watch_levels(current_price, cfg)
    # Technical signal alerts (main source of trade alerts)
    check_technical_signal_alerts(current_price)
    # Position health — fires when signals reverse hard against open trade
    check_position_health_alert(current_price)
    # ML model entry signals + position revision
    check_entry_signal(current_price)
    check_revision_signal(current_price)
