# Gold Price Predictor

Streamlit app that predicts XAU/USD direction (1h / 1d / 1w / 1m horizons) using an ensemble of XGBoost, Random Forest and MLP trained on ~600 market and macro variables. Includes day-trading signals, candlestick patterns, and optional Telegram alerts.

## Architecture

Two processes that share the `data_cache/` directory on disk:

- **`app.py`** — Streamlit frontend. Reads pre-computed predictions from `data_cache/`.
- **`scheduler.py`** — Background worker. Refreshes prices every 15 min and retrains every 24 h.

For free hosting (Streamlit Community Cloud), you can skip the scheduler and trigger refreshes manually through the UI, or run only the scheduler locally.

## Secrets required

| Variable | What for | Required |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Telegram alerts | No (alerts disabled without it) |
| `TELEGRAM_CHAT_ID` | Telegram alerts | No |
| `TWELVE_DATA_KEY` | Intraday gold price fallback | No (yfinance is primary) |

---

## Option A: Streamlit Community Cloud (FREE)

Runs the web UI only. The scheduler is not started, but the app has buttons to refresh data manually, and the app still works with cached predictions.

1. Push this project to a **public GitHub repo** (private is paid on Streamlit Cloud).
   ```
   git init
   git add .
   git commit -m "Initial deploy"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/gold-predictor.git
   git push -u origin main
   ```

2. Go to https://share.streamlit.io → **New app**.
3. Connect your GitHub, pick the repo, set:
   - **Main file path:** `app.py`
   - **Python version:** 3.11
4. Click **Advanced settings → Secrets** and paste:
   ```
   TELEGRAM_BOT_TOKEN = "your_token_here"
   TELEGRAM_CHAT_ID   = "your_chat_id"
   TWELVE_DATA_KEY    = "your_key_here"
   ```
5. Click **Deploy**. First build takes 3–5 minutes (xgboost + scikit-learn are large).

**Note:** Streamlit Cloud sleeps the app after ~7 days of no traffic. A visit wakes it back up in about 30 seconds.

---

## Option B: Railway (~$5 USD/month, both processes)

Runs both the web app and the 24/7 scheduler. This is the closest match to how it ran on Replit.

1. Push to GitHub (public or private both work).
2. Sign up at https://railway.app — new accounts get $5 free credit to try it.
3. **New Project → Deploy from GitHub repo** → pick your repo.
4. Railway auto-detects the Procfile and creates **two services**: `web` and `worker`.
5. In **each service → Variables**, add:
   ```
   TELEGRAM_BOT_TOKEN = your_token
   TELEGRAM_CHAT_ID   = your_chat_id
   TWELVE_DATA_KEY    = your_key
   ```
6. For the `web` service: **Settings → Networking → Generate Domain** to get a public URL.
7. Both processes share a volume — Railway attaches one by default for Python apps. If you want the scheduler's writes to survive restarts, go to the project → **+ New → Volume** and mount it at `/app/data_cache` on both services.

---

## Running locally

```bash
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Terminal 1 — web app
streamlit run app.py

# Terminal 2 — scheduler (optional, for live refresh)
python scheduler.py
```

Set secrets by exporting env vars first, or create `.streamlit/secrets.toml` locally (it's gitignored):
```
TELEGRAM_BOT_TOKEN = "..."
TELEGRAM_CHAT_ID   = "..."
TWELVE_DATA_KEY    = "..."
```

---

## Trading disclaimer

This model is for research and educational purposes. Predictions have non-trivial error rates and past accuracy does not guarantee future performance. Do not trade real capital based solely on these signals. Paper-trade first.
