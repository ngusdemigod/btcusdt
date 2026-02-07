# BTCUSDT Futures Bot (4H Regime Switch)

Trades BTC/USDT futures in a simple, rules-based way.

## Setup
1) Create venv:
   python -m venv .venv
   .venv\Scripts\activate  (Windows)
   source .venv/bin/activate (Mac/Linux)

2) Install:
   pip install -r requirements.txt

3) Run paper trading first:
   - In bot_config.yaml set mode: paper
   python main.py

## Live mode (ONLY if you understand futures risk)
1) Create .env in the same folder:
   BINANCE_API_KEY=your_key
   BINANCE_API_SECRET=your_secret

2) In bot_config.yaml set:
   mode: live

3) Run:
   python main.py

## What it does
- Uses 4H candles.
- Detects TREND vs RANGE.
- Trades trend with EMA + ATR trailing stop.
- Trades range with Bollinger + RSI mean reversion.
- Sizes positions by risking a fixed % of equity per trade.
- Saves state.json so it survives restarts.