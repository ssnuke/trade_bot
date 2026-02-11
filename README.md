# 🚀 Delta Trading Bot - Setup Guide

## Quick Start (3 Steps)

### 1. Configure Your API Keys

Edit `.env` file:
```env
DELTA_API_KEY=your_api_key_here
DELTA_API_SECRET=your_api_secret_here
DELTA_BASE_URL=https://api.india.delta.exchange

# IMPORTANT: Start with paper trading!
PAPER_TRADING=True
```

### 2. Run Paper Trading (Test Mode)

```bash
python -m apps.bot.bot_pro
```

The bot will:

- ✅ Trade SOLUSD, UNIUSD, BTCUSD
- ✅ Simulate trades (no real money)
- ✅ Show you how it works
- ✅ Track paper profits

**Run for 1-2 weeks to verify it works!**

### 3. Go Live (After Paper Testing)

Once you're confident:

1. Change `.env`:
   ```env
   PAPER_TRADING=False
   ```

2. Run bot:
   ```bash
   python -m apps.bot.bot_pro
   ```

3. **Monitor closely for first few days!**

---

## 🗂️ Project Layout

- `apps/bot/` - Trading bot + Flask API (`/analysis`, `/set_priority`)
- `apps/dashboard_flask/` - Web dashboard (port 5006)
- `apps/dashboard_streamlit/` - Streamlit dashboard
- `apps/chrome_extension/` - Browser overlay for delta.exchange
- `packages/core/` - Shared trading logic and API client
- `tools/` - Backtests and debug utilities
- `data/` - Runtime state, dashboard JSON, paper trades
- `scripts/windows/` - Windows run scripts

---

## 📊 Bot Configuration

### Symbols Traded (Top 3)

- **SOLUSD** - 115% monthly return (best performer)
- **UNIUSD** - 113% monthly return (second best)
- **BTCUSD** - 74% monthly return (most stable)

### Strategy

- **Entry**: Trend-following + RSI momentum
- **Exit**: Tiered profit-taking (TP1/TP2/TP3)
- **Timeframes**: 15m trend + 5m execution

### Risk Management

- **Starting Risk**: 10% per trade
- **Leverage**: 15x
- **Max Positions**: 3 (one per symbol)
- **Daily Loss Limit**: 20%
- **Max Consecutive Losses**: 3 (then pause)

### Adaptive Risk (Auto-adjusts as you grow)

| Your Equity | Risk/Trade | Leverage |
|-------------|------------|----------|
| < 10,000 | 10% | 15x |
| 10,000 - 20,000 | 7% | 12x |
| 20,000 - 70,000 | 5% | 10x |
| > 70,000 | 3% | 8x |

---

## 📈 Expected Performance

### Month 1 (5000 → ?)

**Conservative (70% avg):**

- 5,000 → 8,500 INR

**Moderate (90% avg):**

- 5,000 → 9,500 INR

**Aggressive (110% avg):**

- 5,000 → 10,500 INR

### Path to 80,000 INR

**If you achieve 90% monthly:**

| Month | Starting | Ending |
|-------|----------|--------|
| 1 | 5,000 | 9,500 |
| 2 | 9,500 | 18,050 |
| 3 | 18,050 | 34,295 |
| 4 | 34,295 | 65,161 |
| 5 | 65,161 | **123,806** ✅ |

**Reach 80k in 4-5 months**

---

## 🎯 How The Bot Works

### 1. Trend Detection (15m)
- Checks if price is in uptrend or downtrend
- Uses EMA20, EMA50, EMA200

### 2. Entry Signal (5m)
- **LONG**: RSI crosses above 50 in uptrend
- **SHORT**: RSI crosses below 50 in downtrend

### 3. Position Sizing
- Calculates risk based on stop-loss distance
- Uses 10% of equity with 15x leverage
- Adjusts size to risk exactly 10% if stopped

### 4. Tiered Exits
- **TP1 (50%)**: 1.5x risk → Move SL to breakeven
- **TP2 (30%)**: 2.5x risk → Activate trailing stop
- **TP3 (20%)**: 4x risk → Let winners run

### 5. Risk Controls
- Stop loss at recent swing low/high
- Daily loss limit (20%)
- Consecutive loss limit (3)
- Max positions (3)

---

## 🔧 Bot Commands

### Start Bot
```bash
python -m apps.bot.bot_pro
```

### Stop Bot
Press `Ctrl+C`

### Check Status
Bot prints status every 10 cycles:
```
📊 Status: 5234.50 INR | Positions: 2 | Progress: 26.2%
```

---

## 📝 What You'll See

### When Bot Starts
```
======================================================================
🚀 DELTA TRADING BOT - MAXIMUM PROFIT MODE
======================================================================
Capital: 5000 INR
Symbols: SOLUSD, UNIUSD, BTCUSD
Leverage: 15x
Mode: PAPER TRADING
======================================================================

📊 Initializing products...
   [PAPER] SOLUSD configured (ID: 14969)
   [PAPER] UNIUSD configured (ID: 15041)
   [PAPER] BTCUSD configured (ID: 14830)
✅ Loaded 3 products

🤖 Bot is now running...
```

### When Signal Found
```
🎯 SIGNAL: BUY SOLUSD
   Entry: $145.2340
   Stop: $142.8910
   Size: 12 contracts
   Risk: 10% | Leverage: 15x
   Trend: UP
   [PAPER] Position opened
```

### When Profit Taken
```
💰 SOLUSD TP1 HIT! Closing 50% at $148.7650
   PnL: 2.43% | ROE: 36.45% | +182.25 INR
   New Equity: 5182.25 INR
```

### When Stopped Out

```
🛑 UNIUSD STOPPED OUT at $12.3450
   PnL: -1.85% | ROE: -27.75% | -138.75 INR
   New Equity: 5043.50 INR
```

---

## ⚠️ Important Warnings

### Before Going Live

1. **Paper trade for 1-2 weeks minimum**
2. **Verify bot is profitable in paper mode**
3. **Understand the risks** (can lose money)
4. **Start with money you can afford to lose**
5. **Monitor the bot daily**

### Risks

- ⚠️ **High Leverage (15x)**: 1% move = 15% gain/loss
- ⚠️ **Volatile Markets**: Crypto can move fast
- ⚠️ **API Issues**: Network problems can cause missed trades
- ⚠️ **Slippage**: Real prices may differ from signals
- ⚠️ **Drawdowns**: You can lose 20% in a bad day

### Not Financial Advice

This bot is for educational purposes. Past backtest performance does not guarantee future results. You are responsible for your own trading decisions.

---

## 🐛 Troubleshooting

### Bot Won't Start

## Error: "Failed to fetch products"

- Double-check API key and secret
- Make sure keys are for India exchange
- Regenerate keys if needed

### No Trades Happening

**Bot running but no signals:**

- This is normal! Bot waits for high-quality setups
- Can take hours or days between trades
- Don't force trades - quality over quantity

**"Daily loss limit hit":**

- Bot stops trading after -20% day
- Will resume next day automatically
- This is a safety feature

### Positions Not Closing

**TP levels not hit:**

- Market needs to move in your favor
- Can take hours or days
- Stop loss will protect you if wrong

**Stop loss triggered:**

- This is normal and expected
- Strategy has 56-64% win rate
- Losses are part of trading

---

## 📊 Monitoring Your Bot

### Daily Checklist

- [ ] Check bot is still running
- [ ] Review open positions
- [ ] Check daily PnL
- [ ] Verify no errors in console
- [ ] Monitor equity growth

### Weekly Review

- [ ] Calculate win rate
- [ ] Check profit factor
- [ ] Review largest wins/losses
- [ ] Adjust if needed
- [ ] Withdraw profits if desired

### Monthly Goals

- **Month 1**: 70-110% return (5k → 8.5-10.5k)
- **Month 2**: 60-90% return
- **Month 3**: 50-70% return
- **Month 4**: 40-60% return

---

## 🎓 Tips for Success

### Do's ✅

- ✅ Start with paper trading
- ✅ Let the bot run 24/7
- ✅ Trust the strategy
- ✅ Take profits regularly
- ✅ Keep a trading journal
- ✅ Monitor but don't interfere

### Don'ts ❌

- ❌ Don't manually close winning trades early
- ❌ Don't move stop losses
- ❌ Don't add to losing positions
- ❌ Don't trade if bot says stop
- ❌ Don't use money you need
- ❌ Don't panic on drawdowns

---

## 📞 Support

### Files

- `apps/bot/bot_pro.py` - Main bot (Flask API on port 5005)
- `packages/core/delta_client.py` - Delta Exchange API client
- `packages/core/patterns.py` - Candlestick and support/resistance helpers
- `packages/core/structure_analyzer.py` - Market structure helpers
- `apps/dashboard_flask/` - Flask dashboard (port 5006)
- `apps/dashboard_streamlit/dashboard.py` - Streamlit dashboard
- `tools/backtests/` - Backtesting scripts
- `data/` - Runtime data (dashboard JSON, trades)
- `.env` - Configuration

### Backtest Results

- `backtest_results_v2/combined_report.csv` - Major coins
- `backtest_results_v2/altcoin_report.csv` - Altcoins

### Documentation

- `README.md` - This file
- `walkthrough.md` - Strategy analysis

---

## 🚀 Docker Deployment

- Bot Dockerfile: `Dockerfile.bot`
- Dashboard Dockerfile: `Dockerfile.dashboard`
- Oracle Cloud guide: `docs/DEPLOY_ORACLE.md`
- Render guide: `docs/DEPLOY_RENDER.md`

---

## 🚀 Ready to Start?

1. ✅ Set up `.env` with your API keys
2. ✅ Set `PAPER_TRADING=True`
3. ✅ Run `python -m apps.bot.bot_pro`
4. ✅ Watch it trade for 1-2 weeks
5. ✅ If profitable, switch to live
6. ✅ Monitor and enjoy!

## Good luck! May your trades be profitable! 📈
