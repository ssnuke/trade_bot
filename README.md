# ⚡ Delta Pro Sniper Trading Bot

A high-frequency algorithmic trading bot designed for Delta Exchange (India), featuring a premium glassmorphic dashboard, real-time market scanning, and IST-based session tracking.

## 🚀 Overview

The **Delta Pro Sniper** is a consolidated trading solution that combines a sophisticated technical analysis engine with a real-time monitoring interface. It is optimized for aggressive growth while maintaining strict risk management protocols.

### Key Features
- **Sniper Strategy**: High-precision entry logic using BOS (Break of Structure), ChoCh (Change of Character), and RSI divergence.
- **Real-Time Dashboard**: A premium, integrated web UI for monitoring equity, P&L, and market scanning logs.
- **IST Session Tracking**: Automated 24-hour trading sessions (Midnight-to-Midnight IST) with persistent daily records.
- **Risk Management**: Adaptive leverage (default 25x), daily loss limits (20%), and concurrent position controls.
- **Single-Service Architecture**: Both the trading engine and the dashboard run in a single process for easy deployment.

## 🛠 Project Structure

```text
├── apps/
│   ├── bot/                 # Core Trading Logic (bot_pro.py)
│   └── dashboard_flask/     # Frontend Templates & Assets
├── packages/
│   └── core/                # Shared libraries (client, patterns, metrics)
├── paper_trades/            # Local logs and session records
├── Dockerfile.bot           # Unified Docker configuration
└── render.yaml              # Render deployment template
```

## 🌐 Deployment

### 1. Unified Service (Recommended)
The project is optimized for **Render** or similar Docker-based platforms.

- **URL**: `https://your-bot-name.onrender.com/` (Hosts both API and UI)
- **Health Check**: `/api/data` or `/health`

### 2. Environment Variables
Ensure the following are set in your environment:
- `DELTA_API_KEY`: Your Delta Exchange API Key
- `DELTA_API_SECRET`: Your Delta Exchange API Secret
- `PAPER_TRADING`: `True` (default) or `False`
- `DELTA_BASE_URL`: `https://api.india.delta.exchange`

## 📊 Monitoring
- **Dashboard**: Visit the root URL `/` to view real-time stats.
- **Session History**: Historical daily performance is automatically loaded in the "Session History" panel.
- **Raw Analysis**: Access `/api/data` for a JSON view of the current market state.

## 🛡 Disclaimer
Trading cryptocurrencies involves significant risk. This bot is provided for educational and utility purposes. Always use paper trading mode first to verify strategy performance.
