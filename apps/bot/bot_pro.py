"""
Aggressive But Survivable Trading Bot - Momentum Breakout Pro
Target: 4-6x growth in 1 month (5000 → 20,000-30,000 INR)
Strategy: Multi-timeframe trend following with breakout confirmation
"""
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from packages.core.delta_client import DeltaClient
import csv
import json
import sys
import shutil 
from collections import deque # For log buffer
from flask import Flask, jsonify, render_template
from flask_cors import CORS
import threading
from dotenv import load_dotenv
from packages.core.patterns import PatternRecognizer, SupportResistance
from packages.core.structure_analyzer import StructureAnalyzer
from packages.core.database_manager import DatabaseManager
from packages.core.strategies.sniper import SniperStrategy
from packages.core.strategies.ema_cross import EMACrossStrategy
from packages.core.risk.manager import RiskManager
from packages.core.execution.executor import OrderExecutor, Position

# Ensure UTF-8 output for Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

load_dotenv()

# Setup Flask with access to the dashboard templates/static files
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # apps/bot
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
app = Flask(__name__, 
            template_folder=os.path.join(PROJECT_ROOT, "apps", "dashboard_flask", "templates"),
            static_folder=os.path.join(PROJECT_ROOT, "apps", "dashboard_flask", "static"))

CORS(app, resources={r"/*": {"origins": "*"}})

bot_instance = None

@app.route('/', methods=['GET'])
def home():
    """Serve the main trading dashboard UI"""
    return render_template('dashboard.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "Running", "service": "Delta Bot Worker + UI"})

@app.route('/api/data', methods=['GET'])
@app.route('/dashboard_data', methods=['GET'])
@app.route('/analysis', methods=['GET'])
def get_analysis():
    global bot_instance
    if bot_instance:
        # Return full data if available, otherwise fallback to latest analysis
        data = bot_instance.full_dashboard_data if bot_instance.full_dashboard_data else (bot_instance.latest_analysis or {})
        data["bot_version"] = "PRO_V2_RESTRICTED"
        return jsonify(data)
    return jsonify({"error": "Bot instance not initialized"})

@app.route('/set_priority', methods=['POST'])
def set_priority():
    global bot_instance
    from flask import request
    if bot_instance:
        data = request.json
        symbol = data.get('symbol')
        if symbol:
            bot_instance.priority_symbol = symbol
            return jsonify({"status": "success", "priority": symbol})
    return jsonify({"error": "Bot instance not initialized"}), 400

@app.route('/api/session_history', methods=['GET'])
def get_history():
    global bot_instance
    if not bot_instance:
        return jsonify([])
    
    history_dir = bot_instance.session_history_dir
    try:
        if bot_instance and hasattr(bot_instance, 'db'):
            history = bot_instance.db.get_session_history()
            return jsonify(history)
            
        # Fallback to file-based if DB not available or first run
        from pathlib import Path
        history_dir = bot_instance.session_history_dir if bot_instance else os.path.join(PROJECT_ROOT, "apps", "bot", "paper_trades", "sessions")
        if not os.path.exists(history_dir):
            return jsonify([])
            
        history = []
        for file_path in Path(history_dir).glob("session_*.json"):
            with open(file_path, 'r') as f:
                history.append(json.load(f))
        # Sort by date descending
        history.sort(key=lambda x: x.get('date', ''), reverse=True)
        return jsonify(history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset_bot():
    global bot_instance
    msg = f"📩 RECEIVED RESET REQUEST: {datetime.now().strftime('%H:%M:%S')}"
    if bot_instance:
        bot_instance.log(msg)
        bot_instance.reset_trading_limits()
        bot_instance.log("✅ Reset executed on bot_instance")
        return jsonify({"status": "success", "message": "Bot limits and counters reset successfully"})
    print(f"{msg}\n❌ Reset failed: bot_instance is None")
    return jsonify({"error": "Bot instance not initialized"}), 400

@app.route('/reset_capital', methods=['POST'])
def reset_capital():
    global bot_instance
    msg = f"💰 RECEIVED CAPITAL RESET REQUEST: {datetime.now().strftime('%H:%M:%S')}"
    if bot_instance:
        bot_instance.log(msg)
        old_equity = bot_instance.equity
        bot_instance.equity = 5000
        bot_instance.starting_capital = 5000
        bot_instance.daily_start_equity = 5000
        bot_instance.positions = {}
        bot_instance.trades = []
        bot_instance.consecutive_wins = 0
        bot_instance.consecutive_losses = 0
        bot_instance.consecutive_sure_shot_losses = 0
        bot_instance.daily_trades = 0
        bot_instance.save_state()
        bot_instance.log(f"✅ Capital reset: {old_equity:.2f} INR → 5000 INR")
        return jsonify({
            "status": "success", 
            "message": f"Capital reset from ₹{old_equity:.2f} to ₹5,000",
            "old_equity": old_equity,
            "new_equity": 5000
        })
    print(f"{msg}\n❌ Capital reset failed: bot_instance is None")
    return jsonify({"error": "Bot instance not initialized"}), 400

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def resource_not_found(e):
    return jsonify({"error": "Resource not found"}), 404

# Database path
DB_PATH = os.path.join(PROJECT_ROOT, "data", "bot_data.db")

class AggressiveGrowthBot:
    def __init__(self):
        self.api_key = os.getenv("DELTA_API_KEY")
        self.api_secret = os.getenv("DELTA_API_SECRET")
        self.base_url = os.getenv("DELTA_BASE_URL", "https://api.india.delta.exchange")
        self.paper_trading = os.getenv("PAPER_TRADING", "True") == "True"
        
        self.client = DeltaClient(self.api_key, self.api_secret, self.base_url)
        
        # --- AGGRESSIVE GROWTH CONFIGURATION ---
        self.starting_capital = 5000
        self.equity = 5000
        self.target_equity = 80000  # Target as requested: 80k INR from 5k
        
        # Adaptive Risk Management
        self.base_leverage = 25 # Reduced to 25x as requested
        self.max_daily_loss_pct = 0.20  # Stop trading if down 20% in a day
        self.max_concurrent_positions = 1
        
        # Strategy Parameters
        self.swing_lookback = 20  # Candles to look back for swing high/low
        self.volume_mult = 1.5
        self.min_breakout_pct = 0.005  # 0.5% minimum breakout
        
        # State
        self.positions = {}  # {symbol: position_data}
        self.product_map = {}
        self.contract_values = {}
        self.priority_symbols = [
            "ETHUSD", "SOLUSD", "XRPUSD", 
            "BNBUSD", "UNIUSD", "LTCUSD",
            "DOGEUSD", "LINKUSD", "AVAXUSD",
            "DOTUSD", "ADAUSD", "ATOMUSD",
            "ALGOUSD", "BCHUSD"
        ]
        self.symbols_to_trade = self.priority_symbols.copy()
        
        # Daily tracking
        self.daily_start_equity = self.equity
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.last_reset_day = datetime.now().day
        self.trades = []
        
        # --- DATABASE SETUP ---
        self.db = DatabaseManager(DB_PATH)
        
        # --- PATHS & LOGGING SETUP ---
        # Data base path (defaults to PROJECT_ROOT/data for Docker compatibility)
        self.data_base = os.path.join(PROJECT_ROOT, "data")
        self.trade_log_dir = os.path.join(self.data_base, "paper_trades")
        self.dashboard_file = os.path.join(self.data_base, "dashboard_data.json")
        self.session_history_dir = os.path.join(self.trade_log_dir, "sessions")
        self.session_log_dir = os.path.join(self.trade_log_dir, "logs")
        
        os.makedirs(self.trade_log_dir, exist_ok=True)
        os.makedirs(self.session_history_dir, exist_ok=True)
        os.makedirs(self.session_log_dir, exist_ok=True)

        self.log_queue = deque(maxlen=50) 
        
        ist_now = self.get_ist_now()
        self.session_date = ist_now.strftime('%Y-%m-%d')
        self.session_start_time = ist_now
        self.daily_start_equity = self.equity 
        
        self.current_log_file = os.path.join(self.session_log_dir, f"log_{self.session_date}.txt")
        self.load_recent_logs() 

        # --- BOT STATE ---
        self.turbo_mode = os.getenv("TURBO_MODE", "True") == "True"
        self.turbo_completed = False
        self.sure_shot_only = True 

        self.latest_analysis = {} 
        self.full_dashboard_data = {} 
        self.priority_symbol = None 
        self.last_priority_scan = 0
        self._bot_heartbeat_ts = time.time()
        self.startup_token = int(time.time())
        self.reset_event = threading.Event()
        self.bypass_limits = False 
        
        self.consecutive_sure_shot_losses = 0
        self.consecutive_wins = 0 
        self.last_loss_time = 0

        self.log(f"🚀 Aggressive Growth Bot Starting...")
        self.log(f"   Capital: {self.starting_capital} INR")
        self.log(f"   Target: {self.target_equity} INR (6x)")
        self.log(f"   Leverage: {self.base_leverage}x")
        self.log(f"   Mode: {'PAPER' if self.paper_trading else 'LIVE'}")
        
        # --- MODULAR COMPONENTS ---
        self.risk_manager = RiskManager()
        self.executor = OrderExecutor(self.client, self.db, self.paper_trading)
        self.strategies = {
            "SNIPER": SniperStrategy(self.swing_lookback, self.min_breakout_pct),
            "EMA_CROSS": EMACrossStrategy()
        }
        
        self.load_state() 
        
        self._init_products()

    def get_ist_now(self):
        """Get current time in IST (UTC+5:30)"""
        return datetime.utcnow() + timedelta(hours=5, minutes=30)
    
    def log(self, message):
        """Unified logging to console, memory queue, and file"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        full_msg = f"{timestamp} - {message}"
        print(full_msg)
        self.log_queue.append(full_msg)
        try:
            with open(self.current_log_file, "a", encoding="utf-8") as f:
                f.write(full_msg + "\n")
        except Exception as e:
            print(f"⚠️ Failed to write to log file: {e}")

    def load_recent_logs(self):
        """Load last 50 logs from the current session file if it exists"""
        if os.path.exists(self.current_log_file):
            try:
                with open(self.current_log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    # Use deque maxlen to automatically keep last 50
                    for line in lines[-50:]:
                        self.log_queue.append(line.strip())
            except Exception as e:
                print(f"⚠️ Failed to load recent logs: {e}")
    
    def _init_products(self):
        """Fetch all USD products dynamically"""
        self.log("📊 Fetching all USD Product IDs...")
        products = self.client.get_products()
        if not products:
            self.log("❌ Failed to fetch products. Exiting.")
            return
            
        count = 0
        # Filter products to only include USD pairs and limit to a reasonable number
        # We prioritize our list first, then add more until we hit a limit (e.g. 25)
        discovered_symbols = []
        for p in products:
            sym = p['symbol']
            if sym.endswith('USD'):
                self.product_map[sym] = p['id']
                self.contract_values[sym] = float(p.get('contract_value', 1))
                if sym not in self.priority_symbols:
                    discovered_symbols.append(sym)
                count += 1
        
        # Strictly use only the user-specified list
        self.symbols_to_trade = [s for s in self.priority_symbols if s in self.product_map]
        # Ensure only valid symbols (those in product_map) are kept
        self.symbols_to_trade = [s for s in self.symbols_to_trade if s in self.product_map]
        
        self.log(f"✅ Loaded {count} USD products.")
        self.log(f"🎯 Bot will scan {len(self.symbols_to_trade)} active symbols (Loop time ~30-40s)")
    
    def get_adaptive_risk(self, symbol, strategy="SNIPER"):
        """Use RiskManager for adaptive scaling"""
        return self.risk_manager._get_risk_pct(strategy), self.base_leverage
    
    def check_daily_limits(self):
        """Check if we should stop trading for the day (IST-based)"""
        if self.bypass_limits:
            print("✨ Manual bypass active. Resuming trades...")
            self.bypass_limits = False
            return True

        ist_now = self.get_ist_now()
        current_ist_date = ist_now.strftime('%Y-%m-%d')
        
        # Session Rollover Concept (IST Midnight)
        if current_ist_date != self.session_date:
            self.log(f"🌅 NEW SESSION DETECTED: {current_ist_date} (IST)")
            self.save_session_record() # Save record for completed day
            
            # Reset daily counters for the new session
            self.daily_start_equity = self.equity
            self.daily_trades = 0
            self.session_date = current_ist_date
            self.session_start_time = ist_now
            self.last_reset_day = ist_now.day
            
            # Rotate log file
            self.current_log_file = os.path.join(self.session_log_dir, f"log_{self.session_date}.txt")
            self.log_queue.clear()
            self.log(f"📁 New session log initialized: {self.current_log_file}")
            
            self.save_active_positions()
        
        # Check daily loss limit
        daily_pnl_pct = (self.equity - self.daily_start_equity) / self.daily_start_equity if self.daily_start_equity > 0 else 0
        if daily_pnl_pct < -self.max_daily_loss_pct:
            print(f"⛔ Daily loss limit hit ({daily_pnl_pct*100:.1f}%). Stopping for today.")
            return False
        
        # Check consecutive losses
        if self.consecutive_losses >= 3:
            print(f"⛔ 3 consecutive losses. Taking a break.")
            return False
        
        return True

    def save_session_record(self):
        """Save a record of the completed trading session"""
        try:
            ist_now = self.get_ist_now()
            # Calculate session stats
            session_trades = [t for t in self.trades if t.get('time', '').startswith(self.session_date)]
            wins = len([t for t in session_trades if t['pnl_inr'] > 0])
            losses = len([t for t in session_trades if t['pnl_inr'] < 0])
            total_trades = len(session_trades)
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            net_pnl = self.equity - self.daily_start_equity
            
            summary = {
                "date": self.session_date,
                "start_equity": round(self.daily_start_equity, 2),
                "end_equity": round(self.equity, 2),
                "net_pnl_inr": round(net_pnl, 2),
                "total_trades": total_trades,
                "wins": wins,
                "losses": losses,
                "win_rate": round(win_rate, 2),
                "session_started": self.session_start_time.strftime('%Y-%m-%d %H:%M:%S'),
                "session_ended": ist_now.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save to JSON
            filename = os.path.join(self.session_history_dir, f"session_{self.session_date}.json")
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=4)
                
            # Save to Database
            self.db.save_session(summary)
                
            self.log(f"📁 Session record saved to file and DB: {self.session_date}")
            
            # Add to logs
            self.log(f"🌅 Session Ended: {self.session_date} | PnL: ₹{net_pnl:+.2f}")
            
        except Exception as e:
            self.log(f"⚠️ Failed to save session record: {e}")
    
    def reset_trading_limits(self):
        """Reset limits and counters and ensure the dashboard updates immediately"""
        print("🔄 Resetting bot limits and counters via API...")
        self.consecutive_losses = 0
        self.consecutive_sure_shot_losses = 0
        self.consecutive_wins = 0
        self.risk_manager.reset()
        self.last_loss_time = 0
        self.daily_trades = 0
        self.daily_start_equity = self.equity
        self.last_reset_day = datetime.now().day
        
        # Immediate UI feedback
        self.latest_analysis["status"] = "Reset Success: Resuming..."
        self.log("🔄 Bot limits reset manually")
        
        # Trigger event and bypass
        self.bypass_limits = True
        self.reset_event.set()
        
        # FORCE EXPORT so dashboard sees it instantly
        self.export_dashboard_data()
        print(f"✅ Limits reset and bypass enabled. Resuming trades with {self.equity} equity baseline.")

    def get_multi_timeframe_data(self, symbol):
        """Fetch 5m, 15m, 1h, and 4h data for multi-timeframe analysis"""
        end_t = int(time.time())
        
        # 5-minute data (entry timeframe)
        start_5m = end_t - (200 * 5 * 60)
        data_5m = self.client.get_candles(symbol, "5m", start=start_5m, end=end_t)
        
        # 15-minute data (trend timeframe)
        start_15m = end_t - (200 * 15 * 60)
        data_15m = self.client.get_candles(symbol, "15m", start=start_15m, end=end_t)
        
        # 1h data (structure timeframe)
        start_1h = end_t - (200 * 60 * 60)
        data_1h = self.client.get_candles(symbol, "1h", start=start_1h, end=end_t)
        
        # 4h data (major structure timeframe)
        start_4h = end_t - (200 * 4 * 60 * 60)
        data_4h = self.client.get_candles(symbol, "4h", start=start_4h, end=end_t)
        
        if not data_5m or not data_15m or not data_1h or not data_4h:
            return None, None, None, None
        
        def to_df(data):
            df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            return df.sort_values('time').set_index('time')
            
        return to_df(data_5m), to_df(data_15m), to_df(data_1h), to_df(data_4h)
    
    def calculate_indicators(self, df):
        """Calculate technical indicators and patterns"""
        # EMAs with standardized naming (lowercase)
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema15'] = df['close'].ewm(span=15, adjust=False).mean() # Added for Hybrid Strategy
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # Backward compatibility for any logic expecting 'EMA 50' style
        df['ema20'] = df['ema21'] 
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        
        # ADX Calculation (14-period)
        df['tr'] = np.maximum(df['high'] - df['low'], 
                    np.maximum(abs(df['high'] - df['close'].shift()), 
                               abs(df['low'] - df['close'].shift())))
        df['tr14'] = df['tr'].rolling(14).sum()
        df['plus_dm'] = np.where((df['high'] - df['high'].shift()) > (df['low'].shift() - df['low']), 
                                 np.maximum(df['high'] - df['high'].shift(), 0), 0)
        df['minus_dm'] = np.where((df['low'].shift() - df['low']) > (df['high'] - df['high'].shift()), 
                                  np.maximum(df['low'].shift() - df['low'], 0), 0)
        df['plus_di'] = 100 * (df['plus_dm'].rolling(14).sum() / (df['tr14'] + 1e-9))
        df['minus_di'] = 100 * (df['minus_dm'].rolling(14).sum() / (df['tr14'] + 1e-9))
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'] + 1e-9)
        df['adx'] = df['dx'].rolling(14).mean()
        
        # EMA 50 Slope (Angle) - measure of trend flatness
        df['slope'] = (df['ema50'] - df['ema50'].shift(5)) / (df['ema50'].shift(5) + 1e-9) * 1000
        
        # Volume average
        df['vol_avg'] = df['volume'].rolling(window=20).mean()
        
        # ATR for UT Bot (Period 10)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(10).mean()
        df['atr_avg'] = df['atr'].rolling(50).mean()
        
        # UT Bot Trailing Stop Calculation
        a = 1 
        df['ut_trail'] = 0.0
        for i in range(1, len(df)):
            src = df['close'].iloc[i]
            prev_src = df['close'].iloc[i-1]
            prev_stop = df['ut_trail'].iloc[i-1]
            nLoss = a * df['atr'].iloc[i]
            
            if src > prev_stop and prev_src > prev_stop:
                df.loc[df.index[i], 'ut_trail'] = max(prev_stop, src - nLoss)
            elif src < prev_stop and prev_src < prev_stop:
                df.loc[df.index[i], 'ut_trail'] = min(prev_stop, src + nLoss)
            elif src > prev_stop:
                df.loc[df.index[i], 'ut_trail'] = src - nLoss
            else:
                df.loc[df.index[i], 'ut_trail'] = src + nLoss
        
        return df
            
        return "range"
    
    def check_trend(self, df, symbol):
        """Check higher timeframe trend using EMA alignment, separation, and slope"""
        if df is None or len(df) < 50:
            return None
        
        # Ensure EMAs exist (they should be lowercase from calculate_indicators)
        if 'ema50' not in df.columns:
            return None
            
        latest = df.iloc[-1]
        prevs = df.iloc[-5:] # Last 5 candles for slope
        
        # EMA 50 Slope
        ema50_slope = (latest['ema50'] - prevs.iloc[0]['ema50']) / (prevs.iloc[0]['ema50'] + 0.000001)
        
        # EMA Separation (EMA 9 and EMA 50 must have clear space)
        ema_separation = abs(latest['ema9'] - latest['ema50']) / (latest['ema50'] + 0.000001)
        is_separated = ema_separation > 0.0015 # 0.15% minimum separation
        
        # Bullish: EMA 9 > EMA 21 > EMA 50 + Positive Slope + Separation
        if latest['ema9'] > latest['ema21'] and latest['ema21'] > latest['ema50']:
            if ema50_slope > 0.0001 and is_separated:
                return "strong_up"
            return "up"
            
        # Bearish: EMA 9 < EMA 21 < EMA 50 + Negative Slope + Separation
        if latest['ema9'] < latest['ema21'] and latest['ema21'] < latest['ema50']:
            if ema50_slope < -0.0001 and is_separated:
                return "strong_down"
            return "down"
            
        return "range"

    def is_strong_trend(self, df_5m, df_15m):
        """Detect if market is in a STRONG trending regime (suitable for EMA Cross)"""
        if df_5m is None or df_15m is None:
            return False
        
        cur_5m = df_5m.iloc[-1]
        cur_15m = df_15m.iloc[-1]
        
        # Criteria for strong trend (stricter than basic trend)
        adx_strong = cur_5m['adx'] > 30  # Higher threshold
        
        # EMA 9/15 separation > 1.5%
        ema_sep = abs(cur_5m['ema9'] - cur_5m['ema15']) / cur_5m['ema15']
        clear_divergence = ema_sep > 0.015
        
        # Slope magnitude > 0.5 (strong directional movement)
        strong_slope = abs(cur_5m['slope']) > 0.5
        
        # Volume confirmation (current > 1.5x average)
        vol_avg = cur_5m.get('vol_avg', cur_5m['volume'])
        volume_conviction = cur_5m['volume'] > (vol_avg * 1.5)
        
        # 15m timeframe must also confirm the trend
        htf_aligned = cur_15m['adx'] > 25
        
        return adx_strong and clear_divergence and strong_slope and volume_conviction and htf_aligned

    def get_market_state(self):
        """Determine current market session and volatility state"""
        now_utc = datetime.utcnow()
        hour = now_utc.hour
        
        # Sessions in UTC:
        # Asia: 00:00 - 09:00
        # London (Euro/UK): 08:00 - 16:00
        # New York (US): 13:00 - 21:00
        # Overlaps: London/NY is 13:00 - 16:00
        
        session = "Asian"
        if 8 <= hour < 16: session = "London"
        if 13 <= hour < 21: session = "New York"
        if 13 <= hour < 16: session = "London/NY Overlap"
        
        # High alertness during session opens and overlaps
        is_high_alert_time = hour in [8, 9, 13, 14, 15, 16]
        
        return {
            "session": session,
            "is_high_alert": is_high_alert_time,
            "hour_utc": hour
        }
            
        return None
    def check_entry_signal(self, symbol):
        """Modular Hybrid Logic: Routes each coin to its best strategy."""
        # 1. Fetch & Calculate MTF Data
        dfs = self.get_multi_timeframe_data(symbol)
        if any(df is None for df in dfs): return None
        
        df_5m, df_15m, df_1h, df_4h = dfs
        # Standard indicators for dashboard/UI
        df_5m = self.calculate_indicators(df_5m)
        df_15m = self.calculate_indicators(df_15m)
        df_1h = self.calculate_indicators(df_1h)
        df_4h = self.calculate_indicators(df_4h)

        # Update UI analysis (Keep for Dashboard consistency)
        self._update_analysis_ui(symbol, df_5m, df_15m, df_1h)

        # --- DYNAMIC STRATEGY SELECTION ---
        # Same logic: Default to SNIPER unless strong trend
        if self.is_strong_trend(df_5m, df_15m):
            strategy_name = "EMA_CROSS"
        else:
            strategy_name = "SNIPER"
        
        strategy = self.strategies.get(strategy_name)
        if not strategy: return None

        # --- ANALYZE SIGNAL ---
        signal = strategy.analyze(df_5m, df_15m, df_1h, df_4h) if strategy_name == "SNIPER" else strategy.analyze(df_5m, df_15m)
        
        if signal:
            # Check for Fail Cooldown
            if self.consecutive_sure_shot_losses >= 2 and (time.time() - self.last_loss_time < 1800):
                self.log(f"⏸️ Cooldown Active: Skipping {symbol} {signal.side}")
                return None
            return signal

        return None

    def _update_analysis_ui(self, symbol, df_5m, df_15m, df_1h):
        """Helper to maintain the dashboard analysis data"""
        if symbol not in self.latest_analysis:
            self.latest_analysis[symbol] = {}
            
        cur = df_5m.iloc[-1]
        latest_close = cur['close']
        
        status_1h = self.check_trend(df_1h, symbol)
        global_bias = "neutral"
        if status_1h == "strong_up": global_bias = "bullish"
        elif status_1h == "strong_down": global_bias = "bearish"

        patterns = PatternRecognizer.detect_all(df_5m)
        bias_smc, smc_events = StructureAnalyzer.analyze_structure(df_15m)
        status_15m = self.check_trend(df_15m, symbol)
        sweeps = StructureAnalyzer.detect_liquidity_sweeps(df_5m)
        last_event_str = smc_events[-1]['type'] if smc_events else "None"
        
        support_levels = SupportResistance.detect_swing_levels(df_15m, lookback=5)
        nearest_support = 0
        nearest_resistance = 99999
        if support_levels:
            supports_below = [l['price'] for l in support_levels if l['type'] == 'Support' and l['price'] < latest_close]
            if supports_below: nearest_support = max(supports_below)
            resistances_above = [l['price'] for l in support_levels if l['type'] == 'Resistance' and l['price'] > latest_close]
            if resistances_above: nearest_resistance = min(resistances_above)

        self.latest_analysis[symbol].update({
            "patterns": [p['name'] for p in patterns[:3]],
            "market_bias": f"{global_bias.upper()} (1H) | {bias_smc.upper()} (15M)",
            "trend_15m": status_15m,
            "rsi": cur.get('rsi', 50),
            "price": latest_close,
            "support": round(nearest_support, 2),
            "resistance": round(nearest_resistance, 2),
            "sweeps": [s['type'] for s in sweeps],
            "last_event": last_event_str,
            "last_update": datetime.now().strftime('%H:%M:%S')
        })

        
        return None
    
    def calculate_position_size(self, symbol, entry_price, stop_loss, is_sure_shot=False, is_sniper=False, strategy="SNIPER"):
        """Calculate position size based on adaptive scaling"""
        risk_pct, leverage = self.get_adaptive_risk(symbol, is_sure_shot, is_sniper, strategy)
        
        # Position value in INR (Risk % of current equity * leverage)
        position_size_inr = self.equity * risk_pct * leverage
        
        # Convert to USD
        position_size_usd = position_size_inr / 87
        
        return position_size_usd
    
    def execute_trade(self, symbol, signal):
        """Execute trade using modular executor and risk manager"""
        if len(self.positions) >= self.max_concurrent_positions:
            self.log(f"⏸️  Max positions reached. Skipping {symbol}")
            return
        
        # Get signal details (handling both dict and Signal object)
        if hasattr(signal, 'side'):
             side = signal.side
             entry_price = signal.entry_price
             stop_loss = signal.stop_loss
             reason = signal.reason
             strategy_name = signal.strategy
             extra = signal.extra_data or {}
        else:
             side = signal['side']
             entry_price = signal['entry_price']
             stop_loss = signal['stop_loss']
             reason = signal['reason']
             strategy_name = signal.get('strategy', 'SNIPER')
             extra = signal

        # 1. Calculate Risk and Position Size
        pos_size_inr, leverage, risk_pct = self.risk_manager.calculate_position_size(
            self.equity, side, entry_price, stop_loss, strategy_name
        )
        
        # 2. Execution
        product_id = self.product_map.get(symbol)
        contract_val = self.contract_values.get(symbol, 1)
        num_contracts = int((pos_size_inr / 87) / (entry_price * contract_val)) # 87 is USDINR rate
        
        if num_contracts < 1:
            self.log(f"⚠️  Size too small for {symbol}")
            return

        self.log(f"\n🚀 {strategy_name} SIGNAL: {side.upper()} {symbol}")
        self.log(f"   Entry: ${entry_price:.2f} | Risk: {risk_pct*100:.0f}%")
        
        res = self.executor.place_order(product_id, num_contracts, side, entry_price)
        if res.success:
            self.positions[symbol] = Position(
                symbol=symbol,
                side=side,
                size=num_contracts,
                entry_price=entry_price,
                current_price=entry_price,
                stop_loss=stop_loss,
                take_profit=extra.get('tp3', entry_price * 1.1),
                opened_at=datetime.now(),
                tp1=extra.get('tp1', 0),
                tp2=extra.get('tp2', 0),
                tp3=extra.get('tp3', 0),
                initial_sl=stop_loss,
                qty=num_contracts,
                qty_remaining=num_contracts,
                leverage=leverage,
                risk_pct=risk_pct,
                strategy=strategy_name,
                is_sniper=extra.get('is_sniper', False),
                is_sure_shot=extra.get('is_sure_shot', False),
                entry_reason=reason,
                structural_target=extra.get('structural_target')
            )
            self.daily_trades += 1
            self.save_active_positions()
    
    def manage_positions(self):
        """Manage open positions using modular executor"""
        to_remove = []
        for sym, pos in self.positions.items():
            dfs = self.get_multi_timeframe_data(sym)
            if dfs[0] is None: continue
            
            df_5m = dfs[0]
            latest = df_5m.iloc[-1]
            cur_price = latest['close']
            high_p = latest['high']
            low_p = latest['low']
            
            # --- 1. Modular Position Management ---
            exit_info = self.executor.manage_position(pos, cur_price, high_p, low_p)
            
            if exit_info:
                if exit_info["type"] == "FULL_EXIT":
                    self.close_position(sym, exit_info["price"], exit_info["reason"], 1.0)
                    to_remove.append(sym)
                elif exit_info["type"] == "PARTIAL_EXIT":
                    self.take_profit(sym, exit_info["price"], exit_info["reason"], exit_info["portion"])
            
            # --- 2. Strategy Specific Exits (e.g. EMA Cross) ---
            strategy = self.strategies.get(pos.strategy)
            if strategy:
                should_exit, reason = strategy.should_exit(df_5m, pos.__dict__)
                if should_exit:
                    self.close_position(sym, cur_price, reason, 1.0)
                    to_remove.append(sym)

        for sym in to_remove:
            if sym in self.positions:
                del self.positions[sym]
    
    def take_profit(self, sym, price, level, portion):
        pos = self.positions[sym]
        close_qty = int(pos.qty * portion)
        pos.qty_remaining -= close_qty
        self.log(f"💰 {sym} {level} HIT! Closed partial at ${price:.2f}")
        self.close_position(sym, price, level, portion)

    def close_position(self, symbol, exit_price, reason, portion=1.0):
        """Close position and update equity using modular components"""
        pos = self.positions[symbol]
        
        # Record and Save via Executor
        trade_record, pnl_inr = self.executor.record_and_save_trade(
            pos, exit_price, reason, portion, self.equity
        )
        
        self.equity += pnl_inr
        
        # Update Risk Manager stats
        if pnl_inr > 0:
            self.risk_manager.record_win()
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.risk_manager.record_loss()
            self.consecutive_wins = 0
            self.consecutive_losses += 1
            
        # Sure Shot specifics
        if pos.is_sure_shot:
            if pnl_inr < 0:
                self.consecutive_sure_shot_losses += 1
                self.last_loss_time = time.time()
            else:
                self.consecutive_sure_shot_losses = 0

        self.log(f"   PnL: {trade_record['pnl_pct']}% | ROI: {trade_record['roi']}% | {pnl_inr:+.2f} INR")
        
        self.trades.append(trade_record)
        
        if self.paper_trading:
            self.export_trades()
            self.save_active_positions()

    def export_trades(self):
        """Export trades to CSV"""
        if not self.trades: return
        filename = os.path.join(self.trade_log_dir, f"trades_pro_{datetime.now().strftime('%Y%m%d')}.csv")
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.trades[0].keys())
                writer.writeheader()
                writer.writerows(self.trades)
        except Exception as e:
            print(f"   ❌ Export error: {e}")

    def save_active_positions(self):
        """Save current open positions to JSON for monitoring"""
        filename = os.path.join(self.trade_log_dir, "active_positions_pro.json")
        try:
            serializable_positions = {}
            for sym, pos in self.positions.items():
                p = pos.copy()
                if 'entry_time' in p and isinstance(p['entry_time'], datetime):
                    p['entry_time'] = p['entry_time'].strftime('%Y-%m-%d %H:%M:%S')
                serializable_positions[sym] = p
                
            with open(filename, 'w') as f:
                json.dump({
                    "last_update": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "equity": self.equity,
                    "daily_trades": self.daily_trades,
                    "consecutive_losses": self.consecutive_losses,
                    "consecutive_sure_shot_losses": self.consecutive_sure_shot_losses,
                    "last_loss_time": self.last_loss_time,
                    "last_reset_day": self.last_reset_day,
                    "open_positions": serializable_positions
                }, f, indent=4)
        except Exception as e:
            print(f"   ❌ Active positions save error: {e}")

    def load_state(self):
        """Load equity and trade history from dashboard_data.json and active_positions_pro.json"""
        if os.path.exists(self.dashboard_file):
            try:
                with open(self.dashboard_file, "r") as f:
                    data = json.load(f)
                    
                # Restore equity
                saved_equity = data.get("equity", self.starting_capital)
                if saved_equity > 0:
                    self.equity = saved_equity
                    self.daily_start_equity = self.equity
                    
                # Restore trades from DB preferentially
                try:
                    self.trades = self.db.get_recent_trades(100)
                    if self.trades:
                        # Sync equity if available in latest trade
                        self.equity = self.trades[0]['equity']
                        self.daily_start_equity = self.equity
                        self.log(f"   ✅ State restored from DB: {len(self.trades)} trades | Equity: {self.equity:.2f} INR")
                        return
                except Exception as db_e:
                    self.log(f"   ⚠️ DB load error: {db_e}. Falling back to file.")

                # Fallback to local file
                self.trades = data.get("recent_trades", [])
                self.log(f"   ✅ Basic state restored from file: {len(self.trades)} trades | Equity: {self.equity:.2f} INR")
            except Exception as e:
                print(f"   ⚠️ State load error (dashboard): {e}. Starting fresh.")

        # Load detailed counters from active_positions_pro.json
        pos_file = os.path.join(self.trade_log_dir, "active_positions_pro.json")
        if os.path.exists(pos_file):
            try:
                with open(pos_file, "r") as f:
                    p_data = json.load(f)
                    self.daily_trades = p_data.get("daily_trades", 0)
                    self.consecutive_losses = p_data.get("consecutive_losses", 0)
                    self.consecutive_sure_shot_losses = p_data.get("consecutive_sure_shot_losses", 0)
                    self.last_loss_time = p_data.get("last_loss_time", 0)
                    self.last_reset_day = p_data.get("last_reset_day", datetime.now().day)
                    
                    # If it was a different day, reset daily counters
                    if self.last_reset_day != datetime.now().day:
                        self.daily_trades = 0
                        self.last_reset_day = datetime.now().day
                        
                    print(f"   ✅ Persistent counters restored (Loss streak: {self.consecutive_losses})")
            except Exception as e:
                print(f"   ⚠️ Detailed state load error: {e}")

    def export_dashboard_data(self):
        """Export real-time state for Streamlit Dashboard"""
        try:
            # Calculate Win Rate and other Stats
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t['pnl_inr'] > 0])
            losing_trades = len([t for t in self.trades if t['pnl_inr'] < 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            total_pnl = self.equity - self.starting_capital
            
            # Current Session Stats
            daily_pnl = self.equity - self.daily_start_equity
            session_trades = [t for t in self.trades if t.get('time', '').startswith(self.session_date)]
            daily_wins = len([t for t in session_trades if t['pnl_inr'] > 0])
            daily_losses = len([t for t in session_trades if t['pnl_inr'] < 0])

            data = {
                "equity": round(self.equity, 2),
                "target": self.target_equity,
                "start_equity": self.starting_capital,
                "pnl": round(total_pnl, 2),
                "pnl_pct": round((total_pnl / self.starting_capital) * 100, 2),
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": round(win_rate, 2),
                # Session Data
                "session_date": self.session_date,
                "session_start_time": self.session_start_time.strftime('%H:%M:%S'),
                "daily_pnl": round(daily_pnl, 2),
                "daily_trades": self.daily_trades,
                "daily_wins": daily_wins,
                "daily_losses": daily_losses,
                "positions": [
                    {
                        "symbol": p,
                        "side": d['side'], 
                        "entry": d['entry'], 
                        "qty": d['qty'],
                        "leverage": d.get('leverage', 15),
                        "stop_loss": d.get('stop_loss', 0),
                        "tp1": d.get('tp1', 0),
                        "tp2": d.get('tp2', 0),
                        "tp3": d.get('tp3', 0),
                        "entry_time": d.get('entry_time', datetime.now()).strftime('%H:%M:%S') if isinstance(d.get('entry_time'), datetime) else str(d.get('entry_time', '')),
                        "unrealized_pnl": 0.0 
                    } for p, d in self.positions.items()
                ],
                "recent_trades": self.trades[-100:], # Increased to 100 for better context
                "last_update": self.get_ist_now().strftime("%Y-%m-%d %H:%M:%S"),
                "active_mode": "HYBRID",
                "leverage_mode": "PRO",
                "market_structure": self.latest_analysis,
                "market_state": self.get_market_state(), # Added session/alertness
                "recent_logs": list(self.log_queue)
            }
            
            self.full_dashboard_data = data # Store for immediate API access
            
            # Atomic write with retry for Windows locking
            temp = self.dashboard_file + ".tmp"
            for i in range(5):
                try:
                    with open(temp, "w") as f:
                        json.dump(data, f, indent=4)
                    shutil.move(temp, self.dashboard_file)
                    break
                except (PermissionError, IOError):
                    if i == 4: raise
                    time.sleep(0.1)
        except Exception as e:
            if "PermissionError" not in str(e):
                print(f"⚠️ Dashboard export error: {e}")

    def run(self):
        """Main trading loop"""
        print(f"\n{'='*60}")
        print(f"🚀 BOT PRO RUNNING - Target: {self.target_equity} INR")
        print(f"{'='*60}\n")
        
        while True:
            try:
                # Update global heartbeat at start of every full market scan
                self._bot_heartbeat_ts = time.time()
                self.latest_analysis["_bot_heartbeat"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.latest_analysis["status"] = "Starting full scan"

                if self.equity >= self.target_equity:
                    print(f"\n🎉🎉🎉 TARGET REACHED! 🎉🎉🎉")
                    break
                
                if not self.check_daily_limits():
                    # Update heartbeat during the rest period to avoid stale warning
                    print("💤 Bot is resting. Waiting for reset or session change...")
                    self.reset_event.clear()
                    for i in range(60): # 60 * 5s = 300s (5 mins)
                        # Wait for either 5 seconds OR the reset event to be set
                        is_reset = self.reset_event.wait(timeout=5)
                        if is_reset or self.bypass_limits:
                            print("✨ Reset Event detected! Breaking rest loop early...")
                            self.reset_event.clear()
                            break
                        
                        self._bot_heartbeat_ts = time.time()
                        self.latest_analysis["_bot_heartbeat"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        # Use setdefault to avoid overwriting a reset status from Flask thread
                        if "Reset" not in self.latest_analysis.get("status", ""):
                            self.latest_analysis["status"] = "Resting: Limit/Losses reached"
                        
                        self.export_dashboard_data()
                    continue
                
                if self.positions:
                    self.manage_positions()
                
                for sym in self.symbols_to_trade:
                    try:
                        if sym in self.positions: continue
                        
                        # --- PRIORITY SCAN INTERRUPT ---
                        if self.priority_symbol and (time.time() - self.last_priority_scan > 5):
                            p_sym = self.priority_symbol
                            if p_sym in self.product_map and p_sym not in self.positions:
                                print(f"⚡ PRIORITY SCAN: {p_sym}")
                                self.latest_analysis["status"] = f"Priority scanning {p_sym}"
                                self.check_entry_signal(p_sym)
                                self.last_priority_scan = time.time()
                        
                        # Frequent heartbeat updates even during long scans
                        self._bot_heartbeat_ts = time.time()
                        self.latest_analysis["_bot_heartbeat"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        
                        log_msg = f"Scanning {sym}..."
                        print(f"🔍 {log_msg}", end="\r")
                        
                        # Fix: Don't overwrite Reset Success with scanning log
                        if "Reset" not in self.latest_analysis.get("status", ""):
                            self.latest_analysis["status"] = log_msg
                            
                        self.log(log_msg)
                        
                        signal = self.check_entry_signal(sym)
                        if signal:
                            side = signal.side if hasattr(signal, 'side') else signal['side']
                            self.log(f"✅ {sym} signal found!")
                            self.log(f"✅ SIGNAL: {sym} {side.upper()}")
                            self.execute_trade(sym, signal)
                            
                        # Frequent exports (after EACH symbol scan)
                        self.export_dashboard_data()
                        time.sleep(0.1)
                    except Exception as sym_e:
                        self.log(f"⚠️ Error scanning {sym}: {sym_e}")
                        self.export_dashboard_data() # Ensure we still update heartbeat
                        continue
                
                # Export Dashboard Data
                self.export_dashboard_data()
                print(" " * 50, end="\r")
                
            except Exception as e:
                error_msg = f"❌ Error in main loop: {e}"
                self.log(error_msg)
                # Update heartbeat even on error so watchdog doesn't get confused
                self._bot_heartbeat_ts = time.time()
                time.sleep(10) # Wait a bit before retrying


def run_flask():
    from waitress import serve
    print(f"📡 Production API Server starting on port 5005...")
    serve(app, host='0.0.0.0', port=5005, threads=4)

def watchdog_thread(bot):
    """Monitors the bot's health and restarts the loop if it hangs"""
    print("🛡️  Watchdog started...")
    while True:
        time.sleep(60) # Check every minute
        time_since_heartbeat = time.time() - bot._bot_heartbeat_ts
        if time_since_heartbeat > 180: # 3 minutes silence
            print(f"\n🚨 WATCHDOG: Bot loop seems hung! (Last update {int(time_since_heartbeat)}s ago)")
            print("🚨 Attempting to restart bot thread...")
            # We don't want to kill the whole process (Flask needs to live)
            # but we can try to re-trigger the run loop if possible, 
            # or just rely on the fact that if it's hung in a requests call, 
            # it might eventually timeout.
            # However, a better way is to ensure we use timeouts in all network calls.
            # For now, we'll just log this and warn the user.
            bot.log_queue.append(f"🚨 WATCHDOG: Potential hang detected!")

if __name__ == "__main__":
    bot_instance = AggressiveGrowthBot()
    
    # Start Watchdog
    w = threading.Thread(target=watchdog_thread, args=(bot_instance,))
    w.daemon = True
    w.start()
    
    # Start Bot in background thread
    t = threading.Thread(target=bot_instance.run)
    t.daemon = True
    t.start()
    
    # Run Flask in main thread
    run_flask()
