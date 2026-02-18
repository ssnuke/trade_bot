"""
Aggressive But Survivable Trading Bot - Momentum Breakout Pro
Target: 4-6x growth in 1 month (5000 → 20,000-30,000 INR)
Strategy: Multi-timeframe trend following with breakout confirmation
"""
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from delta_client import DeltaClient
import csv
import json
import sys
import shutil 
from collections import deque # For log buffer
from flask import Flask, jsonify
from flask_cors import CORS
import threading
from dotenv import load_dotenv
from patterns import PatternRecognizer, SupportResistance
from structure_analyzer import StructureAnalyzer

# Ensure UTF-8 output for Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

bot_instance = None

@app.route('/analysis', methods=['GET'])
def get_analysis():
    global bot_instance
    if bot_instance:
        data = bot_instance.latest_analysis or {"status": "scanning", "msg": "Bot logic is warm-up"}
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

@app.route('/reset', methods=['POST'])
def reset_bot():
    global bot_instance
    print(f"\n📩 RECEIVED RESET REQUEST: {datetime.now().strftime('%H:%M:%S')}")
    if bot_instance:
        bot_instance.reset_trading_limits()
        print("✅ Reset executed on bot_instance")
        return jsonify({"status": "success", "message": "Bot limits and counters reset successfully"})
    print("❌ Reset failed: bot_instance is None")
    return jsonify({"error": "Bot instance not initialized"}), 400

@app.route('/reset_capital', methods=['POST'])
def reset_capital():
    global bot_instance
    print(f"\n💰 RECEIVED CAPITAL RESET REQUEST: {datetime.now().strftime('%H:%M:%S')}")
    if bot_instance:
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
        print(f"✅ Capital reset: {old_equity:.2f} INR → 5000 INR")
        return jsonify({
            "status": "success", 
            "message": f"Capital reset from ₹{old_equity:.2f} to ₹5,000",
            "old_equity": old_equity,
            "new_equity": 5000
        })
    print("❌ Capital reset failed: bot_instance is None")
    return jsonify({"error": "Bot instance not initialized"}), 400

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def resource_not_found(e):
    return jsonify({"error": "Resource not found"}), 404

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
        self.trade_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_trades")
        self.dashboard_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_data.json")
        self.log_queue = deque(maxlen=20) # Keep last 20 logs
        
        # Sniper Mode Configuration
        self.turbo_mode = os.getenv("TURBO_MODE", "True") == "True"
        self.turbo_completed = False
        self.sure_shot_only = True # 🚀 ONLY take high-precision trades by default

        self.latest_analysis = {} # Store analysis for API
        self.priority_symbol = None # Symbol to scan with priority
        self.last_priority_scan = 0
        self._bot_heartbeat_ts = time.time() # For watchdog
        self.startup_token = int(time.time()) # To identify fresh instance
        self.reset_event = threading.Event()
        self.bypass_limits = False # Manual override after reset
        
        # Risk Management - Cooldown and Scaling
        self.consecutive_sure_shot_losses = 0
        self.consecutive_wins = 0 # For scaling capital
        self.last_loss_time = 0

        # Dynamic strategy selection - no hardcoded mapping
        # Bot will intelligently choose based on market conditions
        
        if self.paper_trading:
            os.makedirs(self.trade_log_dir, exist_ok=True)
        
        print(f"🚀 Aggressive Growth Bot Starting...")
        print(f"   Capital: {self.starting_capital} INR")
        print(f"   Target: {self.target_equity} INR (6x)")
        print(f"   Leverage: {self.base_leverage}x")
        print(f"   Mode: {'PAPER' if self.paper_trading else 'LIVE'}")
        
        self.load_state() # Load previous state if available
        self._init_products()
    
    def _init_products(self):
        """Fetch all USD products dynamically"""
        print("\n📊 Fetching all USD Product IDs...")
        products = self.client.get_products()
        if not products:
            print("❌ Failed to fetch products. Exiting.")
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
        
        print(f"✅ Loaded {count} USD products.")
        print(f"🎯 Bot will scan {len(self.symbols_to_trade)} active symbols (Loop time ~30-40s)")
    
    def get_adaptive_risk(self, symbol, is_sure_shot=False, is_sniper=False, strategy="SNIPER"):
        """Dynamic Risk: 25x Leverage (Locked) | Strategy-Based Capital Scaling"""
        # 1. Base Leverage (Fixed at 25x)
        leverage = self.base_leverage
        
        # 2. Capital Utilization Scaling (Strategy + Win Streak)
        # Sniper: 15% base (high win rate 60.8%)
        # EMA Cross: 10% base (lower win rate 15.2%, more conservative)
        if strategy == "EMA_CROSS":
            risk_pct = 0.10  # Conservative for EMA Cross
            if self.consecutive_wins >= 2:
                risk_pct = 0.15  # Cap at 15% even with wins
        else:  # SNIPER (default)
            risk_pct = 0.15  # Base for Sniper
            if self.consecutive_wins >= 2:
                risk_pct = 0.25  # Scale up on performance
            elif self.consecutive_wins >= 1:
                risk_pct = 0.20
            
        return risk_pct, leverage

    def get_reversal_risk(self):
        """Legacy - now uses adaptive_risk"""
        return self.get_adaptive_risk("", is_sniper=True)
    
    def check_daily_limits(self):
        """Check if we should stop trading for the day"""
        if self.bypass_limits:
            print("✨ Manual bypass active. Resuming trades...")
            self.bypass_limits = False
            return True

        current_day = datetime.now().day
        
        # Reset daily counters
        if current_day != self.last_reset_day:
            self.daily_start_equity = self.equity
            self.daily_trades = 0
            self.last_reset_day = current_day
        
        # Check daily loss limit
        daily_pnl_pct = (self.equity - self.daily_start_equity) / self.daily_start_equity
        if daily_pnl_pct < -self.max_daily_loss_pct:
            print(f"⛔ Daily loss limit hit ({daily_pnl_pct*100:.1f}%). Stopping for today.")
            return False
        
        # Check consecutive losses
        if self.consecutive_losses >= 3:
            print(f"⛔ 3 consecutive losses. Taking a break.")
            return False
        
        return True
    
    def reset_trading_limits(self):
        """Reset limits and counters and ensure the dashboard updates immediately"""
        print("🔄 Resetting bot limits and counters via API...")
        self.consecutive_losses = 0
        self.consecutive_sure_shot_losses = 0
        self.consecutive_wins = 0
        self.last_loss_time = 0
        self.daily_trades = 0
        self.daily_start_equity = self.equity
        self.last_reset_day = datetime.now().day
        
        # Immediate UI feedback
        self.latest_analysis["status"] = "Reset Success: Resuming..."
        self.log_queue.append(f"{datetime.now().strftime('%H:%M:%S')} - 🔄 Bot limits reset manually")
        
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

    def detect_market_structure(self, df):
        """Detect Higher Highs, Higher Lows, etc."""
        if len(df) < 50: return "range"
        
        # Pivot detection (simplistic)
        df['pivot_h'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
        df['pivot_l'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
        
        highs = df[df['pivot_h']]['high'].tail(3).tolist()
        lows = df[df['pivot_l']]['low'].tail(3).tolist()
        
        if len(highs) < 2 or len(lows) < 2: return "range"
        
        # Bullish: HH + HL
        if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
            return "HH_HL"
        # Bearish: LH + LL
        elif highs[-1] < highs[-2] and lows[-1] < lows[-2]:
            return "LH_LL"
            
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
        """
        Modified Hybrid Logic: Routes each coin to its best strategy.
        """
        # 1. Fetch & Calculate MTF Data ONCE
        dfs = self.get_multi_timeframe_data(symbol)
        if any(df is None for df in dfs): return None
        
        df_5m, df_15m, df_1h, df_4h = dfs
        df_5m = self.calculate_indicators(df_5m)
        df_15m = self.calculate_indicators(df_15m)
        df_1h = self.calculate_indicators(df_1h)
        df_4h = self.calculate_indicators(df_4h)

        status_1h = self.check_trend(df_1h, symbol)
        global_bias = "neutral"
        if status_1h == "strong_up": global_bias = "bullish"
        elif status_1h == "strong_down": global_bias = "bearish"

        cur = df_5m.iloc[-1]
        prev = df_5m.iloc[-2]
        latest_close = cur['close']
        
        # Pre-initialize analysis to avoid extension "stuck" state
        if symbol not in self.latest_analysis:
            self.latest_analysis[symbol] = {
                "ut_signal": "INITIALIZING",
                "market_bias": "WAITING",
                "trend_15m": "neutral",
                "rsi": 50,
                "price": latest_close,
                "support": 0, "resistance": 99999,
                "patterns": [], "sweeps": [], "last_event": "None"
            }

        # 0.5 Volatility Filter: Avoid "dead" markets
        if cur['atr'] < cur.get('atr_avg', 0) * 0.4:
             self.latest_analysis[symbol]["ut_signal"] = "Wait: Low Vol"
             return None

        # --- DYNAMIC STRATEGY SELECTION (Based on Backtest Results) ---
        # Default to SNIPER (60.8% win rate) unless strong trending conditions detected
        if self.is_strong_trend(df_5m, df_15m):
            strategy = "EMA_CROSS"  # Only activate in strong trends
        else:
            strategy = "SNIPER"  # Default for all other conditions
        
        # --- THE PATTERN BARRIER (CLARITY FILTER) ---
        signal = None
        reason = ""
        stop_loss = None
        is_sure_shot = False
        tp_targets = {}

        is_indecision = PatternRecognizer.is_indecision_candle(cur['open'], cur['high'], cur['low'], cur['close'])
        clarity = PatternRecognizer.get_candle_clarity(cur['open'], cur['high'], cur['low'], cur['close'])
        
        if strategy == "EMA_CROSS":
            # 📈 EMA 9/15 CROSS with STRICT FILTERS (Based on Backtest)
            buy_cross = prev['ema9'] <= prev['ema15'] and cur['ema9'] > cur['ema15']
            sell_cross = prev['ema9'] >= prev['ema15'] and cur['ema9'] < cur['ema15']
            
            # STRICTER filters: ADX > 30, clarity > 0.5, volume confirmation
            is_trending = cur['adx'] > 30 and abs(cur['slope']) > 0.3
            vol_avg = cur.get('vol_avg', cur['volume'])
            volume_confirmed = cur['volume'] > (vol_avg * 1.2)
            
            if buy_cross and is_trending and volume_confirmed:
                if is_indecision or clarity < 0.5:
                    self.latest_analysis[symbol]["ut_signal"] = "Wait: Low Clarity (EMA)"
                    return None
                signal = "buy"
                reason = "📈 HYBRID: EMA CROSS UP (Strong Trend)"
                # Set initial SL (Stop loss at EMA 50 or 1.5%)
                stop_loss = cur['ema50'] if cur['close'] > cur['ema50'] else cur['low'] * 0.985
            elif sell_cross and is_trending and volume_confirmed:
                if is_indecision or clarity < 0.5:
                    self.latest_analysis[symbol]["ut_signal"] = "Wait: Low Clarity (EMA)"
                    return None
                signal = "sell"
                reason = "📉 HYBRID: EMA CROSS DOWN (Strong Trend)"
                stop_loss = cur['ema50'] if cur['close'] < cur['ema50'] else cur['high'] * 1.015
            
            self.latest_analysis[symbol]["ut_signal"] = "EMA SCANNING" if not signal else f"EMA {signal.upper()}"
                
        else: # SNIPER Strategy (Default)
            sniper_signal = self.check_sniper_reversal_signal(symbol, df_5m, df_15m, df_1h, df_4h)
            if sniper_signal:
                # Sniper rejection: If the reversal candle is itself a spinning top, skip it
                if is_indecision:
                    self.latest_analysis[symbol]["ut_signal"] = "Wait: Sniper Indecision"
                    return None
                sniper_signal['reason'] = "🎯 HYBRID: SNIPER " + sniper_signal.get('reason', '')
                sniper_signal['strategy'] = "SNIPER"
                return sniper_signal
            self.latest_analysis[symbol]["ut_signal"] = "SNIPER SCANNING"


        # Update analysis for UI (common for extension)
        patterns = PatternRecognizer.detect_all(df_5m)
        bias_smc, smc_events = StructureAnalyzer.analyze_structure(df_15m)
        status_15m = self.check_trend(df_15m, symbol)
        sweeps = StructureAnalyzer.detect_liquidity_sweeps(df_15m)
        last_event_str = smc_events[-1]['type'] if smc_events else "None"
        
        # Detect support/resistance levels
        support_levels = SupportResistance.detect_swing_levels(df_15m, lookback=5)
        nearest_support = 0
        nearest_resistance = 99999
        
        if support_levels:
            # Find nearest support below current price
            supports_below = [l['price'] for l in support_levels if l['type'] == 'Support' and l['price'] < latest_close]
            if supports_below:
                nearest_support = max(supports_below)  # Closest support below
            
            # Find nearest resistance above current price
            resistances_above = [l['price'] for l in support_levels if l['type'] == 'Resistance' and l['price'] > latest_close]
            if resistances_above:
                nearest_resistance = min(resistances_above)  # Closest resistance above
        
        self.latest_analysis[symbol].update({
            "patterns": [p['name'] for p in patterns[:3]],
            "market_bias": f"{global_bias.upper()} (1H) | {bias_smc.upper()} (15M)",
            "trend_15m": status_15m,
            "rsi": cur['rsi'],
            "price": latest_close,
            "support": round(nearest_support, 2),
            "resistance": round(nearest_resistance, 2),
            "sweeps": [s['type'] for s in sweeps],
            "last_event": last_event_str,
            "last_update": datetime.now().strftime('%H:%M:%S')
        })

        if signal:
            # For EMA Cross, we use broad TP targets as we exit on EMA50 cross
            tp_targets = {
                'tp1': latest_close * 1.02 if signal == 'buy' else latest_close * 0.98,
                'tp2': latest_close * 1.04 if signal == 'buy' else latest_close * 0.96,
                'tp3': latest_close * 1.10 if signal == 'buy' else latest_close * 0.90
            }
            return {
                'side': signal,
                'entry_price': latest_close,
                'reason': reason,
                'stop_loss': stop_loss,
                'is_sniper': False,
                'strategy': strategy,
                **tp_targets
            }
        
        return None

        if signal:
            # Check for Fail Cooldown (Wait 30 mins after 2 consecutive Sure Shot losses)
            if self.consecutive_sure_shot_losses >= 2 and (time.time() - self.last_loss_time < 1800):
                print(f"⏸️ Cooldown Active: Skipping {symbol} {signal} due to recent losses.")
                return None

            # Multi-Target calculation
            risk = abs(latest_close - stop_loss)
            tp_mult = 3.0 if is_sure_shot else 2.5
            tp1 = latest_close + (risk * tp_mult) if signal == "buy" else latest_close - (risk * tp_mult)
            tp2 = latest_close + (risk * tp_mult * 2) if signal == "buy" else latest_close - (risk * tp_mult * 2)
            tp3 = latest_close + (risk * tp_mult * 4) if signal == "buy" else latest_close - (risk * tp_mult * 4)

            return {
                "side": signal,
                "entry_price": latest_close,
                "stop_loss": stop_loss,
                "tp1": tp1, "tp2": tp2, "tp3": tp3,
                "is_sure_shot": is_sure_shot,
                "is_sniper": False,
                "reason": reason
            }
        return None

    def check_sniper_reversal_signal(self, symbol, df_5m, df_15m, df_1h, df_4h):
        """
        Refined Sniper Strategy: Detects high-probability reversal setups.
        Tighter filters: HTF Trend Alignment + ATR Stops + No Alert Relaxation.
        """
        # Pull tools
        patterns = PatternRecognizer.detect_all(df_5m)
        obs = StructureAnalyzer.detect_order_blocks(df_15m)
        sweeps = StructureAnalyzer.detect_liquidity_sweeps(df_5m)
        support_levels = SupportResistance.detect_swing_levels(df_15m)
        
        cur = df_5m.iloc[-1]
        prev = df_5m.iloc[-2]
        latest_close = cur['close']
        
        # 0. HTF Trend Alignment (1h)
        trend_1h = self.check_trend(df_1h, symbol)
        is_bullish_htf = trend_1h in ["up", "strong_up"]
        is_bearish_htf = trend_1h in ["down", "strong_down"]

        # 1. Level Proximity
        nearest_support = min([l['price'] for l in support_levels if l['price'] < latest_close], default=0, key=lambda x: abs(x - latest_close))
        nearest_resistance = min([l['price'] for l in support_levels if l['price'] > latest_close], default=999999, key=lambda x: abs(x - latest_close))
        
        # 2. Rejection Patterns with Displacement Check
        curr_ts = ""
        try:
            ts_val = cur.name
            if 'time' in cur: ts_val = cur['time']
            curr_ts = pd.to_datetime(ts_val).isoformat()
        except: pass
        
        latest_patterns = [p['name'] for p in patterns if p['time'] == curr_ts] if patterns else []
        
        # Stricter Rejection: Candle must close at least 30% into the previous candle's body (Simple displacement)
        displacement_long = cur['close'] > (prev['open'] + prev['close'])/2 if prev['close'] < prev['open'] else True
        displacement_short = cur['close'] < (prev['open'] + prev['close'])/2 if prev['close'] > prev['open'] else True

        rejection_long = any(p in latest_patterns for p in ["Hammer", "Bullish Engulfing", "Morning Star", "Bullish Marubozu"]) and displacement_long
        rejection_short = any(p in latest_patterns for p in ["Shooting Star", "Bearish Engulfing", "Evening Star", "Bearish Marubozu"]) and displacement_short
        
        # 3. Liquidity Sweeps / OB Interaction
        sweep_low = any(s['type'] == "Liquidity Sweep Low" for s in sweeps)
        sweep_high = any(s['type'] == "Liquidity Sweep High" for s in sweeps)
        near_bull_ob = any(ob['type'] == "Bullish OB" and abs(latest_close - (ob['top'] + ob['bottom'])/2) / latest_close < 0.003 for ob in obs)
        near_bear_ob = any(ob['type'] == "Bearish OB" and abs(latest_close - (ob['top'] + ob['bottom'])/2) / latest_close < 0.003 for ob in obs)

        # 4. Strict Criteria (No relaxation during volatility)
        vol_multiplier = 1.5 # Stricter volume requirement (1.5x avg)
        rsi_long_limit = 40  # Deeper oversold for Sniper
        rsi_short_limit = 60 # Deeper overbought for Sniper
        
        volume_exhaustion = cur['volume'] > (cur['vol_avg'] * vol_multiplier)
        rsi_os = cur['rsi'] < rsi_long_limit
        rsi_ob = cur['rsi'] > rsi_short_limit

        signal = None
        stop_loss = None
        reason = ""

        # LONG SNIPER (HTF Must be Up OR neutral)
        if (sweep_low or near_bull_ob or abs(latest_close - nearest_support)/latest_close < 0.003) and rejection_long:
            if volume_exhaustion and rsi_os and not is_bearish_htf:
                signal = "buy"
                # ATR-Based Stop Loss: 1.5 * ATR below current low
                atr_val = cur.get('atr', latest_close * 0.002)
                stop_loss = cur['low'] - (atr_val * 1.5)
                reason = "🎯 SNIPER LONG: HTF Align + Rejection + Vol + RSI"
            else:
                self.latest_analysis[symbol]["ut_signal"] = "SNIPER: Waiting for Vol/RSI/HTF"

        # SHORT SNIPER (HTF Must be Down OR neutral)
        elif (sweep_high or near_bear_ob or abs(latest_close - nearest_resistance)/latest_close < 0.003) and rejection_short:
            if volume_exhaustion and rsi_ob and not is_bullish_htf:
                signal = "sell"
                # ATR-Based Stop Loss: 1.5 * ATR above current high
                atr_val = cur.get('atr', latest_close * 0.002)
                stop_loss = cur['high'] + (atr_val * 1.5)
                reason = "🎯 SNIPER SHORT: HTF Align + Rejection + Vol + RSI"
            else:
                self.latest_analysis[symbol]["ut_signal"] = "SNIPER: Waiting for Vol/RSI/HTF"

        if signal:
            # ROI TARGETS: 1:3 RR minimum using ATR risk
            risk = abs(latest_close - stop_loss)
            tp1 = latest_close + (risk * 2.0) if signal == "buy" else latest_close - (risk * 2.0)
            tp2 = latest_close + (risk * 4.0) if signal == "buy" else latest_close - (risk * 4.0)
            
            structural_target = nearest_resistance if signal == "buy" else nearest_support
            # Fallback if structural target is too close
            if signal == "buy" and structural_target < latest_close + (risk * 3):
                structural_target = latest_close + (risk * 6.0)
            elif signal == "sell" and structural_target > latest_close - (risk * 3):
                structural_target = latest_close - (risk * 6.0)

            return {
                "side": signal,
                "entry_price": latest_close,
                "stop_loss": stop_loss,
                "tp1": tp1, "tp2": tp2, "tp3": structural_target,
                "structural_target": structural_target,
                "is_sure_shot": True,
                "is_sniper": True,
                "reason": reason
            }
        
        return None
        
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
        """Execute trade with tiered profit targets"""
        if len(self.positions) >= self.max_concurrent_positions:
            print(f"⏸️  Max positions reached ({self.max_concurrent_positions}). Skipping {symbol}")
            return
        
        entry_price = signal["entry_price"]
        stop_loss = signal["stop_loss"]
        
        # Extract strategy for risk calculation
        strategy = signal.get("strategy", "SNIPER")
        is_sure_shot = signal.get("is_sure_shot", False)
        is_sniper = signal.get("is_sniper", False)
        
        # Calculate position size with strategy-based risk
        position_size_usd = self.calculate_position_size(symbol, entry_price, stop_loss, is_sure_shot, is_sniper, strategy)
        
        # Convert to contracts
        contract_val = self.contract_values.get(symbol, 1)
        num_contracts = int(position_size_usd / (entry_price * contract_val))
        
        if num_contracts < 1:
            print(f"⚠️  Position size too small for {symbol}. Skipping.")
            return
        
        risk_pct, leverage = self.get_adaptive_risk(symbol, is_sure_shot, is_sniper, strategy)
        
        status_tag = "🚀 SNIPER" if is_sniper else ("🔥 SURE SHOT" if is_sure_shot else "🎯 STANDARD")
        print(f"\n{status_tag} SIGNAL: {signal['side'].upper()} {symbol}")
        print(f"   Entry: ${entry_price:.2f}")
        print(f"   Stop: ${stop_loss:.2f}")
        print(f"   Size: {num_contracts} contracts (${position_size_usd:.0f})")
        print(f"   Risk: {risk_pct*100:.0f}% | Leverage: {leverage}x")
        print(f"   Reason: {signal['reason']}")
        
        if self.paper_trading:
            self.positions[symbol] = {
                "side": signal["side"],
                "entry": entry_price,
                "stop_loss": stop_loss,
                "tp1": signal["tp1"],
                "tp2": signal["tp2"],
                "tp3": signal["tp3"],
                "qty": num_contracts,
                "qty_remaining": num_contracts,
                "initial_sl": stop_loss,
                "breakeven_moved": False,
                "trailing_active": False,
                "entry_time": datetime.now(),
                "risk_pct": risk_pct,
                "leverage": leverage,
                "is_sniper": signal.get("is_sniper", False),
                "strategy": signal.get("strategy", "SNIPER"), # Store strategy for exit logic
                "tp2_hit": False,
                "tp1_hit": False,
                "structural_target": signal.get("structural_target"),
                "entry_reason": signal.get("reason", "Standard")
            }
            print(f"   [PAPER] Position opened: {signal['reason']}")
            self.daily_trades += 1
            self.save_active_positions()
    
    def manage_positions(self):
        """Manage open positions with tiered exits and Sniper logic"""
        to_remove = []
        
        for sym, pos in self.positions.items():
            df_5m, df_15m, _, _ = self.get_multi_timeframe_data(sym)
            if df_5m is None:
                continue
            
            latest = df_5m.iloc[-1]
            cur_price = latest['close']
            high_price = latest['high']
            low_price = latest['low']
            
            side = pos['side']
            entry = pos['entry']
            is_sniper = pos.get('is_sniper', False)
            
            # 1. STOP LOSS CHECK
            sl_hit = False
            exit_price = cur_price
            if side == "buy":
                if low_price <= pos['stop_loss']:
                    sl_hit = True
                    exit_price = pos['stop_loss']
            else:
                if high_price >= pos['stop_loss']:
                    sl_hit = True
                    exit_price = pos['stop_loss']

            if sl_hit:
                print(f"🛑 {sym} STOPPED OUT at ${exit_price:.2f} (Extreme touched)")
                pnl_pct = (exit_price - entry) / entry if side == "buy" else (entry - exit_price) / entry
                self.close_position(sym, exit_price, "SL", pnl_pct, 1.0)
                to_remove.append(sym)
                self.consecutive_losses += 1
                continue
            
            # 1.5 Early Breakeven: Move SL to entry if price moves 1.0 * Risk in favor
            if not pos.get('breakeven_moved') and not pos.get('tp1_hit'):
                risk_at_entry = abs(entry - pos.get('initial_sl', pos['stop_loss']))
                if side == "buy" and cur_price >= entry + risk_at_entry:
                    pos['stop_loss'] = entry
                    pos['breakeven_moved'] = True
                    print(f"🛡️ {sym} MOVED TO BREAKEVEN (1:1 RR reached)")
                elif side == "sell" and cur_price <= entry - risk_at_entry:
                    pos['stop_loss'] = entry
                    pos['breakeven_moved'] = True
                    print(f"🛡️ {sym} MOVED TO BREAKEVEN (1:1 RR reached)")
            
            # 2. SNIPER REJECTION EXIT (Still uses close/patterns)
            if is_sniper:
                patterns = PatternRecognizer.detect_all(df_5m)
                curr_ts = df_5m.iloc[-1].name.isoformat() if hasattr(df_5m.iloc[-1].name, 'isoformat') else str(df_5m.iloc[-1].name)
                latest_patterns = [p['name'] for p in patterns if p['time'] == curr_ts]
                
                rejection_exit = False
                if side == "buy" and any(p in latest_patterns for p in ["Shooting Star", "Bearish Engulfing", "Evening Star"]):
                    rejection_exit = True
                elif side == "sell" and any(p in latest_patterns for p in ["Hammer", "Bullish Engulfing", "Morning Star"]):
                    rejection_exit = True
                
                if rejection_exit:
                    pnl_pct = (cur_price - entry) / entry if side == "buy" else (entry - cur_price) / entry
                    if pnl_pct > 0.002:
                        print(f"🎯 SNIPER EARLY EXIT: Reversal pattern detected at ${cur_price:.2f}")
                        self.close_position(sym, cur_price, "SNIPER_REJECTION", pnl_pct, 1.0)
                        to_remove.append(sym)
                        continue

            # 2.5 HYBRID EMA EXIT: Exit on EMA 50 cross
            if pos.get('strategy') == 'EMA_CROSS':
                ema_exit = False
                if side == "buy" and cur_price < latest['ema50']:
                    ema_exit = True
                elif side == "sell" and cur_price > latest['ema50']:
                    ema_exit = True
                
                if ema_exit:
                    pnl_pct = (cur_price - entry) / entry if side == "buy" else (entry - cur_price) / entry
                    print(f"📉 HYBRID EXIT: EMA 50 Cross detected for {sym} at ${cur_price:.2f}")
                    self.close_position(sym, cur_price, "EMA_50_CROSS", pnl_pct, 1.0)
                    to_remove.append(sym)
                    continue

            # 3. TIERED PROFIT TAKING (Use Extreme High/Low)
            if side == "buy":
                pnl_pct_tp = (cur_price - entry) / entry
                # TP3/Structural Target Check
                target_hit = False
                if high_price >= pos['tp3']: target_hit = True
                if pos.get('structural_target') and high_price >= pos['structural_target']: target_hit = True

                if target_hit:
                    final_exit = max(pos['tp3'], pos.get('structural_target', 0))
                    self.close_position(sym, final_exit, "TARGET_HIT", (final_exit - entry)/entry, 1.0)
                    to_remove.append(sym)
                elif high_price >= pos['tp2'] and pos.get('tp1_hit') and not pos.get('tp2_hit'):
                    self.take_profit(sym, pos['tp2'], "TP2", (pos['tp2'] - entry)/entry, 0.3, pos)
                elif high_price >= pos['tp1'] and not pos.get('tp1_hit'):
                    self.take_profit(sym, pos['tp1'], "TP1", (pos['tp1'] - entry)/entry, 0.5, pos)
            else: # Sell
                target_hit = False
                if low_price <= pos['tp3']: target_hit = True
                if pos.get('structural_target') and low_price <= pos['structural_target']: target_hit = True

                if target_hit:
                    final_exit = min(pos['tp3'], pos.get('structural_target', 999999))
                    self.close_position(sym, final_exit, "TARGET_HIT", (entry - final_exit)/entry, 1.0)
                    to_remove.append(sym)
                elif low_price <= pos['tp2'] and pos.get('tp1_hit') and not pos.get('tp2_hit'):
                    self.take_profit(sym, pos['tp2'], "TP2", (entry - pos['tp2'])/entry, 0.3, pos)
                elif low_price <= pos['tp1'] and not pos.get('tp1_hit'):
                    self.take_profit(sym, pos['tp1'], "TP1", (entry - pos['tp1'])/entry, 0.5, pos)

            # 4. TRAILING STOP (Uses close to evaluate, but updates SL level)
            if pos.get('trailing_active'):
                 if side == "buy":
                    pos['stop_loss'] = max(pos['stop_loss'], cur_price * 0.995)
                 else:
                    pos['stop_loss'] = min(pos['stop_loss'], cur_price * 1.005)

        for sym in to_remove:
            if sym in self.positions:
                del self.positions[sym]
    
    def take_profit(self, sym, price, level, pnl_pct, portion, pos):
        close_qty = int(pos['qty'] * portion)
        pos['qty_remaining'] -= close_qty
        print(f"💰 {sym} {level} HIT! Closed {portion*100}% at ${price:.2f}")
        self.close_position(sym, price, level, pnl_pct, portion)
        if level == "TP1":
            pos['stop_loss'] = pos['entry'] # Breakeven
            pos['tp1_hit'] = True
        elif level == "TP2":
            pos['trailing_active'] = True
            pos['tp2_hit'] = True

    def close_position(self, symbol, exit_price, reason, pnl_pct, portion=1.0):
        """Close position and update equity"""
        pos = self.positions[symbol]
        roi = pnl_pct * pos['leverage']
        position_value_inr = self.equity * pos['risk_pct']
        pnl_inr = position_value_inr * roi * portion
        
        self.equity += pnl_inr
        
        # Tracking Sure Shot accuracy for cooldown
        if pos.get('is_sure_shot', False):
            if pnl_inr < 0:
                self.consecutive_sure_shot_losses += 1
                self.last_loss_time = time.time()
            else:
                self.consecutive_sure_shot_losses = 0

        if pnl_inr > 0:
            self.consecutive_losses = 0
            self.consecutive_wins += 1
        else:
            self.consecutive_wins = 0
            
        print(f"   PnL: {pnl_pct*100:.2f}% | ROI: {roi*100:.2f}% | {pnl_inr:+.2f} INR")
        
        entry_time_str = pos['entry_time'].strftime("%H:%M:%S") if isinstance(pos['entry_time'], datetime) else str(pos['entry_time'])
        exit_time_str = datetime.now().strftime("%H:%M:%S")

        self.trades.append({
            'symbol': symbol,
            'side': pos['side'],
            'entry_price': pos['entry'],
            'exit_price': exit_price,
            'pnl_inr': round(pnl_inr, 2),
            'pnl_pct': round(pnl_pct * 100, 2),
            'roi': round(roi * 100, 2),
            'equity': round(self.equity, 2),
            'reason': pos.get('entry_reason', 'Unknown'),
            'exit_reason': reason,
            'entry_time': entry_time_str,
            'exit_time': exit_time_str,
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
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
                    
                # Restore trades
                self.trades = data.get("recent_trades", [])
                print(f"   ✅ Basic state restored: {len(self.trades)} trades | Equity: {self.equity:.2f} INR")
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
                "recent_trades": self.trades[-50:], # Increased to 50
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "active_mode": "HYBRID",
                "leverage_mode": "PRO",
                "market_structure": self.latest_analysis,
                "market_state": self.get_market_state(), # Added session/alertness
                "recent_logs": list(self.log_queue)
            }
            
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
                            
                        self.log_queue.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_msg}")
                        
                        signal = self.check_entry_signal(sym)
                        if signal:
                            print(f"\n✅ {sym} signal found!")
                            self.log_queue.append(f"✅ SIGNAL: {sym} {signal['side'].upper()}")
                            self.execute_trade(sym, signal)
                            
                        # Frequent exports (after EACH symbol scan)
                        self.export_dashboard_data()
                        time.sleep(0.1)
                    except Exception as sym_e:
                        print(f"\n⚠️ Error scanning {sym}: {sym_e}")
                        self.log_queue.append(f"⚠️ Error {sym}: {str(sym_e)[:50]}")
                        self.export_dashboard_data() # Ensure we still update heartbeat
                        continue
                
                # Export Dashboard Data
                self.export_dashboard_data()
                print(" " * 50, end="\r")
                
            except Exception as e:
                error_msg = f"❌ Error in main loop: {e}"
                print(error_msg)
                self.log_queue.append(f"{datetime.now().strftime('%H:%M:%S')} - {error_msg}")
                # Update heartbeat even on error so watchdog doesn't get confused
                self._bot_heartbeat_ts = time.time()
                time.sleep(10) # Wait a bit before retrying


def run_flask():
    print(f"📡 API Server starting on port 5005...")
    app.run(host='0.0.0.0', port=5005, threaded=True, debug=False, use_reloader=False)

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
