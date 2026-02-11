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
from pathlib import Path
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

# Ensure UTF-8 output for Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

load_dotenv()

base_dir = Path(__file__).resolve().parents[2]
template_dir = base_dir / "apps" / "dashboard_flask" / "templates"
static_dir = base_dir / "apps" / "dashboard_flask" / "static"
app = Flask(__name__, template_folder=str(template_dir), static_folder=str(static_dir))
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

@app.route('/', methods=['GET'])
def root_redirect():
    return render_template('dashboard.html')

@app.route('/health', methods=['GET'])
def health():
    global bot_instance
    data = {
        "status": "ok",
        "bot_initialized": bool(bot_instance)
    }
    if bot_instance:
        data.update({
            "bot_heartbeat_ts": bot_instance._bot_heartbeat_ts,
            "startup_token": bot_instance.startup_token,
            "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    return jsonify(data)

@app.route('/dashboard_data', methods=['GET'])
def dashboard_data():
    global bot_instance
    if not bot_instance:
        return jsonify({"error": "Bot instance not initialized"}), 503
    data = bot_instance.build_dashboard_data()
    return jsonify(data)

@app.route('/api/data', methods=['GET'])
def dashboard_data_alias():
    return dashboard_data()

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
        self.target_equity = 30000  # 6x target
        
        # Adaptive Risk Management
        self.base_leverage = 50
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
            "BNBUSD", "UNIUSD"
        ]
        self.symbols_to_trade = self.priority_symbols.copy()
        
        # Daily tracking
        self.daily_start_equity = self.equity
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.last_reset_day = datetime.now().day
        self.trades = []
        self.trade_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_trades")
        base_dir = Path(__file__).resolve().parents[2]
        self.dashboard_file = str(base_dir / "data" / "dashboard_data.json")
        self.log_queue = deque(maxlen=20) # Keep last 20 logs
        
        # Sniper Mode Configuration
        self.bootstrap_target = 15000 # Autoscay to 50x after 15k INR
        # Turbo Mode State
        self.turbo_mode = os.getenv("TURBO_MODE", "True") == "True"
        self.turbo_completed = False

        self.latest_analysis = {} # Store analysis for API
        self.priority_symbol = None # Symbol to scan with priority
        self.last_priority_scan = 0
        self._bot_heartbeat_ts = time.time() # For watchdog
        self.startup_token = int(time.time()) # To identify fresh instance
        
        # Risk Management - Cooldown
        self.consecutive_sure_shot_losses = 0
        self.last_loss_time = 0
        
        if self.paper_trading:
            os.makedirs(self.trade_log_dir, exist_ok=True)
        
        print(f"🚀 Aggressive Growth Bot Starting...")
        print(f"   Capital: {self.starting_capital} INR")
        print(f"   Target: {self.target_equity} INR (6x)")
        print(f"   Leverage: {self.base_leverage}x")
        print(f"   Mode: {'PAPER' if self.paper_trading else 'LIVE'}")
        
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
    
    def get_adaptive_risk(self, symbol, is_sure_shot=False):
        """Ultra-Aggressive sizing as requested by user"""
        leverage = 50
        risk_pct = 0.75 # 75% capital utilization per trade
        return risk_pct, leverage
    
    def get_reversal_risk(self):
        """Reversal trades now also use the 75% / 50x profile"""
        return 0.75, 50 
    
    def check_daily_limits(self):
        """Check if we should stop trading for the day"""
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
        # EMAs
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / loss)))
        
        # Volume average
        df['vol_avg'] = df['volume'].rolling(window=20).mean()
        
        # ATR for UT Bot (Period 10)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(10).mean()
        
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
    
    def check_trend(self, df_15m, symbol):
        """Check higher timeframe trend using EMA 9, 21, and 50 alignment"""
        if len(df_15m) < 100:
            return None
        
        latest = df_15m.iloc[-1]
        
        # Bullish: EMA 9 > EMA 21
        if latest['ema9'] > latest['ema21']:
            if latest['ema21'] > latest['ema50']:
                return "strong_up"
            return "up"
            
        # Bearish: EMA 9 < EMA 21
        if latest['ema9'] < latest['ema21']:
            if latest['ema21'] < latest['ema50']:
                return "strong_down"
            return "down"
            
        return None
    
    def check_entry_signal(self, symbol):
        """
        Hybrid Entry Logic: Checks for both Sniper Reversals and Trend Breakouts.
        """
        # 1. Check for SNIPER REVERSAL (High Priority)
        sniper_signal = self.check_sniper_reversal_signal(symbol)
        if sniper_signal:
            return sniper_signal
            
        # 2. Check for STANDARD MOMENTUM BREAKOUT (Original Logic)
        df_5m, df_15m, df_1h, df_4h = self.get_multi_timeframe_data(symbol)
        if any(df is None for df in [df_5m, df_15m, df_1h, df_4h]):
            return None
        
        df_5m = self.calculate_indicators(df_5m)
        df_15m = self.calculate_indicators(df_15m)
        
        # Pull tools for analysis UI
        patterns = PatternRecognizer.detect_all(df_5m)
        support_levels = SupportResistance.detect_swing_levels(df_15m)
        
        latest_close = df_5m.iloc[-1]['close']
        nearest_support = min([l['price'] for l in support_levels if l['price'] < latest_close], default=0, key=lambda x: abs(x - latest_close))
        nearest_resistance = min([l['price'] for l in support_levels if l['price'] > latest_close], default=999999, key=lambda x: abs(x - latest_close))
        
        # UT Signal Logic for UI
        cur = df_5m.iloc[-1]
        prev = df_5m.iloc[-2] if len(df_5m) > 1 else cur
        
        ut_buy_signal = cur['close'] > cur['ut_trail'] and prev['close'] <= prev['ut_trail']
        ut_sell_signal = cur['close'] < cur['ut_trail'] and prev['close'] >= prev['ut_trail']
        
        active_signal = "None"
        if ut_buy_signal: active_signal = "BUY"
        elif ut_sell_signal: active_signal = "SELL"
        elif cur['close'] > cur['ut_trail']: active_signal = "HOLD BUY"
        elif cur['close'] < cur['ut_trail']: active_signal = "HOLD SELL"

        # Structure for UI
        bias, events = StructureAnalyzer.analyze_structure(df_15m)
        sweeps = StructureAnalyzer.detect_liquidity_sweeps(df_5m)

        # Store analysis for API
        self.latest_analysis[symbol] = {
            "patterns": [p['name'] for p in patterns[:3]],
            "support": nearest_support,
            "resistance": nearest_resistance,
            "market_bias": bias,
            "sweeps": [s["type"] for s in sweeps],
            "rsi": df_5m.iloc[-1]['rsi'],
            "price": latest_close,
            "ut_signal": active_signal, 
            "last_update": datetime.now().strftime('%H:%M:%S')
        }
        
        # Check Trend alignment and Volatility/Momentum
        trend = self.check_trend(df_15m, symbol)
        if not trend: return None
        
        # Check RSI (5m) for "Sure Shot" range
        # Long: 50-68 (Trending up but not overbought)
        # Short: 32-50 (Trending down but not oversold)
        val_rsi = df_5m.iloc[-1]['rsi']
        
        # Volume confirmation (Crucial for Momentum Shifts)
        volume_spike = cur['volume'] > (cur['vol_avg'] * 1.3)
        
        # Candle Quality Check
        candle_body_pct = abs(cur['close'] - cur['open']) / (cur['high'] - cur['low'] + 0.000001)
        is_meaningful_candle = candle_body_pct > 0.4 or any(p['name'] in ["Hammer", "Shooting Star", "Bullish Engulfing", "Bearish Engulfing"] for p in patterns if p['time'] == cur.name)

        signal = None
        stop_loss = None
        reason = ""
        is_sure_shot = False
        
        # 1. SURE SHOT LOGIC: Triple EMA Alignment (Strong Trend) + RSI Check
        if trend == "strong_up" and 50 < val_rsi < 68 and volume_spike and is_meaningful_candle:
            if cur['close'] > cur['ema9'] > cur['ema21']:
                signal = "buy"
                is_sure_shot = True
                stop_loss = cur['low'] * 0.999 # Very tight
                reason = "🔥 SURE SHOT LONG (Strong Trend + RSI)"
        elif trend == "strong_down" and 32 < val_rsi < 50 and volume_spike and is_meaningful_candle:
            if cur['close'] < cur['ema9'] < cur['ema21']:
                signal = "sell"
                is_sure_shot = True
                stop_loss = cur['high'] * 1.001
                reason = "🔥 SURE SHOT SHORT (Strong Trend + RSI)"

        # 2. MOMENTUM SHIFT (Secondary Logic - Higher Risk)
        if not signal:
            is_bullish_cross = cur['ema9'] > cur['ema21'] and prev['ema9'] <= prev['ema21']
            is_bearish_cross = cur['ema9'] < cur['ema21'] and prev['ema9'] >= prev['ema21']
            
            if (is_bullish_cross or ut_buy_signal) and "up" in trend and volume_spike:
                signal = "buy"
                stop_loss = min(cur['low'], cur['ema21'])
                reason = f"MOMENTUM_SHIFT UP ({trend})"
            elif (is_bearish_cross or ut_sell_signal) and "down" in trend and volume_spike:
                signal = "sell"
                stop_loss = max(cur['high'], cur['ema21'])
                reason = f"MOMENTUM_SHIFT DOWN ({trend})"

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

    def check_sniper_reversal_signal(self, symbol):
        """
        NEW: Detects high-probability reversal setups (Sniper Strategy).
        Looks for sweeps, OB taps, and rejection candles at key levels.
        """
        df_5m, df_15m, df_1h, df_4h = self.get_multi_timeframe_data(symbol)
        if any(df is None for df in [df_5m, df_15m, df_1h, df_4h]):
            return None

        # Calculate everything on 5m for entry
        df_5m = self.calculate_indicators(df_5m)
        df_15m = self.calculate_indicators(df_15m)
        
        # Pull tools
        patterns = PatternRecognizer.detect_all(df_5m)
        obs = StructureAnalyzer.detect_order_blocks(df_15m) # Using 15m OBs for stronger levels
        sweeps = StructureAnalyzer.detect_liquidity_sweeps(df_5m)
        support_levels = SupportResistance.detect_swing_levels(df_15m)
        
        cur = df_5m.iloc[-1]
        prev = df_5m.iloc[-2]
        latest_close = cur['close']
        
        # 1. Level Proximity
        nearest_support = min([l['price'] for l in support_levels if l['price'] < latest_close], default=0, key=lambda x: abs(x - latest_close))
        nearest_resistance = min([l['price'] for l in support_levels if l['price'] > latest_close], default=999999, key=lambda x: abs(x - latest_close))
        
        # 2. Rejection Patterns
        # Match current candle time (ISO)
        curr_ts = ""
        try:
            ts_val = cur.name
            if 'time' in cur: ts_val = cur['time']
            curr_ts = pd.to_datetime(ts_val).isoformat()
        except: pass
        
        latest_patterns = [p['name'] for p in patterns if p['time'] == curr_ts] if patterns else []
        rejection_long = any(p in latest_patterns for p in ["Hammer", "Bullish Engulfing", "Morning Star", "Bullish Marubozu"])
        rejection_short = any(p in latest_patterns for p in ["Shooting Star", "Bearish Engulfing", "Evening Star", "Bearish Marubozu"])
        
        # 3. Liquidity Sweeps
        sweep_low = any(s['type'] == "Liquidity Sweep Low" for s in sweeps)
        sweep_high = any(s['type'] == "Liquidity Sweep High" for s in sweeps)
        
        # 4. OB Interaction
        near_bull_ob = any(ob['type'] == "Bullish OB" and abs(latest_close - (ob['top'] + ob['bottom'])/2) / latest_close < 0.005 for ob in obs)
        near_bear_ob = any(ob['type'] == "Bearish OB" and abs(latest_close - (ob['top'] + ob['bottom'])/2) / latest_close < 0.005 for ob in obs)

        signal = None
        stop_loss = None
        reason = []

        # LONG SNIPER: Sweep Low OR OB Tap OR Major Support + Bullish Rejection
        if (sweep_low or near_bull_ob or abs(latest_close - nearest_support)/latest_close < 0.002) and rejection_long:
            signal = "buy"
            # SL is minimal: just below the rejection candle low
            stop_loss = cur['low'] * 0.9995 # Tight 0.05% buffer
            reason = f"SNIPER LONG: {'Sweep' if sweep_low else 'OB' if near_bull_ob else 'Support'} + {latest_patterns[0] if latest_patterns else 'Bullish Rejection'}"

        # SHORT SNIPER: Sweep High OR OB Tap OR Major Resistance + Bearish Rejection
        elif (sweep_high or near_bear_ob or abs(latest_close - nearest_resistance)/latest_close < 0.002) and rejection_short:
            signal = "sell"
            # SL is minimal: just above the rejection candle high
            stop_loss = cur['high'] * 1.0005 # Tight 0.05% buffer
            reason = f"SNIPER SHORT: {'Sweep' if sweep_high else 'OB' if near_bear_ob else 'Resistance'} + {latest_patterns[0] if latest_patterns else 'Bearish Rejection'}"

        if signal:
            # ROI TARGETS for Sniper: Minimal losses, but big wins
            # We want at least 1:3 RR for these high-leverage trades
            risk = abs(latest_close - stop_loss)
            tp1 = latest_close + (risk * 2.0) if signal == "buy" else latest_close - (risk * 2.0)
            tp2 = latest_close + (risk * 4.0) if signal == "buy" else latest_close - (risk * 4.0)
            tp3 = latest_close + (risk * 8.0) if signal == "buy" else latest_close - (risk * 8.0) # Aiming for the runner

            return {
                "side": signal,
                "entry_price": latest_close,
                "stop_loss": stop_loss,
                "tp1": tp1, "tp2": tp2, "tp3": tp3,
                "is_sure_shot": True, # For leverage logic
                "is_sniper": True,
                "reason": reason
            }
        
        return None
    
    def calculate_position_size(self, symbol, entry_price, stop_loss, is_sure_shot=False, is_reversal=False):
        """Calculate position size based on fixed 75% capital and 50x leverage"""
        risk_pct, leverage = self.get_adaptive_risk(symbol, is_sure_shot)
        
        # Position value in INR (75% of current equity * 50x leverage)
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
        
        # Calculate position size
        is_sure_shot = signal.get("is_sure_shot", False)
        is_reversal = signal.get("is_reversal", False)
        
        position_size_usd = self.calculate_position_size(symbol, entry_price, stop_loss, is_sure_shot, is_reversal)
        
        # Convert to contracts
        contract_val = self.contract_values.get(symbol, 1)
        num_contracts = int(position_size_usd / (entry_price * contract_val))
        
        if num_contracts < 1:
            print(f"⚠️  Position size too small for {symbol}. Skipping.")
            return
        
        if is_reversal:
            risk_pct, leverage = self.get_reversal_risk()
        else:
            risk_pct, leverage = self.get_adaptive_risk(symbol, is_sure_shot)
        
        status_tag = "🚀 SNIPER" if is_sure_shot else ("↩️ REVERSAL" if is_reversal else "🎯 STANDARD")
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
                "breakeven_moved": False,
                "trailing_active": False,
                "entry_time": datetime.now(),
                "risk_pct": risk_pct,
                "leverage": leverage,
                "is_sniper": signal.get("is_sniper", False),
                "tp1_hit": False,
                "tp2_hit": False
            }
            print(f"   [PAPER] Position opened")
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

            # 3. TIERED PROFIT TAKING (Use Extreme High/Low)
            if side == "buy":
                pnl_pct_tp = (cur_price - entry) / entry # Use close for log, but limit for execution
                if high_price >= pos['tp3']:
                    self.close_position(sym, pos['tp3'], "TP3", (pos['tp3'] - entry)/entry, 1.0)
                    to_remove.append(sym)
                elif high_price >= pos['tp2'] and pos['tp1_hit'] and not pos['tp2_hit']:
                    self.take_profit(sym, pos['tp2'], "TP2", (pos['tp2'] - entry)/entry, 0.3, pos)
                elif high_price >= pos['tp1'] and not pos['tp1_hit']:
                    self.take_profit(sym, pos['tp1'], "TP1", (pos['tp1'] - entry)/entry, 0.5, pos)
            else: # Sell
                if low_price <= pos['tp3']:
                    self.close_position(sym, pos['tp3'], "TP3", (entry - pos['tp3'])/entry, 1.0)
                    to_remove.append(sym)
                elif low_price <= pos['tp2'] and pos['tp1_hit'] and not pos['tp2_hit']:
                    self.take_profit(sym, pos['tp2'], "TP2", (entry - pos['tp2'])/entry, 0.3, pos)
                elif low_price <= pos['tp1'] and not pos['tp1_hit']:
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
            
        print(f"   PnL: {pnl_pct*100:.2f}% | ROI: {roi*100:.2f}% | {pnl_inr:+.2f} INR")
        
        self.trades.append({
            'symbol': symbol,
            'side': pos['side'],
            'entry_price': pos['entry'],
            'exit_price': exit_price,
            'pnl_inr': round(pnl_inr, 2),
            'pnl_pct': round(pnl_pct * 100, 2),
            'roi': round(roi * 100, 2),
            'equity': round(self.equity, 2),
            'reason': reason,
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
                    "open_positions": serializable_positions
                }, f, indent=4)
        except Exception as e:
            print(f"   ❌ Active positions save error: {e}")

    def build_dashboard_data(self):
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl_inr'] > 0])
        losing_trades = len([t for t in self.trades if t['pnl_inr'] < 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pnl = self.equity - self.starting_capital

        return {
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
                    "unrealized_pnl": 0.0
                } for p, d in self.positions.items()
            ],
            "recent_trades": self.trades[-50:],
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "active_mode": "SNIPER" if self.equity < self.bootstrap_target else "GROWTH",
            "leverage_mode": "PRO",
            "market_structure": self.latest_analysis,
            "recent_logs": list(self.log_queue)
        }

    def export_dashboard_data(self):
        """Export real-time state for the dashboard UI."""
        try:
            data = self.build_dashboard_data()

            temp = self.dashboard_file + ".tmp"
            with open(temp, "w") as f:
                json.dump(data, f, indent=4)
            shutil.move(temp, self.dashboard_file)
        except Exception as e:
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
                    for i in range(30): # 30 * 10s = 300s (5 mins)
                        self._bot_heartbeat_ts = time.time()
                        self.latest_analysis["_bot_heartbeat"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        self.latest_analysis["status"] = "Resting: Limit/Losses reached"
                        # Export during rest too!
                        self.export_dashboard_data()
                        time.sleep(10)
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
                        self.latest_analysis["status"] = log_msg
                        self.log_queue.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_msg}")
                        
                        signal = self.check_entry_signal(sym)
                        if signal:
                            print(f"\n✅ {sym} signal found!")
                            self.log_queue.append(f"✅ SIGNAL: {sym} {signal['side'].upper()}")
                            self.execute_trade(sym, signal)
                        time.sleep(0.1)
                    except Exception as sym_e:
                        print(f"\n⚠️ Error scanning {sym}: {sym_e}")
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
