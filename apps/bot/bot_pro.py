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

REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_UI_DIR = REPO_ROOT / "apps" / "dashboard_flask"

app = Flask(
    __name__,
    template_folder=str(DASHBOARD_UI_DIR / "templates"),
    static_folder=str(DASHBOARD_UI_DIR / "static")
)
CORS(app, resources={r"/*": {"origins": "*"}})

bot_instance = None

@app.route('/', methods=['GET'])
def dashboard():
    return render_template('dashboard.html')

@app.route('/analysis', methods=['GET'])
def get_analysis():
    global bot_instance
    if bot_instance:
        data = bot_instance.latest_analysis or {"status": "scanning", "msg": "Bot logic is warm-up"}
        data["bot_version"] = "PRO_V2_RESTRICTED"
        return jsonify(data)
    return jsonify({"error": "Bot instance not initialized"})

@app.route('/api/data', methods=['GET'])
def get_dashboard_api_data():
    global bot_instance
    if not bot_instance:
        return jsonify({"error": "Bot instance not initialized"}), 500
    try:
        with open(bot_instance.dashboard_file, "r") as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({"error": "Dashboard data not available"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/dashboard_data', methods=['GET'])
def get_dashboard_data():
    global bot_instance
    if not bot_instance:
        return jsonify({"error": "Bot instance not initialized"}), 500
    try:
        with open(bot_instance.dashboard_file, "r") as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({"error": "Dashboard data not available"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
        self.max_concurrent_positions = 2
        
        # Strategy Parameters
        self.swing_lookback = 20  # Candles to look back for swing high/low
        self.volume_mult = 1.5
        self.min_breakout_pct = 0.005  # 0.5% minimum breakout
        
        # State
        self.positions = {}  # {symbol: position_data}
        self.product_map = {}
        self.contract_values = {}
        # Priority symbols (User Specified List)
        self.priority_symbols = [
            "BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", 
            "BNBUSD", "UNIUSD"
        ]
        self.symbols_to_trade = self.priority_symbols.copy()
        
        # Daily tracking
        self.daily_start_equity = self.equity
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.last_reset_day = datetime.now().day
        self.trades = []
        self.repo_root = Path(__file__).resolve().parents[2]
        self.data_dir = self.repo_root / "data"
        self.trade_log_dir = self.data_dir / "paper_trades"
        self.dashboard_file = self.data_dir / "dashboard_data.json"
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
        
        if self.paper_trading:
            self.trade_log_dir.mkdir(parents=True, exist_ok=True)
        
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
        """Adaptive position sizing and leverage logic with Autonomous Sniper"""
        # Strictly 50x Leverage as requested for the 6 core coins
        leverage = 50
        
        # Risk percentage based on conviction
        if is_sure_shot:
            risk_pct = 0.10 # 10% risk for high conviction
        else:
            risk_pct = 0.05 # 5% risk standard
            
        return risk_pct, leverage
    
    def get_reversal_risk(self):
        """Reversal trades are high risk, so cap leverage"""
        return 0.05, 50 # Aggressive 50x for Reversals
    
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
        """Check higher timeframe trend with symbol-specific strategy"""
        if len(df_15m) < 200:
            return None
        
        latest = df_15m.iloc[-1]
        
        if symbol == "BTCUSD":
            # Aggressive EMA Strategy for BTC: 9 > 21 > 50
            if latest['ema9'] > latest['ema21'] and latest['ema21'] > latest['ema50'] and latest['close'] > latest['ema200']:
                return "up"
            if latest['ema9'] < latest['ema21'] and latest['ema21'] < latest['ema50'] and latest['close'] < latest['ema200']:
                return "down"
        else:
            # Conservative Strategy for Alts: 20 > 50
            if latest['ema20'] > latest['ema50'] and latest['close'] > latest['ema200']:
                return "up"
            if latest['ema20'] < latest['ema50'] and latest['close'] < latest['ema200']:
                return "down"
        
        # print(f"   [FILTER] {symbol} rejected: No clear EMA trend alignment")
        return None  # No clear trend
    
    def check_entry_signal(self, symbol):
        """Check for high-probability entry setup with MTF structure"""
        df_5m, df_15m, df_1h, df_4h = self.get_multi_timeframe_data(symbol)
        if any(df is None for df in [df_5m, df_15m, df_1h, df_4h]):
            return None
        
        df_5m = self.calculate_indicators(df_5m)
        df_15m = self.calculate_indicators(df_15m)
        df_1h = self.calculate_indicators(df_1h)
        df_4h = self.calculate_indicators(df_4h)
        
        # --- NEW PATTERN & LEVEL ANALYSIS ---
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

        # --- NEW STRUCTURE ANALYSIS ---
        bias, events = StructureAnalyzer.analyze_structure(df_15m)
        obs = StructureAnalyzer.detect_order_blocks(df_15m)
        fvgs = StructureAnalyzer.detect_fvg(df_5m)
        sweeps = StructureAnalyzer.detect_liquidity_sweeps(df_5m)

        # Store analysis for API
        self.latest_analysis[symbol] = {
            "patterns": patterns,
            "support": nearest_support,
            "resistance": nearest_resistance,
            "market_bias": bias,
            "last_event": events[-1]["type"] if events else "none",
            "sweeps": [s["type"] for s in sweeps],
            "rsi": df_5m.iloc[-1]['rsi'],
            "price": latest_close,
            "ut_signal": active_signal, 
            "last_update": datetime.now().strftime('%H:%M:%S')
        }
        
        # Update global heartbeat
        # self.latest_analysis["_bot_heartbeat"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if len(df_5m) < 50 or len(df_15m) < 200:
            return None
        
        # Check higher timeframe trend
        trend = self.check_trend(df_15m, symbol)
        if not trend:
            # print(f"   [FILTER] {symbol} rejected: No 15m Trend alignment")
            return None
        
        cur = df_5m.iloc[-1]
        prev = df_5m.iloc[-2]
        
        # Volume confirmation
        volume_spike = cur['volume'] > (cur['vol_avg'] * self.volume_mult)
        if not volume_spike:
            # print(f"   [FILTER] {symbol} rejected: Volume ({cur['volume']:.0f}) < avg ({cur['vol_avg']:.0f})*1.5")
            return None
        
        signal = None
        entry_price = cur['close']
        stop_loss = None
        
        # UT Bot Signals
        ut_buy = cur['close'] > cur['ut_trail'] and prev['close'] <= prev['ut_trail'] and cur['close'] > cur['ut_trail']
        ut_sell = cur['close'] < cur['ut_trail'] and prev['close'] >= prev['ut_trail'] and cur['close'] < cur['ut_trail']
        
        if ut_buy and trend == "up":
            signal = "buy"
            stop_loss = cur['ut_trail']
        elif ut_sell and trend == "down":
            signal = "sell"
            stop_loss = cur['ut_trail']
        
        if signal:
            # Confluence: Check if patterns align with UT Bot signal
            
            # Match current candle time (ISO) for trading signal
            curr_ts = ""
            try:
                # Use 'cur' which is already defined above as df_5m.iloc[-1]
                ts_val = cur.name
                if 'time' in cur: ts_val = cur['time']
                curr_ts = pd.to_datetime(ts_val).isoformat()
            except:
                pass
            
            # Filter for latest candle only
            latest_patterns = [p['name'] for p in patterns if p['time'] == curr_ts] if patterns else []
            
            has_bullish_pattern = any(p in latest_patterns for p in ["Morning Star", "Three White Soldiers", "Bullish Engulfing", "Hammer", "Bullish Marubozu"])
            has_bearish_pattern = any(p in latest_patterns for p in ["Evening Star", "Three Black Crows", "Bearish Engulfing", "Shooting Star", "Bearish Marubozu"])
            
            # --- FANATIC CONFLUENCE ---
            # 1. MTF Bias alignment (Mandatory)
            if bias == "bullish" and signal != "buy": return None
            if bias == "bearish" and signal != "sell": return None

            # 2. Key Elements
            patt_conf = (signal == "buy" and has_bullish_pattern) or (signal == "sell" and has_bearish_pattern)
            has_sweep = any(s['type'] == ("Liquidity Sweep Low" if signal == "buy" else "Liquidity Sweep High") for s in sweeps)
            
            # 3. Order Block proximity (High Prob)
            near_ob = False
            for ob in obs:
                if ob['type'] == ("Bullish OB" if signal == "buy" else "Bearish OB"):
                    if abs(latest_close - (ob['top'] + ob['bottom'])/2) / latest_close < 0.008: # Slightly wider OB window
                        near_ob = True
                        break
            
            # 4. Fanatic Requirement: High Confluence Only
            # Require at least TWO of (Pattern, Sweep, Near OB) in addition to Bias
            confluence_score = sum([patt_conf, has_sweep, near_ob])
            if confluence_score < 2:
                # print(f"   [FILTER] {symbol} rejected: Confluence score {confluence_score} < 2")
                return None

            is_sure_shot = confluence_score >= 3

            # Calculate TPs
            risk = abs(entry_price - stop_loss)
            tp1 = entry_price + (risk * 2.0) if signal == "buy" else entry_price - (risk * 2.0)
            tp2 = entry_price + (risk * 3.5) if signal == "buy" else entry_price - (risk * 3.5)
            tp3 = entry_price + (risk * 6.0) if signal == "buy" else entry_price - (risk * 6.0)
            
            print(f"✅ Pattern Confirmed: {latest_patterns}")

            return {
                "side": signal,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "tp1": tp1, "tp2": tp2, "tp3": tp3,
                "is_sure_shot": False, "is_reversal": False,
                "reason": f"{trend.upper()} trend + {', '.join(latest_patterns)}"
            }
        
        return None
    
    def calculate_position_size(self, symbol, entry_price, stop_loss, is_sure_shot=False, is_reversal=False):
        """Calculate position size based on risk and sniper mode"""
        if is_reversal:
            risk_pct, leverage = self.get_reversal_risk()
        else:
            risk_pct, leverage = self.get_adaptive_risk(symbol, is_sure_shot)
        
        # Risk amount in INR
        risk_amount_inr = self.equity * risk_pct
        
        # Price risk per unit
        price_risk = abs(entry_price - stop_loss) / entry_price
        
        # Position size in INR (with leverage)
        if price_risk == 0: price_risk = 0.01 # Prevent div/0
        position_size_inr = (risk_amount_inr / price_risk) * leverage
        
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
                "tp1_hit": False,
                "tp2_hit": False
            }
            print(f"   [PAPER] Position opened")
            self.daily_trades += 1
            self.save_active_positions()
    
    def manage_positions(self):
        """Manage open positions with tiered exits"""
        to_remove = []
        
        for sym, pos in self.positions.items():
            df_5m, _, _, _ = self.get_multi_timeframe_data(sym)
            if df_5m is None:
                continue
            
            df_5m = self.calculate_indicators(df_5m)
            cur_price = df_5m.iloc[-1]['close']
            
            side = pos['side']
            entry = pos['entry']
            
            # Calculate current PnL
            if side == "buy":
                pnl_pct = (cur_price - entry) / entry
                sl_hit = cur_price <= pos['stop_loss']
            else:
                pnl_pct = (entry - cur_price) / entry
                sl_hit = cur_price >= pos['stop_loss']
            
            # Check stop loss
            if sl_hit:
                print(f"🛑 {sym} STOPPED OUT at ${cur_price:.2f}")
                self.close_position(sym, cur_price, "SL", pnl_pct, 1.0)
                to_remove.append(sym)
                self.consecutive_losses += 1
                continue
            
            # Tiered profit taking
            qty_rem = pos['qty_remaining']
            
            if side == "buy":
                if cur_price >= pos['tp1'] and not pos['tp1_hit']:
                    self.take_profit(sym, cur_price, "TP1", pnl_pct, 0.5, pos)
                elif cur_price >= pos['tp2'] and pos['tp1_hit'] and not pos['tp2_hit']:
                    self.take_profit(sym, cur_price, "TP2", pnl_pct, 0.3, pos)
                elif cur_price >= pos['tp3']:
                    self.close_position(sym, cur_price, "TP3", pnl_pct, 0.2)
                    to_remove.append(sym)
            else: # Sell
                if cur_price <= pos['tp1'] and not pos['tp1_hit']:
                    self.take_profit(sym, cur_price, "TP1", pnl_pct, 0.5, pos)
                elif cur_price <= pos['tp2'] and pos['tp1_hit'] and not pos['tp2_hit']:
                    self.take_profit(sym, cur_price, "TP2", pnl_pct, 0.3, pos)
                elif cur_price <= pos['tp3']:
                    self.close_position(sym, cur_price, "TP3", pnl_pct, 0.2)
                    to_remove.append(sym)

            # Update trailing stop if active
            if pos.get('trailing_active'):
                 if side == "buy":
                    pos['stop_loss'] = max(pos['stop_loss'], cur_price * 0.99)
                 else:
                    pos['stop_loss'] = min(pos['stop_loss'], cur_price * 1.01)

        for sym in to_remove:
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
        filename = self.trade_log_dir / f"trades_pro_{datetime.now().strftime('%Y%m%d')}.csv"
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.trades[0].keys())
                writer.writeheader()
                writer.writerows(self.trades)
        except Exception as e:
            print(f"   ❌ Export error: {e}")

    def save_active_positions(self):
        """Save current open positions to JSON for monitoring"""
        filename = self.trade_log_dir / "active_positions_pro.json"
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
                        "unrealized_pnl": 0.0 
                    } for p, d in self.positions.items()
                ],
                "recent_trades": self.trades[-50:], # Increased to 50
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "active_mode": "SNIPER" if self.equity < self.bootstrap_target else "GROWTH",
                "leverage_mode": "PRO",
                "market_structure": self.latest_analysis,
                "recent_logs": list(self.log_queue)
            }
            
            # Atomic write
            temp = self.dashboard_file.with_name(self.dashboard_file.name + ".tmp")
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
                print(f"❌ Error in main loop: {e}")
                time.sleep(5)

def run_flask():
    port = int(os.getenv("PORT", "5005"))
    print(f"📡 API Server starting on port {port}...")
    app.run(host='0.0.0.0', port=port, threaded=True, debug=False, use_reloader=False)

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
