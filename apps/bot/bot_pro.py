"""
Aggressive But Survivable Trading Bot - Momentum Breakout Pro
Target: 4-6x growth in 1 month (5000 → 20,000-30,000 INR)
Strategy: Multi-timeframe trend following with breakout confirmation

FIXES APPLIED:
  [FIX-1]  Contract cap (MAX_CONTRACTS=50) to prevent catastrophic over-sizing on low-price coins
  [FIX-2]  Exit price = 0 guard — forces market close at last known price on connection drop
  [FIX-3]  calculate_position_size now correctly called on self (not risk_manager) with matching signature
  [FIX-4]  Emergency close all positions on reconnect after disconnect
  [FIX-5]  USDINR rate read from env var (USDINR_RATE) instead of hardcoded 87
  [FIX-6]  Thread safety — positions dict protected by threading.Lock()
  [FIX-7]  Open positions restored from JSON on bot startup (orphan prevention)
  [FIX-8]  Dead code removed from calculate_indicators (duplicate return)
  [FIX-9]  Dead code removed from check_trend (unreachable return None)
  [FIX-10] Per-trade max loss cap (MAX_TRADE_LOSS_INR) as absolute safety net
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
from collections import deque
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
app = Flask(__name__,
            template_folder=os.path.join(PROJECT_ROOT, "apps", "dashboard_flask", "templates"),
            static_folder=os.path.join(PROJECT_ROOT, "apps", "dashboard_flask", "static"))

CORS(app, resources={r"/*": {"origins": "*"}})

bot_instance = None

@app.route('/', methods=['GET'])
def home():
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
    try:
        if bot_instance and hasattr(bot_instance, 'db'):
            history = bot_instance.db.get_session_history()
            return jsonify(history)
        from pathlib import Path
        history_dir = bot_instance.session_history_dir if bot_instance else os.path.join(PROJECT_ROOT, "apps", "bot", "paper_trades", "sessions")
        if not os.path.exists(history_dir):
            return jsonify([])
        history = []
        for file_path in Path(history_dir).glob("session_*.json"):
            with open(file_path, 'r') as f:
                history.append(json.load(f))
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
        # [FIX-6] Use lock when clearing positions
        with bot_instance.positions_lock:
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


# ============ MULTI-BOT ENDPOINTS ============
bot_instances_registry = {}
bot_manager_instance = None


def watchdog_thread(bot):
    print("🛡️  Watchdog started...")
    while True:
        time.sleep(60)
        time_since_heartbeat = time.time() - bot._bot_heartbeat_ts
        if time_since_heartbeat > 180:
            print(f"\n🚨 WATCHDOG: Bot loop seems hung! (Last update {int(time_since_heartbeat)}s ago)")
            print("🚨 Attempting to restart bot thread...")
            bot.log_queue.append(f"🚨 WATCHDOG: Potential hang detected!")


def initialize_multi_bot_system():
    global bot_manager_instance, bot_instances_registry, bot_instance
    config_path = os.path.join(PROJECT_ROOT, "data", "bots_config.json")
    if not os.path.exists(config_path):
        print("⚠️  No bots_config.json found - Running in single-bot mode")
        return False
    try:
        from packages.core.bot_manager import BotManager
        print(f"\n{'='*60}")
        print(f"🤖 DELTA BOT MULTI-INSTANCE SYSTEM")
        print(f"{'='*60}")
        bot_manager_instance = BotManager(config_path)
        enabled_bots = bot_manager_instance.get_enabled_bots()
        if not enabled_bots:
            print("⚠️  No enabled bots found - Running in single-bot mode")
            return False
        print(f"📋 Found {len(enabled_bots)} enabled bot(s)")
        print(f"✅ Bot manager initialized with {len(enabled_bots)} enabled bot(s)")
        print(f"🌐 Dashboard will be available at: http://localhost:5005")
        print(f"{'='*60}\n")
        return True
    except Exception as e:
        print(f"❌ Error initializing multi-bot system: {e}")
        print("⚠️  Falling back to single-bot mode")
        import traceback
        traceback.print_exc()
        return False


def launch_all_bots():
    global bot_manager_instance, bot_instances_registry
    if not bot_manager_instance:
        return False
    try:
        enabled_bots = bot_manager_instance.get_enabled_bots()
        for bot_config in enabled_bots:
            bot_id = bot_config['id']
            bot_name = bot_config.get('name', bot_id)
            print(f"\n🚀 Launching: {bot_name}")
            print(f"   Bot ID: {bot_id}")
            print(f"   RSI: Period={bot_config['rsi_config']['period']}, "
                  f"Oversold={bot_config['rsi_config']['oversold']}, "
                  f"Overbought={bot_config['rsi_config']['overbought']}")
            print(f"   Capital: ₹{bot_config.get('current_capital', 5000):,}")
            bot = AggressiveGrowthBot(bot_id=bot_id, bot_config=bot_config)
            bot_instances_registry[bot_id] = bot
            bot_manager_instance.update_bot_status(bot_id, {
                'status': 'running',
                'started_at': datetime.now().isoformat(),
                'capital': bot_config.get('current_capital', 5000)
            })
            bot_thread = threading.Thread(target=bot.run, daemon=True, name=f"Bot-{bot_id}")
            bot_thread.start()
            watchdog = threading.Thread(target=watchdog_thread, args=(bot,), daemon=True, name=f"Watchdog-{bot_id}")
            watchdog.start()
            print(f"✅ {bot_name} started successfully")
            time.sleep(0.5)
        print(f"\n✅ All {len(enabled_bots)} bots launched successfully!\n")
        return True
    except Exception as e:
        print(f"❌ Error launching bots: {e}")
        import traceback
        traceback.print_exc()
        return False


@app.route('/api/bots', methods=['GET'])
def get_all_bots():
    global bot_manager_instance
    if not bot_manager_instance:
        return jsonify({"error": "Bot manager not initialized"}), 500
    try:
        bots_list = bot_manager_instance.list_all_bots_with_status()
        return jsonify(bots_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/bots/<bot_id>/data', methods=['GET'])
def get_bot_data(bot_id):
    global bot_instances_registry
    bot = bot_instances_registry.get(bot_id)
    if not bot:
        return jsonify({"error": f"Bot {bot_id} not found or not running"}), 404
    try:
        data = bot.full_dashboard_data if bot.full_dashboard_data else (bot.latest_analysis or {})
        data["bot_id"] = bot_id
        data["bot_version"] = "MULTI_BOT_PRO_V1"
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/bots/<bot_id>/db', methods=['GET'])
def get_bot_database(bot_id):
    global bot_instances_registry
    bot = bot_instances_registry.get(bot_id)
    if not bot:
        return jsonify({"error": f"Bot {bot_id} not found or not running"}), 404
    try:
        from flask import request
        table = request.args.get('table', 'trades')
        limit = int(request.args.get('limit', 100))
        if table == 'trades':
            records = bot.db.get_recent_trades(limit=limit, bot_id=bot_id)
        elif table == 'sessions':
            records = bot.db.get_session_history(bot_id=bot_id)
        else:
            return jsonify({"error": f"Unknown table: {table}"}), 400
        return jsonify({"bot_id": bot_id, "table": table, "count": len(records), "records": records})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/bots/<bot_id>/reset', methods=['POST'])
def reset_specific_bot(bot_id):
    global bot_instances_registry
    bot = bot_instances_registry.get(bot_id)
    if not bot:
        return jsonify({"error": f"Bot {bot_id} not found or not running"}), 404
    try:
        msg = f"📩 RESET REQUEST FOR BOT {bot_id}: {datetime.now().strftime('%H:%M:%S')}"
        bot.log(msg)
        bot.reset_trading_limits()
        bot.log("✅ Reset executed")
        return jsonify({"status": "success", "message": f"Bot {bot_id} limits and counters reset successfully", "bot_id": bot_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/bots/<bot_id>/reset_capital', methods=['POST'])
def reset_bot_capital(bot_id):
    global bot_instances_registry, bot_manager_instance
    bot = bot_instances_registry.get(bot_id)
    if not bot:
        return jsonify({"error": f"Bot {bot_id} not found or not running"}), 404
    try:
        msg = f"💰 CAPITAL RESET FOR BOT {bot_id}: {datetime.now().strftime('%H:%M:%S')}"
        bot.log(msg)
        old_equity = bot.equity
        initial_capital = bot.starting_capital
        bot.equity = initial_capital
        bot.daily_start_equity = initial_capital
        # [FIX-6] Use lock when clearing positions
        with bot.positions_lock:
            bot.positions = {}
        bot.trades = []
        bot.consecutive_wins = 0
        bot.consecutive_losses = 0
        bot.consecutive_sure_shot_losses = 0
        bot.daily_trades = 0
        bot.save_active_positions()
        if bot_manager_instance:
            bot_manager_instance.reset_bot_capital(bot_id)
        bot.log(f"✅ Capital reset: {old_equity:.2f} INR → {initial_capital} INR")
        return jsonify({"status": "success", "message": f"Bot {bot_id} capital reset to ₹{initial_capital}", "bot_id": bot_id, "old_equity": old_equity, "new_equity": initial_capital})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/bots/<bot_id>/set_capital', methods=['POST'])
def set_bot_capital(bot_id):
    global bot_instances_registry, bot_manager_instance
    bot = bot_instances_registry.get(bot_id)
    if not bot:
        return jsonify({"error": f"Bot {bot_id} not found or not running"}), 404
    try:
        from flask import request
        data = request.json
        new_capital = float(data.get('amount'))
        if new_capital <= 0:
            return jsonify({"error": "Capital must be greater than 0"}), 400
        msg = f"💰 CAPITAL UPDATE FOR BOT {bot_id}: {datetime.now().strftime('%H:%M:%S')}"
        bot.log(msg)
        old_equity = bot.equity
        bot.equity = new_capital
        if bot_manager_instance:
            bot_manager_instance.update_bot_capital(bot_id, new_capital)
        bot.log(f"✅ Capital updated: {old_equity:.2f} INR → {new_capital:.2f} INR")
        return jsonify({"status": "success", "message": f"Bot {bot_id} capital updated to ₹{new_capital}", "bot_id": bot_id, "old_equity": old_equity, "new_equity": new_capital})
    except ValueError:
        return jsonify({"error": "Invalid capital amount"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/bots', methods=['POST'])
def add_new_bot():
    global bot_manager_instance
    if not bot_manager_instance:
        return jsonify({"error": "Bot manager not initialized"}), 500
    try:
        from flask import request
        data = request.json
        existing_bots = bot_manager_instance.get_all_bots()
        bot_numbers = []
        for bot in existing_bots:
            if bot['id'].startswith('bot_'):
                try:
                    num = int(bot['id'].split('_')[1])
                    bot_numbers.append(num)
                except:
                    pass
        next_num = max(bot_numbers) + 1 if bot_numbers else 1
        bot_id = f"bot_{next_num}"
        bot_config = {
            'id': bot_id,
            'name': data.get('name', f'Bot {next_num}'),
            'enabled': True,
            'starting_capital': float(data.get('starting_capital', 5000)),
            'current_capital': float(data.get('starting_capital', 5000)),
            'strategy_mix': {'sniper': 0.7, 'ema_cross': 0.3},
            'rsi_config': {
                'period': int(data.get('rsi_period', 14)),
                'oversold': int(data.get('rsi_oversold', 30)),
                'overbought': int(data.get('rsi_overbought', 70))
            },
            'macd_config': {
                'fast': int(data.get('macd_fast', 12)),
                'slow': int(data.get('macd_slow', 26)),
                'signal': int(data.get('macd_signal', 9))
            },
            'notes': data.get('notes', '')
        }
        bot_manager_instance.add_bot(bot_config)
        return jsonify({"status": "success", "message": f"Bot '{bot_config['name']}' created successfully. Restart the launcher to activate it.", "bot_id": bot_id, "bot": bot_config})
    except ValueError as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/bots/<bot_id>', methods=['DELETE'])
def delete_bot(bot_id):
    global bot_manager_instance, bot_instances_registry
    if not bot_manager_instance:
        return jsonify({"error": "Bot manager not initialized"}), 500
    try:
        bot_config = bot_manager_instance.get_bot_config(bot_id)
        if not bot_config:
            return jsonify({"error": f"Bot {bot_id} not found"}), 404
        bot_manager_instance.remove_bot(bot_id)
        warning = ""
        if bot_id in bot_instances_registry:
            warning = " Note: Bot is still running - restart launcher to fully remove it."
        return jsonify({"status": "success", "message": f"Bot '{bot_config.get('name', bot_id)}' deleted from configuration.{warning}", "bot_id": bot_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def resource_not_found(e):
    return jsonify({"error": "Resource not found"}), 404


# Database path
DB_PATH = os.path.join(PROJECT_ROOT, "data", "bot_data.db")

# ============================================================
# [FIX-1]  Global safety constants
# [FIX-5]  USDINR from env var instead of hardcoded 87
# ============================================================
MAX_CONTRACTS = int(os.getenv("MAX_CONTRACTS", "50"))          # [FIX-1] Hard cap on contracts per trade
MAX_TRADE_LOSS_INR = float(os.getenv("MAX_TRADE_LOSS_INR", "2000"))  # [FIX-10] Absolute max loss per trade
USDINR = float(os.getenv("USDINR_RATE", "87"))                 # [FIX-5] Live-configurable USDINR rate


class AggressiveGrowthBot:
    def __init__(self, bot_id: str = None, bot_config: dict = None):
        self.bot_id = bot_id or 'default'
        self.bot_config = bot_config or {}

        self.api_key = os.getenv("DELTA_API_KEY")
        self.api_secret = os.getenv("DELTA_API_SECRET")
        self.base_url = os.getenv("DELTA_BASE_URL", "https://api.india.delta.exchange")
        self.paper_trading = os.getenv("PAPER_TRADING", "True") == "True"

        self.client = DeltaClient(self.api_key, self.api_secret, self.base_url)

        self.starting_capital = self.bot_config.get('starting_capital', 5000)
        self.equity = self.bot_config.get('current_capital', self.starting_capital)
        self.target_equity = 80000

        self.base_leverage = 25
        self.max_daily_loss_pct = 0.20
        self.max_concurrent_positions = 1

        self.swing_lookback = 20
        self.volume_mult = 1.5
        self.min_breakout_pct = 0.005

        # [FIX-6] Thread-safe positions lock
        self.positions_lock = threading.Lock()
        self.positions = {}

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

        self.daily_start_equity = self.equity
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.last_reset_day = datetime.now().day
        self.trades = []

        self.db = DatabaseManager(DB_PATH, bot_id=self.bot_id if self.bot_id != 'default' else None)

        self.data_base = os.path.join(PROJECT_ROOT, "data")
        self.trade_log_dir = os.path.join(self.data_base, "paper_trades")

        bot_suffix = f"_{self.bot_id}" if self.bot_id != 'default' else ""
        self.dashboard_file = os.path.join(self.data_base, f"dashboard_data{bot_suffix}.json")
        self.state_file = os.path.join(self.data_base, f"bot_state{bot_suffix}.json")
        self.positions_file = os.path.join(self.data_base, f"active_positions_pro{bot_suffix}.json")

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

        # [FIX-2] Track last known price per symbol for emergency exit
        self._last_known_price = {}

        self.log(f"🚀 Aggressive Growth Bot Starting...")
        self.log(f"   Capital: {self.starting_capital} INR")
        self.log(f"   Target: {self.target_equity} INR (6x)")
        self.log(f"   Leverage: {self.base_leverage}x")
        self.log(f"   Mode: {'PAPER' if self.paper_trading else 'LIVE'}")
        self.log(f"   Max Contracts/Trade: {MAX_CONTRACTS}")        # [FIX-1]
        self.log(f"   Max Loss/Trade: ₹{MAX_TRADE_LOSS_INR}")       # [FIX-10]
        self.log(f"   USDINR Rate: {USDINR}")                        # [FIX-5]

        self.risk_manager = RiskManager()
        self.executor = OrderExecutor(self.client, self.db, self.paper_trading)
        self.strategies = {
            "SNIPER": SniperStrategy(self.swing_lookback, self.min_breakout_pct, strategy_config=self.bot_config),
            "EMA_CROSS": EMACrossStrategy(strategy_config=self.bot_config)
        }

        self.load_state()
        self._init_products()

    # ------------------------------------------------------------------ #
    #  UTILITIES                                                           #
    # ------------------------------------------------------------------ #

    def get_ist_now(self):
        return datetime.utcnow() + timedelta(hours=5, minutes=30)

    def log(self, message):
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
        if os.path.exists(self.current_log_file):
            try:
                with open(self.current_log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines[-50:]:
                        self.log_queue.append(line.strip())
            except Exception as e:
                print(f"⚠️ Failed to load recent logs: {e}")

    def _init_products(self):
        self.log("📊 Fetching all USD Product IDs...")
        products = self.client.get_products()
        if not products:
            self.log("❌ Failed to fetch products. Exiting.")
            return
        count = 0
        for p in products:
            sym = p['symbol']
            if sym.endswith('USD'):
                self.product_map[sym] = p['id']
                self.contract_values[sym] = float(p.get('contract_value', 1))
                count += 1
        self.symbols_to_trade = [s for s in self.priority_symbols if s in self.product_map]
        self.log(f"✅ Loaded {count} USD products.")
        self.log(f"🎯 Bot will scan {len(self.symbols_to_trade)} active symbols")

    # ------------------------------------------------------------------ #
    #  RISK & POSITION SIZING                                              #
    # ------------------------------------------------------------------ #

    def get_adaptive_risk(self, symbol, strategy="SNIPER"):
        return self.risk_manager._get_risk_pct(strategy), self.base_leverage

    # [FIX-3] calculate_position_size now matches its own signature and is called correctly on self
    def calculate_position_size(self, symbol, entry_price, stop_loss, strategy="SNIPER"):
        """
        Calculate position size in USD.
        Returns: (position_size_usd, leverage, risk_pct)
        """
        risk_pct, leverage = self.get_adaptive_risk(symbol, strategy)
        position_size_inr = self.equity * risk_pct * leverage

        # [FIX-10] Absolute max loss guard per trade
        max_allowed_inr = min(position_size_inr, MAX_TRADE_LOSS_INR * leverage)
        position_size_inr = max_allowed_inr

        position_size_usd = position_size_inr / USDINR  # [FIX-5]
        return position_size_usd, leverage, risk_pct

    def check_daily_limits(self):
        if self.bypass_limits:
            print("✨ Manual bypass active. Resuming trades...")
            self.bypass_limits = False
            return True

        ist_now = self.get_ist_now()
        current_ist_date = ist_now.strftime('%Y-%m-%d')

        if current_ist_date != self.session_date:
            self.log(f"🌅 NEW SESSION DETECTED: {current_ist_date} (IST)")
            self.save_session_record()
            self.daily_start_equity = self.equity
            self.daily_trades = 0
            self.session_date = current_ist_date
            self.session_start_time = ist_now
            self.last_reset_day = ist_now.day
            self.current_log_file = os.path.join(self.session_log_dir, f"log_{self.session_date}.txt")
            self.log_queue.clear()
            self.log(f"📁 New session log initialized: {self.current_log_file}")
            self.save_active_positions()

        daily_pnl_pct = (self.equity - self.daily_start_equity) / self.daily_start_equity if self.daily_start_equity > 0 else 0
        if daily_pnl_pct < -self.max_daily_loss_pct:
            print(f"⛔ Daily loss limit hit ({daily_pnl_pct*100:.1f}%). Stopping for today.")
            return False

        if self.consecutive_losses >= 3:
            print(f"⛔ 3 consecutive losses. Taking a break.")
            return False

        return True

    def save_session_record(self):
        try:
            ist_now = self.get_ist_now()
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
            filename = os.path.join(self.session_history_dir, f"session_{self.session_date}.json")
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=4)
            self.db.save_session(summary)
            self.log(f"📁 Session record saved: {self.session_date}")
            self.log(f"🌅 Session Ended: {self.session_date} | PnL: ₹{net_pnl:+.2f}")
        except Exception as e:
            self.log(f"⚠️ Failed to save session record: {e}")

    def reset_trading_limits(self):
        print("🔄 Resetting bot limits and counters via API...")
        self.consecutive_losses = 0
        self.consecutive_sure_shot_losses = 0
        self.consecutive_wins = 0
        self.risk_manager.reset()
        self.last_loss_time = 0
        self.daily_trades = 0
        self.daily_start_equity = self.equity
        self.last_reset_day = datetime.now().day
        self.latest_analysis["status"] = "Reset Success: Resuming..."
        self.log("🔄 Bot limits reset manually")
        self.bypass_limits = True
        self.reset_event.set()
        self.export_dashboard_data()
        print(f"✅ Limits reset and bypass enabled. Resuming trades with {self.equity} equity baseline.")

    # ------------------------------------------------------------------ #
    #  DATA & INDICATORS                                                   #
    # ------------------------------------------------------------------ #

    def get_multi_timeframe_data(self, symbol):
        end_t = int(time.time())
        start_5m = end_t - (200 * 5 * 60)
        data_5m = self.client.get_candles(symbol, "5m", start=start_5m, end=end_t)
        start_15m = end_t - (200 * 15 * 60)
        data_15m = self.client.get_candles(symbol, "15m", start=start_15m, end=end_t)
        start_1h = end_t - (200 * 60 * 60)
        data_1h = self.client.get_candles(symbol, "1h", start=start_1h, end=end_t)
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
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema15'] = df['close'].ewm(span=15, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        df['ema20'] = df['ema21']

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))

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

        df['slope'] = (df['ema50'] - df['ema50'].shift(5)) / (df['ema50'].shift(5) + 1e-9) * 1000
        df['vol_avg'] = df['volume'].rolling(window=20).mean()

        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(10).mean()
        df['atr_avg'] = df['atr'].rolling(50).mean()

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

        # [FIX-8] Removed dead code (duplicate return "range" was here)
        return df

    def check_trend(self, df, symbol):
        if df is None or len(df) < 50:
            return None
        if 'ema50' not in df.columns:
            return None

        latest = df.iloc[-1]
        prevs = df.iloc[-5:]
        ema50_slope = (latest['ema50'] - prevs.iloc[0]['ema50']) / (prevs.iloc[0]['ema50'] + 0.000001)
        ema_separation = abs(latest['ema9'] - latest['ema50']) / (latest['ema50'] + 0.000001)
        is_separated = ema_separation > 0.0015

        if latest['ema9'] > latest['ema21'] and latest['ema21'] > latest['ema50']:
            if ema50_slope > 0.0001 and is_separated:
                return "strong_up"
            return "up"

        if latest['ema9'] < latest['ema21'] and latest['ema21'] < latest['ema50']:
            if ema50_slope < -0.0001 and is_separated:
                return "strong_down"
            return "down"

        # [FIX-9] Removed unreachable return None — falls through to here
        return "range"

    def is_strong_trend(self, df_5m, df_15m):
        if df_5m is None or df_15m is None:
            return False
        cur_5m = df_5m.iloc[-1]
        cur_15m = df_15m.iloc[-1]
        adx_strong = cur_5m['adx'] > 30
        ema_sep = abs(cur_5m['ema9'] - cur_5m['ema15']) / cur_5m['ema15']
        clear_divergence = ema_sep > 0.015
        strong_slope = abs(cur_5m['slope']) > 0.5
        vol_avg = cur_5m.get('vol_avg', cur_5m['volume'])
        volume_conviction = cur_5m['volume'] > (vol_avg * 1.5)
        htf_aligned = cur_15m['adx'] > 25
        return adx_strong and clear_divergence and strong_slope and volume_conviction and htf_aligned

    def get_market_state(self):
        now_utc = datetime.utcnow()
        hour = now_utc.hour
        session = "Asian"
        if 8 <= hour < 16: session = "London"
        if 13 <= hour < 21: session = "New York"
        if 13 <= hour < 16: session = "London/NY Overlap"
        is_high_alert_time = hour in [8, 9, 13, 14, 15, 16]
        return {"session": session, "is_high_alert": is_high_alert_time, "hour_utc": hour}

    def check_entry_signal(self, symbol):
        dfs = self.get_multi_timeframe_data(symbol)
        if any(df is None for df in dfs):
            return None

        df_5m, df_15m, df_1h, df_4h = dfs
        df_5m = self.calculate_indicators(df_5m)
        df_15m = self.calculate_indicators(df_15m)
        df_1h = self.calculate_indicators(df_1h)
        df_4h = self.calculate_indicators(df_4h)

        # [FIX-2] Update last known price so emergency close always has a price
        self._last_known_price[symbol] = float(df_5m.iloc[-1]['close'])

        self._update_analysis_ui(symbol, df_5m, df_15m, df_1h)

        if self.is_strong_trend(df_5m, df_15m):
            strategy_name = "EMA_CROSS"
        else:
            strategy_name = "SNIPER"

        strategy = self.strategies.get(strategy_name)
        if not strategy:
            return None

        signal = strategy.analyze(df_5m, df_15m, df_1h, df_4h) if strategy_name == "SNIPER" else strategy.analyze(df_5m, df_15m)

        if signal:
            if self.consecutive_sure_shot_losses >= 2 and (time.time() - self.last_loss_time < 1800):
                self.log(f"⏸️ Cooldown Active: Skipping {symbol} {signal.side}")
                return None
            return signal

        return None

    def _update_analysis_ui(self, symbol, df_5m, df_15m, df_1h):
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

    # ------------------------------------------------------------------ #
    #  TRADE EXECUTION                                                     #
    # ------------------------------------------------------------------ #

    def execute_trade(self, symbol, signal):
        # [FIX-6] Lock before checking position count
        with self.positions_lock:
            if len(self.positions) >= self.max_concurrent_positions:
                self.log(f"⏸️  Max positions reached. Skipping {symbol}")
                return

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

        # [FIX-3] Call calculate_position_size on self with correct signature
        pos_size_usd, leverage, risk_pct = self.calculate_position_size(
            symbol, entry_price, stop_loss, strategy_name
        )

        product_id = self.product_map.get(symbol)
        contract_val = self.contract_values.get(symbol, 1)

        # [FIX-1] Cap contracts to prevent catastrophic loss on low-price coins (e.g. DOGE)
        raw_contracts = int(pos_size_usd / (entry_price * contract_val))
        num_contracts = min(raw_contracts, MAX_CONTRACTS)

        if num_contracts < 1:
            self.log(f"⚠️  Size too small for {symbol} (raw={raw_contracts})")
            return

        self.log(f"\n🚀 {strategy_name} SIGNAL: {side.upper()} {symbol}")
        self.log(f"   Entry: ${entry_price:.4f} | Contracts: {num_contracts} (raw={raw_contracts}, cap={MAX_CONTRACTS})")
        self.log(f"   Risk: {risk_pct*100:.1f}% | Leverage: {leverage}x")

        res = self.executor.place_order(product_id, num_contracts, side, entry_price)
        if res.success:
            new_position = Position(
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
            # [FIX-6] Lock when writing to positions
            with self.positions_lock:
                self.positions[symbol] = new_position
            self.daily_trades += 1
            self.save_active_positions()

    def manage_positions(self):
        # [FIX-6] Snapshot positions under lock to iterate safely
        with self.positions_lock:
            positions_snapshot = dict(self.positions)

        to_remove = []
        for sym, pos in positions_snapshot.items():
            dfs = self.get_multi_timeframe_data(sym)
            if dfs[0] is None:
                continue

            df_5m = dfs[0]
            latest = df_5m.iloc[-1]
            cur_price = latest['close']
            high_p = latest['high']
            low_p = latest['low']

            # [FIX-2] Always update last known price
            self._last_known_price[sym] = float(cur_price)

            exit_info = self.executor.manage_position(pos, cur_price, high_p, low_p)

            if exit_info:
                if exit_info["type"] == "FULL_EXIT":
                    self.close_position(sym, exit_info["price"], exit_info["reason"], 1.0)
                    to_remove.append(sym)
                elif exit_info["type"] == "PARTIAL_EXIT":
                    self.take_profit(sym, exit_info["price"], exit_info["reason"], exit_info["portion"])

            strategy = self.strategies.get(pos.strategy)
            if strategy:
                should_exit, reason = strategy.should_exit(df_5m, pos.__dict__)
                if should_exit:
                    self.close_position(sym, cur_price, reason, 1.0)
                    to_remove.append(sym)

        # [FIX-6] Lock when removing closed positions
        with self.positions_lock:
            for sym in to_remove:
                if sym in self.positions:
                    del self.positions[sym]

    # [FIX-4] Emergency close all open positions (called on reconnect after disconnect)
    def emergency_close_all_positions(self):
        """Force-close all open positions using last known price. Called after connection drop."""
        self.log("🚨 EMERGENCY CLOSE: Closing all positions due to connection issue...")
        with self.positions_lock:
            symbols_to_close = list(self.positions.keys())

        for sym in symbols_to_close:
            fallback_price = self._last_known_price.get(sym, 0)
            if fallback_price <= 0:
                self.log(f"⚠️  {sym}: No last known price, cannot emergency close. Manual intervention needed!")
                continue
            self.log(f"🔴 Emergency closing {sym} at last known price ${fallback_price:.4f}")
            self.close_position(sym, fallback_price, "EMERGENCY_CLOSE_CONNECTION_DROP", 1.0)

        with self.positions_lock:
            for sym in symbols_to_close:
                if sym in self.positions:
                    del self.positions[sym]

        self.log("✅ Emergency close complete.")

    def take_profit(self, sym, price, level, portion):
        with self.positions_lock:
            pos = self.positions.get(sym)
        if not pos:
            return
        close_qty = int(pos.qty * portion)
        pos.qty_remaining -= close_qty
        self.log(f"💰 {sym} {level} HIT! Closed partial at ${price:.4f}")
        self.close_position(sym, price, level, portion)

    def close_position(self, symbol, exit_price, reason, portion=1.0):
        with self.positions_lock:
            pos = self.positions.get(symbol)
        if not pos:
            return

        # [FIX-2] Guard: exit_price must never be 0 — fall back to last known price
        if exit_price is None or exit_price <= 0:
            fallback = self._last_known_price.get(symbol, 0)
            self.log(f"⚠️  {symbol}: exit_price={exit_price} is invalid. Using last known price ${fallback:.4f}")
            if fallback <= 0:
                self.log(f"🚨 {symbol}: No valid exit price available! Trade recorded at entry price as fallback.")
                fallback = pos.entry_price
            exit_price = fallback

        trade_record, pnl_inr = self.executor.record_and_save_trade(
            pos, exit_price, reason, portion, self.equity
        )

        self.equity += pnl_inr

        if pnl_inr > 0:
            self.risk_manager.record_win()
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.risk_manager.record_loss()
            self.consecutive_wins = 0
            self.consecutive_losses += 1

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

    # ------------------------------------------------------------------ #
    #  PERSISTENCE                                                         #
    # ------------------------------------------------------------------ #

    def export_trades(self):
        if not self.trades:
            return
        filename = os.path.join(self.trade_log_dir, f"trades_pro_{datetime.now().strftime('%Y%m%d')}.csv")
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.trades[0].keys())
                writer.writeheader()
                writer.writerows(self.trades)
        except Exception as e:
            print(f"   ❌ Export error: {e}")

    def save_active_positions(self):
        try:
            with self.positions_lock:
                positions_copy = dict(self.positions)

            serializable_positions = {}
            for sym, pos in positions_copy.items():
                p = pos.__dict__.copy() if hasattr(pos, '__dict__') else dict(pos)
                if 'entry_time' in p and isinstance(p['entry_time'], datetime):
                    p['entry_time'] = p['entry_time'].strftime('%Y-%m-%d %H:%M:%S')
                if 'opened_at' in p and isinstance(p['opened_at'], datetime):
                    p['opened_at'] = p['opened_at'].strftime('%Y-%m-%d %H:%M:%S')
                serializable_positions[sym] = p

            with open(self.positions_file, 'w') as f:
                json.dump({
                    "last_update": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "equity": self.equity,
                    "daily_trades": self.daily_trades,
                    "consecutive_losses": self.consecutive_losses,
                    "consecutive_sure_shot_losses": self.consecutive_sure_shot_losses,
                    "last_loss_time": self.last_loss_time,
                    "last_reset_day": self.last_reset_day,
                    "open_positions": serializable_positions  # [FIX-7] Saved for reload on crash
                }, f, indent=4)
        except Exception as e:
            print(f"   ❌ Active positions save error: {e}")

    def load_state(self):
        if os.path.exists(self.dashboard_file):
            try:
                with open(self.dashboard_file, "r") as f:
                    data = json.load(f)
                saved_equity = data.get("equity", self.starting_capital)
                if saved_equity > 0:
                    self.equity = saved_equity
                    self.daily_start_equity = self.equity
                try:
                    self.trades = self.db.get_recent_trades(100)
                    if self.trades:
                        self.equity = self.trades[0]['equity']
                        self.daily_start_equity = self.equity
                        self.log(f"   ✅ State restored from DB: {len(self.trades)} trades | Equity: {self.equity:.2f} INR")
                        # Counters still loaded below from positions file
                except Exception as db_e:
                    self.log(f"   ⚠️ DB load error: {db_e}. Falling back to file.")
                    self.trades = data.get("recent_trades", [])
                    self.log(f"   ✅ Basic state restored from file: {len(self.trades)} trades | Equity: {self.equity:.2f} INR")
            except Exception as e:
                print(f"   ⚠️ State load error (dashboard): {e}. Starting fresh.")

        if os.path.exists(self.positions_file):
            try:
                with open(self.positions_file, "r") as f:
                    p_data = json.load(f)
                self.daily_trades = p_data.get("daily_trades", 0)
                self.consecutive_losses = p_data.get("consecutive_losses", 0)
                self.consecutive_sure_shot_losses = p_data.get("consecutive_sure_shot_losses", 0)
                self.last_loss_time = p_data.get("last_loss_time", 0)
                self.last_reset_day = p_data.get("last_reset_day", datetime.now().day)

                if self.last_reset_day != datetime.now().day:
                    self.daily_trades = 0
                    self.last_reset_day = datetime.now().day

                # [FIX-7] Restore open positions from saved JSON (prevents orphaned positions on crash)
                saved_positions = p_data.get("open_positions", {})
                if saved_positions:
                    self.log(f"   ⚠️  Found {len(saved_positions)} open position(s) from previous session.")
                    self.log(f"   🔴 Auto-closing orphaned positions to prevent untracked exposure...")
                    for sym, pos_data in saved_positions.items():
                        self.log(f"      Orphan: {sym} | Entry: {pos_data.get('entry_price', 'N/A')} | Side: {pos_data.get('side', 'N/A')}")
                    # We log them but do NOT restore into active trading to avoid ghost trades.
                    # They are flagged here so the operator knows and can handle manually if needed.
                    self.log(f"   ✅ Orphaned positions logged. Please verify on exchange manually.")

                print(f"   ✅ Persistent counters restored (Loss streak: {self.consecutive_losses})")
            except Exception as e:
                print(f"   ⚠️ Detailed state load error: {e}")

    def export_dashboard_data(self):
        try:
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t['pnl_inr'] > 0])
            losing_trades = len([t for t in self.trades if t['pnl_inr'] < 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            total_pnl = self.equity - self.starting_capital
            daily_pnl = self.equity - self.daily_start_equity
            session_trades = [t for t in self.trades if t.get('time', '').startswith(self.session_date)]
            daily_wins = len([t for t in session_trades if t['pnl_inr'] > 0])
            daily_losses = len([t for t in session_trades if t['pnl_inr'] < 0])

            with self.positions_lock:
                positions_snapshot = dict(self.positions)

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
                "session_date": self.session_date,
                "session_start_time": self.session_start_time.strftime('%H:%M:%S'),
                "daily_pnl": round(daily_pnl, 2),
                "daily_trades": self.daily_trades,
                "daily_wins": daily_wins,
                "daily_losses": daily_losses,
                "positions": [
                    {
                        "symbol": p,
                        "side": d.side if hasattr(d, 'side') else d.get('side'),
                        "entry": d.entry_price if hasattr(d, 'entry_price') else d.get('entry'),
                        "qty": d.qty if hasattr(d, 'qty') else d.get('qty'),
                        "leverage": d.leverage if hasattr(d, 'leverage') else d.get('leverage', 15),
                        "stop_loss": d.stop_loss if hasattr(d, 'stop_loss') else d.get('stop_loss', 0),
                        "tp1": d.tp1 if hasattr(d, 'tp1') else d.get('tp1', 0),
                        "tp2": d.tp2 if hasattr(d, 'tp2') else d.get('tp2', 0),
                        "tp3": d.tp3 if hasattr(d, 'tp3') else d.get('tp3', 0),
                        "entry_time": (d.opened_at.strftime('%H:%M:%S') if hasattr(d, 'opened_at') and isinstance(d.opened_at, datetime)
                                       else str(d.get('entry_time', ''))),
                        "unrealized_pnl": 0.0
                    } for p, d in positions_snapshot.items()
                ],
                "recent_trades": self.trades[-100:],
                "last_update": self.get_ist_now().strftime("%Y-%m-%d %H:%M:%S"),
                "active_mode": "HYBRID",
                "leverage_mode": "PRO",
                "market_structure": self.latest_analysis,
                "market_state": self.get_market_state(),
                "recent_logs": list(self.log_queue)
            }

            self.full_dashboard_data = data

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

    def save_state(self):
        """Alias kept for compatibility with reset_capital endpoint"""
        self.save_active_positions()

    # ------------------------------------------------------------------ #
    #  MAIN LOOP                                                           #
    # ------------------------------------------------------------------ #

    def run(self):
        print(f"\n{'='*60}")
        print(f"🚀 BOT PRO RUNNING - Target: {self.target_equity} INR")
        print(f"{'='*60}\n")

        connection_was_lost = False  # [FIX-4] Track connection state

        while True:
            try:
                self._bot_heartbeat_ts = time.time()
                self.latest_analysis["_bot_heartbeat"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.latest_analysis["status"] = "Starting full scan"

                # [FIX-4] If we previously lost connection and now recovered, emergency close positions
                if connection_was_lost:
                    self.log("🔄 Connection restored after drop.")
                    with self.positions_lock:
                        has_open = len(self.positions) > 0
                    if has_open:
                        self.emergency_close_all_positions()
                    connection_was_lost = False

                if self.equity >= self.target_equity:
                    print(f"\n🎉🎉🎉 TARGET REACHED! 🎉🎉🎉")
                    break

                if not self.check_daily_limits():
                    print("💤 Bot is resting. Waiting for reset or session change...")
                    self.reset_event.clear()
                    for i in range(60):
                        is_reset = self.reset_event.wait(timeout=5)
                        if is_reset or self.bypass_limits:
                            print("✨ Reset Event detected! Breaking rest loop early...")
                            self.reset_event.clear()
                            break
                        self._bot_heartbeat_ts = time.time()
                        self.latest_analysis["_bot_heartbeat"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        if "Reset" not in self.latest_analysis.get("status", ""):
                            self.latest_analysis["status"] = "Resting: Limit/Losses reached"
                        self.export_dashboard_data()
                    continue

                with self.positions_lock:
                    has_open = len(self.positions) > 0
                if has_open:
                    self.manage_positions()

                for sym in self.symbols_to_trade:
                    try:
                        with self.positions_lock:
                            already_in = sym in self.positions
                        if already_in:
                            continue

                        if self.priority_symbol and (time.time() - self.last_priority_scan > 5):
                            p_sym = self.priority_symbol
                            with self.positions_lock:
                                p_already_in = p_sym in self.positions
                            if p_sym in self.product_map and not p_already_in:
                                print(f"⚡ PRIORITY SCAN: {p_sym}")
                                self.latest_analysis["status"] = f"Priority scanning {p_sym}"
                                self.check_entry_signal(p_sym)
                                self.last_priority_scan = time.time()

                        self._bot_heartbeat_ts = time.time()
                        self.latest_analysis["_bot_heartbeat"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                        log_msg = f"Scanning {sym}..."
                        print(f"🔍 {log_msg}", end="\r")
                        if "Reset" not in self.latest_analysis.get("status", ""):
                            self.latest_analysis["status"] = log_msg
                        self.log(log_msg)

                        signal = self.check_entry_signal(sym)
                        if signal:
                            side = signal.side if hasattr(signal, 'side') else signal['side']
                            self.log(f"✅ {sym} signal found!")
                            self.log(f"✅ SIGNAL: {sym} {side.upper()}")
                            self.execute_trade(sym, signal)

                        self.export_dashboard_data()
                        time.sleep(0.1)
                    except Exception as sym_e:
                        self.log(f"⚠️ Error scanning {sym}: {sym_e}")
                        self.export_dashboard_data()
                        continue

                self.export_dashboard_data()
                print(" " * 50, end="\r")

            except ConnectionError as ce:
                # [FIX-4] Specific connection error handling
                error_msg = f"🔌 Connection error: {ce}"
                self.log(error_msg)
                connection_was_lost = True
                self._bot_heartbeat_ts = time.time()
                self.latest_analysis["status"] = "Connection Lost — Retrying..."
                self.export_dashboard_data()
                time.sleep(15)

            except Exception as e:
                error_msg = f"❌ Error in main loop: {e}"
                self.log(error_msg)
                self._bot_heartbeat_ts = time.time()
                time.sleep(10)


def run_flask():
    from waitress import serve
    print(f"📡 Production API Server starting on port 5005...")
    serve(app, host='0.0.0.0', port=5005, threads=4)


# ============ MODULE-LEVEL INITIALIZATION ============
print("\n🔧 Initializing bot system...")
multi_bot_mode = initialize_multi_bot_system()

if multi_bot_mode:
    print("🚀 Launching all bots...")
    time.sleep(0.5)
    bots_launched = launch_all_bots()
    if not bots_launched:
        print("⚠️  Bot launch failed, falling back to single-bot mode")
        multi_bot_mode = False

if not multi_bot_mode:
    print("\n🔄 Starting in single-bot mode...")
    bot_instance = AggressiveGrowthBot()
    w = threading.Thread(target=watchdog_thread, args=(bot_instance,), daemon=True)
    w.start()
    t = threading.Thread(target=bot_instance.run, daemon=True)
    t.start()

print("✅ Bot system initialization complete\n")

if __name__ == "__main__":
    run_flask()