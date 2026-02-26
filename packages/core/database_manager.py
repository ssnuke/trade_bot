import sqlite3
import os
import json
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path, bot_id: str = None):
        """
        Initialize DatabaseManager
        
        Args:
            db_path: Path to database file or directory for multi-bot setup
            bot_id: Optional bot identifier. If provided, creates bot-specific DB file.
                    If None, uses db_path directly as single shared DB.
        """
        self.bot_id = bot_id
        
        if bot_id:
            # Multi-bot mode: create separate DB per bot
            db_dir = os.path.dirname(db_path) if os.path.dirname(db_path) else '.'
            self.db_path = os.path.join(db_dir, f"bot_{bot_id}.db")
        else:
            # Single bot mode: use provided path
            self.db_path = db_path
            
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Create tables if they don't exist"""
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else '.', exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Trades table with bot_id support
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bot_id TEXT,
                    symbol TEXT,
                    side TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    pnl_inr REAL,
                    pnl_pct REAL,
                    roi REAL,
                    equity REAL,
                    reason TEXT,
                    exit_reason TEXT,
                    entry_time TEXT,
                    exit_time TEXT,
                    timestamp TEXT
                )
            ''')
            
            # Sessions table with bot_id support
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bot_id TEXT,
                    date TEXT,
                    start_equity REAL,
                    end_equity REAL,
                    net_pnl_inr REAL,
                    total_trades INTEGER,
                    wins INTEGER,
                    losses INTEGER,
                    win_rate REAL,
                    session_started TEXT,
                    session_ended TEXT,
                    UNIQUE(bot_id, date)
                )
            ''')
            conn.commit()

    def save_trade(self, trade_data, bot_id: str = None):
        """Save a single trade to the database"""
        bot_id = bot_id or self.bot_id or 'default'
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO trades (
                    bot_id, symbol, side, entry_price, exit_price, pnl_inr, 
                    pnl_pct, roi, equity, reason, exit_reason, 
                    entry_time, exit_time, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                bot_id,
                trade_data.get('symbol'),
                trade_data.get('side'),
                trade_data.get('entry_price'),
                trade_data.get('exit_price'),
                trade_data.get('pnl_inr'),
                trade_data.get('pnl_pct'),
                trade_data.get('roi'),
                trade_data.get('equity'),
                trade_data.get('reason'),
                trade_data.get('exit_reason'),
                trade_data.get('entry_time'),
                trade_data.get('exit_time'),
                trade_data.get('time')  # This is the timestamp field in existing dict
            ))
            conn.commit()

    def save_session(self, session_data, bot_id: str = None):
        """Save or update a session record"""
        bot_id = bot_id or self.bot_id or 'default'
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO sessions (
                    bot_id, date, start_equity, end_equity, net_pnl_inr, 
                    total_trades, wins, losses, win_rate, 
                    session_started, session_ended
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                bot_id,
                session_data.get('date'),
                session_data.get('start_equity'),
                session_data.get('end_equity'),
                session_data.get('net_pnl_inr'),
                session_data.get('total_trades'),
                session_data.get('wins'),
                session_data.get('losses'),
                session_data.get('win_rate'),
                session_data.get('session_started'),
                session_data.get('session_ended')
            ))
            conn.commit()

    def get_recent_trades(self, limit=100, bot_id: str = None):
        """Fetch recent trades for a specific bot or all bots"""
        bot_id = bot_id or self.bot_id
        
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if bot_id:
                cursor.execute(
                    'SELECT * FROM trades WHERE bot_id = ? ORDER BY id DESC LIMIT ?', 
                    (bot_id, limit)
                )
            else:
                cursor.execute('SELECT * FROM trades ORDER BY id DESC LIMIT ?', (limit,))
                
            rows = cursor.fetchall()
            trades = [dict(row) for row in rows]
            
            # Convert keys to match existing trade dict structure if needed
            for t in trades:
                if 'timestamp' in t:
                    t['time'] = t.pop('timestamp', None)
            
            # Return in ASC order (oldest first) to match bot's self.trades list
            trades.reverse()
            return trades

    def get_session_history(self, bot_id: str = None):
        """Fetch session history for a specific bot or all bots"""
        bot_id = bot_id or self.bot_id
        
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if bot_id:
                cursor.execute(
                    'SELECT * FROM sessions WHERE bot_id = ? ORDER BY date DESC',
                    (bot_id,)
                )
            else:
                cursor.execute('SELECT * FROM sessions ORDER BY date DESC')
                
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_all_trades(self, bot_id: str = None):
        """Fetch all trades"""
        return self.get_recent_trades(limit=999999, bot_id=bot_id)

    def get_bot_summary(self, bot_id: str = None):
        """Get summary stats for a bot"""
        bot_id = bot_id or self.bot_id
        
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if bot_id:
                cursor.execute(
                    '''SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN roi > 0 THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN roi <= 0 THEN 1 ELSE 0 END) as losses,
                        SUM(pnl_inr) as total_pnl,
                        MAX(equity) as max_equity,
                        MIN(equity) as min_equity
                    FROM trades WHERE bot_id = ?''',
                    (bot_id,)
                )
            else:
                cursor.execute(
                    '''SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN roi > 0 THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN roi <= 0 THEN 1 ELSE 0 END) as losses,
                        SUM(pnl_inr) as total_pnl,
                        MAX(equity) as max_equity,
                        MIN(equity) as min_equity
                    FROM trades'''
                )
            
            row = cursor.fetchone()
            return dict(row) if row else {}

