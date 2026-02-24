import sqlite3
import os
import json
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Create tables if they don't exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            
            # Sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT UNIQUE,
                    start_equity REAL,
                    end_equity REAL,
                    net_pnl_inr REAL,
                    total_trades INTEGER,
                    wins INTEGER,
                    losses INTEGER,
                    win_rate REAL,
                    session_started TEXT,
                    session_ended TEXT
                )
            ''')
            conn.commit()

    def save_trade(self, trade_data):
        """Save a single trade to the database"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO trades (
                    symbol, side, entry_price, exit_price, pnl_inr, 
                    pnl_pct, roi, equity, reason, exit_reason, 
                    entry_time, exit_time, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
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

    def save_session(self, session_data):
        """Save or update a session record"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO sessions (
                    date, start_equity, end_equity, net_pnl_inr, 
                    total_trades, wins, losses, win_rate, 
                    session_started, session_ended
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
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

    def get_recent_trades(self, limit=100):
        """Fetch recent trades"""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM trades ORDER BY id DESC LIMIT ?', (limit,))
            rows = cursor.fetchall()
            # Convert to list of dicts for API compatibility
            trades = [dict(row) for row in rows]
            # Convert keys to match existing trade dict structure if needed
            for t in trades:
                t['time'] = t.pop('timestamp')
            
            # Return in ASC order (oldest first) to match bot's self.trades list
            trades.reverse()
            return trades

    def get_session_history(self):
        """Fetch session history"""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM sessions ORDER BY date DESC')
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
