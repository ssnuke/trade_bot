import os
import sys
import sqlite3
from datetime import datetime

# Add direct paths to sys.path
PROJECT_ROOT = "/Users/snehith/Desktop/Development/delta_bot"
sys.path.append(PROJECT_ROOT)

from packages.core.database_manager import DatabaseManager

def test_database():
    print("Testing DatabaseManager Integration...")
    db_path = os.path.join(PROJECT_ROOT, "data", "test_bot_data.db")
    
    # Remove old test DB if exists
    if os.path.exists(db_path):
        os.remove(db_path)
        
    try:
        db = DatabaseManager(db_path)
        
        # Test 1: Schema creation
        if os.path.exists(db_path):
            print(f"✅ Database file created at {db_path}")
        else:
            print("❌ Database file NOT created.")
            return

        # Test 2: Save trade
        trade_data = {
            'symbol': 'BTCUSD',
            'side': 'buy',
            'entry_price': 50000.0,
            'exit_price': 51000.0,
            'pnl_inr': 8700.0,
            'pnl_pct': 2.0,
            'roi': 50.0,
            'equity': 13700.0,
            'reason': 'Structural Breakout',
            'exit_reason': 'TP1',
            'entry_time': '12:00:00',
            'exit_time': '12:05:00',
            'time': '2026-02-24 12:05:00'
        }
        db.save_trade(trade_data)
        print("✅ Trade saved successfully.")

        # Test 3: Retrieve recent trades
        recent = db.get_recent_trades(1)
        if len(recent) > 0:
            print(f"✅ Retrieved {len(recent)} trade(s).")
            if recent[0]['symbol'] == 'BTCUSD':
                print("✅ Trade data matches!")
            else:
                print(f"❌ Trade data mismatch: {recent[0]}")
        else:
            print("❌ No trades retrieved.")

        # Test 4: Save session
        session_data = {
            'date': '2026-02-24',
            'start_equity': 5000.0,
            'end_equity': 13700.0,
            'net_pnl_inr': 8700.0,
            'total_trades': 1,
            'wins': 1,
            'losses': 0,
            'win_rate': 100.0,
            'session_started': '2026-02-24 00:00:00',
            'session_ended': '2026-02-24 23:59:59'
        }
        db.save_session(session_data)
        print("✅ Session saved successfully.")

        # Test 5: Retrieve session history
        history = db.get_session_history()
        if len(history) > 0:
            print(f"✅ Retrieved {len(history)} session(s).")
            if history[0]['date'] == '2026-02-24':
                print("✅ Session data matches!")
            else:
                print(f"❌ Session data mismatch: {history[0]}")
        else:
            print("❌ No session history retrieved.")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
    finally:
        # Clean up
        if os.path.exists(db_path):
            os.remove(db_path)

if __name__ == "__main__":
    test_database()
