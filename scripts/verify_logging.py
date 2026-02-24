import os
import sys
from datetime import datetime, timedelta

# Add direct paths to sys.path
BASE_DIR = "/Users/snehith/Desktop/Development/delta_bot/apps/bot"
PROJECT_ROOT = "/Users/snehith/Desktop/Development/delta_bot"
sys.path.append(PROJECT_ROOT)

# Mock environment variables needed for initialization
os.environ["DELTA_API_KEY"] = "mock_key"
os.environ["DELTA_API_SECRET"] = "mock_secret"
os.environ["PAPER_TRADING"] = "True"

from apps.bot.bot_pro import AggressiveGrowthBot

def test_logging():
    print("Testing AggressiveGrowthBot Logging...")
    # Initialize bot (this will call _init_products, hence we need to mock some stuff if it fails)
    # But initialization also calls log(), which is what we want to test.
    try:
        bot = AggressiveGrowthBot()
        
        # Test direct log call
        test_msg = "VERIFICATION TEST LOG MESSAGE"
        bot.log(test_msg)
        
        # Verify file exists
        log_file = bot.current_log_file
        print(f"Checking log file: {log_file}")
        
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                content = f.read()
                if test_msg in content:
                    print("✅ Log message found in file!")
                else:
                    print("❌ Log message NOT found in file.")
        else:
            print(f"❌ Log file not found at {log_file}")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

if __name__ == "__main__":
    test_logging()
