
import sys
import os
import pandas as pd

# Add the project root to sys.path
PROJECT_ROOT = "/Users/snehith/Desktop/Development/delta_bot"
sys.path.append(PROJECT_ROOT)

try:
    from packages.core.strategies.sniper import SniperStrategy
    from packages.core.strategies.ema_cross import EMACrossStrategy
    
    print("Attempting to instantiate SniperStrategy...")
    sniper = SniperStrategy()
    print("✅ SniperStrategy instantiated successfully.")
    
    print("Attempting to instantiate EMACrossStrategy...")
    ema = EMACrossStrategy()
    print("✅ EMACrossStrategy instantiated successfully.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
