import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from packages.core.delta_client import DeltaClient
from dotenv import load_dotenv

load_dotenv()

class TradeReplay:
    def __init__(self):
        self.api_key = os.getenv("DELTA_API_KEY")
        self.api_secret = os.getenv("DELTA_API_SECRET")
        self.base_url = os.getenv("DELTA_BASE_URL", "https://api.india.delta.exchange")
        self.client = DeltaClient(self.api_key, self.api_secret, self.base_url)
        self.volume_mult = 1.5

    def get_candles(self, symbol, resolution):
        # Just fetch latest, assume it covers the last 5-8 hours
        candles = self.client.get_candles(symbol, resolution)
        if not candles: return None
        df = pd.DataFrame(candles, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df.sort_values('time').drop_duplicates().reset_index(drop=True)

    def analyze_moment(self, symbol, target_iso_str):
        target_dt = datetime.fromisoformat(target_iso_str)
        print(f"\nANALYZING {symbol} at {target_dt} (IST approx)")
        print("-" * 60)
        
        # Simple fetch
        df_5m = self.get_candles(symbol, "5m")
        df_15m = self.get_candles(symbol, "15m")
        df_1h = self.get_candles(symbol, "1h")
        df_4h = self.get_candles(symbol, "4h")
        
        if any(x is None for x in [df_5m, df_15m, df_1h, df_4h]):
            print("[Error] Error fetching data.")
            return
            
        print(f"DEBUG: 5m Range: {df_5m['time'].min()} to {df_5m['time'].max()}")

        df_5m = self.calculate_indicators(df_5m)
        df_15m = self.calculate_indicators(df_15m)
        df_1h = self.calculate_indicators(df_1h)
        df_4h = self.calculate_indicators(df_4h)
        
        target_ts = pd.Timestamp(target_dt)
        row_5m = df_5m[df_5m['time'] == target_ts]
        
        if row_5m.empty:
            print(f"[Warning] No 5m candle found exactly at {target_ts}. Checking nearby...")
            # print(df_5m['time'].tail(10))
            row_5m = df_5m.iloc[[-1]] # Default to last if specific not found? No that's bad.
            # actually we should look for the closest candle <= target
            row_5m = df_5m[df_5m['time'] <= target_ts].iloc[[-1]]

        idx = row_5m.index[0]
        cur = df_5m.iloc[idx]
        prev = df_5m.iloc[idx-1]
        
        print(f"Candle Time: {cur['time']}")
        print(f"Close: {cur['close']} | UT Trail: {cur['ut_trail']:.2f}")
        
        # 1. Check Signal
        ut_buy = cur['close'] > cur['ut_trail'] and prev['close'] <= prev['ut_trail']
        ut_sell = cur['close'] < cur['ut_trail'] and prev['close'] >= prev['ut_trail']
        
        print(f"1. UT Signal: {'BUY' if ut_buy else 'SELL' if ut_sell else 'NONE'}")
        
        if not (ut_buy or ut_sell):
            print("   -> No crossover at this specific candle.")
            return

        # 2. Check 15m Trend
        row_15m = df_15m[df_15m['time'] <= cur['time']].iloc[-1]
        t_up = row_15m['ema20'] > row_15m['ema50'] and row_15m['close'] > row_15m['ema200']
        t_down = row_15m['ema20'] < row_15m['ema50'] and row_15m['close'] < row_15m['ema200']
        trend = "up" if t_up else "down" if t_down else "none"
        print(f"2. 15m Trend: {trend.upper()} (EMA20={row_15m['ema20']:.2f}, EMA50={row_15m['ema50']:.2f})")
        
        sig_type = "buy" if ut_buy else "sell"
        if (sig_type == "buy" and trend != "up") or (sig_type == "sell" and trend != "down"):
            print("   [REJECTED] Trend mismatch.")
        
        # 3. Check HTF Structure
        row_1h = df_1h[df_1h['time'] <= cur['time']].iloc[-1]
        row_4h = df_4h[df_4h['time'] <= cur['time']].iloc[-1]
        
        idx_1h = df_1h.index.get_loc(row_1h.name)
        idx_4h = df_4h.index.get_loc(row_4h.name)
        
        # Slicing safely
        # We need to pass a slice of the dataframe up to that point to detect_structure
        sub_1h = df_1h.iloc[:idx_1h+1]
        sub_4h = df_4h.iloc[:idx_4h+1]
        
        s_1h = self.detect_structure(sub_1h)
        s_4h = self.detect_structure(sub_4h)
        
        print(f"3. HTF Structure: 1h={s_1h}, 4h={s_4h}")
        
        if sig_type == "buy" and (s_1h == "LH_LL" or s_4h == "LH_LL"):
             print("   [REJECTED] HTF Bearish Structure.")
        elif sig_type == "sell" and (s_1h == "HH_HL" or s_4h == "HH_HL"):
             print("   [REJECTED] HTF Bullish Structure.")

        # 4. Volume
        vol_ratio = cur['volume'] / cur['vol_avg']
        print(f"4. Volume: {cur['volume']:.0f} (Avg: {cur['vol_avg']:.0f}) -> Ratio: {vol_ratio:.2f}x")
        if vol_ratio < 1.5:
            print("   [REJECTED] Volume Spike too small (< 1.5x)")

if __name__ == "__main__":
    replay = TradeReplay()
    # 2026-02-06 09:00:00 UTC (14:30 IST)
    replay.analyze_moment("UNIUSD", "2026-02-06 09:00:00")
