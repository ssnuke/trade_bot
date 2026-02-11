import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from packages.core.delta_client import DeltaClient

load_dotenv()

class AutonomousSniperTester:
    def __init__(self, symbol, starting_capital=5000):
        self.symbol = symbol
        self.starting_capital = starting_capital
        self.bootstrap_target = 15000
        self.client = DeltaClient(
            os.getenv("DELTA_API_KEY"),
            os.getenv("DELTA_API_SECRET"),
            os.getenv("DELTA_BASE_URL", "https://api.india.delta.exchange")
        )

    def fetch_data(self, days=30):
        end_time = int(time.time())
        start_time = end_time - (days * 24 * 60 * 60 + (4 * 3600)) # buffer for indicators
        
        def fetch_tf(tf, chunk_days=5):
            all_candles = []
            curr_start = start_time
            while curr_start < end_time:
                curr_end = min(curr_start + (chunk_days * 86400), end_time)
                candles = self.client.get_candles(self.symbol, tf, start=curr_start, end=curr_end)
                if candles: all_candles.extend(candles)
                curr_start = curr_end
                time.sleep(0.1)
            df = pd.DataFrame(all_candles, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df.sort_values('time').drop_duplicates().reset_index(drop=True)

        return fetch_tf("5m"), fetch_tf("15m"), fetch_tf("1h"), fetch_tf("4h")

    def indicators(self, df):
        df = df.copy()
        # ATR 10
        hl = df['high'] - df['low']
        hc = np.abs(df['high'] - df['close'].shift())
        lc = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['atr'] = tr.rolling(10).mean()
        
        # EMAs
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # UT Bot Trail
        a = 1 
        df['ut_trail'] = 0.0
        for i in range(1, len(df)):
            src, p_src, p_stop, nL = df['close'].iloc[i], df['close'].iloc[i-1], df['ut_trail'].iloc[i-1], a*df['atr'].iloc[i]
            if src > p_stop and p_src > p_stop: df.loc[df.index[i], 'ut_trail'] = max(p_stop, src - nL)
            elif src < p_stop and p_src < p_stop: df.loc[df.index[i], 'ut_trail'] = min(p_stop, src + nL)
            elif src > p_stop: df.loc[df.index[i], 'ut_trail'] = src - nL
            else: df.loc[df.index[i], 'ut_trail'] = src + nL
        
        # Pivot points for structure
        df['pivot_h'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
        df['pivot_l'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
        
        # Volume Avg
        df['vol_avg'] = df['volume'].rolling(20).mean()
        
        return df

    def get_market_structure(self, df, idx):
        subset = df.iloc[max(0, idx-50):idx+1]
        highs = subset[subset['pivot_h']]['high'].tail(2).tolist()
        lows = subset[subset['pivot_l']]['low'].tail(2).tolist()
        if len(highs) < 2 or len(lows) < 2: return "range"
        if highs[-1] > highs[-2] and lows[-1] > lows[-2]: return "HH_HL"
        if highs[-1] < highs[-2] and lows[-1] < lows[-2]: return "LH_LL"
        return "range"

    def get_candlestick(self, df, idx):
        if idx < 1: return None
        cur = df.iloc[idx]
        prev = df.iloc[idx-1]
        # Engulfing
        bull_eng = cur['close'] > cur['open'] and prev['close'] < prev['open'] and cur['close'] >= prev['open'] and cur['open'] <= prev['close']
        bear_eng = cur['close'] < cur['open'] and prev['close'] > prev['open'] and cur['close'] <= prev['open'] and cur['open'] >= prev['close']
        if bull_eng: return "bullish"
        if bear_eng: return "bearish"
        return None

    def run_autonomous_sniper(self, days=30):
        df_5m, df_15m, df_1h, df_4h = self.fetch_data(days)
        df_5m = self.indicators(df_5m)
        df_15m = self.indicators(df_15m)
        df_1h = self.indicators(df_1h)
        df_4h = self.indicators(df_4h)
        
        equity = self.starting_capital
        position = None
        trades = []
        
        lev_map = {"BTCUSD": 50, "ETHUSD": 25, "default": 15}

        for i in range(100, len(df_5m)):
            cur, prev = df_5m.iloc[i], df_5m.iloc[i-1]
            time_now = cur['time']
            
            if position:
                pnl = ((cur['close'] - position['entry']) / position['entry']) if position['side'] == 'buy' else ((position['entry'] - cur['close']) / position['entry'])
                # Liquidation check for 100x (1% move)
                if position['leverage'] >= 100 and pnl <= -0.01:
                    equity += (equity * 0.1) * -1.0 # Full 10% risk wiped
                    position = None
                    continue
                    
                if cur['low'] <= position['sl'] if position['side'] == 'buy' else cur['high'] >= position['sl']:
                    pnl_inr = (equity * 0.1) * -0.01 * position['leverage'] # 1% risk of 10% balance
                    equity += pnl_inr
                    trades.append({'pnl': pnl_inr})
                    position = None
                elif cur['high'] >= position['tp'] if position['side'] == 'buy' else cur['low'] <= position['tp']:
                    pnl_inr = (equity * 0.1) * 0.05 * position['leverage'] # 5% profit target
                    equity += pnl_inr
                    trades.append({'pnl': pnl_inr})
                    position = None
                continue

            # UT Crossover
            buy = cur['close'] > cur['ut_trail'] and prev['close'] <= prev['ut_trail']
            sell = cur['close'] < cur['ut_trail'] and prev['close'] >= prev['ut_trail']
            
            if buy or sell:
                # MTF Trend check (15m)
                matches_15m = df_15m[df_15m['time'] <= time_now]
                if len(matches_15m) == 0: continue
                row_15m = matches_15m.iloc[-1]
                trend_ok = (buy and row_15m['ema20'] > row_15m['ema50']) or (sell and row_15m['ema20'] < row_15m['ema50'])
                if not trend_ok: continue

                # HTF check (1h, 4h)
                matches_1h = df_1h[df_1h['time'] <= time_now]
                matches_4h = df_4h[df_4h['time'] <= time_now]
                if len(matches_1h) == 0 or len(matches_4h) == 0: continue
                struct_1h = self.get_market_structure(df_1h, len(matches_1h)-1)
                struct_4h = self.get_market_structure(df_4h, len(matches_4h)-1)
                
                # Full Confluence for Sure-Shot
                is_sure_shot = False
                if buy:
                    if struct_1h == "HH_HL" and struct_4h == "HH_HL":
                        struct_5m = self.get_market_structure(df_5m, i)
                        # Check last 3 candles for bullish pattern
                        candl_5m = any(self.get_candlestick(df_5m, i-j) == "bullish" for j in range(3))
                        if (struct_5m == "HH_HL" or candl_5m) and cur['volume'] > (cur['vol_avg'] * 1.5):
                            is_sure_shot = True
                else:
                    if struct_1h == "LH_LL" and struct_4h == "LH_LL":
                        struct_5m = self.get_market_structure(df_5m, i)
                        # Check last 3 candles for bearish pattern
                        candl_5m = any(self.get_candlestick(df_5m, i-j) == "bearish" for j in range(3))
                        if (struct_5m == "LH_LL" or candl_5m) and cur['volume'] > (cur['vol_avg'] * 1.5):
                            is_sure_shot = True

                # Adaptive Leverage Logic
                if is_sure_shot and equity < self.bootstrap_target:
                    lev = 100
                elif is_sure_shot:
                    lev = lev_map.get(self.symbol, lev_map["default"])
                else:
                    lev = 15 # Standard setup
                
                # For this backtest, only take Sure-Shot trades to see "Sniper" performance
                if not is_sure_shot: continue

                position = {
                    'side': 'buy' if buy else 'sell',
                    'entry': cur['close'],
                    'leverage': lev,
                    'sl': cur['close'] * (0.99 if buy else 1.01),
                    'tp': cur['close'] * (1.05 if buy else 0.95)
                }
        return equity, len(trades)

if __name__ == "__main__":
    symbols = ["BTCUSD", "ETHUSD", "SOLUSD", "UNIUSD"]
    ranges = [("Last Week", 7), ("Last Month", 30)]
    
    print("\n" + "="*80)
    print(f"AUTONOMOUS SNIPER (100x -> 50x) BACKTEST")
    print(f"Sure-Shot ONLY | Bootstrap Target: 15,000 INR")
    print("="*80)
    
    for label, days in ranges:
        print(f"\n--- Period: {label} ---")
        print(f"{'Symbol':<10} | {'Final Equity':<15} | {'Trades':<8} | {'PnL %':<10}")
        print("-" * 55)
        for sym in symbols:
            tester = AutonomousSniperTester(sym)
            eq, count = tester.run_autonomous_sniper(days)
            pnl = ((eq/5000)-1)*100
            print(f"{sym:<10} | {eq:>15.2f} | {count:<8} | {pnl:>+8.1f}%")
    print("="*80)
