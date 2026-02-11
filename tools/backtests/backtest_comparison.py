import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from packages.core.delta_client import DeltaClient
import csv
import sys

load_dotenv()

class StrategyTester:
    def __init__(self, symbol, starting_capital=5000):
        self.symbol = symbol
        self.starting_capital = starting_capital
        
        # API client
        self.client = DeltaClient(
            os.getenv("DELTA_API_KEY"),
            os.getenv("DELTA_API_SECRET"),
            os.getenv("DELTA_BASE_URL", "https://api.india.delta.exchange")
        )
        
        # Get contract specs
        products = self.client.get_products()
        product = next((p for p in products if p['symbol'] == symbol), None)
        self.contract_value = float(product.get('contract_value', 1)) if product else 1
        
        print(f"Initialized tester for {symbol}")

    def fetch_data(self, months=6):
        print(f"\nFetching {months} month(s) of data for {self.symbol}...")
        end_time = int(time.time())
        start_time = end_time - (months * 30 * 24 * 60 * 60)
        
        def fetch_timeframe(tf, chunk_days=5):
            all_candles = []
            current_start = start_time
            chunk_seconds = chunk_days * 24 * 60 * 60
            
            while current_start < end_time:
                current_end = min(current_start + chunk_seconds, end_time)
                candles = self.client.get_candles(self.symbol, tf, start=current_start, end=current_end)
                if candles:
                    all_candles.extend(candles)
                current_start = current_end
                time.sleep(0.2)
            
            df = pd.DataFrame(all_candles, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df.sort_values('time').drop_duplicates().reset_index(drop=True)

        print("   Fetching 5m...")
        df_5m = fetch_timeframe("5m", chunk_days=3)
        print("   Fetching 15m...")
        df_15m = fetch_timeframe("15m", chunk_days=7)
        print("   Fetching 1h...")
        df_1h = fetch_timeframe("1h", chunk_days=20)
        print("   Fetching 4h...")
        df_4h = fetch_timeframe("4h", chunk_days=50)
        
        return df_5m, df_15m, df_1h, df_4h

    def calculate_indicators(self, df):
        df = df.copy()
        # ATR for UT Bot
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(10).mean()
        
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
        
        # Market structure pivots
        df['pivot_h'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
        df['pivot_l'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
        
        # UT Bot Trailing Stop Calculation
        a = 1 # Key Value
        df['xATRTrailingStop'] = 0.0
        for i in range(1, len(df)):
            src = df['close'].iloc[i]
            prev_src = df['close'].iloc[i-1]
            prev_stop = df['xATRTrailingStop'].iloc[i-1]
            nLoss = a * df['atr'].iloc[i]
            
            if src > prev_stop and prev_src > prev_stop:
                df.loc[df.index[i], 'xATRTrailingStop'] = max(prev_stop, src - nLoss)
            elif src < prev_stop and prev_src < prev_stop:
                df.loc[df.index[i], 'xATRTrailingStop'] = min(prev_stop, src + nLoss)
            elif src > prev_stop:
                df.loc[df.index[i], 'xATRTrailingStop'] = src - nLoss
            else:
                df.loc[df.index[i], 'xATRTrailingStop'] = src + nLoss
                
        return df

    def run_backtest_ut(self, dfs, leverage=15):
        df_5m, df_15m, df_1h, df_4h = dfs
        equity = self.starting_capital
        position = None
        trades = []
        
        for idx in range(100, len(df_5m)):
            cur_5m = df_5m.iloc[idx]
            prev_5m = df_5m.iloc[idx-1]
            time_now = cur_5m['time']
            
            if position:
                cur_price = cur_5m['close']
                side = position['side']
                if side == "buy":
                    pnl_pct = (cur_price - position['entry']) / position['entry']
                    hit = cur_price <= position['sl'] or cur_price >= position['tp']
                else:
                    pnl_pct = (position['entry'] - cur_price) / position['entry']
                    hit = cur_price >= position['sl'] or cur_price <= position['tp']
                
                if hit:
                    equity += (equity * 0.1) * pnl_pct * leverage 
                    if equity < 0: equity = 0
                    trades.append({'pnl': pnl_pct * leverage})
                    position = None
                continue

            # UT Bot Signal (5m crossover)
            # Pine: buy = src > xATRTrailingStop and crossover(ema(src,1), xATRTrailingStop)
            # crossover(ema, stop) -> prev_ema <= prev_stop and cur_ema > cur_stop
            ema = cur_5m['close']
            prev_ema = prev_5m['close']
            stop = cur_5m['xATRTrailingStop']
            prev_stop = prev_5m['xATRTrailingStop']
            
            buy_signal = cur_5m['close'] > stop and prev_ema <= prev_stop and ema > stop
            sell_signal = cur_5m['close'] < stop and prev_ema >= prev_stop and ema < stop
            
            if buy_signal or sell_signal:
                # MTF Trend Filter (15m)
                idx_15m_matches = df_15m[df_15m['time'] <= time_now]
                if len(idx_15m_matches) == 0: continue
                row_15m = idx_15m_matches.iloc[-1]
                
                if buy_signal:
                    trend_ok = row_15m['ema20'] > row_15m['ema50']
                    if not trend_ok: continue
                else:
                    trend_ok = row_15m['ema20'] < row_15m['ema50']
                    if not trend_ok: continue

                position = {
                    'side': "buy" if buy_signal else "sell",
                    'entry': cur_5m['close'],
                    'sl': cur_5m['close'] * (0.985 if buy_signal else 1.015),
                    'tp': cur_5m['close'] * (1.05 if buy_signal else 0.95)
                }
        
        return equity, trades

if __name__ == "__main__":
    symbols = ["BTCUSD", "ETHUSD", "SOLUSD", "UNIUSD"]
    leverages = [15, 25, 50, 100]
    
    print("\n" + "="*80)
    print(f"UT BOT ALERTS (ATR TRAIL) 6-MONTH BACKTEST")
    print("="*80)
    print(f"{'Symbol':<10} | {'Lev':<4} | {'Final Equity':<15} | {'Trades':<8} | {'PnL %':<10}")
    print("-" * 80)
    
    for symbol in symbols:
        try:
            tester = StrategyTester(symbol)
            data = tester.fetch_data(months=6)
            dfs = [tester.calculate_indicators(df) for df in data]
            
            for lev in leverages:
                eq, trades = tester.run_backtest_ut(dfs, leverage=lev)
                pnl_pct = ((eq / 5000) - 1) * 100
                print(f"{symbol:<10} | {lev:<4} | {eq:>15.2f} | {len(trades):<8} | {pnl_pct:>+8.1f}%")
            print("-" * 80)
            
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    print("="*80)
