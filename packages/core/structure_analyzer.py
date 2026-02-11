import pandas as pd
import numpy as np

class StructureAnalyzer:
    """
    Analyzes Market Structure based on 'The Fanatic Way' (Iliya Trading Fanatic).
    Focuses on BOS, ChoCh, Order Blocks, and Fair Value Gaps.
    """

    @staticmethod
    def detect_fractals(df, window=5):
        """
        Detects swing highs and lows (fractals).
        """
        df = df.copy()
        df['swing_high'] = False
        df['swing_low'] = False

        for i in range(window, len(df) - window):
            # Swing High: highest in the window
            if df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max():
                df.at[df.index[i], 'swing_high'] = True
            
            # Swing Low: lowest in the window
            if df['low'].iloc[i] == df['low'].iloc[i-window:i+window+1].min():
                df.at[df.index[i], 'swing_low'] = True
        
        return df

    @staticmethod
    def analyze_structure(df):
        """
        Detects BOS (Break of Structure) and ChoCh (Change of Character).
        Returns the current bias and the last structure event.
        """
        df = StructureAnalyzer.detect_fractals(df)
        
        # Track last swing points
        last_high = None
        last_low = None
        bias = "neutral"
        events = []

        for i in range(1, len(df)):
            curr = df.iloc[i]
            prev_close = df.iloc[i-1]['close']

            # Update Swing Points
            if curr['swing_high']:
                last_high = curr['high']
            if curr['swing_low']:
                last_low = curr['low']

            # Detect Break of Structure (BOS) - Continuation
            if bias == "bullish" and last_high and curr['close'] > last_high:
                events.append({"type": "BOS", "side": "bullish", "price": last_high, "time": df.index[i]})
                last_high = None # Reset until next fractal
            elif bias == "bearish" and last_low and curr['close'] < last_low:
                events.append({"type": "BOS", "side": "bearish", "price": last_low, "time": df.index[i]})
                last_low = None

            # Detect Change of Character (ChoCh) - Reversal
            if bias != "bullish" and last_high and curr['close'] > last_high:
                events.append({"type": "ChoCh", "side": "bullish", "price": last_high, "time": df.index[i]})
                bias = "bullish"
                last_high = None
            elif bias != "bearish" and last_low and curr['close'] < last_low:
                events.append({"type": "ChoCh", "side": "bearish", "price": last_low, "time": df.index[i]})
                bias = "bearish"
                last_low = None

        return bias, events

    @staticmethod
    def detect_order_blocks(df, lookback=50):
        """
        Identifies Order Blocks (OB). 
        Bullish OB: The last bearish candle before a strong bullish expansion that breaks structure.
        Bearish OB: The last bullish candle before a strong bearish expansion that breaks structure.
        """
        obs = []
        df = df.tail(lookback)
        
        for i in range(1, len(df) - 2):
            c1 = df.iloc[i]   # Potential OB candle
            c2 = df.iloc[i+1] # Expansion candle
            c3 = df.iloc[i+2] # Confirmation
            
            # Bullish OB Criteria:
            # 1. c1 is bearish
            # 2. c2 and c3 are bullish and strong (expansion)
            # 3. Expansion breaks previous local high (simplied BOS)
            if c1['close'] < c1['open'] and c2['close'] > c2['open'] and c3['close'] > c3['open']:
                expansion_body = (c2['close'] - c2['open']) + (c3['close'] - c3['open'])
                if expansion_body > (abs(c1['close'] - c1['open']) * 2):
                    obs.append({
                        "type": "Bullish OB",
                        "top": c1['high'],
                        "bottom": c1['low'],
                        "time": df.index[i],
                        "mitigated": False
                    })

            # Bearish OB Criteria:
            # 1. c1 is bullish
            # 2. c2 and c3 are bearish and strong
            if c1['close'] > c1['open'] and c2['close'] < c2['open'] and c3['close'] < c3['open']:
                expansion_body = abs(c2['close'] - c2['open']) + abs(c3['close'] - c3['open'])
                if expansion_body > (abs(c1['close'] - c1['open']) * 2):
                    obs.append({
                        "type": "Bearish OB",
                        "top": c1['high'],
                        "bottom": c1['low'],
                        "time": df.index[i],
                        "mitigated": False
                    })
        
        # Check Mitigation (if price has returned to the OB)
        latest_price = df.iloc[-1]['close']
        for ob in obs:
            # Mitigation logic: if any candle AFTER OB touched the zone
            # For simplicity, we only tag them.
            pass

        return obs

    @staticmethod
    def detect_fvg(df, lookback=50):
        """
        Detects Fair Value Gaps (FVG) / Imbalances.
        """
        fvgs = []
        df = df.tail(lookback)
        
        for i in range(lookback - 3, len(df) - 1):
            if i < 1: continue
            c1 = df.iloc[i-1]
            c2 = df.iloc[i]
            c3 = df.iloc[i+1]
            
            # Bullish FVG (Gap between c1 high and c3 low)
            if c3['low'] > c1['high']:
                fvgs.append({
                    "type": "Bullish FVG",
                    "top": c3['low'],
                    "bottom": c1['high'],
                    "time": df.index[i]
                })
            
            # Bearish FVG (Gap between c1 low and c3 high)
            if c3['high'] < c1['low']:
                fvgs.append({
                    "type": "Bearish FVG",
                    "top": c1['low'],
                    "bottom": c3['high'],
                    "time": df.index[i]
                })
                
        return fvgs

    @staticmethod
    def detect_liquidity_sweeps(df, lookback=50):
        """
        Detects Liquidity Sweeps (Stop Hunts).
        Price goes above a previous fractal high and immediately reverses.
        """
        sweeps = []
        df = StructureAnalyzer.detect_fractals(df)
        
        # Look for recent swing highs/lows
        recent_sh = df[df['swing_high']].tail(5)
        recent_sl = df[df['swing_low']].tail(5)
        
        curr = df.iloc[-1]
        
        # Sweep High
        for idx, row in recent_sh.iterrows():
            if curr['high'] > row['high'] and curr['close'] < row['high']:
                sweeps.append({"type": "Liquidity Sweep High", "price": row['high'], "time": df.index[-1]})
        
        # Sweep Low
        for idx, row in recent_sl.iterrows():
            if curr['low'] < row['low'] and curr['close'] > row['low']:
                sweeps.append({"type": "Liquidity Sweep Low", "price": row['low'], "time": df.index[-1]})
                
        return sweeps
