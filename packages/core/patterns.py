import pandas as pd
import numpy as np

class PatternRecognizer:
    """
    Identifies candlestick patterns in a DataFrame.
    """
    
    @staticmethod
    def is_doji(open_px, high, low, close):
        body = abs(close - open_px)
        rng = high - low
        return body <= (rng * 0.1) and rng > 0

    @staticmethod
    def is_hammer(open_px, high, low, close):
        body = abs(close - open_px)
        rng = high - low
        lower_wick = min(open_px, close) - low
        upper_wick = high - max(open_px, close)
        return lower_wick >= (body * 2) and upper_wick <= (body * 0.5) and rng > 0

    @staticmethod
    def is_shooting_star(open_px, high, low, close):
        body = abs(close - open_px)
        rng = high - low
        lower_wick = min(open_px, close) - low
        upper_wick = high - max(open_px, close)
        return upper_wick >= (body * 2) and lower_wick <= (body * 0.5) and rng > 0

    @staticmethod
    def is_bullish_engulfing(curr, prev):
        return (curr['close'] > curr['open']) and \
               (prev['close'] < prev['open']) and \
               (curr['close'] >= prev['open']) and \
               (curr['open'] <= prev['close'])

    @staticmethod
    def is_bearish_engulfing(curr, prev):
        return (curr['close'] < curr['open']) and \
               (prev['close'] > prev['open']) and \
               (curr['close'] <= prev['open']) and \
               (curr['open'] >= prev['close'])

    @staticmethod
    def is_marubozu(open_px, high, low, close):
        body = abs(close - open_px)
        rng = high - low
        return body >= (rng * 0.9) and rng > 0

    @staticmethod
    def is_morning_star(candles):
        # candles: [first (bearish), second (doji/small), third (bullish)]
        if len(candles) < 3: return False
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        
        is_c1_bear = c1['close'] < c1['open']
        is_c2_small = abs(c2['close'] - c2['open']) < (abs(c1['close'] - c1['open']) * 0.5)
        is_c3_bull = c3['close'] > c3['open']
        
        # Gap down (optional but classic) logic simplified: c2 body below c1 body
        gap_down = max(c2['open'], c2['close']) < min(c1['open'], c1['close'])
        # C3 closes deeply into C1
        piercing = c3['close'] > (c1['close'] + (c1['open'] - c1['close']) / 2)
        
        return is_c1_bear and is_c2_small and is_c3_bull and piercing

    @staticmethod
    def is_evening_star(candles):
        if len(candles) < 3: return False
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        
        is_c1_bull = c1['close'] > c1['open']
        is_c2_small = abs(c2['close'] - c2['open']) < (abs(c1['close'] - c1['open']) * 0.5)
        is_c3_bear = c3['close'] < c3['open']
        
        # Gap up: c2 body above c1 body
        gap_up = min(c2['open'], c2['close']) > max(c1['open'], c1['close'])
        # C3 closes deeply into C1
        piercing = c3['close'] < (c1['close'] - (c1['close'] - c1['open']) / 2)
        
        return is_c1_bull and is_c2_small and is_c3_bear and piercing

    @staticmethod
    def is_three_white_soldiers(candles):
        if len(candles) < 3: return False
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        
        return (c1['close'] > c1['open']) and \
               (c2['close'] > c2['open']) and \
               (c3['close'] > c3['open']) and \
               (c2['close'] > c1['close']) and \
               (c3['close'] > c2['close']) and \
               (c2['open'] > c1['open'] and c2['open'] < c1['close']) and \
               (c3['open'] > c2['open'] and c3['open'] < c2['close'])

    @staticmethod
    def is_three_black_crows(candles):
        if len(candles) < 3: return False
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        
        return (c1['close'] < c1['open']) and \
               (c2['close'] < c2['open']) and \
               (c3['close'] < c3['open']) and \
               (c2['close'] < c1['close']) and \
               (c3['close'] < c2['close']) and \
               (c2['open'] < c1['open'] and c2['open'] > c1['close']) and \
               (c3['open'] < c2['open'] and c3['open'] > c2['close'])

    @staticmethod
    def detect_all(df):
        if len(df) < 50: return []
        
        all_detected = []
        # Scan last 10 candles (newest at end of df)
        # We look at i, i-1, i-2
        scan_window_start = len(df) - 10
        if scan_window_start < 2: scan_window_start = 2
        
        for i in range(scan_window_start, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            prev2 = df.iloc[i-2]
            candles_3 = [prev2, prev, curr]
            
            def add_p(name):
                # ISO Format Timestamp for better JS compatibility
                # curr.name is now a Timestamp object since we set it as index in bot_pro.py
                try:
                    ts_str = curr.name.isoformat()
                except:
                    ts_str = str(curr.name)

                all_detected.append({
                    "name": name,
                    "price": float(curr['close']),
                    "time": ts_str
                })

            # Single candle patterns
            if PatternRecognizer.is_doji(curr['open'], curr['high'], curr['low'], curr['close']):
                add_p("Doji")
            if PatternRecognizer.is_hammer(curr['open'], curr['high'], curr['low'], curr['close']):
                add_p("Hammer")
            if PatternRecognizer.is_shooting_star(curr['open'], curr['high'], curr['low'], curr['close']):
                add_p("Shooting Star")
            if PatternRecognizer.is_marubozu(curr['open'], curr['high'], curr['low'], curr['close']):
                if curr['close'] > curr['open']: add_p("Bullish Marubozu")
                else: add_p("Bearish Marubozu")
                
            # Two candle patterns
            if PatternRecognizer.is_bullish_engulfing(curr, prev):
                add_p("Bullish Engulfing")
            if PatternRecognizer.is_bearish_engulfing(curr, prev):
                add_p("Bearish Engulfing")
                
            # Three candle patterns
            if PatternRecognizer.is_morning_star(candles_3):
                add_p("Morning Star")
            if PatternRecognizer.is_evening_star(candles_3):
                add_p("Evening Star")
            if PatternRecognizer.is_three_white_soldiers(candles_3):
                add_p("Three White Soldiers")
            if PatternRecognizer.is_three_black_crows(candles_3):
                add_p("Three Black Crows")
            
        # Return reversed (Newest First)
        return all_detected[::-1]

class SupportResistance:
    """
    Identifies support and resistance levels.
    """
    
    @staticmethod
    def calculate_pivots(high, low, close):
        p = (high + low + close) / 3
        r1 = (2 * p) - low
        s1 = (2 * p) - high
        r2 = p + (high - low)
        s2 = p - (high - low)
        return {"P": p, "R1": r1, "S1": s1, "R2": r2, "S2": s2}

    @staticmethod
    def detect_swing_levels(df, lookback=5):
        """
        Detects swing highs and lows using a fractal-like approach.
        """
        levels = []
        for i in range(lookback, len(df) - lookback):
            # Swing High
            if all(df['high'].iloc[i] > df['high'].iloc[i-k] for k in range(1, lookback+1)) and \
               all(df['high'].iloc[i] > df['high'].iloc[i+k] for k in range(1, lookback+1)):
                levels.append({"type": "Resistance", "price": df['high'].iloc[i], "index": i})
                
            # Swing Low
            if all(df['low'].iloc[i] < df['low'].iloc[i-k] for k in range(1, lookback+1)) and \
               all(df['low'].iloc[i] < df['low'].iloc[i+k] for k in range(1, lookback+1)):
                levels.append({"type": "Support", "price": df['low'].iloc[i], "index": i})
        
        return levels
