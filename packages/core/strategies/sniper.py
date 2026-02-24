from typing import Optional, List, Dict
import pandas as pd
import numpy as np
from datetime import datetime
from packages.core.strategies.base import BaseStrategy, Signal
from packages.core.patterns import PatternRecognizer
from packages.core.structure_analyzer import StructureAnalyzer
from packages.core.config import DEFAULT_CONFIG


class SniperStrategy(BaseStrategy):
    """
    High-precision entry logic using BOS (Break of Structure), 
    ChoCh (Change of Character), and RSI divergence.
    Refined with logic from bot_pro.py.
    """
    
    def __init__(self, swing_lookback: int = 20, min_breakout_pct: float = 0.005):
        super().__init__("SNIPER")
        self.swing_lookback = swing_lookback
        self.min_breakout_pct = min_breakout_pct
        self.pattern_recognizer = PatternRecognizer()
        self.structure_analyzer = StructureAnalyzer()
    
    def analyze(self, df_5m: pd.DataFrame, df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> Optional[Signal]:
        """
        Analyze multi-timeframe data for a Sniper entry signal.
        """
        if df_5m is None or len(df_5m) < 50:
            return None
            
        cur = df_5m.iloc[-1]
        prev = df_5m.iloc[-2]
        latest_close = cur['close']
        
        # 0. HTF Trend Alignment (1h)
        # Assuming check_trend logic is simplified here or handled by structure analyzer
        bias_1h, _ = self.structure_analyzer.analyze_structure(df_1h)
        is_bullish_htf = bias_1h == "bullish"
        is_bearish_htf = bias_1h == "bearish"

        # 1. Level Proximity & Rejection Patterns
        patterns = self.pattern_recognizer.detect_all(df_5m)
        obs = self.structure_analyzer.detect_order_blocks(df_15m)
        sweeps = self.structure_analyzer.detect_liquidity_sweeps(df_5m)
        
        # Rejection Patterns with Displacement Check
        curr_ts = df_5m.index[-1].isoformat() if hasattr(df_5m.index[-1], 'isoformat') else str(df_5m.index[-1])
        latest_patterns = [p['name'] for p in patterns if p['time'] == curr_ts] if patterns else []
        
        displacement_long = cur['close'] > (prev['open'] + prev['close'])/2 if prev['close'] < prev['open'] else True
        displacement_short = cur['close'] < (prev['open'] + prev['close'])/2 if prev['close'] > prev['open'] else True

        rejection_long = any(p in latest_patterns for p in ["Hammer", "Bullish Engulfing", "Morning Star", "Bullish Marubozu"]) and displacement_long
        rejection_short = any(p in latest_patterns for p in ["Shooting Star", "Bearish Engulfing", "Evening Star", "Bearish Marubozu"]) and displacement_short
        
        # 3. Liquidity Sweeps / OB Interaction
        sweep_low = any(s['type'] == "Liquidity Sweep Low" for s in sweeps)
        sweep_high = any(s['type'] == "Liquidity Sweep High" for s in sweeps)
        near_bull_ob = any(ob['type'] == "Bullish OB" and abs(latest_close - (ob['top'] + ob['bottom'])/2) / latest_close < 0.003 for ob in obs)
        near_bear_ob = any(ob['type'] == "Bearish OB" and abs(latest_close - (ob['top'] + ob['bottom'])/2) / latest_close < 0.003 for ob in obs)

        # 4. Filter Criteria
        vol_multiplier = 1.5 
        rsi_long_limit = 40  
        rsi_short_limit = 60 
        
        vol_avg = df_5m['volume'].tail(20).mean()
        volume_exhaustion = cur['volume'] > (vol_avg * vol_multiplier)
        
        # RSI calculation is assumed to be present in df or calculated here
        rsi = cur.get('rsi', 50.0)
        rsi_os = rsi < rsi_long_limit
        rsi_ob = rsi > rsi_short_limit

        # LONG SNIPER
        if (sweep_low or near_bull_ob) and rejection_long:
            if volume_exhaustion and rsi_os and not is_bearish_htf:
                atr_val = cur.get('atr', latest_close * 0.002)
                sl = cur['low'] - (atr_val * 1.5)
                risk = abs(latest_close - sl)
                return Signal(
                    symbol=df_5m.attrs.get('symbol', 'UNKNOWN'),
                    side="buy",
                    entry_price=latest_close,
                    stop_loss=sl,
                    take_profit=latest_close + (risk * 6.0), # Structural/Max target
                    confidence=0.8,
                    strategy=self.name,
                    reason="🎯 SNIPER LONG: Rejection + Vol + RSI + Sweep/OB",
                    extra_data={
                        "tp1": latest_close + (risk * 2.0),
                        "tp2": latest_close + (risk * 4.0),
                        "is_sure_shot": True,
                        "is_sniper": True
                    }
                )

        # SHORT SNIPER
        elif (sweep_high or near_bear_ob) and rejection_short:
            if volume_exhaustion and rsi_ob and not is_bullish_htf:
                atr_val = cur.get('atr', latest_close * 0.002)
                sl = cur['high'] + (atr_val * 1.5)
                risk = abs(latest_close - sl)
                return Signal(
                    symbol=df_5m.attrs.get('symbol', 'UNKNOWN'),
                    side="sell",
                    entry_price=latest_close,
                    stop_loss=sl,
                    take_profit=latest_close - (risk * 6.0), # Structural/Max target
                    confidence=0.8,
                    strategy=self.name,
                    reason="🎯 SNIPER SHORT: Rejection + Vol + RSI + Sweep/OB",
                    extra_data={
                        "tp1": latest_close - (risk * 2.0),
                        "tp2": latest_close - (risk * 4.0),
                        "is_sure_shot": True,
                        "is_sniper": True
                    }
                )

        return None
