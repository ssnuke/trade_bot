from typing import Optional
import pandas as pd
from packages.core.strategies.base import BaseStrategy, Signal


class EMACrossStrategy(BaseStrategy):
    """
    Trend-following logic using EMA 9/15 crossovers. 
    Refined with strict trending filters from bot_pro.py.
    """
    
    def __init__(self):
        super().__init__("EMA_CROSS")
    
    def analyze(self, df_5m: pd.DataFrame, df_15m: pd.DataFrame) -> Optional[Signal]:
        """
        Analyze 5m and 15m data for an EMA cross signal.
        """
        if df_5m is None or len(df_5m) < 50:
            return None
            
        cur = df_5m.iloc[-1]
        prev = df_5m.iloc[-2]
        latest_close = cur['close']
        
        # 0. Trending Filters: ADX > 30, Slope check, Volume confirmation
        is_trending = cur.get('adx', 0) > 30 and abs(cur.get('slope', 0)) > 0.3
        vol_avg = df_5m['volume'].tail(20).mean()
        volume_confirmed = cur['volume'] > (vol_avg * 1.2)
        
        # Clarity/Indecision check (simplified version of PatternRecognizer check)
        # In a real impl, we'd use PatternRecognizer here too
        
        # 📈 EMA 9/15 CROSS UP
        buy_cross = prev.get('ema9', 0) <= prev.get('ema15', 0) and cur.get('ema9', 0) > cur.get('ema15', 0)
        
        # 📉 EMA 9/15 CROSS DOWN
        sell_cross = prev.get('ema9', 0) >= prev.get('ema15', 0) and cur.get('ema9', 0) < cur.get('ema15', 0)
        
        if is_trending and volume_confirmed:
            if buy_cross:
                # Initial SL (Stop loss at EMA 50 or 1.5%)
                sl = cur.get('ema50', latest_close * 0.985)
                if latest_close <= sl: sl = latest_close * 0.985
                
                return Signal(
                    symbol=df_5m.attrs.get('symbol', 'UNKNOWN'),
                    side="buy",
                    entry_price=latest_close,
                    stop_loss=sl,
                    take_profit=latest_close * 1.10, # Broad Target (EXIT_ON_CROSS)
                    confidence=0.6,
                    strategy=self.name,
                    reason="📈 HYBRID: EMA CROSS UP (Strong Trend)",
                    extra_data={
                        "tp1": latest_close * 1.02,
                        "tp2": latest_close * 1.04,
                        "is_sure_shot": False,
                        "is_sniper": False
                    }
                )
            elif sell_cross:
                sl = cur.get('ema50', latest_close * 1.015)
                if latest_close >= sl: sl = latest_close * 1.015
                
                return Signal(
                    symbol=df_5m.attrs.get('symbol', 'UNKNOWN'),
                    side="sell",
                    entry_price=latest_close,
                    stop_loss=sl,
                    take_profit=latest_close * 0.90, # Broad Target (EXIT_ON_CROSS)
                    confidence=0.6,
                    strategy=self.name,
                    reason="📉 HYBRID: EMA CROSS DOWN (Strong Trend)",
                    extra_data={
                        "tp1": latest_close * 0.98,
                        "tp2": latest_close * 0.96,
                        "is_sure_shot": False,
                        "is_sniper": False
                    }
                )

        return None

    def should_exit(self, df_5m: pd.DataFrame, position: dict) -> tuple[bool, str]:
        """
        Special exit logic for EMA Cross: Exit on EMA 50 cross.
        """
        cur = df_5m.iloc[-1]
        cur_price = cur['close']
        side = position.get('side', '')
        
        ema_exit = False
        if side == "buy" and cur_price < cur.get('ema50', 0):
            ema_exit = True
        elif side == "sell" and cur_price > cur.get('ema50', 0):
            ema_exit = True
            
        if ema_exit:
            return True, "EMA_50_CROSS"
            
        return super().should_exit(df_5m, position)
