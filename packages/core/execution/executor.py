from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
import os
from packages.core.delta_client import DeltaClient
from packages.core.database_manager import DatabaseManager


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str] = None
    message: str = ""
    filled_price: Optional[float] = None


@dataclass
class Position:
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float # Final target (TP3 or structural)
    opened_at: datetime
    pnl: float = 0.0
    pnl_pct: float = 0.0
    
    # Tiered TP and state info from bot_pro.py
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    initial_sl: float = 0.0
    qty: int = 0
    qty_remaining: int = 0
    leverage: int = 25
    risk_pct: float = 0.15
    strategy: str = "SNIPER"
    is_sniper: bool = False
    is_sure_shot: bool = False
    tp1_hit: bool = False
    tp2_hit: bool = False
    breakeven_moved: bool = False
    trailing_active: bool = False
    entry_reason: str = "Standard"
    structural_target: Optional[float] = None


class OrderExecutor:
    """Handles order execution, position management, and DB persistence."""
    
    def __init__(self, client: DeltaClient, db: DatabaseManager, paper_trading: bool = True):
        self.client = client
        self.db = db
        self.paper_trading = paper_trading
    
    def place_order(
        self,
        product_id: int,
        size: float,
        side: str,
        price: Optional[float] = None,
        order_type: str = "limit_order",
    ) -> OrderResult:
        if self.paper_trading:
            return self._paper_order(side, price)
        
        result = self.client.place_order(
            product_id=product_id,
            size=size,
            side=side,
            price=price,
            order_type=order_type,
        )
        
        if result and result.get('result'):
            order_data = result['result']
            return OrderResult(
                success=True,
                order_id=order_data.get('id'),
                message="Order placed successfully",
                filled_price=price,
            )
        
        return OrderResult(
            success=False,
            message=result.get('error', 'Unknown error') if result else 'No response'
        )
    
    def _paper_order(self, side: str, price: Optional[float]) -> OrderResult:
        return OrderResult(
            success=True,
            order_id=f"PAPER_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            message=f"Paper {side} order simulated",
            filled_price=price,
        )

    def manage_position(self, position: Position, current_price: float, high_price: float, low_price: float) -> Optional[Dict[str, Any]]:
        """
        Logic for tiered exits, SL, and trailing stops.
        Returns exit_info if a partial or full exit happens.
        """
        position.current_price = current_price
        side = position.side
        entry = position.entry_price
        
        # 1. STOP LOSS CHECK
        sl_hit = False
        exit_price = current_price
        if side == "buy":
            if low_price <= position.stop_loss:
                sl_hit = True
                exit_price = position.stop_loss
        else:
            if high_price >= position.stop_loss:
                sl_hit = True
                exit_price = position.stop_loss

        if sl_hit:
            return {"type": "FULL_EXIT", "reason": "SL", "price": exit_price}
            
        # 2. BREAKEVEN MOVE (1:1 RR)
        if not position.breakeven_moved and not position.tp1_hit:
            risk_at_entry = abs(entry - position.initial_sl)
            if side == "buy" and current_price >= entry + risk_at_entry:
                position.stop_loss = entry
                position.breakeven_moved = True
            elif side == "sell" and current_price <= entry - risk_at_entry:
                position.stop_loss = entry
                position.breakeven_moved = True
                
        # 3. TIERED PROFIT TAKING
        if side == "buy":
            # TP3/Structural Target
            final_target = max(position.tp3, position.structural_target or 0)
            if high_price >= final_target:
                return {"type": "FULL_EXIT", "reason": "TARGET_HIT", "price": final_target}
            
            # TP2
            if high_price >= position.tp2 and position.tp1_hit and not position.tp2_hit:
                position.tp2_hit = True
                position.trailing_active = True
                return {"type": "PARTIAL_EXIT", "reason": "TP2", "price": position.tp2, "portion": 0.3}
            
            # TP1
            if high_price >= position.tp1 and not position.tp1_hit:
                position.tp1_hit = True
                position.stop_loss = entry # Move to breakeven on TP1
                return {"type": "PARTIAL_EXIT", "reason": "TP1", "price": position.tp1, "portion": 0.5}
                
        else: # Sell
            final_target = min(position.tp3, position.structural_target or 999999)
            if low_price <= final_target:
                return {"type": "FULL_EXIT", "reason": "TARGET_HIT", "price": final_target}
                
            if low_price <= position.tp2 and position.tp1_hit and not position.tp2_hit:
                position.tp2_hit = True
                position.trailing_active = True
                return {"type": "PARTIAL_EXIT", "reason": "TP2", "price": position.tp2, "portion": 0.3}
                
            if low_price <= position.tp1 and not position.tp1_hit:
                position.tp1_hit = True
                position.stop_loss = entry
                return {"type": "PARTIAL_EXIT", "reason": "TP1", "price": position.tp1, "portion": 0.5}

        # 4. TRAILING STOP
        if position.trailing_active:
             if side == "buy":
                position.stop_loss = max(position.stop_loss, current_price * 0.995)
             else:
                position.stop_loss = min(position.stop_loss, current_price * 1.005)
                
        return None

    def record_and_save_trade(self, position: Position, exit_price: float, reason: str, portion: float, current_equity: float):
        """Record trade details and save to SQLite DB."""
        pnl_pct = (exit_price - position.entry_price) / position.entry_price if position.side == "buy" else (position.entry_price - exit_price) / position.entry_price
        roi = pnl_pct * position.leverage
        position_value_inr = current_equity * position.risk_pct
        pnl_inr = position_value_inr * roi * portion
        
        trade_record = {
            'symbol': position.symbol,
            'side': position.side,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'pnl_inr': round(pnl_inr, 2),
            'pnl_pct': round(pnl_pct * 100, 2),
            'roi': round(roi * 100, 2),
            'equity': round(current_equity + pnl_inr, 2),
            'reason': position.entry_reason,
            'exit_reason': reason,
            'entry_time': position.opened_at.strftime("%H:%M:%S"),
            'exit_time': datetime.now().strftime("%H:%M:%S"),
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            self.db.save_trade(trade_record)
        except Exception as e:
            print(f"⚠️ Failed to save trade to DB: {e}")
            
        return trade_record, pnl_inr
