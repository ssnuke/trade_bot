from dataclasses import dataclass
from typing import Optional
from packages.core.config import RiskConfig, DEFAULT_CONFIG


@dataclass
class RiskMetrics:
    current_capital: float
    risk_amount: float
    position_size: float
    leverage: int
    risk_pct: float


class RiskManager:
    """Manages risk parameters and position sizing with adaptive scaling."""
    
    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or DEFAULT_CONFIG.risk
        self.consecutive_losses = 0
        self.consecutive_wins = 0
    
    def calculate_position_size(
        self,
        capital: float,
        side: str,
        entry_price: float,
        stop_loss: float,
        strategy: str = "SNIPER",
    ) -> tuple[float, int, float]:
        """
        Calculate position size based on risk parameters and adaptive scaling.
        
        Returns:
            Tuple of (position_size, leverage, risk_pct)
        """
        risk_pct = self._get_risk_pct(strategy)
        leverage = 25 # Locked at 25x as per user preference in bot_pro.py
        
        if side == "buy":
            risk_per_unit = entry_price - stop_loss
        else:
            risk_per_unit = stop_loss - entry_price
        
        # Avoid division by zero or negative risk
        if risk_per_unit <= 0:
            risk_per_unit = entry_price * 0.01
        
        # bot_pro.py calculation: position_size_inr = equity * risk_pct * leverage
        position_size_inr = capital * risk_pct * leverage
        
        # position_size here is in number of units/contracts
        # But for Delta API it's usually number of contracts
        # Wait, the MONOLITH used: position_size_usd = position_size_inr / 87
        # num_contracts = int(position_size_usd / (entry_price * contract_val))
        
        # Let's return the risk_amount (in INR) and risk_pct
        return position_size_inr, leverage, risk_pct
    
    def _get_risk_pct(self, strategy: str) -> float:
        """
        Adaptive Scaling logic from bot_pro.py:
        Sniper: 15% base -> 20% (1 win) -> 25% (2+ wins)
        EMA Cross: 10% base -> 15% (2+ wins)
        """
        if strategy == "EMA_CROSS":
            risk_pct = 0.10
            if self.consecutive_wins >= 2:
                risk_pct = 0.15
        else:  # SNIPER
            risk_pct = 0.15
            if self.consecutive_wins >= 2:
                risk_pct = 0.25
            elif self.consecutive_wins >= 1:
                risk_pct = 0.20
        
        return risk_pct
    
    def should_take_trade(
        self,
        capital: float,
        daily_trades: int,
        daily_pnl_pct: float,
        consecutive_losses: int
    ) -> tuple[bool, str]:
        """
        Check if a new trade should be taken based on risk limits.
        """
        if consecutive_losses >= 3:
            return False, f"Consecutive loss limit reached (3)"
        
        if daily_pnl_pct < -0.20:
            return False, "Daily loss limit hit (20%)"
        
        return True, ""
    
    def record_win(self) -> None:
        self.consecutive_wins += 1
        self.consecutive_losses = 0
    
    def record_loss(self) -> None:
        self.consecutive_losses += 1
        self.consecutive_wins = 0
    
    def reset(self) -> None:
        self.consecutive_wins = 0
        self.consecutive_losses = 0
