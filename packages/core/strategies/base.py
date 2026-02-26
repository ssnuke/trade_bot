from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd


@dataclass
class Signal:
    symbol: str
    side: str  # "buy" or "sell"
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float  # 0-1
    strategy: str
    reason: str
    extra_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_data is None:
            self.extra_data = {}


class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, name: str, strategy_config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.strategy_config = strategy_config or {}
    
    @abstractmethod
    def analyze(self, *args, **kwargs) -> Optional[Signal]:
        """
        Analyze the dataframe and return a trading signal if available.
        
        Signature varies by strategy implementation. See subclasses for details.
        
        Returns:
            Signal if entry condition met, None otherwise
        """
        pass
    
    def should_exit(self, df: pd.DataFrame, position: dict) -> tuple[bool, str]:
        """
        Default exit check (none). Can be overridden by subclasses.
        
        Args:
            df: DataFrame with current OHLCV data
            position: Current position data
            
        Returns:
            Tuple of (should_exit: bool, reason: str)
        """
        return False, ""
    
    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Validate that dataframe has required columns."""
        required = ["open", "high", "low", "close", "volume"]
        return all(col in df.columns for col in required) and len(df) >= 50
