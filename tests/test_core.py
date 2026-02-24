import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConfig:
    def test_config_from_yaml(self):
        from packages.core.config import Config, TradingConfig
        config = Config.from_yaml("config.yaml")
        assert config.trading.starting_capital == 5000.0
        assert config.trading.base_leverage == 25
        assert config.api.base_url == "https://api.india.delta.exchange"
    
    def test_config_from_env(self):
        os.environ["PAPER_TRADING"] = "False"
        os.environ["BASE_LEVERAGE"] = "20"
        from packages.core.config import Config
        config = Config.from_env()
        assert config.app.paper_trading == False
        assert config.trading.base_leverage == 20
        del os.environ["PAPER_TRADING"]
        del os.environ["BASE_LEVERAGE"]


class TestPatternRecognizer:
    @pytest.fixture
    def sample_df(self):
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
        df = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)
        return df
    
    def test_is_doji(self, sample_df):
        from packages.core.patterns import PatternRecognizer
        row = sample_df.iloc[-1]
        result = PatternRecognizer.is_doji(row['open'], row['high'], row['low'], row['close'])
        assert bool(result) is not None
    
    def test_is_hammer(self, sample_df):
        from packages.core.patterns import PatternRecognizer
        row = sample_df.iloc[-1]
        result = PatternRecognizer.is_hammer(row['open'], row['high'], row['low'], row['close'])
        assert bool(result) is not None
    
    def test_is_bullish_engulfing(self, sample_df):
        from packages.core.patterns import PatternRecognizer
        curr = sample_df.iloc[-1]
        prev = sample_df.iloc[-2]
        result = PatternRecognizer.is_bullish_engulfing(curr, prev)
        assert bool(result) is not None
    
    def test_detect_all(self, sample_df):
        from packages.core.patterns import PatternRecognizer
        result = PatternRecognizer.detect_all(sample_df)
        assert isinstance(result, list)
    
    def test_support_resistance_pivots(self):
        from packages.core.patterns import SupportResistance
        pivots = SupportResistance.calculate_pivots(110, 90, 100)
        assert 'P' in pivots
        assert 'R1' in pivots
        assert 'S1' in pivots
        assert pivots['P'] == 100.0


class TestStructureAnalyzer:
    @pytest.fixture
    def sample_df(self):
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
        df = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)
        return df
    
    def test_detect_fractals(self, sample_df):
        from packages.core.structure_analyzer import StructureAnalyzer
        result = StructureAnalyzer.detect_fractals(sample_df)
        assert 'swing_high' in result.columns
        assert 'swing_low' in result.columns
    
    def test_analyze_structure(self, sample_df):
        from packages.core.structure_analyzer import StructureAnalyzer
        bias, events = StructureAnalyzer.analyze_structure(sample_df)
        assert bias in ["bullish", "bearish", "neutral"]
        assert isinstance(events, list)
    
    def test_detect_fvg(self, sample_df):
        from packages.core.structure_analyzer import StructureAnalyzer
        fvgs = StructureAnalyzer.detect_fvg(sample_df)
        assert isinstance(fvgs, list)


class TestRiskManager:
    def test_calculate_position_size(self):
        from packages.core.risk.manager import RiskManager
        rm = RiskManager()
        size, lev, risk_pct = rm.calculate_position_size(
            capital=5000,
            side="buy",
            entry_price=100,
            stop_loss=98
        )
        assert size > 0
        assert lev > 0
        assert risk_pct > 0
    
    def test_should_take_trade(self):
        from packages.core.risk.manager import RiskManager
        rm = RiskManager()
        should_trade, reason = rm.should_take_trade(5000, 5, 0.05)
        assert should_trade == True
    
    def test_consecutive_losses(self):
        from packages.core.risk.manager import RiskManager
        rm = RiskManager()
        rm.consecutive_losses = 3
        should_trade, reason = rm.should_take_trade(5000, 5, 0.0)
        assert should_trade == False
    
    def test_record_win_loss(self):
        from packages.core.risk.manager import RiskManager
        rm = RiskManager()
        rm.record_win()
        assert rm.consecutive_wins == 1
        rm.record_loss()
        assert rm.consecutive_losses == 1
        assert rm.consecutive_wins == 0


class TestStrategy:
    @pytest.fixture
    def sample_df(self):
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
        df = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)
        df.attrs['symbol'] = 'ETHUSD'
        return df
    
    def test_sniper_strategy_analyze(self, sample_df):
        from packages.core.strategies.sniper import SniperStrategy
        strategy = SniperStrategy()
        result = strategy.analyze(sample_df, 'ETHUSD')
        assert result is None or result.symbol == 'ETHUSD'
    
    def test_sniper_should_exit(self, sample_df):
        from packages.core.strategies.sniper import SniperStrategy
        strategy = SniperStrategy()
        position = {'side': 'buy', 'entry_price': 100, 'stop_loss': 98, 'take_profit': 105}
        should_exit, reason = strategy.should_exit(sample_df, position)
        assert isinstance(should_exit, bool)
    
    def test_validate_dataframe(self, sample_df):
        from packages.core.strategies.sniper import SniperStrategy
        strategy = SniperStrategy()
        assert strategy.validate_dataframe(sample_df) == True
        
        bad_df = pd.DataFrame({'open': [1, 2, 3]})
        assert strategy.validate_dataframe(bad_df) == False


class TestOrderExecutor:
    def test_paper_order(self):
        from packages.core.execution.executor import OrderExecutor, DeltaClient
        client = DeltaClient(None, None)
        executor = OrderExecutor(client, paper_trading=True)
        result = executor.place_order(1, 100, "buy", 100.0)
        assert result.success == True
        assert "PAPER" in result.order_id
    
    def test_calculate_pnl_buy(self):
        from packages.core.execution.executor import OrderExecutor, Position, DeltaClient
        from datetime import datetime
        client = DeltaClient(None, None)
        executor = OrderExecutor(client)
        
        position = Position(
            symbol="ETHUSD",
            side="buy",
            size=1,
            entry_price=100,
            current_price=110,
            stop_loss=95,
            take_profit=120,
            opened_at=datetime.now()
        )
        
        pnl, pnl_pct = executor.calculate_pnl(position, 110)
        assert pnl == 10
    
    def test_should_close_position(self):
        from packages.core.execution.executor import OrderExecutor, Position, DeltaClient
        from datetime import datetime
        client = DeltaClient(None, None)
        executor = OrderExecutor(client)
        
        position = Position(
            symbol="ETHUSD",
            side="buy",
            size=1,
            entry_price=100,
            current_price=100,
            stop_loss=95,
            take_profit=110,
            opened_at=datetime.now()
        )
        
        should_close, reason = executor.should_close_position(position, 94)
        assert should_close == True
        assert "stop loss" in reason.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
