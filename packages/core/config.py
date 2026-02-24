import os
from typing import Optional
from dataclasses import dataclass, field
import yaml


@dataclass
class TradingConfig:
    starting_capital: float = 5000.0
    target_equity: float = 80000.0
    base_leverage: int = 25
    max_daily_loss_pct: float = 0.20
    max_concurrent_positions: int = 1
    swing_lookback: int = 20
    volume_mult: float = 1.5
    min_breakout_pct: float = 0.005


@dataclass
class RiskConfig:
    base_risk_pct: float = 0.15
    max_risk_pct: float = 0.25
    consecutive_loss_limit: int = 3
    min_win_streak_for_increase: int = 2
    ema_cross_risk_pct: float = 0.10
    ema_cross_max_risk_pct: float = 0.15


@dataclass
class APIConfig:
    base_url: str = "https://api.india.delta.exchange"
    timeout: int = 15
    max_retries: int = 3


@dataclass
class SessionConfig:
    timezone: str = "IST"
    utc_offset_hours: int = 5
    utc_offset_minutes: int = 30


@dataclass
class SymbolsConfig:
    priority: list[str] = field(default_factory=lambda: [
        "ETHUSD", "SOLUSD", "XRPUSD", "BNBUSD", "UNIUSD", "LTCUSD",
        "DOGEUSD", "LINKUSD", "AVAXUSD", "DOTUSD", "ADAUSD", "ATOMUSD",
        "ALGOUSD", "BCHUSD"
    ])


@dataclass
class AppConfig:
    paper_trading: bool = True
    turbo_mode: bool = True
    sure_shot_only: bool = True
    log_dir: str = "paper_trades"
    dashboard_file: str = "dashboard_data.json"


@dataclass
class Config:
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    api: APIConfig = field(default_factory=APIConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    symbols: SymbolsConfig = field(default_factory=SymbolsConfig)
    app: AppConfig = field(default_factory=AppConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        if not os.path.exists(path):
            return cls()
        
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        
        return cls(
            trading=TradingConfig(**data.get("trading", {})),
            risk=RiskConfig(**data.get("risk", {})),
            api=APIConfig(**data.get("api", {})),
            session=SessionConfig(**data.get("session", {})),
            symbols=SymbolsConfig(**data.get("symbols", {})),
            app=AppConfig(**data.get("app", {}))
        )

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            trading=TradingConfig(
                starting_capital=float(os.getenv("STARTING_CAPITAL", "5000")),
                target_equity=float(os.getenv("TARGET_EQUITY", "80000")),
                base_leverage=int(os.getenv("BASE_LEVERAGE", "25")),
                max_daily_loss_pct=float(os.getenv("MAX_DAILY_LOSS_PCT", "0.20")),
            ),
            api=APIConfig(
                base_url=os.getenv("DELTA_BASE_URL", "https://api.india.delta.exchange"),
            ),
            app=AppConfig(
                paper_trading=os.getenv("PAPER_TRADING", "True") == "True",
                turbo_mode=os.getenv("TURBO_MODE", "True") == "True",
                sure_shot_only=os.getenv("SURE_SHOT_ONLY", "True") == "True",
            )
        )


DEFAULT_CONFIG = Config()
