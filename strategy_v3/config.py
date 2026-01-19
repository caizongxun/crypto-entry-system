"""
Configuration module for Strategy V3.

Defines all configurable parameters for technical indicators, model training,
and signal generation thresholds.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class TechnicalIndicatorConfig:
    """
    Configuration for technical indicators.
    """
    # SMA periods
    sma_short: int = 10
    sma_medium: int = 20
    sma_long: int = 50

    # EMA periods
    ema_short: int = 12
    ema_long: int = 26

    # Momentum indicators
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    # Bollinger Bands
    bb_period: int = 20
    bb_std_dev: float = 2.0

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # ATR
    atr_period: int = 14

    # Stochastic
    stoch_period: int = 14
    stoch_smooth_k: int = 3
    stoch_smooth_d: int = 3


@dataclass
class ModelConfig:
    """
    Configuration for XGBoost models.
    """
    # Tree-based parameters
    max_depth: int = 6
    learning_rate: float = 0.05
    n_estimators: int = 200
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 1
    gamma: float = 0.0

    # Regularization
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0

    # Training
    train_test_split: float = 0.7
    early_stopping_rounds: int = 50
    eval_metric: str = 'rmse'
    seed: int = 42


@dataclass
class SignalGenerationConfig:
    """
    Configuration for signal generation logic.
    """
    # Signal thresholds (lower values = more signals)
    min_confidence: float = 0.40
    buy_signal_threshold: float = 0.45
    sell_signal_threshold: float = 0.45

    # Support/Resistance tolerance (% of price)
    price_tolerance_pct: float = 0.5

    # Technical indicator weights
    rsi_weight: float = 0.25
    macd_weight: float = 0.25
    bb_weight: float = 0.25
    atr_weight: float = 0.25

    # Risk management
    max_slippage_pct: float = 0.05
    max_commission_pct: float = 0.1


@dataclass
class DataConfig:
    """
    Configuration for data loading and preprocessing.
    """
    # HuggingFace dataset info
    hf_repo: str = 'zongowo111/v2-crypto-ohlcv-data'
    cache_dir: str = './data_cache'

    # Data preprocessing
    lookback_period: int = 100
    future_window: int = 5
    missing_value_fill: str = 'forward_fill'

    # Scaling
    scale_features: bool = True
    scaler_type: str = 'standard'


@dataclass
class ReversalConfig:
    """
    Configuration for binary reversal prediction using Swing High/Low detection.
    """
    lookback_window: int = 20
    atr_multiplier: float = 1.5
    profit_target_ratio: float = 1.5
    forward_window: int = 100
    min_trend_candles: int = 3
    volume_sma_period: int = 20
    swing_left_bars: int = 5   # Lookback bars for swing high/low detection
    swing_right_bars: int = 5  # Lookahead bars for swing high/low detection


@dataclass
class StrategyConfig:
    """
    Main configuration class combining all sub-configs.
    """
    # Sub-configurations
    indicators: TechnicalIndicatorConfig = field(default_factory=TechnicalIndicatorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    signals: SignalGenerationConfig = field(default_factory=SignalGenerationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    reversal: ReversalConfig = field(default_factory=ReversalConfig)

    # General settings
    verbose: bool = False
    debug: bool = False
    random_seed: int = 42
    num_threads: int = -1

    # Output paths
    model_save_dir: str = './models/v3'
    results_save_dir: str = './results/v3'

    # Convenience accessors for reversal config
    @property
    def lookback_window(self) -> int:
        return self.reversal.lookback_window

    @property
    def atr_multiplier(self) -> float:
        return self.reversal.atr_multiplier

    @property
    def profit_target_ratio(self) -> float:
        return self.reversal.profit_target_ratio

    @property
    def swing_left_bars(self) -> int:
        return self.reversal.swing_left_bars

    @property
    def swing_right_bars(self) -> int:
        return self.reversal.swing_right_bars

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        """
        return {
            'indicators': self.indicators.__dict__,
            'model': self.model.__dict__,
            'signals': self.signals.__dict__,
            'data': self.data.__dict__,
            'reversal': self.reversal.__dict__,
            'general': {
                'verbose': self.verbose,
                'debug': self.debug,
                'random_seed': self.random_seed,
                'num_threads': self.num_threads,
                'model_save_dir': self.model_save_dir,
                'results_save_dir': self.results_save_dir,
            }
        }

    @staticmethod
    def get_default() -> 'StrategyConfig':
        """
        Get default configuration.
        """
        return StrategyConfig()

    @staticmethod
    def get_conservative() -> 'StrategyConfig':
        """
        Get conservative configuration (stricter swing detection, fewer signals).
        """
        config = StrategyConfig()
        config.reversal.swing_left_bars = 7   # More bars to left
        config.reversal.swing_right_bars = 7  # More bars to right
        config.signals.min_confidence = 0.55
        config.signals.buy_signal_threshold = 0.60
        config.signals.sell_signal_threshold = 0.60
        return config

    @staticmethod
    def get_aggressive() -> 'StrategyConfig':
        """
        Get aggressive configuration (looser swing detection, more signals).
        """
        config = StrategyConfig()
        config.reversal.swing_left_bars = 3   # Fewer bars to left
        config.reversal.swing_right_bars = 3  # Fewer bars to right
        config.signals.min_confidence = 0.30
        config.signals.buy_signal_threshold = 0.35
        config.signals.sell_signal_threshold = 0.35
        return config
