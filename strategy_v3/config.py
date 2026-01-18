"""
Configuration settings for Strategy V3
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class HuggingFaceConfig:
    """HuggingFace dataset configuration"""
    repo_id: str = "zongowo111/v2-crypto-ohlcv-data"
    repo_type: str = "dataset"
    cache_dir: str = ".cache/huggingface"


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    # Technical indicator periods
    sma_periods: List[int] = None
    ema_periods: List[int] = None
    rsi_period: int = 14
    rsi_threshold_high: float = 70.0
    rsi_threshold_low: float = 30.0
    
    # MACD parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Bollinger Bands
    bb_period: int = 20
    bb_std_dev: float = 2.0
    
    # ATR (Average True Range)
    atr_period: int = 14
    
    # Volume parameters
    volume_ma_period: int = 20
    
    # Multi-timeframe
    timeframes: List[str] = None
    
    def __post_init__(self):
        if self.sma_periods is None:
            self.sma_periods = [5, 10, 20, 50, 200]
        if self.ema_periods is None:
            self.ema_periods = [5, 12, 26]
        if self.timeframes is None:
            self.timeframes = ['15m', '1h', '1d']


@dataclass
class ModelConfig:
    """XGBoost model configuration"""
    # Multi-output targets
    # 0: support level, 1: resistance level, 2: breakout probability
    n_targets: int = 3
    
    # XGBoost parameters
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 1
    gamma: float = 0.0
    lambda_reg: float = 1.0  # L2 regularization
    alpha_reg: float = 0.0   # L1 regularization
    
    # Training
    train_test_split: float = 0.7
    random_state: int = 42
    eval_metric: str = 'rmse'
    
    # Early stopping
    early_stopping_rounds: int = 10
    eval_set_size: float = 0.1
    
    # Device
    tree_method: str = 'hist'  # 'exact', 'approx', 'hist', 'gpu_hist'
    gpu_id: int = 0
    n_jobs: int = -1  # -1 means use all processors
    
    # Multi-output strategy
    multi_strategy: str = 'one_output_per_tree'  # or 'multi_output_tree'


@dataclass
class SignalConfig:
    """Trading signal configuration"""
    # Signal generation thresholds
    support_resistance_tolerance: float = 0.005  # 0.5% tolerance
    breakout_probability_threshold: float = 0.6  # 60% confidence
    
    # Entry conditions (all must be met for strong signal)
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    
    # Signal strength multipliers
    high_confidence_threshold: float = 0.8
    medium_confidence_threshold: float = 0.6
    low_confidence_threshold: float = 0.4
    
    # Risk management
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 3.0
    
    # Position sizing
    max_position_size: float = 0.1  # Max 10% of capital per trade
    

@dataclass
class BacktestConfig:
    """Backtest configuration"""
    initial_capital: float = 10000.0
    commission: float = 0.001  # 0.1% commission
    slippage: float = 0.0005   # 0.05% slippage
    
    # Backtest period
    train_period: str = '90d'  # Training period
    test_period: str = '30d'   # Testing period
    
    # Performance metrics
    calculate_sharpe: bool = True
    calculate_sortino: bool = True
    calculate_calmar: bool = True
    risk_free_rate: float = 0.02  # 2% annual
    

@dataclass
class Config:
    """Main configuration class"""
    
    # Data settings
    symbol: str = 'BTCUSDT'
    timeframe: str = '15m'
    
    # Sub-configurations
    huggingface: HuggingFaceConfig = None
    feature: FeatureConfig = None
    model: ModelConfig = None
    signal: SignalConfig = None
    backtest: BacktestConfig = None
    
    # Logging
    verbose: bool = True
    log_level: str = 'INFO'  # DEBUG, INFO, WARNING, ERROR
    
    # Output paths
    model_dir: str = 'models/v3'
    result_dir: str = 'results/v3'
    log_dir: str = 'logs/v3'
    
    def __post_init__(self):
        if self.huggingface is None:
            self.huggingface = HuggingFaceConfig()
        if self.feature is None:
            self.feature = FeatureConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.signal is None:
            self.signal = SignalConfig()
        if self.backtest is None:
            self.backtest = BacktestConfig()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary"""
        return cls(**config_dict)
