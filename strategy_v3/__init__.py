"""
Strategy V3: Binary Reversal Prediction with Historical Lookback

Core modules:
- StrategyConfig: Configuration management
- DataLoader: Load OHLCV data from HuggingFace
- FeatureEngineer: Generate technical features
- create_reversal_target: Create binary reversal targets with HOLD labeling
"""

from .config import StrategyConfig
from .data_loader import DataLoader
from .feature_engineer import FeatureEngineer
from .targets import create_reversal_target

__all__ = [
    'StrategyConfig',
    'DataLoader',
    'FeatureEngineer',
    'create_reversal_target',
]

__version__ = '3.0.0'
