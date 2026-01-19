"""
Strategy V3: 7-Classifier Entry Signal System for 15m Crypto Trading

Core modules:
- StrategyConfig: Configuration management
- DataLoader: Load OHLCV data from HuggingFace
- FeatureEngineer: Generate technical features
"""

from .config import StrategyConfig
from .data_loader import DataLoader
from .feature_engineer import FeatureEngineer

__all__ = [
    'StrategyConfig',
    'DataLoader',
    'FeatureEngineer',
]

__version__ = '3.0.0'
