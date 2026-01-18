"""
Strategy V3: Advanced Cryptocurrency Entry Signal System

A machine learning-based cryptocurrency entry signal system combining XGBoost
multi-output regression with comprehensive technical indicators.
"""

__version__ = '3.0.0'
__author__ = 'Trading Strategy Team'

from strategy_v3.config import StrategyConfig
from strategy_v3.data_loader import DataLoader
from strategy_v3.feature_engineer import FeatureEngineer
from strategy_v3.models import ModelEnsemble
from strategy_v3.signal_generator import SignalGenerator

__all__ = [
    'StrategyConfig',
    'DataLoader',
    'FeatureEngineer',
    'ModelEnsemble',
    'SignalGenerator',
]
