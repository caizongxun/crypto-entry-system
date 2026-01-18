"""
Strategy V3 Package - Advanced cryptocurrency entry signal system
Combining XGBoost multi-output prediction with technical indicators
"""

from .data_loader import DataLoader
from .feature_engineer import FeatureEngineer
from .models import MultiOutputModel
from .signal_generator import SignalGenerator
from .config import Config

__version__ = '3.0.0'
__all__ = [
    'DataLoader',
    'FeatureEngineer',
    'MultiOutputModel',
    'SignalGenerator',
    'Config',
]
