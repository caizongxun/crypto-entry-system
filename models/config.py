import os
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / 'models' / 'cache'
DATA_DIR = CACHE_DIR / 'data'

CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

HUGGINGFACE_CONFIG = {
    'repo_id': 'zongowo111/v2-crypto-ohlcv-data',
    'repo_type': 'dataset',
    'base_path': 'klines',
}

DEFAULT_SYMBOL = 'BTCUSDT'
DEFAULT_TIMEFRAME = '15m'

TIMEFRAMES: List[str] = ['15m', '1h', '1d']

CRYPTO_SYMBOLS: List[str] = [
    'AAVEUSDT', 'ADAUSDT', 'ALGOUSDT', 'ARBUSDT', 'ATOMUSDT',
    'AVAXUSDT', 'BALUSDT', 'BATUSDT', 'BCHUSDT', 'BNBUSDT',
    'BTCUSDT', 'COMPUSDT', 'CRVUSDT', 'DOGEUSDT', 'DOTUSDT',
    'ENJUSDT', 'ENSUSDT', 'ETCUSDT', 'ETHUSDT', 'FILUSDT',
    'GALAUSDT', 'GRTUSDT', 'IMXUSDT', 'KAVAUSDT', 'LINKUSDT',
    'LTCUSDT', 'MANAUSDT', 'MATICUSDT', 'MKRUSDT', 'NEARUSDT',
    'OPUSDT', 'SANDUSDT', 'SNXUSDT', 'SOLUSDT', 'SPELLUSDT',
    'UNIUSDT', 'XRPUSDT', 'ZRXUSDT'
]

ML_CONFIG = {
    'lookback_period': 50,
    'train_test_split': 0.8,
    'random_state': 42,
    'test_size': 0.2,
    'validation_size': 0.1,
}

TECHNICAL_INDICATORS_CONFIG: Dict[str, Dict] = {
    'sma_fast': {'period': 20},
    'sma_medium': {'period': 50},
    'sma_slow': {'period': 200},
    'ema_fast': {'period': 12},
    'ema_slow': {'period': 26},
    'rsi': {'period': 14},
    'macd': {'fast': 12, 'slow': 26, 'signal': 9},
    'bb': {'period': 20, 'std_dev': 2},
    'atr': {'period': 14},
}

FEATURE_ENGINEERING_CONFIG = {
    'normalize_features': True,
    'handle_missing': 'forward_fill',
    'outlier_method': 'iqr',
}

MODEL_CONFIG = {
    'model_type': 'xgboost',
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 5,
        'random_state': 42,
    },
}

EVALUATION_CONFIG = {
    'quality_score_threshold': 65,
    'momentum_weight': 0.25,
    'volatility_weight': 0.2,
    'trend_weight': 0.3,
    'rr_ratio_weight': 0.25,
}

SIGNAL_QUALITY_RANGES = {
    'excellent': (80, 100),
    'good': (65, 79),
    'moderate': (50, 64),
    'poor': (0, 49),
}

TIMEFRAME_CONFIGS = {
    '15m': {
        'bb_period': 20,
        'bb_std': 2.0,
        'lookforward': 5,
        'bounce_threshold': 0.005,
        'model_type': 'xgboost',
    },
    '1h': {
        'bb_period': 20,
        'bb_std': 2.0,
        'lookforward': 4,
        'bounce_threshold': 0.005,
        'model_type': 'xgboost',
    },
    '1d': {
        'bb_period': 20,
        'bb_std': 2.0,
        'lookforward': 3,
        'bounce_threshold': 0.01,
        'model_type': 'xgboost',
    },
}
