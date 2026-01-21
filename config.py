import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'cache'
MODEL_DIR = BASE_DIR / 'models' / 'artifacts'
LOG_DIR = BASE_DIR / 'logs'

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

HUGGINGFACE_CONFIG = {
    'repo_id': 'zongowo111/v2-crypto-ohlcv-data',
    'repo_type': 'dataset',
    'token': os.getenv('HF_TOKEN', ''),
}

DEFAULT_SYMBOL = 'BTCUSDT'
DEFAULT_TIMEFRAME = '15m'

TRAINING_CONFIG = {
    'symbol': DEFAULT_SYMBOL,
    'timeframe': DEFAULT_TIMEFRAME,
    'test_split': 0.2,
    'validation_split': 0.1,
    'random_state': 42,
}

XGBOOST_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'seed': 42,
}

LIGHTGBM_PARAMS = {
    'num_leaves': 31,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_data_in_leaf': 20,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'objective': 'binary',
    'metric': 'auc',
    'seed': 42,
}

NEURAL_NETWORK_CONFIG = {
    'input_shape': None,
    'lstm_units': 128,
    'lstm_layers': 2,
    'dense_units': 64,
    'dense_layers': 2,
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 10,
    'validation_split': 0.2,
}

FEATURE_CONFIG = {
    'sma_periods': [5, 10, 20, 50, 200],
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bollinger_period': 20,
    'bollinger_std': 2,
    'atr_period': 14,
    'stoch_period': 14,
    'stoch_smooth': 3,
    'obv_smooth': 20,
    'adl_period': 20,
}

VOLATILITY_CONFIG = {
    'short_window': 5,
    'long_window': 20,
    'expand_threshold': 1.5,
    'squeeze_threshold': 0.7,
}

MICROSTRUCTURE_CONFIG = {
    'order_flow_window': 20,
    'volume_profile_bins': 10,
    'aggressive_ratio_smooth': 10,
    'accumulation_threshold': 0.65,
}

REGIME_CONFIG = {
    'momentum_threshold_up': 0.55,
    'momentum_threshold_down': 0.45,
    'volume_threshold': 1.2,
    'volatility_threshold': 1.3,
}

BACKTEST_CONFIG = {
    'initial_capital': 10000,
    'position_size_percent': 1.0,
    'slippage_percent': 0.05,
    'commission_percent': 0.1,
    'min_confidence': 0.60,
    'take_profit_percent': 1.0,
    'stop_loss_percent': 0.5,
}

API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'reload': True,
    'workers': 4,
}

DASHBOARD_CONFIG = {
    'page_title': 'Crypto Reversal Prediction System',
    'page_icon': ':chart_with_upwards_trend:',
    'layout': 'wide',
    'theme': 'dark',
}

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': str(LOG_DIR / 'app.log'),
}
