import logging
from pathlib import Path
import joblib

from config import TRAINING_CONFIG, XGBOOST_PARAMS, LIGHTGBM_PARAMS, NEURAL_NETWORK_CONFIG, FEATURE_CONFIG
from data.loader import KlinesDataLoader
from data.preprocessor import DataPreprocessor
from models.ensemble import EnsembleModel
from models.predictor import UnifiedPredictor
from backtest.engine import BacktestEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_full_pipeline(
    symbol: str = 'BTCUSDT',
    timeframe: str = '15m',
    test_split: float = 0.2,
    validation_split: float = 0.1,
    epochs: int = 100,
    batch_size: int = 32
):
    """Complete training pipeline.
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        timeframe: '15m', '1h', or '1d'
        test_split: Test set ratio
        validation_split: Validation set ratio
        epochs: Neural network epochs
        batch_size: Neural network batch size
    
    Returns:
        Dict with model artifacts
    """
    logger.info(f"Starting training pipeline for {symbol} {timeframe}")
    
    model_dir = Path('models/artifacts')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    data_loader = KlinesDataLoader()
    preprocessor = DataPreprocessor(config=FEATURE_CONFIG)
    ensemble = EnsembleModel(model_dir=model_dir)
    
    logger.info("Step 1: Loading data...")
    df = data_loader.load_klines(symbol, timeframe, use_cache=True)
    if not data_loader.validate_data(df):
        logger.error("Data validation failed")
        return None
    logger.info(f"Loaded {len(df)} candles")
    
    logger.info("Step 2: Preprocessing and feature engineering...")
    X, y = preprocessor.preprocess(df, create_target=True, normalize=True)
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    logger.info("Step 3: Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        X, y, test_split=test_split, validation_split=validation_split, time_series=True
    )
    
    logger.info("Step 4: Training ensemble models...")
    
    ensemble.train_xgboost(X_train, y_train, XGBOOST_PARAMS)
    ensemble.train_lightgbm(X_train, y_train, LIGHTGBM_PARAMS)
    
    nn_config = NEURAL_NETWORK_CONFIG.copy()
    nn_config['epochs'] = epochs
    nn_config['batch_size'] = batch_size
    ensemble.train_neural_network(X_train.values, y_train.values, nn_config)
    
    ensemble.train_logistic_regression(X_train, y_train)
    
    logger.info("Step 5: Validation metrics...")
    val_metrics = ensemble.evaluate(X_val, y_val)
    logger.info(f"Validation metrics: {val_metrics}")
    
    logger.info("Step 6: Test metrics...")
    test_metrics = ensemble.evaluate(X_test, y_test)
    logger.info(f"Test metrics: {test_metrics}")
    
    logger.info("Step 7: Saving models...")
    ensemble.save()
    joblib.dump(preprocessor.scaler, model_dir / 'scaler.pkl')
    
    logger.info("Step 8: Backtesting...")
    predictor = UnifiedPredictor(model_dir=model_dir)
    
    predictions_test = predictor.predict_multiple_candles(df.iloc[len(df) - len(X_test):])
    backtest_results = predictor.backtest_predictions(df.iloc[len(df) - len(X_test):], min_confidence=0.60)
    
    logger.info(f"Backtest results: {backtest_results}")
    
    artifacts = {
        'ensemble': ensemble,
        'preprocessor': preprocessor,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'backtest_results': backtest_results,
        'feature_names': preprocessor.feature_names,
    }
    
    logger.info("Training complete!")
    return artifacts

if __name__ == '__main__':
    train_full_pipeline(
        symbol='BTCUSDT',
        timeframe='15m',
        test_split=0.2,
        validation_split=0.1,
        epochs=100,
        batch_size=32
    )
