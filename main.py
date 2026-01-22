import logging
from pathlib import Path
import joblib
import numpy as np

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
    """Complete training pipeline with regularization to prevent overfitting.
    
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
    X, y = preprocessor.preprocess(df, create_target=True, lookahead=3, normalize=True)
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Calculate class weights for imbalanced data
    from sklearn.utils.class_weight import compute_class_weight
    unique_classes = np.array([0, 1])
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y.values)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    logger.info(f"Class weights: {class_weight_dict}")
    
    logger.info("Step 3: Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        X, y, test_split=test_split, validation_split=validation_split, time_series=True
    )
    
    logger.info("Step 4: Training ensemble models...")
    
    # Conservative XGBoost parameters - prevent overfitting
    xgb_params = XGBOOST_PARAMS.copy()
    xgb_params.update({
        'max_depth': 5,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'gamma': 2,
        'reg_alpha': 1.0,
        'reg_lambda': 2.0,
        'scale_pos_weight': class_weight_dict[0] / class_weight_dict[1],
    })
    logger.info(f"XGBoost params: {xgb_params}")
    ensemble.train_xgboost(X_train, y_train, xgb_params)
    
    # Aggressively regularized LightGBM to prevent overfitting
    # Key: very shallow trees, high minimum leaf size, strong L1/L2 penalty
    lgb_params = LIGHTGBM_PARAMS.copy()
    lgb_params.update({
        'num_leaves': 15,  # Very shallow - fewer splits
        'learning_rate': 0.05,
        'n_estimators': 150,  # Fewer trees
        'subsample': 0.7,  # Less data per iteration
        'colsample_bytree': 0.7,  # Use fewer features per tree
        'min_data_in_leaf': 50,  # Much higher - require 50 samples per leaf
        'min_child_samples': 50,  # Explicit minimum
        'reg_alpha': 2.0,  # Strong L1 regularization
        'reg_lambda': 3.0,  # Strong L2 regularization
        'min_gain_to_split': 0.05,  # Need 5% gain to split
        'max_depth': 4,  # Shallow trees
        'feature_fraction': 0.7,  # Don't use all features
        'bagging_fraction': 0.7,  # Bagging
        'bagging_freq': 5,  # Bagging frequency
    })
    logger.info(f"LightGBM params: {lgb_params}")
    ensemble.train_lightgbm(X_train, y_train, lgb_params)
    
    # Smaller Neural Network with strong regularization
    nn_config = NEURAL_NETWORK_CONFIG.copy()
    nn_config.update({
        'lstm_units': 64,
        'lstm_layers': 2,
        'dense_units': 64,
        'dropout_rate': 0.5,
        'learning_rate': 0.001,
        'batch_size': batch_size,
        'epochs': epochs,
        'early_stopping_patience': 10,
    })
    logger.info(f"Neural Network config: {nn_config}")
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
    backtest_results = predictor.backtest_predictions(df.iloc[len(df) - len(X_test):], min_confidence=0.55)
    
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
    logger.info(f"Final test accuracy: {test_metrics['accuracy']:.2%}")
    logger.info(f"Final test F1-score: {test_metrics['f1']:.2%}")
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
