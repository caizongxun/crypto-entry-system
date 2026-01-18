#!/usr/bin/env python3
"""
Main Script for Strategy V3 - Advanced Cryptocurrency Entry Signal System

Usage:
    python main_v3.py --mode train --symbol BTCUSDT --timeframe 15m
    python main_v3.py --mode predict --symbol BTCUSDT --timeframe 15m
    python main_v3.py --mode backtest --symbol BTCUSDT --timeframe 15m
"""

import argparse
import logging
from pathlib import Path
import sys
import warnings

import pandas as pd
import numpy as np

from strategy_v3 import (
    Config,
    DataLoader,
    FeatureEngineer,
    MultiOutputModel,
    SignalGenerator
)


# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/strategy_v3.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def ensure_dirs(config: Config) -> None:
    """Create necessary directories"""
    Path(config.model_dir).mkdir(parents=True, exist_ok=True)
    Path(config.result_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(parents=True, exist_ok=True)


def create_targets(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Create target variables for ML model:
    1. Support level: lowest price in next lookback periods
    2. Resistance level: highest price in next lookback periods
    3. Breakout probability: 1 if price breaks resistance, 0 otherwise
    
    Args:
        df: DataFrame with OHLCV data and technical indicators
        lookback: Number of periods to look ahead
        
    Returns:
        DataFrame with targets
    """
    df = df.copy()
    
    # Forward looking - get future low/high
    df['future_low'] = df['low'].rolling(window=lookback).min().shift(-lookback)
    df['future_high'] = df['high'].rolling(window=lookback).max().shift(-lookback)
    
    # Support: 0.98 * current low (normalized)
    df['target_support'] = df['future_low'] / df['close']
    
    # Resistance: 1.02 * current high (normalized)
    df['target_resistance'] = df['future_high'] / df['close']
    
    # Breakout probability: binary indicator
    # 1 if high > current price * 1.02, 0 otherwise
    df['target_breakout'] = ((df['future_high'] / df['close']) > 1.02).astype(float)
    
    # Remove NaN
    df = df.dropna()
    
    return df[['target_support', 'target_resistance', 'target_breakout']]


def train_mode(config: Config) -> None:
    """
    Train the model on historical data
    
    Args:
        config: Configuration object
    """
    logger.info("Starting training mode...")
    logger.info(f"Symbol: {config.symbol}, Timeframe: {config.timeframe}")
    
    # Load data
    loader = DataLoader(config)
    df = loader.load_klines(config.symbol, config.timeframe)
    
    # Validate data
    is_valid, message = loader.validate_data(df)
    if not is_valid:
        logger.error(f"Data validation failed: {message}")
        return
    
    logger.info(f"Loaded {len(df)} records")
    
    # Engineer features
    engineer = FeatureEngineer(config)
    df_features, _ = engineer.engineer_all_features(df)
    
    logger.info(f"Engineered features: {len([c for c in df_features.columns if c not in ['open', 'high', 'low', 'close', 'volume']])}")
    
    # Create targets
    targets = create_targets(df_features, lookback=10)
    
    # Align features and targets
    valid_idx = targets.index
    X = df_features.loc[valid_idx].copy()
    y = targets.copy()
    
    logger.info(f"Training set size: {len(X)}")
    
    # Select features
    feature_engineer = FeatureEngineer(config)
    feature_cols = [c for c in X.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'close_time']]
    X_features = X[feature_cols].copy()
    
    X_selected = feature_engineer.select_features(X_features, y.iloc[:, 0], n_features=25)
    
    # Build and train models
    model = MultiOutputModel(config)
    model.build_models()
    
    # Use selected features for training
    X_train_selected = pd.DataFrame(
        X_selected,
        index=X.index,
        columns=feature_engineer.selected_features if feature_engineer.selected_features else feature_cols[:25]
    )
    
    results = model.train(X_train_selected, y, verbose=True)
    
    # Print training summary
    logger.info(model.get_training_summary())
    
    # Save models
    model.save_models(config.model_dir)
    logger.info(f"Models saved to {config.model_dir}")
    
    # Save feature importance
    importance = model.get_feature_importance()
    logger.info("\nTop features for each target:")
    for target, features in importance.items():
        logger.info(f"\n{target}:")
        for feat, imp in list(features.items())[:5]:
            logger.info(f"  {feat}: {imp:.6f}")
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(Path(config.result_dir) / f"{config.symbol}_{config.timeframe}_training_results.csv")
    logger.info(f"Results saved to {config.result_dir}")


def predict_mode(config: Config) -> None:
    """
    Make predictions on latest data
    
    Args:
        config: Configuration object
    """
    logger.info("Starting prediction mode...")
    logger.info(f"Symbol: {config.symbol}, Timeframe: {config.timeframe}")
    
    # Load data
    loader = DataLoader(config)
    df = loader.load_klines(config.symbol, config.timeframe)
    df = loader.get_latest_bars(df, n_bars=200)
    
    logger.info(f"Loaded {len(df)} records for prediction")
    
    # Engineer features
    engineer = FeatureEngineer(config)
    df_features, _ = engineer.engineer_all_features(df)
    
    # Load trained model
    model = MultiOutputModel(config)
    model.load_models(config.model_dir)
    
    # Get feature columns
    feature_cols = [c for c in df_features.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'close_time']]
    X = df_features[feature_cols].copy()
    
    # Make predictions
    predictions = model.predict(X)
    
    # Generate signals
    signal_gen = SignalGenerator(config)
    current_price = df.iloc[-1]['close']
    
    signals = signal_gen.generate_signals(df_features, predictions, float(current_price))
    
    logger.info(signal_gen.get_signal_summary())
    
    # Save signals
    signals_df = signal_gen.get_signals_dataframe()
    if not signals_df.empty:
        signals_df.to_csv(Path(config.result_dir) / f"{config.symbol}_{config.timeframe}_latest_signals.csv")
        logger.info(f"Signals saved to {config.result_dir}")
        logger.info("\nLatest signals:")
        logger.info(signals_df.to_string())


def backtest_mode(config: Config) -> None:
    """
    Backtest strategy on historical data
    
    Args:
        config: Configuration object
    """
    logger.info("Starting backtest mode...")
    logger.info(f"Symbol: {config.symbol}, Timeframe: {config.timeframe}")
    
    # Load data
    loader = DataLoader(config)
    df = loader.load_klines(config.symbol, config.timeframe)
    
    # Split data
    train_df, test_df = loader.split_train_test(df, train_ratio=0.7)
    
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Engineer features on train data
    engineer = FeatureEngineer(config)
    train_features, _ = engineer.engineer_all_features(train_df)
    
    # Create targets
    targets = create_targets(train_features, lookback=10)
    
    # Align
    valid_idx = targets.index
    X_train = train_features.loc[valid_idx].copy()
    y_train = targets.copy()
    
    # Train model
    logger.info("Training model on train set...")
    model = MultiOutputModel(config)
    model.build_models()
    
    feature_cols = [c for c in X_train.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'close_time']]
    X_train_features = X_train[feature_cols].copy()
    
    results = model.train(X_train_features, y_train, verbose=False)
    logger.info("Training completed")
    
    # Test on unseen data
    logger.info("Generating predictions on test set...")
    test_features, _ = engineer.engineer_all_features(test_df)
    
    X_test = test_features[feature_cols].copy()
    X_test = X_test.dropna()
    
    predictions = model.predict(X_test)
    
    # Generate signals on test set
    signal_gen = SignalGenerator(config)
    all_signals = []
    
    for idx in range(len(X_test)):
        current_price = test_features.iloc[idx]['close']
        current_predictions = predictions[idx:idx+1]
        df_slice = test_features.iloc[idx:idx+1]
        
        signals = signal_gen.generate_signals(df_slice, current_predictions, float(current_price))
        all_signals.extend(signals)
    
    logger.info(signal_gen.get_signal_summary())
    
    # Calculate basic backtest metrics
    if all_signals:
        buy_signals = [s for s in all_signals if s.signal_type == 'BUY']
        sell_signals = [s for s in all_signals if s.signal_type == 'SELL']
        
        if buy_signals and sell_signals:
            # Simple PnL calculation
            total_pnl = 0
            for buy_sig in buy_signals:
                # Find closest sell after this buy
                future_sells = [s for s in sell_signals if s.timestamp > buy_sig.timestamp]
                if future_sells:
                    sell_sig = future_sells[0]
                    pnl_pct = (sell_sig.entry_price - buy_sig.entry_price) / buy_sig.entry_price
                    total_pnl += pnl_pct
            
            logger.info(f"\nBacktest Results:")
            logger.info(f"Total Buy Signals: {len(buy_signals)}")
            logger.info(f"Total Sell Signals: {len(sell_signals)}")
            logger.info(f"Estimated Total PnL: {total_pnl:.2%}")
            if len(buy_signals) > 0:
                logger.info(f"Average PnL per trade: {total_pnl/len(buy_signals):.2%}")
    
    # Save backtest signals
    signals_df = signal_gen.get_signals_dataframe()
    if not signals_df.empty:
        signals_df.to_csv(Path(config.result_dir) / f"{config.symbol}_{config.timeframe}_backtest_signals.csv")
        logger.info(f"Backtest signals saved to {config.result_dir}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Strategy V3 - Advanced Cryptocurrency Entry Signal System')
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict', 'backtest'],
                        help='Mode to run: train, predict, or backtest')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='Trading symbol (e.g., BTCUSDT, ETHUSDT)')
    parser.add_argument('--timeframe', type=str, default='15m', choices=['15m', '1h', '1d'],
                        help='Timeframe (15m, 1h, 1d)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Create config
    config = Config(
        symbol=args.symbol,
        timeframe=args.timeframe,
        verbose=args.verbose
    )
    
    # Ensure directories
    ensure_dirs(config)
    
    try:
        if args.mode == 'train':
            train_mode(config)
        elif args.mode == 'predict':
            predict_mode(config)
        elif args.mode == 'backtest':
            backtest_mode(config)
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        sys.exit(1)
    
    logger.info(f"Completed {args.mode} mode")


if __name__ == '__main__':
    main()
