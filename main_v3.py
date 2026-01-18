#!/usr/bin/env python3
"""
Main training and prediction script for Strategy V3.

Usage:
    python main_v3.py --mode train --symbol BTCUSDT --timeframe 15m
    python main_v3.py --mode predict --symbol BTCUSDT --timeframe 15m
    python main_v3.py --mode backtest --symbol BTCUSDT --timeframe 15m
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger

from strategy_v3 import (
    StrategyConfig,
    DataLoader,
    FeatureEngineer,
    ModelEnsemble,
    SignalGenerator,
)


def setup_logging(verbose: bool = False):
    """
    Configure logging.
    """
    log_level = 'DEBUG' if verbose else 'INFO'
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}'
    )


def _create_regression_targets(df: pd.DataFrame, lookback: int = 20, forward_lookback: int = 20) -> tuple:
    """
    Create regression targets by predicting future support/resistance.
    
    Key insight: Shift targets forward so we predict FUTURE price extremes,
    not past ones. This avoids data leakage.
    
    Args:
        df: DataFrame with OHLCV data
        lookback: Number of periods to calculate current support/resistance
        forward_lookback: Number of periods ahead to predict support/resistance
        
    Returns:
        Tuple of (current_support, current_resistance, future_support, future_resistance, direction)
    """
    # Current support/resistance (calculated from past lookback periods)
    current_support = df['low'].rolling(window=lookback).min()
    current_resistance = df['high'].rolling(window=lookback).max()
    
    # Future support/resistance (what will be the extremes in next forward_lookback periods)
    # This is the target we want to predict
    future_support = df['low'].rolling(window=forward_lookback).min().shift(-forward_lookback)
    future_resistance = df['high'].rolling(window=forward_lookback).max().shift(-forward_lookback)
    
    # Price direction in forward window
    future_close = df['close'].shift(-forward_lookback)
    direction = np.where(future_close > df['close'], 1, -1)
    
    return current_support, current_resistance, future_support, future_resistance, direction


def train_model(args):
    """
    Train the strategy model.
    """
    logger.info('Starting model training...')
    logger.info(f'Symbol: {args.symbol}, Timeframe: {args.timeframe}')

    # Initialize configuration
    config = StrategyConfig.get_default()
    config.verbose = args.verbose

    # Create output directories
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.results_save_dir, exist_ok=True)

    # Load data
    logger.info('Loading data...')
    loader = DataLoader(
        hf_repo=config.data.hf_repo,
        cache_dir=config.data.cache_dir,
        verbose=args.verbose
    )

    try:
        df = loader.load_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            cache=True
        )
    except Exception as e:
        logger.error(f'Failed to load data: {str(e)}')
        return False

    # Validate data
    if not loader.validate_data(df):
        logger.error('Data validation failed')
        return False

    logger.info(f'Loaded {len(df)} candles')

    # Engineer features on full dataset
    logger.info('Engineering features...')
    engineer = FeatureEngineer(config)
    df_features = engineer.engineer_features(df)

    logger.info(f'Total features: {len(df_features.columns) - 5}')

    # Create targets: predict FUTURE support/resistance, not current ones
    logger.info('Creating targets (predicting future support/resistance)...')
    current_support, current_resistance, future_support, future_resistance, direction = _create_regression_targets(
        df, lookback=20, forward_lookback=20
    )
    
    # Remove rows with NaN - this creates clean aligned data
    mask = future_support.notna() & future_resistance.notna()
    df_clean = df_features[mask].copy()
    future_support_clean = future_support[mask]
    future_resistance_clean = future_resistance[mask]
    direction_clean = direction[mask]

    logger.info(f'Clean data: {len(df_clean)} samples')

    # Now split the CLEAN data into train/test
    logger.info('Splitting clean data into train/test sets...')
    split_idx = int(len(df_clean) * 0.7)
    
    df_train = df_clean.iloc[:split_idx].copy()
    df_test = df_clean.iloc[split_idx:].copy()
    
    y_train_support = future_support_clean.iloc[:split_idx]
    y_train_resistance = future_resistance_clean.iloc[:split_idx]
    y_train_breakout = (future_resistance_clean.iloc[:split_idx] > df_train['close']).astype(float)
    
    y_test_support = future_support_clean.iloc[split_idx:]
    y_test_resistance = future_resistance_clean.iloc[split_idx:]
    y_test_breakout = (future_resistance_clean.iloc[split_idx:] > df_test['close']).astype(float)

    logger.info(f'Training samples: {len(df_train)}')
    logger.info(f'Testing samples: {len(df_test)}')
    logger.info(f'Average support prediction (train): {y_train_support.mean():.2f}')
    logger.info(f'Average resistance prediction (train): {y_train_resistance.mean():.2f}')
    logger.info(f'Current price range: {df["close"].mean():.2f}')

    # Get feature columns (exclude OHLCV)
    feature_cols = [col for col in df_clean.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]

    # Train models
    logger.info('Training models...')
    ensemble = ModelEnsemble(config)

    metrics = ensemble.train(
        X_train=X_train,
        y_train_support=y_train_support,
        y_train_resistance=y_train_resistance,
        y_train_breakout=y_train_breakout,
        X_test=X_test,
        y_test_support=y_test_support,
        y_test_resistance=y_test_resistance,
        y_test_breakout=y_test_breakout
    )

    # Save models
    ensemble.save_models(config.model_save_dir)

    # Save training results
    results_file = os.path.join(
        config.results_save_dir,
        f'{args.symbol}_{args.timeframe}_training_results.csv'
    )
    
    results_data = []
    for model_name, model_metrics in metrics.items():
        if isinstance(model_metrics.get('test_r2'), (int, float)):
            results_data.append({
                'model': model_name,
                'train_rmse': model_metrics['train_rmse'],
                'train_mae': model_metrics['train_mae'],
                'train_r2': model_metrics['train_r2'],
                'test_rmse': model_metrics['test_rmse'],
                'test_mae': model_metrics['test_mae'],
                'test_r2': model_metrics['test_r2'],
                'overfitting_gap': model_metrics['train_r2'] - model_metrics['test_r2'],
                'timestamp': datetime.now().isoformat()
            })
        else:
            results_data.append({
                'model': model_name,
                'train_rmse': model_metrics['rmse'],
                'train_mae': model_metrics['mae'],
                'train_r2': model_metrics['r2'],
                'test_rmse': None,
                'test_mae': None,
                'test_r2': None,
                'overfitting_gap': None,
                'timestamp': datetime.now().isoformat()
            })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(results_file, index=False)

    logger.info(f'Training results saved to {results_file}')
    logger.info('Model training completed successfully')

    # Print summary
    logger.info('Training Summary:')
    for idx, row in results_df.iterrows():
        logger.info(f"\n{row['model'].upper()}:")
        logger.info(f"  Train R2: {row['train_r2']:.6f}")
        if pd.notna(row['test_r2']):
            logger.info(f"  Test R2:  {row['test_r2']:.6f}")
            logger.info(f"  Overfitting Gap: {row['overfitting_gap']:.6f}")
            if row['overfitting_gap'] > 0.2:
                logger.warning(f"  WARNING: Significant overfitting detected!")
            elif row['overfitting_gap'] > 0.1:
                logger.info(f"  Status: Mild overfitting")
            else:
                logger.info(f"  Status: Healthy model")
        logger.info(f"  Test MAE: {row['test_mae']:.4f} (avg prediction error)")

    return True


def predict_signals(args):
    """
    Generate trading signals on latest data.
    """
    logger.info('Generating trading signals...')
    logger.info(f'Symbol: {args.symbol}, Timeframe: {args.timeframe}')

    # Initialize configuration
    config = StrategyConfig.get_default()
    config.verbose = args.verbose

    # Load data
    logger.info('Loading data...')
    loader = DataLoader(
        hf_repo=config.data.hf_repo,
        cache_dir=config.data.cache_dir,
        verbose=args.verbose
    )

    try:
        df = loader.load_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            cache=True
        )
    except Exception as e:
        logger.error(f'Failed to load data: {str(e)}')
        return False

    # Engineer features
    logger.info('Engineering features...')
    engineer = FeatureEngineer(config)
    df_features = engineer.engineer_features(df)

    # Load models
    logger.info('Loading trained models...')
    ensemble = ModelEnsemble(config)

    try:
        ensemble.load_models(config.model_save_dir)
    except FileNotFoundError:
        logger.error('Models not found. Please train the model first.')
        return False

    # Generate predictions
    logger.info('Generating predictions...')
    feature_cols = ensemble.selected_features
    X = df_features[feature_cols]

    support, resistance, breakout_prob = ensemble.predict(X)

    # Generate signals
    logger.info('Creating signals...')
    signal_gen = SignalGenerator(config)
    signals_df = signal_gen.generate_signals(df_features, support, resistance, breakout_prob)

    # Get summary
    summary = signal_gen.get_signal_summary(signals_df)

    logger.info('Signal Summary:')
    logger.info(f'  Total Candles: {summary["total_candles"]}')
    logger.info(f'  Buy Signals: {summary["buy_signals"]}')
    logger.info(f'  Sell Signals: {summary["sell_signals"]}')
    logger.info(f'  Signal Density: {summary["signal_density"]:.2%}')
    logger.info(f'  Avg Buy Confidence: {summary["avg_buy_confidence"]:.2%}')
    logger.info(f'  Avg Sell Confidence: {summary["avg_sell_confidence"]:.2%}')

    # Save signals
    os.makedirs(config.results_save_dir, exist_ok=True)
    signals_file = os.path.join(
        config.results_save_dir,
        f'{args.symbol}_{args.timeframe}_latest_signals.csv'
    )

    # Keep only relevant columns
    output_cols = ['open', 'high', 'low', 'close', 'volume', 'support', 'resistance',
                   'breakout_prob', 'buy_confidence', 'sell_confidence', 'signal_type']
    signals_df[output_cols].to_csv(signals_file)

    logger.info(f'Signals saved to {signals_file}')

    # Print recent signals
    recent_signals = signals_df[signals_df['signal_type'] != 'HOLD'].tail(10)
    if len(recent_signals) > 0:
        logger.info('Recent signals:')
        for idx, (i, row) in enumerate(recent_signals.iterrows()):
            logger.info(f"  {row.name.strftime('%Y-%m-%d %H:%M')} - {row['signal_type']}: "
                       f"Price {row['close']:.2f}, Conf {max(row['buy_confidence'], row['sell_confidence']):.2%}")

    logger.info('Signal generation completed successfully')
    return True


def backtest_strategy(args):
    """
    Backtest the strategy on unseen test data.
    """
    logger.info('Running strategy backtest...')
    logger.info(f'Symbol: {args.symbol}, Timeframe: {args.timeframe}')

    # Initialize configuration
    config = StrategyConfig.get_default()
    config.verbose = args.verbose

    # Load data
    logger.info('Loading data...')
    loader = DataLoader(
        hf_repo=config.data.hf_repo,
        cache_dir=config.data.cache_dir,
        verbose=args.verbose
    )

    try:
        df = loader.load_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            cache=True
        )
    except Exception as e:
        logger.error(f'Failed to load data: {str(e)}')
        return False

    # Engineer features
    logger.info('Engineering features...')
    engineer = FeatureEngineer(config)
    df_features = engineer.engineer_features(df)

    # Create targets
    current_support, current_resistance, future_support, future_resistance, direction = _create_regression_targets(
        df, lookback=20, forward_lookback=20
    )

    # Remove NaN
    mask = future_support.notna() & future_resistance.notna()
    df_clean = df_features[mask].copy()
    future_support = future_support[mask]
    future_resistance = future_resistance[mask]

    # Split into train/test
    split_idx = int(len(df_clean) * 0.7)
    df_test = df_clean.iloc[split_idx:].copy()
    
    y_test_support = future_support.iloc[split_idx:]
    y_test_resistance = future_resistance.iloc[split_idx:]
    y_test_breakout = (future_resistance.iloc[split_idx:] > df_test['close']).astype(float)

    # Load models
    logger.info('Loading trained models...')
    ensemble = ModelEnsemble(config)

    try:
        ensemble.load_models(config.model_save_dir)
    except FileNotFoundError:
        logger.error('Models not found. Please train the model first.')
        return False

    # Generate predictions on test set
    logger.info('Generating backtest predictions...')
    feature_cols = ensemble.selected_features
    X_test = df_test[feature_cols]

    support_pred, resistance_pred, breakout_pred = ensemble.predict(X_test)

    # Generate signals
    logger.info('Creating backtest signals...')
    signal_gen = SignalGenerator(config)
    signals_df = signal_gen.generate_signals(df_test, support_pred, resistance_pred, breakout_pred)

    # Get summary
    summary = signal_gen.get_signal_summary(signals_df)

    logger.info('Backtest Summary:')
    logger.info(f'  Total Candles: {summary["total_candles"]}')
    logger.info(f'  Buy Signals: {summary["buy_signals"]}')
    logger.info(f'  Sell Signals: {summary["sell_signals"]}')
    logger.info(f'  Signal Density: {summary["signal_density"]:.2%}')

    # Save backtest results
    os.makedirs(config.results_save_dir, exist_ok=True)
    backtest_file = os.path.join(
        config.results_save_dir,
        f'{args.symbol}_{args.timeframe}_backtest_signals.csv'
    )

    output_cols = ['open', 'high', 'low', 'close', 'volume', 'support', 'resistance',
                   'breakout_prob', 'buy_confidence', 'sell_confidence', 'signal_type']
    signals_df[output_cols].to_csv(backtest_file)

    logger.info(f'Backtest results saved to {backtest_file}')
    logger.info('Backtest completed successfully')

    return True


def main():
    """
    Main entry point.
    """
    parser = argparse.ArgumentParser(
        description='Strategy V3: Advanced Cryptocurrency Entry Signal System'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'predict', 'backtest'],
        default='train',
        help='Operation mode'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTCUSDT',
        help='Trading symbol (default: BTCUSDT)'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        default='15m',
        choices=['15m', '1h', '1d'],
        help='Timeframe (default: 15m)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Execute appropriate mode
    if args.mode == 'train':
        success = train_model(args)
    elif args.mode == 'predict':
        success = predict_signals(args)
    elif args.mode == 'backtest':
        success = backtest_strategy(args)

    if success:
        logger.info('Process completed successfully')
        sys.exit(0)
    else:
        logger.error('Process failed')
        sys.exit(1)


if __name__ == '__main__':
    main()
