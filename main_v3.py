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

    # Engineer features
    logger.info('Engineering features...')
    engineer = FeatureEngineer(config)
    df_features = engineer.engineer_features(df)

    logger.info(f'Total features: {len(df_features.columns) - 5}')

    # Create targets
    logger.info('Creating targets...')
    
    # Support: 5-candle low
    y_support = df_features['low'].rolling(window=5).min().shift(-5)
    
    # Resistance: 5-candle high
    y_resistance = df_features['high'].rolling(window=5).max().shift(-5)
    
    # Breakout probability: if price breaks resistance within 5 candles
    future_high = df_features['high'].rolling(window=5).max().shift(-5)
    y_breakout = (future_high > df_features['resistance'].fillna(df_features['high'])).astype(float)

    # Remove rows with NaN targets
    mask = y_support.notna() & y_resistance.notna() & y_breakout.notna()
    df_clean = df_features[mask].copy()
    y_support = y_support[mask]
    y_resistance = y_resistance[mask]
    y_breakout = y_breakout[mask]

    logger.info(f'Clean data: {len(df_clean)} samples')

    # Split train/test
    train_df, test_df = loader.split_train_test(df_clean, train_ratio=0.7)
    y_train_support = y_support[train_df.index]
    y_train_resistance = y_resistance[train_df.index]
    y_train_breakout = y_breakout[train_df.index]

    # Get feature columns (exclude OHLCV)
    feature_cols = [col for col in df_clean.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    X_train = train_df[feature_cols]

    # Train models
    logger.info('Training models...')
    ensemble = ModelEnsemble(config)

    metrics = ensemble.train(
        X_train=X_train,
        y_train_support=y_train_support,
        y_train_resistance=y_train_resistance,
        y_train_breakout=y_train_breakout
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
        results_data.append({
            'model': model_name,
            'rmse': model_metrics['rmse'],
            'mae': model_metrics['mae'],
            'r2': model_metrics['r2'],
            'timestamp': datetime.now().isoformat()
        })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(results_file, index=False)

    logger.info(f'Training results saved to {results_file}')
    logger.info('Model training completed successfully')

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

    # Create targets for analysis
    y_support = df_features['low'].rolling(window=5).min().shift(-5)
    y_resistance = df_features['high'].rolling(window=5).max().shift(-5)
    y_breakout = (df_features['high'].rolling(window=5).max().shift(-5) > df_features['close']).astype(float)

    # Remove NaN
    mask = y_support.notna() & y_resistance.notna()
    df_clean = df_features[mask].copy()
    y_support = y_support[mask]
    y_resistance = y_resistance[mask]
    y_breakout = y_breakout[mask]

    # Split and get test set
    train_df, test_df = loader.split_train_test(df_clean, train_ratio=0.7)
    y_test_support = y_support[test_df.index]
    y_test_resistance = y_resistance[test_df.index]
    y_test_breakout = y_breakout[test_df.index]

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
    X_test = test_df[feature_cols]

    support_pred, resistance_pred, breakout_pred = ensemble.predict(X_test)

    # Generate signals
    logger.info('Creating backtest signals...')
    signal_gen = SignalGenerator(config)
    signals_df = signal_gen.generate_signals(test_df, support_pred, resistance_pred, breakout_pred)

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
