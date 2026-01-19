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
from sklearn.preprocessing import StandardScaler

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


def _create_trading_targets(df: pd.DataFrame, forward_window: int = 5, min_profit_pct: float = 0.5) -> tuple:
    """
    Create practical trading classification targets.
    
    Instead of predicting exact levels, predict if the next N candles will have:
    1. Profitable uptrend opportunity (min_profit_pct % gain)
    2. Profitable downtrend opportunity (min_profit_pct % loss)
    3. High volatility (tradeable momentum)
    4. Good entry signal (price touches support then bounces)
    
    Args:
        df: DataFrame with OHLCV data
        forward_window: Candles to look ahead
        min_profit_pct: Minimum percentage move to count as "profitable"
        
    Returns:
        Tuple of (buy_signal, sell_signal, momentum_signal, volatility)
    """
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    buy_signal = np.zeros(len(df), dtype=int)
    sell_signal = np.zeros(len(df), dtype=int)
    momentum_signal = np.zeros(len(df), dtype=int)
    volatility = np.zeros(len(df), dtype=float)
    
    for i in range(len(df) - forward_window):
        # Future price action
        future_high = high[i+1:i+1+forward_window].max()
        future_low = low[i+1:i+1+forward_window].min()
        future_close = close[i+1:i+1+forward_window][-1]
        current_price = close[i]
        
        # 1. BUY SIGNAL: Price will go up at least min_profit_pct %
        upside_pct = ((future_high - current_price) / current_price) * 100
        if upside_pct >= min_profit_pct:
            buy_signal[i] = 1
        
        # 2. SELL SIGNAL: Price will go down at least min_profit_pct %
        downside_pct = ((current_price - future_low) / current_price) * 100
        if downside_pct >= min_profit_pct:
            sell_signal[i] = 1
        
        # 3. MOMENTUM: Strong directional move
        future_range = future_high - future_low
        momentum_pct = (future_range / current_price) * 100
        if momentum_pct >= min_profit_pct * 2:
            momentum_signal[i] = 1
        
        # 4. VOLATILITY: How much will price move
        volatility[i] = momentum_pct
    
    return buy_signal, sell_signal, momentum_signal, volatility


def train_model(args):
    """
    Train the strategy model.
    """
    logger.info('Starting model training...')
    logger.info(f'Symbol: {args.symbol}, Timeframe: {args.timeframe}')
    logger.info('Mode: Classification - Predict profitable entry/exit opportunities')

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
    logger.info(f'Date range: {df.index[0]} to {df.index[-1]}')

    # Engineer features on full dataset
    logger.info('Engineering features...')
    engineer = FeatureEngineer(config)
    df_features = engineer.engineer_features(df)

    logger.info(f'Total features: {len(df_features.columns) - 5}')

    # Create targets: practical trading signals
    logger.info('Creating trading targets...')
    logger.info('  - BUY: Next 5 candles will have +0.5% opportunity')
    logger.info('  - SELL: Next 5 candles will have -0.5% opportunity')
    logger.info('  - MOMENTUM: Next 5 candles will move +1.0% or more')
    logger.info('  - VOLATILITY: Price swing magnitude')
    
    buy_signal, sell_signal, momentum_signal, volatility = _create_trading_targets(
        df, forward_window=5, min_profit_pct=0.5
    )
    
    # Remove last rows that have NaN
    valid_idx = len(df) - 5
    df_clean = df_features.iloc[:valid_idx].copy()
    buy_signal_clean = buy_signal[:valid_idx]
    sell_signal_clean = sell_signal[:valid_idx]
    momentum_signal_clean = momentum_signal[:valid_idx]
    volatility_clean = volatility[:valid_idx]

    logger.info(f'Clean data: {len(df_clean)} samples')
    logger.info(f'BUY signal frequency: {buy_signal_clean.mean():.2%} ({buy_signal_clean.sum()} signals)')
    logger.info(f'SELL signal frequency: {sell_signal_clean.mean():.2%} ({sell_signal_clean.sum()} signals)')
    logger.info(f'MOMENTUM signal frequency: {momentum_signal_clean.mean():.2%} ({momentum_signal_clean.sum()} signals)')
    logger.info(f'Volatility range: {volatility_clean.min():.2f}% to {volatility_clean.max():.2f}%')
    logger.info(f'Mean volatility: {volatility_clean.mean():.2f}%')

    # Split into train/test
    logger.info('Splitting clean data into train/test sets (70/30)...')
    split_idx = int(len(df_clean) * 0.7)
    
    df_train = df_clean.iloc[:split_idx].copy()
    df_test = df_clean.iloc[split_idx:].copy()
    
    y_train_buy = buy_signal_clean[:split_idx]
    y_train_sell = sell_signal_clean[:split_idx]
    y_train_momentum = momentum_signal_clean[:split_idx]
    y_train_volatility = volatility_clean[:split_idx]
    
    y_test_buy = buy_signal_clean[split_idx:]
    y_test_sell = sell_signal_clean[split_idx:]
    y_test_momentum = momentum_signal_clean[split_idx:]
    y_test_volatility = volatility_clean[split_idx:]

    logger.info(f'Training samples: {len(df_train)}')
    logger.info(f'Testing samples: {len(df_test)}')
    logger.info(f'Train BUY signals: {y_train_buy.sum()} ({y_train_buy.mean():.2%})')
    logger.info(f'Train SELL signals: {y_train_sell.sum()} ({y_train_sell.mean():.2%})')
    logger.info(f'Test BUY signals: {y_test_buy.sum()} ({y_test_buy.mean():.2%})')
    logger.info(f'Test SELL signals: {y_test_sell.sum()} ({y_test_sell.mean():.2%})')

    # Get feature columns
    feature_cols = [col for col in df_clean.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]

    logger.info(f'Using {len(feature_cols)} features for training')

    # Train models for classification
    logger.info('Training classification models...')
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    results_data = []
    
    # Train BUY classifier
    logger.info('Training BUY opportunity classifier...')
    buy_model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
    buy_model.fit(X_train, y_train_buy)
    buy_pred_train = buy_model.predict(X_train)
    buy_pred_test = buy_model.predict(X_test)
    buy_proba_test = buy_model.predict_proba(X_test)[:, 1]
    
    buy_metrics = {
        'model': 'BUY Classifier',
        'train_accuracy': accuracy_score(y_train_buy, buy_pred_train),
        'test_accuracy': accuracy_score(y_test_buy, buy_pred_test),
        'test_precision': precision_score(y_test_buy, buy_pred_test, zero_division=0),
        'test_recall': recall_score(y_test_buy, buy_pred_test, zero_division=0),
        'test_f1': f1_score(y_test_buy, buy_pred_test, zero_division=0),
        'test_auc': roc_auc_score(y_test_buy, buy_proba_test) if len(np.unique(y_test_buy)) > 1 else 0,
        'signal_ratio': y_test_buy.mean(),
        'timestamp': datetime.now().isoformat()
    }
    results_data.append(buy_metrics)
    logger.info(f'  Accuracy: {buy_metrics["test_accuracy"]:.4f}')
    logger.info(f'  Precision: {buy_metrics["test_precision"]:.4f} (when it predicts BUY, is it correct?)')
    logger.info(f'  Recall: {buy_metrics["test_recall"]:.4f} (can it catch BUY signals?)')
    logger.info(f'  F1-Score: {buy_metrics["test_f1"]:.4f}')
    
    # Train SELL classifier
    logger.info('Training SELL opportunity classifier...')
    sell_model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
    sell_model.fit(X_train, y_train_sell)
    sell_pred_train = sell_model.predict(X_train)
    sell_pred_test = sell_model.predict(X_test)
    sell_proba_test = sell_model.predict_proba(X_test)[:, 1]
    
    sell_metrics = {
        'model': 'SELL Classifier',
        'train_accuracy': accuracy_score(y_train_sell, sell_pred_train),
        'test_accuracy': accuracy_score(y_test_sell, sell_pred_test),
        'test_precision': precision_score(y_test_sell, sell_pred_test, zero_division=0),
        'test_recall': recall_score(y_test_sell, sell_pred_test, zero_division=0),
        'test_f1': f1_score(y_test_sell, sell_pred_test, zero_division=0),
        'test_auc': roc_auc_score(y_test_sell, sell_proba_test) if len(np.unique(y_test_sell)) > 1 else 0,
        'signal_ratio': y_test_sell.mean(),
        'timestamp': datetime.now().isoformat()
    }
    results_data.append(sell_metrics)
    logger.info(f'  Accuracy: {sell_metrics["test_accuracy"]:.4f}')
    logger.info(f'  Precision: {sell_metrics["test_precision"]:.4f}')
    logger.info(f'  Recall: {sell_metrics["test_recall"]:.4f}')
    logger.info(f'  F1-Score: {sell_metrics["test_f1"]:.4f}')
    
    # Train MOMENTUM classifier
    logger.info('Training MOMENTUM classifier...')
    momentum_model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
    momentum_model.fit(X_train, y_train_momentum)
    momentum_pred_train = momentum_model.predict(X_train)
    momentum_pred_test = momentum_model.predict(X_test)
    momentum_proba_test = momentum_model.predict_proba(X_test)[:, 1]
    
    momentum_metrics = {
        'model': 'MOMENTUM Classifier',
        'train_accuracy': accuracy_score(y_train_momentum, momentum_pred_train),
        'test_accuracy': accuracy_score(y_test_momentum, momentum_pred_test),
        'test_precision': precision_score(y_test_momentum, momentum_pred_test, zero_division=0),
        'test_recall': recall_score(y_test_momentum, momentum_pred_test, zero_division=0),
        'test_f1': f1_score(y_test_momentum, momentum_pred_test, zero_division=0),
        'test_auc': roc_auc_score(y_test_momentum, momentum_proba_test) if len(np.unique(y_test_momentum)) > 1 else 0,
        'signal_ratio': y_test_momentum.mean(),
        'timestamp': datetime.now().isoformat()
    }
    results_data.append(momentum_metrics)
    logger.info(f'  Accuracy: {momentum_metrics["test_accuracy"]:.4f}')
    logger.info(f'  Precision: {momentum_metrics["test_precision"]:.4f}')
    logger.info(f'  Recall: {momentum_metrics["test_recall"]:.4f}')
    logger.info(f'  F1-Score: {momentum_metrics["test_f1"]:.4f}')
    
    # Save models
    import pickle
    logger.info('Saving models...')
    with open(os.path.join(config.model_save_dir, 'buy_classifier.pkl'), 'wb') as f:
        pickle.dump(buy_model, f)
    with open(os.path.join(config.model_save_dir, 'sell_classifier.pkl'), 'wb') as f:
        pickle.dump(sell_model, f)
    with open(os.path.join(config.model_save_dir, 'momentum_classifier.pkl'), 'wb') as f:
        pickle.dump(momentum_model, f)
    with open(os.path.join(config.model_save_dir, 'feature_cols.pkl'), 'wb') as f:
        pickle.dump(feature_cols, f)
    
    # Save results
    results_file = os.path.join(
        config.results_save_dir,
        f'{args.symbol}_{args.timeframe}_training_results.csv'
    )
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(results_file, index=False)
    logger.info(f'Training results saved to {results_file}')

    # Print summary
    logger.info('\n' + '='*70)
    logger.info('TRAINING SUMMARY - CLASSIFICATION MODELS')
    logger.info('='*70)
    for idx, row in results_df.iterrows():
        logger.info(f"\n{row['model']}:")
        logger.info(f"  Train Accuracy: {row['train_accuracy']:.4f}")
        logger.info(f"  Test Accuracy:  {row['test_accuracy']:.4f}")
        logger.info(f"  Precision: {row['test_precision']:.4f} (信號準確度)")
        logger.info(f"  Recall: {row['test_recall']:.4f} (信號捕捉率)")
        logger.info(f"  F1-Score: {row['test_f1']:.4f}")
        if row['test_auc'] > 0:
            logger.info(f"  ROC-AUC: {row['test_auc']:.4f}")
        logger.info(f"  Signal Ratio: {row['signal_ratio']:.2%}")
        
        if row['test_f1'] > 0.3:
            logger.info(f"  Status: 可用的模型 (Usable model)")
        elif row['test_f1'] > 0.1:
            logger.info(f"  Status: 弱信號 (Weak signal)")
        else:
            logger.info(f"  Status: 無效 (Invalid)")
    logger.info('\n' + '='*70)

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
    import pickle
    try:
        with open(os.path.join(config.model_save_dir, 'buy_classifier.pkl'), 'rb') as f:
            buy_model = pickle.load(f)
        with open(os.path.join(config.model_save_dir, 'sell_classifier.pkl'), 'rb') as f:
            sell_model = pickle.load(f)
        with open(os.path.join(config.model_save_dir, 'momentum_classifier.pkl'), 'rb') as f:
            momentum_model = pickle.load(f)
        with open(os.path.join(config.model_save_dir, 'feature_cols.pkl'), 'rb') as f:
            feature_cols = pickle.load(f)
    except FileNotFoundError:
        logger.error('Models not found. Please train the model first.')
        return False

    # Generate predictions
    logger.info('Generating predictions...')
    X = df_features[feature_cols]
    
    buy_pred = buy_model.predict(X)
    buy_proba = buy_model.predict_proba(X)[:, 1]
    
    sell_pred = sell_model.predict(X)
    sell_proba = sell_model.predict_proba(X)[:, 1]
    
    momentum_pred = momentum_model.predict(X)
    momentum_proba = momentum_model.predict_proba(X)[:, 1]
    
    # Create signals DataFrame
    signals_df = df_features.copy()
    signals_df['buy_signal'] = buy_pred
    signals_df['buy_confidence'] = buy_proba
    signals_df['sell_signal'] = sell_pred
    signals_df['sell_confidence'] = sell_proba
    signals_df['momentum_signal'] = momentum_pred
    signals_df['momentum_confidence'] = momentum_proba
    
    # Generate combined signal
    signals_df['signal_type'] = 'HOLD'
    signals_df.loc[buy_pred == 1, 'signal_type'] = 'BUY'
    signals_df.loc[sell_pred == 1, 'signal_type'] = 'SELL'
    signals_df.loc[(buy_pred == 1) & (momentum_pred == 1), 'signal_type'] = 'BUY (Strong)'
    signals_df.loc[(sell_pred == 1) & (momentum_pred == 1), 'signal_type'] = 'SELL (Strong)'

    # Get summary
    buy_count = (signals_df['signal_type'].str.contains('BUY')).sum()
    sell_count = (signals_df['signal_type'].str.contains('SELL')).sum()
    strong_count = signals_df['signal_type'].str.contains('Strong').sum()

    logger.info('Signal Summary:')
    logger.info(f'  Total Candles: {len(signals_df)}')
    logger.info(f'  BUY Signals: {buy_count}')
    logger.info(f'  SELL Signals: {sell_count}')
    logger.info(f'  Strong Signals: {strong_count}')
    logger.info(f'  Signal Density: {(buy_count + sell_count) / len(signals_df):.2%}')

    # Save signals
    os.makedirs(config.results_save_dir, exist_ok=True)
    signals_file = os.path.join(
        config.results_save_dir,
        f'{args.symbol}_{args.timeframe}_latest_signals.csv'
    )
    
    output_cols = ['open', 'high', 'low', 'close', 'volume', 'buy_confidence', 'sell_confidence',
                   'momentum_confidence', 'signal_type']
    signals_df[output_cols].to_csv(signals_file)
    logger.info(f'Signals saved to {signals_file}')

    # Print recent signals
    recent_signals = signals_df[signals_df['signal_type'] != 'HOLD'].tail(10)
    if len(recent_signals) > 0:
        logger.info('\nRecent Signals:')
        for idx, (i, row) in enumerate(recent_signals.iterrows()):
            logger.info(f"  {row.name.strftime('%Y-%m-%d %H:%M')} - {row['signal_type']}: "
                       f"Price {row['close']:.2f}")

    logger.info('Signal generation completed successfully')
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
        choices=['train', 'predict'],
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

    if success:
        logger.info('Process completed successfully')
        sys.exit(0)
    else:
        logger.error('Process failed')
        sys.exit(1)


if __name__ == '__main__':
    main()
