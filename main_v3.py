#!/usr/bin/env python3
"""
Strategy V3: 7-Classifier Entry Signal System for 15m Crypto Trading

Predicts 7 classification targets optimized for 15m timeframe:
  1. MOMENTUM_STATUS: Dead/Weak/Normal/Strong
  2. VOLUME_SIGNAL: Normal/Spike/Abnormal
  3. PULLBACK_COMING: Yes/No
  4. REVERSAL_RISK: Low/Medium/High
  5. MA_ALIGNMENT: Bullish/Neutral/Bearish
  6. MTF_STRENGTH: 1-5 (Multi-timeframe strength)
  7. ENTRY_TIMING: Now/Wait/Too-Late

Usage:
    python main_v3.py --mode train --symbol BTCUSDT --timeframe 15m --verbose
    python main_v3.py --mode predict --symbol BTCUSDT --timeframe 15m --verbose
"""

import argparse
import os
import sys
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from strategy_v3 import (
    StrategyConfig,
    DataLoader,
    FeatureEngineer,
)


def setup_logging(verbose: bool = False):
    log_level = 'DEBUG' if verbose else 'INFO'
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}'
    )


def _create_15m_targets(df: pd.DataFrame, forward_window: int = 5, min_profit_pct: float = 0.5) -> dict:
    """
    Create 7 classification targets optimized for 15m trading.
    
    Args:
        df: DataFrame with OHLCV data
        forward_window: Candles to look ahead
        min_profit_pct: Minimum move to count as profitable
        
    Returns:
        Dictionary with 7 target arrays
    """
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    
    targets = {
        'momentum_status': np.zeros(len(df), dtype=int),
        'volume_signal': np.zeros(len(df), dtype=int),
        'pullback_coming': np.zeros(len(df), dtype=int),
        'reversal_risk': np.zeros(len(df), dtype=int),
        'ma_alignment': np.zeros(len(df), dtype=int),
        'mtf_strength': np.zeros(len(df), dtype=int),
        'entry_timing': np.zeros(len(df), dtype=int),
    }
    
    avg_volume = pd.Series(volume).rolling(20).mean()
    price_range = high - low
    avg_price_range = pd.Series(price_range).rolling(20).mean()
    
    for i in range(len(df) - forward_window):
        current_price = close[i]
        future_high = high[i+1:i+1+forward_window].max()
        future_low = low[i+1:i+1+forward_window].min()
        future_range = future_high - future_low
        future_volume_avg = volume[i+1:i+1+forward_window].mean()
        
        upside_pct = ((future_high - current_price) / current_price) * 100
        downside_pct = ((current_price - future_low) / current_price) * 100
        momentum_pct = ((future_high - future_low) / current_price) * 100
        
        # 1. MOMENTUM_STATUS
        if momentum_pct < 0.3:
            targets['momentum_status'][i] = 0
        elif momentum_pct < 0.6:
            targets['momentum_status'][i] = 1
        elif momentum_pct < 1.2:
            targets['momentum_status'][i] = 2
        else:
            targets['momentum_status'][i] = 3
        
        # 2. VOLUME_SIGNAL
        if avg_volume[i] > 0:
            volume_ratio = future_volume_avg / avg_volume[i]
            if volume_ratio > 2.0:
                targets['volume_signal'][i] = 1
            elif volume_ratio < 0.5:
                targets['volume_signal'][i] = 2
            else:
                targets['volume_signal'][i] = 0
        
        # 3. PULLBACK_COMING
        if avg_price_range[i] > 0 and future_range > avg_price_range[i] * 1.5:
            targets['pullback_coming'][i] = 1
        
        # 4. REVERSAL_RISK
        if upside_pct > 1.5 or downside_pct > 1.5:
            targets['reversal_risk'][i] = 2
        elif upside_pct > 0.8 or downside_pct > 0.8:
            targets['reversal_risk'][i] = 1
        else:
            targets['reversal_risk'][i] = 0
        
        # 5. MA_ALIGNMENT
        ema9 = pd.Series(close[:i+1]).ewm(span=9, adjust=False).mean().iloc[-1] if i > 0 else current_price
        ema21 = pd.Series(close[:i+1]).ewm(span=21, adjust=False).mean().iloc[-1] if i > 0 else current_price
        
        if ema9 > ema21:
            targets['ma_alignment'][i] = 2
        elif ema9 < ema21:
            targets['ma_alignment'][i] = 0
        else:
            targets['ma_alignment'][i] = 1
        
        # 6. MTF_STRENGTH
        strength = 1
        if upside_pct > 0.5:
            strength += 1
        if downside_pct > 0.5:
            strength += 1
        if avg_volume[i] > 0 and future_volume_avg / avg_volume[i] > 1.2:
            strength += 1
        if momentum_pct > 1.0:
            strength += 1
        targets['mtf_strength'][i] = min(strength, 5)
        
        # 7. ENTRY_TIMING
        if momentum_pct > 1.5 and upside_pct > 1.0:
            targets['entry_timing'][i] = 2
        elif momentum_pct > 0.5:
            targets['entry_timing'][i] = 1
        else:
            targets['entry_timing'][i] = 0
    
    return targets


def train_model(args):
    logger.info('Starting V3 training: 7-Classifier Model')
    logger.info(f'Symbol: {args.symbol}, Timeframe: {args.timeframe}')
    logger.info('Predicting: Momentum, Volume, Pullback, Reversal, MA, MTF Strength, Entry Timing')
    
    config = StrategyConfig.get_default()
    config.verbose = args.verbose
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
    
    if not loader.validate_data(df):
        logger.error('Data validation failed')
        return False
    
    logger.info(f'Loaded {len(df)} candles')
    logger.info(f'Date range: {df.index[0]} to {df.index[-1]}')
    
    # Engineer features
    logger.info('Engineering features...')
    engineer = FeatureEngineer(config)
    df_features = engineer.engineer_features(df)
    logger.info(f'Total features: {len(df_features.columns) - 5}')
    
    # Create targets
    logger.info('Creating 7 classification targets...')
    targets_dict = _create_15m_targets(df, forward_window=5, min_profit_pct=0.5)
    
    # Prepare clean data
    valid_idx = len(df) - 5
    df_clean = df_features.iloc[:valid_idx].copy()
    targets_clean = {k: v[:valid_idx] for k, v in targets_dict.items()}
    
    logger.info(f'Clean data: {len(df_clean)} samples')
    for key, target in targets_clean.items():
        pos_count = len(np.where(target != 0)[0])
        logger.info(f'  {key}: {pos_count} positive samples')
    
    # Split data
    split_idx = int(len(df_clean) * 0.7)
    df_train = df_clean.iloc[:split_idx]
    df_test = df_clean.iloc[split_idx:]
    
    y_train = {k: v[:split_idx] for k, v in targets_clean.items()}
    y_test = {k: v[split_idx:] for k, v in targets_clean.items()}
    
    logger.info(f'Train: {len(df_train)}, Test: {len(df_test)}')
    
    # Prepare features
    feature_cols = [col for col in df_clean.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]
    logger.info(f'Using {len(feature_cols)} features')
    
    # Train 7 classifiers
    logger.info('Training 7 classifiers...')
    models = {}
    results_data = []
    
    classifier_config = {
        'momentum_status': 'Momentum Status',
        'volume_signal': 'Volume Signal',
        'pullback_coming': 'Pullback Coming',
        'reversal_risk': 'Reversal Risk',
        'ma_alignment': 'MA Alignment',
        'mtf_strength': 'MTF Strength',
        'entry_timing': 'Entry Timing',
    }
    
    for key, name in classifier_config.items():
        logger.info(f'Training {name}...')
        model = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbosity=0
        )
        
        model.fit(X_train, y_train[key])
        models[key] = model
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test[key], y_pred)
        
        try:
            if len(np.unique(y_test[key])) > 1:
                auc = roc_auc_score(y_test[key], model.predict_proba(X_test)[:, 1], multi_class='ovr')
            else:
                auc = 0
        except:
            auc = 0
        
        results_data.append({
            'classifier': key,
            'name': name,
            'accuracy': accuracy,
            'auc': auc,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f'  Accuracy: {accuracy:.4f}, AUC: {auc:.4f}')
    
    # Save models
    logger.info('Saving models...')
    for key, model in models.items():
        with open(os.path.join(config.model_save_dir, f'{key}_model.pkl'), 'wb') as f:
            pickle.dump(model, f)
    
    with open(os.path.join(config.model_save_dir, 'feature_cols.pkl'), 'wb') as f:
        pickle.dump(feature_cols, f)
    
    # Save results
    results_df = pd.DataFrame(results_data)
    results_file = os.path.join(
        config.results_save_dir,
        f'{args.symbol}_{args.timeframe}_v3_training_results.csv'
    )
    results_df.to_csv(results_file, index=False)
    logger.info(f'Results saved to {results_file}')
    
    # Summary
    logger.info('\n' + '='*70)
    logger.info('TRAINING SUMMARY - V3 7-CLASSIFIER MODEL')
    logger.info('='*70)
    for idx, row in results_df.iterrows():
        logger.info(f"{row['name']}: Accuracy {row['accuracy']:.4f}, AUC {row['auc']:.4f}")
    logger.info('='*70)
    
    return True


def predict_signals(args):
    logger.info('Generating V3 trading signals...')
    logger.info(f'Symbol: {args.symbol}, Timeframe: {args.timeframe}')
    
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
    try:
        models = {}
        for key in ['momentum_status', 'volume_signal', 'pullback_coming', 'reversal_risk',
                   'ma_alignment', 'mtf_strength', 'entry_timing']:
            with open(os.path.join(config.model_save_dir, f'{key}_model.pkl'), 'rb') as f:
                models[key] = pickle.load(f)
        with open(os.path.join(config.model_save_dir, 'feature_cols.pkl'), 'rb') as f:
            feature_cols = pickle.load(f)
    except FileNotFoundError as e:
        logger.error(f'Models not found: {str(e)}')
        return False
    
    # Generate predictions
    logger.info('Generating predictions...')
    X = df_features[feature_cols]
    
    predictions = {}
    probabilities = {}
    for key, model in models.items():
        predictions[key] = model.predict(X)
        probabilities[key] = model.predict_proba(X).max(axis=1)
    
    # Create signals dataframe
    signals_df = df_features.copy()
    for key in predictions:
        signals_df[f'{key}_pred'] = predictions[key]
        signals_df[f'{key}_conf'] = probabilities[key]
    
    # Calculate composite confidence
    confidence_cols = [f'{k}_conf' for k in models.keys()]
    signals_df['composite_confidence'] = signals_df[confidence_cols].mean(axis=1)
    
    # Generate signal type
    signals_df['signal_type'] = 'HOLD'
    
    # BUY signal
    buy_mask = (
        (predictions['entry_timing'] == 2) &
        (predictions['momentum_status'] >= 2) &
        (predictions['ma_alignment'] == 2)
    )
    signals_df.loc[buy_mask, 'signal_type'] = 'BUY'
    
    # SELL signal
    sell_mask = (
        (predictions['entry_timing'] == 2) &
        (predictions['momentum_status'] >= 2) &
        (predictions['ma_alignment'] == 0)
    )
    signals_df.loc[sell_mask, 'signal_type'] = 'SELL'
    
    # Strong signals
    strong_mask = signals_df['composite_confidence'] > 0.7
    signals_df.loc[buy_mask & strong_mask, 'signal_type'] = 'BUY (Strong)'
    signals_df.loc[sell_mask & strong_mask, 'signal_type'] = 'SELL (Strong)'
    
    # Summary
    buy_count = (signals_df['signal_type'].str.contains('BUY')).sum()
    sell_count = (signals_df['signal_type'].str.contains('SELL')).sum()
    strong_count = signals_df['signal_type'].str.contains('Strong').sum()
    
    logger.info('Signal Summary:')
    logger.info(f'  Total Candles: {len(signals_df)}')
    logger.info(f'  BUY Signals: {buy_count}')
    logger.info(f'  SELL Signals: {sell_count}')
    logger.info(f'  Strong Signals: {strong_count}')
    logger.info(f'  Signal Density: {(buy_count + sell_count) / len(signals_df) * 100:.2f}%')
    
    # Save signals
    os.makedirs(config.results_save_dir, exist_ok=True)
    signals_file = os.path.join(
        config.results_save_dir,
        f'{args.symbol}_{args.timeframe}_v3_signals.csv'
    )
    
    output_cols = ['open', 'high', 'low', 'close', 'volume']
    output_cols += [f'{k}_pred' for k in models.keys()]
    output_cols += [f'{k}_conf' for k in models.keys()]
    output_cols += ['composite_confidence', 'signal_type']
    
    signals_df[output_cols].to_csv(signals_file)
    logger.info(f'Signals saved to {signals_file}')
    
    # Print recent signals
    recent = signals_df[signals_df['signal_type'] != 'HOLD'].tail(10)
    if len(recent) > 0:
        logger.info('Recent signals:')
        for idx, (i, row) in enumerate(recent.iterrows()):
            logger.info(f"  {row.name.strftime('%Y-%m-%d %H:%M')} - {row['signal_type']}: Price {row['close']:.2f}")
    
    logger.info('Signal generation completed')
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Strategy V3: 7-Classifier Entry Signal System for 15m Crypto Trading'
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
        help='Trading symbol'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        default='15m',
        choices=['15m', '1h', '1d'],
        help='Timeframe'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
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
