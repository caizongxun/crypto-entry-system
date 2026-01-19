#!/usr/bin/env python3
"""
Strategy V3 V2: Binary Reversal Prediction with Multi-Confirmation Detection

Improved reversal detection with:
- Volume spike confirmation
- ATR expansion confirmation
- Trend continuation confirmation

Usage:
    python main_v3_v2.py --mode train --symbol BTCUSDT --timeframe 15m --verbose
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
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score

from strategy_v3 import (
    StrategyConfig,
    DataLoader,
    FeatureEngineer,
)
from strategy_v3.targets_v2 import create_reversal_target_v2


def setup_logging(verbose: bool = False):
    log_level = 'DEBUG' if verbose else 'INFO'
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}'
    )


def _clean_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    df = df.copy()
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    nan_before = df[feature_cols].isna().sum().sum()
    df[feature_cols] = df[feature_cols].fillna(0)
    df[feature_cols] = df[feature_cols].clip(-1e6, 1e6)
    if nan_before > 0:
        logger.info(f'Cleaned {nan_before} inf/NaN values')
    return df


def train_model_v2(args):
    logger.info('Starting V3 V2 training: Multi-Confirmation Reversal Detection')
    logger.info(f'Symbol: {args.symbol}, Timeframe: {args.timeframe}')
    logger.info('Improvements: volume spike + ATR expansion + trend confirmation')
    
    config = StrategyConfig.get_default()
    config.verbose = args.verbose
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.results_save_dir, exist_ok=True)
    
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
    
    logger.info('Engineering features...')
    engineer = FeatureEngineer(config)
    df_features = engineer.engineer_features(df)
    logger.info(f'Generated {len(df_features.columns) - 5} features')
    
    logger.info('Creating reversal target with multi-confirmation filters...')
    logger.info('Filters: volume >= 1.2x mean, ATR >= 1.3x mean, trend >= 2 bars')
    
    target = create_reversal_target_v2(
        df_features,
        lookback=config.lookback_window,
        left_bars=config.swing_left_bars,
        right_bars=config.swing_right_bars,
        volume_multiplier=1.2,
        atr_multiplier=1.3,
        trend_bars=2
    )
    
    df_features['reversal_target'] = target
    
    positive_count = (target == 1).sum()
    negative_count = (target == 0).sum()
    
    logger.info(f'Clean data: {len(df_features)} candles')
    logger.info(f'Reversal opportunities (target=1): {positive_count} ({positive_count/len(target)*100:.2f}%)')
    logger.info(f'No reversal (target=0): {negative_count} ({negative_count/len(target)*100:.2f}%)')
    logger.info(f'Reversal ratio: {positive_count}:{negative_count}')
    
    train_size = int(len(df_features) * 0.7)
    df_train = df_features.iloc[:train_size]
    df_test = df_features.iloc[train_size:]
    
    y_train = df_train['reversal_target'].values
    y_test = df_test['reversal_target'].values
    
    logger.info(f'Train: {len(df_train)}, Test: {len(df_test)}')
    logger.info(f'Train positive ratio: {(y_train == 1).sum() / len(y_train) * 100:.2f}%')
    logger.info(f'Test positive ratio: {(y_test == 1).sum() / len(y_test) * 100:.2f}%')
    
    feature_cols = [col for col in df_features.columns 
                    if col not in ['open', 'high', 'low', 'close', 'volume', 'reversal_target']]
    
    logger.info('Cleaning features...')
    df_train = _clean_features(df_train, feature_cols)
    df_test = _clean_features(df_test, feature_cols)
    
    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]
    logger.info(f'Using {len(feature_cols)} features')
    
    logger.info('Training binary reversal classifier...')
    
    scale_pos_weight = (y_train == 0).sum() / ((y_train == 1).sum() + 1)
    logger.info(f'Class weight (negative:positive): {scale_pos_weight:.2f}')
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        verbosity=0
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except:
        auc = 0
    
    logger.info(f'Accuracy: {accuracy:.4f}')
    logger.info(f'Precision: {precision:.4f}')
    logger.info(f'Recall: {recall:.4f}')
    logger.info(f'F1-Score: {f1:.4f}')
    logger.info(f'AUC: {auc:.4f}')
    
    logger.info('Confusion Matrix:')
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    logger.info(f'True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}')
    logger.info(f'False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}')
    
    logger.info('Feature Importance (Top 20):')
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(20).iterrows():
        logger.info(f'  {row["feature"]}: {row["importance"]:.4f}')
    
    logger.info('Saving model...')
    with open(os.path.join(config.model_save_dir, 'reversal_model_v2.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    with open(os.path.join(config.model_save_dir, 'feature_cols_v2.pkl'), 'wb') as f:
        pickle.dump(feature_cols, f)
    
    results_data = [{
        'metric': 'accuracy',
        'value': accuracy,
        'timestamp': datetime.now().isoformat()
    }, {
        'metric': 'precision',
        'value': precision,
        'timestamp': datetime.now().isoformat()
    }, {
        'metric': 'recall',
        'value': recall,
        'timestamp': datetime.now().isoformat()
    }, {
        'metric': 'f1_score',
        'value': f1,
        'timestamp': datetime.now().isoformat()
    }, {
        'metric': 'auc',
        'value': auc,
        'timestamp': datetime.now().isoformat()
    }]
    
    results_df = pd.DataFrame(results_data)
    results_file = os.path.join(
        config.results_save_dir,
        f'{args.symbol}_{args.timeframe}_v3_v2_training_results.csv'
    )
    results_df.to_csv(results_file, index=False)
    logger.info(f'Results saved to {results_file}')
    
    logger.info('='*70)
    logger.info('TRAINING SUMMARY - V3 V2 MULTI-CONFIRMATION REVERSAL DETECTION')
    logger.info('='*70)
    logger.info(f'Model: XGBoost Binary Classifier')
    logger.info(f'Target Definition: Multi-confirmed reversals')
    logger.info(f'Confirmations: Volume spike + ATR expansion + Trend')
    logger.info(f'Reversal signals: {positive_count} ({positive_count/len(target)*100:.2f}%)')
    logger.info(f'Accuracy: {accuracy:.4f}')
    logger.info(f'Precision: {precision:.4f}')
    logger.info(f'Recall: {recall:.4f}')
    logger.info(f'F1-Score: {f1:.4f}')
    logger.info(f'AUC: {auc:.4f}')
    logger.info('='*70)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Strategy V3 V2: Multi-Confirmation Reversal Detection'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train'],
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
    
    success = train_model_v2(args)
    
    if success:
        logger.info('Process completed successfully')
        sys.exit(0)
    else:
        logger.error('Process failed')
        sys.exit(1)


if __name__ == '__main__':
    main()
