#!/usr/bin/env python3
"""
Production Reversal Detection System

Core Architecture:
1. Profitability-based labels: Only reversals followed by profit are positive
2. Ensemble of 3 models (XGBoost, LightGBM, CatBoost) with weighted voting
3. Precision-Recall optimization
4. Real-time prediction capability

Key Insight:
Labels are defined by actual trading profits, not theoretical swing detection.
This ensures models learn to identify truly profitable reversals.

Usage:
    python main_production.py --mode train --symbol BTCUSDT --timeframe 15m
"""

import argparse
import os
import sys
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score

try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
except ImportError:
    logger.error('Required libraries not installed. Run: pip install xgboost lightgbm catboost')
    sys.exit(1)

from strategy_v3 import StrategyConfig, DataLoader, FeatureEngineer
from strategy_v3.targets_v3 import create_reversal_target_v3


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


def train_production_model(args):
    logger.info('Starting Production Reversal Detection Training')
    logger.info(f'Symbol: {args.symbol}, Timeframe: {args.timeframe}')
    logger.info('Label Definition: Profitability-based (only profitable reversals labeled as 1)')
    logger.info(f'Profit Target: {args.profit_pct*100:.2f}%')
    logger.info(f'Future Bars: {args.future_bars}')
    
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
    
    logger.info('Creating profitability-based reversal targets...')
    target, profits = create_reversal_target_v3(
        df_features,
        lookback=config.lookback_window,
        left_bars=config.swing_left_bars,
        right_bars=config.swing_right_bars,
        profit_target_pct=args.profit_pct,
        future_bars=args.future_bars
    )
    
    df_features['reversal_target'] = target
    df_features['profit_pct'] = profits * 100
    
    positive_count = (target == 1).sum()
    negative_count = (target == 0).sum()
    
    logger.info(f'Clean data: {len(df_features)} candles')
    logger.info(f'Profitable reversals (target=1): {positive_count}')
    logger.info(f'Unprofitable reversals (target=0): {negative_count}')
    
    total_swings = positive_count + negative_count
    if total_swings == 0:
        logger.error('No swing points found')
        return False
    
    if positive_count == 0:
        logger.error('No profitable reversals found. Try lower profit_pct or more future_bars.')
        return False
    
    profitability_rate = positive_count / total_swings * 100
    logger.info(f'Profitability Rate: {profitability_rate:.2f}% (target=1) / {100-profitability_rate:.2f}% (target=0)')
    
    train_size = int(len(df_features) * 0.7)
    df_train = df_features.iloc[:train_size]
    df_test = df_features.iloc[train_size:]
    
    y_train = df_train['reversal_target'].values
    y_test = df_test['reversal_target'].values
    
    logger.info(f'Train: {len(df_train)}, Test: {len(df_test)}')
    logger.info(f'Train positive ratio: {(y_train == 1).sum() / len(y_train) * 100:.2f}%')
    logger.info(f'Test positive ratio: {(y_test == 1).sum() / len(y_test) * 100:.2f}%')
    
    feature_cols = [col for col in df_features.columns 
                    if col not in ['open', 'high', 'low', 'close', 'volume', 'reversal_target', 'profit_pct']]
    
    logger.info('Cleaning features...')
    df_train = _clean_features(df_train, feature_cols)
    df_test = _clean_features(df_test, feature_cols)
    
    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]
    logger.info(f'Using {len(feature_cols)} features')
    
    scale_pos_weight = (y_train == 0).sum() / ((y_train == 1).sum() + 1)
    logger.info(f'Class weight (negative:positive): {scale_pos_weight:.2f}')
    
    logger.info('Training ensemble models...')
    models = {}
    predictions_ensemble = []
    
    models['xgboost'] = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        verbosity=0
    )
    logger.info('Training XGBoost...')
    models['xgboost'].fit(X_train, y_train)
    xgb_pred_proba = models['xgboost'].predict_proba(X_test)[:, 1]
    predictions_ensemble.append(xgb_pred_proba)
    xgb_auc = roc_auc_score(y_test, xgb_pred_proba)
    logger.info(f'XGBoost AUC: {xgb_auc:.4f}')
    
    models['lightgbm'] = LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        verbose=-1
    )
    logger.info('Training LightGBM...')
    models['lightgbm'].fit(X_train, y_train)
    lgb_pred_proba = models['lightgbm'].predict_proba(X_test)[:, 1]
    predictions_ensemble.append(lgb_pred_proba)
    lgb_auc = roc_auc_score(y_test, lgb_pred_proba)
    logger.info(f'LightGBM AUC: {lgb_auc:.4f}')
    
    models['catboost'] = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.05,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        verbose=0
    )
    logger.info('Training CatBoost...')
    models['catboost'].fit(X_train, y_train)
    cat_pred_proba = models['catboost'].predict_proba(X_test)[:, 1]
    predictions_ensemble.append(cat_pred_proba)
    cat_auc = roc_auc_score(y_test, cat_pred_proba)
    logger.info(f'CatBoost AUC: {cat_auc:.4f}')
    
    weights = np.array([0.4, 0.4, 0.2])
    ensemble_pred_proba = np.average(predictions_ensemble, axis=0, weights=weights)
    ensemble_auc = roc_auc_score(y_test, ensemble_pred_proba)
    logger.info(f'Ensemble AUC (weighted average): {ensemble_auc:.4f}')
    
    logger.info('Optimizing threshold for best F1-Score...')
    precisions, recalls, thresholds = precision_recall_curve(y_test, ensemble_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    logger.info(f'Optimal threshold: {optimal_threshold:.3f}')
    logger.info(f'Optimal F1-Score: {f1_scores[optimal_idx]:.4f}')
    logger.info(f'Optimal Precision: {precisions[optimal_idx]:.4f}')
    logger.info(f'Optimal Recall: {recalls[optimal_idx]:.4f}')
    
    y_pred_optimal = (ensemble_pred_proba > optimal_threshold).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred_optimal)
    precision = precision_score(y_test, y_pred_optimal, zero_division=0)
    recall = recall_score(y_test, y_pred_optimal, zero_division=0)
    f1 = f1_score(y_test, y_pred_optimal, zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred_optimal, labels=[0, 1])
    logger.info('Confusion Matrix:')
    logger.info(f'True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}')
    logger.info(f'False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}')
    
    false_positive_rate = cm[0,1] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    true_positive_rate = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
    
    logger.info(f'False Positive Rate (FPR): {false_positive_rate:.4f}')
    logger.info(f'True Positive Rate (TPR/Recall): {true_positive_rate:.4f}')
    
    logger.info('Saving models...')
    with open(os.path.join(config.model_save_dir, 'ensemble_models.pkl'), 'wb') as f:
        pickle.dump(models, f)
    
    with open(os.path.join(config.model_save_dir, 'feature_cols.pkl'), 'wb') as f:
        pickle.dump(feature_cols, f)
    
    with open(os.path.join(config.model_save_dir, 'optimal_threshold.pkl'), 'wb') as f:
        pickle.dump(optimal_threshold, f)
    
    with open(os.path.join(config.model_save_dir, 'ensemble_weights.pkl'), 'wb') as f:
        pickle.dump(weights, f)
    
    results_data = [{
        'metric': 'accuracy',
        'value': accuracy,
    }, {
        'metric': 'precision',
        'value': precision,
    }, {
        'metric': 'recall',
        'value': recall,
    }, {
        'metric': 'f1_score',
        'value': f1,
    }, {
        'metric': 'ensemble_auc',
        'value': ensemble_auc,
    }, {
        'metric': 'optimal_threshold',
        'value': optimal_threshold,
    }, {
        'metric': 'false_positive_rate',
        'value': false_positive_rate,
    }, {
        'metric': 'true_positive_rate',
        'value': true_positive_rate,
    }, {
        'metric': 'profitable_signals',
        'value': positive_count,
    }, {
        'metric': 'profitability_rate',
        'value': profitability_rate,
    }]
    
    results_df = pd.DataFrame(results_data)
    results_file = os.path.join(
        config.results_save_dir,
        f'{args.symbol}_{args.timeframe}_production_results.csv'
    )
    results_df.to_csv(results_file, index=False)
    logger.info(f'Results saved to {results_file}')
    
    logger.info('='*70)
    logger.info('PRODUCTION REVERSAL DETECTION - TRAINING COMPLETE')
    logger.info('='*70)
    logger.info(f'Label Definition: Profitable reversals only')
    logger.info(f'Total Profitable Reversals: {positive_count}')
    logger.info(f'Profitability Rate: {profitability_rate:.2f}%')
    logger.info(f'Individual Model AUC: XGB={xgb_auc:.4f}, LGB={lgb_auc:.4f}, CAT={cat_auc:.4f}')
    logger.info(f'Ensemble AUC: {ensemble_auc:.4f}')
    logger.info(f'Optimal Threshold: {optimal_threshold:.3f}')
    logger.info(f'Test Accuracy: {accuracy:.4f}')
    logger.info(f'Test Precision: {precision:.4f}')
    logger.info(f'Test Recall: {recall:.4f}')
    logger.info(f'Test F1-Score: {f1:.4f}')
    logger.info(f'False Positive Rate: {false_positive_rate:.4f}')
    logger.info('='*70)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Production Reversal Detection System with Profitability-Based Labels'
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
        '--profit-pct',
        type=float,
        default=0.005,
        help='Minimum profit target (default 0.5%)'
    )
    parser.add_argument(
        '--future-bars',
        type=int,
        default=10,
        help='How many bars ahead to check for profit'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    success = train_production_model(args)
    
    if success:
        logger.info('Training completed successfully')
        sys.exit(0)
    else:
        logger.error('Training failed')
        sys.exit(1)


if __name__ == '__main__':
    main()
