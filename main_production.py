#!/usr/bin/env python3
"""
Production Reversal Detection - LEAD-LAG ARCHITECTURE

Core Concept:
Features at time t predict reversal profit at time t+lead_bars

Example:
- At bar 100: Market shows reversal patterns (volume, momentum, etc)
- At bar 103-120: Profitable reversal occurs
- Model learns: These bar-100 features predict profitable reversal

CRITICAL FIX: Only trains on labeled data (target != -1)
- Unlabeled bars are ignored
- Model learns from actual swing points only
- No bias from non-swing bars

Usage:
    python main_production.py --mode train --symbol BTCUSDT --timeframe 15m --lead-bars 3
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
    logger.info('='*70)
    logger.info('Production Reversal Detection - LEAD-LAG ARCHITECTURE')
    logger.info('='*70)
    logger.info(f'Symbol: {args.symbol}, Timeframe: {args.timeframe}')
    logger.info(f'Label Method: LEAD-LAG (features at t predict reversal at t+{args.lead_bars})')
    logger.info(f'Training Strategy: Only labeled data (target != -1)')
    logger.info(f'Trading Parameters: PT={args.profit_pct*100:.2f}%, SL={args.stop_loss_pct*100:.2f}%, MaxBars={args.max_hold_bars}')
    logger.info('='*70)
    
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
    
    logger.info('Creating LEAD-LAG labels...')
    logger.info(f'Features at time t will predict reversal at time t+{args.lead_bars}')
    target, profits = create_reversal_target_v3(
        df_features,
        lookback=config.lookback_window,
        left_bars=config.swing_left_bars,
        right_bars=config.swing_right_bars,
        profit_target_pct=args.profit_pct,
        stop_loss_pct=args.stop_loss_pct,
        max_hold_bars=args.max_hold_bars,
        lead_bars=args.lead_bars
    )
    
    df_features['reversal_target'] = target
    df_features['trade_pnl'] = profits * 100
    
    labeled_mask = target != -1
    df_labeled = df_features[labeled_mask].copy()
    
    positive_count = (df_labeled['reversal_target'] == 1).sum()
    negative_count = (df_labeled['reversal_target'] == 0).sum()
    total_labeled = positive_count + negative_count
    
    logger.info(f'Label Analysis:')
    logger.info(f'Total data points: {len(df_features)}')
    logger.info(f'Labeled data points: {total_labeled}')
    logger.info(f'Unlabeled data points: {len(df_features) - total_labeled}')
    logger.info(f'Profitable reversals (target=1): {positive_count}')
    logger.info(f'Unprofitable reversals (target=0): {negative_count}')
    
    if total_labeled == 0:
        logger.error('No labeled data found')
        return False
    
    if positive_count == 0:
        logger.error('No profitable reversals found')
        return False
    
    label_win_rate = positive_count / total_labeled * 100
    logger.info(f'Win rate (among labeled points): {label_win_rate:.2f}%')
    
    train_size = int(len(df_labeled) * 0.7)
    df_train = df_labeled.iloc[:train_size]
    df_test = df_labeled.iloc[train_size:]
    
    y_train = df_train['reversal_target'].values
    y_test = df_test['reversal_target'].values
    
    logger.info(f'Train/Test Split: {len(df_train)}/{len(df_test)}')
    logger.info(f'Train profitable: {(y_train == 1).sum()} / {len(y_train)}')
    logger.info(f'Test profitable: {(y_test == 1).sum()} / {len(y_test)}')
    
    feature_cols = [col for col in df_features.columns 
                    if col not in ['open', 'high', 'low', 'close', 'volume', 'reversal_target', 'trade_pnl']]
    
    logger.info('Cleaning features...')
    df_train = _clean_features(df_train, feature_cols)
    df_test = _clean_features(df_test, feature_cols)
    
    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]
    logger.info(f'Using {len(feature_cols)} features')
    
    scale_pos_weight = (y_train == 0).sum() / ((y_train == 1).sum() + 1)
    logger.info(f'Class weight: {scale_pos_weight:.2f}:1')
    
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
    logger.info(f'Ensemble AUC (weighted): {ensemble_auc:.4f}')
    
    logger.info('Optimizing prediction threshold...')
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
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    
    logger.info('Confusion Matrix:')
    logger.info(f'TN: {tn}, FP: {fp}')
    logger.info(f'FN: {fn}, TP: {tp}')
    
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0
    tpr = tp / (fn + tp) if (fn + tp) > 0 else 0
    
    logger.info('Performance Metrics:')
    logger.info(f'Accuracy: {accuracy:.4f}')
    logger.info(f'Precision: {precision:.4f}')
    logger.info(f'Recall (True Positive Rate): {recall:.4f}')
    logger.info(f'False Positive Rate: {fpr:.4f}')
    logger.info(f'F1-Score: {f1:.4f}')
    
    logger.info('Saving models and metadata...')
    with open(os.path.join(config.model_save_dir, 'ensemble_models.pkl'), 'wb') as f:
        pickle.dump(models, f)
    
    with open(os.path.join(config.model_save_dir, 'feature_cols.pkl'), 'wb') as f:
        pickle.dump(feature_cols, f)
    
    with open(os.path.join(config.model_save_dir, 'optimal_threshold.pkl'), 'wb') as f:
        pickle.dump(optimal_threshold, f)
    
    with open(os.path.join(config.model_save_dir, 'ensemble_weights.pkl'), 'wb') as f:
        pickle.dump(weights, f)
    
    config_dict = {
        'profit_target': args.profit_pct,
        'stop_loss': args.stop_loss_pct,
        'max_hold_bars': args.max_hold_bars,
        'lead_bars': args.lead_bars
    }
    with open(os.path.join(config.model_save_dir, 'trading_config.pkl'), 'wb') as f:
        pickle.dump(config_dict, f)
    
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
        'value': fpr,
    }, {
        'metric': 'win_rate',
        'value': label_win_rate,
    }, {
        'metric': 'profitable_signals',
        'value': positive_count,
    }, {
        'metric': 'total_labeled',
        'value': total_labeled,
    }, {
        'metric': 'lead_bars',
        'value': args.lead_bars,
    }]
    
    results_df = pd.DataFrame(results_data)
    results_file = os.path.join(
        config.results_save_dir,
        f'{args.symbol}_{args.timeframe}_production_results.csv'
    )
    results_df.to_csv(results_file, index=False)
    logger.info(f'Results saved to {results_file}')
    
    logger.info('='*70)
    logger.info('TRAINING COMPLETE')
    logger.info('='*70)
    logger.info(f'Architecture: LEAD-LAG (lead={args.lead_bars} bars)')
    logger.info(f'Training Strategy: Labeled data only')
    logger.info(f'Trading Parameters: PT={args.profit_pct*100:.2f}%, SL={args.stop_loss_pct*100:.2f}%, MaxBars={args.max_hold_bars}')
    logger.info(f'Total labeled: {total_labeled}')
    logger.info(f'Label Win Rate: {label_win_rate:.2f}%')
    logger.info(f'Profitable Signals: {positive_count}')
    logger.info(f'Ensemble AUC: {ensemble_auc:.4f}')
    logger.info(f'Test Precision: {precision:.4f}')
    logger.info(f'Test Recall: {recall:.4f}')
    logger.info(f'Test F1-Score: {f1:.4f}')
    logger.info('='*70)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Production Reversal Detection - LEAD-LAG Architecture'
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
        help='Profit target (default 0.5%)'
    )
    parser.add_argument(
        '--stop-loss-pct',
        type=float,
        default=0.01,
        help='Stop loss level (default 1%)'
    )
    parser.add_argument(
        '--max-hold-bars',
        type=int,
        default=20,
        help='Maximum bars to hold trade (default 20)'
    )
    parser.add_argument(
        '--lead-bars',
        type=int,
        default=3,
        help='How many bars ahead to place label (default 3)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    success = train_production_model(args)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
