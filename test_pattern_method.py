#!/usr/bin/env python3
"""
Test pattern-based reversal detection
"""

import os
import sys
import pandas as pd
import numpy as np
from loguru import logger

from strategy_v3 import StrategyConfig, DataLoader, FeatureEngineer
from strategy_v3.targets_pattern_v1 import create_pattern_labels


def setup_logging():
    logger.remove()
    logger.add(
        sys.stderr,
        level='INFO',
        format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}'
    )


def main():
    logger.info('='*70)
    logger.info('Pattern-Based Reversal Detection Test')
    logger.info('='*70)
    
    config = StrategyConfig.get_default()
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.results_save_dir, exist_ok=True)
    
    logger.info('Loading data...')
    loader = DataLoader(
        hf_repo=config.data.hf_repo,
        cache_dir=config.data.cache_dir,
        verbose=False
    )
    
    df = loader.load_data(
        symbol='BTCUSDT',
        timeframe='15m',
        cache=True
    )
    
    if not loader.validate_data(df):
        logger.error('Data validation failed')
        return False
    
    logger.info(f'Loaded {len(df)} candles')
    logger.info(f'Date range: {df.index[0]} to {df.index[-1]}')
    
    logger.info('Engineering features...')
    engineer = FeatureEngineer(config)
    df_features = engineer.engineer_features(df)
    logger.info(f'Generated {len(df_features.columns) - 5} features')
    
    logger.info('Creating pattern-based labels...')
    target, profits = create_pattern_labels(
        df_features,
        profit_target_pct=0.01,
        stop_loss_pct=0.01,
        max_hold_bars=20,
        min_breakout_pct=0.002
    )
    
    df_features['pattern_target'] = target
    df_features['pattern_pnl'] = profits * 100
    
    labeled_mask = target != -1
    df_labeled = df_features[labeled_mask].copy()
    
    positive_count = (df_labeled['pattern_target'] == 1).sum()
    negative_count = (df_labeled['pattern_target'] == 0).sum()
    total_labeled = positive_count + negative_count
    
    logger.info('='*70)
    logger.info('Pattern Method Results')
    logger.info('='*70)
    logger.info(f'Total labeled patterns: {total_labeled}')
    logger.info(f'Profitable patterns: {positive_count}')
    logger.info(f'Unprofitable patterns: {negative_count}')
    
    if total_labeled > 0:
        win_rate = positive_count / total_labeled * 100
        logger.info(f'Win rate: {win_rate:.2f}%')
    
    logger.info('='*70)
    logger.info('Pattern Method vs Swing Point Method')
    logger.info('='*70)
    logger.info('Pattern Method:')
    logger.info(f'  - Detected patterns: {total_labeled}')
    logger.info(f'  - Data points used: {len(df_labeled)} out of {len(df_features)}')
    logger.info(f'  - Coverage: {len(df_labeled) / len(df_features) * 100:.2f}%')
    logger.info('')
    logger.info('Swing Point Method (previous):')
    logger.info(f'  - Detected points: 29,181')
    logger.info(f'  - Win rate: 43.44%')
    logger.info(f'  - AUC: 0.6023')
    logger.info(f'  - Precision: 0.3955')
    logger.info('')
    
    if total_labeled == 0:
        logger.warning('No patterns detected. This might be normal for short data or tight parameters.')
    else:
        logger.info(f'Pattern method found {total_labeled} tradeable patterns')
        logger.info('Next step: Train model with pattern features to see if AUC improves')
    
    logger.info('='*70)
    
    return True


if __name__ == '__main__':
    setup_logging()
    success = main()
    sys.exit(0 if success else 1)
