#!/usr/bin/env python3
"""
Test pattern-based reversal detection on 1 hour timeframe
Hypothesis: Maybe 15m is too noisy, patterns work better on 1h
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
    logger.info('Pattern-Based Detection on 1H TIMEFRAME')
    logger.info('='*70)
    
    config = StrategyConfig.get_default()
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.results_save_dir, exist_ok=True)
    
    logger.info('Loading data on 1H timeframe...')
    loader = DataLoader(
        hf_repo=config.data.hf_repo,
        cache_dir=config.data.cache_dir,
        verbose=False
    )
    
    df = loader.load_data(
        symbol='BTCUSDT',
        timeframe='1h',  # Changed from 15m to 1h
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
    
    logger.info('Creating pattern-based labels on 1H timeframe...')
    target, profits = create_pattern_labels(
        df_features,
        profit_target_pct=0.01,
        stop_loss_pct=0.01,
        max_hold_bars=20,  # Still 20 bars, but now 20 hours instead of 5
        min_breakout_pct=0.005
    )
    
    df_features['pattern_target'] = target
    df_features['pattern_pnl'] = profits * 100
    
    labeled_mask = target != -1
    df_labeled = df_features[labeled_mask].copy()
    
    positive_count = (df_labeled['pattern_target'] == 1).sum()
    negative_count = (df_labeled['pattern_target'] == 0).sum()
    total_labeled = positive_count + negative_count
    
    logger.info('='*70)
    logger.info('1H Timeframe Results')
    logger.info('='*70)
    logger.info(f'Total labeled patterns: {total_labeled}')
    logger.info(f'Profitable patterns: {positive_count}')
    logger.info(f'Unprofitable patterns: {negative_count}')
    
    if total_labeled > 0:
        win_rate = positive_count / total_labeled * 100
        logger.info(f'Win rate: {win_rate:.2f}%')
    
    logger.info('='*70)
    logger.info('Comparison: Different Timeframes')
    logger.info('='*70)
    logger.info('15 Minute Timeframe:')
    logger.info(f'  - 1 bar = 15 minutes')
    logger.info(f'  - 20 bar window = 5 hours')
    logger.info(f'  - Win rate: 30.61%')
    logger.info('')
    logger.info('1 Hour Timeframe:')
    logger.info(f'  - 1 bar = 1 hour')
    logger.info(f'  - 20 bar window = 20 hours')
    logger.info(f'  - Win rate: {win_rate:.2f}%' if total_labeled > 0 else f'  - Win rate: N/A')
    logger.info('')
    
    if total_labeled > 0 and win_rate > 40:
        logger.info('✓ FINDING: 1H timeframe SIGNIFICANTLY improves win rate!')
        logger.info('  Implication: 15m is too noisy for pattern trading')
        logger.info('  Recommendation: Use 1H or longer timeframe')
    elif total_labeled > 0 and win_rate > 35:
        logger.info('✓ FINDING: 1H timeframe MARGINALLY improves win rate')
        logger.info('  Implication: Timeframe matters, but still not good enough')
    else:
        logger.info('✗ FINDING: 1H timeframe does NOT improve win rate')
        logger.info('  Implication: Problem is not timeframe-specific')
    
    logger.info('='*70)
    
    return True


if __name__ == '__main__':
    setup_logging()
    success = main()
    sys.exit(0 if success else 1)
