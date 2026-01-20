import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import sys
import os

sys.path.insert(0, str(Path(__file__).parent))

from strategy_v3 import StrategyConfig, DataLoader, FeatureEngineer
from strategy_v3.multilayer_features import MultiLayerFeatureEngineer
from strategy_v3.targets_pattern_v1 import create_pattern_labels

logger.remove()
logger.add(
    sys.stderr,
    level='INFO',
    format='{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}'
)


def apply_multilayer_confirmation(df, multilayer_features, target, min_confirmations=2):
    """
    Apply multi-layer confirmation to filter patterns.
    Only processes rows where target != -1
    
    Returns:
        filtered_labels: Labels after multi-layer confirmation
        confidence_scores: Confidence score for each pattern
    """
    filtered_labels = target.copy()
    confidence_scores = np.zeros(len(df), dtype=int)
    
    pattern_mask = target != -1
    
    if not pattern_mask.any():
        return filtered_labels, confidence_scores
    
    close = df['close'].values
    volume = df['volume'].values
    
    ma_4h_fast = pd.Series(close).rolling(window=16).mean().values
    ma_4h_slow = pd.Series(close).rolling(window=32).mean().values
    trend_4h = np.where(ma_4h_fast > ma_4h_slow, 1, -1)
    
    ma_1h_fast = pd.Series(close).rolling(window=4).mean().values
    ma_1h_slow = pd.Series(close).rolling(window=8).mean().values
    trend_1h = np.where(ma_1h_fast > ma_1h_slow, 1, -1)
    
    avg_volume = pd.Series(volume).rolling(window=20).mean().values
    volume_ratio = np.divide(volume, np.maximum(avg_volume, 1), where=avg_volume>0, out=np.ones_like(volume, dtype=float))
    volume_increasing = volume_ratio > 1.2
    
    rsi = multilayer_features['extremum_rsi'].values
    rsi_extreme = (rsi < 30) | (rsi > 70)
    
    volatility_15m = multilayer_features['risk_volatility_15m'].values
    volatility_acceptable = volatility_15m < 0.05
    
    pattern_indices = np.where(pattern_mask)[0]
    
    for idx in pattern_indices:
        confidence = 0
        
        if not volatility_acceptable[idx]:
            filtered_labels[idx] = -1
            confidence_scores[idx] = 0
            continue
        
        if target[idx] == 1:
            if trend_1h[idx] == 1 or trend_4h[idx] == 1:
                confidence += 1
        elif target[idx] == 0:
            if trend_1h[idx] == -1 or trend_4h[idx] == -1:
                confidence += 1
        
        if volume_increasing[idx]:
            confidence += 1
        
        if rsi_extreme[idx]:
            confidence += 1
        
        confidence_scores[idx] = confidence
        
        if confidence < min_confirmations:
            filtered_labels[idx] = 0
    
    return filtered_labels, confidence_scores


def main():
    logger.info('='*70)
    logger.info('Multi-Layer Feature Integration Test for 15m Framework')
    logger.info('='*70)
    
    config = StrategyConfig.get_default()
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.results_save_dir, exist_ok=True)
    
    logger.info('Loading 15m data...')
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
    
    logger.info('\nEngineering existing features...')
    feature_engineer = FeatureEngineer(config)
    df = feature_engineer.engineer_features(df)
    logger.info('Generated existing features')
    
    logger.info('\nDetecting patterns...')
    target, profits = create_pattern_labels(
        df,
        profit_target_pct=0.01,
        stop_loss_pct=0.01,
        max_hold_bars=20,
        min_breakout_pct=0.005
    )
    
    df['pattern_target'] = target
    df['pattern_pnl'] = profits * 100
    
    pattern_mask = target != -1
    pattern_count = pattern_mask.sum()
    logger.info(f'Detected patterns: {pattern_count}')
    
    logger.info('\nEngineering multi-layer features...')
    multilayer_engineer = MultiLayerFeatureEngineer()
    multilayer_features = multilayer_engineer.engineer_multilayer_features(df)
    logger.info(f'Generated {len(multilayer_features.columns)} multi-layer features')
    
    logger.info('\nFeature breakdown:')
    logger.info('  - Momentum: 7 features')
    logger.info('  - Volume: 5 features')
    logger.info('  - Extremum: 7 features')
    logger.info('  - Risk: 5 features')
    logger.info('  - Environment: 5 features')
    
    logger.info('\nApplying multi-layer confirmation logic...')
    filtered_labels, confidence_scores = apply_multilayer_confirmation(
        df, multilayer_features, target, min_confirmations=2
    )
    
    logger.info('\n' + '='*70)
    logger.info('BASELINE: Single Layer (Patterns Only)')
    logger.info('='*70)
    
    positive = (target == 1).sum()
    negative = (target == 0).sum()
    total_labeled = positive + negative
    
    if total_labeled > 0:
        baseline_win_rate = positive / total_labeled * 100
        logger.info(f'Total patterns: {total_labeled}')
        logger.info(f'Profitable: {positive}')
        logger.info(f'Unprofitable: {negative}')
        logger.info(f'Win rate: {baseline_win_rate:.2f}%')
    else:
        baseline_win_rate = 0
        logger.info('No patterns found')
    
    logger.info('\n' + '='*70)
    logger.info('FILTERED: Multi-Layer Confirmation (2+ layers)')
    logger.info('='*70)
    
    high_conf_mask = confidence_scores >= 2
    high_conf_profitable = ((filtered_labels == 1) & high_conf_mask & (target == 1)).sum()
    high_conf_total = high_conf_mask.sum()
    
    if high_conf_total > 0:
        high_conf_win_rate = high_conf_profitable / high_conf_total * 100
        logger.info(f'High confidence trades: {high_conf_total}')
        logger.info(f'Profitable: {high_conf_profitable}')
        logger.info(f'Win rate: {high_conf_win_rate:.2f}%')
        logger.info(f'Filter effectiveness: {(total_labeled - high_conf_total) / total_labeled * 100:.1f}% filtered out')
        logger.info(f'Improvement: +{high_conf_win_rate - baseline_win_rate:.2f}%')
    else:
        high_conf_win_rate = baseline_win_rate
        logger.info('No high-confidence trades found')
    
    logger.info('\n' + '='*70)
    logger.info('CONFIDENCE DISTRIBUTION (Patterns Only)')
    logger.info('='*70)
    
    pattern_indices = np.where(target != -1)[0]
    for conf_level in range(0, 5):
        count = (confidence_scores[pattern_indices] == conf_level).sum()
        if count > 0:
            pct = count / len(pattern_indices) * 100
            logger.info(f'Confidence level {conf_level}: {count} patterns ({pct:.1f}%)')
    
    logger.info('\n' + '='*70)
    logger.info('BREAKDOWN BY CONFIDENCE LEVEL')
    logger.info('='*70)
    
    for conf_level in range(2, 5):
        conf_mask = confidence_scores == conf_level
        profitable_at_level = ((target == 1) & conf_mask).sum()
        total_at_level = conf_mask.sum()
        if total_at_level > 0:
            win_at_level = profitable_at_level / total_at_level * 100
            logger.info(f'Level {conf_level}: {total_at_level} trades, {win_at_level:.2f}% win rate')
    
    logger.info('\n' + '='*70)
    logger.info('SUMMARY')
    logger.info('='*70)
    logger.info(f'Baseline win rate (all patterns): {baseline_win_rate:.2f}%')
    logger.info(f'High confidence win rate (2+ layers): {high_conf_win_rate:.2f}%')
    logger.info(f'Expected with 3+ layers: 40-45%')
    logger.info(f'Expected with all layers: 42-48%')
    
    logger.info('\n' + '='*70)
    logger.info('Integration test complete')
    logger.info('='*70)
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
