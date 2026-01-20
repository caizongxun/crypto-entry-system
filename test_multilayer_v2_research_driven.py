import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import sys
import os

sys.path.insert(0, str(Path(__file__).parent))

from strategy_v3 import StrategyConfig, DataLoader, FeatureEngineer
from strategy_v3.multilayer_confirmation_v2 import MultiLayerConfirmationV2
from strategy_v3.targets_pattern_v1 import create_pattern_labels

logger.remove()
logger.add(
    sys.stderr,
    level='INFO',
    format='{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}'
)


def main():
    logger.info('='*70)
    logger.info('Multi-Layer Confirmation V2: Research-Driven Implementation')
    logger.info('='*70)
    logger.info('Based on: StockTiming.com 11-year backtest + Reddit multi-confirmation analysis')
    logger.info('')
    
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
    
    logger.info('\nApplying research-driven multi-layer confirmation...')
    confirmer = MultiLayerConfirmationV2()
    confirmed_labels, stats = confirmer.apply_confirmation(df, target)
    
    logger.info('\n' + '='*70)
    logger.info('RESEARCH FRAMEWORK ANALYSIS')
    logger.info('='*70)
    logger.info('\nConfirmation Layers (Independent):')
    logger.info('1. HTF Trend (4h MA alignment with pattern direction)')
    logger.info('2. Price Action (Proper follow-through on breakout)')
    logger.info('3. Volume (Absolute increase > avg + 1 std dev)')
    logger.info('4. Technical Support (Aligns with historical levels)')
    logger.info('5. Extremum Timing (RSI extremes for bonus points)')
    logger.info('')
    
    logger.info('='*70)
    logger.info('BASELINE RESULTS')
    logger.info('='*70)
    logger.info(f'Total patterns: {stats["original_total"]}')
    logger.info(f'Baseline win rate: {stats["original_win_rate"]:.2f}%')
    
    logger.info('\n' + '='*70)
    logger.info('AFTER MULTI-LAYER CONFIRMATION (2+ layers required)')
    logger.info('='*70)
    logger.info(f'High-confidence trades: {stats["high_confidence_count"]}')
    logger.info(f'High-confidence win rate: {stats["high_confidence_win_rate"]:.2f}%')
    logger.info(f'Improvement: +{stats["improvement"]:.2f}%')
    logger.info(f'Trades filtered out: {stats["filter_percentage"]:.1f}%')
    
    logger.info('\n' + '='*70)
    logger.info('CONFIRMATION DISTRIBUTION')
    logger.info('='*70)
    for level in sorted(stats['confirmation_distribution'].keys()):
        count = stats['confirmation_distribution'][level]
        pct = count / stats['original_total'] * 100
        logger.info(f'Level {level} ({level} layers): {count} patterns ({pct:.1f}%)')
    
    logger.info('\n' + '='*70)
    logger.info('RESEARCH VALIDATION')
    logger.info('='*70)
    logger.info('\nComparison with Published Research:')
    logger.info('  - StockTiming.com (11-year backtest):')
    logger.info('    • Double Top/Bottom base: 78%')
    logger.info('    • With proper confirmations: 90%')
    logger.info(f'  Our result: {stats["high_confidence_win_rate"]:.1f}%')
    
    logger.info('\n  - Reddit multi-confirmation (500+ trades):')
    logger.info('    • Single indicator: 38% win rate')
    logger.info('    • 3+ layer confirmation: 62% win rate')
    logger.info(f'  Our baseline: {stats["original_win_rate"]:.1f}%')
    logger.info(f'  Our high confidence: {stats["high_confidence_win_rate"]:.1f}%')
    
    logger.info('\n' + '='*70)
    logger.info('KEY INSIGHTS')
    logger.info('='*70)
    logger.info('\n1. INDEPENDENT LAYERS MATTER')
    logger.info('   - Each layer must measure different aspects')
    logger.info('   - HTF trend is separate from pattern momentum')
    logger.info('   - Volume measure different from price action')
    
    logger.info('\n2. MULTI-TIMEFRAME CONFIRMATION')
    logger.info('   - 15m pattern + 4h trend alignment reduces false signals')
    logger.info('   - Research shows 60-80% accuracy for higher timeframes')
    
    logger.info('\n3. MARKET ENVIRONMENT MATTERS')
    logger.info('   - Patterns aligned with technical support more reliable')
    logger.info('   - Requires checking historical levels')
    
    logger.info('\n4. NOT ALL FILTERS ARE EQUAL')
    logger.info('   - Proper confirmation ≠ Strict filtering')
    logger.info('   - Should improve quality, not just reduce quantity')
    
    logger.info('\n' + '='*70)
    logger.info('NEXT IMPROVEMENTS')
    logger.info('='*70)
    logger.info('1. Add fibonacci level detection')
    logger.info('2. Implement volatility regime detection')
    logger.info('3. Add correlated asset strength check')
    logger.info('4. Implement adaptive confidence thresholds per market regime')
    logger.info('5. Add walk-forward validation')
    
    logger.info('\n' + '='*70)
    logger.info('Test complete')
    logger.info('='*70)
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
