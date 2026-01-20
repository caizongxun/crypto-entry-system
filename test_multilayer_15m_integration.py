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


def calculate_htf_trend(df, htf_window=16):
    """
    Calculate higher timeframe trend.
    For 15m data, window=16 represents ~4 hours (15m * 16)
    Returns: 1 (uptrend), -1 (downtrend), 0 (no clear trend)
    """
    close = df['close'].values
    ma_fast = pd.Series(close).rolling(window=htf_window).mean().values
    ma_slow = pd.Series(close).rolling(window=htf_window * 2).mean().values
    
    trend = np.zeros(len(df), dtype=int)
    trend[ma_fast > ma_slow] = 1
    trend[ma_fast < ma_slow] = -1
    
    return trend


def calculate_price_action_confirmation(df, target):
    """
    Price action confirmation:
    - Bullish pattern should have follow-through
    - Bearish pattern should have follow-through
    Returns: 1 if confirmed, 0 otherwise
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    confirmation = np.zeros(len(df), dtype=int)
    pattern_mask = target != -1
    
    for i in range(1, min(len(df) - 2, len(df))):
        if not pattern_mask[i]:
            continue
        
        # Check for follow-through (not just spike)
        if target[i] == 1 and i + 2 < len(df):  # Bullish
            avg_close_next = (close[i+1] + close[i+2]) / 2
            if close[i] > close[i-1] and avg_close_next > close[i]:
                confirmation[i] = 1
        
        elif target[i] == 0 and i + 2 < len(df):  # Bearish
            avg_close_next = (close[i+1] + close[i+2]) / 2
            if close[i] < close[i-1] and avg_close_next < close[i]:
                confirmation[i] = 1
    
    return confirmation


def calculate_volume_confirmation(df, target, volume_window=20):
    """
    Volume confirmation based on absolute volume increase.
    Breakout volume should be > average + 1 std dev (~40% increase)
    """
    volume = df['volume'].values
    avg_volume = pd.Series(volume).rolling(window=volume_window).mean().values
    std_volume = pd.Series(volume).rolling(window=volume_window).std().values
    
    confirmation = np.zeros(len(df), dtype=int)
    pattern_mask = target != -1
    
    for idx in np.where(pattern_mask)[0]:
        if idx < volume_window:
            continue
        
        current_vol = volume[idx]
        avg_vol = avg_volume[idx]
        std_vol = std_volume[idx]
        
        # Volume confirmation: current > average + 1 std dev
        if not np.isnan(std_vol) and current_vol > (avg_vol + std_vol):
            confirmation[idx] = 1
    
    return confirmation


def calculate_technical_support(df, target, lookback=100):
    """
    Check if pattern aligns with technical support/resistance.
    Higher reliability when pattern forms at historical levels.
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    confirmation = np.zeros(len(df), dtype=int)
    pattern_mask = target != -1
    
    for idx in np.where(pattern_mask)[0]:
        if idx < lookback:
            continue
        
        # Get recent swing levels
        recent_high = np.max(high[idx-lookback:idx])
        recent_low = np.min(low[idx-lookback:idx])
        range_size = recent_high - recent_low
        
        if range_size == 0:
            continue
        
        current_price = close[idx]
        
        # Check if at support/resistance (within 1% of historical level)
        if abs(current_price - recent_high) < range_size * 0.01:
            confirmation[idx] = 1
        elif abs(current_price - recent_low) < range_size * 0.01:
            confirmation[idx] = 1
        # Also check round numbers (BTC loves 30000, 40000, etc)
        elif current_price % 1000 < range_size * 0.01:
            confirmation[idx] = 1
    
    return confirmation


def calculate_rsi_timing(df):
    """
    Use RSI for timing bonus, not filtering.
    Returns: 1 if RSI is extreme (< 30 or > 70), 0 otherwise
    """
    if 'extremum_rsi' in df.columns:
        rsi = df['extremum_rsi'].values
    else:
        # Calculate RSI if not provided
        close = df['close'].values
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).rolling(window=14).mean().values
        avg_loss = pd.Series(loss).rolling(window=14).mean().values
        
        rs = np.divide(avg_gain, avg_loss, where=avg_loss != 0, out=np.zeros_like(avg_gain))
        rsi = 100 - (100 / (1 + rs))
    
    timing = np.zeros(len(df), dtype=int)
    timing[(rsi < 30) | (rsi > 70)] = 1
    
    return timing


def apply_multilayer_confirmation(df, multilayer_features, target, min_confirmations=2):
    """
    Apply research-driven multi-layer confirmation.
    Based on: StockTiming.com 11-year backtest + Reddit multi-confirmation analysis
    
    Returns:
        filtered_labels: Labels after multi-layer confirmation
        confidence_scores: Confidence score for each pattern (0-5)
    """
    filtered_labels = target.copy()
    confidence_scores = np.zeros(len(df), dtype=int)
    
    pattern_mask = target != -1
    
    if not pattern_mask.any():
        return filtered_labels, confidence_scores
    
    # Calculate each independent confirmation layer
    htf_trend = calculate_htf_trend(df)
    price_action = calculate_price_action_confirmation(df, target)
    volume_conf = calculate_volume_confirmation(df, target)
    technical = calculate_technical_support(df, target)
    rsi_timing = calculate_rsi_timing(df)
    
    pattern_indices = np.where(pattern_mask)[0]
    
    for idx in pattern_indices:
        score = 0
        
        # Layer 1: HTF Trend alignment (must align with pattern direction)
        if target[idx] == 1 and htf_trend[idx] == 1:
            score += 1
        elif target[idx] == 0 and htf_trend[idx] == -1:
            score += 1
        
        # Layer 2: Price Action confirmation
        score += price_action[idx]
        
        # Layer 3: Volume confirmation
        score += volume_conf[idx]
        
        # Layer 4: Technical Support
        score += technical[idx]
        
        # Layer 5: RSI Timing (bonus point only)
        score += rsi_timing[idx]
        
        confidence_scores[idx] = score
        
        # Filter: require at least 2 independent layers
        if score < min_confirmations:
            filtered_labels[idx] = 0  # Mark as low confidence
    
    return filtered_labels, confidence_scores


def main():
    logger.info('='*70)
    logger.info('Multi-Layer Feature Integration Test for 15m Framework')
    logger.info('='*70)
    logger.info('Research-Driven: Independent layers + HTF confirmation')
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
    logger.info('Confirmation Layers (Independent & Complementary)')
    logger.info('='*70)
    logger.info('1. HTF Trend (4h MA) - Pattern aligns with higher timeframe')
    logger.info('2. Price Action - Proper follow-through on breakout')
    logger.info('3. Volume - Absolute increase > avg + 1 std dev')
    logger.info('4. Technical Support - Aligns with historical levels')
    logger.info('5. RSI Timing - Bonus point for extreme readings')
    
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
    logger.info('AFTER MULTI-LAYER CONFIRMATION (2+ layers)')
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
    for conf_level in range(0, 6):
        count = (confidence_scores[pattern_indices] == conf_level).sum()
        if count > 0:
            pct = count / len(pattern_indices) * 100
            logger.info(f'Confidence level {conf_level}: {count} patterns ({pct:.1f}%)')
    
    logger.info('\n' + '='*70)
    logger.info('BREAKDOWN BY CONFIDENCE LEVEL')
    logger.info('='*70)
    
    for conf_level in range(2, 6):
        conf_mask = confidence_scores == conf_level
        profitable_at_level = ((target == 1) & conf_mask).sum()
        total_at_level = conf_mask.sum()
        if total_at_level > 0:
            win_at_level = profitable_at_level / total_at_level * 100
            logger.info(f'Level {conf_level}: {total_at_level} trades, {win_at_level:.2f}% win rate')
    
    logger.info('\n' + '='*70)
    logger.info('RESEARCH INSIGHTS')
    logger.info('='*70)
    logger.info('\nKey findings (from published research):')
    logger.info('  StockTiming.com (11-year backtest):')
    logger.info('    - Double Top/Bottom base: 78%')
    logger.info('    - With proper confirmations: 90%')
    logger.info(f'  Our baseline: {baseline_win_rate:.1f}%')
    logger.info(f'  Our high confidence: {high_conf_win_rate:.1f}%')
    
    logger.info('\n' + '='*70)
    logger.info('SUMMARY')
    logger.info('='*70)
    logger.info(f'Baseline win rate: {baseline_win_rate:.2f}%')
    logger.info(f'High confidence win rate (2+ layers): {high_conf_win_rate:.2f}%')
    logger.info(f'Expected with all confirmations: 35-42%')
    
    logger.info('\n' + '='*70)
    logger.info('Integration test complete')
    logger.info('='*70)
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
