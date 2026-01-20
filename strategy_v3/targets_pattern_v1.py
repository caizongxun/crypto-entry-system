import numpy as np
import pandas as pd
from loguru import logger
from typing import Tuple
from strategy_v3.pattern_detector import PatternDetector
from strategy_v3.targets_v3 import simulate_trade


def create_pattern_labels(df: pd.DataFrame,
                         profit_target_pct: float = 0.01,
                         stop_loss_pct: float = 0.01,
                         max_hold_bars: int = 20,
                         min_breakout_pct: float = 0.005,
                         min_quality_score: float = 60.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create labels based on detected chart patterns (double top/bottom).
    
    Improved architecture:
    1. Detect double top and double bottom patterns
    2. Apply quality filtering (min 60/100 score)
    3. Wait for confirmed breakout
    4. Simulate trade from breakout point
    5. Label the pattern completion point based on breakout result
    
    Args:
    - profit_target_pct: Target profit (e.g., 1%)
    - stop_loss_pct: Stop loss (e.g., 1%)
    - max_hold_bars: Maximum hold period
    - min_breakout_pct: Minimum move to confirm breakout (0.5%)
    - min_quality_score: Minimum quality score (0-100) to use pattern
    
    Returns:
    - targets: Array with pattern-based labels
    - profits: Actual P&L percentages
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    volume = df['volume'].values if 'volume' in df.columns else np.ones(len(df))
    
    n = len(df)
    targets = np.full(n, -1, dtype=int)
    profits = np.zeros(n, dtype=float)
    features_list = []
    
    detector = PatternDetector(
        min_height_ratio=0.005,      # 0.5% max height difference
        min_bars=15,                 # At least 15 bars for pattern formation
        max_bars=40,                 # At most 40 bars to avoid noise
        min_quality_score=min_quality_score
    )
    
    logger.info('Detecting chart patterns...')
    
    double_tops = detector.detect_double_top(high, low, volume)
    double_bottoms = detector.detect_double_bottom(high, low, volume)
    
    total_patterns = len(double_tops) + len(double_bottoms)
    logger.info(f'Found {len(double_tops)} quality double tops and {len(double_bottoms)} quality double bottoms')
    logger.info(f'Total high-quality patterns: {total_patterns}')
    
    profitable_patterns = 0
    unprofitable_patterns = 0
    labeled_patterns = 0
    avg_quality_score = 0.0
    
    for pattern in double_tops:
        peak2_idx = pattern.indices['peak2']
        neckline = pattern.prices['valley']
        quality_score = pattern.quality_score
        
        if peak2_idx + max_hold_bars >= n:
            continue
        
        # More strict breakout condition
        breakout_idx = -1
        for i in range(peak2_idx + 1, min(peak2_idx + 50, n)):
            if low[i] <= neckline * (1 - min_breakout_pct):
                # Confirm breakout with next bar
                if i + 1 < n and low[i + 1] <= neckline * (1 - min_breakout_pct * 0.5):
                    breakout_idx = i
                    break
        
        if breakout_idx == -1:
            continue
        
        result, profit = simulate_trade(
            high, low, close, breakout_idx,
            is_short=True,
            profit_target=profit_target_pct,
            stop_loss=stop_loss_pct,
            max_bars=max_hold_bars
        )
        
        if result == 1:
            profitable_patterns += 1
        else:
            unprofitable_patterns += 1
        
        targets[peak2_idx] = result
        profits[peak2_idx] = profit
        labeled_patterns += 1
        avg_quality_score += quality_score
        
        features = detector.extract_features(pattern, high, low, volume)
        features['actual_result'] = result
        features['actual_profit'] = profit
        features_list.append(features)
    
    for pattern in double_bottoms:
        valley2_idx = pattern.indices['valley2']
        support = pattern.prices['valley1']
        quality_score = pattern.quality_score
        
        if valley2_idx + max_hold_bars >= n:
            continue
        
        # More strict breakout condition
        breakout_idx = -1
        for i in range(valley2_idx + 1, min(valley2_idx + 50, n)):
            if high[i] >= support * (1 + min_breakout_pct):
                # Confirm breakout with next bar
                if i + 1 < n and high[i + 1] >= support * (1 + min_breakout_pct * 0.5):
                    breakout_idx = i
                    break
        
        if breakout_idx == -1:
            continue
        
        result, profit = simulate_trade(
            high, low, close, breakout_idx,
            is_short=False,
            profit_target=profit_target_pct,
            stop_loss=stop_loss_pct,
            max_bars=max_hold_bars
        )
        
        if result == 1:
            profitable_patterns += 1
        else:
            unprofitable_patterns += 1
        
        targets[valley2_idx] = result
        profits[valley2_idx] = profit
        labeled_patterns += 1
        avg_quality_score += quality_score
        
        features = detector.extract_features(pattern, high, low, volume)
        features['actual_result'] = result
        features['actual_profit'] = profit
        features_list.append(features)
    
    total_labeled = profitable_patterns + unprofitable_patterns
    if total_labeled > 0:
        win_rate = profitable_patterns / total_labeled * 100
        avg_quality_score = avg_quality_score / total_labeled
    else:
        win_rate = 0
        avg_quality_score = 0
    
    logger.info('Improved Pattern-Based Label Analysis:')
    logger.info(f'Total detected patterns: {total_patterns}')
    logger.info(f'Total labeled patterns: {labeled_patterns}')
    logger.info(f'Profitable patterns: {profitable_patterns}')
    logger.info(f'Unprofitable patterns: {unprofitable_patterns}')
    logger.info(f'Pattern win rate: {win_rate:.2f}%')
    logger.info(f'Average quality score: {avg_quality_score:.1f}/100')
    logger.info(f'Total data points: {n}')
    logger.info(f'Labeled data points: {(targets != -1).sum()}')
    logger.info(f'Unlabeled data points: {(targets == -1).sum()}')
    
    return targets, profits
