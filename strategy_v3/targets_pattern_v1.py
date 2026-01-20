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
                         min_breakout_pct: float = 0.002) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create labels based on detected chart patterns (double top/bottom).
    
    Architecture:
    1. Detect double top and double bottom patterns
    2. Wait for breakout of neckline/support
    3. Simulate trade from breakout point
    4. Label the pattern completion point based on breakout result
    
    Args:
    - profit_target_pct: Target profit (e.g., 1%)
    - stop_loss_pct: Stop loss (e.g., 1%)
    - max_hold_bars: Maximum hold period
    - min_breakout_pct: Minimum move to confirm breakout
    
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
    
    detector = PatternDetector(min_height_ratio=0.01, min_bars=10, max_bars=50)
    
    logger.info('Detecting chart patterns...')
    
    double_tops = detector.detect_double_top(high, low)
    double_bottoms = detector.detect_double_bottom(high, low)
    
    total_patterns = len(double_tops) + len(double_bottoms)
    logger.info(f'Found {len(double_tops)} double tops and {len(double_bottoms)} double bottoms')
    logger.info(f'Total patterns: {total_patterns}')
    
    profitable_patterns = 0
    unprofitable_patterns = 0
    labeled_patterns = 0
    
    for pattern in double_tops:
        peak2_idx = pattern.indices['peak2']
        neckline = pattern.prices['valley']
        
        if peak2_idx + max_hold_bars >= n:
            continue
        
        breakout_idx = -1
        for i in range(peak2_idx + 1, min(peak2_idx + 50, n)):
            if low[i] <= neckline * (1 - min_breakout_pct):
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
        
        features = detector.extract_features(pattern, high, low, volume)
        features['actual_result'] = result
        features['actual_profit'] = profit
        features_list.append(features)
    
    for pattern in double_bottoms:
        valley2_idx = pattern.indices['valley2']
        support = pattern.prices['valley1']
        
        if valley2_idx + max_hold_bars >= n:
            continue
        
        breakout_idx = -1
        for i in range(valley2_idx + 1, min(valley2_idx + 50, n)):
            if high[i] >= support * (1 + min_breakout_pct):
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
        
        features = detector.extract_features(pattern, high, low, volume)
        features['actual_result'] = result
        features['actual_profit'] = profit
        features_list.append(features)
    
    total_labeled = profitable_patterns + unprofitable_patterns
    if total_labeled > 0:
        win_rate = profitable_patterns / total_labeled * 100
    else:
        win_rate = 0
    
    logger.info('Pattern-Based Label Analysis:')
    logger.info(f'Total labeled patterns: {labeled_patterns}')
    logger.info(f'Profitable patterns: {profitable_patterns}')
    logger.info(f'Unprofitable patterns: {unprofitable_patterns}')
    logger.info(f'Pattern win rate: {win_rate:.2f}%')
    logger.info(f'Total data points: {n}')
    logger.info(f'Labeled data points: {(targets != -1).sum()}')
    logger.info(f'Unlabeled data points: {(targets == -1).sum()}')
    
    return targets, profits
