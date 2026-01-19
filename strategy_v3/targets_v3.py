import numpy as np
import pandas as pd
from loguru import logger
from typing import Tuple


def detect_swing_points(high: np.ndarray, low: np.ndarray, left_bars: int = 5, right_bars: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect swing high and swing low points.
    """
    n = len(high)
    swing_highs = np.zeros(n, dtype=int)
    swing_lows = np.zeros(n, dtype=int)
    
    if n < left_bars + right_bars + 1:
        return swing_highs, swing_lows
    
    for i in range(left_bars, n - right_bars):
        is_swing_high = (
            high[i] >= high[max(0, i - left_bars):i].max() and
            high[i] >= high[i + 1:min(n, i + right_bars + 1)].max()
        )
        if is_swing_high:
            swing_highs[i] = 1
        
        is_swing_low = (
            low[i] <= low[max(0, i - left_bars):i].min() and
            low[i] <= low[i + 1:min(n, i + right_bars + 1)].min()
        )
        if is_swing_low:
            swing_lows[i] = 1
    
    return swing_highs, swing_lows


def create_reversal_target_v3(df: pd.DataFrame, lookback: int = 20,
                              left_bars: int = 5, right_bars: int = 5,
                              profit_target_pct: float = 0.005,
                              future_bars: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create reversal targets based on actual profitability.
    
    CRITICAL CHANGE: Labels are now defined by whether a swing point
    is followed by profitable movement within future_bars candles.
    
    Not all swing points are labeled as reversals - only those that
    result in at least profit_target_pct gain.
    
    Args:
    - profit_target_pct: Minimum profit (default 0.5%)
    - future_bars: How many candles ahead to check (default 10)
    
    Returns:
    - targets: Binary array (1 if profitable reversal, 0 otherwise)
    - strengths: Profit percentages for analysis (can be negative)
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    n = len(df)
    targets = np.zeros(n, dtype=int)
    profits = np.zeros(n, dtype=float)
    
    swing_highs, swing_lows = detect_swing_points(high, low, left_bars, right_bars)
    high_indices = np.where(swing_highs == 1)[0]
    low_indices = np.where(swing_lows == 1)[0]
    
    logger.info(f'Found {len(high_indices)} swing highs and {len(low_indices)} swing lows')
    
    profitable_count = 0
    unprofitable_count = 0
    
    for idx in high_indices:
        if idx + future_bars >= n:
            continue
        
        future_low = low[idx + 1:idx + future_bars + 1].min()
        profit = (future_low - close[idx]) / close[idx]
        profits[idx] = profit
        
        if profit <= -profit_target_pct:
            targets[idx] = 1
            profitable_count += 1
        else:
            unprofitable_count += 1
    
    for idx in low_indices:
        if idx + future_bars >= n:
            continue
        
        future_high = high[idx + 1:idx + future_bars + 1].max()
        profit = (future_high - close[idx]) / close[idx]
        profits[idx] = profit
        
        if profit >= profit_target_pct:
            targets[idx] = 1
            profitable_count += 1
        else:
            unprofitable_count += 1
    
    logger.info(f'Profitable reversals: {profitable_count} ({profitable_count/(profitable_count + unprofitable_count)*100:.2f}%)')
    logger.info(f'Unprofitable reversals: {unprofitable_count} ({unprofitable_count/(profitable_count + unprofitable_count)*100:.2f}%)')
    logger.info(f'Total swing points: {profitable_count + unprofitable_count}')
    
    if profitable_count == 0:
        logger.warning(f'No profitable reversals found with target {profit_target_pct*100:.2f}%')
    else:
        profitable_profits = profits[targets == 1]
        logger.info(f'Profit distribution: min={profitable_profits.min()*100:.2f}%, max={profitable_profits.max()*100:.2f}%, mean={profitable_profits.mean()*100:.2f}%')
    
    return targets, profits
