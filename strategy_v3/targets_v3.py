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


def simulate_trade(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   entry_idx: int, is_short: bool = False,
                   profit_target: float = 0.005, stop_loss: float = 0.01,
                   max_bars: int = 20) -> Tuple[int, float]:
    """
    Simulate a real trade from entry point.
    """
    n = len(close)
    if entry_idx + 1 >= n:
        return 0, 0.0
    
    entry_price = close[entry_idx]
    
    for i in range(entry_idx + 1, min(entry_idx + max_bars + 1, n)):
        if is_short:
            current_high = high[i]
            current_low = low[i]
            
            profit = (entry_price - current_low) / entry_price
            loss = (current_high - entry_price) / entry_price
            
            if profit >= profit_target:
                return 1, profit
            if loss >= stop_loss:
                return 0, -loss
        else:
            current_high = high[i]
            current_low = low[i]
            
            profit = (current_high - entry_price) / entry_price
            loss = (entry_price - current_low) / entry_price
            
            if profit >= profit_target:
                return 1, profit
            if loss >= stop_loss:
                return 0, -loss
    
    final_price = close[min(entry_idx + max_bars, n - 1)]
    final_profit = (final_price - entry_price) / entry_price if not is_short else (entry_price - final_price) / entry_price
    
    if final_profit >= profit_target:
        return 1, final_profit
    else:
        return 0, final_profit


def create_reversal_target_v3(df: pd.DataFrame, lookback: int = 20,
                              left_bars: int = 5, right_bars: int = 5,
                              profit_target_pct: float = 0.005,
                              stop_loss_pct: float = 0.01,
                              max_hold_bars: int = 20,
                              lead_bars: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create LEAD-LAG labels for reversal prediction.
    
    CRITICAL FIX: Only label swing points, ignore non-swing bars
    
    Architecture:
    1. Detect swing points (high and low)
    2. For each swing point at index i:
       - Simulate trade from i
       - Place label at i - lead_bars
    3. All other bars remain unlabeled (NaN)
    
    This ensures:
    - Model only learns from actual trading opportunities
    - No bias from labeling non-swing points
    - Clean training signal
    
    Args:
    - lead_bars: How many bars ahead to place the label (default 3)
    - profit_target_pct: Profit target (e.g., 0.5%)
    - stop_loss_pct: Stop loss level (e.g., 1%)
    - max_hold_bars: Maximum bars to hold (e.g., 20)
    
    Returns:
    - targets: Array with labels only at lead bars before swing points
               Non-labeled bars are -1 (to be ignored in training)
    - profits: Actual profit/loss percentages
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    n = len(df)
    targets = np.full(n, -1, dtype=int)
    profits = np.zeros(n, dtype=float)
    
    swing_highs, swing_lows = detect_swing_points(high, low, left_bars, right_bars)
    high_indices = np.where(swing_highs == 1)[0]
    low_indices = np.where(swing_lows == 1)[0]
    
    logger.info(f'Found {len(high_indices)} swing highs and {len(low_indices)} swing lows')
    logger.info(f'Lead bars: {lead_bars} (features at t predict reversal at t+{lead_bars})')
    
    profitable_count = 0
    unprofitable_count = 0
    labeled_count = 0
    
    for idx in high_indices:
        if idx + max_hold_bars >= n:
            continue
        
        result, profit = simulate_trade(
            high, low, close, idx,
            is_short=True,
            profit_target=profit_target_pct,
            stop_loss=stop_loss_pct,
            max_bars=max_hold_bars
        )
        
        if result == 1:
            profitable_count += 1
        else:
            unprofitable_count += 1
        
        label_idx = idx - lead_bars
        if label_idx >= 0:
            targets[label_idx] = result
            profits[label_idx] = profit
            labeled_count += 1
    
    for idx in low_indices:
        if idx + max_hold_bars >= n:
            continue
        
        result, profit = simulate_trade(
            high, low, close, idx,
            is_short=False,
            profit_target=profit_target_pct,
            stop_loss=stop_loss_pct,
            max_bars=max_hold_bars
        )
        
        if result == 1:
            profitable_count += 1
        else:
            unprofitable_count += 1
        
        label_idx = idx - lead_bars
        if label_idx >= 0:
            targets[label_idx] = result
            profits[label_idx] = profit
            labeled_count += 1
    
    total_swings = profitable_count + unprofitable_count
    win_rate = profitable_count / total_swings * 100 if total_swings > 0 else 0
    
    logger.info(f'Swing Point Simulation:')
    logger.info(f'Profitable swings: {profitable_count}')
    logger.info(f'Unprofitable swings: {unprofitable_count}')
    logger.info(f'Total swings: {total_swings}')
    logger.info(f'Win rate (at swing points): {win_rate:.2f}%')
    
    logger.info(f'Labels created: {labeled_count}')
    profitable_labels = (targets == 1).sum()
    unprofitable_labels = (targets == 0).sum()
    logger.info(f'Labels predicting profitable reversal: {profitable_labels}')
    logger.info(f'Labels predicting unprofitable reversal: {unprofitable_labels}')
    logger.info(f'Unlabeled bars (ignored in training): {(targets == -1).sum()}')
    
    if labeled_count > 0:
        label_win_rate = profitable_labels / labeled_count * 100
        logger.info(f'Win rate among labeled bars: {label_win_rate:.2f}%')
    
    return targets, profits
