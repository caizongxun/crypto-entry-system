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
    
    CRITICAL CHANGE: Features at time t predict reversal profit at time t+lead_bars
    
    This avoids look-ahead bias and allows models to learn:
    "When these market microstructure features appear NOW,
     a profitable reversal will occur SOON"
    
    Args:
    - lead_bars: How many bars ahead to place the label (default 3)
                 This is the time window for features to "predict" the reversal
    - profit_target_pct: Profit target (e.g., 0.5%)
    - stop_loss_pct: Stop loss level (e.g., 1%)
    - max_hold_bars: Maximum bars to hold (e.g., 20)
    
    Returns:
    - targets: Binary array (1 if profitable reversal within lead_bars, 0 otherwise)
    - profits: Actual profit/loss percentages for analysis
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
    logger.info(f'Lead bars: {lead_bars} (features at t predict reversal at t+{lead_bars})')
    
    profitable_count = 0
    unprofitable_count = 0
    forward_looking_count = 0
    
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
            forward_looking_count += 1
    
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
            forward_looking_count += 1
    
    total_swings = profitable_count + unprofitable_count
    win_rate = profitable_count / total_swings * 100 if total_swings > 0 else 0
    
    logger.info(f'Swing Point Simulation:')
    logger.info(f'Profitable swings: {profitable_count}')
    logger.info(f'Unprofitable swings: {unprofitable_count}')
    logger.info(f'Total swings: {total_swings}')
    logger.info(f'Win rate (at swing points): {win_rate:.2f}%')
    
    logger.info(f'Forward-looking labels created: {forward_looking_count}')
    logger.info(f'Labels that predict profitable reversal: {(targets == 1).sum()}')
    logger.info(f'Labels that predict unprofitable reversal: {(targets == 0).sum()}')
    
    if forward_looking_count > 0:
        profitable_labels = (targets == 1).sum()
        label_win_rate = profitable_labels / forward_looking_count * 100
        logger.info(f'Label win rate: {label_win_rate:.2f}% (at feature time t)')
    
    return targets, profits
