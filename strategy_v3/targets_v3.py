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
    
    Args:
    - entry_idx: Entry candle index
    - is_short: True for short (sell), False for long (buy)
    - profit_target: Profit target (e.g., 0.5%)
    - stop_loss: Stop loss level (e.g., 1%)
    - max_bars: Maximum bars to hold trade
    
    Returns:
    - result: 1 if profitable, 0 if stopped out
    - exit_profit: Actual profit/loss percentage
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
                              max_hold_bars: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create labels based on REAL TRADING SIMULATION.
    
    CRITICAL: This simulates actual trade execution from swing points.
    
    - At swing high: SHORT (sell)
    - At swing low: LONG (buy)
    - Trade succeeds if profit_target reached before stop_loss
    - Trade fails if stop_loss hit before profit_target
    - Trade timeout if max_hold_bars exceeded
    
    This avoids look-ahead bias: we only use data AT and AFTER entry,
    simulating real trading decisions.
    
    Args:
    - profit_target_pct: Profit target (e.g., 0.5%)
    - stop_loss_pct: Stop loss level (e.g., 1%)
    - max_hold_bars: Maximum bars to hold (e.g., 20)
    
    Returns:
    - targets: Binary array (1 if trade profitable, 0 if stopped out)
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
    
    profitable_trades = 0
    stopped_out_trades = 0
    timeout_trades = 0
    total_profit = 0.0
    
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
        
        targets[idx] = result
        profits[idx] = profit
        total_profit += profit
        
        if result == 1:
            profitable_trades += 1
        else:
            stopped_out_trades += 1
    
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
        
        targets[idx] = result
        profits[idx] = profit
        total_profit += profit
        
        if result == 1:
            profitable_trades += 1
        else:
            stopped_out_trades += 1
    
    total_trades = profitable_trades + stopped_out_trades
    
    logger.info(f'Trade Simulation Results:')
    logger.info(f'Profitable trades (target=1): {profitable_trades}')
    logger.info(f'Stopped out trades (target=0): {stopped_out_trades}')
    logger.info(f'Total trades: {total_trades}')
    logger.info(f'Win rate: {profitable_trades/total_trades*100 if total_trades > 0 else 0:.2f}%')
    logger.info(f'Total P&L: {total_profit*100:.2f}%')
    logger.info(f'Average P&L per trade: {total_profit/total_trades*100 if total_trades > 0 else 0:.2f}%')
    
    if profitable_trades > 0:
        avg_profit = profits[profits > 0].mean()
        logger.info(f'Average profit (winning trades): {avg_profit*100:.2f}%')
    if stopped_out_trades > 0:
        avg_loss = profits[profits < 0].mean()
        logger.info(f'Average loss (stopped out): {avg_loss*100:.2f}%')
    
    return targets, profits
