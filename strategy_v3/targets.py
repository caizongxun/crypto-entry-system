import numpy as np
import pandas as pd
from loguru import logger
from typing import Tuple


def detect_swing_points(high: np.ndarray, low: np.ndarray, left_bars: int = 5, right_bars: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect swing high and swing low points using local extrema detection.
    
    A swing high occurs when a bar's high is >= all bars within left_bars to the left
    AND >= all bars within right_bars to the right.
    
    A swing low occurs when a bar's low is <= all bars within left_bars to the left
    AND <= all bars within right_bars to the right.
    
    Args:
        high: Array of high prices
        low: Array of low prices
        left_bars: Number of bars to check on the left side
        right_bars: Number of bars to check on the right side
        
    Returns:
        - swing_highs: Binary array (1 where swing high detected, 0 otherwise)
        - swing_lows: Binary array (1 where swing low detected, 0 otherwise)
    """
    n = len(high)
    swing_highs = np.zeros(n, dtype=int)
    swing_lows = np.zeros(n, dtype=int)
    
    # Need at least left_bars + 1 + right_bars data points
    if n < left_bars + right_bars + 1:
        return swing_highs, swing_lows
    
    for i in range(left_bars, n - right_bars):
        # Check swing high
        # High at i must be >= max of surrounding bars
        is_swing_high = (
            high[i] >= high[max(0, i - left_bars):i].max() and
            high[i] >= high[i + 1:min(n, i + right_bars + 1)].max()
        )
        if is_swing_high:
            swing_highs[i] = 1
        
        # Check swing low
        # Low at i must be <= min of surrounding bars
        is_swing_low = (
            low[i] <= low[max(0, i - left_bars):i].min() and
            low[i] <= low[i + 1:min(n, i + right_bars + 1)].min()
        )
        if is_swing_low:
            swing_lows[i] = 1
    
    return swing_highs, swing_lows


def identify_reversals(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                      left_bars: int = 5, right_bars: int = 5) -> np.ndarray:
    """
    Identify reversal points based on swing high/low pattern breaks.
    
    A reversal is confirmed when:
    1. We have a swing high followed by a swing low (potential downtrend reversal)
    2. We have a swing low followed by a swing high (potential uptrend reversal)
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        left_bars: Lookback period for swing detection
        right_bars: Lookahead period for swing detection
        
    Returns:
        Array where 1 = reversal point detected, 0 = no reversal
    """
    n = len(high)
    reversals = np.zeros(n, dtype=int)
    
    # Detect all swing points
    swing_highs, swing_lows = detect_swing_points(high, low, left_bars, right_bars)
    
    # Find indices of swings
    high_indices = np.where(swing_highs == 1)[0]
    low_indices = np.where(swing_lows == 1)[0]
    
    # Mark reversals: transitions from high to low or low to high
    for i in range(1, len(high_indices)):
        # Check if there's a low between two highs (high->low->high = downtrend reversal)
        lows_between = low_indices[(low_indices > high_indices[i-1]) & (low_indices < high_indices[i])]
        if len(lows_between) > 0:
            # Mark the lower swing low as reversal point
            reversal_idx = lows_between[np.argmin(low[lows_between])]
            reversals[reversal_idx] = 1
    
    for i in range(1, len(low_indices)):
        # Check if there's a high between two lows (low->high->low = uptrend reversal)
        highs_between = high_indices[(high_indices > low_indices[i-1]) & (high_indices < low_indices[i])]
        if len(highs_between) > 0:
            # Mark the higher swing high as reversal point
            reversal_idx = highs_between[np.argmax(high[highs_between])]
            reversals[reversal_idx] = 1
    
    return reversals


def create_reversal_target(df: pd.DataFrame, lookback: int = 20, 
                          left_bars: int = 5, right_bars: int = 5) -> np.ndarray:
    """
    Create binary reversal target using swing high/low based detection.
    
    Logic:
    - For each candle i, check if there's a swing reversal within next 5-20 candles
    - If reversal found, mark as 1 (reversal opportunity)
    - Otherwise mark as 0
    - Skip lookback period (need past data for features)
    
    Args:
        df: DataFrame with OHLCV data
        lookback: Number of past candles used as features
        left_bars: Lookback period for swing detection
        right_bars: Lookahead period for swing detection
        
    Returns:
        Array of targets (0 or 1) for each candle
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    targets = np.zeros(len(df), dtype=int)
    
    # Identify all reversal points
    reversals = identify_reversals(high, low, close, left_bars, right_bars)
    reversal_indices = np.where(reversals == 1)[0]
    
    logger.info(f'Found {len(reversal_indices)} swing-based reversal points')
    logger.info(f'Reversal detection parameters: left_bars={left_bars}, right_bars={right_bars}')
    
    if len(reversal_indices) == 0:
        logger.warning(f'No reversals found with left_bars={left_bars}, right_bars={right_bars}')
        return targets
    
    in_hold_phase = False
    hold_end_idx = -1
    
    for i in range(lookback, len(df)):
        # Skip if in HOLD phase
        if in_hold_phase:
            if i <= hold_end_idx:
                targets[i] = 0
            else:
                in_hold_phase = False
            continue
        
        # Check if there's a reversal within next 5-20 candles
        lookahead_start = i + 1
        lookahead_end = min(i + 20, len(df))
        
        reversals_ahead = reversal_indices[
            (reversal_indices >= lookahead_start) & (reversal_indices < lookahead_end)
        ]
        
        if len(reversals_ahead) > 0:
            # Reversal point found ahead
            reversal_idx = reversals_ahead[0]
            targets[i] = 1
            
            # Mark subsequent candles as HOLD
            in_hold_phase = True
            hold_end_idx = min(reversal_idx + 5, len(df) - 1)
    
    return targets
