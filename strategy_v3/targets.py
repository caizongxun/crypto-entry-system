import numpy as np
import pandas as pd
from loguru import logger
from typing import List, Tuple


def calculate_zigzag(high: np.ndarray, low: np.ndarray, threshold_pct: float = 2.0) -> Tuple[np.ndarray, List[int]]:
    """
    Calculate Zigzag pattern based on price swings.
    
    Args:
        high: Array of high prices
        low: Array of low prices
        threshold_pct: Minimum percentage move to confirm swing
        
    Returns:
        - Array of zigzag values (0 for no zigzag point, 1 for high, -1 for low)
        - List of swing indices
    """
    zigzag = np.zeros(len(high), dtype=int)
    swings = []
    
    if len(high) < 3:
        return zigzag, swings
    
    current_high_idx = 0
    current_low_idx = 0
    current_high = high[0]
    current_low = low[0]
    last_swing_idx = 0
    last_swing_type = None
    
    for i in range(1, len(high)):
        if high[i] > current_high:
            current_high = high[i]
            current_high_idx = i
        
        if low[i] < current_low:
            current_low = low[i]
            current_low_idx = i
        
        # Check if high-to-low swing (reversal from up to down)
        if last_swing_type != 'low':
            swing_pct = (current_high - current_low) / current_low * 100
            if swing_pct >= threshold_pct and current_low_idx > current_high_idx:
                if last_swing_idx > 0:
                    zigzag[current_high_idx] = 1
                    swings.append(('high', current_high_idx, current_high))
                    last_swing_type = 'high'
                    last_swing_idx = current_high_idx
                    current_low = low[current_low_idx]
                    current_low_idx = current_low_idx
        
        # Check if low-to-high swing (reversal from down to up)
        if last_swing_type != 'high':
            swing_pct = (current_high - current_low) / current_low * 100
            if swing_pct >= threshold_pct and current_high_idx > current_low_idx:
                if last_swing_idx > 0:
                    zigzag[current_low_idx] = -1
                    swings.append(('low', current_low_idx, current_low))
                    last_swing_type = 'low'
                    last_swing_idx = current_low_idx
                    current_high = high[current_high_idx]
                    current_high_idx = current_high_idx
    
    return zigzag, swings


def identify_zigzag_reversals(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                             threshold_pct: float = 2.0) -> np.ndarray:
    """
    Identify reversal points using zigzag pattern.
    
    A zigzag reversal is confirmed when:
    1. Price makes a significant swing (threshold_pct% move)
    2. Direction changes (from up to down or down to up)
    3. New high or low is formed and then broken
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        threshold_pct: Minimum percentage move for zigzag swing
        
    Returns:
        Array where 1 = reversal point, 0 = not reversal
    """
    reversal_points = np.zeros(len(high), dtype=int)
    
    if len(high) < 5:
        return reversal_points
    
    # Track local highs and lows
    local_high_idx = 0
    local_low_idx = 0
    local_high = high[0]
    local_low = low[0]
    direction = None  # 'up' or 'down'
    
    for i in range(1, len(high) - 1):
        curr_high = high[i]
        curr_low = low[i]
        
        # Update running high and low
        if curr_high > local_high:
            local_high = curr_high
            local_high_idx = i
        
        if curr_low < local_low:
            local_low = curr_low
            local_low_idx = i
        
        # Detect reversal: price was going up, now goes down significantly
        if direction == 'up':
            swing_pct = (local_high - low[i]) / low[i] * 100
            if swing_pct >= threshold_pct and local_low_idx > local_high_idx:
                reversal_points[local_high_idx] = 1
                local_high = high[i]
                local_high_idx = i
                direction = 'down'
        
        # Detect reversal: price was going down, now goes up significantly
        elif direction == 'down':
            swing_pct = (high[i] - local_low) / local_low * 100
            if swing_pct >= threshold_pct and local_high_idx > local_low_idx:
                reversal_points[local_low_idx] = 1
                local_low = low[i]
                local_low_idx = i
                direction = 'up'
        
        # Initialize direction if not set
        else:
            if high[i] > high[i-1]:
                direction = 'up'
            elif high[i] < high[i-1]:
                direction = 'down'
    
    return reversal_points


def create_reversal_target(df: pd.DataFrame, lookback: int = 20, atr_mult: float = 1.5,
                          profit_target_ratio: float = 1.5, zigzag_threshold_pct: float = 2.0) -> np.ndarray:
    """
    Create binary reversal target using Zigzag-based reversal identification.
    
    Logic:
    - For each candle, check if a zigzag reversal point is found ahead (within lookback distance)
    - If reversal found within next N candles, mark as 1 (reversal opportunity)
    - Otherwise mark as 0
    - Also mark subsequent candles in the reversal move as 0 (HOLD phase)
    
    Args:
        df: DataFrame with OHLCV and atr_14
        lookback: Number of past candles used as features
        atr_mult: ATR multiplier for context (not used directly in zigzag)
        profit_target_ratio: Profit target ratio for context
        zigzag_threshold_pct: Minimum percentage move for zigzag swing detection
        
    Returns:
        Array of targets (0 or 1) for each candle
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    atr = df['atr_14'].values
    
    targets = np.zeros(len(df), dtype=int)
    
    # Identify all zigzag reversal points
    zigzag_reversals = identify_zigzag_reversals(high, low, close, zigzag_threshold_pct)
    
    reversal_indices = np.where(zigzag_reversals == 1)[0]
    
    logger.info(f'Found {len(reversal_indices)} zigzag reversal points')
    
    if len(reversal_indices) == 0:
        logger.warning('No zigzag reversals found with threshold={zigzag_threshold_pct}%')
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
        
        # Check if there's a zigzag reversal within next 5-20 candles
        lookahead_start = i + 1
        lookahead_end = min(i + 20, len(df))
        
        reversals_ahead = reversal_indices[
            (reversal_indices >= lookahead_start) & (reversal_indices < lookahead_end)
        ]
        
        if len(reversals_ahead) > 0:
            # Reversal point found ahead
            reversal_idx = reversals_ahead[0]
            targets[i] = 1
            
            # Mark subsequent candles as HOLD until reversal completes
            # Estimate hold duration as distance to reversal point
            in_hold_phase = True
            hold_end_idx = min(reversal_idx + 5, len(df) - 1)
    
    return targets
