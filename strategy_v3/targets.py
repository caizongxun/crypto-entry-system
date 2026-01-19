import numpy as np
import pandas as pd
from loguru import logger
from typing import List, Tuple


def identify_zigzag_reversals(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                             threshold_pct: float = 2.0, max_lookback: int = 50) -> np.ndarray:
    """
    Identify reversal points using zigzag pattern with time constraint.
    
    Key improvement: Limit each swing to max_lookback candles to prevent
    oversized swings that span hundreds of candles.
    
    A zigzag reversal is confirmed when:
    1. Price makes a significant swing (threshold_pct% move)
    2. Direction changes (from up to down or down to up)
    3. Swing does not exceed max_lookback candles
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        threshold_pct: Minimum percentage move for zigzag swing
        max_lookback: Maximum candles allowed for one swing (prevents oversized swings)
        
    Returns:
        Array where 1 = reversal point, 0 = not reversal
    """
    reversal_points = np.zeros(len(high), dtype=int)
    
    if len(high) < 5:
        return reversal_points
    
    last_reversal_idx = 0
    last_reversal_type = None
    
    i = 0
    while i < len(high) - 1:
        # If last reversal was 'high', look for 'low'
        if last_reversal_type == 'high':
            local_low = high[i]
            local_low_idx = i
            swing_start = i
            
            # Scan ahead for significant down move within max_lookback
            for j in range(i, min(i + max_lookback, len(high))):
                if low[j] < local_low:
                    local_low = low[j]
                    local_low_idx = j
            
            # Check if significant down move found
            swing_pct = (high[last_reversal_idx] - local_low) / local_low * 100
            if swing_pct >= threshold_pct:
                reversal_points[local_low_idx] = 1
                last_reversal_idx = local_low_idx
                last_reversal_type = 'low'
                i = local_low_idx + 1
            else:
                i += 1
        
        # If last reversal was 'low', look for 'high'
        elif last_reversal_type == 'low':
            local_high = low[i]
            local_high_idx = i
            swing_start = i
            
            # Scan ahead for significant up move within max_lookback
            for j in range(i, min(i + max_lookback, len(high))):
                if high[j] > local_high:
                    local_high = high[j]
                    local_high_idx = j
            
            # Check if significant up move found
            swing_pct = (local_high - low[last_reversal_idx]) / low[last_reversal_idx] * 100
            if swing_pct >= threshold_pct:
                reversal_points[local_high_idx] = 1
                last_reversal_idx = local_high_idx
                last_reversal_type = 'high'
                i = local_high_idx + 1
            else:
                i += 1
        
        # Initialize: find first significant move (either up or down)
        else:
            # Try to find initial high
            local_high = high[i]
            local_high_idx = i
            for j in range(i, min(i + max_lookback, len(high))):
                if high[j] > local_high:
                    local_high = high[j]
                    local_high_idx = j
            
            # Try to find initial low from starting point
            local_low = low[i]
            local_low_idx = i
            for j in range(i, min(i + max_lookback, len(high))):
                if low[j] < local_low:
                    local_low = low[j]
                    local_low_idx = j
            
            # Determine which extreme is closer and more significant
            up_swing_pct = (local_high - low[i]) / low[i] * 100
            down_swing_pct = (high[i] - local_low) / local_low * 100
            
            if up_swing_pct >= threshold_pct and local_high_idx <= local_low_idx:
                # Up move is primary
                reversal_points[local_high_idx] = 1
                last_reversal_idx = local_high_idx
                last_reversal_type = 'high'
                i = local_high_idx + 1
            elif down_swing_pct >= threshold_pct and local_low_idx <= local_high_idx:
                # Down move is primary
                reversal_points[local_low_idx] = 1
                last_reversal_idx = local_low_idx
                last_reversal_type = 'low'
                i = local_low_idx + 1
            else:
                i += 1
    
    return reversal_points


def create_reversal_target(df: pd.DataFrame, lookback: int = 20, atr_mult: float = 1.5,
                          profit_target_ratio: float = 1.5, zigzag_threshold_pct: float = 2.0,
                          zigzag_max_lookback: int = 50) -> np.ndarray:
    """
    Create binary reversal target using improved Zigzag-based reversal identification.
    
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
        zigzag_max_lookback: Maximum candles allowed for one swing
        
    Returns:
        Array of targets (0 or 1) for each candle
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    atr = df['atr_14'].values
    
    targets = np.zeros(len(df), dtype=int)
    
    # Identify all zigzag reversal points with time constraint
    zigzag_reversals = identify_zigzag_reversals(
        high, low, close, 
        threshold_pct=zigzag_threshold_pct,
        max_lookback=zigzag_max_lookback
    )
    
    reversal_indices = np.where(zigzag_reversals == 1)[0]
    
    logger.info(f'Found {len(reversal_indices)} zigzag reversal points')
    logger.info(f'Zigzag parameters: threshold={zigzag_threshold_pct}%, max_lookback={zigzag_max_lookback}')
    
    if len(reversal_indices) == 0:
        logger.warning(f'No zigzag reversals found with threshold={zigzag_threshold_pct}%, max_lookback={zigzag_max_lookback}')
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
