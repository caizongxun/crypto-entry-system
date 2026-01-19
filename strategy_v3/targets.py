import numpy as np
import pandas as pd
from loguru import logger
from typing import List, Tuple


def identify_zigzag_reversals(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                             atr: np.ndarray, atr_multiplier: float = 1.5,
                             max_lookback: int = 50) -> np.ndarray:
    """
    Identify reversal points using zigzag pattern with ATR-based volatility.
    
    Key improvement: Use ATR multiplier instead of fixed percentage.
    This automatically adjusts for market volatility:
    - High volatility markets: Larger ATR -> larger required swings
    - Low volatility markets: Smaller ATR -> smaller required swings
    - Works equally for BTC ($100+ moves = 1% ATR move) and small caps
    
    A zigzag reversal is confirmed when:
    1. Price makes a significant swing (atr_multiplier * current_atr)
    2. Direction changes (from up to down or down to up)
    3. Swing does not exceed max_lookback candles
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        atr: Array of ATR values
        atr_multiplier: Number of ATRs for swing threshold (e.g., 1.5 ATR)
        max_lookback: Maximum candles allowed for one swing
        
    Returns:
        Array where 1 = reversal point, 0 = not reversal
    """
    reversal_points = np.zeros(len(high), dtype=int)
    
    if len(high) < 5 or len(atr) != len(high):
        return reversal_points
    
    last_reversal_idx = 0
    last_reversal_type = None
    
    i = 0
    while i < len(high) - 1:
        current_atr = atr[i] if atr[i] > 0 else np.nanmean(atr[:max(i, 1)])
        current_atr = max(current_atr, 0.0001)  # Prevent division by zero
        threshold = atr_multiplier * current_atr
        
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
            
            # Check if significant down move found (using ATR)
            swing_amount = high[last_reversal_idx] - local_low
            if swing_amount >= threshold:
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
            
            # Check if significant up move found (using ATR)
            swing_amount = local_high - low[last_reversal_idx]
            if swing_amount >= threshold:
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
            up_swing = local_high - low[i]
            down_swing = high[i] - local_low
            
            if up_swing >= threshold and local_high_idx <= local_low_idx:
                # Up move is primary
                reversal_points[local_high_idx] = 1
                last_reversal_idx = local_high_idx
                last_reversal_type = 'high'
                i = local_high_idx + 1
            elif down_swing >= threshold and local_low_idx <= local_high_idx:
                # Down move is primary
                reversal_points[local_low_idx] = 1
                last_reversal_idx = local_low_idx
                last_reversal_type = 'low'
                i = local_low_idx + 1
            else:
                i += 1
    
    return reversal_points


def create_reversal_target(df: pd.DataFrame, lookback: int = 20, atr_mult: float = 1.5,
                          profit_target_ratio: float = 1.5, atr_multiplier: float = 1.5,
                          max_lookback: int = 50) -> np.ndarray:
    """
    Create binary reversal target using ATR-based Zigzag reversal identification.
    
    Logic:
    - For each candle, check if a zigzag reversal point is found ahead (within lookahead distance)
    - If reversal found within next N candles, mark as 1 (reversal opportunity)
    - Otherwise mark as 0
    - Also mark subsequent candles in the reversal move as 0 (HOLD phase)
    
    Args:
        df: DataFrame with OHLCV and atr_14
        lookback: Number of past candles used as features
        atr_mult: ATR multiplier for context (deprecated, use atr_multiplier)
        profit_target_ratio: Profit target ratio for context
        atr_multiplier: Number of ATRs for swing threshold (e.g., 1.5 ATR)
        max_lookback: Maximum candles allowed for one swing
        
    Returns:
        Array of targets (0 or 1) for each candle
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    atr = df['atr_14'].values
    
    targets = np.zeros(len(df), dtype=int)
    
    # Identify all zigzag reversal points with ATR-based threshold
    zigzag_reversals = identify_zigzag_reversals(
        high, low, close, atr,
        atr_multiplier=atr_multiplier,
        max_lookback=max_lookback
    )
    
    reversal_indices = np.where(zigzag_reversals == 1)[0]
    
    logger.info(f'Found {len(reversal_indices)} ATR-based zigzag reversal points')
    logger.info(f'Zigzag parameters: atr_multiplier={atr_multiplier}x, max_lookback={max_lookback} candles')
    
    if len(reversal_indices) == 0:
        logger.warning(f'No zigzag reversals found with atr_multiplier={atr_multiplier}x, max_lookback={max_lookback}')
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
