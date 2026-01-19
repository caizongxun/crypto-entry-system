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


def identify_reversals_with_confirmation(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                                        volume: np.ndarray, atr: np.ndarray,
                                        left_bars: int = 5, right_bars: int = 5,
                                        volume_multiplier: float = 1.2,
                                        atr_multiplier: float = 1.3,
                                        trend_bars: int = 2) -> np.ndarray:
    """
    Identify reversal points using Swing High/Low with multi-confirmation filters:
    1. Volume spike confirmation
    2. ATR expansion confirmation
    3. Trend continuation confirmation
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        volume: Array of volumes
        atr: Array of ATR values
        left_bars: Lookback period for swing detection
        right_bars: Lookahead period for swing detection
        volume_multiplier: Volume must be >= mean * multiplier
        atr_multiplier: ATR must be >= mean * multiplier
        trend_bars: Minimum bars to confirm trend reversal
        
    Returns:
        Array where 1 = confirmed reversal point, 0 = no reversal
    """
    n = len(high)
    reversals = np.zeros(n, dtype=int)
    
    swing_highs, swing_lows = detect_swing_points(high, low, left_bars, right_bars)
    high_indices = np.where(swing_highs == 1)[0]
    low_indices = np.where(swing_lows == 1)[0]
    
    lookback_vol = 20
    lookback_atr = 20
    
    for i in range(len(high_indices)):
        curr_idx = high_indices[i]
        
        if curr_idx + trend_bars >= n:
            continue
        
        vol_mean = volume[max(0, curr_idx - lookback_vol):curr_idx].mean()
        atr_mean = atr[max(0, curr_idx - lookback_atr):curr_idx].mean()
        
        vol_check = volume[curr_idx] >= vol_mean * volume_multiplier
        atr_check = atr[curr_idx] >= atr_mean * atr_multiplier
        
        trend_check = False
        if curr_idx + trend_bars < n:
            lower_bars = np.sum(close[curr_idx+1:curr_idx+trend_bars+1] < close[curr_idx])
            if lower_bars >= trend_bars:
                trend_check = True
        
        if vol_check and atr_check and trend_check:
            reversals[curr_idx] = 1
    
    for i in range(len(low_indices)):
        curr_idx = low_indices[i]
        
        if curr_idx + trend_bars >= n:
            continue
        
        vol_mean = volume[max(0, curr_idx - lookback_vol):curr_idx].mean()
        atr_mean = atr[max(0, curr_idx - lookback_atr):curr_idx].mean()
        
        vol_check = volume[curr_idx] >= vol_mean * volume_multiplier
        atr_check = atr[curr_idx] >= atr_mean * atr_multiplier
        
        trend_check = False
        if curr_idx + trend_bars < n:
            higher_bars = np.sum(close[curr_idx+1:curr_idx+trend_bars+1] > close[curr_idx])
            if higher_bars >= trend_bars:
                trend_check = True
        
        if vol_check and atr_check and trend_check:
            reversals[curr_idx] = 1
    
    return reversals


def create_reversal_target_v2(df: pd.DataFrame, lookback: int = 20,
                              left_bars: int = 5, right_bars: int = 5,
                              volume_multiplier: float = 1.2,
                              atr_multiplier: float = 1.3,
                              trend_bars: int = 2) -> np.ndarray:
    """
    Create binary reversal target using swing high/low with multi-confirmation filters.
    
    Multi-confirmation ensures:
    - Volume spike at reversal point (volatility increase)
    - ATR expansion (price momentum increase)
    - Trend continuation (minimum trend bars after reversal)
    
    Args:
        df: DataFrame with OHLCV data and technical indicators (must have 'atr_14')
        lookback: Number of past candles used as features
        left_bars: Lookback period for swing detection
        right_bars: Lookahead period for swing detection
        volume_multiplier: Volume threshold multiplier
        atr_multiplier: ATR threshold multiplier
        trend_bars: Minimum bars to confirm trend continuation
        
    Returns:
        Array of targets (0 or 1) for each candle
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    volume = df['volume'].values
    atr = df['atr_14'].values
    
    targets = np.zeros(len(df), dtype=int)
    
    reversals = identify_reversals_with_confirmation(
        high, low, close, volume, atr,
        left_bars=left_bars,
        right_bars=right_bars,
        volume_multiplier=volume_multiplier,
        atr_multiplier=atr_multiplier,
        trend_bars=trend_bars
    )
    
    reversal_indices = np.where(reversals == 1)[0]
    
    logger.info(f'Found {len(reversal_indices)} multi-confirmed reversal points')
    logger.info(f'Parameters: left_bars={left_bars}, right_bars={right_bars}, '
                f'vol_mult={volume_multiplier}, atr_mult={atr_multiplier}, trend_bars={trend_bars}')
    
    if len(reversal_indices) == 0:
        logger.warning('No reversals found with current parameters')
        return targets
    
    in_hold_phase = False
    hold_end_idx = -1
    
    for i in range(lookback, len(df)):
        if in_hold_phase:
            if i <= hold_end_idx:
                targets[i] = 0
            else:
                in_hold_phase = False
            continue
        
        lookahead_start = i + 1
        lookahead_end = min(i + 20, len(df))
        
        reversals_ahead = reversal_indices[
            (reversal_indices >= lookahead_start) & (reversal_indices < lookahead_end)
        ]
        
        if len(reversals_ahead) > 0:
            reversal_idx = reversals_ahead[0]
            targets[i] = 1
            
            in_hold_phase = True
            hold_end_idx = min(reversal_idx + 5, len(df) - 1)
    
    return targets
