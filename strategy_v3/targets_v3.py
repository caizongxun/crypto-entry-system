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


def calculate_reversal_strength(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                               volume: np.ndarray, atr: np.ndarray,
                               swing_idx: int, is_high: bool,
                               lookback: int = 20) -> float:
    """
    Calculate reversal signal strength (0-100).
    
    Improved scoring with confirmation logic:
    - Base: 20 (confirmed swing point)
    - Volume confirmation: +15
    - ATR expansion: +15
    - Trend confirmation: +15
    - Reversal profitability: +25 (most important)
    - Support/Resistance: +10
    
    Total max: 100
    """
    n = len(close)
    if swing_idx < lookback or swing_idx + 10 >= n:
        return 0.0
    
    score = 20.0
    
    vol_mean = volume[max(0, swing_idx - lookback):swing_idx].mean()
    atr_mean = atr[max(0, swing_idx - lookback):swing_idx].mean()
    
    vol_ratio = volume[swing_idx] / (vol_mean + 1e-10)
    atr_ratio = atr[swing_idx] / (atr_mean + 1e-10)
    
    if vol_ratio >= 1.3:
        score += 15.0
    elif vol_ratio >= 1.1:
        score += 8.0
    elif vol_ratio >= 1.0:
        score += 3.0
    
    if atr_ratio >= 1.5:
        score += 15.0
    elif atr_ratio >= 1.2:
        score += 8.0
    elif atr_ratio >= 1.0:
        score += 3.0
    
    if is_high:
        trend_confirm = sum(1 for i in range(swing_idx + 1, min(swing_idx + 4, n)) if close[i] < close[swing_idx])
    else:
        trend_confirm = sum(1 for i in range(swing_idx + 1, min(swing_idx + 4, n)) if close[i] > close[swing_idx])
    
    if trend_confirm >= 3:
        score += 15.0
    elif trend_confirm >= 2:
        score += 8.0
    elif trend_confirm >= 1:
        score += 3.0
    
    if is_high:
        if swing_idx + 10 < n:
            future_low = low[swing_idx + 1:min(swing_idx + 11, n)].min()
            reversal_range = (close[swing_idx] - future_low) / close[swing_idx]
        else:
            reversal_range = 0
    else:
        if swing_idx + 10 < n:
            future_high = high[swing_idx + 1:min(swing_idx + 11, n)].max()
            reversal_range = (future_high - close[swing_idx]) / close[swing_idx]
        else:
            reversal_range = 0
    
    if reversal_range >= 0.015:
        score += 25.0
    elif reversal_range >= 0.010:
        score += 15.0
    elif reversal_range >= 0.005:
        score += 8.0
    elif reversal_range >= 0.002:
        score += 3.0
    
    high_20 = high[max(0, swing_idx - lookback):swing_idx].max()
    low_20 = low[max(0, swing_idx - lookback):swing_idx].min()
    
    if is_high:
        if high[swing_idx] >= high_20 * 1.01:
            score += 10.0
        elif high[swing_idx] >= high_20 * 0.995:
            score += 5.0
    else:
        if low[swing_idx] <= low_20 * 0.99:
            score += 10.0
        elif low[swing_idx] <= low_20 * 1.005:
            score += 5.0
    
    return min(score, 100.0)


def create_reversal_target_v3(df: pd.DataFrame, lookback: int = 20,
                              left_bars: int = 5, right_bars: int = 5,
                              strength_threshold: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create reversal targets using improved strength scoring.
    
    Returns:
    - targets: Binary array (1 if strong reversal ahead, 0 otherwise)
    - strengths: Strength scores for all points (for analysis)
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    volume = df['volume'].values
    atr = df['atr_14'].values
    
    n = len(df)
    targets = np.zeros(n, dtype=int)
    strengths = np.zeros(n, dtype=float)
    
    swing_highs, swing_lows = detect_swing_points(high, low, left_bars, right_bars)
    high_indices = np.where(swing_highs == 1)[0]
    low_indices = np.where(swing_lows == 1)[0]
    
    logger.info(f'Found {len(high_indices)} swing highs and {len(low_indices)} swing lows')
    
    reversal_indices = []
    reversal_strengths = []
    
    for idx in high_indices:
        strength = calculate_reversal_strength(high, low, close, volume, atr, idx, True, lookback)
        strengths[idx] = strength
        if strength >= strength_threshold:
            reversal_indices.append(idx)
            reversal_strengths.append(strength)
    
    for idx in low_indices:
        strength = calculate_reversal_strength(high, low, close, volume, atr, idx, False, lookback)
        strengths[idx] = strength
        if strength >= strength_threshold:
            reversal_indices.append(idx)
            reversal_strengths.append(strength)
    
    reversal_indices = np.array(reversal_indices)
    reversal_strengths = np.array(reversal_strengths)
    
    if len(reversal_indices) > 0:
        logger.info(f'Found {len(reversal_indices)} strong reversal signals')
        logger.info(f'Strength distribution: min={reversal_strengths.min():.1f}, max={reversal_strengths.max():.1f}, mean={reversal_strengths.mean():.1f}')
    else:
        logger.warning(f'No reversals found with strength >= {strength_threshold}')
        return targets, strengths
    
    in_hold_phase = False
    hold_end_idx = -1
    
    for i in range(lookback, n):
        if in_hold_phase:
            if i <= hold_end_idx:
                targets[i] = 0
            else:
                in_hold_phase = False
            continue
        
        lookahead_start = i + 1
        lookahead_end = min(i + 20, n)
        
        reversals_ahead = reversal_indices[
            (reversal_indices >= lookahead_start) & (reversal_indices < lookahead_end)
        ]
        
        if len(reversals_ahead) > 0:
            reversal_idx = reversals_ahead[0]
            targets[i] = 1
            
            in_hold_phase = True
            hold_end_idx = min(reversal_idx + 5, n - 1)
    
    return targets, strengths
