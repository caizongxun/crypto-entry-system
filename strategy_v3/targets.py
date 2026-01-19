import numpy as np
import pandas as pd
from loguru import logger


def create_reversal_target(df: pd.DataFrame, lookback: int = 20, atr_mult: float = 1.5,
                          profit_target_ratio: float = 1.5, forward_window: int = 100) -> np.ndarray:
    """
    Create binary reversal target with proper HOLD labeling.

    Logic:
    - For candle t, use features from t-lookback to t-1 for prediction
    - Entry at close of candle t
    - SL at entry_price - (atr * atr_mult)
    - TP at entry_price + (atr * atr_mult * profit_target_ratio)
    - If TP hit before SL within forward_window candles: label=1
    - Candles t+1 to t+N (while in position): label=0 (HOLD)
    - Otherwise: label=0 (no reversal opportunity)

    Args:
        df: DataFrame with OHLCV and atr_14
        lookback: Number of past candles to use as features
        atr_mult: ATR multiplier for stop loss
        profit_target_ratio: TP to SL ratio
        forward_window: Max candles to look ahead

    Returns:
        Array of targets (0 or 1) for each candle
    """
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    atr = df['atr_14'].values

    targets = np.zeros(len(df), dtype=int)
    in_position = False
    position_end_idx = -1

    for i in range(lookback, len(df) - forward_window):
        # If in position, mark as HOLD
        if in_position:
            if i <= position_end_idx:
                targets[i] = 0
            else:
                in_position = False
                position_end_idx = -1

        if not in_position:
            entry_price = close[i]
            sl_price = entry_price - (atr[i] * atr_mult)
            tp_price = entry_price + (atr[i] * atr_mult * profit_target_ratio)

            future_high = high[i + 1:i + 1 + forward_window].max()
            future_low = low[i + 1:i + 1 + forward_window].min()

            # Check if TP or SL would be hit
            if future_high >= tp_price or future_low <= sl_price:
                # Find which one hits first
                tp_hit_idx = None
                sl_hit_idx = None

                for j in range(i + 1, min(i + 1 + forward_window, len(df))):
                    if high[j] >= tp_price and tp_hit_idx is None:
                        tp_hit_idx = j
                    if low[j] <= sl_price and sl_hit_idx is None:
                        sl_hit_idx = j

                    if tp_hit_idx is not None and sl_hit_idx is not None:
                        break

                # TP hit first = successful reversal
                if tp_hit_idx is not None and (sl_hit_idx is None or tp_hit_idx < sl_hit_idx):
                    targets[i] = 1
                    in_position = True
                    position_end_idx = tp_hit_idx

    return targets
