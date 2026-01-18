"""
Signal generation module for Strategy V3.

Generates trading signals based on model predictions and technical indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from .config import StrategyConfig


class SignalGenerator:
    """
    Generates trading signals from model predictions.
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize SignalGenerator.

        Args:
            config: Strategy configuration
        """
        self.config = config
        self.signal_config = config.signals

    def generate_signals(
        self,
        df: pd.DataFrame,
        support: np.ndarray,
        resistance: np.ndarray,
        breakout_prob: np.ndarray
    ) -> pd.DataFrame:
        """
        Generate trading signals.

        Args:
            df: DataFrame with OHLCV and technical indicators
            support: Predicted support levels
            resistance: Predicted resistance levels
            breakout_prob: Predicted breakout probabilities

        Returns:
            DataFrame with signals
        """
        signals_df = df.copy()
        signals_df['support'] = support
        signals_df['resistance'] = resistance
        signals_df['breakout_prob'] = breakout_prob

        # Calculate signal confidence
        signals_df['buy_confidence'] = self._calculate_buy_confidence(signals_df)
        signals_df['sell_confidence'] = self._calculate_sell_confidence(signals_df)

        # Generate signal types
        signals_df['signal_type'] = self._determine_signal_type(signals_df)

        return signals_df

    def _calculate_buy_confidence(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate buy signal confidence.

        Args:
            df: DataFrame with features

        Returns:
            Confidence scores [0, 1]
        """
        confidence = np.zeros(len(df))

        # RSI component (oversold signal)
        rsi_signal = np.clip((30 - df['rsi']) / 30, 0, 1)
        confidence += self.signal_config.rsi_weight * rsi_signal

        # MACD component (positive momentum)
        macd_signal = np.where(df['macd'] > df['macd_signal'], 1, 0)
        confidence += self.signal_config.macd_weight * macd_signal

        # Bollinger Bands component (near lower band)
        bb_signal = 1 - np.clip(df['bb_position'], 0, 1)
        confidence += self.signal_config.bb_weight * bb_signal

        # Volatility component (expanding)
        atr_signal = np.clip(df['atr_pct'] * 5, 0, 1)
        confidence += self.signal_config.atr_weight * atr_signal

        return np.clip(confidence, 0, 1)

    def _calculate_sell_confidence(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate sell signal confidence.

        Args:
            df: DataFrame with features

        Returns:
            Confidence scores [0, 1]
        """
        confidence = np.zeros(len(df))

        # RSI component (overbought signal)
        rsi_signal = np.clip((df['rsi'] - 70) / 30, 0, 1)
        confidence += self.signal_config.rsi_weight * rsi_signal

        # MACD component (negative momentum)
        macd_signal = np.where(df['macd'] < df['macd_signal'], 1, 0)
        confidence += self.signal_config.macd_weight * macd_signal

        # Bollinger Bands component (near upper band)
        bb_signal = np.clip(df['bb_position'], 0, 1)
        confidence += self.signal_config.bb_weight * bb_signal

        # Volatility component (expanding)
        atr_signal = np.clip(df['atr_pct'] * 5, 0, 1)
        confidence += self.signal_config.atr_weight * atr_signal

        return np.clip(confidence, 0, 1)

    def _determine_signal_type(self, df: pd.DataFrame) -> np.ndarray:
        """
        Determine signal type based on confidence scores.

        Args:
            df: DataFrame with confidence scores

        Returns:
            Array of signal types
        """
        signals = np.full(len(df), 'HOLD', dtype=object)

        # Buy signals
        buy_mask = (
            (df['buy_confidence'] >= self.signal_config.buy_signal_threshold) &
            (df['buy_confidence'] > df['sell_confidence']) &
            (df['close'] > df['support'] * (1 - self.signal_config.price_tolerance_pct / 100))
        )
        signals[buy_mask] = 'BUY'

        # Sell signals
        sell_mask = (
            (df['sell_confidence'] >= self.signal_config.sell_signal_threshold) &
            (df['sell_confidence'] > df['buy_confidence']) &
            (df['close'] < df['resistance'] * (1 + self.signal_config.price_tolerance_pct / 100))
        )
        signals[sell_mask] = 'SELL'

        return signals

    def get_signal_summary(self, signals_df: pd.DataFrame) -> Dict[str, any]:
        """
        Get summary statistics of signals.

        Args:
            signals_df: DataFrame with signals

        Returns:
            Dictionary with summary statistics
        """
        total_candles = len(signals_df)
        buy_signals = (signals_df['signal_type'] == 'BUY').sum()
        sell_signals = (signals_df['signal_type'] == 'SELL').sum()
        total_signals = buy_signals + sell_signals

        buy_df = signals_df[signals_df['signal_type'] == 'BUY']
        sell_df = signals_df[signals_df['signal_type'] == 'SELL']

        avg_buy_confidence = buy_df['buy_confidence'].mean() if len(buy_df) > 0 else 0
        avg_sell_confidence = sell_df['sell_confidence'].mean() if len(sell_df) > 0 else 0

        return {
            'total_candles': total_candles,
            'buy_signals': int(buy_signals),
            'sell_signals': int(sell_signals),
            'total_signals': int(total_signals),
            'signal_density': total_signals / total_candles if total_candles > 0 else 0,
            'avg_buy_confidence': float(avg_buy_confidence),
            'avg_sell_confidence': float(avg_sell_confidence),
        }
