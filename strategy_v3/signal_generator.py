"""
Signal generator module for Strategy V3.

Combines model predictions with technical indicators to generate trading signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from loguru import logger


class SignalGenerator:
    """
    Generates trading signals based on model predictions and technical indicators.
    """

    def __init__(self, config):
        """
        Initialize SignalGenerator.

        Args:
            config: StrategyConfig object
        """
        self.cfg = config
        self.signal_cfg = config.signals
        self.indicator_cfg = config.indicators
        self.verbose = config.verbose

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
            df: OHLCV DataFrame with technical indicators
            support: Predicted support levels
            resistance: Predicted resistance levels
            breakout_prob: Predicted breakout probability (0-1)

        Returns:
            DataFrame with signals
        """
        signals_df = df.copy()
        signals_df['support'] = support
        signals_df['resistance'] = resistance
        signals_df['breakout_prob'] = breakout_prob

        # Calculate signal components
        signals_df['price_to_support_dist'] = (signals_df['close'] - signals_df['support']) / signals_df['support']
        signals_df['price_to_resistance_dist'] = (signals_df['resistance'] - signals_df['close']) / signals_df['resistance']

        # Generate raw signals
        signals_df['buy_signal_raw'] = self._generate_buy_signal(
            signals_df
        )
        signals_df['sell_signal_raw'] = self._generate_sell_signal(
            signals_df
        )

        # Calculate confidence scores
        signals_df['buy_confidence'] = signals_df['buy_signal_raw']
        signals_df['sell_confidence'] = signals_df['sell_signal_raw']

        # Filter signals by minimum confidence
        signals_df['buy_signal'] = (
            (signals_df['buy_confidence'] >= self.signal_cfg.buy_signal_threshold) &
            (signals_df['buy_confidence'] >= self.signal_cfg.min_confidence)
        ).astype(int)
        signals_df['sell_signal'] = (
            (signals_df['sell_confidence'] >= self.signal_cfg.sell_signal_threshold) &
            (signals_df['sell_confidence'] >= self.signal_cfg.min_confidence)
        ).astype(int)

        # Signal type
        signals_df['signal_type'] = 'HOLD'
        signals_df.loc[signals_df['buy_signal'] == 1, 'signal_type'] = 'BUY'
        signals_df.loc[signals_df['sell_signal'] == 1, 'signal_type'] = 'SELL'

        return signals_df

    def _generate_buy_signal(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate buy signal confidence.

        Args:
            df: DataFrame with OHLCV and indicators

        Returns:
            Confidence array (0-1)
        """
        confidence = np.zeros(len(df))

        # Component 1: Price near support (weight 0.25)
        price_to_support_dist = (df['close'] - df['support']) / df['support']
        support_proximity = 1.0 - np.clip(np.abs(price_to_support_dist) / 0.01, 0, 1)
        confidence += support_proximity * self.signal_cfg.rsi_weight

        # Component 2: RSI oversold (weight 0.25)
        rsi_signal = np.where(
            df['rsi'] < self.indicator_cfg.rsi_oversold,
            (self.indicator_cfg.rsi_oversold - df['rsi']) / self.indicator_cfg.rsi_oversold,
            0.0
        )
        confidence += rsi_signal * self.signal_cfg.macd_weight

        # Component 3: MACD positive (weight 0.25)
        macd_signal = np.where(
            df['macd_histogram'] > 0,
            np.clip(df['macd_histogram'] / (np.abs(df['macd_histogram']).max() + 1e-8), 0, 1),
            0.0
        )
        confidence += macd_signal * self.signal_cfg.bb_weight

        # Component 4: High breakout probability (weight 0.25)
        confidence += df['breakout_prob'].values * self.signal_cfg.atr_weight

        # Normalize to 0-1
        confidence = np.clip(confidence / np.sum([
            self.signal_cfg.rsi_weight,
            self.signal_cfg.macd_weight,
            self.signal_cfg.bb_weight,
            self.signal_cfg.atr_weight
        ]), 0, 1)

        return confidence

    def _generate_sell_signal(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate sell signal confidence.

        Args:
            df: DataFrame with OHLCV and indicators

        Returns:
            Confidence array (0-1)
        """
        confidence = np.zeros(len(df))

        # Component 1: Price near resistance (weight 0.25)
        price_to_resistance_dist = (df['resistance'] - df['close']) / df['resistance']
        resistance_proximity = 1.0 - np.clip(np.abs(price_to_resistance_dist) / 0.01, 0, 1)
        confidence += resistance_proximity * self.signal_cfg.rsi_weight

        # Component 2: RSI overbought (weight 0.25)
        rsi_signal = np.where(
            df['rsi'] > self.indicator_cfg.rsi_overbought,
            (df['rsi'] - self.indicator_cfg.rsi_overbought) / (100 - self.indicator_cfg.rsi_overbought),
            0.0
        )
        confidence += rsi_signal * self.signal_cfg.macd_weight

        # Component 3: MACD negative (weight 0.25)
        macd_signal = np.where(
            df['macd_histogram'] < 0,
            np.clip(np.abs(df['macd_histogram']) / (np.abs(df['macd_histogram']).max() + 1e-8), 0, 1),
            0.0
        )
        confidence += macd_signal * self.signal_cfg.bb_weight

        # Component 4: Low breakout probability (weight 0.25)
        confidence += (1.0 - df['breakout_prob'].values) * self.signal_cfg.atr_weight

        # Normalize to 0-1
        confidence = np.clip(confidence / np.sum([
            self.signal_cfg.rsi_weight,
            self.signal_cfg.macd_weight,
            self.signal_cfg.bb_weight,
            self.signal_cfg.atr_weight
        ]), 0, 1)

        return confidence

    def get_signal_summary(self, signals_df: pd.DataFrame) -> Dict[str, any]:
        """
        Generate summary statistics from signals.

        Args:
            signals_df: DataFrame with generated signals

        Returns:
            Dictionary with summary statistics
        """
        total_signals = len(signals_df)
        buy_signals = (signals_df['signal_type'] == 'BUY').sum()
        sell_signals = (signals_df['signal_type'] == 'SELL').sum()
        hold_periods = (signals_df['signal_type'] == 'HOLD').sum()

        avg_buy_confidence = signals_df[signals_df['signal_type'] == 'BUY']['buy_confidence'].mean() if buy_signals > 0 else 0
        avg_sell_confidence = signals_df[signals_df['signal_type'] == 'SELL']['sell_confidence'].mean() if sell_signals > 0 else 0

        return {
            'total_candles': total_signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'hold_periods': hold_periods,
            'signal_density': (buy_signals + sell_signals) / total_signals if total_signals > 0 else 0,
            'avg_buy_confidence': avg_buy_confidence,
            'avg_sell_confidence': avg_sell_confidence,
        }
