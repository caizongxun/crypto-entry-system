"""
Feature engineering module for Strategy V3.

Computes technical indicators and features for machine learning models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from .config import StrategyConfig, TechnicalIndicatorConfig


class FeatureEngineer:
    """
    Computes technical indicators and features.
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize FeatureEngineer.

        Args:
            config: Strategy configuration
        """
        self.config = config
        self.indicators_config = config.indicators

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all features from OHLCV data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with OHLCV and all computed features
        """
        df = df.copy()

        # Basic OHLCV features
        df['hl_ratio'] = df['high'] / df['low']
        df['oc_ratio'] = df['close'] / df['open']
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['price_range'] = df['high'] - df['low']
        df['body_size'] = np.abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']

        # Volume features
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_change'] = df['volume'].pct_change()

        # Moving Averages
        df['sma_short'] = df['close'].rolling(window=self.indicators_config.sma_short).mean()
        df['sma_medium'] = df['close'].rolling(window=self.indicators_config.sma_medium).mean()
        df['sma_long'] = df['close'].rolling(window=self.indicators_config.sma_long).mean()

        df['ema_short'] = df['close'].ewm(span=self.indicators_config.ema_short, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=self.indicators_config.ema_long, adjust=False).mean()

        # Moving average positions
        df['price_vs_sma_short'] = (df['close'] - df['sma_short']) / df['sma_short']
        df['price_vs_sma_long'] = (df['close'] - df['sma_long']) / df['sma_long']
        df['sma_trend'] = np.where(df['sma_short'] > df['sma_long'], 1, -1)

        # RSI (Relative Strength Index)
        df['rsi'] = self._calculate_rsi(df['close'], self.indicators_config.rsi_period)
        df['rsi_normalized'] = (df['rsi'] - 50) / 50

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=self.indicators_config.bb_period).mean()
        df['bb_std'] = df['close'].rolling(window=self.indicators_config.bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (self.indicators_config.bb_std_dev * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (self.indicators_config.bb_std_dev * df['bb_std'])
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # MACD
        ema_fast = df['close'].ewm(span=self.indicators_config.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.indicators_config.macd_slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.indicators_config.macd_signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['macd_trend'] = np.where(df['macd'] > df['macd_signal'], 1, -1)

        # ATR (Average True Range)
        df['atr'] = self._calculate_atr(df, self.indicators_config.atr_period)
        df['atr_pct'] = df['atr'] / df['close']

        # Stochastic
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(
            df['high'],
            df['low'],
            df['close'],
            self.indicators_config.stoch_period,
            self.indicators_config.stoch_smooth_k,
            self.indicators_config.stoch_smooth_d
        )

        # Support and Resistance
        df['support'] = df['low'].rolling(window=20).min()
        df['resistance'] = df['high'].rolling(window=20).max()
        df['midpoint'] = (df['support'] + df['resistance']) / 2

        # Momentum and Rate of Change
        df['roc_5'] = df['close'].pct_change(periods=5)
        df['roc_10'] = df['close'].pct_change(periods=10)
        df['momentum'] = df['close'] - df['close'].shift(10)

        # Fill NaN values
        df = df.fillna(method='bfill')
        df = df.fillna(method='ffill')
        df = df.fillna(0)

        return df

    @staticmethod
    def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.

        Args:
            series: Price series
            period: RSI period

        Returns:
            RSI values
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range.

        Args:
            df: DataFrame with high, low, close
            period: ATR period

        Returns:
            ATR values
        """
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        return atr

    @staticmethod
    def _calculate_stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3
    ) -> tuple:
        """
        Calculate Stochastic Oscillator.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Stochastic period
            smooth_k: K smoothing period
            smooth_d: D smoothing period

        Returns:
            Tuple of (K%, D%)
        """
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()

        k_raw = 100 * (close - lowest_low) / (highest_high - lowest_low)
        k = k_raw.rolling(window=smooth_k).mean()
        d = k.rolling(window=smooth_d).mean()

        return k, d
