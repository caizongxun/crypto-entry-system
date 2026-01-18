"""
Feature engineering module for Strategy V3.

Comprehensive technical indicators and feature engineering.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from loguru import logger


class FeatureEngineer:
    """
    Computes technical indicators and engineered features.
    """

    def __init__(self, config):
        """
        Initialize FeatureEngineer.

        Args:
            config: StrategyConfig object
        """
        self.cfg = config
        self.indicators = config.indicators
        self.verbose = config.verbose

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features from OHLCV data.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with engineered features
        """
        features = df.copy()

        # Price action
        features = self._add_sma(features)
        features = self._add_ema(features)
        features = self._add_momentum(features)

        # Oscillators
        features = self._add_rsi(features)
        features = self._add_macd(features)
        features = self._add_stochastic(features)

        # Volatility
        features = self._add_atr(features)
        features = self._add_bollinger_bands(features)

        # Volume
        features = self._add_volume_indicators(features)

        # Price relationships
        features = self._add_price_relationships(features)

        # Remove NaN rows
        features = features.dropna()

        if self.verbose:
            logger.info(f'Engineered {len(features.columns) - 5} features')

        return features

    def _add_sma(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Simple Moving Averages.
        """
        df['sma_short'] = df['close'].rolling(window=self.indicators.sma_short).mean()
        df['sma_medium'] = df['close'].rolling(window=self.indicators.sma_medium).mean()
        df['sma_long'] = df['close'].rolling(window=self.indicators.sma_long).mean()

        df['sma_short_slope'] = df['sma_short'].diff()
        df['sma_long_slope'] = df['sma_long'].diff()

        return df

    def _add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Exponential Moving Averages.
        """
        df['ema_short'] = df['close'].ewm(span=self.indicators.ema_short, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=self.indicators.ema_long, adjust=False).mean()

        df['price_to_ema_short'] = (df['close'] - df['ema_short']) / df['ema_short']
        df['price_to_ema_long'] = (df['close'] - df['ema_long']) / df['ema_long']

        return df

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Relative Strength Index.
        """
        delta = df['close'].diff()
        gains = (delta.where(delta > 0, 0)).rolling(window=self.indicators.rsi_period).mean()
        losses = (-delta.where(delta < 0, 0)).rolling(window=self.indicators.rsi_period).mean()

        rs = gains / (losses + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))

        df['rsi_normalized'] = df['rsi'] / 100.0
        df['rsi_divergence'] = df['rsi'].diff()

        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add MACD indicator.
        """
        ema_fast = df['close'].ewm(span=self.indicators.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.indicators.macd_slow, adjust=False).mean()

        df['macd_line'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd_line'].ewm(span=self.indicators.macd_signal, adjust=False).mean()
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']

        df['macd_direction'] = (df['macd_histogram'] > 0).astype(int)

        return df

    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Bollinger Bands.
        """
        sma = df['close'].rolling(window=self.indicators.bb_period).mean()
        std = df['close'].rolling(window=self.indicators.bb_period).std()

        df['bb_upper'] = sma + (std * self.indicators.bb_std_dev)
        df['bb_lower'] = sma - (std * self.indicators.bb_std_dev)
        df['bb_middle'] = sma
        df['bb_width'] = df['bb_upper'] - df['bb_lower']

        # Price position within bands
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        df['bb_position'] = df['bb_position'].clip(0, 1)

        return df

    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Average True Range.
        """
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=self.indicators.atr_period).mean()

        # ATR ratio to close
        df['atr_ratio'] = df['atr'] / df['close']

        return df

    def _add_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Stochastic Oscillator.
        """
        low_min = df['low'].rolling(window=self.indicators.stoch_period).min()
        high_max = df['high'].rolling(window=self.indicators.stoch_period).max()

        stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-8)
        df['stoch_k'] = stoch_k.rolling(window=self.indicators.stoch_smooth_k).mean()
        df['stoch_d'] = df['stoch_k'].rolling(window=self.indicators.stoch_smooth_d).mean()

        df['stoch_k_normalized'] = df['stoch_k'] / 100.0

        return df

    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum indicators.
        """
        df['momentum_1'] = df['close'] - df['close'].shift(1)
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_10'] = df['close'] - df['close'].shift(10)

        df['returns_1'] = df['close'].pct_change(1)
        df['returns_5'] = df['close'].pct_change(5)

        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based indicators.
        """
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-8)

        # On-Balance Volume (simplified)
        obv = (pd.Series(np.where(df['close'] > df['close'].shift(), df['volume'], 0)) -
               pd.Series(np.where(df['close'] < df['close'].shift(), df['volume'], 0))).cumsum()
        df['obv'] = obv
        df['obv_sma'] = obv.rolling(window=20).mean()

        return df

    def _add_price_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived price relationships.
        """
        df['high_low_ratio'] = df['high'] / (df['low'] + 1e-8)
        df['close_to_high_ratio'] = df['close'] / (df['high'] + 1e-8)
        df['close_to_low_ratio'] = df['close'] / (df['low'] + 1e-8)

        df['price_range'] = df['high'] - df['low']
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']

        return df

    @staticmethod
    def get_feature_names() -> List[str]:
        """
        Get list of all engineered feature names.
        """
        return [
            # SMA
            'sma_short', 'sma_medium', 'sma_long', 'sma_short_slope', 'sma_long_slope',
            # EMA
            'ema_short', 'ema_long', 'price_to_ema_short', 'price_to_ema_long',
            # RSI
            'rsi', 'rsi_normalized', 'rsi_divergence',
            # MACD
            'macd_line', 'macd_signal', 'macd_histogram', 'macd_direction',
            # Bollinger Bands
            'bb_upper', 'bb_lower', 'bb_middle', 'bb_width', 'bb_position',
            # ATR
            'atr', 'atr_ratio',
            # Stochastic
            'stoch_k', 'stoch_d', 'stoch_k_normalized',
            # Momentum
            'momentum_1', 'momentum_5', 'momentum_10', 'returns_1', 'returns_5',
            # Volume
            'volume_sma', 'volume_ratio', 'obv', 'obv_sma',
            # Price relationships
            'high_low_ratio', 'close_to_high_ratio', 'close_to_low_ratio',
            'price_range', 'body_size', 'upper_shadow', 'lower_shadow',
        ]
