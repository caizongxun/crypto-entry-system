"""
Feature engineering module for Strategy V3.

Generates 40+ features optimized for 15m crypto trading prediction.
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, List


class FeatureEngineer:
    """
    Generates technical features for machine learning models.
    Optimized for 15m trading on crypto markets.
    """

    def __init__(self, config):
        """
        Initialize FeatureEngineer.

        Args:
            config: StrategyConfig instance
        """
        self.config = config
        self.verbose = config.verbose

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features from OHLCV data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with original OHLCV + engineered features
        """
        df = df.copy()
        
        logger.info('Generating features...')
        
        # Price-based features
        df = self._add_price_features(df)
        # Momentum features
        df = self._add_momentum_features(df)
        # Volume features
        df = self._add_volume_features(df)
        # Trend features
        df = self._add_trend_features(df)
        # Volatility features
        df = self._add_volatility_features(df)
        # Pattern features
        df = self._add_pattern_features(df)
        
        # Fill NaN values
        df = df.ffill().bfill()
        
        if self.verbose:
            feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            logger.info(f'Generated {len(feature_cols)} features')
            logger.info(f'Feature columns: {feature_cols[:10]}... (showing first 10)')
        
        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-based features.
        """
        df['hl_ratio'] = df['high'] / (df['low'] + 1e-8)
        df['cl_ratio'] = df['close'] / (df['low'] + 1e-8)
        df['co_ratio'] = df['close'] / (df['open'] + 1e-8)
        df['oc_ratio'] = df['open'] / (df['close'] + 1e-8)
        
        df['range'] = df['high'] - df['low']
        df['body'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        df['returns'] = df['close'].pct_change() * 100
        df['returns_abs'] = abs(df['returns'])
        
        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum-based features.
        """
        # RSI (14 period)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Momentum indicators
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df['momentum_20'] = df['close'] - df['close'].shift(20)
        
        # Rate of change
        df['roc_5'] = (df['close'] / df['close'].shift(5) - 1) * 100
        df['roc_10'] = (df['close'] / df['close'].shift(10) - 1) * 100
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based features.
        """
        # Volume ratios
        df['vol_ratio_5'] = df['volume'] / (df['volume'].rolling(5).mean() + 1e-8)
        df['vol_ratio_20'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-8)
        df['vol_sma_5'] = df['volume'].rolling(5).mean()
        df['vol_sma_20'] = df['volume'].rolling(20).mean()
        
        # Volume changes
        df['vol_change'] = df['volume'].pct_change() * 100
        df['vol_mom'] = df['volume'] - df['volume'].shift(1)
        
        # OBV (On-Balance Volume)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()
        
        return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend-based features.
        """
        # Moving averages
        for period in [5, 9, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # Price position in moving average
        df['close_sma5_ratio'] = df['close'] / (df['sma_5'] + 1e-8)
        df['close_sma20_ratio'] = df['close'] / (df['sma_20'] + 1e-8)
        
        # EMA crossovers
        df['ema9_ema21'] = df['ema_9'] - df['ema_21']
        df['sma5_sma20'] = df['sma_5'] - df['sma_20']
        
        # Trend strength (ADX-like)
        df['trend_up'] = (df['close'] > df['close'].shift(1)).astype(int)
        df['trend_down'] = (df['close'] < df['close'].shift(1)).astype(int)
        df['trend_strength'] = abs(df['trend_up'] - df['trend_down']).rolling(14).sum()
        
        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility-based features.
        """
        # Standard deviation
        df['std_5'] = df['close'].rolling(5).std()
        df['std_20'] = df['close'].rolling(20).std()
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        df['atr_ratio'] = df['atr_14'] / (df['close'] + 1e-8)
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma_20 + 2 * std_20
        df['bb_lower'] = sma_20 - 2 * std_20
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_width'] + 1e-8)
        
        return df

    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add pattern-based features for 15m trading.
        """
        # Recent price action (last 5 candles)
        for i in range(1, 6):
            df[f'ret_lag{i}'] = df['close'].pct_change(i) * 100
            df[f'high_lag{i}'] = df['high'].shift(i)
            df[f'low_lag{i}'] = df['low'].shift(i)
        
        # Consecutive up/down candles
        df['direction'] = np.sign(df['close'] - df['open'])
        df['consecutive_up'] = (df['direction'] == 1).astype(int).rolling(5, min_periods=1).sum()
        df['consecutive_down'] = (df['direction'] == -1).astype(int).rolling(5, min_periods=1).sum()
        
        # Highest/lowest in last N candles
        df['high_5'] = df['high'].rolling(5).max()
        df['low_5'] = df['low'].rolling(5).min()
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        
        # Distance to recent highs/lows
        df['dist_high_5'] = df['close'] - df['high_5']
        df['dist_low_5'] = df['close'] - df['low_5']
        df['dist_high_20'] = df['close'] - df['high_20']
        df['dist_low_20'] = df['close'] - df['low_20']
        
        return df
