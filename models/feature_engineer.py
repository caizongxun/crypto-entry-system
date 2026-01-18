import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from models.config import TECHNICAL_INDICATORS_CONFIG
from models.advanced_feature_engineer import AdvancedFeatureEngineer


class FeatureEngineer:
    """Calculate technical indicators and engineer features from OHLCV data.
    
    Combines traditional technical indicators with advanced ML-optimized features:
    - Technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV
    - Advanced features: Bounce failure memory, volume anomalies, reversal strength, time patterns
    """

    def __init__(self, config: Dict = None):
        self.config = config or TECHNICAL_INDICATORS_CONFIG
        self.df = None
        self.advanced_engineer = AdvancedFeatureEngineer()

    def calculate_sma(self, df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        """Calculate Simple Moving Average."""
        return df[column].rolling(window=period).mean()

    def calculate_ema(self, df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        """Calculate Exponential Moving Average."""
        return df[column].ewm(span=period, adjust=False).mean()

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
        """Calculate Relative Strength Index (0-100)."""
        delta = df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def calculate_macd(self, df: pd.DataFrame, fast: int = 12,
                      slow: int = 26, signal: int = 9,
                      column: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD and Signal line."""
        ema_fast = df[column].ewm(span=fast, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line.fillna(0), signal_line.fillna(0), histogram.fillna(0)

    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20,
                                 std_dev: int = 2, column: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = df[column].rolling(window=period).mean()
        std = df[column].rolling(window=period).std()
        bb_upper = sma + (std * std_dev)
        bb_lower = sma - (std * std_dev)
        return bb_upper.fillna(0), sma.fillna(0), bb_lower.fillna(0)

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr.fillna(0)

    def calculate_momentum(self, df: pd.DataFrame, period: int = 14,
                          column: str = 'close') -> pd.Series:
        """Calculate Price Momentum (Rate of Change)."""
        momentum = ((df[column] - df[column].shift(period)) / df[column].shift(period)) * 100
        return momentum.fillna(0).replace([np.inf, -np.inf], 0)

    def calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv.replace([np.inf, -np.inf], 0)

    def calculate_volatility(self, df: pd.DataFrame, period: int = 20,
                            column: str = 'close') -> pd.Series:
        """Calculate Price Volatility (Standard Deviation)."""
        returns = df[column].pct_change()
        volatility = returns.rolling(window=period).std() * 100
        return volatility.fillna(0).replace([np.inf, -np.inf], 0)

    def calculate_volume_profile(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Volume-weighted price profile."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        volume_weighted = (typical_price * df['volume']).rolling(window=period).sum()
        return volume_weighted.fillna(0)

    def calculate_divergence(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Price-RSI Divergence indicator."""
        rsi = self.calculate_rsi(df, period)
        price_range = df['close'].rolling(window=period).apply(
            lambda x: (x.max() - x.min()) / x.mean() if x.mean() != 0 else 0,
            raw=False
        )
        price_range = price_range.replace([np.inf, -np.inf], 0).fillna(0)
        price_range_max = price_range.max()
        if price_range_max == 0:
            price_range_max = 1
        divergence = (rsi / 100) - (price_range / price_range_max)
        return divergence.fillna(0).replace([np.inf, -np.inf], 0)

    def calculate_trend_strength(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate trend strength using ADX-like calculation."""
        plus_dm = pd.Series(0.0, index=df.index)
        minus_dm = pd.Series(0.0, index=df.index)
        tr = self.calculate_atr(df, 1)

        for i in range(1, len(df)):
            up = df['high'].iloc[i] - df['high'].iloc[i-1]
            down = df['low'].iloc[i-1] - df['low'].iloc[i]

            if up > down and up > 0:
                plus_dm.iloc[i] = up
            else:
                plus_dm.iloc[i] = 0

            if down > up and down > 0:
                minus_dm.iloc[i] = down
            else:
                minus_dm.iloc[i] = 0

        tr_rolling = tr.rolling(window=period).mean()
        tr_rolling = tr_rolling.replace(0, 1)
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr_rolling)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr_rolling)
        
        denominator = plus_di + minus_di
        denominator = denominator.replace(0, 1)
        dx = 100 * abs(plus_di - minus_di) / denominator
        adx = dx.rolling(window=period).mean()

        return adx.fillna(0).replace([np.inf, -np.inf], 0)

    def calculate_volume_momentum(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Volume Momentum - rate of change of OBV."""
        obv = self.calculate_obv(df)
        obv_shift = obv.shift(period)
        obv_shift_abs = obv_shift.abs()
        obv_shift_abs = obv_shift_abs.replace(0, 1)
        volume_momentum = ((obv - obv_shift) / obv_shift_abs + 1) * 100
        return volume_momentum.fillna(0).replace([np.inf, -np.inf], 0)

    def calculate_price_position_in_range(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate where price sits in its range (0-100)."""
        highest = df['high'].rolling(window=period).max()
        lowest = df['low'].rolling(window=period).min()
        price_range = highest - lowest
        price_range = price_range.replace(0, 1)
        position = ((df['close'] - lowest) / price_range) * 100
        return position.fillna(50).replace([np.inf, -np.inf], 50)

    def calculate_volume_relative_strength(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate relative strength of trading volume."""
        avg_volume = df['volume'].rolling(window=period).mean()
        avg_volume = avg_volume.replace(0, 1)
        volume_ratio = df['volume'] / avg_volume
        return volume_ratio.fillna(1.0).replace([np.inf, -np.inf], 1.0)

    def calculate_close_location(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate close location relative to high-low range (0-1)."""
        high_low_diff = df['high'] - df['low']
        high_low_diff = high_low_diff.replace(0, 1)
        close_high_diff = df['high'] - df['close']
        close_location = (high_low_diff - close_high_diff) / high_low_diff
        return close_location.fillna(0.5).replace([np.inf, -np.inf], 0.5)

    def calculate_momentum_divergence(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate divergence between price momentum and volume momentum."""
        price_momentum = self.calculate_momentum(df, period)
        volume_momentum = self.calculate_volume_momentum(df, period)
        divergence = (price_momentum - volume_momentum) / 100
        return divergence.fillna(0).replace([np.inf, -np.inf], 0)

    def calculate_volatility_acceleration(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate rate of change of volatility."""
        volatility = self.calculate_volatility(df, period)
        vol_acceleration = volatility.diff(period) / volatility.shift(period).replace(0, 1)
        return vol_acceleration.fillna(0).replace([np.inf, -np.inf], 0)

    def calculate_multi_timeframe_strength(self, df: pd.DataFrame) -> pd.Series:
        """Combine multiple timeframe signals (simple multi-TF indicator)."""
        fast_ema = self.calculate_ema(df, 5)
        medium_ema = self.calculate_ema(df, 13)
        slow_ema = self.calculate_ema(df, 50)
        
        fast_signal = (df['close'] > fast_ema).astype(int)
        medium_signal = (df['close'] > medium_ema).astype(int)
        slow_signal = (df['close'] > slow_ema).astype(int)
        
        return (fast_signal + medium_signal + slow_signal) / 3

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators and create feature matrix.
        
        Includes both traditional technical indicators and advanced ML features.
        """
        df_features = df.copy()

        sma_config = self.config.get('sma_fast', {})
        sma_medium_config = self.config.get('sma_medium', {})
        sma_slow_config = self.config.get('sma_slow', {})
        ema_fast_config = self.config.get('ema_fast', {})
        ema_slow_config = self.config.get('ema_slow', {})
        rsi_config = self.config.get('rsi', {})
        macd_config = self.config.get('macd', {})
        bb_config = self.config.get('bb', {})
        atr_config = self.config.get('atr', {})

        df_features['sma_fast'] = self.calculate_sma(df_features, sma_config.get('period', 20))
        df_features['sma_medium'] = self.calculate_sma(df_features, sma_medium_config.get('period', 50))
        df_features['sma_slow'] = self.calculate_sma(df_features, sma_slow_config.get('period', 200))
        df_features['ema_fast'] = self.calculate_ema(df_features, ema_fast_config.get('period', 12))
        df_features['ema_slow'] = self.calculate_ema(df_features, ema_slow_config.get('period', 26))
        df_features['rsi'] = self.calculate_rsi(df_features, rsi_config.get('period', 14))

        macd_line, signal_line, histogram = self.calculate_macd(
            df_features,
            fast=macd_config.get('fast', 12),
            slow=macd_config.get('slow', 26),
            signal=macd_config.get('signal', 9)
        )
        df_features['macd'] = macd_line
        df_features['macd_signal'] = signal_line
        df_features['macd_histogram'] = histogram

        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(
            df_features,
            period=bb_config.get('period', 20),
            std_dev=bb_config.get('std_dev', 2)
        )
        df_features['bb_upper'] = bb_upper
        df_features['bb_middle'] = bb_middle
        df_features['bb_lower'] = bb_lower

        df_features['atr'] = self.calculate_atr(df_features, atr_config.get('period', 14))
        df_features['momentum'] = self.calculate_momentum(df_features)
        df_features['volatility'] = self.calculate_volatility(df_features)
        df_features['obv'] = self.calculate_obv(df_features)
        df_features['trend_strength'] = self.calculate_trend_strength(df_features)
        
        df_features['volume_momentum'] = self.calculate_volume_momentum(df_features)
        df_features['price_position'] = self.calculate_price_position_in_range(df_features)
        df_features['volume_relative_strength'] = self.calculate_volume_relative_strength(df_features)
        df_features['close_location'] = self.calculate_close_location(df_features)
        df_features['momentum_divergence'] = self.calculate_momentum_divergence(df_features)
        df_features['volatility_acceleration'] = self.calculate_volatility_acceleration(df_features)
        df_features['multi_timeframe_strength'] = self.calculate_multi_timeframe_strength(df_features)
        
        # Add Bollinger Bands width and position (for BB bounce analysis)
        df_features['bb_width'] = df_features['bb_upper'] - df_features['bb_lower']
        df_features['bb_position'] = (df_features['close'] - df_features['bb_lower']) / (df_features['bb_width'] + 1e-6)
        df_features['bb_position'] = df_features['bb_position'].clip(0, 1)
        df_features['basis_slope'] = df_features['bb_middle'].diff()

        # Apply advanced feature engineering AFTER BB metrics are available
        print("\nIntegrating advanced ML-optimized features...")
        df_features = self.advanced_engineer.engineer_all_features(df_features)

        df_features = df_features.bfill().ffill()
        
        for col in df_features.select_dtypes(include=[np.number]).columns:
            df_features[col] = df_features[col].replace([np.inf, -np.inf], 0)
            df_features[col] = df_features[col].fillna(0)
            if df_features[col].max() > 1e10:
                df_features[col] = df_features[col].clip(lower=-1e10, upper=1e10)
        
        print(f"Features engineered: {len(df_features)} rows, {len(df_features.columns)} columns")
        return df_features

    def get_feature_names(self) -> List[str]:
        """Return list of engineered feature names."""
        technical_features = [
            # Moving Averages
            'sma_fast', 'sma_medium', 'sma_slow', 'ema_fast', 'ema_slow',
            # RSI and Momentum
            'rsi', 'macd', 'macd_signal', 'macd_histogram', 'momentum',
            # Bollinger Bands
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position', 'basis_slope',
            # Volatility and Trend
            'atr', 'volatility', 'trend_strength',
            # Volume
            'obv', 'volume_momentum', 'volume_relative_strength',
            # Derivatives
            'price_position', 'close_location', 'momentum_divergence', 'volatility_acceleration',
            'multi_timeframe_strength',
        ]
        
        advanced_features = self.advanced_engineer.get_feature_list()
        
        return technical_features + advanced_features
