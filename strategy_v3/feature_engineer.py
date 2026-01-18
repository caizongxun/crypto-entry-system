"""
Feature Engineering for trading signals
Calculates technical indicators and derives ML features
"""

import logging
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

from .config import Config, FeatureConfig


logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Calculate and engineer features for trading signals"""
    
    def __init__(self, config: Config):
        """
        Initialize FeatureEngineer
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.feature_config = config.feature
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = None
    
    def calculate_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price action features (SMA, EMA, momentum)
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with added features
        """
        df = df.copy()
        
        # Simple Moving Averages
        for period in self.feature_config.sma_periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        # Exponential Moving Averages
        for period in self.feature_config.ema_periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # Momentum (Price change)
        df['momentum_1'] = df['close'].diff(1)
        df['momentum_5'] = df['close'].diff(5)
        df['momentum_10'] = df['close'].diff(10)
        
        # Price rate of change
        df['roc_5'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5)) * 100
        df['roc_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
        
        # Returns
        df['returns_1'] = df['close'].pct_change(1)
        df['returns_5'] = df['close'].pct_change(5)
        
        return df
    
    def calculate_oscillators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate oscillator features (RSI, MACD, Stochastic)
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with added features
        """
        df = df.copy()
        
        # RSI (Relative Strength Index)
        rsi = self._calculate_rsi(df['close'], self.feature_config.rsi_period)
        df['rsi'] = rsi
        df['rsi_overbought'] = (rsi > self.feature_config.rsi_threshold_high).astype(int)
        df['rsi_oversold'] = (rsi < self.feature_config.rsi_threshold_low).astype(int)
        
        # MACD (Moving Average Convergence Divergence)
        macd, signal, histogram = self._calculate_macd(
            df['close'],
            self.feature_config.macd_fast,
            self.feature_config.macd_slow,
            self.feature_config.macd_signal
        )
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_histogram'] = histogram
        df['macd_crossover'] = ((macd > signal) & (macd.shift(1) <= signal.shift(1))).astype(int)
        
        # Stochastic Oscillator
        k_percent, d_percent = self._calculate_stochastic(df['high'], df['low'], df['close'])
        df['stoch_k'] = k_percent
        df['stoch_d'] = d_percent
        
        return df
    
    def calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility features (ATR, Bollinger Bands, std)
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with added features
        """
        df = df.copy()
        
        # ATR (Average True Range)
        atr = self._calculate_atr(df['high'], df['low'], df['close'], self.feature_config.atr_period)
        df['atr'] = atr
        df['atr_pct'] = (atr / df['close']) * 100
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(
            df['close'],
            self.feature_config.bb_period,
            self.feature_config.bb_std_dev
        )
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Standard deviation of returns
        df['volatility_5'] = df['returns_1'].rolling(window=5).std()
        df['volatility_10'] = df['returns_1'].rolling(window=10).std()
        df['volatility_20'] = df['returns_1'].rolling(window=20).std()
        
        return df
    
    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based features
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with added features
        """
        df = df.copy()
        
        # Volume moving average
        df['volume_ma'] = df['volume'].rolling(window=self.feature_config.volume_ma_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_ma'] = df['obv'].rolling(window=12).mean()
        df['obv_signal'] = ((df['obv'] > df['obv_ma']) & (df['obv'].shift(1) <= df['obv_ma'].shift(1))).astype(int)
        
        # Volume momentum
        df['volume_momentum'] = df['volume'].diff()
        
        return df
    
    def engineer_all_features(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """
        Engineer all features from raw OHLCV data
        
        Args:
            df: Raw OHLCV DataFrame
            target: Optional target variable for feature selection
            
        Returns:
            Tuple of (engineered_df, selected_features_array)
        """
        logger.info("Engineering features...")
        
        # Calculate all features
        df = self.calculate_price_action_features(df)
        df = self.calculate_oscillators(df)
        df = self.calculate_volatility_features(df)
        df = self.calculate_volume_features(df)
        
        # Remove NaN rows
        df = df.dropna()
        
        logger.info(f"Total features created: {len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']])}")
        
        # Feature selection if target provided
        X_selected = None
        if target is not None:
            X_selected = self.select_features(df, target, n_features=25)
        
        return df, X_selected
    
    def select_features(self, df: pd.DataFrame, target: pd.Series, n_features: int = 25) -> np.ndarray:
        """
        Select top features using SelectKBest
        
        Args:
            df: Feature DataFrame
            target: Target variable
            n_features: Number of features to select
            
        Returns:
            Array of selected features
        """
        # Select only numeric feature columns
        feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'close_time']]
        X = df[feature_cols].copy()
        
        # Remove any rows with NaN
        mask = ~(X.isna().any(axis=1) | target.isna())
        X = X[mask]
        target_clean = target[mask]
        
        # Select features
        selector = SelectKBest(score_func=f_regression, k=min(n_features, X.shape[1]))
        X_selected = selector.fit_transform(X, target_clean)
        
        self.selected_features = X.columns[selector.get_support()].tolist()
        self.feature_selector = selector
        
        logger.info(f"Selected {len(self.selected_features)} features")
        
        return X_selected
    
    def normalize_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Normalize features using StandardScaler
        
        Args:
            X: Features DataFrame
            
        Returns:
            Scaled features array
        """
        return self.scaler.fit_transform(X)
    
    # Private helper methods
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    @staticmethod
    def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate ATR"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def _calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    @staticmethod
    def _calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=3).mean()
        return k_percent, d_percent
