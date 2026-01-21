import pandas as pd
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Calculate technical indicators for feature engineering."""
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD and Signal Line."""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(close: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands."""
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range."""
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, smooth: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator."""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        k_percent = k_percent.rolling(window=smooth).mean()
        d_percent = k_percent.rolling(window=smooth).mean()
        
        return k_percent, d_percent
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume."""
        obv = np.where(close.diff() > 0, volume, np.where(close.diff() < 0, -volume, 0))
        return pd.Series(np.cumsum(obv), index=close.index)
    
    @staticmethod
    def adl(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Accumulation/Distribution Line."""
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        ad = clv * volume
        return ad.cumsum()
    
    @staticmethod
    def momentum(close: pd.Series, period: int = 10) -> pd.Series:
        """Momentum indicator."""
        return close.diff(period)
    
    @staticmethod
    def roc(close: pd.Series, period: int = 12) -> pd.Series:
        """Rate of Change."""
        return ((close - close.shift(period)) / close.shift(period)) * 100
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
        """Calculate all technical indicators.
        
        Args:
            df: OHLCV DataFrame
            config: Configuration dict with indicator parameters
        
        Returns:
            DataFrame with added indicator columns
        """
        if config is None:
            config = {
                'sma_periods': [5, 10, 20, 50, 200],
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'bollinger_period': 20,
                'bollinger_std': 2,
                'atr_period': 14,
                'stoch_period': 14,
                'stoch_smooth': 3,
            }
        
        df = df.copy()
        
        for period in config.get('sma_periods', [5, 10, 20, 50, 200]):
            df[f'sma_{period}'] = TechnicalIndicators.sma(df['close'], period)
        
        df['rsi'] = TechnicalIndicators.rsi(df['close'], config.get('rsi_period', 14))
        
        macd_line, signal_line, histogram = TechnicalIndicators.macd(
            df['close'],
            config.get('macd_fast', 12),
            config.get('macd_slow', 26),
            config.get('macd_signal', 9)
        )
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = histogram
        
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(
            df['close'],
            config.get('bollinger_period', 20),
            config.get('bollinger_std', 2)
        )
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        df['atr'] = TechnicalIndicators.atr(
            df['high'], df['low'], df['close'],
            config.get('atr_period', 14)
        )
        
        stoch_k, stoch_d = TechnicalIndicators.stochastic(
            df['high'], df['low'], df['close'],
            config.get('stoch_period', 14),
            config.get('stoch_smooth', 3)
        )
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        df['obv'] = TechnicalIndicators.obv(df['close'], df['volume'])
        df['adl'] = TechnicalIndicators.adl(df['high'], df['low'], df['close'], df['volume'])
        
        df['momentum'] = TechnicalIndicators.momentum(df['close'], 10)
        df['roc'] = TechnicalIndicators.roc(df['close'], 12)
        
        return df
