import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class VolatilityRegime:
    """Detect and analyze volatility regimes for multi-timeframe analysis."""
    
    @staticmethod
    def calculate_volatility(close: pd.Series, period: int = 20) -> pd.Series:
        """Calculate rolling volatility (standard deviation of returns)."""
        returns = close.pct_change()
        return returns.rolling(window=period).std()
    
    @staticmethod
    def get_regime_state(
        volatility: pd.Series,
        short_window: int = 5,
        long_window: int = 20,
        expand_threshold: float = 1.5,
        squeeze_threshold: float = 0.7
    ) -> pd.Series:
        """Determine volatility regime state.
        
        Returns:
            'expansion': High volatility (short_vol > long_vol * expand_threshold)
            'squeeze': Low volatility (short_vol < long_vol * squeeze_threshold)
            'normal': Between expansion and squeeze
        """
        short_vol = volatility.rolling(window=short_window).mean()
        long_vol = volatility.rolling(window=long_window).mean()
        
        vol_ratio = short_vol / long_vol.replace(0, np.nan)
        
        regime = pd.Series('normal', index=volatility.index)
        regime[vol_ratio > expand_threshold] = 'expansion'
        regime[vol_ratio < squeeze_threshold] = 'squeeze'
        
        return regime
    
    @staticmethod
    def calculate_bollinger_squeeze_factor(high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Bollinger Band squeeze factor.
        
        Values close to 1.0 indicate squeeze, close to 0 indicates expansion.
        """
        hl_range = high - low
        avg_range = hl_range.rolling(window=period).mean()
        current_range = high.rolling(window=1).max() - low.rolling(window=1).min()
        
        squeeze_factor = current_range / avg_range.replace(0, np.nan)
        return squeeze_factor
    
    @staticmethod
    def calculate_price_momentum(close: pd.Series, period: int = 20) -> pd.Series:
        """Calculate price momentum for regime confirmation."""
        return close.pct_change(periods=period)
    
    @staticmethod
    def calculate_volume_regime(volume: pd.Series, period: int = 20) -> pd.Series:
        """Calculate volume regime state relative to historical average."""
        avg_vol = volume.rolling(window=period).mean()
        vol_ratio = volume / avg_vol.replace(0, np.nan)
        return vol_ratio
    
    @staticmethod
    def detect_reversal_zones(
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        period: int = 50
    ) -> tuple:
        """Detect potential reversal zones based on recent highs/lows.
        
        Returns:
            (resistance_level, support_level, distance_to_resistance, distance_to_support)
        """
        recent_high = high.rolling(window=period).max()
        recent_low = low.rolling(window=period).min()
        
        dist_to_resistance = (recent_high - close) / close * 100
        dist_to_support = (close - recent_low) / close * 100
        
        return recent_high, recent_low, dist_to_resistance, dist_to_support
    
    @staticmethod
    def calculate_regime_features(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
        """Calculate all volatility regime features.
        
        Args:
            df: OHLCV DataFrame
            config: Configuration dict
        
        Returns:
            DataFrame with added regime features
        """
        if config is None:
            config = {
                'short_window': 5,
                'long_window': 20,
                'expand_threshold': 1.5,
                'squeeze_threshold': 0.7,
            }
        
        df = df.copy()
        
        vol = VolatilityRegime.calculate_volatility(df['close'], period=20)
        df['volatility'] = vol
        
        regime = VolatilityRegime.get_regime_state(
            vol,
            config.get('short_window', 5),
            config.get('long_window', 20),
            config.get('expand_threshold', 1.5),
            config.get('squeeze_threshold', 0.7)
        )
        df['regime'] = regime
        df['regime_expansion'] = (regime == 'expansion').astype(int)
        df['regime_squeeze'] = (regime == 'squeeze').astype(int)
        df['regime_normal'] = (regime == 'normal').astype(int)
        
        bb_squeeze = VolatilityRegime.calculate_bollinger_squeeze_factor(
            df['high'], df['low'], period=20
        )
        df['bb_squeeze_factor'] = bb_squeeze
        
        momentum = VolatilityRegime.calculate_price_momentum(df['close'], period=20)
        df['momentum_20'] = momentum
        
        vol_regime = VolatilityRegime.calculate_volume_regime(df['volume'], period=20)
        df['volume_regime'] = vol_regime
        df['volume_surge'] = (vol_regime > 1.5).astype(int)
        
        res, sup, dist_res, dist_sup = VolatilityRegime.detect_reversal_zones(
            df['close'], df['high'], df['low'], period=50
        )
        df['resistance_level'] = res
        df['support_level'] = sup
        df['dist_to_resistance_percent'] = dist_res
        df['dist_to_support_percent'] = dist_sup
        
        return df
