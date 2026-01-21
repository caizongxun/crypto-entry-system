import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class RegimeDetector:
    """Multi-timeframe regime detection for signal confirmation."""
    
    @staticmethod
    def detect_mean_reversion_zone(
        close: pd.Series,
        sma: pd.Series,
        bb_upper: pd.Series,
        bb_lower: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """Detect mean reversion zones - price is extreme from average.
        
        Returns:
            1 if price is in mean reversion zone, 0 otherwise
        """
        distance_to_sma = (close - sma).abs() / sma
        bb_width = (bb_upper - bb_lower) / sma
        
        # In mean reversion zone if:
        # 1. Price is > 1.5 standard deviations from SMA
        # 2. Bollinger Band width is narrow (squeeze)
        mean_reversion = ((distance_to_sma > 0.015) & (bb_width < 0.08)).astype(int)
        return mean_reversion
    
    @staticmethod
    def detect_volume_breakout(
        volume: pd.Series,
        close: pd.Series,
        period: int = 20,
        volume_threshold: float = 1.5
    ) -> pd.Series:
        """Detect volume breakout followed by reversal.
        
        High volume + trend reversal = potential reversal point.
        """
        avg_volume = volume.rolling(window=period).mean()
        volume_ratio = volume / avg_volume.replace(0, np.nan)
        
        # Price momentum reversal
        returns = close.pct_change()
        momentum_reversal = (returns.diff().abs() > 0.002).astype(int)
        
        breakout = ((volume_ratio > volume_threshold) & (momentum_reversal == 1)).astype(int)
        return breakout
    
    @staticmethod
    def detect_divergence_reversal(
        close: pd.Series,
        rsi: pd.Series,
        macd: pd.Series,
        period: int = 10
    ) -> pd.Series:
        """Detect divergence patterns that often precede reversals.
        
        Bearish divergence: Price makes new high but RSI/MACD makes lower high
        Bullish divergence: Price makes new low but RSI/MACD makes higher low
        """
        close_high = close.rolling(window=period).max()
        close_low = close.rolling(window=period).min()
        
        rsi_high = rsi.rolling(window=period).max()
        rsi_low = rsi.rolling(window=period).min()
        
        macd_high = macd.rolling(window=period).max()
        macd_low = macd.rolling(window=period).min()
        
        # Bearish divergence
        bearish = ((close == close_high) & (rsi < rsi_high)).astype(int)
        
        # Bullish divergence
        bullish = ((close == close_low) & (rsi > rsi_low)).astype(int)
        
        divergence = bearish | bullish
        return divergence.astype(int)
    
    @staticmethod
    def detect_exhaustion_candle(
        open_price: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """Detect exhaustion candles - large moves on high volume with small close change.
        
        Often precedes reversals.
        """
        candle_range = (high - low) / open_price
        body_size = (close - open_price).abs() / open_price
        avg_volume = volume.rolling(window=period).mean()
        volume_surge = volume > avg_volume * 1.5
        
        # Exhaustion: large range, small body, high volume
        exhaustion = (
            (candle_range > 0.02) &
            (body_size < candle_range * 0.3) &
            volume_surge
        ).astype(int)
        
        return exhaustion
    
    @staticmethod
    def detect_reversal_confirmation(
        close: pd.Series,
        volume: pd.Series,
        rsi: pd.Series,
        macd: pd.Series,
        bb_upper: pd.Series,
        bb_lower: pd.Series,
        period: int = 20
    ) -> pd.DataFrame:
        """Combined reversal detection with multiple confirmations.
        
        Returns:
            DataFrame with multiple reversal signals
        """
        result = pd.DataFrame(index=close.index)
        
        # Individual signals
        result['mean_reversion_zone'] = RegimeDetector.detect_mean_reversion_zone(
            close, close.rolling(period).mean(), bb_upper, bb_lower, period
        )
        
        result['volume_breakout'] = RegimeDetector.detect_volume_breakout(
            volume, close, period, 1.5
        )
        
        result['divergence'] = RegimeDetector.detect_divergence_reversal(
            close, rsi, macd, period
        )
        
        result['exhaustion'] = RegimeDetector.detect_exhaustion_candle(
            close.shift(1), close.rolling(period).max(), close.rolling(period).min(),
            close, volume, period
        )
        
        # Combined signal strength (0-4)
        result['reversal_strength'] = (
            result['mean_reversion_zone'] +
            result['volume_breakout'] +
            result['divergence'] +
            result['exhaustion']
        )
        
        return result
    
    @staticmethod
    def calculate_regime_features(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
        """Calculate all regime detection features."""
        if config is None:
            config = {}
        
        df = df.copy()
        period = config.get('period', 20)
        
        regime_df = RegimeDetector.detect_reversal_confirmation(
            df['close'],
            df['volume'],
            df['rsi'],
            df['macd'],
            df['bb_upper'],
            df['bb_lower'],
            period
        )
        
        for col in regime_df.columns:
            df[f'regime_{col}'] = regime_df[col]
        
        return df
