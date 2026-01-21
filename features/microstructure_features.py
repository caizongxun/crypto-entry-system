import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class MicrostructureFeatures:
    """Extract order flow and market microstructure features."""
    
    @staticmethod
    def calculate_taker_buy_ratio(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate ratio of aggressive buyer volume to total volume.
        
        High ratio indicates strong buying pressure.
        """
        taker_ratio = df['taker_buy_quote_asset_volume'] / df['quote_asset_volume'].replace(0, np.nan)
        smoothed_ratio = taker_ratio.rolling(window=window).mean()
        return smoothed_ratio
    
    @staticmethod
    def calculate_order_flow_imbalance(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate cumulative order flow imbalance.
        
        Positive values indicate buying pressure, negative indicate selling pressure.
        """
        buy_volume = df['taker_buy_quote_asset_volume']
        sell_volume = df['quote_asset_volume'] - df['taker_buy_quote_asset_volume']
        
        imbalance = buy_volume - sell_volume
        cumulative_imbalance = imbalance.rolling(window=window).sum()
        
        return cumulative_imbalance
    
    @staticmethod
    def calculate_volume_strength(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate volume strength relative to price movement.
        
        High values indicate strong conviction in price moves.
        """
        price_change = df['close'].pct_change().abs()
        volume_change = df['volume'].pct_change().rolling(window=period).mean()
        
        strength = volume_change / price_change.rolling(window=period).mean().replace(0, np.nan)
        return strength
    
    @staticmethod
    def calculate_volume_price_trend(df: pd.DataFrame) -> pd.Series:
        """Volume Price Trend indicator.
        
        Combines volume and price direction.
        """
        price_change = df['close'].pct_change()
        vpt = (price_change * df['volume']).cumsum()
        return vpt
    
    @staticmethod
    def detect_accumulation_distribution(df: pd.DataFrame, threshold: float = 0.65) -> pd.Series:
        """Detect accumulation or distribution phases.
        
        Accumulation: high close near opening, high volume
        Distribution: low close near opening, high volume
        """
        # Close Location Value
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        
        # Normalize to 0-1 range
        clv_normalized = (clv + 1) / 2
        
        # Accumulation when close is in upper half and volume is high
        volume_avg = df['volume'].rolling(window=20).mean()
        volume_surge = df['volume'] > volume_avg
        
        accumulation = (clv_normalized > threshold) & volume_surge
        distribution = (clv_normalized < (1 - threshold)) & volume_surge
        
        phase = pd.Series('neutral', index=df.index)
        phase[accumulation] = 'accumulation'
        phase[distribution] = 'distribution'
        
        return phase
    
    @staticmethod
    def calculate_trade_intensity(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate trading intensity based on number of trades.
        
        Higher intensity suggests strong market participation.
        """
        avg_trades = df['number_of_trades'].rolling(window=period).mean()
        intensity = df['number_of_trades'] / avg_trades.replace(0, np.nan)
        return intensity
    
    @staticmethod
    def calculate_aggressive_volume_ratio(df: pd.DataFrame, window: int = 10) -> pd.Series:
        """Calculate ratio of aggressive (taker) buy volume to total volume.
        
        Smoothed version shows consistent buying/selling pressure.
        """
        aggressive_buy = df['taker_buy_base_asset_volume']
        total_volume = df['volume']
        
        ratio = aggressive_buy / total_volume.replace(0, np.nan)
        smoothed = ratio.rolling(window=window).mean()
        
        return smoothed
    
    @staticmethod
    def detect_climactic_volume(df: pd.DataFrame, threshold: float = 2.0) -> pd.Series:
        """Detect climactic volume spikes (potential reversal signals).
        
        Climactic volume often precedes reversals.
        """
        volume_avg = df['volume'].rolling(window=20).mean()
        volume_ratio = df['volume'] / volume_avg.replace(0, np.nan)
        
        climactic = (volume_ratio > threshold).astype(int)
        return climactic
    
    @staticmethod
    def calculate_price_volume_correlation(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate correlation between price change and volume.
        
        High positive correlation indicates strong conviction.
        """
        price_change = df['close'].pct_change()
        volume_change = df['volume'].pct_change()
        
        correlation = price_change.rolling(window=period).corr(volume_change)
        return correlation
    
    @staticmethod
    def calculate_vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Volume Weighted Average Price."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        return vwap
    
    @staticmethod
    def calculate_all_microstructure_features(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
        """Calculate all microstructure features.
        
        Args:
            df: OHLCV DataFrame
            config: Configuration dict
        
        Returns:
            DataFrame with added microstructure features
        """
        if config is None:
            config = {
                'order_flow_window': 20,
                'volume_profile_bins': 10,
                'aggressive_ratio_smooth': 10,
                'accumulation_threshold': 0.65,
            }
        
        df = df.copy()
        
        # Order Flow Features
        df['taker_buy_ratio'] = MicrostructureFeatures.calculate_taker_buy_ratio(
            df, config.get('order_flow_window', 20)
        )
        
        df['order_flow_imbalance'] = MicrostructureFeatures.calculate_order_flow_imbalance(
            df, config.get('order_flow_window', 20)
        )
        
        df['volume_strength'] = MicrostructureFeatures.calculate_volume_strength(df, 14)
        df['volume_price_trend'] = MicrostructureFeatures.calculate_volume_price_trend(df)
        
        # Accumulation/Distribution
        ad_phase = MicrostructureFeatures.detect_accumulation_distribution(
            df, config.get('accumulation_threshold', 0.65)
        )
        df['ad_phase'] = ad_phase
        df['accumulation'] = (ad_phase == 'accumulation').astype(int)
        df['distribution'] = (ad_phase == 'distribution').astype(int)
        
        # Trade Intensity
        df['trade_intensity'] = MicrostructureFeatures.calculate_trade_intensity(df, 20)
        
        # Aggressive Volume
        df['aggressive_volume_ratio'] = MicrostructureFeatures.calculate_aggressive_volume_ratio(
            df, config.get('aggressive_ratio_smooth', 10)
        )
        
        # Climactic Volume
        df['climactic_volume'] = MicrostructureFeatures.detect_climactic_volume(df, 2.0)
        
        # Price-Volume Correlation
        df['pv_correlation'] = MicrostructureFeatures.calculate_price_volume_correlation(df, 20)
        
        # VWAP
        df['vwap'] = MicrostructureFeatures.calculate_vwap(df, 20)
        df['price_vs_vwap'] = df['close'] / df['vwap'].replace(0, np.nan) - 1
        
        return df
