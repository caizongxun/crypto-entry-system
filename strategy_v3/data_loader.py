"""
Data Loader for HuggingFace crypto OHLCV datasets
"""

import logging
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from pathlib import Path

from .config import Config, HuggingFaceConfig


logger = logging.getLogger(__name__)


class DataLoader:
    """Load cryptocurrency OHLCV data from HuggingFace"""
    
    def __init__(self, config: Config):
        """
        Initialize DataLoader
        
        Args:
            config: Config object containing dataset settings
        """
        self.config = config
        self.hf_config = config.huggingface
        self.cache_dir = Path(self.hf_config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_klines(
        self,
        symbol: str,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Load K-line data from HuggingFace
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: Timeframe ('15m', '1h', '1d')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Extract base asset name
            base = symbol.replace('USDT', '')
            filename = f"{base}_{timeframe}.parquet"
            path_in_repo = f"klines/{symbol}/{filename}"
            
            logger.info(f"Loading {symbol} {timeframe} data from HuggingFace...")
            
            # Download from HuggingFace
            local_path = hf_hub_download(
                repo_id=self.hf_config.repo_id,
                filename=path_in_repo,
                repo_type=self.hf_config.repo_type,
                cache_dir=str(self.cache_dir)
            )
            
            # Load parquet file
            df = pd.read_parquet(local_path)
            
            # Ensure datetime columns
            df['open_time'] = pd.to_datetime(df['open_time'])
            df['close_time'] = pd.to_datetime(df['close_time'])
            
            # Set index
            df = df.set_index('open_time')
            df = df.sort_index()
            
            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]
            
            logger.info(f"Loaded {len(df)} records from {df.index[0]} to {df.index[-1]}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load {symbol} {timeframe}: {str(e)}")
            raise
    
    def load_multiple_timeframes(
        self,
        symbol: str,
        timeframes: list = None
    ) -> dict:
        """
        Load multiple timeframe data for same symbol
        
        Args:
            symbol: Trading pair
            timeframes: List of timeframes to load
            
        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        if timeframes is None:
            timeframes = self.config.feature.timeframes
        
        data = {}
        for timeframe in timeframes:
            try:
                data[timeframe] = self.load_klines(symbol, timeframe)
            except Exception as e:
                logger.warning(f"Failed to load {symbol} {timeframe}: {str(e)}")
                continue
        
        return data
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate data quality
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        issues = []
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for NaN values
        nan_counts = df[required_cols].isna().sum()
        if nan_counts.sum() > 0:
            issues.append(f"Found NaN values: {nan_counts[nan_counts > 0].to_dict()}")
        
        # Check price logic
        if (df['high'] < df['low']).any():
            issues.append("Found high < low values")
        if (df['close'] < df['low']).any() or (df['close'] > df['high']).any():
            issues.append("Found close outside high-low range")
        
        # Check volume
        if (df['volume'] < 0).any():
            issues.append("Found negative volume")
        
        if issues:
            message = "; ".join(issues)
            return False, message
        
        return True, "Data validation passed"
    
    def resample_data(
        self,
        df: pd.DataFrame,
        target_timeframe: str
    ) -> pd.DataFrame:
        """
        Resample OHLCV data to different timeframe
        
        Args:
            df: Original DataFrame
            target_timeframe: Target timeframe (e.g., '1h', '4h')
            
        Returns:
            Resampled DataFrame
        """
        ohlc_mapping = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'quote_asset_volume': 'sum',
            'number_of_trades': 'sum',
            'taker_buy_base_asset_volume': 'sum',
            'taker_buy_quote_asset_volume': 'sum'
        }
        
        # Select available columns
        available_cols = {k: v for k, v in ohlc_mapping.items() if k in df.columns}
        
        resampled = df.resample(target_timeframe).agg(available_cols)
        
        # Remove rows with NaN
        resampled = resampled.dropna()
        
        return resampled
    
    def get_latest_bars(
        self,
        df: pd.DataFrame,
        n_bars: int = 100
    ) -> pd.DataFrame:
        """
        Get the latest N bars from data
        
        Args:
            df: Full DataFrame
            n_bars: Number of latest bars to return
            
        Returns:
            DataFrame with latest bars
        """
        return df.tail(n_bars).copy()
    
    def split_train_test(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets
        
        Args:
            df: Full DataFrame
            train_ratio: Ratio of training data
            
        Returns:
            Tuple of (train_df, test_df)
        """
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        logger.info(f"Split data: {len(train_df)} train, {len(test_df)} test")
        
        return train_df, test_df
