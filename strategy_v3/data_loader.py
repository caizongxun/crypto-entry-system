"""
Data loader module for Strategy V3.

Handles loading OHLCV data from HuggingFace datasets.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
from loguru import logger
from huggingface_hub import hf_hub_download


class DataLoader:
    """
    Loads OHLCV data from HuggingFace datasets.
    """

    def __init__(self, hf_repo: str, cache_dir: str = './data_cache', verbose: bool = False):
        """
        Initialize DataLoader.

        Args:
            hf_repo: HuggingFace repository name
            cache_dir: Directory to cache downloaded data
            verbose: Enable verbose logging
        """
        self.hf_repo = hf_repo
        self.cache_dir = cache_dir
        self.verbose = verbose

        os.makedirs(cache_dir, exist_ok=True)

    def load_data(
        self,
        symbol: str,
        timeframe: str,
        cache: bool = True
    ) -> pd.DataFrame:
        """
        Load OHLCV data from HuggingFace dataset.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '15m', '1h', '1d')
            cache: Use cached data if available

        Returns:
            DataFrame with OHLCV data indexed by open_time
        """
        # Extract base asset from symbol (BTCUSDT -> BTC)
        base = symbol.replace('USDT', '').replace('BUSD', '')
        filename = f'{base}_{timeframe}.parquet'

        # Construct remote path following the dataset structure
        remote_path = f'klines/{symbol}/{filename}'

        try:
            if self.verbose:
                logger.info(f'Downloading {symbol} {timeframe} data from HuggingFace...')

            # Download from HuggingFace
            local_path = hf_hub_download(
                repo_id=self.hf_repo,
                filename=remote_path,
                repo_type='dataset',
                cache_dir=self.cache_dir,
                force_download=not cache
            )

            if self.verbose:
                logger.info(f'Loading data from {local_path}')

            # Read parquet file
            df = pd.read_parquet(local_path)
            return self._preprocess_data(df)

        except Exception as e:
            logger.error(f'Failed to load data for {symbol} {timeframe}: {str(e)}')
            raise

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess OHLCV data from HuggingFace dataset.

        Args:
            df: Raw DataFrame from parquet file

        Returns:
            Preprocessed DataFrame with proper index and columns
        """
        # Create copy to avoid modifying original
        df = df.copy()

        # Convert open_time to datetime and set as index
        if 'open_time' in df.columns:
            df['open_time'] = pd.to_datetime(df['open_time'])
            df.set_index('open_time', inplace=True)
        else:
            raise ValueError('Missing open_time column in dataset')

        # Select required OHLCV columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # Check if all required columns exist
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f'Missing required columns. Expected {required_cols}, got {list(df.columns)}')

        # Keep only OHLCV columns
        df = df[required_cols].copy()

        # Ensure numeric types
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle missing values using new pandas API
        df = df.ffill()
        df = df.bfill()

        # Remove any remaining NaN rows
        df = df.dropna()

        # Sort by index (should already be sorted, but ensure it)
        df = df.sort_index()

        if self.verbose:
            logger.info(f'Loaded {len(df)} candles')
            logger.info(f'Date range: {df.index[0]} to {df.index[-1]}')
            logger.info(f'Columns: {list(df.columns)}')

        return df

    def split_train_test(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets.

        Args:
            df: Input DataFrame
            train_ratio: Proportion for training (0-1)

        Returns:
            Tuple of (train_df, test_df)
        """
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        if self.verbose:
            logger.info(f'Train set: {len(train_df)} candles ({train_ratio*100:.1f}%)')
            logger.info(f'Test set: {len(test_df)} candles ({(1-train_ratio)*100:.1f}%)')

        return train_df, test_df

    @staticmethod
    def validate_data(df: pd.DataFrame, min_candles: int = 50) -> bool:
        """
        Validate OHLCV data quality.

        Args:
            df: Input DataFrame
            min_candles: Minimum required candles

        Returns:
            True if valid, False otherwise
        """
        # Check minimum candles
        if len(df) < min_candles:
            logger.error(f'Insufficient data: {len(df)} candles < {min_candles} minimum')
            return False

        # Check required columns
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(set(df.columns)):
            logger.error(f'Missing required columns: {required_cols - set(df.columns)}')
            return False

        # Check for NaN values
        if df[['open', 'high', 'low', 'close', 'volume']].isna().any().any():
            logger.error('Data contains NaN values')
            return False

        # Check price relationships
        if (df['high'] < df['low']).any():
            logger.error('Invalid price relationship: high < low')
            return False
            
        # Allow low to be less than or equal to both open and close
        if ((df['low'] > df['open']) & (df['low'] > df['close'])).any():
            logger.error('Invalid price relationship: low > both open and close')
            return False

        # Check that prices are positive
        if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
            logger.error('Invalid prices: contains non-positive values')
            return False

        return True
