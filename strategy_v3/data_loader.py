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

        if verbose:
            logger.enable('strategy_v3')
        else:
            logger.disable('strategy_v3')

    def load_data(
        self,
        symbol: str,
        timeframe: str,
        cache: bool = True
    ) -> pd.DataFrame:
        """
        Load OHLCV data.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '15m', '1h', '1d')
            cache: Use cached data if available

        Returns:
            DataFrame with OHLCV data
        """
        # Extract base asset from symbol
        base = symbol.replace('USDT', '').replace('BUSD', '')
        filename = f'{base}_{timeframe}.parquet'

        # Construct remote path
        remote_path = f'klines/{symbol}/{filename}'

        # Try to download and cache
        try:
            if self.verbose:
                logger.info(f'Downloading {symbol} {timeframe} data from HuggingFace...')

            local_path = hf_hub_download(
                repo_id=self.hf_repo,
                filename=remote_path,
                repo_type='dataset',
                cache_dir=self.cache_dir,
                force_download=not cache
            )

            if self.verbose:
                logger.info(f'Loading data from {local_path}')

            df = pd.read_parquet(local_path)
            return self._preprocess_data(df)

        except Exception as e:
            logger.error(f'Failed to load data for {symbol} {timeframe}: {str(e)}')
            raise

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess OHLCV data.

        Args:
            df: Raw DataFrame

        Returns:
            Preprocessed DataFrame
        """
        # Create copy to avoid modifying original
        df = df.copy()

        # Ensure correct column names
        expected_cols = {'open', 'high', 'low', 'close', 'volume'}
        actual_cols = set(df.columns.str.lower())

        if not expected_cols.issubset(actual_cols):
            raise ValueError(f'Missing required columns. Expected {expected_cols}, got {actual_cols}')

        # Normalize column names to lowercase
        df.columns = df.columns.str.lower()

        # Handle timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

        # Sort by index
        df = df.sort_index()

        # Handle missing values
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')

        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove any remaining NaN rows
        df = df.dropna()

        if self.verbose:
            logger.info(f'Loaded {len(df)} candles')
            logger.info(f'Date range: {df.index[0]} to {df.index[-1]}')

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
            logger.info(f'Train set: {len(train_df)} candles')
            logger.info(f'Test set: {len(test_df)} candles')

        return train_df, test_df

    @staticmethod
    def validate_data(df: pd.DataFrame, min_candles: int = 50) -> bool:
        """
        Validate OHLCV data.

        Args:
            df: Input DataFrame
            min_candles: Minimum required candles

        Returns:
            True if valid, False otherwise
        """
        if len(df) < min_candles:
            return False

        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(set(df.columns)):
            return False

        # Check for NaN values
        if df[['open', 'high', 'low', 'close', 'volume']].isna().any().any():
            return False

        # Check price relationships
        if (df['high'] < df['low']).any():
            return False
        if (df['low'] < df['open']).any() or (df['low'] < df['close']).any():
            return False

        return True
