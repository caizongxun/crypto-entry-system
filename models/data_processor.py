import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from huggingface_hub import hf_hub_download
from models.config import (
    HUGGINGFACE_CONFIG, CACHE_DIR, DATA_DIR, DEFAULT_SYMBOL, DEFAULT_TIMEFRAME
)
import hashlib
import json


class DataProcessor:
    """Handle cryptocurrency OHLCV data loading and caching."""

    def __init__(self, symbol: str = DEFAULT_SYMBOL, timeframe: str = DEFAULT_TIMEFRAME):
        self.symbol = symbol
        self.timeframe = timeframe
        self.base = symbol.replace('USDT', '')
        self.repo_id = HUGGINGFACE_CONFIG['repo_id']
        self.cache_dir = DATA_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self) -> Path:
        """Generate cache file path based on symbol and timeframe."""
        filename = f"{self.base}_{self.timeframe}.parquet"
        return self.cache_dir / filename

    def check_cache_exists(self) -> bool:
        """Check if data already cached locally."""
        cache_path = self.get_cache_path()
        exists = cache_path.exists()
        print(f"Cache check for {self.symbol} ({self.timeframe}): {'Found' if exists else 'Not found'}")
        return exists

    def get_hf_file_path(self) -> str:
        """Construct HuggingFace dataset file path."""
        filename = f"{self.base}_{self.timeframe}.parquet"
        return f"{HUGGINGFACE_CONFIG['base_path']}/{self.symbol}/{filename}"

    def download_data(self) -> pd.DataFrame:
        """Download data from HuggingFace if not cached."""
        cache_path = self.get_cache_path()

        if cache_path.exists():
            print(f"Loading cached data from {cache_path}")
            return pd.read_parquet(cache_path)

        print(f"Downloading {self.symbol} {self.timeframe} from HuggingFace...")
        try:
            hf_file_path = self.get_hf_file_path()
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=hf_file_path,
                repo_type=HUGGINGFACE_CONFIG['repo_type'],
                cache_dir=str(self.cache_dir)
            )
            df = pd.read_parquet(local_path)
            print(f"Download successful. Saving to cache at {cache_path}")
            df.to_parquet(cache_path, index=False)
            return df
        except Exception as e:
            print(f"Error downloading data: {str(e)}")
            raise

    def load_data(self) -> pd.DataFrame:
        """Load data from cache or download if needed."""
        return self.download_data()

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and validate data."""
        df_copy = df.copy()

        df_copy['open_time'] = pd.to_datetime(df_copy['open_time'], utc=True)
        df_copy['close_time'] = pd.to_datetime(df_copy['close_time'], utc=True)

        df_copy = df_copy.sort_values('open_time').reset_index(drop=True)
        df_copy = df_copy.drop_duplicates(subset=['open_time'], keep='last')

        numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                       'quote_asset_volume', 'taker_buy_base_asset_volume',
                       'taker_buy_quote_asset_volume']
        for col in numeric_cols:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

        df_copy = df_copy.dropna(subset=['close', 'volume'])

        print(f"Data prepared: {len(df_copy)} candles from {df_copy['open_time'].min()} to {df_copy['open_time'].max()}")
        return df_copy

    def get_latest_candles(self, df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
        """Extract last N completed candles."""
        if len(df) < n:
            print(f"Warning: Requested {n} candles but only {len(df)} available")
            return df.copy()
        return df.tail(n).reset_index(drop=True)

    def validate_data_integrity(self, df: pd.DataFrame) -> bool:
        """Validate data integrity and consistency."""
        required_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"Error: Missing columns {missing_cols}")
            return False

        if len(df) == 0:
            print("Error: DataFrame is empty")
            return False

        if (df['high'] < df['low']).any():
            print("Warning: Found high < low in some candles")

        if (df['close'] < df['low']).any() or (df['close'] > df['high']).any():
            print("Warning: Found close outside high-low range")

        print(f"Data validation passed: {len(df)} rows")
        return True


class DataManager:
    """High-level manager for data operations across multiple symbols."""

    def __init__(self):
        self.processors = {}
        self.loaded_data = {}

    def load_symbol_data(self, symbol: str, timeframe: str = DEFAULT_TIMEFRAME) -> pd.DataFrame:
        """Load data for specific symbol."""
        key = f"{symbol}_{timeframe}"
        if key in self.loaded_data:
            return self.loaded_data[key]

        processor = DataProcessor(symbol, timeframe)
        df = processor.load_data()
        df = processor.prepare_data(df)
        processor.validate_data_integrity(df)

        self.loaded_data[key] = df
        return df

    def get_cached_symbols(self) -> list:
        """List all cached data files."""
        if not self.cache_dir.exists():
            return []
        return [f.stem for f in self.cache_dir.glob('*.parquet')]

    @property
    def cache_dir(self) -> Path:
        return DATA_DIR