import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from huggingface_hub import hf_hub_download
import logging

logger = logging.getLogger(__name__)

class KlinesDataLoader:
    """Load OHLCV data from HuggingFace dataset."""
    
    REPO_ID = 'zongowo111/v2-crypto-ohlcv-data'
    TIMEFRAMES = ['15m', '1h', '1d']
    SUPPORTED_SYMBOLS = [
        'AAVEUSDT', 'ADAUSDT', 'ALGOUSDT', 'ARBUSDT', 'ATOMUSDT', 'AVAXUSDT',
        'BALUSDT', 'BATUSDT', 'BCHUSDT', 'BNBUSDT', 'BTCUSDT', 'COMPUSDT',
        'CRVUSDT', 'DOGEUSDT', 'DOTUSDT', 'ENJUSDT', 'ENSUSDT', 'ETCUSDT',
        'ETHUSDT', 'FILUSDT', 'GALAUSDT', 'GRTUSDT', 'IMXUSDT', 'KAVAUSDT',
        'LINKUSDT', 'LTCUSDT', 'MANAUSDT', 'MATICUSDT', 'MKRUSDT', 'NEARUSDT',
        'OPUSDT', 'SANDUSDT', 'SNXUSDT', 'SOLUSDT', 'SPELLUSDT', 'UNIUSDT',
        'XRPUSDT', 'ZRXUSDT'
    ]
    
    def __init__(self, cache_dir: Optional[Path] = None, hf_token: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path('data/cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token = hf_token
    
    @staticmethod
    def _get_base_from_symbol(symbol: str) -> str:
        """Extract base currency from symbol (remove USDT)."""
        if symbol.endswith('USDT'):
            return symbol[:-4]
        return symbol
    
    def load_klines(
        self,
        symbol: str,
        timeframe: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Load K-line data for a symbol and timeframe.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: '15m', '1h', or '1d'
            use_cache: Whether to use cached files
        
        Returns:
            DataFrame with OHLCV data
        """
        if symbol not in self.SUPPORTED_SYMBOLS:
            raise ValueError(f"Unsupported symbol: {symbol}. Supported: {self.SUPPORTED_SYMBOLS}")
        
        if timeframe not in self.TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {self.TIMEFRAMES}")
        
        cache_path = self.cache_dir / f"{symbol}_{timeframe}.parquet"
        
        if use_cache and cache_path.exists():
            logger.info(f"Loading cached data: {cache_path}")
            return pd.read_parquet(cache_path)
        
        base = self._get_base_from_symbol(symbol)
        filename = f"{base}_{timeframe}.parquet"
        path_in_repo = f"klines/{symbol}/{filename}"
        
        logger.info(f"Downloading from HuggingFace: {path_in_repo}")
        
        local_path = hf_hub_download(
            repo_id=self.REPO_ID,
            filename=path_in_repo,
            repo_type='dataset',
            token=self.hf_token,
            cache_dir=str(self.cache_dir)
        )
        
        df = pd.read_parquet(local_path)
        
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
        
        df = df.sort_values('open_time').reset_index(drop=True)
        
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        logger.info(f"Cached to: {cache_path}")
        
        return df
    
    def load_multiple(
        self,
        symbol: str,
        timeframes: list = None,
        use_cache: bool = True
    ) -> dict:
        """Load multiple timeframes for same symbol.
        
        Args:
            symbol: Trading pair
            timeframes: List of timeframes. Default: ['15m', '1h', '1d']
            use_cache: Whether to use cached files
        
        Returns:
            Dict with timeframe as key, DataFrame as value
        """
        if timeframes is None:
            timeframes = self.TIMEFRAMES
        
        data = {}
        for tf in timeframes:
            try:
                data[tf] = self.load_klines(symbol, tf, use_cache)
                logger.info(f"Loaded {symbol} {tf}: {len(data[tf])} candles")
            except Exception as e:
                logger.error(f"Error loading {symbol} {tf}: {e}")
        
        return data
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data integrity.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            True if valid, False otherwise
        """
        required_cols = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
        ]
        
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing required columns. Expected: {required_cols}")
            return False
        
        if df.isnull().any().any():
            logger.error("Found null values in data")
            return False
        
        if len(df) < 100:
            logger.warning(f"Data has only {len(df)} candles, recommend at least 100")
            return False
        
        return True
