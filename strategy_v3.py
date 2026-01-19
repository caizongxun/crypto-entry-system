import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from loguru import logger
import os
from typing import Optional, Dict, List
from huggingface_hub import hf_hub_download


@dataclass
class DataConfig:
    hf_repo: str = 'zongowo111/v2-crypto-ohlcv-data'
    cache_dir: str = './data_cache'


@dataclass
class StrategyConfig:
    lookback_window: int = 20
    atr_multiplier: float = 1.5
    profit_target_ratio: float = 1.5
    min_trend_candles: int = 3
    volume_sma_period: int = 20
    
    data: DataConfig = field(default_factory=DataConfig)
    model_save_dir: str = './models/v3'
    results_save_dir: str = './results/v3'
    verbose: bool = False
    
    @staticmethod
    def get_default() -> 'StrategyConfig':
        return StrategyConfig()


class DataLoader:
    def __init__(self, hf_repo: str, cache_dir: str, verbose: bool = False):
        self.hf_repo = hf_repo
        self.cache_dir = cache_dir
        self.verbose = verbose
    
    def load_data(self, symbol: str, timeframe: str, cache: bool = True) -> pd.DataFrame:
        """Load OHLCV data from HuggingFace dataset."""
        base = symbol.replace('USDT', '')
        filename = f'{base}_{timeframe}.parquet'
        path_in_repo = f'klines/{symbol}/{filename}'
        
        logger.info(f'Downloading {symbol} {timeframe} data from HuggingFace...')
        
        try:
            local_path = hf_hub_download(
                repo_id=self.hf_repo,
                filename=path_in_repo,
                repo_type='dataset',
                cache_dir=self.cache_dir
            )
            
            df = pd.read_parquet(local_path)
            
            if 'open_time' in df.columns:
                df = df.rename(columns={'open_time': 'datetime'})
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f'Failed to load {symbol} {timeframe}: {str(e)}')
            raise
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate OHLCV data structure."""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(f'Missing required columns. Found: {df.columns.tolist()}')
            return False
        
        if len(df) < 100:
            logger.error(f'Insufficient data: {len(df)} rows')
            return False
        
        if df[['open', 'high', 'low', 'close', 'volume']].isnull().any().any():
            logger.error('Data contains null values')
            return False
        
        return True


class FeatureEngineer:
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.lookback = config.lookback_window
        self.atr_mult = config.atr_multiplier
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features optimized for reversal prediction."""
        logger.info('Generating features...')
        
        df_feat = df.copy()
        
        logger.info('Calculating price action features...')
        df_feat = self._add_price_action_features(df_feat)
        
        logger.info('Calculating volatility features...')
        df_feat = self._add_volatility_features(df_feat)
        
        logger.info('Calculating volume features...')
        df_feat = self._add_volume_features(df_feat)
        
        logger.info('Calculating momentum features...')
        df_feat = self._add_momentum_features(df_feat)
        
        logger.info('Calculating structure features...')
        df_feat = self._add_structure_features(df_feat)
        
        feature_cols = [col for col in df_feat.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        logger.info(f'Generated {len(feature_cols)} features')
        
        return df_feat
    
    def _add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price action confluence features."""
        close = df['close']
        high = df['high']
        low = df['low']
        
        df['hl_range'] = (high - low) / close
        df['cl_ratio'] = (close - low) / (high - low + 1e-10)
        df['body_ratio'] = abs(close - df['open']) / (high - low + 1e-10)
        df['upper_wick'] = (high - np.maximum(close, df['open'])) / (high - low + 1e-10)
        df['lower_wick'] = (np.minimum(close, df['open']) - low) / (high - low + 1e-10)
        
        df['is_bullish'] = (close > df['open']).astype(int)
        df['is_bearish'] = (close < df['open']).astype(int)
        
        for period in [5, 10, 20]:
            df[f'price_above_sma{period}'] = (close > close.rolling(period).mean()).astype(int)
            df[f'price_below_sma{period}'] = (close < close.rolling(period).mean()).astype(int)
        
        df['consecutive_up'] = (close > df['open']).rolling(window=5).sum()
        df['consecutive_down'] = (close < df['open']).rolling(window=5).sum()
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility and ATR features for reversal detection."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        for period in [14, 20, 30]:
            atr = tr.rolling(period).mean()
            df[f'atr{period}'] = atr
            df[f'volatility_ratio{period}'] = (high - low) / atr
        
        df['historical_volatility_20'] = close.pct_change().rolling(20).std()
        
        df['current_range'] = (high - low) / close
        df['avg_range_20'] = (high - low).rolling(20).mean() / close
        df['volatility_expansion'] = df['current_range'] / (df['avg_range_20'] + 1e-10)
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume profile and liquidity features."""
        volume = df['volume']
        close = df['close']
        
        df['volume_sma_20'] = volume.rolling(20).mean()
        df['volume_ratio'] = volume / (df['volume_sma_20'] + 1e-10)
        
        df['on_balance_volume'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        df['obv_sma_20'] = df['on_balance_volume'].rolling(20).mean()
        df['obv_momentum'] = df['on_balance_volume'] - df['obv_sma_20']
        
        df['volume_trend'] = volume.rolling(5).mean().diff()
        
        volume_open = df['volume'] * ((close - df['open']) / close)
        df['buy_volume'] = volume_open.clip(lower=0)
        df['sell_volume'] = (-volume_open).clip(lower=0)
        df['buy_sell_ratio'] = (df['buy_volume'] + 1) / (df['sell_volume'] + 1)
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum and rate of change features."""
        close = df['close']
        
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = close.pct_change(period)
            df[f'momentum_{period}'] = close.diff(period)
        
        for period in [9, 21]:
            df[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
        
        df['macd_line'] = df['ema_12'] = close.ewm(span=12, adjust=False).mean()
        df['macd_signal'] = df['ema_26'] = close.ewm(span=26, adjust=False).mean()
        df['macd_line'] = df['ema_12'] - df['ema_26']
        df['macd_histogram'] = df['macd_line'] - df['macd_line'].rolling(9).mean()
        
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _add_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Support/Resistance and structure features."""
        low = df['low']
        high = df['high']
        
        for period in [20, 50]:
            df[f'support_{period}'] = low.rolling(period).min()
            df[f'resistance_{period}'] = high.rolling(period).max()
            df[f'distance_to_support_{period}'] = (df['close'] - df[f'support_{period}']) / df['close']
            df[f'distance_to_resistance_{period}'] = (df[f'resistance_{period}'] - df['close']) / df['close']
        
        df['highest_20'] = high.rolling(20).max()
        df['lowest_20'] = low.rolling(20).min()
        df['range_position_20'] = (df['close'] - df['lowest_20']) / (df['highest_20'] - df['lowest_20'] + 1e-10)
        
        return df
    
    def add_atr_for_targets(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add ATR calculation for target definition."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        df['atr_14'] = tr.rolling(period).mean()
        return df


def create_reversal_target(df: pd.DataFrame, lookback: int = 20, atr_mult: float = 1.5, 
                           profit_target_ratio: float = 1.5, forward_window: int = 100) -> np.ndarray:
    """
    Create binary reversal target with proper labeling.
    
    Rules:
    - Use past lookback candles (t-lookback to t-1) to predict if t-th candle is reversal
    - Reversal success: Entry at t close + stop loss at 1.5 ATR, TP at 1.5:1 ratio
    - Labels t+1, t+2, ... as HOLD until TP/SL hit
    
    Returns:
        - Array with labels for each candle
        - 1: Can open reversal trade at this candle
        - 0: Cannot open (either no setup or in position/HOLD)
    """
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    atr = df['atr_14'].values
    
    targets = np.zeros(len(df), dtype=int)
    in_position = False
    position_entry = None
    position_stop = None
    position_tp = None
    
    for i in range(lookback, len(df) - forward_window):
        if in_position:
            current_high = high[i]
            current_low = low[i]
            
            if current_low <= position_stop or current_high >= position_tp:
                in_position = False
                position_entry = None
            else:
                targets[i] = 0
            continue
        
        entry_price = close[i]
        sl_price = entry_price - (atr[i] * atr_mult)
        tp_price = entry_price + (atr[i] * atr_mult * profit_target_ratio)
        
        future_high = high[i+1:i+1+forward_window].max()
        future_low = low[i+1:i+1+forward_window].min()
        
        if future_high >= tp_price or future_low <= sl_price:
            found_tp_first = False
            for j in range(i+1, i+1+forward_window):
                if high[j] >= tp_price:
                    found_tp_first = True
                    break
                if low[j] <= sl_price:
                    break
            
            if found_tp_first:
                targets[i] = 1
                in_position = True
                position_entry = entry_price
                position_stop = sl_price
                position_tp = tp_price
    
    return targets
