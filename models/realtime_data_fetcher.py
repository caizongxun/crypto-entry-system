# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import requests
import yfinance as yf
from binance.client import Client
from binance.exceptions import BinanceClientException
import logging

logger = logging.getLogger(__name__)

class RealtimeDataFetcher:
    """Fetch completed candles from Binance or yfinance for realtime trading analysis"""

    BINANCE_TIMEFRAMES = {
        '1m': Client.KLINE_INTERVAL_1MINUTE,
        '5m': Client.KLINE_INTERVAL_5MINUTE,
        '15m': Client.KLINE_INTERVAL_15MINUTE,
        '30m': Client.KLINE_INTERVAL_30MINUTE,
        '1h': Client.KLINE_INTERVAL_1HOUR,
        '4h': Client.KLINE_INTERVAL_4HOUR,
        '1d': Client.KLINE_INTERVAL_1DAY,
    }

    def __init__(self, data_source: str = 'binance', api_key: Optional[str] = None, 
                 api_secret: Optional[str] = None):
        self.data_source = data_source
        self.binance_client = None

        if data_source == 'binance':
            try:
                self.binance_client = Client(api_key or '', api_secret or '')
                self.binance_client.ping()
                logger.info("Binance client connected successfully")
            except Exception as e:
                logger.warning(f"Binance client connection failed: {str(e)}. Using yfinance instead.")
                self.data_source = 'yfinance'

    def get_completed_candles(self, symbol: str, timeframe: str, limit: int = 50) -> pd.DataFrame:
        """
        Fetch completed candles (excluding current forming candle)
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'BTC-USD')
            timeframe: Timeframe (e.g., '15m', '1h', '1d')
            limit: Number of candles to fetch (minimum 50)
        
        Returns:
            DataFrame with columns: open_time, close_time, open, high, low, close, volume
        """
        limit = max(limit, 50)

        if self.data_source == 'binance' and self.binance_client:
            return self._fetch_binance_completed_candles(symbol, timeframe, limit)
        else:
            return self._fetch_yfinance_completed_candles(symbol, timeframe, limit)

    def _fetch_binance_completed_candles(self, symbol: str, timeframe: str, 
                                        limit: int) -> pd.DataFrame:
        """Fetch completed candles from Binance (excludes current forming candle)"""
        try:
            if timeframe not in self.BINANCE_TIMEFRAMES:
                raise ValueError(f"Unsupported timeframe: {timeframe}")

            binance_interval = self.BINANCE_TIMEFRAMES[timeframe]
            klines = self.binance_client.get_klines(
                symbol=symbol,
                interval=binance_interval,
                limit=limit + 1
            )

            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)

            numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                           'quote_asset_volume', 'taker_buy_base_asset_volume',
                           'taker_buy_quote_asset_volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df[['open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume',
                     'quote_asset_volume', 'taker_buy_base_asset_volume',
                     'taker_buy_quote_asset_volume']]

            df = df[:-1]

            df = df.sort_values('open_time').reset_index(drop=True)

            logger.info(f"Fetched {len(df)} completed {timeframe} candles from Binance for {symbol}")
            return df

        except BinanceClientException as e:
            logger.error(f"Binance API error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error fetching Binance data: {str(e)}")
            raise

    def _fetch_yfinance_completed_candles(self, symbol: str, timeframe: str,
                                         limit: int) -> pd.DataFrame:
        """Fetch completed candles from yfinance (excludes current forming candle)"""
        try:
            symbol_yf = symbol.replace('USDT', '-USD')

            period_map = {
                '1m': '7d',
                '5m': '60d',
                '15m': '60d',
                '30m': '60d',
                '1h': '730d',
                '4h': '730d',
                '1d': '5y'
            }

            interval_map = {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '1h': '1h',
                '4h': '4h',
                '1d': '1d'
            }

            if timeframe not in interval_map:
                raise ValueError(f"Unsupported timeframe: {timeframe}")

            period = period_map.get(timeframe, '60d')
            interval = interval_map[timeframe]

            data = yf.download(symbol_yf, period=period, interval=interval, 
                              progress=False, prepost=False)

            if data.empty:
                raise ValueError(f"No data found for {symbol_yf}")

            data = data.reset_index()
            data.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'adj_close']

            data['close_time'] = data['open_time'] + pd.to_timedelta(
                self._get_timeframe_minutes(timeframe), unit='min'
            )

            data = data[['open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume']]

            data = data.iloc[:-1]

            data = data.tail(limit).reset_index(drop=True)

            logger.info(f"Fetched {len(data)} completed {timeframe} candles from yfinance for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Error fetching yfinance data: {str(e)}")
            raise

    def _get_timeframe_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        mapping = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        return mapping.get(timeframe, 15)

    def get_current_market_price(self, symbol: str) -> Optional[float]:
        """Get current market price"""
        try:
            if self.data_source == 'binance' and self.binance_client:
                ticker = self.binance_client.get_symbol_ticker(symbol=symbol)
                return float(ticker['price'])
            else:
                symbol_yf = symbol.replace('USDT', '-USD')
                data = yf.Ticker(symbol_yf)
                return data.info.get('currentPrice')
        except Exception as e:
            logger.error(f"Error fetching current price: {str(e)}")
            return None

    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol exists"""
        try:
            if self.data_source == 'binance' and self.binance_client:
                self.binance_client.get_symbol_ticker(symbol=symbol)
                return True
            else:
                symbol_yf = symbol.replace('USDT', '-USD')
                data = yf.Ticker(symbol_yf)
                return data.info.get('regularMarketPrice') is not None
        except:
            return False

    def get_multiple_timeframes(self, symbol: str, timeframes: List[str],
                               limit: int = 50) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple timeframes"""
        result = {}
        for tf in timeframes:
            try:
                result[tf] = self.get_completed_candles(symbol, tf, limit)
            except Exception as e:
                logger.error(f"Error fetching {tf} data: {str(e)}")
                result[tf] = pd.DataFrame()
        return result

    def get_candle_status(self, symbol: str, timeframe: str) -> Dict:
        """Get status of current candle (forming or completed)"""
        try:
            if self.data_source == 'binance' and self.binance_client:
                return self._get_binance_candle_status(symbol, timeframe)
            else:
                return self._get_yfinance_candle_status(symbol, timeframe)
        except Exception as e:
            logger.error(f"Error getting candle status: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def _get_binance_candle_status(self, symbol: str, timeframe: str) -> Dict:
        """Get Binance candle formation status"""
        try:
            binance_interval = self.BINANCE_TIMEFRAMES.get(timeframe)
            if not binance_interval:
                return {'status': 'error', 'message': f'Unsupported timeframe: {timeframe}'}

            klines = self.binance_client.get_klines(
                symbol=symbol,
                interval=binance_interval,
                limit=2
            )

            current_candle = klines[-1]
            current_open_time = datetime.fromtimestamp(current_candle[0] / 1000, tz=pd.Timestamp.now().tz)
            current_close_time = datetime.fromtimestamp(current_candle[6] / 1000, tz=pd.Timestamp.now().tz)
            now = datetime.now(tz=pd.Timestamp.now().tz)

            is_completed = now > current_close_time
            progress_pct = int(((now - current_open_time).total_seconds() / 
                               (current_close_time - current_open_time).total_seconds()) * 100)
            progress_pct = min(progress_pct, 99) if not is_completed else 100

            return {
                'status': 'completed' if is_completed else 'forming',
                'current_open_time': current_open_time.isoformat(),
                'current_close_time': current_close_time.isoformat(),
                'progress_percentage': progress_pct,
                'current_price': float(current_candle[4]),
                'current_high': float(current_candle[2]),
                'current_low': float(current_candle[3])
            }

        except Exception as e:
            logger.error(f"Error getting Binance candle status: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def _get_yfinance_candle_status(self, symbol: str, timeframe: str) -> Dict:
        """Get yfinance candle formation status"""
        try:
            symbol_yf = symbol.replace('USDT', '-USD')
            minutes = self._get_timeframe_minutes(timeframe)

            data = yf.download(symbol_yf, period='1d', interval=timeframe,
                              progress=False, prepost=False)

            if data.empty:
                return {'status': 'error', 'message': 'No data available'}

            last_row = data.iloc[-1]
            now = datetime.now()
            last_time = data.index[-1]

            is_completed = now > (last_time + timedelta(minutes=minutes))
            progress_pct = int(((now - last_time).total_seconds() / (minutes * 60)) * 100)
            progress_pct = min(progress_pct, 99) if not is_completed else 100

            return {
                'status': 'completed' if is_completed else 'forming',
                'current_open_time': last_time.isoformat(),
                'current_close_time': (last_time + timedelta(minutes=minutes)).isoformat(),
                'progress_percentage': progress_pct,
                'current_price': float(last_row['Close']),
                'current_high': float(last_row['High']),
                'current_low': float(last_row['Low'])
            }

        except Exception as e:
            logger.error(f"Error getting yfinance candle status: {str(e)}")
            return {'status': 'error', 'message': str(e)}
