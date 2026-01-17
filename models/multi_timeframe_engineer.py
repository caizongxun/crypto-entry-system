import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from models.data_processor import DataProcessor
from models.feature_engineer import FeatureEngineer
from pathlib import Path


class MultiTimeframeEngineer:
    """Engineer features from immediate higher timeframe only to reduce noise."""

    TIMEFRAME_MINUTES = {
        '15m': 15,
        '1h': 60,
        '4h': 240,
        '1d': 1440
    }

    TIMEFRAME_HIERARCHY = {
        '15m': '1h',
        '1h': '4h',
        '4h': '1d',
        '1d': None
    }

    def __init__(self, symbol: str, base_timeframe: str = '15m'):
        self.symbol = symbol
        self.base_timeframe = base_timeframe
        self.base_minutes = self.TIMEFRAME_MINUTES[base_timeframe]
        self.feature_engineer = FeatureEngineer()
        self.data_processor = DataProcessor(symbol, base_timeframe)
        self.higher_timeframe = self._get_immediate_higher_timeframe()

    def _get_immediate_higher_timeframe(self) -> str:
        """Get immediate higher timeframe only."""
        next_tf = self.TIMEFRAME_HIERARCHY.get(self.base_timeframe)
        
        if next_tf is None:
            return None
        
        cache_dir = Path(__file__).parent / 'cache' / 'data'
        cache_file = cache_dir / f"{self.symbol}_{next_tf}.parquet"
        
        if cache_file.exists():
            return next_tf
        
        return None

    def load_timeframe_data(self, timeframe: str) -> pd.DataFrame:
        """Load data for specific timeframe."""
        try:
            processor = DataProcessor(self.symbol, timeframe)
            df = processor.load_data()
            if df is None or len(df) == 0:
                print(f"Warning: No data available for {timeframe}")
                return None
            df = processor.prepare_data(df)
            return df
        except Exception as e:
            print(f"Warning: Failed to load {timeframe} data: {str(e)}")
            return None

    def align_timeframes(self, base_df: pd.DataFrame, higher_df: pd.DataFrame,
                        base_tf: str, higher_tf: str) -> pd.DataFrame:
        """Align higher timeframe data to base timeframe using forward fill."""
        try:
            if higher_df is None or len(higher_df) == 0:
                return None
            
            base_minutes = self.TIMEFRAME_MINUTES[base_tf]
            higher_minutes = self.TIMEFRAME_MINUTES[higher_tf]
            
            if higher_minutes <= base_minutes:
                print(f"Warning: {higher_tf} is not higher than {base_tf}")
                return None
            
            ratio = higher_minutes // base_minutes
            aligned_data = pd.DataFrame(index=base_df.index)
            
            numeric_cols = higher_df.select_dtypes(include=[np.number]).columns.tolist()
            
            for col in numeric_cols:
                try:
                    reindexed = higher_df[col].reindex(base_df.index, method='ffill')
                    aligned_data[col] = reindexed
                except Exception as e:
                    print(f"Warning: Failed to align column {col}: {str(e)}")
                    continue
            
            if len(aligned_data.columns) == 0:
                return None
            
            aligned_data = aligned_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            return aligned_data
        except Exception as e:
            print(f"Warning: Timeframe alignment failed for {higher_tf}: {str(e)}")
            return None

    def engineer_higher_timeframe_features(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from immediate higher timeframe only."""
        df_features = base_df.copy()
        
        if self.higher_timeframe is None:
            print(f"  No higher timeframe data available for {self.base_timeframe}")
            return df_features
        
        try:
            print(f"  Processing {self.higher_timeframe} timeframe...")
            higher_df = self.load_timeframe_data(self.higher_timeframe)
            
            if higher_df is None or len(higher_df) == 0:
                print(f"  Skipping {self.higher_timeframe}: No data")
                return df_features
            
            higher_features = self.feature_engineer.engineer_features(higher_df)
            
            if higher_features is None or len(higher_features) == 0:
                print(f"  Skipping {self.higher_timeframe}: Feature engineering failed")
                return df_features

            aligned_features = self.align_timeframes(
                base_df,
                higher_features,
                self.base_timeframe,
                self.higher_timeframe
            )
            
            if aligned_features is None or len(aligned_features.columns) == 0:
                print(f"  Skipping {self.higher_timeframe}: Alignment failed")
                return df_features

            added_count = 0
            for col in aligned_features.columns:
                if col not in ['open', 'high', 'low', 'close', 'volume', 'open_time', 'close_time']:
                    new_col_name = f"{col}_{self.higher_timeframe}"
                    if new_col_name not in df_features.columns:
                        df_features[new_col_name] = aligned_features[col]
                        added_count += 1

            print(f"  Added {added_count} features from {self.higher_timeframe}")
            return df_features
            
        except Exception as e:
            print(f"  Failed to process {self.higher_timeframe}: {str(e)}")
            return df_features

    def calculate_timeframe_confirmation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI confirmation from higher timeframe."""
        df_result = df.copy()

        if 'rsi' not in df_result.columns:
            df_result['timeframe_confirmation'] = 0.5
            return df_result

        if self.higher_timeframe is None:
            df_result['timeframe_confirmation'] = 0.5
            return df_result

        rsi_higher_col = f"rsi_{self.higher_timeframe}"
        
        if rsi_higher_col not in df_result.columns:
            df_result['timeframe_confirmation'] = 0.5
            return df_result

        try:
            base_rsi = df_result['rsi']
            higher_rsi = df_result[rsi_higher_col]
            
            base_bullish = (base_rsi > 50).astype(int)
            higher_bullish = (higher_rsi > 50).astype(int)
            
            df_result['timeframe_confirmation'] = (base_bullish + higher_bullish) / 2.0
            df_result['timeframe_confirmation'] = df_result['timeframe_confirmation'].fillna(0.5)
            
            return df_result
        except Exception as e:
            print(f"Warning: Timeframe confirmation failed: {str(e)}")
            df_result['timeframe_confirmation'] = 0.5
            return df_result

    def calculate_trend_alignment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend alignment between base and higher timeframe."""
        df_result = df.copy()

        if 'sma_fast' not in df_result.columns or 'sma_slow' not in df_result.columns:
            df_result['trend_alignment'] = 0.5
            return df_result

        if self.higher_timeframe is None:
            df_result['trend_alignment'] = 0.5
            return df_result

        sma_fast_higher_col = f"sma_fast_{self.higher_timeframe}"
        sma_slow_higher_col = f"sma_slow_{self.higher_timeframe}"
        
        if sma_fast_higher_col not in df_result.columns or sma_slow_higher_col not in df_result.columns:
            df_result['trend_alignment'] = 0.5
            return df_result

        try:
            base_trend = (df_result['sma_fast'] > df_result['sma_slow']).astype(int)
            higher_trend = (df_result[sma_fast_higher_col] > df_result[sma_slow_higher_col]).astype(int)
            
            df_result['trend_alignment'] = (base_trend + higher_trend) / 2.0
            df_result['trend_alignment'] = df_result['trend_alignment'].fillna(0.5)
            
            return df_result
        except Exception as e:
            print(f"Warning: Trend alignment failed: {str(e)}")
            df_result['trend_alignment'] = 0.5
            return df_result

    def calculate_volatility_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate relative volatility context."""
        df_result = df.copy()

        if 'volatility' not in df_result.columns:
            df_result['high_volatility_context'] = 0
            return df_result

        if self.higher_timeframe is None:
            df_result['high_volatility_context'] = 0
            return df_result

        vol_higher_col = f"volatility_{self.higher_timeframe}"
        
        if vol_higher_col not in df_result.columns:
            df_result['high_volatility_context'] = 0
            return df_result

        try:
            relative_vol = df_result['volatility'] / (df_result[vol_higher_col] + 1e-6)
            relative_vol = np.clip(relative_vol, 0, 10)
            df_result['high_volatility_context'] = relative_vol
            df_result['high_volatility_context'] = df_result['high_volatility_context'].fillna(0)
            
            return df_result
        except Exception as e:
            print(f"Warning: Volatility context failed: {str(e)}")
            df_result['high_volatility_context'] = 0
            return df_result

    def calculate_momentum_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum divergence between base and higher timeframe."""
        df_result = df.copy()

        if 'momentum' not in df_result.columns:
            df_result['momentum_divergence_multi'] = 0
            return df_result

        if self.higher_timeframe is None:
            df_result['momentum_divergence_multi'] = 0
            return df_result

        momentum_higher_col = f"momentum_{self.higher_timeframe}"
        
        if momentum_higher_col not in df_result.columns:
            df_result['momentum_divergence_multi'] = 0
            return df_result

        try:
            base_momentum = df_result['momentum']
            higher_momentum = df_result[momentum_higher_col]
            
            divergence = abs(base_momentum - higher_momentum)
            df_result['momentum_divergence_multi'] = divergence
            df_result['momentum_divergence_multi'] = df_result['momentum_divergence_multi'].fillna(0)
            
            return df_result
        except Exception as e:
            print(f"Warning: Momentum divergence failed: {str(e)}")
            df_result['momentum_divergence_multi'] = 0
            return df_result

    def engineer_comprehensive_features(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer multi-timeframe features using only immediate higher timeframe."""
        print("Starting multi-timeframe feature engineering...")
        
        if base_df is None or len(base_df) == 0:
            print("Warning: Base dataframe is empty, skipping multi-timeframe features")
            return base_df

        print(f"  Base timeframe: {self.base_timeframe}")
        print(f"  Higher timeframe: {self.higher_timeframe if self.higher_timeframe else 'None'}")
        print(f"  Base data: {len(base_df)} rows")

        try:
            df_features = self.engineer_higher_timeframe_features(base_df)
        except Exception as e:
            print(f"Warning: Higher timeframe feature engineering failed: {str(e)}")
            df_features = base_df.copy()

        try:
            print("Calculating timeframe confirmation (RSI)...")
            df_features = self.calculate_timeframe_confirmation(df_features)
        except Exception as e:
            print(f"Warning: Timeframe confirmation failed: {str(e)}")

        try:
            print("Calculating trend alignment (SMA)...")
            df_features = self.calculate_trend_alignment(df_features)
        except Exception as e:
            print(f"Warning: Trend alignment failed: {str(e)}")

        try:
            print("Calculating volatility context...")
            df_features = self.calculate_volatility_context(df_features)
        except Exception as e:
            print(f"Warning: Volatility context failed: {str(e)}")

        try:
            print("Calculating momentum divergence...")
            df_features = self.calculate_momentum_divergence(df_features)
        except Exception as e:
            print(f"Warning: Momentum divergence failed: {str(e)}")

        for col in df_features.select_dtypes(include=[np.number]).columns:
            try:
                df_features[col] = df_features[col].fillna(0)
                df_features[col] = df_features[col].replace([np.inf, -np.inf], 0)
            except:
                continue

        initial_cols = len(base_df.columns)
        final_cols = len(df_features.columns)
        new_cols = final_cols - initial_cols
        
        print(f"Multi-timeframe engineering complete: {new_cols} features added (total: {final_cols})")
        return df_features
