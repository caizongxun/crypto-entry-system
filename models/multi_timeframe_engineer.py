import pandas as pd
import numpy as np
from typing import Dict, Tuple
from models.data_processor import DataProcessor
from models.feature_engineer import FeatureEngineer


class MultiTimeframeEngineer:
    """Engineer features from multiple timeframes with proper data alignment."""

    TIMEFRAME_MINUTES = {
        '15m': 15,
        '1h': 60,
        '4h': 240,
        '1d': 1440
    }

    def __init__(self, symbol: str, base_timeframe: str = '15m'):
        self.symbol = symbol
        self.base_timeframe = base_timeframe
        self.base_minutes = self.TIMEFRAME_MINUTES[base_timeframe]
        self.feature_engineer = FeatureEngineer()
        self.data_processor = DataProcessor(symbol, base_timeframe)

    def get_higher_timeframes(self) -> list:
        """Get list of higher timeframes relative to base timeframe."""
        higher_tf = []
        if self.base_timeframe == '15m':
            higher_tf = ['1h', '4h', '1d']
        elif self.base_timeframe == '1h':
            higher_tf = ['4h', '1d']
        elif self.base_timeframe == '4h':
            higher_tf = ['1d']
        return higher_tf

    def load_timeframe_data(self, timeframe: str) -> pd.DataFrame:
        """Load data for specific timeframe."""
        print(f"Loading {timeframe} data...")
        processor = DataProcessor(self.symbol, timeframe)
        df = processor.load_data()
        df = processor.prepare_data(df)
        return df

    def align_timeframes(self, base_df: pd.DataFrame, higher_df: pd.DataFrame,
                        base_tf: str, higher_tf: str) -> pd.DataFrame:
        """Align higher timeframe data to base timeframe using forward fill."""
        base_minutes = self.TIMEFRAME_MINUTES[base_tf]
        higher_minutes = self.TIMEFRAME_MINUTES[higher_tf]
        ratio = higher_minutes // base_minutes

        aligned_data = pd.DataFrame(index=base_df.index)

        for col in higher_df.columns:
            if col not in ['open_time', 'close_time']:
                resampled = higher_df[col].reindex(base_df.index, method='ffill')
                aligned_data[col] = resampled

        return aligned_data

    def engineer_higher_timeframe_features(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from multiple timeframes."""
        df_features = base_df.copy()
        higher_timeframes = self.get_higher_timeframes()

        for higher_tf in higher_timeframes:
            try:
                print(f"Processing {higher_tf} timeframe...")
                higher_df = self.load_timeframe_data(higher_tf)
                higher_features = self.feature_engineer.engineer_features(higher_df)

                aligned_features = self.align_timeframes(
                    base_df,
                    higher_features,
                    self.base_timeframe,
                    higher_tf
                )

                for col in aligned_features.columns:
                    if col not in ['open', 'high', 'low', 'close', 'volume', 'open_time', 'close_time']:
                        new_col_name = f"{col}_{higher_tf}"
                        df_features[new_col_name] = aligned_features[col]

                print(f"Added {len(aligned_features.columns)} features from {higher_tf}")
            except Exception as e:
                print(f"Warning: Failed to load {higher_tf} data: {str(e)}")
                continue

        print(f"Total features engineered: {len(df_features.columns)} columns")
        return df_features

    def calculate_multi_timeframe_signals(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate consensus signals across multiple timeframes."""
        df_signals = base_df.copy()
        higher_timeframes = self.get_higher_timeframes()

        if len(higher_timeframes) == 0:
            return df_signals

        for higher_tf in higher_timeframes:
            try:
                higher_df = self.load_timeframe_data(higher_tf)
                higher_features = self.feature_engineer.engineer_features(higher_df)

                aligned_features = self.align_timeframes(
                    base_df,
                    higher_features,
                    self.base_timeframe,
                    higher_tf
                )

                rsi_col = f"rsi_{higher_tf}"
                if 'rsi' in aligned_features.columns:
                    df_signals[rsi_col] = aligned_features['rsi']

                macd_col = f"macd_{higher_tf}"
                if 'macd' in aligned_features.columns:
                    df_signals[macd_col] = aligned_features['macd']

            except Exception as e:
                print(f"Warning: Failed to process signals for {higher_tf}: {str(e)}")
                continue

        return df_signals

    def calculate_timeframe_confirmation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate signal confirmation across timeframes."""
        df_result = df.copy()
        higher_timeframes = self.get_higher_timeframes()

        if 'rsi' not in df_result.columns:
            return df_result

        df_result['timeframe_confirmation'] = 0

        base_rsi = df_result['rsi']
        confirmation_count = 0

        if base_rsi > 50:
            df_result['base_is_bullish'] = 1
        else:
            df_result['base_is_bullish'] = 0

        for higher_tf in higher_timeframes:
            rsi_col = f"rsi_{higher_tf}"
            if rsi_col in df_result.columns:
                higher_rsi = df_result[rsi_col]
                is_bullish = (higher_rsi > 50).astype(int)
                df_result[f"is_bullish_{higher_tf}"] = is_bullish
                confirmation_count += 1

        if confirmation_count > 0:
            bullish_cols = [col for col in df_result.columns if col.startswith('is_bullish_')]
            if len(bullish_cols) > 0:
                df_result['timeframe_confirmation'] = df_result[bullish_cols].sum(axis=1) / len(bullish_cols)

        df_result['timeframe_confirmation'] = df_result['timeframe_confirmation'].fillna(0)
        return df_result

    def calculate_trend_alignment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate if trends are aligned across timeframes."""
        df_result = df.copy()
        higher_timeframes = self.get_higher_timeframes()

        trend_columns = []

        if 'sma_fast' in df_result.columns and 'sma_slow' in df_result.columns:
            df_result['base_trend'] = (
                (df_result['sma_fast'] > df_result['sma_slow']).astype(int)
            )
            trend_columns.append('base_trend')

        for higher_tf in higher_timeframes:
            sma_fast_col = f"sma_fast_{higher_tf}"
            sma_slow_col = f"sma_slow_{higher_tf}"

            if sma_fast_col in df_result.columns and sma_slow_col in df_result.columns:
                df_result[f"trend_{higher_tf}"] = (
                    (df_result[sma_fast_col] > df_result[sma_slow_col]).astype(int)
                )
                trend_columns.append(f"trend_{higher_tf}")

        if len(trend_columns) > 1:
            df_result['trend_alignment'] = df_result[trend_columns].sum(axis=1) / len(trend_columns)
        else:
            df_result['trend_alignment'] = df_result[trend_columns[0]] if trend_columns else 0

        df_result['trend_alignment'] = df_result['trend_alignment'].fillna(0.5)
        return df_result

    def calculate_volatility_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility context from higher timeframes."""
        df_result = df.copy()
        higher_timeframes = self.get_higher_timeframes()

        volatility_cols = []

        if 'volatility' in df_result.columns:
            df_result['base_volatility'] = df_result['volatility']
            volatility_cols.append('base_volatility')

        for higher_tf in higher_timeframes:
            vol_col = f"volatility_{higher_tf}"
            if vol_col in df_result.columns:
                df_result[f"rel_volatility_{higher_tf}"] = (
                    df_result['volatility'] / (df_result[vol_col] + 1e-6)
                )
                volatility_cols.append(f"rel_volatility_{higher_tf}")

        if len(volatility_cols) > 0:
            df_result['high_volatility_context'] = (
                df_result[[col for col in volatility_cols if col in df_result.columns]].mean(axis=1)
            )
        else:
            df_result['high_volatility_context'] = 0

        return df_result

    def calculate_momentum_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum divergence across timeframes."""
        df_result = df.copy()
        higher_timeframes = self.get_higher_timeframes()

        if 'momentum' not in df_result.columns:
            return df_result

        base_momentum = df_result['momentum']
        df_result['momentum_divergence_multi'] = 0

        divergence_count = 0
        for higher_tf in higher_timeframes:
            momentum_col = f"momentum_{higher_tf}"
            if momentum_col in df_result.columns:
                higher_momentum = df_result[momentum_col]
                divergence = abs(base_momentum - higher_momentum)
                df_result[f"divergence_{higher_tf}"] = divergence
                divergence_count += 1

        if divergence_count > 0:
            div_cols = [col for col in df_result.columns if col.startswith('divergence_')]
            df_result['momentum_divergence_multi'] = df_result[div_cols].mean(axis=1)
        else:
            df_result['momentum_divergence_multi'] = 0

        return df_result

    def engineer_comprehensive_features(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer all multi-timeframe features comprehensively."""
        print("Starting comprehensive multi-timeframe feature engineering...")

        df_features = self.engineer_higher_timeframe_features(base_df)

        print("Calculating timeframe confirmation...")
        df_features = self.calculate_timeframe_confirmation(df_features)

        print("Calculating trend alignment...")
        df_features = self.calculate_trend_alignment(df_features)

        print("Calculating volatility context...")
        df_features = self.calculate_volatility_context(df_features)

        print("Calculating momentum divergence...")
        df_features = self.calculate_momentum_divergence(df_features)

        for col in df_features.select_dtypes(include=[np.number]).columns:
            df_features[col] = df_features[col].fillna(0)
            df_features[col] = df_features[col].replace([np.inf, -np.inf], 0)

        print(f"Comprehensive feature engineering complete: {len(df_features.columns)} total features")
        return df_features
