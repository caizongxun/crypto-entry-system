import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from scipy import stats


class AdvancedFeatureEngineer:
    """Advanced feature engineering for Bollinger Bands bounce prediction.
    
    Implements four key high-signal features:
    1. Bounce Failure Memory: Historical bounce failure rate at this price level
    2. Volume Anomaly Detection: Abnormal volume spike during BB touch
    3. Reversal Strength: Speed and magnitude of price reversal
    4. Time Structure: Hourly and daily patterns affecting bounce probability
    
    Note: Requires BB indicators (touched_lower, touched_upper, bb_lower, bb_upper)
    to be pre-calculated. Will skip if not available.
    """

    def __init__(self, lookback_period: int = 252):
        """Initialize advanced feature engineer.
        
        Args:
            lookback_period: Number of historical candles to analyze
        """
        self.lookback_period = lookback_period
        self.bb_failure_cache = {}
        self.volume_baseline = {}

    def engineer_bounce_failure_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate bounce failure rate at each price level.
        
        This is the most important feature. Measures:
        - How many times this price level bounced successfully in the past
        - How many times it failed (continued down/up)
        - Success rate of bounces at similar volatility levels
        
        Result: High success rate locations get higher weight
        """
        df = df.copy()
        
        # Check if BB indicators exist
        if 'touched_lower' not in df.columns or 'touched_upper' not in df.columns:
            print("Warning: BB indicators not found, using neutral bounce failure memory")
            df['bounce_failure_memory'] = 0.5
            return df
        
        # Bin price into volatility-adjusted levels
        df['price_bin'] = pd.qcut(df['close'], q=20, duplicates='drop')
        
        # For each bin, calculate historical bounce success
        bounce_success_by_bin = {}
        for price_bin in df['price_bin'].unique():
            bin_data = df[df['price_bin'] == price_bin]
            
            if len(bin_data) < 5:
                bounce_success_by_bin[price_bin] = 0.5  # Default neutral
                continue
            
            # Count successful bounces (used BB touch + had future bounce)
            successful = (bin_data['touched_lower'] | bin_data['touched_upper']).sum()
            failed = (bin_data['touched_lower'] & (bin_data['low'].shift(-1) < bin_data['bb_lower'])).sum()
            
            if successful == 0:
                success_rate = 0.5
            else:
                success_rate = max(0, (successful - failed) / successful)
            
            bounce_success_by_bin[price_bin] = success_rate
        
        # Map back to dataframe
        df['bounce_failure_memory'] = df['price_bin'].map(bounce_success_by_bin).fillna(0.5)
        
        # Smooth the feature
        df['bounce_failure_memory'] = df['bounce_failure_memory'].rolling(
            window=5, center=True
        ).mean().fillna(0.5)
        
        return df.drop('price_bin', axis=1)

    def engineer_volume_anomaly(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Detect abnormal volume during BB touch events.
        
        Volume spikes during BB touch indicate institutional buying/selling,
        which strengthens the bounce probability.
        
        Features:
        - volume_zscore: How many std devs above average volume is the touch
        - volume_ratio: Ratio of touch volume to recent average
        - volume_momentum: Is volume accelerating or decelerating
        """
        df = df.copy()
        
        # Calculate rolling volume statistics
        df['volume_sma'] = df['volume'].rolling(window=window).mean()
        df['volume_std'] = df['volume'].rolling(window=window).std()
        
        # Z-score of volume
        df['volume_zscore'] = (df['volume'] - df['volume_sma']) / (df['volume_std'] + 1e-6)
        df['volume_zscore'] = df['volume_zscore'].clip(-5, 5)  # Clip outliers
        
        # Ratio of current volume to recent average
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-6)
        
        # Volume momentum (acceleration)
        df['volume_change'] = df['volume'].pct_change()
        df['volume_momentum'] = df['volume_change'].rolling(window=5).mean()
        
        # Combined volume anomaly score (0-1)
        # High volume + accelerating = strong signal
        df['volume_anomaly_score'] = 0.0
        
        # Normalize components to [0, 1] range
        zscore_norm = (df['volume_zscore'] - df['volume_zscore'].min()) / (
            df['volume_zscore'].max() - df['volume_zscore'].min() + 1e-6
        )
        ratio_norm = (df['volume_ratio'] - 1.0).clip(0, 5) / 5.0  # Cap at 5x
        momentum_norm = (df['volume_momentum'].clip(-0.5, 0.5) + 0.5) / 1.0
        
        # Weighted combination
        df['volume_anomaly_score'] = (
            zscore_norm * 0.4 +  # Z-score weight 40%
            ratio_norm * 0.4 +   # Ratio weight 40%
            momentum_norm * 0.2   # Momentum weight 20%
        )
        
        # Flag for extreme volume anomalies (signal boosters)
        df['volume_extreme'] = (df['volume_zscore'] > 2.0).astype(int)
        
        return df.drop(['volume_sma', 'volume_std', 'volume_change'], axis=1)

    def engineer_reversal_strength(self, df: pd.DataFrame, lookforward: int = 6) -> pd.DataFrame:
        """Measure the speed and magnitude of price reversal after BB touch.
        
        Fast reversals (bounce within 2-3 candles) are more reliable
        than slow reversals. This captures reversal kinetics.
        
        Features:
        - reversal_speed: How quickly price reverses after touch
        - reversal_magnitude: How strong is the reversal
        - reversal_acceleration: Is reversal speeding up or slowing down
        """
        df = df.copy()
        
        # Check if BB indicators exist
        if 'touched_lower' not in df.columns:
            print("Warning: BB indicators not found, using neutral reversal strength")
            df['reversal_speed'] = 0.5
            df['reversal_magnitude'] = 0.5
            df['reversal_acceleration'] = 0.5
            return df
        
        # For lower band touches
        df['reversal_speed_lower'] = 0.0
        for i in range(len(df) - lookforward):
            if df['touched_lower'].iloc[i]:
                # Find how many candles until bounce reaches 0.3%
                future_prices = df['low'].iloc[i:i+lookforward]
                bounce_threshold = df['close'].iloc[i] * 1.003
                
                for j, future_price in enumerate(future_prices):
                    if future_price >= bounce_threshold:
                        # Speed score: faster bounce = higher score
                        df['reversal_speed_lower'].iloc[i] = 1.0 - (j / lookforward)
                        break
        
        # For upper band touches
        df['reversal_speed_upper'] = 0.0
        for i in range(len(df) - lookforward):
            if df['touched_upper'].iloc[i]:
                future_prices = df['high'].iloc[i:i+lookforward]
                bounce_threshold = df['close'].iloc[i] * 0.997
                
                for j, future_price in enumerate(future_prices):
                    if future_price <= bounce_threshold:
                        df['reversal_speed_upper'].iloc[i] = 1.0 - (j / lookforward)
                        break
        
        # Combined reversal speed
        df['reversal_speed'] = df['reversal_speed_lower'] + df['reversal_speed_upper']
        df['reversal_speed'] = df['reversal_speed'].rolling(window=3, center=True).mean()
        
        # Reversal magnitude (how far does it go in lookforward period)
        df['future_high_5'] = df['high'].rolling(window=lookforward).max().shift(-lookforward)
        df['future_low_5'] = df['low'].rolling(window=lookforward).min().shift(-lookforward)
        
        df['reversal_magnitude'] = 0.0
        for i in range(len(df) - lookforward):
            if df['touched_lower'].iloc[i]:
                magnitude = (df['future_high_5'].iloc[i] - df['close'].iloc[i]) / df['close'].iloc[i]
                df['reversal_magnitude'].iloc[i] = min(magnitude, 0.05)  # Cap at 5%
            elif df['touched_upper'].iloc[i]:
                magnitude = (df['close'].iloc[i] - df['future_low_5'].iloc[i]) / df['close'].iloc[i]
                df['reversal_magnitude'].iloc[i] = min(magnitude, 0.05)
        
        # Normalize reversal magnitude to [0, 1]
        max_magnitude = df['reversal_magnitude'].max()
        if max_magnitude > 0:
            df['reversal_magnitude'] = df['reversal_magnitude'] / max_magnitude
        
        # Reversal acceleration (is it speeding up)
        df['reversal_acceleration'] = df['reversal_speed'].diff().clip(-0.1, 0.1)
        df['reversal_acceleration'] = (df['reversal_acceleration'] + 0.1) / 0.2  # Normalize to [0, 1]
        
        return df.drop(['future_high_5', 'future_low_5'], axis=1)

    def engineer_time_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode time-based patterns affecting bounce probability.
        
        Research shows different times have different bounce characteristics:
        - Asian trading hours: Higher volatility, faster reversals
        - US trading hours: More stability, slower but more reliable reversals
        - Low liquidity hours: Higher noise, less reliable
        - Day of week: Mondays often more volatile than Fridays
        
        Features:
        - time_of_day_score: Hour-based bounce probability
        - day_of_week_score: Day-based volatility pattern
        - session_type: Which trading session (Asian/European/US/Overnight)
        """
        df = df.copy()
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            try:
                df.index = pd.to_datetime(df.index)
            except:
                print("Warning: Could not convert index to datetime, using neutral time scores")
                df['time_of_day_score'] = 0.85
                df['day_of_week_score'] = 1.0
                df['session_type'] = 2
                df['time_quality'] = 0.85
                return df
        
        # Extract time components
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek  # 0=Monday, 6=Sunday
        
        # Time of day scoring (based on crypto market patterns)
        # Asian hours (0-8 UTC): Higher volatility, faster reversals (score 0.85)
        # European hours (8-16 UTC): Stable, reliable reversals (score 0.95)
        # US hours (16-24 UTC): Moderate volatility (score 0.80)
        time_scores = {
            0: 0.85, 1: 0.85, 2: 0.85, 3: 0.85, 4: 0.85, 5: 0.85, 6: 0.85, 7: 0.85,  # Asian
            8: 0.95, 9: 0.95, 10: 0.95, 11: 0.95, 12: 0.95, 13: 0.95, 14: 0.95, 15: 0.95,  # European
            16: 0.80, 17: 0.80, 18: 0.80, 19: 0.80, 20: 0.80, 21: 0.80, 22: 0.80, 23: 0.80,  # US
        }
        df['time_of_day_score'] = df['hour'].map(time_scores).fillna(0.75)
        
        # Day of week scoring
        # Monday-Thursday: Normal (1.0)
        # Friday: Lower volatility (0.90)
        # Saturday-Sunday: Higher noise (0.70)
        day_scores = {
            0: 1.0,  # Monday
            1: 1.0,  # Tuesday
            2: 1.0,  # Wednesday
            3: 1.0,  # Thursday
            4: 0.90,  # Friday
            5: 0.70,  # Saturday
            6: 0.70,  # Sunday
        }
        df['day_of_week_score'] = df['day_of_week'].map(day_scores)
        
        # Session type encoding
        session_map = {}
        for hour in range(0, 8):
            session_map[hour] = 1  # Asian
        for hour in range(8, 16):
            session_map[hour] = 2  # European
        for hour in range(16, 24):
            session_map[hour] = 3  # US
        
        df['session_type'] = df['hour'].map(session_map)
        
        # Combined time quality score
        df['time_quality'] = df['time_of_day_score'] * df['day_of_week_score']
        
        return df.drop(['hour', 'day_of_week'], axis=1)

    def engineer_combined_reversal_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine all advanced features into a single reversal quality score.
        
        Uses weighted ensemble:
        - Bounce Failure Memory: 40% weight (most predictive)
        - Volume Anomaly: 30% weight (strong confirmation)
        - Reversal Strength: 20% weight (validates pattern)
        - Time Quality: 10% weight (context)
        """
        df = df.copy()
        
        # Ensure all components exist
        required_cols = [
            'bounce_failure_memory',
            'volume_anomaly_score',
            'reversal_speed',
            'time_quality'
        ]
        
        for col in required_cols:
            if col not in df.columns:
                print(f"Warning: {col} not found, creating neutral values")
                df[col] = 0.5
        
        # Normalize each component to [0, 1] range
        for col in required_cols:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max > col_min:
                df[f'{col}_norm'] = (df[col] - col_min) / (col_max - col_min)
            else:
                df[f'{col}_norm'] = 0.5
        
        # Weighted combination (ensemble)
        df['advanced_reversal_score'] = (
            df['bounce_failure_memory_norm'] * 0.40 +
            df['volume_anomaly_score'] * 0.30 +  # Already normalized
            df['reversal_speed_norm'] * 0.20 +
            df['time_quality'] * 0.10  # Already normalized (0-1)
        )
        
        # Quality indicators
        df['is_strong_setup'] = (
            (df['bounce_failure_memory'] > 0.60) &
            (df['volume_anomaly_score'] > 0.50) &
            (df['reversal_speed'] > 0.40)
        ).astype(int)
        
        df['is_weak_setup'] = (
            (df['bounce_failure_memory'] < 0.40) |
            (df['volume_anomaly_score'] < 0.30) |
            (df['reversal_speed'] < 0.20)
        ).astype(int)
        
        # Clean up normalized columns
        norm_cols = [col for col in df.columns if col.endswith('_norm')]
        df = df.drop(norm_cols, axis=1)
        
        return df

    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer all advanced features in sequence.
        
        Args:
            df: DataFrame with OHLCV data and BB indicators
            
        Returns:
            DataFrame with additional advanced features
        """
        print("Applying advanced feature engineering...")
        
        # Check if BB indicators exist - if not, skip advanced features
        if 'touched_lower' not in df.columns or 'bb_lower' not in df.columns:
            print("  Note: BB indicators not yet available, deferring advanced features to training phase")
            # Create placeholder advanced features
            for feature_name in self.get_feature_list():
                df[feature_name] = 0.5
            return df
        
        # Step 1: Bounce failure memory
        print("  1. Engineering bounce failure memory...")
        df = self.engineer_bounce_failure_memory(df)
        
        # Step 2: Volume anomaly detection
        print("  2. Engineering volume anomaly detection...")
        df = self.engineer_volume_anomaly(df)
        
        # Step 3: Reversal strength
        print("  3. Engineering reversal strength...")
        df = self.engineer_reversal_strength(df)
        
        # Step 4: Time structure
        print("  4. Engineering time structure features...")
        df = self.engineer_time_structure(df)
        
        # Step 5: Combined signal
        print("  5. Creating combined reversal signal...")
        df = self.engineer_combined_reversal_signal(df)
        
        print("Advanced feature engineering complete")
        
        return df

    def get_feature_list(self) -> List[str]:
        """Return list of all advanced features generated."""
        return [
            'bounce_failure_memory',
            'volume_zscore',
            'volume_ratio',
            'volume_momentum',
            'volume_anomaly_score',
            'volume_extreme',
            'reversal_speed',
            'reversal_magnitude',
            'reversal_acceleration',
            'time_of_day_score',
            'day_of_week_score',
            'session_type',
            'time_quality',
            'advanced_reversal_score',
            'is_strong_setup',
            'is_weak_setup',
        ]
