import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import logging

from features.technical import TechnicalIndicators
from features.volatility import VolatilityRegime
from features.microstructure_features import MicrostructureFeatures
from models.regime_detector import RegimeDetector

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Preprocess OHLCV data and engineer features for model training."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.scaler = StandardScaler()
        self.feature_names = []
    
    @staticmethod
    def create_target_variable(
        df: pd.DataFrame,
        lookahead: int = 3,
        threshold: float = 0.002
    ) -> pd.Series:
        """Create binary target based on simple price movement.
        
        Simple approach:
        - If price goes UP by threshold% in next lookahead candles: 1
        - If price goes DOWN by threshold% in next lookahead candles: 0
        - Otherwise: NaN (exclude from training)
        
        Args:
            df: DataFrame with OHLCV data and indicators
            lookahead: Number of candles to look ahead
            threshold: Minimum price change threshold (0.002 = 0.2%)
        
        Returns:
            Binary series: 1 for UP, 0 for DOWN, NaN for neutral
        """
        future_high = df['high'].shift(-lookahead)
        future_low = df['low'].shift(-lookahead)
        current_close = df['close']
        
        # Calculate max up move and max down move
        up_move = (future_high - current_close) / current_close
        down_move = (current_close - future_low) / current_close
        
        target = pd.Series(np.nan, index=df.index, dtype='float64')
        
        # UP if high is reached before low (and exceeds threshold)
        up_condition = up_move >= threshold
        # DOWN if low is reached before high (and exceeds threshold)  
        down_condition = down_move >= threshold
        
        # If both exceed threshold, compare which happens first
        both_condition = up_condition & down_condition
        target[both_condition] = (up_move[both_condition] >= down_move[both_condition]).astype(float)
        
        # Only UP moves
        target[up_condition & ~both_condition] = 1.0
        # Only DOWN moves
        target[down_condition & ~both_condition] = 0.0
        
        return target
    
    @staticmethod
    def calculate_returns_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate return-based features."""
        df = df.copy()
        
        # Returns at different periods
        for period in [1, 5, 10, 20]:
            df[f'returns_{period}'] = df['close'].pct_change(period)
            df[f'log_returns_{period}'] = np.log(df['close'] / df['close'].shift(period))
        
        # Intra-candle returns
        df['intra_candle_return'] = (df['close'] - df['open']) / df['open']
        df['high_low_range'] = (df['high'] - df['low']) / df['low']
        
        return df
    
    @staticmethod
    def calculate_price_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price-based features."""
        df = df.copy()
        
        # Price position within recent range
        for period in [5, 10, 20, 50]:
            high_period = df['high'].rolling(window=period).max()
            low_period = df['low'].rolling(window=period).min()
            df[f'price_position_{period}'] = (df['close'] - low_period) / (high_period - low_period + 1e-10)
        
        # Distance from moving averages
        for ma in [5, 10, 20, 50]:
            df[f'dist_sma_{ma}'] = (df['close'] - df[f'sma_{ma}']) / df['close']
        
        return df
    
    @staticmethod
    def calculate_divergence_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum divergence features."""
        df = df.copy()
        
        # RSI divergence
        rsi_diff = df['rsi'].diff()
        price_diff = df['close'].diff()
        df['rsi_divergence'] = (np.sign(rsi_diff) != np.sign(price_diff)).astype(int)
        
        # MACD divergence
        macd_diff = df['macd'].diff()
        df['macd_divergence'] = (np.sign(macd_diff) != np.sign(price_diff)).astype(int)
        
        # Volume divergence
        vol_diff = df['volume'].diff()
        df['volume_divergence'] = (np.sign(vol_diff) != np.sign(price_diff)).astype(int)
        
        return df
    
    @staticmethod
    def select_features(
        df: pd.DataFrame,
        exclude_cols: List[str] = None,
        include_patterns: List[str] = None
    ) -> List[str]:
        """Select feature columns for model training.
        
        Args:
            df: Preprocessed DataFrame
            exclude_cols: Columns to exclude
            include_patterns: Patterns to include
        
        Returns:
            List of feature column names
        """
        exclude_cols = exclude_cols or [
            'open_time', 'close_time', 'open', 'high', 'low', 'close',
            'volume', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
            'ignore', 'resistance_level', 'support_level',
            'regime', 'ad_phase', 'target'  # Exclude categorical columns
        ]
        
        features = [col for col in df.columns if col not in exclude_cols]
        
        # Filter out any remaining string columns
        numeric_features = []
        for col in features:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                numeric_features.append(col)
        
        if include_patterns:
            numeric_features = [col for col in numeric_features if any(pattern in col for pattern in include_patterns)]
        
        return numeric_features
    
    def preprocess(
        self,
        df: pd.DataFrame,
        create_target: bool = True,
        lookahead: int = 3,
        normalize: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Complete preprocessing pipeline.
        
        Args:
            df: Raw OHLCV DataFrame
            create_target: Whether to create target variable
            lookahead: Lookahead period for target (3 candles = 45 minutes for 15m)
            normalize: Whether to normalize features
        
        Returns:
            (processed_features, target)
        """
        df = df.copy()
        logger.info(f"Starting preprocessing: {len(df)} rows")
        
        # Calculate indicators
        logger.info("Calculating technical indicators...")
        df = TechnicalIndicators.calculate_all_indicators(df, self.config)
        
        # Calculate volatility regime
        logger.info("Calculating volatility regime...")
        df = VolatilityRegime.calculate_regime_features(df, self.config)
        
        # Calculate microstructure features
        logger.info("Calculating microstructure features...")
        df = MicrostructureFeatures.calculate_all_microstructure_features(df, self.config)
        
        # Calculate regime detection features
        logger.info("Calculating regime detection features...")
        df = RegimeDetector.calculate_regime_features(df, self.config)
        
        # Calculate additional features
        logger.info("Calculating derived features...")
        df = self.calculate_returns_features(df)
        df = self.calculate_price_features(df)
        df = self.calculate_divergence_features(df)
        
        # Create target variable BEFORE removing NaN
        if create_target:
            logger.info(f"Creating target variable (lookahead={lookahead})...")
            target = self.create_target_variable(df, lookahead=lookahead, threshold=0.002)
            df['target'] = target
        
        # Remove rows with NaN values
        initial_rows = len(df)
        df = df.dropna()
        logger.info(f"Removed {initial_rows - len(df)} rows with NaN values")
        
        if len(df) == 0:
            logger.error("No valid data after preprocessing")
            return pd.DataFrame(), pd.Series()
        
        # Select features (numeric only)
        self.feature_names = self.select_features(df)
        logger.info(f"Selected {len(self.feature_names)} features")
        
        X = df[self.feature_names].copy()
        
        # Normalize features
        if normalize:
            logger.info("Normalizing features...")
            X_normalized = self.scaler.fit_transform(X)
            X = pd.DataFrame(X_normalized, columns=self.feature_names, index=X.index)
        
        y = df['target'] if 'target' in df.columns else None
        
        if y is not None:
            target_dist = y.value_counts()
            logger.info(f"Target distribution: {target_dist.to_dict()}")
            if len(target_dist) > 1:
                logger.info(f"Class balance: {(target_dist[1] / len(y) * 100):.1f}% UP, {(target_dist[0] / len(y) * 100):.1f}% DOWN")
        
        logger.info(f"Preprocessing complete: {len(X)} samples, {len(self.feature_names)} features")
        return X, y
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_split: float = 0.2,
        validation_split: float = 0.1,
        time_series: bool = True
    ) -> Tuple:
        """Split data into train, validation, and test sets.
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_split: Test set ratio
            validation_split: Validation set ratio (of training data)
            time_series: If True, use temporal split. If False, use random split.
        
        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if time_series:
            # Time series split - no data leakage
            n = len(X)
            test_size = int(n * test_split)
            val_size = int((n - test_size) * validation_split)
            
            X_train = X.iloc[:-test_size-val_size]
            X_val = X.iloc[-test_size-val_size:-test_size]
            X_test = X.iloc[-test_size:]
            
            y_train = y.iloc[:-test_size-val_size]
            y_val = y.iloc[-test_size-val_size:-test_size]
            y_test = y.iloc[-test_size:]
        else:
            # Random split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_split, random_state=42
            )
            
            val_size = validation_split / (1 - test_split)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size, random_state=42
            )
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info(f"Train target distribution: {y_train.value_counts().to_dict()}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
