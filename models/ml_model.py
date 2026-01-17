import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

from models.config import ML_CONFIG, MODEL_CONFIG, DEFAULT_SYMBOL, DEFAULT_TIMEFRAME
from models.data_processor import DataProcessor
from models.feature_engineer import FeatureEngineer
from models.signal_evaluator import SignalEvaluator


class CryptoEntryModel:
    """Machine learning model for cryptocurrency entry signal prediction."""

    def __init__(self, symbol: str = DEFAULT_SYMBOL, timeframe: str = DEFAULT_TIMEFRAME,
                 model_type: str = 'xgboost'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_type = model_type
        self.data_processor = DataProcessor(symbol, timeframe)
        self.feature_engineer = FeatureEngineer()
        self.signal_evaluator = SignalEvaluator()

        self.raw_data = None
        self.feature_data = None
        self.quality_scores = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.model = None
        self.model_path = Path(__file__).parent / f"cache/{symbol}_{timeframe}_{model_type}.joblib"

    def load_data(self) -> pd.DataFrame:
        """Load and prepare data."""
        print(f"Loading data for {self.symbol} ({self.timeframe})...")
        self.raw_data = self.data_processor.load_data()
        self.raw_data = self.data_processor.prepare_data(self.raw_data)
        self.data_processor.validate_data_integrity(self.raw_data)
        print(f"Data loaded: {len(self.raw_data)} candles")
        return self.raw_data

    def engineer_features(self) -> pd.DataFrame:
        """Engineer technical features."""
        if self.raw_data is None:
            self.load_data()

        print("Engineering features...")
        self.feature_data = self.feature_engineer.engineer_features(self.raw_data)
        return self.feature_data

    def prepare_training_data(self, target_variable: str = 'future_return') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for model training."""
        if self.feature_data is None:
            self.engineer_features()

        print("Preparing training data...")

        lookback = ML_CONFIG.get('lookback_period', 50)
        df = self.feature_data.copy()

        df['future_return'] = df['close'].shift(-1) / df['close'] - 1
        df['future_return_signal'] = (df['future_return'] > 0).astype(int)
        df['target_quality'] = pd.cut(df['close'].pct_change().rolling(20).mean() * 100,
                                      bins=[0, 0.5, 1, 2, 5],
                                      labels=[0, 1, 2, 3]).astype(float)

        feature_names = self.feature_engineer.get_feature_names()
        feature_cols = [col for col in feature_names if col in df.columns]

        X = df[feature_cols].dropna().values
        y = df[target_variable].loc[df[feature_cols].index].values

        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]

        print(f"Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y

    def train(self, epochs: int = None) -> Dict:
        """Train the ML model."""
        X, y = self.prepare_training_data()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=ML_CONFIG.get('test_size', 0.2),
            random_state=ML_CONFIG.get('random_state', 42)
        )

        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

        print(f"Training {self.model_type} model...")
        model_config = MODEL_CONFIG.get(self.model_type, {})

        if self.model_type == 'xgboost':
            self.model = XGBClassifier(**model_config)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(**model_config)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model.fit(self.X_train, self.y_train, verbose=0)

        train_score = self.model.score(self.X_train, self.y_train)
        test_score = self.model.score(self.X_test, self.y_test)

        print(f"Model training completed:")
        print(f"  Train accuracy: {train_score:.4f}")
        print(f"  Test accuracy: {test_score:.4f}")

        self.save_model()

        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'model_type': self.model_type,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
        }

    def evaluate_entries(self, lookback: int = 50) -> pd.DataFrame:
        """Evaluate entry signals for recent candles."""
        if self.feature_data is None:
            self.engineer_features()

        if self.model is None:
            self.load_model()
            if self.model is None:
                print("Warning: No model available. Training new model...")
                self.train()

        df = self.feature_data.copy()
        recent_df = df.tail(lookback).copy()

        feature_names = self.feature_engineer.get_feature_names()
        feature_cols = [col for col in feature_names if col in recent_df.columns]

        X_recent = recent_df[feature_cols].values
        X_scaled = self.scaler.transform(X_recent)

        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1] if hasattr(self.model, 'predict_proba') else predictions

        recent_df['ml_prediction'] = predictions
        recent_df['ml_probability'] = probabilities * 100

        quality_df = self.signal_evaluator.calculate_quality_score(recent_df)
        quality_df['combined_score'] = (
            quality_df['quality_score'] * 0.4 +
            quality_df['ml_probability'] * 0.6
        )

        print(f"Entry evaluation complete: {len(quality_df)} recent candles analyzed")
        return quality_df

    def save_model(self) -> None:
        """Save trained model to disk."""
        if self.model is None:
            print("No model to save")
            return

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'model_type': self.model_type,
        }, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self) -> bool:
        """Load trained model from disk."""
        if not self.model_path.exists():
            print(f"Model file not found: {self.model_path}")
            return False

        try:
            checkpoint = joblib.load(self.model_path)
            self.model = checkpoint['model']
            self.scaler = checkpoint['scaler']
            print(f"Model loaded from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def get_diagnostic_info(self) -> Dict:
        """Get comprehensive diagnostic information."""
        if self.feature_data is None:
            return {}

        latest = self.feature_data.iloc[-1]
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'latest_price': latest['close'],
            'timestamp': latest['open_time'],
            'technical_indicators': {
                'rsi': latest.get('rsi', None),
                'macd': latest.get('macd', None),
                'sma_fast': latest.get('sma_fast', None),
                'sma_slow': latest.get('sma_slow', None),
                'bb_upper': latest.get('bb_upper', None),
                'bb_lower': latest.get('bb_lower', None),
                'atr': latest.get('atr', None),
                'volatility': latest.get('volatility', None),
            },
            'model_info': {
                'type': self.model_type,
                'trained': self.model is not None,
                'path': str(self.model_path) if self.model_path else None,
            }
        }