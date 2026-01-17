import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
from pathlib import Path

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

from models.config import ML_CONFIG, MODEL_CONFIG, DEFAULT_SYMBOL, DEFAULT_TIMEFRAME, TIMEFRAME_CONFIGS
from models.data_processor import DataProcessor
from models.feature_engineer import FeatureEngineer
from models.signal_evaluator import SignalEvaluator


class CryptoEntryModel:
    """ML model for BB channel bounce prediction with optimization levels."""

    def __init__(self, symbol: str = DEFAULT_SYMBOL, timeframe: str = DEFAULT_TIMEFRAME,
                 model_type: str = 'xgboost', optimization_level: str = 'balanced'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_type = model_type
        self.optimization_level = optimization_level
        
        self.data_processor = DataProcessor(symbol, timeframe)
        self.feature_engineer = FeatureEngineer()
        self.signal_evaluator = SignalEvaluator()

        self.timeframe_config = TIMEFRAME_CONFIGS.get(timeframe, TIMEFRAME_CONFIGS['15m'])
        self._apply_optimization_config()

        self.raw_data = None
        self.feature_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.model = None
        self.ensemble_models = None
        self.model_path = Path(__file__).parent / f"cache/{symbol}_{timeframe}_{model_type}.joblib"

    def _apply_optimization_config(self):
        """Apply optimization-specific configuration."""
        if self.optimization_level == 'conservative':
            self.use_smote = True
            self.smote_ratio = 0.7
            self.bounce_threshold_multiplier = 1.2
            self.use_ensemble = True
            self.hyperparams = {
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 200,
            }
        elif self.optimization_level == 'aggressive':
            self.use_smote = False
            self.bounce_threshold_multiplier = 0.8
            self.use_ensemble = False
            self.hyperparams = {
                'max_depth': 8,
                'learning_rate': 0.1,
                'n_estimators': 100,
            }
        else:
            self.use_smote = True
            self.smote_ratio = 0.5
            self.bounce_threshold_multiplier = 1.0
            self.use_ensemble = False
            self.hyperparams = {
                'max_depth': 7,
                'learning_rate': 0.08,
                'n_estimators': 150,
            }
        
        self.timeframe_config['bounce_threshold'] *= self.bounce_threshold_multiplier

    def load_data(self) -> pd.DataFrame:
        """Load and prepare data."""
        print(f"Loading data for {self.symbol} ({self.timeframe})...")
        self.raw_data = self.data_processor.load_data()
        self.raw_data = self.data_processor.prepare_data(self.raw_data)
        self.data_processor.validate_data_integrity(self.raw_data)
        print(f"Data loaded: {len(self.raw_data)} candles")
        return self.raw_data

    def engineer_features(self) -> pd.DataFrame:
        """Engineer technical features including BB indicators."""
        if self.raw_data is None:
            self.load_data()

        print("Engineering features...")
        self.feature_data = self.feature_engineer.engineer_features(self.raw_data)
        return self.feature_data

    def calculate_bb_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands metrics for bounce detection using timeframe config."""
        df_bb = df.copy()

        bb_period = self.timeframe_config['bb_period']
        bb_std = self.timeframe_config['bb_std']

        bb_basis = df_bb['close'].rolling(window=bb_period).mean()
        bb_dev = df_bb['close'].rolling(window=bb_period).std()
        bb_upper = bb_basis + (bb_dev * bb_std)
        bb_lower = bb_basis - (bb_dev * bb_std)

        df_bb['bb_basis'] = bb_basis
        df_bb['bb_upper'] = bb_upper
        df_bb['bb_lower'] = bb_lower
        df_bb['bb_width'] = bb_upper - bb_lower
        df_bb['bb_position'] = (df_bb['close'] - bb_lower) / (bb_upper - bb_lower)
        df_bb['basis_slope'] = bb_basis.diff()

        df_bb['touched_upper'] = (df_bb['close'] >= bb_upper * 0.98) & (df_bb['close'] <= bb_upper)
        df_bb['touched_lower'] = (df_bb['close'] <= bb_lower * 1.02) & (df_bb['close'] >= bb_lower)
        df_bb['broke_upper'] = (df_bb['close'] > bb_upper) & (df_bb['high'].shift(1) <= bb_upper.shift(1))
        df_bb['broke_lower'] = (df_bb['close'] < bb_lower) & (df_bb['low'].shift(1) >= bb_lower.shift(1))

        return df_bb

    def calculate_bounce_target(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate if BB touch/break resulted in effective bounce."""
        lookforward = self.timeframe_config['lookforward']
        bounce_threshold = self.timeframe_config['bounce_threshold']

        bounce_signal = np.zeros(len(df))

        for i in range(len(df) - lookforward):
            touched_lower = df['touched_lower'].iloc[i]
            touched_upper = df['touched_upper'].iloc[i]
            broke_lower = df['broke_lower'].iloc[i]
            broke_upper = df['broke_upper'].iloc[i]

            if touched_lower or broke_lower:
                future_high = df['high'].iloc[i:i+lookforward].max()
                current_close = df['close'].iloc[i]
                bounce_pct = (future_high - current_close) / current_close

                if bounce_pct > bounce_threshold:
                    bounce_signal[i] = 1

            elif touched_upper or broke_upper:
                future_low = df['low'].iloc[i:i+lookforward].min()
                current_close = df['close'].iloc[i]
                bounce_pct = (current_close - future_low) / current_close

                if bounce_pct > bounce_threshold:
                    bounce_signal[i] = 1

        return bounce_signal

    def _validate_training_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and clean training data."""
        print(f"Validating training data... (shape: {X.shape})")
        
        if np.isnan(X).any() or np.isinf(X).any():
            print("WARNING: Found NaN or infinity values in features. Cleaning data...")
            
            X_clean = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            
            for col_idx in range(X_clean.shape[1]):
                col_data = X_clean[:, col_idx]
                col_mean = np.nanmean(col_data[np.isfinite(col_data)])
                col_std = np.nanstd(col_data[np.isfinite(col_data)])
                
                if col_std == 0:
                    col_std = 1
                
                X_clean[:, col_idx] = np.clip(
                    col_data,
                    col_mean - 5*col_std,
                    col_mean + 5*col_std
                )
            
            X = X_clean
            valid_mask = np.all(np.isfinite(X), axis=1)
            X = X[valid_mask]
            y = y[valid_mask]
            
            print(f"Cleaned data shape: {X.shape} (removed {len(y) - X.shape[0]} invalid rows)")
        
        return X, y

    def prepare_training_data(self, lookback: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for BB bounce prediction model."""
        if self.feature_data is None:
            self.engineer_features()

        print("Preparing training data for BB bounce prediction...")

        df = self.calculate_bb_metrics(self.feature_data.copy())
        df['bounce_target'] = self.calculate_bounce_target(df)

        feature_names = [
            'sma_fast', 'sma_medium', 'sma_slow', 'ema_fast', 'ema_slow',
            'rsi', 'macd', 'macd_signal', 'macd_histogram', 'momentum', 'volatility',
            'bb_basis', 'bb_middle', 'bb_width', 'bb_position', 'basis_slope',
            'atr', 'trend_strength', 'obv',
            'volume_momentum', 'price_position', 'volume_relative_strength',
            'close_location', 'momentum_divergence', 'volatility_acceleration',
            'multi_timeframe_strength'
        ]

        feature_cols = [col for col in feature_names if col in df.columns]

        valid_mask = (
            (df['touched_lower'] | df['touched_upper'] | df['broke_lower'] | df['broke_upper']) &
            df[feature_cols].notna().all(axis=1) &
            df['bounce_target'].notna()
        )

        df_valid = df[valid_mask].copy()

        X = df_valid[feature_cols].values
        y = df_valid['bounce_target'].values.astype(int)

        X, y = self._validate_training_data(X, y)

        print(f"Training data prepared: {X.shape[0]} BB touch/break events, {X.shape[1]} features")
        print(f"Timeframe: {self.timeframe}")
        print(f"Effective bounces: {(y == 1).sum()}, Ineffective: {(y == 0).sum()}")
        print(f"Bounce rate: {(y == 1).sum() / len(y) * 100:.2f}%")

        return X, y

    def _apply_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE balancing to training data."""
        if not SMOTE_AVAILABLE:
            print("Warning: imbalanced-learn not installed. Skipping SMOTE.")
            print("Install with: pip install imbalanced-learn")
            return X, y
        
        print(f"Applying SMOTE balancing (ratio={self.smote_ratio})...")
        try:
            smote = SMOTE(sampling_strategy=self.smote_ratio, random_state=42)
            X_smote, y_smote = smote.fit_resample(X, y)
            print(f"SMOTE completed: {X_smote.shape[0]} samples (from {X.shape[0]})")
            print(f"Class distribution: {(y_smote == 0).sum()} negative, {(y_smote == 1).sum()} positive")
            return X_smote, y_smote
        except Exception as e:
            print(f"SMOTE error: {str(e)}. Continuing without SMOTE.")
            return X, y

    def train(self, epochs: int = None) -> Dict:
        """Train the BB bounce prediction model with optimization."""
        X, y = self.prepare_training_data()

        if len(X) < 100:
            print("Warning: Not enough training data. Need at least 100 samples.")
            return {}

        if self.use_smote:
            X, y = self._apply_smote(X, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=ML_CONFIG.get('test_size', 0.2),
            random_state=ML_CONFIG.get('random_state', 42),
            stratify=y
        )

        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

        print(f"Training {self.model_type} model for {self.timeframe}...")
        print(f"Hyperparameters: {self.hyperparams}")

        if self.use_ensemble:
            self.ensemble_models = self._train_ensemble()
            predictions_train = self._ensemble_predict(self.X_train)
            predictions_test = self._ensemble_predict(self.X_test)
        else:
            if self.model_type == 'xgboost':
                self.model = XGBClassifier(
                    n_estimators=self.hyperparams['n_estimators'],
                    max_depth=self.hyperparams['max_depth'],
                    learning_rate=self.hyperparams['learning_rate'],
                    subsample=0.9,
                    colsample_bytree=0.9,
                    min_child_weight=3,
                    random_state=42,
                    eval_metric='logloss',
                    scale_pos_weight=(self.y_train == 0).sum() / (self.y_train == 1).sum()
                )
                self.model.fit(self.X_train, self.y_train, verbose=0)
            elif self.model_type == 'lightgbm':
                self.model = LGBMClassifier(
                    n_estimators=self.hyperparams['n_estimators'],
                    max_depth=self.hyperparams['max_depth'],
                    learning_rate=self.hyperparams['learning_rate'],
                    random_state=42,
                    verbose=-1,
                    scale_pos_weight=(self.y_train == 0).sum() / (self.y_train == 1).sum()
                )
                self.model.fit(self.X_train, self.y_train)
            elif self.model_type == 'random_forest':
                self.model = RandomForestClassifier(
                    n_estimators=150,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    class_weight='balanced'
                )
                self.model.fit(self.X_train, self.y_train)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            predictions_train = self.model.predict(self.X_train)
            predictions_test = self.model.predict(self.X_test)

        train_score = (predictions_train == self.y_train).mean()
        test_score = (predictions_test == self.y_test).mean()

        train_precision = self._calculate_precision(predictions_train, self.y_train)
        test_precision = self._calculate_precision(predictions_test, self.y_test)
        train_recall = self._calculate_recall(predictions_train, self.y_train)
        test_recall = self._calculate_recall(predictions_test, self.y_test)

        print(f"Model training completed for {self.timeframe}:")
        print(f"  Train accuracy: {train_score:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
        print(f"  Test accuracy: {test_score:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")

        self.save_model()

        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'train_precision': train_precision,
            'test_precision': test_precision,
            'train_recall': train_recall,
            'test_recall': test_recall,
            'model_type': self.model_type,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'optimization': self.optimization_level,
        }

    def _train_ensemble(self):
        """Train ensemble of multiple models."""
        print("Training ensemble models...")
        models = [
            XGBClassifier(
                n_estimators=self.hyperparams['n_estimators'],
                max_depth=self.hyperparams['max_depth'],
                learning_rate=self.hyperparams['learning_rate'],
                random_state=42,
                eval_metric='logloss',
                verbose=0,
            ),
            RandomForestClassifier(
                n_estimators=150,
                max_depth=15,
                min_samples_split=10,
                random_state=42,
                class_weight='balanced'
            ),
            LGBMClassifier(
                n_estimators=self.hyperparams['n_estimators'],
                max_depth=self.hyperparams['max_depth'],
                learning_rate=self.hyperparams['learning_rate'],
                random_state=42,
                verbose=-1,
            )
        ]
        
        for model in models:
            model.fit(self.X_train, self.y_train)
        
        return models

    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions with voting."""
        predictions = np.array([model.predict(X) for model in self.ensemble_models])
        return (predictions.sum(axis=0) >= 2).astype(int)

    def _calculate_precision(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate precision score."""
        true_positives = ((y_pred == 1) & (y_true == 1)).sum()
        false_positives = ((y_pred == 1) & (y_true == 0)).sum()
        if true_positives + false_positives == 0:
            return 0.0
        return true_positives / (true_positives + false_positives)

    def _calculate_recall(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate recall score."""
        true_positives = ((y_pred == 1) & (y_true == 1)).sum()
        false_negatives = ((y_pred == 0) & (y_true == 1)).sum()
        if true_positives + false_negatives == 0:
            return 0.0
        return true_positives / (true_positives + false_negatives)

    def evaluate_entries(self, lookback: int = 50) -> pd.DataFrame:
        """Evaluate current BB touch/break for bounce probability."""
        if self.feature_data is None:
            self.engineer_features()

        if self.model is None and self.ensemble_models is None:
            self.load_model()
            if self.model is None and self.ensemble_models is None:
                print(f"Warning: No model available for {self.timeframe}. Training new model...")
                self.train()

        df = self.calculate_bb_metrics(self.feature_data.copy())
        recent_df = df.tail(lookback).copy()

        feature_names = [
            'sma_fast', 'sma_medium', 'sma_slow', 'ema_fast', 'ema_slow',
            'rsi', 'macd', 'macd_signal', 'macd_histogram', 'momentum', 'volatility',
            'bb_basis', 'bb_middle', 'bb_width', 'bb_position', 'basis_slope',
            'atr', 'trend_strength', 'obv',
            'volume_momentum', 'price_position', 'volume_relative_strength',
            'close_location', 'momentum_divergence', 'volatility_acceleration',
            'multi_timeframe_strength'
        ]

        feature_cols = [col for col in feature_names if col in recent_df.columns]
        valid_mask = recent_df[feature_cols].notna().all(axis=1)
        recent_df_valid = recent_df[valid_mask].copy()

        X_recent = recent_df_valid[feature_cols].values
        X_scaled = self.scaler.transform(X_recent)

        if self.ensemble_models is not None:
            predictions = self._ensemble_predict(X_scaled)
            probabilities = np.mean([m.predict_proba(X_scaled)[:, 1] for m in self.ensemble_models], axis=0) * 100
        else:
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)[:, 1] * 100 if hasattr(self.model, 'predict_proba') else predictions * 100

        recent_df_valid['bounce_prediction'] = predictions
        recent_df_valid['bounce_probability'] = probabilities

        recent_df_valid['is_bb_touch'] = recent_df_valid['touched_lower'] | recent_df_valid['touched_upper']
        recent_df_valid['is_bb_break'] = recent_df_valid['broke_lower'] | recent_df_valid['broke_upper']
        recent_df_valid['signal_type'] = 'none'
        recent_df_valid.loc[recent_df_valid['touched_lower'], 'signal_type'] = 'lower_touch'
        recent_df_valid.loc[recent_df_valid['touched_upper'], 'signal_type'] = 'upper_touch'
        recent_df_valid.loc[recent_df_valid['broke_lower'], 'signal_type'] = 'lower_break'
        recent_df_valid.loc[recent_df_valid['broke_upper'], 'signal_type'] = 'upper_break'

        print(f"Entry evaluation complete for {self.timeframe}: {len(recent_df_valid)} candles analyzed")
        return recent_df_valid

    def save_model(self) -> None:
        """Save trained model to disk."""
        if self.model is None and self.ensemble_models is None:
            print("No model to save")
            return

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'ensemble_models': self.ensemble_models,
            'scaler': self.scaler,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'model_type': self.model_type,
            'optimization_level': self.optimization_level,
        }, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self) -> bool:
        """Load trained model from disk."""
        if not self.model_path.exists():
            print(f"Model file not found: {self.model_path}")
            return False

        try:
            checkpoint = joblib.load(self.model_path)
            self.model = checkpoint.get('model')
            self.ensemble_models = checkpoint.get('ensemble_models')
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
                'volatility': latest.get('volatility', None),
            },
            'model_info': {
                'type': self.model_type,
                'optimization': self.optimization_level,
                'trained': self.model is not None or self.ensemble_models is not None,
                'path': str(self.model_path) if self.model_path else None,
            }
        }
