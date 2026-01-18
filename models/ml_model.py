import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
from pathlib import Path

try:
    from imblearn.over_sampling import BorderlineSMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

from models.config import ML_CONFIG, MODEL_CONFIG, DEFAULT_SYMBOL, DEFAULT_TIMEFRAME, TIMEFRAME_CONFIGS
from models.data_processor import DataProcessor
from models.feature_engineer import FeatureEngineer
from models.multi_timeframe_engineer import MultiTimeframeEngineer
from models.feature_selector import FeatureSelector
from models.signal_evaluator import SignalEvaluator


class CryptoEntryModel:
    """ML model for BB channel bounce prediction with multi-timeframe features."""

    TIMEFRAME_HIERARCHY = {
        '15m': '1h',
        '1h': '4h',
        '4h': '1d',
        '1d': None
    }

    def __init__(self, symbol: str = DEFAULT_SYMBOL, timeframe: str = DEFAULT_TIMEFRAME,
                 model_type: str = 'xgboost', optimization_level: str = 'balanced',
                 use_multi_timeframe: bool = True, use_feature_selection: bool = False,
                 enable_debug: bool = False):
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_type = model_type
        self.optimization_level = optimization_level
        self.use_multi_timeframe = use_multi_timeframe
        self.use_feature_selection = use_feature_selection
        self.enable_debug = enable_debug
        
        self.data_processor = DataProcessor(symbol, timeframe)
        self.feature_engineer = FeatureEngineer()
        self.signal_evaluator = SignalEvaluator()
        self.feature_selector = FeatureSelector(n_features=25) if use_feature_selection else None
        
        if self.use_multi_timeframe:
            self.multi_tf_engineer = MultiTimeframeEngineer(symbol, timeframe)
        else:
            self.multi_tf_engineer = None

        self.timeframe_config = TIMEFRAME_CONFIGS.get(timeframe, TIMEFRAME_CONFIGS['15m'])
        self._apply_optimization_config()

        self.raw_data = None
        self.higher_tf_data = None
        self.feature_data = None
        self.selected_feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.model = None
        self.ensemble_models = None
        self.model_path = Path(__file__).parent / f"cache/{symbol}_{timeframe}_{model_type}.joblib"
        
        self.debug_info = {}

    def _apply_optimization_config(self):
        """Apply optimization-specific configuration with aggressive overfitting prevention."""
        if self.optimization_level == 'conservative':
            self.use_smote = True
            self.smote_ratio = 0.1
            self.bounce_threshold_multiplier = 1.2
            self.use_ensemble = True
            self.hyperparams = {
                'max_depth': 5,
                'learning_rate': 0.03,
                'n_estimators': 80,
            }
        elif self.optimization_level == 'aggressive':
            self.use_smote = True
            self.smote_ratio = 0.15
            self.bounce_threshold_multiplier = 0.8
            self.use_ensemble = False
            self.hyperparams = {
                'max_depth': 7,
                'learning_rate': 0.08,
                'n_estimators': 120,
            }
        else:
            self.use_smote = True
            self.smote_ratio = 0.1
            self.bounce_threshold_multiplier = 1.0
            self.use_ensemble = False
            self.hyperparams = {
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 100,
            }
        
        self.timeframe_config['bounce_threshold'] *= self.bounce_threshold_multiplier

    def _get_higher_timeframe(self) -> Optional[str]:
        """Get immediate higher timeframe."""
        return self.TIMEFRAME_HIERARCHY.get(self.timeframe)

    def load_data(self) -> pd.DataFrame:
        """Load base timeframe and higher timeframe data if available."""
        print(f"Loading data for {self.symbol} ({self.timeframe})...")
        self.raw_data = self.data_processor.load_data()
        self.raw_data = self.data_processor.prepare_data(self.raw_data)
        self.data_processor.validate_data_integrity(self.raw_data)
        print(f"Data loaded: {len(self.raw_data)} candles")
        
        if self.use_multi_timeframe:
            higher_tf = self._get_higher_timeframe()
            if higher_tf:
                print(f"Preloading {higher_tf} data for multi-timeframe features...")
                try:
                    higher_processor = DataProcessor(self.symbol, higher_tf)
                    self.higher_tf_data = higher_processor.load_data()
                    self.higher_tf_data = higher_processor.prepare_data(self.higher_tf_data)
                    higher_processor.validate_data_integrity(self.higher_tf_data)
                    print(f"Higher timeframe data loaded: {len(self.higher_tf_data)} candles")
                except Exception as e:
                    print(f"Warning: Failed to load {higher_tf} data: {str(e)}")
                    self.higher_tf_data = None
            else:
                print(f"No higher timeframe available for {self.timeframe}")
                self.higher_tf_data = None
        
        return self.raw_data

    def engineer_features(self) -> pd.DataFrame:
        """Engineer technical features including BB indicators and multi-timeframe data."""
        if self.raw_data is None:
            self.load_data()

        print("Engineering features...")
        self.feature_data = self.feature_engineer.engineer_features(self.raw_data)
        
        if self.use_multi_timeframe and self.higher_tf_data is not None:
            print("Integrating multi-timeframe features...")
            try:
                higher_tf_features = self.feature_engineer.engineer_features(self.higher_tf_data)
                self.feature_data = self.multi_tf_engineer.engineer_comprehensive_features(
                    self.feature_data,
                    higher_tf_features
                )
                print("Multi-timeframe features integrated successfully")
            except Exception as e:
                print(f"Warning: Multi-timeframe feature engineering failed: {str(e)}")
                print("Continuing with base timeframe features only")
        else:
            if self.use_multi_timeframe:
                print("No higher timeframe data available, using base features only")
        
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

    def calculate_bounce_target_improved(self, df: pd.DataFrame) -> np.ndarray:
        """Improved bounce target calculation with multiple lookforward periods.
        
        Stricter criteria: requires sustained bounce with minimum threshold.
        """
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
                
                high_above_bb = (df['high'].iloc[i:i+lookforward] > df['bb_upper'].iloc[i]).sum()
                
                if bounce_pct > bounce_threshold and high_above_bb >= 1:
                    bounce_signal[i] = 1

            elif touched_upper or broke_upper:
                future_low = df['low'].iloc[i:i+lookforward].min()
                current_close = df['close'].iloc[i]
                bounce_pct = (current_close - future_low) / current_close
                
                low_below_bb = (df['low'].iloc[i:i+lookforward] < df['bb_lower'].iloc[i]).sum()
                
                if bounce_pct > bounce_threshold and low_below_bb >= 1:
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

    def prepare_training_data(self, lookback: int = 50) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for BB bounce prediction model with improved labeling."""
        if self.feature_data is None:
            self.engineer_features()

        print("Preparing training data for BB bounce prediction...")

        df = self.calculate_bb_metrics(self.feature_data.copy())
        df['bounce_target'] = self.calculate_bounce_target_improved(df)

        base_features = [
            'sma_fast', 'sma_medium', 'sma_slow', 'ema_fast', 'ema_slow',
            'rsi', 'macd', 'macd_signal', 'macd_histogram', 'momentum', 'volatility',
            'bb_basis', 'bb_middle', 'bb_width', 'bb_position', 'basis_slope',
            'atr', 'trend_strength', 'obv',
            'volume_momentum', 'price_position', 'volume_relative_strength',
            'close_location', 'momentum_divergence', 'volatility_acceleration',
            'multi_timeframe_strength'
        ]

        multi_tf_features = [
            'timeframe_confirmation', 'trend_alignment', 'high_volatility_context',
            'momentum_divergence_multi'
        ]
        
        feature_cols = [col for col in base_features if col in df.columns]
        if self.use_multi_timeframe:
            multi_tf_cols = [col for col in multi_tf_features if col in df.columns]
            feature_cols.extend(multi_tf_cols)

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
        
        self.debug_info['raw_positive_count'] = (y == 1).sum()
        self.debug_info['raw_negative_count'] = (y == 0).sum()
        self.debug_info['raw_bounce_rate'] = (y == 1).sum() / len(y) * 100

        return X, y, feature_cols

    def _apply_feature_selection(self, X: np.ndarray, y: np.ndarray, 
                                feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Apply feature selection to reduce dimensionality and improve precision."""
        if not self.use_feature_selection:
            return X, feature_names
        
        print("\nApplying feature selection...")
        print(f"Selecting top 25 features using random forest importance...")
        X_selected, selected_names = self.feature_selector.select_by_random_forest(X, y, feature_names)
        self.selected_feature_names = selected_names
        print(f"Selected {len(selected_names)} features\n")
        
        if self.enable_debug:
            print("Top selected features:")
            for i, feat in enumerate(selected_names[:10], 1):
                print(f"  {i}. {feat}")
        
        print(f"Reduced from {len(feature_names)} to {len(selected_names)} features\n")
        
        self.debug_info['original_feature_count'] = len(feature_names)
        self.debug_info['selected_feature_count'] = len(selected_names)
        
        return X_selected, selected_names

    def _apply_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Borderline-SMOTE balancing to training data.
        
        Borderline-SMOTE only generates synthetic samples near decision boundaries,
        reducing noise compared to standard SMOTE.
        """
        if not SMOTE_AVAILABLE:
            print("Warning: imbalanced-learn not installed. Skipping SMOTE.")
            print("Install with: pip install imbalanced-learn")
            return X, y
        
        print(f"Applying Borderline-SMOTE balancing (ratio={self.smote_ratio})...")
        try:
            smote = BorderlineSMOTE(sampling_strategy=self.smote_ratio, random_state=42, kind='borderline-2')
            X_smote, y_smote = smote.fit_resample(X, y)
            print(f"Borderline-SMOTE completed: {X_smote.shape[0]} samples (from {X.shape[0]})")
            print(f"Class distribution: {(y_smote == 0).sum()} negative, {(y_smote == 1).sum()} positive")
            print(f"Augmentation ratio: {(X_smote.shape[0] / X.shape[0]):.2f}x")
            print(f"Note: Borderline-SMOTE generates fewer samples near decision boundary to reduce noise.")
            
            self.debug_info['smote_original_samples'] = X.shape[0]
            self.debug_info['smote_augmented_samples'] = X_smote.shape[0]
            self.debug_info['smote_augmentation_ratio'] = X_smote.shape[0] / X.shape[0]
            self.debug_info['smote_synthetic_positive'] = (y_smote == 1).sum() - (y == 1).sum()
            
            return X_smote, y_smote
        except Exception as e:
            print(f"SMOTE error: {str(e)}. Continuing without SMOTE.")
            return X, y

    def train(self, epochs: int = None) -> Dict:
        """Train the BB bounce prediction model with optimization."""
        X, y, feature_names = self.prepare_training_data()

        if len(X) < 100:
            print("Warning: Not enough training data. Need at least 100 samples.")
            return {}

        X, feature_names = self._apply_feature_selection(X, y, feature_names)

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
        
        self.debug_info['train_set_size'] = len(X_train)
        self.debug_info['test_set_size'] = len(X_test)
        self.debug_info['train_positive'] = (y_train == 1).sum()
        self.debug_info['train_negative'] = (y_train == 0).sum()
        self.debug_info['test_positive'] = (y_test == 1).sum()
        self.debug_info['test_negative'] = (y_test == 0).sum()
        self.debug_info['train_positive_ratio'] = (y_train == 1).sum() / len(y_train) * 100
        self.debug_info['test_positive_ratio'] = (y_test == 1).sum() / len(y_test) * 100

        print(f"Training {self.model_type} model for {self.timeframe}...")
        print(f"Hyperparameters: {self.hyperparams}")

        if self.use_ensemble:
            self.ensemble_models = self._train_ensemble()
            predictions_train = self._ensemble_predict(self.X_train)
            predictions_test = self._ensemble_predict(self.X_test)
        else:
            if self.model_type == 'xgboost':
                scale_pos_weight = (self.y_train == 0).sum() / max((self.y_train == 1).sum(), 1)
                self.model = XGBClassifier(
                    n_estimators=self.hyperparams.get('n_estimators', 100),
                    max_depth=self.hyperparams.get('max_depth', 6),
                    learning_rate=self.hyperparams.get('learning_rate', 0.05),
                    subsample=0.75,
                    colsample_bytree=0.75,
                    min_child_weight=10,
                    reg_lambda=10.0,
                    reg_alpha=1.0,
                    random_state=42,
                    eval_metric='logloss',
                    scale_pos_weight=scale_pos_weight,
                    early_stopping_rounds=20,
                    verbosity=0
                )
                
                eval_set = [(self.X_test, self.y_test)]
                self.model.fit(
                    self.X_train, self.y_train,
                    eval_set=eval_set,
                    verbose=False
                )
            elif self.model_type == 'lightgbm':
                scale_pos_weight = (self.y_train == 0).sum() / max((self.y_train == 1).sum(), 1)
                self.model = LGBMClassifier(
                    n_estimators=self.hyperparams.get('n_estimators', 100),
                    max_depth=self.hyperparams.get('max_depth', 6),
                    learning_rate=self.hyperparams.get('learning_rate', 0.05),
                    num_leaves=31,
                    subsample=0.75,
                    colsample_bytree=0.75,
                    reg_lambda=10.0,
                    reg_alpha=1.0,
                    random_state=42,
                    verbose=-1,
                    scale_pos_weight=scale_pos_weight
                )
                self.model.fit(self.X_train, self.y_train)
            elif self.model_type == 'random_forest':
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=42,
                    class_weight='balanced',
                    n_jobs=-1
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
        
        train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0
        test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0

        print(f"Model training completed for {self.timeframe}:")
        print(f"  Train accuracy: {train_score:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
        print(f"  Test accuracy: {test_score:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
        
        self.debug_info['train_predictions_positive'] = (predictions_train == 1).sum()
        self.debug_info['train_predictions_negative'] = (predictions_train == 0).sum()
        self.debug_info['test_predictions_positive'] = (predictions_test == 1).sum()
        self.debug_info['test_predictions_negative'] = (predictions_test == 0).sum()
        self.debug_info['train_f1'] = train_f1
        self.debug_info['test_f1'] = test_f1

        self.save_model()

        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'train_precision': train_precision,
            'test_precision': test_precision,
            'train_recall': train_recall,
            'test_recall': test_recall,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'model_type': self.model_type,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'optimization': self.optimization_level,
            'multi_timeframe': self.use_multi_timeframe,
            'feature_selection': self.use_feature_selection,
            'feature_count': len(feature_names),
        }

    def _train_ensemble(self):
        """Train ensemble of multiple models."""
        print("Training ensemble models...")
        models = [
            XGBClassifier(
                n_estimators=self.hyperparams.get('n_estimators', 100),
                max_depth=self.hyperparams.get('max_depth', 6),
                learning_rate=self.hyperparams.get('learning_rate', 0.05),
                random_state=42,
                eval_metric='logloss',
                verbose=0,
            ),
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            LGBMClassifier(
                n_estimators=self.hyperparams.get('n_estimators', 100),
                max_depth=self.hyperparams.get('max_depth', 6),
                learning_rate=self.hyperparams.get('learning_rate', 0.05),
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

        base_features = [
            'sma_fast', 'sma_medium', 'sma_slow', 'ema_fast', 'ema_slow',
            'rsi', 'macd', 'macd_signal', 'macd_histogram', 'momentum', 'volatility',
            'bb_basis', 'bb_middle', 'bb_width', 'bb_position', 'basis_slope',
            'atr', 'trend_strength', 'obv',
            'volume_momentum', 'price_position', 'volume_relative_strength',
            'close_location', 'momentum_divergence', 'volatility_acceleration',
            'multi_timeframe_strength'
        ]

        multi_tf_features = [
            'timeframe_confirmation', 'trend_alignment', 'high_volatility_context',
            'momentum_divergence_multi'
        ]
        
        feature_cols = [col for col in base_features if col in recent_df.columns]
        if self.use_multi_timeframe:
            multi_tf_cols = [col for col in multi_tf_features if col in recent_df.columns]
            feature_cols.extend(multi_tf_cols)
        
        if self.selected_feature_names is not None:
            feature_cols = [f for f in self.selected_feature_names if f in recent_df.columns]
        
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
            'use_multi_timeframe': self.use_multi_timeframe,
            'selected_feature_names': self.selected_feature_names,
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
            self.selected_feature_names = checkpoint.get('selected_feature_names')
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
                'multi_timeframe': self.use_multi_timeframe,
                'feature_selection': self.use_feature_selection,
            },
            'debug_info': self.debug_info if self.enable_debug else {}
        }
