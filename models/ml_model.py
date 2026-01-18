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

from models.config import ML_CONFIG, MODEL_CONFIG, DEFAULT_SYMBOL, DEFAULT_TIMEFRAME, TIMEFRAME_CONFIGS
from models.data_processor import DataProcessor
from models.feature_engineer import FeatureEngineer
from models.multi_timeframe_engineer import MultiTimeframeEngineer
from models.feature_selector import FeatureSelector
from models.signal_evaluator import SignalEvaluator


class CryptoEntryModel:
    """ML model for BB channel bounce prediction with multi-timeframe features.
    
    Strategy: Use cost-sensitive learning (scale_pos_weight) with threshold calibration
    instead of SMOTE to achieve 80%+ recall and precision on imbalanced data.
    """

    TIMEFRAME_HIERARCHY = {
        '15m': '1h',
        '1h': '4h',
        '4h': '1d',
        '1d': None
    }

    def __init__(self, symbol: str = DEFAULT_SYMBOL, timeframe: str = DEFAULT_TIMEFRAME,
                 model_type: str = 'xgboost', optimization_level: str = 'balanced',
                 use_multi_timeframe: bool = True, use_feature_selection: bool = False,
                 enable_debug: bool = False, decision_threshold: float = 0.5):
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_type = model_type
        self.optimization_level = optimization_level
        self.use_multi_timeframe = use_multi_timeframe
        self.use_feature_selection = use_feature_selection
        self.enable_debug = enable_debug
        self.decision_threshold = decision_threshold
        
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
        self.optimal_threshold = decision_threshold
        self.model_path = Path(__file__).parent / f"cache/{symbol}_{timeframe}_{model_type}.joblib"
        
        self.debug_info = {}

    def _apply_optimization_config(self):
        """Apply conservative hyperparameters optimized for generalization."""
        if self.optimization_level == 'conservative':
            self.hyperparams = {
                'max_depth': 5,
                'learning_rate': 0.05,
                'n_estimators': 100,
                'min_child_weight': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_lambda': 2.0,
                'reg_alpha': 0.5,
            }
        elif self.optimization_level == 'aggressive':
            self.hyperparams = {
                'max_depth': 7,
                'learning_rate': 0.1,
                'n_estimators': 150,
                'min_child_weight': 3,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'reg_lambda': 0.5,
                'reg_alpha': 0.1,
            }
        else:
            self.hyperparams = {
                'max_depth': 6,
                'learning_rate': 0.08,
                'n_estimators': 120,
                'min_child_weight': 4,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'reg_lambda': 1.0,
                'reg_alpha': 0.3,
            }
        
        self.bounce_threshold_multiplier = 1.0
        if self.optimization_level == 'conservative':
            self.bounce_threshold_multiplier = 1.2
        elif self.optimization_level == 'aggressive':
            self.bounce_threshold_multiplier = 0.8
        
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
        """Calculate Bollinger Bands metrics for bounce detection using timeframe config.
        
        Uses stricter tolerance parameters for more precise BB touch/break detection.
        """
        df_bb = df.copy()

        bb_period = self.timeframe_config['bb_period']
        bb_std = self.timeframe_config['bb_std']
        bb_touch_tol_upper = self.timeframe_config.get('bb_touch_tolerance_upper', 0.02)
        bb_touch_tol_lower = self.timeframe_config.get('bb_touch_tolerance_lower', 0.02)

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

        # Stricter BB touch definition (1% tolerance instead of 2%)
        df_bb['touched_upper'] = (
            (df_bb['close'] >= bb_upper * (1 - bb_touch_tol_upper)) & 
            (df_bb['close'] <= bb_upper * (1 + bb_touch_tol_upper))
        )
        df_bb['touched_lower'] = (
            (df_bb['close'] <= bb_lower * (1 + bb_touch_tol_lower)) & 
            (df_bb['close'] >= bb_lower * (1 - bb_touch_tol_lower))
        )
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
        """Prepare data for BB bounce prediction model with improved labeling.
        
        Critical: Calculate BB metrics FIRST before using multi-timeframe features
        to ensure all features are properly initialized.
        """
        if self.raw_data is None:
            self.load_data()

        print("Preparing training data for BB bounce prediction...")

        # Step 1: Calculate BB metrics on base data
        df = self.calculate_bb_metrics(self.raw_data.copy())
        
        # Step 2: Now engineer features with BB metrics available
        print("\nEngineering features with BB metrics...")
        df = self.feature_engineer.engineer_features(df)
        
        # Step 3: Integrate multi-timeframe features if available
        if self.use_multi_timeframe and self.higher_tf_data is not None:
            print("Integrating multi-timeframe features...")
            try:
                higher_tf_data = self.calculate_bb_metrics(self.higher_tf_data.copy())
                higher_tf_features = self.feature_engineer.engineer_features(higher_tf_data)
                df = self.multi_tf_engineer.engineer_comprehensive_features(
                    df,
                    higher_tf_features
                )
                print("Multi-timeframe features integrated successfully")
            except Exception as e:
                print(f"Warning: Multi-timeframe integration failed: {str(e)}")
        
        # Step 4: Calculate bounce targets
        df['bounce_target'] = self.calculate_bounce_target_improved(df)
        self.feature_data = df

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
        
        advanced_features = [
            'bounce_failure_memory', 'volume_zscore', 'volume_ratio', 'volume_momentum',
            'volume_anomaly_score', 'volume_extreme', 'reversal_speed', 'reversal_magnitude',
            'reversal_acceleration', 'time_of_day_score', 'day_of_week_score', 'session_type',
            'time_quality', 'advanced_reversal_score', 'is_strong_setup', 'is_weak_setup'
        ]
        
        feature_cols = [col for col in base_features if col in df.columns]
        if self.use_multi_timeframe:
            multi_tf_cols = [col for col in multi_tf_features if col in df.columns]
            feature_cols.extend(multi_tf_cols)
        
        adv_cols = [col for col in advanced_features if col in df.columns]
        feature_cols.extend(adv_cols)

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
        X_selected, selected_names = self.feature_selector.select_by_random_forest(X, y, feature_names)
        self.selected_feature_names = selected_names
        
        print(f"Reduced from {len(feature_names)} to {len(selected_names)} features\n")
        
        self.debug_info['original_feature_count'] = len(feature_names)
        self.debug_info['selected_feature_count'] = len(selected_names)
        
        return X_selected, selected_names

    def _find_optimal_threshold(self, y_proba: np.ndarray, y_true: np.ndarray) -> Tuple[float, Dict]:
        """Find optimal decision threshold to maximize F1 score.
        
        Research shows threshold calibration is more effective than SMOTE for imbalanced data.
        """
        best_f1 = 0
        best_threshold = 0.5
        best_metrics = {}
        
        thresholds = np.arange(0.1, 0.95, 0.05)
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1 and precision >= 0.80 and recall >= 0.80:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                }
        
        if not best_metrics:
            for threshold in thresholds:
                y_pred = (y_proba >= threshold).astype(int)
                
                tp = ((y_pred == 1) & (y_true == 1)).sum()
                fp = ((y_pred == 1) & (y_true == 0)).sum()
                fn = ((y_pred == 0) & (y_true == 1)).sum()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_metrics = {
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                    }
        
        return best_threshold, best_metrics

    def train(self, epochs: int = None) -> Dict:
        """Train BB bounce prediction model using cost-sensitive learning.
        
        Strategy: Use scale_pos_weight for class imbalance, then calibrate threshold
        for 80%+ recall and precision targets.
        """
        X, y, feature_names = self.prepare_training_data()

        if len(X) < 100:
            print("Warning: Not enough training data. Need at least 100 samples.")
            return {}

        X, feature_names = self._apply_feature_selection(X, y, feature_names)

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
        
        # Calculate scale_pos_weight with multiplier from config
        base_scale_pos_weight = (self.y_train == 0).sum() / max((self.y_train == 1).sum(), 1)
        scale_pos_weight_multiplier = self.timeframe_config.get('scale_pos_weight_multiplier', 1.0)
        scale_pos_weight = base_scale_pos_weight * scale_pos_weight_multiplier
        
        print(f"\nClass imbalance ratio: {base_scale_pos_weight:.2f}:1")
        print(f"Scale pos weight multiplier: {scale_pos_weight_multiplier}x")
        print(f"Final scale_pos_weight = {scale_pos_weight:.2f}")
        
        self.debug_info['train_set_size'] = len(X_train)
        self.debug_info['test_set_size'] = len(X_test)
        self.debug_info['train_positive'] = (y_train == 1).sum()
        self.debug_info['train_negative'] = (y_train == 0).sum()
        self.debug_info['test_positive'] = (y_test == 1).sum()
        self.debug_info['test_negative'] = (y_test == 0).sum()
        self.debug_info['base_scale_pos_weight'] = base_scale_pos_weight
        self.debug_info['scale_pos_weight'] = scale_pos_weight
        self.debug_info['scale_pos_weight_multiplier'] = scale_pos_weight_multiplier

        print(f"\nTraining {self.model_type} model for {self.timeframe}...")
        print(f"Hyperparameters: {self.hyperparams}")

        if self.model_type == 'xgboost':
            self.model = XGBClassifier(
                n_estimators=self.hyperparams.get('n_estimators', 120),
                max_depth=self.hyperparams.get('max_depth', 6),
                learning_rate=self.hyperparams.get('learning_rate', 0.08),
                subsample=self.hyperparams.get('subsample', 0.85),
                colsample_bytree=self.hyperparams.get('colsample_bytree', 0.85),
                min_child_weight=self.hyperparams.get('min_child_weight', 4),
                reg_lambda=self.hyperparams.get('reg_lambda', 1.0),
                reg_alpha=self.hyperparams.get('reg_alpha', 0.3),
                random_state=42,
                eval_metric='logloss',
                scale_pos_weight=scale_pos_weight
            )
            self.model.fit(self.X_train, self.y_train, verbose=0)
        elif self.model_type == 'lightgbm':
            self.model = LGBMClassifier(
                n_estimators=self.hyperparams.get('n_estimators', 120),
                max_depth=self.hyperparams.get('max_depth', 6),
                learning_rate=self.hyperparams.get('learning_rate', 0.08),
                subsample=self.hyperparams.get('subsample', 0.85),
                colsample_bytree=self.hyperparams.get('colsample_bytree', 0.85),
                reg_lambda=self.hyperparams.get('reg_lambda', 1.0),
                reg_alpha=self.hyperparams.get('reg_alpha', 0.3),
                random_state=42,
                verbose=-1,
                scale_pos_weight=scale_pos_weight
            )
            self.model.fit(self.X_train, self.y_train)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=15,
                min_samples_leaf=8,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            self.model.fit(self.X_train, self.y_train)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        y_proba_train = self.model.predict_proba(self.X_train)[:, 1]
        y_proba_test = self.model.predict_proba(self.X_test)[:, 1]
        
        print("\nApplying decision threshold strategy...")
        print(f"Decision threshold (Strategy A): {self.decision_threshold:.3f}")
        self.optimal_threshold = self.decision_threshold
        
        predictions_train = (y_proba_train >= self.optimal_threshold).astype(int)
        predictions_test = (y_proba_test >= self.optimal_threshold).astype(int)

        train_score = (predictions_train == self.y_train).mean()
        test_score = (predictions_test == self.y_test).mean()

        train_precision = self._calculate_precision(predictions_train, self.y_train)
        test_precision = self._calculate_precision(predictions_test, self.y_test)
        train_recall = self._calculate_recall(predictions_train, self.y_train)
        test_recall = self._calculate_recall(predictions_test, self.y_test)
        
        train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0
        test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0

        print(f"\nModel training completed for {self.timeframe}:")
        print(f"  Train accuracy: {train_score:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
        print(f"  Test accuracy: {test_score:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
        
        precision_gap = train_precision - test_precision
        recall_gap = train_recall - test_recall
        
        print(f"\nGeneralization Analysis:")
        print(f"  Precision gap: {precision_gap:.4f} ({precision_gap*100:.1f}%)")
        print(f"  Recall gap: {recall_gap:.4f} ({recall_gap*100:.1f}%)")
        
        if precision_gap < 0.05 and recall_gap < 0.05:
            print(f"  Status: EXCELLENT generalization")
        elif precision_gap < 0.10 and recall_gap < 0.10:
            print(f"  Status: GOOD generalization")
        elif precision_gap < 0.15 and recall_gap < 0.15:
            print(f"  Status: ACCEPTABLE generalization")
        else:
            print(f"  Status: POOR generalization - consider retraining")
        
        if test_precision >= 0.70 and test_recall >= 0.80:
            print(f"\nRESULT: Precision {test_precision:.4f} + Recall {test_recall:.4f} (Strategy A successful)")
        
        self.debug_info['train_predictions_positive'] = (predictions_train == 1).sum()
        self.debug_info['train_predictions_negative'] = (predictions_train == 0).sum()
        self.debug_info['test_predictions_positive'] = (predictions_test == 1).sum()
        self.debug_info['test_predictions_negative'] = (predictions_test == 0).sum()
        self.debug_info['train_f1'] = train_f1
        self.debug_info['test_f1'] = test_f1
        self.debug_info['optimal_threshold'] = self.optimal_threshold
        self.debug_info['decision_strategy'] = 'threshold_0.5'

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
            'optimal_threshold': self.optimal_threshold,
        }

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
        """Evaluate current BB touch/break for bounce probability.
        
        Note: bounce_probability is in range [0, 1], not percentage.
        """
        if self.feature_data is None:
            self.engineer_features()

        if self.model is None:
            self.load_model()
            if self.model is None:
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

        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        predictions = (probabilities >= self.optimal_threshold).astype(int)

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
            'optimization_level': self.optimization_level,
            'use_multi_timeframe': self.use_multi_timeframe,
            'selected_feature_names': self.selected_feature_names,
            'optimal_threshold': self.optimal_threshold,
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
            self.scaler = checkpoint['scaler']
            self.selected_feature_names = checkpoint.get('selected_feature_names')
            self.optimal_threshold = checkpoint.get('optimal_threshold', 0.5)
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
                'trained': self.model is not None,
                'path': str(self.model_path) if self.model_path else None,
                'multi_timeframe': self.use_multi_timeframe,
                'feature_selection': self.use_feature_selection,
                'optimal_threshold': self.optimal_threshold,
            },
            'debug_info': self.debug_info if self.enable_debug else {}
        }
