"""
Multi-Output XGBoost Model for Support/Resistance/Breakout Prediction
"""

import logging
from typing import Tuple, Optional, Dict, Any
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

from .config import Config, ModelConfig


logger = logging.getLogger(__name__)


class MultiOutputModel:
    """XGBoost model for multi-output regression (support, resistance, breakout)"""
    
    def __init__(self, config: Config):
        """
        Initialize MultiOutputModel
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.model_config = config.model
        self.models = {}  # Dict to hold individual models for each output
        self.scaler = StandardScaler()
        self.feature_names = None
        self.training_history = {}
    
    def build_models(self) -> None:
        """
        Build XGBoost models for each output target
        """
        logger.info(f"Building {self.model_config.n_targets} XGBoost models...")
        
        for target_idx in range(self.model_config.n_targets):
            target_names = ['support_level', 'resistance_level', 'breakout_probability']
            target_name = target_names[target_idx] if target_idx < len(target_names) else f'output_{target_idx}'
            
            model = xgb.XGBRegressor(
                n_estimators=self.model_config.n_estimators,
                max_depth=self.model_config.max_depth,
                learning_rate=self.model_config.learning_rate,
                subsample=self.model_config.subsample,
                colsample_bytree=self.model_config.colsample_bytree,
                min_child_weight=self.model_config.min_child_weight,
                gamma=self.model_config.gamma,
                reg_lambda=self.model_config.lambda_reg,
                reg_alpha=self.model_config.alpha_reg,
                objective='reg:squarederror',
                tree_method=self.model_config.tree_method,
                random_state=self.model_config.random_state,
                n_jobs=self.model_config.n_jobs,
                eval_metric='rmse',
                verbosity=1
            )
            
            self.models[target_name] = model
            logger.info(f"Built model for {target_name}")
    
    def prepare_training_data(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training and test data
        
        Args:
            X: Features DataFrame
            y: Targets DataFrame with 3 columns (support, resistance, breakout)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Remove NaN
        mask = ~(X.isna().any(axis=1) | y.isna().any(axis=1))
        X_clean = X[mask].values
        y_clean = y[mask].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean,
            y_clean,
            test_size=1 - self.model_config.train_test_split,
            random_state=self.model_config.random_state
        )
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        logger.info(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train all models
        
        Args:
            X: Features DataFrame
            y: Targets DataFrame
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training results
        """
        if not self.models:
            self.build_models()
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_training_data(X, y)
        
        results = {}
        target_names = ['support_level', 'resistance_level', 'breakout_probability']
        
        logger.info("Starting training...")
        
        for idx, (target_name, model) in enumerate(self.models.items()):
            logger.info(f"\nTraining {target_name}...")
            
            # Get target column
            y_train_target = y_train[:, idx]
            y_test_target = y_test[:, idx]
            
            # Train with early stopping
            eval_set = [(X_test, y_test_target)]
            
            model.fit(
                X_train,
                y_train_target,
                eval_set=eval_set,
                verbose=1 if verbose else 0,
                early_stopping_rounds=self.model_config.early_stopping_rounds
            )
            
            # Evaluate
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_rmse = np.sqrt(np.mean((train_pred - y_train_target) ** 2))
            test_rmse = np.sqrt(np.mean((test_pred - y_test_target) ** 2))
            
            train_mae = np.mean(np.abs(train_pred - y_train_target))
            test_mae = np.mean(np.abs(test_pred - y_test_target))
            
            # R-squared
            train_r2 = 1 - (np.sum((train_pred - y_train_target) ** 2) / np.sum((y_train_target - np.mean(y_train_target)) ** 2))
            test_r2 = 1 - (np.sum((test_pred - y_test_target) ** 2) / np.sum((y_test_target - np.mean(y_test_target)) ** 2))
            
            results[target_name] = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'best_iteration': model.best_iteration
            }
            
            logger.info(f"{target_name} - Test RMSE: {test_rmse:.6f}, Test MAE: {test_mae:.6f}, Test R2: {test_r2:.4f}")
        
        self.training_history = results
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X: Features DataFrame
            
        Returns:
            Predictions array of shape (n_samples, 3)
        """
        # Scale features
        X_scaled = self.scaler.transform(X.values)
        
        predictions = []
        for model in self.models.values():
            pred = model.predict(X_scaled)
            predictions.append(pred)
        
        # Stack predictions
        predictions = np.column_stack(predictions)
        
        return predictions
    
    def predict_single(self, X: np.ndarray) -> np.ndarray:
        """
        Make prediction on a single sample
        
        Args:
            X: Single sample array (1D)
            
        Returns:
            Prediction array of length 3
        """
        X_scaled = self.scaler.transform(X.reshape(1, -1))
        
        predictions = []
        for model in self.models.values():
            pred = model.predict(X_scaled)[0]
            predictions.append(pred)
        
        return np.array(predictions)
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """
        Get feature importance for each model
        
        Returns:
            Dictionary of feature importances
        """
        importances = {}
        target_names = ['support_level', 'resistance_level', 'breakout_probability']
        
        for (target_name, model), feature_name in zip(self.models.items(), target_names):
            importance_dict = {}
            for feature, importance in zip(self.feature_names, model.feature_importances_):
                importance_dict[feature] = float(importance)
            
            # Sort by importance
            importances[target_name] = dict(sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20])  # Top 20 features
        
        return importances
    
    def save_models(self, model_dir: str) -> None:
        """
        Save trained models to disk
        
        Args:
            model_dir: Directory to save models
        """
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        for target_name, model in self.models.items():
            filepath = model_path / f"{target_name}_model.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved model to {filepath}")
        
        # Save scaler
        scaler_path = model_path / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature names
        features_path = model_path / "features.pkl"
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
    
    def load_models(self, model_dir: str) -> None:
        """
        Load trained models from disk
        
        Args:
            model_dir: Directory containing saved models
        """
        model_path = Path(model_dir)
        
        target_names = ['support_level', 'resistance_level', 'breakout_probability']
        
        for target_name in target_names:
            filepath = model_path / f"{target_name}_model.pkl"
            if filepath.exists():
                with open(filepath, 'rb') as f:
                    self.models[target_name] = pickle.load(f)
        
        # Load scaler
        scaler_path = model_path / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        # Load feature names
        features_path = model_path / "features.pkl"
        if features_path.exists():
            with open(features_path, 'rb') as f:
                self.feature_names = pickle.load(f)
        
        logger.info(f"Loaded {len(self.models)} models from {model_dir}")
    
    def get_training_summary(self) -> str:
        """
        Get summary of training results
        
        Returns:
            Formatted string summary
        """
        summary = "\n" + "="*60 + "\n"
        summary += "TRAINING SUMMARY\n"
        summary += "="*60 + "\n\n"
        
        for target_name, metrics in self.training_history.items():
            summary += f"{target_name.upper()}\n"
            summary += "-" * 40 + "\n"
            for key, value in metrics.items():
                if isinstance(value, float):
                    summary += f"{key:.<30} {value:.6f}\n"
                else:
                    summary += f"{key:.<30} {value}\n"
            summary += "\n"
        
        summary += "="*60 + "\n"
        return summary
