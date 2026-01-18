"""
Model ensemble module for Strategy V3.

Implements XGBoost multi-output regression for support/resistance/breakout prediction.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Any
from pathlib import Path
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from loguru import logger

from .config import StrategyConfig


class ModelEnsemble:
    """
    Ensemble of XGBoost models for multi-output regression.
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize ModelEnsemble.

        Args:
            config: Strategy configuration
        """
        self.config = config
        self.models = {}
        self.scalers = {}
        self.selected_features = []
        self.feature_importance = {}

    def train(
        self,
        X_train: pd.DataFrame,
        y_train_support: pd.Series,
        y_train_resistance: pd.Series,
        y_train_breakout: pd.Series,
        X_test: pd.DataFrame = None,
        y_test_support: pd.Series = None,
        y_test_resistance: pd.Series = None,
        y_test_breakout: pd.Series = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Train ensemble models.

        Args:
            X_train: Training features
            y_train_support: Target support levels
            y_train_resistance: Target resistance levels
            y_train_breakout: Target breakout probability
            X_test: Test features (optional)
            y_test_support: Test support levels (optional)
            y_test_resistance: Test resistance levels (optional)
            y_test_breakout: Test breakout probability (optional)

        Returns:
            Dictionary of metrics for each model
        """
        logger.info('Training XGBoost ensemble models...')

        # Store feature names
        self.selected_features = list(X_train.columns)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        X_scaled = pd.DataFrame(X_scaled, columns=X_train.columns)
        self.scalers['features'] = scaler

        # Scale test features if provided
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

        metrics = {}

        # Train support model
        logger.info('Training support prediction model...')
        self.models['support'] = self._train_single_model(
            X_scaled, y_train_support, 'support'
        )
        if X_test_scaled is not None:
            train_metrics = self._evaluate_model(
                self.models['support'], X_scaled, y_train_support
            )
            test_metrics = self._evaluate_model(
                self.models['support'], X_test_scaled, y_test_support
            )
            metrics['support'] = {
                'train_rmse': train_metrics['rmse'],
                'train_mae': train_metrics['mae'],
                'train_r2': train_metrics['r2'],
                'test_rmse': test_metrics['rmse'],
                'test_mae': test_metrics['mae'],
                'test_r2': test_metrics['r2']
            }
        else:
            metrics['support'] = self._evaluate_model(
                self.models['support'], X_scaled, y_train_support
            )

        # Train resistance model
        logger.info('Training resistance prediction model...')
        self.models['resistance'] = self._train_single_model(
            X_scaled, y_train_resistance, 'resistance'
        )
        if X_test_scaled is not None:
            train_metrics = self._evaluate_model(
                self.models['resistance'], X_scaled, y_train_resistance
            )
            test_metrics = self._evaluate_model(
                self.models['resistance'], X_test_scaled, y_test_resistance
            )
            metrics['resistance'] = {
                'train_rmse': train_metrics['rmse'],
                'train_mae': train_metrics['mae'],
                'train_r2': train_metrics['r2'],
                'test_rmse': test_metrics['rmse'],
                'test_mae': test_metrics['mae'],
                'test_r2': test_metrics['r2']
            }
        else:
            metrics['resistance'] = self._evaluate_model(
                self.models['resistance'], X_scaled, y_train_resistance
            )

        # Train breakout model
        logger.info('Training breakout probability model...')
        self.models['breakout'] = self._train_single_model(
            X_scaled, y_train_breakout, 'breakout'
        )
        if X_test_scaled is not None:
            train_metrics = self._evaluate_model(
                self.models['breakout'], X_scaled, y_train_breakout
            )
            test_metrics = self._evaluate_model(
                self.models['breakout'], X_test_scaled, y_test_breakout
            )
            metrics['breakout'] = {
                'train_rmse': train_metrics['rmse'],
                'train_mae': train_metrics['mae'],
                'train_r2': train_metrics['r2'],
                'test_rmse': test_metrics['rmse'],
                'test_mae': test_metrics['mae'],
                'test_r2': test_metrics['r2']
            }
        else:
            metrics['breakout'] = self._evaluate_model(
                self.models['breakout'], X_scaled, y_train_breakout
            )

        logger.info('Training completed')
        return metrics

    def _train_single_model(self, X: pd.DataFrame, y: pd.Series, model_name: str) -> xgb.XGBRegressor:
        """
        Train a single XGBoost model.

        Args:
            X: Training features
            y: Target variable
            model_name: Name for logging

        Returns:
            Trained XGBoost model
        """
        model = xgb.XGBRegressor(
            max_depth=self.config.model.max_depth,
            learning_rate=self.config.model.learning_rate,
            n_estimators=self.config.model.n_estimators,
            subsample=self.config.model.subsample,
            colsample_bytree=self.config.model.colsample_bytree,
            min_child_weight=self.config.model.min_child_weight,
            gamma=self.config.model.gamma,
            reg_alpha=self.config.model.reg_alpha,
            reg_lambda=self.config.model.reg_lambda,
            random_state=self.config.model.seed,
            n_jobs=self.config.num_threads,
            verbosity=0
        )

        # Train model
        model.fit(X, y, verbose=False)

        # Store feature importance
        self.feature_importance[model_name] = model.feature_importances_

        return model

    def _evaluate_model(self, model: xgb.XGBRegressor, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            model: Trained model
            X: Test features
            y: Test targets

        Returns:
            Dictionary of metrics
        """
        y_pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions using ensemble.

        Args:
            X: Features DataFrame

        Returns:
            Tuple of (support, resistance, breakout_probability)
        """
        # Scale features
        X_scaled = self.scalers['features'].transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        # Generate predictions
        support = self.models['support'].predict(X_scaled)
        resistance = self.models['resistance'].predict(X_scaled)
        breakout = self.models['breakout'].predict(X_scaled)

        # Clip breakout to [0, 1]
        breakout = np.clip(breakout, 0, 1)

        return support, resistance, breakout

    def save_models(self, save_dir: str) -> None:
        """
        Save trained models to disk.

        Args:
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)

        # Save each model
        for model_name, model in self.models.items():
            model_path = os.path.join(save_dir, f'{model_name}_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f'Saved {model_name} model to {model_path}')

        # Save scalers
        scaler_path = os.path.join(save_dir, 'scalers.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scalers, f)

        # Save feature names
        features_path = os.path.join(save_dir, 'features.pkl')
        with open(features_path, 'wb') as f:
            pickle.dump(self.selected_features, f)

        logger.info(f'Models saved to {save_dir}')

    def load_models(self, save_dir: str) -> None:
        """
        Load trained models from disk.

        Args:
            save_dir: Directory containing saved models
        """
        # Load models
        for model_name in ['support', 'resistance', 'breakout']:
            model_path = os.path.join(save_dir, f'{model_name}_model.pkl')
            with open(model_path, 'rb') as f:
                self.models[model_name] = pickle.load(f)
            logger.info(f'Loaded {model_name} model from {model_path}')

        # Load scalers
        scaler_path = os.path.join(save_dir, 'scalers.pkl')
        with open(scaler_path, 'rb') as f:
            self.scalers = pickle.load(f)

        # Load feature names
        features_path = os.path.join(save_dir, 'features.pkl')
        with open(features_path, 'rb') as f:
            self.selected_features = pickle.load(f)

        logger.info(f'Models loaded from {save_dir}')

    def get_feature_importance(self, model_name: str, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance for a model.

        Args:
            model_name: Name of the model
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.feature_importance:
            return pd.DataFrame()

        importance = self.feature_importance[model_name]
        feature_names = self.selected_features

        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)

        return df
