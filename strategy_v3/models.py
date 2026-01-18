"""
Model module for Strategy V3.

XGBoost-based multi-output regression for support/resistance prediction.
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from loguru import logger
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelEnsemble:
    """
    Ensemble of XGBoost regressors for predicting support, resistance, and breakout probability.
    """

    def __init__(self, config):
        """
        Initialize ModelEnsemble.

        Args:
            config: StrategyConfig object
        """
        self.cfg = config
        self.model_config = config.model
        self.verbose = config.verbose

        # Initialize individual models
        self.support_model = None
        self.resistance_model = None
        self.breakout_model = None
        self.scaler = None
        self.selected_features = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train_support: pd.Series,
        y_train_resistance: pd.Series,
        y_train_breakout: pd.Series,
        X_val: pd.DataFrame = None,
        y_val_support: pd.Series = None,
        y_val_resistance: pd.Series = None,
        y_val_breakout: pd.Series = None
    ) -> Dict[str, any]:
        """
        Train all models.

        Args:
            X_train: Training features
            y_train_support: Training support targets
            y_train_resistance: Training resistance targets
            y_train_breakout: Training breakout targets
            X_val: Validation features (optional)
            y_val_*: Validation targets

        Returns:
            Dictionary with training metrics
        """
        if self.verbose:
            logger.info('Starting model training...')

        # Feature selection and scaling
        self.selected_features = self._select_features(X_train, k=25)
        X_train_selected = X_train[self.selected_features]

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_selected)

        metrics = {}

        # Prepare eval set
        eval_set = None
        if X_val is not None:
            X_val_selected = X_val[self.selected_features]
            X_val_scaled = self.scaler.transform(X_val_selected)
            eval_set = [(X_val_scaled, y_val_support)]

        # Train support model
        if self.verbose:
            logger.info('Training support level model...')
        self.support_model = XGBRegressor(
            max_depth=self.model_config.max_depth,
            learning_rate=self.model_config.learning_rate,
            n_estimators=self.model_config.n_estimators,
            subsample=self.model_config.subsample,
            colsample_bytree=self.model_config.colsample_bytree,
            min_child_weight=self.model_config.min_child_weight,
            gamma=self.model_config.gamma,
            reg_alpha=self.model_config.reg_alpha,
            reg_lambda=self.model_config.reg_lambda,
            random_state=self.model_config.seed,
            n_jobs=-1,
            verbosity=0 if not self.verbose else 1,
        )

        self.support_model.fit(
            X_train_scaled,
            y_train_support,
            eval_set=eval_set,
            early_stopping_rounds=self.model_config.early_stopping_rounds if eval_set else None,
            verbose=False
        )

        # Evaluate support model
        y_pred_support = self.support_model.predict(X_train_scaled)
        metrics['support'] = {
            'rmse': np.sqrt(mean_squared_error(y_train_support, y_pred_support)),
            'mae': mean_absolute_error(y_train_support, y_pred_support),
            'r2': r2_score(y_train_support, y_pred_support),
        }

        # Train resistance model
        if self.verbose:
            logger.info('Training resistance level model...')
        self.resistance_model = XGBRegressor(
            max_depth=self.model_config.max_depth,
            learning_rate=self.model_config.learning_rate,
            n_estimators=self.model_config.n_estimators,
            subsample=self.model_config.subsample,
            colsample_bytree=self.model_config.colsample_bytree,
            min_child_weight=self.model_config.min_child_weight,
            gamma=self.model_config.gamma,
            reg_alpha=self.model_config.reg_alpha,
            reg_lambda=self.model_config.reg_lambda,
            random_state=self.model_config.seed,
            n_jobs=-1,
            verbosity=0 if not self.verbose else 1,
        )

        self.resistance_model.fit(
            X_train_scaled,
            y_train_resistance,
            eval_set=eval_set,
            early_stopping_rounds=self.model_config.early_stopping_rounds if eval_set else None,
            verbose=False
        )

        # Evaluate resistance model
        y_pred_resistance = self.resistance_model.predict(X_train_scaled)
        metrics['resistance'] = {
            'rmse': np.sqrt(mean_squared_error(y_train_resistance, y_pred_resistance)),
            'mae': mean_absolute_error(y_train_resistance, y_pred_resistance),
            'r2': r2_score(y_train_resistance, y_pred_resistance),
        }

        # Train breakout probability model
        if self.verbose:
            logger.info('Training breakout probability model...')
        self.breakout_model = XGBRegressor(
            max_depth=self.model_config.max_depth,
            learning_rate=self.model_config.learning_rate,
            n_estimators=self.model_config.n_estimators,
            subsample=self.model_config.subsample,
            colsample_bytree=self.model_config.colsample_bytree,
            min_child_weight=self.model_config.min_child_weight,
            gamma=self.model_config.gamma,
            reg_alpha=self.model_config.reg_alpha,
            reg_lambda=self.model_config.reg_lambda,
            random_state=self.model_config.seed,
            n_jobs=-1,
            verbosity=0 if not self.verbose else 1,
        )

        self.breakout_model.fit(
            X_train_scaled,
            y_train_breakout,
            eval_set=eval_set,
            early_stopping_rounds=self.model_config.early_stopping_rounds if eval_set else None,
            verbose=False
        )

        # Evaluate breakout model
        y_pred_breakout = self.breakout_model.predict(X_train_scaled)
        metrics['breakout'] = {
            'rmse': np.sqrt(mean_squared_error(y_train_breakout, y_pred_breakout)),
            'mae': mean_absolute_error(y_train_breakout, y_pred_breakout),
            'r2': r2_score(y_train_breakout, y_pred_breakout),
        }

        if self.verbose:
            logger.info('Model training completed')
            for model_name, model_metrics in metrics.items():
                logger.info(f'{model_name.upper()}: RMSE={model_metrics["rmse"]:.6f}, MAE={model_metrics["mae"]:.6f}')

        return metrics

    def predict(
        self,
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions.

        Args:
            X: Input features

        Returns:
            Tuple of (support, resistance, breakout_probability)
        """
        if self.scaler is None or self.selected_features is None:
            raise ValueError('Model must be trained before prediction')

        X_selected = X[self.selected_features]
        X_scaled = self.scaler.transform(X_selected)

        support = self.support_model.predict(X_scaled)
        resistance = self.resistance_model.predict(X_scaled)
        breakout = np.clip(self.breakout_model.predict(X_scaled), 0, 1)

        return support, resistance, breakout

    def save_models(self, save_dir: str):
        """
        Save trained models to disk.

        Args:
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)

        if self.support_model:
            with open(os.path.join(save_dir, 'support_level_model.pkl'), 'wb') as f:
                pickle.dump(self.support_model, f)

        if self.resistance_model:
            with open(os.path.join(save_dir, 'resistance_level_model.pkl'), 'wb') as f:
                pickle.dump(self.resistance_model, f)

        if self.breakout_model:
            with open(os.path.join(save_dir, 'breakout_probability_model.pkl'), 'wb') as f:
                pickle.dump(self.breakout_model, f)

        if self.scaler:
            with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
                pickle.dump(self.scaler, f)

        if self.selected_features:
            with open(os.path.join(save_dir, 'features.pkl'), 'wb') as f:
                pickle.dump(self.selected_features, f)

        if self.verbose:
            logger.info(f'Models saved to {save_dir}')

    def load_models(self, save_dir: str):
        """
        Load trained models from disk.

        Args:
            save_dir: Directory containing saved models
        """
        try:
            with open(os.path.join(save_dir, 'support_level_model.pkl'), 'rb') as f:
                self.support_model = pickle.load(f)

            with open(os.path.join(save_dir, 'resistance_level_model.pkl'), 'rb') as f:
                self.resistance_model = pickle.load(f)

            with open(os.path.join(save_dir, 'breakout_probability_model.pkl'), 'rb') as f:
                self.breakout_model = pickle.load(f)

            with open(os.path.join(save_dir, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)

            with open(os.path.join(save_dir, 'features.pkl'), 'rb') as f:
                self.selected_features = pickle.load(f)

            if self.verbose:
                logger.info(f'Models loaded from {save_dir}')

        except FileNotFoundError as e:
            logger.error(f'Failed to load models: {str(e)}')
            raise

    @staticmethod
    def _select_features(X: pd.DataFrame, k: int = 25) -> List[str]:
        """
        Select most important features using variance-based filtering and correlation analysis.

        Args:
            X: Feature DataFrame
            k: Number of features to select

        Returns:
            List of selected feature names
        """
        # Remove features with low variance
        variances = X.var()
        high_variance_features = variances[variances > variances.quantile(0.1)].index.tolist()

        # Take top k features by variance
        selected = sorted(high_variance_features, key=lambda f: variances[f], reverse=True)[:k]

        return selected
