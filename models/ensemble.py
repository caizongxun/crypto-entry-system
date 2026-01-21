import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import logging
import joblib
from pathlib import Path

import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

logger = logging.getLogger(__name__)

class EnsembleModel:
    """Ensemble model combining XGBoost, LightGBM, Neural Network, and Logistic Regression."""
    
    def __init__(self, model_dir: Path = None):
        self.model_dir = Path(model_dir) if model_dir else Path('models/artifacts')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.xgb_model = None
        self.lgb_model = None
        self.nn_model = None
        self.lr_model = None
        
        self.ensemble_weights = {'xgb': 0.35, 'lgb': 0.35, 'nn': 0.20, 'lr': 0.10}
        self.scaler = StandardScaler()
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series, params: Dict = None):
        """Train XGBoost classifier."""
        if params is None:
            params = {
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'seed': 42,
            }
        
        logger.info("Training XGBoost...")
        self.xgb_model = xgb.XGBClassifier(**params)
        self.xgb_model.fit(X_train, y_train)
        
        train_pred = self.xgb_model.predict_proba(X_train)[:, 1]
        train_acc = roc_auc_score(y_train, train_pred)
        logger.info(f"XGBoost train AUC: {train_acc:.4f}")
    
    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series, params: Dict = None):
        """Train LightGBM classifier."""
        if params is None:
            params = {
                'num_leaves': 31,
                'learning_rate': 0.05,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_data_in_leaf': 20,
                'objective': 'binary',
                'metric': 'auc',
                'seed': 42,
            }
        
        logger.info("Training LightGBM...")
        self.lgb_model = lgb.LGBMClassifier(**params)
        self.lgb_model.fit(X_train, y_train)
        
        train_pred = self.lgb_model.predict_proba(X_train)[:, 1]
        train_acc = roc_auc_score(y_train, train_pred)
        logger.info(f"LightGBM train AUC: {train_acc:.4f}")
    
    def train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray, config: Dict = None):
        """Train neural network classifier."""
        if config is None:
            config = {
                'lstm_units': 128,
                'lstm_layers': 2,
                'dense_units': 64,
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'early_stopping_patience': 10,
            }
        
        logger.info("Training Neural Network...")
        
        X_train_3d = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        
        self.nn_model = Sequential()
        
        # LSTM layers
        for i in range(config['lstm_layers']):
            if i == 0:
                self.nn_model.add(Bidirectional(LSTM(
                    config['lstm_units'],
                    return_sequences=(i < config['lstm_layers'] - 1),
                    input_shape=(X_train_3d.shape[1], 1)
                )))
            else:
                self.nn_model.add(Bidirectional(LSTM(
                    config['lstm_units'],
                    return_sequences=(i < config['lstm_layers'] - 1)
                )))
            self.nn_model.add(Dropout(config['dropout_rate']))
        
        # Dense layers
        for i in range(2):
            self.nn_model.add(Dense(config['dense_units'], activation='relu'))
            self.nn_model.add(Dropout(config['dropout_rate']))
        
        self.nn_model.add(Dense(1, activation='sigmoid'))
        
        self.nn_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=config['early_stopping_patience'],
            restore_best_weights=True
        )
        
        self.nn_model.fit(
            X_train_3d, y_train,
            batch_size=config['batch_size'],
            epochs=config['epochs'],
            validation_split=config.get('validation_split', 0.2),
            callbacks=[early_stop],
            verbose=0
        )
        
        logger.info("Neural Network training completed")
    
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train Logistic Regression baseline."""
        logger.info("Training Logistic Regression...")
        self.lr_model = LogisticRegression(max_iter=1000, random_state=42)
        self.lr_model.fit(X_train, y_train)
        
        train_pred = self.lr_model.predict_proba(X_train)[:, 1]
        train_acc = roc_auc_score(y_train, train_pred)
        logger.info(f"Logistic Regression train AUC: {train_acc:.4f}")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions.
        
        Returns:
            Probability array (n_samples, 2)
        """
        predictions = {}
        
        if self.xgb_model is not None:
            predictions['xgb'] = self.xgb_model.predict_proba(X)[:, 1]
        
        if self.lgb_model is not None:
            predictions['lgb'] = self.lgb_model.predict_proba(X)[:, 1]
        
        if self.nn_model is not None:
            X_3d = X.values.reshape((X.shape[0], X.shape[1], 1))
            predictions['nn'] = self.nn_model.predict(X_3d, verbose=0).flatten()
        
        if self.lr_model is not None:
            predictions['lr'] = self.lr_model.predict_proba(X)[:, 1]
        
        # Weighted ensemble
        ensemble_pred = np.zeros(len(X))
        total_weight = 0
        
        for model_name, pred in predictions.items():
            weight = self.ensemble_weights.get(model_name, 0)
            ensemble_pred += pred * weight
            total_weight += weight
        
        ensemble_pred /= total_weight
        
        return np.column_stack([1 - ensemble_pred, ensemble_pred])
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate binary predictions."""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate model performance."""
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
        }
        
        logger.info(f"Test Metrics: {metrics}")
        return metrics
    
    def save(self, path: Path = None):
        """Save all model components."""
        if path is None:
            path = self.model_dir / 'ensemble_model'
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self.xgb_model:
            joblib.dump(self.xgb_model, path / 'xgb_model.pkl')
        if self.lgb_model:
            joblib.dump(self.lgb_model, path / 'lgb_model.pkl')
        if self.nn_model:
            self.nn_model.save(str(path / 'nn_model.keras'))
        if self.lr_model:
            joblib.dump(self.lr_model, path / 'lr_model.pkl')
        
        joblib.dump(self.ensemble_weights, path / 'weights.pkl')
        logger.info(f"Models saved to {path}")
    
    def load(self, path: Path = None):
        """Load all model components."""
        if path is None:
            path = self.model_dir / 'ensemble_model'
        
        path = Path(path)
        
        if (path / 'xgb_model.pkl').exists():
            self.xgb_model = joblib.load(path / 'xgb_model.pkl')
            logger.info("XGBoost model loaded")
        
        if (path / 'lgb_model.pkl').exists():
            self.lgb_model = joblib.load(path / 'lgb_model.pkl')
            logger.info("LightGBM model loaded")
        
        if (path / 'nn_model.keras').exists():
            from tensorflow.keras.models import load_model
            self.nn_model = load_model(str(path / 'nn_model.keras'))
            logger.info("Neural Network model loaded")
        
        if (path / 'lr_model.pkl').exists():
            self.lr_model = joblib.load(path / 'lr_model.pkl')
            logger.info("Logistic Regression model loaded")
        
        if (path / 'weights.pkl').exists():
            self.ensemble_weights = joblib.load(path / 'weights.pkl')
        
        logger.info(f"Models loaded from {path}")
