import numpy as np
from typing import Dict, Tuple, List, Callable
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from optuna import create_study, Trial
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class OptimizationMetric(Enum):
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    BALANCED = "balanced"


@dataclass
class HyperparameterSpace:
    """Define search space for hyperparameters."""
    max_depth: Tuple[int, int] = (5, 12)
    learning_rate: Tuple[float, float] = (0.01, 0.2)
    n_estimators: Tuple[int, int] = (50, 300)
    min_child_weight: Tuple[int, int] = (1, 10)
    subsample: Tuple[float, float] = (0.6, 1.0)
    colsample_bytree: Tuple[float, float] = (0.6, 1.0)
    reg_lambda: Tuple[float, float] = (0.0, 5.0)
    reg_alpha: Tuple[float, float] = (0.0, 5.0)


@dataclass
class OptimizationResult:
    """Store optimization results."""
    best_params: Dict
    best_score: float
    metric: str
    train_precision: float
    train_recall: float
    test_precision: float
    test_recall: float
    test_f1: float
    trials_count: int

    def to_dict(self) -> Dict:
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'metric': self.metric,
            'train_precision': self.train_precision,
            'train_recall': self.train_recall,
            'test_precision': self.test_precision,
            'test_recall': self.test_recall,
            'test_f1': self.test_f1,
            'trials_count': self.trials_count,
        }


class HyperparameterOptimizer:
    """Optimize hyperparameters for ML models."""

    def __init__(self, model_type: str = 'xgboost', metric: OptimizationMetric = OptimizationMetric.BALANCED):
        self.model_type = model_type
        self.metric = metric
        self.search_space = HyperparameterSpace()
        self.best_params = None
        self.best_score = None
        self.optimization_history = []

    def _calculate_f1_score(self, precision: float, recall: float) -> float:
        """Calculate F1 score from precision and recall."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def _calculate_precision(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate precision."""
        true_positives = ((y_pred == 1) & (y_true == 1)).sum()
        false_positives = ((y_pred == 1) & (y_true == 0)).sum()
        if true_positives + false_positives == 0:
            return 0.0
        return true_positives / (true_positives + false_positives)

    def _calculate_recall(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate recall."""
        true_positives = ((y_pred == 1) & (y_true == 1)).sum()
        false_negatives = ((y_pred == 0) & (y_true == 1)).sum()
        if true_positives + false_negatives == 0:
            return 0.0
        return true_positives / (true_positives + false_negatives)

    def _objective_function(self, trial: Trial, X: np.ndarray, y: np.ndarray,
                           scale_pos_weight: float) -> float:
        """Objective function for Optuna optimization."""
        max_depth = trial.suggest_int('max_depth', self.search_space.max_depth[0], 
                                      self.search_space.max_depth[1])
        learning_rate = trial.suggest_float('learning_rate', self.search_space.learning_rate[0],
                                           self.search_space.learning_rate[1])
        n_estimators = trial.suggest_int('n_estimators', self.search_space.n_estimators[0],
                                        self.search_space.n_estimators[1])

        if self.model_type == 'xgboost':
            min_child_weight = trial.suggest_int('min_child_weight', 
                                                self.search_space.min_child_weight[0],
                                                self.search_space.min_child_weight[1])
            subsample = trial.suggest_float('subsample', self.search_space.subsample[0],
                                           self.search_space.subsample[1])
            colsample_bytree = trial.suggest_float('colsample_bytree', 
                                                  self.search_space.colsample_bytree[0],
                                                  self.search_space.colsample_bytree[1])
            reg_lambda = trial.suggest_float('reg_lambda', self.search_space.reg_lambda[0],
                                            self.search_space.reg_lambda[1])
            reg_alpha = trial.suggest_float('reg_alpha', self.search_space.reg_alpha[0],
                                           self.search_space.reg_alpha[1])

            model = XGBClassifier(
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_lambda=reg_lambda,
                reg_alpha=reg_alpha,
                random_state=42,
                eval_metric='logloss',
                scale_pos_weight=scale_pos_weight,
                verbosity=0
            )
        elif self.model_type == 'lightgbm':
            model = LGBMClassifier(
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                random_state=42,
                verbose=-1,
                scale_pos_weight=scale_pos_weight
            )
        elif self.model_type == 'random_forest':
            model = RandomForestClassifier(
                max_depth=max_depth,
                n_estimators=n_estimators,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                             random_state=42, stratify=y)

        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)

        precision = self._calculate_precision(y_pred_test, y_test)
        recall = self._calculate_recall(y_pred_test, y_test)
        f1 = self._calculate_f1_score(precision, recall)

        if self.metric == OptimizationMetric.PRECISION:
            score = precision
        elif self.metric == OptimizationMetric.RECALL:
            score = recall
        elif self.metric == OptimizationMetric.F1_SCORE:
            score = f1
        elif self.metric == OptimizationMetric.BALANCED:
            score = 0.6 * precision + 0.4 * recall
        else:
            score = f1

        return score

    def optimize(self, X: np.ndarray, y: np.ndarray, n_trials: int = 50,
                verbose: bool = True) -> OptimizationResult:
        """Run hyperparameter optimization using Optuna."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not installed. Install with: pip install optuna")

        scale_pos_weight = (y == 0).sum() / (y == 1).sum()

        sampler = TPESampler(seed=42)
        study = create_study(sampler=sampler, direction='maximize')

        def objective_wrapper(trial):
            return self._objective_function(trial, X, y, scale_pos_weight)

        if verbose:
            print(f"Starting hyperparameter optimization...")
            print(f"Model: {self.model_type}")
            print(f"Metric: {self.metric.value}")
            print(f"Trials: {n_trials}")
            print()

        study.optimize(objective_wrapper, n_trials=n_trials, show_progress_bar=verbose)

        if verbose:
            print(f"\nOptimization completed. Best score: {study.best_value:.4f}")

        self.best_params = study.best_params
        self.best_score = study.best_value
        self.optimization_history = study.trials

        result = self._evaluate_best_params(X, y, self.best_params, scale_pos_weight)
        return result

    def _evaluate_best_params(self, X: np.ndarray, y: np.ndarray,
                            params: Dict, scale_pos_weight: float) -> OptimizationResult:
        """Evaluate best parameters on full dataset."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                             random_state=42, stratify=y)

        if self.model_type == 'xgboost':
            model = XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                scale_pos_weight=scale_pos_weight,
                verbosity=0,
                **params
            )
        elif self.model_type == 'lightgbm':
            model = LGBMClassifier(
                random_state=42,
                verbose=-1,
                scale_pos_weight=scale_pos_weight,
                **params
            )
        elif self.model_type == 'random_forest':
            model = RandomForestClassifier(
                random_state=42,
                class_weight='balanced',
                n_jobs=-1,
                **params
            )

        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_precision = self._calculate_precision(y_pred_train, y_train)
        train_recall = self._calculate_recall(y_pred_train, y_train)
        test_precision = self._calculate_precision(y_pred_test, y_test)
        test_recall = self._calculate_recall(y_pred_test, y_test)
        test_f1 = self._calculate_f1_score(test_precision, test_recall)

        return OptimizationResult(
            best_params=params,
            best_score=self.best_score,
            metric=self.metric.value,
            train_precision=train_precision,
            train_recall=train_recall,
            test_precision=test_precision,
            test_recall=test_recall,
            test_f1=test_f1,
            trials_count=len(self.optimization_history) if self.optimization_history else 0,
        )

    def save_results(self, filepath: str) -> None:
        """Save optimization results to JSON file."""
        if self.best_params is None:
            print("No optimization results to save. Run optimize() first.")
            return

        result_data = {
            'best_params': self.best_params,
            'best_score': float(self.best_score),
            'metric': self.metric.value,
            'model_type': self.model_type,
            'timestamp': pd.Timestamp.now().isoformat() if 'pd' in dir() else str(pd.Timestamp.now()),
        }

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(result_data, f, indent=2)

        print(f"Results saved to {filepath}")

    def load_results(self, filepath: str) -> Dict:
        """Load optimization results from JSON file."""
        path = Path(filepath)
        if not path.exists():
            print(f"Results file not found: {filepath}")
            return {}

        with open(path, 'r') as f:
            data = json.load(f)

        self.best_params = data.get('best_params', {})
        self.best_score = data.get('best_score', 0.0)

        print(f"Results loaded from {filepath}")
        return data

    def print_results_summary(self, result: OptimizationResult) -> None:
        """Print formatted results summary."""
        print("\n" + "="*60)
        print("HYPERPARAMETER OPTIMIZATION RESULTS")
        print("="*60)
        print(f"Model Type: {self.model_type}")
        print(f"Optimization Metric: {result.metric}")
        print(f"Total Trials: {result.trials_count}")
        print(f"Best Score: {result.best_score:.4f}")
        print("\nBest Parameters:")
        for key, value in result.best_params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print("\nPerformance Metrics:")
        print(f"  Train Precision: {result.train_precision:.4f}")
        print(f"  Train Recall: {result.train_recall:.4f}")
        print(f"  Test Precision: {result.test_precision:.4f}")
        print(f"  Test Recall: {result.test_recall:.4f}")
        print(f"  Test F1 Score: {result.test_f1:.4f}")
        print("="*60)


import pandas as pd
