import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


class FeatureSelector:
    """Select most important features to improve model precision."""

    def __init__(self, n_features: int = 25):
        self.n_features = n_features
        self.selected_features = None
        self.feature_importance = None
        self.selector = None

    def select_by_mutual_information(self, X: np.ndarray, y: np.ndarray,
                                     feature_names: List[str] = None) -> Tuple[np.ndarray, List[str]]:
        """Select features using mutual information."""
        print(f"Selecting top {self.n_features} features using mutual information...")
        
        selector = SelectKBest(score_func=mutual_info_classif, k=min(self.n_features, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        scores = selector.scores_
        selected_indices = selector.get_support(indices=True)
        
        if feature_names is not None:
            selected_names = [feature_names[i] for i in selected_indices]
        else:
            selected_names = [f"feature_{i}" for i in selected_indices]
        
        self.selected_features = selected_names
        self.feature_importance = dict(zip(selected_names, scores[selected_indices]))
        self.selector = selector
        
        print(f"Selected {len(selected_names)} features")
        self._print_feature_importance(self.feature_importance)
        
        return X_selected, selected_names

    def select_by_random_forest(self, X: np.ndarray, y: np.ndarray,
                               feature_names: List[str] = None) -> Tuple[np.ndarray, List[str]]:
        """Select features using random forest importance."""
        print(f"Selecting top {self.n_features} features using random forest importance...")
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        importances = rf.feature_importances_
        
        if feature_names is not None:
            feature_importance_dict = dict(zip(feature_names, importances))
        else:
            feature_importance_dict = {f"feature_{i}": imp for i, imp in enumerate(importances)}
        
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:self.n_features]
        
        selected_names = [f[0] for f in top_features]
        selected_indices = [feature_names.index(f[0]) for f in top_features] if feature_names is not None else None
        
        if selected_indices is not None:
            X_selected = X[:, selected_indices]
        else:
            X_selected = X[:, :self.n_features]
        
        self.selected_features = selected_names
        self.feature_importance = dict(top_features)
        
        print(f"Selected {len(selected_names)} features")
        self._print_feature_importance(self.feature_importance)
        
        return X_selected, selected_names

    def select_by_variance(self, X: np.ndarray, y: np.ndarray,
                          feature_names: List[str] = None) -> Tuple[np.ndarray, List[str]]:
        """Select features by variance and correlation with target."""
        print(f"Selecting top {self.n_features} features using variance and correlation...")
        
        variances = np.var(X, axis=0)
        correlations = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
        correlations = np.nan_to_num(correlations, nan=0.0)
        
        combined_scores = (variances / (np.max(variances) + 1e-6)) * 0.5 + (correlations / (np.max(correlations) + 1e-6)) * 0.5
        
        top_indices = np.argsort(combined_scores)[-self.n_features:][::-1]
        X_selected = X[:, top_indices]
        
        if feature_names is not None:
            selected_names = [feature_names[i] for i in top_indices]
        else:
            selected_names = [f"feature_{i}" for i in top_indices]
        
        self.selected_features = selected_names
        self.feature_importance = dict(zip(selected_names, combined_scores[top_indices]))
        
        print(f"Selected {len(selected_names)} features")
        self._print_feature_importance(self.feature_importance)
        
        return X_selected, selected_names

    def _print_feature_importance(self, importance_dict: dict):
        """Print top features with their importance scores."""
        print("\nTop selected features:")
        for i, (feature, score) in enumerate(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10], 1):
            print(f"  {i}. {feature}: {score:.4f}")

    def transform(self, X: np.ndarray, feature_names: List[str] = None) -> np.ndarray:
        """Transform data using selected features."""
        if self.selected_features is None:
            raise ValueError("No features selected. Call select_* method first.")
        
        if feature_names is None:
            raise ValueError("Feature names required for transform")
        
        indices = [feature_names.index(f) for f in self.selected_features]
        return X[:, indices]

    def get_selected_features(self) -> List[str]:
        """Get list of selected features."""
        return self.selected_features

    def get_feature_importance(self) -> dict:
        """Get feature importance dictionary."""
        return self.feature_importance
