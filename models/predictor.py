import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import logging

from data.loader import KlinesDataLoader
from data.preprocessor import DataPreprocessor
from models.ensemble import EnsembleModel

logger = logging.getLogger(__name__)

class UnifiedPredictor:
    """Unified interface for making predictions on new data."""
    
    def __init__(self, model_dir: Path = None, scaler_path: Path = None):
        self.model_dir = Path(model_dir) if model_dir else Path('models/artifacts')
        self.scaler_path = Path(scaler_path) if scaler_path else Path('models/artifacts/scaler.pkl')
        
        self.ensemble = EnsembleModel(self.model_dir)
        self.ensemble.load()
        
        self.preprocessor = DataPreprocessor()
        self.data_loader = KlinesDataLoader()
        
        import joblib
        if self.scaler_path.exists():
            self.preprocessor.scaler = joblib.load(self.scaler_path)
    
    def predict_single_candle(self, df: pd.DataFrame) -> Dict:
        """Predict for the last candle in the dataframe.
        
        Args:
            df: Raw OHLCV DataFrame
        
        Returns:
            Dict with prediction results
        """
        # Preprocess
        X, _ = self.preprocessor.preprocess(df, create_target=False, normalize=True)
        
        # Get last row
        X_last = X.iloc[-1:]
        
        # Predict
        proba = self.ensemble.predict_proba(X_last)
        pred_label = self.ensemble.predict(X_last)[0]
        
        prob_up = proba[0, 1]
        prob_down = proba[0, 0]
        
        # Determine signal strength
        max_prob = max(prob_up, prob_down)
        if max_prob > 0.75:
            strength = 'VERY_STRONG'
        elif max_prob > 0.65:
            strength = 'STRONG'
        elif max_prob > 0.55:
            strength = 'MODERATE'
        else:
            strength = 'WEAK'
        
        result = {
            'direction': 'UP' if pred_label == 1 else 'DOWN',
            'confidence': max_prob,
            'probability_up': prob_up,
            'probability_down': prob_down,
            'signal_strength': strength,
            'timestamp': df['close_time'].iloc[-1],
        }
        
        return result
    
    def predict_next_candle(
        self,
        symbol: str,
        timeframe: str,
        use_cache: bool = True
    ) -> Dict:
        """Load data and predict for next candle.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: '15m', '1h', or '1d'
            use_cache: Whether to use cached data
        
        Returns:
            Dict with prediction results
        """
        # Load data
        df = self.data_loader.load_klines(symbol, timeframe, use_cache=use_cache)
        
        # Predict
        result = self.predict_single_candle(df)
        result['symbol'] = symbol
        result['timeframe'] = timeframe
        
        return result
    
    def predict_multiple_candles(
        self,
        df: pd.DataFrame,
        step: int = 1
    ) -> pd.DataFrame:
        """Generate predictions for multiple candles.
        
        Args:
            df: Raw OHLCV DataFrame
            step: Step size for generating predictions
        
        Returns:
            DataFrame with predictions for each candle
        """
        # Preprocess
        X, _ = self.preprocessor.preprocess(df, create_target=False, normalize=True)
        
        # Predict for all candles
        proba = self.ensemble.predict_proba(X)
        
        results = pd.DataFrame({
            'timestamp': df['close_time'].iloc[len(df) - len(X):].values,
            'probability_up': proba[:, 1],
            'probability_down': proba[:, 0],
            'direction': np.where(proba[:, 1] > 0.5, 'UP', 'DOWN'),
        })
        
        # Add confidence
        results['confidence'] = results[['probability_up', 'probability_down']].max(axis=1)
        
        # Add signal strength
        results['signal_strength'] = pd.cut(
            results['confidence'],
            bins=[0, 0.55, 0.65, 0.75, 1.0],
            labels=['WEAK', 'MODERATE', 'STRONG', 'VERY_STRONG']
        )
        
        return results[::step].reset_index(drop=True)
    
    def backtest_predictions(
        self,
        df: pd.DataFrame,
        min_confidence: float = 0.60
    ) -> Dict:
        """Backtest predictions against actual price movements.
        
        Args:
            df: Raw OHLCV DataFrame
            min_confidence: Minimum confidence threshold for signals
        
        Returns:
            Dict with backtest metrics
        """
        predictions = self.predict_multiple_candles(df)
        
        # Filter by confidence
        signals = predictions[predictions['confidence'] >= min_confidence].copy()
        
        # Get actual price movements
        df_aligned = df.iloc[len(df) - len(predictions):].reset_index(drop=True)
        df_aligned['next_close'] = df_aligned['close'].shift(-1)
        
        # Calculate actual direction
        df_aligned['actual_direction'] = np.where(
            df_aligned['next_close'] > df_aligned['close'],
            'UP',
            'DOWN'
        )
        
        # Align signals with actual
        idx_valid = predictions.index.isin(signals.index)
        df_valid = df_aligned[idx_valid].copy()
        signals_valid = signals[signals.index.isin(df_valid.index)].copy()
        
        # Calculate metrics
        if len(signals_valid) == 0:
            return {'error': 'No signals generated'}
        
        correct = (signals_valid['direction'].values == df_valid['actual_direction'].values).sum()
        accuracy = correct / len(signals_valid)
        
        metrics = {
            'total_signals': len(signals_valid),
            'accuracy': accuracy,
            'correct_predictions': int(correct),
            'avg_confidence': signals_valid['confidence'].mean(),
            'min_confidence': signals_valid['confidence'].min(),
            'max_confidence': signals_valid['confidence'].max(),
        }
        
        logger.info(f"Backtest Results: {metrics}")
        return metrics
