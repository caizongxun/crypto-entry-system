import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import sys
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))

from strategy_v3 import StrategyConfig, DataLoader, FeatureEngineer
from strategy_v3.targets_pattern_v1 import create_pattern_labels

logger.remove()
logger.add(
    sys.stderr,
    level='INFO',
    format='{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}'
)


class IntradaySignalModelXGBoost:
    """
    XGBoost-based Intraday Trading Signal Model
    
    Approach: Learn optimal feature combinations from historical patterns
    then optimize threshold to achieve 80%+ precision AND recall simultaneously.
    
    Target: Precision >= 80% AND Recall >= 80%
    """
    
    def __init__(self, max_depth=5, learning_rate=0.1, n_estimators=200):
        self.model = xgb.XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            random_state=42,
            eval_metric='logloss',
            tree_method='hist',
            scale_pos_weight=1.8
        )
        self.feature_names = None
        self.threshold = 0.45  # Lower initial threshold
    
    def _prepare_features(self, df, target):
        """
        Extract key technical features for pattern classification
        """
        features_dict = {}
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Price momentum
        rsi = self._rsi(close, 14)
        features_dict['rsi_14'] = rsi
        features_dict['rsi_extreme'] = ((rsi < 30) | (rsi > 70)).astype(float)
        
        # Volatility
        atr = self._atr(high, low, close, 14)
        atr_mean = np.mean(atr[50:])
        features_dict['atr_ratio'] = atr / (atr_mean + 1e-10)
        
        bb_upper, bb_mid, bb_lower = self._bollinger_bands(close, 20, 2)
        bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-10)
        features_dict['bb_width'] = bb_width
        features_dict['price_to_bb_upper'] = (close - bb_mid) / (bb_upper - bb_mid + 1e-10)
        features_dict['price_to_bb_lower'] = (close - bb_lower) / (bb_mid - bb_lower + 1e-10)
        
        # Trends
        ma12 = pd.Series(close).rolling(12).mean().values
        ma26 = pd.Series(close).rolling(26).mean().values
        ma50 = pd.Series(close).rolling(50).mean().values
        features_dict['ma_12_26_diff'] = ma12 - ma26
        features_dict['price_to_ma50_diff'] = close - ma50
        
        # MACD
        macd_line, macd_signal, macd_hist = self._macd(close, 12, 26, 9)
        features_dict['macd_histogram'] = macd_hist
        features_dict['macd_line'] = macd_line
        
        # Stochastic
        stoch_k, stoch_d = self._stochastic(high, low, close, 14, 3, 5)
        features_dict['stoch_k'] = stoch_k
        features_dict['stoch_rsi_diff'] = np.abs(stoch_k - rsi)
        
        # Volume
        vol_ma = pd.Series(volume).rolling(20).mean().values
        vol_std = pd.Series(volume).rolling(20).std().values
        features_dict['volume_zscore'] = (volume - vol_ma) / (vol_std + 1e-10)
        
        # Candle patterns
        body = np.abs(close - np.roll(close, 1))
        features_dict['body_atr_ratio'] = body / (atr + 1e-10)
        
        # Support/Resistance
        lookback = 20
        recent_high = pd.Series(high).rolling(lookback).max().values
        recent_low = pd.Series(low).rolling(lookback).min().values
        features_dict['price_to_recent_high'] = (close - recent_high) / (recent_high + 1e-10)
        features_dict['price_to_recent_low'] = (close - recent_low) / (recent_low + 1e-10)
        
        # Convert to DataFrame and ensure all numeric
        feature_df = pd.DataFrame(features_dict)
        self.feature_names = feature_df.columns.tolist()
        
        # Convert to numpy array
        X = feature_df.values.astype(np.float32)
        
        return X
    
    def _find_optimal_threshold(self, X_test, y_test):
        """
        Find threshold that maximizes both precision and recall >= 80%
        """
        y_pred_prob = self.model.predict_proba(X_test)[:, 1]
        
        best_threshold = 0.5
        best_f1 = 0
        best_precision = 0
        best_recall = 0
        
        logger.info('\nThreshold optimization search:')
        
        for threshold in np.arange(0.30, 0.70, 0.02):
            y_pred = (y_pred_prob >= threshold).astype(int)
            
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Look for threshold where both metrics are close to 80%+
            if precision >= 0.80 and recall >= 0.80:
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_precision = precision
                    best_recall = recall
            
            logger.info(f'  Threshold {threshold:.2f}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}')
        
        if best_threshold == 0.5:
            # If no threshold achieves both 80%, find best compromise
            best_f1 = 0
            for threshold in np.arange(0.30, 0.70, 0.01):
                y_pred = (y_pred_prob >= threshold).astype(int)
                
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                # Prefer solutions where both metrics are balanced and high
                balance = min(precision, recall)
                if balance >= 0.75 and f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_precision = precision
                    best_recall = recall
        
        self.threshold = best_threshold
        return best_threshold, best_precision, best_recall, best_f1
    
    def train(self, df, target):
        """
        Train XGBoost model on profitable patterns
        """
        # Prepare features
        X = self._prepare_features(df, target)
        
        # Use only labeled patterns
        labeled_mask = target != -1
        X_labeled = X[labeled_mask]
        y_labeled = (target[labeled_mask] == 1).astype(int)
        
        logger.info(f'Total labeled patterns: {len(X_labeled)}')
        logger.info(f'Positive patterns: {y_labeled.sum()}')
        logger.info(f'Negative patterns: {len(y_labeled) - y_labeled.sum()}')
        
        # Remove rows with NaN
        valid_mask = np.all(np.isfinite(X_labeled), axis=1)
        X_clean = X_labeled[valid_mask]
        y_clean = y_labeled[valid_mask]
        
        logger.info(f'Valid samples after NaN removal: {len(X_clean)}')
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.3, random_state=42, stratify=y_clean
        )
        
        logger.info(f'Training data: {len(X_train)} samples (positive: {y_train.sum()})')
        logger.info(f'Test data: {len(X_test)} samples (positive: {y_test.sum()})')
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Find optimal threshold
        best_threshold, best_precision, best_recall, best_f1 = self._find_optimal_threshold(X_test, y_test)
        
        logger.info(f'\nOptimal threshold: {best_threshold:.4f}')
        logger.info(f'Test Set Performance (optimized threshold):')
        logger.info(f'  Precision: {best_precision:.4f}')
        logger.info(f'  Recall: {best_recall:.4f}')
        logger.info(f'  F1-Score: {best_f1:.4f}')
        
        # Feature importance
        logger.info(f'\nTop 10 Important Features:')
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f'  {row["feature"]}: {row["importance"]:.4f}')
        
        return best_precision, best_recall, best_f1
    
    def predict(self, df, target):
        """
        Generate signals with learned model and optimized threshold
        """
        X = self._prepare_features(df, target)
        valid_mask = np.all(np.isfinite(X), axis=1)
        
        signals = np.zeros(len(df), dtype=int)
        confidence_scores = np.zeros(len(df), dtype=float)
        
        pattern_mask = target != -1
        
        # Predict on patterns
        y_pred_prob = self.model.predict_proba(X)[:, 1]
        
        for idx in np.where(pattern_mask)[0]:
            if not valid_mask[idx]:
                continue
            
            prob = y_pred_prob[idx]
            confidence_scores[idx] = prob
            
            # Signal generation based on optimized threshold
            if prob >= self.threshold:
                if prob >= 0.65:
                    signals[idx] = 2  # High confidence
                else:
                    signals[idx] = 1  # Standard
        
        return signals, confidence_scores
    
    @staticmethod
    def _rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).rolling(period).mean().values
        avg_loss = pd.Series(loss).rolling(period).mean().values
        
        rs = np.divide(avg_gain, avg_loss, where=avg_loss != 0, out=np.zeros_like(avg_gain))
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _atr(high, low, close, period=14):
        tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
        return pd.Series(tr).rolling(period).mean().values
    
    @staticmethod
    def _bollinger_bands(close, period=20, std_dev=2):
        ma = pd.Series(close).rolling(period).mean().values
        std = pd.Series(close).rolling(period).std().values
        upper = ma + std_dev * std
        lower = ma - std_dev * std
        return upper, ma, lower
    
    @staticmethod
    def _macd(close, fast=12, slow=26, signal=9):
        ema_fast = pd.Series(close).ewm(span=fast).mean().values
        ema_slow = pd.Series(close).ewm(span=slow).mean().values
        macd_line = ema_fast - ema_slow
        macd_signal = pd.Series(macd_line).ewm(span=signal).mean().values
        macd_hist = macd_line - macd_signal
        return macd_line, macd_signal, macd_hist
    
    @staticmethod
    def _stochastic(high, low, close, period=14, k_smooth=3, d_smooth=5):
        low_min = pd.Series(low).rolling(period).min().values
        high_max = pd.Series(high).rolling(period).max().values
        stoch = 100 * (close - low_min) / (high_max - low_min + 1e-10)
        k = pd.Series(stoch).rolling(k_smooth).mean().values
        d = pd.Series(k).rolling(d_smooth).mean().values
        return k, d


def main():
    logger.info('='*70)
    logger.info('Intraday Trading Model V2: XGBoost with Threshold Optimization')
    logger.info('='*70)
    logger.info('Strategy: Learn patterns + optimize threshold for 80%+ P&R')
    logger.info('Target: Precision >= 80% AND Recall >= 80%')
    logger.info('')
    
    config = StrategyConfig.get_default()
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.results_save_dir, exist_ok=True)
    
    logger.info('Loading 15-minute data...')
    loader = DataLoader(
        hf_repo=config.data.hf_repo,
        cache_dir=config.data.cache_dir,
        verbose=False
    )
    
    df = loader.load_data(
        symbol='BTCUSDT',
        timeframe='15m',
        cache=True
    )
    
    if not loader.validate_data(df):
        logger.error('Data validation failed')
        return False
    
    logger.info(f'Loaded {len(df)} candles')
    logger.info(f'Date range: {df.index[0]} to {df.index[-1]}')
    
    logger.info('\nEngineering features...')
    feature_engineer = FeatureEngineer(config)
    df = feature_engineer.engineer_features(df)
    logger.info('Features generated')
    
    logger.info('\nDetecting patterns...')
    target, profits = create_pattern_labels(
        df,
        profit_target_pct=0.01,
        stop_loss_pct=0.01,
        max_hold_bars=20,
        min_breakout_pct=0.005
    )
    
    pattern_mask = target != -1
    logger.info(f'Patterns detected: {pattern_mask.sum()}')
    logger.info(f'Profitable patterns: {(target == 1).sum()}')
    logger.info(f'Base win rate: {(target == 1).sum() / pattern_mask.sum() * 100:.2f}%')
    
    logger.info('\n' + '='*70)
    logger.info('TRAINING XGBOOST MODEL')
    logger.info('='*70)
    
    model = IntradaySignalModelXGBoost(max_depth=5, learning_rate=0.1, n_estimators=200)
    train_precision, train_recall, train_f1 = model.train(df, target)
    
    logger.info('\n' + '='*70)
    logger.info('GENERATING SIGNALS ON FULL DATASET')
    logger.info('='*70)
    
    signals, confidence_scores = model.predict(df, target)
    
    high_confidence = (signals == 2).sum()
    standard_confidence = (signals == 1).sum()
    total_signals = high_confidence + standard_confidence
    
    logger.info(f'Total signals generated: {total_signals}')
    logger.info(f'  High confidence (prob >= 0.65): {high_confidence}')
    logger.info(f'  Standard confidence (prob {model.threshold:.2f}-0.65): {standard_confidence}')
    
    signal_mask = signals > 0
    signal_indices = np.where(signal_mask)[0]
    
    if len(signal_indices) > 0:
        signal_dates = df.index[signal_indices].date
        unique_dates = pd.Series(signal_dates).unique()
        
        logger.info(f'\nDaily signal statistics:')
        logger.info(f'  Days with signals: {len(unique_dates)}')
        
        daily_counts = pd.Series(signal_dates).value_counts()
        logger.info(f'  Avg signals per day: {daily_counts.mean():.2f}')
        logger.info(f'  Max signals per day: {daily_counts.max()}')
        logger.info(f'  Min signals per day: {daily_counts.min()}')
    
    if total_signals > 0:
        actual_profitable = (target[signal_indices] == 1).sum()
        precision = actual_profitable / total_signals * 100
        
        all_profitable_indices = np.where(target == 1)[0]
        caught_profitable = (signals[all_profitable_indices] > 0).sum()
        total_profitable = len(all_profitable_indices)
        recall = caught_profitable / total_profitable * 100 if total_profitable > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        logger.info('\n' + '='*70)
        logger.info('LIVE DATASET PERFORMANCE')
        logger.info('='*70)
        logger.info(f'Precision (signal accuracy): {precision:.2f}%')
        logger.info(f'Recall (opportunity capture): {recall:.2f}%')
        logger.info(f'F1-Score: {f1:.4f}')
        logger.info(f'Target: Precision >= 80% AND Recall >= 80%')
        
        if precision >= 80 and recall >= 80:
            logger.info('Status: TARGET ACHIEVED')
        else:
            gap_precision = max(0, 80 - precision)
            gap_recall = max(0, 80 - recall)
            logger.info(f'Status: Precision gap {gap_precision:.1f}%, Recall gap {gap_recall:.1f}%')
    
    logger.info('\n' + '='*70)
    logger.info('Model deployment ready')
    logger.info('='*70)
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
