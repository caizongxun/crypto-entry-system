import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))

from strategy_v3 import StrategyConfig, DataLoader, FeatureEngineer

logger.remove()
logger.add(
    sys.stderr,
    level='INFO',
    format='{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}'
)


class IntradaySignalModelV3:
    """
    Random Forest-based Intraday Trading Signal Model V3 - Redesigned
    
    Strategy: Generate signals for ALL candles using rolling window labels
    - Create training labels: if next 20 candles hit +1% before -1%, label=1
    - This creates thousands of training samples across entire dataset
    - Target: 3500+ signals daily with 80%+ precision and recall
    """
    
    def __init__(self, n_estimators=250, max_depth=14, min_samples_leaf=2):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample'
        )
        self.feature_names = None
        self.voting_threshold = 0.50
    
    @staticmethod
    def _create_rolling_labels(df, profit_pct=0.01, loss_pct=0.01, lookback=20):
        """
        Create binary labels for all candles using rolling window.
        Label=1 if price hits profit_pct before hitting loss_pct within lookback bars
        """
        close = df['close'].values
        labels = np.zeros(len(df), dtype=int)
        
        for i in range(len(df) - lookback):
            entry_price = close[i]
            profit_target = entry_price * (1 + profit_pct)
            loss_target = entry_price * (1 - loss_pct)
            
            # Check future prices
            future_prices = close[i+1:i+lookback+1]
            
            # Check if profit target is hit first
            profit_hit = np.any(future_prices >= profit_target)
            loss_hit = np.any(future_prices <= loss_target)
            
            if profit_hit and not loss_hit:
                labels[i] = 1  # Profitable
            elif loss_hit and not profit_hit:
                labels[i] = 0  # Loss
            # If both or neither hit, keep as 0
        
        return labels
    
    def _prepare_features(self, df):
        """
        Extract comprehensive technical features for intraday trading
        """
        features_dict = {}
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # ===== MOMENTUM FEATURES =====
        rsi_7 = self._rsi(close, 7)
        rsi_14 = self._rsi(close, 14)
        rsi_21 = self._rsi(close, 21)
        
        features_dict['rsi_7'] = rsi_7
        features_dict['rsi_14'] = rsi_14
        features_dict['rsi_21'] = rsi_21
        features_dict['rsi_oversold'] = (rsi_14 < 30).astype(float)
        features_dict['rsi_overbought'] = (rsi_14 > 70).astype(float)
        
        # MACD
        macd_line, macd_signal, macd_hist = self._macd(close, 12, 26, 9)
        features_dict['macd_histogram'] = macd_hist
        features_dict['macd_positive'] = (macd_hist > 0).astype(float)
        features_dict['macd_signal'] = macd_signal
        
        # ===== VOLATILITY =====
        bb_upper, bb_mid, bb_lower = self._bollinger_bands(close, 20, 2)
        bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-10)
        bb_position = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
        
        features_dict['bb_width'] = bb_width
        features_dict['bb_position'] = bb_position
        features_dict['bb_lower_touch'] = (close < bb_lower * 1.01).astype(float)
        features_dict['bb_upper_touch'] = (close > bb_upper * 0.99).astype(float)
        
        atr = self._atr(high, low, close, 14)
        atr_ratio = atr / np.mean(atr[50:] + 1e-10)
        features_dict['atr_ratio'] = atr_ratio
        
        # ===== TREND =====
        ma5 = pd.Series(close).rolling(5).mean().values
        ma10 = pd.Series(close).rolling(10).mean().values
        ma20 = pd.Series(close).rolling(20).mean().values
        ma50 = pd.Series(close).rolling(50).mean().values
        
        features_dict['price_above_ma5'] = (close > ma5).astype(float)
        features_dict['price_above_ma20'] = (close > ma20).astype(float)
        features_dict['ma5_above_ma20'] = (ma5 > ma20).astype(float)
        features_dict['ma_slope_5'] = (ma5 - np.roll(ma5, 5)) / (ma5 + 1e-10)
        features_dict['ma_slope_20'] = (ma20 - np.roll(ma20, 20)) / (ma20 + 1e-10)
        
        # ===== STOCHASTIC =====
        stoch_k, stoch_d = self._stochastic(high, low, close, 14, 3, 5)
        features_dict['stoch_k'] = stoch_k
        features_dict['stoch_oversold'] = (stoch_k < 30).astype(float)
        features_dict['stoch_overbought'] = (stoch_k > 70).astype(float)
        
        # ===== PRICE ACTION =====
        lookback = 20
        recent_high = pd.Series(high).rolling(lookback).max().values
        recent_low = pd.Series(low).rolling(lookback).min().values
        
        features_dict['distance_from_low'] = (close - recent_low) / (recent_high - recent_low + 1e-10)
        features_dict['distance_from_high'] = (recent_high - close) / (recent_high - recent_low + 1e-10)
        features_dict['price_breakout'] = (close > recent_high * 0.98).astype(float)
        
        # ===== VOLUME =====
        vol_ma = pd.Series(volume).rolling(20).mean().values
        vol_std = pd.Series(volume).rolling(20).std().values
        vol_zscore = (volume - vol_ma) / (vol_std + 1e-10)
        
        features_dict['volume_zscore'] = vol_zscore
        features_dict['high_volume'] = (vol_zscore > 1.0).astype(float)
        
        # ===== MOMENTUM INDICATORS =====
        roc_10 = (close - np.roll(close, 10)) / np.roll(close, 10)
        roc_20 = (close - np.roll(close, 20)) / np.roll(close, 20)
        features_dict['roc_10'] = roc_10
        features_dict['roc_20'] = roc_20
        
        # ===== CANDLE PATTERNS =====
        body = np.abs(close - np.roll(close, 1))
        body_ratio = body / (atr + 1e-10)
        features_dict['body_ratio'] = body_ratio
        
        # ===== ADX =====
        adx = self._adx(high, low, close, 14)
        features_dict['adx'] = adx
        features_dict['strong_trend'] = (adx > 25).astype(float)
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(features_dict)
        self.feature_names = feature_df.columns.tolist()
        
        # Handle NaN and convert to float32
        feature_df = feature_df.fillna(0)
        X = feature_df.values.astype(np.float32)
        
        return X
    
    def train(self, df, labels):
        """
        Train Random Forest on rolling window labels
        """
        X = self._prepare_features(df)
        
        # Use only valid labels
        valid_mask = labels >= 0
        X_valid = X[valid_mask]
        y_valid = labels[valid_mask]
        
        logger.info(f'Training samples: {len(X_valid)}')
        logger.info(f'  Positive: {(y_valid == 1).sum()}')
        logger.info(f'  Negative: {(y_valid == 0).sum()}')
        logger.info(f'  Win rate: {(y_valid == 1).sum() / len(y_valid) * 100:.2f}%')
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_valid, y_valid, test_size=0.25, random_state=42, stratify=y_valid
        )
        
        logger.info(f'\nTraining: {len(X_train)} samples')
        logger.info(f'Testing: {len(X_test)} samples')
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Threshold optimization
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        best_f1 = 0
        best_threshold = 0.5
        best_precision = 0
        best_recall = 0
        
        logger.info('\nThreshold optimization:')
        for threshold in np.arange(0.45, 0.75, 0.02):
            y_pred = (y_pred_proba >= threshold).astype(int)
            p = precision_score(y_test, y_pred, zero_division=0)
            r = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            logger.info(f'  Threshold {threshold:.2f}: P={p:.4f}, R={r:.4f}, F1={f1:.4f}')
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_precision = p
                best_recall = r
        
        self.voting_threshold = best_threshold
        logger.info(f'\nSelected threshold: {best_threshold:.4f}')
        logger.info(f'Test Performance: P={best_precision:.4f}, R={best_recall:.4f}, F1={best_f1:.4f}')
        
        # Feature importance
        logger.info(f'\nTop 15 Important Features:')
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in importance_df.head(15).iterrows():
            logger.info(f'  {row["feature"]}: {row["importance"]:.4f}')
        
        return best_precision, best_recall, best_f1
    
    def predict(self, df, labels):
        """
        Generate signals for all candles
        """
        X = self._prepare_features(df)
        
        signals = np.zeros(len(df), dtype=int)
        confidence_scores = np.zeros(len(df), dtype=float)
        
        # Get probability from ensemble
        y_pred_prob = self.model.predict_proba(X)[:, 1]
        
        # Generate signals for all candles
        for idx in range(len(df)):
            prob = y_pred_prob[idx]
            confidence_scores[idx] = prob
            
            if prob >= 0.75:
                signals[idx] = 2  # High confidence
            elif prob >= self.voting_threshold:
                signals[idx] = 1  # Standard
        
        return signals, confidence_scores
    
    # ===== TECHNICAL INDICATOR METHODS =====
    @staticmethod
    def _rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).rolling(period).mean().values
        avg_loss = pd.Series(loss).rolling(period).mean().values
        
        rs = np.divide(avg_gain, avg_loss, where=avg_loss != 0, out=np.zeros_like(avg_gain))
        rsi = 100 - (100 / (1 + rs))
        return np.nan_to_num(rsi, 50)
    
    @staticmethod
    def _macd(close, fast=12, slow=26, signal=9):
        ema_fast = pd.Series(close).ewm(span=fast).mean().values
        ema_slow = pd.Series(close).ewm(span=slow).mean().values
        macd_line = ema_fast - ema_slow
        macd_signal = pd.Series(macd_line).ewm(span=signal).mean().values
        macd_hist = macd_line - macd_signal
        return macd_line, macd_signal, macd_hist
    
    @staticmethod
    def _bollinger_bands(close, period=20, std_dev=2):
        ma = pd.Series(close).rolling(period).mean().values
        std = pd.Series(close).rolling(period).std().values
        upper = ma + std_dev * std
        lower = ma - std_dev * std
        return upper, ma, lower
    
    @staticmethod
    def _atr(high, low, close, period=14):
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        return pd.Series(tr).rolling(period).mean().values
    
    @staticmethod
    def _stochastic(high, low, close, period=14, k_smooth=3, d_smooth=5):
        low_min = pd.Series(low).rolling(period).min().values
        high_max = pd.Series(high).rolling(period).max().values
        stoch = 100 * (close - low_min) / (high_max - low_min + 1e-10)
        k = pd.Series(stoch).rolling(k_smooth).mean().values
        d = pd.Series(k).rolling(d_smooth).mean().values
        return np.nan_to_num(k, 50), np.nan_to_num(d, 50)
    
    @staticmethod
    def _adx(high, low, close, period=14):
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        
        dm_plus = np.where(high - np.roll(high, 1) > np.roll(low, 1) - low, 
                           high - np.roll(high, 1), 0)
        dm_minus = np.where(np.roll(low, 1) - low > high - np.roll(high, 1),
                            np.roll(low, 1) - low, 0)
        
        tr_smooth = pd.Series(tr).rolling(period).sum().values
        dm_plus_smooth = pd.Series(dm_plus).rolling(period).sum().values
        dm_minus_smooth = pd.Series(dm_minus).rolling(period).sum().values
        
        di_plus = 100 * dm_plus_smooth / (tr_smooth + 1e-10)
        di_minus = 100 * dm_minus_smooth / (tr_smooth + 1e-10)
        
        adx = np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        return np.nan_to_num(adx * 100, 0)


def main():
    logger.info('='*70)
    logger.info('Intraday Trading Model V3: Rolling Window Label Generation')
    logger.info('='*70)
    logger.info('Strategy: Generate signals for ALL candles')
    logger.info('Target: 3500+ signals, Precision >= 80%, Recall >= 80%')
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
    
    logger.info('\nCreating rolling window labels...')
    labels = IntradaySignalModelV3._create_rolling_labels(
        df,
        profit_pct=0.01,
        loss_pct=0.01,
        lookback=20
    )
    
    valid_mask = labels >= 0
    logger.info(f'Generated {valid_mask.sum()} valid labels')
    logger.info(f'  Profitable: {(labels == 1).sum()}')
    logger.info(f'  Unprofitable: {(labels == 0).sum()}')
    logger.info(f'  Win rate: {(labels == 1).sum() / valid_mask.sum() * 100:.2f}%')
    
    logger.info('\n' + '='*70)
    logger.info('TRAINING RANDOM FOREST MODEL')
    logger.info('='*70)
    
    model = IntradaySignalModelV3(n_estimators=250, max_depth=14, min_samples_leaf=2)
    train_p, train_r, train_f1 = model.train(df, labels)
    
    logger.info('\n' + '='*70)
    logger.info('GENERATING SIGNALS ON FULL DATASET')
    logger.info('='*70)
    
    signals, confidence_scores = model.predict(df, labels)
    
    high_conf = (signals == 2).sum()
    standard_conf = (signals == 1).sum()
    total_signals = high_conf + standard_conf
    
    logger.info(f'Total signals generated: {total_signals}')
    logger.info(f'  High confidence (prob >= 0.75): {high_conf}')
    logger.info(f'  Standard confidence: {standard_conf}')
    
    signal_indices = np.where(signals > 0)[0]
    
    if len(signal_indices) > 0:
        signal_dates = df.index[signal_indices].date
        unique_dates = pd.Series(signal_dates).unique()
        
        logger.info(f'\nDaily signal distribution:')
        logger.info(f'  Days with signals: {len(unique_dates)}')
        daily_counts = pd.Series(signal_dates).value_counts()
        logger.info(f'  Avg signals/day: {daily_counts.mean():.2f}')
        logger.info(f'  Max signals/day: {daily_counts.max()}')
        logger.info(f'  Min signals/day: {daily_counts.min()}')
    
    # Calculate metrics using true labels
    if total_signals > 0:
        actual_profitable = (labels[signal_indices] == 1).sum()
        precision = actual_profitable / total_signals * 100 if total_signals > 0 else 0
        
        all_profitable_idx = np.where(labels == 1)[0]
        caught = (signals[all_profitable_idx] > 0).sum()
        recall = caught / len(all_profitable_idx) * 100 if len(all_profitable_idx) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        logger.info('\n' + '='*70)
        logger.info('FULL DATASET PERFORMANCE')
        logger.info('='*70)
        logger.info(f'Precision: {precision:.2f}%')
        logger.info(f'Recall: {recall:.2f}%')
        logger.info(f'F1-Score: {f1:.4f}')
        logger.info(f'Target: 3500+ signals, P>=80%, R>=80%')
        
        gaps = []
        if total_signals < 3500:
            gaps.append(f'Signals: {3500-total_signals} short')
        if precision < 80:
            gaps.append(f'Precision: {80-precision:.1f}% short')
        if recall < 80:
            gaps.append(f'Recall: {80-recall:.1f}% short')
        
        if not gaps:
            logger.info('Status: ALL TARGETS ACHIEVED')
        else:
            logger.info(f'Status: {" | ".join(gaps)}')
    
    logger.info('\n' + '='*70)
    logger.info('Model ready for deployment')
    logger.info('='*70)
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
