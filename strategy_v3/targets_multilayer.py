import numpy as np
import pandas as pd
from loguru import logger


class MultiLayerLabelGenerator:
    """
    Generate multi-layer labels for 15-minute pattern detection.
    
    Confirms patterns using multiple layers:
    1. Pattern quality check
    2. Momentum confirmation (1h and 4h trends match expected direction)
    3. Volume confirmation (volume increasing or price accelerating)
    4. Extremum confirmation (RSI/MACD at extremes)
    5. Risk filtering (acceptable volatility and gaps)
    6. Environment confirmation (broader trend alignment)
    
    Only trade when at least 2-3 confirmation layers align.
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logger
        
        self.quality_threshold = self.config.get('quality_threshold', 60)
        self.min_confirmations = self.config.get('min_confirmations', 2)
        self.volatility_threshold = self.config.get('volatility_threshold', 0.03)
        self.gap_threshold = self.config.get('gap_threshold', 0.02)
    
    def check_pattern_quality(self, df, pattern_quality_scores):
        """
        Layer 1: Check if pattern quality meets threshold.
        
        Returns: Boolean array
        """
        return pattern_quality_scores >= self.quality_threshold
    
    def check_momentum_confirmation(self, df, pattern_direction):
        """
        Layer 2: Check if momentum (trend) aligns with expected pattern direction.
        
        pattern_direction: 1 for double bottom (expect up), -1 for double top (expect down)
        
        Returns: Boolean array
        """
        confirmations = np.zeros(len(df), dtype=bool)
        
        close = df['close'].values
        
        ma_4h_fast = pd.Series(close).rolling(window=16).mean().values
        ma_4h_slow = pd.Series(close).rolling(window=32).mean().values
        trend_4h = np.where(ma_4h_fast > ma_4h_slow, 1, -1)
        
        ma_1h_fast = pd.Series(close).rolling(window=4).mean().values
        ma_1h_slow = pd.Series(close).rolling(window=8).mean().values
        trend_1h = np.where(ma_1h_fast > ma_1h_slow, 1, -1)
        
        for i in range(len(df)):
            if pattern_direction[i] == 0:
                confirmations[i] = False
                continue
            
            trend_match = (trend_1h[i] == pattern_direction[i]) or (trend_4h[i] == pattern_direction[i])
            confirmations[i] = trend_match
        
        return confirmations
    
    def check_volume_confirmation(self, df):
        """
        Layer 3: Check if volume is increasing or price is accelerating.
        
        Returns: Boolean array
        """
        confirmations = np.zeros(len(df), dtype=bool)
        
        volume = df['volume'].values
        close = df['close'].values
        
        avg_volume = pd.Series(volume).rolling(window=20).mean().values
        volume_increasing = volume > avg_volume
        
        acceleration = np.gradient(np.gradient(close, edge_order=2), edge_order=2)
        acceleration_positive = acceleration > 0
        
        confirmations = volume_increasing | acceleration_positive
        
        return confirmations
    
    def check_extremum_confirmation(self, df):
        """
        Layer 4: Check if RSI or MACD at extremes (strong signal).
        
        Returns: Boolean array
        """
        confirmations = np.zeros(len(df), dtype=bool)
        
        close = df['close'].values
        
        delta = np.concatenate([[0], np.diff(close)])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).rolling(window=14).mean().values
        avg_loss = pd.Series(loss).rolling(window=14).mean().values
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        
        rsi_extreme = (rsi < 30) | (rsi > 70)
        
        ema_12 = pd.Series(close).ewm(span=12).mean().values
        ema_26 = pd.Series(close).ewm(span=26).mean().values
        macd = ema_12 - ema_26
        signal = pd.Series(macd).ewm(span=9).mean().values
        
        macd_signal_crossover = np.concatenate([[False], np.diff(np.sign(macd - signal)) != 0])
        
        confirmations = rsi_extreme | macd_signal_crossover
        
        return confirmations
    
    def check_risk_filter(self, df):
        """
        Layer 5: Filter out trades with excessive risk.
        
        Check volatility and gaps.
        
        Returns: Boolean array (True = acceptable risk)
        """
        acceptable = np.ones(len(df), dtype=bool)
        
        returns = np.concatenate([[0], np.diff(df['close'].values) / df['close'].values[:-1]])
        volatility = pd.Series(returns).rolling(window=20).std().values
        
        high_volatility = volatility > self.volatility_threshold
        acceptable = acceptable & ~high_volatility
        
        gap = np.abs(df['open'].values - np.concatenate([[df['close'].values[0]], df['close'].values[:-1]])) / np.concatenate([[df['close'].values[0]], df['close'].values[:-1]])
        excessive_gap = gap > self.gap_threshold
        acceptable = acceptable & ~excessive_gap
        
        return acceptable
    
    def check_environment_confirmation(self, df):
        """
        Layer 6: Check broader environment (trend alignment).
        
        Returns: Boolean array
        """
        confirmations = np.zeros(len(df), dtype=bool)
        
        close = df['close'].values
        
        ma_4h_fast = pd.Series(close).rolling(window=16).mean().values
        ma_4h_slow = pd.Series(close).rolling(window=32).mean().values
        
        trend_4h_strong = (ma_4h_fast > ma_4h_slow) | (ma_4h_fast < ma_4h_slow)
        
        confirmations = trend_4h_strong
        
        return confirmations
    
    def generate_multilayer_labels(self, df, patterns_df):
        """
        Generate multi-layer confirmed labels.
        
        Parameters:
        -----------
        df : DataFrame
            OHLCV data with multi-layer features
        patterns_df : DataFrame
            Pattern detection results with columns:
            - 'pattern_type': 'double_top' or 'double_bottom'
            - 'pattern_quality_score': quality score 0-100
            - 'pattern_index': index in df
            - 'profitable': whether it was profitable
        
        Returns:
        --------
        labels : array of -1 (no trade), 0 (uncertain), 1 (confident trade)
        confidences : array of confidence scores (number of layers confirmed)
        statistics : dict with label statistics
        """
        labels = np.zeros(len(df), dtype=int)
        confidences = np.zeros(len(df), dtype=int)
        
        pattern_direction = np.zeros(len(df), dtype=int)
        pattern_quality_scores = np.zeros(len(df), dtype=float)
        
        for idx, row in patterns_df.iterrows():
            pattern_idx = row['pattern_index']
            if pattern_idx >= len(df):
                continue
            
            if row['pattern_type'] == 'double_bottom':
                pattern_direction[pattern_idx] = 1
            else:
                pattern_direction[pattern_idx] = -1
            
            pattern_quality_scores[pattern_idx] = row['pattern_quality_score']
        
        layer1 = self.check_pattern_quality(df, pattern_quality_scores)
        layer2 = self.check_momentum_confirmation(df, pattern_direction)
        layer3 = self.check_volume_confirmation(df)
        layer4 = self.check_extremum_confirmation(df)
        layer5 = self.check_risk_filter(df)
        layer6 = self.check_environment_confirmation(df)
        
        for i in range(len(df)):
            if pattern_direction[i] == 0:
                labels[i] = -1
                continue
            
            confirmation_count = 0
            if layer1[i]:
                confirmation_count += 1
            if layer2[i]:
                confirmation_count += 1
            if layer3[i]:
                confirmation_count += 1
            if layer4[i]:
                confirmation_count += 1
            if layer6[i]:
                confirmation_count += 1
            
            if not layer5[i]:
                labels[i] = -1
                confidences[i] = 0
                continue
            
            if confirmation_count >= self.min_confirmations:
                labels[i] = 1 if pattern_direction[i] > 0 else -1
                confidences[i] = confirmation_count
            else:
                labels[i] = 0
                confidences[i] = confirmation_count
        
        statistics = self._compute_statistics(labels, confidences, patterns_df)
        
        return labels, confidences, statistics
    
    def _compute_statistics(self, labels, confidences, patterns_df):
        """
        Compute statistics for label generation.
        """
        total_patterns = len(patterns_df)
        labeled = np.sum(labels != -1)
        high_confidence = np.sum(confidences >= 3)
        
        profitable_count = np.sum(patterns_df['profitable'].values)
        
        stats = {
            'total_patterns': total_patterns,
            'labeled_patterns': labeled,
            'high_confidence_patterns': high_confidence,
            'label_rate': labeled / total_patterns if total_patterns > 0 else 0,
            'profitable_rate': profitable_count / total_patterns if total_patterns > 0 else 0,
            'average_confidence': np.mean(confidences[confidences > 0]) if np.sum(confidences > 0) > 0 else 0,
        }
        
        return stats
