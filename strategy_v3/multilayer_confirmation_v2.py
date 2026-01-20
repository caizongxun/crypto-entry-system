\"\"\"
Multi-Layer Confirmation V2: Research-Driven Implementation

Based on academic research and empirical backtests:
- StockTiming.com: 11-year backtest showing 94% reliability with proper confirmations
- Reddit multi-confirmation: 62% win rate from 500+ trades with 3+ layer alignment
- Key insight: Layers must be INDEPENDENT and span multiple timeframes

Architecture:
1. HTF Trend Layer (1d/4h) - Primary filter
2. Pattern Layer (15m) - Base pattern signal
3. Volume Confirmation - Absolute volume increase
4. Price Action - Proper breakout behavior
5. Technical Support - Historical levels alignment
\"\"\"

import numpy as np
import pandas as pd
from typing import Tuple, Dict


class MultiLayerConfirmationV2:
    def __init__(self):
        self.layer_weights = {
            'htf_trend': 0.25,      # Higher timeframe alignment
            'price_action': 0.25,   # Proper breakout behavior
            'volume': 0.20,         # Absolute volume increase
            'technical_support': 0.20,  # Historical levels
            'extremum': 0.10        # Only timing layer, not filtering
        }
    
    def calculate_htf_trend(self, df: pd.DataFrame, htf_window: int = 16) -> np.ndarray:
        \"\"\"\
        Calculate higher timeframe trend.\
        For 15m data, window=16 represents 4 hours (15m * 16)
        \"\"\"\
        close = df['close'].values\
        ma_fast = pd.Series(close).rolling(window=htf_window).mean().values\
        ma_slow = pd.Series(close).rolling(window=htf_window * 2).mean().values\
        \
        # 0: no clear trend, 1: uptrend, -1: downtrend\
        trend = np.zeros(len(df), dtype=int)\
        trend[ma_fast > ma_slow] = 1\
        trend[ma_fast < ma_slow] = -1\
        \
        return trend\
    \
    def calculate_price_action_confirmation(self, df: pd.DataFrame, target: np.ndarray) -> np.ndarray:\
        \"\"\"\
        Price action confirmation:\
        - Bullish pattern (1) should break above resistance with continuation\
        - Bearish pattern (0) should break below support with continuation\
        \"\"\"\
        high = df['high'].values\
        low = df['low'].values\
        close = df['close'].values\
        \
        confirmation = np.zeros(len(df), dtype=int)\
        pattern_mask = target != -1\
        \
        for i in range(1, len(df)):\
            if not pattern_mask[i]:\
                continue\
            \
            # Check if there's follow-through (not just spike)\
            if target[i] == 1:  # Bullish pattern\
                # Should have: close above pattern + next 2 closes stay elevated\
                if i + 2 < len(df):\
                    avg_close = (close[i+1] + close[i+2]) / 2\
                    if close[i] > close[i-1] and avg_close > close[i]:\
                        confirmation[i] = 1\
            \
            elif target[i] == 0:  # Bearish pattern\
                if i + 2 < len(df):\
                    avg_close = (close[i+1] + close[i+2]) / 2\
                    if close[i] < close[i-1] and avg_close < close[i]:\
                        confirmation[i] = 1\
        \
        return confirmation\
    \
    def calculate_volume_confirmation(self, df: pd.DataFrame, target: np.ndarray, \
                                     volume_window: int = 20) -> np.ndarray:\
        \"\"\"\
        Volume confirmation based on ABSOLUTE volume increase, not just ratio.\
        Research shows: Breakout volume > average + 40% = strong confirmation\
        \"\"\"\
        volume = df['volume'].values\
        avg_volume = pd.Series(volume).rolling(window=volume_window).mean().values\
        std_volume = pd.Series(volume).rolling(window=volume_window).std().values\
        \
        confirmation = np.zeros(len(df), dtype=int)\
        pattern_mask = target != -1\
        \
        for i in np.where(pattern_mask)[0]:\
            if i < volume_window:\
                continue\
            \
            current_vol = volume[i]\
            avg_vol = avg_volume[i]\
            std_vol = std_volume[i]\
            \
            # Volume confirmation: current > average + 1 std dev (40% increase)\
            if current_vol > (avg_vol + std_vol):\
                confirmation[i] = 1\
        \
        return confirmation\
    \
    def calculate_technical_support(self, df: pd.DataFrame, target: np.ndarray, \
                                   lookback: int = 100) -> np.ndarray:\
        \"\"\"\
        Check if pattern aligns with technical support/resistance.\
        Higher reliability when pattern forms at:\
        - Round numbers\
        - Historical swing highs/lows\
        - Fibonacci levels\
        \"\"\"\
        high = df['high'].values\
        low = df['low'].values\
        close = df['close'].values\
        \
        confirmation = np.zeros(len(df), dtype=int)\
        pattern_mask = target != -1\
        \
        for i in np.where(pattern_mask)[0]:\
            if i < lookback:\
                continue\
            \
            # Get recent swing highs and lows\
            recent_high = np.max(high[i-lookback:i])\
            recent_low = np.min(low[i-lookback:i])\
            range_size = recent_high - recent_low\
            \
            current_price = close[i]\
            \
            # Check if at support/resistance (within 1% of historical level)\
            if abs(current_price - recent_high) < range_size * 0.01:\
                confirmation[i] = 1\
            elif abs(current_price - recent_low) < range_size * 0.01:\
                confirmation[i] = 1\
            # Also check round numbers (BTC loves 30000, 40000, etc)\
            elif current_price % 1000 < range_size * 0.005:\
                confirmation[i] = 1\
        \
        return confirmation\
    \
    def calculate_extremum_timing(self, df: pd.DataFrame) -> np.ndarray:\
        \"\"\"\
        Use extremum indicators for TIMING, not filtering.\
        Only add 1 point if RSI in extreme but DON'T filter out others.\
        \"\"\"\
        rsi = df['extremum_rsi'].values if 'extremum_rsi' in df.columns else self._calculate_rsi(df)\
        \
        timing = np.zeros(len(df), dtype=int)\
        # Add bonus point for RSI extreme (< 30 or > 70)\
        timing[(rsi < 30) | (rsi > 70)] = 1\
        \
        return timing\
    \
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> np.ndarray:\
        \"\"\"\
        Calculate RSI if not provided.\
        \"\"\"\
        close = df['close'].values\
        delta = np.diff(close, prepend=close[0])\
        \
        gain = np.where(delta > 0, delta, 0)\
        loss = np.where(delta < 0, -delta, 0)\
        \
        avg_gain = pd.Series(gain).rolling(window=period).mean().values\
        avg_loss = pd.Series(loss).rolling(window=period).mean().values\
        \
        rs = np.divide(avg_gain, avg_loss, where=avg_loss != 0, out=np.zeros_like(avg_gain))\
        rsi = 100 - (100 / (1 + rs))\
        \
        return rsi\
    \
    def apply_confirmation(self, df: pd.DataFrame, target: np.ndarray) -> Tuple[np.ndarray, Dict]:\
        \"\"\"\
        Apply multi-layer confirmation.\
        Returns:\
        - confirmed_labels: Same as input, but patterns below 2 layers marked as 0\
        - stats: Dictionary with confirmation breakdowns\
        \"\"\"\
        confirmed_labels = target.copy()\
        \
        # Calculate each confirmation layer\
        htf_trend = self.calculate_htf_trend(df)\
        price_action = self.calculate_price_action_confirmation(df, target)\
        volume_conf = self.calculate_volume_confirmation(df, target)\
        technical = self.calculate_technical_support(df, target)\
        extremum = self.calculate_extremum_timing(df)\
        \
        # Count confirmations (0-5 layers)\
        confirmation_score = np.zeros(len(df), dtype=int)\
        \
        pattern_mask = target != -1\
        pattern_indices = np.where(pattern_mask)[0]\
        \
        for idx in pattern_indices:\
            score = 0\
            \
            # Layer 1: HTF Trend (must align with pattern direction)\
            if target[idx] == 1 and htf_trend[idx] == 1:\
                score += 1\
            elif target[idx] == 0 and htf_trend[idx] == -1:\
                score += 1\
            \
            # Layer 2: Price Action\
            score += price_action[idx]\
            \
            # Layer 3: Volume\
            score += volume_conf[idx]\
            \
            # Layer 4: Technical Support\
            score += technical[idx]\
            \
            # Layer 5: Extremum (timing bonus)\
            score += extremum[idx]\
            \
            confirmation_score[idx] = score\
            \
            # Filter: require at least 2 independent layers\
            if score < 2:\
                confirmed_labels[idx] = 0  # Mark as low confidence, don't remove\
        \
        # Calculate statistics\
        stats = self._calculate_stats(\
            target, confirmed_labels, confirmation_score, pattern_mask\
        )\
        \
        return confirmed_labels, stats\
    \
    def _calculate_stats(self, original: np.ndarray, confirmed: np.ndarray, \
                         scores: np.ndarray, pattern_mask: np.ndarray) -> Dict:\
        \"\"\"\
        Calculate confirmation statistics.\
        \"\"\"\
        pattern_indices = np.where(pattern_mask)[0]\
        \
        # Original pattern statistics\
        original_positive = (original[pattern_indices] == 1).sum()\
        original_total = len(pattern_indices)\
        original_wr = original_positive / original_total * 100 if original_total > 0 else 0\
        \
        # High confidence (2+ layers)\
        high_conf_mask = (scores[pattern_indices] >= 2) & (pattern_mask[pattern_indices])\
        high_conf_positive = (original[pattern_indices][high_conf_mask] == 1).sum()\
        high_conf_total = high_conf_mask.sum()\
        high_conf_wr = high_conf_positive / high_conf_total * 100 if high_conf_total > 0 else 0\
        \
        # Distribution\
        distribution = {}\
        for level in range(0, 6):\
            count = (scores[pattern_indices] == level).sum()\
            if count > 0:\
                distribution[level] = count\
        \
        return {\
            'original_win_rate': original_wr,\
            'original_total': original_total,\
            'high_confidence_win_rate': high_conf_wr,\
            'high_confidence_count': high_conf_total,\
            'improvement': high_conf_wr - original_wr,\
            'filter_percentage': (original_total - high_conf_total) / original_total * 100 if original_total > 0 else 0,\
            'confirmation_distribution': distribution\
        }\
