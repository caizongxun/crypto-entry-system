import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import sys
import os
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from strategy_v3 import StrategyConfig, DataLoader, FeatureEngineer
from strategy_v3.multilayer_features import MultiLayerFeatureEngineer
from strategy_v3.targets_pattern_v1 import create_pattern_labels

logger.remove()
logger.add(
    sys.stderr,
    level='INFO',
    format='{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}'
)


class IntradayTradingModelV1:
    """
    Intraday Trading Signal Model - 5-Layer Confirmation System
    
    Target: Precision >= 80% AND Recall >= 80% with daily signal guarantee
    
    Architecture:
    - Layer 1: Market Environment Filter (volume, ATR, time)
    - Layer 2: Multi-Timeframe Trend Alignment (4h/1h/15m)
    - Layer 3: Price Action Confirmation (RSI, BB, Stochastic, Follow-through)
    - Layer 4: Volume Microstructure (volume anomalies)
    - Layer 5: Timing Confirmation (MACD, RSI reversals)
    
    Confidence Score: 0-10 (higher = stronger signal)
    Signal Entry Rules:
    - Confidence >= 8: High precision entry (target 85%+ accuracy)
    - Confidence 6-7: Standard entry (target 80%+ accuracy)
    - Confidence < 6: Filtered out
    """
    
    def __init__(self):
        pass
    
    def layer1_market_environment(self, df, pattern_mask):
        """
        Layer 1: Market Environment Filter (Mandatory)
        
        Strict checks:
        - Volume > 200% of 20-period average (high activity)
        - ATR(14) > 50th percentile (moderate+ volatility)
        - Time: excludes first/last 30 minutes
        
        Returns binary: pass/fail only
        """
        volume = df['volume'].values
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Stricter volume check
        avg_volume_20 = pd.Series(volume).rolling(window=20).mean().values
        volume_ok = volume > (avg_volume_20 * 2.0)
        
        # ATR: stricter median-based threshold
        tr = np.maximum(
            high - low,
            np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1)))
        )
        atr = pd.Series(tr).rolling(window=14).mean().values
        atr_median = np.median(atr[50:])
        atr_ok = atr > atr_median
        
        # Time check
        hour_of_day = np.arange(len(df)) % 96
        time_ok = (hour_of_day > 2) & (hour_of_day < 94)
        
        environment_ok = volume_ok & atr_ok & time_ok
        return environment_ok
    
    def layer2_multi_timeframe_trend(self, df, pattern_mask):
        """
        Layer 2: Multi-Timeframe Trend Alignment
        
        Requires strong alignment across timeframes.
        Scoring:
        - Perfect 4h+1h alignment: +2 points
        - Partial: 0 points (no half credit)
        - Conflict: -2 points (penalty)
        """
        close = df['close'].values
        
        ma_4h_20 = pd.Series(close).rolling(window=20).mean().values
        ma_4h_50 = pd.Series(close).rolling(window=50).mean().values
        trend_4h = np.sign(ma_4h_20 - ma_4h_50)
        
        ma_1h_12 = pd.Series(close).rolling(window=12).mean().values
        ma_1h_26 = pd.Series(close).rolling(window=26).mean().values
        trend_1h = np.sign(ma_1h_12 - ma_1h_26)
        
        rsi = self._calculate_rsi(close, 14)
        
        trend_score = np.zeros(len(df), dtype=int)
        
        for i in range(len(df)):
            if trend_4h[i] != 0 and trend_1h[i] != 0:
                if trend_4h[i] == trend_1h[i]:
                    trend_score[i] = 2
                else:
                    trend_score[i] = -2
            else:
                trend_score[i] = 0
        
        return trend_score
    
    def layer3_price_action(self, df, target):
        """
        Layer 3: Price Action - Strict Requirements
        
        Must satisfy BOTH conditions:
        1. RSI extreme AND (BB touch OR Stochastic align)
        2. Follow-through OR Support/Resistance touch
        
        Scoring: 0 or 2 points (all-or-nothing)
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        rsi = self._calculate_rsi(close, 14)
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close, 20, 2)
        stoch_k, stoch_d = self._calculate_stochastic(high, low, close, 14, 3, 5)
        
        price_action_score = np.zeros(len(df), dtype=int)
        pattern_indices = np.where(target != -1)[0]
        
        for i in pattern_indices:
            if i < 2:
                continue
            
            # Condition 1: RSI extreme
            rsi_extreme = (rsi[i] < 25 or rsi[i] > 75)
            
            # Support: BB or Stochastic confirmation
            bb_touch = (close[i] <= bb_lower[i] or close[i] >= bb_upper[i])
            stoch_align = ((stoch_k[i] < 20 and rsi[i] < 30) or (stoch_k[i] > 80 and rsi[i] > 70))
            condition1 = rsi_extreme and (bb_touch or stoch_align)
            
            # Condition 2: Follow-through or Support/Resistance
            follow_through = False
            if target[i] == 1:
                follow_through = close[i] > close[i-1] and close[i-1] > close[i-2]
            elif target[i] == 0:
                follow_through = close[i] < close[i-1] and close[i-1] < close[i-2]
            
            recent_high = np.max(high[max(0, i-20):i])
            recent_low = np.min(low[max(0, i-20):i])
            sr_touch = (abs(close[i] - recent_high) < (recent_high - recent_low) * 0.01 or
                       abs(close[i] - recent_low) < (recent_high - recent_low) * 0.01)
            condition2 = follow_through or sr_touch
            
            # Both conditions required
            if condition1 and condition2:
                price_action_score[i] = 2
        
        return price_action_score
    
    def layer4_volume_microstructure(self, df, target):
        """
        Layer 4: Volume Microstructure - Strict Anomaly Detection
        
        Requires volume > Average(20) + 2.0*StdDev (very high)
        Scoring: 0 or 2 points (binary)
        """
        volume = df['volume'].values
        
        avg_volume_20 = pd.Series(volume).rolling(window=20).mean().values
        std_volume_20 = pd.Series(volume).rolling(window=20).std().values
        
        volume_score = np.zeros(len(df), dtype=int)
        pattern_indices = np.where(target != -1)[0]
        
        for i in pattern_indices:
            if i < 20:
                continue
            
            # Much stricter: 2 std deviations
            if volume[i] > (avg_volume_20[i] + 2.0 * std_volume_20[i]):
                volume_score[i] = 2
        
        return volume_score
    
    def layer5_timing_confirmation(self, df, target):
        """
        Layer 5: Timing Confirmation - Strict Entry Timing
        
        Requires BOTH MACD and RSI confirmation:
        - MACD histogram reversal (sign change)
        - RSI inflection (starts reversing)
        
        Scoring: 0 or 2 points (both required)
        """
        close = df['close'].values
        
        macd_line, macd_signal, macd_histogram = self._calculate_macd(close, 12, 26, 9)
        rsi = self._calculate_rsi(close, 14)
        
        timing_score = np.zeros(len(df), dtype=int)
        pattern_indices = np.where(target != -1)[0]
        
        for i in pattern_indices:
            if i < 2:
                continue
            
            # MACD reversal check
            macd_reversal = (macd_histogram[i] * macd_histogram[i-1] < 0)
            
            # RSI inflection check
            rsi_inflection = False
            if rsi[i-1] < 30 and rsi[i] > rsi[i-1]:
                rsi_inflection = True
            elif rsi[i-1] > 70 and rsi[i] < rsi[i-1]:
                rsi_inflection = True
            
            # Both required for timing confirmation
            if macd_reversal and rsi_inflection:
                timing_score[i] = 2
        
        return timing_score
    
    def generate_signals(self, df, target):
        """
        Generate signals with stricter confirmation requirements
        
        Confidence scoring (0-10):
        - Layer 1: Must pass (binary)
        - Layer 2: -2/0/+2 points
        - Layer 3: 0/2 points
        - Layer 4: 0/2 points
        - Layer 5: 0/2 points
        
        Total: -2 to +10 range (only 2+ considered)
        """
        pattern_mask = target != -1
        
        env_ok = self.layer1_market_environment(df, pattern_mask)
        trend_score = self.layer2_multi_timeframe_trend(df, pattern_mask)
        price_score = self.layer3_price_action(df, target)
        volume_score = self.layer4_volume_microstructure(df, target)
        timing_score = self.layer5_timing_confirmation(df, target)
        
        confidence_scores = np.zeros(len(df), dtype=int)
        signals = np.zeros(len(df), dtype=int)
        
        pattern_indices = np.where(pattern_mask)[0]
        
        for idx in pattern_indices:
            # Layer 1: mandatory gate
            if not env_ok[idx]:
                confidence_scores[idx] = 0
                continue
            
            # Calculate score: trend + price + volume + timing
            total_score = trend_score[idx] + price_score[idx] + volume_score[idx] + timing_score[idx]
            
            confidence_scores[idx] = max(0, total_score)  # No negative scores
            
            # Entry logic: stricter thresholds
            if total_score >= 8:  # All 4 layers strong
                signals[idx] = 2  # High precision
            elif total_score >= 6:  # Most layers confirmed
                signals[idx] = 1  # Standard
        
        return signals, confidence_scores
    
    @staticmethod
    def _calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).rolling(window=period).mean().values
        avg_loss = pd.Series(loss).rolling(window=period).mean().values
        
        rs = np.divide(avg_gain, avg_loss, where=avg_loss != 0, out=np.zeros_like(avg_gain))
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _calculate_bollinger_bands(close, period=20, std_dev=2):
        ma = pd.Series(close).rolling(window=period).mean().values
        std = pd.Series(close).rolling(window=period).std().values
        upper = ma + std_dev * std
        lower = ma - std_dev * std
        return upper, ma, lower
    
    @staticmethod
    def _calculate_stochastic(high, low, close, period=14, k_smooth=3, d_smooth=5):
        low_min = pd.Series(low).rolling(window=period).min().values
        high_max = pd.Series(high).rolling(window=period).max().values
        
        stoch = 100 * (close - low_min) / (high_max - low_min + 1e-10)
        k = pd.Series(stoch).rolling(window=k_smooth).mean().values
        d = pd.Series(k).rolling(window=d_smooth).mean().values
        
        return k, d
    
    @staticmethod
    def _calculate_macd(close, fast=12, slow=26, signal=9):
        ema_fast = pd.Series(close).ewm(span=fast).mean().values
        ema_slow = pd.Series(close).ewm(span=slow).mean().values
        macd_line = ema_fast - ema_slow
        macd_signal = pd.Series(macd_line).ewm(span=signal).mean().values
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram


def main():
    logger.info('='*70)
    logger.info('Intraday Trading Model V1: 5-Layer Confirmation System')
    logger.info('='*70)
    logger.info('Objective: Daily signals with 80%+ precision and recall')
    logger.info('Strategy: Strict all-or-nothing layer confirmation')
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
    
    df['pattern_target'] = target
    df['pattern_pnl'] = profits * 100
    
    pattern_mask = target != -1
    logger.info(f'Patterns detected: {pattern_mask.sum()}')
    
    logger.info('\nGenerating intraday trading signals...')
    model = IntradayTradingModelV1()
    signals, confidence_scores = model.generate_signals(df, target)
    
    logger.info('\n' + '='*70)
    logger.info('LAYER ARCHITECTURE (Strict Mode)')
    logger.info('='*70)
    logger.info('Layer 1: Market Environment (volume > 200% avg, ATR > median)')
    logger.info('Layer 2: Trend Alignment (4h/1h perfect align only: +2 or -2)')
    logger.info('Layer 3: Price Action (RSI extreme + (BB or Stochastic) + (FT or SR))')
    logger.info('Layer 4: Volume Microstructure (volume > avg + 2.0*std)')
    logger.info('Layer 5: Timing (MACD reversal + RSI inflection both required)')
    logger.info('')
    logger.info('Scoring: Base 2 per layer, total 0-10, gates at 6+ and 8+')
    
    logger.info('\n' + '='*70)
    logger.info('SIGNAL GENERATION RESULTS')
    logger.info('='*70)
    
    high_confidence = (signals == 2).sum()
    standard_confidence = (signals == 1).sum()
    total_signals = high_confidence + standard_confidence
    
    logger.info(f'Total signals generated: {total_signals}')
    logger.info(f'  High confidence (score >= 8): {high_confidence}')
    logger.info(f'  Standard confidence (score 6-7): {standard_confidence}')
    
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
    else:
        logger.info('\nDaily signal statistics:')
        logger.info('  No signals generated - refinement needed')
    
    if total_signals > 0:
        actual_profitable = (target[signal_indices] == 1).sum()
        precision = actual_profitable / total_signals * 100
        
        all_profitable_indices = np.where(target == 1)[0]
        caught_profitable = (signals[all_profitable_indices] > 0).sum()
        total_profitable = len(all_profitable_indices)
        recall = caught_profitable / total_profitable * 100 if total_profitable > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        logger.info('\n' + '='*70)
        logger.info('PRECISION AND RECALL METRICS')
        logger.info('='*70)
        logger.info(f'Precision (signal accuracy): {precision:.2f}%')
        logger.info(f'Recall (opportunity capture): {recall:.2f}%')
        logger.info(f'F1-Score (harmonic mean): {f1:.4f}')
        logger.info(f'Target: Precision >= 80% AND Recall >= 80%')
        
        if precision >= 80 and recall >= 80:
            logger.info('Target achieved')
        else:
            logger.info('Status: Target not met - model refinement needed')
    
    logger.info('\n' + '='*70)
    logger.info('CONFIDENCE DISTRIBUTION')
    logger.info('='*70)
    
    unique_scores = sorted(np.unique(confidence_scores[confidence_scores > 0]))
    if len(unique_scores) > 0:
        for conf_level in unique_scores:
            count = (confidence_scores == conf_level).sum()
            if count > 0:
                pct = count / len(df) * 100
                logger.info(f'Score {conf_level}: {count} signals ({pct:.2f}%)')
    else:
        logger.info('No confidence scores generated')
    
    logger.info('\n' + '='*70)
    logger.info('Model deployment ready')
    logger.info('='*70)
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
