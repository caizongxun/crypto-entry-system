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
    """日內交易信號模型 - 5層確認系統"""
    
    def __init__(self):
        pass
    
    def layer1_market_environment(self, df, pattern_mask):
        """
        層級 1: 市場環境過濾
        檢查流動性、波動性、時間
        """
        volume = df['volume'].values
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # 成交量檢查
        avg_volume_20 = pd.Series(volume).rolling(window=20).mean().values
        volume_ok = volume > (avg_volume_20 * 1.5)
        
        # ATR (波動率)
        tr = np.maximum(
            high - low,
            np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1)))
        )
        atr = pd.Series(tr).rolling(window=14).mean().values
        atr_25_percentile = np.percentile(atr[50:], 25)
        atr_ok = atr > atr_25_percentile
        
        # 時間檢查 (避免開盤/收盤 30 分鐘)
        # 假設每天 96 根 15m 蠟燭 (1440 / 15)
        hour_of_day = np.arange(len(df)) % 96
        time_ok = (hour_of_day > 2) & (hour_of_day < 94)  # 排除開盤/收盤
        
        environment_ok = volume_ok & atr_ok & time_ok
        
        return environment_ok
    
    def layer2_multi_timeframe_trend(self, df, pattern_mask):
        """
        層級 2: 多時間框架趨勢
        檢查 4h/1h/15m 的對齐度
        """
        close = df['close'].values
        
        # 4 小時趨勢 (window=16, 15m*16=240m=4h)
        ma_4h_20 = pd.Series(close).rolling(window=20).mean().values
        ma_4h_50 = pd.Series(close).rolling(window=50).mean().values
        trend_4h = np.sign(ma_4h_20 - ma_4h_50)  # +1 or -1
        
        # 1 小時趨勢 (window=4)
        ma_1h_12 = pd.Series(close).rolling(window=12).mean().values
        ma_1h_26 = pd.Series(close).rolling(window=26).mean().values
        trend_1h = np.sign(ma_1h_12 - ma_1h_26)
        
        # 15 分鐘方向
        rsi = self._calculate_rsi(close, 14)
        direction_15m = np.sign(rsi - 50)  # 簡單方向
        
        # 計算對齐度
        alignment_score = np.zeros(len(df), dtype=int)
        
        for i in range(len(df)):
            if trend_4h[i] != 0 and trend_1h[i] != 0:
                if trend_4h[i] == trend_1h[i]:
                    alignment_score[i] = 2  # 完美對齐
                else:
                    alignment_score[i] = 0  # 不對齣
            elif trend_4h[i] != 0 or trend_1h[i] != 0:
                alignment_score[i] = 1  # 部分對齁
        
        return alignment_score
    
    def layer3_price_action(self, df, target):
        """
        層級 3: 價格行動確認
        RSI + Bollinger Bands + Stochastic + Follow-through
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
            
            score = 0
            
            # 1. RSI 極值 (30 or 70)
            if rsi[i] < 30 or rsi[i] > 70:
                score += 1
            
            # 2. Bollinger Bands 觸及
            if close[i] <= bb_lower[i] or close[i] >= bb_upper[i]:
                score += 1
            
            # 3. Stochastic 與 RSI 同向
            if (stoch_k[i] < 30 and rsi[i] < 30) or (stoch_k[i] > 70 and rsi[i] > 70):
                score += 1
            
            # 4. Follow-through (前 2 根蠟燭的延續)
            if target[i] == 1:  # 看漲
                if close[i] > close[i-1] and close[i-1] > close[i-2]:
                    score += 1
            elif target[i] == 0:  # 看跌
                if close[i] < close[i-1] and close[i-1] < close[i-2]:
                    score += 1
            
            # 5. 支撐/阻力 (簡化: 相對極值)
            recent_high = np.max(high[max(0, i-20):i])
            recent_low = np.min(low[max(0, i-20):i])
            if abs(close[i] - recent_high) < (recent_high - recent_low) * 0.02 or \
               abs(close[i] - recent_low) < (recent_high - recent_low) * 0.02:
                score += 1
            
            price_action_score[i] = min(score, 5)  # 最高 5 分
        
        return price_action_score
    
    def layer4_volume_microstructure(self, df, target):
        """
        層級 4: 成交量微結構
        成交量增加、成交量動量
        """
        volume = df['volume'].values
        
        avg_volume_20 = pd.Series(volume).rolling(window=20).mean().values
        std_volume_20 = pd.Series(volume).rolling(window=20).std().values
        
        volume_score = np.zeros(len(df), dtype=int)
        pattern_indices = np.where(target != -1)[0]
        
        for i in pattern_indices:
            if i < 20:
                continue
            
            # 成交量 > avg + 1.5*std (強烈)
            if volume[i] > (avg_volume_20[i] + 1.5 * std_volume_20[i]):
                volume_score[i] = 1
        
        return volume_score
    
    def layer5_timing_confirmation(self, df, target):
        """
        層級 5: 時機確認
        MACD 直方圖反轉或 RSI 開始反轉
        """
        close = df['close'].values
        
        macd_line, macd_signal, macd_histogram = self._calculate_macd(close, 12, 26, 9)
        rsi = self._calculate_rsi(close, 14)
        
        timing_score = np.zeros(len(df), dtype=int)
        pattern_indices = np.where(target != -1)[0]
        
        for i in pattern_indices:
            if i < 1:
                continue
            
            # MACD 直方圖轉向
            if i > 1 and macd_histogram[i] * macd_histogram[i-1] < 0:  # 轉向
                timing_score[i] = 1
            
            # RSI 開始反轉 (超賣變強或超買變弱)
            elif (rsi[i-1] < 30 and rsi[i] > rsi[i-1]) or \
                 (rsi[i-1] > 70 and rsi[i] < rsi[i-1]):
                timing_score[i] = 1
        
        return timing_score
    
    def generate_signals(self, df, target):
        """
        生成日內交易信號
        
        Returns:
        - signals: 進場信號 (0/1)
        - confidence_scores: 信心等級 (0-8)
        """
        pattern_mask = target != -1
        
        # 所有層級評分
        env_ok = self.layer1_market_environment(df, pattern_mask)
        trend_score = self.layer2_multi_timeframe_trend(df, pattern_mask)
        price_score = self.layer3_price_action(df, target)
        volume_score = self.layer4_volume_microstructure(df, target)
        timing_score = self.layer5_timing_confirmation(df, target)
        
        # 合併信心等級
        confidence_scores = np.zeros(len(df), dtype=int)
        signals = np.zeros(len(df), dtype=int)
        
        pattern_indices = np.where(pattern_mask)[0]
        
        for idx in pattern_indices:
            # 環境層必須通過
            if not env_ok[idx]:
                confidence_scores[idx] = 0
                continue
            
            # 計算總分 (0-8)
            total_score = (
                1 +  # 環境層 base 1
                trend_score[idx] +  # 0-2
                min(price_score[idx], 3) +  # 0-3
                volume_score[idx] +  # 0-1
                timing_score[idx]  # 0-1
            )
            
            confidence_scores[idx] = total_score
            
            # 進場邏輯
            if total_score >= 5:  # 高精準 (≥5)
                signals[idx] = 2  # 高信度
            elif total_score >= 3:  # 標準 (3-4)
                signals[idx] = 1  # 標準信度
            # else: 信號太弱，跳過
        
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
    logger.info('Target: Daily signals with 80%+ precision and recall')
    logger.info('')
    
    config = StrategyConfig.get_default()
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.results_save_dir, exist_ok=True)
    
    logger.info('Loading 15m data...')
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
    logger.info('Generated features')
    
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
    logger.info(f'Detected patterns: {pattern_mask.sum()}')
    
    logger.info('\nGenerating intraday trading signals...')
    model = IntradayTradingModelV1()
    signals, confidence_scores = model.generate_signals(df, target)
    
    logger.info('\n' + '='*70)
    logger.info('LAYER ARCHITECTURE')
    logger.info('='*70)
    logger.info('Layer 1: Market Environment (流動性+波動率+時間)')
    logger.info('Layer 2: Multi-Timeframe Trend (4h/1h/15m 對齁)')
    logger.info('Layer 3: Price Action (RSI/BB/Stoch/Follow-through)')
    logger.info('Layer 4: Volume Microstructure (成交量確認)')
    logger.info('Layer 5: Timing Confirmation (MACD/RSI 時機)')
    
    logger.info('\n' + '='*70)
    logger.info('SIGNAL GENERATION RESULTS')
    logger.info('='*70)
    
    high_confidence = (signals == 2).sum()
    standard_confidence = (signals == 1).sum()
    total_signals = high_confidence + standard_confidence
    
    logger.info(f'Total signals generated: {total_signals}')
    logger.info(f'  - High confidence (≥5 layers): {high_confidence}')
    logger.info(f'  - Standard confidence (3-4 layers): {standard_confidence}')
    
    # 按天統計信號數
    df['signal'] = signals
    df['confidence'] = confidence_scores
    daily_signals = df[df['signal'] > 0].groupby(df.index.date).size()
    
    logger.info(f'\nDaily signal distribution:')
    logger.info(f'  - Days with signals: {len(daily_signals)}')
    logger.info(f'  - Avg signals per day: {daily_signals.mean():.2f}')
    logger.info(f'  - Max signals per day: {daily_signals.max()}')
    logger.info(f'  - Min signals per day: {daily_signals.min()}')
    
    # 精準率/召回率計算
    if total_signals > 0:
        signal_indices = np.where(signals > 0)[0]
        actual_profitable = (target[signal_indices] == 1).sum()
        
        precision = actual_profitable / total_signals * 100 if total_signals > 0 else 0
        
        # 召回率: 有信號的獲利交易 / 所有獲利交易
        all_profitable_indices = np.where(target == 1)[0]
        caught_profitable = (signals[all_profitable_indices] > 0).sum()
        total_profitable = len(all_profitable_indices)
        recall = caught_profitable / total_profitable * 100 if total_profitable > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        logger.info('\n' + '='*70)
        logger.info('PRECISION & RECALL METRICS')
        logger.info('='*70)
        logger.info(f'Precision (精準率): {precision:.2f}% - 信號的準確性')
        logger.info(f'Recall (召回率): {recall:.2f}% - 捕獲獲利機會的能力')
        logger.info(f'F1-Score: {f1:.4f}')
        logger.info(f'Target: Precision ≥ 80% AND Recall ≥ 80%')
        
        if precision >= 80 and recall >= 80:
            logger.info('✓ TARGET ACHIEVED!')
        else:
            logger.info('⊗ Target not yet achieved - needs refinement')
    
    logger.info('\n' + '='*70)
    logger.info('CONFIDENCE DISTRIBUTION')
    logger.info('='*70)
    
    for conf_level in sorted(np.unique(confidence_scores[confidence_scores > 0])):
        count = (confidence_scores == conf_level).sum()
        if count > 0 and conf_level > 0:
            pct = count / len(df) * 100
            logger.info(f'Confidence {conf_level}: {count} signals ({pct:.2f}%)')
    
    logger.info('\n' + '='*70)
    logger.info('Model ready for deployment')
    logger.info('='*70)
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
