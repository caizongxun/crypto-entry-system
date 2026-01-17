import pandas as pd
import numpy as np
from typing import Dict, Tuple
from models.config import EVALUATION_CONFIG, SIGNAL_QUALITY_RANGES


class SignalEvaluator:
    """Evaluate and score entry signals based on technical indicators."""

    def __init__(self, config: Dict = None):
        self.config = config or EVALUATION_CONFIG

    def calculate_momentum_strength(self, df: pd.DataFrame, lookback: int = 5) -> np.ndarray:
        """Calculate momentum strength as percentage of recent candles moving upward."""
        if len(df) < lookback:
            return np.full(len(df), 50.0)

        momentum_strength = np.zeros(len(df))
        for i in range(lookback, len(df)):
            recent_close = df['close'].iloc[i-lookback:i].values
            upward_count = np.sum(np.diff(recent_close) > 0)
            momentum_strength[i] = (upward_count / lookback) * 100

        return momentum_strength

    def calculate_volatility_score(self, df: pd.DataFrame) -> np.ndarray:
        """Normalize volatility to 0-100 scale (lower volatility = higher score)."""
        if 'volatility' not in df.columns:
            df['volatility'] = df['close'].pct_change().rolling(20).std() * 100

        volatility = df['volatility'].values
        vol_max = np.nanmax(volatility)
        vol_min = np.nanmin(volatility)

        if vol_max == vol_min:
            vol_score = np.full(len(volatility), 50.0)
        else:
            vol_score = 100 - ((volatility - vol_min) / (vol_max - vol_min) * 100)

        return vol_score

    def calculate_trend_confirmation_score(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate trend confirmation based on moving average alignment."""
        if 'sma_fast' not in df.columns or 'sma_slow' not in df.columns:
            return np.full(len(df), 50.0)

        score = np.zeros(len(df))
        for i in range(len(df)):
            price = df['close'].iloc[i]
            sma_f = df['sma_fast'].iloc[i]
            sma_s = df['sma_slow'].iloc[i]

            if pd.isna(sma_f) or pd.isna(sma_s):
                score[i] = 50.0
                continue

            if price > sma_f > sma_s:
                score[i] = 90.0
            elif price > sma_s:
                score[i] = 70.0
            elif price < sma_f < sma_s:
                score[i] = 10.0
            elif price < sma_s:
                score[i] = 30.0
            else:
                score[i] = 50.0

        return score

    def calculate_rsi_signal_strength(self, df: pd.DataFrame) -> np.ndarray:
        """Generate RSI-based signal strength (0-100)."""
        if 'rsi' not in df.columns:
            return np.full(len(df), 50.0)

        rsi = df['rsi'].values
        signal = np.zeros(len(rsi))

        for i in range(len(rsi)):
            if pd.isna(rsi[i]):
                signal[i] = 50.0
            elif rsi[i] < 30:
                signal[i] = 80.0 + ((30 - rsi[i]) / 30) * 20
            elif rsi[i] > 70:
                signal[i] = 20.0 - ((rsi[i] - 70) / 30) * 20
            else:
                signal[i] = 50.0

        return signal

    def calculate_macd_signal_strength(self, df: pd.DataFrame) -> np.ndarray:
        """Generate MACD-based signal strength."""
        if 'macd' not in df.columns or 'macd_signal' not in df.columns:
            return np.full(len(df), 50.0)

        macd = df['macd'].values
        signal_line = df['macd_signal'].values
        histogram = df['macd_histogram'].values

        signal_strength = np.zeros(len(macd))
        for i in range(len(macd)):
            if pd.isna(macd[i]) or pd.isna(signal_line[i]):
                signal_strength[i] = 50.0
                continue

            if macd[i] > signal_line[i]:
                strength = 50 + min(50, abs(histogram[i]) * 100 if not pd.isna(histogram[i]) else 0)
                signal_strength[i] = strength
            else:
                strength = 50 - min(50, abs(histogram[i]) * 100 if not pd.isna(histogram[i]) else 0)
                signal_strength[i] = strength

        return signal_strength

    def calculate_risk_reward_ratio(self, df: pd.DataFrame, lookback: int = 20) -> np.ndarray:
        """Calculate risk-reward ratio based on recent price action."""
        if len(df) < lookback:
            return np.full(len(df), 50.0)

        rr_ratio = np.zeros(len(df))
        for i in range(lookback, len(df)):
            recent = df.iloc[i-lookback:i]
            current_price = df['close'].iloc[i]
            recent_low = recent['low'].min()
            recent_high = recent['high'].max()

            if recent_low == recent_high:
                rr_ratio[i] = 50.0
                continue

            potential_reward = recent_high - current_price
            potential_risk = current_price - recent_low

            if potential_risk == 0:
                rr_ratio[i] = 100.0
            else:
                ratio = potential_reward / potential_risk
                rr_score = min(100, (ratio / 3.0) * 100)
                rr_ratio[i] = rr_score

        return rr_ratio

    def calculate_quality_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive entry quality score."""
        result_df = df.copy()

        momentum = self.calculate_momentum_strength(df)
        volatility = self.calculate_volatility_score(df)
        trend = self.calculate_trend_confirmation_score(df)
        rsi_strength = self.calculate_rsi_signal_strength(df)
        macd_strength = self.calculate_macd_signal_strength(df)
        rr_ratio = self.calculate_risk_reward_ratio(df)

        m_weight = self.config.get('momentum_weight', 0.25)
        v_weight = self.config.get('volatility_weight', 0.2)
        t_weight = self.config.get('trend_weight', 0.3)
        rr_weight = self.config.get('rr_ratio_weight', 0.25)

        quality_score = (
            momentum * m_weight * 0.5 +
            rsi_strength * m_weight * 0.5 +
            volatility * v_weight +
            trend * t_weight +
            rr_ratio * rr_weight
        )

        result_df['quality_score'] = np.clip(quality_score, 0, 100)
        result_df['momentum_strength'] = momentum
        result_df['volatility_score'] = volatility
        result_df['trend_score'] = trend
        result_df['rsi_strength'] = rsi_strength
        result_df['macd_strength'] = macd_strength
        result_df['rr_ratio'] = rr_ratio

        return result_df

    def get_signal_quality_label(self, score: float) -> str:
        """Classify signal quality based on score."""
        if score >= SIGNAL_QUALITY_RANGES['excellent'][0]:
            return 'excellent'
        elif score >= SIGNAL_QUALITY_RANGES['good'][0]:
            return 'good'
        elif score >= SIGNAL_QUALITY_RANGES['moderate'][0]:
            return 'moderate'
        else:
            return 'poor'

    def generate_entry_recommendations(self, df: pd.DataFrame, threshold: float = 65) -> pd.DataFrame:
        """Generate entry recommendations based on quality scores."""
        eval_df = self.calculate_quality_score(df.copy())
        eval_df['signal_quality'] = eval_df['quality_score'].apply(self.get_signal_quality_label)
        eval_df['entry_recommended'] = eval_df['quality_score'] >= threshold

        print(f"Entry recommendations generated: {eval_df['entry_recommended'].sum()} signals above threshold {threshold}")
        return eval_df

    def get_latest_signal(self, df: pd.DataFrame, threshold: float = 65) -> Dict:
        """Get the latest signal evaluation."""
        if len(df) == 0:
            return {}

        eval_df = self.calculate_quality_score(df.copy())
        latest = eval_df.iloc[-1]

        return {
            'timestamp': df['open_time'].iloc[-1],
            'price': latest['close'],
            'quality_score': latest['quality_score'],
            'quality_label': self.get_signal_quality_label(latest['quality_score']),
            'momentum_strength': latest['momentum_strength'],
            'trend_score': latest['trend_score'],
            'volatility_score': latest['volatility_score'],
            'rsi': latest.get('rsi', np.nan),
            'rr_ratio': latest['rr_ratio'],
            'entry_recommended': latest['quality_score'] >= threshold,
            'recommendation_details': {
                'strong_momentum': latest['momentum_strength'] > 70,
                'good_trend': latest['trend_score'] > 70,
                'low_volatility': latest['volatility_score'] > 60,
                'favorable_rr': latest['rr_ratio'] > 50,
            }
        }