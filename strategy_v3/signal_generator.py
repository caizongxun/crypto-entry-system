"""
Trading Signal Generation based on ML predictions and technical analysis
"""

import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass

from .config import Config, SignalConfig


logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """Represents a trading signal"""
    timestamp: pd.Timestamp
    signal_type: str  # 'BUY', 'SELL', 'NONE'
    confidence: float  # 0.0 to 1.0
    entry_price: float
    support_level: float
    resistance_level: float
    stop_loss: float
    take_profit: float
    reasoning: str
    atr_value: float
    
    def __repr__(self):
        return f"Signal({self.signal_type}, conf={self.confidence:.2f}, entry={self.entry_price:.2f}, SL={self.stop_loss:.2f}, TP={self.take_profit:.2f})"


class SignalGenerator:
    """Generate trading signals from ML predictions and technical indicators"""
    
    def __init__(self, config: Config):
        """
        Initialize SignalGenerator
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.signal_config = config.signal
        self.signals = []
    
    def generate_signals(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        current_price: float
    ) -> List[TradeSignal]:
        """
        Generate trading signals based on ML predictions and technical indicators
        
        Args:
            df: DataFrame with OHLCV and technical indicators
            predictions: Array of model predictions (support, resistance, breakout_prob)
            current_price: Current market price
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Get latest bar
        latest = df.iloc[-1]
        timestamp = df.index[-1]
        
        # Unpack predictions
        support_pred = predictions[-1, 0]
        resistance_pred = predictions[-1, 1]
        breakout_prob = predictions[-1, 2]  # Should be 0-1
        
        # Get technical indicators
        rsi = latest.get('rsi', 50)
        atr = latest.get('atr', 0)
        macd = latest.get('macd', 0)
        macd_signal = latest.get('macd_signal', 0)
        macd_crossover = latest.get('macd_crossover', 0)
        
        # Normalize breakout probability to 0-1 if needed
        breakout_prob = np.clip(float(breakout_prob), 0.0, 1.0)
        
        # Calculate position targets
        stop_loss = support_pred * (1 - self.signal_config.stop_loss_atr_multiplier * atr / current_price) if atr > 0 else support_pred
        take_profit = resistance_pred * (1 + self.signal_config.take_profit_atr_multiplier * atr / current_price) if atr > 0 else resistance_pred
        
        # Generate buy signal
        buy_signal, buy_confidence = self._generate_buy_signal(
            current_price,
            support_pred,
            resistance_pred,
            breakout_prob,
            rsi,
            macd,
            macd_signal,
            macd_crossover
        )
        
        if buy_signal and buy_confidence >= self.signal_config.medium_confidence_threshold:
            reasoning = self._build_reasoning('BUY', current_price, support_pred, breakout_prob, rsi)
            
            signal = TradeSignal(
                timestamp=timestamp,
                signal_type='BUY',
                confidence=buy_confidence,
                entry_price=current_price,
                support_level=support_pred,
                resistance_level=resistance_pred,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=reasoning,
                atr_value=atr
            )
            signals.append(signal)
            logger.info(f"Generated BUY signal: {signal}")
        
        # Generate sell signal
        sell_signal, sell_confidence = self._generate_sell_signal(
            current_price,
            support_pred,
            resistance_pred,
            breakout_prob,
            rsi,
            macd,
            macd_signal
        )
        
        if sell_signal and sell_confidence >= self.signal_config.medium_confidence_threshold:
            reasoning = self._build_reasoning('SELL', current_price, resistance_pred, breakout_prob, rsi)
            
            signal = TradeSignal(
                timestamp=timestamp,
                signal_type='SELL',
                confidence=sell_confidence,
                entry_price=current_price,
                support_level=support_pred,
                resistance_level=resistance_pred,
                stop_loss=take_profit,  # Inverted for short
                take_profit=stop_loss,
                reasoning=reasoning,
                atr_value=atr
            )
            signals.append(signal)
            logger.info(f"Generated SELL signal: {signal}")
        
        self.signals.extend(signals)
        return signals
    
    def _generate_buy_signal(
        self,
        current_price: float,
        support: float,
        resistance: float,
        breakout_prob: float,
        rsi: float,
        macd: float,
        macd_signal: float,
        macd_crossover: int
    ) -> Tuple[bool, float]:
        """
        Generate buy signal based on multiple conditions
        
        Args:
            current_price: Current price
            support: Predicted support level
            resistance: Predicted resistance level
            breakout_prob: Probability of breakout
            rsi: RSI value
            macd: MACD value
            macd_signal: MACD signal line
            macd_crossover: Whether MACD crossed above signal
            
        Returns:
            Tuple of (signal_triggered, confidence)
        """
        confidence = 0.0
        conditions_met = 0
        conditions_total = 0
        
        # Condition 1: Price near support (buy dip)
        conditions_total += 1
        if support > 0 and support * (1 + self.signal_config.support_resistance_tolerance) >= current_price >= support * (1 - self.signal_config.support_resistance_tolerance):
            confidence += 0.3
            conditions_met += 1
        
        # Condition 2: RSI oversold
        conditions_total += 1
        if rsi < self.signal_config.rsi_oversold:
            confidence += 0.25
            conditions_met += 1
        
        # Condition 3: MACD bullish crossover
        conditions_total += 1
        if macd_crossover == 1 or (macd > macd_signal and macd > 0):
            confidence += 0.25
            conditions_met += 1
        
        # Condition 4: Breakout probability high
        conditions_total += 1
        if breakout_prob > self.signal_config.breakout_probability_threshold:
            confidence += 0.2
            conditions_met += 1
        
        # Require at least 2 conditions met
        trigger = conditions_met >= 2 and confidence >= self.signal_config.low_confidence_threshold
        
        return trigger, min(confidence, 1.0)
    
    def _generate_sell_signal(
        self,
        current_price: float,
        support: float,
        resistance: float,
        breakout_prob: float,
        rsi: float,
        macd: float,
        macd_signal: float
    ) -> Tuple[bool, float]:
        """
        Generate sell signal based on multiple conditions
        
        Args:
            current_price: Current price
            support: Predicted support level
            resistance: Predicted resistance level
            breakout_prob: Probability of breakout
            rsi: RSI value
            macd: MACD value
            macd_signal: MACD signal line
            
        Returns:
            Tuple of (signal_triggered, confidence)
        """
        confidence = 0.0
        conditions_met = 0
        conditions_total = 0
        
        # Condition 1: Price near resistance (take profit)
        conditions_total += 1
        if resistance > 0 and resistance * (1 + self.signal_config.support_resistance_tolerance) >= current_price >= resistance * (1 - self.signal_config.support_resistance_tolerance):
            confidence += 0.3
            conditions_met += 1
        
        # Condition 2: RSI overbought
        conditions_total += 1
        if rsi > self.signal_config.rsi_overbought:
            confidence += 0.25
            conditions_met += 1
        
        # Condition 3: MACD bearish signal (crossing below)
        conditions_total += 1
        if macd < macd_signal and macd < 0:
            confidence += 0.25
            conditions_met += 1
        
        # Condition 4: Low breakout probability
        conditions_total += 1
        if breakout_prob < (1 - self.signal_config.breakout_probability_threshold):
            confidence += 0.2
            conditions_met += 1
        
        # Require at least 2 conditions met
        trigger = conditions_met >= 2 and confidence >= self.signal_config.low_confidence_threshold
        
        return trigger, min(confidence, 1.0)
    
    def _build_reasoning(self, signal_type: str, price: float, level: float, breakout_prob: float, rsi: float) -> str:
        """
        Build reasoning text for signal
        
        Args:
            signal_type: 'BUY' or 'SELL'
            price: Current price
            level: Support/Resistance level
            breakout_prob: Breakout probability
            rsi: RSI value
            
        Returns:
            Reasoning text
        """
        parts = []
        
        if signal_type == 'BUY':
            parts.append(f"Price {price:.2f} near support {level:.2f}")
            if rsi < self.signal_config.rsi_oversold:
                parts.append(f"RSI {rsi:.1f} shows oversold conditions")
            parts.append(f"Breakout probability {breakout_prob:.1%}")
        else:
            parts.append(f"Price {price:.2f} near resistance {level:.2f}")
            if rsi > self.signal_config.rsi_overbought:
                parts.append(f"RSI {rsi:.1f} shows overbought conditions")
            parts.append(f"Breakout probability {breakout_prob:.1%}")
        
        return "; ".join(parts)
    
    def get_signals_dataframe(self) -> pd.DataFrame:
        """
        Convert signals to DataFrame for analysis
        
        Returns:
            DataFrame with all signals
        """
        if not self.signals:
            return pd.DataFrame()
        
        data = {
            'timestamp': [s.timestamp for s in self.signals],
            'signal_type': [s.signal_type for s in self.signals],
            'confidence': [s.confidence for s in self.signals],
            'entry_price': [s.entry_price for s in self.signals],
            'support_level': [s.support_level for s in self.signals],
            'resistance_level': [s.resistance_level for s in self.signals],
            'stop_loss': [s.stop_loss for s in self.signals],
            'take_profit': [s.take_profit for s in self.signals],
            'atr_value': [s.atr_value for s in self.signals],
        }
        
        return pd.DataFrame(data)
    
    def filter_signals_by_confidence(
        self,
        signals: List[TradeSignal],
        min_confidence: float
    ) -> List[TradeSignal]:
        """
        Filter signals by minimum confidence
        
        Args:
            signals: List of signals
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered signals
        """
        return [s for s in signals if s.confidence >= min_confidence]
    
    def get_signal_summary(self) -> str:
        """
        Get summary of generated signals
        
        Returns:
            Formatted summary string
        """
        if not self.signals:
            return "No signals generated"
        
        buy_signals = [s for s in self.signals if s.signal_type == 'BUY']
        sell_signals = [s for s in self.signals if s.signal_type == 'SELL']
        
        avg_buy_conf = np.mean([s.confidence for s in buy_signals]) if buy_signals else 0
        avg_sell_conf = np.mean([s.confidence for s in sell_signals]) if sell_signals else 0
        
        summary = "\n" + "="*60 + "\n"
        summary += "SIGNAL SUMMARY\n"
        summary += "="*60 + "\n"
        summary += f"Total Signals: {len(self.signals)}\n"
        summary += f"Buy Signals: {len(buy_signals)} (avg confidence: {avg_buy_conf:.2%})\n"
        summary += f"Sell Signals: {len(sell_signals)} (avg confidence: {avg_sell_conf:.2%})\n"
        summary += "="*60 + "\n"
        
        return summary
