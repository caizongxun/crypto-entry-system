import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    LOW = 'LOW'
    MEDIUM = 'MEDIUM'
    HIGH = 'HIGH'
    CRITICAL = 'CRITICAL'


class WhaleSignal(Enum):
    STRONG_BUY = 'STRONG_BUY'      # Major whale inflow
    BUY = 'BUY'                    # Minor whale inflow
    NEUTRAL = 'NEUTRAL'             # Balanced activity
    SELL = 'SELL'                  # Minor whale outflow
    STRONG_SELL = 'STRONG_SELL'    # Major whale outflow


@dataclass
class TradeRiskAssessment:
    """Complete risk assessment for a trade"""
    is_allowed: bool
    risk_level: RiskLevel
    whale_signal: WhaleSignal
    confidence: float
    position_size_multiplier: float  # Adjust position size based on risk
    warnings: List[str]
    recommendations: List[str]
    
    def to_dict(self):
        return {
            'is_allowed': self.is_allowed,
            'risk_level': self.risk_level.value,
            'whale_signal': self.whale_signal.value,
            'confidence': round(self.confidence, 2),
            'position_size_multiplier': round(self.position_size_multiplier, 2),
            'warnings': self.warnings,
            'recommendations': self.recommendations
        }


class TradingRiskFilter:
    """
    Advanced risk filtering system that integrates on-chain whale activity
    with trading decisions to enable/block/adjust positions
    """
    
    def __init__(self):
        self.min_confidence_buy = 0.5
        self.min_confidence_sell = 0.4
        self.whale_outflow_threshold = -1000  # USD millions
        self.whale_inflow_threshold = 1000     # USD millions
        self.max_risk_level_for_large_position = RiskLevel.MEDIUM
        self.critical_outflow_threshold = -5000
        
    def should_allow_trade(
        self,
        order_type: str,
        whale_summary: Dict,
        risk_assessment: Dict,
        ml_confidence: Optional[float] = None,
        position_size: Optional[float] = None
    ) -> Dict:
        """
        Main decision function: should a trade be executed?
        
        Args:
            order_type: 'BUY' or 'SHORT'
            whale_summary: Whale activity data from on-chain analysis
            risk_assessment: Risk assessment from on-chain analysis
            ml_confidence: ML model confidence (0-100)
            position_size: Position size in USD
            
        Returns:
            Dict with decision and risk details
        """
        
        # Calculate whale signal
        whale_signal = self._calculate_whale_signal(whale_summary)
        
        # Calculate risk level
        risk_level = self._get_risk_level(risk_assessment)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(
            whale_signal,
            ml_confidence or 50,
            risk_level
        )
        
        # Determine if trade should be allowed
        is_allowed = self._determine_if_allowed(
            order_type=order_type,
            risk_level=risk_level,
            whale_signal=whale_signal,
            confidence=confidence,
            position_size=position_size
        )
        
        # Calculate position size multiplier
        position_multiplier = self._calculate_position_multiplier(
            risk_level,
            whale_signal,
            confidence
        )
        
        # Generate warnings and recommendations
        warnings = self._generate_warnings(
            risk_level,
            whale_signal,
            whale_summary,
            risk_assessment
        )
        
        recommendations = self._generate_recommendations(
            order_type,
            whale_signal,
            risk_level,
            confidence
        )
        
        assessment = TradeRiskAssessment(
            is_allowed=is_allowed,
            risk_level=risk_level,
            whale_signal=whale_signal,
            confidence=confidence,
            position_size_multiplier=position_multiplier,
            warnings=warnings,
            recommendations=recommendations
        )
        
        return assessment.to_dict()
    
    def _calculate_whale_signal(self, whale_summary: Dict) -> WhaleSignal:
        """
        Determine whale signal based on net flow and activity
        """
        net_flow = whale_summary.get('net_flow', 0)
        inflow_value = whale_summary.get('inflow_value_usd', 0)
        outflow_value = whale_summary.get('outflow_value_usd', 0)
        
        # Calculate flow intensity
        total_activity = inflow_value + outflow_value
        flow_ratio = (inflow_value - outflow_value) / total_activity if total_activity > 0 else 0
        
        # Determine signal
        if net_flow > self.whale_inflow_threshold:
            return WhaleSignal.STRONG_BUY
        elif flow_ratio > 0.3:
            return WhaleSignal.BUY
        elif -0.3 <= flow_ratio <= 0.3:
            return WhaleSignal.NEUTRAL
        elif flow_ratio < -0.3:
            return WhaleSignal.SELL
        elif net_flow < self.whale_outflow_threshold:
            return WhaleSignal.STRONG_SELL
        else:
            return WhaleSignal.NEUTRAL
    
    def _get_risk_level(self, risk_assessment: Dict) -> RiskLevel:
        """
        Extract or calculate risk level
        """
        risk_level_str = risk_assessment.get('risk_level', 'LOW')
        
        try:
            return RiskLevel[risk_level_str.upper()]
        except (KeyError, AttributeError):
            return RiskLevel.MEDIUM
    
    def _calculate_confidence(
        self,
        whale_signal: WhaleSignal,
        ml_confidence: float,
        risk_level: RiskLevel
    ) -> float:
        """
        Calculate overall confidence in trade
        """
        # Base confidence from whale signal
        signal_confidence = self._whale_signal_confidence(whale_signal)
        
        # ML confidence (0-100) normalized to 0-1
        ml_score = ml_confidence / 100.0
        
        # Risk adjustment
        risk_penalty = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 0.8,
            RiskLevel.HIGH: 0.5,
            RiskLevel.CRITICAL: 0.1
        }.get(risk_level, 0.5)
        
        # Weighted average: 40% whale signal, 50% ML, 10% risk adjustment
        confidence = (
            0.4 * signal_confidence +
            0.5 * ml_score +
            0.1 * risk_penalty
        )
        
        return min(1.0, max(0.0, confidence))
    
    def _whale_signal_confidence(self, signal: WhaleSignal) -> float:
        """
        Convert whale signal to confidence score
        """
        return {
            WhaleSignal.STRONG_BUY: 0.95,
            WhaleSignal.BUY: 0.75,
            WhaleSignal.NEUTRAL: 0.50,
            WhaleSignal.SELL: 0.25,
            WhaleSignal.STRONG_SELL: 0.05
        }.get(signal, 0.50)
    
    def _determine_if_allowed(
        self,
        order_type: str,
        risk_level: RiskLevel,
        whale_signal: WhaleSignal,
        confidence: float,
        position_size: Optional[float] = None
    ) -> bool:
        """
        Determine if trade should be executed
        """
        # Block trades in critical risk
        if risk_level == RiskLevel.CRITICAL:
            return False
        
        # Block BUY/LONG during strong sell signals in high risk
        if order_type.upper() == 'BUY' and risk_level == RiskLevel.HIGH:
            if whale_signal in [WhaleSignal.STRONG_SELL, WhaleSignal.SELL]:
                return False
        
        # Block SHORT during strong buy signals in high risk
        if order_type.upper() == 'SHORT' and risk_level == RiskLevel.HIGH:
            if whale_signal in [WhaleSignal.STRONG_BUY, WhaleSignal.BUY]:
                return False
        
        # Check minimum confidence thresholds
        min_confidence = self.min_confidence_buy if order_type.upper() == 'BUY' else self.min_confidence_sell
        if confidence < min_confidence:
            return False
        
        # Block large positions in high risk environments
        if position_size and position_size > 10000 and risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            return False
        
        return True
    
    def _calculate_position_multiplier(
        self,
        risk_level: RiskLevel,
        whale_signal: WhaleSignal,
        confidence: float
    ) -> float:
        """
        Calculate position size multiplier based on risk
        """
        # Base multiplier from risk level
        risk_multiplier = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 0.7,
            RiskLevel.HIGH: 0.4,
            RiskLevel.CRITICAL: 0.0
        }.get(risk_level, 0.5)
        
        # Whale signal adjustment
        signal_multiplier = {
            WhaleSignal.STRONG_BUY: 1.2,
            WhaleSignal.BUY: 1.0,
            WhaleSignal.NEUTRAL: 0.8,
            WhaleSignal.SELL: 0.6,
            WhaleSignal.STRONG_SELL: 0.3
        }.get(whale_signal, 0.8)
        
        # Confidence adjustment
        confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5-1.0
        
        # Combined multiplier
        total_multiplier = risk_multiplier * signal_multiplier * confidence_multiplier
        
        return min(2.0, max(0.0, total_multiplier))
    
    def _generate_warnings(self, risk_level, whale_signal, whale_summary, risk_assessment) -> List[str]:
        """
        Generate warnings for the trader
        """
        warnings = []
        
        if risk_level == RiskLevel.CRITICAL:
            warnings.append('CRITICAL: Market conditions extremely risky')
        elif risk_level == RiskLevel.HIGH:
            warnings.append('HIGH RISK: Proceed with caution')
        
        if whale_signal == WhaleSignal.STRONG_SELL:
            warnings.append('Major whale selling pressure detected')
        elif whale_signal == WhaleSignal.STRONG_BUY:
            warnings.append('Major whale buying accumulation detected')
        
        net_flow = whale_summary.get('net_flow', 0)
        if net_flow < self.critical_outflow_threshold:
            warnings.append(f'EXTREME: ${abs(net_flow):.0f}M whale outflow')
        
        # Add risk factor warnings
        risk_factors = risk_assessment.get('risk_factors', [])
        for factor in risk_factors[:2]:  # Limit to 2 most important
            if 'outflow' in factor.lower() or 'exchange' in factor.lower():
                warnings.append(f'Risk factor: {factor}')
        
        return warnings
    
    def _generate_recommendations(self, order_type, whale_signal, risk_level, confidence) -> List[str]:
        """
        Generate trading recommendations
        """
        recommendations = []
        
        if risk_level == RiskLevel.LOW and confidence > 0.7:
            recommendations.append('Optimal conditions: Consider increasing position size')
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.append('Standard risk management: Use 2-3% position sizing')
        elif risk_level == RiskLevel.HIGH:
            recommendations.append('Reduce position size to 0.5-1%')
        
        if whale_signal == WhaleSignal.STRONG_BUY and order_type.upper() == 'BUY':
            recommendations.append('Strong alignment: Whale buying supports your signal')
        elif whale_signal == WhaleSignal.STRONG_SELL and order_type.upper() == 'SHORT':
            recommendations.append('Strong alignment: Whale selling supports your signal')
        elif (whale_signal == WhaleSignal.STRONG_BUY and order_type.upper() == 'SHORT') or \
             (whale_signal == WhaleSignal.STRONG_SELL and order_type.upper() == 'BUY'):
            recommendations.append('WARNING: Your trade direction opposes whale activity')
        
        if confidence > 0.85:
            recommendations.append('High confidence: Reduce trailing stop distance')
        elif confidence < 0.55:
            recommendations.append('Low confidence: Consider waiting for more confirmation')
        
        return recommendations
    
    def should_close_position(
        self,
        position_type: str,
        entry_price: float,
        current_price: float,
        whale_summary: Dict,
        pnl_percentage: float
    ) -> Tuple[bool, str, float]:
        """
        Determine if position should be closed based on risk changes
        
        Returns:
            (should_close, reason, suggested_stop_loss)
        """
        
        whale_signal = self._calculate_whale_signal(whale_summary)
        net_flow = whale_summary.get('net_flow', 0)
        
        # Close LONG position on extreme whale outflow
        if position_type.upper() == 'BUY' and whale_signal == WhaleSignal.STRONG_SELL:
            if pnl_percentage > -2:  # Don't close at big loss
                return True, 'Strong whale selling detected', current_price * 0.98
        
        # Close SHORT position on extreme whale inflow
        if position_type.upper() == 'SHORT' and whale_signal == WhaleSignal.STRONG_BUY:
            if pnl_percentage > -2:
                return True, 'Strong whale buying detected', current_price * 1.02
        
        # Close on critical outflow
        if net_flow < -5000 and position_type.upper() == 'BUY':
            return True, 'Critical whale outflow detected', current_price * 0.97
        
        # Close on critical inflow (SHORT)
        if net_flow > 5000 and position_type.upper() == 'SHORT':
            return True, 'Critical whale inflow detected', current_price * 1.03
        
        return False, '', None
    
    def get_position_size_recommendation(
        self,
        account_balance: float,
        risk_level: RiskLevel,
        whale_signal: WhaleSignal,
        confidence: float
    ) -> float:
        """
        Calculate recommended position size based on risk parameters
        """
        # Base position is 2% of account
        base_position = account_balance * 0.02
        
        # Apply multiplier
        multiplier = self._calculate_position_multiplier(
            risk_level,
            whale_signal,
            confidence
        )
        
        recommended_position = base_position * multiplier
        
        return recommended_position
