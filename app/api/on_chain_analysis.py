from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta
import logging
import json
from collections import defaultdict

from app.services.whale_tracking import WhaleTracker
from app.services.exchange_flow import ExchangeFlowAnalyzer
from app.services.on_chain_metrics import OnChainMetrics
from app.services.trading_risk_filter import TradingRiskFilter

logger = logging.getLogger(__name__)
on_chain_bp = Blueprint('on_chain', __name__, url_prefix='/api/on-chain')

whale_tracker = WhaleTracker()
exchange_flow = ExchangeFlowAnalyzer()
on_chain_metrics = OnChainMetrics()
risk_filter = TradingRiskFilter()


@on_chain_bp.route('', methods=['GET'])
def get_on_chain_analysis():
    """Get comprehensive on-chain analysis including whale activity and risk metrics"""
    try:
        symbol = request.args.get('symbol', 'BTCUSDT').upper()
        limit = request.args.get('limit', 10, type=int)
        timeframe_hours = request.args.get('timeframe', 24, type=int)
        
        coin = symbol.replace('USDT', '').upper()
        if coin == 'BTC':
            contract = 'bitcoin'
        elif coin == 'ETH':
            contract = 'ethereum'
        else:
            contract = coin.lower()
        
        logger.info(f'Fetching on-chain analysis for {contract}')
        
        # Fetch whale transactions
        large_transactions = whale_tracker.get_large_transactions(
            contract=contract,
            limit=limit,
            hours=timeframe_hours
        )
        
        # Fetch exchange flows
        exchange_flows = exchange_flow.get_exchange_flows(
            contract=contract,
            hours=timeframe_hours
        )
        
        # Calculate whale summary statistics
        whale_summary = _calculate_whale_summary(
            large_transactions,
            exchange_flows,
            contract
        )
        
        # Get on-chain metrics
        on_chain_metrics_data = on_chain_metrics.get_metrics(
            contract=contract,
            hours=timeframe_hours
        )
        
        # Calculate risk score
        risk_assessment = _assess_trading_risk(
            whale_summary,
            on_chain_metrics_data,
            contract
        )
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'large_transactions': large_transactions,
            'exchange_flows': exchange_flows,
            'whale_summary': whale_summary,
            'on_chain_metrics': on_chain_metrics_data,
            'risk_assessment': risk_assessment,
            'trading_recommendation': _get_trading_recommendation(whale_summary, risk_assessment)
        }), 200
        
    except Exception as e:
        logger.error(f'Error in on-chain analysis: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@on_chain_bp.route('/whale-activity', methods=['GET'])
def get_whale_activity():
    """Get real-time whale activity alerts"""
    try:
        symbol = request.args.get('symbol', 'BTCUSDT').upper()
        hours = request.args.get('hours', 1, type=int)
        min_volume = request.args.get('min_volume', 100, type=float)  # In USD millions
        
        coin = symbol.replace('USDT', '').upper()
        contract = 'bitcoin' if coin == 'BTC' else 'ethereum' if coin == 'ETH' else coin.lower()
        
        logger.info(f'Fetching whale activity for {contract} in last {hours} hour(s)')
        
        # Get recent large transactions
        transactions = whale_tracker.get_large_transactions(
            contract=contract,
            limit=50,
            hours=hours
        )
        
        # Filter by volume
        significant_txs = [
            tx for tx in transactions 
            if float(tx.get('value_usd', 0)) >= min_volume * 1_000_000
        ]
        
        # Categorize by type
        inflows = [tx for tx in significant_txs if tx.get('type') == 'inflow']
        outflows = [tx for tx in significant_txs if tx.get('type') == 'outflow']
        
        # Calculate net flow
        inflow_volume = sum(float(tx.get('value_usd', 0)) for tx in inflows)
        outflow_volume = sum(float(tx.get('value_usd', 0)) for tx in outflows)
        net_flow = inflow_volume - outflow_volume
        
        # Determine market pressure
        market_pressure = _calculate_market_pressure(net_flow, inflow_volume, outflow_volume)
        
        # Generate alerts
        alerts = _generate_whale_alerts(
            inflows, outflows, net_flow, market_pressure
        )
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'timeframe_hours': hours,
            'significant_transactions': significant_txs,
            'inflows': inflows,
            'outflows': outflows,
            'statistics': {
                'inflow_volume_usd': inflow_volume,
                'outflow_volume_usd': outflow_volume,
                'net_flow_usd': net_flow,
                'inflow_count': len(inflows),
                'outflow_count': len(outflows)
            },
            'market_pressure': market_pressure,
            'alerts': alerts,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f'Error getting whale activity: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@on_chain_bp.route('/risk-check', methods=['POST'])
def check_trading_risk():
    """Check if conditions are safe for automated trading based on on-chain data"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'BTCUSDT').upper()
        order_type = data.get('order_type', 'BUY')
        
        coin = symbol.replace('USDT', '').upper()
        contract = 'bitcoin' if coin == 'BTC' else 'ethereum' if coin == 'ETH' else coin.lower()
        
        logger.info(f'Risk check for {symbol} {order_type}')
        
        # Get whale summary
        large_txs = whale_tracker.get_large_transactions(
            contract=contract, limit=20, hours=6
        )
        exchange_flows = exchange_flow.get_exchange_flows(
            contract=contract, hours=6
        )
        
        whale_summary = _calculate_whale_summary(large_txs, exchange_flows, contract)
        risk_assessment = _assess_trading_risk(
            whale_summary,
            on_chain_metrics.get_metrics(contract, 6),
            contract
        )
        
        # Determine if trade is allowed
        is_safe = risk_filter.should_allow_trade(
            order_type=order_type,
            whale_summary=whale_summary,
            risk_assessment=risk_assessment
        )
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'order_type': order_type,
            'is_safe': is_safe,
            'risk_level': risk_assessment.get('risk_level'),
            'whale_summary': whale_summary,
            'risk_factors': risk_assessment.get('risk_factors'),
            'warnings': risk_assessment.get('warnings'),
            'confidence_score': risk_assessment.get('confidence_score')
        }), 200
        
    except Exception as e:
        logger.error(f'Error in risk check: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@on_chain_bp.route('/exchange-flows', methods=['GET'])
def get_exchange_flows():
    """Get exchange inflow/outflow data"""
    try:
        symbol = request.args.get('symbol', 'BTCUSDT').upper()
        hours = request.args.get('hours', 24, type=int)
        
        coin = symbol.replace('USDT', '').upper()
        contract = 'bitcoin' if coin == 'BTC' else 'ethereum' if coin == 'ETH' else coin.lower()
        
        logger.info(f'Fetching exchange flows for {contract}')
        
        flows = exchange_flow.get_exchange_flows(
            contract=contract,
            hours=hours
        )
        
        # Aggregate by exchange
        exchange_summary = _aggregate_exchange_flows(flows)
        
        # Detect unusual patterns
        anomalies = _detect_flow_anomalies(flows)
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'flows': flows,
            'exchange_summary': exchange_summary,
            'anomalies': anomalies,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f'Error getting exchange flows: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@on_chain_bp.route('/metrics', methods=['GET'])
def get_metrics():
    """Get on-chain metrics for analysis"""
    try:
        symbol = request.args.get('symbol', 'BTCUSDT').upper()
        hours = request.args.get('hours', 24, type=int)
        
        coin = symbol.replace('USDT', '').upper()
        contract = 'bitcoin' if coin == 'BTC' else 'ethereum' if coin == 'ETH' else coin.lower()
        
        logger.info(f'Fetching on-chain metrics for {contract}')
        
        metrics = on_chain_metrics.get_metrics(
            contract=contract,
            hours=hours
        )
        
        # Calculate trends
        metrics['trends'] = _calculate_metric_trends(metrics)
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'metrics': metrics,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f'Error getting metrics: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# Helper Functions

def _calculate_whale_summary(transactions, exchange_flows, contract):
    """Calculate aggregated whale activity summary"""
    inflows = [tx for tx in transactions if tx.get('type') == 'inflow']
    outflows = [tx for tx in transactions if tx.get('type') == 'outflow']
    
    inflow_sum = sum(float(tx.get('amount', 0)) for tx in inflows)
    outflow_sum = sum(float(tx.get('amount', 0)) for tx in outflows)
    inflow_value = sum(float(tx.get('value_usd', 0)) for tx in inflows)
    outflow_value = sum(float(tx.get('value_usd', 0)) for tx in outflows)
    
    # Exchange holding analysis
    total_exchange_holding = sum(
        float(flow.get('total_reserve', 0)) for flow in exchange_flows
    )
    exchange_inflow = sum(
        float(flow.get('amount', 0)) for flow in exchange_flows 
        if flow.get('flow_type') == 'inflow'
    )
    exchange_outflow = sum(
        float(flow.get('amount', 0)) for flow in exchange_flows 
        if flow.get('flow_type') == 'outflow'
    )
    
    return {
        'inflow': inflow_sum,
        'outflow': outflow_sum,
        'net_flow': inflow_sum - outflow_sum,
        'inflow_value_usd': inflow_value,
        'outflow_value_usd': outflow_value,
        'inflow_count': len(inflows),
        'outflow_count': len(outflows),
        'exchange_holdings': total_exchange_holding,
        'exchange_inflow': exchange_inflow,
        'exchange_outflow': exchange_outflow,
        'exchange_net_flow': exchange_inflow - exchange_outflow,
        'active_addresses': len(set(
            tx.get('from_address') for tx in transactions 
        )) + len(set(
            tx.get('to_address') for tx in transactions
        ))
    }


def _assess_trading_risk(whale_summary, metrics, contract):
    """Assess overall trading risk based on on-chain data"""
    risk_factors = []
    warnings = []
    risk_score = 0
    
    # Check whale net flow
    net_flow = whale_summary.get('net_flow', 0)
    if net_flow < -1000:
        risk_factors.append('Major whale outflow detected')
        risk_score += 35
        warnings.append('Significant selling pressure from whales')
    elif net_flow < -100:
        risk_factors.append('Whale outflow trend')
        risk_score += 15
    elif net_flow > 1000:
        risk_factors.append('Major whale inflow')
        risk_score -= 20
    
    # Check exchange flows
    exchange_net = whale_summary.get('exchange_net_flow', 0)
    if exchange_net > 500:
        risk_factors.append('Large inflow to exchanges')
        risk_score += 20
        warnings.append('Potential selling pressure incoming')
    elif exchange_net < -500:
        risk_factors.append('Large outflow from exchanges')
        risk_score -= 15
    
    # Check metrics
    if metrics.get('network_stress') > 0.7:
        risk_factors.append('High network stress')
        risk_score += 10
    
    # Determine risk level
    if risk_score >= 50:
        risk_level = 'HIGH'
    elif risk_score >= 20:
        risk_level = 'MEDIUM'
    else:
        risk_level = 'LOW'
    
    confidence_score = max(0, 100 - abs(net_flow) / 10)
    
    return {
        'risk_score': risk_score,
        'risk_level': risk_level,
        'risk_factors': risk_factors,
        'warnings': warnings,
        'confidence_score': confidence_score,
        'timestamp': datetime.utcnow().isoformat()
    }


def _get_trading_recommendation(whale_summary, risk_assessment):
    """Generate trading recommendation based on analysis"""
    risk_level = risk_assessment.get('risk_level')
    net_flow = whale_summary.get('net_flow', 0)
    
    if risk_level == 'HIGH':
        return {
            'action': 'HOLD',
            'reason': 'High risk detected from whale activity',
            'suggested_position': None
        }
    elif risk_level == 'MEDIUM':
        if net_flow > 100:
            return {
                'action': 'ACCUMULATE',
                'reason': 'Whale buying detected, medium risk manageable',
                'suggested_position': 'LONG'
            }
        else:
            return {
                'action': 'CAUTIOUS',
                'reason': 'Medium risk, reduce position size',
                'suggested_position': None
            }
    else:  # LOW
        if net_flow > 500:
            return {
                'action': 'BUY',
                'reason': 'Strong whale buying signal, low risk',
                'suggested_position': 'LONG'
            }
        elif net_flow < -500:
            return {
                'action': 'SHORT',
                'reason': 'Whale selling detected',
                'suggested_position': 'SHORT'
            }
        else:
            return {
                'action': 'NEUTRAL',
                'reason': 'Low risk, balanced whale activity',
                'suggested_position': None
            }


def _calculate_market_pressure(net_flow, inflow_vol, outflow_vol):
    """Calculate market pressure indicator"""
    total_volume = inflow_vol + outflow_vol
    if total_volume == 0:
        return 0
    
    pressure = (net_flow / total_volume) * 100
    return round(pressure, 2)


def _generate_whale_alerts(inflows, outflows, net_flow, market_pressure):
    """Generate alerts based on whale activity patterns"""
    alerts = []
    
    if len(inflows) > len(outflows) * 2:
        alerts.append({
            'type': 'ACCUMULATION',
            'severity': 'INFO',
            'message': 'Whales actively accumulating'
        })
    
    if len(outflows) > len(inflows) * 2:
        alerts.append({
            'type': 'DISTRIBUTION',
            'severity': 'WARNING',
            'message': 'Whales actively distributing'
        })
    
    if abs(net_flow) > 1000:
        alerts.append({
            'type': 'EXTREME_ACTIVITY',
            'severity': 'HIGH',
            'message': f'Extreme whale activity detected: ${abs(net_flow):.2f}M net flow'
        })
    
    if abs(market_pressure) > 80:
        alerts.append({
            'type': 'EXTREME_PRESSURE',
            'severity': 'HIGH',
            'message': f'Market pressure at {abs(market_pressure):.1f}%'
        })
    
    return alerts


def _aggregate_exchange_flows(flows):
    """Aggregate flows by exchange"""
    summary = defaultdict(lambda: {'inflow': 0, 'outflow': 0})
    
    for flow in flows:
        exchange = flow.get('exchange_name', 'Unknown')
        amount = float(flow.get('amount', 0))
        flow_type = flow.get('flow_type', 'unknown')
        
        if flow_type == 'inflow':
            summary[exchange]['inflow'] += amount
        else:
            summary[exchange]['outflow'] += amount
    
    return dict(summary)


def _detect_flow_anomalies(flows):
    """Detect unusual flow patterns"""
    anomalies = []
    
    if not flows:
        return anomalies
    
    # Calculate average flow
    amounts = [float(f.get('amount', 0)) for f in flows]
    avg_amount = sum(amounts) / len(amounts) if amounts else 0
    std_dev = (sum((x - avg_amount) ** 2 for x in amounts) / len(amounts)) ** 0.5 if amounts else 0
    
    # Detect outliers (> 2 std devs)
    for flow in flows:
        amount = float(flow.get('amount', 0))
        if abs(amount - avg_amount) > 2 * std_dev:
            anomalies.append({
                'exchange': flow.get('exchange_name'),
                'amount': amount,
                'type': flow.get('flow_type'),
                'severity': 'HIGH' if abs(amount - avg_amount) > 3 * std_dev else 'MEDIUM'
            })
    
    return anomalies


def _calculate_metric_trends(metrics):
    """Calculate trends in on-chain metrics"""
    trends = {}
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            trends[key] = 'stable'  # Would implement real trend analysis here
    
    return trends
