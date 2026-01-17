from flask import Blueprint, request, jsonify
from app.services.whale_tracking import WhaleTracker
from app.services.exchange_flow import ExchangeFlowAnalyzer
import logging

logger = logging.getLogger(__name__)
on_chain_bp = Blueprint('on_chain', __name__, url_prefix='/api/on-chain')

whale_tracker = WhaleTracker()
exchange_analyzer = ExchangeFlowAnalyzer()


@on_chain_bp.route('/whale-transactions', methods=['GET'])
def get_whale_transactions():
    """
    Get recent large whale transactions.
    Query params:
    - symbol: BTCUSDT, ETHUSDT (default: BTCUSDT)
    - limit: max results (default: 20)
    - hours: time window in hours (default: 24)
    """
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        limit = int(request.args.get('limit', 20))
        hours = int(request.args.get('hours', 24))
        
        contract = 'bitcoin' if 'BTC' in symbol else 'ethereum'
        
        transactions = whale_tracker.get_large_transactions(
            contract=contract,
            limit=limit,
            hours=hours
        )
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'data': transactions,
            'count': len(transactions)
        })
    
    except Exception as e:
        logger.error(f'Error in get_whale_transactions: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@on_chain_bp.route('/exchange-flows', methods=['GET'])
def get_exchange_flows():
    """
    Get exchange inflow/outflow data.
    Query params:
    - symbol: BTCUSDT, ETHUSDT (default: BTCUSDT)
    - hours: time window in hours (default: 24)
    """
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        hours = int(request.args.get('hours', 24))
        
        contract = 'bitcoin' if 'BTC' in symbol else 'ethereum'
        
        flows = exchange_analyzer.get_exchange_flows(
            contract=contract,
            hours=hours
        )
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'hours': hours,
            'data': flows,
            'count': len(flows)
        })
    
    except Exception as e:
        logger.error(f'Error in get_exchange_flows: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@on_chain_bp.route('/exchange-pressure', methods=['GET'])
def get_exchange_pressure():
    """
    Get exchange selling/buying pressure analysis.
    Query params:
    - symbol: BTCUSDT, ETHUSDT (default: BTCUSDT)
    """
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        contract = 'bitcoin' if 'BTC' in symbol else 'ethereum'
        
        pressure = exchange_analyzer.analyze_exchange_pressure(contract=contract)
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'data': pressure
        })
    
    except Exception as e:
        logger.error(f'Error in get_exchange_pressure: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@on_chain_bp.route('/whale-wallets', methods=['GET'])
def get_whale_wallets():
    """
    Get top whale wallets by balance.
    Query params:
    - symbol: BTCUSDT, ETHUSDT (default: BTCUSDT)
    - min_balance: minimum balance (default: 1000 for BTC)
    """
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        min_balance = float(request.args.get('min_balance', 1000))
        
        contract = 'bitcoin' if 'BTC' in symbol else 'ethereum'
        
        wallets = whale_tracker.get_whale_wallets(
            contract=contract,
            min_balance_btc=min_balance
        )
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'data': wallets,
            'count': len(wallets)
        })
    
    except Exception as e:
        logger.error(f'Error in get_whale_wallets: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@on_chain_bp.route('/transaction-volume', methods=['GET'])
def get_transaction_volume():
    """
    Get total transaction volume metrics.
    Query params:
    - symbol: BTCUSDT, ETHUSDT (default: BTCUSDT)
    - hours: time window in hours (default: 24)
    """
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        hours = int(request.args.get('hours', 24))
        
        contract = 'bitcoin' if 'BTC' in symbol else 'ethereum'
        
        volume = whale_tracker.get_transaction_volume(
            contract=contract,
            hours=hours
        )
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'hours': hours,
            'data': volume
        })
    
    except Exception as e:
        logger.error(f'Error in get_transaction_volume: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@on_chain_bp.route('/exchange-transactions', methods=['GET'])
def get_exchange_transactions():
    """
    Get whale transactions grouped by exchange.
    Query params:
    - symbol: BTCUSDT, ETHUSDT (default: BTCUSDT)
    """
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        contract = 'bitcoin' if 'BTC' in symbol else 'ethereum'
        
        grouped = whale_tracker.get_whale_transactions_by_exchange(
            contract=contract
        )
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'data': grouped
        })
    
    except Exception as e:
        logger.error(f'Error in get_exchange_transactions: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@on_chain_bp.route('/status', methods=['GET'])
def get_api_status():
    """
    Check on-chain data provider status.
    """
    return jsonify({
        'status': 'success',
        'provider': 'Glassnode',
        'supported_contracts': ['bitcoin', 'ethereum'],
        'endpoints': [
            '/whale-transactions',
            '/exchange-flows',
            '/exchange-pressure',
            '/whale-wallets',
            '/transaction-volume',
            '/exchange-transactions'
        ]
    })
