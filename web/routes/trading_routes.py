from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
import logging

trading_bp = Blueprint('trading', __name__, url_prefix='/api/trading')
logger = logging.getLogger(__name__)


@trading_bp.before_request
def init_auto_trader():
    if not hasattr(current_app, 'auto_trader'):
        from trading.auto_trader import AutoTrader
        current_app.auto_trader = AutoTrader(
            initial_balance=1000.0,
            position_size_percent=0.1,
            confidence_threshold=0.5
        )


@trading_bp.route('/account-summary', methods=['GET'])
def get_account_summary():
    try:
        summary = current_app.auto_trader.get_account_summary()
        return jsonify({
            'status': 'success',
            'summary': summary,
            'positions': summary.get('open_positions', []),
            'trade_history': [t for t in current_app.auto_trader.export_trades() if t['status'] == 'CLOSED']
        })
    except Exception as e:
        logger.error(f"Account summary error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@trading_bp.route('/parameters', methods=['GET', 'POST'])
def manage_parameters():
    try:
        if request.method == 'GET':
            return jsonify({
                'status': 'success',
                'parameters': {
                    'initial_balance': current_app.auto_trader.initial_balance,
                    'position_size_percent': current_app.auto_trader.position_size_percent,
                    'confidence_threshold': current_app.auto_trader.confidence_threshold
                }
            })
        
        elif request.method == 'POST':
            data = request.get_json()
            initial_balance = data.get('initial_balance')
            position_size_percent = data.get('position_size_percent')
            confidence_threshold = data.get('confidence_threshold')
            
            current_app.auto_trader.set_parameters(
                initial_balance=initial_balance,
                position_size_percent=position_size_percent,
                confidence_threshold=confidence_threshold
            )
            
            return jsonify({
                'status': 'success',
                'message': 'Parameters updated',
                'parameters': {
                    'initial_balance': current_app.auto_trader.initial_balance,
                    'position_size_percent': current_app.auto_trader.position_size_percent,
                    'confidence_threshold': current_app.auto_trader.confidence_threshold
                }
            })
    except Exception as e:
        logger.error(f"Parameter management error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@trading_bp.route('/execute-signal', methods=['POST'])
def execute_signal():
    try:
        signal_data = request.get_json()
        
        if not signal_data:
            return jsonify({'status': 'error', 'message': 'No signal data provided'}), 400
        
        signal_data['timestamp'] = signal_data.get('timestamp', datetime.now().isoformat())
        
        trade = current_app.auto_trader.process_signal(signal_data)
        
        if not trade:
            return jsonify({
                'status': 'rejected',
                'message': f"Signal rejected - confidence below threshold ({current_app.auto_trader.confidence_threshold:.0%})",
                'reason': 'confidence_threshold',
                'confidence': signal_data.get('confidence', 0),
                'threshold': current_app.auto_trader.confidence_threshold
            })
        
        summary = current_app.auto_trader.get_account_summary()
        return jsonify({
            'status': 'success',
            'message': 'Signal executed',
            'trade': trade.to_dict(),
            'account_summary': summary
        })
    except Exception as e:
        logger.error(f"Signal execution error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@trading_bp.route('/trade-history', methods=['GET'])
def get_trade_history():
    try:
        limit = request.args.get('limit', 50, type=int)
        trades = current_app.auto_trader.export_trades()
        return jsonify({
            'status': 'success',
            'trades': trades[-limit:],
            'total': len(trades)
        })
    except Exception as e:
        logger.error(f"Trade history error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@trading_bp.route('/reset', methods=['POST'])
def reset_trading():
    try:
        data = request.get_json() or {}
        initial_balance = data.get('initial_balance')
        
        current_app.auto_trader.reset(initial_balance=initial_balance)
        
        return jsonify({
            'status': 'success',
            'message': 'Trading reset',
            'account_summary': current_app.auto_trader.get_account_summary()
        })
    except Exception as e:
        logger.error(f"Reset error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@trading_bp.route('/auto-trading/status', methods=['GET'])
def get_auto_trading_status():
    try:
        return jsonify({
            'status': 'success',
            'auto_trading_enabled': True,
            'parameters': {
                'confidence_threshold': current_app.auto_trader.confidence_threshold,
                'position_size_percent': current_app.auto_trader.position_size_percent,
                'initial_balance': current_app.auto_trader.initial_balance,
                'current_balance': current_app.auto_trader.current_balance
            }
        })
    except Exception as e:
        logger.error(f"Status error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
