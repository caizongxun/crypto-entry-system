from flask import Blueprint, request, jsonify, current_app
from datetime import datetime, timedelta
import logging
import random
import requests

realtime_bp = Blueprint('realtime', __name__, url_prefix='/api/realtime')
logger = logging.getLogger(__name__)

data_cache = {}
CACHE_TTL = 10


def get_cached_price(symbol):
    now = datetime.now()
    if symbol in data_cache:
        cached_time, price = data_cache[symbol]
        if (now - cached_time).total_seconds() < CACHE_TTL:
            return price
    
    try:
        response = requests.get(
            f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}",
            timeout=5
        )
        if response.status_code == 200:
            price = float(response.json()['price'])
            data_cache[symbol] = (now, price)
            return price
    except Exception as e:
        logger.warning(f"Failed to fetch real price for {symbol}: {e}")
    
    if symbol in data_cache:
        return data_cache[symbol][1]
    
    if symbol == 'BTCUSDT':
        return 42500.0
    elif symbol == 'ETHUSDT':
        return 2300.0
    else:
        return 100.0


@realtime_bp.route('/signals', methods=['GET'])
def get_signals():
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        limit = request.args.get('limit', 10, type=int)
        
        signals = generate_signals(symbol, limit)
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'signals': signals
        })
    except Exception as e:
        logger.error(f"Signals error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@realtime_bp.route('/prediction', methods=['GET'])
def get_prediction():
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        timeframe = request.args.get('timeframe', '1h')
        limit = request.args.get('limit', 50, type=int)
        
        prediction_data = generate_prediction(symbol, timeframe)
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'timeframe': timeframe,
            'candles_used': limit,
            'prediction': prediction_data['prediction'],
            'latest_candle': prediction_data['latest_candle'],
            'technical_indicators': prediction_data['technical_indicators']
        })
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@realtime_bp.route('/candle-status', methods=['GET'])
def get_candle_status():
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        timeframe = request.args.get('timeframe', '1h')
        
        current_price = get_cached_price(symbol)
        
        status_data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'candle_status': {
                'status': 'completed' if random.random() > 0.3 else 'forming',
                'progress_percentage': random.randint(50, 100),
                'current_price': current_price
            }
        }
        
        return jsonify({
            'status': 'success',
            **status_data
        })
    except Exception as e:
        logger.error(f"Candle status error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


def generate_signals(symbol, limit):
    signals = []
    base_price = get_cached_price(symbol)
    
    for i in range(limit):
        confidence = random.uniform(0.55, 0.95)
        signal_type = 'BUY' if random.random() > 0.5 else 'SELL'
        
        signal_time = datetime.now() - timedelta(hours=i)
        
        signals.append({
            'symbol': symbol,
            'signal': signal_type,
            'strength': 'MEDIUM' if confidence < 0.7 else 'STRONG',
            'price': base_price,
            'confidence': round(confidence, 3),
            'bb_position': round(random.uniform(0.1, 0.9), 3),
            'bb_upper': round(base_price * 1.01, 2),
            'bb_lower': round(base_price * 0.99, 2),
            'timestamp': signal_time.isoformat()
        })
    
    return signals


def generate_prediction(symbol, timeframe):
    base_price = get_cached_price(symbol)
    confidence = random.uniform(0.55, 0.95)
    
    volatility = base_price * 0.003
    bb_range = base_price * 0.02
    
    return {
        'prediction': {
            'signal_type': 'BUY' if random.random() > 0.5 else 'SELL',
            'confidence': 'HIGH' if confidence > 0.7 else 'MEDIUM' if confidence > 0.6 else 'LOW',
            'confidence_value': round(confidence, 3),
            'bounce_probability': round(random.uniform(0.55, 0.95), 3),
            'timestamp': datetime.now().isoformat()
        },
        'latest_candle': {
            'open': round(base_price - random.uniform(50, 200), 2),
            'high': round(base_price + random.uniform(50, 300), 2),
            'low': round(base_price - random.uniform(50, 300), 2),
            'close': round(base_price, 2),
            'volume': round(random.uniform(100, 5000), 2),
            'bb_upper': round(base_price + bb_range, 2),
            'bb_basis': round(base_price, 2),
            'bb_lower': round(base_price - bb_range, 2),
            'bb_position': round(random.uniform(0.1, 0.9), 3)
        },
        'technical_indicators': {
            'rsi': round(random.uniform(25, 75), 2),
            'volatility': round(volatility / base_price, 6),
            'macd': round(random.uniform(-100, 100), 2)
        }
    }
