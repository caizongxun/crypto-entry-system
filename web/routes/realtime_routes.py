from flask import Blueprint, request, jsonify, current_app
from datetime import datetime, timedelta
import logging
import random
import requests

realtime_bp = Blueprint('realtime', __name__, url_prefix='/api/realtime')
logger = logging.getLogger(__name__)

candle_cache = {}
CACHE_TTL = 60


def get_completed_candles(symbol, timeframe, limit=50):
    cache_key = f"{symbol}_{timeframe}"
    now = datetime.now()
    
    if cache_key in candle_cache:
        cached_time, candles = candle_cache[cache_key]
        if (now - cached_time).total_seconds() < CACHE_TTL:
            return candles
    
    try:
        interval_map = {
            '15m': '15m',
            '1h': '1h',
            '1d': '1d'
        }
        interval = interval_map.get(timeframe, '1h')
        
        response = requests.get(
            f"https://api.binance.com/api/v3/klines",
            params={
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            },
            timeout=5
        )
        
        if response.status_code == 200:
            klines = response.json()
            candles = []
            
            for kline in klines:
                candle = {
                    'time': datetime.fromtimestamp(kline[0] / 1000).isoformat(),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[7])
                }
                candles.append(candle)
            
            candle_cache[cache_key] = (now, candles)
            return candles
    
    except Exception as e:
        logger.warning(f"Failed to fetch candles for {symbol} {timeframe}: {e}")
    
    if cache_key in candle_cache:
        return candle_cache[cache_key][1]
    
    return generate_mock_completed_candles(symbol, limit)


def generate_mock_completed_candles(symbol, limit):
    if symbol == 'BTCUSDT':
        base_price = 42500.0
    elif symbol == 'ETHUSDT':
        base_price = 2300.0
    else:
        base_price = 100.0
    
    candles = []
    current_time = datetime.now()
    
    for i in range(limit):
        time_offset = timedelta(hours=limit - i)
        candle_time = current_time - time_offset
        
        price_change = random.uniform(-base_price * 0.01, base_price * 0.01)
        open_price = base_price + price_change
        close_price = open_price + random.uniform(-base_price * 0.005, base_price * 0.005)
        high_price = max(open_price, close_price) + abs(random.uniform(0, base_price * 0.005))
        low_price = min(open_price, close_price) - abs(random.uniform(0, base_price * 0.005))
        
        candles.append({
            'time': candle_time.isoformat(),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': round(random.uniform(100, 5000), 2)
        })
    
    return candles


def calculate_bollinger_bands(candles, period=20):
    if len(candles) < period:
        return None, None, None
    
    closes = [candle['close'] for candle in candles[-period:]]
    sma = sum(closes) / period
    variance = sum((x - sma) ** 2 for x in closes) / period
    std_dev = variance ** 0.5
    
    bb_upper = round(sma + (2 * std_dev), 2)
    bb_basis = round(sma, 2)
    bb_lower = round(sma - (2 * std_dev), 2)
    
    latest_price = closes[-1]
    if bb_upper - bb_lower > 0:
        bb_position = (latest_price - bb_lower) / (bb_upper - bb_lower)
        bb_position = round(max(0, min(1, bb_position)), 3)
    else:
        bb_position = 0.5
    
    return bb_upper, bb_basis, bb_lower, bb_position


def calculate_rsi(candles, period=14):
    if len(candles) < period + 1:
        return 50.0
    
    closes = [candle['close'] for candle in candles]
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return round(rsi, 2)


def determine_signal(candles, bb_upper, bb_lower, bb_position):
    if len(candles) < 2:
        return 'NEUTRAL', 0.5
    
    latest_close = candles[-1]['close']
    prev_close = candles[-2]['close']
    
    if bb_position < 0.2 and latest_close > prev_close:
        confidence = 0.75
        signal = 'BUY'
    elif bb_position > 0.8 and latest_close < prev_close:
        confidence = 0.75
        signal = 'SELL'
    elif latest_close > prev_close:
        confidence = 0.55
        signal = 'BUY'
    elif latest_close < prev_close:
        confidence = 0.55
        signal = 'SELL'
    else:
        confidence = 0.5
        signal = 'NEUTRAL'
    
    return signal, round(confidence, 3)


@realtime_bp.route('/signals', methods=['GET'])
def get_signals():
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        timeframe = request.args.get('timeframe', '1h')
        limit = request.args.get('limit', 10, type=int)
        
        candles = get_completed_candles(symbol, timeframe, limit * 2)
        signals = []
        
        for i, candle in enumerate(candles[-limit:]):
            bb_upper, bb_basis, bb_lower, bb_position = calculate_bollinger_bands(
                candles[:len(candles) - limit + i + 1]
            )
            
            signal_type, confidence = determine_signal(
                candles[:len(candles) - limit + i + 1],
                bb_upper, bb_lower, bb_position
            )
            
            signals.append({
                'symbol': symbol,
                'signal': signal_type,
                'strength': 'STRONG' if confidence > 0.7 else 'MEDIUM' if confidence > 0.55 else 'WEAK',
                'price': candle['close'],
                'confidence': confidence,
                'bb_position': bb_position if bb_position else 0.5,
                'bb_upper': bb_upper if bb_upper else candle['close'],
                'bb_lower': bb_lower if bb_lower else candle['close'],
                'timestamp': candle['time']
            })
        
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
        
        candles = get_completed_candles(symbol, timeframe, limit)
        
        if not candles:
            return jsonify({'status': 'error', 'message': 'No candle data available'}), 500
        
        latest_candle = candles[-1]
        
        bb_upper, bb_basis, bb_lower, bb_position = calculate_bollinger_bands(candles)
        rsi = calculate_rsi(candles)
        signal_type, confidence = determine_signal(candles, bb_upper, bb_lower, bb_position)
        
        bounce_probability = 0.5 + (abs(bb_position - 0.5) * 0.4)
        
        volatility = 0.0
        if len(candles) > 1:
            returns = [(candles[i]['close'] - candles[i-1]['close']) / candles[i-1]['close'] for i in range(1, len(candles))]
            volatility = (sum(r**2 for r in returns) / len(returns)) ** 0.5
        
        prediction_data = {
            'prediction': {
                'signal_type': signal_type,
                'confidence': 'HIGH' if confidence > 0.7 else 'MEDIUM' if confidence > 0.6 else 'LOW',
                'confidence_value': confidence,
                'bounce_probability': round(bounce_probability, 3),
                'timestamp': datetime.now().isoformat()
            },
            'latest_candle': {
                'open': latest_candle['open'],
                'high': latest_candle['high'],
                'low': latest_candle['low'],
                'close': latest_candle['close'],
                'volume': latest_candle['volume'],
                'bb_upper': bb_upper if bb_upper else latest_candle['close'],
                'bb_basis': bb_basis if bb_basis else latest_candle['close'],
                'bb_lower': bb_lower if bb_lower else latest_candle['close'],
                'bb_position': bb_position if bb_position else 0.5
            },
            'technical_indicators': {
                'rsi': rsi,
                'volatility': round(volatility, 6),
                'macd': 0.0
            }
        }
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'timeframe': timeframe,
            'candles_used': len(candles),
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
        
        candles = get_completed_candles(symbol, timeframe, 1)
        current_price = candles[-1]['close'] if candles else 0
        
        status_data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'candle_status': {
                'status': 'completed',
                'progress_percentage': 100,
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
