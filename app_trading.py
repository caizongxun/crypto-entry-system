# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
from datetime import datetime
import os
import sys
from dotenv import load_dotenv
import joblib
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from models.ml_model import CryptoEntryModel
from models.paper_trading import PaperTradingEngine
from models.data_processor import DataProcessor
from models.realtime_data_fetcher import RealtimeDataFetcher
from models.feature_engineer import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
CORS(app)

ml_model_15m = None
paper_trading = PaperTradingEngine(initial_balance=10000.0)
realtime_fetcher = None

try:
    binance_api_key = os.getenv('BINANCE_API_KEY', '')
    binance_api_secret = os.getenv('BINANCE_API_SECRET', '')
    
    realtime_fetcher = RealtimeDataFetcher(
        data_source='binance' if binance_api_key else 'yfinance',
        api_key=binance_api_key,
        api_secret=binance_api_secret
    )
    logger.info(f"Realtime data fetcher initialized with source: {realtime_fetcher.data_source}")
except Exception as e:
    logger.warning(f"Could not initialize realtime fetcher: {str(e)}")
    realtime_fetcher = None

try:
    ml_model_15m = CryptoEntryModel('BTCUSDT', '15m', model_type='lightgbm')
    ml_model_15m.load_model()
    logger.info('ML Model (15m, LightGBM) loaded successfully')
except Exception as e:
    logger.warning(f'ML model not loaded: {str(e)}')

BITGET_MAKER_FEE = 0.0002
BITGET_TAKER_FEE = 0.0005

@app.route('/')
def index():
    return render_template('trading_dashboard.html')

@app.route('/api/realtime/candles', methods=['GET'])
def get_realtime_candles():
    """Fetch completed candles for realtime analysis"""
    try:
        if not realtime_fetcher:
            return jsonify({
                'status': 'error',
                'message': 'Realtime data fetcher not initialized'
            }), 500

        symbol = request.args.get('symbol', 'BTCUSDT')
        timeframe = request.args.get('timeframe', '15m')
        limit = int(request.args.get('limit', 50))
        limit = max(limit, 50)

        df = realtime_fetcher.get_completed_candles(symbol, timeframe, limit)

        if df.empty:
            return jsonify({
                'status': 'error',
                'message': f'No data available for {symbol} {timeframe}'
            }), 404

        candles = []
        for idx, row in df.iterrows():
            candles.append({
                'open_time': row['open_time'].isoformat() if hasattr(row['open_time'], 'isoformat') else str(row['open_time']),
                'close_time': row['close_time'].isoformat() if hasattr(row['close_time'], 'isoformat') else str(row['close_time']),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })

        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'timeframe': timeframe,
            'data_source': realtime_fetcher.data_source,
            'candles': candles,
            'total_candles': len(candles),
            'note': 'All candles are completed (excluding current forming candle)'
        })

    except Exception as e:
        logger.error(f"Error fetching realtime candles: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/realtime/prediction', methods=['GET'])
def get_realtime_prediction():
    """Get ML prediction based on realtime completed candles"""
    try:
        if not ml_model_15m or not realtime_fetcher:
            return jsonify({
                'status': 'error',
                'message': 'ML model or data fetcher not initialized'
            }), 500

        symbol = request.args.get('symbol', 'BTCUSDT')
        timeframe = request.args.get('timeframe', '15m')
        limit = int(request.args.get('limit', 50))
        limit = max(limit, 50)

        df = realtime_fetcher.get_completed_candles(symbol, timeframe, limit)

        if df.empty:
            return jsonify({
                'status': 'error',
                'message': f'No completed candles available'
            }), 404

        if len(df) < 50:
            return jsonify({
                'status': 'warning',
                'message': f'Only {len(df)} completed candles available (need 50 minimum)',
                'available': len(df),
                'required': 50
            }), 400

        df = ml_model_15m.calculate_bb_metrics(df.copy())
        feature_engineer = FeatureEngineer()
        df = feature_engineer.engineer_features(df)

        base_features = [
            'sma_fast', 'sma_medium', 'sma_slow', 'ema_fast', 'ema_slow',
            'rsi', 'macd', 'macd_signal', 'macd_histogram', 'momentum', 'volatility',
            'bb_basis', 'bb_middle', 'bb_width', 'bb_position', 'basis_slope',
            'atr', 'trend_strength', 'obv',
            'volume_momentum', 'price_position', 'volume_relative_strength',
            'close_location', 'momentum_divergence', 'volatility_acceleration',
            'multi_timeframe_strength'
        ]

        feature_cols = [col for col in base_features if col in df.columns]

        if ml_model_15m.selected_feature_names:
            feature_cols = [f for f in ml_model_15m.selected_feature_names if f in df.columns]

        latest_row = df.iloc[-1]

        X_latest = df[feature_cols].tail(1).values
        X_scaled = ml_model_15m.scaler.transform(X_latest)

        bounce_prob = float(ml_model_15m.model.predict_proba(X_scaled)[0, 1])
        prediction = int(ml_model_15m.model.predict(X_scaled)[0])

        signal_type = 'NONE'
        if latest_row.get('touched_lower') or latest_row.get('broke_lower'):
            signal_type = 'BUY' if prediction == 1 else 'NO_SIGNAL_BUY'
        elif latest_row.get('touched_upper') or latest_row.get('broke_upper'):
            signal_type = 'SELL' if prediction == 1 else 'NO_SIGNAL_SELL'

        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'timeframe': timeframe,
            'data_source': realtime_fetcher.data_source,
            'prediction': {
                'bounce_probability': bounce_prob,
                'bounce_probability_pct': round(bounce_prob * 100, 2),
                'prediction': prediction,
                'signal_type': signal_type,
                'confidence': 'HIGH' if bounce_prob > 0.7 else 'MEDIUM' if bounce_prob > 0.5 else 'LOW'
            },
            'latest_candle': {
                'open_time': latest_row['open_time'].isoformat() if hasattr(latest_row['open_time'], 'isoformat') else str(latest_row['open_time']),
                'close_time': latest_row['close_time'].isoformat() if hasattr(latest_row['close_time'], 'isoformat') else str(latest_row['close_time']),
                'close': float(latest_row['close']),
                'bb_upper': float(latest_row.get('bb_upper', 0)),
                'bb_basis': float(latest_row.get('bb_basis', 0)),
                'bb_lower': float(latest_row.get('bb_lower', 0)),
                'bb_position': float(latest_row.get('bb_position', 0)),
                'rsi': float(latest_row.get('rsi', 0))
            },
            'technical_indicators': {
                'rsi': float(latest_row.get('rsi', 0)),
                'volatility': float(latest_row.get('volatility', 0)),
                'trend_strength': float(latest_row.get('trend_strength', 0)),
                'momentum': float(latest_row.get('momentum', 0))
            },
            'candles_used': len(df),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting realtime prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/realtime/candle-status', methods=['GET'])
def get_candle_status():
    """Get current candle formation status"""
    try:
        if not realtime_fetcher:
            return jsonify({
                'status': 'error',
                'message': 'Realtime data fetcher not initialized'
            }), 500

        symbol = request.args.get('symbol', 'BTCUSDT')
        timeframe = request.args.get('timeframe', '15m')

        status = realtime_fetcher.get_candle_status(symbol, timeframe)

        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'timeframe': timeframe,
            'candle_status': status,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting candle status: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/ml-prediction-15m', methods=['GET'])
def get_ml_prediction_15m():
    """Legacy endpoint: Get ML prediction from historical data"""
    try:
        if not ml_model_15m:
            return jsonify({'status': 'error', 'message': 'ML model not loaded'}), 500

        symbol = request.args.get('symbol', 'BTCUSDT')

        eval_results = ml_model_15m.evaluate_entries(lookback=50)

        bb_events = eval_results[
            (eval_results['is_bb_touch'] | eval_results['is_bb_break']) &
            (eval_results['signal_type'] != 'none')
        ]

        predictions = []
        for idx, row in bb_events.tail(20).iterrows():
            signal_type = 'BUY' if row['signal_type'] == 'lower_touch' or row['signal_type'] == 'lower_break' else 'SELL'
            bounce_prob = float(row['bounce_probability'])

            predictions.append({
                'timestamp': str(row['open_time']),
                'signal_type': signal_type,
                'price': float(row['close']),
                'bounce_probability': bounce_prob,
                'bb_position': float(row['bb_position']),
                'bb_width': float(row['bb_width']),
                'bb_upper': float(row.get('bb_upper', 0)),
                'bb_lower': float(row.get('bb_lower', 0)),
                'bb_basis': float(row.get('bb_basis', 0))
            })

        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'timeframe': '15m',
            'model': 'LightGBM',
            'predictions': predictions,
            'total_events': len(bb_events),
            'model_metrics': {
                'precision': 0.6919,
                'recall': 0.8043,
                'f1': 0.7439
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/trading/open-position', methods=['POST'])
def open_position():
    try:
        data = request.json

        entry_price = float(data.get('entry_price', 0))
        quantity = float(data.get('quantity', 0))

        taker_fee = entry_price * quantity * BITGET_TAKER_FEE

        result = paper_trading.open_position(
            symbol=data.get('symbol', 'BTCUSDT'),
            order_type=data.get('order_type', 'BUY'),
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=float(data.get('stop_loss', 0)),
            take_profit=float(data.get('take_profit', 0)),
            notes=data.get('notes', '')
        )

        if result['status'] == 'success':
            result['fee'] = taker_fee
            result['fee_rate'] = BITGET_TAKER_FEE

        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/trading/close-position', methods=['POST'])
def close_position():
    try:
        data = request.json

        result = paper_trading.close_position(
            position_id=data.get('position_id'),
            close_price=float(data.get('close_price', 0)),
            notes=data.get('notes', '')
        )

        if result['status'] == 'success' and 'position' in result:
            maker_fee = result['position']['quantity'] * result['position'].get('close_price', 0) * BITGET_MAKER_FEE
            result['fee'] = maker_fee

        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/trading/account-summary', methods=['GET'])
def get_account_summary():
    try:
        summary = paper_trading.get_account_summary()
        positions = paper_trading.get_positions()
        trade_history = paper_trading.get_trade_history(limit=50)

        return jsonify({
            'status': 'success',
            'summary': summary,
            'positions': positions,
            'trade_history': trade_history,
            'fees': {
                'maker': BITGET_MAKER_FEE,
                'taker': BITGET_TAKER_FEE
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/system/status', methods=['GET'])
def get_system_status():
    """Get system status and available data sources"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'ml_model': 'loaded' if ml_model_15m else 'not_loaded',
            'realtime_fetcher': realtime_fetcher.data_source if realtime_fetcher else 'not_initialized',
            'paper_trading': 'ready'
        },
        'data_source': realtime_fetcher.data_source if realtime_fetcher else 'unavailable'
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model': 'LightGBM loaded' if ml_model_15m else 'not loaded',
        'trading': 'ready',
        'realtime_data': realtime_fetcher.data_source if realtime_fetcher else 'unavailable',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
