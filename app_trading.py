# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
from datetime import datetime
import os
import sys
from dotenv import load_dotenv
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from models.ml_model import CryptoEntryModel
from models.paper_trading import PaperTradingEngine
from models.data_processor import DataProcessor

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
CORS(app)

ml_model_15m = None
paper_trading = PaperTradingEngine(initial_balance=10000.0)

try:
    ml_model_15m = CryptoEntryModel('BTCUSDT', '15m', model_type='lightgbm')
    ml_model_15m.load_model()
    print('ML Model (15m, LightGBM) loaded successfully')
except Exception as e:
    print(f'Warning: Could not load ML model: {str(e)}')

BITGET_MAKER_FEE = 0.0002
BITGET_TAKER_FEE = 0.0005

@app.route('/')
def index():
    return render_template('trading_dashboard.html')

@app.route('/api/ml-prediction-15m', methods=['GET'])
def get_ml_prediction_15m():
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

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model': 'LightGBM loaded' if ml_model_15m else 'not loaded',
        'trading': 'ready',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
