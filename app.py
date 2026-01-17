from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
from datetime import datetime, timedelta
import requests
from binance.client import Client
import os
from dotenv import load_dotenv

from models.ml_model import CryptoEntryModel
from models.on_chain_data import OnChainDataProvider
from models.market_sentiment import MarketSentimentAnalyzer
from models.paper_trading import PaperTradingEngine

load_dotenv()

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
CORS(app)

ml_model = CryptoEntryModel('BTCUSDT', '1h')
ml_model.load_model()

on_chain_provider = OnChainDataProvider('BTCUSDT')
on_chain_provider.set_api_keys(
    glassnode_key=os.getenv('GLASSNODE_API_KEY'),
    alternative_key=os.getenv('ALTERNATIVE_API_KEY')
)

sentiment_analyzer = MarketSentimentAnalyzer('BTC')

paper_trading = PaperTradingEngine(initial_balance=10000.0)

binance_api_key = os.getenv('BINANCE_API_KEY')
binance_api_secret = os.getenv('BINANCE_API_SECRET')

if binance_api_key and binance_api_secret:
    client = Client(api_key=binance_api_key, api_secret=binance_api_secret)
else:
    client = None
    print('Warning: Binance API keys not configured. Price data will use public endpoints.')


@app.route('/')
def index():
    """Render main dashboard."""
    return render_template('dashboard.html')


@app.route('/api/price', methods=['GET'])
def get_price():
    """Get current price data from Binance."""
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        interval = request.args.get('interval', '1h')
        limit = int(request.args.get('limit', 100))
        
        if not symbol.endswith('USDT'):
            symbol = symbol + 'USDT'
        
        klines = requests.get(
            'https://api.binance.com/api/v3/klines',
            params={
                'symbol': symbol,
                'interval': interval,
                'limit': min(limit, 1000)
            },
            timeout=10
        ).json()
        
        if isinstance(klines, dict) and 'code' in klines:
            return jsonify({'status': 'error', 'message': klines.get('msg', 'Binance API error')}), 400
        
        data = {
            'timestamps': [],
            'opens': [],
            'highs': [],
            'lows': [],
            'closes': [],
            'volumes': []
        }
        
        for kline in klines:
            data['timestamps'].append(datetime.fromtimestamp(kline[0] / 1000).strftime('%Y-%m-%d %H:%M:%S'))
            data['opens'].append(float(kline[1]))
            data['highs'].append(float(kline[2]))
            data['lows'].append(float(kline[3]))
            data['closes'].append(float(kline[4]))
            data['volumes'].append(float(kline[7]))
        
        result = {
            'status': 'success',
            'symbol': symbol,
            'interval': interval,
            'data': data,
            'latest': {
                'price': data['closes'][-1],
                'timestamp': data['timestamps'][-1]
            }
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/ml-prediction', methods=['GET'])
def get_ml_prediction():
    """Get ML model predictions."""
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        
        ml_model_instance = CryptoEntryModel(symbol, '1h')
        ml_model_instance.load_model()
        
        eval_results = ml_model_instance.evaluate_entries(lookback=50)
        
        bb_events = eval_results[
            (eval_results['is_bb_touch'] | eval_results['is_bb_break']) &
            (eval_results['signal_type'] != 'none')
        ]
        
        predictions = []
        for idx, row in bb_events.tail(20).iterrows():
            predictions.append({
                'timestamp': str(row['open_time']),
                'signal_type': row['signal_type'],
                'price': float(row['close']),
                'bounce_probability': float(row['bounce_probability']),
                'bb_position': float(row['bb_position']),
                'bb_width': float(row['bb_width'])
            })
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'predictions': predictions,
            'total_events': len(bb_events)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/sentiment', methods=['GET'])
def get_sentiment():
    """Get market sentiment data."""
    try:
        fgi_data = sentiment_analyzer.get_fear_greed_index()
        funding_data = sentiment_analyzer.get_funding_rates()
        
        return jsonify({
            'status': 'success',
            'fear_greed_index': fgi_data,
            'funding_rates': funding_data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/on-chain', methods=['GET'])
def get_on_chain_data():
    """Get on-chain data."""
    try:
        on_chain_data = on_chain_provider.get_all_on_chain_data()
        
        return jsonify({
            'status': 'success',
            'data': on_chain_data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/trading/open-position', methods=['POST'])
def open_position():
    """Open a new position."""
    try:
        data = request.json
        
        result = paper_trading.open_position(
            symbol=data.get('symbol', 'BTCUSDT'),
            order_type=data.get('order_type', 'BUY'),
            quantity=float(data.get('quantity', 0.1)),
            entry_price=float(data.get('entry_price', 0)),
            stop_loss=float(data.get('stop_loss', 0)),
            take_profit=float(data.get('take_profit', 0)),
            notes=data.get('notes', '')
        )
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/trading/close-position', methods=['POST'])
def close_position():
    """Close a position."""
    try:
        data = request.json
        
        result = paper_trading.close_position(
            position_id=data.get('position_id'),
            close_price=float(data.get('close_price', 0)),
            notes=data.get('notes', '')
        )
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/trading/account-summary', methods=['GET'])
def get_account_summary():
    """Get trading account summary."""
    try:
        summary = paper_trading.get_account_summary()
        positions = paper_trading.get_positions()
        trade_history = paper_trading.get_trade_history(limit=20)
        
        return jsonify({
            'status': 'success',
            'summary': summary,
            'positions': positions,
            'trade_history': trade_history
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/trading/update-price', methods=['POST'])
def update_price():
    """Update market price for positions."""
    try:
        data = request.json
        paper_trading.update_market_price(
            symbol=data.get('symbol', 'BTCUSDT'),
            current_price=float(data.get('price', 0))
        )
        
        return jsonify({'status': 'success', 'message': 'Price updated'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
