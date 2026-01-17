from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
from datetime import datetime, timedelta
import requests
import os
import sys
from dotenv import load_dotenv

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

# Import models
from models.ml_model import CryptoEntryModel
from models.on_chain_data import OnChainDataProvider
from models.market_sentiment import MarketSentimentAnalyzer
from models.paper_trading import PaperTradingEngine

# Import services
from app.services.whale_tracking import WhaleTracker
from app.services.exchange_flow import ExchangeFlowAnalyzer

# Import routes
from app.routes.on_chain import on_chain_bp

# Try to import Binance client, but handle if not installed
try:
    from binance.client import Client
except ImportError:
    Client = None
    print('Warning: python-binance not installed. Some features may not work.')

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
CORS(app)

# Register blueprints
app.register_blueprint(on_chain_bp)

# Initialize ML model
try:
    ml_model = CryptoEntryModel('BTCUSDT', '1h')
    ml_model.load_model()
except Exception as e:
    print(f'Warning: Could not load ML model: {str(e)}')
    ml_model = None

# Initialize on-chain data provider
try:
    on_chain_provider = OnChainDataProvider('BTCUSDT')
    on_chain_provider.set_api_keys(
        glassnode_key=os.getenv('GLASSNODE_API_KEY'),
        alternative_key=os.getenv('ALTERNATIVE_API_KEY')
    )
except Exception as e:
    print(f'Warning: Could not initialize on-chain provider: {str(e)}')
    on_chain_provider = None

# Initialize sentiment analyzer
try:
    sentiment_analyzer = MarketSentimentAnalyzer('BTC')
except Exception as e:
    print(f'Warning: Could not initialize sentiment analyzer: {str(e)}')
    sentiment_analyzer = None

# Initialize paper trading engine
try:
    paper_trading = PaperTradingEngine(initial_balance=10000.0)
except Exception as e:
    print(f'Warning: Could not initialize paper trading: {str(e)}')
    paper_trading = None

# Initialize new services
try:
    whale_tracker = WhaleTracker(api_key=os.getenv('GLASSNODE_API_KEY'))
    exchange_analyzer = ExchangeFlowAnalyzer(api_key=os.getenv('GLASSNODE_API_KEY'))
except Exception as e:
    print(f'Warning: Could not initialize whale tracking services: {str(e)}')
    whale_tracker = None
    exchange_analyzer = None

# Initialize Binance client
binance_api_key = os.getenv('BINANCE_API_KEY')
binance_api_secret = os.getenv('BINANCE_API_SECRET')

if binance_api_key and binance_api_secret and Client:
    try:
        client = Client(api_key=binance_api_key, api_secret=binance_api_secret)
    except Exception as e:
        print(f'Warning: Could not initialize Binance client: {str(e)}')
        client = None
else:
    client = None
    print('Warning: Binance API keys not configured or python-binance not installed. Price data will use public endpoints.')


@app.route('/')
def index():
    """Render main dashboard."""
    return render_template('dashboard.html')


@app.route('/chart')
def advanced_chart():
    """Render advanced TradingView-style chart."""
    return render_template('chart.html')


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
        if not ml_model:
            return jsonify({'status': 'error', 'message': 'ML model not initialized'}), 500
        
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
        if not sentiment_analyzer:
            return jsonify({'status': 'error', 'message': 'Sentiment analyzer not initialized'}), 500
        
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
        if not on_chain_provider:
            return jsonify({'status': 'error', 'message': 'On-chain provider not initialized'}), 500
        
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
        if not paper_trading:
            return jsonify({'status': 'error', 'message': 'Paper trading not initialized'}), 500
        
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
        if not paper_trading:
            return jsonify({'status': 'error', 'message': 'Paper trading not initialized'}), 500
        
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
        if not paper_trading:
            return jsonify({'status': 'error', 'message': 'Paper trading not initialized'}), 500
        
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
        if not paper_trading:
            return jsonify({'status': 'error', 'message': 'Paper trading not initialized'}), 500
        
        data = request.json
        paper_trading.update_market_price(
            symbol=data.get('symbol', 'BTCUSDT'),
            current_price=float(data.get('price', 0))
        )
        
        return jsonify({'status': 'success', 'message': 'Price updated'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'whale_tracker': 'enabled' if whale_tracker else 'disabled',
            'exchange_analyzer': 'enabled' if exchange_analyzer else 'disabled',
            'ml_model': 'loaded' if ml_model else 'not_loaded',
            'paper_trading': 'enabled' if paper_trading else 'disabled',
            'binance_client': 'connected' if client else 'using_public_endpoints'
        }
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
