from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
from datetime import datetime, timedelta
import requests
import os
import sys
from dotenv import load_dotenv
import yfinance as yf
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from models.ml_model import CryptoEntryModel
from models.on_chain_data import OnChainDataProvider
from models.market_sentiment import MarketSentimentAnalyzer
from models.paper_trading import PaperTradingEngine
from app.services.whale_tracking import WhaleTracker
from app.services.exchange_flow import ExchangeFlowAnalyzer
from app.routes.on_chain import on_chain_bp

try:
    from binance.client import Client
except ImportError:
    Client = None
    print('Warning: python-binance not installed. Some features may not work.')

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
CORS(app)

app.register_blueprint(on_chain_bp)

try:
    ml_model_1h = CryptoEntryModel('BTCUSDT', '1h')
    ml_model_1h.load_model()
except Exception as e:
    print(f'Warning: Could not load 1h ML model: {str(e)}')
    ml_model_1h = None

try:
    ml_model_15m = CryptoEntryModel('BTCUSDT', '15m')
    ml_model_15m.load_model()
except Exception as e:
    print(f'Warning: Could not load 15m ML model: {str(e)}')
    ml_model_15m = None

try:
    on_chain_provider = OnChainDataProvider('BTCUSDT')
    on_chain_provider.set_api_keys(
        glassnode_key=os.getenv('GLASSNODE_API_KEY'),
        alternative_key=os.getenv('ALTERNATIVE_API_KEY')
    )
except Exception as e:
    print(f'Warning: Could not initialize on-chain provider: {str(e)}')
    on_chain_provider = None

try:
    sentiment_analyzer = MarketSentimentAnalyzer('BTC')
except Exception as e:
    print(f'Warning: Could not initialize sentiment analyzer: {str(e)}')
    sentiment_analyzer = None

try:
    paper_trading = PaperTradingEngine(initial_balance=10000.0)
except Exception as e:
    print(f'Warning: Could not initialize paper trading: {str(e)}')
    paper_trading = None

try:
    whale_tracker = WhaleTracker(api_key=os.getenv('GLASSNODE_API_KEY'))
    exchange_analyzer = ExchangeFlowAnalyzer(api_key=os.getenv('GLASSNODE_API_KEY'))
except Exception as e:
    print(f'Warning: Could not initialize whale tracking services: {str(e)}')
    whale_tracker = None
    exchange_analyzer = None

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
    print('Warning: Binance API keys not configured or python-binance not installed. Using public endpoints.')


class RealtimeCandleProvider:
    TIMEFRAME_MS = {
        '15m': 15 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000
    }
    
    TIMEFRAME_YFINANCE = {
        '15m': '15m',
        '1h': '1h',
        '1d': '1d'
    }
    
    @staticmethod
    def _convert_interval_to_binance(timeframe):
        interval_map = {'15m': '15m', '1h': '1h', '1d': '1d'}
        return interval_map.get(timeframe, '1h')
    
    @staticmethod
    def get_completed_candles_binance(symbol, timeframe, limit=50):
        try:
            if limit < 50:
                limit = 50
            
            interval = RealtimeCandleProvider._convert_interval_to_binance(timeframe)
            
            klines = requests.get(
                'https://api.binance.com/api/v3/klines',
                params={
                    'symbol': symbol,
                    'interval': interval,
                    'limit': min(limit + 1, 1000)
                },
                timeout=10
            ).json()
            
            if isinstance(klines, dict) and 'code' in klines:
                return None
            
            candles = []
            current_time = int(time.time() * 1000)
            timeframe_ms = RealtimeCandleProvider.TIMEFRAME_MS.get(timeframe, 60 * 60 * 1000)
            
            for kline in klines[:-1]:
                open_time = kline[0]
                is_completed = (current_time - open_time) >= timeframe_ms
                
                if is_completed:
                    candles.append({
                        'open_time': open_time,
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[7]),
                        'is_completed': True
                    })
            
            return candles[-limit:] if len(candles) >= limit else (candles if candles else None)
        
        except Exception as e:
            print(f'Error fetching from Binance: {str(e)}')
            return None
    
    @staticmethod
    def get_completed_candles_yfinance(symbol, timeframe, limit=50):
        try:
            if limit < 50:
                limit = 50
            
            ticker = yf.Ticker(symbol)
            yf_interval = RealtimeCandleProvider.TIMEFRAME_YFINANCE.get(timeframe, '1h')
            
            df = ticker.history(interval=yf_interval, period='60d')
            
            if df.empty:
                return None
            
            candles = []
            current_time = pd.Timestamp.now()
            timeframe_td = pd.Timedelta(minutes=15 if timeframe == '15m' else (60 if timeframe == '1h' else 24*60))
            
            for idx, row in df.iterrows():
                if (current_time - idx) >= timeframe_td * 0.9:
                    candles.append({
                        'open_time': int(idx.timestamp() * 1000),
                        'open': float(row['Open']),
                        'high': float(row['High']),
                        'low': float(row['Low']),
                        'close': float(row['Close']),
                        'volume': float(row['Volume']) if 'Volume' in row else 0,
                        'is_completed': True
                    })
            
            return candles[-limit:] if len(candles) >= limit else (candles if candles else None)
        
        except Exception as e:
            print(f'Error fetching from yfinance: {str(e)}')
            return None
    
    @staticmethod
    def get_current_candle_status(symbol, timeframe, data_source='binance'):
        try:
            if data_source == 'binance':
                interval = RealtimeCandleProvider._convert_interval_to_binance(timeframe)
                klines = requests.get(
                    'https://api.binance.com/api/v3/klines',
                    params={'symbol': symbol, 'interval': interval, 'limit': 2},
                    timeout=10
                ).json()
                
                if isinstance(klines, dict) or len(klines) < 2:
                    return None
                
                current_candle = klines[-1]
                open_time = current_candle[0]
                current_time = int(time.time() * 1000)
                timeframe_ms = RealtimeCandleProvider.TIMEFRAME_MS.get(timeframe, 60 * 60 * 1000)
                
                progress = ((current_time - open_time) / timeframe_ms) * 100
                
                return {
                    'status': 'completed' if progress >= 100 else 'forming',
                    'progress_percentage': min(100, round(progress, 2)),
                    'open_time': open_time,
                    'current_price': float(current_candle[4]),
                    'open_price': float(current_candle[1]),
                    'high': float(current_candle[2]),
                    'low': float(current_candle[3])
                }
            else:
                ticker = yf.Ticker(symbol)
                yf_interval = RealtimeCandleProvider.TIMEFRAME_YFINANCE.get(timeframe, '1h')
                df = ticker.history(interval=yf_interval, period='5d')
                
                if df.empty:
                    return None
                
                current_row = df.iloc[-1]
                open_time_idx = df.index[-1]
                current_time = pd.Timestamp.now()
                timeframe_td = pd.Timedelta(minutes=15 if timeframe == '15m' else (60 if timeframe == '1h' else 24*60))
                
                elapsed = (current_time - open_time_idx)
                progress = (elapsed / timeframe_td) * 100
                
                return {
                    'status': 'completed' if progress >= 100 else 'forming',
                    'progress_percentage': min(100, round(progress.total_seconds() / timeframe_td.total_seconds() * 100, 2)),
                    'open_time': int(open_time_idx.timestamp() * 1000),
                    'current_price': float(current_row['Close']),
                    'open_price': float(current_row['Open']),
                    'high': float(current_row['High']),
                    'low': float(current_row['Low'])
                }
        
        except Exception as e:
            print(f'Error getting candle status: {str(e)}')
            return None


@app.route('/')
def index():
    return render_template('trading_dashboard.html')


@app.route('/15m-analysis')
def analysis_15m():
    return render_template('15m_analysis.html')


@app.route('/chart')
def advanced_chart():
    return render_template('chart.html')


@app.route('/api/price', methods=['GET'])
def get_price():
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


@app.route('/api/system/status', methods=['GET'])
def system_status():
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'ml_model': 'loaded' if ml_model_15m else 'not_initialized',
            'realtime_fetcher': 'initialized'
        }
    })


@app.route('/api/realtime/prediction', methods=['GET'])
def realtime_prediction():
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        timeframe = request.args.get('timeframe', '15m')
        limit = int(request.args.get('limit', 50))
        data_source = request.args.get('source', 'binance')
        
        if not symbol.endswith('USDT'):
            symbol = symbol + 'USDT'
        
        if limit < 50:
            limit = 50
        
        if data_source == 'binance':
            candles = RealtimeCandleProvider.get_completed_candles_binance(symbol, timeframe, limit)
        else:
            candles = RealtimeCandleProvider.get_completed_candles_yfinance(symbol, timeframe, limit)
        
        if not candles or len(candles) < 50:
            return jsonify({
                'status': 'error',
                'message': f'Insufficient completed candles. Got {len(candles) if candles else 0}, need at least 50'
            }), 400
        
        df = pd.DataFrame(candles)
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        
        df['bb_basis'] = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_basis'] + (std * 2)
        df['bb_lower'] = df['bb_basis'] - (std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width']
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        
        latest_row = df.iloc[-1]
        
        signal_type = 'NEUTRAL'
        bounce_probability = 0.5
        confidence = 'LOW'
        
        if latest_row['bb_upper'] > 0:
            if latest_row['close'] >= latest_row['bb_upper']:
                signal_type = 'SELL_SIGNAL'
                bounce_probability = min(0.95, 0.6 + (1 - latest_row['bb_position']) * 0.3)
                confidence = 'HIGH'
            elif latest_row['close'] <= latest_row['bb_lower']:
                signal_type = 'BUY_SIGNAL'
                bounce_probability = min(0.95, 0.6 + latest_row['bb_position'] * 0.3)
                confidence = 'HIGH'
            elif latest_row['bb_position'] > 0.8:
                signal_type = 'POTENTIAL_SELL'
                bounce_probability = min(0.9, 0.5 + (1 - latest_row['bb_position']) * 0.3)
                confidence = 'MEDIUM'
            elif latest_row['bb_position'] < 0.2:
                signal_type = 'POTENTIAL_BUY'
                bounce_probability = min(0.9, 0.5 + latest_row['bb_position'] * 0.3)
                confidence = 'MEDIUM'
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'timeframe': timeframe,
            'data_source': data_source,
            'candles_used': len(candles),
            'latest_candle': {
                'timestamp': latest_row['timestamp'].isoformat(),
                'open': float(latest_row['open']),
                'high': float(latest_row['high']),
                'low': float(latest_row['low']),
                'close': float(latest_row['close']),
                'volume': float(latest_row['volume']),
                'bb_upper': float(latest_row['bb_upper']) if pd.notna(latest_row['bb_upper']) else None,
                'bb_basis': float(latest_row['bb_basis']) if pd.notna(latest_row['bb_basis']) else None,
                'bb_lower': float(latest_row['bb_lower']) if pd.notna(latest_row['bb_lower']) else None,
                'bb_position': float(latest_row['bb_position']) if pd.notna(latest_row['bb_position']) else None,
                'bb_width': float(latest_row['bb_width']) if pd.notna(latest_row['bb_width']) else None
            },
            'technical_indicators': {
                'rsi': float(latest_row['rsi']) if pd.notna(latest_row['rsi']) else None,
                'volatility': float(latest_row['volatility']) if pd.notna(latest_row['volatility']) else None
            },
            'prediction': {
                'signal_type': signal_type,
                'bounce_probability': float(bounce_probability),
                'confidence': confidence
            }
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/realtime/candle-status', methods=['GET'])
def candle_status():
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        timeframe = request.args.get('timeframe', '15m')
        data_source = request.args.get('source', 'binance')
        
        if not symbol.endswith('USDT'):
            symbol = symbol + 'USDT'
        
        status = RealtimeCandleProvider.get_current_candle_status(symbol, timeframe, data_source)
        
        if status is None:
            return jsonify({
                'status': 'error',
                'candle_status': {'status': 'error', 'message': 'Could not fetch candle status'}
            }), 400
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'timeframe': timeframe,
            'data_source': data_source,
            'candle_status': status
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e), 'candle_status': {'status': 'error'}}), 500


@app.route('/api/realtime/candles', methods=['GET'])
def get_candles():
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        timeframe = request.args.get('timeframe', '15m')
        limit = int(request.args.get('limit', 50))
        data_source = request.args.get('source', 'binance')
        
        if not symbol.endswith('USDT'):
            symbol = symbol + 'USDT'
        
        if limit < 10:
            limit = 10
        if limit > 1000:
            limit = 1000
        
        if data_source == 'binance':
            candles = RealtimeCandleProvider.get_completed_candles_binance(symbol, timeframe, limit)
        else:
            candles = RealtimeCandleProvider.get_completed_candles_yfinance(symbol, timeframe, limit)
        
        if not candles:
            return jsonify({
                'status': 'error',
                'message': 'Could not fetch candles from data source'
            }), 400
        
        formatted_candles = []
        for candle in candles:
            formatted_candles.append({
                'open_time': candle['open_time'],
                'timestamp': datetime.fromtimestamp(candle['open_time'] / 1000).isoformat(),
                'open': round(candle['open'], 2),
                'high': round(candle['high'], 2),
                'low': round(candle['low'], 2),
                'close': round(candle['close'], 2),
                'volume': round(candle['volume'], 2),
                'is_completed': candle.get('is_completed', True)
            })
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'timeframe': timeframe,
            'data_source': data_source,
            'candles_count': len(formatted_candles),
            'candles': formatted_candles
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/ml-prediction', methods=['GET'])
def get_ml_prediction():
    try:
        if not ml_model_1h:
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
            'timeframe': '1h',
            'predictions': predictions,
            'total_events': len(bb_events)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/ml-prediction-15m', methods=['GET'])
def get_ml_prediction_15m():
    try:
        if not ml_model_15m:
            return jsonify({'status': 'error', 'message': '15m ML model not initialized'}), 500
        
        symbol = request.args.get('symbol', 'BTCUSDT')
        lookback = int(request.args.get('lookback', 50))
        
        eval_results = ml_model_15m.evaluate_entries(lookback=lookback)
        
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
                'bb_width': float(row['bb_width']),
                'bb_upper': float(row['bb_upper']),
                'bb_lower': float(row['bb_lower']),
                'bb_basis': float(row['bb_basis'])
            })
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'timeframe': '15m',
            'predictions': predictions,
            'total_events': len(bb_events),
            'model_trained': ml_model_15m.model is not None
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/sentiment', methods=['GET'])
def get_sentiment():
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
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'whale_tracker': 'enabled' if whale_tracker else 'disabled',
            'exchange_analyzer': 'enabled' if exchange_analyzer else 'disabled',
            'ml_model_1h': 'loaded' if ml_model_1h else 'not_loaded',
            'ml_model_15m': 'loaded' if ml_model_15m else 'not_loaded',
            'paper_trading': 'enabled' if paper_trading else 'disabled',
            'binance_client': 'connected' if client else 'using_public_endpoints',
            'realtime_provider': 'initialized'
        }
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
