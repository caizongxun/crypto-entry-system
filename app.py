import os
import logging
from flask import Flask, render_template
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
app.config['JSON_AS_ASCII'] = False

from trading.auto_trader import AutoTrader
from web.routes.trading_routes import trading_bp

app.register_blueprint(trading_bp)

app.auto_trader = AutoTrader(
    initial_balance=1000.0,
    position_size_percent=0.1,
    confidence_threshold=0.5
)

logger.info('Auto Trader initialized with: initial_balance=1000, position_size=10%, confidence_threshold=50%')

@app.route('/')
def dashboard():
    return render_template('trading_dashboard.html')

@app.route('/health')
def health():
    return {'status': 'healthy'}

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
