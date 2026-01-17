import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from models.ml_model import CryptoEntryModel


def train_mode(symbol, timeframe, model_type, optimization_level):
    print("="*60)
    print(f"Training {timeframe.upper()} BB Model for {symbol}")
    print(f"Optimization Level: {optimization_level}")
    print("="*60)
    
    try:
        print(f"\nStep 1: Loading data...")
        model = CryptoEntryModel(
            symbol=symbol,
            timeframe=timeframe,
            model_type=model_type,
            optimization_level=optimization_level
        )
        model.load_data()
        
        print(f"\nStep 2: Engineering features...")
        model.engineer_features()
        
        print(f"\nStep 3: Training model...")
        results = model.train()
        
        print(f"\n" + "="*60)
        print(f"Training Results:")
        print("="*60)
        print(f"Symbol: {results['symbol']}")
        print(f"Timeframe: {results['timeframe']}")
        print(f"Model Type: {results['model_type']}")
        print(f"Optimization: {results.get('optimization', 'none')}")
        print(f"Train Accuracy: {results['train_accuracy']:.4f}")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"Train Precision: {results['train_precision']:.4f}")
        print(f"Test Precision: {results['test_precision']:.4f}")
        
        if 'train_recall' in results:
            print(f"Train Recall: {results['train_recall']:.4f}")
            print(f"Test Recall: {results['test_recall']:.4f}")
        
        if 'bounce_rate' in results:
            print(f"Bounce Rate: {results['bounce_rate']:.4f}")
        
        print(f"="*60)
        print(f"\nTraining completed successfully!")
        print(f"Model saved to: models/cache/{symbol}_{timeframe}_{model_type}.joblib")
        return True
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def server_mode():
    from app import app
    print("Starting Crypto Entry System server...")
    print("Access at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)


def main():
    parser = argparse.ArgumentParser(
        description='Crypto Entry System - ML-based Trading Signal Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              Start web server
  python main.py --train                      Train model in interactive mode
  python main.py --symbol BTCUSDT --timeframe 15m             Train with parameters
  python main.py --symbol ETHUSDT --timeframe 1h --opt aggressive
  python main.py --symbol LTCUSDT --timeframe 4h --model-type lightgbm --opt balanced

Optimization Levels:
  conservative    High precision, fewer signals (precision 65-75%)
  balanced        Good precision and recall (precision 55-65%)
  aggressive      More signals, moderate precision (precision 45-55%)
        """
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Enter training mode'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTCUSDT',
        help='Trading pair symbol (default: BTCUSDT)'
    )
    
    parser.add_argument(
        '--timeframe',
        type=str,
        default='1h',
        choices=['15m', '1h', '4h', '1d'],
        help='Timeframe for model training (default: 1h)'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        default='xgboost',
        choices=['xgboost', 'lightgbm', 'random_forest'],
        help='Model algorithm type (default: xgboost)'
    )
    
    parser.add_argument(
        '--opt',
        type=str,
        default='balanced',
        choices=['conservative', 'balanced', 'aggressive'],
        help='Optimization level (default: balanced)'
    )
    
    args = parser.parse_args()
    
    if args.train or args.symbol != 'BTCUSDT' or args.timeframe != '1h' or args.model_type != 'xgboost' or args.opt != 'balanced':
        symbol = args.symbol.upper()
        if not symbol.endswith('USDT'):
            symbol = symbol + 'USDT'
        
        success = train_mode(
            symbol=symbol,
            timeframe=args.timeframe,
            model_type=args.model_type,
            optimization_level=args.opt
        )
        sys.exit(0 if success else 1)
    else:
        server_mode()


if __name__ == '__main__':
    main()
