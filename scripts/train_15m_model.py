import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.ml_model import CryptoEntryModel


def train_15m_model(symbol='BTCUSDT'):
    """Train 15m Bollinger Band bounce prediction model."""
    
    print(f"="*60)
    print(f"Training 15m BB Model for {symbol}")
    print(f"="*60)
    
    try:
        model = CryptoEntryModel(symbol=symbol, timeframe='15m', model_type='xgboost')
        
        print(f"\nStep 1: Loading data...")
        model.load_data()
        
        print(f"\nStep 2: Engineering features...")
        model.engineer_features()
        
        print(f"\nStep 3: Training model...")
        results = model.train()
        
        print(f"\n" + "="*60)
        print(f"Training Results:")
        print(f"="*60)
        print(f"Symbol: {results['symbol']}")
        print(f"Timeframe: {results['timeframe']}")
        print(f"Model Type: {results['model_type']}")
        print(f"Train Accuracy: {results['train_accuracy']:.4f}")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"Train Precision: {results['train_precision']:.4f}")
        print(f"Test Precision: {results['test_precision']:.4f}")
        print(f"="*60)
        
        return results
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None


if __name__ == '__main__':
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'BTCUSDT'
    results = train_15m_model(symbol)
    
    if results:
        print(f"\nTraining completed successfully!")
        print(f"Model saved to: models/cache/{symbol}_15m_xgboost.joblib")
    else:
        print(f"\nTraining failed!")
