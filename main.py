import sys
from pathlib import Path
from models import CryptoEntryModel


def main():
    """Main execution function."""
    print("="*70)
    print("Cryptocurrency Entry System - Model Training and Evaluation")
    print("="*70)

    symbol = 'BTCUSDT'
    timeframe = '15m'

    print(f"\nTarget: {symbol} at {timeframe} timeframe")
    print("-"*70)

    model = CryptoEntryModel(symbol=symbol, timeframe=timeframe, model_type='xgboost')

    print("\n[Step 1] Loading Data...")
    raw_data = model.load_data()
    print(f"Loaded {len(raw_data)} candles")

    print("\n[Step 2] Engineering Features...")
    feature_data = model.engineer_features()
    print(f"Engineered features: {len(feature_data)} rows, {len(feature_data.columns)} columns")

    print("\n[Step 3] Training ML Model...")
    training_results = model.train()
    print(f"Train Accuracy: {training_results['train_accuracy']:.4f}")
    print(f"Test Accuracy: {training_results['test_accuracy']:.4f}")

    print("\n[Step 4] Evaluating Latest Signals...")
    eval_results = model.evaluate_entries(lookback=50)
    latest_signal = eval_results.iloc[-1]

    print(f"\nLatest Signal Analysis:")
    print(f"  Timestamp: {latest_signal['open_time']}")
    print(f"  Price: ${latest_signal['close']:.2f}")
    print(f"  Quality Score: {latest_signal['quality_score']:.2f}/100")
    print(f"  Momentum: {latest_signal['momentum_strength']:.2f}")
    print(f"  Trend Score: {latest_signal['trend_score']:.2f}")
    print(f"  Volatility: {latest_signal['volatility_score']:.2f}")
    print(f"  RSI: {latest_signal.get('rsi', 'N/A')}")
    print(f"  ML Probability: {latest_signal['ml_probability']:.2f}%")
    print(f"  Combined Score: {latest_signal['combined_score']:.2f}")

    print("\n" + "="*70)
    print("System Status: Ready for trading signal integration")
    print("="*70)


if __name__ == '__main__':
    main()