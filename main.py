import sys
from pathlib import Path
from models import CryptoEntryModel


def main():
    """Main execution function for BB bounce prediction model."""
    print("="*70)
    print("Cryptocurrency Entry System - BB Channel Bounce Prediction")
    print("="*70)

    symbol = 'BTCUSDT'
    timeframe = '1h'
    model_type = 'xgboost'

    print(f"\nTarget: {symbol} at {timeframe} timeframe")
    print(f"Model Type: {model_type}")
    print("-"*70)

    model = CryptoEntryModel(symbol=symbol, timeframe=timeframe, model_type=model_type)

    print("\n[Step 1] Loading Data...")
    raw_data = model.load_data()
    print(f"Loaded {len(raw_data)} candles")

    print("\n[Step 2] Engineering Features...")
    feature_data = model.engineer_features()
    print(f"Engineered features: {len(feature_data)} rows, {len(feature_data.columns)} columns")

    print("\n[Step 3] Training ML Model for BB Bounce Prediction...")
    training_results = model.train()
    if training_results:
        print(f"Train Accuracy: {training_results.get('train_accuracy', 0):.4f}")
        print(f"Test Accuracy: {training_results.get('test_accuracy', 0):.4f}")
        print(f"Train Precision: {training_results.get('train_precision', 0):.4f}")
        print(f"Test Precision: {training_results.get('test_precision', 0):.4f}")

    print("\n[Step 4] Evaluating Recent BB Touches/Breaks for Bounce Probability...")
    eval_results = model.evaluate_entries(lookback=50)

    if len(eval_results) > 0:
        print(f"\nAnalyzed {len(eval_results)} candles")

        bb_events = eval_results[
            (eval_results['is_bb_touch'] | eval_results['is_bb_break']) &
            (eval_results['signal_type'] != 'none')
        ]

        if len(bb_events) > 0:
            print(f"\nBB Touch/Break Events Found: {len(bb_events)}")
            print("-" * 70)
            print(f"{'Timestamp':<20} {'Signal':<15} {'Price':<12} {'Bounce Prob':<15} {'Position':<10}")
            print("-" * 70)

            for idx, row in bb_events.tail(10).iterrows():
                timestamp = str(row['open_time'])[:16]
                signal_type = row['signal_type']
                price = f"${row['close']:.2f}"
                bounce_prob = f"{row['bounce_probability']:.1f}%"
                position = f"{row['bb_position']:.2f}"

                print(f"{timestamp:<20} {signal_type:<15} {price:<12} {bounce_prob:<15} {position:<10}")

            print("-" * 70)
            high_prob_events = bb_events[bb_events['bounce_probability'] >= 60]
            if len(high_prob_events) > 0:
                print(f"\nHigh Probability Events (>= 60%): {len(high_prob_events)}")
                print("Recommended for entry consideration.")
        else:
            print("\nNo BB touch/break events in recent data.")

        latest_signal = eval_results.iloc[-1]
        print(f"\n" + "="*70)
        print(f"Latest Candle Analysis:")
        print(f"  Timestamp: {latest_signal['open_time']}")
        print(f"  Price: ${latest_signal['close']:.2f}")
        print(f"  BB Position: {latest_signal['bb_position']:.3f}")
        print(f"  BB Width: ${latest_signal['bb_width']:.2f}")
        print(f"  Bounce Prediction: {'Effective' if latest_signal['bounce_prediction'] == 1 else 'Ineffective'}")
        print(f"  Bounce Probability: {latest_signal['bounce_probability']:.1f}%")
        if 'rsi' in latest_signal.index:
            print(f"  RSI: {latest_signal['rsi']:.2f}")
        if 'macd_histogram' in latest_signal.index:
            print(f"  MACD Histogram: {latest_signal['macd_histogram']:.4f}")
    else:
        print("\nNo valid evaluation data available.")

    print("\n" + "="*70)
    print("System Status: Model training and evaluation completed")
    print("Save the model with model.save_model() after training")
    print("="*70)


if __name__ == '__main__':
    main()