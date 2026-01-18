# Strategy V3 - Advanced Cryptocurrency Entry Signal System

## Overview

Strategy V3 is an advanced machine learning-based cryptocurrency entry signal system that combines XGBoost multi-output regression with comprehensive technical indicators to generate high-confidence trading signals.

## Features

- **Multi-Output XGBoost Regression**: Simultaneously predicts support level, resistance level, and breakout probability
- **Comprehensive Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, and more
- **Signal Generation**: Combines ML predictions with technical analysis for robust signal generation
- **HuggingFace Data Integration**: Seamless loading of OHLCV data from HuggingFace datasets
- **Three Operational Modes**:
  - Train: Build and train models on historical data
  - Predict: Generate signals on latest data
  - Backtest: Test strategy performance on unseen data

## Installation

```bash
pip install -r requirements_v3.txt
```

## Quick Start

### Training

Train the model on BTCUSDT 15m data:

```bash
python main_v3.py --mode train --symbol BTCUSDT --timeframe 15m
```

### Prediction

Generate trading signals on latest data:

```bash
python main_v3.py --mode predict --symbol BTCUSDT --timeframe 15m
```

### Backtesting

Test strategy on historical data:

```bash
python main_v3.py --mode backtest --symbol BTCUSDT --timeframe 15m
```

## Configuration

Modify `strategy_v3/config.py` to customize:

- Technical indicator parameters
- Model hyperparameters
- Signal generation thresholds
- Risk management settings

## System Architecture

```
strategy_v3/
├── __init__.py           # Package initialization
├── config.py             # Configuration classes
├── data_loader.py        # HuggingFace data loading
├── feature_engineer.py   # Technical indicators & feature engineering
├── models.py            # XGBoost multi-output models
└── signal_generator.py  # Trading signal generation

main_v3.py              # Main training/prediction/backtest script
requirements_v3.txt     # Python dependencies
```

## Data Structure

Data is loaded from HuggingFace dataset: `zongowo111/v2-crypto-ohlcv-data`

Path structure:
```
klines/{SYMBOL}/{BASE}_{TIMEFRAME}.parquet
```

Example:
- `klines/BTCUSDT/BTC_15m.parquet`
- `klines/ETHUSDT/ETH_1h.parquet`
- `klines/ADAUSDT/ADA_1d.parquet`

## Model Outputs

The XGBoost ensemble predicts:

1. **Support Level**: Expected support price
2. **Resistance Level**: Expected resistance price
3. **Breakout Probability**: Likelihood of price breaking resistance (0-1)

## Signal Generation Logic

### Buy Signal Requirements
- Price near support level
- RSI below 30 (oversold)
- MACD positive crossover
- High breakout probability
- Minimum confidence threshold: 0.4

### Sell Signal Requirements
- Price near resistance level
- RSI above 70 (overbought)
- MACD negative signal
- Low breakout probability
- Minimum confidence threshold: 0.4

## Error Tolerance

System maintains 0.1% accuracy tolerance for price predictions:
- Slippage: 0.05%
- Commission: 0.1%
- Support/Resistance tolerance: 0.5%

## Output Files

Generated in `results/v3/`:

- `{SYMBOL}_{TIMEFRAME}_training_results.csv`: Training metrics
- `{SYMBOL}_{TIMEFRAME}_latest_signals.csv`: Latest signals
- `{SYMBOL}_{TIMEFRAME}_backtest_signals.csv`: Backtest signals

Models saved in `models/v3/`:

- `support_level_model.pkl`
- `resistance_level_model.pkl`
- `breakout_probability_model.pkl`
- `scaler.pkl`
- `features.pkl`

## Performance Metrics

Training reports include:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R-squared
- Best iteration (early stopping)

## Command Line Options

```
--mode {train, predict, backtest}  Operation mode
--symbol SYMBOL                     Trading symbol (default: BTCUSDT)
--timeframe {15m, 1h, 1d}          Timeframe (default: 15m)
--verbose                          Enable verbose logging
```

## Notes

- First run will download data from HuggingFace (cached for subsequent runs)
- Training time varies with data size (typically 5-15 minutes for 1 year of data)
- Backtest uses 70% train / 30% test split
- All timestamps are UTC

## Limitations

- Models are trained on historical data and may not capture future market regime changes
- Strategy does not account for gap events or market holidays
- Slippage and commission estimates may differ from actual execution
- Technical indicators may lag in highly volatile markets

## Future Enhancements

- LSTM neural networks for temporal pattern recognition
- Sentiment analysis integration
- Multi-symbol correlation analysis
- Real-time order execution integration
- Advanced portfolio optimization
