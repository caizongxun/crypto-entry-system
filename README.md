# Crypto Entry System - Strategy V3

Advanced machine learning-based cryptocurrency entry signal system using XGBoost multi-output regression with comprehensive technical indicators.

## Features

### Core Capabilities
- **Multi-Output Regression**: Simultaneous prediction of support levels, resistance levels, and breakout probability
- **40+ Technical Indicators**: Comprehensive technical analysis including SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic, ATR, and more
- **Ensemble Learning**: XGBoost models for robust and accurate predictions
- **Automated Signal Generation**: Buy/Sell signals with confidence scoring based on technical indicators
- **Risk Management**: Built-in risk parameters for slippage and commission considerations

### Training Modes
- **train**: Train models on historical data with automatic train/test splitting
- **predict**: Generate real-time signals on latest market data
- **backtest**: Evaluate strategy performance on unseen test data

## Installation

```bash
# Clone the repository
git clone https://github.com/caizongxun/crypto-entry-system.git
cd crypto-entry-system

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- loguru >= 0.7.0
- huggingface-hub >= 0.16.0
- pyarrow >= 12.0.0

## Quick Start

### 1. Train Models

```bash
python main_v3.py --mode train --symbol BTCUSDT --timeframe 15m --verbose
```

This will:
- Download OHLCV data from HuggingFace dataset
- Engineer 40+ technical indicators
- Train XGBoost models for support, resistance, and breakout prediction
- Save models to `./models/v3/`
- Generate training metrics

### 2. Generate Signals

```bash
python main_v3.py --mode predict --symbol BTCUSDT --timeframe 15m --verbose
```

This will:
- Load latest market data
- Generate real-time trading signals
- Display signal summary and recent signals
- Save signals to `./results/v3/`

### 3. Backtest Strategy

```bash
python main_v3.py --mode backtest --symbol BTCUSDT --timeframe 15m --verbose
```

This will:
- Load market data and split into train/test sets
- Generate predictions on unseen test data
- Evaluate strategy performance
- Save backtest results

## Project Structure

```
crypto-entry-system/
├── strategy_v3/
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration dataclasses
│   ├── data_loader.py           # OHLCV data loading and preprocessing
│   ├── feature_engineer.py      # Technical indicator computation
│   ├── models.py                # XGBoost ensemble models
│   └── signal_generator.py      # Trading signal generation
├── main_v3.py                   # Main entry point
├── requirements.txt             # Project dependencies
├── README.md                    # This file
├── models/v3/                   # Saved trained models
├── results/v3/                  # Generated signals and results
└── data_cache/                  # Cached OHLCV data
```

## Configuration

All configuration is managed through `StrategyConfig` dataclasses in `strategy_v3/config.py`:

### Technical Indicators Configuration
```python
config.indicators.sma_short = 10
config.indicators.rsi_period = 14
config.indicators.bb_period = 20
config.indicators.macd_fast = 12
```

### Model Configuration
```python
config.model.max_depth = 6
config.model.learning_rate = 0.05
config.model.n_estimators = 200
```

### Signal Generation Thresholds
```python
config.signals.min_confidence = 0.40
config.signals.buy_signal_threshold = 0.45
config.signals.sell_signal_threshold = 0.45
```

### Predefined Configurations

```python
# Conservative (higher confidence, fewer signals)
config = StrategyConfig.get_conservative()

# Aggressive (lower confidence, more signals)
config = StrategyConfig.get_aggressive()

# Default (balanced)
config = StrategyConfig.get_default()
```

## Technical Indicators

The system computes 40+ technical indicators:

### Trend Indicators
- Simple Moving Averages (SMA): 10, 20, 50 periods
- Exponential Moving Averages (EMA): 12, 26 periods
- MACD (Moving Average Convergence Divergence)

### Momentum Indicators
- RSI (Relative Strength Index)
- Stochastic Oscillator (%K, %D)
- MACD Histogram

### Volatility Indicators
- Bollinger Bands (position, width)
- ATR (Average True Range)

### Support/Resistance
- Dynamic support (20-period rolling minimum)
- Dynamic resistance (20-period rolling maximum)

### Candlestick Patterns
- High-Low ratio
- Open-Close ratio
- Upper/Lower shadows
- Body size

### Volume Analysis
- Volume SMA
- Volume ratio
- Volume change

## Model Architecture

### Support & Resistance Models
- **Target**: Predicted support/resistance levels for next 5 candles
- **Type**: XGBoost Regressor
- **Output**: Continuous price levels
- **Metrics**: RMSE, MAE, R²

### Breakout Model
- **Target**: Probability of price breakout within 5 candles
- **Type**: XGBoost Regressor
- **Output**: Probability [0, 1]
- **Metrics**: RMSE, MAE, R²

## Signal Generation Logic

### Buy Signal Confidence
Calculated from:
1. **RSI Component** (25%): Low RSI indicates oversold
2. **MACD Component** (25%): Positive momentum
3. **Bollinger Bands** (25%): Price near lower band
4. **ATR Component** (25%): Increasing volatility

### Sell Signal Confidence
Calculated from:
1. **RSI Component** (25%): High RSI indicates overbought
2. **MACD Component** (25%): Negative momentum
3. **Bollinger Bands** (25%): Price near upper band
4. **ATR Component** (25%): Increasing volatility

### Signal Rules
- **BUY**: Confidence > threshold AND Price > Support AND Buy > Sell
- **SELL**: Confidence > threshold AND Price < Resistance AND Sell > Buy
- **HOLD**: Otherwise

## Output Files

### Training Results
```
results/v3/{SYMBOL}_{TIMEFRAME}_training_results.csv
- model: Model name (support, resistance, breakout)
- rmse: Root Mean Square Error
- mae: Mean Absolute Error
- r2: R² score
```

### Prediction Signals
```
results/v3/{SYMBOL}_{TIMEFRAME}_latest_signals.csv
- open, high, low, close, volume: OHLCV data
- support, resistance: Predicted levels
- breakout_prob: Breakout probability
- buy_confidence, sell_confidence: Signal confidence scores
- signal_type: BUY/SELL/HOLD
```

### Backtest Results
```
results/v3/{SYMBOL}_{TIMEFRAME}_backtest_signals.csv
- Same format as prediction signals
- Generated on test data (30% of total)
```

## Performance Tips

1. **Data Quality**: Ensure continuous, high-quality OHLCV data
2. **Hyperparameter Tuning**: Adjust model parameters in `config.py` for your market
3. **Signal Tuning**: Modify thresholds for buy/sell confidence based on risk tolerance
4. **Regular Retraining**: Retrain models regularly with fresh data
5. **Symbol Selection**: Works best on liquid cryptocurrencies (BTC, ETH, etc.)

## Supported Symbols & Timeframes

**Symbols**: BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, etc.

**Timeframes**: 15m, 1h, 1d (configurable, based on available data)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - See LICENSE file for details

## Disclaimer

This system is for educational and research purposes. Use at your own risk. Always conduct thorough backtesting and validation before using in live trading.

## Contact

For questions or support, please open an issue on GitHub.
