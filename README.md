# Crypto Entry System

A sophisticated cryptocurrency trading entry system combining TradingView Pine Script indicators with machine learning-based signal filtering for optimized trade entries.

## System Architecture

### Overview

The system implements a three-layer decision architecture to identify high-probability entry points in cryptocurrency markets:

1. **Layer 1: Coarse Entry Signals** - TradingView Pine Script indicators generating broad entry signals
2. **Layer 2: ML-Based Filtering** - Machine learning model evaluating signal quality based on historical patterns
3. **Layer 3: Confirmation** - Dual-confirmation logic ensuring alignment between technical and ML signals

### Layer 1: Initial Entry Signals (Pine Script)

Generates entry signals using multiple technical indicators:

- Simple Moving Average (SMA) crossovers
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index) with overbought/oversold levels
- Bollinger Bands breakouts

**Key Feature:** Deliberately loose thresholds to capture maximum trading opportunities. False signals are filtered in Layer 2.

### Layer 2: ML-Based Filtering

Analyzes the last N completed candles and provides:

- Entry quality score (0-100)
- Multiple diagnostic metrics:
  - Momentum strength indicator
  - Volatility assessment
  - Trend confirmation score
  - Risk-reward ratio estimate

The model uses supervised learning trained on historical cryptocurrency data to predict signal reliability.

### Layer 3: Confirmation Logic

Entry executed only when:
- Layer 1 produces buy/sell signal AND
- Layer 2 quality score exceeds defined threshold

## Data Source

All OHLCV data sourced from HuggingFace Dataset: `zongowo111/v2-crypto-ohlcv-data`

Supported trading pairs: 38 major cryptocurrencies including BTC, ETH, SOL, etc.

Timeframes: 15m, 1h, 1d

## Directory Structure

```
crypto-entry-system/
├── README.md
├── pine_scripts/
│   ├── entry_signal_v1.pine
│   ├── confirmation_filter_v1.pine
│   └── monitoring.pine
├── models/
│   ├── data_processor.py
│   ├── feature_engineer.py
│   ├── ml_model.py
│   ├── signal_evaluator.py
│   ├── config.py
│   └── cache/
└── .gitignore
```

## Technical Details

### Indicators Used

**Momentum Indicators:**
- Relative Strength Index (RSI, period=14)
- MACD (fast=12, slow=26, signal=9)

**Trend Indicators:**
- Simple Moving Average (SMA, periods=20, 50, 200)
- Exponential Moving Average (EMA, periods=12, 26)

**Volatility Indicators:**
- Bollinger Bands (period=20, std_dev=2)
- Average True Range (ATR, period=14)

**Volume Indicators:**
- Volume-weighted metrics for signal confirmation

### Machine Learning Features

Primary features extracted from candlestick data:

1. Price momentum (rate of change)
2. Volatility measure (standard deviation)
3. Volume profile analysis
4. Trend strength (ADX-inspired metric)
5. Support/resistance proximity
6. Historical win rate at similar price levels

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, xgboost
- huggingface_hub for data loading
- Technical analysis libraries (ta-lib or equivalent)

## Installation

```bash
git clone https://github.com/caizongxun/crypto-entry-system.git
cd crypto-entry-system
pip install -r requirements.txt
```

## Usage

### Pine Script Setup

1. Copy `entry_signal_v1.pine` into TradingView Pine Script Editor
2. Add to chart on cryptocurrency pairs at 15-minute timeframe
3. Configure alert conditions for buy/sell signals

### ML Model Training and Inference

```python
from models.ml_model import CryptoEntryModel

model = CryptoEntryModel(symbol='BTCUSDT', timeframe='15m')
model.load_data()
model.engineer_features()
model.train()
signals = model.evaluate_entries()
```

## Performance Metrics

The system tracks:
- Win rate on generated signals
- Average profit per winning trade
- Drawdown statistics
- Signal false-positive rate

## Future Enhancements

- Multi-timeframe confluence analysis
- Real-time signal websocket streaming
- Portfolio-level optimization
- Advanced ensemble methods

## References

- Cryptocurrency Trading Strategy: Deep Q-Learning (2024)
- Blockchain Metrics in Cryptocurrency Trading (2024)
- Machine Learning for Bitcoin Price Prediction (2023)
- Technical Analysis Indicators Comprehensive Review (2020)

## License

MIT License

## Author

Cryptocurrency trading system developer