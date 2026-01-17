# Crypto Entry System

Advanced machine learning system for identifying Bollinger Band bounce trading opportunities in cryptocurrency markets using multi-timeframe analysis and automatic hyperparameter optimization.

## Overview

This project leverages gradient boosting models (XGBoost, LightGBM) combined with technical analysis to predict effective bounce entries when prices touch or break Bollinger Band boundaries. The system incorporates multi-timeframe feature engineering and automated hyperparameter optimization to maximize precision while maintaining adequate recall rates.

## Key Features

- **Multi-Timeframe Analysis**: Integrates higher timeframe context (1h, 4h) with base timeframe signals (15m, 1h)
- **Advanced Feature Engineering**: 68 technical indicators including price action, momentum, and volatility metrics
- **Automatic Feature Selection**: Identifies 25 most important features to reduce noise and improve model precision
- **Hyperparameter Optimization**: Bayesian optimization using Optuna to find optimal model parameters
- **Balanced Optimization**: Customizable optimization profiles (conservative, balanced, aggressive)
- **Real-Time Evaluation**: Current entry signal assessment with bounce probability estimates

## System Architecture

```
models/
  ml_model.py                  - Core ML model with training pipeline
  feature_engineer.py          - Technical indicator calculation
  multi_timeframe_engineer.py  - Cross-timeframe feature integration
  feature_selector.py          - Automated feature selection module
  hyperparameter_optimizer.py  - Bayesian optimization engine
  signal_evaluator.py          - Signal quality evaluation
  config.py                    - Configuration management
  cache/                       - Model artifacts and optimization results

main.py          - Production training and inference
main_v2.py       - Hyperparameter optimization experiments
README.md        - Documentation
```

## Installation

Requirements:
- Python 3.8 or higher
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0
- lightgbm >= 3.3.0
- optuna >= 3.0.0 (for optimization)
- imbalanced-learn >= 0.8.0 (for SMOTE)

Setup:

```bash
git clone https://github.com/caizongxun/crypto-entry-system.git
cd crypto-entry-system
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Usage Guide

### Production Training

Train model with balanced default parameters:

```bash
python main.py --train --symbol BTCUSDT --timeframe 15m --model-type xgboost --opt balanced
```

Evaluate current trading signals:

```bash
python main.py --eval --symbol BTCUSDT --timeframe 15m
```

### Hyperparameter Optimization

Find optimal hyperparameters using Bayesian optimization (50 trials):

```bash
python main_v2.py --mode optimize --symbol BTCUSDT --timeframe 15m --model-type xgboost --n-trials 50
```

Train using optimized parameters:

```bash
python main_v2.py --mode train --symbol BTCUSDT --timeframe 15m --model-type xgboost \
  --results-path models/cache/optimization/BTCUSDT_15m_xgboost_results.json
```

Compare performance across optimization levels:

```bash
python main_v2.py --mode compare --symbol BTCUSDT --timeframe 15m --model-type xgboost
```

## Model Performance

Current performance on BTCUSDT 15m (with feature selection and improved bounce criteria):

| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 84.30% | 82.88% |
| Precision | 74.24% | 72.41% |
| Recall | 80.99% | 78.58% |
| F1 Score | 0.7745 | 0.7546 |

Expected trading profitability (assuming 2% win / 1% loss ratio):
- Win Rate: 72.41%
- Expected Return per Trade: 1.17%
- Monthly Return (20 trades): 23.4%

## Configuration

Model hyperparameters configured in `models/config.py`:

```python
TIMEFRAME_CONFIGS = {
    '15m': {
        'bb_period': 20,
        'bb_std': 2.0,
        'lookforward': 5,
        'bounce_threshold': 0.005,
    },
    '1h': {...},
}
```

Optimization levels adjust bounce criteria and ensemble settings:
- **Conservative**: Ensemble model, stricter criteria, higher precision
- **Balanced**: Single optimized model, standard criteria, 72-73% precision
- **Aggressive**: Relaxed criteria, maximum trades, requires strict risk management

## Optimization Workflow

### Step 1: Run Optimization
Let Optuna search for best hyperparameters across 50 trials:
```bash
python main_v2.py --mode optimize --symbol BTCUSDT --timeframe 15m --n-trials 50
```

Results saved to: `models/cache/optimization/BTCUSDT_15m_xgboost_results.json`

### Step 2: Review Results
Examine the generated JSON file to verify:
- Best score achieved
- Optimal parameters (max_depth, learning_rate, n_estimators)
- Train/test precision and recall metrics

### Step 3: Train with Optimized Parameters
Train model using discovered optimal hyperparameters:
```bash
python main_v2.py --mode train --results-path models/cache/optimization/BTCUSDT_15m_xgboost_results.json
```

### Step 4: Validate Performance
Use production script to evaluate on real data:
```bash
python main.py --eval --symbol BTCUSDT --timeframe 15m
```

## Feature Engineering

**Base Features (36)**:
- Price action: SMA, EMA, momentum
- Oscillators: RSI, MACD, Stochastic
- Volatility: ATR, Bollinger Bands width
- Volume: OBV, volume momentum

**Multi-Timeframe Features (32)**:
- Timeframe confirmation using RSI alignment
- Trend alignment via SMA comparison
- Volatility context detection
- Momentum divergence analysis

**Selected Features (25)**:
Automatic feature selection identifies most predictive features, reducing from 68 to 25 dimensions.

## Data Requirements

- Minimum 6 months historical OHLCV data for training
- Format: Open, High, Low, Close, Volume, Timestamp
- Supported sources:
  - Binance Futures API
  - Local parquet files

## Risk Management

Before production deployment:

1. Backtest on out-of-sample data (different time periods)
2. Start with position sizes 1-2% of account
3. Implement stop-loss at 1% loss threshold
4. Monitor model performance weekly
5. Reoptimize monthly with fresh data
6. Account for 0.1-0.2% transaction costs
7. Use position sizing appropriate to risk tolerance

## Development Practices

- All source files modified in-place (no version files)
- Optimization results saved to `models/cache/optimization/`
- Results tracked with JSON metadata for reproducibility
- Main production entry point remains `main.py`
- Experimental optimizations use `main_v2.py`
- All code and documentation free of emojis or symbols

## Troubleshooting

**Model precision not improving**:
- Verify feature engineering output
- Check data integrity (no gaps, outliers)
- Increase optimization trials to 100
- Try alternative timeframes

**Memory constraints**:
- Reduce historical data window
- Lower SMOTE sampling ratio
- Disable multi-timeframe features temporarily

**Optuna library missing**:
```bash
pip install optuna
```

## Performance Tuning

To improve precision:

1. Increase optimization trials: `--n-trials 100`
2. Tighten bounce criteria: Lower `bounce_threshold` in config
3. Reduce feature dimensions: Select top 20 features only
4. Use ensemble: Set optimization level to conservative
5. Add domain-specific features: Extend feature engineering

## References

System based on established quantitative trading practices:
- Multi-timeframe technical analysis principles
- Machine learning model selection and optimization
- Proper backtesting and performance evaluation methodology

## License

Private project for quantitative trading research.

## Repository Maintenance

Maintain professional structure:
- Clean commit history with descriptive messages
- No experimental branches in main repository
- Documentation kept current with code changes
- Results tracked in dedicated cache directory
- Consistent code style and naming conventions
