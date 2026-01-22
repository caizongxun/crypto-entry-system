# Pine Script Indicators

TradingView Pine Script v5 indicators designed for cryptocurrency trading strategies.

## Available Indicators

### 1. Momentum Divergence Indicator (momentum_divergence_indicator.pine)

Comprehensive momentum analysis tool combining multiple indicators for divergence detection.

**Features:**
- Momentum with EMA smoothing for trend confirmation
- RSI with overbought/oversold detection (levels: 30, 70)
- KDJ Stochastic with signal line crossing
- Bollinger Bands integration for volatility context
- Automated divergence detection
- Signal strength scoring system

**Parameters:**
- BB Length: 20 (default)
- BB Std Dev: 2.0 (default)
- Momentum Period: 12 (default)
- RSI Period: 14 (default)
- KDJ Period: 9 (default)

**Signals:**
- U Symbol: Bullish divergence (RSI or KDJ + momentum acceleration)
- D Symbol: Bearish divergence (RSI or KDJ + momentum deceleration)
- Green Bars: Positive momentum trend
- Red Bars: Negative momentum trend

---

### 2. Squeeze Momentum BB Enhanced (squeeze_momentum_bb_enhanced.pine)

Advanced momentum indicator addressing lag reduction through squeeze detection and linear regression.

**Features:**
- Bollinger Bands and Keltner Channels squeeze detection
- Linear regression momentum calculation for faster response
- Dual EMA smoothing (3-period and 8-period)
- Momentum strength meter
- Squeeze release breakout signals
- Automatic divergence alerts

**Parameters:**
- Basis Length: 20 (default)
- BB Length: 20, Multiplier: 2.0 (default)
- KC Length: 20, Multiplier: 1.5 (default)
- Momentum Length: 12 (default)
- Momentum Smooth: 2 (default)

**Signals:**
- B Symbol: Bullish divergence at squeeze release
- S Symbol: Bearish divergence at squeeze release
- Q Symbol: Squeeze status (minimal volatility)
- Green Gradient: Strengthening bullish momentum
- Red Gradient: Strengthening bearish momentum

**Lag Reduction Methods:**
1. Linear Regression: Responds faster to momentum changes than simple differences
2. Squeeze Detection: Identifies low-volatility periods before expansion
3. Dual EMA: Fast EMA (3) for immediate response, slow EMA (8) for confirmation
4. Keltner Channels: ATR-based channels provide dynamic support/resistance

---

## Integration with Bollinger Bands

### Optimal Strategy

1. **Squeeze Identification Phase**
   - Monitor Bollinger Bands width (BB Upper - BB Lower) / Middle
   - When width < 0.05 (5%): Market in squeeze
   - Historical volatility compression indicates breakout potential

2. **Momentum Divergence Detection**
   - Track RSI or KDJ crossing key levels (30/70)
   - If price creates lower low but momentum creates higher low: Bullish divergence
   - If price creates higher high but momentum creates lower high: Bearish divergence

3. **Confirmation Rules**
   - Squeeze MUST be released (BB width expanding)
   - Momentum EMA cross (fast crossing slow)
   - BB Bands boundary break (close beyond upper or lower band)
   - Entry triggered at confluence of 2+ signals

### Configuration for Crypto (15m - 1h timeframes)

```
Momentum Divergence Indicator:
- BB Length: 18-22
- BB Std Dev: 1.8-2.2
- Momentum Period: 10-14
- RSI Period: 12-16
- KDJ Period: 7-11

Squeeze Momentum Enhanced:
- Basis Length: 18-22
- BB Multiplier: 1.8-2.2
- KC Multiplier: 1.3-1.7
- Momentum Length: 10-14
```

---

## Pine Script Syntax Critical Points

### Line Continuation Rules

**INCORRECT - WILL CAUSE 'end' ERROR:**
```pine
myvar = input1 + input2 +
        input3

myvar = 
        input1 + input2
```

**CORRECT - Use proper indentation:**
```pine
myvar = input1 + input2 +
     input3

myvar = input1 + input2 +
     input3
```

Key rules:
- Operator (+, -, *, /) MUST be at end of line
- Next line indentation: Use 1, 2, 3, 5, 6, 7 spaces (NOT 4, 8, 12 - reserved for blocks)
- NO backslash continuation (\\)
- All operators, assignments, and function parameters must follow operator-end rule

### Variable Declaration Scope

```pine
// Global scope
global_var = 0

if condition
    // This variable accessible outside if block
    scoped_var := 10

// Access scoped_var here - it exists in parent scope
plot(scoped_var)

// Single initialization across all bars
var persistent_var = 0
if condition
    persistent_var := persistent_var + 1
```

### Common Patterns

**Ternary Operator (Line Continuation Safe):**
```pine
color_value = mom > 0 ? 
     color.green : color.red

color_value = mom > 0 ? (mom > mom[1] ? 
     color.new(color.green, 0) : color.new(color.lime, 40)) : 
     (mom < mom[1] ? color.new(color.red, 0) : color.new(color.maroon, 40))
```

**Function Calls (Operator-End Safe):**
```pine
result = ta.sma(source1, 20) +
     ta.ema(source2, 14)

alert_cond = condition1 and
     condition2 and
     condition3
```

---

## Usage Instructions

1. Copy indicator code from .pine files
2. Open TradingView Chart -> Pine Script Editor
3. Create New Script -> Paste code
4. Click "Add to Chart"
5. Configure parameters based on timeframe
6. Set alerts: Hammer menu -> Alerts

## Testing Recommendations

- Backtest on 1h timeframe with 6-12 month data
- Optimize parameters for your trading pair
- Use "squeeze_momentum_bb_enhanced" for faster responses
- Combine with volatility filters for confirmation
- Forward-test 2 weeks minimum before live trading
