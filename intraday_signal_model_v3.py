import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import ta
from loguru import logger
import warnings
import os
warnings.filterwarnings('ignore')

logger.remove()
logger.add(lambda msg: print(msg, end=''), format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}")

def load_klines(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Load K-line data from HuggingFace dataset.
    
    symbol: e.g. 'BTCUSDT', 'ETHUSDT'
    timeframe: '15m', '1h', '1d'
    """
    from huggingface_hub import hf_hub_download
    
    repo_id = "zongowo111/v2-crypto-ohlcv-data"
    base = symbol.replace("USDT", "")
    filename = f"{base}_{timeframe}.parquet"
    path_in_repo = f"klines/{symbol}/{filename}"
    
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=path_in_repo,
        repo_type="dataset"
    )
    return pd.read_parquet(local_path)

def load_data(symbol: str = "BTCUSDT", timeframe: str = "15m"):
    logger.info(f"Loading {timeframe} data for {symbol}...")
    
    try:
        df = load_klines(symbol, timeframe)
    except Exception as e:
        logger.error(f"Failed to load from HuggingFace: {e}")
        logger.info("Attempting to load from local parquet file...")
        
        parquet_path = f"{symbol.replace('USDT', '')}_{timeframe}.parquet"
        if os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path)
        else:
            raise FileNotFoundError(f"Cannot find {parquet_path} locally or on HuggingFace")
    
    df['timestamp'] = pd.to_datetime(df['open_time'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.rename(columns={'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'v'})
    
    logger.info(f"Loaded {len(df)} candles")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df

def generate_features(df):
    logger.info("Generating features...")
    
    df['close'] = df['c']
    df['high'] = df['h']
    df['low'] = df['l']
    df['volume'] = df['v']
    df['open'] = df['o']
    
    bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
    df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
    
    df['rsi_7'] = ta.momentum.RSIIndicator(close=df['close'], window=7).rsi()
    df['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['rsi_21'] = ta.momentum.RSIIndicator(close=df['close'], window=21).rsi()
    
    macd = ta.trend.MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_histogram'] = macd.macd_diff()
    
    atr = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['atr'] = atr.average_true_range()
    df['atr_ratio'] = df['atr'] / df['close']
    
    df['roc_10'] = ta.momentum.ROCIndicator(close=df['close'], window=10).roc()
    df['roc_20'] = ta.momentum.ROCIndicator(close=df['close'], window=20).roc()
    
    stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    df['ma_5'] = ta.trend.SMAIndicator(close=df['close'], window=5).sma_indicator()
    df['ma_20'] = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
    
    df['ma_slope_5'] = df['ma_5'].diff()
    df['ma_slope_20'] = df['ma_20'].diff()
    
    adx = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['adx'] = adx.adx()
    
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_zscore'] = (df['volume'] - df['volume_ma_20']) / (df['volume'].rolling(window=20).std() + 1e-8)
    
    df['candle_body'] = abs(df['close'] - df['open'])
    df['candle_range'] = df['high'] - df['low']
    df['candle_body_ratio'] = df['candle_body'] / (df['candle_range'] + 1e-8)
    df['range_from_low'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    
    stoch_rsi = ta.momentum.StochasticRSIIndicator(close=df['close'], window=14, smooth1=14, smooth2=3)
    df['stoch_rsi_k'] = stoch_rsi.stochasticrsi()
    df['stoch_rsi_divergence'] = (df['stoch_rsi_k'] - df['stoch_rsi_k'].shift(1)).abs()
    
    df = df.fillna(method='bfill').fillna(method='ffill')
    logger.info("Features generated")
    return df

def generate_improved_labels(df, lookback_window=20, rr_ratio_threshold=1.5, max_loss_pct=0.5, min_profit_pct=1.0):
    """
    Generate labels based on risk-reward ratio analysis.
    
    Label 1 (quality signal): Risk/Reward >= threshold AND profit >= min_profit_pct
    Label 0 (poor signal): Otherwise
    """
    logger.info("Creating improved rolling window labels with risk-reward analysis...")
    
    labels = np.zeros(len(df), dtype=int)
    valid_count = 0
    profitable_count = 0
    
    for i in range(len(df) - lookback_window):
        entry_price = df.iloc[i]['close']
        future_slice = df.iloc[i:i+lookback_window]
        
        max_high = future_slice['high'].max()
        min_low = future_slice['low'].min()
        
        upside_pct = ((max_high - entry_price) / entry_price) * 100
        downside_pct = ((entry_price - min_low) / entry_price) * 100
        
        if downside_pct < 1e-8:
            downside_pct = 0.01
        rr_ratio = upside_pct / downside_pct
        
        if (upside_pct >= min_profit_pct and 
            downside_pct <= max_loss_pct and 
            rr_ratio >= rr_ratio_threshold):
            labels[i] = 1
            profitable_count += 1
        else:
            labels[i] = 0
        
        valid_count += 1
    
    df['label'] = labels
    
    win_rate = (profitable_count / valid_count * 100) if valid_count > 0 else 0
    logger.info(f"Generated {valid_count} valid labels")
    logger.info(f"  Quality signals: {profitable_count}")
    logger.info(f"  Poor signals: {valid_count - profitable_count}")
    logger.info(f"  Quality rate: {win_rate:.2f}%")
    
    return df

def prepare_features(df):
    feature_cols = [
        'bb_width', 'bb_position', 'rsi_7', 'rsi_14', 'rsi_21',
        'macd', 'macd_signal', 'macd_histogram',
        'atr_ratio', 'roc_10', 'roc_20',
        'stoch_k', 'stoch_d', 'ma_slope_5', 'ma_slope_20', 'adx',
        'volume_zscore', 'candle_body_ratio', 'range_from_low',
        'stoch_rsi_divergence'
    ]
    
    X = df[feature_cols].fillna(0)
    y = df['label']
    
    valid_mask = (X.notna().all(axis=1)) & (y.notna())
    return X[valid_mask], y[valid_mask]

def calculate_profit_factor(y_true, y_pred, threshold=0.5):
    """
    Profit Factor = Correct Predictions / Incorrect Predictions
    Higher is better. Target: > 1.5
    """
    predictions = (y_pred >= threshold).astype(int)
    
    correct = (predictions == y_true).sum()
    incorrect = (predictions != y_true).sum()
    
    if incorrect == 0:
        return 0
    return correct / incorrect

def optimize_threshold_by_profit_factor(y_true, y_pred_proba):
    """Find optimal threshold based on Profit Factor instead of F1-Score"""
    logger.info("Optimizing threshold by Profit Factor...")
    
    best_pf = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.40, 0.75, 0.02):
        pf = calculate_profit_factor(y_true, y_pred_proba, threshold)
        precision = precision_score(y_true, (y_pred_proba >= threshold).astype(int), zero_division=0)
        recall = recall_score(y_true, (y_pred_proba >= threshold).astype(int), zero_division=0)
        
        logger.info(f"  Threshold {threshold:.2f}: PF={pf:.4f}, P={precision:.4f}, R={recall:.4f}")
        
        if pf > best_pf:
            best_pf = pf
            best_threshold = threshold
    
    logger.info(f"\nSelected threshold: {best_threshold:.4f} (Profit Factor: {best_pf:.4f})")
    return best_threshold

def train_model(X_train, y_train, X_test, y_test):
    logger.info("======================================================================")
    logger.info("TRAINING OPTIMIZED RANDOM FOREST MODEL")
    logger.info("======================================================================")
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"  Positive: {(y_train == 1).sum()}")
    logger.info(f"  Negative: {(y_train == 0).sum()}")
    logger.info(f"  Base rate: {(y_train == 1).sum() / len(y_train) * 100:.2f}%")
    logger.info("")
    
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=3,
        min_samples_split=5,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    rf.fit(X_train, y_train)
    logger.info("Random Forest training complete")
    
    logger.info("Applying probability calibration (Isotonic Regression)...")
    calibrated_rf = CalibratedClassifierCV(
        estimator=rf,
        method='isotonic',
        cv=5
    )
    calibrated_rf.fit(X_train, y_train)
    logger.info("Calibration complete")
    
    probs_train = calibrated_rf.predict_proba(X_train)[:, 1]
    probs_test = calibrated_rf.predict_proba(X_test)[:, 1]
    
    optimal_threshold = optimize_threshold_by_profit_factor(y_test, probs_test)
    
    y_pred_test = (probs_test >= optimal_threshold).astype(int)
    precision_test = precision_score(y_test, y_pred_test, zero_division=0)
    recall_test = recall_score(y_test, y_pred_test, zero_division=0)
    f1_test = f1_score(y_test, y_pred_test, zero_division=0)
    
    logger.info("")
    logger.info("Test Set Performance (Calibrated):")
    logger.info(f"  Precision: {precision_test:.4f}")
    logger.info(f"  Recall: {recall_test:.4f}")
    logger.info(f"  F1-Score: {f1_test:.4f}")
    
    feature_cols = [
        'bb_width', 'bb_position', 'rsi_7', 'rsi_14', 'rsi_21',
        'macd', 'macd_signal', 'macd_histogram',
        'atr_ratio', 'roc_10', 'roc_20',
        'stoch_k', 'stoch_d', 'ma_slope_5', 'ma_slope_20', 'adx',
        'volume_zscore', 'candle_body_ratio', 'range_from_low',
        'stoch_rsi_divergence'
    ]
    
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    logger.info("")
    logger.info("Top 15 Important Features:")
    for idx, row in feature_importance_df.head(15).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    return calibrated_rf, optimal_threshold

def generate_signals(df, model, threshold):
    logger.info("======================================================================")
    logger.info("GENERATING SIGNALS ON FULL DATASET")
    logger.info("======================================================================")
    
    feature_cols = [
        'bb_width', 'bb_position', 'rsi_7', 'rsi_14', 'rsi_21',
        'macd', 'macd_signal', 'macd_histogram',
        'atr_ratio', 'roc_10', 'roc_20',
        'stoch_k', 'stoch_d', 'ma_slope_5', 'ma_slope_20', 'adx',
        'volume_zscore', 'candle_body_ratio', 'range_from_low',
        'stoch_rsi_divergence'
    ]
    
    X_full = df[feature_cols].fillna(0)
    probabilities = model.predict_proba(X_full)[:, 1]
    
    df['signal_prob'] = probabilities
    df['signal'] = (probabilities >= threshold).astype(int)
    
    total_signals = df['signal'].sum()
    high_conf_signals = (df['signal_prob'] >= 0.75).sum()
    
    logger.info(f"Total signals generated: {total_signals}")
    logger.info(f"  High confidence (prob >= 0.75): {high_conf_signals}")
    logger.info(f"  Standard confidence: {total_signals - high_conf_signals}")
    
    signal_df = df[df['signal'] == 1].copy()
    daily_signals = signal_df.groupby(signal_df['timestamp'].dt.date).size()
    
    logger.info("")
    logger.info("Daily signal distribution:")
    logger.info(f"  Days with signals: {len(daily_signals)}")
    logger.info(f"  Avg signals/day: {daily_signals.mean():.2f}")
    logger.info(f"  Max signals/day: {daily_signals.max()}")
    logger.info(f"  Min signals/day: {daily_signals.min()}")
    
    return df

def evaluate_full_dataset(df):
    logger.info("======================================================================")
    logger.info("FULL DATASET PERFORMANCE")
    logger.info("======================================================================")
    
    y_true = df['label']
    y_pred = df['signal']
    
    if y_pred.sum() == 0:
        logger.info("No signals generated, skipping evaluation")
        return
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    logger.info(f"Precision: {precision:.2%}")
    logger.info(f"Recall: {recall:.2%}")
    logger.info(f"F1-Score: {f1:.4f}")
    
    logger.info("")
    logger.info("Target: P >= 80%, R >= 80%")
    precision_gap = max(0, 0.80 - precision)
    recall_gap = max(0, 0.80 - recall)
    
    if precision_gap > 0:
        logger.info(f"Status: Precision {precision_gap*100:.1f}% short")
    if recall_gap > 0:
        logger.info(f"Status: Recall {recall_gap*100:.1f}% short")
    
    if precision >= 0.80 and recall >= 0.80:
        logger.info("Status: TARGET ACHIEVED")
    
    logger.info("")

def main(symbol: str = "BTCUSDT", timeframe: str = "15m"):
    logger.info("======================================================================")
    logger.info("Intraday Trading Model V3 Optimized: Risk-Reward + Calibration")
    logger.info("======================================================================")
    logger.info(f"Symbol: {symbol}, Timeframe: {timeframe}")
    logger.info("Strategy: Improved label definition with risk-reward ratio")
    logger.info("Enhancement: Probability calibration + class weight balancing")
    logger.info("")
    
    df = load_data(symbol, timeframe)
    logger.info("")
    
    logger.info("Engineering features...")
    df = generate_features(df)
    logger.info("")
    
    logger.info("Creating improved labels...")
    df = generate_improved_labels(
        df,
        lookback_window=20,
        rr_ratio_threshold=1.5,
        max_loss_pct=0.5,
        min_profit_pct=1.0
    )
    logger.info("")
    
    X, y = prepare_features(df)
    logger.info(f"Total valid data points: {len(X)}")
    logger.info("")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    model, threshold = train_model(X_train, y_train, X_test, y_test)
    logger.info("")
    
    df = generate_signals(df, model, threshold)
    logger.info("")
    
    evaluate_full_dataset(df)
    
    logger.info("======================================================================")
    logger.info("Model optimization complete")
    logger.info("======================================================================")

if __name__ == "__main__":
    main(symbol="BTCUSDT", timeframe="15m")
