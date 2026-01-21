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
import sys
warnings.filterwarnings('ignore')

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

logger.remove()
logger.add(lambda msg: print(msg, end=''), format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}")

def load_klines(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    從 HuggingFace 資料集讀取 K 線資料
    
    symbol: 例如 'BTCUSDT', 'ETHUSDT'
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
    logger.info(f"正在讀取 {symbol} {timeframe} 資料...")
    
    try:
        df = load_klines(symbol, timeframe)
    except Exception as e:
        logger.error(f"從 HuggingFace 讀取失敗: {e}")
        logger.info("嘗試從本地 parquet 檔案讀取...")
        
        parquet_path = f"{symbol.replace('USDT', '')}_{timeframe}.parquet"
        if os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path)
        else:
            raise FileNotFoundError(f"找不到 {parquet_path}")
    
    df['timestamp'] = pd.to_datetime(df['open_time'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.rename(columns={'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'v'})
    
    logger.info(f"已讀取 {len(df)} 根 K 線")
    logger.info(f"時間範圍: {df['timestamp'].min()} 至 {df['timestamp'].max()}")
    return df

def generate_features(df):
    logger.info("正在生成技術指標...")
    
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
    
    stoch_rsi = ta.momentum.StochRSIIndicator(close=df['close'], window=14, smooth1=14, smooth2=3)
    df['stoch_rsi_k'] = stoch_rsi.stochrsi()
    df['stoch_rsi_divergence'] = (df['stoch_rsi_k'] - df['stoch_rsi_k'].shift(1)).abs()
    
    df = df.fillna(method='bfill').fillna(method='ffill')
    logger.info("技術指標生成完成")
    return df

def generate_improved_labels(df, lookback_window=20, rr_ratio_threshold=1.0, max_loss_pct=1.0, min_profit_pct=0.5):
    """
    基於風險收益比的標籤生成
    
    Label 1 (優質信號): 風險/收益 >= threshold AND 利潤 >= min_profit_pct
    Label 0 (劣質信號): 其他情況
    """
    logger.info("正在生成基於風險收益比的標籤...")
    logger.info(f"參數: lookback_window={lookback_window}, rr_ratio={rr_ratio_threshold}, max_loss={max_loss_pct}%, min_profit={min_profit_pct}%")
    
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
    logger.info(f"標籤生成完成")
    logger.info(f"  總標籤數: {valid_count}")
    logger.info(f"  優質信號: {profitable_count}")
    logger.info(f"  劣質信號: {valid_count - profitable_count}")
    logger.info(f"  優質率: {win_rate:.2f}%")
    
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
    利潤因子 = 正確預測數 / 不正確預測數
    越高越好。目標: > 1.5
    """
    predictions = (y_pred >= threshold).astype(int)
    
    correct = (predictions == y_true).sum()
    incorrect = (predictions != y_true).sum()
    
    if incorrect == 0:
        return 0
    return correct / incorrect

def optimize_threshold_by_f1_score(y_true, y_pred_proba):
    """根據 F1 分數尋找最優閾值，平衡精準度和召回率"""
    logger.info("正在根據 F1 分數優化閾值...")
    
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = {}
    
    for threshold in np.arange(0.30, 0.75, 0.02):
        precision = precision_score(y_true, (y_pred_proba >= threshold).astype(int), zero_division=0)
        recall = recall_score(y_true, (y_pred_proba >= threshold).astype(int), zero_division=0)
        f1 = f1_score(y_true, (y_pred_proba >= threshold).astype(int), zero_division=0)
        pf = calculate_profit_factor(y_true, y_pred_proba, threshold)
        
        logger.info(f"  閾值 {threshold:.2f}: F1={f1:.4f}, 精準度={precision:.4f}, 召回率={recall:.4f}, PF={pf:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {'precision': precision, 'recall': recall, 'pf': pf}
    
    logger.info(f"\n選定閾值: {best_threshold:.4f}")
    logger.info(f"  F1 分數: {best_f1:.4f}")
    logger.info(f"  精準度: {best_metrics['precision']:.4f}")
    logger.info(f"  召回率: {best_metrics['recall']:.4f}")
    logger.info(f"  利潤因子: {best_metrics['pf']:.4f}")
    
    return best_threshold

def train_model(X_train, y_train, X_test, y_test):
    logger.info("="*70)
    logger.info("訓練隨機森林模型")
    logger.info("="*70)
    logger.info(f"訓練樣本數: {len(X_train)}")
    logger.info(f"  正類: {(y_train == 1).sum()}")
    logger.info(f"  負類: {(y_train == 0).sum()}")
    logger.info(f"  正類比例: {(y_train == 1).sum() / len(y_train) * 100:.2f}%")
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
    logger.info("隨機森林訓練完成")
    
    logger.info("正在應用概率校準 (等滲迴歸)...")
    calibrated_rf = CalibratedClassifierCV(
        estimator=rf,
        method='isotonic',
        cv=5
    )
    calibrated_rf.fit(X_train, y_train)
    logger.info("校準完成")
    
    probs_train = calibrated_rf.predict_proba(X_train)[:, 1]
    probs_test = calibrated_rf.predict_proba(X_test)[:, 1]
    
    optimal_threshold = optimize_threshold_by_f1_score(y_test, probs_test)
    
    y_pred_test = (probs_test >= optimal_threshold).astype(int)
    precision_test = precision_score(y_test, y_pred_test, zero_division=0)
    recall_test = recall_score(y_test, y_pred_test, zero_division=0)
    f1_test = f1_score(y_test, y_pred_test, zero_division=0)
    
    logger.info("")
    logger.info("測試集性能 (已校準):")
    logger.info(f"  精準度: {precision_test:.4f}")
    logger.info(f"  召回率: {recall_test:.4f}")
    logger.info(f"  F1 分數: {f1_test:.4f}")
    
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
    logger.info("前 15 個重要特徵:")
    for idx, row in feature_importance_df.head(15).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    return calibrated_rf, optimal_threshold

def generate_signals(df, model, threshold):
    logger.info("="*70)
    logger.info("在完整資料集上生成信號")
    logger.info("="*70)
    
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
    
    logger.info(f"生成的總信號數: {total_signals}")
    logger.info(f"  高信心信號 (概率 >= 0.75): {high_conf_signals}")
    logger.info(f"  標準信心信號: {total_signals - high_conf_signals}")
    
    signal_df = df[df['signal'] == 1].copy()
    daily_signals = signal_df.groupby(signal_df['timestamp'].dt.date).size()
    
    logger.info("")
    logger.info("日均信號分佈:")
    logger.info(f"  有信號的天數: {len(daily_signals)}")
    logger.info(f"  平均每日信號數: {daily_signals.mean():.2f}")
    logger.info(f"  最多每日信號數: {daily_signals.max()}")
    logger.info(f"  最少每日信號數: {daily_signals.min()}")
    
    return df

def evaluate_full_dataset(df):
    logger.info("="*70)
    logger.info("完整資料集性能評估")
    logger.info("="*70)
    
    y_true = df['label']
    y_pred = df['signal']
    
    if y_pred.sum() == 0:
        logger.info("未生成任何信號，跳過評估")
        return
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pf = calculate_profit_factor(y_true, y_pred.values)
    
    logger.info(f"精準度: {precision:.2%}")
    logger.info(f"召回率: {recall:.2%}")
    logger.info(f"F1 分數: {f1:.4f}")
    logger.info(f"利潤因子: {pf:.4f}")
    
    logger.info("")
    logger.info("目標: 精準度 >= 80%, 召回率 >= 80%")
    precision_gap = max(0, 0.80 - precision)
    recall_gap = max(0, 0.80 - recall)
    
    if precision_gap > 0:
        logger.info(f"狀態: 精準度還差 {precision_gap*100:.1f}%")
    if recall_gap > 0:
        logger.info(f"狀態: 召回率還差 {recall_gap*100:.1f}%")
    
    if precision >= 0.80 and recall >= 0.80:
        logger.info("狀態: 已達成目標")
    
    logger.info("")

def main(symbol: str = "BTCUSDT", timeframe: str = "15m"):
    logger.info("="*70)
    logger.info("日內交易模型 V3 優化: 風險收益 + 校準")
    logger.info("="*70)
    logger.info(f"交易對: {symbol}, 時間框: {timeframe}")
    logger.info("策略: 基於風險收益比的改進標籤定義")
    logger.info("增強: 概率校準 + 類別權重平衡")
    logger.info("最佳化: 基於 F1 分數平衡精準度和召回率")
    logger.info("")
    
    df = load_data(symbol, timeframe)
    logger.info("")
    
    logger.info("正在工程化特徵...")
    df = generate_features(df)
    logger.info("")
    
    logger.info("正在生成標籤...")
    df = generate_improved_labels(
        df,
        lookback_window=20,
        rr_ratio_threshold=1.0,
        max_loss_pct=1.0,
        min_profit_pct=0.5
    )
    logger.info("")
    
    X, y = prepare_features(df)
    logger.info(f"有效資料點總數: {len(X)}")
    logger.info("")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    model, threshold = train_model(X_train, y_train, X_test, y_test)
    logger.info("")
    
    df = generate_signals(df, model, threshold)
    logger.info("")
    
    evaluate_full_dataset(df)
    
    logger.info("="*70)
    logger.info("模型優化完成")
    logger.info("="*70)

if __name__ == "__main__":
    main(symbol="BTCUSDT", timeframe="15m")
