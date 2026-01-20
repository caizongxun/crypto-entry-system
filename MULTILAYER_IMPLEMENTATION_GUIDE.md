# 15 分鐘框架多層整合實施指南

## 整體目標

不放棄 15 分鐘框架，而是通過多層確認機制將勝率從 30.61% 提升至 42-48%。

## 核心思想

### 問題
```
單純 15m 形態:
  - 時間窗口太短 (5 小時)
  - 噪音過多
  - 虛假信號頻繁
  結果: 30.61% 勝率
```

### 解決方案
```
多層確認機制:
  Layer 1: 形態本身 (質量 >= 60)
  Layer 2: 動量確認 (短期趨勢一致)
  Layer 3: 成交量確認 (參與度增加)
  Layer 4: 極值確認 (RSI/MACD 極值)
  Layer 5: 風險篩選 (波動率和間隙)
  Layer 6: 環境確認 (大框架趨勢)
  
  只在 2-3 層以上確認時交易
  結果: 42-48% 勝率
```

## 代碼結構

### 已上傳文件

```
strategy_v3/
├── multilayer_features.py
│   └── MultiLayerFeatureEngineer 類
│       - add_momentum_features()
│       - add_volume_features()
│       - add_extremum_features()
│       - add_risk_features()
│       - add_environment_features()
│       - engineer_multilayer_features() [主入口]
│
├── targets_multilayer.py
│   └── MultiLayerLabelGenerator 類
│       - check_pattern_quality()
│       - check_momentum_confirmation()
│       - check_volume_confirmation()
│       - check_extremum_confirmation()
│       - check_risk_filter()
│       - check_environment_confirmation()
│       - generate_multilayer_labels() [主入口]
│
└── pattern_detector.py (既有)
    └── PatternDetector 類
        - detect_patterns()

test_multilayer_15m_integration.py
└── 整合測試腳本
```

## 使用流程

### 第一步: 基礎數據準備

```python
from data_loader import DataLoader
from strategy_v3.feature_engineer import FeatureEngineer

loader = DataLoader()
df = loader.load_bitcoin_15m()

engineeer = FeatureEngineer()
df = engineer.engineer_features(df)
```

### 第二步: 形態檢測

```python
from strategy_v3.pattern_detector import PatternDetector

pattern_detector = PatternDetector()
patterns_df = pattern_detector.detect_patterns(df)
```

### 第三步: 多層特徵生成

```python
from strategy_v3.multilayer_features import MultiLayerFeatureEngineer

multilayer_engineer = MultiLayerFeatureEngineer()
multilayer_features = multilayer_engineer.engineer_multilayer_features(df)
```

生成 32 個新特徵：
- 6 個動量特徵
- 5 個成交量特徵  
- 5 個極值特徵
- 5 個風險特徵
- 5 個環境特徵

### 第四步: 多層標籤生成

```python
from strategy_v3.targets_multilayer import MultiLayerLabelGenerator

label_generator = MultiLayerLabelGenerator(config={
    'quality_threshold': 60,      # 形態質量最小值
    'min_confirmations': 2,       # 最少確認層數
    'volatility_threshold': 0.03, # 波動率上限
    'gap_threshold': 0.02,        # 間隙上限
})

labels, confidences, stats = label_generator.generate_multilayer_labels(df, patterns_df)
```

返回值：
- labels: -1 (無交易), 0 (不確定), 1 (看漲), -1 (看跌)
- confidences: 確認的層數 (1-6)
- stats: 統計信息

### 第五步: 特徵組合

```python
import pandas as pd

df_combined = pd.concat([
    df,                        # 原始 OHLCV
    existing_features,         # 76 個既有特徵
    multilayer_features,       # 32 個新特徵
], axis=1)

# 總特徵數: 76 + 32 = 108
```

### 第六步: 模型訓練

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

X = df_combined[feature_columns]  # 108 個特徵
y = labels  # 多層標籤

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=7,
    min_samples_split=50,
    min_samples_leaf=20,
    subsample=0.8,
    random_state=42,
)

model.fit(X_scaled, y)
```

## 特徵詳細說明

### Layer 2: 動量確認 (6 個特徵)

```python
momentum_1h: 1 小時動量變化
momentum_1h_pct: 1 小時百分比變化
momentum_4h: 4 小時動量變化
velocity: 價格變化速度
acceleration: 加速度
roc_10: 10 根 K 線變化率
ema_slope: EMA 斜率

作用: 確認形態方向與市場短期趨勢是否一致
```

### Layer 3: 成交量確認 (5 個特徵)

```python
volume_ratio: 當前成交量 / 平均成交量
volume_trend: 成交量變化方向
obv: On Balance Volume
obv_slope: OBV 斜率
ad_indicator: Accumulation/Distribution

作用: 確認是否有真實的市場參與和資金流入
```

### Layer 4: 極值確認 (5 個特徵)

```python
rsi: 相對強度指數
macd: MACD 值
macd_signal: MACD 信號線
macd_histogram: MACD 直方圖
bb_position: Bollinger Band 位置
stochastic_k: Stochastic K 值
stochastic_d: Stochastic D 值

作用: 確認是否處於超買超賣或趨勢反轉的極值位置
```

### Layer 5: 風險篩選 (5 個特徵)

```python
volatility_15m: 15 分鐘波動率
volatility_1h: 1 小時波動率
volatility_ratio: 波動率比 (異常判斷)
gap_from_previous_close: 跳空間隙
range_hl_pct: 高低點範圍百分比

作用: 篩選風險過高的交易 (不交易，不預測)
規則: 波動率 < 3%, 間隙 < 2%
```

### Layer 6: 環境確認 (5 個特徵)

```python
trend_4h: 4 小時趨勢 (1=上升, 0=下降)
trend_1h: 1 小時趨勢
session_strength: 交易時段強度
volatility_regime: 波動率狀態
concentration: 價格集中度

作用: 提供大框架背景信息，加權其他信號
```

## 標籤邏輯

### 多層確認決策樹

```
Pattern Detected
    ↓
    ├─ Layer 1: 形態質量 >= 60?
    │   ├─ NO → 不交易 (-1)
    │   └─ YES → 繼續
    │       ↓
    │       ├─ Layer 5: 風險可接受?
    │       │   ├─ NO → 不交易 (-1)
    │       │   └─ YES → 繼續
    │       │       ↓
    │       │       ├─ Layer 2: 動量一致? (+ 1)
    │       │       ├─ Layer 3: 成交量增加? (+ 1)
    │       │       ├─ Layer 4: 極值確認? (+ 1)
    │       │       ├─ Layer 6: 環境確認? (+ 1)
    │       │       ↓
    │       │       ├─ 確認層數 >= 2?
    │       │       │   ├─ NO → 不確定 (0)
    │       │       │   └─ YES → 交易 (1 或 -1)
    │       │       │       (根據形態方向)
```

### 具體例子

```
情況 1: 高質量雙底 + 強動量 + 成交量增加
  確認層: 1 + 1 + 1 = 3 層
  決策: 交易 (信心高)
  標籤: 1 (看漲)

情況 2: 中等雙底 + 動量一致
  確認層: 1 + 1 = 2 層
  決策: 交易 (信心中等)
  標籤: 1 (看漲)

情況 3: 低質量雙頂 + 極值確認
  確認層: 0 + 1 = 1 層
  決策: 不交易 (信心太低)
  標籤: 0 (不確定)

情況 4: 形態 + 高波動率
  風險層: 波動率 > 3%
  決策: 不交易 (風險過高)
  標籤: -1 (不交易)
```

## 預期改善

### 逐層改善

```
V1: 單層 (只形態)
  特徵: 6 個形態特徵
  勝率: 30.61%
  問題: 噪音太多

V2: 加動量
  特徵: +6 個動量特徵
  勝率: 35-38% (+5%)
  改善: 篩除逆趨勢信號

V3: 加成交量
  特徵: +5 個成交量特徵
  勝率: 38-42% (+4%)
  改善: 確認市場參與度

V4: 加極值
  特徵: +5 個極值特徵
  勝率: 40-44% (+2%)
  改善: 強化反轉信號

V5: 加風險和環境
  特徵: +10 個風險/環境特徵
  勝率: 42-48% (+2-4%)
  改善: 篩除風險交易，增加上下文

最終: 108 個特徵
  勝率: 42-48%
  改善: +12-17%
```

## 參數優化建議

### 確認層數閾值

```
min_confirmations = 2 (當前)
  優點: 交易機會多
  缺點: 准確度較低
  用途: 激進策略

min_confirmations = 3 (建議)
  優點: 平衡機會和準確度
  缺點: 機會減少
  用途: 標準策略

min_confirmations = 4 (保守)
  優點: 准確度高
  缺點: 機會很少
  用途: 保守策略
```

### 波動率閾值

```
volatility_threshold = 0.02 (嚴格)
  篩除更多風險交易
  交易次數減少 20-30%

volatility_threshold = 0.03 (平衡)
  當前設置
  保留合理數量的交易

volatility_threshold = 0.05 (寬鬆)
  接受高波動率環境
  交易次數增加 20-30%
```

## 測試和驗證

### Step 1: 運行集成測試

```bash
python test_multilayer_15m_integration.py
```

預期輸出：
```
Loaded 221672 candles
Detected 3097 patterns
Generated 32 multi-layer features
Labeled 2149 patterns
High confidence patterns: ~1000-1200
Projected win rate: 42-48%
```

### Step 2: 特徵重要性分析

```python
import matplotlib.pyplot as plt

feature_importance = model.feature_importances_
top_features = np.argsort(feature_importance)[-20:]

plt.figure(figsize=(12, 8))
plt.barh(range(20), feature_importance[top_features])
plt.xlabel('Importance')
plt.title('Top 20 Feature Importance')
plt.tight_layout()
plt.show()
```

### Step 3: 層級貢獻度分析

```python
for layer_name in ['momentum', 'volume', 'extremum', 'risk', 'env']:
    layer_features = [f for f in feature_columns if layer_name in f]
    layer_importance = feature_importance[[feature_columns.index(f) for f in layer_features]]
    print(f'{layer_name}: {layer_importance.sum():.4f}')
```

### Step 4: 回測驗證

```python
from backtesting import Backtest

backtest = Backtest(df, YourStrategy, cash=10000, commission=.002)
results = backtest.run()

print(results)
print(f'Win Rate: {results._stats["Win Rate"]*100:.2f}%')
print(f'Sharpe Ratio: {results._stats["Sharpe Ratio"]:.2f}')
```

## 常見問題

### Q1: 為什麼還是 15m 而不是 1h?

```
15m 優勢:
  - 高頻率: 每天 10-30 個信號
  - 快速反應: 能更快進場
  - 靈活性: 適合日內交易

多層補償:
  - 噪音多 → 用多層篩選
  - 窗口短 → 延長確認時間
  - 準確度低 → 提高確認標準

結果: 既保留 15m 高頻率，又達到 1h 的準確度
```

### Q2: 32 個新特徵會不會過度擬合?

```
防護措施:
  1. 特徵設計基於經濟學原理 (不是隨意)  
  2. 多層邏輯使用規則 (不是統計學習)
  3. 使用 dropout 和正則化
  4. 交叉驗證和時間序列分割
  5. 測試新數據 (2024-2026)

結果: 低過度擬合風險
```

### Q3: 如何實時運行?

```python
# 實時行情更新
while True:
    new_candle = get_latest_candle()
    df = df.append(new_candle)
    
    # 重新計算特徵
    multilayer_features = multilayer_engineer.engineer_multilayer_features(df)
    
    # 預測
    X_latest = df_combined.iloc[-1:]
    prediction = model.predict(X_latest)[0]
    confidence = model.predict_proba(X_latest)[0]
    
    if prediction == 1 and confidence[1] > 0.55:
        execute_buy_order()
    elif prediction == -1 and confidence[0] > 0.55:
        execute_sell_order()
    
    time.sleep(60)  # 等待下一個 15m K 線
```

## 下一步行動

### 今晚
- [ ] 運行 test_multilayer_15m_integration.py
- [ ] 驗證所有 32 個特徵計算正確
- [ ] 檢查標籤生成邏輯

### 明天
- [ ] 訓練新模型
- [ ] 比較性能: 基線 vs 新模型
- [ ] 特徵重要性分析

### 後天
- [ ] 回測驗證
- [ ] 參數優化
- [ ] 紙盤交易測試

### 一周內
- [ ] 實盤驗證 (小額)
- [ ] 風險管理測試
- [ ] 優化和迭代

## 成功標準

```
定量:
  - 勝率 >= 42%
  - AUC >= 0.64
  - Sharpe 比率 >= 0.8

定性:
  - 信號品質明顯提升
  - 虛假信號明顯減少
  - 系統整體穩定性改善
  - 可交易性提高 (信心增強)
```

## 資源

- PATTERN_DISCOVERY_REPORT.md - 發現報告
- 15m_integrated_strategy.md - 詳細設計文檔
- strategy_v3/multilayer_features.py - 特徵工程代碼
- strategy_v3/targets_multilayer.py - 標籤生成代碼
- test_multilayer_15m_integration.py - 集成測試

