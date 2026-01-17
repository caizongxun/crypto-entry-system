# 15m Bollinger Band 模型訓練指南

## 概述

本指南說明如何訓練 15 分鐘時間框架的 Bollinger Band 反彈預測模型，以及如何在網頁中查看預測結果。

## 快速開始

### 方式 1：使用命令行訓練

```bash
cd C:\Users\zong\PycharmProjects\crypto-entry-system

python scripts/train_15m_model.py BTCUSDT
```

輸出示例：
```
============================================================
Training 15m BB Model for BTCUSDT
============================================================

Step 1: Loading data...
Loading data for BTCUSDT (15m)...
Data loaded: 2500 candles

Step 2: Engineering features...
Engineering features...

Step 3: Training model...
Preparing training data for BB bounce prediction...
Training data prepared: 450 BB touch/break events, 13 features
Timeframe: 15m
Effective bounces: 280, Ineffective: 170
Bounce rate: 62.22%
Training xgboost model for 15m...

Model training completed for 15m:
  Train accuracy: 0.7234, Precision: 0.6850
  Test accuracy: 0.7012, Precision: 0.6520
Model saved to models/cache/BTCUSDT_15m_xgboost.joblib

============================================================
Training Results:
============================================================
Symbol: BTCUSDT
Timeframe: 15m
Model Type: xgboost
Train Accuracy: 0.7234
Test Accuracy: 0.7012
Train Precision: 0.6850
Test Precision: 0.6520
============================================================

Training completed successfully!
Model saved to: models/cache/BTCUSDT_15m_xgboost.joblib
```

### 方式 2：通過網頁界面訓練

1. 啟動應用
   ```bash
   python app.py
   ```

2. 訪問網頁 http://localhost:5000/15m-analysis

3. 輸入交易對（如 BTCUSDT）

4. 點擊 "Train Model" 按鈕

5. 等待訓練完成（通常 30-60 秒）

## 模型工作原理

### 15m 特定配置

```python
{
    'bb_period': 20,          # BB 周期為 20
    'bb_std': 2.0,            # 標準差倍數為 2
    'lookforward': 5,         # 向前看 5 根 K 線
    'bounce_threshold': 0.5%  # 反彈幅度閾值 0.5%
}
```

### 訓練流程

1. **數據加載**：從本地緩存或 HuggingFace 加載 BTCUSDT 15m OHLCV 數據
2. **特徵工程**：計算 RSI、MACD、BB、ATR 等 13 個技術指標
3. **目標計算**：計算過去 5 根 K 線內的有效反彈信號
4. **數據分割**：80% 訓練集，20% 測試集
5. **模型訓練**：使用 XGBoost 分類器訓練
6. **模型保存**：保存到 `models/cache/BTCUSDT_15m_xgboost.joblib`

### 評估指標

- **準確率 (Accuracy)**：模型整體預測正確率
- **精確率 (Precision)**：預測正確的反彈信號比例
- **反彈率 (Bounce Rate)**：有效反彈占所有 BB 觸及信號的比例

## 使用模型進行預測

### API 端點

```bash
GET /api/ml-prediction-15m?symbol=BTCUSDT&lookback=50
```

**響應示例：**

```json
{
  "status": "success",
  "symbol": "BTCUSDT",
  "timeframe": "15m",
  "predictions": [
    {
      "timestamp": "2026-01-17 08:45:00",
      "signal_type": "lower_touch",
      "price": 42500.50,
      "bounce_probability": 75.3,
      "bb_position": 0.15,
      "bb_width": 250.5,
      "bb_upper": 42750.00,
      "bb_lower": 42500.00,
      "bb_basis": 42625.00
    }
  ],
  "total_events": 12,
  "model_trained": true
}
```

**參數說明：**

- `timestamp`：信號發生時間
- `signal_type`：信號類型
  - `lower_touch`：下軌觸及
  - `lower_break`：下軌突破
  - `upper_touch`：上軌觸及
  - `upper_break`：上軌突破
- `bounce_probability`：反彈概率（0-100）
- `bb_position`：價格在 BB 通道內的位置（0-1）
  - 0：在下軌
  - 0.5：在中線
  - 1：在上軌
- `bb_width`：BB 通道寬度（價格差）

## 網頁儀表板

訪問 http://localhost:5000/15m-analysis 會看到：

### 模型訓練卡片
- 模型狀態（已訓練/未訓練）
- 上次訓練時間
- 模型類型（XGBoost）

### 統計信息卡片
- 總 BB 信號數
- 高概率信號數（>70%）
- 最後信號時間

### BB 指標卡片
- BB 位置百分比
- BB 通道寬度
- 當前價格

### 預測表格
顯示最近 20 個 BB 反彈信號的詳細信息

## 常見問題

### Q: 訓練需要多久？
A: 取決於數據量，通常 30-90 秒。

### Q: 準確率多少才算好？
A: 測試集精確率 > 65% 已經不錯。實際取決於市場情況。

### Q: 可以用不同交易對訓練嗎？
A: 可以。運行 `python scripts/train_15m_model.py ETHUSDT` 訓練以太坊。

### Q: 訓練數據從哪來？
A: 從本地緩存或 HuggingFace Hub（zongowo111/v2-crypto-ohlcv-data）。

### Q: 多久重新訓練一次？
A: 建議每周或月初重新訓練以適應市場變化。

## 預測信號解釋

### 下軌觸及 + 高概率
買進信號：價格觸及下軌且模型預測會反彈
```
入場：市價買進或輕微超跌反彈時買進
止損：突破下軌 0.5% 以下
止盈：反彈到 BB 中線或上軌
```

### 上軌觸及 + 高概率
賣出信號：價格觸及上軌且模型預測會回落
```
入場：市價賣出或輕微超漲時賣出
止損：突破上軌 0.5% 以上
止盈：回落到 BB 中線或下軌
```

### 低概率信號
忽略：模型信心不足，跳過該信號

## 性能優化建議

1. **定期檢查模型性能**
   - 每周回測新數據
   - 如精確率下降 > 10%，重新訓練

2. **多模型組合**
   - 同時訓練 1h 和 4h 模型
   - 信號確認時更加可靠

3. **市場適應**
   - 牛市增加上軌信號權重
   - 熊市增加下軌信號權重
   - 盤整期降低信號頻率

## 故障排除

### 錯誤："No model available"
```
原因：15m 模型未訓練
解決：運行 python scripts/train_15m_model.py BTCUSDT
```

### 錯誤："Not enough training data"
```
原因：數據不足 100 個 BB 信號
解決：等待更多數據或使用其他交易對
```

### 模型精確率低 (< 50%)
```
原因：當前市場不適合該策略
解決：
1. 檢查市場趨勢
2. 調整 BB 參數
3. 增加訓練數據量
```

## 文件位置

- 訓練腳本：`scripts/train_15m_model.py`
- 模型文件：`models/cache/BTCUSDT_15m_xgboost.joblib`
- 前端頁面：`web/templates/15m_analysis.html`
- 後端路由：`app.py` 中的 `/api/ml-prediction-15m` 和 `/api/train-model-15m`

## 下一步

1. 訓練 1h 和 4h 模型進行多時間框架確認
2. 添加風險管理規則（止損、止盈）
3. 整合實時交易執行
4. 監控績效并定期調整參數
