# 模型訓練管理頁面 - 使用指南

## 功能概述

應用現在包含完整的模型訓練和管理系統，支持：

- 選擇任意交易對（BTC、ETH、LTC 等）
- 選擇任意時間框架（15m、1h、4h、1d）
- 選擇模型類型（XGBoost、LightGBM、Random Forest）
- 查看本地所有已訓練的模型
- 刪除不需要的模型
- 訓練結果實時顯示

## 快速開始

### 1. 啟動應用

```bash
python app.py
```

### 2. 訪問訓練頁面

打開瀏覽器訪問：**http://localhost:5000/training**

## 訓練頁面功能

### Tab 1: Train Model（訓練模型）

#### 輸入參數

1. **Trading Pair**（交易對）
   - 輸入交易對代碼（如 BTCUSDT、ETHUSDT、LTCUSDT）
   - 自動添加 USDT 後綴（如果沒有）
   - 默認值：BTCUSDT

2. **Timeframe**（時間框架）
   - 15 Minutes（15分鐘）
   - 1 Hour（1小時）
   - 4 Hours（4小時）
   - 1 Day（1天）
   - 默認值：1h

3. **Model Type**（模型類型）
   - XGBoost - 梯度提升模型，精度最高
   - LightGBM - 輕量級梯度提升，速度最快
   - Random Forest - 隨機森林，泛化能力最好
   - 默認值：XGBoost

#### 訓練流程

1. 填入參數
2. 點擊 "Start Training" 按鈕
3. 頁面顯示進度動畫和訓練狀態
4. 訓練完成後顯示詳細的訓練結果

#### 訓練結果

訓練完成後會顯示：

```
Symbol: BTCUSDT
Timeframe: 15m
Model Type: xgboost
Train Accuracy: 0.7234 (72.34%)
Test Accuracy: 0.7012 (70.12%)
Train Precision: 0.6850 (68.50%)
Test Precision: 0.6520 (65.20%)
```

**重要指標解釋：**

- **Train Accuracy**：訓練集準確率，值越高越好（通常會過擬合）
- **Test Accuracy**：測試集準確率，最重要指標，代表實際預測能力
- **Train Precision**：訓練集精確率，預測正確的反彈信號比例
- **Test Precision**：測試集精確率，實際預測精確性

**評估標準：**
- Test Accuracy > 65% 為可用
- Test Accuracy > 70% 為優秀
- Test Precision > 60% 為可用
- Test Precision > 65% 為優秀

### Tab 2: Local Models（本地模型）

#### 模型統計

頁面頂部顯示：
- **Total Models** - 已訓練模型總數
- 按時間框架統計
- 按交易對統計

#### 模型卡片

每個已訓練的模型顯示為一張卡片，包含：

| 信息 | 說明 |
|------|------|
| 交易對 | 如 BTCUSDT、ETHUSDT |
| 時間框架 | 15m、1h、4h、1d |
| Model Type | xgboost、lightgbm、random_forest |
| Size | 模型文件大小（KB） |
| Modified | 最後修改時間 |

#### 模型操作

每張卡片右下角有兩個按鈕：

1. **Analyze**（分析）
   - 跳轉到對應的分析頁面
   - 15m 模型跳到 15m Analysis 頁面
   - 其他時間框架跳到主儀表板

2. **Delete**（刪除）
   - 刪除該模型文件
   - 需要確認
   - 刪除後無法恢復

## 使用場景

### 場景 1：訓練第一個 15m 模型

```
1. Trading Pair: BTCUSDT
2. Timeframe: 15 Minutes
3. Model Type: XGBoost
4. 點擊 "Start Training"
5. 等待 30-90 秒
6. 查看訓練結果
7. 切換到 "Local Models" 標籤
8. 看到新訓練的模型卡片
9. 點擊 "Analyze" 查看實時預測
```

### 場景 2：為多個交易對訓練模型

```
訓練 BTC 1h 模型
交易對: BTCUSDT, 時間框架: 1h, 類型: XGBoost

訓練 ETH 4h 模型
交易對: ETHUSDT, 時間框架: 4h, 類型: XGBoost

訓練 BTC 15m 對比模型
交易對: BTCUSDT, 時間框架: 15m, 類型: LightGBM
```

### 場景 3：比較不同模型

在 "Local Models" 標籤中查看：
- BTCUSDT 的所有時間框架模型
- 同一交易對、同一時間框架、不同模型類型的性能差異
- 選擇性能最好的模型用於實盤

### 場景 4：清理過期模型

```
1. 切換到 "Local Models" 標籤
2. 查看所有模型列表
3. 點擊要刪除模型的 "Delete" 按鈕
4. 確認刪除
5. 模型文件被刪除，頁面自動刷新
```

## 文件位置

### 模型文件位置

所有訓練的模型保存在：

```
models/cache/
├── BTCUSDT_15m_xgboost.joblib
├── BTCUSDT_1h_xgboost.joblib
├── ETHUSDT_4h_lightgbm.joblib
└── ...
```

### 命名規則

```
{SYMBOL}_{TIMEFRAME}_{MODEL_TYPE}.joblib

示例：
- BTCUSDT_15m_xgboost.joblib
- ETHUSDT_1h_lightgbm.joblib
- LTCUSDT_4h_random_forest.joblib
```

## 常見問題

### Q: 訓練需要多久？
A: 根據數據量和模型複雜度，通常 30-120 秒。XGBoost 通常需要 60-90 秒，LightGBM 30-60 秒，Random Forest 60-120 秒。

### Q: 訓練失敗怎麼辦？
A: 檢查：
1. 交易對是否存在（Binance 有數據）
2. 網絡連接是否正常
3. 本地 `models/cache` 目錄是否存在
4. 磁盤空間是否充足

### Q: 如何選擇模型類型？
A: 
- **精度優先**：使用 XGBoost
- **速度優先**：使用 LightGBM
- **穩定性優先**：使用 Random Forest

### Q: Test Accuracy 很低怎麼辦？
A: 
1. 檢查市場情況（熊市/牛市/盤整）
2. 嘗試不同的時間框架
3. 嘗試不同的交易對
4. 等待更多數據後重新訓練
5. 調整 BB 參數（需修改代碼）

### Q: 可以同時訓練多個模型嗎？
A: 不建議。系統會逐個訓練。請等待一個訓練完成再開始下一個。

### Q: 如何更新已訓練的模型？
A: 重新訓練會自動覆蓋舊模型。無需手動刪除。

### Q: 模型文件能否移動或備份？
A: 可以。模型文件是標準的 joblib 格式，可自由複製或備份。建議定期備份 `models/cache` 目錄。

## 最佳實踐

### 1. 定期重新訓練

```
每週：訓練一次 1h 和 4h 模型
每天：訓練一次 15m 模型
```

### 2. 多模型對比

```
為同一交易對和時間框架訓練多個模型類型
選擇 Test Accuracy 最高的模型用於交易
```

### 3. 交易對多樣化

```
同時訓練 BTC、ETH、LTC 等多個交易對
在不同市場條件下評估模型性能
```

### 4. 定期清理

```
刪除舊的、性能差的模型
Test Accuracy < 60% 的模型應該刪除
保持 cache 目錄清潔
```

## 下一步

1. **訓練你的第一個模型**
   - 打開 http://localhost:5000/training
   - 選擇交易對和時間框架
   - 點擊 "Start Training"

2. **查看本地模型**
   - 切換到 "Local Models" 標籤
   - 查看所有已訓練的模型

3. **分析模型結果**
   - 點擊模型卡片上的 "Analyze" 按鈕
   - 查看實時預測信號

4. **管理多個模型**
   - 為不同交易對和時間框架訓練模型
   - 比較不同模型的性能
   - 刪除不需要的模型

## 技術細節

### API 端點

**訓練模型：**
```bash
POST /api/train-model
Body: {
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "model_type": "xgboost"
}
```

**列出本地模型：**
```bash
GET /api/models/list
```

**刪除模型：**
```bash
DELETE /api/models/delete/{symbol}/{timeframe}
```

### 前端組件

- 訓練表單：支持實時驗證和表單提交
- 進度指示：訓練進行中顯示加載動畫
- 結果展示：訓練完成後顯示詳細指標
- 模型列表：網格布局展示所有模型
- 模型卡片：顯示模型信息和操作按鈕

## 導航結構

```
Main Dashboard (/)
├── Dashboard 標籤 (默認)
├── Training 標籤 (/training) [新增]
│   ├── Train Model 標籤
│   │   ├── 輸入參數表單
│   │   ├── 訓練進度
│   │   └── 訓練結果
│   └── Local Models 標籤
│       ├── 模型統計
│       └── 模型卡片列表
├── 15m Analysis 標籤 (/15m-analysis)
└── Chart 標籤 (/chart)
```

## 常見工作流

### 工作流 1：日常訓練

```
1. 打開 http://localhost:5000/training
2. Train Model 標籤中訓練今日模型
3. 查看訓練結果
4. 切換到 Local Models 查看模型列表
5. 點擊 Analyze 查看預測
```

### 工作流 2：多模型對比

```
1. 訓練 XGBoost 模型
2. 訓練 LightGBM 模型
3. 訓練 Random Forest 模型
4. 查看 Local Models 標籤對比結果
5. 選擇最好的模型進行交易
```

### 工作流 3：模型管理

```
1. 定期查看 Local Models 標籤
2. 識別性能差的模型（Test Accuracy < 60%）
3. 刪除舊的、冗余的模型
4. 保持 cache 目錄整潔
5. 定期備份 models/cache 目錄
```
