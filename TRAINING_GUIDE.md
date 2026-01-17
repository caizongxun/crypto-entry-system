# 模型訓練指南

## 快速開始

基礎訓練命令:

```bash
python main.py --train
```

完整訓練命令(指定所有參數):

```bash
python main.py --train --symbol BTCUSDT --timeframe 1h --model-type xgboost --opt balanced
```

## 命令參數詳解

### 1. --train

進入訓練模式(必須指定才會執行訓練)

```bash
python main.py --train
```

### 2. --symbol

交易對符號,要訓練哪個加密貨幣

可選值:任何交易對(系統會自動添加 USDT 後綴)

預設值:BTCUSDT

常見交易對:
- BTCUSDT - 比特幣
- ETHUSDT - 以太坊
- LTCUSDT - 萊特幣
- BNBUSDT - 幣安幣
- ADAUSDT - 卡爾達諾

使用方式:

```bash
python main.py --train --symbol ETHUSDT
python main.py --train --symbol LTCUSDT
python main.py --train --symbol BTC
```

### 3. --timeframe

時間框架,決定使用哪個級別的K線數據

可選值:
- 15m - 15分鐘
- 1h - 1小時
- 4h - 4小時
- 1d - 1天

預設值:1h

使用方式:

```bash
python main.py --train --timeframe 15m
python main.py --train --timeframe 4h
python main.py --train --timeframe 1d
```

### 4. --model-type

機器學習模型算法類型

可選值:
- xgboost - XGBoost(推薦,平衡性能和速度)
- lightgbm - LightGBM(速度最快,記憶體效率高)
- random_forest - 隨機森林(更穩定,容易過擬合較少)

預設值:xgboost

效能對比:

| 模型 | 速度 | 精準度 | 記憶體 | 推薦場景 |
|------|------|--------|--------|----------|
| XGBoost | 中 | 高 | 中 | 一般用途 |
| LightGBM | 快 | 高 | 低 | 大數據集 |
| Random Forest | 慢 | 中 | 高 | 防止過擬合 |

使用方式:

```bash
python main.py --train --model-type lightgbm
python main.py --train --model-type random_forest
```

### 5. --opt

優化等級,決定模型的精準度和靈敏度權衡

可選值:
- conservative - 保守型(高精準度,低誤報)
- balanced - 均衡型(平衡精準度和靈敏度)
- aggressive - 激進型(高靈敏度,可能增加誤報)

預設值:balanced

詳細對比:

| 參數 | Conservative | Balanced | Aggressive |
|------|--------------|----------|------------|
| 使用SMOTE | 是 | 是 | 否 |
| 集合模型 | 是 | 否 | 否 |
| 精準度 | 85-88% | 82-85% | 78-82% |
| 召回率 | 75-80% | 78-82% | 85-90% |
| 誤報率 | 低 | 中 | 高 |
| 訊號數量 | 較少 | 中等 | 較多 |
| 訓練時間 | 較長 | 中等 | 最短 |

使用方式:

```bash
python main.py --train --opt conservative
python main.py --train --opt aggressive
```

## 常見訓練命令組合

### 組合 1: 快速訓練(推薦首次使用)

```bash
python main.py --train --symbol BTCUSDT --timeframe 1h --opt balanced
```

說明:比特幣1小時時間框,均衡優化

### 組合 2: 保守精準度優先

```bash
python main.py --train --symbol ETHUSDT --timeframe 4h --model-type xgboost --opt conservative
```

說明:以太坊4小時時間框,優先精準度,使用集合模型

### 組合 3: 高靈敏度多信號

```bash
python main.py --train --symbol BTCUSDT --timeframe 15m --model-type lightgbm --opt aggressive
```

說明:比特幣15分鐘,高靈敏度,快速模型

### 組合 4: 完整測試所有參數

```bash
python main.py --train --symbol LTCUSDT --timeframe 1d --model-type random_forest --opt balanced
```

說明:萊特幣1天時間框,隨機森林,均衡優化

### 組合 5: 批量訓練多個模型

```bash
python main.py --train --symbol BTCUSDT --timeframe 1h --opt conservative
python main.py --train --symbol ETHUSDT --timeframe 1h --opt conservative
python main.py --train --symbol LTCUSDT --timeframe 1h --opt conservative
```

## 訓練過程輸出說明

執行訓練後會看到以下輸出:

```
============================================================
Training 1h BB Model for BTCUSDT
Optimization Level: balanced
============================================================

Step 1: Loading data...
Data loaded: 5000 candles

Step 2: Engineering features...
Engineering features...

Step 3: Training model...
Preparing training data for BB bounce prediction...
Training data prepared: 450 BB touch/break events, 25 features
Timeframe: 1h
Effective bounces: 120, Ineffective: 330
Bounce rate: 26.67%
Applying SMOTE balancing (ratio=0.5)...
SMOTE completed: 580 samples (from 450)
Class distribution: 290 negative, 290 positive
Training xgboost model for 1h...
Hyperparameters: {'max_depth': 7, 'learning_rate': 0.08, 'n_estimators': 150}
Model training completed for 1h:
  Train accuracy: 0.8234, Precision: 0.8412, Recall: 0.7891
  Test accuracy: 0.7956, Precision: 0.8023, Recall: 0.7654

============================================================
Training Results:
============================================================
Symbol: BTCUSDT
Timeframe: 1h
Model Type: xgboost
Optimization: balanced
Train Accuracy: 0.8234
Test Accuracy: 0.7956
Train Precision: 0.8412
Test Precision: 0.8023
Train Recall: 0.7891
Test Recall: 0.7654
============================================================

Training completed successfully!
Model saved to: models/cache/BTCUSDT_1h_xgboost.joblib
```

### 結果解釋

- **Accuracy(準確率)**:模型預測正確的比例
- **Precision(精準度)**:預測為正的結果中,實際為正的比例(低誤報)
- **Recall(召回率)**:實際為正的結果中,被正確預測為正的比例(低漏報)

目標:Precision > 80%,Recall > 75%

## 訓練前準備

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 配置環境變數

複製 .env.example 到 .env 並配置:

```bash
cp .env.example .env
```

編輯 .env 檔案,填入必要的 API 密鑰

### 3. 確保數據充足

訓練需要至少 100 個 Bollinger Bands 觸及/突破事件。數據來源:
- Binance API(推薦)
- Glassnode API(鏈上數據)

## 訓練後檢查

### 模型文件位置

訓練後模型保存在:

```
models/cache/{symbol}_{timeframe}_{model_type}.joblib
```

例如:
- models/cache/BTCUSDT_1h_xgboost.joblib
- models/cache/ETHUSDT_4h_lightgbm.joblib

### 評估模型

檢查精準度指標:

- Test Precision > 80%:模型質量良好
- Test Precision 70-80%:模型質量一般
- Test Precision < 70%:需要重新訓練或調整參數

## 疑難排解

### 問題 1:"Not enough training data"

解決方案:

```bash
python main.py --train --symbol BTCUSDT --timeframe 4h
```

使用更長的時間框架獲得更多數據

### 問題 2:模型精準度低

嘗試:

```bash
python main.py --train --symbol BTCUSDT --timeframe 1h --opt conservative --model-type random_forest
```

使用保守優化和隨機森林模型

### 問題 3:訓練速度太慢

使用:

```bash
python main.py --train --symbol BTCUSDT --timeframe 1h --model-type lightgbm --opt aggressive
```

LightGBM 是最快的模型

## 啟動 Web 服務器

訓練完成後,可以啟動 Web 應用:

```bash
python main.py
```

或完整指定參數:

```bash
python app.py
```

然後訪問:http://localhost:5000
