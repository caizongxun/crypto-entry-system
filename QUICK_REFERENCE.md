# 訓練指令快速參考卡

## 最常用指令

```bash
python main.py --train --symbol BTCUSDT --timeframe 1h --opt balanced
```

## 所有參數速查表

### 符號 (--symbol)

| 符號 | 說明 |
|------|------|
| BTCUSDT | 比特幣(預設) |
| ETHUSDT | 以太坊 |
| LTCUSDT | 萊特幣 |
| BNBUSDT | 幣安幣 |
| ADAUSDT | 卡爾達諾 |
| BTC | 系統自動補USDT後綴 |
| ETH | 系統自動補USDT後綴 |

### 時間框架 (--timeframe)

| 框架 | 說明 |
|------|------|
| 15m | 15分鐘(短期) |
| 1h | 1小時(預設,推薦) |
| 4h | 4小時 |
| 1d | 1天(長期) |

### 模型類型 (--model-type)

| 模型 | 速度 | 精度 | 記憶 | 推薦場景 |
|------|------|------|------|----------|
| xgboost | 中 | 高 | 中 | 一般用途(預設) |
| lightgbm | 快 | 高 | 低 | 大數據集 |
| random_forest | 慢 | 中 | 高 | 防止過擬合 |

### 優化等級 (--opt)

| 等級 | 精度 | 靈敏 | 誤報 | SMOTE | 集合 | 用途 |
|------|------|------|------|-------|------|------|
| conservative | 85-88% | 低 | 低 | 是 | 是 | 優先精確 |
| balanced | 82-85% | 中 | 中 | 是 | 否 | 均衡(預設) |
| aggressive | 78-82% | 高 | 高 | 否 | 否 | 多信號 |

## 場景化指令範本

### 場景1: 第一次訓練(推薦)

```bash
python main.py --train
```

或完整版:

```bash
python main.py --train --symbol BTCUSDT --timeframe 1h --model-type xgboost --opt balanced
```

### 場景2: 優先精準度

```bash
python main.py --train --symbol BTCUSDT --timeframe 1h --opt conservative
```

### 場景3: 多信號,快速

```bash
python main.py --train --symbol BTCUSDT --timeframe 1h --model-type lightgbm --opt aggressive
```

### 場景4: 長期K線

```bash
python main.py --train --symbol BTCUSDT --timeframe 4h
```

### 場景5: 短期K線

```bash
python main.py --train --symbol BTCUSDT --timeframe 15m
```

### 場景6: 訓練其他幣種

```bash
python main.py --train --symbol ETHUSDT
python main.py --train --symbol LTCUSDT
```

### 場景7: 批量訓練

```bash
python main.py --train --symbol BTCUSDT --opt conservative
python main.py --train --symbol ETHUSDT --opt conservative
python main.py --train --symbol LTCUSDT --opt conservative
```

## 訓練輸出結果解讀

```
Training data prepared: 450 BB touch/break events, 25 features
 ^
 訊號數量                     使用特徵數

Bounce rate: 26.67%
 有效反彈比例(越高越容易訓練)

Test Precision: 0.8023
 精準度(目標>0.80)

Test Recall: 0.7654
 召回率(目標>0.75)
```

### 評估標準

| 指標 | 優秀 | 良好 | 一般 | 不佳 |
|------|------|------|------|------|
| Precision | >85% | 80-85% | 70-80% | <70% |
| Recall | >85% | 80-85% | 70-80% | <70% |
| Accuracy | >85% | 80-85% | 70-80% | <70% |

## 檔案位置

### 訓練數據

```
data/
  historical/
    {symbol}_{timeframe}.csv
```

### 已訓練模型

```
models/cache/
  {symbol}_{timeframe}_{model_type}.joblib
```

例如:
- models/cache/BTCUSDT_1h_xgboost.joblib
- models/cache/ETHUSDT_4h_lightgbm.joblib

## 常見問題快速解決

### 訊號太少

```bash
python main.py --train --symbol BTCUSDT --timeframe 4h
```
(使用更長時間框架)

### 精準度太低

```bash
python main.py --train --symbol BTCUSDT --timeframe 1h --opt conservative
```
(使用保守優化)

### 訓練太慢

```bash
python main.py --train --symbol BTCUSDT --timeframe 1h --model-type lightgbm --opt aggressive
```
(使用LightGBM和激進模式)

### 記憶體不足

```bash
python main.py --train --symbol BTCUSDT --timeframe 15m --model-type lightgbm
```
(使用LightGBM和短時間框架)

## 完整參數組合數

符號: 7種 x 時間框架: 4種 x 模型: 3種 x 優化: 3種 = 252種組合

推薦優先嘗試:

1. python main.py --train
2. python main.py --train --opt conservative
3. python main.py --train --model-type lightgbm
4. python main.py --train --symbol ETHUSDT

## 啟動Web服務

```bash
python main.py
```

訪問: http://localhost:5000
