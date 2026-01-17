# 15m Bollinger Band 模型 - 快速開始

## 一行命令開始

### 1. 命令行訓練模型

```bash
python scripts/train_15m_model.py BTCUSDT
```

### 2. 啟動應用並訪問網頁

```bash
python app.py
```

然後打開瀏覽器訪問：**http://localhost:5000/15m-analysis**

## 網頁界面功能

### 訓練模型
1. 輸入交易對（如 BTCUSDT、ETHUSDT）
2. 點擊 "Train Model" 按鈕
3. 等待訓練完成（通常 30-90 秒）
4. 查看訓練結果

### 查看預測
- 自動加載最新的 BB 反彈預測信號
- 每 60 秒自動刷新一次
- 手動點擊 "Refresh Predictions" 立即刷新

### 模型統計
- **總 BB 信號數**：所有 BB 觸及或突破事件
- **高概率信號**：反彈概率 > 70% 的信號
- **BB 位置**：價格在 BB 通道內的百分比位置
- **BB 寬度**：Bollinger Band 通道的寬度

## API 端點

### 獲取 15m 預測
```bash
GET /api/ml-prediction-15m?symbol=BTCUSDT&lookback=50
```

### 訓練 15m 模型
```bash
POST /api/train-model-15m
Body: {"symbol": "BTCUSDT"}
```

## 信號解釋

| 信號類型 | 含義 | 交易策略 |
|---------|------|----------|
| lower_touch | 下跌觸及下軌 | 買入信號，預期反彈上升 |
| lower_break | 下跌突破下軌 | 強買入信號，下跌已確認 |
| upper_touch | 上漲觸及上軌 | 賣出信號，預期回落下跌 |
| upper_break | 上漲突破上軌 | 強賣出信號，上漲已確認 |

## 快速配置調整

編輯 `models/ml_model.py` 中的 `BB_CONFIG_15m`：

```python
BB_CONFIG_15m = {
    'bb_period': 20,        # 提高到 25 使帶變寬
    'bb_std': 2.0,          # 改為 2.5 捕捉更多信號
    'lookforward': 5,       # 改為 10 看更長期反彈
    'bounce_threshold': 0.005  # 改為 0.01 提高反彈幅度要求
}
```

## 常見問題

**Q: 訓練失敗怎麼辦？**
A: 檢查 `models/cache/` 目錄是否存在。如不存在，創建該目錄。

**Q: 精確率太低？**
A: 可能市場不適合該策略。嘗試調整 BB 參數或增加訓練數據。

**Q: 如何訓練其他交易對？**
A: 
```bash
python scripts/train_15m_model.py ETHUSDT
python scripts/train_15m_model.py LTCUSDT
```

**Q: 預測實時更新嗎？**
A: 網頁每 60 秒自動刷新預測。手動點擊 "Refresh" 立即更新。

## 文件結構

```
crypto-entry-system/
├── app.py                          # Flask 應用主文件
├── scripts/
│   └── train_15m_model.py         # 訓練腳本
├── web/
│   └── templates/
│       └── 15m_analysis.html       # 15m 分析頁面
├── models/
│   ├── ml_model.py                # ML 模型
│   └── cache/                     # 模型保存目錄
│       └── BTCUSDT_15m_xgboost.joblib
└── docs/
    └── TRAINING_15M_MODEL.md      # 詳細指南
```

## 性能指標

模型訓練後會輸出：

```
Train Accuracy: 0.7234   # 訓練集準確率
Test Accuracy: 0.7012    # 測試集準確率 (重要)
Train Precision: 0.6850  # 訓練集精確率
Test Precision: 0.6520   # 測試集精確率 (重要)
```

重點看測試集指標（Test），這代表實際預測能力。

## 下一步

1. 訓練 1h 和 4h 模型進行多時框架確認
2. 添加風險管理規則（止損、止盈）
3. 回測歷史數據驗證策略
4. 實盤前紙交易驗證

## 更多信息

詳見 [TRAINING_15M_MODEL.md](docs/TRAINING_15M_MODEL.md)
