# 快速開始指南

## 5 分鐘快速設置

### 1. 複製環境變數模板
```bash
cp .env.example .env
```

### 2. 填入 API 密鑰
```bash
vim .env
```

填入以下三個 API 密鑰（至少一個）：
```
WHALE_ALERT_API_KEY=your_key
GLASSNODE_API_KEY=your_key
CRYPTOQUANT_API_KEY=your_key
```

### 3. 載入環境變數
```bash
source .env
```

### 4. 執行數據蒐集
```bash
# 蒐集市場情緒
python scripts/update_market_sentiment.py

# 蒐集大戶動向
python scripts/update_whale_data.py --all

# 或同時蒐集兩者
sh data_collection/run_all_collections.sh
```

---

## 數據檔案位置

```
data_collection/
├── market_fear_greed_sentiment.csv    # 市場恐慌情緒 (2018~今)
└── daily_whale_movements.csv          # 每日大戶動向 (2015~今)
```

---

## 常用命令

### 個別蒐集特定數據源
```bash
# 只蒐集 Whale Alert 數據
python scripts/update_whale_data.py --source whale-alert

# 只蒐集 Glassnode 數據
python scripts/update_whale_data.py --source glassnode

# 只蒐集 CryptoQuant 數據
python scripts/update_whale_data.py --source cryptoquant
```

### 查看最新數據
```bash
# 查看市場情緒最後 5 筆記錄
tail -5 data_collection/market_fear_greed_sentiment.csv

# 查看大戶動向最後 10 筆記錄
tail -10 data_collection/daily_whale_movements.csv
```

### 統計數據
```bash
# 計算市場情緒檔案行數 (含表頭)
wc -l data_collection/market_fear_greed_sentiment.csv

# 統計大戶動向的不同區塊鏈
cut -d',' -f3 data_collection/daily_whale_movements.csv | sort | uniq -c
```

---

## 定時自動更新 (Cron)

### Linux/macOS
```bash
crontab -e

# 添加以下行
# 每天 UTC 00:00 蒐集市場情緒（台灣時間 08:00）
0 0 * * * cd /path/to/crypto-entry-system && source .env && python scripts/update_market_sentiment.py

# 每天 UTC 01:00 蒐集大戶動向（台灣時間 09:00）
0 1 * * * cd /path/to/crypto-entry-system && source .env && python scripts/update_whale_data.py --all
```

---

## 數據分析示例

### 導入並查看數據
```python
import pandas as pd

# 讀取市場情緒數據
sentiment = pd.read_csv('data_collection/market_fear_greed_sentiment.csv')
print(f"總記錄數: {len(sentiment)}")
print(sentiment.tail(10))

# 讀取大戶動向數據
whales = pd.read_csv('data_collection/daily_whale_movements.csv')
print(f"總交易數: {len(whales)}")
print(whales.head())
```

### 計算統計信息
```python
import pandas as pd
import numpy as np

sentiment = pd.read_csv('data_collection/market_fear_greed_sentiment.csv')

# 當前恐慌指數
latest = sentiment.iloc[-1]
print(f"當前恐慌指數: {latest['fear_greed_value']}")
print(f"情緒分類: {latest['sentiment_classification']}")

# 過去 30 天的平均
recent_30 = sentiment['fear_greed_value'].tail(30).mean()
print(f"過去 30 天平均: {recent_30:.2f}")

# 過去 90 天的最高和最低
recent_90 = sentiment['fear_greed_value'].tail(90)
print(f"90 天最高: {recent_90.max()}")
print(f"90 天最低: {recent_90.min()}")
```

---

## API 密鑰獲取連結

| 平台 | 註冊連結 | 免費配額 |
|------|--------|--------|
| **Whale Alert** | https://whale-alert.io | 100/月 |
| **Glassnode** | https://glassnode.com | 有限 |
| **CryptoQuant** | https://cryptoquant.com | 基本指標 |

---

## 常見問題

**Q: 需要三個 API 密鑰都有嗎？**
A: 不需要。至少有一個就能開始蒐集數據。但三個密鑰能提供最完整的數據。

**Q: 數據會自動更新嗎？**
A: 不會。需要通過 Cron（Linux/Mac）或工作排程器（Windows）設置自動執行。

**Q: CSV 檔案太大怎麼辦？**
A: 使用 Pandas 的 chunksize 參數分塊讀取：
```python
df = pd.read_csv('file.csv', chunksize=50000)
```

**Q: 如何備份數據？**
A: 定期複製 CSV 檔案到其他位置或使用 Git 提交：
```bash
cp data_collection/*.csv backups/
cd backups && date '+%Y%m%d' >> backup_log.txt
```

---

## 文件目錄結構

```
crypto-entry-system/
├── data_collection/
│   ├── README.md                          # 目錄說明
│   ├── QUICK_START.md                     # 本文件
│   ├── DATA_COLLECTION_GUIDE.md           # 完整指南
│   ├── market_fear_greed_sentiment.csv    # 市場情緒數據
│   └── daily_whale_movements.csv          # 大戶動向數據
├── scripts/
│   ├── update_market_sentiment.py         # 市場情緒蒐集腳本
│   └── update_whale_data.py               # 大戶動向蒐集腳本
├── .env.example                           # 環境變數模板
└── ...
```

---

## 下一步

1. 閱讀 [完整指南](DATA_COLLECTION_GUIDE.md) 了解更詳細的設置
2. 在量化策略中使用這些數據
3. 結合 Pine Script 指標和機器學習模型
4. 優化交易信號生成

---

**支援時間:** 2026 年 1 月 17 日
**版本:** 1.0.0
