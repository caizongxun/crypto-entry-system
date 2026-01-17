# 加密貨幣市場數據蒐集指南

本指南說明如何設置和運行自動化數據蒐集系統，以獲取市場恐慌情緒指數和大戶動向數據。

## 目錄

1. [快速開始](#快速開始)
2. [API 密鑰設置](#api-密鑰設置)
3. [數據檔案說明](#數據檔案說明)
4. [自動化執行](#自動化執行)
5. [數據分析](#數據分析)
6. [故障排除](#故障排除)

---

## 快速開始

### 前置需求

```bash
python >= 3.8
pip install pandas requests
```

### 設置環境變數

```bash
# 複製環境變數模板
cp .env.example .env

# 編輯 .env 並填入你的 API 密鑰
vim .env

# 載入環境變數
source .env  # Linux/macOS
# 或在 Windows 中
set /p < .env
```

### 第一次運行

```bash
# 蒐集市場情緒數據
python scripts/update_market_sentiment.py

# 蒐集大戶動向數據
python scripts/update_whale_data.py --all
```

---

## API 密鑰設置

### 1. Whale Alert API

**用途：** 追蹤超過 100 萬 USD 的大額交易

**取得步驟：**
1. 訪問 [https://whale-alert.io](https://whale-alert.io)
2. 創建免費帳戶
3. 進入 Dashboard > API
4. 複製 API Key
5. 設置環境變數：
   ```bash
   export WHALE_ALERT_API_KEY="your_key_here"
   ```

**功能特性：**
- 10+ 區塊鏈支持（比特幣、以太坊、瑞波幣等）
- 實時交易監控
- 自動標記已知實體（交易所、託管機構）

**API 限制：**
- 免費層：100 次請求/月
- 專業層：無限制

---

### 2. Glassnode API

**用途：** 取得深入的鏈上分析指標

**取得步驟：**
1. 訪問 [https://glassnode.com](https://glassnode.com)
2. 註冊免費帳戶
3. 進入 API 部分
4. 複製 API Key
5. 設置環境變數：
   ```bash
   export GLASSNODE_API_KEY="your_key_here"
   ```

**功能特性：**
- 超過 200+ 鏈上指標
- 大戶實體分析（Whale Distribution）
- 活躍地址計數
- 每日至 10 分鐘更新頻率

**API 限制：**
- 免費層：限制指標和頻率
- 專業層：完整訪問

---

### 3. CryptoQuant API

**用途：** 交易所進出分析和宏觀指標

**取得步驟：**
1. 訪問 [https://cryptoquant.com](https://cryptoquant.com)
2. 創建帳戶
3. 進入 API Console
4. 生成 API Token
5. 設置環境變數：
   ```bash
   export CRYPTOQUANT_API_KEY="your_key_here"
   ```

**功能特性：**
- Exchange Inflow/Outflow（交易所流入/流出）
- Netflow Analysis（淨流分析）
- Miner Revenue 和 Flow
- 自 2015 年的歷史數據

**API 限制：**
- 免費層：基本指標
- 專業層：所有指標

---

## 數據檔案說明

### market_fear_greed_sentiment.csv

**來源：** Alternative.me Fear and Greed Index

**更新頻率：** 每日（建議 UTC 00:00）

**欄位說明：**

| 欄位 | 類型 | 描述 |
|------|------|------|
| date | Date | YYYY-MM-DD 格式日期 |
| timestamp | Integer | Unix 時間戳（秒） |
| fear_greed_value | Integer | 0-100 分數 |
| sentiment_classification | String | 情緒分類 |
| data_source | String | 數據來源 |
| notes | String | 備註說明 |

**情緒分類：**
- 0-24：**Extreme Fear**（極度恐慌）
- 25-46：**Fear**（恐慌）
- 47-54：**Neutral**（中立）
- 55-75：**Greed**（貪婪）
- 76-100：**Extreme Greed**（極度貪婪）

**範例數據：**
```csv
2026-01-17,1737302400,67,Greed,Alternative.me,
2026-01-16,1737216000,65,Greed,Alternative.me,Latest data point
```

### daily_whale_movements.csv

**來源：** Whale Alert + Glassnode + CryptoQuant

**更新頻率：** 每日（建議 UTC 01:00）

**欄位說明：**

| 欄位 | 類型 | 描述 |
|------|------|------|
| date | Date | YYYY-MM-DD 格式日期 |
| timestamp | Integer | Unix 時間戳（秒） |
| blockchain | String | 區塊鏈名稱 |
| transaction_type | String | 交易類型 |
| from_address | String | 源地址或實體 |
| to_address | String | 目標地址或實體 |
| amount | Float | 交易數量（原生單位） |
| usd_value | Float | USD 等價值 |
| whale_classification | String | 鯨魚分類 |
| exchange_flow | String | 流向類型 |
| transaction_hash | String | 區塊鏈交易哈希 |
| data_source | String | 數據提供者 |
| notes | String | 備註說明 |

**鯨魚分類：**
- Whale>1k BTC：持有 1,000+ BTC 的實體
- Whale>10k BTC：持有 10,000+ BTC 的實體
- Large ETH Holder：大型以太坊持有者
- Exchange Flow：交易所聚合流
- Whale Alert Signal：Whale Alert 信號

**流向類型：**
- inflow：資金進入交易所（通常表示拋售意願）
- outflow：資金流出交易所（通常表示持有意願）
- transfer：實體間轉移
- mining：挖礦產出

---

## 自動化執行

### Linux/macOS - 使用 Cron

**編輯 Crontab：**
```bash
crontab -e
```

**每天早上 6 點收集市場情緒，下午 2 點收集鯨魚數據：**
```cron
# 市場情緒 - 每天 UTC 00:00 (台灣 08:00)
0 0 * * * cd /path/to/crypto-entry-system && source .env && python scripts/update_market_sentiment.py >> logs/sentiment.log 2>&1

# 鯨魚動向 - 每天 UTC 01:00 (台灣 09:00)
0 1 * * * cd /path/to/crypto-entry-system && source .env && python scripts/update_whale_data.py --all >> logs/whale.log 2>&1
```

**建立日誌目錄：**
```bash
mkdir -p logs
```

### Windows - 使用任務排程器

**建立批處理檔案 (`update_data.bat`)：**
```batch
@echo off
cd "C:\path\to\crypto-entry-system"
for /f "tokens=*" %%a in (.env) do set %%a
python scripts/update_market_sentiment.py
python scripts/update_whale_data.py --all
```

**設置任務排程：**
1. 打開「任務排程器」
2. 建立基本任務
3. 設置時間表
4. 操作：啟動程序 → 選擇 `update_data.bat`

### Docker - 容器化執行

**Dockerfile：**
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY scripts/ ./scripts/
COPY .env .
CMD ["python", "scripts/update_market_sentiment.py"] && ["python", "scripts/update_whale_data.py", "--all"]
```

**運行容器：**
```bash
docker build -t crypto-data-collector .
docker run --env-file .env crypto-data-collector
```

---

## 數據分析

### 基礎分析示例

**Python 載入和分析數據：**

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 載入市場情緒數據
sentiment_df = pd.read_csv('data_collection/market_fear_greed_sentiment.csv')
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

# 載入大戶動向數據
whale_df = pd.read_csv('data_collection/daily_whale_movements.csv')
whale_df['date'] = pd.to_datetime(whale_df['date'])

# 最近 30 天的平均恐慌指數
recent_30days = sentiment_df[sentiment_df['date'] > datetime.now() - timedelta(days=30)]
avg_fear = recent_30days['fear_greed_value'].mean()
print(f"過去 30 天平均恐慌指數: {avg_fear:.2f}")

# 分析大戶淨流
recent_whale = whale_df[whale_df['date'] > datetime.now() - timedelta(days=7)]
total_outflow = recent_whale[recent_whale['exchange_flow'] == 'outflow']['usd_value'].sum()
total_inflow = recent_whale[recent_whale['exchange_flow'] == 'inflow']['usd_value'].sum()
net_flow = total_inflow - total_outflow
print(f"過去 7 天淨流: ${net_flow:,.2f}")

# 相關性分析
sentiment_df = sentiment_df.set_index('date')
whale_daily = whale_df.groupby('date')['usd_value'].sum()
correlation = sentiment_df['fear_greed_value'].corr(whale_daily)
print(f"恐慌指數與大戶交易量相關性: {correlation:.3f}")
```

### 使用 Jupyter Notebook

**安裝：**
```bash
pip install jupyter
jupyter notebook
```

**在 notebook 中：**
```python
%matplotlib inline
import matplotlib.pyplot as plt

# 繪製恐慌指數趨勢
sentiment_df.plot(figsize=(14, 6))
plt.title('市場恐慌指數趨勢')
plt.xlabel('日期')
plt.ylabel('恐慌指數 (0-100)')
plt.show()
```

---

## 故障排除

### 常見問題

**Q1: API 密鑰無效**
- 驗證 .env 檔案中的密鑰是否正確
- 確認 API 帳戶仍然活躍
- 檢查是否超過 API 限制

**Q2: 網絡連接超時**
- 檢查互聯網連接
- 嘗試使用 VPN
- 檢查 API 提供商的狀態頁面

**Q3: 重複數據**
- 腳本已內建重複檢測
- 檢查 .csv 檔案中的 date 和 transaction_hash 欄位

**Q4: 記憶體不足**
- 對大型 CSV 檔案使用 chunks：
  ```python
  df = pd.read_csv('file.csv', chunksize=10000)
  ```

### 日誌檢查

```bash
# 檢查執行日誌
tail -f logs/sentiment.log
tail -f logs/whale.log

# 檢查錯誤
grep ERROR logs/*.log
```

---

## 最佳實踐

1. **定期備份**
   ```bash
   cp data_collection/*.csv backups/
   ```

2. **監控數據質量**
   - 每週檢查數據更新狀態
   - 驗證記錄計數

3. **安全管理 API 密鑰**
   - 永遠不要提交 .env 到版本控制
   - 定期輪換密鑰
   - 使用密鑰管理系統（如 AWS Secrets Manager）

4. **版本控制**
   ```bash
   # .gitignore
   .env
   data_collection/*.csv
   logs/
   backups/
   ```

---

## 支援和反饋

如有問題或建議，請提交 Issue 或 Pull Request。

---

## 參考資源

- [Alternative.me Fear & Greed Index](https://alternative.me/crypto/fear-and-greed-index/)
- [Whale Alert Documentation](https://docs.whale-alert.io)
- [Glassnode API Docs](https://docs.glassnode.com)
- [CryptoQuant API Docs](https://docs.cryptoquant.com)

---

**最後更新：** 2026-01-17
**版本：** 1.0.0
