# Data Collection Directory

此目錄用於存放加密貨幣市場數據蒐集檔案。

## 目錄結構

```
data_collection/
├── README.md                           # 本檔案
├── market_fear_greed_sentiment.csv     # 市場恐慌情緒指數 (Fear & Greed Index)
└── daily_whale_movements.csv           # 每日大戶動向數據
```

## 數據源

### 市場恐慌情緒指數 (market_fear_greed_sentiment.csv)
- **數據源**: Alternative.me Fear and Greed Index
- **更新頻率**: 每日
- **覆蓋時間**: 2018年2月至今
- **欄位**: 日期、恐慌指數值(0-100)、情緒分類

### 每日大戶動向 (daily_whale_movements.csv)
- **數據源**: Whale Alert + Glassnode + CryptoQuant
- **更新頻率**: 每日
- **覆蓋時間**: 2015年至今
- **欄位**: 日期、區塊鏈、交易類型、交易量、USD價值、大戶分類、交易所流向

## 使用說明

這些檔案可用於：
- 量化策略開發
- 市場情緒分析
- 大戶行動預測
- 交易信號生成

## 數據更新方式

使用提供的 Python 腳本自動更新這些檔案：
```bash
python scripts/update_market_sentiment.py
python scripts/update_whale_data.py
```

## 相關資源

- Fear & Greed Index: https://alternative.me/crypto/fear-and-greed-index/
- Whale Alert: https://whale-alert.io
- Glassnode: https://glassnode.com
- CryptoQuant: https://cryptoquant.com
