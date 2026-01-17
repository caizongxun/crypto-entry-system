# 大戶追蹤系統文檔

## 功能概述

本系統集成 Glassnode API，提供以下大戶鏈上動向分析功能：

### 1. 大額交易追蹤 (Whale Transactions)
監控超大額交易活動，幫助識別鯨魚錢包的進出情況。

### 2. 交易所資金流向分析 (Exchange Flows)
追蹤資金進出各大交易所的動向：
- Binance
- Coinbase
- Kraken
- Huobi
- FTX

### 3. 市場情緒指標 (Exchange Pressure)
根據交易所資金流向計算市場情緒：
- 正數 = 買盤情緒強
- 負數 = 賣盤情緒強
- 接近 0 = 市場平衡

### 4. 鯨魚錢包排名 (Top Whale Wallets)
監控大戶持倉排名和動向。

## API 端點

### 獲取大額交易
```bash
GET /api/on-chain/whale-transactions
?symbol=BTCUSDT&limit=20&hours=24
```

**參數**:
- `symbol`: 交易對 (BTCUSDT, ETHUSDT)
- `limit`: 返回最多交易數 (默認: 20)
- `hours`: 時間窗口，小時 (默認: 24)

**響應示例**:
```json
{
  "status": "success",
  "symbol": "BTCUSDT",
  "data": [
    {
      "txid": "a1234...",
      "timestamp": "2026-01-17T12:00:00Z",
      "from_address": "1A1z7aga...",
      "to_address": "3J98t1wp...",
      "amount": 150.5,
      "value_usd": 6321000,
      "type": "outflow",
      "block_height": 821234
    }
  ],
  "count": 1
}
```

### 交易所資金流向
```bash
GET /api/on-chain/exchange-flows
?symbol=BTCUSDT&hours=24
```

**響應示例**:
```json
{
  "status": "success",
  "symbol": "BTCUSDT",
  "data": [
    {
      "exchange_name": "Binance",
      "exchange_key": "binance",
      "inflow": 1250.5,
      "outflow": 980.75,
      "net_flow": 269.75,
      "total_reserve": 585420.3,
      "flow_type": "inflow",
      "timestamp": "2026-01-17T12:00:00Z"
    },
    {
      "exchange_name": "Coinbase",
      "exchange_key": "coinbase",
      "inflow": 450.25,
      "outflow": 320.1,
      "net_flow": 130.15,
      "total_reserve": 285610.2,
      "flow_type": "inflow",
      "timestamp": "2026-01-17T12:00:00Z"
    }
  ]
}
```

**字段解釋**:
- `inflow`: 進入交易所的資金 (可能預示下跌)
- `outflow`: 離開交易所的資金 (可能預示上漲)
- `net_flow`: 淨流量 (正 = 進入, 負 = 離開)
- `total_reserve`: 交易所總儲備量
- `flow_type`: 主要流向 (inflow/outflow)

### 市場情緒分析
```bash
GET /api/on-chain/exchange-pressure
?symbol=BTCUSDT
```

**響應示例**:
```json
{
  "status": "success",
  "data": {
    "total_inflow": 2201.8,
    "total_outflow": 1971.55,
    "net_flow": 230.25,
    "inflow_outflow_ratio": 1.117,
    "pressure_score": 5.5,
    "interpretation": "Balanced",
    "timestamp": "2026-01-17T12:00:00Z"
  }
}
```

**情緒等級**:
- 60 以上: 強烈買盤情緒
- 30-60: 溫和買盤情緒
- -30 到 30: 市場平衡
- -30 到 -60: 溫和賣盤情緒
- 低於 -60: 強烈賣盤情緒

### 鯨魚錢包排名
```bash
GET /api/on-chain/whale-wallets
?symbol=BTCUSDT&min_balance=1000
```

**參數**:
- `symbol`: 交易對
- `min_balance`: 最小持倉量 (BTC 單位)

### 交易量統計
```bash
GET /api/on-chain/transaction-volume
?symbol=BTCUSDT&hours=24
```

**響應示例**:
```json
{
  "status": "success",
  "data": {
    "total_volume_24h": 45000000000,
    "avg_hourly_volume": 1875000000,
    "peak_volume": 3500000000,
    "datapoints": 24
  }
}
```

### 按交易所分組交易
```bash
GET /api/on-chain/exchange-transactions
?symbol=BTCUSDT
```

**響應示例**:
```json
{
  "status": "success",
  "data": {
    "binance_inflow": [...],
    "binance_outflow": [...],
    "coinbase_inflow": [...],
    "coinbase_outflow": [...],
    "kraken_inflow": [...],
    "kraken_outflow": [...],
    "other": [...]
  }
}
```

## 使用場景

### 1. 識別底部買進信號
```
情景: 
- 交易所 outflow 增加 (大戶取出資金)
- Exchange Pressure Score > 50 (買盤情緒強)
- 大額買單進入冷錢包

解釋: 鯨魚在積累頭寸，市場可能築底
```

### 2. 識別頂部賣出信號
```
情景:
- 交易所 inflow 增加 (大戶進入資金)
- Exchange Pressure Score < -50 (賣盤情緒強)
- 大額轉帳進入交易所

解釋: 鯨魚在清倉頭寸，市場可能見頂
```

### 3. 監控交易所流動性
```
情景:
- 某交易所 total_reserve 快速下降
- 多個交易所同時出現 outflow

解釋: 市場進入風險事件，大戶急速出逃
```

## 數據更新頻率

- **大額交易**: 實時 (需要 Glassnode 訂閱)
- **交易所流向**: 每小時更新
- **市場情緒**: 每小時更新
- **鯨魚錢包**: 每 6 小時更新

## 無 API Key 的運行模式

所有端點都支持模擬數據模式（使用合理的虛假數據），無需配置 API Key 即可測試功能。生產環境中建議配置 Glassnode API Key 以獲得實時準確數據。

## 故障排除

### API 返回 403 Forbidden
```
原因: API Key 無效或過期
解決: 
1. 檢查 .env 文件中的 GLASSNODE_API_KEY
2. 登錄 Glassnode 確認 API Key 有效
3. 檢查訂閱計劃是否支持所需功能
```

### 請求超時
```
原因: 網路連線不穩定或 API 響應緩慢
解決:
1. 檢查網路連線
2. 減少 limit 參數
3. 增加 timeout 時間
```

### 數據異常
```
原因: 區塊鏈數據延遲或提供商同步問題
解決:
1. 等待 5-10 分鐘後重試
2. 檢查 Glassnode 狀態頁面
3. 嘗試其他時間範圍
```

## 最佳實踐

1. **不要過度查詢**: 遵守 API 配額限制
2. **組合信號**: 不要單獨依賴一個指標
3. **驗證數據**: 交叉驗證多個數據源
4. **風險管理**: 基於鯨魚動向調整倉位
5. **定時檢查**: 設置定時任務監控關鍵指標
