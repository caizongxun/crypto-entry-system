# API 設置指南

## 概述

本系統使用以下數據提供商來获取加密貨幣市場數據：

### 1. Glassnode API (大戶鏈上動向)
**用途**: 追蹤大額交易、交易所進出資金、鯨魚錢包

**註冊步驟**:
1. 訪問 https://glassnode.com/
2. 點擊「Sign Up」
3. 使用 Email 和密碼註冊
4. 驗證 Email
5. 登入後進入 Settings → API
6. 複製 API Key

**定價**:
- Free Plan: 基礎功能
- Professional Plan: $999/月 (推薦用於交易系統)
- Enterprise: 自訂價格

**API Key 配置**:
```bash
# .env 文件
GLASSNODE_API_KEY=your_api_key_here
```

### 2. Binance API (市場數據和交易執行)
**用途**: 獲取實時價格、執行交易、查詢賬戶餘額

**註冊步驟**:
1. 登入 https://www.binance.com/
2. 進入「User Center" → "API Management"
3. 點擊「Create API"
4. 設置 API Label (如 "Crypto Entry System")
5. 完成 2FA 驗證
6. 複製 API Key 和 Secret Key
7. 限制 IP 訪問 (推薦: 只允許你的服務器 IP)

**配置**:
```bash
# .env 文件
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

**安全設置**:
- 啟用「Restrict access to trusted IPs only"
- 禁用「Enable Futures" 和 "Enable Margin"
- 啟用 "Enable Withdrawals"

## Glassnode 數據端點

### 1. 獲取大額交易
```bash
GET /api/on-chain/whale-transactions?symbol=BTCUSDT&limit=20&hours=24
```

**響應示例**:
```json
{
  "status": "success",
  "symbol": "BTCUSDT",
  "data": [
    {
      "txid": "abc123...",
      "timestamp": "2026-01-17T12:00:00Z",
      "from_address": "1A1z7aga...",
      "to_address": "3J98t1wp...",
      "amount": 150.5,
      "value_usd": 6321000,
      "type": "outflow",
      "block_height": 821234
    }
  ]
}
```

### 2. 交易所進出資金流
```bash
GET /api/on-chain/exchange-flows?symbol=BTCUSDT&hours=24
```

**解釋**:
- **inflow**: 進入交易所的資金 (可能預示下跌)
- **outflow**: 離開交易所的資金 (可能預示上漲)
- **net_flow**: 淨流量 (正值 = 進入, 負值 = 流出)

### 3. 市場壓力分析
```bash
GET /api/on-chain/exchange-pressure?symbol=BTCUSDT
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
    "interpretation": "Balanced"
  }
}
```

**壓力評級**:
- > 60: 強勁買盤壓力
- 30-60: 溫和買盤壓力
- -30 to 30: 平衡
- -30 to -60: 溫和賣盤壓力
- < -60: 強勁賣盤壓力

### 4. 鯨魚錢包排名
```bash
GET /api/on-chain/whale-wallets?symbol=BTCUSDT&min_balance=1000
```

### 5. 交易量統計
```bash
GET /api/on-chain/transaction-volume?symbol=BTCUSDT&hours=24
```

## 如果沒有 API Key

所有端點都使用**模擬數據**返回，系統仍然可以正常運作。要啟用實時數據，需要配置 Glassnode API Key。

## 替代數據源

### CryptoQuant
- URL: https://cryptoquant.com/
- 專注於交易所數據和鏈上分析
- 可作為 Glassnode 的補充

### Whale Alert
- URL: https://whale-alert.io/
- 實時大額交易通知
- 適合即時監控

## 故障排除

### 常見錯誤

**1. "未授權"或 403 錯誤**
```
原因: API Key 無效或過期
解決: 檢查 .env 文件中的 API Key
```

**2. "配額超出"**
```
原因: 超過免費計劃限制
解決: 升級 Glassnode 訂閱或減少請求頻率
```

**3. 超時錯誤**
```
原因: 網路連線問題或 API 響應緩慢
解決: 檢查網路連線，增加超時時間
```

## 成本估算

| 服務 | 免費層 | 專業層 | 用途 |
|------|-------|--------|------|
| Glassnode | 有限 | $999/月 | 鏈上數據 |
| Binance API | 免費 | 免費 | 市場數據和交易 |
| Whale Alert | $20/月 | 可選 | 實時通知 |

## 環境變數設置

複製 `.env.example` 為 `.env`：

```bash
cp .env.example .env
```

編輯 `.env` 並填入您的 API Key：

```bash
# 大戶鏈上動向
GLASSNODE_API_KEY=your_key_here

# 現貨交易
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_key_here
```

## 驗證設置

啟動應用後，訪問健康檢查端點：

```bash
GET http://localhost:5000/api/on-chain/status
```

**預期響應**:
```json
{
  "status": "success",
  "provider": "Glassnode",
  "supported_contracts": ["bitcoin", "ethereum"]
}
```
