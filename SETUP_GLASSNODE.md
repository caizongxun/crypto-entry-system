# Glassnode API 快速設置指南

## 一. 註冊 Glassnode API

### 步驟 1: 註冊帳戶
1. 訪問 https://glassnode.com/
2. 點擊 "Sign Up" 或 "Start Free"
3. 使用 Email 和密碼註冊
4. 驗證 Email

### 步驟 2: 獲取 API Key
1. 登入 Glassnode 决寶板
2. 進入 **Settings** > **API**
3. 點擊 **Generate New Key**
4. 複製交五時生成的 API Key

### 步驟 3: 配置环境變數

```bash
# 1. 複製 .env 樣本檔
cp .env.example .env

# 2. 編輯 .env 檔
vi .env

# 3. 填入你的 API Key
GLASSNODE_API_KEY=your_api_key_here

# 4. 可選: 填入 Binance API Key (成機上線需要)
BINANCE_API_KEY=your_binance_key
BINANCE_API_SECRET=your_binance_secret
```

## 二. 檢詳 API 盟接

### 方法 1: 使用 curl

```bash
# 檢户帳戶是否接成功
 curl -H "x-api-key: your_glassnode_api_key" \
  https://api.glassnode.com/v1/entities/top_holders?a=bitcoin&limit=5

# 查看系統是否正常運作
http://localhost:5000/api/health
```

### 方法 2: 使用 Python

```python
import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GLASSNODE_API_KEY')

headers = {'x-api-key': api_key}
response = requests.get(
    'https://api.glassnode.com/v1/entities/top_holders',
    params={'a': 'bitcoin', 'limit': 5},
    headers=headers
)
print(response.json())
```

## 三. 驗證 API 端點

### 1. 檢查系統狀態

```bash
GET http://localhost:5000/api/health
```

**預期響應** (API Key 配置應會顯示 'enabled'):
```json
{
  "status": "healthy",
  "services": {
    "whale_tracker": "enabled",
    "exchange_analyzer": "enabled",
    "ml_model": "loaded",
    "paper_trading": "enabled"
  }
}
```

### 2. 獲取大額交易

```bash
GET http://localhost:5000/api/on-chain/whale-transactions?symbol=BTCUSDT&limit=5
```

### 3. 查看交易所資金流向

```bash
GET http://localhost:5000/api/on-chain/exchange-flows?symbol=BTCUSDT
```

### 4. 設取少 简易测試脚本

```python
import requests

base_url = 'http://localhost:5000/api/on-chain'

# 檢查系統是否正常
response = requests.get(f'{base_url}/status')
print('Status:', response.json())

# 獲取大額交易
response = requests.get(f'{base_url}/whale-transactions?symbol=BTCUSDT&limit=5')
print('Whale Transactions:', response.json()['count'], 'records')

# 查看交易所流向
response = requests.get(f'{base_url}/exchange-flows?symbol=BTCUSDT')
for exchange in response.json()['data']:
    print(f"{exchange['exchange_name']}: {exchange['net_flow']:+.2f}")

# 取得市場情緒
response = requests.get(f'{base_url}/exchange-pressure?symbol=BTCUSDT')
data = response.json()['data']
print(f"Market Pressure: {data['interpretation']} (Score: {data['pressure_score']:.1f})")
```

## 四. 安裝依賴

### 接該碼一次

```bash
# 安裝所有依賴
 pip install -r requirements.txt

# 或手動安裝
 pip install requests python-binance Flask flask-cors pandas python-dotenv
```

### requirements.txt

```
Flask==2.3.0
flask-cors==4.0.0
python-binance==1.0.17
pandas==2.0.0
python-dotenv==1.0.0
requests==2.31.0
scikit-learn==1.3.0
Joblib==1.3.0
```

## 五. 啟動應用

```bash
# 開發模式
python app.py

# 或使用 Flask CLI
flask run

# 特定端口
flask run --port 5000
```

應用即會在 http://localhost:5000 上運行。

## 六. API Key 匹配鐡你标

| 需求 | Glassnode Plan | 免費測試 | 成本 |
|------|----------------|---------|---------|
| 大戶追蹤 | Professional+ | 4 weeks | $999/月 |
| 交易所流向 | Professional+ | 是 | $999/月 |
| 鯨魚錢包 | Professional+ | 是 | $999/月 |
| 基礎功能 | Free | 是 | 免費 |

## 七. 常見幫助

### 錯誤: "API Key not found"
```
解決: 確認 .env 檔存在且包含 GLASSNODE_API_KEY
echo $GLASSNODE_API_KEY  # 檢查是否已設置
```

### 錯誤: "403 Forbidden"
```
解決: API Key 無效或訂閱訂間已過戉
1. 稽紅 Glassnode 兜好板
2. 確認訂閱訂間是否仍有效
3. 爱你的 API Key 並重新訪堵
```

### 錯誤: "Timeout"
```
解決: 網路連線不達
1. 梡骗 VPN 是否開啟
2. 確認你的網一絥
3. 減少 API 請求頻率
```

## 八. 下一步

1. 閱讀 [API 文檔](./docs/API_SETUP.md) 了解所有端點
2. 查看 [大戶追蹤功能](./docs/WHALE_TRACKING.md) 的使用場景
3. 集成市場情緒到你的交易武器
4. 站庨安民並開啟是你的抃金機

## 乞. 获得支持

- Glassnode 官方文檔: https://docs.glassnode.com/
- 我們的 GitHub Issues: https://github.com/caizongxun/crypto-entry-system/issues

祝你成功配置︁
