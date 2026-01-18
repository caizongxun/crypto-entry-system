class CookieManager {
  constructor() {
    this.cookieName = 'trading_config';
    this.defaultValues = {
      dataSource: 'binance',
      defaultSymbol: 'BTCUSDT',
      defaultTimeframe: '1h',
      minCandles: 50,
      confidenceThreshold: 0.7,
      initialBalance: 1000,
      positionSize: 10,
      autoTradingEnabled: true,
      chartHeight: 600,
      updateInterval: 2
    };
  }

  setCookie(key, value, days = 30) {
    const date = new Date();
    date.setTime(date.getTime() + days * 24 * 60 * 60 * 1000);
    const expires = 'expires=' + date.toUTCString();
    document.cookie = `${this.cookieName}_${key}=${encodeURIComponent(JSON.stringify(value))}; ${expires}; path=/; SameSite=Lax`;
    console.log(`Cookie set: ${key} = ${JSON.stringify(value)}`);
  }

  getCookie(key) {
    const name = `${this.cookieName}_${key}=`;
    const decodedCookie = decodeURIComponent(document.cookie);
    const cookieArray = decodedCookie.split(';');
    
    for (let cookie of cookieArray) {
      cookie = cookie.trim();
      if (cookie.indexOf(name) === 0) {
        try {
          return JSON.parse(cookie.substring(name.length));
        } catch (e) {
          console.error(`Error parsing cookie ${key}:`, e);
          return null;
        }
      }
    }
    return null;
  }

  getAllSettings() {
    const settings = {};
    
    for (const key of Object.keys(this.defaultValues)) {
      const value = this.getCookie(key);
      settings[key] = value !== null ? value : this.defaultValues[key];
    }
    
    return settings;
  }

  saveSettings(settings) {
    for (const [key, value] of Object.entries(settings)) {
      this.setCookie(key, value);
    }
  }

  deleteCookie(key) {
    document.cookie = `${this.cookieName}_${key}=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/; SameSite=Lax`;
    console.log(`Cookie deleted: ${key}`);
  }

  clearAllSettings() {
    for (const key of Object.keys(this.defaultValues)) {
      this.deleteCookie(key);
    }
  }

  resetToDefaults() {
    this.clearAllSettings();
    return this.defaultValues;
  }
}

const cookieManager = new CookieManager();
