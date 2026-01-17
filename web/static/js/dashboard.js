const API_BASE = 'http://localhost:5000/api';
let currentSymbol = 'BTCUSDT';
let priceChart = null;
let priceMovementChart = null;
let volumeChart = null;
let fgiChart = null;
let mlPredictions = [];
let onChainData = {};
let currentTimeframe = '60';
let notifications = [];
let chartHeight = 600;
let refreshInterval = 15000;
let enableOnChainFilter = true;

const timeframeMap = {
    '1': { minutes: 1, interval: '1m', label: '1m' },
    '3': { minutes: 3, interval: '3m', label: '3m' },
    '5': { minutes: 5, interval: '5m', label: '5m' },
    '15': { minutes: 15, interval: '15m', label: '15m' },
    '30': { minutes: 30, interval: '30m', label: '30m' },
    '60': { minutes: 60, interval: '1h', label: '1h' },
    '240': { minutes: 240, interval: '4h', label: '4h' },
    '1440': { minutes: 1440, interval: '1d', label: '1d' }
};

const chartColors = {
    upCandle: '#00c853',
    downCandle: '#ff3860',
    volume: '#1f6feb',
    gridLine: '#3a4556',
    text: '#b0b8c8',
    background: '#0d1117'
};

document.addEventListener('DOMContentLoaded', () => {
    console.log('Dashboard initialized');
    loadSettings();
    initializeEventListeners();
    initializeResizeHandles();
    initializeCharts();
    loadDashboardData();
    setupTimeframeButtons();
    setInterval(loadDashboardData, refreshInterval);
});

function loadSettings() {
    const savedSettings = localStorage.getItem('dashboardSettings');
    if (savedSettings) {
        try {
            const settings = JSON.parse(savedSettings);
            currentTimeframe = settings.defaultTimeframe || '60';
            refreshInterval = (settings.refreshInterval || 15) * 1000;
            enableOnChainFilter = settings.enableOnChainFilter !== false;
            
            document.getElementById('themeSelect').value = settings.theme || 'dark';
            document.getElementById('defaultTimeframe').value = currentTimeframe;
            document.getElementById('enableOnChainFilter').checked = enableOnChainFilter;
            document.getElementById('enableNotifications').checked = settings.enableNotifications !== false;
            document.getElementById('refreshInterval').value = refreshInterval / 1000;
            
            if (settings.theme === 'light') {
                document.documentElement.style.colorScheme = 'light';
            }
        } catch (error) {
            console.error('Error loading settings:', error);
        }
    }
}

function setupTimeframeButtons() {
    const buttons = document.querySelectorAll('.timeframe-btn');
    buttons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            buttons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentTimeframe = btn.dataset.timeframe;
            loadPriceData();
        });
    });
}

function initializeResizeHandles() {
    const handles = document.querySelectorAll('.chart-resize-handle');
    let isResizing = false;
    let startY = 0;
    let startHeight = 0;
    let currentHandle = null;

    handles.forEach(handle => {
        handle.addEventListener('mousedown', (e) => {
            isResizing = true;
            startY = e.clientY;
            currentHandle = handle;
            const wrapper = handle.previousElementSibling;
            startHeight = wrapper.clientHeight;
            document.body.style.userSelect = 'none';
            document.body.style.cursor = 'ns-resize';
        });
    });

    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;
        const delta = e.clientY - startY;
        const wrapper = currentHandle.previousElementSibling;
        wrapper.style.minHeight = (startHeight + delta) + 'px';
        if (priceChart) priceChart.resize();
    });

    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            document.body.style.userSelect = 'auto';
            document.body.style.cursor = 'default';
        }
    });
}

function initializeCharts() {
    initializePriceChart();
    initializePriceMovementChart();
    initializeVolumeChart();
    initializeFGIChart();
}

function initializePriceChart() {
    const ctx = document.getElementById('priceChart');
    if (!ctx) return;
    
    if (priceChart) priceChart.destroy();
    
    priceChart = new Chart(ctx, {
        type: 'candlestick',
        data: {
            labels: [],
            datasets: [{
                label: 'Price',
                data: [],
                borderColor: chartColors.upCandle,
                backgroundColor: chartColors.upCandle,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    labels: { color: chartColors.text }
                }
            },
            scales: {
                x: {
                    grid: { color: chartColors.gridLine, drawBorder: false },
                    ticks: { color: chartColors.text }
                },
                y: {
                    grid: { color: chartColors.gridLine, drawBorder: false },
                    ticks: { color: chartColors.text }
                }
            }
        }
    });
}

function initializePriceMovementChart() {
    const ctx = document.getElementById('priceMovementChart');
    if (!ctx) return;
    
    if (priceMovementChart) priceMovementChart.destroy();
    
    priceMovementChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: '24h Price',
                data: [],
                borderColor: chartColors.volume,
                backgroundColor: 'rgba(31, 111, 235, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    labels: { color: chartColors.text }
                }
            },
            scales: {
                x: {
                    grid: { color: chartColors.gridLine, drawBorder: false },
                    ticks: { color: chartColors.text }
                },
                y: {
                    grid: { color: chartColors.gridLine, drawBorder: false },
                    ticks: { color: chartColors.text }
                }
            }
        }
    });
}

function initializeVolumeChart() {
    const ctx = document.getElementById('volumeChart');
    if (!ctx) return;
    
    if (volumeChart) volumeChart.destroy();
    
    volumeChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Volume',
                data: [],
                backgroundColor: chartColors.volume,
                borderColor: chartColors.upCandle,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    labels: { color: chartColors.text }
                }
            },
            scales: {
                x: {
                    grid: { color: chartColors.gridLine, drawBorder: false },
                    ticks: { color: chartColors.text }
                },
                y: {
                    grid: { color: chartColors.gridLine, drawBorder: false },
                    ticks: { color: chartColors.text }
                }
            }
        }
    });
}

function initializeFGIChart() {
    const ctx = document.getElementById('fgiChart');
    if (!ctx) return;
    
    if (fgiChart) fgiChart.destroy();
    
    fgiChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Fear', 'Greed'],
            datasets: [{
                data: [50, 50],
                backgroundColor: ['#ff3860', '#00c853'],
                borderColor: chartColors.background,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    labels: { color: chartColors.text },
                    position: 'bottom'
                }
            }
        }
    });
}

function initializeEventListeners() {
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const section = item.dataset.section;
            switchSection(section);
        });
    });

    const symbolInput = document.getElementById('symbolInput');
    if (symbolInput) {
        symbolInput.addEventListener('change', (e) => {
            currentSymbol = e.target.value.toUpperCase();
            if (!currentSymbol.endsWith('USDT')) {
                currentSymbol = currentSymbol + 'USDT';
            }
            document.getElementById('chartTitle').textContent = currentSymbol.replace('USDT', ' / USDT');
            loadPriceData();
            loadDashboardData();
        });
    }

    const refreshBtn = document.getElementById('refreshBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', loadDashboardData);
    }

    const form = document.getElementById('openPositionForm');
    if (form) {
        form.addEventListener('submit', openPosition);
    }

    const notificationBell = document.getElementById('notificationBell');
    if (notificationBell) {
        notificationBell.addEventListener('click', openNotificationsModal);
    }

    const settingsBtn = document.getElementById('settingsBtn');
    if (settingsBtn) {
        settingsBtn.addEventListener('click', openSettingsModal);
    }

    document.getElementById('notificationClose').addEventListener('click', closeNotificationsModal);
    document.getElementById('settingsClose').addEventListener('click', closeSettingsModal);
    document.getElementById('saveSettings').addEventListener('click', saveSettings);

    document.getElementById('notificationModal').addEventListener('click', (e) => {
        if (e.target.id === 'notificationModal') closeNotificationsModal();
    });

    document.getElementById('settingsModal').addEventListener('click', (e) => {
        if (e.target.id === 'settingsModal') closeSettingsModal();
    });
}

function openNotificationsModal() {
    document.getElementById('notificationModal').classList.add('open');
    loadNotifications();
}

function closeNotificationsModal() {
    document.getElementById('notificationModal').classList.remove('open');
}

function openSettingsModal() {
    document.getElementById('settingsModal').classList.add('open');
}

function closeSettingsModal() {
    document.getElementById('settingsModal').classList.remove('open');
}

function loadNotifications() {
    const notificationList = document.getElementById('notificationList');
    if (notifications.length === 0) {
        notificationList.innerHTML = '<p style="color: #8b949e; text-align: center;">No notifications yet</p>';
        return;
    }

    notificationList.innerHTML = notifications.slice().reverse().map((notif, idx) => `
        <div class="notification-item ${notif.read ? '' : 'unread'}">
            <div>
                <h4 style="margin: 0 0 4px 0; color: #f0f6fc;">${notif.title}</h4>
                <p style="margin: 0; color: #8b949e; font-size: 13px;">${notif.message}</p>
            </div>
            <span class="notification-time">${new Date(notif.timestamp).toLocaleString()}</span>
        </div>
    `).join('');
}

function saveSettings() {
    const settings = {
        theme: document.getElementById('themeSelect').value,
        defaultTimeframe: document.getElementById('defaultTimeframe').value,
        enableOnChainFilter: document.getElementById('enableOnChainFilter').checked,
        enableNotifications: document.getElementById('enableNotifications').checked,
        refreshInterval: parseInt(document.getElementById('refreshInterval').value),
        apiKey: document.getElementById('apiKey').value,
        apiSecret: document.getElementById('apiSecret').value
    };

    localStorage.setItem('dashboardSettings', JSON.stringify(settings));
    
    currentTimeframe = settings.defaultTimeframe;
    refreshInterval = settings.refreshInterval * 1000;
    enableOnChainFilter = settings.enableOnChainFilter;

    if (settings.theme === 'light') {
        document.documentElement.style.colorScheme = 'light';
    } else {
        document.documentElement.style.colorScheme = 'dark';
    }

    addNotification('Settings Saved', 'Your preferences have been saved successfully.');
    closeSettingsModal();
}

function switchSection(section) {
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    
    const sectionElement = document.getElementById(`${section}-section`);
    if (sectionElement) {
        sectionElement.classList.add('active');
    }
    
    const navItem = document.querySelector(`[data-section="${section}"]`);
    if (navItem) {
        navItem.classList.add('active');
    }
}

async function loadPriceData() {
    try {
        const response = await fetch(`${API_BASE}/market/kline?symbol=${currentSymbol}&interval=${timeframeMap[currentTimeframe].interval}&limit=50`);
        const data = await response.json();
        
        if (data.status === 'success') {
            updatePriceChart(data.klines);
        }
    } catch (error) {
        console.error('Error loading price data:', error);
    }
}

function updatePriceChart(klines) {
    if (!priceChart || !klines || klines.length === 0) return;
    
    const labels = klines.map(k => {
        const date = new Date(k.timestamp || k.open_time);
        return date.toLocaleTimeString();
    });
    
    const data = klines.map(k => ({
        o: parseFloat(k.open),
        h: parseFloat(k.high),
        l: parseFloat(k.low),
        c: parseFloat(k.close)
    }));
    
    priceChart.data.labels = labels;
    priceChart.data.datasets[0].data = data;
    priceChart.update('none');
}

async function loadDashboardData() {
    try {
        console.log('Loading dashboard data...');
        
        const promises = [
            fetch(`${API_BASE}/ml-prediction?symbol=${currentSymbol}`).then(r => r.json()).catch(err => ({ status: 'error', message: err.message })),
            fetch(`${API_BASE}/sentiment`).then(r => r.json()).catch(err => ({ status: 'error', message: err.message })),
            fetch(`${API_BASE}/trading/account-summary`).then(r => r.json()).catch(err => ({ status: 'error', message: err.message })),
            fetch(`${API_BASE}/on-chain?symbol=${currentSymbol}`).then(r => r.json()).catch(err => ({ status: 'error', message: err.message })),
            fetch(`${API_BASE}/market/kline?symbol=${currentSymbol}&interval=${timeframeMap[currentTimeframe].interval}&limit=24`).then(r => r.json()).catch(err => ({ status: 'error', message: err.message }))
        ];
        
        const [predictions, sentiment, accountData, onChain, marketData] = await Promise.all(promises);
        
        if (predictions.status === 'success') {
            updateMLPredictions(predictions);
            checkSignalNotifications(predictions);
        } else {
            console.error('ML predictions error:', predictions.message);
        }
        
        if (sentiment.status === 'success') {
            updateSentiment(sentiment);
        } else {
            console.error('Sentiment error:', sentiment.message);
        }
        
        if (accountData.status === 'success') {
            updateAccountInfo(accountData);
        } else {
            console.error('Account data error:', accountData.message);
        }

        if (onChain.status === 'success') {
            onChainData = onChain;
            updateOnChainData(onChain);
            checkWhaleAlerts(onChain);
        } else {
            console.error('On-chain data error:', onChain.message);
        }
        
        if (marketData.status === 'success') {
            updateMarketCharts(marketData.klines);
        }
    } catch (error) {
        console.error('Error loading dashboard data:', error);
    }
}

function updateMarketCharts(klines) {
    if (!klines || klines.length === 0) return;
    
    const labels = klines.map(k => {
        const date = new Date(k.timestamp || k.open_time);
        return date.toLocaleTimeString();
    });
    
    const prices = klines.map(k => parseFloat(k.close));
    const volumes = klines.map(k => parseFloat(k.volume));
    
    if (priceMovementChart) {
        priceMovementChart.data.labels = labels;
        priceMovementChart.data.datasets[0].data = prices;
        priceMovementChart.update('none');
    }
    
    if (volumeChart) {
        volumeChart.data.labels = labels;
        volumeChart.data.datasets[0].data = volumes;
        volumeChart.update('none');
    }
}

function getOnChainRiskLevel() {
    if (!onChainData.whale_summary) return 'Low';
    
    const summary = onChainData.whale_summary;
    const netFlow = (summary.inflow || 0) - (summary.outflow || 0);
    
    if (netFlow < -1000) return 'High';
    if (netFlow < -100) return 'Medium';
    return 'Low';
}

function updateOnChainData(data) {
    try {
        const summary = data.whale_summary || {};
        
        const onChainRisk = document.getElementById('onChainRisk');
        const whaleNetFlow = document.getElementById('whaleNetFlow');
        const networkActivity = document.getElementById('networkActivity');
        const exchangeReserve = document.getElementById('exchangeReserve');
        const whaleHoldings = document.getElementById('whaleHoldings');
        const netFlow24h = document.getElementById('netFlow24h');
        const whaleTransactionsList = document.getElementById('whaleTransactionsList');
        const exchangeFlowList = document.getElementById('exchangeFlowList');
        
        if (onChainRisk) {
            onChainRisk.textContent = getOnChainRiskLevel();
        }
        
        if (whaleNetFlow && summary.inflow !== undefined) {
            const netFlow = (summary.inflow || 0) - (summary.outflow || 0);
            whaleNetFlow.textContent = `${netFlow.toFixed(0)}`;
            whaleNetFlow.style.color = netFlow >= 0 ? '#00c853' : '#ff3860';
        }
        
        if (networkActivity) networkActivity.textContent = summary.active_addresses || '--';
        if (exchangeReserve) exchangeReserve.textContent = `${summary.exchange_holdings || '--'} BTC`;
        if (whaleHoldings) whaleHoldings.textContent = `${summary.whale_holdings || '--'}%`;
        if (netFlow24h && summary.inflow !== undefined) {
            const netFlow = (summary.inflow || 0) - (summary.outflow || 0);
            netFlow24h.textContent = `${netFlow >= 0 ? '+' : ''}${netFlow.toFixed(0)}`;
        }
        
        const transactions = data.large_transactions || [];
        if (whaleTransactionsList) {
            if (transactions.length === 0) {
                whaleTransactionsList.innerHTML = '<p class="empty-state">No large transactions</p>';
            } else {
                whaleTransactionsList.innerHTML = transactions.slice(0, 10).map(tx => `
                    <div class="whale-activity-item">
                        <div class="whale-flow">
                            <span style="color: #f0f6fc; font-weight: 600;">${tx.amount} BTC</span>
                            <span class="whale-amount ${tx.type === 'inflow' ? 'inflow' : 'outflow'}">${tx.type === 'inflow' ? 'IN' : 'OUT'}</span>
                        </div>
                        <div class="whale-info">${tx.from_address} ${tx.type === 'inflow' ? '->' : '<-'} ${tx.to_address}</div>
                        <div class="whale-info">Value: ${tx.value_usd || 'N/A'}</div>
                    </div>
                `).join('');
            }
        }
        
        const exchangeFlow = data.exchange_flows || [];
        if (exchangeFlowList) {
            if (exchangeFlow.length === 0) {
                exchangeFlowList.innerHTML = '<p class="empty-state">No exchange flow data</p>';
            } else {
                exchangeFlowList.innerHTML = exchangeFlow.slice(0, 10).map(flow => `
                    <div class="whale-activity-item">
                        <div class="whale-flow">
                            <span style="color: #f0f6fc;">${flow.exchange_name}</span>
                            <span class="whale-amount ${flow.flow_type === 'inflow' ? 'inflow' : 'outflow'}">${flow.amount} ${flow.flow_type === 'inflow' ? 'IN' : 'OUT'}</span>
                        </div>
                        <div class="whale-info">Total Reserve: ${flow.total_reserve} BTC</div>
                    </div>
                `).join('');
            }
        }
    } catch (error) {
        console.error('Error updating on-chain data:', error);
    }
}

function checkWhaleAlerts(data) {
    const summary = data.whale_summary || {};
    const netFlow = (summary.inflow || 0) - (summary.outflow || 0);
    const whaleAlertsList = document.getElementById('whaleAlertsList');
    
    if (!whaleAlertsList) return;
    
    let alerts = [];
    if (netFlow < -1000) {
        alerts.push('Major whale outflow detected - High selling pressure');
        addNotification('Whale Alert', 'Major outflow: ' + Math.abs(netFlow).toFixed(0) + ' - Caution advised');
    } else if (netFlow > 1000) {
        alerts.push('Large whale inflow detected - Buying accumulation');
        addNotification('Whale Alert', 'Major inflow: ' + netFlow.toFixed(0) + ' - Bullish signal');
    }
    
    if (alerts.length === 0) {
        whaleAlertsList.innerHTML = '<p class="empty-state">No whale activity detected</p>';
    } else {
        whaleAlertsList.innerHTML = alerts.map(alert => `
            <div class="whale-activity-item" style="border-left: 3px solid #ff3860;">
                <p style="margin: 0; color: #f0f6fc;">${alert}</p>
            </div>
        `).join('');
    }
}

function checkSignalNotifications(data) {
    if (data.predictions && data.predictions.length > 0) {
        const latestSignal = data.predictions[0];
        if (latestSignal.signal_type && latestSignal.bounce_probability > 60) {
            addNotification(
                'Strong Trading Signal',
                `${latestSignal.signal_type} signal detected at ${latestSignal.price} with ${latestSignal.bounce_probability.toFixed(1)}% probability`
            );
        }
    }
}

function addNotification(title, message) {
    const notification = {
        id: Date.now(),
        title: title,
        message: message,
        timestamp: new Date().toISOString(),
        read: false
    };
    
    notifications.push(notification);
    updateNotificationBadge();
}

function updateNotificationBadge() {
    const badge = document.getElementById('notificationBadge');
    const unreadCount = notifications.filter(n => !n.read).length;
    badge.textContent = unreadCount;
    badge.style.display = unreadCount > 0 ? 'inline-flex' : 'none';
}

function formatTimestamp(timestamp) {
    if (!timestamp) return 'N/A';
    try {
        const date = new Date(timestamp);
        const hours = String(date.getHours()).padStart(2, '0');
        const minutes = String(date.getMinutes()).padStart(2, '0');
        const seconds = String(date.getSeconds()).padStart(2, '0');
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        return `${month}/${day} ${hours}:${minutes}:${seconds}`;
    } catch (error) {
        return 'N/A';
    }
}

function updateMLPredictions(data) {
    if (data.status === 'error') {
        console.error('ML predictions error:', data.message);
        const container = document.getElementById('mlPredictionsList');
        if (container) {
            container.innerHTML = `<p class="empty-state">Error loading predictions: ${data.message}</p>`;
        }
        return;
    }

    mlPredictions = data.predictions || [];
    const container = document.getElementById('mlPredictionsList');
    const table = document.getElementById('mlSignalsTable');

    if (!container) {
        console.error('ML predictions container not found');
        return;
    }

    if (mlPredictions.length === 0) {
        container.innerHTML = '<p class="empty-state">No predictions available</p>';
        if (table) {
            table.innerHTML = '<tr><td colspan="6" class="empty-state">No signals</td></tr>';
        }
        return;
    }

    container.innerHTML = mlPredictions.slice(-5).reverse().map(pred => `
        <div class="prediction-item" style="padding: 12px; border-bottom: 1px solid #3a4556;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <div>
                    <span style="font-size: 13px; color: #00d4ff;">${pred.signal_type || 'N/A'}</span>
                    <span style="font-size: 11px; color: #8b949e; margin-left: 8px;">${formatTimestamp(pred.timestamp)}</span>
                </div>
                <span style="font-weight: 600; color: #00c853;">${pred.bounce_probability ? pred.bounce_probability.toFixed(1) : '--'}%</span>
            </div>
            <div style="font-size: 12px; color: #b0b8c8;">
                Price: ${pred.price ? parseFloat(pred.price).toFixed(2) : 'N/A'} | BB Position: ${pred.bb_position ? pred.bb_position.toFixed(3) : 'N/A'}
            </div>
        </div>
    `).join('');

    if (table) {
        table.innerHTML = mlPredictions.map(pred => `
            <tr>
                <td>${formatTimestamp(pred.timestamp)}</td>
                <td><span style="color: ${pred.signal_type && pred.signal_type.includes('lower') ? '#00c853' : '#ff3860'};">${pred.signal_type || 'N/A'}</span></td>
                <td>${pred.price ? parseFloat(pred.price).toFixed(2) : 'N/A'}</td>
                <td style="color: ${pred.bounce_probability > 60 ? '#00c853' : pred.bounce_probability > 40 ? '#ffc107' : '#ff3860'};">${pred.bounce_probability ? pred.bounce_probability.toFixed(1) : '--'}%</td>
                <td>${pred.bb_position ? pred.bb_position.toFixed(3) : 'N/A'}</td>
                <td>
                    <button class="btn btn-primary" style="padding: 5px 10px; font-size: 12px;" onclick="quickTrade('${pred.signal_type}')">Trade</button>
                </td>
            </tr>
        `).join('');
    }
}

function updateSentiment(data) {
    try {
        if (data.fear_greed_index && data.fear_greed_index.status === 'success') {
            const fgi = data.fear_greed_index.index_value;
            const fgiValue = document.getElementById('fgiValue');
            const fgiLabel = document.getElementById('fgiLabel');
            
            if (fgiValue) fgiValue.textContent = fgi;
            if (fgiLabel) fgiLabel.textContent = data.fear_greed_index.interpretation || 'N/A';
            
            if (fgiChart) {
                fgiChart.data.datasets[0].data = [100 - fgi, fgi];
                fgiChart.update('none');
            }
        }

        if (data.funding_rates && data.funding_rates.status === 'success') {
            const fundingRateElement = document.getElementById('fundingRate');
            if (fundingRateElement) {
                fundingRateElement.textContent = (data.funding_rates.current_funding_rate * 100).toFixed(3) + '%';
            }
        }
    } catch (error) {
        console.error('Error updating sentiment:', error);
    }
}

function updateAccountInfo(data) {
    if (data.status === 'error') {
        console.error('Account data error:', data.message);
        return;
    }

    try {
        const summary = data.summary || {};
        
        const accountBalance = document.getElementById('accountBalance');
        if (accountBalance) {
            accountBalance.textContent = (summary.total_balance || 0).toFixed(2);
        }
        
        const balanceChange = document.getElementById('balanceChange');
        if (balanceChange) {
            const changeValue = summary.balance_change || 0;
            const changePct = summary.balance_change_pct || 0;
            balanceChange.textContent = `${changeValue >= 0 ? '+' : ''}${changePct.toFixed(2)}%`;
            balanceChange.className = `stat-change ${changeValue >= 0 ? 'positive' : 'negative'}`;
        }
        
        const activePositions = document.getElementById('activePositions');
        if (activePositions) {
            activePositions.textContent = summary.active_positions || 0;
        }
        
        const unrealizedPnl = document.getElementById('unrealizedPnl');
        if (unrealizedPnl) {
            unrealizedPnl.textContent = (summary.unrealized_pnl || 0).toFixed(2);
        }
        
        const totalTrades = document.getElementById('totalTrades');
        if (totalTrades) {
            totalTrades.textContent = summary.total_trades || 0;
        }
        
        const winRate = document.getElementById('winRate');
        if (winRate) {
            winRate.textContent = `${(summary.win_rate || 0).toFixed(1)}%`;
        }

        const positions = data.positions || [];
        const posContainer = document.getElementById('activePositionsList');
        
        if (posContainer) {
            if (positions.length === 0) {
                posContainer.innerHTML = '<p class="empty-state">No active positions</p>';
            } else {
                posContainer.innerHTML = positions.map(pos => `
                    <div class="position-item">
                        <div class="position-info">
                            <h4>${pos.position_type} ${pos.quantity} ${pos.symbol}</h4>
                            <p>Entry: ${pos.entry_price.toFixed(2)} | Current: ${pos.current_price.toFixed(2)}</p>
                        </div>
                        <div class="position-pnl ${pos.unrealized_pnl >= 0 ? 'positive' : 'negative'}">
                            ${pos.unrealized_pnl >= 0 ? '+' : ''}${pos.unrealized_pnl.toFixed(2)}
                            <div style="font-size: 12px;">(${pos.pnl_percentage >= 0 ? '+' : ''}${pos.pnl_percentage.toFixed(2)}%)</div>
                        </div>
                        <button class="btn btn-secondary" style="margin-left: 15px; padding: 8px 12px; font-size: 12px;" onclick="closePositionUI('${pos.position_id}', ${pos.current_price})">Close</button>
                    </div>
                `).join('');
            }
        }

        const tradeHistory = data.trade_history || [];
        const historyTable = document.getElementById('tradeHistoryTable');
        
        if (historyTable) {
            if (tradeHistory.length === 0) {
                historyTable.innerHTML = '<tr><td colspan="8" class="empty-state">No trades yet</td></tr>';
            } else {
                historyTable.innerHTML = tradeHistory.map(trade => `
                    <tr>
                        <td>${formatTimestamp(trade.entry_time)}</td>
                        <td>${formatTimestamp(trade.close_time)}</td>
                        <td>${trade.position_type || 'N/A'}</td>
                        <td>${trade.entry_price ? parseFloat(trade.entry_price).toFixed(2) : 'N/A'}</td>
                        <td>${trade.close_price ? parseFloat(trade.close_price).toFixed(2) : 'N/A'}</td>
                        <td>${trade.quantity ? parseFloat(trade.quantity).toFixed(4) : 'N/A'}</td>
                        <td style="color: ${(trade.pnl || 0) >= 0 ? '#00c853' : '#ff3860'};">${(trade.pnl || 0) >= 0 ? '+' : ''}${(trade.pnl || 0).toFixed(2)}</td>
                        <td style="color: ${(trade.pnl_percentage || 0) >= 0 ? '#00c853' : '#ff3860'};">${(trade.pnl_percentage || 0) >= 0 ? '+' : ''}${(trade.pnl_percentage || 0).toFixed(2)}%</td>
                    </tr>
                `).join('');
            }
        }
    } catch (error) {
        console.error('Error updating account info:', error);
    }
}

async function openPosition(e) {
    e.preventDefault();
    
    const riskLevel = getOnChainRiskLevel();
    const shouldBlock = enableOnChainFilter && riskLevel === 'High';
    
    if (shouldBlock) {
        addNotification('Trade Blocked', 'High on-chain risk detected. Whale outflow warning active.');
        alert('Trade blocked: High whale outflow detected. Disable on-chain filter in settings to override.');
        return;
    }
    
    const formData = {
        symbol: currentSymbol,
        order_type: document.getElementById('orderType')?.value || 'BUY',
        quantity: parseFloat(document.getElementById('quantity')?.value || 0.1),
        entry_price: parseFloat(document.getElementById('entryPrice')?.value || 0),
        stop_loss: parseFloat(document.getElementById('stopLoss')?.value || 0),
        take_profit: parseFloat(document.getElementById('takeProfit')?.value || 0),
        ml_confidence: mlPredictions.length > 0 ? mlPredictions[0].bounce_probability : 0,
        on_chain_risk: riskLevel
    };

    try {
        const response = await fetch(`${API_BASE}/trading/open-position`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });
        const result = await response.json();
        
        if (result.status === 'success') {
            addNotification('Position Opened', 'Position opened successfully with on-chain confirmation: ' + result.message);
            document.getElementById('openPositionForm').reset();
            loadDashboardData();
        } else {
            addNotification('Error', 'Error opening position: ' + result.message);
        }
    } catch (error) {
        console.error('Error opening position:', error);
        addNotification('Error', 'Error opening position: ' + error.message);
    }
}

async function closePositionUI(positionId, currentPrice) {
    try {
        const response = await fetch(`${API_BASE}/trading/close-position`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                position_id: positionId,
                close_price: currentPrice
            })
        });
        const result = await response.json();
        
        if (result.status === 'success') {
            addNotification('Position Closed', 'Position closed successfully: ' + result.message);
            loadDashboardData();
        } else {
            addNotification('Error', 'Error closing position: ' + result.message);
        }
    } catch (error) {
        console.error('Error closing position:', error);
        addNotification('Error', 'Error closing position: ' + error.message);
    }
}

function quickTrade(signalType) {
    const orderTypeSelect = document.getElementById('orderType');
    if (orderTypeSelect) {
        const orderType = signalType && signalType.includes('lower') ? 'BUY' : 'SHORT';
        orderTypeSelect.value = orderType;
    }
    
    const latestSignal = mlPredictions.length > 0 ? mlPredictions[0] : null;
    if (latestSignal) {
        document.getElementById('signalConfidence').textContent = latestSignal.bounce_probability.toFixed(1) + '%';
    }
    
    const riskLevel = getOnChainRiskLevel();
    document.getElementById('riskLevel').textContent = riskLevel;
    document.getElementById('riskLevel').style.color = 
        riskLevel === 'High' ? '#ff3860' : riskLevel === 'Medium' ? '#ffc107' : '#00c853';
    
    const summary = onChainData.whale_summary || {};
    const netFlow = (summary.inflow || 0) - (summary.outflow || 0);
    document.getElementById('whaleActivityStatus').textContent = Math.abs(netFlow) > 500 ? 'Active' : 'Normal';
    
    switchSection('trading');
    document.getElementById('entryPrice')?.focus();
}
