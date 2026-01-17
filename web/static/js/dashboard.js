const API_BASE = 'http://localhost:5000/api';
let currentSymbol = 'BTCUSDT';
let tvChart = null;
let tvLiveChart = null;
let mlPredictions = [];
let currentTimeframe = '60';
let notifications = [];
let chartHeight = 400;
let refreshInterval = 15000;

const timeframeMap = {
    '1': { minutes: 1, interval: '1m', tvInterval: '1' },
    '3': { minutes: 3, interval: '3m', tvInterval: '3' },
    '5': { minutes: 5, interval: '5m', tvInterval: '5' },
    '15': { minutes: 15, interval: '15m', tvInterval: '15' },
    '30': { minutes: 30, interval: '30m', tvInterval: '30' },
    '60': { minutes: 60, interval: '1h', tvInterval: '60' },
    '240': { minutes: 240, interval: '4h', tvInterval: '240' },
    '1440': { minutes: 1440, interval: '1d', tvInterval: '1D' }
};

document.addEventListener('DOMContentLoaded', () => {
    console.log('Dashboard initialized');
    loadSettings();
    initializeChart();
    initializeLiveChart();
    initializeEventListeners();
    initializeResizeHandles();
    loadDashboardData();
    setInterval(loadDashboardData, refreshInterval);
});

function loadSettings() {
    const savedSettings = localStorage.getItem('dashboardSettings');
    if (savedSettings) {
        try {
            const settings = JSON.parse(savedSettings);
            currentTimeframe = settings.defaultTimeframe || '60';
            refreshInterval = (settings.refreshInterval || 15) * 1000;
            
            document.getElementById('themeSelect').value = settings.theme || 'dark';
            document.getElementById('defaultTimeframe').value = currentTimeframe;
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
    });

    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            document.body.style.userSelect = 'auto';
            document.body.style.cursor = 'default';
        }
    });
}

function initializeChart() {
    const container = document.getElementById('priceChart');
    
    const symbol = 'BINANCE:' + currentSymbol.replace('USDT', '') + 'USDT';
    const interval = timeframeMap[currentTimeframe]?.tvInterval || '60';
    
    tvChart = new TradingView.widget({
        autosize: false,
        width: '100%',
        height: chartHeight,
        symbol: symbol,
        interval: interval,
        timezone: 'Etc/UTC',
        theme: 'dark',
        style: '1',
        locale: 'en',
        toolbar_bg: '#1e2139',
        enable_publishing: false,
        allow_symbol_change: false,
        container_id: 'priceChart',
        hide_side_toolbar: false,
        hide_top_toolbar: false,
        withdateranges: true,
        studies: ['MAExp@tv-basicstudies', 'Volume@tv-basicstudies'],
        overrides: {
            'mainSeriesProperties.candleStyle.upColor': '#00c853',
            'mainSeriesProperties.candleStyle.downColor': '#ff3860',
            'mainSeriesProperties.candleStyle.borderUpColor': '#00c853',
            'mainSeriesProperties.candleStyle.borderDownColor': '#ff3860',
            'mainSeriesProperties.candleStyle.wickUpColor': '#00c853',
            'mainSeriesProperties.candleStyle.wickDownColor': '#ff3860',
            'volumePaneSize': 'small',
        }
    });
}

function initializeLiveChart() {
    const container = document.getElementById('liveChart');
    if (!container) return;
    
    const symbol = 'BINANCE:' + currentSymbol.replace('USDT', '') + 'USDT';
    
    tvLiveChart = new TradingView.widget({
        autosize: true,
        symbol: symbol,
        interval: '60',
        timezone: 'Etc/UTC',
        theme: 'dark',
        style: '1',
        locale: 'en',
        toolbar_bg: '#1e2139',
        enable_publishing: false,
        allow_symbol_change: true,
        container_id: 'liveChart',
        hide_side_toolbar: false,
        hide_top_toolbar: false,
        withdateranges: true,
        studies: [],
        overrides: {
            'mainSeriesProperties.candleStyle.upColor': '#00c853',
            'mainSeriesProperties.candleStyle.downColor': '#ff3860',
            'mainSeriesProperties.candleStyle.borderUpColor': '#00c853',
            'mainSeriesProperties.candleStyle.borderDownColor': '#ff3860',
            'mainSeriesProperties.candleStyle.wickUpColor': '#00c853',
            'mainSeriesProperties.candleStyle.wickDownColor': '#ff3860',
            'volumePaneSize': 'small',
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
            updateChart();
            updateLiveChart();
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
        enableNotifications: document.getElementById('enableNotifications').checked,
        refreshInterval: parseInt(document.getElementById('refreshInterval').value),
        apiKey: document.getElementById('apiKey').value,
        apiSecret: document.getElementById('apiSecret').value
    };

    localStorage.setItem('dashboardSettings', JSON.stringify(settings));
    
    currentTimeframe = settings.defaultTimeframe;
    refreshInterval = settings.refreshInterval * 1000;

    if (settings.theme === 'light') {
        document.documentElement.style.colorScheme = 'light';
    } else {
        document.documentElement.style.colorScheme = 'dark';
    }

    addNotification('Settings Saved', 'Your preferences have been saved successfully.');
    closeSettingsModal();
}

function updateChart() {
    if (tvChart) {
        tvChart.remove();
        tvChart = null;
    }
    initializeChart();
}

function updateLiveChart() {
    if (tvLiveChart) {
        tvLiveChart.remove();
        tvLiveChart = null;
    }
    initializeLiveChart();
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

async function loadDashboardData() {
    try {
        console.log('Loading dashboard data...');
        
        const promises = [
            fetch(`${API_BASE}/ml-prediction?symbol=${currentSymbol}`).then(r => r.json()).catch(err => ({ status: 'error', message: err.message })),
            fetch(`${API_BASE}/sentiment`).then(r => r.json()).catch(err => ({ status: 'error', message: err.message })),
            fetch(`${API_BASE}/trading/account-summary`).then(r => r.json()).catch(err => ({ status: 'error', message: err.message }))
        ];
        
        const [predictions, sentiment, accountData] = await Promise.all(promises);
        
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
    } catch (error) {
        console.error('Error loading dashboard data:', error);
    }
}

function checkSignalNotifications(data) {
    if (data.predictions && data.predictions.length > 0) {
        const latestSignal = data.predictions[0];
        if (latestSignal.signal_type && latestSignal.bounce_probability > 60) {
            addNotification(
                'Strong Trading Signal',
                `${latestSignal.signal_type} signal detected at $${latestSignal.price} with ${latestSignal.bounce_probability.toFixed(1)}% probability`
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
                <span style="font-size: 13px; color: #00d4ff;">${pred.signal_type || 'N/A'}</span>
                <span style="font-weight: 600; color: #00c853;">${pred.bounce_probability ? pred.bounce_probability.toFixed(1) : '--'}%</span>
            </div>
            <div style="font-size: 12px; color: #b0b8c8;">
                $${pred.price ? parseFloat(pred.price).toFixed(2) : 'N/A'}
            </div>
        </div>
    `).join('');

    if (table) {
        table.innerHTML = mlPredictions.map(pred => `
            <tr>
                <td>${pred.timestamp ? pred.timestamp.slice(-8) : 'N/A'}</td>
                <td><span style="color: ${pred.signal_type && pred.signal_type.includes('lower') ? '#00c853' : '#ff3860'};">${pred.signal_type || 'N/A'}</span></td>
                <td>$${pred.price ? parseFloat(pred.price).toFixed(2) : 'N/A'}</td>
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
            const fgiIndicator = document.getElementById('fgiIndicator');
            
            if (fgiValue) fgiValue.textContent = fgi;
            if (fgiLabel) fgiLabel.textContent = data.fear_greed_index.interpretation || 'N/A';
            if (fgiIndicator) {
                const percentage = (fgi / 100) * 282;
                fgiIndicator.style.strokeDashoffset = 282 - percentage;
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
            accountBalance.textContent = `$${(summary.total_balance || 0).toFixed(2)}`;
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
            unrealizedPnl.textContent = `$${(summary.unrealized_pnl || 0).toFixed(2)}`;
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
                            <p>Entry: $${pos.entry_price.toFixed(2)} | Current: $${pos.current_price.toFixed(2)}</p>
                        </div>
                        <div class="position-pnl ${pos.unrealized_pnl >= 0 ? 'positive' : 'negative'}">
                            ${pos.unrealized_pnl >= 0 ? '+' : ''}$${pos.unrealized_pnl.toFixed(2)}
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
                        <td>${trade.entry_time ? trade.entry_time.slice(-16) : 'N/A'}</td>
                        <td>${trade.close_time ? trade.close_time.slice(-16) : 'N/A'}</td>
                        <td>${trade.position_type || 'N/A'}</td>
                        <td>$${trade.entry_price ? parseFloat(trade.entry_price).toFixed(2) : 'N/A'}</td>
                        <td>$${trade.close_price ? parseFloat(trade.close_price).toFixed(2) : 'N/A'}</td>
                        <td>${trade.quantity ? parseFloat(trade.quantity).toFixed(4) : 'N/A'}</td>
                        <td style="color: ${(trade.pnl || 0) >= 0 ? '#00c853' : '#ff3860'};">${(trade.pnl || 0) >= 0 ? '+' : ''}$${(trade.pnl || 0).toFixed(2)}</td>
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
    
    const formData = {
        symbol: currentSymbol,
        order_type: document.getElementById('orderType')?.value || 'BUY',
        quantity: parseFloat(document.getElementById('quantity')?.value || 0.1),
        entry_price: parseFloat(document.getElementById('entryPrice')?.value || 0),
        stop_loss: parseFloat(document.getElementById('stopLoss')?.value || 0),
        take_profit: parseFloat(document.getElementById('takeProfit')?.value || 0)
    };

    try {
        const response = await fetch(`${API_BASE}/trading/open-position`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });
        const result = await response.json();
        
        if (result.status === 'success') {
            addNotification('Position Opened', 'Position opened successfully: ' + result.message);
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
    switchSection('trading');
    document.getElementById('entryPrice')?.focus();
}
