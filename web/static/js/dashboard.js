const API_BASE = 'http://localhost:5000/api';
let currentSymbol = 'BTCUSDT';
let tvChart = null;
let tvLiveChart = null;
let mlPredictions = [];
let currentTimeframe = '60';
let activeIndicators = {};
let chartData = { opens: [], highs: [], lows: [], closes: [] };

const timeframeMap = {
    '1': { minutes: 1, interval: '1m' },
    '5': { minutes: 5, interval: '5m' },
    '60': { minutes: 60, interval: '1h' },
    '240': { minutes: 240, interval: '4h' },
    '1440': { minutes: 1440, interval: '1d' }
};

// Technical Indicator Calculations
const indicators = {
    sma: (prices, period = 20) => {
        const result = [];
        for (let i = period - 1; i < prices.length; i++) {
            const sum = prices.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
            result.push(sum / period);
        }
        return result;
    },
    
    ema: (prices, period = 20) => {
        const result = [];
        const multiplier = 2 / (period + 1);
        let ema = prices.slice(0, period).reduce((a, b) => a + b, 0) / period;
        result.push(ema);
        
        for (let i = period; i < prices.length; i++) {
            ema = (prices[i] - ema) * multiplier + ema;
            result.push(ema);
        }
        return result;
    },
    
    bollinger: (prices, period = 20, stdDev = 2) => {
        const sma = indicators.sma(prices, period);
        const result = { upper: [], middle: sma, lower: [] };
        
        for (let i = period - 1; i < prices.length; i++) {
            const values = prices.slice(i - period + 1, i + 1);
            const mean = values.reduce((a, b) => a + b, 0) / period;
            const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / period;
            const std = Math.sqrt(variance);
            result.upper.push(mean + std * stdDev);
            result.lower.push(mean - std * stdDev);
        }
        return result;
    },
    
    rsi: (prices, period = 14) => {
        const result = [];
        const changes = [];
        
        for (let i = 1; i < prices.length; i++) {
            changes.push(prices[i] - prices[i - 1]);
        }
        
        let avgGain = 0, avgLoss = 0;
        for (let i = 0; i < period; i++) {
            avgGain += Math.max(changes[i], 0);
            avgLoss += Math.abs(Math.min(changes[i], 0));
        }
        avgGain /= period;
        avgLoss /= period;
        
        for (let i = period; i < changes.length; i++) {
            const gain = Math.max(changes[i], 0);
            const loss = Math.abs(Math.min(changes[i], 0));
            avgGain = (avgGain * (period - 1) + gain) / period;
            avgLoss = (avgLoss * (period - 1) + loss) / period;
            const rs = avgGain / (avgLoss || 1);
            const rsi = 100 - (100 / (1 + rs));
            result.push(rsi);
        }
        return result;
    },
    
    macd: (prices, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) => {
        const fastEma = indicators.ema(prices, fastPeriod);
        const slowEma = indicators.ema(prices, slowPeriod);
        const macdLine = [];
        
        for (let i = 0; i < Math.min(fastEma.length, slowEma.length); i++) {
            macdLine.push(fastEma[i] - slowEma[i]);
        }
        
        const signalLine = indicators.ema(macdLine, signalPeriod);
        return { macdLine, signalLine };
    }
};

document.addEventListener('DOMContentLoaded', () => {
    console.log('Dashboard initialized');
    initializeEventListeners();
    loadDashboardData();
    setInterval(loadDashboardData, 15000);
});

function initializeEventListeners() {
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const section = item.dataset.section;
            switchSection(section);
        });
    });

    // Timeframe buttons
    document.querySelectorAll('.tf-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const container = e.target.closest('.card-header');
            if (!container) return;
            
            container.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            
            const newTimeframe = e.target.dataset.tf;
            if (newTimeframe !== currentTimeframe) {
                currentTimeframe = newTimeframe;
                console.log('Timeframe changed to:', currentTimeframe);
                loadDashboardData();
            }
        });
    });

    // Indicator buttons - Dashboard
    const dashboardIndicators = document.getElementById('dashboardIndicators');
    if (dashboardIndicators) {
        dashboardIndicators.querySelectorAll('.indicator-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const indicator = e.target.dataset.indicator;
                e.target.classList.toggle('active');
                activeIndicators[indicator] = e.target.classList.contains('active');
                console.log('Indicator toggled:', indicator, activeIndicators[indicator]);
                updateChartIndicators();
            });
        });
    }

    // Indicator buttons - Market
    const marketIndicators = document.getElementById('marketIndicators');
    if (marketIndicators) {
        marketIndicators.querySelectorAll('.indicator-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const indicator = e.target.dataset.indicator;
                e.target.classList.toggle('active');
                activeIndicators[indicator] = e.target.classList.contains('active');
                console.log('Indicator toggled:', indicator, activeIndicators[indicator]);
                updateChartIndicators();
            });
        });
    }

    const symbolInput = document.getElementById('symbolInput');
    if (symbolInput) {
        symbolInput.addEventListener('change', (e) => {
            currentSymbol = e.target.value.toUpperCase();
            if (!currentSymbol.endsWith('USDT')) {
                currentSymbol = currentSymbol + 'USDT';
            }
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
    
    setTimeout(() => {
        if (section === 'market' && tvLiveChart) {
            tvLiveChart.applyOptions({ width: document.getElementById('liveChart')?.clientWidth || 800 });
        }
    }, 100);
}

function updateChartIndicators() {
    if (!tvChart || !chartData.closes.length) return;
    
    const closes = chartData.closes.map(v => parseFloat(v));
    const baseLine = closes[closes.length - 1];
    const baseTime = Math.floor(Date.now() / 1000);
    const timeframeConfig = timeframeMap[currentTimeframe];
    const intervalSeconds = (timeframeConfig?.minutes || 60) * 60;
    
    // Clear existing series except candlestick
    if (tvChart.series && tvChart.series.length > 1) {
        for (let i = tvChart.series.length - 1; i >= 1; i--) {
            tvChart.removeSeries(tvChart.series[i]);
        }
    }
    
    // Add SMA
    if (activeIndicators.sma) {
        const smaValues = indicators.sma(closes, 20);
        const smaData = smaValues.map((val, idx) => ({
            time: baseTime - (closes.length - (idx + closes.length - smaValues.length)) * intervalSeconds,
            value: val
        }));
        const smaSeries = tvChart.addLineSeries({ color: '#FFD700', lineWidth: 1 });
        smaSeries.setData(smaData);
    }
    
    // Add EMA
    if (activeIndicators.ema) {
        const emaValues = indicators.ema(closes, 20);
        const emaData = emaValues.map((val, idx) => ({
            time: baseTime - (closes.length - (idx + closes.length - emaValues.length)) * intervalSeconds,
            value: val
        }));
        const emaSeries = tvChart.addLineSeries({ color: '#FF6B9D', lineWidth: 1 });
        emaSeries.setData(emaData);
    }
    
    // Add Bollinger Bands
    if (activeIndicators.bb) {
        const bbValues = indicators.bollinger(closes, 20, 2);
        const bbUpper = bbValues.upper.map((val, idx) => ({
            time: baseTime - (closes.length - (idx + closes.length - bbValues.upper.length)) * intervalSeconds,
            value: val
        }));
        const bbLower = bbValues.lower.map((val, idx) => ({
            time: baseTime - (closes.length - (idx + closes.length - bbValues.lower.length)) * intervalSeconds,
            value: val
        }));
        
        const bbUpperSeries = tvChart.addLineSeries({ color: '#00FF00', lineWidth: 1, lineStyle: 2 });
        const bbLowerSeries = tvChart.addLineSeries({ color: '#00FF00', lineWidth: 1, lineStyle: 2 });
        bbUpperSeries.setData(bbUpper);
        bbLowerSeries.setData(bbLower);
    }
    
    console.log('Chart indicators updated:', activeIndicators);
}

async function loadDashboardData() {
    try {
        console.log('Loading dashboard data with timeframe:', currentTimeframe);
        
        const interval = timeframeMap[currentTimeframe]?.interval || '1h';
        const limit = currentTimeframe === '1' ? 200 : currentTimeframe === '5' ? 150 : 100;
        
        const promises = [
            fetch(`${API_BASE}/price?symbol=${currentSymbol}&interval=${interval}&limit=${limit}`).then(r => r.json()).catch(err => ({ status: 'error', message: err.message })),
            fetch(`${API_BASE}/ml-prediction?symbol=${currentSymbol}`).then(r => r.json()).catch(err => ({ status: 'error', message: err.message })),
            fetch(`${API_BASE}/sentiment`).then(r => r.json()).catch(err => ({ status: 'error', message: err.message })),
            fetch(`${API_BASE}/trading/account-summary`).then(r => r.json()).catch(err => ({ status: 'error', message: err.message }))
        ];
        
        const [priceData, predictions, sentiment, accountData] = await Promise.all(promises);

        if (priceData.status === 'success') {
            chartData = priceData.data;
            updatePriceChart(priceData);
        } else {
            console.error('Price data error:', priceData.message);
        }
        
        if (predictions.status === 'success') {
            updateMLPredictions(predictions);
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

function updatePriceChart(data) {
    if (data.status === 'error') {
        console.error('Price data error:', data.message);
        return;
    }

    const container = document.getElementById('priceChart');
    if (!container) {
        console.error('Chart container not found');
        return;
    }

    if (!data.data || !data.data.closes) {
        console.error('Invalid price data structure');
        return;
    }

    try {
        const LWC = window.LightweightCharts;
        if (!LWC) {
            console.error('LightweightCharts library not loaded');
            return;
        }

        if (!tvChart) {
            tvChart = LWC.createChart(container, {
                layout: {
                    textColor: '#b0b8c8',
                    background: { color: '#1e2139' }
                },
                width: container.clientWidth,
                height: 400,
                timeScale: {
                    timeVisible: true,
                    secondsVisible: false
                },
                localization: {
                    priceFormatter: price => '$' + price.toFixed(2)
                }
            });

            const candlestickSeries = tvChart.addCandlestickSeries({
                upColor: '#00c853',
                downColor: '#ff3860',
                borderVisible: false,
                wickUpColor: '#00c853',
                wickDownColor: '#ff3860'
            });

            tvChart.candlestickSeries = candlestickSeries;
        }

        const candleData = [];
        const baseTime = Math.floor(Date.now() / 1000);
        const timeframeConfig = timeframeMap[currentTimeframe];
        const intervalSeconds = (timeframeConfig?.minutes || 60) * 60;

        for (let i = 0; i < data.data.closes.length; i++) {
            candleData.push({
                time: baseTime - (data.data.closes.length - i) * intervalSeconds,
                open: parseFloat(data.data.opens[i]),
                high: parseFloat(data.data.highs[i]),
                low: parseFloat(data.data.lows[i]),
                close: parseFloat(data.data.closes[i])
            });
        }

        tvChart.candlestickSeries.setData(candleData);
        tvChart.timeScale().fitContent();
        
        // Update indicators when chart updates
        updateChartIndicators();
        
        console.log('TradingView chart updated with', candleData.length, 'candles');
    } catch (err) {
        console.error('Failed to update TradingView chart:', err);
    }

    const currentPriceElement = document.getElementById('currentPrice');
    if (currentPriceElement && data.latest) {
        currentPriceElement.textContent = `$${data.latest.price.toFixed(2)}`;
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
            alert('Position opened: ' + result.message);
            document.getElementById('openPositionForm').reset();
            loadDashboardData();
        } else {
            alert('Error: ' + result.message);
        }
    } catch (error) {
        console.error('Error opening position:', error);
        alert('Error opening position: ' + error.message);
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
            alert('Position closed: ' + result.message);
            loadDashboardData();
        } else {
            alert('Error: ' + result.message);
        }
    } catch (error) {
        console.error('Error closing position:', error);
        alert('Error closing position: ' + error.message);
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
