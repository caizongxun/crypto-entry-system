const API_BASE = 'http://localhost:5000/api';
let currentSymbol = 'BTCUSDT';
let tvChart = null;
let tvLiveChart = null;
let mlPredictions = [];
let currentTimeframe = '60';

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

    document.querySelectorAll('.tf-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const container = e.target.closest('.card-header');
            if (!container) return;
            
            container.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            
            currentTimeframe = e.target.dataset.tf;
            console.log('Timeframe changed to:', currentTimeframe);
        });
    });

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

async function loadDashboardData() {
    try {
        console.log('Loading dashboard data...');
        
        const promises = [
            fetch(`${API_BASE}/price?symbol=${currentSymbol}&interval=1h&limit=100`).then(r => r.json()).catch(err => ({ status: 'error', message: err.message })),
            fetch(`${API_BASE}/ml-prediction?symbol=${currentSymbol}`).then(r => r.json()).catch(err => ({ status: 'error', message: err.message })),
            fetch(`${API_BASE}/sentiment`).then(r => r.json()).catch(err => ({ status: 'error', message: err.message })),
            fetch(`${API_BASE}/trading/account-summary`).then(r => r.json()).catch(err => ({ status: 'error', message: err.message }))
        ];
        
        const [priceData, predictions, sentiment, accountData] = await Promise.all(promises);

        if (priceData.status === 'success') {
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
        const interval = 3600;

        for (let i = 0; i < data.data.closes.length; i++) {
            candleData.push({
                time: baseTime - (data.data.closes.length - i) * interval,
                open: parseFloat(data.data.opens[i]),
                high: parseFloat(data.data.highs[i]),
                low: parseFloat(data.data.lows[i]),
                close: parseFloat(data.data.closes[i])
            });
        }

        tvChart.candlestickSeries.setData(candleData);
        tvChart.timeScale().fitContent();
        console.log('TradingView chart updated successfully');
    } catch (err) {
        console.error('Failed to create TradingView chart:', err);
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
