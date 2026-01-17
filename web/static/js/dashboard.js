const API_BASE = 'http://localhost:5000/api';
let currentSymbol = 'BTCUSDT';
let priceChart = null;
let mlPredictions = [];

document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    loadDashboardData();
    setInterval(loadDashboardData, 10000);
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
            document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
        });
    });

    document.getElementById('symbolInput').addEventListener('change', (e) => {
        currentSymbol = e.target.value.toUpperCase();
        loadDashboardData();
    });

    document.getElementById('refreshBtn').addEventListener('click', loadDashboardData);

    document.getElementById('openPositionForm').addEventListener('submit', openPosition);
}

function switchSection(section) {
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    
    document.getElementById(`${section}-section`).classList.add('active');
    document.querySelector(`[data-section="${section}"]`).classList.add('active');
}

async function loadDashboardData() {
    try {
        const [priceData, predictions, sentiment, accountData] = await Promise.all([
            fetch(`${API_BASE}/price?symbol=${currentSymbol}`).then(r => r.json()),
            fetch(`${API_BASE}/ml-prediction?symbol=${currentSymbol}`).then(r => r.json()),
            fetch(`${API_BASE}/sentiment`).then(r => r.json()),
            fetch(`${API_BASE}/trading/account-summary`).then(r => r.json())
        ]);

        updatePriceChart(priceData);
        updateMLPredictions(predictions);
        updateSentiment(sentiment);
        updateAccountInfo(accountData);
    } catch (error) {
        console.error('Error loading dashboard data:', error);
    }
}

function updatePriceChart(data) {
    if (data.status === 'error') return;

    const ctx = document.getElementById('priceChart')?.getContext('2d');
    if (!ctx) return;

    const chartData = {
        labels: data.data.timestamps,
        datasets: [{
            label: currentSymbol,
            data: data.data.closes,
            borderColor: '#00d4ff',
            backgroundColor: 'rgba(0, 212, 255, 0.1)',
            fill: true,
            tension: 0.4,
            borderWidth: 2,
            pointRadius: 0,
            pointBackgroundColor: '#00d4ff'
        }]
    };

    if (priceChart) {
        priceChart.data = chartData;
        priceChart.update();
    } else {
        priceChart = new Chart(ctx, {
            type: 'line',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        },
                        ticks: {
                            color: '#b0b8c8',
                            font: {
                                size: 12
                            }
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        },
                        ticks: {
                            color: '#b0b8c8',
                            font: {
                                size: 12
                            },
                            maxTicksLimit: 10
                        }
                    }
                }
            }
        });
    }

    document.getElementById('currentPrice').textContent = `$${data.latest.price.toFixed(2)}`;
}

function updateMLPredictions(data) {
    if (data.status === 'error') return;

    mlPredictions = data.predictions || [];
    const container = document.getElementById('mlPredictionsList');
    const table = document.getElementById('mlSignalsTable');

    if (mlPredictions.length === 0) {
        container.innerHTML = '<p class="empty-state">No predictions available</p>';
        table.innerHTML = '<tr><td colspan="6" class="empty-state">No signals</td></tr>';
        return;
    }

    container.innerHTML = mlPredictions.slice(-5).reverse().map(pred => `
        <div class="prediction-item" style="padding: 12px; border-bottom: 1px solid #3a4556;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="font-size: 13px; color: #00d4ff;">${pred.signal_type}</span>
                <span style="font-weight: 600; color: #00c853;">${pred.bounce_probability.toFixed(1)}%</span>
            </div>
            <div style="font-size: 12px; color: #b0b8c8;">
                $${parseFloat(pred.price).toFixed(2)}
            </div>
        </div>
    `).join('');

    table.innerHTML = mlPredictions.map(pred => `
        <tr>
            <td>${pred.timestamp.slice(-8)}</td>
            <td><span style="color: ${pred.signal_type.includes('lower') ? '#00c853' : '#ff3860'};">${pred.signal_type}</span></td>
            <td>$${parseFloat(pred.price).toFixed(2)}</td>
            <td style="color: ${pred.bounce_probability > 60 ? '#00c853' : pred.bounce_probability > 40 ? '#ffc107' : '#ff3860'};">${pred.bounce_probability.toFixed(1)}%</td>
            <td>${pred.bb_position.toFixed(3)}</td>
            <td>
                <button class="btn btn-primary" style="padding: 5px 10px; font-size: 12px;" onclick="quickTrade('${pred.signal_type}')">Trade</button>
            </td>
        </tr>
    `).join('');
}

function updateSentiment(data) {
    if (data.fear_greed_index && data.fear_greed_index.status === 'success') {
        const fgi = data.fear_greed_index.index_value;
        document.getElementById('fgiValue').textContent = fgi;
        document.getElementById('fgiLabel').textContent = data.fear_greed_index.interpretation;
        
        const percentage = (fgi / 100) * 282;
        document.getElementById('fgiIndicator').style.strokeDashoffset = 282 - percentage;
    }

    if (data.funding_rates && data.funding_rates.status === 'success') {
        document.getElementById('fundingRate').textContent = (data.funding_rates.current_funding_rate * 100).toFixed(3) + '%';
    }
}

function updateAccountInfo(data) {
    if (data.status === 'error') return;

    const summary = data.summary;
    document.getElementById('accountBalance').textContent = `$${summary.total_balance.toFixed(2)}`;
    document.getElementById('balanceChange').textContent = `${summary.balance_change >= 0 ? '+' : ''}${summary.balance_change_pct.toFixed(2)}%`;
    document.getElementById('balanceChange').className = `stat-change ${summary.balance_change >= 0 ? 'positive' : 'negative'}`;
    
    document.getElementById('activePositions').textContent = summary.active_positions;
    document.getElementById('unrealizedPnl').textContent = `$${summary.unrealized_pnl.toFixed(2)}`;
    
    document.getElementById('totalTrades').textContent = summary.total_trades;
    document.getElementById('winRate').textContent = `${summary.win_rate.toFixed(1)}%`;

    const positions = data.positions || [];
    const posContainer = document.getElementById('activePositionsList');
    
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

    const tradeHistory = data.trade_history || [];
    const historyTable = document.getElementById('tradeHistoryTable');
    
    if (tradeHistory.length === 0) {
        historyTable.innerHTML = '<tr><td colspan="8" class="empty-state">No trades yet</td></tr>';
    } else {
        historyTable.innerHTML = tradeHistory.map(trade => `
            <tr>
                <td>${trade.entry_time.slice(-16)}</td>
                <td>${trade.close_time.slice(-16)}</td>
                <td>${trade.position_type}</td>
                <td>$${parseFloat(trade.entry_price).toFixed(2)}</td>
                <td>$${parseFloat(trade.close_price).toFixed(2)}</td>
                <td>${parseFloat(trade.quantity).toFixed(4)}</td>
                <td style="color: ${trade.pnl >= 0 ? '#00c853' : '#ff3860'};">${trade.pnl >= 0 ? '+' : ''}$${trade.pnl.toFixed(2)}</td>
                <td style="color: ${trade.pnl_percentage >= 0 ? '#00c853' : '#ff3860'};">${trade.pnl_percentage >= 0 ? '+' : ''}${trade.pnl_percentage.toFixed(2)}%</td>
            </tr>
        `).join('');
    }
}

async function openPosition(e) {
    e.preventDefault();
    
    const formData = {
        symbol: currentSymbol,
        order_type: document.getElementById('orderType').value,
        quantity: parseFloat(document.getElementById('quantity').value),
        entry_price: parseFloat(document.getElementById('entryPrice').value),
        stop_loss: parseFloat(document.getElementById('stopLoss').value),
        take_profit: parseFloat(document.getElementById('takeProfit').value)
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
    }
}

function quickTrade(signalType) {
    const orderType = signalType.includes('lower') ? 'BUY' : 'SHORT';
    document.getElementById('orderType').value = orderType;
    switchSection('trading');
    document.getElementById('entryPrice').focus();
}
