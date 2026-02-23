async function fetchData() {
    try {
        const response = await fetch('/api/data');
        if (!response.ok) throw new Error('Network response was not ok');
        const data = await response.json();
        updateUI(data);
    } catch (error) {
        console.error('Error fetching data:', error);
        document.getElementById('botStatusText').innerText = 'Connection Lost...';
        document.getElementById('botStatus').classList.remove('running');
        document.getElementById('botStatus').classList.add('stopped');
    }
}

function updateUI(data) {
    if (!data) return;

    // Header
    const botStatusText = document.getElementById('botStatusText');
    if (botStatusText) botStatusText.innerText = data.market_structure?.status || 'Active';

    document.getElementById('botStatus')?.classList.add('running');
    document.getElementById('activeMode').innerText = data.active_mode || 'SNIPER';

    // Safety split for last_update
    const lastUpdate = data.last_update || '';
    document.getElementById('lastHeartbeat').innerText = lastUpdate.includes(' ') ? lastUpdate.split(' ')[1] : lastUpdate;
    
    // Trading Day
    if (data.session_date) {
        document.getElementById('tradingDay').innerText = data.session_date;
    }

    // Stats
    document.getElementById('equity').innerText = `₹${(data.equity || 0).toLocaleString()}`;
    const target = data.target || 30000;
    const targetProgress = (((data.equity || 0) / target) * 100).toFixed(1);
    document.getElementById('pnlProgress').innerText = `${targetProgress}% of target (₹${target.toLocaleString()})`;

    const totalPnl = data.pnl || 0;
    const pnlEl = document.getElementById('totalPnl');
    if (pnlEl) {
        pnlEl.innerText = `₹${totalPnl.toLocaleString()}`;
        pnlEl.className = totalPnl >= 0 ? 'stat-value pnl-pos' : 'stat-value pnl-neg';
    }

    const pnlPctEl = document.getElementById('pnlPct');
    if (pnlPctEl) {
        pnlPctEl.innerText = `${totalPnl >= 0 ? '+' : ''}${data.pnl_pct || 0}%`;
        pnlPctEl.className = totalPnl >= 0 ? 'stat-sub pnl-pos' : 'stat-sub pnl-neg';
    }

    document.getElementById('winRate').innerText = `${data.win_rate || 0}%`;
    document.getElementById('tradeCount').innerText = `${data.total_trades || 0} Trades`;

    // --- DAILY SESSION STATS ---
    const sessionPnlEl = document.getElementById('sessionPnl');
    if (sessionPnlEl) {
        sessionPnlEl.innerText = `₹${(data.daily_pnl || 0).toLocaleString()}`;
        sessionPnlEl.className = (data.daily_pnl || 0) >= 0 ? 'value pnl-pos' : 'value pnl-neg';
    }
    document.getElementById('sessionTrades').innerText = data.daily_trades || 0;
    document.getElementById('sessionWinLoss').innerText = `${data.daily_wins || 0} / ${data.daily_losses || 0}`;
    document.getElementById('sessionStart').innerText = data.session_start_time || '--:--';

    // Positions
    const posContainer = document.getElementById('positionsContainer');
    const positions = data.positions || [];
    document.getElementById('positionCount').innerText = positions.length;

    if (positions.length > 0) {
        posContainer.innerHTML = positions.map(pos => `
            <div class="position-item">
                <div class="pos-symbol">${pos.symbol}</div>
                <div class="pos-side ${(pos.side || '').toLowerCase()}">${(pos.side || '').toUpperCase()} ${pos.leverage || ''}x</div>
                <div class="pos-entry">Entry: ${pos.entry}</div>
                <div class="pos-pnl ${(pos.unrealized_pnl || 0) >= 0 ? 'pnl-pos' : 'pnl-neg'}">
                    ${(pos.unrealized_pnl || 0) >= 0 ? '+' : ''}${pos.unrealized_pnl || 0}
                </div>
            </div>
        `).join('');
    } else {
        posContainer.innerHTML = '<div class="empty-state">No active positions. Scanning... 🔍</div>';
    }

    // Trade History
    const historyBody = document.getElementById('tradeHistoryBody');
    const recentTrades = data.recent_trades || [];
    if (recentTrades.length > 0) {
        historyBody.innerHTML = recentTrades.slice().reverse().map(trade => {
            const roi = trade.roi !== undefined ? trade.roi : (trade.roe !== undefined ? trade.roe : '--');
            const pnlInr = trade.pnl_inr !== undefined ? trade.pnl_inr.toLocaleString() : '--';
            const time = trade.time ? (trade.time.includes(' ') ? trade.time.split(' ')[1] : trade.time) : '--';

            return `
                <tr>
                    <td>${time}</td>
                    <td class="pos-symbol">${trade.symbol}</td>
                    <td class="pos-side ${(trade.side || '').toLowerCase()}">${(trade.side || '').toUpperCase()}</td>
                    <td>${trade.exit_price}</td>
                    <td class="${roi >= 0 || roi === '--' ? 'pnl-pos' : 'pnl-neg'}">${roi >= 0 ? '+' : ''}${roi}${roi !== '--' ? '%' : ''}</td>
                    <td class="${trade.pnl_inr >= 0 || trade.pnl_inr === undefined ? 'pnl-pos' : 'pnl-neg'}">₹${pnlInr}</td>
                    <td><span class="badge">${trade.reason}</span></td>
                </tr>
            `;
        }).join('');
    } else {
        historyBody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding: 20px; opacity: 0.5;">No trade history available yet.</td></tr>';
    }

    // Logs
    const logsContainer = document.getElementById('logsContainer');
    const recentLogs = data.recent_logs || [];
    if (recentLogs.length > 0) {
        logsContainer.innerHTML = recentLogs.slice().reverse().map(log => `
            <div class="log-entry">${log}</div>
        `).join('');
    }

    // Analysis
    const analysisContainer = document.getElementById('analysisContainer');
    if (data.market_structure) {
        const symbols = Object.keys(data.market_structure).filter(k => k !== 'status' && k !== '_bot_heartbeat');
        analysisContainer.innerHTML = symbols.map(sym => {
            const s = data.market_structure[sym];
            const bias = s.market_bias || 'neutral';
            return `
                <div class="analysis-item">
                    <span><strong>${sym}</strong>: ${s.ut_signal || 'Wait'}</span>
                    <span class="${bias === 'bullish' ? 'pnl-pos' : (bias === 'bearish' ? 'pnl-neg' : '')}">${bias.toUpperCase()}</span>
                </div>
            `;
        }).join('');
    }
}

async function fetchSessionHistory() {
    try {
        const response = await fetch('/api/session_history');
        if (!response.ok) throw new Error('History fetch failed');
        const history = await response.json();
        const container = document.getElementById('historyContainer');
        
        if (history.length > 0) {
            container.innerHTML = history.map(session => `
                <div class="history-item">
                    <div class="history-date">📅 ${session.date}</div>
                    <div class="history-data">Net PnL: <span class="history-pnl ${session.net_pnl_inr >= 0 ? 'pnl-pos' : 'pnl-neg'}">₹${session.net_pnl_inr.toLocaleString()}</span></div>
                    <div class="history-data">Trades: <span>${session.total_trades}</span></div>
                    <div class="history-data">Win/Loss: <span>${session.wins} / ${session.losses}</span></div>
                    <div class="history-data">Win Rate: <span>${session.win_rate}%</span></div>
                </div>
            `).join('');
        } else {
            container.innerHTML = '<div class="empty-state">No past sessions recorded yet.</div>';
        }
    } catch (error) {
        console.error('Error fetching history:', error);
    }
}

// Initial fetch and start interval
fetchData();
fetchSessionHistory();
setInterval(fetchData, 1000);
setInterval(fetchSessionHistory, 30000); // Check history every 30s

// Export CSV handler
document.getElementById('exportCsv')?.addEventListener('click', () => {
    alert('Exporting to paper_trades folder on server...');
});
