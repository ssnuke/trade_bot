// Multi-Bot Dashboard Manager
class DashboardManager {
    constructor() {
        this.currentBotId = null;
        this.allBots = [];
        this.botInstances = {};
        this.refreshInterval = null;
        this.init();
    }

    async init() {
        await this.loadBots();
        this.setupEventListeners();
        this.startRefresh();
    }

    async loadBots() {
        try {
            const response = await fetch('/api/bots');
            if (!response.ok) {
                // Fallback to single bot mode
                this.allBots = [{ id: 'default', name: 'Bot', enabled: true }];
            } else {
                this.allBots = await response.json();
            }
            
            this.renderBotTabs();
            
            // Load first enabled bot or default
            const firstBot = this.allBots.find(b => b.enabled) || this.allBots[0];
            if (firstBot) {
                this.switchBot(firstBot.id);
            }
        } catch (error) {
            console.error('Error loading bots:', error);
            // Single bot fallback
            this.allBots = [{ id: 'default', name: 'Bot', enabled: true }];
            this.renderBotTabs();
        }
    }

    renderBotTabs() {
        const tabsContainer = document.getElementById('botTabs');
        if (!tabsContainer) return;

        tabsContainer.innerHTML = this.allBots.map(bot => `
            <button class="bot-tab ${bot.id === this.currentBotId ? 'active' : ''}" 
                    data-bot-id="${bot.id}"
                    title="${bot.name}">
                ${bot.name}
            </button>
        `).join('');

        // Add click handlers
        document.querySelectorAll('.bot-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.switchBot(e.target.getAttribute('data-bot-id'));
            });
        });
    }

    async switchBot(botId) {
        this.currentBotId = botId;
        
        // Update active tab
        document.querySelectorAll('.bot-tab').forEach(tab => {
            tab.classList.toggle('active', tab.getAttribute('data-bot-id') === botId);
        });

        // Fetch and display bot data
        await this.fetchBotData(botId);
        await this.fetchSessionHistory(botId);
    }

    async fetchBotData(botId = null) {
        botId = botId || this.currentBotId || 'default';
        
        try {
            let url = botId === 'default' ? '/api/data' : `/api/bots/${botId}/data`;
            const response = await fetch(url);
            if (!response.ok) throw new Error('Fetch failed');
            
            const data = await response.json();
            this.updateUI(data, botId);
        } catch (error) {
            console.error(`Error fetching data for bot ${botId}:`, error);
            document.getElementById('botStatusText').innerText = 'Connection Lost...';
        }
    }

    updateUI(data) {
        if (!data) return;

        // Header
        const botStatusText = document.getElementById('botStatusText');
        if (botStatusText) botStatusText.innerText = data.market_structure?.status || 'Active';

        document.getElementById('botStatus')?.classList.add('running');
        document.getElementById('activeMode').innerText = data.active_mode || 'SNIPER';

        //Heartbeat
        const lastUpdate = data.last_update || '';
        document.getElementById('lastHeartbeat').innerText = lastUpdate.includes(' ') ? lastUpdate.split(' ')[1] : lastUpdate;
        
        // Trading Day
        if (data.session_date) {
            document.getElementById('tradingDay').innerText = data.session_date;
        }

        // Stats
        document.getElementById('equity').innerText = `₹${(data.equity || 0).toLocaleString()}`;
        document.getElementById('startingCapitalDisplay').innerText = data.starting_capital || 5000;
        
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

        // Daily Session Stats
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

    async fetchSessionHistory(botId = null) {
        botId = botId || this.currentBotId || 'default';
        
        try {
            let url = botId === 'default' ? '/api/session_history' : `/api/bots/${botId}/db?table=sessions`;
            const response = await fetch(url);
            if (!response.ok) throw new Error('History fetch failed');
            
            let history = await response.json();
            if (botId !== 'default') history = history.records || [];
            
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

    async openDatabaseViewer() {
        const modal = document.getElementById('dbModal');
        if (!modal) return;
        
        modal.classList.add('active');
        await this.loadDatabaseRecords();
    }

    async loadDatabaseRecords() {
        const botId = this.currentBotId || 'default';
        const table = document.getElementById('dbTable').value || 'trades';
        const limit = document.getElementById('dbLimit').value || 100;
        
        try {
            const url = botId === 'default' 
                ? `/api/session_history?table=${table}`
                : `/api/bots/${botId}/db?table=${table}&limit=${limit}`;
            
            const response = await fetch(url);
            if (!response.ok) throw new Error('DB fetch failed');
            
            const result = await response.json();
            const records = result.records || result;
            
            this.renderDatabaseTable(records, table);
        } catch (error) {
            console.error('Error loading database:', error);
            document.getElementById('dbTableBody').innerHTML = '<tr><td colspan="10">Error loading data</td></tr>';
        }
    }

    renderDatabaseTable(records, table) {
        const thead = document.getElementById('dbTableHead');
        const tbody = document.getElementById('dbTableBody');
        
        if (!records || records.length === 0) {
            thead.innerHTML = '';
            tbody.innerHTML = '<tr><td colspan="10" style="text-align: center;">No records found</td></tr>';
            return;
        }

        // Get columns from first record
        const columns = Object.keys(records[0]);
        
        thead.innerHTML = `<tr>${columns.map(col => `<th>${col}</th>`).join('')}</tr>`;
        
        tbody.innerHTML = records.map(record => `
            <tr>${columns.map(col => {
                const val = record[col];
                let displayVal = val;
                if (typeof val === 'number') {
                    displayVal = val.toLocaleString();
                }
                return `<td>${displayVal}</td>`;
            }).join('')}</tr>
        `).join('');
    }

    async performBotAction(action) {
        const botId = this.currentBotId || 'default';
        if (botId === 'default') {
            // Use old endpoints for single bot mode
            const endpoint = action === 'reset' ? '/reset' : '/reset_capital';
            try {
                const response = await fetch(endpoint, { method: 'POST' });
                const data = await response.json();
                alert(data.message || 'Action completed');
                this.fetchBotData();
            } catch (error) {
                alert('Error: ' + error.message);
            }
        } else {
            // Use new multi-bot endpoints
            const endpoint = `/api/bots/${botId}/${action}`;
            try {
                const response = await fetch(endpoint, { method: 'POST' });
                const data = await response.json();
                alert(data.message || 'Action completed');
                this.fetchBotData();
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
    }

    async addNewBot() {
        // Get form values
        const botName = document.getElementById('botName').value;
        const startingCapital = document.getElementById('botStartingCapital').value;
        const rsiPeriod = document.getElementById('rsiPeriod').value;
        const rsiOversold = document.getElementById('rsiOversold').value;
        const rsiOverbought = document.getElementById('rsiOverbought').value;
        const macdFast = document.getElementById('macdFast').value;
        const macdSlow = document.getElementById('macdSlow').value;
        const macdSignal = document.getElementById('macdSignal').value;
        const notes = document.getElementById('botNotes').value;

        // Validate
        if (!botName) {
            alert('Please enter a bot name');
            return;
        }

        // Build request
        const botData = {
            name: botName,
            starting_capital: parseFloat(startingCapital),
            rsi_period: parseInt(rsiPeriod),
            rsi_oversold: parseInt(rsiOversold),
            rsi_overbought: parseInt(rsiOverbought),
            macd_fast: parseInt(macdFast),
            macd_slow: parseInt(macdSlow),
            macd_signal: parseInt(macdSignal),
            notes: notes
        };

        try {
            const response = await fetch('/api/bots', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(botData)
            });

            const result = await response.json();
            
            if (response.ok) {
                alert(result.message || 'Bot created successfully!');
                document.getElementById('addBotModal').classList.remove('active');
                
                // Clear form
                document.getElementById('botName').value = '';
                document.getElementById('botNotes').value = '';
                
                // Reload bots list
                await this.loadBots();
            } else {
                alert('Error: ' + (result.error || 'Failed to create bot'));
            }
        } catch (error) {
            alert('Error: ' + error.message);
        }
    }

    async deleteCurrentBot() {
        const botId = this.currentBotId;
        
        if (!botId || botId === 'default') {
            alert('Cannot delete the default bot');
            return;
        }

        const botConfig = this.allBots.find(b => b.id === botId);
        const botName = botConfig ? botConfig.name : botId;

        if (!confirm(`Are you sure you want to delete "${botName}"?\n\nThis will remove the bot configuration. The bot will stop running after restart.`)) {
            return;
        }

        try {
            const response = await fetch(`/api/bots/${botId}`, {
                method: 'DELETE'
            });

            const result = await response.json();
            
            if (response.ok) {
                alert(result.message || 'Bot deleted successfully!');
                
                // Reload bots list and switch to first available
                await this.loadBots();
            } else {
                alert('Error: ' + (result.error || 'Failed to delete bot'));
            }
        } catch (error) {
            alert('Error: ' + error.message);
        }
    }

    async updateCapital() {
        const botId = this.currentBotId || 'default';
        const amount = parseFloat(document.getElementById('newCapital').value);
        
        if (!amount || amount <= 0) {
            alert('Please enter a valid amount');
            return;
        }

        if (botId === 'default') {
            alert('Capital update not available for single-bot mode');
            return;
        }

        try {
            const response = await fetch(`/api/bots/${botId}/set_capital`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ amount })
            });
            const data = await response.json();
            alert(data.message || 'Capital updated');
            document.getElementById('newCapital').value = '';
            this.fetchBotData();
        } catch (error) {
            alert('Error: ' + error.message);
        }
    }

    startRefresh() {
        // Start auto-refresh
        this.refreshInterval = setInterval(() => {
            this.fetchBotData();
            this.fetchSessionHistory();
        }, 1000);
    }

    stopRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
    }

    setupEventListeners() {
        // Capital button
        document.getElementById('capitalBtn')?.addEventListener('click', () => {
            document.getElementById('capitalModal').classList.add('active');
        });

        // Reset button
        document.getElementById('resetBtn')?.addEventListener('click', () => {
            if (confirm('Reset bot trading limits?')) {
                this.performBotAction('reset');
            }
        });

        // Add Bot button
        document.getElementById('addBotBtn')?.addEventListener('click', () => {
            document.getElementById('addBotModal').classList.add('active');
        });

        // Delete Bot button
        document.getElementById('deleteBotBtn')?.addEventListener('click', () => {
            this.deleteCurrentBot();
        });

        // Add Bot Modal
        document.getElementById('addBotModalClose')?.addEventListener('click', () => {
            document.getElementById('addBotModal').classList.remove('active');
        });
        document.getElementById('createBotBtn')?.addEventListener('click', () => {
            this.addNewBot();
        });

        // Database modal
        document.getElementById('viewDbBtn')?.addEventListener('click', () => {
            this.openDatabaseViewer();
        });
        document.getElementById('dbModalClose')?.addEventListener('click', () => {
            document.getElementById('dbModal').classList.remove('active');
        });
        document.getElementById('dbRefreshBtn')?.addEventListener('click', () => {
            this.loadDatabaseRecords();
        });
        document.getElementById('dbTable')?.addEventListener('change', () => {
            this.loadDatabaseRecords();
        });

        // Capital modal
        document.getElementById('capitalModalClose')?.addEventListener('click', () => {
            document.getElementById('capitalModal').classList.remove('active');
        });
        document.getElementById('resetCapitalBtn')?.addEventListener('click', () => {
            if (confirm('Reset capital to starting amount?')) {
                this.performBotAction('reset_capital');
            }
        });
        document.getElementById('updateCapitalBtn')?.addEventListener('click', () => {
            this.updateCapital();
        });

        // Close modal on background click
        document.querySelectorAll('.modal').forEach(modal => {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    modal.classList.remove('active');
                }
            });
        });
    }
}

// Initialize dashboard on page load
let dashboard;
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new DashboardManager();
});

// Fallback buttons (for backward compatibility)
document.getElementById('exportCsv')?.addEventListener('click', () => {
    alert('Exporting to paper_trades folder on server...');
});

