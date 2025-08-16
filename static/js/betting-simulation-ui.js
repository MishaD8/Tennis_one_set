/**
 * Betting Simulation UI Components
 * User interface for betting simulation and performance visualization
 */

class BettingSimulationUI {
    constructor(container, options = {}) {
        this.container = typeof container === 'string' ? document.getElementById(container) : container;
        this.simulation = new BettingSimulation(options);
        this.chartInstances = new Map();
        
        this.init();
    }

    init() {
        this.render();
        this.bindEvents();
        
        // Generate sample data for demonstration
        this.simulation.generateSampleData(50);
        this.updateDisplay();
    }

    render() {
        this.container.innerHTML = `
            <div class="betting-simulation-dashboard">
                <!-- Controls Section -->
                <div class="simulation-controls">
                    <h2>📊 Betting Simulation Dashboard</h2>
                    
                    <div class="controls-grid">
                        <div class="control-group">
                            <label for="initial-capital">Initial Capital ($)</label>
                            <input type="number" id="initial-capital" value="${this.simulation.initialCapital}" min="10" max="10000" step="10">
                        </div>
                        
                        <div class="control-group">
                            <label for="bankroll-strategy">Staking Strategy</label>
                            <select id="bankroll-strategy">
                                <option value="flat">Flat Stake</option>
                                <option value="kelly">Kelly Criterion</option>
                                <option value="proportional">Proportional</option>
                            </select>
                        </div>
                        
                        <div class="control-group">
                            <label for="flat-stake">Flat Stake ($)</label>
                            <input type="number" id="flat-stake" value="${this.simulation.flatStake}" min="1" max="100" step="1">
                        </div>
                        
                        <div class="control-group">
                            <label for="kelly-fraction">Kelly Fraction</label>
                            <input type="number" id="kelly-fraction" value="${this.simulation.kellyFraction}" min="0.1" max="1" step="0.05">
                        </div>
                        
                        <div class="control-buttons">
                            <button id="reset-simulation" class="btn btn-secondary">🔄 Reset</button>
                            <button id="generate-data" class="btn btn-primary">🎲 Generate Sample Data</button>
                            <button id="export-data" class="btn btn-info">📁 Export Results</button>
                        </div>
                    </div>
                </div>

                <!-- Statistics Overview -->
                <div class="simulation-stats">
                    <h3>📈 Performance Overview</h3>
                    <div class="stats-grid">
                        <div class="stat-card profit-card">
                            <div class="stat-value" id="net-profit">$0</div>
                            <div class="stat-label">Net Profit</div>
                        </div>
                        <div class="stat-card roi-card">
                            <div class="stat-value" id="roi-percent">0%</div>
                            <div class="stat-label">ROI</div>
                        </div>
                        <div class="stat-card winrate-card">
                            <div class="stat-value" id="win-rate">0%</div>
                            <div class="stat-label">Win Rate</div>
                        </div>
                        <div class="stat-card sharpe-card">
                            <div class="stat-value" id="sharpe-ratio">0.00</div>
                            <div class="stat-label">Sharpe Ratio</div>
                        </div>
                        <div class="stat-card drawdown-card">
                            <div class="stat-value" id="max-drawdown">0%</div>
                            <div class="stat-label">Max Drawdown</div>
                        </div>
                        <div class="stat-card total-bets-card">
                            <div class="stat-value" id="total-bets">0</div>
                            <div class="stat-label">Total Bets</div>
                        </div>
                    </div>
                </div>

                <!-- Charts Section -->
                <div class="simulation-charts">
                    <div class="chart-container">
                        <h3>💰 Capital Growth Curve</h3>
                        <canvas id="capital-chart"></canvas>
                    </div>
                    
                    <div class="chart-container">
                        <h3>📊 Profit/Loss Distribution</h3>
                        <canvas id="profit-distribution-chart"></canvas>
                    </div>
                </div>

                <!-- Detailed Analytics -->
                <div class="detailed-analytics">
                    <div class="analytics-grid">
                        <div class="analytics-card">
                            <h4>🔥 Streak Analysis</h4>
                            <div id="streak-analysis">
                                <div class="streak-item">
                                    <span class="streak-label">Longest Winning Streak:</span>
                                    <span class="streak-value" id="max-win-streak">0 bets</span>
                                </div>
                                <div class="streak-item">
                                    <span class="streak-label">Longest Losing Streak:</span>
                                    <span class="streak-value" id="max-lose-streak">0 bets</span>
                                </div>
                                <div class="streak-item">
                                    <span class="streak-label">Current Streak:</span>
                                    <span class="streak-value" id="current-streak">None</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="analytics-card">
                            <h4>💸 Drawdown Analysis</h4>
                            <div id="drawdown-analysis">
                                <div class="drawdown-item">
                                    <span class="drawdown-label">Max Drawdown:</span>
                                    <span class="drawdown-value" id="max-dd-value">0%</span>
                                </div>
                                <div class="drawdown-item">
                                    <span class="drawdown-label">Avg Drawdown:</span>
                                    <span class="drawdown-value" id="avg-drawdown">0%</span>
                                </div>
                                <div class="drawdown-item">
                                    <span class="drawdown-label">Recovery Time:</span>
                                    <span class="drawdown-value" id="recovery-time">0 days</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="analytics-card">
                            <h4>🎯 Model Performance</h4>
                            <div id="model-performance">
                                <div class="model-stats">
                                    <span class="model-label">Best Model:</span>
                                    <span class="model-value" id="best-model">-</span>
                                </div>
                                <div class="model-stats">
                                    <span class="model-label">Avg Confidence:</span>
                                    <span class="model-value" id="avg-confidence">0%</span>
                                </div>
                                <div class="model-stats">
                                    <span class="model-label">Avg Edge:</span>
                                    <span class="model-value" id="avg-edge">0%</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Recent Bets Table -->
                <div class="recent-bets">
                    <h3>📋 Recent Betting Activity</h3>
                    <div class="table-container">
                        <table id="bets-table">
                            <thead>
                                <tr>
                                    <th>Date</th>
                                    <th>Match</th>
                                    <th>Odds</th>
                                    <th>Stake</th>
                                    <th>Result</th>
                                    <th>Profit/Loss</th>
                                    <th>Edge</th>
                                    <th>Model</th>
                                </tr>
                            </thead>
                            <tbody id="bets-tbody">
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;
    }

    bindEvents() {
        // Control events
        document.getElementById('reset-simulation').addEventListener('click', () => this.resetSimulation());
        document.getElementById('generate-data').addEventListener('click', () => this.generateNewData());
        document.getElementById('export-data').addEventListener('click', () => this.exportData());
        
        // Settings changes
        document.getElementById('initial-capital').addEventListener('change', (e) => {
            this.simulation.initialCapital = parseFloat(e.target.value);
            this.resetSimulation();
        });
        
        document.getElementById('bankroll-strategy').addEventListener('change', (e) => {
            this.simulation.bankrollManagement = e.target.value;
            this.resetAndRerun();
        });
        
        document.getElementById('flat-stake').addEventListener('change', (e) => {
            this.simulation.flatStake = parseFloat(e.target.value);
            this.resetAndRerun();
        });
        
        document.getElementById('kelly-fraction').addEventListener('change', (e) => {
            this.simulation.kellyFraction = parseFloat(e.target.value);
            this.resetAndRerun();
        });
    }

    updateDisplay() {
        const stats = this.simulation.getStats();
        
        // Update main statistics
        this.updateElement('net-profit', `$${stats.netProfit.toFixed(2)}`);
        this.updateElement('roi-percent', `${stats.roi.toFixed(1)}%`);
        this.updateElement('win-rate', `${stats.winRate.toFixed(1)}%`);
        this.updateElement('sharpe-ratio', stats.sharpeRatio.toFixed(2));
        this.updateElement('max-drawdown', `${stats.maxDrawdown.toFixed(1)}%`);
        this.updateElement('total-bets', stats.totalBets.toString());
        
        // Update streak analysis
        this.updateStreakAnalysis();
        
        // Update drawdown analysis
        this.updateDrawdownAnalysis();
        
        // Update model performance
        this.updateModelPerformance();
        
        // Update charts
        this.updateCharts();
        
        // Update recent bets table
        this.updateBetsTable();
    }

    updateElement(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
            
            // Add visual feedback for profit/loss
            if (id === 'net-profit') {
                const profit = parseFloat(value.replace('$', ''));
                element.className = profit >= 0 ? 'profit' : 'loss';
            }
        }
    }

    updateStreakAnalysis() {
        const stats = this.simulation.getStats();
        const currentStreak = this.simulation.currentStreak;
        
        this.updateElement('max-win-streak', `${stats.winningStreak} bets`);
        this.updateElement('max-lose-streak', `${stats.losingStreak} bets`);
        
        if (currentStreak) {
            const streakText = `${currentStreak.type} (${currentStreak.length} bets)`;
            this.updateElement('current-streak', streakText);
        } else {
            this.updateElement('current-streak', 'None');
        }
    }

    updateDrawdownAnalysis() {
        const drawdowns = this.simulation.drawdowns;
        if (drawdowns.length === 0) {
            this.updateElement('max-dd-value', '0%');
            this.updateElement('avg-drawdown', '0%');
            this.updateElement('recovery-time', '0 days');
            return;
        }
        
        const maxDrawdown = Math.max(...drawdowns.map(d => d.drawdownPercent));
        const avgDrawdown = drawdowns.reduce((sum, d) => sum + d.drawdownPercent, 0) / drawdowns.length;
        const avgRecovery = drawdowns.reduce((sum, d) => sum + (d.duration || 0), 0) / drawdowns.length;
        
        this.updateElement('max-dd-value', `${maxDrawdown.toFixed(1)}%`);
        this.updateElement('avg-drawdown', `${avgDrawdown.toFixed(1)}%`);
        this.updateElement('recovery-time', `${avgRecovery.toFixed(0)} days`);
    }

    updateModelPerformance() {
        const bets = this.simulation.betResults;
        if (bets.length === 0) return;
        
        // Find best performing model
        const modelStats = {};
        bets.forEach(bet => {
            if (!modelStats[bet.model]) {
                modelStats[bet.model] = { profits: [], count: 0 };
            }
            modelStats[bet.model].profits.push(bet.profit);
            modelStats[bet.model].count++;
        });
        
        let bestModel = '';
        let bestProfit = -Infinity;
        
        Object.entries(modelStats).forEach(([model, stats]) => {
            const totalProfit = stats.profits.reduce((sum, p) => sum + p, 0);
            if (totalProfit > bestProfit) {
                bestProfit = totalProfit;
                bestModel = model;
            }
        });
        
        const avgConfidence = bets.reduce((sum, bet) => sum + bet.confidence, 0) / bets.length;
        const avgEdge = bets.reduce((sum, bet) => sum + bet.edge, 0) / bets.length;
        
        this.updateElement('best-model', bestModel);
        this.updateElement('avg-confidence', `${(avgConfidence * 100).toFixed(1)}%`);
        this.updateElement('avg-edge', `${(avgEdge * 100).toFixed(1)}%`);
    }

    updateCharts() {
        this.updateCapitalChart();
        this.updateProfitDistributionChart();
    }

    updateCapitalChart() {
        const ctx = document.getElementById('capital-chart').getContext('2d');
        
        // Destroy existing chart
        if (this.chartInstances.has('capital')) {
            this.chartInstances.get('capital').destroy();
        }
        
        const capitalHistory = this.simulation.capitalHistory;
        const labels = capitalHistory.map(h => h.date.toLocaleDateString());
        const data = capitalHistory.map(h => h.capital);
        
        // Add initial point
        if (capitalHistory.length > 0) {
            labels.unshift('Start');
            data.unshift(this.simulation.initialCapital);
        }
        
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label: 'Capital ($)',
                    data,
                    borderColor: '#6bcf7f',
                    backgroundColor: 'rgba(107, 207, 127, 0.1)',
                    fill: true,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false,
                        title: {
                            display: true,
                            text: 'Capital ($)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
        
        this.chartInstances.set('capital', chart);
    }

    updateProfitDistributionChart() {
        const ctx = document.getElementById('profit-distribution-chart').getContext('2d');
        
        // Destroy existing chart
        if (this.chartInstances.has('profit')) {
            this.chartInstances.get('profit').destroy();
        }
        
        const profits = this.simulation.betResults.map(bet => bet.profit);
        if (profits.length === 0) return;
        
        // Create histogram bins
        const bins = 10;
        const min = Math.min(...profits);
        const max = Math.max(...profits);
        const binSize = (max - min) / bins;
        
        const binCounts = new Array(bins).fill(0);
        const binLabels = [];
        
        for (let i = 0; i < bins; i++) {
            const binStart = min + i * binSize;
            const binEnd = min + (i + 1) * binSize;
            binLabels.push(`${binStart.toFixed(1)} to ${binEnd.toFixed(1)}`);
            
            profits.forEach(profit => {
                if (profit >= binStart && (profit < binEnd || i === bins - 1)) {
                    binCounts[i]++;
                }
            });
        }
        
        const chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: binLabels,
                datasets: [{
                    label: 'Frequency',
                    data: binCounts,
                    backgroundColor: binLabels.map((_, i) => {
                        const binStart = min + i * binSize;
                        return binStart >= 0 ? 'rgba(107, 207, 127, 0.7)' : 'rgba(255, 107, 107, 0.7)';
                    }),
                    borderColor: binLabels.map((_, i) => {
                        const binStart = min + i * binSize;
                        return binStart >= 0 ? '#6bcf7f' : '#ff6b6b';
                    }),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Number of Bets'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Profit/Loss Range ($)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
        
        this.chartInstances.set('profit', chart);
    }

    updateBetsTable() {
        const tbody = document.getElementById('bets-tbody');
        const recentBets = this.simulation.betResults.slice(-20).reverse(); // Last 20 bets, newest first
        
        tbody.innerHTML = recentBets.map(bet => `
            <tr class="${bet.won ? 'bet-won' : 'bet-lost'}">
                <td>${bet.date.toLocaleDateString()}</td>
                <td class="match-cell">${bet.match}</td>
                <td>${bet.odds.toFixed(2)}</td>
                <td>$${bet.stake.toFixed(2)}</td>
                <td class="result-cell">${bet.won ? '✅ Won' : '❌ Lost'}</td>
                <td class="profit-cell ${bet.profit >= 0 ? 'profit' : 'loss'}">
                    ${bet.profit >= 0 ? '+' : ''}$${bet.profit.toFixed(2)}
                </td>
                <td>${(bet.edge * 100).toFixed(1)}%</td>
                <td>${bet.model}</td>
            </tr>
        `).join('');
    }

    resetSimulation() {
        this.simulation.reset();
        this.updateDisplay();
    }

    generateNewData() {
        this.simulation.reset();
        this.simulation.generateSampleData(50);
        this.updateDisplay();
    }

    resetAndRerun() {
        const oldResults = [...this.simulation.betResults];
        this.simulation.reset();
        
        // Reprocess all bets with new settings
        oldResults.forEach(bet => {
            // Remove calculated fields
            const cleanBet = {
                ...bet,
                stake: 0,
                profit: 0,
                payout: 0
            };
            this.simulation.addBetResult(cleanBet);
        });
        
        this.updateDisplay();
    }

    exportData() {
        const data = this.simulation.exportData();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `betting-simulation-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    destroy() {
        // Clean up chart instances
        this.chartInstances.forEach(chart => chart.destroy());
        this.chartInstances.clear();
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { BettingSimulationUI };
}

// Global availability
window.BettingSimulationUI = BettingSimulationUI;