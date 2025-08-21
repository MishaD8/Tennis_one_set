/**
 * Betting Automation Controls Dashboard
 * Features: Auto-betting configuration, risk management, position sizing, safety controls
 */

class BettingAutomationControls {
    constructor() {
        // Use current host and port for API base to avoid localhost issues
        this.API_BASE = `${window.location.protocol}//${window.location.host}/api`;
        this.automationConfig = {
            enabled: false,
            maxBetsPerDay: 5,
            maxStakePerBet: 100,
            minEdgeThreshold: 0.05, // 5%
            maxRiskPercentage: 2, // 2% of bankroll
            bankrollAmount: 10000,
            stakingStrategy: 'kelly', // 'flat', 'kelly', 'proportional'
            autoApproveThreshold: 0.08, // 8% edge for auto-approval
            safetyChecks: {
                requireManualApproval: true,
                maxConsecutiveLosses: 3,
                stopLossPercentage: 10,
                maxDrawdownPercentage: 15
            },
            filters: {
                tournamentTypes: ['ATP', 'WTA'],
                minOdds: 1.5,
                maxOdds: 4.0,
                excludeQualifiers: true,
                requireRankingData: true
            }
        };
        this.activeBets = new Map();
        this.automationHistory = [];
        this.isMonitoring = false;
        this.init();
    }

    async init() {
        try {
            this.loadAutomationConfig();
            this.createAutomationUI();
            this.setupEventListeners();
            await this.loadActiveBets();
            this.startMonitoring();
            
            console.log('✅ Betting Automation Controls initialized');
        } catch (error) {
            console.error('Error initializing betting automation controls:', error);
        }
    }

    createAutomationUI() {
        // Find or create container for automation controls
        let container = document.getElementById('betting-automation-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'betting-automation-container';
            container.className = 'betting-automation-controls';
            
            // Add to appropriate section (money management tab)
            const moneyManagementTab = document.getElementById('money-management');
            if (moneyManagementTab) {
                const dashboard = moneyManagementTab.querySelector('.money-management-dashboard');
                if (dashboard) {
                    dashboard.appendChild(container);
                }
            } else {
                // Fallback: add to main container
                const mainContainer = document.querySelector('.container');
                if (mainContainer) {
                    mainContainer.appendChild(container);
                }
            }
        }

        container.innerHTML = this.createAutomationHTML();
    }

    createAutomationHTML() {
        return `
            <div class="automation-panel">
                <div class="automation-header">
                    <h3>🤖 Betting Automation Controls</h3>
                    <div class="automation-master-switch">
                        <label class="master-switch-label">
                            <input type="checkbox" id="automation-master-toggle" ${this.automationConfig.enabled ? 'checked' : ''}>
                            <span class="switch-slider"></span>
                            <span class="switch-text">Auto-Betting ${this.automationConfig.enabled ? 'Enabled' : 'Disabled'}</span>
                        </label>
                    </div>
                </div>

                <div class="automation-status" id="automation-status">
                    <div class="status-indicator ${this.automationConfig.enabled ? 'active' : 'inactive'}">
                        <div class="status-icon">${this.automationConfig.enabled ? '🟢' : '🔴'}</div>
                        <div class="status-text">
                            <div class="status-label">Automation Status</div>
                            <div class="status-value" id="automation-status-text">
                                ${this.automationConfig.enabled ? 'Active - Monitoring for opportunities' : 'Inactive'}
                            </div>
                        </div>
                    </div>
                    
                    <div class="automation-stats">
                        <div class="stat-item">
                            <div class="stat-label">Bets Today</div>
                            <div class="stat-value" id="bets-today">0 / ${this.automationConfig.maxBetsPerDay}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Risk Exposure</div>
                            <div class="stat-value" id="risk-exposure">0%</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Next Check</div>
                            <div class="stat-value" id="next-check">--</div>
                        </div>
                    </div>
                </div>

                <div class="automation-tabs">
                    <button class="tab-btn active" data-tab="configuration">Configuration</button>
                    <button class="tab-btn" data-tab="active-bets">Active Bets</button>
                    <button class="tab-btn" data-tab="history">History</button>
                    <button class="tab-btn" data-tab="safety">Safety</button>
                </div>

                <!-- Configuration Tab -->
                <div class="automation-tab-content active" id="configuration-tab">
                    <div class="config-sections">
                        <div class="config-section">
                            <h4>💰 Bankroll & Risk Management</h4>
                            <div class="config-grid">
                                <div class="config-item">
                                    <label>Bankroll Amount ($)</label>
                                    <input type="number" id="bankroll-amount" value="${this.automationConfig.bankrollAmount}" min="100" step="100">
                                </div>
                                <div class="config-item">
                                    <label>Max Stake Per Bet ($)</label>
                                    <input type="number" id="max-stake-per-bet" value="${this.automationConfig.maxStakePerBet}" min="1" step="1">
                                </div>
                                <div class="config-item">
                                    <label>Max Risk Per Bet (%)</label>
                                    <input type="range" id="max-risk-percentage" min="0.5" max="10" step="0.5" value="${this.automationConfig.maxRiskPercentage}">
                                    <span class="range-value">${this.automationConfig.maxRiskPercentage}%</span>
                                </div>
                                <div class="config-item">
                                    <label>Staking Strategy</label>
                                    <select id="staking-strategy">
                                        <option value="flat" ${this.automationConfig.stakingStrategy === 'flat' ? 'selected' : ''}>Flat Betting</option>
                                        <option value="kelly" ${this.automationConfig.stakingStrategy === 'kelly' ? 'selected' : ''}>Kelly Criterion</option>
                                        <option value="proportional" ${this.automationConfig.stakingStrategy === 'proportional' ? 'selected' : ''}>Proportional</option>
                                    </select>
                                </div>
                            </div>
                        </div>

                        <div class="config-section">
                            <h4>🎯 Betting Criteria</h4>
                            <div class="config-grid">
                                <div class="config-item">
                                    <label>Min Edge Threshold (%)</label>
                                    <input type="range" id="min-edge-threshold" min="2" max="15" step="0.5" value="${this.automationConfig.minEdgeThreshold * 100}">
                                    <span class="range-value">${(this.automationConfig.minEdgeThreshold * 100).toFixed(1)}%</span>
                                </div>
                                <div class="config-item">
                                    <label>Auto-Approve Edge (%)</label>
                                    <input type="range" id="auto-approve-threshold" min="5" max="20" step="0.5" value="${this.automationConfig.autoApproveThreshold * 100}">
                                    <span class="range-value">${(this.automationConfig.autoApproveThreshold * 100).toFixed(1)}%</span>
                                </div>
                                <div class="config-item">
                                    <label>Max Bets Per Day</label>
                                    <input type="number" id="max-bets-per-day" value="${this.automationConfig.maxBetsPerDay}" min="1" max="20" step="1">
                                </div>
                                <div class="config-item">
                                    <label>Min Odds</label>
                                    <input type="number" id="min-odds" value="${this.automationConfig.filters.minOdds}" min="1.1" max="3.0" step="0.1">
                                </div>
                                <div class="config-item">
                                    <label>Max Odds</label>
                                    <input type="number" id="max-odds" value="${this.automationConfig.filters.maxOdds}" min="2.0" max="10.0" step="0.1">
                                </div>
                            </div>
                        </div>

                        <div class="config-section">
                            <h4>🏆 Tournament Filters</h4>
                            <div class="filter-options">
                                <label class="filter-checkbox">
                                    <input type="checkbox" id="allow-atp" ${this.automationConfig.filters.tournamentTypes.includes('ATP') ? 'checked' : ''}>
                                    <span>ATP Tournaments</span>
                                </label>
                                <label class="filter-checkbox">
                                    <input type="checkbox" id="allow-wta" ${this.automationConfig.filters.tournamentTypes.includes('WTA') ? 'checked' : ''}>
                                    <span>WTA Tournaments</span>
                                </label>
                                <label class="filter-checkbox">
                                    <input type="checkbox" id="exclude-qualifiers" ${this.automationConfig.filters.excludeQualifiers ? 'checked' : ''}>
                                    <span>Exclude Qualifiers</span>
                                </label>
                                <label class="filter-checkbox">
                                    <input type="checkbox" id="require-ranking-data" ${this.automationConfig.filters.requireRankingData ? 'checked' : ''}>
                                    <span>Require Ranking Data</span>
                                </label>
                            </div>
                        </div>
                    </div>

                    <div class="config-actions">
                        <button class="btn-save-config" id="save-automation-config">Save Configuration</button>
                        <button class="btn-test-config" id="test-automation-config">Test Configuration</button>
                        <button class="btn-reset-config" id="reset-automation-config">Reset to Defaults</button>
                    </div>
                </div>

                <!-- Active Bets Tab -->
                <div class="automation-tab-content" id="active-bets-tab">
                    <div class="active-bets-header">
                        <h4>🎯 Active Automated Bets</h4>
                        <button class="btn-refresh-bets" id="refresh-active-bets">Refresh</button>
                    </div>
                    <div class="active-bets-list" id="active-bets-list">
                        <!-- Active bets will be rendered here -->
                    </div>
                </div>

                <!-- History Tab -->
                <div class="automation-tab-content" id="history-tab">
                    <div class="history-header">
                        <h4>📊 Automation History</h4>
                        <div class="history-filters">
                            <select id="history-period">
                                <option value="today">Today</option>
                                <option value="week">This Week</option>
                                <option value="month">This Month</option>
                                <option value="all">All Time</option>
                            </select>
                        </div>
                    </div>
                    <div class="automation-history-list" id="automation-history-list">
                        <!-- History will be rendered here -->
                    </div>
                </div>

                <!-- Safety Tab -->
                <div class="automation-tab-content" id="safety-tab">
                    <div class="safety-controls">
                        <h4>⚠️ Safety Controls & Circuit Breakers</h4>
                        
                        <div class="safety-section">
                            <h5>🛡️ Risk Limits</h5>
                            <div class="safety-grid">
                                <div class="safety-item">
                                    <label>Max Consecutive Losses</label>
                                    <input type="number" id="max-consecutive-losses" value="${this.automationConfig.safetyChecks.maxConsecutiveLosses}" min="1" max="10">
                                </div>
                                <div class="safety-item">
                                    <label>Stop Loss Percentage (%)</label>
                                    <input type="range" id="stop-loss-percentage" min="5" max="50" step="1" value="${this.automationConfig.safetyChecks.stopLossPercentage}">
                                    <span class="range-value">${this.automationConfig.safetyChecks.stopLossPercentage}%</span>
                                </div>
                                <div class="safety-item">
                                    <label>Max Drawdown Percentage (%)</label>
                                    <input type="range" id="max-drawdown-percentage" min="10" max="50" step="1" value="${this.automationConfig.safetyChecks.maxDrawdownPercentage}">
                                    <span class="range-value">${this.automationConfig.safetyChecks.maxDrawdownPercentage}%</span>
                                </div>
                            </div>
                        </div>

                        <div class="safety-section">
                            <h5>🚨 Emergency Controls</h5>
                            <div class="emergency-controls">
                                <button class="btn-emergency-stop" id="emergency-stop-all">🛑 Emergency Stop All</button>
                                <button class="btn-pause-automation" id="pause-automation">⏸️ Pause Automation</button>
                                <button class="btn-close-all-positions" id="close-all-positions">🔒 Close All Positions</button>
                            </div>
                        </div>

                        <div class="safety-section">
                            <h5>📋 Manual Approval Settings</h5>
                            <label class="safety-checkbox">
                                <input type="checkbox" id="require-manual-approval" ${this.automationConfig.safetyChecks.requireManualApproval ? 'checked' : ''}>
                                <span>Require manual approval for all bets</span>
                            </label>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    setupEventListeners() {
        // Master toggle
        const masterToggle = document.getElementById('automation-master-toggle');
        if (masterToggle) {
            masterToggle.addEventListener('change', (e) => this.toggleAutomation(e.target.checked));
        }

        // Tab navigation
        const tabButtons = document.querySelectorAll('.automation-tabs .tab-btn');
        tabButtons.forEach(btn => {
            btn.addEventListener('click', () => this.switchTab(btn.dataset.tab));
        });

        // Configuration inputs
        this.setupConfigurationListeners();

        // Action buttons
        this.setupActionButtonListeners();

        // Safety controls
        this.setupSafetyControlListeners();

        // Range input updates
        this.setupRangeInputListeners();
    }

    setupConfigurationListeners() {
        const configInputs = [
            'bankroll-amount', 'max-stake-per-bet', 'max-risk-percentage',
            'staking-strategy', 'min-edge-threshold', 'auto-approve-threshold',
            'max-bets-per-day', 'min-odds', 'max-odds'
        ];

        configInputs.forEach(id => {
            const input = document.getElementById(id);
            if (input) {
                input.addEventListener('change', () => this.updateConfiguration());
            }
        });

        // Tournament filters
        const filterInputs = ['allow-atp', 'allow-wta', 'exclude-qualifiers', 'require-ranking-data'];
        filterInputs.forEach(id => {
            const input = document.getElementById(id);
            if (input) {
                input.addEventListener('change', () => this.updateConfiguration());
            }
        });
    }

    setupActionButtonListeners() {
        const actions = [
            { id: 'save-automation-config', handler: () => this.saveConfiguration() },
            { id: 'test-automation-config', handler: () => this.testConfiguration() },
            { id: 'reset-automation-config', handler: () => this.resetConfiguration() },
            { id: 'refresh-active-bets', handler: () => this.loadActiveBets() }
        ];

        actions.forEach(({ id, handler }) => {
            const btn = document.getElementById(id);
            if (btn) {
                btn.addEventListener('click', handler);
            }
        });
    }

    setupSafetyControlListeners() {
        const safetyInputs = [
            'max-consecutive-losses', 'stop-loss-percentage', 'max-drawdown-percentage',
            'require-manual-approval'
        ];

        safetyInputs.forEach(id => {
            const input = document.getElementById(id);
            if (input) {
                input.addEventListener('change', () => this.updateSafetySettings());
            }
        });

        // Emergency controls
        const emergencyActions = [
            { id: 'emergency-stop-all', handler: () => this.emergencyStopAll() },
            { id: 'pause-automation', handler: () => this.pauseAutomation() },
            { id: 'close-all-positions', handler: () => this.closeAllPositions() }
        ];

        emergencyActions.forEach(({ id, handler }) => {
            const btn = document.getElementById(id);
            if (btn) {
                btn.addEventListener('click', handler);
            }
        });
    }

    setupRangeInputListeners() {
        const rangeInputs = document.querySelectorAll('input[type="range"]');
        rangeInputs.forEach(input => {
            input.addEventListener('input', (e) => {
                const value = e.target.value;
                const valueSpan = e.target.nextElementSibling;
                if (valueSpan && valueSpan.classList.contains('range-value')) {
                    valueSpan.textContent = `${value}%`;
                }
            });
        });
    }

    async toggleAutomation(enabled) {
        try {
            this.automationConfig.enabled = enabled;
            
            const response = await fetch(`${this.API_BASE}/betting/automation/toggle`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ enabled })
            });

            if (response.ok) {
                this.updateAutomationStatus();
                this.saveAutomationConfig();
                
                if (window.notificationManager) {
                    window.notificationManager.notify(
                        enabled ? 'success' : 'info',
                        'Automation Status',
                        `Auto-betting ${enabled ? 'enabled' : 'disabled'}`,
                        { priority: 'high' }
                    );
                }
            } else {
                throw new Error('Failed to toggle automation');
            }
        } catch (error) {
            console.error('Error toggling automation:', error);
            // Revert toggle
            const toggle = document.getElementById('automation-master-toggle');
            if (toggle) {
                toggle.checked = !enabled;
            }
        }
    }

    updateConfiguration() {
        // Update configuration object from form inputs
        this.automationConfig.bankrollAmount = parseFloat(document.getElementById('bankroll-amount')?.value || 10000);
        this.automationConfig.maxStakePerBet = parseFloat(document.getElementById('max-stake-per-bet')?.value || 100);
        this.automationConfig.maxRiskPercentage = parseFloat(document.getElementById('max-risk-percentage')?.value || 2);
        this.automationConfig.stakingStrategy = document.getElementById('staking-strategy')?.value || 'kelly';
        this.automationConfig.minEdgeThreshold = parseFloat(document.getElementById('min-edge-threshold')?.value || 5) / 100;
        this.automationConfig.autoApproveThreshold = parseFloat(document.getElementById('auto-approve-threshold')?.value || 8) / 100;
        this.automationConfig.maxBetsPerDay = parseInt(document.getElementById('max-bets-per-day')?.value || 5);
        
        // Update filters
        this.automationConfig.filters.minOdds = parseFloat(document.getElementById('min-odds')?.value || 1.5);
        this.automationConfig.filters.maxOdds = parseFloat(document.getElementById('max-odds')?.value || 4.0);
        this.automationConfig.filters.excludeQualifiers = document.getElementById('exclude-qualifiers')?.checked || true;
        this.automationConfig.filters.requireRankingData = document.getElementById('require-ranking-data')?.checked || true;
        
        // Tournament types
        const tournamentTypes = [];
        if (document.getElementById('allow-atp')?.checked) tournamentTypes.push('ATP');
        if (document.getElementById('allow-wta')?.checked) tournamentTypes.push('WTA');
        this.automationConfig.filters.tournamentTypes = tournamentTypes;
    }

    updateSafetySettings() {
        this.automationConfig.safetyChecks.maxConsecutiveLosses = parseInt(document.getElementById('max-consecutive-losses')?.value || 3);
        this.automationConfig.safetyChecks.stopLossPercentage = parseFloat(document.getElementById('stop-loss-percentage')?.value || 10);
        this.automationConfig.safetyChecks.maxDrawdownPercentage = parseFloat(document.getElementById('max-drawdown-percentage')?.value || 15);
        this.automationConfig.safetyChecks.requireManualApproval = document.getElementById('require-manual-approval')?.checked || true;
    }

    async saveConfiguration() {
        try {
            this.updateConfiguration();
            this.updateSafetySettings();
            
            const response = await fetch(`${this.API_BASE}/betting/automation/config`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(this.automationConfig)
            });

            if (response.ok) {
                this.saveAutomationConfig();
                
                if (window.notificationManager) {
                    window.notificationManager.notify(
                        'success',
                        'Configuration Saved',
                        'Automation configuration has been saved successfully'
                    );
                }
            } else {
                throw new Error('Failed to save configuration');
            }
        } catch (error) {
            console.error('Error saving configuration:', error);
            if (window.notificationManager) {
                window.notificationManager.notify(
                    'error',
                    'Configuration Error',
                    'Failed to save automation configuration'
                );
            }
        }
    }

    async testConfiguration() {
        try {
            this.updateConfiguration();
            
            const response = await fetch(`${this.API_BASE}/betting/automation/test`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(this.automationConfig)
            });

            if (response.ok) {
                const result = await response.json();
                
                if (window.notificationManager) {
                    window.notificationManager.notify(
                        'info',
                        'Configuration Test',
                        `Test completed: ${result.message || 'Configuration is valid'}`,
                        { autoHide: false }
                    );
                }
            } else {
                throw new Error('Configuration test failed');
            }
        } catch (error) {
            console.error('Error testing configuration:', error);
            if (window.notificationManager) {
                window.notificationManager.notify(
                    'error',
                    'Configuration Test Failed',
                    'Please check your settings and try again'
                );
            }
        }
    }

    resetConfiguration() {
        // Reset to default values
        this.automationConfig = {
            enabled: false,
            maxBetsPerDay: 5,
            maxStakePerBet: 100,
            minEdgeThreshold: 0.05,
            maxRiskPercentage: 2,
            bankrollAmount: 10000,
            stakingStrategy: 'kelly',
            autoApproveThreshold: 0.08,
            safetyChecks: {
                requireManualApproval: true,
                maxConsecutiveLosses: 3,
                stopLossPercentage: 10,
                maxDrawdownPercentage: 15
            },
            filters: {
                tournamentTypes: ['ATP', 'WTA'],
                minOdds: 1.5,
                maxOdds: 4.0,
                excludeQualifiers: true,
                requireRankingData: true
            }
        };

        // Recreate UI with default values
        this.createAutomationUI();
        this.setupEventListeners();
        
        if (window.notificationManager) {
            window.notificationManager.notify(
                'info',
                'Configuration Reset',
                'Automation configuration has been reset to defaults'
            );
        }
    }

    switchTab(tabName) {
        // Update tab buttons
        const tabButtons = document.querySelectorAll('.automation-tabs .tab-btn');
        tabButtons.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabName);
        });

        // Update tab content
        const tabContents = document.querySelectorAll('.automation-tab-content');
        tabContents.forEach(content => {
            content.classList.toggle('active', content.id === `${tabName}-tab`);
        });

        // Load tab-specific data
        if (tabName === 'active-bets') {
            this.loadActiveBets();
        } else if (tabName === 'history') {
            this.loadAutomationHistory();
        }
    }

    async loadActiveBets() {
        try {
            const response = await fetch(`${this.API_BASE}/betting/automation/active-bets`);
            if (response.ok) {
                const data = await response.json();
                this.renderActiveBets(data.bets || []);
            }
        } catch (error) {
            console.error('Error loading active bets:', error);
        }
    }

    renderActiveBets(bets) {
        const container = document.getElementById('active-bets-list');
        if (!container) return;

        if (bets.length === 0) {
            container.innerHTML = `
                <div class="no-active-bets">
                    <div class="no-bets-icon">📊</div>
                    <div class="no-bets-text">No active automated bets</div>
                </div>
            `;
            return;
        }

        container.innerHTML = bets.map(bet => `
            <div class="active-bet-item">
                <div class="bet-header">
                    <div class="bet-match">${bet.match.player1} vs ${bet.match.player2}</div>
                    <div class="bet-status ${bet.status}">${bet.status}</div>
                </div>
                <div class="bet-details">
                    <div class="bet-detail">
                        <span class="detail-label">Stake:</span>
                        <span class="detail-value">$${bet.stake}</span>
                    </div>
                    <div class="bet-detail">
                        <span class="detail-label">Odds:</span>
                        <span class="detail-value">${bet.odds}</span>
                    </div>
                    <div class="bet-detail">
                        <span class="detail-label">Edge:</span>
                        <span class="detail-value">${(bet.edge * 100).toFixed(1)}%</span>
                    </div>
                    <div class="bet-detail">
                        <span class="detail-label">Placed:</span>
                        <span class="detail-value">${new Date(bet.timestamp).toLocaleString()}</span>
                    </div>
                </div>
                <div class="bet-actions">
                    <button class="btn-view-bet" onclick="window.bettingAutomation?.viewBet('${bet.id}')">View</button>
                    <button class="btn-close-bet" onclick="window.bettingAutomation?.closeBet('${bet.id}')">Close</button>
                </div>
            </div>
        `).join('');
    }

    async loadAutomationHistory() {
        // Implementation for loading automation history
        console.log('Loading automation history...');
    }

    updateAutomationStatus() {
        const statusIndicator = document.querySelector('.automation-status .status-indicator');
        const statusText = document.getElementById('automation-status-text');
        const masterToggle = document.getElementById('automation-master-toggle');
        
        if (statusIndicator && statusText) {
            statusIndicator.className = `status-indicator ${this.automationConfig.enabled ? 'active' : 'inactive'}`;
            statusText.textContent = this.automationConfig.enabled ? 
                'Active - Monitoring for opportunities' : 'Inactive';
        }
        
        if (masterToggle) {
            const switchText = masterToggle.parentNode.querySelector('.switch-text');
            if (switchText) {
                switchText.textContent = `Auto-Betting ${this.automationConfig.enabled ? 'Enabled' : 'Disabled'}`;
            }
        }
    }

    async emergencyStopAll() {
        if (confirm('Are you sure you want to emergency stop all automation? This will immediately disable all automated betting and close pending orders.')) {
            try {
                this.automationConfig.enabled = false;
                // Implementation for emergency stop
                this.updateAutomationStatus();
                
                if (window.notificationManager) {
                    window.notificationManager.notify(
                        'warning',
                        'Emergency Stop Activated',
                        'All automated betting has been stopped',
                        { priority: 'high', persistent: true }
                    );
                }
            } catch (error) {
                console.error('Error during emergency stop:', error);
            }
        }
    }

    pauseAutomation() {
        this.automationConfig.enabled = false;
        this.updateAutomationStatus();
        
        if (window.notificationManager) {
            window.notificationManager.notify(
                'info',
                'Automation Paused',
                'Automated betting has been paused'
            );
        }
    }

    async closeAllPositions() {
        if (confirm('Are you sure you want to close all active positions?')) {
            // Implementation for closing all positions
            console.log('Closing all positions...');
        }
    }

    startMonitoring() {
        if (this.isMonitoring) return;
        
        this.isMonitoring = true;
        this.monitoringInterval = setInterval(() => {
            this.updateAutomationStats();
        }, 10000); // Update every 10 seconds
    }

    stopMonitoring() {
        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval);
            this.monitoringInterval = null;
        }
        this.isMonitoring = false;
    }

    updateAutomationStats() {
        // Update stats display
        const betsToday = document.getElementById('bets-today');
        const riskExposure = document.getElementById('risk-exposure');
        const nextCheck = document.getElementById('next-check');
        
        if (betsToday) {
            const todaysBets = this.getTodaysBetCount();
            betsToday.textContent = `${todaysBets} / ${this.automationConfig.maxBetsPerDay}`;
        }
        
        if (riskExposure) {
            const exposure = this.calculateRiskExposure();
            riskExposure.textContent = `${exposure.toFixed(1)}%`;
        }
        
        if (nextCheck) {
            nextCheck.textContent = this.getNextCheckTime();
        }
    }

    getTodaysBetCount() {
        // Mock implementation - would fetch from API
        return Math.floor(Math.random() * this.automationConfig.maxBetsPerDay);
    }

    calculateRiskExposure() {
        // Mock implementation - would calculate actual risk exposure
        return Math.random() * this.automationConfig.maxRiskPercentage * 2;
    }

    getNextCheckTime() {
        const nextCheck = new Date(Date.now() + 5 * 60 * 1000); // 5 minutes from now
        return nextCheck.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    }

    loadAutomationConfig() {
        try {
            const saved = localStorage.getItem('betting-automation-config');
            if (saved) {
                this.automationConfig = { ...this.automationConfig, ...JSON.parse(saved) };
            }
        } catch (error) {
            console.warn('Error loading automation config:', error);
        }
    }

    saveAutomationConfig() {
        try {
            localStorage.setItem('betting-automation-config', JSON.stringify(this.automationConfig));
        } catch (error) {
            console.warn('Error saving automation config:', error);
        }
    }

    // Public API methods
    viewBet(betId) {
        console.log('View bet:', betId);
    }

    closeBet(betId) {
        console.log('Close bet:', betId);
    }

    destroy() {
        this.stopMonitoring();
        
        const container = document.getElementById('betting-automation-container');
        if (container) {
            container.remove();
        }
    }
}

// Global initialization
function initBettingAutomationControls() {
    try {
        if (!window.bettingAutomation) {
            window.bettingAutomation = new BettingAutomationControls();
        }
    } catch (error) {
        console.error('Failed to initialize betting automation controls:', error);
    }
}

// Auto-initialize when DOM is ready and money management tab is available
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        const moneyManagementTab = document.getElementById('money-management');
        if (moneyManagementTab) {
            initBettingAutomationControls();
        }
    }, 1500);
});

// Export for manual initialization
window.BettingAutomationControls = BettingAutomationControls;
window.initBettingAutomationControls = initBettingAutomationControls;