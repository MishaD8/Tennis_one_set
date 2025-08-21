/**
 * System Health Monitor - Real-time Backend Status Tracking
 * Monitors API health, database connectivity, ML model status, and system performance
 */

class SystemHealthMonitor {
    constructor() {
        // Use current host and port for API base to avoid localhost issues
        this.API_BASE = `${window.location.protocol}//${window.location.host}/api`;
        this.healthData = {
            overall: { status: 'unknown', score: 0 },
            api: { status: 'unknown', responseTime: 0, errorRate: 0 },
            database: { status: 'unknown', connectivity: false },
            mlModels: { status: 'unknown', modelsLoaded: 0, totalModels: 0 },
            dataCollection: { status: 'unknown', lastUpdate: null },
            performance: { cpuUsage: 0, memoryUsage: 0, activeConnections: 0 }
        };
        this.isMonitoring = false;
        this.monitoringInterval = null;
        this.healthHistory = [];
        this.maxHistoryLength = 50;
        this.init();
    }

    async init() {
        try {
            this.createHealthMonitorUI();
            await this.performInitialHealthCheck();
            this.startMonitoring();
            this.setupEventListeners();
        } catch (error) {
            console.error('Error initializing system health monitor:', error);
        }
    }

    createHealthMonitorUI() {
        // Add health monitor to dashboard if not already present
        let healthContainer = document.getElementById('system-health-container');
        if (!healthContainer) {
            healthContainer = document.createElement('div');
            healthContainer.id = 'system-health-container';
            healthContainer.className = 'system-health-monitor';
            
            // Find a good place to insert (after stats grid)
            const statsGrid = document.querySelector('.stats-grid');
            if (statsGrid && statsGrid.parentNode) {
                statsGrid.parentNode.insertAdjacentElement('afterend', healthContainer);
            } else {
                // Fallback: add to main container
                const container = document.querySelector('.container');
                if (container) {
                    container.insertAdjacentElement('afterbegin', healthContainer);
                }
            }
        }

        healthContainer.innerHTML = this.createHealthMonitorHTML();
    }

    createHealthMonitorHTML() {
        return `
            <div class="health-monitor-panel">
                <div class="health-header">
                    <h3>🔍 System Health Monitor</h3>
                    <div class="health-controls">
                        <button class="health-toggle-btn" id="health-toggle-monitoring" title="Toggle monitoring">
                            <span class="btn-icon">⏸️</span>
                            <span class="btn-text">Pause</span>
                        </button>
                        <button class="health-refresh-btn" id="health-manual-refresh" title="Manual refresh">
                            <span class="btn-icon">🔄</span>
                        </button>
                        <button class="health-details-btn" id="health-toggle-details" title="Toggle details">
                            <span class="btn-icon">📊</span>
                        </button>
                    </div>
                </div>

                <div class="health-overview">
                    <div class="overall-health-indicator">
                        <div class="health-status-circle" id="overall-health-circle">
                            <div class="health-score" id="overall-health-score">--</div>
                        </div>
                        <div class="health-status-text">
                            <div class="status-label">System Status</div>
                            <div class="status-value" id="overall-health-status">Checking...</div>
                        </div>
                    </div>

                    <div class="health-metrics-quick">
                        <div class="quick-metric">
                            <div class="metric-icon">⚡</div>
                            <div class="metric-info">
                                <div class="metric-label">API Response</div>
                                <div class="metric-value" id="api-response-time">--ms</div>
                            </div>
                        </div>
                        <div class="quick-metric">
                            <div class="metric-icon">🤖</div>
                            <div class="metric-info">
                                <div class="metric-label">ML Models</div>
                                <div class="metric-value" id="ml-models-status">--/--</div>
                            </div>
                        </div>
                        <div class="quick-metric">
                            <div class="metric-icon">💾</div>
                            <div class="metric-info">
                                <div class="metric-label">Database</div>
                                <div class="metric-value" id="database-status">--</div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="health-details" id="health-details-panel" style="display: none;">
                    <div class="health-components">
                        <div class="health-component">
                            <div class="component-header">
                                <div class="component-title">
                                    <span class="component-icon">🌐</span>
                                    API Health
                                </div>
                                <div class="component-status" id="api-component-status">Unknown</div>
                            </div>
                            <div class="component-metrics">
                                <div class="component-metric">
                                    <span class="metric-name">Response Time:</span>
                                    <span class="metric-val" id="api-response-detail">--ms</span>
                                </div>
                                <div class="component-metric">
                                    <span class="metric-name">Error Rate:</span>
                                    <span class="metric-val" id="api-error-rate">--%</span>
                                </div>
                                <div class="component-metric">
                                    <span class="metric-name">Last Check:</span>
                                    <span class="metric-val" id="api-last-check">--</span>
                                </div>
                            </div>
                        </div>

                        <div class="health-component">
                            <div class="component-header">
                                <div class="component-title">
                                    <span class="component-icon">💾</span>
                                    Database
                                </div>
                                <div class="component-status" id="db-component-status">Unknown</div>
                            </div>
                            <div class="component-metrics">
                                <div class="component-metric">
                                    <span class="metric-name">Connection:</span>
                                    <span class="metric-val" id="db-connectivity">--</span>
                                </div>
                                <div class="component-metric">
                                    <span class="metric-name">Query Time:</span>
                                    <span class="metric-val" id="db-query-time">--ms</span>
                                </div>
                                <div class="component-metric">
                                    <span class="metric-name">Tables Status:</span>
                                    <span class="metric-val" id="db-tables-status">--</span>
                                </div>
                            </div>
                        </div>

                        <div class="health-component">
                            <div class="component-header">
                                <div class="component-title">
                                    <span class="component-icon">🤖</span>
                                    ML Models
                                </div>
                                <div class="component-status" id="ml-component-status">Unknown</div>
                            </div>
                            <div class="component-metrics">
                                <div class="component-metric">
                                    <span class="metric-name">Models Loaded:</span>
                                    <span class="metric-val" id="ml-models-loaded">--/--</span>
                                </div>
                                <div class="component-metric">
                                    <span class="metric-name">Prediction Time:</span>
                                    <span class="metric-val" id="ml-prediction-time">--ms</span>
                                </div>
                                <div class="component-metric">
                                    <span class="metric-name">Last Training:</span>
                                    <span class="metric-val" id="ml-last-training">--</span>
                                </div>
                            </div>
                        </div>

                        <div class="health-component">
                            <div class="component-header">
                                <div class="component-title">
                                    <span class="component-icon">📊</span>
                                    Data Collection
                                </div>
                                <div class="component-status" id="data-component-status">Unknown</div>
                            </div>
                            <div class="component-metrics">
                                <div class="component-metric">
                                    <span class="metric-name">API Quotas:</span>
                                    <span class="metric-val" id="data-api-quotas">--</span>
                                </div>
                                <div class="component-metric">
                                    <span class="metric-name">Last Collection:</span>
                                    <span class="metric-val" id="data-last-collection">--</span>
                                </div>
                                <div class="component-metric">
                                    <span class="metric-name">Cache Status:</span>
                                    <span class="metric-val" id="data-cache-status">--</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="health-history">
                        <h4>Health History (Last 24h)</h4>
                        <div class="health-chart-container">
                            <canvas id="health-history-chart" width="400" height="100"></canvas>
                        </div>
                    </div>
                </div>

                <div class="health-alerts" id="health-alerts-container"></div>
            </div>
        `;
    }

    setupEventListeners() {
        // Toggle monitoring
        const toggleBtn = document.getElementById('health-toggle-monitoring');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => this.toggleMonitoring());
        }

        // Manual refresh
        const refreshBtn = document.getElementById('health-manual-refresh');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.performHealthCheck());
        }

        // Toggle details panel
        const detailsBtn = document.getElementById('health-toggle-details');
        if (detailsBtn) {
            detailsBtn.addEventListener('click', () => this.toggleDetailsPanel());
        }
    }

    async performInitialHealthCheck() {
        try {
            await this.performHealthCheck();
        } catch (error) {
            console.error('Initial health check failed:', error);
            this.updateHealthData('overall', { status: 'critical', score: 0 });
        }
    }

    async performHealthCheck() {
        const startTime = Date.now();
        
        try {
            // Check API health
            await this.checkAPIHealth();
            
            // Check database connectivity
            await this.checkDatabaseHealth();
            
            // Check ML models status
            await this.checkMLModelsHealth();
            
            // Check data collection status
            await this.checkDataCollectionHealth();
            
            // Calculate overall health score
            this.calculateOverallHealth();
            
            // Update UI
            this.updateHealthUI();
            
            // Store in history
            this.addToHistory(Date.now(), this.healthData.overall.score);
            
            console.log(`Health check completed in ${Date.now() - startTime}ms`);
            
        } catch (error) {
            console.error('Health check failed:', error);
            this.updateHealthData('overall', { status: 'critical', score: 0 });
            this.updateHealthUI();
        }
    }

    async checkAPIHealth() {
        const startTime = Date.now();
        
        try {
            const response = await fetch(`${this.API_BASE}/health-check`, {
                method: 'GET',
                headers: { 'Accept': 'application/json' },
                cache: 'no-cache'
            });
            
            const responseTime = Date.now() - startTime;
            
            if (response.ok) {
                const data = await response.json();
                this.updateHealthData('api', {
                    status: 'healthy',
                    responseTime: responseTime,
                    errorRate: 0,
                    lastCheck: new Date()
                });
            } else {
                throw new Error(`API returned ${response.status}`);
            }
            
        } catch (error) {
            const responseTime = Date.now() - startTime;
            this.updateHealthData('api', {
                status: 'unhealthy',
                responseTime: responseTime,
                errorRate: 100,
                lastCheck: new Date(),
                error: error.message
            });
        }
    }

    async checkDatabaseHealth() {
        try {
            const response = await fetch(`${this.API_BASE}/system/database-health`, {
                method: 'GET',
                headers: { 'Accept': 'application/json' },
                cache: 'no-cache'
            });
            
            if (response.ok) {
                const data = await response.json();
                const isHealthy = data.status === 'healthy';
                this.updateHealthData('database', {
                    status: isHealthy ? 'healthy' : 'unhealthy',
                    connectivity: isHealthy && data.database?.connected,
                    queryTime: data.database?.query_time || 0,
                    tablesStatus: data.database?.tables_status || 'connected'
                });
            } else {
                throw new Error(`Database check failed: ${response.status}`);
            }
            
        } catch (error) {
            this.updateHealthData('database', {
                status: 'unhealthy',
                connectivity: false,
                error: error.message
            });
        }
    }

    async checkMLModelsHealth() {
        try {
            const response = await fetch(`${this.API_BASE}/system/ml-health`, {
                method: 'GET',
                headers: { 'Accept': 'application/json' },
                cache: 'no-cache'
            });
            
            if (response.ok) {
                const data = await response.json();
                const isHealthy = data.status === 'healthy';
                const mlSystem = data.ml_system || {};
                
                this.updateHealthData('mlModels', {
                    status: isHealthy ? 'healthy' : 'degraded',
                    modelsLoaded: mlSystem.models_count || 0,
                    totalModels: Math.max(mlSystem.models_count || 0, 4), // Expected models
                    predictionTime: 0, // Not provided in current response
                    lastTraining: null, // Not provided in current response
                    predictionTest: mlSystem.prediction_test || 'unknown',
                    realPredictor: mlSystem.real_predictor?.available || false,
                    predictionService: mlSystem.prediction_service?.available || false
                });
            } else {
                throw new Error(`ML health check failed: ${response.status}`);
            }
            
        } catch (error) {
            this.updateHealthData('mlModels', {
                status: 'unhealthy',
                modelsLoaded: 0,
                totalModels: 0,
                error: error.message
            });
        }
    }

    async checkDataCollectionHealth() {
        try {
            const response = await fetch(`${this.API_BASE}/api-economy-status`, {
                method: 'GET',
                headers: { 'Accept': 'application/json' },
                cache: 'no-cache'
            });
            
            if (response.ok) {
                const data = await response.json();
                const usage = data.api_usage || {};
                
                this.updateHealthData('dataCollection', {
                    status: data.success ? 'healthy' : 'degraded',
                    apiQuotas: `${usage.remaining_hour || 0}/${usage.max_per_hour || 0}`,
                    lastCollection: usage.last_update || null,
                    cacheStatus: `${usage.cache_items || 0} items`
                });
            } else {
                throw new Error(`Data collection check failed: ${response.status}`);
            }
            
        } catch (error) {
            this.updateHealthData('dataCollection', {
                status: 'unhealthy',
                error: error.message
            });
        }
    }

    updateHealthData(component, data) {
        this.healthData[component] = { ...this.healthData[component], ...data };
    }

    calculateOverallHealth() {
        const components = ['api', 'database', 'mlModels', 'dataCollection'];
        let totalScore = 0;
        let criticalIssues = 0;
        
        components.forEach(component => {
            const status = this.healthData[component].status;
            let score = 0;
            
            switch (status) {
                case 'healthy':
                    score = 100;
                    break;
                case 'degraded':
                    score = 60;
                    break;
                case 'unhealthy':
                    score = 20;
                    criticalIssues++;
                    break;
                default:
                    score = 0;
                    criticalIssues++;
            }
            
            totalScore += score;
        });
        
        const averageScore = Math.round(totalScore / components.length);
        let overallStatus = 'healthy';
        
        if (criticalIssues > 1 || averageScore < 30) {
            overallStatus = 'critical';
        } else if (criticalIssues > 0 || averageScore < 70) {
            overallStatus = 'degraded';
        }
        
        this.updateHealthData('overall', {
            status: overallStatus,
            score: averageScore
        });
    }

    updateHealthUI() {
        // Update overall health indicator
        const healthCircle = document.getElementById('overall-health-circle');
        const healthScore = document.getElementById('overall-health-score');
        const healthStatus = document.getElementById('overall-health-status');
        
        if (healthCircle && healthScore && healthStatus) {
            const overall = this.healthData.overall;
            
            healthScore.textContent = overall.score;
            healthStatus.textContent = this.formatStatus(overall.status);
            
            // Update circle color based on status
            healthCircle.className = `health-status-circle ${overall.status}`;
        }
        
        // Update quick metrics
        this.updateQuickMetrics();
        
        // Update detailed components
        this.updateDetailedComponents();
        
        // Update health history chart
        this.updateHealthChart();
    }

    updateQuickMetrics() {
        const apiResponseTime = document.getElementById('api-response-time');
        const mlModelsStatus = document.getElementById('ml-models-status');
        const databaseStatus = document.getElementById('database-status');
        
        if (apiResponseTime) {
            apiResponseTime.textContent = `${this.healthData.api.responseTime || 0}ms`;
        }
        
        if (mlModelsStatus) {
            const ml = this.healthData.mlModels;
            mlModelsStatus.textContent = `${ml.modelsLoaded || 0}/${ml.totalModels || 0}`;
        }
        
        if (databaseStatus) {
            databaseStatus.textContent = this.formatStatus(this.healthData.database.status);
        }
    }

    updateDetailedComponents() {
        // Update API component
        this.updateComponent('api', {
            'api-component-status': this.healthData.api.status,
            'api-response-detail': `${this.healthData.api.responseTime || 0}ms`,
            'api-error-rate': `${this.healthData.api.errorRate || 0}%`,
            'api-last-check': this.formatTime(this.healthData.api.lastCheck)
        });
        
        // Update Database component
        this.updateComponent('db', {
            'db-component-status': this.healthData.database.status,
            'db-connectivity': this.healthData.database.connectivity ? 'Connected' : 'Disconnected',
            'db-query-time': `${this.healthData.database.queryTime || 0}ms`,
            'db-tables-status': this.healthData.database.tablesStatus || 'Unknown'
        });
        
        // Update ML Models component
        this.updateComponent('ml', {
            'ml-component-status': this.healthData.mlModels.status,
            'ml-models-loaded': `${this.healthData.mlModels.modelsLoaded || 0}/${this.healthData.mlModels.totalModels || 0}`,
            'ml-prediction-time': `${this.healthData.mlModels.predictionTime || 0}ms`,
            'ml-last-training': this.formatTime(this.healthData.mlModels.lastTraining)
        });
        
        // Update Data Collection component
        this.updateComponent('data', {
            'data-component-status': this.healthData.dataCollection.status,
            'data-api-quotas': this.healthData.dataCollection.apiQuotas || '--',
            'data-last-collection': this.formatTime(this.healthData.dataCollection.lastCollection),
            'data-cache-status': this.healthData.dataCollection.cacheStatus || '--'
        });
    }

    updateComponent(prefix, updates) {
        Object.entries(updates).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
                
                // Update status class for status elements
                if (id.includes('status')) {
                    element.className = `component-status ${value}`;
                }
            }
        });
    }

    addToHistory(timestamp, score) {
        this.healthHistory.push({ timestamp, score });
        
        // Keep only recent history
        if (this.healthHistory.length > this.maxHistoryLength) {
            this.healthHistory.shift();
        }
    }

    updateHealthChart() {
        // Simple text-based chart for now
        // In a real implementation, you might use Chart.js or similar
        const chartContainer = document.querySelector('.health-chart-container');
        if (chartContainer && this.healthHistory.length > 0) {
            const canvas = document.getElementById('health-history-chart');
            if (canvas) {
                this.drawSimpleChart(canvas);
            }
        }
    }

    drawSimpleChart(canvas) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        if (this.healthHistory.length < 2) return;
        
        // Draw background
        ctx.fillStyle = 'rgba(255, 255, 255, 0.05)';
        ctx.fillRect(0, 0, width, height);
        
        // Draw grid lines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = (height / 4) * i;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }
        
        // Draw health score line
        ctx.strokeStyle = '#6bcf7f';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        this.healthHistory.forEach((point, index) => {
            const x = (width / (this.healthHistory.length - 1)) * index;
            const y = height - (height * (point.score / 100));
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
        
        // Draw points
        ctx.fillStyle = '#6bcf7f';
        this.healthHistory.forEach((point, index) => {
            const x = (width / (this.healthHistory.length - 1)) * index;
            const y = height - (height * (point.score / 100));
            
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
            ctx.fill();
        });
    }

    toggleMonitoring() {
        if (this.isMonitoring) {
            this.stopMonitoring();
        } else {
            this.startMonitoring();
        }
    }

    startMonitoring() {
        if (this.isMonitoring) return;
        
        this.isMonitoring = true;
        this.monitoringInterval = setInterval(() => {
            this.performHealthCheck();
        }, 30000); // Check every 30 seconds
        
        // Update toggle button
        const toggleBtn = document.getElementById('health-toggle-monitoring');
        if (toggleBtn) {
            toggleBtn.querySelector('.btn-icon').textContent = '⏸️';
            toggleBtn.querySelector('.btn-text').textContent = 'Pause';
            toggleBtn.title = 'Pause monitoring';
        }
    }

    stopMonitoring() {
        if (!this.isMonitoring) return;
        
        this.isMonitoring = false;
        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval);
            this.monitoringInterval = null;
        }
        
        // Update toggle button
        const toggleBtn = document.getElementById('health-toggle-monitoring');
        if (toggleBtn) {
            toggleBtn.querySelector('.btn-icon').textContent = '▶️';
            toggleBtn.querySelector('.btn-text').textContent = 'Resume';
            toggleBtn.title = 'Resume monitoring';
        }
    }

    toggleDetailsPanel() {
        const detailsPanel = document.getElementById('health-details-panel');
        const toggleBtn = document.getElementById('health-toggle-details');
        
        if (detailsPanel && toggleBtn) {
            const isVisible = detailsPanel.style.display !== 'none';
            detailsPanel.style.display = isVisible ? 'none' : 'block';
            toggleBtn.querySelector('.btn-icon').textContent = isVisible ? '📊' : '📈';
            toggleBtn.title = isVisible ? 'Show details' : 'Hide details';
        }
    }

    formatStatus(status) {
        const statusMap = {
            'healthy': 'Healthy',
            'degraded': 'Degraded',
            'unhealthy': 'Unhealthy',
            'critical': 'Critical',
            'unknown': 'Unknown'
        };
        return statusMap[status] || 'Unknown';
    }

    formatTime(timestamp) {
        if (!timestamp) return '--';
        
        const date = new Date(timestamp);
        if (isNaN(date.getTime())) return '--';
        
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        
        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins}m ago`;
        
        const diffHours = Math.floor(diffMins / 60);
        if (diffHours < 24) return `${diffHours}h ago`;
        
        const diffDays = Math.floor(diffHours / 24);
        return `${diffDays}d ago`;
    }

    destroy() {
        this.stopMonitoring();
        
        // Remove UI elements
        const container = document.getElementById('system-health-container');
        if (container) {
            container.remove();
        }
    }
}

// Global initialization
function initSystemHealthMonitor() {
    try {
        if (!window.systemHealthMonitor) {
            window.systemHealthMonitor = new SystemHealthMonitor();
        }
    } catch (error) {
        console.error('Failed to initialize system health monitor:', error);
    }
}

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Initialize with a small delay to ensure other components are ready
    setTimeout(initSystemHealthMonitor, 1000);
});

// Export for manual initialization
window.SystemHealthMonitor = SystemHealthMonitor;
window.initSystemHealthMonitor = initSystemHealthMonitor;