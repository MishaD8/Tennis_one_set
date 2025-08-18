/**
 * Betting Statistics Dashboard - Comprehensive Performance Analysis
 * Features: Time period filtering, interactive charts, risk analysis
 */

class BettingStatistics {
    constructor() {
        this.API_BASE = window.location.origin + '/api';
        this.currentTimeframe = '1_week';
        this.charts = {};
        this.data = null;
        this.init();
    }

    async init() {
        try {
            this.setupEventListeners();
            await this.loadStatistics();
            this.initializeCharts();
        } catch (error) {
            console.error('Error initializing betting statistics:', error);
            this.showError('Failed to initialize betting statistics dashboard');
        }
    }

    setupEventListeners() {
        // Time period selector buttons
        const periodButtons = document.querySelectorAll('.period-btn');
        periodButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                this.selectTimeframe(btn.dataset.period);
            });
        });
    }

    async selectTimeframe(timeframe) {
        try {
            // Update active button
            document.querySelectorAll('.period-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            document.querySelector(`[data-period="${timeframe}"]`).classList.add('active');

            this.currentTimeframe = timeframe;
            await this.loadStatistics();
            this.updateCharts();
        } catch (error) {
            console.error('Error selecting timeframe:', error);
            this.showError('Failed to load statistics for selected timeframe');
        }
    }

    async loadStatistics() {
        try {
            this.showLoading();

            const response = await fetch(`${this.API_BASE}/betting/statistics?timeframe=${this.currentTimeframe}&test_mode=live`);
            const data = await response.json();

            if (data.success) {
                this.data = data.statistics;
                this.updateMetrics();
                this.updateDataQuality();
            } else {
                throw new Error(data.error || 'Failed to load statistics');
            }
        } catch (error) {
            console.error('Error loading statistics:', error);
            this.showError('Unable to load betting statistics. Please try again later.');
        }
    }

    updateMetrics() {
        try {
            if (!this.data) return;

            const basicMetrics = this.data.basic_metrics || {};
            const financialMetrics = this.data.financial_metrics || {};
            const averageMetrics = this.data.average_metrics || {};
            const riskMetrics = this.data.risk_metrics || {};
            const streakAnalysis = this.data.streak_analysis || {};

            // Update basic metrics
            this.updateElement('stats-total-bets', basicMetrics.total_bets || 0);
            this.updateElement('stats-win-rate', `${basicMetrics.win_rate || 0}%`);

            // Update financial metrics with enhanced calculations
            const netProfit = financialMetrics.net_profit || 0;
            const roi = financialMetrics.roi_percentage || 0;
            
            this.updateElement('stats-net-profit', this.formatCurrency(netProfit), netProfit >= 0 ? 'positive' : 'negative');
            this.updateElement('stats-roi', `${roi.toFixed(2)}%`, roi >= 0 ? 'positive' : 'negative');

            // Calculate and display capital growth scenarios
            this.updateCapitalGrowthScenarios(netProfit, basicMetrics.total_bets || 0, roi);

            // Update average metrics
            this.updateElement('stats-avg-odds', averageMetrics.average_odds?.toFixed(2) || '0.00');

            // Update risk metrics
            this.updateElement('stats-sharpe-ratio', riskMetrics.sharpe_ratio?.toFixed(3) || '0.000');
            this.updateElement('stats-largest-win', this.formatCurrency(riskMetrics.largest_win || 0));
            this.updateElement('stats-largest-loss', this.formatCurrency(riskMetrics.largest_loss || 0));

            // Update streak analysis
            const currentStreak = streakAnalysis.current_streak || {};
            const streakText = currentStreak.count > 0 ? 
                `${currentStreak.count} ${currentStreak.type}` : 'No streak';
            this.updateElement('stats-current-streak', streakText);
            
            this.updateElement('stats-longest-win-streak', streakAnalysis.longest_winning_streak || 0);
            this.updateElement('stats-longest-loss-streak', streakAnalysis.longest_losing_streak || 0);
            this.updateElement('stats-max-drawdown', this.formatCurrency(riskMetrics.max_drawdown || 0));

            // Update match selection criteria display
            this.updateMatchSelectionCriteria();

        } catch (error) {
            console.error('Error updating metrics:', error);
        }
    }

    updateCapitalGrowthScenarios(netProfit, totalBets, roi) {
        try {
            // Calculate growth scenarios with fixed $100 bet
            const fixedBetSize = 100;
            const projectedProfit = totalBets > 0 ? (netProfit / totalBets) * fixedBetSize : 0;
            
            // Calculate different timeframe projections
            const scenarios = {
                daily: projectedProfit,
                weekly: projectedProfit * 7,
                monthly: projectedProfit * 30,
                yearly: projectedProfit * 365
            };

            // Update or create capital growth display
            let growthSection = document.querySelector('.capital-growth-scenarios');
            if (!growthSection) {
                growthSection = this.createCapitalGrowthSection();
                const metricsGrid = document.querySelector('.betting-metrics-grid');
                if (metricsGrid) {
                    metricsGrid.insertAdjacentElement('afterend', growthSection);
                }
            }

            // Update scenario values
            Object.entries(scenarios).forEach(([period, profit]) => {
                const element = document.getElementById(`growth-${period}`);
                if (element) {
                    element.textContent = this.formatCurrency(profit);
                    element.className = `scenario-value ${profit >= 0 ? 'positive' : 'negative'}`;
                }
            });

            // Update confidence level based on sample size and performance
            const confidenceLevel = this.calculateConfidenceLevel(totalBets, roi);
            const confidenceElement = document.getElementById('growth-confidence');
            if (confidenceElement) {
                confidenceElement.textContent = confidenceLevel.label;
                confidenceElement.className = `confidence-level ${confidenceLevel.class}`;
            }

        } catch (error) {
            console.error('Error updating capital growth scenarios:', error);
        }
    }

    createCapitalGrowthSection() {
        const section = document.createElement('div');
        section.className = 'capital-growth-scenarios';
        section.innerHTML = `
            <h3>💵 Capital Growth Scenarios (Fixed $100 Bet)</h3>
            <p class="scenario-description">Projected profits based on current performance metrics</p>
            <div class="growth-scenarios-grid">
                <div class="scenario-card">
                    <div class="scenario-label">Per Bet</div>
                    <div class="scenario-value" id="growth-daily">$0.00</div>
                </div>
                <div class="scenario-card">
                    <div class="scenario-label">Weekly (7 bets)</div>
                    <div class="scenario-value" id="growth-weekly">$0.00</div>
                </div>
                <div class="scenario-card">
                    <div class="scenario-label">Monthly (30 bets)</div>
                    <div class="scenario-value" id="growth-monthly">$0.00</div>
                </div>
                <div class="scenario-card">
                    <div class="scenario-label">Yearly (365 bets)</div>
                    <div class="scenario-value" id="growth-yearly">$0.00</div>
                </div>
            </div>
            <div class="confidence-indicator">
                <span class="confidence-label">Confidence Level:</span>
                <span class="confidence-level" id="growth-confidence">Calculating...</span>
            </div>
        `;
        return section;
    }

    calculateConfidenceLevel(totalBets, roi) {
        if (totalBets === 0) return { label: 'No Data', class: 'no-data' };
        if (totalBets < 10) return { label: 'Very Low (< 10 bets)', class: 'very-low' };
        if (totalBets < 30) return { label: 'Low (< 30 bets)', class: 'low' };
        if (totalBets < 100) return { label: 'Moderate (30-100 bets)', class: 'moderate' };
        if (totalBets < 500) return { label: 'Good (100-500 bets)', class: 'good' };
        return { label: 'High (500+ bets)', class: 'high' };
    }

    updateMatchSelectionCriteria() {
        try {
            // Update or create match selection criteria display
            let criteriaSection = document.querySelector('.match-selection-criteria');
            if (!criteriaSection) {
                criteriaSection = this.createMatchSelectionCriteriaSection();
                const riskAnalysis = document.querySelector('.risk-analysis-section');
                if (riskAnalysis) {
                    riskAnalysis.insertAdjacentElement('beforebegin', criteriaSection);
                }
            }

            // Update criteria based on current data
            const criteria = this.getMatchSelectionCriteria();
            const criteriaList = criteriaSection.querySelector('.criteria-list');
            if (criteriaList) {
                criteriaList.innerHTML = criteria.map(criterion => `
                    <div class="criterion-item">
                        <div class="criterion-icon">${criterion.icon}</div>
                        <div class="criterion-content">
                            <div class="criterion-label">${criterion.label}</div>
                            <div class="criterion-description">${criterion.description}</div>
                        </div>
                        <div class="criterion-status ${criterion.status}">${criterion.value}</div>
                    </div>
                `).join('');
            }

        } catch (error) {
            console.error('Error updating match selection criteria:', error);
        }
    }

    createMatchSelectionCriteriaSection() {
        const section = document.createElement('div');
        section.className = 'match-selection-criteria';
        section.innerHTML = `
            <h3>🎯 Match Selection Criteria & Reasoning</h3>
            <p class="criteria-description">Our systematic approach to identifying value betting opportunities</p>
            <div class="criteria-list"></div>
        `;
        return section;
    }

    getMatchSelectionCriteria() {
        return [
            {
                icon: '📊',
                label: 'Edge Threshold',
                description: 'Minimum edge required for bet consideration',
                value: '> 5%',
                status: 'active'
            },
            {
                icon: '🎲',
                label: 'Odds Range',
                description: 'Preferred odds range for optimal value',
                value: '1.80 - 3.50',
                status: 'active'
            },
            {
                icon: '🏆',
                label: 'Tournament Filter',
                description: 'Focus on ATP/WTA professional events',
                value: 'ATP/WTA Only',
                status: 'active'
            },
            {
                icon: '🤖',
                label: 'ML Confidence',
                description: 'Minimum ML model confidence threshold',
                value: '> 65%',
                status: 'active'
            },
            {
                icon: '📈',
                label: 'Historical Performance',
                description: 'Player recent form and head-to-head',
                value: 'Last 10 matches',
                status: 'active'
            },
            {
                icon: '💰',
                label: 'Kelly Criterion',
                description: 'Maximum stake based on edge and bankroll',
                value: '≤ 5% bankroll',
                status: 'active'
            }
        ];
    }

    updateElement(id, value, className = '') {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
            if (className) {
                element.className = `metric-value ${className}`;
            }
        }
    }

    updateDataQuality() {
        try {
            const dataQuality = this.data?.data_quality;
            const qualityStatus = document.getElementById('data-quality-status');
            
            if (qualityStatus && dataQuality) {
                const sampleSize = dataQuality.sample_size || 0;
                const completeness = dataQuality.data_completeness || 'unknown';
                const significance = dataQuality.statistical_significance || 'unknown';
                const qualityScore = dataQuality.quality_score || 0;
                
                let statusText = '';
                let statusClass = '';
                
                // Enhanced quality assessment
                if (completeness === 'excellent' && qualityScore >= 3) {
                    statusText = `Excellent (${sampleSize} bets, ${significance} significance)`;
                    statusClass = 'quality-excellent';
                } else if (completeness === 'good' && qualityScore >= 2) {
                    statusText = `Good (${sampleSize} bets, ${significance} significance)`;
                    statusClass = 'quality-good';
                } else if (completeness === 'limited' || sampleSize < 30) {
                    statusText = `Limited (${sampleSize} bets, ${significance} significance)`;
                    statusClass = 'quality-limited';
                } else if (completeness === 'very_limited' || sampleSize < 10) {
                    statusText = `Very Limited (${sampleSize} bets)`;
                    statusClass = 'quality-limited';
                } else if (completeness === 'no_data' || sampleSize === 0) {
                    statusText = 'No Data Available';
                    statusClass = 'quality-error';
                } else {
                    statusText = `${sampleSize} bets available`;
                    statusClass = 'quality-good';
                }
                
                qualityStatus.textContent = statusText;
                qualityStatus.className = `quality-status ${statusClass}`;
                
                // Show recommendations if available
                const recommendations = dataQuality.recommendations || [];
                if (recommendations.length > 0) {
                    this.showDataQualityTooltip(recommendations);
                }
            }
        } catch (error) {
            console.error('Error updating data quality:', error);
        }
    }

    async initializeCharts() {
        try {
            await this.createProfitTimelineChart();
            await this.createWinRateTrendChart();
            await this.createOddsDistributionChart();
            await this.createMonthlyPerformanceChart();
        } catch (error) {
            console.error('Error initializing charts:', error);
        }
    }

    async createProfitTimelineChart() {
        try {
            const chartData = await this.getChartData('profit_timeline');
            const ctx = document.getElementById('profit-timeline-chart');
            
            if (!ctx || !chartData) return;

            this.destroyChart('profitTimeline');

            this.charts.profitTimeline = new Chart(ctx, {
                type: 'line',
                data: chartData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: {
                                color: 'rgba(255, 255, 255, 0.9)'
                            }
                        }
                    },
                    scales: {
                        x: {
                            ticks: { color: 'rgba(255, 255, 255, 0.7)' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        },
                        y: {
                            ticks: { color: 'rgba(255, 255, 255, 0.7)' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error creating profit timeline chart:', error);
        }
    }

    async createWinRateTrendChart() {
        try {
            const chartData = await this.getChartData('win_rate_trend');
            const ctx = document.getElementById('win-rate-trend-chart');
            
            if (!ctx || !chartData) return;

            this.destroyChart('winRateTrend');

            this.charts.winRateTrend = new Chart(ctx, {
                type: 'line',
                data: chartData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: {
                                color: 'rgba(255, 255, 255, 0.9)'
                            }
                        }
                    },
                    scales: {
                        x: {
                            ticks: { color: 'rgba(255, 255, 255, 0.7)' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        },
                        y: {
                            min: 0,
                            max: 100,
                            ticks: { 
                                color: 'rgba(255, 255, 255, 0.7)',
                                callback: function(value) {
                                    return value + '%';
                                }
                            },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error creating win rate trend chart:', error);
        }
    }

    async createOddsDistributionChart() {
        try {
            const chartData = await this.getChartData('odds_distribution');
            const ctx = document.getElementById('odds-distribution-chart');
            
            if (!ctx || !chartData) return;

            this.destroyChart('oddsDistribution');

            this.charts.oddsDistribution = new Chart(ctx, {
                type: 'doughnut',
                data: chartData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                color: 'rgba(255, 255, 255, 0.9)',
                                padding: 20
                            }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error creating odds distribution chart:', error);
        }
    }

    async createMonthlyPerformanceChart() {
        try {
            const chartData = await this.getChartData('monthly_performance');
            const ctx = document.getElementById('monthly-performance-chart');
            
            if (!ctx || !chartData) return;

            this.destroyChart('monthlyPerformance');

            this.charts.monthlyPerformance = new Chart(ctx, {
                type: 'bar',
                data: chartData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: {
                                color: 'rgba(255, 255, 255, 0.9)'
                            }
                        }
                    },
                    scales: {
                        x: {
                            ticks: { color: 'rgba(255, 255, 255, 0.7)' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        },
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            ticks: { color: 'rgba(255, 255, 255, 0.7)' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            ticks: { 
                                color: 'rgba(255, 255, 255, 0.7)',
                                callback: function(value) {
                                    return value + '%';
                                }
                            },
                            grid: {
                                drawOnChartArea: false,
                            },
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error creating monthly performance chart:', error);
        }
    }

    async getChartData(chartType) {
        try {
            const response = await fetch(`${this.API_BASE}/betting/charts-data?timeframe=${this.currentTimeframe}&chart_type=${chartType}&test_mode=live`);
            const data = await response.json();
            
            if (data.success) {
                // Check data quality for chart generation
                const chartData = data.data;
                if (chartData.data_quality) {
                    const sampleSize = chartData.data_quality.sample_size || 0;
                    if (sampleSize === 0) {
                        console.warn(`No data available for ${chartType} chart in ${this.currentTimeframe} timeframe`);
                    } else if (sampleSize < 10) {
                        console.warn(`Limited data (${sampleSize} bets) for ${chartType} chart - results may not be reliable`);
                    }
                }
                return chartData;
            } else {
                throw new Error(data.error || 'Failed to load chart data');
            }
        } catch (error) {
            console.error(`Error getting chart data for ${chartType}:`, error);
            return {
                labels: ['No Data'],
                datasets: [{
                    label: 'Chart data unavailable',
                    data: [0],
                    backgroundColor: 'rgba(255, 107, 107, 0.7)',
                    borderColor: '#ff6b6b'
                }],
                error: error.message
            };
        }
    }

    async updateCharts() {
        try {
            await this.createProfitTimelineChart();
            await this.createWinRateTrendChart();
            await this.createOddsDistributionChart();
            await this.createMonthlyPerformanceChart();
        } catch (error) {
            console.error('Error updating charts:', error);
        }
    }

    destroyChart(chartName) {
        if (this.charts[chartName]) {
            this.charts[chartName].destroy();
            delete this.charts[chartName];
        }
    }

    formatCurrency(amount) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(amount);
    }

    showLoading() {
        // Update all metric values to show loading state
        const metricElements = document.querySelectorAll('.metric-value');
        metricElements.forEach(element => {
            element.textContent = '...';
            element.className = 'metric-value loading';
        });

        const dataQualityStatus = document.getElementById('data-quality-status');
        if (dataQualityStatus) {
            dataQualityStatus.textContent = 'Loading...';
        }
    }

    showDataQualityTooltip(recommendations) {
        try {
            const indicator = document.getElementById('data-quality-indicator');
            if (!indicator) return;
            
            // Remove existing tooltip
            const existingTooltip = indicator.querySelector('.data-quality-tooltip');
            if (existingTooltip) {
                existingTooltip.remove();
            }
            
            // Create tooltip with recommendations
            const tooltip = document.createElement('div');
            tooltip.className = 'data-quality-tooltip';
            tooltip.innerHTML = `
                <div class="tooltip-content">
                    <h4>Data Quality Recommendations:</h4>
                    <ul>
                        ${recommendations.map(rec => `<li>${rec}</li>`).join('')}
                    </ul>
                </div>
            `;
            
            indicator.appendChild(tooltip);
            
            // Auto-hide after 10 seconds
            setTimeout(() => {
                if (tooltip.parentNode) {
                    tooltip.remove();
                }
            }, 10000);
            
        } catch (error) {
            console.error('Error showing data quality tooltip:', error);
        }
    }

    showError(message) {
        // Show error in data quality indicator
        const dataQualityStatus = document.getElementById('data-quality-status');
        if (dataQualityStatus) {
            dataQualityStatus.textContent = 'Error loading data';
            dataQualityStatus.className = 'quality-status quality-error';
        }

        // Show notification if available
        if (window.tennisDashboard && typeof window.tennisDashboard.showNotification === 'function') {
            window.tennisDashboard.showNotification('Betting Statistics Error', message, 'error');
        } else {
            console.error('Betting Statistics Error:', message);
        }
    }

    destroy() {
        // Clean up charts
        Object.keys(this.charts).forEach(chartName => {
            this.destroyChart(chartName);
        });
    }
}

// Global initialization function
function initBettingStatistics() {
    try {
        if (!window.bettingStatistics) {
            window.bettingStatistics = new BettingStatistics();
        }
    } catch (error) {
        console.error('Failed to initialize betting statistics:', error);
    }
}

// Auto-initialize if in betting statistics tab
document.addEventListener('DOMContentLoaded', () => {
    const bettingStatsTab = document.getElementById('betting-statistics');
    if (bettingStatsTab && !bettingStatsTab.hasAttribute('aria-hidden')) {
        initBettingStatistics();
    }
});