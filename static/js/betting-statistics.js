/**
 * Betting Statistics Dashboard - Enhanced Performance Analysis
 * Features: Time period filtering, interactive charts, risk analysis,
 * enhanced error handling, loading states, and mobile responsiveness
 */

class BettingStatistics {
    constructor() {
        // Use current host and port for API base to avoid localhost issues
        this.API_BASE = `${window.location.protocol}//${window.location.host}/api`;
        this.currentTimeframe = '1_week';
        this.charts = {};
        this.data = null;
        this.isLoading = false;
        this.retryCount = 0;
        this.maxRetries = 3;
        this.loadingSteps = [
            { id: 'fetch', text: 'Fetching betting data...', icon: '📊' },
            { id: 'process', text: 'Processing statistics...', icon: '🔄' },
            { id: 'charts', text: 'Generating charts...', icon: '📈' },
            { id: 'complete', text: 'Analysis complete!', icon: '✅' }
        ];
        this.currentStep = 0;
        this.init();
    }

    async init() {
        try {
            this.setupEventListeners();
            this.setupResizeHandler();
            await this.loadStatistics();
            this.initializeCharts();
            this.setupDataRefresh();
        } catch (error) {
            console.error('Error initializing betting statistics:', error);
            this.showError('Failed to initialize betting statistics dashboard');
            this.showRetryOption();
        }
    }

    setupResizeHandler() {
        // Debounced resize handler for responsive charts
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                this.resizeCharts();
            }, 300);
        });
    }

    setupDataRefresh() {
        // Auto-refresh data every 5 minutes
        setInterval(async () => {
            if (document.getElementById('betting-statistics').classList.contains('active')) {
                await this.refreshData();
            }
        }, 5 * 60 * 1000);
    }

    async refreshData(silent = true) {
        try {
            if (!silent) this.showLoading();
            await this.loadStatistics();
            this.updateCharts();
            if (!silent) {
                this.showNotification('Data refreshed successfully', 'success');
            }
        } catch (error) {
            console.error('Error refreshing data:', error);
            if (!silent) {
                this.showError('Failed to refresh data');
            }
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
            if (this.isLoading) return; // Prevent multiple concurrent requests
            
            // Add smooth transition effect
            this.addTransitionEffect();
            
            // Update active button with animation
            const currentActive = document.querySelector('.period-btn.active');
            const newActive = document.querySelector(`[data-period="${timeframe}"]`);
            
            if (currentActive) currentActive.classList.remove('active');
            if (newActive) {
                newActive.classList.add('active');
                this.animateButtonSelection(newActive);
            }

            this.currentTimeframe = timeframe;
            await this.loadStatistics();
            this.updateCharts();
            
            // Track selection for analytics
            this.trackTimeframeSelection(timeframe);
        } catch (error) {
            console.error('Error selecting timeframe:', error);
            this.showError('Failed to load statistics for selected timeframe');
        }
    }

    addTransitionEffect() {
        const container = document.querySelector('.betting-statistics-dashboard');
        if (container) {
            container.style.opacity = '0.7';
            container.style.transition = 'opacity 0.3s ease';
            setTimeout(() => {
                container.style.opacity = '1';
            }, 300);
        }
    }

    animateButtonSelection(button) {
        button.style.transform = 'scale(0.95)';
        button.style.transition = 'transform 0.2s ease';
        setTimeout(() => {
            button.style.transform = 'scale(1)';
        }, 200);
    }

    trackTimeframeSelection(timeframe) {
        // Analytics tracking (non-invasive)
        if (window.gtag) {
            window.gtag('event', 'timeframe_selected', {
                event_category: 'betting_statistics',
                event_label: timeframe
            });
        }
    }

    async loadStatistics() {
        try {
            if (this.isLoading) return;
            this.isLoading = true;
            this.retryCount = 0;
            this.currentStep = 0;
            
            this.showEnhancedLoading();
            this.updateLoadingStep('fetch');

            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

            try {
                const response = await fetch(
                    `${this.API_BASE}/betting/statistics?timeframe=${this.currentTimeframe}&test_mode=live`,
                    { signal: controller.signal }
                );
                clearTimeout(timeoutId);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                this.updateLoadingStep('process');
                const data = await response.json();

                if (data.success) {
                    this.data = data.statistics;
                    this.updateLoadingStep('charts');
                    await this.updateMetrics();
                    this.updateDataQuality();
                    this.updateLoadingStep('complete');
                    
                    // Show success state briefly
                    setTimeout(() => {
                        this.hideLoading();
                        this.showSuccessBanner();
                    }, 500);
                } else {
                    throw new Error(data.error || 'Failed to load statistics');
                }
            } catch (error) {
                clearTimeout(timeoutId);
                throw error;
            }
        } catch (error) {
            console.error('Error loading statistics:', error);
            this.handleLoadingError(error);
        } finally {
            this.isLoading = false;
        }
    }

    handleLoadingError(error) {
        this.hideLoading();
        
        if (error.name === 'AbortError') {
            this.showError('Request timed out. Please check your connection and try again.', true);
        } else if (error.message.includes('HTTP 5')) {
            this.showError('Server error. Our team has been notified.', true);
        } else if (error.message.includes('HTTP 4')) {
            this.showError('Invalid request. Please refresh the page.', true);
        } else if (!navigator.onLine) {
            this.showError('No internet connection. Please check your network.', true);
        } else {
            this.showError('Unable to load betting statistics. Please try again later.', true);
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
            const sharpeRatio = riskMetrics.sharpe_ratio || riskMetrics.risk_adjusted_return || 0;
            this.updateElement('stats-sharpe-ratio', sharpeRatio.toFixed(3));
            this.updateElement('stats-largest-win', this.formatCurrency(riskMetrics.largest_win || riskMetrics.best_bet || 0));
            this.updateElement('stats-largest-loss', this.formatCurrency(riskMetrics.largest_loss || riskMetrics.worst_bet || 0));

            // Update streak analysis
            const currentStreak = streakAnalysis.current_streak || {};
            const streakText = currentStreak.count > 0 ? 
                `${currentStreak.count} ${currentStreak.type}` : 'No streak';
            this.updateElement('stats-current-streak', streakText);
            
            this.updateElement('stats-longest-win-streak', streakAnalysis.longest_winning_streak || 0);
            this.updateElement('stats-longest-loss-streak', streakAnalysis.longest_losing_streak || 0);
            this.updateElement('stats-max-drawdown', this.formatCurrency(riskMetrics.max_drawdown || riskMetrics.maximum_drawdown || 0));

            // Update match selection criteria display
            this.updateMatchSelectionCriteria();
            
            // Cache updated data
            this.cacheData();

        } catch (error) {
            console.error('Error updating metrics:', error);
            this.announceError('Failed to update statistics display');
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
            // Add smooth transition effect
            element.style.transition = 'all 0.3s ease';
            
            // Update value with animation
            element.style.opacity = '0.5';
            
            setTimeout(() => {
                element.textContent = value;
                if (className) {
                    element.className = `metric-value ${className}`;
                }
                element.style.opacity = '1';
                
                // Add brief highlight effect for positive/negative values
                if (className === 'positive' || className === 'negative') {
                    element.style.transform = 'scale(1.05)';
                    setTimeout(() => {
                        element.style.transform = 'scale(1)';
                    }, 200);
                }
            }, 150);
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
            
            if (!ctx || !chartData) {
                this.showChartError('profit-timeline-chart', 'Profit Timeline');
                return;
            }

            this.destroyChart('profitTimeline');

            // Enhanced chart configuration with mobile responsiveness
            const isMobile = window.innerWidth < 768;
            
            this.charts.profitTimeline = new Chart(ctx, {
                type: 'line',
                data: {
                    ...chartData,
                    datasets: chartData.datasets?.map(dataset => ({
                        ...dataset,
                        borderColor: dataset.borderColor || '#6bcf7f',
                        backgroundColor: dataset.backgroundColor || 'rgba(107, 207, 127, 0.1)',
                        borderWidth: isMobile ? 2 : 3,
                        pointRadius: isMobile ? 3 : 4,
                        pointHoverRadius: isMobile ? 5 : 6,
                        tension: 0.4,
                        fill: true
                    })) || []
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    },
                    plugins: {
                        legend: {
                            display: !isMobile,
                            position: isMobile ? 'bottom' : 'top',
                            labels: {
                                color: 'rgba(255, 255, 255, 0.9)',
                                font: {
                                    size: isMobile ? 11 : 12
                                },
                                usePointStyle: true,
                                padding: isMobile ? 10 : 15
                            }
                        },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.9)',
                            titleColor: '#6bcf7f',
                            bodyColor: 'rgba(255, 255, 255, 0.9)',
                            borderColor: '#6bcf7f',
                            borderWidth: 1,
                            cornerRadius: 8,
                            titleFont: { size: isMobile ? 12 : 14 },
                            bodyFont: { size: isMobile ? 11 : 13 },
                            callbacks: {
                                label: (context) => {
                                    const value = context.parsed.y;
                                    return `Profit: ${this.formatCurrency(value)}`;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            ticks: { 
                                color: 'rgba(255, 255, 255, 0.7)',
                                font: { size: isMobile ? 10 : 12 },
                                maxTicksLimit: isMobile ? 5 : 10
                            },
                            grid: { 
                                color: 'rgba(255, 255, 255, 0.1)',
                                display: !isMobile
                            }
                        },
                        y: {
                            ticks: { 
                                color: 'rgba(255, 255, 255, 0.7)',
                                font: { size: isMobile ? 10 : 12 },
                                callback: (value) => this.formatCurrency(value)
                            },
                            grid: { 
                                color: 'rgba(255, 255, 255, 0.1)' 
                            },
                            beginAtZero: false
                        }
                    },
                    animation: {
                        duration: 1000,
                        easing: 'easeInOutCubic'
                    }
                }
            });
            
            this.addChartLoadingAnimation(ctx);
        } catch (error) {
            console.error('Error creating profit timeline chart:', error);
            this.showChartError('profit-timeline-chart', 'Profit Timeline');
        }
    }

    async createWinRateTrendChart() {
        try {
            const chartData = await this.getChartData('win_rate_trend');
            const ctx = document.getElementById('win-rate-trend-chart');
            
            if (!ctx || !chartData) {
                this.showChartError('win-rate-trend-chart', 'Win Rate Trend');
                return;
            }

            this.destroyChart('winRateTrend');

            const isMobile = window.innerWidth < 768;

            this.charts.winRateTrend = new Chart(ctx, {
                type: 'line',
                data: {
                    ...chartData,
                    datasets: chartData.datasets?.map(dataset => ({
                        ...dataset,
                        borderColor: dataset.borderColor || '#4a9eff',
                        backgroundColor: dataset.backgroundColor || 'rgba(74, 158, 255, 0.1)',
                        borderWidth: isMobile ? 2 : 3,
                        pointRadius: isMobile ? 3 : 4,
                        pointHoverRadius: isMobile ? 5 : 6,
                        tension: 0.4,
                        fill: true
                    })) || []
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    },
                    plugins: {
                        legend: {
                            display: !isMobile,
                            position: isMobile ? 'bottom' : 'top',
                            labels: {
                                color: 'rgba(255, 255, 255, 0.9)',
                                font: {
                                    size: isMobile ? 11 : 12
                                },
                                usePointStyle: true
                            }
                        },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.9)',
                            titleColor: '#4a9eff',
                            bodyColor: 'rgba(255, 255, 255, 0.9)',
                            borderColor: '#4a9eff',
                            borderWidth: 1,
                            cornerRadius: 8,
                            callbacks: {
                                label: (context) => {
                                    const value = context.parsed.y;
                                    return `Win Rate: ${value.toFixed(1)}%`;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            ticks: { 
                                color: 'rgba(255, 255, 255, 0.7)',
                                font: { size: isMobile ? 10 : 12 },
                                maxTicksLimit: isMobile ? 5 : 10
                            },
                            grid: { 
                                color: 'rgba(255, 255, 255, 0.1)',
                                display: !isMobile
                            }
                        },
                        y: {
                            min: 0,
                            max: 100,
                            ticks: { 
                                color: 'rgba(255, 255, 255, 0.7)',
                                font: { size: isMobile ? 10 : 12 },
                                callback: function(value) {
                                    return value + '%';
                                }
                            },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        }
                    },
                    animation: {
                        duration: 1000,
                        easing: 'easeInOutCubic'
                    }
                }
            });
            
            this.addChartLoadingAnimation(ctx);
        } catch (error) {
            console.error('Error creating win rate trend chart:', error);
            this.showChartError('win-rate-trend-chart', 'Win Rate Trend');
        }
    }

    async createOddsDistributionChart() {
        try {
            const chartData = await this.getChartData('odds_distribution');
            const ctx = document.getElementById('odds-distribution-chart');
            
            if (!ctx || !chartData) {
                this.showChartError('odds-distribution-chart', 'Odds Distribution');
                return;
            }

            this.destroyChart('oddsDistribution');

            const isMobile = window.innerWidth < 768;
            
            // Enhanced color palette for better visibility
            const colors = [
                '#6bcf7f', '#4a9eff', '#ff6b6b', '#ffa500', 
                '#9c88ff', '#ff8c94', '#a8e6cf', '#88d8c0'
            ];

            this.charts.oddsDistribution = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    ...chartData,
                    datasets: chartData.datasets?.map((dataset, index) => ({
                        ...dataset,
                        backgroundColor: colors,
                        borderColor: colors.map(color => color + '40'),
                        borderWidth: 2,
                        hoverBorderWidth: 3,
                        hoverBackgroundColor: colors.map(color => color + 'CC')
                    })) || []
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: isMobile ? '50%' : '60%',
                    plugins: {
                        legend: {
                            position: isMobile ? 'bottom' : 'right',
                            labels: {
                                color: 'rgba(255, 255, 255, 0.9)',
                                padding: isMobile ? 15 : 20,
                                font: {
                                    size: isMobile ? 11 : 12
                                },
                                usePointStyle: true,
                                pointStyle: 'circle'
                            }
                        },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.9)',
                            titleColor: '#6bcf7f',
                            bodyColor: 'rgba(255, 255, 255, 0.9)',
                            borderColor: '#6bcf7f',
                            borderWidth: 1,
                            cornerRadius: 8,
                            callbacks: {
                                label: (context) => {
                                    const label = context.label || '';
                                    const value = context.parsed;
                                    const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                    const percentage = ((value / total) * 100).toFixed(1);
                                    return `${label}: ${value} bets (${percentage}%)`;
                                }
                            }
                        }
                    },
                    animation: {
                        duration: 1500,
                        easing: 'easeInOutCubic'
                    }
                }
            });
            
            this.addChartLoadingAnimation(ctx);
        } catch (error) {
            console.error('Error creating odds distribution chart:', error);
            this.showChartError('odds-distribution-chart', 'Odds Distribution');
        }
    }

    async createMonthlyPerformanceChart() {
        try {
            const chartData = await this.getChartData('monthly_performance');
            const ctx = document.getElementById('monthly-performance-chart');
            
            if (!ctx || !chartData) {
                this.showChartError('monthly-performance-chart', 'Monthly Performance');
                return;
            }

            this.destroyChart('monthlyPerformance');

            const isMobile = window.innerWidth < 768;

            this.charts.monthlyPerformance = new Chart(ctx, {
                type: 'bar',
                data: {
                    ...chartData,
                    datasets: chartData.datasets?.map((dataset, index) => {
                        if (dataset.type === 'line') {
                            return {
                                ...dataset,
                                borderColor: '#4a9eff',
                                backgroundColor: 'rgba(74, 158, 255, 0.2)',
                                borderWidth: 3,
                                pointRadius: isMobile ? 3 : 4,
                                pointHoverRadius: isMobile ? 5 : 6,
                                tension: 0.4
                            };
                        } else {
                            return {
                                ...dataset,
                                backgroundColor: dataset.data?.map(value => 
                                    value >= 0 ? 'rgba(107, 207, 127, 0.8)' : 'rgba(255, 107, 107, 0.8)'
                                ) || 'rgba(107, 207, 127, 0.8)',
                                borderColor: dataset.data?.map(value => 
                                    value >= 0 ? '#6bcf7f' : '#ff6b6b'
                                ) || '#6bcf7f',
                                borderWidth: 2,
                                borderRadius: isMobile ? 4 : 6,
                                borderSkipped: false
                            };
                        }
                    }) || []
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    },
                    plugins: {
                        legend: {
                            display: !isMobile,
                            position: 'top',
                            labels: {
                                color: 'rgba(255, 255, 255, 0.9)',
                                font: {
                                    size: isMobile ? 11 : 12
                                },
                                usePointStyle: true
                            }
                        },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.9)',
                            titleColor: '#6bcf7f',
                            bodyColor: 'rgba(255, 255, 255, 0.9)',
                            borderColor: '#6bcf7f',
                            borderWidth: 1,
                            cornerRadius: 8,
                            callbacks: {
                                label: (context) => {
                                    const dataset = context.dataset;
                                    const value = context.parsed.y;
                                    
                                    if (dataset.type === 'line') {
                                        return `Win Rate: ${value.toFixed(1)}%`;
                                    } else {
                                        return `Profit: ${this.formatCurrency(value)}`;
                                    }
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            ticks: { 
                                color: 'rgba(255, 255, 255, 0.7)',
                                font: { size: isMobile ? 10 : 12 },
                                maxRotation: isMobile ? 45 : 0
                            },
                            grid: { 
                                color: 'rgba(255, 255, 255, 0.1)',
                                display: false
                            }
                        },
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            ticks: { 
                                color: 'rgba(255, 255, 255, 0.7)',
                                font: { size: isMobile ? 10 : 12 },
                                callback: (value) => this.formatCurrency(value)
                            },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        },
                        y1: {
                            type: 'linear',
                            display: !isMobile,
                            position: 'right',
                            ticks: { 
                                color: 'rgba(255, 255, 255, 0.7)',
                                font: { size: isMobile ? 10 : 12 },
                                callback: function(value) {
                                    return value.toFixed(1) + '%';
                                }
                            },
                            grid: {
                                drawOnChartArea: false,
                            },
                        }
                    },
                    animation: {
                        duration: 1200,
                        easing: 'easeInOutCubic'
                    }
                }
            });
            
            this.addChartLoadingAnimation(ctx);
        } catch (error) {
            console.error('Error creating monthly performance chart:', error);
            this.showChartError('monthly-performance-chart', 'Monthly Performance');
        }
    }

    async getChartData(chartType) {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 15000);
            
            const response = await fetch(
                `${this.API_BASE}/betting/charts-data?timeframe=${this.currentTimeframe}&chart_type=${chartType}&test_mode=live`,
                { signal: controller.signal }
            );
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                const chartData = data.data;
                
                // Enhanced data quality check
                if (chartData.data_quality) {
                    const sampleSize = chartData.data_quality.sample_size || 0;
                    const quality = chartData.data_quality.quality_score || 0;
                    
                    if (sampleSize === 0) {
                        console.warn(`No data available for ${chartType} chart in ${this.currentTimeframe} timeframe`);
                        return this.getEmptyChartData(chartType, 'No data available for selected period');
                    } else if (sampleSize < 5) {
                        console.warn(`Very limited data (${sampleSize} bets) for ${chartType} chart`);
                        chartData.warning = `Based on ${sampleSize} bets - results may not be reliable`;
                    } else if (sampleSize < 20) {
                        console.warn(`Limited data (${sampleSize} bets) for ${chartType} chart`);
                        chartData.warning = `Based on ${sampleSize} bets - trends may not be statistically significant`;
                    }
                }
                
                return chartData;
            } else {
                throw new Error(data.error || 'Failed to load chart data');
            }
        } catch (error) {
            console.error(`Error getting chart data for ${chartType}:`, error);
            
            if (error.name === 'AbortError') {
                return this.getEmptyChartData(chartType, 'Request timed out');
            } else {
                return this.getEmptyChartData(chartType, 'Data temporarily unavailable');
            }
        }
    }

    getEmptyChartData(chartType, message) {
        const isMobile = window.innerWidth < 768;
        
        // Return appropriate empty chart structure based on chart type
        if (chartType === 'odds_distribution') {
            return {
                labels: ['No Data'],
                datasets: [{
                    label: message,
                    data: [1],
                    backgroundColor: ['rgba(255, 107, 107, 0.3)'],
                    borderColor: ['#ff6b6b'],
                    borderWidth: 2
                }],
                empty: true,
                message
            };
        } else {
            return {
                labels: ['No Data'],
                datasets: [{
                    label: message,
                    data: [0],
                    backgroundColor: 'rgba(255, 107, 107, 0.3)',
                    borderColor: '#ff6b6b',
                    borderWidth: 2,
                    pointRadius: isMobile ? 3 : 4,
                    tension: 0.4
                }],
                empty: true,
                message
            };
        }
    }

    showChartError(chartId, chartName) {
        const chartContainer = document.getElementById(chartId);
        if (!chartContainer) return;
        
        const container = chartContainer.parentElement;
        if (!container) return;
        
        // Create error display
        const errorDiv = document.createElement('div');
        errorDiv.className = 'chart-error-display';
        errorDiv.innerHTML = `
            <div class="chart-error-content">
                <div class="error-icon">📈</div>
                <h4>Chart Unavailable</h4>
                <p>${chartName} data could not be loaded</p>
                <button class="btn btn-sm btn-primary" onclick="window.bettingStatistics.retryChart('${chartId}', '${chartName}')">
                    🔄 Retry
                </button>
            </div>
        `;
        
        // Hide canvas and show error
        chartContainer.style.display = 'none';
        container.appendChild(errorDiv);
    }

    async retryChart(chartId, chartName) {
        const errorDisplay = document.querySelector('.chart-error-display');
        if (errorDisplay) {
            errorDisplay.remove();
        }
        
        const canvas = document.getElementById(chartId);
        if (canvas) {
            canvas.style.display = 'block';
        }
        
        // Retry chart creation based on chart ID
        try {
            switch (chartId) {
                case 'profit-timeline-chart':
                    await this.createProfitTimelineChart();
                    break;
                case 'win-rate-trend-chart':
                    await this.createWinRateTrendChart();
                    break;
                case 'odds-distribution-chart':
                    await this.createOddsDistributionChart();
                    break;
                case 'monthly-performance-chart':
                    await this.createMonthlyPerformanceChart();
                    break;
            }
        } catch (error) {
            console.error(`Error retrying chart ${chartName}:`, error);
            this.showChartError(chartId, chartName);
        }
    }

    addChartLoadingAnimation(ctx) {
        // Add subtle loading animation to newly created charts
        if (ctx && ctx.parentElement) {
            ctx.parentElement.style.opacity = '0';
            ctx.parentElement.style.transition = 'opacity 0.5s ease';
            
            setTimeout(() => {
                ctx.parentElement.style.opacity = '1';
            }, 100);
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

    showEnhancedLoading() {
        // Create or update enhanced loading display
        const existingLoading = document.getElementById('enhanced-loading-container');
        if (existingLoading) {
            existingLoading.remove();
        }

        const loadingContainer = document.createElement('div');
        loadingContainer.id = 'enhanced-loading-container';
        loadingContainer.className = 'enhanced-loading';
        loadingContainer.innerHTML = `
            <div class="loading-header">
                <div class="loading-spinner"></div>
                <h3>📊 Loading Betting Statistics</h3>
                <p>Analyzing your betting performance data...</p>
            </div>
            <div class="loading-progress">
                <div class="progress-bar">
                    <div class="progress-fill" id="loading-progress-fill"></div>
                </div>
                <div class="progress-text" id="loading-progress-text">Initializing...</div>
            </div>
            <div class="loading-steps" id="loading-steps">
                ${this.loadingSteps.map((step, index) => `
                    <div class="loading-step" id="step-${step.id}">
                        <div class="step-icon">${step.icon}</div>
                        <div class="step-text">${step.text}</div>
                    </div>
                `).join('')}
            </div>
        `;

        const dashboard = document.querySelector('.betting-statistics-dashboard');
        if (dashboard) {
            dashboard.insertBefore(loadingContainer, dashboard.firstChild);
        }
    }

    updateLoadingStep(stepId) {
        const stepIndex = this.loadingSteps.findIndex(step => step.id === stepId);
        if (stepIndex === -1) return;

        this.currentStep = stepIndex;
        const progressPercent = ((stepIndex + 1) / this.loadingSteps.length) * 100;
        
        const progressFill = document.getElementById('loading-progress-fill');
        const progressText = document.getElementById('loading-progress-text');
        
        if (progressFill) {
            progressFill.style.width = `${progressPercent}%`;
        }
        
        if (progressText) {
            progressText.textContent = this.loadingSteps[stepIndex].text;
        }

        // Update step states
        this.loadingSteps.forEach((step, index) => {
            const stepElement = document.getElementById(`step-${step.id}`);
            if (stepElement) {
                stepElement.classList.remove('active', 'completed');
                if (index < stepIndex) {
                    stepElement.classList.add('completed');
                } else if (index === stepIndex) {
                    stepElement.classList.add('active');
                }
            }
        });
    }

    hideLoading() {
        const loadingContainer = document.getElementById('enhanced-loading-container');
        if (loadingContainer) {
            loadingContainer.style.opacity = '0';
            loadingContainer.style.transition = 'opacity 0.3s ease';
            setTimeout(() => {
                if (loadingContainer.parentNode) {
                    loadingContainer.remove();
                }
            }, 300);
        }
        
        // Remove loading state from metrics
        const metricElements = document.querySelectorAll('.metric-value.loading');
        metricElements.forEach(element => {
            element.classList.remove('loading');
        });
    }

    showSuccessBanner() {
        const existingBanner = document.querySelector('.success-banner');
        if (existingBanner) {
            existingBanner.remove();
        }

        if (!this.data) return;

        const banner = document.createElement('div');
        banner.className = 'success-banner';
        banner.innerHTML = `
            <div class="success-header">
                <h2>✅ Statistics Loaded Successfully</h2>
            </div>
            <div class="success-metrics">
                <div class="success-metric">
                    <span class="metric-icon">📊</span>
                    <span class="metric-text">${this.data.basic_metrics?.total_bets || 0} bets analyzed</span>
                </div>
                <div class="success-metric">
                    <span class="metric-icon">📈</span>
                    <span class="metric-text">${this.data.basic_metrics?.win_rate || 0}% win rate</span>
                </div>
                <div class="success-metric">
                    <span class="metric-icon">💰</span>
                    <span class="metric-text">${this.formatCurrency(this.data.financial_metrics?.net_profit || 0)} net profit</span>
                </div>
            </div>
            <div class="data-freshness-indicator">
                <div class="freshness-status live">
                    <div class="status-dot"></div>
                    <span class="status-text">Live Data</span>
                </div>
                <div class="next-update-info">
                    Next update in: <span id="next-update-countdown">5:00</span>
                </div>
            </div>
        `;

        const dashboard = document.querySelector('.betting-statistics-dashboard');
        if (dashboard && dashboard.firstChild) {
            dashboard.insertBefore(banner, dashboard.firstChild);
            
            // Start countdown timer
            this.startUpdateCountdown();
            
            // Auto-hide banner after 8 seconds
            setTimeout(() => {
                if (banner.parentNode) {
                    banner.style.opacity = '0';
                    banner.style.transition = 'opacity 0.5s ease';
                    setTimeout(() => {
                        if (banner.parentNode) {
                            banner.remove();
                        }
                    }, 500);
                }
            }, 8000);
        }
    }

    startUpdateCountdown() {
        let seconds = 300; // 5 minutes
        const countdownElement = document.getElementById('next-update-countdown');
        
        const countdown = setInterval(() => {
            if (!countdownElement || !document.contains(countdownElement)) {
                clearInterval(countdown);
                return;
            }
            
            const minutes = Math.floor(seconds / 60);
            const remainingSeconds = seconds % 60;
            countdownElement.textContent = `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
            
            seconds--;
            
            if (seconds < 0) {
                clearInterval(countdown);
                countdownElement.textContent = 'Updating...';
            }
        }, 1000);
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

    showError(message, showRetry = false) {
        // Hide loading first
        this.hideLoading();
        
        // Show error in data quality indicator
        const dataQualityStatus = document.getElementById('data-quality-status');
        if (dataQualityStatus) {
            dataQualityStatus.textContent = 'Error loading data';
            dataQualityStatus.className = 'quality-status quality-error';
        }

        // Create comprehensive error display
        this.showErrorBanner(message, showRetry);

        // Show notification if available
        if (window.tennisDashboard && typeof window.tennisDashboard.showNotification === 'function') {
            window.tennisDashboard.showNotification('Betting Statistics Error', message, 'error');
        } else {
            console.error('Betting Statistics Error:', message);
        }
    }

    showErrorBanner(message, showRetry) {
        const existingError = document.querySelector('.error-banner');
        if (existingError) {
            existingError.remove();
        }

        const errorBanner = document.createElement('div');
        errorBanner.className = 'error-banner';
        errorBanner.innerHTML = `
            <div class="error-content">
                <div class="error-header">
                    <h3>❌ Error Loading Statistics</h3>
                    <p>${message}</p>
                </div>
                ${showRetry ? `
                    <div class="error-actions">
                        <button class="btn btn-primary" onclick="window.bettingStatistics.retryLoad()">🔄 Retry</button>
                        <button class="btn btn-secondary" onclick="window.bettingStatistics.showOfflineMode()">📱 View Cached Data</button>
                    </div>
                ` : ''}
                <div class="error-details">
                    <details>
                        <summary>🔧 Troubleshooting Tips</summary>
                        <ul>
                            <li>Check your internet connection</li>
                            <li>Refresh the page and try again</li>
                            <li>Clear your browser cache</li>
                            <li>Contact support if the problem persists</li>
                        </ul>
                    </details>
                </div>
            </div>
        `;

        const dashboard = document.querySelector('.betting-statistics-dashboard');
        if (dashboard) {
            dashboard.insertBefore(errorBanner, dashboard.firstChild);
        }
    }

    async retryLoad() {
        this.retryCount++;
        if (this.retryCount > this.maxRetries) {
            this.showError('Maximum retry attempts reached. Please refresh the page.', false);
            return;
        }
        
        // Remove error banner
        const errorBanner = document.querySelector('.error-banner');
        if (errorBanner) {
            errorBanner.remove();
        }
        
        // Wait before retrying
        const delay = Math.min(1000 * Math.pow(2, this.retryCount - 1), 5000);
        await new Promise(resolve => setTimeout(resolve, delay));
        
        await this.loadStatistics();
    }

    showRetryOption() {
        const dashboard = document.querySelector('.betting-statistics-dashboard');
        if (!dashboard) return;

        const retryButton = document.createElement('div');
        retryButton.className = 'retry-container';
        retryButton.innerHTML = `
            <button class="btn btn-primary btn-lg" onclick="window.bettingStatistics.retryLoad()" style="margin: 20px auto; display: block;">
                🔄 Retry Loading Statistics
            </button>
        `;

        dashboard.appendChild(retryButton);
    }

    showOfflineMode() {
        // Try to load cached data from localStorage
        const cachedData = localStorage.getItem(`betting-stats-${this.currentTimeframe}`);
        if (cachedData) {
            try {
                this.data = JSON.parse(cachedData);
                this.updateMetrics();
                this.showNotification('Showing cached data from previous session', 'info');
            } catch (error) {
                this.showNotification('No cached data available', 'warning');
            }
        } else {
            this.showNotification('No cached data available', 'warning');
        }
    }

    showNotification(message, type = 'info') {
        if (window.tennisDashboard && typeof window.tennisDashboard.showNotification === 'function') {
            window.tennisDashboard.showNotification('Betting Statistics', message, type);
        } else {
            // Fallback notification
            const notification = document.createElement('div');
            notification.className = `notification notification-${type} show`;
            notification.innerHTML = `
                <div class="notification-content">
                    <div class="notification-title">Betting Statistics</div>
                    <div class="notification-message">${message}</div>
                    <button class="notification-close" onclick="this.parentElement.parentElement.remove()">&times;</button>
                </div>
            `;
            document.body.appendChild(notification);
            
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.remove();
                }
            }, 5000);
        }
    }

    resizeCharts() {
        // Resize all charts for responsive design
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.resize === 'function') {
                chart.resize();
            }
        });
    }
    
    // New methods for enhanced functionality
    async loadFreshDataInBackground() {
        try {
            // Load fresh data without showing loading UI
            const response = await fetch(`${this.API_BASE}/betting/statistics?timeframe=${this.currentTimeframe}&test_mode=live`);
            const data = await response.json();
            
            if (data.success && data.statistics) {
                const oldData = this.data;
                this.data = data.statistics;
                
                // Update UI if data has changed significantly
                if (this.hasSignificantChanges(oldData, this.data)) {
                    this.updateMetrics();
                    this.updateDataQuality();
                    this.showNotification('Statistics updated with fresh data', 'success');
                }
            }
        } catch (error) {
            console.warn('Background data refresh failed:', error);
        }
    }
    
    hasSignificantChanges(oldData, newData) {
        if (!oldData || !newData) return true;
        
        const oldBasic = oldData.basic_metrics || {};
        const newBasic = newData.basic_metrics || {};
        const oldFinancial = oldData.financial_metrics || {};
        const newFinancial = newData.financial_metrics || {};
        
        // Check for significant changes in key metrics
        const totalBetsChanged = (newBasic.total_bets || 0) !== (oldBasic.total_bets || 0);
        const profitChanged = Math.abs((newFinancial.net_profit || 0) - (oldFinancial.net_profit || 0)) > 10;
        const winRateChanged = Math.abs((newBasic.win_rate || 0) - (oldBasic.win_rate || 0)) > 1;
        
        return totalBetsChanged || profitChanged || winRateChanged;
    }
    
    announceToScreenReader(message) {
        const announcer = document.getElementById('screen-reader-announcer');
        if (announcer) {
            announcer.textContent = message;
            setTimeout(() => {
                if (announcer.textContent === message) {
                    announcer.textContent = '';
                }
            }, 3000);
        }
    }
    
    announceError(message) {
        this.announceToScreenReader(`Error: ${message}`);
    }

    // Enhanced caching with cleanup and error handling
    cacheData() {
        if (this.data) {
            try {
                const cacheData = {
                    data: this.data,
                    timestamp: Date.now(),
                    version: '1.0'
                };
                localStorage.setItem(`betting-stats-${this.currentTimeframe}`, JSON.stringify(cacheData));
                
                // Clean up old cache entries
                this.cleanupCache();
            } catch (error) {
                console.warn('Failed to cache betting statistics data:', error);
                this.cleanupCache(true);
            }
        }
    }
    
    cleanupCache(aggressive = false) {
        try {
            const keysToKeep = aggressive ? 1 : 3;
            const cacheKeys = [];
            
            for (let i = 0; i < localStorage.length; i++) {
                const key = localStorage.key(i);
                if (key && key.startsWith('betting-stats-') && !key.endsWith('-timestamp')) {
                    try {
                        const data = JSON.parse(localStorage.getItem(key));
                        if (data && data.timestamp) {
                            cacheKeys.push({ key, timestamp: data.timestamp });
                        }
                    } catch (e) {
                        localStorage.removeItem(key);
                    }
                }
            }
            
            cacheKeys.sort((a, b) => b.timestamp - a.timestamp);
            const keysToRemove = cacheKeys.slice(keysToKeep);
            
            keysToRemove.forEach(({ key }) => {
                localStorage.removeItem(key);
            });
            
        } catch (error) {
            console.warn('Cache cleanup failed:', error);
        }
    }
    
    getCachedData(timeframe) {
        try {
            const cached = localStorage.getItem(`betting-stats-${timeframe}`);
            if (cached) {
                const cacheData = JSON.parse(cached);
                const age = Date.now() - cacheData.timestamp;
                const maxAge = 10 * 60 * 1000; // 10 minutes
                
                if (age < maxAge && cacheData.data) {
                    return {
                        data: cacheData.data,
                        cached: true,
                        age: Math.floor(age / 60000)
                    };
                }
            }
        } catch (error) {
            console.warn('Failed to retrieve cached data:', error);
        }
        return null;
    }
    
    // Performance monitoring
    getPerformanceMetrics() {
        if (!window.performance) return null;
        
        return {
            loadTime: performance.now(),
            memoryUsage: performance.memory ? {
                used: Math.round(performance.memory.usedJSHeapSize / 1048576),
                total: Math.round(performance.memory.totalJSHeapSize / 1048576)
            } : null,
            chartCount: Object.keys(this.charts).length,
            dataSize: this.data ? JSON.stringify(this.data).length : 0
        };
    }

    destroy() {
        // Clean up charts
        Object.keys(this.charts).forEach(chartName => {
            this.destroyChart(chartName);
        });
        
        // Cache current data before destroying
        this.cacheData();
        
        // Remove event listeners
        window.removeEventListener('resize', this.resizeHandler);
    }
}

// Global initialization function
function initBettingStatistics() {
    try {
        if (!window.bettingStatistics) {
            window.bettingStatistics = new BettingStatistics();
            console.log('Betting statistics initialized successfully');
        } else {
            // Refresh data if already initialized
            window.bettingStatistics.refreshData(false);
        }
    } catch (error) {
        console.error('Failed to initialize betting statistics:', error);
        // Show error notification in UI
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-banner';
        errorDiv.innerHTML = `
            <div class="error-content">
                <div class="error-header">
                    <h3>⚠️ Initialization Error</h3>
                    <p>Failed to initialize betting statistics. Please refresh the page.</p>
                </div>
                <div class="error-actions">
                    <button class="btn btn-primary" onclick="location.reload()">🔄 Refresh Page</button>
                </div>
            </div>
        `;
        const dashboard = document.querySelector('.betting-statistics-dashboard');
        if (dashboard) {
            dashboard.insertBefore(errorDiv, dashboard.firstChild);
        }
    }
}

// Enhanced tab integration
function initBettingStatisticsTab() {
    const isActive = !document.getElementById('betting-statistics').hasAttribute('aria-hidden');
    
    if (isActive) {
        // Initialize immediately if tab is active
        initBettingStatistics();
    } else {
        // Set up lazy loading for when tab becomes active
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'attributes' && mutation.attributeName === 'aria-hidden') {
                    const target = mutation.target;
                    if (target.id === 'betting-statistics' && !target.hasAttribute('aria-hidden')) {
                        initBettingStatistics();
                        observer.disconnect(); // Stop observing once initialized
                    }
                }
            });
        });
        
        observer.observe(document.getElementById('betting-statistics'), {
            attributes: true,
            attributeFilter: ['aria-hidden']
        });
    }
}

// Auto-initialize with enhanced integration
document.addEventListener('DOMContentLoaded', () => {
    // Wait for tab navigation to be set up
    setTimeout(() => {
        initBettingStatisticsTab();
    }, 100);
    
    // Ensure Chart.js is loaded
    if (typeof Chart === 'undefined') {
        console.warn('Chart.js not loaded. Charts may not display properly.');
    }
    
    // Add visibility change handler for background refresh
    document.addEventListener('visibilitychange', () => {
        if (!document.hidden && window.bettingStatistics) {
            // Refresh data when page becomes visible again (user returns to tab)
            const bettingStatsTab = document.getElementById('betting-statistics');
            if (bettingStatsTab && bettingStatsTab.classList.contains('active')) {
                setTimeout(() => {
                    window.bettingStatistics.refreshData(true);
                }, 2000); // Wait 2 seconds before refreshing
            }
        }
    });
});

// Ensure global access for debugging and external integration
window.initBettingStatistics = initBettingStatistics;

// Add performance monitoring
if (window.performance && performance.mark) {
    performance.mark('betting-statistics-script-loaded');
}