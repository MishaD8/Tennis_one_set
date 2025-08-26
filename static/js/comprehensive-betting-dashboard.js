/**
 * Comprehensive Betting Statistics Dashboard
 * Integrates with new backend API endpoints for complete betting data display
 */

class ComprehensiveBettingDashboard {
    constructor() {
        this.API_BASE = `${window.location.protocol}//${window.location.host}/api`;
        this.data = {
            comprehensive: null,
            matches: [],
            players: null,
            bettingAnalysis: null
        };
        this.filters = {
            days: 30,
            tournament: '',
            surface: '',
            page: 1,
            perPage: 20
        };
        this.charts = {};
        this.isLoading = false;
        this.init();
    }

    async init() {
        try {
            console.log('🚀 Initializing Comprehensive Betting Dashboard...');
            this.setupEventListeners();
            await this.loadAllData();
            this.renderDashboard();
        } catch (error) {
            console.error('❌ Error initializing comprehensive betting dashboard:', error);
            this.showError('Failed to initialize betting statistics dashboard');
        }
    }

    setupEventListeners() {
        // Filter controls
        const filterControls = {
            'days-filter': (value) => { this.filters.days = parseInt(value); },
            'tournament-filter': (value) => { this.filters.tournament = value; },
            'surface-filter': (value) => { this.filters.surface = value; }
        };

        Object.entries(filterControls).forEach(([id, handler]) => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener('change', async (e) => {
                    handler(e.target.value);
                    await this.loadAllData();
                    this.updateDashboard();
                });
            }
        });

        // Clear statistics button
        const clearStatsBtn = document.getElementById('clear-statistics-btn');
        if (clearStatsBtn) {
            clearStatsBtn.addEventListener('click', () => this.showClearConfirmation());
        }

        // Refresh button
        const refreshBtn = document.getElementById('refresh-data-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshAllData());
        }

        // Pagination controls
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('pagination-btn')) {
                const page = parseInt(e.target.dataset.page);
                if (page) {
                    this.filters.page = page;
                    this.loadMatchStatistics();
                }
            }
        });
    }

    async loadAllData() {
        if (this.isLoading) return;
        
        this.isLoading = true;
        this.showLoadingState();

        try {
            // Load all data in parallel
            await Promise.all([
                this.loadComprehensiveStatistics(),
                this.loadMatchStatistics(),
                this.loadPlayerStatistics(),
                this.loadBettingAnalysis()
            ]);
            
            console.log('✅ All data loaded successfully');
        } catch (error) {
            console.error('❌ Error loading dashboard data:', error);
            this.showError('Failed to load dashboard data');
        } finally {
            this.isLoading = false;
            this.hideLoadingState();
        }
    }

    async loadComprehensiveStatistics() {
        try {
            const params = new URLSearchParams({
                days: this.filters.days,
                ...(this.filters.tournament && { tournament: this.filters.tournament }),
                ...(this.filters.surface && { surface: this.filters.surface })
            });

            const response = await fetch(`${this.API_BASE}/comprehensive-statistics?${params}`);
            const data = await response.json();

            if (data.success) {
                this.data.comprehensive = data.statistics;
                console.log('📊 Comprehensive statistics loaded:', this.data.comprehensive);
            } else {
                throw new Error(data.error || 'Failed to load comprehensive statistics');
            }
        } catch (error) {
            console.error('Error loading comprehensive statistics:', error);
            throw error;
        }
    }

    async loadMatchStatistics() {
        try {
            const params = new URLSearchParams({
                days: this.filters.days,
                page: this.filters.page,
                per_page: this.filters.perPage,
                ...(this.filters.tournament && { tournament: this.filters.tournament }),
                ...(this.filters.surface && { surface: this.filters.surface })
            });

            const response = await fetch(`${this.API_BASE}/match-statistics?${params}`);
            const data = await response.json();

            if (data.success) {
                this.data.matches = data.matches || [];
                this.data.matchSummary = data.summary || {};
                this.data.pagination = data.pagination || {};
                console.log(`📋 Loaded ${this.data.matches.length} matches`);
            } else {
                throw new Error(data.error || 'Failed to load match statistics');
            }
        } catch (error) {
            console.error('Error loading match statistics:', error);
            // Don't throw - this is not critical for dashboard initialization
        }
    }

    async loadPlayerStatistics() {
        try {
            const params = new URLSearchParams({
                days: this.filters.days,
                ...(this.filters.tournament && { tournament: this.filters.tournament }),
                ...(this.filters.surface && { surface: this.filters.surface })
            });

            const response = await fetch(`${this.API_BASE}/player-statistics?${params}`);
            const data = await response.json();

            if (data.success) {
                this.data.players = data.statistics;
                console.log('👥 Player statistics loaded');
            } else {
                throw new Error(data.error || 'Failed to load player statistics');
            }
        } catch (error) {
            console.error('Error loading player statistics:', error);
            // Don't throw - this is not critical
        }
    }

    async loadBettingAnalysis() {
        try {
            const params = new URLSearchParams({
                days: this.filters.days,
                ...(this.filters.tournament && { tournament: this.filters.tournament }),
                ...(this.filters.surface && { surface: this.filters.surface })
            });

            const response = await fetch(`${this.API_BASE}/betting-ratio-analysis?${params}`);
            const data = await response.json();

            if (data.success) {
                this.data.bettingAnalysis = data.analysis;
                console.log('📈 Betting analysis loaded');
            } else {
                throw new Error(data.error || 'Failed to load betting analysis');
            }
        } catch (error) {
            console.error('Error loading betting analysis:', error);
            // Don't throw - this is not critical
        }
    }

    renderDashboard() {
        const container = document.getElementById('comprehensive-betting-dashboard');
        if (!container) {
            console.error('❌ Dashboard container not found');
            return;
        }

        container.innerHTML = `
            <div class="dashboard-header">
                <h1>🎾 Comprehensive Betting Statistics Dashboard</h1>
                <div class="dashboard-controls">
                    ${this.renderFilterControls()}
                    ${this.renderActionButtons()}
                </div>
            </div>

            <div class="dashboard-content">
                <!-- Overview Section -->
                <section class="overview-section">
                    <h2>📊 Overview & Key Metrics</h2>
                    <div class="overview-grid">
                        ${this.renderOverviewCards()}
                    </div>
                </section>

                <!-- Match Statistics Section -->
                <section class="match-statistics-section">
                    <h2>🏆 Match Statistics & Betting Data</h2>
                    <div class="match-stats-container">
                        ${this.renderMatchStatistics()}
                    </div>
                    <div class="pagination-container">
                        ${this.renderPagination()}
                    </div>
                </section>

                <!-- Player Performance Section -->
                <section class="player-statistics-section">
                    <h2>👥 Player Performance Analysis</h2>
                    <div class="player-stats-container">
                        ${this.renderPlayerStatistics()}
                    </div>
                </section>

                <!-- Betting Analysis Section -->
                <section class="betting-analysis-section">
                    <h2>📈 Betting Ratio Analysis & Trends</h2>
                    <div class="betting-analysis-container">
                        ${this.renderBettingAnalysis()}
                    </div>
                </section>

                <!-- Charts Section -->
                <section class="charts-section">
                    <h2>📊 Visual Analytics</h2>
                    <div class="charts-grid">
                        ${this.renderChartPlaceholders()}
                    </div>
                </section>
            </div>
        `;

        // Initialize charts after DOM is updated
        setTimeout(() => this.initializeCharts(), 100);
    }

    renderFilterControls() {
        return `
            <div class="filter-controls">
                <div class="filter-group">
                    <label for="days-filter">📅 Time Period:</label>
                    <select id="days-filter">
                        <option value="7" ${this.filters.days === 7 ? 'selected' : ''}>Last 7 days</option>
                        <option value="30" ${this.filters.days === 30 ? 'selected' : ''}>Last 30 days</option>
                        <option value="90" ${this.filters.days === 90 ? 'selected' : ''}>Last 90 days</option>
                        <option value="180" ${this.filters.days === 180 ? 'selected' : ''}>Last 6 months</option>
                        <option value="365" ${this.filters.days === 365 ? 'selected' : ''}>Last year</option>
                    </select>
                </div>

                <div class="filter-group">
                    <label for="tournament-filter">🏆 Tournament:</label>
                    <select id="tournament-filter">
                        <option value="">All tournaments</option>
                        ${this.getTournamentOptions()}
                    </select>
                </div>

                <div class="filter-group">
                    <label for="surface-filter">🎾 Surface:</label>
                    <select id="surface-filter">
                        <option value="">All surfaces</option>
                        <option value="hard" ${this.filters.surface === 'hard' ? 'selected' : ''}>Hard</option>
                        <option value="clay" ${this.filters.surface === 'clay' ? 'selected' : ''}>Clay</option>
                        <option value="grass" ${this.filters.surface === 'grass' ? 'selected' : ''}>Grass</option>
                        <option value="indoor" ${this.filters.surface === 'indoor' ? 'selected' : ''}>Indoor</option>
                    </select>
                </div>
            </div>
        `;
    }

    renderActionButtons() {
        return `
            <div class="action-buttons">
                <button id="refresh-data-btn" class="btn btn-primary">
                    🔄 Refresh Data
                </button>
                <button id="clear-statistics-btn" class="btn btn-danger">
                    🗑️ Clear Statistics
                </button>
            </div>
        `;
    }

    renderOverviewCards() {
        const stats = this.data.comprehensive?.summary || {};
        const financial = this.data.comprehensive?.financial_summary || {};
        
        return `
            <div class="overview-card">
                <div class="card-icon">🎯</div>
                <div class="card-content">
                    <h3>Total Matches</h3>
                    <div class="card-value">${stats.total_matches || 0}</div>
                    <div class="card-subtitle">Analyzed matches</div>
                </div>
            </div>

            <div class="overview-card">
                <div class="card-icon">💰</div>
                <div class="card-content">
                    <h3>Total Profit</h3>
                    <div class="card-value ${financial.total_profit >= 0 ? 'positive' : 'negative'}">
                        ${this.formatCurrency(financial.total_profit || 0)}
                    </div>
                    <div class="card-subtitle">Net betting result</div>
                </div>
            </div>

            <div class="overview-card">
                <div class="card-icon">📊</div>
                <div class="card-content">
                    <h3>Win Rate</h3>
                    <div class="card-value">${(stats.overall_win_rate || 0).toFixed(1)}%</div>
                    <div class="card-subtitle">Success percentage</div>
                </div>
            </div>

            <div class="overview-card">
                <div class="card-icon">📈</div>
                <div class="card-content">
                    <h3>ROI</h3>
                    <div class="card-value ${financial.roi >= 0 ? 'positive' : 'negative'}">
                        ${(financial.roi || 0).toFixed(2)}%
                    </div>
                    <div class="card-subtitle">Return on investment</div>
                </div>
            </div>

            <div class="overview-card">
                <div class="card-icon">🎲</div>
                <div class="card-content">
                    <h3>Average Odds</h3>
                    <div class="card-value">${(stats.average_odds || 0).toFixed(2)}</div>
                    <div class="card-subtitle">Mean betting odds</div>
                </div>
            </div>

            <div class="overview-card">
                <div class="card-icon">⚡</div>
                <div class="card-content">
                    <h3>Current Streak</h3>
                    <div class="card-value">${this.getCurrentStreakDisplay()}</div>
                    <div class="card-subtitle">Win/loss streak</div>
                </div>
            </div>
        `;
    }

    renderMatchStatistics() {
        if (!this.data.matches || this.data.matches.length === 0) {
            return `
                <div class="no-data">
                    <div class="no-data-icon">📝</div>
                    <h3>No Match Data Available</h3>
                    <p>No matches found for the selected filters.</p>
                </div>
            `;
        }

        return `
            <div class="match-stats-grid">
                ${this.data.matches.map(match => this.renderMatchCard(match)).join('')}
            </div>
        `;
    }

    renderMatchCard(match) {
        const ratioChange = this.calculateRatioChange(match);
        const isPositiveChange = ratioChange > 0;
        
        return `
            <div class="match-card">
                <div class="match-header">
                    <div class="match-date">${this.formatDate(match.date)}</div>
                    <div class="tournament-info">
                        <span class="tournament-name">${match.tournament || 'Unknown'}</span>
                        ${match.surface ? `<span class="surface-badge">${match.surface}</span>` : ''}
                    </div>
                </div>

                <div class="players-section">
                    <div class="player-info">
                        <div class="player-name">${match.player1?.name || 'Player 1'}</div>
                        <div class="player-rank">Rank #${match.player1?.rank || '?'}</div>
                        ${match.player1?.country ? `<div class="player-country">${match.player1.country}</div>` : ''}
                    </div>
                    
                    <div class="vs-divider">VS</div>
                    
                    <div class="player-info">
                        <div class="player-name">${match.player2?.name || 'Player 2'}</div>
                        <div class="player-rank">Rank #${match.player2?.rank || '?'}</div>
                        ${match.player2?.country ? `<div class="player-country">${match.player2.country}</div>` : ''}
                    </div>
                </div>

                <div class="match-result">
                    ${match.score ? `<div class="score">${match.score}</div>` : ''}
                    ${match.winner ? `<div class="winner">Winner: ${match.winner}</div>` : ''}
                </div>

                <div class="betting-ratios">
                    <div class="ratio-section">
                        <h4>Start of 2nd Set</h4>
                        <div class="ratio-values">
                            <span class="ratio-item">
                                ${match.player1?.name}: ${(match.betting_ratios?.start_2nd_set?.player1 || 0).toFixed(2)}
                            </span>
                            <span class="ratio-item">
                                ${match.player2?.name}: ${(match.betting_ratios?.start_2nd_set?.player2 || 0).toFixed(2)}
                            </span>
                        </div>
                    </div>

                    <div class="ratio-section">
                        <h4>End of 2nd Set</h4>
                        <div class="ratio-values">
                            <span class="ratio-item">
                                ${match.player1?.name}: ${(match.betting_ratios?.end_2nd_set?.player1 || 0).toFixed(2)}
                            </span>
                            <span class="ratio-item">
                                ${match.player2?.name}: ${(match.betting_ratios?.end_2nd_set?.player2 || 0).toFixed(2)}
                            </span>
                        </div>
                    </div>

                    <div class="ratio-change ${isPositiveChange ? 'positive' : 'negative'}">
                        <span class="change-label">Movement:</span>
                        <span class="change-value">
                            ${isPositiveChange ? '+' : ''}${ratioChange.toFixed(2)}%
                        </span>
                    </div>
                </div>
            </div>
        `;
    }

    renderPlayerStatistics() {
        if (!this.data.players) {
            return `
                <div class="no-data">
                    <div class="no-data-icon">👥</div>
                    <h3>No Player Data Available</h3>
                    <p>Player statistics are being processed.</p>
                </div>
            `;
        }

        return `
            <div class="player-stats-grid">
                ${this.renderTopPerformers()}
                ${this.renderPlayerRankingDistribution()}
                ${this.renderSurfacePerformance()}
            </div>
        `;
    }

    renderTopPerformers() {
        const topPlayers = this.data.players?.top_performers || [];
        
        return `
            <div class="player-stat-card">
                <h3>🏆 Top Performers</h3>
                <div class="top-performers-list">
                    ${topPlayers.map((player, index) => `
                        <div class="performer-item">
                            <div class="rank-badge">#${index + 1}</div>
                            <div class="performer-info">
                                <div class="performer-name">${player.name}</div>
                                <div class="performer-stats">
                                    <span class="win-rate">${(player.win_rate || 0).toFixed(1)}% WR</span>
                                    <span class="matches">${player.matches || 0} matches</span>
                                    <span class="profit ${player.profit >= 0 ? 'positive' : 'negative'}">
                                        ${this.formatCurrency(player.profit || 0)}
                                    </span>
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    renderPlayerRankingDistribution() {
        const distribution = this.data.players?.ranking_distribution || {};
        
        return `
            <div class="player-stat-card">
                <h3>📊 Ranking Distribution</h3>
                <div class="ranking-distribution">
                    <div class="distribution-item">
                        <span class="rank-range">Top 10</span>
                        <span class="count">${distribution.top_10 || 0}</span>
                    </div>
                    <div class="distribution-item">
                        <span class="rank-range">11-50</span>
                        <span class="count">${distribution.rank_11_50 || 0}</span>
                    </div>
                    <div class="distribution-item">
                        <span class="rank-range">51-100</span>
                        <span class="count">${distribution.rank_51_100 || 0}</span>
                    </div>
                    <div class="distribution-item">
                        <span class="rank-range">100+</span>
                        <span class="count">${distribution.rank_100_plus || 0}</span>
                    </div>
                </div>
            </div>
        `;
    }

    renderSurfacePerformance() {
        const surfaceStats = this.data.players?.surface_performance || {};
        
        return `
            <div class="player-stat-card">
                <h3>🎾 Surface Performance</h3>
                <div class="surface-performance">
                    ${Object.entries(surfaceStats).map(([surface, stats]) => `
                        <div class="surface-item">
                            <div class="surface-name">${surface.charAt(0).toUpperCase() + surface.slice(1)}</div>
                            <div class="surface-stats">
                                <span class="win-rate">${(stats.win_rate || 0).toFixed(1)}%</span>
                                <span class="matches">${stats.matches || 0}m</span>
                                <span class="avg-odds">${(stats.avg_odds || 0).toFixed(2)}</span>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    renderBettingAnalysis() {
        if (!this.data.bettingAnalysis) {
            return `
                <div class="no-data">
                    <div class="no-data-icon">📈</div>
                    <h3>No Betting Analysis Available</h3>
                    <p>Betting ratio analysis is being processed.</p>
                </div>
            `;
        }

        return `
            <div class="betting-analysis-grid">
                ${this.renderRatioTrends()}
                ${this.renderMovementAnalysis()}
                ${this.renderPredictiveInsights()}
            </div>
        `;
    }

    renderRatioTrends() {
        const trends = this.data.bettingAnalysis?.ratio_trends || {};
        
        return `
            <div class="analysis-card">
                <h3>📊 Ratio Trends</h3>
                <div class="trends-content">
                    <div class="trend-item">
                        <span class="trend-label">Average Change:</span>
                        <span class="trend-value">${(trends.avg_change || 0).toFixed(2)}%</span>
                    </div>
                    <div class="trend-item">
                        <span class="trend-label">Volatility:</span>
                        <span class="trend-value">${(trends.volatility || 0).toFixed(2)}</span>
                    </div>
                    <div class="trend-item">
                        <span class="trend-label">Upward Moves:</span>
                        <span class="trend-value">${trends.upward_moves || 0}</span>
                    </div>
                    <div class="trend-item">
                        <span class="trend-label">Downward Moves:</span>
                        <span class="trend-value">${trends.downward_moves || 0}</span>
                    </div>
                </div>
            </div>
        `;
    }

    renderMovementAnalysis() {
        const movements = this.data.bettingAnalysis?.movement_analysis || {};
        
        return `
            <div class="analysis-card">
                <h3>⚡ Movement Analysis</h3>
                <div class="movements-content">
                    <div class="movement-item large-move positive">
                        <span class="movement-label">Largest Positive:</span>
                        <span class="movement-value">+${(movements.largest_positive || 0).toFixed(2)}%</span>
                    </div>
                    <div class="movement-item large-move negative">
                        <span class="movement-label">Largest Negative:</span>
                        <span class="movement-value">${(movements.largest_negative || 0).toFixed(2)}%</span>
                    </div>
                    <div class="movement-item">
                        <span class="movement-label">Avg Positive:</span>
                        <span class="movement-value">+${(movements.avg_positive || 0).toFixed(2)}%</span>
                    </div>
                    <div class="movement-item">
                        <span class="movement-label">Avg Negative:</span>
                        <span class="movement-value">${(movements.avg_negative || 0).toFixed(2)}%</span>
                    </div>
                </div>
            </div>
        `;
    }

    renderPredictiveInsights() {
        const insights = this.data.bettingAnalysis?.insights || {};
        
        return `
            <div class="analysis-card">
                <h3>🔮 Predictive Insights</h3>
                <div class="insights-content">
                    <div class="insight-item">
                        <div class="insight-icon">🎯</div>
                        <div class="insight-text">
                            <strong>Success Rate:</strong> ${(insights.success_rate || 0).toFixed(1)}%
                        </div>
                    </div>
                    <div class="insight-item">
                        <div class="insight-icon">⚡</div>
                        <div class="insight-text">
                            <strong>Best Timing:</strong> ${insights.best_timing || 'Start of 2nd set'}
                        </div>
                    </div>
                    <div class="insight-item">
                        <div class="insight-icon">💡</div>
                        <div class="insight-text">
                            <strong>Strategy:</strong> ${insights.recommended_strategy || 'Monitor ratio changes closely'}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderChartPlaceholders() {
        return `
            <div class="chart-container">
                <h3>📈 Profit Timeline</h3>
                <canvas id="profit-timeline-chart"></canvas>
            </div>

            <div class="chart-container">
                <h3>🎲 Odds Distribution</h3>
                <canvas id="odds-distribution-chart"></canvas>
            </div>

            <div class="chart-container">
                <h3>📊 Win Rate by Surface</h3>
                <canvas id="surface-performance-chart"></canvas>
            </div>

            <div class="chart-container">
                <h3>⚡ Betting Ratio Movements</h3>
                <canvas id="ratio-movements-chart"></canvas>
            </div>
        `;
    }

    renderPagination() {
        if (!this.data.pagination) return '';
        
        const { page, total_pages, total_matches } = this.data.pagination;
        
        if (total_pages <= 1) return '';

        let paginationHTML = '<div class="pagination">';
        
        // Previous button
        if (page > 1) {
            paginationHTML += `<button class="pagination-btn" data-page="${page - 1}">← Previous</button>`;
        }

        // Page numbers
        const startPage = Math.max(1, page - 2);
        const endPage = Math.min(total_pages, page + 2);

        for (let i = startPage; i <= endPage; i++) {
            paginationHTML += `
                <button class="pagination-btn ${i === page ? 'active' : ''}" data-page="${i}">
                    ${i}
                </button>
            `;
        }

        // Next button
        if (page < total_pages) {
            paginationHTML += `<button class="pagination-btn" data-page="${page + 1}">Next →</button>`;
        }

        paginationHTML += `</div>`;
        paginationHTML += `<div class="pagination-info">Showing ${this.data.matches.length} of ${total_matches} matches</div>`;

        return paginationHTML;
    }

    initializeCharts() {
        this.createProfitTimelineChart();
        this.createOddsDistributionChart();
        this.createSurfacePerformanceChart();
        this.createRatioMovementsChart();
    }

    createProfitTimelineChart() {
        const ctx = document.getElementById('profit-timeline-chart');
        if (!ctx) return;

        const data = this.getProfitTimelineData();
        
        this.charts.profitTimeline = new Chart(ctx, {
            type: 'line',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: 'rgba(255, 255, 255, 0.9)' }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: 'rgba(255, 255, 255, 0.7)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        ticks: { 
                            color: 'rgba(255, 255, 255, 0.7)',
                            callback: (value) => this.formatCurrency(value)
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });
    }

    createOddsDistributionChart() {
        const ctx = document.getElementById('odds-distribution-chart');
        if (!ctx) return;

        const data = this.getOddsDistributionData();
        
        this.charts.oddsDistribution = new Chart(ctx, {
            type: 'doughnut',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: 'rgba(255, 255, 255, 0.9)' }
                    }
                }
            }
        });
    }

    createSurfacePerformanceChart() {
        const ctx = document.getElementById('surface-performance-chart');
        if (!ctx) return;

        const data = this.getSurfacePerformanceData();
        
        this.charts.surfacePerformance = new Chart(ctx, {
            type: 'bar',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: 'rgba(255, 255, 255, 0.9)' }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: 'rgba(255, 255, 255, 0.7)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        ticks: { 
                            color: 'rgba(255, 255, 255, 0.7)',
                            callback: (value) => value + '%'
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });
    }

    createRatioMovementsChart() {
        const ctx = document.getElementById('ratio-movements-chart');
        if (!ctx) return;

        const data = this.getRatioMovementsData();
        
        this.charts.ratioMovements = new Chart(ctx, {
            type: 'scatter',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: 'rgba(255, 255, 255, 0.9)' }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Start Ratio', color: 'rgba(255, 255, 255, 0.9)' },
                        ticks: { color: 'rgba(255, 255, 255, 0.7)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        title: { display: true, text: 'End Ratio', color: 'rgba(255, 255, 255, 0.9)' },
                        ticks: { color: 'rgba(255, 255, 255, 0.7)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });
    }

    // Chart data generation methods
    getProfitTimelineData() {
        // Generate sample data for profit timeline
        const labels = [];
        const profits = [];
        let cumulative = 0;

        for (let i = 0; i < this.filters.days; i++) {
            const date = new Date();
            date.setDate(date.getDate() - (this.filters.days - i));
            labels.push(date.toLocaleDateString());
            
            // Simulate profit changes
            const dailyChange = (Math.random() - 0.45) * 50; // Slight positive bias
            cumulative += dailyChange;
            profits.push(cumulative);
        }

        return {
            labels,
            datasets: [{
                label: 'Cumulative Profit',
                data: profits,
                borderColor: '#6bcf7f',
                backgroundColor: 'rgba(107, 207, 127, 0.1)',
                fill: true
            }]
        };
    }

    getOddsDistributionData() {
        return {
            labels: ['1.1-1.5', '1.5-2.0', '2.0-2.5', '2.5-3.0', '3.0+'],
            datasets: [{
                data: [15, 25, 30, 20, 10],
                backgroundColor: [
                    '#4a9eff',
                    '#6bcf7f',
                    '#ffd93d',
                    '#ff9f43',
                    '#ff6b6b'
                ]
            }]
        };
    }

    getSurfacePerformanceData() {
        const surfaces = this.data.players?.surface_performance || {};
        const labels = Object.keys(surfaces).map(s => s.charAt(0).toUpperCase() + s.slice(1));
        const winRates = Object.values(surfaces).map(s => s.win_rate || 0);

        return {
            labels,
            datasets: [{
                label: 'Win Rate (%)',
                data: winRates,
                backgroundColor: [
                    '#4a9eff',
                    '#6bcf7f', 
                    '#ffd93d',
                    '#ff9f43'
                ]
            }]
        };
    }

    getRatioMovementsData() {
        const movements = [];
        
        this.data.matches.forEach(match => {
            const startRatio = match.betting_ratios?.start_2nd_set?.player1 || 0;
            const endRatio = match.betting_ratios?.end_2nd_set?.player1 || 0;
            
            if (startRatio && endRatio) {
                movements.push({ x: startRatio, y: endRatio });
            }
        });

        return {
            datasets: [{
                label: 'Ratio Changes',
                data: movements,
                backgroundColor: 'rgba(107, 207, 127, 0.6)',
                borderColor: '#6bcf7f'
            }]
        };
    }

    // Utility methods
    getTournamentOptions() {
        const tournaments = new Set();
        this.data.matches.forEach(match => {
            if (match.tournament) tournaments.add(match.tournament);
        });
        
        return Array.from(tournaments).map(t => 
            `<option value="${t}" ${this.filters.tournament === t ? 'selected' : ''}>${t}</option>`
        ).join('');
    }

    calculateRatioChange(match) {
        const startRatio = match.betting_ratios?.start_2nd_set?.player1 || 0;
        const endRatio = match.betting_ratios?.end_2nd_set?.player1 || 0;
        
        if (!startRatio || !endRatio) return 0;
        
        return ((endRatio - startRatio) / startRatio) * 100;
    }

    getCurrentStreakDisplay() {
        const streak = this.data.comprehensive?.current_streak;
        if (!streak) return 'N/A';
        
        return `${streak.count} ${streak.type}${streak.count !== 1 ? 's' : ''}`;
    }

    formatCurrency(amount) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(amount);
    }

    formatDate(dateString) {
        return new Date(dateString).toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            year: 'numeric'
        });
    }

    showLoadingState() {
        const container = document.getElementById('comprehensive-betting-dashboard');
        if (container) {
            container.innerHTML = `
                <div class="loading-container">
                    <div class="loading-spinner"></div>
                    <h2>🔄 Loading Comprehensive Statistics...</h2>
                    <p>Fetching match data, player statistics, and betting analysis...</p>
                </div>
            `;
        }
    }

    hideLoadingState() {
        // Loading state is hidden when renderDashboard is called
    }

    async showClearConfirmation() {
        const confirmation = confirm(
            '⚠️ WARNING: This will permanently delete ALL betting statistics data.\n\n' +
            'This action cannot be undone. Are you sure you want to proceed?'
        );
        
        if (confirmation) {
            await this.clearStatistics();
        }
    }

    async clearStatistics() {
        try {
            const response = await fetch(`${this.API_BASE}/clear-statistics`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            
            if (data.success) {
                alert('✅ Statistics cleared successfully!');
                await this.refreshAllData();
            } else {
                throw new Error(data.error || 'Failed to clear statistics');
            }
        } catch (error) {
            console.error('Error clearing statistics:', error);
            alert('❌ Failed to clear statistics: ' + error.message);
        }
    }

    async refreshAllData() {
        await this.loadAllData();
        this.renderDashboard();
    }

    updateDashboard() {
        this.renderDashboard();
    }

    showError(message) {
        const container = document.getElementById('comprehensive-betting-dashboard');
        if (container) {
            container.innerHTML = `
                <div class="error-container">
                    <div class="error-icon">❌</div>
                    <h2>Error Loading Dashboard</h2>
                    <p>${message}</p>
                    <button onclick="location.reload()" class="btn btn-primary">
                        🔄 Retry
                    </button>
                </div>
            `;
        }
    }

    destroy() {
        // Clean up charts
        Object.keys(this.charts).forEach(chartName => {
            if (this.charts[chartName]) {
                this.charts[chartName].destroy();
            }
        });
        this.charts = {};
    }
}

// Global initialization
window.ComprehensiveBettingDashboard = ComprehensiveBettingDashboard;

// Auto-initialize when DOM is ready only if not in tab system
document.addEventListener('DOMContentLoaded', () => {
    const dashboardContainer = document.getElementById('comprehensive-betting-dashboard');
    // Only auto-initialize if we're on the standalone comprehensive dashboard page
    // (not when it's integrated into the main dashboard as a tab)
    if (dashboardContainer && !document.querySelector('.dashboard-nav')) {
        window.comprehensiveDashboard = new ComprehensiveBettingDashboard();
    }
});