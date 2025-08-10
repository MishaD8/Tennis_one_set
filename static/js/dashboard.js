/**
 * Enhanced Dashboard - Modern JavaScript Architecture
 * Features: Component-based, Error Handling, Performance Optimizations, Accessibility
 */

class TennisDashboard {
    constructor() {
        this.API_BASE = window.location.origin + '/api';
        this.SUBSCRIBER_MODE = window.SUBSCRIBER_MODE || false;
        this.state = {
            matches: [],
            stats: {},
            loading: false,
            error: null,
            lastUpdate: null,
            cachedData: null
        };
        this.components = new Map();
        this.observers = new Set();
        this.init();
    }

    async init() {
        try {
            this.setupErrorHandling();
            this.clearUTRCache();
            this.setupEventListeners();
            this.setupIntersectionObserver();
            await this.loadInitialData();
            this.startAutoRefresh();
        } catch (error) {
            this.handleError(error, 'Dashboard initialization');
        }
    }

    setupErrorHandling() {
        window.addEventListener('error', (event) => {
            this.handleError(event.error, 'Global error');
        });

        window.addEventListener('unhandledrejection', (event) => {
            this.handleError(event.reason, 'Unhandled promise rejection');
        });
    }

    clearUTRCache() {
        const cachedData = localStorage.getItem('lastSuccessfulMatches');
        if (cachedData) {
            try {
                const cached = JSON.parse(cachedData);
                const filteredMatches = cached.data.matches.filter(match => {
                    const tournament = match.tournament.toLowerCase();
                    return !tournament.includes('utr') && 
                           !tournament.includes('ptt') && 
                           !tournament.includes('lovedale') &&
                           !tournament.includes('group');
                });
                
                if (filteredMatches.length !== cached.data.matches.length) {
                    console.log('🧹 Clearing UTR/PTT tournaments from cache');
                    if (filteredMatches.length > 0) {
                        cached.data.matches = filteredMatches;
                        localStorage.setItem('lastSuccessfulMatches', JSON.stringify(cached));
                    } else {
                        localStorage.removeItem('lastSuccessfulMatches');
                    }
                }
            } catch (e) {
                console.log('🧹 Clearing corrupted cache');
                localStorage.removeItem('lastSuccessfulMatches');
            }
        }
    }

    setupEventListeners() {
        document.addEventListener('DOMContentLoaded', () => {
            this.bindControlEvents();
            this.setupKeyboardNavigation();
        });

        // Visibility API for performance
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden && this.shouldRefresh()) {
                this.loadUnderdogOpportunities();
            }
        });
    }

    bindControlEvents() {
        // In subscriber mode, don't bind control events as buttons don't exist
        if (this.SUBSCRIBER_MODE) {
            return;
        }
        
        const controls = {
            'load-underdog': () => this.loadUnderdogOpportunities(),
            'test-underdog': () => this.testUnderdogAnalysis(),
            'manual-api-update': () => this.manualAPIUpdate(),
            'check-api-status': () => this.checkAPIStatus()
        };

        Object.entries(controls).forEach(([id, handler]) => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener('click', handler);
            }
        });
    }

    setupKeyboardNavigation() {
        document.addEventListener('keydown', (e) => {
            // In subscriber mode, only allow escape key for dismissing errors
            if (this.SUBSCRIBER_MODE) {
                if (e.key === 'Escape') {
                    this.dismissErrors();
                }
                return;
            }
            
            if (e.ctrlKey || e.metaKey) {
                switch(e.key) {
                    case 'r':
                        e.preventDefault();
                        this.loadUnderdogOpportunities();
                        break;
                    case 'u':
                        e.preventDefault();
                        this.manualAPIUpdate();
                        break;
                }
            }
            
            // Escape to dismiss errors
            if (e.key === 'Escape') {
                this.dismissErrors();
            }
        });
    }

    setupIntersectionObserver() {
        if ('IntersectionObserver' in window) {
            this.observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('animate-in');
                    }
                });
            }, {
                threshold: 0.1,
                rootMargin: '50px'
            });
        }
    }

    async loadInitialData() {
        this.setState({ loading: true });
        
        try {
            const [matches, stats] = await Promise.allSettled([
                this.fetchMatches(),
                this.fetchStats()
            ]);

            this.setState({
                matches: matches.status === 'fulfilled' ? matches.value : [],
                stats: stats.status === 'fulfilled' ? stats.value : {},
                loading: false,
                lastUpdate: new Date()
            });

            this.render();
        } catch (error) {
            this.handleError(error, 'Initial data load');
        }
    }

    // Auto-load data for subscribers
    async autoLoadData() {
        if (!this.SUBSCRIBER_MODE) {
            return;
        }
        
        try {
            await this.loadUnderdogOpportunities();
        } catch (error) {
            console.warn('Auto-load failed, using cached data if available');
            // Try to show cached data if available
            const cachedData = localStorage.getItem('lastSuccessfulMatches');
            if (cachedData) {
                this.displayCachedData(JSON.parse(cachedData));
            }
        }
    }

    async loadUnderdogOpportunities() {
        const container = document.getElementById('matches-container');
        if (!container) return;

        this.setState({ loading: true, error: null });
        
        // Show skeleton loading
        container.innerHTML = this.createSkeletonLoading();
        this.announceToScreenReader('Loading underdog opportunities');

        try {
            const response = await this.fetchWithTimeout('/matches', 10000);
            const data = await response.json();

            if (data.success && data.matches?.length > 0) {
                this.handleSuccessfulData(data);
            } else {
                await this.handleNoData(data);
            }
        } catch (error) {
            await this.handleLoadError(error);
        }
    }

    async fetchWithTimeout(endpoint, timeout = 5000) {
        const controller = new AbortController();
        const id = setTimeout(() => controller.abort(), timeout);

        try {
            const response = await fetch(`${this.API_BASE}${endpoint}`, {
                signal: controller.signal,
                headers: {
                    'Accept': 'application/json',
                    'Cache-Control': 'no-cache'
                }
            });
            clearTimeout(id);
            return response;
        } catch (error) {
            clearTimeout(id);
            throw error;
        }
    }

    handleSuccessfulData(data) {
        // Save to cache
        this.cacheData(data);
        
        // Update state
        this.setState({
            matches: data.matches,
            loading: false,
            lastUpdate: new Date()
        });

        // Render matches
        this.renderMatches(data.matches);
        this.updateStats(data.matches);
        
        this.announceToScreenReader(`Found ${data.matches.length} underdog opportunities`);
    }

    async handleNoData(data) {
        const isNoRealData = data.source === 'NO_REAL_DATA';
        const cachedData = await this.getCachedData();

        if (cachedData && !isNoRealData) {
            this.renderCachedData(cachedData);
        } else {
            this.renderEmptyState(isNoRealData);
        }
    }

    async handleLoadError(error) {
        const cachedData = await this.getCachedData();
        
        if (cachedData) {
            this.renderOfflineData(cachedData);
        } else {
            this.renderErrorState(error);
        }
        
        this.handleError(error, 'Loading opportunities');
    }

    createSkeletonLoading() {
        const skeletonCards = Array.from({ length: 3 }, () => 
            new TennisComponents.SkeletonCard().render()
        ).join('');

        return `
            <div role="status" aria-live="polite" aria-label="Loading content">
                ${skeletonCards}
            </div>
        `;
    }

    renderMatches(matches) {
        const container = document.getElementById('matches-container');
        if (!container) return;

        const header = this.createSuccessHeader(matches.length);
        const matchCards = matches.map(match => {
            const card = new TennisComponents.MatchCard(match);
            return card.render();
        }).join('');

        container.innerHTML = header + matchCards;
        
        // Animate cards in
        this.animateCardsIn(container);
    }

    createSuccessHeader(count) {
        return `
            <div class="success-banner" role="banner">
                <h2>🎯 UNDERDOG OPPORTUNITIES FOUND</h2>
                <p>Matches: ${count} • Last updated: ${this.formatTime(new Date())}</p>
            </div>
        `;
    }

    animateCardsIn(container) {
        if (this.observer) {
            container.querySelectorAll('.match-card').forEach(card => {
                this.observer.observe(card);
            });
        }
    }

    updateStats(matches) {
        const stats = this.calculateStats(matches);
        
        Object.entries(stats).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                this.animateValue(element, value);
            }
        });
    }

    calculateStats(matches) {
        const excellentCount = matches.filter(m => 
            m.underdog_analysis?.quality === 'EXCELLENT'
        ).length;

        const totalProbability = matches.reduce((sum, match) => 
            sum + (match.underdog_analysis?.underdog_probability || 0.5), 0
        );

        return {
            'underdog-count': matches.length,
            'avg-probability': `${(totalProbability / matches.length * 100).toFixed(1)}%`,
            'excellent-quality': excellentCount
        };
    }

    animateValue(element, newValue) {
        const currentValue = element.textContent;
        if (currentValue !== String(newValue)) {
            element.style.transform = 'scale(1.1)';
            element.style.color = '#6bcf7f';
            
            setTimeout(() => {
                element.textContent = newValue;
                element.style.transform = 'scale(1)';
                element.style.color = '';
            }, 150);
        }
    }

    async getCachedData() {
        try {
            const cached = localStorage.getItem('lastSuccessfulMatches');
            if (!cached) return null;

            const data = JSON.parse(cached);
            const cacheAge = (new Date() - new Date(data.timestamp)) / 1000 / 60;
            
            if (cacheAge > 120) return null; // Cache expires after 2 hours

            // Filter UTR/PTT tournaments
            const filteredMatches = data.data.matches.filter(match => {
                const tournament = match.tournament.toLowerCase();
                return !tournament.includes('utr') && 
                       !tournament.includes('ptt') && 
                       !tournament.includes('lovedale');
            });

            return filteredMatches.length > 0 ? { matches: filteredMatches, age: cacheAge } : null;
        } catch (error) {
            console.warn('Failed to parse cached data:', error);
            return null;
        }
    }

    cacheData(data) {
        try {
            localStorage.setItem('lastSuccessfulMatches', JSON.stringify({
                data: data,
                timestamp: new Date().toISOString()
            }));
        } catch (error) {
            console.warn('Failed to cache data:', error);
        }
    }

    renderCachedData(cachedData) {
        const container = document.getElementById('matches-container');
        const header = `
            <div class="cached-banner" role="banner">
                <h2>📋 CACHED UNDERDOG OPPORTUNITIES</h2>
                <p>Showing cached data (${Math.round(cachedData.age)} minutes old)</p>
            </div>
        `;
        
        const matchCards = cachedData.matches.map(match => {
            const card = new TennisComponents.MatchCard(match);
            return `<div class="cached-card">${card.render()}</div>`;
        }).join('');

        container.innerHTML = header + matchCards;
    }

    renderOfflineData(cachedData) {
        const container = document.getElementById('matches-container');
        const header = `
            <div class="offline-banner" role="banner">
                <h2>🔌 OFFLINE - CACHED DATA</h2>
                <p>Connection error. Showing cached data (${Math.round(cachedData.age)} minutes old)</p>
            </div>
        `;
        
        const matchCards = cachedData.matches.map(match => {
            const card = new TennisComponents.MatchCard(match);
            return `<div class="offline-card">${card.render()}</div>`;
        }).join('');

        container.innerHTML = header + matchCards;
    }

    renderEmptyState(isNoRealData) {
        const container = document.getElementById('matches-container');
        const message = isNoRealData 
            ? 'API quotas exhausted. Data will refresh when APIs are available.'
            : 'Try refreshing or check back later';

        container.innerHTML = `
            <div class="empty-state" role="status">
                <div class="empty-icon">${isNoRealData ? '⏳' : '❌'}</div>
                <h3>${isNoRealData ? 'No Live Data Available' : 'No underdog opportunities found'}</h3>
                <p>${message}</p>
                ${isNoRealData ? '<p class="success-note">✨ System working correctly - waiting for fresh data</p>' : ''}
            </div>
        `;
    }

    renderErrorState(error) {
        const container = document.getElementById('matches-container');
        const errorBoundary = new TennisComponents.ErrorBoundary(error, 'Loading opportunities');
        container.innerHTML = errorBoundary.render();
    }

    async testUnderdogAnalysis() {
        try {
            const response = await this.fetchWithTimeout('/test-underdog', 8000);
            const data = await response.json();

            if (data.success) {
                this.showTestResults(data);
            } else {
                throw new Error(data.error || 'Test failed');
            }
        } catch (error) {
            this.handleError(error, 'Underdog analysis test');
        }
    }

    showTestResults(data) {
        const analysis = data.underdog_analysis;
        const scenario = analysis.underdog_scenario;
        
        const results = {
            match: `${data.match_info.player1} vs ${data.match_info.player2}`,
            underdog: `${scenario.underdog} (Rank #${scenario.underdog_rank})`,
            favorite: `${scenario.favorite} (Rank #${scenario.favorite_rank})`,
            type: scenario.underdog_type,
            probability: `${(analysis.underdog_probability * 100).toFixed(1)}%`,
            quality: analysis.quality,
            system: analysis.ml_system_used
        };

        this.showNotification('Test Results', results, 'success');
        console.log('✅ Underdog Analysis Test Results:', results);
    }

    async manualAPIUpdate() {
        try {
            const response = await this.fetchWithTimeout('/manual-api-update', 10000);
            const data = await response.json();

            if (data.success) {
                this.updateAPIStatus('🔄 Updating');
                this.showNotification('API Update', 'Manual update triggered successfully', 'success');
            } else {
                throw new Error(data.error || 'Update failed');
            }
        } catch (error) {
            this.handleError(error, 'Manual API update');
        }
    }

    async checkAPIStatus() {
        try {
            const response = await this.fetchWithTimeout('/api-economy-status', 5000);
            const data = await response.json();

            if (data.success) {
                const usage = data.api_usage;
                this.updateAPIStatus(`${usage.remaining_hour}/${usage.max_per_hour}`);
                
                console.log('📊 API Economy Status:', {
                    requests_this_hour: usage.requests_this_hour,
                    max_per_hour: usage.max_per_hour,
                    remaining: usage.remaining_hour,
                    cache_items: usage.cache_items,
                    manual_update_status: usage.manual_update_status
                });
            } else {
                this.updateAPIStatus('❌ Error');
            }
        } catch (error) {
            this.updateAPIStatus('❌ Error');
            this.handleError(error, 'API status check');
        }
    }

    updateAPIStatus(status) {
        const element = document.getElementById('api-status');
        if (element) {
            element.textContent = status;
        }
    }

    handleError(error, context) {
        console.error(`🚨 ${context}:`, error);
        
        this.setState({ error: { message: error.message, context } });
        
        // Show user-friendly error message
        this.showNotification('Error', this.getErrorMessage(error), 'error');
    }

    getErrorMessage(error) {
        if (error.name === 'AbortError') return 'Request timed out. Please try again.';
        if (error.name === 'TypeError') return 'Network error. Check your connection.';
        if (error.message?.includes('fetch')) return 'Unable to connect to server.';
        return 'An unexpected error occurred. Please try again.';
    }

    showNotification(title, message, type = 'info') {
        // Create or update notification
        let notification = document.getElementById('notification');
        if (!notification) {
            notification = document.createElement('div');
            notification.id = 'notification';
            notification.className = 'notification';
            document.body.appendChild(notification);
        }

        notification.className = `notification notification-${type} show`;
        notification.innerHTML = `
            <div class="notification-content">
                <div class="notification-title">${title}</div>
                <div class="notification-message">${typeof message === 'object' ? JSON.stringify(message, null, 2) : message}</div>
                <button class="notification-close" onclick="this.parentElement.parentElement.classList.remove('show')">✖</button>
            </div>
        `;

        // Auto-hide after 5 seconds
        setTimeout(() => {
            notification.classList.remove('show');
        }, 5000);
    }

    dismissErrors() {
        const notifications = document.querySelectorAll('.notification.show');
        notifications.forEach(notification => {
            notification.classList.remove('show');
        });
    }

    announceToScreenReader(message) {
        const announcer = document.getElementById('screen-reader-announcer') || this.createAnnouncer();
        announcer.textContent = message;
        
        setTimeout(() => {
            announcer.textContent = '';
        }, 1000);
    }

    createAnnouncer() {
        const announcer = document.createElement('div');
        announcer.id = 'screen-reader-announcer';
        announcer.setAttribute('aria-live', 'polite');
        announcer.setAttribute('aria-atomic', 'true');
        announcer.className = 'sr-only';
        document.body.appendChild(announcer);
        return announcer;
    }

    shouldRefresh() {
        return !this.state.lastUpdate || 
               (new Date() - this.state.lastUpdate) > 300000; // 5 minutes
    }

    startAutoRefresh() {
        // Refresh every 10 minutes instead of 2
        setInterval(() => {
            if (!document.hidden && this.shouldRefresh()) {
                this.loadUnderdogOpportunities();
            }
        }, 600000);
    }

    formatTime(date) {
        return date.toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit',
            hour12: false
        });
    }

    setState(newState) {
        this.state = { ...this.state, ...newState };
        this.notifyObservers();
    }

    subscribe(callback) {
        this.observers.add(callback);
        return () => this.observers.delete(callback);
    }

    notifyObservers() {
        this.observers.forEach(callback => {
            try {
                callback(this.state);
            } catch (error) {
                console.error('Observer error:', error);
            }
        });
    }

    async fetchMatches() {
        // Placeholder - integrate with actual API
        return [];
    }

    async fetchStats() {
        // Placeholder - integrate with actual API
        return {};
    }

    render() {
        // Main render method - placeholder for future enhancements
        console.log('Dashboard rendered with state:', this.state);
    }

    destroy() {
        // Cleanup method
        if (this.observer) {
            this.observer.disconnect();
        }
        this.observers.clear();
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.tennisDashboard = new TennisDashboard();
});

// Global functions for backward compatibility
window.loadUnderdogOpportunities = () => window.tennisDashboard?.loadUnderdogOpportunities();
window.testUnderdogAnalysis = () => window.tennisDashboard?.testUnderdogAnalysis();
window.manualAPIUpdate = () => window.tennisDashboard?.manualAPIUpdate();
window.checkAPIStatus = () => window.tennisDashboard?.checkAPIStatus();