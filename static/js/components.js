/**
 * Dashboard Components
 * Modular, reusable components for better maintainability
 */

class TennisComponents {
    static StatCard = class {
        constructor(value, label, icon = '') {
            this.value = value;
            this.label = label;
            this.icon = icon;
        }

        render() {
            return `
                <div class="stat-card" role="article" aria-label="${this.label}: ${this.value}">
                    <div class="stat-value" aria-label="Current value">${this.value}</div>
                    <div class="stat-label">${this.icon} ${this.label}</div>
                </div>
            `;
        }
    };

    static LoadingSpinner = class {
        constructor(message = 'Loading...', subMessage = '') {
            this.message = message;
            this.subMessage = subMessage;
        }

        render() {
            return `
                <div class="loading-container" role="status" aria-live="polite">
                    <div class="loading-spinner"></div>
                    <h3 class="loading-title">${this.message}</h3>
                    ${this.subMessage ? `<p class="loading-subtitle">${this.subMessage}</p>` : ''}
                </div>
            `;
        }
    };

    static SkeletonCard = class {
        render() {
            return `
                <div class="skeleton-card" aria-label="Loading content">
                    <div class="skeleton-header">
                        <div class="skeleton-line skeleton-title"></div>
                        <div class="skeleton-badge"></div>
                    </div>
                    <div class="skeleton-players">
                        <div class="skeleton-player">
                            <div class="skeleton-line skeleton-name"></div>
                            <div class="skeleton-line skeleton-rank"></div>
                        </div>
                        <div class="skeleton-divider"></div>
                        <div class="skeleton-player">
                            <div class="skeleton-line skeleton-name"></div>
                            <div class="skeleton-line skeleton-rank"></div>
                        </div>
                    </div>
                    <div class="skeleton-probability"></div>
                    <div class="skeleton-stats">
                        <div class="skeleton-stat"></div>
                        <div class="skeleton-stat"></div>
                        <div class="skeleton-stat"></div>
                    </div>
                </div>
            `;
        }
    };

    static PlayerCard = class {
        constructor(player, type = 'player') {
            this.player = player;
            this.type = type; // 'favorite', 'underdog', or 'player'
        }

        getTypeStyles() {
            const styles = {
                favorite: { border: '2px solid #4a9eff', title: '👑 FAVORITE', color: '#4a9eff' },
                underdog: { border: '2px solid #6bcf7f', title: '🎯 UNDERDOG', color: '#6bcf7f' },
                player: { border: '1px solid rgba(255, 255, 255, 0.1)', title: '', color: '#ffffff' }
            };
            return styles[this.type] || styles.player;
        }

        render() {
            const styles = this.getTypeStyles();
            return `
                <div class="player-info" style="border: ${styles.border}" role="article">
                    ${styles.title ? `<div style="font-weight: bold; color: ${styles.color};">${styles.title}</div>` : ''}
                    <div style="font-size: 1.1rem; margin: 5px 0;" class="player-name">
                        ${this.player.name || 'Player'}
                    </div>
                    <div style="font-size: 0.9rem; opacity: 0.8;" class="player-rank">
                        Rank #${this.player.rank || '?'}
                    </div>
                    ${this.player.country ? `<div class="player-country" aria-label="Country">${this.getCountryFlag(this.player.country)} ${this.player.country}</div>` : ''}
                </div>
            `;
        }

        getCountryFlag(country) {
            const flags = {
                'Serbia': '🇷🇸', 'Spain': '🇪🇸', 'Switzerland': '🇨🇭', 'Italy': '🇮🇹',
                'Russia': '🇷🇺', 'Germany': '🇩🇪', 'Greece': '🇬🇷', 'USA': '🇺🇸',
                'Norway': '🇳🇴', 'Denmark': '🇩🇰', 'Bulgaria': '🇧🇬', 'Australia': '🇦🇺',
                'France': '🇫🇷', 'UK': '🇬🇧', 'Canada': '🇨🇦', 'Argentina': '🇦🇷',
                'Croatia': '🇭🇷', 'Poland': '🇵🇱', 'Czech': '🇨🇿'
            };
            return flags[country] || '🏁';
        }
    };

    static MatchCard = class {
        constructor(match) {
            this.match = match;
        }

        render() {
            const analysis = this.match.underdog_analysis || {};
            const scenario = analysis.underdog_scenario || {};
            const probability = analysis.underdog_probability || 0.5;
            const quality = analysis.quality || 'FAIR';
            const qualityClass = `quality-${quality.toLowerCase()}`;

            return `
                <article class="match-card ${qualityClass}" role="article" aria-label="Tennis match analysis">
                    ${this.renderQualityBadge(quality, analysis)}
                    ${this.renderMatchHeader()}
                    ${this.renderPlayersSection(scenario)}
                    ${this.renderProbabilitySection(probability, scenario)}
                    ${this.renderStatsSection()}
                    ${this.renderKeyFactors()}
                    ${this.renderSystemInfo(analysis)}
                </article>
            `;
        }

        renderQualityBadge(quality, analysis) {
            return `
                <div class="quality-badge" role="img" aria-label="Match quality: ${quality}">
                    ${quality} ${(analysis.underdog_type || 'UNDERDOG').replace('_', ' ')}
                </div>
            `;
        }

        renderMatchHeader() {
            return `
                <div style="margin-bottom: 20px;">
                    <div style="font-size: 1.4rem; font-weight: bold; margin-bottom: 10px;" class="tournament-name">
                        ${this.match.tournament} • ${this.match.surface}
                    </div>
                </div>
            `;
        }

        renderPlayersSection(scenario) {
            const favoritePlayer = {
                name: scenario.favorite || 'Player',
                rank: scenario.favorite_rank || '?'
            };
            const underdogPlayer = {
                name: scenario.underdog || 'Player',
                rank: scenario.underdog_rank || '?'
            };

            const favoriteCard = new TennisComponents.PlayerCard(favoritePlayer, 'favorite');
            const underdogCard = new TennisComponents.PlayerCard(underdogPlayer, 'underdog');

            return `
                <div class="favorite-vs-underdog" role="region" aria-label="Match participants">
                    ${favoriteCard.render()}
                    <div class="vs-divider" aria-hidden="true">VS</div>
                    ${underdogCard.render()}
                </div>
            `;
        }

        renderProbabilitySection(probability, scenario) {
            return `
                <div class="underdog-highlight" role="region" aria-label="Prediction analysis">
                    <div class="probability" aria-label="Probability percentage">${(probability * 100).toFixed(1)}%</div>
                    <div class="confidence">${scenario.underdog || 'Underdog'} chance to win at least one set</div>
                </div>
            `;
        }

        renderStatsSection() {
            const scenario = this.match.underdog_analysis?.underdog_scenario || {};
            const analysis = this.match.underdog_analysis || {};
            
            return `
                <div class="odds-display" role="region" aria-label="Match statistics">
                    <div class="odds-item">
                        <div style="font-weight: bold;">Rank Difference</div>
                        <div style="font-size: 1.2rem; color: #ffd93d;" aria-label="Ranking difference">
                            ${scenario.rank_difference || '?'}
                        </div>
                    </div>
                    <div class="odds-item">
                        <div style="font-weight: bold;">Quality Rating</div>
                        <div style="font-size: 1.2rem; color: #6bcf7f;" aria-label="Analysis quality">
                            ${analysis.quality || 'FAIR'}
                        </div>
                    </div>
                    <div class="odds-item">
                        <div style="font-weight: bold;">ML Confidence</div>
                        <div style="font-size: 1.2rem; color: #4a9eff;" aria-label="Machine learning confidence">
                            ${analysis.confidence || 'Medium'}
                        </div>
                    </div>
                </div>
            `;
        }

        renderKeyFactors() {
            if (!this.match.key_factors || !this.match.key_factors.length) return '';
            
            return `
                <div class="factors-list" role="region" aria-label="Key analysis factors">
                    <strong>🔍 Key Factors:</strong>
                    ${this.match.key_factors.slice(0, 3).map(factor => 
                        `<div class="factor-item" role="listitem">${factor}</div>`
                    ).join('')}
                </div>
            `;
        }

        renderSystemInfo(analysis) {
            return `
                <div style="margin-top: 15px; text-align: center; font-size: 0.85rem; opacity: 0.7;" 
                     role="contentinfo" aria-label="System information">
                    ML System: ${analysis.ml_system_used || 'Basic'} • Type: ${analysis.prediction_type || 'Analysis'}
                </div>
            `;
        }
    };

    static ErrorBoundary = class {
        constructor(error, context = '') {
            this.error = error;
            this.context = context;
        }

        render() {
            return `
                <div class="error-boundary" role="alert" aria-live="assertive">
                    <div class="error-icon">⚠️</div>
                    <h3 class="error-title">Oops! Something went wrong</h3>
                    <p class="error-message">${this.getErrorMessage()}</p>
                    <div class="error-actions">
                        <button class="btn btn-primary" onclick="location.reload()">
                            🔄 Refresh Page
                        </button>
                        <button class="btn btn-secondary" onclick="this.parentElement.parentElement.style.display='none'">
                            ✖ Dismiss
                        </button>
                    </div>
                    ${process.env.NODE_ENV === 'development' ? this.renderDebugInfo() : ''}
                </div>
            `;
        }

        getErrorMessage() {
            if (this.error instanceof TypeError) return 'Data format error. Please try refreshing.';
            if (this.error instanceof ReferenceError) return 'Component error. Please contact support.';
            if (this.error.name === 'NetworkError') return 'Connection problem. Check your internet connection.';
            return 'An unexpected error occurred. Please try again.';
        }

        renderDebugInfo() {
            return `
                <details class="error-debug">
                    <summary>Debug Information</summary>
                    <pre class="error-stack">${this.error.stack || this.error.message}</pre>
                    <p><strong>Context:</strong> ${this.context}</p>
                </details>
            `;
        }
    };
}

// Export for ES6 modules (if using)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TennisComponents;
}

// Global availability for vanilla JS
window.TennisComponents = TennisComponents;