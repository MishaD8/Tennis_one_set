/**
 * Notification Management System - Betting Alerts & System Notifications
 * Features: Priority-based notifications, persistent alerts, sound alerts, user preferences
 */

class NotificationManager {
    constructor() {
        // Use current host and port for API base to avoid localhost issues
        this.API_BASE = `${window.location.protocol}//${window.location.host}/api`;
        this.notifications = new Map();
        this.notificationQueue = [];
        this.settings = {
            soundEnabled: true,
            persistentAlerts: true,
            priorityFiltering: true,
            maxNotifications: 5,
            autoHideTimeout: 8000,
            soundVolume: 0.5
        };
        this.sounds = {
            bet: null,
            alert: null,
            success: null,
            error: null
        };
        this.isInitialized = false;
        this.init();
    }

    async init() {
        try {
            this.loadUserSettings();
            this.createNotificationContainer();
            await this.loadSounds();
            this.setupEventListeners();
            this.startNotificationPolling();
            this.isInitialized = true;
            
            console.log('✅ Notification Manager initialized');
        } catch (error) {
            console.error('Error initializing notification manager:', error);
        }
    }

    createNotificationContainer() {
        // Remove existing container if present
        const existing = document.getElementById('notification-manager-container');
        if (existing) {
            existing.remove();
        }

        // Create main notification container
        const container = document.createElement('div');
        container.id = 'notification-manager-container';
        container.className = 'notification-manager';
        container.innerHTML = this.createNotificationHTML();
        
        document.body.appendChild(container);
    }

    createNotificationHTML() {
        return `
            <!-- Notification Settings Panel -->
            <div class="notification-settings" id="notification-settings" style="display: none;">
                <div class="settings-header">
                    <h4>🔔 Notification Settings</h4>
                    <button class="settings-close-btn" id="close-notification-settings">×</button>
                </div>
                
                <div class="settings-content">
                    <div class="setting-group">
                        <label class="setting-label">
                            <input type="checkbox" id="sound-enabled" ${this.settings.soundEnabled ? 'checked' : ''}>
                            <span class="setting-text">Sound Alerts</span>
                        </label>
                        <div class="setting-description">Play sound notifications for betting alerts</div>
                    </div>
                    
                    <div class="setting-group">
                        <label class="setting-label">
                            <input type="checkbox" id="persistent-alerts" ${this.settings.persistentAlerts ? 'checked' : ''}>
                            <span class="setting-text">Persistent Alerts</span>
                        </label>
                        <div class="setting-description">Keep important notifications until manually dismissed</div>
                    </div>
                    
                    <div class="setting-group">
                        <label class="setting-label">
                            <input type="checkbox" id="priority-filtering" ${this.settings.priorityFiltering ? 'checked' : ''}>
                            <span class="setting-text">Priority Filtering</span>
                        </label>
                        <div class="setting-description">Only show high-priority betting opportunities</div>
                    </div>
                    
                    <div class="setting-group">
                        <label class="setting-range-label">
                            <span class="setting-text">Sound Volume</span>
                            <input type="range" id="sound-volume" min="0" max="1" step="0.1" value="${this.settings.soundVolume}">
                        </label>
                        <div class="setting-description">Adjust notification sound volume</div>
                    </div>
                    
                    <div class="setting-group">
                        <label class="setting-range-label">
                            <span class="setting-text">Auto-hide Timer (seconds)</span>
                            <input type="range" id="auto-hide-timeout" min="3" max="30" step="1" value="${this.settings.autoHideTimeout / 1000}">
                        </label>
                        <div class="setting-description">How long to show notifications before auto-hiding</div>
                    </div>
                </div>
                
                <div class="settings-actions">
                    <button class="btn-test-notification" id="test-notification">Test Notification</button>
                    <button class="btn-clear-all" id="clear-all-notifications">Clear All</button>
                </div>
            </div>

            <!-- Notification Stack -->
            <div class="notification-stack" id="notification-stack"></div>

            <!-- Notification Toggle Button -->
            <div class="notification-toggle-btn" id="notification-toggle" title="Notification Settings">
                <span class="toggle-icon">🔔</span>
                <span class="notification-count" id="notification-count" style="display: none;">0</span>
            </div>

            <!-- Persistent Alerts Panel -->
            <div class="persistent-alerts-panel" id="persistent-alerts-panel" style="display: none;">
                <div class="alerts-header">
                    <h4>⚠️ Active Alerts</h4>
                    <button class="alerts-close-btn" id="close-persistent-alerts">×</button>
                </div>
                <div class="alerts-content" id="persistent-alerts-content">
                    <!-- Persistent alerts will be rendered here -->
                </div>
            </div>
        `;
    }

    setupEventListeners() {
        // Settings toggle
        const toggleBtn = document.getElementById('notification-toggle');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => this.toggleSettings());
        }

        // Settings close
        const closeBtn = document.getElementById('close-notification-settings');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.hideSettings());
        }

        // Setting changes
        this.setupSettingListeners();

        // Test notification
        const testBtn = document.getElementById('test-notification');
        if (testBtn) {
            testBtn.addEventListener('click', () => this.sendTestNotification());
        }

        // Clear all notifications
        const clearBtn = document.getElementById('clear-all-notifications');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearAllNotifications());
        }

        // Persistent alerts panel
        const alertsCloseBtn = document.getElementById('close-persistent-alerts');
        if (alertsCloseBtn) {
            alertsCloseBtn.addEventListener('click', () => this.hidePersistentAlerts());
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'n') {
                e.preventDefault();
                this.toggleSettings();
            }
            if (e.key === 'Escape') {
                this.hideSettings();
                this.hidePersistentAlerts();
            }
        });
    }

    setupSettingListeners() {
        const settings = [
            { id: 'sound-enabled', key: 'soundEnabled', type: 'checkbox' },
            { id: 'persistent-alerts', key: 'persistentAlerts', type: 'checkbox' },
            { id: 'priority-filtering', key: 'priorityFiltering', type: 'checkbox' },
            { id: 'sound-volume', key: 'soundVolume', type: 'range' },
            { id: 'auto-hide-timeout', key: 'autoHideTimeout', type: 'range', multiplier: 1000 }
        ];

        settings.forEach(setting => {
            const element = document.getElementById(setting.id);
            if (element) {
                element.addEventListener('change', (e) => {
                    let value = setting.type === 'checkbox' ? e.target.checked : parseFloat(e.target.value);
                    if (setting.multiplier) value *= setting.multiplier;
                    
                    this.settings[setting.key] = value;
                    this.saveUserSettings();
                    
                    // Update sound volume immediately
                    if (setting.key === 'soundVolume') {
                        this.updateSoundVolume(value);
                    }
                });
            }
        });
    }

    async loadSounds() {
        try {
            // Load notification sounds (if available)
            const soundFiles = {
                bet: '/static/sounds/bet-alert.mp3',
                alert: '/static/sounds/alert.mp3',
                success: '/static/sounds/success.mp3',
                error: '/static/sounds/error.mp3'
            };

            for (const [type, url] of Object.entries(soundFiles)) {
                try {
                    const audio = new Audio(url);
                    audio.volume = this.settings.soundVolume;
                    audio.preload = 'auto';
                    this.sounds[type] = audio;
                } catch (error) {
                    // Sound file not available, use fallback
                    console.warn(`Sound file not available: ${url}`);
                }
            }
        } catch (error) {
            console.warn('Error loading notification sounds:', error);
        }
    }

    updateSoundVolume(volume) {
        Object.values(this.sounds).forEach(audio => {
            if (audio) {
                audio.volume = volume;
            }
        });
    }

    loadUserSettings() {
        try {
            const saved = localStorage.getItem('notification-settings');
            if (saved) {
                this.settings = { ...this.settings, ...JSON.parse(saved) };
            }
        } catch (error) {
            console.warn('Error loading notification settings:', error);
        }
    }

    saveUserSettings() {
        try {
            localStorage.setItem('notification-settings', JSON.stringify(this.settings));
        } catch (error) {
            console.warn('Error saving notification settings:', error);
        }
    }

    async startNotificationPolling() {
        // Poll for new betting opportunities every 30 seconds
        setInterval(async () => {
            try {
                await this.checkForBettingAlerts();
            } catch (error) {
                console.warn('Error checking for betting alerts:', error);
            }
        }, 30000);

        // Initial check
        setTimeout(() => this.checkForBettingAlerts(), 2000);
    }

    async checkForBettingAlerts() {
        try {
            const response = await fetch(`${this.API_BASE}/betting/alerts`, {
                method: 'GET',
                headers: { 'Accept': 'application/json' },
                cache: 'no-cache'
            });

            if (response.ok) {
                const data = await response.json();
                if (data.success && data.alerts && data.alerts.length > 0) {
                    this.processBettingAlerts(data.alerts);
                }
            }
        } catch (error) {
            console.warn('Error fetching betting alerts:', error);
        }
    }

    processBettingAlerts(alerts) {
        alerts.forEach(alert => {
            // Apply priority filtering if enabled
            if (this.settings.priorityFiltering && alert.priority !== 'high') {
                return;
            }

            // Create betting notification
            this.showNotification({
                id: `betting-alert-${alert.id}`,
                type: 'bet',
                title: '💰 New Betting Opportunity',
                message: this.formatBettingMessage(alert),
                priority: alert.priority,
                persistent: alert.priority === 'high',
                data: alert
            });
        });
    }

    formatBettingMessage(alert) {
        const { match, edge, odds, recommendation } = alert;
        return `
            <div class="betting-alert-content">
                <div class="match-info">
                    <strong>${match.player1} vs ${match.player2}</strong>
                </div>
                <div class="betting-details">
                    <span class="bet-edge">Edge: ${(edge * 100).toFixed(1)}%</span>
                    <span class="bet-odds">Odds: ${odds}</span>
                    <span class="bet-recommendation ${recommendation.toLowerCase()}">${recommendation}</span>
                </div>
                <div class="tournament-info">${match.tournament}</div>
            </div>
        `;
    }

    showNotification(options) {
        const {
            id = `notification-${Date.now()}`,
            type = 'info',
            title,
            message,
            priority = 'normal',
            persistent = false,
            autoHide = true,
            data = null
        } = options;

        // Check if notification already exists
        if (this.notifications.has(id)) {
            return;
        }

        // Create notification object
        const notification = {
            id,
            type,
            title,
            message,
            priority,
            persistent,
            autoHide,
            data,
            timestamp: new Date(),
            dismissed: false
        };

        // Add to notifications map
        this.notifications.set(id, notification);

        // Render notification
        this.renderNotification(notification);

        // Play sound if enabled
        if (this.settings.soundEnabled) {
            this.playNotificationSound(type);
        }

        // Auto-hide if not persistent
        if (autoHide && !persistent) {
            setTimeout(() => {
                this.hideNotification(id);
            }, this.settings.autoHideTimeout);
        }

        // Update notification count
        this.updateNotificationCount();

        // Add to persistent alerts if applicable
        if (persistent) {
            this.addToPersistentAlerts(notification);
        }
    }

    renderNotification(notification) {
        const stack = document.getElementById('notification-stack');
        if (!stack) return;

        // Remove oldest notification if we've reached the max
        const visibleNotifications = stack.querySelectorAll('.notification-item');
        if (visibleNotifications.length >= this.settings.maxNotifications) {
            const oldest = visibleNotifications[0];
            if (oldest) {
                oldest.remove();
            }
        }

        // Create notification element
        const notificationEl = document.createElement('div');
        notificationEl.className = `notification-item ${notification.type} priority-${notification.priority}`;
        notificationEl.setAttribute('data-id', notification.id);

        notificationEl.innerHTML = `
            <div class="notification-content">
                <div class="notification-header">
                    <div class="notification-icon">${this.getNotificationIcon(notification.type)}</div>
                    <div class="notification-title">${notification.title}</div>
                    <div class="notification-timestamp">${this.formatTimestamp(notification.timestamp)}</div>
                    <button class="notification-close" onclick="window.notificationManager?.hideNotification('${notification.id}')">×</button>
                </div>
                <div class="notification-message">${notification.message}</div>
                ${notification.persistent ? '<div class="notification-persistent-indicator">📌 Persistent</div>' : ''}
                ${notification.data ? this.renderNotificationActions(notification) : ''}
            </div>
            <div class="notification-progress" ${notification.autoHide && !notification.persistent ? 'style="animation: notificationProgress ' + this.settings.autoHideTimeout + 'ms linear"' : ''}></div>
        `;

        // Add animation
        notificationEl.style.animation = 'notificationSlideIn 0.3s ease-out';

        // Insert at the top of the stack
        stack.insertBefore(notificationEl, stack.firstChild);
    }

    renderNotificationActions(notification) {
        if (notification.type === 'bet' && notification.data) {
            const alert = notification.data;
            return `
                <div class="notification-actions">
                    <button class="btn-action btn-view-match" onclick="window.notificationManager?.viewMatch('${alert.match.id}')">
                        View Match
                    </button>
                    <button class="btn-action btn-place-bet" onclick="window.notificationManager?.placeBet('${alert.id}')">
                        Place Bet
                    </button>
                    <button class="btn-action btn-dismiss" onclick="window.notificationManager?.dismissAlert('${alert.id}')">
                        Dismiss
                    </button>
                </div>
            `;
        }
        return '';
    }

    getNotificationIcon(type) {
        const icons = {
            bet: '💰',
            alert: '⚠️',
            success: '✅',
            error: '❌',
            info: 'ℹ️',
            warning: '⚠️'
        };
        return icons[type] || 'ℹ️';
    }

    formatTimestamp(timestamp) {
        const now = new Date();
        const diff = now - timestamp;
        
        if (diff < 60000) return 'Now';
        if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
        if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
        
        return timestamp.toLocaleDateString();
    }

    hideNotification(id) {
        const notificationEl = document.querySelector(`[data-id="${id}"]`);
        if (notificationEl) {
            notificationEl.style.animation = 'notificationSlideOut 0.3s ease-in';
            setTimeout(() => {
                notificationEl.remove();
            }, 300);
        }

        // Remove from notifications map
        this.notifications.delete(id);
        this.updateNotificationCount();
    }

    playNotificationSound(type) {
        try {
            const sound = this.sounds[type] || this.sounds.alert;
            if (sound) {
                sound.currentTime = 0;
                sound.play().catch(e => {
                    console.warn('Could not play notification sound:', e);
                });
            }
        } catch (error) {
            console.warn('Error playing notification sound:', error);
        }
    }

    addToPersistentAlerts(notification) {
        const alertsContent = document.getElementById('persistent-alerts-content');
        if (!alertsContent) return;

        const alertEl = document.createElement('div');
        alertEl.className = 'persistent-alert-item';
        alertEl.setAttribute('data-id', notification.id);

        alertEl.innerHTML = `
            <div class="alert-header">
                <div class="alert-icon">${this.getNotificationIcon(notification.type)}</div>
                <div class="alert-title">${notification.title}</div>
                <div class="alert-priority ${notification.priority}">${notification.priority}</div>
                <button class="alert-remove" onclick="window.notificationManager?.removePersistentAlert('${notification.id}')">×</button>
            </div>
            <div class="alert-message">${notification.message}</div>
            <div class="alert-timestamp">${notification.timestamp.toLocaleString()}</div>
        `;

        alertsContent.appendChild(alertEl);
    }

    removePersistentAlert(id) {
        const alertEl = document.querySelector(`#persistent-alerts-content [data-id="${id}"]`);
        if (alertEl) {
            alertEl.remove();
        }
        
        // Also remove from notifications if still active
        this.hideNotification(id);
    }

    updateNotificationCount() {
        const countEl = document.getElementById('notification-count');
        const count = this.notifications.size;
        
        if (countEl) {
            if (count > 0) {
                countEl.textContent = count > 99 ? '99+' : count;
                countEl.style.display = 'flex';
            } else {
                countEl.style.display = 'none';
            }
        }

        // Update toggle button appearance
        const toggleBtn = document.getElementById('notification-toggle');
        if (toggleBtn) {
            toggleBtn.classList.toggle('has-notifications', count > 0);
        }
    }

    toggleSettings() {
        const settings = document.getElementById('notification-settings');
        if (settings) {
            const isVisible = settings.style.display !== 'none';
            settings.style.display = isVisible ? 'none' : 'block';
            
            if (!isVisible) {
                settings.style.animation = 'settingsPanelSlideIn 0.3s ease-out';
            }
        }
    }

    hideSettings() {
        const settings = document.getElementById('notification-settings');
        if (settings) {
            settings.style.animation = 'settingsPanelSlideOut 0.3s ease-in';
            setTimeout(() => {
                settings.style.display = 'none';
            }, 300);
        }
    }

    togglePersistentAlerts() {
        const panel = document.getElementById('persistent-alerts-panel');
        if (panel) {
            const isVisible = panel.style.display !== 'none';
            panel.style.display = isVisible ? 'none' : 'block';
        }
    }

    hidePersistentAlerts() {
        const panel = document.getElementById('persistent-alerts-panel');
        if (panel) {
            panel.style.display = 'none';
        }
    }

    clearAllNotifications() {
        // Clear notification stack
        const stack = document.getElementById('notification-stack');
        if (stack) {
            stack.innerHTML = '';
        }

        // Clear notifications map
        this.notifications.clear();
        this.updateNotificationCount();

        // Show confirmation
        this.showNotification({
            type: 'success',
            title: 'Notifications Cleared',
            message: 'All notifications have been cleared',
            autoHide: true,
            persistent: false
        });
    }

    sendTestNotification() {
        const testTypes = ['bet', 'alert', 'success', 'error', 'info'];
        const randomType = testTypes[Math.floor(Math.random() * testTypes.length)];
        
        const testMessages = {
            bet: 'Test betting opportunity: Djokovic vs Federer - Edge: 8.5%',
            alert: 'System alert: API rate limit approaching threshold',
            success: 'Model training completed successfully',
            error: 'Connection error: Unable to fetch live data',
            info: 'Daily report: 15 new betting opportunities identified'
        };

        this.showNotification({
            type: randomType,
            title: `Test ${randomType.charAt(0).toUpperCase() + randomType.slice(1)} Notification`,
            message: testMessages[randomType],
            priority: randomType === 'bet' ? 'high' : 'normal',
            persistent: randomType === 'bet',
            autoHide: true
        });
    }

    // Public API methods for external use
    viewMatch(matchId) {
        console.log('View match:', matchId);
        // Implement match viewing logic
        this.showNotification({
            type: 'info',
            title: 'Match View',
            message: `Opening match details for ID: ${matchId}`,
            autoHide: true
        });
    }

    placeBet(alertId) {
        console.log('Place bet for alert:', alertId);
        // Implement bet placement logic
        this.showNotification({
            type: 'success',
            title: 'Bet Placement',
            message: 'Bet placement interface would open here',
            autoHide: true
        });
    }

    dismissAlert(alertId) {
        console.log('Dismiss alert:', alertId);
        this.hideNotification(`betting-alert-${alertId}`);
    }

    // Utility method to show notifications from external code
    notify(type, title, message, options = {}) {
        this.showNotification({
            type,
            title,
            message,
            ...options
        });
    }

    destroy() {
        // Stop polling
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
        }

        // Remove UI elements
        const container = document.getElementById('notification-manager-container');
        if (container) {
            container.remove();
        }

        // Clear notifications
        this.notifications.clear();
    }
}

// Global initialization
function initNotificationManager() {
    try {
        if (!window.notificationManager) {
            window.notificationManager = new NotificationManager();
        }
    } catch (error) {
        console.error('Failed to initialize notification manager:', error);
    }
}

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Initialize with a small delay to ensure other components are ready
    setTimeout(initNotificationManager, 500);
});

// Export for manual initialization
window.NotificationManager = NotificationManager;
window.initNotificationManager = initNotificationManager;