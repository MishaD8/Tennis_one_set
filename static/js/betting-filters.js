/**
 * Advanced Betting Filters System
 * Comprehensive filtering and drill-down functionality for betting analytics
 */

class BettingFilters {
    constructor(dashboard) {
        this.dashboard = dashboard;
        this.filters = {
            tournament: [],
            surface: [],
            playerRankRange: { min: 1, max: 300 },
            oddsRange: { min: 1.1, max: 10.0 },
            edgeRange: { min: 0, max: 0.5 },
            confidenceRange: { min: 0, max: 1 },
            model: [],
            dateRange: { start: null, end: null },
            minStake: 0,
            maxStake: 1000
        };
        
        this.originalData = [];
        this.filteredData = [];
        this.filterHistory = [];
        this.presets = this.getDefaultPresets();
        
        this.init();
    }

    init() {
        this.createFilterUI();
        this.bindEvents();
        this.loadFilterPresets();
    }

    createFilterUI() {
        const filterContainer = document.createElement('div');
        filterContainer.className = 'betting-filters-container';
        filterContainer.innerHTML = `
            <div class="filters-header">
                <h3>🔍 Advanced Filters & Analytics</h3>
                <div class="filters-actions">
                    <button class="btn btn-secondary" id="reset-filters">Reset All</button>
                    <button class="btn btn-info" id="save-preset">Save Preset</button>
                    <select id="filter-presets">
                        <option value="">Select Preset...</option>
                    </select>
                    <button class="btn btn-primary" id="apply-filters">Apply Filters</button>
                </div>
            </div>

            <div class="filters-content">
                <!-- Quick Filter Chips -->
                <div class="quick-filters">
                    <h4>Quick Filters</h4>
                    <div class="filter-chips">
                        <button class="filter-chip" data-filter="high-edge">High Edge (>10%)</button>
                        <button class="filter-chip" data-filter="strong-bets">Strong Bets Only</button>
                        <button class="filter-chip" data-filter="clay-specialists">Clay Court Specialists</button>
                        <button class="filter-chip" data-filter="low-ranked">Underdogs (Rank 50+)</button>
                        <button class="filter-chip" data-filter="recent">Last 7 Days</button>
                        <button class="filter-chip" data-filter="high-odds">High Odds (>3.0)</button>
                    </div>
                </div>

                <!-- Advanced Filters Grid -->
                <div class="advanced-filters-grid">
                    <!-- Tournament & Surface Filters -->
                    <div class="filter-group">
                        <h4>Tournament & Surface</h4>
                        <div class="filter-controls">
                            <div class="multi-select-container">
                                <label for="tournament-select">Tournaments:</label>
                                <div class="multi-select" id="tournament-select">
                                    <div class="selected-items" id="selected-tournaments"></div>
                                    <div class="dropdown-content" id="tournament-options"></div>
                                </div>
                            </div>
                            
                            <div class="multi-select-container">
                                <label for="surface-select">Surfaces:</label>
                                <div class="multi-select" id="surface-select">
                                    <div class="selected-items" id="selected-surfaces"></div>
                                    <div class="dropdown-content" id="surface-options"></div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Player Ranking Filters -->
                    <div class="filter-group">
                        <h4>Player Rankings</h4>
                        <div class="filter-controls">
                            <div class="range-slider-container">
                                <label>Player Rank Range:</label>
                                <div class="range-slider">
                                    <input type="range" id="rank-min" min="1" max="300" value="1" class="range-input">
                                    <input type="range" id="rank-max" min="1" max="300" value="300" class="range-input">
                                </div>
                                <div class="range-values">
                                    <span id="rank-min-value">1</span> - <span id="rank-max-value">300</span>
                                </div>
                            </div>
                            
                            <div class="ranking-presets">
                                <button class="preset-btn" data-rank-preset="top-10">Top 10</button>
                                <button class="preset-btn" data-rank-preset="top-50">Top 50</button>
                                <button class="preset-btn" data-rank-preset="outsiders">Rank 100+</button>
                            </div>
                        </div>
                    </div>

                    <!-- Odds & Edge Filters -->
                    <div class="filter-group">
                        <h4>Odds & Edge Analysis</h4>
                        <div class="filter-controls">
                            <div class="range-slider-container">
                                <label>Bookmaker Odds Range:</label>
                                <div class="range-slider">
                                    <input type="range" id="odds-min" min="1.1" max="10" step="0.1" value="1.1" class="range-input">
                                    <input type="range" id="odds-max" min="1.1" max="10" step="0.1" value="10" class="range-input">
                                </div>
                                <div class="range-values">
                                    <span id="odds-min-value">1.1</span> - <span id="odds-max-value">10.0</span>
                                </div>
                            </div>
                            
                            <div class="range-slider-container">
                                <label>Edge Percentage Range:</label>
                                <div class="range-slider">
                                    <input type="range" id="edge-min" min="0" max="50" step="1" value="0" class="range-input">
                                    <input type="range" id="edge-max" min="0" max="50" step="1" value="50" class="range-input">
                                </div>
                                <div class="range-values">
                                    <span id="edge-min-value">0</span>% - <span id="edge-max-value">50</span>%
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- ML Model & Confidence Filters -->
                    <div class="filter-group">
                        <h4>ML Models & Confidence</h4>
                        <div class="filter-controls">
                            <div class="multi-select-container">
                                <label for="model-select">ML Models:</label>
                                <div class="multi-select" id="model-select">
                                    <div class="selected-items" id="selected-models"></div>
                                    <div class="dropdown-content" id="model-options"></div>
                                </div>
                            </div>
                            
                            <div class="range-slider-container">
                                <label>Model Confidence:</label>
                                <div class="range-slider">
                                    <input type="range" id="confidence-min" min="0" max="100" step="1" value="0" class="range-input">
                                    <input type="range" id="confidence-max" min="0" max="100" step="1" value="100" class="range-input">
                                </div>
                                <div class="range-values">
                                    <span id="confidence-min-value">0</span>% - <span id="confidence-max-value">100</span>%
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Date & Time Filters -->
                    <div class="filter-group">
                        <h4>Date & Time Period</h4>
                        <div class="filter-controls">
                            <div class="date-inputs">
                                <div class="date-input-group">
                                    <label for="start-date">Start Date:</label>
                                    <input type="date" id="start-date" class="date-input">
                                </div>
                                <div class="date-input-group">
                                    <label for="end-date">End Date:</label>
                                    <input type="date" id="end-date" class="date-input">
                                </div>
                            </div>
                            
                            <div class="date-presets">
                                <button class="preset-btn" data-date-preset="today">Today</button>
                                <button class="preset-btn" data-date-preset="week">Last 7 Days</button>
                                <button class="preset-btn" data-date-preset="month">Last 30 Days</button>
                                <button class="preset-btn" data-date-preset="all">All Time</button>
                            </div>
                        </div>
                    </div>

                    <!-- Stake Size Filters -->
                    <div class="filter-group">
                        <h4>Stake & Risk Management</h4>
                        <div class="filter-controls">
                            <div class="range-slider-container">
                                <label>Stake Size Range ($):</label>
                                <div class="range-slider">
                                    <input type="range" id="stake-min" min="0" max="1000" step="5" value="0" class="range-input">
                                    <input type="range" id="stake-max" min="0" max="1000" step="5" value="1000" class="range-input">
                                </div>
                                <div class="range-values">
                                    $<span id="stake-min-value">0</span> - $<span id="stake-max-value">1000</span>
                                </div>
                            </div>
                            
                            <div class="risk-presets">
                                <button class="preset-btn" data-risk-preset="conservative">Conservative</button>
                                <button class="preset-btn" data-risk-preset="moderate">Moderate Risk</button>
                                <button class="preset-btn" data-risk-preset="aggressive">Aggressive</button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Filter Results Summary -->
                <div class="filter-results">
                    <div class="results-summary">
                        <span class="results-count">
                            Showing <strong id="filtered-count">0</strong> of <strong id="total-count">0</strong> betting opportunities
                        </span>
                        <span class="results-stats">
                            Avg Edge: <strong id="avg-edge-filtered">0%</strong> | 
                            Total Value: <strong id="total-value-filtered">$0</strong>
                        </span>
                    </div>
                    
                    <div class="active-filters" id="active-filters">
                        <!-- Active filter chips will appear here -->
                    </div>
                </div>
            </div>
        `;
        
        // Insert filters before the main content
        const mainContent = document.getElementById('main-content');
        if (mainContent) {
            mainContent.parentNode.insertBefore(filterContainer, mainContent);
        }
    }

    bindEvents() {
        // Apply filters button
        document.getElementById('apply-filters')?.addEventListener('click', () => this.applyFilters());
        
        // Reset filters button
        document.getElementById('reset-filters')?.addEventListener('click', () => this.resetFilters());
        
        // Save preset button
        document.getElementById('save-preset')?.addEventListener('click', () => this.savePreset());
        
        // Preset selection
        document.getElementById('filter-presets')?.addEventListener('change', (e) => this.loadPreset(e.target.value));
        
        // Quick filter chips
        document.querySelectorAll('.filter-chip').forEach(chip => {
            chip.addEventListener('click', (e) => this.applyQuickFilter(e.target.dataset.filter));
        });
        
        // Range sliders
        this.bindRangeSliders();
        
        // Multi-select dropdowns
        this.bindMultiSelects();
        
        // Preset buttons
        this.bindPresetButtons();
    }

    bindRangeSliders() {
        const sliders = [
            { id: 'rank', min: 'rank-min', max: 'rank-max', minVal: 'rank-min-value', maxVal: 'rank-max-value' },
            { id: 'odds', min: 'odds-min', max: 'odds-max', minVal: 'odds-min-value', maxVal: 'odds-max-value' },
            { id: 'edge', min: 'edge-min', max: 'edge-max', minVal: 'edge-min-value', maxVal: 'edge-max-value' },
            { id: 'confidence', min: 'confidence-min', max: 'confidence-max', minVal: 'confidence-min-value', maxVal: 'confidence-max-value' },
            { id: 'stake', min: 'stake-min', max: 'stake-max', minVal: 'stake-min-value', maxVal: 'stake-max-value' }
        ];

        sliders.forEach(slider => {
            const minSlider = document.getElementById(slider.min);
            const maxSlider = document.getElementById(slider.max);
            const minValue = document.getElementById(slider.minVal);
            const maxValue = document.getElementById(slider.maxVal);

            if (minSlider && maxSlider && minValue && maxValue) {
                minSlider.addEventListener('input', () => {
                    const min = parseFloat(minSlider.value);
                    const max = parseFloat(maxSlider.value);
                    
                    if (min > max) {
                        minSlider.value = max;
                    }
                    
                    minValue.textContent = slider.id === 'stake' ? minSlider.value : 
                                          slider.id === 'odds' ? parseFloat(minSlider.value).toFixed(1) : 
                                          minSlider.value;
                });

                maxSlider.addEventListener('input', () => {
                    const min = parseFloat(minSlider.value);
                    const max = parseFloat(maxSlider.value);
                    
                    if (max < min) {
                        maxSlider.value = min;
                    }
                    
                    maxValue.textContent = slider.id === 'stake' ? maxSlider.value : 
                                          slider.id === 'odds' ? parseFloat(maxSlider.value).toFixed(1) : 
                                          maxSlider.value;
                });
            }
        });
    }

    bindMultiSelects() {
        const multiSelects = ['tournament', 'surface', 'model'];
        
        multiSelects.forEach(type => {
            const container = document.getElementById(`${type}-select`);
            const selectedItems = document.getElementById(`selected-${type}s`);
            const options = document.getElementById(`${type}-options`);
            
            if (container && selectedItems && options) {
                // Toggle dropdown
                selectedItems.addEventListener('click', () => {
                    options.style.display = options.style.display === 'block' ? 'none' : 'block';
                });
                
                // Close dropdown when clicking outside
                document.addEventListener('click', (e) => {
                    if (!container.contains(e.target)) {
                        options.style.display = 'none';
                    }
                });
            }
        });
    }

    bindPresetButtons() {
        // Ranking presets
        document.querySelectorAll('[data-rank-preset]').forEach(btn => {
            btn.addEventListener('click', (e) => this.applyRankingPreset(e.target.dataset.rankPreset));
        });
        
        // Date presets
        document.querySelectorAll('[data-date-preset]').forEach(btn => {
            btn.addEventListener('click', (e) => this.applyDatePreset(e.target.dataset.datePreset));
        });
        
        // Risk presets
        document.querySelectorAll('[data-risk-preset]').forEach(btn => {
            btn.addEventListener('click', (e) => this.applyRiskPreset(e.target.dataset.riskPreset));
        });
    }

    applyQuickFilter(filterType) {
        this.resetFilters();
        
        switch (filterType) {
            case 'high-edge':
                document.getElementById('edge-min').value = 10;
                document.getElementById('edge-min-value').textContent = '10';
                break;
                
            case 'strong-bets':
                document.getElementById('edge-min').value = 8;
                document.getElementById('confidence-min').value = 70;
                document.getElementById('edge-min-value').textContent = '8';
                document.getElementById('confidence-min-value').textContent = '70';
                break;
                
            case 'clay-specialists':
                this.selectSurface('Clay');
                break;
                
            case 'low-ranked':
                document.getElementById('rank-min').value = 50;
                document.getElementById('rank-min-value').textContent = '50';
                break;
                
            case 'recent':
                this.applyDatePreset('week');
                break;
                
            case 'high-odds':
                document.getElementById('odds-min').value = 3.0;
                document.getElementById('odds-min-value').textContent = '3.0';
                break;
        }
        
        this.applyFilters();
    }

    applyRankingPreset(preset) {
        const rankMin = document.getElementById('rank-min');
        const rankMax = document.getElementById('rank-max');
        const rankMinValue = document.getElementById('rank-min-value');
        const rankMaxValue = document.getElementById('rank-max-value');
        
        switch (preset) {
            case 'top-10':
                rankMin.value = 1;
                rankMax.value = 10;
                break;
            case 'top-50':
                rankMin.value = 1;
                rankMax.value = 50;
                break;
            case 'outsiders':
                rankMin.value = 100;
                rankMax.value = 300;
                break;
        }
        
        rankMinValue.textContent = rankMin.value;
        rankMaxValue.textContent = rankMax.value;
    }

    applyDatePreset(preset) {
        const startDate = document.getElementById('start-date');
        const endDate = document.getElementById('end-date');
        const today = new Date();
        
        switch (preset) {
            case 'today':
                startDate.value = today.toISOString().split('T')[0];
                endDate.value = today.toISOString().split('T')[0];
                break;
            case 'week':
                const weekAgo = new Date(today);
                weekAgo.setDate(weekAgo.getDate() - 7);
                startDate.value = weekAgo.toISOString().split('T')[0];
                endDate.value = today.toISOString().split('T')[0];
                break;
            case 'month':
                const monthAgo = new Date(today);
                monthAgo.setDate(monthAgo.getDate() - 30);
                startDate.value = monthAgo.toISOString().split('T')[0];
                endDate.value = today.toISOString().split('T')[0];
                break;
            case 'all':
                startDate.value = '';
                endDate.value = '';
                break;
        }
    }

    applyRiskPreset(preset) {
        const stakeMin = document.getElementById('stake-min');
        const stakeMax = document.getElementById('stake-max');
        const stakeMinValue = document.getElementById('stake-min-value');
        const stakeMaxValue = document.getElementById('stake-max-value');
        
        switch (preset) {
            case 'conservative':
                stakeMin.value = 5;
                stakeMax.value = 25;
                break;
            case 'moderate':
                stakeMin.value = 10;
                stakeMax.value = 50;
                break;
            case 'aggressive':
                stakeMin.value = 25;
                stakeMax.value = 100;
                break;
        }
        
        stakeMinValue.textContent = stakeMin.value;
        stakeMaxValue.textContent = stakeMax.value;
    }

    applyFilters() {
        // Collect current filter values
        this.updateFiltersFromUI();
        
        // Apply filters to data
        this.filteredData = this.filterData(this.originalData);
        
        // Update UI
        this.updateFilterResults();
        this.updateActiveFiltersDisplay();
        
        // Notify dashboard of filtered data
        if (this.dashboard && typeof this.dashboard.onDataFiltered === 'function') {
            this.dashboard.onDataFiltered(this.filteredData);
        }
        
        // Save filter state to history
        this.saveFilterState();
    }

    updateFiltersFromUI() {
        // Range filters
        this.filters.playerRankRange = {
            min: parseInt(document.getElementById('rank-min').value),
            max: parseInt(document.getElementById('rank-max').value)
        };
        
        this.filters.oddsRange = {
            min: parseFloat(document.getElementById('odds-min').value),
            max: parseFloat(document.getElementById('odds-max').value)
        };
        
        this.filters.edgeRange = {
            min: parseFloat(document.getElementById('edge-min').value) / 100,
            max: parseFloat(document.getElementById('edge-max').value) / 100
        };
        
        this.filters.confidenceRange = {
            min: parseFloat(document.getElementById('confidence-min').value) / 100,
            max: parseFloat(document.getElementById('confidence-max').value) / 100
        };
        
        this.filters.minStake = parseFloat(document.getElementById('stake-min').value);
        this.filters.maxStake = parseFloat(document.getElementById('stake-max').value);
        
        // Date filters
        const startDate = document.getElementById('start-date').value;
        const endDate = document.getElementById('end-date').value;
        
        this.filters.dateRange = {
            start: startDate ? new Date(startDate) : null,
            end: endDate ? new Date(endDate) : null
        };
    }

    filterData(data) {
        return data.filter(item => {
            // Player rank filter
            const playerRank = item.player_rank || 999;
            if (playerRank < this.filters.playerRankRange.min || playerRank > this.filters.playerRankRange.max) {
                return false;
            }
            
            // Odds filter
            const odds = item.odds?.player1 || 2.0;
            if (odds < this.filters.oddsRange.min || odds > this.filters.oddsRange.max) {
                return false;
            }
            
            // Edge filter
            const edge = item.betting_edge || 0;
            if (edge < this.filters.edgeRange.min || edge > this.filters.edgeRange.max) {
                return false;
            }
            
            // Confidence filter
            const confidence = item.confidence || 0.5;
            if (confidence < this.filters.confidenceRange.min || confidence > this.filters.confidenceRange.max) {
                return false;
            }
            
            // Date filter
            if (this.filters.dateRange.start || this.filters.dateRange.end) {
                const itemDate = new Date(item.date);
                if (this.filters.dateRange.start && itemDate < this.filters.dateRange.start) {
                    return false;
                }
                if (this.filters.dateRange.end && itemDate > this.filters.dateRange.end) {
                    return false;
                }
            }
            
            // Tournament filter
            if (this.filters.tournament.length > 0) {
                if (!this.filters.tournament.includes(item.tournament)) {
                    return false;
                }
            }
            
            // Surface filter
            if (this.filters.surface.length > 0) {
                if (!this.filters.surface.includes(item.surface)) {
                    return false;
                }
            }
            
            // Model filter
            if (this.filters.model.length > 0) {
                if (!this.filters.model.includes(item.model)) {
                    return false;
                }
            }
            
            return true;
        });
    }

    updateFilterResults() {
        const filteredCount = this.filteredData.length;
        const totalCount = this.originalData.length;
        
        document.getElementById('filtered-count').textContent = filteredCount;
        document.getElementById('total-count').textContent = totalCount;
        
        if (filteredCount > 0) {
            const avgEdge = this.filteredData.reduce((sum, item) => sum + (item.betting_edge || 0), 0) / filteredCount;
            const totalValue = this.filteredData.reduce((sum, item) => sum + (item.potential_profit || 0), 0);
            
            document.getElementById('avg-edge-filtered').textContent = `${(avgEdge * 100).toFixed(1)}%`;
            document.getElementById('total-value-filtered').textContent = `$${totalValue.toFixed(2)}`;
        } else {
            document.getElementById('avg-edge-filtered').textContent = '0%';
            document.getElementById('total-value-filtered').textContent = '$0';
        }
    }

    updateActiveFiltersDisplay() {
        const activeFiltersContainer = document.getElementById('active-filters');
        const activeFilters = [];
        
        // Check each filter type and add to active filters
        if (this.filters.playerRankRange.min > 1 || this.filters.playerRankRange.max < 300) {
            activeFilters.push(`Rank ${this.filters.playerRankRange.min}-${this.filters.playerRankRange.max}`);
        }
        
        if (this.filters.edgeRange.min > 0 || this.filters.edgeRange.max < 0.5) {
            activeFilters.push(`Edge ${(this.filters.edgeRange.min * 100).toFixed(0)}-${(this.filters.edgeRange.max * 100).toFixed(0)}%`);
        }
        
        if (this.filters.tournament.length > 0) {
            activeFilters.push(`Tournaments: ${this.filters.tournament.join(', ')}`);
        }
        
        if (this.filters.surface.length > 0) {
            activeFilters.push(`Surfaces: ${this.filters.surface.join(', ')}`);
        }
        
        if (this.filters.dateRange.start || this.filters.dateRange.end) {
            const start = this.filters.dateRange.start ? this.filters.dateRange.start.toLocaleDateString() : 'Start';
            const end = this.filters.dateRange.end ? this.filters.dateRange.end.toLocaleDateString() : 'End';
            activeFilters.push(`Date: ${start} - ${end}`);
        }
        
        activeFiltersContainer.innerHTML = activeFilters.map(filter => 
            `<span class="active-filter-chip">${filter} <button class="remove-filter">×</button></span>`
        ).join('');
    }

    resetFilters() {
        // Reset UI elements to default values
        document.getElementById('rank-min').value = 1;
        document.getElementById('rank-max').value = 300;
        document.getElementById('odds-min').value = 1.1;
        document.getElementById('odds-max').value = 10;
        document.getElementById('edge-min').value = 0;
        document.getElementById('edge-max').value = 50;
        document.getElementById('confidence-min').value = 0;
        document.getElementById('confidence-max').value = 100;
        document.getElementById('stake-min').value = 0;
        document.getElementById('stake-max').value = 1000;
        document.getElementById('start-date').value = '';
        document.getElementById('end-date').value = '';
        
        // Update value displays
        document.getElementById('rank-min-value').textContent = '1';
        document.getElementById('rank-max-value').textContent = '300';
        document.getElementById('odds-min-value').textContent = '1.1';
        document.getElementById('odds-max-value').textContent = '10.0';
        document.getElementById('edge-min-value').textContent = '0';
        document.getElementById('edge-max-value').textContent = '50';
        document.getElementById('confidence-min-value').textContent = '0';
        document.getElementById('confidence-max-value').textContent = '100';
        document.getElementById('stake-min-value').textContent = '0';
        document.getElementById('stake-max-value').textContent = '1000';
        
        // Reset filter object
        this.filters = {
            tournament: [],
            surface: [],
            playerRankRange: { min: 1, max: 300 },
            oddsRange: { min: 1.1, max: 10.0 },
            edgeRange: { min: 0, max: 0.5 },
            confidenceRange: { min: 0, max: 1 },
            model: [],
            dateRange: { start: null, end: null },
            minStake: 0,
            maxStake: 1000
        };
        
        // Apply reset filters
        this.applyFilters();
    }

    savePreset() {
        const name = prompt('Enter a name for this filter preset:');
        if (name && name.trim()) {
            const preset = {
                name: name.trim(),
                filters: JSON.parse(JSON.stringify(this.filters))
            };
            
            // Save to localStorage
            const presets = JSON.parse(localStorage.getItem('bettingFilterPresets') || '[]');
            presets.push(preset);
            localStorage.setItem('bettingFilterPresets', JSON.stringify(presets));
            
            // Update preset dropdown
            this.loadFilterPresets();
            
            // Show success message
            this.showNotification('Filter preset saved successfully!', 'success');
        }
    }

    loadPreset(presetName) {
        if (!presetName) return;
        
        const presets = JSON.parse(localStorage.getItem('bettingFilterPresets') || '[]');
        const preset = presets.find(p => p.name === presetName);
        
        if (preset) {
            this.filters = JSON.parse(JSON.stringify(preset.filters));
            this.updateUIFromFilters();
            this.applyFilters();
        }
    }

    loadFilterPresets() {
        const presetsSelect = document.getElementById('filter-presets');
        if (!presetsSelect) return;
        
        const presets = JSON.parse(localStorage.getItem('bettingFilterPresets') || '[]');
        
        presetsSelect.innerHTML = '<option value="">Select Preset...</option>' +
            presets.map(preset => `<option value="${preset.name}">${preset.name}</option>`).join('');
    }

    updateUIFromFilters() {
        // Update range sliders
        document.getElementById('rank-min').value = this.filters.playerRankRange.min;
        document.getElementById('rank-max').value = this.filters.playerRankRange.max;
        document.getElementById('odds-min').value = this.filters.oddsRange.min;
        document.getElementById('odds-max').value = this.filters.oddsRange.max;
        document.getElementById('edge-min').value = this.filters.edgeRange.min * 100;
        document.getElementById('edge-max').value = this.filters.edgeRange.max * 100;
        document.getElementById('confidence-min').value = this.filters.confidenceRange.min * 100;
        document.getElementById('confidence-max').value = this.filters.confidenceRange.max * 100;
        document.getElementById('stake-min').value = this.filters.minStake;
        document.getElementById('stake-max').value = this.filters.maxStake;
        
        // Update value displays
        document.getElementById('rank-min-value').textContent = this.filters.playerRankRange.min;
        document.getElementById('rank-max-value').textContent = this.filters.playerRankRange.max;
        document.getElementById('odds-min-value').textContent = this.filters.oddsRange.min.toFixed(1);
        document.getElementById('odds-max-value').textContent = this.filters.oddsRange.max.toFixed(1);
        document.getElementById('edge-min-value').textContent = (this.filters.edgeRange.min * 100).toFixed(0);
        document.getElementById('edge-max-value').textContent = (this.filters.edgeRange.max * 100).toFixed(0);
        document.getElementById('confidence-min-value').textContent = (this.filters.confidenceRange.min * 100).toFixed(0);
        document.getElementById('confidence-max-value').textContent = (this.filters.confidenceRange.max * 100).toFixed(0);
        document.getElementById('stake-min-value').textContent = this.filters.minStake;
        document.getElementById('stake-max-value').textContent = this.filters.maxStake;
        
        // Update date inputs
        if (this.filters.dateRange.start) {
            document.getElementById('start-date').value = this.filters.dateRange.start.toISOString().split('T')[0];
        }
        if (this.filters.dateRange.end) {
            document.getElementById('end-date').value = this.filters.dateRange.end.toISOString().split('T')[0];
        }
    }

    getDefaultPresets() {
        return [
            {
                name: 'High Value Bets',
                filters: {
                    edgeRange: { min: 0.1, max: 0.5 },
                    confidenceRange: { min: 0.7, max: 1.0 }
                }
            },
            {
                name: 'Conservative',
                filters: {
                    edgeRange: { min: 0.05, max: 0.15 },
                    confidenceRange: { min: 0.8, max: 1.0 },
                    minStake: 5,
                    maxStake: 25
                }
            },
            {
                name: 'Underdogs Only',
                filters: {
                    playerRankRange: { min: 50, max: 300 },
                    oddsRange: { min: 2.5, max: 10.0 }
                }
            }
        ];
    }

    saveFilterState() {
        this.filterHistory.push({
            timestamp: new Date(),
            filters: JSON.parse(JSON.stringify(this.filters)),
            resultCount: this.filteredData.length
        });
        
        // Keep only last 10 filter states
        if (this.filterHistory.length > 10) {
            this.filterHistory = this.filterHistory.slice(-10);
        }
    }

    showNotification(message, type = 'info') {
        // Use existing notification system if available
        if (window.tennisDashboard && typeof window.tennisDashboard.showNotification === 'function') {
            window.tennisDashboard.showNotification('Filter', message, type);
        } else {
            console.log(`[${type.toUpperCase()}] ${message}`);
        }
    }

    setData(data) {
        this.originalData = data;
        this.filteredData = [...data];
        this.updateFilterResults();
        this.populateFilterOptions();
    }

    populateFilterOptions() {
        // Extract unique values for dropdowns
        const tournaments = [...new Set(this.originalData.map(item => item.tournament))];
        const surfaces = [...new Set(this.originalData.map(item => item.surface))];
        const models = [...new Set(this.originalData.map(item => item.model))];
        
        // Populate tournament options
        this.populateMultiSelect('tournament', tournaments);
        this.populateMultiSelect('surface', surfaces);
        this.populateMultiSelect('model', models);
    }

    populateMultiSelect(type, options) {
        const optionsContainer = document.getElementById(`${type}-options`);
        if (optionsContainer) {
            optionsContainer.innerHTML = options.map(option => `
                <div class="multi-select-option" data-value="${option}">
                    <input type="checkbox" id="${type}-${option}" value="${option}">
                    <label for="${type}-${option}">${option}</label>
                </div>
            `).join('');
            
            // Bind checkbox events
            optionsContainer.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
                checkbox.addEventListener('change', () => this.updateMultiSelectSelection(type));
            });
        }
    }

    updateMultiSelectSelection(type) {
        const checkboxes = document.querySelectorAll(`#${type}-options input[type="checkbox"]:checked`);
        const selected = Array.from(checkboxes).map(cb => cb.value);
        
        this.filters[type] = selected;
        
        // Update display
        const selectedContainer = document.getElementById(`selected-${type}s`);
        if (selectedContainer) {
            selectedContainer.innerHTML = selected.length > 0 
                ? selected.map(item => `<span class="selected-item">${item}</span>`).join('')
                : `Select ${type}s...`;
        }
    }

    selectSurface(surface) {
        const checkbox = document.querySelector(`#surface-options input[value="${surface}"]`);
        if (checkbox) {
            checkbox.checked = true;
            this.updateMultiSelectSelection('surface');
        }
    }

    getFilteredData() {
        return this.filteredData;
    }

    destroy() {
        // Clean up event listeners and references
        this.dashboard = null;
        this.originalData = [];
        this.filteredData = [];
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { BettingFilters };
}

// Global availability
window.BettingFilters = BettingFilters;