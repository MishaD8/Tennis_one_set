/**
 * Betting Simulation System
 * Advanced betting simulation with capital growth, ROI tracking, and performance analysis
 */

/**
 * Data Models (TypeScript-style interfaces using JSDoc)
 */

/**
 * @typedef {Object} BetResult
 * @property {string} id - Unique bet identifier
 * @property {Date} date - Date of the bet
 * @property {string} match - Match description
 * @property {string} selection - Player/outcome selected
 * @property {number} odds - Betting odds
 * @property {number} stake - Amount wagered
 * @property {boolean} won - Whether the bet won
 * @property {number} payout - Amount returned (0 if lost)
 * @property {number} profit - Net profit/loss
 * @property {number} edge - Expected edge percentage
 * @property {string} strategy - Betting strategy used
 * @property {string} model - ML model used
 * @property {number} confidence - Model confidence
 */

/**
 * @typedef {Object} SimulationPeriod
 * @property {Date} startDate
 * @property {Date} endDate
 * @property {number} initialCapital
 * @property {number} finalCapital
 * @property {number} totalStaked
 * @property {number} totalReturns
 * @property {number} netProfit
 * @property {number} roi - Return on Investment
 * @property {number} maxDrawdown
 * @property {number} winRate
 * @property {number} avgOdds
 * @property {number} totalBets
 * @property {number} winningStreak
 * @property {number} losingStreak
 * @property {number} sharpeRatio
 */

/**
 * @typedef {Object} DrawdownPeriod
 * @property {Date} startDate
 * @property {Date} endDate
 * @property {number} peakCapital
 * @property {number} troughCapital
 * @property {number} drawdownAmount
 * @property {number} drawdownPercent
 * @property {number} duration - Duration in days
 */

/**
 * @typedef {Object} StreakPeriod
 * @property {string} type - 'winning' or 'losing'
 * @property {Date} startDate
 * @property {Date} endDate
 * @property {number} length - Number of consecutive bets
 * @property {number} profitLoss - Total profit/loss during streak
 */

class BettingSimulation {
    constructor(options = {}) {
        this.initialCapital = options.initialCapital || 100;
        this.bankrollManagement = options.bankrollManagement || 'flat'; // 'flat', 'kelly', 'proportional'
        this.flatStake = options.flatStake || 10;
        this.kellyFraction = options.kellyFraction || 0.25; // Conservative Kelly
        this.maxStakePercent = options.maxStakePercent || 5; // Max 5% of bankroll per bet
        
        // Simulation data
        this.betResults = [];
        this.capitalHistory = [];
        this.drawdowns = [];
        this.streaks = [];
        this.currentCapital = this.initialCapital;
        this.currentStreak = null;
        this.peakCapital = this.initialCapital;
        this.currentDrawdown = null;
    }

    /**
     * Add a bet result to the simulation
     * @param {BetResult} betResult 
     */
    addBetResult(betResult) {
        // Calculate stake based on strategy
        const stake = this.calculateStake(betResult);
        
        // Update bet result with actual stake
        betResult.stake = stake;
        betResult.profit = betResult.won ? (stake * betResult.odds - stake) : -stake;
        betResult.payout = betResult.won ? stake * betResult.odds : 0;
        
        // Update capital
        this.currentCapital += betResult.profit;
        
        // Record capital snapshot
        this.capitalHistory.push({
            date: betResult.date,
            capital: this.currentCapital,
            betId: betResult.id,
            profit: betResult.profit
        });
        
        // Update peak capital and track drawdowns
        if (this.currentCapital > this.peakCapital) {
            this.endDrawdown(betResult.date);
            this.peakCapital = this.currentCapital;
        } else if (this.currentCapital < this.peakCapital) {
            this.updateDrawdown(betResult.date);
        }
        
        // Track streaks
        this.updateStreaks(betResult);
        
        // Store bet result
        this.betResults.push(betResult);
    }

    /**
     * Calculate stake based on bankroll management strategy
     * @param {BetResult} betResult 
     * @returns {number}
     */
    calculateStake(betResult) {
        switch (this.bankrollManagement) {
            case 'flat':
                return Math.min(this.flatStake, this.currentCapital * 0.1); // Never stake more than 10% on flat
                
            case 'kelly':
                const kellyStake = this.calculateKellyStake(betResult);
                return Math.min(kellyStake, this.currentCapital * this.maxStakePercent / 100);
                
            case 'proportional':
                return this.currentCapital * (this.maxStakePercent / 100);
                
            default:
                return this.flatStake;
        }
    }

    /**
     * Calculate Kelly criterion stake
     * @param {BetResult} betResult 
     * @returns {number}
     */
    calculateKellyStake(betResult) {
        const p = betResult.edge + (1 / betResult.odds); // True probability
        const q = 1 - p; // Probability of losing
        const b = betResult.odds - 1; // Net odds received
        
        const kellyFraction = (b * p - q) / b;
        const conservativeKelly = kellyFraction * this.kellyFraction;
        
        return Math.max(0, this.currentCapital * conservativeKelly);
    }

    /**
     * Update drawdown tracking
     * @param {Date} date 
     */
    updateDrawdown(date) {
        if (!this.currentDrawdown) {
            this.currentDrawdown = {
                startDate: date,
                peakCapital: this.peakCapital,
                troughCapital: this.currentCapital,
                drawdownAmount: this.peakCapital - this.currentCapital,
                drawdownPercent: ((this.peakCapital - this.currentCapital) / this.peakCapital) * 100
            };
        } else {
            this.currentDrawdown.troughCapital = Math.min(this.currentDrawdown.troughCapital, this.currentCapital);
            this.currentDrawdown.drawdownAmount = this.peakCapital - this.currentDrawdown.troughCapital;
            this.currentDrawdown.drawdownPercent = (this.currentDrawdown.drawdownAmount / this.peakCapital) * 100;
        }
    }

    /**
     * End current drawdown period
     * @param {Date} date 
     */
    endDrawdown(date) {
        if (this.currentDrawdown) {
            this.currentDrawdown.endDate = date;
            this.currentDrawdown.duration = Math.ceil((date - this.currentDrawdown.startDate) / (1000 * 60 * 60 * 24));
            this.drawdowns.push(this.currentDrawdown);
            this.currentDrawdown = null;
        }
    }

    /**
     * Update winning/losing streaks
     * @param {BetResult} betResult 
     */
    updateStreaks(betResult) {
        if (!this.currentStreak) {
            this.currentStreak = {
                type: betResult.won ? 'winning' : 'losing',
                startDate: betResult.date,
                length: 1,
                profitLoss: betResult.profit,
                bets: [betResult.id]
            };
        } else if ((betResult.won && this.currentStreak.type === 'winning') || 
                   (!betResult.won && this.currentStreak.type === 'losing')) {
            // Continue current streak
            this.currentStreak.length++;
            this.currentStreak.profitLoss += betResult.profit;
            this.currentStreak.bets.push(betResult.id);
        } else {
            // End current streak and start new one
            this.currentStreak.endDate = betResult.date;
            this.streaks.push({...this.currentStreak});
            
            this.currentStreak = {
                type: betResult.won ? 'winning' : 'losing',
                startDate: betResult.date,
                length: 1,
                profitLoss: betResult.profit,
                bets: [betResult.id]
            };
        }
    }

    /**
     * Get simulation statistics
     * @returns {SimulationPeriod}
     */
    getStats() {
        if (this.betResults.length === 0) {
            return this.getEmptyStats();
        }

        const totalStaked = this.betResults.reduce((sum, bet) => sum + bet.stake, 0);
        const totalReturns = this.betResults.reduce((sum, bet) => sum + bet.payout, 0);
        const winningBets = this.betResults.filter(bet => bet.won);
        const maxDrawdown = Math.max(0, ...this.drawdowns.map(d => d.drawdownPercent));
        const maxWinStreak = Math.max(0, ...this.streaks.filter(s => s.type === 'winning').map(s => s.length));
        const maxLoseStreak = Math.max(0, ...this.streaks.filter(s => s.type === 'losing').map(s => s.length));

        return {
            startDate: this.betResults[0].date,
            endDate: this.betResults[this.betResults.length - 1].date,
            initialCapital: this.initialCapital,
            finalCapital: this.currentCapital,
            totalStaked,
            totalReturns,
            netProfit: this.currentCapital - this.initialCapital,
            roi: ((this.currentCapital - this.initialCapital) / this.initialCapital) * 100,
            maxDrawdown,
            winRate: (winningBets.length / this.betResults.length) * 100,
            avgOdds: this.betResults.reduce((sum, bet) => sum + bet.odds, 0) / this.betResults.length,
            totalBets: this.betResults.length,
            winningStreak: maxWinStreak,
            losingStreak: maxLoseStreak,
            sharpeRatio: this.calculateSharpeRatio()
        };
    }

    /**
     * Calculate Sharpe ratio for the betting strategy
     * @returns {number}
     */
    calculateSharpeRatio() {
        if (this.betResults.length < 2) return 0;

        const returns = this.betResults.map(bet => bet.profit / bet.stake);
        const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
        const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / (returns.length - 1);
        const stdDev = Math.sqrt(variance);

        return stdDev === 0 ? 0 : avgReturn / stdDev;
    }

    /**
     * Get empty stats object
     * @returns {SimulationPeriod}
     */
    getEmptyStats() {
        return {
            startDate: new Date(),
            endDate: new Date(),
            initialCapital: this.initialCapital,
            finalCapital: this.currentCapital,
            totalStaked: 0,
            totalReturns: 0,
            netProfit: 0,
            roi: 0,
            maxDrawdown: 0,
            winRate: 0,
            avgOdds: 0,
            totalBets: 0,
            winningStreak: 0,
            losingStreak: 0,
            sharpeRatio: 0
        };
    }

    /**
     * Generate sample betting data for demonstration
     * @param {number} numBets - Number of bets to simulate
     */
    generateSampleData(numBets = 50) {
        const players = [
            'Novak Djokovic', 'Carlos Alcaraz', 'Jannik Sinner', 'Alexander Zverev', 
            'Daniil Medvedev', 'Stefanos Tsitsipas', 'Casper Ruud', 'Andrey Rublev'
        ];
        const tournaments = ['Australian Open', 'French Open', 'Wimbledon', 'US Open', 'ATP Masters', 'ATP 500'];
        const models = ['LightGBM', 'XGBoost', 'Random Forest', 'Neural Network'];

        let currentDate = new Date();
        currentDate.setDate(currentDate.getDate() - numBets);

        for (let i = 0; i < numBets; i++) {
            currentDate.setDate(currentDate.getDate() + 1);
            
            const player1 = players[Math.floor(Math.random() * players.length)];
            const player2 = players[Math.floor(Math.random() * players.length)];
            if (player1 === player2) continue;

            const odds = 1.5 + Math.random() * 3; // Odds between 1.5 and 4.5
            const edge = (Math.random() - 0.3) * 0.2; // Edge between -0.3 and 0.2
            const confidence = 0.6 + Math.random() * 0.4;
            const won = Math.random() < (0.5 + edge); // Slight bias towards edge

            const betResult = {
                id: `bet_${i + 1}`,
                date: new Date(currentDate),
                match: `${player1} vs ${player2}`,
                selection: player1,
                odds: parseFloat(odds.toFixed(2)),
                stake: 0, // Will be calculated
                won,
                payout: 0, // Will be calculated
                profit: 0, // Will be calculated
                edge: parseFloat(edge.toFixed(3)),
                strategy: this.bankrollManagement,
                model: models[Math.floor(Math.random() * models.length)],
                confidence: parseFloat(confidence.toFixed(3))
            };

            this.addBetResult(betResult);
        }
    }

    /**
     * Reset simulation to initial state
     */
    reset() {
        this.betResults = [];
        this.capitalHistory = [];
        this.drawdowns = [];
        this.streaks = [];
        this.currentCapital = this.initialCapital;
        this.currentStreak = null;
        this.peakCapital = this.initialCapital;
        this.currentDrawdown = null;
    }

    /**
     * Export simulation data
     * @returns {Object}
     */
    exportData() {
        return {
            settings: {
                initialCapital: this.initialCapital,
                bankrollManagement: this.bankrollManagement,
                flatStake: this.flatStake,
                kellyFraction: this.kellyFraction,
                maxStakePercent: this.maxStakePercent
            },
            results: {
                betResults: this.betResults,
                capitalHistory: this.capitalHistory,
                drawdowns: this.drawdowns,
                streaks: this.streaks,
                stats: this.getStats()
            }
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { BettingSimulation };
}

// Global availability
window.BettingSimulation = BettingSimulation;