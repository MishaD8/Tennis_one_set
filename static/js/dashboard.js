/* Tennis Underdog Analytics Dashboard JavaScript */

const API_BASE = window.location.origin + '/api';

async function loadUnderdogOpportunities() {
    const container = document.getElementById('matches-container');
    container.innerHTML = '<div class="loading"><h3>🔍 Analyzing underdog opportunities...</h3><p>Using advanced ML models...</p></div>';
    
    try {
        const response = await fetch(API_BASE + '/matches');
        const data = await response.json();
        
        if (data.success && data.matches && data.matches.length > 0) {
            let html = `<div style="background: linear-gradient(135deg, rgba(107, 207, 127, 0.1), rgba(255, 217, 61, 0.1)); border: 1px solid rgba(107, 207, 127, 0.3); padding: 20px; border-radius: 15px; margin-bottom: 25px; text-align: center;">
                <h2>🎯 UNDERDOG OPPORTUNITIES FOUND</h2>
                <p>Source: ${data.source} • Matches: ${data.matches.length}</p>
            </div>`;
            
            // Статистика
            let excellentCount = 0;
            let totalProbability = 0;
            
            data.matches.forEach(match => {
                const analysis = match.underdog_analysis || {};
                const scenario = analysis.underdog_scenario || {};
                const probability = analysis.underdog_probability || 0.5;
                const quality = analysis.quality || 'FAIR';
                
                if (quality === 'EXCELLENT') excellentCount++;
                totalProbability += probability;
                
                const qualityClass = `quality-${quality.toLowerCase()}`;
                
                html += `
                    <div class="match-card ${qualityClass}">
                        <div class="quality-badge">
                            ${quality} ${(analysis.underdog_type || 'UNDERDOG').replace('_', ' ')}
                        </div>
                        
                        <div style="margin-bottom: 20px;">
                            <div style="font-size: 1.4rem; font-weight: bold; margin-bottom: 10px;">
                                ${match.tournament} • ${match.surface}
                            </div>
                            
                            <div class="favorite-vs-underdog">
                                <div class="player-info favorite-player">
                                    <div style="font-weight: bold; color: #4a9eff;">👑 FAVORITE</div>
                                    <div style="font-size: 1.1rem; margin: 5px 0;">${scenario.favorite || 'Player'}</div>
                                    <div style="font-size: 0.9rem; opacity: 0.8;">Rank #${scenario.favorite_rank || '?'}</div>
                                </div>
                                
                                <div class="vs-divider">VS</div>
                                
                                <div class="player-info underdog-player">
                                    <div style="font-weight: bold; color: #6bcf7f;">🎯 UNDERDOG</div>
                                    <div style="font-size: 1.1rem; margin: 5px 0;">${scenario.underdog || 'Player'}</div>
                                    <div style="font-size: 0.9rem; opacity: 0.8;">Rank #${scenario.underdog_rank || '?'}</div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="underdog-highlight">
                            <div class="probability">${(probability * 100).toFixed(1)}%</div>
                            <div class="confidence">${scenario.underdog || 'Underdog'} chance to win at least one set</div>
                        </div>
                        
                        <div class="odds-display">
                            <div class="odds-item">
                                <div style="font-weight: bold;">Rank Difference</div>
                                <div style="font-size: 1.2rem; color: #ffd93d;">${scenario.rank_difference || '?'}</div>
                            </div>
                            <div class="odds-item">
                                <div style="font-weight: bold;">Quality Rating</div>
                                <div style="font-size: 1.2rem; color: #6bcf7f;">${quality}</div>
                            </div>
                            <div class="odds-item">
                                <div style="font-weight: bold;">ML Confidence</div>
                                <div style="font-size: 1.2rem; color: #4a9eff;">${analysis.confidence || 'Medium'}</div>
                            </div>
                        </div>
                        
                        ${match.key_factors && match.key_factors.length > 0 ? `
                        <div class="factors-list">
                            <strong>🔍 Key Factors:</strong>
                            ${match.key_factors.slice(0, 3).map(factor => `<div class="factor-item">${factor}</div>`).join('')}
                        </div>
                        ` : ''}
                        
                        <div style="margin-top: 15px; text-align: center; font-size: 0.85rem; opacity: 0.7;">
                            ML System: ${analysis.ml_system_used || 'Basic'} • Type: ${analysis.prediction_type || 'Analysis'}
                        </div>
                    </div>
                `;
            });
            
            container.innerHTML = html;
            
            // Обновляем статистику
            document.getElementById('underdog-count').textContent = data.matches.length;
            document.getElementById('avg-probability').textContent = `${(totalProbability / data.matches.length * 100).toFixed(1)}%`;
            document.getElementById('excellent-quality').textContent = excellentCount;
            
        } else {
            const message = data.message || 'No underdog opportunities found';
            const isNoRealData = data.source === 'NO_REAL_DATA';
            
            container.innerHTML = `<div class="loading">
                <h3>${isNoRealData ? '': '❌ No underdog opportunities found'}</h3>
                <p>${isNoRealData ? 'No live tournaments currently available.' : 'Try refreshing or check back later'}</p>
                ${isNoRealData ? '<p style="color: #6bcf7f; margin-top: 10px;">✨ System working correctly </p>' : ''}
            </div>`;
            
            // Reset stats when no real data
            if (isNoRealData) {
                document.getElementById('underdog-count').textContent = '0';
                document.getElementById('avg-probability').textContent = '0%';
                document.getElementById('excellent-quality').textContent = '0';
            }
        }
    } catch (error) {
        container.innerHTML = '<div class="loading"><h3>❌ Error loading opportunities</h3><p>Connection issues detected</p></div>';
        console.error('Matches error:', error);
    }
}

async function testUnderdogAnalysis() {
    try {
        const response = await fetch(API_BASE + '/test-underdog', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                player1: 'Fabio Fognini',
                player2: 'Carlos Alcaraz',
                tournament: 'US Open',
                surface: 'Hard'
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            const analysis = data.underdog_analysis;
            const scenario = analysis.underdog_scenario;
            
            let message = `🎯 UNDERDOG ANALYSIS TEST\\n\\n`;
            message += `Match: ${data.match_info.player1} vs ${data.match_info.player2}\\n`;
            message += `Underdog: ${scenario.underdog} (Rank #${scenario.underdog_rank})\\n`;
            message += `Favorite: ${scenario.favorite} (Rank #${scenario.favorite_rank})\\n`;
            message += `Type: ${scenario.underdog_type}\\n`;
            message += `Set Probability: ${(analysis.underdog_probability * 100).toFixed(1)}%\\n`;
            message += `Quality: ${analysis.quality}\\n`;
            message += `ML System: ${analysis.ml_system_used}\\n\\n`;
            message += `✅ Underdog analysis working correctly!`;
            
            alert(message);
        } else {
            alert(`❌ Test failed: ${data.error}`);
        }
    } catch (error) {
        alert(`❌ Test error: ${error.message}`);
    }
}

async function manualAPIUpdate() {
    try {
        const response = await fetch(API_BASE + '/manual-api-update', { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            alert('✅ Manual API update triggered! Fresh data will be available on next request.');
            document.getElementById('api-status').textContent = '🔄 Updating';
        } else {
            alert(`❌ Update failed: ${data.error}`);
        }
    } catch (error) {
        alert(`❌ Update error: ${error.message}`);
    }
}

async function checkAPIStatus() {
    try {
        const response = await fetch(API_BASE + '/api-economy-status');
        const data = await response.json();
        
        if (data.success) {
            const usage = data.api_usage;
            document.getElementById('api-status').textContent = `${usage.remaining_hour}/${usage.max_per_hour}`;
            
            alert(`📊 API Economy Status:\\n\\nRequests this hour: ${usage.requests_this_hour}/${usage.max_per_hour}\\nRemaining: ${usage.remaining_hour}\\nCache items: ${usage.cache_items}\\nManual update: ${usage.manual_update_status}`);
        } else {
            document.getElementById('api-status').textContent = '❌ Error';
            alert('❌ Failed to get API status');
        }
    } catch (error) {
        document.getElementById('api-status').textContent = '❌ Error';
        alert(`❌ Status error: ${error.message}`);
    }
}

// Auto-load on page ready
document.addEventListener('DOMContentLoaded', function() {
    loadUnderdogOpportunities();
    checkAPIStatus().catch(console.error);
    setInterval(loadUnderdogOpportunities, 120000);
});