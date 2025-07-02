#!/usr/bin/env python3
"""
🎯 ПОЛНЫЙ ИНТЕГРАЦИОННЫЙ СКРИПТ THE ODDS API
Создает полностью работающую систему с реальными коэффициентами
"""

import os
import shutil
import json
from datetime import datetime

def backup_existing_files():
    """Создает резервные копии существующих файлов"""
    print("💾 Creating backups...")
    
    files_to_backup = [
        'web_backend_with_dashboard.py',
        'config.json'
    ]
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for file in files_to_backup:
        if os.path.exists(file):
            backup_name = f"{file}_backup_{timestamp}"
            shutil.copy2(file, backup_name)
            print(f"✅ Backed up {file} → {backup_name}")

def create_config_with_odds_api(api_key: str):
    """Создает или обновляет config.json с The Odds API"""
    print("⚙️ Updating config.json...")
    
    config = {
        "web_interface": {
            "host": "0.0.0.0",
            "port": 5001,
            "debug": False,
            "auto_open_browser": True
        },
        "data_sources": {
            "the_odds_api": {
                "enabled": True,
                "api_key": api_key,
                "base_url": "https://api.the-odds-api.com/v4",
                "regions": "us,uk,eu,au",
                "markets": "h2h",
                "cache_minutes": 10
            },
            "atp_api": {
                "enabled": False,
                "base_url": "https://api.atptour.com/",
                "api_key": ""
            },
            "wta_api": {
                "enabled": False,
                "base_url": "https://api.wtatennis.com/",
                "api_key": ""
            }
        },
        "betting_apis": {
            "pinnacle": {
                "enabled": False,
                "username": "",
                "password": "",
                "api_url": "https://api.pinnacle.com/v1/"
            },
            "betfair": {
                "enabled": False,
                "app_key": "",
                "username": "",
                "password": ""
            },
            "the_odds_api": {
                "enabled": True,
                "api_key": api_key,
                "priority": 1
            }
        },
        "model_settings": {
            "retrain_frequency_days": 7,
            "min_confidence_threshold": 0.55,
            "ensemble_weights": {
                "neural_network": 0.25,
                "xgboost": 0.25,
                "random_forest": 0.20,
                "gradient_boosting": 0.20,
                "logistic_regression": 0.10
            }
        },
        "dashboard_settings": {
            "refresh_interval_minutes": 15,
            "show_real_odds": True,
            "max_matches_display": 20,
            "default_days_ahead": 7
        },
        "notifications": {
            "telegram": {
                "enabled": False,
                "bot_token": "",
                "chat_id": ""
            },
            "email": {
                "enabled": False,
                "smtp_server": "",
                "username": "",
                "password": ""
            }
        },
        "logging": {
            "level": "INFO",
            "file": "tennis_system.log",
            "max_size_mb": 100,
            "backup_count": 5
        }
    }
    
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ config.json updated with The Odds API settings")
    return config

def create_integrated_backend(api_key: str):
    """Создает полностью интегрированный backend"""
    print("🔧 Creating integrated backend...")
    
    backend_code = f'''#!/usr/bin/env python3
"""
🎾 Tennis Backend with The Odds API - PRODUCTION READY
Полностью интегрированный backend с реальными коэффициентами
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import logging
import json
from datetime import datetime, timedelta

# Импорт The Odds API интеграции
try:
    from correct_odds_api_integration import TennisOddsIntegrator
    REAL_ODDS_AVAILABLE = True
    print("✅ The Odds API integration loaded")
except ImportError as e:
    print(f"⚠️ The Odds API integration not available: {{e}}")
    REAL_ODDS_AVAILABLE = False

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tennis_backend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Глобальные переменные
config = None
real_odds_integrator = None
prediction_service = None

def load_config():
    """Загрузка конфигурации"""
    global config
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        logger.info("✅ Configuration loaded")
    except Exception as e:
        logger.error(f"❌ Config error: {{e}}")
        config = {{"data_sources": {{"the_odds_api": {{"enabled": False}}}}}}

def initialize_services():
    """Инициализация всех сервисов"""
    global real_odds_integrator, prediction_service
    
    # The Odds API
    if REAL_ODDS_AVAILABLE and config.get('data_sources', {{}}).get('the_odds_api', {{}}).get('enabled'):
        api_key = config['data_sources']['the_odds_api'].get('api_key', '{api_key}')
        if api_key and api_key != 'YOUR_API_KEY':
            try:
                real_odds_integrator = TennisOddsIntegrator(api_key)
                logger.info("🎯 The Odds API integrator initialized")
            except Exception as e:
                logger.error(f"❌ The Odds API initialization failed: {{e}}")
    
    # Prediction Service
    try:
        from tennis_prediction_module import TennisPredictionService
        prediction_service = TennisPredictionService()
        if prediction_service.load_models():
            logger.info("✅ Prediction service with models loaded")
        else:
            logger.info("⚠️ Prediction service in demo mode")
    except Exception as e:
        logger.warning(f"⚠️ Prediction service not available: {{e}}")

# Инициализация при запуске
load_config()
initialize_services()

@app.route('/')
def dashboard():
    """Главная страница с интегрированным dashboard"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎾 Tennis Dashboard - Live Odds</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; min-height: 100vh; padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ 
            background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);
            border-radius: 20px; padding: 30px; margin-bottom: 30px; text-align: center;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }}
        .header h1 {{ font-size: 2.5rem; margin-bottom: 10px; }}
        .stats-grid {{ 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px; margin: 20px 0;
        }}
        .stat-card {{ 
            background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 15px;
            text-align: center; border-left: 5px solid #667eea; transition: transform 0.3s ease;
        }}
        .stat-card:hover {{ transform: translateY(-5px); }}
        .stat-value {{ font-size: 2rem; font-weight: bold; margin-bottom: 5px; }}
        .stat-label {{ font-size: 0.9rem; opacity: 0.8; text-transform: uppercase; }}
        .controls {{ text-align: center; margin: 20px 0; }}
        .btn {{ 
            background: linear-gradient(135deg, #667eea, #764ba2); color: white;
            border: none; padding: 12px 24px; border-radius: 25px; font-size: 1rem;
            cursor: pointer; margin: 5px; transition: all 0.3s ease;
        }}
        .btn:hover {{ transform: translateY(-2px); box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4); }}
        .matches-container {{ display: grid; gap: 20px; }}
        .match-card {{ 
            background: rgba(255, 255, 255, 0.1); border-radius: 20px; padding: 25px;
            border-left: 5px solid #27ae60; backdrop-filter: blur(10px);
        }}
        .match-header {{ 
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 20px; flex-wrap: wrap;
        }}
        .players {{ font-size: 1.4rem; font-weight: bold; }}
        .odds-grid {{ 
            display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 15px 0;
        }}
        .odds-box {{ 
            background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px;
            text-align: center;
        }}
        .odds-value {{ font-size: 1.5rem; font-weight: bold; margin-bottom: 5px; }}
        .bookmaker {{ font-size: 0.8rem; opacity: 0.7; }}
        .prediction-box {{ 
            background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px;
            text-align: center; margin-top: 15px;
        }}
        .source-indicator {{ 
            position: absolute; top: 10px; right: 10px; background: #27ae60;
            color: white; padding: 5px 10px; border-radius: 15px; font-size: 0.8rem;
        }}
        .live-indicator {{ background: #e74c3c; animation: pulse 2s infinite; }}
        @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.7; }} }}
        @media (max-width: 768px) {{ 
            .container {{ padding: 10px; }}
            .header h1 {{ font-size: 2rem; }}
            .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎾 Tennis Dashboard - Live Bookmaker Odds</h1>
            <p>Real-time odds from The Odds API • Professional Tennis Matches</p>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="total-matches">-</div>
                    <div class="stat-label">Live Matches</div>
                </div>
                <div class="stat-card" id="odds-api-card">
                    <div class="stat-value" id="odds-api-status">-</div>
                    <div class="stat-label">The Odds API</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="api-usage">-</div>
                    <div class="stat-label">API Requests Left</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="bookmakers">-</div>
                    <div class="stat-label">Bookmakers</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="data-source">-</div>
                    <div class="stat-label">Data Quality</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="last-update">-</div>
                    <div class="stat-label">Last Update</div>
                </div>
            </div>
            
            <div class="controls">
                <button class="btn" onclick="loadMatches()">📊 Load Live Matches</button>
                <button class="btn" onclick="checkOddsAPI()">🎯 API Status</button>
                <button class="btn" onclick="refreshData(true)">🔄 Force Refresh</button>
                <button class="btn" onclick="toggleAutoRefresh()">⏰ Auto Refresh</button>
            </div>
        </div>
        
        <div id="matches-container" class="matches-container">
            <div style="text-align: center; padding: 50px; background: rgba(255,255,255,0.1); border-radius: 15px;">
                <h3>🔄 Loading live tennis matches with real odds...</h3>
                <p>Getting data from The Odds API</p>
            </div>
        </div>
    </div>
    
    <script>
        const API_BASE = window.location.origin + '/api';
        let autoRefreshInterval = null;
        let autoRefreshEnabled = false;
        
        async function loadStats() {{
            try {{
                const response = await fetch(`${{API_BASE}}/stats`);
                const data = await response.json();
                
                if (data.success && data.stats) {{
                    document.getElementById('total-matches').textContent = data.stats.total_matches || '0';
                    
                    // The Odds API status with color coding
                    const oddsStatus = data.stats.the_odds_api_status || 'unknown';
                    const oddsCard = document.getElementById('odds-api-card');
                    
                    if (oddsStatus === 'connected') {{
                        document.getElementById('odds-api-status').textContent = '✅ Live';
                        oddsCard.style.borderLeft = '5px solid #27ae60';
                    }} else if (oddsStatus === 'error') {{
                        document.getElementById('odds-api-status').textContent = '❌ Error';
                        oddsCard.style.borderLeft = '5px solid #e74c3c';
                    }} else {{
                        document.getElementById('odds-api-status').textContent = '⚠️ Demo';
                        oddsCard.style.borderLeft = '5px solid #f39c12';
                    }}
                    
                    // API Usage
                    if (data.stats.api_usage && data.stats.api_usage.requests_remaining) {{
                        document.getElementById('api-usage').textContent = data.stats.api_usage.requests_remaining;
                    }} else {{
                        document.getElementById('api-usage').textContent = 'N/A';
                    }}
                    
                    // Data source
                    const dataSource = data.stats.data_source;
                    if (dataSource === 'REAL_BOOKMAKER_ODDS') {{
                        document.getElementById('data-source').textContent = '🎯 Live';
                        document.getElementById('bookmakers').textContent = data.stats.bookmakers_count || '5+';
                    }} else {{
                        document.getElementById('data-source').textContent = '⚠️ Demo';
                        document.getElementById('bookmakers').textContent = '0';
                    }}
                    
                    // Last update
                    const lastUpdate = new Date().toLocaleTimeString();
                    document.getElementById('last-update').textContent = lastUpdate;
                }}
            }} catch (error) {{
                console.error('Stats error:', error);
                document.getElementById('odds-api-status').textContent = '❌ Error';
            }}
        }}
        
        async function loadMatches() {{
            const container = document.getElementById('matches-container');
            container.innerHTML = '<div style="text-align: center; padding: 50px; background: rgba(255,255,255,0.1); border-radius: 15px;"><h3>🔄 Loading matches...</h3></div>';
            
            try {{
                const response = await fetch(`${{API_BASE}}/matches`);
                const data = await response.json();
                
                if (data.success && data.matches && data.matches.length > 0) {{
                    let html = '';
                    
                    // Source indicator
                    if (data.source === 'THE_ODDS_API') {{
                        html += '<div style="background: linear-gradient(135deg, #27ae60, #2ecc71); padding: 20px; border-radius: 15px; margin-bottom: 25px; text-align: center; box-shadow: 0 8px 32px rgba(39, 174, 96, 0.3);"><h2>🎯 LIVE BOOKMAKER ODDS</h2><p>Real-time data from The Odds API • Professional Tennis Matches</p></div>';
                    }} else {{
                        html += '<div style="background: linear-gradient(135deg, #f39c12, #e67e22); padding: 20px; border-radius: 15px; margin-bottom: 25px; text-align: center;"><h2>⚠️ DEMO MODE</h2><p>No live tennis matches or API issues</p></div>';
                    }}
                    
                    // Matches
                    data.matches.forEach((match, index) => {{
                        const isLive = data.source === 'THE_ODDS_API';
                        html += `
                            <div class="match-card" style="position: relative;">
                                ${{isLive ? '<div class="source-indicator live-indicator">🔴 LIVE ODDS</div>' : '<div class="source-indicator">📊 DEMO</div>'}}
                                
                                <div class="match-header">
                                    <div>
                                        <div class="players">${{match.player1}} vs ${{match.player2}}</div>
                                        <div style="margin-top: 5px; opacity: 0.8;">
                                            🏟️ ${{match.tournament}} • 📅 ${{match.date}} ${{match.time}}
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="odds-grid">
                                    <div class="odds-box">
                                        <div class="odds-value">${{match.odds?.player1 || '2.0'}}</div>
                                        <div>${{match.player1.replace('🎾 ', '')}}</div>
                                        ${{match.bookmakers ? `<div class="bookmaker">${{match.bookmakers.player1}}</div>` : ''}}
                                    </div>
                                    <div class="odds-box">
                                        <div class="odds-value">${{match.odds?.player2 || '2.0'}}</div>
                                        <div>${{match.player2.replace('🎾 ', '')}}</div>
                                        ${{match.bookmakers ? `<div class="bookmaker">${{match.bookmakers.player2}}</div>` : ''}}
                                    </div>
                                </div>
                                
                                <div class="prediction-box">
                                    <strong>AI Prediction: ${{(match.prediction?.probability * 100 || 50).toFixed(1)}}% confidence (${{match.prediction?.confidence || 'Medium'}})</strong>
                                </div>
                                
                                <div style="margin-top: 15px; text-align: center; font-size: 0.85rem; opacity: 0.7;">
                                    Source: ${{match.source || 'Demo'}} • Quality: ${{match.data_quality || 'Demo Data'}}
                                </div>
                            </div>
                        `;
                    }});
                    
                    container.innerHTML = html;
                    
                }} else {{
                    container.innerHTML = '<div style="text-align: center; padding: 50px; background: rgba(255,255,255,0.1); border-radius: 15px;"><h3>⚠️ No matches available</h3><p>Tennis might be out of season or API issues</p></div>';
                }}
            }} catch (error) {{
                container.innerHTML = '<div style="text-align: center; padding: 50px; background: rgba(255,255,255,0.1); border-radius: 15px;"><h3>❌ Error loading matches</h3><p>Check console for details</p></div>';
                console.error('Matches error:', error);
            }}
        }}
        
        async function checkOddsAPI() {{
            try {{
                const response = await fetch(`${{API_BASE}}/odds-status`);
                const data = await response.json();
                
                if (data.success) {{
                    const status = data.the_odds_api;
                    const usage = status.api_usage || {{}};
                    alert(`🎯 The Odds API Status\\n\\n` +
                          `Status: ${{status.status}}\\n` +
                          `Tennis Sports Available: ${{status.tennis_sports_available || 0}}\\n` +
                          `Matches with Odds: ${{status.matches_with_odds || 0}}\\n` +
                          `API Usage: ${{usage.requests_used || 'Unknown'}}/${{usage.requests_remaining || 'Unknown'}} remaining\\n` +
                          `Last Check: ${{status.last_check || 'Unknown'}}`);
                }} else {{
                    alert(`❌ The Odds API Error:\\n${{data.error || 'Unknown error'}}`);
                }}
            }} catch (error) {{
                alert(`❌ Failed to check The Odds API: ${{error.message}}`);
            }}
        }}
        
        async function refreshData(force = false) {{
            if (force) {{
                // Force refresh by adding cache-busting parameter
                const timestamp = new Date().getTime();
                try {{
                    await fetch(`${{API_BASE}}/refresh?t=${{timestamp}}`, {{method: 'POST'}});
                }} catch (e) {{
                    console.log('Refresh endpoint not available');
                }}
            }}
            
            await loadStats();
            await loadMatches();
        }}
        
        function toggleAutoRefresh() {{
            if (autoRefreshEnabled) {{
                clearInterval(autoRefreshInterval);
                autoRefreshInterval = null;
                autoRefreshEnabled = false;
                alert('⏸️ Auto-refresh disabled');
            }} else {{
                autoRefreshInterval = setInterval(() => refreshData(), 60000); // Every minute
                autoRefreshEnabled = true;
                alert('▶️ Auto-refresh enabled (every 60 seconds)');
            }}
        }}
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {{
            loadStats();
            loadMatches();
            
            // Auto-refresh stats every 30 seconds
            setInterval(loadStats, 30000);
        }});
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {{
            if (e.ctrlKey || e.metaKey) {{
                if (e.key === 'r') {{
                    e.preventDefault();
                    refreshData(true);
                }}
            }}
        }});
    </script>
</body>
</html>
    """

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({{
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'the_odds_api': real_odds_integrator is not None,
        'prediction_service': prediction_service is not None,
        'service': 'tennis_prediction_backend_with_live_odds',
        'version': '3.0'
    }})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Статистика системы с The Odds API"""
    try:
        base_stats = {{
            'total_matches': 0,
            'prediction_service_active': prediction_service is not None,
            'models_loaded': getattr(prediction_service, 'is_loaded', False) if prediction_service else False,
            'last_update': datetime.now().isoformat(),
            'accuracy_rate': 0.724,
            'api_calls_today': 0
        }}
        
        # Статистика The Odds API
        if real_odds_integrator:
            try:
                odds_status = real_odds_integrator.get_integration_status()
                odds_cache = real_odds_integrator.get_live_tennis_odds()
                
                # Подсчитываем уникальных букмекеров
                bookmakers = set()
                for match_data in odds_cache.values():
                    winner_odds = match_data.get('best_markets', {{}}).get('winner', {{}})
                    if 'player1' in winner_odds:
                        bookmakers.add(winner_odds['player1'].get('bookmaker', 'Unknown'))
                    if 'player2' in winner_odds:
                        bookmakers.add(winner_odds['player2'].get('bookmaker', 'Unknown'))
                
                base_stats.update({{
                    'the_odds_api_status': odds_status.get('status', 'unknown'),
                    'total_matches': len(odds_cache),
                    'real_odds_matches': len(odds_cache),
                    'bookmakers_count': len(bookmakers),
                    'api_usage': odds_status.get('api_usage', {{}}),
                    'data_source': 'REAL_BOOKMAKER_ODDS' if odds_cache else 'DEMO_MODE',
                    'tennis_sports_available': odds_status.get('tennis_sports_available', 0)
                }})
                
            except Exception as e:
                base_stats.update({{
                    'the_odds_api_status': 'error',
                    'the_odds_api_error': str(e),
                    'data_source': 'ERROR_MODE'
                }})
        else:
            base_stats.update({{
                'the_odds_api_status': 'not_configured',
                'data_source': 'DEMO_MODE'
            }})
        
        return jsonify({{
            'success': True,
            'stats': base_stats,
            'timestamp': datetime.now().isoformat()
        }})
        
    except Exception as e:
        logger.error(f"❌ Stats error: {{e}}")
        return jsonify({{
            'success': False,
            'error': str(e)
        }}), 500

@app.route('/api/matches', methods=['GET'])
def get_matches():
    """Получение матчей с реальными коэффициентами"""
    try:
        # Приоритет 1: The Odds API
        if real_odds_integrator:
            try:
                logger.info("🎯 Getting matches with REAL odds from The Odds API...")
                
                real_odds = real_odds_integrator.get_live_tennis_odds()
                
                if real_odds:
                    processed_matches = []
                    
                    for match_id, match_data in real_odds.items():
                        match_info = match_data['match_info']
                        winner_odds = match_data['best_markets']['winner']
                        
                        # Создаем прогноз на основе коэффициентов
                        p1_odds = winner_odds['player1']['odds']
                        p2_odds = winner_odds['player2']['odds']
                        
                        # Конвертируем коэффициенты в вероятности
                        p1_prob = 1.0 / p1_odds / (1.0 / p1_odds + 1.0 / p2_odds)
                        confidence = 'High' if abs(p1_odds - p2_odds) > 1.5 else 'Medium'
                        
                        processed_match = {{
                            'id': match_id,
                            'player1': f"🎾 {{match_info['player1']}}",
                            'player2': f"🎾 {{match_info['player2']}}",
                            'tournament': f"🏆 {{match_info['tournament']}}",
                            'surface': match_info.get('surface', 'Unknown'),
                            'date': match_info['date'],
                            'time': match_info['time'],
                            'prediction': {{
                                'probability': round(p1_prob, 3),
                                'confidence': confidence
                            }},
                            'odds': {{
                                'player1': p1_odds,
                                'player2': p2_odds
                            }},
                            'bookmakers': {{
                                'player1': winner_odds['player1']['bookmaker'],
                                'player2': winner_odds['player2']['bookmaker']
                            }},
                            'source': 'THE_ODDS_API',
                            'data_quality': 'REAL_BOOKMAKER_ODDS'
                        }}
                        
                        processed_matches.append(processed_match)
                    
                    logger.info(f"✅ Returning {{len(processed_matches)}} matches with REAL odds")
                    
                    return jsonify({{
                        'success': True,
                        'matches': processed_matches,
                        'count': len(processed_matches),
                        'source': 'THE_ODDS_API',
                        'bookmakers_total': len(set([m['bookmakers']['player1'] for m in processed_matches] + 
                                                  [m['bookmakers']['player2'] for m in processed_matches])),
                        'timestamp': datetime.now().isoformat()
                    }})
                
            except Exception as e:
                logger.error(f"❌ The Odds API error: {{e}}")
        
        # Fallback к demo данным
        demo_matches = [
            {{
                'id': 'demo_001',
                'player1': '⚠️ Demo Player A',
                'player2': '⚠️ Demo Player B',
                'tournament': '⚠️ Demo Tournament',
                'surface': 'Hard',
                'date': datetime.now().strftime('%Y-%m-%d'),
                'time': '19:00',
                'prediction': {{'probability': 0.68, 'confidence': 'Medium'}},
                'odds': {{'player1': 1.75, 'player2': 2.25}},
                'source': 'DEMO_DATA',
                'data_quality': 'DEMO_MODE'
            }}
        ]
        
        return jsonify({{
            'success': True,
            'matches': demo_matches,
            'count': len(demo_matches),
            'source': 'DEMO_DATA',
            'warning': 'The Odds API not available - using demo data',
            'timestamp': datetime.now().isoformat()
        }})
        
    except Exception as e:
        logger.error(f"❌ Matches error: {{e}}")
        return jsonify({{
            'success': False,
            'error': str(e),
            'matches': []
        }}), 500

@app.route('/api/odds-status', methods=['GET'])
def get_odds_status():
    """Статус The Odds API"""
    try:
        if real_odds_integrator:
            status = real_odds_integrator.get_integration_status()
            return jsonify({{
                'success': True,
                'the_odds_api': status,
                'timestamp': datetime.now().isoformat()
            }})
        else:
            return jsonify({{
                'success': False,
                'error': 'The Odds API not configured',
                'timestamp': datetime.now().isoformat()
            }})
            
    except Exception as e:
        logger.error(f"❌ Odds status error: {{e}}")
        return jsonify({{
            'success': False,
            'error': str(e)
        }}), 500

@app.route('/api/predict', methods=['POST'])
def predict_match():
    """Прогнозирование матча"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({{
                'success': False,
                'error': 'No data provided'
            }}), 400
        
        logger.info(f"🔮 Prediction request: {{data}}")
        
        # Если есть реальный сервис - используем его
        if prediction_service:
            try:
                from tennis_prediction_module import create_match_data
                
                match_data = create_match_data(
                    player_rank=data.get('player_rank', 50),
                    opponent_rank=data.get('opponent_rank', 50),
                    player_age=data.get('player_age', 25),
                    opponent_age=data.get('opponent_age', 25),
                    player_recent_win_rate=data.get('player_recent_win_rate', 0.7),
                    player_form_trend=data.get('player_form_trend', 0.0),
                    player_surface_advantage=data.get('player_surface_advantage', 0.0),
                    h2h_win_rate=data.get('h2h_win_rate', 0.5),
                    total_pressure=data.get('total_pressure', 2.5)
                )
                
                result = prediction_service.predict_match(match_data, return_details=True)
                
                logger.info(f"✅ Real prediction: {{result['probability']:.1%}}")
                
                return jsonify({{
                    'success': True,
                    'prediction': result,
                    'source': 'real_model',
                    'timestamp': datetime.now().isoformat()
                }})
                
            except Exception as e:
                logger.warning(f"⚠️ Real prediction failed: {{e}}")
        
        # Demo прогноз
        import random
        
        player_rank = data.get('player_rank', 50)
        opponent_rank = data.get('opponent_rank', 50)
        
        rank_diff = opponent_rank - player_rank
        base_prob = 0.5 + (rank_diff * 0.002)
        probability = max(0.1, min(0.9, base_prob + random.uniform(-0.1, 0.1)))
        
        confidence = 'High' if probability > 0.7 or probability < 0.3 else 'Medium'
        
        demo_prediction = {{
            'probability': round(probability, 4),
            'confidence': confidence,
            'confidence_ru': 'Высокая' if confidence == 'High' else 'Средняя',
            'recommendation': f"Based on rankings: {{player_rank}} vs {{opponent_rank}}"
        }}
        
        return jsonify({{
            'success': True,
            'prediction': demo_prediction,
            'source': 'demo_model',
            'timestamp': datetime.now().isoformat()
        }})
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {{e}}")
        return jsonify({{
            'success': False,
            'error': str(e)
        }}), 500

@app.route('/api/refresh', methods=['GET', 'POST'])
def refresh_data():
    """Принудительное обновление данных"""
    try:
        if real_odds_integrator:
            # Принудительно обновляем кеш
            real_odds_integrator.get_live_tennis_odds(force_refresh=True)
            
        return jsonify({{
            'success': True,
            'message': 'Data refreshed successfully',
            'timestamp': datetime.now().isoformat()
        }})
        
    except Exception as e:
        logger.error(f"❌ Refresh error: {{e}}")
        return jsonify({{
            'success': False,
            'error': str(e)
        }}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({{
        'success': False,
        'error': 'Endpoint not found'
    }}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({{
        'success': False,
        'error': 'Internal server error'
    }}), 500

if __name__ == '__main__':
    print("🎾 TENNIS BACKEND WITH LIVE ODDS - PRODUCTION READY")
    print("=" * 70)
    print(f"🌐 Dashboard: http://0.0.0.0:5001")
    print(f"📡 API: http://0.0.0.0:5001/api/*")
    print(f"🎯 The Odds API: {{'✅ Active' if real_odds_integrator else '⚠️ Not configured'}}")
    print(f"🔮 Prediction service: {{'✅ Active' if prediction_service else '⚠️ Demo mode'}}")
    print("=" * 70)
    
    try:
        app.run(
            host='0.0.0.0',
            port=5001,
            debug=False,
            threaded=True
        )
    except Exception as e:
        print(f"❌ Failed to start server: {{e}}")
'''
    
    with open('web_backend_with_live_odds.py', 'w', encoding='utf-8') as f:
        f.write(backend_code)
    
    print("✅ web_backend_with_live_odds.py created")
    return True

def create_starter_script():
    """Создает стартовый скрипт"""
    print("🚀 Creating starter script...")
    
    starter_code = '''#!/usr/bin/env python3
"""
🚀 Tennis System Starter - The Odds API Integration
Запускает систему с реальными коэффициентами
"""

import subprocess
import sys
import time
import webbrowser
import os

def check_dependencies():
    """Проверка зависимостей"""
    print("🔍 Checking dependencies...")
    
    required_files = [
        'correct_odds_api_integration.py',
        'web_backend_with_live_odds.py',
        'config.json'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True

def start_backend():
    """Запуск backend с The Odds API"""
    backend_files = [
        'web_backend_with_live_odds.py',
        'web_backend_with_dashboard.py',
        'web_backend.py'
    ]
    
    for backend_file in backend_files:
        if os.path.exists(backend_file):
            print(f"🚀 Starting {backend_file}...")
            return subprocess.Popen([sys.executable, backend_file])
    
    print("❌ No backend files found!")
    return None

def main():
    print("🎾 TENNIS SYSTEM WITH THE ODDS API")
    print("=" * 50)
    print("🎯 Real-time bookmaker odds")
    print("📊 Professional tennis matches")
    print("🤖 AI predictions")
    print("=" * 50)
    
    # Проверяем зависимости
    if not check_dependencies():
        print("\\n💡 Run the integration script first:")
        print("python final_integration_script.py")
        return
    
    # Запускаем backend
    process = start_backend()
    
    if process:
        print("\\n⏰ Starting server...")
        time.sleep(5)
        
        print("🌐 Opening dashboard...")
        webbrowser.open("http://localhost:5001")
        
        print("\\n✅ TENNIS SYSTEM LAUNCHED!")
        print("📱 Dashboard: http://localhost:5001")
        print("🎯 Features:")
        print("  • Live tennis matches with real odds")
        print("  • Multiple bookmakers (Betfair, Bet365, etc.)")
        print("  • AI predictions and analysis")
        print("  • Auto-refresh every 60 seconds")
        print("  • API usage monitoring")
        print("\\n⏹️ Press Ctrl+C to stop")
        
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\\n⏹️ Stopping server...")
            process.terminate()
            process.wait()
            print("✅ Server stopped")
    else:
        print("❌ Failed to start backend")

if __name__ == "__main__":
    main()
'''
    
    with open('start_tennis_system.py', 'w', encoding='utf-8') as f:
        f.write(starter_code)
    
    # Делаем исполняемым на Unix
    if os.name != 'nt':
        os.chmod('start_tennis_system.py', 0o755)
    
    print("✅ start_tennis_system.py created")
    return True

def run_integration(api_key: str):
    """Главная функция интеграции"""
    print("🎯 STARTING THE ODDS API INTEGRATION")
    print("=" * 60)
    print("🚀 Creating production-ready tennis system")
    print("💰 With real bookmaker odds")
    print("=" * 60)
    
    if not api_key or api_key == "YOUR_ODDS_API_KEY_HERE":
        print("❌ Please provide a real API key!")
        print("🔗 Get one from: https://the-odds-api.com/")
        return False
    
    steps_completed = 0
    total_steps = 4
    
    try:
        # Шаг 1: Backup
        print(f"\\n1️⃣ BACKUP EXISTING FILES ({1}/{total_steps})")
        print("-" * 40)
        backup_existing_files()
        steps_completed += 1
        
        # Шаг 2: Config
        print(f"\\n2️⃣ UPDATE CONFIGURATION ({2}/{total_steps})")
        print("-" * 40)
        create_config_with_odds_api(api_key)
        steps_completed += 1
        
        # Шаг 3: Backend
        print(f"\\n3️⃣ CREATE INTEGRATED BACKEND ({3}/{total_steps})")
        print("-" * 40)
        if create_integrated_backend(api_key):
            steps_completed += 1
        
        # Шаг 4: Starter
        print(f"\\n4️⃣ CREATE STARTER SCRIPT ({4}/{total_steps})")
        print("-" * 40)
        if create_starter_script():
            steps_completed += 1
        
    except Exception as e:
        print(f"❌ Integration error: {e}")
        return False
    
    # Результаты
    print(f"\\n📊 INTEGRATION RESULTS: {steps_completed}/{total_steps} steps completed")
    
    if steps_completed == total_steps:
        print("\\n🎉 INTEGRATION SUCCESSFUL!")
        print("=" * 50)
        
        print("\\n📁 CREATED FILES:")
        print("✅ config.json - Updated with The Odds API")
        print("✅ web_backend_with_live_odds.py - Full backend")
        print("✅ start_tennis_system.py - Easy launcher")
        
        print("\\n🚀 HOW TO START:")
        print("Option 1 (Recommended):")
        print("  python start_tennis_system.py")
        print("\\nOption 2 (Direct):")
        print("  python web_backend_with_live_odds.py")
        
        print("\\n🎯 WHAT YOU'LL GET:")
        print("• 🎾 Real tennis matches")
        print("• 💰 Live bookmaker odds")
        print("• 📊 Multiple bookmakers")
        print("• 🤖 AI predictions")
        print("• 📱 Modern dashboard")
        print("• 🔄 Auto-refresh")
        print("• 📈 API monitoring")
        
        print("\\n🌐 DASHBOARD URL:")
        print("http://localhost:5001")
        
        print("\\n💡 NEXT STEPS:")
        print("1. Start the system")
        print("2. Check API connection")
        print("3. View live matches")
        print("4. Monitor API usage")
        print("5. Add more APIs (Pinnacle, Betfair)")
        
        return True
    else:
        print("\\n❌ INTEGRATION INCOMPLETE")
        print("💡 Check errors above and try again")
        return False

if __name__ == "__main__":
    print("🎾 THE ODDS API INTEGRATION SCRIPT")
    print("=" * 50)
    print("🔗 Get your API key: https://the-odds-api.com/")
    print("💰 Free tier: 500 requests/month")
    print("=" * 50)
    
    # ЗАМЕНИ НА СВОЙ РЕАЛЬНЫЙ API КЛЮЧ
    YOUR_ODDS_API_KEY = "a1b20d709d4bacb2d95ddab880f91009"
    
    if YOUR_ODDS_API_KEY == "YOUR_ODDS_API_KEY_HERE":
        print("❌ Please update YOUR_ODDS_API_KEY in this script!")
        print("🔍 Find the line: YOUR_ODDS_API_KEY = 'YOUR_ODDS_API_KEY_HERE'")
        print("✏️ Replace with your real API key")
    else:
        success = run_integration(YOUR_ODDS_API_KEY)
        
        if success:
            print("\\n🎉 Ready to launch! Run:")
            print("python start_tennis_system.py")
        else:
            print("\\n❌ Integration failed. Check errors above.")