#!/usr/bin/env python3
"""
🔍 DEBUG Tennis Backend - Диагностика проблем
Проверяем почему не загружаются матчи
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import traceback
from datetime import datetime
import random

# Подробное логирование
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class DebugTennisPredictor:
    """Отладочная версия предиктора"""
    
    def __init__(self):
        logger.info("🔍 Initializing DebugTennisPredictor...")
        
        self.players = {
            'carlos alcaraz': {
                'rank': 2, 'form': 0.85, 'age': 21,
                'grass_adv': 0.05, 'big_match': 0.8, 'stamina': 0.9
            },
            'novak djokovic': {
                'rank': 5, 'form': 0.78, 'age': 37,
                'grass_adv': 0.18, 'big_match': 0.95, 'stamina': 0.85
            },
            'jannik sinner': {
                'rank': 1, 'form': 0.88, 'age': 23,
                'grass_adv': 0.08, 'big_match': 0.75, 'stamina': 0.9
            },
            'daniil medvedev': {
                'rank': 4, 'form': 0.72, 'age': 28,
                'grass_adv': -0.05, 'big_match': 0.85, 'stamina': 0.8
            }
        }
        
        logger.info(f"✅ Loaded {len(self.players)} players")
    
    def get_player_data(self, name):
        """Получение данных игрока с отладкой"""
        logger.debug(f"🔍 Looking up player: {name}")
        
        name_clean = name.lower().strip()
        
        if name_clean in self.players:
            logger.debug(f"✅ Found exact match for: {name}")
            return self.players[name_clean]
        
        # Поиск по частям
        for player_name, data in self.players.items():
            if any(part in player_name for part in name_clean.split()):
                logger.debug(f"✅ Found partial match: {player_name} for {name}")
                return data
        
        logger.warning(f"⚠️ Player not found: {name}, using default data")
        return {
            'rank': 30, 'form': 0.65, 'age': 25,
            'grass_adv': 0.0, 'big_match': 0.5, 'stamina': 0.75
        }
    
    def predict_match(self, player1, player2, tournament, surface):
        """Прогноз с подробной отладкой"""
        logger.info(f"🎾 Predicting: {player1} vs {player2} at {tournament} ({surface})")
        
        try:
            p1_data = self.get_player_data(player1)
            p2_data = self.get_player_data(player2)
            
            logger.debug(f"Player 1 data: {p1_data}")
            logger.debug(f"Player 2 data: {p2_data}")
            
            # Простые расчеты для отладки
            rank_factor = (p2_data['rank'] - p1_data['rank']) * 0.01
            form_factor = (p1_data['form'] - p2_data['form']) * 0.5
            
            base_probability = 0.5 + rank_factor + form_factor
            probability = max(0.2, min(0.8, base_probability))
            
            confidence = 'High' if abs(probability - 0.5) > 0.2 else 'Medium'
            
            factors = [
                f"🏆 Rank difference: {p1_data['rank']} vs {p2_data['rank']}",
                f"🔥 Form: {p1_data['form']:.1%} vs {p2_data['form']:.1%}"
            ]
            
            result = {
                'probability': round(probability, 3),
                'confidence': confidence,
                'key_factors': factors,
                'prediction_type': 'DEBUG_ANALYSIS'
            }
            
            logger.info(f"✅ Prediction successful: {probability:.1%} ({confidence})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            logger.error(traceback.format_exc())
            raise

# Глобальный предиктор
predictor = None

def init_predictor():
    """Инициализация с отладкой"""
    global predictor
    try:
        logger.info("🚀 Initializing predictor...")
        predictor = DebugTennisPredictor()
        logger.info("✅ Predictor initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Predictor initialization failed: {e}")
        logger.error(traceback.format_exc())
        return False

# Инициализируем при старте
init_success = init_predictor()

@app.route('/')
def dashboard():
    """Debug dashboard с дополнительной информацией"""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔍 Tennis Debug Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; min-height: 100vh; padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { 
            background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);
            border-radius: 20px; padding: 30px; margin-bottom: 30px; text-align: center;
        }
        .debug-info {
            background: rgba(255, 0, 0, 0.2); padding: 15px; border-radius: 10px;
            margin-bottom: 20px; border: 2px solid #e74c3c;
        }
        .success { background: rgba(0, 255, 0, 0.2); border-color: #27ae60; }
        .controls { text-align: center; margin: 20px 0; }
        .btn { 
            background: linear-gradient(135deg, #667eea, #764ba2); color: white;
            border: none; padding: 12px 24px; border-radius: 25px; font-size: 1rem;
            cursor: pointer; margin: 5px; transition: all 0.3s ease;
        }
        .btn:hover { transform: translateY(-2px); }
        .matches-container { display: grid; gap: 20px; }
        .match-card { 
            background: rgba(255, 255, 255, 0.1); border-radius: 20px; padding: 25px;
            border-left: 5px solid #27ae60;
        }
        .error { color: #e74c3c; }
        .success-text { color: #27ae60; }
        .debug-log {
            background: rgba(0, 0, 0, 0.5); padding: 15px; border-radius: 10px;
            font-family: monospace; font-size: 0.9rem; margin-top: 10px;
            white-space: pre-wrap; max-height: 200px; overflow-y: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Tennis Debug Dashboard</h1>
            <p>Диагностика проблем загрузки матчей</p>
            
            <div id="debug-status" class="debug-info">
                <strong>🔍 Debug Status: Checking...</strong>
            </div>
            
            <div class="controls">
                <button class="btn" onclick="testAPI()">🔌 Test API</button>
                <button class="btn" onclick="testPredictor()">🧠 Test Predictor</button>
                <button class="btn" onclick="loadMatches()">🎾 Load Matches</button>
                <button class="btn" onclick="showLogs()">📋 Show Logs</button>
            </div>
        </div>
        
        <div id="matches-container" class="matches-container">
            <div style="text-align: center; padding: 50px;">
                <h3>🔍 Ready for debugging...</h3>
                <p>Click buttons above to test different components</p>
            </div>
        </div>
        
        <div id="debug-logs" class="debug-log" style="display: none;"></div>
    </div>
    
    <script>
        const API_BASE = window.location.origin + '/api';
        let debugLogs = [];
        
        function addLog(message, type = 'info') {
            const timestamp = new Date().toLocaleTimeString();
            const logEntry = `[${timestamp}] ${type.toUpperCase()}: ${message}`;
            debugLogs.push(logEntry);
            console.log(logEntry);
        }
        
        async function testAPI() {
            addLog('Testing API connection...');
            
            try {
                const response = await fetch(`${API_BASE}/health`);
                const data = await response.json();
                
                if (response.ok) {
                    addLog(`API Health: ${data.status}`, 'success');
                    document.getElementById('debug-status').innerHTML = 
                        '<strong class="success-text">✅ API Connection: OK</strong>';
                    document.getElementById('debug-status').className = 'debug-info success';
                } else {
                    addLog(`API Error: ${response.status}`, 'error');
                }
            } catch (error) {
                addLog(`API Connection Failed: ${error.message}`, 'error');
                document.getElementById('debug-status').innerHTML = 
                    '<strong class="error">❌ API Connection: FAILED</strong>';
            }
        }
        
        async function testPredictor() {
            addLog('Testing predictor...');
            
            try {
                const response = await fetch(`${API_BASE}/test-predictor`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        player1: 'Carlos Alcaraz',
                        player2: 'Novak Djokovic'
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    addLog(`Predictor Test: SUCCESS (${data.prediction.probability})`, 'success');
                    alert(`✅ Predictor Working!\\n\\nPrediction: ${(data.prediction.probability * 100).toFixed(1)}%\\nConfidence: ${data.prediction.confidence}`);
                } else {
                    addLog(`Predictor Test: FAILED (${data.error})`, 'error');
                    alert(`❌ Predictor Failed: ${data.error}`);
                }
            } catch (error) {
                addLog(`Predictor Test Error: ${error.message}`, 'error');
                alert(`❌ Predictor Error: ${error.message}`);
            }
        }
        
        async function loadMatches() {
            addLog('Loading matches...');
            const container = document.getElementById('matches-container');
            container.innerHTML = '<div style="text-align: center; padding: 50px;"><h3>🔄 Loading matches...</h3></div>';
            
            try {
                const response = await fetch(`${API_BASE}/matches`);
                addLog(`Matches API Response: ${response.status}`);
                
                const data = await response.json();
                addLog(`Matches Data: ${JSON.stringify(data, null, 2)}`);
                
                if (data.success && data.matches) {
                    addLog(`Matches loaded: ${data.matches.length}`, 'success');
                    
                    let html = '<h2>🎾 DEBUG MATCHES:</h2>';
                    
                    data.matches.forEach((match, index) => {
                        html += `
                            <div class="match-card">
                                <div style="font-size: 1.4rem; font-weight: bold; margin-bottom: 10px;">
                                    ${match.player1} vs ${match.player2}
                                </div>
                                <div style="opacity: 0.8; margin-bottom: 10px;">
                                    🏆 ${match.tournament} • ${match.surface}
                                </div>
                                <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 8px;">
                                    <strong>Prediction: ${(match.prediction.probability * 100).toFixed(1)}% (${match.prediction.confidence})</strong>
                                </div>
                                <div style="margin-top: 10px; font-size: 0.9rem;">
                                    🔍 Debug Info: ID=${match.id}, Type=${match.prediction_type}
                                </div>
                            </div>
                        `;
                    });
                    
                    container.innerHTML = html;
                } else {
                    addLog(`Matches loading failed: ${data.error || 'Unknown error'}`, 'error');
                    container.innerHTML = '<div style="text-align: center; padding: 50px; color: #e74c3c;"><h3>❌ Failed to load matches</h3><p>Check debug logs for details</p></div>';
                }
            } catch (error) {
                addLog(`Matches loading error: ${error.message}`, 'error');
                container.innerHTML = `<div style="text-align: center; padding: 50px; color: #e74c3c;"><h3>❌ Connection Error</h3><p>${error.message}</p></div>`;
            }
        }
        
        function showLogs() {
            const logsDiv = document.getElementById('debug-logs');
            logsDiv.innerHTML = debugLogs.join('\\n');
            logsDiv.style.display = logsDiv.style.display === 'none' ? 'block' : 'none';
        }
        
        // Auto-test на старте
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(testAPI, 1000);
        });
    </script>
</body>
</html>'''

@app.route('/api/health', methods=['GET'])
def health_check():
    """Подробный health check"""
    logger.info("🔍 Health check requested")
    
    try:
        status = {
            'status': 'healthy',
            'predictor_initialized': predictor is not None,
            'init_success': init_success,
            'timestamp': datetime.now().isoformat(),
            'players_loaded': len(predictor.players) if predictor else 0
        }
        
        logger.info(f"✅ Health check: {status}")
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/test-predictor', methods=['POST'])
def test_predictor():
    """Тест предиктора"""
    logger.info("🧠 Testing predictor...")
    
    try:
        data = request.get_json()
        
        if not predictor:
            logger.error("❌ Predictor not initialized")
            return jsonify({
                'success': False,
                'error': 'Predictor not initialized'
            }), 500
        
        player1 = data.get('player1', 'Carlos Alcaraz')
        player2 = data.get('player2', 'Novak Djokovic')
        
        logger.info(f"🎾 Testing prediction: {player1} vs {player2}")
        
        prediction = predictor.predict_match(player1, player2, 'Test Tournament', 'Hard')
        
        logger.info(f"✅ Test prediction successful: {prediction}")
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Predictor test error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/matches', methods=['GET'])
def get_matches():
    """Получение матчей с подробной отладкой"""
    logger.info("🎾 Matches endpoint called")
    
    try:
        if not predictor:
            logger.error("❌ Predictor not available")
            return jsonify({
                'success': False,
                'error': 'Predictor not initialized',
                'matches': []
            }), 500
        
        # Тестовые матчи
        test_matches = [
            ('Carlos Alcaraz', 'Novak Djokovic', 'Wimbledon', 'Grass'),
            ('Jannik Sinner', 'Daniil Medvedev', 'US Open', 'Hard'),
            ('Alexander Zverev', 'Taylor Fritz', 'ATP Finals', 'Hard')
        ]
        
        logger.info(f"🔍 Processing {len(test_matches)} matches...")
        
        processed_matches = []
        
        for i, (player1, player2, tournament, surface) in enumerate(test_matches):
            logger.info(f"🎾 Processing match {i+1}: {player1} vs {player2}")
            
            try:
                prediction = predictor.predict_match(player1, player2, tournament, surface)
                
                match = {
                    'id': f"debug_{i+1}",
                    'player1': f"🎾 {player1}",
                    'player2': f"🎾 {player2}",
                    'tournament': f"🏆 {tournament}",
                    'surface': surface,
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'time': '14:00',
                    'prediction': prediction,
                    'prediction_type': prediction.get('prediction_type', 'DEBUG')
                }
                
                processed_matches.append(match)
                logger.info(f"✅ Match {i+1} processed successfully")
                
            except Exception as e:
                logger.error(f"❌ Error processing match {i+1}: {e}")
                continue
        
        logger.info(f"✅ Successfully processed {len(processed_matches)} matches")
        
        return jsonify({
            'success': True,
            'matches': processed_matches,
            'count': len(processed_matches),
            'debug_info': {
                'predictor_available': True,
                'players_loaded': len(predictor.players),
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Matches endpoint error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'matches': []
        }), 500

if __name__ == '__main__':
    print("🔍 TENNIS DEBUG BACKEND")
    print("=" * 40)
    print(f"🌐 Dashboard: http://localhost:5001")
    print(f"🔧 Debug mode: ON")
    print(f"📋 Predictor: {'✅' if init_success else '❌'}")
    print("=" * 40)
    
    try:
        app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)
    except Exception as e:
        print(f"❌ Server error: {e}")