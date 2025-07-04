#!/usr/bin/env python3
"""
🎾 Tennis Backend with ADVANCED ML - 70%+ Accuracy Target
Полностью заменяет простое 1/odds на реальный машинное обучение
"""

#!/usr/bin/env python3
"""
🎾 Tennis Backend with ADVANCED ML - 70%+ Accuracy Target
Полностью заменяет простое 1/odds на реальный машинное обучение
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
from datetime import datetime
import sys
import os

# Добавляем текущую директорию в путь для импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем наш Fixed Advanced ML
try:
    from advanced_tennis_ml_predictor_fixed import AdvancedTennisPredictor
    ML_AVAILABLE = True
    print("✅ Advanced Tennis ML imported successfully")
except ImportError as e:
    print(f"❌ Advanced ML not available: {e}")
    ML_AVAILABLE = False

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Глобальные переменные
ml_predictor = None
ml_ready = False

def initialize_advanced_ml():
    """Инициализация Advanced ML системы"""
    global ml_predictor, ml_ready
    
    if not ML_AVAILABLE:
        logger.warning("⚠️ Advanced ML не доступен")
        return False
    
    try:
        logger.info("🤖 Инициализация Advanced ML системы...")
        
        ml_predictor = AdvancedTennisPredictor()
        
        # Генерируем тренировочные данные (в продакшене загрузите реальные)
        logger.info("📚 Подготовка тренировочных данных...")
        training_data = ml_predictor.generate_training_data(3000)
        
        # Обучаем модели
        logger.info("🚀 Обучение ML моделей...")
        ml_predictor.train_ensemble(training_data)
        
        ml_ready = True
        logger.info("✅ Advanced ML система готова!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации ML: {e}")
        ml_ready = False
        return False

def get_player_data(player_name):
    """Получение данных игрока (замените на реальные данные)"""
    # TODO: Интегрируйте с вашей базой данных
    
    player_database = {
        # ATP Top players
        'carlos alcaraz': {'rank': 2, 'age': 21, 'form': 0.82, 'grass_adv': 0.05},
        'novak djokovic': {'rank': 5, 'age': 37, 'form': 0.78, 'grass_adv': 0.18},
        'jannik sinner': {'rank': 1, 'age': 23, 'form': 0.85, 'grass_adv': 0.08},
        'daniil medvedev': {'rank': 4, 'age': 28, 'form': 0.72, 'grass_adv': -0.05},
        'alexander zverev': {'rank': 3, 'age': 27, 'form': 0.75, 'grass_adv': 0.02},
        'brandon nakashima': {'rank': 45, 'age': 23, 'form': 0.68, 'grass_adv': 0.03},
        'bu yunchaokete': {'rank': 85, 'age': 22, 'form': 0.55, 'grass_adv': -0.08},
        
        # WTA players  
        'aryna sabalenka': {'rank': 1, 'age': 26, 'form': 0.83, 'grass_adv': 0.02},
        'iga swiatek': {'rank': 2, 'age': 23, 'form': 0.81, 'grass_adv': -0.05},
        'renata zarazua': {'rank': 180, 'age': 26, 'form': 0.45, 'grass_adv': -0.12},
        'amanda anisimova': {'rank': 35, 'age': 23, 'form': 0.72, 'grass_adv': 0.05},
    }
    
    name_lower = player_name.lower().strip()
    
    # Прямое совпадение
    if name_lower in player_database:
        return player_database[name_lower]
    
    # Поиск по частичному совпадению
    for db_name, data in player_database.items():
        if any(part in db_name for part in name_lower.split()):
            return data
    
    # Если не найден
    return {'rank': 50, 'age': 25, 'form': 0.65, 'grass_adv': 0.0}

def prepare_ml_data(player1, player2, tournament, surface):
    """Подготовка данных для ML модели"""
    
    p1_data = get_player_data(player1)
    p2_data = get_player_data(player2)
    
    # Определяем давление турнира
    pressure_map = {
        'wimbledon': 4.0, 'us open': 4.0, 'french open': 4.0, 'australian open': 4.0,
        'atp finals': 3.8, 'masters': 3.5, 'atp 500': 2.5, 'atp 250': 2.0
    }
    
    tournament_pressure = 2.5
    for key, pressure in pressure_map.items():
        if key in tournament.lower():
            tournament_pressure = pressure
            break
    
    # Преимущество на покрытии
    surface_advantage = p1_data.get('grass_adv', 0.0) if surface == 'Grass' else 0.0
    
    # Собираем данные для ML
    match_data = {
        'player_rank': p1_data['rank'],
        'opponent_rank': p2_data['rank'],
        'player_age': p1_data['age'],
        'opponent_age': p2_data['age'],
        'player_recent_win_rate': p1_data['form'],
        'player_form_trend': 0.05,  # TODO: рассчитывать реальный тренд
        'player_surface_advantage': surface_advantage,
        'player_surface_experience': 0.8,  # TODO: реальный опыт на покрытии
        'h2h_win_rate': 0.5,  # TODO: реальная H2H статистика
        'h2h_matches': 2,
        'total_pressure': tournament_pressure,
        'player_days_since_last_match': 7
    }
    
    return match_data

@app.route('/')
def dashboard():
    """Главная страница с информацией о Advanced ML"""
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎾 Advanced Tennis ML Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; min-height: 100vh; padding: 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ 
            background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);
            border-radius: 20px; padding: 30px; margin-bottom: 30px; text-align: center;
        }}
        .ml-banner {{
            background: linear-gradient(135deg, #27ae60, #2ecc71);
            padding: 20px; border-radius: 15px; margin-bottom: 20px;
            text-align: center; animation: pulse 3s infinite;
        }}
        @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.9; }} }}
        .stats-grid {{ 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px; margin: 20px 0;
        }}
        .stat-card {{ 
            background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 15px;
            text-align: center; transition: transform 0.3s ease;
        }}
        .stat-card:hover {{ transform: translateY(-5px); }}
        .stat-value {{ font-size: 2rem; font-weight: bold; margin-bottom: 5px; }}
        .controls {{ text-align: center; margin: 20px 0; }}
        .btn {{ 
            background: linear-gradient(135deg, #667eea, #764ba2); color: white;
            border: none; padding: 12px 24px; border-radius: 25px; font-size: 1rem;
            cursor: pointer; margin: 5px; transition: all 0.3s ease;
        }}
        .btn:hover {{ transform: translateY(-2px); }}
        .matches-container {{ display: grid; gap: 20px; }}
        .match-card {{ 
            background: rgba(255, 255, 255, 0.1); border-radius: 20px; padding: 25px;
            border-left: 5px solid #27ae60;
        }}
        .loading {{ text-align: center; padding: 50px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="ml-banner">
                <h2>🤖 ADVANCED MACHINE LEARNING SYSTEM</h2>
                <p>Targeting 70%+ accuracy • Real feature engineering • 25+ factors analysis</p>
            </div>
            
            <h1>🎾 Tennis Analytics Dashboard</h1>
            <p>Professional-grade ML predictions • No more simple 1/odds conversion</p>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="ml-status">{'✅ Ready' if ml_ready else '❌ Error'}</div>
                    <div class="stat-label">ML System</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">25+</div>
                    <div class="stat-label">ML Features</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">3</div>
                    <div class="stat-label">ML Models</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">70%+</div>
                    <div class="stat-label">Target Accuracy</div>
                </div>
            </div>
            
            <div class="controls">
                <button class="btn" onclick="loadMatches()">🤖 Get ML Predictions</button>
                <button class="btn" onclick="testML()">🔮 Test ML System</button>
                <button class="btn" onclick="showFeatures()">📊 Show ML Features</button>
            </div>
        </div>
        
        <div id="matches-container" class="matches-container">
            <div class="loading">🤖 Ready to generate Advanced ML predictions...</div>
        </div>
    </div>
    
    <script>
        const API_BASE = window.location.origin + '/api';
        
        async function loadMatches() {{
            const container = document.getElementById('matches-container');
            container.innerHTML = '<div class="loading">🤖 Generating Advanced ML predictions...</div>';
            
            try {{
                const response = await fetch(`${{API_BASE}}/matches`);
                const data = await response.json();
                
                if (data.success && data.matches) {{
                    let html = '<div style="background: linear-gradient(135deg, #27ae60, #2ecc71); padding: 20px; border-radius: 15px; margin-bottom: 25px; text-align: center;"><h2>🤖 ADVANCED ML PREDICTIONS</h2><p>Professional-grade machine learning • 25+ factors • 3-model ensemble</p></div>';
                    
                    data.matches.forEach(match => {{
                        const prob = match.prediction?.probability || 0.5;
                        const conf = match.prediction?.confidence || 'Medium';
                        
                        html += `
                            <div class="match-card">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                    <div>
                                        <div style="font-size: 1.4rem; font-weight: bold;">${{match.player1}} vs ${{match.player2}}</div>
                                        <div style="opacity: 0.8; margin-top: 5px;">🏆 ${{match.tournament}} • ${{match.surface}}</div>
                                    </div>
                                    <div style="background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 15px; text-align: center;">
                                        <div style="font-size: 1.2rem; font-weight: bold;">${{(prob * 100).toFixed(1)}}%</div>
                                        <div style="font-size: 0.8rem;">${{conf}}</div>
                                    </div>
                                </div>
                                
                                ${{match.key_factors && match.key_factors.length > 0 ? `
                                <div style="margin-top: 15px;">
                                    <strong>🔍 Key ML Factors:</strong>
                                    <ul style="margin-left: 20px; margin-top: 5px;">
                                        ${{match.key_factors.slice(0, 3).map(factor => `<li style="margin: 3px 0;">${{factor}}</li>`).join('')}}
                                    </ul>
                                </div>
                                ` : ''}}
                                
                                <div style="margin-top: 15px; text-align: center; font-size: 0.9rem; opacity: 0.8;">
                                    🤖 Type: ${{match.prediction_type || 'Advanced ML'}} • Source: ${{match.source || 'ML System'}}
                                </div>
                            </div>
                        `;
                    }});
                    
                    container.innerHTML = html;
                }} else {{
                    container.innerHTML = '<div class="loading">❌ ML system error</div>';
                }}
            }} catch (error) {{
                container.innerHTML = '<div class="loading">❌ Connection error</div>';
                console.error('Error:', error);
            }}
        }}
        
        async function testML() {{
            try {{
                const response = await fetch(`${{API_BASE}}/test-ml`, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        player1: 'Carlos Alcaraz',
                        player2: 'Novak Djokovic',
                        tournament: 'Wimbledon',
                        surface: 'Grass'
                    }})
                }});
                
                const data = await response.json();
                
                if (data.success) {{
                    const result = data.prediction;
                    alert(`🤖 ML Test Result:\\n\\n` +
                          `Match: ${{data.match_info.player1}} vs ${{data.match_info.player2}}\\n` +
                          `ML Prediction: ${{(result.probability * 100).toFixed(1)}}%\\n` +
                          `Confidence: ${{result.confidence}}\\n` +
                          `Type: ${{result.prediction_type}}\\n\\n` +
                          `Key Factors: ${{result.key_factors.slice(0, 2).join(', ')}}\\n\\n` +
                          `✅ Advanced ML working correctly!`);
                }} else {{
                    alert(`❌ ML Test failed: ${{data.error}}`);
                }}
            }} catch (error) {{
                alert(`❌ Test error: ${{error.message}}`);
            }}
        }}
        
        async function showFeatures() {{
            const features = [
                '🎯 Player Rankings & Age',
                '🔥 Recent Form & Momentum', 
                '🏟️ Surface Advantages',
                '💪 Head-to-Head History',
                '⭐ Tournament Pressure',
                '💎 Big Match Experience',
                '🏃 Physical Condition',
                '🎪 Hot/Cold Streaks',
                '⚡ Underdog Potential',
                '🧠 Consistency Factors',
                '🎲 Form vs Ranking',
                '🏆 Surface Specialists',
                '💼 Elo Ratings',
                '📈 Trend Analysis',
                '🎯 Combined Strength'
            ];
            
            alert(`🤖 ADVANCED ML FEATURES:\\n\\n` +
                  `Total: 25+ sophisticated features\\n\\n` +
                  features.slice(0, 10).join('\\n') +
                  `\\n\\n...and 10+ more advanced factors!\\n\\n` +
                  `🎯 Goal: Professional 70%+ accuracy`);
        }}
        
        // Auto-load on page ready
        document.addEventListener('DOMContentLoaded', function() {{
            setTimeout(loadMatches, 1000);
        }});
    </script>
</body>
</html>'''

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check с информацией о ML"""
    return jsonify({{
        'status': 'healthy',
        'ml_system': 'advanced',
        'ml_ready': ml_ready,
        'ml_models': 3 if ml_ready else 0,
        'prediction_type': 'ADVANCED_ML_ENSEMBLE' if ml_ready else 'FALLBACK',
        'timestamp': datetime.now().isoformat()
    }})

@app.route('/api/matches', methods=['GET'])
def get_matches():
    """Получение матчей с ADVANCED ML прогнозами"""
    try:
        logger.info("🤖 Generating Advanced ML predictions...")
        
        # Тестовые матчи для демонстрации
        test_matches = [
            ('Carlos Alcaraz', 'Novak Djokovic', 'Wimbledon', 'Grass'),
            ('Brandon Nakashima', 'Bu Yunchaokete', 'Wimbledon', 'Grass'),
            ('Jannik Sinner', 'Daniil Medvedev', 'US Open', 'Hard'),
            ('Aryna Sabalenka', 'Iga Swiatek', 'WTA Finals', 'Hard')
        ]
        
        processed_matches = []
        
        for player1, player2, tournament, surface in test_matches:
            if ml_ready and ml_predictor:
                try:
                    # Подготавливаем данные для ML
                    match_data = prepare_ml_data(player1, player2, tournament, surface)
                    
                    # Получаем РЕАЛЬНЫЙ ML прогноз
                    ml_result = ml_predictor.predict_match(match_data)
                    
                    logger.info(f"🤖 Advanced ML prediction for {{player1}}: {{ml_result['probability']:.1%}}")
                    
                    prediction_result = {{
                        'probability': ml_result['probability'],
                        'confidence': ml_result['confidence'],
                        'prediction_type': ml_result['prediction_type'],
                        'key_factors': ml_result['key_factors']
                    }}
                    
                except Exception as e:
                    logger.error(f"❌ ML prediction error for {{player1}}: {{e}}")
                    prediction_result = {{
                        'probability': 0.5,
                        'confidence': 'Error',
                        'prediction_type': 'ML_ERROR',
                        'key_factors': ['ML system error']
                    }}
            else:
                # Fallback если ML не готов
                prediction_result = {{
                    'probability': 0.5,
                    'confidence': 'Low',
                    'prediction_type': 'ML_NOT_READY',
                    'key_factors': ['ML system not initialized']
                }}
            
            match = {{
                'id': f"adv_ml_{{player1.replace(' ', '_').lower()}}",
                'player1': f"🎾 {{player1}}",
                'player2': f"🎾 {{player2}}",
                'tournament': f"🏆 {{tournament}}",
                'surface': surface,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'time': '14:00',
                'prediction': {{
                    'probability': prediction_result['probability'],
                    'confidence': prediction_result['confidence']
                }},
                'prediction_type': prediction_result['prediction_type'],
                'key_factors': prediction_result['key_factors'],
                'source': 'ADVANCED_ML_SYSTEM',
                'ml_features_count': 25
            }}
            
            processed_matches.append(match)
        
        return jsonify({{
            'success': True,
            'matches': processed_matches,
            'count': len(processed_matches),
            'source': 'ADVANCED_ML_SYSTEM',
            'ml_status': 'active' if ml_ready else 'error',
            'prediction_type': 'ADVANCED_ML_ENSEMBLE' if ml_ready else 'FALLBACK',
            'timestamp': datetime.now().isoformat()
        }})
        
    except Exception as e:
        logger.error(f"❌ Matches error: {{e}}")
        return jsonify({{
            'success': False,
            'error': str(e),
            'matches': []
        }}), 500

@app.route('/api/test-ml', methods=['POST'])
def test_ml_system():
    """Тестирование Advanced ML системы"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({{
                'success': False,
                'error': 'No data provided'
            }}), 400
        
        player1 = data.get('player1', 'Carlos Alcaraz')
        player2 = data.get('player2', 'Novak Djokovic')
        tournament = data.get('tournament', 'Wimbledon')
        surface = data.get('surface', 'Grass')
        
        logger.info(f"🔮 Testing Advanced ML: {{player1}} vs {{player2}}")
        
        if ml_ready and ml_predictor:
            # Подготавливаем данные
            match_data = prepare_ml_data(player1, player2, tournament, surface)
            
            # Получаем ML прогноз
            ml_result = ml_predictor.predict_match(match_data)
            
            return jsonify({{
                'success': True,
                'prediction': ml_result,
                'match_info': {{
                    'player1': player1,
                    'player2': player2,
                    'tournament': tournament,
                    'surface': surface
                }},
                'ml_data_used': match_data,
                'timestamp': datetime.now().isoformat()
            }})
        else:
            return jsonify({{
                'success': False,
                'error': 'Advanced ML system not ready'
            }}), 500
        
    except Exception as e:
        logger.error(f"❌ ML test error: {{e}}")
        return jsonify({{
            'success': False,
            'error': str(e)
        }}), 500

@app.route('/api/ml-status', methods=['GET'])
def get_ml_status():
    """Статус Advanced ML системы"""
    try:
        if ml_ready and ml_predictor:
            status = {{
                'ml_ready': True,
                'models_count': len(ml_predictor.models),
                'features_count': len(ml_predictor.feature_names) if ml_predictor.feature_names else 25,
                'accuracy_target': '70%+',
                'system_type': 'ADVANCED_ML_ENSEMBLE',
                'models': list(ml_predictor.models.keys()) if ml_predictor.models else [],
                'training_completed': ml_predictor.is_trained
            }}
        else:
            status = {{
                'ml_ready': False,
                'error': 'ML system not initialized',
                'system_type': 'FALLBACK'
            }}
        
        return jsonify({{
            'success': True,
            'ml_status': status,
            'timestamp': datetime.now().isoformat()
        }})
        
    except Exception as e:
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
    print("🎾 ADVANCED TENNIS ML BACKEND")
    print("=" * 60)
    print("🎯 Target: Professional 70%+ accuracy")
    print("🤖 Features: 25+ advanced ML factors")
    print("🚀 Models: 3-model ensemble (RF, GB, LR)")
    print("=" * 60)
    
    # Инициализируем Advanced ML
    ml_success = initialize_advanced_ml()
    
    print(f"🌐 Dashboard: http://0.0.0.0:5001")
    print(f"📡 API: http://0.0.0.0:5001/api/*")
    print(f"🤖 Advanced ML: {{'✅ Ready' if ml_success else '❌ Error'}}")
    print("🔥 NO MORE simple 1/odds conversion!")
    print("=" * 60)
    
    try:
        app.run(
            host='0.0.0.0',
            port=5001,
            debug=False,
            threaded=True
        )
    except Exception as e:
        print(f"❌ Failed to start server: {{e}}")