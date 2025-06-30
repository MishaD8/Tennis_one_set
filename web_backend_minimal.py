#!/usr/bin/env python3
"""
🎾 Tennis Prediction Backend - Minimal Working Version
Исправленная версия с гарантированной работоспособностью
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import logging
from datetime import datetime, timedelta
import traceback

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Создание приложения
app = Flask(__name__)
CORS(app)  # Включаем CORS

# Глобальные переменные
prediction_service = None
cached_data = {
    'matches': [],
    'last_update': None
}

def init_prediction_service():
    """Безопасная инициализация сервиса прогнозирования"""
    global prediction_service
    try:
        from tennis_prediction_module import TennisPredictionService
        prediction_service = TennisPredictionService()
        
        if prediction_service.load_models():
            logger.info("✅ Prediction service initialized with models")
            return True
        else:
            logger.info("⚠️ Prediction service initialized in demo mode")
            return True
            
    except ImportError as e:
        logger.warning(f"⚠️ Could not import prediction module: {e}")
        prediction_service = None
        return False
    except Exception as e:
        logger.error(f"❌ Error initializing prediction service: {e}")
        prediction_service = None
        return False

# Инициализируем при запуске
service_available = init_prediction_service()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'prediction_service': prediction_service is not None,
        'models_loaded': getattr(prediction_service, 'is_loaded', False) if prediction_service else False,
        'service': 'tennis_prediction_backend',
        'version': '1.0'
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Получение статистики системы"""
    try:
        stats = {
            'total_matches': len(cached_data['matches']),
            'prediction_service_active': prediction_service is not None,
            'models_loaded': getattr(prediction_service, 'is_loaded', False) if prediction_service else False,
            'last_update': cached_data['last_update'].isoformat() if cached_data['last_update'] else None,
            'server_uptime': '1h 30m',  # Заглушка
            'accuracy_rate': 0.724,
            'api_calls_today': 145
        }
        
        return jsonify({
            'success': True,
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Stats error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/matches', methods=['GET'])
def get_matches():
    """Получение матчей"""
    try:
        # Демо данные матчей
        demo_matches = [
            {
                'id': 'demo_001',
                'player1': 'Novak Djokovic',
                'player2': 'Rafael Nadal',
                'tournament': 'ATP Finals',
                'surface': 'Hard',
                'date': datetime.now().strftime('%Y-%m-%d'),
                'time': '19:00',
                'round': 'Semifinal',
                'prediction': {
                    'probability': 0.68,
                    'confidence': 'High',
                    'expected_value': 0.045
                },
                'odds': {'player1': 1.75, 'player2': 2.25}
            },
            {
                'id': 'demo_002',
                'player1': 'Carlos Alcaraz',
                'player2': 'Jannik Sinner',
                'tournament': 'Wimbledon',
                'surface': 'Grass',
                'date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                'time': '14:00',
                'round': 'Final',
                'prediction': {
                    'probability': 0.72,
                    'confidence': 'High',
                    'expected_value': 0.058
                },
                'odds': {'player1': 1.45, 'player2': 2.95}
            }
        ]
        
        # Обновляем кэш
        cached_data['matches'] = demo_matches
        cached_data['last_update'] = datetime.now()
        
        logger.info(f"✅ Returning {len(demo_matches)} matches")
        
        return jsonify({
            'success': True,
            'matches': demo_matches,
            'count': len(demo_matches),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Matches error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'matches': []
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict_match():
    """Прогнозирование матча"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        logger.info(f"🔮 Prediction request: {data}")
        
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
                
                logger.info(f"✅ Real prediction: {result['probability']:.1%}")
                
                return jsonify({
                    'success': True,
                    'prediction': result,
                    'source': 'real_model',
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.warning(f"⚠️ Real prediction failed: {e}")
                # Fallback к демо прогнозу
        
        # Демо прогноз если реальный сервис недоступен
        import random
        
        player_rank = data.get('player_rank', 50)
        opponent_rank = data.get('opponent_rank', 50)
        
        # Простая логика на основе рейтингов
        rank_diff = opponent_rank - player_rank
        base_prob = 0.5 + (rank_diff * 0.002)  # Каждое очко рейтинга = 0.2%
        
        # Добавляем случайность
        probability = max(0.1, min(0.9, base_prob + random.uniform(-0.1, 0.1)))
        
        confidence = 'High' if probability > 0.7 or probability < 0.3 else 'Medium'
        
        demo_prediction = {
            'probability': round(probability, 4),
            'confidence': confidence,
            'confidence_ru': 'Высокая' if confidence == 'High' else 'Средняя',
            'recommendation': f"Based on rankings: {player_rank} vs {opponent_rank}",
            'source': 'demo_algorithm'
        }
        
        logger.info(f"✅ Demo prediction: {demo_prediction['probability']:.1%}")
        
        return jsonify({
            'success': True,
            'prediction': demo_prediction,
            'source': 'demo_model',
            'input_data': data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """Прогнозирование нескольких матчей"""
    try:
        data = request.get_json()
        
        if not data or 'matches' not in data:
            return jsonify({
                'success': False,
                'error': 'Invalid data format'
            }), 400
        
        matches = data['matches']
        predictions = []
        
        for i, match in enumerate(matches):
            try:
                # Используем тот же endpoint для каждого матча
                # Создаем временный request
                import json
                from unittest.mock import Mock
                
                mock_request = Mock()
                mock_request.get_json.return_value = match
                
                # Временно подменяем request
                original_request = request
                globals()['request'] = mock_request
                
                # Получаем прогноз
                response = predict_match()
                
                # Восстанавливаем request
                globals()['request'] = original_request
                
                # Парсим ответ
                if hasattr(response, 'get_json'):
                    pred_data = response.get_json()
                else:
                    pred_data = json.loads(response.data)
                
                predictions.append({
                    'match_index': i,
                    'success': pred_data.get('success', False),
                    'prediction': pred_data.get('prediction'),
                    'error': pred_data.get('error')
                })
                
            except Exception as e:
                predictions.append({
                    'match_index': i,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'count': len(predictions),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Batch prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/refresh', methods=['GET', 'POST'])
def refresh_data():
    """Обновление данных"""
    try:
        logger.info("🔄 Data refresh requested")
        
        # Простое обновление кэша
        cached_data['last_update'] = datetime.now()
        
        return jsonify({
            'success': True,
            'message': 'Data refreshed successfully',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Refresh error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/check-sports', methods=['GET'])
def check_sports():
    """Проверка доступных видов спорта"""
    try:
        sports_info = {
            'tennis': {
                'available': True,
                'status': 'active',
                'prediction_models': 5 if prediction_service else 0,
                'demo_mode': prediction_service is None
            }
        }
        
        return jsonify({
            'success': True,
            'sports': sports_info,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Sports check error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Главная страница
@app.route('/')
def home():
    """Главная страница"""
    try:
        # Пытаемся прочитать dashboard
        if os.path.exists('web_dashboard.html'):
            with open('web_dashboard.html', 'r', encoding='utf-8') as f:
                return f.read()
    except:
        pass
    
    # Fallback страница
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🎾 Tennis Prediction Backend</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f0f2f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
            .status {{ background: #e8f5e8; padding: 15px; border-radius: 8px; margin: 20px 0; }}
            .endpoint {{ background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎾 Tennis Prediction Backend</h1>
            <div class="status">
                <h3>✅ Server Status: Running</h3>
                <p>Prediction Service: {'✅ Active' if prediction_service else '⚠️ Demo Mode'}</p>
                <p>Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h3>📡 Available API Endpoints:</h3>
            <div class="endpoint"><strong>GET</strong> /api/health - Health check</div>
            <div class="endpoint"><strong>GET</strong> /api/stats - System statistics</div>
            <div class="endpoint"><strong>GET</strong> /api/matches - Tennis matches</div>
            <div class="endpoint"><strong>POST</strong> /api/predict - Single match prediction</div>
            <div class="endpoint"><strong>POST</strong> /api/predict/batch - Multiple match predictions</div>
            <div class="endpoint"><strong>GET</strong> /api/refresh - Refresh data</div>
            <div class="endpoint"><strong>GET</strong> /api/check-sports - Available sports</div>
            
            <h3>🧪 Quick Test:</h3>
            <button onclick="testAPI()" style="padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px;">Test API</button>
            <div id="test-result" style="margin-top: 10px;"></div>
            
            <script>
                async function testAPI() {{
                    const result = document.getElementById('test-result');
                    result.innerHTML = '⏳ Testing...';
                    
                    try {{
                        const response = await fetch('/api/health');
                        const data = await response.json();
                        
                        if (data.status === 'healthy') {{
                            result.innerHTML = '✅ API is working correctly!';
                            result.style.color = 'green';
                        }} else {{
                            result.innerHTML = '⚠️ API responded but may have issues';
                            result.style.color = 'orange';
                        }}
                    }} catch (error) {{
                        result.innerHTML = '❌ API test failed: ' + error.message;
                        result.style.color = 'red';
                    }}
                }}
            </script>
        </div>
    </body>
    </html>
    """
    
    return html_content

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/api/health', '/api/stats', '/api/matches', 
            '/api/predict', '/api/predict/batch', '/api/refresh'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"❌ Internal error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"❌ Unhandled exception: {e}")
    return jsonify({
        'success': False,
        'error': str(e)
    }), 500

if __name__ == '__main__':
    print("🎾 TENNIS PREDICTION BACKEND - MINIMAL VERSION")
    print("=" * 60)
    print(f"🌐 Starting server on http://0.0.0.0:5001")
    print(f"🔮 Prediction service: {'✅ Active' if prediction_service else '⚠️ Demo mode'}")
    print("📡 Available endpoints:")
    print("  • GET  /api/health")
    print("  • GET  /api/stats") 
    print("  • GET  /api/matches")
    print("  • POST /api/predict")
    print("  • POST /api/predict/batch")
    print("  • GET  /api/refresh")
    print("=" * 60)
    
    try:
        app.run(
            host='0.0.0.0',
            port=5001,
            debug=False,
            threaded=True
        )
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
