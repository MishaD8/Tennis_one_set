#!/usr/bin/env python3
"""
🎾 TENNIS BACKEND С ПОДРОБНОЙ ДИАГНОСТИКОЙ
Покажет где именно происходит ошибка
"""

import sys
import traceback
from datetime import datetime

print("🎾 ЗАПУСК TENNIS BACKEND С ДИАГНОСТИКОЙ")
print("=" * 50)
print(f"🕐 Время запуска: {datetime.now()}")
print(f"🐍 Python: {sys.version}")
print("=" * 50)

try:
    print("📦 Импорт Flask...")
    from flask import Flask, jsonify, request
    print("✅ Flask импортирован")
    
    print("📦 Импорт CORS...")
    try:
        from flask_cors import CORS
        CORS_AVAILABLE = True
        print("✅ Flask-CORS импортирован")
    except ImportError:
        CORS_AVAILABLE = False
        print("⚠️ Flask-CORS не найден (не критично)")
    
    print("📦 Импорт стандартных модулей...")
    import logging
    import random
    import json
    print("✅ Стандартные модули импортированы")
    
    print("📦 Импорт API Economy...")
    try:
        from api_economy_patch import (
            init_api_economy, 
            economical_tennis_request, 
            get_api_usage, 
            trigger_manual_update,
            clear_api_cache
        )
        print("✅ API Economy импортирован")
        API_ECONOMY_AVAILABLE = True
    except ImportError as e:
        print(f"⚠️ API Economy не найден: {e}")
        API_ECONOMY_AVAILABLE = False
    
    print("\n🔧 Создание Flask приложения...")
    app = Flask(__name__)
    
    if CORS_AVAILABLE:
        CORS(app)
        print("✅ CORS настроен")
    
    print("🔧 Настройка логирования...")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    print("✅ Логирование настроено")
    
    if API_ECONOMY_AVAILABLE:
        print("💰 Инициализация API Economy...")
        try:
            init_api_economy(
                api_key="a1b20d709d4bacb2d95ddab880f91009",
                max_per_hour=30,
                cache_minutes=20
            )
            print("✅ API Economy инициализирован")
        except Exception as e:
            print(f"⚠️ Ошибка инициализации API Economy: {e}")
    
    # Простые демо данные
    DEMO_MATCHES = [
        {
            'id': 'match_1',
            'player1': '🎾 Marin Cilic',
            'player2': '🎾 Flavio Cobolli',
            'tournament': '🏆 ATP Tournament',
            'surface': 'Hard',
            'round': 'R32',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': '14:00',
            'odds': {'player1': 1.99, 'player2': 2.00},
            'underdog_analysis': {
                'underdog': 'Flavio Cobolli',
                'favorite': 'Marin Cilic',
                'underdog_odds': 2.00,
                'prediction': {
                    'probability': 0.78,
                    'confidence': 'High',
                    'key_factors': [
                        '🎯 Реалистичные коэффициенты',
                        '⚖️ Равные силы игроков',
                        '💪 Хорошие шансы взять сет'
                    ]
                },
                'quality_rating': 'HIGH'
            },
            'focus': '💎 Flavio Cobolli взять хотя бы 1 сет',
            'recommendation': '78% шанс взять сет',
            'data_source': 'DEMO_SYSTEM'
        }
    ]
    
    print("🎨 Создание HTML dashboard...")
    
    @app.route('/')
    def dashboard():
        """Упрощенный dashboard"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎾 Tennis Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; min-height: 100vh; padding: 20px;
        }
        .container { max-width: 1000px; margin: 0 auto; }
        .header { 
            background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);
            border-radius: 20px; padding: 30px; margin-bottom: 30px; text-align: center;
        }
        .title { font-size: 2.5rem; margin-bottom: 10px; }
        .btn { 
            background: rgba(255,255,255,0.2); border: 2px solid rgba(255,255,255,0.3);
            color: white; padding: 15px 30px; border-radius: 25px; font-size: 1.1rem;
            cursor: pointer; margin: 10px; transition: all 0.3s ease;
        }
        .btn:hover { background: rgba(255,255,255,0.3); transform: translateY(-2px); }
        .status { margin: 20px 0; padding: 15px; background: rgba(39, 174, 96, 0.2); border-radius: 10px; }
        .matches { display: grid; gap: 20px; margin-top: 20px; }
        .match { 
            background: rgba(255,255,255,0.1); border-radius: 15px; padding: 20px;
            border-left: 4px solid #27ae60;
        }
        .loading { text-align: center; padding: 40px; font-size: 1.2rem; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">🎾 Tennis Dashboard</div>
            <div class="status">
                ✅ <strong>Система запущена успешно!</strong><br>
                Flask работает, dashboard загружен, API готов к работе
            </div>
            <button class="btn" onclick="loadMatches()">💎 Загрузить матчи</button>
            <button class="btn" onclick="testAPI()">🧪 Тест API</button>
            <button class="btn" onclick="showStatus()">📊 Статус системы</button>
        </div>
        
        <div id="content" class="loading">
            💎 Dashboard успешно загружен!<br>
            <small>Нажмите любую кнопку для тестирования</small>
        </div>
    </div>

    <script>
        // Проверяем что JavaScript работает
        console.log('✅ JavaScript загружен');
        
        async function loadMatches() {
            document.getElementById('content').innerHTML = '<div class="loading">🔄 Загрузка матчей...</div>';
            
            try {
                const response = await fetch('/api/matches');
                const data = await response.json();
                
                if (data.success) {
                    displayMatches(data.matches);
                } else {
                    document.getElementById('content').innerHTML = '<div class="loading">❌ Ошибка: ' + data.error + '</div>';
                }
            } catch (error) {
                document.getElementById('content').innerHTML = '<div class="loading">❌ Ошибка сети: ' + error + '</div>';
            }
        }
        
        function displayMatches(matches) {
            let html = '<div class="matches">';
            html += '<h2 style="text-align: center; margin-bottom: 20px;">💎 Найдены матчи!</h2>';
            
            matches.forEach(match => {
                const analysis = match.underdog_analysis;
                const prediction = analysis.prediction;
                
                html += `
                    <div class="match">
                        <div style="font-size: 1.3rem; font-weight: bold; margin-bottom: 10px;">
                            ${match.player1} vs ${match.player2}
                        </div>
                        <div style="margin: 10px 0;">
                            🏆 ${match.tournament} • ${match.surface} • ${match.round}
                        </div>
                        <div style="background: rgba(255, 217, 61, 0.2); padding: 15px; border-radius: 10px; margin: 15px 0;">
                            <strong>${match.focus}</strong><br>
                            ${match.recommendation}
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
                            <div style="text-align: center; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 8px;">
                                <div>${match.player1.replace('🎾 ', '')}</div>
                                <div style="font-size: 1.2rem; font-weight: bold;">${match.odds.player1}</div>
                            </div>
                            <div style="text-align: center; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 8px;">
                                <div>${match.player2.replace('🎾 ', '')}</div>
                                <div style="font-size: 1.2rem; font-weight: bold;">${match.odds.player2}</div>
                            </div>
                        </div>
                    </div>
                `;
            });
            
            html += '</div>';
            document.getElementById('content').innerHTML = html;
        }
        
        async function testAPI() {
            try {
                const response = await fetch('/api/health');
                const data = await response.json();
                
                alert(`🧪 API ТЕСТ:\\n\\n` +
                      `Статус: ${data.status}\\n` +
                      `Система: ${data.system}\\n` +
                      `Время: ${data.timestamp}\\n\\n` +
                      `✅ API работает отлично!`);
            } catch (error) {
                alert('❌ Ошибка API: ' + error);
            }
        }
        
        async function showStatus() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                
                if (data.success) {
                    alert(`📊 СТАТУС СИСТЕМЫ:\\n\\n` +
                          `Матчей: ${data.count}\\n` +
                          `API Economy: ${data.api_economy ? 'Активен' : 'Отключен'}\\n` +
                          `Источник: ${data.source}\\n` +
                          `Время: ${data.timestamp}\\n\\n` +
                          `✅ Все системы работают!`);
                } else {
                    alert('❌ Ошибка получения статуса');
                }
            } catch (error) {
                alert('❌ Ошибка: ' + error);
            }
        }
        
        // Автотест при загрузке
        document.addEventListener('DOMContentLoaded', function() {
            console.log('✅ DOM загружен');
            setTimeout(() => {
                console.log('✅ Автотест через 2 секунды...');
            }, 2000);
        });
    </script>
</body>
</html>'''
    
    print("📡 Создание API endpoints...")
    
    @app.route('/api/matches')
    def get_matches():
        try:
            logger.info("💎 API: Запрос матчей")
            return jsonify({
                'success': True,
                'matches': DEMO_MATCHES,
                'count': len(DEMO_MATCHES),
                'source': 'DEMO_WITH_DEBUG',
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"❌ Ошибка API matches: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/health')
    def health_check():
        return jsonify({
            'status': 'healthy',
            'system': 'debug_tennis_backend',
            'api_economy': API_ECONOMY_AVAILABLE,
            'timestamp': datetime.now().isoformat()
        })
    
    @app.route('/api/stats')
    def get_stats():
        try:
            stats_data = {
                'success': True,
                'count': len(DEMO_MATCHES),
                'api_economy': API_ECONOMY_AVAILABLE,
                'source': 'DEBUG_SYSTEM',
                'timestamp': datetime.now().isoformat()
            }
            
            if API_ECONOMY_AVAILABLE:
                try:
                    usage = get_api_usage()
                    stats_data['api_usage'] = usage
                except:
                    stats_data['api_usage'] = 'error'
            
            return jsonify(stats_data)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    print("\n🚀 ВСЕ КОМПОНЕНТЫ ЗАГРУЖЕНЫ УСПЕШНО!")
    print("=" * 50)
    print("🌐 Запуск веб-сервера...")
    
    if __name__ == '__main__':
        try:
            print("🔄 Попытка запуска на порту 5001...")
            app.run(
                host='0.0.0.0',
                port=5001,
                debug=True,
                use_reloader=False
            )
        except OSError as e:
            if "Address already in use" in str(e):
                print("⚠️ Порт 5001 занят, пробуем 8080...")
                app.run(
                    host='0.0.0.0', 
                    port=8080,
                    debug=True,
                    use_reloader=False
                )
            else:
                raise e

except ImportError as e:
    print(f"❌ ОШИБКА ИМПОРТА: {e}")
    print("💡 Решение: pip install flask flask-cors")
    traceback.print_exc()

except Exception as e:
    print(f"❌ НЕОЖИДАННАЯ ОШИБКА: {e}")
    print("\n🔍 ПОЛНАЯ ТРАССИРОВКА ОШИБКИ:")
    traceback.print_exc()
    print("\n💡 ОБРАТИТЕСЬ ЗА ПОМОЩЬЮ С ЭТОЙ ИНФОРМАЦИЕЙ")