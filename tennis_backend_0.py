@app.route('/api/quality-matches')
def get_quality_matches():
    """ОБНОВЛЕНО: Использует API Economy систему"""
    try:
        logger.info("💰 Using API Economy for quality matches...")
        
        quality_matches = generate_quality_matches()
        
        # Получаем контекст от универсальной системы
        season_context = "Unknown Season"
        data_source = "API_ECONOMY"
        
        if universal_data_ready and universal_data_collector:
            season_context = universal_data_collector.get_season_context()
            data_source = "API_ECONOMY_UNIVERSAL"
        
        if not quality_matches:
            return jsonify({
                'success': False,
                'message': 'No quality matches found via API Economy',
                'matches': [],
                'season_context': season_context,
                'source': data_source + '_NO_MATCHES'
            })
        
        # Статистика
        probabilities = [m['underdog_analysis']['prediction']['probability'] for m in quality_matches]
        strong_underdogs = len([p for p in probabilities if p > 0.70])
        
        return jsonify({
            'success': True,
            'matches': quality_matches,
            'count': len(quality_matches),
            'source': data_source,
            'season_context': season_context,
            'stats': {
                'total_matches': len(quality_matches),
                'strong_underdogs': strong_underdogs,
                'avg_probability': f"{(sum(probabilities) / len(probabilities) * 100):.0f}%"
            },
            'system_info': {
                'api_economy_active': True,
                'data_source': data_source,
                'prediction_type': 'API_ECONOMY_ML_UNDERDOG'
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Quality matches error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'matches': []
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check с информацией об API Economy"""
    try:
        # Получаем статистику API Economy
        usage_stats = get_api_usage()
        
        return jsonify({
            'status': 'healthy',
            'system': 'tennis_backend_with_api_economy',
            'api_economy_active': True,
            'api_requests_remaining': usage_stats.get('remaining_hour', 0),
            'cache_items': usage_stats.get('cache_items', 0),
            'universal_data_system': universal_data_ready,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'limited',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Статистика системы с API Economy"""
    try:
        # Получаем статистику API Economy
        usage_stats = get_api_usage()
        
        return jsonify({
            'success': True,
            'stats': {
                'system_type': 'Tennis Backend with API Economy',
                'api_economy_active': True,
                'api_requests_hour': f"{usage_stats.get('requests_this_hour', 0)}/{usage_stats.get('max_per_hour', 30)}",
                'api_requests_remaining': usage_stats.get('remaining_hour', 0),
                'cache_items': usage_stats.get('cache_items', 0),
                'cache_duration_minutes': usage_stats.get('cache_minutes', 20),
                'manual_update_status': usage_stats.get('manual_update_status', 'не требуется'),
                'universal_data_active': universal_data_ready,
                'last_update': datetime.now().isoformat()
            },
            'api_usage': usage_stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("🎾 TENNIS BACKEND С ИНТЕГРИРОВАННОЙ API ECONOMY")
    print("=" * 70)
    print("💰 API ECONOMY FEATURES:")
    print("• ✅ Автоматическое кеширование API запросов")
    print("• ✅ Контроль лимитов запросов в час")
    print("• ✅ Ручное обновление данных")
    print("• ✅ Fallback на кеш при превышении лимитов")
    print("• ✅ Статистика использования API")
    print("=" * 70)
    print(f"🌐 Dashboard: http://localhost:5001")
    print(f"📡 API: http://localhost:5001/api/*")
    print(f"💰 API Economy: ✅ Active")
    print(f"🌍 Universal Data: {'✅ Active' if universal_data_ready else '⚠️ Fallback mode'}")
    print("=" * 70)
    print("🔧 УПРАВЛЕНИЕ API:")
    print("• Ручное обновление: POST /api/trigger-manual-update")
    print("• Статистика API: GET /api/api-stats")
    print("• Очистка кеша: POST /api/clear-cache")
    print("=" * 70)
    
    # Показываем текущий статус API
    try:
        usage = get_api_usage()
        print(f"📊 ТЕКУЩИЙ СТАТУС API:")
        print(f"   Запросов в час: {usage.get('requests_this_hour', 0)}/{usage.get('max_per_hour', 30)}")
        print(f"   Остается запросов: {usage.get('remaining_hour', 0)}")
        print(f"   Элементов в кеше: {usage.get('cache_items', 0)}")
        print(f"   Ручное обновление: {usage.get('manual_update_status', 'не требуется')}")
    except Exception as e:
        print(f"⚠️ Не удалось получить статистику: {e}")
    
    print("=" * 70)
    
    try:
        app.run(
            host='0.0.0.0',
            port=5001,
            debug=False,
            threaded=True
        )
    except Exception as e:
        print(f"❌ Server error: {e}")#!/usr/bin/env python3
"""
🎾 ИСПРАВЛЕННЫЙ Tennis Backend с полной интеграцией API Economy
"""

import os
from dotenv import load_dotenv
from typing import Optional
from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import random
import math
from datetime import datetime, timedelta
import numpy as np

# ИНТЕГРАЦИЯ API ECONOMY
from api_economy_patch import (
    init_api_economy, 
    economical_tennis_request, 
    get_api_usage, 
    trigger_manual_update,
    check_manual_update_status,
    clear_api_cache
)

# ИНИЦИАЛИЗАЦИЯ API ECONOMY ПРИ ЗАПУСКЕ
init_api_economy(
    api_key="a1b20d709d4bacb2d95ddab880f91009",
    max_per_hour=30,
    cache_minutes=20
)

# НОВОЕ: Импорт универсальной системы данных
try:
    from universal_data_fix import UniversalTennisDataFix
    UNIVERSAL_DATA_AVAILABLE = True
    print("✅ Universal tennis data system imported")
except ImportError as e:
    print(f"⚠️ Universal data system not available: {e}")
    UNIVERSAL_DATA_AVAILABLE = False

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Глобальные переменные
universal_data_collector = None

def initialize_universal_data():
    """Инициализация универсальной системы данных"""
    global universal_data_collector
    
    if UNIVERSAL_DATA_AVAILABLE:
        try:
            universal_data_collector = UniversalTennisDataFix()
            logger.info("🌍 Universal data collector initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Universal data initialization failed: {e}")
            return False
    else:
        logger.warning("⚠️ Universal data collector not available")
        return False

# Инициализируем при запуске
universal_data_ready = initialize_universal_data()

class SmartUnderdogPredictor:
    """Умный предиктор для андердогов взять хотя бы один сет"""
    
    def __init__(self):
        # Реальные данные игроков на июль 2025
        self.player_database = {
            # ATP топ-игроки
            'jannik sinner': {'rank': 1, 'age': 23, 'form': 0.90, 'grass_skill': 0.78, 'set_tenacity': 0.85, 'big_match': 0.85},
            'carlos alcaraz': {'rank': 2, 'age': 21, 'form': 0.88, 'grass_skill': 0.75, 'set_tenacity': 0.80, 'big_match': 0.82},
            'alexander zverev': {'rank': 3, 'age': 27, 'form': 0.82, 'grass_skill': 0.68, 'set_tenacity': 0.75, 'big_match': 0.78},
            'daniil medvedev': {'rank': 4, 'age': 28, 'form': 0.78, 'grass_skill': 0.65, 'set_tenacity': 0.85, 'big_match': 0.82},
            'novak djokovic': {'rank': 6, 'age': 37, 'form': 0.75, 'grass_skill': 0.95, 'set_tenacity': 0.95, 'big_match': 0.95},
            
            # Средние игроки (потенциальные андердоги)
            'ben shelton': {'rank': 15, 'age': 22, 'form': 0.72, 'grass_skill': 0.70, 'set_tenacity': 0.75, 'big_match': 0.60},
            'tommy paul': {'rank': 12, 'age': 27, 'form': 0.75, 'grass_skill': 0.72, 'set_tenacity': 0.78, 'big_match': 0.70},
            'frances tiafoe': {'rank': 18, 'age': 26, 'form': 0.70, 'grass_skill': 0.68, 'set_tenacity': 0.80, 'big_match': 0.65},
            'brandon nakashima': {'rank': 45, 'age': 23, 'form': 0.68, 'grass_skill': 0.62, 'set_tenacity': 0.72, 'big_match': 0.50},
            'fabio fognini': {'rank': 85, 'age': 37, 'form': 0.62, 'grass_skill': 0.58, 'set_tenacity': 0.65, 'big_match': 0.75},
            
            # WTA
            'aryna sabalenka': {'rank': 1, 'age': 26, 'form': 0.85, 'grass_skill': 0.72, 'set_tenacity': 0.82, 'big_match': 0.80},
            'iga swiatek': {'rank': 2, 'age': 23, 'form': 0.88, 'grass_skill': 0.65, 'set_tenacity': 0.85, 'big_match': 0.85},
            'emma raducanu': {'rank': 90, 'age': 22, 'form': 0.62, 'grass_skill': 0.68, 'set_tenacity': 0.70, 'big_match': 0.50},
            'katie boulter': {'rank': 28, 'age': 27, 'form': 0.68, 'grass_skill': 0.75, 'set_tenacity': 0.72, 'big_match': 0.60},
            'amanda anisimova': {'rank': 35, 'age': 23, 'form': 0.72, 'grass_skill': 0.58, 'set_tenacity': 0.75, 'big_match': 0.55},
        }
    
    def get_player_data(self, player_name):
        """Получение данных игрока"""
        name_lower = player_name.lower().strip()
        
        # Прямое совпадение
        if name_lower in self.player_database:
            return self.player_database[name_lower]
        
        # Поиск по частям имени
        for known_player, data in self.player_database.items():
            if any(part in known_player for part in name_lower.split()):
                return data
        
        # Генерируем данные для неизвестного игрока
        rank = random.randint(40, 150)
        return {
            'rank': rank,
            'age': random.randint(20, 32),
            'form': max(0.4, 0.8 - rank/200),
            'grass_skill': random.uniform(0.5, 0.7),
            'set_tenacity': random.uniform(0.6, 0.8),
            'big_match': max(0.3, 0.8 - rank/150)
        }
    
    def determine_underdog_from_odds(self, player1, player2, odds1, odds2):
        """Правильное определение андердога по коэффициентам"""
        # Андердог = игрок с БОЛЬШИМИ коэффициентами (менее вероятная победа)
        if odds1 > odds2:
            return {
                'underdog': player1,
                'favorite': player2,
                'underdog_odds': odds1,
                'favorite_odds': odds2,
                'is_player1_underdog': True
            }
        else:
            return {
                'underdog': player2,
                'favorite': player1, 
                'underdog_odds': odds2,
                'favorite_odds': odds1,
                'is_player1_underdog': False
            }
    
    def calculate_smart_set_probability(self, underdog_name, favorite_name, underdog_odds, favorite_odds):
        """Умный расчёт вероятности андердога взять хотя бы один сет"""
        
        underdog_data = self.get_player_data(underdog_name)
        favorite_data = self.get_player_data(favorite_name)
        
        # 1. Базовая вероятность из коэффициентов
        match_prob = 1.0 / underdog_odds
        base_set_prob = min(0.85, match_prob + 0.25)
        
        # 2. Факторы
        tenacity_factor = underdog_data['set_tenacity'] * 0.3
        grass_factor = (underdog_data['grass_skill'] - 0.6) * 0.2
        big_match_factor = underdog_data['big_match'] * 0.15
        form_factor = (underdog_data['form'] - 0.65) * 0.2
        
        # 3. Возраст
        age_factor = 0
        if underdog_data['age'] < 25:
            age_factor = 0.05
        elif underdog_data['age'] > 32:
            age_factor = -0.03
        
        # 4. Итоговая вероятность
        final_probability = (base_set_prob + tenacity_factor + grass_factor + 
                           big_match_factor + form_factor + age_factor)
        
        final_probability = max(0.25, min(0.92, final_probability))
        
        confidence = 'Very High' if final_probability > 0.8 else \
                    'High' if final_probability > 0.7 else 'Medium'
        
        factors = self._analyze_key_factors(underdog_data, favorite_data, underdog_odds, final_probability)
        
        return {
            'probability': round(final_probability, 3),
            'confidence': confidence,
            'key_factors': factors
        }
    
    def _analyze_key_factors(self, underdog_data, favorite_data, underdog_odds, probability):
        """Анализ ключевых факторов для андердога"""
        factors = []
        
        if underdog_data['set_tenacity'] > 0.75:
            factors.append(f"🔥 Высокое упорство в сетах ({underdog_data['set_tenacity']:.0%})")
        
        if underdog_data['grass_skill'] > 0.70:
            factors.append(f"🌱 Хорошо играет на траве")
        
        if underdog_data['form'] > 0.70:
            factors.append(f"📈 Хорошая текущая форма")
        elif underdog_data['form'] < 0.60:
            factors.append(f"📉 Проблемы с формой - но может сыграть без давления")
        
        if underdog_data['age'] < 24:
            factors.append(f"⚡ Молодой игрок - может играть без страха")
        
        if underdog_data['big_match'] > 0.70:
            factors.append(f"💎 Опыт важных матчей")
        
        if underdog_odds > 4.0:
            factors.append(f"🎯 Большой андердог (коэф. {underdog_odds}) - высокий потенциал сенсации")
        elif underdog_odds > 2.5:
            factors.append(f"⚖️ Средний андердог - разумные шансы")
        
        return factors[:4]

# ИНИЦИАЛИЗАЦИЯ ПРЕДИКТОРА ПОСЛЕ ОПРЕДЕЛЕНИЯ КЛАССА
predictor = SmartUnderdogPredictor()

def get_fallback_demo_matches():
    """Fallback демо матчи"""
    return [
        {
            'player1': 'Demo Player A', 'player2': 'Demo Player B',
            'odds1': 2.20, 'odds2': 1.75,
            'tournament': 'Demo Tournament', 'round': 'Demo',
            'court': 'Demo Court', 'time': 'TBD',
            'surface': 'Hard', 'level': 'Demo'
        }
    ]

def extract_best_odds_from_api(bookmakers):
    """Извлекает лучшие коэффициенты из данных The Odds API"""
    best_odds1 = None
    best_odds2 = None
    
    for bookmaker in bookmakers:
        for market in bookmaker.get('markets', []):
            if market.get('key') == 'h2h':
                outcomes = market.get('outcomes', [])
                if len(outcomes) >= 2:
                    odds1 = outcomes[0].get('price')
                    odds2 = outcomes[1].get('price')
                    
                    if odds1 and odds2:
                        # Берем максимальные коэффициенты (лучшие для игрока)
                        if not best_odds1 or odds1 > best_odds1:
                            best_odds1 = odds1
                        if not best_odds2 or odds2 > best_odds2:
                            best_odds2 = odds2
    
    return best_odds1, best_odds2

def get_live_matches_with_api_economy():
    """НОВОЕ: Получение матчей через API Economy"""
    
    try:
        # ИСПОЛЬЗУЕМ API ECONOMY ДЛЯ ПОЛУЧЕНИЯ ДАННЫХ
        logger.info("🌍 Using API Economy for live matches")
        
        # Получаем данные через экономичную систему
        result = economical_tennis_request('tennis')
        
        if result['success']:
            raw_data = result['data']
            source_info = f"API_ECONOMY_{result['status']}"
            
            # Преобразуем данные The Odds API в наш формат
            converted_matches = []
            
            for api_match in raw_data:
                try:
                    # Извлекаем базовую информацию
                    player1 = api_match.get('home_team', 'Player 1')
                    player2 = api_match.get('away_team', 'Player 2')
                    
                    # Извлекаем лучшие коэффициенты
                    odds1, odds2 = extract_best_odds_from_api(api_match.get('bookmakers', []))
                    
                    if odds1 and odds2:
                        match = {
                            'player1': player1,
                            'player2': player2,
                            'odds1': odds1,
                            'odds2': odds2,
                            'tournament': 'Live Tournament',
                            'surface': 'Hard',  # По умолчанию
                            'round': 'Live',
                            'court': 'TBD',
                            'time': api_match.get('commence_time', '14:00')[:5] if api_match.get('commence_time') else '14:00'
                        }
                        converted_matches.append(match)
                        
                except Exception as e:
                    logger.warning(f"Ошибка обработки матча: {e}")
                    continue
            
            if converted_matches:
                return {
                    'matches': converted_matches,
                    'source': source_info,
                    'api_status': result.get('emoji', '📡'),
                    'success': True
                }
            else:
                logger.warning("Не удалось преобразовать матчи из API")
                
        else:
            logger.warning(f"API Economy error: {result.get('error', 'Unknown')}")
            
    except Exception as e:
        logger.error(f"❌ API Economy error: {e}")
    
    # Fallback к демо данным
    return {
        'matches': get_fallback_demo_matches(),
        'source': 'FALLBACK_DEMO', 
        'success': True
    }

def generate_quality_matches():
    """ОБНОВЛЕНО: Генерация с использованием API Economy"""
    
    # Получаем live матчи через API Economy
    live_data = get_live_matches_with_api_economy()
    potential_matches = live_data['matches']
    
    quality_matches = []
    
    for match_data in potential_matches:
        # Определяем андердога
        underdog_info = predictor.determine_underdog_from_odds(
            match_data['player1'], match_data['player2'],
            match_data['odds1'], match_data['odds2']
        )
        
        # Рассчитываем вероятность взять сет
        prediction = predictor.calculate_smart_set_probability(
            underdog_info['underdog'],
            underdog_info['favorite'], 
            underdog_info['underdog_odds'],
            underdog_info['favorite_odds']
        )
        
        # Проверяем качество
        if (prediction['probability'] >= 0.45 and 
            prediction['probability'] <= 0.88 and
            1.8 <= underdog_info['underdog_odds'] <= 8.0):
            
            match = {
                'id': f"api_economy_{len(quality_matches)+1}",
                'player1': f"🎾 {match_data['player1']}",
                'player2': f"🎾 {match_data['player2']}",
                'tournament': f"🏆 {match_data['tournament']}",
                'surface': match_data['surface'],
                'round': match_data['round'],
                'date': datetime.now().strftime('%Y-%m-%d'),
                'time': match_data['time'],
                
                'odds': {
                    'player1': match_data['odds1'],
                    'player2': match_data['odds2']
                },
                
                'underdog_analysis': {
                    'underdog': underdog_info['underdog'],
                    'favorite': underdog_info['favorite'],
                    'underdog_odds': underdog_info['underdog_odds'],
                    'prediction': prediction,
                    'quality_rating': 'HIGH' if prediction['probability'] > 0.70 else 'MEDIUM'
                },
                
                'focus': f"💎 {underdog_info['underdog']} взять хотя бы 1 сет",
                'recommendation': f"{prediction['probability']:.0%} шанс взять сет",
                'data_source': live_data['source'],
                'api_status': live_data.get('api_status', '📡')
            }
            
            quality_matches.append(match)
    
    # Сортируем по вероятности
    quality_matches.sort(key=lambda x: x['underdog_analysis']['prediction']['probability'], reverse=True)
    
    return quality_matches

def get_live_matches_with_api_economy():
    """НОВОЕ: Получение матчей через API Economy"""
    
    try:
        # ИСПОЛЬЗУЕМ API ECONOMY ДЛЯ ПОЛУЧЕНИЯ ДАННЫХ
        logger.info("🌍 Using API Economy for live matches")
        
        # Получаем данные через экономичную систему
        result = economical_tennis_request('tennis')
        
        if result['success']:
            raw_data = result['data']
            source_info = f"API_ECONOMY_{result['status']}"
            
            # Преобразуем данные The Odds API в наш формат
            converted_matches = []
            
            for api_match in raw_data:
                try:
                    # Извлекаем базовую информацию
                    player1 = api_match.get('home_team', 'Player 1')
                    player2 = api_match.get('away_team', 'Player 2')
                    
                    # Извлекаем лучшие коэффициенты
                    odds1, odds2 = extract_best_odds_from_api(api_match.get('bookmakers', []))
                    
                    if odds1 and odds2:
                        match = {
                            'player1': player1,
                            'player2': player2,
                            'odds1': odds1,
                            'odds2': odds2,
                            'tournament': 'Live Tournament',
                            'surface': 'Hard',  # По умолчанию
                            'round': 'Live',
                            'court': 'TBD',
                            'time': api_match.get('commence_time', '14:00')[:5] if api_match.get('commence_time') else '14:00'
                        }
                        converted_matches.append(match)
                        
                except Exception as e:
                    logger.warning(f"Ошибка обработки матча: {e}")
                    continue
            
            if converted_matches:
                return {
                    'matches': converted_matches,
                    'source': source_info,
                    'api_status': result.get('emoji', '📡'),
                    'success': True
                }
            else:
                logger.warning("Не удалось преобразовать матчи из API")
                
        else:
            logger.warning(f"API Economy error: {result.get('error', 'Unknown')}")
            
    except Exception as e:
        logger.error(f"❌ API Economy error: {e}")
    
    # Fallback к демо данным
    return {
        'matches': get_fallback_demo_matches(),
        'source': 'FALLBACK_DEMO', 
        'success': True
    }

def extract_best_odds_from_api(bookmakers):
    """Извлекает лучшие коэффициенты из данных The Odds API"""
    best_odds1 = None
    best_odds2 = None
    
    for bookmaker in bookmakers:
        for market in bookmaker.get('markets', []):
            if market.get('key') == 'h2h':
                outcomes = market.get('outcomes', [])
                if len(outcomes) >= 2:
                    odds1 = outcomes[0].get('price')
                    odds2 = outcomes[1].get('price')
                    
                    if odds1 and odds2:
                        # Берем максимальные коэффициенты (лучшие для игрока)
                        if not best_odds1 or odds1 > best_odds1:
                            best_odds1 = odds1
                        if not best_odds2 or odds2 > best_odds2:
                            best_odds2 = odds2
    
    return best_odds1, best_odds2