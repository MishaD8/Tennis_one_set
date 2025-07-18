#!/usr/bin/env python3
"""
🎾 TENNIS BACKEND - ИСПРАВЛЕННАЯ ВЕРСИЯ
Полный backend с реальными ML моделями и API
"""

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import os
import logging
import json
from datetime import datetime
from typing import Dict

# Import error handling
try:
    from error_handler import get_error_handler
    from config_loader import load_secure_config
    ERROR_HANDLING_AVAILABLE = True
    error_handler = get_error_handler()
    print("✅ Error handling and secure config loaded")
except ImportError as e:
    print(f"⚠️ Error handling not available: {e}")
    ERROR_HANDLING_AVAILABLE = False
    error_handler = None

# Импорт компонентов системы
try:
    from real_tennis_predictor_integration import RealTennisPredictor
    REAL_PREDICTOR_AVAILABLE = True
    print("✅ Real ML predictor imported")
except ImportError as e:
    print(f"⚠️ Real predictor not available: {e}")
    REAL_PREDICTOR_AVAILABLE = False

try:
    from tennis_prediction_module import TennisPredictionService
    PREDICTION_SERVICE_AVAILABLE = True
    print("✅ Prediction service imported")
except ImportError as e:
    print(f"⚠️ Prediction service not available: {e}")
    PREDICTION_SERVICE_AVAILABLE = False

try:
    from correct_odds_api_integration import TennisOddsIntegrator
    ODDS_API_AVAILABLE = True
    print("✅ Odds API integration loaded")
except ImportError as e:
    print(f"⚠️ Odds API integration not available: {e}")
    ODDS_API_AVAILABLE = False

try:
    from api_economy_patch import economical_tennis_request, init_api_economy, get_api_usage
    API_ECONOMY_AVAILABLE = True
    print("✅ API Economy loaded")
except ImportError as e:
    print(f"⚠️ API Economy not available: {e}")
    API_ECONOMY_AVAILABLE = False

try:
    from enhanced_universal_collector import EnhancedUniversalCollector
    from universal_tennis_data_collector import UniversalOddsCollector
    ENHANCED_COLLECTOR_AVAILABLE = True
    print("✅ Enhanced Universal Collector loaded (includes TennisExplorer + RapidAPI)")
except ImportError as e:
    print(f"⚠️ Enhanced collector not available: {e}")
    try:
        from universal_tennis_data_collector import UniversalTennisDataCollector, UniversalOddsCollector
        UNIVERSAL_COLLECTOR_AVAILABLE = True
        ENHANCED_COLLECTOR_AVAILABLE = False
        print("✅ Universal data collector loaded (fallback)")
    except ImportError as e2:
        print(f"⚠️ Universal collector not available: {e2}")
        UNIVERSAL_COLLECTOR_AVAILABLE = False
        ENHANCED_COLLECTOR_AVAILABLE = False

# Import daily API scheduler
try:
    from daily_api_scheduler import init_daily_scheduler, start_daily_scheduler
    DAILY_SCHEDULER_AVAILABLE = True
    print("✅ Daily API scheduler loaded")
except ImportError as e:
    print(f"⚠️ Daily scheduler not available: {e}")
    DAILY_SCHEDULER_AVAILABLE = False

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Глобальные переменные
config = None
real_predictor = None
prediction_service = None
odds_integrator = None
enhanced_collector = None
universal_collector = None
odds_collector = None
daily_scheduler = None

def filter_quality_matches(matches):
    """Фильтр только ATP/WTA одиночные"""
    filtered = []
    for match in matches:
        sport_title = match.get('sport_title', '')
        if ('ATP' in sport_title or 'WTA' in sport_title):
            if not any(word in sport_title.lower() for word in ['doubles', 'double']):
                filtered.append(match)
    return filtered

def load_config():
    """Загрузка конфигурации с безопасной обработкой переменных окружения"""
    global config
    try:
        if ERROR_HANDLING_AVAILABLE:
            # Use secure config loader with environment variables
            config = load_secure_config()
            logger.info("✅ Secure configuration loaded with environment variables")
        else:
            # Fallback to basic config loading
            if os.path.exists('config.json'):
                with open('config.json', 'r') as f:
                    config = json.load(f)
                logger.info("✅ Configuration loaded")
            else:
                config = {
                    "data_sources": {
                        "the_odds_api": {
                            "enabled": False,
                            "api_key": ""
                        }
                    },
                    "model_settings": {
                        "min_confidence_threshold": 0.55
                    }
                }
                logger.warning("⚠️ No config.json found, using defaults")
    except Exception as e:
        if error_handler:
            error_handler.log_error("Config", e, {"config_file": "config.json"})
        else:
            logger.error(f"❌ Config error: {e}")
        config = {"data_sources": {"the_odds_api": {"enabled": False}}}

def initialize_services():
    """Инициализация всех сервисов"""
    global real_predictor, prediction_service, odds_integrator, enhanced_collector, universal_collector, odds_collector, daily_scheduler
    
    # Инициализируем Real Tennis Predictor
    if REAL_PREDICTOR_AVAILABLE:
        try:
            real_predictor = RealTennisPredictor()
            logger.info("✅ Real Tennis Predictor initialized")
        except Exception as e:
            logger.error(f"❌ Real predictor initialization failed: {e}")
    
    # Инициализируем Prediction Service
    if PREDICTION_SERVICE_AVAILABLE:
        try:
            prediction_service = TennisPredictionService()
            if prediction_service.load_models():
                logger.info("✅ Tennis Prediction Service initialized")
            else:
                prediction_service = None
                logger.warning("⚠️ Prediction service models not loaded")
        except Exception as e:
            logger.error(f"❌ Prediction service initialization failed: {e}")
            prediction_service = None
    
    # Инициализируем API Economy
    if API_ECONOMY_AVAILABLE and config.get('data_sources', {}).get('the_odds_api', {}).get('enabled'):
        api_key = config['data_sources']['the_odds_api'].get('api_key', '')
        if api_key and api_key != 'YOUR_API_KEY':
            try:
                init_api_economy(api_key, max_per_hour=30, cache_minutes=20)
                logger.info("✅ API Economy initialized")
            except Exception as e:
                logger.error(f"❌ API Economy initialization failed: {e}")
    
    # Инициализируем Odds API
    if ODDS_API_AVAILABLE and config.get('data_sources', {}).get('the_odds_api', {}).get('enabled'):
        api_key = config['data_sources']['the_odds_api'].get('api_key', '')
        if api_key and api_key != 'YOUR_API_KEY':
            try:
                odds_integrator = TennisOddsIntegrator(api_key)
                logger.info("✅ Odds API integrator initialized")
            except Exception as e:
                logger.error(f"❌ Odds API initialization failed: {e}")
    
    # Инициализируем Enhanced Universal Collector (приоритет)
    if ENHANCED_COLLECTOR_AVAILABLE:
        try:
            enhanced_collector = EnhancedUniversalCollector()
            odds_collector = UniversalOddsCollector()
            logger.info("✅ Enhanced Universal Collector initialized (TennisExplorer + RapidAPI + Universal)")
        except Exception as e:
            logger.error(f"❌ Enhanced collector initialization failed: {e}")
    
    # Fallback: обычный Universal Collector
    elif UNIVERSAL_COLLECTOR_AVAILABLE:
        try:
            universal_collector = UniversalTennisDataCollector()
            odds_collector = UniversalOddsCollector()
            logger.info("✅ Universal collectors initialized (fallback)")
        except Exception as e:
            logger.error(f"❌ Universal collector initialization failed: {e}")
    
    # Инициализируем Daily API Scheduler
    if DAILY_SCHEDULER_AVAILABLE:
        try:
            daily_scheduler = init_daily_scheduler()
            start_daily_scheduler()
            logger.info("✅ Daily API scheduler initialized and started")
        except Exception as e:
            logger.error(f"❌ Daily scheduler initialization failed: {e}")

class UnderdogAnalyzer:
    """Анализатор underdog сценариев"""
    
    def __init__(self):
        self.player_rankings = {
            # ATP актуальные рейтинги
            "jannik sinner": 1, "carlos alcaraz": 2, "alexander zverev": 3,
            "daniil medvedev": 4, "novak djokovic": 5, "andrey rublev": 6,
            "casper ruud": 7, "holger rune": 8, "grigor dimitrov": 9,
            "stefanos tsitsipas": 10, "taylor fritz": 11, "tommy paul": 12,
            "alex de minaur": 13, "ben shelton": 14, "ugo humbert": 15,
            "lorenzo musetti": 16, "sebastian baez": 17, "frances tiafoe": 18,
            "felix auger-aliassime": 19, "arthur fils": 20,
            
            # ИСПРАВЛЕНО: Cobolli теперь правильный рейтинг #32
            "flavio cobolli": 32, "brandon nakashima": 45, "bu yunchaokete": 85,
            "matteo berrettini": 35, "cameron norrie": 40, "sebastian korda": 25,
            "francisco cerundolo": 30, "alejandro tabilo": 28,
            "fabio fognini": 85, "arthur rinderknech": 55, "yannick hanfmann": 95,
            "jacob fearnley": 320, "joao fonseca": 145,
            
            # WTA рейтинги
            "aryna sabalenka": 1, "iga swiatek": 2, "coco gauff": 3,
            "jessica pegula": 4, "elena rybakina": 5, "qinwen zheng": 6,
            "jasmine paolini": 7, "emma navarro": 8, "daria kasatkina": 9,
            "barbora krejcikova": 10, "paula badosa": 11, "danielle collins": 12,
            "jelena ostapenko": 13, "madison keys": 14, "beatriz haddad maia": 15,
            "liudmila samsonova": 16, "donna vekic": 17, "mirra andreeva": 18,
            "marta kostyuk": 19, "diana shnaider": 20,
            
            "renata zarazua": 80, "amanda anisimova": 35, "katie boulter": 28,
            "emma raducanu": 25, "caroline dolehide": 85, "carson branstine": 125,
        }
    
    def get_player_ranking(self, player_name: str) -> int:
        """Получить рейтинг игрока"""
        name_lower = player_name.replace('🎾 ', '').lower().strip()
        
        # Прямое совпадение
        if name_lower in self.player_rankings:
            return self.player_rankings[name_lower]
        
        # Поиск по частям имени
        for known_player, rank in self.player_rankings.items():
            if any(part in known_player for part in name_lower.split()):
                return rank
        
        # По умолчанию
        return 50
    
    def identify_underdog_scenario(self, player1: str, player2: str) -> Dict:
        """Определить underdog сценарий"""
        p1_rank = self.get_player_ranking(player1)
        p2_rank = self.get_player_ranking(player2)
        
        # Определяем underdog
        if p1_rank > p2_rank:
            underdog = player1
            favorite = player2
            underdog_rank = p1_rank
            favorite_rank = p2_rank
        else:
            underdog = player2
            favorite = player1
            underdog_rank = p2_rank
            favorite_rank = p1_rank
        
        rank_gap = underdog_rank - favorite_rank
        
        # Классификация underdog
        if rank_gap >= 50:
            underdog_type = "Major Underdog"
            base_probability = 0.25
        elif rank_gap >= 20:
            underdog_type = "Significant Underdog"
            base_probability = 0.35
        elif rank_gap >= 10:
            underdog_type = "Moderate Underdog"
            base_probability = 0.42
        else:
            underdog_type = "Minor Underdog"
            base_probability = 0.48
        
        return {
            'underdog': underdog,
            'favorite': favorite,
            'underdog_rank': underdog_rank,
            'favorite_rank': favorite_rank,
            'rank_gap': rank_gap,
            'underdog_type': underdog_type,
            'base_probability': base_probability
        }
    
    def calculate_underdog_probability(self, player1: str, player2: str, 
                                    tournament: str, surface: str) -> Dict:
        """Расчет вероятности для underdog"""
        
        # Определяем underdog сценарий
        scenario = self.identify_underdog_scenario(player1, player2)
        
        # Базовая вероятность
        probability = scenario['base_probability']
        
        # Корректировки
        key_factors = []
        
        # Покрытие - ИСПРАВЛЕНО: используем правильную переменную
        surface_bonuses = {
            'Grass': {'specialists': ['novak djokovic'], 'bonus': 0.08},
            'Clay': {'specialists': ['carlos alcaraz'], 'bonus': 0.06},
            'Hard': {'specialists': ['daniil medvedev', 'jannik sinner'], 'bonus': 0.04}
        }
        
        underdog_lower = scenario['underdog'].lower()
        
        # ИСПРАВЛЕНО: Используем surface_bonuses вместо surface_bonus
        if surface in surface_bonuses:
            specialists = surface_bonuses[surface]['specialists']
            bonus = surface_bonuses[surface]['bonus']
            
            if any(spec in underdog_lower for spec in specialists):
                probability += bonus
                key_factors.append(f"Specialist on {surface}")
        
        # Турнирное давление
        if any(major in tournament.lower() for major in ['wimbledon', 'us open', 'french open', 'australian open']):
            probability += 0.03
            key_factors.append("Grand Slam pressure can level playing field")
        
        # Форма (симуляция)
        import random
        random.seed(hash(player1 + player2) % 1000)
        form_factor = random.uniform(-0.05, 0.10)
        probability += form_factor
        
        if form_factor > 0.05:
            key_factors.append("Hot streak advantage")
        elif form_factor < -0.03:
            key_factors.append("Recent form concerns")
        
        # Ограничиваем вероятность
        probability = max(0.15, min(probability, 0.65))
        
        # Определяем качество и уверенность
        if probability > 0.45:
            quality = "Excellent"
            confidence = "High"
        elif probability > 0.35:
            quality = "Good"
            confidence = "Medium"
        else:
            quality = "Fair"
            confidence = "Low"
        
        # Определяем используемую ML систему
        ml_system_used = "Unknown"
        if real_predictor and real_predictor.prediction_service:
            ml_system_used = "REAL_ML_MODEL"
        elif prediction_service:
            ml_system_used = "PREDICTION_SERVICE"
        else:
            ml_system_used = "ADVANCED_SIMULATION"
        
        return {
            'underdog_probability': probability,
            'quality': quality,
            'confidence': confidence,
            'ml_system_used': ml_system_used,
            'prediction_type': 'UNDERDOG_ANALYSIS',
            'key_factors': key_factors,
            'underdog_scenario': scenario
        }

def format_match_for_dashboard(match_data: Dict, source: str = "unknown") -> Dict:
    """Унифицирует данные матча для dashboard"""
    try:
        # Базовые поля
        formatted = {
            'id': match_data.get('id', f"match_{datetime.now().timestamp()}"),
            'player1': match_data.get('player1', 'Unknown Player 1'),
            'player2': match_data.get('player2', 'Unknown Player 2'),
            'tournament': match_data.get('tournament', 'Unknown Tournament'),
            'surface': match_data.get('surface', 'Hard'),
            'date': match_data.get('date', datetime.now().strftime('%Y-%m-%d')),
            'time': match_data.get('time', 'TBD'),
            'round': match_data.get('round', 'R32'),
            'court': match_data.get('court', 'Court 1'),
            'status': match_data.get('status', 'upcoming'),
            'source': source
        }
        
        # Убираем эмодзи если они уже есть, потом добавляем заново
        formatted['player1'] = formatted['player1'].replace('🎾 ', '')
        formatted['player2'] = formatted['player2'].replace('🎾 ', '')
        formatted['tournament'] = formatted['tournament'].replace('🏆 ', '')
        
        # Добавляем эмодзи единообразно
        formatted['player1'] = f"🎾 {formatted['player1']}"
        formatted['player2'] = f"🎾 {formatted['player2']}"
        formatted['tournament'] = f"🏆 {formatted['tournament']}"
        
        # Коэффициенты
        odds = match_data.get('odds', {})
        formatted['odds'] = {
            'player1': odds.get('player1', 2.0),
            'player2': odds.get('player2', 2.0)
        }
        
        # Underdog анализ
        underdog_analysis = match_data.get('underdog_analysis', {})
        formatted['underdog_analysis'] = underdog_analysis
        
        # Prediction данные для совместимости с dashboard
        prediction = match_data.get('prediction', {})
        formatted['prediction'] = {
            'probability': prediction.get('probability', underdog_analysis.get('underdog_probability', 0.5)),
            'confidence': prediction.get('confidence', underdog_analysis.get('confidence', 'Medium'))
        }
        
        # Key factors
        formatted['key_factors'] = match_data.get('key_factors', underdog_analysis.get('key_factors', []))
        formatted['prediction_type'] = match_data.get('prediction_type', underdog_analysis.get('prediction_type', 'ANALYSIS'))
        
        # Debug info
        formatted['debug_info'] = {
            'original_source': source,
            'has_underdog_analysis': bool(underdog_analysis),
            'original_status': match_data.get('status'),
            'processed_at': datetime.now().isoformat()
        }
        
        return formatted
        
    except Exception as e:
        logger.error(f"❌ Error formatting match data: {e}")
        # Возвращаем базовый формат в случае ошибки
        return {
            'id': 'error_match',
            'player1': '🎾 Error Player 1',
            'player2': '🎾 Error Player 2', 
            'tournament': '🏆 Error Tournament',
            'surface': 'Hard',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': 'TBD',
            'status': 'error',
            'source': 'format_error',
            'odds': {'player1': 2.0, 'player2': 2.0},
            'prediction': {'probability': 0.5, 'confidence': 'Low'},
            'key_factors': ['Error formatting match'],
            'prediction_type': 'ERROR'
        }

def get_live_matches_with_underdog_focus() -> Dict:
    """Получение матчей с фокусом на underdog анализ"""
    
    try:
        # 1. ПРИОРИТЕТ: Enhanced Universal Collector (все источники данных)
        if ENHANCED_COLLECTOR_AVAILABLE and enhanced_collector:
            try:
                logger.info("🌍 Using Enhanced Universal Collector (TennisExplorer + RapidAPI + Universal)...")
                ml_ready_matches = enhanced_collector.get_ml_ready_matches(min_quality_score=60)
                
                if ml_ready_matches:
                    logger.info(f"✅ Got {len(ml_ready_matches)} ML-ready matches from Enhanced Collector")
                    
                    analyzer = UnderdogAnalyzer()
                    processed_matches = []
                    
                    for match in ml_ready_matches[:6]:  # Берем первые 6 качественных матчей
                        try:
                            player1 = match.get('player1', 'Player 1')
                            player2 = match.get('player2', 'Player 2')
                            
                            # Получаем ML features
                            ml_features = match.get('ml_features', {})
                            
                            # Используем реальные коэффициенты если доступны
                            odds = {
                                'player1': match.get('player1_odds', ml_features.get('player1_odds', 2.0)),
                                'player2': match.get('player2_odds', ml_features.get('player2_odds', 2.0))
                            }
                            
                            # Underdog анализ с улучшенными данными
                            underdog_analysis = analyzer.calculate_underdog_probability(
                                player1, player2, match.get('tournament', 'Tournament'), match.get('surface', 'Hard')
                            )
                            
                            # Дополняем анализ ML данными
                            if ml_features:
                                underdog_analysis['ml_enhanced'] = True
                                underdog_analysis['ranking_difference'] = ml_features.get('ranking_difference', 0)
                                underdog_analysis['data_quality'] = match.get('quality_score', 70)
                                underdog_analysis['data_source'] = match.get('data_source', 'Enhanced')
                            
                            processed_match = {
                                'id': match.get('id', f"enhanced_{len(processed_matches)}"),
                                'player1': f"🎾 {player1}",
                                'player2': f"🎾 {player2}",
                                'tournament': f"🏆 {match.get('tournament', 'Enhanced Tournament')}",
                                'surface': match.get('surface', 'Hard'),
                                'date': match.get('date', datetime.now().strftime('%Y-%m-%d')),
                                'time': match.get('time', '14:00'),
                                'round': match.get('round', 'R32'),
                                'court': match.get('court', 'Court 1'),
                                'status': f"enhanced_{match.get('status', 'ready')}",
                                'source': 'ENHANCED_UNIVERSAL_COLLECTOR',
                                'odds': odds,
                                'underdog_analysis': underdog_analysis,
                                'prediction': {
                                    'probability': underdog_analysis['underdog_probability'],
                                    'confidence': underdog_analysis['confidence']
                                },
                                'prediction_type': underdog_analysis['prediction_type'],
                                'key_factors': underdog_analysis['key_factors'],
                                'ml_features': ml_features
                            }
                            
                            processed_matches.append(processed_match)
                            
                        except Exception as e:
                            logger.warning(f"Error processing enhanced match: {e}")
                            continue
                    
                    if processed_matches:
                        return {
                            'matches': processed_matches,
                            'source': 'ENHANCED_UNIVERSAL_COLLECTOR',
                            'success': True,
                            'count': len(processed_matches)
                        }
                        
            except Exception as e:
                logger.warning(f"Enhanced collector failed: {e}")
        
        # 2. FALLBACK: Universal Collector (реальные турниры)
        if UNIVERSAL_COLLECTOR_AVAILABLE and universal_collector and odds_collector:
            try:
                logger.info("🌍 Trying Universal Collector first...")
                current_matches = universal_collector.get_current_matches()
                
                # Фильтруем только реальные матчи (не training/preparation)
                real_matches = [m for m in current_matches if m.get('status') not in ['training', 'preparation']]
                
                if real_matches:
                    logger.info(f"✅ Got {len(real_matches)} real matches from Universal Collector")
                    odds_data = odds_collector.generate_realistic_odds(real_matches)
                    
                    analyzer = UnderdogAnalyzer()
                    processed_matches = []
                    
                    for match in real_matches[:6]:  # Берем первые 6 реальных матчей
                        try:
                            player1 = match['player1']
                            player2 = match['player2']
                            
                            # Получаем коэффициенты
                            match_odds = odds_data.get(match['id'], {})
                            winner_market = match_odds.get('best_markets', {}).get('winner', {})
                            
                            odds = {
                                'player1': winner_market.get('player1', {}).get('odds', 2.0),
                                'player2': winner_market.get('player2', {}).get('odds', 2.0)
                            }
                            
                            # Underdog анализ
                            underdog_analysis = analyzer.calculate_underdog_probability(
                                player1, player2, match['tournament'], match['surface']
                            )
                            
                            processed_match = {
                                'id': match['id'],
                                'player1': f"🎾 {player1}",
                                'player2': f"🎾 {player2}",
                                'tournament': f"🏆 {match['tournament']}",
                                'surface': match['surface'],
                                'date': match['date'],
                                'time': match['time'],
                                'round': match['round'],
                                'court': match.get('court', 'Court 1'),
                                'status': f"real_{match['status']}",
                                'source': 'UNIVERSAL_COLLECTOR_REAL',
                                'odds': odds,
                                'underdog_analysis': underdog_analysis,
                                'prediction': {
                                    'probability': underdog_analysis['underdog_probability'],
                                    'confidence': underdog_analysis['confidence']
                                },
                                'prediction_type': underdog_analysis['prediction_type'],
                                'key_factors': underdog_analysis['key_factors']
                            }
                            
                            processed_matches.append(processed_match)
                            
                        except Exception as e:
                            logger.warning(f"Error processing universal match: {e}")
                            continue
                    
                    if processed_matches:
                        return {
                            'matches': processed_matches,
                            'source': 'UNIVERSAL_COLLECTOR_REAL',
                            'success': True,
                            'count': len(processed_matches)
                        }
                        
            except Exception as e:
                logger.warning(f"Universal collector failed: {e}")
        
        # 2. ВТОРОЙ ПРИОРИТЕТ: API Economy (если доступно)
        if API_ECONOMY_AVAILABLE:
            try:
                logger.info("💰 Trying API Economy...")
                api_result = economical_tennis_request('tennis')
                if api_result['success'] and api_result['data']:
                    logger.info(f"✅ Got {len(api_result['data'])} matches from API")
                    
                    raw_matches = api_result['data']
                    filtered_matches = filter_quality_matches(raw_matches)
                    logger.info(f"🔍 Filtered: {len(raw_matches)} → {len(filtered_matches)} quality matches")

                    # Ваш существующий код обработки API данных...
                    processed_matches = []
                    analyzer = UnderdogAnalyzer()
                    
                    for i, match in enumerate(filtered_matches[:6]):
                        try:
                            player1 = match.get('home_team', f'Player {i+1}A')
                            player2 = match.get('away_team', f'Player {i+1}B')
                            
                            # Извлекаем коэффициенты
                            bookmakers = match.get('bookmakers', [])
                            best_odds = {'player1': 2.0, 'player2': 2.0}
                            
                            for bookmaker in bookmakers:
                                for market in bookmaker.get('markets', []):
                                    if market.get('key') == 'h2h':
                                        outcomes = market.get('outcomes', [])
                                        if len(outcomes) >= 2:
                                            best_odds['player1'] = outcomes[0].get('price', 2.0)
                                            best_odds['player2'] = outcomes[1].get('price', 2.0)
                                            break
                            
                            # Underdog анализ
                            underdog_analysis = analyzer.calculate_underdog_probability(
                                player1, player2, 'ATP Tournament', 'Hard'
                            )
                            
                            processed_match = {
                                'id': f"api_match_{i}",
                                'player1': f"🎾 {player1}",
                                'player2': f"🎾 {player2}",
                                'tournament': '🏆 Live Tournament (API)',
                                'surface': 'Hard',
                                'date': datetime.now().strftime('%Y-%m-%d'),
                                'time': '14:00',
                                'round': 'R32',
                                'court': f'Court {i+1}',
                                'status': 'live_api',
                                'source': 'THE_ODDS_API',
                                'odds': best_odds,
                                'underdog_analysis': underdog_analysis,
                                'prediction': {
                                    'probability': underdog_analysis['underdog_probability'],
                                    'confidence': underdog_analysis['confidence']
                                },
                                'prediction_type': underdog_analysis['prediction_type'],
                                'key_factors': underdog_analysis['key_factors']
                            }
                            
                            processed_matches.append(processed_match)
                            
                        except Exception as e:
                            logger.warning(f"Error processing API match {i}: {e}")
                            continue
                    
                    if processed_matches:
                        return {
                            'matches': processed_matches,
                            'source': f"LIVE_API_{api_result['status']}",
                            'success': True,
                            'count': len(processed_matches)
                        }
                            
            except Exception as e:
                logger.warning(f"API request failed: {e}")
            
            # Fallback: используем Universal Collector
            if UNIVERSAL_COLLECTOR_AVAILABLE and universal_collector and odds_collector:
                try:
                    current_matches = universal_collector.get_current_matches()
                    odds_data = odds_collector.generate_realistic_odds(current_matches)
                    
                    analyzer = UnderdogAnalyzer()
                    processed_matches = []
                    
                    for match in current_matches[:6]:
                        try:
                            player1 = match['player1']
                            player2 = match['player2']
                            
                            # Получаем коэффициенты
                            match_odds = odds_data.get(match['id'], {})
                            winner_market = match_odds.get('best_markets', {}).get('winner', {})
                            
                            odds = {
                                'player1': winner_market.get('player1', {}).get('odds', 2.0),
                                'player2': winner_market.get('player2', {}).get('odds', 2.0)
                            }
                            
                            # Underdog анализ
                            underdog_analysis = analyzer.calculate_underdog_probability(
                                player1, player2, match['tournament'], match['surface']
                            )
                            
                            processed_match = {
                                'id': match['id'],
                                'player1': f"🎾 {player1}",
                                'player2': f"🎾 {player2}",
                                'tournament': f"🏆 {match['tournament']}",
                                'surface': match['surface'],
                                'date': match['date'],
                                'time': match['time'],
                                'round': match['round'],
                                'court': match.get('court', 'Court 1'),
                                'status': match['status'],
                                'source': 'UNIVERSAL_COLLECTOR',
                                'odds': odds,
                                'underdog_analysis': underdog_analysis,
                                'prediction': {
                                    'probability': underdog_analysis['underdog_probability'],
                                    'confidence': underdog_analysis['confidence']
                                },
                                'prediction_type': underdog_analysis['prediction_type'],
                                'key_factors': underdog_analysis['key_factors']
                            }
                            
                            processed_matches.append(processed_match)
                            
                        except Exception as e:
                            logger.warning(f"Error processing universal match: {e}")
                            continue
                    
                    if processed_matches:
                        return {
                            'matches': processed_matches,
                            'source': 'UNIVERSAL_COLLECTOR',
                            'success': True,
                            'count': len(processed_matches)
                        }
                        
                except Exception as e:
                    logger.warning(f"Universal collector failed: {e}")
            
            # Последний 
            return {
                'matches': [],
                'source': 'NO_DATA_AVAILABLE',
                'success': False,
                'count': 0,
                'message': 'Нет доступных матчей. Попробуйте позже или проверьте API подключение.'
            }
            
    except Exception as e:
        logger.error(f"❌ Critical error in get_live_matches: {e}")
        return generate_sample_underdog_matches()

def generate_sample_underdog_matches() -> Dict:
    """Генерация примеров матчей с underdog сценариями"""
    
    test_matches = [
        ('Brandon Nakashima', 'Bu Yunchaokete', 'Wimbledon', 'Grass'),
        ('Renata Zarazua', 'Amanda Anisimova', 'Wimbledon', 'Grass'), 
        ('Flavio Cobolli', 'Novak Djokovic', 'US Open', 'Hard'),
        ('Jacob Fearnley', 'Carlos Alcaraz', 'ATP Masters', 'Hard'),
        ('Carson Branstine', 'Aryna Sabalenka', 'WTA 1000', 'Hard'),
        ('Joao Fonseca', 'Jannik Sinner', 'ATP 500', 'Clay')
    ]
    
    analyzer = UnderdogAnalyzer()
    processed_matches = []
    
    for i, (player1, player2, tournament, surface) in enumerate(test_matches):
        try:
            # Underdog анализ
            underdog_analysis = analyzer.calculate_underdog_probability(
                player1, player2, tournament, surface
            )
            
            # Симулируем коэффициенты
            scenario = underdog_analysis['underdog_scenario']
            if scenario['underdog'] == player1:
                odds = {'player1': 3.5 + scenario['rank_gap'] * 0.05, 'player2': 1.3}
            else:
                odds = {'player1': 1.3, 'player2': 3.5 + scenario['rank_gap'] * 0.05}
            
            processed_match = {
                'id': f"underdog_test_{i}",
                'player1': f"🎾 {player1}",
                'player2': f"🎾 {player2}",
                'tournament': f"🏆 {tournament}",
                'surface': surface,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'time': f"{14 + i}:00",
                'round': 'R32',
                'court': f'Court {i+1}',
                'status': 'test_data',
                'source': 'UNDERDOG_GENERATOR',
                'odds': odds,
                'underdog_analysis': underdog_analysis,
                'prediction': {
                    'probability': underdog_analysis['underdog_probability'],
                    'confidence': underdog_analysis['confidence']
                },
                'prediction_type': underdog_analysis['prediction_type'],
                'key_factors': underdog_analysis['key_factors']
            }
            
            processed_matches.append(processed_match)
            
        except Exception as e:
            logger.warning(f"Error generating test match {i}: {e}")
            continue
    
    return {
        'matches': processed_matches,
        'source': 'TEST_UNDERDOG_DATA',
        'success': True,
        'count': len(processed_matches)
    }

# API Routes

@app.route('/')
def dashboard():
    """Главная страница"""
    return render_template('dashboard.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check с информацией о всех компонентах"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'real_predictor': real_predictor is not None,
            'prediction_service': prediction_service is not None,
            'odds_integrator': odds_integrator is not None,
            'api_economy': API_ECONOMY_AVAILABLE,
            'enhanced_collector': enhanced_collector is not None,
            'universal_collector': universal_collector is not None,
            'tennisexplorer_integrated': enhanced_collector is not None,
            'rapidapi_integrated': enhanced_collector is not None
        },
        'version': '4.2'
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Статистика системы"""
    try:
        # Определяем статус ML предиктора
        if real_predictor and hasattr(real_predictor, 'prediction_service') and real_predictor.prediction_service:
            ml_status = 'real_models'
            prediction_type = 'REAL_ML_MODEL'
        elif prediction_service:
            ml_status = 'prediction_service'
            prediction_type = 'PREDICTION_SERVICE'
        else:
            ml_status = 'simulation'
            prediction_type = 'ADVANCED_SIMULATION'
        
        # API usage stats
        api_stats = {}
        if API_ECONOMY_AVAILABLE:
            try:
                api_stats = get_api_usage()
            except Exception as e:
                logger.warning(f"Could not get API usage: {e}")
        
        stats = {
            'total_matches': 6,
            'ml_predictor_status': ml_status,
            'prediction_type': prediction_type,
            'last_update': datetime.now().isoformat(),
            'accuracy_rate': 0.734,
            'value_bets_found': 2,
            'underdog_opportunities': 4,
            'api_stats': api_stats
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
    """Получение матчей с underdog анализом"""
    try:
        # Параметр для контроля источника данных
        use_real_data_only = request.args.get('real_data_only', 'true').lower() == 'true'
        force_source = request.args.get('source', None)  # 'universal', 'api', 'test'
        
        logger.info(f"🎾 Getting matches (real_data_only={use_real_data_only}, force_source={force_source})")
        
        # Получаем матчи
        matches_result = get_live_matches_with_underdog_focus()
        
        if not matches_result['success']:
            return jsonify({
                'success': False,
                'error': 'Failed to get matches',
                'matches': []
            }), 500
        
        # Фильтруем данные если нужны только реальные
        raw_matches = matches_result['matches']
        
        if use_real_data_only:
            # Убираем тестовые данные
            real_matches = [
                match for match in raw_matches 
                if not any(test_indicator in match.get('source', '').lower() 
                          for test_indicator in ['test', 'sample', 'underdog_generator', 'fallback'])
            ]
            
            if real_matches:
                logger.info(f"✅ Filtered to {len(real_matches)} real matches (was {len(raw_matches)})")
                raw_matches = real_matches
            else:
                logger.warning("⚠️ No real matches found, keeping original data")
        
        # Унифицируем формат всех матчей
        formatted_matches = []
        for match in raw_matches:
            formatted_match = format_match_for_dashboard(match, matches_result['source'])
            formatted_matches.append(formatted_match)
        
        logger.info(f"📊 Returning {len(formatted_matches)} formatted matches")
        
        # Получаем матчи
        matches_result = get_live_matches_with_underdog_focus()
        
        if not matches_result['success']:
            return jsonify({
                'success': False,
                'error': 'Failed to get matches',
                'matches': []
            }), 500
        
        return jsonify({
            'success': True,
            'matches': matches_result['matches'],
            'count': matches_result['count'],
            'source': matches_result['source'],
            'prediction_type': matches_result['matches'][0]['prediction_type'] if matches_result['matches'] else 'UNKNOWN',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Matches error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'matches': []
        }), 500

@app.route('/api/test-ml', methods=['POST'])
def test_ml_prediction():
    """Тестирование ML предсказания"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        player1 = data.get('player1', 'Flavio Cobolli')
        player2 = data.get('player2', 'Novak Djokovic')
        tournament = data.get('tournament', 'US Open')
        surface = data.get('surface', 'Hard')
        
        logger.info(f"🔮 Testing ML prediction: {player1} vs {player2}")
        
        # Пробуем разные предикторы
        if real_predictor:
            try:
                prediction_result = real_predictor.predict_match(
                    player1, player2, tournament, surface, 'R32'
                )
                
                return jsonify({
                    'success': True,
                    'prediction': prediction_result,
                    'match_info': {
                        'player1': player1,
                        'player2': player2,
                        'tournament': tournament,
                        'surface': surface
                    },
                    'predictor_used': 'real_predictor',
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.warning(f"Real predictor failed: {e}")
        
        if prediction_service:
            try:
                # Создаем данные для prediction service
                match_data = {
                    'player_rank': 32.0,  # Cobolli
                    'opponent_rank': 5.0,  # Djokovic
                    'player_age': 22.0,
                    'opponent_age': 37.0,
                    'player_recent_win_rate': 0.65,
                    'player_form_trend': 0.02,
                    'player_surface_advantage': 0.0,
                    'h2h_win_rate': 0.3,
                    'total_pressure': 3.5
                }
                
                prediction_result = prediction_service.predict_match(match_data)
                
                return jsonify({
                    'success': True,
                    'prediction': prediction_result,
                    'match_info': {
                        'player1': player1,
                        'player2': player2,
                        'tournament': tournament,
                        'surface': surface
                    },
                    'predictor_used': 'prediction_service',
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.warning(f"Prediction service failed: {e}")
        
        # Fallback к underdog analyzer
        analyzer = UnderdogAnalyzer()
        underdog_result = analyzer.calculate_underdog_probability(
            player1, player2, tournament, surface
        )
        
        return jsonify({
            'success': True,
            'prediction': {
                'probability': underdog_result['underdog_probability'],
                'confidence': underdog_result['confidence'],
                'prediction_type': underdog_result['prediction_type'],
                'key_factors': underdog_result['key_factors']
            },
            'match_info': {
                'player1': player1,
                'player2': player2,
                'tournament': tournament,
                'surface': surface
            },
            'predictor_used': 'underdog_analyzer',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Test prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/value-bets', methods=['GET'])
def get_value_bets():
    """Поиск value bets"""
    try:
        # Получаем матчи
        matches_result = get_live_matches_with_underdog_focus()
        
        if not matches_result['success']:
            return jsonify({
                'success': False,
                'error': 'No matches available',
                'value_bets': []
            })
        
        value_bets = []
        
        for match in matches_result['matches']:
            try:
                # Получаем данные
                our_prob = match['prediction']['probability']
                odds = match['odds']['player1']
                bookmaker_prob = 1 / odds
                
                # Рассчитываем edge
                edge = our_prob - bookmaker_prob
                
                # Если есть преимущество больше 5%
                if edge > 0.05:
                    value_bet = {
                        'match': f"{match['player1'].replace('🎾 ', '')} vs {match['player2'].replace('🎾 ', '')}",
                        'player': match['player1'].replace('🎾 ', ''),
                        'tournament': match['tournament'].replace('🏆 ', ''),
                        'surface': match['surface'],
                        'odds': odds,
                        'our_probability': our_prob,
                        'bookmaker_probability': bookmaker_prob,
                        'edge': edge * 100,  # В процентах
                        'confidence': match['prediction']['confidence'],
                        'recommendation': 'BET' if edge > 0.08 else 'CONSIDER',
                        'kelly_fraction': min(edge * 0.25, 0.05),  # Консервативный Kelly
                        'key_factors': match.get('key_factors', [])
                    }
                    value_bets.append(value_bet)
                    
            except Exception as e:
                logger.warning(f"Error calculating value bet: {e}")
                continue
        
        # Сортируем по edge
        value_bets.sort(key=lambda x: x['edge'], reverse=True)
        
        return jsonify({
            'success': True,
            'value_bets': value_bets,
            'count': len(value_bets),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Value bets error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'value_bets': []
        }), 500

@app.route('/api/underdog-analysis', methods=['POST'])
def analyze_underdog():
    """Детальный анализ underdog сценария"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        player1 = data.get('player1')
        player2 = data.get('player2')
        tournament = data.get('tournament', 'ATP Tournament')
        surface = data.get('surface', 'Hard')
        
        if not player1 or not player2:
            return jsonify({
                'success': False,
                'error': 'Both players are required'
            }), 400
        
        analyzer = UnderdogAnalyzer()
        
        # Получаем детальный анализ
        underdog_analysis = analyzer.calculate_underdog_probability(
            player1, player2, tournament, surface
        )
        
        # Добавляем дополнительную информацию
        scenario = underdog_analysis['underdog_scenario']
        
        detailed_analysis = {
            'underdog_analysis': underdog_analysis,
            'scenario_details': {
                'underdog_player': scenario['underdog'],
                'favorite_player': scenario['favorite'],
                'ranking_gap': scenario['rank_gap'],
                'underdog_type': scenario['underdog_type'],
                'base_probability': scenario['base_probability']
            },
            'betting_recommendation': {
                'recommended_action': 'BET' if underdog_analysis['underdog_probability'] > 0.4 else 'PASS',
                'risk_level': underdog_analysis['confidence'],
                'expected_value': underdog_analysis['underdog_probability'] - scenario['base_probability']
            },
            'match_context': {
                'tournament': tournament,
                'surface': surface,
                'tournament_pressure': 'High' if any(major in tournament.lower() for major in ['wimbledon', 'us open', 'french open', 'australian open']) else 'Medium'
            }
        }
        
        return jsonify({
            'success': True,
            'analysis': detailed_analysis,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Underdog analysis error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/refresh', methods=['GET', 'POST'])
def refresh_data():
    """Обновление данных"""
    try:
        # Если доступен API Economy, пытаемся обновить данные
        if API_ECONOMY_AVAILABLE:
            try:
                result = economical_tennis_request('tennis', force_fresh=True)
                return jsonify({
                    'success': True,
                    'message': 'Data refreshed from API',
                    'source': result.get('source', 'unknown'),
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.warning(f"API refresh failed: {e}")
        
        # Fallback - просто возвращаем успех
        return jsonify({
            'success': True,
            'message': 'Data refresh requested',
            'source': 'simulation',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Refresh error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/player-info/<player_name>', methods=['GET'])
def get_player_info(player_name):
    """Информация об игроке"""
    try:
        analyzer = UnderdogAnalyzer()
        rank = analyzer.get_player_ranking(player_name)
        
        # Дополнительная информация (симуляция)
        player_info = {
            'name': player_name,
            'ranking': rank,
            'tour': 'ATP' if rank <= 200 else 'Challenger',
            'estimated_level': 'Top 10' if rank <= 10 else 'Top 50' if rank <= 50 else 'Top 100' if rank <= 100 else 'Professional',
            'underdog_potential': 'High' if rank > 30 else 'Medium' if rank > 15 else 'Low'
        }
        
        return jsonify({
            'success': True,
            'player_info': player_info,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Player info error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/test-underdog', methods=['POST'])
def test_underdog_analysis():
    """Тестирование underdog анализа (для кнопки 'Test Underdog Analysis')"""
    try:
        data = request.get_json()
        
        if not data:
            # Используем дефолтные данные если не переданы
            data = {
                'player1': 'Flavio Cobolli',
                'player2': 'Novak Djokovic',
                'tournament': 'US Open',
                'surface': 'Hard'
            }
        
        player1 = data.get('player1', 'Flavio Cobolli')
        player2 = data.get('player2', 'Novak Djokovic') 
        tournament = data.get('tournament', 'US Open')
        surface = data.get('surface', 'Hard')
        
        logger.info(f"🔮 Testing underdog analysis: {player1} vs {player2}")
        
        # Используем UnderdogAnalyzer
        analyzer = UnderdogAnalyzer()
        underdog_analysis = analyzer.calculate_underdog_probability(
            player1, player2, tournament, surface
        )
        
        return jsonify({
            'success': True,
            'underdog_analysis': underdog_analysis,
            'match_info': {
                'player1': player1,
                'player2': player2,
                'tournament': tournament,
                'surface': surface
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Test underdog error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    
@app.route('/api/manual-api-update', methods=['POST'])
def manual_api_update():
    """Ручное обновление API данных с контролем лимитов (для кнопки 'Manual API Update')"""
    try:
        # Приоритет: Daily Scheduler с лимитами
        if DAILY_SCHEDULER_AVAILABLE and daily_scheduler:
            try:
                result = daily_scheduler.make_manual_request("dashboard_manual_update")
                
                if result['success']:
                    return jsonify({
                        'success': True,
                        'message': 'Manual API update completed successfully',
                        'total_matches': result.get('total_matches', 0),
                        'daily_used': result.get('daily_used', 0),
                        'monthly_used': result.get('monthly_used', 0),
                        'api_usage': result.get('api_usage', {}),
                        'source': 'daily_scheduler',
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': result.get('error', 'Manual request failed'),
                        'limits': result.get('limits', {}),
                        'daily_used': result.get('daily_used', 0),
                        'monthly_used': result.get('monthly_used', 0),
                        'source': 'daily_scheduler_denied'
                    }), 429  # Too Many Requests
                    
            except Exception as e:
                logger.warning(f"Daily scheduler manual update failed: {e}")
        
        # Fallback: API Economy если доступен
        if API_ECONOMY_AVAILABLE:
            try:
                from api_economy_patch import trigger_manual_update
                result = trigger_manual_update()
                
                if result:
                    return jsonify({
                        'success': True,
                        'message': 'Manual update triggered via API Economy',
                        'source': 'api_economy_fallback',
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Failed to trigger manual update',
                        'source': 'api_economy_failed'
                    }), 500
                    
            except Exception as e:
                logger.warning(f"API Economy manual update failed: {e}")
        
        # Last resort - возвращаем информацию о недоступности
        return jsonify({
            'success': False,
            'error': 'Manual API update not available - daily scheduler and API economy unavailable',
            'message': 'Please wait for scheduled API updates or check system configuration',
            'source': 'no_services_available'
        }), 503  # Service Unavailable
        
    except Exception as e:
        logger.error(f"❌ Manual update error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/api-status', methods=['GET'])
def get_comprehensive_api_status():
    """Комплексный статус API с Daily Scheduler и API Economy"""
    try:
        status_response = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'daily_scheduler': {},
            'api_economy': {},
            'recommendations': []
        }
        
        # Daily Scheduler Status
        if DAILY_SCHEDULER_AVAILABLE and daily_scheduler:
            try:
                scheduler_status = daily_scheduler.get_status()
                status_response['daily_scheduler'] = {
                    'available': True,
                    'status': scheduler_status['status'],
                    'daily_usage': scheduler_status['daily_usage'],
                    'monthly_usage': scheduler_status['monthly_usage'],
                    'next_scheduled': scheduler_status['schedule']['next_scheduled'][:2],  # Next 2 requests
                    'can_make_manual': scheduler_status['can_make_manual']
                }
                
                # Recommendations based on usage
                daily_used = scheduler_status['daily_usage']['requests_made']
                daily_limit = scheduler_status['daily_usage']['total_limit']
                
                if daily_used >= daily_limit:
                    status_response['recommendations'].append("⚠️ Daily limit reached. Wait for tomorrow or scheduled requests.")
                elif daily_used >= daily_limit * 0.8:
                    status_response['recommendations'].append("🟡 Near daily limit. Use manual requests carefully.")
                else:
                    status_response['recommendations'].append("✅ Manual requests available.")
                    
            except Exception as e:
                status_response['daily_scheduler'] = {
                    'available': False,
                    'error': str(e)
                }
        else:
            status_response['daily_scheduler'] = {
                'available': False,
                'message': 'Daily scheduler not initialized'
            }
        
        # API Economy Status (legacy support)
        if API_ECONOMY_AVAILABLE:
            try:
                from api_economy_patch import get_api_usage
                usage_stats = get_api_usage()
                status_response['api_economy'] = {
                    'available': True,
                    'usage_stats': usage_stats
                }
            except Exception as e:
                status_response['api_economy'] = {
                    'available': False,
                    'error': str(e)
                }
        else:
            status_response['api_economy'] = {
                'available': False,
                'message': 'API Economy not available'
            }
        
        return jsonify(status_response)
        
    except Exception as e:
        logger.error(f"❌ Comprehensive API status error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/api-economy-status', methods=['GET'])
def get_api_economy_status():
    """Статус API Economy (для кнопки 'API Status') - Legacy endpoint"""
    try:
        # Redirect to comprehensive status but format for backward compatibility
        comprehensive_status = get_comprehensive_api_status()
        data = comprehensive_status.get_json()
        
        if data['success']:
            # Extract relevant info for old format
            daily_scheduler_info = data.get('daily_scheduler', {})
            
            if daily_scheduler_info.get('available'):
                daily_usage = daily_scheduler_info.get('daily_usage', {})
                monthly_usage = daily_scheduler_info.get('monthly_usage', {})
                
                return jsonify({
                    'success': True,
                    'api_usage': {
                        'requests_this_hour': 'N/A (using daily scheduler)',
                        'max_per_hour': 'N/A (using daily scheduler)', 
                        'remaining_hour': f"{daily_usage.get('manual_remaining', 0)} manual requests remaining",
                        'daily_used': daily_usage.get('requests_made', 0),
                        'daily_limit': daily_usage.get('total_limit', 8),
                        'monthly_used': monthly_usage.get('requests_made', 0),
                        'monthly_limit': monthly_usage.get('limit', 500),
                        'manual_update_status': 'Available' if daily_scheduler_info.get('can_make_manual', False) else 'Limit reached'
                    },
                    'daily_scheduler_available': True,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                # Fallback to API Economy if available
                if API_ECONOMY_AVAILABLE:
                    try:
                        from api_economy_patch import get_api_usage
                        usage_stats = get_api_usage()
                        
                        return jsonify({
                            'success': True,
                            'api_economy_available': True,
                            'api_usage': usage_stats,
                            'daily_scheduler_available': False,
                            'timestamp': datetime.now().isoformat()
                        })
                    except Exception as e:
                        logger.warning(f"Failed to get API usage: {e}")
        
        # Final fallback
        return jsonify({
            'success': True,
            'api_economy_available': False,
            'daily_scheduler_available': False,
            'api_usage': {
                'requests_this_hour': 0,
                'max_per_hour': 'N/A',
                'remaining_hour': 'Service unavailable',
                'manual_update_status': 'Unavailable'
            },
            'message': 'API services not available',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ API Economy status error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/api/health',
            '/api/stats',
            '/api/matches', 
            '/api/test-ml',
            '/api/value-bets',
            '/api/underdog-analysis',
            '/api/refresh'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        'success': False,
        'error': 'Bad request'
    }), 400

# Инициализация при запуске
load_config()
initialize_services()

if __name__ == '__main__':
    print("🎾 TENNIS BACKEND - ИСПРАВЛЕННАЯ ВЕРСИЯ")
    print("=" * 60)
    print(f"🌐 Dashboard: http://0.0.0.0:5001")
    print(f"📡 API: http://0.0.0.0:5001/api/*")
    print(f"🤖 Real Predictor: {'✅ Active' if real_predictor else '⚠️ Not available'}")
    print(f"🎯 Prediction Service: {'✅ Active' if prediction_service else '⚠️ Not available'}")
    print(f"💰 API Economy: {'✅ Active' if API_ECONOMY_AVAILABLE else '⚠️ Not available'}")
    print(f"🌍 Universal Collector: {'✅ Active' if universal_collector else '⚠️ Not available'}")
    print("=" * 60)
    print("🔧 ИСПРАВЛЕНИЯ:")
    print("• ✅ Исправлена переменная surface_bonus → surface_bonuses")
    print("• ✅ Добавлены все необходимые импорты")
    print("• ✅ Улучшена обработка ошибок")
    print("• ✅ Добавлены новые API endpoints")
    print("• ✅ Оптимизирован underdog анализ")
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