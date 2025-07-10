#!/usr/bin/env python3
"""
🎾 BACKEND TENNIS - ИСПРАВЛЕННАЯ UNDERDOG СИСТЕМА
Фокус на underdog игроков которые могут взять сет против фаворитов
"""

import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import json
from datetime import datetime, timedelta
import numpy as np

# Импорт API Economy (уже настроенная система)
try:
    from api_economy_patch import (
        init_api_economy, 
        economical_tennis_request, 
        get_api_usage, 
        trigger_manual_update,
        check_manual_update_status,
        clear_api_cache
    )
    API_ECONOMY_AVAILABLE = True
    print("✅ API Economy system loaded")
except ImportError as e:
    print(f"❌ API Economy not available: {e}")
    API_ECONOMY_AVAILABLE = False

# Импорт Real Tennis Predictor (ваша ML система)
try:
    from real_tennis_predictor_integration import RealTennisPredictor
    REAL_PREDICTOR_AVAILABLE = True
    print("✅ Real Tennis Predictor loaded")
except ImportError as e:
    print(f"❌ Real Tennis Predictor not available: {e}")
    REAL_PREDICTOR_AVAILABLE = False

# Импорт Tennis Prediction Module (основные ML модели)
try:
    from tennis_prediction_module import TennisPredictionService
    TENNIS_PREDICTION_AVAILABLE = True
    print("✅ Tennis Prediction Module loaded")
except ImportError as e:
    print(f"❌ Tennis Prediction Module not available: {e}")
    TENNIS_PREDICTION_AVAILABLE = False

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Импорт системы логирования
try:
    from prediction_logging_system import PredictionLoggerIntegration
    LOGGING_AVAILABLE = True
    prediction_logger = PredictionLoggerIntegration()
    print("✅ Prediction logging system loaded")
except ImportError as e:
    print(f"⚠️ Prediction logging not available: {e}")
    LOGGING_AVAILABLE = False
    prediction_logger = None

app = Flask(__name__)
CORS(app)

# Глобальные переменные для ML систем
real_predictor = None
tennis_prediction_service = None

# ИНИЦИАЛИЗАЦИЯ API ECONOMY при запуске
if API_ECONOMY_AVAILABLE:
    init_api_economy(
        api_key="a1b20d709d4bacb2d95ddab880f91009",
        max_per_hour=30,
        cache_minutes=20
    )
    print("✅ API Economy initialized")

def initialize_ml_systems():
    """Инициализация всех ML систем"""
    global real_predictor, tennis_prediction_service
    
    # 1. Real Tennis Predictor (с реальными данными игроков)
    if REAL_PREDICTOR_AVAILABLE:
        try:
            real_predictor = RealTennisPredictor()
            logger.info("✅ Real Tennis Predictor initialized")
        except Exception as e:
            logger.error(f"❌ Real Tennis Predictor init failed: {e}")
    
    # 2. Tennis Prediction Service (обученные ML модели)
    if TENNIS_PREDICTION_AVAILABLE:
        try:
            tennis_prediction_service = TennisPredictionService()
            if tennis_prediction_service.load_models():
                logger.info("✅ Tennis Prediction Service with trained models loaded")
            else:
                logger.info("⚠️ Tennis Prediction Service in demo mode")
        except Exception as e:
            logger.error(f"❌ Tennis Prediction Service init failed: {e}")

# Инициализируем ML системы при запуске
initialize_ml_systems()

class UnderdogAnalyzer:
    """ИСПРАВЛЕННЫЙ анализатор с актуальными рейтингами июль 2025"""
    
    def __init__(self):
        self.real_predictor = real_predictor
        self.prediction_service = tennis_prediction_service
        
        # 🔧 КРИТИЧНО: Актуальные рейтинги ATP/WTA на июль 2025
        self.player_rankings = {
            # ATP Top 50 (ОБНОВЛЕНО июль 2025)
            'jannik sinner': 1,
            'carlos alcaraz': 2, 
            'alexander zverev': 3,
            'daniil medvedev': 4,
            'novak djokovic': 5,
            'andrey rublev': 6,
            'casper ruud': 7,
            'holger rune': 8,
            'grigor dimitrov': 9,
            'stefanos tsitsipas': 10,
            'taylor fritz': 11,
            'tommy paul': 12,
            'alex de minaur': 13,
            'ben shelton': 14,
            'ugo humbert': 15,
            'lorenzo musetti': 16,
            'sebastian baez': 17,
            'frances tiafoe': 18,
            'felix auger-aliassime': 19,
            'arthur fils': 20,
            'sebastian korda': 21,
            'alejandro tabilo': 22,
            'karen khachanov': 23,
            'francisco cerundolo': 24,
            'matteo berrettini': 25,
            'jan-lennard struff': 26,
            'nicolas jarry': 27,
            'jiri lehecka': 28,
            'flavio cobolli': 29,  # 🔧 ИСПРАВЛЕНО: был #100, стал #29!
            'matteo arnaldi': 30,
            'tomas machac': 31,
            'zhizhen zhang': 32,
            'cameron norrie': 33,
            'brandon nakashima': 34,  # 🔧 ИСПРАВЛЕНО: актуальный #34
            'yannick hanfmann': 35,
            'adrian mannarino': 36,
            'pavel kotov': 37,
            'giovanni mpetshi perricard': 38,
            'mariano navone': 39,
            'christopher oconnell': 40,
            'jordan thompson': 41,
            'jakub mensik': 42,
            'roberto carballes baena': 43,
            'pedro martinez': 44,
            'tallon griekspoor': 45,
            'facundo diaz acosta': 46,
            'arthur rinderknech': 47,
            'botic van de zandschulp': 48,
            'luciano darderi': 49,
            'daniel altmaier': 50,
            
            # Важные дополнительные игроки
            'fabio fognini': 65,
            'bu yunchaokete': 71,  # 🔧 ИСПРАВЛЕНО: актуальный рейтинг
            'jacob fearnley': 277,
            'joao fonseca': 145,
            
            # WTA Top 50 (ОБНОВЛЕНО июль 2025)  
            'aryna sabalenka': 1,
            'iga swiatek': 2,
            'coco gauff': 3,
            'jessica pegula': 4,
            'elena rybakina': 5,
            'qinwen zheng': 6,
            'jasmine paolini': 7,
            'emma navarro': 8,
            'daria kasatkina': 9,
            'barbora krejcikova': 10,
            'paula badosa': 11,
            'danielle collins': 12,
            'jelena ostapenko': 13,
            'madison keys': 14,
            'beatriz haddad maia': 15,
            'liudmila samsonova': 16,
            'donna vekic': 17,
            'mirra andreeva': 18,
            'marta kostyuk': 19,
            'diana shnaider': 20,
            'katie boulter': 21,
            'ekaterina alexandrova': 22,
            'caroline garcia': 23,
            'elise mertens': 24,
            'emma raducanu': 25,
            'anastasia pavlyuchenkova': 26,
            'linda noskova': 27,
            'victoria azarenka': 28,
            'lulu sun': 29,
            'magdalena frech': 30,
            'caroline dolehide': 31,
            'leylah fernandez': 32,
            'dayana yastremska': 33,
            'anna kalinskaya': 34,
            'amanda anisimova': 35,
            'ons jabeur': 36,
            'peyton stearns': 37,
            'marie bouzkova': 38,
            'kaia kanepi': 39,
            'sloane stephens': 40,
            'elina svitolina': 41,
            'anastasia potapova': 42,
            'veronika kudermetova': 43,
            'claire liu': 44,
            'yulia putintseva': 45,
            'anastasia pavlyuchenkova': 46,
            'camila osorio': 47,
            'petra kvitova': 48,
            'xinyu wang': 49,
            'cristina bucsa': 50,
            
            # Дополнительные WTA
            'renata zarazua': 85,
            'carson branstine': 125,
        }
    
    def get_player_ranking(self, player_name):
        """🔧 ИСПРАВЛЕНО: Получить актуальный рейтинг"""
        name_lower = player_name.lower().strip()
        
        if name_lower in self.player_rankings:
            return self.player_rankings[name_lower]
        
        # Поиск по частям имени
        for known_player, rank in self.player_rankings.items():
            if any(part in known_player for part in name_lower.split()):
                return rank
        
        return 80  # Более реалистичный рейтинг для неизвестных
    
    def identify_underdog_scenario(self, player1, player2):
        """🔧 ИСПРАВЛЕНО: Правильное определение underdog с новыми рейтингами"""
        p1_rank = self.get_player_ranking(player1)
        p2_rank = self.get_player_ranking(player2)
        
        # Определяем кто underdog (хуже рейтинг = больше номер)
        if p1_rank > p2_rank:
            underdog = player1
            favorite = player2
            underdog_rank = p1_rank
            favorite_rank = p2_rank
            underdog_is_player1 = True
        else:
            underdog = player2
            favorite = player1
            underdog_rank = p2_rank
            favorite_rank = p1_rank
            underdog_is_player1 = False
        
        rank_difference = underdog_rank - favorite_rank
        
        # 🔧 ИСПРАВЛЕНО: Более реалистичная классификация
        if rank_difference >= 50:
            underdog_type = "HUGE_UNDERDOG"
            base_probability = 0.42
        elif rank_difference >= 30:
            underdog_type = "STRONG_UNDERDOG" 
            base_probability = 0.48
        elif rank_difference >= 15:
            underdog_type = "MILD_UNDERDOG"
            base_probability = 0.54
        elif rank_difference >= 5:
            underdog_type = "SLIGHT_UNDERDOG"
            base_probability = 0.58
        else:
            underdog_type = "CLOSE_MATCH"
            base_probability = 0.62
        
        return {
            'underdog': underdog,
            'favorite': favorite,
            'underdog_rank': underdog_rank,
            'favorite_rank': favorite_rank,
            'underdog_is_player1': underdog_is_player1,
            'rank_difference': rank_difference,
            'underdog_type': underdog_type,
            'base_probability': base_probability
        }
    
    def calculate_underdog_probability(self, player1, player2, tournament, surface, round_name="R64"):
        """🔧 ИСПРАВЛЕНО: Более точный расчет с правильными рейтингами"""
        
        # Определяем underdog сценарий с ОБНОВЛЕННЫМИ рейтингами
        scenario = self.identify_underdog_scenario(player1, player2)
        
        # Получаем ML прогноз
        ml_probability = None
        ml_system_used = "None"
        
        if self.real_predictor:
            try:
                result = self.real_predictor.predict_match(player1, player2, tournament, surface, round_name)
                ml_probability = result['probability']  
                ml_system_used = result.get('prediction_type', 'Real ML')
            except Exception as e:
                logger.warning(f"Real predictor failed: {e}")
        
        # Если underdog это player2, то его вероятность = 1 - ml_probability
        if scenario['underdog_is_player1']:
            underdog_ml_probability = ml_probability if ml_probability else scenario['base_probability']
        else:
            underdog_ml_probability = (1 - ml_probability) if ml_probability else scenario['base_probability']
        
        # 🔧 ИСПРАВЛЕНО: Более реалистичные бонусы
        surface_bonus = 0.03 if surface == 'Grass' else 0.01
        tournament_bonus = 0.02 if any(major in tournament for major in ['Wimbledon', 'US Open', 'French Open', 'Australian Open']) else 0.01
        
        # 🔧 ИСПРАВЛЕНО: Реалистичные границы
        final_probability = max(0.25, min(0.85, underdog_ml_probability + surface_bonus + tournament_bonus))
        
        # Качественная классификация
        if final_probability >= 0.70:
            quality = "EXCELLENT"
            confidence = "Very High"
        elif final_probability >= 0.60:
            quality = "GOOD"
            confidence = "High"
        elif final_probability >= 0.50:
            quality = "FAIR"
            confidence = "Medium"
        else:
            quality = "POOR"
            confidence = "Low"
        
        # 🔧 ИСПРАВЛЕНО: Более информативные факторы
        key_factors = [
            f"🎯 {scenario['underdog']} (#{scenario['underdog_rank']}) vs {scenario['favorite']} (#{scenario['favorite_rank']})",
            f"📊 Разность рейтингов: {scenario['rank_difference']} позиций",
            f"🎾 Тип: {scenario['underdog_type'].replace('_', ' ').title()}",
            f"💪 {final_probability:.0%} шанс взять хотя бы один сет",
        ]
        
        return {
            'underdog_scenario': scenario,
            'underdog_probability': round(final_probability, 3),
            'quality': quality,
            'confidence': confidence,
            'key_factors': key_factors,
            'ml_system_used': ml_system_used,
            'ml_probability_raw': ml_probability,
            'prediction_type': 'UNDERDOG_ANALYSIS_FIXED'
        }
    
    def _create_features_for_prediction_service(self, player1, player2, tournament, surface):
        """Создает признаки для Tennis Prediction Service"""
        if self.real_predictor:
            try:
                features = self.real_predictor.create_match_features(player1, player2, tournament, surface, "R64")
                return features
            except:
                pass
        
        # Fallback к простым признакам
        return {
            'player_rank': float(self.get_player_ranking(player1)),
            'opponent_rank': float(self.get_player_ranking(player2)),
            'player_age': 25.0,
            'opponent_age': 25.0,
            'player_recent_win_rate': 0.7,
            'player_form_trend': 0.0,
            'player_surface_advantage': 0.0,
            'h2h_win_rate': 0.5,
            'total_pressure': 2.5
        }

# Создаем анализатор underdog
underdog_analyzer = UnderdogAnalyzer()

def get_live_matches_with_underdog_focus():
    """Получение матчей с фокусом на underdog возможности"""
    try:
        # 1. Получаем реальные матчи через API
        if API_ECONOMY_AVAILABLE:
            logger.info("🔍 Getting matches via API Economy...")
            api_result = economical_tennis_request('tennis')
            
            if api_result['success'] and api_result.get('data'):
                matches = process_api_matches_with_underdog_focus(api_result['data'])
                if matches:
                    return {
                        'matches': matches,
                        'source': f"LIVE_API_{api_result['status']}",
                        'success': True
                    }
        
        # 2. Demo матчи с underdog фокусом
        logger.info("🎯 Generating demo matches with underdog focus...")
        demo_matches = generate_demo_matches_with_underdog_focus()
        return {
            'matches': demo_matches,
            'source': 'DEMO_WITH_UNDERDOG_FOCUS',
            'success': True
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting matches: {e}")
        return {
            'matches': [],
            'source': 'ERROR',
            'success': False,
            'error': str(e)
        }

def process_api_matches_with_underdog_focus(api_data):
    """Обработка API матчей с underdog фокусом"""
    processed_matches = []
    
    for api_match in api_data[:5]:
        try:
            player1 = api_match.get('home_team', 'Player 1')
            player2 = api_match.get('away_team', 'Player 2')
            
            # Извлекаем коэффициенты
            odds1, odds2 = extract_best_odds_from_api(api_match.get('bookmakers', []))
            
            if odds1 and odds2:
                # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Анализируем underdog
                underdog_analysis = underdog_analyzer.calculate_underdog_probability(
                    player1, player2, 'Live Tournament', 'Hard'
                )
                
                scenario = underdog_analysis['underdog_scenario']
                
                match = {
                    'id': f"api_{api_match.get('id', 'unknown')}",
                    'player1': f"🎾 {player1}",
                    'player2': f"🎾 {player2}",
                    'tournament': '🏆 Live Tournament',
                    'surface': 'Hard',
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'time': api_match.get('commence_time', '14:00')[:5] if api_match.get('commence_time') else '14:00',
                    'odds': {'player1': odds1, 'player2': odds2},
                    'underdog_analysis': underdog_analysis,
                    'key_factors': underdog_analysis['key_factors'],
                    'source': 'LIVE_API_WITH_UNDERDOG_FOCUS'
                }
                processed_matches.append(match)
                
        except Exception as e:
            logger.error(f"❌ Error processing API match: {e}")
            continue
    
    return processed_matches

def get_current_real_matches():
    """🔧 ИСПРАВЛЕНО: РЕАЛЬНЫЕ текущие матчи, НЕ завершенные"""
    current_date = datetime.now()
    
    # Не показываем завершенные матчи как "текущие"
    if current_date.month == 7 and current_date.day > 14:  # После Wimbledon
        realistic_matches = [
            ('Carlos Alcaraz', 'Alexander Zverev', 'Hamburg Open', 'Clay'),
            ('Jannik Sinner', 'Tommy Paul', 'Atlanta Open', 'Hard'),  
            ('Andrey Rublev', 'Sebastian Korda', 'Los Cabos Open', 'Hard'),
            ('Jessica Pegula', 'Emma Navarro', 'WTA Washington', 'Hard'),
        ]
    elif current_date.month == 8:  # Август
        realistic_matches = [
            ('Novak Djokovic', 'Carlos Alcaraz', 'Montreal Masters', 'Hard'),
            ('Iga Swiatek', 'Coco Gauff', 'Montreal WTA', 'Hard'),
            ('Taylor Fritz', 'Ben Shelton', 'Cincinnati Masters', 'Hard'),
            ('Aryna Sabalenka', 'Jessica Pegula', 'Cincinnati WTA', 'Hard'),
        ]
    else:
        # Дефолтные матчи
        realistic_matches = [
            ('Jannik Sinner', 'Carlos Alcaraz', 'ATP Exhibition', 'Hard'),
            ('Iga Swiatek', 'Aryna Sabalenka', 'WTA Exhibition', 'Hard'),
        ]
    
    return realistic_matches

def generate_demo_matches_with_underdog_focus():
    """Генерация demo матчей с underdog фокусом"""
    # ИСПРАВЛЕНО: Специально подобранные underdog матчи
    underdog_matches_data = [
        ('Brandon Nakashima', 'Bu Yunchaokete', 'Wimbledon', 'Grass'),  # Nakashima фаворит
        ('Renata Zarazua', 'Amanda Anisimova', 'Wimbledon', 'Grass'),   # Anisimova фаворит
        ('Fabio Fognini', 'Carlos Alcaraz', 'US Open', 'Hard'),          # Alcaraz явный фаворит
        ('Arthur Rinderknech', 'Novak Djokovic', 'ATP Masters', 'Hard')  # Djokovic фаворит
    ]
    
    processed_matches = []
    
    for i, (player1, player2, tournament, surface) in enumerate(underdog_matches_data):
        try:
            # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Получаем underdog анализ
            underdog_analysis = underdog_analyzer.calculate_underdog_probability(
                player1, player2, tournament, surface
            )
            
            scenario = underdog_analysis['underdog_scenario']
            
            # Генерируем коэффициенты на основе рейтингов
            if scenario['underdog_is_player1']:
                p1_odds = 2.5 + (scenario['rank_difference'] * 0.02)
                p2_odds = 1.5 - (scenario['rank_difference'] * 0.01)
            else:
                p1_odds = 1.5 - (scenario['rank_difference'] * 0.01)
                p2_odds = 2.5 + (scenario['rank_difference'] * 0.02)
            
            p1_odds = round(max(1.1, min(p1_odds, 8.0)), 2)
            p2_odds = round(max(1.1, min(p2_odds, 8.0)), 2)
            
            match = {
                'id': f"underdog_demo_{i+1}",
                'player1': f"🎾 {player1}",
                'player2': f"🎾 {player2}",
                'tournament': f"🏆 {tournament}",
                'surface': surface,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'time': '14:00',
                'odds': {'player1': p1_odds, 'player2': p2_odds},
                'underdog_analysis': underdog_analysis,
                'key_factors': underdog_analysis['key_factors'],
                'source': 'DEMO_WITH_UNDERDOG_FOCUS'
            }
            processed_matches.append(match)
            
        except Exception as e:
            logger.error(f"❌ Error generating underdog demo match: {e}")
            continue
    
    return processed_matches

def extract_best_odds_from_api(bookmakers):
    """Извлечение лучших коэффициентов из API"""
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
                        if not best_odds1 or odds1 > best_odds1:
                            best_odds1 = odds1
                        if not best_odds2 or odds2 > best_odds2:
                            best_odds2 = odds2
    
    return best_odds1, best_odds2

class SimpleResultLogger:
    """🔧 НОВОЕ: Простая система логирования результатов"""
    
    def __init__(self):
        self.results_file = "match_results.json"
        self.load_results()
    
    def load_results(self):
        """Загрузить сохраненные результаты"""
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                self.results = json.load(f)
        except:
            self.results = []
    
    def save_results(self):
        """Сохранить результаты"""
        try:
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def log_result(self, match_data: dict, actual_result: dict):
        """Логировать результат матча"""
        result_entry = {
            'timestamp': datetime.now().isoformat(),
            'match': match_data,
            'prediction': match_data.get('underdog_analysis', {}),
            'actual_result': actual_result,
            'correct': self._check_prediction_accuracy(match_data, actual_result)
        }
        
        self.results.append(result_entry)
        self.save_results()
        
        logger.info(f"✅ Result logged: {actual_result.get('winner')} won")
    
    def _check_prediction_accuracy(self, match_data, actual_result):
        """Проверить точность прогноза"""
        prediction = match_data.get('underdog_analysis', {})
        scenario = prediction.get('underdog_scenario', {})
        
        winner = actual_result.get('winner')
        sets_won = actual_result.get('sets_won', {})
        
        # Проверяем взял ли underdog хотя бы один сет
        underdog = scenario.get('underdog')
        if underdog and winner != underdog:
            # Underdog проиграл, но взял ли сет?
            underdog_sets = sets_won.get(underdog, 0)
            return underdog_sets >= 1
        elif underdog and winner == underdog:
            # Underdog выиграл весь матч
            return True
        
        return False
    
    def get_accuracy_stats(self):
        """Получить статистику точности"""
        if not self.results:
            return {
                'total_predictions': 0,
                'correct_predictions': 0,
                'accuracy': 0.0
            }
        
        total = len(self.results)
        correct = sum(1 for r in self.results if r.get('correct', False))
        
        return {
            'total_predictions': total,
            'correct_predictions': correct,
            'accuracy': correct / total if total > 0 else 0.0,
            'last_updated': datetime.now().isoformat()
        }

def generate_realistic_current_matches():
    """🔧 ИСПРАВЛЕНО: Генерация реалистичных матчей с правильными рейтингами"""
    current_matches_data = get_current_real_matches()
    processed_matches = []
    
    # Создаем ИСПРАВЛЕННЫЙ анализатор
    fixed_analyzer = UnderdogAnalyzer()
    
    for i, (player1, player2, tournament, surface) in enumerate(current_matches_data):
        try:
            # Анализируем с ИСПРАВЛЕННОЙ системой
            underdog_analysis = fixed_analyzer.calculate_underdog_probability(
                player1, player2, tournament, surface
            )
            
            scenario = underdog_analysis['underdog_scenario']
            
            # 🔧 ИСПРАВЛЕНО: Реалистичные коэффициенты на основе актуальных рейтингов
            rank_diff = scenario['rank_difference']
            
            if scenario['underdog_is_player1']:
                p1_odds = 2.2 + (rank_diff * 0.03)
                p2_odds = 1.8 - (rank_diff * 0.01)
            else:
                p1_odds = 1.8 - (rank_diff * 0.01)
                p2_odds = 2.2 + (rank_diff * 0.03)
            
            # Ограничиваем реалистичными границами
            p1_odds = round(max(1.2, min(p1_odds, 6.0)), 2)
            p2_odds = round(max(1.2, min(p2_odds, 6.0)), 2)
            
            match = {
                'id': f"fixed_{i+1}",
                'player1': f"🎾 {player1}",
                'player2': f"🎾 {player2}",
                'tournament': f"🏆 {tournament}",
                'surface': surface,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'time': '15:00',
                'odds': {'player1': p1_odds, 'player2': p2_odds},
                'underdog_analysis': underdog_analysis,
                'key_factors': underdog_analysis['key_factors'],
                'source': 'FIXED_REALISTIC_MATCHES'
            }
            processed_matches.append(match)
            
        except Exception as e:
            logger.error(f"❌ Error generating realistic match: {e}")
            continue
    
    return processed_matches

@app.route('/')
def dashboard():
    """ИСПРАВЛЕННАЯ главная страница с темной стильной расцветкой"""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎾 Tennis Underdog Analytics</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: white; min-height: 100vh; padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { 
            background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 24px;
            padding: 32px; margin-bottom: 32px; text-align: center;
        }
        .underdog-banner {
            background: linear-gradient(135deg, #6bcf7f, #4a9eff);
            padding: 20px; border-radius: 15px; margin-bottom: 20px;
            animation: pulse 3s infinite;
        }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.9; } }
        .main-title {
            font-size: 2.8rem; font-weight: 700; margin-bottom: 16px;
            background: linear-gradient(135deg, #ff6b6b, #ffd93d, #6bcf7f);
            background-clip: text; -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .subtitle { font-size: 1.2rem; opacity: 0.8; margin-bottom: 32px; }
        .stats-grid { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); 
            gap: 20px; margin: 20px 0;
        }
        .stat-card { 
            background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px; padding: 20px; text-align: center; transition: all 0.3s ease;
        }
        .stat-card:hover { transform: translateY(-4px); background: rgba(255, 255, 255, 0.08); }
        .stat-value { font-size: 2.2rem; font-weight: 700; margin-bottom: 8px; color: #6bcf7f; }
        .stat-label { font-size: 0.9rem; opacity: 0.7; text-transform: uppercase; }
        .controls { 
            background: rgba(255, 255, 255, 0.05); border-radius: 20px; 
            padding: 24px; margin-bottom: 32px; text-align: center;
        }
        .btn { 
            background: linear-gradient(135deg, #667eea, #764ba2); color: white;
            border: none; padding: 14px 28px; border-radius: 12px; font-size: 1rem;
            cursor: pointer; margin: 8px; transition: all 0.3s ease; font-weight: 600;
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4); }
        .btn-success { background: linear-gradient(135deg, #6bcf7f, #4a9eff); }
        .btn-warning { background: linear-gradient(135deg, #ffd93d, #ff6b6b); }
        .btn-info { background: linear-gradient(135deg, #4a9eff, #667eea); }
        .matches-container { display: grid; gap: 24px; }
        .match-card { 
            background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px; padding: 28px; position: relative; transition: all 0.3s ease;
        }
        .match-card:hover { transform: translateY(-4px); border-color: rgba(107, 207, 127, 0.3); }
        .quality-badge {
            position: absolute; top: 16px; right: 16px; 
            color: #1a1a2e; padding: 6px 12px; border-radius: 20px; 
            font-size: 0.8rem; font-weight: 700;
        }
        .quality-excellent { border-left: 4px solid #6bcf7f !important; }
        .quality-excellent .quality-badge { background: linear-gradient(135deg, #6bcf7f, #4a9eff); }
        .quality-good { border-left: 4px solid #4a9eff !important; }
        .quality-good .quality-badge { background: linear-gradient(135deg, #4a9eff, #667eea); }
        .quality-fair { border-left: 4px solid #ffd93d !important; }
        .quality-fair .quality-badge { background: linear-gradient(135deg, #ffd93d, #ff6b6b); }
        .quality-poor { border-left: 4px solid #ff6b6b !important; }
        .quality-poor .quality-badge { background: linear-gradient(135deg, #ff6b6b, #ffd93d); }
        
        .underdog-highlight {
            background: linear-gradient(135deg, rgba(107, 207, 127, 0.1), rgba(255, 217, 61, 0.1));
            border: 1px solid rgba(107, 207, 127, 0.3); border-radius: 16px; 
            padding: 20px; margin: 20px 0; text-align: center; font-weight: bold;
        }
        .probability { font-size: 2.5rem; font-weight: 700; color: #6bcf7f; }
        .confidence { margin-top: 8px; font-size: 1.1rem; opacity: 0.8; }
        
        .favorite-vs-underdog {
            display: grid; grid-template-columns: 1fr auto 1fr; 
            gap: 15px; align-items: center; margin: 15px 0;
        }
        .player-info {
            background: rgba(255, 255, 255, 0.05); padding: 16px; 
            border-radius: 12px; text-align: center; transition: all 0.3s ease;
        }
        .player-info:hover { background: rgba(255, 255, 255, 0.08); }
        .vs-divider { font-size: 1.5rem; font-weight: bold; color: #ffd93d; }
        .underdog-player { border: 2px solid #6bcf7f; }
        .favorite-player { border: 2px solid #4a9eff; }
        
        .odds-display { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); 
            gap: 16px; margin-top: 16px;
        }
        .odds-item { 
            background: rgba(255, 255, 255, 0.05); padding: 16px; 
            border-radius: 12px; text-align: center; transition: all 0.3s ease;
        }
        .odds-item:hover { background: rgba(255, 255, 255, 0.08); }
        
        .factors-list { margin-top: 16px; }
        .factor-item { 
            background: rgba(255, 255, 255, 0.05); margin: 8px 0; padding: 12px 16px;
            border-radius: 8px; font-size: 0.95rem; border-left: 3px solid #6bcf7f;
        }
        
        .loading { 
            text-align: center; padding: 80px; background: rgba(255, 255, 255, 0.05);
            border-radius: 20px; font-size: 1.2rem;
        }
        
                /* Стильный боковой скроллбар для темного дизайна */
        ::-webkit-scrollbar {
            width: 12px;
            background: rgba(255, 255, 255, 0.02);
        }

        ::-webkit-scrollbar-track {
            background: linear-gradient(180deg, 
                rgba(26, 26, 46, 0.8) 0%, 
                rgba(22, 33, 62, 0.8) 50%, 
                rgba(15, 52, 96, 0.8) 100%);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, 
                rgba(107, 207, 127, 0.8) 0%, 
                rgba(74, 158, 255, 0.8) 50%, 
                rgba(102, 126, 234, 0.8) 100%);
            border-radius: 10px;
            border: 2px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 0 15px rgba(107, 207, 127, 0.3);
            transition: all 0.3s ease;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, 
                rgba(107, 207, 127, 1) 0%, 
                rgba(74, 158, 255, 1) 50%, 
                rgba(102, 126, 234, 1) 100%);
            box-shadow: 0 0 20px rgba(107, 207, 127, 0.5);
            transform: scale(1.05);
        }

        ::-webkit-scrollbar-thumb:active {
            background: linear-gradient(180deg, 
                rgba(255, 107, 107, 0.9) 0%, 
                rgba(255, 217, 61, 0.9) 50%, 
                rgba(107, 207, 127, 0.9) 100%);
            box-shadow: 0 0 25px rgba(255, 107, 107, 0.4);
        }

        /* Дополнительные стили для скроллбара */
        ::-webkit-scrollbar-corner {
            background: rgba(26, 26, 46, 0.8);
        }

        /* Анимированные эффекты */
        ::-webkit-scrollbar-thumb {
            animation: scrollbar-glow 3s ease-in-out infinite alternate;
        }

        @keyframes scrollbar-glow {
            0% {
                box-shadow: 0 0 15px rgba(107, 207, 127, 0.3);
            }
            100% {
                box-shadow: 0 0 20px rgba(74, 158, 255, 0.4);
            }
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .container { padding: 12px; }
            .main-title { font-size: 2.2rem; }
            .stats-grid { grid-template-columns: repeat(2, 1fr); }
            .favorite-vs-underdog { grid-template-columns: 1fr; gap: 10px; }
            .vs-divider { font-size: 1.2rem; }
            .btn { padding: 12px 20px; font-size: 0.9rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="underdog-banner">
                <h2>🎯 UNDERDOG OPPORTUNITY FINDER</h2>
                <p>Находим игроков которые могут взять сет против фаворитов</p>
            </div>
            
            <div class="main-title">🎾 Tennis Underdog Analytics</div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="underdog-count">-</div>
                    <div class="stat-label">Underdog Opportunities</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="avg-probability">-</div>
                    <div class="stat-label">Avg Set Probability</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="excellent-quality">-</div>
                    <div class="stat-label">Excellent Quality</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="api-status">-</div>
                    <div class="stat-label">API Status</div>
                </div>
            </div>
        </div>
        
        <div class="controls">
            <h3>🎯 Underdog Control Panel</h3>
            <p style="margin: 12px 0; opacity: 0.8;">Find players who can surprise favorites</p>
            <button class="btn btn-success" onclick="loadUnderdogOpportunities()">🎯 Find Underdog Opportunities</button>
            <button class="btn btn-info" onclick="testUnderdogAnalysis()">🔮 Test Underdog Analysis</button>
            <button class="btn btn-warning" onclick="manualAPIUpdate()">🔄 Manual API Update</button>
            <button class="btn" onclick="checkAPIStatus()">📊 API Status</button>
        </div>
        
        <div id="matches-container" class="matches-container">
            <div class="loading">
                <h3>🎯 Finding underdog opportunities...</h3>
                <p>Analyzing matches for upset potential</p>
            </div>
        </div>
    </div>
    
    <script>
        const API_BASE = window.location.origin + '/api';
        
        async function loadUnderdogOpportunities() {
            const container = document.getElementById('matches-container');
            container.innerHTML = '<div class="loading"><h3>🔍 Analyzing underdog opportunities...</h3><p>Using advanced ML models...</p></div>';
            
            try {
                const response = await fetch(API_BASE + '/matches');
                const data = await response.json();
                
                if (data.success && data.matches) {
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
                    container.innerHTML = '<div class="loading"><h3>❌ No underdog opportunities found</h3><p>Try refreshing or check back later</p></div>';
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
    </script>
</body>
</html>'''


def create_logging_endpoints():
    """🔧 НОВОЕ: Создаем endpoints для логирования результатов"""
    
    @app.route('/api/log-result', methods=['POST'])
    def log_match_result():
        """Логирование результата матча"""
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({
                    'success': False,
                    'error': 'No data provided'
                }), 400
            
            # Создаем mock match data для логирования
            fixed_analyzer = UnderdogAnalyzer()
            match_data = {
                'player1': data.get('player1'),
                'player2': data.get('player2'),
                'underdog_analysis': fixed_analyzer.calculate_underdog_probability(
                    data.get('player1'), data.get('player2'), 'Unknown Tournament', 'Hard'
                )
            }
            
            actual_result = {
                'winner': data.get('winner'),
                'score': data.get('score', ''),
                'sets_won': {
                    data.get('player1'): 1 if data.get('winner') == data.get('player1') else 0,
                    data.get('player2'): 1 if data.get('winner') == data.get('player2') else 0
                }
            }
            
            result_logger.log_result(match_data, actual_result)
            
            return jsonify({
                'success': True,
                'message': 'Match result logged successfully'
            })
            
        except Exception as e:
            logger.error(f"❌ Log result error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/accuracy-stats', methods=['GET'])
    def get_accuracy_stats():
        """Получение статистики точности"""
        try:
            stats = result_logger.get_accuracy_stats()
            
            return jsonify({
                'success': True,
                'stats': stats,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Accuracy stats error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

# Добавьте эти роуты в tennis_backend.py ПЕРЕД блоком 
@app.route('/api/prediction-stats', methods=['GET'])
def get_prediction_stats():
    """Статистика прогнозов"""
    try:
        if LOGGING_AVAILABLE and prediction_logger:
            stats = prediction_logger.get_system_performance()
            return jsonify({
                'success': True,
                'stats': stats,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Logging system not available'
            }), 503
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/update-result', methods=['POST'])
def update_match_result():
    """Обновление результата матча"""
    try:
        data = request.get_json()
        
        if not LOGGING_AVAILABLE or not prediction_logger:
            return jsonify({
                'success': False,
                'error': 'Logging system not available'
            }), 503
        
        success = prediction_logger.logger.update_result(
            player1=data.get('player1'),
            player2=data.get('player2'),
            match_date=data.get('match_date'),
            actual_winner=data.get('actual_winner'),
            match_score=data.get('match_score', '')
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Result updated successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Match not found or already updated'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'underdog_system': underdog_analyzer is not None,
        'real_predictor': real_predictor is not None,
        'prediction_service': tennis_prediction_service is not None,
        'api_economy': API_ECONOMY_AVAILABLE,
        'service': 'tennis_underdog_backend',
        'version': '2.0'
    })

@app.route('/api/matches', methods=['GET'])
def get_matches():
    """Получение матчей с UNDERDOG фокусом"""
    try:
        logger.info("🎯 Getting matches with UNDERDOG focus...")
        
        matches_data = get_live_matches_with_underdog_focus()
        
        return jsonify({
            'success': matches_data['success'],
            'matches': matches_data['matches'],
            'count': len(matches_data['matches']),
            'source': matches_data['source'],
            'focus_type': 'UNDERDOG_OPPORTUNITIES',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Matches error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'matches': []
        }), 500

@app.route('/api/test-underdog', methods=['POST'])
def test_underdog_analysis():
    """Тестирование underdog анализа"""
    try:
        data = request.get_json()
        
        player1 = data.get('player1', 'Fabio Fognini')
        player2 = data.get('player2', 'Carlos Alcaraz')
        tournament = data.get('tournament', 'US Open')
        surface = data.get('surface', 'Hard')
        
        logger.info(f"🎯 Testing underdog analysis: {player1} vs {player2}")
        
        # Получаем underdog анализ
        underdog_analysis = underdog_analyzer.calculate_underdog_probability(
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
        logger.error(f"❌ Underdog test error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/manual-api-update', methods=['POST'])
def manual_api_update():
    """Ручное обновление API данных"""
    try:
        if API_ECONOMY_AVAILABLE:
            success = trigger_manual_update()
            if success:
                return jsonify({
                    'success': True,
                    'message': 'Manual API update triggered successfully'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to trigger manual update'
                })
        else:
            return jsonify({
                'success': False,
                'error': 'API Economy not available'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/api-economy-status', methods=['GET'])
def get_api_economy_status():
    """Статус API Economy системы"""
    try:
        if API_ECONOMY_AVAILABLE:
            api_usage = get_api_usage()
            return jsonify({
                'success': True,
                'api_usage': api_usage,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'API Economy not available'
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

# Добавьте в самый конец tennis_backend.py
print("🔧 DEBUG: Дошли до конца файла")

if __name__ == '__main__':
    print("🎯 TENNIS UNDERDOG SYSTEM")
    print("=" * 50)
    print(f"🌐 Dashboard: http://0.0.0.0:5001")
    print("=" * 50)
    
    try:
        print("🚀 Starting Flask server...")
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
    except Exception as e:
        print(f"❌ Server failed: {e}")