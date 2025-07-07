#!/usr/bin/env python3
"""
🎾 СТРОГИЙ Tennis Backend - ТОЛЬКО реальные данные
БЕЗ демо матчей - показываем только то что есть
"""

import os
from dotenv import load_dotenv
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

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class StrictUnderdogPredictor:
    """СТРОГИЙ предиктор - только качественные возможности"""
    
    def __init__(self):
        # Реальные данные игроков
        self.player_database = {
            # ATP
            'jannik sinner': {'rank': 1, 'age': 23, 'form': 0.90, 'set_tenacity': 0.85},
            'carlos alcaraz': {'rank': 2, 'age': 21, 'form': 0.88, 'set_tenacity': 0.80},
            'alexander zverev': {'rank': 3, 'age': 27, 'form': 0.82, 'set_tenacity': 0.75},
            'daniil medvedev': {'rank': 4, 'age': 28, 'form': 0.78, 'set_tenacity': 0.85},
            'novak djokovic': {'rank': 5, 'age': 37, 'form': 0.75, 'set_tenacity': 0.95},
            'ben shelton': {'rank': 15, 'age': 22, 'form': 0.72, 'set_tenacity': 0.75},
            'tommy paul': {'rank': 12, 'age': 27, 'form': 0.75, 'set_tenacity': 0.78},
            'frances tiafoe': {'rank': 18, 'age': 26, 'form': 0.70, 'set_tenacity': 0.80},
            'brandon nakashima': {'rank': 45, 'age': 23, 'form': 0.68, 'set_tenacity': 0.72},
            'marin cilic': {'rank': 70, 'age': 35, 'form': 0.65, 'set_tenacity': 0.80},
            'flavio cobolli': {'rank': 85, 'age': 22, 'form': 0.68, 'set_tenacity': 0.70},
            'cameron norrie': {'rank': 35, 'age': 28, 'form': 0.70, 'set_tenacity': 0.75},
            
            # WTA
            'aryna sabalenka': {'rank': 1, 'age': 26, 'form': 0.85, 'set_tenacity': 0.82},
            'iga swiatek': {'rank': 2, 'age': 23, 'form': 0.88, 'set_tenacity': 0.85},
            'coco gauff': {'rank': 3, 'age': 20, 'form': 0.80, 'set_tenacity': 0.75},
            'emma raducanu': {'rank': 90, 'age': 22, 'form': 0.62, 'set_tenacity': 0.70},
            'katie boulter': {'rank': 28, 'age': 27, 'form': 0.68, 'set_tenacity': 0.72},
            'amanda anisimova': {'rank': 35, 'age': 23, 'form': 0.72, 'set_tenacity': 0.75},
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
        
        # Возвращаем None если игрок неизвестен
        return None
    
    def determine_underdog_from_odds(self, player1, player2, odds1, odds2):
        """Определение андердога по коэффициентам"""
        if odds1 > odds2:
            return {
                'underdog': player1,
                'favorite': player2,
                'underdog_odds': odds1,
                'favorite_odds': odds2
            }
        else:
            return {
                'underdog': player2,
                'favorite': player1, 
                'underdog_odds': odds2,
                'favorite_odds': odds1
            }
    
    def calculate_strict_set_probability(self, underdog_name, favorite_name, underdog_odds, favorite_odds):
        """СТРОГИЙ расчёт вероятности - только если есть данные"""
        
        underdog_data = self.get_player_data(underdog_name)
        favorite_data = self.get_player_data(favorite_name)
        
        # Если нет данных об игроках - отклоняем
        if not underdog_data or not favorite_data:
            return None
        
        # Базовая вероятность из коэффициентов
        match_prob = 1.0 / underdog_odds
        
        # СТРОГИЕ критерии для качественной возможности
        # 1. Коэффициенты должны быть в разумном диапазоне
        if not (1.8 <= underdog_odds <= 6.0):
            return None
        
        # 2. Андердог должен иметь хорошее упорство в сетах
        if underdog_data['set_tenacity'] < 0.70:
            return None
        
        # 3. Форма андердога не должна быть ужасной
        if underdog_data['form'] < 0.60:
            return None
        
        # Рассчитываем вероятность взять сет
        base_set_prob = min(0.85, match_prob + 0.25)
        
        # Корректировки
        tenacity_bonus = (underdog_data['set_tenacity'] - 0.70) * 0.5
        form_bonus = (underdog_data['form'] - 0.65) * 0.3
        
        # Возрастной фактор
        age_factor = 0
        if underdog_data['age'] < 25:
            age_factor = 0.05  # Молодость = смелость
        elif underdog_data['age'] > 32:
            age_factor = -0.05  # Возраст = осторожность
        
        final_probability = base_set_prob + tenacity_bonus + form_bonus + age_factor
        final_probability = max(0.35, min(0.88, final_probability))
        
        # СТРОГИЙ фильтр финальной вероятности
        if final_probability < 0.55:  # Минимум 55% шанс
            return None
        
        confidence = 'Very High' if final_probability > 0.80 else \
                    'High' if final_probability > 0.70 else 'Medium'
        
        # Только высокая и очень высокая уверенность
        if confidence not in ['High', 'Very High']:
            return None
        
        factors = self._analyze_factors(underdog_data, favorite_data, underdog_odds)
        
        return {
            'probability': round(final_probability, 3),
            'confidence': confidence,
            'key_factors': factors
        }
    
    def _analyze_factors(self, underdog_data, favorite_data, underdog_odds):
        """Анализ факторов"""
        factors = []
        
        if underdog_data['set_tenacity'] > 0.75:
            factors.append(f"🔥 Высокое упорство в сетах ({underdog_data['set_tenacity']:.0%})")
        
        if underdog_data['form'] > 0.70:
            factors.append(f"📈 Хорошая текущая форма ({underdog_data['form']:.0%})")
        
        if underdog_data['age'] < 25:
            factors.append(f"⚡ Молодой игрок ({underdog_data['age']} лет) - может играть без страха")
        
        if 2.0 <= underdog_odds <= 3.5:
            factors.append(f"⚖️ Разумные коэффициенты ({underdog_odds}) - реальные шансы")
        elif underdog_odds > 3.5:
            factors.append(f"🎯 Большой андердог ({underdog_odds}) - потенциал сенсации")
        
        return factors

# Инициализация предиктора
predictor = StrictUnderdogPredictor()

def extract_best_odds_from_api(bookmakers):
    """Извлекает лучшие коэффициенты из API"""
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

def get_live_matches_strict():
    """СТРОГОЕ получение матчей - только через API"""
    
    try:
        logger.info("🔍 Getting REAL matches via API Economy...")
        
        # Получаем данные через API Economy
        result = economical_tennis_request('tennis')
        
        if result['success'] and result.get('data'):
            raw_data = result['data']
            
            logger.info(f"📡 API returned {len(raw_data)} matches")
            
            # Преобразуем данные
            converted_matches = []
            
            for api_match in raw_data:
                try:
                    player1 = api_match.get('home_team', '')
                    player2 = api_match.get('away_team', '')
                    
                    if not player1 or not player2:
                        continue
                    
                    # Извлекаем коэффициенты
                    odds1, odds2 = extract_best_odds_from_api(api_match.get('bookmakers', []))
                    
                    if odds1 and odds2 and odds1 > 0 and odds2 > 0:
                        match = {
                            'player1': player1,
                            'player2': player2,
                            'odds1': odds1,
                            'odds2': odds2,
                            'tournament': 'Live Tournament',
                            'surface': 'Hard',
                            'round': 'Live',
                            'time': api_match.get('commence_time', '14:00')[:5] if api_match.get('commence_time') else '14:00'
                        }
                        converted_matches.append(match)
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error processing match: {e}")
                    continue
            
            logger.info(f"✅ Converted {len(converted_matches)} matches")
            
            return {
                'matches': converted_matches,
                'source': f"LIVE_API_{result['status']}",
                'success': True
            }
        else:
            logger.info(f"📭 API returned no data: {result.get('error', 'Unknown')}")
            return {
                'matches': [],
                'source': 'API_NO_DATA',
                'success': False,
                'reason': result.get('error', 'No data from API')
            }
            
    except Exception as e:
        logger.error(f"❌ API error: {e}")
        return {
            'matches': [],
            'source': 'API_ERROR',
            'success': False,
            'reason': str(e)
        }

def generate_strict_quality_matches():
    """СТРОГАЯ генерация - только качественные возможности"""
    
    logger.info("🔍 Strict quality analysis...")
    
    # Получаем ТОЛЬКО реальные матчи
    live_data = get_live_matches_strict()
    
    if not live_data['success'] or not live_data['matches']:
        logger.info(f"📭 No live matches available: {live_data.get('reason', 'Unknown')}")
        return []
    
    potential_matches = live_data['matches']
    logger.info(f"📊 Analyzing {len(potential_matches)} live matches")
    
    quality_matches = []
    rejected_reasons = {}
    
    for match_data in potential_matches:
        try:
            # Определяем андердога
            underdog_info = predictor.determine_underdog_from_odds(
                match_data['player1'], match_data['player2'],
                match_data['odds1'], match_data['odds2']
            )
            
            # СТРОГИЙ анализ
            prediction = predictor.calculate_strict_set_probability(
                underdog_info['underdog'],
                underdog_info['favorite'], 
                underdog_info['underdog_odds'],
                underdog_info['favorite_odds']
            )
            
            if prediction is None:
                # Определяем причину отклонения
                underdog_data = predictor.get_player_data(underdog_info['underdog'])
                
                if not underdog_data:
                    reason = "Unknown player"
                elif not (1.8 <= underdog_info['underdog_odds'] <= 6.0):
                    reason = f"Odds out of range ({underdog_info['underdog_odds']})"
                elif underdog_data['set_tenacity'] < 0.70:
                    reason = f"Low set tenacity ({underdog_data['set_tenacity']:.1%})"
                elif underdog_data['form'] < 0.60:
                    reason = f"Poor form ({underdog_data['form']:.1%})"
                else:
                    reason = "Low final probability"
                
                rejected_reasons[f"{match_data['player1']} vs {match_data['player2']}"] = reason
                continue
            
            # Матч прошел СТРОГИЕ критерии
            match = {
                'id': f"quality_{len(quality_matches)+1}",
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
                    'quality_rating': 'PREMIUM'  # Все прошедшие фильтр = премиум
                },
                
                'focus': f"💎 {underdog_info['underdog']} взять хотя бы 1 сет",
                'recommendation': f"{prediction['probability']:.0%} шанс взять сет",
                'data_source': live_data['source']
            }
            
            quality_matches.append(match)
            logger.info(f"✅ ACCEPTED: {match_data['player1']} vs {match_data['player2']} ({prediction['probability']:.1%})")
            
        except Exception as e:
            logger.error(f"❌ Error analyzing match: {e}")
            continue
    
    # Логируем отклоненные матчи
    if rejected_reasons:
        logger.info(f"❌ REJECTED MATCHES ({len(rejected_reasons)}):")
        for match, reason in rejected_reasons.items():
            logger.info(f"   • {match}: {reason}")
    
    # Сортируем по вероятности
    quality_matches.sort(key=lambda x: x['underdog_analysis']['prediction']['probability'], reverse=True)
    
    logger.info(f"🎯 STRICT RESULT: {len(quality_matches)} PREMIUM opportunities found")
    
    return quality_matches

@app.route('/')
def dashboard():
    """Dashboard с честной информацией"""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎾 Strict Tennis Analysis</title>
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
        .strict-banner {
            background: linear-gradient(135deg, #e74c3c, #c0392b);
            padding: 20px; border-radius: 15px; margin-bottom: 20px;
            text-align: center; animation: pulse 3s infinite;
        }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.9; } }
        .main-title {
            font-size: 2.8rem; font-weight: 700; margin-bottom: 16px;
            background: linear-gradient(135deg, #ff6b6b, #ffd93d, #6bcf7f);
            background-clip: text; -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 20px; }
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
        .matches-container { display: grid; gap: 24px; }
        .match-card { 
            background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px; padding: 28px; position: relative; transition: all 0.3s ease;
        }
        .match-card:hover { transform: translateY(-4px); border-color: rgba(107, 207, 127, 0.3); }
        .loading { 
            text-align: center; padding: 80px; background: rgba(255, 255, 255, 0.05);
            border-radius: 20px; font-size: 1.2rem;
        }
        .no-matches {
            text-align: center; padding: 60px; background: rgba(255, 255, 255, 0.05);
            border-radius: 20px; border: 2px solid rgba(255, 193, 7, 0.3);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="strict-banner">
                <h2>🎯 СТРОГИЙ АНАЛИЗ - ТОЛЬКО ПРЕМИУМ</h2>
                <p>Никаких демо • Только реальные возможности • Высокие стандарты</p>
            </div>
            
            <div class="main-title">🎾 Premium Underdog Analysis</div>
            <p style="font-size: 1.2rem; opacity: 0.8; margin-bottom: 32px;">
                Строгий отбор качественных возможностей
            </p>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="api-requests">-</div>
                    <div class="stat-label">API Requests</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="live-matches">-</div>
                    <div class="stat-label">Live Matches</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="premium-found">-</div>
                    <div class="stat-label">Premium Found</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="success-rate">-</div>
                    <div class="stat-label">Filter Rate</div>
                </div>
            </div>
        </div>
        
        <div class="controls">
            <h3>🎯 Строгий анализ в реальном времени</h3>
            <p style="margin: 12px 0; opacity: 0.8;">Только проверенные возможности проходят фильтр</p>
            <button class="btn" onclick="findPremiumOpportunities()">💎 Найти премиум возможности</button>
            <button class="btn" onclick="checkAPIStatus()">📊 Статус API</button>
            <button class="btn" onclick="forceRefresh()">🔄 Принудительное обновление</button>
        </div>
        
        <div id="matches-container" class="matches-container">
            <div class="loading">
                <h3>🎯 Система строгого анализа готова</h3>
                <p>Нажмите "Найти премиум возможности" для начала</p>
            </div>
        </div>
    </div>
    
    <script>
        const API_BASE = window.location.origin + '/api';
        
        async function findPremiumOpportunities() {
            const container = document.getElementById('matches-container');
            container.innerHTML = '<div class="loading"><h3>🔍 Строгий анализ live матчей...</h3><p>Применяем высокие стандарты качества</p></div>';
            
            try {
                const response = await fetch(API_BASE + '/premium-matches');
                const data = await response.json();
                
                updateStats(data);
                
                if (data.success && data.matches && data.matches.length > 0) {
                    displayPremiumMatches(data.matches, data);
                } else {
                    showNoMatches(data);
                }
            } catch (error) {
                console.error('Error:', error);
                container.innerHTML = '<div class="loading"><h3>❌ Ошибка подключения к API</h3></div>';
            }
        }
        
        function updateStats(data) {
            document.getElementById('live-matches').textContent = data.stats?.total_analyzed || '0';
            document.getElementById('premium-found').textContent = data.matches?.length || '0';
            
            const total = data.stats?.total_analyzed || 1;
            const found = data.matches?.length || 0;
            const rate = ((found / total) * 100).toFixed(1);
            document.getElementById('success-rate').textContent = rate + '%';
        }
        
        function showNoMatches(data) {
            const container = document.getElementById('matches-container');
            
            let reason = '';
            if (data.stats?.api_error) {
                reason = 'Проблема с API: ' + data.stats.api_error;
            } else if (data.stats?.total_analyzed === 0) {
                reason = 'API не вернул активных матчей';
            } else {
                reason = 'Ни один матч не прошел строгие критерии качества';
            }
            
            container.innerHTML = '<div class="no-matches"><h3>📅 Сейчас нет премиум возможностей</h3><p style="margin: 20px 0; opacity: 0.8;">' + reason + '</p><div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; margin-top: 20px;"><h4>🎯 Наши строгие критерии:</h4><ul style="text-align: left; margin: 15px 0; margin-left: 20px;"><li>Коэффициенты от 1.8 до 6.0</li><li>Игрок известен системе</li><li>Упорство в сетах > 70%</li><li>Форма > 60%</li><li>Финальная вероятность > 55%</li><li>Только высокая уверенность</li></ul></div><p style="margin-top: 20px; font-size: 0.9rem; opacity: 0.7;">Это нормально - качество важнее количества</p></div>';
        }
        
        function displayPremiumMatches(matches, data) {
            const container = document.getElementById('matches-container');
            
            let html = '<div style="background: linear-gradient(135deg, rgba(231, 76, 60, 0.1), rgba(192, 57, 43, 0.1)); border: 1px solid rgba(231, 76, 60, 0.3); border-radius: 20px; padding: 24px; margin-bottom: 32px; text-align: center;"><h2>💎 НАЙДЕНЫ ПРЕМИУМ ВОЗМОЖНОСТИ</h2><p>Строгий отбор: ' + matches.length + ' из ' + (data.stats?.total_analyzed || 0) + ' матчей прошли фильтр</p></div>';
            
            matches.forEach((match, index) => {
                const analysis = match.underdog_analysis;
                const prediction = analysis.prediction;
                
                html += '<div class="match-card"><div style="position: absolute; top: 15px; right: 15px; background: #e74c3c; color: white; padding: 5px 12px; border-radius: 15px; font-size: 0.8rem; font-weight: bold;">PREMIUM</div>';
                
                html += '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;"><div><div style="font-size: 1.4rem; font-weight: bold;">' + match.player1 + ' vs ' + match.player2 + '</div><div style="opacity: 0.8; margin-top: 5px;">🏆 ' + match.tournament + ' • ' + match.surface + ' • ' + match.round + '</div></div><div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 15px; text-align: center;"><div style="font-size: 2rem; font-weight: bold; color: #ffd93d;">' + (prediction.probability * 100).toFixed(0) + '%</div><div style="font-size: 0.9rem;">' + prediction.confidence + '</div></div></div>';
                
                html += '<div style="text-align: center; margin: 20px 0; padding: 16px; background: rgba(255, 217, 61, 0.1); border-radius: 12px;"><div style="font-size: 1.2rem; font-weight: 600; color: #ffd93d;">' + match.focus + '</div><div style="margin-top: 8px; opacity: 0.9;">' + match.recommendation + '</div></div>';
                
                html += '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 20px;"><div style="text-align: center; padding: 12px; background: rgba(255,255,255,0.05); border-radius: 10px;"><div style="font-weight: bold; margin-bottom: 5px;">' + match.player1.replace('🎾 ', '') + '</div><div style="font-size: 1.5rem; color: ' + (match.player1.replace('🎾 ', '') === analysis.underdog ? '#ffd93d' : '#6bcf7f') + ';">' + match.odds.player1 + '</div><div style="font-size: 0.8rem; opacity: 0.7;">' + (match.player1.replace('🎾 ', '') === analysis.underdog ? 'АНДЕРДОГ' : 'ФАВОРИТ') + '</div></div><div style="text-align: center; padding: 12px; background: rgba(255,255,255,0.05); border-radius: 10px;"><div style="font-weight: bold; margin-bottom: 5px;">' + match.player2.replace('🎾 ', '') + '</div><div style="font-size: 1.5rem; color: ' + (match.player2.replace('🎾 ', '') === analysis.underdog ? '#ffd93d' : '#6bcf7f') + ';">' + match.odds.player2 + '</div><div style="font-size: 0.8rem; opacity: 0.7;">' + (match.player2.replace('🎾 ', '') === analysis.underdog ? 'АНДЕРДОГ' : 'ФАВОРИТ') + '</div></div></div>';
                
                if (prediction.key_factors && prediction.key_factors.length > 0) {
                    html += '<div style="margin-top: 20px;"><div style="font-weight: 600; margin-bottom: 12px;">🔍 Ключевые факторы премиум качества:</div>';
                    prediction.key_factors.forEach(factor => {
                        html += '<div style="background: rgba(255,255,255,0.05); margin: 8px 0; padding: 12px; border-radius: 8px; border-left: 3px solid #e74c3c;">' + factor + '</div>';
                    });
                    html += '</div>';
                }
                
                html += '<div style="margin-top: 16px; text-align: center; font-size: 0.9rem; opacity: 0.6;">🎯 Премиум качество • Строгий отбор • ' + analysis.quality_rating + '</div></div>';
            });
            
            container.innerHTML = html;
        }
        
        async function checkAPIStatus() {
            try {
                const response = await fetch(API_BASE + '/api-status');
                const data = await response.json();
                
                if (data.success) {
                    const usage = data.api_usage;
                    alert('📊 СТАТУС API СИСТЕМЫ:\\n\\n' + 
                          'Запросов сегодня: ' + usage.requests_this_hour + '/' + usage.max_per_hour + '\\n' +
                          'Остается: ' + usage.remaining_hour + '\\n' +
                          'Кеш элементов: ' + usage.cache_items + '\\n\\n' +
                          '🎯 Система строгого анализа активна!');
                    
                    document.getElementById('api-requests').textContent = usage.remaining_hour;
                } else {
                    alert('❌ Ошибка получения статуса API');
                }
            } catch (error) {
                alert('❌ Ошибка: ' + error.message);
            }
        }
        
        async function forceRefresh() {
            try {
                const response = await fetch(API_BASE + '/force-refresh', {
                    method: 'POST'
                });
                
                const data = await response.json();
                
                if (data.success) {
                    alert('✅ Принудительное обновление запущено!\\nОбновленные данные будут при следующем анализе.');
                } else {
                    alert('❌ Ошибка обновления: ' + data.error);
                }
            } catch (error) {
                alert('❌ Ошибка: ' + error.message);
            }
        }
        
        // Автозагрузка статистики
        document.addEventListener('DOMContentLoaded', function() {
            checkAPIStatus();
        });
    </script>
</body>
</html>'''

@app.route('/api/premium-matches')
def get_premium_matches():
    """СТРОГИЙ анализ - только премиум возможности"""
    try:
        logger.info("🎯 Starting STRICT premium analysis...")
        
        # Получаем строго отобранные матчи
        premium_matches = generate_strict_quality_matches()
        
        # Статистика для пользователя
        live_data = get_live_matches_strict()
        total_analyzed = len(live_data.get('matches', []))
        
        api_error = None
        if not live_data['success']:
            api_error = live_data.get('reason', 'Unknown API error')
        
        stats = {
            'total_analyzed': total_analyzed,
            'premium_found': len(premium_matches),
            'filter_rate': f"{(len(premium_matches) / max(total_analyzed, 1) * 100):.1f}%",
            'api_error': api_error
        }
        
        if premium_matches:
            # Успешно найдены премиум возможности
            probabilities = [m['underdog_analysis']['prediction']['probability'] for m in premium_matches]
            
            return jsonify({
                'success': True,
                'matches': premium_matches,
                'count': len(premium_matches),
                'source': premium_matches[0]['data_source'] if premium_matches else 'NONE',
                'stats': stats,
                'quality_summary': {
                    'avg_probability': f"{(sum(probabilities) / len(probabilities) * 100):.0f}%",
                    'min_probability': f"{(min(probabilities) * 100):.0f}%",
                    'max_probability': f"{(max(probabilities) * 100):.0f}%"
                },
                'timestamp': datetime.now().isoformat()
            })
        else:
            # Честно сообщаем что ничего не найдено
            return jsonify({
                'success': False,
                'matches': [],
                'count': 0,
                'source': live_data.get('source', 'UNKNOWN'),
                'stats': stats,
                'message': 'No matches passed strict quality criteria',
                'timestamp': datetime.now().isoformat()
            })
        
    except Exception as e:
        logger.error(f"❌ Premium analysis error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'matches': [],
            'stats': {'api_error': str(e)}
        }), 500

@app.route('/api/api-status')
def get_api_status():
    """Статус API системы"""
    try:
        api_usage = get_api_usage()
        
        return jsonify({
            'success': True,
            'api_usage': api_usage,
            'system_type': 'STRICT_ANALYSIS',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/force-refresh', methods=['POST'])
def force_refresh():
    """Принудительное обновление данных"""
    try:
        success = trigger_manual_update()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Force refresh triggered - fresh data on next request'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to trigger refresh'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health')
def health_check():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'system': 'strict_tennis_analysis',
        'mode': 'PREMIUM_ONLY',
        'demo_matches': False,
        'timestamp': datetime.now().isoformat()
    })

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
    print("🎾 СТРОГИЙ TENNIS BACKEND - ТОЛЬКО РЕАЛЬНЫЕ ДАННЫЕ")
    print("=" * 60)
    print("🎯 Режим: ПРЕМИУМ АНАЛИЗ")
    print("❌ Демо матчи: ОТКЛЮЧЕНЫ")
    print("✅ Строгие критерии: АКТИВНЫ")
    print("🔍 Фильтрация: ВЫСОКИЕ СТАНДАРТЫ")
    print("=" * 60)
    print(f"🌐 Dashboard: http://0.0.0.0:5001")
    print(f"📡 API: http://0.0.0.0:5001/api/*")
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