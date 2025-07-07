#!/usr/bin/env python3
"""
🎾 BACKEND TENNIS - ИНТЕГРИРОВАННАЯ ML СИСТЕМА
Полная интеграция всех ML компонентов с ручным управлением API
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

class IntegratedMLPredictor:
    """Интегрированный ML предиктор использующий все доступные системы"""
    
    def __init__(self):
        self.real_predictor = real_predictor
        self.prediction_service = tennis_prediction_service
        
    def predict_match_advanced(self, player1, player2, tournament, surface, round_name="R64"):
        """Продвинутый прогноз использующий все доступные ML системы"""
        predictions = {}
        final_result = None
        
        # 1. Пробуем Real Tennis Predictor (приоритет)
        if self.real_predictor:
            try:
                result = self.real_predictor.predict_match(
                    player1, player2, tournament, surface, round_name
                )
                predictions['real_predictor'] = result
                
                if result['prediction_type'] == 'REAL_ML_MODEL':
                    final_result = result
                    final_result['ml_system_used'] = 'Real ML Models'
                    logger.info(f"🤖 Used Real ML Models: {result['probability']:.1%}")
                else:
                    logger.info(f"🎯 Used Advanced Simulation: {result['probability']:.1%}")
                
            except Exception as e:
                logger.error(f"❌ Real predictor error: {e}")
        
        # 2. Пробуем Tennis Prediction Service (если Real ML недоступен)
        if self.prediction_service and not final_result:
            try:
                # Создаем данные для модели
                match_features = self._create_features_for_prediction_service(
                    player1, player2, tournament, surface
                )
                
                result = self.prediction_service.predict_match(match_features, return_details=True)
                
                # Адаптируем формат ответа
                adapted_result = {
                    'prediction_type': 'TRAINED_ML_MODELS',
                    'probability': result['probability'],
                    'confidence': result['confidence'],
                    'confidence_ru': result.get('confidence_ru', result['confidence']),
                    'key_factors': result.get('key_factors', []),
                    'model_details': result,
                    'ml_system_used': 'Trained ML Ensemble'
                }
                
                predictions['prediction_service'] = adapted_result
                final_result = adapted_result
                
                logger.info(f"🧠 Used Trained ML Models: {result['probability']:.1%}")
                
            except Exception as e:
                logger.error(f"❌ Prediction service error: {e}")
        
        # 3. Fallback к простой логике
        if not final_result:
            final_result = self._fallback_prediction(player1, player2, tournament, surface)
            predictions['fallback'] = final_result
            logger.info(f"⚠️ Used Fallback prediction: {final_result['probability']:.1%}")
        
        # Добавляем метаданные
        final_result['all_predictions'] = predictions
        final_result['prediction_timestamp'] = datetime.now().isoformat()
        
        return final_result
    
    def _create_features_for_prediction_service(self, player1, player2, tournament, surface):
        """Создает признаки для Tennis Prediction Service"""
        # Получаем базовые данные игроков (если есть Real Predictor)
        if self.real_predictor:
            try:
                features = self.real_predictor.create_match_features(
                    player1, player2, tournament, surface, "R64"
                )
                return features
            except:
                pass
        
        # Fallback к простым признакам
        return {
            'player_rank': float(self._estimate_ranking(player1)),
            'opponent_rank': float(self._estimate_ranking(player2)),
            'player_age': 25.0,
            'opponent_age': 25.0,
            'player_recent_win_rate': 0.7,
            'player_form_trend': 0.0,
            'player_surface_advantage': 0.0,
            'h2h_win_rate': 0.5,
            'total_pressure': 2.5
        }
    
    def _estimate_ranking(self, player_name):
        """Простая оценка рейтинга"""
        top_players = {
            'jannik sinner': 1, 'carlos alcaraz': 2, 'alexander zverev': 3,
            'daniil medvedev': 4, 'novak djokovic': 5, 'andrey rublev': 6,
            'aryna sabalenka': 1, 'iga swiatek': 2, 'coco gauff': 3
        }
        
        name_lower = player_name.lower()
        for known_player, rank in top_players.items():
            if known_player in name_lower or any(part in known_player for part in name_lower.split()):
                return rank
        
        return 50  # Средний рейтинг
    
    def _fallback_prediction(self, player1, player2, tournament, surface):
        """Fallback прогноз"""
        p1_rank = self._estimate_ranking(player1)
        p2_rank = self._estimate_ranking(player2)
        
        rank_diff = p2_rank - p1_rank
        probability = 0.5 + (rank_diff * 0.01)
        probability = max(0.1, min(0.9, probability))
        
        confidence = 'High' if abs(rank_diff) > 20 else 'Medium'
        
        return {
            'prediction_type': 'FALLBACK_LOGIC',
            'probability': probability,
            'confidence': confidence,
            'confidence_ru': 'Высокая' if confidence == 'High' else 'Средняя',
            'key_factors': [f'Ranking advantage: {rank_diff} positions'],
            'ml_system_used': 'Fallback Logic'
        }

# Создаем интегрированный предиктор
integrated_predictor = IntegratedMLPredictor()

def get_live_matches_with_ml():
    """Получение live матчей с ML прогнозами"""
    try:
        # 1. Получаем реальные матчи через API
        if API_ECONOMY_AVAILABLE:
            logger.info("🔍 Getting matches via API Economy...")
            api_result = economical_tennis_request('tennis')
            
            if api_result['success'] and api_result.get('data'):
                matches = process_api_matches_with_ml(api_result['data'])
                if matches:
                    return {
                        'matches': matches,
                        'source': f"LIVE_API_{api_result['status']}",
                        'success': True
                    }
        
        # 2. Demo матчи с ML
        logger.info("🎯 Generating demo matches with ML...")
        demo_matches = generate_demo_matches_with_ml()
        return {
            'matches': demo_matches,
            'source': 'DEMO_WITH_ML',
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

def process_api_matches_with_ml(api_data):
    """Обработка API матчей с ML прогнозами"""
    processed_matches = []
    
    for api_match in api_data[:5]:
        try:
            player1 = api_match.get('home_team', 'Player 1')
            player2 = api_match.get('away_team', 'Player 2')
            
            # Извлекаем коэффициенты
            odds1, odds2 = extract_best_odds_from_api(api_match.get('bookmakers', []))
            
            if odds1 and odds2:
                # Получаем ML прогноз
                ml_result = integrated_predictor.predict_match_advanced(
                    player1, player2, 'Live Tournament', 'Hard'
                )
                
                match = {
                    'id': f"api_{api_match.get('id', 'unknown')}",
                    'player1': f"🎾 {player1}",
                    'player2': f"🎾 {player2}",
                    'tournament': '🏆 Live Tournament',
                    'surface': 'Hard',
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'time': api_match.get('commence_time', '14:00')[:5] if api_match.get('commence_time') else '14:00',
                    'odds': {'player1': odds1, 'player2': odds2},
                    'ml_prediction': {
                        'probability': ml_result['probability'],
                        'confidence': ml_result['confidence'],
                        'system_used': ml_result.get('ml_system_used', 'Unknown'),
                        'prediction_type': ml_result['prediction_type']
                    },
                    'key_factors': ml_result.get('key_factors', []),
                    'source': 'LIVE_API_WITH_ML'
                }
                processed_matches.append(match)
                
        except Exception as e:
            logger.error(f"❌ Error processing API match: {e}")
            continue
    
    return processed_matches

def generate_demo_matches_with_ml():
    """Генерация demo матчей с ML прогнозами"""
    demo_matches_data = [
        ('Carlos Alcaraz', 'Novak Djokovic', 'Wimbledon', 'Grass'),
        ('Jannik Sinner', 'Daniil Medvedev', 'US Open', 'Hard'),
        ('Aryna Sabalenka', 'Iga Swiatek', 'WTA Finals', 'Hard'),
        ('Alexander Zverev', 'Andrey Rublev', 'ATP Masters', 'Hard')
    ]
    
    processed_matches = []
    
    for i, (player1, player2, tournament, surface) in enumerate(demo_matches_data):
        try:
            # Получаем ML прогноз
            ml_result = integrated_predictor.predict_match_advanced(
                player1, player2, tournament, surface
            )
            
            # Генерируем коэффициенты на основе ML прогноза
            prob = ml_result['probability']
            p1_odds = round(1 / max(prob, 0.1), 2)
            p2_odds = round(1 / max(1 - prob, 0.1), 2)
            
            match = {
                'id': f"demo_ml_{i+1}",
                'player1': f"🎾 {player1}",
                'player2': f"🎾 {player2}",
                'tournament': f"🏆 {tournament}",
                'surface': surface,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'time': '14:00',
                'odds': {'player1': p1_odds, 'player2': p2_odds},
                'ml_prediction': {
                    'probability': ml_result['probability'],
                    'confidence': ml_result['confidence'],
                    'system_used': ml_result.get('ml_system_used', 'Demo'),
                    'prediction_type': ml_result['prediction_type']
                },
                'key_factors': ml_result.get('key_factors', []),
                'source': 'DEMO_WITH_ML'
            }
            processed_matches.append(match)
            
        except Exception as e:
            logger.error(f"❌ Error generating demo match: {e}")
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

@app.route('/')
def dashboard():
    """Главная страница с интегрированным dashboard"""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎾 Tennis ML Dashboard - Integrated System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; min-height: 100vh; padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { 
            background: rgba(255,255,255,0.1); backdrop-filter: blur(10px);
            border-radius: 20px; padding: 30px; margin-bottom: 30px; text-align: center;
        }
        .ml-banner {
            background: linear-gradient(135deg, #e74c3c, #c0392b);
            padding: 20px; border-radius: 15px; margin-bottom: 20px;
            animation: pulse 3s infinite;
        }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.9; } }
        .stats-grid { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px; margin: 20px 0;
        }
        .stat-card { 
            background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px;
            text-align: center; transition: transform 0.3s ease;
        }
        .stat-card:hover { transform: translateY(-5px); }
        .stat-value { font-size: 2rem; font-weight: bold; margin-bottom: 5px; }
        .controls { text-align: center; margin: 20px 0; }
        .btn { 
            background: linear-gradient(135deg, #667eea, #764ba2); color: white;
            border: none; padding: 12px 24px; border-radius: 25px; font-size: 1rem;
            cursor: pointer; margin: 5px; transition: all 0.3s ease;
        }
        .btn:hover { transform: translateY(-2px); }
        .matches-container { display: grid; gap: 20px; }
        .match-card { 
            background: rgba(255,255,255,0.1); border-radius: 20px; padding: 25px;
            border-left: 5px solid #27ae60; position: relative;
        }
        .ml-indicator { 
            position: absolute; top: 10px; right: 10px; background: #e74c3c;
            color: white; padding: 5px 10px; border-radius: 15px; font-size: 0.8rem;
        }
        .loading { text-align: center; padding: 50px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="ml-banner">
                <h2>🤖 INTEGRATED ML SYSTEM</h2>
                <p>Real ML Models • Trained Ensembles • Manual API Control</p>
            </div>
            
            <h1>🎾 Tennis ML Dashboard</h1>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="ml-systems">-</div>
                    <div class="stat-label">ML Systems</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="api-status">-</div>
                    <div class="stat-label">API Status</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="matches-count">-</div>
                    <div class="stat-label">Live Matches</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="prediction-type">-</div>
                    <div class="stat-label">Prediction Type</div>
                </div>
            </div>
            
            <div class="controls">
                <button class="btn" onclick="loadMatches()">🤖 Load ML Predictions</button>
                <button class="btn" onclick="testMLSystem()">🔮 Test ML System</button>
                <button class="btn" onclick="manualAPIUpdate()">🔄 Manual API Update</button>
                <button class="btn" onclick="checkAPIStatus()">📊 API Economy Status</button>
            </div>
        </div>
        
        <div id="matches-container" class="matches-container">
            <div class="loading">🤖 Loading integrated ML predictions...</div>
        </div>
    </div>
    
    <script>
        const API_BASE = window.location.origin + '/api';
        
        async function loadMatches() {
            const container = document.getElementById('matches-container');
            container.innerHTML = '<div class="loading">🤖 Processing matches with ML systems...</div>';
            
            try {
                const response = await fetch(API_BASE + '/matches');
                const data = await response.json();
                
                if (data.success && data.matches) {
                    let html = `<div style="background: linear-gradient(135deg, #e74c3c, #c0392b); padding: 20px; border-radius: 15px; margin-bottom: 25px; text-align: center;">
                        <h2>🤖 INTEGRATED ML PREDICTIONS</h2>
                        <p>Source: ${data.source} • Matches: ${data.matches.length}</p>
                    </div>`;
                    
                    data.matches.forEach(match => {
                        const mlPred = match.ml_prediction || {};
                        
                        html += `
                            <div class="match-card">
                                <div class="ml-indicator">${mlPred.system_used || 'ML'}</div>
                                
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                    <div>
                                        <div style="font-size: 1.4rem; font-weight: bold;">${match.player1} vs ${match.player2}</div>
                                        <div style="opacity: 0.8; margin-top: 5px;">${match.tournament} • ${match.surface}</div>
                                    </div>
                                    <div style="background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 15px; text-align: center;">
                                        <div style="font-size: 1.2rem; font-weight: bold;">${(mlPred.probability * 100).toFixed(1)}%</div>
                                        <div style="font-size: 0.8rem;">${mlPred.confidence}</div>
                                    </div>
                                </div>
                                
                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 15px 0;">
                                    <div style="text-align: center; padding: 12px; background: rgba(255,255,255,0.1); border-radius: 10px;">
                                        <div style="font-weight: bold;">Odds: ${match.odds.player1} vs ${match.odds.player2}</div>
                                    </div>
                                    <div style="text-align: center; padding: 12px; background: rgba(255,255,255,0.1); border-radius: 10px;">
                                        <div style="font-size: 0.9rem;">${mlPred.prediction_type || 'ML'}</div>
                                    </div>
                                </div>
                                
                                <!-- НОВАЯ КНОПКА ТЕСТА ДЛЯ КАЖДОГО МАТЧА -->
                                <div style="text-align: center; margin: 15px 0;">
                                    <button class="btn" style="font-size: 0.9rem; padding: 8px 16px;" 
                                            onclick="testSpecificMatch('${match.player1.replace('🎾 ', '')}', '${match.player2.replace('🎾 ', '')}', '${match.tournament.replace('🏆 ', '')}', '${match.surface}')">
                                        🔍 Test This Match ML
                                    </button>
                                </div>
                                
                                ${match.key_factors && match.key_factors.length > 0 ? `
                                <div style="margin-top: 15px;">
                                    <strong>🔍 ML Factors:</strong>
                                    <ul style="margin-left: 20px; margin-top: 5px;">
                                        ${match.key_factors.slice(0, 3).map(factor => `<li style="margin: 3px 0;">${factor}</li>`).join('')}
                                    </ul>
                                </div>
                                ` : ''}
                            </div>
                        `;
                    });
                    
                    container.innerHTML = html;
                    document.getElementById('matches-count').textContent = data.matches.length;
                    
                    // Обновляем тип прогноза на основе первого матча
                    if (data.matches.length > 0) {
                        const firstMatch = data.matches[0];
                        const systemUsed = firstMatch.ml_prediction?.system_used || 'Unknown';
                        document.getElementById('prediction-type').textContent = systemUsed;
                    }
                    
                } else {
                    container.innerHTML = '<div class="loading">❌ No matches available</div>';
                }
            } catch (error) {
                container.innerHTML = '<div class="loading">❌ Error loading matches</div>';
            }
        }
        
        async function testSpecificMatch(player1, player2, tournament, surface) {
            try {
                const response = await fetch(API_BASE + '/test-ml', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        player1: player1,
                        player2: player2,
                        tournament: tournament,
                        surface: surface
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    const pred = data.prediction;
                    let message = `🔍 DETAILED ML ANALYSIS\\n`;
                    message += `═══════════════════════════════\\n\\n`;
                    message += `🎾 Match: ${player1} vs ${player2}\\n`;
                    message += `🏆 Tournament: ${tournament}\\n`;
                    message += `🏟️ Surface: ${surface}\\n\\n`;
                    message += `🤖 ML PREDICTION:\\n`;
                    message += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n`;
                    message += `📊 Probability: ${(pred.probability * 100).toFixed(1)}%\\n`;
                    message += `🎯 Confidence: ${pred.confidence}\\n`;
                    message += `🔧 ML System: ${pred.ml_system_used}\\n`;
                    message += `⚡ Type: ${pred.prediction_type}\\n\\n`;
                    
                    if (pred.key_factors && pred.key_factors.length > 0) {
                        message += `🔍 KEY FACTORS:\\n`;
                        message += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n`;
                        pred.key_factors.forEach((factor, i) => {
                            message += `${i + 1}. ${factor}\\n`;
                        });
                    }
                    
                    message += `\\n✅ ML Analysis Complete!`;
                    
                    alert(message);
                } else {
                    alert(`❌ Test failed: ${data.error}`);
                }
            } catch (error) {
                alert(`❌ Test error: ${error.message}`);
            }
        }
        
        async function testMLSystem() {
            try {
                const response = await fetch(API_BASE + '/test-ml', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        player1: 'Carlos Alcaraz',
                        player2: 'Novak Djokovic',
                        tournament: 'US Open',
                        surface: 'Hard'
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    const pred = data.prediction;
                    alert(`🤖 ML Test Result:\\n\\nProbability: ${(pred.probability * 100).toFixed(1)}%\\nConfidence: ${pred.confidence}\\nSystem: ${pred.ml_system_used}\\nType: ${pred.prediction_type}`);
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
                    
                    // Обновляем статистики в dashboard
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
        
        async function updateDashboardStats() {
            try {
                // Получаем детальную статистику
                const statsResponse = await fetch(API_BASE + '/dashboard-stats');
                const statsData = await statsResponse.json();
                
                if (statsData.success) {
                    const stats = statsData.stats;
                    
                    // Обновляем карточки
                    document.getElementById('ml-systems').textContent = stats.ml_systems_count;
                    document.getElementById('prediction-type').textContent = stats.prediction_type;
                    
                    if (stats.api_stats.status === 'active') {
                        document.getElementById('api-status').textContent = `${stats.api_stats.remaining}/${stats.api_stats.max_per_hour}`;
                    } else {
                        document.getElementById('api-status').textContent = '❌ N/A';
                    }
                } else {
                    // Fallback статистика
                    document.getElementById('ml-systems').textContent = '?';
                    document.getElementById('api-status').textContent = '?';
                    document.getElementById('prediction-type').textContent = '?';
                }
                
            } catch (error) {
                console.error('Failed to update dashboard stats:', error);
                // Показываем ошибку в интерфейсе
                document.getElementById('ml-systems').textContent = '!';
                document.getElementById('api-status').textContent = '!';
            }
        }
        
        // Auto-load on page ready
        document.addEventListener('DOMContentLoaded', function() {
            updateDashboardStats();
            loadMatches();
            
            // Периодическое обновление статистик
            setInterval(updateDashboardStats, 30000); // каждые 30 секунд
        });
    </script>
</body>
</html>'''

@app.route('/api/dashboard-stats', methods=['GET'])
def get_dashboard_stats():
    """Получение статистики для dashboard"""
    try:
        # Подсчет активных ML систем
        ml_systems_count = 0
        ml_systems_details = {}
        
        if real_predictor:
            ml_systems_count += 1
            ml_systems_details['real_predictor'] = 'Active'
        
        if tennis_prediction_service:
            ml_systems_count += 1
            ml_systems_details['prediction_service'] = 'Active'
        
        if API_ECONOMY_AVAILABLE:
            ml_systems_count += 1
            ml_systems_details['api_economy'] = 'Active'
        
        # API статистика
        api_stats = {}
        if API_ECONOMY_AVAILABLE:
            try:
                api_usage = get_api_usage()
                api_stats = {
                    'remaining': api_usage.get('remaining_hour', 0),
                    'max_per_hour': api_usage.get('max_per_hour', 30),
                    'status': 'active'
                }
            except:
                api_stats = {'status': 'error'}
        else:
            api_stats = {'status': 'unavailable'}
        
        return jsonify({
            'success': True,
            'stats': {
                'ml_systems_count': ml_systems_count,
                'ml_systems_details': ml_systems_details,
                'api_stats': api_stats,
                'prediction_type': 'Real ML Models' if real_predictor else 'Trained Models' if tennis_prediction_service else 'Fallback'
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Dashboard stats error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check с информацией о всех ML системах"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'ml_systems': {
            'real_predictor': real_predictor is not None,
            'prediction_service': tennis_prediction_service is not None,
            'api_economy': API_ECONOMY_AVAILABLE
        },
        'service': 'integrated_tennis_ml',
        'version': '1.0'
    })

@app.route('/api/matches', methods=['GET'])
def get_matches():
    """Получение матчей с интегрированными ML прогнозами"""
    try:
        logger.info("🤖 Getting matches with integrated ML predictions...")
        
        matches_data = get_live_matches_with_ml()
        
        return jsonify({
            'success': matches_data['success'],
            'matches': matches_data['matches'],
            'count': len(matches_data['matches']),
            'source': matches_data['source'],
            'ml_systems_active': {
                'real_predictor': real_predictor is not None,
                'prediction_service': tennis_prediction_service is not None
            },
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
def test_ml_system():
    """Тестирование интегрированной ML системы"""
    try:
        data = request.get_json()
        
        player1 = data.get('player1', 'Carlos Alcaraz')
        player2 = data.get('player2', 'Novak Djokovic')
        tournament = data.get('tournament', 'US Open')
        surface = data.get('surface', 'Hard')
        
        logger.info(f"🔮 Testing ML system: {player1} vs {player2}")
        
        prediction = integrated_predictor.predict_match_advanced(
            player1, player2, tournament, surface
        )
        
        # Очищаем результат от circular references для JSON
        clean_prediction = {
            'prediction_type': prediction['prediction_type'],
            'probability': prediction['probability'], 
            'confidence': prediction['confidence'],
            'confidence_ru': prediction.get('confidence_ru', prediction['confidence']),
            'key_factors': prediction.get('key_factors', []),
            'ml_system_used': prediction.get('ml_system_used', 'Unknown'),
            'prediction_timestamp': prediction.get('prediction_timestamp', datetime.now().isoformat())
        }
        
        return jsonify({
            'success': True,
            'prediction': clean_prediction,
            'match_info': {
                'player1': player1,
                'player2': player2,
                'tournament': tournament,
                'surface': surface
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ ML test error: {e}")
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
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🎾 INTEGRATED TENNIS ML BACKEND")
    print("=" * 60)
    print(f"🌐 Dashboard: http://0.0.0.0:5001")
    print(f"📡 API: http://0.0.0.0:5001/api/*")
    print(f"🤖 Real Predictor: {'✅ Active' if real_predictor else '❌ Not available'}")
    print(f"🧠 ML Service: {'✅ Active' if tennis_prediction_service else '❌ Not available'}")
    print(f"💰 API Economy: {'✅ Active' if API_ECONOMY_AVAILABLE else '❌ Not available'}")
    print("=" * 60)
    
    try:
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")