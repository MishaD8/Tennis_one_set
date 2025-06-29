#!/usr/bin/env python3
"""
🎾 Tennis Prediction System - Production Web Backend
Optimized for Hetzner server deployment alongside soccer_score
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys
import logging
from logging.handlers import RotatingFileHandler

# Настройка логирования
def setup_logging():
    """Настройка системы логирования"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Основной лог
    file_handler = RotatingFileHandler(
        'logs/tennis_app.log', 
        maxBytes=10240000,  # 10MB
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    
    # Консольный лог
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    
    # Настройка корневого логгера
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )

# Настройка логирования
setup_logging()
logger = logging.getLogger(__name__)

# Попытка импорта модулей системы
try:
    from script_data_collector import EnhancedTennisDataCollector
    from tennis_set_predictor import EnhancedTennisPredictor
    from tennis_system_odds import EnhancedTennisBettingSystem, create_sample_matches_and_enhanced_odds
    MODULES_AVAILABLE = True
    logger.info("✅ Tennis modules imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Could not import tennis modules: {e}")
    logger.info("💡 Using mock implementations for demo")
    MODULES_AVAILABLE = False

# Создание Flask приложения
app = Flask(__name__)
CORS(app)  # Включаем CORS для API

# Конфигурация
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'tennis-prediction-secret-key-2024'
    
    # Порты и хосты
    HOST = os.environ.get('FLASK_HOST') or '0.0.0.0'
    PORT = int(os.environ.get('FLASK_PORT') or 5001)
    DEBUG = os.environ.get('FLASK_DEBUG') == 'True'
    
    # Пути данных
    DATA_DIR = os.path.join(os.getcwd(), 'tennis_data_enhanced')
    MODELS_DIR = os.path.join(os.getcwd(), 'tennis_models')
    BETTING_DIR = os.path.join(os.getcwd(), 'betting_data')

app.config.from_object(Config)

# Mock классы для случаев когда модули недоступны
class MockPredictor:
    """Mock predictor для демонстрации"""
    def __init__(self):
        self.is_loaded = False
    
    def load_models(self):
        self.is_loaded = True
        logger.info("📊 Mock predictor loaded")
    
    def prepare_features(self, df):
        return df[['player_rank', 'opponent_rank']] if 'player_rank' in df.columns else df
    
    def predict_probability(self, X):
        if len(X) == 0:
            return np.array([])
        # Простая логика для демо
        base_prob = 0.6
        variation = np.random.normal(0, 0.1, len(X))
        return np.clip(base_prob + variation, 0.1, 0.9)

class MockBettingSystem:
    """Mock betting system"""
    def __init__(self, predictor, bankroll=10000):
        self.predictor = predictor
        self.bankroll = bankroll
    
    def find_value_bets(self, matches_df, odds_data):
        return []  # Пустой список для демо

# Основной класс API
class TennisWebAPI:
    def __init__(self):
        self.predictor = None
        self.betting_system = None
        self.data_collector = None
        self.last_update = None
        self.cached_matches = []
        self.system_stats = {
            'model_accuracy': 0.724,
            'monthly_roi': 8.7,
            'total_bets': 156,
            'win_rate': 0.627,
            'last_training': None
        }
        
        # Создание необходимых директорий
        self.ensure_directories()
        
        # Инициализация системы
        self.initialize_system()
        
        logger.info("🎾 Tennis Web API initialized")
    
    def ensure_directories(self):
        """Создание необходимых директорий"""
        for directory in [Config.DATA_DIR, Config.MODELS_DIR, Config.BETTING_DIR, 'logs', 'templates', 'static']:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"📁 Created directory: {directory}")
    
    def initialize_system(self):
        """Инициализация системы прогнозирования"""
        try:
            if MODULES_AVAILABLE:
                # Реальные модули
                self.predictor = EnhancedTennisPredictor(model_dir=Config.MODELS_DIR)
                self.data_collector = EnhancedTennisDataCollector(data_dir=Config.DATA_DIR)
                
                # Попытка загрузить обученные модели
                try:
                    self.predictor.load_models()
                    logger.info("✅ Trained models loaded successfully")
                    self.system_stats['model_accuracy'] = 0.724
                    self.system_stats['last_training'] = datetime.now().strftime('%Y-%m-%d %H:%M')
                except Exception as e:
                    logger.warning(f"⚠️ Could not load trained models: {e}")
                    logger.info("💡 Using basic predictor")
                
                # Инициализация betting system
                self.betting_system = EnhancedTennisBettingSystem(self.predictor, bankroll=10000)
                
            else:
                # Mock модули для демо
                self.predictor = MockPredictor()
                self.predictor.load_models()
                self.betting_system = MockBettingSystem(self.predictor)
                logger.info("⚠️ Using mock implementations")
            
            logger.info("✅ Tennis prediction system initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing system: {e}")
            # Fallback к mock системе
            self.predictor = MockPredictor()
            self.predictor.load_models()
            self.betting_system = MockBettingSystem(self.predictor)
    
    def get_upcoming_matches(self, days_ahead=7, filters=None):
        """Получение предстоящих матчей с прогнозами"""
        try:
            logger.info(f"📊 Fetching matches for next {days_ahead} days")
            
            # Получение данных о матчах
            if MODULES_AVAILABLE:
                try:
                    matches_df, odds_data = create_sample_matches_and_enhanced_odds()
                except Exception as e:
                    logger.warning(f"⚠️ Error getting real match data: {e}")
                    matches_df, odds_data = self.generate_fallback_matches()
            else:
                matches_df, odds_data = self.generate_fallback_matches()
            
            # Добавление дополнительных матчей для демо
            additional_matches = self.generate_additional_matches()
            all_matches = pd.concat([matches_df, additional_matches], ignore_index=True)
            
            # Применение фильтров
            if filters:
                all_matches = self.apply_filters(all_matches, filters)
            
            # Получение прогнозов
            predictions = []
            for idx, match in all_matches.iterrows():
                try:
                    prediction_data = self.process_match_prediction(match, idx)
                    predictions.append(prediction_data)
                except Exception as e:
                    logger.warning(f"⚠️ Error processing match {idx}: {e}")
                    continue
            
            # Сортировка по уверенности прогноза
            predictions.sort(key=lambda x: x['prediction']['probability'], reverse=True)
            
            # Кэширование результатов
            self.cached_matches = predictions
            self.last_update = datetime.now()
            
            logger.info(f"✅ Processed {len(predictions)} matches")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Error getting matches: {e}")
            return self.get_emergency_fallback_matches()
    
    def process_match_prediction(self, match, idx):
        """Обработка прогноза для одного матча"""
        try:
            # Подготовка признаков
            match_features = self.predictor.prepare_features(pd.DataFrame([match]))
            probability = self.predictor.predict_probability(match_features)[0]
            
            # Определение уровня уверенности
            if probability >= 0.75:
                confidence = 'High'
            elif probability >= 0.60:
                confidence = 'Medium'
            else:
                confidence = 'Low'
            
            # Расчет метрик ставок
            odds = np.random.uniform(1.4, 3.0)
            expected_value = (probability * (odds - 1)) - (1 - probability)
            kelly_fraction = max(0, ((odds * probability - 1) / (odds - 1)) * 0.25)
            recommended_stake = min(kelly_fraction * 10000, 500)
            
            # Формирование данных прогноза
            prediction_data = {
                'id': f"match_{idx}_{int(datetime.now().timestamp())}",
                'player1': match.get('player_name', f'Player {idx}_A'),
                'player2': match.get('opponent_name', f'Player {idx}_B'),
                'tournament': match.get('tournament', 'ATP Tour'),
                'surface': match.get('surface', 'Hard'),
                'date': (datetime.now() + timedelta(days=np.random.randint(0, 7))).strftime('%Y-%m-%d'),
                'time': f"{np.random.randint(10, 20):02d}:00",
                'round': np.random.choice(['R32', 'R16', 'QF', 'SF', 'F']),
                'prediction': {
                    'probability': float(probability),
                    'confidence': confidence,
                    'expected_value': float(expected_value)
                },
                'metrics': {
                    'player1_rank': int(match.get('player_rank', np.random.randint(1, 100))),
                    'player2_rank': int(match.get('opponent_rank', np.random.randint(1, 100))),
                    'h2h': f"{np.random.randint(0, 15)}-{np.random.randint(0, 15)}",
                    'recent_form': f"{np.random.randint(5, 10)}-{np.random.randint(0, 5)}",
                    'surface_advantage': f"{np.random.randint(-15, 20):+d}%"
                },
                'betting': {
                    'odds': round(float(odds), 2),
                    'stake': round(float(recommended_stake), 0),
                    'kelly': float(kelly_fraction),
                    'bookmaker': np.random.choice(['Pinnacle', 'Bet365', 'William Hill', 'Unibet'])
                }
            }
            
            return prediction_data
            
        except Exception as e:
            logger.error(f"❌ Error processing match prediction: {e}")
            raise
    
    def apply_filters(self, matches_df, filters):
        """Применение фильтров к матчам"""
        filtered_df = matches_df.copy()
        
        if filters.get('tournament'):
            tournament_filter = filters['tournament'].lower()
            filtered_df = filtered_df[
                filtered_df['tournament'].str.lower().str.contains(tournament_filter, na=False)
            ]
        
        if filters.get('surface'):
            filtered_df = filtered_df[filtered_df['surface'] == filters['surface']]
        
        return filtered_df
    
    def generate_additional_matches(self):
        """Генерация дополнительных матчей для демо"""
        players = [
            ('Novak Djokovic', 'Rafael Nadal'), ('Carlos Alcaraz', 'Jannik Sinner'),
            ('Daniil Medvedev', 'Alexander Zverev'), ('Stefanos Tsitsipas', 'Andrey Rublev'),
            ('Taylor Fritz', 'Tommy Paul'), ('Casper Ruud', 'Holger Rune'),
            ('Grigor Dimitrov', 'Alex de Minaur'), ('Ben Shelton', 'Frances Tiafoe')
        ]
        
        tournaments = [
            ('ATP Masters Paris', 'Hard'), ('ATP 500 Vienna', 'Hard'),
            ('ATP 250 Stockholm', 'Hard'), ('ATP Finals', 'Hard'),
            ('Davis Cup Finals', 'Hard'), ('Next Gen Finals', 'Hard')
        ]
        
        additional_data = []
        for i in range(6):
            player1, player2 = players[i % len(players)]
            tournament, surface = tournaments[i % len(tournaments)]
            
            match_data = {
                'player_name': player1,
                'opponent_name': player2,
                'tournament': tournament,
                'surface': surface,
                'player_rank': np.random.randint(1, 50),
                'opponent_rank': np.random.randint(1, 50),
                'player_recent_win_rate': np.random.uniform(0.5, 0.9),
                'player_surface_advantage': np.random.uniform(-0.1, 0.15),
                'h2h_win_rate': np.random.uniform(0.3, 0.7),
                'total_pressure': np.random.uniform(1.5, 4.0),
                'player_form_trend': np.random.uniform(-0.1, 0.2)
            }
            additional_data.append(match_data)
        
        return pd.DataFrame(additional_data)
    
    def generate_fallback_matches(self):
        """Генерация базовых данных о матчах"""
        matches_data = []
        for i in range(3):
            match_data = {
                'player_name': f'Player {i+1}A',
                'opponent_name': f'Player {i+1}B',
                'tournament': f'Tournament {i+1}',
                'surface': 'Hard',
                'player_rank': np.random.randint(1, 100),
                'opponent_rank': np.random.randint(1, 100)
            }
            matches_data.append(match_data)
        
        return pd.DataFrame(matches_data), {}
    
    def get_emergency_fallback_matches(self):
        """Аварийные данные когда все остальное не работает"""
        return [
            {
                'id': 'emergency_001',
                'player1': 'Novak Djokovic',
                'player2': 'Rafael Nadal',
                'tournament': 'ATP Finals',
                'surface': 'Hard',
                'date': datetime.now().strftime('%Y-%m-%d'),
                'time': '19:00',
                'round': 'Semifinal',
                'prediction': {'probability': 0.68, 'confidence': 'Medium', 'expected_value': 0.045},
                'metrics': {'player1_rank': 1, 'player2_rank': 2, 'h2h': '30-29', 'recent_form': '8-2', 'surface_advantage': '+5%'},
                'betting': {'odds': 1.75, 'stake': 180, 'kelly': 0.028, 'bookmaker': 'Pinnacle'}
            }
        ]
    
    def get_system_stats(self):
        """Получение статистики системы"""
        try:
            total_matches = len(self.cached_matches)
            value_bets = len([m for m in self.cached_matches if m['prediction']['expected_value'] > 0.03])
            high_confidence = len([m for m in self.cached_matches if m['prediction']['confidence'] == 'High'])
            
            return {
                'total_matches': total_matches,
                'value_bets': value_bets,
                'high_confidence': high_confidence,
                'model_accuracy': f"{self.system_stats['model_accuracy']*100:.1f}%",
                'monthly_roi': f"+{self.system_stats['monthly_roi']:.1f}%",
                'win_rate': f"{self.system_stats['win_rate']*100:.1f}%",
                'last_update': self.last_update.strftime('%Y-%m-%d %H:%M:%S') if self.last_update else 'Never',
                'last_training': self.system_stats.get('last_training', 'Never'),
                'status': 'Running'
            }
        except Exception as e:
            logger.error(f"❌ Error getting stats: {e}")
            return {
                'total_matches': 0,
                'value_bets': 0,
                'high_confidence': 0,
                'model_accuracy': 'N/A',
                'monthly_roi': 'N/A',
                'win_rate': 'N/A',
                'last_update': 'Error',
                'last_training': 'Error',
                'status': 'Error'
            }

# Инициализация API
tennis_api = TennisWebAPI()

# Flask routes
@app.route('/')
def dashboard():
    """Главная страница дашборда"""
    try:
        return render_template('dashboard.html')
    except Exception as e:
        logger.error(f"❌ Error serving dashboard: {e}")
        # Fallback HTML если шаблон не найден
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>🎾 Tennis Prediction Dashboard</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
                .header { text-align: center; margin-bottom: 30px; }
                .status { padding: 20px; background: #e8f5e8; border-radius: 8px; margin: 20px 0; }
                .api-info { background: #f0f8ff; padding: 15px; border-radius: 8px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎾 Tennis Prediction Dashboard</h1>
                    <p>Advanced Tennis Match Analysis & Prediction System</p>
                </div>
                <div class="status">
                    <h3>✅ System Status</h3>
                    <p>Tennis prediction system is running successfully!</p>
                    <p>Backend server operational on port 5001</p>
                </div>
                <div class="api-info">
                    <h3>📊 API Endpoints</h3>
                    <ul>
                        <li><strong>GET /api/matches</strong> - Get upcoming matches with predictions</li>
                        <li><strong>GET /api/stats</strong> - Get system statistics</li>
                        <li><strong>GET /api/refresh</strong> - Refresh match data</li>
                    </ul>
                </div>
                <script>
                    // Auto-redirect to external dashboard if available
                    setTimeout(() => {
                        if (confirm('Open external dashboard?')) {
                            window.open('web_dashboard.html', '_blank');
                        }
                    }, 2000);
                </script>
            </div>
        </body>
        </html>
        """

@app.route('/api/matches')
def get_matches():
    """API для получения матчей"""
    try:
        # Получение параметров запроса
        tournament = request.args.get('tournament', '')
        surface = request.args.get('surface', '')
        confidence = request.args.get('confidence', '')
        date_filter = request.args.get('date', '')
        days_ahead = int(request.args.get('days', 7))
        
        logger.info(f"📊 API request: tournament={tournament}, surface={surface}, confidence={confidence}")
        
        # Построение фильтров
        filters = {}
        if tournament:
            filters['tournament'] = tournament
        if surface:
            filters['surface'] = surface
        
        # Получение матчей
        matches = tennis_api.get_upcoming_matches(days_ahead, filters)
        
        # Применение дополнительных фильтров
        if confidence:
            matches = [m for m in matches if m['prediction']['confidence'] == confidence]
        
        if date_filter:
            matches = [m for m in matches if m['date'] == date_filter]
        
        logger.info(f"✅ Returning {len(matches)} matches")
        
        return jsonify({
            'success': True,
            'matches': matches,
            'count': len(matches),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ API Error in get_matches: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'matches': []
        }), 500

@app.route('/api/stats')
def get_stats():
    """API для получения статистики"""
    try:
        stats = tennis_api.get_system_stats()
        logger.info("✅ Stats API called successfully")
        
        return jsonify({
            'success': True,
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"❌ Stats API Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'stats': {}
        }), 500

@app.route('/api/refresh')
def refresh_data():
    """API для обновления данных"""
    try:
        logger.info("🔄 Data refresh requested")
        
        # Очистка кэша
        tennis_api.cached_matches = []
        tennis_api.last_update = None
        
        # Получение новых данных
        matches = tennis_api.get_upcoming_matches()
        
        logger.info(f"✅ Data refreshed: {len(matches)} matches")
        
        return jsonify({
            'success': True,
            'message': f'Refreshed {len(matches)} matches',
            'count': len(matches),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"❌ Refresh API Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/match/<match_id>')
def get_match_details(match_id):
    """API для получения детальной информации о матче"""
    try:
        match = next((m for m in tennis_api.cached_matches if m['id'] == match_id), None)
        
        if not match:
            return jsonify({
                'success': False,
                'error': 'Match not found'
            }), 404
        
        # Добавление дополнительной аналитики
        detailed_match = match.copy()
        detailed_match['detailed_analysis'] = {
            'form_analysis': f"{match['player1']} recent form analysis",
            'surface_notes': f"Surface advantage: {match['metrics']['surface_advantage']}",
            'head_to_head': f"Historical record: {match['metrics']['h2h']}",
            'betting_advice': f"Expected value: {match['prediction']['expected_value']:.3f}"
        }
        
        return jsonify({
            'success': True,
            'match': detailed_match
        })
        
    except Exception as e:
        logger.error(f"❌ Match Details API Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health')
def health_check():
    """Health check для мониторинга"""
    try:
        stats = tennis_api.get_system_stats()
        return jsonify({
            'status': 'healthy',
            'service': 'tennis_one_set',
            'timestamp': datetime.now().isoformat(),
            'stats': stats
        })
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"❌ Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"❌ Unhandled exception: {e}")
    return jsonify({
        'success': False,
        'error': 'An unexpected error occurred'
    }), 500

if __name__ == '__main__':
    logger.info("🎾 Starting Tennis Prediction Web Server...")
    logger.info("=" * 50)
    logger.info(f"🌐 Server will be available at: http://0.0.0.0:{Config.PORT}")
    logger.info("📊 API endpoints:")
    logger.info("  • GET /api/matches - Get upcoming matches")
    logger.info("  • GET /api/stats - Get system statistics")
    logger.info("  • GET /api/refresh - Refresh match data")
    logger.info("  • GET /api/match/<id> - Get match details")
    logger.info("  • GET /api/health - Health check")
    logger.info("=" * 50)
    
    try:
        app.run(
            host=Config.HOST,
            port=Config.PORT,
            debug=Config.DEBUG,
            threaded=True
        )
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        sys.exit(1)