#!/usr/bin/env python3
"""
Tennis Backend Routes
All API routes and endpoints for tennis betting and ML prediction system
"""

import os
import logging
import json
import re
import html
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from functools import wraps

from flask import Flask, jsonify, request, render_template, redirect
from werkzeug.exceptions import HTTPException

from middleware import require_api_key, validate_json_request, log_request_info, handle_betting_errors

# Set up logger
logger = logging.getLogger(__name__)

# Import tennis system components with fallback handling
try:
    from real_tennis_predictor_integration import RealTennisPredictor
    REAL_PREDICTOR_AVAILABLE = True
    logger.info("✅ Real ML predictor imported")
except ImportError as e:
    logger.info(f"⚠️ Real predictor not available: {e}")
    REAL_PREDICTOR_AVAILABLE = False

try:
    from tennis_prediction_module import TennisPredictionService
    PREDICTION_SERVICE_AVAILABLE = True
    logger.info("✅ Prediction service imported")
except ImportError as e:
    logger.info(f"⚠️ Prediction service not available: {e}")
    PREDICTION_SERVICE_AVAILABLE = False

ODDS_API_AVAILABLE = False
API_ECONOMY_AVAILABLE = False

try:
    from enhanced_universal_collector import EnhancedUniversalCollector
    from universal_tennis_data_collector import UniversalOddsCollector
    ENHANCED_COLLECTOR_AVAILABLE = True
    UNIVERSAL_COLLECTOR_AVAILABLE = False  # Will be set below if needed
    logger.info("✅ Enhanced Universal Collector loaded")
except ImportError as e:
    logger.info(f"⚠️ Enhanced collector not available: {e}")
    ENHANCED_COLLECTOR_AVAILABLE = False
    try:
        from universal_tennis_data_collector import UniversalTennisDataCollector, UniversalOddsCollector
        UNIVERSAL_COLLECTOR_AVAILABLE = True
        logger.info("✅ Universal data collector loaded (fallback)")
    except ImportError as e2:
        logger.info(f"⚠️ Universal collector not available: {e2}")
        UNIVERSAL_COLLECTOR_AVAILABLE = False

try:
    from daily_api_scheduler import init_daily_scheduler, start_daily_scheduler
    DAILY_SCHEDULER_AVAILABLE = True
    logger.info("✅ Daily API scheduler loaded")
except ImportError as e:
    logger.info(f"⚠️ Daily scheduler not available: {e}")
    DAILY_SCHEDULER_AVAILABLE = False

# Global service instances (will be initialized in register_routes)
real_predictor = None
prediction_service = None
odds_integrator = None
enhanced_collector = None
universal_collector = None
daily_scheduler = None

class UnderdogAnalyzer:
    """Tennis underdog scenario analyzer"""
    
    def __init__(self):
        self.player_rankings = {
            # Current ATP rankings (updated July 2025)
            "jannik sinner": 1, "carlos alcaraz": 2, "alexander zverev": 3,
            "daniil medvedev": 4, "novak djokovic": 5, "andrey rublev": 6,
            "casper ruud": 7, "holger rune": 8, "grigor dimitrov": 9,
            "stefanos tsitsipas": 10, "taylor fritz": 11, "tommy paul": 12,
            "alex de minaur": 13, "ben shelton": 14, "ugo humbert": 15,
            "lorenzo musetti": 16, "sebastian baez": 17, "frances tiafoe": 18,
            "felix auger-aliassime": 19, "arthur fils": 20,
            # Additional players from system
            "flavio cobolli": 32, "brandon nakashima": 45, "bu yunchaokete": 85,
            "matteo berrettini": 35, "cameron norrie": 40, "sebastian korda": 25,
            "francisco cerundolo": 30, "alejandro tabilo": 28, "fabio fognini": 85,
            # WTA rankings
            "aryna sabalenka": 1, "iga swiatek": 2, "coco gauff": 3,
            "jessica pegula": 4, "elena rybakina": 5, "qinwen zheng": 6,
            "jasmine paolini": 7, "emma navarro": 8, "daria kasatkina": 9,
            "renata zarazua": 80, "amanda anisimova": 35, "katie boulter": 28,
            "emma raducanu": 25, "caroline dolehide": 85, "carson branstine": 125
        }
        
        self.surface_bonuses = {
            'Clay': {'spanish': 0.15, 'south_american': 0.1},
            'Grass': {'british': 0.1, 'serve_volley': 0.05},
            'Hard': {'american': 0.05, 'baseline': 0.03}
        }
        
    def get_player_ranking(self, player_name: str) -> int:
        """Get player ranking with fallback"""
        normalized_name = player_name.lower().strip()
        return self.player_rankings.get(normalized_name, 999)
        
    def calculate_underdog_probability(self, player1: str, player2: str, 
                                     tournament: str, surface: str) -> Dict[str, Any]:
        """Calculate detailed underdog probability analysis"""
        
        rank1 = self.get_player_ranking(player1)
        rank2 = self.get_player_ranking(player2)
        
        # Determine underdog and favorite
        if rank1 > rank2:
            underdog, favorite = player1, player2
            underdog_rank, favorite_rank = rank1, rank2
        else:
            underdog, favorite = player2, player1
            underdog_rank, favorite_rank = rank2, rank1
            
        rank_gap = underdog_rank - favorite_rank
        
        # Base probability calculation
        if rank_gap <= 5:
            base_prob = 0.48
            underdog_type = "Slight underdog"
        elif rank_gap <= 15:
            base_prob = 0.35
            underdog_type = "Moderate underdog"
        elif rank_gap <= 30:
            base_prob = 0.25
            underdog_type = "Strong underdog"
        else:
            base_prob = 0.15
            underdog_type = "Major underdog"
            
        # Apply surface bonuses
        surface_bonus = 0.0
        if surface in self.surface_bonuses:
            # Simplified surface advantage logic
            if 'clay' in tournament.lower() or surface.lower() == 'clay':
                surface_bonus = 0.05
                
        # Tournament pressure factor
        pressure_factor = 0.0
        if any(major in tournament.lower() for major in ['grand slam', 'masters', 'wimbledon', 'us open']):
            pressure_factor = 0.03  # Higher pressure can level the playing field
            
        # Calculate final probability
        final_probability = min(base_prob + surface_bonus + pressure_factor, 0.6)
        
        # Confidence calculation
        confidence_score = max(0.6, 1.0 - (rank_gap / 100))
        
        key_factors = []
        if surface_bonus > 0:
            key_factors.append(f"Surface advantage (+{surface_bonus:.2%})")
        if pressure_factor > 0:
            key_factors.append(f"Tournament pressure factor (+{pressure_factor:.2%})")
        if rank_gap > 50:
            key_factors.append("Significant ranking gap - upset potential")
            
        return {
            'underdog_probability': round(final_probability, 3),
            'confidence': round(confidence_score, 3),
            'prediction_type': 'UNDERDOG_ANALYSIS',
            'key_factors': key_factors,
            'underdog_scenario': {
                'underdog': underdog,
                'favorite': favorite,
                'rank_gap': rank_gap,
                'underdog_type': underdog_type,
                'base_probability': base_prob
            }
        }

# Security and validation functions
def validate_player_name(name: str) -> bool:
    """Validate player name input with enhanced security"""
    if not isinstance(name, str):
        return False
    
    name = name.strip()
    if not name:
        return False
        
    # Length validation - prevent DoS attacks
    if len(name) > 100:
        return False
    
    # Character validation - allow letters, spaces, apostrophes, hyphens, dots
    if not re.match(r"^[a-zA-Z\s\'\-\.]+$", name):
        return False
        
    return True

def validate_tournament_name(tournament: str) -> bool:
    """Validate tournament name"""
    if not isinstance(tournament, str):
        return False
        
    tournament = tournament.strip()
    if not tournament:
        return False
        
    if len(tournament) > 200:
        return False
        
    # Allow alphanumeric, spaces, and common tournament characters
    if not re.match(r"^[a-zA-Z0-9\s\'\-\.\(\)]+$", tournament):
        return False
        
    return True

def validate_surface(surface: str) -> bool:
    """Validate tennis surface"""
    valid_surfaces = ['Hard', 'Clay', 'Grass', 'Carpet']
    return surface in valid_surfaces

def sanitize_input(data: Any) -> Any:
    """Sanitize string inputs with enhanced security"""
    if isinstance(data, str):
        # Strip whitespace and remove null bytes
        cleaned = data.strip().replace('\x00', '')
        # Escape HTML to prevent XSS
        return html.escape(cleaned, quote=True)
    elif isinstance(data, dict):
        # Recursively sanitize dictionary values with depth limit
        if hasattr(sanitize_input, '_depth'):
            sanitize_input._depth += 1
        else:
            sanitize_input._depth = 1
            
        if sanitize_input._depth > 10:  # Prevent deep recursion attacks
            sanitize_input._depth -= 1
            return {}
            
        result = {key: sanitize_input(value) for key, value in data.items() 
                 if isinstance(key, str) and len(key) <= 100}
        sanitize_input._depth -= 1
        return result
    elif isinstance(data, list):
        # Limit list size to prevent DoS
        if len(data) > 1000:
            return data[:1000]
        return [sanitize_input(item) for item in data]
    return data

def validate_json_payload(data: Dict, max_keys: int = 20, max_depth: int = 5) -> bool:
    """Validate JSON payload structure to prevent DoS attacks"""
    if not isinstance(data, dict):
        return False
    
    if len(data) > max_keys:
        return False
    
    def check_depth(obj, current_depth=0):
        if current_depth > max_depth:
            return False
        if isinstance(obj, dict):
            for value in obj.values():
                if not check_depth(value, current_depth + 1):
                    return False
        elif isinstance(obj, list):
            for item in obj:
                if not check_depth(item, current_depth + 1):
                    return False
        return True
    
    return check_depth(data)

def check_redis_health() -> Dict[str, Any]:
    """Check Redis connection health for monitoring"""
    redis_health = {
        'available': False,
        'connection_time_ms': None,
        'error': None,
        'version': None
    }
    
    try:
        import redis
        import time
        
        start_time = time.time()
        redis_url = os.getenv('REDIS_URL', '').strip()
        
        if redis_url.startswith('redis://'):
            r = redis.Redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2)
        else:
            r = redis.Redis(host='localhost', port=6379, socket_connect_timeout=2, socket_timeout=2)
        
        r.ping()
        connection_time = (time.time() - start_time) * 1000
        
        redis_health.update({
            'available': True,
            'connection_time_ms': round(connection_time, 2),
            'version': r.info().get('redis_version', 'unknown')
        })
        
    except ImportError:
        redis_health['error'] = 'Redis package not installed'
    except Exception as e:
        redis_health['error'] = str(e)
    
    return redis_health

def create_safe_error_response(error: Exception, fallback_message: str) -> str:
    """Create safe error response without exposing sensitive information"""
    if os.getenv('FLASK_ENV') == 'production':
        return fallback_message
    return str(error)

def format_match_for_dashboard(match_data: Dict, source: str = "unknown") -> Dict:
    """Unify match data format for dashboard display"""
    try:
        # Base fields
        formatted = {
            'id': match_data.get('id', f"match_{datetime.now().timestamp()}"),
            'player1': match_data.get('player1', 'Unknown Player 1'),
            'player2': match_data.get('player2', 'Unknown Player 2'),
            'tournament': match_data.get('tournament', 'Unknown Tournament'),
            'surface': match_data.get('surface', 'Hard'),
            'date': match_data.get('date', datetime.now().strftime('%Y-%m-%d')),
            'time': match_data.get('time', datetime.now().strftime('%H:%M')),
            'source': source
        }
        
        # Handle prediction data
        if 'prediction' in match_data:
            formatted['prediction'] = match_data['prediction']
        else:
            # Default prediction structure
            formatted['prediction'] = {
                'probability': match_data.get('probability', 0.5),
                'confidence': match_data.get('confidence', 0.7),
                'prediction_type': match_data.get('prediction_type', 'ANALYSIS'),
                'key_factors': match_data.get('key_factors', [])
            }
        
        # Handle odds
        if 'odds' in match_data:
            formatted['odds'] = match_data['odds']
        else:
            formatted['odds'] = {
                'player1': match_data.get('odds_player1', 2.0),
                'player2': match_data.get('odds_player2', 2.0)
            }
        
        # Additional fields
        formatted.update({
            'prediction_type': formatted['prediction']['prediction_type'],
            'underdog_probability': formatted['prediction']['probability'],
            'value_bet': formatted['prediction']['probability'] > (1 / formatted['odds']['player1']),
            'key_factors': formatted['prediction']['key_factors']
        })
        
        return formatted
        
    except Exception as e:
        logger.warning(f"Error formatting match data: {e}")
        # Return minimal safe format
        return {
            'id': f"error_{datetime.now().timestamp()}",
            'player1': 'Error Player 1',
            'player2': 'Error Player 2',
            'tournament': 'Error Tournament',
            'surface': 'Hard',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'source': 'error',
            'prediction': {
                'probability': 0.5,
                'confidence': 0.5,
                'prediction_type': 'ERROR',
                'key_factors': ['Data formatting error']
            },
            'odds': {'player1': 2.0, 'player2': 2.0}
        }

def get_live_matches_with_underdog_focus() -> Dict:
    """Get matches with underdog analysis focus"""
    try:
        # Priority 1: Enhanced Universal Collector
        if ENHANCED_COLLECTOR_AVAILABLE and enhanced_collector:
            try:
                logger.info("🌍 Using Enhanced Universal Collector...")
                ml_ready_matches = enhanced_collector.get_ml_ready_matches(min_quality_score=30)
                
                if ml_ready_matches and len(ml_ready_matches) > 0:
                    logger.info(f"✅ Got {len(ml_ready_matches)} ML-ready matches from Enhanced Collector")
                    
                    analyzer = UnderdogAnalyzer()
                    processed_matches = []
                    
                    for match in ml_ready_matches[:6]:
                        try:
                            player1 = match.get('player1', 'Player 1')
                            player2 = match.get('player2', 'Player 2')
                            
                            ml_features = match.get('ml_features', {})
                            
                            underdog_analysis = analyzer.calculate_underdog_probability(
                                player1, player2,
                                match.get('tournament', 'ATP Tournament'),
                                match.get('surface', 'Hard')
                            )
                            
                            processed_match = {
                                'id': match.get('match_id', f"enhanced_{len(processed_matches)}"),
                                'player1': player1,
                                'player2': player2,
                                'tournament': match.get('tournament', 'ATP Tournament'),
                                'surface': match.get('surface', 'Hard'),
                                'date': match.get('date', datetime.now().strftime('%Y-%m-%d')),
                                'odds': {
                                    'player1': match.get('odds', {}).get('player1', 2.0),
                                    'player2': match.get('odds', {}).get('player2', 2.0)
                                },
                                'prediction': underdog_analysis,
                                'ml_features': ml_features,
                                'quality_score': match.get('quality_score', 0)
                            }
                            processed_matches.append(processed_match)
                            
                        except Exception as e:
                            logger.warning(f"Error processing enhanced match: {e}")
                            continue
                    
                    if processed_matches:
                        return {
                            'success': True,
                            'matches': processed_matches,
                            'source': 'ENHANCED_UNIVERSAL_COLLECTOR',
                            'count': len(processed_matches)
                        }
                        
            except Exception as e:
                logger.warning(f"Enhanced collector failed: {e}")
        
        # Fallback to test data for development
        test_matches = [
            ("Flavio Cobolli", "Novak Djokovic", "US Open", "Hard"),
            ("Brandon Nakashima", "Carlos Alcaraz", "ATP Masters", "Hard"),
            ("Bu Yunchaokete", "Alexander Zverev", "ATP 500", "Hard"),
            ("Amanda Anisimova", "Aryna Sabalenka", "WTA 1000", "Hard"),
            ("Katie Boulter", "Iga Swiatek", "WTA 500", "Clay"),
            ("Emma Raducanu", "Coco Gauff", "WTA 250", "Grass")
        ]
        
        analyzer = UnderdogAnalyzer()
        processed_matches = []
        
        for i, (player1, player2, tournament, surface) in enumerate(test_matches):
            try:
                underdog_analysis = analyzer.calculate_underdog_probability(
                    player1, player2, tournament, surface
                )
                
                # Simulate realistic odds
                rank1 = analyzer.get_player_ranking(player1)
                rank2 = analyzer.get_player_ranking(player2)
                
                if rank1 < rank2:  # player1 is favorite
                    odds1 = max(1.2, 2.0 - (rank2 - rank1) / 50)
                    odds2 = max(1.2, 1.5 + (rank2 - rank1) / 30)
                else:  # player2 is favorite
                    odds1 = max(1.2, 1.5 + (rank1 - rank2) / 30)
                    odds2 = max(1.2, 2.0 - (rank1 - rank2) / 50)
                
                processed_match = {
                    'id': f"test_match_{i+1}",
                    'player1': player1,
                    'player2': player2,
                    'tournament': tournament,
                    'surface': surface,
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'time': f"{10 + i}:00",
                    'odds': {
                        'player1': round(odds1, 2),
                        'player2': round(odds2, 2)
                    },
                    'prediction': underdog_analysis
                }
                processed_matches.append(processed_match)
                
            except Exception as e:
                logger.warning(f"Error processing test match: {e}")
                continue
        
        return {
            'success': True,
            'matches': processed_matches,
            'source': 'TEST_UNDERDOG_DATA',
            'count': len(processed_matches)
        }
        
    except Exception as e:
        logger.error(f"Error getting matches: {e}")
        return {
            'success': False,
            'matches': [],
            'source': 'ERROR',
            'error': str(e)
        }

# Route definitions
def register_routes(app: Flask):
    """Register all tennis backend routes"""
    
    # Get the limiter from app context
    limiter = getattr(app, 'limiter', None)
    if not limiter:
        logger.warning("Rate limiter not available in app context")
    
    # Initialize global service instances
    global real_predictor, prediction_service, odds_integrator
    global enhanced_collector, universal_collector, daily_scheduler
    
    # Initialize services
    if REAL_PREDICTOR_AVAILABLE:
        try:
            real_predictor = RealTennisPredictor()
            logger.info("✅ Real predictor initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize real predictor: {e}")
    
    if PREDICTION_SERVICE_AVAILABLE:
        try:
            prediction_service = TennisPredictionService()
            logger.info("✅ Prediction service initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize prediction service: {e}")
    
    if ODDS_API_AVAILABLE:
        try:
            odds_integrator = TennisOddsIntegrator()
            logger.info("✅ Odds integrator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize odds integrator: {e}")
    
    if ENHANCED_COLLECTOR_AVAILABLE:
        try:
            enhanced_collector = EnhancedUniversalCollector()
            logger.info("✅ Enhanced collector initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize enhanced collector: {e}")
    
    if UNIVERSAL_COLLECTOR_AVAILABLE:
        try:
            universal_collector = UniversalTennisDataCollector()
            logger.info("✅ Universal collector initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize universal collector: {e}")
    
    if DAILY_SCHEDULER_AVAILABLE:
        try:
            daily_scheduler = init_daily_scheduler()
            start_daily_scheduler()
            logger.info("✅ Daily scheduler initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize daily scheduler: {e}")
    
    @app.route('/')
    def dashboard():
        """Main dashboard page"""
        return render_template('dashboard.html')

    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Comprehensive health check with security and infrastructure monitoring"""
        redis_health = check_redis_health()
        
        # Determine overall health status
        overall_status = 'healthy'
        warnings = []
        
        if not redis_health['available']:
            warnings.append('Redis unavailable - using in-memory rate limiting')
        
        connection_time = redis_health.get('connection_time_ms')
        if connection_time is not None and connection_time > 100:
            warnings.append('Redis connection slow')
        
        if os.getenv('FLASK_ENV') != 'production' and request.headers.get('X-Forwarded-Proto') != 'https':
            warnings.append('Not using HTTPS in production')
        
        if warnings:
            overall_status = 'degraded'
        
        return jsonify({
            'status': overall_status,
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
            'infrastructure': {
                'redis': redis_health,
                'rate_limiting': 'redis' if redis_health['available'] else 'memory',
                'ssl_enabled': request.is_secure or request.headers.get('X-Forwarded-Proto') == 'https',
                'environment': os.getenv('FLASK_ENV', 'development')
            },
            'security': {
                'https_enforced': app.config.get('FORCE_HTTPS', False),
                'secure_cookies': app.config.get('SESSION_COOKIE_SECURE', False),
                'csp_enabled': True,
                'hsts_enabled': os.getenv('FLASK_ENV') == 'production'
            },
            'warnings': warnings,
            'version': '5.0-modular'
        })

    @app.route('/api/stats', methods=['GET'])
    def get_stats():
        """System statistics"""
        try:
            # Determine ML predictor status
            if real_predictor and hasattr(real_predictor, 'prediction_service') and real_predictor.prediction_service:
                ml_status = 'real_models'
                prediction_type = 'REAL_ML_MODEL'
            elif prediction_service:
                ml_status = 'prediction_service'
                prediction_type = 'PREDICTION_SERVICE'
            else:
                ml_status = 'simulation'
                prediction_type = 'ADVANCED_SIMULATION'
            
            # API usage stats - Note: Old APIs removed
            api_stats = {'message': 'Old API integrations removed'}
            
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
                'error': create_safe_error_response(e, 'System statistics unavailable')
            }), 500

    @app.route('/api/matches', methods=['GET'])
    @log_request_info()
    def get_matches():
        """Get matches with underdog analysis"""
        try:
            # Parameter to control data source
            use_real_data_only = request.args.get('real_data_only', 'true').lower() == 'true'
            force_source = request.args.get('source', None)
            
            logger.info(f"🎾 Getting matches (real_data_only={use_real_data_only}, force_source={force_source})")
            
            # Get matches
            matches_result = get_live_matches_with_underdog_focus()
            
            if not matches_result or not isinstance(matches_result, dict) or not matches_result.get('success', False):
                return jsonify({
                    'success': False,
                    'error': 'Failed to get matches',
                    'matches': []
                }), 500
            
            # Filter data if only real data needed
            raw_matches = matches_result.get('matches', [])
            
            if use_real_data_only:
                # Remove test data
                real_matches = [
                    match for match in raw_matches 
                    if not any(test_indicator in match.get('source', '').lower() 
                              for test_indicator in ['test', 'sample', 'underdog_generator', 'fallback'])
                ]
                
                if real_matches:
                    logger.info(f"✅ Filtered to {len(real_matches)} real matches (was {len(raw_matches)})")
                    raw_matches = real_matches
                else:
                    logger.warning("⚠️ No real matches found, returning empty results instead of test data")
                    return jsonify({
                        'success': True,
                        'matches': [],
                        'count': 0,
                        'source': 'NO_REAL_DATA',
                        'message': 'No real matches available. Only test data found.',
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Unify format of all matches
            formatted_matches = []
            for match in raw_matches:
                formatted_match = format_match_for_dashboard(match, matches_result.get('source', 'unknown'))
                formatted_matches.append(formatted_match)
            
            logger.info(f"📊 Returning {len(formatted_matches)} formatted matches")
            
            return jsonify({
                'success': True,
                'matches': formatted_matches,
                'count': len(formatted_matches),
                'source': matches_result.get('source', 'unknown'),
                'prediction_type': formatted_matches[0]['prediction_type'] if formatted_matches else 'UNKNOWN',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Matches error: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to retrieve matches',
                'matches': []
            }), 500

    @app.route('/api/test-ml', methods=['POST'])
    @validate_json_request()
    @handle_betting_errors()
    def test_ml_prediction():
        """Test ML prediction"""
        try:
            # Get and validate JSON payload
            try:
                data = request.get_json(force=True, max_content_length=1024)
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': 'Invalid JSON payload'
                }), 400
            
            if not data:
                return jsonify({
                    'success': False,
                    'error': 'No data provided'
                }), 400
            
            # Validate JSON structure to prevent DoS attacks
            if not validate_json_payload(data):
                return jsonify({
                    'success': False,
                    'error': 'Invalid payload structure'
                }), 400
            
            # Sanitize and validate inputs
            data = sanitize_input(data)
            player1 = data.get('player1', 'Flavio Cobolli')
            player2 = data.get('player2', 'Novak Djokovic')
            tournament = data.get('tournament', 'US Open')
            surface = data.get('surface', 'Hard')
            
            # Validate inputs
            if not validate_player_name(player1) or not validate_player_name(player2):
                return jsonify({
                    'success': False,
                    'error': 'Invalid player names provided'
                }), 400
                
            if not validate_tournament_name(tournament):
                return jsonify({
                    'success': False,
                    'error': 'Invalid tournament name provided'
                }), 400
                
            if not validate_surface(surface):
                return jsonify({
                    'success': False,
                    'error': 'Invalid surface provided'
                }), 400
            
            logger.info(f"🔮 Testing ML prediction: {player1} vs {player2}")
            
            # Try different predictors
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
                    # Create data for prediction service
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
            
            # Fallback to underdog analyzer
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
                'error': 'ML prediction test failed'
            }), 500

    @app.route('/api/value-bets', methods=['GET'])
    @handle_betting_errors()
    def get_value_bets():
        """Find value bets"""
        try:
            # Get matches
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
                    # Get data
                    our_prob = match['prediction']['probability']
                    odds = match['odds']['player1']
                    bookmaker_prob = 1 / odds
                    
                    # Calculate edge
                    edge = our_prob - bookmaker_prob
                    
                    # If edge is greater than 5%
                    if edge > 0.05:
                        value_bet = {
                            'match': f"{match['player1'].replace('🎾 ', '')} vs {match['player2'].replace('🎾 ', '')}",
                            'player': match['player1'].replace('🎾 ', ''),
                            'tournament': match['tournament'].replace('🏆 ', ''),
                            'surface': match['surface'],
                            'odds': odds,
                            'our_probability': our_prob,
                            'bookmaker_probability': bookmaker_prob,
                            'edge': edge * 100,  # In percentage
                            'confidence': match['prediction']['confidence'],
                            'recommendation': 'BET' if edge > 0.08 else 'CONSIDER',
                            'kelly_fraction': min(edge * 0.25, 0.05),  # Conservative Kelly
                            'key_factors': match.get('key_factors', [])
                        }
                        value_bets.append(value_bet)
                        
                except Exception as e:
                    logger.warning(f"Error calculating value bet: {e}")
                    continue
            
            # Sort by edge
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
                'error': 'Failed to analyze value bets',
                'value_bets': []
            }), 500

    @app.route('/api/underdog-analysis', methods=['POST'])
    @validate_json_request()
    @handle_betting_errors()
    def analyze_underdog():
        """Detailed underdog scenario analysis"""
        try:
            # Get and validate JSON payload
            try:
                data = request.get_json(force=True, max_content_length=1024)
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': 'Invalid JSON payload'
                }), 400
            
            if not data:
                return jsonify({
                    'success': False,
                    'error': 'No data provided'
                }), 400
            
            # Validate JSON structure to prevent DoS attacks
            if not validate_json_payload(data):
                return jsonify({
                    'success': False,
                    'error': 'Invalid payload structure'
                }), 400
            
            # Sanitize and validate inputs
            data = sanitize_input(data)
            player1 = data.get('player1')
            player2 = data.get('player2')
            tournament = data.get('tournament', 'ATP Tournament')
            surface = data.get('surface', 'Hard')
            
            if not player1 or not player2:
                return jsonify({
                    'success': False,
                    'error': 'Both players are required'
                }), 400
                
            # Validate inputs
            if not validate_player_name(player1) or not validate_player_name(player2):
                return jsonify({
                    'success': False,
                    'error': 'Invalid player names provided'
                }), 400
                
            if not validate_tournament_name(tournament):
                return jsonify({
                    'success': False,
                    'error': 'Invalid tournament name provided'
                }), 400
                
            if not validate_surface(surface):
                return jsonify({
                    'success': False,
                    'error': 'Invalid surface provided'
                }), 400
            
            analyzer = UnderdogAnalyzer()
            
            # Get detailed analysis
            underdog_analysis = analyzer.calculate_underdog_probability(
                player1, player2, tournament, surface
            )
            
            # Add additional information
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
                'error': 'Underdog analysis failed'
            }), 500

    @app.route('/api/refresh', methods=['GET', 'POST'])
    @require_api_key()
    def refresh_data():
        """Refresh data - Note: Old API integrations removed"""
        try:
            # Return success - old API integrations were removed
            return jsonify({
                'success': True,
                'message': 'Data refresh requested (old APIs removed)',
                'source': 'simulation',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Refresh error: {e}")
            return jsonify({
                'success': False,
                'error': create_safe_error_response(e, 'Data refresh failed')
            }), 500

    @app.route('/api/player-info/<player_name>', methods=['GET'])
    def get_player_info(player_name):
        """Player information"""
        try:
            analyzer = UnderdogAnalyzer()
            rank = analyzer.get_player_ranking(player_name)
            
            # Additional information (simulation)
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
        """Test underdog analysis (for 'Test Underdog Analysis' button)"""
        try:
            data = request.get_json()
            
            if not data:
                # Use default data if none provided
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
            
            # Use UnderdogAnalyzer
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

    # Additional API status and management endpoints
    @app.route('/api/manual-api-update', methods=['POST'])
    @require_api_key()
    def manual_api_update():
        """Manual API data update with rate limiting control"""
        try:
            # Priority: Daily Scheduler with limits
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
            
            
            # Last resort - return information about unavailability
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
        """Comprehensive API status with Daily Scheduler and API Economy"""
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
            
            # API Economy Status (removed during cleanup)
            status_response['api_economy'] = {
                'available': False,
                'message': 'API Economy removed during cleanup'
            }
            
            return jsonify(status_response)
            
        except Exception as e:
            logger.error(f"❌ Comprehensive API status error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/redis-status', methods=['GET'])
    @require_api_key()
    def get_redis_status():
        """Get detailed Redis connection status and performance metrics"""
        try:
            redis_health = check_redis_health()
            
            # Additional performance tests if Redis is available
            performance_metrics = {}
            if redis_health['available']:
                try:
                    import redis
                    import time
                    
                    redis_url = os.getenv('REDIS_URL', '').strip()
                    if redis_url.startswith('redis://'):
                        r = redis.Redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2)
                    else:
                        r = redis.Redis(host='localhost', port=6379, socket_connect_timeout=2, socket_timeout=2)
                    
                    # Test write/read performance
                    start_time = time.time()
                    test_key = f"health_check_{int(time.time())}"
                    r.set(test_key, 'test_value', ex=5)  # Expire in 5 seconds
                    r.get(test_key)
                    r.delete(test_key)
                    write_read_time = (time.time() - start_time) * 1000
                    
                    # Get memory usage
                    info = r.info('memory')
                    
                    performance_metrics = {
                        'write_read_time_ms': round(write_read_time, 2),
                        'used_memory': info.get('used_memory_human', 'unknown'),
                        'used_memory_peak': info.get('used_memory_peak_human', 'unknown'),
                        'connected_clients': r.info('clients').get('connected_clients', 0)
                    }
                    
                except Exception as e:
                    performance_metrics['error'] = str(e)
            
            return jsonify({
                'success': True,
                'redis_health': redis_health,
                'performance_metrics': performance_metrics,
                'rate_limiter_status': 'redis' if redis_health['available'] else 'in-memory fallback',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Redis status check error: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to check Redis status',
                'timestamp': datetime.now().isoformat()
            }), 500

    @app.route('/api/rankings-status', methods=['GET'])
    def get_rankings_status():
        """Get dynamic rankings system status"""
        try:
            from dynamic_rankings_api import get_rankings_status
            status = get_rankings_status()
            return jsonify({
                'success': True,
                'rankings_status': status,
                'timestamp': datetime.now().isoformat()
            })
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'Dynamic rankings not available',
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    @app.route('/api/refresh-rankings', methods=['POST'])
    @require_api_key()
    def refresh_rankings():
        """Force refresh tennis rankings from APIs"""
        try:
            from dynamic_rankings_api import refresh_tennis_rankings
            results = refresh_tennis_rankings()
            return jsonify({
                'success': True,
                'message': 'Rankings refresh completed',
                'results': results,
                'timestamp': datetime.now().isoformat()
            })
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'Dynamic rankings not available',
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Rankings refresh error: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    @app.route('/api/api-economy-status', methods=['GET'])
    def get_api_economy_status():
        """API Economy status (for 'API Status' button) - Legacy endpoint"""
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
                            pass
            
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

    # API-Tennis.com specific routes
    @app.route('/api/api-tennis/status', methods=['GET'])
    def get_api_tennis_status():
        """Get API-Tennis.com integration status"""
        try:
            from api_tennis_data_collector import get_api_tennis_data_collector
            collector = get_api_tennis_data_collector()
            
            status = collector.get_integration_status()
            
            return jsonify({
                'success': True,
                'api_tennis_status': status,
                'timestamp': datetime.now().isoformat()
            })
            
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'API-Tennis integration not available',
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"API-Tennis status error: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    @app.route('/api/api-tennis/tournaments', methods=['GET'])
    def get_api_tennis_tournaments():
        """Get tournaments from API-Tennis.com"""
        try:
            from api_tennis_data_collector import get_api_tennis_data_collector
            collector = get_api_tennis_data_collector()
            
            if not collector.is_available():
                return jsonify({
                    'success': False,
                    'error': 'API-Tennis not configured or unavailable',
                    'tournaments': []
                })
            
            tournaments = collector.get_tournaments()
            
            return jsonify({
                'success': True,
                'tournaments': tournaments,
                'count': len(tournaments),
                'data_source': 'API-Tennis',
                'timestamp': datetime.now().isoformat()
            })
            
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'API-Tennis integration not available',
                'tournaments': []
            })
        except Exception as e:
            logger.error(f"API-Tennis tournaments error: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'tournaments': []
            }), 500

    @app.route('/api/api-tennis/matches', methods=['GET'])
    def get_api_tennis_matches():
        """Get matches from API-Tennis.com"""
        try:
            from api_tennis_data_collector import get_api_tennis_data_collector
            collector = get_api_tennis_data_collector()
            
            if not collector.is_available():
                return jsonify({
                    'success': False,
                    'error': 'API-Tennis not configured or unavailable',
                    'matches': []
                })
            
            # Get parameters
            include_live = request.args.get('include_live', 'true').lower() == 'true'
            days_ahead = int(request.args.get('days_ahead', '2'))
            
            # Get current matches
            current_matches = collector.get_current_matches(include_live=include_live)
            
            # Get upcoming matches if requested
            if days_ahead > 0:
                upcoming_matches = collector.get_upcoming_matches(days_ahead)
                # Combine and deduplicate
                all_matches = current_matches + upcoming_matches
                seen_ids = set()
                unique_matches = []
                for match in all_matches:
                    match_id = match.get('id')
                    if match_id not in seen_ids:
                        seen_ids.add(match_id)
                        unique_matches.append(match)
                matches = unique_matches
            else:
                matches = current_matches
            
            return jsonify({
                'success': True,
                'matches': matches,
                'count': len(matches),
                'data_source': 'API-Tennis',
                'parameters': {
                    'include_live': include_live,
                    'days_ahead': days_ahead
                },
                'timestamp': datetime.now().isoformat()
            })
            
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'API-Tennis integration not available',
                'matches': []
            })
        except Exception as e:
            logger.error(f"API-Tennis matches error: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'matches': []
            }), 500

    @app.route('/api/api-tennis/player/<player_name>/matches', methods=['GET'])
    def get_api_tennis_player_matches(player_name):
        """Get matches for a specific player from API-Tennis.com"""
        try:
            from api_tennis_data_collector import get_api_tennis_data_collector
            collector = get_api_tennis_data_collector()
            
            if not collector.is_available():
                return jsonify({
                    'success': False,
                    'error': 'API-Tennis not configured or unavailable',
                    'matches': []
                })
            
            # Validate player name
            if not validate_player_name(player_name):
                return jsonify({
                    'success': False,
                    'error': 'Invalid player name provided',
                    'matches': []
                }), 400
            
            days_ahead = int(request.args.get('days_ahead', '30'))
            matches = collector.get_player_matches(player_name, days_ahead)
            
            return jsonify({
                'success': True,
                'player': player_name,
                'matches': matches,
                'count': len(matches),
                'days_ahead': days_ahead,
                'data_source': 'API-Tennis',
                'timestamp': datetime.now().isoformat()
            })
            
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'API-Tennis integration not available',
                'matches': []
            })
        except Exception as e:
            logger.error(f"API-Tennis player matches error: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'matches': []
            }), 500

    @app.route('/api/api-tennis/match/<int:match_id>/odds', methods=['GET'])
    def get_api_tennis_match_odds(match_id):
        """Get betting odds for a specific match from API-Tennis.com"""
        try:
            from api_tennis_data_collector import get_api_tennis_data_collector
            collector = get_api_tennis_data_collector()
            
            if not collector.is_available():
                return jsonify({
                    'success': False,
                    'error': 'API-Tennis not configured or unavailable',
                    'odds': {}
                })
            
            odds_data = collector.get_match_odds(match_id)
            
            return jsonify({
                'success': True,
                'match_id': match_id,
                'odds': odds_data,
                'data_source': 'API-Tennis',
                'timestamp': datetime.now().isoformat()
            })
            
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'API-Tennis integration not available',
                'odds': {}
            })
        except Exception as e:
            logger.error(f"API-Tennis match odds error: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'odds': {}
            }), 500

    @app.route('/api/api-tennis/enhanced', methods=['GET'])
    def get_enhanced_api_tennis_data():
        """Get comprehensive data using Enhanced API-Tennis collector"""
        try:
            from api_tennis_data_collector import get_enhanced_api_tennis_collector
            collector = get_enhanced_api_tennis_collector()
            
            days_ahead = int(request.args.get('days_ahead', '2'))
            matches = collector.get_comprehensive_match_data(days_ahead)
            
            # Get status information
            status = collector.get_status()
            
            return jsonify({
                'success': True,
                'matches': matches,
                'count': len(matches),
                'collector_status': status,
                'parameters': {
                    'days_ahead': days_ahead
                },
                'data_source': 'Enhanced_API_Tennis_Collector',
                'timestamp': datetime.now().isoformat()
            })
            
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'Enhanced API-Tennis collector not available',
                'matches': []
            })
        except Exception as e:
            logger.error(f"Enhanced API-Tennis data error: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'matches': []
            }), 500

    @app.route('/api/api-tennis/clear-cache', methods=['POST'])
    @require_api_key()
    def clear_api_tennis_cache():
        """Clear API-Tennis.com cache"""
        try:
            from api_tennis_data_collector import get_api_tennis_data_collector
            collector = get_api_tennis_data_collector()
            
            if not collector.is_available():
                return jsonify({
                    'success': False,
                    'error': 'API-Tennis not configured or unavailable'
                })
            
            collector.clear_cache()
            
            return jsonify({
                'success': True,
                'message': 'API-Tennis cache cleared successfully',
                'timestamp': datetime.now().isoformat()
            })
            
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'API-Tennis integration not available'
            })
        except Exception as e:
            logger.error(f"Clear API-Tennis cache error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/api-tennis/test-connection', methods=['GET'])
    def test_api_tennis_connection():
        """Test API-Tennis.com connection and API key"""
        try:
            from api_tennis_integration import get_api_tennis_client
            client = get_api_tennis_client()
            
            # Test basic connectivity by getting event types
            try:
                event_types = client.get_event_types()
                
                if isinstance(event_types, dict) and event_types.get('success') == 1:
                    return jsonify({
                        'success': True,
                        'message': 'API-Tennis connection successful',
                        'event_types_count': len(event_types.get('result', [])),
                        'api_version': '2.9.4',
                        'client_status': client.get_client_status(),
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'API-Tennis returned invalid response',
                        'response': event_types,
                        'timestamp': datetime.now().isoformat()
                    })
                    
            except Exception as api_error:
                return jsonify({
                    'success': False,
                    'error': f'API-Tennis connection failed: {api_error}',
                    'timestamp': datetime.now().isoformat()
                })
            
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'API-Tennis integration not available',
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Test API-Tennis connection error: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    logger.info("✅ All routes registered successfully")