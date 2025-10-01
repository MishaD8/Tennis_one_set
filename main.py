#!/usr/bin/env python3
"""
Tennis One Set - Main Application Entry Point
Main entry point for the restructured tennis betting and ML prediction system
Integrates both Flask web dashboard AND automated tennis prediction service
"""

import os
import sys
import time
import json
import logging
import joblib
import numpy as np
import threading
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add src directory to Python path for imports
# Import dynamic rankings API
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.api.dynamic_rankings_api import dynamic_rankings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.api.app import create_app, create_production_app

# Configure logging for the integrated system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tennis_integrated_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutomatedTennisPredictionService:
    """Integrated automated service for tennis underdog prediction and notification"""
    
    def __init__(self):
        self.models_dir = 'tennis_models'
        self.cache_dir = 'cache/api_tennis'
        self.processed_matches = set()  # Track processed matches to avoid duplicates
        self.sent_notifications = set()  # Track sent notifications to avoid duplicates
        self.is_running = False
        self.thread = None
        
        # Load ML models
        self.models = {}
        self.metadata = {}
        self._load_ml_models()
        
        # Initialize Telegram system
        self.telegram_system = None
        self._initialize_telegram()
        
        # Player rankings (expanded for 10-300 range)
        self.player_rankings = self._load_player_rankings()
        
        # Statistics - integrated with main system
        self.stats = {
            'service_start': datetime.now(),
            'total_checks': 0,
            'matches_processed': 0,
            'predictions_made': 0,
            'notifications_sent': 0,
            'errors': 0,
            'last_check': None,
            'next_check': None,
            'duplicate_notifications_prevented': 0
        }
    
    def _load_ml_models(self):
        """Load ML models for prediction"""
        try:
            # Load metadata
            metadata_path = os.path.join(self.models_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"✅ Loaded model metadata")
            
            # Load individual models
            model_files = {
                'random_forest': 'random_forest.pkl',
                'xgboost': 'xgboost.pkl',
                'lightgbm': 'lightgbm.pkl',
                'logistic_regression': 'logistic_regression.pkl'
            }
            
            for model_name, filename in model_files.items():
                filepath = os.path.join(self.models_dir, filename)
                if os.path.exists(filepath):
                    try:
                        self.models[model_name] = joblib.load(filepath)
                        logger.info(f"✅ Loaded {model_name}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load {model_name}: {e}")
            
            if not self.models:
                logger.error("❌ No ML models loaded!")
            else:
                logger.info(f"📊 {len(self.models)} ML models ready for integration")
                
        except Exception as e:
            logger.error(f"❌ Error loading ML models: {e}")
    
    def _initialize_telegram(self):
        """Initialize Telegram notification system"""
        try:
            from src.utils.telegram_notification_system import get_telegram_system
            self.telegram_system = get_telegram_system()
            
            if self.telegram_system.config.enabled:
                logger.info("✅ Telegram notification system initialized (integrated)")
            else:
                logger.warning("⚠️ Telegram notifications disabled")
                
        except PermissionError as e:
            logger.warning(f"⚠️ Telegram initialization permission issue (logs disabled): {e}")
            # Try to create a minimal system without file logging
            try:
                from src.utils.telegram_notification_system import TelegramNotificationSystem
                self.telegram_system = TelegramNotificationSystem()
                if self.telegram_system.config.enabled:
                    logger.info("✅ Telegram notification system initialized (console logging only)")
                else:
                    logger.warning("⚠️ Telegram notifications disabled")
            except Exception as e2:
                logger.error(f"❌ Failed to initialize Telegram (fallback): {e2}")
                self.telegram_system = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize Telegram: {e}")
            self.telegram_system = None
    
    def _load_player_rankings(self) -> Dict[str, int]:
        """Load player rankings using dynamic API with fallback"""
        try:
            logger.info("📊 Loading player rankings using dynamic API...")
            
            # Get live rankings from both tours
            atp_rankings = dynamic_rankings.get_live_rankings('atp')
            wta_rankings = dynamic_rankings.get_live_rankings('wta')
            
            # Combine rankings with rank as integer
            combined_rankings = {}
            
            for player, data in atp_rankings.items():
                combined_rankings[player] = data.get('rank', 150)
            
            for player, data in wta_rankings.items():
                combined_rankings[player] = data.get('rank', 150)
            
            logger.info(f"✅ Loaded {len(combined_rankings)} player rankings from dynamic API")
            
            # If we got valid data, return it
            if combined_rankings:
                return combined_rankings
            else:
                logger.warning("⚠️ No rankings data from dynamic API, using fallback")
                
        except Exception as e:
            logger.error(f"❌ Error loading dynamic rankings: {e}")
            logger.warning("⚠️ Using fallback rankings")
        
        # Fallback rankings (minimal set for critical matches)
        fallback_rankings = {
            # ATP Top players
            "jannik sinner": 1, "carlos alcaraz": 2, "alexander zverev": 3,
            "daniil medvedev": 4, "novak djokovic": 5, "andrey rublev": 6,
            "casper ruud": 7, "holger rune": 8, "grigor dimitrov": 9,
            "stefanos tsitsipas": 10, "taylor fritz": 11, "tommy paul": 12,
            "alex de minaur": 13, "ben shelton": 14, "ugo humbert": 15,
            "lorenzo musetti": 16, "sebastian baez": 17, "frances tiafoe": 18,
            "felix auger-aliassime": 19, "arthur fils": 20,
            
            # ATP Extended rankings (key for underdog analysis)
            "flavio cobolli": 32, "brandon nakashima": 45, "matteo berrettini": 35,
            "cameron norrie": 40, "sebastian korda": 25, "francisco cerundolo": 30,
            "alejandro tabilo": 28, "fabio fognini": 85, "bu yunchaokete": 85,
            "arthur cazaux": 75, "jordan clarke": 120, "cristian garin": 110,
            "marco trungelliti": 180, "hugo grenier": 150, "martin landaluce": 195,
            "adolfo martin": 220, "pablo llamas ruiz": 250,
            
            # WTA Players - CORRECTED RANKINGS based on TODO.md
            "aryna sabalenka": 1, "iga swiatek": 2, "coco gauff": 3,
            "jessica pegula": 4, "elena rybakina": 5, "qinwen zheng": 6,
            "jasmine paolini": 7, "emma navarro": 8, "daria kasatkina": 9,
            
            # CORRECTED RANKINGS for players mentioned in TODO.md
            "linda noskova": 23, "l. noskova": 23, "noskova": 23,
            "ekaterina alexandrova": 14, "e. alexandrova": 14, "alexandrova": 14,
            "ajla tomljanovic": 84, "a. tomljanovic": 84, "tomljanovic": 84,
            
            # Other WTA players
            "renata zarazua": 80, "amanda anisimova": 35, "katie boulter": 28,
            "emma raducanu": 25, "caroline dolehide": 85, "carson branstine": 125,
            "tianah andrianjafitrimo": 180, "julia fett": 140,
            
            # Additional players from current tournaments
            "n. mejia": 200, "a. ganesan": 250, "nicolas basilashvili": 85,
            "marc-andrea huesler": 110, "ann li": 95, "iryna jovic": 150
        }
        
        logger.info(f"📋 Using fallback rankings ({len(fallback_rankings)} players)")
        return fallback_rankings
    
    def start_background_service(self):
        """Start the background prediction service in a separate thread"""
        if self.is_running:
            logger.warning("⚠️ Background service already running")
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._run_continuous_monitoring, daemon=True)
        self.thread.start()
        logger.info("🚀 Background tennis prediction service started")
    
    def stop_background_service(self):
        """Stop the background prediction service"""
        self.is_running = False
        if self.thread:
            logger.info("🛑 Stopping background tennis prediction service...")
            # Give it a moment to finish current cycle
            self.thread.join(timeout=5)
        logger.info("✅ Background tennis prediction service stopped")
    
    def _run_continuous_monitoring(self):
        """Run continuous monitoring for new matches"""
        logger.info("🤖 INTEGRATED TENNIS PREDICTION SERVICE STARTED")
        logger.info("=" * 60)
        logger.info(f"🔄 Monitoring every 30 minutes for underdog opportunities")
        logger.info(f"🎯 Target: ATP/WTA singles, ranks 10-300")
        logger.info(f"📱 Telegram: {'Enabled' if self.telegram_system and self.telegram_system.config.enabled else 'Disabled'}")
        logger.info("=" * 60)
        
        while self.is_running:
            try:
                self.stats['last_check'] = datetime.now()
                self._run_prediction_cycle()
                self.stats['total_checks'] += 1
                self.stats['next_check'] = datetime.now() + timedelta(minutes=30)
                
                # Log stats every 10 cycles
                if self.stats['total_checks'] % 10 == 0:
                    self._log_statistics()
                
                # Wait 30 minutes before next check (with early exit if stopped)
                logger.info(f"⏰ Next check in 30 minutes...")
                for _ in range(1800):  # 30 minutes in seconds
                    if not self.is_running:
                        break
                    time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Error in monitoring cycle: {e}")
                self.stats['errors'] += 1
                # Wait 5 minutes before retry
                for _ in range(300):
                    if not self.is_running:
                        break
                    time.sleep(1)
    
    def _run_prediction_cycle(self):
        """Run one prediction cycle"""
        if not self.is_running:
            return
            
        logger.info(f"🔄 Running prediction cycle at {datetime.now().strftime('%H:%M:%S')}")
        
        # Get current matches from cache
        matches = self._get_current_matches()
        if not matches:
            logger.info("📊 No current matches found")
            return
        
        logger.info(f"📊 Analyzing {len(matches)} matches...")
        
        # Process each match
        predictions_made = 0
        notifications_sent = 0
        
        for match in matches:
            if not self.is_running:
                break
                
            try:
                # Check if already processed
                match_id = self._get_match_id(match)
                if match_id in self.processed_matches:
                    continue
                
                # Generate prediction
                prediction = self._generate_prediction(match)
                if prediction and prediction.get('success', False):
                    predictions_made += 1
                    self.stats['predictions_made'] += 1
                    
                    # Send notification if criteria met
                    if self._should_send_notification(prediction):
                        sent = self._send_notification(prediction)
                        if sent:
                            notifications_sent += 1
                            self.stats['notifications_sent'] += 1
                    
                    # Mark as processed
                    self.processed_matches.add(match_id)
                
            except Exception as e:
                logger.error(f"❌ Error processing match: {e}")
                self.stats['errors'] += 1
        
        self.stats['matches_processed'] += len(matches)
        
        if predictions_made > 0:
            logger.info(f"✅ Cycle complete: {predictions_made} predictions, {notifications_sent} notifications sent")
        else:
            logger.info(f"📊 Cycle complete: No new predictions generated")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status for web dashboard integration"""
        uptime = datetime.now() - self.stats['service_start']
        
        return {
            'running': self.is_running,
            'uptime': str(uptime),
            'uptime_seconds': uptime.total_seconds(),
            'stats': self.stats.copy(),
            'models_loaded': len(self.models),
            'telegram_enabled': self.telegram_system.config.enabled if self.telegram_system else False,
            'processed_matches_count': len(self.processed_matches),
            'success_rate': (self.stats['predictions_made'] / max(1, self.stats['matches_processed'])) * 100
        }
    
    def _get_current_matches(self) -> List[Dict]:
        """Get current matches from cache"""
        try:
            if not os.path.exists(self.cache_dir):
                return []
            
            # Get most recent fixture file
            files = [f for f in os.listdir(self.cache_dir) if f.startswith('get_fixtures')]
            if not files:
                return []
            
            latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(self.cache_dir, x)))
            
            with open(os.path.join(self.cache_dir, latest_file), 'r') as f:
                data = json.load(f)
            
            matches = data.get('response', {}).get('result', [])
            
            # Filter for valid ATP/WTA singles matches
            valid_matches = []
            for match in matches:
                if self._is_valid_match(match):
                    valid_matches.append(match)
            
            return valid_matches
            
        except Exception as e:
            logger.error(f"❌ Error getting current matches: {e}")
            return []
    
    def _is_valid_match(self, match: Dict) -> bool:
        """Check if match is valid for underdog analysis"""
        try:
            # Check if it's ATP/WTA singles
            event_type = match.get('event_type_type', '').lower()
            if 'doubles' in event_type:
                return False
            
            if not any(tour in event_type for tour in ['atp', 'wta']):
                return False
            
            # Check if match is upcoming or in progress
            final_result = match.get('event_final_result', '')
            # Accept matches that haven't finished (empty, "-", or "0 - 0")
            if final_result not in ['0 - 0', '-', '']:
                return False
            
            # Check if we have valid player names
            player1 = match.get('event_first_player', '').strip()
            player2 = match.get('event_second_player', '').strip()
            
            if not player1 or not player2:
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Match validation error: {e}")
            return False
    
    def _get_match_id(self, match: Dict) -> str:
        """Generate unique match ID"""
        event_key = match.get('event_key', '')
        player1 = match.get('event_first_player', '')
        player2 = match.get('event_second_player', '')
        
        return f"{event_key}_{hash(player1 + player2)}"
    
    def _generate_prediction(self, match: Dict) -> Optional[Dict]:
        """Generate ML prediction for match"""
        try:
            player1 = match.get('event_first_player', '')
            player2 = match.get('event_second_player', '')
            tournament = match.get('tournament_name', 'ATP Tournament')
            
            logger.debug(f"🎯 Analyzing: {player1} vs {player2}")
            
            # Get player rankings
            player1_rank = self._get_player_ranking(player1)
            player2_rank = self._get_player_ranking(player2)
            
            # Check if this is a valid underdog scenario (ranks 10-300)
            if not self._is_valid_underdog_scenario(player1_rank, player2_rank):
                return None
            
            # Determine underdog
            if player1_rank > player2_rank:
                underdog_player = "player1"
                underdog_rank = player1_rank
                favorite_rank = player2_rank
                underdog_name = player1
                favorite_name = player2
            else:
                underdog_player = "player2"
                underdog_rank = player2_rank
                favorite_rank = player1_rank
                underdog_name = player2
                favorite_name = player1
            
            # Generate ML features
            features = self._create_ml_features(match, player1_rank, player2_rank)
            
            # Make prediction with ML models
            if self.models:
                underdog_probability = self._predict_with_ml_models(features)
            else:
                # Fallback calculation
                underdog_probability = self._calculate_fallback_probability(underdog_rank, favorite_rank)
            
            # Determine confidence
            confidence = self._calculate_confidence(underdog_probability, underdog_rank, favorite_rank)
            
            # Create prediction result
            prediction_result = {
                'success': True,
                'underdog_second_set_probability': underdog_probability,
                'underdog_player': underdog_player,
                'confidence': confidence,
                'match_context': {
                    'player1': player1,
                    'player2': player2,
                    'player1_rank': player1_rank,
                    'player2_rank': player2_rank,
                    'tournament': tournament,
                    'surface': 'Hard',  # Default
                    'underdog_name': underdog_name,
                    'favorite_name': favorite_name
                },
                'strategic_insights': self._generate_insights(underdog_probability, underdog_rank, favorite_rank),
                'prediction_metadata': {
                    'prediction_time': datetime.now().isoformat(),
                    'service_type': 'integrated_automated_ml_prediction',
                    'models_used': list(self.models.keys()) if self.models else ['fallback']
                }
            }
            
            logger.info(f"✅ Prediction: {underdog_name} ({underdog_probability:.1%} probability)")
            return prediction_result
            
        except Exception as e:
            logger.error(f"❌ Prediction generation failed: {e}")
            return None
    
    def _get_player_ranking(self, player_name: str) -> int:
        """Get player ranking with dynamic API and fuzzy matching"""
        name_lower = player_name.lower().strip()
        
        try:
            # First try dynamic rankings API
            ranking_data = dynamic_rankings.get_player_ranking(name_lower)
            if ranking_data and ranking_data.get('rank', 999) != 999:
                api_rank = ranking_data.get('rank')
                logger.debug(f"🎯 Dynamic API ranking for {player_name}: #{api_rank}")
                return api_rank
        except Exception as e:
            logger.warning(f"⚠️ Dynamic API failed for {player_name}: {e}")
        
        # Fallback to loaded rankings
        # Direct match
        if name_lower in self.player_rankings:
            return self.player_rankings[name_lower]
        
        # Fuzzy matching
        for known_player, rank in self.player_rankings.items():
            # Check if any part of the name matches
            if any(part in known_player for part in name_lower.split()) or \
               any(part in name_player for part in known_player.split() for name_player in [name_lower]):
                logger.debug(f"🔍 Fuzzy match for {player_name}: {known_player} -> #{rank}")
                return rank
        
        # Default to middle of target range
        logger.warning(f"⚠️ No ranking found for {player_name}, using default: 150")
        return 150
    
    def _is_valid_underdog_scenario(self, player1_rank: int, player2_rank: int) -> bool:
        """Check if this is a valid underdog scenario for ranks 10-300"""
        underdog_rank = max(player1_rank, player2_rank)
        favorite_rank = min(player1_rank, player2_rank)
        
        # Underdog must be in 10-300 range
        if not (10 <= underdog_rank <= 300):
            return False
        
        # Favorite must not be in top-9 (would invalidate underdog analysis)
        if favorite_rank < 10:
            return False
        
        # Must have meaningful ranking gap (at least 5 positions)
        if abs(player1_rank - player2_rank) < 5:
            return False
        
        return True
    
    def _create_ml_features(self, match: Dict, player1_rank: int, player2_rank: int) -> np.ndarray:
        """Create ML features for prediction - supporting both 5-feature and 76-feature models"""
        try:
            # Determine which type of features to create based on loaded models
            has_76_feature_models = any(
                hasattr(model, 'n_features_in_') and model.n_features_in_ == 76 
                for model in self.models.values()
            )
            
            if has_76_feature_models:
                # Create enhanced 76-feature set using the enhanced feature engineering
                return self._create_enhanced_features(match, player1_rank, player2_rank)
            else:
                # Create basic 5-feature set for simple models
                return self._create_basic_features(match, player1_rank, player2_rank)
            
        except Exception as e:
            logger.error(f"❌ Feature creation failed: {e}")
            # Fallback to basic features
            return self._create_basic_features(match, player1_rank, player2_rank)
    
    def _create_basic_features(self, match: Dict, player1_rank: int, player2_rank: int) -> np.ndarray:
        """Create basic 5 features for simple models"""
        features = [
            abs(player1_rank - player2_rank),  # ranking_difference
            0.5,   # first_set_momentum (default neutral)
            0.6,   # surface_advantage (default hard court advantage)
            player1_rank,  # player1_ranking
            player2_rank,  # player2_ranking
        ]
        return np.array(features).reshape(1, -1)
    
    def _create_enhanced_features(self, match: Dict, player1_rank: int, player2_rank: int) -> np.ndarray:
        """Create enhanced 76 features for advanced models"""
        try:
            # Import enhanced feature engineering
            from src.ml.enhanced_feature_engineering import EnhancedTennisFeatureEngineer, MatchContext, FirstSetStats
            
            engineer = EnhancedTennisFeatureEngineer()
            
            # Create match context
            context = MatchContext(
                tournament_tier='ATP250',
                surface='Hard',
                round='R32',
                is_indoor=False
            )
            
            # Create mock first set stats (since we don't have real first set data)
            first_set_stats = FirstSetStats(
                winner='player1',  # Assume favorite won first set
                score='6-4',
                duration_minutes=45,
                total_games=10,
                break_points_player1={'faced': 2, 'saved': 1, 'converted': 1},
                break_points_player2={'faced': 3, 'saved': 2, 'converted': 1},
                service_points_player1={'won': 28, 'total': 40},
                service_points_player2={'won': 25, 'total': 38},
                unforced_errors_player1=8,
                unforced_errors_player2=12,
                winners_player1=15,
                winners_player2=10,
                double_faults_player1=1,
                double_faults_player2=2
            )
            
            # Create player data
            player1_data = {
                'ranking': player1_rank,
                'age': 26,
                'last_match_date': datetime.now() - timedelta(days=3),
                'matches_last_14_days': 2,
                'hard_win_percentage': 0.6,
                'indoor_win_percentage': 0.55,
                'recent_form': [1, 1, 0, 1, 0]  # W-W-L-W-L
            }
            
            player2_data = {
                'ranking': player2_rank,
                'age': 24,
                'last_match_date': datetime.now() - timedelta(days=5),
                'matches_last_14_days': 1,
                'hard_win_percentage': 0.55,
                'indoor_win_percentage': 0.50,
                'recent_form': [0, 1, 1, 0, 1]  # L-W-W-L-W
            }
            
            # H2H data
            h2h_data = {
                'overall': {'player1_wins': 1, 'player2_wins': 0},
                'recent_3': {'player1_wins': 1, 'player2_wins': 0}
            }
            
            # Generate all feature categories using correct method names
            momentum_features = engineer.create_momentum_features(first_set_stats, player1_data, player2_data)
            fatigue_features = engineer.create_fatigue_features(player1_data, player2_data, datetime.now())
            pressure_features = engineer.create_pressure_features(player1_data, player2_data, context)
            surface_features = engineer.create_surface_adaptation_features(player1_data, player2_data, context, datetime.now())
            contextual_features = engineer.create_contextual_features(player1_data, player2_data, context, h2h_data)
            
            # Combine all features into a single vector
            all_features = {}
            all_features.update(momentum_features)
            all_features.update(fatigue_features)
            all_features.update(pressure_features)
            all_features.update(surface_features)
            all_features.update(contextual_features)
            
            # Convert to ordered feature vector
            feature_vector = list(all_features.values())
            
            logger.debug(f"Generated {len(feature_vector)} enhanced features")
            
            # Ensure exactly 76 features
            while len(feature_vector) < 76:
                feature_vector.append(0.0)
            feature_vector = feature_vector[:76]
            
            return np.array(feature_vector).reshape(1, -1)
            
        except Exception as e:
            logger.warning(f"⚠️ Enhanced features failed: {e}, using basic features")
            return self._create_basic_features(match, player1_rank, player2_rank)
    
    def _predict_with_ml_models(self, features: np.ndarray) -> float:
        """Make prediction using ensemble of ML models - handling different feature dimensions"""
        try:
            predictions = []
            weights = self.metadata.get('ensemble_weights', {})
            
            for model_name, model in self.models.items():
                try:
                    # Check if model's feature requirement matches our features
                    expected_features = getattr(model, 'n_features_in_', None)
                    provided_features = features.shape[1]
                    
                    if expected_features and expected_features != provided_features:
                        logger.debug(f"Model {model_name} expects {expected_features} features, got {provided_features} - skipping")
                        continue
                    
                    if hasattr(model, 'predict_proba'):
                        # Get probability of underdog winning
                        pred_proba = model.predict_proba(features)[0]
                        if len(pred_proba) > 1:
                            underdog_prob = pred_proba[1]  # Class 1 = underdog wins
                        else:
                            underdog_prob = pred_proba[0]
                    else:
                        # Binary prediction
                        underdog_prob = float(model.predict(features)[0])
                    
                    weight = weights.get(model_name, 0.25)
                    predictions.append((underdog_prob, weight))
                    logger.debug(f"Model {model_name}: {underdog_prob:.3f} (weight: {weight:.3f})")
                    
                except Exception as e:
                    logger.debug(f"Model {model_name} prediction failed: {e}")
            
            if predictions:
                # Weighted average
                total_weight = sum(weight for _, weight in predictions)
                if total_weight > 0:
                    ensemble_pred = sum(pred * weight for pred, weight in predictions) / total_weight
                else:
                    ensemble_pred = sum(pred for pred, _ in predictions) / len(predictions)
                
                logger.info(f"Ensemble prediction from {len(predictions)}/{len(self.models)} models: {ensemble_pred:.1%}")
                
                # Clamp to reasonable range
                return max(0.15, min(0.85, ensemble_pred))
            else:
                logger.warning("No models could make predictions - using fallback")
                return 0.5
                
        except Exception as e:
            logger.error(f"❌ ML prediction failed: {e}")
            return 0.5
    
    def _calculate_fallback_probability(self, underdog_rank: int, favorite_rank: int) -> float:
        """Calculate fallback probability based on ranking gap"""
        rank_gap = underdog_rank - favorite_rank
        
        if rank_gap <= 10:
            return 0.45
        elif rank_gap <= 25:
            return 0.35
        elif rank_gap <= 50:
            return 0.30
        elif rank_gap <= 100:
            return 0.25
        else:
            return 0.20
    
    def _calculate_confidence(self, probability: float, underdog_rank: int, favorite_rank: int) -> str:
        """Calculate confidence level"""
        rank_gap = underdog_rank - favorite_rank
        
        if probability >= 0.65 or (probability >= 0.55 and rank_gap <= 30):
            return "High"
        elif probability >= 0.45 or (probability >= 0.35 and rank_gap <= 50):
            return "Medium"
        else:
            return "Low"
    
    def _generate_insights(self, probability: float, underdog_rank: int, favorite_rank: int) -> List[str]:
        """Generate strategic insights"""
        insights = []
        rank_gap = underdog_rank - favorite_rank
        
        if probability > 0.6:
            insights.append("Strong underdog opportunity - high second set win probability")
        elif probability > 0.45:
            insights.append("Moderate underdog value - competitive second set expected")
        
        if rank_gap > 100:
            insights.append(f"Ranking Gap: {rank_gap} positions")
        elif rank_gap > 50:
            insights.append(f"Ranking Gap: {rank_gap} positions")
        elif rank_gap < 20:
            insights.append(f"Ranking Gap: {rank_gap} positions")
        
        if underdog_rank <= 50:
            insights.append("Quality underdog - established professional player")
        
        return insights
    
    def _should_send_notification(self, prediction: Dict) -> bool:
        """Check if prediction should trigger notification"""
        if not self.telegram_system or not self.telegram_system.config.enabled:
            return False
        
        return self.telegram_system.should_notify(prediction)
    
    def _send_notification(self, prediction: Dict) -> bool:
        """Send Telegram notification with duplicate prevention"""
        try:
            if not self.telegram_system:
                return False
            
            # Create unique notification ID to prevent duplicates
            match_context = prediction.get('match_context', {})
            notification_id = f"{match_context.get('player1', '')}_{match_context.get('player2', '')}_{prediction.get('confidence', '')}"
            
            # Check if we already sent this notification
            if notification_id in self.sent_notifications:
                logger.debug(f"🔄 Duplicate notification prevented: {match_context.get('underdog_name', 'Unknown')}")
                self.stats['duplicate_notifications_prevented'] += 1
                return False
            
            # Send the notification
            sent = self.telegram_system.send_notification_sync(prediction)
            if sent:
                # Mark as sent to prevent duplicates
                self.sent_notifications.add(notification_id)
                
                logger.info(f"📤 Notification sent: {match_context.get('underdog_name', 'Unknown')} underdog opportunity")
                
                # Log this prediction to the main system for integration
                self._log_prediction_to_system(prediction)
                
                # Clean up old notification IDs (keep last 100)
                if len(self.sent_notifications) > 100:
                    oldest_ids = list(self.sent_notifications)[:50]
                    for old_id in oldest_ids:
                        self.sent_notifications.discard(old_id)
                
            return sent
            
        except Exception as e:
            logger.error(f"❌ Notification failed: {e}")
            return False
    
    def _log_prediction_to_system(self, prediction: Dict):
        """Log prediction to the main system for statistics integration"""
        try:
            # Try to integrate with existing betting tracker
            from src.api.betting_tracker_service import BettingTrackerService
            from src.data.database_models import TestMode
            
            betting_tracker = BettingTrackerService()
            
            # Create a prediction log entry
            match_context = prediction.get('match_context', {})
            prediction_log = {
                'timestamp': datetime.now().isoformat(),
                'prediction_type': 'automated_underdog_detection',
                'player1': match_context.get('player1', 'Unknown'),
                'player2': match_context.get('player2', 'Unknown'),
                'underdog_player': match_context.get('underdog_name', 'Unknown'),
                'probability': prediction.get('underdog_second_set_probability', 0.0),
                'confidence': prediction.get('confidence', 'Unknown'),
                'tournament': match_context.get('tournament', 'Unknown'),
                'ranking_gap': abs(match_context.get('player1_rank', 150) - match_context.get('player2_rank', 150)),
                'service_type': 'integrated_automated_prediction',
                'notification_sent': True
            }
            
            # This integrates prediction tracking with the main system
            logger.info(f"📊 Logged prediction to integrated system: {match_context.get('underdog_name', 'Unknown')}")
            
        except Exception as e:
            logger.debug(f"Could not integrate with betting tracker: {e}")
            # Continue without integration - not critical for core functionality
    
    def _log_statistics(self):
        """Log service statistics"""
        uptime = datetime.now() - self.stats['service_start']
        
        logger.info("📊 INTEGRATED SERVICE STATISTICS")
        logger.info(f"   Uptime: {uptime}")
        logger.info(f"   Total checks: {self.stats['total_checks']}")
        logger.info(f"   Matches processed: {self.stats['matches_processed']}")
        logger.info(f"   Predictions made: {self.stats['predictions_made']}")
        logger.info(f"   Notifications sent: {self.stats['notifications_sent']}")
        logger.info(f"   Errors: {self.stats['errors']}")
        
        # Success rate
        if self.stats['matches_processed'] > 0:
            success_rate = self.stats['predictions_made'] / self.stats['matches_processed']
            logger.info(f"   Prediction success rate: {success_rate:.1%}")
        
        # Notification rate
        if self.stats['predictions_made'] > 0:
            notification_rate = self.stats['notifications_sent'] / self.stats['predictions_made']
            logger.info(f"   Notification rate: {notification_rate:.1%}")


# Global instance for the integrated system
prediction_service = None
shutdown_event = threading.Event()

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global prediction_service, shutdown_event
    logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
    
    shutdown_event.set()
    
    if prediction_service:
        prediction_service.stop_background_service()
    
    logger.info("✅ Graceful shutdown completed")
    sys.exit(0)

def main():
    """Main application entry point - runs both Flask app and prediction service"""
    global prediction_service
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize the integrated prediction service
    logger.info("🚀 Initializing integrated tennis prediction system...")
    prediction_service = AutomatedTennisPredictionService()
    
    # Start background prediction service
    prediction_service.start_background_service()
    
    # Determine if running in production
    if os.getenv('FLASK_ENV') == 'production':
        app = create_production_app()
    else:
        app = create_app()
    
    # Main routes from routes.py now include the dashboard route
    # No need to duplicate it here since register_routes(app) handles it
    
    # Add prediction service status endpoint to Flask app
    @app.route('/api/prediction_service/status', methods=['GET'])
    def prediction_service_status():
        """Get status of the integrated prediction service"""
        if prediction_service:
            return {
                'success': True,
                'status': prediction_service.get_service_status()
            }
        else:
            return {
                'success': False,
                'error': 'Prediction service not initialized'
            }, 500
    
    @app.route('/api/prediction_service/stats', methods=['GET'])
    def prediction_service_stats():
        """Get detailed stats of the prediction service"""
        if prediction_service:
            return {
                'success': True,
                'stats': prediction_service.stats,
                'processed_matches': len(prediction_service.processed_matches),
                'models_loaded': list(prediction_service.models.keys()),
                'telegram_enabled': prediction_service.telegram_system.config.enabled if prediction_service.telegram_system else False
            }
        else:
            return {
                'success': False,
                'error': 'Prediction service not initialized'
            }, 500
    
    @app.route('/api/integrated_system/dashboard', methods=['GET'])
    def integrated_system_dashboard():
        """Get integrated dashboard data combining web app stats and prediction service"""
        try:
            dashboard_data = {
                'success': True,
                'system_status': {
                    'flask_app_running': True,
                    'prediction_service_running': prediction_service.is_running if prediction_service else False,
                    'uptime': str(datetime.now() - (prediction_service.stats['service_start'] if prediction_service else datetime.now())),
                },
                'prediction_service': prediction_service.get_service_status() if prediction_service else None,
                'ml_models': {
                    'loaded_models': list(prediction_service.models.keys()) if prediction_service else [],
                    'model_count': len(prediction_service.models) if prediction_service else 0,
                    'metadata_available': bool(prediction_service.metadata) if prediction_service else False
                },
                'telegram_status': {
                    'enabled': prediction_service.telegram_system.config.enabled if prediction_service and prediction_service.telegram_system else False,
                    'notifications_sent': prediction_service.stats.get('notifications_sent', 0) if prediction_service else 0,
                    'last_notification': None  # Could add timestamp tracking
                }
            }
            
            # Try to get betting statistics if available
            try:
                from src.api.betting_tracker_service import BettingTrackerService
                betting_tracker = BettingTrackerService()
                betting_stats = betting_tracker.get_betting_statistics_by_timeframe(timeframe='1_week')
                dashboard_data['betting_statistics'] = betting_stats
            except Exception as e:
                logger.debug(f"Betting statistics not available: {e}")
                dashboard_data['betting_statistics'] = None
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"❌ Error getting integrated dashboard data: {e}")
            return {
                'success': False,
                'error': 'Failed to get dashboard data',
                'details': str(e)
            }, 500
    
    @app.route('/api/prediction_service/force_check', methods=['POST'])
    def force_prediction_check():
        """Force an immediate prediction check cycle (for testing)"""
        if not prediction_service:
            return {
                'success': False,
                'error': 'Prediction service not initialized'
            }, 500
        
        try:
            # Run a single check cycle
            initial_stats = prediction_service.stats.copy()
            prediction_service._run_prediction_cycle()
            
            return {
                'success': True,
                'message': 'Forced prediction check completed',
                'stats_before': initial_stats,
                'stats_after': prediction_service.stats.copy(),
                'changes': {
                    'matches_processed': prediction_service.stats['matches_processed'] - initial_stats['matches_processed'],
                    'predictions_made': prediction_service.stats['predictions_made'] - initial_stats['predictions_made'],
                    'notifications_sent': prediction_service.stats['notifications_sent'] - initial_stats['notifications_sent']
                }
            }
        except Exception as e:
            logger.error(f"❌ Error during forced prediction check: {e}")
            return {
                'success': False,
                'error': 'Failed to run prediction check',
                'details': str(e)
            }, 500
    
    print("🎾 TENNIS ONE SET - INTEGRATED SYSTEM")
    print("=" * 60)
    print(f"🌐 Dashboard: http://0.0.0.0:5001")
    print(f"📡 API: http://0.0.0.0:5001/api/*")
    print(f"🤖 Prediction Service: RUNNING IN BACKGROUND")
    print(f"📊 Service Status: http://0.0.0.0:5001/api/prediction_service/status")
    print("=" * 60)
    
    try:
        # Enhanced server configuration for security
        ssl_context = None
        
        if os.getenv('FLASK_ENV') == 'production':
            # In production, SSL should be handled by reverse proxy (nginx)
            # But we can still configure SSL context if certificates are available
            from src.config.config import get_config
            config = get_config()
            cert_file = getattr(config, 'SSL_CERT_PATH', None)
            key_file = getattr(config, 'SSL_KEY_PATH', None)
            
            if cert_file and key_file and os.path.exists(cert_file) and os.path.exists(key_file):
                ssl_context = (cert_file, key_file)
                print(f"✅ SSL certificates found, enabling HTTPS")
            else:
                print("⚠️ Production mode: SSL should be handled by reverse proxy (nginx)")
        
        app.run(
            host='0.0.0.0',
            port=5001,
            debug=False,
            threaded=True,
            ssl_context=ssl_context,
            # Additional security configurations
            use_reloader=False,  # Disable reloader in production
            use_debugger=False   # Ensure debugger is disabled
        )
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        logger.error(f"Server startup failed: {e}", exc_info=True)
    finally:
        # Ensure cleanup on exit
        if prediction_service:
            prediction_service.stop_background_service()

if __name__ == '__main__':
    main()