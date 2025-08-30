#!/usr/bin/env python3
"""
🤖 AUTOMATED TENNIS PREDICTION SERVICE

This service runs continuously to monitor tennis matches, generate underdog predictions,
and send Telegram notifications when strong opportunities are found.

Key Features:
- Monitors cached tennis data for new ATP/WTA singles matches
- Applies ML models to identify underdog opportunities (ranks 10-300)
- Sends Telegram notifications for predictions above 55% confidence
- Runs every 30 minutes to catch new matches
- Comprehensive logging and error handling

Author: Claude Code (Anthropic)
Date: 2025-08-21
"""

import sys
import os
import time
import json
import logging
import requests
import joblib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automated_tennis_predictions.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutomatedTennisPredictionService:
    """Automated service for tennis underdog prediction and notification"""
    
    def __init__(self):
        self.models_dir = 'tennis_models'
        self.cache_dir = 'cache/api_tennis'
        self.processed_matches = set()  # Track processed matches to avoid duplicates
        
        # Load ML models
        self.models = {}
        self.metadata = {}
        self._load_ml_models()
        
        # Initialize Telegram system
        self.telegram_system = None
        self._initialize_telegram()
        
        # Enhanced player rankings with comprehensive data (expanded for 10-300 range)
        try:
            self.enhanced_rankings = self._load_enhanced_player_rankings()
            self.player_rankings = {name: data['rank'] for name, data in self.enhanced_rankings.items()}
            logger.info(f"✅ Loaded enhanced rankings for {len(self.enhanced_rankings)} players")
        except Exception as e:
            logger.error(f"❌ Enhanced rankings failed: {e}")
            self.enhanced_rankings = {}
            self.player_rankings = self._load_player_rankings()
        
        # Statistics
        self.stats = {
            'service_start': datetime.now(),
            'total_checks': 0,
            'matches_processed': 0,
            'predictions_made': 0,
            'notifications_sent': 0,
            'errors': 0
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
                logger.info(f"📊 {len(self.models)} ML models ready")
                
        except Exception as e:
            logger.error(f"❌ Error loading ML models: {e}")
    
    def _initialize_telegram(self):
        """Initialize Telegram notification system"""
        try:
            from utils.telegram_notification_system import get_telegram_system
            self.telegram_system = get_telegram_system()
            
            if self.telegram_system.config.enabled:
                logger.info("✅ Telegram notification system initialized")
            else:
                logger.warning("⚠️ Telegram notifications disabled")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize Telegram: {e}")
    
    def _load_enhanced_player_rankings(self) -> Dict[str, Dict[str, Any]]:
        """Load enhanced player rankings from paid API tier with comprehensive data"""
        try:
            from enhanced_api_tennis_integration import EnhancedAPITennisIntegration
            enhanced_api = EnhancedAPITennisIntegration()
            
            # Get comprehensive rankings with points, movement, country data
            rankings_data = enhanced_api.get_enhanced_player_rankings()
            
            # Convert to name -> comprehensive data mapping
            enhanced_rankings = {}
            for player_id, data in rankings_data.items():
                if data.get('name') and data.get('rank'):
                    name_key = data['name'].lower().strip()
                    enhanced_rankings[name_key] = {
                        'rank': data['rank'],
                        'points': data.get('points', 1000),
                        'country': data.get('country', ''),
                        'tour': data.get('tour', 'ATP'),
                        'movement': data.get('movement', 'same'),
                        'player_id': player_id
                    }
                    
                    # Add alternate name formats for better matching
                    name_parts = name_key.split()
                    if len(name_parts) >= 2:
                        # Add "First Last" format
                        alt_name = f"{name_parts[0]} {name_parts[-1]}"
                        enhanced_rankings[alt_name] = enhanced_rankings[name_key].copy()
                        
                        # Add abbreviated format like "F. Last"
                        abbrev_name = f"{name_parts[0][0].lower()}. {name_parts[-1]}"
                        enhanced_rankings[abbrev_name] = enhanced_rankings[name_key].copy()
            
            logger.info(f"✅ Enhanced rankings loaded: {len(enhanced_rankings)} player mappings")
            logger.info(f"📊 Coverage: {len([r for r in enhanced_rankings.values() if r['tour'] == 'ATP'])} ATP, "
                       f"{len([r for r in enhanced_rankings.values() if r['tour'] == 'WTA'])} WTA players")
            
            return enhanced_rankings
            
        except Exception as e:
            logger.error(f"❌ Enhanced rankings failed, using fallback: {e}")
            return self._load_fallback_player_rankings()
    
    def _load_fallback_player_rankings(self) -> Dict[str, Dict[str, Any]]:
        """Fallback to existing ranking system with enhanced format"""
        fallback_rankings = {
            # ATP Top players  
            "jannik sinner": {'rank': 1, 'points': 9500, 'tour': 'ATP', 'movement': 'same', 'country': 'ITA'},
            "carlos alcaraz": {'rank': 2, 'points': 8800, 'tour': 'ATP', 'movement': 'same', 'country': 'ESP'},
            "alexander zverev": {'rank': 3, 'points': 7400, 'tour': 'ATP', 'movement': 'up', 'country': 'GER'},
            "daniil medvedev": {'rank': 4, 'points': 6500, 'tour': 'ATP', 'movement': 'down', 'country': 'RUS'},
            "novak djokovic": {'rank': 5, 'points': 5900, 'tour': 'ATP', 'movement': 'same', 'country': 'SRB'},
            "andrey rublev": {'rank': 6, 'points': 4600, 'tour': 'ATP', 'movement': 'up', 'country': 'RUS'},
            "casper ruud": {'rank': 7, 'points': 4200, 'tour': 'ATP', 'movement': 'down', 'country': 'NOR'},
            "holger rune": {'rank': 8, 'points': 3900, 'tour': 'ATP', 'movement': 'up', 'country': 'DEN'},
            "grigor dimitrov": {'rank': 9, 'points': 3600, 'tour': 'ATP', 'movement': 'same', 'country': 'BUL'},
            "stefanos tsitsipas": {'rank': 10, 'points': 3400, 'tour': 'ATP', 'movement': 'down', 'country': 'GRE'},
            "taylor fritz": {'rank': 11, 'points': 3200, 'tour': 'ATP', 'movement': 'up', 'country': 'USA'},
            "tommy paul": {'rank': 12, 'points': 3000, 'tour': 'ATP', 'movement': 'same', 'country': 'USA'},
            
            # Extended ATP with enhanced data
            "matteo arnaldi": {'rank': 64, 'points': 1200, 'tour': 'ATP', 'movement': 'up', 'country': 'ITA'},
            "gael monfils": {'rank': 45, 'points': 1600, 'tour': 'ATP', 'movement': 'down', 'country': 'FRA'},
            "alexandre muller": {'rank': 240, 'points': 200, 'tour': 'ATP', 'movement': 'same', 'country': 'FRA'},
            "francisco cerundolo": {'rank': 19, 'points': 2800, 'tour': 'ATP', 'movement': 'up', 'country': 'ARG'},
            
            # WTA Players with enhanced data
            "aryna sabalenka": {'rank': 1, 'points': 9500, 'tour': 'WTA', 'movement': 'same', 'country': 'BLR'},
            "iga swiatek": {'rank': 2, 'points': 8200, 'tour': 'WTA', 'movement': 'down', 'country': 'POL'},
            "coco gauff": {'rank': 3, 'points': 6900, 'tour': 'WTA', 'movement': 'up', 'country': 'USA'},
            "jessica pegula": {'rank': 4, 'points': 6200, 'tour': 'WTA', 'movement': 'same', 'country': 'USA'},
            "elena rybakina": {'rank': 5, 'points': 5800, 'tour': 'WTA', 'movement': 'up', 'country': 'KAZ'},
            "ekaterina alexandrova": {'rank': 14, 'points': 3200, 'tour': 'WTA', 'movement': 'same', 'country': 'RUS'},
            "anastasija sevastova": {'rank': 251, 'points': 150, 'tour': 'WTA', 'movement': 'down', 'country': 'LAT'},
        }
        
        # Add alternate formats
        enhanced_fallback = {}
        for name, data in fallback_rankings.items():
            enhanced_fallback[name] = data
            name_parts = name.split()
            if len(name_parts) >= 2:
                alt_name = f"{name_parts[0]} {name_parts[-1]}"
                enhanced_fallback[alt_name] = data.copy()
        
        logger.info(f"📋 Using fallback rankings: {len(enhanced_fallback)} mappings")
        return enhanced_fallback
    
    def _load_player_rankings(self) -> Dict[str, int]:
        """Load player rankings for backward compatibility"""
        try:
            enhanced_rankings = self._load_enhanced_player_rankings()
            # Convert to simple rank mapping for backward compatibility
            return {name: data['rank'] for name, data in enhanced_rankings.items()}
        except Exception as e:
            logger.error(f"❌ Failed to load rankings: {e}")
            return {}
    
    def run_continuous_monitoring(self):
        """Run continuous monitoring for new matches"""
        logger.info("🤖 AUTOMATED TENNIS PREDICTION SERVICE STARTED")
        logger.info("=" * 60)
        logger.info(f"🔄 Monitoring every 30 minutes for underdog opportunities")
        logger.info(f"🎯 Target: ATP/WTA singles, ranks 10-300")
        logger.info(f"📱 Telegram: {'Enabled' if self.telegram_system and self.telegram_system.config.enabled else 'Disabled'}")
        logger.info("=" * 60)
        
        while True:
            try:
                self._run_prediction_cycle()
                self.stats['total_checks'] += 1
                
                # Log stats every 10 cycles
                if self.stats['total_checks'] % 10 == 0:
                    self._log_statistics()
                
                # Wait 30 minutes before next check
                logger.info(f"⏰ Next check in 30 minutes...")
                time.sleep(1800)  # 30 minutes
                
            except KeyboardInterrupt:
                logger.info("🛑 Service stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in monitoring cycle: {e}")
                self.stats['errors'] += 1
                time.sleep(300)  # 5 minutes before retry
    
    def _run_prediction_cycle(self):
        """Run one prediction cycle"""
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
    
    def _get_enhanced_current_matches(self) -> List[Dict]:
        """Get current matches using enhanced API integration"""
        try:
            from enhanced_api_tennis_integration import EnhancedAPITennisIntegration
            enhanced_api = EnhancedAPITennisIntegration()
            
            # Get enhanced matches with comprehensive ranking data
            enhanced_matches = enhanced_api.get_enhanced_fixtures_with_rankings()
            
            # Convert to format expected by existing system
            compatible_matches = []
            for match in enhanced_matches:
                if match.get('enhanced_data', {}).get('is_underdog_scenario'):
                    compatible_match = {
                        'event_key': match.get('id'),
                        'event_first_player': match.get('player1', {}).get('name', ''),
                        'event_second_player': match.get('player2', {}).get('name', ''),
                        'tournament_name': match.get('tournament_name', ''),
                        'event_type_type': match.get('event_type', ''),
                        'event_final_result': '-',  # Upcoming match
                        'player1_rank': match.get('player1', {}).get('ranking'),
                        'player2_rank': match.get('player2', {}).get('ranking'),
                        'player1_points': match.get('player1', {}).get('points'),
                        'player2_points': match.get('player2', {}).get('points'),
                        'player1_movement': match.get('player1', {}).get('ranking_movement', 'same'),
                        'player2_movement': match.get('player2', {}).get('ranking_movement', 'same'),
                        'player1_country': match.get('player1', {}).get('country', ''),
                        'player2_country': match.get('player2', {}).get('country', ''),
                        'surface': match.get('surface', 'Hard'),
                        'enhanced_data': match.get('enhanced_data', {}),
                        'data_source': 'enhanced_api'
                    }
                    compatible_matches.append(compatible_match)
            
            logger.info(f"✅ Enhanced API provided {len(compatible_matches)} underdog opportunities")
            return compatible_matches
            
        except Exception as e:
            logger.warning(f"⚠️ Enhanced API unavailable, using fallback: {e}")
            return self._get_current_matches_fallback()
    
    def _get_current_matches_fallback(self) -> List[Dict]:
        """Fallback to cached match data"""
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
                    # Add basic enhanced data structure for compatibility
                    match['data_source'] = 'cache_fallback'
                    valid_matches.append(match)
            
            return valid_matches
            
        except Exception as e:
            logger.error(f"❌ Error getting current matches: {e}")
            return []
    
    def _get_current_matches(self) -> List[Dict]:
        """Get current matches - enhanced version with fallback"""
        try:
            # Always try enhanced API first
            enhanced_matches = self._get_enhanced_current_matches()
            if enhanced_matches:
                logger.info(f"✅ Using enhanced API data: {len(enhanced_matches)} matches")
                return enhanced_matches
            else:
                logger.warning("⚠️ Enhanced API returned no matches, trying fallback")
                return self._get_current_matches_fallback()
        except Exception as e:
            logger.error(f"❌ Enhanced match getting failed: {e}")
            return self._get_current_matches_fallback()
    
    def _is_valid_match(self, match: Dict) -> bool:
        """Check if match is valid for underdog analysis"""
        try:
            # Check if it's ATP/WTA singles
            event_type = match.get('event_type_type', '').lower()
            if 'doubles' in event_type:
                return False
            
            if not any(tour in event_type for tour in ['atp', 'wta']):
                return False
            
            # Check if match is upcoming or in progress (not finished)
            final_result = match.get('event_final_result', '')
            # Match is finished if it has a score like "3-1", "2-0", etc.
            # Pending matches typically show "-" or empty string
            if final_result and final_result not in ['-', '', '0 - 0']:
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
            
            logger.info(f"🎯 Analyzing: {player1} vs {player2}")
            
            # Get player rankings
            player1_rank = self._get_player_ranking(player1)
            player2_rank = self._get_player_ranking(player2)
            
            # Check if this is a valid underdog scenario (ranks 10-300)
            if not self._is_valid_underdog_scenario(player1_rank, player2_rank):
                logger.debug(f"⚠️ Not valid underdog scenario: ranks {player1_rank} vs {player2_rank}")
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
            
            # Generate enhanced ML features with comprehensive data
            try:
                # Check if we have enhanced API data
                if match.get('data_source') == 'enhanced_api':
                    logger.info("🔬 Using enhanced API feature engineering")
                    from enhanced_api_feature_engineering import EnhancedAPIFeatureEngineer
                    engineer = EnhancedAPIFeatureEngineer()
                    features = engineer.create_comprehensive_features(match)
                else:
                    logger.info("📊 Using enhanced ML features with cached data")
                    features = self._create_enhanced_ml_features(match, player1, player2)
            except Exception as e:
                logger.debug(f"Enhanced features unavailable: {e}")
                features = self._create_ml_features(match, player1_rank, player2_rank)
            
            # Make prediction with enhanced ML models
            if self.models:
                underdog_probability = self._predict_with_enhanced_ensemble(match, underdog_name, favorite_name)
            else:
                # Fallback calculation
                underdog_probability = self._calculate_fallback_probability(underdog_rank, favorite_rank)
            
            # Enhanced confidence calculation using comprehensive data
            confidence = self._calculate_enhanced_confidence(
                underdog_probability, underdog_rank, favorite_rank, match, underdog_name, favorite_name
            )
            
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
                'strategic_insights': self._generate_enhanced_insights(
                    underdog_probability, underdog_rank, favorite_rank, match, underdog_name, favorite_name
                ),
                'prediction_metadata': {
                    'prediction_time': datetime.now().isoformat(),
                    'service_type': 'automated_ml_prediction',
                    'models_used': list(self.models.keys()) if self.models else ['fallback']
                }
            }
            
            logger.info(f"✅ Prediction: {underdog_name} ({underdog_probability:.1%} probability)")
            return prediction_result
            
        except Exception as e:
            logger.error(f"❌ Prediction generation failed: {e}")
            return None
    
    def _get_enhanced_player_data(self, player_name: str) -> Dict[str, Any]:
        """Get comprehensive player data with enhanced matching"""
        name_lower = player_name.lower().strip()
        
        # Direct match in enhanced rankings
        if hasattr(self, 'enhanced_rankings') and name_lower in self.enhanced_rankings:
            return self.enhanced_rankings[name_lower]
        
        # Fuzzy matching in enhanced rankings
        if hasattr(self, 'enhanced_rankings'):
            for known_player, data in self.enhanced_rankings.items():
                # Check if any part of the name matches
                if any(part in known_player for part in name_lower.split()) or \
                   any(part in name_lower for part in known_player.split()):
                    return data
        
        # Fallback to basic ranking
        rank = self._get_player_ranking(player_name)
        return {
            'rank': rank,
            'points': rank * 10 if rank <= 100 else max(100, 1000 - rank),  # Estimated points
            'tour': 'ATP' if rank <= 300 else 'WTA',  # Basic estimation
            'movement': 'same',
            'country': ''
        }
    
    def _get_player_ranking(self, player_name: str) -> int:
        """Get player ranking with fuzzy matching - fallback method"""
        name_lower = player_name.lower().strip()
        
        # Direct match in basic rankings
        if name_lower in self.player_rankings:
            return self.player_rankings[name_lower]
        
        # Fuzzy matching in basic rankings
        for known_player, rank in self.player_rankings.items():
            if any(part in known_player for part in name_lower.split()) or \
               any(part in name_lower for part in known_player.split()):
                return rank
        
        # Default to middle of target range
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
    
    def _create_enhanced_ml_features(self, match: Dict, player1_name: str, player2_name: str) -> np.ndarray:
        """Create enhanced ML features using comprehensive player data"""
        try:
            # Use enhanced data from the match if available (from enhanced API)
            if match.get('data_source') == 'enhanced_api':
                player1_rank = match.get('player1_rank', 150)
                player2_rank = match.get('player2_rank', 150)
                player1_points = match.get('player1_points', player1_rank * 10)
                player2_points = match.get('player2_points', player2_rank * 10)
                player1_movement = match.get('player1_movement', 'same')
                player2_movement = match.get('player2_movement', 'same')
                player1_country = match.get('player1_country', '')
                player2_country = match.get('player2_country', '')
                surface = match.get('surface', 'Hard')
                
                logger.info(f"🔬 Enhanced API data - P1: {player1_name} (#{player1_rank}, {player1_points}pts, {player1_movement}) vs P2: {player2_name} (#{player2_rank}, {player2_points}pts, {player2_movement})")
            else:
                # Fallback to cached player data
                player1_data = self._get_enhanced_player_data(player1_name)
                player2_data = self._get_enhanced_player_data(player2_name)
                
                player1_rank = player1_data['rank']
                player2_rank = player2_data['rank']
                player1_points = player1_data.get('points', player1_rank * 10)
                player2_points = player2_data.get('points', player2_rank * 10)
                player1_movement = player1_data.get('movement', 'same')
                player2_movement = player2_data.get('movement', 'same')
                player1_country = player1_data.get('country', '')
                player2_country = player2_data.get('country', '')
                surface = match.get('surface', 'Hard')
            
            # Core ranking features
            features = [
                player1_rank,
                player2_rank,
                abs(player1_rank - player2_rank),  # ranking gap
                min(player1_rank, player2_rank),   # favorite rank
                max(player1_rank, player2_rank),   # underdog rank
            ]
            
            # Enhanced features with points differential
            if player1_points and player2_points and player2_points > 0:
                points_ratio = player1_points / player2_points
                features.append(min(max(points_ratio, 0.1), 10.0))  # Clamped ratio
            else:
                features.append(1.0)  # Neutral if no points data
            
            # Data quality score from enhanced API
            data_quality = match.get('enhanced_data', {}).get('data_quality_score', 0.5)
            features.append(data_quality)
            
            # Tournament importance scoring
            tournament_name = match.get('tournament_name', '').lower()
            if any(slam in tournament_name for slam in ['us open', 'wimbledon', 'french', 'australian']):
                importance = 4.0  # Grand Slam
            elif 'masters' in tournament_name or '1000' in tournament_name:
                importance = 3.0  # Masters
            elif '500' in tournament_name:
                importance = 2.5  # ATP 500
            else:
                importance = 2.0  # ATP 250 / WTA regular
            features.append(importance)
            
            # Surface features (enhanced detection)
            surface = match.get('surface', 'Hard').lower()
            features.extend([
                1.0 if surface == 'hard' else 0.0,
                1.0 if surface == 'clay' else 0.0,
                1.0 if surface == 'grass' else 0.0,
            ])
            
            # Form indicators from ranking movement (using actual API data)
            movement_score = 0.0
            if player1_movement == 'up':
                movement_score += 0.1
            elif player1_movement == 'down':
                movement_score -= 0.1
                
            if player2_movement == 'up':
                movement_score -= 0.1
            elif player2_movement == 'down':
                movement_score += 0.1
            
            features.append(movement_score)
            
            # Enhanced data quality from API
            if match.get('data_source') == 'enhanced_api':
                data_quality = match.get('enhanced_data', {}).get('data_quality_score', 0.9)
                ranking_gap = match.get('enhanced_data', {}).get('ranking_gap', abs(player1_rank - player2_rank))
                features.append(data_quality)
                features.append(min(ranking_gap / 100.0, 2.0))  # Normalized ranking gap
            else:
                features.append(0.7)  # Default data quality
                features.append(abs(player1_rank - player2_rank) / 100.0)
            
            # Country performance patterns with enhanced data
            european_countries = ['ESP', 'FRA', 'ITA', 'GER', 'SRB', 'SUI', 'AUT']
            south_american_countries = ['ARG', 'BRA', 'CHI', 'URU', 'COL']
            
            euro_advantage = 0.0
            if surface.lower() == 'clay':
                if player1_country in european_countries or player1_country in south_american_countries:
                    euro_advantage += 0.05
                if player2_country in european_countries or player2_country in south_american_countries:
                    euro_advantage -= 0.05
            features.append(euro_advantage)
            
            # Tournament importance from enhanced data
            tournament_name = match.get('tournament_name', '').lower()
            if 'enhanced_data' in match and match.get('data_source') == 'enhanced_api':
                # Use tournament importance from match data if available
                tournament_info = match.get('tournament_info', {})
                if tournament_info.get('is_grand_slam'):
                    importance = 4.0
                elif 'masters' in tournament_name or '1000' in tournament_name:
                    importance = 3.0
                elif '500' in tournament_name:
                    importance = 2.5
                else:
                    importance = 2.0
            else:
                # Default tournament importance calculation
                importance = 2.5
            features.append(importance)
            
            # Tour-specific features (determine from enhanced data or match context)
            if match.get('data_source') == 'enhanced_api':
                # Try to determine tour from event type or player rankings context
                event_type = match.get('event_type_type', '').lower()
                if 'atp' in event_type:
                    tour_atp, tour_wta = 1.0, 0.0
                elif 'wta' in event_type:
                    tour_atp, tour_wta = 0.0, 1.0
                else:
                    # Default based on ranking ranges (rough estimation)
                    avg_rank = (player1_rank + player2_rank) / 2
                    tour_atp, tour_wta = (1.0, 0.0) if avg_rank < 200 else (0.5, 0.5)
            else:
                # Use cached player data
                player1_data = self._get_enhanced_player_data(player1_name)
                tour_atp = 1.0 if player1_data.get('tour') == 'ATP' else 0.0
                tour_wta = 1.0 if player1_data.get('tour') == 'WTA' else 0.0
            
            features.extend([tour_atp, tour_wta])
            
            # Pad to expected feature count if needed
            expected_features = self.metadata.get('feature_columns', [])
            if expected_features:
                while len(features) < len(expected_features):
                    features.append(0.0)
                features = features[:len(expected_features)]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.warning(f"Enhanced features failed, using fallback: {e}")
            return self._create_ml_features(match, 
                                          self._get_player_ranking(player1_name),
                                          self._get_player_ranking(player2_name))
    
    def _create_ml_features(self, match: Dict, player1_rank: int, player2_rank: int) -> np.ndarray:
        """Create basic ML features for prediction (fallback method)"""
        try:
            # Basic features based on ranking and tournament
            features = [
                player1_rank,
                player2_rank,
                abs(player1_rank - player2_rank),  # ranking gap
                min(player1_rank, player2_rank),   # favorite rank
                max(player1_rank, player2_rank),   # underdog rank
                1.0,  # hard court (default)
                0.0,  # clay court
                0.0,  # grass court
                2.5,  # tournament importance (ATP 250 level)
                1.0,  # first set assumed close
                0.5,  # break points saved
                0.65, # first serve percentage
                0.0,  # had tiebreak
            ]
            
            # Pad to expected feature count (76 from metadata)
            expected_features = self.metadata.get('feature_columns', [])
            if expected_features:
                while len(features) < len(expected_features):
                    features.append(0.0)
                features = features[:len(expected_features)]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"❌ Feature creation failed: {e}")
            # Return minimal feature set
            return np.array([player1_rank, player2_rank, abs(player1_rank - player2_rank)]).reshape(1, -1)
    
    def _predict_with_enhanced_ensemble(self, match: Dict, underdog_name: str, favorite_name: str) -> float:
        """Make prediction using enhanced ensemble with comprehensive API data"""
        try:
            # Import enhanced ML predictor
            from src.models.enhanced_ml_integration import EnhancedMLPredictor
            
            # Create enhanced predictor instance
            enhanced_predictor = EnhancedMLPredictor(self.models_dir)
            
            # Get enhanced player data
            underdog_data = self._get_enhanced_player_data(underdog_name)
            favorite_data = self._get_enhanced_player_data(favorite_name)
            
            # Create enhanced features
            features = enhanced_predictor.create_enhanced_features(match, underdog_data, favorite_data)
            
            # Get enhanced ensemble prediction
            prediction_result = enhanced_predictor.predict_with_ensemble(features)
            
            if prediction_result.get('success'):
                probability = prediction_result['probability']
                confidence_level = prediction_result['confidence_level']
                model_agreement = prediction_result.get('model_agreement', 0.5)
                
                logger.info(f"🤖 Enhanced ML prediction: {probability:.3f} (confidence: {confidence_level}, agreement: {model_agreement:.3f})")
                
                # Store enhanced prediction metadata for insights
                if hasattr(self, 'last_prediction_metadata'):
                    self.last_prediction_metadata = {
                        'enhanced_prediction': True,
                        'models_used': prediction_result.get('models_used', []),
                        'model_agreement': model_agreement,
                        'confidence_level': confidence_level,
                        'individual_probabilities': prediction_result.get('individual_probabilities', {})
                    }
                
                return max(0.15, min(0.85, probability))
            else:
                logger.warning(f"Enhanced prediction failed: {prediction_result.get('error')}")
                return self._predict_with_ml_models_fallback(match, underdog_name, favorite_name)
                
        except Exception as e:
            logger.warning(f"Enhanced ensemble prediction failed: {e}")
            return self._predict_with_ml_models_fallback(match, underdog_name, favorite_name)
    
    def _predict_with_ml_models_fallback(self, match: Dict, underdog_name: str, favorite_name: str) -> float:
        """Fallback ML prediction using basic features"""
        try:
            # Get basic rankings
            underdog_rank = self._get_player_ranking(underdog_name)
            favorite_rank = self._get_player_ranking(favorite_name)
            
            # Create basic features
            features = self._create_ml_features(match, underdog_rank, favorite_rank)
            
            return self._predict_with_ml_models(features)
            
        except Exception as e:
            logger.error(f"❌ Fallback ML prediction failed: {e}")
            return 0.5

    def _predict_with_ml_models(self, features: np.ndarray) -> float:
        """Make prediction using ensemble of ML models (basic method)"""
        try:
            predictions = []
            weights = self.metadata.get('ensemble_weights', {})
            
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        # Get probability of underdog winning
                        pred_proba = model.predict_proba(features)[0, 1] if len(model.predict_proba(features)[0]) > 1 else model.predict_proba(features)[0, 0]
                    else:
                        # Binary prediction
                        pred_proba = float(model.predict(features)[0])
                    
                    weight = weights.get(model_name, 0.25)
                    predictions.append((pred_proba, weight))
                    
                except Exception as e:
                    logger.debug(f"Model {model_name} prediction failed: {e}")
            
            if predictions:
                # Weighted average
                total_weight = sum(weight for _, weight in predictions)
                if total_weight > 0:
                    ensemble_pred = sum(pred * weight for pred, weight in predictions) / total_weight
                else:
                    ensemble_pred = sum(pred for pred, _ in predictions) / len(predictions)
                
                # Clamp to reasonable range
                return max(0.15, min(0.85, ensemble_pred))
            else:
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
    
    def _calculate_enhanced_confidence(self, probability: float, underdog_rank: int, favorite_rank: int, 
                                     match: Dict, underdog_name: str, favorite_name: str) -> str:
        """Calculate enhanced confidence level using comprehensive data"""
        try:
            rank_gap = underdog_rank - favorite_rank
            base_confidence = probability
            
            # Get enhanced player data for confidence adjustments
            underdog_data = self._get_enhanced_player_data(underdog_name)
            favorite_data = self._get_enhanced_player_data(favorite_name)
            
            # Confidence boosters
            confidence_boost = 0.0
            
            # Data quality boost
            data_quality = match.get('enhanced_data', {}).get('data_quality_score', 0.5)
            if data_quality > 0.8:
                confidence_boost += 0.05
            
            # Form boost (underdog on the rise)
            if underdog_data.get('movement') == 'up':
                confidence_boost += 0.03
            if favorite_data.get('movement') == 'down':
                confidence_boost += 0.02
                
            # Points differential consideration
            underdog_points = underdog_data.get('points', 0)
            favorite_points = favorite_data.get('points', 0)
            if underdog_points and favorite_points and favorite_points > 0:
                points_ratio = underdog_points / favorite_points
                if points_ratio > 0.7:  # Points closer than rankings suggest
                    confidence_boost += 0.04
            
            # Tournament importance (more predictable in major events)
            tournament_name = match.get('tournament_name', '').lower()
            if any(slam in tournament_name for slam in ['us open', 'wimbledon', 'french', 'australian']):
                confidence_boost += 0.02  # Grand Slams are more predictable
            
            # Surface advantage
            surface = match.get('surface', 'Hard').lower()
            underdog_country = underdog_data.get('country', '')
            if surface == 'clay' and underdog_country in ['ESP', 'FRA', 'ITA', 'ARG']:
                confidence_boost += 0.03
            
            # Apply confidence boost
            adjusted_probability = base_confidence + confidence_boost
            
            # Enhanced confidence thresholds
            if adjusted_probability >= 0.70 or (adjusted_probability >= 0.60 and rank_gap <= 25):
                return "High"
            elif adjusted_probability >= 0.50 or (adjusted_probability >= 0.40 and rank_gap <= 40):
                return "Medium"
            else:
                return "Low"
                
        except Exception as e:
            logger.debug(f"Enhanced confidence calculation failed: {e}")
            return self._calculate_confidence(probability, underdog_rank, favorite_rank)
    
    def _calculate_confidence(self, probability: float, underdog_rank: int, favorite_rank: int) -> str:
        """Calculate basic confidence level (fallback method)"""
        rank_gap = underdog_rank - favorite_rank
        
        if probability >= 0.65 or (probability >= 0.55 and rank_gap <= 30):
            return "High"
        elif probability >= 0.45 or (probability >= 0.35 and rank_gap <= 50):
            return "Medium"
        else:
            return "Low"
    
    def _generate_enhanced_insights(self, probability: float, underdog_rank: int, favorite_rank: int,
                                  match: Dict, underdog_name: str, favorite_name: str) -> List[str]:
        """Generate enhanced strategic insights using comprehensive data"""
        try:
            insights = []
            rank_gap = underdog_rank - favorite_rank
            
            # Get enhanced player data
            underdog_data = self._get_enhanced_player_data(underdog_name)
            favorite_data = self._get_enhanced_player_data(favorite_name)
            
            # Probability-based insights
            if probability > 0.65:
                insights.append("🔥 STRONG underdog opportunity - high second set win probability")
            elif probability > 0.50:
                insights.append("⚡ GOOD underdog value - competitive second set expected")
            elif probability > 0.35:
                insights.append("📊 Moderate underdog potential - decent second set chances")
            
            # Enhanced ranking analysis
            underdog_points = underdog_data.get('points', 0)
            favorite_points = favorite_data.get('points', 0)
            
            if underdog_points and favorite_points and favorite_points > 0:
                points_ratio = underdog_points / favorite_points
                if points_ratio > 0.8:
                    insights.append(f"📈 Points gap smaller than rank gap suggests ({points_ratio:.2f} ratio)")
                elif points_ratio < 0.3:
                    insights.append(f"📉 Significant points differential ({points_ratio:.2f} ratio)")
            
            # Form insights
            underdog_movement = underdog_data.get('movement', 'same')
            favorite_movement = favorite_data.get('movement', 'same')
            
            if underdog_movement == 'up' and favorite_movement == 'down':
                insights.append("📈 Perfect form scenario: Underdog rising, favorite declining")
            elif underdog_movement == 'up':
                insights.append("📈 Underdog showing positive form (ranking rising)")
            elif favorite_movement == 'down':
                insights.append("📉 Favorite struggling with recent form (ranking declining)")
            
            # Tournament and surface insights
            tournament_name = match.get('tournament_name', '').lower()
            surface = match.get('surface', 'Hard').lower()
            
            if any(slam in tournament_name for slam in ['us open', 'wimbledon', 'french', 'australian']):
                insights.append("🏆 Grand Slam match - higher unpredictability factor")
            
            underdog_country = underdog_data.get('country', '')
            if surface == 'clay' and underdog_country in ['ESP', 'FRA', 'ITA', 'ARG']:
                insights.append(f"🎾 Clay court advantage: {underdog_country} player on preferred surface")
            
            # Data quality insights
            data_quality = match.get('enhanced_data', {}).get('data_quality_score', 0.5)
            if data_quality > 0.9:
                insights.append("✅ Excellent data quality - high confidence in analysis")
            elif data_quality < 0.6:
                insights.append("⚠️ Limited data quality - exercise caution")
            
            # Professional level insights
            if underdog_rank <= 50:
                insights.append("⭐ Quality underdog - top 50 professional player")
            elif underdog_rank <= 100:
                insights.append("💪 Solid underdog - top 100 ranking")
            elif underdog_rank > 200:
                insights.append("🎯 Deep underdog - high risk/reward scenario")
            
            # Gap analysis
            if rank_gap > 150:
                insights.append(f"🏔️ Major ranking gap: {rank_gap} positions")
            elif rank_gap > 75:
                insights.append(f"📊 Significant gap: {rank_gap} positions")
            elif rank_gap < 25:
                insights.append(f"🤝 Close ranking match: {rank_gap} positions apart")
            
            return insights
            
        except Exception as e:
            logger.debug(f"Enhanced insights generation failed: {e}")
            return self._generate_insights(probability, underdog_rank, favorite_rank)
    
    def _generate_insights(self, probability: float, underdog_rank: int, favorite_rank: int) -> List[str]:
        """Generate basic strategic insights (fallback method)"""
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
        """Send Telegram notification"""
        try:
            if self.telegram_system:
                sent = self.telegram_system.send_notification_sync(prediction)
                if sent:
                    match_context = prediction.get('match_context', {})
                    logger.info(f"📤 Notification sent: {match_context.get('underdog_name', 'Unknown')} underdog opportunity")
                return sent
            return False
        except Exception as e:
            logger.error(f"❌ Notification failed: {e}")
            return False
    
    def _log_statistics(self):
        """Log service statistics"""
        uptime = datetime.now() - self.stats['service_start']
        
        logger.info("📊 SERVICE STATISTICS")
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
    
    def run_single_check(self):
        """Run a single prediction check (for testing)"""
        logger.info("🧪 Running single prediction check...")
        self._run_prediction_cycle()
        self._log_statistics()

def main():
    """Main service entry point"""
    service = AutomatedTennisPredictionService()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Run single check for testing
        service.run_single_check()
    else:
        # Run continuous monitoring
        service.run_continuous_monitoring()

if __name__ == "__main__":
    main()