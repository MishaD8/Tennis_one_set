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
        
        # Player rankings (expanded for 10-300 range)
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
    
    def _load_player_rankings(self) -> Dict[str, int]:
        """Load player rankings for underdog analysis"""
        # Comprehensive rankings for ATP/WTA players (ranks 10-300 focus)
        rankings = {
            # ATP Top players  
            "jannik sinner": 1, "carlos alcaraz": 2, "alexander zverev": 3,
            "daniil medvedev": 4, "novak djokovic": 5, "andrey rublev": 6,
            "casper ruud": 7, "holger rune": 8, "grigor dimitrov": 9,
            "stefanos tsitsipas": 10, "taylor fritz": 11, "tommy paul": 12,
            "alex de minaur": 13, "ben shelton": 14, "ugo humbert": 15,
            "lorenzo musetti": 16, "sebastian baez": 17, "frances tiafoe": 18,
            "felix auger-aliassime": 19, "arthur fils": 20,
            
            # ATP Extended rankings (updated based on recent matches)
            "flavio cobolli": 32, "brandon nakashima": 45, "matteo berrettini": 35,
            "cameron norrie": 40, "sebastian korda": 25, "francisco cerundolo": 19,  # Fixed from 30
            "alejandro tabilo": 28, "fabio fognini": 85, "bu yunchaokete": 85,
            "arthur cazaux": 75, "jordan clarke": 120, "cristian garin": 110,
            "marco trungelliti": 180, "hugo grenier": 150, "martin landaluce": 195,
            "adolfo martin": 220, "pablo llamas ruiz": 250,
            "matteo arnaldi": 64, "m. arnaldi": 64,  # Fixed Arnaldi ranking
            "gael monfils": 45, "g. monfils": 45, "roman safiullin": 62, "r. safiullin": 62,
            "giovanni mpetshi perricard": 95, "g. mpetshi perricard": 95,
            "alexandre muller": 240, "a. muller": 240,  # Fixed Muller ranking
            
            # WTA Players (updated rankings)
            "aryna sabalenka": 1, "iga swiatek": 2, "coco gauff": 3,
            "jessica pegula": 4, "elena rybakina": 5, "qinwen zheng": 6,
            "jasmine paolini": 7, "emma navarro": 8, "daria kasatkina": 9,
            "ekaterina alexandrova": 14, "e. alexandrova": 14,  # Fixed from fallback
            "anastasija sevastova": 251, "a. sevastova": 251,  # Fixed Sevastova ranking
            "amanda anisimova": 35, "a. anisimova": 35, "kimberly birrell": 95, "k. birrell": 95,
            "katie boulter": 28, "k. boulter": 28, "marta kostyuk": 45, "m. kostyuk": 45,
            "caroline dolehide": 85, "c. dolehide": 85, "wang xinyu": 55, "xin. wang": 55,
            "laura siegemund": 125, "l. siegemund": 125, "diana shnaider": 25, "d. shnaider": 25,
            "emma raducanu": 25, "carson branstine": 125,
            
            # Additional players from current tournaments
            "n. mejia": 200, "a. ganesan": 250, "nicolas basilashvili": 85,
            "marc-andrea huesler": 110, "ann li": 95, "iryna jovic": 150,
            "billy harris": 120, "b. harris": 120, "f. cerundolo": 19, "f.cerundolo": 19
        }
        
        return rankings
    
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
                    'service_type': 'automated_ml_prediction',
                    'models_used': list(self.models.keys()) if self.models else ['fallback']
                }
            }
            
            logger.info(f"✅ Prediction: {underdog_name} ({underdog_probability:.1%} probability)")
            return prediction_result
            
        except Exception as e:
            logger.error(f"❌ Prediction generation failed: {e}")
            return None
    
    def _get_player_ranking(self, player_name: str) -> int:
        """Get player ranking with fuzzy matching"""
        name_lower = player_name.lower().strip()
        
        # Direct match
        if name_lower in self.player_rankings:
            return self.player_rankings[name_lower]
        
        # Fuzzy matching
        for known_player, rank in self.player_rankings.items():
            # Check if any part of the name matches
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
    
    def _create_ml_features(self, match: Dict, player1_rank: int, player2_rank: int) -> np.ndarray:
        """Create ML features for prediction"""
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
    
    def _predict_with_ml_models(self, features: np.ndarray) -> float:
        """Make prediction using ensemble of ML models"""
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