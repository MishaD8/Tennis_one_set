#!/usr/bin/env python3
"""
API-Tennis ML Integration
Connects API-Tennis.com data with the ML prediction system
"""

import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Import API-Tennis integration
from api_tennis_integration import initialize_api_tennis_client, TennisMatch, TennisPlayer

# Import existing ML components
try:
    from enhanced_ml_integration import EnhancedMLOrchestrator
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    from second_set_prediction_service import SecondSetPredictionService
    PREDICTION_SERVICE_AVAILABLE = True
except ImportError:
    PREDICTION_SERVICE_AVAILABLE = False

try:
    from tennis_prediction_module import TennisPredictionModule
    TENNIS_PREDICTION_AVAILABLE = True
except ImportError:
    TENNIS_PREDICTION_AVAILABLE = False

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APITennisMLIntegration:
    """
    Integration layer between API-Tennis.com and ML prediction system
    """
    
    def __init__(self):
        """Initialize the integration system"""
        self.api_client = None
        self.ml_orchestrator = None
        self.prediction_service = None
        self.tennis_predictor = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components"""
        try:
            # Initialize API-Tennis client
            api_key = os.getenv('API_TENNIS_KEY')
            if api_key:
                self.api_client = initialize_api_tennis_client(api_key=api_key)
                logger.info("API-Tennis client initialized successfully")
            else:
                logger.warning("API_TENNIS_KEY not found in environment variables")
            
            # Initialize ML components
            if ML_AVAILABLE:
                self.ml_orchestrator = EnhancedMLOrchestrator()
                logger.info("ML orchestrator initialized")
            
            if PREDICTION_SERVICE_AVAILABLE:
                self.prediction_service = SecondSetPredictionService()
                logger.info("Prediction service initialized")
            
            if TENNIS_PREDICTION_AVAILABLE:
                self.tennis_predictor = TennisPredictionModule()
                logger.info("Tennis predictor initialized")
                
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get status of API and ML components"""
        status = {
            'api_tennis': {
                'available': self.api_client is not None,
                'status': self.api_client.get_client_status() if self.api_client else None
            },
            'ml_components': {
                'orchestrator_available': ML_AVAILABLE and self.ml_orchestrator is not None,
                'prediction_service_available': PREDICTION_SERVICE_AVAILABLE and self.prediction_service is not None,
                'tennis_predictor_available': TENNIS_PREDICTION_AVAILABLE and self.tennis_predictor is not None
            }
        }
        return status
    
    def fetch_live_data(self) -> Dict[str, Any]:
        """Fetch live tennis data from API-Tennis"""
        if not self.api_client:
            return {'error': 'API client not initialized'}
        
        try:
            # Get today's matches
            today_matches = self.api_client.get_today_matches()
            
            # Get upcoming matches
            upcoming_matches = self.api_client.get_upcoming_matches(days_ahead=7)
            
            # Get live matches
            live_matches = self.api_client.get_live_matches()
            
            return {
                'success': True,
                'data': {
                    'today_matches_count': len(today_matches),
                    'upcoming_matches_count': len(upcoming_matches),
                    'live_matches_count': len(live_matches),
                    'today_matches': [match.to_dict() for match in today_matches[:10]],
                    'upcoming_matches': [match.to_dict() for match in upcoming_matches[:10]],
                    'live_matches': [match.to_dict() for match in live_matches[:10]]
                }
            }
            
        except Exception as e:
            logger.error(f"Error fetching live data: {e}")
            return {'error': str(e)}
    
    def convert_api_match_to_ml_format(self, match: TennisMatch) -> Dict[str, Any]:
        """Convert API-Tennis match data to ML model input format"""
        try:
            # Extract player information
            player1_data = {
                'name': match.player1.name if match.player1 else '',
                'country': match.player1.country if match.player1 else '',
                'ranking': match.player1.ranking if match.player1 else None,
                'age': match.player1.age if match.player1 else None
            }
            
            player2_data = {
                'name': match.player2.name if match.player2 else '',
                'country': match.player2.country if match.player2 else '',
                'ranking': match.player2.ranking if match.player2 else None,
                'age': match.player2.age if match.player2 else None
            }
            
            # Convert to ML format
            ml_data = {
                'match_id': match.id,
                'tournament': match.tournament_name,
                'surface': match.surface,
                'round': match.round,
                'player_1': player1_data['name'],
                'player_2': player2_data['name'],
                'player_1_country': player1_data['country'],
                'player_2_country': player2_data['country'],
                'player_1_ranking': player1_data['ranking'],
                'player_2_ranking': player2_data['ranking'],
                'player_1_age': player1_data['age'],
                'player_2_age': player2_data['age'],
                'odds_player_1': match.odds_player1,
                'odds_player_2': match.odds_player2,
                'start_time': match.start_time.isoformat() if match.start_time else None,
                'status': match.status,
                'location': match.location,
                'level': match.level
            }
            
            return ml_data
            
        except Exception as e:
            logger.error(f"Error converting match to ML format: {e}")
            return {}
    
    def generate_predictions_for_matches(self, matches: List[TennisMatch]) -> List[Dict[str, Any]]:
        """Generate ML predictions for a list of matches"""
        predictions = []
        
        for match in matches:
            try:
                # Convert to ML format
                ml_data = self.convert_api_match_to_ml_format(match)
                
                if not ml_data:
                    continue
                
                # Generate prediction using available ML components
                prediction_result = self._generate_single_prediction(ml_data)
                
                if prediction_result:
                    prediction_result['match_data'] = ml_data
                    predictions.append(prediction_result)
                    
            except Exception as e:
                logger.error(f"Error generating prediction for match {match.id}: {e}")
        
        return predictions
    
    def _generate_single_prediction(self, match_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate prediction for a single match using available ML models"""
        try:
            # Try different prediction methods based on available components
            
            # Method 1: Use SecondSetPredictionService if available
            if self.prediction_service:
                try:
                    # Convert data to the format expected by prediction service
                    prediction = self.prediction_service.predict_match_outcome(match_data)
                    if prediction:
                        return {
                            'method': 'second_set_prediction_service',
                            'prediction': prediction,
                            'confidence': prediction.get('confidence', 0.5),
                            'timestamp': datetime.now().isoformat()
                        }
                except Exception as e:
                    logger.debug(f"SecondSetPredictionService failed: {e}")
            
            # Method 2: Use TennisPredictionModule if available
            if self.tennis_predictor:
                try:
                    prediction = self.tennis_predictor.predict_match(match_data)
                    if prediction:
                        return {
                            'method': 'tennis_prediction_module',
                            'prediction': prediction,
                            'confidence': prediction.get('confidence', 0.5),
                            'timestamp': datetime.now().isoformat()
                        }
                except Exception as e:
                    logger.debug(f"TennisPredictionModule failed: {e}")
            
            # Method 3: Basic prediction based on rankings (fallback)
            return self._generate_basic_prediction(match_data)
            
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            return None
    
    def _generate_basic_prediction(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic prediction based on rankings and odds"""
        try:
            ranking_1 = match_data.get('player_1_ranking', 1000)
            ranking_2 = match_data.get('player_2_ranking', 1000)
            odds_1 = match_data.get('odds_player_1', 2.0)
            odds_2 = match_data.get('odds_player_2', 2.0)
            
            # Simple ranking-based prediction
            if ranking_1 and ranking_2:
                if ranking_1 < ranking_2:  # Lower ranking number = better rank
                    winner_prob = 0.6 + (ranking_2 - ranking_1) / 1000 * 0.3
                else:
                    winner_prob = 0.4 - (ranking_1 - ranking_2) / 1000 * 0.3
            else:
                # Use odds if rankings not available
                if odds_1 and odds_2:
                    implied_prob_1 = 1 / odds_1
                    implied_prob_2 = 1 / odds_2
                    total_prob = implied_prob_1 + implied_prob_2
                    winner_prob = implied_prob_1 / total_prob if total_prob > 0 else 0.5
                else:
                    winner_prob = 0.5
            
            # Clamp between 0.1 and 0.9
            winner_prob = max(0.1, min(0.9, winner_prob))
            
            return {
                'method': 'basic_ranking_odds',
                'prediction': {
                    'winner': match_data.get('player_1', 'Player 1'),
                    'player_1_win_probability': winner_prob,
                    'player_2_win_probability': 1 - winner_prob,
                    'confidence': abs(winner_prob - 0.5) * 2  # Higher confidence for more extreme probabilities
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating basic prediction: {e}")
            return {
                'method': 'basic_fallback',
                'prediction': {
                    'winner': match_data.get('player_1', 'Player 1'),
                    'player_1_win_probability': 0.5,
                    'player_2_win_probability': 0.5,
                    'confidence': 0.0
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def run_live_prediction_pipeline(self) -> Dict[str, Any]:
        """Run the complete live prediction pipeline"""
        try:
            logger.info("Starting live prediction pipeline...")
            
            # Step 1: Fetch live data
            live_data = self.fetch_live_data()
            if 'error' in live_data:
                return {'error': f"Failed to fetch live data: {live_data['error']}"}
            
            # Step 2: Get upcoming matches for prediction
            if not self.api_client:
                return {'error': 'API client not available'}
            
            upcoming_matches = self.api_client.get_upcoming_matches(days_ahead=3)
            
            if not upcoming_matches:
                return {
                    'success': True,
                    'message': 'No upcoming matches found for prediction',
                    'live_data_summary': live_data.get('data', {}),
                    'predictions': []
                }
            
            # Step 3: Generate predictions
            logger.info(f"Generating predictions for {len(upcoming_matches)} matches...")
            predictions = self.generate_predictions_for_matches(upcoming_matches)
            
            return {
                'success': True,
                'live_data_summary': live_data.get('data', {}),
                'predictions_generated': len(predictions),
                'predictions': predictions,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in live prediction pipeline: {e}")
            return {'error': str(e)}


def main():
    """Main function to test the integration"""
    print("=== API-Tennis ML Integration Test ===")
    
    # Initialize integration
    integration = APITennisMLIntegration()
    
    # Check status
    print("\\nComponent Status:")
    status = integration.get_api_status()
    for component, info in status.items():
        print(f"  {component}: {info}")
    
    # Run live prediction pipeline
    print("\\nRunning live prediction pipeline...")
    result = integration.run_live_prediction_pipeline()
    
    if result.get('success'):
        print(f"✅ Pipeline completed successfully!")
        print(f"Generated {result.get('predictions_generated', 0)} predictions")
        
        # Show sample predictions
        for i, prediction in enumerate(result.get('predictions', [])[:3]):
            print(f"\\nPrediction {i+1}:")
            match_data = prediction.get('match_data', {})
            pred_data = prediction.get('prediction', {})
            print(f"  Match: {match_data.get('player_1', 'N/A')} vs {match_data.get('player_2', 'N/A')}")
            print(f"  Tournament: {match_data.get('tournament', 'N/A')}")
            print(f"  Prediction: {pred_data.get('winner', 'N/A')} (Confidence: {pred_data.get('confidence', 0):.2f})")
            print(f"  Method: {prediction.get('method', 'N/A')}")
    else:
        print(f"❌ Pipeline failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()