#!/usr/bin/env python3
"""
Simple API-Tennis ML Integration
Connects API-Tennis.com data with the ML prediction system without complex dependencies
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Import API-Tennis integration
from api_tennis_integration import initialize_api_tennis_client, TennisMatch, TennisPlayer

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleAPITennisMLIntegration:
    """
    Simple integration layer between API-Tennis.com and ML prediction system
    """
    
    def __init__(self):
        """Initialize the integration system"""
        self.api_client = None
        self._initialize_api_client()
    
    def _initialize_api_client(self):
        """Initialize API-Tennis client"""
        try:
            api_key = os.getenv('API_TENNIS_KEY')
            if api_key:
                self.api_client = initialize_api_tennis_client(api_key=api_key)
                logger.info("✅ API-Tennis client initialized successfully")
            else:
                logger.warning("⚠️ API_TENNIS_KEY not found in environment variables")
                
        except Exception as e:
            logger.error(f"❌ Error initializing API client: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of the integration"""
        return {
            'api_tennis_available': self.api_client is not None,
            'api_key_configured': bool(os.getenv('API_TENNIS_KEY')),
            'client_status': self.api_client.get_client_status() if self.api_client else None,
            'timestamp': datetime.now().isoformat()
        }
    
    def fetch_live_matches(self) -> Dict[str, Any]:
        """Fetch live tennis data from API-Tennis"""
        if not self.api_client:
            return {'error': 'API client not initialized'}
        
        try:
            logger.info("🔄 Fetching live tennis data...")
            
            # Get today's matches
            today_matches = self.api_client.get_today_matches()
            logger.info(f"📅 Today's matches: {len(today_matches)}")
            
            # Get upcoming matches
            upcoming_matches = self.api_client.get_upcoming_matches(days_ahead=7)
            logger.info(f"⏳ Upcoming matches (7 days): {len(upcoming_matches)}")
            
            # Get live matches
            live_matches = self.api_client.get_live_matches()
            logger.info(f"🔴 Live matches: {len(live_matches)}")
            
            return {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'data': {
                    'today_matches': len(today_matches),
                    'upcoming_matches': len(upcoming_matches), 
                    'live_matches': len(live_matches),
                    'sample_upcoming': [self._match_to_dict(match) for match in upcoming_matches[:5]],
                    'sample_live': [self._match_to_dict(match) for match in live_matches[:5]]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error fetching live data: {e}")
            return {'error': str(e)}
    
    def _match_to_dict(self, match: TennisMatch) -> Dict[str, Any]:
        """Convert match to dictionary format"""
        return {
            'id': match.id,
            'player1': match.player1.name if match.player1 else 'Unknown',
            'player2': match.player2.name if match.player2 else 'Unknown',
            'tournament': match.tournament_name,
            'surface': match.surface,
            'round': match.round,
            'status': match.status,
            'start_time': match.start_time.isoformat() if match.start_time else None,
            'location': match.location
        }
    
    def generate_simple_predictions(self, matches: List[TennisMatch]) -> List[Dict[str, Any]]:
        """Generate simple predictions based on available data"""
        predictions = []
        
        for match in matches:
            try:
                # Create basic prediction based on player names and available data
                prediction = self._create_basic_prediction(match)
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"❌ Error generating prediction for match {match.id}: {e}")
                continue
        
        return predictions
    
    def _create_basic_prediction(self, match: TennisMatch) -> Dict[str, Any]:
        """Create a basic prediction for a match"""
        # Simple prediction logic based on available information
        player1_name = match.player1.name if match.player1 else "Player 1"
        player2_name = match.player2.name if match.player2 else "Player 2"
        
        # Basic prediction logic (can be enhanced with actual ML models)
        # For now, use random but deterministic prediction based on player names
        import hashlib
        name_hash = hashlib.md5(f"{player1_name}{player2_name}".encode()).hexdigest()
        hash_value = int(name_hash[:8], 16) / (16**8)  # Normalize to 0-1
        
        # Adjust based on tournament level and surface if available
        confidence_modifier = 0.1
        if match.tournament_name:
            if any(word in match.tournament_name.lower() for word in ['grand slam', 'masters', 'atp', 'wta']):
                confidence_modifier += 0.1
        
        player1_prob = 0.3 + hash_value * 0.4  # Between 0.3 and 0.7
        player2_prob = 1.0 - player1_prob
        
        winner = player1_name if player1_prob > player2_prob else player2_name
        confidence = abs(player1_prob - 0.5) * 2 + confidence_modifier
        confidence = min(0.9, confidence)  # Cap at 90%
        
        return {
            'match_id': match.id,
            'tournament': match.tournament_name,
            'surface': match.surface,
            'round': match.round,
            'player1': player1_name,
            'player2': player2_name,
            'prediction': {
                'predicted_winner': winner,
                'player1_win_probability': round(player1_prob, 3),
                'player2_win_probability': round(player2_prob, 3),
                'confidence': round(confidence, 3),
                'method': 'basic_algorithm'
            },
            'timestamp': datetime.now().isoformat(),
            'start_time': match.start_time.isoformat() if match.start_time else None
        }
    
    def run_prediction_pipeline(self) -> Dict[str, Any]:
        """Run the complete prediction pipeline"""
        try:
            logger.info("🚀 Starting prediction pipeline...")
            
            # Step 1: Check API status
            status = self.get_status()
            if not status['api_tennis_available']:
                return {'error': 'API-Tennis client not available'}
            
            # Step 2: Fetch live data
            live_data = self.fetch_live_matches()
            if 'error' in live_data:
                return {'error': f"Failed to fetch live data: {live_data['error']}"}
            
            # Step 3: Get matches for prediction
            if not self.api_client:
                return {'error': 'API client not available'}
            
            upcoming_matches = self.api_client.get_upcoming_matches(days_ahead=3)
            
            if not upcoming_matches:
                logger.info("📭 No upcoming matches found")
                return {
                    'success': True,
                    'message': 'No upcoming matches found for prediction',
                    'live_data': live_data.get('data', {}),
                    'predictions': []
                }
            
            # Step 4: Generate predictions
            logger.info(f"🎯 Generating predictions for {len(upcoming_matches)} matches...")
            predictions = self.generate_simple_predictions(upcoming_matches)
            
            logger.info(f"✅ Generated {len(predictions)} predictions")
            
            return {
                'success': True,
                'pipeline_completed': True,
                'live_data': live_data.get('data', {}),
                'predictions_count': len(predictions),
                'predictions': predictions,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error in prediction pipeline: {e}")
            return {'error': str(e)}
    
    def test_api_connection(self) -> Dict[str, Any]:
        """Test API connection with detailed diagnostics"""
        try:
            if not self.api_client:
                return {'success': False, 'error': 'API client not initialized'}
            
            # Test basic API call
            logger.info("🔍 Testing API connection...")
            event_types = self.api_client.get_event_types()
            
            if isinstance(event_types, dict) and event_types.get('success') == 1:
                logger.info("✅ API connection successful")
                return {
                    'success': True,
                    'api_working': True,
                    'event_types_count': len(event_types.get('result', [])),
                    'sample_events': event_types.get('result', [])[:5],
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'api_working': False,
                    'response': event_types,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ API connection test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


def main():
    """Main function to test the simple integration"""
    print("🎾 === Simple API-Tennis ML Integration Test ===")
    
    # Initialize integration
    integration = SimpleAPITennisMLIntegration()
    
    # Test API connection first
    print("\\n🔍 Testing API connection...")
    api_test = integration.test_api_connection()
    
    if api_test.get('success'):
        print(f"✅ API connection successful!")
        print(f"   Event types found: {api_test.get('event_types_count', 0)}")
    else:
        print(f"❌ API connection failed: {api_test.get('error', 'Unknown error')}")
        return
    
    # Check overall status
    print("\\n📊 System Status:")
    status = integration.get_status()
    for key, value in status.items():
        if key != 'client_status':
            print(f"   {key}: {value}")
    
    # Run prediction pipeline
    print("\\n🚀 Running prediction pipeline...")
    result = integration.run_prediction_pipeline()
    
    if result.get('success'):
        print(f"✅ Pipeline completed successfully!")
        print(f"   Generated predictions: {result.get('predictions_count', 0)}")
        
        # Show live data summary
        live_data = result.get('live_data', {})
        print(f"   Today's matches: {live_data.get('today_matches', 0)}")
        print(f"   Upcoming matches: {live_data.get('upcoming_matches', 0)}")
        print(f"   Live matches: {live_data.get('live_matches', 0)}")
        
        # Show sample predictions
        predictions = result.get('predictions', [])
        for i, pred in enumerate(predictions[:3]):
            print(f"\\n🎯 Prediction {i+1}:")
            print(f"   Match: {pred['player1']} vs {pred['player2']}")
            print(f"   Tournament: {pred['tournament']}")
            prediction_data = pred['prediction']
            print(f"   Winner: {prediction_data['predicted_winner']}")
            print(f"   Confidence: {prediction_data['confidence']:.1%}")
            print(f"   Win probabilities: {prediction_data['player1_win_probability']:.1%} / {prediction_data['player2_win_probability']:.1%}")
            
    else:
        print(f"❌ Pipeline failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()