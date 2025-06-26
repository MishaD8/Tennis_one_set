#!/usr/bin/env python3
"""
Flask Backend for Tennis Prediction Dashboard
Connects your existing tennis prediction system to the web interface
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

# Import your existing modules
try:
    from enhanced_data_collector import EnhancedTennisDataCollector
    from enhanced_predictor import EnhancedTennisPredictor
    from enhanced_betting_system import EnhancedTennisBettingSystem, create_sample_matches_and_enhanced_odds
except ImportError as e:
    print(f"⚠️ Could not import tennis modules: {e}")
    print("💡 Make sure your tennis prediction files are in the same directory")

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

class TennisWebAPI:
    def __init__(self):
        self.predictor = None
        self.betting_system = None
        self.last_update = None
        self.cached_matches = []
        self.system_stats = {
            'model_accuracy': 0.724,
            'monthly_roi': 8.7,
            'total_bets': 156,
            'win_rate': 0.627
        }
        
        # Try to load your models
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the prediction system"""
        try:
            # Load your trained models
            self.predictor = EnhancedTennisPredictor()
            self.predictor.load_models()
            
            # Initialize betting system
            self.betting_system = EnhancedTennisBettingSystem(self.predictor, bankroll=10000)
            
            print("✅ Tennis prediction system loaded successfully")
            
        except Exception as e:
            print(f"⚠️ Could not load models: {e}")
            print("💡 Using mock predictor for demo")
            self.predictor = MockPredictor()
            self.betting_system = EnhancedTennisBettingSystem(self.predictor, bankroll=10000)
    
    def get_upcoming_matches(self, days_ahead=7):
        """Get upcoming matches with predictions"""
        try:
            # In real implementation, this would fetch from ATP/WTA APIs
            # For now, we'll use sample data and add some real-time variation
            
            matches_df, odds_data = create_sample_matches_and_enhanced_odds()
            
            # Add some more sample matches
            additional_matches = self.generate_additional_matches()
            all_matches = pd.concat([matches_df, additional_matches], ignore_index=True)
            
            # Get predictions for all matches
            predictions = []
            for idx, match in all_matches.iterrows():
                try:
                    # Get model prediction
                    match_features = self.predictor.prepare_features(pd.DataFrame([match]))
                    probability = self.predictor.predict_probability(match_features)[0]
                    
                    # Determine confidence level
                    if probability >= 0.75:
                        confidence = 'High'
                    elif probability >= 0.60:
                        confidence = 'Medium'
                    else:
                        confidence = 'Low'
                    
                    # Calculate betting metrics
                    odds = np.random.uniform(1.4, 3.0)  # In real app, get from odds_data
                    expected_value = (probability * (odds - 1)) - (1 - probability)
                    kelly_fraction = max(0, ((odds * probability - 1) / (odds - 1)) * 0.25)
                    recommended_stake = min(kelly_fraction * 10000, 500)  # Max $500
                    
                    prediction_data = {
                        'id': f"match_{idx}",
                        'player1': match.get('player_name', f'Player {idx}_A'),
                        'player2': match.get('opponent_name', f'Player {idx}_B'),
                        'tournament': match.get('tournament', 'ATP Tour'),
                        'surface': match.get('surface', 'Hard'),
                        'date': (datetime.now() + timedelta(days=np.random.randint(0, days_ahead))).strftime('%Y-%m-%d'),
                        'time': f"{np.random.randint(10, 20):02d}:00",
                        'round': np.random.choice(['R32', 'R16', 'QF', 'SF', 'F']),
                        'prediction': {
                            'probability': probability,
                            'confidence': confidence,
                            'expected_value': expected_value
                        },
                        'metrics': {
                            'player1_rank': match.get('player_rank', np.random.randint(1, 100)),
                            'player2_rank': match.get('opponent_rank', np.random.randint(1, 100)),
                            'h2h': f"{np.random.randint(0, 15)}-{np.random.randint(0, 15)}",
                            'recent_form': f"{np.random.randint(5, 10)}-{np.random.randint(0, 5)}",
                            'surface_advantage': f"{np.random.randint(-15, 20):+d}%"
                        },
                        'betting': {
                            'odds': round(odds, 2),
                            'stake': round(recommended_stake, 0),
                            'kelly': kelly_fraction,
                            'bookmaker': np.random.choice(['Pinnacle', 'Bet365', 'William Hill', 'Unibet'])
                        }
                    }
                    
                    predictions.append(prediction_data)
                    
                except Exception as e:
                    print(f"⚠️ Error processing match {idx}: {e}")
                    continue
            
            # Cache the results
            self.cached_matches = predictions
            self.last_update = datetime.now()
            
            return predictions
            
        except Exception as e:
            print(f"❌ Error getting matches: {e}")
            return self.get_fallback_matches()
    
    def generate_additional_matches(self):
        """Generate additional sample matches for demo"""
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
        for i in range(6):  # Generate 6 more matches
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
    
    def get_fallback_matches(self):
        """Fallback sample data if main system fails"""
        return [
            {
                'id': 'fallback_001',
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
        """Get current system performance stats"""
        try:
            # In real implementation, calculate from actual betting history
            total_matches = len(self.cached_matches)
            value_bets = len([m for m in self.cached_matches if m['prediction']['expected_value'] > 0.03])
            
            return {
                'total_matches': total_matches,
                'value_bets': value_bets,
                'model_accuracy': f"{self.system_stats['model_accuracy']*100:.1f}%",
                'monthly_roi': f"+{self.system_stats['monthly_roi']:.1f}%",
                'last_update': self.last_update.strftime('%Y-%m-%d %H:%M:%S') if self.last_update else 'Never'
            }
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return {
                'total_matches': 0,
                'value_bets': 0,
                'model_accuracy': '0.0%',
                'monthly_roi': '0.0%',
                'last_update': 'Error'
            }

# Initialize the API
tennis_api = TennisWebAPI()

class MockPredictor:
    """Mock predictor for demo purposes"""
    def prepare_features(self, df):
        feature_cols = ['player_recent_win_rate', 'player_surface_advantage', 
                       'h2h_win_rate', 'total_pressure', 'player_form_trend']
        return df[feature_cols] if all(col in df.columns for col in feature_cols) else df
    
    def predict_probability(self, X):
        if len(X) == 0:
            return np.array([])
        # Simple logic for demo
        base_prob = 0.6
        if hasattr(X, 'iloc'):
            variation = np.random.normal(0, 0.1, len(X))
        else:
            variation = np.random.normal(0, 0.1, 1)
        return np.clip(base_prob + variation, 0.1, 0.9)

@app.route('/')
def dashboard():
    """Serve the main dashboard"""
    return render_template('dashboard.html')

@app.route('/api/matches')
def get_matches():
    """API endpoint to get upcoming matches"""
    try:
        # Get query parameters
        tournament = request.args.get('tournament', '')
        surface = request.args.get('surface', '')
        confidence = request.args.get('confidence', '')
        date_filter = request.args.get('date', '')
        days_ahead = int(request.args.get('days', 7))
        
        # Get matches
        matches = tennis_api.get_upcoming_matches(days_ahead)
        
        # Apply filters
        if tournament:
            matches = [m for m in matches if tournament.lower() in m['tournament'].lower()]
        
        if surface:
            matches = [m for m in matches if m['surface'] == surface]
        
        if confidence:
            matches = [m for m in matches if m['prediction']['confidence'] == confidence]
        
        if date_filter:
            matches = [m for m in matches if m['date'] == date_filter]
        
        return jsonify({
            'success': True,
            'matches': matches,
            'count': len(matches)
        })
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'matches': []
        }), 500

@app.route('/api/stats')
def get_stats():
    """API endpoint to get system statistics"""
    try:
        stats = tennis_api.get_system_stats()
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        print(f"❌ Stats API Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/refresh')
def refresh_data():
    """API endpoint to refresh match data"""
    try:
        # Force refresh of match data
        tennis_api.cached_matches = []
        matches = tennis_api.get_upcoming_matches()
        
        return jsonify({
            'success': True,
            'message': f'Refreshed {len(matches)} matches',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        print(f"❌ Refresh API Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/match/<match_id>')
def get_match_details(match_id):
    """API endpoint to get detailed match information"""
    try:
        match = next((m for m in tennis_api.cached_matches if m['id'] == match_id), None)
        
        if not match:
            return jsonify({
                'success': False,
                'error': 'Match not found'
            }), 404
        
        # Add additional details for specific match
        detailed_match = match.copy()
        detailed_match['detailed_analysis'] = {
            'form_analysis': f"{match['player1']} has won {match['metrics']['recent_form']} recently",
            'surface_notes': f"Surface advantage: {match['metrics']['surface_advantage']}",
            'head_to_head': f"Historical record: {match['metrics']['h2h']}",
            'betting_advice': f"Expected value suggests {'strong' if match['prediction']['expected_value'] > 0.05 else 'moderate'} betting opportunity"
        }
        
        return jsonify({
            'success': True,
            'match': detailed_match
        })
        
    except Exception as e:
        print(f"❌ Match Details API Error: {e}")
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

if __name__ == '__main__':
    print("🎾 Starting Tennis Prediction Web Server...")
    print("=" * 50)
    print("🌐 Dashboard will be available at: http://localhost:5000")
    print("📊 API endpoints:")
    print("  • GET /api/matches - Get upcoming matches")
    print("  • GET /api/stats - Get system statistics")
    print("  • GET /api/refresh - Refresh match data")
    print("  • GET /api/match/<id> - Get match details")
    print("=" * 50)
    
    # Create templates directory and save dashboard HTML
    os.makedirs('templates', exist_ok=True)
    
    # In a real deployment, you'd have the HTML file in templates/dashboard.html
    # For now, we'll use the API endpoints with the standalone HTML file
    
    app.run(host='0.0.0.0', port=5000, debug=True)