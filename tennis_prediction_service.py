#!/usr/bin/env python3
"""
Production Tennis Prediction Service
====================================

This module provides a production-ready service for making real-time tennis
second-set underdog predictions using the trained ensemble models.

Features:
- Real-time inference with confidence scoring
- Betting recommendation system with Kelly Criterion
- Feature preprocessing and validation
- Model performance monitoring
- Risk management and position sizing

Author: Tennis Analytics ML System
Date: 2025-08-15
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class TennisPredictionService:
    """Production service for tennis match predictions."""
    
    def __init__(self, model_path: str = "tennis_models"):
        """Initialize prediction service with trained models."""
        self.model_path = Path(model_path)
        self.models = {}
        self.metadata = {}
        self.feature_names = []
        self.load_models()
    
    def load_models(self):
        """Load all trained models and metadata."""
        print("Loading trained models...")
        
        # Load metadata
        metadata_path = self.model_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                self.feature_names = self.metadata['feature_names']
        
        # Load individual models
        model_files = ['xgboost.pkl', 'lightgbm.pkl', 'random_forest.pkl', 'logistic_regression.pkl']
        
        for model_file in model_files:
            model_path = self.model_path / model_file
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    model_name = model_file.replace('.pkl', '')
                    self.models[model_name] = pickle.load(f)
                    print(f"Loaded {model_name} model")
        
        print(f"Loaded {len(self.models)} models with {len(self.feature_names)} features")
    
    def preprocess_match_data(self, match_data: Dict[str, Any]) -> np.ndarray:
        """Preprocess raw match data into model features."""
        
        # Create feature engineering similar to training
        features = {}
        
        # Core ranking features
        player_rank = match_data.get('player_rank', 100)
        opponent_rank = match_data.get('opponent_rank', 100)
        
        features['player_rank'] = player_rank
        features['opponent_rank'] = opponent_rank
        features['rank_difference'] = player_rank - opponent_rank
        features['rank_ratio'] = player_rank / (opponent_rank + 1)
        features['is_underdog'] = 1 if player_rank > opponent_rank else 0
        features['underdog_magnitude'] = max(0, (player_rank - opponent_rank) / player_rank)
        features['rank_percentile'] = player_rank / 300
        
        # Form features
        features['player_recent_win_rate'] = match_data.get('player_recent_win_rate', 0.5)
        features['player_form_trend'] = match_data.get('player_form_trend', 0.0)
        features['form_momentum'] = features['player_form_trend'] * features['player_recent_win_rate']
        features['form_consistency'] = abs(features['player_form_trend'])
        features['match_frequency'] = 1 / (match_data.get('player_days_since_last_match', 7) + 1)
        
        # Surface features
        features['player_surface_win_rate'] = match_data.get('player_surface_win_rate', 0.5)
        features['player_surface_advantage'] = match_data.get('player_surface_advantage', 0.0)
        features['surface_specialization'] = features['player_surface_win_rate'] - features['player_recent_win_rate']
        features['player_surface_experience'] = match_data.get('player_surface_experience', 1.0)
        
        # H2H features
        features['h2h_win_rate'] = match_data.get('h2h_win_rate', 0.5)
        h2h_matches = match_data.get('h2h_matches', 0)
        features['h2h_dominance'] = (features['h2h_win_rate'] - 0.5) * h2h_matches if h2h_matches > 0 else 0
        features['h2h_momentum'] = match_data.get('h2h_recent_form', 0.0) * h2h_matches
        features['h2h_experience'] = np.log1p(h2h_matches)
        
        # Pressure features
        features['total_pressure'] = match_data.get('total_pressure', 1.0)
        features['momentum_pressure'] = features['total_pressure'] * features['player_form_trend']
        features['pressure_rank_interaction'] = features['total_pressure'] * features['rank_percentile']
        
        # Tournament features
        round_importance = {
            'First Round': 1, 'Second Round': 2, 'Third Round': 3,
            'Fourth Round': 4, 'Quarterfinals': 5, 'Semifinals': 6, 'Final': 7
        }
        features['round_importance_score'] = round_importance.get(match_data.get('round', 'First Round'), 1)
        
        # Age features
        player_age = match_data.get('player_age', 25)
        opponent_age = match_data.get('opponent_age', 25)
        features['player_age'] = player_age
        features['age_difference'] = player_age - opponent_age
        features['age_advantage'] = 1 if features['age_difference'] < -2 else (-1 if features['age_difference'] > 5 else 0)
        
        # Categorical features (encoded)
        surface_mapping = {'Hard': 0, 'Clay': 1, 'Grass': 2}
        features['surface_encoded'] = surface_mapping.get(match_data.get('surface', 'Hard'), 0)
        
        round_mapping = {
            'First Round': 0, 'Second Round': 1, 'Third Round': 2, 'Fourth Round': 3,
            'Quarterfinals': 4, 'Semifinals': 5, 'Final': 6
        }
        features['round_encoded'] = round_mapping.get(match_data.get('round', 'First Round'), 0)
        
        # Create feature vector in correct order
        feature_vector = []
        for feature_name in self.feature_names:
            feature_vector.append(features.get(feature_name, 0.0))
        
        return np.array(feature_vector).reshape(1, -1)
    
    def predict_ensemble(self, X: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Make ensemble prediction with individual model scores."""
        predictions = {}
        
        # Get predictions from each model
        if 'xgboost' in self.models:
            predictions['xgboost'] = self.models['xgboost'].predict_proba(X)[0, 1]
        
        if 'lightgbm' in self.models:
            predictions['lightgbm'] = self.models['lightgbm'].predict_proba(X)[0, 1]
        
        if 'random_forest' in self.models:
            predictions['random_forest'] = self.models['random_forest'].predict_proba(X)[0, 1]
        
        if 'logistic_regression' in self.models:
            X_scaled = self.models['logistic_regression'].scaler.transform(X)
            predictions['logistic_regression'] = self.models['logistic_regression'].predict_proba(X_scaled)[0, 1]
        
        # Ensemble prediction (weighted average)
        weights = {'xgboost': 0.3, 'lightgbm': 0.3, 'random_forest': 0.25, 'logistic_regression': 0.15}
        
        ensemble_score = 0.0
        total_weight = 0.0
        
        for model_name, score in predictions.items():
            if model_name in weights:
                ensemble_score += weights[model_name] * score
                total_weight += weights[model_name]
        
        if total_weight > 0:
            ensemble_score /= total_weight
        
        return ensemble_score, predictions
    
    def calculate_betting_recommendation(self, confidence: float, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate betting recommendation using Kelly Criterion."""
        
        # Default underdog odds estimation
        rank_diff = match_data.get('player_rank', 100) - match_data.get('opponent_rank', 100)
        
        # Estimate odds based on ranking difference
        if rank_diff > 50:
            estimated_odds = 3.5
        elif rank_diff > 20:
            estimated_odds = 2.8
        elif rank_diff > 5:
            estimated_odds = 2.2
        else:
            estimated_odds = 1.8
        
        # Apply provided odds if available
        odds = match_data.get('underdog_odds', estimated_odds)
        
        # Kelly Criterion calculation
        win_probability = confidence
        b = odds - 1  # Net odds
        q = 1 - win_probability
        
        kelly_fraction = (b * win_probability - q) / b
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25% of bankroll
        
        # Betting recommendation
        recommendation = {
            'should_bet': confidence >= 0.6 and kelly_fraction > 0.02,
            'confidence': confidence,
            'kelly_fraction': kelly_fraction,
            'odds': odds,
            'win_probability': win_probability,
            'expected_value': (odds * win_probability) - 1,
            'risk_level': 'Low' if confidence >= 0.8 else ('Medium' if confidence >= 0.65 else 'High')
        }
        
        # Recommended stake as percentage of bankroll
        if recommendation['should_bet']:
            if confidence >= 0.8:
                recommendation['recommended_stake_pct'] = min(kelly_fraction, 0.15)
            elif confidence >= 0.7:
                recommendation['recommended_stake_pct'] = min(kelly_fraction, 0.10)
            else:
                recommendation['recommended_stake_pct'] = min(kelly_fraction, 0.05)
        else:
            recommendation['recommended_stake_pct'] = 0.0
        
        return recommendation
    
    def predict_match(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make complete match prediction with betting analysis."""
        
        # Preprocess data
        X = self.preprocess_match_data(match_data)
        
        # Make prediction
        ensemble_confidence, individual_predictions = self.predict_ensemble(X)
        
        # Calculate betting recommendation
        betting_rec = self.calculate_betting_recommendation(ensemble_confidence, match_data)
        
        # Compile results
        prediction_result = {
            'match_info': {
                'player_rank': match_data.get('player_rank'),
                'opponent_rank': match_data.get('opponent_rank'),
                'surface': match_data.get('surface'),
                'tournament': match_data.get('tournament', 'Unknown'),
                'round': match_data.get('round', 'Unknown')
            },
            'prediction': {
                'ensemble_confidence': ensemble_confidence,
                'prediction': 'Win Second Set' if ensemble_confidence >= 0.5 else 'Lose Second Set',
                'individual_models': individual_predictions
            },
            'betting_analysis': betting_rec,
            'timestamp': datetime.now().isoformat()
        }
        
        return prediction_result
    
    def analyze_historical_performance(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze model performance on historical data."""
        
        results = []
        for match in test_data:
            prediction = self.predict_match(match)
            results.append({
                'confidence': prediction['prediction']['ensemble_confidence'],
                'predicted': 1 if prediction['prediction']['ensemble_confidence'] >= 0.5 else 0,
                'actual': match.get('actual_result', 0),
                'betting_rec': prediction['betting_analysis']['should_bet']
            })
        
        # Calculate metrics
        predictions = [r['predicted'] for r in results]
        actuals = [r['actual'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        accuracy = np.mean([p == a for p, a in zip(predictions, actuals)])
        
        # Betting simulation
        betting_results = [r for r in results if r['betting_rec']]
        if betting_results:
            betting_accuracy = np.mean([r['predicted'] == r['actual'] for r in betting_results])
            num_bets = len(betting_results)
        else:
            betting_accuracy = 0
            num_bets = 0
        
        performance_analysis = {
            'overall_accuracy': accuracy,
            'total_predictions': len(results),
            'betting_accuracy': betting_accuracy,
            'betting_opportunities': num_bets,
            'avg_confidence': np.mean(confidences),
            'high_confidence_accuracy': np.mean([
                r['predicted'] == r['actual'] for r in results if r['confidence'] >= 0.7
            ]) if any(r['confidence'] >= 0.7 for r in results) else 0
        }
        
        return performance_analysis


def create_example_prediction():
    """Demonstrate the prediction service with example data."""
    
    # Initialize service
    service = TennisPredictionService()
    
    # Example match data for an underdog scenario
    example_match = {
        'player_rank': 150,  # Underdog
        'opponent_rank': 25,  # Favorite
        'player_recent_win_rate': 0.65,
        'player_form_trend': 0.1,
        'player_surface_win_rate': 0.70,
        'player_surface_advantage': 0.05,
        'player_surface_experience': 2.5,
        'h2h_win_rate': 0.33,
        'h2h_matches': 3,
        'h2h_recent_form': 0.5,
        'total_pressure': 2.0,
        'player_age': 24,
        'opponent_age': 28,
        'surface': 'Clay',
        'round': 'Second Round',
        'tournament': 'ATP Barcelona',
        'player_days_since_last_match': 5,
        'underdog_odds': 2.8
    }
    
    # Make prediction
    result = service.predict_match(example_match)
    
    return result


if __name__ == "__main__":
    # Demonstrate the service
    print("Tennis Prediction Service Demo")
    print("=" * 50)
    
    try:
        result = create_example_prediction()
        
        print(f"Match: Rank {result['match_info']['player_rank']} vs Rank {result['match_info']['opponent_rank']}")
        print(f"Surface: {result['match_info']['surface']}")
        print(f"Tournament: {result['match_info']['tournament']}")
        print(f"Round: {result['match_info']['round']}")
        print()
        
        print("PREDICTION RESULTS:")
        print(f"Ensemble Confidence: {result['prediction']['ensemble_confidence']:.3f}")
        print(f"Prediction: {result['prediction']['prediction']}")
        print()
        
        print("INDIVIDUAL MODEL SCORES:")
        for model, score in result['prediction']['individual_models'].items():
            print(f"  {model}: {score:.3f}")
        print()
        
        print("BETTING ANALYSIS:")
        betting = result['betting_analysis']
        print(f"Should Bet: {betting['should_bet']}")
        print(f"Confidence: {betting['confidence']:.3f}")
        print(f"Kelly Fraction: {betting['kelly_fraction']:.3f}")
        print(f"Recommended Stake: {betting['recommended_stake_pct']:.1%}")
        print(f"Expected Value: {betting['expected_value']:.3f}")
        print(f"Risk Level: {betting['risk_level']}")
        
    except Exception as e:
        print(f"Error running prediction service: {e}")
        print("Please ensure models are trained and saved in the tennis_models directory.")