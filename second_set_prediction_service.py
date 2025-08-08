#!/usr/bin/env python3
"""
🎾 SECOND SET PREDICTION SERVICE
Specialized ML service for predicting underdog second set wins
"""

import pandas as pd
import numpy as np
import joblib
import os
from tensorflow import keras
import warnings
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Import our feature engineering
from second_set_feature_engineering import SecondSetFeatureEngineer

class SecondSetPredictionService:
    """
    Specialized service for predicting second set outcomes
    Target: Probability that underdog wins the second set
    """
    
    def __init__(self, models_dir="tennis_models"):
        self.models_dir = models_dir
        self.models = {}
        self.scaler = None
        self.feature_engineer = SecondSetFeatureEngineer()
        
        # Second set specific model weights (will be trained/optimized for second set data)
        self.second_set_weights = {
            'neural_network': 0.25,      # Better at capturing complex set dynamics
            'xgboost': 0.25,            # Good with feature interactions  
            'random_forest': 0.20,      # Stable baseline performance
            'gradient_boosting': 0.15,   # Reduced weight for second set
            'logistic_regression': 0.15  # Reduced weight for second set
        }
        
        # Expected features for second set prediction (expanded from original)
        self.expected_features = [
            # Original features (subset relevant for second set)
            'player_rank', 'player_age', 'opponent_rank', 'opponent_age',
            'tournament_importance', 'total_pressure', 'surface_advantage',
            
            # NEW: First set context features
            'first_set_games_diff', 'first_set_total_games', 'first_set_close',
            'first_set_dominant', 'player1_won_first_set', 'player2_won_first_set',
            'first_set_duration', 'first_set_long', 'first_set_quick',
            
            # NEW: Break point and serving features  
            'breaks_difference', 'total_breaks', 'break_fest',
            'player1_bp_save_rate', 'player2_bp_save_rate', 'bp_save_difference',
            'player1_serve_percentage', 'player2_serve_percentage', 'serve_percentage_diff',
            'had_tiebreak', 'momentum_with_loser',
            
            # NEW: Momentum and adaptation features
            'player1_second_set_improvement', 'player2_second_set_improvement',
            'player1_comeback_ability', 'player2_comeback_ability',
            'winner_momentum', 'loser_pressure_to_respond', 'active_comeback_scenario',
            'fatigue_factor_player1', 'fatigue_factor_player2', 'mental_fatigue_bonus',
            
            # NEW: Adaptation features
            'player1_adaptation_age_factor', 'player2_adaptation_age_factor',
            'player1_tactical_experience', 'player2_tactical_experience',
            'player1_adaptation_pressure', 'player2_adaptation_pressure',
            'player1_tactical_versatility', 'player2_tactical_versatility',
            
            # NEW: Underdog specific features
            'player1_is_underdog', 'player2_is_underdog', 'ranking_gap',
            'underdog_won_first_set', 'underdog_lost_first_set',
            'underdog_confidence_boost', 'underdog_pressure_as_leader',
            'favorite_desperation_factor', 'underdog_nothing_to_lose',
            'underdog_relaxation_factor', 'favorite_comfort_zone',
            'underdog_competitive_indicator', 'second_set_underdog_value',
            'underdog_mental_toughness',
            
            # NEW: Engineered combinations
            'momentum_times_adaptation', 'pressure_fatigue_interaction',
            'rank_gap_times_first_set_closeness'
        ]
        
        self.is_loaded = False
        
    def load_models(self, retrain_for_second_set=False):
        """
        Load existing models and optionally retrain for second set prediction
        
        Args:
            retrain_for_second_set: If True, will prepare models for second set retraining
        """
        try:
            print("📂 Loading models for second set prediction...")
            
            # Load scaler
            scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print("✅ Scaler loaded")
            else:
                print("⚠️ Creating new scaler for second set features")
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
            
            # Load existing models
            model_files = {
                'neural_network': 'neural_network.h5',
                'xgboost': 'xgboost.pkl', 
                'random_forest': 'random_forest.pkl',
                'gradient_boosting': 'gradient_boosting.pkl',
                'logistic_regression': 'logistic_regression.pkl'
            }
            
            loaded_count = 0
            for model_name, filename in model_files.items():
                filepath = os.path.join(self.models_dir, filename)
                if os.path.exists(filepath):
                    try:
                        if model_name == 'neural_network':
                            self.models[model_name] = keras.models.load_model(filepath)
                        else:
                            self.models[model_name] = joblib.load(filepath)
                        loaded_count += 1
                    except Exception as e:
                        print(f"⚠️ Could not load {model_name}: {e}")
                else:
                    print(f"⚠️ Model file not found: {filepath}")
            
            if loaded_count == 0:
                print("❌ No models loaded - will use simulation mode")
                self.is_loaded = False
                return False
            
            self.is_loaded = True
            print(f"✅ Loaded {loaded_count} models for second set prediction")
            print(f"🎯 Target: Underdog probability to win second set")
            print(f"🔧 Expected features: {len(self.expected_features)}")
            
            if retrain_for_second_set:
                print("🔄 Models loaded but may need retraining for second set target")
                print("💡 Consider generating second set training data")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.is_loaded = False
            return False
    
    def prepare_features_for_prediction(self, features_dict: Dict) -> pd.DataFrame:
        """
        Prepare features for second set prediction
        Handle missing features and ensure correct format
        """
        # Create DataFrame with expected features
        features_df = pd.DataFrame([features_dict])
        
        # Ensure all expected features are present
        for feature in self.expected_features:
            if feature not in features_df.columns:
                # Provide reasonable defaults for missing features
                if 'percentage' in feature or 'rate' in feature:
                    features_df[feature] = 0.5
                elif 'factor' in feature or 'advantage' in feature:
                    features_df[feature] = 0.0
                elif 'pressure' in feature or 'importance' in feature:
                    features_df[feature] = 2.0
                elif 'gap' in feature or 'difference' in feature:
                    features_df[feature] = 0.0
                elif feature.endswith('_is_underdog') or feature.startswith('underdog_'):
                    features_df[feature] = 0.0
                elif 'won_first_set' in feature:
                    features_df[feature] = 0.0
                else:
                    features_df[feature] = 0.0
        
        # Select only expected features in correct order
        features_df = features_df[self.expected_features]
        
        # Fill any remaining NaN values
        features_df = features_df.fillna(features_df.median())
        
        return features_df
    
    def predict_second_set(self, player1_name: str, player2_name: str,
                          player1_data: Dict, player2_data: Dict,
                          match_context: Dict, first_set_data: Dict,
                          return_details: bool = True) -> Dict:
        """
        Predict second set outcome for underdog scenario
        
        Args:
            player1_name: Name of player 1
            player2_name: Name of player 2
            player1_data: Player 1 stats (rank, age, etc.)
            player2_data: Player 2 stats (rank, age, etc.)
            match_context: Tournament, surface, pressure data
            first_set_data: First set outcome and statistics
            return_details: Whether to return detailed predictions
        
        Returns:
            Dict: Prediction results focused on underdog second set probability
        """
        
        if not self.is_loaded:
            if not self.load_models():
                print("⚠️ Models not available, using advanced simulation")
                return self._simulate_second_set_prediction(
                    player1_name, player2_name, player1_data, player2_data,
                    match_context, first_set_data
                )
        
        try:
            # Create complete feature set for second set
            features = self.feature_engineer.create_complete_feature_set(
                player1_name, player2_name, player1_data, player2_data,
                match_context, first_set_data
            )
            
            # Prepare features for ML models
            features_df = self.prepare_features_for_prediction(features)
            
            # Scale features
            if self.scaler:
                X_scaled = self.scaler.transform(features_df)
            else:
                X_scaled = features_df.values
            
            # Get predictions from each model
            individual_predictions = {}
            
            for model_name, model in self.models.items():
                try:
                    if model_name == 'neural_network':
                        # Neural network prediction
                        pred = float(model.predict(X_scaled, verbose=0)[0][0])
                    elif model_name == 'logistic_regression':
                        # Logistic regression probability
                        pred = float(model.predict_proba(X_scaled)[0, 1])
                    else:
                        # Tree-based models probability
                        pred = float(model.predict_proba(features_df)[0, 1])
                    
                    individual_predictions[model_name] = pred
                    
                except Exception as e:
                    print(f"⚠️ Error with {model_name}: {e}")
                    individual_predictions[model_name] = 0.5  # Neutral fallback
            
            # Ensemble prediction with second set specific weights
            ensemble_pred = sum(pred * self.second_set_weights.get(name, 0.2) 
                               for name, pred in individual_predictions.items())
            total_weight = sum(self.second_set_weights.get(name, 0.2) 
                              for name in individual_predictions.keys())
            ensemble_pred /= total_weight if total_weight > 0 else 1
            
            # Post-process for underdog second set context
            underdog_result = self._post_process_for_underdog_second_set(
                ensemble_pred, features, player1_data, player2_data, first_set_data
            )
            
            # Determine confidence
            underdog_prob = underdog_result['underdog_second_set_probability']
            if underdog_prob >= 0.65 or underdog_prob <= 0.25:
                confidence = "High"
            elif underdog_prob >= 0.55 or underdog_prob <= 0.35:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            # Build result
            result = {
                'prediction_type': 'SECOND_SET_ML_UNDERDOG',
                'underdog_second_set_probability': underdog_prob,
                'underdog_player': underdog_result['underdog_player'],
                'favorite_player': underdog_result['favorite_player'],
                'confidence': confidence,
                'key_factors': self._analyze_second_set_factors(features, underdog_prob),
                'first_set_context': {
                    'winner': first_set_data.get('winner', 'unknown'),
                    'score': first_set_data.get('score', 'unknown'),
                    'was_close': features.get('first_set_close', 0) > 0.5
                }
            }
            
            if return_details:
                result.update({
                    'individual_predictions': individual_predictions,
                    'ensemble_weights': self.second_set_weights,
                    'raw_ensemble_prediction': ensemble_pred,
                    'features_used': len(features),
                    'underdog_analysis': underdog_result
                })
            
            return result
            
        except Exception as e:
            print(f"❌ Second set prediction error: {e}")
            return self._simulate_second_set_prediction(
                player1_name, player2_name, player1_data, player2_data,
                match_context, first_set_data
            )
    
    def _post_process_for_underdog_second_set(self, raw_prediction: float, 
                                            features: Dict, player1_data: Dict, 
                                            player2_data: Dict, first_set_data: Dict) -> Dict:
        """
        Post-process ML prediction specifically for underdog second set scenario
        """
        p1_rank = player1_data.get('rank', 100)
        p2_rank = player2_data.get('rank', 100)
        
        # Identify underdog
        if p1_rank > p2_rank:  # Player1 is underdog
            underdog_player = "player1"
            favorite_player = "player2"
            underdog_rank = p1_rank
            favorite_rank = p2_rank
            # raw_prediction is probability player1 wins second set
            underdog_second_set_prob = raw_prediction
        else:  # Player2 is underdog
            underdog_player = "player2"
            favorite_player = "player1"
            underdog_rank = p2_rank
            favorite_rank = p1_rank
            # raw_prediction is probability player1 wins second set
            # Need to invert for player2 (underdog)
            underdog_second_set_prob = 1 - raw_prediction
        
        # Apply second set specific adjustments
        rank_gap = underdog_rank - favorite_rank
        
        # Boost underdog probability in second set scenarios
        if features.get('underdog_lost_first_set', 0) > 0.5:
            # Underdog lost first set - "nothing to lose" boost
            nothing_to_lose_boost = min(0.15, rank_gap / 200)
            underdog_second_set_prob += nothing_to_lose_boost
            
        if features.get('first_set_close', 0) > 0.5:
            # First set was close - underdog showed competitiveness
            competitive_boost = min(0.10, rank_gap / 300)
            underdog_second_set_prob += competitive_boost
        
        if features.get('momentum_with_loser', 0) > 0.1:
            # Momentum indicators favor the loser
            momentum_boost = features.get('momentum_with_loser', 0)
            underdog_second_set_prob += momentum_boost
        
        # Player-specific second set patterns
        underdog_improvement = features.get(f'{underdog_player}_second_set_improvement', 0)
        underdog_second_set_prob += underdog_improvement
        
        # Limit probability range (underdog can win sets but shouldn't be too favored)
        if rank_gap > 200:
            underdog_second_set_prob = max(0.15, min(underdog_second_set_prob, 0.45))
        elif rank_gap > 100:
            underdog_second_set_prob = max(0.20, min(underdog_second_set_prob, 0.55))
        elif rank_gap > 50:
            underdog_second_set_prob = max(0.25, min(underdog_second_set_prob, 0.65))
        else:
            underdog_second_set_prob = max(0.30, min(underdog_second_set_prob, 0.70))
        
        return {
            'underdog_second_set_probability': underdog_second_set_prob,
            'underdog_player': underdog_player,
            'favorite_player': favorite_player,
            'underdog_rank': underdog_rank,
            'favorite_rank': favorite_rank,
            'ranking_gap': rank_gap,
            'raw_ml_prediction': raw_prediction
        }
    
    def _analyze_second_set_factors(self, features: Dict, prediction: float) -> List[str]:
        """Analyze key factors for second set underdog prediction"""
        factors = []
        
        # First set context
        if features.get('underdog_lost_first_set', 0) > 0.5:
            factors.append("🎯 Underdog lost first set - 'nothing to lose' mentality")
            
        if features.get('first_set_close', 0) > 0.5:
            factors.append("⚡ First set was close - underdog showed competitiveness")
            
        if features.get('momentum_with_loser', 0) > 0.1:
            factors.append("📈 Momentum indicators favor the first set loser")
        
        # Break point and serving patterns
        bp_diff = features.get('bp_save_difference', 0)
        if bp_diff < -0.2:  # Underdog struggled with break points
            factors.append("🔥 Break point conversion opportunities in second set")
        
        # Player-specific patterns
        improvement = max(
            features.get('player1_second_set_improvement', 0),
            features.get('player2_second_set_improvement', 0)
        )
        if improvement > 0.1:
            factors.append("📊 Strong historical second set improvement pattern")
        
        # Fatigue and adaptation
        if features.get('first_set_long', 0) > 0.5:
            factors.append("⏱️ Long first set may affect stamina and tactics")
        
        comeback_ability = max(
            features.get('player1_comeback_ability', 0.5),
            features.get('player2_comeback_ability', 0.5)
        )
        if comeback_ability > 0.7:
            factors.append("💪 Strong comeback ability and mental resilience")
        
        # Ranking and expectation factors
        rank_gap = features.get('ranking_gap', 20)
        if rank_gap > 50:
            factors.append(f"🎲 Large ranking gap ({int(rank_gap)}) creates upset potential")
        
        return factors
    
    def _simulate_second_set_prediction(self, player1_name: str, player2_name: str,
                                      player1_data: Dict, player2_data: Dict,
                                      match_context: Dict, first_set_data: Dict) -> Dict:
        """Fallback simulation for second set prediction when models unavailable"""
        
        # Create features using feature engineering
        features = self.feature_engineer.create_complete_feature_set(
            player1_name, player2_name, player1_data, player2_data,
            match_context, first_set_data
        )
        
        # Simple rule-based simulation for second set
        p1_rank = player1_data.get('rank', 100)
        p2_rank = player2_data.get('rank', 100)
        
        # Base probability from rankings
        rank_diff = p2_rank - p1_rank
        base_prob = 0.5 + (rank_diff * 0.002)  # Small ranking effect
        
        # First set context adjustments
        if features.get('underdog_lost_first_set', 0) > 0.5:
            base_prob += 0.12  # Nothing to lose boost
        
        if features.get('first_set_close', 0) > 0.5:
            base_prob += 0.08  # Competitiveness boost
        
        if features.get('momentum_with_loser', 0) > 0:
            base_prob += features['momentum_with_loser']
        
        # Player patterns
        improvement = features.get('player1_second_set_improvement', 0)
        base_prob += improvement
        
        # Limit range
        base_prob = max(0.15, min(base_prob, 0.75))
        
        # Determine underdog
        if p1_rank > p2_rank:
            underdog_prob = base_prob
            underdog_player = "player1"
        else:
            underdog_prob = 1 - base_prob
            underdog_player = "player2"
        
        return {
            'prediction_type': 'SECOND_SET_SIMULATION',
            'underdog_second_set_probability': underdog_prob,
            'underdog_player': underdog_player,
            'confidence': 'Medium',
            'key_factors': self._analyze_second_set_factors(features, underdog_prob),
            'simulation_mode': True
        }
    
    def get_training_data_requirements(self) -> Dict:
        """
        Specify what training data is needed for second set models
        """
        return {
            'target_variable': 'underdog_won_second_set',
            'target_description': 'Binary: 1 if underdog won the second set, 0 if favorite won',
            'required_features': self.expected_features,
            'minimum_samples': 5000,
            'recommended_samples': 15000,
            'data_collection_strategy': [
                '1. Collect historical match data with set-by-set results',
                '2. Identify underdog in each match based on pre-match rankings',
                '3. Extract first set statistics and context',
                '4. Label second set outcome (underdog won/lost)',
                '5. Generate features using SecondSetFeatureEngineer',
                '6. Balance dataset for underdog wins/losses',
                '7. Split by time period to avoid data leakage'
            ],
            'key_considerations': [
                'Focus on matches where underdog lost first set',
                'Include various tournament levels and surfaces',
                'Ensure sufficient examples of different ranking gaps',
                'Consider seasonal and surface-specific patterns'
            ]
        }

# Example usage
if __name__ == "__main__":
    print("🎾 SECOND SET PREDICTION SERVICE TEST")
    print("=" * 60)
    
    service = SecondSetPredictionService()
    
    # Test scenario: Cobolli (underdog) vs Djokovic (favorite) 
    # Cobolli lost first set 4-6, can he win second set?
    player1_data = {"rank": 32, "age": 22}  # Cobolli
    player2_data = {"rank": 5, "age": 37}   # Djokovic
    
    match_context = {
        "tournament_importance": 4,
        "total_pressure": 3.8,
        "player1_surface_advantage": -0.05
    }
    
    first_set_data = {
        "winner": "player2",
        "score": "4-6", 
        "duration_minutes": 48,
        "breaks_won_player1": 0,
        "breaks_won_player2": 1,
        "break_points_saved_player1": 0.4,
        "break_points_saved_player2": 0.8,
        "first_serve_percentage_player1": 0.68,
        "first_serve_percentage_player2": 0.72,
        "had_tiebreak": False
    }
    
    result = service.predict_second_set(
        "flavio cobolli", "novak djokovic",
        player1_data, player2_data, match_context, first_set_data
    )
    
    print(f"\n🎯 SECOND SET PREDICTION RESULT:")
    print(f"Underdog ({result['underdog_player']}): {result['underdog_second_set_probability']:.1%} chance to win set 2")
    print(f"Confidence: {result['confidence']}")
    print(f"Prediction type: {result['prediction_type']}")
    
    print(f"\n🔍 Key factors:")
    for factor in result['key_factors']:
        print(f"  • {factor}")
    
    # Show training requirements
    print(f"\n📚 Training Data Requirements:")
    requirements = service.get_training_data_requirements()
    print(f"Target: {requirements['target_variable']}")
    print(f"Features needed: {len(requirements['required_features'])}")
    print(f"Minimum samples: {requirements['minimum_samples']:,}")
    
    print(f"\n✅ Second set prediction service ready!")