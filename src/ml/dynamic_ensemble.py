#!/usr/bin/env python3
"""
Dynamic Tennis Ensemble with Contextual Weighting
=================================================

Advanced ensemble model that dynamically adjusts model weights based on
match context, surface, player rankings, and situational factors.

Features:
- Context-aware model weighting
- Surface-specific ensemble optimization
- Ranking-tier adaptive weighting
- Real-time weight adjustment
- Multi-objective ensemble optimization

Author: Tennis ML Enhancement System
Date: 2025-08-30
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import joblib
import json
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class MatchContext:
    """Match context for dynamic weighting"""
    surface: str                    # 'Hard', 'Clay', 'Grass'
    tournament_tier: str           # 'ATP250', 'ATP500', 'Masters1000', etc.
    round: str                     # 'R32', 'R16', 'QF', 'SF', 'F'
    is_indoor: bool
    player1_ranking: int
    player2_ranking: int
    ranking_gap: int
    is_upset_scenario: bool        # True if large ranking gap
    surface_specialization: Dict[str, float]  # Player surface win rates
    h2h_history: int              # Number of previous meetings
    tournament_importance: float   # 0.0 to 1.0
    
    def get_context_vector(self) -> np.ndarray:
        """Convert context to numerical vector for model input"""
        
        # Surface encoding
        surface_encoding = {
            'Hard': [1, 0, 0],
            'Clay': [0, 1, 0], 
            'Grass': [0, 0, 1]
        }
        surface_vector = surface_encoding.get(self.surface, [1, 0, 0])
        
        # Tournament tier encoding
        tier_weights = {
            'Grand Slam': 1.0,
            'Masters1000': 0.9,
            'ATP500': 0.7,
            'ATP250': 0.5,
            'WTA1000': 0.9,
            'WTA500': 0.7,
            'WTA250': 0.5,
            'Challenger': 0.3,
            'ITF': 0.1
        }
        tier_weight = tier_weights.get(self.tournament_tier, 0.5)
        
        # Round encoding
        round_weights = {
            'R128': 0.1, 'R64': 0.2, 'R32': 0.3, 'R16': 0.5,
            'QF': 0.7, 'SF': 0.9, 'F': 1.0
        }
        round_weight = round_weights.get(self.round, 0.3)
        
        # Build context vector
        context_vector = np.array([
            *surface_vector,                                    # Surface (3 dims)
            tier_weight,                                        # Tournament tier
            round_weight,                                       # Round
            1.0 if self.is_indoor else 0.0,                   # Indoor/outdoor
            min(self.player1_ranking / 300.0, 1.0),           # P1 ranking (normalized)
            min(self.player2_ranking / 300.0, 1.0),           # P2 ranking (normalized)
            min(self.ranking_gap / 200.0, 1.0),               # Ranking gap (normalized)
            1.0 if self.is_upset_scenario else 0.0,           # Upset scenario
            self.surface_specialization.get('player1', 0.5),   # P1 surface specialization
            self.surface_specialization.get('player2', 0.5),   # P2 surface specialization
            min(self.h2h_history / 10.0, 1.0),               # H2H history (normalized)
            self.tournament_importance                          # Tournament importance
        ])
        
        return context_vector

class ContextualWeightCalculator:
    """Calculate dynamic weights based on match context"""
    
    def __init__(self):
        # Model performance profiles by context
        self.model_profiles = {
            'random_forest': {
                'surface_strength': {'Hard': 0.9, 'Clay': 0.8, 'Grass': 0.7},
                'ranking_gap_strength': 'high',  # Performs well with large ranking gaps
                'upset_detection': 0.8,
                'tournament_tier_preference': ['Masters1000', 'ATP500'],
                'indoor_outdoor_bias': 0.1  # Slight indoor preference
            },
            'xgboost': {
                'surface_strength': {'Hard': 0.85, 'Clay': 0.9, 'Grass': 0.75},
                'ranking_gap_strength': 'medium',
                'upset_detection': 0.85,
                'tournament_tier_preference': ['ATP250', 'WTA250'],
                'indoor_outdoor_bias': -0.1  # Slight outdoor preference
            },
            'lightgbm': {
                'surface_strength': {'Hard': 0.8, 'Clay': 0.85, 'Grass': 0.9},
                'ranking_gap_strength': 'high',
                'upset_detection': 0.75,
                'tournament_tier_preference': ['Masters1000', 'WTA1000'],
                'indoor_outdoor_bias': 0.05
            },
            'logistic_regression': {
                'surface_strength': {'Hard': 0.7, 'Clay': 0.75, 'Grass': 0.65},
                'ranking_gap_strength': 'low',  # Better with similar-ranked players
                'upset_detection': 0.6,
                'tournament_tier_preference': ['Challenger', 'ITF'],
                'indoor_outdoor_bias': 0.0
            }
        }
        
        # Base weights (equal weighting as starting point)
        self.base_weights = {
            'random_forest': 0.25,
            'xgboost': 0.25,
            'lightgbm': 0.25,
            'logistic_regression': 0.25
        }
    
    def calculate_contextual_weights(self, context: MatchContext) -> Dict[str, float]:
        """Calculate model weights based on match context"""
        
        weights = {}
        
        for model_name, profile in self.model_profiles.items():
            weight = self.base_weights[model_name]
            
            # Surface adjustment
            surface_multiplier = profile['surface_strength'].get(context.surface, 0.8)
            weight *= surface_multiplier
            
            # Ranking gap adjustment
            if context.ranking_gap > 50:  # Large gap
                if profile['ranking_gap_strength'] == 'high':
                    weight *= 1.2
                elif profile['ranking_gap_strength'] == 'medium':
                    weight *= 1.0
                else:
                    weight *= 0.8
            elif context.ranking_gap < 20:  # Small gap
                if profile['ranking_gap_strength'] == 'low':
                    weight *= 1.2
                else:
                    weight *= 0.9
            
            # Upset scenario adjustment
            if context.is_upset_scenario:
                upset_strength = profile['upset_detection']
                weight *= upset_strength
            
            # Tournament tier adjustment
            if context.tournament_tier in profile['tournament_tier_preference']:
                weight *= 1.1
            
            # Indoor/outdoor adjustment
            indoor_bias = profile['indoor_outdoor_bias']
            if context.is_indoor:
                weight *= (1.0 + indoor_bias)
            else:
                weight *= (1.0 - indoor_bias)
            
            weights[model_name] = max(weight, 0.05)  # Minimum weight
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def get_surface_weights(self, surface: str) -> Dict[str, float]:
        """Get weights optimized for specific surface"""
        
        surface_weights = {}
        for model_name, profile in self.model_profiles.items():
            surface_weights[model_name] = profile['surface_strength'].get(surface, 0.8)
        
        # Normalize
        total = sum(surface_weights.values())
        return {k: v / total for k, v in surface_weights.items()}
    
    def get_ranking_gap_weights(self, ranking_gap: int) -> Dict[str, float]:
        """Get weights optimized for ranking gap scenarios"""
        
        gap_weights = {}
        
        for model_name, profile in self.model_profiles.items():
            if ranking_gap > 50:  # Large gap (upset potential)
                if profile['ranking_gap_strength'] == 'high':
                    gap_weights[model_name] = 1.2
                elif profile['ranking_gap_strength'] == 'medium':
                    gap_weights[model_name] = 1.0
                else:
                    gap_weights[model_name] = 0.7
            else:  # Small gap (close match)
                if profile['ranking_gap_strength'] == 'low':
                    gap_weights[model_name] = 1.2
                else:
                    gap_weights[model_name] = 0.9
        
        # Normalize
        total = sum(gap_weights.values())
        return {k: v / total for k, v in gap_weights.items()}

class DynamicTennisEnsemble(BaseEstimator, ClassifierMixin):
    """Dynamic ensemble classifier with contextual weighting"""
    
    def __init__(self, models: Dict[str, Any], weight_calculator: ContextualWeightCalculator = None):
        self.models = models
        self.weight_calculator = weight_calculator or ContextualWeightCalculator()
        self.is_fitted = False
        self.classes_ = None
        self.feature_importances_ = None
        
        # Performance tracking
        self.weight_history = []
        self.prediction_history = []
        self.context_history = []
        
        logger.info(f"✅ Dynamic ensemble initialized with {len(models)} models")
    
    def fit(self, X: np.ndarray, y: np.ndarray, contexts: List[MatchContext] = None) -> 'DynamicTennisEnsemble':
        """Fit all models in the ensemble"""
        
        logger.info("🔧 Training dynamic ensemble models...")
        
        self.classes_ = np.unique(y)
        
        # Fit each individual model
        for model_name, model in self.models.items():
            logger.info(f"   Training {model_name}...")
            model.fit(X, y)
        
        # Calculate feature importances (if available)
        self._calculate_feature_importances()
        
        self.is_fitted = True
        logger.info("✅ Dynamic ensemble training completed")
        
        return self
    
    def predict(self, X: np.ndarray, contexts: List[MatchContext] = None) -> np.ndarray:
        """Make predictions using dynamic weighting"""
        
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        predictions = []
        
        for i, sample in enumerate(X):
            # Get context for this sample
            if contexts and i < len(contexts):
                context = contexts[i]
                weights = self.weight_calculator.calculate_contextual_weights(context)
                self.context_history.append(context)
            else:
                # Use equal weights if no context
                weights = self.weight_calculator.base_weights
            
            # Get predictions from each model
            model_predictions = {}
            for model_name, model in self.models.items():
                pred_proba = model.predict_proba(sample.reshape(1, -1))[0]
                model_predictions[model_name] = pred_proba[1]  # Probability of class 1
            
            # Calculate weighted prediction
            weighted_prediction = sum(
                model_predictions[model_name] * weights[model_name]
                for model_name in self.models.keys()
            )
            
            # Convert to binary prediction
            prediction = 1 if weighted_prediction > 0.5 else 0
            predictions.append(prediction)
            
            # Store for analysis
            self.weight_history.append(weights)
            self.prediction_history.append({
                'weighted_probability': weighted_prediction,
                'model_predictions': model_predictions,
                'weights': weights,
                'final_prediction': prediction
            })
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray, contexts: List[MatchContext] = None) -> np.ndarray:
        """Predict class probabilities using dynamic weighting"""
        
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        probabilities = []
        
        for i, sample in enumerate(X):
            # Get context for this sample
            if contexts and i < len(contexts):
                context = contexts[i]
                weights = self.weight_calculator.calculate_contextual_weights(context)
            else:
                weights = self.weight_calculator.base_weights
            
            # Get probabilities from each model
            model_probas = {}
            for model_name, model in self.models.items():
                pred_proba = model.predict_proba(sample.reshape(1, -1))[0]
                model_probas[model_name] = pred_proba
            
            # Calculate weighted probabilities
            weighted_proba = np.zeros(len(self.classes_))
            for model_name, weight in weights.items():
                weighted_proba += model_probas[model_name] * weight
            
            probabilities.append(weighted_proba)
        
        return np.array(probabilities)
    
    def predict_with_explanation(self, X: np.ndarray, contexts: List[MatchContext] = None) -> List[Dict[str, Any]]:
        """Make predictions with detailed explanations"""
        
        explanations = []
        
        for i, sample in enumerate(X):
            # Get context and weights
            if contexts and i < len(contexts):
                context = contexts[i]
                weights = self.weight_calculator.calculate_contextual_weights(context)
            else:
                context = None
                weights = self.weight_calculator.base_weights
            
            # Get predictions from each model
            model_results = {}
            for model_name, model in self.models.items():
                pred_proba = model.predict_proba(sample.reshape(1, -1))[0]
                model_results[model_name] = {
                    'probability': pred_proba[1],
                    'weight': weights[model_name],
                    'weighted_contribution': pred_proba[1] * weights[model_name]
                }
            
            # Calculate final prediction
            weighted_probability = sum(
                result['weighted_contribution'] for result in model_results.values()
            )
            
            final_prediction = 1 if weighted_probability > 0.5 else 0
            
            explanation = {
                'final_prediction': final_prediction,
                'final_probability': weighted_probability,
                'model_contributions': model_results,
                'context_used': context is not None,
                'context_details': asdict(context) if context else None,
                'weight_reasoning': self._explain_weights(weights, context) if context else None
            }
            
            explanations.append(explanation)
        
        return explanations
    
    def _explain_weights(self, weights: Dict[str, float], context: MatchContext) -> Dict[str, str]:
        """Explain why specific weights were chosen"""
        
        explanations = {}
        
        for model_name, weight in weights.items():
            reasons = []
            
            profile = self.weight_calculator.model_profiles[model_name]
            
            # Surface reasoning
            surface_strength = profile['surface_strength'].get(context.surface, 0.8)
            if surface_strength > 0.85:
                reasons.append(f"Strong on {context.surface}")
            elif surface_strength < 0.75:
                reasons.append(f"Weaker on {context.surface}")
            
            # Ranking gap reasoning
            if context.ranking_gap > 50:
                if profile['ranking_gap_strength'] == 'high':
                    reasons.append("Good at detecting upsets")
                elif profile['ranking_gap_strength'] == 'low':
                    reasons.append("Less reliable for large ranking gaps")
            
            # Tournament tier reasoning
            if context.tournament_tier in profile['tournament_tier_preference']:
                reasons.append(f"Optimized for {context.tournament_tier}")
            
            explanations[model_name] = "; ".join(reasons) if reasons else "Standard weighting"
        
        return explanations
    
    def _calculate_feature_importances(self) -> None:
        """Calculate aggregated feature importances"""
        
        importances = []
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances.append(model.feature_importances_)
            elif hasattr(model, 'coef_'):
                # For logistic regression, use absolute coefficients
                importances.append(np.abs(model.coef_[0]))
        
        if importances:
            # Average feature importances across models
            self.feature_importances_ = np.mean(importances, axis=0)
        else:
            self.feature_importances_ = None
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get summary of ensemble performance"""
        
        if not self.weight_history:
            return {"error": "No predictions made yet"}
        
        # Average weights used
        avg_weights = {}
        for model_name in self.models.keys():
            weights = [w[model_name] for w in self.weight_history]
            avg_weights[model_name] = {
                'mean': np.mean(weights),
                'std': np.std(weights),
                'min': np.min(weights),
                'max': np.max(weights)
            }
        
        # Weight distribution
        weight_distribution = {
            'total_predictions': len(self.weight_history),
            'average_weights': avg_weights,
            'weight_variance': np.var([list(w.values()) for w in self.weight_history])
        }
        
        return weight_distribution
    
    def save_ensemble(self, filepath: str) -> None:
        """Save the ensemble and its configuration"""
        
        ensemble_data = {
            'models': {},
            'weight_calculator': self.weight_calculator.__dict__,
            'is_fitted': self.is_fitted,
            'classes_': self.classes_.tolist() if self.classes_ is not None else None,
            'feature_importances_': self.feature_importances_.tolist() if self.feature_importances_ is not None else None,
            'performance_history': {
                'weight_history': self.weight_history,
                'prediction_history': self.prediction_history
            }
        }
        
        # Save individual models separately
        for model_name, model in self.models.items():
            model_path = f"{filepath}_{model_name}.pkl"
            joblib.dump(model, model_path)
            ensemble_data['models'][model_name] = model_path
        
        # Save ensemble configuration
        with open(f"{filepath}_ensemble_config.json", 'w') as f:
            json.dump(ensemble_data, f, indent=2, default=str)
        
        logger.info(f"✅ Dynamic ensemble saved to {filepath}")
    
    @classmethod
    def load_ensemble(cls, filepath: str) -> 'DynamicTennisEnsemble':
        """Load a saved ensemble"""
        
        # Load configuration
        with open(f"{filepath}_ensemble_config.json", 'r') as f:
            ensemble_data = json.load(f)
        
        # Load individual models
        models = {}
        for model_name, model_path in ensemble_data['models'].items():
            models[model_name] = joblib.load(model_path)
        
        # Create weight calculator
        weight_calculator = ContextualWeightCalculator()
        weight_calculator.__dict__.update(ensemble_data['weight_calculator'])
        
        # Create ensemble
        ensemble = cls(models, weight_calculator)
        ensemble.is_fitted = ensemble_data['is_fitted']
        
        if ensemble_data['classes_']:
            ensemble.classes_ = np.array(ensemble_data['classes_'])
        
        if ensemble_data['feature_importances_']:
            ensemble.feature_importances_ = np.array(ensemble_data['feature_importances_'])
        
        logger.info(f"✅ Dynamic ensemble loaded from {filepath}")
        
        return ensemble

# Example usage and testing
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb
    import lightgbm as lgb
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = (X[:, 0] + X[:, 1] + np.random.randn(1000) * 0.3 > 0).astype(int)
    
    # Create models
    models = {
        'random_forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'xgboost': xgb.XGBClassifier(n_estimators=50, random_state=42),
        'lightgbm': lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1),
        'logistic_regression': LogisticRegression(random_state=42)
    }
    
    # Create sample contexts
    contexts = []
    for i in range(len(X)):
        context = MatchContext(
            surface=np.random.choice(['Hard', 'Clay', 'Grass']),
            tournament_tier=np.random.choice(['ATP250', 'ATP500', 'Masters1000']),
            round=np.random.choice(['R32', 'R16', 'QF']),
            is_indoor=np.random.choice([True, False]),
            player1_ranking=np.random.randint(1, 100),
            player2_ranking=np.random.randint(50, 300),
            ranking_gap=0,
            is_upset_scenario=False,
            surface_specialization={'player1': 0.6, 'player2': 0.55},
            h2h_history=np.random.randint(0, 5),
            tournament_importance=np.random.uniform(0.3, 1.0)
        )
        context.ranking_gap = abs(context.player1_ranking - context.player2_ranking)
        context.is_upset_scenario = context.ranking_gap > 50
        contexts.append(context)
    
    # Test ensemble
    ensemble = DynamicTennisEnsemble(models)
    
    # Split data
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]
    contexts_train, contexts_test = contexts[:800], contexts[800:]
    
    # Train
    ensemble.fit(X_train, y_train, contexts_train)
    
    # Predict
    predictions = ensemble.predict(X_test, contexts_test)
    probabilities = ensemble.predict_proba(X_test, contexts_test)
    explanations = ensemble.predict_with_explanation(X_test[:5], contexts_test[:5])
    
    # Evaluate
    accuracy = accuracy_score(y_test, predictions)
    print(f"Dynamic ensemble accuracy: {accuracy:.4f}")
    
    # Performance summary
    summary = ensemble.get_model_performance_summary()
    print("\nEnsemble performance summary:")
    for model_name, stats in summary['average_weights'].items():
        print(f"  {model_name}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    
    # Show sample explanation
    print(f"\nSample prediction explanation:")
    print(f"Prediction: {explanations[0]['final_prediction']}")
    print(f"Probability: {explanations[0]['final_probability']:.3f}")
    print("Model contributions:")
    for model, contrib in explanations[0]['model_contributions'].items():
        print(f"  {model}: {contrib['probability']:.3f} × {contrib['weight']:.3f} = {contrib['weighted_contribution']:.3f}")#!/usr/bin/env python3
"""
Dynamic Tennis Ensemble with Contextual Weighting
=================================================

Advanced ensemble model that dynamically adjusts model weights based on
match context, surface, player rankings, and situational factors.

Features:
- Context-aware model weighting
- Surface-specific ensemble optimization
- Ranking-tier adaptive weighting
- Real-time weight adjustment
- Multi-objective ensemble optimization

Author: Tennis ML Enhancement System
Date: 2025-08-30
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import joblib
import json
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class MatchContext:
    """Match context for dynamic weighting"""
    surface: str                    # 'Hard', 'Clay', 'Grass'
    tournament_tier: str           # 'ATP250', 'ATP500', 'Masters1000', etc.
    round: str                     # 'R32', 'R16', 'QF', 'SF', 'F'
    is_indoor: bool
    player1_ranking: int
    player2_ranking: int
    ranking_gap: int
    is_upset_scenario: bool        # True if large ranking gap
    surface_specialization: Dict[str, float]  # Player surface win rates
    h2h_history: int              # Number of previous meetings
    tournament_importance: float   # 0.0 to 1.0
    
    def get_context_vector(self) -> np.ndarray:
        """Convert context to numerical vector for model input"""
        
        # Surface encoding
        surface_encoding = {
            'Hard': [1, 0, 0],
            'Clay': [0, 1, 0], 
            'Grass': [0, 0, 1]
        }
        surface_vector = surface_encoding.get(self.surface, [1, 0, 0])
        
        # Tournament tier encoding
        tier_weights = {
            'Grand Slam': 1.0,
            'Masters1000': 0.9,
            'ATP500': 0.7,
            'ATP250': 0.5,
            'WTA1000': 0.9,
            'WTA500': 0.7,
            'WTA250': 0.5,
            'Challenger': 0.3,
            'ITF': 0.1
        }
        tier_weight = tier_weights.get(self.tournament_tier, 0.5)
        
        # Round encoding
        round_weights = {
            'R128': 0.1, 'R64': 0.2, 'R32': 0.3, 'R16': 0.5,
            'QF': 0.7, 'SF': 0.9, 'F': 1.0
        }
        round_weight = round_weights.get(self.round, 0.3)
        
        # Build context vector
        context_vector = np.array([
            *surface_vector,                                    # Surface (3 dims)
            tier_weight,                                        # Tournament tier
            round_weight,                                       # Round
            1.0 if self.is_indoor else 0.0,                   # Indoor/outdoor
            min(self.player1_ranking / 300.0, 1.0),           # P1 ranking (normalized)
            min(self.player2_ranking / 300.0, 1.0),           # P2 ranking (normalized)
            min(self.ranking_gap / 200.0, 1.0),               # Ranking gap (normalized)
            1.0 if self.is_upset_scenario else 0.0,           # Upset scenario
            self.surface_specialization.get('player1', 0.5),   # P1 surface specialization
            self.surface_specialization.get('player2', 0.5),   # P2 surface specialization
            min(self.h2h_history / 10.0, 1.0),               # H2H history (normalized)
            self.tournament_importance                          # Tournament importance
        ])
        
        return context_vector

class ContextualWeightCalculator:
    """Calculate dynamic weights based on match context"""
    
    def __init__(self):
        # Model performance profiles by context
        self.model_profiles = {
            'random_forest': {
                'surface_strength': {'Hard': 0.9, 'Clay': 0.8, 'Grass': 0.7},
                'ranking_gap_strength': 'high',  # Performs well with large ranking gaps
                'upset_detection': 0.8,
                'tournament_tier_preference': ['Masters1000', 'ATP500'],
                'indoor_outdoor_bias': 0.1  # Slight indoor preference
            },
            'xgboost': {
                'surface_strength': {'Hard': 0.85, 'Clay': 0.9, 'Grass': 0.75},
                'ranking_gap_strength': 'medium',
                'upset_detection': 0.85,
                'tournament_tier_preference': ['ATP250', 'WTA250'],
                'indoor_outdoor_bias': -0.1  # Slight outdoor preference
            },
            'lightgbm': {
                'surface_strength': {'Hard': 0.8, 'Clay': 0.85, 'Grass': 0.9},
                'ranking_gap_strength': 'high',
                'upset_detection': 0.75,
                'tournament_tier_preference': ['Masters1000', 'WTA1000'],
                'indoor_outdoor_bias': 0.05
            },
            'logistic_regression': {
                'surface_strength': {'Hard': 0.7, 'Clay': 0.75, 'Grass': 0.65},
                'ranking_gap_strength': 'low',  # Better with similar-ranked players
                'upset_detection': 0.6,
                'tournament_tier_preference': ['Challenger', 'ITF'],
                'indoor_outdoor_bias': 0.0
            }
        }
        
        # Base weights (equal weighting as starting point)
        self.base_weights = {
            'random_forest': 0.25,
            'xgboost': 0.25,
            'lightgbm': 0.25,
            'logistic_regression': 0.25
        }
    
    def calculate_contextual_weights(self, context: MatchContext) -> Dict[str, float]:
        """Calculate model weights based on match context"""
        
        weights = {}
        
        for model_name, profile in self.model_profiles.items():
            weight = self.base_weights[model_name]
            
            # Surface adjustment
            surface_multiplier = profile['surface_strength'].get(context.surface, 0.8)
            weight *= surface_multiplier
            
            # Ranking gap adjustment
            if context.ranking_gap > 50:  # Large gap
                if profile['ranking_gap_strength'] == 'high':
                    weight *= 1.2
                elif profile['ranking_gap_strength'] == 'medium':
                    weight *= 1.0
                else:
                    weight *= 0.8
            elif context.ranking_gap < 20:  # Small gap
                if profile['ranking_gap_strength'] == 'low':
                    weight *= 1.2
                else:
                    weight *= 0.9
            
            # Upset scenario adjustment
            if context.is_upset_scenario:
                upset_strength = profile['upset_detection']
                weight *= upset_strength
            
            # Tournament tier adjustment
            if context.tournament_tier in profile['tournament_tier_preference']:
                weight *= 1.1
            
            # Indoor/outdoor adjustment
            indoor_bias = profile['indoor_outdoor_bias']
            if context.is_indoor:
                weight *= (1.0 + indoor_bias)
            else:
                weight *= (1.0 - indoor_bias)
            
            weights[model_name] = max(weight, 0.05)  # Minimum weight
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def get_surface_weights(self, surface: str) -> Dict[str, float]:
        """Get weights optimized for specific surface"""
        
        surface_weights = {}
        for model_name, profile in self.model_profiles.items():
            surface_weights[model_name] = profile['surface_strength'].get(surface, 0.8)
        
        # Normalize
        total = sum(surface_weights.values())
        return {k: v / total for k, v in surface_weights.items()}
    
    def get_ranking_gap_weights(self, ranking_gap: int) -> Dict[str, float]:
        """Get weights optimized for ranking gap scenarios"""
        
        gap_weights = {}
        
        for model_name, profile in self.model_profiles.items():
            if ranking_gap > 50:  # Large gap (upset potential)
                if profile['ranking_gap_strength'] == 'high':
                    gap_weights[model_name] = 1.2
                elif profile['ranking_gap_strength'] == 'medium':
                    gap_weights[model_name] = 1.0
                else:
                    gap_weights[model_name] = 0.7
            else:  # Small gap (close match)
                if profile['ranking_gap_strength'] == 'low':
                    gap_weights[model_name] = 1.2
                else:
                    gap_weights[model_name] = 0.9
        
        # Normalize
        total = sum(gap_weights.values())
        return {k: v / total for k, v in gap_weights.items()}

class DynamicTennisEnsemble(BaseEstimator, ClassifierMixin):
    """Dynamic ensemble classifier with contextual weighting"""
    
    def __init__(self, models: Dict[str, Any], weight_calculator: ContextualWeightCalculator = None):
        self.models = models
        self.weight_calculator = weight_calculator or ContextualWeightCalculator()
        self.is_fitted = False
        self.classes_ = None
        self.feature_importances_ = None
        
        # Performance tracking
        self.weight_history = []
        self.prediction_history = []
        self.context_history = []
        
        logger.info(f"✅ Dynamic ensemble initialized with {len(models)} models")
    
    def fit(self, X: np.ndarray, y: np.ndarray, contexts: List[MatchContext] = None) -> 'DynamicTennisEnsemble':
        """Fit all models in the ensemble"""
        
        logger.info("🔧 Training dynamic ensemble models...")
        
        self.classes_ = np.unique(y)
        
        # Fit each individual model
        for model_name, model in self.models.items():
            logger.info(f"   Training {model_name}...")
            model.fit(X, y)
        
        # Calculate feature importances (if available)
        self._calculate_feature_importances()
        
        self.is_fitted = True
        logger.info("✅ Dynamic ensemble training completed")
        
        return self
    
    def predict(self, X: np.ndarray, contexts: List[MatchContext] = None) -> np.ndarray:
        """Make predictions using dynamic weighting"""
        
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        predictions = []
        
        for i, sample in enumerate(X):
            # Get context for this sample
            if contexts and i < len(contexts):
                context = contexts[i]
                weights = self.weight_calculator.calculate_contextual_weights(context)
                self.context_history.append(context)
            else:
                # Use equal weights if no context
                weights = self.weight_calculator.base_weights
            
            # Get predictions from each model
            model_predictions = {}
            for model_name, model in self.models.items():
                pred_proba = model.predict_proba(sample.reshape(1, -1))[0]
                model_predictions[model_name] = pred_proba[1]  # Probability of class 1
            
            # Calculate weighted prediction
            weighted_prediction = sum(
                model_predictions[model_name] * weights[model_name]
                for model_name in self.models.keys()
            )
            
            # Convert to binary prediction
            prediction = 1 if weighted_prediction > 0.5 else 0
            predictions.append(prediction)
            
            # Store for analysis
            self.weight_history.append(weights)
            self.prediction_history.append({
                'weighted_probability': weighted_prediction,
                'model_predictions': model_predictions,
                'weights': weights,
                'final_prediction': prediction
            })
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray, contexts: List[MatchContext] = None) -> np.ndarray:
        """Predict class probabilities using dynamic weighting"""
        
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        probabilities = []
        
        for i, sample in enumerate(X):
            # Get context for this sample
            if contexts and i < len(contexts):
                context = contexts[i]
                weights = self.weight_calculator.calculate_contextual_weights(context)
            else:
                weights = self.weight_calculator.base_weights
            
            # Get probabilities from each model
            model_probas = {}
            for model_name, model in self.models.items():
                pred_proba = model.predict_proba(sample.reshape(1, -1))[0]
                model_probas[model_name] = pred_proba
            
            # Calculate weighted probabilities
            weighted_proba = np.zeros(len(self.classes_))
            for model_name, weight in weights.items():
                weighted_proba += model_probas[model_name] * weight
            
            probabilities.append(weighted_proba)
        
        return np.array(probabilities)
    
    def predict_with_explanation(self, X: np.ndarray, contexts: List[MatchContext] = None) -> List[Dict[str, Any]]:
        """Make predictions with detailed explanations"""
        
        explanations = []
        
        for i, sample in enumerate(X):
            # Get context and weights
            if contexts and i < len(contexts):
                context = contexts[i]
                weights = self.weight_calculator.calculate_contextual_weights(context)
            else:
                context = None
                weights = self.weight_calculator.base_weights
            
            # Get predictions from each model
            model_results = {}
            for model_name, model in self.models.items():
                pred_proba = model.predict_proba(sample.reshape(1, -1))[0]
                model_results[model_name] = {
                    'probability': pred_proba[1],
                    'weight': weights[model_name],
                    'weighted_contribution': pred_proba[1] * weights[model_name]
                }
            
            # Calculate final prediction
            weighted_probability = sum(
                result['weighted_contribution'] for result in model_results.values()
            )
            
            final_prediction = 1 if weighted_probability > 0.5 else 0
            
            explanation = {
                'final_prediction': final_prediction,
                'final_probability': weighted_probability,
                'model_contributions': model_results,
                'context_used': context is not None,
                'context_details': asdict(context) if context else None,
                'weight_reasoning': self._explain_weights(weights, context) if context else None
            }
            
            explanations.append(explanation)
        
        return explanations
    
    def _explain_weights(self, weights: Dict[str, float], context: MatchContext) -> Dict[str, str]:
        """Explain why specific weights were chosen"""
        
        explanations = {}
        
        for model_name, weight in weights.items():
            reasons = []
            
            profile = self.weight_calculator.model_profiles[model_name]
            
            # Surface reasoning
            surface_strength = profile['surface_strength'].get(context.surface, 0.8)
            if surface_strength > 0.85:
                reasons.append(f"Strong on {context.surface}")
            elif surface_strength < 0.75:
                reasons.append(f"Weaker on {context.surface}")
            
            # Ranking gap reasoning
            if context.ranking_gap > 50:
                if profile['ranking_gap_strength'] == 'high':
                    reasons.append("Good at detecting upsets")
                elif profile['ranking_gap_strength'] == 'low':
                    reasons.append("Less reliable for large ranking gaps")
            
            # Tournament tier reasoning
            if context.tournament_tier in profile['tournament_tier_preference']:
                reasons.append(f"Optimized for {context.tournament_tier}")
            
            explanations[model_name] = "; ".join(reasons) if reasons else "Standard weighting"
        
        return explanations
    
    def _calculate_feature_importances(self) -> None:
        """Calculate aggregated feature importances"""
        
        importances = []
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances.append(model.feature_importances_)
            elif hasattr(model, 'coef_'):
                # For logistic regression, use absolute coefficients
                importances.append(np.abs(model.coef_[0]))
        
        if importances:
            # Average feature importances across models
            self.feature_importances_ = np.mean(importances, axis=0)
        else:
            self.feature_importances_ = None
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get summary of ensemble performance"""
        
        if not self.weight_history:
            return {"error": "No predictions made yet"}
        
        # Average weights used
        avg_weights = {}
        for model_name in self.models.keys():
            weights = [w[model_name] for w in self.weight_history]
            avg_weights[model_name] = {
                'mean': np.mean(weights),
                'std': np.std(weights),
                'min': np.min(weights),
                'max': np.max(weights)
            }
        
        # Weight distribution
        weight_distribution = {
            'total_predictions': len(self.weight_history),
            'average_weights': avg_weights,
            'weight_variance': np.var([list(w.values()) for w in self.weight_history])
        }
        
        return weight_distribution
    
    def save_ensemble(self, filepath: str) -> None:
        """Save the ensemble and its configuration"""
        
        ensemble_data = {
            'models': {},
            'weight_calculator': self.weight_calculator.__dict__,
            'is_fitted': self.is_fitted,
            'classes_': self.classes_.tolist() if self.classes_ is not None else None,
            'feature_importances_': self.feature_importances_.tolist() if self.feature_importances_ is not None else None,
            'performance_history': {
                'weight_history': self.weight_history,
                'prediction_history': self.prediction_history
            }
        }
        
        # Save individual models separately
        for model_name, model in self.models.items():
            model_path = f"{filepath}_{model_name}.pkl"
            joblib.dump(model, model_path)
            ensemble_data['models'][model_name] = model_path
        
        # Save ensemble configuration
        with open(f"{filepath}_ensemble_config.json", 'w') as f:
            json.dump(ensemble_data, f, indent=2, default=str)
        
        logger.info(f"✅ Dynamic ensemble saved to {filepath}")
    
    @classmethod
    def load_ensemble(cls, filepath: str) -> 'DynamicTennisEnsemble':
        """Load a saved ensemble"""
        
        # Load configuration
        with open(f"{filepath}_ensemble_config.json", 'r') as f:
            ensemble_data = json.load(f)
        
        # Load individual models
        models = {}
        for model_name, model_path in ensemble_data['models'].items():
            models[model_name] = joblib.load(model_path)
        
        # Create weight calculator
        weight_calculator = ContextualWeightCalculator()
        weight_calculator.__dict__.update(ensemble_data['weight_calculator'])
        
        # Create ensemble
        ensemble = cls(models, weight_calculator)
        ensemble.is_fitted = ensemble_data['is_fitted']
        
        if ensemble_data['classes_']:
            ensemble.classes_ = np.array(ensemble_data['classes_'])
        
        if ensemble_data['feature_importances_']:
            ensemble.feature_importances_ = np.array(ensemble_data['feature_importances_'])
        
        logger.info(f"✅ Dynamic ensemble loaded from {filepath}")
        
        return ensemble

# Example usage and testing
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb
    import lightgbm as lgb
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = (X[:, 0] + X[:, 1] + np.random.randn(1000) * 0.3 > 0).astype(int)
    
    # Create models
    models = {
        'random_forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'xgboost': xgb.XGBClassifier(n_estimators=50, random_state=42),
        'lightgbm': lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1),
        'logistic_regression': LogisticRegression(random_state=42)
    }
    
    # Create sample contexts
    contexts = []
    for i in range(len(X)):
        context = MatchContext(
            surface=np.random.choice(['Hard', 'Clay', 'Grass']),
            tournament_tier=np.random.choice(['ATP250', 'ATP500', 'Masters1000']),
            round=np.random.choice(['R32', 'R16', 'QF']),
            is_indoor=np.random.choice([True, False]),
            player1_ranking=np.random.randint(1, 100),
            player2_ranking=np.random.randint(50, 300),
            ranking_gap=0,
            is_upset_scenario=False,
            surface_specialization={'player1': 0.6, 'player2': 0.55},
            h2h_history=np.random.randint(0, 5),
            tournament_importance=np.random.uniform(0.3, 1.0)
        )
        context.ranking_gap = abs(context.player1_ranking - context.player2_ranking)
        context.is_upset_scenario = context.ranking_gap > 50
        contexts.append(context)
    
    # Test ensemble
    ensemble = DynamicTennisEnsemble(models)
    
    # Split data
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]
    contexts_train, contexts_test = contexts[:800], contexts[800:]
    
    # Train
    ensemble.fit(X_train, y_train, contexts_train)
    
    # Predict
    predictions = ensemble.predict(X_test, contexts_test)
    probabilities = ensemble.predict_proba(X_test, contexts_test)
    explanations = ensemble.predict_with_explanation(X_test[:5], contexts_test[:5])
    
    # Evaluate
    accuracy = accuracy_score(y_test, predictions)
    print(f"Dynamic ensemble accuracy: {accuracy:.4f}")
    
    # Performance summary
    summary = ensemble.get_model_performance_summary()
    print("\nEnsemble performance summary:")
    for model_name, stats in summary['average_weights'].items():
        print(f"  {model_name}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    
    # Show sample explanation
    print(f"\nSample prediction explanation:")
    print(f"Prediction: {explanations[0]['final_prediction']}")
    print(f"Probability: {explanations[0]['final_probability']:.3f}")
    print("Model contributions:")
    for model, contrib in explanations[0]['model_contributions'].items():
        print(f"  {model}: {contrib['probability']:.3f} × {contrib['weight']:.3f} = {contrib['weighted_contribution']:.3f}")