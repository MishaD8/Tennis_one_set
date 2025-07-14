#!/usr/bin/env python3
"""
🧠 Adaptive Ensemble Weight Optimizer
Dynamically adjusts ensemble weights based on recent model performance
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import logging
from dataclasses import dataclass, asdict
from enhanced_cache_manager import cache_get, cache_set, get_cache_manager

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformance:
    """Track individual model performance metrics"""
    model_name: str
    predictions: deque  # Recent predictions with actual results
    accuracy_window: int = 50  # Rolling window size
    
    def __post_init__(self):
        if not isinstance(self.predictions, deque):
            self.predictions = deque(maxlen=self.accuracy_window)
    
    def add_prediction(self, prediction: float, actual: float, match_info: Dict = None):
        """Add a prediction with actual result"""
        self.predictions.append({
            'prediction': prediction,
            'actual': actual,
            'correct': abs(prediction - actual) < 0.5,  # Binary classification threshold
            'confidence': abs(prediction - 0.5),  # Distance from 0.5 indicates confidence
            'timestamp': datetime.now().isoformat(),
            'match_info': match_info or {}
        })
    
    def get_recent_accuracy(self, days: int = 7) -> float:
        """Get accuracy for recent period"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_predictions = [
            p for p in self.predictions 
            if datetime.fromisoformat(p['timestamp']) > cutoff_date
        ]
        
        if not recent_predictions:
            return 0.5  # Default accuracy
        
        correct_count = sum(1 for p in recent_predictions if p['correct'])
        return correct_count / len(recent_predictions)
    
    def get_confidence_weighted_accuracy(self, days: int = 7) -> float:
        """Get accuracy weighted by prediction confidence"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_predictions = [
            p for p in self.predictions 
            if datetime.fromisoformat(p['timestamp']) > cutoff_date
        ]
        
        if not recent_predictions:
            return 0.5
        
        total_weight = 0
        weighted_correct = 0
        
        for p in recent_predictions:
            weight = p['confidence']  # Higher confidence gets more weight
            total_weight += weight
            if p['correct']:
                weighted_correct += weight
        
        return weighted_correct / total_weight if total_weight > 0 else 0.5
    
    def get_trend(self, days: int = 14) -> float:
        """Get performance trend (positive = improving, negative = declining)"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_predictions = [
            p for p in self.predictions 
            if datetime.fromisoformat(p['timestamp']) > cutoff_date
        ]
        
        if len(recent_predictions) < 10:
            return 0.0  # Not enough data
        
        # Split into two halves and compare accuracy
        mid_point = len(recent_predictions) // 2
        first_half = recent_predictions[:mid_point]
        second_half = recent_predictions[mid_point:]
        
        first_accuracy = sum(1 for p in first_half if p['correct']) / len(first_half)
        second_accuracy = sum(1 for p in second_half if p['correct']) / len(second_half)
        
        return second_accuracy - first_accuracy

class AdaptiveEnsembleOptimizer:
    """
    Adaptive ensemble weight optimizer
    Tracks model performance and adjusts weights dynamically
    """
    
    def __init__(self, base_weights: Dict[str, float] = None, performance_file: str = "model_performance.json"):
        self.base_weights = base_weights or {
            'neural_network': 0.2054,
            'xgboost': 0.2027,
            'random_forest': 0.1937,
            'gradient_boosting': 0.1916,
            'logistic_regression': 0.2065
        }
        
        self.performance_file = performance_file
        self.model_performances = {}
        
        # Optimization parameters
        self.adaptation_rate = 0.1  # How quickly to adapt weights
        self.min_weight = 0.05  # Minimum weight for any model
        self.max_weight = 0.5   # Maximum weight for any model
        self.performance_window = 50  # Number of recent predictions to track
        
        # Initialize model performance trackers
        for model_name in self.base_weights:
            self.model_performances[model_name] = ModelPerformance(
                model_name=model_name,
                predictions=deque(maxlen=self.performance_window)
            )
        
        self._load_performance_history()
    
    def _load_performance_history(self):
        """Load historical performance data"""
        try:
            if os.path.exists(self.performance_file):
                with open(self.performance_file, 'r') as f:
                    data = json.load(f)
                
                for model_name, perf_data in data.get('model_performances', {}).items():
                    if model_name in self.model_performances:
                        # Convert list back to deque
                        predictions = deque(
                            perf_data.get('predictions', []),
                            maxlen=self.performance_window
                        )
                        self.model_performances[model_name].predictions = predictions
                
                logger.info(f"📊 Loaded performance history for {len(self.model_performances)} models")
                
        except Exception as e:
            logger.warning(f"Could not load performance history: {e}")
    
    def _save_performance_history(self):
        """Save performance data to disk and cache"""
        try:
            data = {
                'model_performances': {},
                'last_updated': datetime.now().isoformat(),
                'base_weights': self.base_weights
            }
            
            for model_name, perf in self.model_performances.items():
                data['model_performances'][model_name] = {
                    'model_name': model_name,
                    'predictions': list(perf.predictions),
                    'accuracy_window': perf.accuracy_window
                }
            
            # Save to file
            with open(self.performance_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Cache the data
            cache_set('ensemble', 'performance_history', data, 'api_response')
            
        except Exception as e:
            logger.warning(f"Could not save performance history: {e}")
    
    def record_predictions(self, predictions: Dict[str, float], actual_result: float, 
                          match_info: Dict = None):
        """
        Record model predictions and actual result
        
        Args:
            predictions: Dict of model_name -> prediction (0-1 probability)
            actual_result: Actual match result (0 or 1)
            match_info: Additional match information
        """
        for model_name, prediction in predictions.items():
            if model_name in self.model_performances:
                self.model_performances[model_name].add_prediction(
                    prediction, actual_result, match_info
                )
        
        self._save_performance_history()
        logger.info(f"📊 Recorded predictions for {len(predictions)} models")
    
    def get_adaptive_weights(self, context: Dict = None) -> Dict[str, float]:
        """
        Calculate adaptive weights based on recent performance
        
        Args:
            context: Match context for contextual weight adjustment
        """
        context = context or {}
        
        # Get recent performance metrics
        performance_metrics = {}
        for model_name, perf in self.model_performances.items():
            metrics = {
                'recent_accuracy': perf.get_recent_accuracy(days=7),
                'confidence_weighted_accuracy': perf.get_confidence_weighted_accuracy(days=7),
                'trend': perf.get_trend(days=14),
                'prediction_count': len(perf.predictions)
            }
            performance_metrics[model_name] = metrics
        
        # Calculate adaptive weights
        adaptive_weights = {}
        total_performance_score = 0
        
        for model_name, base_weight in self.base_weights.items():
            metrics = performance_metrics.get(model_name, {})
            
            # Calculate performance score
            accuracy = metrics.get('recent_accuracy', 0.5)
            confidence_accuracy = metrics.get('confidence_weighted_accuracy', 0.5)
            trend = metrics.get('trend', 0.0)
            prediction_count = metrics.get('prediction_count', 0)
            
            # Weight by confidence-weighted accuracy and trend
            performance_score = (
                confidence_accuracy * 0.7 +  # Main factor: weighted accuracy
                (0.5 + trend) * 0.2 +         # Trend bonus/penalty
                min(prediction_count / 20, 1.0) * 0.1  # Data availability factor
            )
            
            # Apply contextual adjustments
            performance_score = self._apply_contextual_adjustments(
                model_name, performance_score, context
            )
            
            adaptive_weights[model_name] = performance_score
            total_performance_score += performance_score
        
        # Normalize weights
        if total_performance_score > 0:
            for model_name in adaptive_weights:
                adaptive_weights[model_name] /= total_performance_score
        else:
            # Fallback to base weights
            adaptive_weights = self.base_weights.copy()
        
        # Apply constraints and blend with base weights
        final_weights = self._blend_with_base_weights(adaptive_weights)
        final_weights = self._apply_weight_constraints(final_weights)
        
        logger.info(f"🧠 Calculated adaptive weights: {final_weights}")
        return final_weights
    
    def _apply_contextual_adjustments(self, model_name: str, performance_score: float, 
                                    context: Dict) -> float:
        """Apply context-specific adjustments to performance score"""
        adjusted_score = performance_score
        
        # Surface-specific adjustments (if we have surface data)
        surface = context.get('surface', '').lower()
        if surface in ['clay', 'hard', 'grass']:
            # Example: Neural networks might perform better on clay
            surface_bonuses = {
                'neural_network': {'clay': 0.05, 'hard': 0.02, 'grass': 0.0},
                'xgboost': {'clay': 0.02, 'hard': 0.05, 'grass': 0.03},
                'random_forest': {'clay': 0.01, 'hard': 0.02, 'grass': 0.04}
            }
            
            bonus = surface_bonuses.get(model_name, {}).get(surface, 0.0)
            adjusted_score += bonus
        
        # Tournament importance adjustments
        tournament_importance = context.get('tournament_importance', 0)
        if tournament_importance > 0.8:  # High importance tournament
            # Boost models that perform better under pressure
            pressure_bonuses = {
                'neural_network': 0.03,
                'logistic_regression': 0.02,
                'xgboost': 0.01
            }
            adjusted_score += pressure_bonuses.get(model_name, 0.0)
        
        # Player ranking context
        rank_diff = abs(context.get('rank_difference', 0))
        if rank_diff > 50:  # Large ranking difference
            # Boost models that handle upsets well
            upset_bonuses = {
                'random_forest': 0.02,
                'gradient_boosting': 0.015
            }
            adjusted_score += upset_bonuses.get(model_name, 0.0)
        
        return adjusted_score
    
    def _blend_with_base_weights(self, adaptive_weights: Dict[str, float]) -> Dict[str, float]:
        """Blend adaptive weights with base weights to prevent extreme changes"""
        blended_weights = {}
        
        for model_name, base_weight in self.base_weights.items():
            adaptive_weight = adaptive_weights.get(model_name, base_weight)
            
            # Blend: 70% adaptive, 30% base (conservative approach)
            blended_weight = (
                adaptive_weight * (1 - self.adaptation_rate) +
                base_weight * self.adaptation_rate
            )
            
            blended_weights[model_name] = blended_weight
        
        return blended_weights
    
    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply min/max constraints and ensure weights sum to 1.0"""
        constrained_weights = {}
        
        # Apply min/max constraints
        for model_name, weight in weights.items():
            constrained_weights[model_name] = max(
                self.min_weight, 
                min(self.max_weight, weight)
            )
        
        # Normalize to sum to 1.0
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            for model_name in constrained_weights:
                constrained_weights[model_name] /= total_weight
        
        return constrained_weights
    
    def get_performance_report(self) -> Dict:
        """Generate detailed performance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_performances': {},
            'current_weights': self.get_adaptive_weights(),
            'base_weights': self.base_weights,
            'summary': {
                'total_predictions': 0,
                'best_performing_model': None,
                'worst_performing_model': None,
                'average_accuracy': 0.0
            }
        }
        
        accuracies = {}
        total_predictions = 0
        
        for model_name, perf in self.model_performances.items():
            accuracy_7d = perf.get_recent_accuracy(days=7)
            accuracy_30d = perf.get_recent_accuracy(days=30)
            weighted_accuracy = perf.get_confidence_weighted_accuracy(days=7)
            trend = perf.get_trend(days=14)
            prediction_count = len(perf.predictions)
            
            report['model_performances'][model_name] = {
                'accuracy_7d': round(accuracy_7d, 4),
                'accuracy_30d': round(accuracy_30d, 4),
                'confidence_weighted_accuracy': round(weighted_accuracy, 4),
                'trend': round(trend, 4),
                'prediction_count': prediction_count,
                'current_weight': round(report['current_weights'][model_name], 4),
                'base_weight': round(self.base_weights[model_name], 4)
            }
            
            accuracies[model_name] = accuracy_7d
            total_predictions += prediction_count
        
        # Summary statistics
        if accuracies:
            report['summary']['best_performing_model'] = max(accuracies, key=accuracies.get)
            report['summary']['worst_performing_model'] = min(accuracies, key=accuracies.get)
            report['summary']['average_accuracy'] = round(
                sum(accuracies.values()) / len(accuracies), 4
            )
        
        report['summary']['total_predictions'] = total_predictions
        
        return report
    
    def update_base_weights(self, new_base_weights: Dict[str, float]):
        """Update base weights (e.g., after model retraining)"""
        self.base_weights = new_base_weights.copy()
        logger.info(f"🔄 Updated base weights: {self.base_weights}")
        self._save_performance_history()
    
    def reset_performance_history(self):
        """Reset all performance history"""
        for perf in self.model_performances.values():
            perf.predictions.clear()
        
        self._save_performance_history()
        logger.info("🧹 Reset all performance history")

# Global optimizer instance
_ensemble_optimizer = None

def init_ensemble_optimizer(base_weights: Dict[str, float] = None) -> AdaptiveEnsembleOptimizer:
    """Initialize global ensemble optimizer"""
    global _ensemble_optimizer
    _ensemble_optimizer = AdaptiveEnsembleOptimizer(base_weights)
    logger.info("🧠 Adaptive ensemble optimizer initialized")
    return _ensemble_optimizer

def get_ensemble_optimizer() -> Optional[AdaptiveEnsembleOptimizer]:
    """Get global ensemble optimizer instance"""
    return _ensemble_optimizer

# Convenience functions
def get_optimized_weights(context: Dict = None) -> Dict[str, float]:
    """Get optimized weights using global optimizer"""
    if _ensemble_optimizer:
        return _ensemble_optimizer.get_adaptive_weights(context)
    return {}

def record_model_predictions(predictions: Dict[str, float], actual_result: float, 
                           match_info: Dict = None):
    """Record predictions using global optimizer"""
    if _ensemble_optimizer:
        _ensemble_optimizer.record_predictions(predictions, actual_result, match_info)

def get_ensemble_performance_report() -> Dict:
    """Get performance report using global optimizer"""
    if _ensemble_optimizer:
        return _ensemble_optimizer.get_performance_report()
    return {'error': 'Ensemble optimizer not initialized'}

if __name__ == "__main__":
    # Test the adaptive ensemble optimizer
    print("🧠 Testing Adaptive Ensemble Optimizer")
    print("=" * 50)
    
    # Load base weights from metadata
    try:
        with open('tennis_models/metadata.json', 'r') as f:
            metadata = json.load(f)
            base_weights = metadata['ensemble_weights']
    except Exception:
        base_weights = {
            'neural_network': 0.2054,
            'xgboost': 0.2027,
            'random_forest': 0.1937,
            'gradient_boosting': 0.1916,
            'logistic_regression': 0.2065
        }
    
    # Initialize optimizer
    optimizer = init_ensemble_optimizer(base_weights)
    
    # Simulate some predictions
    print("1. Simulating model predictions...")
    
    # Simulate 20 matches with varying performance
    for i in range(20):
        # Generate mock predictions (neural network performing slightly better)
        mock_predictions = {
            'neural_network': 0.7 + np.random.normal(0, 0.1),
            'xgboost': 0.65 + np.random.normal(0, 0.1),
            'random_forest': 0.6 + np.random.normal(0, 0.15),
            'gradient_boosting': 0.6 + np.random.normal(0, 0.12),
            'logistic_regression': 0.68 + np.random.normal(0, 0.1)
        }
        
        # Clip to [0, 1] range
        for model in mock_predictions:
            mock_predictions[model] = max(0, min(1, mock_predictions[model]))
        
        # Generate actual result (slightly favor neural network predictions)
        actual_result = 1 if np.random.random() < mock_predictions['neural_network'] else 0
        
        # Record predictions
        record_model_predictions(mock_predictions, actual_result, {'match_id': f'test_{i}'})
    
    print("2. Getting optimized weights...")
    optimized_weights = get_optimized_weights()
    
    print("3. Base vs Optimized weights:")
    for model_name in base_weights:
        base_w = base_weights[model_name]
        opt_w = optimized_weights.get(model_name, base_w)
        change = ((opt_w - base_w) / base_w) * 100
        print(f"   {model_name:20} {base_w:.4f} → {opt_w:.4f} ({change:+.1f}%)")
    
    # Get performance report
    report = get_ensemble_performance_report()
    print(f"\n4. Performance Summary:")
    print(f"   Best model: {report['summary']['best_performing_model']}")
    print(f"   Average accuracy: {report['summary']['average_accuracy']:.1%}")
    print(f"   Total predictions: {report['summary']['total_predictions']}")
    
    print("\n✅ Adaptive ensemble optimizer test completed!")