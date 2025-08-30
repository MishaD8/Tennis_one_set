#!/usr/bin/env python3
"""
Integration Tests for Enhanced Tennis ML System
==============================================

Tests the integration between all enhanced ML components:
- Enhanced feature engineering
- Bayesian optimization
- Dynamic ensemble
- Real-time data processing

Author: Tennis ML Enhancement System
Date: 2025-08-30
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import tempfile
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.ml.enhanced_feature_engineering import (
    EnhancedTennisFeatureEngineer,
    MatchContext,
    FirstSetStats
)

from src.ml.bayesian_hyperparameter_optimizer import TennisBayesianOptimizer

from src.ml.dynamic_ensemble import (
    DynamicTennisEnsemble,
    ContextualWeightCalculator
)

# Mock models for testing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class TestMLIntegration:
    """Integration tests for enhanced ML system"""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data"""
        np.random.seed(42)
        
        # Create synthetic features
        n_samples = 200
        n_features = 30
        
        X = np.random.randn(n_samples, n_features)
        # Create realistic tennis prediction target (slightly better than random)
        y = ((X[:, 0] + X[:, 1] * 0.5 + X[:, 2] * 0.3 + np.random.randn(n_samples) * 0.8) > 0).astype(int)
        
        return X, y
    
    @pytest.fixture
    def sample_match_contexts(self):
        """Create sample match contexts for dynamic ensemble"""
        contexts = []
        
        for i in range(200):
            context = MatchContext(
                surface=np.random.choice(['Hard', 'Clay', 'Grass']),
                tournament_tier=np.random.choice(['ATP250', 'ATP500', 'Masters1000']),
                round=np.random.choice(['R32', 'R16', 'QF']),
                is_indoor=np.random.choice([True, False]),
                player1_ranking=np.random.randint(10, 100),
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
        
        return contexts
    
    def test_feature_engineering_integration(self):
        """Test that enhanced feature engineering works with ML pipeline"""
        engineer = EnhancedTennisFeatureEngineer()
        
        # Create sample match data
        match_data = {
            'player1': {
                'ranking': 25,
                'age': 26,
                'last_match_date': datetime.now() - timedelta(days=3),
                'matches_last_14_days': 2,
                'hard_win_percentage': 0.65,
                'indoor_win_percentage': 0.70
            },
            'player2': {
                'ranking': 85,
                'age': 24,
                'last_match_date': datetime.now() - timedelta(days=5),
                'matches_last_14_days': 1,
                'hard_win_percentage': 0.55,
                'indoor_win_percentage': 0.50
            },
            'first_set_stats': FirstSetStats(
                winner='player1',
                score='6-4',
                duration_minutes=45,
                total_games=10,
                break_points_player1={'faced': 2, 'saved': 2, 'converted': 1},
                break_points_player2={'faced': 3, 'saved': 1, 'converted': 0},
                service_points_player1={'won': 35, 'total': 50},
                service_points_player2={'won': 25, 'total': 40},
                unforced_errors_player1=8,
                unforced_errors_player2=12,
                winners_player1=15,
                winners_player2=8,
                double_faults_player1=1,
                double_faults_player2=2
            ),
            'match_context': MatchContext(
                tournament_tier='ATP500',
                surface='Hard',
                round='R16',
                is_indoor=True,
                altitude=500.0
            ),
            'h2h_data': {
                'overall': {'player1_wins': 2, 'player2_wins': 1},
                'recent_3': {'player1_wins': 2, 'player2_wins': 1}
            },
            'match_date': datetime.now()
        }
        
        # Generate features
        features = engineer.create_all_enhanced_features(match_data)
        
        # Should create meaningful number of features
        assert len(features) >= 40, f"Expected at least 40 features, got {len(features)}"
        
        # Features should be numerical and valid
        feature_values = list(features.values())
        assert all(isinstance(v, (int, float)) for v in feature_values), "All features should be numerical"
        assert all(not np.isnan(v) for v in feature_values), "No features should be NaN"
        assert all(not np.isinf(v) for v in feature_values), "No features should be infinite"
        
        # Convert to array format for ML models
        feature_array = np.array(list(features.values())).reshape(1, -1)
        assert feature_array.shape[1] == len(features), "Feature array shape should match feature count"
    
    def test_bayesian_optimization_integration(self, sample_training_data):
        """Test Bayesian optimization integration"""
        X, y = sample_training_data
        
        # Test with minimal configuration for speed
        optimizer = TennisBayesianOptimizer(n_calls=5, cv_folds=3)
        
        # Test optimization for one model
        result = optimizer.optimize_model('random_forest', X, y)
        
        # Should return optimization results
        assert 'best_params' in result
        assert 'best_scores' in result or 'optimization_score' in result
        assert 'model_type' in result
        
        # Parameters should be reasonable for Random Forest
        if 'best_params' in result:
            params = result['best_params']
            if 'n_estimators' in params:
                assert 50 <= params['n_estimators'] <= 300
            if 'max_depth' in params:
                assert 3 <= params['max_depth'] <= 20
    
    def test_dynamic_ensemble_integration(self, sample_training_data, sample_match_contexts):
        """Test dynamic ensemble integration"""
        X, y = sample_training_data
        contexts = sample_match_contexts
        
        # Create simple models for testing
        models = {
            'random_forest': RandomForestClassifier(n_estimators=10, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=200)
        }
        
        # Create dynamic ensemble
        ensemble = DynamicTennisEnsemble(models)
        
        # Split data
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]
        contexts_train, contexts_test = contexts[:150], contexts[150:]
        
        # Fit ensemble
        ensemble.fit(X_train, y_train, contexts_train)
        
        # Make predictions
        predictions = ensemble.predict(X_test, contexts_test)
        probabilities = ensemble.predict_proba(X_test, contexts_test)
        explanations = ensemble.predict_with_explanation(X_test[:5], contexts_test[:5])
        
        # Validate predictions
        assert predictions.shape == (len(X_test),), "Predictions shape should match test set"
        assert probabilities.shape == (len(X_test), 2), "Probabilities should be 2D"
        assert len(explanations) == 5, "Should have 5 explanations"
        
        # Check that predictions are binary
        assert all(p in [0, 1] for p in predictions), "Predictions should be binary"
        
        # Check that probabilities sum to 1
        assert all(abs(sum(prob) - 1.0) < 1e-6 for prob in probabilities), "Probabilities should sum to 1"
        
        # Check explanations structure
        for explanation in explanations:
            assert 'final_prediction' in explanation
            assert 'final_probability' in explanation
            assert 'model_contributions' in explanation
            assert 'context_used' in explanation
    
    def test_weight_calculator_contextual_adaptation(self, sample_match_contexts):
        """Test that weight calculator adapts to different contexts"""
        calculator = ContextualWeightCalculator()
        
        # Test different surface contexts
        hard_context = MatchContext(
            surface='Hard', tournament_tier='ATP500', round='QF', is_indoor=True,
            player1_ranking=20, player2_ranking=80, ranking_gap=60, is_upset_scenario=True,
            surface_specialization={'player1': 0.8, 'player2': 0.6}, h2h_history=3, tournament_importance=0.8
        )
        
        clay_context = MatchContext(
            surface='Clay', tournament_tier='ATP500', round='QF', is_indoor=False,
            player1_ranking=20, player2_ranking=80, ranking_gap=60, is_upset_scenario=True,
            surface_specialization={'player1': 0.6, 'player2': 0.8}, h2h_history=3, tournament_importance=0.8
        )
        
        hard_weights = calculator.calculate_contextual_weights(hard_context)
        clay_weights = calculator.calculate_contextual_weights(clay_context)
        
        # Weights should be different for different surfaces
        assert hard_weights != clay_weights, "Weights should adapt to surface"
        
        # All weights should sum to approximately 1
        assert abs(sum(hard_weights.values()) - 1.0) < 1e-6, "Hard court weights should sum to 1"
        assert abs(sum(clay_weights.values()) - 1.0) < 1e-6, "Clay court weights should sum to 1"
        
        # All weights should be positive
        assert all(w > 0 for w in hard_weights.values()), "All hard court weights should be positive"
        assert all(w > 0 for w in clay_weights.values()), "All clay court weights should be positive"
    
    def test_model_persistence_integration(self, sample_training_data, sample_match_contexts):
        """Test that models can be saved and loaded correctly"""
        X, y = sample_training_data
        contexts = sample_match_contexts[:50]  # Use subset for speed
        
        # Create and train ensemble
        models = {
            'random_forest': RandomForestClassifier(n_estimators=10, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=200)
        }
        
        ensemble = DynamicTennisEnsemble(models)
        ensemble.fit(X[:50], y[:50], contexts)
        
        # Make predictions before saving
        original_predictions = ensemble.predict(X[50:60], contexts[:10])
        
        # Save ensemble
        with tempfile.TemporaryDirectory() as tmpdir:
            ensemble_path = os.path.join(tmpdir, 'test_ensemble')
            ensemble.save_ensemble(ensemble_path)
            
            # Load ensemble
            loaded_ensemble = DynamicTennisEnsemble.load_ensemble(ensemble_path)
            
            # Make predictions with loaded ensemble
            loaded_predictions = loaded_ensemble.predict(X[50:60], contexts[:10])
            
            # Predictions should be identical
            np.testing.assert_array_equal(original_predictions, loaded_predictions,
                                        "Loaded ensemble should make identical predictions")
    
    def test_performance_comparison_workflow(self, sample_training_data):
        """Test workflow for comparing enhanced vs baseline performance"""
        X, y = sample_training_data
        
        # Create baseline model
        baseline_model = RandomForestClassifier(n_estimators=50, random_state=42)
        baseline_model.fit(X[:150], y[:150])
        baseline_predictions = baseline_model.predict(X[150:])
        baseline_accuracy = np.mean(baseline_predictions == y[150:])
        
        # Create enhanced model with optimization
        optimizer = TennisBayesianOptimizer(n_calls=5, cv_folds=3)  # Minimal for testing
        optimization_result = optimizer.optimize_model('random_forest', X[:150], y[:150])
        
        # Train optimized model
        if 'best_params' in optimization_result and optimization_result['best_params']:
            enhanced_model = RandomForestClassifier(**optimization_result['best_params'], random_state=42)
        else:
            enhanced_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        
        enhanced_model.fit(X[:150], y[:150])
        enhanced_predictions = enhanced_model.predict(X[150:])
        enhanced_accuracy = np.mean(enhanced_predictions == y[150:])
        
        # Both models should achieve reasonable performance
        assert baseline_accuracy > 0.3, f"Baseline accuracy too low: {baseline_accuracy}"
        assert enhanced_accuracy > 0.3, f"Enhanced accuracy too low: {enhanced_accuracy}"
        
        # Performance should be in reasonable range for random data
        assert 0.3 <= baseline_accuracy <= 0.8, "Baseline accuracy should be reasonable"
        assert 0.3 <= enhanced_accuracy <= 0.8, "Enhanced accuracy should be reasonable"
    
    def test_end_to_end_prediction_pipeline(self):
        """Test complete prediction pipeline from features to final prediction"""
        # Create feature engineer
        engineer = EnhancedTennisFeatureEngineer()
        
        # Create sample match data
        match_data = {
            'player1': {
                'ranking': 30, 'age': 25, 'last_match_date': datetime.now() - timedelta(days=2),
                'matches_last_14_days': 3, 'hard_win_percentage': 0.70, 'indoor_win_percentage': 0.65
            },
            'player2': {
                'ranking': 120, 'age': 27, 'last_match_date': datetime.now() - timedelta(days=6),
                'matches_last_14_days': 2, 'hard_win_percentage': 0.55, 'indoor_win_percentage': 0.50
            },
            'first_set_stats': FirstSetStats(
                winner='player2', score='7-5', duration_minutes=55, total_games=12,
                break_points_player1={'faced': 4, 'saved': 2, 'converted': 1},
                break_points_player2={'faced': 2, 'saved': 1, 'converted': 2},
                service_points_player1={'won': 40, 'total': 60}, service_points_player2={'won': 35, 'total': 50},
                unforced_errors_player1=15, unforced_errors_player2=10,
                winners_player1=12, winners_player2=18, double_faults_player1=3, double_faults_player2=1
            ),
            'match_context': MatchContext(
                tournament_tier='Masters1000', surface='Hard', round='QF', is_indoor=True, altitude=200.0
            ),
            'h2h_data': {'overall': {'player1_wins': 0, 'player2_wins': 1}, 'recent_3': {'player1_wins': 0, 'player2_wins': 1}},
            'match_date': datetime.now()
        }
        
        # Generate enhanced features
        features = engineer.create_all_enhanced_features(match_data)
        assert len(features) > 0, "Should generate enhanced features"
        
        # Create feature vector
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        
        # Create and train a simple model (for demonstration)
        # In practice, this would be your trained tennis prediction model
        dummy_model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Create dummy training data with same feature dimensions
        n_features = feature_vector.shape[1]
        dummy_X = np.random.randn(100, n_features)
        dummy_y = np.random.choice([0, 1], size=100)
        dummy_model.fit(dummy_X, dummy_y)
        
        # Make prediction
        prediction = dummy_model.predict(feature_vector)[0]
        prediction_proba = dummy_model.predict_proba(feature_vector)[0]
        
        # Validate prediction format
        assert prediction in [0, 1], "Prediction should be binary"
        assert len(prediction_proba) == 2, "Should have probabilities for both classes"
        assert abs(sum(prediction_proba) - 1.0) < 1e-6, "Probabilities should sum to 1"
        
        # Create match context for dynamic ensemble
        context = MatchContext(
            surface=match_data['match_context'].surface,
            tournament_tier=match_data['match_context'].tournament_tier,
            round=match_data['match_context'].round,
            is_indoor=match_data['match_context'].is_indoor,
            player1_ranking=match_data['player1']['ranking'],
            player2_ranking=match_data['player2']['ranking'],
            ranking_gap=abs(match_data['player1']['ranking'] - match_data['player2']['ranking']),
            is_upset_scenario=abs(match_data['player1']['ranking'] - match_data['player2']['ranking']) > 50,
            surface_specialization={'player1': 0.7, 'player2': 0.55},
            h2h_history=1, tournament_importance=0.9
        )
        
        # Test weight calculation
        calculator = ContextualWeightCalculator()
        weights = calculator.calculate_contextual_weights(context)
        
        assert abs(sum(weights.values()) - 1.0) < 1e-6, "Weights should sum to 1"
        assert all(w > 0 for w in weights.values()), "All weights should be positive"

if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v"])