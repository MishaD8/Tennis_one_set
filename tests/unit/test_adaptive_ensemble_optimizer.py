"""
Unit tests for Adaptive Ensemble Optimizer
"""
import pytest
import numpy as np
import json
import os
import tempfile
from datetime import datetime, timedelta
from collections import deque
from unittest.mock import Mock, patch, MagicMock

# Import the module under test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from adaptive_ensemble_optimizer import (
    ModelPerformance, EnsembleOptimizer, 
    get_ensemble_optimizer, get_optimized_weights,
    record_model_predictions, init_ensemble_optimizer
)


class TestModelPerformance:
    """Test suite for ModelPerformance class"""
    
    @pytest.fixture
    def model_performance(self):
        """Create ModelPerformance instance"""
        return ModelPerformance(
            model_name="test_model",
            predictions=deque(maxlen=50),
            accuracy_window=50
        )
    
    def test_init_default_values(self):
        """Test ModelPerformance initialization with default values"""
        mp = ModelPerformance(model_name="neural_network", predictions=deque())
        
        assert mp.model_name == "neural_network"
        assert isinstance(mp.predictions, deque)
        assert mp.accuracy_window == 50
        assert mp.predictions.maxlen == 50
    
    def test_init_with_custom_window(self):
        """Test ModelPerformance initialization with custom window size"""
        mp = ModelPerformance(
            model_name="xgboost", 
            predictions=deque(),
            accuracy_window=100
        )
        
        assert mp.accuracy_window == 100
        assert mp.predictions.maxlen == 100
    
    def test_post_init_converts_predictions(self):
        """Test that __post_init__ converts predictions to deque if needed"""
        mp = ModelPerformance(
            model_name="test_model",
            predictions=[],  # List instead of deque
            accuracy_window=30
        )
        
        assert isinstance(mp.predictions, deque)
        assert mp.predictions.maxlen == 30
    
    def test_add_prediction_correct(self, model_performance):
        """Test adding a correct prediction"""
        model_performance.add_prediction(0.8, 1.0)  # Correct prediction (>0.5 threshold)
        
        assert len(model_performance.predictions) == 1
        prediction = model_performance.predictions[0]
        
        assert prediction['prediction'] == 0.8
        assert prediction['actual'] == 1.0
        assert prediction['correct'] is True
        assert prediction['confidence'] == 0.3  # |0.8 - 0.5|
        assert 'timestamp' in prediction
        assert prediction['match_info'] == {}
    
    def test_add_prediction_incorrect(self, model_performance):
        """Test adding an incorrect prediction"""
        model_performance.add_prediction(0.8, 0.0)  # Incorrect prediction
        
        prediction = model_performance.predictions[0]
        assert prediction['correct'] is False
    
    def test_add_prediction_with_match_info(self, model_performance):
        """Test adding prediction with match information"""
        match_info = {
            'surface': 'Clay',
            'tournament': 'French Open',
            'player_rank': 5
        }
        
        model_performance.add_prediction(0.7, 1.0, match_info)
        
        prediction = model_performance.predictions[0]
        assert prediction['match_info'] == match_info
    
    def test_deque_maxlen_behavior(self, model_performance):
        """Test that deque respects maxlen and removes old predictions"""
        # Add more predictions than maxlen
        for i in range(60):
            model_performance.add_prediction(0.6, 1.0)
        
        # Should only keep the last 50 predictions
        assert len(model_performance.predictions) == 50
    
    def test_get_recent_accuracy_with_data(self, model_performance):
        """Test calculating recent accuracy with valid data"""
        # Add recent correct predictions
        for i in range(5):
            model_performance.add_prediction(0.8, 1.0)  # Correct
        
        # Add recent incorrect predictions
        for i in range(3):
            model_performance.add_prediction(0.8, 0.0)  # Incorrect
            
        accuracy = model_performance.get_recent_accuracy(days=7)
        expected_accuracy = 5 / 8  # 5 correct out of 8 total
        
        assert accuracy == expected_accuracy
    
    def test_get_recent_accuracy_no_data(self, model_performance):
        """Test calculating recent accuracy with no data"""
        accuracy = model_performance.get_recent_accuracy(days=7)
        assert accuracy == 0.5  # Default accuracy when no data
    
    def test_get_recent_accuracy_old_data_only(self, model_performance):
        """Test recent accuracy when all data is older than requested period"""
        # Mock old timestamps
        old_prediction = {
            'prediction': 0.8,
            'actual': 1.0,
            'correct': True,
            'confidence': 0.3,
            'timestamp': (datetime.now() - timedelta(days=10)).isoformat(),
            'match_info': {}
        }
        
        model_performance.predictions.append(old_prediction)
        
        accuracy = model_performance.get_recent_accuracy(days=7)
        assert accuracy == 0.5  # Should return default when no recent data
    
    def test_confidence_calculation(self, model_performance):
        """Test confidence calculation for different prediction values"""
        test_cases = [
            (0.5, 0.0),  # Lowest confidence
            (0.0, 0.5),  # High confidence
            (1.0, 0.5),  # High confidence
            (0.7, 0.2),  # Medium confidence
            (0.3, 0.2)   # Medium confidence
        ]
        
        for prediction, expected_confidence in test_cases:
            model_performance.add_prediction(prediction, 1.0)
            actual_confidence = model_performance.predictions[-1]['confidence']
            assert actual_confidence == expected_confidence


class TestEnsembleOptimizer:
    """Test suite for EnsembleOptimizer class"""
    
    @pytest.fixture
    def base_weights(self):
        """Base ensemble weights for testing"""
        return {
            'neural_network': 0.25,
            'xgboost': 0.20,
            'random_forest': 0.20,
            'gradient_boosting': 0.20,
            'logistic_regression': 0.15
        }
    
    @pytest.fixture
    def ensemble_optimizer(self, base_weights):
        """Create EnsembleOptimizer instance"""
        return EnsembleOptimizer(base_weights=base_weights)
    
    def test_init_creates_model_performance_trackers(self, ensemble_optimizer, base_weights):
        """Test that initialization creates ModelPerformance for each model"""
        assert len(ensemble_optimizer.model_performances) == len(base_weights)
        
        for model_name in base_weights.keys():
            assert model_name in ensemble_optimizer.model_performances
            assert isinstance(ensemble_optimizer.model_performances[model_name], ModelPerformance)
            assert ensemble_optimizer.model_performances[model_name].model_name == model_name
    
    def test_init_sets_base_weights(self, ensemble_optimizer, base_weights):
        """Test that base weights are properly set"""
        assert ensemble_optimizer.base_weights == base_weights
        assert ensemble_optimizer.current_weights == base_weights
    
    def test_init_default_parameters(self, ensemble_optimizer):
        """Test default parameter values"""
        assert ensemble_optimizer.learning_rate == 0.1
        assert ensemble_optimizer.confidence_threshold == 0.6
        assert ensemble_optimizer.min_predictions == 10
        assert ensemble_optimizer.max_weight_change == 0.1
    
    def test_record_predictions_all_models(self, ensemble_optimizer):
        """Test recording predictions for all models"""
        predictions = {
            'neural_network': 0.8,
            'xgboost': 0.7,
            'random_forest': 0.75,
            'gradient_boosting': 0.72,
            'logistic_regression': 0.73
        }
        actual_result = 1.0
        match_context = {'surface': 'Clay', 'tournament': 'French Open'}
        
        ensemble_optimizer.record_predictions(predictions, actual_result, match_context)
        
        # Check that all models have recorded the prediction
        for model_name, prediction in predictions.items():
            model_perf = ensemble_optimizer.model_performances[model_name]
            assert len(model_perf.predictions) == 1
            recorded = model_perf.predictions[0]
            assert recorded['prediction'] == prediction
            assert recorded['actual'] == actual_result
            assert recorded['match_info'] == match_context
    
    def test_record_predictions_partial_models(self, ensemble_optimizer):
        """Test recording predictions when not all models provide predictions"""
        predictions = {
            'neural_network': 0.8,
            'xgboost': 0.7
            # Missing other models
        }
        actual_result = 1.0
        
        ensemble_optimizer.record_predictions(predictions, actual_result)
        
        # Only specified models should have predictions recorded
        assert len(ensemble_optimizer.model_performances['neural_network'].predictions) == 1
        assert len(ensemble_optimizer.model_performances['xgboost'].predictions) == 1
        assert len(ensemble_optimizer.model_performances['random_forest'].predictions) == 0
    
    def test_calculate_performance_scores_sufficient_data(self, ensemble_optimizer):
        """Test performance score calculation with sufficient data"""
        # Add enough predictions to exceed min_predictions threshold
        model_perf = ensemble_optimizer.model_performances['neural_network']
        
        # Add 15 predictions (> min_predictions = 10)
        for i in range(15):
            correct = i < 12  # 12 correct, 3 incorrect = 80% accuracy
            actual = 1.0 if correct else 0.0
            model_perf.add_prediction(0.8, actual)
        
        scores = ensemble_optimizer._calculate_performance_scores()
        
        # Neural network should have a score based on 80% accuracy
        assert 'neural_network' in scores
        assert scores['neural_network'] > 0  # Should be positive for good performance
        
        # Models without sufficient data should have score of 0
        assert scores['xgboost'] == 0
    
    def test_calculate_performance_scores_insufficient_data(self, ensemble_optimizer):
        """Test performance scores when models have insufficient data"""
        # Add only a few predictions (< min_predictions)
        model_perf = ensemble_optimizer.model_performances['neural_network']
        for i in range(5):
            model_perf.add_prediction(0.8, 1.0)
        
        scores = ensemble_optimizer._calculate_performance_scores()
        
        # All models should have score of 0 due to insufficient data
        for score in scores.values():
            assert score == 0
    
    def test_adjust_weights_performance_based(self, ensemble_optimizer):
        """Test weight adjustment based on performance"""
        # Setup: Give neural_network good performance, xgboost poor performance
        nn_perf = ensemble_optimizer.model_performances['neural_network']
        xgb_perf = ensemble_optimizer.model_performances['xgboost']
        
        # Neural network: 90% accuracy (excellent)
        for i in range(15):
            actual = 1.0 if i < 13 else 0.0  # 13 correct, 2 incorrect
            nn_perf.add_prediction(0.8, actual)
        
        # XGBoost: 40% accuracy (poor)
        for i in range(15):
            actual = 1.0 if i < 6 else 0.0  # 6 correct, 9 incorrect
            xgb_perf.add_prediction(0.8, actual)
        
        original_weights = ensemble_optimizer.current_weights.copy()
        new_weights = ensemble_optimizer._adjust_weights()
        
        # Neural network weight should increase
        assert new_weights['neural_network'] > original_weights['neural_network']
        
        # XGBoost weight should decrease
        assert new_weights['xgboost'] < original_weights['xgboost']
        
        # Weights should still sum to approximately 1
        assert abs(sum(new_weights.values()) - 1.0) < 0.01
    
    def test_adjust_weights_max_change_limit(self, ensemble_optimizer):
        """Test that weight changes are limited by max_weight_change"""
        # Setup extreme performance difference
        nn_perf = ensemble_optimizer.model_performances['neural_network']
        
        # Perfect performance for neural network
        for i in range(20):
            nn_perf.add_prediction(0.9, 1.0)  # All correct
        
        original_weight = ensemble_optimizer.current_weights['neural_network']
        new_weights = ensemble_optimizer._adjust_weights()
        
        # Weight change should be limited by max_weight_change (0.1)
        weight_change = abs(new_weights['neural_network'] - original_weight)
        assert weight_change <= ensemble_optimizer.max_weight_change + 0.001  # Small tolerance
    
    def test_get_optimized_weights_no_context(self, ensemble_optimizer):
        """Test getting optimized weights without match context"""
        # Add some performance data
        nn_perf = ensemble_optimizer.model_performances['neural_network']
        for i in range(15):
            nn_perf.add_prediction(0.8, 1.0)
        
        weights = ensemble_optimizer.get_optimized_weights()
        
        assert isinstance(weights, dict)
        assert len(weights) == len(ensemble_optimizer.base_weights)
        assert abs(sum(weights.values()) - 1.0) < 0.01
    
    def test_get_optimized_weights_with_context(self, ensemble_optimizer):
        """Test getting optimized weights with match context"""
        match_context = {
            'surface': 'Clay',
            'tournament_importance': 'Grand Slam',
            'rank_difference': 50
        }
        
        weights = ensemble_optimizer.get_optimized_weights(match_context)
        
        assert isinstance(weights, dict)
        assert len(weights) == len(ensemble_optimizer.base_weights)
    
    def test_contextual_adjustments_surface_specific(self, ensemble_optimizer):
        """Test contextual weight adjustments for different surfaces"""
        clay_context = {'surface': 'Clay'}
        hard_context = {'surface': 'Hard'}
        
        clay_weights = ensemble_optimizer._apply_contextual_adjustments(
            ensemble_optimizer.current_weights.copy(), clay_context
        )
        hard_weights = ensemble_optimizer._apply_contextual_adjustments(
            ensemble_optimizer.current_weights.copy(), hard_context
        )
        
        # Weights might be different for different surfaces
        # (specific behavior depends on implementation)
        assert isinstance(clay_weights, dict)
        assert isinstance(hard_weights, dict)
        
        # Weights should still sum to 1
        assert abs(sum(clay_weights.values()) - 1.0) < 0.01
        assert abs(sum(hard_weights.values()) - 1.0) < 0.01
    
    def test_save_and_load_state(self, ensemble_optimizer, temp_dir):
        """Test saving and loading optimizer state"""
        # Add some performance data
        nn_perf = ensemble_optimizer.model_performances['neural_network']
        for i in range(5):
            nn_perf.add_prediction(0.8, 1.0)
        
        # Save state
        state_file = os.path.join(temp_dir, 'optimizer_state.json')
        ensemble_optimizer.save_state(state_file)
        
        assert os.path.exists(state_file)
        
        # Create new optimizer and load state
        new_optimizer = EnsembleOptimizer(base_weights=ensemble_optimizer.base_weights)
        new_optimizer.load_state(state_file)
        
        # Check that state was loaded correctly
        assert len(new_optimizer.model_performances['neural_network'].predictions) == 5
        assert new_optimizer.current_weights == ensemble_optimizer.current_weights


class TestModuleFunctions:
    """Test module-level functions"""
    
    def test_init_ensemble_optimizer(self, mock_config):
        """Test initializing the global ensemble optimizer"""
        base_weights = mock_config['models']['ensemble_weights']
        
        # Clear any existing optimizer
        import adaptive_ensemble_optimizer
        adaptive_ensemble_optimizer._global_optimizer = None
        
        result = init_ensemble_optimizer(base_weights)
        
        assert result is True
        assert get_ensemble_optimizer() is not None
    
    def test_get_ensemble_optimizer_not_initialized(self):
        """Test getting optimizer when not initialized"""
        import adaptive_ensemble_optimizer
        adaptive_ensemble_optimizer._global_optimizer = None
        
        optimizer = get_ensemble_optimizer()
        assert optimizer is None
    
    def test_get_optimized_weights_no_optimizer(self):
        """Test getting weights when no optimizer is initialized"""
        import adaptive_ensemble_optimizer
        adaptive_ensemble_optimizer._global_optimizer = None
        
        weights = get_optimized_weights()
        assert weights is None
    
    @patch('adaptive_ensemble_optimizer.get_ensemble_optimizer')
    def test_record_model_predictions_success(self, mock_get_optimizer):
        """Test recording model predictions through module function"""
        mock_optimizer = Mock()
        mock_get_optimizer.return_value = mock_optimizer
        
        predictions = {'neural_network': 0.8, 'xgboost': 0.7}
        actual_result = 1.0
        match_context = {'surface': 'Clay'}
        
        result = record_model_predictions(predictions, actual_result, match_context)
        
        assert result is True
        mock_optimizer.record_predictions.assert_called_once_with(
            predictions, actual_result, match_context
        )
    
    @patch('adaptive_ensemble_optimizer.get_ensemble_optimizer')
    def test_record_model_predictions_no_optimizer(self, mock_get_optimizer):
        """Test recording predictions when no optimizer exists"""
        mock_get_optimizer.return_value = None
        
        result = record_model_predictions({}, 1.0)
        
        assert result is False
    
    @patch('adaptive_ensemble_optimizer.get_ensemble_optimizer')
    def test_record_model_predictions_exception(self, mock_get_optimizer):
        """Test recording predictions when optimizer raises exception"""
        mock_optimizer = Mock()
        mock_optimizer.record_predictions.side_effect = Exception("Test error")
        mock_get_optimizer.return_value = mock_optimizer
        
        result = record_model_predictions({}, 1.0)
        
        assert result is False


@pytest.mark.integration
class TestAdaptiveOptimizerIntegration:
    """Integration tests for the adaptive optimizer system"""
    
    def test_complete_optimization_workflow(self, mock_config):
        """Test complete workflow from initialization to weight optimization"""
        base_weights = mock_config['models']['ensemble_weights']
        
        # Initialize optimizer
        assert init_ensemble_optimizer(base_weights) is True
        
        optimizer = get_ensemble_optimizer()
        assert optimizer is not None
        
        # Simulate several matches with predictions
        match_data = [
            ({'neural_network': 0.8, 'xgboost': 0.6}, 1.0, {'surface': 'Clay'}),
            ({'neural_network': 0.9, 'xgboost': 0.7}, 1.0, {'surface': 'Hard'}),
            ({'neural_network': 0.4, 'xgboost': 0.3}, 0.0, {'surface': 'Grass'}),
            ({'neural_network': 0.7, 'xgboost': 0.8}, 1.0, {'surface': 'Clay'}),
        ]
        
        # Record predictions
        for predictions, actual, context in match_data:
            assert record_model_predictions(predictions, actual, context) is True
        
        # Get optimized weights
        optimized_weights = get_optimized_weights({'surface': 'Clay'})
        
        assert optimized_weights is not None
        assert isinstance(optimized_weights, dict)
        assert abs(sum(optimized_weights.values()) - 1.0) < 0.01
    
    def test_performance_tracking_accuracy(self, mock_config):
        """Test that performance tracking correctly calculates accuracy"""
        base_weights = mock_config['models']['ensemble_weights']
        init_ensemble_optimizer(base_weights)
        
        optimizer = get_ensemble_optimizer()
        
        # Add many predictions with known accuracy
        correct_predictions = 15
        total_predictions = 20
        
        for i in range(total_predictions):
            prediction = 0.8
            actual = 1.0 if i < correct_predictions else 0.0
            
            record_model_predictions(
                {'neural_network': prediction}, 
                actual, 
                {'surface': 'Clay'}
            )
        
        # Check accuracy calculation
        nn_performance = optimizer.model_performances['neural_network']
        accuracy = nn_performance.get_recent_accuracy(days=7)
        expected_accuracy = correct_predictions / total_predictions
        
        assert accuracy == expected_accuracy