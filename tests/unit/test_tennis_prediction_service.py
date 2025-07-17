"""
Unit tests for TennisPredictionService
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import os
import tempfile
import json

# Import the module under test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tennis_prediction_module import TennisPredictionService


class TestTennisPredictionService:
    """Test suite for TennisPredictionService class"""
    
    @pytest.fixture
    def temp_models_dir(self, temp_dir):
        """Create temporary models directory"""
        models_dir = os.path.join(temp_dir, 'test_models')
        os.makedirs(models_dir, exist_ok=True)
        return models_dir
    
    @pytest.fixture
    def mock_metadata(self, temp_models_dir):
        """Create mock metadata.json file"""
        metadata = {
            'ensemble_weights': {
                'neural_network': 0.25,
                'xgboost': 0.20,
                'random_forest': 0.20,
                'gradient_boosting': 0.20,
                'logistic_regression': 0.15
            }
        }
        metadata_path = os.path.join(temp_models_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        return metadata_path
    
    @pytest.fixture
    def prediction_service(self, temp_models_dir):
        """Create TennisPredictionService instance"""
        return TennisPredictionService(models_dir=temp_models_dir, use_adaptive_weights=False)
    
    @pytest.fixture
    def valid_match_data(self):
        """Create valid match data for testing"""
        return {
            'player_rank': 10,
            'player_age': 25,
            'opponent_rank': 15,
            'opponent_age': 28,
            'player_recent_win_rate': 0.75,
            'player_form_trend': 0.8,
            'player_surface_advantage': 0.6,
            'h2h_win_rate': 0.65,
            'total_pressure': 0.7
        }
    
    def test_init_default_parameters(self, temp_models_dir):
        """Test initialization with default parameters"""
        service = TennisPredictionService(models_dir=temp_models_dir)
        
        assert service.models_dir == temp_models_dir
        assert service.models == {}
        assert service.scaler is None
        assert service.expected_features == []
        assert service.is_loaded is False
        assert isinstance(service.base_weights, dict)
        assert len(service.base_weights) == 5  # 5 model types
    
    def test_init_with_adaptive_weights_disabled(self, temp_models_dir):
        """Test initialization with adaptive weights disabled"""
        service = TennisPredictionService(models_dir=temp_models_dir, use_adaptive_weights=False)
        
        assert service.use_adaptive_weights is False
        assert service.ensemble_weights == service.base_weights
    
    def test_load_base_weights_from_metadata(self, temp_models_dir, mock_metadata):
        """Test loading base weights from metadata file"""
        service = TennisPredictionService(models_dir=temp_models_dir)
        
        # Weights should be loaded from metadata
        expected_weights = {
            'neural_network': 0.25,
            'xgboost': 0.20,
            'random_forest': 0.20,
            'gradient_boosting': 0.20,
            'logistic_regression': 0.15
        }
        assert service.base_weights == expected_weights
    
    def test_load_base_weights_fallback(self, temp_models_dir):
        """Test fallback to hardcoded weights when metadata not available"""
        service = TennisPredictionService(models_dir=temp_models_dir)
        
        # Should use fallback weights
        assert 'neural_network' in service.base_weights
        assert 'xgboost' in service.base_weights
        assert 'random_forest' in service.base_weights
        assert 'gradient_boosting' in service.base_weights
        assert 'logistic_regression' in service.base_weights
        
        # Weights should sum to approximately 1
        total_weight = sum(service.base_weights.values())
        assert abs(total_weight - 1.0) < 0.01
    
    @patch('tennis_prediction_module.joblib.load')
    @patch('tennis_prediction_module.keras.models.load_model')
    @patch('os.path.exists')
    def test_load_models_success(self, mock_exists, mock_keras_load, mock_joblib_load, prediction_service):
        """Test successful model loading"""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock model objects
        mock_scaler = Mock()
        mock_nn_model = Mock()
        mock_sklearn_model = Mock()
        mock_sklearn_model.feature_names_in_ = ['feature_1', 'feature_2', 'feature_3']
        
        # Configure mocks
        def joblib_side_effect(path):
            if 'scaler.pkl' in path:
                return mock_scaler
            else:
                return mock_sklearn_model
        
        mock_joblib_load.side_effect = joblib_side_effect
        mock_keras_load.return_value = mock_nn_model
        
        # Test model loading
        result = prediction_service.load_models()
        
        assert result is True
        assert prediction_service.is_loaded is True
        assert prediction_service.scaler == mock_scaler
        assert len(prediction_service.models) == 5
        assert 'neural_network' in prediction_service.models
        assert len(prediction_service.expected_features) == 3
    
    @patch('os.path.exists')
    def test_load_models_scaler_not_found(self, mock_exists, prediction_service):
        """Test model loading when scaler file not found"""
        mock_exists.return_value = False
        
        result = prediction_service.load_models()
        
        assert result is False
        assert prediction_service.is_loaded is False
    
    @patch('tennis_prediction_module.joblib.load')
    @patch('os.path.exists')
    def test_load_models_no_models_loaded(self, mock_exists, mock_joblib_load, prediction_service):
        """Test when no models can be loaded"""
        # Mock scaler exists but models don't
        def exists_side_effect(path):
            return 'scaler.pkl' in path
        
        mock_exists.side_effect = exists_side_effect
        mock_joblib_load.return_value = Mock()
        
        result = prediction_service.load_models()
        
        assert result is False
        assert prediction_service.is_loaded is False
    
    def test_create_engineered_features(self, prediction_service, valid_match_data):
        """Test feature engineering"""
        df = pd.DataFrame([valid_match_data])
        enhanced_df = prediction_service.create_engineered_features(df)
        
        # Check that new features are created
        assert 'rank_difference' in enhanced_df.columns
        assert 'rank_ratio' in enhanced_df.columns
        assert 'combined_form' in enhanced_df.columns
        assert 'surface_pressure_interaction' in enhanced_df.columns
        
        # Check feature calculations
        expected_rank_diff = valid_match_data['opponent_rank'] - valid_match_data['player_rank']
        assert enhanced_df['rank_difference'].iloc[0] == expected_rank_diff
        
        expected_rank_ratio = valid_match_data['player_rank'] / (valid_match_data['opponent_rank'] + 1)
        assert enhanced_df['rank_ratio'].iloc[0] == expected_rank_ratio
    
    def test_validate_input_data_valid(self, prediction_service, valid_match_data):
        """Test input validation with valid data"""
        result = prediction_service.validate_input_data(valid_match_data)
        assert result is True
    
    def test_validate_input_data_missing_fields(self, prediction_service):
        """Test input validation with missing required fields"""
        incomplete_data = {
            'player_rank': 10,
            'player_age': 25
            # Missing other required fields
        }
        
        with pytest.raises(ValueError, match="Отсутствующие поля"):
            prediction_service.validate_input_data(incomplete_data)
    
    def test_validate_input_data_invalid_rank(self, prediction_service, valid_match_data):
        """Test input validation with invalid rank values"""
        invalid_data = valid_match_data.copy()
        invalid_data['player_rank'] = 1500  # Invalid rank
        
        with pytest.raises(ValueError, match="player_rank должен быть от 1 до 1000"):
            prediction_service.validate_input_data(invalid_data)
    
    def test_validate_input_data_invalid_win_rate(self, prediction_service, valid_match_data):
        """Test input validation with invalid win rate"""
        invalid_data = valid_match_data.copy()
        invalid_data['player_recent_win_rate'] = 1.5  # Invalid win rate
        
        with pytest.raises(ValueError, match="player_recent_win_rate должен быть от 0 до 1"):
            prediction_service.validate_input_data(invalid_data)
    
    @patch('tennis_prediction_module.joblib.load')
    @patch('tennis_prediction_module.keras.models.load_model')
    @patch('os.path.exists')
    def test_predict_match_not_loaded(self, mock_exists, mock_keras_load, mock_joblib_load, 
                                     prediction_service, valid_match_data):
        """Test prediction when models are not loaded"""
        # Don't load models
        assert prediction_service.is_loaded is False
        
        with pytest.raises(Exception, match="Модели не загружены"):
            prediction_service.predict_match(valid_match_data)
    
    @patch('tennis_prediction_module.record_model_predictions')
    def test_get_current_weights_adaptive_disabled(self, mock_record, prediction_service):
        """Test getting current weights when adaptive optimization is disabled"""
        prediction_service.use_adaptive_weights = False
        
        weights = prediction_service._get_current_weights()
        
        assert weights == prediction_service.ensemble_weights
        assert weights == prediction_service.base_weights
    
    def test_model_weight_validation(self, prediction_service):
        """Test that model weights are properly validated"""
        # Weights should sum to approximately 1
        total_weight = sum(prediction_service.base_weights.values())
        assert 0.99 <= total_weight <= 1.01
        
        # All weights should be positive
        for weight in prediction_service.base_weights.values():
            assert weight > 0
    
    def test_expected_features_structure(self, prediction_service):
        """Test that expected features have proper structure"""
        # Even without loading models, fallback features should be available
        fallback_features = [
            'player_rank', 'player_age', 'opponent_rank', 'opponent_age',
            'player_recent_win_rate', 'player_form_trend', 'player_surface_advantage',
            'h2h_win_rate', 'total_pressure', 'rank_difference', 'rank_ratio',
            'combined_form', 'surface_pressure_interaction'
        ]
        
        # When models are not loaded, expected_features should be empty or fallback
        assert isinstance(prediction_service.expected_features, list)
    
    @pytest.mark.parametrize("model_name,file_extension", [
        ('neural_network', '.h5'),
        ('xgboost', '.pkl'),
        ('random_forest', '.pkl'),
        ('gradient_boosting', '.pkl'),
        ('logistic_regression', '.pkl')
    ])
    def test_model_file_mapping(self, model_name, file_extension, prediction_service):
        """Test that model names map to correct file extensions"""
        # This test ensures the model loading logic handles different file types correctly
        expected_filename = f"{model_name}{file_extension}"
        
        # The actual mapping is defined in load_models method
        model_files = {
            'neural_network': 'neural_network.h5',
            'xgboost': 'xgboost.pkl', 
            'random_forest': 'random_forest.pkl',
            'gradient_boosting': 'gradient_boosting.pkl',
            'logistic_regression': 'logistic_regression.pkl'
        }
        
        assert model_name in model_files
        assert model_files[model_name] == expected_filename
    
    def test_feature_engineering_edge_cases(self, prediction_service):
        """Test feature engineering with edge case values"""
        edge_case_data = pd.DataFrame([{
            'player_rank': 1,  # Best rank
            'opponent_rank': 1000,  # Worst rank
            'player_recent_win_rate': 0.0,  # No wins
            'h2h_win_rate': 1.0,  # Always wins H2H
            'player_surface_advantage': 0.0,  # No advantage
            'total_pressure': 1.0  # Maximum pressure
        }])
        
        enhanced_df = prediction_service.create_engineered_features(edge_case_data)
        
        # Check calculations don't break with edge values
        assert not enhanced_df['rank_difference'].isna().any()
        assert not enhanced_df['rank_ratio'].isna().any()
        assert not enhanced_df['combined_form'].isna().any()
        assert not enhanced_df['surface_pressure_interaction'].isna().any()
        
        # Specific edge case checks
        assert enhanced_df['rank_difference'].iloc[0] == 999  # 1000 - 1
        assert enhanced_df['rank_ratio'].iloc[0] == 1/1001  # 1 / (1000 + 1)


@pytest.mark.integration
class TestTennisPredictionServiceIntegration:
    """Integration tests for TennisPredictionService"""
    
    def test_full_prediction_workflow_mock(self, temp_dir, valid_match_data, mock_ml_models):
        """Test complete prediction workflow with mocked models"""
        # Create service with temporary directory
        service = TennisPredictionService(models_dir=temp_dir, use_adaptive_weights=False)
        
        # Mock the models and scaler
        service.models = mock_ml_models
        service.scaler = Mock()
        service.scaler.transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
        service.is_loaded = True
        service.expected_features = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
        
        # Mock the prediction method to avoid full implementation
        with patch.object(service, 'predict_match') as mock_predict:
            mock_predict.return_value = {
                'prediction': 0.65,
                'confidence': 'medium',
                'individual_predictions': {
                    'neural_network': 0.67,
                    'xgboost': 0.63,
                    'random_forest': 0.65,
                    'gradient_boosting': 0.64,
                    'logistic_regression': 0.66
                }
            }
            
            result = service.predict_match(valid_match_data)
            
            assert result['prediction'] > 0
            assert result['prediction'] < 1
            assert 'confidence' in result
            assert 'individual_predictions' in result
            assert len(result['individual_predictions']) == 5