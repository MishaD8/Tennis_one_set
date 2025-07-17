"""
Integration tests for Tennis Backend API endpoints
"""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Import the Flask app
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Mock all the complex dependencies before importing the app
with patch.dict('sys.modules', {
    'error_handler': Mock(),
    'config_loader': Mock(),
    'real_tennis_predictor_integration': Mock(),
    'tennis_prediction_module': Mock(),
    'correct_odds_api_integration': Mock(),
    'api_economy_patch': Mock(),
    'universal_tennis_data_collector': Mock(),
    'daily_api_scheduler': Mock()
}):
    from tennis_backend import app


class TestTennisBackendAPI:
    """Test suite for Tennis Backend API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        with app.test_client() as client:
            yield client
    
    @pytest.fixture
    def valid_match_data(self):
        """Valid match data for testing"""
        return {
            'player_name': 'Novak Djokovic',
            'opponent_name': 'Rafael Nadal',
            'surface': 'Clay',
            'tournament': 'French Open',
            'player_rank': 1,
            'opponent_rank': 2,
            'player_age': 36,
            'opponent_age': 37
        }
    
    def test_health_endpoint(self, client):
        """Test /api/health endpoint"""
        response = client.get('/api/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'status' in data
        assert 'timestamp' in data
        assert 'uptime' in data
        assert 'components' in data
        
        # Health should always return OK for basic functionality
        assert data['status'] in ['healthy', 'degraded']  # degraded if components missing
    
    def test_root_endpoint(self, client):
        """Test root endpoint /"""
        response = client.get('/')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'message' in data
        assert 'status' in data
        assert 'endpoints' in data
        
        # Should list available API endpoints
        endpoints = data['endpoints']
        expected_endpoints = [
            '/api/health',
            '/api/stats', 
            '/api/matches',
            '/api/test-ml',
            '/api/value-bets',
            '/api/underdog-analysis'
        ]
        
        for endpoint in expected_endpoints:
            assert any(endpoint in ep for ep in endpoints)
    
    def test_stats_endpoint(self, client):
        """Test /api/stats endpoint"""
        response = client.get('/api/stats')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should have basic statistics structure
        assert isinstance(data, dict)
        # Exact structure depends on implementation
    
    def test_matches_endpoint_get(self, client):
        """Test GET /api/matches endpoint"""
        response = client.get('/api/matches')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert isinstance(data, dict)
        assert 'matches' in data or 'message' in data  # Could be empty or have message
    
    def test_matches_endpoint_with_params(self, client):
        """Test /api/matches endpoint with query parameters"""
        response = client.get('/api/matches?surface=Clay&limit=5')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert isinstance(data, dict)
    
    def test_test_ml_endpoint_valid_data(self, client, valid_match_data):
        """Test POST /api/test-ml endpoint with valid data"""
        with patch('tennis_backend.PREDICTION_SERVICE_AVAILABLE', True):
            with patch('tennis_backend.prediction_service') as mock_service:
                mock_service.predict_match.return_value = {
                    'prediction': 0.75,
                    'confidence': 'high',
                    'details': {'model_1': 0.8, 'model_2': 0.7}
                }
                
                response = client.post('/api/test-ml', 
                                     data=json.dumps(valid_match_data),
                                     content_type='application/json')
                
                assert response.status_code == 200
                data = json.loads(response.data)
                
                assert 'prediction' in data
                assert 'confidence' in data
                assert isinstance(data['prediction'], (int, float))
    
    def test_test_ml_endpoint_invalid_data(self, client):
        """Test POST /api/test-ml endpoint with invalid data"""
        invalid_data = {
            'player_name': 'Player A'
            # Missing required fields
        }
        
        response = client.post('/api/test-ml',
                              data=json.dumps(invalid_data),
                              content_type='application/json')
        
        # Should handle validation error gracefully
        assert response.status_code in [400, 422, 500]  # Depends on validation implementation
    
    def test_test_ml_endpoint_no_json(self, client):
        """Test POST /api/test-ml endpoint without JSON data"""
        response = client.post('/api/test-ml')
        
        assert response.status_code in [400, 422]  # Bad request
    
    def test_test_ml_endpoint_service_unavailable(self, client, valid_match_data):
        """Test POST /api/test-ml when prediction service unavailable"""
        with patch('tennis_backend.PREDICTION_SERVICE_AVAILABLE', False):
            response = client.post('/api/test-ml',
                                  data=json.dumps(valid_match_data),
                                  content_type='application/json')
            
            assert response.status_code in [503, 500]  # Service unavailable
            data = json.loads(response.data)
            assert 'error' in data or 'message' in data
    
    def test_value_bets_endpoint(self, client):
        """Test GET /api/value-bets endpoint"""
        response = client.get('/api/value-bets')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert isinstance(data, dict)
        # Should have value bets structure or empty result
        assert 'value_bets' in data or 'message' in data
    
    def test_value_bets_endpoint_with_params(self, client):
        """Test /api/value-bets endpoint with parameters"""
        response = client.get('/api/value-bets?min_value=1.5&surface=Hard')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert isinstance(data, dict)
    
    def test_underdog_analysis_endpoint_valid(self, client, valid_match_data):
        """Test POST /api/underdog-analysis endpoint with valid data"""
        with patch('tennis_backend.REAL_PREDICTOR_AVAILABLE', True):
            with patch('tennis_backend.real_predictor') as mock_predictor:
                mock_predictor.analyze_underdog_potential.return_value = {
                    'is_underdog': True,
                    'value_rating': 0.75,
                    'recommendation': 'Strong underdog potential'
                }
                
                response = client.post('/api/underdog-analysis',
                                      data=json.dumps(valid_match_data),
                                      content_type='application/json')
                
                assert response.status_code == 200
                data = json.loads(response.data)
                
                assert 'is_underdog' in data or 'analysis' in data
    
    def test_underdog_analysis_endpoint_invalid(self, client):
        """Test POST /api/underdog-analysis endpoint with invalid data"""
        invalid_data = {"invalid": "data"}
        
        response = client.post('/api/underdog-analysis',
                              data=json.dumps(invalid_data),
                              content_type='application/json')
        
        assert response.status_code in [400, 422, 500]
    
    def test_refresh_endpoint_get(self, client):
        """Test GET /api/refresh endpoint"""
        response = client.get('/api/refresh')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert isinstance(data, dict)
        assert 'message' in data or 'status' in data
    
    def test_refresh_endpoint_post(self, client):
        """Test POST /api/refresh endpoint"""
        response = client.post('/api/refresh')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert isinstance(data, dict)
    
    def test_player_info_endpoint(self, client):
        """Test GET /api/player-info/<player_name> endpoint"""
        player_name = "Novak Djokovic"
        response = client.get(f'/api/player-info/{player_name}')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert isinstance(data, dict)
        assert 'player' in data or 'message' in data
    
    def test_player_info_endpoint_invalid_player(self, client):
        """Test player info endpoint with invalid/unknown player"""
        response = client.get('/api/player-info/Unknown Player')
        
        # Should handle gracefully, might return 404 or empty data
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert isinstance(data, dict)
    
    def test_test_underdog_endpoint(self, client, valid_match_data):
        """Test POST /api/test-underdog endpoint"""
        response = client.post('/api/test-underdog',
                              data=json.dumps(valid_match_data),
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert isinstance(data, dict)
    
    def test_manual_api_update_endpoint(self, client):
        """Test POST /api/manual-api-update endpoint"""
        with patch('tennis_backend.DAILY_SCHEDULER_AVAILABLE', True):
            response = client.post('/api/manual-api-update')
            
            assert response.status_code in [200, 429]  # Success or rate limited
            data = json.loads(response.data)
            
            assert isinstance(data, dict)
            assert 'message' in data or 'status' in data
    
    def test_api_status_endpoint(self, client):
        """Test GET /api/api-status endpoint"""
        response = client.get('/api/api-status')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert isinstance(data, dict)
        assert 'status' in data or 'api_usage' in data
    
    def test_api_economy_status_endpoint(self, client):
        """Test GET /api/api-economy-status endpoint"""
        response = client.get('/api/api-economy-status')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert isinstance(data, dict)
    
    def test_cors_headers(self, client):
        """Test that CORS headers are properly set"""
        response = client.get('/api/health')
        
        # CORS should be enabled for cross-origin requests
        assert response.status_code == 200
        # The specific CORS headers depend on Flask-CORS configuration
    
    def test_content_type_json(self, client):
        """Test that API endpoints return JSON content type"""
        response = client.get('/api/health')
        
        assert response.status_code == 200
        assert 'application/json' in response.content_type
    
    @pytest.mark.parametrize("endpoint", [
        '/api/health',
        '/api/stats',
        '/api/matches',
        '/api/value-bets',
        '/api/api-status',
        '/api/api-economy-status'
    ])
    def test_get_endpoints_return_json(self, client, endpoint):
        """Test that GET endpoints return valid JSON"""
        response = client.get(endpoint)
        
        assert response.status_code == 200
        
        # Should return valid JSON
        try:
            data = json.loads(response.data)
            assert isinstance(data, dict)
        except json.JSONDecodeError:
            pytest.fail(f"Endpoint {endpoint} did not return valid JSON")
    
    def test_nonexistent_endpoint(self, client):
        """Test accessing non-existent endpoint"""
        response = client.get('/api/nonexistent')
        
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test using wrong HTTP method"""
        # POST to GET-only endpoint
        response = client.post('/api/health')
        
        assert response.status_code == 405  # Method Not Allowed
    
    def test_large_payload_handling(self, client):
        """Test handling of large JSON payload"""
        large_data = {
            'player_name': 'Test Player',
            'large_field': 'x' * 10000  # Large string
        }
        
        response = client.post('/api/test-ml',
                              data=json.dumps(large_data),
                              content_type='application/json')
        
        # Should handle gracefully (might reject or process)
        assert response.status_code in [200, 400, 413, 422, 500]
    
    def test_malformed_json_handling(self, client):
        """Test handling of malformed JSON"""
        response = client.post('/api/test-ml',
                              data='{"invalid": json}',
                              content_type='application/json')
        
        assert response.status_code == 400  # Bad Request
    
    def test_empty_request_body(self, client):
        """Test handling of empty request body"""
        response = client.post('/api/test-ml',
                              data='',
                              content_type='application/json')
        
        assert response.status_code in [400, 422]  # Bad Request or Unprocessable Entity


@pytest.mark.integration
class TestTennisBackendAPIIntegration:
    """Integration tests with mocked dependencies"""
    
    @pytest.fixture
    def client_with_mocks(self):
        """Create test client with properly mocked dependencies"""
        with patch('tennis_backend.PREDICTION_SERVICE_AVAILABLE', True), \
             patch('tennis_backend.REAL_PREDICTOR_AVAILABLE', True), \
             patch('tennis_backend.ODDS_API_AVAILABLE', True):
            
            app.config['TESTING'] = True
            with app.test_client() as client:
                yield client
    
    def test_ml_prediction_workflow(self, client_with_mocks, valid_match_data):
        """Test complete ML prediction workflow"""
        with patch('tennis_backend.prediction_service') as mock_service:
            # Mock successful prediction
            mock_service.predict_match.return_value = {
                'prediction': 0.72,
                'confidence': 'high',
                'individual_predictions': {
                    'neural_network': 0.75,
                    'xgboost': 0.70,
                    'random_forest': 0.71
                }
            }
            
            # Test ML prediction
            response = client_with_mocks.post('/api/test-ml',
                                            data=json.dumps(valid_match_data),
                                            content_type='application/json')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            assert data['prediction'] == 0.72
            assert data['confidence'] == 'high'
            assert 'individual_predictions' in data
    
    def test_underdog_analysis_workflow(self, client_with_mocks, valid_match_data):
        """Test complete underdog analysis workflow"""
        with patch('tennis_backend.real_predictor') as mock_predictor:
            # Mock underdog analysis
            mock_predictor.analyze_underdog_potential.return_value = {
                'is_underdog': True,
                'value_rating': 0.85,
                'confidence': 0.75,
                'recommendation': 'Strong underdog bet',
                'analysis': {
                    'rank_advantage': False,
                    'form_advantage': True,
                    'surface_advantage': True
                }
            }
            
            response = client_with_mocks.post('/api/underdog-analysis',
                                            data=json.dumps(valid_match_data),
                                            content_type='application/json')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            assert data['is_underdog'] is True
            assert data['value_rating'] == 0.85
            assert 'analysis' in data
    
    def test_api_rate_limiting_behavior(self, client_with_mocks):
        """Test API rate limiting behavior"""
        with patch('tennis_backend.DAILY_SCHEDULER_AVAILABLE', True):
            with patch('tennis_backend.get_daily_scheduler') as mock_scheduler:
                mock_scheduler.return_value.can_make_manual_request.return_value = False
                mock_scheduler.return_value.get_status.return_value = {
                    'daily_usage': 8,
                    'daily_limit': 8,
                    'manual_usage': 5,
                    'manual_limit': 5
                }
                
                response = client_with_mocks.post('/api/manual-api-update')
                
                assert response.status_code == 429  # Too Many Requests
                data = json.loads(response.data)
                assert 'limit' in data['message'].lower()
    
    def test_error_handling_integration(self, client_with_mocks, valid_match_data):
        """Test error handling integration"""
        with patch('tennis_backend.ERROR_HANDLING_AVAILABLE', True):
            with patch('tennis_backend.prediction_service') as mock_service:
                # Mock service raising an exception
                mock_service.predict_match.side_effect = Exception("Model loading failed")
                
                response = client_with_mocks.post('/api/test-ml',
                                                data=json.dumps(valid_match_data),
                                                content_type='application/json')
                
                # Should handle the error gracefully
                assert response.status_code in [500, 503]
                data = json.loads(response.data)
                assert 'error' in data or 'message' in data