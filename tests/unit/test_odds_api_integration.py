"""
Unit tests for TheOddsAPICorrect (Odds API Integration)
"""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
import requests
import sys
import os

# Import the module under test
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from correct_odds_api_integration import TheOddsAPICorrect


class TestTheOddsAPICorrect:
    """Test suite for TheOddsAPICorrect class"""
    
    @pytest.fixture
    def api_key(self):
        """Test API key"""
        return "test_api_key_12345"
    
    @pytest.fixture
    def odds_api(self, api_key):
        """Create TheOddsAPICorrect instance"""
        return TheOddsAPICorrect(api_key)
    
    @pytest.fixture
    def mock_response_success(self):
        """Mock successful API response"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {
            'x-requests-used': '5',
            'x-requests-remaining': '495'
        }
        mock_response.json.return_value = {
            'data': [
                {
                    'id': 'test_match_1',
                    'sport_title': 'Tennis',
                    'home_team': 'Novak Djokovic',
                    'away_team': 'Rafael Nadal'
                }
            ]
        }
        return mock_response
    
    @pytest.fixture
    def mock_response_auth_error(self):
        """Mock authentication error response"""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Invalid API key"
        mock_response.headers = {}
        return mock_response
    
    @pytest.fixture
    def mock_response_no_data(self):
        """Mock no data available response"""
        mock_response = Mock()
        mock_response.status_code = 422
        mock_response.text = "No data available"
        mock_response.headers = {}
        return mock_response
    
    def test_init_sets_properties(self, api_key):
        """Test initialization sets correct properties"""
        api = TheOddsAPICorrect(api_key)
        
        assert api.api_key == api_key
        assert api.base_url == "https://api.the-odds-api.com/v4"
        assert api.requests_used == 0
        assert api.requests_remaining is None
    
    @patch('correct_odds_api_integration.requests.get')
    def test_make_request_success(self, mock_get, odds_api, mock_response_success):
        """Test successful API request"""
        mock_get.return_value = mock_response_success
        
        result = odds_api._make_request("test_endpoint", {"param1": "value1"})
        
        # Check request was made correctly
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        
        assert "test_endpoint" in call_args[0][0]  # URL contains endpoint
        assert call_args[1]['params']['apiKey'] == odds_api.api_key
        assert call_args[1]['params']['param1'] == "value1"
        assert call_args[1]['timeout'] == 10
        
        # Check response handling
        assert result is not None
        assert odds_api.requests_used == '5'
        assert odds_api.requests_remaining == '495'
    
    @patch('correct_odds_api_integration.requests.get')
    def test_make_request_auth_error(self, mock_get, odds_api, mock_response_auth_error):
        """Test API request with authentication error"""
        mock_get.return_value = mock_response_auth_error
        
        result = odds_api._make_request("test_endpoint", {})
        
        assert result is None
        mock_get.assert_called_once()
    
    @patch('correct_odds_api_integration.requests.get')
    def test_make_request_no_data_error(self, mock_get, odds_api, mock_response_no_data):
        """Test API request with no data available"""
        mock_get.return_value = mock_response_no_data
        
        result = odds_api._make_request("test_endpoint", {})
        
        assert result is None
        mock_get.assert_called_once()
    
    @patch('correct_odds_api_integration.requests.get')
    def test_make_request_network_error(self, mock_get, odds_api):
        """Test API request with network error"""
        mock_get.side_effect = requests.RequestException("Network error")
        
        result = odds_api._make_request("test_endpoint", {})
        
        assert result is None
        mock_get.assert_called_once()
    
    @patch('correct_odds_api_integration.requests.get')
    def test_make_request_timeout(self, mock_get, odds_api):
        """Test API request timeout"""
        mock_get.side_effect = requests.Timeout("Request timed out")
        
        result = odds_api._make_request("test_endpoint", {})
        
        assert result is None
        mock_get.assert_called_once()
    
    @patch('correct_odds_api_integration.requests.get')
    def test_make_request_adds_api_key(self, mock_get, odds_api, mock_response_success):
        """Test that API key is added to request parameters"""
        mock_get.return_value = mock_response_success
        
        odds_api._make_request("test_endpoint", {"existing_param": "value"})
        
        call_args = mock_get.call_args
        params = call_args[1]['params']
        
        assert 'apiKey' in params
        assert params['apiKey'] == odds_api.api_key
        assert params['existing_param'] == "value"
    
    @patch.object(TheOddsAPICorrect, '_make_request')
    def test_get_available_sports_success(self, mock_request, odds_api):
        """Test getting available sports successfully"""
        mock_sports_data = [
            {'key': 'tennis', 'title': 'Tennis', 'active': True},
            {'key': 'tennis_atp', 'title': 'ATP Tennis', 'active': True},
            {'key': 'tennis_wta', 'title': 'WTA Tennis', 'active': True},
            {'key': 'basketball', 'title': 'Basketball', 'active': True}  # Non-tennis sport
        ]
        mock_request.return_value = mock_sports_data
        
        result = odds_api.get_available_sports()
        
        mock_request.assert_called_once_with("sports", {})
        
        # Should only return tennis sports
        assert len(result) == 3
        tennis_keys = [sport['key'] for sport in result]
        assert 'tennis' in tennis_keys
        assert 'tennis_atp' in tennis_keys
        assert 'tennis_wta' in tennis_keys
        assert 'basketball' not in tennis_keys
    
    @patch.object(TheOddsAPICorrect, '_make_request')
    def test_get_available_sports_no_data(self, mock_request, odds_api):
        """Test getting available sports when no data returned"""
        mock_request.return_value = None
        
        result = odds_api.get_available_sports()
        
        assert result == []
        mock_request.assert_called_once_with("sports", {})
    
    @patch.object(TheOddsAPICorrect, '_make_request')
    def test_get_available_sports_no_tennis(self, mock_request, odds_api):
        """Test getting available sports when no tennis sports available"""
        mock_sports_data = [
            {'key': 'basketball', 'title': 'Basketball', 'active': True},
            {'key': 'football', 'title': 'Football', 'active': True}
        ]
        mock_request.return_value = mock_sports_data
        
        result = odds_api.get_available_sports()
        
        assert result == []
    
    @patch.object(TheOddsAPICorrect, '_make_request')
    def test_get_tennis_odds_default_params(self, mock_request, odds_api):
        """Test getting tennis odds with default parameters"""
        mock_odds_data = [
            {
                'id': 'match_1',
                'sport_title': 'Tennis',
                'home_team': 'Player A',
                'away_team': 'Player B',
                'bookmakers': []
            }
        ]
        mock_request.return_value = mock_odds_data
        
        result = odds_api.get_tennis_odds()
        
        expected_params = {
            'regions': 'us,uk,eu,au',
            'markets': 'h2h',
            'oddsFormat': 'decimal',
            'dateFormat': 'iso'
        }
        mock_request.assert_called_once_with(f"sports/tennis/odds", expected_params)
        assert result == mock_odds_data
    
    @patch.object(TheOddsAPICorrect, '_make_request')
    def test_get_tennis_odds_custom_params(self, mock_request, odds_api):
        """Test getting tennis odds with custom parameters"""
        mock_request.return_value = []
        
        result = odds_api.get_tennis_odds(
            sport_key="tennis_atp",
            regions="us,uk"
        )
        
        expected_params = {
            'regions': 'us,uk',
            'markets': 'h2h',
            'oddsFormat': 'decimal',
            'dateFormat': 'iso'
        }
        mock_request.assert_called_once_with(f"sports/tennis_atp/odds", expected_params)
        assert result == []
    
    @patch.object(TheOddsAPICorrect, '_make_request')
    def test_get_tennis_odds_no_data(self, mock_request, odds_api):
        """Test getting tennis odds when no data available"""
        mock_request.return_value = None
        
        result = odds_api.get_tennis_odds()
        
        assert result == []
        mock_request.assert_called_once()
    
    def test_api_usage_tracking(self, odds_api):
        """Test that API usage is properly tracked"""
        # Initially should be 0 and None
        assert odds_api.requests_used == 0
        assert odds_api.requests_remaining is None
        
        # After making a request, should be updated
        with patch('correct_odds_api_integration.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {
                'x-requests-used': '10',
                'x-requests-remaining': '490'
            }
            mock_response.json.return_value = {}
            mock_get.return_value = mock_response
            
            odds_api._make_request("test", {})
            
            assert odds_api.requests_used == '10'
            assert odds_api.requests_remaining == '490'
    
    def test_api_usage_tracking_missing_headers(self, odds_api):
        """Test API usage tracking when headers are missing"""
        with patch('correct_odds_api_integration.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {}  # No usage headers
            mock_response.json.return_value = {}
            mock_get.return_value = mock_response
            
            odds_api._make_request("test", {})
            
            assert odds_api.requests_used == 'Unknown'
            assert odds_api.requests_remaining == 'Unknown'
    
    @pytest.mark.parametrize("status_code,expected_result", [
        (200, "success"),
        (401, None),
        (422, None),
        (500, None),
        (404, None)
    ])
    @patch('correct_odds_api_integration.requests.get')
    def test_make_request_status_codes(self, mock_get, odds_api, status_code, expected_result):
        """Test handling of various HTTP status codes"""
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.headers = {}
        mock_response.text = f"Error {status_code}"
        
        if status_code == 200:
            mock_response.json.return_value = {"status": "success"}
        
        mock_get.return_value = mock_response
        
        result = odds_api._make_request("test", {})
        
        if expected_result == "success":
            assert result == {"status": "success"}
        else:
            assert result is None
    
    def test_base_url_construction(self, odds_api):
        """Test that base URL is constructed correctly"""
        assert odds_api.base_url == "https://api.the-odds-api.com/v4"
        
        # Test that endpoint construction works
        with patch('correct_odds_api_integration.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_response.json.return_value = {}
            mock_get.return_value = mock_response
            
            odds_api._make_request("sports/tennis/odds", {})
            
            expected_url = "https://api.the-odds-api.com/v4/sports/tennis/odds"
            mock_get.assert_called_once()
            assert mock_get.call_args[0][0] == expected_url
    
    def test_request_timeout_configuration(self, odds_api):
        """Test that request timeout is properly configured"""
        with patch('correct_odds_api_integration.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_response.json.return_value = {}
            mock_get.return_value = mock_response
            
            odds_api._make_request("test", {})
            
            # Check that timeout is set to 10 seconds
            call_kwargs = mock_get.call_args[1]
            assert call_kwargs['timeout'] == 10


class TestTheOddsAPIIntegration:
    """Integration tests for TheOddsAPICorrect"""
    
    @pytest.fixture
    def odds_api_with_invalid_key(self):
        """Create API instance with invalid key for testing"""
        return TheOddsAPICorrect("invalid_key")
    
    @patch('correct_odds_api_integration.requests.get')
    def test_full_workflow_mock(self, mock_get, odds_api):
        """Test complete workflow with mocked responses"""
        # Mock sports response
        sports_response = Mock()
        sports_response.status_code = 200
        sports_response.headers = {'x-requests-used': '1', 'x-requests-remaining': '499'}
        sports_response.json.return_value = [
            {'key': 'tennis', 'title': 'Tennis', 'active': True}
        ]
        
        # Mock odds response
        odds_response = Mock()
        odds_response.status_code = 200
        odds_response.headers = {'x-requests-used': '2', 'x-requests-remaining': '498'}
        odds_response.json.return_value = [
            {
                'id': 'match_1',
                'sport_title': 'Tennis',
                'home_team': 'Player A',
                'away_team': 'Player B',
                'bookmakers': [
                    {
                        'key': 'test_bookmaker',
                        'markets': [
                            {
                                'key': 'h2h',
                                'outcomes': [
                                    {'name': 'Player A', 'price': 1.85},
                                    {'name': 'Player B', 'price': 1.95}
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
        
        mock_get.side_effect = [sports_response, odds_response]
        
        # Test workflow
        sports = odds_api.get_available_sports()
        assert len(sports) == 1
        assert sports[0]['key'] == 'tennis'
        
        odds = odds_api.get_tennis_odds()
        assert len(odds) == 1
        assert odds[0]['home_team'] == 'Player A'
        
        # Check API usage tracking
        assert odds_api.requests_used == '2'
        assert odds_api.requests_remaining == '498'
    
    def test_error_handling_resilience(self, odds_api):
        """Test that the API client handles various error conditions gracefully"""
        # Test with connection error
        with patch('correct_odds_api_integration.requests.get', side_effect=requests.ConnectionError):
            result = odds_api.get_available_sports()
            assert result == []
            
            result = odds_api.get_tennis_odds()
            assert result == []
        
        # Test with timeout
        with patch('correct_odds_api_integration.requests.get', side_effect=requests.Timeout):
            result = odds_api.get_available_sports()
            assert result == []
    
    @patch('correct_odds_api_integration.requests.get')
    def test_rate_limiting_awareness(self, mock_get, odds_api):
        """Test that the client is aware of rate limiting"""
        # Mock response indicating low remaining requests
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {
            'x-requests-used': '495',
            'x-requests-remaining': '5'
        }
        mock_response.json.return_value = []
        mock_get.return_value = mock_response
        
        odds_api.get_tennis_odds()
        
        # Client should track the low remaining count
        assert odds_api.requests_remaining == '5'
        assert odds_api.requests_used == '495'