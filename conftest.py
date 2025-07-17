"""
Pytest configuration and shared fixtures for Tennis Prediction System
"""
import pytest
import tempfile
import shutil
import os
import sys
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory"""
    return os.path.dirname(os.path.abspath(__file__))

@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session")
def sample_tennis_data():
    """Create sample tennis match data for testing"""
    return pd.DataFrame({
        'player1': ['Novak Djokovic', 'Rafael Nadal', 'Roger Federer'],
        'player2': ['Rafael Nadal', 'Roger Federer', 'Novak Djokovic'],
        'player1_rank': [1, 2, 3],
        'player2_rank': [2, 3, 1],
        'surface': ['Clay', 'Hard', 'Grass'],
        'tournament': ['French Open', 'US Open', 'Wimbledon'],
        'result': [1, 0, 1]  # 1 = player1 wins, 0 = player2 wins
    })

@pytest.fixture(scope="session")
def sample_features():
    """Create sample feature matrix for ML testing"""
    np.random.seed(42)  # For reproducible tests
    n_samples = 100
    n_features = 23  # Based on existing feature engineering
    
    return pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

@pytest.fixture(scope="session")
def sample_labels():
    """Create sample labels for ML testing"""
    np.random.seed(42)
    return np.random.randint(0, 2, size=100)

@pytest.fixture
def mock_api_response():
    """Mock API response for odds data"""
    return {
        'success': True,
        'data': [
            {
                'id': 'test_match_1',
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
    }

@pytest.fixture
def mock_database():
    """Mock database connection and operations"""
    mock_db = Mock()
    mock_db.connect.return_value = True
    mock_db.execute.return_value = True
    mock_db.fetch.return_value = []
    mock_db.close.return_value = True
    return mock_db

@pytest.fixture
def mock_ml_models():
    """Mock ML models for testing predictions"""
    models = {}
    for model_name in ['neural_network', 'xgboost', 'random_forest', 'gradient_boosting', 'logistic_regression']:
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.6])  # Prediction probability
        mock_model.predict_proba.return_value = np.array([[0.4, 0.6]])
        models[model_name] = mock_model
    return models

@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return {
        'odds_api': {
            'api_key': 'test_api_key',
            'base_url': 'https://api.the-odds-api.com/v4',
            'rate_limit': 500
        },
        'database': {
            'host': 'localhost',
            'port': 5432,
            'name': 'test_tennis_db',
            'user': 'test_user',
            'password': 'test_password'
        },
        'cache': {
            'redis_url': 'redis://localhost:6379/1',
            'ttl': 3600
        },
        'models': {
            'model_dir': './tennis_models',
            'ensemble_weights': {
                'neural_network': 0.25,
                'xgboost': 0.20,
                'random_forest': 0.20,
                'gradient_boosting': 0.20,
                'logistic_regression': 0.15
            }
        }
    }

@pytest.fixture
def mock_player_rankings():
    """Mock player rankings data"""
    return {
        'atp_rankings': [
            {'rank': 1, 'player': 'Novak Djokovic', 'points': 12000},
            {'rank': 2, 'player': 'Rafael Nadal', 'points': 11000},
            {'rank': 3, 'player': 'Roger Federer', 'points': 10000}
        ],
        'wta_rankings': [
            {'rank': 1, 'player': 'Serena Williams', 'points': 9000},
            {'rank': 2, 'player': 'Naomi Osaka', 'points': 8500},
            {'rank': 3, 'player': 'Simona Halep', 'points': 8000}
        ]
    }

@pytest.fixture
def mock_tournament_data():
    """Mock tournament calendar data"""
    return {
        'tournaments': [
            {
                'name': 'Australian Open',
                'surface': 'Hard',
                'category': 'Grand Slam',
                'start_date': '2024-01-15',
                'end_date': '2024-01-28'
            },
            {
                'name': 'French Open',
                'surface': 'Clay',
                'category': 'Grand Slam',
                'start_date': '2024-05-26',
                'end_date': '2024-06-09'
            }
        ]
    }

@pytest.fixture
def mock_cache():
    """Mock cache manager for testing"""
    cache = Mock()
    cache.get.return_value = None
    cache.set.return_value = True
    cache.delete.return_value = True
    cache.clear.return_value = True
    cache.exists.return_value = False
    return cache

# Environment setup fixtures
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment before running tests"""
    # Set test environment variables
    os.environ['TENNIS_ENV'] = 'test'
    os.environ['ODDS_API_KEY'] = 'test_api_key'
    os.environ['DATABASE_URL'] = 'postgresql://test:test@localhost:5432/test_tennis'
    os.environ['REDIS_URL'] = 'redis://localhost:6379/1'
    
    yield
    
    # Cleanup after tests
    test_vars = ['TENNIS_ENV', 'ODDS_API_KEY', 'DATABASE_URL', 'REDIS_URL']
    for var in test_vars:
        os.environ.pop(var, None)

# Custom pytest hooks
def pytest_configure(config):
    """Configure pytest with custom markers and settings"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "api: mark test as an API test"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file paths"""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "api" in str(item.fspath):
            item.add_marker(pytest.mark.api)
        
        # Add slow marker for tests that might take longer
        if any(keyword in item.name.lower() for keyword in ['api', 'database', 'ml_model']):
            item.add_marker(pytest.mark.slow)