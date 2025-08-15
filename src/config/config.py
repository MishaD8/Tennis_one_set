#!/usr/bin/env python3
"""
Tennis Backend Configuration Management
Centralized configuration for tennis betting and ML prediction system
"""

import os
import secrets
from typing import Dict, Any, Optional

class Config:
    """Base configuration class for tennis backend"""
    
    # Flask Configuration
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(32))
    WTF_CSRF_TIME_LIMIT = None
    
    # Security Configuration
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # CORS Configuration
    ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:5001').split(',')
    
    # Rate Limiting Configuration
    RATELIMIT_STORAGE_URL = os.getenv('REDIS_URL', '').strip() or None
    RATELIMIT_DEFAULT = ["100 per day", "20 per hour", "5 per minute"]
    RATELIMIT_STRATEGY = "fixed-window"
    RATELIMIT_HEADERS_ENABLED = True
    RATELIMIT_SWALLOW_ERRORS = True
    RATELIMIT_IN_MEMORY_FALLBACK_ENABLED = True
    
    # Trusted proxy configuration for rate limiting security
    TRUSTED_PROXIES = os.getenv('TRUSTED_PROXIES', '').split(',')
    
    # API Configuration
    TENNIS_API_KEY = os.getenv('TENNIS_API_KEY', '')
    RAPIDAPI_KEY = os.getenv('RAPIDAPI_KEY', '')
    API_TENNIS_KEY = os.getenv('API_TENNIS_KEY', '')  # API-Tennis.com key
    BETFAIR_APP_KEY = os.getenv('BETFAIR_APP_KEY', '')
    BETFAIR_USERNAME = os.getenv('BETFAIR_USERNAME', '')
    BETFAIR_PASSWORD = os.getenv('BETFAIR_PASSWORD', '')
    
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///tennis_data.db')
    
    # ML Model Configuration
    ML_MODEL_PATH = os.getenv('ML_MODEL_PATH', './tennis_models/')
    PREDICTION_CONFIDENCE_THRESHOLD = float(os.getenv('PREDICTION_CONFIDENCE_THRESHOLD', '0.6'))
    
    # Tennis Betting Configuration
    DEFAULT_STAKE = float(os.getenv('DEFAULT_STAKE', '10.0'))
    MAX_STAKE = float(os.getenv('MAX_STAKE', '100.0'))
    RISK_MANAGEMENT_ENABLED = os.getenv('RISK_MANAGEMENT_ENABLED', 'true').lower() == 'true'
    
    # API Rate Limits
    DAILY_API_LIMIT = int(os.getenv('DAILY_API_LIMIT', '8'))
    MONTHLY_API_LIMIT = int(os.getenv('MONTHLY_API_LIMIT', '500'))
    
    @staticmethod
    def get_redis_url() -> Optional[str]:
        """Get Redis URL with validation"""
        redis_url = os.getenv('REDIS_URL', '').strip()
        
        if redis_url and redis_url != 'memory://':
            try:
                import redis
                # Test Redis connection
                if redis_url.startswith('redis://'):
                    r = redis.Redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2)
                else:
                    r = redis.Redis(host='localhost', port=6379, socket_connect_timeout=2, socket_timeout=2)
                
                r.ping()
                return redis_url
            except Exception:
                pass
        
        return None

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    PREFERRED_URL_SCHEME = 'http'
    SESSION_COOKIE_SECURE = False
    FORCE_HTTPS = False

class ProductionConfig(Config):
    """Production configuration with enhanced security"""
    DEBUG = False
    PREFERRED_URL_SCHEME = 'https'
    SESSION_COOKIE_SECURE = True
    FORCE_HTTPS = True
    
    # SSL Configuration
    SSL_CERT_PATH = os.getenv('SSL_CERT_PATH', '')
    SSL_KEY_PATH = os.getenv('SSL_KEY_PATH', '')
    
    # Enhanced security headers
    SECURITY_HEADERS = {
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Content-Security-Policy': "default-src 'self'"
    }

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    WTF_CSRF_ENABLED = False
    DATABASE_URL = 'sqlite:///:memory:'

def get_config() -> Config:
    """Get configuration based on environment"""
    env = os.getenv('FLASK_ENV', 'development').lower()
    
    if env == 'production':
        return ProductionConfig()
    elif env == 'testing':
        return TestingConfig()
    else:
        return DevelopmentConfig()

def validate_config() -> Dict[str, Any]:
    """Validate critical configuration settings"""
    config = get_config()
    validation_results = {
        'valid': True,
        'warnings': [],
        'errors': []
    }
    
    # Check critical API keys
    if not config.TENNIS_API_KEY:
        validation_results['warnings'].append('TENNIS_API_KEY not configured')
    
    if not config.RAPIDAPI_KEY:
        validation_results['warnings'].append('RAPIDAPI_KEY not configured')
    
    if not config.API_TENNIS_KEY:
        validation_results['warnings'].append('API_TENNIS_KEY not configured - API-Tennis.com integration disabled')
    
    # Check Betfair configuration for betting functionality
    if not config.BETFAIR_APP_KEY:
        validation_results['warnings'].append('BETFAIR_APP_KEY not configured - betting functionality limited')
    
    # Check ML model path
    if not os.path.exists(config.ML_MODEL_PATH):
        validation_results['errors'].append(f'ML model path does not exist: {config.ML_MODEL_PATH}')
        validation_results['valid'] = False
    
    # Check Redis configuration
    redis_url = config.get_redis_url()
    if not redis_url:
        validation_results['warnings'].append('Redis not available - using in-memory rate limiting')
    
    return validation_results

# Configuration constants for tennis betting
TENNIS_SURFACES = ['Hard', 'Clay', 'Grass', 'Carpet']
TOURNAMENT_CATEGORIES = ['ATP Masters 1000', 'ATP 500', 'ATP 250', 'WTA 1000', 'WTA 500', 'WTA 250']
BET_TYPES = ['Match Winner', 'Set Betting', 'Handicap', 'Total Games', 'Over/Under']

# Tennis ranking configuration
RANKING_RANGE = {
    'MIN_RANK': 10,
    'MAX_RANK': 300,
    'TIER_SIZE': 290,  # 300 - 10
    'MILESTONES': {
        'top_10_threshold': 10,
        'top_30_threshold': 30,
        'top_50_threshold': 50,
        'top_100_threshold': 100,
        'relegation_risk': 280,
        'mid_tier': 155,  # (300 + 10) / 2
        'upper_tier': 50,
        'promotion_zone': 75
    }
}

# ML Model configurations
ML_MODELS = {
    'random_forest': {'file': 'random_forest.pkl', 'type': 'sklearn'},
    'xgboost': {'file': 'xgboost.pkl', 'type': 'xgboost'},
    'neural_network': {'file': 'neural_network.h5', 'type': 'tensorflow'},
    'logistic_regression': {'file': 'logistic_regression.pkl', 'type': 'sklearn'},
    'gradient_boosting': {'file': 'gradient_boosting.pkl', 'type': 'sklearn'}
}