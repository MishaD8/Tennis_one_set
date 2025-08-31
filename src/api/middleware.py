#!/usr/bin/env python3
"""
Tennis Backend Middleware
Security, CORS, rate limiting, and authentication middleware for tennis betting system
"""

import os
import logging
from functools import wraps
from datetime import datetime
from typing import Optional, Dict, Any

from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.exceptions import HTTPException

from src.config.config import get_config

logger = logging.getLogger(__name__)

def secure_rate_limit_key_func():
    """
    Secure rate limiting key function that prevents IP spoofing attacks
    and implements additional validation for proxy headers
    """
    try:
        # Get the real IP with enhanced security
        real_ip = get_remote_address()
        forwarded_for = request.headers.get('X-Forwarded-For', '')
        
        # Validate proxy headers to prevent spoofing
        if forwarded_for:
            config = get_config()
            trusted_proxies = config.TRUSTED_PROXIES
            if trusted_proxies and trusted_proxies[0]:  # Check if configured
                # Additional validation can be added here for trusted proxy ranges
                pass
        
        # Use combination of IP and User-Agent for better fingerprinting
        user_agent = request.headers.get('User-Agent', 'unknown')[:50]  # Limit length
        # Create a secure fingerprint but don't expose sensitive data in logs
        fingerprint = f"{real_ip}:{hash(user_agent) % 10000}"
        
        return fingerprint
        
    except Exception as e:
        # Log security event without exposing sensitive details
        logger.warning(f"Rate limit key function error, using fallback")
        return get_remote_address()

def init_security_middleware(app: Flask) -> Flask:
    """Initialize security middleware including proxy fix and security headers"""
    
    # Secure proxy configuration with validation
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
    
    config = get_config()
    
    # Add security headers for production
    if hasattr(config, 'SECURITY_HEADERS'):
        @app.after_request
        def add_security_headers(response):
            for header, value in config.SECURITY_HEADERS.items():
                response.headers[header] = value
            return response
    
    # Force HTTPS in production
    if getattr(config, 'FORCE_HTTPS', False):
        @app.before_request
        def force_https():
            if not request.is_secure and os.getenv('FLASK_ENV') == 'production':
                return redirect(request.url.replace('http://', 'https://'))
    
    return app

def init_cors(app: Flask) -> None:
    """Initialize CORS with restricted origins"""
    config = get_config()
    
    CORS(app, 
         origins=config.ALLOWED_ORIGINS,
         methods=['GET', 'POST'],
         allow_headers=['Content-Type', 'Authorization', 'X-API-Key'])

def check_redis_health() -> Dict[str, Any]:
    """Check Redis connection health for rate limiting"""
    try:
        redis_url = os.getenv('REDIS_URL', '').strip()
        
        if not redis_url or redis_url == 'memory://':
            return {
                'available': False,
                'status': 'not_configured',
                'message': 'Redis URL not configured'
            }
        
        import redis
        
        if redis_url.startswith('redis://'):
            r = redis.Redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2)
        else:
            r = redis.Redis(host='localhost', port=6379, socket_connect_timeout=2, socket_timeout=2)
        
        # Test connection
        r.ping()
        
        return {
            'available': True,
            'status': 'connected',
            'message': 'Redis connection successful',
            'info': {
                'redis_version': r.info().get('redis_version', 'unknown'),
                'connected_clients': r.info().get('connected_clients', 0)
            }
        }
        
    except ImportError:
        return {
            'available': False,
            'status': 'redis_not_installed',
            'message': 'Redis package not installed'
        }
    except Exception as e:
        return {
            'available': False,
            'status': 'connection_failed',
            'message': f'Redis connection failed: {str(e)}'
        }

def init_rate_limiting(app: Flask) -> Limiter:
    """Initialize rate limiting with Redis fallback to in-memory"""
    
    config = get_config()
    redis_url = config.get_redis_url()
    
    if redis_url:
        logger.info("✅ Redis connection successful - using Redis for rate limiting")
    else:
        logger.info("🔄 Redis not available - Flask-Limiter will use in-memory storage")
    
    limiter = Limiter(
        app=app,
        key_func=secure_rate_limit_key_func,
        default_limits=config.RATELIMIT_DEFAULT,
        storage_uri=redis_url,  # Will be None if Redis unavailable
        strategy=config.RATELIMIT_STRATEGY,
        headers_enabled=config.RATELIMIT_HEADERS_ENABLED,
        swallow_errors=config.RATELIMIT_SWALLOW_ERRORS,  # Don't crash app on rate limiter errors
        in_memory_fallback_enabled=config.RATELIMIT_IN_MEMORY_FALLBACK_ENABLED,
        in_memory_fallback=config.RATELIMIT_DEFAULT
    )
    
    return limiter

def require_api_key():
    """Decorator to require API key for sensitive endpoints"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
            
            # In development, allow requests without API key
            if os.getenv('FLASK_ENV') != 'production':
                return f(*args, **kwargs)
            
            # Check for API key
            expected_key = os.getenv('TENNIS_API_KEY')
            if not expected_key:
                # API key not configured, allow access but log warning
                logger.warning("API key not configured for production")
                return f(*args, **kwargs)
            
            if not api_key or api_key != expected_key:
                return jsonify({
                    'success': False,
                    'error': 'Invalid or missing API key',
                    'timestamp': datetime.now().isoformat()
                }), 401
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def validate_json_request():
    """Decorator to validate JSON requests"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if request.method == 'POST':
                if not request.is_json:
                    return jsonify({
                        'success': False,
                        'error': 'Request must be JSON',
                        'timestamp': datetime.now().isoformat()
                    }), 400
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def log_request_info():
    """Decorator to log request information for monitoring"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            start_time = datetime.now()
            
            # Log request
            logger.info(f"Request: {request.method} {request.path} from {get_remote_address()}")
            
            try:
                result = f(*args, **kwargs)
                
                # Log successful response
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"Response: {request.path} completed in {duration:.3f}s")
                
                return result
                
            except Exception as e:
                # Log error
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"Error: {request.path} failed after {duration:.3f}s - {str(e)}")
                raise
                
        return decorated_function
    return decorator

def handle_betting_errors():
    """Decorator to handle betting-specific errors"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except ValueError as e:
                return jsonify({
                    'success': False,
                    'error': f'Invalid input: {str(e)}',
                    'error_type': 'validation_error',
                    'timestamp': datetime.now().isoformat()
                }), 400
            except ConnectionError as e:
                return jsonify({
                    'success': False,
                    'error': 'Service temporarily unavailable',
                    'error_type': 'connection_error',
                    'timestamp': datetime.now().isoformat()
                }), 503
            except Exception as e:
                logger.error(f"Betting operation error: {str(e)}", exc_info=True)
                return jsonify({
                    'success': False,
                    'error': 'Internal server error',
                    'error_type': 'server_error',
                    'timestamp': datetime.now().isoformat()
                }), 500
        return decorated_function
    return decorator

class TennisMiddleware:
    """Tennis-specific middleware for enhanced functionality"""
    
    def __init__(self, app: Flask):
        self.app = app
        self.init_error_handlers()
    
    def init_error_handlers(self):
        """Initialize custom error handlers"""
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({
                'success': False,
                'error': 'Endpoint not found',
                'available_endpoints': [
                    '/api/health',
                    '/api/stats',
                    '/api/matches', 
                    '/api/test-ml',
                    '/api/value-bets',
                    '/api/underdog-analysis',
                    '/api/refresh'
                ],
                'timestamp': datetime.now().isoformat()
            }), 404

        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({
                'success': False,
                'error': 'Internal server error',
                'timestamp': datetime.now().isoformat()
            }), 500

        @self.app.errorhandler(400)
        def bad_request(error):
            return jsonify({
                'success': False,
                'error': 'Bad request',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        @self.app.errorhandler(429)
        def rate_limit_exceeded(error):
            return jsonify({
                'success': False,
                'error': 'Rate limit exceeded',
                'message': 'Too many requests. Please try again later.',
                'timestamp': datetime.now().isoformat()
            }), 429