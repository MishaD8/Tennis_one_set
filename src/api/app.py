#!/usr/bin/env python3
"""
Tennis Backend Flask Application Factory
Main application initialization for tennis betting and ML prediction system
"""

import os
import logging
from flask import Flask

from src.config.config import get_config, validate_config
from src.api.middleware import (
    init_security_middleware, 
    init_cors, 
    init_rate_limiting,
    TennisMiddleware
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app() -> Flask:
    """
    Application factory pattern for creating Flask app with proper configuration
    """
    # Get the project root directory (two levels up from src/api/)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    template_dir = os.path.join(project_root, 'templates')
    static_dir = os.path.join(project_root, 'static')
    
    app = Flask(__name__, 
                template_folder=template_dir,
                static_folder=static_dir)
    
    # Load configuration
    config = get_config()
    app.config.from_object(config)
    
    # Validate configuration
    validation = validate_config()
    if not validation['valid']:
        logger.error("Critical configuration errors found:")
        for error in validation['errors']:
            logger.error(f"  - {error}")
        raise RuntimeError("Application cannot start due to configuration errors")
    
    if validation['warnings']:
        logger.warning("Configuration warnings:")
        for warning in validation['warnings']:
            logger.warning(f"  - {warning}")
    
    # Initialize security middleware
    app = init_security_middleware(app)
    
    # Initialize CORS
    init_cors(app)
    
    # Initialize rate limiting
    limiter = init_rate_limiting(app)
    
    # Initialize tennis-specific middleware
    tennis_middleware = TennisMiddleware(app)
    
    # Store limiter in app context for use in routes
    app.limiter = limiter
    
    # Import and register routes here to avoid circular imports
    routes_registered = False
    
    try:
        from src.api.simple_routes import register_simple_routes
        register_simple_routes(app)
        logger.info("✅ Simple routes registered successfully")
        routes_registered = True
    except ImportError as e:
        logger.warning(f"⚠️ Simple routes not available: {e}")
    
    # Register enhanced routes with advanced features
    try:
        from src.api.enhanced_routes import register_enhanced_routes
        register_enhanced_routes(app)
        logger.info("✅ Enhanced routes registered successfully")
        routes_registered = True
    except ImportError as e:
        logger.warning(f"⚠️ Enhanced routes not available: {e}")
    
    # Fallback minimal health endpoint only if no routes were registered
    if not routes_registered:
        @app.route('/api/health', methods=['GET'])
        def health():
            return {
                'success': True,
                'status': 'healthy',
                'message': 'Tennis backend is running (minimal mode)',
                'version': '1.0.0'
            }
    
    logger.info("✅ Flask application initialized successfully")
    logger.info(f"✅ Configuration: {config.__class__.__name__}")
    logger.info(f"✅ Environment: {os.getenv('FLASK_ENV', 'development')}")
    
    return app

def create_production_app() -> Flask:
    """
    Create production-ready app with enhanced security and monitoring
    """
    os.environ['FLASK_ENV'] = 'production'
    app = create_app()
    
    # Additional production configurations
    if not app.debug:
        # Add production-specific error handling
        @app.errorhandler(Exception)
        def handle_exception(e):
            # Log the error
            logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
            
            # Return generic error to prevent information disclosure
            return {
                'success': False,
                'error': 'An internal error occurred'
            }, 500
    
    return app

# Create the application instance
app = create_app()

if __name__ == '__main__':
    print("🎾 TENNIS BACKEND - MODULAR VERSION")
    print("=" * 60)
    print(f"🌐 Dashboard: http://0.0.0.0:5001")
    print(f"📡 API: http://0.0.0.0:5001/api/*")
    print("=" * 60)
    
    try:
        # Enhanced server configuration for security
        ssl_context = None
        config = get_config()
        
        if os.getenv('FLASK_ENV') == 'production':
            # In production, SSL should be handled by reverse proxy (nginx)
            # But we can still configure SSL context if certificates are available
            cert_file = getattr(config, 'SSL_CERT_PATH', None)
            key_file = getattr(config, 'SSL_KEY_PATH', None)
            
            if cert_file and key_file and os.path.exists(cert_file) and os.path.exists(key_file):
                ssl_context = (cert_file, key_file)
                print(f"✅ SSL certificates found, enabling HTTPS")
            else:
                print("⚠️ Production mode: SSL should be handled by reverse proxy (nginx)")
        
        app.run(
            host='0.0.0.0',
            port=5001,
            debug=False,
            threaded=True,
            ssl_context=ssl_context,
            # Additional security configurations
            use_reloader=False,  # Disable reloader in production
            use_debugger=False   # Ensure debugger is disabled
        )
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        # Log the full error for debugging
        logger.error(f"Server startup failed: {e}", exc_info=True)