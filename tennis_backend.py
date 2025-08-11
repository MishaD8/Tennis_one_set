#!/usr/bin/env python3
"""
Tennis Backend - Modular Version
Main entry point for tennis betting and ML prediction system
Refactored for better maintainability and separation of concerns
"""

import os
import logging
from flask import Flask

# Import our new modular components
try:
    from app import create_app, create_production_app
    from config import get_config, validate_config
    MODULAR_COMPONENTS_AVAILABLE = True
    print("✅ Modular components loaded successfully")
except ImportError as e:
    print(f"❌ Critical error: Modular components not available: {e}")
    print("❌ This suggests the refactoring is incomplete or there are import issues")
    MODULAR_COMPONENTS_AVAILABLE = False

# Import legacy error handling for backward compatibility
try:
    from error_handler import get_error_handler
    from config_loader import load_secure_config
    ERROR_HANDLING_AVAILABLE = True
    error_handler = get_error_handler()
    print("✅ Legacy error handling loaded")
except ImportError as e:
    print(f"⚠️ Legacy error handling not available: {e}")
    ERROR_HANDLING_AVAILABLE = False
    error_handler = None

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_tennis_app():
    """Create tennis application with proper error handling"""
    
    if not MODULAR_COMPONENTS_AVAILABLE:
        # Fallback to create a minimal Flask app if modular components aren't available
        logger.error("Creating minimal fallback app due to missing modular components")
        
        app = Flask(__name__)
        app.config['SECRET_KEY'] = 'fallback-key-not-for-production'
        
        @app.route('/api/health')
        def health():
            return {
                'status': 'degraded',
                'message': 'Tennis backend running in fallback mode',
                'error': 'Modular components not available',
                'version': '5.0-fallback'
            }
        
        @app.errorhandler(404)
        def not_found(error):
            return {
                'success': False,
                'error': 'Endpoint not found',
                'message': 'Tennis backend is running in fallback mode'
            }, 404
        
        return app
    
    # Create the proper modular application
    try:
        if os.getenv('FLASK_ENV') == 'production':
            app = create_production_app()
        else:
            app = create_app()
        
        logger.info("✅ Tennis backend application created successfully")
        return app
        
    except Exception as e:
        logger.error(f"❌ Failed to create tennis application: {e}")
        raise

# Create the Flask application instance
app = create_tennis_app()

# Legacy service initialization for backward compatibility
real_predictor = None
prediction_service = None
odds_integrator = None
enhanced_collector = None
universal_collector = None
daily_scheduler = None

def load_config():
    """Legacy configuration loading for backward compatibility"""
    try:
        if ERROR_HANDLING_AVAILABLE:
            config = load_secure_config()
            logger.info("✅ Legacy configuration loaded")
            return config
        else:
            logger.info("⚠️ Using modular configuration system")
            return get_config()
    except Exception as e:
        logger.warning(f"Configuration loading error: {e}")
        return None

def initialize_services():
    """Legacy service initialization for backward compatibility"""
    global real_predictor, prediction_service, odds_integrator
    global enhanced_collector, universal_collector, daily_scheduler
    
    try:
        # Import tennis system components with fallback handling
        try:
            from real_tennis_predictor_integration import RealTennisPredictor
            real_predictor = RealTennisPredictor()
            logger.info("✅ Real ML predictor initialized")
        except ImportError as e:
            logger.info(f"⚠️ Real predictor not available: {e}")

        try:
            from tennis_prediction_module import TennisPredictionService
            prediction_service = TennisPredictionService()
            logger.info("✅ Prediction service initialized")
        except ImportError as e:
            logger.info(f"⚠️ Prediction service not available: {e}")

        try:
            odds_integrator = TennisOddsIntegrator()
            logger.info("✅ Odds API integration loaded")
        except ImportError as e:
            logger.info(f"⚠️ Odds API integration not available: {e}")

        try:
            from enhanced_universal_collector import EnhancedUniversalCollector
            enhanced_collector = EnhancedUniversalCollector()
            logger.info("✅ Enhanced Universal Collector loaded")
        except ImportError as e:
            logger.info(f"⚠️ Enhanced collector not available: {e}")
            try:
                from universal_tennis_data_collector import UniversalTennisDataCollector
                universal_collector = UniversalTennisDataCollector()
                logger.info("✅ Universal data collector loaded (fallback)")
            except ImportError as e2:
                logger.info(f"⚠️ Universal collector not available: {e2}")

        try:
            from daily_api_scheduler import init_daily_scheduler, start_daily_scheduler
            daily_scheduler = init_daily_scheduler()
            start_daily_scheduler()
            logger.info("✅ Daily API scheduler loaded")
        except ImportError as e:
            logger.info(f"⚠️ Daily scheduler not available: {e}")
            
        logger.info("✅ Legacy services initialization completed")
        
    except Exception as e:
        logger.error(f"❌ Service initialization error: {e}")

if __name__ == '__main__':
    print("🎾 TENNIS BACKEND - MODULAR VERSION 5.0")
    print("=" * 60)
    print("🔧 ARCHITECTURE IMPROVEMENTS:")
    print("• ✅ Modular structure with separated concerns")
    print("• ✅ Centralized configuration management")
    print("• ✅ Enhanced security middleware")
    print("• ✅ Improved error handling and validation")
    print("• ✅ Better separation of routes and business logic")
    print("• ✅ Standardized English comments")
    print("=" * 60)
    print(f"🌐 Dashboard: http://0.0.0.0:5001")
    print(f"📡 API: http://0.0.0.0:5001/api/*")
    print("=" * 60)
    
    # Legacy configuration and service initialization
    config = load_config()
    initialize_services()
    
    # Component status
    print("🔍 COMPONENT STATUS:")
    print(f"🤖 Real Predictor: {'✅ Active' if real_predictor else '⚠️ Not available'}")
    print(f"🎯 Prediction Service: {'✅ Active' if prediction_service else '⚠️ Not available'}")
    print(f"💰 Odds Integrator: {'✅ Active' if odds_integrator else '⚠️ Not available'}")
    print(f"🌍 Enhanced Collector: {'✅ Active' if enhanced_collector else '⚠️ Not available'}")
    print(f"📊 Universal Collector: {'✅ Active' if universal_collector else '⚠️ Not available'}")
    print(f"⏰ Daily Scheduler: {'✅ Active' if daily_scheduler else '⚠️ Not available'}")
    print("=" * 60)
    
    try:
        # Get configuration for SSL setup
        config_obj = get_config() if MODULAR_COMPONENTS_AVAILABLE else None
        ssl_context = None
        
        if os.getenv('FLASK_ENV') == 'production':
            # In production, SSL should be handled by reverse proxy (nginx)
            # But we can still configure SSL context if certificates are available
            if config_obj:
                cert_file = getattr(config_obj, 'SSL_CERT_PATH', None)
                key_file = getattr(config_obj, 'SSL_KEY_PATH', None)
                
                if cert_file and key_file and os.path.exists(cert_file) and os.path.exists(key_file):
                    ssl_context = (cert_file, key_file)
                    print(f"✅ SSL certificates found, enabling HTTPS")
                else:
                    print("⚠️ Production mode: SSL should be handled by reverse proxy (nginx)")
            else:
                print("⚠️ Production mode: Configuration not available for SSL setup")
        
        print("🚀 Starting Tennis Backend Server...")
        
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
        
        # If modular components are not available, provide troubleshooting info
        if not MODULAR_COMPONENTS_AVAILABLE:
            print("\n🔧 TROUBLESHOOTING:")
            print("The modular components (config.py, app.py, middleware.py, routes.py) are required.")
            print("Please ensure all new modules are properly created and accessible.")
            print("If you're running this for the first time after refactoring, verify:")
            print("1. config.py exists and is valid")
            print("2. middleware.py exists and is valid") 
            print("3. app.py exists and is valid")
            print("4. routes.py exists and is valid")
            print("5. All required dependencies are installed")