#!/usr/bin/env python3
"""
Tennis One Set - Main Application Entry Point
Main entry point for the restructured tennis betting and ML prediction system
"""

import os
import sys

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.api.app import create_app, create_production_app

def main():
    """Main application entry point"""
    # Determine if running in production
    if os.getenv('FLASK_ENV') == 'production':
        app = create_production_app()
    else:
        app = create_app()
    
    print("🎾 TENNIS ONE SET - RESTRUCTURED VERSION")
    print("=" * 60)
    print(f"🌐 Dashboard: http://0.0.0.0:5001")
    print(f"📡 API: http://0.0.0.0:5001/api/*")
    print("=" * 60)
    
    try:
        # Enhanced server configuration for security
        ssl_context = None
        
        if os.getenv('FLASK_ENV') == 'production':
            # In production, SSL should be handled by reverse proxy (nginx)
            # But we can still configure SSL context if certificates are available
            from src.config.config import get_config
            config = get_config()
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
        import logging
        logging.error(f"Server startup failed: {e}", exc_info=True)

if __name__ == '__main__':
    main()