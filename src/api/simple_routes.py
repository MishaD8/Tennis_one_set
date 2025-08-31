#!/usr/bin/env python3
"""
Simple Tennis API Routes
Basic routes for testing and core functionality
"""

import os
import sys
import logging
from datetime import datetime
from flask import Flask, jsonify, request

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

def register_simple_routes(app: Flask):
    """Register simplified routes for basic functionality"""
    
    # Import basic services
    try:
        from models.basic_prediction_service import tennis_prediction_service
        from data.api_tennis_collector import api_tennis_collector
        SERVICES_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"⚠️ Services not available: {e}")
        SERVICES_AVAILABLE = False
    
    @app.route('/', methods=['GET'])
    def home():
        """Home endpoint"""
        return {
            'service': 'Tennis One Set - Underdog Prediction System',
            'status': 'operational',
            'version': '1.0.0',
            'api_endpoints': {
                'health': '/api/health',
                'prediction': '/api/predict',
                'opportunities': '/api/opportunities',
                'status': '/api/status'
            },
            'timestamp': datetime.now().isoformat()
        }
    
    @app.route('/api/health', methods=['GET'])
    def health():
        """Health check endpoint"""
        try:
            # Check API key
            api_key_status = bool(os.getenv('API_TENNIS_KEY'))
            
            # Check database
            db_status = os.path.exists('data/tennis_predictions.db')
            
            # Check models
            models_status = os.path.exists('tennis_models/metadata.json')
            
            status = {
                'success': True,
                'status': 'healthy',
                'service': 'Tennis Underdog Prediction System',
                'version': '1.0.0',
                'components': {
                    'api_key_configured': api_key_status,
                    'database': 'available' if db_status else 'not_found',
                    'ml_models': 'available' if models_status else 'not_found',
                    'services': 'available' if SERVICES_AVAILABLE else 'not_available'
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Overall health status
            if api_key_status and db_status and models_status:
                status['overall_status'] = 'operational'
            else:
                status['overall_status'] = 'degraded'
                status['warnings'] = []
                
                if not api_key_status:
                    status['warnings'].append('API_TENNIS_KEY not configured')
                if not db_status:
                    status['warnings'].append('Database not found')
                if not models_status:
                    status['warnings'].append('ML models not found')
            
            return status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'success': False,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, 500
    
    @app.route('/api/predict', methods=['POST'])
    def predict():
        """Basic prediction endpoint"""
        try:
            if not SERVICES_AVAILABLE:
                return {
                    'success': False,
                    'error': 'Prediction services not available',
                    'timestamp': datetime.now().isoformat()
                }, 503
            
            data = request.get_json()
            if not data:
                return {
                    'success': False,
                    'error': 'JSON data required',
                    'timestamp': datetime.now().isoformat()
                }, 400
            
            # Get prediction from service
            analysis = tennis_prediction_service.analyze_second_set_prediction(data)
            
            return {
                'success': True,
                'prediction': analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, 500
    
    @app.route('/api/opportunities', methods=['GET'])
    def opportunities():
        """Get current underdog opportunities"""
        try:
            if not SERVICES_AVAILABLE:
                return {
                    'success': False,
                    'error': 'Services not available',
                    'timestamp': datetime.now().isoformat()
                }, 503
            
            # Get opportunities from service
            opportunities_list = tennis_prediction_service.find_underdog_opportunities()
            
            return {
                'success': True,
                'opportunities': opportunities_list,
                'count': len(opportunities_list),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Opportunities search failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, 500
    
    @app.route('/api/status', methods=['GET'])
    def status():
        """Get system status"""
        try:
            status_info = {
                'success': True,
                'system_status': 'operational',
                'timestamp': datetime.now().isoformat()
            }
            
            if SERVICES_AVAILABLE:
                # Get service status
                service_status = tennis_prediction_service.get_service_status()
                collector_status = api_tennis_collector.get_status()
                
                status_info.update({
                    'prediction_service': service_status,
                    'data_collector': collector_status
                })
            else:
                status_info['warning'] = 'Services not fully available'
            
            return status_info
            
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, 500
    
    logger.info("✅ Simple routes registered successfully")