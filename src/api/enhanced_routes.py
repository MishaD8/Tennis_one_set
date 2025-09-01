#!/usr/bin/env python3
"""
Enhanced Tennis API Routes with Advanced Features
New endpoints for enhanced ML models and real-time prediction pipeline
"""

import os
import sys
import logging
import asyncio
from datetime import datetime
from flask import Flask, jsonify, request

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

def register_enhanced_routes(app: Flask):
    """Register enhanced routes with advanced features"""
    
    # Import advanced components
    try:
        from models.enhanced_feature_engineering import feature_engineer
        from models.advanced_ensemble import advanced_ensemble
        from models.realtime_prediction_pipeline import realtime_pipeline
        from models.performance_monitor import performance_monitor
        from data.live_data_feed import live_data_feed
        ADVANCED_FEATURES_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"⚠️ Advanced features not available: {e}")
        ADVANCED_FEATURES_AVAILABLE = False
    
    @app.route('/api/v2/enhanced-prediction', methods=['POST'])
    def enhanced_prediction():
        """Enhanced prediction with advanced feature engineering"""
        try:
            if not ADVANCED_FEATURES_AVAILABLE:
                return {
                    'success': False,
                    'error': 'Advanced features not available',
                    'timestamp': datetime.now().isoformat()
                }, 503
            
            data = request.get_json()
            if not data:
                return {
                    'success': False,
                    'error': 'JSON data required',
                    'timestamp': datetime.now().isoformat()
                }, 400
            
            # Validate required fields
            required_fields = ['player1', 'player2', 'player1_ranking', 'player2_ranking', 
                             'tournament', 'surface', 'first_set_score', 'first_set_winner']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                return {
                    'success': False,
                    'error': f'Missing required fields: {missing_fields}',
                    'timestamp': datetime.now().isoformat()
                }, 400
            
            # Generate enhanced features
            features = feature_engineer.create_enhanced_feature_vector(data)
            feature_explanation = feature_engineer.get_feature_importance_explanation(features)
            
            # Get ensemble prediction
            if not advanced_ensemble.is_trained:
                logger.info("🧠 Training ensemble for first use...")
                X_train, y_train = advanced_ensemble.generate_training_data(1000)
                advanced_ensemble.train_ensemble(X_train, y_train)
            
            ensemble_result = advanced_ensemble.predict_ensemble(features)
            
            # Determine underdog
            p1_rank = data['player1_ranking']
            p2_rank = data['player2_ranking']
            
            if p1_rank > p2_rank:
                underdog = data['player1']
                favorite = data['player2']
                underdog_rank = p1_rank
                favorite_rank = p2_rank
            else:
                underdog = data['player2']
                favorite = data['player1']
                underdog_rank = p2_rank
                favorite_rank = p1_rank
            
            # Enhanced analysis
            ranking_gap = abs(p1_rank - p2_rank)
            
            response = {
                'success': True,
                'prediction_type': 'enhanced_v2',
                'match_analysis': {
                    'underdog': underdog,
                    'favorite': favorite,
                    'underdog_ranking': underdog_rank,
                    'favorite_ranking': favorite_rank,
                    'ranking_gap': ranking_gap,
                    'tournament': data['tournament'],
                    'surface': data['surface'],
                    'first_set_result': {
                        'score': data['first_set_score'],
                        'winner': data['first_set_winner']
                    }
                },
                'enhanced_prediction': {
                    'underdog_second_set_probability': round(ensemble_result.get('probability', 0), 4),
                    'model_confidence': round(ensemble_result.get('confidence', 0), 4),
                    'individual_model_predictions': {
                        k: round(v, 4) for k, v in ensemble_result.get('individual_predictions', {}).items()
                    },
                    'ensemble_weights': ensemble_result.get('ensemble_weights', {}),
                    'model_count': ensemble_result.get('model_count', 0)
                },
                'advanced_features': {
                    'feature_count': len(features),
                    'feature_categories': feature_explanation,
                    'key_insights': _generate_key_insights(data, ensemble_result, ranking_gap)
                },
                'recommendation': _get_enhanced_recommendation(ensemble_result, ranking_gap),
                'timestamp': datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Enhanced prediction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, 500
    
    @app.route('/api/v2/live-monitoring', methods=['GET'])
    def live_monitoring_status():
        """Get live data monitoring status"""
        try:
            if not ADVANCED_FEATURES_AVAILABLE:
                return {
                    'success': False,
                    'error': 'Live monitoring not available',
                    'timestamp': datetime.now().isoformat()
                }, 503
            
            # Get live feed stats
            feed_stats = live_data_feed.get_monitoring_stats()
            
            # Get pipeline stats
            pipeline_stats = realtime_pipeline.get_pipeline_stats()
            
            # Get active matches
            active_matches = live_data_feed.get_active_matches()
            eligible_matches = [match for match in active_matches if match.is_eligible_for_prediction()]
            
            return {
                'success': True,
                'live_monitoring': {
                    'feed_status': feed_stats,
                    'pipeline_status': pipeline_stats,
                    'active_matches': len(active_matches),
                    'eligible_matches': len(eligible_matches),
                    'matches_with_completed_first_set': len([m for m in active_matches if m.is_first_set_complete()])
                },
                'recent_predictions': len(realtime_pipeline.prediction_history[-10:]),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Live monitoring status failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, 500
    
    @app.route('/api/v2/performance-metrics', methods=['GET'])
    def performance_metrics():
        """Get model performance metrics"""
        try:
            if not ADVANCED_FEATURES_AVAILABLE:
                return {
                    'success': False,
                    'error': 'Performance monitoring not available',
                    'timestamp': datetime.now().isoformat()
                }, 503
            
            time_period = request.args.get('period', '7_days')
            
            # Get performance metrics
            metrics = performance_monitor.calculate_performance_metrics(time_period)
            
            # Get drift analysis
            drift_analysis = performance_monitor.detect_model_drift([])
            
            return {
                'success': True,
                'performance_analysis': metrics,
                'drift_detection': drift_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Performance metrics failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, 500
    
    @app.route('/api/v2/record-result', methods=['POST'])
    def record_match_result():
        """Record actual match result for performance tracking"""
        try:
            if not ADVANCED_FEATURES_AVAILABLE:
                return {
                    'success': False,
                    'error': 'Performance tracking not available',
                    'timestamp': datetime.now().isoformat()
                }, 503
            
            data = request.get_json()
            if not data:
                return {
                    'success': False,
                    'error': 'JSON data required',
                    'timestamp': datetime.now().isoformat()
                }, 400
            
            # Validate required fields
            required_fields = ['prediction_data', 'actual_second_set_winner']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                return {
                    'success': False,
                    'error': f'Missing required fields: {missing_fields}',
                    'timestamp': datetime.now().isoformat()
                }, 400
            
            # Record the result
            success = performance_monitor.record_prediction_result(
                data['prediction_data'],
                data['actual_second_set_winner']
            )
            
            if success:
                return {
                    'success': True,
                    'message': 'Match result recorded successfully',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to record match result',
                    'timestamp': datetime.now().isoformat()
                }, 500
            
        except Exception as e:
            logger.error(f"❌ Record result failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, 500
    
    @app.route('/api/v2/feature-analysis', methods=['POST'])
    def feature_analysis():
        """Analyze feature importance for a specific match"""
        try:
            if not ADVANCED_FEATURES_AVAILABLE:
                return {
                    'success': False,
                    'error': 'Feature analysis not available',
                    'timestamp': datetime.now().isoformat()
                }, 503
            
            data = request.get_json()
            if not data:
                return {
                    'success': False,
                    'error': 'JSON data required',
                    'timestamp': datetime.now().isoformat()
                }, 400
            
            # Generate features
            features = feature_engineer.create_enhanced_feature_vector(data)
            
            # Get detailed feature analysis
            momentum_features = feature_engineer.calculate_first_set_momentum(data)
            fatigue_features = feature_engineer.calculate_fatigue_model(data)
            psychological_features = feature_engineer.calculate_psychological_factors(data)
            surface_features = feature_engineer.calculate_surface_adaptation(data)
            
            return {
                'success': True,
                'feature_analysis': {
                    'total_features': len(features),
                    'feature_vector': features.tolist(),
                    'feature_names': feature_engineer.feature_names,
                    'detailed_breakdown': {
                        'momentum_indicators': momentum_features,
                        'fatigue_modeling': fatigue_features,
                        'psychological_factors': psychological_features,
                        'surface_adaptation': surface_features
                    },
                    'category_summary': feature_engineer.get_feature_importance_explanation(features)
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Feature analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, 500
    
    @app.route('/api/v2/start-live-monitoring', methods=['POST'])
    def start_live_monitoring():
        """Start live data monitoring and prediction pipeline"""
        try:
            if not ADVANCED_FEATURES_AVAILABLE:
                return {
                    'success': False,
                    'error': 'Live monitoring not available',
                    'timestamp': datetime.now().isoformat()
                }, 503
            
            # This would start the async pipeline in a background task
            # For demonstration, we'll just return status
            
            return {
                'success': True,
                'message': 'Live monitoring start requested',
                'note': 'Live monitoring requires async setup - use dedicated script',
                'current_status': realtime_pipeline.is_running,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Start live monitoring failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, 500
    
    def _generate_key_insights(match_data: dict, ensemble_result: dict, ranking_gap: int) -> list:
        """Generate key insights from prediction"""
        insights = []
        
        probability = ensemble_result.get('probability', 0)
        confidence = ensemble_result.get('confidence', 0)
        
        # Ranking insight
        if ranking_gap > 50:
            insights.append(f"Large ranking gap ({ranking_gap}) creates significant underdog opportunity")
        elif ranking_gap < 20:
            insights.append(f"Close ranking gap ({ranking_gap}) suggests evenly matched players")
        
        # Probability insight
        if probability > 0.45:
            insights.append("High underdog probability - strong second set potential")
        elif probability > 0.35:
            insights.append("Moderate underdog probability - worth monitoring")
        else:
            insights.append("Low underdog probability - favorite likely to maintain dominance")
        
        # Confidence insight
        if confidence > 0.8:
            insights.append("High model confidence in prediction")
        elif confidence < 0.6:
            insights.append("Low model confidence - unpredictable match scenario")
        
        # First set insight
        first_set_score = match_data.get('first_set_score', '')
        if '-' in first_set_score:
            try:
                games = first_set_score.split('-')
                game_diff = abs(int(games[0]) - int(games[1]))
                if game_diff <= 2:
                    insights.append("Close first set suggests momentum can shift easily")
                elif game_diff >= 4:
                    insights.append("Dominant first set indicates strong form advantage")
            except:
                pass
        
        return insights[:4]  # Limit to 4 key insights
    
    def _get_enhanced_recommendation(ensemble_result: dict, ranking_gap: int) -> dict:
        """Get enhanced betting recommendation"""
        probability = ensemble_result.get('probability', 0)
        confidence = ensemble_result.get('confidence', 0)
        
        # Recommendation logic
        if probability >= 0.45 and confidence >= 0.8:
            action = "STRONG BET"
            reasoning = "High probability with high confidence"
        elif probability >= 0.38 and confidence >= 0.7:
            action = "CONSIDER"
            reasoning = "Good probability with decent confidence"
        elif probability >= 0.30 and confidence >= 0.65:
            action = "MONITOR"
            reasoning = "Moderate signals - watch for additional indicators"
        else:
            action = "PASS"
            reasoning = "Insufficient probability or confidence"
        
        # Risk assessment
        if ranking_gap > 100:
            risk = "HIGH"
        elif ranking_gap > 50:
            risk = "MEDIUM"
        else:
            risk = "LOW"
        
        return {
            'action': action,
            'reasoning': reasoning,
            'risk_level': risk,
            'suggested_stake': 'small' if action in ['STRONG BET', 'CONSIDER'] else 'none',
            'confidence_threshold_met': confidence >= 0.7
        }
    
    logger.info("✅ Enhanced routes registered successfully")

if __name__ == "__main__":
    from flask import Flask
    app = Flask(__name__)
    register_enhanced_routes(app)
    print("✅ Enhanced routes module ready")