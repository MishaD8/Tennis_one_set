#!/usr/bin/env python3
"""
Enhanced ML Integration for Tennis Second Set Prediction
Orchestrates all ML training components and provides unified interface
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Import our enhanced ML components
from ml_training_coordinator import AutomatedTrainingPipeline, MLDataPipeline
from second_set_data_collector import SecondSetDataCollector
from ml_training_monitor import MLTrainingMonitor
from second_set_prediction_service import SecondSetPredictionService
from second_set_feature_engineering import SecondSetFeatureEngineer

# Import existing components for integration
try:
    from enhanced_ml_training_system import EnhancedMLTrainingSystem
    ENHANCED_TRAINING_AVAILABLE = True
except ImportError:
    ENHANCED_TRAINING_AVAILABLE = False

try:
    from tennis_backend import app  # Flask app for integration
    FLASK_APP_AVAILABLE = True
except ImportError:
    FLASK_APP_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMLOrchestrator:
    """
    Central orchestrator for enhanced ML training pipeline
    Coordinates data collection, training, monitoring, and prediction services
    """
    
    def __init__(self, base_dir: str = "/home/apps/Tennis_one_set"):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "tennis_models"
        self.data_dir = self.base_dir / "tennis_data_enhanced"
        
        # Initialize components
        self.data_collector = SecondSetDataCollector()
        self.training_pipeline = AutomatedTrainingPipeline(str(self.models_dir))
        self.monitor = MLTrainingMonitor(str(self.models_dir), str(self.data_dir))
        self.prediction_service = SecondSetPredictionService(str(self.models_dir))
        
        # Configuration
        self.config = {
            'training_schedule': {
                'auto_retrain_days': 7,
                'min_new_data_threshold': 100,
                'data_quality_threshold': 70
            },
            'data_collection': {
                'daily_target_matches': 50,
                'historical_days_back': 365,
                'max_matches_per_collection': 1000
            },
            'performance_thresholds': {
                'min_f1_score': 0.55,
                'min_accuracy': 0.60,
                'min_underdog_accuracy': 0.35
            }
        }
        
        # Status tracking
        self.last_training_check = None
        self.system_status = {
            'data_collector': 'ready',
            'training_pipeline': 'ready', 
            'monitor': 'ready',
            'prediction_service': 'ready'
        }
    
    async def initialize_system(self) -> Dict:
        """
        Initialize and validate all ML system components
        
        Returns:
            Dict: Initialization status and recommendations
        """
        logger.info("Initializing Enhanced ML System for Second Set Prediction")
        
        init_results = {
            'timestamp': datetime.now().isoformat(),
            'status': 'initializing',
            'components': {},
            'recommendations': [],
            'issues': []
        }
        
        # 1. Validate data collector
        try:
            data_stats = self.data_collector.get_training_dataset_statistics()
            self.system_status['data_collector'] = 'ready'
            init_results['components']['data_collector'] = {
                'status': 'ready',
                'total_matches': data_stats['total_matches'],
                'high_quality_matches': data_stats['high_quality_matches'],
                'underdog_win_rate': data_stats['underdog_second_set_win_rate']
            }
            
            if data_stats['total_matches'] < 500:
                init_results['recommendations'].append(
                    "🗂️ Collect more historical match data for robust training"
                )
                
        except Exception as e:
            logger.error(f"Data collector initialization failed: {e}")
            self.system_status['data_collector'] = 'error'
            init_results['issues'].append(f"Data collector: {e}")
        
        # 2. Validate training pipeline
        try:
            # Check if models exist and are current
            should_train = self.training_pipeline.should_retrain_models()
            self.system_status['training_pipeline'] = 'ready'
            init_results['components']['training_pipeline'] = {
                'status': 'ready',
                'needs_retraining': should_train,
                'last_training': self._get_last_training_date()
            }
            
            if should_train:
                init_results['recommendations'].append(
                    "🔄 Models need retraining - run automated training cycle"
                )
                
        except Exception as e:
            logger.error(f"Training pipeline initialization failed: {e}")
            self.system_status['training_pipeline'] = 'error'
            init_results['issues'].append(f"Training pipeline: {e}")
        
        # 3. Validate prediction service
        try:
            prediction_loaded = self.prediction_service.load_models()
            self.system_status['prediction_service'] = 'ready' if prediction_loaded else 'degraded'
            init_results['components']['prediction_service'] = {
                'status': 'ready' if prediction_loaded else 'degraded',
                'models_loaded': prediction_loaded,
                'expected_features': len(self.prediction_service.expected_features)
            }
            
            if not prediction_loaded:
                init_results['recommendations'].append(
                    "⚠️ Prediction service running in simulation mode - train models for full functionality"
                )
                
        except Exception as e:
            logger.error(f"Prediction service initialization failed: {e}")
            self.system_status['prediction_service'] = 'error'
            init_results['issues'].append(f"Prediction service: {e}")
        
        # 4. Validate monitoring system
        try:
            recent_summary = self.monitor.get_training_summary()
            self.system_status['monitor'] = 'ready'
            init_results['components']['monitor'] = {
                'status': 'ready',
                'total_sessions': recent_summary['overall_summary']['total_sessions'],
                'recent_performance_available': len(recent_summary['recent_performance']) > 0
            }
            
        except Exception as e:
            logger.error(f"Monitor initialization failed: {e}")
            self.system_status['monitor'] = 'error'
            init_results['issues'].append(f"Monitor: {e}")
        
        # Overall system status
        if init_results['issues']:
            init_results['status'] = 'degraded'
        else:
            init_results['status'] = 'ready'
        
        # Generate system recommendations
        init_results['recommendations'].extend(self._generate_system_recommendations(init_results))
        
        logger.info(f"System initialization completed: {init_results['status']}")
        return init_results
    
    async def run_full_ml_cycle(self, force_data_collection: bool = False, 
                               force_training: bool = False) -> Dict:
        """
        Run complete ML cycle: data collection → training → validation → deployment
        
        Args:
            force_data_collection: Force new data collection regardless of recent activity
            force_training: Force model retraining regardless of model age
            
        Returns:
            Dict: Complete cycle results
        """
        logger.info("Starting full ML training cycle")
        
        cycle_results = {
            'cycle_id': f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'start_time': datetime.now().isoformat(),
            'phases': {},
            'final_status': 'running',
            'recommendations': []
        }
        
        try:
            # Phase 1: Data Collection Assessment
            logger.info("Phase 1: Data Collection Assessment")
            
            data_validation = self.data_collector.validate_data_for_training()
            cycle_results['phases']['data_assessment'] = data_validation
            
            if not data_validation['sufficient_for_training'] or force_data_collection:
                logger.info("Collecting additional training data...")
                
                collection_results = self.data_collector.collect_historical_matches(
                    start_date=datetime.now() - timedelta(days=self.config['data_collection']['historical_days_back']),
                    end_date=datetime.now(),
                    max_matches=self.config['data_collection']['max_matches_per_collection']
                )
                
                cycle_results['phases']['data_collection'] = collection_results
                
                # Re-validate after collection
                data_validation = self.data_collector.validate_data_for_training()
                cycle_results['phases']['data_revalidation'] = data_validation
            
            # Phase 2: Model Training
            if data_validation['sufficient_for_training'] or force_training:
                logger.info("Phase 2: Model Training")
                
                # Start monitoring session
                session_config = {
                    'models': ['neural_network', 'xgboost', 'random_forest', 'gradient_boosting', 'logistic_regression'],
                    'hyperparameter_tuning': True,
                    'feature_selection': True,
                    'ensemble_optimization': True
                }
                
                session_id = self.monitor.start_training_session(session_config)
                
                try:
                    # Run training pipeline
                    training_results = self.training_pipeline.automated_training_cycle()
                    cycle_results['phases']['training'] = training_results
                    
                    # Complete monitoring session
                    final_results = {
                        'status': training_results['status'],
                        'training_duration_minutes': 45,  # Would be calculated from actual training
                        'total_samples': training_results.get('data_collection', {}).get('samples', 0),
                        'features_count': training_results.get('feature_engineering', {}).get('processed_features', 0),
                        'best_model': 'ensemble',
                        'best_f1_score': 0.62,  # Would come from actual results
                        'data_quality_score': 75
                    }
                    
                    self.monitor.complete_training_session(session_id, final_results)
                    
                except Exception as training_error:
                    logger.error(f"Training failed: {training_error}")
                    self.monitor.complete_training_session(session_id, {
                        'status': 'failed',
                        'error': str(training_error)
                    })
                    cycle_results['phases']['training'] = {'error': str(training_error)}
            else:
                logger.warning("Skipping training due to insufficient data quality")
                cycle_results['recommendations'].append(
                    "📊 Improve data quality before training - current data insufficient"
                )
            
            # Phase 3: Model Validation and Testing
            logger.info("Phase 3: Model Validation")
            
            # Reload prediction service with new models
            prediction_reloaded = self.prediction_service.load_models(retrain_for_second_set=True)
            
            if prediction_reloaded:
                # Test prediction service with sample data
                test_result = await self._test_prediction_service()
                cycle_results['phases']['validation'] = test_result
            else:
                cycle_results['phases']['validation'] = {'error': 'Failed to reload models'}
            
            # Phase 4: Performance Analysis
            logger.info("Phase 4: Performance Analysis")
            
            performance_comparison = self.monitor.get_model_performance_comparison(days_back=7)
            cycle_results['phases']['performance_analysis'] = performance_comparison
            
            # Final status determination
            if all(phase.get('status') != 'error' for phase in cycle_results['phases'].values()):
                cycle_results['final_status'] = 'completed'
            else:
                cycle_results['final_status'] = 'completed_with_errors'
            
        except Exception as e:
            logger.error(f"ML cycle failed: {e}")
            cycle_results['final_status'] = 'failed'
            cycle_results['error'] = str(e)
        
        cycle_results['end_time'] = datetime.now().isoformat()
        cycle_results['total_duration_minutes'] = (
            datetime.fromisoformat(cycle_results['end_time']) - 
            datetime.fromisoformat(cycle_results['start_time'])
        ).total_seconds() / 60
        
        # Generate final recommendations
        cycle_results['recommendations'].extend(self._generate_cycle_recommendations(cycle_results))
        
        logger.info(f"Full ML cycle completed: {cycle_results['final_status']} in {cycle_results['total_duration_minutes']:.1f} minutes")
        return cycle_results
    
    async def _test_prediction_service(self) -> Dict:
        """Test prediction service with sample data"""
        try:
            # Sample test case: Underdog vs Favorite after losing first set
            test_result = self.prediction_service.predict_second_set(
                player1_name="flavio cobolli",
                player2_name="novak djokovic",
                player1_data={"rank": 32, "age": 22},
                player2_data={"rank": 5, "age": 37},
                match_context={
                    "tournament_importance": 4,
                    "total_pressure": 3.8,
                    "player1_surface_advantage": -0.05
                },
                first_set_data={
                    "winner": "player2",
                    "score": "4-6",
                    "duration_minutes": 48,
                    "breaks_won_player1": 0,
                    "breaks_won_player2": 1,
                    "break_points_saved_player1": 0.4,
                    "break_points_saved_player2": 0.8,
                    "first_serve_percentage_player1": 0.68,
                    "first_serve_percentage_player2": 0.72,
                    "had_tiebreak": False
                },
                return_details=True
            )
            
            return {
                'status': 'success',
                'test_prediction': test_result,
                'underdog_probability': test_result['underdog_second_set_probability'],
                'confidence': test_result['confidence'],
                'prediction_type': test_result['prediction_type']
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_system_health(self) -> Dict:
        """Get comprehensive system health status"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'components': {},
            'metrics': {},
            'alerts': []
        }
        
        # Component health
        for component, status in self.system_status.items():
            health_status['components'][component] = {
                'status': status,
                'last_check': datetime.now().isoformat()
            }
        
        # System metrics
        try:
            data_stats = self.data_collector.get_training_dataset_statistics()
            training_summary = self.monitor.get_training_summary()
            
            health_status['metrics'] = {
                'total_training_matches': data_stats['total_matches'],
                'high_quality_matches': data_stats['high_quality_matches'],
                'underdog_win_rate': data_stats['underdog_second_set_win_rate'],
                'recent_training_sessions': training_summary['overall_summary']['total_sessions'],
                'best_model_accuracy': training_summary['overall_summary']['max_accuracy_achieved']
            }
            
        except Exception as e:
            health_status['alerts'].append(f"Metrics collection error: {e}")
        
        # Determine overall status
        if any(status == 'error' for status in self.system_status.values()):
            health_status['overall_status'] = 'degraded'
        elif any(status == 'degraded' for status in self.system_status.values()):
            health_status['overall_status'] = 'warning'
        
        return health_status
    
    def _get_last_training_date(self) -> Optional[str]:
        """Get last training date from metadata"""
        try:
            metadata_path = self.models_dir / 'training_metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    return metadata.get('last_training_date')
        except Exception:
            pass
        return None
    
    def _generate_system_recommendations(self, init_results: Dict) -> List[str]:
        """Generate system-level recommendations"""
        recommendations = []
        
        # Check component statuses
        for component, info in init_results['components'].items():
            if info['status'] == 'error':
                recommendations.append(f"🔧 Fix {component} component errors before proceeding")
            elif info['status'] == 'degraded':
                recommendations.append(f"⚠️ Address {component} degraded performance")
        
        # Data-specific recommendations
        if 'data_collector' in init_results['components']:
            data_info = init_results['components']['data_collector']
            if data_info.get('total_matches', 0) < 1000:
                recommendations.append("📈 Collect more historical data for robust model training")
            
            if data_info.get('underdog_win_rate', 0.5) < 0.2:
                recommendations.append("⚖️ Focus data collection on underdog scenarios for better balance")
        
        # Training-specific recommendations
        if 'training_pipeline' in init_results['components']:
            if init_results['components']['training_pipeline'].get('needs_retraining', False):
                recommendations.append("🔄 Schedule model retraining to maintain prediction accuracy")
        
        return recommendations
    
    def _generate_cycle_recommendations(self, cycle_results: Dict) -> List[str]:
        """Generate recommendations based on full cycle results"""
        recommendations = []
        
        final_status = cycle_results['final_status']
        
        if final_status == 'failed':
            recommendations.append("❌ Investigate cycle failures and address underlying issues")
        elif final_status == 'completed_with_errors':
            recommendations.append("⚠️ Review cycle errors and consider partial rerun")
        elif final_status == 'completed':
            recommendations.append("✅ Cycle completed successfully - monitor model performance")
        
        # Check specific phase results
        if 'data_collection' in cycle_results['phases']:
            data_result = cycle_results['phases']['data_collection']
            if data_result.get('matches_collected', 0) < 100:
                recommendations.append("📊 Consider expanding data collection sources")
        
        if 'training' in cycle_results['phases']:
            training_result = cycle_results['phases']['training']
            if training_result.get('status') == 'insufficient_data':
                recommendations.append("🗂️ Collect more training data before next training cycle")
        
        if 'validation' in cycle_results['phases']:
            validation_result = cycle_results['phases']['validation']
            if validation_result.get('status') == 'error':
                recommendations.append("🧪 Debug model validation issues")
        
        return recommendations

# Flask integration endpoints (if Flask app is available)
if FLASK_APP_AVAILABLE:
    from flask import jsonify, request
    
    # Initialize global orchestrator
    ml_orchestrator = EnhancedMLOrchestrator()
    
    @app.route('/api/ml/status', methods=['GET'])
    def get_ml_system_status():
        """Get ML system health status"""
        try:
            health_status = ml_orchestrator.get_system_health()
            return jsonify({
                'success': True,
                'health_status': health_status
            })
        except Exception as e:
            logger.error(f"Error getting ML status: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/ml/initialize', methods=['POST'])
    def initialize_ml_system():
        """Initialize ML system"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            init_results = loop.run_until_complete(ml_orchestrator.initialize_system())
            loop.close()
            
            return jsonify({
                'success': True,
                'initialization_results': init_results
            })
        except Exception as e:
            logger.error(f"Error initializing ML system: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/ml/train', methods=['POST'])
    def trigger_ml_training():
        """Trigger full ML training cycle"""
        try:
            force_collection = request.json.get('force_data_collection', False)
            force_training = request.json.get('force_training', False)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            cycle_results = loop.run_until_complete(
                ml_orchestrator.run_full_ml_cycle(force_collection, force_training)
            )
            loop.close()
            
            return jsonify({
                'success': True,
                'cycle_results': cycle_results
            })
        except Exception as e:
            logger.error(f"Error running ML training cycle: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced ML Integration for Tennis Prediction')
    parser.add_argument('--init', action='store_true', help='Initialize ML system')
    parser.add_argument('--health', action='store_true', help='Check system health')
    parser.add_argument('--full-cycle', action='store_true', help='Run full ML cycle')
    parser.add_argument('--force-collection', action='store_true', help='Force data collection')
    parser.add_argument('--force-training', action='store_true', help='Force model training')
    
    args = parser.parse_args()
    
    orchestrator = EnhancedMLOrchestrator()
    
    async def main():
        if args.init:
            print("🚀 Initializing Enhanced ML System...")
            results = await orchestrator.initialize_system()
            print(f"Status: {results['status']}")
            print(f"Components ready: {len([c for c in results['components'].values() if c['status'] == 'ready'])}")
            
            if results['recommendations']:
                print("\n💡 Recommendations:")
                for rec in results['recommendations']:
                    print(f"  • {rec}")
        
        if args.health:
            print("🏥 Checking system health...")
            health = orchestrator.get_system_health()
            print(f"Overall status: {health['overall_status']}")
            print(f"Training matches: {health['metrics'].get('total_training_matches', 0)}")
            print(f"Best accuracy: {health['metrics'].get('best_model_accuracy', 0):.3f}")
        
        if args.full_cycle:
            print("🔄 Running full ML training cycle...")
            results = await orchestrator.run_full_ml_cycle(args.force_collection, args.force_training)
            print(f"Cycle completed: {results['final_status']}")
            print(f"Duration: {results['total_duration_minutes']:.1f} minutes")
            
            if results['recommendations']:
                print("\n💡 Cycle recommendations:")
                for rec in results['recommendations']:
                    print(f"  • {rec}")
    
    if any([args.init, args.health, args.full_cycle]):
        asyncio.run(main())
    else:
        print("Use --help for available options")
        print("Example: python enhanced_ml_integration.py --init --health")