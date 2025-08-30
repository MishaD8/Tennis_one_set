#!/usr/bin/env python3
"""
Tennis Betting System Deployment Script
Complete deployment and validation of the automated tennis betting system
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('deployment.log')
    ]
)
logger = logging.getLogger(__name__)

class TennisBettingSystemDeployer:
    """Comprehensive deployment manager for tennis betting system"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.deployment_results = {
            'timestamp': datetime.now().isoformat(),
            'deployment_status': 'starting',
            'components_validated': {},
            'system_ready': False,
            'warnings': [],
            'errors': [],
            'next_steps': []
        }
    
    async def deploy_system(self, simulation_mode: bool = True) -> Dict[str, Any]:
        """Deploy complete tennis betting system"""
        logger.info("🚀 Starting Tennis Betting System Deployment")
        
        try:
            # 1. Validate environment and dependencies
            await self._validate_environment()
            
            # 2. Initialize core components
            await self._initialize_components()
            
            # 3. Test system integration
            await self._test_integration(simulation_mode)
            
            # 4. Validate ML models
            await self._validate_ml_models()
            
            # 5. Test prediction pipeline
            await self._test_prediction_pipeline()
            
            # 6. Test risk management
            await self._test_risk_management()
            
            # 7. Final system validation
            await self._final_system_validation()
            
            # 8. Generate deployment report
            self._generate_deployment_report()
            
            self.deployment_results['deployment_status'] = 'completed'
            self.deployment_results['system_ready'] = len(self.deployment_results['errors']) == 0
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            self.deployment_results['deployment_status'] = 'failed'
            self.deployment_results['errors'].append(str(e))
        
        return self.deployment_results
    
    async def _validate_environment(self):
        """Validate environment and dependencies"""
        logger.info("📋 Validating Environment")
        
        validation_results = {
            'python_version': sys.version,
            'required_directories': [],
            'required_files': [],
            'dependencies': [],
            'missing_items': []
        }
        
        # Check Python version
        if sys.version_info < (3, 8):
            self.deployment_results['errors'].append("Python 3.8+ required")
        
        # Check required directories
        required_dirs = [
            'src/api',
            'src/config', 
            'src/models',
            'tennis_models',
            'cache'
        ]
        
        for dir_path in required_dirs:
            full_path = os.path.join(self.base_dir, dir_path)
            if os.path.exists(full_path):
                validation_results['required_directories'].append(f"✅ {dir_path}")
            else:
                validation_results['required_directories'].append(f"❌ {dir_path}")
                validation_results['missing_items'].append(dir_path)
        
        # Check key files
        required_files = [
            'src/api/comprehensive_tennis_betting_integration.py',
            'src/api/betfair_api_client.py',
            'src/api/automated_betting_engine.py',
            'src/api/risk_management_system.py',
            'automated_tennis_prediction_service.py',
            'enhanced_api_tennis_integration.py'
        ]
        
        for file_path in required_files:
            full_path = os.path.join(self.base_dir, file_path)
            if os.path.exists(full_path):
                validation_results['required_files'].append(f"✅ {file_path}")
            else:
                validation_results['required_files'].append(f"❌ {file_path}")
                validation_results['missing_items'].append(file_path)
        
        # Check dependencies
        try:
            import requests
            validation_results['dependencies'].append("✅ requests")
        except ImportError:
            validation_results['dependencies'].append("❌ requests")
            validation_results['missing_items'].append("requests")
        
        try:
            import numpy
            validation_results['dependencies'].append("✅ numpy")
        except ImportError:
            validation_results['dependencies'].append("❌ numpy")
            validation_results['missing_items'].append("numpy")
        
        try:
            import pandas
            validation_results['dependencies'].append("✅ pandas")
        except ImportError:
            validation_results['dependencies'].append("❌ pandas")
            validation_results['missing_items'].append("pandas")
        
        try:
            import scikit_learn
            validation_results['dependencies'].append("✅ scikit-learn")
        except ImportError:
            validation_results['dependencies'].append("❌ scikit-learn")
            validation_results['missing_items'].append("scikit-learn")
        
        self.deployment_results['components_validated']['environment'] = validation_results
        
        if validation_results['missing_items']:
            self.deployment_results['warnings'].append(
                f"Missing items: {', '.join(validation_results['missing_items'])}"
            )
    
    async def _initialize_components(self):
        """Initialize core system components"""
        logger.info("🔧 Initializing Components")
        
        # Add src to Python path
        sys.path.insert(0, os.path.join(self.base_dir, 'src'))
        sys.path.insert(0, self.base_dir)
        
        component_results = {}
        
        # Test automated tennis prediction service
        try:
            from automated_tennis_prediction_service import AutomatedTennisPredictionService
            service = AutomatedTennisPredictionService()
            component_results['tennis_prediction_service'] = {
                'status': 'ready',
                'models_loaded': len(getattr(service, 'models', {})),
                'rankings_loaded': len(getattr(service, 'enhanced_rankings', {}))
            }
        except Exception as e:
            component_results['tennis_prediction_service'] = {'status': 'error', 'error': str(e)}
            self.deployment_results['warnings'].append(f"Tennis prediction service: {e}")
        
        # Test enhanced API integration
        try:
            from enhanced_api_tennis_integration import EnhancedAPITennisIntegration
            api_client = EnhancedAPITennisIntegration()
            component_results['enhanced_api_integration'] = {
                'status': 'ready',
                'client_available': api_client.client is not None
            }
        except Exception as e:
            component_results['enhanced_api_integration'] = {'status': 'error', 'error': str(e)}
            self.deployment_results['warnings'].append(f"Enhanced API integration: {e}")
        
        # Test Betfair API client
        try:
            from src.api.betfair_api_client import BetfairAPIClient
            betfair_client = BetfairAPIClient()  # Will run in simulation mode
            component_results['betfair_api_client'] = {
                'status': 'ready',
                'simulation_mode': True
            }
        except Exception as e:
            component_results['betfair_api_client'] = {'status': 'error', 'error': str(e)}
            self.deployment_results['warnings'].append(f"Betfair API client: {e}")
        
        # Test risk management
        try:
            from src.api.risk_management_system import create_risk_manager, RiskLevel
            risk_manager = create_risk_manager(RiskLevel.MODERATE, 10000.0)
            component_results['risk_management'] = {
                'status': 'ready',
                'risk_level': 'moderate',
                'initial_bankroll': 10000.0
            }
        except Exception as e:
            component_results['risk_management'] = {'status': 'error', 'error': str(e)}
            self.deployment_results['warnings'].append(f"Risk management: {e}")
        
        # Test betting engine
        try:
            from src.api.automated_betting_engine import AutomatedBettingEngine, RiskManagementConfig
            risk_config = RiskManagementConfig.moderate()
            betting_engine = AutomatedBettingEngine(risk_config, 10000.0)
            component_results['betting_engine'] = {
                'status': 'ready',
                'risk_config': 'moderate'
            }
        except Exception as e:
            component_results['betting_engine'] = {'status': 'error', 'error': str(e)}
            self.deployment_results['warnings'].append(f"Betting engine: {e}")
        
        self.deployment_results['components_validated']['core_components'] = component_results
    
    async def _test_integration(self, simulation_mode: bool):
        """Test comprehensive system integration"""
        logger.info("🔄 Testing System Integration")
        
        integration_results = {
            'initialization': 'failed',
            'component_health': {},
            'system_start': 'failed',
            'worker_threads': 0,
            'errors': []
        }
        
        try:
            from src.api.comprehensive_tennis_betting_integration import ComprehensiveTennisBettingIntegration
            
            # Initialize system
            system = ComprehensiveTennisBettingIntegration()
            if simulation_mode:
                system.config['simulation_mode'] = True
            
            # Test initialization
            init_results = await system.initialize_system()
            integration_results['initialization'] = init_results['status']
            integration_results['component_health'] = system.component_health
            
            if init_results['ready_for_operation']:
                # Test system start
                start_result = system.start_system()
                if start_result['success']:
                    integration_results['system_start'] = 'success'
                    integration_results['worker_threads'] = start_result['active_threads']
                    
                    # Let it run briefly
                    await asyncio.sleep(2)
                    
                    # Test system stop
                    stop_result = system.stop_system()
                    if stop_result['success']:
                        integration_results['system_stop'] = 'success'
                    else:
                        integration_results['errors'].append("System stop failed")
                else:
                    integration_results['errors'].append(start_result.get('error', 'Unknown start error'))
            else:
                integration_results['errors'].append("System not ready for operation")
                
        except Exception as e:
            integration_results['errors'].append(str(e))
            self.deployment_results['errors'].append(f"Integration test: {e}")
        
        self.deployment_results['components_validated']['integration_test'] = integration_results
    
    async def _validate_ml_models(self):
        """Validate ML models are loaded and functional"""
        logger.info("🤖 Validating ML Models")
        
        ml_results = {
            'models_directory_exists': False,
            'models_found': [],
            'models_loaded': [],
            'prediction_test': 'failed',
            'errors': []
        }
        
        try:
            models_dir = os.path.join(self.base_dir, 'tennis_models')
            ml_results['models_directory_exists'] = os.path.exists(models_dir)
            
            if os.path.exists(models_dir):
                model_files = os.listdir(models_dir)
                ml_results['models_found'] = model_files
                
                # Test model loading
                from src.models.enhanced_ml_integration import EnhancedMLPredictor
                predictor = EnhancedMLPredictor()
                ml_results['models_loaded'] = list(predictor.models.keys())
                
                # Test prediction
                if predictor.models:
                    # Create sample features
                    import numpy as np
                    sample_features = np.array([
                        30, 15, 15, 15, 30,  # ranking features
                        2.0, 0.5,  # points features
                        0.0, 1.0, 1.0,  # movement features
                        2.0,  # tournament importance
                        1.0, 0.0, 0.0,  # surface features
                        0.0, 0.0,  # surface advantages
                        1.0, 0.0, 1.0, 0.0,  # tour features
                        0.75  # data quality
                    ], dtype=np.float32).reshape(1, -1)
                    
                    prediction = predictor.predict_with_ensemble(sample_features)
                    if prediction.get('success'):
                        ml_results['prediction_test'] = 'success'
                        ml_results['sample_prediction'] = prediction['probability']
                    else:
                        ml_results['errors'].append("Prediction test failed")
                else:
                    ml_results['errors'].append("No models loaded")
            
        except Exception as e:
            ml_results['errors'].append(str(e))
            self.deployment_results['warnings'].append(f"ML models: {e}")
        
        self.deployment_results['components_validated']['ml_models'] = ml_results
    
    async def _test_prediction_pipeline(self):
        """Test prediction generation pipeline"""
        logger.info("🔮 Testing Prediction Pipeline")
        
        pipeline_results = {
            'prediction_generation': 'failed',
            'predictions_count': 0,
            'confidence_levels': [],
            'errors': []
        }
        
        try:
            # Test via comprehensive integration
            from src.api.comprehensive_tennis_betting_integration import ComprehensiveTennisBettingIntegration, TennisMatch
            from datetime import datetime, timedelta
            
            system = ComprehensiveTennisBettingIntegration()
            await system.initialize_system()
            
            # Create sample match
            sample_match = TennisMatch(
                match_id="test_match_001",
                player1="Novak Djokovic",
                player2="Rafael Nadal",
                tournament="French Open",
                surface="Clay",
                start_time=datetime.now() + timedelta(hours=2),
                player1_rank=1,
                player2_rank=2
            )
            
            # Generate predictions
            predictions = system._generate_predictions_for_match(sample_match)
            
            pipeline_results['predictions_count'] = len(predictions)
            pipeline_results['confidence_levels'] = [p.confidence for p in predictions]
            
            if predictions:
                pipeline_results['prediction_generation'] = 'success'
                pipeline_results['sample_predictions'] = [
                    {
                        'source': p.source.value,
                        'underdog_player': p.underdog_player,
                        'probability': p.underdog_probability,
                        'confidence': p.confidence
                    } for p in predictions
                ]
            else:
                pipeline_results['errors'].append("No predictions generated")
                
        except Exception as e:
            pipeline_results['errors'].append(str(e))
            self.deployment_results['warnings'].append(f"Prediction pipeline: {e}")
        
        self.deployment_results['components_validated']['prediction_pipeline'] = pipeline_results
    
    async def _test_risk_management(self):
        """Test risk management system"""
        logger.info("⚖️ Testing Risk Management")
        
        risk_results = {
            'risk_manager_creation': 'failed',
            'bet_evaluation': 'failed',
            'risk_limits_applied': False,
            'sample_evaluations': [],
            'errors': []
        }
        
        try:
            from src.api.risk_management_system import create_risk_manager, RiskLevel
            
            # Create risk manager
            risk_manager = create_risk_manager(RiskLevel.MODERATE, 10000.0)
            risk_results['risk_manager_creation'] = 'success'
            
            # Test bet evaluations with different scenarios
            test_cases = [
                {
                    'name': 'High confidence bet',
                    'prediction': {'confidence': 0.8, 'edge': 0.08, 'probability': 0.65},
                    'market_data': {'odds': 2.1, 'liquidity': 5000, 'volatility': 0.15, 'spread': 0.05},
                    'match_info': {'match_id': 'test1', 'player1': 'Player A', 'tournament': 'ATP 500'}
                },
                {
                    'name': 'Low confidence bet',
                    'prediction': {'confidence': 0.5, 'edge': 0.01, 'probability': 0.52},
                    'market_data': {'odds': 1.9, 'liquidity': 3000, 'volatility': 0.2, 'spread': 0.08},
                    'match_info': {'match_id': 'test2', 'player1': 'Player B', 'tournament': 'ATP 250'}
                },
                {
                    'name': 'High edge bet',
                    'prediction': {'confidence': 0.75, 'edge': 0.12, 'probability': 0.68},
                    'market_data': {'odds': 2.5, 'liquidity': 8000, 'volatility': 0.1, 'spread': 0.03},
                    'match_info': {'match_id': 'test3', 'player1': 'Player C', 'tournament': 'Masters 1000'}
                }
            ]
            
            for test_case in test_cases:
                evaluation = risk_manager.evaluate_bet_request(
                    test_case['prediction'],
                    test_case['market_data'],
                    test_case['match_info']
                )
                
                risk_results['sample_evaluations'].append({
                    'test_name': test_case['name'],
                    'approved': evaluation.get('approved', False),
                    'stake': evaluation.get('stake', 0),
                    'reason': evaluation.get('reason', 'No reason'),
                    'risk_score': evaluation.get('risk_score', 0)
                })
            
            # Check if risk limits are being applied
            approved_count = sum(1 for eval in risk_results['sample_evaluations'] if eval['approved'])
            risk_results['risk_limits_applied'] = approved_count < len(test_cases)  # Some should be rejected
            risk_results['bet_evaluation'] = 'success'
            
        except Exception as e:
            risk_results['errors'].append(str(e))
            self.deployment_results['warnings'].append(f"Risk management: {e}")
        
        self.deployment_results['components_validated']['risk_management'] = risk_results
    
    async def _final_system_validation(self):
        """Final comprehensive system validation"""
        logger.info("✅ Final System Validation")
        
        final_results = {
            'overall_health': 'unknown',
            'critical_components_ready': 0,
            'total_components': 0,
            'system_readiness_score': 0.0,
            'deployment_recommendations': []
        }
        
        # Count ready components
        components = self.deployment_results['components_validated']
        ready_components = 0
        total_components = 0
        
        for component_name, component_data in components.items():
            total_components += 1
            if isinstance(component_data, dict):
                if component_data.get('status') == 'ready' or \
                   component_data.get('initialization') == 'ready' or \
                   component_data.get('prediction_generation') == 'success' or \
                   component_data.get('bet_evaluation') == 'success':
                    ready_components += 1
        
        final_results['critical_components_ready'] = ready_components
        final_results['total_components'] = total_components
        final_results['system_readiness_score'] = ready_components / total_components if total_components > 0 else 0
        
        # Determine overall health
        if final_results['system_readiness_score'] >= 0.8:
            final_results['overall_health'] = 'excellent'
        elif final_results['system_readiness_score'] >= 0.6:
            final_results['overall_health'] = 'good'
        elif final_results['system_readiness_score'] >= 0.4:
            final_results['overall_health'] = 'degraded'
        else:
            final_results['overall_health'] = 'critical'
        
        # Generate recommendations
        if final_results['system_readiness_score'] >= 0.7:
            final_results['deployment_recommendations'].append(
                "✅ System ready for deployment in simulation mode"
            )
        else:
            final_results['deployment_recommendations'].append(
                "⚠️ Address component issues before production deployment"
            )
        
        if len(self.deployment_results['errors']) == 0:
            final_results['deployment_recommendations'].append(
                "🚀 No critical errors detected - system is stable"
            )
        else:
            final_results['deployment_recommendations'].append(
                f"❌ Resolve {len(self.deployment_results['errors'])} critical errors"
            )
        
        if len(self.deployment_results['warnings']) > 0:
            final_results['deployment_recommendations'].append(
                f"⚠️ Address {len(self.deployment_results['warnings'])} warnings for optimal performance"
            )
        
        self.deployment_results['components_validated']['final_validation'] = final_results
    
    def _generate_deployment_report(self):
        """Generate comprehensive deployment report"""
        logger.info("📊 Generating Deployment Report")
        
        report_path = os.path.join(self.base_dir, 'deployment_report.json')
        
        # Add summary to deployment results
        self.deployment_results['deployment_summary'] = {
            'total_components_tested': len(self.deployment_results['components_validated']),
            'critical_errors': len(self.deployment_results['errors']),
            'warnings': len(self.deployment_results['warnings']),
            'system_ready': self.deployment_results['system_ready'],
            'recommended_next_steps': self._get_next_steps()
        }
        
        # Save deployment report
        try:
            with open(report_path, 'w') as f:
                json.dump(self.deployment_results, f, indent=2, default=str)
            logger.info(f"📝 Deployment report saved to: {report_path}")
        except Exception as e:
            logger.error(f"Failed to save deployment report: {e}")
    
    def _get_next_steps(self) -> List[str]:
        """Get recommended next steps"""
        next_steps = []
        
        if self.deployment_results['system_ready']:
            next_steps.extend([
                "1. Configure Betfair API credentials for live trading",
                "2. Set up monitoring and alerting systems",
                "3. Start system in simulation mode for testing",
                "4. Monitor performance for 24-48 hours",
                "5. Gradually increase stake sizes if performance is good",
                "6. Implement additional risk controls as needed"
            ])
        else:
            next_steps.extend([
                "1. Address all critical errors listed in the report",
                "2. Resolve high-priority warnings",
                "3. Re-run deployment validation",
                "4. Test individual components that failed",
                "5. Check system dependencies and configuration"
            ])
        
        return next_steps
    
    def print_deployment_summary(self):
        """Print deployment summary to console"""
        print("\n" + "="*60)
        print("🎾 TENNIS BETTING SYSTEM DEPLOYMENT REPORT")
        print("="*60)
        
        print(f"📅 Deployment Time: {self.deployment_results['timestamp']}")
        print(f"📊 Status: {self.deployment_results['deployment_status'].upper()}")
        print(f"✅ System Ready: {self.deployment_results['system_ready']}")
        
        summary = self.deployment_results['deployment_summary']
        print(f"🔧 Components Tested: {summary['total_components_tested']}")
        print(f"❌ Critical Errors: {summary['critical_errors']}")
        print(f"⚠️ Warnings: {summary['warnings']}")
        
        if self.deployment_results['warnings']:
            print("\n⚠️ WARNINGS:")
            for warning in self.deployment_results['warnings']:
                print(f"  • {warning}")
        
        if self.deployment_results['errors']:
            print("\n❌ ERRORS:")
            for error in self.deployment_results['errors']:
                print(f"  • {error}")
        
        print("\n🎯 NEXT STEPS:")
        for step in summary['recommended_next_steps']:
            print(f"  {step}")
        
        print("\n" + "="*60)
        
        if self.deployment_results['system_ready']:
            print("🚀 DEPLOYMENT SUCCESSFUL - System ready for operation!")
        else:
            print("⚠️ DEPLOYMENT NEEDS ATTENTION - Address issues before production")
        
        print("="*60)


async def main():
    """Main deployment function"""
    deployer = TennisBettingSystemDeployer()
    
    # Run deployment with simulation mode
    results = await deployer.deploy_system(simulation_mode=True)
    
    # Print summary
    deployer.print_deployment_summary()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy Tennis Betting System')
    parser.add_argument('--production', action='store_true', help='Deploy for production (default: simulation)')
    
    args = parser.parse_args()
    
    # Run deployment
    results = asyncio.run(main())
    
    # Exit with appropriate code
    sys.exit(0 if results['system_ready'] else 1)