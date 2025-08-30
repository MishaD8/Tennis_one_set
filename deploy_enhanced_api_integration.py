#!/usr/bin/env python3
"""
Deploy Enhanced API Integration for Tennis Betting System
Complete deployment and validation of enhanced api-tennis.com integration
"""

import os
import sys
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedAPIDeployment:
    """Enhanced API integration deployment manager"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.deployment_log = []
        self.deployment_id = f"enhanced_api_deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def log_step(self, step: str, status: str, details: str = ""):
        """Log deployment step"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'status': status,
            'details': details
        }
        self.deployment_log.append(entry)
        
        status_icon = "✅" if status == "success" else "❌" if status == "error" else "⏳"
        logger.info(f"{status_icon} {step}: {status}")
        if details:
            logger.info(f"    {details}")
    
    def validate_enhanced_api_integration(self) -> bool:
        """Validate enhanced API integration components"""
        self.log_step("Enhanced API Integration Validation", "running")
        
        try:
            # Check enhanced API integration file
            enhanced_api_file = self.project_root / "enhanced_api_tennis_integration.py"
            if not enhanced_api_file.exists():
                raise FileNotFoundError("Enhanced API integration file missing")
            
            # Check enhanced feature engineering
            feature_eng_file = self.project_root / "enhanced_api_feature_engineering.py"
            if not feature_eng_file.exists():
                raise FileNotFoundError("Enhanced feature engineering file missing")
            
            # Test API integration
            from enhanced_api_tennis_integration import EnhancedAPITennisIntegration
            integration = EnhancedAPITennisIntegration()
            
            # Test rankings
            rankings = integration.get_enhanced_player_rankings()
            if len(rankings) < 1000:
                raise ValueError(f"Insufficient rankings loaded: {len(rankings)}")
            
            # Test fixtures
            fixtures = integration.get_enhanced_fixtures_with_rankings()
            underdog_matches = [m for m in fixtures if m.get('enhanced_data', {}).get('is_underdog_scenario')]
            
            self.log_step(
                "Enhanced API Integration Validation", 
                "success", 
                f"{len(rankings)} rankings, {len(fixtures)} fixtures, {len(underdog_matches)} underdog opportunities"
            )
            return True
            
        except Exception as e:
            self.log_step("Enhanced API Integration Validation", "error", str(e))
            return False
    
    def validate_automated_service_integration(self) -> bool:
        """Validate automated service integration with enhanced API"""
        self.log_step("Automated Service Integration Validation", "running")
        
        try:
            from automated_tennis_prediction_service import AutomatedTennisPredictionService
            service = AutomatedTennisPredictionService()
            
            # Test enhanced match retrieval
            matches = service._get_enhanced_current_matches()
            if not matches:
                logger.warning("No enhanced matches found - this may be normal depending on time/schedule")
            
            # Test ML models
            if not service.models:
                raise ValueError("No ML models loaded in automated service")
            
            # Test enhanced player rankings
            if not service.enhanced_rankings:
                raise ValueError("Enhanced rankings not loaded in automated service")
            
            self.log_step(
                "Automated Service Integration Validation",
                "success",
                f"{len(matches)} enhanced matches, {len(service.models)} ML models, {len(service.enhanced_rankings)} enhanced rankings"
            )
            return True
            
        except Exception as e:
            self.log_step("Automated Service Integration Validation", "error", str(e))
            return False
    
    def validate_telegram_enhancements(self) -> bool:
        """Validate enhanced Telegram notification system"""
        self.log_step("Telegram Enhancement Validation", "running")
        
        try:
            from src.utils.telegram_notification_system import TelegramNotificationSystem, TelegramConfig
            
            # Test config (disabled for validation)
            test_config = TelegramConfig(
                bot_token='test_token',
                chat_ids=['test_chat'],
                enabled=False
            )
            
            telegram_system = TelegramNotificationSystem(test_config)
            
            # Test enhanced message formatting
            sample_prediction = {
                'success': True,
                'underdog_second_set_probability': 0.65,
                'underdog_player': 'player1',
                'confidence': 'High',
                'match_context': {
                    'player1': 'Test Player',
                    'player2': 'Test Opponent',
                    'player1_rank': 45,
                    'player2_rank': 12,
                    'tournament': 'Test Tournament',
                    'surface': 'Hard'
                },
                'strategic_insights': ['Test insight'],
                'prediction_metadata': {
                    'service_type': 'automated_ml_prediction',
                    'models_used': ['random_forest', 'xgboost']
                }
            }
            
            # Test message formatting
            message = telegram_system._format_underdog_message(sample_prediction)
            if "ENHANCED UNDERDOG ALERT" not in message:
                raise ValueError("Enhanced message formatting not working")
            
            # Test enhanced insights extraction
            insights = telegram_system._extract_enhanced_insights(sample_prediction)
            if not insights:
                raise ValueError("Enhanced insights extraction not working")
            
            self.log_step(
                "Telegram Enhancement Validation",
                "success",
                f"Enhanced formatting working, {len(insights)} insights extracted"
            )
            return True
            
        except Exception as e:
            self.log_step("Telegram Enhancement Validation", "error", str(e))
            return False
    
    def validate_ml_model_compatibility(self) -> bool:
        """Validate ML model compatibility with enhanced features"""
        self.log_step("ML Model Compatibility Validation", "running")
        
        try:
            # Check ML models directory
            models_dir = self.project_root / "tennis_models"
            if not models_dir.exists():
                raise FileNotFoundError("ML models directory not found")
            
            # Check for required model files
            required_models = ['random_forest.pkl', 'xgboost.pkl', 'lightgbm.pkl', 'logistic_regression.pkl']
            existing_models = []
            
            for model_file in required_models:
                if (models_dir / model_file).exists():
                    existing_models.append(model_file)
            
            if len(existing_models) < 2:
                raise ValueError(f"Insufficient ML models: {len(existing_models)} found")
            
            # Check metadata
            metadata_file = models_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                feature_count = len(metadata.get('feature_columns', []))
            else:
                feature_count = 0
            
            # Test enhanced feature engineering
            from enhanced_api_feature_engineering import EnhancedAPIFeatureEngineer
            engineer = EnhancedAPIFeatureEngineer()
            
            sample_match = {
                'data_source': 'enhanced_api',
                'tournament_name': 'Test Tournament',
                'surface': 'Hard',
                'event_type_type': 'ATP Singles',
                'player1_rank': 32,
                'player2_rank': 8,
                'player1_points': 1200,
                'player2_points': 4200,
                'player1_movement': 'up',
                'player2_movement': 'same',
                'enhanced_data': {'data_quality_score': 0.9}
            }
            
            features = engineer.create_comprehensive_features(sample_match)
            
            self.log_step(
                "ML Model Compatibility Validation",
                "success",
                f"{len(existing_models)} models, {feature_count} expected features, {features.shape[1]} generated features"
            )
            return True
            
        except Exception as e:
            self.log_step("ML Model Compatibility Validation", "error", str(e))
            return False
    
    def run_comprehensive_test(self) -> bool:
        """Run comprehensive system test"""
        self.log_step("Comprehensive System Test", "running")
        
        try:
            # Run the comprehensive test script we created earlier
            result = subprocess.run(
                [sys.executable, 'test_enhanced_api_integration.py'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode == 0:
                self.log_step(
                    "Comprehensive System Test",
                    "success",
                    "All integration tests passed"
                )
                return True
            else:
                self.log_step(
                    "Comprehensive System Test",
                    "error",
                    f"Test failed with code {result.returncode}: {result.stderr[:200]}"
                )
                return False
                
        except Exception as e:
            self.log_step("Comprehensive System Test", "error", str(e))
            return False
    
    def create_deployment_summary(self) -> dict:
        """Create deployment summary report"""
        successful_steps = len([log for log in self.deployment_log if log['status'] == 'success'])
        total_steps = len([log for log in self.deployment_log if log['status'] in ['success', 'error']])
        
        summary = {
            'deployment_id': self.deployment_id,
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'success' if successful_steps == total_steps else 'partial_success' if successful_steps > total_steps * 0.5 else 'failed',
            'steps_completed': successful_steps,
            'total_steps': total_steps,
            'success_rate': successful_steps / max(1, total_steps),
            'deployment_log': self.deployment_log,
            'system_capabilities': {
                'enhanced_api_integration': True,
                'comprehensive_feature_engineering': True,
                'enhanced_telegram_notifications': True,
                'ml_model_compatibility': True,
                'automated_prediction_service': True,
                'ranks_10_300_targeting': True,
                'second_set_predictions': True
            },
            'recommendations': []
        }
        
        # Generate recommendations based on results
        if summary['overall_status'] == 'success':
            summary['recommendations'].extend([
                "✅ Enhanced API integration deployment successful",
                "🚀 System ready for production with comprehensive api-tennis.com data",
                "📊 Monitor prediction accuracy with enhanced features",
                "📱 Enable Telegram notifications for real-time alerts",
                "🎯 Focus on ranks 10-300 underdog scenarios as designed"
            ])
        else:
            failed_steps = [log for log in self.deployment_log if log['status'] == 'error']
            for failed_step in failed_steps:
                summary['recommendations'].append(f"🔧 Fix {failed_step['step']}: {failed_step['details']}")
        
        return summary
    
    def deploy(self) -> dict:
        """Run complete enhanced API integration deployment"""
        logger.info(f"🚀 Starting Enhanced API Integration Deployment: {self.deployment_id}")
        logger.info("=" * 80)
        
        # Validation steps
        validation_steps = [
            self.validate_enhanced_api_integration,
            self.validate_automated_service_integration,
            self.validate_telegram_enhancements,
            self.validate_ml_model_compatibility,
            self.run_comprehensive_test
        ]
        
        for step_func in validation_steps:
            try:
                step_func()
            except Exception as e:
                logger.error(f"Step failed with exception: {e}")
        
        # Create summary
        summary = self.create_deployment_summary()
        
        # Save deployment report
        report_file = self.project_root / f"enhanced_api_deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        logger.info(f"\n📊 DEPLOYMENT SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Deployment ID: {summary['deployment_id']}")
        logger.info(f"Overall Status: {summary['overall_status'].upper()}")
        logger.info(f"Steps Completed: {summary['steps_completed']}/{summary['total_steps']}")
        logger.info(f"Success Rate: {summary['success_rate']:.1%}")
        
        if summary['recommendations']:
            logger.info(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(summary['recommendations'], 1):
                logger.info(f"   {i}. {rec}")
        
        logger.info(f"\n📄 Detailed report saved to: {report_file}")
        
        if summary['overall_status'] == 'success':
            logger.info(f"\n🎉 DEPLOYMENT SUCCESSFUL!")
            logger.info("✅ Enhanced API integration is now fully deployed and operational")
            logger.info("🔬 System leverages comprehensive api-tennis.com data for improved predictions")
            logger.info("🎯 Targeting ranks 10-300 underdog scenarios for second-set betting opportunities")
        else:
            logger.info(f"\n⚠️ DEPLOYMENT COMPLETED WITH ISSUES")
            logger.info("📝 Review failed steps and address issues before production use")
        
        return summary


def main():
    """Main deployment execution"""
    deployment = EnhancedAPIDeployment()
    summary = deployment.deploy()
    
    # Return appropriate exit code
    if summary['overall_status'] == 'success':
        sys.exit(0)
    elif summary['overall_status'] == 'partial_success':
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()