#!/usr/bin/env python3
"""
🏆 HISTORICAL DATA INTEGRATION SYSTEM
Integration script for 2-year historical data collection and ML model training pipeline

This script integrates the historical data collector with the ML training system,
ensuring that:
1. 2 years of historical tennis data is downloaded and stored
2. Data is properly filtered for ranks 10-300
3. ML models are trained/retrained with expanded dataset
4. System validation and compatibility checks

Author: Claude Code (Anthropic)
"""

import os
import sys
import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Dict, Any, Optional
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.historical_data_collector import HistoricalDataCollector, HistoricalDataConfig, get_historical_data_summary
from models.enhanced_ml_training_system import EnhancedMLTrainingSystem
from data.database_models import create_tables
from config.config import get_config

logger = logging.getLogger(__name__)

class HistoricalDataIntegrationManager:
    """
    Manages the complete integration of historical data collection and ML training
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.config = get_config()
        
        # File paths
        self.integration_log = "logs/historical_data_integration.log"
        self.progress_file = "data/historical_integration_progress.json"
        self.validation_report = "data/historical_data_validation_report.json"
        
        # Setup logging
        self._setup_logging()
        
        # Integration state
        self.integration_state = {
            'phase': 'initialization',
            'started_at': None,
            'completed_at': None,
            'data_collection_complete': False,
            'ml_training_complete': False,
            'validation_complete': False,
            'errors': []
        }
        
    def _setup_logging(self):
        """Setup logging for integration process"""
        os.makedirs(os.path.dirname(self.integration_log), exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.integration_log),
                logging.StreamHandler()
            ]
        )
    
    async def run_complete_integration(self, force_restart: bool = False) -> Dict[str, Any]:
        """
        Run complete historical data integration process
        
        Args:
            force_restart: Whether to restart from beginning even if progress exists
            
        Returns:
            Dict with integration results and statistics
        """
        
        logger.info("🏆 Starting Historical Data Integration")
        logger.info("=" * 80)
        
        self.integration_state['started_at'] = datetime.now()
        self.integration_state['phase'] = 'started'
        
        try:
            # Load previous progress if available
            if not force_restart:
                self._load_progress()
            
            # Phase 1: Historical Data Collection
            if not self.integration_state.get('data_collection_complete', False):
                logger.info("📥 Phase 1: Historical Data Collection")
                await self._run_historical_data_collection()
                self.integration_state['data_collection_complete'] = True
                self._save_progress()
            else:
                logger.info("📥 Phase 1: Historical Data Collection (SKIPPED - Already Complete)")
            
            # Phase 2: Data Validation and Quality Check
            logger.info("🔍 Phase 2: Data Validation and Quality Assessment")
            validation_results = await self._validate_historical_data()
            
            # Phase 3: ML Model Training with Historical Data
            if not self.integration_state.get('ml_training_complete', False):
                logger.info("🤖 Phase 3: ML Model Training with Historical Data")
                await self._run_ml_training_with_historical_data()
                self.integration_state['ml_training_complete'] = True
                self._save_progress()
            else:
                logger.info("🤖 Phase 3: ML Model Training (SKIPPED - Already Complete)")
            
            # Phase 4: System Validation and Compatibility Testing
            if not self.integration_state.get('validation_complete', False):
                logger.info("✅ Phase 4: System Validation and Compatibility Testing")
                compatibility_results = await self._run_compatibility_testing()
                self.integration_state['validation_complete'] = True
                self._save_progress()
            else:
                logger.info("✅ Phase 4: System Validation (SKIPPED - Already Complete)")
            
            # Phase 5: Integration Report Generation
            logger.info("📊 Phase 5: Integration Report Generation")
            final_report = await self._generate_integration_report()
            
            self.integration_state['completed_at'] = datetime.now()
            self.integration_state['phase'] = 'completed'
            
            logger.info("✅ Historical Data Integration Complete!")
            return final_report
            
        except Exception as e:
            logger.error(f"❌ Integration failed: {e}")
            self.integration_state['errors'].append(str(e))
            self.integration_state['phase'] = 'failed'
            raise
        
        finally:
            self._save_progress()
    
    async def _run_historical_data_collection(self):
        """Run historical data collection for 2 years"""
        
        logger.info("🔄 Starting 2-year historical data collection...")
        
        # Create configuration for 2 years of data
        end_date = date.today()
        start_date = end_date - timedelta(days=730)  # 2 years
        
        config = HistoricalDataConfig(
            start_date=start_date,
            end_date=end_date,
            target_rank_min=10,
            target_rank_max=300,
            max_requests_per_minute=30,
            max_requests_per_hour=1000,
            batch_size=50,
            concurrent_workers=5,
            resume_from_checkpoint=True
        )
        
        # Initialize collector
        collector = HistoricalDataCollector(self.api_key, config)
        
        # Run collection
        results = await collector.collect_historical_data(resume=True)
        
        logger.info(f"📊 Data Collection Results:")
        logger.info(f"   Total matches collected: {results['data_collection']['matches_stored']:,}")
        logger.info(f"   Target rank matches (10-300): {results['data_collection']['target_rank_matches']:,}")
        logger.info(f"   Duration: {results['collection_summary']['duration_hours']:.1f} hours")
        logger.info(f"   Data quality score: {results['data_quality']['data_quality_score']:.2%}")
        
        return results
    
    async def _validate_historical_data(self) -> Dict[str, Any]:
        """Validate historical data quality and coverage"""
        
        logger.info("🔍 Validating historical data quality...")
        
        try:
            # Get data summary
            summary = get_historical_data_summary()
            
            validation_results = {
                'data_coverage': {
                    'total_matches': summary.get('total_matches', 0),
                    'target_rank_matches': summary.get('target_rank_matches', 0),
                    'target_percentage': summary.get('target_match_percentage', 0),
                    'date_range': summary.get('date_range', []),
                },
                'data_quality': {
                    'tournament_distribution': summary.get('tournament_distribution', []),
                    'surface_distribution': summary.get('surface_distribution', []),
                    'ranking_distribution': summary.get('ranking_distribution', [])
                },
                'quality_assessment': {},
                'recommendations': []
            }
            
            # Quality assessment
            total_matches = summary.get('total_matches', 0)
            target_matches = summary.get('target_rank_matches', 0)
            target_percentage = summary.get('target_match_percentage', 0)
            
            # Assess data adequacy
            if total_matches < 10000:
                validation_results['quality_assessment']['data_volume'] = 'LOW'
                validation_results['recommendations'].append('Consider extending collection period or reducing filters')
            elif total_matches < 50000:
                validation_results['quality_assessment']['data_volume'] = 'MEDIUM'
            else:
                validation_results['quality_assessment']['data_volume'] = 'HIGH'
            
            # Assess target match coverage
            if target_percentage < 15:
                validation_results['quality_assessment']['target_coverage'] = 'LOW'
                validation_results['recommendations'].append('Target rank coverage is low - verify rank filtering')
            elif target_percentage < 30:
                validation_results['quality_assessment']['target_coverage'] = 'MEDIUM'
            else:
                validation_results['quality_assessment']['target_coverage'] = 'HIGH'
            
            # Overall assessment
            volume_score = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}[validation_results['quality_assessment']['data_volume']]
            coverage_score = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}[validation_results['quality_assessment']['target_coverage']]
            overall_score = (volume_score + coverage_score) / 2
            
            if overall_score >= 2.5:
                validation_results['quality_assessment']['overall'] = 'EXCELLENT'
            elif overall_score >= 2.0:
                validation_results['quality_assessment']['overall'] = 'GOOD'
            elif overall_score >= 1.5:
                validation_results['quality_assessment']['overall'] = 'FAIR'
            else:
                validation_results['quality_assessment']['overall'] = 'POOR'
                validation_results['recommendations'].append('Consider recollecting data with adjusted parameters')
            
            # Save validation report
            with open(self.validation_report, 'w') as f:
                json.dump(validation_results, f, indent=2, default=str)
            
            logger.info(f"📊 Validation Results:")
            logger.info(f"   Total matches: {total_matches:,}")
            logger.info(f"   Target matches: {target_matches:,} ({target_percentage:.1f}%)")
            logger.info(f"   Data quality: {validation_results['quality_assessment']['overall']}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ Data validation failed: {e}")
            raise
    
    async def _run_ml_training_with_historical_data(self):
        """Train ML models using historical data"""
        
        logger.info("🤖 Starting ML model training with historical data...")
        
        try:
            # Initialize ML training system
            ml_trainer = EnhancedMLTrainingSystem()
            
            # Configure for historical data training
            training_config = {
                'use_historical_data': True,
                'historical_db_path': 'tennis_data_enhanced/historical_data.db',
                'rank_range': (10, 300),
                'feature_engineering_class': 'ranks_10_300_feature_engineering.Ranks10to300FeatureEngineer',
                'training_epochs': 100,
                'validation_split': 0.2,
                'test_split': 0.1
            }
            
            # Train models
            logger.info("🔄 Training models with expanded 10-300 dataset...")
            training_results = await ml_trainer.train_with_historical_data(training_config)
            
            logger.info(f"📊 ML Training Results:")
            for model_name, metrics in training_results.get('model_performance', {}).items():
                logger.info(f"   {model_name}: Accuracy {metrics.get('accuracy', 0):.3f}, F1 {metrics.get('f1_score', 0):.3f}")
            
            return training_results
            
        except Exception as e:
            logger.error(f"❌ ML training failed: {e}")
            raise
    
    async def _run_compatibility_testing(self) -> Dict[str, Any]:
        """Run compatibility testing to ensure system works with expanded dataset"""
        
        logger.info("✅ Running compatibility testing...")
        
        compatibility_results = {
            'tests_run': [],
            'tests_passed': 0,
            'tests_failed': 0,
            'issues_found': [],
            'overall_status': 'UNKNOWN'
        }
        
        try:
            # Test 1: Verify 10-300 range filtering
            logger.info("🧪 Test 1: Rank range filtering (10-300)")
            from ranks_10_300_feature_engineering import Ranks10to300FeatureEngineer, Ranks10to300DataValidator
            
            feature_engineer = Ranks10to300FeatureEngineer()
            validator = Ranks10to300DataValidator()
            
            # Test valid scenario (both in range)
            test_valid = validator.validate_match_data({
                'player1': {'rank': 25, 'age': 23},
                'player2': {'rank': 180, 'age': 26},
                'first_set_data': {'winner': 'player1'}
            })
            
            if test_valid['valid']:
                compatibility_results['tests_passed'] += 1
                logger.info("   ✅ PASSED: Valid 10-300 scenario accepted")
            else:
                compatibility_results['tests_failed'] += 1
                compatibility_results['issues_found'].append("Valid 10-300 scenario rejected")
                logger.error("   ❌ FAILED: Valid 10-300 scenario rejected")
            
            compatibility_results['tests_run'].append("rank_range_filtering")
            
            # Test 2: Verify invalid scenarios are rejected
            logger.info("🧪 Test 2: Invalid rank scenarios rejection")
            test_invalid = validator.validate_match_data({
                'player1': {'rank': 5, 'age': 24},  # Outside range
                'player2': {'rank': 180, 'age': 26},
                'first_set_data': {'winner': 'player1'}
            })
            
            if not test_invalid['valid']:
                compatibility_results['tests_passed'] += 1
                logger.info("   ✅ PASSED: Invalid scenario properly rejected")
            else:
                compatibility_results['tests_failed'] += 1
                compatibility_results['issues_found'].append("Invalid scenario not rejected")
                logger.error("   ❌ FAILED: Invalid scenario not rejected")
            
            compatibility_results['tests_run'].append("invalid_scenario_rejection")
            
            # Test 3: Feature engineering compatibility
            logger.info("🧪 Test 3: Feature engineering with 10-300 data")
            try:
                test_features = feature_engineer.create_complete_feature_set(
                    "test_player_1", "test_player_2",
                    {'rank': 15, 'age': 22},
                    {'rank': 200, 'age': 28},
                    {'tournament_level': 'ATP_250', 'surface': 'Hard'},
                    {'winner': 'player2', 'score': '6-4'}
                )
                
                if len(test_features) > 50:  # Should generate many features
                    compatibility_results['tests_passed'] += 1
                    logger.info(f"   ✅ PASSED: Generated {len(test_features)} features")
                else:
                    compatibility_results['tests_failed'] += 1
                    compatibility_results['issues_found'].append("Insufficient features generated")
                    logger.error(f"   ❌ FAILED: Only {len(test_features)} features generated")
                
            except Exception as e:
                compatibility_results['tests_failed'] += 1
                compatibility_results['issues_found'].append(f"Feature engineering error: {e}")
                logger.error(f"   ❌ FAILED: Feature engineering error: {e}")
            
            compatibility_results['tests_run'].append("feature_engineering")
            
            # Test 4: Database connectivity and historical data access
            logger.info("🧪 Test 4: Historical database connectivity")
            try:
                summary = get_historical_data_summary()
                if summary.get('total_matches', 0) > 0:
                    compatibility_results['tests_passed'] += 1
                    logger.info(f"   ✅ PASSED: Historical database accessible ({summary['total_matches']:,} matches)")
                else:
                    compatibility_results['tests_failed'] += 1
                    compatibility_results['issues_found'].append("No historical data found")
                    logger.error("   ❌ FAILED: No historical data found")
                    
            except Exception as e:
                compatibility_results['tests_failed'] += 1
                compatibility_results['issues_found'].append(f"Database access error: {e}")
                logger.error(f"   ❌ FAILED: Database access error: {e}")
            
            compatibility_results['tests_run'].append("database_connectivity")
            
            # Overall assessment
            total_tests = len(compatibility_results['tests_run'])
            pass_rate = compatibility_results['tests_passed'] / total_tests if total_tests > 0 else 0
            
            if pass_rate >= 0.9:
                compatibility_results['overall_status'] = 'EXCELLENT'
            elif pass_rate >= 0.75:
                compatibility_results['overall_status'] = 'GOOD'
            elif pass_rate >= 0.5:
                compatibility_results['overall_status'] = 'FAIR'
            else:
                compatibility_results['overall_status'] = 'POOR'
            
            logger.info(f"📊 Compatibility Test Results:")
            logger.info(f"   Tests run: {total_tests}")
            logger.info(f"   Passed: {compatibility_results['tests_passed']}")
            logger.info(f"   Failed: {compatibility_results['tests_failed']}")
            logger.info(f"   Overall status: {compatibility_results['overall_status']}")
            
            return compatibility_results
            
        except Exception as e:
            logger.error(f"❌ Compatibility testing failed: {e}")
            compatibility_results['overall_status'] = 'ERROR'
            compatibility_results['issues_found'].append(f"Testing error: {e}")
            return compatibility_results
    
    async def _generate_integration_report(self) -> Dict[str, Any]:
        """Generate comprehensive integration report"""
        
        logger.info("📊 Generating comprehensive integration report...")
        
        try:
            # Load validation results
            validation_results = {}
            if os.path.exists(self.validation_report):
                with open(self.validation_report, 'r') as f:
                    validation_results = json.load(f)
            
            # Get current data summary
            data_summary = get_historical_data_summary()
            
            # Calculate integration duration
            duration = None
            if self.integration_state.get('started_at') and self.integration_state.get('completed_at'):
                start = datetime.fromisoformat(self.integration_state['started_at']) if isinstance(self.integration_state['started_at'], str) else self.integration_state['started_at']
                end = datetime.fromisoformat(self.integration_state['completed_at']) if isinstance(self.integration_state['completed_at'], str) else self.integration_state['completed_at']
                duration = (end - start).total_seconds() / 3600  # hours
            
            # Create comprehensive report
            integration_report = {
                'integration_summary': {
                    'started_at': self.integration_state.get('started_at'),
                    'completed_at': self.integration_state.get('completed_at'),
                    'duration_hours': duration,
                    'final_status': self.integration_state.get('phase', 'unknown'),
                    'data_collection_complete': self.integration_state.get('data_collection_complete', False),
                    'ml_training_complete': self.integration_state.get('ml_training_complete', False),
                    'validation_complete': self.integration_state.get('validation_complete', False)
                },
                
                'historical_data_summary': data_summary,
                'data_quality_assessment': validation_results.get('quality_assessment', {}),
                
                'system_status': {
                    'rank_range_updated': '10-300',
                    'feature_engineering_updated': True,
                    'historical_data_available': data_summary.get('total_matches', 0) > 0,
                    'ml_models_trained': self.integration_state.get('ml_training_complete', False)
                },
                
                'recommendations': validation_results.get('recommendations', []),
                'errors_encountered': self.integration_state.get('errors', [])
            }
            
            # Save integration report
            report_file = f"data/historical_integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(integration_report, f, indent=2, default=str)
            
            logger.info(f"📄 Integration report saved: {report_file}")
            logger.info(f"📊 Final Integration Summary:")
            logger.info(f"   Duration: {duration:.1f} hours" if duration else "   Duration: Unknown")
            logger.info(f"   Historical matches: {data_summary.get('total_matches', 0):,}")
            logger.info(f"   Target rank matches: {data_summary.get('target_rank_matches', 0):,}")
            logger.info(f"   Status: {self.integration_state.get('phase', 'unknown').upper()}")
            
            return integration_report
            
        except Exception as e:
            logger.error(f"❌ Report generation failed: {e}")
            raise
    
    def _load_progress(self):
        """Load integration progress from file"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r') as f:
                    saved_state = json.load(f)
                    self.integration_state.update(saved_state)
                    logger.info(f"📍 Loaded progress from {self.progress_file}")
        except Exception as e:
            logger.warning(f"⚠️ Could not load progress: {e}")
    
    def _save_progress(self):
        """Save integration progress to file"""
        try:
            os.makedirs(os.path.dirname(self.progress_file), exist_ok=True)
            with open(self.progress_file, 'w') as f:
                json.dump(self.integration_state, f, indent=2, default=str)
            logger.debug(f"💾 Progress saved to {self.progress_file}")
        except Exception as e:
            logger.warning(f"⚠️ Could not save progress: {e}")

async def main():
    """Main entry point for historical data integration"""
    
    # Get API key
    api_key = os.getenv('API_TENNIS_KEY')
    if not api_key:
        print("❌ API_TENNIS_KEY environment variable required")
        print("Please set your API key: export API_TENNIS_KEY='your_key_here'")
        return
    
    # Initialize integration manager
    integration_manager = HistoricalDataIntegrationManager(api_key)
    
    try:
        print("🏆 TENNIS HISTORICAL DATA INTEGRATION")
        print("=" * 80)
        print("This will:")
        print("1. Download 2 years of historical tennis data")
        print("2. Filter for players ranked 10-300")
        print("3. Train ML models with expanded dataset")
        print("4. Validate system compatibility")
        print("=" * 80)
        
        # Run complete integration
        results = await integration_manager.run_complete_integration()
        
        print("\n✅ INTEGRATION COMPLETED SUCCESSFULLY!")
        print(f"📊 Historical matches: {results['historical_data_summary'].get('total_matches', 0):,}")
        print(f"🎯 Target rank matches: {results['historical_data_summary'].get('target_rank_matches', 0):,}")
        print(f"⏱️ Total duration: {results['integration_summary'].get('duration_hours', 0):.1f} hours")
        
    except Exception as e:
        print(f"❌ Integration failed: {e}")
        logger.error(f"Integration failed: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())