#!/usr/bin/env python3
"""
Workflow Integration Test
Test complete prediction → notification → tracking pipeline
"""

import os
import sys
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.enhanced_database_service import get_database_service, EnhancedDatabaseService
from data.enhanced_tennis_data_collector import get_tennis_collector, EnhancedTennisDataCollector
from utils.enhanced_rate_limiting import get_rate_limit_manager, EnhancedRateLimitManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WorkflowIntegrationTester:
    """Comprehensive workflow integration tester"""
    
    def __init__(self):
        self.db_service: Optional[EnhancedDatabaseService] = None
        self.tennis_collector: Optional[EnhancedTennisDataCollector] = None
        self.rate_limiter: Optional[EnhancedRateLimitManager] = None
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'errors': [],
            'warnings': [],
            'details': {}
        }
    
    def initialize_services(self) -> bool:
        """Initialize all required services"""
        logger.info("🔄 Initializing services for workflow testing...")
        
        try:
            # Initialize database service
            self.db_service = get_database_service()
            if not self.db_service.test_connection():
                raise Exception("Database connection failed")
            logger.info("✅ Database service initialized")
            
            # Initialize tennis data collector
            self.tennis_collector = get_tennis_collector()
            logger.info("✅ Tennis data collector initialized")
            
            # Initialize rate limiter
            self.rate_limiter = get_rate_limit_manager()
            logger.info("✅ Rate limiter initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Service initialization failed: {e}")
            self.test_results['errors'].append(f"Service initialization: {str(e)}")
            return False
    
    def test_database_operations(self) -> bool:
        """Test database operations"""
        logger.info("🧪 Testing database operations...")
        self.test_results['tests_run'] += 1
        
        try:
            # Test prediction logging
            prediction_data = {
                'match_date': datetime.now() + timedelta(hours=2),
                'player1': 'Test Player 1',
                'player2': 'Test Player 2',
                'tournament': 'Test Tournament',
                'surface': 'Hard',
                'round_name': 'Round 1',
                'our_probability': 0.65,
                'confidence': 'High',
                'ml_system': 'test_model',
                'prediction_type': 'match_winner',
                'key_factors': 'Test factors',
                'bookmaker_odds': 1.8,
                'bookmaker_probability': 0.556,
                'edge': 9.4,
                'recommendation': 'BET'
            }
            
            prediction_id = self.db_service.log_prediction(prediction_data)
            logger.info(f"✅ Prediction logged with ID: {prediction_id}")
            
            # Test betting record logging
            betting_data = {
                'player1': 'Test Player 1',
                'player2': 'Test Player 2',
                'tournament': 'Test Tournament',
                'match_date': datetime.now() + timedelta(hours=2),
                'our_probability': 0.65,
                'bookmaker_odds': 1.8,
                'implied_probability': 0.556,
                'edge_percentage': 9.4,
                'confidence_level': 'High',
                'bet_recommendation': 'BET',
                'suggested_stake': 25.0
            }
            
            betting_id = self.db_service.log_betting_record(betting_data, prediction_id)
            logger.info(f"✅ Betting record logged with ID: {betting_id}")
            
            # Test data retrieval
            recent_predictions = self.db_service.get_recent_predictions(days=1)
            recent_betting = self.db_service.get_recent_betting_records(days=1)
            
            logger.info(f"✅ Retrieved {len(recent_predictions)} predictions and {len(recent_betting)} betting records")
            
            # Test system stats
            stats = self.db_service.get_system_stats()
            logger.info(f"✅ System stats: {stats}")
            
            self.test_results['tests_passed'] += 1
            self.test_results['details']['database_test'] = {
                'status': 'passed',
                'prediction_id': prediction_id,
                'betting_id': betting_id,
                'predictions_count': len(recent_predictions),
                'betting_records_count': len(recent_betting),
                'stats': stats
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Database test failed: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"Database test: {str(e)}")
            self.test_results['details']['database_test'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_data_collection(self) -> bool:
        """Test tennis data collection"""
        logger.info("🧪 Testing tennis data collection...")
        self.test_results['tests_run'] += 1
        
        try:
            # Test API status
            api_status = self.tennis_collector.get_api_status()
            logger.info(f"✅ API status: {api_status}")
            
            # Test comprehensive data collection
            collection_results = self.tennis_collector.collect_comprehensive_data()
            
            total_matches = len(collection_results['live_matches']) + len(collection_results['upcoming_matches'])
            logger.info(f"✅ Data collection complete: {total_matches} matches, {len(collection_results['rankings'])} rankings")
            
            self.test_results['tests_passed'] += 1
            self.test_results['details']['data_collection_test'] = {
                'status': 'passed',
                'api_status': api_status,
                'collection_results': {
                    'sources_attempted': len(collection_results['sources_attempted']),
                    'sources_successful': len(collection_results['sources_successful']),
                    'live_matches': len(collection_results['live_matches']),
                    'upcoming_matches': len(collection_results['upcoming_matches']),
                    'rankings': len(collection_results['rankings']),
                    'errors': len(collection_results['errors'])
                }
            }
            
            # Test notification if any matches found
            if total_matches > 0:
                notification_message = f"🎾 Tennis Data Collection Complete\n\n"
                notification_message += f"📊 Matches Found: {total_matches}\n"
                notification_message += f"🏆 Rankings: {len(collection_results['rankings'])}\n"
                notification_message += f"🔄 Sources: {len(collection_results['sources_successful'])}/{len(collection_results['sources_attempted'])}\n"
                notification_message += f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}"
                
                notification_sent = self.tennis_collector.send_telegram_notification(notification_message)
                logger.info(f"✅ Notification test: {'sent' if notification_sent else 'skipped (not configured)'}")
                
                self.test_results['details']['data_collection_test']['notification_sent'] = notification_sent
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Data collection test failed: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"Data collection test: {str(e)}")
            self.test_results['details']['data_collection_test'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_rate_limiting(self) -> bool:
        """Test rate limiting functionality"""
        logger.info("🧪 Testing rate limiting...")
        self.test_results['tests_run'] += 1
        
        try:
            # Test rate limit status
            system_status = self.rate_limiter.get_system_status()
            logger.info(f"✅ Rate limiter status: {system_status['rate_limiter']}")
            
            # Test rate limit checking
            test_identifier = f"test_{int(time.time())}"
            
            # Test API rate limit
            allowed, rate_info = self.rate_limiter.check_rate_limit('api_requests', test_identifier)
            logger.info(f"✅ API rate limit test: allowed={allowed}, remaining={rate_info.remaining}")
            
            # Test prediction rate limit
            allowed, rate_info = self.rate_limiter.check_rate_limit('predictions', test_identifier)
            logger.info(f"✅ Prediction rate limit test: allowed={allowed}, remaining={rate_info.remaining}")
            
            # Test rate limit status retrieval
            status = self.rate_limiter.get_rate_limit_status('api_requests', test_identifier)
            logger.info(f"✅ Rate limit status: {status}")
            
            # Test rate limit reset
            reset_success = self.rate_limiter.reset_rate_limit('api_requests', test_identifier)
            logger.info(f"✅ Rate limit reset: {reset_success}")
            
            self.test_results['tests_passed'] += 1
            self.test_results['details']['rate_limiting_test'] = {
                'status': 'passed',
                'system_status': system_status,
                'test_results': {
                    'api_limit_check': allowed,
                    'rate_info': {
                        'limit': rate_info.limit,
                        'remaining': rate_info.remaining,
                        'window_size': rate_info.window_size
                    },
                    'reset_success': reset_success
                }
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Rate limiting test failed: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"Rate limiting test: {str(e)}")
            self.test_results['details']['rate_limiting_test'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_complete_workflow(self) -> bool:
        """Test complete prediction workflow"""
        logger.info("🧪 Testing complete workflow...")
        self.test_results['tests_run'] += 1
        
        try:
            workflow_start = time.time()
            
            # Step 1: Collect tennis data
            logger.info("Step 1: Collecting tennis data...")
            collection_results = self.tennis_collector.collect_comprehensive_data()
            
            # Step 2: Create prediction based on collected data
            if collection_results['upcoming_matches']:
                match = collection_results['upcoming_matches'][0]
                logger.info(f"Step 2: Creating prediction for {match['player1']} vs {match['player2']}")
                
                # Simulate ML prediction
                prediction_data = {
                    'match_date': datetime.now() + timedelta(hours=4),
                    'player1': match['player1'],
                    'player2': match['player2'],
                    'tournament': match['tournament'],
                    'surface': match.get('surface', 'Hard'),
                    'round_name': match.get('round_name', 'Unknown'),
                    'our_probability': 0.68,
                    'confidence': 'Medium',
                    'ml_system': 'enhanced_predictor',
                    'prediction_type': 'match_winner',
                    'key_factors': 'Recent form, surface advantage',
                    'bookmaker_odds': match.get('odds_player1', 1.75),
                    'bookmaker_probability': 0.571,
                    'edge': 10.9,
                    'recommendation': 'STRONG_BET'
                }
                
                prediction_id = self.db_service.log_prediction(prediction_data)
                logger.info(f"✅ Prediction created with ID: {prediction_id}")
                
                # Step 3: Create betting record
                logger.info("Step 3: Creating betting record...")
                betting_data = {
                    'player1': match['player1'],
                    'player2': match['player2'],
                    'tournament': match['tournament'],
                    'match_date': datetime.now() + timedelta(hours=4),
                    'our_probability': 0.68,
                    'bookmaker_odds': match.get('odds_player1', 1.75),
                    'implied_probability': 0.571,
                    'edge_percentage': 10.9,
                    'confidence_level': 'Medium',
                    'bet_recommendation': 'STRONG_BET',
                    'suggested_stake': 30.0
                }
                
                betting_id = self.db_service.log_betting_record(betting_data, prediction_id)
                logger.info(f"✅ Betting record created with ID: {betting_id}")
                
                # Step 4: Send notification
                logger.info("Step 4: Sending notification...")
                notification_message = f"🎾 <b>Tennis Betting Opportunity</b>\n\n"
                notification_message += f"🏆 <b>{match['tournament']}</b>\n"
                notification_message += f"⚔️ {match['player1']} vs {match['player2']}\n"
                notification_message += f"📊 Our Probability: {prediction_data['our_probability']:.1%}\n"
                notification_message += f"💰 Bookmaker Odds: {betting_data['bookmaker_odds']}\n"
                notification_message += f"📈 Edge: {betting_data['edge_percentage']:.1f}%\n"
                notification_message += f"💵 Suggested Stake: ${betting_data['suggested_stake']}\n"
                notification_message += f"🎯 Recommendation: {betting_data['bet_recommendation']}\n"
                notification_message += f"⏰ {datetime.now().strftime('%H:%M:%S')}"
                
                notification_sent = self.tennis_collector.send_telegram_notification(notification_message)
                logger.info(f"✅ Notification: {'sent' if notification_sent else 'skipped'}")
                
                # Step 5: Verify tracking
                logger.info("Step 5: Verifying tracking...")
                recent_predictions = self.db_service.get_recent_predictions(days=1)
                recent_betting = self.db_service.get_recent_betting_records(days=1)
                
                workflow_duration = time.time() - workflow_start
                
                logger.info(f"✅ Complete workflow test passed in {workflow_duration:.2f}s")
                
                self.test_results['tests_passed'] += 1
                self.test_results['details']['complete_workflow_test'] = {
                    'status': 'passed',
                    'duration_seconds': workflow_duration,
                    'steps_completed': 5,
                    'prediction_id': prediction_id,
                    'betting_id': betting_id,
                    'notification_sent': notification_sent,
                    'match_used': {
                        'player1': match['player1'],
                        'player2': match['player2'],
                        'tournament': match['tournament']
                    },
                    'tracking_verified': {
                        'predictions_count': len(recent_predictions),
                        'betting_records_count': len(recent_betting)
                    }
                }
                
                return True
            else:
                logger.warning("⚠️ No upcoming matches found for workflow test")
                self.test_results['warnings'].append("No upcoming matches found for complete workflow test")
                
                # Create a mock workflow test instead
                return self._test_mock_workflow()
        
        except Exception as e:
            logger.error(f"❌ Complete workflow test failed: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"Complete workflow test: {str(e)}")
            self.test_results['details']['complete_workflow_test'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def _test_mock_workflow(self) -> bool:
        """Test workflow with mock data"""
        logger.info("🧪 Testing workflow with mock data...")
        
        try:
            # Mock match data
            mock_match = {
                'player1': 'Mock Player 1',
                'player2': 'Mock Player 2',
                'tournament': 'Mock Tournament',
                'surface': 'Hard',
                'round_name': 'Round 1',
                'odds_player1': 1.65
            }
            
            # Create prediction
            prediction_data = {
                'match_date': datetime.now() + timedelta(hours=6),
                'player1': mock_match['player1'],
                'player2': mock_match['player2'],
                'tournament': mock_match['tournament'],
                'surface': mock_match['surface'],
                'round_name': mock_match['round_name'],
                'our_probability': 0.72,
                'confidence': 'High',
                'ml_system': 'mock_predictor',
                'prediction_type': 'match_winner',
                'key_factors': 'Mock factors for testing',
                'bookmaker_odds': mock_match['odds_player1'],
                'bookmaker_probability': 0.606,
                'edge': 11.4,
                'recommendation': 'BET'
            }
            
            prediction_id = self.db_service.log_prediction(prediction_data)
            
            # Create betting record
            betting_data = {
                'player1': mock_match['player1'],
                'player2': mock_match['player2'],
                'tournament': mock_match['tournament'],
                'match_date': datetime.now() + timedelta(hours=6),
                'our_probability': 0.72,
                'bookmaker_odds': mock_match['odds_player1'],
                'implied_probability': 0.606,
                'edge_percentage': 11.4,
                'confidence_level': 'High',
                'bet_recommendation': 'BET',
                'suggested_stake': 35.0
            }
            
            betting_id = self.db_service.log_betting_record(betting_data, prediction_id)
            
            # Send mock notification
            notification_message = f"🧪 <b>Mock Tennis Workflow Test</b>\n\n"
            notification_message += f"✅ Prediction ID: {prediction_id}\n"
            notification_message += f"✅ Betting ID: {betting_id}\n"
            notification_message += f"🎾 {mock_match['player1']} vs {mock_match['player2']}\n"
            notification_message += f"⏰ {datetime.now().strftime('%H:%M:%S')}"
            
            notification_sent = self.tennis_collector.send_telegram_notification(notification_message)
            
            self.test_results['tests_passed'] += 1
            self.test_results['details']['complete_workflow_test'] = {
                'status': 'passed_mock',
                'prediction_id': prediction_id,
                'betting_id': betting_id,
                'notification_sent': notification_sent,
                'mock_data_used': True
            }
            
            logger.info("✅ Mock workflow test completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Mock workflow test failed: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"Mock workflow test: {str(e)}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all workflow tests"""
        logger.info("🚀 Starting comprehensive workflow integration tests...")
        
        # Initialize services
        if not self.initialize_services():
            self.test_results['critical_failure'] = "Service initialization failed"
            return self.test_results
        
        # Run individual tests
        tests = [
            ('Database Operations', self.test_database_operations),
            ('Data Collection', self.test_data_collection),
            ('Rate Limiting', self.test_rate_limiting),
            ('Complete Workflow', self.test_complete_workflow)
        ]
        
        for test_name, test_function in tests:
            logger.info(f"🧪 Running {test_name} test...")
            try:
                test_function()
            except Exception as e:
                logger.error(f"❌ {test_name} test crashed: {e}")
                self.test_results['tests_failed'] += 1
                self.test_results['errors'].append(f"{test_name} test crashed: {str(e)}")
        
        # Calculate success rate
        if self.test_results['tests_run'] > 0:
            success_rate = (self.test_results['tests_passed'] / self.test_results['tests_run']) * 100
            self.test_results['success_rate'] = round(success_rate, 2)
        else:
            self.test_results['success_rate'] = 0.0
        
        # Overall status
        if self.test_results['tests_failed'] == 0:
            self.test_results['overall_status'] = 'PASSED'
        elif self.test_results['tests_passed'] > 0:
            self.test_results['overall_status'] = 'PARTIAL'
        else:
            self.test_results['overall_status'] = 'FAILED'
        
        # Log summary
        logger.info("📊 Workflow Integration Test Summary:")
        logger.info(f"   - Tests Run: {self.test_results['tests_run']}")
        logger.info(f"   - Tests Passed: {self.test_results['tests_passed']}")
        logger.info(f"   - Tests Failed: {self.test_results['tests_failed']}")
        logger.info(f"   - Success Rate: {self.test_results['success_rate']}%")
        logger.info(f"   - Overall Status: {self.test_results['overall_status']}")
        
        if self.test_results['errors']:
            logger.error(f"❌ {len(self.test_results['errors'])} errors occurred:")
            for error in self.test_results['errors']:
                logger.error(f"   - {error}")
        
        if self.test_results['warnings']:
            logger.warning(f"⚠️ {len(self.test_results['warnings'])} warnings:")
            for warning in self.test_results['warnings']:
                logger.warning(f"   - {warning}")
        
        return self.test_results

def main():
    """Run workflow integration tests"""
    print("🎾 TENNIS BACKEND WORKFLOW INTEGRATION TEST")
    print("=" * 60)
    
    tester = WorkflowIntegrationTester()
    results = tester.run_all_tests()
    
    print("\n" + "=" * 60)
    print("📋 FINAL TEST REPORT")
    print("=" * 60)
    print(f"Overall Status: {results['overall_status']}")
    print(f"Success Rate: {results['success_rate']}%")
    print(f"Tests: {results['tests_passed']}/{results['tests_run']} passed")
    
    if results['errors']:
        print(f"\n❌ Errors ({len(results['errors'])}):")
        for error in results['errors']:
            print(f"  - {error}")
    
    if results['warnings']:
        print(f"\n⚠️ Warnings ({len(results['warnings'])}):")
        for warning in results['warnings']:
            print(f"  - {warning}")
    
    # Save results to file
    results_file = f"workflow_test_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    return results['overall_status'] == 'PASSED'

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)