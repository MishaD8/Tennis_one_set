#!/usr/bin/env python3
"""
Backend Service Verifier
Comprehensive verification of all backend services and components
"""

import os
import sys
import logging
import time
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.enhanced_database_service import get_database_service
from data.enhanced_tennis_data_collector import get_tennis_collector
from utils.enhanced_rate_limiting import get_rate_limit_manager
from api.enhanced_betting_tracker import get_betting_tracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BackendServiceVerifier:
    """Comprehensive backend service verification system"""
    
    def __init__(self):
        self.verification_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'UNKNOWN',
            'services': {},
            'critical_issues': [],
            'warnings': [],
            'recommendations': [],
            'summary': {}
        }
    
    def verify_environment_configuration(self) -> Tuple[bool, Dict[str, Any]]:
        """Verify environment configuration and API keys"""
        logger.info("🔧 Verifying environment configuration...")
        
        config_status = {
            'status': 'checking',
            'env_vars_checked': 0,
            'env_vars_configured': 0,
            'critical_missing': [],
            'optional_missing': [],
            'details': {}
        }
        
        # Critical environment variables
        critical_vars = {
            'DATABASE_URL': 'Database connection string',
            'API_TENNIS_KEY': 'API-Tennis.com API key for live data',
            'TELEGRAM_BOT_TOKEN': 'Telegram bot for notifications',
            'TELEGRAM_CHAT_IDS': 'Telegram chat IDs for notifications'
        }
        
        # Optional environment variables
        optional_vars = {
            'REDIS_URL': 'Redis for rate limiting',
            'RAPIDAPI_KEY': 'RapidAPI for backup data',
            'FLASK_SECRET_KEY': 'Flask secret key',
            'BETFAIR_APP_KEY': 'Betfair API for betting',
            'BETFAIR_USERNAME': 'Betfair username',
            'BETFAIR_PASSWORD': 'Betfair password'
        }
        
        try:
            all_vars = {**critical_vars, **optional_vars}
            config_status['env_vars_checked'] = len(all_vars)
            
            for var_name, description in all_vars.items():
                value = os.getenv(var_name, '').strip()
                is_configured = bool(value)
                
                config_status['details'][var_name] = {
                    'configured': is_configured,
                    'description': description,
                    'value_length': len(value) if is_configured else 0
                }
                
                if is_configured:
                    config_status['env_vars_configured'] += 1
                elif var_name in critical_vars:
                    config_status['critical_missing'].append(var_name)
                else:
                    config_status['optional_missing'].append(var_name)
            
            # Determine overall status
            if not config_status['critical_missing']:
                config_status['status'] = 'good'
            elif len(config_status['critical_missing']) <= 1:
                config_status['status'] = 'warning'
            else:
                config_status['status'] = 'critical'
            
            logger.info(f"✅ Environment check: {config_status['env_vars_configured']}/{config_status['env_vars_checked']} configured")
            
            return config_status['status'] != 'critical', config_status
            
        except Exception as e:
            logger.error(f"❌ Environment verification failed: {e}")
            config_status['status'] = 'error'
            config_status['error'] = str(e)
            return False, config_status
    
    def verify_database_service(self) -> Tuple[bool, Dict[str, Any]]:
        """Verify database service functionality"""
        logger.info("🗄️ Verifying database service...")
        
        db_status = {
            'status': 'checking',
            'connection_successful': False,
            'database_type': None,
            'operations_tested': 0,
            'operations_successful': 0,
            'test_results': {},
            'performance': {}
        }
        
        try:
            start_time = time.time()
            
            # Initialize database service
            db_service = get_database_service()
            db_status['database_type'] = db_service.database_type
            
            # Test connection
            if db_service.test_connection():
                db_status['connection_successful'] = True
                logger.info(f"✅ Database connection successful ({db_service.database_type})")
            else:
                raise Exception("Database connection test failed")
            
            # Test prediction logging
            db_status['operations_tested'] += 1
            test_prediction = {
                'match_date': datetime.now() + timedelta(hours=1),
                'player1': 'Test Player A',
                'player2': 'Test Player B',
                'tournament': 'Verification Test',
                'surface': 'Hard',
                'round_name': 'Test Round',
                'our_probability': 0.75,
                'confidence': 'High',
                'ml_system': 'verification_test',
                'prediction_type': 'match_winner',
                'key_factors': 'Service verification test',
                'bookmaker_odds': 1.5,
                'bookmaker_probability': 0.667,
                'edge': 8.3,
                'recommendation': 'BET'
            }
            
            prediction_id = db_service.log_prediction(test_prediction)
            db_status['operations_successful'] += 1
            db_status['test_results']['prediction_logging'] = {
                'success': True,
                'prediction_id': prediction_id
            }
            logger.info(f"✅ Prediction logging test passed (ID: {prediction_id})")
            
            # Test betting record logging
            db_status['operations_tested'] += 1
            test_betting = {
                'player1': 'Test Player A',
                'player2': 'Test Player B',
                'tournament': 'Verification Test',
                'match_date': datetime.now() + timedelta(hours=1),
                'our_probability': 0.75,
                'bookmaker_odds': 1.5,
                'implied_probability': 0.667,
                'edge_percentage': 8.3,
                'confidence_level': 'High',
                'bet_recommendation': 'BET',
                'suggested_stake': 20.0
            }
            
            betting_id = db_service.log_betting_record(test_betting, prediction_id)
            db_status['operations_successful'] += 1
            db_status['test_results']['betting_logging'] = {
                'success': True,
                'betting_id': betting_id
            }
            logger.info(f"✅ Betting record logging test passed (ID: {betting_id})")
            
            # Test data retrieval
            db_status['operations_tested'] += 1
            recent_predictions = db_service.get_recent_predictions(days=1)
            recent_betting = db_service.get_recent_betting_records(days=1)
            
            db_status['operations_successful'] += 1
            db_status['test_results']['data_retrieval'] = {
                'success': True,
                'predictions_count': len(recent_predictions),
                'betting_records_count': len(recent_betting)
            }
            logger.info(f"✅ Data retrieval test passed")
            
            # Test system stats
            db_status['operations_tested'] += 1
            stats = db_service.get_system_stats()
            db_status['operations_successful'] += 1
            db_status['test_results']['system_stats'] = {
                'success': True,
                'stats': stats
            }
            logger.info(f"✅ System stats test passed")
            
            # Performance metrics
            end_time = time.time()
            db_status['performance'] = {
                'total_time_seconds': round(end_time - start_time, 3),
                'operations_per_second': round(db_status['operations_tested'] / (end_time - start_time), 2)
            }
            
            # Overall status
            if db_status['operations_successful'] == db_status['operations_tested']:
                db_status['status'] = 'operational'
            elif db_status['operations_successful'] > 0:
                db_status['status'] = 'partial'
            else:
                db_status['status'] = 'failed'
            
            return db_status['status'] == 'operational', db_status
            
        except Exception as e:
            logger.error(f"❌ Database verification failed: {e}")
            db_status['status'] = 'error'
            db_status['error'] = str(e)
            db_status['traceback'] = traceback.format_exc()
            return False, db_status
    
    def verify_data_collection_service(self) -> Tuple[bool, Dict[str, Any]]:
        """Verify tennis data collection service"""
        logger.info("📡 Verifying data collection service...")
        
        collector_status = {
            'status': 'checking',
            'apis_configured': 0,
            'apis_operational': 0,
            'data_sources': {},
            'collection_test': {},
            'performance': {}
        }
        
        try:
            start_time = time.time()
            
            # Initialize data collector
            tennis_collector = get_tennis_collector()
            
            # Check API status
            api_status = tennis_collector.get_api_status()
            collector_status['data_sources'] = api_status['apis']
            
            for api_name, api_info in api_status['apis'].items():
                if api_info['configured']:
                    collector_status['apis_configured'] += 1
                if api_info.get('operational', False):
                    collector_status['apis_operational'] += 1
            
            logger.info(f"✅ APIs: {collector_status['apis_operational']}/{collector_status['apis_configured']} operational")
            
            # Test data collection
            collection_results = tennis_collector.collect_comprehensive_data()
            
            total_matches = len(collection_results['live_matches']) + len(collection_results['upcoming_matches'])
            
            collector_status['collection_test'] = {
                'sources_attempted': len(collection_results['sources_attempted']),
                'sources_successful': len(collection_results['sources_successful']),
                'live_matches': len(collection_results['live_matches']),
                'upcoming_matches': len(collection_results['upcoming_matches']),
                'rankings': len(collection_results['rankings']),
                'total_matches': total_matches,
                'errors': len(collection_results['errors'])
            }
            
            logger.info(f"✅ Data collection test: {total_matches} matches, {len(collection_results['rankings'])} rankings")
            
            # Test notification (if configured)
            test_message = f"🧪 Backend Verification Test\n⏰ {datetime.now().strftime('%H:%M:%S')}"
            notification_sent = tennis_collector.send_telegram_notification(test_message)
            collector_status['notification_test'] = {
                'sent': notification_sent,
                'configured': bool(tennis_collector.telegram_token and tennis_collector.telegram_chat_ids)
            }
            
            # Performance metrics
            end_time = time.time()
            collector_status['performance'] = {
                'collection_time_seconds': round(end_time - start_time, 3),
                'matches_per_second': round(total_matches / (end_time - start_time), 2) if total_matches > 0 else 0
            }
            
            # Overall status
            if collector_status['apis_operational'] > 0 and total_matches > 0:
                collector_status['status'] = 'operational'
            elif collector_status['apis_configured'] > 0:
                collector_status['status'] = 'partial'
            else:
                collector_status['status'] = 'failed'
            
            return collector_status['status'] in ['operational', 'partial'], collector_status
            
        except Exception as e:
            logger.error(f"❌ Data collection verification failed: {e}")
            collector_status['status'] = 'error'
            collector_status['error'] = str(e)
            collector_status['traceback'] = traceback.format_exc()
            return False, collector_status
    
    def verify_rate_limiting_service(self) -> Tuple[bool, Dict[str, Any]]:
        """Verify rate limiting service"""
        logger.info("⚡ Verifying rate limiting service...")
        
        rate_limit_status = {
            'status': 'checking',
            'limiter_type': None,
            'redis_available': False,
            'test_results': {},
            'performance': {}
        }
        
        try:
            start_time = time.time()
            
            # Initialize rate limiter
            rate_limiter = get_rate_limit_manager()
            
            # Get system status
            system_status = rate_limiter.get_system_status()
            rate_limit_status['limiter_type'] = system_status['rate_limiter']
            rate_limit_status['redis_available'] = system_status['redis_available']
            
            logger.info(f"✅ Rate limiter: {rate_limit_status['limiter_type']}")
            
            # Test rate limiting
            test_identifier = f"verification_test_{int(time.time())}"
            
            # Test API rate limit
            allowed, rate_info = rate_limiter.check_rate_limit('api_requests', test_identifier, limit=5, window=60)
            rate_limit_status['test_results']['api_limit'] = {
                'allowed': allowed,
                'limit': rate_info.limit,
                'remaining': rate_info.remaining,
                'window_size': rate_info.window_size
            }
            
            # Test multiple requests to verify limiting
            test_requests = []
            for i in range(3):
                allowed, rate_info = rate_limiter.check_rate_limit('api_requests', test_identifier, limit=5, window=60)
                test_requests.append({
                    'request_number': i + 1,
                    'allowed': allowed,
                    'remaining': rate_info.remaining
                })
            
            rate_limit_status['test_results']['multiple_requests'] = test_requests
            
            # Test rate limit reset
            reset_success = rate_limiter.reset_rate_limit('api_requests', test_identifier)
            rate_limit_status['test_results']['reset_test'] = reset_success
            
            # Test cleanup
            cleaned_count = rate_limiter.cleanup_expired_limits()
            rate_limit_status['test_results']['cleanup_test'] = cleaned_count
            
            # Performance metrics
            end_time = time.time()
            rate_limit_status['performance'] = {
                'test_time_seconds': round(end_time - start_time, 3),
                'requests_per_second': round(len(test_requests) / (end_time - start_time), 2)
            }
            
            rate_limit_status['status'] = 'operational'
            logger.info(f"✅ Rate limiting verification passed")
            
            return True, rate_limit_status
            
        except Exception as e:
            logger.error(f"❌ Rate limiting verification failed: {e}")
            rate_limit_status['status'] = 'error'
            rate_limit_status['error'] = str(e)
            rate_limit_status['traceback'] = traceback.format_exc()
            return False, rate_limit_status
    
    def verify_betting_tracker_service(self) -> Tuple[bool, Dict[str, Any]]:
        """Verify betting tracker service"""
        logger.info("💰 Verifying betting tracker service...")
        
        betting_status = {
            'status': 'checking',
            'test_results': {},
            'performance': {}
        }
        
        try:
            start_time = time.time()
            
            # Initialize betting tracker
            betting_tracker = get_betting_tracker()
            
            # Test betting decision logging
            match_info = {
                'match_id': f'test_match_{int(time.time())}',
                'player1': 'Verification Player 1',
                'player2': 'Verification Player 2',
                'tournament': 'Backend Verification Test',
                'surface': 'Hard',
                'round_name': 'Test Round',
                'match_date': datetime.now() + timedelta(hours=2)
            }
            
            prediction_info = {
                'our_probability': 0.68,
                'confidence_level': 'High',
                'model_used': 'verification_model',
                'key_factors': 'Backend verification test',
                'prediction_type': 'match_winner'
            }
            
            betting_info = {
                'bookmaker': 'Test Bookmaker',
                'odds': 1.85,
                'implied_probability': 0.541,
                'edge_percentage': 13.9,
                'stake_amount': 25.0,
                'decision': 'BET',
                'risk_level': 'medium'
            }
            
            record_id = betting_tracker.log_betting_decision(match_info, prediction_info, betting_info)
            betting_status['test_results']['decision_logging'] = {
                'success': True,
                'record_id': record_id
            }
            logger.info(f"✅ Betting decision logged: {record_id}")
            
            # Test outcome update
            from api.enhanced_betting_tracker import BettingOutcome
            outcome_updated = betting_tracker.update_betting_outcome(
                record_id, 
                BettingOutcome.WIN,
                actual_return=46.25,
                notes="Backend verification test win"
            )
            betting_status['test_results']['outcome_update'] = {
                'success': outcome_updated
            }
            logger.info(f"✅ Betting outcome updated: {outcome_updated}")
            
            # Test data retrieval
            active_bets = betting_tracker.get_active_bets()
            summary = betting_tracker.get_betting_summary(days=1)
            metrics = betting_tracker.get_performance_metrics()
            
            betting_status['test_results']['data_retrieval'] = {
                'active_bets_count': len(active_bets),
                'summary_available': bool(summary),
                'metrics_available': bool(metrics)
            }
            logger.info(f"✅ Betting data retrieval successful")
            
            # Test export
            export_filename = betting_tracker.export_betting_data(format='json', days=1)
            betting_status['test_results']['export'] = {
                'success': bool(export_filename),
                'filename': export_filename
            }
            logger.info(f"✅ Betting data export: {export_filename}")
            
            # Performance metrics
            end_time = time.time()
            betting_status['performance'] = {
                'test_time_seconds': round(end_time - start_time, 3),
                'operations_completed': 5
            }
            
            betting_status['status'] = 'operational'
            logger.info(f"✅ Betting tracker verification passed")
            
            return True, betting_status
            
        except Exception as e:
            logger.error(f"❌ Betting tracker verification failed: {e}")
            betting_status['status'] = 'error'
            betting_status['error'] = str(e)
            betting_status['traceback'] = traceback.format_exc()
            return False, betting_status
    
    def run_comprehensive_verification(self) -> Dict[str, Any]:
        """Run comprehensive verification of all backend services"""
        logger.info("🚀 Starting comprehensive backend service verification...")
        
        verification_start = time.time()
        
        # Services to verify
        services = [
            ('Environment Configuration', self.verify_environment_configuration),
            ('Database Service', self.verify_database_service),
            ('Data Collection Service', self.verify_data_collection_service),
            ('Rate Limiting Service', self.verify_rate_limiting_service),
            ('Betting Tracker Service', self.verify_betting_tracker_service)
        ]
        
        operational_count = 0
        
        for service_name, verify_function in services:
            logger.info(f"🔍 Verifying {service_name}...")
            
            try:
                is_operational, service_status = verify_function()
                
                self.verification_results['services'][service_name] = service_status
                
                if is_operational:
                    operational_count += 1
                    logger.info(f"✅ {service_name}: OPERATIONAL")
                else:
                    if service_status.get('status') == 'critical':
                        self.verification_results['critical_issues'].append(f"{service_name}: {service_status.get('error', 'Critical failure')}")
                    else:
                        self.verification_results['warnings'].append(f"{service_name}: {service_status.get('error', 'Partial functionality')}")
                    logger.warning(f"⚠️ {service_name}: {service_status.get('status', 'FAILED').upper()}")
            
            except Exception as e:
                logger.error(f"❌ {service_name} verification crashed: {e}")
                self.verification_results['services'][service_name] = {
                    'status': 'crashed',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                self.verification_results['critical_issues'].append(f"{service_name}: Verification crashed - {str(e)}")
        
        # Calculate overall status
        total_services = len(services)
        if operational_count == total_services:
            self.verification_results['overall_status'] = 'FULLY_OPERATIONAL'
        elif operational_count >= total_services * 0.8:
            self.verification_results['overall_status'] = 'MOSTLY_OPERATIONAL'
        elif operational_count > 0:
            self.verification_results['overall_status'] = 'PARTIALLY_OPERATIONAL'
        else:
            self.verification_results['overall_status'] = 'CRITICAL_FAILURES'
        
        # Generate summary
        verification_end = time.time()
        self.verification_results['summary'] = {
            'total_services': total_services,
            'operational_services': operational_count,
            'operational_percentage': round((operational_count / total_services) * 100, 2),
            'verification_time_seconds': round(verification_end - verification_start, 3),
            'critical_issues_count': len(self.verification_results['critical_issues']),
            'warnings_count': len(self.verification_results['warnings'])
        }
        
        # Generate recommendations
        self._generate_recommendations()
        
        # Log final summary
        logger.info("📊 Backend Service Verification Complete:")
        logger.info(f"   - Overall Status: {self.verification_results['overall_status']}")
        logger.info(f"   - Operational Services: {operational_count}/{total_services}")
        logger.info(f"   - Success Rate: {self.verification_results['summary']['operational_percentage']}%")
        logger.info(f"   - Verification Time: {self.verification_results['summary']['verification_time_seconds']}s")
        
        if self.verification_results['critical_issues']:
            logger.error(f"❌ {len(self.verification_results['critical_issues'])} critical issues found")
        
        if self.verification_results['warnings']:
            logger.warning(f"⚠️ {len(self.verification_results['warnings'])} warnings")
        
        return self.verification_results
    
    def _generate_recommendations(self):
        """Generate recommendations based on verification results"""
        recommendations = []
        
        # Check environment configuration
        env_status = self.verification_results['services'].get('Environment Configuration', {})
        if env_status.get('critical_missing'):
            recommendations.append("Configure missing critical environment variables: " + 
                                 ", ".join(env_status['critical_missing']))
        
        # Check database
        db_status = self.verification_results['services'].get('Database Service', {})
        if db_status.get('status') != 'operational':
            if db_status.get('database_type') == 'sqlite':
                recommendations.append("Database is using SQLite fallback. Consider configuring PostgreSQL for production.")
            else:
                recommendations.append("Database service issues detected. Check connection and credentials.")
        
        # Check data collection
        collector_status = self.verification_results['services'].get('Data Collection Service', {})
        if collector_status.get('apis_operational', 0) == 0:
            recommendations.append("No APIs are operational for data collection. Configure API keys.")
        elif collector_status.get('apis_operational', 0) < collector_status.get('apis_configured', 1):
            recommendations.append("Some APIs are not responding. Check API keys and quotas.")
        
        # Check rate limiting
        rate_status = self.verification_results['services'].get('Rate Limiting Service', {})
        if not rate_status.get('redis_available'):
            recommendations.append("Redis not available. Using in-memory rate limiting (not suitable for production).")
        
        # Check betting tracker
        betting_status = self.verification_results['services'].get('Betting Tracker Service', {})
        if betting_status.get('status') != 'operational':
            recommendations.append("Betting tracker service issues. Check database connectivity.")
        
        # Overall recommendations
        if self.verification_results['overall_status'] == 'CRITICAL_FAILURES':
            recommendations.append("URGENT: Multiple critical services are failing. System is not operational.")
        elif self.verification_results['overall_status'] == 'PARTIALLY_OPERATIONAL':
            recommendations.append("System has partial functionality. Address critical issues for full operation.")
        
        self.verification_results['recommendations'] = recommendations

def main():
    """Run backend service verification"""
    print("🎾 TENNIS BACKEND SERVICE VERIFICATION")
    print("=" * 60)
    
    verifier = BackendServiceVerifier()
    results = verifier.run_comprehensive_verification()
    
    print("\n" + "=" * 60)
    print("📋 VERIFICATION REPORT")
    print("=" * 60)
    
    print(f"Overall Status: {results['overall_status']}")
    print(f"Success Rate: {results['summary']['operational_percentage']}%")
    print(f"Services: {results['summary']['operational_services']}/{results['summary']['total_services']} operational")
    
    if results['critical_issues']:
        print(f"\n❌ Critical Issues ({len(results['critical_issues'])}):")
        for issue in results['critical_issues']:
            print(f"  - {issue}")
    
    if results['warnings']:
        print(f"\n⚠️ Warnings ({len(results['warnings'])}):")
        for warning in results['warnings']:
            print(f"  - {warning}")
    
    if results['recommendations']:
        print(f"\n🔧 Recommendations ({len(results['recommendations'])}):")
        for rec in results['recommendations']:
            print(f"  - {rec}")
    
    # Save detailed results
    results_file = f"backend_verification_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    # Return success code
    return results['overall_status'] in ['FULLY_OPERATIONAL', 'MOSTLY_OPERATIONAL']

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)