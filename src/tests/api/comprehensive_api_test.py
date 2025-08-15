#!/usr/bin/env python3
"""
Comprehensive API Testing Script
Tests all documented API endpoints from API_DOCUMENTATION.md
"""

import json
import requests
import time
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

class ComprehensiveApiTester:
    def __init__(self, base_url: str = "http://localhost:5001"):
        self.base_url = base_url
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'base_url': base_url,
            'internal_api_tests': {},
            'external_api_tests': {},
            'summary': {
                'total_endpoints': 0,
                'working_endpoints': 0,
                'failed_endpoints': 0,
                'authentication_required': [],
                'no_auth_required': []
            }
        }
        
        # Try to get API keys from environment
        self.api_tennis_key = os.getenv('API_TENNIS_KEY', '')
        self.rapidapi_key = os.getenv('RAPIDAPI_KEY', '')
        self.internal_api_key = os.getenv('API_KEY', 'test-key-123')
        
        print(f"🧪 COMPREHENSIVE API TESTING SUITE")
        print(f"📍 Testing server: {base_url}")
        print(f"🔑 API-Tennis Key: {'✅ Found' if self.api_tennis_key else '❌ Missing'}")
        print(f"🔑 RapidAPI Key: {'✅ Found' if self.rapidapi_key else '❌ Missing'}")
        print("=" * 70)

    def make_request(self, method: str, endpoint: str, headers: Optional[Dict] = None, 
                    data: Optional[Dict] = None, params: Optional[Dict] = None,
                    timeout: int = 10) -> Dict[str, Any]:
        """Make HTTP request with error handling"""
        try:
            url = f"{self.base_url}{endpoint}"
            
            # Default headers
            default_headers = {'Content-Type': 'application/json'}
            if headers:
                default_headers.update(headers)
            
            # Make request
            if method.upper() == 'GET':
                response = requests.get(url, headers=default_headers, params=params, timeout=timeout)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=default_headers, json=data, params=params, timeout=timeout)
            else:
                response = requests.request(method, url, headers=default_headers, json=data, params=params, timeout=timeout)
            
            # Parse response
            try:
                json_response = response.json()
            except:
                json_response = {'raw_response': response.text}
            
            return {
                'success': response.status_code < 400,
                'status_code': response.status_code,
                'response': json_response,
                'response_time_ms': response.elapsed.total_seconds() * 1000,
                'url': url
            }
            
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': 'Request timeout',
                'status_code': 0,
                'response_time_ms': timeout * 1000
            }
        except requests.exceptions.ConnectionError:
            return {
                'success': False,
                'error': 'Connection error - server may be down',
                'status_code': 0
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'status_code': 0
            }

    def test_internal_api_endpoints(self):
        """Test all internal Flask API endpoints"""
        print("\n🏠 TESTING INTERNAL FLASK API ENDPOINTS")
        print("=" * 50)
        
        # Basic endpoints (no authentication required)
        basic_endpoints = [
            {
                'name': 'Health Check',
                'method': 'GET',
                'endpoint': '/api/health',
                'description': 'Basic health status'
            },
            {
                'name': 'System Statistics',
                'method': 'GET',
                'endpoint': '/api/stats',
                'description': 'System statistics and ML predictor status'
            },
            {
                'name': 'Match Data',
                'method': 'GET',
                'endpoint': '/api/matches',
                'description': 'Get matches with underdog analysis'
            },
            {
                'name': 'Value Bets',
                'method': 'GET',
                'endpoint': '/api/value-bets',
                'description': 'Find value betting opportunities'
            },
            {
                'name': 'Player Info - Djokovic',
                'method': 'GET',
                'endpoint': '/api/player-info/Novak Djokovic',
                'description': 'Get player information'
            },
            {
                'name': 'Dashboard',
                'method': 'GET',
                'endpoint': '/',
                'description': 'Main dashboard page'
            }
        ]
        
        # Test basic endpoints
        for endpoint_test in basic_endpoints:
            print(f"\n📡 Testing: {endpoint_test['name']}")
            result = self.make_request(
                endpoint_test['method'],
                endpoint_test['endpoint']
            )
            
            self.results['internal_api_tests'][endpoint_test['name']] = {
                **result,
                'description': endpoint_test['description'],
                'requires_auth': False
            }
            
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"   {status} [{result.get('status_code', 0)}] {endpoint_test['description']}")
            if result.get('response_time_ms'):
                print(f"   ⏱️  Response Time: {result['response_time_ms']:.0f}ms")
            
            if not result['success']:
                print(f"   🚨 Error: {result.get('error', 'Unknown error')}")
        
        # POST endpoints requiring JSON data
        post_endpoints = [
            {
                'name': 'Test ML Prediction',
                'method': 'POST',
                'endpoint': '/api/test-ml',
                'data': {
                    'player1': 'Flavio Cobolli',
                    'player2': 'Novak Djokovic',
                    'tournament': 'US Open',
                    'surface': 'Hard'
                },
                'description': 'Test ML prediction engine'
            },
            {
                'name': 'Underdog Analysis',
                'method': 'POST',
                'endpoint': '/api/underdog-analysis',
                'data': {
                    'player1': 'Emma Raducanu',
                    'player2': 'Iga Swiatek',
                    'tournament': 'Wimbledon',
                    'surface': 'Grass'
                },
                'description': 'Detailed underdog scenario analysis'
            },
            {
                'name': 'Test Underdog',
                'method': 'POST',
                'endpoint': '/api/test-underdog',
                'data': {
                    'player1': 'Brandon Nakashima',
                    'player2': 'Carlos Alcaraz',
                    'tournament': 'ATP Masters',
                    'surface': 'Hard'
                },
                'description': 'Test underdog analysis system'
            }
        ]
        
        for endpoint_test in post_endpoints:
            print(f"\n📡 Testing: {endpoint_test['name']}")
            result = self.make_request(
                endpoint_test['method'],
                endpoint_test['endpoint'],
                data=endpoint_test['data']
            )
            
            self.results['internal_api_tests'][endpoint_test['name']] = {
                **result,
                'description': endpoint_test['description'],
                'requires_auth': False,
                'test_data': endpoint_test['data']
            }
            
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"   {status} [{result.get('status_code', 0)}] {endpoint_test['description']}")
            if result.get('response_time_ms'):
                print(f"   ⏱️  Response Time: {result['response_time_ms']:.0f}ms")
            
            if result['success'] and 'response' in result:
                resp = result['response']
                if isinstance(resp, dict):
                    if 'prediction' in resp:
                        pred = resp['prediction']
                        print(f"   📊 Prediction: {pred.get('probability', 'N/A')} confidence: {pred.get('confidence', 'N/A')}")
                    elif 'underdog_analysis' in resp:
                        analysis = resp['underdog_analysis']
                        print(f"   📊 Underdog Prob: {analysis.get('underdog_probability', 'N/A')}")
            
            if not result['success']:
                print(f"   🚨 Error: {result.get('error', 'Unknown error')}")

    def test_authenticated_endpoints(self):
        """Test endpoints requiring API key authentication"""
        print("\n🔐 TESTING AUTHENTICATED ENDPOINTS")
        print("=" * 50)
        
        auth_headers = {'X-API-Key': self.internal_api_key}
        
        auth_endpoints = [
            {
                'name': 'Refresh Data',
                'method': 'POST',
                'endpoint': '/api/refresh',
                'description': 'Manual data refresh'
            },
            {
                'name': 'Manual API Update',
                'method': 'POST',
                'endpoint': '/api/manual-api-update',
                'description': 'Manual API data update'
            },
            {
                'name': 'Redis Status',
                'method': 'GET',
                'endpoint': '/api/redis-status',
                'description': 'Redis connection status'
            },
            {
                'name': 'Clear API-Tennis Cache',
                'method': 'POST',
                'endpoint': '/api/api-tennis/clear-cache',
                'description': 'Clear API-Tennis cache'
            },
            {
                'name': 'Refresh Rankings',
                'method': 'POST',
                'endpoint': '/api/refresh-rankings',
                'description': 'Force refresh tennis rankings'
            }
        ]
        
        for endpoint_test in auth_endpoints:
            print(f"\n🔑 Testing: {endpoint_test['name']}")
            
            # Test without auth first
            result_no_auth = self.make_request(
                endpoint_test['method'],
                endpoint_test['endpoint']
            )
            
            # Test with auth
            result_with_auth = self.make_request(
                endpoint_test['method'],
                endpoint_test['endpoint'],
                headers=auth_headers
            )
            
            self.results['internal_api_tests'][f"{endpoint_test['name']} (No Auth)"] = {
                **result_no_auth,
                'description': f"{endpoint_test['description']} - no auth",
                'requires_auth': True
            }
            
            self.results['internal_api_tests'][f"{endpoint_test['name']} (With Auth)"] = {
                **result_with_auth,
                'description': f"{endpoint_test['description']} - with auth",
                'requires_auth': True
            }
            
            status_no_auth = "✅ PASS" if result_no_auth['success'] else "❌ FAIL"
            status_with_auth = "✅ PASS" if result_with_auth['success'] else "❌ FAIL"
            
            print(f"   🚫 No Auth: {status_no_auth} [{result_no_auth.get('status_code', 0)}]")
            print(f"   🔑 With Auth: {status_with_auth} [{result_with_auth.get('status_code', 0)}]")
            
            # Check if authentication is properly enforced
            if result_no_auth.get('status_code') == 401 and result_with_auth['success']:
                print(f"   ✅ Authentication properly enforced")
            elif result_no_auth['success'] and result_with_auth['success']:
                print(f"   ⚠️  Endpoint doesn't require authentication")
            else:
                print(f"   🚨 Authentication behavior unclear")

    def test_api_tennis_endpoints(self):
        """Test API-Tennis.com integration endpoints"""
        print("\n🎾 TESTING API-TENNIS.COM INTEGRATION ENDPOINTS")
        print("=" * 50)
        
        api_tennis_endpoints = [
            {
                'name': 'API-Tennis Status',
                'method': 'GET',
                'endpoint': '/api/api-tennis/status',
                'description': 'API-Tennis integration status'
            },
            {
                'name': 'API-Tennis Test Connection',
                'method': 'GET',
                'endpoint': '/api/api-tennis/test-connection',
                'description': 'Test API-Tennis connection'
            },
            {
                'name': 'API-Tennis Tournaments',
                'method': 'GET',
                'endpoint': '/api/api-tennis/tournaments',
                'description': 'Get tournaments from API-Tennis'
            },
            {
                'name': 'API-Tennis Matches',
                'method': 'GET',
                'endpoint': '/api/api-tennis/matches',
                'description': 'Get matches from API-Tennis'
            },
            {
                'name': 'API-Tennis Enhanced Data',
                'method': 'GET',
                'endpoint': '/api/api-tennis/enhanced',
                'description': 'Enhanced API-Tennis data collection'
            },
            {
                'name': 'API-Tennis Player Matches',
                'method': 'GET',
                'endpoint': '/api/api-tennis/player/Novak Djokovic/matches',
                'description': 'Get player-specific matches'
            },
            {
                'name': 'API-Tennis Match Odds',
                'method': 'GET',
                'endpoint': '/api/api-tennis/match/159923/odds',
                'description': 'Get match odds'
            }
        ]
        
        for endpoint_test in api_tennis_endpoints:
            print(f"\n🎾 Testing: {endpoint_test['name']}")
            result = self.make_request(
                endpoint_test['method'],
                endpoint_test['endpoint']
            )
            
            self.results['internal_api_tests'][endpoint_test['name']] = {
                **result,
                'description': endpoint_test['description'],
                'requires_auth': False,
                'integration_type': 'API-Tennis'
            }
            
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"   {status} [{result.get('status_code', 0)}] {endpoint_test['description']}")
            if result.get('response_time_ms'):
                print(f"   ⏱️  Response Time: {result['response_time_ms']:.0f}ms")
            
            if result['success'] and 'response' in result:
                resp = result['response']
                if isinstance(resp, dict):
                    if 'count' in resp:
                        print(f"   📊 Data Count: {resp['count']}")
                    if 'data_source' in resp:
                        print(f"   📡 Data Source: {resp['data_source']}")
                    if 'api_tennis_status' in resp:
                        status_info = resp['api_tennis_status']
                        print(f"   🔗 Integration Status: {status_info.get('status', 'Unknown')}")
            
            if not result['success']:
                print(f"   🚨 Error: {result.get('error', 'Unknown error')}")

    def test_external_api_tennis(self):
        """Test direct external API-Tennis.com endpoints"""
        print("\n🌐 TESTING EXTERNAL API-TENNIS.COM ENDPOINTS")
        print("=" * 50)
        
        if not self.api_tennis_key:
            print("⚠️  API-Tennis key not found. Skipping external API tests.")
            print("   Set API_TENNIS_KEY environment variable to test external APIs")
            return
        
        base_api_url = "https://api.api-tennis.com/tennis/"
        
        external_endpoints = [
            {
                'name': 'Get Events',
                'method': 'GET',
                'params': {
                    'method': 'get_events',
                    'APIkey': self.api_tennis_key
                },
                'description': 'Get supported tournament types'
            },
            {
                'name': 'Get Tournaments',
                'method': 'GET',
                'params': {
                    'method': 'get_tournaments',
                    'APIkey': self.api_tennis_key
                },
                'description': 'Get available tournaments'
            },
            {
                'name': 'Get Today Fixtures',
                'method': 'GET',
                'params': {
                    'method': 'get_fixtures',
                    'APIkey': self.api_tennis_key,
                    'date_start': datetime.now().strftime('%Y-%m-%d'),
                    'date_stop': datetime.now().strftime('%Y-%m-%d')
                },
                'description': 'Get today\'s fixtures'
            },
            {
                'name': 'Get Livescore',
                'method': 'GET',
                'params': {
                    'method': 'get_livescore',
                    'APIkey': self.api_tennis_key
                },
                'description': 'Get live tennis matches'
            },
            {
                'name': 'Get ATP Standings',
                'method': 'GET',
                'params': {
                    'method': 'get_standings',
                    'APIkey': self.api_tennis_key,
                    'event_type': 'ATP'
                },
                'description': 'Get ATP rankings'
            },
            {
                'name': 'Get WTA Standings',
                'method': 'GET',
                'params': {
                    'method': 'get_standings',
                    'APIkey': self.api_tennis_key,
                    'event_type': 'WTA'
                },
                'description': 'Get WTA rankings'
            },
            {
                'name': 'Get Player Info',
                'method': 'GET',
                'params': {
                    'method': 'get_players',
                    'APIkey': self.api_tennis_key,
                    'player_key': '1905'  # Djokovic
                },
                'description': 'Get player profile (Djokovic)'
            },
            {
                'name': 'Get H2H',
                'method': 'GET',
                'params': {
                    'method': 'get_H2H',
                    'APIkey': self.api_tennis_key,
                    'first_player_key': '1905',  # Djokovic
                    'second_player_key': '1642'  # Alcaraz
                },
                'description': 'Head-to-head analysis'
            }
        ]
        
        for endpoint_test in external_endpoints:
            print(f"\n🌐 Testing: {endpoint_test['name']}")
            
            try:
                response = requests.get(base_api_url, params=endpoint_test['params'], timeout=15)
                
                result = {
                    'success': response.status_code == 200,
                    'status_code': response.status_code,
                    'response_time_ms': response.elapsed.total_seconds() * 1000,
                    'url': response.url
                }
                
                try:
                    json_response = response.json()
                    result['response'] = json_response
                    
                    # Check if API returned success
                    if isinstance(json_response, dict):
                        api_success = json_response.get('success') == 1
                        result['api_success'] = api_success
                        
                        if not api_success:
                            result['api_error'] = json_response.get('error', 'Unknown API error')
                    
                except:
                    result['response'] = {'raw_text': response.text[:500]}
                
            except Exception as e:
                result = {
                    'success': False,
                    'error': str(e),
                    'status_code': 0
                }
            
            self.results['external_api_tests'][endpoint_test['name']] = {
                **result,
                'description': endpoint_test['description']
            }
            
            status = "✅ PASS" if result['success'] and result.get('api_success', True) else "❌ FAIL"
            print(f"   {status} [{result.get('status_code', 0)}] {endpoint_test['description']}")
            if result.get('response_time_ms'):
                print(f"   ⏱️  Response Time: {result['response_time_ms']:.0f}ms")
            
            if result['success'] and 'response' in result:
                resp = result['response']
                if isinstance(resp, dict):
                    if 'result' in resp and isinstance(resp['result'], list):
                        print(f"   📊 Results Count: {len(resp['result'])}")
                    elif 'result' in resp and isinstance(resp['result'], dict):
                        print(f"   📊 Result Type: Dictionary")
                    
                    if result.get('api_success') == False:
                        print(f"   🚨 API Error: {result.get('api_error', 'Unknown')}")
            
            if not result['success']:
                print(f"   🚨 Connection Error: {result.get('error', 'Unknown error')}")
            
            # Rate limiting - be respectful to external APIs
            time.sleep(0.5)

    def test_status_endpoints(self):
        """Test various status and monitoring endpoints"""
        print("\n📊 TESTING STATUS & MONITORING ENDPOINTS")
        print("=" * 50)
        
        status_endpoints = [
            {
                'name': 'API Economy Status',
                'method': 'GET',
                'endpoint': '/api/api-economy-status',
                'description': 'API economy usage status'
            },
            {
                'name': 'Comprehensive API Status',
                'method': 'GET',
                'endpoint': '/api/api-status',
                'description': 'Comprehensive API status'
            },
            {
                'name': 'Rankings Status',
                'method': 'GET',
                'endpoint': '/api/rankings-status',
                'description': 'Dynamic rankings system status'
            }
        ]
        
        for endpoint_test in status_endpoints:
            print(f"\n📊 Testing: {endpoint_test['name']}")
            result = self.make_request(
                endpoint_test['method'],
                endpoint_test['endpoint']
            )
            
            self.results['internal_api_tests'][endpoint_test['name']] = {
                **result,
                'description': endpoint_test['description'],
                'requires_auth': False,
                'category': 'status'
            }
            
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"   {status} [{result.get('status_code', 0)}] {endpoint_test['description']}")
            if result.get('response_time_ms'):
                print(f"   ⏱️  Response Time: {result['response_time_ms']:.0f}ms")
            
            if result['success'] and 'response' in result:
                resp = result['response']
                if isinstance(resp, dict):
                    # Show key status information
                    for key in ['success', 'status', 'available', 'daily_scheduler', 'api_usage']:
                        if key in resp:
                            print(f"   📋 {key}: {resp[key]}")
            
            if not result['success']:
                print(f"   🚨 Error: {result.get('error', 'Unknown error')}")

    def generate_summary(self):
        """Generate test summary and statistics"""
        print("\n📋 GENERATING TEST SUMMARY")
        print("=" * 50)
        
        # Count results
        all_tests = {**self.results['internal_api_tests'], **self.results['external_api_tests']}
        
        total = len(all_tests)
        working = sum(1 for test in all_tests.values() if test.get('success', False))
        failed = total - working
        
        auth_required = [name for name, test in all_tests.items() if test.get('requires_auth', False)]
        no_auth = [name for name, test in all_tests.items() if not test.get('requires_auth', False)]
        
        self.results['summary'].update({
            'total_endpoints': total,
            'working_endpoints': working,
            'failed_endpoints': failed,
            'authentication_required': auth_required,
            'no_auth_required': no_auth
        })
        
        print(f"📊 TOTAL ENDPOINTS TESTED: {total}")
        print(f"✅ WORKING: {working}")
        print(f"❌ FAILED: {failed}")
        print(f"📊 SUCCESS RATE: {(working/total*100):.1f}%")
        print(f"🔐 AUTH REQUIRED: {len(auth_required)}")
        print(f"🔓 NO AUTH: {len(no_auth)}")
        
        # Categorize by integration type
        internal_tests = [name for name, test in all_tests.items() if 'external_api_tests' not in str(test)]
        external_tests = list(self.results['external_api_tests'].keys())
        api_tennis_tests = [name for name, test in all_tests.items() if test.get('integration_type') == 'API-Tennis']
        
        print(f"\n📊 BY CATEGORY:")
        print(f"   🏠 Internal Flask APIs: {len(internal_tests)}")
        print(f"   🌐 External APIs: {len(external_tests)}")
        print(f"   🎾 API-Tennis Integration: {len(api_tennis_tests)}")
        
        # Show working external APIs
        working_external = [name for name, test in self.results['external_api_tests'].items() 
                          if test.get('success', False) and test.get('api_success', True)]
        
        if working_external:
            print(f"\n✅ WORKING EXTERNAL APIs:")
            for api_name in working_external:
                print(f"   - {api_name}")
        
        # Show failed endpoints
        failed_endpoints = [name for name, test in all_tests.items() if not test.get('success', False)]
        if failed_endpoints:
            print(f"\n❌ FAILED ENDPOINTS:")
            for endpoint_name in failed_endpoints:
                test_data = all_tests[endpoint_name]
                error = test_data.get('error', test_data.get('api_error', 'Unknown error'))
                print(f"   - {endpoint_name}: {error}")

    def save_results(self, filename: str = None):
        """Save detailed test results to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"/home/apps/Tennis_one_set/api_test_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"\n💾 Test results saved to: {filename}")
        except Exception as e:
            print(f"\n❌ Failed to save results: {e}")

    def run_all_tests(self):
        """Run comprehensive test suite"""
        print(f"🚀 STARTING COMPREHENSIVE API TEST SUITE")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # Test internal API endpoints
        self.test_internal_api_endpoints()
        
        # Test authenticated endpoints
        self.test_authenticated_endpoints()
        
        # Test API-Tennis integration endpoints
        self.test_api_tennis_endpoints()
        
        # Test status endpoints
        self.test_status_endpoints()
        
        # Test external API-Tennis.com endpoints
        self.test_external_api_tennis()
        
        # Generate summary
        self.generate_summary()
        
        # Save results
        self.save_results()
        
        print(f"\n🏁 TESTING COMPLETED")
        print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

if __name__ == "__main__":
    tester = ComprehensiveApiTester()
    tester.run_all_tests()