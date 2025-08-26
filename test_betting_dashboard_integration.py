#!/usr/bin/env python3
"""
🎾 Betting Dashboard Integration Test
Test all betting statistics endpoints for the main dashboard
"""

import sys
import os
import requests
import json
from time import sleep

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def test_betting_dashboard_integration():
    """Test betting dashboard API integration"""
    print("🎾 BETTING DASHBOARD INTEGRATION TEST")
    print("=" * 60)
    
    # Dashboard server URL
    base_url = "http://65.109.135.2:5001"
    
    # Test endpoints
    endpoints = [
        {
            'name': 'Dashboard Health Check',
            'url': f'{base_url}/api/betting-dashboard/health',
            'expected_keys': ['success', 'overall_status', 'components']
        },
        {
            'name': 'Betting Statistics (1 week)',
            'url': f'{base_url}/api/betting-dashboard/statistics?timeframe=1_week',
            'expected_keys': ['success', 'statistics']
        },
        {
            'name': 'Betting Statistics (1 month)',
            'url': f'{base_url}/api/betting-dashboard/statistics?timeframe=1_month',
            'expected_keys': ['success', 'statistics']
        },
        {
            'name': 'Profit Timeline Chart Data',
            'url': f'{base_url}/api/betting-dashboard/charts-data?timeframe=1_week&chart_type=profit_timeline',
            'expected_keys': ['success', 'data']
        },
        {
            'name': 'Win Rate Trend Chart Data',
            'url': f'{base_url}/api/betting-dashboard/charts-data?timeframe=1_week&chart_type=win_rate_trend',
            'expected_keys': ['success', 'data']
        },
        {
            'name': 'Odds Distribution Chart Data',
            'url': f'{base_url}/api/betting-dashboard/charts-data?timeframe=1_week&chart_type=odds_distribution',
            'expected_keys': ['success', 'data']
        },
        {
            'name': 'Monthly Performance Chart Data',
            'url': f'{base_url}/api/betting-dashboard/charts-data?timeframe=1_month&chart_type=monthly_performance',
            'expected_keys': ['success', 'data']
        },
        {
            'name': 'Fallback Betting Statistics',
            'url': f'{base_url}/api/betting/statistics?timeframe=1_week&test_mode=live',
            'expected_keys': ['success']
        },
        {
            'name': 'Fallback Chart Data',
            'url': f'{base_url}/api/betting/charts-data?timeframe=1_week&chart_type=profit_timeline&test_mode=live',
            'expected_keys': ['success']
        }
    ]
    
    print(f"Testing {len(endpoints)} endpoints...")
    print()
    
    success_count = 0
    warning_count = 0
    error_count = 0
    
    for i, endpoint in enumerate(endpoints, 1):
        print(f"{i:2d}. Testing {endpoint['name']}...")
        
        try:
            response = requests.get(endpoint['url'], timeout=10)
            print(f"    Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Check expected keys
                    missing_keys = []
                    for key in endpoint['expected_keys']:
                        if key not in data:
                            missing_keys.append(key)
                    
                    if missing_keys:
                        print(f"    ⚠️ Missing keys: {', '.join(missing_keys)}")
                        warning_count += 1
                    else:
                        print(f"    ✅ Success - All expected keys present")
                        success_count += 1
                    
                    # Additional checks based on endpoint type
                    if 'statistics' in endpoint['url'] and data.get('success'):
                        stats = data.get('statistics', {})
                        basic_metrics = stats.get('basic_metrics', {})
                        print(f"    📊 Total bets: {basic_metrics.get('total_bets', 0)}")
                        print(f"    📊 Win rate: {basic_metrics.get('win_rate', 0)}%")
                        print(f"    📊 Data quality: {stats.get('data_quality', {}).get('data_completeness', 'unknown')}")
                    
                    elif 'charts-data' in endpoint['url'] and data.get('success'):
                        chart_data = data.get('data', {})
                        labels = chart_data.get('labels', [])
                        datasets = chart_data.get('datasets', [])
                        print(f"    📈 Chart labels: {len(labels)} items")
                        print(f"    📈 Chart datasets: {len(datasets)} series")
                    
                    elif 'health' in endpoint['url'] and data.get('success'):
                        status = data.get('overall_status', 'unknown')
                        components = data.get('components', {})
                        print(f"    🏥 Overall status: {status}")
                        print(f"    🏥 Components: {len(components)} checked")
                    
                except json.JSONDecodeError:
                    print(f"    ❌ Invalid JSON response")
                    error_count += 1
                except Exception as e:
                    print(f"    ⚠️ Data parsing error: {e}")
                    warning_count += 1
            
            elif response.status_code == 404:
                print(f"    ❌ Endpoint not found (404)")
                error_count += 1
            
            elif response.status_code == 500:
                print(f"    ❌ Server error (500)")
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        print(f"    ❌ Error details: {error_data['error']}")
                except:
                    pass
                error_count += 1
            
            else:
                print(f"    ⚠️ Unexpected status code: {response.status_code}")
                warning_count += 1
        
        except requests.exceptions.ConnectionError:
            print(f"    ❌ Connection failed - Server may not be running")
            error_count += 1
        
        except requests.exceptions.Timeout:
            print(f"    ❌ Request timeout")
            error_count += 1
        
        except Exception as e:
            print(f"    ❌ Unexpected error: {e}")
            error_count += 1
        
        print()
    
    # Summary
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Successful: {success_count}")
    print(f"⚠️ Warnings:   {warning_count}")
    print(f"❌ Errors:     {error_count}")
    print(f"📊 Total:      {len(endpoints)}")
    print()
    
    success_rate = (success_count / len(endpoints)) * 100
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("🎉 EXCELLENT - Dashboard integration is working perfectly!")
        return True
    elif success_rate >= 70:
        print("✅ GOOD - Dashboard integration is mostly working")
        return True
    elif success_rate >= 50:
        print("⚠️ FAIR - Dashboard integration has some issues")
        return False
    else:
        print("❌ POOR - Dashboard integration needs attention")
        return False

def test_javascript_compatibility():
    """Test JavaScript compatibility with API responses"""
    print()
    print("🔧 JAVASCRIPT COMPATIBILITY TEST")
    print("=" * 60)
    
    base_url = "http://65.109.135.2:5001"
    
    try:
        # Test main statistics endpoint
        response = requests.get(f'{base_url}/api/betting-dashboard/statistics?timeframe=1_week')
        if response.status_code == 200:
            data = response.json()
            
            if data.get('success'):
                stats = data.get('statistics', {})
                
                # Check JavaScript expected structure
                js_expectations = [
                    ('basic_metrics', dict),
                    ('financial_metrics', dict),
                    ('risk_metrics', dict),
                    ('data_quality', dict),
                    ('basic_metrics.total_bets', (int, type(None))),
                    ('basic_metrics.win_rate', (int, float, type(None))),
                    ('financial_metrics.net_profit', (int, float, type(None))),
                    ('financial_metrics.roi_percentage', (int, float, type(None))),
                    ('data_quality.sample_size', (int, type(None))),
                    ('data_quality.data_completeness', (str, type(None)))
                ]
                
                compatibility_issues = []
                
                for path, expected_types in js_expectations:
                    try:
                        value = stats
                        for key in path.split('.'):
                            value = value[key]
                        
                        if not isinstance(value, expected_types):
                            compatibility_issues.append(f"{path}: expected {expected_types}, got {type(value)}")
                    
                    except KeyError:
                        compatibility_issues.append(f"{path}: missing key")
                    except Exception as e:
                        compatibility_issues.append(f"{path}: error accessing - {e}")
                
                if compatibility_issues:
                    print("⚠️ JavaScript compatibility issues found:")
                    for issue in compatibility_issues:
                        print(f"    - {issue}")
                    return False
                else:
                    print("✅ JavaScript structure compatibility: PASSED")
                    return True
            else:
                print("❌ Statistics endpoint returned failure")
                return False
        else:
            print(f"❌ Statistics endpoint returned status {response.status_code}")
            return False
    
    except Exception as e:
        print(f"❌ JavaScript compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting comprehensive betting dashboard integration test...")
    print()
    
    # Run API integration tests
    api_success = test_betting_dashboard_integration()
    
    # Run JavaScript compatibility tests
    js_success = test_javascript_compatibility()
    
    print()
    print("FINAL RESULT")
    print("=" * 60)
    
    if api_success and js_success:
        print("🎉 ALL TESTS PASSED - Betting dashboard integration is ready!")
        exit_code = 0
    elif api_success or js_success:
        print("⚠️ PARTIAL SUCCESS - Some issues need to be addressed")
        exit_code = 1
    else:
        print("❌ INTEGRATION FAILED - Significant issues found")
        exit_code = 2
    
    print()
    print("Next steps:")
    print("1. Navigate to http://65.109.135.2:5001/ in your browser")
    print("2. Click on the 'Betting Statistics' tab")
    print("3. Select different time periods to test functionality")
    print("4. Verify that all metrics, charts, and data display correctly")
    
    sys.exit(exit_code)