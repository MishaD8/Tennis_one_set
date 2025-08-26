#!/usr/bin/env python3
"""
🎾 Dashboard Betting Statistics Integration Verification
Final verification that all betting statistics functionality works correctly
"""

import requests
import json
import sys

def verify_dashboard_integration():
    """Verify the betting statistics integration is working properly"""
    print("🎾 DASHBOARD BETTING STATISTICS VERIFICATION")
    print("=" * 60)
    
    base_url = "http://65.109.135.2:5001"
    
    # Test the main API endpoints used by the JavaScript
    tests = [
        {
            'name': 'Main Dashboard Page',
            'url': f'{base_url}/',
            'type': 'html',
            'expected_content': ['Betting Statistics', 'dashboard.html']
        },
        {
            'name': 'Betting Statistics API (1 week)',
            'url': f'{base_url}/api/betting/statistics?timeframe=1_week&test_mode=live',
            'type': 'json',
            'required_keys': ['success', 'statistics']
        },
        {
            'name': 'Betting Statistics API (1 month)', 
            'url': f'{base_url}/api/betting/statistics?timeframe=1_month&test_mode=live',
            'type': 'json',
            'required_keys': ['success', 'statistics']
        },
        {
            'name': 'Chart Data - Profit Timeline',
            'url': f'{base_url}/api/betting/charts-data?timeframe=1_week&chart_type=profit_timeline&test_mode=live',
            'type': 'json',
            'required_keys': ['success', 'data']
        },
        {
            'name': 'Chart Data - Win Rate Trend',
            'url': f'{base_url}/api/betting/charts-data?timeframe=1_week&chart_type=win_rate_trend&test_mode=live',
            'type': 'json',
            'required_keys': ['success', 'data']
        },
        {
            'name': 'Chart Data - Odds Distribution',
            'url': f'{base_url}/api/betting/charts-data?timeframe=1_week&chart_type=odds_distribution&test_mode=live',
            'type': 'json',
            'required_keys': ['success', 'data']
        },
        {
            'name': 'Chart Data - Monthly Performance',
            'url': f'{base_url}/api/betting/charts-data?timeframe=1_month&chart_type=monthly_performance&test_mode=live',
            'type': 'json',
            'required_keys': ['success', 'data']
        },
        {
            'name': 'JavaScript File',
            'url': f'{base_url}/static/js/betting-statistics.js',
            'type': 'javascript',
            'expected_content': ['BettingStatistics', 'betting/statistics', 'betting/charts-data']
        }
    ]
    
    print(f"Running {len(tests)} verification tests...\n")
    
    results = {'passed': 0, 'failed': 0, 'details': []}
    
    for test in tests:
        print(f"Testing: {test['name']}")
        
        try:
            response = requests.get(test['url'], timeout=15)
            status = response.status_code
            
            if status == 200:
                if test['type'] == 'json':
                    try:
                        data = response.json()
                        
                        # Check required keys
                        missing_keys = [key for key in test['required_keys'] if key not in data]
                        
                        if missing_keys:
                            results['failed'] += 1
                            results['details'].append(f"❌ {test['name']}: Missing keys {missing_keys}")
                            print(f"  ❌ FAIL - Missing keys: {missing_keys}")
                        else:
                            results['passed'] += 1
                            results['details'].append(f"✅ {test['name']}: All required keys present")
                            print("  ✅ PASS - Structure valid")
                            
                            # Show key metrics for statistics endpoints
                            if 'statistics' in data and data.get('success'):
                                stats = data['statistics']
                                if 'basic_metrics' in stats:
                                    bm = stats['basic_metrics']
                                    print(f"    📊 Total bets: {bm.get('total_bets', 0)}")
                                    print(f"    📊 Win rate: {bm.get('win_rate', 0)}%")
                                    
                                if 'financial_metrics' in stats:
                                    fm = stats['financial_metrics']
                                    print(f"    💰 Net profit: ${fm.get('net_profit', 0)}")
                                    print(f"    💰 ROI: {fm.get('roi_percentage', 0)}%")
                                    
                                if 'data_quality' in stats:
                                    dq = stats['data_quality']
                                    print(f"    📈 Data quality: {dq.get('data_completeness', 'unknown')}")
                                    print(f"    📈 Sample size: {dq.get('sample_size', 0)} bets")
                            
                            # Show chart info
                            elif 'data' in data and data.get('success'):
                                chart_data = data['data']
                                labels = chart_data.get('labels', [])
                                datasets = chart_data.get('datasets', [])
                                print(f"    📊 Chart data: {len(labels)} labels, {len(datasets)} datasets")
                                
                    except json.JSONDecodeError:
                        results['failed'] += 1
                        results['details'].append(f"❌ {test['name']}: Invalid JSON response")
                        print("  ❌ FAIL - Invalid JSON")
                        
                elif test['type'] == 'html':
                    content = response.text
                    missing_content = [item for item in test['expected_content'] if item not in content]
                    
                    if missing_content:
                        results['failed'] += 1
                        results['details'].append(f"❌ {test['name']}: Missing content {missing_content}")
                        print(f"  ❌ FAIL - Missing content: {missing_content}")
                    else:
                        results['passed'] += 1
                        results['details'].append(f"✅ {test['name']}: All expected content found")
                        print("  ✅ PASS - Content valid")
                        
                elif test['type'] == 'javascript':
                    content = response.text
                    missing_content = [item for item in test['expected_content'] if item not in content]
                    
                    if missing_content:
                        results['failed'] += 1
                        results['details'].append(f"❌ {test['name']}: Missing JS content {missing_content}")
                        print(f"  ❌ FAIL - Missing JS content: {missing_content}")
                    else:
                        results['passed'] += 1
                        results['details'].append(f"✅ {test['name']}: JavaScript properly configured")
                        print("  ✅ PASS - JavaScript valid")
            else:
                results['failed'] += 1
                results['details'].append(f"❌ {test['name']}: HTTP {status}")
                print(f"  ❌ FAIL - HTTP {status}")
                
        except requests.exceptions.ConnectionError:
            results['failed'] += 1
            results['details'].append(f"❌ {test['name']}: Connection failed")
            print("  ❌ FAIL - Connection failed")
            
        except requests.exceptions.Timeout:
            results['failed'] += 1
            results['details'].append(f"❌ {test['name']}: Timeout")
            print("  ❌ FAIL - Timeout")
            
        except Exception as e:
            results['failed'] += 1
            results['details'].append(f"❌ {test['name']}: {str(e)}")
            print(f"  ❌ FAIL - {str(e)}")
            
        print()
    
    # Results summary
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    total_tests = results['passed'] + results['failed']
    success_rate = (results['passed'] / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"✅ Passed: {results['passed']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"📊 Success Rate: {success_rate:.1f}%")
    print()
    
    # Detailed results
    print("DETAILED RESULTS:")
    for detail in results['details']:
        print(f"  {detail}")
    print()
    
    # Final assessment
    if success_rate >= 90:
        print("🎉 EXCELLENT - Betting statistics integration is fully functional!")
        print("The dashboard should display all metrics, charts, and data correctly.")
        return True
    elif success_rate >= 70:
        print("✅ GOOD - Betting statistics integration is mostly working.")
        print("Minor issues present but core functionality should work.")
        return True
    elif success_rate >= 50:
        print("⚠️ FAIR - Betting statistics integration has some issues.")
        print("Some features may not work correctly.")
        return False
    else:
        print("❌ POOR - Betting statistics integration needs significant work.")
        print("Major functionality issues detected.")
        return False

def print_usage_instructions():
    """Print instructions for using the betting statistics dashboard"""
    print("\n" + "=" * 60)
    print("HOW TO USE THE BETTING STATISTICS DASHBOARD")
    print("=" * 60)
    
    instructions = [
        "1. Open your web browser and navigate to: http://65.109.135.2:5001/",
        "2. You should see the main tennis dashboard with multiple tabs",
        "3. Click on the 'Betting Statistics' tab (📈 Betting Statistics)",
        "4. The tab will load and display:",
        "   • Key performance metrics (Total Bets, Win Rate, Net Profit, ROI, etc.)",
        "   • Interactive charts showing profit/loss timeline and trends",
        "   • Time period selector (1 Week, 1 Month, 1 Year, All Time)",
        "   • Risk analysis data (Sharpe ratio, largest win/loss, streaks)",
        "   • Data quality indicators",
        "",
        "5. Test the functionality:",
        "   • Click different time period buttons to filter data",
        "   • Verify that metrics update when changing time periods",
        "   • Check that charts display data appropriately",
        "   • Look for the data quality indicator at the bottom",
        "",
        "6. Expected behavior:",
        "   • Metrics should display actual values (not '-' or 'Loading...')",
        "   • Charts should show real data or 'No Data' placeholders",
        "   • Time period changes should trigger new API requests",
        "   • System should handle errors gracefully",
        "",
        "7. If you see issues:",
        "   • Check browser console (F12) for JavaScript errors",
        "   • Verify network requests are reaching the API endpoints",
        "   • Confirm betting data exists in the system",
        "",
        "The integration provides comprehensive betting analytics including:",
        "• Financial performance tracking with ROI calculations",
        "• Risk management metrics and drawdown analysis", 
        "• Model performance evaluation and confidence breakdowns",
        "• Interactive data visualization with multiple chart types",
        "• Automated data quality assessment and recommendations"
    ]
    
    for instruction in instructions:
        print(instruction)
    
    print("=" * 60)

if __name__ == "__main__":
    print("Starting betting statistics dashboard verification...\n")
    
    success = verify_dashboard_integration()
    
    print_usage_instructions()
    
    if success:
        print("\n🎉 VERIFICATION COMPLETE - Integration is ready for use!")
        sys.exit(0)
    else:
        print("\n⚠️ VERIFICATION COMPLETE - Some issues found, check details above")
        sys.exit(1)