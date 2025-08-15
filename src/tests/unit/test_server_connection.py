#!/usr/bin/env python3
"""
Test script to verify the server is working correctly after TLS/HTTPS fixes
"""

import requests
import json
import time
import sys
import warnings

# Suppress SSL warnings for testing
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

def test_endpoint(url, description, expected_status=200):
    """Test a specific endpoint and return results"""
    try:
        print(f"Testing {description}...")
        response = requests.get(url, timeout=10, verify=False)
        
        if response.status_code == expected_status:
            print(f"✅ {description}: OK (Status: {response.status_code})")
            return True, response
        else:
            print(f"❌ {description}: Failed (Status: {response.status_code})")
            return False, response
            
    except requests.exceptions.ConnectionError as e:
        print(f"❌ {description}: Connection Error - {str(e)}")
        return False, None
    except requests.exceptions.Timeout:
        print(f"❌ {description}: Timeout")
        return False, None
    except Exception as e:
        print(f"❌ {description}: Unexpected error - {str(e)}")
        return False, None

def main():
    """Run comprehensive server tests"""
    print("🎾 TENNIS BACKEND CONNECTION TESTS")
    print("=" * 50)
    
    base_urls = [
        "http://localhost:5001",
        "http://127.0.0.1:5001", 
        "http://0.0.0.0:5001"
    ]
    
    results = {}
    
    for base_url in base_urls:
        print(f"\n🔍 Testing base URL: {base_url}")
        print("-" * 40)
        
        url_results = []
        
        # Test main dashboard
        success, response = test_endpoint(f"{base_url}/", "Dashboard Home")
        url_results.append(("Dashboard", success))
        
        # Test health endpoint
        success, response = test_endpoint(f"{base_url}/api/health", "Health Check")
        url_results.append(("Health", success))
        if success and response:
            try:
                health_data = response.json()
                print(f"   Components: {health_data.get('components', {})}")
            except:
                pass
        
        # Test stats endpoint
        success, response = test_endpoint(f"{base_url}/api/stats", "Statistics")
        url_results.append(("Stats", success))
        
        # Test matches endpoint
        success, response = test_endpoint(f"{base_url}/api/matches", "Matches")
        url_results.append(("Matches", success))
        if success and response:
            try:
                matches_data = response.json()
                match_count = matches_data.get('count', 0)
                print(f"   Matches found: {match_count}")
            except:
                pass
        
        results[base_url] = url_results
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    for base_url, url_results in results.items():
        print(f"\n{base_url}:")
        for test_name, success in url_results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {test_name}: {status}")
    
    # Overall status
    all_tests = [result for url_results in results.values() for _, result in url_results]
    passed_tests = sum(all_tests)
    total_tests = len(all_tests)
    
    print(f"\n🎯 OVERALL: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("✅ All tests passed! The server is working correctly.")
        return 0
    elif passed_tests > 0:
        print("⚠️ Some tests passed. The server is partially working.")
        return 1
    else:
        print("❌ All tests failed. The server may not be running or accessible.")
        return 2

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)