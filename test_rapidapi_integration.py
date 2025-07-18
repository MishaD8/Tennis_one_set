#!/usr/bin/env python3
"""
Test RapidAPI Tennis Integration
Tests the integration without making actual API calls
"""

import os
import sys
from unittest.mock import patch, MagicMock
import json

def test_rapidapi_config():
    """Test if RapidAPI configuration is properly loaded"""
    print("=== Testing RapidAPI Configuration ===")
    
    try:
        from config_loader import load_secure_config
        config = load_secure_config()
        
        rapidapi_config = config.get('data_sources', {}).get('rapidapi_tennis', {})
        
        if not rapidapi_config:
            print("❌ RapidAPI configuration not found in config.json")
            return False
        
        print(f"✅ RapidAPI configuration found")
        print(f"   - Enabled: {rapidapi_config.get('enabled', False)}")
        print(f"   - Base URL: {rapidapi_config.get('base_url', 'Not set')}")
        print(f"   - Host: {rapidapi_config.get('host', 'Not set')}")
        print(f"   - Daily Limit: {rapidapi_config.get('daily_limit', 'Not set')}")
        print(f"   - Username: {rapidapi_config.get('username', 'Not set')}")
        
        endpoints = rapidapi_config.get('endpoints', {})
        if endpoints:
            print(f"   - Endpoints configured: {len(endpoints)}")
            for name, endpoint in endpoints.items():
                print(f"     * {name}: {endpoint}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing configuration: {e}")
        return False

def test_rapidapi_client_structure():
    """Test if RapidAPI client can be imported and initialized"""
    print("\n=== Testing RapidAPI Client Structure ===")
    
    try:
        # Mock environment variable for API key
        with patch.dict(os.environ, {'RAPIDAPI_KEY': 'test_key_123'}):
            from rapidapi_tennis_client import RapidAPITennisClient, RapidAPIRateLimiter
            
            print("✅ RapidAPI client imports successfully")
            
            # Test rate limiter
            rate_limiter = RapidAPIRateLimiter(daily_limit=50)
            print(f"✅ Rate limiter initialized with {rate_limiter.daily_limit} daily limit")
            print(f"   - Requests today: {rate_limiter.requests_today}")
            print(f"   - Remaining: {rate_limiter.get_remaining_requests()}")
            
            # Test client initialization (mocked)
            try:
                client = RapidAPITennisClient()
                print("✅ RapidAPI client can be initialized")
                
                # Test status method
                status = client.get_status()
                print(f"✅ Client status method works")
                print(f"   - Daily limit: {status['daily_limit']}")
                print(f"   - Requests remaining: {status['requests_remaining']}")
                print(f"   - Cache size: {status['cache_size']}")
                
                return True
                
            except Exception as e:
                print(f"❌ Client initialization failed: {e}")
                return False
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_rapidapi_methods():
    """Test if all RapidAPI methods are available"""
    print("\n=== Testing RapidAPI Methods ===")
    
    try:
        with patch.dict(os.environ, {'RAPIDAPI_KEY': 'test_key_123'}):
            from rapidapi_tennis_client import RapidAPITennisClient
            
            client = RapidAPITennisClient()
            
            # Check if all expected methods exist
            expected_methods = [
                'get_wta_rankings',
                'get_atp_rankings', 
                'get_live_matches',
                'get_player_details',
                'get_tournament_details',
                'get_status',
                'clear_cache'
            ]
            
            for method_name in expected_methods:
                if hasattr(client, method_name):
                    print(f"✅ Method {method_name} exists")
                else:
                    print(f"❌ Method {method_name} missing")
                    return False
            
            print("✅ All expected methods are available")
            return True
            
    except Exception as e:
        print(f"❌ Error testing methods: {e}")
        return False

def test_mock_api_calls():
    """Test API calls with mocked responses"""
    print("\n=== Testing Mocked API Calls ===")
    
    try:
        with patch.dict(os.environ, {'RAPIDAPI_KEY': 'test_key_123'}):
            from rapidapi_tennis_client import RapidAPITennisClient
            
            client = RapidAPITennisClient()
            
            # Mock requests.get
            mock_wta_response = {
                'rankings': [
                    {'rank': 1, 'player': {'name': 'Test Player 1'}},
                    {'rank': 2, 'player': {'name': 'Test Player 2'}}
                ]
            }
            
            with patch('requests.get') as mock_get:
                # Configure mock response
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = mock_wta_response
                mock_get.return_value = mock_response
                
                # Test WTA rankings
                wta_rankings = client.get_wta_rankings()
                if wta_rankings and len(wta_rankings) == 2:
                    print("✅ WTA rankings method works with mock data")
                else:
                    print("❌ WTA rankings method failed")
                    return False
                
                # Test that rate limiter recorded the request
                if client.rate_limiter.requests_today > 0:
                    print("✅ Rate limiter properly records requests")
                else:
                    print("❌ Rate limiter not recording requests")
                    return False
                
            print("✅ Mocked API calls work correctly")
            return True
            
    except Exception as e:
        print(f"❌ Error testing mocked calls: {e}")
        return False

def test_integration_with_existing_system():
    """Test integration with existing tennis system"""
    print("\n=== Testing Integration with Existing System ===")
    
    try:
        # Test getting RapidAPI key through config system
        with patch.dict(os.environ, {'RAPIDAPI_KEY': 'test_integration_key'}):
            # Force reload config
            from config_loader import SecureConfigLoader
            loader = SecureConfigLoader()
            config = loader.load_config()
            api_key = loader.get_api_key('rapidapi')
            
            if api_key == 'test_integration_key':
                print("✅ RapidAPI key retrieval through config system works")
            else:
                print(f"❌ Expected 'test_integration_key', got '{api_key}'")
                return False
        
        # Test that we can import existing modules without conflicts
        existing_modules = [
            'tennis_backend',
            'database_service',
            'enhanced_api_integration'
        ]
        
        for module_name in existing_modules:
            try:
                __import__(module_name)
                print(f"✅ Can import {module_name} alongside RapidAPI client")
            except ImportError:
                print(f"⚠️ Could not import {module_name} (may be expected)")
            except Exception as e:
                print(f"❌ Error importing {module_name}: {e}")
                return False
        
        print("✅ Integration with existing system looks good")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🎾 RapidAPI Tennis Integration Test Suite")
    print("=" * 50)
    
    tests = [
        test_rapidapi_config,
        test_rapidapi_client_structure,
        test_rapidapi_methods,
        test_mock_api_calls,
        test_integration_with_existing_system
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("🎾 Test Results Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! RapidAPI integration is ready.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)