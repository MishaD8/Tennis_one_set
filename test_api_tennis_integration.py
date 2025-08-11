#!/usr/bin/env python3
"""
API-Tennis.com Integration Test Script
Tests the API-Tennis.com integration with the tennis backend system
"""

import os
import sys
import requests
import json
from datetime import datetime

def test_api_tennis_integration():
    """Test API-Tennis.com integration endpoints"""
    
    base_url = "http://localhost:5001"
    
    # Test endpoints
    endpoints = [
        "/api/api-tennis/status",
        "/api/api-tennis/test-connection", 
        "/api/api-tennis/tournaments",
        "/api/api-tennis/matches",
        "/api/api-tennis/enhanced"
    ]
    
    print("🎾 API-Tennis.com Integration Test")
    print("=" * 50)
    
    # Check if API key is configured
    api_key = os.getenv('API_TENNIS_KEY')
    if api_key:
        print(f"✅ API_TENNIS_KEY configured (length: {len(api_key)})")
    else:
        print("⚠️  API_TENNIS_KEY not configured - testing without key")
    
    print()
    
    for endpoint in endpoints:
        try:
            print(f"Testing: {endpoint}")
            response = requests.get(f"{base_url}{endpoint}", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    print(f"  ✅ Success")
                    
                    # Print specific information based on endpoint
                    if 'status' in endpoint:
                        status = data.get('api_tennis_status', {})
                        print(f"     Available: {status.get('available', False)}")
                        print(f"     API Key: {'✅' if status.get('api_key_configured') else '❌'}")
                    
                    elif 'connection' in endpoint:
                        if data.get('success'):
                            print(f"     Connection: ✅ OK")
                            print(f"     Event types: {data.get('event_types_count', 0)}")
                        else:
                            print(f"     Connection: ❌ {data.get('error', 'Unknown error')}")
                    
                    elif 'tournaments' in endpoint:
                        count = data.get('count', 0)
                        print(f"     Tournaments: {count}")
                    
                    elif 'matches' in endpoint:
                        count = data.get('count', 0)
                        source = data.get('data_source', 'Unknown')
                        print(f"     Matches: {count} from {source}")
                    
                else:
                    print(f"  ❌ Failed: {data.get('error', 'Unknown error')}")
                    
            else:
                print(f"  ❌ HTTP {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"  ❌ Connection failed - is the server running?")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        print()
    
    # Test with player name if API key is available
    if api_key:
        test_player = "Novak Djokovic"
        endpoint = f"/api/api-tennis/player/{test_player}/matches"
        
        try:
            print(f"Testing player search: {endpoint}")
            response = requests.get(f"{base_url}{endpoint}", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                count = data.get('count', 0)
                print(f"  ✅ Found {count} matches for {test_player}")
            else:
                print(f"  ❌ HTTP {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("=" * 50)
    print("Integration test completed")

def test_direct_api_client():
    """Test direct API-Tennis client without server"""
    print("\n🔧 Direct API Client Test")
    print("=" * 30)
    
    try:
        from api_tennis_integration import APITennisClient
        
        # Test with environment variable
        client = APITennisClient()
        status = client.get_client_status()
        
        print(f"API Key configured: {status['api_key_configured']}")
        print(f"Caching enabled: {status['caching_enabled']}")
        print(f"Rate limit: {status['rate_limit']} req/min")
        
        if status['api_key_configured']:
            print("\nTesting API connection...")
            try:
                event_types = client.get_event_types()
                if isinstance(event_types, dict) and event_types.get('success') == 1:
                    print(f"✅ API connection successful")
                    print(f"✅ Found {len(event_types.get('result', []))} event types")
                else:
                    print(f"❌ API connection failed: {event_types}")
            except Exception as e:
                print(f"❌ API test failed: {e}")
        else:
            print("⚠️  Cannot test API connection - no key configured")
            
    except Exception as e:
        print(f"❌ Direct client test failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "direct":
        test_direct_api_client()
    elif len(sys.argv) > 1 and sys.argv[1] == "both":
        test_direct_api_client()
        print()
        test_api_tennis_integration()
    else:
        test_api_tennis_integration()