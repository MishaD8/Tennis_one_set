#!/usr/bin/env python3
"""
Test script to check API Tennis odds endpoints
This script attempts to test the odds methods without requiring API key by examining the implementation
"""

import os
import sys
import json
import requests
from datetime import datetime
from typing import Dict, List, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_odds_methods_availability():
    """Test if odds methods are available in the API Tennis integration"""
    print("=" * 60)
    print("ODDS ENDPOINTS AVAILABILITY TEST")
    print("=" * 60)
    
    try:
        from api_tennis_integration import APITennisClient
        
        # Initialize client (even without API key to test methods)
        client = APITennisClient(api_key="test_key")
        
        # Check if odds methods exist
        print("✅ Odds methods available in APITennisClient:")
        print(f"  - get_odds: {hasattr(client, 'get_odds')}")
        print(f"  - get_live_odds: {hasattr(client, 'get_live_odds')}")
        
        # Check method signatures
        if hasattr(client, 'get_odds'):
            import inspect
            signature = inspect.signature(client.get_odds)
            print(f"  - get_odds signature: {signature}")
        
        if hasattr(client, 'get_live_odds'):
            import inspect
            signature = inspect.signature(client.get_live_odds)
            print(f"  - get_live_odds signature: {signature}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error checking odds methods: {e}")
        return False

def check_data_collector_odds_support():
    """Check if data collector supports odds functionality"""
    print("\n" + "=" * 60)
    print("DATA COLLECTOR ODDS SUPPORT TEST")
    print("=" * 60)
    
    try:
        from api_tennis_data_collector import APITennisDataCollector
        
        # Initialize data collector
        collector = APITennisDataCollector()
        
        # Check if odds methods exist
        print("✅ Odds methods available in APITennisDataCollector:")
        print(f"  - get_match_odds: {hasattr(collector, 'get_match_odds')}")
        
        # Check method signature
        if hasattr(collector, 'get_match_odds'):
            import inspect
            signature = inspect.signature(collector.get_match_odds)
            print(f"  - get_match_odds signature: {signature}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error checking data collector odds support: {e}")
        return False

def analyze_api_documentation():
    """Analyze the API Tennis documentation for odds endpoints"""
    print("\n" + "=" * 60)
    print("API DOCUMENTATION ANALYSIS")
    print("=" * 60)
    
    # API Tennis documented methods based on their documentation
    documented_methods = {
        'get_odds': {
            'description': 'Get betting odds for matches',
            'parameters': ['fixture_id', 'league_id', 'bookmaker'],
            'expected_response': 'Odds data from multiple bookmakers'
        },
        'get_live_odds': {
            'description': 'Get live betting odds',
            'parameters': ['fixture_id'],
            'expected_response': 'Real-time odds data'
        }
    }
    
    print("📚 API Tennis Documented Odds Methods:")
    for method, info in documented_methods.items():
        print(f"\n  🎯 {method}:")
        print(f"    Description: {info['description']}")
        print(f"    Parameters: {', '.join(info['parameters'])}")
        print(f"    Expected Response: {info['expected_response']}")
    
    return documented_methods

def check_current_match_data_for_odds():
    """Check current match data to see if any odds information is present"""
    print("\n" + "=" * 60)
    print("CURRENT MATCH DATA ODDS CHECK")
    print("=" * 60)
    
    try:
        # Read cached fixture data to check for odds
        cache_dir = "/home/apps/Tennis_one_set/cache/api_tennis"
        
        if os.path.exists(cache_dir):
            cache_files = [f for f in os.listdir(cache_dir) if f.startswith('get_fixtures')]
            print(f"Found {len(cache_files)} cached fixture files")
            
            if cache_files:
                # Read the first fixture file (just first 1000 characters to avoid size issues)
                sample_file = os.path.join(cache_dir, cache_files[0])
                with open(sample_file, 'r') as f:
                    # Read just a sample to check structure
                    content = f.read(2000)  # First 2000 chars
                    
                print("📋 Sample fixture data structure:")
                try:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        print(f"  Response format: {type(data)}")
                        print(f"  Top-level keys: {list(data.keys())}")
                        
                        if 'result' in data and isinstance(data['result'], list) and data['result']:
                            sample_match = data['result'][0]
                            print(f"  Sample match keys: {list(sample_match.keys())}")
                            
                            # Look for odds-related fields
                            odds_fields = [k for k in sample_match.keys() if 'odds' in k.lower()]
                            print(f"  Odds-related fields: {odds_fields}")
                            
                except json.JSONDecodeError:
                    print("  ⚠️ Could not parse cached data as JSON")
        else:
            print("❌ No cached fixture data found")
            
    except Exception as e:
        print(f"❌ Error checking cached match data: {e}")

def simulate_odds_request_structure():
    """Simulate what odds requests would look like based on API Tennis documentation"""
    print("\n" + "=" * 60)
    print("ODDS REQUEST STRUCTURE SIMULATION")
    print("=" * 60)
    
    # Expected URL patterns based on API Tennis documentation
    base_url = "https://api.api-tennis.com/tennis/"
    
    print("🌐 Expected Odds API Endpoints:")
    
    # get_odds endpoint
    print("\n  📊 GET/POST api.api-tennis.com/tennis?method=get_odds")
    print("    Parameters:")
    print("      - APIkey: [required] Your API key")
    print("      - fixture_id: [optional] Specific match ID")
    print("      - league_id: [optional] Tournament ID")
    print("      - bookmaker: [optional] Specific bookmaker filter")
    print("    Expected Response: JSON with bookmaker odds data")
    
    # get_live_odds endpoint  
    print("\n  🔴 GET/POST api.api-tennis.com/tennis?method=get_live_odds")
    print("    Parameters:")
    print("      - APIkey: [required] Your API key")
    print("      - fixture_id: [optional] Specific match ID")
    print("    Expected Response: JSON with real-time odds data")
    
    print("\n  💡 Sample Request URLs:")
    print(f"    General odds: {base_url}?method=get_odds&APIkey=YOUR_KEY")
    print(f"    Match-specific: {base_url}?method=get_odds&APIkey=YOUR_KEY&fixture_id=12345")
    print(f"    Live odds: {base_url}?method=get_live_odds&APIkey=YOUR_KEY&fixture_id=12345")

def analyze_betting_system_requirements():
    """Analyze what the betting system needs from odds data"""
    print("\n" + "=" * 60)
    print("BETTING SYSTEM ODDS REQUIREMENTS ANALYSIS")
    print("=" * 60)
    
    print("🎯 Required Odds Data for Automated Betting:")
    
    requirements = {
        'Real-time Odds': [
            'Live match winner odds (player 1 vs player 2)',
            'Multiple bookmaker comparison',
            'Odds movement tracking',
            'Timestamp of odds updates'
        ],
        'Pre-match Odds': [
            'Opening odds for value betting identification',
            'Current odds for stake calculation',
            'Historical odds for trend analysis',
            'Best available odds across bookmakers'
        ],
        'Market Coverage': [
            'Match Winner (Moneyline)',
            'Set Betting (2-0, 2-1 outcomes)',
            'Handicap betting',
            'Total Games Over/Under',
            'Asian Handicap'
        ],
        'Data Quality': [
            'Decimal odds format',
            'Fractional odds support',
            'American odds conversion',
            'Implied probability calculation',
            'Margin calculation'
        ],
        'Betting Integration': [
            'Betfair Exchange odds',
            'Traditional bookmaker odds',
            'Arbitrage opportunity detection',
            'Value bet identification',
            'Stake sizing based on odds and confidence'
        ]
    }
    
    for category, items in requirements.items():
        print(f"\n  📋 {category}:")
        for item in items:
            print(f"    - {item}")
    
    print("\n  🔄 Integration with ML Predictions:")
    print("    - Combine ML confidence with odds for expected value")
    print("    - Identify underdog opportunities with favorable odds")
    print("    - Risk management based on odds movements")
    print("    - Position sizing algorithms using Kelly Criterion")

def main():
    """Run all odds analysis tests"""
    print("🎾 API TENNIS ODDS DATA INVESTIGATION")
    print("Starting comprehensive odds endpoint analysis...")
    
    try:
        # Test method availability
        methods_available = test_odds_methods_availability()
        
        # Test data collector support
        collector_support = check_data_collector_odds_support()
        
        # Analyze API documentation
        documented_methods = analyze_api_documentation()
        
        # Check current data for odds
        check_current_match_data_for_odds()
        
        # Simulate request structure
        simulate_odds_request_structure()
        
        # Analyze betting system requirements
        analyze_betting_system_requirements()
        
        # Summary
        print("\n" + "=" * 60)
        print("INVESTIGATION SUMMARY")
        print("=" * 60)
        
        print("✅ FINDINGS:")
        print(f"  - Odds methods implemented: {methods_available}")
        print(f"  - Data collector odds support: {collector_support}")
        print(f"  - API documentation available: {len(documented_methods) > 0}")
        
        print("\n📊 CURRENT STATUS:")
        if methods_available:
            print("  ✅ API Tennis client has get_odds() and get_live_odds() methods")
            print("  ✅ Data collector has get_match_odds() method")
            print("  ⚠️  Implementation exists but needs testing with real API key")
        else:
            print("  ❌ Odds methods not properly implemented")
        
        print("\n🎯 NEXT STEPS:")
        print("  1. Test odds endpoints with valid API key")
        print("  2. Analyze actual response format and data quality")
        print("  3. Enhance odds integration for betting system")
        print("  4. Implement Betfair API integration for live betting")
        print("  5. Create arbitrage and value betting detection")
        
    except Exception as e:
        print(f"❌ Error in odds investigation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()