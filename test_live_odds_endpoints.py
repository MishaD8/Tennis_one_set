#!/usr/bin/env python3
"""
Live Odds Endpoints Testing Script
Tests the actual API Tennis odds endpoints with real API calls
Run this script when you have a valid API_TENNIS_KEY configured
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api_tennis_integration import APITennisClient, get_api_tennis_client
from api_tennis_data_collector import APITennisDataCollector

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_api_key_configuration():
    """Check if API key is properly configured"""
    api_key = os.getenv('API_TENNIS_KEY')
    if not api_key:
        print("❌ API_TENNIS_KEY environment variable not configured")
        print("Please set your API key with: export API_TENNIS_KEY='your_key_here'")
        return False
    
    print(f"✅ API_TENNIS_KEY configured (length: {len(api_key)} characters)")
    return True

def test_basic_odds_endpoints():
    """Test basic odds endpoints without specific match IDs"""
    print("\n" + "=" * 60)
    print("TESTING BASIC ODDS ENDPOINTS")
    print("=" * 60)
    
    try:
        client = APITennisClient()
        
        # Test general get_odds call
        print("🎯 Testing get_odds() - General call...")
        try:
            odds_response = client.get_odds()
            print(f"✅ get_odds() successful")
            print(f"   Response type: {type(odds_response)}")
            print(f"   Response keys: {list(odds_response.keys()) if isinstance(odds_response, dict) else 'Not a dict'}")
            
            if isinstance(odds_response, dict):
                if odds_response.get('success') == 1:
                    result = odds_response.get('result', [])
                    print(f"   Number of odds records: {len(result) if isinstance(result, list) else 'Not a list'}")
                    
                    if isinstance(result, list) and result:
                        sample_record = result[0]
                        print(f"   Sample record keys: {list(sample_record.keys())}")
                        print(f"   Sample record: {json.dumps(sample_record, indent=2)[:200]}...")
                else:
                    print(f"   API Error: {odds_response.get('error', 'Unknown error')}")
            
        except Exception as e:
            print(f"❌ get_odds() failed: {e}")
        
        # Test general get_live_odds call
        print("\n🔴 Testing get_live_odds() - General call...")
        try:
            live_odds_response = client.get_live_odds()
            print(f"✅ get_live_odds() successful")
            print(f"   Response type: {type(live_odds_response)}")
            print(f"   Response keys: {list(live_odds_response.keys()) if isinstance(live_odds_response, dict) else 'Not a dict'}")
            
            if isinstance(live_odds_response, dict):
                if live_odds_response.get('success') == 1:
                    result = live_odds_response.get('result', [])
                    print(f"   Number of live odds records: {len(result) if isinstance(result, list) else 'Not a list'}")
                    
                    if isinstance(result, list) and result:
                        sample_record = result[0]
                        print(f"   Sample live record keys: {list(sample_record.keys())}")
                        print(f"   Sample live record: {json.dumps(sample_record, indent=2)[:200]}...")
                else:
                    print(f"   API Error: {live_odds_response.get('error', 'Unknown error')}")
            
        except Exception as e:
            print(f"❌ get_live_odds() failed: {e}")
            
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")

def test_match_specific_odds():
    """Test odds endpoints with specific match IDs"""
    print("\n" + "=" * 60)
    print("TESTING MATCH-SPECIFIC ODDS")
    print("=" * 60)
    
    try:
        client = APITennisClient()
        
        # Get today's matches first
        print("📅 Getting today's matches to test odds...")
        today_matches = client.get_today_matches()
        print(f"Found {len(today_matches)} matches today")
        
        if not today_matches:
            # Try upcoming matches if no matches today
            print("📅 No matches today, trying upcoming matches...")
            upcoming_matches = client.get_upcoming_matches(days_ahead=3)
            print(f"Found {len(upcoming_matches)} upcoming matches")
            today_matches = upcoming_matches[:5]  # Test first 5
        
        if today_matches:
            print(f"\n🎾 Testing odds for {min(3, len(today_matches))} specific matches:")
            
            for i, match in enumerate(today_matches[:3], 1):
                if not match.id:
                    print(f"  {i}. Skipping match without ID")
                    continue
                
                player1_name = match.player1.name if match.player1 else "Unknown"
                player2_name = match.player2.name if match.player2 else "Unknown"
                print(f"\n  {i}. Testing: {player1_name} vs {player2_name} (ID: {match.id})")
                
                # Test specific match odds
                try:
                    match_odds = client.get_odds(fixture_id=match.id)
                    print(f"     ✅ get_odds(fixture_id={match.id}) successful")
                    
                    if isinstance(match_odds, dict) and match_odds.get('result'):
                        result = match_odds['result']
                        if isinstance(result, list):
                            print(f"     📊 Found odds from {len(result)} sources")
                            for j, odds_data in enumerate(result[:2], 1):  # Show first 2 bookmakers
                                bookmaker = odds_data.get('bookmaker', 'Unknown')
                                home_odds = odds_data.get('home_odds', odds_data.get('player1_odds'))
                                away_odds = odds_data.get('away_odds', odds_data.get('player2_odds'))
                                print(f"       {j}. {bookmaker}: {home_odds} vs {away_odds}")
                        else:
                            print(f"     📊 Odds result: {match_odds}")
                    else:
                        print(f"     ⚠️ No odds data in response: {match_odds}")
                        
                except Exception as e:
                    print(f"     ❌ get_odds() failed: {e}")
                
                # Test specific match live odds
                try:
                    live_match_odds = client.get_live_odds(fixture_id=match.id)
                    print(f"     ✅ get_live_odds(fixture_id={match.id}) successful")
                    
                    if isinstance(live_match_odds, dict) and live_match_odds.get('result'):
                        result = live_match_odds['result']
                        if isinstance(result, list) and result:
                            print(f"     🔴 Found live odds from {len(result)} sources")
                        else:
                            print(f"     🔴 Live odds result: {live_match_odds}")
                    else:
                        print(f"     ⚠️ No live odds data: {live_match_odds}")
                        
                except Exception as e:
                    print(f"     ❌ get_live_odds() failed: {e}")
        else:
            print("❌ No matches available to test odds")
            
    except Exception as e:
        print(f"❌ Failed to test match-specific odds: {e}")

def test_data_collector_odds_integration():
    """Test the data collector odds integration"""
    print("\n" + "=" * 60)
    print("TESTING DATA COLLECTOR ODDS INTEGRATION")
    print("=" * 60)
    
    try:
        collector = APITennisDataCollector()
        
        if not collector.is_available():
            print("❌ Data collector not available (API key missing)")
            return
        
        print("✅ Data collector initialized successfully")
        
        # Get current matches
        print("\n📊 Getting current matches...")
        current_matches = collector.get_current_matches()
        print(f"Found {len(current_matches)} current matches")
        
        if current_matches:
            # Test odds integration for first few matches
            print(f"\n🎯 Testing odds integration for {min(3, len(current_matches))} matches:")
            
            for i, match in enumerate(current_matches[:3], 1):
                match_id = match.get('id', '').replace('api_tennis_', '')
                if not match_id.isdigit():
                    print(f"  {i}. Skipping match with invalid ID: {match.get('id', 'None')}")
                    continue
                
                print(f"\n  {i}. Testing odds for: {match['player1']} vs {match['player2']}")
                print(f"     Match ID: {match_id}")
                
                try:
                    odds_data = collector.get_match_odds(int(match_id))
                    print(f"     ✅ get_match_odds() successful")
                    print(f"     📊 Odds data structure: {list(odds_data.keys())}")
                    
                    if odds_data.get('bookmakers'):
                        print(f"     📈 Found {len(odds_data['bookmakers'])} bookmakers")
                        for bookmaker in odds_data['bookmakers'][:2]:  # Show first 2
                            name = bookmaker.get('bookmaker', 'Unknown')
                            p1_odds = bookmaker.get('player1_odds', 'N/A')
                            p2_odds = bookmaker.get('player2_odds', 'N/A')
                            print(f"       - {name}: {p1_odds} vs {p2_odds}")
                    else:
                        print(f"     ⚠️ No bookmaker data found")
                        
                except Exception as e:
                    print(f"     ❌ get_match_odds() failed: {e}")
        else:
            print("❌ No current matches available for odds testing")
            
    except Exception as e:
        print(f"❌ Failed to test data collector odds: {e}")

def analyze_odds_data_quality():
    """Analyze the quality and structure of odds data"""
    print("\n" + "=" * 60)
    print("ODDS DATA QUALITY ANALYSIS")
    print("=" * 60)
    
    try:
        client = APITennisClient()
        
        # Get general odds data for analysis
        print("📊 Analyzing general odds data structure...")
        odds_response = client.get_odds()
        
        if isinstance(odds_response, dict) and odds_response.get('success') == 1:
            result = odds_response.get('result', [])
            
            if isinstance(result, list) and result:
                print(f"✅ Found {len(result)} odds records for analysis")
                
                # Analyze data structure
                sample_record = result[0]
                print(f"\n📋 Odds Record Structure:")
                for key, value in sample_record.items():
                    print(f"  {key}: {type(value).__name__} = {value}")
                
                # Check for common bookmakers
                bookmakers = set()
                fixture_ids = set()
                
                for record in result[:20]:  # Analyze first 20 records
                    if 'bookmaker' in record:
                        bookmakers.add(record['bookmaker'])
                    if 'fixture_id' in record:
                        fixture_ids.add(record['fixture_id'])
                
                print(f"\n📈 Data Coverage Analysis (first 20 records):")
                print(f"  Unique bookmakers: {len(bookmakers)}")
                if bookmakers:
                    print(f"  Bookmaker examples: {list(bookmakers)[:5]}")
                print(f"  Unique fixtures: {len(fixture_ids)}")
                
                # Check for odds format
                odds_fields = []
                for key in sample_record.keys():
                    if 'odds' in key.lower():
                        odds_fields.append(key)
                
                print(f"\n💰 Odds Fields Detected: {odds_fields}")
                
                # Check update timestamps
                timestamp_fields = []
                for key in sample_record.keys():
                    if any(time_word in key.lower() for time_word in ['time', 'date', 'updated', 'timestamp']):
                        timestamp_fields.append(key)
                
                print(f"⏰ Timestamp Fields: {timestamp_fields}")
                
            else:
                print("⚠️ No odds records found for analysis")
        else:
            print(f"❌ Failed to get odds data: {odds_response}")
            
    except Exception as e:
        print(f"❌ Failed to analyze odds data quality: {e}")

def generate_recommendations():
    """Generate recommendations based on test results"""
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS BASED ON TEST RESULTS")
    print("=" * 60)
    
    print("📋 Next Steps Based on Testing:")
    print("1. ✅ Verify odds endpoints are working with your API subscription")
    print("2. 📊 Analyze actual bookmaker coverage and data quality")
    print("3. 🔄 Implement automatic odds monitoring for value opportunities")
    print("4. 🎯 Integrate odds data with ML predictions for betting signals")
    print("5. ⚡ Add Betfair Exchange integration for live betting execution")
    
    print("\n🚀 Implementation Priority:")
    print("  🔥 Critical: Validate odds data availability and quality")
    print("  ⚠️  High: Create odds-ML integration for value detection")
    print("  💡 Medium: Add historical odds tracking for analysis")
    print("  🔧 Enhancement: Implement arbitrage detection algorithms")
    
    print("\n📈 Success Metrics:")
    print("  - Odds coverage: >80% of professional tennis matches")
    print("  - Update frequency: <5 minutes for live odds")
    print("  - Bookmaker count: >5 major bookmakers represented")
    print("  - Data accuracy: >95% accuracy vs. actual bookmaker sites")

def main():
    """Run all odds testing procedures"""
    print("🎾 API TENNIS LIVE ODDS ENDPOINTS TESTING")
    print("Testing odds functionality with real API calls...")
    
    # Check API key configuration
    if not check_api_key_configuration():
        print("\n❌ Cannot proceed without API key. Please configure API_TENNIS_KEY.")
        return
    
    try:
        # Run all tests
        test_basic_odds_endpoints()
        test_match_specific_odds()
        test_data_collector_odds_integration()
        analyze_odds_data_quality()
        generate_recommendations()
        
        print("\n" + "=" * 60)
        print("✅ ODDS TESTING COMPLETED")
        print("=" * 60)
        print("Review the results above to understand odds data availability")
        print("and quality for your API Tennis subscription.")
        
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during odds testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()