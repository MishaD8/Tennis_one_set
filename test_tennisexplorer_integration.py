#!/usr/bin/env python3
"""
🧪 Test TennisExplorer Integration
Tests the complete integration with your tennis prediction system
"""

import os
import sys
from tennisexplorer_integration import TennisExplorerIntegration

def test_integration():
    """Test the complete TennisExplorer integration"""
    print("🧪 TennisExplorer Integration Test")
    print("=" * 50)
    
    # Check environment variables
    print("🔍 Checking environment configuration...")
    username = os.getenv('TENNISEXPLORER_USERNAME')
    password = os.getenv('TENNISEXPLORER_PASSWORD')
    
    if username and password:
        print(f"✅ Credentials found for user: {username}")
    else:
        print("⚠️ No credentials found in environment variables")
        print("Please add TENNISEXPLORER_USERNAME and TENNISEXPLORER_PASSWORD to .env file")
    
    # Initialize integration
    print("\n🔧 Initializing integration...")
    integration = TennisExplorerIntegration()
    
    if integration.initialize():
        print("✅ Integration initialized successfully")
    else:
        print("❌ Integration initialization failed")
        return False
    
    # Test enhanced match data
    print("\n📊 Testing enhanced match data collection...")
    matches = integration.get_enhanced_match_data(days_ahead=3)
    
    print(f"Found {len(matches)} enhanced matches")
    
    if matches:
        print("\n🎾 Sample Enhanced Matches:")
        for i, match in enumerate(matches[:3]):
            print(f"\n{i+1}. {match['player1']} vs {match['player2']}")
            print(f"   Tournament: {match['tournament']}")
            print(f"   Surface: {match.get('surface_confirmed', match['surface'])}")
            print(f"   Data Source: {match.get('data_source', 'Unknown')}")
            print(f"   Data Quality: {match.get('data_quality', 'Not assessed')}")
            
            if match.get('odds_player1'):
                print(f"   Odds: {match['odds_player1']} - {match['odds_player2']}")
    else:
        print("⚠️ No matches found - this might be normal during off-season")
    
    # Test prediction-ready matches
    print(f"\n🎯 Testing prediction-ready match filtering...")
    pred_matches = integration.get_matches_for_prediction()
    print(f"Found {len(pred_matches)} matches ready for prediction")
    
    if pred_matches:
        print("\nPrediction-ready matches:")
        for match in pred_matches[:2]:
            print(f"- {match['player1']} vs {match['player2']} (Quality: {match.get('data_quality', 'Unknown')})")
    
    # Test data saving
    print(f"\n💾 Testing data persistence...")
    if integration.save_enhanced_data('test_enhanced_data.json'):
        print("✅ Data saved successfully")
        
        # Check file exists
        if os.path.exists('test_enhanced_data.json'):
            file_size = os.path.getsize('test_enhanced_data.json')
            print(f"📁 File size: {file_size} bytes")
        else:
            print("❌ File was not created")
    else:
        print("❌ Data saving failed")
    
    print("\n🎉 Integration test completed!")
    return True

def test_scraper_connection():
    """Test basic scraper functionality"""
    print("\n🌐 Testing TennisExplorer Connection...")
    
    from tennisexplorer_scraper import TennisExplorerScraper
    
    scraper = TennisExplorerScraper()
    
    # Test connection
    if scraper.test_connection():
        print("✅ Connection successful")
    else:
        print("❌ Connection failed")
        return False
    
    # Test login
    if scraper.login():
        print("✅ Login successful")
    else:
        print("⚠️ Login failed (credentials might be incorrect or site structure changed)")
    
    return True

if __name__ == "__main__":
    print("🚀 Starting TennisExplorer Integration Tests")
    print("=" * 60)
    
    # Test basic connection first
    if not test_scraper_connection():
        print("❌ Basic connection test failed")
        sys.exit(1)
    
    # Test full integration
    if test_integration():
        print("\n✅ All tests completed successfully!")
        print("\n📋 Next Steps:")
        print("1. Add your TennisExplorer credentials to .env file if not already done")
        print("2. Test with: python test_tennisexplorer_integration.py") 
        print("3. Use integration.get_enhanced_match_data() in your prediction system")
        print("4. The scraper respects rate limits and includes error handling")
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)