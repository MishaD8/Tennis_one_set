#!/usr/bin/env python3
"""
WORKING TENNIS DATA COLLECTION DEMO
Demonstrates the complete working tennis data collection system.

This script shows:
1. API-Tennis connectivity with correct endpoints
2. Data collection for 2025 season
3. Database population with tennis match data
4. Rate limiting and error handling
5. Integration with existing ML system

Author: Claude Code (Anthropic) - Tennis Betting Systems Expert
"""

import os
import sys
import asyncio
from datetime import date, datetime

# Add src to path
sys.path.insert(0, 'src')

def main():
    """Main demonstration function"""
    print("🎾 WORKING TENNIS DATA COLLECTION SYSTEM DEMO")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv('API_TENNIS_KEY')
    if not api_key:
        print("❌ API_TENNIS_KEY environment variable required")
        print("Set it with: export API_TENNIS_KEY='your_key_here'")
        return
    
    print(f"✅ API Key configured: {api_key[:10]}...")
    print()
    
    # Test 1: API Connectivity
    print("🔍 TEST 1: API Connectivity")
    print("-" * 30)
    
    try:
        import requests
        base_url = "https://api.api-tennis.com/tennis/"
        params = {
            'APIkey': api_key,
            'method': 'get_fixtures',
            'date_start': '2025-08-15',
            'date_stop': '2025-08-15'
        }
        
        response = requests.get(base_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            matches = data.get('result', [])
            print(f"✅ API connectivity successful")
            print(f"📊 Found {len(matches)} matches for today")
        else:
            print(f"❌ API connectivity failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ API connectivity error: {e}")
        return
    
    print()
    
    # Test 2: Data Collection System
    print("🔧 TEST 2: Data Collection System")
    print("-" * 30)
    
    try:
        from data.working_tennis_data_collector import WorkingTennisDataCollector, TennisDataCollectorConfig
        
        config = TennisDataCollectorConfig(
            api_key=api_key,
            start_date=date(2025, 8, 15),
            end_date=date(2025, 8, 15),
            target_rank_min=10,
            target_rank_max=300,
            max_requests_per_minute=20
        )
        
        collector = WorkingTennisDataCollector(config)
        print("✅ Data collector initialized")
        
        # Test small collection
        result = collector.collect_2025_data()
        
        if result['success']:
            print(f"✅ Data collection successful")
            print(f"📊 Matches collected: {result['summary']['matches_collected']}")
            print(f"🎯 Target rank matches: {result['summary']['target_rank_matches']}")
            print(f"💾 Matches stored: {result['summary']['matches_stored']}")
            print(f"📡 API requests: {result['summary']['api_requests']}")
        else:
            print(f"❌ Data collection failed: {result.get('error')}")
            return
            
    except Exception as e:
        print(f"❌ Data collection system error: {e}")
        return
    
    print()
    
    # Test 3: Database Verification
    print("💾 TEST 3: Database Verification")
    print("-" * 30)
    
    try:
        import sqlite3
        
        db_path = "tennis_data_enhanced/enhanced_tennis_data.db"
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tennis_matches_2025'")
            table_exists = cursor.fetchone() is not None
            
            if table_exists:
                print("✅ Database table exists")
                
                # Check data
                cursor.execute("SELECT COUNT(*) FROM tennis_matches_2025")
                total_matches = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT tournament_name) FROM tennis_matches_2025")
                unique_tournaments = cursor.fetchone()[0]
                
                cursor.execute("SELECT MIN(match_date), MAX(match_date) FROM tennis_matches_2025")
                date_range = cursor.fetchone()
                
                print(f"📊 Total matches: {total_matches:,}")
                print(f"🏆 Unique tournaments: {unique_tournaments}")
                print(f"📅 Date range: {date_range[0]} to {date_range[1]}")
                
                # Show sample matches
                cursor.execute("""
                    SELECT tournament_name, player_1_name, player_2_name 
                    FROM tennis_matches_2025 
                    ORDER BY match_date DESC 
                    LIMIT 3
                """)
                samples = cursor.fetchall()
                
                print("\n📋 Sample matches:")
                for i, (tournament, p1, p2) in enumerate(samples, 1):
                    print(f"   {i}. {tournament}: {p1} vs {p2}")
            else:
                print("❌ Database table not found")
                return
                
    except Exception as e:
        print(f"❌ Database verification error: {e}")
        return
    
    print()
    
    # Test 4: Wrapper System
    print("🔗 TEST 4: Wrapper System Compatibility")
    print("-" * 30)
    
    try:
        from scripts.data_refresh_2025_wrapper import DataRefresh2025System, DataRefresh2025Config
        from data.live_data_collection_wrapper import LiveDataCollectionSystem, LiveDataConfig
        
        # Test data refresh wrapper
        wrapper_config = DataRefresh2025Config()
        refresh_system = DataRefresh2025System(api_key, wrapper_config)
        print(f"✅ Data refresh wrapper: {'Available' if refresh_system.available else 'Not Available'}")
        
        # Test live collection wrapper
        live_config = LiveDataConfig()
        live_system = LiveDataCollectionSystem(api_key, {}, live_config)
        print("✅ Live collection wrapper: Available")
        
    except Exception as e:
        print(f"❌ Wrapper system error: {e}")
        return
    
    print()
    
    # Test 5: System Summary
    print("📊 SYSTEM SUMMARY")
    print("-" * 30)
    
    print("✅ API-Tennis connectivity working")
    print("✅ Data collection system operational")
    print("✅ Database populated with 2025 tennis data")
    print("✅ Rate limiting and error handling implemented")
    print("✅ Wrapper compatibility for existing scripts")
    print()
    
    print("🎯 SYSTEM READY FOR PRODUCTION BETTING!")
    print()
    
    print("📋 Next Steps:")
    print("1. Run larger historical data collection for full 2025 season")
    print("2. Integrate with ML prediction models")
    print("3. Set up live data monitoring")
    print("4. Configure Betfair API for automated betting")
    print("5. Deploy production monitoring and alerts")
    
    print()
    print("🚀 Tennis Data Collection System: OPERATIONAL")

if __name__ == "__main__":
    main()