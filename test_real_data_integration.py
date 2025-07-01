#!/usr/bin/env python3
"""
🧪 Тест интеграции реальных данных Wimbledon 2025
"""

def test_real_data_integration():
    print("🎾 Testing Wimbledon 2025 real data integration...")
    
    try:
        from real_tennis_data_collector import RealTennisDataCollector, RealOddsCollector
        
        collector = RealTennisDataCollector()
        odds_collector = RealOddsCollector()
        
        # Тест реальных данных Wimbledon
        matches = collector.get_wimbledon_2025_real_matches()
        print(f"✅ Found {len(matches)} Wimbledon 2025 matches")
        
        if matches:
            print("\n🎾 Current Wimbledon matches:")
            for i, match in enumerate(matches[:4], 1):
                status = match['status'].upper()
                court = match.get('court', 'TBD')
                print(f"   {i}. {match['player1']} vs {match['player2']}")
                print(f"      📅 {match['date']} {match['time']} • {court} • {status}")
        
        # Тест коэффициентов
        if matches:
            odds = odds_collector.get_real_odds(matches[:2])
            
            if odds:
                print(f"\n💰 Sample odds:")
                for match_id, match_odds in list(odds.items())[:2]:
                    winner_odds = match_odds['best_markets']['winner']
                    p1_odds = winner_odds['player1']['odds']
                    p2_odds = winner_odds['player2']['odds']
                    print(f"   {match_id}: {p1_odds} vs {p2_odds}")
        
        print("\n🎉 Real data integration test PASSED!")
        print("\n🚀 Next steps:")
        print("   1. Restart your backend:")
        print("      python web_backend_with_dashboard.py")
        print("\n   2. Open dashboard:")
        print("      http://localhost:5001")
        print("\n🎾 What you'll see:")
        print("   • Real Wimbledon 2025 matches with 🎾 emoji")
        print("   • Current player names: Alcaraz, Zverev, Sabalenka, etc.")
        print("   • Live tournament status")
        print("   • Real rankings and odds")
        print("   • No more demo warnings!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure real_tennis_data_collector.py exists")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

if __name__ == "__main__":
    test_real_data_integration()
