#!/usr/bin/env python3
"""
🧪 ТЕСТ реальных данных
"""

def test_real_data():
    try:
        print("🎾 Testing real tennis data collection...")
        
        from real_tennis_data_collector import RealTennisDataCollector, RealOddsCollector
        
        collector = RealTennisDataCollector()
        
        # Тест Wimbledon данных
        wimbledon = collector.get_wimbledon_2025_schedule()
        print(f"✅ Wimbledon: {len(wimbledon)} matches found")
        
        if wimbledon:
            for i, match in enumerate(wimbledon[:2]):  # Показываем первые 2
                print(f"   {i+1}. {match['player1']} vs {match['player2']}")
                print(f"      📅 {match['date']} {match['time']} • {match['tournament']}")
        
        # Тест коэффициентов
        odds_collector = RealOddsCollector()
        if wimbledon:
            odds = odds_collector.get_real_odds(wimbledon[:1])
            
            if odds:
                match_id = list(odds.keys())[0]
                match_odds = odds[match_id]['best_markets']['winner']
                print(f"💰 Sample odds: {match_odds['player1']['odds']} vs {match_odds['player2']['odds']}")
        
        print("\n🎉 Real data test PASSED!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure real_tennis_data_collector.py exists!")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

if __name__ == "__main__":
    test_real_data()
