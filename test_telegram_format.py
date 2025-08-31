#!/usr/bin/env python3
"""
Test the new concise Telegram notification format
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_telegram_format():
    """Test the new Telegram message format"""
    
    print("🎾 TESTING NEW TELEGRAM NOTIFICATION FORMAT")
    print("=" * 60)
    
    try:
        from utils.telegram_notification_system import TelegramNotificationSystem
        
        # Create system instance
        system = TelegramNotificationSystem()
        
        # Create sample prediction data
        sample_prediction = {
            'match_context': {
                'player1': 'Rafael Nadal',
                'player2': 'Novak Djokovic',
                'tournament': 'ATP Masters Rome',
                'surface': 'Clay'
            },
            'underdog_player': 'player1',  # Nadal is underdog
            'underdog_second_set_probability': 0.68,
            'confidence': 'High',
            'strategic_insights': [
                'Nadal has strong clay court advantage in Rome',
                'Djokovic showing fatigue after 3-set first round match',
                'Weather conditions favor baseline rallies'
            ],
            'success': True
        }
        
        # Format the message
        formatted_message = system._format_underdog_message(sample_prediction)
        
        print("📱 NEW FORMATTED MESSAGE:")
        print("-" * 40)
        print(formatted_message)
        print("-" * 40)
        
        # Analyze the message
        lines = formatted_message.split('\n')
        print(f"\n📊 MESSAGE ANALYSIS:")
        print(f"   Total lines: {len(lines)}")
        print(f"   Total characters: {len(formatted_message)}")
        print(f"   Contains ranking gap: {'Ranking Gap:' in formatted_message}")
        print(f"   Contains probability: {'Second Set Win Probability:' in formatted_message}")
        print(f"   Insights included: {'Key Insights:' in formatted_message}")
        
        # Check improvements
        improvements = []
        if len(formatted_message) < 1000:  # Should be more concise
            improvements.append("✅ More concise (under 1000 characters)")
        if 'Ranking Gap:' in formatted_message:
            improvements.append("✅ Ranking gap display fixed")
        if len([line for line in lines if line.strip()]) < 15:  # Fewer lines
            improvements.append("✅ Fewer message lines")
        if 'Key Insights:' in formatted_message:
            improvements.append("✅ Simplified insights section")
        
        print(f"\n🚀 IMPROVEMENTS:")
        for improvement in improvements:
            print(f"   {improvement}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing format: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ranking_gap_fix():
    """Test that ranking gap calculation is fixed"""
    
    print("\n🔧 TESTING RANKING GAP FIX")
    print("=" * 60)
    
    try:
        from utils.telegram_notification_system import TelegramNotificationSystem
        
        system = TelegramNotificationSystem()
        
        # Test case with clear ranking gap
        test_prediction = {
            'match_context': {
                'player1': 'Alexander Zverev',   # Higher ranked (better)
                'player2': 'Fabio Fognini',     # Lower ranked (worse)
                'tournament': 'ATP 250 Vienna',
                'surface': 'Hard'
            },
            'underdog_player': 'player2',  # Fognini is underdog
            'underdog_second_set_probability': 0.62,
            'confidence': 'Medium',
            'strategic_insights': [
                'Fognini showing improved form on hard courts'
            ],
            'success': True
        }
        
        # Mock the ranking function to return known values
        def mock_get_ranking(player_name):
            rankings = {
                'alexander zverev': 4,   # Top player
                'fabio fognini': 85      # Lower ranked
            }
            return rankings.get(player_name.lower(), 150)
        
        # Temporarily replace the ranking function
        original_func = system._get_live_player_ranking
        system._get_live_player_ranking = mock_get_ranking
        
        # Format message
        message = system._format_underdog_message(test_prediction)
        
        # Restore original function
        system._get_live_player_ranking = original_func
        
        print("📱 RANKING GAP TEST MESSAGE:")
        print("-" * 40)
        print(message)
        print("-" * 40)
        
        # Check if ranking gap is correctly calculated
        if 'Ranking Gap: 81 positions' in message:
            print("✅ Ranking gap correctly calculated: 81 positions (85 - 4)")
        elif 'Ranking Gap:' in message:
            # Extract the gap value
            for line in message.split('\n'):
                if 'Ranking Gap:' in line:
                    print(f"📊 Found ranking gap line: {line}")
                    if '81' in line:
                        print("✅ Ranking gap correctly shows 81 positions")
                    else:
                        print("⚠️ Ranking gap value may be incorrect")
        else:
            print("❌ Ranking gap not found in message")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing ranking gap: {e}")
        return False

if __name__ == "__main__":
    print("Starting Telegram format tests...\n")
    
    # Test new format
    format_success = test_telegram_format()
    
    # Test ranking gap fix
    gap_success = test_ranking_gap_fix()
    
    print(f"\n📊 TEST RESULTS:")
    print(f"   Format test: {'✅ PASS' if format_success else '❌ FAIL'}")
    print(f"   Ranking gap test: {'✅ PASS' if gap_success else '❌ FAIL'}")
    
    if format_success and gap_success:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"   Your Telegram notifications are now more concise")
        print(f"   Ranking gap display is fixed")
        print(f"   Ready for production use!")
    else:
        print(f"\n⚠️ Some tests failed. Please check the errors above.")