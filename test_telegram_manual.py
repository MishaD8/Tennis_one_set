#!/usr/bin/env python3
"""
Manual test for Telegram notification with actual API call
"""

import asyncio
import sys
import os
sys.path.append('src/utils')
from telegram_notification_system import TelegramNotificationSystem

async def test_telegram_manual():
    """Test telegram with manual notification"""
    
    # Create a strong prediction that should trigger notification
    strong_prediction = {
        'success': True,
        'underdog_second_set_probability': 0.72,  # Above 0.55 threshold
        'confidence': 'High',  # Required confidence level  
        'underdog_player': 'J. Ostapenko',
        'match_context': {
            'player1': 'J. Ostapenko',
            'player2': 'M. Stakusic',
            'tournament': 'WTA Guadalajara',
            'surface': 'Hard',
            'player1_rank': 45,
            'player2_rank': 180
        },
        'strategic_insights': ['Strong underdog potential based on recent form']
    }
    
    telegram_system = TelegramNotificationSystem()
    
    print("🎾 Telegram Manual Test")
    print("=" * 40)
    print(f"System enabled: {telegram_system.config.enabled}")
    print(f"Min probability: {telegram_system.config.min_probability}")
    print(f"Chat IDs: {telegram_system.config.chat_ids}")
    print(f"Should notify: {telegram_system.should_notify(strong_prediction)}")
    
    if telegram_system.should_notify(strong_prediction):
        print("\n📤 Attempting to send notification...")
        result = await telegram_system.send_underdog_notification(strong_prediction)
        print(f"Result: {'✅ Success' if result else '❌ Failed'}")
    else:
        print("\n❌ Prediction doesn't meet notification criteria")
        print("   - Check if probability > 0.55")
        print("   - Check if confidence is 'High' or 'Medium'")
        print("   - Check if system is enabled")

if __name__ == "__main__":
    asyncio.run(test_telegram_manual())#!/usr/bin/env python3
"""
Manual test for Telegram notification with actual API call
"""

import asyncio
import sys
import os
sys.path.append('src/utils')
from telegram_notification_system import TelegramNotificationSystem

async def test_telegram_manual():
    """Test telegram with manual notification"""
    
    # Create a strong prediction that should trigger notification
    strong_prediction = {
        'success': True,
        'underdog_second_set_probability': 0.72,  # Above 0.55 threshold
        'confidence': 'High',  # Required confidence level  
        'underdog_player': 'J. Ostapenko',
        'match_context': {
            'player1': 'J. Ostapenko',
            'player2': 'M. Stakusic',
            'tournament': 'WTA Guadalajara',
            'surface': 'Hard',
            'player1_rank': 45,
            'player2_rank': 180
        },
        'strategic_insights': ['Strong underdog potential based on recent form']
    }
    
    telegram_system = TelegramNotificationSystem()
    
    print("🎾 Telegram Manual Test")
    print("=" * 40)
    print(f"System enabled: {telegram_system.config.enabled}")
    print(f"Min probability: {telegram_system.config.min_probability}")
    print(f"Chat IDs: {telegram_system.config.chat_ids}")
    print(f"Should notify: {telegram_system.should_notify(strong_prediction)}")
    
    if telegram_system.should_notify(strong_prediction):
        print("\n📤 Attempting to send notification...")
        result = await telegram_system.send_underdog_notification(strong_prediction)
        print(f"Result: {'✅ Success' if result else '❌ Failed'}")
    else:
        print("\n❌ Prediction doesn't meet notification criteria")
        print("   - Check if probability > 0.55")
        print("   - Check if confidence is 'High' or 'Medium'")
        print("   - Check if system is enabled")

if __name__ == "__main__":
    asyncio.run(test_telegram_manual())