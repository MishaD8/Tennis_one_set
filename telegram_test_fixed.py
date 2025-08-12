#!/usr/bin/env python3
"""
Fixed Telegram test using correct methods
"""

import asyncio
import os
from dotenv import load_dotenv
from telegram_notification_system import TelegramNotificationSystem

# Load environment variables
load_dotenv()

async def test_telegram_system():
    """Test the Telegram notification system"""
    
    print("🎾 Telegram Notification System Test")
    print("=" * 50)
    
    # Initialize the notification system
    system = TelegramNotificationSystem()
    
    # Check system status
    print(f"✅ System enabled: {system.config.enabled}")
    print(f"📱 Bot token configured: {'✅' if system.config.bot_token else '❌'}")
    print(f"💬 Chat IDs: {system.config.chat_ids}")
    print(f"📊 Min probability threshold: {system.config.min_probability}")
    
    # Get current stats
    stats = system.get_notification_stats()
    print(f"📈 Notifications this hour: {stats.get('notifications_last_hour', 0)}")
    print(f"🔄 Rate limit remaining: {10 - stats.get('notifications_last_hour', 0)}")
    
    if not system.config.enabled:
        print("❌ System is disabled - cannot send test")
        return
    
    # Test 1: Send simple test message
    print(f"\n📤 Test 1: Sending simple test message...")
    try:
        result = await system.send_test_message()
        print(f"   Result: {'✅ SUCCESS' if result else '❌ FAILED'}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Send underdog notification
    print(f"\n📤 Test 2: Sending underdog notification...")
    
    test_prediction = {
        'player_1': 'Carlos Alcaraz',
        'player_2': 'Novak Djokovic',
        'player_1_second_set_prob': 0.42,  # Alcaraz as underdog
        'player_2_second_set_prob': 0.58,  # Djokovic as favorite  
        'tournament': 'Australian Open',
        'surface': 'Hard',
        'confidence_level': 'High',
        'strategic_insights': [
            'Djokovic has strong second set recovery rate',
            'Alcaraz tends to fade in longer matches',
            'Hard court favors Djokovic\'s style'
        ],
        'success': True
    }
    
    print(f"   Match: {test_prediction['player_1']} vs {test_prediction['player_2']}")
    print(f"   Underdog prob: {test_prediction['player_1_second_set_prob']*100:.1f}%")
    
    # Check if it should notify
    should_notify = system.should_notify(test_prediction)
    print(f"   Should notify: {should_notify}")
    
    if should_notify:
        try:
            result = await system.send_underdog_notification(test_prediction)
            print(f"   Result: {'✅ SUCCESS' if result else '❌ FAILED'}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    else:
        print(f"   ⏭️ Skipped - doesn't meet notification criteria")
    
    # Test 3: High probability underdog (should notify)
    print(f"\n📤 Test 3: High probability underdog (should notify)...")
    
    strong_underdog = {
        'player_1': 'Stefanos Tsitsipas',
        'player_2': 'Rafael Nadal',
        'player_1_second_set_prob': 0.67,  # Strong underdog
        'player_2_second_set_prob': 0.33,  # Clear favorite
        'tournament': 'French Open',
        'surface': 'Clay',
        'confidence_level': 'High',
        'strategic_insights': [
            'Tsitsipas has improved clay court game significantly',
            'Nadal showing fatigue in recent matches',
            'Weather conditions favor aggressive play'
        ],
        'success': True
    }
    
    print(f"   Match: {strong_underdog['player_1']} vs {strong_underdog['player_2']}")
    print(f"   Underdog prob: {strong_underdog['player_1_second_set_prob']*100:.1f}%")
    
    should_notify = system.should_notify(strong_underdog)
    print(f"   Should notify: {should_notify}")
    
    if should_notify:
        try:
            result = await system.send_underdog_notification(strong_underdog)
            print(f"   Result: {'✅ SUCCESS' if result else '❌ FAILED'}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Final stats
    final_stats = system.get_notification_stats()
    print(f"\n📊 Final Statistics:")
    print(f"   Notifications sent this hour: {final_stats.get('notifications_last_hour', 0)}")
    print(f"   Rate limit remaining: {10 - final_stats.get('notifications_last_hour', 0)}")
    
    print(f"\n🎉 Telegram test completed!")

if __name__ == "__main__":
    asyncio.run(test_telegram_system())