#!/usr/bin/env python3
"""
Send a test notification using the tennis prediction notification system
"""

import asyncio
import os
from dotenv import load_dotenv
from telegram_notification_system import TelegramNotificationSystem

# Load environment variables
load_dotenv()

async def send_test_notification():
    """Send a test tennis prediction notification"""
    
    # Initialize the notification system
    notification_system = TelegramNotificationSystem()
    
    print("🎾 Tennis Prediction Notification Test")
    print("=" * 50)
    
    # Check if system is properly configured
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_ids_str = os.getenv('TELEGRAM_CHAT_IDS', '')
    
    print(f"Bot token configured: {'✅' if bot_token else '❌'}")
    print(f"Chat IDs configured: {chat_ids_str}")
    
    if not bot_token:
        print("❌ TELEGRAM_BOT_TOKEN not found in environment")
        return
    
    # Create a realistic test prediction notification
    test_prediction = {
        'player_1': 'Rafael Nadal',
        'player_2': 'Novak Djokovic', 
        'player_1_second_set_prob': 0.35,  # Nadal underdog
        'player_2_second_set_prob': 0.65,  # Djokovic favorite
        'tournament': 'ATP Masters Miami Open',
        'surface': 'Hard',
        'confidence_level': 'High',
        'strategic_insights': [
            'Djokovic has won 4 of last 5 matches on hard courts',
            'Nadal struggling with second set consistency recently',
            'Historical H2H favors Djokovic on hard courts'
        ],
        'success': True
    }
    
    print("\n📊 Test Prediction Details:")
    print(f"   Match: {test_prediction['player_1']} vs {test_prediction['player_2']}")
    print(f"   Tournament: {test_prediction['tournament']}")
    print(f"   Underdog: {test_prediction['player_1']} ({test_prediction['player_1_second_set_prob']*100:.1f}%)")
    print(f"   Favorite: {test_prediction['player_2']} ({test_prediction['player_2_second_set_prob']*100:.1f}%)")
    print(f"   Confidence: {test_prediction['confidence_level']}")
    
    # Send the notification
    print(f"\n📤 Attempting to send notification...")
    
    try:
        result = await notification_system.send_underdog_notification(test_prediction)
        
        if result:
            print("✅ Test notification sent successfully!")
            print("   Check your Telegram for the message")
        else:
            print("❌ Failed to send test notification")
            
        # Get and display system stats
        stats = notification_system.get_stats()
        print(f"\n📈 System Statistics:")
        print(f"   Notifications this hour: {stats.get('notifications_last_hour', 0)}")
        print(f"   Rate limit remaining: {10 - stats.get('notifications_last_hour', 0)}")
        
    except Exception as e:
        print(f"❌ Error sending notification: {e}")

if __name__ == "__main__":
    asyncio.run(send_test_notification())