#!/usr/bin/env python3
"""
Quick test of Telegram integration without full data collection
"""

import os
from dotenv import load_dotenv
from telegram_notification_system import TelegramNotificationSystem, TelegramConfig
from datetime import datetime

def test_telegram_with_manual_prediction():
    """Test Telegram notification with a manually created prediction"""
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Set up configuration
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '').strip()
    chat_ids_str = os.getenv('TELEGRAM_CHAT_IDS', '').strip()
    
    chat_ids = []
    if chat_ids_str:
        chat_ids = [chat_id.strip() for chat_id in chat_ids_str.split(',') if chat_id.strip()]
    
    config = TelegramConfig(
        bot_token=bot_token,
        chat_ids=chat_ids,
        enabled=True,
        min_probability=0.55
    )
    
    if not config.bot_token or not config.chat_ids:
        print("❌ Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_IDS environment variables or in .env file")
        print("  Option 1: export TELEGRAM_BOT_TOKEN='your_bot_token_here'")
        print("  Option 2: Add TELEGRAM_BOT_TOKEN=your_bot_token_here to .env file")
        print("Use 'python get_chat_id.py' to find your chat ID")
        return False
    
    print("🤖 Testing Telegram Notification System")
    print("=" * 45)
    
    # Initialize system
    telegram_system = TelegramNotificationSystem(config)
    print(f"✅ System initialized: {telegram_system.config.enabled}")
    
    # Create a high-quality underdog prediction (should trigger notification)
    strong_prediction = {
        'success': True,
        'underdog_second_set_probability': 0.68,  # 68% chance
        'confidence': 'High',
        'underdog_player': 'player1',
        'match_context': {
            'player1': 'A. Rublev',
            'player2': 'S. Tsitsipas', 
            'player1_rank': 175,
            'player2_rank': 8,
            'tournament': 'ATP 500 Dubai Open',
            'surface': 'Hard'
        },
        'strategic_insights': [
            '🔥 Strong underdog opportunity - high second set win probability',
            '📊 Large ranking gap (167 positions) creates significant upset potential',
            '🎾 Hard court surface favors aggressive baseline play',
            '🏆 High-pressure tournament environment may favor mental toughness'
        ],
        'prediction_metadata': {
            'service_type': 'comprehensive_ml_service',
            'prediction_time': datetime.now().isoformat(),
            'models_used': ['random_forest', 'gradient_boosting', 'xgboost', 'logistic_regression'],
            'processing_duration_ms': 1250,
            'data_sources': ['odds_api', 'tennis_explorer', 'rapidapi']
        }
    }
    
    print(f"\n🎯 Testing Strong Prediction Notification:")
    print(f"   Match: {strong_prediction['match_context']['player1']} vs {strong_prediction['match_context']['player2']}")
    print(f"   Underdog probability: {strong_prediction['underdog_second_set_probability']:.1%}")
    print(f"   Confidence: {strong_prediction['confidence']}")
    
    # Test notification decision
    should_notify = telegram_system.should_notify(strong_prediction)
    print(f"   Should notify: {should_notify}")
    
    if should_notify:
        print(f"\n📤 Sending notification...")
        success = telegram_system.send_notification_sync(strong_prediction)
        
        if success:
            print(f"✅ Notification sent successfully!")
            print(f"Check your Telegram for the underdog alert message.")
        else:
            print(f"❌ Failed to send notification")
    else:
        print(f"\n📱 Notification criteria not met")
    
    # Test a weak prediction (should NOT notify)
    print(f"\n\n🔍 Testing Weak Prediction (should NOT notify):")
    
    weak_prediction = {
        'success': True,
        'underdog_second_set_probability': 0.42,  # 42% chance - below threshold
        'confidence': 'Low',
        'underdog_player': 'player2',
        'match_context': {
            'player1': 'R. Nadal',
            'player2': 'C. Alcaraz',
            'tournament': 'ATP 250 Barcelona'
        }
    }
    
    should_notify_weak = telegram_system.should_notify(weak_prediction)
    print(f"   Weak prediction (42% prob): {'✅ Correctly filtered' if not should_notify_weak else '❌ Should not notify'}")
    
    # Show notification stats
    stats = telegram_system.get_notification_stats()
    print(f"\n📊 Notification Statistics:")
    print(f"   Enabled: {stats['enabled']}")
    print(f"   Notifications sent last hour: {stats['notifications_last_hour']}")
    print(f"   Rate limit remaining: {stats['rate_limit_remaining']}")
    
    return True

if __name__ == "__main__":
    success = test_telegram_with_manual_prediction()
    if success:
        print(f"\n🎉 Telegram integration test completed successfully!")
        print(f"\nYour tennis prediction system is now configured to automatically")
        print(f"send Telegram notifications when strong underdog opportunities are found!")
    else:
        print(f"\n❌ Test failed - please check configuration")