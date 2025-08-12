#!/usr/bin/env python3
"""
🤖 Telegram Bot Setup and Testing Script

This script helps set up and test the Telegram notification system
for tennis underdog predictions.

Usage:
    python telegram_setup.py --test-config
    python telegram_setup.py --send-test
    python telegram_setup.py --check-bot-info

Author: Claude Code (Anthropic)
"""

import os
import asyncio
import argparse
import json
from dotenv import load_dotenv
from telegram_notification_system import TelegramNotificationSystem, TelegramConfig, send_test_notification
from datetime import datetime

def print_setup_instructions():
    """Print setup instructions for Telegram bot"""
    
    print("🤖 TELEGRAM BOT SETUP INSTRUCTIONS")
    print("=" * 50)
    print()
    print("1. Create a Telegram Bot:")
    print("   • Open Telegram and search for @BotFather")
    print("   • Send /newbot command")
    print("   • Follow prompts to create your bot")
    print("   • Copy the bot token")
    print()
    print("2. Get your Chat ID:")
    print("   • Start a chat with your bot")
    print("   • Send any message to your bot")
    print("   • Visit: https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates")
    print("   • Find 'chat':{'id': XXXXXXX} in the response")
    print()
    print("3. Set Environment Variables:")
    print("   export TELEGRAM_BOT_TOKEN=")
    print("   export TELEGRAM_CHAT_IDS=")
    print("   export TELEGRAM_NOTIFICATIONS_ENABLED='true'")
    print("   export TELEGRAM_MIN_PROBABILITY='0.55'")
    print()
    print("4. Test the setup:")
    print("   python telegram_setup.py --test-config")
    print("   python telegram_setup.py --send-test")
    print()

def test_configuration():
    """Test Telegram configuration"""
    
    print("🧪 TESTING TELEGRAM CONFIGURATION")
    print("=" * 40)
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check environment variables
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
    chat_ids = os.getenv('TELEGRAM_CHAT_IDS', '')
    enabled = os.getenv('TELEGRAM_NOTIFICATIONS_ENABLED', 'true').lower() == 'true'
    
    try:
        min_prob = float(os.getenv('TELEGRAM_MIN_PROBABILITY', '0.55'))
    except ValueError:
        print("⚠️  Invalid TELEGRAM_MIN_PROBABILITY value, using default 0.55")
        min_prob = 0.55
    
    print(f"📊 Configuration Check:")
    print(f"  Bot Token: {'✅ Set' if bot_token else '❌ Missing'}")
    print(f"  Chat IDs: {'✅ Set' if chat_ids else '❌ Missing'}")
    print(f"  Enabled: {enabled}")
    print(f"  Min Probability: {min_prob:.1%}")
    
    if not bot_token or not chat_ids:
        print("\n❌ Configuration incomplete!")
        print("Please set the required environment variables:")
        print("  Option 1: Export environment variables:")
        print("    export TELEGRAM_BOT_TOKEN='your_bot_token_here'")
        print("    export TELEGRAM_CHAT_IDS='your_chat_id_here'")
        print("  Option 2: Add to .env file:")
        print("    TELEGRAM_BOT_TOKEN=your_bot_token_here")
        print("    TELEGRAM_CHAT_IDS=your_chat_id_here")
        print("\nUse 'python get_chat_id.py' to find your chat ID.")
        return False
    
    # Test system initialization
    try:
        config = TelegramConfig(
            bot_token=bot_token,
            chat_ids=chat_ids.split(','),
            enabled=enabled,
            min_probability=min_prob
        )
        
        system = TelegramNotificationSystem(config)
        
        print(f"\n✅ System initialized successfully!")
        print(f"  Valid config: {system._validate_config()}")
        print(f"  Chat count: {len(system.config.chat_ids)}")
        
        # Check notification stats
        stats = system.get_notification_stats()
        print(f"\n📈 Current Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ System initialization failed: {e}")
        return False

async def check_bot_info():
    """Check bot information via Telegram API"""
    
    # Load environment variables from .env file
    load_dotenv()
    
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
    if not bot_token:
        print("❌ TELEGRAM_BOT_TOKEN not set in environment variables or .env file")
        print("Please set your bot token first using 'python telegram_setup.py --setup' for instructions")
        return
    
    print("🔍 CHECKING BOT INFORMATION")
    print("=" * 35)
    
    try:
        import aiohttp
        
        url = f"https://api.telegram.org/bot{bot_token}/getMe"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('ok'):
                        bot_info = data.get('result', {})
                        print("✅ Bot Information:")
                        print(f"  Name: {bot_info.get('first_name', 'N/A')}")
                        print(f"  Username: @{bot_info.get('username', 'N/A')}")
                        print(f"  ID: {bot_info.get('id', 'N/A')}")
                        print(f"  Can join groups: {bot_info.get('can_join_groups', False)}")
                        print(f"  Can read all group messages: {bot_info.get('can_read_all_group_messages', False)}")
                    else:
                        print(f"❌ API Error: {data.get('description', 'Unknown error')}")
                else:
                    print(f"❌ HTTP Error: {response.status}")
                    
    except Exception as e:
        print(f"❌ Error checking bot info: {e}")

def send_test_message():
    """Send test message to verify setup"""
    
    print("📤 SENDING TEST MESSAGE")
    print("=" * 30)
    
    try:
        success = send_test_notification(
            "🧪 <b>Test Message from Tennis Prediction Bot</b>\n\n"
            "✅ Your Telegram integration is working correctly!\n"
            f"📅 Test conducted on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            "🎾 You will now receive notifications when strong underdog opportunities are detected."
        )
        
        if success:
            print("✅ Test message sent successfully!")
        else:
            print("❌ Failed to send test message")
            print("Please check your bot token and chat ID configuration")
            
    except Exception as e:
        print(f"❌ Error sending test message: {e}")

def simulate_underdog_prediction():
    """Simulate a strong underdog prediction and test notification"""
    
    print("🎯 SIMULATING UNDERDOG PREDICTION")
    print("=" * 40)
    
    # Create realistic sample prediction
    sample_prediction = {
        'success': True,
        'underdog_second_set_probability': 0.67,  # Strong underdog opportunity
        'confidence': 'High',
        'underdog_player': 'player1',
        'match_context': {
            'player1': 'A. Karatsev',  # Real player name
            'player2': 'D. Thiem',    # Real player name
            'player1_rank': 185,
            'player2_rank': 45,
            'tournament': 'ATP 250 Vienna',
            'surface': 'Hard'
        },
        'strategic_insights': [
            '🔥 Strong underdog opportunity - high second set win probability',
            '📊 Large ranking gap (140 positions) creates significant upset potential',
            '⚡ Hard court surface favors aggressive baseline play'
        ],
        'prediction_metadata': {
            'service_type': 'comprehensive_ml_service',
            'prediction_time': datetime.now().isoformat(),
            'models_used': ['random_forest', 'gradient_boosting', 'xgboost'],
            'processing_duration_ms': 850
        }
    }
    
    print("📊 Sample Prediction Details:")
    print(f"  Match: {sample_prediction['match_context']['player1']} vs {sample_prediction['match_context']['player2']}")
    print(f"  Underdog probability: {sample_prediction['underdog_second_set_probability']:.1%}")
    print(f"  Confidence: {sample_prediction['confidence']}")
    print(f"  Tournament: {sample_prediction['match_context']['tournament']}")
    
    # Test if notification would be sent
    system = TelegramNotificationSystem()
    should_notify = system.should_notify(sample_prediction)
    
    print(f"\n🤔 Should notify: {should_notify}")
    
    if should_notify:
        print("\n📤 Sending sample notification...")
        try:
            success = system.send_notification_sync(sample_prediction)
            if success:
                print("✅ Sample notification sent successfully!")
            else:
                print("❌ Failed to send sample notification")
        except Exception as e:
            print(f"❌ Error sending sample notification: {e}")
    else:
        print("\n📱 Notification criteria not met (this is expected for testing)")
        print("Reasons notification might not be sent:")
        print("  • Telegram notifications disabled")
        print("  • Probability below threshold")
        print("  • Confidence too low")
        print("  • Rate limiting active")

def main():
    parser = argparse.ArgumentParser(description='Telegram Bot Setup and Testing')
    parser.add_argument('--setup', action='store_true', help='Show setup instructions')
    parser.add_argument('--test-config', action='store_true', help='Test configuration')
    parser.add_argument('--check-bot', action='store_true', help='Check bot information')
    parser.add_argument('--send-test', action='store_true', help='Send test message')
    parser.add_argument('--simulate', action='store_true', help='Simulate underdog prediction')
    
    args = parser.parse_args()
    
    if args.setup or not any(vars(args).values()):
        print_setup_instructions()
    
    if args.test_config:
        print()
        test_configuration()
    
    if args.check_bot:
        print()
        asyncio.run(check_bot_info())
    
    if args.send_test:
        print()
        send_test_message()
    
    if args.simulate:
        print()
        simulate_underdog_prediction()

if __name__ == "__main__":
    main()