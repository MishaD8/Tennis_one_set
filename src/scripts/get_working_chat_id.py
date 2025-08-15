#!/usr/bin/env python3
"""
Get working chat ID for Telegram bot
This script will help you get the correct chat ID to use in the .env file
"""

import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

def get_bot_info():
    """Get bot information"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getMe"
    response = requests.get(url)
    return response.json()

def get_updates():
    """Get recent updates/messages"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    response = requests.get(url)
    return response.json()

def send_test_to_chat_id(chat_id):
    """Test sending a message to a specific chat ID"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    
    payload = {
        'chat_id': chat_id,
        'text': '🎾 Test message from Tennis Prediction Bot!\n\nIf you received this, the chat ID is working correctly.',
        'parse_mode': 'HTML'
    }
    
    response = requests.post(url, json=payload)
    return response.status_code, response.json()

def main():
    if not BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN not found in environment")
        return
    
    print("🤖 Telegram Bot Chat ID Helper")
    print("=" * 40)
    
    # Get bot info
    bot_info = get_bot_info()
    if bot_info.get('ok'):
        bot_data = bot_info['result']
        print(f"✅ Bot: @{bot_data['username']} ({bot_data['first_name']})")
        print(f"   Bot ID: {bot_data['id']}")
    else:
        print(f"❌ Failed to get bot info: {bot_info}")
        return
    
    # Get recent messages
    print(f"\n📨 Checking for recent messages...")
    updates = get_updates()
    
    if updates.get('ok'):
        messages = updates['result']
        
        if not messages:
            print("❌ No messages found!")
            print("\n📱 To get your chat ID:")
            print("1. Open Telegram")
            print("2. Search for @underdog_one_set_bot")
            print("3. Start a conversation by sending: /start")
            print("4. Send any message like 'hello'")
            print("5. Run this script again")
            return
        
        print(f"✅ Found {len(messages)} recent messages")
        
        # Extract unique chat IDs
        chat_ids = set()
        for msg in messages:
            if 'message' in msg:
                chat_data = msg['message']['chat']
                chat_id = chat_data['id']
                chat_type = chat_data['type']
                chat_title = chat_data.get('title', chat_data.get('first_name', 'Unknown'))
                
                chat_ids.add(chat_id)
                print(f"   Chat: {chat_id} ({chat_type}) - {chat_title}")
        
        # Test each chat ID
        print(f"\n📤 Testing chat IDs...")
        working_chat_ids = []
        
        for chat_id in chat_ids:
            print(f"\n   Testing chat ID: {chat_id}")
            status, result = send_test_to_chat_id(chat_id)
            
            if status == 200 and result.get('ok'):
                print(f"   ✅ SUCCESS! Chat ID {chat_id} works")
                working_chat_ids.append(chat_id)
            else:
                print(f"   ❌ Failed: {result}")
        
        # Provide instructions
        if working_chat_ids:
            print(f"\n🎉 Found {len(working_chat_ids)} working chat ID(s)!")
            print(f"\n🔧 Update your .env file:")
            
            if len(working_chat_ids) == 1:
                print(f"   TELEGRAM_CHAT_IDS={working_chat_ids[0]}")
            else:
                chat_ids_str = ','.join(map(str, working_chat_ids))
                print(f"   TELEGRAM_CHAT_IDS={chat_ids_str}")
            
            print(f"\n✅ After updating .env, your notifications will work!")
        else:
            print(f"\n❌ No working chat IDs found")
    else:
        print(f"❌ Failed to get updates: {updates}")

if __name__ == "__main__":
    main()