#!/usr/bin/env python3
"""
Direct Telegram test to find working chat ID and send notification
"""

import os
import requests
import asyncio
import aiohttp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
if not BOT_TOKEN:
    print("❌ TELEGRAM_BOT_TOKEN not found in environment")
    exit(1)

print(f"🤖 Using bot token: {BOT_TOKEN[:20]}...")

async def send_test_message(chat_id, message):
    """Send test message to a specific chat ID"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'HTML'
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            result = await response.json()
            return response.status, result

async def get_bot_info():
    """Get bot information"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getMe"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            result = await response.json()
            return response.status, result

async def get_updates():
    """Get recent messages to bot"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            result = await response.json()
            return response.status, result

async def main():
    print("📋 Getting bot information...")
    status, bot_info = await get_bot_info()
    
    if status == 200 and bot_info.get('ok'):
        bot_data = bot_info['result']
        print(f"✅ Bot connected: @{bot_data['username']} ({bot_data['first_name']})")
        print(f"   Bot ID: {bot_data['id']}")
    else:
        print(f"❌ Failed to get bot info: {bot_info}")
        return
    
    print("\n📨 Getting recent updates...")
    status, updates = await get_updates()
    
    chat_ids = []
    if status == 200 and updates.get('ok'):
        messages = updates['result']
        if messages:
            print(f"✅ Found {len(messages)} recent messages")
            for msg in messages:
                if 'message' in msg:
                    chat_id = msg['message']['chat']['id']
                    chat_type = msg['message']['chat']['type']
                    if chat_id not in chat_ids:
                        chat_ids.append(chat_id)
                        print(f"   Chat ID: {chat_id} (type: {chat_type})")
        else:
            print("❌ No recent messages found")
            print("   Please send a message to @underdog_one_set_bot first")
    
    # Test message
    test_message = """🎾 <b>Tennis Prediction System Test</b> 🎾

✅ Bot connection successful!
🤖 Automated notifications are working

This is a test message from your tennis prediction system."""
    
    # Try different chat ID approaches
    test_chat_ids = []
    
    # Add found chat IDs
    if chat_ids:
        test_chat_ids.extend(chat_ids)
    
    # Try common patterns if no chat IDs found
    if not chat_ids:
        print("\n🔍 No chat IDs found. Please:")
        print("   1. Open Telegram")
        print("   2. Search for @underdog_one_set_bot")
        print("   3. Send any message to the bot")
        print("   4. Run this script again")
        return
    
    print(f"\n📤 Testing notification with {len(test_chat_ids)} chat IDs...")
    
    for chat_id in test_chat_ids:
        print(f"\n   Testing chat ID: {chat_id}")
        status, result = await send_test_message(chat_id, test_message)
        
        if status == 200 and result.get('ok'):
            print(f"   ✅ SUCCESS! Message sent to chat {chat_id}")
            print(f"      Message ID: {result['result']['message_id']}")
            
            # Update .env file with working chat ID
            print(f"\n🔧 Update your .env file:")
            print(f"   TELEGRAM_CHAT_IDS={chat_id}")
            
        else:
            print(f"   ❌ FAILED: {result}")

if __name__ == "__main__":
    asyncio.run(main())