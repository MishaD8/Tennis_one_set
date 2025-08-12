#!/usr/bin/env python3
"""
Get Telegram Chat ID helper script
"""

import requests
import os
import sys
from dotenv import load_dotenv

def get_chat_id():
    # Load environment variables from .env file
    load_dotenv()
    
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    
    if not bot_token:
        print("❌ TELEGRAM_BOT_TOKEN not set in environment variables or .env file")
        print("Please set your bot token:")
        print("  Option 1: export TELEGRAM_BOT_TOKEN='your_bot_token_here'")
        print("  Option 2: Add TELEGRAM_BOT_TOKEN=your_bot_token_here to .env file")
        print("\nTo get a bot token:")
        print("  1. Open Telegram and search for @BotFather")
        print("  2. Send /newbot command and follow instructions")
        print("  3. Copy the bot token provided")
        return
    
    print("🔍 Fetching recent messages to find your chat ID...")
    
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if data.get('ok'):
            updates = data.get('result', [])
            
            if not updates:
                print("❌ No messages found!")
                print("Please:")
                print("1. Open Telegram")
                print("2. Search for @underdog_one_set_bot")
                print("3. Send any message to the bot (like 'hello')")
                print("4. Run this script again")
                return
            
            print("✅ Found messages! Here are the chat IDs:")
            print()
            
            chat_ids = set()
            for update in updates:
                message = update.get('message', {})
                chat = message.get('chat', {})
                chat_id = chat.get('id')
                chat_type = chat.get('type')
                first_name = chat.get('first_name', '')
                username = chat.get('username', '')
                
                if chat_id:
                    chat_ids.add(chat_id)
                    print(f"Chat ID: {chat_id}")
                    print(f"  Type: {chat_type}")
                    print(f"  Name: {first_name}")
                    if username:
                        print(f"  Username: @{username}")
                    print()
            
            if chat_ids:
                chat_ids_list = list(chat_ids)
                print("🎯 Use this command to set your chat ID:")
                if len(chat_ids_list) == 1:
                    print(f"export TELEGRAM_CHAT_IDS='{chat_ids_list[0]}'")
                else:
                    chat_ids_str = ','.join(map(str, chat_ids_list))
                    print(f"export TELEGRAM_CHAT_IDS='{chat_ids_str}'")
                
        else:
            print(f"❌ API Error: {data.get('description', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error fetching updates: {e}")

if __name__ == "__main__":
    get_chat_id()