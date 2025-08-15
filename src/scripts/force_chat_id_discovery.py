#!/usr/bin/env python3
"""
Force discovery of chat ID by getting all updates and bot interactions
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

def get_all_updates(offset=0):
    """Get all updates including old ones"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    params = {
        'offset': offset,
        'limit': 100,
        'timeout': 0
    }
    response = requests.get(url, params=params)
    return response.json()

def send_message_to_username():
    """Try sending message using the bot's username approach"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    
    # Try different chat ID formats
    chat_formats = [
        '@underdog_one_set_bot',
        'underdog_one_set_bot',
        '8369911887',  # Bot's own ID
        'me'
    ]
    
    for chat_id in chat_formats:
        print(f"\n📤 Trying chat ID: {chat_id}")
        
        payload = {
            'chat_id': chat_id,
            'text': f'🎾 Test from Tennis Bot\n\nTesting chat ID: {chat_id}',
            'parse_mode': 'HTML'
        }
        
        response = requests.post(url, json=payload)
        result = response.json()
        
        if response.status_code == 200 and result.get('ok'):
            print(f"   ✅ SUCCESS with chat ID: {chat_id}")
            return chat_id, result
        else:
            print(f"   ❌ Failed: {result.get('description', 'Unknown error')}")
    
    return None, None

def main():
    if not BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN not found")
        return
    
    print("🔍 Telegram Chat ID Discovery Tool")
    print("=" * 50)
    
    # Get bot info
    bot_url = f"https://api.telegram.org/bot{BOT_TOKEN}/getMe"
    bot_response = requests.get(bot_url)
    bot_info = bot_response.json()
    
    if bot_info.get('ok'):
        bot_data = bot_info['result']
        print(f"✅ Bot: @{bot_data['username']} (ID: {bot_data['id']})")
    
    # Try to get all historical updates
    print(f"\n📨 Searching for ALL bot interactions...")
    
    all_updates = []
    offset = 0
    
    # Get updates in batches
    for batch in range(10):  # Get up to 1000 messages
        updates = get_all_updates(offset)
        if updates.get('ok') and updates['result']:
            batch_updates = updates['result']
            all_updates.extend(batch_updates)
            offset = batch_updates[-1]['update_id'] + 1
            print(f"   Batch {batch + 1}: Found {len(batch_updates)} messages")
        else:
            break
    
    print(f"📊 Total messages found: {len(all_updates)}")
    
    # Extract all unique chat IDs
    unique_chats = {}
    
    for update in all_updates:
        if 'message' in update:
            msg = update['message']
            chat = msg['chat']
            chat_id = chat['id']
            chat_type = chat['type']
            
            if chat_id not in unique_chats:
                unique_chats[chat_id] = {
                    'type': chat_type,
                    'title': chat.get('title', chat.get('first_name', chat.get('username', 'Unknown'))),
                    'username': chat.get('username', ''),
                    'message_count': 0
                }
            
            unique_chats[chat_id]['message_count'] += 1
    
    if unique_chats:
        print(f"\n💬 Found {len(unique_chats)} unique chat(s):")
        for chat_id, info in unique_chats.items():
            print(f"   {chat_id} ({info['type']}) - {info['title']} - {info['message_count']} messages")
        
        # Test each chat ID
        print(f"\n📤 Testing each chat ID...")
        working_chats = []
        
        for chat_id in unique_chats.keys():
            print(f"\n   Testing: {chat_id}")
            
            test_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
            test_payload = {
                'chat_id': chat_id,
                'text': f'🎾 Tennis Bot Test\n\nYour chat ID is: {chat_id}\n\nThis confirms notifications will work!',
                'parse_mode': 'HTML'
            }
            
            response = requests.post(test_url, json=test_payload)
            result = response.json()
            
            if response.status_code == 200 and result.get('ok'):
                print(f"   ✅ SUCCESS! Working chat ID: {chat_id}")
                working_chats.append(chat_id)
            else:
                print(f"   ❌ Failed: {result.get('description', 'Unknown')}")
        
        if working_chats:
            print(f"\n🎉 Found {len(working_chats)} working chat ID(s)!")
            
            # Create .env update instruction
            if len(working_chats) == 1:
                new_chat_id = working_chats[0]
                print(f"\n🔧 Update your .env file:")
                print(f"   Change: TELEGRAM_CHAT_IDS=@underdog_one_set_bot")
                print(f"   To:     TELEGRAM_CHAT_IDS={new_chat_id}")
                
                # Automatically update .env file
                try:
                    with open('.env', 'r') as f:
                        content = f.read()
                    
                    updated_content = content.replace(
                        'TELEGRAM_CHAT_IDS=@underdog_one_set_bot',
                        f'TELEGRAM_CHAT_IDS={new_chat_id}'
                    )
                    
                    with open('.env', 'w') as f:
                        f.write(updated_content)
                    
                    print(f"✅ .env file automatically updated!")
                    print(f"   New chat ID: {new_chat_id}")
                    
                except Exception as e:
                    print(f"❌ Failed to auto-update .env: {e}")
                    print(f"   Please update manually")
            
            else:
                chat_ids_str = ','.join(map(str, working_chats))
                print(f"\n🔧 Multiple chat IDs found. Update .env:")
                print(f"   TELEGRAM_CHAT_IDS={chat_ids_str}")
        
        else:
            print(f"\n❌ No working chat IDs found")
            print(f"   This might mean the bot hasn't received any messages yet")
    
    else:
        print(f"\n❌ No message history found")
        print(f"   The bot may not have received any messages yet")
        print(f"\n📱 To fix this:")
        print(f"   1. Open Telegram")
        print(f"   2. Search: @underdog_one_set_bot") 
        print(f"   3. Send any message like: /start")
        print(f"   4. Run this script again")

if __name__ == "__main__":
    main()