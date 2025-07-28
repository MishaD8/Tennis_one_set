#!/usr/bin/env python3
"""
Test script to check scheduled matches functionality
"""

import sys
import json
from rapidapi_tennis_client import RapidAPITennisClient

def test_scheduled_matches():
    """Test the scheduled matches functionality"""
    try:
        print("🧪 Testing RapidAPI scheduled matches...")
        client = RapidAPITennisClient()
        
        print(f"📊 Status: {client.get_status()}")
        
        # Test 1: Get all events
        print("\n1️⃣ Testing all events endpoint...")
        all_events = client.get_all_events()
        if all_events:
            print(f"✅ Found {len(all_events)} total events")
            # Show first few with status
            for i, event in enumerate(all_events[:5]):
                status = event.get('status', {})
                tournament = event.get('tournament', {}).get('name', 'Unknown')
                home_team = event.get('homeTeam', {}).get('name', 'Player1')
                away_team = event.get('awayTeam', {}).get('name', 'Player2')
                print(f"  Event {i+1}: {home_team} vs {away_team} ({tournament}) - Status: {status.get('type', 'unknown')}")
        else:
            print("❌ No events found")
        
        # Test 2: Get scheduled matches
        print("\n2️⃣ Testing scheduled matches...")
        scheduled_matches = client.get_scheduled_matches()
        if scheduled_matches:
            print(f"✅ Found {len(scheduled_matches)} scheduled matches")
            for i, match in enumerate(scheduled_matches[:3]):
                home_team = match.get('homeTeam', {}).get('name', 'Player1')
                away_team = match.get('awayTeam', {}).get('name', 'Player2')
                tournament = match.get('tournament', {}).get('name', 'Tournament')
                status = match.get('status', {}).get('type', 'unknown')
                start_time = match.get('startTimestamp', 'No time')
                print(f"  Match {i+1}: {home_team} vs {away_team}")
                print(f"    Tournament: {tournament}")
                print(f"    Status: {status}")
                print(f"    Start time: {start_time}")
                print()
        else:
            print("❌ No scheduled matches found")
        
        # Test 3: Get tournaments
        print("\n3️⃣ Testing tournaments endpoint...")
        tournaments = client.get_tournaments()
        if tournaments:
            print(f"✅ Found {len(tournaments)} tournaments")
            for i, tournament in enumerate(tournaments[:3]):
                name = tournament.get('name', 'Unknown')
                category = tournament.get('category', {}).get('name', 'Unknown')
                print(f"  Tournament {i+1}: {name} ({category})")
        else:
            print("❌ No tournaments found")
            
        print(f"\n📊 Final Status: {client.get_status()}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_scheduled_matches()