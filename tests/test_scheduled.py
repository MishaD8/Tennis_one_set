#!/usr/bin/env python3
import requests
import json

def test_scheduled_matches():
    try:
        print("🧪 Testing scheduled matches endpoint...")
        response = requests.get("http://localhost:5001/api/matches", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Got {data.get('count', 0)} matches")
            print(f"📊 Source: {data.get('source', 'unknown')}")
            
            for i, match in enumerate(data.get('matches', [])[:3]):
                print(f"\nMatch {i+1}:")
                print(f"  Players: {match.get('player1', 'Unknown')} vs {match.get('player2', 'Unknown')}")
                print(f"  Tournament: {match.get('tournament', 'Unknown')}")
                print(f"  Time: {match.get('time', 'Unknown')}")
                print(f"  Status: {match.get('status', 'unknown')}")
                print(f"  Source: {match.get('debug_info', {}).get('original_source', 'Unknown')}")
                
                # Check if it's scheduled or live
                if match.get('time') != 'live' and match.get('status') != 'inprogress':
                    print("  ✅ SCHEDULED MATCH (not live)")
                else:
                    print("  ⚠️ LIVE MATCH (already started)")
        else:
            print(f"❌ Request failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_scheduled_matches()