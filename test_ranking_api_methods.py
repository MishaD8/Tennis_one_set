#!/usr/bin/env python3
"""
Test script to investigate get_players and get_standings API methods
According to the API documentation, these methods should provide ranking data
"""

import os
import sys
import json
import requests
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api_tennis_integration import APITennisClient

def test_get_standings():
    """Test the get_standings method according to API documentation"""
    print("🏆 Testing get_standings method")
    print("=" * 50)
    
    client = APITennisClient()
    
    if not client.api_key:
        print("❌ API_TENNIS_KEY not configured - cannot test")
        return
    
    # According to docs: Parameters are event_type ('ATP' or 'WTA') not league_id
    for event_type in ['ATP', 'WTA']:
        try:
            print(f"\nTesting get_standings for {event_type}...")
            
            # Make direct API request with correct parameters
            params = {
                'method': 'get_standings',
                'APIkey': client.api_key,
                'event_type': event_type
            }
            
            response = client.session.get(client.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            print(f"Response for {event_type}:")
            print(f"Success: {data.get('success')}")
            
            if data.get('success') == 1:
                result = data.get('result', [])
                print(f"Found {len(result)} players in {event_type} rankings")
                
                # Show top 5 players
                for i, player in enumerate(result[:5]):
                    print(f"  {player.get('place')}. {player.get('player')} "
                          f"(Key: {player.get('player_key')}, Points: {player.get('points')})")
                
                # Save sample data
                with open(f'ranking_sample_{event_type.lower()}.json', 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"Sample data saved to ranking_sample_{event_type.lower()}.json")
                
            else:
                print(f"❌ Failed: {data}")
                
        except Exception as e:
            print(f"❌ Error testing {event_type} standings: {e}")

def test_get_players():
    """Test the get_players method according to API documentation"""
    print("\n👤 Testing get_players method")
    print("=" * 50)
    
    client = APITennisClient()
    
    if not client.api_key:
        print("❌ API_TENNIS_KEY not configured - cannot test")
        return
    
    # Test with specific player keys from the docs
    test_player_keys = [1905, 137, 30, 5]  # Example player keys from documentation
    
    for player_key in test_player_keys:
        try:
            print(f"\nTesting get_players for player_key {player_key}...")
            
            # Make direct API request with correct parameters
            params = {
                'method': 'get_players',
                'APIkey': client.api_key,
                'player_key': player_key
            }
            
            response = client.session.get(client.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            print(f"Response for player_key {player_key}:")
            print(f"Success: {data.get('success')}")
            
            if data.get('success') == 1:
                result = data.get('result', [])
                print(f"Found {len(result)} player records")
                
                for player in result:
                    print(f"  Player: {player.get('player_name')}")
                    print(f"  Country: {player.get('player_country')}")
                    print(f"  Birthday: {player.get('player_bday')}")
                    
                    # Check stats for ranking information
                    stats = player.get('stats', [])
                    print(f"  Stats records: {len(stats)}")
                    for stat in stats[:3]:  # Show first 3 stat records
                        print(f"    Season: {stat.get('season')}, Type: {stat.get('type')}, "
                              f"Rank: {stat.get('rank')}, Titles: {stat.get('titles')}")
                
                # Save sample data
                with open(f'player_sample_{player_key}.json', 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"Sample data saved to player_sample_{player_key}.json")
                
            else:
                print(f"❌ Failed: {data}")
                
        except Exception as e:
            print(f"❌ Error testing player_key {player_key}: {e}")

def test_current_implementation():
    """Test current implementation to identify issues"""
    print("\n🔍 Testing current implementation")
    print("=" * 50)
    
    try:
        client = APITennisClient()
        
        print("Current get_standings method (incorrect implementation):")
        # This will fail because it uses wrong parameter
        try:
            standings = client.get_standings(league_id=123)
            print(f"Current method returned: {standings}")
        except Exception as e:
            print(f"Current method failed: {e}")
        
        print("\nCurrent get_players method (incorrect implementation):")
        # This calls 'get_teams' instead of 'get_players'
        try:
            players = client.get_players(league_id=123)
            print(f"Current method returned: {len(players)} players")
        except Exception as e:
            print(f"Current method failed: {e}")
            
    except Exception as e:
        print(f"❌ Error testing current implementation: {e}")

def analyze_fixture_data_for_ranking():
    """Analyze current fixture data to see if player keys can be used for ranking lookup"""
    print("\n📊 Analyzing fixture data for ranking potential")
    print("=" * 50)
    
    try:
        client = APITennisClient()
        
        # Get some current matches
        today = datetime.now().strftime('%Y-%m-%d')
        matches = client.get_fixtures(date_start=today, date_stop=today)
        
        print(f"Found {len(matches)} matches today")
        
        player_keys = []
        for match in matches[:5]:  # Check first 5 matches
            print(f"\nMatch: {match.player1.name} vs {match.player2.name}")
            print(f"Player1 Key: {match.player1.id}, Player2 Key: {match.player2.id}")
            print(f"Current Rankings: P1={match.player1.ranking}, P2={match.player2.ranking}")
            
            if match.player1.id:
                player_keys.append(match.player1.id)
            if match.player2.id:
                player_keys.append(match.player2.id)
        
        print(f"\nCollected player keys: {player_keys[:10]}")  # Show first 10
        return player_keys
        
    except Exception as e:
        print(f"❌ Error analyzing fixture data: {e}")
        return []

def main():
    """Run all tests to understand ranking data availability"""
    print("🎾 Tennis Ranking API Investigation")
    print("=" * 60)
    
    # Check API key configuration
    api_key = os.getenv('API_TENNIS_KEY')
    if api_key:
        print(f"✅ API_TENNIS_KEY configured (length: {len(api_key)})")
    else:
        print("❌ API_TENNIS_KEY not configured - tests will fail")
        return
    
    # Run tests
    test_get_standings()
    test_get_players()
    test_current_implementation()
    
    # Analyze current data
    player_keys = analyze_fixture_data_for_ranking()
    
    # If we have player keys, test ranking lookup
    if player_keys:
        print(f"\n🔍 Testing ranking lookup with real player keys")
        for player_key in player_keys[:3]:  # Test first 3
            test_get_players_with_key(player_key)

def test_get_players_with_key(player_key):
    """Test get_players with a specific player key"""
    try:
        client = APITennisClient()
        
        params = {
            'method': 'get_players',
            'APIkey': client.api_key,
            'player_key': player_key
        }
        
        response = client.session.get(client.BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get('success') == 1:
            result = data.get('result', [])
            if result:
                player = result[0]
                print(f"  Player {player_key}: {player.get('player_name')} "
                      f"from {player.get('player_country')}")
                
                # Look for ranking in stats
                stats = player.get('stats', [])
                if stats:
                    latest_stat = stats[0]  # Most recent stat
                    rank = latest_stat.get('rank')
                    print(f"    Latest rank: {rank} ({latest_stat.get('season')}, {latest_stat.get('type')})")
                else:
                    print(f"    No stats/ranking data available")
        else:
            print(f"  Player {player_key}: Failed to get data")
            
    except Exception as e:
        print(f"  Player {player_key}: Error - {e}")

if __name__ == "__main__":
    main()