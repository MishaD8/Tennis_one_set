#!/usr/bin/env python3
"""
External API-Tennis.com Demo & Testing Script
Demonstrates all documented API-Tennis.com endpoints and their responses
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, Any

class ApiTennisDemo:
    def __init__(self, api_key: str = "demo_key"):
        self.base_url = "https://api.api-tennis.com/tennis/"
        self.api_key = api_key
        self.test_results = {}
        
        print("🎾 API-TENNIS.COM EXTERNAL API DEMONSTRATION")
        print("=" * 60)
        print(f"📍 Base URL: {self.base_url}")
        print(f"🔑 API Key: {'[CONFIGURED]' if api_key != 'demo_key' else '[DEMO MODE]'}")
        print("=" * 60)

    def make_api_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make request to API-Tennis.com with proper error handling"""
        try:
            params['APIkey'] = self.api_key
            
            print(f"\n🌐 API Call: {method}")
            print(f"📋 Parameters: {dict(params)}")
            
            response = requests.get(self.base_url, params=params, timeout=15)
            response_time = response.elapsed.total_seconds() * 1000
            
            print(f"⏱️  Response Time: {response_time:.0f}ms")
            print(f"📊 Status Code: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    json_data = response.json()
                    
                    # Check API success flag
                    if isinstance(json_data, dict):
                        api_success = json_data.get('success') == 1
                        print(f"✅ API Success: {api_success}")
                        
                        if 'result' in json_data:
                            result = json_data['result']
                            if isinstance(result, list):
                                print(f"📊 Results Count: {len(result)}")
                                if len(result) > 0:
                                    print(f"📋 Sample Result Keys: {list(result[0].keys()) if isinstance(result[0], dict) else 'N/A'}")
                            elif isinstance(result, dict):
                                print(f"📋 Result Keys: {list(result.keys())}")
                        
                        if not api_success and 'error' in json_data:
                            print(f"🚨 API Error: {json_data['error']}")
                    
                    return {
                        'success': True,
                        'api_success': json_data.get('success') == 1 if isinstance(json_data, dict) else False,
                        'status_code': response.status_code,
                        'response_time_ms': response_time,
                        'data': json_data,
                        'url': response.url
                    }
                    
                except json.JSONDecodeError:
                    print(f"❌ Invalid JSON response")
                    return {
                        'success': False,
                        'error': 'Invalid JSON response',
                        'status_code': response.status_code,
                        'raw_response': response.text[:500]
                    }
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                return {
                    'success': False,
                    'error': f'HTTP {response.status_code}',
                    'status_code': response.status_code,
                    'raw_response': response.text[:500]
                }
                
        except requests.exceptions.Timeout:
            print(f"⏰ Request Timeout")
            return {'success': False, 'error': 'Timeout'}
        except requests.exceptions.ConnectionError:
            print(f"🔌 Connection Error")
            return {'success': False, 'error': 'Connection Error'}
        except Exception as e:
            print(f"❌ Unexpected Error: {e}")
            return {'success': False, 'error': str(e)}

    def demo_get_events(self):
        """Demo: Get supported event types"""
        print("\n" + "="*50)
        print("📅 GET EVENTS - Supported Tournament Types")
        print("="*50)
        
        result = self.make_api_request('get_events', {'method': 'get_events'})
        self.test_results['get_events'] = result
        
        if result.get('api_success'):
            print("\n📋 DOCUMENTED RESPONSE STRUCTURE:")
            print("""
{
  "success": 1,
  "result": [
    {
      "event_type_key": "267",
      "event_type_type": "Atp Doubles"
    },
    {
      "event_type_key": "265", 
      "event_type_type": "Atp Singles"
    },
    ...
  ]
}""")
        
        return result

    def demo_get_tournaments(self):
        """Demo: Get available tournaments"""
        print("\n" + "="*50)
        print("🏆 GET TOURNAMENTS - Available Tournaments")
        print("="*50)
        
        result = self.make_api_request('get_tournaments', {'method': 'get_tournaments'})
        self.test_results['get_tournaments'] = result
        
        if result.get('api_success'):
            print("\n📋 DOCUMENTED RESPONSE STRUCTURE:")
            print("""
{
  "success": 1,
  "result": [
    {
      "tournament_key": "2833",
      "tournament_name": "Aachen",
      "event_type_key": "281",
      "event_type_type": "Challenger Men Singles"
    },
    ...
  ]
}""")
        
        return result

    def demo_get_fixtures(self):
        """Demo: Get tennis fixtures"""
        print("\n" + "="*50)
        print("📅 GET FIXTURES - Tennis Fixtures")
        print("="*50)
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        params = {
            'method': 'get_fixtures',
            'date_start': today,
            'date_stop': today
        }
        
        result = self.make_api_request('get_fixtures', params)
        self.test_results['get_fixtures'] = result
        
        if result.get('api_success'):
            print("\n📋 DOCUMENTED RESPONSE STRUCTURE:")
            print("""
{
  "success": 1,
  "result": [
    {
      "event_key": "143104",
      "event_date": "2022-06-17",
      "event_time": "18:00",
      "event_first_player": "M. Navone",
      "first_player_key": "949",
      "event_second_player": "C. Gomez-Herrera",
      "second_player_key": "3474",
      "event_final_result": "-",
      "event_status": "",
      "tournament_name": "Corrientes Challenger Men",
      "scores": [],
      "pointbypoint": []
    },
    ...
  ]
}""")
        
        return result

    def demo_get_livescore(self):
        """Demo: Get live tennis matches"""
        print("\n" + "="*50)
        print("🔴 GET LIVESCORE - Live Tennis Matches")
        print("="*50)
        
        result = self.make_api_request('get_livescore', {'method': 'get_livescore'})
        self.test_results['get_livescore'] = result
        
        if result.get('api_success'):
            print("\n📋 DOCUMENTED RESPONSE STRUCTURE:")
            print("""
{
  "success": 1,
  "result": [
    {
      "event_key": "143192",
      "event_date": "2022-06-17",
      "event_time": "10:10",
      "event_first_player": "S. Bejlek",
      "event_second_player": "R. Zarazua",
      "event_final_result": "0 - 0",
      "event_game_result": "0 - 0",
      "event_serve": "First Player",
      "event_status": "Set 1",
      "event_live": "1",
      "pointbypoint": [...],
      "scores": [...]
    },
    ...
  ]
}""")
        
        return result

    def demo_get_standings(self):
        """Demo: Get ATP/WTA standings"""
        print("\n" + "="*50)
        print("🏅 GET STANDINGS - ATP/WTA Rankings")
        print("="*50)
        
        # Test ATP standings
        print("\n🎾 Testing ATP Standings:")
        atp_result = self.make_api_request('get_standings_atp', {
            'method': 'get_standings',
            'event_type': 'ATP'
        })
        
        # Test WTA standings  
        print("\n🎾 Testing WTA Standings:")
        wta_result = self.make_api_request('get_standings_wta', {
            'method': 'get_standings',
            'event_type': 'WTA'
        })
        
        self.test_results['get_standings_atp'] = atp_result
        self.test_results['get_standings_wta'] = wta_result
        
        if atp_result.get('api_success') or wta_result.get('api_success'):
            print("\n📋 DOCUMENTED RESPONSE STRUCTURE:")
            print("""
{
  "success": 1,
  "result": [
    {
      "place": "1",
      "player": "Iga Swiatek",
      "player_key": "1910",
      "league": "WTA",
      "movement": "same",
      "country": "Poland", 
      "points": "8501"
    },
    ...
  ]
}""")
        
        return {'atp': atp_result, 'wta': wta_result}

    def demo_get_players(self):
        """Demo: Get player information"""
        print("\n" + "="*50)
        print("👤 GET PLAYERS - Player Profiles")
        print("="*50)
        
        # Test with Djokovic's player key
        params = {
            'method': 'get_players',
            'player_key': '1905'  # Djokovic
        }
        
        result = self.make_api_request('get_players', params)
        self.test_results['get_players'] = result
        
        if result.get('api_success'):
            print("\n📋 DOCUMENTED RESPONSE STRUCTURE:")
            print("""
{
  "success": 1,
  "result": [
    {
      "player_key": "1905",
      "player_name": "N. Djokovic",
      "player_country": "Serbia",
      "player_bday": "22.05.1987",
      "player_logo": "https://api.api-tennis.com/logo-tennis/1905_n-djokovic.jpg",
      "stats": [
        {
          "season": "2021",
          "type": "doubles",
          "rank": "255",
          "titles": "0",
          "matches_won": "6",
          "matches_lost": "4",
          ...
        },
        ...
      ]
    }
  ]
}""")
        
        return result

    def demo_get_h2h(self):
        """Demo: Get head-to-head analysis"""
        print("\n" + "="*50)
        print("⚔️  GET H2H - Head to Head Analysis")
        print("="*50)
        
        # Test Djokovic vs Alcaraz H2H
        params = {
            'method': 'get_H2H',
            'first_player_key': '1905',  # Djokovic
            'second_player_key': '1642'  # Alcaraz (example)
        }
        
        result = self.make_api_request('get_h2h', params)
        self.test_results['get_h2h'] = result
        
        if result.get('api_success'):
            print("\n📋 DOCUMENTED RESPONSE STRUCTURE:")
            print("""
{
  "success": 1,
  "result": {
    "H2H": [...],
    "firstPlayerResults": [
      {
        "event_key": "112163",
        "event_date": "2022-05-11",
        "event_first_player": "Player Name",
        "event_winner": "Second Player",
        "tournament_name": "Tournament Name",
        ...
      },
      ...
    ],
    "secondPlayerResults": [...]
  }
}""")
        
        return result

    def demo_get_odds(self):
        """Demo: Get betting odds"""
        print("\n" + "="*50)
        print("💰 GET ODDS - Betting Odds")
        print("="*50)
        
        # Test with sample match key from documentation
        params = {
            'method': 'get_odds',
            'match_key': '159923'
        }
        
        result = self.make_api_request('get_odds', params)
        self.test_results['get_odds'] = result
        
        if result.get('api_success'):
            print("\n📋 DOCUMENTED RESPONSE STRUCTURE:")
            print("""
{
  "success": 1,
  "result": {
    "159923": {
      "Home/Away": {
        "Home": {
          "bwin": "2.40",
          "bet365": "2.50",
          "Betsson": "2.45",
          ...
        },
        "Away": {
          "bwin": "1.48",
          "bet365": "1.50",
          ...
        }
      },
      "Set Betting": {...},
      "Correct Score 1st Half": {...}
    }
  }
}""")
        
        return result

    def demo_get_live_odds(self):
        """Demo: Get live betting odds"""
        print("\n" + "="*50)
        print("🔴💰 GET LIVE ODDS - Live Betting Odds")
        print("="*50)
        
        result = self.make_api_request('get_live_odds', {'method': 'get_live_odds'})
        self.test_results['get_live_odds'] = result
        
        if result.get('api_success'):
            print("\n📋 DOCUMENTED RESPONSE STRUCTURE:")
            print("""
{
  "success": 1,
  "result": {
    "11976653": {
      "event_key": 11976653,
      "event_date": "2024-08-22",
      "event_game_result": "30 - 30",
      "event_serve": "First Player",
      "event_status": "Set 2",
      "live_odds": [
        {
          "odd_name": "Set 1 to Break Serve",
          "suspended": "Yes",
          "type": "1/Yes",
          "value": "1.125",
          "handicap": null,
          "upd": "2024-08-22 09:15:10"
        },
        ...
      ]
    }
  }
}""")
        
        return result

    def run_complete_demo(self):
        """Run complete demonstration of all API-Tennis.com endpoints"""
        print("🚀 RUNNING COMPLETE API-TENNIS.COM DEMONSTRATION")
        print("⏰ Started at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print("=" * 70)
        
        # Rate limiting delay between requests
        delay = 1.0
        
        try:
            # Test all endpoints
            self.demo_get_events()
            time.sleep(delay)
            
            self.demo_get_tournaments()
            time.sleep(delay)
            
            self.demo_get_fixtures()
            time.sleep(delay)
            
            self.demo_get_livescore()
            time.sleep(delay)
            
            self.demo_get_standings()
            time.sleep(delay)
            
            self.demo_get_players()
            time.sleep(delay)
            
            self.demo_get_h2h()
            time.sleep(delay)
            
            self.demo_get_odds()
            time.sleep(delay)
            
            self.demo_get_live_odds()
            
        except KeyboardInterrupt:
            print("\n⚠️  Demo interrupted by user")
        except Exception as e:
            print(f"\n❌ Demo error: {e}")
        
        # Generate summary
        self.generate_summary()

    def generate_summary(self):
        """Generate demo summary"""
        print("\n" + "="*70)
        print("📊 API-TENNIS.COM DEMO SUMMARY")
        print("="*70)
        
        total_tests = len(self.test_results)
        successful_connections = sum(1 for result in self.test_results.values() if result.get('success', False))
        api_successful = sum(1 for result in self.test_results.values() if result.get('api_success', False))
        
        print(f"📊 Total Endpoints Tested: {total_tests}")
        print(f"🔌 Successful Connections: {successful_connections}")
        print(f"✅ API Successful Responses: {api_successful}")
        
        if self.api_key == "demo_key":
            print(f"\n⚠️  DEMO MODE RESULTS:")
            print(f"   - Tests run with demo API key")
            print(f"   - Real API responses will require valid API-Tennis.com subscription")
            print(f"   - Set environment variable API_TENNIS_KEY for live testing")
        
        print(f"\n📋 ENDPOINT STATUS:")
        for endpoint, result in self.test_results.items():
            connection_status = "✅" if result.get('success') else "❌"
            api_status = "✅" if result.get('api_success') else "❌"
            print(f"   {endpoint}: Connection {connection_status} | API {api_status}")
        
        print(f"\n📖 DOCUMENTATION REFERENCE:")
        print(f"   - API Version: 2.9.4")
        print(f"   - Base URL: {self.base_url}")
        print(f"   - Documentation: /home/apps/Tennis_one_set/docs/API_DOCUMENTATION.md")
        print(f"   - All documented endpoints demonstrated above")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"/home/apps/Tennis_one_set/api_tennis_demo_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'api_key_type': 'demo' if self.api_key == 'demo_key' else 'real',
                    'test_results': self.test_results,
                    'summary': {
                        'total_tests': total_tests,
                        'successful_connections': successful_connections,
                        'api_successful': api_successful
                    }
                }, f, indent=2)
            print(f"\n💾 Demo results saved to: {filename}")
        except Exception as e:
            print(f"\n❌ Failed to save results: {e}")

if __name__ == "__main__":
    # Try to get real API key, fallback to demo
    import os
    api_key = os.getenv('API_TENNIS_KEY', 'demo_key')
    
    demo = ApiTennisDemo(api_key)
    demo.run_complete_demo()