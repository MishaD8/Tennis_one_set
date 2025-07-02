#!/usr/bin/env python3
"""
🧪 The Odds API Tennis Tester
Тестируем твой The Odds API для получения реальных коэффициентов
"""

import requests
import json
from datetime import datetime
from typing import Dict, List

class TheOddsAPITester:
    """Тестер для The Odds API"""
    
    def __init__(self, api_key: str = "a1b20d709d4bacb2d95ddab880f91009"):
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4"
        
    def test_connection(self) -> bool:
        """Тест подключения к API"""
        print("🔌 Testing connection to The Odds API...")
        
        try:
            response = requests.get(
                f"{self.base_url}/sports",
                params={'apiKey': self.api_key},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Connection successful! Found {len(data)} sports")
                return True
            elif response.status_code == 401:
                print("❌ Invalid API key! Get your key from: https://the-odds-api.com/")
                return False
            else:
                print(f"⚠️ API returned status {response.status_code}")
                return False
                
        except requests.RequestException as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def get_available_sports(self) -> List[Dict]:
        """Получение доступных видов спорта"""
        print("\n🏃 Getting available sports...")
        
        try:
            response = requests.get(
                f"{self.base_url}/sports",
                params={'apiKey': self.api_key}
            )
            
            if response.status_code == 200:
                sports = response.json()
                tennis_sports = [s for s in sports if 'tennis' in s.get('key', '').lower()]
                
                print(f"📊 Found {len(sports)} total sports")
                print(f"🎾 Found {len(tennis_sports)} tennis-related sports:")
                
                for sport in tennis_sports:
                    print(f"  • {sport.get('key')} - {sport.get('title')}")
                
                return tennis_sports
            else:
                print(f"❌ Failed to get sports: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return []
    
    def get_tennis_odds(self, sport_key: str = "tennis") -> Dict:
        """Получение коэффициентов на теннис"""
        print(f"\n🎾 Getting tennis odds for sport: {sport_key}")
        
        try:
            params = {
                'apiKey': self.api_key,
                'regions': 'us,uk,eu,au',  # Расширяем регионы
                'markets': 'h2h',  # Head-to-head (winner)
                'oddsFormat': 'decimal',
                'dateFormat': 'iso'
            }
            
            response = requests.get(
                f"{self.base_url}/sports/{sport_key}/odds",
                params=params
            )
            
            print(f"🔗 Request URL: {response.url}")
            print(f"📡 Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success! Found {len(data)} matches")
                
                # Показываем первые несколько матчей
                for i, match in enumerate(data[:3], 1):
                    print(f"\n🎾 Match {i}:")
                    print(f"   🏠 Home: {match.get('home_team')}")
                    print(f"   🏃 Away: {match.get('away_team')}")
                    print(f"   📅 Start: {match.get('commence_time')}")
                    print(f"   📊 Bookmakers: {len(match.get('bookmakers', []))}")
                    
                    # Показываем коэффициенты
                    for bookmaker in match.get('bookmakers', [])[:2]:
                        print(f"     💰 {bookmaker.get('title')}:")
                        for market in bookmaker.get('markets', []):
                            if market.get('key') == 'h2h':
                                for outcome in market.get('outcomes', []):
                                    print(f"       {outcome.get('name')}: {outcome.get('price')}")
                
                return data
            
            elif response.status_code == 422:
                print("⚠️ Sport not found or no active matches")
                print("💡 Try: 'tennis_wta_aus_open' or check available sports")
                return {}
            
            elif response.status_code == 401:
                print("❌ Invalid API key")
                return {}
            
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                return {}
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return {}
    
    def get_usage_info(self) -> Dict:
        """Получение информации об использовании API"""
        print("\n📊 Checking API usage...")
        
        try:
            # The Odds API возвращает usage info в заголовках
            response = requests.get(
                f"{self.base_url}/sports",
                params={'apiKey': self.api_key}
            )
            
            headers = response.headers
            usage_info = {
                'requests_remaining': headers.get('x-requests-remaining'),
                'requests_used': headers.get('x-requests-used'),
                'requests_limit': headers.get('x-requests-limit')
            }
            
            print(f"📈 Requests used: {usage_info['requests_used']}")
            print(f"📉 Requests remaining: {usage_info['requests_remaining']}")
            print(f"📊 Requests limit: {usage_info['requests_limit']}")
            
            return usage_info
            
        except Exception as e:
            print(f"❌ Error getting usage info: {e}")
            return {}
    
    def test_specific_tennis_sports(self) -> None:
        """Тест специфических теннисных видов спорта"""
        print("\n🎯 Testing specific tennis sports...")
        
        # Известные теннисные sport_keys в The Odds API
        tennis_keys = [
            'tennis',
            'tennis_wta',
            'tennis_atp',
            'tennis_wta_aus_open',
            'tennis_atp_aus_open'
        ]
        
        for sport_key in tennis_keys:
            print(f"\n🔍 Testing: {sport_key}")
            try:
                response = requests.get(
                    f"{self.base_url}/sports/{sport_key}/odds",
                    params={
                        'apiKey': self.api_key,
                        'regions': 'us,uk',
                        'markets': 'h2h',
                        'oddsFormat': 'decimal'
                    },
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"  ✅ {sport_key}: {len(data)} matches")
                elif response.status_code == 422:
                    print(f"  ⚠️ {sport_key}: No matches or out of season")
                else:
                    print(f"  ❌ {sport_key}: Error {response.status_code}")
                    
            except Exception as e:
                print(f"  ❌ {sport_key}: {e}")
    
    def integration_test(self) -> Dict:
        """Полный интеграционный тест"""
        print("🚀 STARTING FULL INTEGRATION TEST")
        print("=" * 50)
        
        results = {
            'connection': False,
            'sports_available': 0,
            'tennis_sports': 0,
            'matches_found': 0,
            'bookmakers_count': 0,
            'usage_info': {},
            'status': 'FAILED'
        }
        
        # 1. Тест подключения
        if not self.test_connection():
            return results
        results['connection'] = True
        
        # 2. Получение видов спорта
        sports = self.get_available_sports()
        results['sports_available'] = len(sports)
        results['tennis_sports'] = len([s for s in sports if 'tennis' in s.get('key', '')])
        
        # 3. Получение коэффициентов
        odds_data = self.get_tennis_odds()
        if odds_data:
            results['matches_found'] = len(odds_data)
            if odds_data:
                bookmakers = set()
                for match in odds_data:
                    for bm in match.get('bookmakers', []):
                        bookmakers.add(bm.get('key'))
                results['bookmakers_count'] = len(bookmakers)
        
        # 4. Информация об использовании
        results['usage_info'] = self.get_usage_info()
        
        # 5. Тест специфических видов спорта
        self.test_specific_tennis_sports()
        
        # Определяем общий статус
        if results['connection'] and results['matches_found'] > 0:
            results['status'] = 'SUCCESS'
        elif results['connection']:
            results['status'] = 'CONNECTED_NO_MATCHES'
        
        return results

def main():
    """Главная функция тестирования"""
    print("🎾 THE ODDS API TENNIS TESTER")
    print("=" * 50)
    print("🔑 Set your API key below:")
    print("💡 Get free key: https://the-odds-api.com/")
    print("📊 Free tier: 500 requests/month")
    print("=" * 50)
    
    # ЗАМЕНИ НА СВОЙ API КЛЮЧ
    api_key = "YOUR_ODDS_API_KEY_HERE"
    
    if api_key == "YOUR_ODDS_API_KEY_HERE":
        print("❌ Please set your real API key!")
        print("🔗 Get it from: https://the-odds-api.com/")
        return
    
    # Создаем тестер
    tester = TheOddsAPITester(api_key)
    
    # Запускаем полный тест
    results = tester.integration_test()
    
    # Показываем итоги
    print("\n" + "=" * 50)
    print("📋 INTEGRATION TEST RESULTS")
    print("=" * 50)
    print(f"🔌 Connection: {'✅' if results['connection'] else '❌'}")
    print(f"🏃 Sports available: {results['sports_available']}")
    print(f"🎾 Tennis sports: {results['tennis_sports']}")
    print(f"⚽ Matches found: {results['matches_found']}")
    print(f"💰 Bookmakers: {results['bookmakers_count']}")
    print(f"📊 Status: {results['status']}")
    
    if results['usage_info']:
        usage = results['usage_info']
        print(f"📈 API Usage: {usage.get('requests_used', 'N/A')}/{usage.get('requests_limit', 'N/A')}")
    
    print("\n🎯 NEXT STEPS:")
    if results['status'] == 'SUCCESS':
        print("✅ The Odds API is working!")
        print("🔧 Update your config.json:")
        print(f'   "the_odds_api": {{"enabled": true, "api_key": "{api_key[:8]}..."}}')
        print("🚀 Run your tennis system!")
    elif results['status'] == 'CONNECTED_NO_MATCHES':
        print("⚠️ API connected but no tennis matches available")
        print("💡 Tennis might be out of season or use different sport_key")
        print("🔍 Try different sport keys like 'tennis_wta' or 'tennis_atp'")
    else:
        print("❌ API connection failed")
        print("🔑 Check your API key")
        print("🌐 Check internet connection")

if __name__ == "__main__":
    main()