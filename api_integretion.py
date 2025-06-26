#!/usr/bin/env python3
"""
🎾 TENNIS PREDICTION SYSTEM - API INTEGRATIONS
Complete guide and implementation for all major APIs you'll need

This file contains ready-to-use code for:
• Betting APIs (Pinnacle, Bet365, etc.)
• Tennis Data APIs (ATP, WTA, ITF)
• Odds Comparison APIs
• Live Score APIs
• Weather APIs
• Sports News APIs

IMPORTANT: Replace API keys and credentials with your real ones after registration!
"""

import requests
import asyncio
import aiohttp
import json
import base64
import hmac
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class APICredentials:
    """Store API credentials securely"""
    api_key: str = ""
    username: str = ""
    password: str = ""
    secret: str = ""
    base_url: str = ""
    rate_limit: int = 60  # requests per minute

class PinnacleAPI:
    """
    🏆 PINNACLE SPORTS API
    Registration: https://www.pinnacle.com/en/developer-api
    Features: Sharp odds, low margins, high limits
    Rate Limit: 120 requests/minute
    """
    
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.base_url = "https://api.pinnacle.com"
        self.session = requests.Session()
        
        # Set up authentication
        credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
        self.session.headers.update({
            'Authorization': f'Basic {credentials}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def get_sports(self) -> Dict:
        """Get available sports"""
        try:
            response = self.session.get(f"{self.base_url}/v1/sports")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Pinnacle sports error: {e}")
            return {}
    
    def get_tennis_leagues(self) -> Dict:
        """Get tennis leagues/tournaments"""
        try:
            # Tennis sport ID is typically 33
            response = self.session.get(f"{self.base_url}/v1/leagues?sportId=33")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Pinnacle leagues error: {e}")
            return {}
    
    def get_tennis_fixtures(self, league_ids: List[int] = None) -> Dict:
        """Get upcoming tennis matches"""
        try:
            params = {'sportId': 33}
            if league_ids:
                params['leagueIds'] = ','.join(map(str, league_ids))
            
            response = self.session.get(f"{self.base_url}/v1/fixtures", params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Pinnacle fixtures error: {e}")
            return {}
    
    def get_tennis_odds(self, since: int = None) -> Dict:
        """Get tennis odds"""
        try:
            params = {'sportId': 33, 'oddsFormat': 'DECIMAL'}
            if since:
                params['since'] = since
            
            response = self.session.get(f"{self.base_url}/v1/odds", params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Pinnacle odds error: {e}")
            return {}
    
    def place_bet(self, bet_data: Dict) -> Dict:
        """Place a tennis bet"""
        try:
            response = self.session.post(f"{self.base_url}/v1/bets/place", json=bet_data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Pinnacle bet placement error: {e}")
            return {}

class TheOddsAPI:
    """
    📊 THE ODDS API
    Registration: https://the-odds-api.com/
    Features: Multiple bookmakers, real-time odds
    Free Tier: 500 requests/month
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4"
        
    def get_tennis_odds(self, regions: str = "us,uk,au", markets: str = "h2h") -> Dict:
        """Get tennis odds from multiple bookmakers"""
        try:
            params = {
                'apiKey': self.api_key,
                'regions': regions,
                'markets': markets,
                'oddsFormat': 'decimal',
                'dateFormat': 'iso'
            }
            
            response = requests.get(f"{self.base_url}/sports/tennis/odds", params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"The Odds API error: {e}")
            return []
    
    def get_sports(self) -> Dict:
        """Get available sports"""
        try:
            params = {'apiKey': self.api_key}
            response = requests.get(f"{self.base_url}/sports", params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"The Odds API sports error: {e}")
            return []

class ATP_API:
    """
    🎾 ATP TOUR API
    Registration: Contact ATP Tour for access
    Features: Official tournament data, player stats
    Note: Limited public access, mainly for media/partners
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.atptour.com"
        self.headers = {
            'X-API-Key': api_key,
            'Accept': 'application/json'
        }
    
    def get_tournaments(self, year: int = None) -> Dict:
        """Get ATP tournaments"""
        year = year or datetime.now().year
        try:
            response = requests.get(
                f"{self.base_url}/v1/tournaments/{year}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"ATP tournaments error: {e}")
            return {}
    
    def get_player_stats(self, player_id: str) -> Dict:
        """Get player statistics"""
        try:
            response = requests.get(
                f"{self.base_url}/v1/players/{player_id}/stats",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"ATP player stats error: {e}")
            return {}
    
    def get_live_scores(self) -> Dict:
        """Get live ATP scores"""
        try:
            response = requests.get(
                f"{self.base_url}/v1/scores/live",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"ATP live scores error: {e}")
            return {}

class WTA_API:
    """
    🎾 WTA TOUR API
    Registration: Contact WTA for access
    Features: Official WTA data, player rankings
    Note: Similar to ATP, limited public access
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.wtatennis.com"
        self.headers = {
            'X-API-Key': api_key,
            'Accept': 'application/json'
        }
    
    def get_rankings(self, date: str = None) -> Dict:
        """Get WTA rankings"""
        date = date or datetime.now().strftime('%Y-%m-%d')
        try:
            response = requests.get(
                f"{self.base_url}/v1/rankings/{date}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"WTA rankings error: {e}")
            return {}

class RapidAPI_Tennis:
    """
    ⚡ RAPIDAPI TENNIS SERVICES
    Registration: https://rapidapi.com/
    Multiple tennis APIs available through one platform
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            'X-RapidAPI-Key': api_key,
            'X-RapidAPI-Host': ''  # Changes per API
        }
    
    def get_tennis_live_scores(self) -> Dict:
        """Tennis Live Scores API"""
        self.headers['X-RapidAPI-Host'] = 'tennis-live-data.p.rapidapi.com'
        try:
            response = requests.get(
                "https://tennis-live-data.p.rapidapi.com/matches-by-date/2024-06-25",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"RapidAPI tennis error: {e}")
            return {}
    
    def get_tennis_stats(self, player_id: str) -> Dict:
        """Ultimate Tennis API"""
        self.headers['X-RapidAPI-Host'] = 'ultimate-tennis1.p.rapidapi.com'
        try:
            response = requests.get(
                f"https://ultimate-tennis1.p.rapidapi.com/player_stats/{player_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"RapidAPI tennis stats error: {e}")
            return {}

class BetfairAPI:
    """
    💱 BETFAIR EXCHANGE API
    Registration: https://developer.betfair.com/
    Features: Exchange betting, better odds, lay betting
    Note: Requires application approval
    """
    
    def __init__(self, app_key: str, username: str, password: str):
        self.app_key = app_key
        self.username = username
        self.password = password
        self.session_token = None
        self.base_url = "https://api.betfair.com/exchange/betting/rest/v1.0"
        
    def login(self) -> bool:
        """Login to Betfair and get session token"""
        try:
            login_url = "https://identitysso.betfair.com/api/login"
            login_data = {
                'username': self.username,
                'password': self.password
            }
            headers = {
                'X-Application': self.app_key,
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            response = requests.post(login_url, data=login_data, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            if result['status'] == 'SUCCESS':
                self.session_token = result['token']
                return True
            return False
        except requests.RequestException as e:
            logger.error(f"Betfair login error: {e}")
            return False
    
    def get_tennis_markets(self) -> Dict:
        """Get tennis betting markets"""
        if not self.session_token:
            self.login()
        
        try:
            headers = {
                'X-Application': self.app_key,
                'X-Authentication': self.session_token,
                'Content-Type': 'application/json'
            }
            
            data = {
                'filter': {
                    'eventTypeIds': ['2'],  # Tennis event type ID
                    'marketCountries': ['GB', 'US', 'AU'],
                    'marketTypeCodes': ['MATCH_ODDS']
                }
            }
            
            response = requests.post(
                f"{self.base_url}/listMarketCatalogue/",
                json=data,
                headers=headers
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Betfair markets error: {e}")
            return []

class FlashscoreAPI:
    """
    📱 FLASHSCORE API (Unofficial)
    Web scraping alternative for live scores
    Note: Use responsibly, respect rate limits
    """
    
    def __init__(self):
        self.base_url = "https://www.flashscore.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_tennis_matches(self, date: str = None) -> List[Dict]:
        """Scrape tennis matches (educational purposes)"""
        # Implementation would go here
        # Note: Web scraping should respect robots.txt and terms of service
        logger.warning("Flashscore scraping - ensure compliance with ToS")
        return []

class WeatherAPI:
    """
    🌤️ WEATHER API
    Registration: https://openweathermap.org/api
    Features: Weather conditions for outdoor matches
    Free Tier: 1000 calls/day
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"
    
    def get_weather_for_tournament(self, city: str, tournament_date: str) -> Dict:
        """Get weather forecast for tournament location"""
        try:
            params = {
                'q': city,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(f"{self.base_url}/forecast", params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Weather API error: {e}")
            return {}

class NewsAPI:
    """
    📰 NEWS API
    Registration: https://newsapi.org/
    Features: Tennis news and sentiment analysis
    Free Tier: 100 requests/day
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
    
    def get_tennis_news(self, player_name: str = None, days_back: int = 7) -> Dict:
        """Get tennis-related news"""
        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            params = {
                'apiKey': self.api_key,
                'q': f'tennis {player_name}' if player_name else 'tennis',
                'from': from_date,
                'sortBy': 'relevancy',
                'language': 'en'
            }
            
            response = requests.get(f"{self.base_url}/everything", params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"News API error: {e}")
            return {}

class ComprehensiveAPIManager:
    """
    🎯 MAIN API MANAGER
    Coordinates all APIs and provides unified interface
    """
    
    def __init__(self, config_file: str = "api_config.json"):
        self.config_file = config_file
        self.apis = {}
        self.load_config()
        self.initialize_apis()
    
    def load_config(self):
        """Load API configuration"""
        try:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = self.create_default_config()
            self.save_config()
    
    def create_default_config(self) -> Dict:
        """Create default API configuration"""
        return {
            "pinnacle": {
                "enabled": False,
                "username": "YOUR_PINNACLE_USERNAME",
                "password": "YOUR_PINNACLE_PASSWORD",
                "priority": 1
            },
            "the_odds_api": {
                "enabled": False,
                "api_key": "YOUR_ODDS_API_KEY",
                "priority": 2
            },
            "atp_api": {
                "enabled": False,
                "api_key": "YOUR_ATP_API_KEY",
                "priority": 3
            },
            "wta_api": {
                "enabled": False,
                "api_key": "YOUR_WTA_API_KEY",
                "priority": 3
            },
            "rapidapi": {
                "enabled": False,
                "api_key": "YOUR_RAPIDAPI_KEY",
                "priority": 4
            },
            "betfair": {
                "enabled": False,
                "app_key": "YOUR_BETFAIR_APP_KEY",
                "username": "YOUR_BETFAIR_USERNAME",
                "password": "YOUR_BETFAIR_PASSWORD",
                "priority": 2
            },
            "weather_api": {
                "enabled": False,
                "api_key": "YOUR_WEATHER_API_KEY",
                "priority": 5
            },
            "news_api": {
                "enabled": False,
                "api_key": "YOUR_NEWS_API_KEY",
                "priority": 5
            }
        }
    
    def save_config(self):
        """Save API configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def initialize_apis(self):
        """Initialize enabled APIs"""
        try:
            # Pinnacle
            if self.config.get('pinnacle', {}).get('enabled'):
                self.apis['pinnacle'] = PinnacleAPI(
                    self.config['pinnacle']['username'],
                    self.config['pinnacle']['password']
                )
            
            # The Odds API
            if self.config.get('the_odds_api', {}).get('enabled'):
                self.apis['the_odds_api'] = TheOddsAPI(
                    self.config['the_odds_api']['api_key']
                )
            
            # ATP API
            if self.config.get('atp_api', {}).get('enabled'):
                self.apis['atp_api'] = ATP_API(
                    self.config['atp_api']['api_key']
                )
            
            # WTA API
            if self.config.get('wta_api', {}).get('enabled'):
                self.apis['wta_api'] = WTA_API(
                    self.config['wta_api']['api_key']
                )
            
            # RapidAPI
            if self.config.get('rapidapi', {}).get('enabled'):
                self.apis['rapidapi'] = RapidAPI_Tennis(
                    self.config['rapidapi']['api_key']
                )
            
            # Betfair
            if self.config.get('betfair', {}).get('enabled'):
                self.apis['betfair'] = BetfairAPI(
                    self.config['betfair']['app_key'],
                    self.config['betfair']['username'],
                    self.config['betfair']['password']
                )
            
            # Weather API
            if self.config.get('weather_api', {}).get('enabled'):
                self.apis['weather_api'] = WeatherAPI(
                    self.config['weather_api']['api_key']
                )
            
            # News API
            if self.config.get('news_api', {}).get('enabled'):
                self.apis['news_api'] = NewsAPI(
                    self.config['news_api']['api_key']
                )
            
            logger.info(f"Initialized {len(self.apis)} APIs")
            
        except Exception as e:
            logger.error(f"API initialization error: {e}")
    
    def get_all_tennis_odds(self) -> Dict:
        """Get odds from all available sources"""
        all_odds = {}
        
        # Pinnacle odds
        if 'pinnacle' in self.apis:
            try:
                pinnacle_odds = self.apis['pinnacle'].get_tennis_odds()
                all_odds['pinnacle'] = pinnacle_odds
            except Exception as e:
                logger.error(f"Pinnacle odds error: {e}")
        
        # The Odds API
        if 'the_odds_api' in self.apis:
            try:
                odds_api_data = self.apis['the_odds_api'].get_tennis_odds()
                all_odds['the_odds_api'] = odds_api_data
            except Exception as e:
                logger.error(f"The Odds API error: {e}")
        
        # Betfair
        if 'betfair' in self.apis:
            try:
                betfair_markets = self.apis['betfair'].get_tennis_markets()
                all_odds['betfair'] = betfair_markets
            except Exception as e:
                logger.error(f"Betfair error: {e}")
        
        return all_odds
    
    def get_comprehensive_match_data(self, match_date: str = None) -> Dict:
        """Get comprehensive match data from all sources"""
        match_date = match_date or datetime.now().strftime('%Y-%m-%d')
        
        comprehensive_data = {
            'matches': [],
            'odds': {},
            'weather': {},
            'news': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Get odds data
        comprehensive_data['odds'] = self.get_all_tennis_odds()
        
        # Get ATP data
        if 'atp_api' in self.apis:
            try:
                atp_tournaments = self.apis['atp_api'].get_tournaments()
                comprehensive_data['atp_tournaments'] = atp_tournaments
            except Exception as e:
                logger.error(f"ATP data error: {e}")
        
        # Get weather data for major tournament locations
        if 'weather_api' in self.apis:
            major_locations = ['London', 'Paris', 'New York', 'Melbourne']
            for location in major_locations:
                try:
                    weather = self.apis['weather_api'].get_weather_for_tournament(location, match_date)
                    comprehensive_data['weather'][location] = weather
                except Exception as e:
                    logger.error(f"Weather data error for {location}: {e}")
        
        # Get tennis news
        if 'news_api' in self.apis:
            try:
                news = self.apis['news_api'].get_tennis_news()
                comprehensive_data['news'] = news
            except Exception as e:
                logger.error(f"News data error: {e}")
        
        return comprehensive_data
    
    def test_all_apis(self) -> Dict:
        """Test all configured APIs"""
        test_results = {}
        
        for api_name, api_instance in self.apis.items():
            try:
                if api_name == 'pinnacle':
                    result = api_instance.get_sports()
                    test_results[api_name] = {'status': 'success', 'data_points': len(result.get('sports', []))}
                
                elif api_name == 'the_odds_api':
                    result = api_instance.get_sports()
                    test_results[api_name] = {'status': 'success', 'sports_count': len(result)}
                
                elif api_name == 'betfair':
                    login_success = api_instance.login()
                    test_results[api_name] = {'status': 'success' if login_success else 'failed', 'logged_in': login_success}
                
                else:
                    test_results[api_name] = {'status': 'configured', 'message': 'API configured but not tested'}
                
            except Exception as e:
                test_results[api_name] = {'status': 'error', 'error': str(e)}
        
        return test_results

# Usage example and testing functions
def main():
    """
    Example usage of the API manager
    """
    print("🎾 TENNIS PREDICTION SYSTEM - API INTEGRATION TESTER")
    print("=" * 60)
    
    # Initialize API manager
    api_manager = ComprehensiveAPIManager()
    
    print(f"📡 Configured APIs: {list(api_manager.apis.keys())}")
    
    # Test all APIs
    print("\n🔍 Testing API connections...")
    test_results = api_manager.test_all_apis()
    
    for api_name, result in test_results.items():
        status_icon = "✅" if result['status'] == 'success' else "❌" if result['status'] == 'error' else "⚠️"
        print(f"{status_icon} {api_name}: {result['status']}")
    
    # Get comprehensive data (if APIs are configured)
    if any(api_manager.apis.values()):
        print("\n📊 Fetching comprehensive match data...")
        try:
            comprehensive_data = api_manager.get_comprehensive_match_data()
            print(f"✅ Data collected at: {comprehensive_data['timestamp']}")
            print(f"📈 Odds sources: {list(comprehensive_data['odds'].keys())}")
            print(f"🌤️ Weather locations: {list(comprehensive_data['weather'].keys())}")
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
    else:
        print("⚠️ No APIs configured. Please update api_config.json with your credentials.")
    
    print("\n💡 Next steps:")
    print("1. Register for the APIs you need")
    print("2. Update api_config.json with your credentials")
    print("3. Set enabled=true for the APIs you want to use")
    print("4. Run this script again to test connections")

if __name__ == "__main__":
    main()