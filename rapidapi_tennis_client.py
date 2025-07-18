#!/usr/bin/env python3
"""
RapidAPI Tennis Data Client
High-quality tennis data from RapidAPI with rate limiting
"""

import os
import time
import json
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from config_loader import load_secure_config
import logging

logger = logging.getLogger(__name__)

class RapidAPIRateLimiter:
    """Rate limiter for RapidAPI requests (50 requests per day)"""
    
    def __init__(self, daily_limit: int = 50):
        self.daily_limit = daily_limit
        self.request_log_file = "rapidapi_requests.json"
        self.requests_today = 0
        self.last_reset_date = None
        self._load_request_log()
    
    def _load_request_log(self):
        """Load request log from file"""
        try:
            if os.path.exists(self.request_log_file):
                with open(self.request_log_file, 'r') as f:
                    data = json.load(f)
                    self.requests_today = data.get('requests_today', 0)
                    self.last_reset_date = data.get('last_reset_date')
                    
                    # Reset if new day
                    today = datetime.now().strftime('%Y-%m-%d')
                    if self.last_reset_date != today:
                        self.requests_today = 0
                        self.last_reset_date = today
                        self._save_request_log()
        except Exception as e:
            logger.warning(f"Could not load request log: {e}")
            self.requests_today = 0
            self.last_reset_date = datetime.now().strftime('%Y-%m-%d')
    
    def _save_request_log(self):
        """Save request log to file"""
        try:
            with open(self.request_log_file, 'w') as f:
                json.dump({
                    'requests_today': self.requests_today,
                    'last_reset_date': self.last_reset_date
                }, f)
        except Exception as e:
            logger.warning(f"Could not save request log: {e}")
    
    def can_make_request(self) -> bool:
        """Check if we can make a request within daily limit"""
        return self.requests_today < self.daily_limit
    
    def record_request(self):
        """Record that a request was made"""
        self.requests_today += 1
        self._save_request_log()
    
    def get_remaining_requests(self) -> int:
        """Get number of remaining requests today"""
        return max(0, self.daily_limit - self.requests_today)

class RapidAPITennisClient:
    """Client for RapidAPI Tennis data with rate limiting and caching"""
    
    def __init__(self):
        self.config = load_secure_config()
        self.rapidapi_config = self.config.get('data_sources', {}).get('rapidapi_tennis', {})
        
        if not self.rapidapi_config.get('enabled', False):
            raise Exception("RapidAPI Tennis is not enabled in configuration")
        
        self.api_key = self.rapidapi_config.get('api_key', '')
        self.base_url = self.rapidapi_config.get('base_url', '')
        self.host = self.rapidapi_config.get('host', '')
        self.username = self.rapidapi_config.get('username', '')
        self.daily_limit = self.rapidapi_config.get('daily_limit', 50)
        self.endpoints = self.rapidapi_config.get('endpoints', {})
        self.cache_minutes = self.rapidapi_config.get('cache_minutes', 60)
        
        if not self.api_key or self.api_key.startswith('MISSING_'):
            raise Exception("RapidAPI key not configured. Set RAPIDAPI_KEY environment variable.")
        
        self.rate_limiter = RapidAPIRateLimiter(self.daily_limit)
        self.cache = {}
        
        # Headers for RapidAPI
        self.headers = {
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': self.host,
            'User-Agent': f'Tennis-Predictor/{self.username}'
        }
        
        logger.info(f"RapidAPI Tennis client initialized. Remaining requests: {self.rate_limiter.get_remaining_requests()}")
    
    def _get_cache_key(self, endpoint: str, params: Dict = None) -> str:
        """Generate cache key for request"""
        cache_key = f"{endpoint}"
        if params:
            sorted_params = sorted(params.items())
            cache_key += "_" + "_".join([f"{k}={v}" for k, v in sorted_params])
        return cache_key
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid"""
        if not cache_entry:
            return False
        
        cached_time = datetime.fromisoformat(cache_entry['timestamp'])
        expiry_time = cached_time + timedelta(minutes=self.cache_minutes)
        return datetime.now() < expiry_time
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make HTTP request to RapidAPI with rate limiting"""
        
        # Check rate limit
        if not self.rate_limiter.can_make_request():
            logger.error(f"Daily rate limit ({self.daily_limit}) exceeded for RapidAPI")
            return None
        
        # Check cache first
        cache_key = self._get_cache_key(endpoint, params)
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            logger.info(f"Using cached data for {endpoint}")
            return self.cache[cache_key]['data']
        
        # Make request
        url = f"{self.base_url}{endpoint}"
        
        try:
            logger.info(f"Making RapidAPI request to {endpoint} (Remaining: {self.rate_limiter.get_remaining_requests()})")
            
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            
            # Record the request
            self.rate_limiter.record_request()
            
            if response.status_code == 200:
                data = response.json()
                
                # Cache the response
                self.cache[cache_key] = {
                    'data': data,
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"Successfully fetched data from {endpoint}")
                return data
            
            elif response.status_code == 429:
                logger.error("Rate limit exceeded by RapidAPI")
                return None
            
            else:
                logger.error(f"RapidAPI request failed: {response.status_code} - {response.text}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Request error for {endpoint}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error for {endpoint}: {e}")
            return None
    
    def get_wta_rankings(self) -> Optional[List[Dict]]:
        """Get WTA live rankings"""
        endpoint = self.endpoints.get('wta_rankings', '/rankings/wta/live')
        data = self._make_request(endpoint)
        
        if data and 'rankings' in data:
            logger.info(f"Retrieved {len(data['rankings'])} WTA rankings")
            return data['rankings']
        
        return None
    
    def get_atp_rankings(self) -> Optional[List[Dict]]:
        """Get ATP live rankings"""
        endpoint = self.endpoints.get('atp_rankings', '/rankings/atp/live')
        data = self._make_request(endpoint)
        
        if data and 'rankings' in data:
            logger.info(f"Retrieved {len(data['rankings'])} ATP rankings")
            return data['rankings']
        
        return None
    
    def get_live_matches(self, tour: str = 'both') -> Optional[List[Dict]]:
        """Get live matches for ATP/WTA
        
        Args:
            tour: 'atp', 'wta', or 'both'
        """
        matches = []
        
        if tour in ['atp', 'both']:
            atp_endpoint = self.endpoints.get('atp_matches', '/matches/live')
            atp_data = self._make_request(atp_endpoint)
            if atp_data and 'events' in atp_data:
                for match in atp_data['events']:
                    match['tour'] = 'ATP'
                    matches.append(match)
        
        if tour in ['wta', 'both']:
            wta_endpoint = self.endpoints.get('wta_matches', '/matches/live')
            wta_data = self._make_request(wta_endpoint)
            if wta_data and 'events' in wta_data:
                for match in wta_data['events']:
                    match['tour'] = 'WTA'
                    matches.append(match)
        
        if matches:
            logger.info(f"Retrieved {len(matches)} live matches")
            return matches
        
        return None
    
    def get_player_details(self, player_id: str) -> Optional[Dict]:
        """Get detailed player information"""
        endpoint = self.endpoints.get('player_details', '/player/{id}').format(id=player_id)
        data = self._make_request(endpoint)
        
        if data and 'player' in data:
            logger.info(f"Retrieved player details for ID {player_id}")
            return data['player']
        
        return None
    
    def get_tournament_details(self, tournament_id: str) -> Optional[Dict]:
        """Get detailed tournament information"""
        endpoint = self.endpoints.get('tournament_details', '/tournament/{id}').format(id=tournament_id)
        data = self._make_request(endpoint)
        
        if data and 'tournament' in data:
            logger.info(f"Retrieved tournament details for ID {tournament_id}")
            return data['tournament']
        
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get client status and remaining requests"""
        return {
            'enabled': self.rapidapi_config.get('enabled', False),
            'daily_limit': self.daily_limit,
            'requests_used_today': self.rate_limiter.requests_today,
            'requests_remaining': self.rate_limiter.get_remaining_requests(),
            'last_reset_date': self.rate_limiter.last_reset_date,
            'cache_size': len(self.cache),
            'cache_minutes': self.cache_minutes
        }
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        logger.info("RapidAPI cache cleared")

def test_rapidapi_client():
    """Test function for RapidAPI client"""
    try:
        client = RapidAPITennisClient()
        
        print(f"Client Status: {client.get_status()}")
        
        # Test WTA rankings
        print("\n=== Testing WTA Rankings ===")
        wta_rankings = client.get_wta_rankings()
        if wta_rankings:
            print(f"Retrieved {len(wta_rankings)} WTA rankings")
            if wta_rankings:
                print(f"Top player: {wta_rankings[0]}")
        else:
            print("No WTA rankings data retrieved")
        
        # Test ATP rankings
        print("\n=== Testing ATP Rankings ===")
        atp_rankings = client.get_atp_rankings()
        if atp_rankings:
            print(f"Retrieved {len(atp_rankings)} ATP rankings")
            if atp_rankings:
                print(f"Top player: {atp_rankings[0]}")
        else:
            print("No ATP rankings data retrieved")
        
        # Test live matches
        print("\n=== Testing Live Matches ===")
        live_matches = client.get_live_matches()
        if live_matches:
            print(f"Retrieved {len(live_matches)} live matches")
            for match in live_matches[:3]:  # Show first 3
                print(f"Match: {match}")
        else:
            print("No live matches data retrieved")
        
        print(f"\nFinal Status: {client.get_status()}")
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_rapidapi_client()