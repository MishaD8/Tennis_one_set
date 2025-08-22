#!/usr/bin/env python3
"""
Dynamic Tennis Rankings API Integration
Replaces hardcoded rankings with live API-driven data
"""

import os
import logging
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from functools import lru_cache
import threading

logger = logging.getLogger(__name__)

class DynamicTennisRankings:
    """Dynamic tennis rankings fetcher with caching and fallback"""
    
    def __init__(self):
        self.cache_duration_hours = 24  # Cache rankings for 24 hours
        self.cache_file_atp = "cache/atp_rankings.json"
        self.cache_file_wta = "cache/wta_rankings.json"
        self.api_timeout = 10
        self.lock = threading.Lock()
        
        # API configuration
        self.rapidapi_key = os.getenv('RAPIDAPI_KEY', '')
        self.tennis_api_key = os.getenv('TENNIS_API_KEY', '')
        
        # Ensure cache directory exists
        os.makedirs('cache', exist_ok=True)
        
        # Fallback rankings (minimal set for critical matches) - CORRECTED
        self.fallback_rankings = {
            "atp": {
                "jannik sinner": {"rank": 1, "points": 11830, "age": 23},
                "carlos alcaraz": {"rank": 2, "points": 8580, "age": 21},
                "alexander zverev": {"rank": 3, "points": 7915, "age": 27},
                "daniil medvedev": {"rank": 4, "points": 6230, "age": 28},
                "novak djokovic": {"rank": 5, "points": 5560, "age": 37},
                "andrey rublev": {"rank": 6, "points": 4805, "age": 26},
                "flavio cobolli": {"rank": 32, "points": 1456, "age": 22},
                "brandon nakashima": {"rank": 45, "points": 1255, "age": 23},
            },
            "wta": {
                "aryna sabalenka": {"rank": 1, "points": 9706, "age": 26},
                "iga swiatek": {"rank": 2, "points": 8370, "age": 23},
                "coco gauff": {"rank": 3, "points": 6530, "age": 20},
                "jessica pegula": {"rank": 4, "points": 5945, "age": 30},
                "elena rybakina": {"rank": 5, "points": 5471, "age": 25},
                
                # CORRECTED RANKINGS for players mentioned in TODO.md  
                "linda noskova": {"rank": 23, "points": 1650, "age": 19},
                "l. noskova": {"rank": 23, "points": 1650, "age": 19},
                "noskova": {"rank": 23, "points": 1650, "age": 19},
                "ekaterina alexandrova": {"rank": 14, "points": 2875, "age": 29},
                "e. alexandrova": {"rank": 14, "points": 2875, "age": 29},
                "alexandrova": {"rank": 14, "points": 2875, "age": 29},
                "ajla tomljanovic": {"rank": 84, "points": 790, "age": 31},
                "a. tomljanovic": {"rank": 84, "points": 790, "age": 31},
                "tomljanovic": {"rank": 84, "points": 790, "age": 31},
                
                # Other WTA players
                "renata zarazua": {"rank": 80, "points": 825, "age": 26},
                "amanda anisimova": {"rank": 35, "points": 1456, "age": 23},
            }
        }
        
        logger.info("Dynamic tennis rankings system initialized")
    
    def _load_cached_rankings(self, tour: str) -> Optional[Dict]:
        """Load cached rankings from file"""
        cache_file = self.cache_file_atp if tour.lower() == 'atp' else self.cache_file_wta
        
        try:
            if not os.path.exists(cache_file):
                return None
                
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            # Check if cache is still valid
            cached_time = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - cached_time < timedelta(hours=self.cache_duration_hours):
                logger.info(f"Using cached {tour.upper()} rankings from {cached_time}")
                return data['rankings']
            else:
                logger.info(f"Cached {tour.upper()} rankings expired")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to load cached {tour.upper()} rankings: {e}")
            return None
    
    def _save_cached_rankings(self, tour: str, rankings: Dict):
        """Save rankings to cache file"""
        cache_file = self.cache_file_atp if tour.lower() == 'atp' else self.cache_file_wta
        
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'rankings': rankings,
                'source': 'api'
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
            logger.info(f"Cached {tour.upper()} rankings saved ({len(rankings)} players)")
            
        except Exception as e:
            logger.error(f"Failed to save cached {tour.upper()} rankings: {e}")
    
    def _fetch_rankings_from_rapidapi(self, tour: str) -> Optional[Dict]:
        """Fetch rankings from RapidAPI tennis API"""
        if not self.rapidapi_key:
            logger.warning("RapidAPI key not configured")
            return None
        
        try:
            # RapidAPI tennis rankings endpoint
            url = f"https://tennis-live-data.p.rapidapi.com/rankings/{tour.lower()}"
            
            headers = {
                "X-RapidAPI-Key": self.rapidapi_key,
                "X-RapidAPI-Host": "tennis-live-data.p.rapidapi.com"
            }
            
            response = requests.get(url, headers=headers, timeout=self.api_timeout)
            
            if response.status_code == 200:
                data = response.json()
                rankings = {}
                
                # Parse the response format (adjust based on actual API response)
                if 'results' in data:
                    for player in data['results'][:100]:  # Top 100 players
                        name = f"{player.get('player', {}).get('firstname', '')} {player.get('player', {}).get('lastname', '')}".strip().lower()
                        if name:
                            rankings[name] = {
                                'rank': player.get('ranking', 999),
                                'points': player.get('points', 0),
                                'age': player.get('player', {}).get('age', 25)
                            }
                
                if rankings:
                    logger.info(f"Fetched {len(rankings)} {tour.upper()} players from RapidAPI")
                    return rankings
                else:
                    logger.warning(f"No valid {tour.upper()} rankings data from RapidAPI")
                    return None
                    
            else:
                logger.warning(f"RapidAPI returned status {response.status_code} for {tour.upper()} rankings")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch {tour.upper()} rankings from RapidAPI: {e}")
            return None
    
    def _fetch_rankings_from_tennis_api(self, tour: str) -> Optional[Dict]:
        """Fetch rankings from alternative tennis API"""
        if not self.tennis_api_key:
            logger.warning("Tennis API key not configured")
            return None
        
        try:
            # Alternative tennis API endpoint (adjust URL based on actual API)
            url = f"https://api.tennisdata.net/v1/rankings/{tour.lower()}"
            
            headers = {
                "Authorization": f"Bearer {self.tennis_api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(url, headers=headers, timeout=self.api_timeout)
            
            if response.status_code == 200:
                data = response.json()
                rankings = {}
                
                # Parse the response (adjust based on actual API format)
                for player in data.get('rankings', [])[:100]:
                    name = player.get('name', '').strip().lower()
                    if name:
                        rankings[name] = {
                            'rank': player.get('rank', 999),
                            'points': player.get('points', 0),
                            'age': player.get('age', 25)
                        }
                
                if rankings:
                    logger.info(f"Fetched {len(rankings)} {tour.upper()} players from Tennis API")
                    return rankings
                else:
                    logger.warning(f"No valid {tour.upper()} rankings data from Tennis API")
                    return None
                    
            else:
                logger.warning(f"Tennis API returned status {response.status_code} for {tour.upper()} rankings")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch {tour.upper()} rankings from Tennis API: {e}")
            return None
    
    def get_live_rankings(self, tour: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Get live rankings with caching and fallback"""
        with self.lock:
            tour_lower = tour.lower()
            
            # Try cache first unless force refresh
            if not force_refresh:
                cached = self._load_cached_rankings(tour_lower)
                if cached:
                    return cached
            
            logger.info(f"Fetching fresh {tour.upper()} rankings from APIs...")
            
            # Try RapidAPI first
            rankings = self._fetch_rankings_from_rapidapi(tour_lower)
            
            # Try alternative API if RapidAPI fails
            if not rankings:
                rankings = self._fetch_rankings_from_tennis_api(tour_lower)
            
            # If both APIs fail, use fallback
            if not rankings:
                logger.warning(f"All APIs failed for {tour.upper()} rankings, using fallback data")
                rankings = self.fallback_rankings.get(tour_lower, {})
            else:
                # Save successful API fetch to cache
                self._save_cached_rankings(tour_lower, rankings)
            
            return rankings
    
    def get_player_ranking(self, player_name: str, tour: str = None) -> Dict[str, Any]:
        """Get specific player ranking with automatic tour detection"""
        player_lower = player_name.lower().strip()
        
        # If tour is specified, search only that tour
        if tour:
            rankings = self.get_live_rankings(tour)
            if player_lower in rankings:
                return {"tour": tour.lower(), **rankings[player_lower]}
        
        # Search both tours
        for tour_name in ['atp', 'wta']:
            rankings = self.get_live_rankings(tour_name)
            if player_lower in rankings:
                return {"tour": tour_name, **rankings[player_lower]}
        
        # Partial name matching
        for tour_name in ['atp', 'wta']:
            rankings = self.get_live_rankings(tour_name)
            for known_player, data in rankings.items():
                if any(part in known_player for part in player_lower.split()):
                    return {"tour": tour_name, **data}
        
        # Return default if not found
        logger.warning(f"Player '{player_name}' not found in rankings, using default")
        return {"tour": "unknown", "rank": 100, "points": 500, "age": 25}
    
    @lru_cache(maxsize=200)  # Cache recent player lookups
    def get_player_data_cached(self, player_name: str) -> Dict[str, Any]:
        """Cached version of get_player_ranking for frequent lookups"""
        return self.get_player_ranking(player_name)
    
    def refresh_all_rankings(self):
        """Force refresh all rankings from APIs"""
        logger.info("Force refreshing all rankings...")
        
        results = {}
        for tour in ['atp', 'wta']:
            try:
                rankings = self.get_live_rankings(tour, force_refresh=True)
                results[tour] = {
                    'success': True,
                    'player_count': len(rankings),
                    'last_updated': datetime.now().isoformat()
                }
            except Exception as e:
                results[tour] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def get_ranking_stats(self) -> Dict[str, Any]:
        """Get statistics about current rankings cache"""
        stats = {
            'atp': {'cached': False, 'age_hours': None, 'player_count': 0},
            'wta': {'cached': False, 'age_hours': None, 'player_count': 0},
            'api_keys_configured': {
                'rapidapi': bool(self.rapidapi_key),
                'tennis_api': bool(self.tennis_api_key)
            }
        }
        
        for tour in ['atp', 'wta']:
            cache_file = self.cache_file_atp if tour == 'atp' else self.cache_file_wta
            
            try:
                if os.path.exists(cache_file):
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                    
                    cached_time = datetime.fromisoformat(data['timestamp'])
                    age_hours = (datetime.now() - cached_time).total_seconds() / 3600
                    
                    stats[tour] = {
                        'cached': True,
                        'age_hours': round(age_hours, 2),
                        'player_count': len(data.get('rankings', {})),
                        'last_updated': data['timestamp'],
                        'source': data.get('source', 'unknown')
                    }
            except Exception as e:
                logger.error(f"Error reading cache stats for {tour}: {e}")
        
        return stats

# Global instance for use across the application
dynamic_rankings = DynamicTennisRankings()

def get_dynamic_player_ranking(player_name: str, tour: str = None) -> Dict[str, Any]:
    """Convenience function to get player ranking"""
    return dynamic_rankings.get_player_ranking(player_name, tour)

def refresh_tennis_rankings():
    """Convenience function to refresh rankings"""
    return dynamic_rankings.refresh_all_rankings()

def get_rankings_status():
    """Convenience function to get rankings status"""
    return dynamic_rankings.get_ranking_stats()