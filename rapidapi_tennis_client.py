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
                    # ENHANCED FILTERING: Only professional tournaments
                    tournament = match.get('tournament', {})
                    if not self._is_professional_tournament(tournament):
                        continue  # Skip non-professional tournaments like UTR PTT
                    
                    match['tour'] = 'ATP'
                    matches.append(match)
        
        if tour in ['wta', 'both']:
            wta_endpoint = self.endpoints.get('wta_matches', '/matches/live')
            wta_data = self._make_request(wta_endpoint)
            if wta_data and 'events' in wta_data:
                for match in wta_data['events']:
                    # ENHANCED FILTERING: Only professional tournaments
                    tournament = match.get('tournament', {})
                    if not self._is_professional_tournament(tournament):
                        continue  # Skip non-professional tournaments like UTR PTT
                    
                    match['tour'] = 'WTA'
                    matches.append(match)
        
        if matches:
            logger.info(f"Retrieved {len(matches)} live matches")
            return matches
        
        return None
    
    def get_all_events(self) -> Optional[List[Dict]]:
        """Get ALL events (including scheduled matches) from all tournaments"""
        endpoint = self.endpoints.get('all_events', '/api/tennis/events')
        data = self._make_request(endpoint)
        
        if data and 'events' in data:
            logger.info(f"Retrieved {len(data['events'])} total events (live + scheduled)")
            return data['events']
        
        return None
    
    def get_scheduled_matches(self, tour: str = 'both') -> Optional[List[Dict]]:
        """Get scheduled (upcoming) matches that haven't started yet
        
        Args:
            tour: 'atp', 'wta', or 'both'
        """
        # First try to get all events
        all_events = self.get_all_events()
        if not all_events:
            return None
        
        scheduled_matches = []
        for event in all_events:
            status = event.get('status', {})
            status_type = status.get('type', '')
            
            # Filter for scheduled matches (not live, finished, etc.)
            if status_type in ['scheduled', 'notstarted', 'postponed']:
                # Apply tour filter if specified
                tournament = event.get('tournament', {})
                category = tournament.get('category', {}).get('name', '').upper()
                
                # ENHANCED FILTERING: Only professional ATP/WTA tournaments
                if not self._is_professional_tournament(tournament):
                    continue  # Skip non-professional tournaments like UTR PTT
                
                if tour == 'both' or \
                   (tour == 'atp' and 'ATP' in category) or \
                   (tour == 'wta' and 'WTA' in category):
                    event['tour'] = 'ATP' if 'ATP' in category else 'WTA'
                    scheduled_matches.append(event)
        
        if scheduled_matches:
            logger.info(f"Retrieved {len(scheduled_matches)} scheduled matches")
            return scheduled_matches
        
        return None
    
    def get_tournaments(self) -> Optional[List[Dict]]:
        """Get current active tournaments"""
        endpoint = self.endpoints.get('tournaments', '/api/tennis/tournaments')
        data = self._make_request(endpoint)
        
        if data and 'tournaments' in data:
            logger.info(f"Retrieved {len(data['tournaments'])} tournaments")
            return data['tournaments']
        
        return None
    
    def get_tournament_matches(self, tournament_id: str) -> Optional[List[Dict]]:
        """Get all matches for a specific tournament
        
        Args:
            tournament_id: The tournament ID
        """
        endpoint = self.endpoints.get('tournament_matches', '/api/tennis/tournament/{id}/matches').format(id=tournament_id)
        data = self._make_request(endpoint)
        
        if data and 'events' in data:
            logger.info(f"Retrieved {len(data['events'])} matches for tournament {tournament_id}")
            return data['events']
        
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
    
    def get_all_events(self, tour: str = 'both', include_status: List[str] = None) -> Optional[List[Dict]]:
        """Get all tennis events, with optional status filtering
        
        Args:
            tour: 'atp', 'wta', or 'both'
            include_status: List of status types to include (e.g., ['notstarted', 'inprogress', 'finished'])
        """
        matches = []
        current_time = datetime.now().timestamp()
        
        # Get data from the working live events endpoint  
        if tour in ['atp', 'both']:
            atp_endpoint = self.endpoints.get('atp_matches', '/api/tennis/events/live')
            atp_data = self._make_request(atp_endpoint)
            if atp_data and 'events' in atp_data:
                for match in atp_data['events']:
                    match['tour'] = 'ATP'
                    matches.append(match)
        
        if tour in ['wta', 'both']:
            wta_endpoint = self.endpoints.get('wta_matches', '/api/tennis/events/live')  
            wta_data = self._make_request(wta_endpoint)
            if wta_data and 'events' in wta_data:
                for match in wta_data['events']:
                    match['tour'] = 'WTA'
                    matches.append(match)
        
        # Filter by status if specified
        if include_status:
            filtered_matches = []
            for match in matches:
                match_status = match.get('status', {}).get('type', '').lower()
                if match_status in [s.lower() for s in include_status]:
                    filtered_matches.append(match)
            matches = filtered_matches
        
        if matches:
            logger.info(f"Retrieved {len(matches)} events (status filter: {include_status})")
            return matches
        
        return None
    
    def _is_professional_tournament(self, tournament: Dict) -> bool:
        """Check if tournament is ATP/WTA professional level only"""
        
        # Get tournament information
        tournament_name = tournament.get('name', '').lower()
        category = tournament.get('category', {})
        category_name = category.get('name', '').upper()
        
        # Exclude non-professional tournaments
        excluded_keywords = [
            'utr', 'ptt', 'junior', 'college', 'university', 
            'challenger', 'futures', 'itf', 'amateur',
            'qualifying', 'q1', 'q2', 'q3', 'youth',
            'exhibition', 'invitational', 'lovedale',
            'utr ptt', 'group a', 'group b', 'group c', 'group d',
            'men 03', 'women 03', 'ciguenza', 'errey', 'karnani',
            'baker', 'mihulka', 'dejanovic'
        ]
        
        # Check if tournament name contains excluded keywords
        for keyword in excluded_keywords:
            if keyword in tournament_name:
                logger.info(f"Excluding non-professional tournament: {tournament_name} (contains '{keyword}')")
                return False
        
        # Only allow specific professional categories
        professional_categories = [
            'ATP', 'WTA', 'GRAND SLAM', 'MASTERS', 'PREMIER',
            'ATP 250', 'ATP 500', 'ATP 1000', 'ATP FINALS',
            'WTA 250', 'WTA 500', 'WTA 1000', 'WTA FINALS'
        ]
        
        # Check if category is professional
        for prof_category in professional_categories:
            if prof_category in category_name:
                return True
        
        # If no professional category found, log and exclude
        logger.info(f"Excluding tournament without professional category: {tournament_name} (category: {category_name})")
        return False
    
    def get_scheduled_matches(self, tour: str = 'both') -> Optional[List[Dict]]:
        """Get scheduled/upcoming matches by filtering all events for non-live status
        
        Args:
            tour: 'atp', 'wta', or 'both'
        """
        # Try to get events that are not currently in progress
        # Common status types: 'notstarted', 'scheduled', 'postponed', 'cancelled'
        potential_scheduled_statuses = ['notstarted', 'scheduled', 'postponed']
        
        all_matches = self.get_all_events(tour=tour)
        if not all_matches:
            return None
        
        scheduled_matches = []
        current_time = datetime.now().timestamp()
        
        for match in all_matches:
            match_status = match.get('status', {}).get('type', '').lower()
            start_timestamp = match.get('startTimestamp', 0)
            
            # Include matches that:
            # 1. Have a 'notstarted' or similar status, OR
            # 2. Have a future start timestamp, OR  
            # 3. Are not currently 'inprogress' or 'finished'
            is_scheduled = (
                match_status in potential_scheduled_statuses or
                (start_timestamp > current_time) or
                match_status not in ['inprogress', 'finished', 'ended']
            )
            
            if is_scheduled:
                match['scheduling_reason'] = f"Status: {match_status}, Future: {start_timestamp > current_time}"
                scheduled_matches.append(match)
        
        if scheduled_matches:
            logger.info(f"Retrieved {len(scheduled_matches)} scheduled matches from {len(all_matches)} total events")
            return scheduled_matches
        
        logger.info("No scheduled matches found among available events")
        return None
    
    def get_matches_by_date(self, date: str, tour: str = 'both') -> Optional[List[Dict]]:
        """Get matches for a specific date (YYYY-MM-DD format)
        
        Args:
            date: Date in YYYY-MM-DD format
            tour: 'atp', 'wta', or 'both'
        """
        matches = []
        
        # Try date-based endpoints
        date_endpoints = {
            'events_all': self.endpoints.get('events_all', '')
        }
        
        for endpoint_key, endpoint in date_endpoints.items():
            if endpoint and not '{date}' in endpoint:  # Make sure date was replaced
                data = self._make_request(endpoint)
                
                if data and 'events' in data:
                    for match in data['events']:
                        match['source_endpoint'] = endpoint_key
                        matches.append(match)
                elif data and 'fixtures' in data:
                    for match in data['fixtures']:
                        match['source_endpoint'] = endpoint_key
                        matches.append(match)
        
        if matches:
            logger.info(f"Retrieved {len(matches)} matches for date {date}")
            return matches
        
        return None
    
    def discover_working_endpoints(self) -> Dict[str, bool]:
        """Test various endpoint patterns to find what actually works"""
        test_endpoints = [
            "/api/tennis/events",
            "/api/tennis/matches", 
            "/api/tennis/calendar",
            "/api/tennis/schedule",
            "/api/tennis/fixtures",
            "/api/tennis/tournaments",
            "/api/tennis/events/today",
            "/api/tennis/events/upcoming",
            "/api/tennis/matches/scheduled"
        ]
        
        results = {}
        for endpoint in test_endpoints:
            if self.rate_limiter.can_make_request():
                data = self._make_request(endpoint)
                results[endpoint] = data is not None
            else:
                results[endpoint] = "rate_limited"
        
        return results
    
    def get_tournament_schedule(self, tournament_id: str, season_id: str = None) -> Optional[List[Dict]]:
        """Get schedule for a specific tournament
        
        Args:
            tournament_id: Tournament ID
            season_id: Season ID (optional)
        """
        matches = []
        
        # Try different tournament endpoint patterns
        tournament_endpoints = []
        
        if season_id:
            season_endpoint = self.endpoints.get('season_events', '').format(id=tournament_id, season_id=season_id)
            if season_endpoint and '{' not in season_endpoint:
                tournament_endpoints.append(('season_events', season_endpoint))
        
        tournament_endpoint = self.endpoints.get('tournament_matches', '').format(id=tournament_id)
        if tournament_endpoint and '{id}' not in tournament_endpoint:
            tournament_endpoints.append(('tournament_matches', tournament_endpoint))
        
        for endpoint_key, endpoint in tournament_endpoints:
            data = self._make_request(endpoint)
            
            if data and 'events' in data:
                logger.info(f"Retrieved tournament events from {endpoint_key} for ID {tournament_id}")
                return data['events']
            elif data and 'matches' in data:
                logger.info(f"Retrieved tournament matches from {endpoint_key} for ID {tournament_id}")
                return data['matches']
        
        return None

    def get_remaining_requests(self) -> int:
        """Get number of remaining requests today"""
        return self.rate_limiter.get_remaining_requests()
    
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
        
        # Test scheduled matches
        print("\n=== Testing Scheduled Matches ===")
        scheduled_matches = client.get_scheduled_matches()
        if scheduled_matches:
            print(f"Retrieved {len(scheduled_matches)} scheduled matches")
            for match in scheduled_matches[:3]:  # Show first 3
                print(f"Scheduled Match: {match}")
        else:
            print("No scheduled matches data retrieved")
        
        # Test all events to understand available statuses
        print("\n=== Testing All Events (Status Analysis) ===")
        all_events = client.get_all_events()
        if all_events:
            print(f"Retrieved {len(all_events)} total events")
            
            # Analyze status types
            status_analysis = {}
            future_matches = 0
            current_time = time.time()
            from datetime import datetime
            
            for match in all_events:
                status_type = match.get('status', {}).get('type', 'unknown')
                status_code = match.get('status', {}).get('code', 'unknown')
                start_timestamp = match.get('startTimestamp', 0)
                
                status_key = f"{status_type} (code: {status_code})"
                if status_key not in status_analysis:
                    status_analysis[status_key] = 0
                status_analysis[status_key] += 1
                
                if start_timestamp > current_time:
                    future_matches += 1
            
            print(f"Status analysis:")
            for status, count in status_analysis.items():
                print(f"  {status}: {count} matches")
            print(f"Matches with future start time: {future_matches}")
            
            # Show a few sample matches with different statuses
            print(f"\nSample matches:")
            shown_statuses = set()
            for match in all_events[:10]:
                status_type = match.get('status', {}).get('type', 'unknown')
                if status_type not in shown_statuses:
                    shown_statuses.add(status_type)
                    start_time = datetime.fromtimestamp(match.get('startTimestamp', 0))
                    print(f"  {match.get('homeTeam', {}).get('name', 'Unknown')} vs {match.get('awayTeam', {}).get('name', 'Unknown')}")
                    print(f"    Status: {status_type} | Start: {start_time} | Tournament: {match.get('tournament', {}).get('name', 'Unknown')}")
        else:
            print("No events data retrieved")
        
        # Test scheduled matches with improved filtering
        print("\n=== Testing Scheduled Matches (Improved) ===")
        scheduled_matches = client.get_scheduled_matches()
        if scheduled_matches:
            print(f"Retrieved {len(scheduled_matches)} scheduled matches")
            for match in scheduled_matches[:3]:  # Show first 3
                start_time = datetime.fromtimestamp(match.get('startTimestamp', 0))
                print(f"Scheduled: {match.get('homeTeam', {}).get('name', 'Unknown')} vs {match.get('awayTeam', {}).get('name', 'Unknown')}")
                print(f"  Start: {start_time} | {match.get('scheduling_reason', 'Unknown reason')}")
        else:
            print("No scheduled matches data retrieved")
        
        print(f"\nFinal Status: {client.get_status()}")
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_rapidapi_client()