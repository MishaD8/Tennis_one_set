#!/usr/bin/env python3
"""
API-Tennis.com Integration for Tennis Prediction System
Production-ready integration with comprehensive error handling, rate limiting, and caching
"""

import os
import logging
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from functools import lru_cache
import threading
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class TournamentType(Enum):
    """Tournament types supported by API-Tennis"""
    ATP = "atp"
    WTA = "wta" 
    ITF = "itf"
    CHALLENGER = "challenger"
    ALL = "all"


class MatchStatus(Enum):
    """Match status types"""
    UPCOMING = "upcoming"
    LIVE = "live"
    FINISHED = "finished"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


@dataclass
class TennisPlayer:
    """Tennis player data model"""
    id: Optional[int] = None
    name: str = ""
    country: str = ""
    ranking: Optional[int] = None
    seed: Optional[int] = None
    age: Optional[int] = None
    points: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TennisMatch:
    """Tennis match data model"""
    id: Optional[int] = None
    player1: TennisPlayer = None
    player2: TennisPlayer = None
    tournament_id: Optional[int] = None
    tournament_name: str = ""
    surface: str = ""
    round: str = ""
    status: str = ""
    start_time: Optional[datetime] = None
    score: str = ""
    odds_player1: Optional[float] = None
    odds_player2: Optional[float] = None
    event_type: str = ""
    location: str = ""
    level: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.player1:
            data['player1'] = self.player1.to_dict()
        if self.player2:
            data['player2'] = self.player2.to_dict()
        if self.start_time:
            data['start_time'] = self.start_time.isoformat()
        return data


@dataclass
class Tournament:
    """Tournament data model"""
    id: Optional[int] = None
    name: str = ""
    location: str = ""
    surface: str = ""
    prize_money: Optional[int] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    category: str = ""
    level: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.start_date:
            data['start_date'] = self.start_date.isoformat()
        if self.end_date:
            data['end_date'] = self.end_date.isoformat()
        return data


class APITennisRateLimiter:
    """Rate limiter for API-Tennis requests"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests_made = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        with self.lock:
            now = time.time()
            
            # Remove requests older than 1 minute
            self.requests_made = [req_time for req_time in self.requests_made 
                                if now - req_time < 60]
            
            # Check if we're at the limit
            if len(self.requests_made) >= self.requests_per_minute:
                sleep_time = 60 - (now - self.requests_made[0])
                if sleep_time > 0:
                    logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
            
            # Record this request
            self.requests_made.append(now)


class APITennisClient:
    """Production-ready API-Tennis.com client"""
    
    BASE_URL = "https://api.api-tennis.com/tennis/"
    
    def __init__(self, api_key: str = None, timeout: int = 30, enable_caching: bool = True):
        """
        Initialize API-Tennis client
        
        Args:
            api_key: API key from API-Tennis.com account
            timeout: Request timeout in seconds
            enable_caching: Enable local caching of responses
        """
        self.api_key = api_key or os.getenv('API_TENNIS_KEY', '')
        self.timeout = timeout
        self.enable_caching = enable_caching
        
        # Rate limiting
        self.rate_limiter = APITennisRateLimiter(requests_per_minute=50)  # Conservative limit
        
        # Caching
        self.cache_dir = "cache/api_tennis"
        self.cache_duration_minutes = 15  # Cache for 15 minutes
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TennisPredictionSystem/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
        # Ensure cache directory exists
        if self.enable_caching:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"API-Tennis client initialized with caching={'enabled' if enable_caching else 'disabled'}")
    
    def _validate_config(self):
        """Validate client configuration"""
        if not self.api_key:
            logger.warning("API_TENNIS_KEY not configured - some methods may fail")
            
        if self.timeout < 5:
            logger.warning(f"Timeout {self.timeout}s may be too low for API responses")
    
    def _get_cache_path(self, method: str, params: Dict[str, Any]) -> str:
        """Generate cache file path for request"""
        param_string = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(param_string.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{method}_{param_hash}.json")
    
    def _load_cached_response(self, method: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Load cached API response if still valid"""
        if not self.enable_caching:
            return None
            
        cache_path = self._get_cache_path(method, params)
        
        try:
            if not os.path.exists(cache_path):
                return None
            
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            
            # Check if cache is still valid
            cached_time = datetime.fromisoformat(cached_data['timestamp'])
            if datetime.now() - cached_time < timedelta(minutes=self.cache_duration_minutes):
                logger.debug(f"Using cached response for {method}")
                return cached_data['response']
            else:
                # Remove expired cache
                os.remove(cache_path)
                return None
                
        except Exception as e:
            logger.warning(f"Failed to load cached response: {e}")
            return None
    
    def _save_cached_response(self, method: str, params: Dict[str, Any], response: Dict[str, Any]):
        """Save API response to cache"""
        if not self.enable_caching:
            return
            
        cache_path = self._get_cache_path(method, params)
        
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'method': method,
                'params': params,
                'response': response
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save cached response: {e}")
    
    def _make_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make API request with rate limiting, caching, and error handling
        
        Args:
            method: API method to call
            params: Request parameters
            
        Returns:
            API response data
        """
        params = params or {}
        
        # Check cache first
        cached_response = self._load_cached_response(method, params)
        if cached_response:
            return cached_response
        
        # Prepare request parameters
        request_params = {
            'method': method,
            'APIkey': self.api_key,
            **params
        }
        
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()
        
        try:
            logger.debug(f"Making API request: {method} with params: {params}")
            
            response = self.session.get(
                self.BASE_URL,
                params=request_params,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            
            # Check for API error responses
            if isinstance(data, dict) and data.get('error'):
                raise Exception(f"API Error: {data['error']}")
            
            # Cache successful response
            self._save_cached_response(method, params, data)
            
            logger.debug(f"Successful API response for {method}")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {method}: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response for {method}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error for {method}: {e}")
            raise
    
    def get_event_types(self) -> List[Dict[str, Any]]:
        """Get available event types"""
        return self._make_request('get_events')
    
    def get_tournaments(self, event_id: int = None) -> List[Tournament]:
        """
        Get tournaments
        
        Args:
            event_id: Optional event ID to filter tournaments
            
        Returns:
            List of Tournament objects
        """
        params = {}
        if event_id:
            params['event_id'] = event_id
        
        data = self._make_request('get_leagues', params)
        
        tournaments = []
        if isinstance(data, list):
            for tournament_data in data:
                tournament = Tournament(
                    id=tournament_data.get('id'),
                    name=tournament_data.get('name', ''),
                    location=tournament_data.get('country', ''),
                    category=tournament_data.get('type', ''),
                    level=tournament_data.get('tier', '')
                )
                tournaments.append(tournament)
        
        logger.info(f"Retrieved {len(tournaments)} tournaments")
        return tournaments
    
    def get_fixtures(self, 
                     date_start: Union[str, datetime] = None,
                     date_stop: Union[str, datetime] = None,
                     league_id: int = None,
                     team_id: int = None) -> List[TennisMatch]:
        """
        Get upcoming matches (fixtures)
        
        Args:
            date_start: Start date for matches (YYYY-MM-DD or datetime)
            date_stop: End date for matches (YYYY-MM-DD or datetime)
            league_id: Optional league/tournament ID filter
            team_id: Optional player ID filter
            
        Returns:
            List of TennisMatch objects
        """
        params = {}
        
        # Handle date parameters
        if date_start:
            if isinstance(date_start, datetime):
                params['date_start'] = date_start.strftime('%Y-%m-%d')
            else:
                params['date_start'] = date_start
        
        if date_stop:
            if isinstance(date_stop, datetime):
                params['date_stop'] = date_stop.strftime('%Y-%m-%d')
            else:
                params['date_stop'] = date_stop
        
        if league_id:
            params['league_id'] = league_id
        if team_id:
            params['team_id'] = team_id
        
        data = self._make_request('get_fixtures', params)
        
        matches = []
        # API-Tennis returns data in format: {"success": 1, "result": [...]}
        if isinstance(data, dict) and data.get('success') == 1:
            result = data.get('result', [])
            if isinstance(result, list):
                for match_data in result:
                    match = self._parse_match_data(match_data)
                    if match:
                        matches.append(match)
        elif isinstance(data, list):
            # Fallback for direct list response
            for match_data in data:
                match = self._parse_match_data(match_data)
                if match:
                    matches.append(match)
        
        logger.info(f"Retrieved {len(matches)} fixtures")
        return matches
    
    def get_live_matches(self) -> List[TennisMatch]:
        """Get currently live matches"""
        data = self._make_request('get_livescore')
        
        matches = []
        # API-Tennis returns data in format: {"success": 1, "result": [...]}
        if isinstance(data, dict) and data.get('success') == 1:
            result = data.get('result', [])
            if isinstance(result, list):
                for match_data in result:
                    match = self._parse_match_data(match_data)
                    if match:
                        match.status = MatchStatus.LIVE.value
                        matches.append(match)
        elif isinstance(data, list):
            # Fallback for direct list response
            for match_data in data:
                match = self._parse_match_data(match_data)
                if match:
                    match.status = MatchStatus.LIVE.value
                    matches.append(match)
        
        logger.info(f"Retrieved {len(matches)} live matches")
        return matches
    
    def get_head_to_head(self, player1_id: int, player2_id: int) -> Dict[str, Any]:
        """
        Get head-to-head statistics between two players
        
        Args:
            player1_id: First player ID
            player2_id: Second player ID
            
        Returns:
            Head-to-head statistics
        """
        params = {
            'team1_id': player1_id,
            'team2_id': player2_id
        }
        
        return self._make_request('get_H2H', params)
    
    def get_standings(self, event_type: str = 'ATP') -> List[Dict[str, Any]]:
        """
        Get tournament standings/rankings - CORRECTED according to API documentation
        
        Args:
            event_type: 'ATP' or 'WTA' (correct parameter according to docs)
            
        Returns:
            List of player standings with rankings
        """
        params = {'event_type': event_type}
        data = self._make_request('get_standings', params)
        
        # Handle the response format
        if isinstance(data, dict) and data.get('success') == 1:
            return data.get('result', [])
        return data if isinstance(data, list) else []
    
    def get_players(self, player_key: int = None) -> List[TennisPlayer]:
        """
        Get players list - CORRECTED according to API documentation
        
        Args:
            player_key: Optional specific player key to get details for
            
        Returns:
            List of TennisPlayer objects with ranking data
        """
        params = {}
        if player_key:
            params['player_key'] = player_key
        
        data = self._make_request('get_players', params)
        
        players = []
        # Handle the response format
        if isinstance(data, dict) and data.get('success') == 1:
            result = data.get('result', [])
            if isinstance(result, list):
                for player_data in result:
                    player = self._parse_player_data(player_data)
                    if player:
                        players.append(player)
        elif isinstance(data, list):
            # Fallback for direct list response
            for player_data in data:
                player = self._parse_player_data(player_data)
                if player:
                    players.append(player)
        
        logger.info(f"Retrieved {len(players)} players")
        return players
    
    def get_odds(self, 
                 fixture_id: int = None,
                 league_id: int = None,
                 bookmaker: str = None) -> Dict[str, Any]:
        """
        Get betting odds
        
        Args:
            fixture_id: Specific match ID
            league_id: Tournament ID
            bookmaker: Specific bookmaker filter
            
        Returns:
            Odds data
        """
        params = {}
        if fixture_id:
            params['fixture_id'] = fixture_id
        if league_id:
            params['league_id'] = league_id
        if bookmaker:
            params['bookmaker'] = bookmaker
        
        return self._make_request('get_odds', params)
    
    def get_live_odds(self, fixture_id: int = None) -> Dict[str, Any]:
        """
        Get live betting odds
        
        Args:
            fixture_id: Optional specific match ID
            
        Returns:
            Live odds data
        """
        params = {}
        if fixture_id:
            params['fixture_id'] = fixture_id
        
        return self._make_request('get_liveodds', params)
    
    def _parse_player_data(self, player_data: Dict[str, Any]) -> Optional[TennisPlayer]:
        """Parse API player data into TennisPlayer object with ranking"""
        try:
            # Extract basic player information
            player = TennisPlayer(
                id=player_data.get('player_key'),
                name=player_data.get('player_name', ''),
                country=player_data.get('player_country', '')
            )
            
            # Extract ranking from stats
            stats = player_data.get('stats', [])
            if stats:
                # Look for the most recent singles ranking
                current_year = str(datetime.now().year)
                rankings = []
                
                for stat in stats:
                    if stat.get('type', '').lower() == 'singles':
                        rank = stat.get('rank')
                        season = stat.get('season', '')
                        
                        if rank and rank.isdigit():
                            rankings.append({
                                'rank': int(rank),
                                'season': season,
                                'year_priority': 1 if season == current_year else 0
                            })
                
                if rankings:
                    # Sort by year priority (current year first), then by rank
                    rankings.sort(key=lambda x: (x['year_priority'], x['rank']), reverse=True)
                    player.ranking = rankings[0]['rank']
            
            return player
            
        except Exception as e:
            logger.warning(f"Failed to parse player data: {e}")
            return None
    
    def _parse_match_data(self, match_data: Dict[str, Any]) -> Optional[TennisMatch]:
        """Parse API match data into TennisMatch object"""
        try:
            # Extract player information from API-Tennis format
            player1 = TennisPlayer(
                id=match_data.get('first_player_key'),
                name=match_data.get('event_first_player', ''),
                country=''  # API-Tennis doesn't provide country in fixtures
            )
            
            player2 = TennisPlayer(
                id=match_data.get('second_player_key'),
                name=match_data.get('event_second_player', ''),
                country=''  # API-Tennis doesn't provide country in fixtures
            )
            
            # Parse start time from event_date and event_time
            start_time = None
            if match_data.get('event_date'):
                try:
                    date_str = match_data['event_date']
                    time_str = match_data.get('event_time', '00:00')
                    
                    # Combine date and time
                    datetime_str = f"{date_str} {time_str}"
                    start_time = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M')
                except Exception as e:
                    logger.debug(f"Failed to parse datetime: {e}")
                    pass
            
            # Determine surface from tournament name (API-Tennis doesn't provide surface directly)
            surface = self._determine_surface_from_tournament(match_data.get('tournament_name', ''))
            
            # Create match object
            match = TennisMatch(
                id=match_data.get('event_key'),
                player1=player1,
                player2=player2,
                tournament_id=match_data.get('tournament_key'),
                tournament_name=match_data.get('tournament_name', ''),
                surface=surface,
                round=match_data.get('tournament_round', ''),
                status=match_data.get('event_status', ''),
                start_time=start_time,
                score=match_data.get('event_final_result', ''),
                event_type=match_data.get('event_type_type', ''),
                location=self._extract_location_from_tournament(match_data.get('tournament_name', '')),
                level=self._determine_level_from_event_type(match_data.get('event_type_type', ''))
            )
            
            return match
            
        except Exception as e:
            logger.warning(f"Failed to parse match data: {e}")
            return None
    
    def _determine_surface_from_tournament(self, tournament_name: str) -> str:
        """Determine surface from tournament name"""
        if not tournament_name:
            return 'Hard'
        
        tournament_lower = tournament_name.lower()
        if any(keyword in tournament_lower for keyword in ['french', 'roland garros', 'monte carlo', 'rome', 'madrid']):
            return 'Clay'
        elif any(keyword in tournament_lower for keyword in ['wimbledon', 'grass']):
            return 'Grass'
        else:
            return 'Hard'  # Default for most tournaments
    
    def _extract_location_from_tournament(self, tournament_name: str) -> str:
        """Extract location from tournament name"""
        if not tournament_name:
            return ''
        
        # Common tournament location mappings
        location_map = {
            'wimbledon': 'London',
            'french open': 'Paris',
            'roland garros': 'Paris',
            'us open': 'New York',
            'australian open': 'Melbourne',
            'indian wells': 'Indian Wells',
            'miami': 'Miami',
            'monte carlo': 'Monte Carlo',
            'madrid': 'Madrid',
            'rome': 'Rome',
            'cincinnati': 'Cincinnati'
        }
        
        tournament_lower = tournament_name.lower()
        for tournament, location in location_map.items():
            if tournament in tournament_lower:
                return location
        
        # Try to extract location from tournament name patterns
        words = tournament_name.split()
        if len(words) >= 2:
            return words[-1]  # Often the last word is the location
        
        return tournament_name
    
    def _determine_level_from_event_type(self, event_type: str) -> str:
        """Determine tournament level from event type"""
        if not event_type:
            return 'Unknown'
        
        event_lower = event_type.lower()
        if 'atp' in event_lower:
            return 'ATP'
        elif 'wta' in event_lower:
            return 'WTA'
        elif 'challenger' in event_lower:
            return 'Challenger'
        elif 'itf' in event_lower:
            return 'ITF'
        else:
            return 'Other'
    
    def get_today_matches(self) -> List[TennisMatch]:
        """Get today's matches"""
        today = datetime.now().strftime('%Y-%m-%d')
        return self.get_fixtures(date_start=today, date_stop=today)
    
    def get_ranking_mapping(self, event_types: List[str] = ['ATP', 'WTA']) -> Dict[int, int]:
        """
        Create a mapping of player_key -> current_ranking from standings
        
        Args:
            event_types: List of event types to get rankings for
            
        Returns:
            Dictionary mapping player keys to rankings
        """
        ranking_map = {}
        
        for event_type in event_types:
            try:
                standings = self.get_standings(event_type)
                
                for player in standings:
                    player_key = player.get('player_key')
                    place = player.get('place')
                    
                    if player_key and place:
                        try:
                            ranking_map[int(player_key)] = int(place)
                        except (ValueError, TypeError):
                            continue
                            
                logger.info(f"Added {len(standings)} {event_type} rankings to mapping")
                
            except Exception as e:
                logger.error(f"Error getting rankings for {event_type}: {e}")
        
        logger.info(f"Total ranking mapping: {len(ranking_map)} players")
        return ranking_map
    
    def enhance_matches_with_rankings(self, matches: List[TennisMatch]) -> List[TennisMatch]:
        """
        Enhance matches with player rankings
        
        Args:
            matches: List of TennisMatch objects
            
        Returns:
            List of enhanced matches with ranking data
        """
        if not matches:
            return matches
        
        logger.info(f"Enhancing {len(matches)} matches with ranking data")
        
        # Get rankings mapping once for all matches
        rankings = self.get_ranking_mapping()
        
        enhanced_matches = []
        for match in matches:
            try:
                # Enhance player 1
                if match.player1 and match.player1.id and match.player1.id in rankings:
                    match.player1.ranking = rankings[match.player1.id]
                
                # Enhance player 2
                if match.player2 and match.player2.id and match.player2.id in rankings:
                    match.player2.ranking = rankings[match.player2.id]
                
                enhanced_matches.append(match)
                
            except Exception as e:
                logger.error(f"Error enhancing match {match.id}: {e}")
                enhanced_matches.append(match)  # Add original match on error
        
        # Count how many matches have rankings
        ranked_count = sum(1 for m in enhanced_matches 
                          if (m.player1 and m.player1.ranking) or (m.player2 and m.player2.ranking))
        
        logger.info(f"Enhanced {ranked_count}/{len(enhanced_matches)} matches with ranking data")
        return enhanced_matches
    
    def get_fixtures_with_rankings(self, 
                                 date_start: str = None,
                                 date_stop: str = None,
                                 **kwargs) -> List[TennisMatch]:
        """
        Get fixtures enhanced with ranking data
        
        Returns:
            List of matches with ranking information
        """
        # Get basic fixtures
        matches = self.get_fixtures(date_start=date_start, date_stop=date_stop, **kwargs)
        
        # Enhance with rankings
        return self.enhance_matches_with_rankings(matches)
    
    def get_upcoming_matches(self, days_ahead: int = 7) -> List[TennisMatch]:
        """
        Get matches in the next N days
        
        Args:
            days_ahead: Number of days to look ahead
            
        Returns:
            List of upcoming matches
        """
        start_date = datetime.now()
        end_date = start_date + timedelta(days=days_ahead)
        
        return self.get_fixtures(
            date_start=start_date.strftime('%Y-%m-%d'),
            date_stop=end_date.strftime('%Y-%m-%d')
        )
    
    def search_matches_by_player(self, player_name: str, days_ahead: int = 30) -> List[TennisMatch]:
        """
        Search for matches involving a specific player
        
        Args:
            player_name: Player name to search for
            days_ahead: Number of days to search ahead
            
        Returns:
            List of matches involving the player
        """
        matches = self.get_upcoming_matches(days_ahead)
        player_matches = []
        
        player_name_lower = player_name.lower()
        for match in matches:
            if (player_name_lower in match.player1.name.lower() or 
                player_name_lower in match.player2.name.lower()):
                player_matches.append(match)
        
        return player_matches
    
    def get_client_status(self) -> Dict[str, Any]:
        """Get client status and configuration"""
        return {
            'api_key_configured': bool(self.api_key),
            'caching_enabled': self.enable_caching,
            'cache_directory': self.cache_dir if self.enable_caching else None,
            'timeout': self.timeout,
            'rate_limit': self.rate_limiter.requests_per_minute,
            'cache_duration_minutes': self.cache_duration_minutes
        }
    
    def clear_cache(self):
        """Clear all cached API responses"""
        if not self.enable_caching:
            return
        
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info("API-Tennis cache cleared")
    
    def __del__(self):
        """Clean up session"""
        if hasattr(self, 'session'):
            self.session.close()


# Global instance
api_tennis_client = None

def get_api_tennis_client() -> APITennisClient:
    """Get global API-Tennis client instance"""
    global api_tennis_client
    if api_tennis_client is None:
        api_tennis_client = APITennisClient()
    return api_tennis_client


def initialize_api_tennis_client(api_key: str = None, **kwargs) -> APITennisClient:
    """Initialize global API-Tennis client with custom settings"""
    global api_tennis_client
    api_tennis_client = APITennisClient(api_key=api_key, **kwargs)
    return api_tennis_client