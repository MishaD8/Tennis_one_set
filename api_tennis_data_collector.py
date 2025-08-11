#!/usr/bin/env python3
"""
API-Tennis.com Data Collector Integration
Integrates API-Tennis.com with the existing UniversalCollector system
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os

from api_tennis_integration import (
    APITennisClient, TennisMatch, Tournament, TennisPlayer, 
    get_api_tennis_client, initialize_api_tennis_client
)
from config import get_config

logger = logging.getLogger(__name__)


class APITennisDataCollector:
    """Data collector that integrates API-Tennis.com with the tennis prediction system"""
    
    def __init__(self, api_key: str = None, enable_caching: bool = True):
        """
        Initialize API-Tennis data collector
        
        Args:
            api_key: API-Tennis.com API key (will use environment variable if not provided)
            enable_caching: Enable local caching for better performance
        """
        self.config = get_config()
        self.api_key = api_key or self.config.API_TENNIS_KEY
        
        # Initialize API client
        if self.api_key:
            self.client = initialize_api_tennis_client(
                api_key=self.api_key,
                enable_caching=enable_caching
            )
            self.available = True
            logger.info("API-Tennis data collector initialized with API key")
        else:
            self.client = None
            self.available = False
            logger.warning("API-Tennis data collector initialized without API key - will be inactive")
        
        # Professional tournament filters
        self.professional_keywords = [
            'atp', 'wta', 'grand slam', 'masters', 'wimbledon',
            'french open', 'us open', 'australian open', 'miami',
            'indian wells', 'madrid', 'rome', 'montreal', 'cincinnati',
            'roland garros', 'open', 'championships', 'masters 1000',
            'premier', 'tier'
        ]
        
        self.non_professional_keywords = [
            'utr', 'ptt', 'junior', 'college', 'university',
            'challenger', 'futures', 'itf', 'qualifying',
            'youth', 'exhibition', 'invitational', 'amateur'
        ]
    
    def is_available(self) -> bool:
        """Check if API-Tennis integration is available"""
        return self.available and self.client is not None
    
    def get_current_matches(self, include_live: bool = True) -> List[Dict[str, Any]]:
        """
        Get current matches in Universal Collector format
        
        Args:
            include_live: Include live matches in addition to today's fixtures
            
        Returns:
            List of matches in Universal Collector format
        """
        if not self.is_available():
            logger.warning("API-Tennis not available - returning empty match list")
            return []
        
        try:
            matches = []
            
            # Get today's fixtures
            today_matches = self.client.get_today_matches()
            matches.extend(today_matches)
            
            # Get live matches if requested
            if include_live:
                live_matches = self.client.get_live_matches()
                matches.extend(live_matches)
            
            # Convert to Universal Collector format
            universal_matches = []
            for match in matches:
                if self._is_professional_match(match):
                    universal_match = self._convert_to_universal_format(match)
                    if universal_match:
                        universal_matches.append(universal_match)
            
            logger.info(f"Retrieved {len(universal_matches)} professional matches from API-Tennis")
            return universal_matches
            
        except Exception as e:
            logger.error(f"Failed to get current matches from API-Tennis: {e}")
            return []
    
    def get_upcoming_matches(self, days_ahead: int = 7) -> List[Dict[str, Any]]:
        """
        Get upcoming matches in the next N days
        
        Args:
            days_ahead: Number of days to look ahead
            
        Returns:
            List of matches in Universal Collector format
        """
        if not self.is_available():
            return []
        
        try:
            matches = self.client.get_upcoming_matches(days_ahead)
            
            # Convert and filter
            universal_matches = []
            for match in matches:
                if self._is_professional_match(match):
                    universal_match = self._convert_to_universal_format(match)
                    if universal_match:
                        universal_matches.append(universal_match)
            
            logger.info(f"Retrieved {len(universal_matches)} upcoming professional matches")
            return universal_matches
            
        except Exception as e:
            logger.error(f"Failed to get upcoming matches from API-Tennis: {e}")
            return []
    
    def get_tournaments(self) -> List[Dict[str, Any]]:
        """
        Get current tournaments
        
        Returns:
            List of tournaments in Universal Collector format
        """
        if not self.is_available():
            return []
        
        try:
            tournaments = self.client.get_tournaments()
            
            # Convert to universal format
            universal_tournaments = []
            for tournament in tournaments:
                if self._is_professional_tournament(tournament):
                    universal_tournament = {
                        'id': tournament.id,
                        'name': tournament.name,
                        'location': tournament.location,
                        'surface': self._normalize_surface(tournament.surface),
                        'level': self._determine_tournament_level(tournament),
                        'category': tournament.category,
                        'start_date': tournament.start_date.isoformat() if tournament.start_date else None,
                        'end_date': tournament.end_date.isoformat() if tournament.end_date else None,
                        'status': 'active',
                        'data_source': 'API-Tennis'
                    }
                    universal_tournaments.append(universal_tournament)
            
            logger.info(f"Retrieved {len(universal_tournaments)} professional tournaments")
            return universal_tournaments
            
        except Exception as e:
            logger.error(f"Failed to get tournaments from API-Tennis: {e}")
            return []
    
    def get_player_matches(self, player_name: str, days_ahead: int = 30) -> List[Dict[str, Any]]:
        """
        Get matches for a specific player
        
        Args:
            player_name: Player name to search for
            days_ahead: Number of days to search ahead
            
        Returns:
            List of matches involving the player
        """
        if not self.is_available():
            return []
        
        try:
            matches = self.client.search_matches_by_player(player_name, days_ahead)
            
            universal_matches = []
            for match in matches:
                if self._is_professional_match(match):
                    universal_match = self._convert_to_universal_format(match)
                    if universal_match:
                        universal_matches.append(universal_match)
            
            logger.info(f"Found {len(universal_matches)} matches for player '{player_name}'")
            return universal_matches
            
        except Exception as e:
            logger.error(f"Failed to get matches for player '{player_name}': {e}")
            return []
    
    def get_match_odds(self, match_id: int) -> Dict[str, Any]:
        """
        Get betting odds for a specific match
        
        Args:
            match_id: API-Tennis match ID
            
        Returns:
            Odds data
        """
        if not self.is_available():
            return {}
        
        try:
            odds_data = self.client.get_odds(fixture_id=match_id)
            
            # Normalize odds format
            normalized_odds = {
                'match_id': match_id,
                'bookmakers': [],
                'best_odds': {},
                'last_updated': datetime.now().isoformat(),
                'data_source': 'API-Tennis'
            }
            
            # Process odds data (format will depend on actual API response)
            if isinstance(odds_data, list):
                for odds_item in odds_data:
                    bookmaker_data = {
                        'bookmaker': odds_item.get('bookmaker', 'Unknown'),
                        'player1_odds': odds_item.get('home_odds'),
                        'player2_odds': odds_item.get('away_odds')
                    }
                    normalized_odds['bookmakers'].append(bookmaker_data)
            
            return normalized_odds
            
        except Exception as e:
            logger.error(f"Failed to get odds for match {match_id}: {e}")
            return {}
    
    def _convert_to_universal_format(self, match: TennisMatch) -> Optional[Dict[str, Any]]:
        """Convert API-Tennis match to Universal Collector format"""
        try:
            return {
                'id': f"api_tennis_{match.id}",
                'player1': match.player1.name if match.player1 else "Unknown",
                'player2': match.player2.name if match.player2 else "Unknown",
                'tournament': match.tournament_name,
                'location': match.location,
                'surface': self._normalize_surface(match.surface),
                'level': self._determine_match_level(match),
                'date': match.start_time.strftime('%Y-%m-%d') if match.start_time else datetime.now().strftime('%Y-%m-%d'),
                'time': match.start_time.strftime('%H:%M') if match.start_time else 'TBD',
                'round': match.round,
                'court': 'TBD',  # API-Tennis doesn't provide court info
                'status': self._normalize_status(match.status),
                'score': match.score,
                'data_source': 'API-Tennis',
                'api_tennis_id': match.id,
                'quality_score': 95,  # High quality - real API data
                
                # Player details
                'player1_country': match.player1.country if match.player1 else '',
                'player2_country': match.player2.country if match.player2 else '',
                'player1_ranking': match.player1.ranking if match.player1 else None,
                'player2_ranking': match.player2.ranking if match.player2 else None,
                
                # Odds if available
                'player1_odds': match.odds_player1,
                'player2_odds': match.odds_player2,
                
                # Tournament details
                'tournament_id': match.tournament_id,
                'event_type': match.event_type,
                
                # Metadata
                'last_updated': datetime.now().isoformat(),
                'source_reliability': 'high'
            }
            
        except Exception as e:
            logger.warning(f"Failed to convert match to universal format: {e}")
            return None
    
    def _is_professional_match(self, match: TennisMatch) -> bool:
        """Check if match is a professional ATP/WTA match"""
        tournament_name = match.tournament_name.lower()
        
        # Exclude non-professional tournaments
        for keyword in self.non_professional_keywords:
            if keyword in tournament_name:
                return False
        
        # Require professional indicators
        for keyword in self.professional_keywords:
            if keyword in tournament_name:
                return True
        
        # Also check event type and level
        event_type = (match.event_type or '').lower()
        level = (match.level or '').lower()
        
        professional_indicators = ['atp', 'wta', 'grand slam', 'masters']
        return any(indicator in event_type + level for indicator in professional_indicators)
    
    def _is_professional_tournament(self, tournament: Tournament) -> bool:
        """Check if tournament is professional level"""
        tournament_name = tournament.name.lower()
        category = (tournament.category or '').lower()
        
        # Exclude non-professional
        for keyword in self.non_professional_keywords:
            if keyword in tournament_name + category:
                return False
        
        # Require professional indicators
        for keyword in self.professional_keywords:
            if keyword in tournament_name + category:
                return True
        
        return False
    
    def _normalize_surface(self, surface: str) -> str:
        """Normalize surface type"""
        if not surface:
            return 'Hard'
        
        surface_lower = surface.lower()
        if 'clay' in surface_lower:
            return 'Clay'
        elif 'grass' in surface_lower:
            return 'Grass'
        elif 'carpet' in surface_lower:
            return 'Carpet'
        else:
            return 'Hard'
    
    def _determine_tournament_level(self, tournament: Tournament) -> str:
        """Determine tournament level from tournament data"""
        name = tournament.name.lower()
        category = (tournament.category or '').lower()
        level = (tournament.level or '').lower()
        
        combined = name + category + level
        
        if any(keyword in combined for keyword in ['grand slam', 'wimbledon', 'french open', 'us open', 'australian open']):
            return 'Grand Slam'
        elif any(keyword in combined for keyword in ['masters 1000', 'atp 1000', 'wta 1000']):
            return 'ATP 1000' if 'atp' in combined else 'WTA 1000'
        elif any(keyword in combined for keyword in ['500', 'atp 500', 'wta 500']):
            return 'ATP 500' if 'atp' in combined else 'WTA 500'
        elif any(keyword in combined for keyword in ['250', 'atp 250', 'wta 250']):
            return 'ATP 250' if 'atp' in combined else 'WTA 250'
        else:
            return 'ATP 250'  # Default
    
    def _determine_match_level(self, match: TennisMatch) -> str:
        """Determine match level from match data"""
        return self._determine_tournament_level(
            Tournament(name=match.tournament_name, level=match.level)
        )
    
    def _normalize_status(self, status: str) -> str:
        """Normalize match status"""
        if not status:
            return 'upcoming'
        
        status_lower = status.lower()
        if any(keyword in status_lower for keyword in ['live', 'playing', 'in progress']):
            return 'live'
        elif any(keyword in status_lower for keyword in ['finished', 'completed', 'final']):
            return 'finished'
        elif any(keyword in status_lower for keyword in ['postponed', 'delayed']):
            return 'postponed'
        elif any(keyword in status_lower for keyword in ['cancelled', 'canceled']):
            return 'cancelled'
        else:
            return 'upcoming'
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get integration status and statistics"""
        status = {
            'available': self.is_available(),
            'api_key_configured': bool(self.api_key),
            'client_status': None,
            'last_successful_request': None,
            'error_count': 0
        }
        
        if self.client:
            try:
                status['client_status'] = self.client.get_client_status()
                
                # Test API connectivity
                test_tournaments = self.client.get_tournaments()
                status['connectivity_test'] = 'success'
                status['tournament_count'] = len(test_tournaments)
                status['last_successful_request'] = datetime.now().isoformat()
                
            except Exception as e:
                status['connectivity_test'] = 'failed'
                status['error_message'] = str(e)
                status['error_count'] = 1
        
        return status
    
    def clear_cache(self):
        """Clear API-Tennis cache"""
        if self.client:
            self.client.clear_cache()
            logger.info("API-Tennis cache cleared")


# Enhanced Universal Collector integration
class EnhancedAPITennisCollector:
    """Enhanced collector that integrates API-Tennis with existing Universal Collector"""
    
    def __init__(self):
        self.api_tennis_collector = APITennisDataCollector()
        
        # Try to import existing collectors
        try:
            from enhanced_universal_collector import EnhancedUniversalCollector
            self.universal_collector = EnhancedUniversalCollector()
            self.has_universal_collector = True
        except ImportError:
            self.universal_collector = None
            self.has_universal_collector = False
            logger.warning("Enhanced Universal Collector not available")
    
    def get_comprehensive_match_data(self, days_ahead: int = 2) -> List[Dict[str, Any]]:
        """Get comprehensive match data from all sources including API-Tennis"""
        all_matches = []
        
        # Get API-Tennis data
        if self.api_tennis_collector.is_available():
            try:
                api_tennis_matches = self.api_tennis_collector.get_current_matches()
                upcoming_matches = self.api_tennis_collector.get_upcoming_matches(days_ahead)
                
                # Combine and deduplicate API-Tennis matches
                combined_api_tennis = api_tennis_matches + upcoming_matches
                seen_ids = set()
                unique_api_tennis = []
                
                for match in combined_api_tennis:
                    match_id = match.get('id')
                    if match_id not in seen_ids:
                        seen_ids.add(match_id)
                        unique_api_tennis.append(match)
                
                all_matches.extend(unique_api_tennis)
                logger.info(f"Added {len(unique_api_tennis)} matches from API-Tennis")
                
            except Exception as e:
                logger.error(f"Failed to get API-Tennis matches: {e}")
        
        # Get Universal Collector data
        if self.has_universal_collector:
            try:
                universal_matches = self.universal_collector.get_comprehensive_match_data(days_ahead)
                all_matches.extend(universal_matches)
                logger.info(f"Added {len(universal_matches)} matches from Universal Collector")
            except Exception as e:
                logger.error(f"Failed to get Universal Collector matches: {e}")
        
        # Deduplicate across all sources
        if all_matches:
            all_matches = self._deduplicate_matches(all_matches)
        
        logger.info(f"Total comprehensive matches after deduplication: {len(all_matches)}")
        return all_matches
    
    def _deduplicate_matches(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate matches from multiple sources with priority"""
        if not matches:
            return matches
        
        # Group matches by player pair and date
        match_groups = {}
        
        for match in matches:
            key = self._create_dedup_key(match)
            if key not in match_groups:
                match_groups[key] = []
            match_groups[key].append(match)
        
        # Select best match from each group
        deduplicated = []
        source_priority = {
            'API-Tennis': 100,
            'TennisExplorer': 90,
            'RapidAPI_Scheduled': 80,
            'UniversalCollector': 70
        }
        
        for key, group_matches in match_groups.items():
            if len(group_matches) == 1:
                deduplicated.append(group_matches[0])
            else:
                # Select best match based on source priority and quality
                best_match = max(group_matches, key=lambda m: (
                    source_priority.get(m.get('data_source', ''), 0),
                    m.get('quality_score', 0)
                ))
                deduplicated.append(best_match)
        
        return deduplicated
    
    def _create_dedup_key(self, match: Dict[str, Any]) -> str:
        """Create deduplication key for match"""
        player1 = self._normalize_player_name(match.get('player1', ''))
        player2 = self._normalize_player_name(match.get('player2', ''))
        
        # Ensure consistent ordering
        if player1 > player2:
            player1, player2 = player2, player1
        
        date = match.get('date', datetime.now().strftime('%Y-%m-%d'))
        
        return f"{player1}_vs_{player2}_{date}"
    
    def _normalize_player_name(self, name: str) -> str:
        """Normalize player name for comparison"""
        if not name:
            return 'unknown'
        
        import re
        normalized = re.sub(r'[^\w\s]', '', name)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized.strip().lower()
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all data sources"""
        return {
            'api_tennis': self.api_tennis_collector.get_integration_status(),
            'universal_collector': self.has_universal_collector,
            'total_sources_available': sum([
                self.api_tennis_collector.is_available(),
                self.has_universal_collector
            ])
        }


# Global instances
api_tennis_data_collector = APITennisDataCollector()
enhanced_api_tennis_collector = EnhancedAPITennisCollector()


def get_api_tennis_data_collector() -> APITennisDataCollector:
    """Get global API-Tennis data collector instance"""
    return api_tennis_data_collector


def get_enhanced_api_tennis_collector() -> EnhancedAPITennisCollector:
    """Get enhanced collector with all data sources"""
    return enhanced_api_tennis_collector