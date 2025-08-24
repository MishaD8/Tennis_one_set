#!/usr/bin/env python3
"""
Enhanced Tennis Ranking Integration
Fixes the player ranking issue by correctly implementing get_players and get_standings methods
according to the API-Tennis.com documentation
"""

import os
import sys

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, using system environment variables
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import time

# Add current directory for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api_tennis_integration import APITennisClient, TennisPlayer, TennisMatch

logger = logging.getLogger(__name__)

class EnhancedRankingClient(APITennisClient):
    """Enhanced API-Tennis client with corrected ranking methods"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ranking_cache = {}
        self.player_cache = {}
        self.cache_duration = 3600  # 1 hour cache for rankings
    
    def get_standings_corrected(self, event_type: str = 'ATP') -> List[Dict[str, Any]]:
        """
        Get tournament standings/rankings - CORRECTED implementation
        
        According to API documentation, this method uses event_type parameter, not league_id
        
        Args:
            event_type: 'ATP' or 'WTA' (not league_id as in old implementation)
            
        Returns:
            List of player standings with rankings
        """
        cache_key = f"standings_{event_type}"
        
        # Check cache first
        if cache_key in self.ranking_cache:
            cached_data = self.ranking_cache[cache_key]
            if datetime.now() - cached_data['timestamp'] < timedelta(seconds=self.cache_duration):
                logger.debug(f"Using cached standings for {event_type}")
                return cached_data['data']
        
        try:
            # Correct parameters according to API documentation
            params = {'event_type': event_type}
            data = self._make_request('get_standings', params)
            
            standings = []
            if isinstance(data, dict) and data.get('success') == 1:
                result = data.get('result', [])
                standings = result
                
                # Cache the result
                self.ranking_cache[cache_key] = {
                    'data': standings,
                    'timestamp': datetime.now()
                }
                
                logger.info(f"Retrieved {len(standings)} {event_type} rankings")
            else:
                logger.error(f"Failed to get standings for {event_type}: {data}")
                
            return standings
            
        except Exception as e:
            logger.error(f"Error getting standings for {event_type}: {e}")
            return []
    
    def get_player_details_corrected(self, player_key: int) -> Optional[Dict[str, Any]]:
        """
        Get player details including ranking - CORRECTED implementation
        
        According to API documentation, this method uses player_key parameter
        
        Args:
            player_key: Player's unique key from the API
            
        Returns:
            Player details with ranking information
        """
        cache_key = f"player_{player_key}"
        
        # Check cache first
        if cache_key in self.player_cache:
            cached_data = self.player_cache[cache_key]
            if datetime.now() - cached_data['timestamp'] < timedelta(seconds=self.cache_duration):
                logger.debug(f"Using cached player data for {player_key}")
                return cached_data['data']
        
        try:
            # Correct parameters according to API documentation
            params = {'player_key': player_key}
            data = self._make_request('get_players', params)
            
            player_data = None
            if isinstance(data, dict) and data.get('success') == 1:
                result = data.get('result', [])
                if result:
                    player_data = result[0]  # Get first (and usually only) result
                    
                    # Cache the result
                    self.player_cache[cache_key] = {
                        'data': player_data,
                        'timestamp': datetime.now()
                    }
                    
                    logger.debug(f"Retrieved player data for {player_key}: {player_data.get('player_name', 'Unknown')}")
            else:
                logger.error(f"Failed to get player details for {player_key}: {data}")
                
            return player_data
            
        except Exception as e:
            logger.error(f"Error getting player details for {player_key}: {e}")
            return None
    
    def extract_player_ranking(self, player_data: Dict[str, Any], match_type: str = 'singles') -> Optional[int]:
        """
        Extract current ranking from player data
        
        Args:
            player_data: Player data from get_players API
            match_type: 'singles' or 'doubles'
            
        Returns:
            Current ranking or None if not available
        """
        if not player_data:
            return None
        
        stats = player_data.get('stats', [])
        if not stats:
            return None
        
        # Look for the most recent singles ranking
        current_year = str(datetime.now().year)
        rankings = []
        
        for stat in stats:
            if stat.get('type', '').lower() == match_type.lower():
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
            return rankings[0]['rank']
        
        return None
    
    def get_rankings_mapping(self, event_types: List[str] = ['ATP', 'WTA']) -> Dict[int, int]:
        """
        Create a mapping of player_key -> current_ranking
        
        Args:
            event_types: List of event types to get rankings for
            
        Returns:
            Dictionary mapping player keys to rankings
        """
        ranking_map = {}
        
        for event_type in event_types:
            try:
                standings = self.get_standings_corrected(event_type)
                
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
    
    def enhance_match_with_rankings(self, match: TennisMatch) -> TennisMatch:
        """
        Enhance a match object with player rankings
        
        Args:
            match: TennisMatch object to enhance
            
        Returns:
            Enhanced match with ranking data
        """
        try:
            # Get rankings mapping
            rankings = self.get_rankings_mapping()
            
            # Enhance player 1
            if match.player1 and match.player1.id:
                if match.player1.id in rankings:
                    match.player1.ranking = rankings[match.player1.id]
                    logger.debug(f"Set player1 ranking: {match.player1.name} = {match.player1.ranking}")
                else:
                    # Try to get detailed player info
                    player_data = self.get_player_details_corrected(match.player1.id)
                    if player_data:
                        ranking = self.extract_player_ranking(player_data)
                        if ranking:
                            match.player1.ranking = ranking
                            logger.debug(f"Found player1 ranking via details: {match.player1.name} = {ranking}")
            
            # Enhance player 2
            if match.player2 and match.player2.id:
                if match.player2.id in rankings:
                    match.player2.ranking = rankings[match.player2.id]
                    logger.debug(f"Set player2 ranking: {match.player2.name} = {match.player2.ranking}")
                else:
                    # Try to get detailed player info
                    player_data = self.get_player_details_corrected(match.player2.id)
                    if player_data:
                        ranking = self.extract_player_ranking(player_data)
                        if ranking:
                            match.player2.ranking = ranking
                            logger.debug(f"Found player2 ranking via details: {match.player2.name} = {ranking}")
            
            return match
            
        except Exception as e:
            logger.error(f"Error enhancing match with rankings: {e}")
            return match
    
    def enhance_matches_with_rankings(self, matches: List[TennisMatch]) -> List[TennisMatch]:
        """
        Enhance multiple matches with player rankings
        
        Args:
            matches: List of TennisMatch objects
            
        Returns:
            List of enhanced matches
        """
        if not matches:
            return matches
        
        logger.info(f"Enhancing {len(matches)} matches with ranking data")
        
        # Get rankings mapping once for all matches
        rankings = self.get_rankings_mapping()
        
        enhanced_matches = []
        for match in matches:
            try:
                enhanced_match = self._enhance_single_match(match, rankings)
                enhanced_matches.append(enhanced_match)
            except Exception as e:
                logger.error(f"Error enhancing match {match.id}: {e}")
                enhanced_matches.append(match)  # Add original match on error
        
        # Count how many matches have rankings
        ranked_count = sum(1 for m in enhanced_matches 
                          if (m.player1 and m.player1.ranking) or (m.player2 and m.player2.ranking))
        
        logger.info(f"Enhanced {ranked_count}/{len(enhanced_matches)} matches with ranking data")
        return enhanced_matches
    
    def _enhance_single_match(self, match: TennisMatch, rankings: Dict[int, int]) -> TennisMatch:
        """Enhance a single match with rankings from the mapping"""
        # Enhance player 1
        if match.player1 and match.player1.id and match.player1.id in rankings:
            match.player1.ranking = rankings[match.player1.id]
        
        # Enhance player 2
        if match.player2 and match.player2.id and match.player2.id in rankings:
            match.player2.ranking = rankings[match.player2.id]
        
        return match
    
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
    
    def clear_ranking_cache(self):
        """Clear ranking and player caches"""
        self.ranking_cache.clear()
        self.player_cache.clear()
        logger.info("Ranking caches cleared")


class RankingIntegrationService:
    """Service to integrate ranking data into the existing tennis system"""
    
    def __init__(self, api_client: EnhancedRankingClient = None):
        self.client = api_client or EnhancedRankingClient()
    
    def update_match_rankings(self, matches: List[TennisMatch]) -> List[TennisMatch]:
        """
        Update a list of matches with current ranking data
        
        Args:
            matches: List of matches to update
            
        Returns:
            Updated matches with ranking data
        """
        return self.client.enhance_matches_with_rankings(matches)
    
    def get_player_ranking(self, player_key: int, match_type: str = 'singles') -> Optional[int]:
        """
        Get current ranking for a specific player
        
        Args:
            player_key: Player's API key
            match_type: 'singles' or 'doubles'
            
        Returns:
            Current ranking or None
        """
        player_data = self.client.get_player_details_corrected(player_key)
        if player_data:
            return self.client.extract_player_ranking(player_data, match_type)
        return None
    
    def get_ranking_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about available ranking data
        
        Returns:
            Statistics about rankings
        """
        try:
            atp_rankings = self.client.get_standings_corrected('ATP')
            wta_rankings = self.client.get_standings_corrected('WTA')
            
            return {
                'atp_players': len(atp_rankings),
                'wta_players': len(wta_rankings),
                'total_players': len(atp_rankings) + len(wta_rankings),
                'cache_size': len(self.client.ranking_cache) + len(self.client.player_cache),
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting ranking statistics: {e}")
            return {
                'error': str(e),
                'last_updated': datetime.now().isoformat()
            }


def test_ranking_integration():
    """Test the enhanced ranking integration"""
    print("🏆 Testing Enhanced Ranking Integration")
    print("=" * 50)
    
    try:
        # Initialize enhanced client
        client = EnhancedRankingClient()
        
        if not client.api_key:
            print("❌ API_TENNIS_KEY not configured - cannot test")
            return
        
        # Test get_standings_corrected
        print("\n1. Testing get_standings_corrected...")
        for event_type in ['ATP', 'WTA']:
            standings = client.get_standings_corrected(event_type)
            print(f"   {event_type}: {len(standings)} players")
            
            if standings:
                top_player = standings[0]
                print(f"   Top {event_type}: {top_player.get('player')} "
                      f"(Rank {top_player.get('place')}, Points: {top_player.get('points')})")
        
        # Test get_player_details_corrected with sample player keys
        print("\n2. Testing get_player_details_corrected...")
        sample_keys = [2172, 521]  # From cached data
        
        for key in sample_keys:
            player_data = client.get_player_details_corrected(key)
            if player_data:
                name = player_data.get('player_name', 'Unknown')
                country = player_data.get('player_country', 'Unknown')
                ranking = client.extract_player_ranking(player_data)
                print(f"   Player {key}: {name} ({country}) - Ranking: {ranking}")
            else:
                print(f"   Player {key}: No data found")
        
        # Test fixtures with rankings
        print("\n3. Testing fixtures with rankings...")
        today = datetime.now().strftime('%Y-%m-%d')
        matches = client.get_fixtures_with_rankings(date_start=today, date_stop=today)
        
        ranked_matches = [m for m in matches if 
                         (m.player1 and m.player1.ranking) or (m.player2 and m.player2.ranking)]
        
        print(f"   Total matches: {len(matches)}")
        print(f"   Matches with rankings: {len(ranked_matches)}")
        
        for match in ranked_matches[:3]:  # Show first 3
            p1_rank = match.player1.ranking if match.player1 else "N/A"
            p2_rank = match.player2.ranking if match.player2 else "N/A"
            print(f"   {match.player1.name if match.player1 else 'Unknown'} (#{p1_rank}) vs "
                  f"{match.player2.name if match.player2 else 'Unknown'} (#{p2_rank})")
        
        # Test ranking service
        print("\n4. Testing RankingIntegrationService...")
        service = RankingIntegrationService(client)
        stats = service.get_ranking_statistics()
        print(f"   Stats: {stats}")
        
        print("\n✅ Enhanced ranking integration test completed successfully")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run test
    test_ranking_integration()