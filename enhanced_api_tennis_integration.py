#!/usr/bin/env python3
"""
Enhanced API-Tennis Integration After Payment Upgrade
Optimized to leverage paid tier benefits and fix identified issues
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedAPITennisIntegration:
    """Enhanced API-Tennis integration leveraging paid tier benefits"""
    
    def __init__(self):
        # Load API client
        try:
            from api.api_tennis_integration import get_api_tennis_client
            self.client = get_api_tennis_client()
            logger.info("✅ Enhanced API-Tennis client loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load API client: {e}")
            self.client = None
        
        # Player ranking cache
        self.ranking_cache = {}
        self.ranking_cache_timestamp = None
        self.cache_duration = timedelta(hours=6)  # Cache rankings for 6 hours
    
    def get_enhanced_player_rankings(self, force_refresh: bool = False) -> Dict[int, Dict[str, Any]]:
        """
        Get comprehensive player rankings with enhanced data from paid tier
        
        Args:
            force_refresh: Force refresh of ranking cache
            
        Returns:
            Dictionary mapping player_key to ranking data
        """
        # Check if cache is still valid
        if (not force_refresh and 
            self.ranking_cache and 
            self.ranking_cache_timestamp and 
            datetime.now() - self.ranking_cache_timestamp < self.cache_duration):
            logger.info("Using cached ranking data")
            return self.ranking_cache
        
        logger.info("🏆 Fetching enhanced player rankings...")
        
        ranking_data = {}
        
        try:
            # Get ATP rankings - paid tier provides comprehensive data
            logger.info("Fetching ATP rankings...")
            atp_standings = self.client.get_standings('ATP')
            
            if isinstance(atp_standings, list):
                for player in atp_standings:
                    if isinstance(player, dict):
                        player_key = player.get('player_key')
                        if player_key:
                            ranking_data[int(player_key)] = {
                                'name': player.get('player', ''),
                                'rank': int(player.get('place', 0)) if player.get('place') else None,
                                'points': int(player.get('points', 0)) if player.get('points') else None,
                                'country': player.get('country', ''),
                                'tour': 'ATP',
                                'movement': player.get('movement', ''),
                                'last_updated': datetime.now().isoformat()
                            }
                
                logger.info(f"✅ Loaded {len([r for r in ranking_data.values() if r['tour'] == 'ATP'])} ATP rankings")
            
            # Get WTA rankings
            logger.info("Fetching WTA rankings...")
            wta_standings = self.client.get_standings('WTA')
            
            if isinstance(wta_standings, list):
                for player in wta_standings:
                    if isinstance(player, dict):
                        player_key = player.get('player_key')
                        if player_key:
                            ranking_data[int(player_key)] = {
                                'name': player.get('player', ''),
                                'rank': int(player.get('place', 0)) if player.get('place') else None,
                                'points': int(player.get('points', 0)) if player.get('points') else None,
                                'country': player.get('country', ''),
                                'tour': 'WTA',
                                'movement': player.get('movement', ''),
                                'last_updated': datetime.now().isoformat()
                            }
                
                logger.info(f"✅ Loaded {len([r for r in ranking_data.values() if r['tour'] == 'WTA'])} WTA rankings")
            
            # Update cache
            self.ranking_cache = ranking_data
            self.ranking_cache_timestamp = datetime.now()
            
            # Save to persistent cache file
            cache_file = 'cache/enhanced_player_rankings.json'
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            
            cache_data = {
                'timestamp': self.ranking_cache_timestamp.isoformat(),
                'rankings': ranking_data,
                'total_players': len(ranking_data),
                'atp_players': len([r for r in ranking_data.values() if r['tour'] == 'ATP']),
                'wta_players': len([r for r in ranking_data.values() if r['tour'] == 'WTA'])
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"💾 Cached {len(ranking_data)} player rankings to {cache_file}")
            
            return ranking_data
            
        except Exception as e:
            logger.error(f"❌ Error fetching rankings: {e}")
            # Try to load from persistent cache as fallback
            return self._load_ranking_fallback()
    
    def _load_ranking_fallback(self) -> Dict[int, Dict[str, Any]]:
        """Load rankings from persistent cache as fallback"""
        try:
            cache_file = 'cache/enhanced_player_rankings.json'
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                cache_timestamp = datetime.fromisoformat(cache_data['timestamp'])
                if datetime.now() - cache_timestamp < timedelta(days=1):  # Accept day-old cache as fallback
                    logger.info(f"✅ Loaded {cache_data['total_players']} rankings from fallback cache")
                    return cache_data['rankings']
        except Exception as e:
            logger.error(f"❌ Fallback cache load failed: {e}")
        
        return {}
    
    def get_enhanced_fixtures_with_rankings(self, 
                                          date_start: str = None, 
                                          date_stop: str = None) -> List[Dict[str, Any]]:
        """
        Get fixtures enhanced with comprehensive ranking data
        
        Args:
            date_start: Start date (YYYY-MM-DD)
            date_stop: End date (YYYY-MM-DD)
            
        Returns:
            List of enhanced match data with rankings
        """
        if not date_start:
            date_start = datetime.now().strftime('%Y-%m-%d')
        
        if not date_stop:
            date_stop = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')
        
        logger.info(f"🎾 Fetching enhanced fixtures from {date_start} to {date_stop}")
        
        try:
            # Get fixtures from API
            matches = self.client.get_fixtures(date_start=date_start, date_stop=date_stop)
            
            # Get enhanced rankings
            rankings = self.get_enhanced_player_rankings()
            
            enhanced_matches = []
            
            for match in matches:
                # Convert to dictionary format for processing
                match_data = match.to_dict() if hasattr(match, 'to_dict') else match
                
                # Enhance with ranking data
                if match_data.get('player1') and match_data['player1'].get('id'):
                    player1_id = match_data['player1']['id']
                    if player1_id in rankings:
                        ranking_info = rankings[player1_id]
                        match_data['player1'].update({
                            'ranking': ranking_info.get('rank'),
                            'points': ranking_info.get('points'),
                            'country': ranking_info.get('country'),
                            'tour': ranking_info.get('tour'),
                            'ranking_movement': ranking_info.get('movement')
                        })
                
                if match_data.get('player2') and match_data['player2'].get('id'):
                    player2_id = match_data['player2']['id']
                    if player2_id in rankings:
                        ranking_info = rankings[player2_id]
                        match_data['player2'].update({
                            'ranking': ranking_info.get('rank'),
                            'points': ranking_info.get('points'),
                            'country': ranking_info.get('country'),
                            'tour': ranking_info.get('tour'),
                            'ranking_movement': ranking_info.get('movement')
                        })
                
                # Add enhanced match metadata
                match_data['enhanced_data'] = {
                    'has_rankings': bool(
                        match_data.get('player1', {}).get('ranking') and 
                        match_data.get('player2', {}).get('ranking')
                    ),
                    'is_underdog_scenario': self._is_underdog_scenario(match_data),
                    'ranking_gap': self._calculate_ranking_gap(match_data),
                    'data_quality_score': self._calculate_data_quality_score(match_data),
                    'enhanced_at': datetime.now().isoformat()
                }
                
                enhanced_matches.append(match_data)
            
            # Filter and sort by relevance for underdog analysis
            relevant_matches = self._filter_relevant_matches(enhanced_matches)
            
            logger.info(f"✅ Enhanced {len(enhanced_matches)} matches")
            logger.info(f"📊 Found {len(relevant_matches)} relevant matches for underdog analysis")
            
            return relevant_matches
            
        except Exception as e:
            logger.error(f"❌ Error fetching enhanced fixtures: {e}")
            return []
    
    def _is_underdog_scenario(self, match_data: Dict[str, Any]) -> bool:
        """Check if match is a valid underdog scenario (ranks 10-300)"""
        try:
            player1_rank = match_data.get('player1', {}).get('ranking')
            player2_rank = match_data.get('player2', {}).get('ranking')
            
            if not player1_rank or not player2_rank:
                return False
            
            underdog_rank = max(player1_rank, player2_rank)
            favorite_rank = min(player1_rank, player2_rank)
            
            # Underdog must be in 10-300 range
            if not (10 <= underdog_rank <= 300):
                return False
            
            # Favorite must not be in top-9
            if favorite_rank < 10:
                return False
            
            # Must have meaningful ranking gap (at least 5 positions)
            if abs(player1_rank - player2_rank) < 5:
                return False
            
            return True
            
        except (TypeError, ValueError):
            return False
    
    def _calculate_ranking_gap(self, match_data: Dict[str, Any]) -> Optional[int]:
        """Calculate ranking gap between players"""
        try:
            player1_rank = match_data.get('player1', {}).get('ranking')
            player2_rank = match_data.get('player2', {}).get('ranking')
            
            if player1_rank and player2_rank:
                return abs(player1_rank - player2_rank)
        except (TypeError, ValueError):
            pass
        
        return None
    
    def _calculate_data_quality_score(self, match_data: Dict[str, Any]) -> float:
        """Calculate data quality score (0.0 to 1.0)"""
        score = 0.0
        
        # Player data completeness
        if match_data.get('player1', {}).get('name'):
            score += 0.15
        if match_data.get('player2', {}).get('name'):
            score += 0.15
        
        # Ranking data availability
        if match_data.get('player1', {}).get('ranking'):
            score += 0.25
        if match_data.get('player2', {}).get('ranking'):
            score += 0.25
        
        # Tournament information
        if match_data.get('tournament_name'):
            score += 0.1
        
        # Match timing
        if match_data.get('start_time'):
            score += 0.1
        
        return score
    
    def _filter_relevant_matches(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter matches relevant for underdog betting analysis"""
        relevant_matches = []
        
        for match in matches:
            # Must be ATP or WTA singles
            event_type = match.get('event_type', '').lower()
            if not any(tour in event_type for tour in ['atp', 'wta']):
                continue
            
            if 'doubles' in event_type:
                continue
            
            # Must have ranking data
            if not match.get('enhanced_data', {}).get('has_rankings'):
                continue
            
            # Must be a valid underdog scenario
            if not match.get('enhanced_data', {}).get('is_underdog_scenario'):
                continue
            
            # Must have good data quality
            quality_score = match.get('enhanced_data', {}).get('data_quality_score', 0)
            if quality_score < 0.7:
                continue
            
            relevant_matches.append(match)
        
        # Sort by relevance (ranking gap, tournament importance, data quality)
        relevant_matches.sort(key=lambda m: (
            -m.get('enhanced_data', {}).get('ranking_gap', 0),
            -m.get('enhanced_data', {}).get('data_quality_score', 0)
        ))
        
        return relevant_matches
    
    def get_us_open_matches(self) -> List[Dict[str, Any]]:
        """Get US Open matches with enhanced data"""
        us_open_start = '2025-08-25'
        us_open_end = '2025-09-08'
        
        logger.info("🏆 Fetching US Open matches with enhanced data...")
        
        all_matches = self.get_enhanced_fixtures_with_rankings(us_open_start, us_open_end)
        
        # Filter for US Open specifically
        us_open_matches = []
        for match in all_matches:
            tournament_name = match.get('tournament_name', '').lower()
            if 'us open' in tournament_name:
                match['tournament_info'] = {
                    'is_grand_slam': True,
                    'surface': 'Hard',
                    'location': 'New York',
                    'tournament_level': 'Grand Slam'
                }
                us_open_matches.append(match)
        
        logger.info(f"🎾 Found {len(us_open_matches)} US Open matches")
        return us_open_matches
    
    def create_enhanced_match_report(self) -> Dict[str, Any]:
        """Create comprehensive match report with enhanced data"""
        logger.info("📊 Creating enhanced match report...")
        
        # Get today's and tomorrow's matches
        today_matches = self.get_enhanced_fixtures_with_rankings()
        us_open_matches = self.get_us_open_matches()
        
        # Get rankings summary
        rankings = self.get_enhanced_player_rankings()
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'api_status': {
                'client_configured': self.client is not None,
                'ranking_cache_valid': bool(self.ranking_cache),
                'total_players_in_rankings': len(rankings)
            },
            'today_matches': {
                'total_matches': len(today_matches),
                'underdog_opportunities': len([m for m in today_matches if m.get('enhanced_data', {}).get('is_underdog_scenario')]),
                'high_quality_matches': len([m for m in today_matches if m.get('enhanced_data', {}).get('data_quality_score', 0) > 0.8]),
                'matches': today_matches[:10]  # Top 10 matches
            },
            'us_open': {
                'total_matches': len(us_open_matches),
                'underdog_opportunities': len([m for m in us_open_matches if m.get('enhanced_data', {}).get('is_underdog_scenario')]),
                'sample_matches': us_open_matches[:5]
            },
            'ranking_summary': {
                'atp_players': len([r for r in rankings.values() if r['tour'] == 'ATP']),
                'wta_players': len([r for r in rankings.values() if r['tour'] == 'WTA']),
                'top_10_atp': [
                    {'name': r['name'], 'rank': r['rank'], 'points': r['points']} 
                    for r in sorted(rankings.values(), key=lambda x: x.get('rank', 999)) 
                    if r['tour'] == 'ATP' and r.get('rank') and r['rank'] <= 10
                ][:10],
                'top_10_wta': [
                    {'name': r['name'], 'rank': r['rank'], 'points': r['points']} 
                    for r in sorted(rankings.values(), key=lambda x: x.get('rank', 999)) 
                    if r['tour'] == 'WTA' and r.get('rank') and r['rank'] <= 10
                ][:10]
            },
            'data_quality_improvements': {
                'comprehensive_rankings': True,
                'player_points_available': bool(rankings and any(r.get('points') for r in rankings.values())),
                'country_data_available': bool(rankings and any(r.get('country') for r in rankings.values())),
                'movement_tracking': bool(rankings and any(r.get('movement') for r in rankings.values())),
                'tournament_detection_improved': len(us_open_matches) > 0
            }
        }
        
        # Save report
        report_filename = f"enhanced_api_tennis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Enhanced report saved to: {report_filename}")
        return report
    
    def test_automated_system_integration(self):
        """Test integration with the automated tennis prediction service"""
        logger.info("🧪 Testing automated system integration...")
        
        # Get enhanced matches
        matches = self.get_enhanced_fixtures_with_rankings()
        
        # Test compatibility with existing prediction service
        compatible_matches = []
        
        for match in matches:
            # Convert to format expected by automated service
            if match.get('enhanced_data', {}).get('is_underdog_scenario'):
                # Create compatible format
                compatible_match = {
                    'event_key': match.get('id'),
                    'event_first_player': match.get('player1', {}).get('name', ''),
                    'event_second_player': match.get('player2', {}).get('name', ''),
                    'tournament_name': match.get('tournament_name', ''),
                    'event_type_type': match.get('event_type', ''),
                    'event_final_result': '-',  # Upcoming match
                    'player1_rank': match.get('player1', {}).get('ranking'),
                    'player2_rank': match.get('player2', {}).get('ranking'),
                    'ranking_gap': match.get('enhanced_data', {}).get('ranking_gap'),
                    'data_quality_score': match.get('enhanced_data', {}).get('data_quality_score')
                }
                compatible_matches.append(compatible_match)
        
        logger.info(f"✅ {len(compatible_matches)} matches ready for automated prediction system")
        
        return compatible_matches

def main():
    """Main function to demonstrate enhanced API integration"""
    integration = EnhancedAPITennisIntegration()
    
    # Create comprehensive report
    report = integration.create_enhanced_match_report()
    
    # Test automated system integration
    compatible_matches = integration.test_automated_system_integration()
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("🚀 ENHANCED API-TENNIS INTEGRATION SUMMARY")
    logger.info("="*70)
    logger.info(f"📊 Total players in rankings: {report['api_status']['total_players_in_rankings']}")
    logger.info(f"🎾 Today's matches: {report['today_matches']['total_matches']}")
    logger.info(f"🎯 Underdog opportunities: {report['today_matches']['underdog_opportunities']}")
    logger.info(f"🏆 US Open matches: {report['us_open']['total_matches']}")
    logger.info(f"🔥 High-quality matches: {report['today_matches']['high_quality_matches']}")
    logger.info(f"🤖 Compatible with automated system: {len(compatible_matches)} matches")
    
    improvements = report['data_quality_improvements']
    logger.info("\n💡 DATA QUALITY IMPROVEMENTS:")
    for key, value in improvements.items():
        status = "✅" if value else "❌"
        logger.info(f"   {status} {key.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    main()