#!/usr/bin/env python3
"""
🌍 Enhanced Universal Tennis Data Collector
Integrates ALL data sources: TennisExplorer + RapidAPI + Universal Collector
Feeds comprehensive data to ML models for better predictions
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import existing components
from universal_tennis_data_collector import UniversalTennisDataCollector, UniversalOddsCollector
from tennisexplorer_integration import TennisExplorerIntegration
from rapidapi_tennis_client import RapidAPITennisClient

logger = logging.getLogger(__name__)

class EnhancedUniversalCollector:
    """Enhanced Universal Collector that integrates all data sources"""
    
    def __init__(self):
        """Initialize all data sources"""
        self.universal_collector = UniversalTennisDataCollector()
        self.odds_collector = UniversalOddsCollector()
        
        # Initialize TennisExplorer
        self.tennisexplorer = None
        try:
            self.tennisexplorer = TennisExplorerIntegration()
            if self.tennisexplorer.initialize():
                logger.info("✅ TennisExplorer integration initialized")
            else:
                logger.warning("⚠️ TennisExplorer integration failed")
                self.tennisexplorer = None
        except Exception as e:
            logger.warning(f"TennisExplorer not available: {e}")
            self.tennisexplorer = None
        
        # Initialize RapidAPI
        self.rapidapi = None
        try:
            self.rapidapi = RapidAPITennisClient()
            logger.info("✅ RapidAPI Tennis client initialized")
        except Exception as e:
            logger.warning(f"RapidAPI not available: {e}")
            self.rapidapi = None
        
        # Data cache
        self.data_cache = {}
        self.last_update = None
        
    def get_comprehensive_match_data(self, days_ahead: int = 2) -> List[Dict]:
        """Get comprehensive match data from all sources"""
        
        logger.info("🔍 Collecting data from all sources...")
        all_matches = []
        
        # 1. Get TennisExplorer data (highest priority - real matches)
        if self.tennisexplorer:
            try:
                te_matches = self.tennisexplorer.get_enhanced_match_data(days_ahead)
                if te_matches:
                    logger.info(f"✅ TennisExplorer: {len(te_matches)} matches")
                    for match in te_matches:
                        match['data_source'] = 'TennisExplorer'
                        match['quality_score'] = 95  # Highest quality
                        all_matches.append(match)
            except Exception as e:
                logger.warning(f"TennisExplorer error: {e}")
        
        # 2. Get Universal Collector data (tournament calendar + generated matches)
        try:
            universal_matches = self.universal_collector.get_current_matches()
            if universal_matches:
                logger.info(f"✅ Universal Collector: {len(universal_matches)} matches")
                for match in universal_matches:
                    match['data_source'] = 'UniversalCollector'
                    match['quality_score'] = 70  # Medium quality
                    all_matches.append(match)
        except Exception as e:
            logger.warning(f"Universal Collector error: {e}")
        
        # 3. Enhance with odds data
        if all_matches:
            try:
                odds_data = self.odds_collector.generate_realistic_odds(all_matches)
                all_matches = self._merge_odds_data(all_matches, odds_data)
                logger.info("✅ Enhanced matches with odds data")
            except Exception as e:
                logger.warning(f"Odds enhancement error: {e}")
        
        # 4. Add RapidAPI rankings data for player enhancement
        if self.rapidapi and all_matches:
            try:
                all_matches = self._enhance_with_rapidapi_data(all_matches)
                logger.info("✅ Enhanced matches with RapidAPI rankings")
            except Exception as e:
                logger.warning(f"RapidAPI enhancement error: {e}")
        
        # 5. Calculate ML features for each match
        enhanced_matches = []
        for match in all_matches:
            try:
                enhanced_match = self._calculate_ml_features(match)
                enhanced_matches.append(enhanced_match)
            except Exception as e:
                logger.warning(f"ML feature calculation error: {e}")
                enhanced_matches.append(match)  # Add without ML features
        
        logger.info(f"📊 Total enhanced matches: {len(enhanced_matches)}")
        self.last_update = datetime.now()
        
        return enhanced_matches
    
    def _merge_odds_data(self, matches: List[Dict], odds_data: Dict) -> List[Dict]:
        """Merge odds data with match data"""
        
        enhanced_matches = []
        for match in matches:
            match_id = match.get('id', '')
            if match_id in odds_data:
                odds_info = odds_data[match_id]
                
                # Add odds information
                match['odds'] = {
                    'best_odds': odds_info.get('best_markets', {}).get('winner', {}),
                    'bookmaker_count': len(odds_info.get('bookmakers', [])),
                    'last_updated': odds_info.get('last_updated')
                }
                
                # Extract player odds
                winner_market = odds_info.get('best_markets', {}).get('winner', {})
                match['player1_odds'] = winner_market.get('player1', {}).get('odds', 2.0)
                match['player2_odds'] = winner_market.get('player2', {}).get('odds', 2.0)
            
            enhanced_matches.append(match)
        
        return enhanced_matches
    
    def _enhance_with_rapidapi_data(self, matches: List[Dict]) -> List[Dict]:
        """Enhance matches with RapidAPI rankings data"""
        
        # Get current rankings
        atp_rankings = None
        wta_rankings = None
        
        try:
            if self.rapidapi.get_remaining_requests() > 2:  # Only if we have requests left
                atp_rankings = self.rapidapi.get_atp_rankings()
                wta_rankings = self.rapidapi.get_wta_rankings()
        except Exception as e:
            logger.warning(f"Could not get rankings: {e}")
        
        enhanced_matches = []
        for match in matches:
            try:
                # Try to find player rankings
                player1_rank = self._get_player_ranking_from_rapidapi(
                    match.get('player1', ''), atp_rankings, wta_rankings
                )
                player2_rank = self._get_player_ranking_from_rapidapi(
                    match.get('player2', ''), atp_rankings, wta_rankings
                )
                
                if player1_rank:
                    match['player1_ranking'] = player1_rank
                    match['player1_ranking_source'] = 'RapidAPI'
                
                if player2_rank:
                    match['player2_ranking'] = player2_rank
                    match['player2_ranking_source'] = 'RapidAPI'
                
                enhanced_matches.append(match)
                
            except Exception as e:
                logger.warning(f"Error enhancing match with RapidAPI data: {e}")
                enhanced_matches.append(match)
        
        return enhanced_matches
    
    def _get_player_ranking_from_rapidapi(self, player_name: str, atp_rankings: List, wta_rankings: List) -> Optional[int]:
        """Find player ranking from RapidAPI data"""
        
        if not player_name:
            return None
        
        player_name_clean = player_name.replace('🎾 ', '').lower().strip()
        
        # Search in ATP rankings
        if atp_rankings:
            for player_data in atp_rankings:
                try:
                    name = player_data.get('team', {}).get('name', '').lower()
                    if player_name_clean in name or name in player_name_clean:
                        return player_data.get('ranking', None)
                except:
                    continue
        
        # Search in WTA rankings
        if wta_rankings:
            for player_data in wta_rankings:
                try:
                    name = player_data.get('team', {}).get('name', '').lower()
                    if player_name_clean in name or name in player_name_clean:
                        return player_data.get('ranking', None)
                except:
                    continue
        
        return None
    
    def _calculate_ml_features(self, match: Dict) -> Dict:
        """Calculate ML features for the match"""
        
        # Base match data
        ml_features = {
            'match_id': match.get('id', ''),
            'tournament': match.get('tournament', ''),
            'surface': match.get('surface', 'Hard'),
            'level': match.get('level', 'ATP 250'),
            'data_source': match.get('data_source', 'Unknown'),
            'quality_score': match.get('quality_score', 50)
        }
        
        # Player rankings
        player1_rank = match.get('player1_ranking', 50)
        player2_rank = match.get('player2_ranking', 50)
        
        ml_features.update({
            'player1_ranking': player1_rank,
            'player2_ranking': player2_rank,
            'ranking_difference': abs(player1_rank - player2_rank),
            'higher_ranked_player': 1 if player1_rank < player2_rank else 2,
            'ranking_gap_category': self._categorize_ranking_gap(abs(player1_rank - player2_rank))
        })
        
        # Odds-based features
        player1_odds = match.get('player1_odds', 2.0)
        player2_odds = match.get('player2_odds', 2.0)
        
        ml_features.update({
            'player1_odds': player1_odds,
            'player2_odds': player2_odds,
            'odds_favorite': 1 if player1_odds < player2_odds else 2,
            'odds_difference': abs(player1_odds - player2_odds),
            'total_probability': (1/player1_odds) + (1/player2_odds),
            'implied_probability_p1': 1/player1_odds,
            'implied_probability_p2': 1/player2_odds
        })
        
        # Surface and tournament features
        ml_features.update({
            'surface_encoded': self._encode_surface(match.get('surface', 'Hard')),
            'tournament_level_encoded': self._encode_tournament_level(match.get('level', 'ATP 250')),
            'is_grand_slam': 1 if 'Grand Slam' in match.get('level', '') else 0,
            'is_masters': 1 if 'ATP 1000' in match.get('level', '') or 'WTA 1000' in match.get('level', '') else 0
        })
        
        # Temporal features
        match_date = match.get('date', datetime.now().strftime('%Y-%m-%d'))
        try:
            match_datetime = datetime.strptime(match_date, '%Y-%m-%d')
            ml_features.update({
                'day_of_year': match_datetime.timetuple().tm_yday,
                'month': match_datetime.month,
                'is_weekend': 1 if match_datetime.weekday() >= 5 else 0
            })
        except:
            ml_features.update({
                'day_of_year': datetime.now().timetuple().tm_yday,
                'month': datetime.now().month,
                'is_weekend': 0
            })
        
        # Add ML features to original match data
        match['ml_features'] = ml_features
        match['ml_ready'] = True
        
        return match
    
    def _categorize_ranking_gap(self, gap: int) -> int:
        """Categorize ranking gap for ML"""
        if gap <= 10:
            return 0  # Small gap
        elif gap <= 30:
            return 1  # Medium gap
        elif gap <= 50:
            return 2  # Large gap
        else:
            return 3  # Very large gap
    
    def _encode_surface(self, surface: str) -> int:
        """Encode surface for ML"""
        surface_map = {
            'Hard': 0,
            'Clay': 1,
            'Grass': 2,
            'Indoor': 3
        }
        return surface_map.get(surface, 0)
    
    def _encode_tournament_level(self, level: str) -> int:
        """Encode tournament level for ML"""
        if 'Grand Slam' in level:
            return 4
        elif 'ATP 1000' in level or 'WTA 1000' in level:
            return 3
        elif 'ATP 500' in level or 'WTA 500' in level:
            return 2
        elif 'ATP 250' in level or 'WTA 250' in level:
            return 1
        else:
            return 0
    
    def get_ml_ready_matches(self, min_quality_score: int = 70) -> List[Dict]:
        """Get matches ready for ML prediction with minimum quality"""
        
        all_matches = self.get_comprehensive_match_data()
        
        # If no real matches found but sources are active, generate realistic samples
        if len(all_matches) == 0 and (self.tennisexplorer or self.rapidapi):
            logger.info("🎾 No current matches found, generating tournament-based sample matches...")
            all_matches = self._generate_realistic_tournament_matches()
        
        ml_ready_matches = [
            match for match in all_matches 
            if match.get('ml_ready', False) and match.get('quality_score', 0) >= min_quality_score
        ]
        
        logger.info(f"📊 ML-ready matches (quality ≥ {min_quality_score}): {len(ml_ready_matches)}")
        return ml_ready_matches
    
    def _generate_realistic_tournament_matches(self) -> List[Dict]:
        """Generate realistic tournament matches when no live data is available"""
        from datetime import datetime
        
        # NO FAKE TOURNAMENTS - Only use real live API data
        matches = []
        
        logger.info("🚫 No fake tournament data generated - using only real live API sources")
        return matches
    
    def get_underdog_opportunities(self, min_ranking_gap: int = 20) -> List[Dict]:
        """Get potential underdog opportunities for prediction"""
        
        ml_matches = self.get_ml_ready_matches()
        
        underdog_opportunities = []
        for match in ml_matches:
            ml_features = match.get('ml_features', {})
            ranking_gap = ml_features.get('ranking_difference', 0)
            
            if ranking_gap >= min_ranking_gap:
                # Calculate underdog potential
                match['underdog_potential'] = self._calculate_underdog_potential(match)
                underdog_opportunities.append(match)
        
        # Sort by underdog potential
        underdog_opportunities.sort(key=lambda x: x.get('underdog_potential', 0), reverse=True)
        
        logger.info(f"🎯 Underdog opportunities found: {len(underdog_opportunities)}")
        return underdog_opportunities
    
    def _calculate_underdog_potential(self, match: Dict) -> float:
        """Calculate underdog potential score (0-1)"""
        
        ml_features = match.get('ml_features', {})
        
        # Base score from ranking difference
        ranking_gap = ml_features.get('ranking_difference', 0)
        base_score = min(ranking_gap / 100, 0.5)  # Max 0.5 from ranking
        
        # Bonus for surface specialists (simplified)
        surface = match.get('surface', 'Hard')
        if surface == 'Clay':
            base_score += 0.1
        elif surface == 'Grass':
            base_score += 0.15
        
        # Bonus for data quality
        quality_score = match.get('quality_score', 50)
        quality_bonus = (quality_score - 50) / 200  # 0 to 0.25
        
        # Bonus for comprehensive data
        data_bonus = 0
        if match.get('data_source') == 'TennisExplorer':
            data_bonus += 0.1
        if 'player1_ranking_source' in match:
            data_bonus += 0.05
        
        total_score = min(base_score + quality_bonus + data_bonus, 1.0)
        return round(total_score, 3)
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all data sources"""
        
        status = {
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'data_sources': {
                'universal_collector': True,
                'odds_collector': True,
                'tennisexplorer': self.tennisexplorer is not None,
                'rapidapi': self.rapidapi is not None
            },
            'cache_info': {
                'cached_items': len(self.data_cache),
                'cache_keys': list(self.data_cache.keys())
            }
        }
        
        # Add RapidAPI status if available
        if self.rapidapi:
            try:
                rapidapi_status = self.rapidapi.get_status()
                status['rapidapi_status'] = rapidapi_status
            except:
                status['rapidapi_status'] = {'error': 'Could not get status'}
        
        return status
    
    def clear_cache(self):
        """Clear all cached data"""
        self.data_cache.clear()
        if self.rapidapi:
            self.rapidapi.clear_cache()
        logger.info("🗑️ All caches cleared")

def test_enhanced_collector():
    """Test the enhanced collector"""
    print("🧪 Testing Enhanced Universal Collector")
    print("=" * 50)
    
    collector = EnhancedUniversalCollector()
    
    # Test status
    print("\n📊 Data Sources Status:")
    status = collector.get_status()
    for source, available in status['data_sources'].items():
        status_emoji = "✅" if available else "❌"
        print(f"  {status_emoji} {source}")
    
    # Test comprehensive data collection
    print("\n🔍 Testing comprehensive data collection...")
    matches = collector.get_comprehensive_match_data()
    print(f"Found {len(matches)} total matches")
    
    # Test ML-ready matches
    print("\n🤖 Testing ML-ready matches...")
    ml_matches = collector.get_ml_ready_matches()
    print(f"Found {len(ml_matches)} ML-ready matches")
    
    if ml_matches:
        sample_match = ml_matches[0]
        print(f"\nSample ML features: {list(sample_match.get('ml_features', {}).keys())}")
    
    # Test underdog opportunities
    print("\n🎯 Testing underdog opportunities...")
    underdogs = collector.get_underdog_opportunities()
    print(f"Found {len(underdogs)} underdog opportunities")
    
    if underdogs:
        top_underdog = underdogs[0]
        print(f"Top opportunity: {top_underdog.get('underdog_potential', 0):.3f} potential")

if __name__ == "__main__":
    test_enhanced_collector()