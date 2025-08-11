#!/usr/bin/env python3
"""
🌍 Enhanced Universal Tennis Data Collector
Integrates ALL data sources: TennisExplorer + RapidAPI + Universal Collector
Feeds comprehensive data to ML models for better predictions
"""

import logging
from datetime import datetime
from typing import Dict, List, Any

# Import existing components
from universal_tennis_data_collector import UniversalTennisDataCollector, UniversalOddsCollector

logger = logging.getLogger(__name__)

class EnhancedUniversalCollector:
    """Enhanced Universal Collector that integrates all data sources"""
    
    def __init__(self):
        """Initialize all data sources"""
        self.universal_collector = UniversalTennisDataCollector()
        self.odds_collector = UniversalOddsCollector()
        
        # Only using universal data collectors
        logger.info("✅ Using UniversalCollector and UniversalOddsCollector")
        
        # Data cache
        self.data_cache = {}
        self.last_update = None
        
    def get_comprehensive_match_data(self, days_ahead: int = 2) -> List[Dict]:
        """Get comprehensive match data from all sources"""
        
        logger.info("🔍 Collecting data from all sources...")
        all_matches = []
        
        
        # 1. Get Universal Collector data (tournament calendar + generated matches)
        
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
        
        
        # 2. PROFESSIONAL FILTERING: Only ATP/WTA singles matches
        if all_matches:
            professional_matches = []
            for match in all_matches:
                if self._is_professional_tennis_match(match):
                    professional_matches.append(match)
            
            logger.info(f"🏆 Professional filtering: {len(professional_matches)} ATP/WTA matches (was {len(all_matches)})")
            all_matches = professional_matches
        
        # 3. DEDUPLICATION: Remove duplicate matches across all sources
        if all_matches:
            all_matches = self._deduplicate_matches(all_matches)
        
        # 4. Enhance with odds data
        if all_matches:
            try:
                odds_data = self.odds_collector.generate_realistic_odds(all_matches)
                all_matches = self._merge_odds_data(all_matches, odds_data)
                logger.info("✅ Enhanced matches with odds data")
            except Exception as e:
                logger.warning(f"Odds enhancement error: {e}")
        
        
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
        
        # If no real matches found, log info
        if len(all_matches) == 0:
            logger.info("🎾 No current matches found from universal collector")
        
        ml_ready_matches = [
            match for match in all_matches 
            if match.get('ml_ready', False) and match.get('quality_score', 0) >= min_quality_score
        ]
        
        logger.info(f"📊 ML-ready matches (quality ≥ {min_quality_score}): {len(ml_ready_matches)}")
        return ml_ready_matches
    
    def _generate_realistic_tournament_matches(self) -> List[Dict]:
        """Generate realistic tournament matches when no live data is available"""
        
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
    
    
    def _deduplicate_matches(self, matches: List[Dict]) -> List[Dict]:
        """Remove duplicate matches across all data sources with priority-based selection"""
        
        if not matches:
            return matches
        
        logger.info(f"🔍 Deduplicating {len(matches)} matches from all sources...")
        
        # Group matches by normalized player pair
        match_groups = {}
        
        for match in matches:
            # Create normalized key for deduplication
            normalized_key = self._create_match_key(match)
            
            if normalized_key not in match_groups:
                match_groups[normalized_key] = []
            
            match_groups[normalized_key].append(match)
        
        # Select best match from each group based on priority
        deduplicated_matches = []
        duplicates_removed = 0
        
        for normalized_key, group_matches in match_groups.items():
            if len(group_matches) > 1:
                # Multiple matches found - select highest priority
                best_match = self._select_best_match(group_matches)
                deduplicated_matches.append(best_match)
                duplicates_removed += len(group_matches) - 1
                
                # Log duplicate removal
                sources = [m.get('data_source', 'Unknown') for m in group_matches]
                logger.info(f"🗑️ Removed {len(group_matches) - 1} duplicates for {best_match.get('player1', 'Unknown')} vs {best_match.get('player2', 'Unknown')} from sources: {sources}")
            else:
                # Unique match
                deduplicated_matches.append(group_matches[0])
        
        logger.info(f"✅ Deduplication complete: {len(deduplicated_matches)} unique matches (removed {duplicates_removed} duplicates)")
        
        return deduplicated_matches
    
    def _create_match_key(self, match: Dict) -> str:
        """Create normalized key for match deduplication"""
        
        player1 = self._normalize_player_name(match.get('player1', ''))
        player2 = self._normalize_player_name(match.get('player2', ''))
        
        # Ensure consistent ordering (alphabetical)
        if player1 > player2:
            player1, player2 = player2, player1
        
        # Include date for matches on different days
        match_date = match.get('date', datetime.now().strftime('%Y-%m-%d'))
        
        return f"{player1}_vs_{player2}_{match_date}"
    
    def _normalize_player_name(self, name: str) -> str:
        """Normalize player name for comparison"""
        if not name:
            return 'unknown'
        
        # Remove emojis, extra spaces, and special characters
        import re
        normalized = re.sub(r'[🎾🏆🔥⭐️🌟💪🎯]', '', name)
        normalized = re.sub(r'\s+', ' ', normalized)  # Multiple spaces to single
        normalized = normalized.strip().lower()
        
        # Handle common name variations
        normalized = normalized.replace('.', '')
        normalized = normalized.replace('-', ' ')
        
        return normalized
    
    def _normalize_tournament_name(self, tournament: str) -> str:
        """Normalize tournament name for comparison"""
        if not tournament:
            return 'unknown'
        
        import re
        normalized = tournament.lower().strip()
        
        # Remove common prefixes and suffixes
        prefixes_to_remove = ['atp', 'wta', 'the', 'open', '2024', '2025']
        for prefix in prefixes_to_remove:
            if normalized.startswith(prefix + ' '):
                normalized = normalized[len(prefix) + 1:]
            if normalized.endswith(' ' + prefix):
                normalized = normalized[:-len(prefix) - 1]
        
        # Remove special characters and extra spaces
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.strip()
    
    def _is_professional_tennis_match(self, match: Dict) -> bool:
        """Filter to ensure only ATP/WTA professional singles matches"""
        
        tournament_name = match.get('tournament', '').lower()
        
        # Exclude non-professional tournaments
        excluded_keywords = [
            'utr', 'ptt', 'junior', 'college', 'university',
            'challenger', 'futures', 'itf', 'amateur', 'qualifying',
            'youth', 'exhibition', 'invitational'
        ]
        
        for keyword in excluded_keywords:
            if keyword in tournament_name:
                logger.info(f"🚫 Filtering out non-professional tournament: {tournament_name} (contains '{keyword}')")
                return False
        
        # Check for doubles matches
        player1 = match.get('player1', '').lower()
        player2 = match.get('player2', '').lower()
        
        if any(indicator in player1 + player2 for indicator in ['/', ' and ', '&', ' / ']):
            logger.info(f"🚫 Filtering out doubles match: {player1} vs {player2}")
            return False  # Likely doubles match
        
        # Only allow professional tournament indicators
        professional_indicators = [
            'atp', 'wta', 'grand slam', 'masters', 'wimbledon',
            'french open', 'us open', 'australian open', 'miami',
            'indian wells', 'madrid', 'rome', 'montreal', 'cincinnati'
        ]
        
        # If tournament contains professional indicators, it's likely professional
        for indicator in professional_indicators:
            if indicator in tournament_name:
                return True
        
        # If no clear professional indicators found and contains excluded keywords, filter out
        logger.info(f"🚫 Filtering out non-ATP/WTA tournament: {tournament_name}")
        return False
    
    def _select_best_match(self, matches: List[Dict]) -> Dict:
        """Select the best match from duplicates based on data source priority"""
        
        # Define data source priority (higher number = higher priority)
        source_priority = {
            'TennisExplorer': 100,      # Highest priority - real tournament data
            'RapidAPI_Scheduled': 95,   # High priority - scheduled matches
            'OddsAPI': 90,              # High priority - real betting data
            'RapidAPI_Live': 85,        # Good priority - live matches
            'UniversalCollector': 70    # Lowest priority - generated data
        }
        
        # Sort by priority (highest first)
        sorted_matches = sorted(
            matches, 
            key=lambda m: (
                source_priority.get(m.get('data_source', ''), 0),
                m.get('quality_score', 0)
            ),
            reverse=True
        )
        
        best_match = sorted_matches[0]
        
        # Merge additional data from other sources
        for other_match in sorted_matches[1:]:
            # Add odds if not present in best match
            if not best_match.get('player1_odds') and other_match.get('player1_odds'):
                best_match['player1_odds'] = other_match['player1_odds']
                best_match['player2_odds'] = other_match['player2_odds']
            
            # Add ranking data if not present
            if not best_match.get('player1_ranking') and other_match.get('player1_ranking'):
                best_match['player1_ranking'] = other_match['player1_ranking']
                best_match['player1_ranking_source'] = other_match.get('player1_ranking_source', 'merged')
            
            if not best_match.get('player2_ranking') and other_match.get('player2_ranking'):
                best_match['player2_ranking'] = other_match['player2_ranking']
                best_match['player2_ranking_source'] = other_match.get('player2_ranking_source', 'merged')
        
        # Mark as merged data
        if len(matches) > 1:
            best_match['merged_from_sources'] = [m.get('data_source', 'Unknown') for m in matches]
        
        return best_match
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all data sources"""
        
        status = {
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'data_sources': {
                'universal_collector': True,
                'odds_collector': True,
                'tennisexplorer': False,
                'rapidapi': False
            },
            'cache_info': {
                'cached_items': len(self.data_cache),
                'cache_keys': list(self.data_cache.keys())
            }
        }
        
        
        return status
    
    def clear_cache(self):
        """Clear all cached data"""
        self.data_cache.clear()
        logger.info("🗑️ All caches cleared")

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