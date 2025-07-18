#!/usr/bin/env python3
"""
🔗 TennisExplorer Integration Module
Integrates TennisExplorer scraper with existing tennis prediction system
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os
from dataclasses import asdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from tennisexplorer_scraper import TennisExplorerScraper, TennisMatch
from universal_tennis_data_collector import UniversalTennisDataCollector

logger = logging.getLogger(__name__)

class TennisExplorerIntegration:
    """Integrates TennisExplorer data with existing prediction system"""
    
    def __init__(self):
        self.scraper = TennisExplorerScraper()
        self.universal_collector = UniversalTennisDataCollector()
        self.data_cache = {}
        self.last_update = None
        
    def initialize(self) -> bool:
        """Initialize the integration system"""
        try:
            # Test connection
            if not self.scraper.test_connection():
                logger.error("Failed to connect to TennisExplorer")
                return False
                
            # Attempt login
            if self.scraper.login():
                logger.info("✅ TennisExplorer integration initialized with authentication")
            else:
                logger.warning("⚠️ TennisExplorer integration initialized without authentication")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize TennisExplorer integration: {e}")
            return False
            
    def get_enhanced_match_data(self, days_ahead: int = 2) -> List[Dict]:
        """Get enhanced match data combining TennisExplorer and universal collector"""
        try:
            # Get matches from TennisExplorer
            te_matches = self.scraper.get_current_matches(days_ahead)
            
            # Get matches from universal collector
            universal_matches = self.universal_collector.get_current_matches()
            
            # Combine and enhance the data
            enhanced_matches = []
            
            for te_match in te_matches:
                # Convert to dict for easier manipulation
                match_data = asdict(te_match)
                
                # Try to find corresponding match in universal data
                universal_match = self._find_matching_universal_match(te_match, universal_matches)
                
                if universal_match:
                    # Merge data from both sources
                    match_data.update({
                        'surface_confirmed': universal_match.get('surface', match_data['surface']),
                        'tournament_level': universal_match.get('level', 'Unknown'),
                        'tournament_location': universal_match.get('location', 'Unknown'),
                        'round_detailed': universal_match.get('round', match_data['round_info']),
                        'data_source': 'TennisExplorer + Universal'
                    })
                else:
                    match_data.update({
                        'surface_confirmed': match_data['surface'],
                        'tournament_level': 'Unknown',
                        'tournament_location': 'Unknown',
                        'round_detailed': match_data['round_info'],
                        'data_source': 'TennisExplorer'
                    })
                
                # Add player statistics if available
                match_data = self._enhance_with_player_stats(match_data)
                
                enhanced_matches.append(match_data)
                
            logger.info(f"Enhanced {len(enhanced_matches)} matches with combined data")
            return enhanced_matches
            
        except Exception as e:
            logger.error(f"Error getting enhanced match data: {e}")
            return []
            
    def _find_matching_universal_match(self, te_match: TennisMatch, universal_matches: List[Dict]) -> Optional[Dict]:
        """Find matching match in universal collector data"""
        for universal_match in universal_matches:
            # Simple name matching (can be improved with fuzzy matching)
            universal_players = universal_match.get('players', [])
            
            if len(universal_players) >= 2:
                # Check if player names match (allowing for variations)
                if (self._names_similar(te_match.player1, universal_players[0]) and 
                    self._names_similar(te_match.player2, universal_players[1])) or \
                   (self._names_similar(te_match.player1, universal_players[1]) and 
                    self._names_similar(te_match.player2, universal_players[0])):
                    return universal_match
                    
        return None
        
    def _names_similar(self, name1: str, name2: str) -> bool:
        """Check if two player names are similar (basic matching)"""
        # Simple similarity check - can be enhanced with fuzzy matching
        name1_parts = set(name1.lower().split())
        name2_parts = set(name2.lower().split())
        
        # Check if at least 50% of name parts match
        if len(name1_parts) == 0 or len(name2_parts) == 0:
            return False
            
        intersection = name1_parts.intersection(name2_parts)
        return len(intersection) / max(len(name1_parts), len(name2_parts)) >= 0.5
        
    def _enhance_with_player_stats(self, match_data: Dict) -> Dict:
        """Enhance match data with player statistics"""
        try:
            # Get player stats from cache or fetch new
            player1_stats = self._get_cached_player_stats(match_data['player1'])
            player2_stats = self._get_cached_player_stats(match_data['player2'])
            
            # Add player statistics to match data
            match_data.update({
                'player1_stats': player1_stats,
                'player2_stats': player2_stats,
                'player1_ranking': player1_stats.get('ranking', 'Unknown'),
                'player2_ranking': player2_stats.get('ranking', 'Unknown'),
                'player1_recent_form': self._analyze_recent_form(player1_stats.get('recent_matches', [])),
                'player2_recent_form': self._analyze_recent_form(player2_stats.get('recent_matches', []))
            })
            
        except Exception as e:
            logger.debug(f"Could not enhance match with player stats: {e}")
            
        return match_data
        
    def _get_cached_player_stats(self, player_name: str) -> Dict:
        """Get player stats from cache or fetch new"""
        cache_key = f"player_stats_{player_name}"
        
        # Check cache first (cache for 24 hours)
        if (cache_key in self.data_cache and 
            self.data_cache[cache_key].get('timestamp', 0) > 
            (datetime.now() - timedelta(hours=24)).timestamp()):
            return self.data_cache[cache_key]['data']
            
        # Fetch new stats
        try:
            stats = self.scraper.get_player_stats(player_name)
            
            # Cache the results
            self.data_cache[cache_key] = {
                'data': stats,
                'timestamp': datetime.now().timestamp()
            }
            
            return stats
            
        except Exception as e:
            logger.debug(f"Could not fetch stats for {player_name}: {e}")
            return {}
            
    def _analyze_recent_form(self, recent_matches: List[str]) -> Dict:
        """Analyze player's recent form"""
        if not recent_matches:
            return {'wins': 0, 'losses': 0, 'form': 'Unknown'}
            
        wins = 0
        losses = 0
        
        for match in recent_matches:
            # Simple win/loss detection (can be improved)
            if any(indicator in match.lower() for indicator in ['won', 'win', 'w']):
                wins += 1
            elif any(indicator in match.lower() for indicator in ['lost', 'loss', 'l']):
                losses += 1
                
        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0
        
        return {
            'wins': wins,
            'losses': losses,
            'total': total,
            'win_rate': round(win_rate, 1),
            'form': 'Good' if win_rate >= 60 else 'Average' if win_rate >= 40 else 'Poor'
        }
        
    def get_matches_for_prediction(self, min_ranking_diff: int = 50) -> List[Dict]:
        """Get matches suitable for prediction analysis"""
        all_matches = self.get_enhanced_match_data()
        
        prediction_matches = []
        
        for match in all_matches:
            # Filter criteria for prediction-worthy matches
            if (match.get('odds_player1') and match.get('odds_player2') and
                match.get('tournament_level') != 'Unknown'):
                
                # Add prediction readiness score
                match['prediction_ready'] = True
                match['data_quality'] = self._assess_data_quality(match)
                
                prediction_matches.append(match)
                
        logger.info(f"Found {len(prediction_matches)} matches ready for prediction")
        return prediction_matches
        
    def _assess_data_quality(self, match: Dict) -> str:
        """Assess the quality of match data for prediction"""
        score = 0
        
        # Check data completeness
        if match.get('odds_player1') and match.get('odds_player2'):
            score += 25
        if match.get('surface_confirmed') != 'Unknown':
            score += 20
        if match.get('tournament_level') != 'Unknown':
            score += 20
        if match.get('player1_ranking') != 'Unknown':
            score += 15
        if match.get('player2_ranking') != 'Unknown':
            score += 15
        if match.get('player1_recent_form', {}).get('total', 0) > 0:
            score += 5
            
        if score >= 80:
            return 'Excellent'
        elif score >= 60:
            return 'Good'
        elif score >= 40:
            return 'Fair'
        else:
            return 'Poor'
            
    def save_enhanced_data(self, filepath: str = 'enhanced_tennis_data.json') -> bool:
        """Save enhanced match data to file"""
        try:
            enhanced_matches = self.get_enhanced_match_data()
            
            data_to_save = {
                'timestamp': datetime.now().isoformat(),
                'total_matches': len(enhanced_matches),
                'data_source': 'TennisExplorer + Universal Collector',
                'matches': enhanced_matches
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
                
            logger.info(f"✅ Saved {len(enhanced_matches)} enhanced matches to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save enhanced data: {e}")
            return False

def main():
    """Test the integration"""
    integration = TennisExplorerIntegration()
    
    print("🔗 TennisExplorer Integration Test")
    print("=" * 50)
    
    # Initialize
    if not integration.initialize():
        print("❌ Failed to initialize integration")
        return
        
    print("✅ Integration initialized successfully")
    
    # Get enhanced match data
    print("\n🔍 Getting enhanced match data...")
    matches = integration.get_enhanced_match_data(days_ahead=2)
    
    print(f"Found {len(matches)} enhanced matches")
    
    # Show sample enhanced data
    if matches:
        print("\n📊 Sample Enhanced Match:")
        sample_match = matches[0]
        print(f"Players: {sample_match['player1']} vs {sample_match['player2']}")
        print(f"Tournament: {sample_match['tournament']} ({sample_match.get('tournament_level', 'Unknown')})")
        print(f"Surface: {sample_match.get('surface_confirmed', sample_match['surface'])}")
        print(f"Data Source: {sample_match.get('data_source', 'Unknown')}")
        print(f"Data Quality: {sample_match.get('data_quality', 'Unknown')}")
        
        if sample_match.get('odds_player1'):
            print(f"Odds: {sample_match['odds_player1']} - {sample_match['odds_player2']}")
            
        if sample_match.get('player1_recent_form'):
            form1 = sample_match['player1_recent_form']
            print(f"{sample_match['player1']} Form: {form1.get('form', 'Unknown')} ({form1.get('wins', 0)}W-{form1.get('losses', 0)}L)")
            
    # Get prediction-ready matches
    print(f"\n🎯 Prediction-ready matches:")
    pred_matches = integration.get_matches_for_prediction()
    print(f"Found {len(pred_matches)} matches ready for prediction")
    
    # Save data
    if integration.save_enhanced_data():
        print("✅ Enhanced data saved to file")

if __name__ == "__main__":
    main()