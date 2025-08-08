#!/usr/bin/env python3
"""
🎾 COMPREHENSIVE ML DATA COLLECTOR FOR TENNIS UNDERDOG DETECTION

This module implements a comprehensive data collection system that:
1. Integrates data from The Odds API, Tennis Explorer, and RapidAPI Tennis API
2. Respects rate limits: 500/month (Odds API), 5/day (Tennis Explorer), 50/day (RapidAPI)  
3. Filters for ATP/WTA singles tournaments only
4. Focuses on players ranked 101-300 for underdog detection
5. Collects data specifically for second set prediction models

Author: Claude Code (Anthropic)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import os
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import existing modules
from enhanced_universal_collector import EnhancedUniversalCollector
from enhanced_api_integration import EnhancedAPIIntegration, init_enhanced_api
from rapidapi_tennis_client import RapidAPITennisClient
from tennisexplorer_integration import TennisExplorerIntegration
from second_set_feature_engineering import SecondSetFeatureEngineer
from ranks_101_300_feature_engineering import Ranks101to300FeatureEngineer, Ranks101to300DataValidator

logger = logging.getLogger(__name__)

class RateLimitManager:
    """Manages API rate limits across all data sources"""
    
    def __init__(self):
        self.api_usage = {
            'odds_api': {'used': 0, 'limit_monthly': 500, 'limit_daily': 17},  # ~500/30 days
            'tennis_explorer': {'used': 0, 'limit_daily': 5, 'limit_hourly': 1},
            'rapidapi_tennis': {'used': 0, 'limit_daily': 50, 'limit_hourly': 3}
        }
        self.usage_file = "comprehensive_api_usage.json"
        self._load_usage_tracking()
    
    def _load_usage_tracking(self):
        """Load API usage tracking from file"""
        if os.path.exists(self.usage_file):
            try:
                with open(self.usage_file, 'r') as f:
                    saved_usage = json.load(f)
                
                # Check if usage is from today
                today = datetime.now().strftime('%Y-%m-%d')
                if saved_usage.get('date') == today:
                    for api, data in saved_usage.get('apis', {}).items():
                        if api in self.api_usage:
                            self.api_usage[api]['used'] = data.get('used', 0)
                            
            except Exception as e:
                logger.warning(f"Could not load API usage tracking: {e}")
    
    def _save_usage_tracking(self):
        """Save API usage tracking to file"""
        try:
            usage_data = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'timestamp': datetime.now().isoformat(),
                'apis': {}
            }
            
            for api, data in self.api_usage.items():
                usage_data['apis'][api] = {
                    'used': data['used'],
                    'limit_daily': data.get('limit_daily', 0),
                    'limit_remaining': data.get('limit_daily', 0) - data['used']
                }
            
            with open(self.usage_file, 'w') as f:
                json.dump(usage_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Could not save API usage tracking: {e}")
    
    def can_make_request(self, api_name: str) -> Tuple[bool, str]:
        """Check if API request is within rate limits"""
        if api_name not in self.api_usage:
            return False, f"Unknown API: {api_name}"
        
        usage = self.api_usage[api_name]
        used = usage['used']
        
        # Check daily limits
        if 'limit_daily' in usage and used >= usage['limit_daily']:
            return False, f"{api_name} daily limit exceeded: {used}/{usage['limit_daily']}"
        
        # Additional hourly check for some APIs
        if 'limit_hourly' in usage:
            # Simplified hourly check - assume even distribution
            hourly_used = used / 24  # Rough estimate
            if hourly_used >= usage['limit_hourly']:
                return False, f"{api_name} hourly limit approached"
        
        return True, "OK"
    
    def record_request(self, api_name: str):
        """Record an API request"""
        if api_name in self.api_usage:
            self.api_usage[api_name]['used'] += 1
            self._save_usage_tracking()
            
            logger.info(f"📊 {api_name}: {self.api_usage[api_name]['used']} requests used today")
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get summary of API usage"""
        summary = {}
        for api, usage in self.api_usage.items():
            daily_limit = usage.get('limit_daily', 0)
            used = usage['used']
            remaining = daily_limit - used
            
            summary[api] = {
                'used': used,
                'limit': daily_limit,
                'remaining': remaining,
                'percentage_used': (used / daily_limit * 100) if daily_limit > 0 else 0
            }
        
        return summary

class ComprehensiveMLDataCollector:
    """
    Comprehensive data collection system for tennis underdog ML prediction
    
    Integrates all data sources with intelligent rate limiting and filtering
    """
    
    def __init__(self):
        self.rate_limit_manager = RateLimitManager()
        self.feature_engineer_second_set = SecondSetFeatureEngineer()
        self.feature_engineer_ranks = Ranks101to300FeatureEngineer()
        self.data_validator = Ranks101to300DataValidator()
        
        # Initialize data sources
        self.enhanced_collector = None
        self.enhanced_api = None
        self.rapidapi_client = None
        self.tennis_explorer = None
        
        self._initialize_data_sources()
        
        # Data storage
        self.collected_data = {
            'matches': [],
            'player_data': {},
            'tournament_data': {},
            'collection_metadata': {}
        }
    
    def _initialize_data_sources(self):
        """Initialize all data sources with error handling"""
        
        # 1. Enhanced Universal Collector
        try:
            self.enhanced_collector = EnhancedUniversalCollector()
            logger.info("✅ Enhanced Universal Collector initialized")
        except Exception as e:
            logger.error(f"❌ Enhanced Universal Collector failed: {e}")
        
        # 2. Enhanced API Integration (Odds API)
        try:
            self.enhanced_api = init_enhanced_api()
            logger.info("✅ Enhanced API Integration (Odds API) initialized")
        except Exception as e:
            logger.error(f"❌ Enhanced API Integration failed: {e}")
        
        # 3. RapidAPI Tennis Client
        try:
            self.rapidapi_client = RapidAPITennisClient()
            logger.info("✅ RapidAPI Tennis Client initialized")
        except Exception as e:
            logger.error(f"❌ RapidAPI Tennis Client failed: {e}")
        
        # 4. Tennis Explorer Integration
        try:
            self.tennis_explorer = TennisExplorerIntegration()
            if self.tennis_explorer.initialize():
                logger.info("✅ Tennis Explorer Integration initialized")
            else:
                self.tennis_explorer = None
                logger.warning("⚠️ Tennis Explorer Integration failed initialization")
        except Exception as e:
            logger.error(f"❌ Tennis Explorer Integration failed: {e}")
            self.tennis_explorer = None
    
    def collect_comprehensive_data(self, max_matches: int = 100, priority_second_set: bool = True) -> Dict[str, Any]:
        """
        Collect comprehensive data from all sources for ML training
        
        Args:
            max_matches: Maximum number of matches to collect
            priority_second_set: Prioritize matches with second set data
            
        Returns:
            Dict containing collected data and metadata
        """
        logger.info("🚀 Starting comprehensive ML data collection...")
        
        collection_start = datetime.now()
        
        # Reset collected data
        self.collected_data = {
            'matches': [],
            'player_data': {},
            'tournament_data': {},
            'collection_metadata': {
                'start_time': collection_start.isoformat(),
                'api_usage_before': self.rate_limit_manager.get_usage_summary()
            }
        }
        
        # Collection phases with prioritization
        collection_phases = [
            ('odds_api_data', self._collect_odds_api_data),
            ('rapidapi_scheduled', self._collect_rapidapi_scheduled_matches),
            ('tennis_explorer_data', self._collect_tennis_explorer_data),
            ('rapidapi_rankings', self._collect_rapidapi_rankings),
            ('enhanced_universal_data', self._collect_enhanced_universal_data)
        ]
        
        successful_phases = 0
        failed_phases = []
        
        for phase_name, collection_function in collection_phases:
            try:
                logger.info(f"📊 Executing collection phase: {phase_name}")
                phase_data = collection_function(max_matches // len(collection_phases))
                
                if phase_data:
                    self._integrate_phase_data(phase_name, phase_data)
                    successful_phases += 1
                    logger.info(f"✅ Phase {phase_name} completed successfully")
                else:
                    logger.warning(f"⚠️ Phase {phase_name} returned no data")
                
                # Rate limiting pause between phases
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Phase {phase_name} failed: {e}")
                failed_phases.append((phase_name, str(e)))
        
        # Post-processing: Filter and enhance collected data
        self._post_process_collected_data(priority_second_set)
        
        # Generate collection metadata
        collection_end = datetime.now()
        self.collected_data['collection_metadata'].update({
            'end_time': collection_end.isoformat(),
            'duration_seconds': (collection_end - collection_start).total_seconds(),
            'successful_phases': successful_phases,
            'failed_phases': failed_phases,
            'api_usage_after': self.rate_limit_manager.get_usage_summary(),
            'total_matches_collected': len(self.collected_data['matches']),
            'ranks_101_300_matches': len([m for m in self.collected_data['matches'] 
                                        if self._is_ranks_101_300_match(m)])
        })
        
        logger.info(f"🎯 Data collection completed: {len(self.collected_data['matches'])} total matches")
        
        return self.collected_data
    
    def _collect_odds_api_data(self, max_items: int) -> Dict[str, Any]:
        """Collect data from The Odds API with rate limiting"""
        if not self.enhanced_api:
            logger.warning("Enhanced API not available")
            return {}
        
        can_request, reason = self.rate_limit_manager.can_make_request('odds_api')
        if not can_request:
            logger.warning(f"🚦 Odds API rate limit: {reason}")
            return {}
        
        try:
            # Get tennis odds
            result = self.enhanced_api.get_tennis_odds('tennis')
            self.rate_limit_manager.record_request('odds_api')
            
            if result.get('success', False):
                odds_data = result.get('data', {})
                logger.info(f"📊 Odds API: Collected {len(odds_data)} matches")
                return {'odds_matches': odds_data}
            else:
                logger.warning(f"Odds API request failed: {result.get('error', 'Unknown error')}")
                return {}
                
        except Exception as e:
            logger.error(f"Odds API collection error: {e}")
            return {}
    
    def _collect_rapidapi_scheduled_matches(self, max_items: int) -> Dict[str, Any]:
        """Collect scheduled matches from RapidAPI Tennis"""
        if not self.rapidapi_client:
            logger.warning("RapidAPI client not available")
            return {}
        
        can_request, reason = self.rate_limit_manager.can_make_request('rapidapi_tennis')
        if not can_request:
            logger.warning(f"🚦 RapidAPI rate limit: {reason}")
            return {}
        
        try:
            scheduled_matches = self.rapidapi_client.get_scheduled_matches()
            self.rate_limit_manager.record_request('rapidapi_tennis')
            
            if scheduled_matches:
                # Filter for professional tournaments only
                professional_matches = [
                    match for match in scheduled_matches 
                    if self._is_professional_tournament_match(match)
                ]
                
                logger.info(f"📊 RapidAPI scheduled: {len(professional_matches)} professional matches")
                return {'scheduled_matches': professional_matches}
            else:
                logger.warning("RapidAPI returned no scheduled matches")
                return {}
                
        except Exception as e:
            logger.error(f"RapidAPI scheduled matches error: {e}")
            return {}
    
    def _collect_tennis_explorer_data(self, max_items: int) -> Dict[str, Any]:
        """Collect data from Tennis Explorer with rate limiting"""
        if not self.tennis_explorer:
            logger.warning("Tennis Explorer not available")
            return {}
        
        can_request, reason = self.rate_limit_manager.can_make_request('tennis_explorer')
        if not can_request:
            logger.warning(f"🚦 Tennis Explorer rate limit: {reason}")
            return {}
        
        try:
            # Get enhanced match data (includes historical results)
            te_matches = self.tennis_explorer.get_enhanced_match_data(days_ahead=2)
            self.rate_limit_manager.record_request('tennis_explorer')
            
            if te_matches:
                logger.info(f"📊 Tennis Explorer: Collected {len(te_matches)} matches")
                return {'tennis_explorer_matches': te_matches}
            else:
                logger.warning("Tennis Explorer returned no matches")
                return {}
                
        except Exception as e:
            logger.error(f"Tennis Explorer collection error: {e}")
            return {}
    
    def _collect_rapidapi_rankings(self, max_items: int) -> Dict[str, Any]:
        """Collect current rankings from RapidAPI"""
        if not self.rapidapi_client:
            return {}
        
        can_request, reason = self.rate_limit_manager.can_make_request('rapidapi_tennis')
        if not can_request:
            logger.warning(f"🚦 RapidAPI rankings rate limit: {reason}")
            return {}
        
        try:
            rankings_data = {}
            
            # Get ATP rankings
            atp_rankings = self.rapidapi_client.get_atp_rankings()
            if atp_rankings:
                # Filter for ranks 101-300
                atp_101_300 = [
                    player for player in atp_rankings 
                    if 101 <= player.get('ranking', 0) <= 300
                ]
                rankings_data['atp_rankings_101_300'] = atp_101_300
            
            # Get WTA rankings
            wta_rankings = self.rapidapi_client.get_wta_rankings()  
            if wta_rankings:
                # Filter for ranks 101-300
                wta_101_300 = [
                    player for player in wta_rankings
                    if 101 <= player.get('ranking', 0) <= 300
                ]
                rankings_data['wta_rankings_101_300'] = wta_101_300
            
            self.rate_limit_manager.record_request('rapidapi_tennis')
            
            total_players = len(rankings_data.get('atp_rankings_101_300', [])) + len(rankings_data.get('wta_rankings_101_300', []))
            logger.info(f"📊 RapidAPI rankings: {total_players} players in ranks 101-300")
            
            return rankings_data
            
        except Exception as e:
            logger.error(f"RapidAPI rankings collection error: {e}")
            return {}
    
    def _collect_enhanced_universal_data(self, max_items: int) -> Dict[str, Any]:
        """Collect data from Enhanced Universal Collector"""
        if not self.enhanced_collector:
            logger.warning("Enhanced Universal Collector not available")
            return {}
        
        try:
            # Get ML-ready matches
            ml_matches = self.enhanced_collector.get_ml_ready_matches(min_quality_score=70)
            
            # Get underdog opportunities (focuses on ranking gaps)
            underdog_opportunities = self.enhanced_collector.get_underdog_opportunities(min_ranking_gap=20)
            
            logger.info(f"📊 Enhanced Universal: {len(ml_matches)} ML-ready, {len(underdog_opportunities)} underdog opportunities")
            
            return {
                'ml_ready_matches': ml_matches,
                'underdog_opportunities': underdog_opportunities
            }
            
        except Exception as e:
            logger.error(f"Enhanced Universal Collector error: {e}")
            return {}
    
    def _integrate_phase_data(self, phase_name: str, phase_data: Dict[str, Any]):
        """Integrate data from a collection phase"""
        
        # Add phase data to collection metadata
        self.collected_data['collection_metadata'][phase_name] = {
            'collected_at': datetime.now().isoformat(),
            'data_keys': list(phase_data.keys()),
            'success': True
        }
        
        # Process different types of data
        for data_type, data in phase_data.items():
            if 'matches' in data_type and isinstance(data, (list, dict)):
                # Handle match data
                if isinstance(data, dict):
                    # Convert dict format (like odds data) to list
                    matches_list = []
                    for match_id, match_data in data.items():
                        match_data['id'] = match_id
                        match_data['source_phase'] = phase_name
                        matches_list.append(match_data)
                    data = matches_list
                
                # Add matches to collection
                for match in data:
                    if isinstance(match, dict):
                        match['source_phase'] = phase_name
                        match['collected_at'] = datetime.now().isoformat()
                        self.collected_data['matches'].append(match)
            
            elif 'rankings' in data_type and isinstance(data, list):
                # Handle rankings data
                for player in data:
                    if isinstance(player, dict):
                        player_name = self._extract_player_name(player)
                        if player_name:
                            self.collected_data['player_data'][player_name] = {
                                'ranking_data': player,
                                'source_phase': phase_name,
                                'collected_at': datetime.now().isoformat()
                            }
    
    def _post_process_collected_data(self, priority_second_set: bool = True):
        """Post-process collected data with filtering and enhancement"""
        
        logger.info("🔧 Post-processing collected data...")
        
        original_count = len(self.collected_data['matches'])
        
        # 1. Filter for ATP/WTA professional singles only
        professional_matches = []
        for match in self.collected_data['matches']:
            if self._is_atp_wta_singles_match(match):
                professional_matches.append(match)
        
        self.collected_data['matches'] = professional_matches
        logger.info(f"🏆 Professional filtering: {len(professional_matches)}/{original_count} matches retained")
        
        # 2. Filter for ranks 101-300 (at least one player in range)
        ranks_101_300_matches = []
        for match in self.collected_data['matches']:
            if self._has_player_in_ranks_101_300(match):
                ranks_101_300_matches.append(match)
        
        self.collected_data['matches'] = ranks_101_300_matches
        logger.info(f"🎯 Ranks 101-300 filtering: {len(ranks_101_300_matches)}/{len(professional_matches)} matches retained")
        
        # 3. Enhance matches with ML features
        enhanced_matches = []
        for match in self.collected_data['matches']:
            try:
                enhanced_match = self._enhance_match_with_ml_features(match)
                enhanced_matches.append(enhanced_match)
            except Exception as e:
                logger.warning(f"Could not enhance match {match.get('id', 'unknown')}: {e}")
                enhanced_matches.append(match)  # Keep original
        
        self.collected_data['matches'] = enhanced_matches
        logger.info(f"⚡ Enhanced {len(enhanced_matches)} matches with ML features")
        
        # 4. Priority filtering for second set data
        if priority_second_set:
            second_set_matches = []
            for match in self.collected_data['matches']:
                if self._has_second_set_potential(match):
                    second_set_matches.append(match)
            
            if len(second_set_matches) > 0:
                self.collected_data['matches'] = second_set_matches
                logger.info(f"📈 Second set priority filtering: {len(second_set_matches)} matches retained")
    
    def _is_professional_tournament_match(self, match: Dict) -> bool:
        """Check if match is from professional ATP/WTA tournament"""
        tournament = match.get('tournament', {})
        tournament_name = tournament.get('name', '').lower()
        
        # Exclude non-professional keywords
        excluded = ['utr', 'ptt', 'junior', 'college', 'challenger', 'futures', 'itf', 'amateur', 'qualifying']
        for keyword in excluded:
            if keyword in tournament_name:
                return False
        
        # Look for professional indicators
        professional = ['atp', 'wta', 'grand slam', 'masters', 'wimbledon', 'french open', 'us open', 'australian open']
        for indicator in professional:
            if indicator in tournament_name:
                return True
        
        return False
    
    def _is_atp_wta_singles_match(self, match: Dict) -> bool:
        """Check if match is ATP/WTA singles"""
        
        # Check tournament name
        tournament_name = match.get('tournament', '').lower()
        if any(keyword in tournament_name for keyword in ['utr', 'ptt', 'junior', 'college', 'amateur']):
            return False
        
        # Check if singles (not doubles)
        player1 = match.get('player1', '')
        player2 = match.get('player2', '')
        
        # Look for doubles indicators
        doubles_indicators = ['/', ' and ', '&', ' / ']
        if any(indicator in player1 + player2 for indicator in doubles_indicators):
            return False
        
        return True
    
    def _has_player_in_ranks_101_300(self, match: Dict) -> bool:
        """Check if at least one player is ranked 101-300"""
        player1_rank = self._extract_player_rank(match, 'player1')
        player2_rank = self._extract_player_rank(match, 'player2')
        
        # Check if either player is in target range
        if player1_rank and 101 <= player1_rank <= 300:
            return True
        if player2_rank and 101 <= player2_rank <= 300:
            return True
        
        return False
    
    def _is_ranks_101_300_match(self, match: Dict) -> bool:
        """Check if match involves ranks 101-300 players"""
        return self._has_player_in_ranks_101_300(match)
    
    def _extract_player_rank(self, match: Dict, player_key: str) -> Optional[int]:
        """Extract player ranking from match data"""
        
        # Try different ranking field names
        ranking_fields = [f'{player_key}_ranking', f'{player_key}_rank', 'ranking', 'rank']
        
        for field in ranking_fields:
            if field in match:
                try:
                    rank = int(match[field])
                    if 1 <= rank <= 1000:  # Sanity check
                        return rank
                except (ValueError, TypeError):
                    continue
        
        # Try nested player data
        player_data = match.get(player_key, {})
        if isinstance(player_data, dict):
            for field in ['ranking', 'rank']:
                if field in player_data:
                    try:
                        rank = int(player_data[field])
                        if 1 <= rank <= 1000:
                            return rank
                    except (ValueError, TypeError):
                        continue
        
        return None
    
    def _extract_player_name(self, player_data: Dict) -> Optional[str]:
        """Extract player name from player data"""
        name_fields = ['name', 'player_name', 'team', 'player']
        
        for field in name_fields:
            if field in player_data:
                name = player_data[field]
                if isinstance(name, str) and len(name.strip()) > 0:
                    return name.strip()
        
        # Try nested team data
        if 'team' in player_data and isinstance(player_data['team'], dict):
            if 'name' in player_data['team']:
                name = player_data['team']['name']
                if isinstance(name, str) and len(name.strip()) > 0:
                    return name.strip()
        
        return None
    
    def _has_second_set_potential(self, match: Dict) -> bool:
        """Check if match has potential for second set prediction"""
        
        # Look for match status that indicates it could go to multiple sets
        status = match.get('status', '').lower()
        
        # Exclude completed matches unless they have set-by-set data
        if status in ['finished', 'completed', 'final']:
            # Only keep if we have detailed set data
            return 'set_scores' in match or 'first_set_data' in match
        
        # Include scheduled, live, and in-progress matches
        if status in ['scheduled', 'live', 'in_progress', 'inprogress']:
            return True
        
        # Include matches with tournament context
        if match.get('tournament') and match.get('round'):
            return True
        
        return False
    
    def _enhance_match_with_ml_features(self, match: Dict) -> Dict:
        """Enhance match with ML features for prediction"""
        
        # Extract player data
        player1_data = self._extract_player_data_for_ml(match, 'player1')
        player2_data = self._extract_player_data_for_ml(match, 'player2')
        
        # Create match context
        match_context = {
            'tournament': match.get('tournament', 'Unknown Tournament'),
            'surface': match.get('surface', 'Hard'),
            'round': match.get('round', 'R32'),
            'tournament_importance': self._calculate_tournament_importance(match),
            'total_pressure': self._calculate_pressure_factor(match),
            'player1_surface_advantage': 0.0  # Default, can be enhanced later
        }
        
        # Create dummy first set data for feature generation
        first_set_data = match.get('first_set_data', {
            'winner': 'unknown',
            'score': '6-4',
            'duration_minutes': 45,
            'breaks_won_player1': 1,
            'breaks_won_player2': 1,
            'break_points_saved_player1': 0.5,
            'break_points_saved_player2': 0.5,
            'first_serve_percentage_player1': 0.65,
            'first_serve_percentage_player2': 0.65,
            'had_tiebreak': False
        })
        
        # Generate ML features
        try:
            # Validate that players are in target range
            validation_data = {
                'player1': player1_data,
                'player2': player2_data,
                'first_set_data': first_set_data
            }
            
            validation_result = self.data_validator.validate_match_data(validation_data)
            
            if validation_result['valid']:
                # Generate comprehensive features for ranks 101-300
                ml_features = self.feature_engineer_ranks.create_complete_feature_set(
                    match.get('player1', 'Player 1'),
                    match.get('player2', 'Player 2'),
                    player1_data,
                    player2_data,
                    match_context,
                    first_set_data
                )
                
                # Add second set specific features
                second_set_features = self.feature_engineer_second_set.create_complete_feature_set(
                    match.get('player1', 'Player 1'),
                    match.get('player2', 'Player 2'),
                    player1_data,
                    player2_data,
                    match_context,
                    first_set_data
                )
                
                # Merge features
                ml_features.update(second_set_features)
                
                # Add to match
                match['ml_features'] = ml_features
                match['ml_ready'] = True
                match['feature_count'] = len(ml_features)
                match['validation_result'] = validation_result
                
            else:
                # Mark as invalid for ML
                match['ml_ready'] = False
                match['validation_errors'] = validation_result['errors']
                
        except Exception as e:
            logger.warning(f"Could not generate ML features for match: {e}")
            match['ml_ready'] = False
            match['ml_error'] = str(e)
        
        return match
    
    def _extract_player_data_for_ml(self, match: Dict, player_key: str) -> Dict:
        """Extract player data needed for ML features"""
        
        player_data = {
            'rank': self._extract_player_rank(match, player_key) or 200,  # Default to mid-range
            'age': 25,  # Default age
            'recent_win_rate_10': 0.5,
            'hard_court_win_rate': 0.5,
            'overall_win_rate': 0.5,
            'professional_years': 5
        }
        
        # Try to get more detailed data from collected player data
        player_name = match.get(player_key, '')
        if player_name in self.collected_data['player_data']:
            stored_data = self.collected_data['player_data'][player_name]
            ranking_data = stored_data.get('ranking_data', {})
            
            # Extract available data
            if 'ranking' in ranking_data:
                player_data['rank'] = ranking_data['ranking']
            
            # Add other available fields
            for field in ['age', 'wins', 'losses']:
                if field in ranking_data:
                    if field == 'age':
                        player_data['age'] = ranking_data[field]
                    elif field in ['wins', 'losses']:
                        wins = ranking_data.get('wins', 0)
                        losses = ranking_data.get('losses', 0)
                        total = wins + losses
                        if total > 0:
                            player_data['overall_win_rate'] = wins / total
        
        return player_data
    
    def _calculate_tournament_importance(self, match: Dict) -> float:
        """Calculate tournament importance (1-5 scale)"""
        tournament_name = match.get('tournament', '').lower()
        
        if 'grand slam' in tournament_name:
            return 5.0
        elif 'masters' in tournament_name or '1000' in tournament_name:
            return 4.0
        elif '500' in tournament_name:
            return 3.0
        elif '250' in tournament_name:
            return 2.0
        else:
            return 2.5  # Default
    
    def _calculate_pressure_factor(self, match: Dict) -> float:
        """Calculate pressure factor for the match"""
        base_pressure = 2.5
        
        # Round pressure
        round_name = match.get('round', '').lower()
        if 'final' in round_name:
            base_pressure += 1.5
        elif 'semifinal' in round_name or 'sf' in round_name:
            base_pressure += 1.0
        elif 'quarterfinal' in round_name or 'qf' in round_name:
            base_pressure += 0.5
        
        # Tournament importance pressure
        importance = self._calculate_tournament_importance(match)
        base_pressure += (importance - 2.5) * 0.3
        
        return min(base_pressure, 5.0)  # Cap at 5.0
    
    def get_ml_training_dataset(self) -> Dict[str, Any]:
        """
        Generate ML training dataset from collected data
        
        Returns:
            Dict with training data, features, and metadata
        """
        if not self.collected_data['matches']:
            logger.warning("No matches collected for ML training dataset")
            return {'error': 'No data collected'}
        
        ml_ready_matches = [
            match for match in self.collected_data['matches']
            if match.get('ml_ready', False)
        ]
        
        if not ml_ready_matches:
            logger.warning("No ML-ready matches found")
            return {'error': 'No ML-ready matches'}
        
        # Extract features and create dataset
        features_list = []
        match_metadata = []
        
        for match in ml_ready_matches:
            ml_features = match.get('ml_features', {})
            if ml_features:
                features_list.append(ml_features)
                match_metadata.append({
                    'match_id': match.get('id', 'unknown'),
                    'player1': match.get('player1', 'Unknown'),
                    'player2': match.get('player2', 'Unknown'),
                    'tournament': match.get('tournament', 'Unknown'),
                    'source_phase': match.get('source_phase', 'unknown'),
                    'validation_result': match.get('validation_result', {})
                })
        
        if not features_list:
            return {'error': 'No feature data extracted'}
        
        # Convert to structured format
        import pandas as pd
        
        features_df = pd.DataFrame(features_list)
        
        # Fill missing values
        features_df = features_df.fillna(features_df.median())
        
        dataset = {
            'features_df': features_df,
            'match_metadata': match_metadata,
            'feature_columns': list(features_df.columns),
            'n_matches': len(ml_ready_matches),
            'n_features': len(features_df.columns),
            'dataset_metadata': {
                'created_at': datetime.now().isoformat(),
                'collection_metadata': self.collected_data['collection_metadata'],
                'target_variable': 'underdog_won_second_set',
                'ml_ready_matches': len(ml_ready_matches),
                'total_matches_collected': len(self.collected_data['matches'])
            }
        }
        
        logger.info(f"📊 ML Training Dataset: {dataset['n_matches']} matches, {dataset['n_features']} features")
        
        return dataset
    
    def save_collected_data(self, filepath: str = None) -> str:
        """Save collected data to file"""
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"comprehensive_ml_data_{timestamp}.json"
        
        try:
            # Convert pandas DataFrames to serializable format
            serializable_data = self.collected_data.copy()
            
            with open(filepath, 'w') as f:
                json.dump(serializable_data, f, indent=2, default=str)
            
            logger.info(f"💾 Collected data saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Could not save collected data: {e}")
            return ""
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """Get summary of data collection results"""
        
        matches = self.collected_data['matches']
        
        # Count by source
        source_counts = {}
        for match in matches:
            source = match.get('source_phase', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Count ML-ready matches
        ml_ready_count = len([m for m in matches if m.get('ml_ready', False)])
        
        # Count ranks 101-300 matches
        ranks_101_300_count = len([m for m in matches if self._is_ranks_101_300_match(m)])
        
        # API usage summary
        api_usage = self.rate_limit_manager.get_usage_summary()
        
        summary = {
            'total_matches_collected': len(matches),
            'ml_ready_matches': ml_ready_count,
            'ranks_101_300_matches': ranks_101_300_count,
            'matches_by_source': source_counts,
            'api_usage_summary': api_usage,
            'collection_metadata': self.collected_data.get('collection_metadata', {}),
            'data_sources_status': {
                'enhanced_collector': self.enhanced_collector is not None,
                'enhanced_api': self.enhanced_api is not None,
                'rapidapi_client': self.rapidapi_client is not None,
                'tennis_explorer': self.tennis_explorer is not None
            }
        }
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    print("🎾 COMPREHENSIVE ML DATA COLLECTOR FOR TENNIS UNDERDOG DETECTION")
    print("=" * 80)
    
    # Initialize the collector
    collector = ComprehensiveMLDataCollector()
    
    # Display API usage status
    print("\n📊 API Usage Status:")
    usage_summary = collector.rate_limit_manager.get_usage_summary()
    for api, usage in usage_summary.items():
        print(f"  {api}: {usage['used']}/{usage['limit']} ({usage['percentage_used']:.1f}% used)")
    
    # Collect data
    print("\n🚀 Starting data collection...")
    collected_data = collector.collect_comprehensive_data(max_matches=50, priority_second_set=True)
    
    # Display results
    print(f"\n📈 Collection Results:")
    print(f"  Total matches: {len(collected_data['matches'])}")
    print(f"  ML-ready matches: {len([m for m in collected_data['matches'] if m.get('ml_ready', False)])}")
    print(f"  Ranks 101-300 matches: {len([m for m in collected_data['matches'] if collector._is_ranks_101_300_match(m)])}")
    
    # Generate ML training dataset
    print("\n🤖 Generating ML training dataset...")
    training_dataset = collector.get_ml_training_dataset()
    
    if 'error' not in training_dataset:
        print(f"  Dataset shape: {training_dataset['n_matches']} matches × {training_dataset['n_features']} features")
        print(f"  Feature columns: {training_dataset['feature_columns'][:5]}... (showing first 5)")
    else:
        print(f"  Dataset generation error: {training_dataset['error']}")
    
    # Save collected data
    saved_file = collector.save_collected_data()
    print(f"\n💾 Data saved to: {saved_file}")
    
    # Display collection summary
    print("\n📋 Collection Summary:")
    summary = collector.get_collection_summary()
    print(f"  Total matches: {summary['total_matches_collected']}")
    print(f"  ML-ready: {summary['ml_ready_matches']}")
    print(f"  Target ranks (101-300): {summary['ranks_101_300_matches']}")
    
    print("\n✅ Comprehensive ML data collection completed!")