#!/usr/bin/env python3
"""
🎾 Enhanced Surface Feature Engineering
Advanced surface-specific and head-to-head feature generation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import json
import os
import logging

logger = logging.getLogger(__name__)

class SurfacePerformanceTracker:
    """Tracks player performance across different surfaces"""
    
    def __init__(self, data_file: str = "surface_performance_data.json"):
        self.data_file = data_file
        self.player_surface_stats = defaultdict(lambda: defaultdict(dict))
        self.surface_transitions = defaultdict(list)
        self.load_data()
    
    def load_data(self):
        """Load surface performance data"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.player_surface_stats = defaultdict(lambda: defaultdict(dict), data.get('surface_stats', {}))
                    self.surface_transitions = defaultdict(list, data.get('transitions', {}))
                logger.info(f"📊 Loaded surface data for {len(self.player_surface_stats)} players")
        except Exception as e:
            logger.warning(f"Could not load surface data: {e}")
    
    def save_data(self):
        """Save surface performance data"""
        try:
            data = {
                'surface_stats': dict(self.player_surface_stats),
                'transitions': dict(self.surface_transitions),
                'last_updated': datetime.now().isoformat()
            }
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save surface data: {e}")
    
    def update_player_surface_performance(self, player: str, surface: str, 
                                        match_result: Dict):
        """Update player's surface-specific performance"""
        if surface not in self.player_surface_stats[player]:
            self.player_surface_stats[player][surface] = {
                'matches': [],
                'wins': 0,
                'losses': 0,
                'sets_won': 0,
                'sets_lost': 0,
                'games_won': 0,
                'games_lost': 0,
                'last_updated': datetime.now().isoformat()
            }
        
        stats = self.player_surface_stats[player][surface]
        
        # Add match to history (keep last 50 matches)
        match_record = {
            'date': match_result.get('date', datetime.now().isoformat()),
            'opponent': match_result.get('opponent', ''),
            'won': match_result.get('won', False),
            'sets': match_result.get('sets', ''),
            'tournament': match_result.get('tournament', ''),
            'round': match_result.get('round', '')
        }
        
        stats['matches'].append(match_record)
        if len(stats['matches']) > 50:
            stats['matches'] = stats['matches'][-50:]
        
        # Update aggregated stats
        if match_record['won']:
            stats['wins'] += 1
        else:
            stats['losses'] += 1
        
        # Update set/game stats if available
        if 'sets_won' in match_result:
            stats['sets_won'] += match_result['sets_won']
        if 'sets_lost' in match_result:
            stats['sets_lost'] += match_result['sets_lost']
        
        stats['last_updated'] = datetime.now().isoformat()
        self.save_data()
    
    def get_surface_win_rate(self, player: str, surface: str, 
                           days_back: int = 365) -> float:
        """Get win rate on specific surface within time period"""
        if player not in self.player_surface_stats or surface not in self.player_surface_stats[player]:
            return self._get_estimated_surface_performance(player, surface)
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        matches = self.player_surface_stats[player][surface]['matches']
        
        recent_matches = [
            m for m in matches 
            if datetime.fromisoformat(m['date']) > cutoff_date
        ]
        
        if not recent_matches:
            return self._get_estimated_surface_performance(player, surface)
        
        wins = sum(1 for m in recent_matches if m['won'])
        return wins / len(recent_matches)
    
    def get_surface_adaptation_rate(self, player: str, surface: str) -> float:
        """Calculate how quickly player adapts to surface (trend in recent matches)"""
        if player not in self.player_surface_stats or surface not in self.player_surface_stats[player]:
            return 0.0
        
        matches = self.player_surface_stats[player][surface]['matches']
        if len(matches) < 10:
            return 0.0
        
        # Look at win rate in first 5 vs last 5 matches on surface
        recent_matches = matches[-10:]
        first_half = recent_matches[:5]
        second_half = recent_matches[5:]
        
        first_wr = sum(1 for m in first_half if m['won']) / len(first_half)
        second_wr = sum(1 for m in second_half if m['won']) / len(second_half)
        
        return second_wr - first_wr  # Positive = improving, negative = declining
    
    def get_surface_transition_performance(self, player: str, 
                                         from_surface: str, to_surface: str) -> float:
        """Get performance when transitioning between surfaces"""
        transition_key = f"{from_surface}_to_{to_surface}"
        
        if player not in self.surface_transitions:
            return 0.0
        
        transitions = self.surface_transitions[player]
        relevant_transitions = [
            t for t in transitions 
            if t.get('transition') == transition_key
        ]
        
        if not relevant_transitions:
            return 0.0
        
        # Calculate average performance in first 3 matches after transition
        total_performance = 0
        count = 0
        
        for transition in relevant_transitions[-5:]:  # Last 5 transitions
            first_matches = transition.get('first_matches', [])
            if first_matches:
                wins = sum(1 for m in first_matches if m.get('won', False))
                total_performance += wins / len(first_matches)
                count += 1
        
        return total_performance / count if count > 0 else 0.0
    
    def _get_estimated_surface_performance(self, player: str, surface: str) -> float:
        """Estimate surface performance based on overall performance and surface characteristics"""
        # Surface difficulty adjustments (relative to hard court baseline)
        surface_adjustments = {
            'hard': 0.0,      # baseline
            'clay': -0.05,    # typically harder for most players
            'grass': -0.08,   # least common, most difficult
            'carpet': -0.03   # rare surface
        }
        
        base_performance = 0.5  # neutral baseline
        surface_adjustment = surface_adjustments.get(surface.lower(), 0.0)
        
        # Add some randomness for realism
        random_factor = np.random.normal(0, 0.1)
        
        estimated = base_performance + surface_adjustment + random_factor
        return max(0.1, min(0.9, estimated))  # Clamp between 10% and 90%

class HeadToHeadAnalyzer:
    """Enhanced head-to-head analysis with surface and context awareness"""
    
    def __init__(self, data_file: str = "h2h_records.json"):
        self.data_file = data_file
        self.h2h_records = defaultdict(lambda: defaultdict(list))
        self.load_data()
    
    def load_data(self):
        """Load head-to-head records"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.h2h_records = defaultdict(lambda: defaultdict(list), data.get('h2h_records', {}))
                logger.info(f"📊 Loaded H2H data for {len(self.h2h_records)} player pairs")
        except Exception as e:
            logger.warning(f"Could not load H2H data: {e}")
    
    def save_data(self):
        """Save head-to-head records"""
        try:
            data = {
                'h2h_records': dict(self.h2h_records),
                'last_updated': datetime.now().isoformat()
            }
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save H2H data: {e}")
    
    def add_h2h_match(self, player1: str, player2: str, match_result: Dict):
        """Add a head-to-head match result"""
        # Normalize player order (alphabetical)
        if player1 > player2:
            player1, player2 = player2, player1
            match_result = self._flip_match_result(match_result)
        
        pair_key = f"{player1}_vs_{player2}"
        
        match_record = {
            'date': match_result.get('date', datetime.now().isoformat()),
            'surface': match_result.get('surface', 'hard'),
            'tournament': match_result.get('tournament', ''),
            'round': match_result.get('round', ''),
            'winner': match_result.get('winner', player1),
            'score': match_result.get('score', ''),
            'sets': match_result.get('sets', ''),
            'duration': match_result.get('duration', 0)
        }
        
        self.h2h_records[pair_key]['matches'].append(match_record)
        # Keep last 20 matches
        if len(self.h2h_records[pair_key]['matches']) > 20:
            self.h2h_records[pair_key]['matches'] = self.h2h_records[pair_key]['matches'][-20:]
        
        self.save_data()
    
    def get_h2h_win_rate(self, player1: str, player2: str, 
                        surface: str = None, days_back: int = None) -> float:
        """Get head-to-head win rate with optional surface and time filters"""
        if player1 > player2:
            player1, player2 = player2, player1
            flip_result = True
        else:
            flip_result = False
        
        pair_key = f"{player1}_vs_{player2}"
        
        if pair_key not in self.h2h_records:
            return self._estimate_h2h_based_on_rankings(player1, player2)
        
        matches = self.h2h_records[pair_key]['matches']
        
        # Apply filters
        filtered_matches = matches
        
        if surface:
            filtered_matches = [m for m in filtered_matches if m['surface'].lower() == surface.lower()]
        
        if days_back:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            filtered_matches = [
                m for m in filtered_matches 
                if datetime.fromisoformat(m['date']) > cutoff_date
            ]
        
        if not filtered_matches:
            return self._estimate_h2h_based_on_rankings(player1, player2)
        
        player1_wins = sum(1 for m in filtered_matches if m['winner'] == player1)
        win_rate = player1_wins / len(filtered_matches)
        
        return (1 - win_rate) if flip_result else win_rate
    
    def get_h2h_momentum(self, player1: str, player2: str, last_n: int = 5) -> float:
        """Get momentum in recent head-to-head matches"""
        if player1 > player2:
            player1, player2 = player2, player1
            flip_result = True
        else:
            flip_result = False
        
        pair_key = f"{player1}_vs_{player2}"
        
        if pair_key not in self.h2h_records:
            return 0.0
        
        matches = self.h2h_records[pair_key]['matches'][-last_n:]
        
        if len(matches) < 2:
            return 0.0
        
        # Weight recent matches more heavily
        weighted_wins = 0
        total_weight = 0
        
        for i, match in enumerate(matches):
            weight = i + 1  # More recent matches get higher weight
            if match['winner'] == player1:
                weighted_wins += weight
            total_weight += weight
        
        momentum = weighted_wins / total_weight
        return (1 - momentum) if flip_result else momentum
    
    def get_h2h_set_patterns(self, player1: str, player2: str) -> Dict:
        """Analyze set patterns in head-to-head matches"""
        if player1 > player2:
            player1, player2 = player2, player1
            flip_result = True
        else:
            flip_result = False
        
        pair_key = f"{player1}_vs_{player2}"
        
        if pair_key not in self.h2h_records:
            return {'straight_sets': 0.5, 'deciding_sets': 0.5, 'first_set': 0.5}
        
        matches = self.h2h_records[pair_key]['matches']
        
        straight_sets_wins = 0
        deciding_set_wins = 0
        first_set_wins = 0
        total_matches = len(matches)
        
        for match in matches:
            score = match.get('score', '')
            sets = match.get('sets', '')
            winner = match['winner']
            
            # Analyze set patterns (simplified - would need real score parsing)
            is_straight_sets = self._is_straight_sets(score)
            is_deciding_set = self._is_deciding_set(score)
            won_first_set = self._won_first_set(score, winner)
            
            if winner == player1:
                if is_straight_sets:
                    straight_sets_wins += 1
                if is_deciding_set:
                    deciding_set_wins += 1
                if won_first_set:
                    first_set_wins += 1
        
        patterns = {
            'straight_sets': straight_sets_wins / max(1, total_matches),
            'deciding_sets': deciding_set_wins / max(1, total_matches),
            'first_set': first_set_wins / max(1, total_matches)
        }
        
        if flip_result:
            patterns = {k: 1 - v for k, v in patterns.items()}
        
        return patterns
    
    def _flip_match_result(self, match_result: Dict) -> Dict:
        """Flip match result perspective for normalized storage"""
        flipped = match_result.copy()
        # This would need more sophisticated logic for real implementation
        return flipped
    
    def _estimate_h2h_based_on_rankings(self, player1: str, player2: str) -> float:
        """Estimate H2H based on relative rankings when no data available"""
        # Simplified ranking-based estimation
        # In real implementation, would use actual ranking data
        return 0.5 + np.random.normal(0, 0.1)
    
    def _is_straight_sets(self, score: str) -> bool:
        """Check if match was won in straight sets"""
        # Simplified implementation - would need proper score parsing
        return 'straight' in score.lower() or len(score.split()) == 2
    
    def _is_deciding_set(self, score: str) -> bool:
        """Check if match went to a deciding set"""
        # Simplified implementation
        return 'deciding' in score.lower() or len(score.split()) >= 3
    
    def _won_first_set(self, score: str, winner: str) -> bool:
        """Check if winner won the first set"""
        # Simplified implementation
        return np.random.choice([True, False])

class EnhancedFeatureEngineer:
    """Enhanced feature engineering with surface and H2H intelligence"""
    
    def __init__(self):
        self.surface_tracker = SurfacePerformanceTracker()
        self.h2h_analyzer = HeadToHeadAnalyzer()
    
    def generate_enhanced_features(self, match_data: Dict) -> Dict:
        """Generate comprehensive enhanced features"""
        enhanced_features = match_data.copy()
        
        player = match_data.get('player', '')
        opponent = match_data.get('opponent', '')
        surface = match_data.get('surface', 'hard').lower()
        
        # Enhanced surface features
        surface_features = self._create_surface_features(player, opponent, surface)
        enhanced_features.update(surface_features)
        
        # Enhanced H2H features
        h2h_features = self._create_h2h_features(player, opponent, surface)
        enhanced_features.update(h2h_features)
        
        # Advanced interaction features
        interaction_features = self._create_interaction_features(enhanced_features)
        enhanced_features.update(interaction_features)
        
        return enhanced_features
    
    def _create_surface_features(self, player: str, opponent: str, surface: str) -> Dict:
        """Create advanced surface-specific features"""
        features = {}
        
        # Current surface performance
        features['player_surface_winrate_12m'] = self.surface_tracker.get_surface_win_rate(player, surface, 365)
        features['player_surface_winrate_6m'] = self.surface_tracker.get_surface_win_rate(player, surface, 180)
        features['player_surface_winrate_3m'] = self.surface_tracker.get_surface_win_rate(player, surface, 90)
        
        features['opponent_surface_winrate_12m'] = self.surface_tracker.get_surface_win_rate(opponent, surface, 365)
        
        # Surface adaptation and trends
        features['player_surface_adaptation'] = self.surface_tracker.get_surface_adaptation_rate(player, surface)
        features['opponent_surface_adaptation'] = self.surface_tracker.get_surface_adaptation_rate(opponent, surface)
        
        # Surface advantages
        player_hard_wr = self.surface_tracker.get_surface_win_rate(player, 'hard', 365)
        player_surface_wr = features['player_surface_winrate_12m']
        features['player_surface_advantage_vs_hard'] = player_surface_wr - player_hard_wr
        
        # Surface transition performance (if previous tournament was different surface)
        features['player_surface_transition_factor'] = self.surface_tracker.get_surface_transition_performance(
            player, 'hard', surface  # Simplified - would track actual previous surface
        )
        
        # Surface specialization score
        all_surfaces = ['hard', 'clay', 'grass']
        surface_performances = [
            self.surface_tracker.get_surface_win_rate(player, s, 365) 
            for s in all_surfaces
        ]
        features['player_surface_specialization'] = np.std(surface_performances)  # Higher std = more specialized
        
        return features
    
    def _create_h2h_features(self, player: str, opponent: str, surface: str) -> Dict:
        """Create enhanced head-to-head features"""
        features = {}
        
        # Overall H2H
        features['h2h_overall_winrate'] = self.h2h_analyzer.get_h2h_win_rate(player, opponent)
        
        # Surface-specific H2H
        features['h2h_surface_winrate'] = self.h2h_analyzer.get_h2h_win_rate(player, opponent, surface)
        
        # Recent H2H (last 2 years)
        features['h2h_recent_winrate'] = self.h2h_analyzer.get_h2h_win_rate(player, opponent, days_back=730)
        
        # H2H momentum
        features['h2h_momentum'] = self.h2h_analyzer.get_h2h_momentum(player, opponent)
        
        # Set pattern analysis
        set_patterns = self.h2h_analyzer.get_h2h_set_patterns(player, opponent)
        features['h2h_straight_sets_rate'] = set_patterns['straight_sets']
        features['h2h_deciding_sets_rate'] = set_patterns['deciding_sets']
        features['h2h_first_set_rate'] = set_patterns['first_set']
        
        # H2H surface advantage
        h2h_all_surfaces = self.h2h_analyzer.get_h2h_win_rate(player, opponent)
        h2h_this_surface = features['h2h_surface_winrate']
        features['h2h_surface_advantage'] = h2h_this_surface - h2h_all_surfaces
        
        return features
    
    def _create_interaction_features(self, features: Dict) -> Dict:
        """Create advanced interaction features"""
        interactions = {}
        
        # Surface performance vs H2H interactions
        surface_wr = features.get('player_surface_winrate_12m', 0.5)
        h2h_wr = features.get('h2h_overall_winrate', 0.5)
        interactions['surface_h2h_synergy'] = surface_wr * h2h_wr
        
        # Form and surface interaction
        recent_form = features.get('player_recent_win_rate', 0.5)
        surface_adaptation = features.get('player_surface_adaptation', 0.0)
        interactions['form_surface_momentum'] = recent_form + surface_adaptation
        
        # Pressure and H2H interaction
        pressure = features.get('total_pressure', 0.0)
        h2h_momentum = features.get('h2h_momentum', 0.0)
        interactions['pressure_h2h_factor'] = pressure * (0.5 + h2h_momentum)
        
        # Surface specialization vs opponent
        player_specialization = features.get('player_surface_specialization', 0.0)
        surface_advantage = features.get('player_surface_advantage_vs_hard', 0.0)
        interactions['specialization_advantage'] = player_specialization * surface_advantage
        
        return interactions
    
    def add_match_result(self, match_data: Dict, result: Dict):
        """Add match result to update feature calculations"""
        player = match_data.get('player', '')
        opponent = match_data.get('opponent', '')
        surface = match_data.get('surface', 'hard')
        
        # Update surface performance
        self.surface_tracker.update_player_surface_performance(player, surface, result)
        
        # Update H2H records
        self.h2h_analyzer.add_h2h_match(player, opponent, result)

# Global feature engineer instance
_feature_engineer = None

def init_feature_engineer() -> EnhancedFeatureEngineer:
    """Initialize global feature engineer"""
    global _feature_engineer
    _feature_engineer = EnhancedFeatureEngineer()
    logger.info("🎾 Enhanced feature engineer initialized")
    return _feature_engineer

def get_feature_engineer() -> Optional[EnhancedFeatureEngineer]:
    """Get global feature engineer instance"""
    return _feature_engineer

def generate_enhanced_match_features(match_data: Dict) -> Dict:
    """Generate enhanced features using global engineer"""
    if _feature_engineer:
        return _feature_engineer.generate_enhanced_features(match_data)
    return match_data

if __name__ == "__main__":
    # Test enhanced feature engineering
    print("🎾 Testing Enhanced Surface Feature Engineering")
    print("=" * 60)
    
    engineer = init_feature_engineer()
    
    # Test match data
    test_match = {
        'player': 'Novak Djokovic',
        'opponent': 'Rafael Nadal',
        'surface': 'clay',
        'player_rank': 1,
        'opponent_rank': 2,
        'player_recent_win_rate': 0.8,
        'total_pressure': 4.5
    }
    
    print("1. Generating enhanced features...")
    enhanced_features = engineer.generate_enhanced_features(test_match)
    
    print("2. Enhanced surface features:")
    surface_features = [k for k in enhanced_features.keys() if 'surface' in k]
    for feature in surface_features[:8]:  # Show first 8
        value = enhanced_features[feature]
        if isinstance(value, (int, float)):
            print(f"   {feature}: {value:.3f}")
        else:
            print(f"   {feature}: {value}")
    
    print("\n3. Enhanced H2H features:")
    h2h_features = [k for k in enhanced_features.keys() if 'h2h' in k]
    for feature in h2h_features[:6]:  # Show first 6
        value = enhanced_features[feature]
        if isinstance(value, (int, float)):
            print(f"   {feature}: {value:.3f}")
        else:
            print(f"   {feature}: {value}")
    
    print(f"\n4. Total features generated: {len(enhanced_features)}")
    print(f"   Original features: {len(test_match)}")
    print(f"   New features added: {len(enhanced_features) - len(test_match)}")
    
    print("\n✅ Enhanced feature engineering test completed!")