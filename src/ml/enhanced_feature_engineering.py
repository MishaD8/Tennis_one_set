#!/usr/bin/env python3
"""
Enhanced Feature Engineering for Tennis Second Set Prediction
============================================================

Advanced tennis-specific feature engineering module that creates momentum,
fatigue, pressure, and contextual features for improved second set prediction.

Features:
- First set momentum indicators
- Player fatigue modeling
- Psychological pressure features
- Tournament context features
- Surface adaptation features
- Real-time match state features

Author: Tennis ML Enhancement System
Date: 2025-08-30
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MatchContext:
    """Data class for match context information"""
    tournament_tier: str  # ATP250, ATP500, Masters1000, WTA250, etc.
    surface: str         # Hard, Clay, Grass
    round: str          # R32, R16, QF, SF, F
    is_indoor: bool
    altitude: Optional[float] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None

@dataclass
class FirstSetStats:
    """Data class for first set statistics"""
    winner: str                    # 'player1' or 'player2'
    score: str                    # e.g., "6-4"
    duration_minutes: int
    total_games: int
    break_points_player1: Dict[str, int]  # {'faced': 3, 'saved': 2, 'converted': 1}
    break_points_player2: Dict[str, int]
    service_points_player1: Dict[str, int]  # {'won': 45, 'total': 60}
    service_points_player2: Dict[str, int]
    unforced_errors_player1: int
    unforced_errors_player2: int
    winners_player1: int
    winners_player2: int
    double_faults_player1: int
    double_faults_player2: int

class EnhancedTennisFeatureEngineer:
    """Enhanced feature engineering for tennis second set prediction"""
    
    def __init__(self):
        self.feature_cache = {}
        self.player_history_cache = {}
        
        # Tournament tier weights for pressure calculation
        self.tournament_weights = {
            'Grand Slam': 1.0,      # Excluded but for reference
            'Masters1000': 0.9,
            'ATP500': 0.7,
            'ATP250': 0.5,
            'WTA1000': 0.9,
            'WTA500': 0.7,
            'WTA250': 0.5,
            'Challenger': 0.3,
            'ITF': 0.1
        }
        
        # Surface adaptation periods (days to adapt)
        self.surface_adaptation_periods = {
            'Hard': 7,
            'Clay': 14,
            'Grass': 21  # Shortest season, needs more adaptation
        }
    
    def create_momentum_features(self, first_set_stats: FirstSetStats, 
                                player1_data: Dict, player2_data: Dict) -> Dict[str, float]:
        """Create momentum-based features from first set performance"""
        features = {}
        
        # Basic momentum indicators
        features['first_set_winner_momentum'] = 1.0 if first_set_stats.winner == 'player2' else 0.0
        
        # Break point momentum
        bp1 = first_set_stats.break_points_player1
        bp2 = first_set_stats.break_points_player2
        
        # Break point conversion efficiency
        bp1_efficiency = bp1.get('converted', 0) / max(bp1.get('faced', 1), 1)
        bp2_efficiency = bp2.get('converted', 0) / max(bp2.get('faced', 1), 1)
        features['underdog_bp_efficiency'] = bp2_efficiency
        features['favorite_bp_efficiency'] = bp1_efficiency
        features['bp_efficiency_gap'] = bp2_efficiency - bp1_efficiency
        
        # Break point saving ability (defensive momentum)
        bp1_save_rate = bp1.get('saved', 0) / max(bp1.get('faced', 1), 1)
        bp2_save_rate = bp2.get('saved', 0) / max(bp2.get('faced', 1), 1)
        features['underdog_bp_save_rate'] = bp2_save_rate
        features['favorite_bp_save_rate'] = bp1_save_rate
        features['bp_save_rate_gap'] = bp2_save_rate - bp1_save_rate
        
        # Service momentum
        serve1 = first_set_stats.service_points_player1
        serve2 = first_set_stats.service_points_player2
        
        serve1_percentage = serve1.get('won', 0) / max(serve1.get('total', 1), 1)
        serve2_percentage = serve2.get('won', 0) / max(serve2.get('total', 1), 1)
        features['underdog_first_set_serve_pct'] = serve2_percentage
        features['favorite_first_set_serve_pct'] = serve1_percentage
        features['serve_pct_gap'] = serve2_percentage - serve1_percentage
        
        # Quality of play momentum
        total_points_est = serve1.get('total', 30) + serve2.get('total', 30)
        
        # Error rates
        p1_error_rate = first_set_stats.unforced_errors_player1 / max(total_points_est / 2, 1)
        p2_error_rate = first_set_stats.unforced_errors_player2 / max(total_points_est / 2, 1)
        features['underdog_error_rate'] = p2_error_rate
        features['favorite_error_rate'] = p1_error_rate
        features['error_rate_gap'] = p1_error_rate - p2_error_rate  # Positive if favorite has more errors
        
        # Winner rates
        p1_winner_rate = first_set_stats.winners_player1 / max(total_points_est / 2, 1)
        p2_winner_rate = first_set_stats.winners_player2 / max(total_points_est / 2, 1)
        features['underdog_winner_rate'] = p2_winner_rate
        features['favorite_winner_rate'] = p1_winner_rate
        features['winner_rate_gap'] = p2_winner_rate - p1_winner_rate
        
        # Composite momentum score (0-1, higher = better for underdog)
        momentum_components = [
            features['first_set_winner_momentum'] * 0.3,
            features['bp_efficiency_gap'] * 0.2,
            features['serve_pct_gap'] * 0.2,
            features['error_rate_gap'] * 0.15,
            features['winner_rate_gap'] * 0.15
        ]
        features['composite_momentum_score'] = np.clip(sum(momentum_components), 0, 1)
        
        # Set closeness indicator (tight sets favor underdogs in second set)
        games_diff = abs(int(first_set_stats.score.split('-')[0]) - int(first_set_stats.score.split('-')[1]))
        features['first_set_closeness'] = 1.0 / (1.0 + games_diff)  # Closer to 1 = very close set
        
        return features
    
    def create_fatigue_features(self, player1_data: Dict, player2_data: Dict, 
                               match_date: datetime) -> Dict[str, float]:
        """Create fatigue and form features based on recent match history"""
        features = {}
        
        # Days since last match
        p1_last_match = player1_data.get('last_match_date')
        p2_last_match = player2_data.get('last_match_date')
        
        if p1_last_match:
            p1_days_rest = (match_date - p1_last_match).days
        else:
            p1_days_rest = 7  # Default assumption
            
        if p2_last_match:
            p2_days_rest = (match_date - p2_last_match).days
        else:
            p2_days_rest = 7  # Default assumption
        
        features['favorite_days_rest'] = p1_days_rest
        features['underdog_days_rest'] = p2_days_rest
        features['rest_advantage'] = p2_days_rest - p1_days_rest  # Positive if underdog more rested
        
        # Rest quality scoring (1-7 days is optimal, <1 or >10 is concerning)
        def rest_quality_score(days):
            if days < 1:
                return 0.2  # Very tired
            elif days <= 3:
                return 0.6  # Somewhat tired
            elif days <= 7:
                return 1.0  # Optimal rest
            elif days <= 10:
                return 0.8  # Well rested
            else:
                return 0.5  # Potentially rusty
        
        features['favorite_rest_quality'] = rest_quality_score(p1_days_rest)
        features['underdog_rest_quality'] = rest_quality_score(p2_days_rest)
        features['rest_quality_advantage'] = features['underdog_rest_quality'] - features['favorite_rest_quality']
        
        # Recent match load (matches in last 14 days)
        p1_recent_matches = player1_data.get('matches_last_14_days', 2)
        p2_recent_matches = player2_data.get('matches_last_14_days', 2)
        
        features['favorite_recent_load'] = p1_recent_matches
        features['underdog_recent_load'] = p2_recent_matches
        features['load_difference'] = p1_recent_matches - p2_recent_matches  # Positive if favorite played more
        
        # Travel fatigue (if data available)
        p1_travel_distance = player1_data.get('travel_distance_km', 0)
        p2_travel_distance = player2_data.get('travel_distance_km', 0)
        
        features['favorite_travel_fatigue'] = min(p1_travel_distance / 10000, 1.0)  # Normalize to 0-1
        features['underdog_travel_fatigue'] = min(p2_travel_distance / 10000, 1.0)
        features['travel_fatigue_advantage'] = features['favorite_travel_fatigue'] - features['underdog_travel_fatigue']
        
        return features
    
    def create_pressure_features(self, player1_data: Dict, player2_data: Dict, 
                                match_context: MatchContext) -> Dict[str, float]:
        """Create psychological pressure features"""
        features = {}
        
        # Tournament tier pressure
        tier_weight = self.tournament_weights.get(match_context.tournament_tier, 0.5)
        features['tournament_pressure'] = tier_weight
        
        # Ranking pressure (higher ranked players have more pressure)
        p1_rank = player1_data.get('ranking', 50)
        p2_rank = player2_data.get('ranking', 150)
        
        # Normalize rankings to pressure scores (higher rank = more pressure)
        p1_pressure = 1.0 - (p1_rank / 300.0)  # Higher for better ranked players
        p2_pressure = 1.0 - (p2_rank / 300.0)
        
        features['favorite_ranking_pressure'] = p1_pressure
        features['underdog_ranking_pressure'] = p2_pressure
        features['pressure_differential'] = p1_pressure - p2_pressure
        
        # Round pressure (later rounds = more pressure)
        round_pressure_map = {
            'R128': 0.1, 'R64': 0.2, 'R32': 0.3, 'R16': 0.5, 
            'QF': 0.7, 'SF': 0.9, 'F': 1.0
        }
        round_pressure = round_pressure_map.get(match_context.round, 0.3)
        features['round_pressure'] = round_pressure
        
        # Expected outcome pressure (favorite expected to win)
        rank_gap = p1_rank - p2_rank
        expectation_pressure = min(abs(rank_gap) / 100.0, 1.0)  # Higher gap = more pressure on favorite
        features['expectation_pressure'] = expectation_pressure
        
        # Age pressure (younger players less pressure in big moments)
        p1_age = player1_data.get('age', 25)
        p2_age = player2_data.get('age', 25)
        
        # Age pressure curve (peak pressure around 25-30)
        def age_pressure_score(age):
            if age < 20:
                return 0.3  # Young, less pressure
            elif age < 25:
                return 0.6  # Building pressure
            elif age < 30:
                return 1.0  # Peak pressure years
            elif age < 35:
                return 0.8  # Still high but experience helps
            else:
                return 0.4  # Lower pressure, veteran
        
        features['favorite_age_pressure'] = age_pressure_score(p1_age)
        features['underdog_age_pressure'] = age_pressure_score(p2_age)
        features['age_pressure_advantage'] = features['favorite_age_pressure'] - features['underdog_age_pressure']
        
        # Composite pressure score
        features['total_pressure_on_favorite'] = (
            features['favorite_ranking_pressure'] * 0.3 +
            features['tournament_pressure'] * 0.25 +
            features['round_pressure'] * 0.2 +
            features['expectation_pressure'] * 0.15 +
            features['favorite_age_pressure'] * 0.1
        )
        
        return features
    
    def create_surface_adaptation_features(self, player1_data: Dict, player2_data: Dict, 
                                          match_context: MatchContext, match_date: datetime) -> Dict[str, float]:
        """Create surface-specific adaptation features"""
        features = {}
        
        surface = match_context.surface
        adaptation_period = self.surface_adaptation_periods.get(surface, 14)
        
        # Recent matches on this surface
        p1_surface_matches = player1_data.get(f'matches_on_{surface.lower()}_last_30_days', 1)
        p2_surface_matches = player2_data.get(f'matches_on_{surface.lower()}_last_30_days', 1)
        
        features['favorite_surface_adaptation'] = min(p1_surface_matches / 5.0, 1.0)  # 5+ matches = fully adapted
        features['underdog_surface_adaptation'] = min(p2_surface_matches / 5.0, 1.0)
        features['surface_adaptation_advantage'] = features['underdog_surface_adaptation'] - features['favorite_surface_adaptation']
        
        # Career surface win percentage
        p1_surface_winpct = player1_data.get(f'{surface.lower()}_win_percentage', 0.5)
        p2_surface_winpct = player2_data.get(f'{surface.lower()}_win_percentage', 0.5)
        
        features['favorite_surface_expertise'] = p1_surface_winpct
        features['underdog_surface_expertise'] = p2_surface_winpct
        features['surface_expertise_gap'] = p2_surface_winpct - p1_surface_winpct
        
        # Surface transition penalty (coming from different surface)
        p1_last_surface = player1_data.get('last_match_surface', surface)
        p2_last_surface = player2_data.get('last_match_surface', surface)
        
        features['favorite_surface_transition'] = 0.0 if p1_last_surface == surface else 0.2
        features['underdog_surface_transition'] = 0.0 if p2_last_surface == surface else 0.2
        features['surface_transition_advantage'] = features['favorite_surface_transition'] - features['underdog_surface_transition']
        
        return features
    
    def create_contextual_features(self, player1_data: Dict, player2_data: Dict, 
                                  match_context: MatchContext, h2h_data: Dict) -> Dict[str, float]:
        """Create contextual and situational features"""
        features = {}
        
        # Head-to-head momentum
        h2h_record = h2h_data.get('overall', {'player1_wins': 0, 'player2_wins': 0})
        total_h2h = h2h_record['player1_wins'] + h2h_record['player2_wins']
        
        if total_h2h > 0:
            features['h2h_underdog_winrate'] = h2h_record['player2_wins'] / total_h2h
            features['h2h_total_matches'] = min(total_h2h / 10.0, 1.0)  # More matches = more reliable
        else:
            features['h2h_underdog_winrate'] = 0.5  # No history
            features['h2h_total_matches'] = 0.0
        
        # Recent H2H (last 3 meetings)
        recent_h2h = h2h_data.get('recent_3', {'player1_wins': 0, 'player2_wins': 0})
        recent_total = recent_h2h['player1_wins'] + recent_h2h['player2_wins']
        
        if recent_total > 0:
            features['recent_h2h_underdog_winrate'] = recent_h2h['player2_wins'] / recent_total
        else:
            features['recent_h2h_underdog_winrate'] = 0.5
        
        # Indoor/outdoor preference
        features['is_indoor'] = 1.0 if match_context.is_indoor else 0.0
        
        p1_indoor_pref = player1_data.get('indoor_win_percentage', 0.5)
        p2_indoor_pref = player2_data.get('indoor_win_percentage', 0.5)
        
        if match_context.is_indoor:
            features['indoor_advantage'] = p2_indoor_pref - p1_indoor_pref
        else:
            # Outdoor preference is inverse of indoor disadvantage
            features['indoor_advantage'] = (1 - p1_indoor_pref) - (1 - p2_indoor_pref)
        
        # Environmental factors
        if match_context.altitude:
            # Higher altitude favors players with better fitness
            altitude_factor = min(match_context.altitude / 2000.0, 1.0)  # Normalize to 0-1
            features['altitude_factor'] = altitude_factor
            
            # Fitness proxy - younger players typically better at altitude
            p1_age = player1_data.get('age', 25)
            p2_age = player2_data.get('age', 25)
            fitness_advantage = (30 - p2_age) - (30 - p1_age)  # Positive if underdog younger
            features['altitude_fitness_advantage'] = fitness_advantage * altitude_factor / 10.0
        else:
            features['altitude_factor'] = 0.0
            features['altitude_fitness_advantage'] = 0.0
        
        return features
    
    def create_all_enhanced_features(self, match_data: Dict) -> Dict[str, float]:
        """Create all enhanced features for a match"""
        try:
            # Extract data components
            player1_data = match_data.get('player1', {})
            player2_data = match_data.get('player2', {})
            first_set_stats = match_data.get('first_set_stats')
            match_context = match_data.get('match_context')
            h2h_data = match_data.get('h2h_data', {})
            match_date = match_data.get('match_date', datetime.now())
            
            if not first_set_stats or not match_context:
                logger.warning("Missing required data for enhanced features")
                return {}
            
            # Generate all feature categories
            all_features = {}
            
            # Momentum features
            momentum_features = self.create_momentum_features(first_set_stats, player1_data, player2_data)
            all_features.update({f'momentum_{k}': v for k, v in momentum_features.items()})
            
            # Fatigue features
            fatigue_features = self.create_fatigue_features(player1_data, player2_data, match_date)
            all_features.update({f'fatigue_{k}': v for k, v in fatigue_features.items()})
            
            # Pressure features
            pressure_features = self.create_pressure_features(player1_data, player2_data, match_context)
            all_features.update({f'pressure_{k}': v for k, v in pressure_features.items()})
            
            # Surface adaptation features
            surface_features = self.create_surface_adaptation_features(player1_data, player2_data, match_context, match_date)
            all_features.update({f'surface_{k}': v for k, v in surface_features.items()})
            
            # Contextual features
            contextual_features = self.create_contextual_features(player1_data, player2_data, match_context, h2h_data)
            all_features.update({f'context_{k}': v for k, v in contextual_features.items()})
            
            logger.info(f"Generated {len(all_features)} enhanced features")
            return all_features
            
        except Exception as e:
            logger.error(f"Error creating enhanced features: {e}")
            return {}
    
    def validate_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Validate and clean features"""
        cleaned_features = {}
        
        for name, value in features.items():
            # Handle missing values
            if value is None or np.isnan(value):
                cleaned_features[name] = 0.0
            # Handle infinite values
            elif np.isinf(value):
                cleaned_features[name] = 1.0 if value > 0 else 0.0
            # Clip extreme values
            else:
                cleaned_features[name] = np.clip(value, -5.0, 5.0)
        
        return cleaned_features

# Example usage and testing
if __name__ == "__main__":
    # Example data structure for testing
    sample_match_data = {
        'player1': {
            'ranking': 25,
            'age': 26,
            'last_match_date': datetime.now() - timedelta(days=3),
            'matches_last_14_days': 2,
            'hard_win_percentage': 0.65,
            'indoor_win_percentage': 0.70
        },
        'player2': {
            'ranking': 85,
            'age': 24,
            'last_match_date': datetime.now() - timedelta(days=5),
            'matches_last_14_days': 1,
            'hard_win_percentage': 0.55,
            'indoor_win_percentage': 0.50
        },
        'first_set_stats': FirstSetStats(
            winner='player1',
            score='6-4',
            duration_minutes=45,
            total_games=10,
            break_points_player1={'faced': 2, 'saved': 2, 'converted': 1},
            break_points_player2={'faced': 3, 'saved': 1, 'converted': 0},
            service_points_player1={'won': 35, 'total': 50},
            service_points_player2={'won': 25, 'total': 40},
            unforced_errors_player1=8,
            unforced_errors_player2=12,
            winners_player1=15,
            winners_player2=8,
            double_faults_player1=1,
            double_faults_player2=2
        ),
        'match_context': MatchContext(
            tournament_tier='ATP500',
            surface='Hard',
            round='R16',
            is_indoor=True,
            altitude=500.0
        ),
        'h2h_data': {
            'overall': {'player1_wins': 2, 'player2_wins': 1},
            'recent_3': {'player1_wins': 2, 'player2_wins': 1}
        },
        'match_date': datetime.now()
    }
    
    # Test the feature engineer
    engineer = EnhancedTennisFeatureEngineer()
    features = engineer.create_all_enhanced_features(sample_match_data)
    cleaned_features = engineer.validate_features(features)
    
    print(f"Generated {len(cleaned_features)} enhanced features:")
    for name, value in sorted(cleaned_features.items()):
        print(f"  {name}: {value:.4f}")