#!/usr/bin/env python3
"""
🎯 RANK-SPECIFIC FEATURE ENGINEERING FOR RANKS 50-300
Specialized feature engineering system for second set underdog predictions
focusing exclusively on players ranked 50-300.

Strategic Focus:
- Career trajectory analysis within 50-300 tier
- Tournament level adaptation features
- Rank gap psychology and pressure factors
- Second set comeback patterns specific to this tier

Author: Claude Code (Anthropic)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

class Ranks50to300FeatureEngineer:
    """
    Specialized feature engineering for ranks 50-300 second set predictions
    
    This class creates features specifically designed for the psychological,
    competitive, and strategic dynamics of players ranked 50-300.
    """
    
    def __init__(self):
        self.rank_range = (50, 300)
        self.tier_size = 250  # 300 - 50
        
        # Key milestone ranks
        self.milestones = {
            'top_50_threshold': 50,
            'top_100_threshold': 100,
            'relegation_risk': 280,
            'mid_tier': 175,
            'upper_tier': 100,
            'promotion_zone': 125
        }
        
        # Career stage definitions
        self.career_stages = {
            'rising': {'age_max': 24, 'rank_improvement_6m': 30},
            'established': {'age_range': (25, 29), 'rank_stability_6m': 20},
            'veteran': {'age_min': 30, 'experience_years': 8},
            'declining': {'age_min': 29, 'rank_decline_12m': -50}
        }
    
    def create_complete_feature_set(self, player1_name: str, player2_name: str,
                                   player1_data: Dict, player2_data: Dict,
                                   match_context: Dict, first_set_data: Dict) -> Dict:
        """
        Create complete feature set for ranks 50-300 second set prediction
        
        Args:
            player1_name: Name of player 1
            player2_name: Name of player 2
            player1_data: Player 1 data including rank, age, form
            player2_data: Player 2 data including rank, age, form
            match_context: Tournament, surface, pressure context
            first_set_data: First set outcome and statistics
            
        Returns:
            Dict: Complete feature set optimized for ranks 50-300
        """
        
        features = {}
        
        # Validate players are in target rank range
        if not self._validate_rank_range(player1_data, player2_data):
            raise ValueError("Players must be in ranks 50-300 range")
        
        # 1. Core rank position features
        features.update(self._create_rank_position_features(player1_data, player2_data))
        
        # 2. Career trajectory features
        features.update(self._create_career_trajectory_features(player1_data, player2_data))
        
        # 3. Tournament context features
        features.update(self._create_tournament_context_features(match_context, player1_data, player2_data))
        
        # 4. Underdog/favorite dynamics
        features.update(self._create_underdog_dynamics_features(player1_data, player2_data, first_set_data))
        
        # 5. Second set specific features
        features.update(self._create_second_set_features(first_set_data, player1_data, player2_data))
        
        # 6. Psychological pressure features
        features.update(self._create_psychological_features(player1_data, player2_data, match_context))
        
        # 7. Surface and conditions features
        features.update(self._create_surface_features(match_context, player1_data, player2_data))
        
        # 8. Head-to-head and experience features
        features.update(self._create_experience_features(player1_data, player2_data, match_context))
        
        # 9. Form and momentum features  
        features.update(self._create_form_features(player1_data, player2_data, first_set_data))
        
        # 10. Rank-specific combination features
        features.update(self._create_combination_features(features))
        
        return features
    
    def _validate_rank_range(self, player1_data: Dict, player2_data: Dict) -> bool:
        """Validate both players are in 50-300 rank range"""
        p1_rank = player1_data.get('rank', 999)
        p2_rank = player2_data.get('rank', 999)
        
        return (self.rank_range[0] <= p1_rank <= self.rank_range[1] and
                self.rank_range[0] <= p2_rank <= self.rank_range[1])
    
    def _create_rank_position_features(self, player1_data: Dict, player2_data: Dict) -> Dict:
        """Create features based on rank positions within 50-300 tier"""
        features = {}
        
        p1_rank = player1_data.get('rank', 175)
        p2_rank = player2_data.get('rank', 175)
        
        # Tier position (0 to 1, where 0 = rank 50, 1 = rank 300)
        features['player1_tier_position'] = (p1_rank - 50) / self.tier_size
        features['player2_tier_position'] = (p2_rank - 50) / self.tier_size
        
        # Distance from key milestones
        features['player1_distance_from_50'] = max(0, p1_rank - 50) / 250
        features['player2_distance_from_50'] = max(0, p2_rank - 50) / 250
        features['player1_distance_from_100'] = max(0, p1_rank - 100) / 200
        features['player2_distance_from_100'] = max(0, p2_rank - 100) / 200
        
        # Risk of falling below 300
        features['player1_relegation_risk'] = max(0, (p1_rank - 250) / 50)
        features['player2_relegation_risk'] = max(0, (p2_rank - 250) / 50)
        
        # Promotion opportunity (closer to top 50)
        features['player1_top_50_opportunity'] = max(0, (75 - p1_rank) / 25)
        features['player2_top_50_opportunity'] = max(0, (75 - p2_rank) / 25)
        
        # Top 100 opportunity
        features['player1_top_100_opportunity'] = max(0, (125 - p1_rank) / 75)
        features['player2_top_100_opportunity'] = max(0, (125 - p2_rank) / 75)
        
        # Ranking gap analysis
        rank_gap = abs(p1_rank - p2_rank)
        features['ranking_gap'] = rank_gap
        features['ranking_gap_normalized'] = min(rank_gap / 100, 1.0)  # Normalize to 0-1
        
        # Gap categories for different prediction models
        if rank_gap < 20:
            features['gap_category_small'] = 1
            features['gap_category_medium'] = 0
            features['gap_category_large'] = 0
        elif rank_gap < 60:
            features['gap_category_small'] = 0
            features['gap_category_medium'] = 1
            features['gap_category_large'] = 0
        else:
            features['gap_category_small'] = 0
            features['gap_category_medium'] = 0
            features['gap_category_large'] = 1
        
        # Identify underdog and favorite
        if p1_rank > p2_rank:  # Higher number = lower rank = underdog
            features['player1_is_underdog'] = 1
            features['player2_is_underdog'] = 0
            features['underdog_rank'] = p1_rank
            features['favorite_rank'] = p2_rank
        else:
            features['player1_is_underdog'] = 0
            features['player2_is_underdog'] = 1
            features['underdog_rank'] = p2_rank
            features['favorite_rank'] = p1_rank
        
        return features
    
    def _create_career_trajectory_features(self, player1_data: Dict, player2_data: Dict) -> Dict:
        """Create career trajectory features specific to 50-300 tier"""
        features = {}
        
        for i, player_data in enumerate([player1_data, player2_data], 1):
            prefix = f'player{i}'
            
            current_rank = player_data.get('rank', 175)
            age = player_data.get('age', 25)
            
            # Rank changes over different periods
            rank_3m_ago = player_data.get('rank_3_months_ago', current_rank)
            rank_6m_ago = player_data.get('rank_6_months_ago', current_rank)
            rank_12m_ago = player_data.get('rank_12_months_ago', current_rank)
            
            features[f'{prefix}_rank_change_3m'] = rank_3m_ago - current_rank  # Positive = improvement
            features[f'{prefix}_rank_change_6m'] = rank_6m_ago - current_rank
            features[f'{prefix}_rank_change_12m'] = rank_12m_ago - current_rank
            
            # Career stage classification
            features[f'{prefix}_is_rising'] = 1 if (age < 24 and features[f'{prefix}_rank_change_6m'] > 30) else 0
            features[f'{prefix}_is_veteran'] = 1 if age > 30 else 0
            features[f'{prefix}_is_declining'] = 1 if (age > 29 and features[f'{prefix}_rank_change_12m'] < -50) else 0
            features[f'{prefix}_is_plateau'] = 1 if (abs(features[f'{prefix}_rank_change_6m']) < 20 and 25 <= age <= 29) else 0
            
            # Career high context
            career_high = player_data.get('career_high_rank', 300)
            features[f'{prefix}_career_high_gap'] = current_rank - career_high
            features[f'{prefix}_comeback_attempt'] = 1 if (career_high < 50 and current_rank > 100) else 0
            features[f'{prefix}_former_top_50'] = 1 if career_high <= 50 else 0
            features[f'{prefix}_former_top_100'] = 1 if career_high <= 100 else 0
            features[f'{prefix}_first_time_in_tier'] = 1 if (career_high > current_rank and age < 24) else 0
            
            # Momentum indicators
            features[f'{prefix}_positive_momentum'] = 1 if features[f'{prefix}_rank_change_3m'] > 15 else 0
            features[f'{prefix}_negative_momentum'] = 1 if features[f'{prefix}_rank_change_3m'] < -15 else 0
            
            # Stability vs volatility
            rank_volatility = np.std([
                features[f'{prefix}_rank_change_3m'],
                features[f'{prefix}_rank_change_6m'] / 2,
                features[f'{prefix}_rank_change_12m'] / 4
            ])
            features[f'{prefix}_rank_volatility'] = min(rank_volatility / 50, 1.0)
        
        return features
    
    def _create_tournament_context_features(self, match_context: Dict, 
                                           player1_data: Dict, player2_data: Dict) -> Dict:
        """Create tournament context features for 50-300 players"""
        features = {}
        
        tournament_level = match_context.get('tournament_level', 'ATP_250')
        
        for i, player_data in enumerate([player1_data, player2_data], 1):
            prefix = f'player{i}'
            
            # ATP vs Challenger experience
            total_matches = player_data.get('total_matches', 100)
            atp_matches = player_data.get('atp_main_draw_matches', 20)
            challenger_matches = player_data.get('challenger_matches', 50)
            
            features[f'{prefix}_atp_experience'] = atp_matches / max(total_matches, 1)
            features[f'{prefix}_challenger_dominance'] = player_data.get('challenger_win_rate', 0.5) - 0.5
            
            # Tournament level adaptation
            current_rank = player_data.get('rank', 175)
            
            if tournament_level in ['ATP_500', 'ATP_250']:
                # Many 50-300 players have varying comfort levels at ATP events
                features[f'{prefix}_playing_up_level'] = 1 if current_rank > 100 else 0
                features[f'{prefix}_big_stage_pressure'] = min(1.0, (300 - current_rank) / 250)
                features[f'{prefix}_atp_comfort'] = features[f'{prefix}_atp_experience']
            else:
                # Challenger level - comfort zone for lower ranked players
                features[f'{prefix}_playing_up_level'] = 0
                features[f'{prefix}_comfort_level'] = 1 if current_rank > 150 else 0.5
                features[f'{prefix}_challenger_advantage'] = features[f'{prefix}_challenger_dominance']
            
            # Qualifying experience
            qualifying_matches = player_data.get('qualifying_matches', 0)
            features[f'{prefix}_qualifying_experience'] = min(qualifying_matches / 50, 1.0)
            
            # Tournament importance for ranking
            points_available = match_context.get('ranking_points_available', 50)
            features[f'{prefix}_points_motivation'] = min(points_available / 250, 1.0)
        
        # Tournament context
        features['tournament_importance'] = match_context.get('tournament_importance', 2)
        features['is_challenger_event'] = 1 if tournament_level == 'Challenger' else 0
        features['is_atp_event'] = 1 if tournament_level in ['ATP_250', 'ATP_500'] else 0
        features['is_masters_event'] = 1 if tournament_level == 'ATP_Masters_1000' else 0
        
        return features
    
    def _create_underdog_dynamics_features(self, player1_data: Dict, player2_data: Dict, 
                                         first_set_data: Dict) -> Dict:
        """Create underdog/favorite dynamics specific to 50-300 tier"""
        features = {}
        
        p1_rank = player1_data.get('rank', 175)
        p2_rank = player2_data.get('rank', 175)
        
        # Determine underdog
        underdog_player = 1 if p1_rank > p2_rank else 2
        favorite_player = 2 if p1_rank > p2_rank else 1
        
        underdog_rank = p1_rank if underdog_player == 1 else p2_rank
        favorite_rank = p2_rank if underdog_player == 1 else p1_rank
        
        rank_gap = abs(p1_rank - p2_rank)
        
        # First set outcome context
        first_set_winner = first_set_data.get('winner', 'unknown')
        
        if first_set_winner == f'player{underdog_player}':
            features['underdog_won_first_set'] = 1
            features['underdog_lost_first_set'] = 0
            features['underdog_confidence_boost'] = min(0.3, rank_gap / 100)
            features['favorite_desperation_factor'] = min(0.25, rank_gap / 150)
        elif first_set_winner == f'player{favorite_player}':
            features['underdog_won_first_set'] = 0
            features['underdog_lost_first_set'] = 1
            features['underdog_nothing_to_lose'] = min(0.2, rank_gap / 120)
            features['favorite_comfort_zone'] = min(0.15, (300 - favorite_rank) / 250)
        else:
            # Unknown first set result
            features['underdog_won_first_set'] = 0
            features['underdog_lost_first_set'] = 0
            features['underdog_confidence_boost'] = 0
            features['underdog_nothing_to_lose'] = 0
        
        # Underdog type classification
        if rank_gap < 25:
            features['slight_underdog'] = 1
            features['underdog_base_probability'] = 0.45
        elif rank_gap < 60:
            features['clear_underdog'] = 1
            features['underdog_base_probability'] = 0.38
        elif rank_gap < 120:
            features['strong_underdog'] = 1
            features['underdog_base_probability'] = 0.28
        else:
            features['heavy_underdog'] = 1
            features['underdog_base_probability'] = 0.18
        
        # Psychological factors specific to 50-300 tier
        underdog_data = player1_data if underdog_player == 1 else player2_data
        favorite_data = player2_data if underdog_player == 1 else player1_data
        
        # Underdog motivation factors
        features['underdog_career_high_motivation'] = 1 if underdog_data.get('career_high_rank', 300) < underdog_rank else 0
        features['underdog_breakthrough_potential'] = 1 if (underdog_data.get('age', 30) < 25 and underdog_rank > 150) else 0
        
        # Favorite pressure factors - adjusted for expanded range
        if favorite_rank <= 50:
            features['favorite_ranking_pressure'] = min(0.2, max(0, (favorite_rank - 30) / 20))
        elif favorite_rank <= 100:
            features['favorite_ranking_pressure'] = min(0.3, max(0, (favorite_rank - 75) / 25))
        else:
            features['favorite_ranking_pressure'] = min(0.3, max(0, (favorite_rank - 120) / 80))
        
        features['favorite_expectation_weight'] = min(0.25, rank_gap / 150)
        
        return features
    
    def _create_second_set_features(self, first_set_data: Dict, 
                                   player1_data: Dict, player2_data: Dict) -> Dict:
        """Create second set specific features"""
        features = {}
        
        # First set characteristics
        first_set_score = first_set_data.get('score', '6-4')
        features['first_set_games_diff'] = self._calculate_games_difference(first_set_score)
        features['first_set_total_games'] = self._calculate_total_games(first_set_score)
        
        # Set closeness indicators
        games_diff = abs(features['first_set_games_diff'])
        features['first_set_close'] = 1 if games_diff <= 2 else 0
        features['first_set_dominant'] = 1 if games_diff >= 4 else 0
        
        # Duration and momentum
        duration_minutes = first_set_data.get('duration_minutes', 45)
        features['first_set_duration'] = duration_minutes
        features['first_set_long'] = 1 if duration_minutes > 60 else 0
        features['first_set_quick'] = 1 if duration_minutes < 35 else 0
        
        # Break points and serving
        p1_bp_saved = first_set_data.get('break_points_saved_player1', 0.5)
        p2_bp_saved = first_set_data.get('break_points_saved_player2', 0.5)
        
        features['player1_bp_save_rate'] = p1_bp_saved
        features['player2_bp_save_rate'] = p2_bp_saved
        features['bp_save_difference'] = p1_bp_saved - p2_bp_saved
        
        # Service performance
        p1_serve_pct = first_set_data.get('first_serve_percentage_player1', 0.65)
        p2_serve_pct = first_set_data.get('first_serve_percentage_player2', 0.65)
        
        features['player1_serve_percentage'] = p1_serve_pct
        features['player2_serve_percentage'] = p2_serve_pct
        features['serve_percentage_diff'] = p1_serve_pct - p2_serve_pct
        
        # Tiebreak indicator
        features['had_tiebreak'] = 1 if '7-6' in first_set_score or '6-7' in first_set_score else 0
        
        # Historical second set performance for 50-300 players
        for i, player_data in enumerate([player1_data, player2_data], 1):
            prefix = f'player{i}'
            
            # Second set improvement patterns (estimated from historical data)
            age = player_data.get('age', 25)
            rank = player_data.get('rank', 175)
            
            # Younger players in this tier often improve in second sets
            features[f'{prefix}_second_set_improvement'] = max(0, (25 - age) / 10) * 0.1 if rank > 100 else 0
            
            # Comeback ability based on career stage and ranking tier
            if player_data.get('career_high_rank', 300) <= 50:  # Former top 50
                features[f'{prefix}_comeback_ability'] = 0.8
            elif player_data.get('career_high_rank', 300) <= 100:  # Former top 100
                features[f'{prefix}_comeback_ability'] = 0.7
            elif age < 24:  # Young and hungry
                features[f'{prefix}_comeback_ability'] = 0.6
            else:
                features[f'{prefix}_comeback_ability'] = 0.5
        
        # Momentum indicators
        winner = first_set_data.get('winner', 'player1')
        loser = 'player2' if winner == 'player1' else 'player1'
        
        # Momentum often swings in 50-300 tier after close sets
        if features['first_set_close']:
            features['momentum_with_loser'] = 0.15  # Loser gains momentum
        else:
            features['momentum_with_loser'] = 0.05
        
        return features
    
    def _create_psychological_features(self, player1_data: Dict, player2_data: Dict, 
                                      match_context: Dict) -> Dict:
        """Create psychological pressure features for 50-300 players"""
        features = {}
        
        for i, player_data in enumerate([player1_data, player2_data], 1):
            prefix = f'player{i}'
            
            rank = player_data.get('rank', 175)
            age = player_data.get('age', 25)
            
            # Ranking anxiety (specific to 50-300 tier)
            if 50 < rank <= 75:
                features[f'{prefix}_top_50_anxiety'] = 0.35  # Pressure to break into top 50
            elif 100 < rank <= 120:
                features[f'{prefix}_top_100_anxiety'] = 0.3  # Pressure to break into top 100
            elif 280 <= rank <= 300:
                features[f'{prefix}_relegation_anxiety'] = 0.25  # Fear of falling out of tier
            else:
                features[f'{prefix}_ranking_anxiety'] = 0.1
            
            # Career stage pressure
            if age < 24 and rank > 150:
                features[f'{prefix}_breakthrough_pressure'] = 0.2
            elif age > 30 and rank > 200:
                features[f'{prefix}_career_survival_pressure'] = 0.25
            else:
                features[f'{prefix}_career_pressure'] = 0.1
            
            # Tournament importance pressure
            tournament_level = match_context.get('tournament_level', 'ATP_250')
            if tournament_level in ['ATP_Masters_1000'] and rank > 100:
                features[f'{prefix}_masters_pressure'] = 0.3
            elif tournament_level in ['ATP_500', 'ATP_250'] and rank > 150:
                features[f'{prefix}_big_tournament_pressure'] = 0.2
            else:
                features[f'{prefix}_tournament_pressure'] = 0.05
        
        # Match situation pressure
        round_name = match_context.get('round', 'R32')
        if round_name in ['F', 'SF', 'QF']:
            features['late_round_pressure'] = 0.2
        else:
            features['early_round_pressure'] = 0.05
        
        return features
    
    def _create_surface_features(self, match_context: Dict, 
                                player1_data: Dict, player2_data: Dict) -> Dict:
        """Create surface-specific features for 50-300 players"""
        features = {}
        
        surface = match_context.get('surface', 'Hard').lower()
        
        for i, player_data in enumerate([player1_data, player2_data], 1):
            prefix = f'player{i}'
            
            # Surface win rates
            surface_wr = player_data.get(f'{surface}_court_win_rate', 0.5)
            overall_wr = player_data.get('overall_win_rate', 0.5)
            
            features[f'{prefix}_surface_advantage'] = surface_wr - overall_wr
            features[f'{prefix}_surface_specialist'] = 1 if abs(features[f'{prefix}_surface_advantage']) > 0.15 else 0
            
            # Surface experience (important for 50-300 players)
            surface_matches = player_data.get(f'{surface}_court_matches', 50)
            features[f'{prefix}_surface_experience'] = min(surface_matches / 100, 1.0)
        
        # Relative surface advantages
        features['surface_advantage_gap'] = (features['player1_surface_advantage'] - 
                                            features['player2_surface_advantage'])
        
        return features
    
    def _create_experience_features(self, player1_data: Dict, player2_data: Dict, 
                                   match_context: Dict) -> Dict:
        """Create experience-based features"""
        features = {}
        
        # Head-to-head (more common in 50-300 tier than pure 101-300)
        h2h_matches = match_context.get('h2h_matches', 0)
        features['h2h_matches'] = h2h_matches
        features['limited_h2h'] = 1 if h2h_matches <= 2 else 0
        
        if h2h_matches > 0:
            features['h2h_win_rate_p1'] = match_context.get('h2h_win_rate_p1', 0.5)
        else:
            features['h2h_win_rate_p1'] = 0.5
        
        # Professional experience
        for i, player_data in enumerate([player1_data, player2_data], 1):
            prefix = f'player{i}'
            
            pro_years = player_data.get('professional_years', 5)
            features[f'{prefix}_experience_years'] = min(pro_years / 10, 1.0)
            
            # Big match experience (finals, semifinals)
            big_matches = player_data.get('career_finals', 0) + player_data.get('career_semifinals', 0)
            features[f'{prefix}_big_match_experience'] = min(big_matches / 10, 1.0)
        
        return features
    
    def _create_form_features(self, player1_data: Dict, player2_data: Dict, 
                             first_set_data: Dict) -> Dict:
        """Create recent form and momentum features"""
        features = {}
        
        for i, player_data in enumerate([player1_data, player2_data], 1):
            prefix = f'player{i}'
            
            # Recent match results
            recent_wr = player_data.get('recent_win_rate_10', 0.5)
            features[f'{prefix}_recent_form'] = recent_wr - 0.5  # Centered around 0
            
            # Form trend
            form_trend = player_data.get('form_trend', 0)  # Positive = improving
            features[f'{prefix}_form_trend'] = max(-0.3, min(0.3, form_trend / 100))
            
            # Days since last match (rest vs rhythm)
            days_rest = player_data.get('days_since_last_match', 7)
            if days_rest < 2:
                features[f'{prefix}_rest_factor'] = -0.1  # Too little rest
            elif days_rest > 14:
                features[f'{prefix}_rest_factor'] = -0.05  # Too much rest/lack of rhythm
            else:
                features[f'{prefix}_rest_factor'] = 0.05  # Good rest
        
        # Momentum from first set
        winner = first_set_data.get('winner', 'player1')
        features['winner_momentum'] = 0.1
        features['loser_pressure_to_respond'] = 0.15
        
        return features
    
    def _create_combination_features(self, features: Dict) -> Dict:
        """Create combination features from existing features"""
        combo_features = {}
        
        # Key combination features for 50-300 tier
        
        # Rank gap × first set closeness
        combo_features['rank_gap_times_first_set_closeness'] = (
            features.get('ranking_gap_normalized', 0.3) * 
            features.get('first_set_close', 0)
        )
        
        # Momentum × adaptation ability
        combo_features['momentum_times_adaptation'] = (
            features.get('momentum_with_loser', 0.1) * 
            (features.get('player1_comeback_ability', 0.5) + 
             features.get('player2_comeback_ability', 0.5)) / 2
        )
        
        # Pressure × fatigue interaction
        total_pressure = (features.get('player1_ranking_anxiety', 0.1) + 
                         features.get('player2_ranking_anxiety', 0.1))
        set_length_factor = min(features.get('first_set_duration', 45) / 60, 1.0)
        
        combo_features['pressure_fatigue_interaction'] = total_pressure * set_length_factor
        
        # Career stage × rank gap
        rising_players = (features.get('player1_is_rising', 0) + 
                         features.get('player2_is_rising', 0))
        combo_features['rising_player_upset_potential'] = (
            rising_players * features.get('ranking_gap_normalized', 0.3)
        )
        
        # Experience gap
        exp_gap = abs(features.get('player1_experience_years', 0.5) - 
                     features.get('player2_experience_years', 0.5))
        combo_features['experience_gap'] = exp_gap
        
        # Former elite player comeback potential
        former_elite = (features.get('player1_former_top_50', 0) + 
                       features.get('player2_former_top_50', 0))
        combo_features['former_elite_comeback_factor'] = former_elite * 0.2
        
        return combo_features
    
    def _calculate_games_difference(self, score: str) -> int:
        """Calculate games difference from set score (e.g., '6-4' -> 2)"""
        try:
            if '-' in score:
                games = score.split('-')
                return int(games[0]) - int(games[1])
        except:
            pass
        return 0
    
    def _calculate_total_games(self, score: str) -> int:
        """Calculate total games in set (e.g., '6-4' -> 10)"""
        try:
            if '-' in score:
                games = score.split('-')
                return int(games[0]) + int(games[1])
        except:
            pass
        return 10

class Ranks50to300DataValidator:
    """Validate data quality for ranks 50-300 predictions"""
    
    def __init__(self):
        self.rank_range = (50, 300)
    
    def validate_match_data(self, match_data: Dict) -> Dict[str, any]:
        """Validate match data for ranks 50-300 prediction"""
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check rank range
        p1_rank = match_data.get('player1', {}).get('rank', 999)
        p2_rank = match_data.get('player2', {}).get('rank', 999)
        
        if not (self.rank_range[0] <= p1_rank <= self.rank_range[1]):
            validation_result['errors'].append(f"Player 1 rank {p1_rank} outside target range 50-300")
            validation_result['valid'] = False
        
        if not (self.rank_range[0] <= p2_rank <= self.rank_range[1]):
            validation_result['errors'].append(f"Player 2 rank {p2_rank} outside target range 50-300")
            validation_result['valid'] = False
        
        # Check data completeness
        required_fields = ['rank', 'age']
        for player_key in ['player1', 'player2']:
            player_data = match_data.get(player_key, {})
            for field in required_fields:
                if field not in player_data:
                    validation_result['warnings'].append(f"{player_key} missing {field}")
        
        # Check first set data
        first_set_data = match_data.get('first_set_data', {})
        if 'winner' not in first_set_data:
            validation_result['warnings'].append("First set winner not specified")
        
        return validation_result

# Example usage and testing
if __name__ == "__main__":
    print("🎯 RANKS 50-300 FEATURE ENGINEERING SYSTEM TEST")
    print("=" * 60)
    
    # Test feature engineering
    feature_engineer = Ranks50to300FeatureEngineer()
    
    # Sample data for ranks 50-300 players
    player1_data = {
        'rank': 85,  # Updated to be within new range
        'age': 23,
        'rank_3_months_ago': 120,
        'rank_6_months_ago': 180,
        'rank_12_months_ago': 250,
        'career_high_rank': 85,
        'recent_win_rate_10': 0.6,
        'hard_court_win_rate': 0.55,
        'overall_win_rate': 0.52,
        'professional_years': 4
    }
    
    player2_data = {
        'rank': 165,  # Updated to be within new range
        'age': 27,
        'rank_3_months_ago': 150,
        'rank_6_months_ago': 135,
        'rank_12_months_ago': 120,
        'career_high_rank': 45,  # Former top 50 player
        'recent_win_rate_10': 0.4,
        'hard_court_win_rate': 0.48,
        'overall_win_rate': 0.50,
        'professional_years': 8
    }
    
    match_context = {
        'tournament_level': 'ATP_250',
        'surface': 'Hard',
        'tournament_importance': 3,
        'h2h_matches': 1,
        'h2h_win_rate_p1': 1.0
    }
    
    first_set_data = {
        'winner': 'player2',  # Favorite (lower rank) won first set
        'score': '6-4',
        'duration_minutes': 52,
        'break_points_saved_player1': 0.33,
        'break_points_saved_player2': 0.75,
        'first_serve_percentage_player1': 0.62,
        'first_serve_percentage_player2': 0.71
    }
    
    try:
        features = feature_engineer.create_complete_feature_set(
            "rising player", "veteran comeback",
            player1_data, player2_data, match_context, first_set_data
        )
        
        print(f"\n✅ Generated {len(features)} features for ranks 50-300 prediction")
        
        # Display key features
        key_features = [
            'player1_is_underdog', 'ranking_gap', 'underdog_lost_first_set',
            'player1_is_rising', 'player2_comeback_attempt', 
            'underdog_nothing_to_lose', 'momentum_with_loser',
            'player2_former_top_50'
        ]
        
        print("\n🔍 Key Features:")
        for feature in key_features:
            if feature in features:
                print(f"  {feature}: {features[feature]:.3f}")
        
        # Validate data
        validator = Ranks50to300DataValidator()
        validation = validator.validate_match_data({
            'player1': player1_data,
            'player2': player2_data,
            'first_set_data': first_set_data
        })
        
        print(f"\n🔍 Data Validation: {'✅ VALID' if validation['valid'] else '❌ INVALID'}")
        if validation['warnings']:
            print("⚠️ Warnings:")
            for warning in validation['warnings']:
                print(f"  • {warning}")
        
        print(f"\n✅ Ranks 50-300 feature engineering system ready!")
        print(f"🎯 Optimized for second set underdog predictions in this expanded tier")
        
    except Exception as e:
        print(f"❌ Error in feature engineering: {e}")