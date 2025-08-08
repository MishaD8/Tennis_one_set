#!/usr/bin/env python3
"""
🎾 SECOND SET FEATURE ENGINEERING
Specialized features for predicting second set outcomes for underdogs
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

class SecondSetFeatureEngineer:
    """
    Feature engineering specifically for second set prediction
    Focus: Underdog probability to win the second set given first set context
    """
    
    def __init__(self):
        # Player-specific second set improvement patterns (historical data simulation)
        self.second_set_patterns = {
            # Players who typically improve in second sets
            "carlos alcaraz": {"improvement_rate": 0.15, "comeback_ability": 0.8},
            "novak djokovic": {"improvement_rate": 0.20, "comeback_ability": 0.9},
            "jannik sinner": {"improvement_rate": 0.12, "comeback_ability": 0.7},
            "daniil medvedev": {"improvement_rate": 0.08, "comeback_ability": 0.6},
            
            # Players who may decline in second sets
            "alexander zverev": {"improvement_rate": -0.05, "comeback_ability": 0.4},
            "stefanos tsitsipas": {"improvement_rate": -0.03, "comeback_ability": 0.5},
            
            # Underdog players with strong second set patterns
            "flavio cobolli": {"improvement_rate": 0.18, "comeback_ability": 0.7},
            "brandon nakashima": {"improvement_rate": 0.10, "comeback_ability": 0.6},
            "ben shelton": {"improvement_rate": 0.22, "comeback_ability": 0.8},
        }
    
    def create_first_set_context_features(self, first_set_data: Dict) -> Dict:
        """
        Create features based on first set outcome and patterns
        
        Args:
            first_set_data: {
                'winner': 'player1' or 'player2',
                'score': '6-4', '7-6', etc.,
                'duration_minutes': int,
                'breaks_won_player1': int,
                'breaks_won_player2': int,
                'break_points_saved_player1': float (0-1),
                'break_points_saved_player2': float (0-1),
                'first_serve_percentage_player1': float (0-1),
                'first_serve_percentage_player2': float (0-1),
                'had_tiebreak': bool
            }
        """
        features = {}
        
        # Parse score
        if 'score' in first_set_data and isinstance(first_set_data['score'], str):
            try:
                games = first_set_data['score'].split('-')
                player1_games = int(games[0])
                player2_games = int(games[1])
                
                features['first_set_games_diff'] = player1_games - player2_games
                features['first_set_total_games'] = player1_games + player2_games
                features['first_set_close'] = 1.0 if abs(player1_games - player2_games) <= 2 else 0.0
                features['first_set_dominant'] = 1.0 if abs(player1_games - player2_games) >= 4 else 0.0
                
            except (ValueError, IndexError):
                # Default values if score parsing fails
                features['first_set_games_diff'] = 0
                features['first_set_total_games'] = 12
                features['first_set_close'] = 1.0
                features['first_set_dominant'] = 0.0
        else:
            # Default values
            features['first_set_games_diff'] = 0
            features['first_set_total_games'] = 12
            features['first_set_close'] = 1.0
            features['first_set_dominant'] = 0.0
        
        # Winner context
        features['player1_won_first_set'] = 1.0 if first_set_data.get('winner') == 'player1' else 0.0
        features['player2_won_first_set'] = 1.0 if first_set_data.get('winner') == 'player2' else 0.0
        
        # Set length and intensity
        duration = first_set_data.get('duration_minutes', 45)
        features['first_set_duration'] = float(duration)
        features['first_set_long'] = 1.0 if duration > 60 else 0.0
        features['first_set_quick'] = 1.0 if duration < 35 else 0.0
        
        # Break point dynamics (crucial for second set prediction)
        breaks_p1 = first_set_data.get('breaks_won_player1', 1)
        breaks_p2 = first_set_data.get('breaks_won_player2', 1)
        features['breaks_difference'] = float(breaks_p1 - breaks_p2)
        features['total_breaks'] = float(breaks_p1 + breaks_p2)
        features['break_fest'] = 1.0 if (breaks_p1 + breaks_p2) > 4 else 0.0
        
        # Break point save rates (mental toughness indicator)
        bp_saved_p1 = first_set_data.get('break_points_saved_player1', 0.5)
        bp_saved_p2 = first_set_data.get('break_points_saved_player2', 0.5)
        features['player1_bp_save_rate'] = float(bp_saved_p1)
        features['player2_bp_save_rate'] = float(bp_saved_p2)
        features['bp_save_difference'] = bp_saved_p1 - bp_saved_p2
        
        # Serving performance (affects second set confidence)
        serve_p1 = first_set_data.get('first_serve_percentage_player1', 0.65)
        serve_p2 = first_set_data.get('first_serve_percentage_player2', 0.65)
        features['player1_serve_percentage'] = float(serve_p1)
        features['player2_serve_percentage'] = float(serve_p2)
        features['serve_percentage_diff'] = serve_p1 - serve_p2
        
        # Tiebreak factor
        features['had_tiebreak'] = 1.0 if first_set_data.get('had_tiebreak', False) else 0.0
        
        # Momentum indicators
        # If loser had opportunities but failed to convert, they might improve in set 2
        features['momentum_with_loser'] = 0.0
        if first_set_data.get('winner') == 'player1':
            # Player2 lost but had break chances
            if bp_saved_p1 < 0.6 and breaks_p2 > 0:
                features['momentum_with_loser'] = 0.3
        else:
            # Player1 lost but had break chances
            if bp_saved_p2 < 0.6 and breaks_p1 > 0:
                features['momentum_with_loser'] = 0.3
        
        return features
    
    def create_momentum_features(self, player1_name: str, player2_name: str, 
                                first_set_context: Dict) -> Dict:
        """Create momentum-based features for second set"""
        features = {}
        
        # Player-specific second set patterns
        p1_pattern = self.second_set_patterns.get(player1_name.lower(), 
                                                 {"improvement_rate": 0.0, "comeback_ability": 0.5})
        p2_pattern = self.second_set_patterns.get(player2_name.lower(),
                                                 {"improvement_rate": 0.0, "comeback_ability": 0.5})
        
        features['player1_second_set_improvement'] = p1_pattern['improvement_rate']
        features['player2_second_set_improvement'] = p2_pattern['improvement_rate']
        features['player1_comeback_ability'] = p1_pattern['comeback_ability']
        features['player2_comeback_ability'] = p2_pattern['comeback_ability']
        
        # Context-dependent momentum
        first_set_winner = first_set_context.get('winner', 'player1')
        
        if first_set_winner == 'player1':
            # Player1 won first set
            features['winner_momentum'] = 0.1  # Small momentum advantage
            features['loser_pressure_to_respond'] = 0.2  # Pressure can motivate
            features['active_comeback_scenario'] = p2_pattern['comeback_ability']
        else:
            # Player2 won first set
            features['winner_momentum'] = -0.1  # Negative for player1
            features['loser_pressure_to_respond'] = 0.2
            features['active_comeback_scenario'] = p1_pattern['comeback_ability']
        
        # Fatigue and endurance factors
        set_duration = first_set_context.get('duration_minutes', 45)
        features['fatigue_factor_player1'] = min(0.2, set_duration / 300)  # Max 20% fatigue impact
        features['fatigue_factor_player2'] = min(0.2, set_duration / 300)
        
        # Mental pressure adjustments
        if first_set_context.get('had_tiebreak', False):
            features['mental_fatigue_bonus'] = 0.1  # Tiebreaks are mentally draining
        else:
            features['mental_fatigue_bonus'] = 0.0
        
        return features
    
    def create_adaptation_features(self, player1_data: Dict, player2_data: Dict,
                                 first_set_context: Dict) -> Dict:
        """Features related to player adaptation and tactical changes"""
        features = {}
        
        # Age and experience factors for adaptation
        p1_age = player1_data.get('age', 25)
        p2_age = player2_data.get('age', 25)
        
        # Younger players often adapt better between sets
        features['player1_adaptation_age_factor'] = max(0.5, 1.0 - (p1_age - 20) * 0.02)
        features['player2_adaptation_age_factor'] = max(0.5, 1.0 - (p2_age - 20) * 0.02)
        
        # Experience (based on ranking) helps with tactical adjustments
        p1_rank = player1_data.get('rank', 100)
        p2_rank = player2_data.get('rank', 100)
        
        features['player1_tactical_experience'] = max(0.1, 1.0 - p1_rank / 200)
        features['player2_tactical_experience'] = max(0.1, 1.0 - p2_rank / 200)
        
        # Adaptation pressure based on first set performance
        if first_set_context.get('first_set_dominant', 0) > 0.5:
            # One player was dominant, loser needs to adapt more
            if first_set_context.get('winner') == 'player1':
                features['player2_adaptation_pressure'] = 0.8
                features['player1_adaptation_pressure'] = 0.2
            else:
                features['player1_adaptation_pressure'] = 0.8
                features['player2_adaptation_pressure'] = 0.2
        else:
            # Close set, both need moderate adaptation
            features['player1_adaptation_pressure'] = 0.4
            features['player2_adaptation_pressure'] = 0.4
        
        # Playing style adaptation (simulation based on rankings)
        # Higher ranked players typically have more versatile games
        features['player1_tactical_versatility'] = min(0.9, max(0.3, 1.0 - p1_rank / 100))
        features['player2_tactical_versatility'] = min(0.9, max(0.3, 1.0 - p2_rank / 100))
        
        return features
    
    def create_underdog_specific_features(self, player1_data: Dict, player2_data: Dict,
                                        first_set_context: Dict) -> Dict:
        """Features specifically for underdog second set prediction"""
        features = {}
        
        # Identify underdog
        p1_rank = player1_data.get('rank', 100)
        p2_rank = player2_data.get('rank', 100)
        
        if p1_rank > p2_rank:  # Player1 is underdog
            underdog_rank = p1_rank
            favorite_rank = p2_rank
            features['player1_is_underdog'] = 1.0
            features['player2_is_underdog'] = 0.0
            underdog_won_first = first_set_context.get('winner') == 'player1'
        else:  # Player2 is underdog
            underdog_rank = p2_rank
            favorite_rank = p1_rank
            features['player1_is_underdog'] = 0.0
            features['player2_is_underdog'] = 1.0
            underdog_won_first = first_set_context.get('winner') == 'player2'
        
        rank_gap = underdog_rank - favorite_rank
        features['ranking_gap'] = float(rank_gap)
        
        # Underdog scenarios
        features['underdog_won_first_set'] = 1.0 if underdog_won_first else 0.0
        features['underdog_lost_first_set'] = 1.0 if not underdog_won_first else 0.0
        
        # Different dynamics based on first set outcome
        if underdog_won_first:
            # Underdog ahead - may face pressure, but also confidence boost
            features['underdog_confidence_boost'] = min(0.3, rank_gap / 200)
            features['underdog_pressure_as_leader'] = min(0.2, rank_gap / 300)
            features['favorite_desperation_factor'] = min(0.4, rank_gap / 150)
        else:
            # Underdog behind - nothing to lose mentality
            features['underdog_nothing_to_lose'] = min(0.4, rank_gap / 100)
            features['underdog_relaxation_factor'] = min(0.2, rank_gap / 200)
            features['favorite_comfort_zone'] = min(0.1, rank_gap / 300)
        
        # Underdog improvement potential based on first set patterns
        if first_set_context.get('first_set_close', 0) > 0.5:
            # Close first set suggests underdog can compete
            features['underdog_competitive_indicator'] = 1.0
            features['second_set_underdog_value'] = min(0.3, rank_gap / 150)
        else:
            features['underdog_competitive_indicator'] = 0.0
            features['second_set_underdog_value'] = min(0.1, rank_gap / 300)
        
        # Special underdog factors
        break_point_performance = first_set_context.get('bp_save_difference', 0)
        if features['player1_is_underdog'] > 0.5:
            # Player1 is underdog, check their break point performance
            underdog_bp_performance = first_set_context.get('player1_bp_save_rate', 0.5)
        else:
            # Player2 is underdog
            underdog_bp_performance = first_set_context.get('player2_bp_save_rate', 0.5)
        
        features['underdog_mental_toughness'] = underdog_bp_performance
        
        return features
    
    def create_complete_feature_set(self, player1_name: str, player2_name: str,
                                  player1_data: Dict, player2_data: Dict,
                                  match_context: Dict, first_set_data: Dict) -> Dict:
        """
        Create complete feature set for second set prediction
        
        Args:
            player1_name: Name of player 1
            player2_name: Name of player 2  
            player1_data: Player 1 stats (rank, age, etc.)
            player2_data: Player 2 stats (rank, age, etc.)
            match_context: Tournament, surface, etc.
            first_set_data: First set outcome and statistics
        
        Returns:
            Dict: Complete feature set ready for ML models
        """
        
        # Start with basic match features (from original system)
        features = {}
        
        # Basic player data
        features['player_rank'] = float(player1_data.get('rank', 100))
        features['player_age'] = float(player1_data.get('age', 25))
        features['opponent_rank'] = float(player2_data.get('rank', 100))
        features['opponent_age'] = float(player2_data.get('age', 25))
        
        # Match context
        features['tournament_importance'] = float(match_context.get('tournament_importance', 2))
        features['total_pressure'] = float(match_context.get('total_pressure', 2.5))
        features['surface_advantage'] = float(match_context.get('player1_surface_advantage', 0.0))
        
        # Add all second set specific features
        first_set_features = self.create_first_set_context_features(first_set_data)
        momentum_features = self.create_momentum_features(player1_name, player2_name, first_set_data)
        adaptation_features = self.create_adaptation_features(player1_data, player2_data, first_set_features)
        underdog_features = self.create_underdog_specific_features(player1_data, player2_data, first_set_features)
        
        # Combine all features
        features.update(first_set_features)
        features.update(momentum_features)
        features.update(adaptation_features)
        features.update(underdog_features)
        
        # Calculate engineered combinations for second set
        features['momentum_times_adaptation'] = (
            momentum_features.get('active_comeback_scenario', 0.5) * 
            features.get('underdog_adaptation_pressure', 0.4)
        )
        
        features['pressure_fatigue_interaction'] = (
            features.get('total_pressure', 2.5) * 
            features.get('fatigue_factor_player1', 0.1)
        )
        
        features['rank_gap_times_first_set_closeness'] = (
            features.get('ranking_gap', 20) * 
            features.get('first_set_close', 1.0)
        )
        
        return features

# Example usage and testing
if __name__ == "__main__":
    print("🎾 SECOND SET FEATURE ENGINEERING TEST")
    print("=" * 60)
    
    engineer = SecondSetFeatureEngineer()
    
    # Test data: Underdog Cobolli vs Favorite Djokovic after losing first set
    player1_data = {"rank": 32, "age": 22}  # Cobolli
    player2_data = {"rank": 5, "age": 37}   # Djokovic
    
    match_context = {
        "tournament_importance": 4,  # Grand Slam
        "total_pressure": 3.8,
        "player1_surface_advantage": -0.05  # Slight disadvantage
    }
    
    first_set_data = {
        "winner": "player2",  # Djokovic won first set
        "score": "6-4",
        "duration_minutes": 48,
        "breaks_won_player1": 0,
        "breaks_won_player2": 1,
        "break_points_saved_player1": 0.4,  # Cobolli struggled on BPs
        "break_points_saved_player2": 0.8,  # Djokovic solid
        "first_serve_percentage_player1": 0.68,
        "first_serve_percentage_player2": 0.72,
        "had_tiebreak": False
    }
    
    features = engineer.create_complete_feature_set(
        "flavio cobolli", "novak djokovic",
        player1_data, player2_data, match_context, first_set_data
    )
    
    print(f"Generated {len(features)} features for second set prediction:")
    print("\n🔍 Key Second Set Features:")
    
    important_features = [
        'underdog_lost_first_set', 'underdog_nothing_to_lose', 
        'player1_second_set_improvement', 'first_set_close',
        'ranking_gap', 'underdog_competitive_indicator',
        'momentum_with_loser', 'active_comeback_scenario'
    ]
    
    for feature in important_features:
        if feature in features:
            print(f"  {feature}: {features[feature]:.3f}")
    
    print(f"\n✅ Second set feature engineering complete!")
    print(f"🎯 Focus: Underdog probability to win SET 2 after losing SET 1")