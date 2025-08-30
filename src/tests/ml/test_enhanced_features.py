#!/usr/bin/env python3
"""
Tests for Enhanced Feature Engineering Module
============================================

Comprehensive tests for the enhanced tennis feature engineering system
including momentum, fatigue, and pressure indicators.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.ml.enhanced_feature_engineering import (
    EnhancedTennisFeatureEngineer,
    MatchContext,
    FirstSetStats
)

class TestEnhancedFeatureEngineering:
    """Test suite for enhanced feature engineering"""
    
    @pytest.fixture
    def feature_engineer(self):
        """Create feature engineer instance"""
        return EnhancedTennisFeatureEngineer()
    
    @pytest.fixture
    def sample_first_set_stats(self):
        """Create sample first set statistics"""
        return FirstSetStats(
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
        )
    
    @pytest.fixture
    def sample_match_context(self):
        """Create sample match context"""
        return MatchContext(
            tournament_tier='ATP500',
            surface='Hard',
            round='R16',
            is_indoor=True,
            altitude=500.0
        )
    
    @pytest.fixture
    def sample_player_data(self):
        """Create sample player data"""
        return {
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
            }
        }
    
    def test_momentum_features_creation(self, feature_engineer, sample_first_set_stats, sample_player_data):
        """Test momentum feature creation"""
        player1_data = sample_player_data['player1']
        player2_data = sample_player_data['player2']
        
        features = feature_engineer.create_momentum_features(
            sample_first_set_stats, player1_data, player2_data
        )
        
        # Check that all expected features are present
        expected_features = [
            'first_set_winner_momentum', 'underdog_bp_efficiency', 'favorite_bp_efficiency',
            'bp_efficiency_gap', 'underdog_bp_save_rate', 'favorite_bp_save_rate',
            'bp_save_rate_gap', 'underdog_first_set_serve_pct', 'favorite_first_set_serve_pct',
            'serve_pct_gap', 'underdog_error_rate', 'favorite_error_rate', 'error_rate_gap',
            'underdog_winner_rate', 'favorite_winner_rate', 'winner_rate_gap',
            'composite_momentum_score', 'first_set_closeness'
        ]
        
        for feature in expected_features:
            assert feature in features, f"Missing momentum feature: {feature}"
        
        # Check value ranges
        assert 0 <= features['composite_momentum_score'] <= 1, "Momentum score out of range"
        assert 0 <= features['first_set_closeness'] <= 1, "Closeness score out of range"
        
        # Check that break point features are calculated correctly
        assert features['underdog_bp_efficiency'] == 0.0  # 0/3 break points converted
        assert features['favorite_bp_efficiency'] == 0.5   # 1/2 break points converted
    
    def test_fatigue_features_creation(self, feature_engineer, sample_player_data):
        """Test fatigue feature creation"""
        player1_data = sample_player_data['player1']
        player2_data = sample_player_data['player2']
        match_date = datetime.now()
        
        features = feature_engineer.create_fatigue_features(
            player1_data, player2_data, match_date
        )
        
        # Check expected features
        expected_features = [
            'favorite_days_rest', 'underdog_days_rest', 'rest_advantage',
            'favorite_rest_quality', 'underdog_rest_quality', 'rest_quality_advantage',
            'favorite_recent_load', 'underdog_recent_load', 'load_difference',
            'favorite_travel_fatigue', 'underdog_travel_fatigue', 'travel_fatigue_advantage'
        ]
        
        for feature in expected_features:
            assert feature in features, f"Missing fatigue feature: {feature}"
        
        # Check rest calculations
        assert features['favorite_days_rest'] == 3
        assert features['underdog_days_rest'] == 5
        assert features['rest_advantage'] == 2  # Underdog more rested
        
        # Check rest quality (optimal is 1-7 days)
        assert 0.5 <= features['favorite_rest_quality'] <= 1.0
        assert 0.5 <= features['underdog_rest_quality'] <= 1.0
    
    def test_pressure_features_creation(self, feature_engineer, sample_player_data, sample_match_context):
        """Test pressure feature creation"""
        player1_data = sample_player_data['player1']
        player2_data = sample_player_data['player2']
        
        features = feature_engineer.create_pressure_features(
            player1_data, player2_data, sample_match_context
        )
        
        # Check expected features
        expected_features = [
            'tournament_pressure', 'favorite_ranking_pressure', 'underdog_ranking_pressure',
            'pressure_differential', 'round_pressure', 'expectation_pressure',
            'favorite_age_pressure', 'underdog_age_pressure', 'age_pressure_advantage',
            'total_pressure_on_favorite'
        ]
        
        for feature in expected_features:
            assert feature in features, f"Missing pressure feature: {feature}"
        
        # Check pressure calculations
        assert features['tournament_pressure'] == 0.7  # ATP500 weight
        assert features['round_pressure'] == 0.5       # R16 weight
        
        # Higher ranked players should have more pressure
        assert features['favorite_ranking_pressure'] > features['underdog_ranking_pressure']
        
        # Total pressure should be reasonable
        assert 0 <= features['total_pressure_on_favorite'] <= 1
    
    def test_surface_adaptation_features(self, feature_engineer, sample_player_data, sample_match_context):
        """Test surface adaptation features"""
        player1_data = sample_player_data['player1']
        player2_data = sample_player_data['player2']
        match_date = datetime.now()
        
        features = feature_engineer.create_surface_adaptation_features(
            player1_data, player2_data, sample_match_context, match_date
        )
        
        # Check expected features
        expected_features = [
            'favorite_surface_adaptation', 'underdog_surface_adaptation', 'surface_adaptation_advantage',
            'favorite_surface_expertise', 'underdog_surface_expertise', 'surface_expertise_gap',
            'favorite_surface_transition', 'underdog_surface_transition', 'surface_transition_advantage'
        ]
        
        for feature in expected_features:
            assert feature in features, f"Missing surface feature: {feature}"
        
        # Check adaptation scores (0-1 range)
        assert 0 <= features['favorite_surface_adaptation'] <= 1
        assert 0 <= features['underdog_surface_adaptation'] <= 1
        
        # Check expertise scores (win percentages)
        assert features['favorite_surface_expertise'] == 0.65
        assert features['underdog_surface_expertise'] == 0.55
    
    def test_contextual_features(self, feature_engineer, sample_player_data, sample_match_context):
        """Test contextual features"""
        h2h_data = {
            'overall': {'player1_wins': 2, 'player2_wins': 1},
            'recent_3': {'player1_wins': 2, 'player2_wins': 1}
        }
        
        features = feature_engineer.create_contextual_features(
            sample_player_data['player1'], sample_player_data['player2'], 
            sample_match_context, h2h_data
        )
        
        # Check expected features
        expected_features = [
            'h2h_underdog_winrate', 'h2h_total_matches', 'recent_h2h_underdog_winrate',
            'is_indoor', 'indoor_advantage', 'altitude_factor', 'altitude_fitness_advantage'
        ]
        
        for feature in expected_features:
            assert feature in features, f"Missing contextual feature: {feature}"
        
        # Check H2H calculations
        assert features['h2h_underdog_winrate'] == 1/3  # 1 win out of 3 matches
        assert features['recent_h2h_underdog_winrate'] == 1/3
        
        # Check indoor flag
        assert features['is_indoor'] == 1.0
        
        # Check altitude (normalized)
        assert 0 <= features['altitude_factor'] <= 1
    
    def test_complete_feature_creation(self, feature_engineer, sample_first_set_stats, 
                                     sample_match_context, sample_player_data):
        """Test complete feature creation pipeline"""
        match_data = {
            'player1': sample_player_data['player1'],
            'player2': sample_player_data['player2'],
            'first_set_stats': sample_first_set_stats,
            'match_context': sample_match_context,
            'h2h_data': {
                'overall': {'player1_wins': 1, 'player2_wins': 1},
                'recent_3': {'player1_wins': 1, 'player2_wins': 1}
            },
            'match_date': datetime.now()
        }
        
        features = feature_engineer.create_all_enhanced_features(match_data)
        
        # Should have features from all categories
        assert len(features) > 50, "Should have many enhanced features"
        
        # Check feature categories are present
        momentum_features = [f for f in features.keys() if f.startswith('momentum_')]
        fatigue_features = [f for f in features.keys() if f.startswith('fatigue_')]
        pressure_features = [f for f in features.keys() if f.startswith('pressure_')]
        surface_features = [f for f in features.keys() if f.startswith('surface_')]
        context_features = [f for f in features.keys() if f.startswith('context_')]
        
        assert len(momentum_features) > 0, "Should have momentum features"
        assert len(fatigue_features) > 0, "Should have fatigue features"
        assert len(pressure_features) > 0, "Should have pressure features"
        assert len(surface_features) > 0, "Should have surface features"
        assert len(context_features) > 0, "Should have contextual features"
    
    def test_feature_validation(self, feature_engineer):
        """Test feature validation and cleaning"""
        # Test features with invalid values
        features = {
            'valid_feature': 0.5,
            'nan_feature': np.nan,
            'inf_feature': np.inf,
            'negative_inf_feature': -np.inf,
            'extreme_positive': 100.0,
            'extreme_negative': -100.0,
            'none_feature': None
        }
        
        cleaned_features = feature_engineer.validate_features(features)
        
        # Check that invalid values are handled
        assert cleaned_features['valid_feature'] == 0.5
        assert cleaned_features['nan_feature'] == 0.0
        assert cleaned_features['inf_feature'] == 1.0
        assert cleaned_features['negative_inf_feature'] == 0.0
        assert cleaned_features['none_feature'] == 0.0
        
        # Check that extreme values are clipped
        assert -5.0 <= cleaned_features['extreme_positive'] <= 5.0
        assert -5.0 <= cleaned_features['extreme_negative'] <= 5.0
    
    def test_missing_data_handling(self, feature_engineer):
        """Test handling of missing or incomplete data"""
        # Test with minimal data
        match_data = {
            'player1': {'ranking': 50},
            'player2': {'ranking': 100},
            'first_set_stats': None,  # Missing
            'match_context': None,    # Missing
        }
        
        features = feature_engineer.create_all_enhanced_features(match_data)
        
        # Should return empty dict when required data is missing
        assert len(features) == 0

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])