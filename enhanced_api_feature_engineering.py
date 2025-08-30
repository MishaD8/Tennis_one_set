#!/usr/bin/env python3
"""
Enhanced API Feature Engineering for Tennis Prediction System
Specialized feature engineering using comprehensive api-tennis.com data
Optimized for ranks 10-300 underdog second-set predictions
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedAPIFeatureEngineer:
    """
    Feature engineering specifically designed for enhanced API data
    Converts comprehensive api-tennis.com data into ML-ready features
    """
    
    def __init__(self):
        # Feature mapping configurations
        self.surface_encodings = {
            'hard': [1.0, 0.0, 0.0],
            'clay': [0.0, 1.0, 0.0], 
            'grass': [0.0, 0.0, 1.0],
            'carpet': [0.0, 0.0, 0.0]  # Rare surface
        }
        
        self.movement_encodings = {
            'up': 1.0,
            'same': 0.0,
            'down': -1.0,
            'new': 0.5,  # New to rankings
            '': 0.0  # No movement data
        }
        
        # Country groupings for surface advantages
        self.clay_specialists = [
            'ESP', 'FRA', 'ITA', 'ARG', 'SRB', 'POR', 'BRA', 'CHI', 'URU', 'COL',
            'AUT', 'CRO', 'SLO', 'MNE', 'BEL', 'ECU', 'PER'
        ]
        
        self.grass_specialists = [
            'GBR', 'AUS', 'NZL', 'IRL', 'RSA'
        ]
        
        self.hard_court_specialists = [
            'USA', 'CAN', 'RUS', 'UKR', 'BLR', 'KAZ', 'JPN', 'KOR', 'CHN'
        ]
        
        # Tournament tier mappings
        self.tournament_tiers = {
            'grand_slam': 5.0,
            'masters_1000': 4.0,
            'atp_500': 3.0,
            'atp_250': 2.0,
            'challenger': 1.0,
            'wta_1000': 4.0,
            'wta_500': 3.0,
            'wta_250': 2.0
        }
    
    def create_comprehensive_features(self, match_data: Dict[str, Any]) -> np.ndarray:
        """
        Create comprehensive feature vector from enhanced API match data
        
        Args:
            match_data: Enhanced match data from api-tennis.com integration
            
        Returns:
            Numpy array of engineered features
        """
        try:
            features = []
            
            # Extract player data from enhanced API structure
            player1_data = self._extract_player_data(match_data, 'player1')
            player2_data = self._extract_player_data(match_data, 'player2')
            
            # Core ranking features (most important for underdog detection)
            features.extend(self._create_ranking_features(player1_data, player2_data))
            
            # Enhanced points-based features (from paid API tier)
            features.extend(self._create_points_features(player1_data, player2_data))
            
            # Movement and form features
            features.extend(self._create_movement_features(player1_data, player2_data))
            
            # Tournament context features
            features.extend(self._create_tournament_features(match_data))
            
            # Surface and geographic features
            features.extend(self._create_surface_features(match_data, player1_data, player2_data))
            
            # Competition and pressure features
            features.extend(self._create_competition_features(match_data, player1_data, player2_data))
            
            # Data quality and confidence features
            features.extend(self._create_data_quality_features(match_data))
            
            # Underdog-specific features (key for our use case)
            features.extend(self._create_underdog_features(player1_data, player2_data, match_data))
            
            # Convert to numpy array
            feature_array = np.array(features, dtype=np.float32)
            
            # Ensure no NaN or infinite values
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1.0, neginf=-1.0)
            
            logger.info(f"✅ Generated {len(feature_array)} enhanced API features")
            return feature_array.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"❌ Enhanced API feature creation failed: {e}")
            # Return minimal features as fallback
            return self._create_fallback_features(match_data)
    
    def _extract_player_data(self, match_data: Dict[str, Any], player_key: str) -> Dict[str, Any]:
        """Extract and normalize player data from enhanced API structure"""
        player_data = {}
        
        # From enhanced API structure
        if player_key in match_data:
            api_player_data = match_data[player_key]
            player_data.update({
                'name': api_player_data.get('name', ''),
                'rank': api_player_data.get('ranking', 150),
                'points': api_player_data.get('points', 1000),
                'country': api_player_data.get('country', ''),
                'movement': api_player_data.get('ranking_movement', 'same'),
                'tour': api_player_data.get('tour', 'ATP')
            })
        
        # Fallback from direct match keys
        if player_key == 'player1':
            player_data.update({
                'name': match_data.get('event_first_player', ''),
                'rank': match_data.get('player1_rank', 150),
                'points': match_data.get('player1_points', 1000),
                'country': match_data.get('player1_country', ''),
                'movement': match_data.get('player1_movement', 'same')
            })
        elif player_key == 'player2':
            player_data.update({
                'name': match_data.get('event_second_player', ''),
                'rank': match_data.get('player2_rank', 150),
                'points': match_data.get('player2_points', 1000),
                'country': match_data.get('player2_country', ''),
                'movement': match_data.get('player2_movement', 'same')
            })
        
        return player_data
    
    def _create_ranking_features(self, player1_data: Dict, player2_data: Dict) -> List[float]:
        """Create ranking-based features"""
        p1_rank = player1_data.get('rank', 150)
        p2_rank = player2_data.get('rank', 150)
        
        features = [
            float(p1_rank),
            float(p2_rank),
            float(abs(p1_rank - p2_rank)),  # Ranking gap
            float(min(p1_rank, p2_rank)),   # Favorite rank
            float(max(p1_rank, p2_rank)),   # Underdog rank
        ]
        
        # Ranking tier indicators
        features.extend([
            1.0 if min(p1_rank, p2_rank) <= 10 else 0.0,   # Top 10 favorite
            1.0 if min(p1_rank, p2_rank) <= 20 else 0.0,   # Top 20 favorite
            1.0 if max(p1_rank, p2_rank) <= 50 else 0.0,   # Top 50 underdog
            1.0 if max(p1_rank, p2_rank) <= 100 else 0.0,  # Top 100 underdog
            1.0 if max(p1_rank, p2_rank) >= 200 else 0.0,  # Deep underdog (200+)
        ])
        
        return features
    
    def _create_points_features(self, player1_data: Dict, player2_data: Dict) -> List[float]:
        """Create ATP/WTA points-based features (enhanced API exclusive)"""
        p1_points = player1_data.get('points', 1000)
        p2_points = player2_data.get('points', 1000)
        
        if p1_points and p2_points and p2_points > 0:
            points_ratio = p1_points / p2_points
            points_gap = abs(p1_points - p2_points)
            points_gap_normalized = points_gap / max(p1_points, p2_points)
        else:
            points_ratio = 1.0
            points_gap = 0.0
            points_gap_normalized = 0.0
        
        features = [
            min(max(points_ratio, 0.1), 10.0),  # Clamped points ratio
            min(points_gap_normalized, 1.0),    # Normalized points gap
            float(p1_points / 1000),            # Player 1 points (in thousands)
            float(p2_points / 1000),            # Player 2 points (in thousands)
        ]
        
        # Points competitiveness indicators
        features.extend([
            1.0 if 0.8 <= points_ratio <= 1.25 else 0.0,  # Very competitive points
            1.0 if 0.6 <= points_ratio <= 1.67 else 0.0,  # Moderately competitive
            1.0 if points_gap < 500 else 0.0,              # Close points gap
            1.0 if points_gap > 2000 else 0.0,             # Large points gap
        ])
        
        return features
    
    def _create_movement_features(self, player1_data: Dict, player2_data: Dict) -> List[float]:
        """Create ranking movement and form features"""
        p1_movement = player1_data.get('movement', 'same')
        p2_movement = player2_data.get('movement', 'same')
        
        p1_movement_score = self.movement_encodings.get(p1_movement.lower(), 0.0)
        p2_movement_score = self.movement_encodings.get(p2_movement.lower(), 0.0)
        
        features = [
            p1_movement_score,
            p2_movement_score,
            p1_movement_score - p2_movement_score,  # Movement differential
        ]
        
        # Form combination indicators
        features.extend([
            1.0 if p1_movement == 'up' and p2_movement == 'down' else 0.0,     # Perfect underdog form
            1.0 if p1_movement == 'down' and p2_movement == 'up' else 0.0,     # Perfect favorite form
            1.0 if p1_movement == 'up' else 0.0,                               # Player 1 rising
            1.0 if p2_movement == 'up' else 0.0,                               # Player 2 rising
            1.0 if p1_movement == 'down' else 0.0,                             # Player 1 declining
            1.0 if p2_movement == 'down' else 0.0,                             # Player 2 declining
        ])
        
        return features
    
    def _create_tournament_features(self, match_data: Dict[str, Any]) -> List[float]:
        """Create tournament context features"""
        tournament_name = match_data.get('tournament_name', '').lower()
        
        # Tournament tier detection
        if any(slam in tournament_name for slam in ['us open', 'wimbledon', 'french', 'australian']):
            tier = self.tournament_tiers['grand_slam']
            is_grand_slam = 1.0
        elif 'masters' in tournament_name or '1000' in tournament_name:
            tier = self.tournament_tiers['masters_1000']
            is_grand_slam = 0.0
        elif '500' in tournament_name:
            tier = self.tournament_tiers['atp_500']
            is_grand_slam = 0.0
        elif '250' in tournament_name or 'wta' in tournament_name:
            tier = self.tournament_tiers['atp_250']
            is_grand_slam = 0.0
        else:
            tier = 2.5  # Default tier
            is_grand_slam = 0.0
        
        # Tournament info from enhanced data
        tournament_info = match_data.get('tournament_info', {})
        if tournament_info.get('is_grand_slam'):
            is_grand_slam = 1.0
            tier = self.tournament_tiers['grand_slam']
        
        features = [
            tier / 5.0,      # Normalized tournament importance
            is_grand_slam,   # Grand Slam indicator
        ]
        
        # Tournament tier indicators
        features.extend([
            1.0 if tier >= 4.5 else 0.0,  # Grand Slam
            1.0 if 3.5 <= tier < 4.5 else 0.0,  # Masters/WTA 1000
            1.0 if 2.5 <= tier < 3.5 else 0.0,  # 500 level
            1.0 if tier < 2.5 else 0.0,   # 250 level or lower
        ])
        
        return features
    
    def _create_surface_features(self, match_data: Dict[str, Any], 
                               player1_data: Dict, player2_data: Dict) -> List[float]:
        """Create surface and geographic advantage features"""
        surface = match_data.get('surface', 'Hard').lower()
        p1_country = player1_data.get('country', '')
        p2_country = player2_data.get('country', '')
        
        # Surface encoding
        surface_features = self.surface_encodings.get(surface, [0.5, 0.5, 0.0])
        features = surface_features.copy()
        
        # Surface specialization advantages
        clay_advantage = 0.0
        grass_advantage = 0.0
        hard_advantage = 0.0
        
        if surface == 'clay':
            if p1_country in self.clay_specialists:
                clay_advantage += 0.1
            if p2_country in self.clay_specialists:
                clay_advantage -= 0.1
        
        elif surface == 'grass':
            if p1_country in self.grass_specialists:
                grass_advantage += 0.1
            if p2_country in self.grass_specialists:
                grass_advantage -= 0.1
        
        elif surface == 'hard':
            if p1_country in self.hard_court_specialists:
                hard_advantage += 0.05
            if p2_country in self.hard_court_specialists:
                hard_advantage -= 0.05
        
        features.extend([
            clay_advantage,
            grass_advantage, 
            hard_advantage,
        ])
        
        # Geographic features
        features.extend([
            1.0 if p1_country in self.clay_specialists else 0.0,
            1.0 if p2_country in self.clay_specialists else 0.0,
            1.0 if p1_country == p2_country else 0.0,  # Same nationality
            1.0 if p1_country in ['ESP', 'ARG', 'FRA'] else 0.0,  # Traditional clay powers
            1.0 if p2_country in ['ESP', 'ARG', 'FRA'] else 0.0,
        ])
        
        return features
    
    def _create_competition_features(self, match_data: Dict[str, Any], 
                                   player1_data: Dict, player2_data: Dict) -> List[float]:
        """Create competition-level and pressure features"""
        event_type = match_data.get('event_type_type', '').lower()
        
        # Tour detection
        is_atp = 1.0 if 'atp' in event_type else 0.0
        is_wta = 1.0 if 'wta' in event_type else 0.0
        
        # Competition level from player rankings
        p1_rank = player1_data.get('rank', 150)
        p2_rank = player2_data.get('rank', 150)
        avg_rank = (p1_rank + p2_rank) / 2
        
        competition_level = min(1.0, (400 - avg_rank) / 400)  # Normalized competition level
        
        features = [
            is_atp,
            is_wta,
            competition_level,
            1.0 if avg_rank <= 50 else 0.0,   # Elite level competition
            1.0 if avg_rank <= 100 else 0.0,  # High level competition
            1.0 if avg_rank <= 200 else 0.0,  # Professional level
        ]
        
        return features
    
    def _create_data_quality_features(self, match_data: Dict[str, Any]) -> List[float]:
        """Create data quality and confidence features"""
        # Enhanced data quality from API
        enhanced_data = match_data.get('enhanced_data', {})
        data_quality = enhanced_data.get('data_quality_score', 0.7)
        
        # Data source quality
        data_source_quality = 1.0 if match_data.get('data_source') == 'enhanced_api' else 0.7
        
        # Completeness indicators
        has_points = 1.0 if match_data.get('player1_points') and match_data.get('player2_points') else 0.0
        has_movement = 1.0 if match_data.get('player1_movement') and match_data.get('player2_movement') else 0.0
        has_countries = 1.0 if match_data.get('player1_country') and match_data.get('player2_country') else 0.0
        
        features = [
            data_quality,
            data_source_quality,
            has_points,
            has_movement,
            has_countries,
            (has_points + has_movement + has_countries) / 3,  # Overall completeness
        ]
        
        return features
    
    def _create_underdog_features(self, player1_data: Dict, player2_data: Dict, 
                                match_data: Dict[str, Any]) -> List[float]:
        """Create underdog-specific features (core of our betting strategy)"""
        p1_rank = player1_data.get('rank', 150)
        p2_rank = player2_data.get('rank', 150)
        
        # Identify underdog and favorite
        if p1_rank > p2_rank:
            underdog_rank = p1_rank
            favorite_rank = p2_rank
            underdog_data = player1_data
            favorite_data = player2_data
            underdog_is_p1 = True
        else:
            underdog_rank = p2_rank
            favorite_rank = p1_rank
            underdog_data = player2_data
            favorite_data = player1_data
            underdog_is_p1 = False
        
        ranking_gap = underdog_rank - favorite_rank
        
        # Core underdog features
        features = [
            1.0 if underdog_is_p1 else 0.0,           # Underdog identity
            float(underdog_rank),                      # Underdog absolute rank
            float(favorite_rank),                      # Favorite absolute rank  
            float(ranking_gap),                        # Absolute ranking gap
            min(ranking_gap / 100.0, 2.0),           # Normalized ranking gap
        ]
        
        # Underdog tier analysis (key for our 10-300 range strategy)
        features.extend([
            1.0 if 10 <= underdog_rank <= 50 else 0.0,    # Quality underdog (10-50)
            1.0 if 51 <= underdog_rank <= 100 else 0.0,   # Solid underdog (51-100)
            1.0 if 101 <= underdog_rank <= 200 else 0.0,  # Standard underdog (101-200)
            1.0 if 201 <= underdog_rank <= 300 else 0.0,  # Deep underdog (201-300)
            1.0 if underdog_rank > 300 else 0.0,           # Extreme underdog (300+)
        ])
        
        # Gap analysis for underdog scenarios
        features.extend([
            1.0 if 5 <= ranking_gap <= 20 else 0.0,       # Close gap (dangerous underdog)
            1.0 if 21 <= ranking_gap <= 50 else 0.0,      # Moderate gap
            1.0 if 51 <= ranking_gap <= 100 else 0.0,     # Large gap
            1.0 if ranking_gap > 100 else 0.0,            # Huge gap
        ])
        
        # Underdog form analysis
        underdog_movement = underdog_data.get('movement', 'same').lower()
        favorite_movement = favorite_data.get('movement', 'same').lower()
        
        features.extend([
            1.0 if underdog_movement == 'up' else 0.0,              # Rising underdog
            1.0 if favorite_movement == 'down' else 0.0,            # Declining favorite
            1.0 if underdog_movement == 'up' and favorite_movement == 'down' else 0.0,  # Perfect scenario
        ])
        
        # Points-based underdog analysis (if available)
        underdog_points = underdog_data.get('points', 0)
        favorite_points = favorite_data.get('points', 0)
        
        if underdog_points and favorite_points and favorite_points > 0:
            points_competitiveness = underdog_points / favorite_points
            features.extend([
                points_competitiveness,
                1.0 if points_competitiveness > 0.8 else 0.0,  # Very competitive on points
                1.0 if points_competitiveness > 0.6 else 0.0,  # Moderately competitive
            ])
        else:
            features.extend([0.5, 0.0, 0.0])  # No points data
        
        return features
    
    def _create_fallback_features(self, match_data: Dict[str, Any]) -> np.ndarray:
        """Create minimal features when enhanced processing fails"""
        try:
            p1_rank = match_data.get('player1_rank', 150)
            p2_rank = match_data.get('player2_rank', 150)
            
            basic_features = [
                float(p1_rank),
                float(p2_rank),
                float(abs(p1_rank - p2_rank)),
                float(min(p1_rank, p2_rank)),
                float(max(p1_rank, p2_rank)),
                1.0,  # Hard surface default
                0.0,  # Clay surface
                0.0,  # Grass surface
                2.5,  # Tournament importance default
            ]
            
            # Pad to at least 20 features
            while len(basic_features) < 20:
                basic_features.append(0.0)
            
            return np.array(basic_features, dtype=np.float32).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"❌ Even fallback feature creation failed: {e}")
            return np.zeros((1, 10), dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get descriptive names for all features"""
        feature_names = [
            # Ranking features
            'player1_rank', 'player2_rank', 'ranking_gap', 'favorite_rank', 'underdog_rank',
            'top10_favorite', 'top20_favorite', 'top50_underdog', 'top100_underdog', 'deep_underdog_200plus',
            
            # Points features
            'points_ratio', 'points_gap_normalized', 'player1_points_k', 'player2_points_k',
            'very_competitive_points', 'moderately_competitive_points', 'close_points_gap', 'large_points_gap',
            
            # Movement features  
            'player1_movement', 'player2_movement', 'movement_differential',
            'perfect_underdog_form', 'perfect_favorite_form', 'player1_rising', 'player2_rising',
            'player1_declining', 'player2_declining',
            
            # Tournament features
            'tournament_importance', 'is_grand_slam', 'grand_slam_tier', 'masters_tier', 
            'atp500_tier', 'atp250_tier',
            
            # Surface features
            'hard_surface', 'clay_surface', 'grass_surface', 'clay_advantage', 'grass_advantage',
            'hard_advantage', 'player1_clay_specialist', 'player2_clay_specialist', 'same_nationality',
            'player1_clay_power', 'player2_clay_power',
            
            # Competition features
            'is_atp', 'is_wta', 'competition_level', 'elite_competition', 'high_competition', 
            'professional_competition',
            
            # Data quality features
            'data_quality', 'data_source_quality', 'has_points', 'has_movement', 'has_countries',
            'overall_completeness',
            
            # Underdog features
            'underdog_is_player1', 'underdog_absolute_rank', 'favorite_absolute_rank',
            'absolute_ranking_gap', 'normalized_ranking_gap', 'quality_underdog_10_50',
            'solid_underdog_51_100', 'standard_underdog_101_200', 'deep_underdog_201_300',
            'extreme_underdog_300plus', 'close_gap_5_20', 'moderate_gap_21_50',
            'large_gap_51_100', 'huge_gap_100plus', 'rising_underdog', 'declining_favorite',
            'perfect_underdog_scenario', 'underdog_points_competitiveness',
            'very_competitive_underdog_points', 'moderately_competitive_underdog_points'
        ]
        
        return feature_names


def main():
    """Test the enhanced API feature engineering"""
    engineer = EnhancedAPIFeatureEngineer()
    
    # Test with sample enhanced API data
    sample_match = {
        'data_source': 'enhanced_api',
        'tournament_name': 'US Open',
        'surface': 'Hard',
        'event_type_type': 'ATP Singles',
        'player1': {
            'name': 'Flavio Cobolli',
            'ranking': 32,
            'points': 1180,
            'country': 'ITA',
            'ranking_movement': 'up',
            'tour': 'ATP'
        },
        'player2': {
            'name': 'Novak Djokovic', 
            'ranking': 5,
            'points': 5900,
            'country': 'SRB',
            'ranking_movement': 'down',
            'tour': 'ATP'
        },
        'enhanced_data': {
            'data_quality_score': 0.95,
            'ranking_gap': 27,
            'is_underdog_scenario': True
        },
        'tournament_info': {
            'is_grand_slam': True,
            'surface': 'Hard',
            'location': 'New York'
        }
    }
    
    print("🧪 Testing Enhanced API Feature Engineering")
    print("=" * 60)
    
    features = engineer.create_comprehensive_features(sample_match)
    feature_names = engineer.get_feature_names()
    
    print(f"✅ Generated {features.shape[1]} features")
    print(f"📋 Expected {len(feature_names)} feature names")
    
    # Show first 20 features with names
    print("\n📊 Sample Features:")
    for i, (name, value) in enumerate(zip(feature_names[:20], features[0][:20])):
        print(f"  {i+1:2d}. {name:<25}: {value:.3f}")
    
    print(f"\n... and {features.shape[1] - 20} more features")
    
    # Test underdog detection
    player1_rank = sample_match['player1']['ranking'] 
    player2_rank = sample_match['player2']['ranking']
    underdog_rank = max(player1_rank, player2_rank)
    
    print(f"\n🎯 Underdog Analysis:")
    print(f"   Player 1: {sample_match['player1']['name']} (#{player1_rank})")
    print(f"   Player 2: {sample_match['player2']['name']} (#{player2_rank})")
    print(f"   Underdog: Rank #{underdog_rank} ({'Player 1' if player1_rank > player2_rank else 'Player 2'})")
    print(f"   Valid for strategy: {10 <= underdog_rank <= 300}")


if __name__ == "__main__":
    main()