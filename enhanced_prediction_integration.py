#!/usr/bin/env python3
"""
🎾 Enhanced Prediction Integration
Integrates enhanced surface and H2H features with existing prediction system
"""

from tennis_prediction_module import TennisPredictionService, create_match_data
from enhanced_surface_features import (
    init_feature_engineer, get_feature_engineer, 
    generate_enhanced_match_features
)
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class EnhancedTennisPredictionService(TennisPredictionService):
    """
    Enhanced prediction service with advanced surface and H2H features
    Extends the base TennisPredictionService with enhanced feature engineering
    """
    
    def __init__(self, models_dir="tennis_models", use_adaptive_weights=True, 
                 use_enhanced_features=True):
        super().__init__(models_dir, use_adaptive_weights)
        self.use_enhanced_features = use_enhanced_features
        
        # Initialize enhanced feature engineer
        if self.use_enhanced_features:
            try:
                if not get_feature_engineer():
                    init_feature_engineer()
                self.feature_engineer = get_feature_engineer()
                print("🎾 Enhanced feature engineering enabled")
            except Exception as e:
                print(f"⚠️ Could not initialize enhanced features: {e}")
                self.use_enhanced_features = False
                self.feature_engineer = None
        else:
            self.feature_engineer = None
    
    def predict_match_enhanced(self, match_data: Dict, return_details=True) -> Dict:
        """
        Enhanced prediction with surface-specific and H2H features
        
        Args:
            match_data: Match data with player names and basic info
            return_details: Whether to return detailed analysis
        
        Returns:
            Enhanced prediction result with feature breakdown
        """
        
        # Generate enhanced features
        if self.use_enhanced_features and self.feature_engineer:
            try:
                enhanced_data = self.feature_engineer.generate_enhanced_features(match_data)
                
                # Map enhanced features to expected model features
                model_features = self._map_to_model_features(enhanced_data)
                
                # Get prediction using enhanced features
                result = self.predict_match(model_features, return_details)
                
                # Add enhanced feature analysis
                if return_details:
                    result['enhanced_features'] = self._analyze_enhanced_features(enhanced_data)
                    result['feature_engineering'] = 'enhanced'
                    result['total_features_used'] = len(enhanced_data)
                
                return result
                
            except Exception as e:
                logger.warning(f"Enhanced features failed, falling back to basic: {e}")
        
        # Fallback to basic prediction
        basic_result = self.predict_match(match_data, return_details)
        if return_details:
            basic_result['feature_engineering'] = 'basic'
        return basic_result
    
    def _map_to_model_features(self, enhanced_data: Dict) -> Dict:
        """Map enhanced features to expected model feature format"""
        # Start with enhanced data
        model_features = enhanced_data.copy()
        
        # Ensure all required basic features are present
        required_features = [
            'player_rank', 'player_age', 'opponent_rank', 'opponent_age',
            'player_recent_win_rate', 'player_form_trend', 
            'player_surface_advantage', 'h2h_win_rate', 'total_pressure'
        ]
        
        # Map enhanced features to basic feature names if needed
        feature_mapping = {
            'h2h_overall_winrate': 'h2h_win_rate',
            'player_surface_winrate_12m': 'player_surface_win_rate',
            'player_surface_advantage_vs_hard': 'player_surface_advantage'
        }
        
        for enhanced_name, basic_name in feature_mapping.items():
            if enhanced_name in model_features and basic_name not in model_features:
                model_features[basic_name] = model_features[enhanced_name]
        
        # Fill missing required features with defaults
        defaults = {
            'player_rank': enhanced_data.get('player_rank', 50),
            'player_age': 26,
            'opponent_rank': enhanced_data.get('opponent_rank', 50),
            'opponent_age': 26,
            'player_recent_win_rate': enhanced_data.get('player_recent_win_rate', 0.6),
            'player_form_trend': 0.0,
            'total_pressure': enhanced_data.get('total_pressure', 2.5)
        }
        
        for feature, default_value in defaults.items():
            if feature not in model_features:
                model_features[feature] = default_value
        
        return model_features
    
    def _analyze_enhanced_features(self, enhanced_data: Dict) -> Dict:
        """Analyze and categorize enhanced features for interpretation"""
        analysis = {
            'surface_analysis': {},
            'h2h_analysis': {},
            'form_analysis': {},
            'key_insights': []
        }
        
        # Surface analysis
        surface_features = {k: v for k, v in enhanced_data.items() if 'surface' in k}
        analysis['surface_analysis'] = surface_features
        
        # Identify surface advantages/disadvantages
        surface_adv = enhanced_data.get('player_surface_advantage_vs_hard', 0)
        if surface_adv > 0.1:
            analysis['key_insights'].append(f"Strong surface advantage (+{surface_adv:.1%})")
        elif surface_adv < -0.1:
            analysis['key_insights'].append(f"Surface disadvantage ({surface_adv:.1%})")
        
        # H2H analysis
        h2h_features = {k: v for k, v in enhanced_data.items() if 'h2h' in k}
        analysis['h2h_analysis'] = h2h_features
        
        # H2H insights
        h2h_overall = enhanced_data.get('h2h_overall_winrate', 0.5)
        h2h_surface = enhanced_data.get('h2h_surface_winrate', 0.5)
        
        if h2h_overall > 0.7:
            analysis['key_insights'].append(f"Dominates H2H overall ({h2h_overall:.1%})")
        elif h2h_overall < 0.3:
            analysis['key_insights'].append(f"Poor H2H record ({h2h_overall:.1%})")
        
        if abs(h2h_surface - h2h_overall) > 0.2:
            if h2h_surface > h2h_overall:
                analysis['key_insights'].append("H2H advantage stronger on this surface")
            else:
                analysis['key_insights'].append("H2H advantage weaker on this surface")
        
        # Form analysis
        recent_form = enhanced_data.get('player_recent_win_rate', 0.5)
        surface_form = enhanced_data.get('player_surface_winrate_3m', 0.5)
        adaptation = enhanced_data.get('player_surface_adaptation', 0)
        
        analysis['form_analysis'] = {
            'recent_overall': recent_form,
            'recent_surface': surface_form,
            'surface_adaptation': adaptation
        }
        
        if adaptation > 0.1:
            analysis['key_insights'].append("Improving on this surface")
        elif adaptation < -0.1:
            analysis['key_insights'].append("Declining form on this surface")
        
        return analysis
    
    def record_enhanced_match_result(self, match_data: Dict, prediction_result: Dict, 
                                   actual_result: float, detailed_result: Dict = None):
        """
        Record match result for both adaptive weights and enhanced features
        
        Args:
            match_data: Original match data
            prediction_result: Result from predict_match_enhanced
            actual_result: Actual match outcome (0 or 1)
            detailed_result: Detailed match result (score, sets, etc.)
        """
        
        # Record for adaptive weights
        success = self.record_match_result(match_data, prediction_result, actual_result)
        
        # Record for enhanced features
        if self.use_enhanced_features and self.feature_engineer and detailed_result:
            try:
                # Prepare result data for feature engineer
                result_data = {
                    'date': detailed_result.get('date', ''),
                    'won': actual_result == 1,
                    'opponent': match_data.get('opponent', ''),
                    'surface': match_data.get('surface', 'hard'),
                    'tournament': detailed_result.get('tournament', ''),
                    'score': detailed_result.get('score', ''),
                    'sets': detailed_result.get('sets', '')
                }
                
                self.feature_engineer.add_match_result(match_data, result_data)
                print("📊 Enhanced features updated with match result")
                
            except Exception as e:
                logger.warning(f"Could not update enhanced features: {e}")
        
        return success
    
    def get_enhanced_model_info(self) -> Dict:
        """Get comprehensive model information including enhanced features"""
        info = self.get_model_info()
        
        info.update({
            'enhanced_features_enabled': self.use_enhanced_features,
            'feature_engineer_available': self.feature_engineer is not None
        })
        
        if self.use_enhanced_features:
            info['enhanced_feature_categories'] = [
                'surface_performance_tracking',
                'head_to_head_analysis', 
                'surface_transition_metrics',
                'performance_momentum',
                'advanced_interactions'
            ]
        
        return info

def create_enhanced_match_data(player: str, opponent: str, surface: str = 'hard',
                              player_rank: int = 50, opponent_rank: int = 50,
                              tournament: str = '', **kwargs) -> Dict:
    """
    Create enhanced match data with all necessary fields
    
    Args:
        player: Player name
        opponent: Opponent name  
        surface: Court surface (hard, clay, grass)
        player_rank: Player ATP ranking
        opponent_rank: Opponent ATP ranking
        tournament: Tournament name
        **kwargs: Additional match data
    
    Returns:
        Complete match data dict for enhanced prediction
    """
    
    match_data = {
        'player': player,
        'opponent': opponent,
        'surface': surface.lower(),
        'player_rank': float(player_rank),
        'opponent_rank': float(opponent_rank),
        'tournament': tournament,
        'player_age': kwargs.get('player_age', 26),
        'opponent_age': kwargs.get('opponent_age', 26),
        'player_recent_win_rate': kwargs.get('player_recent_win_rate', 0.7),
        'player_form_trend': kwargs.get('player_form_trend', 0.0),
        'total_pressure': kwargs.get('total_pressure', 2.5)
    }
    
    # Add any additional provided data
    for key, value in kwargs.items():
        if key not in match_data:
            match_data[key] = value
    
    return match_data

def test_enhanced_prediction_system():
    """Comprehensive test of the enhanced prediction system"""
    print("🎾 Testing Enhanced Tennis Prediction System")
    print("=" * 60)
    
    # Initialize enhanced service
    print("1. Initializing enhanced prediction service...")
    service = EnhancedTennisPredictionService(
        use_adaptive_weights=True,
        use_enhanced_features=True
    )
    
    info = service.get_enhanced_model_info()
    print(f"   Enhanced features: {info['enhanced_features_enabled']}")
    print(f"   Adaptive weights: {info.get('adaptive_weights_enabled', 'Unknown')}")
    
    # Test enhanced predictions
    print("\n2. Testing enhanced predictions...")
    
    test_matches = [
        {
            'name': 'Clay Court Specialist vs Hard Court Player',
            'data': create_enhanced_match_data(
                player='Rafael Nadal',
                opponent='John Isner', 
                surface='clay',
                player_rank=3,
                opponent_rank=15,
                player_recent_win_rate=0.85,
                tournament='French Open'
            )
        },
        {
            'name': 'Grass Court Matchup',
            'data': create_enhanced_match_data(
                player='Roger Federer',
                opponent='Novak Djokovic',
                surface='grass', 
                player_rank=8,
                opponent_rank=1,
                player_recent_win_rate=0.75,
                tournament='Wimbledon'
            )
        }
    ]
    
    for i, test_match in enumerate(test_matches, 1):
        print(f"\n   Test {i}: {test_match['name']}")
        
        try:
            result = service.predict_match_enhanced(test_match['data'], return_details=True)
            
            print(f"     Probability: {result['probability']:.3f}")
            print(f"     Confidence: {result['confidence']}")
            print(f"     Features used: {result.get('feature_engineering', 'basic')}")
            
            if 'enhanced_features' in result:
                enhanced = result['enhanced_features']
                
                # Show key insights
                insights = enhanced.get('key_insights', [])
                if insights:
                    print("     Key insights:")
                    for insight in insights[:3]:  # Show top 3
                        print(f"       • {insight}")
                
                # Show surface analysis
                surface_analysis = enhanced.get('surface_analysis', {})
                surface_advantage = surface_analysis.get('player_surface_advantage_vs_hard', 0)
                if abs(surface_advantage) > 0.05:
                    print(f"     Surface factor: {surface_advantage:+.1%}")
            
        except Exception as e:
            print(f"     Error: {e}")
    
    print(f"\n✅ Enhanced prediction system test completed!")

if __name__ == "__main__":
    test_enhanced_prediction_system()