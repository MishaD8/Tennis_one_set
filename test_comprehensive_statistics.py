#!/usr/bin/env python3
"""
Test script for comprehensive statistics system
"""

import sys
import os
sys.path.append('/home/apps/Tennis_one_set')

from src.api.comprehensive_statistics_service import ComprehensiveStatisticsService
from datetime import datetime, timedelta
import json

def test_comprehensive_statistics():
    print("🧪 TESTING COMPREHENSIVE STATISTICS SYSTEM")
    print("=" * 60)
    
    # Initialize service
    stats_service = ComprehensiveStatisticsService()
    
    # Sample match data with comprehensive information
    sample_matches = [
        {
            'match_id': 'us_open_2025_001',
            'player1_name': 'Carlos Alcaraz',
            'player1_rank': 2,
            'player2_name': 'Novak Djokovic',
            'player2_rank': 1,
            'tournament': 'US Open 2025',
            'surface': 'Hard',
            'round_name': 'Semifinals',
            'match_date': (datetime.now() - timedelta(days=2)).isoformat(),
            'winner': 'Carlos Alcaraz',
            'match_score': '6-3, 2-6, 7-6(4), 6-2',
            'sets_won_p1': 3,
            'sets_won_p2': 1,
            'betting_ratios': {
                'start_set2': {
                    'odds_p1': 2.8, 'odds_p2': 1.4,
                    'ratio_p1': 0.35, 'ratio_p2': 0.65,
                    'current_set': 2, 'sets_score': '1-1',
                    'market_volume': 150000
                },
                'end_set2': {
                    'odds_p1': 2.2, 'odds_p2': 1.6,
                    'ratio_p1': 0.45, 'ratio_p2': 0.55,
                    'current_set': 2, 'sets_score': '1-1',
                    'market_volume': 180000,
                    'odds_movement': 'decreasing'
                }
            },
            'prediction': {
                'predicted_winner': 'Carlos Alcaraz',
                'probability': 0.68,
                'confidence': 'High',
                'model_used': 'ensemble_ml_v2',
                'key_factors': ['Hard court advantage', 'Recent head-to-head', 'Current form']
            },
            'betting_recommendation': 'BET',
            'edge_percentage': 12.5,
            'notes': 'High-profile semifinal match'
        },
        {
            'match_id': 'us_open_2025_002',
            'player1_name': 'Jannik Sinner',
            'player1_rank': 3,
            'player2_name': 'Daniil Medvedev',
            'player2_rank': 4,
            'tournament': 'US Open 2025',
            'surface': 'Hard',
            'round_name': 'Semifinals',
            'match_date': (datetime.now() - timedelta(days=1)).isoformat(),
            'winner': 'Daniil Medvedev',
            'match_score': '3-6, 6-4, 6-3, 7-5',
            'sets_won_p1': 1,
            'sets_won_p2': 3,
            'betting_ratios': {
                'start_set2': {
                    'odds_p1': 1.8, 'odds_p2': 2.0,
                    'ratio_p1': 0.52, 'ratio_p2': 0.48,
                    'current_set': 2, 'sets_score': '0-1',
                    'market_volume': 125000
                },
                'end_set2': {
                    'odds_p1': 2.5, 'odds_p2': 1.5,
                    'ratio_p1': 0.38, 'ratio_p2': 0.62,
                    'current_set': 2, 'sets_score': '1-1',
                    'market_volume': 160000,
                    'odds_movement': 'increasing'
                }
            },
            'prediction': {
                'predicted_winner': 'Jannik Sinner',
                'probability': 0.58,
                'confidence': 'Medium',
                'model_used': 'ensemble_ml_v2',
                'key_factors': ['Youth advantage', 'Surface preference']
            },
            'betting_recommendation': 'SMALL_BET',
            'edge_percentage': 8.2
        },
        {
            'match_id': 'us_open_2025_003',
            'player1_name': 'Ben Shelton',
            'player1_rank': 15,
            'player2_name': 'Alexander Zverev',
            'player2_rank': 5,
            'tournament': 'US Open 2025',
            'surface': 'Hard',
            'round_name': 'Quarterfinals',
            'match_date': (datetime.now() - timedelta(days=3)).isoformat(),
            'winner': 'Ben Shelton',
            'match_score': '6-4, 6-7(3), 6-4, 6-2',
            'sets_won_p1': 3,
            'sets_won_p2': 1,
            'betting_ratios': {
                'start_set2': {
                    'odds_p1': 4.2, 'odds_p2': 1.25,
                    'ratio_p1': 0.22, 'ratio_p2': 0.78,
                    'current_set': 2, 'sets_score': '1-0',
                    'market_volume': 95000
                },
                'end_set2': {
                    'odds_p1': 3.8, 'odds_p2': 1.28,
                    'ratio_p1': 0.25, 'ratio_p2': 0.75,
                    'current_set': 2, 'sets_score': '1-1',
                    'market_volume': 110000,
                    'odds_movement': 'stable'
                }
            },
            'prediction': {
                'predicted_winner': 'Alexander Zverev',
                'probability': 0.75,
                'confidence': 'High',
                'model_used': 'ensemble_ml_v2',
                'key_factors': ['Ranking advantage', 'Experience', 'Consistency']
            },
            'betting_recommendation': 'NO_BET',
            'edge_percentage': -5.2,
            'notes': 'Upset victory by American player'
        }
    ]
    
    print(f"1️⃣ Recording {len(sample_matches)} sample matches...")
    
    for i, match in enumerate(sample_matches, 1):
        match_id = stats_service.record_match_statistics(match)
        if match_id:
            print(f"   ✅ Match {i}: {match_id}")
        else:
            print(f"   ❌ Match {i}: Failed to record")
    
    print("\n2️⃣ Testing comprehensive statistics retrieval...")
    
    # Get comprehensive statistics
    stats = stats_service.get_comprehensive_match_statistics(days_back=7)
    
    if 'error' not in stats:
        summary = stats['summary']
        print(f"   📊 Total matches: {summary['total_matches']}")
        print(f"   ✅ Completed matches: {summary['completed_matches']}")
        print(f"   🎯 Prediction accuracy: {summary['prediction_accuracy']}%")
        print(f"   📈 Upsets occurred: {summary['upsets_occurred']}")
        
        print(f"\n   🏆 Tournaments:")
        for tournament, count in summary['tournaments'].items():
            print(f"      - {tournament}: {count} matches")
        
        print(f"\n   🏟️ Surfaces:")
        for surface, count in summary['surfaces'].items():
            print(f"      - {surface}: {count} matches")
    else:
        print(f"   ❌ Error: {stats['error']}")
    
    print("\n3️⃣ Testing player statistics...")
    
    # Get detailed player stats
    player_stats = stats_service.get_player_detailed_statistics('Carlos Alcaraz')
    
    if 'error' not in player_stats:
        print(f"   👤 Player: {player_stats['name']}")
        print(f"   🏅 Current rank: {player_stats['current_rank']}")
        print(f"   📊 Total matches: {player_stats['match_statistics']['total_matches']}")
        print(f"   🏆 Win rate: {player_stats['match_statistics']['win_percentage']}%")
        print(f"   🎯 Prediction accuracy when predicted to win: {player_stats['prediction_performance']['prediction_accuracy']}%")
    else:
        print(f"   ❌ Error: {player_stats['error']}")
    
    print("\n4️⃣ Testing betting ratio analysis...")
    
    if 'betting_analysis' in stats:
        betting_analysis = stats['betting_analysis']
        analysis_summary = betting_analysis.get('analysis_summary', {})
        
        print(f"   📊 Matches with betting ratios: {analysis_summary.get('total_matches_with_ratios', 0)}")
        print(f"   📈 Significant swings: {analysis_summary.get('matches_with_significant_swings', 0)}")
        print(f"   📉 Average ratio change: {analysis_summary.get('average_ratio_change', 0)}")
        
        correlation = betting_analysis.get('prediction_ratio_correlation', {})
        if isinstance(correlation, dict) and 'agreement_rate' in correlation:
            print(f"   🤝 Prediction-Ratio agreement: {correlation['agreement_rate']}%")
    
    print("\n🧪 Comprehensive statistics system test completed!")
    return True

if __name__ == "__main__":
    test_comprehensive_statistics()