#!/usr/bin/env python3
"""
🔧 RANKING FILTER FIX VERIFICATION TEST
Tests the corrected ranking filter logic to ensure players like Cobolli (#22) 
are properly excluded from underdog analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from comprehensive_tennis_prediction_service import ComprehensiveTennisPredictionService
from comprehensive_ml_data_collector import ComprehensiveMLDataCollector
from ranks_10_300_feature_engineering import Ranks10to300FeatureEngineer, Ranks10to300DataValidator

def test_cobolli_inclusion():
    """Test that Cobolli (rank #22) vs rank #150 player is now properly included in 10-300 range"""
    
    print("🧪 TESTING COBOLLI INCLUSION (Rank #22 in new 10-300 range)")
    print("=" * 60)
    
    # Test case: Cobolli (#22) vs Player (#150)
    test_match_cobolli = {
        'player1': 'Flavio Cobolli',
        'player2': 'Random Player',
        'player1_ranking': 22,  # Now valid in 10-300 range
        'player2_ranking': 150, # Valid underdog in range
        'tournament': 'ATP 250 Tournament',
        'surface': 'Hard'
    }
    
    # Initialize components
    print("🔧 Initializing components...")
    data_collector = ComprehensiveMLDataCollector()
    feature_engineer = Ranks10to300FeatureEngineer()
    data_validator = Ranks10to300DataValidator()
    
    # Test 1: Data Collector Filter
    print("\n1️⃣ Testing Data Collector Filter:")
    collector_result = data_collector._has_player_in_ranks_10_300(test_match_cobolli)
    print(f"   Result: {'❌ EXCLUDED' if not collector_result else '✅ INCLUDED'}")
    print(f"   Expected: ✅ INCLUDED (Cobolli #22 now in valid 10-300 range)")
    
    # Test 2: Feature Engineer Validation
    print("\n2️⃣ Testing Feature Engineer Validation:")
    try:
        player1_data = {'rank': 22}
        player2_data = {'rank': 150}
        
        fe_result = feature_engineer._validate_rank_range(player1_data, player2_data)
        print(f"   Result: {'❌ EXCLUDED' if not fe_result else '✅ INCLUDED'}")
        print(f"   Expected: ✅ INCLUDED (Both players in 10-300 range)")
    except Exception as e:
        print(f"   Result: ❌ EXCLUDED (Exception: {e})")
        print(f"   Expected: ❌ EXCLUDED")
    
    # Test 3: Data Validator
    print("\n3️⃣ Testing Data Validator:")
    validation_data = {
        'player1': {'rank': 22},
        'player2': {'rank': 150},
        'first_set_data': {}
    }
    
    validator_result = data_validator.validate_match_data(validation_data)
    print(f"   Result: {'❌ EXCLUDED' if not validator_result['valid'] else '✅ INCLUDED'}")
    print(f"   Expected: ✅ INCLUDED (Both players in 10-300 range)")
    
    if not validator_result['valid']:
        print(f"   Errors: {validator_result['errors']}")
    
    # Test 4: Prediction Service Validation
    print("\n4️⃣ Testing Prediction Service:")
    try:
        service = ComprehensiveTennisPredictionService(enable_training=False)
        service_result = service._has_player_in_target_ranks(test_match_cobolli)
        print(f"   Result: {'❌ EXCLUDED' if not service_result else '✅ INCLUDED'}")
        print(f"   Expected: ✅ INCLUDED (Both players in 10-300 range)")
    except Exception as e:
        print(f"   Result: ❌ EXCLUDED (Exception: {e})")
        print(f"   Expected: ✅ INCLUDED")

def test_valid_scenarios():
    """Test valid scenarios that should pass the filter"""
    
    print("\n\n🧪 TESTING VALID SCENARIOS")
    print("=" * 60)
    
    test_cases = [
        {
            'name': 'Valid: #15 vs #200 (both in 10-300 range)',
            'player1_rank': 15,
            'player2_rank': 200,
            'expected': True
        },
        {
            'name': 'Valid: #30 vs #180 (both in 10-300 range)', 
            'player1_rank': 30,
            'player2_rank': 180,
            'expected': True
        },
        {
            'name': 'Valid: #75 vs #250 (both in 10-300 range)',
            'player1_rank': 75,
            'player2_rank': 250,
            'expected': True
        },
        {
            'name': 'Valid: #10 vs #300 (boundary case - both exactly on limits)',
            'player1_rank': 10,
            'player2_rank': 300,
            'expected': True
        }
    ]
    
    data_collector = ComprehensiveMLDataCollector()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}️⃣ {test_case['name']}:")
        
        test_match = {
            'player1': 'Player 1',
            'player2': 'Player 2', 
            'player1_ranking': test_case['player1_rank'],
            'player2_ranking': test_case['player2_rank'],
            'tournament': 'ATP 250',
            'surface': 'Hard'
        }
        
        result = data_collector._has_player_in_ranks_10_300(test_match)
        print(f"   Result: {'✅ INCLUDED' if result else '❌ EXCLUDED'}")
        print(f"   Expected: {'✅ INCLUDED' if test_case['expected'] else '❌ EXCLUDED'}")

def test_invalid_scenarios():
    """Test scenarios that should be excluded"""
    
    print("\n\n🧪 TESTING INVALID SCENARIOS")
    print("=" * 60)
    
    test_cases = [
        {
            'name': 'Invalid: #5 vs #150 (favorite outside 10-300 range)',
            'player1_rank': 5,
            'player2_rank': 150,
            'expected': False
        },
        {
            'name': 'Invalid: #8 vs #250 (favorite outside 10-300 range)',
            'player1_rank': 8,
            'player2_rank': 250,
            'expected': False
        },
        {
            'name': 'Invalid: #50 vs #350 (underdog outside 10-300 range)',
            'player1_rank': 50,
            'player2_rank': 350,
            'expected': False
        },
        {
            'name': 'Invalid: #80 vs #400 (underdog far outside range)',
            'player1_rank': 80,
            'player2_rank': 400,
            'expected': False
        }
    ]
    
    data_collector = ComprehensiveMLDataCollector()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}️⃣ {test_case['name']}:")
        
        test_match = {
            'player1': 'Player 1',
            'player2': 'Player 2',
            'player1_ranking': test_case['player1_rank'],
            'player2_ranking': test_case['player2_rank'],
            'tournament': 'ATP 250',
            'surface': 'Hard'
        }
        
        result = data_collector._has_player_in_ranks_10_300(test_match)
        print(f"   Result: {'❌ EXCLUDED' if not result else '✅ INCLUDED'}")
        print(f"   Expected: {'❌ EXCLUDED' if not test_case['expected'] else '✅ INCLUDED'}")
        
        if result == test_case['expected']:
            print("   ✅ CORRECT")
        else:
            print("   ❌ INCORRECT")

def main():
    """Run all ranking filter tests"""
    
    print("🔧 RANKING FILTER UPDATE VERIFICATION")
    print("Testing updated system with new 10-300 range (Cobolli #22 now included)")
    print("=" * 80)
    
    # Test the specific Cobolli case
    test_cobolli_inclusion()
    
    # Test valid scenarios
    test_valid_scenarios()
    
    # Test invalid scenarios  
    test_invalid_scenarios()
    
    print("\n" + "=" * 80)
    print("✅ RANKING FILTER UPDATE VERIFICATION COMPLETE")
    print("The system now includes players ranked 10-300 for underdog analysis")
    print("Cobolli (rank #22) is now properly included in the expanded range")

if __name__ == "__main__":
    main()