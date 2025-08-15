#!/usr/bin/env python3
"""
🧪 COMPLETE PIPELINE TEST
Tests the entire tennis prediction pipeline from data collection to ML predictions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from comprehensive_tennis_prediction_service import ComprehensiveTennisPredictionService
import json

def test_pipeline_scenarios():
    """Test various ranking scenarios through the complete pipeline"""
    
    print("🎾 TENNIS PREDICTION PIPELINE TEST")
    print("=" * 80)
    
    # Initialize prediction service
    print("🔧 Initializing Comprehensive Prediction Service...")
    try:
        prediction_service = ComprehensiveTennisPredictionService()
        print("✅ Service initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize service: {e}")
        return
    
    # Test scenarios
    test_cases = [
        {
            "name": "❌ INVALID: Cobolli (#22) vs Player (#150)",
            "data": {
                'player1': 'Flavio Cobolli',
                'player2': 'Random Player',
                'player1_ranking': 22,  # Top player - should be excluded
                'player2_ranking': 150,
                'tournament': 'ATP 250 Vienna',
                'surface': 'Hard',
                'first_set_winner': 'player2',
                'first_set_score': '6-4'
            },
            "expected": "EXCLUDED"
        },
        {
            "name": "✅ VALID: Player (#75) vs Player (#200)",
            "data": {
                'player1': 'Player A',
                'player2': 'Player B', 
                'player1_ranking': 75,
                'player2_ranking': 200,  # Player B is underdog
                'tournament': 'ATP 250 Vienna',
                'surface': 'Hard',
                'first_set_winner': 'player1',
                'first_set_score': '6-4'
            },
            "expected": "INCLUDED"
        },
        {
            "name": "❌ INVALID: Top-10 Player (#8) vs Player (#120)",
            "data": {
                'player1': 'Top Player',
                'player2': 'Mid Player',
                'player1_ranking': 8,   # Too highly ranked
                'player2_ranking': 120,
                'tournament': 'ATP 250 Vienna', 
                'surface': 'Hard',
                'first_set_winner': 'player1',
                'first_set_score': '6-3'
            },
            "expected": "EXCLUDED"
        },
        {
            "name": "✅ VALID: Player (#60) vs Player (#180)",
            "data": {
                'player1': 'Player C',
                'player2': 'Player D',
                'player1_ranking': 60,
                'player2_ranking': 180,  # Player D is underdog
                'tournament': 'ATP 250 Vienna',
                'surface': 'Hard',
                'first_set_winner': 'player1', 
                'first_set_score': '7-5'
            },
            "expected": "INCLUDED"
        }
    ]
    
    print(f"\\n🧪 Testing {len(test_cases)} scenarios through complete pipeline...")
    print("-" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\\n{i}. {test_case['name']}")
        print(f"   Ranks: #{test_case['data']['player1_ranking']} vs #{test_case['data']['player2_ranking']}")
        
        try:
            # Test through prediction service
            result = prediction_service.predict_second_set_underdog(test_case['data'])
            
            if result.get('success', False):
                print(f"   Result: ✅ PREDICTION GENERATED")
                print(f"   Underdog: {result.get('underdog_player', 'Unknown')}")
                print(f"   Probability: {result.get('underdog_second_set_probability', 0):.1%}")
                print(f"   Expected: {test_case['expected']}")
                
                if test_case['expected'] == "EXCLUDED":
                    print(f"   ⚠️  WARNING: Should have been excluded!")
                else:
                    print(f"   ✅ CORRECT: Properly included")
                    
            else:
                print(f"   Result: ❌ PREDICTION REJECTED")
                print(f"   Reason: {result.get('error', 'Unknown error')}")
                print(f"   Expected: {test_case['expected']}")
                
                if test_case['expected'] == "EXCLUDED":
                    print(f"   ✅ CORRECT: Properly excluded")
                else:
                    print(f"   ⚠️  WARNING: Should have been included!")
                    
        except Exception as e:
            print(f"   Result: ❌ ERROR: {e}")
            print(f"   Expected: {test_case['expected']}")
    
    print("\\n" + "=" * 80)
    print("🏁 PIPELINE TEST COMPLETE")

if __name__ == "__main__":
    test_pipeline_scenarios()