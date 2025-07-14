#!/usr/bin/env python3
"""
🎾 Test Enhanced Tennis Prediction System
Tests the complete system with adaptive ensemble weights
"""

from tennis_prediction_module import TennisPredictionService, create_match_data
from adaptive_ensemble_optimizer import init_ensemble_optimizer, get_ensemble_performance_report
import json

def test_complete_system():
    """Test the complete enhanced prediction system"""
    print("🎾 Testing Complete Enhanced Tennis Prediction System")
    print("=" * 60)
    
    # 1. Initialize prediction service with adaptive weights
    print("1. Initializing prediction service...")
    service = TennisPredictionService(use_adaptive_weights=True)
    
    # Check if models are loaded
    info = service.get_model_info()
    print(f"   Models loaded: {info['status']}")
    print(f"   Adaptive weights: {info.get('adaptive_weights_enabled', False)}")
    
    # 2. Test multiple predictions
    print("\n2. Testing multiple predictions...")
    test_matches = [
        {
            'name': 'Djokovic vs Alcaraz',
            'data': create_match_data(
                player_rank=1, opponent_rank=2,
                player_recent_win_rate=0.85, h2h_win_rate=0.6,
                player_surface_advantage=0.1
            )
        },
        {
            'name': 'Sinner vs Medvedev', 
            'data': create_match_data(
                player_rank=4, opponent_rank=3,
                player_recent_win_rate=0.75, h2h_win_rate=0.45,
                player_surface_advantage=-0.05
            )
        },
        {
            'name': 'Underdog scenario',
            'data': create_match_data(
                player_rank=50, opponent_rank=10,
                player_recent_win_rate=0.80, h2h_win_rate=0.2,
                player_surface_advantage=0.15
            )
        }
    ]
    
    predictions = []
    for i, match in enumerate(test_matches):
        print(f"\n   Match {i+1}: {match['name']}")
        try:
            result = service.predict_match(match['data'], return_details=True)
            predictions.append(result)
            
            print(f"     Probability: {result['probability']:.3f}")
            print(f"     Confidence: {result['confidence']}")
            print(f"     Recommendation: {result['recommendation']}")
            
            # Show weight differences if adaptive
            if result.get('adaptive_weights_used'):
                base_weights = result['base_weights']
                current_weights = result['model_weights']
                print("     Weight changes:")
                for model in base_weights:
                    base_w = base_weights[model]
                    curr_w = current_weights.get(model, base_w)
                    change = ((curr_w - base_w) / base_w) * 100
                    if abs(change) > 1:  # Only show significant changes
                        print(f"       {model}: {change:+.1f}%")
            
        except Exception as e:
            print(f"     Error: {e}")
    
    # 3. Simulate recording actual results
    print("\n3. Simulating match results for learning...")
    
    # Simulate some results (in real system, these would come from actual matches)
    simulated_results = [
        {'match_idx': 0, 'actual': 1, 'info': 'Djokovic won'},
        {'match_idx': 1, 'actual': 0, 'info': 'Medvedev won'},
        {'match_idx': 2, 'actual': 1, 'info': 'Underdog won!'}
    ]
    
    for sim_result in simulated_results:
        match_idx = sim_result['match_idx']
        if match_idx < len(predictions):
            success = service.record_match_result(
                test_matches[match_idx]['data'],
                predictions[match_idx],
                sim_result['actual'],
                {'info': sim_result['info']}
            )
            print(f"   Recorded: {sim_result['info']} - Success: {success}")
    
    # 4. Get performance report
    print("\n4. Performance report:")
    try:
        report = service.get_ensemble_performance_report()
        if 'error' not in report:
            print(f"   Best model: {report['summary']['best_performing_model']}")
            print(f"   Average accuracy: {report['summary']['average_accuracy']:.1%}")
            print(f"   Total predictions: {report['summary']['total_predictions']}")
            
            # Show model performances
            print("   Individual model performance:")
            for model, perf in report['model_performances'].items():
                print(f"     {model}: {perf['accuracy_7d']:.1%} accuracy")
        else:
            print(f"   Error: {report['error']}")
    except Exception as e:
        print(f"   Error getting report: {e}")
    
    # 5. Test enhanced API integration
    print("\n5. Testing enhanced API integration...")
    try:
        from enhanced_api_integration import init_enhanced_api
        api = init_enhanced_api()
        
        # Test cache stats
        cache_stats = api.get_cache_stats()
        print(f"   Cache manager available: {cache_stats['cache_manager'].get('redis_available', False)}")
        print(f"   Total cache hit rate: {cache_stats['cache_manager']['hit_rates']['total_hit_rate']:.1f}%")
        
    except Exception as e:
        print(f"   API integration error: {e}")
    
    print("\n🎯 Complete system test finished!")
    return True

if __name__ == "__main__":
    test_complete_system()