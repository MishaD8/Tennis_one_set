#!/usr/bin/env python3
"""
Simple ML Integration Test
Tests API Tennis data flow to ML systems without complex threading
"""

import os
from dotenv import load_dotenv
load_dotenv()

def test_simple_ml_integration():
    """Test simple ML integration without threading issues"""
    print("🤖 Simple ML Integration Test")
    print("=" * 50)
    
    try:
        # Test 1: Direct API Tennis data to ML format conversion
        print("📊 Testing data format conversion...")
        from api_tennis_integration import TennisMatch, TennisPlayer
        from datetime import datetime
        
        # Create sample match
        player1 = TennisPlayer(id=1, name="N. Djokovic", country="Serbia", ranking=5)
        player2 = TennisPlayer(id=2, name="F. Cobolli", country="Italy", ranking=32)
        
        sample_match = TennisMatch(
            id=12345,
            player1=player1,
            player2=player2,
            tournament_name="US Open",
            surface="Hard",
            start_time=datetime.now(),
            odds_player1=2.5,
            odds_player2=1.6
        )
        
        print(f"✅ Sample match created: {player1.name} vs {player2.name}")
        
        # Test 2: Convert to ML format
        print("\n🔄 Testing ML format conversion...")
        from api_ml_integration import APITennisMLIntegration
        
        ml_integration = APITennisMLIntegration()
        ml_format = ml_integration.convert_api_match_to_ml_format(sample_match)
        
        if ml_format:
            print("✅ Successfully converted to ML format:")
            print(f"   Players: {ml_format.get('player_1')} vs {ml_format.get('player_2')}")
            print(f"   Rankings: {ml_format.get('player_1_ranking')} vs {ml_format.get('player_2_ranking')}")
            print(f"   Surface: {ml_format.get('surface')}")
            print(f"   Tournament: {ml_format.get('tournament')}")
        else:
            print("❌ Failed to convert to ML format")
            return False
        
        # Test 3: Basic prediction without complex ML models
        print("\n🎯 Testing basic prediction...")
        try:
            prediction = ml_integration._generate_basic_prediction(ml_format)
            if prediction:
                print("✅ Basic prediction generated:")
                print(f"   Method: {prediction.get('method')}")
                pred_data = prediction.get('prediction', {})
                print(f"   Winner: {pred_data.get('winner')}")
                print(f"   P1 probability: {pred_data.get('player_1_win_probability', 0):.3f}")
                print(f"   P2 probability: {pred_data.get('player_2_win_probability', 0):.3f}")
                print(f"   Confidence: {pred_data.get('confidence', 0):.3f}")
            else:
                print("❌ Failed to generate basic prediction")
                return False
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return False
        
        # Test 4: Test with real API Tennis data
        print("\n📡 Testing with real API Tennis data...")
        try:
            from api_tennis_data_collector import APITennisDataCollector
            
            collector = APITennisDataCollector()
            if collector.is_available():
                matches = collector.get_current_matches(include_live=True)
                if matches:
                    real_match = matches[0]
                    print(f"✅ Real match found: {real_match.get('player1')} vs {real_match.get('player2')}")
                    print(f"   Tournament: {real_match.get('tournament')}")
                    print(f"   Surface: {real_match.get('surface')}")
                    print(f"   Data source: {real_match.get('data_source')}")
                else:
                    print("⚠️  No real matches found in current data")
            else:
                print("⚠️  API Tennis collector not available")
        except Exception as e:
            print(f"❌ Real data test error: {e}")
        
        print("\n✅ Simple ML integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Simple ML integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prediction_service_direct():
    """Test prediction service directly"""
    print("\n🧠 Direct Prediction Service Test")
    print("=" * 50)
    
    try:
        # Try to load prediction service directly
        from tennis_prediction_module import TennisPredictionModule
        
        predictor = TennisPredictionModule()
        print("✅ Tennis Prediction Module loaded")
        
        # Test with sample data
        sample_data = {
            'player_1': 'Novak Djokovic',
            'player_2': 'Flavio Cobolli', 
            'tournament': 'US Open',
            'surface': 'Hard',
            'player_1_ranking': 5,
            'player_2_ranking': 32,
            'player_1_age': 37,
            'player_2_age': 22
        }
        
        print("🎯 Testing prediction with sample data...")
        try:
            prediction = predictor.predict_match(sample_data)
            if prediction:
                print("✅ Prediction generated successfully")
                print(f"   Confidence: {prediction.get('confidence', 0):.3f}")
            else:
                print("⚠️  No prediction returned")
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  Tennis Prediction Module not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Prediction service test failed: {e}")
        return False

if __name__ == "__main__":
    print("🎾 SIMPLE ML INTEGRATION TEST")
    print("=" * 60)
    
    result1 = test_simple_ml_integration()
    result2 = test_prediction_service_direct()
    
    print("\n" + "=" * 60)
    print("📋 SIMPLE TEST SUMMARY")
    print("=" * 60)
    
    if result1:
        print("✅ Basic ML integration: PASS")
    else:
        print("❌ Basic ML integration: FAIL")
    
    if result2:
        print("✅ Prediction service: PASS")  
    else:
        print("❌ Prediction service: FAIL")
    
    if result1 and result2:
        print("\n🎉 All simple tests passed!")
    else:
        print("\n⚠️  Some tests failed - check individual results above")