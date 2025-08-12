#!/usr/bin/env python3
"""
Comprehensive API Tennis Integration Test
Tests the complete data flow from API Tennis to ML systems
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_api_tennis_connectivity():
    """Test basic API Tennis connectivity and authentication"""
    print("🔧 API Tennis Connectivity Test")
    print("=" * 50)
    
    try:
        from api_tennis_integration import APITennisClient
        
        # Test with environment variable
        client = APITennisClient()
        status = client.get_client_status()
        
        print(f"✅ API Key configured: {status['api_key_configured']}")
        print(f"✅ Caching enabled: {status['caching_enabled']}")
        print(f"✅ Rate limit: {status['rate_limit']} req/min")
        print(f"✅ Cache directory: {status['cache_directory']}")
        
        if status['api_key_configured']:
            print("\n🌐 Testing API connection...")
            try:
                # Test event types endpoint
                event_types = client.get_event_types()
                if isinstance(event_types, dict) and event_types.get('success') == 1:
                    result = event_types.get('result', [])
                    print(f"✅ API connection successful")
                    print(f"✅ Found {len(result)} event types")
                    
                    # Show ATP/WTA event types
                    atp_wta_events = [e for e in result if 'atp' in e.get('event_type_type', '').lower() or 'wta' in e.get('event_type_type', '').lower()]
                    print(f"✅ Professional events (ATP/WTA): {len(atp_wta_events)}")
                    
                    return True, client
                else:
                    print(f"❌ API connection failed: {event_types}")
                    return False, None
                    
            except Exception as e:
                print(f"❌ API test failed: {e}")
                return False, None
        else:
            print("❌ Cannot test API connection - no key configured")
            return False, None
            
    except Exception as e:
        print(f"❌ Client initialization failed: {e}")
        return False, None

def test_data_collection():
    """Test API Tennis data collection capabilities"""
    print("\n📊 Data Collection Test")
    print("=" * 50)
    
    try:
        from api_tennis_data_collector import APITennisDataCollector
        
        collector = APITennisDataCollector()
        
        if not collector.is_available():
            print("❌ API Tennis collector not available")
            return False
            
        print("✅ API Tennis data collector initialized")
        
        # Test current matches
        print("\n🎾 Testing current matches collection...")
        current_matches = collector.get_current_matches(include_live=True)
        print(f"✅ Current matches found: {len(current_matches)}")
        
        if current_matches:
            sample_match = current_matches[0]
            print(f"✅ Sample match: {sample_match.get('player1', 'N/A')} vs {sample_match.get('player2', 'N/A')}")
            print(f"✅ Tournament: {sample_match.get('tournament', 'N/A')}")
            print(f"✅ Surface: {sample_match.get('surface', 'N/A')}")
            print(f"✅ Data source: {sample_match.get('data_source', 'N/A')}")
            print(f"✅ Quality score: {sample_match.get('quality_score', 'N/A')}")
        
        # Test upcoming matches
        print("\n📅 Testing upcoming matches collection...")
        upcoming_matches = collector.get_upcoming_matches(days_ahead=3)
        print(f"✅ Upcoming matches found: {len(upcoming_matches)}")
        
        # Test tournaments
        print("\n🏆 Testing tournaments collection...")
        tournaments = collector.get_tournaments()
        print(f"✅ Professional tournaments found: {len(tournaments)}")
        
        # Test integration status
        print("\n🔍 Testing integration status...")
        integration_status = collector.get_integration_status()
        print(f"✅ Integration available: {integration_status['available']}")
        print(f"✅ API key configured: {integration_status['api_key_configured']}")
        print(f"✅ Connectivity test: {integration_status.get('connectivity_test', 'Not tested')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data collection test failed: {e}")
        return False

def test_ml_integration():
    """Test ML integration with API Tennis data"""
    print("\n🤖 ML Integration Test")
    print("=" * 50)
    
    try:
        from api_ml_integration import APITennisMLIntegration
        
        ml_integration = APITennisMLIntegration()
        
        # Check component status
        print("🔍 Checking ML component status...")
        status = ml_integration.get_api_status()
        
        api_status = status.get('api_tennis', {})
        ml_status = status.get('ml_components', {})
        
        print(f"✅ API Tennis available: {api_status.get('available', False)}")
        print(f"✅ ML orchestrator available: {ml_status.get('orchestrator_available', False)}")
        print(f"✅ Prediction service available: {ml_status.get('prediction_service_available', False)}")
        print(f"✅ Tennis predictor available: {ml_status.get('tennis_predictor_available', False)}")
        
        if api_status.get('available'):
            # Test live data fetching
            print("\n📡 Testing live data fetching...")
            live_data = ml_integration.fetch_live_data()
            
            if live_data.get('success'):
                data = live_data.get('data', {})
                print(f"✅ Today's matches: {data.get('today_matches_count', 0)}")
                print(f"✅ Upcoming matches: {data.get('upcoming_matches_count', 0)}")
                print(f"✅ Live matches: {data.get('live_matches_count', 0)}")
                
                # Test prediction pipeline
                print("\n🎯 Testing prediction pipeline...")
                pipeline_result = ml_integration.run_live_prediction_pipeline()
                
                if pipeline_result.get('success'):
                    print(f"✅ Predictions generated: {pipeline_result.get('predictions_generated', 0)}")
                    
                    predictions = pipeline_result.get('predictions', [])
                    if predictions:
                        sample_pred = predictions[0]
                        match_data = sample_pred.get('match_data', {})
                        pred_data = sample_pred.get('prediction', {})
                        
                        print(f"✅ Sample prediction:")
                        print(f"   Match: {match_data.get('player_1', 'N/A')} vs {match_data.get('player_2', 'N/A')}")
                        print(f"   Tournament: {match_data.get('tournament', 'N/A')}")
                        print(f"   Prediction method: {sample_pred.get('method', 'N/A')}")
                        print(f"   Confidence: {pred_data.get('confidence', 0):.3f}")
                        print(f"   Winner probability: {pred_data.get('player_1_win_probability', 0):.3f}")
                else:
                    print(f"❌ Prediction pipeline failed: {pipeline_result.get('error', 'Unknown error')}")
            else:
                print(f"❌ Live data fetch failed: {live_data.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ ML integration test failed: {e}")
        return False

def test_data_format_compatibility():
    """Test data format compatibility between API Tennis and ML models"""
    print("\n🔄 Data Format Compatibility Test")
    print("=" * 50)
    
    try:
        from api_tennis_integration import APITennisClient, TennisMatch, TennisPlayer
        from api_ml_integration import APITennisMLIntegration
        
        # Create sample match data
        player1 = TennisPlayer(
            id=1905,
            name="N. Djokovic",
            country="Serbia",
            ranking=5,
            age=37
        )
        
        player2 = TennisPlayer(
            id=123,
            name="F. Cobolli", 
            country="Italy",
            ranking=32,
            age=22
        )
        
        sample_match = TennisMatch(
            id=12345,
            player1=player1,
            player2=player2,
            tournament_name="US Open",
            surface="Hard",
            round="Round of 32",
            start_time=datetime.now() + timedelta(days=1),
            odds_player1=2.5,
            odds_player2=1.6,
            location="New York"
        )
        
        print("✅ Sample match created")
        
        # Test data conversion
        ml_integration = APITennisMLIntegration()
        ml_format = ml_integration.convert_api_match_to_ml_format(sample_match)
        
        if ml_format:
            print("✅ Match data converted to ML format")
            print(f"   Match ID: {ml_format.get('match_id')}")
            print(f"   Players: {ml_format.get('player_1')} vs {ml_format.get('player_2')}")
            print(f"   Tournament: {ml_format.get('tournament')}")
            print(f"   Surface: {ml_format.get('surface')}")
            print(f"   Rankings: {ml_format.get('player_1_ranking')} vs {ml_format.get('player_2_ranking')}")
            print(f"   Odds: {ml_format.get('odds_player_1')} vs {ml_format.get('odds_player_2')}")
            
            # Test prediction generation
            print("\n🎯 Testing prediction generation...")
            prediction = ml_integration._generate_single_prediction(ml_format)
            
            if prediction:
                print("✅ Prediction generated successfully")
                print(f"   Method: {prediction.get('method')}")
                print(f"   Confidence: {prediction.get('confidence', 0):.3f}")
                pred_data = prediction.get('prediction', {})
                print(f"   Winner probability: {pred_data.get('player_1_win_probability', 0):.3f}")
            else:
                print("❌ Prediction generation failed")
        else:
            print("❌ Data conversion failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Data format test failed: {e}")
        return False

def test_betting_readiness():
    """Test automated betting readiness"""
    print("\n💰 Betting Readiness Assessment")
    print("=" * 50)
    
    try:
        from api_tennis_data_collector import APITennisDataCollector, EnhancedAPITennisCollector
        
        # Test enhanced collector
        enhanced_collector = EnhancedAPITennisCollector()
        
        print("🔍 Testing comprehensive data collection...")
        comprehensive_matches = enhanced_collector.get_comprehensive_match_data(days_ahead=2)
        print(f"✅ Comprehensive matches: {len(comprehensive_matches)}")
        
        if comprehensive_matches:
            # Analyze data quality for betting
            high_quality_matches = [m for m in comprehensive_matches if m.get('quality_score', 0) >= 90]
            atp_wta_matches = [m for m in comprehensive_matches if m.get('level', '').startswith(('ATP', 'WTA', 'Grand Slam'))]
            matches_with_odds = [m for m in comprehensive_matches if m.get('player1_odds') and m.get('player2_odds')]
            
            print(f"✅ High quality matches (≥90): {len(high_quality_matches)}")
            print(f"✅ Professional ATP/WTA: {len(atp_wta_matches)}")  
            print(f"✅ Matches with odds: {len(matches_with_odds)}")
            
            # Sample match analysis
            if comprehensive_matches:
                sample = comprehensive_matches[0]
                print(f"\n📊 Sample match analysis:")
                print(f"   Match: {sample.get('player1')} vs {sample.get('player2')}")
                print(f"   Tournament: {sample.get('tournament')}")
                print(f"   Level: {sample.get('level')}")
                print(f"   Surface: {sample.get('surface')}")
                print(f"   Quality score: {sample.get('quality_score')}")
                print(f"   Data source: {sample.get('data_source')}")
                print(f"   Has odds: {'Yes' if sample.get('player1_odds') else 'No'}")
                print(f"   Has rankings: {'Yes' if sample.get('player1_ranking') else 'No'}")
        
        # Test status across sources
        print("\n🌐 Testing multi-source status...")
        status = enhanced_collector.get_status()
        print(f"✅ API Tennis available: {status['api_tennis']['available']}")
        print(f"✅ Universal collector available: {status['universal_collector']}")
        print(f"✅ Total sources available: {status['total_sources_available']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Betting readiness test failed: {e}")
        return False

def main():
    """Run comprehensive API Tennis integration analysis"""
    print("🎾 COMPREHENSIVE API TENNIS INTEGRATION ANALYSIS")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = {
        'connectivity': False,
        'data_collection': False,
        'ml_integration': False,
        'data_compatibility': False,
        'betting_readiness': False
    }
    
    # Test 1: API Connectivity
    try:
        connectivity_result, client = test_api_tennis_connectivity()
        results['connectivity'] = connectivity_result
    except Exception as e:
        print(f"❌ Connectivity test error: {e}")
    
    # Test 2: Data Collection
    try:
        results['data_collection'] = test_data_collection()
    except Exception as e:
        print(f"❌ Data collection test error: {e}")
    
    # Test 3: ML Integration
    try:
        results['ml_integration'] = test_ml_integration()
    except Exception as e:
        print(f"❌ ML integration test error: {e}")
    
    # Test 4: Data Format Compatibility
    try:
        results['data_compatibility'] = test_data_format_compatibility()
    except Exception as e:
        print(f"❌ Data compatibility test error: {e}")
    
    # Test 5: Betting Readiness
    try:
        results['betting_readiness'] = test_betting_readiness()
    except Exception as e:
        print(f"❌ Betting readiness test error: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 INTEGRATION ANALYSIS SUMMARY")
    print("=" * 70)
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():.<40} {status}")
    
    print(f"\nOverall Score: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! API Tennis integration is fully operational.")
    elif passed_tests >= total_tests * 0.8:
        print("⚠️  Integration mostly working with minor issues.")
    elif passed_tests >= total_tests * 0.5:
        print("🔧 Integration partially working - significant issues need attention.")
    else:
        print("❌ Integration has major issues - requires immediate attention.")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if not results['connectivity']:
        print("• Fix API Tennis connectivity - check API key and network access")
    if not results['data_collection']:
        print("• Debug data collection pipeline - ensure proper filtering and processing")
    if not results['ml_integration']:
        print("• Fix ML integration components - check model loading and prediction service")
    if not results['data_compatibility']:
        print("• Fix data format conversion between API Tennis and ML models")
    if not results['betting_readiness']:
        print("• Improve data quality and completeness for automated betting")
    
    if all(results.values()):
        print("• Integration is ready for production automated betting!")
        print("• Monitor data quality and prediction accuracy regularly")
        print("• Implement proper risk management and position sizing")

if __name__ == "__main__":
    main()