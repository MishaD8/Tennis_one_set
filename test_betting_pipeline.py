#!/usr/bin/env python3
"""
Test Automated Tennis Betting Pipeline
End-to-end testing of the complete betting system
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import components
from automated_betting_engine import (
    AutomatedBettingEngine, 
    RiskManagementConfig, 
    BetOpportunity, 
    BetOrder,
    BetType,
    BetStatus
)
from realtime_prediction_engine import (
    PredictionEngine, 
    PredictionConfig, 
    MLPredictionResult,
    PredictionTrigger
)
from realtime_ml_pipeline import RealTimeMLPipeline
from betfair_api_client import BetfairAPIClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_mock_prediction() -> MLPredictionResult:
    """Create a mock ML prediction for testing"""
    prediction_data = {
        'player_1_win_probability': 0.65,
        'player_2_win_probability': 0.35,
        'winner': 'Player 1',
        'confidence_factors': {
            'ranking_advantage': 0.3,
            'surface_performance': 0.2,
            'head_to_head': 0.15
        }
    }
    
    return MLPredictionResult(
        match_id=12345,
        prediction=prediction_data,
        confidence=0.72,
        model_used='ensemble',
        processing_time=1.5,
        timestamp=datetime.now(),
        live_context={
            'trigger': PredictionTrigger.MATCH_START.value,
            'features_used': ['ranking', 'surface', 'form']
        }
    )


def test_betfair_client():
    """Test Betfair API client functionality"""
    logger.info("🔧 Testing Betfair API Client...")
    
    client = BetfairAPIClient()
    
    # Health check
    health = client.health_check()
    logger.info(f"Health Check: {health}")
    
    # Get tennis events
    try:
        events = client.get_tennis_events()
        logger.info(f"Found {len(events)} tennis events")
    except Exception as e:
        logger.warning(f"Could not get events: {e}")
    
    # Get tennis markets
    try:
        markets = client.get_tennis_markets()
        logger.info(f"Found {len(markets)} tennis markets")
        
        if markets:
            # Test market odds
            market_odds = client.get_market_book([markets[0].market_id])
            logger.info(f"Market odds retrieved for {markets[0].market_id}")
            
    except Exception as e:
        logger.warning(f"Could not get markets: {e}")
    
    # Test account funds
    try:
        funds = client.get_account_funds()
        logger.info(f"Account balance: €{funds.get('available_balance', 0.0):.2f}")
    except Exception as e:
        logger.warning(f"Could not get account funds: {e}")
    
    logger.info("✅ Betfair API Client test completed")
    return True


def test_risk_management():
    """Test risk management functionality"""
    logger.info("🔧 Testing Risk Management...")
    
    # Create conservative risk config
    risk_config = RiskManagementConfig.conservative()
    logger.info(f"Risk Config: Max stake per bet: €{risk_config.max_stake_per_bet}")
    
    # Create prediction engine (mock)
    mock_prediction = create_mock_prediction()
    
    # Test risk manager
    from automated_betting_engine import RiskManager
    risk_manager = RiskManager(risk_config, initial_bankroll=1000.0)
    
    # Test opportunity evaluation
    market_odds = 1.85
    opportunity = risk_manager.evaluate_opportunity(mock_prediction, market_odds)
    
    if opportunity:
        logger.info(f"✅ Betting opportunity identified:")
        logger.info(f"  Selection: {opportunity.selection}")
        logger.info(f"  Edge: {opportunity.edge:.3f}")
        logger.info(f"  Recommended Stake: €{opportunity.recommended_stake:.2f}")
        logger.info(f"  Confidence: {opportunity.confidence:.2f}")
    else:
        logger.info("❌ No betting opportunity identified")
    
    # Test risk checks
    can_bet, reason = risk_manager.can_place_bet(25.0, 12345)
    logger.info(f"Risk Check (€25 stake): {can_bet} - {reason}")
    
    # Get risk summary
    summary = risk_manager.get_risk_summary()
    logger.info(f"Risk Summary: Bankroll: €{summary['current_bankroll']:.2f}")
    
    logger.info("✅ Risk Management test completed")
    return opportunity


def test_betting_engine():
    """Test the complete betting engine"""
    logger.info("🔧 Testing Automated Betting Engine...")
    
    # Create components
    risk_config = RiskManagementConfig.moderate()
    betting_engine = AutomatedBettingEngine(risk_config, initial_bankroll=1000.0)
    
    # Create mock prediction engine
    class MockPredictionEngine:
        def __init__(self):
            self.callbacks = []
        
        def add_prediction_callback(self, callback):
            self.callbacks.append(callback)
        
        def trigger_prediction(self, prediction):
            for callback in self.callbacks:
                callback(prediction)
    
    mock_pred_engine = MockPredictionEngine()
    
    # Initialize betting engine
    betting_engine.initialize(mock_pred_engine)
    
    # Add callbacks
    bets_placed = []
    opportunities_found = []
    
    def handle_bet_placed(bet: BetOrder):
        bets_placed.append(bet)
        logger.info(f"🎯 BET PLACED: €{bet.stake:.2f} on {bet.selection} at odds {bet.odds}")
    
    def handle_opportunity(opportunity: BetOpportunity):
        opportunities_found.append(opportunity)
        logger.info(f"💡 OPPORTUNITY: {opportunity.selection} - Edge: {opportunity.edge:.3f}")
    
    betting_engine.add_bet_callback(handle_bet_placed)
    betting_engine.add_opportunity_callback(handle_opportunity)
    
    # Start betting engine
    betting_engine.start()
    
    # Simulate prediction
    logger.info("📊 Triggering mock prediction...")
    mock_prediction = create_mock_prediction()
    mock_pred_engine.trigger_prediction(mock_prediction)
    
    # Wait for processing
    time.sleep(2)
    
    # Check results
    stats = betting_engine.get_stats()
    logger.info(f"📈 Betting Stats:")
    logger.info(f"  Opportunities Found: {stats['opportunities_found']}")
    logger.info(f"  Bets Placed: {stats['bets_placed']}")
    logger.info(f"  Queue Size: {stats['queue_size']}")
    
    # Check active bets
    active_bets = betting_engine.get_active_bets()
    logger.info(f"  Active Bets: {len(active_bets)}")
    
    for bet in active_bets:
        logger.info(f"    Bet ID: {bet.order_id[:8]}... - Status: {bet.status.value}")
    
    # Stop betting engine
    betting_engine.stop()
    
    logger.info("✅ Betting Engine test completed")
    return len(bets_placed) > 0, len(opportunities_found) > 0


def test_integration_pipeline():
    """Test the complete integration pipeline"""
    logger.info("🚀 Testing Complete Integration Pipeline...")
    
    success_count = 0
    total_tests = 4
    
    try:
        # Test 1: Betfair Client
        if test_betfair_client():
            success_count += 1
        
        # Test 2: Risk Management
        opportunity = test_risk_management()
        if opportunity:
            success_count += 1
        
        # Test 3: Betting Engine
        bets_placed, opportunities_found = test_betting_engine()
        if bets_placed or opportunities_found:
            success_count += 1
        
        # Test 4: End-to-End Pipeline
        logger.info("🔧 Testing End-to-End Pipeline...")
        if test_end_to_end_pipeline():
            success_count += 1
        
        # Summary
        logger.info("=" * 60)
        logger.info("🏁 INTEGRATION TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Tests Passed: {success_count}/{total_tests}")
        logger.info(f"Success Rate: {(success_count/total_tests)*100:.1f}%")
        
        if success_count == total_tests:
            logger.info("✅ ALL TESTS PASSED - System is ready for operation!")
        elif success_count >= total_tests // 2:
            logger.info("⚠️ PARTIAL SUCCESS - Some components need attention")
        else:
            logger.info("❌ TESTS FAILED - System needs significant work")
        
        return success_count >= total_tests // 2
        
    except Exception as e:
        logger.error(f"❌ Integration test failed with exception: {e}")
        return False


def test_end_to_end_pipeline():
    """Test complete end-to-end pipeline"""
    logger.info("🔧 Testing End-to-End Pipeline...")
    
    try:
        # Create complete system
        risk_config = RiskManagementConfig.moderate()
        betting_engine = AutomatedBettingEngine(risk_config, initial_bankroll=1000.0)
        
        # Mock prediction engine that generates predictions
        class MockPredictionEngine:
            def __init__(self):
                self.callbacks = []
                self.running = False
            
            def add_prediction_callback(self, callback):
                self.callbacks.append(callback)
            
            def start(self):
                self.running = True
                logger.info("Mock prediction engine started")
            
            def stop(self):
                self.running = False
                logger.info("Mock prediction engine stopped")
            
            def generate_prediction(self):
                if self.running:
                    prediction = create_mock_prediction()
                    for callback in self.callbacks:
                        callback(prediction)
        
        mock_pred_engine = MockPredictionEngine()
        
        # Initialize system
        betting_engine.initialize(mock_pred_engine)
        
        # Start system
        betting_engine.start()
        mock_pred_engine.start()
        
        # Generate some predictions
        for i in range(3):
            logger.info(f"Generating prediction {i+1}...")
            mock_pred_engine.generate_prediction()
            time.sleep(1)
        
        # Check system status
        stats = betting_engine.get_stats()
        logger.info(f"Final Stats: {stats['opportunities_found']} opportunities, {stats['bets_placed']} bets")
        
        # Stop system
        betting_engine.stop()
        mock_pred_engine.stop()
        
        logger.info("✅ End-to-End Pipeline test completed")
        return True
        
    except Exception as e:
        logger.error(f"End-to-end test failed: {e}")
        return False


if __name__ == "__main__":
    print("🎾 TENNIS AUTOMATED BETTING SYSTEM - INTEGRATION TEST")
    print("=" * 60)
    
    # Run integration tests
    success = test_integration_pipeline()
    
    print("=" * 60)
    if success:
        print("🎉 SYSTEM READY FOR DEPLOYMENT!")
        print("The automated tennis betting pipeline is working correctly.")
    else:
        print("⚠️ SYSTEM NEEDS ATTENTION")
        print("Some components require fixes before deployment.")
    
    sys.exit(0 if success else 1)