#!/usr/bin/env python3
"""
WebSocket Integration Testing Suite
Comprehensive testing for real-time tennis data pipeline
"""

import os
import sys
import json
import time
import logging
import asyncio
import threading
import unittest
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import components to test
from websocket_tennis_client import (
    TennisWebSocketClient, LiveMatchEvent, WSConnectionState,
    LiveDataProcessor, WebSocketManager
)
from realtime_ml_pipeline import (
    RealTimeMLPipeline, LiveDataBuffer, MLPredictionProcessor
)
from realtime_data_validator import (
    LiveDataValidator, DataQualityMonitor, ValidationSeverity
)
from realtime_prediction_engine import (
    PredictionEngine, TriggerDetector, PredictionTrigger
)
from automated_betting_engine import (
    AutomatedBettingEngine, RiskManager, RiskManagementConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestWebSocketClient(unittest.TestCase):
    """Test WebSocket client functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.client = TennisWebSocketClient()
        self.test_events = []
        
        # Add test event callback
        def capture_event(event):
            self.test_events.append(event)
        
        self.client.add_event_callback(capture_event)
    
    def test_client_initialization(self):
        """Test client initialization"""
        self.assertIsNotNone(self.client)
        self.assertEqual(self.client.connection_state, WSConnectionState.DISCONNECTED)
        self.assertIsNotNone(self.client.data_processor)
    
    def test_connection_url_building(self):
        """Test WebSocket URL building"""
        # Mock API key
        self.client.api_key = "test_key"
        
        url = self.client._build_connection_url()
        self.assertIn("wss://wss.api-tennis.com/live", url)
        self.assertIn("APIkey=test_key", url)
    
    def test_filter_setting(self):
        """Test filter configuration"""
        self.client.set_filters(tournament_key=123, player_key=456)
        
        url = self.client._build_connection_url()
        self.assertIn("tournament_key=123", url)
        self.assertIn("player_key=456", url)
    
    def test_event_processing(self):
        """Test live event processing"""
        # Create test event data
        test_data = {
            "event_key": 12345,
            "event_date": "2024-01-15",
            "event_time": "14:30",
            "event_first_player": "Test Player 1",
            "first_player_key": 1001,
            "event_second_player": "Test Player 2",
            "second_player_key": 1002,
            "event_final_result": "1 - 0",
            "event_game_result": "3 - 2",
            "event_serve": "First Player",
            "event_winner": None,
            "event_status": "Set 1",
            "event_type_type": "ATP",
            "tournament_name": "Test Tournament",
            "tournament_key": 5001,
            "tournament_round": "R1",
            "tournament_season": "2024",
            "event_live": "1",
            "scores": [],
            "pointbypoint": [],
            "statistics": []
        }
        
        # Process event
        asyncio.run(self.client._handle_message(json.dumps(test_data)))
        
        # Verify event was processed
        self.assertEqual(len(self.test_events), 1)
        event = self.test_events[0]
        self.assertEqual(event.event_key, 12345)
        self.assertEqual(event.first_player, "Test Player 1")
    
    def test_stats_tracking(self):
        """Test statistics tracking"""
        stats = self.client.get_stats()
        
        self.assertIn('events_received', stats)
        self.assertIn('events_processed', stats)
        self.assertIn('connection_state', stats)


class TestLiveDataProcessor(unittest.TestCase):
    """Test live data processing"""
    
    def setUp(self):
        """Set up test environment"""
        self.processor = LiveDataProcessor()
    
    def test_event_validation(self):
        """Test event data validation"""
        # Valid event data
        valid_data = {
            'event_key': 123,
            'event_first_player': 'Player 1',
            'event_second_player': 'Player 2',
            'tournament_name': 'Test Tournament',
            'event_status': 'Live'
        }
        
        self.assertTrue(self.processor._validate_event_data(valid_data))
        
        # Invalid event data (missing required field)
        invalid_data = {
            'event_key': 123,
            'event_first_player': 'Player 1'
            # Missing required fields
        }
        
        self.assertFalse(self.processor._validate_event_data(invalid_data))
    
    def test_match_cache_update(self):
        """Test match state caching"""
        event = LiveMatchEvent(
            event_key=123,
            event_date="2024-01-15",
            event_time="14:30",
            first_player="Player 1",
            first_player_key=1001,
            second_player="Player 2",
            second_player_key=1002,
            final_result="1 - 0",
            game_result="3 - 2",
            serve="First Player",
            winner=None,
            status="Set 1",
            event_type="ATP",
            tournament_name="Test Tournament",
            tournament_key=5001,
            tournament_round="R1",
            tournament_season="2024",
            live=True,
            scores=[],
            point_by_point=[],
            statistics=[],
            timestamp=datetime.now()
        )
        
        # Process event
        processed_event = self.processor.process_live_event(event.__dict__)
        
        # Check cache update
        match_state = self.processor.get_match_state(123)
        self.assertIsNotNone(match_state)
        self.assertEqual(match_state['event'].event_key, 123)


class TestDataValidator(unittest.TestCase):
    """Test data validation system"""
    
    def setUp(self):
        """Set up test environment"""
        self.validator = LiveDataValidator()
    
    def test_field_validation(self):
        """Test individual field validation"""
        # Test valid data
        valid_data = {
            'event_key': 12345,
            'event_date': '2024-01-15',
            'event_time': '14:30',
            'first_player': 'Rafael Nadal',
            'second_player': 'Novak Djokovic',
            'first_player_key': 1001,
            'second_player_key': 1002,
            'final_result': '1 - 0',
            'game_result': '3 - 2',
            'tournament_name': 'Australian Open',
            'tournament_key': 5001,
            'event_status': 'Set 1',
            'live': True
        }
        
        issues = self.validator._validate_fields(valid_data, 12345)
        error_issues = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        self.assertEqual(len(error_issues), 0)
    
    def test_cross_field_validation(self):
        """Test cross-field validation"""
        # Test data with same player keys (should fail)
        invalid_data = {
            'first_player_key': 1001,
            'second_player_key': 1001,  # Same as first player
            'first_player': 'Player 1',
            'second_player': 'Player 2',
            'final_result': '1 - 0',
            'game_result': '3 - 2'
        }
        
        issues = self.validator._validate_cross_fields(invalid_data, 12345)
        self.assertTrue(any('Player keys should be different' in issue.message for issue in issues))
    
    def test_score_progression_validation(self):
        """Test score progression validation"""
        # Create events with valid progression
        event1 = LiveMatchEvent(
            event_key=123, event_date="2024-01-15", event_time="14:30",
            first_player="Player 1", first_player_key=1001,
            second_player="Player 2", second_player_key=1002,
            final_result="0 - 0", game_result="0 - 0",
            serve="First Player", winner=None, status="Set 1",
            event_type="ATP", tournament_name="Test Tournament",
            tournament_key=5001, tournament_round="R1", tournament_season="2024",
            live=True, scores=[], point_by_point=[], statistics=[],
            timestamp=datetime.now()
        )
        
        event2 = LiveMatchEvent(
            event_key=123, event_date="2024-01-15", event_time="14:30",
            first_player="Player 1", first_player_key=1001,
            second_player="Player 2", second_player_key=1002,
            final_result="1 - 0", game_result="3 - 2",
            serve="First Player", winner=None, status="Set 1",
            event_type="ATP", tournament_name="Test Tournament",
            tournament_key=5001, tournament_round="R1", tournament_season="2024",
            live=True, scores=[], point_by_point=[], statistics=[],
            timestamp=datetime.now()
        )
        
        # Validate first event
        report1 = self.validator.validate_live_event(event1)
        
        # Validate second event (should show valid progression)
        report2 = self.validator.validate_live_event(event2)
        
        # Check that progression issues are minimal
        progression_issues = [i for i in report2.issues if 'progression' in i.message.lower()]
        self.assertEqual(len(progression_issues), 0)


class TestMLPipeline(unittest.TestCase):
    """Test ML pipeline components"""
    
    def setUp(self):
        """Set up test environment"""
        self.buffer = LiveDataBuffer()
        self.processor = MLPredictionProcessor()
    
    def test_data_buffer_features(self):
        """Test feature extraction from buffered data"""
        event = LiveMatchEvent(
            event_key=123, event_date="2024-01-15", event_time="14:30",
            first_player="Player 1", first_player_key=1001,
            second_player="Player 2", second_player_key=1002,
            final_result="1 - 0", game_result="6 - 4",
            serve="Second Player", winner=None, status="Set 2",
            event_type="ATP Masters", tournament_name="Miami Open",
            tournament_key=5001, tournament_round="QF", tournament_season="2024",
            live=True, scores=[{"score_first": "1", "score_second": "0", "score_set": "1"}],
            point_by_point=[], statistics=[],
            timestamp=datetime.now()
        )
        
        # Add event to buffer
        self.buffer.add_live_event(event)
        
        # Extract features
        features = self.buffer.get_match_features(123)
        
        # Verify features
        self.assertIn('match_id', features)
        self.assertIn('player1', features)
        self.assertIn('player2', features)
        self.assertIn('tournament', features)
        self.assertIn('surface', features)
        self.assertIn('current_score', features)
        self.assertEqual(features['match_id'], 123)
        self.assertEqual(features['player1'], 'Player 1')
    
    def test_surface_determination(self):
        """Test surface determination from tournament names"""
        # Test clay tournaments
        clay_features = self.buffer._determine_surface("French Open")
        self.assertEqual(clay_features, "Clay")
        
        # Test grass tournaments
        grass_features = self.buffer._determine_surface("Wimbledon")
        self.assertEqual(grass_features, "Grass")
        
        # Test hard court (default)
        hard_features = self.buffer._determine_surface("US Open")
        self.assertEqual(hard_features, "Hard")


class TestPredictionEngine(unittest.TestCase):
    """Test prediction engine components"""
    
    def setUp(self):
        """Set up test environment"""
        self.trigger_detector = TriggerDetector()
    
    def test_match_state_extraction(self):
        """Test match state extraction"""
        event = LiveMatchEvent(
            event_key=123, event_date="2024-01-15", event_time="14:30",
            first_player="Player 1", first_player_key=1001,
            second_player="Player 2", second_player_key=1002,
            final_result="1 - 0", game_result="3 - 2",
            serve="First Player", winner=None, status="Set 1",
            event_type="ATP", tournament_name="Test Tournament",
            tournament_key=5001, tournament_round="R1", tournament_season="2024",
            live=True, scores=[], point_by_point=[], statistics=[],
            timestamp=datetime.now()
        )
        
        state = self.trigger_detector._extract_match_state(event)
        
        self.assertEqual(state['sets_p1'], 1)
        self.assertEqual(state['sets_p2'], 0)
        self.assertEqual(state['games_p1'], 3)
        self.assertEqual(state['games_p2'], 2)
        self.assertTrue(state['live'])
    
    def test_trigger_detection(self):
        """Test trigger detection logic"""
        from realtime_data_validator import DataQualityReport
        
        # Create mock quality report
        quality_report = DataQualityReport(
            match_id=123,
            timestamp=datetime.now(),
            overall_score=0.9,
            metric_scores={},
            issues=[],
            recommendations=[]
        )
        
        # Test match start detection
        event = LiveMatchEvent(
            event_key=123, event_date="2024-01-15", event_time="14:30",
            first_player="Player 1", first_player_key=1001,
            second_player="Player 2", second_player_key=1002,
            final_result="0 - 0", game_result="0 - 0",
            serve="First Player", winner=None, status="Set 1",
            event_type="ATP", tournament_name="Test Tournament",
            tournament_key=5001, tournament_round="R1", tournament_season="2024",
            live=True, scores=[], point_by_point=[], statistics=[],
            timestamp=datetime.now()
        )
        
        triggers = self.trigger_detector.analyze_event(event, quality_report)
        
        # Should detect match start
        match_start_triggers = [t for t in triggers if t[0] == PredictionTrigger.MATCH_START]
        self.assertTrue(len(match_start_triggers) > 0)


class TestBettingEngine(unittest.TestCase):
    """Test automated betting engine"""
    
    def setUp(self):
        """Set up test environment"""
        self.risk_config = RiskManagementConfig.conservative()
        self.risk_manager = RiskManager(self.risk_config, 1000.0)
    
    def test_risk_management(self):
        """Test risk management rules"""
        # Test stake limits
        can_bet, reason = self.risk_manager.can_place_bet(20.0, 123)
        self.assertTrue(can_bet)
        
        # Test excessive stake
        can_bet, reason = self.risk_manager.can_place_bet(50.0, 123)  # Exceeds max_stake_per_bet
        self.assertFalse(can_bet)
        self.assertIn('maximum per bet', reason)
    
    def test_opportunity_evaluation(self):
        """Test betting opportunity evaluation"""
        from realtime_ml_pipeline import MLPredictionResult
        
        # Create mock prediction result
        prediction = MLPredictionResult(
            match_id=123,
            prediction={
                'winner': 'Player 1',
                'player_1_win_probability': 0.7,
                'player_2_win_probability': 0.3,
                'confidence': 0.8
            },
            confidence=0.8,
            model_used='test_model',
            processing_time=0.1,
            timestamp=datetime.now(),
            live_context={}
        )
        
        # Test with favorable odds (edge exists)
        market_odds = 2.0  # Implied probability = 0.5, our prediction = 0.7
        opportunity = self.risk_manager.evaluate_opportunity(prediction, market_odds)
        
        self.assertIsNotNone(opportunity)
        self.assertTrue(opportunity.edge > 0)
        self.assertEqual(opportunity.confidence, 0.8)
    
    def test_stake_calculation(self):
        """Test stake calculation methods"""
        from automated_betting_engine import StakeCalculator
        
        calculator = StakeCalculator(1000.0)
        
        # Test Kelly criterion
        kelly_stake = calculator.calculate_kelly_stake(0.6, 2.0, 50.0)
        self.assertGreater(kelly_stake, 0)
        self.assertLessEqual(kelly_stake, 50.0)
        
        # Test confidence weighting
        weighted_stake = calculator.calculate_confidence_weighted_stake(0.8, 25.0, 50.0)
        self.assertEqual(weighted_stake, 20.0)  # 25.0 * 0.8


class TestIntegrationWorkflow(unittest.TestCase):
    """Test end-to-end workflow integration"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.events_processed = []
        self.predictions_generated = []
        self.opportunities_found = []
        self.bets_placed = []
    
    @patch('websocket_tennis_client.websockets.connect')
    async def test_full_pipeline(self, mock_websocket):
        """Test complete pipeline from WebSocket to betting"""
        # Mock WebSocket connection
        mock_ws = Mock()
        mock_websocket.return_value = mock_ws
        
        # Create test event
        test_event_data = {
            "event_key": 12345,
            "event_date": "2024-01-15",
            "event_time": "14:30",
            "event_first_player": "Rafael Nadal",
            "first_player_key": 1001,
            "event_second_player": "Novak Djokovic",
            "second_player_key": 1002,
            "event_final_result": "1 - 0",
            "event_game_result": "3 - 2",
            "event_serve": "First Player",
            "event_winner": None,
            "event_status": "Set 1",
            "event_type_type": "ATP Masters",
            "tournament_name": "Miami Open",
            "tournament_key": 5001,
            "tournament_round": "QF",
            "tournament_season": "2024",
            "event_live": "1",
            "scores": [{"score_first": "1", "score_second": "0", "score_set": "1"}],
            "pointbypoint": [],
            "statistics": []
        }
        
        # This would test the full workflow but requires more complex mocking
        # For now, test individual components work together
        
        # 1. WebSocket client processes event
        client = TennisWebSocketClient()
        
        def capture_event(event):
            self.events_processed.append(event)
        
        client.add_event_callback(capture_event)
        
        # Simulate message processing
        await client._handle_message(json.dumps(test_event_data))
        
        # Verify event was captured
        self.assertEqual(len(self.events_processed), 1)
        
        # 2. Data validation
        validator = LiveDataValidator()
        report = validator.validate_live_event(self.events_processed[0])
        self.assertGreater(report.overall_score, 0.7)  # Should be high quality
        
        # 3. Feature extraction
        buffer = LiveDataBuffer()
        buffer.add_live_event(self.events_processed[0])
        features = buffer.get_match_features(12345)
        self.assertIn('match_id', features)


class TestPerformanceAndLoad(unittest.TestCase):
    """Test performance and load handling"""
    
    def test_high_volume_event_processing(self):
        """Test processing many events quickly"""
        processor = LiveDataProcessor()
        start_time = time.time()
        
        # Process 1000 events
        for i in range(1000):
            event_data = {
                'event_key': i,
                'event_first_player': f'Player {i}A',
                'event_second_player': f'Player {i}B',
                'tournament_name': f'Tournament {i}',
                'event_status': 'Live',
                'event_final_result': '1 - 0',
                'event_game_result': '3 - 2'
            }
            processor.process_live_event(event_data)
        
        processing_time = time.time() - start_time
        
        # Should process 1000 events in under 5 seconds
        self.assertLess(processing_time, 5.0)
        
        # Verify all events were processed
        self.assertEqual(len(processor.match_cache), 1000)
    
    def test_memory_management(self):
        """Test memory usage with large data sets"""
        buffer = LiveDataBuffer(max_buffer_size=100)
        
        # Add many events for single match
        for i in range(200):
            event = LiveMatchEvent(
                event_key=123, event_date="2024-01-15", event_time="14:30",
                first_player="Player 1", first_player_key=1001,
                second_player="Player 2", second_player_key=1002,
                final_result=f"{i//10} - {i//15}", game_result=f"{i%10} - {(i+1)%10}",
                serve="First Player", winner=None, status="Set 1",
                event_type="ATP", tournament_name="Test Tournament",
                tournament_key=5001, tournament_round="R1", tournament_season="2024",
                live=True, scores=[], point_by_point=[], statistics=[],
                timestamp=datetime.now()
            )
            buffer.add_live_event(event)
        
        # Buffer should not exceed max size
        self.assertLessEqual(len(buffer.match_buffers[123]), 100)


def run_websocket_integration_tests():
    """Run all WebSocket integration tests"""
    print("=" * 70)
    print("WEBSOCKET INTEGRATION TEST SUITE")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestWebSocketClient))
    suite.addTests(loader.loadTestsFromTestCase(TestLiveDataProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestMLPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictionEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestBettingEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationWorkflow))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceAndLoad))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_websocket_integration_tests()
    sys.exit(0 if success else 1)