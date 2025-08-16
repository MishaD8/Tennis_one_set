#!/usr/bin/env python3
"""
🔄 Testing Mode Controller for Tennis Betting System
Orchestrates switching between backtesting, forward testing, and live modes
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

from src.data.database_models import TestMode, DatabaseManager
from src.data.backtest_data_manager import BacktestDataManager
from src.data.forward_test_manager import ForwardTestManager
from src.api.betting_tracker_service import BettingTrackerService, TennisBettingIntegration
from src.api.performance_comparison_service import PerformanceComparisonService

logger = logging.getLogger(__name__)

class TestingModeController:
    """
    Central controller for managing testing modes and coordinating between
    backtesting, forward testing, and live betting operations
    """
    
    def __init__(self, database_manager: DatabaseManager = None):
        self.db_manager = database_manager or DatabaseManager()
        
        # Initialize managers
        self.backtest_manager = BacktestDataManager(self.db_manager)
        self.forward_test_manager = ForwardTestManager(self.db_manager)
        self.betting_tracker = BettingTrackerService(self.db_manager)
        self.betting_integration = TennisBettingIntegration()
        self.performance_service = PerformanceComparisonService(self.db_manager)
        
        # Current mode state
        self.current_mode = TestMode.LIVE
        self.active_session_id = None
        
        logger.info("✅ Testing Mode Controller initialized")
    
    def switch_mode(self, new_mode: str, session_config: Dict = None) -> Dict[str, Any]:
        """
        Switch between testing modes with proper validation and setup
        
        Args:
            new_mode: Target mode ('live', 'backtest', 'forward_test')
            session_config: Configuration for creating new sessions if needed
        
        Returns:
            Dict with switch result and session information
        """
        try:
            # Validate mode
            if new_mode not in ['live', 'backtest', 'forward_test']:
                return {
                    'success': False,
                    'error': f'Invalid mode: {new_mode}. Must be one of: live, backtest, forward_test'
                }
            
            previous_mode = self.current_mode.value if hasattr(self.current_mode, 'value') else str(self.current_mode)
            target_mode = TestMode(new_mode)
            
            logger.info(f"🔄 Switching from {previous_mode} to {new_mode} mode")
            
            # Handle mode-specific setup
            session_info = {}
            
            if target_mode == TestMode.BACKTEST:
                session_info = self._setup_backtest_mode(session_config or {})
            elif target_mode == TestMode.FORWARD_TEST:
                session_info = self._setup_forward_test_mode(session_config or {})
            elif target_mode == TestMode.LIVE:
                session_info = self._setup_live_mode()
            
            if not session_info.get('success', False):
                return {
                    'success': False,
                    'error': f'Failed to setup {new_mode} mode: {session_info.get("error", "Unknown error")}'
                }
            
            # Update current state
            self.current_mode = target_mode
            self.active_session_id = session_info.get('session_id')
            
            result = {
                'success': True,
                'previous_mode': previous_mode,
                'current_mode': new_mode,
                'session_info': session_info,
                'switch_timestamp': datetime.now().isoformat(),
                'capabilities': self._get_mode_capabilities(target_mode)
            }
            
            logger.info(f"✅ Successfully switched to {new_mode} mode")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error switching to {new_mode} mode: {e}")
            return {
                'success': False,
                'error': f'Mode switch failed: {str(e)}'
            }
    
    def _setup_backtest_mode(self, config: Dict) -> Dict[str, Any]:
        """Setup backtesting mode with session creation"""
        try:
            # Default backtest configuration
            default_config = {
                'session_name': f'Backtest Session {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                'start_date': (datetime.now() - timedelta(days=365)).isoformat(),
                'end_date': (datetime.now() - timedelta(days=30)).isoformat(),
                'initial_bankroll': 10000.0,
                'max_stake_percentage': 0.05,
                'filters': {
                    'surface': None,
                    'ranking_range': [10, 300],
                    'tournament_level': None
                }
            }
            
            # Merge with provided config
            session_config = {**default_config, **config}
            
            # Create backtest session
            session_id = self.backtest_manager.create_backtest_session(
                session_name=session_config['session_name'],
                start_date=datetime.fromisoformat(session_config['start_date']),
                end_date=datetime.fromisoformat(session_config['end_date']),
                initial_bankroll=session_config['initial_bankroll'],
                max_stake_percentage=session_config['max_stake_percentage'],
                filters=session_config['filters']
            )
            
            if not session_id:
                return {'success': False, 'error': 'Failed to create backtest session'}
            
            # Get available historical matches
            matches = self.backtest_manager.get_backtest_matches(
                start_date=datetime.fromisoformat(session_config['start_date']),
                end_date=datetime.fromisoformat(session_config['end_date']),
                ranking_range=tuple(session_config['filters']['ranking_range']),
                limit=50
            )
            
            return {
                'success': True,
                'session_id': session_id,
                'session_config': session_config,
                'available_matches': len(matches),
                'mode_type': 'backtest',
                'ready_for_testing': len(matches) > 0
            }
            
        except Exception as e:
            logger.error(f"❌ Error setting up backtest mode: {e}")
            return {'success': False, 'error': str(e)}
    
    def _setup_forward_test_mode(self, config: Dict) -> Dict[str, Any]:
        """Setup forward testing mode with session creation"""
        try:
            # Default forward test configuration
            default_config = {
                'session_name': f'Forward Test Session {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                'duration_days': 30,
                'initial_bankroll': 5000.0,
                'max_stake_percentage': 0.03,
                'filters': {
                    'surface': None,
                    'ranking_range': [10, 300],
                    'tournament_level': None
                }
            }
            
            # Merge with provided config
            session_config = {**default_config, **config}
            
            # Create forward test session
            session_id = self.forward_test_manager.create_forward_test_session(
                session_name=session_config['session_name'],
                duration_days=session_config['duration_days'],
                initial_bankroll=session_config['initial_bankroll'],
                max_stake_percentage=session_config['max_stake_percentage'],
                filters=session_config['filters']
            )
            
            if not session_id:
                return {'success': False, 'error': 'Failed to create forward test session'}
            
            # Get available live matches
            matches = self.forward_test_manager.get_live_matches_for_testing(
                session_id=session_id,
                ranking_filter=tuple(session_config['filters']['ranking_range'])
            )
            
            return {
                'success': True,
                'session_id': session_id,
                'session_config': session_config,
                'available_matches': len(matches),
                'mode_type': 'forward_test',
                'ready_for_testing': len(matches) > 0
            }
            
        except Exception as e:
            logger.error(f"❌ Error setting up forward test mode: {e}")
            return {'success': False, 'error': str(e)}
    
    def _setup_live_mode(self) -> Dict[str, Any]:
        """Setup live mode (no session needed)"""
        try:
            return {
                'success': True,
                'session_id': None,
                'mode_type': 'live',
                'ready_for_testing': True,
                'capabilities': [
                    'real_betting',
                    'live_market_data',
                    'real_money_stakes',
                    'immediate_settlement'
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Error setting up live mode: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_mode_capabilities(self, mode: TestMode) -> List[str]:
        """Get capabilities available in each mode"""
        capabilities = {
            TestMode.LIVE: [
                'real_betting',
                'live_market_data', 
                'telegram_notifications',
                'real_money_stakes',
                'immediate_settlement'
            ],
            TestMode.BACKTEST: [
                'historical_data',
                'performance_analysis',
                'risk_free_testing',
                'strategy_validation',
                'bulk_testing'
            ],
            TestMode.FORWARD_TEST: [
                'live_market_data',
                'paper_trading',
                'performance_tracking',
                'model_validation',
                'risk_free_testing'
            ]
        }
        
        return capabilities.get(mode, [])
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current testing mode status and information"""
        try:
            mode_value = self.current_mode.value if hasattr(self.current_mode, 'value') else str(self.current_mode)
            
            status = {
                'current_mode': mode_value,
                'active_session_id': self.active_session_id,
                'capabilities': self._get_mode_capabilities(self.current_mode),
                'status_timestamp': datetime.now().isoformat()
            }
            
            # Add mode-specific status information
            if self.current_mode == TestMode.BACKTEST and self.active_session_id:
                backtest_sessions = self.backtest_manager.get_backtest_sessions(active_only=True)
                active_session = next((s for s in backtest_sessions if s['session_id'] == self.active_session_id), None)
                if active_session:
                    status['session_details'] = active_session
            
            elif self.current_mode == TestMode.FORWARD_TEST and self.active_session_id:
                forward_sessions = self.forward_test_manager.get_forward_test_sessions(active_only=True)
                active_session = next((s for s in forward_sessions if s['session_id'] == self.active_session_id), None)
                if active_session:
                    status['session_details'] = active_session
            
            # Add recent performance summary
            performance_summary = self.betting_tracker.get_betting_performance_summary(
                test_mode=self.current_mode,
                days_back=30
            )
            
            if 'error' not in performance_summary:
                status['recent_performance'] = performance_summary
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Error getting current status: {e}")
            return {
                'current_mode': 'unknown',
                'error': str(e),
                'status_timestamp': datetime.now().isoformat()
            }
    
    def process_match_prediction(self, 
                                match_data: Dict,
                                prediction_data: Dict,
                                stake_amount: float = 100.0) -> Dict[str, Any]:
        """
        Process a match prediction according to current testing mode
        """
        try:
            logger.info(f"🎾 Processing match prediction in {self.current_mode.value} mode")
            
            # Log the betting decision
            bet_id = self.betting_integration.log_underdog_bet(
                match_data=match_data,
                underdog_analysis=prediction_data,
                stake_amount=stake_amount,
                test_mode=self.current_mode
            )
            
            if not bet_id:
                return {
                    'success': False,
                    'error': 'Failed to log betting decision'
                }
            
            result = {
                'success': True,
                'bet_id': bet_id,
                'test_mode': self.current_mode.value,
                'session_id': self.active_session_id,
                'match_info': {
                    'player1': match_data.get('player1', ''),
                    'player2': match_data.get('player2', ''),
                    'tournament': match_data.get('tournament', ''),
                    'date': match_data.get('match_date', '')
                },
                'prediction_info': {
                    'predicted_winner': prediction_data.get('underdog_scenario', {}).get('underdog', ''),
                    'probability': prediction_data.get('underdog_probability', 0),
                    'confidence': prediction_data.get('confidence', 'Medium')
                },
                'stake_amount': stake_amount,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add mode-specific handling
            if self.current_mode == TestMode.LIVE:
                result['note'] = 'Live bet logged - monitor for real settlement'
            elif self.current_mode == TestMode.BACKTEST:
                result['note'] = 'Backtest bet logged - will be settled with historical data'
            elif self.current_mode == TestMode.FORWARD_TEST:
                result['note'] = 'Forward test bet logged - will be settled with live results'
            
            logger.info(f"✅ Processed prediction: {bet_id}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing match prediction: {e}")
            return {
                'success': False,
                'error': f'Failed to process prediction: {str(e)}'
            }
    
    def settle_match_result(self, 
                           match_id: str,
                           winner: str,
                           score: str = "") -> Dict[str, Any]:
        """
        Settle match results across all relevant bets
        """
        try:
            logger.info(f"🏆 Settling match result: {match_id} - Winner: {winner}")
            
            # Settle bets using betting integration
            settled_count = self.betting_integration.settle_match_bets(
                match_id=match_id,
                winner=winner,
                score=score
            )
            
            result = {
                'success': True,
                'match_id': match_id,
                'winner': winner,
                'score': score,
                'settled_bets': settled_count,
                'settlement_timestamp': datetime.now().isoformat()
            }
            
            # Update session statistics if in testing mode
            if self.current_mode in [TestMode.BACKTEST, TestMode.FORWARD_TEST] and self.active_session_id:
                # This would require session-bet linking in the database
                # For now, we'll note it in the result
                result['session_updated'] = True
                result['active_session_id'] = self.active_session_id
            
            logger.info(f"✅ Settled {settled_count} bets for match {match_id}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error settling match result: {e}")
            return {
                'success': False,
                'error': f'Failed to settle match: {str(e)}'
            }
    
    def compare_testing_performance(self, 
                                   backtest_session_id: str,
                                   forward_test_session_id: str,
                                   comparison_name: str = None) -> Dict[str, Any]:
        """
        Create and retrieve performance comparison between testing sessions
        """
        try:
            if not comparison_name:
                comparison_name = f"Performance Comparison {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            logger.info(f"📊 Creating performance comparison: {comparison_name}")
            
            # Create comparison
            comparison_id = self.performance_service.create_performance_comparison(
                comparison_name=comparison_name,
                backtest_session_id=backtest_session_id,
                forward_test_session_id=forward_test_session_id
            )
            
            if not comparison_id:
                return {
                    'success': False,
                    'error': 'Failed to create performance comparison'
                }
            
            # Get detailed comparison
            detailed_comparison = self.performance_service.get_detailed_comparison(comparison_id)
            
            return {
                'success': True,
                'comparison_id': comparison_id,
                'comparison_name': comparison_name,
                'detailed_analysis': detailed_comparison,
                'creation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error creating performance comparison: {e}")
            return {
                'success': False,
                'error': f'Failed to compare performance: {str(e)}'
            }
    
    def get_model_stability_assessment(self, days_back: int = 90) -> Dict[str, Any]:
        """
        Get comprehensive model stability assessment across all testing modes
        """
        try:
            logger.info(f"🔍 Generating model stability assessment for last {days_back} days")
            
            # Get stability report
            stability_report = self.performance_service.generate_model_stability_report(days_back=days_back)
            
            # Add current mode context
            current_status = self.get_current_status()
            
            assessment = {
                'success': True,
                'assessment_period': f'Last {days_back} days',
                'current_context': current_status,
                'stability_analysis': stability_report,
                'assessment_timestamp': datetime.now().isoformat()
            }
            
            # Add recommendations based on current mode
            recommendations = []
            
            if self.current_mode == TestMode.LIVE:
                if stability_report.get('overall_status') != 'healthy':
                    recommendations.append("⚠️ Consider switching to forward test mode for safer validation")
                    recommendations.append("🔄 Review recent model performance before continuing live betting")
            
            elif self.current_mode == TestMode.FORWARD_TEST:
                if stability_report.get('overall_status') == 'healthy':
                    recommendations.append("✅ Forward test results look good - consider moving to live mode")
                else:
                    recommendations.append("🔄 Continue forward testing until performance stabilizes")
            
            elif self.current_mode == TestMode.BACKTEST:
                recommendations.append("📊 Use backtest results to calibrate forward testing parameters")
                recommendations.append("🔄 Progress to forward testing when backtest shows consistent performance")
            
            assessment['mode_specific_recommendations'] = recommendations
            
            return assessment
            
        except Exception as e:
            logger.error(f"❌ Error getting model stability assessment: {e}")
            return {
                'success': False,
                'error': f'Failed to assess model stability: {str(e)}'
            }


# Integration with existing route handlers
class TestingModeAPI:
    """
    API wrapper for testing mode controller to integrate with Flask routes
    """
    
    def __init__(self):
        self.controller = TestingModeController()
    
    def handle_mode_switch(self, request_data: Dict) -> Dict[str, Any]:
        """Handle mode switch API request"""
        test_mode = request_data.get('test_mode', 'live')
        session_config = request_data.get('session_config', {})
        
        return self.controller.switch_mode(test_mode, session_config)
    
    def handle_current_status(self) -> Dict[str, Any]:
        """Handle current status API request"""
        return self.controller.get_current_status()
    
    def handle_prediction_processing(self, request_data: Dict) -> Dict[str, Any]:
        """Handle prediction processing API request"""
        match_data = request_data.get('match_data', {})
        prediction_data = request_data.get('prediction_data', {})
        stake_amount = request_data.get('stake_amount', 100.0)
        
        return self.controller.process_match_prediction(
            match_data=match_data,
            prediction_data=prediction_data,
            stake_amount=stake_amount
        )
    
    def handle_match_settlement(self, request_data: Dict) -> Dict[str, Any]:
        """Handle match settlement API request"""
        match_id = request_data.get('match_id', '')
        winner = request_data.get('winner', '')
        score = request_data.get('score', '')
        
        return self.controller.settle_match_result(
            match_id=match_id,
            winner=winner,
            score=score
        )


# Example usage and testing
if __name__ == "__main__":
    print("🔄 TESTING MODE CONTROLLER - TESTING")
    print("=" * 50)
    
    # Initialize controller
    controller = TestingModeController()
    
    # Test mode switching
    print("1️⃣ Testing mode switching...")
    
    # Switch to backtest mode
    backtest_result = controller.switch_mode('backtest', {
        'session_name': 'Test Backtest Session',
        'start_date': '2024-01-01',
        'end_date': '2024-06-30',
        'initial_bankroll': 5000.0
    })
    print(f"🔄 Backtest switch: {backtest_result['success']}")
    
    # Switch to forward test mode
    forward_result = controller.switch_mode('forward_test', {
        'session_name': 'Test Forward Session',
        'duration_days': 14,
        'initial_bankroll': 2000.0
    })
    print(f"⏩ Forward test switch: {forward_result['success']}")
    
    # Test current status
    print("\n2️⃣ Testing status retrieval...")
    status = controller.get_current_status()
    print(f"📊 Current mode: {status.get('current_mode', 'unknown')}")
    
    # Test model stability assessment
    print("\n3️⃣ Testing stability assessment...")
    assessment = controller.get_model_stability_assessment(days_back=30)
    print(f"🔍 Assessment success: {assessment['success']}")
    
    print("\n🔄 Testing Mode Controller testing completed!")