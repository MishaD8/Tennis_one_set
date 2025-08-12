#!/usr/bin/env python3
"""
WebSocket Tennis System Demonstration
Complete demonstration of real-time tennis betting system capabilities
"""

import os
import sys
import json
import time
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import signal
from dataclasses import asdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all components
from websocket_tennis_client import (
    TennisWebSocketClient, LiveMatchEvent, WSConnectionState, WebSocketManager
)
from realtime_ml_pipeline import RealTimeMLPipeline, MLPredictionResult
from realtime_data_validator import DataQualityMonitor, ValidationSeverity
from realtime_prediction_engine import (
    PredictionEngine, PredictionConfig, PredictionTrigger
)
from automated_betting_engine import (
    AutomatedBettingEngine, RiskManagementConfig, BetOpportunity, BetOrder
)
from config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('websocket_demo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class WebSocketSystemDemo:
    """Comprehensive demonstration of the WebSocket tennis system"""
    
    def __init__(self, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode
        self.config = get_config()
        
        # System components
        self.websocket_manager = None
        self.ml_pipeline = None
        self.prediction_engine = None
        self.betting_engine = None
        
        # Demo tracking
        self.demo_start_time = None
        self.events_received = 0
        self.predictions_made = 0
        self.betting_opportunities = 0
        self.bets_placed = 0
        
        # Event storage for analysis
        self.live_events = []
        self.ml_predictions = []
        self.betting_opportunities_list = []
        self.placed_bets = []
        
        # Control
        self.running = False
        self.stop_event = threading.Event()
    
    def initialize_system(self):
        """Initialize all system components"""
        logger.info("Initializing WebSocket Tennis System...")
        
        try:
            # 1. Initialize WebSocket Manager
            self.websocket_manager = WebSocketManager()
            
            # 2. Initialize ML Pipeline
            self.ml_pipeline = RealTimeMLPipeline()
            
            # 3. Initialize Prediction Engine
            prediction_config = PredictionConfig(
                enabled_triggers=[
                    PredictionTrigger.MATCH_START,
                    PredictionTrigger.SCORE_CHANGE,
                    PredictionTrigger.SET_CHANGE,
                    PredictionTrigger.BREAK_POINT,
                    PredictionTrigger.MOMENTUM_SHIFT,
                    PredictionTrigger.PERIODIC
                ],
                min_confidence_threshold=0.6,
                max_predictions_per_match=15,
                min_time_between_predictions=60,
                quality_threshold=0.7,
                tournament_filters=[],  # No filters for demo
                player_filters=[]
            )
            self.prediction_engine = PredictionEngine(prediction_config)
            
            # 4. Initialize Betting Engine
            risk_config = RiskManagementConfig.moderate()
            self.betting_engine = AutomatedBettingEngine(risk_config, initial_bankroll=1000.0)
            
            # 5. Setup connections between components
            self._setup_component_connections()
            
            # 6. Add demo callbacks
            self._setup_demo_callbacks()
            
            logger.info("System initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            return False
    
    def _setup_component_connections(self):
        """Setup connections between system components"""
        
        # Connect prediction engine to ML pipeline
        self.prediction_engine.initialize()
        
        # Connect betting engine to prediction engine
        self.betting_engine.initialize(self.prediction_engine)
    
    def _setup_demo_callbacks(self):
        """Setup callbacks for demonstration tracking"""
        
        # WebSocket event callback
        def handle_live_event(client_id: str, event: LiveMatchEvent):
            self.events_received += 1
            self.live_events.append(event)
            self._log_live_event(event)
        
        # ML prediction callback
        def handle_prediction(result: MLPredictionResult):
            self.predictions_made += 1
            self.ml_predictions.append(result)
            self._log_prediction(result)
        
        # Betting opportunity callback
        def handle_opportunity(opportunity: BetOpportunity):
            self.betting_opportunities += 1
            self.betting_opportunities_list.append(opportunity)
            self._log_betting_opportunity(opportunity)
        
        # Bet placement callback
        def handle_bet_placed(bet: BetOrder):
            self.bets_placed += 1
            self.placed_bets.append(bet)
            self._log_bet_placed(bet)
        
        # Trigger callback
        def handle_trigger(trigger: PredictionTrigger, context: Dict[str, Any]):
            self._log_prediction_trigger(trigger, context)
        
        # Add callbacks to components
        if hasattr(self.websocket_manager, 'event_router'):
            self.websocket_manager.event_router.add_handler('all', handle_live_event)
        
        if self.prediction_engine:
            self.prediction_engine.add_prediction_callback(handle_prediction)
            self.prediction_engine.add_trigger_callback(handle_trigger)
        
        if self.betting_engine:
            self.betting_engine.add_opportunity_callback(handle_opportunity)
            self.betting_engine.add_bet_callback(handle_bet_placed)
    
    def _log_live_event(self, event: LiveMatchEvent):
        """Log live event details"""
        logger.info(f"LIVE EVENT: {event.first_player} vs {event.second_player} - "
                   f"Score: {event.final_result} - Status: {event.status}")
    
    def _log_prediction(self, result: MLPredictionResult):
        """Log ML prediction details"""
        prediction = result.prediction
        logger.info(f"ML PREDICTION: Match {result.match_id} - "
                   f"Winner: {prediction.get('winner', 'Unknown')} - "
                   f"Confidence: {result.confidence:.2f} - "
                   f"Model: {result.model_used}")
    
    def _log_betting_opportunity(self, opportunity: BetOpportunity):
        """Log betting opportunity details"""
        logger.info(f"BETTING OPPORTUNITY: {opportunity.selection} - "
                   f"Edge: {opportunity.edge:.3f} - "
                   f"Recommended Stake: {opportunity.recommended_stake:.2f}€ - "
                   f"Odds: {opportunity.market_odds}")
    
    def _log_bet_placed(self, bet: BetOrder):
        """Log bet placement details"""
        logger.info(f"BET PLACED: {bet.stake:.2f}€ on {bet.selection} - "
                   f"Odds: {bet.odds} - "
                   f"Potential Payout: {bet.potential_payout:.2f}€")
    
    def _log_prediction_trigger(self, trigger: PredictionTrigger, context: Dict[str, Any]):
        """Log prediction trigger details"""
        logger.info(f"PREDICTION TRIGGER: {trigger.value} - "
                   f"Reason: {context.get('reason', 'No reason provided')}")
    
    def start_simulation_mode(self):
        """Start system in simulation mode with generated events"""
        logger.info("Starting system in SIMULATION MODE...")
        
        self.demo_start_time = datetime.now()
        self.running = True
        
        # Start all system components
        if self.prediction_engine:
            threading.Thread(target=self._start_prediction_engine, daemon=True).start()
        
        if self.betting_engine:
            threading.Thread(target=self._start_betting_engine, daemon=True).start()
        
        # Start event simulation
        threading.Thread(target=self._simulate_live_events, daemon=True).start()
        
        # Start monitoring
        threading.Thread(target=self._monitor_system, daemon=True).start()
        
        logger.info("Simulation mode started successfully")
    
    def start_live_mode(self, api_key: str):
        """Start system in live mode with real WebSocket connection"""
        logger.info("Starting system in LIVE MODE...")
        
        if not api_key:
            logger.error("API key required for live mode")
            return False
        
        self.demo_start_time = datetime.now()
        self.running = True
        
        try:
            # Create WebSocket client
            main_client = self.websocket_manager.create_client('main', api_key=api_key)
            
            # Start all components
            if self.prediction_engine:
                self.prediction_engine.start(api_key)
            
            if self.betting_engine:
                self.betting_engine.start()
            
            # Start WebSocket clients
            self.websocket_manager.start_all_clients()
            
            # Start monitoring
            threading.Thread(target=self._monitor_system, daemon=True).start()
            
            logger.info("Live mode started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start live mode: {e}")
            return False
    
    def _start_prediction_engine(self):
        """Start prediction engine"""
        try:
            if not self.simulation_mode:
                self.prediction_engine.start()
        except Exception as e:
            logger.error(f"Error starting prediction engine: {e}")
    
    def _start_betting_engine(self):
        """Start betting engine"""
        try:
            self.betting_engine.start()
        except Exception as e:
            logger.error(f"Error starting betting engine: {e}")
    
    def _simulate_live_events(self):
        """Simulate live tennis events for demonstration"""
        logger.info("Starting event simulation...")
        
        match_scenarios = [
            {
                'event_key': 12345,
                'first_player': 'Rafael Nadal',
                'second_player': 'Novak Djokovic',
                'tournament': 'Miami Open',
                'tournament_key': 5001
            },
            {
                'event_key': 12346,
                'first_player': 'Carlos Alcaraz',
                'second_player': 'Daniil Medvedev',
                'tournament': 'Indian Wells',
                'tournament_key': 5002
            },
            {
                'event_key': 12347,
                'first_player': 'Stefanos Tsitsipas',
                'second_player': 'Alexander Zverev',
                'tournament': 'Monte Carlo Masters',
                'tournament_key': 5003
            }
        ]
        
        event_count = 0
        
        while self.running and not self.stop_event.is_set():
            try:
                for scenario in match_scenarios:
                    if not self.running:
                        break
                    
                    # Generate event progression
                    events = self._generate_match_progression(scenario, event_count)
                    
                    for event_data in events:
                        if not self.running:
                            break
                        
                        # Create live event
                        event = LiveMatchEvent.from_api_data(event_data)
                        
                        # Process through pipeline (simulation)
                        if self.ml_pipeline:
                            self.ml_pipeline._handle_live_event(event)
                        
                        # Small delay between events
                        time.sleep(2)
                        event_count += 1
                
                # Longer delay between match scenarios
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in event simulation: {e}")
                time.sleep(5)
    
    def _generate_match_progression(self, scenario: Dict[str, Any], base_count: int) -> List[Dict[str, Any]]:
        """Generate a realistic match progression"""
        events = []
        
        # Match start
        events.append({
            'event_key': scenario['event_key'],
            'event_date': datetime.now().strftime('%Y-%m-%d'),
            'event_time': datetime.now().strftime('%H:%M'),
            'event_first_player': scenario['first_player'],
            'first_player_key': 1001 + base_count,
            'event_second_player': scenario['second_player'],
            'second_player_key': 2001 + base_count,
            'event_final_result': '0 - 0',
            'event_game_result': '0 - 0',
            'event_serve': 'First Player',
            'event_winner': None,
            'event_status': 'Set 1',
            'event_type_type': 'ATP Masters',
            'tournament_name': scenario['tournament'],
            'tournament_key': scenario['tournament_key'],
            'tournament_round': 'Quarterfinals',
            'tournament_season': '2024',
            'event_live': '1',
            'scores': [],
            'pointbypoint': [],
            'statistics': []
        })
        
        # Game progressions
        game_scores = [
            ('1 - 0', '1 - 0'), ('1 - 0', '2 - 0'), ('1 - 0', '3 - 1'),
            ('1 - 0', '4 - 2'), ('1 - 0', '5 - 3'), ('1 - 0', '6 - 4')
        ]
        
        for set_score, game_score in game_scores:
            events.append({
                **events[0],  # Copy base event
                'event_final_result': set_score,
                'event_game_result': game_score,
                'scores': [{'score_first': set_score.split(' - ')[0], 
                           'score_second': set_score.split(' - ')[1], 
                           'score_set': '1'}],
                'pointbypoint': [{
                    'set_number': 'Set 1',
                    'number_game': str(len([s for s in game_scores[:game_scores.index((set_score, game_score))+1]]),
                    'player_served': 'First Player' if len(events) % 2 == 0 else 'Second Player',
                    'serve_winner': 'First Player' if 'First' in game_score else 'Second Player',
                    'points': []
                }]
            })
        
        return events
    
    def _monitor_system(self):
        """Monitor system performance and display statistics"""
        logger.info("Starting system monitoring...")
        
        while self.running and not self.stop_event.is_set():
            try:
                time.sleep(30)  # Update every 30 seconds
                self._display_system_status()
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
    
    def _display_system_status(self):
        """Display current system status"""
        uptime = datetime.now() - self.demo_start_time if self.demo_start_time else timedelta(0)
        
        print("\n" + "="*80)
        print("WEBSOCKET TENNIS SYSTEM STATUS")
        print("="*80)
        print(f"Uptime: {uptime}")
        print(f"Mode: {'SIMULATION' if self.simulation_mode else 'LIVE'}")
        print(f"Events Received: {self.events_received}")
        print(f"ML Predictions: {self.predictions_made}")
        print(f"Betting Opportunities: {self.betting_opportunities}")
        print(f"Bets Placed: {self.bets_placed}")
        
        # Component status
        if self.prediction_engine:
            pred_stats = self.prediction_engine.get_stats()
            print(f"\nPrediction Engine:")
            print(f"  - Queue Size: {pred_stats.get('queue_size', 0)}")
            print(f"  - Triggers Fired: {pred_stats.get('triggers_fired', 0)}")
        
        if self.betting_engine:
            bet_stats = self.betting_engine.get_stats()
            risk_summary = bet_stats.get('risk_summary', {})
            print(f"\nBetting Engine:")
            print(f"  - Active Bets: {bet_stats.get('active_bets', 0)}")
            print(f"  - Win Rate: {bet_stats.get('win_rate', 0):.1f}%")
            print(f"  - Current Bankroll: {risk_summary.get('current_bankroll', 0):.2f}€")
            print(f"  - P&L: {risk_summary.get('profit_loss', 0):.2f}€")
        
        print("="*80)
    
    def stop_system(self):
        """Stop all system components"""
        logger.info("Stopping WebSocket Tennis System...")
        
        self.running = False
        self.stop_event.set()
        
        # Stop components
        if self.websocket_manager:
            self.websocket_manager.stop_all_clients()
        
        if self.prediction_engine:
            self.prediction_engine.stop()
        
        if self.betting_engine:
            self.betting_engine.stop()
        
        if self.ml_pipeline:
            self.ml_pipeline.stop()
        
        logger.info("System stopped successfully")
    
    def generate_demo_report(self) -> Dict[str, Any]:
        """Generate comprehensive demo report"""
        uptime = datetime.now() - self.demo_start_time if self.demo_start_time else timedelta(0)
        
        report = {
            'demo_summary': {
                'start_time': self.demo_start_time.isoformat() if self.demo_start_time else None,
                'uptime_seconds': uptime.total_seconds(),
                'mode': 'simulation' if self.simulation_mode else 'live',
                'events_received': self.events_received,
                'predictions_made': self.predictions_made,
                'betting_opportunities': self.betting_opportunities,
                'bets_placed': self.bets_placed
            },
            'performance_metrics': {
                'events_per_minute': (self.events_received / max(uptime.total_seconds() / 60, 1)),
                'predictions_per_hour': (self.predictions_made / max(uptime.total_seconds() / 3600, 1)),
                'opportunity_conversion_rate': (self.bets_placed / max(self.betting_opportunities, 1)) * 100
            },
            'system_components': {}
        }
        
        # Add component statistics
        if self.prediction_engine:
            report['system_components']['prediction_engine'] = self.prediction_engine.get_stats()
        
        if self.betting_engine:
            report['system_components']['betting_engine'] = self.betting_engine.get_stats()
        
        if self.websocket_manager:
            report['system_components']['websocket_manager'] = self.websocket_manager.get_aggregated_stats()
        
        # Add sample data
        report['sample_data'] = {
            'recent_events': [event.to_dict() for event in self.live_events[-5:]],
            'recent_predictions': [pred.to_dict() for pred in self.ml_predictions[-5:]],
            'recent_opportunities': [opp.to_dict() for opp in self.betting_opportunities_list[-5:]],
            'recent_bets': [bet.to_dict() for bet in self.placed_bets[-5:]]
        }
        
        return report
    
    def save_demo_report(self, filename: str = None):
        """Save demo report to file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"websocket_demo_report_{timestamp}.json"
        
        report = self.generate_demo_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Demo report saved to {filename}")
        return filename


def signal_handler(signum, frame, demo: WebSocketSystemDemo):
    """Handle shutdown signals gracefully"""
    logger.info("Received shutdown signal, stopping demo...")
    demo.stop_system()
    
    # Generate final report
    report_file = demo.save_demo_report()
    logger.info(f"Final report saved to: {report_file}")
    
    sys.exit(0)


def main():
    """Main demonstration function"""
    print("="*80)
    print("WEBSOCKET TENNIS SYSTEM DEMONSTRATION")
    print("="*80)
    print()
    print("This demonstration showcases the complete real-time tennis betting system:")
    print("1. WebSocket live data streaming")
    print("2. Real-time ML prediction pipeline")
    print("3. Data quality validation")
    print("4. Automated betting engine")
    print("5. Risk management system")
    print()
    
    # Get configuration
    config = get_config()
    api_key = config.API_TENNIS_KEY
    
    # Choose mode
    if api_key:
        mode = input("Choose mode - (L)ive with real data or (S)imulation? [S]: ").strip().upper()
        simulation_mode = mode != 'L'
    else:
        print("No API key configured - running in simulation mode")
        simulation_mode = True
    
    # Duration
    try:
        duration = int(input("Demo duration in minutes [10]: ") or "10")
    except ValueError:
        duration = 10
    
    print(f"\nStarting demonstration in {'SIMULATION' if simulation_mode else 'LIVE'} mode for {duration} minutes...")
    print("Press Ctrl+C to stop early and generate report")
    print()
    
    # Create and initialize demo
    demo = WebSocketSystemDemo(simulation_mode=simulation_mode)
    
    if not demo.initialize_system():
        print("Failed to initialize system. Exiting.")
        return 1
    
    # Setup signal handler
    signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, demo))
    signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, demo))
    
    try:
        # Start system
        if simulation_mode:
            demo.start_simulation_mode()
        else:
            if not demo.start_live_mode(api_key):
                print("Failed to start live mode. Exiting.")
                return 1
        
        # Run for specified duration
        start_time = time.time()
        while time.time() - start_time < duration * 60:
            time.sleep(1)
        
        # Stop system
        demo.stop_system()
        
        # Generate and save final report
        report_file = demo.save_demo_report()
        
        print(f"\nDemo completed successfully!")
        print(f"Final report saved to: {report_file}")
        
        # Display final statistics
        report = demo.generate_demo_report()
        summary = report['demo_summary']
        performance = report['performance_metrics']
        
        print(f"\nFINAL STATISTICS:")
        print(f"  Events Processed: {summary['events_received']}")
        print(f"  ML Predictions: {summary['predictions_made']}")
        print(f"  Betting Opportunities: {summary['betting_opportunities']}")
        print(f"  Bets Placed: {summary['bets_placed']}")
        print(f"  Events/min: {performance['events_per_minute']:.1f}")
        print(f"  Predictions/hour: {performance['predictions_per_hour']:.1f}")
        print(f"  Opportunity Conversion: {performance['opportunity_conversion_rate']:.1f}%")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        demo.stop_system()
        report_file = demo.save_demo_report()
        print(f"Report saved to: {report_file}")
        return 0
    
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        demo.stop_system()
        return 1


if __name__ == "__main__":
    sys.exit(main())