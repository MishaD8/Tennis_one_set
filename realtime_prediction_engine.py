#!/usr/bin/env python3
"""
Real-Time Tennis Prediction Engine
Advanced prediction triggering system for live tennis matches with ML integration
"""

import os
import json
import time
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from queue import Queue, PriorityQueue, Empty
import pickle
from collections import defaultdict, deque
import numpy as np

# Import components
from websocket_tennis_client import LiveMatchEvent, TennisWebSocketClient
from realtime_ml_pipeline import RealTimeMLPipeline, MLPredictionResult
from realtime_data_validator import DataQualityMonitor, DataQualityReport
from api_tennis_integration import TennisMatch
from config import get_config

logger = logging.getLogger(__name__)


class PredictionTrigger(Enum):
    """Types of prediction triggers"""
    MATCH_START = "match_start"
    SCORE_CHANGE = "score_change"
    SET_CHANGE = "set_change"
    BREAK_POINT = "break_point"
    SET_POINT = "set_point"
    MATCH_POINT = "match_point"
    GAME_CHANGE = "game_change"
    MOMENTUM_SHIFT = "momentum_shift"
    PERIODIC = "periodic"
    MANUAL = "manual"
    ODDS_CHANGE = "odds_change"
    TIMEOUT = "timeout"
    INJURY = "injury"
    WEATHER = "weather"


class PredictionPriority(Enum):
    """Prediction priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


@dataclass
class PredictionTask:
    """Prediction task for priority queue"""
    priority: int
    timestamp: datetime
    match_id: int
    trigger: PredictionTrigger
    event_data: Dict[str, Any]
    context: Dict[str, Any]
    
    def __lt__(self, other):
        """For priority queue sorting (higher priority first)"""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.timestamp < other.timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['priority'] = self.priority
        data['timestamp'] = self.timestamp.isoformat()
        data['trigger'] = self.trigger.value
        return data


@dataclass
class PredictionConfig:
    """Configuration for prediction triggers"""
    enabled_triggers: List[PredictionTrigger]
    min_confidence_threshold: float
    max_predictions_per_match: int
    min_time_between_predictions: int  # seconds
    quality_threshold: float
    tournament_filters: List[str]
    player_filters: List[str]
    
    @classmethod
    def default(cls) -> 'PredictionConfig':
        return cls(
            enabled_triggers=[
                PredictionTrigger.MATCH_START,
                PredictionTrigger.SCORE_CHANGE,
                PredictionTrigger.SET_CHANGE,
                PredictionTrigger.BREAK_POINT,
                PredictionTrigger.PERIODIC
            ],
            min_confidence_threshold=0.6,
            max_predictions_per_match=20,
            min_time_between_predictions=120,  # 2 minutes
            quality_threshold=0.7,
            tournament_filters=[],
            player_filters=[]
        )


class TriggerDetector:
    """Detects when prediction triggers should fire"""
    
    def __init__(self):
        self.match_states: Dict[int, Dict[str, Any]] = {}
        self.last_predictions: Dict[int, datetime] = {}
        self.momentum_tracker = MomentumTracker()
    
    def analyze_event(self, event: LiveMatchEvent, quality_report: DataQualityReport) -> List[Tuple[PredictionTrigger, int, Dict[str, Any]]]:
        """
        Analyze live event and return triggered predictions with priorities
        Returns: List of (trigger_type, priority, context)
        """
        triggers = []
        match_id = event.event_key
        
        # Update match state
        previous_state = self.match_states.get(match_id, {})
        current_state = self._extract_match_state(event)
        self.match_states[match_id] = current_state
        
        # Skip if data quality is too low
        if quality_report.overall_score < 0.5:
            logger.warning(f"Skipping prediction triggers for match {match_id} due to low data quality")
            return triggers
        
        # Detect triggers
        triggers.extend(self._detect_match_start(event, previous_state, current_state))
        triggers.extend(self._detect_score_changes(event, previous_state, current_state))
        triggers.extend(self._detect_set_changes(event, previous_state, current_state))
        triggers.extend(self._detect_critical_points(event, previous_state, current_state))
        triggers.extend(self._detect_momentum_shifts(event, previous_state, current_state))
        triggers.extend(self._detect_periodic_triggers(event, match_id))
        
        return triggers
    
    def _extract_match_state(self, event: LiveMatchEvent) -> Dict[str, Any]:
        """Extract current match state from event"""
        try:
            final_score = event.final_result.replace(' ', '').split('-')
            game_score = event.game_result.replace(' ', '').split('-')
            
            return {
                'sets_p1': int(final_score[0]) if len(final_score) == 2 else 0,
                'sets_p2': int(final_score[1]) if len(final_score) == 2 else 0,
                'games_p1': int(game_score[0]) if len(game_score) == 2 else 0,
                'games_p2': int(game_score[1]) if len(game_score) == 2 else 0,
                'status': event.status,
                'serve': event.serve,
                'live': event.live,
                'timestamp': event.timestamp
            }
        except (ValueError, IndexError):
            return {
                'sets_p1': 0, 'sets_p2': 0,
                'games_p1': 0, 'games_p2': 0,
                'status': event.status,
                'serve': event.serve,
                'live': event.live,
                'timestamp': event.timestamp
            }
    
    def _detect_match_start(self, event: LiveMatchEvent, prev_state: Dict, curr_state: Dict) -> List[Tuple[PredictionTrigger, int, Dict[str, Any]]]:
        """Detect match start"""
        triggers = []
        
        # Match is starting if it's live and no previous state exists
        if not prev_state and curr_state['live']:
            triggers.append((
                PredictionTrigger.MATCH_START,
                PredictionPriority.HIGH.value,
                {
                    'reason': 'Match starting',
                    'players': f"{event.first_player} vs {event.second_player}",
                    'tournament': event.tournament_name
                }
            ))
        
        return triggers
    
    def _detect_score_changes(self, event: LiveMatchEvent, prev_state: Dict, curr_state: Dict) -> List[Tuple[PredictionTrigger, int, Dict[str, Any]]]:
        """Detect score changes"""
        triggers = []
        
        if not prev_state:
            return triggers
        
        # Game score change
        if (curr_state['games_p1'] != prev_state['games_p1'] or 
            curr_state['games_p2'] != prev_state['games_p2']):
            
            triggers.append((
                PredictionTrigger.GAME_CHANGE,
                PredictionPriority.NORMAL.value,
                {
                    'reason': 'Game completed',
                    'previous_games': f"{prev_state['games_p1']}-{prev_state['games_p2']}",
                    'current_games': f"{curr_state['games_p1']}-{curr_state['games_p2']}"
                }
            ))
        
        # Significant game advantage
        game_diff = abs(curr_state['games_p1'] - curr_state['games_p2'])
        if game_diff >= 3 and curr_state['live']:
            triggers.append((
                PredictionTrigger.SCORE_CHANGE,
                PredictionPriority.HIGH.value,
                {
                    'reason': 'Significant game advantage',
                    'game_difference': game_diff,
                    'leading_player': 'Player 1' if curr_state['games_p1'] > curr_state['games_p2'] else 'Player 2'
                }
            ))
        
        return triggers
    
    def _detect_set_changes(self, event: LiveMatchEvent, prev_state: Dict, curr_state: Dict) -> List[Tuple[PredictionTrigger, int, Dict[str, Any]]]:
        """Detect set changes"""
        triggers = []
        
        if not prev_state:
            return triggers
        
        # Set completed
        if (curr_state['sets_p1'] != prev_state['sets_p1'] or 
            curr_state['sets_p2'] != prev_state['sets_p2']):
            
            triggers.append((
                PredictionTrigger.SET_CHANGE,
                PredictionPriority.HIGH.value,
                {
                    'reason': 'Set completed',
                    'previous_sets': f"{prev_state['sets_p1']}-{prev_state['sets_p2']}",
                    'current_sets': f"{curr_state['sets_p1']}-{curr_state['sets_p2']}",
                    'set_winner': 'Player 1' if curr_state['sets_p1'] > prev_state['sets_p1'] else 'Player 2'
                }
            ))
        
        return triggers
    
    def _detect_critical_points(self, event: LiveMatchEvent, prev_state: Dict, curr_state: Dict) -> List[Tuple[PredictionTrigger, int, Dict[str, Any]]]:
        """Detect break points, set points, match points"""
        triggers = []
        
        # Analyze point-by-point data
        for point_data in event.point_by_point:
            for point in point_data.get('points', []):
                
                # Break point
                if point.get('break_point'):
                    triggers.append((
                        PredictionTrigger.BREAK_POINT,
                        PredictionPriority.URGENT.value,
                        {
                            'reason': 'Break point situation',
                            'serving_player': point_data.get('player_served', ''),
                            'point_score': point.get('score', '')
                        }
                    ))
                
                # Set point
                if point.get('set_point'):
                    triggers.append((
                        PredictionTrigger.SET_POINT,
                        PredictionPriority.URGENT.value,
                        {
                            'reason': 'Set point situation',
                            'point_score': point.get('score', ''),
                            'set_number': point_data.get('set_number', '')
                        }
                    ))
                
                # Match point
                if point.get('match_point'):
                    triggers.append((
                        PredictionTrigger.MATCH_POINT,
                        PredictionPriority.CRITICAL.value,
                        {
                            'reason': 'Match point situation',
                            'point_score': point.get('score', ''),
                            'set_number': point_data.get('set_number', '')
                        }
                    ))
        
        return triggers
    
    def _detect_momentum_shifts(self, event: LiveMatchEvent, prev_state: Dict, curr_state: Dict) -> List[Tuple[PredictionTrigger, int, Dict[str, Any]]]:
        """Detect momentum shifts in the match"""
        triggers = []
        
        # Update momentum tracker
        momentum_change = self.momentum_tracker.update_momentum(event, prev_state, curr_state)
        
        if abs(momentum_change) > 0.3:  # Significant momentum shift
            triggers.append((
                PredictionTrigger.MOMENTUM_SHIFT,
                PredictionPriority.HIGH.value,
                {
                    'reason': 'Significant momentum shift detected',
                    'momentum_change': momentum_change,
                    'direction': 'Player 1' if momentum_change > 0 else 'Player 2'
                }
            ))
        
        return triggers
    
    def _detect_periodic_triggers(self, event: LiveMatchEvent, match_id: int) -> List[Tuple[PredictionTrigger, int, Dict[str, Any]]]:
        """Detect periodic prediction triggers"""
        triggers = []
        
        if not event.live:
            return triggers
        
        now = datetime.now()
        last_prediction = self.last_predictions.get(match_id)
        
        # Trigger every 10 minutes for live matches
        if not last_prediction or now - last_prediction > timedelta(minutes=10):
            triggers.append((
                PredictionTrigger.PERIODIC,
                PredictionPriority.LOW.value,
                {
                    'reason': 'Periodic update for live match',
                    'minutes_since_last': (now - last_prediction).total_seconds() / 60 if last_prediction else 0
                }
            ))
            self.last_predictions[match_id] = now
        
        return triggers


class MomentumTracker:
    """Tracks match momentum based on recent events"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.match_momentum: Dict[int, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.game_history: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    
    def update_momentum(self, event: LiveMatchEvent, prev_state: Dict, curr_state: Dict) -> float:
        """
        Update momentum and return momentum change
        Returns: momentum change (-1 to 1, negative = Player 2 gaining, positive = Player 1 gaining)
        """
        match_id = event.event_key
        momentum_delta = 0.0
        
        if not prev_state:
            return momentum_delta
        
        # Game won/lost momentum
        if curr_state['games_p1'] > prev_state['games_p1']:
            momentum_delta += 0.2  # Player 1 won game
        elif curr_state['games_p2'] > prev_state['games_p2']:
            momentum_delta -= 0.2  # Player 2 won game
        
        # Set won/lost momentum
        if curr_state['sets_p1'] > prev_state['sets_p1']:
            momentum_delta += 0.5  # Player 1 won set
        elif curr_state['sets_p2'] > prev_state['sets_p2']:
            momentum_delta -= 0.5  # Player 2 won set
        
        # Break of serve momentum
        if self._detect_break_of_serve(event, prev_state, curr_state):
            # Determine who broke serve
            serving_player = prev_state.get('serve', '')
            if serving_player == 'First Player':
                momentum_delta -= 0.4  # Player 2 broke serve
            elif serving_player == 'Second Player':
                momentum_delta += 0.4  # Player 1 broke serve
        
        # Store momentum change
        self.match_momentum[match_id].append(momentum_delta)
        
        return momentum_delta
    
    def _detect_break_of_serve(self, event: LiveMatchEvent, prev_state: Dict, curr_state: Dict) -> bool:
        """Detect if a break of serve occurred"""
        # Simplified detection - would need more sophisticated logic
        games_changed = (curr_state['games_p1'] != prev_state['games_p1'] or 
                        curr_state['games_p2'] != prev_state['games_p2'])
        
        # Analyze point-by-point data for serve breaks
        for point_data in event.point_by_point:
            serve_lost = point_data.get('serve_lost')
            serve_winner = point_data.get('serve_winner')
            
            if serve_lost and serve_winner and serve_lost != serve_winner:
                return True
        
        return False
    
    def get_match_momentum(self, match_id: int) -> float:
        """Get current momentum for a match"""
        if match_id not in self.match_momentum:
            return 0.0
        
        momentum_history = list(self.match_momentum[match_id])
        if not momentum_history:
            return 0.0
        
        # Weight recent momentum changes more heavily
        weights = np.exp(np.linspace(-1, 0, len(momentum_history)))
        weighted_momentum = np.average(momentum_history, weights=weights)
        
        return float(weighted_momentum)


class PredictionEngine:
    """Main real-time prediction engine"""
    
    def __init__(self, config: PredictionConfig = None):
        self.config = config or PredictionConfig.default()
        
        # Components
        self.websocket_client = None
        self.ml_pipeline = None
        self.quality_monitor = DataQualityMonitor()
        self.trigger_detector = TriggerDetector()
        
        # Processing
        self.prediction_queue = PriorityQueue()
        self.result_cache: Dict[int, List[MLPredictionResult]] = defaultdict(list)
        
        # Control
        self.is_running = False
        self.stop_event = threading.Event()
        
        # Callbacks
        self.prediction_callbacks: List[Callable] = []
        self.trigger_callbacks: List[Callable] = []
        
        # Statistics
        self.stats = {
            'total_events': 0,
            'triggers_fired': 0,
            'predictions_generated': 0,
            'start_time': None,
            'trigger_counts': defaultdict(int),
            'prediction_performance': []
        }
        
        # Rate limiting
        self.match_prediction_counts: Dict[int, int] = defaultdict(int)
        self.last_match_predictions: Dict[int, datetime] = {}
    
    def initialize(self, api_key: str = None):
        """Initialize engine components"""
        logger.info("Initializing Real-Time Prediction Engine...")
        
        # Initialize ML pipeline
        self.ml_pipeline = RealTimeMLPipeline()
        self.ml_pipeline.initialize_websocket(api_key)
        
        # Add callbacks
        self.ml_pipeline.add_event_callback(self._handle_live_event)
        self.ml_pipeline.add_prediction_callback(self._handle_prediction_result)
        
        # Add quality alert callback
        self.quality_monitor.add_alert_callback(self._handle_quality_alert)
        
        logger.info("Prediction Engine initialized successfully")
    
    def add_prediction_callback(self, callback: Callable[[MLPredictionResult], None]):
        """Add callback for prediction results"""
        self.prediction_callbacks.append(callback)
    
    def add_trigger_callback(self, callback: Callable[[PredictionTrigger, Dict[str, Any]], None]):
        """Add callback for triggered predictions"""
        self.trigger_callbacks.append(callback)
    
    def _handle_live_event(self, event: LiveMatchEvent):
        """Handle live event from WebSocket"""
        try:
            self.stats['total_events'] += 1
            
            # Filter matches based on configuration
            if not self._should_process_match(event):
                return
            
            # Validate data quality
            quality_report = self.quality_monitor.process_event(event)
            
            # Skip if quality is below threshold
            if quality_report.overall_score < self.config.quality_threshold:
                logger.debug(f"Skipping match {event.event_key} due to low quality: {quality_report.overall_score:.2f}")
                return
            
            # Detect triggers
            triggers = self.trigger_detector.analyze_event(event, quality_report)
            
            # Process each trigger
            for trigger_type, priority, context in triggers:
                if trigger_type in self.config.enabled_triggers:
                    self._queue_prediction_task(event, trigger_type, priority, context)
            
        except Exception as e:
            logger.error(f"Error handling live event: {e}")
    
    def _should_process_match(self, event: LiveMatchEvent) -> bool:
        """Check if match should be processed based on filters"""
        
        # Tournament filter
        if self.config.tournament_filters:
            tournament_match = any(
                filter_term.lower() in event.tournament_name.lower()
                for filter_term in self.config.tournament_filters
            )
            if not tournament_match:
                return False
        
        # Player filter
        if self.config.player_filters:
            player_match = any(
                filter_term.lower() in event.first_player.lower() or
                filter_term.lower() in event.second_player.lower()
                for filter_term in self.config.player_filters
            )
            if not player_match:
                return False
        
        # Rate limiting per match
        match_id = event.event_key
        now = datetime.now()
        
        # Check prediction count limit
        if self.match_prediction_counts[match_id] >= self.config.max_predictions_per_match:
            return False
        
        # Check time between predictions
        last_prediction = self.last_match_predictions.get(match_id)
        if last_prediction:
            time_diff = (now - last_prediction).total_seconds()
            if time_diff < self.config.min_time_between_predictions:
                return False
        
        return True
    
    def _queue_prediction_task(self, event: LiveMatchEvent, trigger: PredictionTrigger, priority: int, context: Dict[str, Any]):
        """Queue a prediction task"""
        try:
            task = PredictionTask(
                priority=priority,
                timestamp=datetime.now(),
                match_id=event.event_key,
                trigger=trigger,
                event_data={
                    'player1': event.first_player,
                    'player2': event.second_player,
                    'tournament': event.tournament_name,
                    'status': event.status,
                    'score': event.final_result,
                    'live': event.live
                },
                context=context
            )
            
            self.prediction_queue.put(task)
            self.stats['triggers_fired'] += 1
            self.stats['trigger_counts'][trigger.value] += 1
            
            # Update prediction tracking
            self.match_prediction_counts[event.event_key] += 1
            self.last_match_predictions[event.event_key] = datetime.now()
            
            # Notify trigger callbacks
            for callback in self.trigger_callbacks:
                try:
                    callback(trigger, context)
                except Exception as e:
                    logger.error(f"Error in trigger callback: {e}")
            
            logger.info(f"Queued prediction task: {trigger.value} for match {event.event_key}")
            
        except Exception as e:
            logger.error(f"Error queuing prediction task: {e}")
    
    def _handle_prediction_result(self, result: MLPredictionResult):
        """Handle ML prediction result"""
        try:
            self.stats['predictions_generated'] += 1
            
            # Cache result
            self.result_cache[result.match_id].append(result)
            
            # Track performance
            self.stats['prediction_performance'].append({
                'timestamp': result.timestamp,
                'confidence': result.confidence,
                'processing_time': result.processing_time,
                'model_used': result.model_used
            })
            
            # Notify callbacks
            for callback in self.prediction_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Error in prediction callback: {e}")
            
            logger.info(f"Generated prediction for match {result.match_id} with confidence {result.confidence:.2f}")
            
        except Exception as e:
            logger.error(f"Error handling prediction result: {e}")
    
    def _handle_quality_alert(self, alert_type: str, data: Dict[str, Any]):
        """Handle data quality alerts"""
        logger.warning(f"Data Quality Alert: {alert_type} - {data}")
    
    def _prediction_worker(self):
        """Worker thread for processing prediction tasks"""
        while not self.stop_event.is_set():
            try:
                # Get highest priority task
                task = self.prediction_queue.get(timeout=1)
                
                # Process task through ML pipeline
                # This would trigger the ML pipeline's prediction process
                logger.debug(f"Processing prediction task: {task.trigger.value} for match {task.match_id}")
                
                self.prediction_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in prediction worker: {e}")
    
    def start(self, api_key: str = None):
        """Start the prediction engine"""
        logger.info("Starting Real-Time Prediction Engine...")
        
        if not self.ml_pipeline:
            self.initialize(api_key)
        
        self.is_running = True
        self.stop_event.clear()
        self.stats['start_time'] = datetime.now()
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._prediction_worker, daemon=True)
        self.worker_thread.start()
        
        # Start ML pipeline
        self.ml_pipeline.start(api_key)
        
        logger.info("Real-Time Prediction Engine started successfully")
    
    def stop(self):
        """Stop the prediction engine"""
        logger.info("Stopping Real-Time Prediction Engine...")
        
        self.is_running = False
        self.stop_event.set()
        
        # Stop ML pipeline
        if self.ml_pipeline:
            self.ml_pipeline.stop()
        
        logger.info("Real-Time Prediction Engine stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        stats = self.stats.copy()
        
        if self.ml_pipeline:
            stats['ml_pipeline'] = self.ml_pipeline.get_stats()
        
        stats['quality_monitor'] = self.quality_monitor.get_quality_summary()
        stats['active_matches'] = len(self.result_cache)
        stats['queue_size'] = self.prediction_queue.qsize()
        
        # Calculate performance metrics
        if self.stats['prediction_performance']:
            performances = self.stats['prediction_performance']
            stats['avg_confidence'] = np.mean([p['confidence'] for p in performances])
            stats['avg_processing_time'] = np.mean([p['processing_time'] for p in performances])
        
        if self.stats['start_time']:
            uptime = datetime.now() - self.stats['start_time']
            stats['uptime_seconds'] = uptime.total_seconds()
        
        return stats
    
    def get_match_predictions(self, match_id: int) -> List[MLPredictionResult]:
        """Get all predictions for a specific match"""
        return self.result_cache.get(match_id, [])
    
    def get_recent_predictions(self, hours: int = 1) -> List[MLPredictionResult]:
        """Get recent predictions from all matches"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_predictions = []
        
        for predictions in self.result_cache.values():
            for prediction in predictions:
                if prediction.timestamp > cutoff_time:
                    recent_predictions.append(prediction)
        
        return sorted(recent_predictions, key=lambda p: p.timestamp, reverse=True)
    
    def update_config(self, new_config: PredictionConfig):
        """Update engine configuration"""
        self.config = new_config
        logger.info("Prediction engine configuration updated")
    
    def force_prediction(self, match_id: int, context: Dict[str, Any] = None):
        """Force a manual prediction for a specific match"""
        task = PredictionTask(
            priority=PredictionPriority.URGENT.value,
            timestamp=datetime.now(),
            match_id=match_id,
            trigger=PredictionTrigger.MANUAL,
            event_data={'match_id': match_id},
            context=context or {'reason': 'Manual trigger'}
        )
        
        self.prediction_queue.put(task)
        logger.info(f"Forced prediction for match {match_id}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create custom configuration
    config = PredictionConfig(
        enabled_triggers=[
            PredictionTrigger.MATCH_START,
            PredictionTrigger.SCORE_CHANGE,
            PredictionTrigger.SET_CHANGE,
            PredictionTrigger.BREAK_POINT,
            PredictionTrigger.MOMENTUM_SHIFT
        ],
        min_confidence_threshold=0.65,
        max_predictions_per_match=15,
        min_time_between_predictions=90,
        quality_threshold=0.75,
        tournament_filters=['ATP', 'WTA'],
        player_filters=[]
    )
    
    # Create engine
    engine = PredictionEngine(config)
    
    # Add callbacks
    def handle_prediction(result: MLPredictionResult):
        print(f"NEW PREDICTION: {result.prediction['winner']} (confidence: {result.confidence:.2f})")
    
    def handle_trigger(trigger: PredictionTrigger, context: Dict[str, Any]):
        print(f"TRIGGER FIRED: {trigger.value} - {context.get('reason', 'No reason')}")
    
    engine.add_prediction_callback(handle_prediction)
    engine.add_trigger_callback(handle_trigger)
    
    try:
        # Start engine
        engine.start()
        
        while True:
            time.sleep(30)
            stats = engine.get_stats()
            print(f"\nEngine Stats:")
            print(f"  Events: {stats['total_events']}")
            print(f"  Triggers: {stats['triggers_fired']}")
            print(f"  Predictions: {stats['predictions_generated']}")
            print(f"  Queue Size: {stats['queue_size']}")
            
    except KeyboardInterrupt:
        engine.stop()