#!/usr/bin/env python3
"""
Real-Time ML Pipeline for Tennis Prediction System
Processes live WebSocket data and triggers ML predictions for automated betting
"""

import os
import json
import time
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from queue import Queue, Empty
import pandas as pd
import numpy as np
from collections import defaultdict, deque

# Import WebSocket client and existing components
from websocket_tennis_client import TennisWebSocketClient, LiveMatchEvent, WSConnectionState
from api_tennis_integration import TennisMatch, TennisPlayer
from api_ml_integration import APITennisMLIntegration
from config import get_config

# Import ML components
try:
    from enhanced_ml_integration import EnhancedMLOrchestrator
    ML_ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ML_ORCHESTRATOR_AVAILABLE = False

try:
    from second_set_prediction_service import SecondSetPredictionService
    SECOND_SET_AVAILABLE = True
except ImportError:
    SECOND_SET_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MLPredictionRequest:
    """ML prediction request data model"""
    match_id: int
    player1: str
    player2: str
    tournament: str
    surface: str
    live_data: Dict[str, Any]
    timestamp: datetime
    priority: int = 1  # Higher number = higher priority
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass 
class MLPredictionResult:
    """ML prediction result data model"""
    match_id: int
    prediction: Dict[str, Any]
    confidence: float
    model_used: str
    processing_time: float
    timestamp: datetime
    live_context: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class LiveDataBuffer:
    """Buffers and processes live match data for ML feature engineering"""
    
    def __init__(self, max_buffer_size: int = 1000):
        self.max_buffer_size = max_buffer_size
        self.match_buffers: Dict[int, deque] = defaultdict(lambda: deque(maxlen=max_buffer_size))
        self.match_metadata: Dict[int, Dict[str, Any]] = {}
        self.last_update: Dict[int, datetime] = {}
    
    def add_live_event(self, event: LiveMatchEvent):
        """Add live event to buffer"""
        match_id = event.event_key
        
        # Add event to buffer
        self.match_buffers[match_id].append(event)
        self.last_update[match_id] = datetime.now()
        
        # Update metadata
        if match_id not in self.match_metadata:
            self.match_metadata[match_id] = {
                'first_player': event.first_player,
                'second_player': event.second_player,
                'tournament': event.tournament_name,
                'tournament_key': event.tournament_key,
                'event_type': event.event_type,
                'first_seen': datetime.now()
            }
    
    def get_match_features(self, match_id: int) -> Dict[str, Any]:
        """Extract ML features from buffered match data"""
        if match_id not in self.match_buffers:
            return {}
        
        events = list(self.match_buffers[match_id])
        if not events:
            return {}
        
        latest_event = events[-1]
        metadata = self.match_metadata.get(match_id, {})
        
        # Calculate features from live data
        features = {
            'match_id': match_id,
            'player1': metadata.get('first_player', ''),
            'player2': metadata.get('second_player', ''),
            'tournament': metadata.get('tournament', ''),
            'surface': self._determine_surface(metadata.get('tournament', '')),
            'current_score': latest_event.final_result,
            'current_game_score': latest_event.game_result,
            'current_set': self._extract_current_set(latest_event.status),
            'serving_player': latest_event.serve,
            'match_duration_minutes': self._calculate_match_duration(events),
            'total_games_played': self._count_total_games(events),
            'momentum_indicator': self._calculate_momentum(events),
            'break_points_converted': self._analyze_break_points(events),
            'service_dominance': self._analyze_service_stats(events),
            'set_score_history': self._extract_set_scores(latest_event.scores),
            'last_update': latest_event.timestamp.isoformat(),
            'match_status': latest_event.status,
            'is_live': latest_event.live
        }
        
        return features
    
    def _determine_surface(self, tournament_name: str) -> str:
        """Determine court surface from tournament name"""
        if not tournament_name:
            return 'Hard'
        
        tournament_lower = tournament_name.lower()
        if any(keyword in tournament_lower for keyword in ['french', 'roland garros', 'monte carlo', 'rome', 'madrid']):
            return 'Clay'
        elif any(keyword in tournament_lower for keyword in ['wimbledon', 'grass']):
            return 'Grass'
        else:
            return 'Hard'
    
    def _extract_current_set(self, status: str) -> int:
        """Extract current set number from status"""
        try:
            if 'Set' in status:
                return int(status.split('Set')[1].strip().split()[0])
        except:
            pass
        return 1
    
    def _calculate_match_duration(self, events: List[LiveMatchEvent]) -> float:
        """Calculate match duration in minutes"""
        if len(events) < 2:
            return 0.0
        
        start_time = events[0].timestamp
        end_time = events[-1].timestamp
        duration = (end_time - start_time).total_seconds() / 60
        return round(duration, 2)
    
    def _count_total_games(self, events: List[LiveMatchEvent]) -> int:
        """Count total games played"""
        if not events:
            return 0
        
        latest_event = events[-1]
        total_games = 0
        
        try:
            for score_data in latest_event.scores:
                total_games += int(score_data.get('score_first', 0))
                total_games += int(score_data.get('score_second', 0))
        except:
            pass
        
        return total_games
    
    def _calculate_momentum(self, events: List[LiveMatchEvent]) -> float:
        """Calculate momentum indicator based on recent game wins"""
        if len(events) < 5:
            return 0.0
        
        recent_events = events[-10:]  # Look at last 10 events
        momentum = 0.0
        
        # Analyze point-by-point data for momentum
        for event in recent_events:
            try:
                for point_data in event.point_by_point:
                    for point in point_data.get('points', []):
                        # Simplified momentum calculation
                        if 'break_point' in point and point['break_point']:
                            momentum += 0.3 if 'First Player' in point.get('winner', '') else -0.3
            except:
                continue
        
        return max(-1.0, min(1.0, momentum))
    
    def _analyze_break_points(self, events: List[LiveMatchEvent]) -> Dict[str, int]:
        """Analyze break point statistics"""
        break_points = {'player1_saved': 0, 'player1_converted': 0, 'player2_saved': 0, 'player2_converted': 0}
        
        for event in events:
            try:
                for point_data in event.point_by_point:
                    for point in point_data.get('points', []):
                        if point.get('break_point'):
                            # Simplified break point analysis
                            if 'First Player' in point.get('winner', ''):
                                if point_data.get('player_served') == 'Second Player':
                                    break_points['player1_converted'] += 1
                                else:
                                    break_points['player1_saved'] += 1
            except:
                continue
        
        return break_points
    
    def _analyze_service_stats(self, events: List[LiveMatchEvent]) -> Dict[str, float]:
        """Analyze service statistics"""
        service_stats = {
            'player1_service_games_won': 0,
            'player1_service_games_total': 0,
            'player2_service_games_won': 0,
            'player2_service_games_total': 0
        }
        
        for event in events:
            try:
                for point_data in event.point_by_point:
                    server = point_data.get('player_served', '')
                    winner = point_data.get('serve_winner', '')
                    
                    if server == 'First Player':
                        service_stats['player1_service_games_total'] += 1
                        if winner == 'First Player':
                            service_stats['player1_service_games_won'] += 1
                    elif server == 'Second Player':
                        service_stats['player2_service_games_total'] += 1
                        if winner == 'Second Player':
                            service_stats['player2_service_games_won'] += 1
            except:
                continue
        
        # Calculate service percentages
        stats = {}
        if service_stats['player1_service_games_total'] > 0:
            stats['player1_service_percentage'] = service_stats['player1_service_games_won'] / service_stats['player1_service_games_total']
        else:
            stats['player1_service_percentage'] = 0.0
            
        if service_stats['player2_service_games_total'] > 0:
            stats['player2_service_percentage'] = service_stats['player2_service_games_won'] / service_stats['player2_service_games_total']
        else:
            stats['player2_service_percentage'] = 0.0
        
        return stats
    
    def _extract_set_scores(self, scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract set score history"""
        return scores
    
    def get_active_matches(self) -> List[int]:
        """Get list of currently active match IDs"""
        cutoff_time = datetime.now() - timedelta(hours=4)
        active_matches = []
        
        for match_id, last_update in self.last_update.items():
            if last_update > cutoff_time:
                active_matches.append(match_id)
        
        return active_matches
    
    def cleanup_old_data(self):
        """Remove old match data to prevent memory leaks"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        matches_to_remove = []
        
        for match_id, last_update in self.last_update.items():
            if last_update < cutoff_time:
                matches_to_remove.append(match_id)
        
        for match_id in matches_to_remove:
            if match_id in self.match_buffers:
                del self.match_buffers[match_id]
            if match_id in self.match_metadata:
                del self.match_metadata[match_id]
            if match_id in self.last_update:
                del self.last_update[match_id]
        
        if matches_to_remove:
            logger.info(f"Cleaned up {len(matches_to_remove)} old matches from buffer")


class MLPredictionProcessor:
    """Processes ML prediction requests using available models"""
    
    def __init__(self):
        self.config = get_config()
        self.ml_integration = APITennisMLIntegration()
        
        # Initialize ML components
        self.orchestrator = None
        self.second_set_service = None
        
        if ML_ORCHESTRATOR_AVAILABLE:
            try:
                self.orchestrator = EnhancedMLOrchestrator()
                logger.info("ML Orchestrator initialized")
            except Exception as e:
                logger.error(f"Failed to initialize ML Orchestrator: {e}")
        
        if SECOND_SET_AVAILABLE:
            try:
                self.second_set_service = SecondSetPredictionService()
                logger.info("Second Set Service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Second Set Service: {e}")
    
    def process_prediction_request(self, request: MLPredictionRequest) -> Optional[MLPredictionResult]:
        """Process ML prediction request"""
        start_time = time.time()
        
        try:
            # Convert live data to ML input format
            ml_input = self._convert_to_ml_input(request)
            
            # Generate prediction using available models
            prediction = self._generate_prediction(ml_input)
            
            if not prediction:
                return None
            
            processing_time = time.time() - start_time
            
            result = MLPredictionResult(
                match_id=request.match_id,
                prediction=prediction['prediction'],
                confidence=prediction.get('confidence', 0.5),
                model_used=prediction.get('model_used', 'unknown'),
                processing_time=processing_time,
                timestamp=datetime.now(),
                live_context=request.live_data
            )
            
            logger.info(f"Generated prediction for match {request.match_id} in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing prediction request: {e}")
            return None
    
    def _convert_to_ml_input(self, request: MLPredictionRequest) -> Dict[str, Any]:
        """Convert prediction request to ML model input format"""
        live_data = request.live_data
        
        ml_input = {
            'match_id': request.match_id,
            'player_1': request.player1,
            'player_2': request.player2,
            'tournament': request.tournament,
            'surface': request.surface,
            'current_score': live_data.get('current_score', ''),
            'current_set': live_data.get('current_set', 1),
            'serving_player': live_data.get('serving_player', ''),
            'match_duration_minutes': live_data.get('match_duration_minutes', 0),
            'total_games_played': live_data.get('total_games_played', 0),
            'momentum_indicator': live_data.get('momentum_indicator', 0),
            'break_points_converted': live_data.get('break_points_converted', {}),
            'service_dominance': live_data.get('service_dominance', {}),
            'is_live': live_data.get('is_live', True),
            'timestamp': request.timestamp.isoformat()
        }
        
        return ml_input
    
    def _generate_prediction(self, ml_input: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate prediction using available ML models"""
        
        # Try ML Orchestrator first
        if self.orchestrator:
            try:
                prediction = self.orchestrator.predict_match_outcome(ml_input)
                if prediction:
                    return {
                        'prediction': prediction,
                        'confidence': prediction.get('confidence', 0.5),
                        'model_used': 'enhanced_ml_orchestrator'
                    }
            except Exception as e:
                logger.debug(f"ML Orchestrator failed: {e}")
        
        # Try Second Set Service
        if self.second_set_service:
            try:
                prediction = self.second_set_service.predict_match_outcome(ml_input)
                if prediction:
                    return {
                        'prediction': prediction,
                        'confidence': prediction.get('confidence', 0.5),
                        'model_used': 'second_set_service'
                    }
            except Exception as e:
                logger.debug(f"Second Set Service failed: {e}")
        
        # Fallback to basic prediction
        return self._generate_basic_prediction(ml_input)
    
    def _generate_basic_prediction(self, ml_input: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic prediction based on current match state"""
        try:
            current_score = ml_input.get('current_score', '0 - 0')
            momentum = ml_input.get('momentum_indicator', 0)
            service_stats = ml_input.get('service_dominance', {})
            
            # Parse current score
            try:
                score_parts = current_score.split(' - ')
                player1_sets = int(score_parts[0])
                player2_sets = int(score_parts[1])
            except:
                player1_sets = 0
                player2_sets = 0
            
            # Calculate win probability based on current state
            base_prob = 0.5
            
            # Adjust for set score
            set_diff = player1_sets - player2_sets
            base_prob += set_diff * 0.15
            
            # Adjust for momentum
            base_prob += momentum * 0.1
            
            # Adjust for service performance
            player1_service = service_stats.get('player1_service_percentage', 0.5)
            player2_service = service_stats.get('player2_service_percentage', 0.5)
            service_diff = player1_service - player2_service
            base_prob += service_diff * 0.2
            
            # Clamp probability
            win_prob = max(0.1, min(0.9, base_prob))
            
            return {
                'prediction': {
                    'winner': ml_input.get('player_1', 'Player 1'),
                    'player_1_win_probability': win_prob,
                    'player_2_win_probability': 1 - win_prob,
                    'recommended_stake': self._calculate_recommended_stake(win_prob),
                    'confidence': abs(win_prob - 0.5) * 2
                },
                'confidence': abs(win_prob - 0.5) * 2,
                'model_used': 'realtime_basic_model'
            }
            
        except Exception as e:
            logger.error(f"Error in basic prediction: {e}")
            return {
                'prediction': {
                    'winner': ml_input.get('player_1', 'Player 1'),
                    'player_1_win_probability': 0.5,
                    'player_2_win_probability': 0.5,
                    'recommended_stake': 0.0,
                    'confidence': 0.0
                },
                'confidence': 0.0,
                'model_used': 'fallback_model'
            }
    
    def _calculate_recommended_stake(self, win_probability: float) -> float:
        """Calculate recommended stake based on win probability"""
        confidence = abs(win_probability - 0.5) * 2
        base_stake = self.config.DEFAULT_STAKE
        
        if confidence > self.config.PREDICTION_CONFIDENCE_THRESHOLD:
            return base_stake * confidence
        else:
            return 0.0  # Don't bet if confidence is too low


class RealTimeMLPipeline:
    """Main real-time ML pipeline that coordinates WebSocket data and ML predictions"""
    
    def __init__(self):
        self.config = get_config()
        
        # Components
        self.websocket_client = None
        self.data_buffer = LiveDataBuffer()
        self.ml_processor = MLPredictionProcessor()
        
        # Processing queues
        self.prediction_queue = Queue()
        self.result_queue = Queue()
        
        # Control flags
        self.is_running = False
        self.stop_event = threading.Event()
        
        # Callbacks
        self.prediction_callbacks: List[Callable] = []
        self.event_callbacks: List[Callable] = []
        
        # Statistics
        self.stats = {
            'events_processed': 0,
            'predictions_generated': 0,
            'errors': 0,
            'start_time': None
        }
        
        # Prediction triggers
        self.prediction_triggers = {
            'score_change': True,
            'set_change': True,
            'break_point': True,
            'periodic': True,
            'periodic_interval': 300  # 5 minutes
        }
        
        self.last_predictions: Dict[int, datetime] = {}
    
    def add_prediction_callback(self, callback: Callable[[MLPredictionResult], None]):
        """Add callback for prediction results"""
        self.prediction_callbacks.append(callback)
    
    def add_event_callback(self, callback: Callable[[LiveMatchEvent], None]):
        """Add callback for live events"""
        self.event_callbacks.append(callback)
    
    def initialize_websocket(self, api_key: str = None):
        """Initialize WebSocket connection"""
        self.websocket_client = TennisWebSocketClient(api_key=api_key)
        self.websocket_client.add_event_callback(self._handle_live_event)
        self.websocket_client.add_connection_callback(self._handle_connection_state)
    
    def _handle_live_event(self, event: LiveMatchEvent):
        """Handle incoming live event from WebSocket"""
        try:
            self.stats['events_processed'] += 1
            
            # Add to data buffer
            self.data_buffer.add_live_event(event)
            
            # Check if prediction should be triggered
            if self._should_trigger_prediction(event):
                self._queue_prediction_request(event)
            
            # Notify event callbacks
            for callback in self.event_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")
            
            logger.debug(f"Processed live event for match {event.event_key}")
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Error handling live event: {e}")
    
    def _handle_connection_state(self, state: WSConnectionState):
        """Handle WebSocket connection state changes"""
        logger.info(f"WebSocket connection state: {state.value}")
    
    def _should_trigger_prediction(self, event: LiveMatchEvent) -> bool:
        """Determine if a prediction should be triggered for this event"""
        match_id = event.event_key
        now = datetime.now()
        
        # Check last prediction time
        if match_id in self.last_predictions:
            time_since_last = now - self.last_predictions[match_id]
            if time_since_last < timedelta(minutes=2):  # Minimum 2 minutes between predictions
                return False
        
        # Trigger on significant events
        if self.prediction_triggers['score_change'] and self._is_score_change(event):
            return True
        
        if self.prediction_triggers['set_change'] and self._is_set_change(event):
            return True
        
        if self.prediction_triggers['break_point'] and self._is_break_point(event):
            return True
        
        # Periodic prediction
        if self.prediction_triggers['periodic']:
            if match_id not in self.last_predictions:
                return True
            time_since_last = now - self.last_predictions[match_id]
            if time_since_last > timedelta(seconds=self.prediction_triggers['periodic_interval']):
                return True
        
        return False
    
    def _is_score_change(self, event: LiveMatchEvent) -> bool:
        """Check if event represents a score change"""
        # Simplified check - could be enhanced with more sophisticated logic
        return event.live and event.final_result != '0 - 0'
    
    def _is_set_change(self, event: LiveMatchEvent) -> bool:
        """Check if event represents a set change"""
        return 'Set' in event.status and event.live
    
    def _is_break_point(self, event: LiveMatchEvent) -> bool:
        """Check if event involves a break point"""
        try:
            for point_data in event.point_by_point:
                for point in point_data.get('points', []):
                    if point.get('break_point'):
                        return True
        except:
            pass
        return False
    
    def _queue_prediction_request(self, event: LiveMatchEvent):
        """Queue a prediction request based on live event"""
        try:
            # Get features from data buffer
            features = self.data_buffer.get_match_features(event.event_key)
            
            if not features:
                return
            
            request = MLPredictionRequest(
                match_id=event.event_key,
                player1=event.first_player,
                player2=event.second_player,
                tournament=event.tournament_name,
                surface=features.get('surface', 'Hard'),
                live_data=features,
                timestamp=datetime.now(),
                priority=self._calculate_priority(event)
            )
            
            self.prediction_queue.put(request)
            self.last_predictions[event.event_key] = datetime.now()
            
            logger.info(f"Queued prediction request for match {event.event_key}")
            
        except Exception as e:
            logger.error(f"Error queuing prediction request: {e}")
    
    def _calculate_priority(self, event: LiveMatchEvent) -> int:
        """Calculate priority for prediction request"""
        priority = 1
        
        # Higher priority for live matches
        if event.live:
            priority += 2
        
        # Higher priority for important tournaments
        if any(keyword in event.tournament_name.lower() for keyword in ['masters', 'open', 'cup']):
            priority += 1
        
        # Higher priority for break points or set points
        try:
            for point_data in event.point_by_point:
                for point in point_data.get('points', []):
                    if point.get('break_point') or point.get('set_point') or point.get('match_point'):
                        priority += 3
                        break
        except:
            pass
        
        return priority
    
    def _prediction_worker(self):
        """Worker thread for processing prediction requests"""
        while not self.stop_event.is_set():
            try:
                # Get prediction request
                request = self.prediction_queue.get(timeout=1)
                
                # Process prediction
                result = self.ml_processor.process_prediction_request(request)
                
                if result:
                    self.stats['predictions_generated'] += 1
                    self.result_queue.put(result)
                    
                    # Notify callbacks
                    for callback in self.prediction_callbacks:
                        try:
                            callback(result)
                        except Exception as e:
                            logger.error(f"Error in prediction callback: {e}")
                
                self.prediction_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                self.stats['errors'] += 1
                logger.error(f"Error in prediction worker: {e}")
    
    def _cleanup_worker(self):
        """Worker thread for periodic cleanup"""
        while not self.stop_event.is_set():
            try:
                time.sleep(300)  # Run every 5 minutes
                self.data_buffer.cleanup_old_data()
                
                # Clean up old prediction timestamps
                cutoff_time = datetime.now() - timedelta(hours=6)
                matches_to_remove = []
                for match_id, last_time in self.last_predictions.items():
                    if last_time < cutoff_time:
                        matches_to_remove.append(match_id)
                
                for match_id in matches_to_remove:
                    del self.last_predictions[match_id]
                
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
    
    def start(self, api_key: str = None):
        """Start the real-time ML pipeline"""
        logger.info("Starting Real-Time ML Pipeline...")
        
        self.is_running = True
        self.stop_event.clear()
        self.stats['start_time'] = datetime.now()
        
        # Initialize WebSocket if not already done
        if not self.websocket_client:
            self.initialize_websocket(api_key)
        
        # Start worker threads
        self.prediction_thread = threading.Thread(target=self._prediction_worker, daemon=True)
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        
        self.prediction_thread.start()
        self.cleanup_thread.start()
        
        # Start WebSocket client
        self.websocket_thread = self.websocket_client.start_threaded()
        
        logger.info("Real-Time ML Pipeline started successfully")
    
    def stop(self):
        """Stop the real-time ML pipeline"""
        logger.info("Stopping Real-Time ML Pipeline...")
        
        self.is_running = False
        self.stop_event.set()
        
        # Stop WebSocket client
        if self.websocket_client:
            self.websocket_client.stop_threaded()
        
        logger.info("Real-Time ML Pipeline stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        stats = self.stats.copy()
        
        if self.websocket_client:
            stats['websocket'] = self.websocket_client.get_stats()
        
        stats['active_matches'] = len(self.data_buffer.get_active_matches())
        stats['prediction_queue_size'] = self.prediction_queue.qsize()
        stats['result_queue_size'] = self.result_queue.qsize()
        
        if self.stats['start_time']:
            uptime = datetime.now() - self.stats['start_time']
            stats['uptime_seconds'] = uptime.total_seconds()
        
        return stats
    
    def get_recent_predictions(self, max_results: int = 10) -> List[MLPredictionResult]:
        """Get recent prediction results"""
        results = []
        try:
            while len(results) < max_results:
                result = self.result_queue.get_nowait()
                results.append(result)
        except Empty:
            pass
        
        return results
    
    def get_active_matches(self) -> List[Dict[str, Any]]:
        """Get currently active matches with their latest features"""
        active_matches = []
        
        for match_id in self.data_buffer.get_active_matches():
            features = self.data_buffer.get_match_features(match_id)
            if features:
                active_matches.append(features)
        
        return active_matches


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    pipeline = RealTimeMLPipeline()
    
    # Add prediction callback
    def handle_prediction(result: MLPredictionResult):
        print(f"New prediction: {result.prediction['winner']} (confidence: {result.confidence:.2f})")
    
    pipeline.add_prediction_callback(handle_prediction)
    
    try:
        pipeline.start()
        
        while True:
            time.sleep(30)
            stats = pipeline.get_stats()
            print(f"Pipeline stats: {stats['events_processed']} events, {stats['predictions_generated']} predictions")
            
    except KeyboardInterrupt:
        pipeline.stop()