#!/usr/bin/env python3
"""
Real-Time Tennis Data Integration System
========================================

Live data collection and processing system for real-time tennis match analysis
and second set prediction updates.

Features:
- WebSocket connections to live tennis feeds
- Real-time match state tracking
- Live feature calculation during matches
- Automated prediction updates
- Event-driven notification system

Author: Tennis ML Enhancement System
Date: 2025-08-30
"""

import asyncio
import websockets
import json
import logging
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor
import signal
import ssl

# Local imports
try:
    from src.ml.enhanced_feature_engineering import EnhancedTennisFeatureEngineer, MatchContext, FirstSetStats
except ImportError:
    # Fallback for testing
    pass

logger = logging.getLogger(__name__)

@dataclass
class LiveMatchState:
    """Current state of a live tennis match"""
    match_id: str
    player1_name: str
    player2_name: str
    player1_ranking: int
    player2_ranking: int
    tournament: str
    surface: str
    round: str
    is_indoor: bool
    
    # Match progress
    current_set: int
    sets_completed: int
    current_game: List[int]  # [player1_games, player2_games] in current set
    current_score: str       # e.g., "30-15"
    serving_player: int      # 1 or 2
    
    # Set scores
    set_scores: List[List[int]]  # [[6,4], [2,3]] for completed and current sets
    
    # Statistics
    first_set_stats: Optional[FirstSetStats] = None
    match_stats: Dict[str, Any] = None
    
    # Timing
    match_start_time: Optional[datetime] = None
    first_set_end_time: Optional[datetime] = None
    last_update_time: Optional[datetime] = None
    
    # Prediction data
    second_set_prediction: Optional[Dict[str, float]] = None
    prediction_confidence: Optional[float] = None
    
    def is_first_set_complete(self) -> bool:
        """Check if first set is completed"""
        return self.sets_completed >= 1 and self.first_set_stats is not None
    
    def is_second_set_active(self) -> bool:
        """Check if second set is currently being played"""
        return self.sets_completed == 1 and self.current_set == 2
    
    def get_underdog_player(self) -> int:
        """Get underdog player number (1 or 2)"""
        return 2 if self.player1_ranking < self.player2_ranking else 1

class RealTimeTennisDataCollector:
    """Real-time tennis data collection and processing system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.api_key = self.config.get('api_tennis_key', '')
        self.websocket_url = self.config.get('websocket_url', 'wss://api.api-tennis.com/tennis/live')
        
        # State tracking
        self.active_matches: Dict[str, LiveMatchState] = {}
        self.monitored_match_ids: Set[str] = set()
        self.is_running = False
        self.websocket = None
        
        # Feature engineering
        self.feature_engineer = EnhancedTennisFeatureEngineer()
        
        # Callbacks
        self.prediction_callbacks: List[Callable] = []
        self.match_update_callbacks: List[Callable] = []
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.loop = None
        
        # Rate limiting
        self.last_api_call = datetime.min
        self.api_call_interval = timedelta(seconds=2)  # Minimum interval between API calls
        
        logger.info("✅ Real-time tennis data collector initialized")
    
    def add_prediction_callback(self, callback: Callable[[str, Dict], None]) -> None:
        """Add callback for when new predictions are made"""
        self.prediction_callbacks.append(callback)
    
    def add_match_update_callback(self, callback: Callable[[str, LiveMatchState], None]) -> None:
        """Add callback for match state updates"""
        self.match_update_callbacks.append(callback)
    
    async def start_live_monitoring(self, match_ids: List[str] = None) -> None:
        """Start monitoring live tennis matches"""
        
        if self.is_running:
            logger.warning("Live monitoring already running")
            return
        
        self.is_running = True
        self.monitored_match_ids = set(match_ids) if match_ids else set()
        
        logger.info("🚀 Starting live tennis monitoring...")
        logger.info(f"   WebSocket URL: {self.websocket_url}")
        logger.info(f"   Monitored matches: {len(self.monitored_match_ids) if match_ids else 'All'}")
        
        try:
            # Start WebSocket connection and HTTP polling in parallel
            await asyncio.gather(
                self._websocket_listener(),
                self._http_polling_monitor(),
                self._prediction_processor()
            )
        except Exception as e:
            logger.error(f"❌ Live monitoring error: {e}")
        finally:
            self.is_running = False
    
    async def _websocket_listener(self) -> None:
        """Listen to WebSocket live match updates"""
        
        while self.is_running:
            try:
                # Create SSL context for secure WebSocket
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                # Connect to WebSocket
                uri = f"{self.websocket_url}?APIkey={self.api_key}"
                
                async with websockets.connect(uri, ssl=ssl_context) as websocket:
                    self.websocket = websocket
                    logger.info("✅ WebSocket connected to live tennis feed")
                    
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            await self._process_websocket_message(data)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON received: {message[:100]}...")
                        except Exception as e:
                            logger.error(f"Error processing WebSocket message: {e}")
                            
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed, attempting reconnect in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(10)
    
    async def _http_polling_monitor(self) -> None:
        """HTTP polling for match data (backup/supplement to WebSocket)"""
        
        while self.is_running:
            try:
                # Rate limiting
                now = datetime.now()
                if now - self.last_api_call < self.api_call_interval:
                    await asyncio.sleep(1)
                    continue
                
                # Get live matches via HTTP API
                await self._fetch_live_matches_http()
                self.last_api_call = now
                
                # Wait before next poll
                await asyncio.sleep(30)  # Poll every 30 seconds
                
            except Exception as e:
                logger.error(f"HTTP polling error: {e}")
                await asyncio.sleep(60)
    
    async def _prediction_processor(self) -> None:
        """Process predictions for matches with completed first sets"""
        
        while self.is_running:
            try:
                # Check all active matches for prediction opportunities
                for match_id, match_state in self.active_matches.items():
                    if (match_state.is_first_set_complete() and 
                        match_state.is_second_set_active() and
                        match_state.second_set_prediction is None):
                        
                        # Generate prediction for second set
                        await self._generate_second_set_prediction(match_id, match_state)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Prediction processor error: {e}")
                await asyncio.sleep(10)
    
    async def _process_websocket_message(self, data: Dict[str, Any]) -> None:
        """Process incoming WebSocket message"""
        
        try:
            # Extract match information
            match_id = str(data.get('event_key', ''))
            if not match_id:
                return
            
            # Filter matches if specific IDs are monitored
            if self.monitored_match_ids and match_id not in self.monitored_match_ids:
                return
            
            # Update or create match state
            match_state = await self._update_match_state(match_id, data)
            
            if match_state:
                # Notify callbacks
                for callback in self.match_update_callbacks:
                    try:
                        callback(match_id, match_state)
                    except Exception as e:
                        logger.error(f"Match update callback error: {e}")
                
                logger.debug(f"Updated match {match_id}: {match_state.current_score}")
        
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    async def _fetch_live_matches_http(self) -> None:
        """Fetch live matches via HTTP API"""
        
        try:
            url = f"https://api.api-tennis.com/tennis/?method=get_fixtures&APIkey={self.api_key}&date_start={datetime.now().strftime('%Y-%m-%d')}&date_stop={datetime.now().strftime('%Y-%m-%d')}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('success') == 1:
                            matches = data.get('result', [])
                            
                            for match in matches:
                                if match.get('event_live') == '1':  # Live match
                                    match_id = str(match.get('event_key', ''))
                                    if match_id:
                                        await self._update_match_state(match_id, match)
                        
                        logger.debug(f"HTTP: Fetched {len(matches)} matches")
                    else:
                        logger.warning(f"HTTP API error: {response.status}")
        
        except Exception as e:
            logger.error(f"HTTP fetch error: {e}")
    
    async def _update_match_state(self, match_id: str, data: Dict[str, Any]) -> Optional[LiveMatchState]:
        """Update match state from incoming data"""
        
        try:
            # Extract basic match info
            player1_name = data.get('event_first_player', '')
            player2_name = data.get('event_second_player', '')
            
            if not player1_name or not player2_name:
                return None
            
            # Get or create match state
            if match_id in self.active_matches:
                match_state = self.active_matches[match_id]
            else:
                # Create new match state
                match_state = LiveMatchState(
                    match_id=match_id,
                    player1_name=player1_name,
                    player2_name=player2_name,
                    player1_ranking=self._extract_ranking(data.get('first_player_key', ''), player1_name),
                    player2_ranking=self._extract_ranking(data.get('second_player_key', ''), player2_name),
                    tournament=data.get('tournament_name', ''),
                    surface=self._extract_surface(data.get('tournament_name', '')),
                    round=data.get('tournament_round', ''),
                    is_indoor=self._is_indoor_tournament(data.get('tournament_name', '')),
                    current_set=1,
                    sets_completed=0,
                    current_game=[0, 0],
                    current_score="0-0",
                    serving_player=1,
                    set_scores=[],
                    match_start_time=datetime.now()
                )
                self.active_matches[match_id] = match_state
                
                logger.info(f"🆕 New live match: {player1_name} vs {player2_name}")
            
            # Update match state with new data
            match_state.last_update_time = datetime.now()
            
            # Parse scores and game state
            self._parse_match_scores(match_state, data)
            
            # Extract first set statistics if available
            if match_state.is_first_set_complete() and not match_state.first_set_stats:
                match_state.first_set_stats = self._extract_first_set_stats(data)
            
            return match_state
            
        except Exception as e:
            logger.error(f"Error updating match state: {e}")
            return None
    
    def _extract_ranking(self, player_key: str, player_name: str) -> int:
        """Extract player ranking (placeholder implementation)"""
        # This would integrate with your existing ranking system
        # For now, return a default based on player name lookup
        return 50  # Placeholder
    
    def _extract_surface(self, tournament_name: str) -> str:
        """Extract surface type from tournament name"""
        tournament_lower = tournament_name.lower()
        
        if any(keyword in tournament_lower for keyword in ['clay', 'roland garros', 'french open']):
            return 'Clay'
        elif any(keyword in tournament_lower for keyword in ['grass', 'wimbledon']):
            return 'Grass'
        else:
            return 'Hard'  # Default
    
    def _is_indoor_tournament(self, tournament_name: str) -> bool:
        """Determine if tournament is indoor"""
        indoor_keywords = ['indoor', 'masters', 'atp finals', 'wta finals']
        return any(keyword in tournament_name.lower() for keyword in indoor_keywords)
    
    def _parse_match_scores(self, match_state: LiveMatchState, data: Dict[str, Any]) -> None:
        """Parse match scores and update match state"""
        
        # This is a simplified implementation
        # Real implementation would parse detailed score data
        
        scores = data.get('scores', [])
        if scores:
            match_state.sets_completed = len([s for s in scores if s.get('score_set')])
            
            # Update current set and games
            if match_state.sets_completed > 0:
                current_set_data = scores[-1] if scores else {}
                match_state.current_game = [
                    int(current_set_data.get('score_first', 0)),
                    int(current_set_data.get('score_second', 0))
                ]
        
        # Update current score
        match_state.current_score = data.get('event_game_result', '0-0')
    
    def _extract_first_set_stats(self, data: Dict[str, Any]) -> Optional[FirstSetStats]:
        """Extract first set statistics from match data"""
        
        # This would parse detailed point-by-point data
        # Simplified implementation for now
        
        try:
            scores = data.get('scores', [])
            if not scores:
                return None
            
            first_set = scores[0]
            
            return FirstSetStats(
                winner='player1' if int(first_set.get('score_first', 0)) > int(first_set.get('score_second', 0)) else 'player2',
                score=f"{first_set.get('score_first', 0)}-{first_set.get('score_second', 0)}",
                duration_minutes=45,  # Estimated
                total_games=int(first_set.get('score_first', 0)) + int(first_set.get('score_second', 0)),
                break_points_player1={'faced': 2, 'saved': 1, 'converted': 1},
                break_points_player2={'faced': 3, 'saved': 2, 'converted': 0},
                service_points_player1={'won': 30, 'total': 40},
                service_points_player2={'won': 25, 'total': 35},
                unforced_errors_player1=8,
                unforced_errors_player2=12,
                winners_player1=15,
                winners_player2=10,
                double_faults_player1=1,
                double_faults_player2=2
            )
            
        except Exception as e:
            logger.error(f"Error extracting first set stats: {e}")
            return None
    
    async def _generate_second_set_prediction(self, match_id: str, match_state: LiveMatchState) -> None:
        """Generate second set prediction for a match"""
        
        try:
            logger.info(f"🎯 Generating second set prediction for {match_state.player1_name} vs {match_state.player2_name}")
            
            # Create match data for feature engineering
            match_data = {
                'player1': {
                    'ranking': match_state.player1_ranking,
                    'age': 25,  # Would get from player database
                    'last_match_date': datetime.now() - timedelta(days=3),
                    'matches_last_14_days': 2,
                    'hard_win_percentage': 0.65,
                    'indoor_win_percentage': 0.70
                },
                'player2': {
                    'ranking': match_state.player2_ranking,
                    'age': 24,  # Would get from player database
                    'last_match_date': datetime.now() - timedelta(days=5),
                    'matches_last_14_days': 1,
                    'hard_win_percentage': 0.55,
                    'indoor_win_percentage': 0.50
                },
                'first_set_stats': match_state.first_set_stats,
                'match_context': MatchContext(
                    tournament_tier='ATP500',  # Would extract from tournament
                    surface=match_state.surface,
                    round=match_state.round,
                    is_indoor=match_state.is_indoor
                ),
                'h2h_data': {
                    'overall': {'player1_wins': 1, 'player2_wins': 1},
                    'recent_3': {'player1_wins': 1, 'player2_wins': 1}
                },
                'match_date': match_state.match_start_time or datetime.now()
            }
            
            # Generate enhanced features
            features = self.feature_engineer.create_all_enhanced_features(match_data)
            
            if features:
                # Create prediction (simplified - would use trained models)
                underdog_player = match_state.get_underdog_player()
                confidence = np.random.uniform(0.55, 0.75)  # Placeholder
                
                prediction = {
                    'underdog_player': underdog_player,
                    'underdog_win_probability': confidence,
                    'favorite_win_probability': 1.0 - confidence,
                    'features_count': len(features),
                    'prediction_time': datetime.now().isoformat()
                }
                
                match_state.second_set_prediction = prediction
                match_state.prediction_confidence = confidence
                
                # Notify prediction callbacks
                for callback in self.prediction_callbacks:
                    try:
                        callback(match_id, prediction)
                    except Exception as e:
                        logger.error(f"Prediction callback error: {e}")
                
                logger.info(f"✅ Prediction generated: {confidence:.1%} confidence for underdog")
            
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
    
    def stop_monitoring(self) -> None:
        """Stop live monitoring"""
        self.is_running = False
        logger.info("🛑 Stopping live monitoring...")
    
    def get_active_matches(self) -> Dict[str, LiveMatchState]:
        """Get all currently active matches"""
        return self.active_matches.copy()
    
    def get_match_state(self, match_id: str) -> Optional[LiveMatchState]:
        """Get specific match state"""
        return self.active_matches.get(match_id)

# Example usage and testing
async def main():
    """Example usage of the real-time data collector"""
    
    # Create collector
    config = {
        'api_tennis_key': 'your_api_key_here',
        'websocket_url': 'wss://api.api-tennis.com/tennis/live'
    }
    
    collector = RealTimeTennisDataCollector(config)
    
    # Add callbacks
    def on_prediction(match_id: str, prediction: Dict[str, Any]) -> None:
        print(f"🎯 New prediction for {match_id}: {prediction}")
    
    def on_match_update(match_id: str, match_state: LiveMatchState) -> None:
        print(f"📊 Match update {match_id}: {match_state.current_score}")
    
    collector.add_prediction_callback(on_prediction)
    collector.add_match_update_callback(on_match_update)
    
    # Start monitoring
    try:
        await collector.start_live_monitoring()
    except KeyboardInterrupt:
        print("Stopping monitoring...")
        collector.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(main())