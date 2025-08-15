#!/usr/bin/env python3
"""
Real-Time Tennis WebSocket Client for API-Tennis.com
Production-ready WebSocket integration for live tennis data streaming
"""

import os
import json
import time
import asyncio
import logging
import threading
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import ssl
import urllib.parse
from queue import Queue, Empty
import signal
import sys
from contextlib import asynccontextmanager

# Import existing components
from api_tennis_integration import TennisMatch, TennisPlayer
from config import get_config

logger = logging.getLogger(__name__)


class WSConnectionState(Enum):
    """WebSocket connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting" 
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    CLOSING = "closing"


@dataclass
class LiveMatchEvent:
    """Live match event data model"""
    event_key: int
    event_date: str
    event_time: str
    first_player: str
    first_player_key: int
    second_player: str
    second_player_key: int
    final_result: str
    game_result: str
    serve: str
    winner: Optional[str]
    status: str
    event_type: str
    tournament_name: str
    tournament_key: int
    tournament_round: Optional[str]
    tournament_season: str
    live: bool
    scores: List[Dict[str, Any]]
    point_by_point: List[Dict[str, Any]]
    statistics: List[Dict[str, Any]]
    timestamp: datetime
    
    @classmethod
    def from_api_data(cls, data: Dict[str, Any]) -> 'LiveMatchEvent':
        """Create LiveMatchEvent from API data"""
        return cls(
            event_key=data.get('event_key', 0),
            event_date=data.get('event_date', ''),
            event_time=data.get('event_time', ''),
            first_player=data.get('event_first_player', ''),
            first_player_key=data.get('first_player_key', 0),
            second_player=data.get('event_second_player', ''),
            second_player_key=data.get('second_player_key', 0),
            final_result=data.get('event_final_result', ''),
            game_result=data.get('event_game_result', ''),
            serve=data.get('event_serve', ''),
            winner=data.get('event_winner'),
            status=data.get('event_status', ''),
            event_type=data.get('event_type_type', ''),
            tournament_name=data.get('tournament_name', ''),
            tournament_key=data.get('tournament_key', 0),
            tournament_round=data.get('tournament_round'),
            tournament_season=data.get('tournament_season', ''),
            live=data.get('event_live') == '1',
            scores=data.get('scores', []),
            point_by_point=data.get('pointbypoint', []),
            statistics=data.get('statistics', []),
            timestamp=datetime.now()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class LiveDataProcessor:
    """Process and validate live tennis data"""
    
    def __init__(self):
        self.match_cache = {}  # Cache for match state tracking
        self.event_history = []  # History of recent events
        self.max_history = 1000
    
    def process_live_event(self, raw_data: Dict[str, Any]) -> Optional[LiveMatchEvent]:
        """Process raw WebSocket data into structured event"""
        try:
            # Validate required fields
            if not self._validate_event_data(raw_data):
                logger.warning(f"Invalid event data: {raw_data}")
                return None
            
            # Create live match event
            event = LiveMatchEvent.from_api_data(raw_data)
            
            # Update match cache
            self._update_match_cache(event)
            
            # Add to history
            self._add_to_history(event)
            
            logger.debug(f"Processed live event: {event.first_player} vs {event.second_player}")
            return event
            
        except Exception as e:
            logger.error(f"Error processing live event: {e}")
            return None
    
    def _validate_event_data(self, data: Dict[str, Any]) -> bool:
        """Validate live event data"""
        required_fields = [
            'event_key', 'event_first_player', 'event_second_player',
            'tournament_name', 'event_status'
        ]
        
        for field in required_fields:
            if field not in data or data[field] is None:
                return False
        
        return True
    
    def _update_match_cache(self, event: LiveMatchEvent):
        """Update internal match state cache"""
        match_key = event.event_key
        
        if match_key in self.match_cache:
            # Update existing match
            cached_match = self.match_cache[match_key]
            cached_match.update({
                'final_result': event.final_result,
                'game_result': event.game_result,
                'serve': event.serve,
                'winner': event.winner,
                'status': event.status,
                'scores': event.scores,
                'last_update': event.timestamp
            })
        else:
            # New match
            self.match_cache[match_key] = {
                'event': event,
                'first_seen': event.timestamp,
                'last_update': event.timestamp
            }
    
    def _add_to_history(self, event: LiveMatchEvent):
        """Add event to history buffer"""
        self.event_history.append(event)
        
        # Maintain history size
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)
    
    def get_match_state(self, event_key: int) -> Optional[Dict[str, Any]]:
        """Get current state of a match"""
        return self.match_cache.get(event_key)
    
    def get_active_matches(self) -> List[Dict[str, Any]]:
        """Get all currently active matches"""
        active_matches = []
        cutoff_time = datetime.now() - timedelta(hours=4)  # Consider matches active if updated in last 4 hours
        
        for match_data in self.match_cache.values():
            if match_data['last_update'] > cutoff_time:
                active_matches.append(match_data)
        
        return active_matches


class TennisWebSocketClient:
    """Production WebSocket client for API-Tennis live data"""
    
    def __init__(self, api_key: str = None, timezone: str = "UTC"):
        """Initialize WebSocket client"""
        self.config = get_config()
        self.api_key = api_key or self.config.API_TENNIS_KEY
        self.timezone = timezone
        
        # Connection settings
        self.ws_url = "wss://wss.api-tennis.com/live"
        self.connection_state = WSConnectionState.DISCONNECTED
        self.websocket = None
        
        # Reconnection settings
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5  # Start with 5 seconds
        self.max_reconnect_delay = 300  # Max 5 minutes
        self.reconnect_attempt = 0
        
        # Data processing
        self.data_processor = LiveDataProcessor()
        self.event_queue = Queue()
        self.stop_event = threading.Event()
        
        # Callbacks
        self.event_callbacks: List[Callable] = []
        self.connection_callbacks: List[Callable] = []
        
        # Statistics
        self.stats = {
            'events_received': 0,
            'events_processed': 0,
            'connection_errors': 0,
            'last_event_time': None,
            'connection_start_time': None,
            'total_uptime': 0
        }
        
        # Filters
        self.tournament_filter = None
        self.match_filter = None
        self.player_filter = None
        
        if not self.api_key:
            logger.warning("API_TENNIS_KEY not configured - WebSocket connection will fail")
    
    def add_event_callback(self, callback: Callable[[LiveMatchEvent], None]):
        """Add callback for live events"""
        self.event_callbacks.append(callback)
    
    def add_connection_callback(self, callback: Callable[[WSConnectionState], None]):
        """Add callback for connection state changes"""
        self.connection_callbacks.append(callback)
    
    def set_filters(self, tournament_key: int = None, match_key: int = None, player_key: int = None):
        """Set filters for specific data"""
        self.tournament_filter = tournament_key
        self.match_filter = match_key
        self.player_filter = player_key
    
    def _build_connection_url(self) -> str:
        """Build WebSocket connection URL with parameters"""
        params = {
            'APIkey': self.api_key,
            'timezone': self.timezone
        }
        
        # Add filters if set
        if self.tournament_filter:
            params['tournament_key'] = self.tournament_filter
        if self.match_filter:
            params['match_key'] = self.match_filter
        if self.player_filter:
            params['player_key'] = self.player_filter
        
        query_string = urllib.parse.urlencode(params)
        return f"{self.ws_url}?{query_string}"
    
    def _update_connection_state(self, state: WSConnectionState):
        """Update connection state and notify callbacks"""
        if self.connection_state != state:
            old_state = self.connection_state
            self.connection_state = state
            
            logger.info(f"WebSocket state changed: {old_state.value} -> {state.value}")
            
            # Update statistics
            if state == WSConnectionState.CONNECTED:
                self.stats['connection_start_time'] = datetime.now()
                self.reconnect_attempt = 0
            elif state == WSConnectionState.ERROR:
                self.stats['connection_errors'] += 1
            
            # Notify callbacks
            for callback in self.connection_callbacks:
                try:
                    callback(state)
                except Exception as e:
                    logger.error(f"Error in connection callback: {e}")
    
    async def _connect(self) -> bool:
        """Establish WebSocket connection"""
        try:
            self._update_connection_state(WSConnectionState.CONNECTING)
            
            url = self._build_connection_url()
            logger.info(f"Connecting to WebSocket: {url}")
            
            # Create SSL context for secure connection
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False  # API-Tennis certificate handling
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Connect with timeout
            self.websocket = await websockets.connect(
                url,
                ssl=ssl_context,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            
            self._update_connection_state(WSConnectionState.CONNECTED)
            logger.info("WebSocket connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self._update_connection_state(WSConnectionState.ERROR)
            return False
    
    async def _handle_message(self, message: str):
        """Handle incoming WebSocket message"""
        try:
            self.stats['events_received'] += 1
            self.stats['last_event_time'] = datetime.now()
            
            # Parse JSON message
            data = json.loads(message)
            
            # Process the event
            event = self.data_processor.process_live_event(data)
            
            if event:
                self.stats['events_processed'] += 1
                
                # Add to queue for async processing
                self.event_queue.put(event)
                
                # Notify callbacks
                for callback in self.event_callbacks:
                    try:
                        callback(event)
                    except Exception as e:
                        logger.error(f"Error in event callback: {e}")
            
            logger.debug(f"Processed WebSocket message for match {data.get('event_key', 'unknown')}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse WebSocket message: {e}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    async def _listen_loop(self):
        """Main WebSocket listening loop"""
        try:
            async for message in self.websocket:
                if self.stop_event.is_set():
                    break
                
                await self._handle_message(message)
                
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed: {e}")
            self._update_connection_state(WSConnectionState.DISCONNECTED)
        except Exception as e:
            logger.error(f"Error in WebSocket listen loop: {e}")
            self._update_connection_state(WSConnectionState.ERROR)
    
    async def _reconnect_loop(self):
        """Handle automatic reconnection"""
        while not self.stop_event.is_set():
            if self.connection_state in [WSConnectionState.DISCONNECTED, WSConnectionState.ERROR]:
                if self.reconnect_attempt < self.max_reconnect_attempts:
                    self._update_connection_state(WSConnectionState.RECONNECTING)
                    
                    logger.info(f"Attempting reconnection {self.reconnect_attempt + 1}/{self.max_reconnect_attempts}")
                    
                    if await self._connect():
                        # Connection successful, start listening
                        await self._listen_loop()
                    else:
                        # Connection failed, wait and retry
                        self.reconnect_attempt += 1
                        delay = min(self.reconnect_delay * (2 ** self.reconnect_attempt), self.max_reconnect_delay)
                        logger.info(f"Reconnection failed, waiting {delay} seconds...")
                        await asyncio.sleep(delay)
                else:
                    logger.error("Max reconnection attempts reached, giving up")
                    break
            else:
                # Wait a bit before checking again
                await asyncio.sleep(1)
    
    async def start(self):
        """Start WebSocket client"""
        logger.info("Starting Tennis WebSocket client...")
        
        if not self.api_key:
            raise ValueError("API_TENNIS_KEY is required for WebSocket connection")
        
        self.stop_event.clear()
        
        # Start initial connection
        if await self._connect():
            await self._listen_loop()
        
        # Start reconnection loop if connection fails
        await self._reconnect_loop()
    
    def start_threaded(self):
        """Start WebSocket client in a separate thread"""
        def run_async():
            try:
                asyncio.run(self.start())
            except Exception as e:
                logger.error(f"Error in WebSocket thread: {e}")
        
        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()
        return thread
    
    async def stop(self):
        """Stop WebSocket client"""
        logger.info("Stopping Tennis WebSocket client...")
        
        self.stop_event.set()
        self._update_connection_state(WSConnectionState.CLOSING)
        
        if self.websocket:
            await self.websocket.close()
        
        self._update_connection_state(WSConnectionState.DISCONNECTED)
        
        # Update uptime statistics
        if self.stats['connection_start_time']:
            uptime = datetime.now() - self.stats['connection_start_time']
            self.stats['total_uptime'] += uptime.total_seconds()
    
    def stop_threaded(self):
        """Stop WebSocket client running in thread"""
        self.stop_event.set()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        stats = self.stats.copy()
        stats['connection_state'] = self.connection_state.value
        stats['active_matches'] = len(self.data_processor.get_active_matches())
        
        if self.stats['connection_start_time']:
            uptime = datetime.now() - self.stats['connection_start_time']
            stats['current_uptime_seconds'] = uptime.total_seconds()
        
        return stats
    
    def get_live_events(self, max_events: int = 100) -> List[LiveMatchEvent]:
        """Get recent live events from queue"""
        events = []
        try:
            while len(events) < max_events:
                event = self.event_queue.get_nowait()
                events.append(event)
        except Empty:
            pass
        
        return events
    
    def get_active_matches(self) -> List[Dict[str, Any]]:
        """Get currently active matches"""
        return self.data_processor.get_active_matches()
    
    def get_match_state(self, event_key: int) -> Optional[Dict[str, Any]]:
        """Get current state of specific match"""
        return self.data_processor.get_match_state(event_key)


class WebSocketManager:
    """Manages multiple WebSocket connections and data routing"""
    
    def __init__(self):
        self.clients: Dict[str, TennisWebSocketClient] = {}
        self.event_router = EventRouter()
        self.is_running = False
    
    def create_client(self, client_id: str, api_key: str = None, **kwargs) -> TennisWebSocketClient:
        """Create a new WebSocket client"""
        client = TennisWebSocketClient(api_key=api_key, **kwargs)
        client.add_event_callback(lambda event: self.event_router.route_event(client_id, event))
        
        self.clients[client_id] = client
        logger.info(f"Created WebSocket client: {client_id}")
        return client
    
    def start_all_clients(self):
        """Start all registered clients"""
        self.is_running = True
        for client_id, client in self.clients.items():
            logger.info(f"Starting WebSocket client: {client_id}")
            client.start_threaded()
    
    def stop_all_clients(self):
        """Stop all registered clients"""
        self.is_running = False
        for client_id, client in self.clients.items():
            logger.info(f"Stopping WebSocket client: {client_id}")
            client.stop_threaded()
    
    def get_aggregated_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics from all clients"""
        stats = {
            'total_clients': len(self.clients),
            'total_events_received': 0,
            'total_events_processed': 0,
            'total_connection_errors': 0,
            'clients': {}
        }
        
        for client_id, client in self.clients.items():
            client_stats = client.get_stats()
            stats['clients'][client_id] = client_stats
            stats['total_events_received'] += client_stats.get('events_received', 0)
            stats['total_events_processed'] += client_stats.get('events_processed', 0)
            stats['total_connection_errors'] += client_stats.get('connection_errors', 0)
        
        return stats


class EventRouter:
    """Routes live events to appropriate handlers"""
    
    def __init__(self):
        self.handlers: Dict[str, List[Callable]] = {}
    
    def add_handler(self, event_type: str, handler: Callable):
        """Add event handler"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
    
    def route_event(self, client_id: str, event: LiveMatchEvent):
        """Route event to appropriate handlers"""
        # Route to general handlers
        for handler in self.handlers.get('all', []):
            try:
                handler(client_id, event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
        
        # Route to specific event type handlers
        event_type = self._determine_event_type(event)
        for handler in self.handlers.get(event_type, []):
            try:
                handler(client_id, event)
            except Exception as e:
                logger.error(f"Error in {event_type} handler: {e}")
    
    def _determine_event_type(self, event: LiveMatchEvent) -> str:
        """Determine event type for routing"""
        if event.winner:
            return 'match_finished'
        elif event.live:
            return 'match_live'
        else:
            return 'match_scheduled'


# Signal handling for graceful shutdown
def setup_signal_handlers(ws_manager: WebSocketManager):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        ws_manager.stop_all_clients()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create WebSocket manager
    manager = WebSocketManager()
    setup_signal_handlers(manager)
    
    # Create main client
    main_client = manager.create_client('main')
    
    # Add event handler
    def handle_live_event(client_id: str, event: LiveMatchEvent):
        print(f"[{client_id}] Live event: {event.first_player} vs {event.second_player} - {event.status}")
    
    manager.event_router.add_handler('all', handle_live_event)
    
    # Start clients
    manager.start_all_clients()
    
    try:
        while True:
            time.sleep(10)
            stats = manager.get_aggregated_stats()
            print(f"Stats: {stats['total_events_processed']} events processed")
    except KeyboardInterrupt:
        manager.stop_all_clients()