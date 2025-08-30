#!/usr/bin/env python3
"""
Comprehensive Tennis Betting Integration Service
End-to-end automated tennis betting system integrating ML predictions with Betfair Exchange API
"""

import os
import json
import time
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from queue import Queue, Empty
from collections import defaultdict, deque

# Import core components
try:
    from betfair_api_client import BetfairAPIClient, BetSide
    BETFAIR_AVAILABLE = True
except ImportError:
    BETFAIR_AVAILABLE = False
    logger.warning("Betfair API client not available")

try:
    from automated_betting_engine import AutomatedBettingEngine, RiskManagementConfig, BetOpportunity, BetOrder
    BETTING_ENGINE_AVAILABLE = True
except ImportError:
    BETTING_ENGINE_AVAILABLE = False

try:
    from risk_management_system import RiskManager, create_risk_manager, RiskLevel, Position, Portfolio
    RISK_MANAGEMENT_AVAILABLE = True
except ImportError:
    RISK_MANAGEMENT_AVAILABLE = False

try:
    from enhanced_ml_integration import EnhancedMLPredictor, EnhancedMLOrchestrator
    ENHANCED_ML_AVAILABLE = True
except ImportError:
    ENHANCED_ML_AVAILABLE = False

# Import tennis prediction components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from automated_tennis_prediction_service import AutomatedTennisPredictionService
    TENNIS_PREDICTION_AVAILABLE = True
except ImportError:
    TENNIS_PREDICTION_AVAILABLE = False

try:
    from enhanced_api_tennis_integration import EnhancedAPITennisIntegration
    ENHANCED_API_AVAILABLE = True
except ImportError:
    ENHANCED_API_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegrationStatus(Enum):
    """Integration system status"""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    DEGRADED = "degraded"
    ERROR = "error"
    STOPPED = "stopped"

class PredictionSource(Enum):
    """Source of tennis predictions"""
    AUTOMATED_SERVICE = "automated_service"
    ENHANCED_ML = "enhanced_ml"
    API_INTEGRATION = "api_integration"
    MANUAL = "manual"

@dataclass
class TennisMatch:
    """Tennis match data structure"""
    match_id: str
    player1: str
    player2: str
    tournament: str
    surface: str
    start_time: datetime
    player1_rank: int
    player2_rank: int
    player1_odds: Optional[float] = None
    player2_odds: Optional[float] = None
    market_id: Optional[str] = None
    status: str = "scheduled"
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        return data

@dataclass
class TennisPrediction:
    """Tennis prediction result"""
    match_id: str
    prediction_id: str
    source: PredictionSource
    underdog_player: str
    underdog_probability: float
    confidence: float
    edge: float
    reasoning: str
    timestamp: datetime
    recommended_stake: float = 0.0
    max_stake: float = 0.0
    market_odds: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['source'] = self.source.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

class ComprehensiveTennisBettingIntegration:
    """
    Main integration service that orchestrates all tennis betting components:
    - Tennis match data collection
    - ML prediction generation
    - Risk management
    - Betfair API integration
    - Automated bet placement
    - Performance monitoring
    """
    
    def __init__(self, config_path: str = None, initial_bankroll: float = 10000.0):
        self.config = self._load_config(config_path)
        self.initial_bankroll = initial_bankroll
        self.status = IntegrationStatus.INITIALIZING
        
        # Core components
        self.tennis_prediction_service = None
        self.enhanced_ml_predictor = None
        self.enhanced_api_client = None
        self.betfair_client = None
        self.betting_engine = None
        self.risk_manager = None
        
        # Integration components
        self.match_monitor = None
        self.prediction_queue = Queue()
        self.betting_queue = Queue()
        
        # Threading and control
        self.stop_event = threading.Event()
        self.worker_threads = []
        
        # Statistics and monitoring
        self.stats = {
            'service_start_time': None,
            'matches_monitored': 0,
            'predictions_generated': 0,
            'opportunities_identified': 0,
            'bets_placed': 0,
            'bets_won': 0,
            'bets_lost': 0,
            'total_profit_loss': 0.0,
            'current_bankroll': initial_bankroll,
            'active_positions': 0
        }
        
        # Component health tracking
        self.component_health = {
            'tennis_predictions': 'unknown',
            'enhanced_ml': 'unknown',
            'api_integration': 'unknown',
            'betfair_connection': 'unknown',
            'risk_management': 'unknown',
            'betting_engine': 'unknown'
        }
        
        # Event callbacks
        self.event_callbacks = {
            'match_found': [],
            'prediction_generated': [],
            'opportunity_identified': [],
            'bet_placed': [],
            'bet_settled': [],
            'error_occurred': []
        }
        
        logger.info("Comprehensive Tennis Betting Integration initialized")
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            'risk_level': 'moderate',
            'max_concurrent_bets': 10,
            'min_confidence_threshold': 0.65,
            'min_edge_threshold': 0.03,
            'max_stake_per_bet': 100.0,
            'prediction_sources': ['automated_service', 'enhanced_ml', 'api_integration'],
            'monitoring_interval': 30,  # seconds
            'match_refresh_interval': 300,  # seconds
            'simulation_mode': False,
            'betfair_credentials': {
                'app_key': os.getenv('BETFAIR_APP_KEY', ''),
                'username': os.getenv('BETFAIR_USERNAME', ''),
                'password': os.getenv('BETFAIR_PASSWORD', '')
            },
            'notification_settings': {
                'telegram_enabled': True,
                'email_enabled': False
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                    default_config.update(file_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    async def initialize_system(self) -> Dict[str, Any]:
        """Initialize all system components"""
        logger.info("🚀 Initializing Comprehensive Tennis Betting System")
        
        initialization_results = {
            'timestamp': datetime.now().isoformat(),
            'status': 'initializing',
            'components': {},
            'warnings': [],
            'errors': [],
            'ready_for_operation': False
        }
        
        try:
            # 1. Initialize Tennis Prediction Service
            await self._initialize_tennis_predictions(initialization_results)
            
            # 2. Initialize Enhanced ML Components
            await self._initialize_enhanced_ml(initialization_results)
            
            # 3. Initialize Enhanced API Integration
            await self._initialize_api_integration(initialization_results)
            
            # 4. Initialize Risk Management
            await self._initialize_risk_management(initialization_results)
            
            # 5. Initialize Betfair Connection
            await self._initialize_betfair_connection(initialization_results)
            
            # 6. Initialize Betting Engine
            await self._initialize_betting_engine(initialization_results)
            
            # 7. Perform system health check
            health_check = await self._perform_system_health_check()
            initialization_results['health_check'] = health_check
            
            # Determine overall readiness
            critical_components = ['tennis_predictions', 'risk_management']
            ready_components = sum(1 for comp in critical_components 
                                 if self.component_health.get(comp) == 'ready')
            
            if ready_components >= len(critical_components):
                initialization_results['ready_for_operation'] = True
                initialization_results['status'] = 'ready'
                self.status = IntegrationStatus.READY
            else:
                initialization_results['status'] = 'degraded'
                self.status = IntegrationStatus.DEGRADED
                initialization_results['warnings'].append(
                    f"Only {ready_components}/{len(critical_components)} critical components ready"
                )
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            initialization_results['status'] = 'error'
            initialization_results['errors'].append(str(e))
            self.status = IntegrationStatus.ERROR
        
        logger.info(f"System initialization completed: {initialization_results['status']}")
        return initialization_results
    
    async def _initialize_tennis_predictions(self, results: Dict[str, Any]):
        """Initialize tennis prediction service"""
        try:
            if TENNIS_PREDICTION_AVAILABLE:
                self.tennis_prediction_service = AutomatedTennisPredictionService()
                self.component_health['tennis_predictions'] = 'ready'
                results['components']['tennis_predictions'] = {
                    'status': 'ready',
                    'enhanced_rankings': len(getattr(self.tennis_prediction_service, 'enhanced_rankings', {})),
                    'models_loaded': len(getattr(self.tennis_prediction_service, 'models', {}))
                }
                logger.info("✅ Tennis prediction service initialized")
            else:
                raise ImportError("Tennis prediction service not available")
                
        except Exception as e:
            logger.error(f"Tennis prediction initialization failed: {e}")
            self.component_health['tennis_predictions'] = 'error'
            results['components']['tennis_predictions'] = {'status': 'error', 'error': str(e)}
            results['errors'].append(f"Tennis predictions: {e}")
    
    async def _initialize_enhanced_ml(self, results: Dict[str, Any]):
        """Initialize enhanced ML components"""
        try:
            if ENHANCED_ML_AVAILABLE:
                self.enhanced_ml_predictor = EnhancedMLPredictor()
                self.component_health['enhanced_ml'] = 'ready' if self.enhanced_ml_predictor.models else 'degraded'
                results['components']['enhanced_ml'] = {
                    'status': self.component_health['enhanced_ml'],
                    'models_loaded': len(self.enhanced_ml_predictor.models),
                    'scaler_available': self.enhanced_ml_predictor.scaler is not None
                }
                logger.info("✅ Enhanced ML predictor initialized")
            else:
                self.component_health['enhanced_ml'] = 'unavailable'
                results['components']['enhanced_ml'] = {'status': 'unavailable'}
                results['warnings'].append("Enhanced ML predictor not available")
                
        except Exception as e:
            logger.error(f"Enhanced ML initialization failed: {e}")
            self.component_health['enhanced_ml'] = 'error'
            results['components']['enhanced_ml'] = {'status': 'error', 'error': str(e)}
            results['warnings'].append(f"Enhanced ML: {e}")
    
    async def _initialize_api_integration(self, results: Dict[str, Any]):
        """Initialize enhanced API integration"""
        try:
            if ENHANCED_API_AVAILABLE:
                self.enhanced_api_client = EnhancedAPITennisIntegration()
                self.component_health['api_integration'] = 'ready'
                results['components']['api_integration'] = {
                    'status': 'ready',
                    'client_configured': self.enhanced_api_client.client is not None
                }
                logger.info("✅ Enhanced API integration initialized")
            else:
                self.component_health['api_integration'] = 'unavailable'
                results['components']['api_integration'] = {'status': 'unavailable'}
                results['warnings'].append("Enhanced API integration not available")
                
        except Exception as e:
            logger.error(f"API integration initialization failed: {e}")
            self.component_health['api_integration'] = 'error'
            results['components']['api_integration'] = {'status': 'error', 'error': str(e)}
            results['warnings'].append(f"API integration: {e}")
    
    async def _initialize_risk_management(self, results: Dict[str, Any]):
        """Initialize risk management system"""
        try:
            if RISK_MANAGEMENT_AVAILABLE:
                risk_level = getattr(RiskLevel, self.config['risk_level'].upper(), RiskLevel.MODERATE)
                self.risk_manager = create_risk_manager(risk_level, self.initial_bankroll)
                self.component_health['risk_management'] = 'ready'
                results['components']['risk_management'] = {
                    'status': 'ready',
                    'risk_level': risk_level.value,
                    'initial_bankroll': self.initial_bankroll,
                    'max_stake_per_bet': self.risk_manager.risk_limits.max_stake_per_bet
                }
                logger.info("✅ Risk management system initialized")
            else:
                # Create basic risk manager
                self.risk_manager = self._create_basic_risk_manager()
                self.component_health['risk_management'] = 'basic'
                results['components']['risk_management'] = {'status': 'basic'}
                results['warnings'].append("Using basic risk management")
                
        except Exception as e:
            logger.error(f"Risk management initialization failed: {e}")
            self.component_health['risk_management'] = 'error'
            results['components']['risk_management'] = {'status': 'error', 'error': str(e)}
            results['errors'].append(f"Risk management: {e}")
    
    async def _initialize_betfair_connection(self, results: Dict[str, Any]):
        """Initialize Betfair API connection"""
        try:
            if BETFAIR_AVAILABLE and not self.config.get('simulation_mode', False):
                self.betfair_client = BetfairAPIClient(
                    app_key=self.config['betfair_credentials']['app_key'],
                    username=self.config['betfair_credentials']['username'],
                    password=self.config['betfair_credentials']['password']
                )
                
                # Test connection
                health_check = self.betfair_client.health_check()
                if health_check['status'] == 'healthy':
                    self.component_health['betfair_connection'] = 'ready'
                    results['components']['betfair_connection'] = {
                        'status': 'ready',
                        'mode': health_check['mode'],
                        'balance': health_check.get('balance', 0)
                    }
                else:
                    self.component_health['betfair_connection'] = 'degraded'
                    results['components']['betfair_connection'] = {
                        'status': 'degraded',
                        'error': health_check.get('error')
                    }
                    results['warnings'].append("Betfair connection issues detected")
                
                logger.info("✅ Betfair connection initialized")
            else:
                # Simulation mode
                self.betfair_client = BetfairAPIClient()  # Will run in simulation mode
                self.component_health['betfair_connection'] = 'simulation'
                results['components']['betfair_connection'] = {'status': 'simulation'}
                results['warnings'].append("Running in Betfair simulation mode")
                
        except Exception as e:
            logger.error(f"Betfair initialization failed: {e}")
            self.component_health['betfair_connection'] = 'error'
            results['components']['betfair_connection'] = {'status': 'error', 'error': str(e)}
            results['warnings'].append(f"Betfair connection: {e}")
    
    async def _initialize_betting_engine(self, results: Dict[str, Any]):
        """Initialize automated betting engine"""
        try:
            if BETTING_ENGINE_AVAILABLE and self.risk_manager:
                # Create risk config from our risk manager
                risk_config = RiskManagementConfig.moderate()  # Use default, will be customized
                risk_config.max_stake_per_bet = self.config['max_stake_per_bet']
                risk_config.max_concurrent_bets = self.config['max_concurrent_bets']
                risk_config.min_confidence_threshold = self.config['min_confidence_threshold']
                risk_config.min_edge_threshold = self.config['min_edge_threshold']
                
                self.betting_engine = AutomatedBettingEngine(risk_config, self.initial_bankroll)
                
                # Add callbacks
                self.betting_engine.add_bet_callback(self._on_bet_placed)
                self.betting_engine.add_opportunity_callback(self._on_opportunity_identified)
                
                self.component_health['betting_engine'] = 'ready'
                results['components']['betting_engine'] = {
                    'status': 'ready',
                    'max_stake_per_bet': risk_config.max_stake_per_bet,
                    'max_concurrent_bets': risk_config.max_concurrent_bets
                }
                logger.info("✅ Betting engine initialized")
            else:
                self.component_health['betting_engine'] = 'unavailable'
                results['components']['betting_engine'] = {'status': 'unavailable'}
                results['warnings'].append("Betting engine not available")
                
        except Exception as e:
            logger.error(f"Betting engine initialization failed: {e}")
            self.component_health['betting_engine'] = 'error'
            results['components']['betting_engine'] = {'status': 'error', 'error': str(e)}
            results['warnings'].append(f"Betting engine: {e}")
    
    def _create_basic_risk_manager(self):
        """Create basic risk manager if full system not available"""
        class BasicRiskManager:
            def __init__(self, max_stake: float):
                self.max_stake = max_stake
                self.current_bankroll = 10000.0
            
            def evaluate_bet_request(self, prediction: Dict, market_data: Dict, match_info: Dict) -> Dict:
                confidence = prediction.get('confidence', 0)
                edge = prediction.get('edge', 0)
                
                if confidence < 0.6 or edge < 0.02:
                    return {'approved': False, 'reason': 'Insufficient confidence/edge'}
                
                stake = min(self.max_stake, self.current_bankroll * 0.02)  # 2% of bankroll
                
                return {
                    'approved': True,
                    'stake': stake,
                    'reason': 'Basic risk check passed',
                    'risk_score': 0.5,
                    'warnings': []
                }
        
        return BasicRiskManager(self.config['max_stake_per_bet'])
    
    async def _perform_system_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        health_check = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': 'unknown',
            'components': dict(self.component_health),
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Count component statuses
        ready_count = sum(1 for status in self.component_health.values() if status == 'ready')
        error_count = sum(1 for status in self.component_health.values() if status == 'error')
        total_components = len(self.component_health)
        
        # Determine overall health
        if error_count == 0 and ready_count >= total_components * 0.8:
            health_check['overall_health'] = 'excellent'
        elif error_count == 0 and ready_count >= total_components * 0.6:
            health_check['overall_health'] = 'good'
        elif error_count <= 2:
            health_check['overall_health'] = 'degraded'
        else:
            health_check['overall_health'] = 'critical'
        
        # Generate specific recommendations
        if self.component_health['betfair_connection'] in ['error', 'unavailable']:
            health_check['warnings'].append("Betfair connection issues - betting will be simulated")
        
        if self.component_health['enhanced_ml'] in ['error', 'unavailable']:
            health_check['warnings'].append("Enhanced ML unavailable - using basic predictions")
        
        if ready_count < 3:
            health_check['critical_issues'].append("Insufficient components ready for operation")
        
        return health_check
    
    def start_system(self) -> Dict[str, Any]:
        """Start the integrated tennis betting system"""
        if self.status != IntegrationStatus.READY:
            return {
                'success': False,
                'error': f'System not ready. Current status: {self.status.value}'
            }
        
        logger.info("🚀 Starting Comprehensive Tennis Betting System")
        
        try:
            self.status = IntegrationStatus.RUNNING
            self.stats['service_start_time'] = datetime.now()
            self.stop_event.clear()
            
            # Start worker threads
            self._start_worker_threads()
            
            # Start betting engine if available
            if self.betting_engine:
                # Create a minimal prediction engine for the betting engine
                prediction_engine = self._create_prediction_engine_adapter()
                self.betting_engine.initialize(prediction_engine)
                self.betting_engine.start()
            
            logger.info("✅ Tennis Betting System started successfully")
            return {
                'success': True,
                'status': 'running',
                'start_time': self.stats['service_start_time'].isoformat(),
                'active_threads': len(self.worker_threads)
            }
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            self.status = IntegrationStatus.ERROR
            return {
                'success': False,
                'error': str(e)
            }
    
    def _start_worker_threads(self):
        """Start worker threads for system operation"""
        # Match monitoring thread
        match_thread = threading.Thread(target=self._match_monitoring_worker, daemon=True)
        match_thread.start()
        self.worker_threads.append(match_thread)
        
        # Prediction processing thread
        prediction_thread = threading.Thread(target=self._prediction_processing_worker, daemon=True)
        prediction_thread.start()
        self.worker_threads.append(prediction_thread)
        
        # System monitoring thread
        monitoring_thread = threading.Thread(target=self._system_monitoring_worker, daemon=True)
        monitoring_thread.start()
        self.worker_threads.append(monitoring_thread)
        
        logger.info(f"Started {len(self.worker_threads)} worker threads")
    
    def _match_monitoring_worker(self):
        """Worker thread for monitoring tennis matches"""
        logger.info("Match monitoring worker started")
        
        while not self.stop_event.is_set():
            try:
                matches = self._get_current_matches()
                
                for match in matches:
                    self.stats['matches_monitored'] += 1
                    
                    # Generate predictions for this match
                    predictions = self._generate_predictions_for_match(match)
                    
                    for prediction in predictions:
                        self.prediction_queue.put(prediction)
                        self.stats['predictions_generated'] += 1
                
                # Wait before next scan
                time.sleep(self.config['match_refresh_interval'])
                
            except Exception as e:
                logger.error(f"Match monitoring error: {e}")
                self._notify_error('match_monitoring', e)
                time.sleep(60)  # Wait longer on error
    
    def _prediction_processing_worker(self):
        """Worker thread for processing predictions and identifying opportunities"""
        logger.info("Prediction processing worker started")
        
        while not self.stop_event.is_set():
            try:
                # Get prediction from queue
                prediction = self.prediction_queue.get(timeout=1)
                
                # Get market data for this match
                market_data = self._get_market_data(prediction.match_id)
                
                if market_data:
                    # Evaluate betting opportunity
                    opportunity = self._evaluate_betting_opportunity(prediction, market_data)
                    
                    if opportunity:
                        self.stats['opportunities_identified'] += 1
                        self._notify_event('opportunity_identified', opportunity)
                        
                        # Add to betting queue if betting engine available
                        if self.betting_engine:
                            self.betting_queue.put(opportunity)
                
                self.prediction_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Prediction processing error: {e}")
                self._notify_error('prediction_processing', e)
    
    def _system_monitoring_worker(self):
        """Worker thread for system health monitoring"""
        logger.info("System monitoring worker started")
        
        while not self.stop_event.is_set():
            try:
                # Update statistics
                self._update_system_statistics()
                
                # Check component health
                self._check_component_health()
                
                # Log periodic status
                if hasattr(self, 'stats') and self.stats['service_start_time']:
                    uptime = datetime.now() - self.stats['service_start_time']
                    if uptime.total_seconds() % 3600 < 60:  # Every hour
                        logger.info(f"System running for {uptime}. "
                                  f"Matches: {self.stats['matches_monitored']}, "
                                  f"Predictions: {self.stats['predictions_generated']}, "
                                  f"Opportunities: {self.stats['opportunities_identified']}")
                
                time.sleep(self.config['monitoring_interval'])
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                time.sleep(60)
    
    def _get_current_matches(self) -> List[TennisMatch]:
        """Get current tennis matches from available sources"""
        matches = []
        
        try:
            # Try enhanced API first
            if self.enhanced_api_client:
                api_matches = self.enhanced_api_client.get_enhanced_fixtures_with_rankings()
                matches.extend(self._convert_api_matches(api_matches))
            
            # Try automated service
            elif self.tennis_prediction_service:
                service_matches = self.tennis_prediction_service._get_current_matches()
                matches.extend(self._convert_service_matches(service_matches))
            
        except Exception as e:
            logger.error(f"Error getting matches: {e}")
        
        return matches
    
    def _convert_api_matches(self, api_matches: List[Dict]) -> List[TennisMatch]:
        """Convert API matches to TennisMatch objects"""
        matches = []
        
        for match_data in api_matches:
            try:
                match = TennisMatch(
                    match_id=str(match_data.get('id', uuid.uuid4())),
                    player1=match_data.get('player1', {}).get('name', ''),
                    player2=match_data.get('player2', {}).get('name', ''),
                    tournament=match_data.get('tournament_name', ''),
                    surface=match_data.get('surface', 'Hard'),
                    start_time=datetime.now() + timedelta(hours=1),  # Default
                    player1_rank=match_data.get('player1', {}).get('ranking', 150),
                    player2_rank=match_data.get('player2', {}).get('ranking', 150),
                    metadata=match_data
                )
                matches.append(match)
            except Exception as e:
                logger.debug(f"Failed to convert API match: {e}")
        
        return matches
    
    def _convert_service_matches(self, service_matches: List[Dict]) -> List[TennisMatch]:
        """Convert service matches to TennisMatch objects"""
        matches = []
        
        for match_data in service_matches:
            try:
                match = TennisMatch(
                    match_id=match_data.get('event_key', str(uuid.uuid4())),
                    player1=match_data.get('event_first_player', ''),
                    player2=match_data.get('event_second_player', ''),
                    tournament=match_data.get('tournament_name', ''),
                    surface='Hard',  # Default
                    start_time=datetime.now() + timedelta(hours=1),  # Default
                    player1_rank=match_data.get('player1_rank', 150),
                    player2_rank=match_data.get('player2_rank', 150),
                    metadata=match_data
                )
                matches.append(match)
            except Exception as e:
                logger.debug(f"Failed to convert service match: {e}")
        
        return matches
    
    def _generate_predictions_for_match(self, match: TennisMatch) -> List[TennisPrediction]:
        """Generate predictions for a tennis match from available sources"""
        predictions = []
        
        # Try automated service prediction
        if self.tennis_prediction_service and 'automated_service' in self.config['prediction_sources']:
            try:
                service_prediction = self.tennis_prediction_service._generate_prediction(match.metadata or {})
                if service_prediction:
                    prediction = TennisPrediction(
                        match_id=match.match_id,
                        prediction_id=f"auto_{uuid.uuid4()}",
                        source=PredictionSource.AUTOMATED_SERVICE,
                        underdog_player=service_prediction.get('match_context', {}).get('underdog_name', ''),
                        underdog_probability=service_prediction.get('underdog_second_set_probability', 0.5),
                        confidence=self._convert_confidence_to_float(service_prediction.get('confidence', 'Low')),
                        edge=0.05,  # Default edge
                        reasoning=service_prediction.get('strategic_insights', ['Automated prediction'])[0],
                        timestamp=datetime.now()
                    )
                    predictions.append(prediction)
                    
            except Exception as e:
                logger.debug(f"Automated service prediction failed: {e}")
        
        # Try enhanced ML prediction
        if self.enhanced_ml_predictor and 'enhanced_ml' in self.config['prediction_sources']:
            try:
                # Create mock player data for ML prediction
                player1_data = {'rank': match.player1_rank, 'tour': 'ATP', 'movement': 'same'}
                player2_data = {'rank': match.player2_rank, 'tour': 'ATP', 'movement': 'same'}
                match_data = {'tournament_name': match.tournament, 'surface': match.surface}
                
                features = self.enhanced_ml_predictor.create_enhanced_features(match_data, player1_data, player2_data)
                ml_result = self.enhanced_ml_predictor.predict_with_ensemble(features)
                
                if ml_result.get('success'):
                    underdog = match.player1 if match.player1_rank > match.player2_rank else match.player2
                    prediction = TennisPrediction(
                        match_id=match.match_id,
                        prediction_id=f"ml_{uuid.uuid4()}",
                        source=PredictionSource.ENHANCED_ML,
                        underdog_player=underdog,
                        underdog_probability=ml_result['probability'],
                        confidence=self._convert_confidence_to_float(ml_result.get('confidence_level', 'Low')),
                        edge=max(0.02, ml_result['probability'] - 0.5),
                        reasoning=f"ML ensemble prediction with {ml_result.get('model_agreement', 0):.2f} agreement",
                        timestamp=datetime.now()
                    )
                    predictions.append(prediction)
                    
            except Exception as e:
                logger.debug(f"Enhanced ML prediction failed: {e}")
        
        return predictions
    
    def _convert_confidence_to_float(self, confidence_str: str) -> float:
        """Convert confidence string to float"""
        confidence_map = {
            'Very High': 0.9,
            'High': 0.8,
            'Medium': 0.7,
            'Low': 0.6,
            'Very Low': 0.5
        }
        return confidence_map.get(confidence_str, 0.6)
    
    def _get_market_data(self, match_id: str) -> Optional[Dict[str, Any]]:
        """Get market data for a match"""
        try:
            if self.betfair_client:
                # In a real implementation, this would map match_id to Betfair market
                # For now, return simulated market data
                import random
                return {
                    'market_id': f'market_{match_id}',
                    'odds': round(random.uniform(1.5, 3.0), 2),
                    'liquidity': random.uniform(1000, 10000),
                    'spread': 0.05
                }
        except Exception as e:
            logger.debug(f"Market data retrieval failed: {e}")
        
        return None
    
    def _evaluate_betting_opportunity(self, prediction: TennisPrediction, market_data: Dict) -> Optional[Dict]:
        """Evaluate if prediction represents a betting opportunity"""
        try:
            if not self.risk_manager:
                return None
            
            # Prepare data for risk manager
            prediction_data = {
                'confidence': prediction.confidence,
                'edge': prediction.edge,
                'probability': prediction.underdog_probability
            }
            
            match_info = {
                'match_id': prediction.match_id,
                'player1': prediction.underdog_player,
                'tournament': 'ATP Tournament'
            }
            
            # Evaluate with risk manager
            evaluation = self.risk_manager.evaluate_bet_request(prediction_data, market_data, match_info)
            
            if evaluation.get('approved'):
                return {
                    'prediction': prediction,
                    'market_data': market_data,
                    'evaluation': evaluation,
                    'recommended_stake': evaluation['stake']
                }
        
        except Exception as e:
            logger.error(f"Opportunity evaluation failed: {e}")
        
        return None
    
    def _create_prediction_engine_adapter(self):
        """Create adapter for betting engine prediction interface"""
        class PredictionEngineAdapter:
            def __init__(self, integration_service):
                self.integration_service = integration_service
                self.prediction_callbacks = []
            
            def add_prediction_callback(self, callback):
                self.prediction_callbacks.append(callback)
            
            def start(self):
                pass  # Already handled by integration service
            
            def stop(self):
                pass
        
        return PredictionEngineAdapter(self)
    
    def _update_system_statistics(self):
        """Update system statistics"""
        if self.betting_engine:
            engine_stats = self.betting_engine.get_stats()
            self.stats.update({
                'bets_placed': engine_stats.get('bets_placed', 0),
                'bets_won': engine_stats.get('bets_won', 0),
                'bets_lost': engine_stats.get('bets_lost', 0),
                'total_profit_loss': engine_stats.get('total_profit_loss', 0.0),
                'current_bankroll': engine_stats.get('risk_summary', {}).get('current_bankroll', self.initial_bankroll),
                'active_positions': engine_stats.get('active_bets', 0)
            })
    
    def _check_component_health(self):
        """Check health of all components"""
        # This would implement health checks for each component
        pass
    
    def _notify_event(self, event_type: str, data: Any):
        """Notify event callbacks"""
        for callback in self.event_callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Event callback error: {e}")
    
    def _notify_error(self, component: str, error: Exception):
        """Notify error callbacks"""
        error_data = {
            'component': component,
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        }
        self._notify_event('error_occurred', error_data)
    
    def _on_bet_placed(self, bet_order):
        """Callback for when bet is placed"""
        self._notify_event('bet_placed', bet_order)
        logger.info(f"Bet placed: {bet_order.stake}€ on {bet_order.selection}")
    
    def _on_opportunity_identified(self, opportunity):
        """Callback for when opportunity is identified"""
        self._notify_event('opportunity_identified', opportunity)
        logger.info(f"Opportunity identified: {opportunity.selection} - Edge: {opportunity.edge:.3f}")
    
    def add_event_callback(self, event_type: str, callback):
        """Add event callback"""
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].append(callback)
    
    def stop_system(self) -> Dict[str, Any]:
        """Stop the tennis betting system"""
        logger.info("Stopping Comprehensive Tennis Betting System")
        
        try:
            self.status = IntegrationStatus.STOPPED
            self.stop_event.set()
            
            # Stop betting engine
            if self.betting_engine:
                self.betting_engine.stop()
            
            # Wait for threads to finish
            for thread in self.worker_threads:
                if thread.is_alive():
                    thread.join(timeout=5)
            
            return {
                'success': True,
                'stopped_at': datetime.now().isoformat(),
                'final_stats': self.get_system_stats()
            }
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = dict(self.stats)
        stats['system_status'] = self.status.value
        stats['component_health'] = dict(self.component_health)
        
        if self.stats['service_start_time']:
            stats['uptime_seconds'] = (datetime.now() - self.stats['service_start_time']).total_seconds()
        
        return stats
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'status': self.status.value,
            'components': dict(self.component_health),
            'statistics': self.get_system_stats(),
            'ready_for_operation': self.status in [IntegrationStatus.READY, IntegrationStatus.RUNNING]
        }


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Tennis Betting Integration')
    parser.add_argument('--init', action='store_true', help='Initialize system')
    parser.add_argument('--start', action='store_true', help='Start system')
    parser.add_argument('--health', action='store_true', help='Check system health')
    parser.add_argument('--simulation', action='store_true', help='Run in simulation mode')
    
    args = parser.parse_args()
    
    async def main():
        # Create integration system
        config = {'simulation_mode': args.simulation} if args.simulation else {}
        system = ComprehensiveTennisBettingIntegration()
        system.config.update(config)
        
        if args.init:
            print("🚀 Initializing Comprehensive Tennis Betting System...")
            results = await system.initialize_system()
            print(f"Status: {results['status']}")
            print(f"Ready for operation: {results['ready_for_operation']}")
            
            if results['warnings']:
                print("\n⚠️ Warnings:")
                for warning in results['warnings']:
                    print(f"  • {warning}")
            
            if results['errors']:
                print("\n❌ Errors:")
                for error in results['errors']:
                    print(f"  • {error}")
        
        if args.start:
            print("▶️ Starting tennis betting system...")
            if system.status == IntegrationStatus.INITIALIZING:
                await system.initialize_system()
            
            start_result = system.start_system()
            if start_result['success']:
                print("✅ System started successfully")
                try:
                    # Run for a short time in demo mode
                    time.sleep(10)
                except KeyboardInterrupt:
                    pass
                finally:
                    stop_result = system.stop_system()
                    print(f"🛑 System stopped: {stop_result['success']}")
            else:
                print(f"❌ Failed to start: {start_result['error']}")
        
        if args.health:
            print("🏥 Checking system health...")
            health = system.get_system_health()
            print(f"Overall status: {health['status']}")
            print("Component health:")
            for component, status in health['components'].items():
                status_icon = "✅" if status == 'ready' else "⚠️" if status in ['degraded', 'simulation'] else "❌"
                print(f"  {status_icon} {component}: {status}")
    
    if any([args.init, args.start, args.health]):
        asyncio.run(main())
    else:
        print("Use --help for available options")
        print("Example: python comprehensive_tennis_betting_integration.py --init --simulation")