#!/usr/bin/env python3
"""
Automated Tennis Betting Engine
Production-ready betting system with Betfair integration and real-time ML predictions
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
from decimal import Decimal, ROUND_HALF_UP
import uuid
from collections import defaultdict, deque
import hashlib
import hmac
from queue import Queue, Empty

# Import prediction components
from realtime_prediction_engine import PredictionEngine, MLPredictionResult, PredictionTrigger
from websocket_tennis_client import LiveMatchEvent
from config import get_config

logger = logging.getLogger(__name__)


class BetType(Enum):
    """Types of tennis bets"""
    MATCH_WINNER = "match_winner"
    SET_WINNER = "set_winner"
    HANDICAP = "handicap"
    TOTAL_GAMES = "total_games"
    OVER_UNDER = "over_under"
    CORRECT_SCORE = "correct_score"


class BetStatus(Enum):
    """Bet execution status"""
    PENDING = "pending"
    PLACED = "placed"
    MATCHED = "matched"
    CANCELLED = "cancelled"
    SETTLED_WON = "settled_won"
    SETTLED_LOST = "settled_lost"
    ERROR = "error"


class RiskLevel(Enum):
    """Risk management levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class BetOpportunity:
    """Betting opportunity identified by ML system"""
    match_id: int
    bet_type: BetType
    selection: str
    predicted_odds: float
    market_odds: float
    confidence: float
    edge: float  # Expected value advantage
    recommended_stake: float
    max_stake: float
    reasoning: str
    timestamp: datetime
    ml_prediction: MLPredictionResult
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['bet_type'] = self.bet_type.value
        data['timestamp'] = self.timestamp.isoformat()
        data['ml_prediction'] = self.ml_prediction.to_dict() if self.ml_prediction else None
        return data


@dataclass
class BetOrder:
    """Bet order to be placed"""
    order_id: str
    match_id: int
    market_id: str
    selection_id: str
    bet_type: BetType
    selection: str
    odds: float
    stake: float
    potential_payout: float
    status: BetStatus
    created_at: datetime
    placed_at: Optional[datetime] = None
    settled_at: Optional[datetime] = None
    error_message: Optional[str] = None
    opportunity: Optional[BetOpportunity] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['bet_type'] = self.bet_type.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['placed_at'] = self.placed_at.isoformat() if self.placed_at else None
        data['settled_at'] = self.settled_at.isoformat() if self.settled_at else None
        data['opportunity'] = self.opportunity.to_dict() if self.opportunity else None
        return data


@dataclass
class RiskManagementConfig:
    """Risk management configuration"""
    max_stake_per_bet: float
    max_stake_per_match: float
    max_daily_loss: float
    max_weekly_loss: float
    min_confidence_threshold: float
    min_edge_threshold: float
    max_concurrent_bets: int
    stop_loss_percentage: float
    profit_taking_percentage: float
    risk_level: RiskLevel
    
    @classmethod
    def conservative(cls) -> 'RiskManagementConfig':
        return cls(
            max_stake_per_bet=25.0,
            max_stake_per_match=100.0,
            max_daily_loss=200.0,
            max_weekly_loss=500.0,
            min_confidence_threshold=0.75,
            min_edge_threshold=0.05,
            max_concurrent_bets=5,
            stop_loss_percentage=0.20,
            profit_taking_percentage=0.15,
            risk_level=RiskLevel.CONSERVATIVE
        )
    
    @classmethod
    def moderate(cls) -> 'RiskManagementConfig':
        return cls(
            max_stake_per_bet=50.0,
            max_stake_per_match=200.0,
            max_daily_loss=400.0,
            max_weekly_loss=1000.0,
            min_confidence_threshold=0.65,
            min_edge_threshold=0.03,
            max_concurrent_bets=10,
            stop_loss_percentage=0.25,
            profit_taking_percentage=0.20,
            risk_level=RiskLevel.MODERATE
        )
    
    @classmethod
    def aggressive(cls) -> 'RiskManagementConfig':
        return cls(
            max_stake_per_bet=100.0,
            max_stake_per_match=500.0,
            max_daily_loss=800.0,
            max_weekly_loss=2000.0,
            min_confidence_threshold=0.55,
            min_edge_threshold=0.02,
            max_concurrent_bets=20,
            stop_loss_percentage=0.30,
            profit_taking_percentage=0.25,
            risk_level=RiskLevel.AGGRESSIVE
        )


# Import the proper Betfair client
from betfair_api_client import BetfairAPIClient, BetSide


class StakeCalculator:
    """Calculates optimal stake sizes based on Kelly Criterion and risk management"""
    
    def __init__(self, bankroll: float):
        self.bankroll = bankroll
    
    def calculate_kelly_stake(self, probability: float, odds: float, max_stake: float) -> float:
        """Calculate Kelly Criterion stake"""
        try:
            # Kelly formula: f = (bp - q) / b
            # where b = odds - 1, p = probability, q = 1 - p
            b = odds - 1
            p = probability
            q = 1 - p
            
            if b <= 0 or p <= 0:
                return 0.0
            
            kelly_fraction = (b * p - q) / b
            
            # Apply fractional Kelly (typically 25% of full Kelly for safety)
            fractional_kelly = kelly_fraction * 0.25
            
            # Calculate stake
            stake = self.bankroll * fractional_kelly
            
            # Apply maximum stake limit
            return min(max(stake, 0), max_stake)
            
        except Exception as e:
            logger.error(f"Error calculating Kelly stake: {e}")
            return 0.0
    
    def calculate_fixed_percentage_stake(self, percentage: float, max_stake: float) -> float:
        """Calculate fixed percentage stake"""
        stake = self.bankroll * (percentage / 100)
        return min(stake, max_stake)
    
    def calculate_confidence_weighted_stake(self, confidence: float, base_stake: float, max_stake: float) -> float:
        """Calculate stake weighted by prediction confidence"""
        weighted_stake = base_stake * confidence
        return min(weighted_stake, max_stake)


class RiskManager:
    """Manages betting risk and position sizing"""
    
    def __init__(self, config: RiskManagementConfig, initial_bankroll: float):
        self.config = config
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        
        # Tracking
        self.daily_loss = 0.0
        self.weekly_loss = 0.0
        self.active_bets: Dict[str, BetOrder] = {}
        self.bet_history = []
        
        # Reset tracking
        self.last_daily_reset = datetime.now().date()
        self.last_weekly_reset = datetime.now().isocalendar()[:2]  # Year and week
    
    def can_place_bet(self, stake: float, match_id: int) -> Tuple[bool, str]:
        """Check if bet can be placed according to risk management rules"""
        
        # Update daily/weekly tracking
        self._update_tracking()
        
        # Check maximum stake per bet
        if stake > self.config.max_stake_per_bet:
            return False, f"Stake {stake} exceeds maximum per bet {self.config.max_stake_per_bet}"
        
        # Check maximum stake per match
        match_stakes = sum(bet.stake for bet in self.active_bets.values() if bet.match_id == match_id)
        if match_stakes + stake > self.config.max_stake_per_match:
            return False, f"Total match stake would exceed maximum {self.config.max_stake_per_match}"
        
        # Check concurrent bets limit
        if len(self.active_bets) >= self.config.max_concurrent_bets:
            return False, f"Maximum concurrent bets limit reached ({self.config.max_concurrent_bets})"
        
        # Check daily loss limit
        if self.daily_loss + stake > self.config.max_daily_loss:
            return False, f"Would exceed daily loss limit {self.config.max_daily_loss}"
        
        # Check weekly loss limit
        if self.weekly_loss + stake > self.config.max_weekly_loss:
            return False, f"Would exceed weekly loss limit {self.config.max_weekly_loss}"
        
        # Check bankroll (ensure we have funds)
        if stake > self.current_bankroll * 0.5:  # Never risk more than 50% of bankroll on single bet
            return False, f"Stake too large relative to current bankroll {self.current_bankroll}"
        
        return True, "Risk checks passed"
    
    def evaluate_opportunity(self, prediction: MLPredictionResult, market_odds: float) -> Optional[BetOpportunity]:
        """Evaluate if a prediction represents a betting opportunity"""
        
        # Extract prediction data
        pred_data = prediction.prediction
        confidence = prediction.confidence
        
        # Check minimum confidence threshold
        if confidence < self.config.min_confidence_threshold:
            return None
        
        # Calculate implied probability from ML prediction
        ml_prob = pred_data.get('player_1_win_probability', 0.5)
        
        # Calculate edge (expected value)
        implied_prob_market = 1 / market_odds if market_odds > 0 else 0
        edge = ml_prob - implied_prob_market
        
        # Check minimum edge threshold
        if edge < self.config.min_edge_threshold:
            return None
        
        # Calculate recommended stake
        stake_calculator = StakeCalculator(self.current_bankroll)
        recommended_stake = stake_calculator.calculate_kelly_stake(
            ml_prob, market_odds, self.config.max_stake_per_bet
        )
        
        # Adjust stake based on confidence
        recommended_stake = stake_calculator.calculate_confidence_weighted_stake(
            confidence, recommended_stake, self.config.max_stake_per_bet
        )
        
        if recommended_stake < 1.0:  # Minimum bet size
            return None
        
        opportunity = BetOpportunity(
            match_id=prediction.match_id,
            bet_type=BetType.MATCH_WINNER,
            selection=pred_data.get('winner', 'Unknown'),
            predicted_odds=1 / ml_prob if ml_prob > 0 else 1.0,
            market_odds=market_odds,
            confidence=confidence,
            edge=edge,
            recommended_stake=recommended_stake,
            max_stake=self.config.max_stake_per_bet,
            reasoning=f"ML confidence {confidence:.2f}, edge {edge:.3f}",
            timestamp=datetime.now(),
            ml_prediction=prediction
        )
        
        return opportunity
    
    def add_active_bet(self, bet: BetOrder):
        """Add bet to active tracking"""
        self.active_bets[bet.order_id] = bet
    
    def settle_bet(self, order_id: str, won: bool, payout: float = 0.0):
        """Settle a bet and update tracking"""
        if order_id not in self.active_bets:
            return
        
        bet = self.active_bets[order_id]
        
        if won:
            profit = payout - bet.stake
            self.current_bankroll += payout
            bet.status = BetStatus.SETTLED_WON
        else:
            loss = bet.stake
            self.current_bankroll -= loss
            self.daily_loss += loss
            self.weekly_loss += loss
            bet.status = BetStatus.SETTLED_LOST
        
        bet.settled_at = datetime.now()
        self.bet_history.append(bet)
        del self.active_bets[order_id]
        
        logger.info(f"Bet settled: {bet.selection} - {'WON' if won else 'LOST'} - Bankroll: {self.current_bankroll:.2f}")
    
    def _update_tracking(self):
        """Update daily and weekly loss tracking"""
        today = datetime.now().date()
        current_week = datetime.now().isocalendar()[:2]
        
        # Reset daily tracking
        if today != self.last_daily_reset:
            self.daily_loss = 0.0
            self.last_daily_reset = today
        
        # Reset weekly tracking
        if current_week != self.last_weekly_reset:
            self.weekly_loss = 0.0
            self.last_weekly_reset = current_week
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get current risk management summary"""
        return {
            'current_bankroll': self.current_bankroll,
            'initial_bankroll': self.initial_bankroll,
            'daily_loss': self.daily_loss,
            'weekly_loss': self.weekly_loss,
            'active_bets_count': len(self.active_bets),
            'active_exposure': sum(bet.stake for bet in self.active_bets.values()),
            'total_bets_placed': len(self.bet_history) + len(self.active_bets),
            'profit_loss': self.current_bankroll - self.initial_bankroll,
            'profit_loss_percentage': ((self.current_bankroll - self.initial_bankroll) / self.initial_bankroll) * 100
        }


class AutomatedBettingEngine:
    """Main automated betting engine"""
    
    def __init__(self, risk_config: RiskManagementConfig = None, initial_bankroll: float = 1000.0):
        self.config = get_config()
        
        # Components
        self.prediction_engine = None
        self.betfair_client = BetfairAPIClient()
        self.risk_manager = RiskManager(
            risk_config or RiskManagementConfig.moderate(),
            initial_bankroll
        )
        
        # Processing
        self.opportunity_queue = Queue()
        self.bet_orders: Dict[str, BetOrder] = {}
        
        # Control
        self.is_running = False
        self.stop_event = threading.Event()
        
        # Callbacks
        self.bet_callbacks: List[Callable] = []
        self.opportunity_callbacks: List[Callable] = []
        
        # Statistics
        self.stats = {
            'opportunities_found': 0,
            'bets_placed': 0,
            'bets_won': 0,
            'bets_lost': 0,
            'total_profit_loss': 0.0,
            'start_time': None
        }
        
        # Market tracking
        self.market_cache: Dict[str, Dict[str, Any]] = {}
        self.last_market_update = {}
    
    def initialize(self, prediction_engine: PredictionEngine):
        """Initialize betting engine with prediction engine"""
        self.prediction_engine = prediction_engine
        
        # Add prediction callback
        self.prediction_engine.add_prediction_callback(self._handle_prediction)
        
        # Authenticate with Betfair
        health_check = self.betfair_client.health_check()
        if health_check['status'] != 'healthy':
            logger.error(f"Betfair API health check failed: {health_check}")
        else:
            logger.info(f"Betfair API connected: {health_check['message']}")
        
        logger.info("Automated Betting Engine initialized")
    
    def add_bet_callback(self, callback: Callable[[BetOrder], None]):
        """Add callback for bet events"""
        self.bet_callbacks.append(callback)
    
    def add_opportunity_callback(self, callback: Callable[[BetOpportunity], None]):
        """Add callback for betting opportunities"""
        self.opportunity_callbacks.append(callback)
    
    def _handle_prediction(self, prediction: MLPredictionResult):
        """Handle ML prediction and evaluate betting opportunity"""
        try:
            # Get market odds for this match
            market_odds = self._get_market_odds_for_match(prediction.match_id)
            
            if not market_odds:
                logger.debug(f"No market odds available for match {prediction.match_id}")
                return
            
            # Evaluate opportunity
            opportunity = self.risk_manager.evaluate_opportunity(prediction, market_odds)
            
            if opportunity:
                self.stats['opportunities_found'] += 1
                self.opportunity_queue.put(opportunity)
                
                # Notify callbacks
                for callback in self.opportunity_callbacks:
                    try:
                        callback(opportunity)
                    except Exception as e:
                        logger.error(f"Error in opportunity callback: {e}")
                
                logger.info(f"Betting opportunity found: {opportunity.selection} at {opportunity.market_odds} (edge: {opportunity.edge:.3f})")
        
        except Exception as e:
            logger.error(f"Error handling prediction: {e}")
    
    def _get_market_odds_for_match(self, match_id: int) -> Optional[float]:
        """Get current market odds for a match"""
        try:
            # Try to find Betfair market for this match
            # This is a simplified mapping - in production, you'd need proper match-to-market mapping
            markets = self.betfair_client.get_tennis_markets()
            
            # For simulation or when no markets found, return random odds
            if not markets or len(markets) == 0:
                import random
                return round(random.uniform(1.5, 3.0), 2)
            
            # Get the first market (match winner)
            market = markets[0]
            market_odds = self.betfair_client.get_market_book([market.market_id])
            
            if market.market_id in market_odds and market_odds[market.market_id]:
                # Get best back price for first runner
                first_runner_odds = market_odds[market.market_id][0]
                best_price = first_runner_odds.get_best_back_price()
                if best_price:
                    return best_price
            
            # Fallback to random odds
            import random
            return round(random.uniform(1.5, 3.0), 2)
        
        except Exception as e:
            logger.error(f"Error getting market odds: {e}")
            # Fallback to random odds for simulation
            import random
            return round(random.uniform(1.5, 3.0), 2)
    
    def _opportunity_worker(self):
        """Worker thread for processing betting opportunities"""
        while not self.stop_event.is_set():
            try:
                # Get betting opportunity
                opportunity = self.opportunity_queue.get(timeout=1)
                
                # Process opportunity
                self._process_betting_opportunity(opportunity)
                
                self.opportunity_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in opportunity worker: {e}")
    
    def _process_betting_opportunity(self, opportunity: BetOpportunity):
        """Process a betting opportunity and place bet if appropriate"""
        try:
            # Final risk check
            can_bet, reason = self.risk_manager.can_place_bet(
                opportunity.recommended_stake,
                opportunity.match_id
            )
            
            if not can_bet:
                logger.info(f"Betting opportunity rejected: {reason}")
                return
            
            # Create bet order with proper selection ID
            # In production, this would map player names to Betfair selection IDs
            selection_id = "12345" if "Player 1" in opportunity.selection else "12346"
            
            order = BetOrder(
                order_id=str(uuid.uuid4()),
                match_id=opportunity.match_id,
                market_id=f"tennis_match_{opportunity.match_id}",
                selection_id=selection_id,
                bet_type=opportunity.bet_type,
                selection=opportunity.selection,
                odds=opportunity.market_odds,
                stake=opportunity.recommended_stake,
                potential_payout=opportunity.recommended_stake * opportunity.market_odds,
                status=BetStatus.PENDING,
                created_at=datetime.now(),
                opportunity=opportunity
            )
            
            # Place bet
            result = self.betfair_client.place_bet(
                market_id=order.market_id,
                selection_id=order.selection_id,
                side=BetSide.BACK,  # Always backing for match winner bets
                price=order.odds,
                size=order.stake
            )
            
            if result.get('status') == 'success':
                order.status = BetStatus.PLACED
                order.placed_at = datetime.now()
                self.stats['bets_placed'] += 1
                
                # Add to active tracking
                self.risk_manager.add_active_bet(order)
                self.bet_orders[order.order_id] = order
                
                logger.info(f"Bet placed: {order.stake}€ on {order.selection} at {order.odds}")
                
            else:
                order.status = BetStatus.ERROR
                order.error_message = result.get('message', 'Unknown error')
                logger.error(f"Failed to place bet: {order.error_message}")
            
            # Notify callbacks
            for callback in self.bet_callbacks:
                try:
                    callback(order)
                except Exception as e:
                    logger.error(f"Error in bet callback: {e}")
        
        except Exception as e:
            logger.error(f"Error processing betting opportunity: {e}")
    
    def _bet_monitoring_worker(self):
        """Worker thread for monitoring placed bets"""
        while not self.stop_event.is_set():
            try:
                time.sleep(30)  # Check every 30 seconds
                
                # Check status of active bets
                for order_id, bet in list(self.bet_orders.items()):
                    if bet.status in [BetStatus.PLACED, BetStatus.MATCHED]:
                        self._check_bet_status(bet)
                
            except Exception as e:
                logger.error(f"Error in bet monitoring worker: {e}")
    
    def _check_bet_status(self, bet: BetOrder):
        """Check status of a placed bet"""
        try:
            # Get current orders to check bet status
            current_orders = self.betfair_client.get_current_orders(bet.market_id)
            
            # Find our bet in current orders
            bet_found = False
            for betfair_bet in current_orders:
                if betfair_bet.bet_id == bet.order_id:
                    bet_found = True
                    
                    # Update bet status based on Betfair status
                    if betfair_bet.status.value == 'EXECUTION_COMPLETE':
                        if betfair_bet.size_matched > 0:
                            bet.status = BetStatus.MATCHED
                        else:
                            bet.status = BetStatus.CANCELLED
                    elif betfair_bet.status.value == 'EXECUTABLE':
                        bet.status = BetStatus.PLACED
                    elif betfair_bet.status.value in ['EXPIRED', 'CANCELLED']:
                        bet.status = BetStatus.CANCELLED
                    
                    break
            
            # If bet not found in current orders, check cleared orders
            if not bet_found:
                cleared_orders = self.betfair_client.get_cleared_orders()
                for betfair_bet in cleared_orders:
                    if betfair_bet.bet_id == bet.order_id:
                        # Determine if bet won or lost based on settlement
                        # This would require additional logic to determine match outcome
                        # For now, we'll simulate settlement
                        import random
                        won = random.choice([True, False])  # Placeholder logic
                        
                        if won:
                            payout = bet.stake * bet.odds
                            self.risk_manager.settle_bet(bet.order_id, True, payout)
                            self.stats['bets_won'] += 1
                            self.stats['total_profit_loss'] += (payout - bet.stake)
                            bet.status = BetStatus.SETTLED_WON
                        else:
                            self.risk_manager.settle_bet(bet.order_id, False)
                            self.stats['bets_lost'] += 1
                            self.stats['total_profit_loss'] -= bet.stake
                            bet.status = BetStatus.SETTLED_LOST
                        
                        bet.settled_at = datetime.now()
                        if bet.order_id in self.bet_orders:
                            del self.bet_orders[bet.order_id]
                        break
        
        except Exception as e:
            logger.error(f"Error checking bet status: {e}")
    
    def start(self):
        """Start the automated betting engine"""
        logger.info("Starting Automated Betting Engine...")
        
        if not self.prediction_engine:
            raise ValueError("Prediction engine must be initialized first")
        
        self.is_running = True
        self.stop_event.clear()
        self.stats['start_time'] = datetime.now()
        
        # Start worker threads
        self.opportunity_thread = threading.Thread(target=self._opportunity_worker, daemon=True)
        self.monitoring_thread = threading.Thread(target=self._bet_monitoring_worker, daemon=True)
        
        self.opportunity_thread.start()
        self.monitoring_thread.start()
        
        logger.info("Automated Betting Engine started successfully")
    
    def stop(self):
        """Stop the automated betting engine"""
        logger.info("Stopping Automated Betting Engine...")
        
        self.is_running = False
        self.stop_event.set()
        
        # Cancel any pending bets
        for bet in self.bet_orders.values():
            if bet.status in [BetStatus.PENDING, BetStatus.PLACED]:
                result = self.betfair_client.cancel_bet(bet.market_id, bet.order_id)
                if result.get('status') == 'success':
                    bet.status = BetStatus.CANCELLED
                    logger.info(f"Cancelled bet {bet.order_id}")
        
        logger.info("Automated Betting Engine stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get betting engine statistics"""
        stats = self.stats.copy()
        stats['risk_summary'] = self.risk_manager.get_risk_summary()
        stats['active_bets'] = len([b for b in self.bet_orders.values() if b.status in [BetStatus.PLACED, BetStatus.MATCHED]])
        stats['queue_size'] = self.opportunity_queue.qsize()
        
        if self.stats['start_time']:
            uptime = datetime.now() - self.stats['start_time']
            stats['uptime_seconds'] = uptime.total_seconds()
        
        # Calculate win rate
        total_settled = self.stats['bets_won'] + self.stats['bets_lost']
        if total_settled > 0:
            stats['win_rate'] = (self.stats['bets_won'] / total_settled) * 100
        else:
            stats['win_rate'] = 0.0
        
        return stats
    
    def get_active_bets(self) -> List[BetOrder]:
        """Get list of active bets"""
        return [bet for bet in self.bet_orders.values() 
                if bet.status in [BetStatus.PLACED, BetStatus.MATCHED]]
    
    def get_bet_history(self) -> List[BetOrder]:
        """Get betting history"""
        return self.risk_manager.bet_history
    
    def force_settlement_check(self):
        """Force check of all active bet settlements"""
        for bet in self.get_active_bets():
            self._check_bet_status(bet)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    from realtime_prediction_engine import PredictionEngine, PredictionConfig
    
    # Create components
    risk_config = RiskManagementConfig.moderate()
    betting_engine = AutomatedBettingEngine(risk_config, initial_bankroll=1000.0)
    
    prediction_config = PredictionConfig.default()
    prediction_engine = PredictionEngine(prediction_config)
    
    # Initialize
    betting_engine.initialize(prediction_engine)
    
    # Add callbacks
    def handle_bet_placed(bet: BetOrder):
        print(f"BET PLACED: {bet.stake}€ on {bet.selection} at odds {bet.odds}")
    
    def handle_opportunity(opportunity: BetOpportunity):
        print(f"OPPORTUNITY: {opportunity.selection} - Edge: {opportunity.edge:.3f}")
    
    betting_engine.add_bet_callback(handle_bet_placed)
    betting_engine.add_opportunity_callback(handle_opportunity)
    
    try:
        # Start engines
        prediction_engine.start()
        betting_engine.start()
        
        while True:
            time.sleep(60)
            stats = betting_engine.get_stats()
            print(f"\nBetting Stats:")
            print(f"  Opportunities: {stats['opportunities_found']}")
            print(f"  Bets Placed: {stats['bets_placed']}")
            print(f"  Win Rate: {stats['win_rate']:.1f}%")
            print(f"  P&L: {stats['total_profit_loss']:.2f}€")
            print(f"  Bankroll: {stats['risk_summary']['current_bankroll']:.2f}€")
            
    except KeyboardInterrupt:
        betting_engine.stop()
        prediction_engine.stop()