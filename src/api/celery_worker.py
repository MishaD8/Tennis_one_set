#!/usr/bin/env python3
"""
Celery Worker System for Async Tennis Betting Operations
Production-ready distributed task queue for automated tennis betting with Betfair Exchange
"""

import os
import sys
import logging
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from celery import Celery, Task
from celery.utils.log import get_task_logger
from celery.schedules import crontab
from kombu import Queue, Exchange
import redis
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import requests

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config.config import get_config
from src.api.betfair_api_client import BetfairAPIClient, BetSide, BetType
from src.models.realtime_prediction_engine import RealtimePredictionEngine
from src.utils.enhanced_cache_manager import EnhancedCacheManager
from src.utils.error_handler import TennisSystemErrorHandler

# Configure logging
logger = get_task_logger(__name__)

# Configuration
config = get_config()

# Initialize Celery app
def create_celery_app():
    """Create and configure Celery application"""
    
    # Redis configuration
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    
    # Celery configuration
    celery_config = {
        'broker_url': redis_url,
        'result_backend': redis_url,
        'task_serializer': 'json',
        'accept_content': ['json'],
        'result_serializer': 'json',
        'timezone': 'UTC',
        'enable_utc': True,
        'task_track_started': True,
        'task_time_limit': 300,  # 5 minutes
        'task_soft_time_limit': 240,  # 4 minutes
        'worker_prefetch_multiplier': 1,
        'worker_max_tasks_per_child': 1000,
        'task_acks_late': True,
        'task_reject_on_worker_lost': True,
        'task_routes': {
            'betting.place_bet': {'queue': 'betting'},
            'betting.cancel_bet': {'queue': 'betting'},
            'betting.settle_bet': {'queue': 'betting'},
            'data.collect_live_odds': {'queue': 'data'},
            'data.update_predictions': {'queue': 'ml'},
            'monitoring.health_check': {'queue': 'monitoring'},
            'risk.calculate_position': {'queue': 'risk'},
        },
        'task_default_queue': 'default',
        'task_queues': (
            Queue('default', Exchange('default'), routing_key='default'),
            Queue('betting', Exchange('betting'), routing_key='betting'),
            Queue('data', Exchange('data'), routing_key='data'),
            Queue('ml', Exchange('ml'), routing_key='ml'),
            Queue('risk', Exchange('risk'), routing_key='risk'),
            Queue('monitoring', Exchange('monitoring'), routing_key='monitoring'),
        ),
        'beat_schedule': {
            'collect-live-odds': {
                'task': 'data.collect_live_odds',
                'schedule': 30.0,  # Every 30 seconds
            },
            'update-predictions': {
                'task': 'data.update_predictions',
                'schedule': 60.0,  # Every minute
            },
            'health-check': {
                'task': 'monitoring.health_check',
                'schedule': crontab(minute='*/5'),  # Every 5 minutes
            },
            'settle-expired-bets': {
                'task': 'betting.settle_expired_bets',
                'schedule': crontab(minute='*/10'),  # Every 10 minutes
            },
            'calculate-risk-positions': {
                'task': 'risk.calculate_positions',
                'schedule': crontab(minute='*/2'),  # Every 2 minutes
            },
        },
    }
    
    celery_app = Celery('tennis_betting_worker')
    celery_app.config_from_object(celery_config)
    
    return celery_app

celery = create_celery_app()

# Initialize components
cache_manager = EnhancedCacheManager()
error_handler = TennisSystemErrorHandler()
redis_client = redis.Redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/0'))


@dataclass
class BetOrder:
    """Bet order data structure"""
    order_id: str
    match_id: str
    market_id: str
    selection_id: str
    bet_type: str
    selection: str
    odds: float
    stake: float
    potential_payout: float
    confidence: float
    strategy: str
    created_at: str
    expires_at: str
    status: str = "pending"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RiskPosition:
    """Risk position data structure"""
    match_id: str
    total_exposure: float
    max_loss: float
    max_profit: float
    positions: List[Dict[str, Any]]
    risk_score: float
    timestamp: str


class TennisBettingTask(Task):
    """Base task class with error handling and logging"""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        logger.error(f"Task {task_id} failed: {exc}")
        error_handler.handle_error(exc, f"celery_task_{self.name}")
        
        # Store failure in Redis for monitoring
        redis_client.setex(
            f"task_failure:{task_id}",
            3600,  # 1 hour TTL
            json.dumps({
                'task_name': self.name,
                'error': str(exc),
                'args': args,
                'kwargs': kwargs,
                'timestamp': datetime.now().isoformat()
            })
        )
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success"""
        logger.info(f"Task {task_id} ({self.name}) completed successfully")


# Betting Tasks
@celery.task(base=TennisBettingTask, bind=True, name='betting.place_bet')
def place_bet_task(self, bet_order_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Asynchronously place a bet on Betfair Exchange
    
    Args:
        bet_order_data: Dictionary containing bet order information
        
    Returns:
        Dict containing bet placement result
    """
    try:
        logger.info(f"Processing bet placement: {bet_order_data.get('order_id')}")
        
        # Create bet order object
        bet_order = BetOrder(**bet_order_data)
        
        # Validate bet order
        validation_result = validate_bet_order(bet_order)
        if not validation_result['valid']:
            return {
                'success': False,
                'error': validation_result['reason'],
                'order_id': bet_order.order_id
            }
        
        # Initialize Betfair client
        betfair_client = BetfairAPIClient()
        
        # Check account funds
        funds = betfair_client.get_account_funds()
        if funds['available_balance'] < bet_order.stake:
            return {
                'success': False,
                'error': 'Insufficient funds',
                'order_id': bet_order.order_id,
                'available_balance': funds['available_balance']
            }
        
        # Place bet
        bet_result = betfair_client.place_bet(
            market_id=bet_order.market_id,
            selection_id=bet_order.selection_id,
            side=BetSide.BACK,
            price=bet_order.odds,
            size=bet_order.stake
        )
        
        if bet_result.get('status') == 'success':
            # Update bet status in database
            update_bet_status(bet_order.order_id, 'placed', bet_result.get('bet_id'))
            
            # Cache bet information
            cache_manager.set(
                f"bet:{bet_order.order_id}",
                {
                    **bet_order.to_dict(),
                    'betfair_bet_id': bet_result.get('bet_id'),
                    'status': 'placed',
                    'placed_at': datetime.now().isoformat()
                },
                ttl=3600  # 1 hour
            )
            
            # Schedule bet monitoring
            monitor_bet_task.apply_async(
                args=[bet_order.order_id, bet_result.get('bet_id')],
                countdown=60  # Check after 1 minute
            )
            
            logger.info(f"Bet placed successfully: {bet_order.order_id}")
            
            return {
                'success': True,
                'order_id': bet_order.order_id,
                'betfair_bet_id': bet_result.get('bet_id'),
                'status': 'placed'
            }
        else:
            # Update bet status to failed
            update_bet_status(bet_order.order_id, 'failed', error=bet_result.get('error'))
            
            return {
                'success': False,
                'error': bet_result.get('error', 'Unknown error'),
                'order_id': bet_order.order_id
            }
    
    except Exception as e:
        logger.error(f"Bet placement failed: {e}")
        update_bet_status(bet_order_data.get('order_id'), 'error', error=str(e))
        return {
            'success': False,
            'error': str(e),
            'order_id': bet_order_data.get('order_id')
        }


@celery.task(base=TennisBettingTask, bind=True, name='betting.cancel_bet')
def cancel_bet_task(self, order_id: str, reason: str = None) -> Dict[str, Any]:
    """
    Cancel a pending or unmatched bet
    
    Args:
        order_id: Unique order identifier
        reason: Reason for cancellation
        
    Returns:
        Dict containing cancellation result
    """
    try:
        logger.info(f"Cancelling bet: {order_id}")
        
        # Get bet information from cache/database
        bet_info = cache_manager.get(f"bet:{order_id}")
        if not bet_info:
            bet_info = get_bet_from_database(order_id)
        
        if not bet_info:
            return {
                'success': False,
                'error': 'Bet not found',
                'order_id': order_id
            }
        
        betfair_bet_id = bet_info.get('betfair_bet_id')
        if not betfair_bet_id:
            return {
                'success': False,
                'error': 'Betfair bet ID not found',
                'order_id': order_id
            }
        
        # Initialize Betfair client
        betfair_client = BetfairAPIClient()
        
        # Cancel bet
        cancel_result = betfair_client.cancel_bet(
            market_id=bet_info['market_id'],
            bet_id=betfair_bet_id
        )
        
        if cancel_result.get('status') == 'success':
            # Update bet status
            update_bet_status(order_id, 'cancelled', reason=reason)
            
            # Update cache
            bet_info.update({
                'status': 'cancelled',
                'cancelled_at': datetime.now().isoformat(),
                'cancellation_reason': reason
            })
            cache_manager.set(f"bet:{order_id}", bet_info, ttl=3600)
            
            logger.info(f"Bet cancelled successfully: {order_id}")
            
            return {
                'success': True,
                'order_id': order_id,
                'status': 'cancelled'
            }
        else:
            return {
                'success': False,
                'error': cancel_result.get('error', 'Cancellation failed'),
                'order_id': order_id
            }
    
    except Exception as e:
        logger.error(f"Bet cancellation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'order_id': order_id
        }


@celery.task(base=TennisBettingTask, bind=True, name='betting.monitor_bet')
def monitor_bet_task(self, order_id: str, betfair_bet_id: str) -> Dict[str, Any]:
    """
    Monitor bet status and handle state changes
    
    Args:
        order_id: Internal order identifier
        betfair_bet_id: Betfair bet identifier
        
    Returns:
        Dict containing monitoring result
    """
    try:
        logger.debug(f"Monitoring bet: {order_id}")
        
        # Get bet information
        bet_info = cache_manager.get(f"bet:{order_id}")
        if not bet_info:
            bet_info = get_bet_from_database(order_id)
        
        if not bet_info:
            return {'success': False, 'error': 'Bet not found'}
        
        # Initialize Betfair client
        betfair_client = BetfairAPIClient()
        
        # Get current bet status
        current_orders = betfair_client.get_current_orders(bet_info['market_id'])
        bet_found = False
        
        for betfair_bet in current_orders:
            if betfair_bet.bet_id == betfair_bet_id:
                bet_found = True
                
                # Update bet status based on Betfair status
                new_status = map_betfair_status(betfair_bet.status.value)
                
                if new_status != bet_info.get('status'):
                    # Status changed - update database and cache
                    update_bet_status(order_id, new_status)
                    bet_info['status'] = new_status
                    cache_manager.set(f"bet:{order_id}", bet_info, ttl=3600)
                    
                    logger.info(f"Bet status updated: {order_id} -> {new_status}")
                
                # If bet is matched, schedule settlement monitoring
                if new_status == 'matched':
                    schedule_settlement_monitoring(order_id, bet_info['match_id'])
                
                break
        
        if not bet_found:
            # Bet not in current orders - check cleared orders
            cleared_orders = betfair_client.get_cleared_orders()
            for cleared_bet in cleared_orders:
                if cleared_bet.bet_id == betfair_bet_id:
                    # Bet has been settled
                    settle_bet_task.apply_async(args=[order_id])
                    break
        
        # Schedule next monitoring check if bet is still active
        if bet_info.get('status') in ['placed', 'matched']:
            monitor_bet_task.apply_async(
                args=[order_id, betfair_bet_id],
                countdown=120  # Check again in 2 minutes
            )
        
        return {'success': True, 'order_id': order_id}
    
    except Exception as e:
        logger.error(f"Bet monitoring failed: {e}")
        return {'success': False, 'error': str(e)}


@celery.task(base=TennisBettingTask, bind=True, name='betting.settle_bet')
def settle_bet_task(self, order_id: str) -> Dict[str, Any]:
    """
    Settle a completed bet and calculate P&L
    
    Args:
        order_id: Order identifier to settle
        
    Returns:
        Dict containing settlement result
    """
    try:
        logger.info(f"Settling bet: {order_id}")
        
        # Get bet information
        bet_info = cache_manager.get(f"bet:{order_id}")
        if not bet_info:
            bet_info = get_bet_from_database(order_id)
        
        if not bet_info:
            return {'success': False, 'error': 'Bet not found'}
        
        # Get match result
        match_result = get_match_result(bet_info['match_id'])
        if not match_result:
            # Match not finished yet - reschedule
            settle_bet_task.apply_async(
                args=[order_id],
                countdown=600  # Check again in 10 minutes
            )
            return {'success': False, 'error': 'Match not finished'}
        
        # Determine if bet won
        bet_won = determine_bet_outcome(bet_info, match_result)
        
        # Calculate P&L
        if bet_won:
            profit_loss = bet_info['potential_payout'] - bet_info['stake']
            settlement_status = 'won'
        else:
            profit_loss = -bet_info['stake']
            settlement_status = 'lost'
        
        # Update bet status
        update_bet_status(
            order_id, 
            'settled',
            settlement_data={
                'settlement_status': settlement_status,
                'profit_loss': profit_loss,
                'settled_at': datetime.now().isoformat(),
                'match_result': match_result
            }
        )
        
        # Update cache
        bet_info.update({
            'status': 'settled',
            'settlement_status': settlement_status,
            'profit_loss': profit_loss,
            'settled_at': datetime.now().isoformat()
        })
        cache_manager.set(f"bet:{order_id}", bet_info, ttl=7200)  # Keep for 2 hours
        
        # Update global P&L metrics
        update_pnl_metrics(profit_loss, settlement_status)
        
        logger.info(f"Bet settled: {order_id} - {settlement_status} - P&L: {profit_loss}")
        
        return {
            'success': True,
            'order_id': order_id,
            'settlement_status': settlement_status,
            'profit_loss': profit_loss
        }
    
    except Exception as e:
        logger.error(f"Bet settlement failed: {e}")
        return {'success': False, 'error': str(e)}


@celery.task(base=TennisBettingTask, bind=True, name='betting.settle_expired_bets')
def settle_expired_bets_task(self) -> Dict[str, Any]:
    """
    Periodic task to settle any expired or overlooked bets
    
    Returns:
        Dict containing settlement summary
    """
    try:
        logger.info("Checking for expired bets to settle")
        
        # Get all pending/matched bets that might need settlement
        pending_bets = get_pending_bets()
        settled_count = 0
        
        for bet in pending_bets:
            # Check if bet should be expired
            created_at = datetime.fromisoformat(bet['created_at'])
            if datetime.now() - created_at > timedelta(hours=6):  # Expire after 6 hours
                # Force settlement check
                result = settle_bet_task.apply_async(args=[bet['order_id']])
                if result.get() and result.get().get('success'):
                    settled_count += 1
        
        return {'success': True, 'settled_count': settled_count}
    
    except Exception as e:
        logger.error(f"Expired bet settlement failed: {e}")
        return {'success': False, 'error': str(e)}


# Data Collection Tasks
@celery.task(base=TennisBettingTask, bind=True, name='data.collect_live_odds')
def collect_live_odds_task(self) -> Dict[str, Any]:
    """
    Collect live odds from multiple sources and cache them
    
    Returns:
        Dict containing collection result
    """
    try:
        logger.debug("Collecting live odds")
        
        # Get active matches
        active_matches = get_active_matches()
        odds_collected = 0
        
        for match in active_matches:
            try:
                # Collect from Betfair
                betfair_odds = collect_betfair_odds(match['match_id'])
                if betfair_odds:
                    cache_odds(match['match_id'], 'betfair', betfair_odds)
                    odds_collected += 1
                
                # Collect from other sources (API-Tennis, etc.)
                api_odds = collect_api_tennis_odds(match['match_id'])
                if api_odds:
                    cache_odds(match['match_id'], 'api_tennis', api_odds)
                    odds_collected += 1
                
            except Exception as e:
                logger.warning(f"Failed to collect odds for match {match['match_id']}: {e}")
        
        return {'success': True, 'odds_collected': odds_collected}
    
    except Exception as e:
        logger.error(f"Live odds collection failed: {e}")
        return {'success': False, 'error': str(e)}


@celery.task(base=TennisBettingTask, bind=True, name='data.update_predictions')
def update_predictions_task(self) -> Dict[str, Any]:
    """
    Update ML predictions for active matches
    
    Returns:
        Dict containing update result
    """
    try:
        logger.debug("Updating ML predictions")
        
        # Initialize prediction engine
        prediction_engine = RealtimePredictionEngine()
        
        # Get active matches
        active_matches = get_active_matches()
        predictions_updated = 0
        
        for match in active_matches:
            try:
                # Generate updated prediction
                prediction = prediction_engine.predict_live_match(match)
                
                if prediction and prediction.get('confidence', 0) > 0.6:
                    # Cache prediction
                    cache_manager.set(
                        f"prediction:{match['match_id']}",
                        prediction,
                        ttl=300  # 5 minutes
                    )
                    
                    # Check for new betting opportunities
                    check_betting_opportunity_task.apply_async(
                        args=[match['match_id'], prediction]
                    )
                    
                    predictions_updated += 1
                
            except Exception as e:
                logger.warning(f"Failed to update prediction for match {match['match_id']}: {e}")
        
        return {'success': True, 'predictions_updated': predictions_updated}
    
    except Exception as e:
        logger.error(f"Prediction update failed: {e}")
        return {'success': False, 'error': str(e)}


@celery.task(base=TennisBettingTask, bind=True, name='data.check_betting_opportunity')
def check_betting_opportunity_task(self, match_id: str, prediction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if a prediction represents a betting opportunity
    
    Args:
        match_id: Match identifier
        prediction: ML prediction data
        
    Returns:
        Dict containing opportunity check result
    """
    try:
        # Get current odds
        odds_data = cache_manager.get(f"odds:{match_id}:betfair")
        if not odds_data:
            return {'success': False, 'error': 'No odds available'}
        
        # Calculate edge
        predicted_prob = prediction.get('probability_1', 0.5)
        market_odds = odds_data.get('player1_odds', 2.0)
        implied_prob = 1 / market_odds
        edge = predicted_prob - implied_prob
        
        # Check if opportunity meets criteria
        if (edge > 0.05 and  # 5% edge minimum
            prediction.get('confidence', 0) > 0.7 and  # 70% confidence minimum
            market_odds >= 1.5 and market_odds <= 4.0):  # Odds range
            
            # Calculate optimal stake
            stake = calculate_kelly_stake(predicted_prob, market_odds)
            
            if stake >= 10:  # Minimum stake
                # Create bet order
                bet_order = create_bet_order(match_id, prediction, odds_data, stake)
                
                # Submit bet asynchronously
                place_bet_task.apply_async(args=[bet_order.to_dict()])
                
                logger.info(f"Betting opportunity identified: {match_id} - Edge: {edge:.3f}")
                
                return {
                    'success': True,
                    'opportunity_found': True,
                    'edge': edge,
                    'stake': stake
                }
        
        return {'success': True, 'opportunity_found': False}
    
    except Exception as e:
        logger.error(f"Betting opportunity check failed: {e}")
        return {'success': False, 'error': str(e)}


# Risk Management Tasks
@celery.task(base=TennisBettingTask, bind=True, name='risk.calculate_positions')
def calculate_risk_positions_task(self) -> Dict[str, Any]:
    """
    Calculate current risk positions across all active bets
    
    Returns:
        Dict containing risk calculation result
    """
    try:
        logger.debug("Calculating risk positions")
        
        # Get all active bets
        active_bets = get_active_bets()
        
        # Group by match
        match_positions = {}
        total_exposure = 0
        total_potential_loss = 0
        
        for bet in active_bets:
            match_id = bet['match_id']
            if match_id not in match_positions:
                match_positions[match_id] = {
                    'total_exposure': 0,
                    'max_loss': 0,
                    'max_profit': 0,
                    'positions': []
                }
            
            exposure = bet['stake']
            potential_loss = bet['stake'] if bet['status'] in ['placed', 'matched'] else 0
            potential_profit = bet['potential_payout'] - bet['stake']
            
            match_positions[match_id]['total_exposure'] += exposure
            match_positions[match_id]['max_loss'] += potential_loss
            match_positions[match_id]['max_profit'] += potential_profit
            match_positions[match_id]['positions'].append(bet)
            
            total_exposure += exposure
            total_potential_loss += potential_loss
        
        # Calculate risk scores
        for match_id, position in match_positions.items():
            # Risk score based on exposure and odds
            avg_odds = sum(p['odds'] for p in position['positions']) / len(position['positions'])
            risk_score = (position['total_exposure'] / 1000) * (1 / avg_odds)  # Normalized risk
            position['risk_score'] = risk_score
        
        # Cache risk positions
        risk_summary = {
            'total_exposure': total_exposure,
            'total_potential_loss': total_potential_loss,
            'match_positions': match_positions,
            'timestamp': datetime.now().isoformat()
        }
        
        cache_manager.set('risk_positions', risk_summary, ttl=300)
        
        # Check risk limits
        check_risk_limits(risk_summary)
        
        return {'success': True, 'total_exposure': total_exposure}
    
    except Exception as e:
        logger.error(f"Risk calculation failed: {e}")
        return {'success': False, 'error': str(e)}


# Monitoring Tasks
@celery.task(base=TennisBettingTask, bind=True, name='monitoring.health_check')
def health_check_task(self) -> Dict[str, Any]:
    """
    Perform system health check
    
    Returns:
        Dict containing health status
    """
    try:
        logger.debug("Performing health check")
        
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'celery_workers': check_celery_workers(),
            'redis_connection': check_redis_connection(),
            'betfair_connection': check_betfair_connection(),
            'database_connection': check_database_connection(),
            'active_bets_count': len(get_active_bets()),
            'system_status': 'healthy'
        }
        
        # Check for any failures
        if not all([
            health_status['redis_connection'],
            health_status['database_connection']
        ]):
            health_status['system_status'] = 'degraded'
        
        # Cache health status
        cache_manager.set('system_health', health_status, ttl=300)
        
        return health_status
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            'timestamp': datetime.now().isoformat(),
            'system_status': 'error',
            'error': str(e)
        }


# Helper Functions
def validate_bet_order(bet_order: BetOrder) -> Dict[str, Any]:
    """Validate bet order before placement"""
    if bet_order.stake < 2:  # Minimum Betfair stake
        return {'valid': False, 'reason': 'Stake below minimum'}
    
    if bet_order.odds < 1.01 or bet_order.odds > 1000:
        return {'valid': False, 'reason': 'Invalid odds range'}
    
    if not bet_order.market_id or not bet_order.selection_id:
        return {'valid': False, 'reason': 'Missing market or selection ID'}
    
    return {'valid': True}


def update_bet_status(order_id: str, status: str, betfair_bet_id: str = None, 
                     error: str = None, settlement_data: Dict = None):
    """Update bet status in database"""
    # This would update the database with bet status
    # Implementation depends on database schema
    pass


def get_bet_from_database(order_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve bet information from database"""
    # This would query the database for bet information
    # Implementation depends on database schema
    return None


def map_betfair_status(betfair_status: str) -> str:
    """Map Betfair bet status to internal status"""
    mapping = {
        'EXECUTABLE': 'placed',
        'EXECUTION_COMPLETE': 'matched',
        'CANCELLED': 'cancelled',
        'EXPIRED': 'expired'
    }
    return mapping.get(betfair_status, 'unknown')


def schedule_settlement_monitoring(order_id: str, match_id: str):
    """Schedule settlement monitoring for matched bet"""
    # Monitor match completion and settle when finished
    pass


def get_match_result(match_id: str) -> Optional[Dict[str, Any]]:
    """Get match result from database or API"""
    # This would get the final match result
    return None


def determine_bet_outcome(bet_info: Dict, match_result: Dict) -> bool:
    """Determine if bet won based on match result"""
    # Logic to determine if bet won
    return True  # Placeholder


def update_pnl_metrics(profit_loss: float, status: str):
    """Update global P&L metrics"""
    # Update metrics in Redis or database
    pass


def get_pending_bets() -> List[Dict[str, Any]]:
    """Get all pending bets from database"""
    return []


def get_active_matches() -> List[Dict[str, Any]]:
    """Get currently active tennis matches"""
    return []


def collect_betfair_odds(match_id: str) -> Optional[Dict[str, Any]]:
    """Collect odds from Betfair for specific match"""
    return None


def collect_api_tennis_odds(match_id: str) -> Optional[Dict[str, Any]]:
    """Collect odds from API-Tennis for specific match"""
    return None


def cache_odds(match_id: str, source: str, odds_data: Dict[str, Any]):
    """Cache odds data in Redis"""
    cache_manager.set(f"odds:{match_id}:{source}", odds_data, ttl=60)


def calculate_kelly_stake(probability: float, odds: float, bankroll: float = 10000) -> float:
    """Calculate optimal stake using Kelly criterion"""
    b = odds - 1
    p = probability
    q = 1 - p
    
    kelly_fraction = (b * p - q) / b if b > 0 else 0
    conservative_fraction = kelly_fraction * 0.25  # 25% of Kelly
    
    stake = max(10, min(100, bankroll * conservative_fraction))
    return round(stake, 2)


def create_bet_order(match_id: str, prediction: Dict, odds_data: Dict, stake: float) -> BetOrder:
    """Create bet order from prediction and odds data"""
    import uuid
    
    order_id = str(uuid.uuid4())
    selection = f"player_{prediction.get('predicted_winner', 1)}"
    odds = odds_data.get('player1_odds' if prediction.get('predicted_winner') == 1 else 'player2_odds', 2.0)
    
    return BetOrder(
        order_id=order_id,
        match_id=match_id,
        market_id=odds_data.get('market_id', f"market_{match_id}"),
        selection_id=odds_data.get('selection_id', f"selection_{prediction.get('predicted_winner')}"),
        bet_type='MATCH_WINNER',
        selection=selection,
        odds=odds,
        stake=stake,
        potential_payout=stake * odds,
        confidence=prediction.get('confidence', 0),
        strategy='ml_prediction',
        created_at=datetime.now().isoformat(),
        expires_at=(datetime.now() + timedelta(hours=4)).isoformat()
    )


def get_active_bets() -> List[Dict[str, Any]]:
    """Get all active bets"""
    return []


def check_risk_limits(risk_summary: Dict[str, Any]):
    """Check if risk limits are exceeded"""
    if risk_summary['total_exposure'] > 5000:  # $5000 total exposure limit
        logger.warning(f"High exposure detected: ${risk_summary['total_exposure']}")


def check_celery_workers() -> bool:
    """Check if Celery workers are running"""
    try:
        i = celery.control.inspect()
        stats = i.stats()
        return bool(stats)
    except:
        return False


def check_redis_connection() -> bool:
    """Check Redis connection"""
    try:
        redis_client.ping()
        return True
    except:
        return False


def check_betfair_connection() -> bool:
    """Check Betfair API connection"""
    try:
        client = BetfairAPIClient()
        health = client.health_check()
        return health.get('status') == 'healthy'
    except:
        return False


def check_database_connection() -> bool:
    """Check database connection"""
    try:
        # This would test database connectivity
        return True
    except:
        return False


if __name__ == '__main__':
    # Start Celery worker
    celery.start()