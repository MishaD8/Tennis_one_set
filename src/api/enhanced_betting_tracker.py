#!/usr/bin/env python3
"""
Enhanced Betting Tracker Service
Comprehensive betting tracking with enhanced database integration
"""

import os
import logging
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from data.enhanced_database_service import get_database_service, EnhancedDatabaseService
from utils.enhanced_rate_limiting import get_rate_limit_manager, betting_rate_limit

logger = logging.getLogger(__name__)

class BettingDecision(Enum):
    """Betting decision types"""
    BET = "BET"
    STRONG_BET = "STRONG_BET"
    AVOID = "AVOID"
    NO_VALUE = "NO_VALUE"
    SKIP = "SKIP"

class BettingOutcome(Enum):
    """Betting outcome types"""
    WIN = "WIN"
    LOSS = "LOSS"
    PUSH = "PUSH"
    VOID = "VOID"
    PENDING = "PENDING"

@dataclass
class MatchInfo:
    """Match information structure"""
    match_id: str
    player1: str
    player2: str
    tournament: str
    surface: str
    round_name: str
    match_date: datetime
    
@dataclass
class PredictionInfo:
    """Prediction information structure"""
    our_probability: float
    confidence_level: str
    model_used: str
    key_factors: str
    prediction_type: str = "match_winner"
    
@dataclass
class BettingInfo:
    """Betting information structure"""
    bookmaker: str
    odds: float
    implied_probability: float
    edge_percentage: float
    stake_amount: float
    decision: BettingDecision
    risk_level: str = "medium"
    
@dataclass
class BettingRecord:
    """Complete betting record"""
    record_id: str
    timestamp: datetime
    match_info: MatchInfo
    prediction_info: PredictionInfo
    betting_info: BettingInfo
    outcome: Optional[BettingOutcome] = None
    actual_return: Optional[float] = None
    profit_loss: Optional[float] = None
    notes: Optional[str] = None

class EnhancedBettingTracker:
    """Enhanced betting tracker with comprehensive logging and analytics"""
    
    def __init__(self):
        self.db_service = get_database_service()
        self.rate_limiter = get_rate_limit_manager()
        self.active_bets: Dict[str, BettingRecord] = {}
        
        logger.info("✅ Enhanced Betting Tracker initialized")
    
    @betting_rate_limit
    def log_betting_decision(self, 
                           match_info: Dict,
                           prediction_info: Dict,
                           betting_info: Dict) -> str:
        """Log a new betting decision"""
        
        try:
            # Generate unique record ID
            record_id = str(uuid.uuid4())
            
            # Create structured objects
            match_obj = MatchInfo(
                match_id=match_info.get('match_id', str(uuid.uuid4())),
                player1=match_info['player1'],
                player2=match_info['player2'],
                tournament=match_info['tournament'],
                surface=match_info.get('surface', 'Unknown'),
                round_name=match_info.get('round_name', 'Unknown'),
                match_date=match_info.get('match_date', datetime.now())
            )
            
            prediction_obj = PredictionInfo(
                our_probability=prediction_info['our_probability'],
                confidence_level=prediction_info.get('confidence_level', 'Medium'),
                model_used=prediction_info.get('model_used', 'default'),
                key_factors=prediction_info.get('key_factors', ''),
                prediction_type=prediction_info.get('prediction_type', 'match_winner')
            )
            
            betting_obj = BettingInfo(
                bookmaker=betting_info.get('bookmaker', 'Unknown'),
                odds=betting_info['odds'],
                implied_probability=betting_info['implied_probability'],
                edge_percentage=betting_info['edge_percentage'],
                stake_amount=betting_info['stake_amount'],
                decision=BettingDecision(betting_info.get('decision', 'BET')),
                risk_level=betting_info.get('risk_level', 'medium')
            )
            
            # Create betting record
            betting_record = BettingRecord(
                record_id=record_id,
                timestamp=datetime.now(),
                match_info=match_obj,
                prediction_info=prediction_obj,
                betting_info=betting_obj
            )
            
            # Store in active bets
            self.active_bets[record_id] = betting_record
            
            # Log to database
            prediction_data = {
                'match_date': match_obj.match_date,
                'player1': match_obj.player1,
                'player2': match_obj.player2,
                'tournament': match_obj.tournament,
                'surface': match_obj.surface,
                'round_name': match_obj.round_name,
                'our_probability': prediction_obj.our_probability,
                'confidence': prediction_obj.confidence_level,
                'ml_system': prediction_obj.model_used,
                'prediction_type': prediction_obj.prediction_type,
                'key_factors': prediction_obj.key_factors,
                'bookmaker_odds': betting_obj.odds,
                'bookmaker_probability': betting_obj.implied_probability,
                'edge': betting_obj.edge_percentage,
                'recommendation': betting_obj.decision.value
            }
            
            prediction_id = self.db_service.log_prediction(prediction_data)
            
            betting_data = {
                'player1': match_obj.player1,
                'player2': match_obj.player2,
                'tournament': match_obj.tournament,
                'match_date': match_obj.match_date,
                'our_probability': prediction_obj.our_probability,
                'bookmaker_odds': betting_obj.odds,
                'implied_probability': betting_obj.implied_probability,
                'edge_percentage': betting_obj.edge_percentage,
                'confidence_level': prediction_obj.confidence_level,
                'bet_recommendation': betting_obj.decision.value,
                'suggested_stake': betting_obj.stake_amount
            }
            
            betting_id = self.db_service.log_betting_record(betting_data, prediction_id)
            
            logger.info(f"✅ Betting decision logged: {match_obj.player1} vs {match_obj.player2}")
            logger.info(f"   - Decision: {betting_obj.decision.value}")
            logger.info(f"   - Stake: ${betting_obj.stake_amount}")
            logger.info(f"   - Edge: {betting_obj.edge_percentage:.1f}%")
            logger.info(f"   - Record ID: {record_id}")
            
            return record_id
            
        except Exception as e:
            logger.error(f"❌ Error logging betting decision: {e}")
            raise
    
    def update_betting_outcome(self, 
                             record_id: str,
                             outcome: BettingOutcome,
                             actual_return: Optional[float] = None,
                             notes: Optional[str] = None) -> bool:
        """Update betting outcome for a recorded bet"""
        
        try:
            if record_id not in self.active_bets:
                logger.warning(f"⚠️ Betting record {record_id} not found in active bets")
                return False
            
            betting_record = self.active_bets[record_id]
            betting_record.outcome = outcome
            betting_record.actual_return = actual_return or 0.0
            betting_record.notes = notes
            
            # Calculate profit/loss
            if outcome == BettingOutcome.WIN and actual_return:
                betting_record.profit_loss = actual_return - betting_record.betting_info.stake_amount
            elif outcome == BettingOutcome.LOSS:
                betting_record.profit_loss = -betting_record.betting_info.stake_amount
            elif outcome in [BettingOutcome.PUSH, BettingOutcome.VOID]:
                betting_record.profit_loss = 0.0
            else:
                betting_record.profit_loss = None
            
            # Update database records would go here
            # For now, we'll keep the record in memory
            
            logger.info(f"✅ Betting outcome updated for {record_id}")
            logger.info(f"   - Outcome: {outcome.value}")
            logger.info(f"   - Return: ${actual_return or 0:.2f}")
            logger.info(f"   - P&L: ${betting_record.profit_loss or 0:.2f}")
            
            # Move to completed bets if final outcome
            if outcome in [BettingOutcome.WIN, BettingOutcome.LOSS, BettingOutcome.VOID]:
                # Could move to a completed_bets storage here
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating betting outcome: {e}")
            return False
    
    def get_active_bets(self) -> List[Dict[str, Any]]:
        """Get all active betting records"""
        return [self._betting_record_to_dict(record) for record in self.active_bets.values()]
    
    def get_betting_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get betting summary and statistics"""
        
        # Get recent betting records from database
        recent_betting = self.db_service.get_recent_betting_records(days=days)
        recent_predictions = self.db_service.get_recent_predictions(days=days)
        
        # Calculate statistics
        total_bets = len(recent_betting)
        total_stake = sum(bet.get('suggested_stake', 0) for bet in recent_betting)
        
        # Count by recommendation
        bet_recommendations = {}
        for bet in recent_betting:
            rec = bet.get('bet_recommendation', 'UNKNOWN')
            bet_recommendations[rec] = bet_recommendations.get(rec, 0) + 1
        
        # Calculate accuracy from predictions
        accurate_predictions = 0
        total_predictions_with_results = 0
        
        for pred in recent_predictions:
            if pred.get('prediction_correct') is not None:
                total_predictions_with_results += 1
                if pred.get('prediction_correct'):
                    accurate_predictions += 1
        
        accuracy = 0.0
        if total_predictions_with_results > 0:
            accuracy = (accurate_predictions / total_predictions_with_results) * 100
        
        # Active bets summary
        active_bets_count = len(self.active_bets)
        active_stake = sum(bet.betting_info.stake_amount for bet in self.active_bets.values())
        
        return {
            'period_days': days,
            'timestamp': datetime.now().isoformat(),
            'database_summary': {
                'total_betting_records': total_bets,
                'total_predictions': len(recent_predictions),
                'total_stake': total_stake,
                'bet_recommendations': bet_recommendations,
                'accuracy_percentage': round(accuracy, 2)
            },
            'active_bets': {
                'count': active_bets_count,
                'total_stake': active_stake,
                'pending_value': active_stake
            },
            'system_status': {
                'database_type': self.db_service.database_type,
                'rate_limiter_status': self.rate_limiter.get_system_status()['rate_limiter']
            }
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        
        # Get completed bets with outcomes
        completed_bets = [bet for bet in self.active_bets.values() 
                         if bet.outcome in [BettingOutcome.WIN, BettingOutcome.LOSS, BettingOutcome.VOID]]
        
        if not completed_bets:
            return {
                'total_completed_bets': 0,
                'message': 'No completed bets available for performance calculation'
            }
        
        # Calculate metrics
        total_completed = len(completed_bets)
        wins = len([bet for bet in completed_bets if bet.outcome == BettingOutcome.WIN])
        losses = len([bet for bet in completed_bets if bet.outcome == BettingOutcome.LOSS])
        voids = len([bet for bet in completed_bets if bet.outcome == BettingOutcome.VOID])
        
        win_rate = (wins / total_completed) * 100 if total_completed > 0 else 0
        
        # P&L calculations
        total_staked = sum(bet.betting_info.stake_amount for bet in completed_bets)
        total_return = sum(bet.actual_return or 0 for bet in completed_bets)
        net_profit = total_return - total_staked
        roi_percentage = (net_profit / total_staked) * 100 if total_staked > 0 else 0
        
        # Average odds and stakes
        avg_odds = sum(bet.betting_info.odds for bet in completed_bets) / total_completed
        avg_stake = total_staked / total_completed
        avg_edge = sum(bet.betting_info.edge_percentage for bet in completed_bets) / total_completed
        
        return {
            'period': 'all_time',
            'timestamp': datetime.now().isoformat(),
            'betting_performance': {
                'total_completed_bets': total_completed,
                'wins': wins,
                'losses': losses,
                'voids': voids,
                'win_rate_percentage': round(win_rate, 2)
            },
            'financial_performance': {
                'total_staked': round(total_staked, 2),
                'total_return': round(total_return, 2),
                'net_profit': round(net_profit, 2),
                'roi_percentage': round(roi_percentage, 2)
            },
            'averages': {
                'avg_odds': round(avg_odds, 2),
                'avg_stake': round(avg_stake, 2),
                'avg_edge_percentage': round(avg_edge, 2)
            }
        }
    
    def export_betting_data(self, format: str = 'json', days: int = 365) -> str:
        """Export betting data to file"""
        
        try:
            # Get data from database
            recent_betting = self.db_service.get_recent_betting_records(days=days)
            recent_predictions = self.db_service.get_recent_predictions(days=days)
            active_bets_data = self.get_active_bets()
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'period_days': days,
                'database_records': {
                    'betting_records': recent_betting,
                    'predictions': recent_predictions
                },
                'active_bets': active_bets_data,
                'summary': self.get_betting_summary(days=days),
                'performance': self.get_performance_metrics()
            }
            
            # Create filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'betting_export_{timestamp}.{format}'
            
            if format.lower() == 'json':
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            else:
                # Could add CSV export here
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"✅ Betting data exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"❌ Error exporting betting data: {e}")
            raise
    
    def _betting_record_to_dict(self, record: BettingRecord) -> Dict[str, Any]:
        """Convert betting record to dictionary"""
        return {
            'record_id': record.record_id,
            'timestamp': record.timestamp.isoformat(),
            'match_info': asdict(record.match_info),
            'prediction_info': asdict(record.prediction_info),
            'betting_info': {
                'bookmaker': record.betting_info.bookmaker,
                'odds': record.betting_info.odds,
                'implied_probability': record.betting_info.implied_probability,
                'edge_percentage': record.betting_info.edge_percentage,
                'stake_amount': record.betting_info.stake_amount,
                'decision': record.betting_info.decision.value,
                'risk_level': record.betting_info.risk_level
            },
            'outcome': record.outcome.value if record.outcome else None,
            'actual_return': record.actual_return,
            'profit_loss': record.profit_loss,
            'notes': record.notes
        }
    
    def cleanup_old_records(self, days: int = 90) -> int:
        """Clean up old active betting records"""
        cutoff_date = datetime.now() - timedelta(days=days)
        cleaned_count = 0
        
        for record_id, record in list(self.active_bets.items()):
            if record.timestamp < cutoff_date:
                del self.active_bets[record_id]
                cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"✅ Cleaned up {cleaned_count} old betting records")
        
        return cleaned_count

# Global instance
betting_tracker = None

def get_betting_tracker() -> EnhancedBettingTracker:
    """Get singleton betting tracker instance"""
    global betting_tracker
    if betting_tracker is None:
        betting_tracker = EnhancedBettingTracker()
    return betting_tracker