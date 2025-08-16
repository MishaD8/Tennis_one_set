#!/usr/bin/env python3
"""
💰 Betting Tracker Service for Tennis Betting System
Comprehensive logging and tracking of actual betting decisions, stakes, and outcomes
"""

import os
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import sessionmaker
from sqlalchemy import and_, or_, desc, asc, func

from src.data.database_models import (
    DatabaseManager, BettingLog, BettingStatus, TestMode, 
    TestingSession, Prediction, Base
)

logger = logging.getLogger(__name__)

class BettingTrackerService:
    """
    Comprehensive betting tracker for logging actual betting decisions and outcomes
    Integrates with backtesting and forward testing systems
    """
    
    def __init__(self, database_manager: DatabaseManager = None):
        self.db_manager = database_manager or DatabaseManager()
        self.session = self.db_manager.get_session()
        
        # Ensure tables exist
        try:
            Base.metadata.create_all(bind=self.db_manager.engine)
            logger.info("✅ Betting tracker tables created/verified")
        except Exception as e:
            logger.error(f"❌ Error creating betting tracker tables: {e}")
    
    def log_betting_decision(self, 
                           match_data: Dict,
                           prediction_data: Dict,
                           betting_data: Dict,
                           test_mode: TestMode = TestMode.LIVE) -> str:
        """
        Log a betting decision with comprehensive details
        
        Args:
            match_data: Match information (players, tournament, etc.)
            prediction_data: ML prediction details
            betting_data: Betting specifics (odds, stakes, etc.)
            test_mode: Testing mode (LIVE, BACKTEST, FORWARD_TEST)
        
        Returns:
            bet_id: Unique identifier for the betting log
        """
        try:
            bet_id = f"bet_{uuid.uuid4().hex[:12]}"
            
            # Extract match data
            match_id = match_data.get('match_id', f"match_{uuid.uuid4().hex[:8]}")
            player1 = match_data.get('player1', '')
            player2 = match_data.get('player2', '')
            tournament = match_data.get('tournament', '')
            match_date_str = match_data.get('match_date', datetime.now().strftime('%Y-%m-%d'))
            match_date = datetime.fromisoformat(match_date_str) if isinstance(match_date_str, str) else match_date_str
            
            # Extract prediction data
            predicted_winner = prediction_data.get('predicted_winner', player1)
            our_probability = prediction_data.get('probability', 0.5)
            confidence_level = prediction_data.get('confidence', 'Medium')
            model_used = prediction_data.get('model_used', 'Unknown')
            key_factors = json.dumps(prediction_data.get('key_factors', []))
            
            # Extract betting data
            bookmaker = betting_data.get('bookmaker', 'Unknown')
            odds_taken = betting_data.get('odds_taken', 2.0)
            stake_amount = betting_data.get('stake_amount', 100.0)
            stake_percentage = betting_data.get('stake_percentage')
            kelly_fraction = betting_data.get('kelly_fraction')
            risk_level = betting_data.get('risk_level', 'Medium')
            
            # Calculate derived values
            implied_probability = 1.0 / odds_taken if odds_taken > 0 else 0.5
            edge_percentage = (our_probability - implied_probability) * 100
            
            # Create betting log
            betting_log = BettingLog(
                bet_id=bet_id,
                match_id=match_id,
                player1=player1,
                player2=player2,
                tournament=tournament,
                match_date=match_date,
                test_mode=test_mode,
                predicted_winner=predicted_winner,
                our_probability=our_probability,
                confidence_level=confidence_level,
                model_used=model_used,
                key_factors=key_factors,
                bookmaker=bookmaker,
                odds_taken=odds_taken,
                implied_probability=implied_probability,
                edge_percentage=edge_percentage,
                stake_amount=stake_amount,
                stake_percentage=stake_percentage,
                kelly_fraction=kelly_fraction,
                risk_level=risk_level,
                betting_status=BettingStatus.PENDING
            )
            
            self.session.add(betting_log)
            self.session.commit()
            
            logger.info(f"💰 Logged betting decision: {bet_id}")
            logger.info(f"   Match: {player1} vs {player2}")
            logger.info(f"   Prediction: {predicted_winner} @ {our_probability:.1%}")
            logger.info(f"   Bet: ${stake_amount} @ {odds_taken} (Edge: {edge_percentage:+.1f}%)")
            
            return bet_id
            
        except Exception as e:
            logger.error(f"❌ Error logging betting decision: {e}")
            self.session.rollback()
            return ""
    
    def update_betting_outcome(self, 
                              bet_id: str,
                              match_result: Dict,
                              settlement_notes: str = "") -> bool:
        """
        Update betting outcome when match is completed
        
        Args:
            bet_id: Betting log identifier
            match_result: Match outcome details
            settlement_notes: Additional notes about settlement
        
        Returns:
            bool: Success status
        """
        try:
            # Find betting log
            betting_log = self.session.query(BettingLog).filter(
                BettingLog.bet_id == bet_id
            ).first()
            
            if not betting_log:
                logger.error(f"❌ Betting log not found: {bet_id}")
                return False
            
            # Extract match result
            actual_winner = match_result.get('winner', '')
            match_score = match_result.get('score', '')
            
            # Update betting log
            betting_log.actual_winner = actual_winner
            betting_log.match_score = match_score
            betting_log.settled_at = datetime.utcnow()
            betting_log.settlement_notes = settlement_notes
            
            # Determine outcome
            if actual_winner == betting_log.predicted_winner:
                # Win
                betting_log.betting_status = BettingStatus.WON
                betting_log.payout_amount = betting_log.stake_amount * betting_log.odds_taken
                betting_log.profit_loss = betting_log.payout_amount - betting_log.stake_amount
                betting_log.prediction_correct = True
                betting_log.value_bet_outcome = 'profitable'
            else:
                # Loss
                betting_log.betting_status = BettingStatus.LOST
                betting_log.payout_amount = 0.0
                betting_log.profit_loss = -betting_log.stake_amount
                betting_log.prediction_correct = False
                betting_log.value_bet_outcome = 'unprofitable'
            
            # Calculate ROI
            betting_log.roi_percentage = (betting_log.profit_loss / betting_log.stake_amount) * 100
            
            self.session.commit()
            
            logger.info(f"💰 Updated betting outcome: {bet_id}")
            logger.info(f"   Result: {actual_winner} won")
            logger.info(f"   Outcome: {'WIN' if betting_log.prediction_correct else 'LOSS'}")
            logger.info(f"   P&L: ${betting_log.profit_loss:.2f} (ROI: {betting_log.roi_percentage:.1f}%)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating betting outcome: {e}")
            self.session.rollback()
            return False
    
    def get_betting_logs(self, 
                        test_mode: TestMode = None,
                        status: BettingStatus = None,
                        start_date: datetime = None,
                        end_date: datetime = None,
                        limit: int = 100) -> List[Dict]:
        """
        Retrieve betting logs with filtering options
        """
        try:
            query = self.session.query(BettingLog)
            
            # Apply filters
            if test_mode:
                query = query.filter(BettingLog.test_mode == test_mode)
            
            if status:
                query = query.filter(BettingLog.betting_status == status)
            
            if start_date:
                query = query.filter(BettingLog.timestamp >= start_date)
            
            if end_date:
                query = query.filter(BettingLog.timestamp <= end_date)
            
            # Order by timestamp and limit
            logs = query.order_by(BettingLog.timestamp.desc()).limit(limit).all()
            
            # Convert to dictionaries
            betting_logs = []
            for log in logs:
                log_dict = {
                    'bet_id': log.bet_id,
                    'timestamp': log.timestamp.isoformat(),
                    'match_id': log.match_id,
                    'player1': log.player1,
                    'player2': log.player2,
                    'tournament': log.tournament,
                    'match_date': log.match_date.isoformat(),
                    'test_mode': log.test_mode.value,
                    'predicted_winner': log.predicted_winner,
                    'our_probability': log.our_probability,
                    'confidence_level': log.confidence_level,
                    'model_used': log.model_used,
                    'key_factors': json.loads(log.key_factors) if log.key_factors else [],
                    'bookmaker': log.bookmaker,
                    'odds_taken': log.odds_taken,
                    'edge_percentage': log.edge_percentage,
                    'stake_amount': log.stake_amount,
                    'stake_percentage': log.stake_percentage,
                    'kelly_fraction': log.kelly_fraction,
                    'risk_level': log.risk_level,
                    'betting_status': log.betting_status.value,
                    'actual_winner': log.actual_winner,
                    'match_score': log.match_score,
                    'payout_amount': log.payout_amount,
                    'profit_loss': log.profit_loss,
                    'roi_percentage': log.roi_percentage,
                    'prediction_correct': log.prediction_correct,
                    'value_bet_outcome': log.value_bet_outcome,
                    'settled_at': log.settled_at.isoformat() if log.settled_at else None,
                    'settlement_notes': log.settlement_notes
                }
                betting_logs.append(log_dict)
            
            return betting_logs
            
        except Exception as e:
            logger.error(f"❌ Error retrieving betting logs: {e}")
            return []
    
    def get_betting_performance_summary(self, 
                                      test_mode: TestMode = None,
                                      days_back: int = 30) -> Dict:
        """
        Get comprehensive betting performance summary
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            query = self.session.query(BettingLog).filter(
                and_(
                    BettingLog.timestamp >= cutoff_date,
                    BettingLog.betting_status.in_([BettingStatus.WON, BettingStatus.LOST])
                )
            )
            
            if test_mode:
                query = query.filter(BettingLog.test_mode == test_mode)
            
            logs = query.all()
            
            if not logs:
                return {
                    'message': f'No betting data available for the last {days_back} days',
                    'test_mode': test_mode.value if test_mode else 'all'
                }
            
            # Basic metrics
            total_bets = len(logs)
            winning_bets = len([log for log in logs if log.betting_status == BettingStatus.WON])
            losing_bets = total_bets - winning_bets
            
            win_rate = (winning_bets / total_bets) * 100
            
            # Financial metrics
            total_staked = sum(log.stake_amount for log in logs)
            total_returned = sum(log.payout_amount or 0 for log in logs)
            net_profit = sum(log.profit_loss for log in logs)
            
            roi_percentage = (net_profit / total_staked) * 100 if total_staked > 0 else 0
            
            # Profit factor
            total_wins_amount = sum(log.profit_loss for log in logs if log.profit_loss > 0)
            total_losses_amount = abs(sum(log.profit_loss for log in logs if log.profit_loss < 0))
            profit_factor = total_wins_amount / total_losses_amount if total_losses_amount > 0 else 0
            
            # Average metrics
            average_stake = total_staked / total_bets
            average_odds = sum(log.odds_taken for log in logs) / total_bets
            average_edge = sum(log.edge_percentage for log in logs) / total_bets
            
            # Streak analysis
            current_streak = self._calculate_current_streak(logs)
            longest_winning_streak = self._calculate_longest_streak(logs, 'win')
            longest_losing_streak = self._calculate_longest_streak(logs, 'loss')
            
            # Model performance breakdown
            model_performance = {}
            for log in logs:
                model = log.model_used
                if model not in model_performance:
                    model_performance[model] = {
                        'total_bets': 0,
                        'winning_bets': 0,
                        'net_profit': 0,
                        'total_staked': 0
                    }
                
                model_performance[model]['total_bets'] += 1
                model_performance[model]['total_staked'] += log.stake_amount
                model_performance[model]['net_profit'] += log.profit_loss
                
                if log.betting_status == BettingStatus.WON:
                    model_performance[model]['winning_bets'] += 1
            
            # Calculate model ROI and win rates
            for model, stats in model_performance.items():
                stats['win_rate'] = (stats['winning_bets'] / stats['total_bets']) * 100
                stats['roi'] = (stats['net_profit'] / stats['total_staked']) * 100 if stats['total_staked'] > 0 else 0
            
            # Risk analysis
            largest_win = max((log.profit_loss for log in logs if log.profit_loss > 0), default=0)
            largest_loss = min((log.profit_loss for log in logs if log.profit_loss < 0), default=0)
            
            # Calculate maximum drawdown
            max_drawdown = self._calculate_max_drawdown(logs)
            
            summary = {
                'period': f'Last {days_back} days',
                'test_mode': test_mode.value if test_mode else 'all',
                'basic_metrics': {
                    'total_bets': total_bets,
                    'winning_bets': winning_bets,
                    'losing_bets': losing_bets,
                    'win_rate': win_rate
                },
                'financial_metrics': {
                    'total_staked': total_staked,
                    'total_returned': total_returned,
                    'net_profit': net_profit,
                    'roi_percentage': roi_percentage,
                    'profit_factor': profit_factor
                },
                'average_metrics': {
                    'average_stake': average_stake,
                    'average_odds': average_odds,
                    'average_edge': average_edge
                },
                'streak_analysis': {
                    'current_streak': current_streak,
                    'longest_winning_streak': longest_winning_streak,
                    'longest_losing_streak': longest_losing_streak
                },
                'risk_analysis': {
                    'largest_win': largest_win,
                    'largest_loss': abs(largest_loss),
                    'max_drawdown': max_drawdown
                },
                'model_performance': model_performance,
                'last_updated': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error getting betting performance summary: {e}")
            return {'error': str(e)}
    
    def _calculate_current_streak(self, logs: List[BettingLog]) -> Dict:
        """Calculate current winning/losing streak"""
        if not logs:
            return {'type': 'none', 'count': 0}
        
        # Sort by timestamp (most recent first)
        sorted_logs = sorted(logs, key=lambda x: x.timestamp, reverse=True)
        
        if not sorted_logs:
            return {'type': 'none', 'count': 0}
        
        current_status = sorted_logs[0].betting_status
        streak_count = 0
        
        for log in sorted_logs:
            if log.betting_status == current_status:
                streak_count += 1
            else:
                break
        
        streak_type = 'winning' if current_status == BettingStatus.WON else 'losing'
        
        return {
            'type': streak_type,
            'count': streak_count
        }
    
    def _calculate_longest_streak(self, logs: List[BettingLog], streak_type: str) -> int:
        """Calculate longest winning or losing streak"""
        if not logs:
            return 0
        
        target_status = BettingStatus.WON if streak_type == 'win' else BettingStatus.LOST
        
        # Sort by timestamp
        sorted_logs = sorted(logs, key=lambda x: x.timestamp)
        
        max_streak = 0
        current_streak = 0
        
        for log in sorted_logs:
            if log.betting_status == target_status:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _calculate_max_drawdown(self, logs: List[BettingLog]) -> float:
        """Calculate maximum drawdown from peak equity"""
        if not logs:
            return 0.0
        
        # Sort by timestamp
        sorted_logs = sorted(logs, key=lambda x: x.timestamp)
        
        running_profit = 0.0
        peak_profit = 0.0
        max_drawdown = 0.0
        
        for log in sorted_logs:
            running_profit += log.profit_loss
            peak_profit = max(peak_profit, running_profit)
            drawdown = peak_profit - running_profit
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def get_pending_bets(self, test_mode: TestMode = None) -> List[Dict]:
        """Get all pending bets awaiting settlement"""
        try:
            query = self.session.query(BettingLog).filter(
                BettingLog.betting_status == BettingStatus.PENDING
            )
            
            if test_mode:
                query = query.filter(BettingLog.test_mode == test_mode)
            
            logs = query.order_by(BettingLog.timestamp.desc()).all()
            
            pending_bets = []
            for log in logs:
                bet_dict = {
                    'bet_id': log.bet_id,
                    'match_id': log.match_id,
                    'player1': log.player1,
                    'player2': log.player2,
                    'tournament': log.tournament,
                    'match_date': log.match_date.isoformat(),
                    'predicted_winner': log.predicted_winner,
                    'odds_taken': log.odds_taken,
                    'stake_amount': log.stake_amount,
                    'potential_payout': log.stake_amount * log.odds_taken,
                    'edge_percentage': log.edge_percentage,
                    'test_mode': log.test_mode.value,
                    'timestamp': log.timestamp.isoformat()
                }
                pending_bets.append(bet_dict)
            
            return pending_bets
            
        except Exception as e:
            logger.error(f"❌ Error getting pending bets: {e}")
            return []
    
    def export_betting_logs(self, 
                           test_mode: TestMode = None,
                           start_date: datetime = None,
                           end_date: datetime = None,
                           format: str = 'csv') -> str:
        """
        Export betting logs to file
        """
        try:
            logs = self.get_betting_logs(
                test_mode=test_mode,
                start_date=start_date,
                end_date=end_date,
                limit=10000  # Large limit for export
            )
            
            if not logs:
                logger.warning("⚠️ No betting logs found for export")
                return ""
            
            # Create export filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            mode_suffix = f"_{test_mode.value}" if test_mode else "_all"
            export_dir = os.path.join(os.getcwd(), 'data', 'exports')
            os.makedirs(export_dir, exist_ok=True)
            
            if format.lower() == 'csv':
                import pandas as pd
                export_file = os.path.join(export_dir, f'betting_logs{mode_suffix}_{timestamp}.csv')
                
                # Convert to DataFrame
                df = pd.DataFrame(logs)
                df.to_csv(export_file, index=False)
                
            elif format.lower() == 'json':
                export_file = os.path.join(export_dir, f'betting_logs{mode_suffix}_{timestamp}.json')
                
                with open(export_file, 'w') as f:
                    json.dump(logs, f, indent=2, default=str)
            
            logger.info(f"📁 Exported {len(logs)} betting logs to {export_file}")
            return export_file
            
        except Exception as e:
            logger.error(f"❌ Error exporting betting logs: {e}")
            return ""
    
    def __del__(self):
        """Cleanup database session"""
        if hasattr(self, 'session'):
            self.session.close()


# Integration with existing tennis betting system
class TennisBettingIntegration:
    """
    Integration layer for tennis betting system
    Connects betting tracker with ML predictions and match data
    """
    
    def __init__(self):
        self.betting_tracker = BettingTrackerService()
    
    def log_underdog_bet(self, 
                        match_data: Dict,
                        underdog_analysis: Dict,
                        stake_amount: float = 100.0,
                        test_mode: TestMode = TestMode.LIVE) -> str:
        """
        Log a bet based on underdog analysis
        """
        try:
            # Extract underdog scenario
            scenario = underdog_analysis.get('underdog_scenario', {})
            underdog_player = scenario.get('underdog', match_data.get('player1', ''))
            
            # Determine odds (use higher odds for underdog)
            odds = match_data.get('odds', {})
            if underdog_player == match_data.get('player1', ''):
                betting_odds = odds.get('player1', 3.0)
            else:
                betting_odds = odds.get('player2', 3.0)
            
            # Calculate Kelly fraction (conservative)
            our_prob = underdog_analysis.get('underdog_probability', 0.3)
            kelly_fraction = max(0, (our_prob * betting_odds - 1) / (betting_odds - 1)) * 0.25  # 25% Kelly
            kelly_fraction = min(kelly_fraction, 0.05)  # Max 5% of bankroll
            
            # Prepare data for betting tracker
            prediction_data = {
                'predicted_winner': underdog_player,
                'probability': our_prob,
                'confidence': underdog_analysis.get('confidence', 'Medium'),
                'model_used': 'underdog_analyzer',
                'key_factors': underdog_analysis.get('key_factors', [])
            }
            
            betting_data = {
                'bookmaker': 'simulation',
                'odds_taken': betting_odds,
                'stake_amount': stake_amount,
                'kelly_fraction': kelly_fraction,
                'risk_level': 'Medium'
            }
            
            bet_id = self.betting_tracker.log_betting_decision(
                match_data=match_data,
                prediction_data=prediction_data,
                betting_data=betting_data,
                test_mode=test_mode
            )
            
            return bet_id
            
        except Exception as e:
            logger.error(f"❌ Error logging underdog bet: {e}")
            return ""
    
    def settle_match_bets(self, match_id: str, winner: str, score: str = "") -> int:
        """
        Settle all bets for a completed match
        """
        try:
            # Find all pending bets for this match
            pending_logs = self.betting_tracker.session.query(BettingLog).filter(
                and_(
                    BettingLog.match_id == match_id,
                    BettingLog.betting_status == BettingStatus.PENDING
                )
            ).all()
            
            settled_count = 0
            
            for log in pending_logs:
                match_result = {
                    'winner': winner,
                    'score': score
                }
                
                success = self.betting_tracker.update_betting_outcome(
                    bet_id=log.bet_id,
                    match_result=match_result
                )
                
                if success:
                    settled_count += 1
            
            logger.info(f"✅ Settled {settled_count} bets for match {match_id}")
            return settled_count
            
        except Exception as e:
            logger.error(f"❌ Error settling match bets: {e}")
            return 0


# Example usage and testing
if __name__ == "__main__":
    print("💰 BETTING TRACKER SERVICE - TESTING")
    print("=" * 50)
    
    # Initialize tracker
    tracker = BettingTrackerService()
    integration = TennisBettingIntegration()
    
    # Test data
    test_match = {
        'match_id': 'test_match_001',
        'player1': 'Flavio Cobolli',
        'player2': 'Novak Djokovic',
        'tournament': 'US Open',
        'match_date': '2025-08-20',
        'odds': {'player1': 4.5, 'player2': 1.22}
    }
    
    test_underdog_analysis = {
        'underdog_probability': 0.25,
        'confidence': 'Medium',
        'prediction_type': 'UNDERDOG_ANALYSIS',
        'key_factors': ['Grass advantage', 'Underdog potential'],
        'underdog_scenario': {
            'underdog': 'Flavio Cobolli',
            'favorite': 'Novak Djokovic'
        }
    }
    
    # Test logging betting decision
    print("1️⃣ Testing betting decision logging...")
    bet_id = integration.log_underdog_bet(
        match_data=test_match,
        underdog_analysis=test_underdog_analysis,
        stake_amount=100.0,
        test_mode=TestMode.FORWARD_TEST
    )
    print(f"💰 Logged bet: {bet_id}")
    
    # Test getting pending bets
    print("\n2️⃣ Testing pending bets retrieval...")
    pending = tracker.get_pending_bets(test_mode=TestMode.FORWARD_TEST)
    print(f"📋 Found {len(pending)} pending bets")
    
    # Test settling bet
    print("\n3️⃣ Testing bet settlement...")
    if bet_id:
        settled = integration.settle_match_bets(
            match_id=test_match['match_id'],
            winner='Novak Djokovic',
            score='6-4, 6-2, 6-3'
        )
        print(f"✅ Settled {settled} bets")
    
    # Test performance summary
    print("\n4️⃣ Testing performance summary...")
    summary = tracker.get_betting_performance_summary(
        test_mode=TestMode.FORWARD_TEST,
        days_back=30
    )
    print(f"📊 Performance summary: {summary}")
    
    print("\n💰 Betting Tracker Service testing completed!")