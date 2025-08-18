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
    
    def get_betting_statistics_by_timeframe(self,
                                           test_mode: TestMode = None,
                                           timeframe: str = '1_month') -> Dict:
        """
        Get comprehensive betting statistics for specific timeframes
        
        Args:
            test_mode: Testing mode filter
            timeframe: '1_week', '1_month', '1_year', 'all_time'
        
        Returns:
            Dict: Comprehensive statistics including all metrics
        """
        try:
            # Calculate date range based on timeframe with timezone awareness
            end_date = datetime.now()
            
            if timeframe == '1_week':
                start_date = end_date - timedelta(days=7)
                days_back = 7
                period_label = 'Last 7 Days'
            elif timeframe == '1_month':
                start_date = end_date - timedelta(days=30)
                days_back = 30
                period_label = 'Last 30 Days'
            elif timeframe == '1_year':
                start_date = end_date - timedelta(days=365)
                days_back = 365
                period_label = 'Last 365 Days'
            else:  # all_time
                start_date = None
                days_back = 999999
                period_label = 'All Time'
            
            # Get betting logs for the timeframe
            query = self.session.query(BettingLog).filter(
                BettingLog.betting_status.in_([BettingStatus.WON, BettingStatus.LOST])
            )
            
            if test_mode:
                query = query.filter(BettingLog.test_mode == test_mode)
            
            if start_date:
                query = query.filter(BettingLog.timestamp >= start_date)
            
            logs = query.order_by(BettingLog.timestamp).all()
            
            if not logs:
                return {
                    'timeframe': timeframe,
                    'test_mode': test_mode.value if test_mode else 'all',
                    'message': f'No betting data available for {timeframe}',
                    'basic_metrics': self._get_empty_metrics(),
                    'date_range': {
                        'start_date': start_date.isoformat() if start_date else None,
                        'end_date': end_date.isoformat(),
                        'days_included': days_back if days_back != 999999 else 'all_time'
                    }
                }
            
            # Calculate all statistics
            statistics = self._calculate_comprehensive_statistics(logs, timeframe)
            statistics['timeframe'] = timeframe
            statistics['period_label'] = period_label
            statistics['test_mode'] = test_mode.value if test_mode else 'all'
            statistics['date_range'] = {
                'start_date': start_date.isoformat() if start_date else None,
                'end_date': end_date.isoformat(),
                'days_included': days_back if days_back != 999999 else 'all_time',
                'timezone': 'UTC'
            }
            
            # Add enhanced data quality metrics
            statistics['data_quality'] = self._calculate_enhanced_data_quality(logs, timeframe)
            
            return statistics
            
        except Exception as e:
            logger.error(f"❌ Error getting betting statistics for {timeframe}: {e}")
            return {
                'timeframe': timeframe,
                'period_label': timeframe.replace('_', ' ').title(),
                'error': str(e),
                'basic_metrics': self._get_empty_metrics(),
                'data_quality': {
                    'sample_size': 0,
                    'data_completeness': 'error',
                    'statistical_significance': 'error',
                    'quality_score': 0
                }
            }

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
    
    def _get_empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {
            'total_bets': 0,
            'winning_bets': 0,
            'losing_bets': 0,
            'win_rate': 0,
            'total_staked': 0,
            'total_returned': 0,
            'net_profit': 0,
            'roi_percentage': 0
        }
    
    def _calculate_comprehensive_statistics(self, logs: List, timeframe: str) -> Dict:
        """
        Calculate comprehensive betting statistics from log records
        
        Args:
            logs: List of BettingLog records
            timeframe: Timeframe for analysis
        
        Returns:
            Dict: Complete statistics breakdown
        """
        try:
            if not logs:
                return {
                    'basic_metrics': self._get_empty_metrics(),
                    'message': f'No betting data available for {timeframe}'
                }
            
            # Basic metrics
            total_bets = len(logs)
            winning_bets = len([log for log in logs if log.betting_status == BettingStatus.WON])
            losing_bets = total_bets - winning_bets
            win_rate = (winning_bets / total_bets) * 100 if total_bets > 0 else 0
            
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
            average_stake = total_staked / total_bets if total_bets > 0 else 0
            average_odds = sum(log.odds_taken for log in logs) / total_bets if total_bets > 0 else 0
            average_edge = sum(log.edge_percentage for log in logs) / total_bets if total_bets > 0 else 0
            
            # Advanced risk metrics
            risk_metrics = self._calculate_advanced_risk_metrics(logs)
            
            # Streak analysis
            current_streak = self._calculate_current_streak(logs)
            longest_winning_streak = self._calculate_longest_streak(logs, 'win')
            longest_losing_streak = self._calculate_longest_streak(logs, 'loss')
            
            # Time-based analysis
            time_analysis = self._calculate_time_based_metrics(logs, timeframe)
            
            # Odds distribution analysis
            odds_analysis = self._calculate_odds_distribution(logs)
            
            # Model performance breakdown
            model_performance = self._calculate_model_performance(logs)
            
            # Rolling performance metrics
            rolling_metrics = self._calculate_rolling_metrics(logs, timeframe)
            
            return {
                'basic_metrics': {
                    'total_bets': total_bets,
                    'winning_bets': winning_bets,
                    'losing_bets': losing_bets,
                    'win_rate': round(win_rate, 2),
                    'pending_bets': 0  # These are settled logs only
                },
                'financial_metrics': {
                    'total_staked': round(total_staked, 2),
                    'total_returned': round(total_returned, 2),
                    'net_profit': round(net_profit, 2),
                    'roi_percentage': round(roi_percentage, 2),
                    'profit_factor': round(profit_factor, 2)
                },
                'average_metrics': {
                    'average_stake': round(average_stake, 2),
                    'average_odds': round(average_odds, 2),
                    'average_edge': round(average_edge, 2),
                    'average_win': round(total_wins_amount / winning_bets, 2) if winning_bets > 0 else 0,
                    'average_loss': round(total_losses_amount / losing_bets, 2) if losing_bets > 0 else 0
                },
                'risk_metrics': risk_metrics,
                'streak_analysis': {
                    'current_streak': current_streak,
                    'longest_winning_streak': longest_winning_streak,
                    'longest_losing_streak': longest_losing_streak
                },
                'time_analysis': time_analysis,
                'odds_analysis': odds_analysis,
                'model_performance': model_performance,
                'rolling_metrics': rolling_metrics,
                'data_quality': {
                    'sample_size': total_bets,
                    'data_completeness': 'complete' if total_bets >= 20 else 'limited',
                    'statistical_significance': 'high' if total_bets >= 50 else 'medium' if total_bets >= 20 else 'low'
                },
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating comprehensive statistics: {e}")
            return {
                'basic_metrics': self._get_empty_metrics(),
                'error': str(e)
            }
    
    def _calculate_advanced_risk_metrics(self, logs: List) -> Dict:
        """Calculate advanced risk metrics including Sharpe ratio, VaR, Sortino ratio"""
        try:
            if not logs:
                return {
                    'sharpe_ratio': 0,
                    'sortino_ratio': 0,
                    'max_drawdown': 0,
                    'var_95': 0,
                    'var_99': 0,
                    'largest_win': 0,
                    'largest_loss': 0
                }
            
            profits = [log.profit_loss for log in logs]
            
            # Basic risk metrics
            largest_win = max(profits) if profits else 0
            largest_loss = abs(min(profits)) if profits else 0
            max_drawdown = self._calculate_max_drawdown(logs)
            
            # Sharpe ratio (risk-free rate assumed to be 0)
            mean_return = sum(profits) / len(profits) if profits else 0
            variance = sum((p - mean_return) ** 2 for p in profits) / len(profits) if profits else 0
            std_dev = variance ** 0.5
            sharpe_ratio = mean_return / std_dev if std_dev > 0 else 0
            
            # Sortino ratio (downside deviation)
            negative_returns = [p for p in profits if p < 0]
            downside_variance = sum(p ** 2 for p in negative_returns) / len(profits) if negative_returns else 0
            downside_deviation = downside_variance ** 0.5
            sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else 0
            
            # Value at Risk (VaR)
            sorted_profits = sorted(profits)
            var_95_index = int(0.05 * len(sorted_profits))
            var_99_index = int(0.01 * len(sorted_profits))
            var_95 = abs(sorted_profits[var_95_index]) if var_95_index < len(sorted_profits) else 0
            var_99 = abs(sorted_profits[var_99_index]) if var_99_index < len(sorted_profits) else 0
            
            return {
                'sharpe_ratio': round(sharpe_ratio, 3),
                'sortino_ratio': round(sortino_ratio, 3),
                'max_drawdown': round(max_drawdown, 2),
                'var_95': round(var_95, 2),  # 95% VaR
                'var_99': round(var_99, 2),  # 99% VaR
                'largest_win': round(largest_win, 2),
                'largest_loss': round(largest_loss, 2),
                'volatility': round(std_dev, 3),
                'downside_deviation': round(downside_deviation, 3)
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating advanced risk metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_time_based_metrics(self, logs: List, timeframe: str) -> Dict:
        """Calculate time-based performance metrics"""
        try:
            if not logs:
                return {'profit_by_period': []}
            
            from collections import defaultdict
            period_data = defaultdict(lambda: {
                'profit': 0,
                'bets': 0,
                'wins': 0,
                'stakes': 0
            })
            
            for log in logs:
                try:
                    timestamp = log.timestamp
                    
                    if timeframe == '1_week':
                        # Group by day
                        period_key = timestamp.strftime('%Y-%m-%d')
                    elif timeframe == '1_month':
                        # Group by week
                        week_start = timestamp - timedelta(days=timestamp.weekday())
                        period_key = week_start.strftime('%Y-W%W')
                    elif timeframe == '1_year':
                        # Group by month
                        period_key = timestamp.strftime('%Y-%m')
                    else:
                        # Group by month for all time
                        period_key = timestamp.strftime('%Y-%m')
                    
                    period_data[period_key]['profit'] += log.profit_loss
                    period_data[period_key]['bets'] += 1
                    period_data[period_key]['stakes'] += log.stake_amount
                    if log.betting_status == BettingStatus.WON:
                        period_data[period_key]['wins'] += 1
                        
                except Exception as e:
                    logger.warning(f"Error processing log timestamp: {e}")
                    continue
            
            # Convert to list and calculate additional metrics
            profit_by_period = []
            for period, data in sorted(period_data.items()):
                win_rate = (data['wins'] / data['bets']) * 100 if data['bets'] > 0 else 0
                roi = (data['profit'] / data['stakes']) * 100 if data['stakes'] > 0 else 0
                
                profit_by_period.append({
                    'period': period,
                    'profit': round(data['profit'], 2),
                    'bets': data['bets'],
                    'win_rate': round(win_rate, 1),
                    'roi': round(roi, 1)
                })
            
            return {
                'profit_by_period': profit_by_period,
                'best_period': max(profit_by_period, key=lambda x: x['profit']) if profit_by_period else None,
                'worst_period': min(profit_by_period, key=lambda x: x['profit']) if profit_by_period else None
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating time-based metrics: {e}")
            return {'profit_by_period': []}
    
    def _calculate_odds_distribution(self, logs: List) -> Dict:
        """Calculate betting distribution by odds ranges"""
        try:
            if not logs:
                return {'odds_ranges': []}
            
            # Define odds ranges
            ranges = [
                (1.0, 1.5, 'Heavy Favorites'),
                (1.5, 2.0, 'Favorites'), 
                (2.0, 3.0, 'Moderate'),
                (3.0, 5.0, 'Underdogs'),
                (5.0, float('inf'), 'Heavy Underdogs')
            ]
            
            range_data = []
            for min_odds, max_odds, label in ranges:
                range_logs = [log for log in logs if min_odds <= log.odds_taken < max_odds]
                
                if range_logs:
                    total_bets = len(range_logs)
                    wins = len([log for log in range_logs if log.betting_status == BettingStatus.WON])
                    win_rate = (wins / total_bets) * 100
                    total_profit = sum(log.profit_loss for log in range_logs)
                    total_stakes = sum(log.stake_amount for log in range_logs)
                    roi = (total_profit / total_stakes) * 100 if total_stakes > 0 else 0
                    
                    range_data.append({
                        'range': label,
                        'odds_range': f"{min_odds}-{max_odds if max_odds != float('inf') else '∞'}",
                        'bets': total_bets,
                        'win_rate': round(win_rate, 1),
                        'profit': round(total_profit, 2),
                        'roi': round(roi, 1)
                    })
            
            return {
                'odds_ranges': range_data,
                'most_profitable_range': max(range_data, key=lambda x: x['roi']) if range_data else None
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating odds distribution: {e}")
            return {'odds_ranges': []}
    
    def _calculate_model_performance(self, logs: List) -> Dict:
        """Calculate performance breakdown by ML model"""
        try:
            if not logs:
                return {}
            
            model_stats = {}
            
            for log in logs:
                model = log.model_used or 'Unknown'
                
                if model not in model_stats:
                    model_stats[model] = {
                        'total_bets': 0,
                        'winning_bets': 0,
                        'net_profit': 0,
                        'total_staked': 0
                    }
                
                model_stats[model]['total_bets'] += 1
                model_stats[model]['total_staked'] += log.stake_amount
                model_stats[model]['net_profit'] += log.profit_loss
                
                if log.betting_status == BettingStatus.WON:
                    model_stats[model]['winning_bets'] += 1
            
            # Calculate derived metrics
            for model, stats in model_stats.items():
                stats['win_rate'] = (stats['winning_bets'] / stats['total_bets']) * 100
                stats['roi'] = (stats['net_profit'] / stats['total_staked']) * 100 if stats['total_staked'] > 0 else 0
                # Round values
                stats['win_rate'] = round(stats['win_rate'], 1)
                stats['roi'] = round(stats['roi'], 1)
                stats['net_profit'] = round(stats['net_profit'], 2)
            
            return model_stats
            
        except Exception as e:
            logger.error(f"❌ Error calculating model performance: {e}")
            return {}
    
    def _calculate_rolling_metrics(self, logs: List, timeframe: str) -> Dict:
        """Calculate rolling performance metrics"""
        try:
            if len(logs) < 10:  # Need minimum data for rolling metrics
                return {'rolling_roi': [], 'rolling_win_rate': []}
            
            # Sort logs by timestamp
            sorted_logs = sorted(logs, key=lambda x: x.timestamp)
            
            # Calculate window size based on timeframe
            if timeframe == '1_week':
                window_size = min(7, len(sorted_logs) // 3)
            elif timeframe == '1_month':
                window_size = min(15, len(sorted_logs) // 3)
            elif timeframe == '1_year':
                window_size = min(30, len(sorted_logs) // 5)
            else:
                window_size = min(50, len(sorted_logs) // 5)
            
            rolling_roi = []
            rolling_win_rate = []
            
            for i in range(window_size, len(sorted_logs) + 1):
                window_logs = sorted_logs[i - window_size:i]
                
                # Calculate window metrics
                total_stakes = sum(log.stake_amount for log in window_logs)
                total_profit = sum(log.profit_loss for log in window_logs)
                wins = len([log for log in window_logs if log.betting_status == BettingStatus.WON])
                
                roi = (total_profit / total_stakes) * 100 if total_stakes > 0 else 0
                win_rate = (wins / len(window_logs)) * 100 if window_logs else 0
                
                rolling_roi.append({
                    'period': window_logs[-1].timestamp.strftime('%Y-%m-%d'),
                    'roi': round(roi, 2)
                })
                
                rolling_win_rate.append({
                    'period': window_logs[-1].timestamp.strftime('%Y-%m-%d'),
                    'win_rate': round(win_rate, 1)
                })
            
            return {
                'rolling_roi': rolling_roi,
                'rolling_win_rate': rolling_win_rate,
                'window_size': window_size
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating rolling metrics: {e}")
            return {'rolling_roi': [], 'rolling_win_rate': []}

    def _calculate_enhanced_data_quality(self, logs: List, timeframe: str) -> Dict:
        """Calculate enhanced data quality metrics and statistical significance"""
        try:
            sample_size = len(logs)
            
            # Determine data completeness
            if sample_size == 0:
                data_completeness = 'no_data'
                statistical_significance = 'insufficient'
                quality_score = 0
            elif sample_size < 10:
                data_completeness = 'very_limited'
                statistical_significance = 'low'
                quality_score = 1
            elif sample_size < 30:
                data_completeness = 'limited'
                statistical_significance = 'medium'
                quality_score = 2
            elif sample_size < 100:
                data_completeness = 'good'
                statistical_significance = 'high'
                quality_score = 3
            else:
                data_completeness = 'excellent'
                statistical_significance = 'very_high'
                quality_score = 4
            
            # Calculate additional quality metrics
            if sample_size > 0:
                # Time distribution quality
                sorted_logs = sorted(logs, key=lambda x: x.timestamp)
                if len(sorted_logs) > 1:
                    time_span = (sorted_logs[-1].timestamp - sorted_logs[0].timestamp).days
                    if timeframe == '1_week' and time_span >= 5:
                        time_distribution_quality = 'good'
                    elif timeframe == '1_month' and time_span >= 20:
                        time_distribution_quality = 'good'
                    elif timeframe == '1_year' and time_span >= 200:
                        time_distribution_quality = 'good'
                    else:
                        time_distribution_quality = 'concentrated'
                else:
                    time_distribution_quality = 'single_point'
                
                # Model diversity
                models_used = set(log.model_used for log in logs if log.model_used)
                model_diversity = len(models_used)
                
                # Stakes consistency
                stakes = [log.stake_amount for log in logs]
                if stakes:
                    stake_std = (sum((s - sum(stakes)/len(stakes))**2 for s in stakes) / len(stakes))**0.5
                    stake_cv = stake_std / (sum(stakes)/len(stakes)) if sum(stakes) > 0 else 0
                    stakes_consistency = 'consistent' if stake_cv < 0.5 else 'variable'
                else:
                    stakes_consistency = 'unknown'
                
                # Calculate confidence intervals for key metrics
                wins = len([log for log in logs if log.betting_status == BettingStatus.WON])
                win_rate = wins / sample_size if sample_size > 0 else 0
                
                # Binomial confidence interval (Wilson score)
                if sample_size >= 5:
                    import math
                    z = 1.96  # 95% confidence
                    p = win_rate
                    n = sample_size
                    
                    denominator = 1 + z**2/n
                    centre_adjusted_probability = (p + z**2/(2*n)) / denominator
                    adjusted_standard_deviation = math.sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator
                    
                    win_rate_ci_lower = centre_adjusted_probability - z * adjusted_standard_deviation
                    win_rate_ci_upper = centre_adjusted_probability + z * adjusted_standard_deviation
                    
                    ci_width = win_rate_ci_upper - win_rate_ci_lower
                    precision = 'high' if ci_width < 0.2 else 'medium' if ci_width < 0.4 else 'low'
                else:
                    win_rate_ci_lower = 0
                    win_rate_ci_upper = 1
                    precision = 'very_low'
            else:
                time_distribution_quality = 'no_data'
                model_diversity = 0
                stakes_consistency = 'no_data'
                win_rate_ci_lower = 0
                win_rate_ci_upper = 1
                precision = 'no_data'
            
            # Generate recommendations
            recommendations = []
            if sample_size < 30:
                recommendations.append("Collect more data for reliable statistics")
            if timeframe != 'all_time' and sample_size == 0:
                recommendations.append(f"No bets found in {timeframe} period")
            if sample_size > 0 and model_diversity <= 1:
                recommendations.append("Consider diversifying ML models for better insights")
            
            return {
                'sample_size': sample_size,
                'data_completeness': data_completeness,
                'statistical_significance': statistical_significance,
                'quality_score': quality_score,
                'time_distribution_quality': time_distribution_quality,
                'model_diversity': model_diversity,
                'stakes_consistency': stakes_consistency,
                'confidence_intervals': {
                    'win_rate_lower': round(win_rate_ci_lower * 100, 1),
                    'win_rate_upper': round(win_rate_ci_upper * 100, 1),
                    'precision': precision
                },
                'recommendations': recommendations,
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating data quality: {e}")
            return {
                'sample_size': 0,
                'data_completeness': 'error',
                'statistical_significance': 'error',
                'quality_score': 0,
                'error': str(e)
            }

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
    
    def get_chart_data_for_timeframe(self, 
                                   test_mode: TestMode = None,
                                   timeframe: str = '1_month',
                                   chart_type: str = 'profit_timeline') -> Dict:
        """
        Generate chart data for betting statistics visualization with enhanced error handling
        
        Args:
            test_mode: Testing mode filter
            timeframe: Time period for data
            chart_type: Type of chart (profit_timeline, win_rate_trend, odds_distribution, etc.)
        
        Returns:
            Dict: Chart data in format suitable for frontend visualization
        """
        try:
            # Validate inputs
            valid_timeframes = ['1_week', '1_month', '1_year', 'all_time']
            valid_chart_types = [
                'profit_timeline', 'win_rate_trend', 'odds_distribution',
                'monthly_performance', 'risk_metrics', 'model_comparison',
                'data_quality_overview'
            ]
            
            if timeframe not in valid_timeframes:
                return {
                    'labels': [], 
                    'datasets': [], 
                    'error': f'Invalid timeframe: {timeframe}. Valid options: {valid_timeframes}'
                }
            
            if chart_type not in valid_chart_types:
                return {
                    'labels': [], 
                    'datasets': [], 
                    'error': f'Invalid chart type: {chart_type}. Valid options: {valid_chart_types}'
                }
            
            # Get statistics data for the timeframe
            statistics = self.get_betting_statistics_by_timeframe(
                test_mode=test_mode,
                timeframe=timeframe
            )
            
            if 'error' in statistics:
                return {
                    'labels': [], 
                    'datasets': [], 
                    'error': statistics['error'],
                    'data_quality': statistics.get('data_quality', {})
                }
            
            # Check data quality before generating charts
            data_quality = statistics.get('data_quality', {})
            sample_size = data_quality.get('sample_size', 0)
            
            if sample_size == 0:
                return {
                    'labels': ['No Data'],
                    'datasets': [{
                        'label': 'No betting data available',
                        'data': [0],
                        'backgroundColor': 'rgba(255, 107, 107, 0.7)',
                        'borderColor': '#ff6b6b'
                    }],
                    'message': f'No betting data available for {timeframe}',
                    'data_quality': data_quality
                }
            
            # Generate chart data based on type
            chart_data = None
            if chart_type == 'profit_timeline':
                chart_data = self._generate_profit_timeline_chart(statistics)
            elif chart_type == 'win_rate_trend':
                chart_data = self._generate_win_rate_chart(statistics)
            elif chart_type == 'odds_distribution':
                chart_data = self._generate_odds_distribution_chart(statistics)
            elif chart_type == 'monthly_performance':
                chart_data = self._generate_monthly_performance_chart(statistics)
            elif chart_type == 'risk_metrics':
                chart_data = self._generate_risk_metrics_chart(statistics)
            elif chart_type == 'model_comparison':
                chart_data = self._generate_model_comparison_chart(statistics)
            elif chart_type == 'data_quality_overview':
                chart_data = self._generate_data_quality_chart(statistics)
            
            # Add metadata to chart data
            if chart_data:
                chart_data['data_quality'] = data_quality
                chart_data['timeframe'] = timeframe
                chart_data['chart_type'] = chart_type
                chart_data['sample_size'] = sample_size
                chart_data['last_updated'] = datetime.utcnow().isoformat()
            
            return chart_data or {'labels': [], 'datasets': [], 'error': 'Failed to generate chart data'}
                
        except Exception as e:
            logger.error(f"❌ Error generating chart data for {chart_type}: {e}")
            return {
                'labels': [], 
                'datasets': [], 
                'error': str(e),
                'chart_type': chart_type,
                'timeframe': timeframe
            }
    
    def _generate_profit_timeline_chart(self, statistics: Dict) -> Dict:
        """Generate profit timeline chart data with enhanced validation"""
        try:
            time_analysis = statistics.get('time_analysis', {})
            profit_by_period = time_analysis.get('profit_by_period', [])
            
            if not profit_by_period:
                return {
                    'labels': ['No Data'],
                    'datasets': [{
                        'label': 'No profit timeline data available',
                        'data': [0],
                        'backgroundColor': 'rgba(255, 193, 7, 0.7)',
                        'borderColor': '#ffc107'
                    }],
                    'message': 'Insufficient data for profit timeline analysis'
                }
            
            labels = [item['period'] for item in profit_by_period]
            cumulative_profit = []
            running_total = 0
            
            for item in profit_by_period:
                running_total += item['profit']
                cumulative_profit.append(running_total)
            
            return {
                'labels': labels,
                'datasets': [
                    {
                        'label': 'Cumulative Profit',
                        'data': cumulative_profit,
                        'borderColor': '#10B981',
                        'backgroundColor': 'rgba(16, 185, 129, 0.1)',
                        'fill': True
                    },
                    {
                        'label': 'Period Profit',
                        'data': [item['profit'] for item in profit_by_period],
                        'borderColor': '#3B82F6',
                        'backgroundColor': 'rgba(59, 130, 246, 0.1)',
                        'type': 'bar'
                    }
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating profit timeline chart: {e}")
            return {
                'labels': ['Error'],
                'datasets': [{
                    'label': 'Chart generation error',
                    'data': [0],
                    'backgroundColor': 'rgba(255, 107, 107, 0.7)',
                    'borderColor': '#ff6b6b'
                }],
                'error': str(e)
            }
    
    def _generate_win_rate_chart(self, statistics: Dict) -> Dict:
        """Generate win rate trend chart data"""
        try:
            rolling_metrics = statistics.get('rolling_metrics', {})
            rolling_win_rate = rolling_metrics.get('rolling_win_rate', [])
            
            if not rolling_win_rate:
                return {'labels': [], 'datasets': []}
            
            labels = [item['period'] for item in rolling_win_rate]
            win_rates = [item['win_rate'] for item in rolling_win_rate]
            
            # Add overall win rate line
            overall_win_rate = statistics.get('basic_metrics', {}).get('win_rate', 0)
            overall_line = [overall_win_rate] * len(labels)
            
            return {
                'labels': labels,
                'datasets': [
                    {
                        'label': f'Rolling Win Rate ({rolling_metrics.get("window_size", 10)} bet window)',
                        'data': win_rates,
                        'borderColor': '#8B5CF6',
                        'backgroundColor': 'rgba(139, 92, 246, 0.1)',
                        'fill': True
                    },
                    {
                        'label': 'Overall Win Rate',
                        'data': overall_line,
                        'borderColor': '#EF4444',
                        'backgroundColor': 'transparent',
                        'borderDash': [5, 5]
                    }
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating win rate chart: {e}")
            return {'labels': [], 'datasets': []}
    
    def _generate_odds_distribution_chart(self, statistics: Dict) -> Dict:
        """Generate odds distribution chart data"""
        try:
            odds_analysis = statistics.get('odds_analysis', {})
            odds_ranges = odds_analysis.get('odds_ranges', [])
            
            if not odds_ranges:
                return {'labels': [], 'datasets': []}
            
            labels = [item['range'] for item in odds_ranges]
            bets = [item['bets'] for item in odds_ranges]
            roi_values = [item['roi'] for item in odds_ranges]
            win_rates = [item['win_rate'] for item in odds_ranges]
            
            return {
                'labels': labels,
                'datasets': [
                    {
                        'label': 'Number of Bets',
                        'data': bets,
                        'backgroundColor': 'rgba(59, 130, 246, 0.7)',
                        'yAxisID': 'y'
                    },
                    {
                        'label': 'ROI (%)',
                        'data': roi_values,
                        'backgroundColor': 'rgba(16, 185, 129, 0.7)',
                        'yAxisID': 'y1',
                        'type': 'line'
                    },
                    {
                        'label': 'Win Rate (%)',
                        'data': win_rates,
                        'backgroundColor': 'rgba(139, 92, 246, 0.7)',
                        'yAxisID': 'y1',
                        'type': 'line'
                    }
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating odds distribution chart: {e}")
            return {'labels': [], 'datasets': []}
    
    def _generate_monthly_performance_chart(self, statistics: Dict) -> Dict:
        """Generate monthly performance breakdown chart"""
        try:
            time_analysis = statistics.get('time_analysis', {})
            profit_by_period = time_analysis.get('profit_by_period', [])
            
            if not profit_by_period:
                return {'labels': [], 'datasets': []}
            
            labels = [item['period'] for item in profit_by_period]
            profits = [item['profit'] for item in profit_by_period]
            roi_values = [item['roi'] for item in profit_by_period]
            bet_counts = [item['bets'] for item in profit_by_period]
            
            return {
                'labels': labels,
                'datasets': [
                    {
                        'label': 'Profit',
                        'data': profits,
                        'backgroundColor': [
                            'rgba(16, 185, 129, 0.7)' if p >= 0 else 'rgba(239, 68, 68, 0.7)' 
                            for p in profits
                        ],
                        'yAxisID': 'y'
                    },
                    {
                        'label': 'ROI (%)',
                        'data': roi_values,
                        'borderColor': '#8B5CF6',
                        'backgroundColor': 'transparent',
                        'type': 'line',
                        'yAxisID': 'y1'
                    },
                    {
                        'label': 'Number of Bets',
                        'data': bet_counts,
                        'borderColor': '#F59E0B',
                        'backgroundColor': 'rgba(245, 158, 11, 0.1)',
                        'type': 'line',
                        'yAxisID': 'y2'
                    }
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating monthly performance chart: {e}")
            return {'labels': [], 'datasets': []}
    
    def _generate_risk_metrics_chart(self, statistics: Dict) -> Dict:
        """Generate risk metrics visualization"""
        try:
            risk_metrics = statistics.get('risk_metrics', {})
            
            if not risk_metrics:
                return {'labels': [], 'datasets': []}
            
            labels = ['Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'VaR 95%', 'VaR 99%']
            values = [
                risk_metrics.get('sharpe_ratio', 0),
                risk_metrics.get('sortino_ratio', 0),
                -risk_metrics.get('max_drawdown', 0),  # Negative for visualization
                -risk_metrics.get('var_95', 0),
                -risk_metrics.get('var_99', 0)
            ]
            
            colors = [
                'rgba(16, 185, 129, 0.7)' if v >= 0 else 'rgba(239, 68, 68, 0.7)'
                for v in values
            ]
            
            return {
                'labels': labels,
                'datasets': [
                    {
                        'label': 'Risk Metrics',
                        'data': values,
                        'backgroundColor': colors,
                        'borderColor': colors,
                        'borderWidth': 1
                    }
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating risk metrics chart: {e}")
            return {'labels': [], 'datasets': []}
    
    def _generate_model_comparison_chart(self, statistics: Dict) -> Dict:
        """Generate model performance comparison chart"""
        try:
            model_performance = statistics.get('model_performance', {})
            
            if not model_performance:
                return {'labels': [], 'datasets': []}
            
            models = list(model_performance.keys())
            win_rates = [model_performance[model]['win_rate'] for model in models]
            roi_values = [model_performance[model]['roi'] for model in models]
            total_bets = [model_performance[model]['total_bets'] for model in models]
            
            return {
                'labels': models,
                'datasets': [
                    {
                        'label': 'Win Rate (%)',
                        'data': win_rates,
                        'backgroundColor': 'rgba(59, 130, 246, 0.7)',
                        'yAxisID': 'y'
                    },
                    {
                        'label': 'ROI (%)',
                        'data': roi_values,
                        'backgroundColor': 'rgba(16, 185, 129, 0.7)',
                        'yAxisID': 'y'
                    },
                    {
                        'label': 'Total Bets',
                        'data': total_bets,
                        'borderColor': '#8B5CF6',
                        'backgroundColor': 'transparent',
                        'type': 'line',
                        'yAxisID': 'y1'
                    }
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating model comparison chart: {e}")
            return {'labels': [], 'datasets': []}

    def _generate_data_quality_chart(self, statistics: Dict) -> Dict:
        """Generate data quality overview chart"""
        try:
            data_quality = statistics.get('data_quality', {})
            
            if not data_quality:
                return {'labels': [], 'datasets': []}
            
            # Quality score breakdown
            quality_labels = [
                'Sample Size Score',
                'Statistical Significance',
                'Time Distribution',
                'Model Diversity',
                'Overall Quality Score'
            ]
            
            # Convert quality metrics to scores (0-100)
            sample_size = data_quality.get('sample_size', 0)
            quality_score = data_quality.get('quality_score', 0) * 25  # Convert 0-4 to 0-100
            model_diversity = min(data_quality.get('model_diversity', 0) * 20, 100)  # Max 5 models
            
            # Time distribution score
            time_quality = data_quality.get('time_distribution_quality', 'no_data')
            time_score = {
                'good': 100,
                'concentrated': 60,
                'single_point': 20,
                'no_data': 0
            }.get(time_quality, 0)
            
            # Statistical significance score
            significance = data_quality.get('statistical_significance', 'insufficient')
            significance_score = {
                'very_high': 100,
                'high': 80,
                'medium': 60,
                'low': 40,
                'insufficient': 20,
                'error': 0
            }.get(significance, 0)
            
            scores = [
                min(sample_size * 2, 100),  # Sample size score (50 bets = 100%)
                significance_score,
                time_score,
                model_diversity,
                quality_score
            ]
            
            # Generate color based on score
            colors = []
            for score in scores:
                if score >= 80:
                    colors.append('rgba(16, 185, 129, 0.7)')  # Green
                elif score >= 60:
                    colors.append('rgba(245, 158, 11, 0.7)')  # Yellow
                elif score >= 40:
                    colors.append('rgba(251, 146, 60, 0.7)')  # Orange
                else:
                    colors.append('rgba(239, 68, 68, 0.7)')   # Red
            
            return {
                'labels': quality_labels,
                'datasets': [
                    {
                        'label': 'Data Quality Scores',
                        'data': scores,
                        'backgroundColor': colors,
                        'borderColor': colors,
                        'borderWidth': 1
                    }
                ],
                'options': {
                    'scales': {
                        'y': {
                            'beginAtZero': True,
                            'max': 100
                        }
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating data quality chart: {e}")
            return {'labels': [], 'datasets': []}
    
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