#!/usr/bin/env python3
"""
🎾 Prediction-to-Betting Integration Service

This service bridges the gap between ML predictions sent as Telegram notifications
and the betting simulation system. Every prediction notification is automatically
converted into a betting record for comprehensive performance tracking.

Features:
- Captures all Telegram notification predictions as betting records
- Converts predictions into simulated bets with proper stake calculations
- Tracks prediction outcomes and calculates ROI
- Integrates with web dashboard for real-time statistics
- Supports both live and historical data integration

Author: Claude Code (Anthropic)
"""

import logging
import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import uuid

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Database imports
from src.data.database_models import (
    DatabaseManager, BettingLog, Prediction, TestMode, BettingStatus
)

# Telegram integration (imported separately to avoid circular imports)

logger = logging.getLogger(__name__)

@dataclass
class PredictionBettingConfig:
    """Configuration for prediction-to-betting integration"""
    default_stake_amount: float = 25.0  # Default stake per bet
    stake_percentage: float = 2.5  # Percentage of bankroll per bet
    minimum_confidence_threshold: float = 0.55  # Minimum confidence to record as bet
    kelly_fraction: float = 0.25  # Conservative Kelly fraction
    max_stake_percentage: float = 5.0  # Maximum stake as % of bankroll
    initial_bankroll: float = 1000.0  # Starting bankroll for simulation

class PredictionBettingIntegrator:
    """
    Integrates ML predictions with betting simulation system
    """
    
    def __init__(self, config: PredictionBettingConfig = None):
        self.config = config or PredictionBettingConfig()
        self.db_manager = DatabaseManager()
        self.current_bankroll = self.config.initial_bankroll
        self._setup_logging()
        self._initialize_database()
        
    def _setup_logging(self):
        """Setup logging for the integrator"""
        log_dir = os.path.join(project_root, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(os.path.join(log_dir, 'prediction_betting_integration.log'))
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.integration_logger = logging.getLogger('prediction_betting_integration')
        self.integration_logger.addHandler(file_handler)
        self.integration_logger.setLevel(logging.INFO)
        
    def _initialize_database(self):
        """Initialize database tables"""
        try:
            self.db_manager.create_tables()
            self.integration_logger.info("✅ Database tables initialized successfully")
        except Exception as e:
            self.integration_logger.error(f"❌ Failed to initialize database: {e}")
            raise
            
    def process_telegram_prediction(self, prediction_result: Dict) -> Optional[str]:
        """
        Process a prediction that was sent as Telegram notification
        and convert it into a betting record
        """
        try:
            # Validate prediction data
            if not self._validate_prediction_data(prediction_result):
                self.integration_logger.warning("❌ Invalid prediction data, skipping betting record creation")
                return None
                
            # Check if we should create a betting record
            if not self._should_create_betting_record(prediction_result):
                self.integration_logger.info("ℹ️ Prediction doesn't meet betting criteria, skipping")
                return None
                
            # Create prediction record
            prediction_id = self._create_prediction_record(prediction_result)
            if not prediction_id:
                self.integration_logger.error("❌ Failed to create prediction record")
                return None
                
            # Create betting record
            betting_record_id = self._create_betting_record(prediction_result, prediction_id)
            if not betting_record_id:
                self.integration_logger.error("❌ Failed to create betting record")
                return None
                
            # Update current bankroll tracking
            self._update_bankroll_tracking(betting_record_id)
            
            self.integration_logger.info(
                f"✅ Successfully created betting record {betting_record_id} for prediction"
            )
            
            return betting_record_id
            
        except Exception as e:
            self.integration_logger.error(f"❌ Error processing telegram prediction: {e}")
            return None
            
    def _validate_prediction_data(self, prediction_result: Dict) -> bool:
        """Validate that prediction data contains required fields"""
        required_fields = [
            'success', 'underdog_second_set_probability', 'confidence',
            'underdog_player', 'match_context'
        ]
        
        for field in required_fields:
            if field not in prediction_result:
                self.integration_logger.warning(f"Missing required field: {field}")
                return False
                
        match_context = prediction_result.get('match_context', {})
        required_match_fields = ['player1', 'player2', 'tournament']
        
        for field in required_match_fields:
            if field not in match_context:
                self.integration_logger.warning(f"Missing required match field: {field}")
                return False
                
        return True
        
    def _should_create_betting_record(self, prediction_result: Dict) -> bool:
        """Determine if this prediction should become a betting record"""
        
        # Check success status
        if not prediction_result.get('success', False):
            return False
            
        # Check confidence threshold
        probability = prediction_result.get('underdog_second_set_probability', 0)
        if probability < self.config.minimum_confidence_threshold:
            return False
            
        # Check confidence level
        confidence = prediction_result.get('confidence', '').lower()
        if confidence not in ['medium', 'high']:
            return False
            
        return True
        
    def _create_prediction_record(self, prediction_result: Dict) -> Optional[int]:
        """Create a prediction record in the database"""
        try:
            session = self.db_manager.get_session()
            
            match_context = prediction_result.get('match_context', {})
            
            # Extract match date
            match_date = self._parse_match_date(match_context)
            
            # Calculate bookmaker probability and edge (estimated)
            our_probability = prediction_result.get('underdog_second_set_probability', 0)
            estimated_odds = self._estimate_bookmaker_odds(our_probability)
            bookmaker_probability = 1 / estimated_odds
            edge = our_probability - bookmaker_probability
            
            prediction = Prediction(
                timestamp=datetime.utcnow(),
                match_date=match_date,
                player1=match_context.get('player1', 'Unknown'),
                player2=match_context.get('player2', 'Unknown'),
                tournament=match_context.get('tournament', 'Unknown'),
                surface=match_context.get('surface', 'Hard'),
                round_name=match_context.get('round', 'Unknown'),
                
                # Prediction data
                our_probability=our_probability,
                confidence=prediction_result.get('confidence', 'Medium'),
                ml_system='SecondSetUnderdogML',
                prediction_type='second_set_underdog',
                key_factors=json.dumps(prediction_result.get('strategic_insights', [])),
                
                # Bookmaker data (estimated)
                bookmaker_odds=estimated_odds,
                bookmaker_probability=bookmaker_probability,
                edge=edge,
                recommendation='BET' if edge > 0.05 else 'SKIP'
            )
            
            session.add(prediction)
            session.commit()
            
            prediction_id = prediction.id
            session.close()
            
            self.integration_logger.info(f"✅ Created prediction record {prediction_id}")
            return prediction_id
            
        except Exception as e:
            self.integration_logger.error(f"❌ Error creating prediction record: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return None
            
    def _create_betting_record(self, prediction_result: Dict, prediction_id: int) -> Optional[str]:
        """Create a betting record in the database"""
        try:
            session = self.db_manager.get_session()
            
            match_context = prediction_result.get('match_context', {})
            
            # Generate unique bet ID
            bet_id = f"telegram_bet_{uuid.uuid4().hex[:8]}"
            
            # Calculate stake
            stake_amount = self._calculate_stake(prediction_result)
            
            # Extract match date
            match_date = self._parse_match_date(match_context)
            
            # Determine predicted winner
            underdog_player = prediction_result.get('underdog_player', 'player1')
            if underdog_player == 'player1':
                predicted_winner = match_context.get('player1', 'Unknown')
            else:
                predicted_winner = match_context.get('player2', 'Unknown')
                
            # Calculate betting metrics
            our_probability = prediction_result.get('underdog_second_set_probability', 0)
            estimated_odds = self._estimate_bookmaker_odds(our_probability)
            implied_probability = 1 / estimated_odds
            edge_percentage = (our_probability - implied_probability) * 100
            
            betting_record = BettingLog(
                bet_id=bet_id,
                timestamp=datetime.utcnow(),
                
                # Match details
                match_id=f"telegram_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                player1=match_context.get('player1', 'Unknown'),
                player2=match_context.get('player2', 'Unknown'),
                tournament=match_context.get('tournament', 'Unknown'),
                match_date=match_date,
                
                # Test mode - mark as simulation
                test_mode=TestMode.LIVE,
                
                # Prediction details
                predicted_winner=predicted_winner,
                our_probability=our_probability,
                confidence_level=prediction_result.get('confidence', 'Medium'),
                model_used='SecondSetUnderdogML',
                key_factors=json.dumps(prediction_result.get('strategic_insights', [])),
                
                # Betting details
                bookmaker='Simulated',
                odds_taken=estimated_odds,
                implied_probability=implied_probability,
                edge_percentage=edge_percentage,
                
                # Stake management
                stake_amount=stake_amount,
                stake_percentage=(stake_amount / self.current_bankroll) * 100,
                kelly_fraction=self.config.kelly_fraction,
                risk_level=self._determine_risk_level(edge_percentage),
                
                # Status
                betting_status=BettingStatus.PLACED,
                
                # Link to prediction
                prediction_id=prediction_id
            )
            
            session.add(betting_record)
            session.commit()
            
            betting_record_id = betting_record.bet_id
            session.close()
            
            self.integration_logger.info(
                f"✅ Created betting record {betting_record_id} with stake ${stake_amount:.2f}"
            )
            
            return betting_record_id
            
        except Exception as e:
            self.integration_logger.error(f"❌ Error creating betting record: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return None
            
    def _calculate_stake(self, prediction_result: Dict) -> float:
        """Calculate stake amount using Kelly criterion"""
        try:
            our_probability = prediction_result.get('underdog_second_set_probability', 0)
            estimated_odds = self._estimate_bookmaker_odds(our_probability)
            
            # Kelly criterion calculation
            p = our_probability  # Probability of winning
            q = 1 - p  # Probability of losing
            b = estimated_odds - 1  # Net odds
            
            kelly_fraction = (b * p - q) / b
            
            # Apply conservative Kelly fraction
            conservative_kelly = kelly_fraction * self.config.kelly_fraction
            
            # Calculate stake amount
            stake = self.current_bankroll * max(0, conservative_kelly)
            
            # Apply limits
            max_stake = self.current_bankroll * (self.config.max_stake_percentage / 100)
            stake = min(stake, max_stake)
            
            # Ensure minimum stake
            stake = max(stake, self.config.default_stake_amount)
            
            return round(stake, 2)
            
        except Exception as e:
            self.integration_logger.warning(f"❌ Error calculating stake, using default: {e}")
            return self.config.default_stake_amount
            
    def _estimate_bookmaker_odds(self, our_probability: float) -> float:
        """Estimate bookmaker odds based on our probability"""
        if our_probability <= 0:
            return 10.0  # Very high odds for very low probability
            
        # Convert probability to decimal odds with bookmaker margin
        fair_odds = 1 / our_probability
        bookmaker_margin = 0.05  # 5% bookmaker margin
        estimated_odds = fair_odds * (1 + bookmaker_margin)
        
        # Clamp to reasonable range
        return max(1.1, min(10.0, estimated_odds))
        
    def _parse_match_date(self, match_context: Dict) -> datetime:
        """Parse match date from context"""
        try:
            # Try to parse match_date if provided
            match_date_str = match_context.get('match_date')
            if match_date_str:
                return datetime.fromisoformat(match_date_str.replace('Z', '+00:00'))
        except:
            pass
            
        # Default to current time + 2 hours (typical for upcoming matches)
        return datetime.utcnow() + timedelta(hours=2)
        
    def _determine_risk_level(self, edge_percentage: float) -> str:
        """Determine risk level based on edge percentage"""
        if edge_percentage >= 10:
            return 'Low'
        elif edge_percentage >= 5:
            return 'Medium'
        else:
            return 'High'
            
    def _update_bankroll_tracking(self, betting_record_id: str):
        """Update current bankroll after placing a bet"""
        try:
            session = self.db_manager.get_session()
            
            betting_record = session.query(BettingLog).filter_by(bet_id=betting_record_id).first()
            if betting_record:
                # Subtract stake from current bankroll
                self.current_bankroll -= betting_record.stake_amount
                self.integration_logger.info(
                    f"💰 Updated bankroll: ${self.current_bankroll:.2f} "
                    f"(staked ${betting_record.stake_amount:.2f})"
                )
                
            session.close()
            
        except Exception as e:
            self.integration_logger.error(f"❌ Error updating bankroll: {e}")
            
    def settle_betting_record(self, bet_id: str, match_result: Dict) -> bool:
        """Settle a betting record when match result is known"""
        try:
            session = self.db_manager.get_session()
            
            betting_record = session.query(BettingLog).filter_by(bet_id=bet_id).first()
            if not betting_record:
                self.integration_logger.warning(f"❌ Betting record {bet_id} not found")
                session.close()
                return False
                
            # Determine if prediction was correct
            actual_winner = match_result.get('winner')
            prediction_correct = (actual_winner == betting_record.predicted_winner)
            
            # Calculate financial results
            if prediction_correct:
                payout = betting_record.stake_amount * betting_record.odds_taken
                profit_loss = payout - betting_record.stake_amount
                betting_status = BettingStatus.WON
                value_bet_outcome = 'profitable'
                
                # Add payout to bankroll
                self.current_bankroll += payout
                
            else:
                payout = 0
                profit_loss = -betting_record.stake_amount
                betting_status = BettingStatus.LOST
                value_bet_outcome = 'unprofitable'
                
            # Calculate ROI
            roi_percentage = (profit_loss / betting_record.stake_amount) * 100
            
            # Update betting record
            betting_record.betting_status = betting_status
            betting_record.actual_winner = actual_winner
            betting_record.match_score = match_result.get('score', '')
            betting_record.payout_amount = payout
            betting_record.profit_loss = profit_loss
            betting_record.roi_percentage = roi_percentage
            betting_record.prediction_correct = prediction_correct
            betting_record.value_bet_outcome = value_bet_outcome
            betting_record.settled_at = datetime.utcnow()
            betting_record.settlement_notes = f"Auto-settled from match result: {actual_winner} won"
            
            # Update prediction record
            if betting_record.prediction:
                betting_record.prediction.actual_result = 'win' if prediction_correct else 'loss'
                betting_record.prediction.actual_winner = actual_winner
                betting_record.prediction.match_score = match_result.get('score', '')
                betting_record.prediction.prediction_correct = prediction_correct
                betting_record.prediction.probability_error = abs(
                    betting_record.our_probability - (1 if prediction_correct else 0)
                )
                betting_record.prediction.roi = roi_percentage
                
            session.commit()
            session.close()
            
            self.integration_logger.info(
                f"✅ Settled bet {bet_id}: {'WON' if prediction_correct else 'LOST'} "
                f"(P&L: ${profit_loss:.2f}, ROI: {roi_percentage:.1f}%)"
            )
            
            return True
            
        except Exception as e:
            self.integration_logger.error(f"❌ Error settling betting record {bet_id}: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False
            
    def get_betting_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get betting statistics for the dashboard"""
        try:
            session = self.db_manager.get_session()
            
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Query betting records
            betting_records = session.query(BettingLog).filter(
                BettingLog.timestamp >= start_date,
                BettingLog.timestamp <= end_date,
                BettingLog.betting_status.in_([BettingStatus.PLACED, BettingStatus.WON, BettingStatus.LOST])
            ).all()
            
            # Calculate statistics
            total_bets = len(betting_records)
            settled_bets = [bet for bet in betting_records if bet.betting_status in [BettingStatus.WON, BettingStatus.LOST]]
            winning_bets = [bet for bet in settled_bets if bet.betting_status == BettingStatus.WON]
            
            if total_bets == 0:
                session.close()
                return self._get_empty_statistics()
                
            # Financial metrics
            total_staked = sum(bet.stake_amount for bet in betting_records)
            total_returned = sum(bet.payout_amount or 0 for bet in settled_bets)
            net_profit = total_returned - total_staked
            roi = (net_profit / total_staked * 100) if total_staked > 0 else 0
            
            # Performance metrics
            win_rate = (len(winning_bets) / len(settled_bets) * 100) if settled_bets else 0
            avg_odds = sum(bet.odds_taken for bet in betting_records) / total_bets
            avg_stake = total_staked / total_bets
            
            # Confidence breakdown
            confidence_stats = self._calculate_confidence_breakdown(betting_records)
            
            # Model performance
            model_stats = self._calculate_model_performance(betting_records)
            
            session.close()
            
            return {
                'period_days': days,
                'total_bets': total_bets,
                'settled_bets': len(settled_bets),
                'pending_bets': total_bets - len(settled_bets),
                'winning_bets': len(winning_bets),
                'losing_bets': len(settled_bets) - len(winning_bets),
                'win_rate': round(win_rate, 1),
                'total_staked': round(total_staked, 2),
                'total_returned': round(total_returned, 2),
                'net_profit': round(net_profit, 2),
                'roi': round(roi, 1),
                'avg_odds': round(avg_odds, 2),
                'avg_stake': round(avg_stake, 2),
                'current_bankroll': round(self.current_bankroll, 2),
                'confidence_breakdown': confidence_stats,
                'model_performance': model_stats,
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.integration_logger.error(f"❌ Error getting betting statistics: {e}")
            return self._get_empty_statistics()
            
    def _calculate_confidence_breakdown(self, betting_records: List[BettingLog]) -> Dict[str, Any]:
        """Calculate statistics by confidence level"""
        confidence_stats = {}
        
        for confidence in ['High', 'Medium', 'Low']:
            confidence_bets = [bet for bet in betting_records if bet.confidence_level == confidence]
            settled_confidence_bets = [bet for bet in confidence_bets if bet.betting_status in [BettingStatus.WON, BettingStatus.LOST]]
            winning_confidence_bets = [bet for bet in settled_confidence_bets if bet.betting_status == BettingStatus.WON]
            
            if confidence_bets:
                win_rate = (len(winning_confidence_bets) / len(settled_confidence_bets) * 100) if settled_confidence_bets else 0
                avg_edge = sum(bet.edge_percentage for bet in confidence_bets) / len(confidence_bets)
                
                confidence_stats[confidence.lower()] = {
                    'total_bets': len(confidence_bets),
                    'win_rate': round(win_rate, 1),
                    'avg_edge': round(avg_edge, 1)
                }
                
        return confidence_stats
        
    def _calculate_model_performance(self, betting_records: List[BettingLog]) -> Dict[str, Any]:
        """Calculate model-specific performance metrics"""
        model_stats = {}
        
        for model in set(bet.model_used for bet in betting_records if bet.model_used):
            model_bets = [bet for bet in betting_records if bet.model_used == model]
            settled_model_bets = [bet for bet in model_bets if bet.betting_status in [BettingStatus.WON, BettingStatus.LOST]]
            winning_model_bets = [bet for bet in settled_model_bets if bet.betting_status == BettingStatus.WON]
            
            if model_bets:
                win_rate = (len(winning_model_bets) / len(settled_model_bets) * 100) if settled_model_bets else 0
                total_profit = sum(bet.profit_loss or 0 for bet in settled_model_bets)
                total_staked = sum(bet.stake_amount for bet in model_bets)
                roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
                
                model_stats[model] = {
                    'total_bets': len(model_bets),
                    'win_rate': round(win_rate, 1),
                    'roi': round(roi, 1),
                    'profit': round(total_profit, 2)
                }
                
        return model_stats
        
    def _get_empty_statistics(self) -> Dict[str, Any]:
        """Return empty statistics structure"""
        return {
            'period_days': 30,
            'total_bets': 0,
            'settled_bets': 0,
            'pending_bets': 0,
            'winning_bets': 0,
            'losing_bets': 0,
            'win_rate': 0,
            'total_staked': 0,
            'total_returned': 0,
            'net_profit': 0,
            'roi': 0,
            'avg_odds': 0,
            'avg_stake': 0,
            'current_bankroll': self.config.initial_bankroll,
            'confidence_breakdown': {},
            'model_performance': {},
            'last_updated': datetime.utcnow().isoformat()
        }

# Global instance
_prediction_betting_integrator = None

def get_prediction_betting_integrator() -> PredictionBettingIntegrator:
    """Get global prediction betting integrator instance"""
    global _prediction_betting_integrator
    if _prediction_betting_integrator is None:
        _prediction_betting_integrator = PredictionBettingIntegrator()
    return _prediction_betting_integrator

def process_telegram_prediction_as_bet(prediction_result: Dict) -> Optional[str]:
    """Convenience function to process a telegram prediction as a bet"""
    integrator = get_prediction_betting_integrator()
    return integrator.process_telegram_prediction(prediction_result)

def get_betting_dashboard_stats(days: int = 30) -> Dict[str, Any]:
    """Convenience function to get betting statistics for dashboard"""
    integrator = get_prediction_betting_integrator()
    return integrator.get_betting_statistics(days)

if __name__ == "__main__":
    print("🎾 PREDICTION-BETTING INTEGRATION TEST")
    print("=" * 60)
    
    # Test configuration
    config = PredictionBettingConfig(
        default_stake_amount=25.0,
        stake_percentage=2.5,
        initial_bankroll=1000.0
    )
    
    integrator = PredictionBettingIntegrator(config)
    
    # Test sample prediction
    sample_prediction = {
        'success': True,
        'underdog_second_set_probability': 0.62,
        'confidence': 'High',
        'underdog_player': 'player1',
        'match_context': {
            'player1': 'Test Underdog',
            'player2': 'Test Favorite',
            'tournament': 'ATP 250 Test Tournament',
            'surface': 'Hard',
            'match_date': datetime.now().isoformat()
        },
        'strategic_insights': [
            'Strong underdog opportunity detected',
            'Ranking gap creates upset potential'
        ]
    }
    
    print("📊 Testing prediction processing...")
    bet_id = integrator.process_telegram_prediction(sample_prediction)
    
    if bet_id:
        print(f"✅ Successfully created betting record: {bet_id}")
        
        # Get statistics
        stats = integrator.get_betting_statistics()
        print(f"📈 Current Statistics:")
        print(f"  Total Bets: {stats['total_bets']}")
        print(f"  Current Bankroll: ${stats['current_bankroll']:.2f}")
        print(f"  Win Rate: {stats['win_rate']:.1f}%")
        
    else:
        print("❌ Failed to create betting record")
        
    print("✅ Integration test completed!")