#!/usr/bin/env python3
"""
Enhanced Odds and Betting Integration System
Complete implementation blueprint for automated tennis betting with odds integration
"""

import os
import sys
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api_tennis_integration import APITennisClient, TennisMatch
from api_tennis_data_collector import APITennisDataCollector

logger = logging.getLogger(__name__)

class BettingMarket(Enum):
    """Supported betting markets"""
    MATCH_WINNER = "match_winner"
    SET_BETTING = "set_betting"
    HANDICAP = "handicap"
    TOTAL_GAMES = "total_games"
    OVER_UNDER = "over_under"

class BettingSignal(Enum):
    """Betting signal types"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    AVOID = "avoid"

@dataclass
class OddsData:
    """Comprehensive odds data structure"""
    match_id: int
    bookmaker: str
    market_type: BettingMarket
    player1_odds: float
    player2_odds: float
    margin: float
    implied_probability_p1: float
    implied_probability_p2: float
    last_updated: datetime
    data_source: str
    
    def calculate_edge(self, ml_probability: float, player: int) -> float:
        """Calculate betting edge based on ML prediction"""
        if player == 1:
            odds = self.player1_odds
            market_prob = self.implied_probability_p1
        else:
            odds = self.player2_odds
            market_prob = self.implied_probability_p2
        
        # Expected value calculation
        edge = (ml_probability * odds) - 1
        return edge

@dataclass
class BettingOpportunity:
    """Identified betting opportunity"""
    match_id: int
    player1: str
    player2: str
    tournament: str
    
    # ML Prediction data
    ml_confidence: float
    predicted_winner: int
    ml_probability: float
    
    # Odds data
    best_odds: OddsData
    alternative_odds: List[OddsData]
    
    # Betting metrics
    expected_value: float
    kelly_stake_pct: float
    recommended_stake: float
    betting_signal: BettingSignal
    
    # Risk metrics
    max_loss: float
    potential_profit: float
    confidence_score: float

class TennisOddsManager:
    """Comprehensive odds management system"""
    
    def __init__(self, api_tennis_key: str = None):
        """Initialize odds manager with API clients"""
        self.api_tennis = APITennisClient(api_key=api_tennis_key)
        self.data_collector = APITennisDataCollector(api_key=api_tennis_key)
        
        # Configuration
        self.minimum_edge = 0.05  # 5% minimum edge for betting
        self.maximum_stake_pct = 0.05  # 5% max stake per bet
        self.minimum_confidence = 0.65  # Minimum ML confidence
        
        # Odds tracking
        self.odds_history = {}
        self.active_opportunities = []
    
    def get_comprehensive_odds(self, match_id: int) -> List[OddsData]:
        """Get comprehensive odds from all available sources"""
        odds_list = []
        
        try:
            # Get odds from API Tennis
            api_odds = self.api_tennis.get_odds(fixture_id=match_id)
            
            if isinstance(api_odds, dict) and api_odds.get('success') == 1:
                result = api_odds.get('result', [])
                
                for odds_item in result:
                    try:
                        odds_data = self._parse_api_tennis_odds(odds_item, match_id)
                        if odds_data:
                            odds_list.append(odds_data)
                    except Exception as e:
                        logger.warning(f"Failed to parse odds item: {e}")
            
            # Get live odds if available
            live_odds = self.api_tennis.get_live_odds(fixture_id=match_id)
            if isinstance(live_odds, dict) and live_odds.get('success') == 1:
                result = live_odds.get('result', [])
                
                for odds_item in result:
                    try:
                        odds_data = self._parse_api_tennis_odds(odds_item, match_id, is_live=True)
                        if odds_data:
                            odds_list.append(odds_data)
                    except Exception as e:
                        logger.warning(f"Failed to parse live odds: {e}")
        
        except Exception as e:
            logger.error(f"Failed to get comprehensive odds for match {match_id}: {e}")
        
        return odds_list
    
    def _parse_api_tennis_odds(self, odds_item: Dict[str, Any], match_id: int, is_live: bool = False) -> Optional[OddsData]:
        """Parse API Tennis odds response into OddsData format"""
        try:
            # Extract odds values (field names may vary)
            player1_odds = odds_item.get('home_odds') or odds_item.get('player1_odds') or odds_item.get('odds_1')
            player2_odds = odds_item.get('away_odds') or odds_item.get('player2_odds') or odds_item.get('odds_2')
            
            if not player1_odds or not player2_odds:
                return None
            
            # Convert to float
            player1_odds = float(player1_odds)
            player2_odds = float(player2_odds)
            
            # Calculate implied probabilities
            implied_prob_p1 = 1 / player1_odds
            implied_prob_p2 = 1 / player2_odds
            
            # Calculate bookmaker margin
            margin = (implied_prob_p1 + implied_prob_p2 - 1) * 100
            
            # Normalize probabilities (remove margin)
            total_prob = implied_prob_p1 + implied_prob_p2
            implied_prob_p1_norm = implied_prob_p1 / total_prob
            implied_prob_p2_norm = implied_prob_p2 / total_prob
            
            return OddsData(
                match_id=match_id,
                bookmaker=odds_item.get('bookmaker', 'Unknown'),
                market_type=BettingMarket.MATCH_WINNER,
                player1_odds=player1_odds,
                player2_odds=player2_odds,
                margin=margin,
                implied_probability_p1=implied_prob_p1_norm,
                implied_probability_p2=implied_prob_p2_norm,
                last_updated=datetime.now(),
                data_source=f"API-Tennis{'(Live)' if is_live else ''}"
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse odds item: {e}")
            return None
    
    def find_best_odds(self, odds_list: List[OddsData]) -> Tuple[OddsData, OddsData]:
        """Find best odds for each player"""
        if not odds_list:
            raise ValueError("No odds available")
        
        best_p1_odds = max(odds_list, key=lambda x: x.player1_odds)
        best_p2_odds = max(odds_list, key=lambda x: x.player2_odds)
        
        return best_p1_odds, best_p2_odds
    
    def calculate_kelly_stake(self, edge: float, odds: float, bankroll: float) -> float:
        """Calculate optimal stake using Kelly Criterion"""
        if edge <= 0:
            return 0
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds - 1, p = probability of winning, q = probability of losing
        b = odds - 1
        p = edge / (odds - 1) + (1 / odds)  # Implied probability adjusted for edge
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Apply safety constraints
        kelly_fraction = max(0, min(kelly_fraction, self.maximum_stake_pct))
        
        return kelly_fraction * bankroll

class AutomatedBettingEngine:
    """Complete automated betting engine with odds integration"""
    
    def __init__(self, api_tennis_key: str = None, initial_bankroll: float = 1000.0):
        """Initialize automated betting engine"""
        self.odds_manager = TennisOddsManager(api_tennis_key)
        self.data_collector = APITennisDataCollector(api_tennis_key)
        
        # Bankroll management
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.total_staked = 0.0
        self.total_profit = 0.0
        
        # Risk management
        self.max_stake_per_bet = initial_bankroll * 0.05  # 5% max
        self.max_daily_stake = initial_bankroll * 0.20    # 20% daily max
        self.daily_stake_used = 0.0
        self.last_reset_date = datetime.now().date()
        
        # Performance tracking
        self.betting_history = []
        self.active_bets = []
    
    def get_ml_prediction(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get ML prediction for a match (placeholder - integrate with your ML system)"""
        # This would integrate with your existing ML prediction system
        # For now, return a sample prediction structure
        return {
            'confidence': 0.72,
            'predicted_winner': 1,  # Player 1
            'probability_p1': 0.72,
            'probability_p2': 0.28,
            'model_ensemble_agreement': 0.85
        }
    
    def identify_betting_opportunities(self, matches: List[Dict[str, Any]]) -> List[BettingOpportunity]:
        """Identify profitable betting opportunities from match list"""
        opportunities = []
        
        for match in matches:
            try:
                # Get match ID (convert from universal format)
                match_id_str = match.get('id', '').replace('api_tennis_', '')
                if not match_id_str.isdigit():
                    continue
                
                match_id = int(match_id_str)
                
                # Get ML prediction
                ml_prediction = self.get_ml_prediction(match)
                
                # Check minimum confidence threshold
                if ml_prediction['confidence'] < self.odds_manager.minimum_confidence:
                    continue
                
                # Get comprehensive odds
                odds_list = self.odds_manager.get_comprehensive_odds(match_id)
                
                if not odds_list:
                    logger.debug(f"No odds available for match {match_id}")
                    continue
                
                # Find best odds for predicted winner
                predicted_winner = ml_prediction['predicted_winner']
                ml_probability = ml_prediction[f'probability_p{predicted_winner}']
                
                # Calculate edges for all available odds
                best_opportunity = None
                best_edge = -1
                
                for odds_data in odds_list:
                    edge = odds_data.calculate_edge(ml_probability, predicted_winner)
                    
                    if edge > best_edge and edge > self.odds_manager.minimum_edge:
                        best_edge = edge
                        best_opportunity = odds_data
                
                if best_opportunity:
                    # Calculate betting metrics
                    kelly_stake_pct = self.odds_manager.calculate_kelly_stake(
                        best_edge, 
                        best_opportunity.player1_odds if predicted_winner == 1 else best_opportunity.player2_odds,
                        self.current_bankroll
                    ) / self.current_bankroll
                    
                    recommended_stake = min(
                        kelly_stake_pct * self.current_bankroll,
                        self.max_stake_per_bet,
                        self.max_daily_stake - self.daily_stake_used
                    )
                    
                    if recommended_stake > 0:
                        # Determine betting signal strength
                        if best_edge > 0.15:  # 15%+ edge
                            signal = BettingSignal.STRONG_BUY
                        elif best_edge > 0.08:  # 8%+ edge
                            signal = BettingSignal.BUY
                        else:
                            signal = BettingSignal.HOLD
                        
                        opportunity = BettingOpportunity(
                            match_id=match_id,
                            player1=match['player1'],
                            player2=match['player2'],
                            tournament=match.get('tournament', ''),
                            ml_confidence=ml_prediction['confidence'],
                            predicted_winner=predicted_winner,
                            ml_probability=ml_probability,
                            best_odds=best_opportunity,
                            alternative_odds=[o for o in odds_list if o != best_opportunity],
                            expected_value=best_edge,
                            kelly_stake_pct=kelly_stake_pct,
                            recommended_stake=recommended_stake,
                            betting_signal=signal,
                            max_loss=recommended_stake,
                            potential_profit=recommended_stake * (best_opportunity.player1_odds if predicted_winner == 1 else best_opportunity.player2_odds) - recommended_stake,
                            confidence_score=ml_prediction['confidence'] * ml_prediction.get('model_ensemble_agreement', 1.0)
                        )
                        
                        opportunities.append(opportunity)
                        
            except Exception as e:
                logger.error(f"Failed to analyze match for opportunities: {e}")
                continue
        
        # Sort by expected value and confidence
        opportunities.sort(key=lambda x: (x.expected_value * x.confidence_score), reverse=True)
        
        return opportunities
    
    def execute_betting_strategy(self, max_bets_per_day: int = 5) -> List[BettingOpportunity]:
        """Execute automated betting strategy"""
        # Reset daily limits if needed
        if datetime.now().date() > self.last_reset_date:
            self.daily_stake_used = 0.0
            self.last_reset_date = datetime.now().date()
        
        # Get current matches
        current_matches = self.data_collector.get_current_matches()
        upcoming_matches = self.data_collector.get_upcoming_matches(days_ahead=1)
        
        all_matches = current_matches + upcoming_matches
        
        logger.info(f"Analyzing {len(all_matches)} matches for betting opportunities")
        
        # Identify opportunities
        opportunities = self.identify_betting_opportunities(all_matches)
        
        logger.info(f"Found {len(opportunities)} potential betting opportunities")
        
        # Execute top opportunities
        executed_bets = []
        bets_placed = 0
        
        for opportunity in opportunities:
            if bets_placed >= max_bets_per_day:
                break
            
            if opportunity.betting_signal in [BettingSignal.STRONG_BUY, BettingSignal.BUY]:
                if self._should_place_bet(opportunity):
                    executed_bet = self._place_bet(opportunity)
                    if executed_bet:
                        executed_bets.append(executed_bet)
                        bets_placed += 1
                        self.daily_stake_used += opportunity.recommended_stake
        
        return executed_bets
    
    def _should_place_bet(self, opportunity: BettingOpportunity) -> bool:
        """Determine if a bet should be placed based on risk management rules"""
        # Check bankroll constraints
        if opportunity.recommended_stake > self.current_bankroll * 0.05:
            return False
        
        # Check daily limits
        if self.daily_stake_used + opportunity.recommended_stake > self.max_daily_stake:
            return False
        
        # Check minimum edge and confidence
        if opportunity.expected_value < self.odds_manager.minimum_edge:
            return False
        
        if opportunity.confidence_score < 0.60:
            return False
        
        return True
    
    def _place_bet(self, opportunity: BettingOpportunity) -> Optional[BettingOpportunity]:
        """Place actual bet (integrate with betting API)"""
        try:
            logger.info(f"PLACING BET: {opportunity.player1} vs {opportunity.player2}")
            logger.info(f"  Predicted Winner: Player {opportunity.predicted_winner}")
            logger.info(f"  Stake: ${opportunity.recommended_stake:.2f}")
            logger.info(f"  Odds: {opportunity.best_odds.player1_odds if opportunity.predicted_winner == 1 else opportunity.best_odds.player2_odds}")
            logger.info(f"  Expected Value: {opportunity.expected_value:.1%}")
            logger.info(f"  Bookmaker: {opportunity.best_odds.bookmaker}")
            
            # Here you would integrate with actual betting API (Betfair, etc.)
            # For now, we'll simulate the bet placement
            
            # Update bankroll tracking
            self.current_bankroll -= opportunity.recommended_stake
            self.total_staked += opportunity.recommended_stake
            
            # Add to active bets
            self.active_bets.append(opportunity)
            
            # Add to betting history
            bet_record = {
                'timestamp': datetime.now(),
                'match_id': opportunity.match_id,
                'match': f"{opportunity.player1} vs {opportunity.player2}",
                'predicted_winner': opportunity.predicted_winner,
                'stake': opportunity.recommended_stake,
                'odds': opportunity.best_odds.player1_odds if opportunity.predicted_winner == 1 else opportunity.best_odds.player2_odds,
                'expected_value': opportunity.expected_value,
                'confidence': opportunity.confidence_score,
                'bookmaker': opportunity.best_odds.bookmaker,
                'status': 'placed'
            }
            
            self.betting_history.append(bet_record)
            
            return opportunity
            
        except Exception as e:
            logger.error(f"Failed to place bet: {e}")
            return None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get betting performance summary"""
        total_bets = len(self.betting_history)
        
        if total_bets == 0:
            return {
                'total_bets': 0,
                'total_staked': 0.0,
                'total_profit': 0.0,
                'roi': 0.0,
                'win_rate': 0.0,
                'average_odds': 0.0,
                'current_bankroll': self.current_bankroll
            }
        
        # Calculate metrics
        total_staked = sum(bet['stake'] for bet in self.betting_history)
        total_profit = self.total_profit
        roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0
        
        settled_bets = [bet for bet in self.betting_history if bet.get('status') in ['won', 'lost']]
        win_rate = (len([bet for bet in settled_bets if bet.get('status') == 'won']) / len(settled_bets)) * 100 if settled_bets else 0
        
        average_odds = sum(bet['odds'] for bet in self.betting_history) / total_bets
        
        return {
            'total_bets': total_bets,
            'total_staked': total_staked,
            'total_profit': total_profit,
            'roi': roi,
            'win_rate': win_rate,
            'average_odds': average_odds,
            'current_bankroll': self.current_bankroll,
            'bankroll_change': ((self.current_bankroll / self.initial_bankroll) - 1) * 100
        }

def main():
    """Example usage of the enhanced betting system"""
    print("🎾 ENHANCED ODDS AND BETTING INTEGRATION SYSTEM")
    print("=" * 60)
    
    # Check configuration
    api_key = os.getenv('API_TENNIS_KEY')
    if not api_key:
        print("❌ API_TENNIS_KEY not configured")
        print("This is a simulation showing the system architecture")
        print("Configure API key to test with real data")
        api_key = "demo_key"
    
    try:
        # Initialize betting engine
        betting_engine = AutomatedBettingEngine(
            api_tennis_key=api_key,
            initial_bankroll=1000.0
        )
        
        print(f"✅ Betting engine initialized")
        print(f"   Initial bankroll: ${betting_engine.initial_bankroll}")
        print(f"   Max stake per bet: ${betting_engine.max_stake_per_bet}")
        print(f"   Max daily stake: ${betting_engine.max_daily_stake}")
        
        # Simulate betting strategy execution
        print(f"\n🎯 Executing automated betting strategy...")
        
        if api_key != "demo_key":
            executed_bets = betting_engine.execute_betting_strategy(max_bets_per_day=3)
            
            print(f"✅ Executed {len(executed_bets)} bets")
            
            for i, bet in enumerate(executed_bets, 1):
                print(f"  {i}. {bet.player1} vs {bet.player2}")
                print(f"     Stake: ${bet.recommended_stake:.2f} | Edge: {bet.expected_value:.1%}")
        else:
            print("  ⚠️ Demo mode - no real bets placed")
        
        # Show performance summary
        print(f"\n📊 Performance Summary:")
        performance = betting_engine.get_performance_summary()
        
        for metric, value in performance.items():
            if isinstance(value, float):
                print(f"   {metric}: {value:.2f}")
            else:
                print(f"   {metric}: {value}")
        
        print(f"\n🎯 System Features Demonstrated:")
        print("   ✅ Comprehensive odds collection from API Tennis")
        print("   ✅ ML prediction integration for value detection")
        print("   ✅ Kelly Criterion stake sizing")
        print("   ✅ Risk management and bankroll protection")
        print("   ✅ Automated opportunity identification")
        print("   ✅ Performance tracking and analysis")
        
        print(f"\n🚀 Next Steps for Production:")
        print("   1. Integrate with real betting API (Betfair Exchange)")
        print("   2. Connect with your ML prediction system")
        print("   3. Add real-time odds monitoring")
        print("   4. Implement bet settlement tracking")
        print("   5. Add advanced risk management rules")
        
    except Exception as e:
        print(f"❌ Error in betting system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()