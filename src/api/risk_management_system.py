#!/usr/bin/env python3
"""
Comprehensive Risk Management & Position Sizing System
Advanced risk controls and position sizing for automated tennis betting operations
"""

import logging
import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
from collections import defaultdict, deque
import threading
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk management levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class PositionType(Enum):
    """Types of betting positions"""
    BACK = "back"
    LAY = "lay"
    HEDGE = "hedge"
    ARBITRAGE = "arbitrage"


@dataclass
class RiskLimits:
    """Comprehensive risk limit configuration"""
    # Stake limits
    max_stake_per_bet: float = 100.0
    max_stake_per_match: float = 500.0
    max_stake_per_player: float = 1000.0
    max_stake_per_tournament: float = 2000.0
    
    # Time-based limits
    max_daily_stake: float = 2000.0
    max_weekly_stake: float = 10000.0
    max_monthly_stake: float = 40000.0
    
    # Loss limits
    max_daily_loss: float = 500.0
    max_weekly_loss: float = 2000.0
    max_monthly_loss: float = 8000.0
    max_drawdown_percentage: float = 20.0
    
    # Position limits
    max_concurrent_bets: int = 20
    max_exposure_percentage: float = 25.0  # % of bankroll
    max_correlation_exposure: float = 1000.0
    
    # Quality thresholds
    min_confidence_threshold: float = 0.65
    min_edge_threshold: float = 0.03
    min_odds_threshold: float = 1.2
    max_odds_threshold: float = 10.0
    
    # Stop-loss and take-profit
    stop_loss_percentage: float = 15.0
    take_profit_percentage: float = 25.0
    trailing_stop_percentage: float = 10.0
    
    # Market conditions
    max_volatility_threshold: float = 0.3
    min_liquidity_threshold: float = 1000.0
    blacklist_periods: List[str] = field(default_factory=list)


@dataclass
class Position:
    """Individual betting position"""
    position_id: str
    match_id: str
    market_id: str
    selection_id: str
    position_type: PositionType
    stake: float
    odds: float
    potential_profit: float
    potential_loss: float
    confidence: float
    edge: float
    created_at: datetime
    status: str = "open"
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    risk_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['position_type'] = self.position_type.value
        return data


@dataclass
class PortfolioMetrics:
    """Portfolio performance and risk metrics"""
    total_positions: int = 0
    total_exposure: float = 0.0
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    expected_shortfall: float = 0.0
    correlation_risk: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class PositionSizer(ABC):
    """Abstract base class for position sizing strategies"""
    
    @abstractmethod
    def calculate_stake(self, prediction: Dict[str, Any], market_data: Dict[str, Any], 
                       portfolio: 'Portfolio', risk_limits: RiskLimits) -> float:
        pass


class KellyCriterionSizer(PositionSizer):
    """Kelly Criterion position sizing"""
    
    def __init__(self, kelly_fraction: float = 0.25):
        self.kelly_fraction = kelly_fraction  # Use fractional Kelly for safety
    
    def calculate_stake(self, prediction: Dict[str, Any], market_data: Dict[str, Any], 
                       portfolio: 'Portfolio', risk_limits: RiskLimits) -> float:
        try:
            # Extract probability and odds
            probability = prediction.get('probability', 0.5)
            odds = market_data.get('odds', 2.0)
            
            # Kelly formula: f = (bp - q) / b
            # where b = odds - 1, p = probability, q = 1 - p
            b = odds - 1
            p = probability
            q = 1 - p
            
            if b <= 0 or p <= 0:
                return 0.0
            
            # Calculate Kelly fraction
            kelly_fraction = (b * p - q) / b
            
            # Apply fractional Kelly
            adjusted_fraction = kelly_fraction * self.kelly_fraction
            
            # Calculate stake based on bankroll
            bankroll = portfolio.get_available_bankroll()
            stake = bankroll * adjusted_fraction
            
            # Apply limits
            stake = max(0, min(stake, risk_limits.max_stake_per_bet))
            
            logger.debug(f"Kelly stake calculated: {stake:.2f} (fraction: {adjusted_fraction:.3f})")
            return stake
            
        except Exception as e:
            logger.error(f"Kelly calculation failed: {e}")
            return 0.0


class FixedPercentageSizer(PositionSizer):
    """Fixed percentage position sizing"""
    
    def __init__(self, percentage: float = 2.0):
        self.percentage = percentage  # % of bankroll per bet
    
    def calculate_stake(self, prediction: Dict[str, Any], market_data: Dict[str, Any], 
                       portfolio: 'Portfolio', risk_limits: RiskLimits) -> float:
        try:
            bankroll = portfolio.get_available_bankroll()
            stake = bankroll * (self.percentage / 100)
            
            # Apply limits
            stake = max(0, min(stake, risk_limits.max_stake_per_bet))
            
            return stake
            
        except Exception as e:
            logger.error(f"Fixed percentage calculation failed: {e}")
            return 0.0


class ConfidenceWeightedSizer(PositionSizer):
    """Confidence-weighted position sizing"""
    
    def __init__(self, base_percentage: float = 2.0, max_multiplier: float = 3.0):
        self.base_percentage = base_percentage
        self.max_multiplier = max_multiplier
    
    def calculate_stake(self, prediction: Dict[str, Any], market_data: Dict[str, Any], 
                       portfolio: 'Portfolio', risk_limits: RiskLimits) -> float:
        try:
            confidence = prediction.get('confidence', 0.5)
            edge = prediction.get('edge', 0.0)
            
            # Scale stake based on confidence and edge
            confidence_multiplier = min(confidence * 2, self.max_multiplier)
            edge_multiplier = min(1 + edge * 10, self.max_multiplier)
            
            total_multiplier = (confidence_multiplier + edge_multiplier) / 2
            
            bankroll = portfolio.get_available_bankroll()
            base_stake = bankroll * (self.base_percentage / 100)
            stake = base_stake * total_multiplier
            
            # Apply limits
            stake = max(0, min(stake, risk_limits.max_stake_per_bet))
            
            return stake
            
        except Exception as e:
            logger.error(f"Confidence weighted calculation failed: {e}")
            return 0.0


class Portfolio:
    """Portfolio management for tracking positions and performance"""
    
    def __init__(self, initial_bankroll: float):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.daily_pnl_history: deque = deque(maxlen=365)  # 1 year history
        self.metrics = PortfolioMetrics()
        self.lock = threading.Lock()
    
    def add_position(self, position: Position) -> bool:
        """Add new position to portfolio"""
        with self.lock:
            try:
                self.positions[position.position_id] = position
                self.current_bankroll -= position.stake
                self._update_metrics()
                
                logger.info(f"Position added: {position.position_id} - Stake: {position.stake}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to add position: {e}")
                return False
    
    def close_position(self, position_id: str, realized_pnl: float) -> bool:
        """Close position and update P&L"""
        with self.lock:
            try:
                if position_id not in self.positions:
                    return False
                
                position = self.positions[position_id]
                position.status = "closed"
                position.realized_pnl = realized_pnl
                
                # Update bankroll
                self.current_bankroll += position.stake + realized_pnl
                
                # Move to closed positions
                self.closed_positions.append(position)
                del self.positions[position_id]
                
                self._update_metrics()
                
                logger.info(f"Position closed: {position_id} - P&L: {realized_pnl}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to close position: {e}")
                return False
    
    def update_position_pnl(self, position_id: str, unrealized_pnl: float):
        """Update unrealized P&L for open position"""
        with self.lock:
            if position_id in self.positions:
                self.positions[position_id].unrealized_pnl = unrealized_pnl
                self._update_metrics()
    
    def get_available_bankroll(self) -> float:
        """Get available bankroll for new positions"""
        return max(0, self.current_bankroll)
    
    def get_total_exposure(self) -> float:
        """Get total exposure across all positions"""
        return sum(pos.stake for pos in self.positions.values())
    
    def get_unrealized_pnl(self) -> float:
        """Get total unrealized P&L"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    def get_realized_pnl(self) -> float:
        """Get total realized P&L"""
        return sum(pos.realized_pnl for pos in self.closed_positions)
    
    def get_position_by_match(self, match_id: str) -> List[Position]:
        """Get all positions for specific match"""
        return [pos for pos in self.positions.values() if pos.match_id == match_id]
    
    def _update_metrics(self):
        """Update portfolio metrics"""
        try:
            # Basic metrics
            self.metrics.total_positions = len(self.positions) + len(self.closed_positions)
            self.metrics.total_exposure = self.get_total_exposure()
            self.metrics.total_realized_pnl = self.get_realized_pnl()
            self.metrics.total_unrealized_pnl = self.get_unrealized_pnl()
            
            # Performance metrics
            if self.closed_positions:
                winning_positions = [p for p in self.closed_positions if p.realized_pnl > 0]
                self.metrics.win_rate = len(winning_positions) / len(self.closed_positions)
                
                total_wins = sum(p.realized_pnl for p in winning_positions)
                total_losses = abs(sum(p.realized_pnl for p in self.closed_positions if p.realized_pnl < 0))
                
                if total_losses > 0:
                    self.metrics.profit_factor = total_wins / total_losses
            
            # Drawdown calculation
            total_pnl = self.metrics.total_realized_pnl + self.metrics.total_unrealized_pnl
            peak_value = self.initial_bankroll + max(0, max([0] + [sum(p.realized_pnl for p in self.closed_positions[:i+1]) 
                                                                 for i in range(len(self.closed_positions))]))
            current_value = self.initial_bankroll + total_pnl
            
            if peak_value > 0:
                self.metrics.current_drawdown = max(0, (peak_value - current_value) / peak_value)
                self.metrics.max_drawdown = max(self.metrics.max_drawdown, self.metrics.current_drawdown)
            
            self.metrics.timestamp = datetime.now()
            
        except Exception as e:
            logger.error(f"Metrics update failed: {e}")


class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, risk_limits: RiskLimits, portfolio: Portfolio, 
                 position_sizer: PositionSizer = None):
        self.risk_limits = risk_limits
        self.portfolio = portfolio
        self.position_sizer = position_sizer or KellyCriterionSizer()
        
        # Risk tracking
        self.daily_stakes = defaultdict(float)
        self.weekly_stakes = defaultdict(float)
        self.monthly_stakes = defaultdict(float)
        self.daily_losses = defaultdict(float)
        self.correlation_matrix = {}
        self.volatility_cache = {}
        
        # Alert system
        self.alerts: List[Dict[str, Any]] = []
        self.alert_callbacks: List[callable] = []
        
        # Performance tracking
        self.risk_decisions = {
            'approved': 0,
            'rejected': 0,
            'reasons': defaultdict(int)
        }
        
        self.lock = threading.Lock()
    
    def evaluate_bet_request(self, prediction: Dict[str, Any], market_data: Dict[str, Any], 
                           match_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive bet evaluation with risk checks
        
        Returns:
        {
            'approved': bool,
            'stake': float,
            'reason': str,
            'risk_score': float,
            'warnings': List[str]
        }
        """
        with self.lock:
            try:
                result = {
                    'approved': False,
                    'stake': 0.0,
                    'reason': '',
                    'risk_score': 0.0,
                    'warnings': []
                }
                
                # Basic validation
                validation_result = self._validate_basic_requirements(prediction, market_data)
                if not validation_result['valid']:
                    result['reason'] = validation_result['reason']
                    self.risk_decisions['rejected'] += 1
                    self.risk_decisions['reasons'][validation_result['reason']] += 1
                    return result
                
                # Calculate position size
                stake = self.position_sizer.calculate_stake(
                    prediction, market_data, self.portfolio, self.risk_limits
                )
                
                if stake < 10:  # Minimum bet size
                    result['reason'] = 'Calculated stake below minimum'
                    self.risk_decisions['rejected'] += 1
                    self.risk_decisions['reasons']['stake_too_small'] += 1
                    return result
                
                # Risk limit checks
                risk_check_result = self._check_risk_limits(stake, match_info)
                if not risk_check_result['approved']:
                    result['reason'] = risk_check_result['reason']
                    result['warnings'] = risk_check_result.get('warnings', [])
                    self.risk_decisions['rejected'] += 1
                    self.risk_decisions['reasons'][risk_check_result['reason']] += 1
                    return result
                
                # Market condition checks
                market_check_result = self._check_market_conditions(market_data)
                if not market_check_result['approved']:
                    result['reason'] = market_check_result['reason']
                    result['warnings'].extend(market_check_result.get('warnings', []))
                    self.risk_decisions['rejected'] += 1
                    self.risk_decisions['reasons'][market_check_result['reason']] += 1
                    return result
                
                # Portfolio correlation check
                correlation_check = self._check_correlation_risk(match_info, stake)
                if not correlation_check['approved']:
                    result['reason'] = correlation_check['reason']
                    result['warnings'].extend(correlation_check.get('warnings', []))
                    self.risk_decisions['rejected'] += 1
                    self.risk_decisions['reasons']['correlation_risk'] += 1
                    return result
                
                # Calculate risk score
                risk_score = self._calculate_risk_score(prediction, market_data, stake)
                
                # Final approval
                result.update({
                    'approved': True,
                    'stake': stake,
                    'reason': 'Risk checks passed',
                    'risk_score': risk_score,
                    'warnings': risk_check_result.get('warnings', []) + 
                               market_check_result.get('warnings', []) +
                               correlation_check.get('warnings', [])
                })
                
                self.risk_decisions['approved'] += 1
                
                # Update tracking
                self._update_stake_tracking(stake)
                
                return result
                
            except Exception as e:
                logger.error(f"Bet evaluation failed: {e}")
                self.risk_decisions['rejected'] += 1
                self.risk_decisions['reasons']['system_error'] += 1
                return {
                    'approved': False,
                    'stake': 0.0,
                    'reason': f'System error: {str(e)}',
                    'risk_score': 0.0,
                    'warnings': []
                }
    
    def _validate_basic_requirements(self, prediction: Dict[str, Any], 
                                   market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate basic bet requirements"""
        
        # Confidence check
        confidence = prediction.get('confidence', 0)
        if confidence < self.risk_limits.min_confidence_threshold:
            return {
                'valid': False,
                'reason': f'Confidence {confidence:.2f} below threshold {self.risk_limits.min_confidence_threshold}'
            }
        
        # Edge check
        edge = prediction.get('edge', 0)
        if edge < self.risk_limits.min_edge_threshold:
            return {
                'valid': False,
                'reason': f'Edge {edge:.3f} below threshold {self.risk_limits.min_edge_threshold}'
            }
        
        # Odds check
        odds = market_data.get('odds', 0)
        if odds < self.risk_limits.min_odds_threshold or odds > self.risk_limits.max_odds_threshold:
            return {
                'valid': False,
                'reason': f'Odds {odds} outside acceptable range [{self.risk_limits.min_odds_threshold}, {self.risk_limits.max_odds_threshold}]'
            }
        
        return {'valid': True}
    
    def _check_risk_limits(self, stake: float, match_info: Dict[str, Any]) -> Dict[str, Any]:
        """Check all risk limits"""
        warnings = []
        
        # Stake limits
        if stake > self.risk_limits.max_stake_per_bet:
            return {
                'approved': False,
                'reason': f'Stake {stake} exceeds max per bet {self.risk_limits.max_stake_per_bet}'
            }
        
        # Match exposure
        match_id = match_info.get('match_id')
        current_match_exposure = sum(p.stake for p in self.portfolio.get_position_by_match(match_id))
        if current_match_exposure + stake > self.risk_limits.max_stake_per_match:
            return {
                'approved': False,
                'reason': f'Match exposure would exceed limit {self.risk_limits.max_stake_per_match}'
            }
        
        # Concurrent positions
        if len(self.portfolio.positions) >= self.risk_limits.max_concurrent_bets:
            return {
                'approved': False,
                'reason': f'Maximum concurrent positions reached {self.risk_limits.max_concurrent_bets}'
            }
        
        # Portfolio exposure
        total_exposure = self.portfolio.get_total_exposure() + stake
        max_exposure = self.portfolio.current_bankroll * (self.risk_limits.max_exposure_percentage / 100)
        if total_exposure > max_exposure:
            return {
                'approved': False,
                'reason': f'Portfolio exposure would exceed {self.risk_limits.max_exposure_percentage}%'
            }
        
        # Daily limits
        today = datetime.now().date()
        daily_stake = self.daily_stakes[today] + stake
        if daily_stake > self.risk_limits.max_daily_stake:
            return {
                'approved': False,
                'reason': f'Daily stake limit exceeded {self.risk_limits.max_daily_stake}'
            }
        
        # Drawdown check
        if self.portfolio.metrics.current_drawdown > (self.risk_limits.max_drawdown_percentage / 100):
            return {
                'approved': False,
                'reason': f'Maximum drawdown exceeded {self.risk_limits.max_drawdown_percentage}%'
            }
        
        # Warnings for approaching limits
        if daily_stake > self.risk_limits.max_daily_stake * 0.8:
            warnings.append(f'Approaching daily stake limit: {daily_stake:.0f}/{self.risk_limits.max_daily_stake}')
        
        if total_exposure > max_exposure * 0.8:
            warnings.append(f'Approaching exposure limit: {total_exposure:.0f}/{max_exposure:.0f}')
        
        return {
            'approved': True,
            'warnings': warnings
        }
    
    def _check_market_conditions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check market conditions and liquidity"""
        warnings = []
        
        # Liquidity check
        liquidity = market_data.get('liquidity', 0)
        if liquidity < self.risk_limits.min_liquidity_threshold:
            return {
                'approved': False,
                'reason': f'Insufficient liquidity: {liquidity} < {self.risk_limits.min_liquidity_threshold}'
            }
        
        # Volatility check
        volatility = market_data.get('volatility', 0)
        if volatility > self.risk_limits.max_volatility_threshold:
            warnings.append(f'High volatility detected: {volatility:.3f}')
        
        # Spread check
        spread = market_data.get('spread', 0)
        if spread > 0.1:  # 10% spread
            warnings.append(f'Wide spread detected: {spread:.3f}')
        
        return {
            'approved': True,
            'warnings': warnings
        }
    
    def _check_correlation_risk(self, match_info: Dict[str, Any], stake: float) -> Dict[str, Any]:
        """Check correlation risk with existing positions"""
        warnings = []
        
        # Get players involved
        player1 = match_info.get('player1')
        player2 = match_info.get('player2')
        tournament = match_info.get('tournament')
        surface = match_info.get('surface')
        
        # Calculate correlation exposure
        correlation_exposure = 0
        
        for position in self.portfolio.positions.values():
            # Same players
            if (position.match_id.endswith(player1) or position.match_id.endswith(player2)):
                correlation_exposure += position.stake
            
            # Same tournament
            if tournament and tournament in position.match_id:
                correlation_exposure += position.stake * 0.5  # Partial correlation
        
        total_correlation_exposure = correlation_exposure + stake
        
        if total_correlation_exposure > self.risk_limits.max_correlation_exposure:
            return {
                'approved': False,
                'reason': f'Correlation exposure limit exceeded: {total_correlation_exposure}'
            }
        
        if total_correlation_exposure > self.risk_limits.max_correlation_exposure * 0.8:
            warnings.append(f'High correlation exposure: {total_correlation_exposure:.0f}')
        
        return {
            'approved': True,
            'warnings': warnings
        }
    
    def _calculate_risk_score(self, prediction: Dict[str, Any], market_data: Dict[str, Any], 
                             stake: float) -> float:
        """Calculate overall risk score for the bet"""
        try:
            # Base risk from odds
            odds = market_data.get('odds', 2.0)
            odds_risk = min(1.0, 1.0 / odds)  # Higher odds = higher risk
            
            # Confidence risk
            confidence = prediction.get('confidence', 0.5)
            confidence_risk = 1.0 - confidence
            
            # Stake risk (relative to bankroll)
            stake_risk = stake / self.portfolio.current_bankroll
            
            # Portfolio concentration risk
            concentration_risk = self.portfolio.get_total_exposure() / self.portfolio.current_bankroll
            
            # Combine risk factors
            risk_score = (
                odds_risk * 0.3 +
                confidence_risk * 0.3 +
                stake_risk * 0.2 +
                concentration_risk * 0.2
            )
            
            return min(1.0, max(0.0, risk_score))
            
        except Exception as e:
            logger.error(f"Risk score calculation failed: {e}")
            return 0.5  # Default medium risk
    
    def _update_stake_tracking(self, stake: float):
        """Update stake tracking for limits"""
        today = datetime.now().date()
        week = today.isocalendar()[:2]  # Year, week
        month = (today.year, today.month)
        
        self.daily_stakes[today] += stake
        self.weekly_stakes[week] += stake
        self.monthly_stakes[month] += stake
    
    def monitor_positions(self) -> Dict[str, Any]:
        """Monitor all open positions for risk management"""
        try:
            alerts = []
            actions = []
            
            current_time = datetime.now()
            
            for position in self.portfolio.positions.values():
                # Check position age
                age = current_time - position.created_at
                if age > timedelta(hours=6):  # 6 hour max position time
                    alerts.append({
                        'level': AlertLevel.WARNING,
                        'message': f'Long-running position: {position.position_id}',
                        'position_id': position.position_id,
                        'age_hours': age.total_seconds() / 3600
                    })
                
                # Check unrealized P&L
                if position.unrealized_pnl < -position.stake * (self.risk_limits.stop_loss_percentage / 100):
                    alerts.append({
                        'level': AlertLevel.CRITICAL,
                        'message': f'Stop-loss triggered: {position.position_id}',
                        'position_id': position.position_id,
                        'unrealized_pnl': position.unrealized_pnl
                    })
                    actions.append({
                        'action': 'close_position',
                        'position_id': position.position_id,
                        'reason': 'stop_loss'
                    })
                
                # Check take-profit
                elif position.unrealized_pnl > position.stake * (self.risk_limits.take_profit_percentage / 100):
                    alerts.append({
                        'level': AlertLevel.INFO,
                        'message': f'Take-profit target reached: {position.position_id}',
                        'position_id': position.position_id,
                        'unrealized_pnl': position.unrealized_pnl
                    })
                    actions.append({
                        'action': 'close_position',
                        'position_id': position.position_id,
                        'reason': 'take_profit'
                    })
            
            # Check portfolio-level risks
            total_drawdown = self.portfolio.metrics.current_drawdown
            if total_drawdown > (self.risk_limits.max_drawdown_percentage / 100) * 0.8:
                alerts.append({
                    'level': AlertLevel.WARNING,
                    'message': f'Approaching maximum drawdown: {total_drawdown:.1%}',
                    'current_drawdown': total_drawdown,
                    'max_drawdown': self.risk_limits.max_drawdown_percentage / 100
                })
            
            # Process alerts
            for alert in alerts:
                self._process_alert(alert)
            
            return {
                'alerts_generated': len(alerts),
                'actions_recommended': len(actions),
                'alerts': alerts,
                'actions': actions,
                'portfolio_health': self._assess_portfolio_health()
            }
            
        except Exception as e:
            logger.error(f"Position monitoring failed: {e}")
            return {'error': str(e)}
    
    def _process_alert(self, alert: Dict[str, Any]):
        """Process and store alert"""
        alert['timestamp'] = datetime.now().isoformat()
        alert['level'] = alert['level'].value if isinstance(alert['level'], AlertLevel) else alert['level']
        
        self.alerts.append(alert)
        
        # Keep only last 1000 alerts
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        # Log critical alerts
        if alert.get('level') in ['CRITICAL', 'EMERGENCY']:
            logger.critical(f"RISK ALERT: {alert['message']}")
        elif alert.get('level') == 'WARNING':
            logger.warning(f"Risk warning: {alert['message']}")
    
    def _assess_portfolio_health(self) -> Dict[str, Any]:
        """Assess overall portfolio health"""
        try:
            metrics = self.portfolio.metrics
            
            # Health score calculation
            health_factors = []
            
            # Drawdown factor
            drawdown_factor = 1.0 - min(1.0, metrics.current_drawdown / 0.2)  # Normalize to 20% max
            health_factors.append(('drawdown', drawdown_factor, 0.3))
            
            # Win rate factor
            win_rate_factor = metrics.win_rate if metrics.win_rate <= 1.0 else 0.5
            health_factors.append(('win_rate', win_rate_factor, 0.2))
            
            # Profit factor
            profit_factor = min(1.0, metrics.profit_factor / 2.0) if metrics.profit_factor > 0 else 0
            health_factors.append(('profit_factor', profit_factor, 0.2))
            
            # Exposure factor
            exposure_ratio = self.portfolio.get_total_exposure() / self.portfolio.current_bankroll
            exposure_factor = 1.0 - min(1.0, exposure_ratio / 0.5)  # Normalize to 50% max
            health_factors.append(('exposure', exposure_factor, 0.15))
            
            # Diversification factor
            position_count = len(self.portfolio.positions)
            diversification_factor = min(1.0, position_count / 10)  # Normalize to 10 positions
            health_factors.append(('diversification', diversification_factor, 0.15))
            
            # Calculate weighted health score
            health_score = sum(factor * weight for name, factor, weight in health_factors)
            
            # Determine health status
            if health_score >= 0.8:
                health_status = 'excellent'
            elif health_score >= 0.6:
                health_status = 'good'
            elif health_score >= 0.4:
                health_status = 'fair'
            elif health_score >= 0.2:
                health_status = 'poor'
            else:
                health_status = 'critical'
            
            return {
                'health_score': health_score,
                'health_status': health_status,
                'factors': {name: factor for name, factor, weight in health_factors},
                'recommendations': self._generate_health_recommendations(health_factors)
            }
            
        except Exception as e:
            logger.error(f"Health assessment failed: {e}")
            return {
                'health_score': 0.0,
                'health_status': 'unknown',
                'error': str(e)
            }
    
    def _generate_health_recommendations(self, health_factors: List[Tuple[str, float, float]]) -> List[str]:
        """Generate portfolio health recommendations"""
        recommendations = []
        
        for name, factor, weight in health_factors:
            if factor < 0.5:  # Poor factor
                if name == 'drawdown':
                    recommendations.append("Consider reducing position sizes to limit drawdown")
                elif name == 'win_rate':
                    recommendations.append("Review betting strategy - win rate below 50%")
                elif name == 'profit_factor':
                    recommendations.append("Focus on higher edge opportunities")
                elif name == 'exposure':
                    recommendations.append("Reduce overall portfolio exposure")
                elif name == 'diversification':
                    recommendations.append("Increase position diversification")
        
        return recommendations
    
    def add_alert_callback(self, callback: callable):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        try:
            return {
                'timestamp': datetime.now().isoformat(),
                'portfolio_metrics': asdict(self.portfolio.metrics),
                'risk_limits': asdict(self.risk_limits),
                'current_positions': len(self.portfolio.positions),
                'total_exposure': self.portfolio.get_total_exposure(),
                'available_bankroll': self.portfolio.get_available_bankroll(),
                'risk_decisions': dict(self.risk_decisions),
                'recent_alerts': self.alerts[-10:],  # Last 10 alerts
                'daily_stakes': dict(self.daily_stakes),
                'portfolio_health': self._assess_portfolio_health()
            }
            
        except Exception as e:
            logger.error(f"Risk report generation failed: {e}")
            return {'error': str(e)}


def create_risk_manager(risk_level: RiskLevel = RiskLevel.MODERATE, 
                       initial_bankroll: float = 10000.0) -> RiskManager:
    """Create risk manager with predefined risk level"""
    
    # Define risk limits by level
    if risk_level == RiskLevel.CONSERVATIVE:
        limits = RiskLimits(
            max_stake_per_bet=50.0,
            max_stake_per_match=200.0,
            max_daily_stake=500.0,
            max_daily_loss=200.0,
            max_concurrent_bets=5,
            max_exposure_percentage=15.0,
            min_confidence_threshold=0.75,
            min_edge_threshold=0.05,
            stop_loss_percentage=10.0
        )
        sizer = FixedPercentageSizer(1.0)  # 1% per bet
        
    elif risk_level == RiskLevel.MODERATE:
        limits = RiskLimits(
            max_stake_per_bet=100.0,
            max_stake_per_match=500.0,
            max_daily_stake=1000.0,
            max_daily_loss=400.0,
            max_concurrent_bets=10,
            max_exposure_percentage=25.0,
            min_confidence_threshold=0.65,
            min_edge_threshold=0.03,
            stop_loss_percentage=15.0
        )
        sizer = KellyCriterionSizer(0.25)
        
    elif risk_level == RiskLevel.AGGRESSIVE:
        limits = RiskLimits(
            max_stake_per_bet=200.0,
            max_stake_per_match=1000.0,
            max_daily_stake=2000.0,
            max_daily_loss=800.0,
            max_concurrent_bets=20,
            max_exposure_percentage=40.0,
            min_confidence_threshold=0.55,
            min_edge_threshold=0.02,
            stop_loss_percentage=20.0
        )
        sizer = ConfidenceWeightedSizer(3.0, 5.0)
    
    else:  # Custom - use moderate defaults
        limits = RiskLimits()
        sizer = KellyCriterionSizer()
    
    portfolio = Portfolio(initial_bankroll)
    return RiskManager(limits, portfolio, sizer)


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create risk manager
    risk_manager = create_risk_manager(RiskLevel.MODERATE, 10000.0)
    
    # Example bet evaluation
    prediction = {
        'confidence': 0.75,
        'edge': 0.08,
        'probability': 0.65
    }
    
    market_data = {
        'odds': 2.1,
        'liquidity': 5000,
        'volatility': 0.15,
        'spread': 0.05
    }
    
    match_info = {
        'match_id': 'test_match_123',
        'player1': 'Novak Djokovic',
        'player2': 'Rafael Nadal',
        'tournament': 'French Open',
        'surface': 'clay'
    }
    
    # Evaluate bet
    result = risk_manager.evaluate_bet_request(prediction, market_data, match_info)
    print(f"Bet evaluation result: {result}")
    
    # Generate risk report
    report = risk_manager.get_risk_report()
    print(f"Risk report: {json.dumps(report, indent=2, default=str)}")
    
    print("✅ Risk Management System test completed")