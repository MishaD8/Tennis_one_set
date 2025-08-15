#!/usr/bin/env python3
"""
Comprehensive Bet Settlement & P&L Tracking System
Advanced settlement processing and performance analytics for automated tennis betting
"""

import logging
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
import numpy as np
from collections import defaultdict
import threading
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


class BetStatus(Enum):
    """Bet status enumeration"""
    PENDING = "pending"
    PLACED = "placed"
    MATCHED = "matched"
    CANCELLED = "cancelled"
    VOIDED = "voided"
    SETTLED_WON = "settled_won"
    SETTLED_LOST = "settled_lost"
    SETTLED_HALF_WON = "settled_half_won"
    SETTLED_HALF_LOST = "settled_half_lost"


class MatchStatus(Enum):
    """Match status enumeration"""
    SCHEDULED = "scheduled"
    LIVE = "live"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    SUSPENDED = "suspended"
    RETIRED = "retired"
    WALKOVER = "walkover"


class SettlementRule(Enum):
    """Settlement rule types"""
    WINNER = "winner"
    RETIREMENT = "retirement"
    WALKOVER = "walkover"
    CANCELLATION = "cancellation"
    DEAD_HEAT = "dead_heat"


@dataclass
class MatchResult:
    """Tennis match result data"""
    match_id: str
    player1_name: str
    player2_name: str
    winner: Optional[int]  # 1 or 2, None for no result
    final_score: str
    match_status: MatchStatus
    completed_at: Optional[datetime]
    retirement_details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['match_status'] = self.match_status.value
        data['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        return data


@dataclass
class BetSettlement:
    """Bet settlement record"""
    settlement_id: str
    bet_id: str
    match_id: str
    settlement_rule: SettlementRule
    original_stake: float
    settlement_amount: float
    profit_loss: float
    settlement_reason: str
    settled_at: datetime
    tax_amount: float = 0.0
    commission_amount: float = 0.0
    net_profit_loss: float = 0.0
    
    def __post_init__(self):
        """Calculate net P&L after taxes and commission"""
        self.net_profit_loss = self.profit_loss - self.tax_amount - self.commission_amount
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['settlement_rule'] = self.settlement_rule.value
        data['settled_at'] = self.settled_at.isoformat()
        return data


@dataclass
class PnLSummary:
    """P&L summary statistics"""
    period_start: datetime
    period_end: datetime
    total_stakes: float = 0.0
    total_returns: float = 0.0
    gross_profit_loss: float = 0.0
    total_commission: float = 0.0
    total_tax: float = 0.0
    net_profit_loss: float = 0.0
    
    # Performance metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    roi: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # Bet counts
    total_bets: int = 0
    winning_bets: int = 0
    losing_bets: int = 0
    void_bets: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['period_start'] = self.period_start.isoformat()
        data['period_end'] = self.period_end.isoformat()
        return data


class BetSettlementEngine:
    """
    Core settlement engine for processing bet outcomes
    """
    
    def __init__(self, db_path: str = "tennis_data_enhanced/betting_settlement.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._initialize_database()
        
        # Settlement rules configuration
        self.settlement_rules = {
            SettlementRule.WINNER: self._settle_winner_market,
            SettlementRule.RETIREMENT: self._settle_retirement,
            SettlementRule.WALKOVER: self._settle_walkover,
            SettlementRule.CANCELLATION: self._settle_cancellation,
            SettlementRule.DEAD_HEAT: self._settle_dead_heat
        }
        
        # Commission and tax rates (configurable)
        self.commission_rate = 0.05  # 5% commission
        self.tax_rate = 0.0  # No tax by default
        
        # Performance tracking
        self.settlement_stats = {
            'total_settlements': 0,
            'successful_settlements': 0,
            'failed_settlements': 0,
            'last_settlement': None
        }
    
    def _initialize_database(self):
        """Initialize SQLite database for settlement tracking"""
        try:
            # Ensure directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Bets table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS bets (
                    bet_id TEXT PRIMARY KEY,
                    match_id TEXT NOT NULL,
                    market_type TEXT NOT NULL,
                    selection TEXT NOT NULL,
                    stake REAL NOT NULL,
                    odds REAL NOT NULL,
                    potential_return REAL NOT NULL,
                    status TEXT NOT NULL,
                    placed_at TIMESTAMP NOT NULL,
                    betfair_bet_id TEXT,
                    strategy TEXT,
                    confidence REAL,
                    edge REAL,
                    model_version TEXT
                )
                ''')
                
                # Match results table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS match_results (
                    match_id TEXT PRIMARY KEY,
                    player1_name TEXT NOT NULL,
                    player2_name TEXT NOT NULL,
                    winner INTEGER,
                    final_score TEXT,
                    match_status TEXT NOT NULL,
                    completed_at TIMESTAMP,
                    retirement_details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # Settlements table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS settlements (
                    settlement_id TEXT PRIMARY KEY,
                    bet_id TEXT NOT NULL,
                    match_id TEXT NOT NULL,
                    settlement_rule TEXT NOT NULL,
                    original_stake REAL NOT NULL,
                    settlement_amount REAL NOT NULL,
                    profit_loss REAL NOT NULL,
                    commission_amount REAL DEFAULT 0.0,
                    tax_amount REAL DEFAULT 0.0,
                    net_profit_loss REAL NOT NULL,
                    settlement_reason TEXT NOT NULL,
                    settled_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (bet_id) REFERENCES bets (bet_id),
                    FOREIGN KEY (match_id) REFERENCES match_results (match_id)
                )
                ''')
                
                # P&L tracking table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS pnl_daily (
                    date DATE PRIMARY KEY,
                    total_stakes REAL DEFAULT 0.0,
                    total_returns REAL DEFAULT 0.0,
                    gross_profit_loss REAL DEFAULT 0.0,
                    total_commission REAL DEFAULT 0.0,
                    total_tax REAL DEFAULT 0.0,
                    net_profit_loss REAL DEFAULT 0.0,
                    total_bets INTEGER DEFAULT 0,
                    winning_bets INTEGER DEFAULT 0,
                    losing_bets INTEGER DEFAULT 0,
                    void_bets INTEGER DEFAULT 0
                )
                ''')
                
                # Performance metrics table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    metric_date DATE,
                    metric_type TEXT,
                    metric_value REAL,
                    period_type TEXT,
                    PRIMARY KEY (metric_date, metric_type, period_type)
                )
                ''')
                
                # Indexes for performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_bets_match_id ON bets(match_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_bets_status ON bets(status)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_settlements_bet_id ON settlements(bet_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_settlements_settled_at ON settlements(settled_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_pnl_date ON pnl_daily(date)')
                
                conn.commit()
                logger.info("✅ Settlement database initialized")
                
        except Exception as e:
            logger.error(f"❌ Settlement database initialization failed: {e}")
            raise
    
    def record_bet(self, bet_data: Dict[str, Any]) -> bool:
        """Record a new bet in the system"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                    INSERT OR REPLACE INTO bets (
                        bet_id, match_id, market_type, selection, stake, odds,
                        potential_return, status, placed_at, betfair_bet_id,
                        strategy, confidence, edge, model_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        bet_data['bet_id'],
                        bet_data['match_id'],
                        bet_data.get('market_type', 'MATCH_ODDS'),
                        bet_data['selection'],
                        bet_data['stake'],
                        bet_data['odds'],
                        bet_data['stake'] * bet_data['odds'],
                        bet_data.get('status', 'pending'),
                        bet_data.get('placed_at', datetime.now().isoformat()),
                        bet_data.get('betfair_bet_id'),
                        bet_data.get('strategy'),
                        bet_data.get('confidence'),
                        bet_data.get('edge'),
                        bet_data.get('model_version')
                    ))
                    
                    conn.commit()
                    logger.info(f"Bet recorded: {bet_data['bet_id']}")
                    return True
                    
            except Exception as e:
                logger.error(f"Failed to record bet: {e}")
                return False
    
    def record_match_result(self, match_result: MatchResult) -> bool:
        """Record match result for settlement"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                    INSERT OR REPLACE INTO match_results (
                        match_id, player1_name, player2_name, winner, final_score,
                        match_status, completed_at, retirement_details
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        match_result.match_id,
                        match_result.player1_name,
                        match_result.player2_name,
                        match_result.winner,
                        match_result.final_score,
                        match_result.match_status.value,
                        match_result.completed_at.isoformat() if match_result.completed_at else None,
                        json.dumps(match_result.retirement_details) if match_result.retirement_details else None
                    ))
                    
                    conn.commit()
                    logger.info(f"Match result recorded: {match_result.match_id}")
                    
                    # Trigger settlement for affected bets
                    self._trigger_bet_settlements(match_result.match_id)
                    
                    return True
                    
            except Exception as e:
                logger.error(f"Failed to record match result: {e}")
                return False
    
    def _trigger_bet_settlements(self, match_id: str):
        """Trigger settlement for all bets on a completed match"""
        try:
            # Get all unsettled bets for this match
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT bet_id, market_type, selection, stake, odds, status
                FROM bets 
                WHERE match_id = ? AND status IN ('placed', 'matched')
                ''', (match_id,))
                
                unsettled_bets = cursor.fetchall()
            
            # Process each bet
            for bet_row in unsettled_bets:
                bet_id = bet_row[0]
                self.settle_bet(bet_id)
                
        except Exception as e:
            logger.error(f"Failed to trigger settlements for match {match_id}: {e}")
    
    def settle_bet(self, bet_id: str) -> Optional[BetSettlement]:
        """Settle individual bet based on match result"""
        with self.lock:
            try:
                # Get bet details
                bet_data = self._get_bet_data(bet_id)
                if not bet_data:
                    logger.error(f"Bet not found: {bet_id}")
                    return None
                
                # Get match result
                match_result = self._get_match_result(bet_data['match_id'])
                if not match_result:
                    logger.warning(f"Match result not available for bet {bet_id}")
                    return None
                
                # Determine settlement rule
                settlement_rule = self._determine_settlement_rule(match_result)
                
                # Apply settlement rule
                settlement_result = self.settlement_rules[settlement_rule](bet_data, match_result)
                
                if settlement_result:
                    # Calculate commission and tax
                    commission = settlement_result['settlement_amount'] * self.commission_rate if settlement_result['profit_loss'] > 0 else 0
                    tax = settlement_result['profit_loss'] * self.tax_rate if settlement_result['profit_loss'] > 0 else 0
                    
                    # Create settlement record
                    settlement = BetSettlement(
                        settlement_id=str(uuid.uuid4()),
                        bet_id=bet_id,
                        match_id=bet_data['match_id'],
                        settlement_rule=settlement_rule,
                        original_stake=bet_data['stake'],
                        settlement_amount=settlement_result['settlement_amount'],
                        profit_loss=settlement_result['profit_loss'],
                        settlement_reason=settlement_result['reason'],
                        settled_at=datetime.now(),
                        commission_amount=commission,
                        tax_amount=tax
                    )
                    
                    # Store settlement
                    if self._store_settlement(settlement):
                        # Update bet status
                        self._update_bet_status(bet_id, settlement_result['status'])
                        
                        # Update daily P&L
                        self._update_daily_pnl(settlement)
                        
                        self.settlement_stats['successful_settlements'] += 1
                        self.settlement_stats['last_settlement'] = datetime.now()
                        
                        logger.info(f"Bet settled: {bet_id} - P&L: {settlement.net_profit_loss:.2f}")
                        return settlement
                    
                self.settlement_stats['failed_settlements'] += 1
                return None
                
            except Exception as e:
                logger.error(f"Bet settlement failed for {bet_id}: {e}")
                self.settlement_stats['failed_settlements'] += 1
                return None
            finally:
                self.settlement_stats['total_settlements'] += 1
    
    def _get_bet_data(self, bet_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve bet data from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT bet_id, match_id, market_type, selection, stake, odds,
                       potential_return, status, placed_at, betfair_bet_id,
                       strategy, confidence, edge, model_version
                FROM bets WHERE bet_id = ?
                ''', (bet_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'bet_id': row[0],
                        'match_id': row[1],
                        'market_type': row[2],
                        'selection': row[3],
                        'stake': row[4],
                        'odds': row[5],
                        'potential_return': row[6],
                        'status': row[7],
                        'placed_at': row[8],
                        'betfair_bet_id': row[9],
                        'strategy': row[10],
                        'confidence': row[11],
                        'edge': row[12],
                        'model_version': row[13]
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get bet data: {e}")
            return None
    
    def _get_match_result(self, match_id: str) -> Optional[MatchResult]:
        """Retrieve match result from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT match_id, player1_name, player2_name, winner, final_score,
                       match_status, completed_at, retirement_details
                FROM match_results WHERE match_id = ?
                ''', (match_id,))
                
                row = cursor.fetchone()
                if row:
                    return MatchResult(
                        match_id=row[0],
                        player1_name=row[1],
                        player2_name=row[2],
                        winner=row[3],
                        final_score=row[4],
                        match_status=MatchStatus(row[5]),
                        completed_at=datetime.fromisoformat(row[6]) if row[6] else None,
                        retirement_details=json.loads(row[7]) if row[7] else None
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get match result: {e}")
            return None
    
    def _determine_settlement_rule(self, match_result: MatchResult) -> SettlementRule:
        """Determine which settlement rule to apply"""
        if match_result.match_status == MatchStatus.COMPLETED:
            return SettlementRule.WINNER
        elif match_result.match_status == MatchStatus.RETIRED:
            return SettlementRule.RETIREMENT
        elif match_result.match_status in [MatchStatus.CANCELLED, MatchStatus.SUSPENDED]:
            return SettlementRule.CANCELLATION
        elif match_result.match_status == MatchStatus.WALKOVER:
            return SettlementRule.WALKOVER
        else:
            return SettlementRule.WINNER  # Default
    
    def _settle_winner_market(self, bet_data: Dict[str, Any], match_result: MatchResult) -> Optional[Dict[str, Any]]:
        """Settle winner market bet"""
        try:
            selection = bet_data['selection']
            winner = match_result.winner
            stake = bet_data['stake']
            odds = bet_data['odds']
            
            # Determine if bet won
            if selection == f"player_{winner}":
                # Winning bet
                settlement_amount = stake * odds
                profit_loss = settlement_amount - stake
                status = BetStatus.SETTLED_WON
                reason = f"Player {winner} won the match"
            else:
                # Losing bet
                settlement_amount = 0.0
                profit_loss = -stake
                status = BetStatus.SETTLED_LOST
                reason = f"Player {winner} won the match (bet on other player)"
            
            return {
                'settlement_amount': settlement_amount,
                'profit_loss': profit_loss,
                'status': status.value,
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"Winner settlement failed: {e}")
            return None
    
    def _settle_retirement(self, bet_data: Dict[str, Any], match_result: MatchResult) -> Optional[Dict[str, Any]]:
        """Settle retirement scenario"""
        try:
            # Check retirement rules (typically bets are void if player retires before completing first set)
            retirement_details = match_result.retirement_details or {}
            sets_completed = retirement_details.get('sets_completed', 0)
            
            stake = bet_data['stake']
            
            if sets_completed < 1:
                # Void bet - return stake
                settlement_amount = stake
                profit_loss = 0.0
                status = BetStatus.VOIDED
                reason = "Match retired before completion of first set - bet voided"
            else:
                # Settle as normal winner market
                return self._settle_winner_market(bet_data, match_result)
            
            return {
                'settlement_amount': settlement_amount,
                'profit_loss': profit_loss,
                'status': status.value,
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"Retirement settlement failed: {e}")
            return None
    
    def _settle_walkover(self, bet_data: Dict[str, Any], match_result: MatchResult) -> Optional[Dict[str, Any]]:
        """Settle walkover scenario"""
        try:
            selection = bet_data['selection']
            winner = match_result.winner
            stake = bet_data['stake']
            odds = bet_data['odds']
            
            # Walkover - settle normally based on winner
            if winner and selection == f"player_{winner}":
                settlement_amount = stake * odds
                profit_loss = settlement_amount - stake
                status = BetStatus.SETTLED_WON
                reason = f"Player {winner} won by walkover"
            else:
                settlement_amount = 0.0
                profit_loss = -stake
                status = BetStatus.SETTLED_LOST
                reason = f"Player {winner} won by walkover (bet on other player)"
            
            return {
                'settlement_amount': settlement_amount,
                'profit_loss': profit_loss,
                'status': status.value,
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"Walkover settlement failed: {e}")
            return None
    
    def _settle_cancellation(self, bet_data: Dict[str, Any], match_result: MatchResult) -> Optional[Dict[str, Any]]:
        """Settle cancellation scenario"""
        try:
            stake = bet_data['stake']
            
            # Cancelled match - void all bets
            settlement_amount = stake
            profit_loss = 0.0
            status = BetStatus.VOIDED
            reason = "Match cancelled - bet voided"
            
            return {
                'settlement_amount': settlement_amount,
                'profit_loss': profit_loss,
                'status': status.value,
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"Cancellation settlement failed: {e}")
            return None
    
    def _settle_dead_heat(self, bet_data: Dict[str, Any], match_result: MatchResult) -> Optional[Dict[str, Any]]:
        """Settle dead heat scenario (rare in tennis)"""
        try:
            stake = bet_data['stake']
            odds = bet_data['odds']
            
            # Dead heat - typically half stakes
            settlement_amount = stake * odds * 0.5
            profit_loss = settlement_amount - stake
            status = BetStatus.SETTLED_HALF_WON
            reason = "Dead heat settlement - half stakes"
            
            return {
                'settlement_amount': settlement_amount,
                'profit_loss': profit_loss,
                'status': status.value,
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"Dead heat settlement failed: {e}")
            return None
    
    def _store_settlement(self, settlement: BetSettlement) -> bool:
        """Store settlement record in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT INTO settlements (
                    settlement_id, bet_id, match_id, settlement_rule,
                    original_stake, settlement_amount, profit_loss,
                    commission_amount, tax_amount, net_profit_loss,
                    settlement_reason, settled_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    settlement.settlement_id,
                    settlement.bet_id,
                    settlement.match_id,
                    settlement.settlement_rule.value,
                    settlement.original_stake,
                    settlement.settlement_amount,
                    settlement.profit_loss,
                    settlement.commission_amount,
                    settlement.tax_amount,
                    settlement.net_profit_loss,
                    settlement.settlement_reason,
                    settlement.settled_at.isoformat()
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to store settlement: {e}")
            return False
    
    def _update_bet_status(self, bet_id: str, status: str):
        """Update bet status in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                UPDATE bets SET status = ? WHERE bet_id = ?
                ''', (status, bet_id))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to update bet status: {e}")
    
    def _update_daily_pnl(self, settlement: BetSettlement):
        """Update daily P&L summary"""
        try:
            settlement_date = settlement.settled_at.date()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get existing daily record
                cursor.execute('''
                SELECT total_stakes, total_returns, gross_profit_loss, total_commission,
                       total_tax, net_profit_loss, total_bets, winning_bets, losing_bets, void_bets
                FROM pnl_daily WHERE date = ?
                ''', (settlement_date,))
                
                row = cursor.fetchone()
                
                if row:
                    # Update existing record
                    total_stakes = row[0] + settlement.original_stake
                    total_returns = row[1] + settlement.settlement_amount
                    gross_profit_loss = row[2] + settlement.profit_loss
                    total_commission = row[3] + settlement.commission_amount
                    total_tax = row[4] + settlement.tax_amount
                    net_profit_loss = row[5] + settlement.net_profit_loss
                    total_bets = row[6] + 1
                    
                    # Update bet counters
                    winning_bets = row[7]
                    losing_bets = row[8] 
                    void_bets = row[9]
                    
                    if settlement.profit_loss > 0:
                        winning_bets += 1
                    elif settlement.profit_loss < 0:
                        losing_bets += 1
                    else:
                        void_bets += 1
                    
                    cursor.execute('''
                    UPDATE pnl_daily SET 
                        total_stakes = ?, total_returns = ?, gross_profit_loss = ?,
                        total_commission = ?, total_tax = ?, net_profit_loss = ?,
                        total_bets = ?, winning_bets = ?, losing_bets = ?, void_bets = ?
                    WHERE date = ?
                    ''', (
                        total_stakes, total_returns, gross_profit_loss,
                        total_commission, total_tax, net_profit_loss,
                        total_bets, winning_bets, losing_bets, void_bets,
                        settlement_date
                    ))
                else:
                    # Create new record
                    winning_bets = 1 if settlement.profit_loss > 0 else 0
                    losing_bets = 1 if settlement.profit_loss < 0 else 0
                    void_bets = 1 if settlement.profit_loss == 0 else 0
                    
                    cursor.execute('''
                    INSERT INTO pnl_daily (
                        date, total_stakes, total_returns, gross_profit_loss,
                        total_commission, total_tax, net_profit_loss,
                        total_bets, winning_bets, losing_bets, void_bets
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        settlement_date, settlement.original_stake, settlement.settlement_amount,
                        settlement.profit_loss, settlement.commission_amount, settlement.tax_amount,
                        settlement.net_profit_loss, 1, winning_bets, losing_bets, void_bets
                    ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to update daily P&L: {e}")


class PnLAnalyzer:
    """
    Advanced P&L analysis and performance metrics calculator
    """
    
    def __init__(self, settlement_engine: BetSettlementEngine):
        self.settlement_engine = settlement_engine
        self.db_path = settlement_engine.db_path
    
    def get_pnl_summary(self, start_date: datetime, end_date: datetime) -> PnLSummary:
        """Generate comprehensive P&L summary for date range"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get aggregated data
                cursor.execute('''
                SELECT 
                    SUM(total_stakes) as total_stakes,
                    SUM(total_returns) as total_returns,
                    SUM(gross_profit_loss) as gross_profit_loss,
                    SUM(total_commission) as total_commission,
                    SUM(total_tax) as total_tax,
                    SUM(net_profit_loss) as net_profit_loss,
                    SUM(total_bets) as total_bets,
                    SUM(winning_bets) as winning_bets,
                    SUM(losing_bets) as losing_bets,
                    SUM(void_bets) as void_bets
                FROM pnl_daily 
                WHERE date BETWEEN ? AND ?
                ''', (start_date.date(), end_date.date()))
                
                row = cursor.fetchone()
                
                if row and row[0]:  # Check if we have data
                    summary = PnLSummary(
                        period_start=start_date,
                        period_end=end_date,
                        total_stakes=row[0] or 0.0,
                        total_returns=row[1] or 0.0,
                        gross_profit_loss=row[2] or 0.0,
                        total_commission=row[3] or 0.0,
                        total_tax=row[4] or 0.0,
                        net_profit_loss=row[5] or 0.0,
                        total_bets=int(row[6]) or 0,
                        winning_bets=int(row[7]) or 0,
                        losing_bets=int(row[8]) or 0,
                        void_bets=int(row[9]) or 0
                    )
                    
                    # Calculate performance metrics
                    self._calculate_performance_metrics(summary, start_date, end_date)
                    
                    return summary
                else:
                    # No data for period
                    return PnLSummary(period_start=start_date, period_end=end_date)
                    
        except Exception as e:
            logger.error(f"Failed to generate P&L summary: {e}")
            return PnLSummary(period_start=start_date, period_end=end_date)
    
    def _calculate_performance_metrics(self, summary: PnLSummary, start_date: datetime, end_date: datetime):
        """Calculate advanced performance metrics"""
        try:
            # Win rate
            if summary.total_bets > 0:
                summary.win_rate = summary.winning_bets / summary.total_bets
            
            # ROI
            if summary.total_stakes > 0:
                summary.roi = summary.net_profit_loss / summary.total_stakes
            
            # Profit factor
            if summary.losing_bets > 0:
                total_losses = abs(self._get_total_losses(start_date, end_date))
                total_wins = self._get_total_wins(start_date, end_date)
                if total_losses > 0:
                    summary.profit_factor = total_wins / total_losses
            
            # Get daily P&L for Sharpe ratio and drawdown
            daily_pnl = self._get_daily_pnl_series(start_date, end_date)
            
            if len(daily_pnl) > 1:
                # Sharpe ratio (assuming 0% risk-free rate)
                pnl_std = np.std(daily_pnl)
                if pnl_std > 0:
                    summary.sharpe_ratio = np.mean(daily_pnl) / pnl_std * np.sqrt(252)  # Annualized
                
                # Maximum drawdown
                cumulative_pnl = np.cumsum(daily_pnl)
                running_max = np.maximum.accumulate(cumulative_pnl)
                drawdowns = (cumulative_pnl - running_max) / np.maximum(running_max, 1)  # Avoid division by zero
                summary.max_drawdown = abs(np.min(drawdowns))
            
        except Exception as e:
            logger.error(f"Failed to calculate performance metrics: {e}")
    
    def _get_total_losses(self, start_date: datetime, end_date: datetime) -> float:
        """Get total losses for period"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT SUM(net_profit_loss)
                FROM settlements s
                JOIN bets b ON s.bet_id = b.bet_id
                WHERE DATE(s.settled_at) BETWEEN ? AND ?
                AND s.net_profit_loss < 0
                ''', (start_date.date(), end_date.date()))
                
                result = cursor.fetchone()
                return result[0] if result[0] else 0.0
                
        except Exception as e:
            logger.error(f"Failed to get total losses: {e}")
            return 0.0
    
    def _get_total_wins(self, start_date: datetime, end_date: datetime) -> float:
        """Get total wins for period"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT SUM(net_profit_loss)
                FROM settlements s
                JOIN bets b ON s.bet_id = b.bet_id
                WHERE DATE(s.settled_at) BETWEEN ? AND ?
                AND s.net_profit_loss > 0
                ''', (start_date.date(), end_date.date()))
                
                result = cursor.fetchone()
                return result[0] if result[0] else 0.0
                
        except Exception as e:
            logger.error(f"Failed to get total wins: {e}")
            return 0.0
    
    def _get_daily_pnl_series(self, start_date: datetime, end_date: datetime) -> List[float]:
        """Get daily P&L series for statistical analysis"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT net_profit_loss
                FROM pnl_daily
                WHERE date BETWEEN ? AND ?
                ORDER BY date
                ''', (start_date.date(), end_date.date()))
                
                rows = cursor.fetchall()
                return [row[0] for row in rows if row[0] is not None]
                
        except Exception as e:
            logger.error(f"Failed to get daily P&L series: {e}")
            return []
    
    def get_performance_breakdown(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get detailed performance breakdown by various dimensions"""
        try:
            breakdown = {}
            
            with sqlite3.connect(self.db_path) as conn:
                # By strategy
                breakdown['by_strategy'] = pd.read_sql_query('''
                SELECT 
                    b.strategy,
                    COUNT(*) as bet_count,
                    SUM(s.original_stake) as total_stakes,
                    SUM(s.net_profit_loss) as net_pnl,
                    AVG(s.net_profit_loss / s.original_stake) as avg_roi,
                    SUM(CASE WHEN s.net_profit_loss > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as win_rate
                FROM settlements s
                JOIN bets b ON s.bet_id = b.bet_id
                WHERE DATE(s.settled_at) BETWEEN ? AND ?
                GROUP BY b.strategy
                ''', conn, params=(start_date.date(), end_date.date())).to_dict('records')
                
                # By model version
                breakdown['by_model'] = pd.read_sql_query('''
                SELECT 
                    b.model_version,
                    COUNT(*) as bet_count,
                    SUM(s.original_stake) as total_stakes,
                    SUM(s.net_profit_loss) as net_pnl,
                    AVG(s.net_profit_loss / s.original_stake) as avg_roi,
                    SUM(CASE WHEN s.net_profit_loss > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as win_rate
                FROM settlements s
                JOIN bets b ON s.bet_id = b.bet_id
                WHERE DATE(s.settled_at) BETWEEN ? AND ?
                AND b.model_version IS NOT NULL
                GROUP BY b.model_version
                ''', conn, params=(start_date.date(), end_date.date())).to_dict('records')
                
                # By odds range
                breakdown['by_odds'] = pd.read_sql_query('''
                SELECT 
                    CASE 
                        WHEN b.odds < 1.5 THEN '< 1.5'
                        WHEN b.odds < 2.0 THEN '1.5 - 2.0'
                        WHEN b.odds < 3.0 THEN '2.0 - 3.0'
                        WHEN b.odds < 5.0 THEN '3.0 - 5.0'
                        ELSE '> 5.0'
                    END as odds_range,
                    COUNT(*) as bet_count,
                    SUM(s.original_stake) as total_stakes,
                    SUM(s.net_profit_loss) as net_pnl,
                    AVG(s.net_profit_loss / s.original_stake) as avg_roi,
                    SUM(CASE WHEN s.net_profit_loss > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as win_rate
                FROM settlements s
                JOIN bets b ON s.bet_id = b.bet_id
                WHERE DATE(s.settled_at) BETWEEN ? AND ?
                GROUP BY odds_range
                ORDER BY MIN(b.odds)
                ''', conn, params=(start_date.date(), end_date.date())).to_dict('records')
            
            return breakdown
            
        except Exception as e:
            logger.error(f"Failed to get performance breakdown: {e}")
            return {}
    
    def get_monthly_summary(self, year: int) -> List[Dict[str, Any]]:
        """Get monthly P&L summary for specified year"""
        try:
            monthly_data = []
            
            for month in range(1, 13):
                start_date = datetime(year, month, 1)
                if month == 12:
                    end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
                else:
                    end_date = datetime(year, month + 1, 1) - timedelta(days=1)
                
                summary = self.get_pnl_summary(start_date, end_date)
                
                monthly_data.append({
                    'month': month,
                    'month_name': start_date.strftime('%B'),
                    'total_stakes': summary.total_stakes,
                    'net_pnl': summary.net_profit_loss,
                    'roi': summary.roi,
                    'win_rate': summary.win_rate,
                    'total_bets': summary.total_bets,
                    'profit_factor': summary.profit_factor
                })
            
            return monthly_data
            
        except Exception as e:
            logger.error(f"Failed to get monthly summary: {e}")
            return []
    
    def export_settlement_data(self, start_date: datetime, end_date: datetime, 
                              format: str = 'csv') -> Optional[str]:
        """Export settlement data for external analysis"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                SELECT 
                    s.settlement_id,
                    s.bet_id,
                    s.match_id,
                    b.market_type,
                    b.selection,
                    b.odds,
                    s.original_stake,
                    s.settlement_amount,
                    s.profit_loss,
                    s.net_profit_loss,
                    s.settlement_reason,
                    s.settled_at,
                    b.strategy,
                    b.model_version,
                    b.confidence,
                    b.edge
                FROM settlements s
                JOIN bets b ON s.bet_id = b.bet_id
                WHERE DATE(s.settled_at) BETWEEN ? AND ?
                ORDER BY s.settled_at
                '''
                
                df = pd.read_sql_query(query, conn, params=(start_date.date(), end_date.date()))
                
                # Export to specified format
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                if format.lower() == 'csv':
                    filename = f"settlement_export_{timestamp}.csv"
                    df.to_csv(filename, index=False)
                elif format.lower() == 'excel':
                    filename = f"settlement_export_{timestamp}.xlsx"
                    df.to_excel(filename, index=False)
                else:
                    logger.error(f"Unsupported export format: {format}")
                    return None
                
                logger.info(f"Settlement data exported to {filename}")
                return filename
                
        except Exception as e:
            logger.error(f"Failed to export settlement data: {e}")
            return None


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize settlement engine
    settlement_engine = BetSettlementEngine()
    
    # Example bet
    bet_data = {
        'bet_id': 'test_bet_123',
        'match_id': 'test_match_123',
        'market_type': 'MATCH_ODDS',
        'selection': 'player_1',
        'stake': 50.0,
        'odds': 2.1,
        'status': 'placed',
        'strategy': 'ml_prediction',
        'confidence': 0.75,
        'edge': 0.08,
        'model_version': 'v1.0'
    }
    
    # Record bet
    success = settlement_engine.record_bet(bet_data)
    print(f"Bet recorded: {success}")
    
    # Example match result
    match_result = MatchResult(
        match_id='test_match_123',
        player1_name='Novak Djokovic',
        player2_name='Rafael Nadal',
        winner=1,
        final_score='6-4, 6-2',
        match_status=MatchStatus.COMPLETED,
        completed_at=datetime.now()
    )
    
    # Record match result (this will trigger settlement)
    success = settlement_engine.record_match_result(match_result)
    print(f"Match result recorded: {success}")
    
    # Initialize P&L analyzer
    analyzer = PnLAnalyzer(settlement_engine)
    
    # Get P&L summary
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    summary = analyzer.get_pnl_summary(start_date, end_date)
    print(f"P&L Summary: {summary.to_dict()}")
    
    print("✅ Bet Settlement System test completed")