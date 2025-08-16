#!/usr/bin/env python3
"""
📊 Performance Comparison Service for Tennis Betting System
Compare backtesting vs forward testing results to detect model degradation
"""

import os
import json
import logging
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import sessionmaker
from sqlalchemy import and_, or_, desc, asc, func

from src.data.database_models import (
    DatabaseManager, TestingSession, PerformanceComparison, 
    BettingLog, TestMode, BettingStatus, Base
)

logger = logging.getLogger(__name__)

class PerformanceComparisonService:
    """
    Advanced performance comparison service for detecting model degradation
    and stability across backtesting and forward testing periods
    """
    
    def __init__(self, database_manager: DatabaseManager = None):
        self.db_manager = database_manager or DatabaseManager()
        self.session = self.db_manager.get_session()
        
        # Ensure tables exist
        try:
            Base.metadata.create_all(bind=self.db_manager.engine)
            logger.info("✅ Performance comparison tables created/verified")
        except Exception as e:
            logger.error(f"❌ Error creating performance comparison tables: {e}")
    
    def create_performance_comparison(self, 
                                    comparison_name: str,
                                    backtest_session_id: str,
                                    forward_test_session_id: str) -> str:
        """
        Create a comprehensive performance comparison between backtest and forward test sessions
        """
        try:
            # Get sessions
            backtest_session = self.session.query(TestingSession).filter(
                TestingSession.session_id == backtest_session_id
            ).first()
            
            forward_test_session = self.session.query(TestingSession).filter(
                TestingSession.session_id == forward_test_session_id
            ).first()
            
            if not backtest_session or not forward_test_session:
                logger.error("❌ One or both sessions not found")
                return ""
            
            if backtest_session.test_mode != TestMode.BACKTEST:
                logger.error("❌ First session is not a backtest session")
                return ""
            
            if forward_test_session.test_mode != TestMode.FORWARD_TEST:
                logger.error("❌ Second session is not a forward test session")
                return ""
            
            comparison_id = f"comparison_{uuid.uuid4().hex[:12]}"
            
            # Calculate comparison metrics
            roi_difference = forward_test_session.roi_percentage - backtest_session.roi_percentage
            win_rate_difference = forward_test_session.win_rate - backtest_session.win_rate
            profit_difference = forward_test_session.net_profit - backtest_session.net_profit
            
            # Advanced stability analysis
            stability_analysis = self._analyze_model_stability(
                backtest_session, forward_test_session
            )
            
            # Create comparison record
            comparison = PerformanceComparison(
                comparison_id=comparison_id,
                comparison_name=comparison_name,
                backtest_session_id=backtest_session.id,
                forward_test_session_id=forward_test_session.id,
                backtest_roi=backtest_session.roi_percentage,
                forward_test_roi=forward_test_session.roi_percentage,
                roi_difference=roi_difference,
                backtest_win_rate=backtest_session.win_rate,
                forward_test_win_rate=forward_test_session.win_rate,
                win_rate_difference=win_rate_difference,
                backtest_profit=backtest_session.net_profit,
                forward_test_profit=forward_test_session.net_profit,
                profit_difference=profit_difference,
                performance_stability_score=stability_analysis['stability_score'],
                model_degradation_indicator=stability_analysis['degradation_detected'],
                analysis_summary=json.dumps(stability_analysis['summary']),
                recommendations=json.dumps(stability_analysis['recommendations'])
            )
            
            self.session.add(comparison)
            self.session.commit()
            
            logger.info(f"📊 Created performance comparison: {comparison_id}")
            logger.info(f"   Backtest ROI: {backtest_session.roi_percentage:.2f}%")
            logger.info(f"   Forward ROI: {forward_test_session.roi_percentage:.2f}%")
            logger.info(f"   Difference: {roi_difference:+.2f}%")
            logger.info(f"   Stability Score: {stability_analysis['stability_score']:.1f}/100")
            
            return comparison_id
            
        except Exception as e:
            logger.error(f"❌ Error creating performance comparison: {e}")
            self.session.rollback()
            return ""
    
    def _analyze_model_stability(self, 
                                backtest_session: TestingSession, 
                                forward_test_session: TestingSession) -> Dict:
        """
        Perform advanced model stability analysis
        """
        try:
            # Basic metric differences
            roi_diff = forward_test_session.roi_percentage - backtest_session.roi_percentage
            win_rate_diff = forward_test_session.win_rate - backtest_session.win_rate
            
            # Calculate stability score (0-100)
            stability_score = 100.0
            
            # ROI stability component (50 points max)
            roi_penalty = min(abs(roi_diff) * 2, 50.0)  # 2 points per 1% ROI difference
            stability_score -= roi_penalty
            
            # Win rate stability component (50 points max)
            win_rate_penalty = min(abs(win_rate_diff), 50.0)  # 1 point per 1% win rate difference
            stability_score -= win_rate_penalty
            
            stability_score = max(0.0, stability_score)
            
            # Degradation detection
            degradation_detected = (
                roi_diff < -5.0 or  # ROI dropped by more than 5%
                win_rate_diff < -10.0 or  # Win rate dropped by more than 10%
                stability_score < 60.0  # Overall stability score below 60
            )
            
            # Get detailed betting logs for deeper analysis
            backtest_analysis = self._analyze_betting_logs(backtest_session.session_id, TestMode.BACKTEST)
            forward_analysis = self._analyze_betting_logs(forward_test_session.session_id, TestMode.FORWARD_TEST)
            
            # Summary of key findings
            summary = {
                'roi_performance': {
                    'backtest': backtest_session.roi_percentage,
                    'forward_test': forward_test_session.roi_percentage,
                    'difference': roi_diff,
                    'status': 'improved' if roi_diff > 0 else 'declined'
                },
                'win_rate_performance': {
                    'backtest': backtest_session.win_rate,
                    'forward_test': forward_test_session.win_rate,
                    'difference': win_rate_diff,
                    'status': 'improved' if win_rate_diff > 0 else 'declined'
                },
                'sample_sizes': {
                    'backtest_bets': backtest_session.total_bets,
                    'forward_test_bets': forward_test_session.total_bets,
                    'sufficient_data': backtest_session.total_bets >= 30 and forward_test_session.total_bets >= 20
                },
                'volatility_analysis': {
                    'backtest_profit_factor': backtest_session.profit_factor or 0,
                    'forward_profit_factor': forward_test_session.profit_factor or 0,
                    'backtest_max_drawdown': backtest_session.max_drawdown or 0,
                    'forward_max_drawdown': forward_test_session.max_drawdown or 0
                }
            }
            
            # Generate recommendations
            recommendations = []
            
            if degradation_detected:
                recommendations.extend([
                    "⚠️ Model degradation detected - immediate attention required",
                    "🔄 Consider retraining ML models with recent data",
                    "📊 Review market conditions and data quality",
                    "💰 Reduce stake sizes until performance stabilizes"
                ])
            elif stability_score < 80:
                recommendations.extend([
                    "🔍 Monitor model performance closely",
                    "📈 Consider minor model adjustments",
                    "📊 Collect more forward test data for validation"
                ])
            else:
                recommendations.extend([
                    "✅ Model performance appears stable",
                    "📊 Continue current strategy",
                    "🔄 Regular monitoring recommended"
                ])
            
            # Add data quality recommendations
            if not summary['sample_sizes']['sufficient_data']:
                recommendations.append("📋 Collect more data for reliable comparison")
            
            if abs(roi_diff) > 15:
                recommendations.append("🚨 Large ROI difference - investigate underlying causes")
            
            if abs(win_rate_diff) > 20:
                recommendations.append("🚨 Large win rate difference - review prediction accuracy")
            
            return {
                'stability_score': stability_score,
                'degradation_detected': degradation_detected,
                'summary': summary,
                'recommendations': recommendations,
                'backtest_analysis': backtest_analysis,
                'forward_analysis': forward_analysis
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing model stability: {e}")
            return {
                'stability_score': 0.0,
                'degradation_detected': True,
                'summary': {'error': str(e)},
                'recommendations': ['❌ Error in analysis - manual review required']
            }
    
    def _analyze_betting_logs(self, session_id: str, test_mode: TestMode) -> Dict:
        """
        Analyze betting logs for a specific session
        """
        try:
            # Get betting logs for this session
            # Note: This assumes we have a way to link betting logs to sessions
            # In practice, you might need to add session_id to BettingLog model
            
            logs = self.session.query(BettingLog).filter(
                and_(
                    BettingLog.test_mode == test_mode,
                    BettingLog.betting_status.in_([BettingStatus.WON, BettingStatus.LOST])
                )
            ).all()
            
            if not logs:
                return {'message': 'No betting logs found'}
            
            # Calculate advanced metrics
            total_bets = len(logs)
            winning_bets = len([log for log in logs if log.betting_status == BettingStatus.WON])
            
            # Edge analysis
            edges = [log.edge_percentage for log in logs]
            avg_edge = np.mean(edges)
            edge_std = np.std(edges)
            
            # Odds analysis
            odds = [log.odds_taken for log in logs]
            avg_odds = np.mean(odds)
            
            # Profit analysis
            profits = [log.profit_loss for log in logs]
            total_profit = sum(profits)
            profit_std = np.std(profits)
            
            # Streak analysis
            current_streak = self._calculate_streak_metrics(logs)
            
            # Model accuracy by confidence level
            confidence_analysis = {}
            for log in logs:
                conf = log.confidence_level
                if conf not in confidence_analysis:
                    confidence_analysis[conf] = {'total': 0, 'correct': 0}
                
                confidence_analysis[conf]['total'] += 1
                if log.prediction_correct:
                    confidence_analysis[conf]['correct'] += 1
            
            # Calculate confidence accuracy rates
            for conf, stats in confidence_analysis.items():
                stats['accuracy'] = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
            
            return {
                'total_bets': total_bets,
                'winning_bets': winning_bets,
                'win_rate': (winning_bets / total_bets) * 100,
                'edge_analysis': {
                    'average_edge': avg_edge,
                    'edge_consistency': edge_std,
                    'positive_edge_bets': len([e for e in edges if e > 0])
                },
                'odds_analysis': {
                    'average_odds': avg_odds,
                    'odds_range': [min(odds), max(odds)]
                },
                'profit_analysis': {
                    'total_profit': total_profit,
                    'profit_volatility': profit_std,
                    'largest_win': max(profits),
                    'largest_loss': min(profits)
                },
                'streak_metrics': current_streak,
                'confidence_analysis': confidence_analysis
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing betting logs: {e}")
            return {'error': str(e)}
    
    def _calculate_streak_metrics(self, logs: List[BettingLog]) -> Dict:
        """Calculate comprehensive streak metrics"""
        if not logs:
            return {}
        
        # Sort by timestamp
        sorted_logs = sorted(logs, key=lambda x: x.timestamp)
        
        streaks = []
        current_streak = {'type': None, 'length': 0}
        
        for log in sorted_logs:
            result = 'win' if log.betting_status == BettingStatus.WON else 'loss'
            
            if current_streak['type'] == result:
                current_streak['length'] += 1
            else:
                if current_streak['type'] is not None:
                    streaks.append(current_streak.copy())
                current_streak = {'type': result, 'length': 1}
        
        # Add final streak
        if current_streak['type'] is not None:
            streaks.append(current_streak)
        
        # Analyze streaks
        win_streaks = [s['length'] for s in streaks if s['type'] == 'win']
        loss_streaks = [s['length'] for s in streaks if s['type'] == 'loss']
        
        return {
            'longest_winning_streak': max(win_streaks) if win_streaks else 0,
            'longest_losing_streak': max(loss_streaks) if loss_streaks else 0,
            'average_winning_streak': np.mean(win_streaks) if win_streaks else 0,
            'average_losing_streak': np.mean(loss_streaks) if loss_streaks else 0,
            'total_streaks': len(streaks),
            'current_streak': streaks[-1] if streaks else {'type': 'none', 'length': 0}
        }
    
    def get_performance_comparisons(self, limit: int = 10) -> List[Dict]:
        """Get all performance comparisons"""
        try:
            comparisons = self.session.query(PerformanceComparison).order_by(
                PerformanceComparison.created_at.desc()
            ).limit(limit).all()
            
            comparison_list = []
            for comp in comparisons:
                comp_dict = {
                    'comparison_id': comp.comparison_id,
                    'comparison_name': comp.comparison_name,
                    'backtest_session_id': comp.backtest_session.session_id if comp.backtest_session else None,
                    'forward_test_session_id': comp.forward_test_session.session_id if comp.forward_test_session else None,
                    'backtest_roi': comp.backtest_roi,
                    'forward_test_roi': comp.forward_test_roi,
                    'roi_difference': comp.roi_difference,
                    'backtest_win_rate': comp.backtest_win_rate,
                    'forward_test_win_rate': comp.forward_test_win_rate,
                    'win_rate_difference': comp.win_rate_difference,
                    'performance_stability_score': comp.performance_stability_score,
                    'model_degradation_indicator': comp.model_degradation_indicator,
                    'analysis_summary': json.loads(comp.analysis_summary) if comp.analysis_summary else {},
                    'recommendations': json.loads(comp.recommendations) if comp.recommendations else [],
                    'created_at': comp.created_at.isoformat()
                }
                comparison_list.append(comp_dict)
            
            return comparison_list
            
        except Exception as e:
            logger.error(f"❌ Error retrieving performance comparisons: {e}")
            return []
    
    def get_detailed_comparison(self, comparison_id: str) -> Dict:
        """Get detailed comparison analysis"""
        try:
            comparison = self.session.query(PerformanceComparison).filter(
                PerformanceComparison.comparison_id == comparison_id
            ).first()
            
            if not comparison:
                return {'error': 'Comparison not found'}
            
            # Get full session details
            backtest_session = comparison.backtest_session
            forward_test_session = comparison.forward_test_session
            
            detailed_analysis = {
                'comparison_info': {
                    'comparison_id': comparison.comparison_id,
                    'comparison_name': comparison.comparison_name,
                    'created_at': comparison.created_at.isoformat(),
                    'stability_score': comparison.performance_stability_score,
                    'degradation_detected': comparison.model_degradation_indicator
                },
                'backtest_session': {
                    'session_id': backtest_session.session_id,
                    'session_name': backtest_session.session_name,
                    'period': f"{backtest_session.start_date.date()} to {backtest_session.end_date.date()}",
                    'total_bets': backtest_session.total_bets,
                    'winning_bets': backtest_session.winning_bets,
                    'win_rate': backtest_session.win_rate,
                    'net_profit': backtest_session.net_profit,
                    'roi_percentage': backtest_session.roi_percentage,
                    'profit_factor': backtest_session.profit_factor,
                    'max_drawdown': backtest_session.max_drawdown
                },
                'forward_test_session': {
                    'session_id': forward_test_session.session_id,
                    'session_name': forward_test_session.session_name,
                    'period': f"{forward_test_session.start_date.date()} to {forward_test_session.end_date.date()}",
                    'total_bets': forward_test_session.total_bets,
                    'winning_bets': forward_test_session.winning_bets,
                    'win_rate': forward_test_session.win_rate,
                    'net_profit': forward_test_session.net_profit,
                    'roi_percentage': forward_test_session.roi_percentage,
                    'profit_factor': forward_test_session.profit_factor,
                    'max_drawdown': forward_test_session.max_drawdown
                },
                'comparison_metrics': {
                    'roi_difference': comparison.roi_difference,
                    'win_rate_difference': comparison.win_rate_difference,
                    'profit_difference': comparison.profit_difference,
                    'stability_assessment': self._assess_stability(comparison.performance_stability_score),
                    'risk_assessment': self._assess_risk_changes(backtest_session, forward_test_session)
                },
                'analysis_summary': json.loads(comparison.analysis_summary) if comparison.analysis_summary else {},
                'recommendations': json.loads(comparison.recommendations) if comparison.recommendations else []
            }
            
            return detailed_analysis
            
        except Exception as e:
            logger.error(f"❌ Error getting detailed comparison: {e}")
            return {'error': str(e)}
    
    def _assess_stability(self, stability_score: float) -> str:
        """Assess stability based on score"""
        if stability_score >= 90:
            return "Excellent - Model very stable"
        elif stability_score >= 80:
            return "Good - Model mostly stable"
        elif stability_score >= 70:
            return "Fair - Some performance variation"
        elif stability_score >= 60:
            return "Concerning - Notable performance changes"
        else:
            return "Poor - Significant model degradation"
    
    def _assess_risk_changes(self, backtest_session: TestingSession, forward_test_session: TestingSession) -> Dict:
        """Assess changes in risk profile between sessions"""
        risk_assessment = {
            'drawdown_change': 'stable',
            'volatility_change': 'stable',
            'risk_level': 'moderate'
        }
        
        # Drawdown comparison
        backtest_dd = backtest_session.max_drawdown or 0
        forward_dd = forward_test_session.max_drawdown or 0
        dd_change = forward_dd - backtest_dd
        
        if dd_change > 500:  # $500 increase in drawdown
            risk_assessment['drawdown_change'] = 'increased'
        elif dd_change < -500:
            risk_assessment['drawdown_change'] = 'decreased'
        
        # Profit factor comparison
        backtest_pf = backtest_session.profit_factor or 1
        forward_pf = forward_test_session.profit_factor or 1
        
        if forward_pf < 1.0:
            risk_assessment['risk_level'] = 'high'
        elif forward_pf > 2.0:
            risk_assessment['risk_level'] = 'low'
        
        return risk_assessment
    
    def generate_model_stability_report(self, days_back: int = 90) -> Dict:
        """
        Generate comprehensive model stability report
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Get recent comparisons
            recent_comparisons = self.session.query(PerformanceComparison).filter(
                PerformanceComparison.created_at >= cutoff_date
            ).order_by(PerformanceComparison.created_at.desc()).all()
            
            if not recent_comparisons:
                return {
                    'message': f'No performance comparisons found in the last {days_back} days',
                    'report_date': datetime.now().isoformat()
                }
            
            # Analyze trends
            stability_scores = [comp.performance_stability_score for comp in recent_comparisons]
            degradation_count = len([comp for comp in recent_comparisons if comp.model_degradation_indicator])
            
            # Calculate trend
            if len(stability_scores) >= 3:
                recent_scores = stability_scores[:3]
                older_scores = stability_scores[3:6] if len(stability_scores) >= 6 else stability_scores[3:]
                
                if older_scores:
                    trend = np.mean(recent_scores) - np.mean(older_scores)
                    trend_direction = 'improving' if trend > 5 else 'declining' if trend < -5 else 'stable'
                else:
                    trend_direction = 'insufficient_data'
            else:
                trend_direction = 'insufficient_data'
            
            # Overall assessment
            avg_stability = np.mean(stability_scores)
            degradation_rate = (degradation_count / len(recent_comparisons)) * 100
            
            # Generate overall status
            if avg_stability >= 80 and degradation_rate < 20:
                overall_status = 'healthy'
            elif avg_stability >= 70 and degradation_rate < 40:
                overall_status = 'monitoring_required'
            else:
                overall_status = 'intervention_required'
            
            report = {
                'report_period': f'Last {days_back} days',
                'report_date': datetime.now().isoformat(),
                'overall_status': overall_status,
                'summary_metrics': {
                    'total_comparisons': len(recent_comparisons),
                    'average_stability_score': avg_stability,
                    'degradation_incidents': degradation_count,
                    'degradation_rate': degradation_rate,
                    'trend_direction': trend_direction
                },
                'recent_comparisons': [
                    {
                        'comparison_id': comp.comparison_id,
                        'stability_score': comp.performance_stability_score,
                        'degradation_detected': comp.model_degradation_indicator,
                        'roi_difference': comp.roi_difference,
                        'created_at': comp.created_at.isoformat()
                    }
                    for comp in recent_comparisons[:5]  # Last 5 comparisons
                ],
                'recommendations': self._generate_stability_recommendations(
                    overall_status, avg_stability, degradation_rate, trend_direction
                )
            }
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Error generating stability report: {e}")
            return {'error': str(e)}
    
    def _generate_stability_recommendations(self, 
                                          overall_status: str, 
                                          avg_stability: float,
                                          degradation_rate: float,
                                          trend_direction: str) -> List[str]:
        """Generate recommendations based on stability analysis"""
        recommendations = []
        
        if overall_status == 'intervention_required':
            recommendations.extend([
                "🚨 Immediate intervention required",
                "🔄 Retrain ML models with latest data",
                "📊 Comprehensive data quality review",
                "💰 Reduce position sizes significantly",
                "⏸️ Consider temporarily halting automated betting"
            ])
        elif overall_status == 'monitoring_required':
            recommendations.extend([
                "⚠️ Enhanced monitoring required",
                "🔍 Daily performance reviews",
                "📈 Consider model parameter adjustments",
                "💰 Conservative stake sizing"
            ])
        else:
            recommendations.extend([
                "✅ System performing within acceptable parameters",
                "📊 Continue regular monitoring",
                "🔄 Maintain current strategy"
            ])
        
        # Trend-specific recommendations
        if trend_direction == 'declining':
            recommendations.append("📉 Performance trend declining - investigate causes")
        elif trend_direction == 'improving':
            recommendations.append("📈 Performance trend improving - maintain current approach")
        
        # Degradation rate recommendations
        if degradation_rate > 50:
            recommendations.append("🚨 High degradation rate - fundamental review needed")
        elif degradation_rate > 30:
            recommendations.append("⚠️ Elevated degradation rate - closer monitoring required")
        
        return recommendations
    
    def __del__(self):
        """Cleanup database session"""
        if hasattr(self, 'session'):
            self.session.close()


# Example usage and testing
if __name__ == "__main__":
    print("📊 PERFORMANCE COMPARISON SERVICE - TESTING")
    print("=" * 50)
    
    # Initialize service
    service = PerformanceComparisonService()
    
    # Note: In a real scenario, you would have actual session IDs
    # For testing, this shows the interface
    
    print("1️⃣ Testing comparison creation...")
    # comparison_id = service.create_performance_comparison(
    #     comparison_name="Test Comparison",
    #     backtest_session_id="backtest_session_id",
    #     forward_test_session_id="forward_test_session_id"
    # )
    # print(f"📊 Created comparison: {comparison_id}")
    
    print("2️⃣ Testing comparison retrieval...")
    comparisons = service.get_performance_comparisons(limit=5)
    print(f"📋 Retrieved {len(comparisons)} comparisons")
    
    print("3️⃣ Testing stability report...")
    report = service.generate_model_stability_report(days_back=30)
    print(f"📊 Stability report: {report}")
    
    print("\n📊 Performance Comparison Service testing completed!")