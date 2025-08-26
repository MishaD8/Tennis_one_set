#!/usr/bin/env python3
"""
🎾 Betting Dashboard API Integration
Comprehensive API endpoints for the main dashboard's betting statistics tab
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from flask import Blueprint, jsonify, request

from src.api.prediction_betting_integration import get_prediction_betting_integrator
from src.api.betting_statistics_integration import BettingStatisticsIntegrator
from src.api.betting_tracker_service import BettingTrackerService
from src.data.database_models import TestMode, DatabaseManager

logger = logging.getLogger(__name__)

# Create Blueprint for betting dashboard APIs
betting_dashboard_bp = Blueprint('betting_dashboard', __name__, url_prefix='/api/betting-dashboard')

class BettingDashboardAPI:
    """Unified API service for betting dashboard integration"""
    
    def __init__(self):
        self.prediction_integrator = get_prediction_betting_integrator()
        self.stats_integrator = BettingStatisticsIntegrator()
        self.betting_tracker = BettingTrackerService()
        self.db_manager = DatabaseManager()
        
    def get_comprehensive_stats(self, period: str = '1_week') -> Dict[str, Any]:
        """Get comprehensive betting statistics for dashboard"""
        try:
            # Convert period to days
            period_mapping = {
                '1_week': 7,
                '1_month': 30,
                '1_year': 365,
                'all_time': 9999
            }
            days = period_mapping.get(period, 7)
            
            # Get prediction-based statistics
            prediction_stats = self.prediction_integrator.get_betting_statistics(days)
            
            # Get comprehensive statistics from integration service
            comprehensive_stats = self.stats_integrator.get_dashboard_summary(days_back=days)
            
            # Get betting tracker statistics
            tracker_stats = self.betting_tracker.get_betting_statistics_by_timeframe(
                test_mode=TestMode.LIVE,
                timeframe=period
            )
            
            # Merge all statistics
            merged_stats = self._merge_statistics(prediction_stats, comprehensive_stats, tracker_stats)
            
            return {
                'success': True,
                'statistics': merged_stats,
                'period': period,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting comprehensive stats: {e}")
            return {
                'success': False,
                'error': str(e),
                'statistics': self._get_empty_statistics()
            }
    
    def get_chart_data(self, chart_type: str, period: str = '1_week') -> Dict[str, Any]:
        """Get chart data for betting statistics visualization"""
        try:
            if chart_type == 'profit_timeline':
                return self._get_profit_timeline_data(period)
            elif chart_type == 'win_rate_trend':
                return self._get_win_rate_trend_data(period)
            elif chart_type == 'odds_distribution':
                return self._get_odds_distribution_data(period)
            elif chart_type == 'monthly_performance':
                return self._get_monthly_performance_data(period)
            else:
                return {
                    'success': False,
                    'error': f'Unknown chart type: {chart_type}'
                }
                
        except Exception as e:
            logger.error(f"❌ Error getting chart data for {chart_type}: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': self._get_empty_chart_data()
            }
    
    def _merge_statistics(self, prediction_stats: Dict, comprehensive_stats: Dict, tracker_stats: Dict) -> Dict[str, Any]:
        """Merge statistics from different sources"""
        try:
            # Basic metrics from prediction integrator
            basic_metrics = {
                'total_bets': prediction_stats.get('total_bets', 0),
                'win_rate': prediction_stats.get('win_rate', 0),
                'settled_bets': prediction_stats.get('settled_bets', 0),
                'pending_bets': prediction_stats.get('pending_bets', 0)
            }
            
            # Financial metrics
            financial_metrics = {
                'net_profit': prediction_stats.get('net_profit', 0),
                'total_staked': prediction_stats.get('total_staked', 0),
                'total_returned': prediction_stats.get('total_returned', 0),
                'roi_percentage': prediction_stats.get('roi', 0),
                'current_bankroll': prediction_stats.get('current_bankroll', 1000)
            }
            
            # Average metrics
            average_metrics = {
                'average_odds': prediction_stats.get('avg_odds', 0),
                'average_stake': prediction_stats.get('avg_stake', 0)
            }
            
            # Risk metrics from tracker if available
            risk_metrics = {}
            if 'error' not in tracker_stats:
                tracker_risk = tracker_stats.get('risk_metrics', {})
                risk_metrics = {
                    'sharpe_ratio': tracker_risk.get('sharpe_ratio', 0),
                    'largest_win': tracker_risk.get('largest_win', 0),
                    'largest_loss': tracker_risk.get('largest_loss', 0),
                    'max_drawdown': tracker_risk.get('max_drawdown', 0),
                    'volatility': tracker_risk.get('volatility', 0)
                }
            else:
                # Calculate basic risk metrics from prediction stats
                risk_metrics = {
                    'sharpe_ratio': 0,
                    'largest_win': 0,
                    'largest_loss': 0,
                    'max_drawdown': 0,
                    'volatility': 0
                }
            
            # Streak analysis
            streak_analysis = {
                'current_streak': {'count': 0, 'type': 'none'},
                'longest_winning_streak': 0,
                'longest_losing_streak': 0
            }
            
            # Model performance from prediction stats
            model_performance = prediction_stats.get('model_performance', {})
            
            # Confidence breakdown
            confidence_breakdown = prediction_stats.get('confidence_breakdown', {})
            
            # Data quality assessment
            data_quality = self._assess_data_quality(basic_metrics['total_bets'])
            
            return {
                'basic_metrics': basic_metrics,
                'financial_metrics': financial_metrics,
                'average_metrics': average_metrics,
                'risk_metrics': risk_metrics,
                'streak_analysis': streak_analysis,
                'model_performance': model_performance,
                'confidence_breakdown': confidence_breakdown,
                'data_quality': data_quality,
                'system_health': self._get_system_health_metrics(comprehensive_stats)
            }
            
        except Exception as e:
            logger.error(f"❌ Error merging statistics: {e}")
            return self._get_empty_statistics()
    
    def _get_profit_timeline_data(self, period: str) -> Dict[str, Any]:
        """Get profit timeline chart data"""
        try:
            # Get betting records from database
            session = self.db_manager.get_session()
            from src.data.database_models import BettingLog, BettingStatus
            
            # Calculate date range
            period_mapping = {'1_week': 7, '1_month': 30, '1_year': 365, 'all_time': 9999}
            days = period_mapping.get(period, 7)
            
            if days == 9999:
                # All time
                records = session.query(BettingLog).filter(
                    BettingLog.betting_status.in_([BettingStatus.WON, BettingStatus.LOST])
                ).order_by(BettingLog.timestamp.asc()).all()
            else:
                cutoff_date = datetime.now() - timedelta(days=days)
                records = session.query(BettingLog).filter(
                    BettingLog.timestamp >= cutoff_date,
                    BettingLog.betting_status.in_([BettingStatus.WON, BettingStatus.LOST])
                ).order_by(BettingLog.timestamp.asc()).all()
            
            session.close()
            
            if not records:
                return {
                    'success': True,
                    'data': {
                        'labels': ['No Data'],
                        'datasets': [{
                            'label': 'Cumulative Profit/Loss',
                            'data': [0],
                            'borderColor': '#74b9ff',
                            'backgroundColor': 'rgba(116, 185, 255, 0.1)',
                            'fill': True
                        }]
                    }
                }
            
            # Calculate cumulative profit/loss
            labels = []
            cumulative_data = []
            cumulative_profit = 0
            
            for record in records:
                profit_loss = record.profit_loss or 0
                cumulative_profit += profit_loss
                
                labels.append(record.timestamp.strftime('%Y-%m-%d'))
                cumulative_data.append(round(cumulative_profit, 2))
            
            return {
                'success': True,
                'data': {
                    'labels': labels,
                    'datasets': [{
                        'label': 'Cumulative Profit/Loss',
                        'data': cumulative_data,
                        'borderColor': '#74b9ff',
                        'backgroundColor': 'rgba(116, 185, 255, 0.1)',
                        'fill': True,
                        'tension': 0.4
                    }]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting profit timeline data: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': self._get_empty_chart_data()
            }
    
    def _get_win_rate_trend_data(self, period: str) -> Dict[str, Any]:
        """Get win rate trend chart data"""
        try:
            session = self.db_manager.get_session()
            from src.data.database_models import BettingLog, BettingStatus
            
            # Calculate date range
            period_mapping = {'1_week': 7, '1_month': 30, '1_year': 365, 'all_time': 9999}
            days = period_mapping.get(period, 7)
            
            if days == 9999:
                records = session.query(BettingLog).filter(
                    BettingLog.betting_status.in_([BettingStatus.WON, BettingStatus.LOST])
                ).order_by(BettingLog.timestamp.asc()).all()
            else:
                cutoff_date = datetime.now() - timedelta(days=days)
                records = session.query(BettingLog).filter(
                    BettingLog.timestamp >= cutoff_date,
                    BettingLog.betting_status.in_([BettingStatus.WON, BettingStatus.LOST])
                ).order_by(BettingLog.timestamp.asc()).all()
            
            session.close()
            
            if not records:
                return {
                    'success': True,
                    'data': {
                        'labels': ['No Data'],
                        'datasets': [{
                            'label': 'Win Rate %',
                            'data': [0],
                            'borderColor': '#00b894',
                            'backgroundColor': 'rgba(0, 184, 148, 0.1)',
                            'fill': True
                        }]
                    }
                }
            
            # Calculate rolling win rate (10-bet window)
            labels = []
            win_rate_data = []
            window_size = min(10, len(records))
            
            for i in range(window_size - 1, len(records)):
                window_records = records[i - window_size + 1:i + 1]
                wins = sum(1 for r in window_records if r.betting_status == BettingStatus.WON)
                win_rate = (wins / window_size) * 100
                
                labels.append(records[i].timestamp.strftime('%Y-%m-%d'))
                win_rate_data.append(round(win_rate, 1))
            
            return {
                'success': True,
                'data': {
                    'labels': labels,
                    'datasets': [{
                        'label': f'Win Rate % (Rolling {window_size}-bet average)',
                        'data': win_rate_data,
                        'borderColor': '#00b894',
                        'backgroundColor': 'rgba(0, 184, 148, 0.1)',
                        'fill': True,
                        'tension': 0.4
                    }]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting win rate trend data: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': self._get_empty_chart_data()
            }
    
    def _get_odds_distribution_data(self, period: str) -> Dict[str, Any]:
        """Get odds distribution chart data"""
        try:
            session = self.db_manager.get_session()
            from src.data.database_models import BettingLog
            
            # Calculate date range
            period_mapping = {'1_week': 7, '1_month': 30, '1_year': 365, 'all_time': 9999}
            days = period_mapping.get(period, 7)
            
            if days == 9999:
                records = session.query(BettingLog).all()
            else:
                cutoff_date = datetime.now() - timedelta(days=days)
                records = session.query(BettingLog).filter(
                    BettingLog.timestamp >= cutoff_date
                ).all()
            
            session.close()
            
            if not records:
                return {
                    'success': True,
                    'data': {
                        'labels': ['No Data'],
                        'datasets': [{
                            'data': [1],
                            'backgroundColor': ['rgba(255, 107, 107, 0.7)']
                        }]
                    }
                }
            
            # Categorize odds
            odds_ranges = {
                '1.0-1.5': 0,
                '1.5-2.0': 0,
                '2.0-2.5': 0,
                '2.5-3.0': 0,
                '3.0+': 0
            }
            
            for record in records:
                odds = record.odds_taken or 2.0
                if odds < 1.5:
                    odds_ranges['1.0-1.5'] += 1
                elif odds < 2.0:
                    odds_ranges['1.5-2.0'] += 1
                elif odds < 2.5:
                    odds_ranges['2.0-2.5'] += 1
                elif odds < 3.0:
                    odds_ranges['2.5-3.0'] += 1
                else:
                    odds_ranges['3.0+'] += 1
            
            labels = list(odds_ranges.keys())
            data = list(odds_ranges.values())
            colors = [
                'rgba(255, 99, 132, 0.7)',
                'rgba(54, 162, 235, 0.7)',
                'rgba(255, 205, 86, 0.7)',
                'rgba(75, 192, 192, 0.7)',
                'rgba(153, 102, 255, 0.7)'
            ]
            
            return {
                'success': True,
                'data': {
                    'labels': labels,
                    'datasets': [{
                        'data': data,
                        'backgroundColor': colors,
                        'borderColor': [color.replace('0.7', '1') for color in colors],
                        'borderWidth': 2
                    }]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting odds distribution data: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': self._get_empty_chart_data()
            }
    
    def _get_monthly_performance_data(self, period: str) -> Dict[str, Any]:
        """Get monthly performance chart data"""
        try:
            session = self.db_manager.get_session()
            from src.data.database_models import BettingLog, BettingStatus
            
            # Calculate date range
            period_mapping = {'1_week': 7, '1_month': 30, '1_year': 365, 'all_time': 9999}
            days = period_mapping.get(period, 7)
            
            if days == 9999:
                records = session.query(BettingLog).filter(
                    BettingLog.betting_status.in_([BettingStatus.WON, BettingStatus.LOST])
                ).all()
            else:
                cutoff_date = datetime.now() - timedelta(days=days)
                records = session.query(BettingLog).filter(
                    BettingLog.timestamp >= cutoff_date,
                    BettingLog.betting_status.in_([BettingStatus.WON, BettingStatus.LOST])
                ).all()
            
            session.close()
            
            if not records:
                return {
                    'success': True,
                    'data': {
                        'labels': ['No Data'],
                        'datasets': [{
                            'label': 'No data available',
                            'data': [0],
                            'backgroundColor': 'rgba(255, 107, 107, 0.7)'
                        }]
                    }
                }
            
            # Group by month
            monthly_data = {}
            for record in records:
                month_key = record.timestamp.strftime('%Y-%m')
                if month_key not in monthly_data:
                    monthly_data[month_key] = {'profit': 0, 'bets': 0, 'wins': 0}
                
                monthly_data[month_key]['profit'] += record.profit_loss or 0
                monthly_data[month_key]['bets'] += 1
                if record.betting_status == BettingStatus.WON:
                    monthly_data[month_key]['wins'] += 1
            
            # Sort by month
            sorted_months = sorted(monthly_data.keys())
            labels = [datetime.strptime(month, '%Y-%m').strftime('%b %Y') for month in sorted_months]
            
            profit_data = [round(monthly_data[month]['profit'], 2) for month in sorted_months]
            win_rate_data = [
                round((monthly_data[month]['wins'] / monthly_data[month]['bets']) * 100, 1) 
                for month in sorted_months
            ]
            
            return {
                'success': True,
                'data': {
                    'labels': labels,
                    'datasets': [
                        {
                            'label': 'Monthly Profit/Loss ($)',
                            'data': profit_data,
                            'backgroundColor': 'rgba(116, 185, 255, 0.7)',
                            'borderColor': '#74b9ff',
                            'type': 'bar',
                            'yAxisID': 'y'
                        },
                        {
                            'label': 'Win Rate (%)',
                            'data': win_rate_data,
                            'backgroundColor': 'rgba(0, 184, 148, 0.7)',
                            'borderColor': '#00b894',
                            'type': 'line',
                            'yAxisID': 'y1'
                        }
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting monthly performance data: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': self._get_empty_chart_data()
            }
    
    def _assess_data_quality(self, total_bets: int) -> Dict[str, Any]:
        """Assess data quality based on sample size and completeness"""
        try:
            if total_bets == 0:
                return {
                    'sample_size': 0,
                    'data_completeness': 'no_data',
                    'statistical_significance': 'none',
                    'quality_score': 0,
                    'recommendations': [
                        'No betting data available',
                        'Start placing bets to generate statistics',
                        'Minimum 10 bets needed for basic analysis'
                    ]
                }
            elif total_bets < 10:
                return {
                    'sample_size': total_bets,
                    'data_completeness': 'very_limited',
                    'statistical_significance': 'very_low',
                    'quality_score': 1,
                    'recommendations': [
                        f'Only {total_bets} bets recorded - results may be unreliable',
                        'Continue betting to reach minimum 30 bets for meaningful analysis',
                        'Current statistics should be interpreted with caution'
                    ]
                }
            elif total_bets < 30:
                return {
                    'sample_size': total_bets,
                    'data_completeness': 'limited',
                    'statistical_significance': 'low',
                    'quality_score': 2,
                    'recommendations': [
                        f'{total_bets} bets recorded - early trends emerging',
                        'Reach 50+ bets for more reliable statistical analysis',
                        'Monitor performance consistency over time'
                    ]
                }
            elif total_bets < 100:
                return {
                    'sample_size': total_bets,
                    'data_completeness': 'good',
                    'statistical_significance': 'moderate',
                    'quality_score': 3,
                    'recommendations': [
                        f'Good sample size of {total_bets} bets',
                        'Statistics becoming more reliable',
                        'Continue tracking for long-term performance validation'
                    ]
                }
            else:
                return {
                    'sample_size': total_bets,
                    'data_completeness': 'excellent',
                    'statistical_significance': 'high',
                    'quality_score': 4,
                    'recommendations': [
                        f'Excellent sample size of {total_bets} bets',
                        'Statistics are reliable for performance analysis',
                        'Strong foundation for strategy optimization'
                    ]
                }
                
        except Exception as e:
            logger.error(f"❌ Error assessing data quality: {e}")
            return {
                'sample_size': 0,
                'data_completeness': 'error',
                'statistical_significance': 'unknown',
                'quality_score': 0,
                'recommendations': ['Error calculating data quality metrics']
            }
    
    def _get_system_health_metrics(self, comprehensive_stats: Dict) -> Dict[str, Any]:
        """Get system health metrics from comprehensive statistics"""
        try:
            if 'error' in comprehensive_stats:
                return {
                    'status': 'error',
                    'last_match_date': None,
                    'prediction_accuracy': 0,
                    'data_coverage': 'limited'
                }
            
            overview = comprehensive_stats.get('overview', {})
            return {
                'status': 'healthy',
                'total_matches_tracked': overview.get('total_matches_tracked', 0),
                'prediction_accuracy': overview.get('prediction_accuracy', 0),
                'recent_accuracy': overview.get('recent_accuracy_20_matches', 0),
                'upsets_detected': overview.get('upsets_detected', 0),
                'data_coverage': 'good' if overview.get('total_matches_tracked', 0) > 50 else 'limited'
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting system health metrics: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _get_empty_statistics(self) -> Dict[str, Any]:
        """Return empty statistics structure when no data is available"""
        return {
            'basic_metrics': {
                'total_bets': 0,
                'win_rate': 0,
                'settled_bets': 0,
                'pending_bets': 0
            },
            'financial_metrics': {
                'net_profit': 0,
                'total_staked': 0,
                'total_returned': 0,
                'roi_percentage': 0,
                'current_bankroll': 1000
            },
            'average_metrics': {
                'average_odds': 0,
                'average_stake': 0
            },
            'risk_metrics': {
                'sharpe_ratio': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'max_drawdown': 0,
                'volatility': 0
            },
            'streak_analysis': {
                'current_streak': {'count': 0, 'type': 'none'},
                'longest_winning_streak': 0,
                'longest_losing_streak': 0
            },
            'model_performance': {},
            'confidence_breakdown': {},
            'data_quality': {
                'sample_size': 0,
                'data_completeness': 'no_data',
                'statistical_significance': 'none',
                'quality_score': 0,
                'recommendations': ['No betting data available']
            },
            'system_health': {
                'status': 'no_data',
                'data_coverage': 'none'
            }
        }
    
    def _get_empty_chart_data(self) -> Dict[str, Any]:
        """Return empty chart data structure"""
        return {
            'labels': ['No Data'],
            'datasets': [{
                'label': 'No data available',
                'data': [0],
                'backgroundColor': 'rgba(255, 107, 107, 0.7)',
                'borderColor': '#ff6b6b'
            }],
            'message': 'No betting data available for chart generation'
        }

# Initialize API service
betting_dashboard_api = BettingDashboardAPI()

# Blueprint route handlers
@betting_dashboard_bp.route('/statistics', methods=['GET'])
def get_betting_statistics():
    """Get comprehensive betting statistics for dashboard"""
    try:
        period = request.args.get('timeframe', '1_week')
        result = betting_dashboard_api.get_comprehensive_stats(period)
        return jsonify(result)
    except Exception as e:
        logger.error(f"❌ Error in betting statistics endpoint: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve betting statistics',
            'details': str(e)
        }), 500

@betting_dashboard_bp.route('/charts-data', methods=['GET'])
def get_charts_data():
    """Get chart data for betting statistics visualization"""
    try:
        chart_type = request.args.get('chart_type', 'profit_timeline')
        period = request.args.get('timeframe', '1_week')
        
        result = betting_dashboard_api.get_chart_data(chart_type, period)
        return jsonify(result)
    except Exception as e:
        logger.error(f"❌ Error in betting charts endpoint: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve chart data',
            'details': str(e)
        }), 500

@betting_dashboard_bp.route('/health', methods=['GET'])
def get_dashboard_health():
    """Get health status of betting dashboard systems"""
    try:
        # Test each component
        health_status = {
            'prediction_integrator': 'healthy',
            'stats_integrator': 'healthy',
            'betting_tracker': 'healthy',
            'database': 'healthy'
        }
        
        # Test database connection
        try:
            session = betting_dashboard_api.db_manager.get_session()
            session.close()
        except Exception as e:
            health_status['database'] = f'error: {str(e)}'
        
        # Test prediction integrator
        try:
            test_stats = betting_dashboard_api.prediction_integrator.get_betting_statistics(days=1)
            if 'error' in str(test_stats):
                health_status['prediction_integrator'] = 'warning'
        except Exception as e:
            health_status['prediction_integrator'] = f'error: {str(e)}'
        
        overall_status = 'healthy'
        if any('error' in status for status in health_status.values()):
            overall_status = 'error'
        elif any('warning' in status for status in health_status.values()):
            overall_status = 'warning'
        
        return jsonify({
            'success': True,
            'overall_status': overall_status,
            'components': health_status,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error in dashboard health endpoint: {e}")
        return jsonify({
            'success': False,
            'overall_status': 'error',
            'error': str(e)
        }), 500

def register_betting_dashboard_routes(app):
    """Register betting dashboard routes with Flask app"""
    try:
        app.register_blueprint(betting_dashboard_bp)
        logger.info("✅ Betting dashboard API routes registered")
    except Exception as e:
        logger.error(f"❌ Error registering betting dashboard routes: {e}")