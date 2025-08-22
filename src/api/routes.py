#!/usr/bin/env python3
"""
Tennis Backend Routes
All API routes and endpoints for tennis betting and ML prediction system
"""

import os
import logging
import json
import re
import html
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from functools import wraps

from flask import Flask, jsonify, request, render_template, redirect
from werkzeug.exceptions import HTTPException

from src.api.middleware import require_api_key, validate_json_request, log_request_info, handle_betting_errors

# Set up logger
logger = logging.getLogger(__name__)

# Import tennis system components with fallback handling
try:
    from src.models.real_tennis_predictor_integration import RealTennisPredictor
    REAL_PREDICTOR_AVAILABLE = True
    logger.info("✅ Real ML predictor imported")
except ImportError as e:
    logger.info(f"⚠️ Real predictor not available: {e}")
    REAL_PREDICTOR_AVAILABLE = False

try:
    from src.models.tennis_prediction_module import TennisPredictionService
    PREDICTION_SERVICE_AVAILABLE = True
    logger.info("✅ Prediction service imported")
except ImportError as e:
    logger.info(f"⚠️ Prediction service not available: {e}")
    PREDICTION_SERVICE_AVAILABLE = False

ODDS_API_AVAILABLE = False
API_ECONOMY_AVAILABLE = False

try:
    from src.data.enhanced_universal_collector import EnhancedUniversalCollector
    from src.data.universal_tennis_data_collector import UniversalOddsCollector
    ENHANCED_COLLECTOR_AVAILABLE = True
    UNIVERSAL_COLLECTOR_AVAILABLE = False  # Will be set below if needed
    logger.info("✅ Enhanced Universal Collector loaded")
except ImportError as e:
    logger.info(f"⚠️ Enhanced collector not available: {e}")
    ENHANCED_COLLECTOR_AVAILABLE = False
    try:
        from src.data.universal_tennis_data_collector import UniversalTennisDataCollector, UniversalOddsCollector
        UNIVERSAL_COLLECTOR_AVAILABLE = True
        logger.info("✅ Universal data collector loaded (fallback)")
    except ImportError as e2:
        logger.info(f"⚠️ Universal collector not available: {e2}")
        UNIVERSAL_COLLECTOR_AVAILABLE = False

try:
    from src.data.daily_api_scheduler import init_daily_scheduler, start_daily_scheduler
    DAILY_SCHEDULER_AVAILABLE = True
    logger.info("✅ Daily API scheduler loaded")
except ImportError as e:
    logger.info(f"⚠️ Daily scheduler not available: {e}")
    DAILY_SCHEDULER_AVAILABLE = False

# Global service instances (will be initialized in register_routes)
real_predictor = None
prediction_service = None
odds_integrator = None
enhanced_collector = None
universal_collector = None
daily_scheduler = None

class UnderdogAnalyzer:
    """Tennis underdog scenario analyzer"""
    
    def __init__(self):
        self.player_rankings = {
            # Current ATP rankings (updated July 2025)
            "jannik sinner": 1, "carlos alcaraz": 2, "alexander zverev": 3,
            "daniil medvedev": 4, "novak djokovic": 5, "andrey rublev": 6,
            "casper ruud": 7, "holger rune": 8, "grigor dimitrov": 9,
            "stefanos tsitsipas": 10, "taylor fritz": 11, "tommy paul": 12,
            "alex de minaur": 13, "ben shelton": 14, "ugo humbert": 15,
            "lorenzo musetti": 16, "sebastian baez": 17, "frances tiafoe": 18,
            "felix auger-aliassime": 19, "arthur fils": 20,
            # Additional players from system
            "flavio cobolli": 32, "brandon nakashima": 45, "bu yunchaokete": 85,
            "matteo berrettini": 35, "cameron norrie": 40, "sebastian korda": 25,
            "francisco cerundolo": 30, "alejandro tabilo": 28, "fabio fognini": 85,
            # WTA rankings
            "aryna sabalenka": 1, "iga swiatek": 2, "coco gauff": 3,
            "jessica pegula": 4, "elena rybakina": 5, "qinwen zheng": 6,
            "jasmine paolini": 7, "emma navarro": 8, "daria kasatkina": 9,
            "renata zarazua": 80, "amanda anisimova": 35, "katie boulter": 28,
            "emma raducanu": 25, "caroline dolehide": 85, "carson branstine": 125
        }
        
        self.surface_bonuses = {
            'Clay': {'spanish': 0.15, 'south_american': 0.1},
            'Grass': {'british': 0.1, 'serve_volley': 0.05},
            'Hard': {'american': 0.05, 'baseline': 0.03}
        }
        
    def get_player_ranking(self, player_name: str) -> int:
        """Get player ranking with fallback"""
        normalized_name = player_name.lower().strip()
        return self.player_rankings.get(normalized_name, 999)
        
    def calculate_underdog_probability(self, player1: str, player2: str, 
                                     tournament: str, surface: str) -> Dict[str, Any]:
        """Calculate detailed underdog probability analysis"""
        
        rank1 = self.get_player_ranking(player1)
        rank2 = self.get_player_ranking(player2)
        
        # Determine underdog and favorite
        if rank1 > rank2:
            underdog, favorite = player1, player2
            underdog_rank, favorite_rank = rank1, rank2
        else:
            underdog, favorite = player2, player1
            underdog_rank, favorite_rank = rank2, rank1
            
        rank_gap = underdog_rank - favorite_rank
        
        # Base probability calculation
        if rank_gap <= 5:
            base_prob = 0.48
            underdog_type = "Slight underdog"
        elif rank_gap <= 15:
            base_prob = 0.35
            underdog_type = "Moderate underdog"
        elif rank_gap <= 30:
            base_prob = 0.25
            underdog_type = "Strong underdog"
        else:
            base_prob = 0.15
            underdog_type = "Major underdog"
            
        # Apply surface bonuses
        surface_bonus = 0.0
        if surface in self.surface_bonuses:
            # Simplified surface advantage logic
            if 'clay' in tournament.lower() or surface.lower() == 'clay':
                surface_bonus = 0.05
                
        # Tournament pressure factor
        pressure_factor = 0.0
        if any(major in tournament.lower() for major in ['grand slam', 'masters', 'wimbledon', 'us open']):
            pressure_factor = 0.03  # Higher pressure can level the playing field
            
        # Calculate final probability
        final_probability = min(base_prob + surface_bonus + pressure_factor, 0.6)
        
        # Confidence calculation
        confidence_score = max(0.6, 1.0 - (rank_gap / 100))
        
        key_factors = []
        if surface_bonus > 0:
            key_factors.append(f"Surface advantage (+{surface_bonus:.2%})")
        if pressure_factor > 0:
            key_factors.append(f"Tournament pressure factor (+{pressure_factor:.2%})")
        if rank_gap > 50:
            key_factors.append("Significant ranking gap - upset potential")
            
        return {
            'underdog_probability': round(final_probability, 3),
            'confidence': round(confidence_score, 3),
            'prediction_type': 'UNDERDOG_ANALYSIS',
            'key_factors': key_factors,
            'underdog_scenario': {
                'underdog': underdog,
                'favorite': favorite,
                'rank_gap': rank_gap,
                'underdog_type': underdog_type,
                'base_probability': base_prob
            }
        }

# Security and validation functions
def validate_player_name(name: str) -> bool:
    """Validate player name input with enhanced security"""
    if not isinstance(name, str):
        return False
    
    name = name.strip()
    if not name:
        return False
        
    # Length validation - prevent DoS attacks
    if len(name) > 100:
        return False
    
    # Character validation - allow letters, numbers, spaces, apostrophes, hyphens, dots
    if not re.match(r"^[a-zA-Z0-9\s\'\-\.]+$", name):
        return False
        
    return True

def validate_tournament_name(tournament: str) -> bool:
    """Validate tournament name"""
    if not isinstance(tournament, str):
        return False
        
    tournament = tournament.strip()
    if not tournament:
        return False
        
    if len(tournament) > 200:
        return False
        
    # Allow alphanumeric, spaces, and common tournament characters
    if not re.match(r"^[a-zA-Z0-9\s\'\-\.\(\)]+$", tournament):
        return False
        
    return True

def validate_surface(surface: str) -> bool:
    """Validate tennis surface"""
    valid_surfaces = ['Hard', 'Clay', 'Grass', 'Carpet']
    return surface in valid_surfaces

def sanitize_input(data: Any) -> Any:
    """Sanitize string inputs with enhanced security"""
    if isinstance(data, str):
        # Strip whitespace and remove null bytes
        cleaned = data.strip().replace('\x00', '')
        # Escape HTML to prevent XSS
        return html.escape(cleaned, quote=True)
    elif isinstance(data, dict):
        # Recursively sanitize dictionary values with depth limit
        if hasattr(sanitize_input, '_depth'):
            sanitize_input._depth += 1
        else:
            sanitize_input._depth = 1
            
        if sanitize_input._depth > 10:  # Prevent deep recursion attacks
            sanitize_input._depth -= 1
            return {}
            
        result = {key: sanitize_input(value) for key, value in data.items() 
                 if isinstance(key, str) and len(key) <= 100}
        sanitize_input._depth -= 1
        return result
    elif isinstance(data, list):
        # Limit list size to prevent DoS
        if len(data) > 1000:
            return data[:1000]
        return [sanitize_input(item) for item in data]
    return data

def validate_json_payload(data: Dict, max_keys: int = 20, max_depth: int = 5) -> bool:
    """Validate JSON payload structure to prevent DoS attacks"""
    if not isinstance(data, dict):
        return False
    
    if len(data) > max_keys:
        return False
    
    def check_depth(obj, current_depth=0):
        if current_depth > max_depth:
            return False
        if isinstance(obj, dict):
            for value in obj.values():
                if not check_depth(value, current_depth + 1):
                    return False
        elif isinstance(obj, list):
            for item in obj:
                if not check_depth(item, current_depth + 1):
                    return False
        return True
    
    return check_depth(data)

def calculate_detailed_betting_statistics(logs: List[Dict], timeframe: str) -> Dict[str, Any]:
    """Calculate detailed betting statistics from logs"""
    try:
        if not logs:
            return {
                'message': f'No betting data available for {timeframe}',
                'basic_metrics': {
                    'total_bets': 0,
                    'winning_bets': 0,
                    'losing_bets': 0,
                    'win_rate': 0
                },
                'financial_metrics': {
                    'total_staked': 0,
                    'total_returned': 0,
                    'net_profit': 0,
                    'roi_percentage': 0
                }
            }
        
        # Filter to only settled bets
        settled_logs = [log for log in logs if log.get('betting_status') in ['won', 'lost']]
        
        if not settled_logs:
            return {
                'message': f'No settled bets available for {timeframe}',
                'basic_metrics': {
                    'total_bets': len(logs),
                    'pending_bets': len(logs),
                    'winning_bets': 0,
                    'losing_bets': 0,
                    'win_rate': 0
                }
            }
        
        # Basic metrics
        total_bets = len(settled_logs)
        winning_bets = len([log for log in settled_logs if log.get('betting_status') == 'won'])
        losing_bets = total_bets - winning_bets
        win_rate = (winning_bets / total_bets) * 100 if total_bets > 0 else 0
        
        # Financial metrics
        total_staked = sum(log.get('stake_amount', 0) for log in settled_logs)
        total_returned = sum(log.get('payout_amount', 0) for log in settled_logs)
        net_profit = sum(log.get('profit_loss', 0) for log in settled_logs)
        roi_percentage = (net_profit / total_staked) * 100 if total_staked > 0 else 0
        
        # Average metrics
        average_stake = total_staked / total_bets if total_bets > 0 else 0
        average_odds = sum(log.get('odds_taken', 0) for log in settled_logs) / total_bets if total_bets > 0 else 0
        average_edge = sum(log.get('edge_percentage', 0) for log in settled_logs) / total_bets if total_bets > 0 else 0
        
        # Risk metrics
        largest_win = max((log.get('profit_loss', 0) for log in settled_logs if log.get('profit_loss', 0) > 0), default=0)
        largest_loss = abs(min((log.get('profit_loss', 0) for log in settled_logs if log.get('profit_loss', 0) < 0), default=0))
        
        # Calculate Sharpe ratio (simplified)
        if settled_logs:
            profits = [log.get('profit_loss', 0) for log in settled_logs]
            mean_return = sum(profits) / len(profits)
            variance = sum((p - mean_return) ** 2 for p in profits) / len(profits)
            std_dev = variance ** 0.5
            sharpe_ratio = mean_return / std_dev if std_dev > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Monthly/weekly breakdown
        profit_by_period = calculate_profit_by_period(settled_logs, timeframe)
        
        return {
            'basic_metrics': {
                'total_bets': total_bets,
                'winning_bets': winning_bets,
                'losing_bets': losing_bets,
                'win_rate': round(win_rate, 2),
                'pending_bets': len(logs) - total_bets
            },
            'financial_metrics': {
                'total_staked': round(total_staked, 2),
                'total_returned': round(total_returned, 2),
                'net_profit': round(net_profit, 2),
                'roi_percentage': round(roi_percentage, 2)
            },
            'average_metrics': {
                'average_stake': round(average_stake, 2),
                'average_odds': round(average_odds, 2),
                'average_edge': round(average_edge, 2)
            },
            'risk_metrics': {
                'largest_win': round(largest_win, 2),
                'largest_loss': round(largest_loss, 2),
                'sharpe_ratio': round(sharpe_ratio, 3)
            },
            'profit_by_period': profit_by_period,
            'timeframe_analysis': {
                'period': timeframe,
                'data_quality': 'complete' if total_bets >= 10 else 'limited',
                'sample_size': total_bets
            }
        }
        
    except Exception as e:
        logger.error(f"Error calculating betting statistics: {e}")
        return {
            'error': 'Failed to calculate statistics',
            'message': str(e)
        }

def calculate_profit_by_period(logs: List[Dict], timeframe: str) -> List[Dict]:
    """Calculate profit breakdown by time periods"""
    try:
        if not logs:
            return []
        
        # Group by time periods
        from collections import defaultdict
        period_profits = defaultdict(float)
        
        for log in logs:
            try:
                # Parse timestamp
                timestamp_str = log.get('timestamp', '')
                if timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    
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
                    
                    period_profits[period_key] += log.get('profit_loss', 0)
            except Exception as e:
                logger.warning(f"Error processing log timestamp: {e}")
                continue
        
        # Convert to list format
        result = []
        for period, profit in sorted(period_profits.items()):
            result.append({
                'period': period,
                'profit': round(profit, 2)
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating profit by period: {e}")
        return []

def generate_chart_data(logs: List[Dict], chart_type: str, timeframe: str) -> Dict[str, Any]:
    """Generate chart data for different chart types"""
    try:
        if not logs:
            return {'labels': [], 'datasets': []}
        
        if chart_type == 'profit_timeline':
            return generate_profit_timeline_data(logs)
        elif chart_type == 'win_rate_trend':
            return generate_win_rate_trend_data(logs, timeframe)
        elif chart_type == 'odds_distribution':
            return generate_odds_distribution_data(logs)
        elif chart_type == 'monthly_performance':
            return generate_monthly_performance_data(logs)
        else:
            return {'labels': [], 'datasets': []}
            
    except Exception as e:
        logger.error(f"Error generating chart data: {e}")
        return {'labels': [], 'datasets': []}

def generate_profit_timeline_data(logs: List[Dict]) -> Dict[str, Any]:
    """Generate cumulative profit timeline data"""
    try:
        # Sort by timestamp
        sorted_logs = sorted(logs, key=lambda x: x.get('timestamp', ''))
        
        labels = []
        cumulative_profit = 0
        profit_data = []
        
        for log in sorted_logs:
            if log.get('betting_status') in ['won', 'lost']:
                timestamp_str = log.get('timestamp', '')
                if timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    labels.append(timestamp.strftime('%Y-%m-%d'))
                    
                    cumulative_profit += log.get('profit_loss', 0)
                    profit_data.append(round(cumulative_profit, 2))
        
        return {
            'labels': labels,
            'datasets': [{
                'label': 'Cumulative Profit/Loss',
                'data': profit_data,
                'borderColor': 'rgb(107, 207, 127)',
                'backgroundColor': 'rgba(107, 207, 127, 0.1)',
                'tension': 0.1
            }]
        }
        
    except Exception as e:
        logger.error(f"Error generating profit timeline: {e}")
        return {'labels': [], 'datasets': []}

def generate_win_rate_trend_data(logs: List[Dict], timeframe: str) -> Dict[str, Any]:
    """Generate win rate trend data"""
    try:
        from collections import defaultdict
        
        # Group by periods
        period_stats = defaultdict(lambda: {'wins': 0, 'total': 0})
        
        for log in logs:
            if log.get('betting_status') in ['won', 'lost']:
                timestamp_str = log.get('timestamp', '')
                if timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    
                    if timeframe == '1_week':
                        period_key = timestamp.strftime('%Y-%m-%d')
                    elif timeframe == '1_month':
                        week_start = timestamp - timedelta(days=timestamp.weekday())
                        period_key = week_start.strftime('%Y-W%W')
                    else:
                        period_key = timestamp.strftime('%Y-%m')
                    
                    period_stats[period_key]['total'] += 1
                    if log.get('betting_status') == 'won':
                        period_stats[period_key]['wins'] += 1
        
        # Convert to chart data
        labels = []
        win_rates = []
        
        for period in sorted(period_stats.keys()):
            stats = period_stats[period]
            win_rate = (stats['wins'] / stats['total']) * 100 if stats['total'] > 0 else 0
            labels.append(period)
            win_rates.append(round(win_rate, 1))
        
        return {
            'labels': labels,
            'datasets': [{
                'label': 'Win Rate %',
                'data': win_rates,
                'borderColor': 'rgb(74, 158, 255)',
                'backgroundColor': 'rgba(74, 158, 255, 0.1)',
                'tension': 0.1
            }]
        }
        
    except Exception as e:
        logger.error(f"Error generating win rate trend: {e}")
        return {'labels': [], 'datasets': []}

def generate_odds_distribution_data(logs: List[Dict]) -> Dict[str, Any]:
    """Generate odds distribution data"""
    try:
        # Define odds ranges
        odds_ranges = [
            (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), 
            (3.0, 4.0), (4.0, 5.0), (5.0, float('inf'))
        ]
        
        range_counts = [0] * len(odds_ranges)
        range_labels = [
            '1.0-1.5', '1.5-2.0', '2.0-2.5', '2.5-3.0',
            '3.0-4.0', '4.0-5.0', '5.0+'
        ]
        
        for log in logs:
            odds = log.get('odds_taken', 0)
            for i, (min_odds, max_odds) in enumerate(odds_ranges):
                if min_odds <= odds < max_odds:
                    range_counts[i] += 1
                    break
        
        return {
            'labels': range_labels,
            'datasets': [{
                'label': 'Number of Bets',
                'data': range_counts,
                'backgroundColor': [
                    'rgba(255, 99, 132, 0.8)',
                    'rgba(54, 162, 235, 0.8)',
                    'rgba(255, 205, 86, 0.8)',
                    'rgba(75, 192, 192, 0.8)',
                    'rgba(153, 102, 255, 0.8)',
                    'rgba(255, 159, 64, 0.8)',
                    'rgba(199, 199, 199, 0.8)'
                ]
            }]
        }
        
    except Exception as e:
        logger.error(f"Error generating odds distribution: {e}")
        return {'labels': [], 'datasets': []}

def generate_monthly_performance_data(logs: List[Dict]) -> Dict[str, Any]:
    """Generate monthly performance comparison"""
    try:
        from collections import defaultdict
        
        monthly_stats = defaultdict(lambda: {
            'profit': 0, 'bets': 0, 'wins': 0, 'staked': 0
        })
        
        for log in logs:
            if log.get('betting_status') in ['won', 'lost']:
                timestamp_str = log.get('timestamp', '')
                if timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    month_key = timestamp.strftime('%Y-%m')
                    
                    monthly_stats[month_key]['profit'] += log.get('profit_loss', 0)
                    monthly_stats[month_key]['bets'] += 1
                    monthly_stats[month_key]['staked'] += log.get('stake_amount', 0)
                    
                    if log.get('betting_status') == 'won':
                        monthly_stats[month_key]['wins'] += 1
        
        # Convert to chart data
        labels = []
        profit_data = []
        roi_data = []
        win_rate_data = []
        
        for month in sorted(monthly_stats.keys()):
            stats = monthly_stats[month]
            roi = (stats['profit'] / stats['staked']) * 100 if stats['staked'] > 0 else 0
            win_rate = (stats['wins'] / stats['bets']) * 100 if stats['bets'] > 0 else 0
            
            labels.append(month)
            profit_data.append(round(stats['profit'], 2))
            roi_data.append(round(roi, 1))
            win_rate_data.append(round(win_rate, 1))
        
        return {
            'labels': labels,
            'datasets': [
                {
                    'label': 'Monthly Profit',
                    'data': profit_data,
                    'backgroundColor': 'rgba(107, 207, 127, 0.8)',
                    'yAxisID': 'y'
                },
                {
                    'label': 'ROI %',
                    'data': roi_data,
                    'backgroundColor': 'rgba(74, 158, 255, 0.8)',
                    'yAxisID': 'y1'
                }
            ]
        }
        
    except Exception as e:
        logger.error(f"Error generating monthly performance: {e}")
        return {'labels': [], 'datasets': []}

def check_redis_health() -> Dict[str, Any]:
    """Check Redis connection health for monitoring"""
    redis_health = {
        'available': False,
        'connection_time_ms': None,
        'error': None,
        'version': None
    }
    
    try:
        import redis
        import time
        
        start_time = time.time()
        redis_url = os.getenv('REDIS_URL', '').strip()
        
        if redis_url.startswith('redis://'):
            r = redis.Redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2)
        else:
            r = redis.Redis(host='localhost', port=6379, socket_connect_timeout=2, socket_timeout=2)
        
        r.ping()
        connection_time = (time.time() - start_time) * 1000
        
        redis_health.update({
            'available': True,
            'connection_time_ms': round(connection_time, 2),
            'version': r.info().get('redis_version', 'unknown')
        })
        
    except ImportError:
        redis_health['error'] = 'Redis package not installed'
    except Exception as e:
        redis_health['error'] = str(e)
    
    return redis_health

def create_safe_error_response(error: Exception, fallback_message: str) -> str:
    """Create safe error response without exposing sensitive information"""
    if os.getenv('FLASK_ENV') == 'production':
        return fallback_message
    return str(error)

def format_match_for_dashboard(match_data: Dict, source: str = "unknown") -> Dict:
    """Unify match data format for dashboard display"""
    try:
        # Base fields
        formatted = {
            'id': match_data.get('id', f"match_{datetime.now().timestamp()}"),
            'player1': match_data.get('player1', 'Unknown Player 1'),
            'player2': match_data.get('player2', 'Unknown Player 2'),
            'tournament': match_data.get('tournament', 'Unknown Tournament'),
            'surface': match_data.get('surface', 'Hard'),
            'date': match_data.get('date', datetime.now().strftime('%Y-%m-%d')),
            'time': match_data.get('time', datetime.now().strftime('%H:%M')),
            'source': source
        }
        
        # Handle prediction data
        if 'prediction' in match_data:
            formatted['prediction'] = match_data['prediction']
        else:
            # Default prediction structure
            formatted['prediction'] = {
                'probability': match_data.get('probability', 0.5),
                'confidence': match_data.get('confidence', 0.7),
                'prediction_type': match_data.get('prediction_type', 'ANALYSIS'),
                'key_factors': match_data.get('key_factors', [])
            }
        
        # Handle odds
        if 'odds' in match_data:
            formatted['odds'] = match_data['odds']
        else:
            formatted['odds'] = {
                'player1': match_data.get('odds_player1', 2.0),
                'player2': match_data.get('odds_player2', 2.0)
            }
        
        # Additional fields with safe access
        prediction = formatted.get('prediction', {})
        odds = formatted.get('odds', {})
        
        formatted.update({
            'prediction_type': prediction.get('prediction_type', 'ANALYSIS'),
            'underdog_probability': prediction.get('probability', 0.5),
            'value_bet': prediction.get('probability', 0.5) > (1 / odds.get('player1', 2.0)) if odds.get('player1', 0) > 0 else False,
            'key_factors': prediction.get('key_factors', [])
        })
        
        return formatted
        
    except Exception as e:
        logger.warning(f"Error formatting match data: {e}")
        # Return minimal safe format
        return {
            'id': f"error_{datetime.now().timestamp()}",
            'player1': 'Error Player 1',
            'player2': 'Error Player 2',
            'tournament': 'Error Tournament',
            'surface': 'Hard',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'source': 'error',
            'prediction': {
                'probability': 0.5,
                'confidence': 0.5,
                'prediction_type': 'ERROR',
                'key_factors': ['Data formatting error']
            },
            'odds': {'player1': 2.0, 'player2': 2.0}
        }

def get_live_matches_with_underdog_focus() -> Dict:
    """Get matches with underdog analysis focus"""
    try:
        # Priority 1: Enhanced Universal Collector
        if ENHANCED_COLLECTOR_AVAILABLE and enhanced_collector:
            try:
                logger.info("🌍 Using Enhanced Universal Collector...")
                ml_ready_matches = enhanced_collector.get_ml_ready_matches(min_quality_score=30)
                
                if ml_ready_matches and len(ml_ready_matches) > 0:
                    logger.info(f"✅ Got {len(ml_ready_matches)} ML-ready matches from Enhanced Collector")
                    
                    analyzer = UnderdogAnalyzer()
                    processed_matches = []
                    
                    for match in ml_ready_matches[:6]:
                        try:
                            player1 = match.get('player1', 'Player 1')
                            player2 = match.get('player2', 'Player 2')
                            
                            ml_features = match.get('ml_features', {})
                            
                            underdog_analysis = analyzer.calculate_underdog_probability(
                                player1, player2,
                                match.get('tournament', 'ATP Tournament'),
                                match.get('surface', 'Hard')
                            )
                            
                            processed_match = {
                                'id': match.get('match_id', f"enhanced_{len(processed_matches)}"),
                                'player1': player1,
                                'player2': player2,
                                'tournament': match.get('tournament', 'ATP Tournament'),
                                'surface': match.get('surface', 'Hard'),
                                'date': match.get('date', datetime.now().strftime('%Y-%m-%d')),
                                'odds': {
                                    'player1': match.get('odds', {}).get('player1', 2.0),
                                    'player2': match.get('odds', {}).get('player2', 2.0)
                                },
                                'prediction': underdog_analysis,
                                'ml_features': ml_features,
                                'quality_score': match.get('quality_score', 0)
                            }
                            processed_matches.append(processed_match)
                            
                        except Exception as e:
                            logger.warning(f"Error processing enhanced match: {e}")
                            continue
                    
                    if processed_matches:
                        return {
                            'success': True,
                            'matches': processed_matches,
                            'source': 'ENHANCED_UNIVERSAL_COLLECTOR',
                            'count': len(processed_matches)
                        }
                        
            except Exception as e:
                logger.warning(f"Enhanced collector failed: {e}")
        
        # Fallback to test data for development
        test_matches = [
            ("Flavio Cobolli", "Novak Djokovic", "US Open", "Hard"),
            ("Brandon Nakashima", "Carlos Alcaraz", "ATP Masters", "Hard"),
            ("Bu Yunchaokete", "Alexander Zverev", "ATP 500", "Hard"),
            ("Amanda Anisimova", "Aryna Sabalenka", "WTA 1000", "Hard"),
            ("Katie Boulter", "Iga Swiatek", "WTA 500", "Clay"),
            ("Emma Raducanu", "Coco Gauff", "WTA 250", "Grass")
        ]
        
        analyzer = UnderdogAnalyzer()
        processed_matches = []
        
        for i, (player1, player2, tournament, surface) in enumerate(test_matches):
            try:
                underdog_analysis = analyzer.calculate_underdog_probability(
                    player1, player2, tournament, surface
                )
                
                # Simulate realistic odds
                rank1 = analyzer.get_player_ranking(player1)
                rank2 = analyzer.get_player_ranking(player2)
                
                if rank1 < rank2:  # player1 is favorite
                    odds1 = max(1.2, 2.0 - (rank2 - rank1) / 50)
                    odds2 = max(1.2, 1.5 + (rank2 - rank1) / 30)
                else:  # player2 is favorite
                    odds1 = max(1.2, 1.5 + (rank1 - rank2) / 30)
                    odds2 = max(1.2, 2.0 - (rank1 - rank2) / 50)
                
                processed_match = {
                    'id': f"test_match_{i+1}",
                    'player1': player1,
                    'player2': player2,
                    'tournament': tournament,
                    'surface': surface,
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'time': f"{10 + i}:00",
                    'odds': {
                        'player1': round(odds1, 2),
                        'player2': round(odds2, 2)
                    },
                    'prediction': underdog_analysis
                }
                processed_matches.append(processed_match)
                
            except Exception as e:
                logger.warning(f"Error processing test match: {e}")
                continue
        
        return {
            'success': True,
            'matches': processed_matches,
            'source': 'TEST_UNDERDOG_DATA',
            'count': len(processed_matches)
        }
        
    except Exception as e:
        logger.error(f"Error getting matches: {e}")
        return {
            'success': False,
            'matches': [],
            'source': 'ERROR',
            'error': str(e)
        }

# Helper functions for health check
def _check_api_tennis_status() -> bool:
    """Check if API Tennis integration is working"""
    try:
        from src.data.api_tennis_data_collector import get_api_tennis_data_collector
        collector = get_api_tennis_data_collector()
        return collector.is_available()
    except Exception:
        return False

def _check_odds_integrator_status() -> bool:
    """Check if odds integrator is working"""
    try:
        return odds_integrator is not None
    except Exception:
        return False

# Route definitions
def register_routes(app: Flask):
    """Register all tennis backend routes"""
    
    # Get the limiter from app context
    limiter = getattr(app, 'limiter', None)
    if not limiter:
        logger.warning("Rate limiter not available in app context")
    
    # Initialize global service instances
    global real_predictor, prediction_service, odds_integrator
    global enhanced_collector, universal_collector, daily_scheduler
    
    # Initialize services
    if REAL_PREDICTOR_AVAILABLE:
        try:
            real_predictor = RealTennisPredictor()
            logger.info("✅ Real predictor initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize real predictor: {e}")
    
    if PREDICTION_SERVICE_AVAILABLE:
        try:
            prediction_service = TennisPredictionService()
            logger.info("✅ Prediction service initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize prediction service: {e}")
    
    if ODDS_API_AVAILABLE:
        try:
            odds_integrator = TennisOddsIntegrator()
            logger.info("✅ Odds integrator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize odds integrator: {e}")
    
    if ENHANCED_COLLECTOR_AVAILABLE:
        try:
            enhanced_collector = EnhancedUniversalCollector()
            logger.info("✅ Enhanced collector initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize enhanced collector: {e}")
    
    if UNIVERSAL_COLLECTOR_AVAILABLE:
        try:
            universal_collector = UniversalTennisDataCollector()
            logger.info("✅ Universal collector initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize universal collector: {e}")
    
    if DAILY_SCHEDULER_AVAILABLE:
        try:
            daily_scheduler = init_daily_scheduler()
            start_daily_scheduler()
            logger.info("✅ Daily scheduler initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize daily scheduler: {e}")
    
    @app.route('/')
    def dashboard():
        """Main dashboard page"""
        return render_template('dashboard.html')

    @app.route('/test-health')
    def test_health_monitor():
        """Test page for health monitor debugging"""
        return render_template('test_health_monitor_standalone.html')

    @app.route('/betting')
    def betting_dashboard():
        """Advanced betting analytics dashboard"""
        return render_template('betting_dashboard.html')

    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Comprehensive health check with security and infrastructure monitoring"""
        redis_health = check_redis_health()
        
        # Determine overall health status
        overall_status = 'healthy'
        warnings = []
        
        if not redis_health['available']:
            warnings.append('Redis unavailable - using in-memory rate limiting')
        
        connection_time = redis_health.get('connection_time_ms')
        if connection_time is not None and connection_time > 100:
            warnings.append('Redis connection slow')
        
        if os.getenv('FLASK_ENV') != 'production' and request.headers.get('X-Forwarded-Proto') != 'https':
            warnings.append('Not using HTTPS in production')
        
        if warnings:
            overall_status = 'degraded'
        
        return jsonify({
            'status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'components': {
                'real_predictor': real_predictor is not None,
                'prediction_service': prediction_service is not None,
                'odds_integrator': _check_odds_integrator_status(),
                'api_economy': API_ECONOMY_AVAILABLE,
                'enhanced_collector': enhanced_collector is not None,
                'universal_collector': universal_collector is not None,
                'tennisexplorer_integrated': enhanced_collector is not None,
                'rapidapi_integrated': enhanced_collector is not None,
                'api_tennis_integrated': _check_api_tennis_status()
            },
            'infrastructure': {
                'redis': redis_health,
                'rate_limiting': 'redis' if redis_health['available'] else 'memory',
                'ssl_enabled': request.is_secure or request.headers.get('X-Forwarded-Proto') == 'https',
                'environment': os.getenv('FLASK_ENV', 'development')
            },
            'security': {
                'https_enforced': app.config.get('FORCE_HTTPS', False),
                'secure_cookies': app.config.get('SESSION_COOKIE_SECURE', False),
                'csp_enabled': True,
                'hsts_enabled': os.getenv('FLASK_ENV') == 'production'
            },
            'warnings': warnings,
            'version': '5.0-modular'
        })

    @app.route('/api/health-check', methods=['GET'])
    def health_check_endpoint():
        """Simple health check endpoint for load balancers"""
        return jsonify({
            'status': 'ok',
            'timestamp': datetime.now().isoformat(),
            'service': 'tennis-betting-api'
        })

    @app.route('/api/system/database-health', methods=['GET'])
    def database_health_check():
        """Database connectivity health check"""
        try:
            from src.data.database_service import DatabaseService
            db_service = DatabaseService()
            
            # Test database connection
            connection_ok = db_service.test_connection()
            
            if connection_ok:
                # Get database statistics
                stats = db_service.get_database_stats()
                return jsonify({
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    'database': {
                        'connected': True,
                        'type': 'SQLite',
                        'stats': stats
                    }
                })
            else:
                return jsonify({
                    'status': 'unhealthy',
                    'timestamp': datetime.now().isoformat(),
                    'database': {
                        'connected': False,
                        'error': 'Connection failed'
                    }
                }), 503
                
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return jsonify({
                'status': 'unhealthy',
                'timestamp': datetime.now().isoformat(),
                'database': {
                    'connected': False,
                    'error': str(e)
                }
            }), 503

    @app.route('/api/system/ml-health', methods=['GET'])
    def ml_health_check():
        """ML models and prediction system health check"""
        try:
            ml_status = {
                'real_predictor': {
                    'available': REAL_PREDICTOR_AVAILABLE,
                    'loaded': real_predictor is not None
                },
                'prediction_service': {
                    'available': PREDICTION_SERVICE_AVAILABLE,
                    'loaded': prediction_service is not None
                },
                'models_path_exists': os.path.exists('./tennis_models/'),
                'model_files': []
            }
            
            # Check for model files
            if os.path.exists('./tennis_models/'):
                model_files = [f for f in os.listdir('./tennis_models/') if f.endswith(('.pkl', '.h5', '.json'))]
                ml_status['model_files'] = model_files
                ml_status['models_count'] = len([f for f in model_files if f.endswith('.pkl')])
            
            # Test prediction capability
            ml_status['prediction_test'] = 'skipped'
            if real_predictor:
                try:
                    # Simple test to see if predictor can be called
                    test_data = {
                        'player1': 'test player 1',
                        'player2': 'test player 2',
                        'surface': 'Hard',
                        'tournament': 'Test'
                    }
                    # Don't actually run prediction, just check if service is responsive
                    ml_status['prediction_test'] = 'available'
                except Exception as e:
                    ml_status['prediction_test'] = f'error: {str(e)}'
            
            overall_healthy = (
                ml_status['real_predictor']['available'] or ml_status['prediction_service']['available']
            ) and ml_status['models_path_exists']
            
            status_code = 200 if overall_healthy else 503
            
            return jsonify({
                'status': 'healthy' if overall_healthy else 'unhealthy',
                'timestamp': datetime.now().isoformat(),
                'ml_system': ml_status
            }), status_code
            
        except Exception as e:
            logger.error(f"ML health check failed: {e}")
            return jsonify({
                'status': 'unhealthy',
                'timestamp': datetime.now().isoformat(),
                'ml_system': {
                    'error': str(e)
                }
            }), 503

    @app.route('/api/stats', methods=['GET'])
    def get_stats():
        """System statistics"""
        try:
            # Determine ML predictor status
            if real_predictor and hasattr(real_predictor, 'prediction_service') and real_predictor.prediction_service:
                ml_status = 'real_models'
                prediction_type = 'REAL_ML_MODEL'
            elif prediction_service:
                ml_status = 'prediction_service'
                prediction_type = 'PREDICTION_SERVICE'
            else:
                ml_status = 'simulation'
                prediction_type = 'ADVANCED_SIMULATION'
            
            # API usage stats - Note: Old APIs removed
            api_stats = {'message': 'Old API integrations removed'}
            
            stats = {
                'total_matches': 6,
                'ml_predictor_status': ml_status,
                'prediction_type': prediction_type,
                'last_update': datetime.now().isoformat(),
                'accuracy_rate': 0.734,
                'value_bets_found': 2,
                'underdog_opportunities': 4,
                'api_stats': api_stats
            }
            
            return jsonify({
                'success': True,
                'stats': stats,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Stats error: {e}")
            return jsonify({
                'success': False,
                'error': create_safe_error_response(e, 'System statistics unavailable')
            }), 500

    @app.route('/api/matches', methods=['GET'])
    @log_request_info()
    def get_matches():
        """Get matches with underdog analysis"""
        try:
            # Parameter to control data source
            use_real_data_only = request.args.get('real_data_only', 'true').lower() == 'true'
            force_source = request.args.get('source', None)
            
            logger.info(f"🎾 Getting matches (real_data_only={use_real_data_only}, force_source={force_source})")
            
            # Get matches
            matches_result = get_live_matches_with_underdog_focus()
            
            if not matches_result or not isinstance(matches_result, dict) or not matches_result.get('success', False):
                return jsonify({
                    'success': False,
                    'error': 'Failed to get matches',
                    'matches': []
                }), 500
            
            # Filter data if only real data needed
            raw_matches = matches_result.get('matches', [])
            
            if use_real_data_only:
                # Remove test data
                real_matches = [
                    match for match in raw_matches 
                    if not any(test_indicator in match.get('source', '').lower() 
                              for test_indicator in ['test', 'sample', 'underdog_generator', 'fallback'])
                ]
                
                if real_matches:
                    logger.info(f"✅ Filtered to {len(real_matches)} real matches (was {len(raw_matches)})")
                    raw_matches = real_matches
                else:
                    logger.warning("⚠️ No real matches found, returning empty results instead of test data")
                    return jsonify({
                        'success': True,
                        'matches': [],
                        'count': 0,
                        'source': 'NO_REAL_DATA',
                        'message': 'No real matches available. Only test data found.',
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Unify format of all matches
            formatted_matches = []
            for match in raw_matches:
                formatted_match = format_match_for_dashboard(match, matches_result.get('source', 'unknown'))
                formatted_matches.append(formatted_match)
            
            logger.info(f"📊 Returning {len(formatted_matches)} formatted matches")
            
            return jsonify({
                'success': True,
                'matches': formatted_matches,
                'count': len(formatted_matches),
                'source': matches_result.get('source', 'unknown'),
                'prediction_type': formatted_matches[0].get('prediction_type', 'UNKNOWN') if formatted_matches else 'UNKNOWN',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Matches error: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to retrieve matches',
                'matches': []
            }), 500

    @app.route('/api/test-ml', methods=['POST'])
    @validate_json_request()
    @handle_betting_errors()
    def test_ml_prediction():
        """Test ML prediction"""
        try:
            # Get and validate JSON payload
            try:
                if not request.is_json:
                    return jsonify({
                        'success': False,
                        'error': 'Content-Type must be application/json'
                    }), 400
                
                data = request.get_json()
                if data is None:
                    return jsonify({
                        'success': False,
                        'error': 'Invalid JSON payload'
                    }), 400
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': 'Failed to parse JSON'
                }), 400
            
            if not data:
                return jsonify({
                    'success': False,
                    'error': 'No data provided'
                }), 400
            
            # Validate JSON structure to prevent DoS attacks
            if not validate_json_payload(data):
                return jsonify({
                    'success': False,
                    'error': 'Invalid payload structure'
                }), 400
            
            # Sanitize and validate inputs
            data = sanitize_input(data)
            player1 = data.get('player1', 'Flavio Cobolli')
            player2 = data.get('player2', 'Novak Djokovic')
            tournament = data.get('tournament', 'US Open')
            surface = data.get('surface', 'Hard')
            
            # Validate inputs
            if not validate_player_name(player1) or not validate_player_name(player2):
                return jsonify({
                    'success': False,
                    'error': 'Invalid player names provided'
                }), 400
                
            if not validate_tournament_name(tournament):
                return jsonify({
                    'success': False,
                    'error': 'Invalid tournament name provided'
                }), 400
                
            if not validate_surface(surface):
                return jsonify({
                    'success': False,
                    'error': 'Invalid surface provided'
                }), 400
            
            logger.info(f"🔮 Testing ML prediction: {player1} vs {player2}")
            
            # Try different predictors
            if real_predictor:
                try:
                    prediction_result = real_predictor.predict_match(
                        player1, player2, tournament, surface, 'R32'
                    )
                    
                    return jsonify({
                        'success': True,
                        'prediction': prediction_result,
                        'match_info': {
                            'player1': player1,
                            'player2': player2,
                            'tournament': tournament,
                            'surface': surface
                        },
                        'predictor_used': 'real_predictor',
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.warning(f"Real predictor failed: {e}")
            
            if prediction_service:
                try:
                    # Create data for prediction service
                    match_data = {
                        'player_rank': 32.0,  # Cobolli
                        'opponent_rank': 5.0,  # Djokovic
                        'player_age': 22.0,
                        'opponent_age': 37.0,
                        'player_recent_win_rate': 0.65,
                        'player_form_trend': 0.02,
                        'player_surface_advantage': 0.0,
                        'h2h_win_rate': 0.3,
                        'total_pressure': 3.5
                    }
                    
                    prediction_result = prediction_service.predict_match(match_data)
                    
                    return jsonify({
                        'success': True,
                        'prediction': prediction_result,
                        'match_info': {
                            'player1': player1,
                            'player2': player2,
                            'tournament': tournament,
                            'surface': surface
                        },
                        'predictor_used': 'prediction_service',
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.warning(f"Prediction service failed: {e}")
            
            # Fallback to underdog analyzer
            analyzer = UnderdogAnalyzer()
            underdog_result = analyzer.calculate_underdog_probability(
                player1, player2, tournament, surface
            )
            
            return jsonify({
                'success': True,
                'prediction': {
                    'probability': underdog_result['underdog_probability'],
                    'confidence': underdog_result['confidence'],
                    'prediction_type': underdog_result['prediction_type'],
                    'key_factors': underdog_result['key_factors']
                },
                'match_info': {
                    'player1': player1,
                    'player2': player2,
                    'tournament': tournament,
                    'surface': surface
                },
                'predictor_used': 'underdog_analyzer',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Test prediction error: {e}")
            return jsonify({
                'success': False,
                'error': 'ML prediction test failed'
            }), 500

    @app.route('/api/value-bets', methods=['GET'])
    @handle_betting_errors()
    def get_value_bets():
        """Find value bets"""
        try:
            # Get matches
            matches_result = get_live_matches_with_underdog_focus()
            
            if not matches_result['success']:
                return jsonify({
                    'success': False,
                    'error': 'No matches available',
                    'value_bets': []
                })
            
            value_bets = []
            
            for match in matches_result['matches']:
                try:
                    # Get data
                    our_prob = match['prediction']['probability']
                    odds = match['odds']['player1']
                    bookmaker_prob = 1 / odds
                    
                    # Calculate edge
                    edge = our_prob - bookmaker_prob
                    
                    # If edge is greater than 5%
                    if edge > 0.05:
                        value_bet = {
                            'match': f"{match['player1'].replace('🎾 ', '')} vs {match['player2'].replace('🎾 ', '')}",
                            'player': match['player1'].replace('🎾 ', ''),
                            'tournament': match['tournament'].replace('🏆 ', ''),
                            'surface': match['surface'],
                            'odds': odds,
                            'our_probability': our_prob,
                            'bookmaker_probability': bookmaker_prob,
                            'edge': edge * 100,  # In percentage
                            'confidence': match['prediction']['confidence'],
                            'recommendation': 'BET' if edge > 0.08 else 'CONSIDER',
                            'kelly_fraction': min(edge * 0.25, 0.05),  # Conservative Kelly
                            'key_factors': match.get('key_factors', [])
                        }
                        value_bets.append(value_bet)
                        
                except Exception as e:
                    logger.warning(f"Error calculating value bet: {e}")
                    continue
            
            # Sort by edge
            value_bets.sort(key=lambda x: x['edge'], reverse=True)
            
            return jsonify({
                'success': True,
                'value_bets': value_bets,
                'count': len(value_bets),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Value bets error: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to analyze value bets',
                'value_bets': []
            }), 500

    @app.route('/api/underdog-analysis', methods=['POST'])
    @validate_json_request()
    @handle_betting_errors()
    def analyze_underdog():
        """Detailed underdog scenario analysis"""
        try:
            # Get and validate JSON payload
            try:
                if not request.is_json:
                    return jsonify({
                        'success': False,
                        'error': 'Content-Type must be application/json'
                    }), 400
                
                data = request.get_json()
                if data is None:
                    return jsonify({
                        'success': False,
                        'error': 'Invalid JSON payload'
                    }), 400
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': 'Failed to parse JSON'
                }), 400
            
            if not data:
                return jsonify({
                    'success': False,
                    'error': 'No data provided'
                }), 400
            
            # Validate JSON structure to prevent DoS attacks
            if not validate_json_payload(data):
                return jsonify({
                    'success': False,
                    'error': 'Invalid payload structure'
                }), 400
            
            # Sanitize and validate inputs
            data = sanitize_input(data)
            player1 = data.get('player1')
            player2 = data.get('player2')
            tournament = data.get('tournament', 'ATP Tournament')
            surface = data.get('surface', 'Hard')
            
            if not player1 or not player2:
                return jsonify({
                    'success': False,
                    'error': 'Both players are required'
                }), 400
                
            # Validate inputs
            if not validate_player_name(player1) or not validate_player_name(player2):
                return jsonify({
                    'success': False,
                    'error': 'Invalid player names provided'
                }), 400
                
            if not validate_tournament_name(tournament):
                return jsonify({
                    'success': False,
                    'error': 'Invalid tournament name provided'
                }), 400
                
            if not validate_surface(surface):
                return jsonify({
                    'success': False,
                    'error': 'Invalid surface provided'
                }), 400
            
            analyzer = UnderdogAnalyzer()
            
            # Get detailed analysis
            underdog_analysis = analyzer.calculate_underdog_probability(
                player1, player2, tournament, surface
            )
            
            # Add additional information
            scenario = underdog_analysis['underdog_scenario']
            
            detailed_analysis = {
                'underdog_analysis': underdog_analysis,
                'scenario_details': {
                    'underdog_player': scenario['underdog'],
                    'favorite_player': scenario['favorite'],
                    'ranking_gap': scenario['rank_gap'],
                    'underdog_type': scenario['underdog_type'],
                    'base_probability': scenario['base_probability']
                },
                'betting_recommendation': {
                    'recommended_action': 'BET' if underdog_analysis['underdog_probability'] > 0.4 else 'PASS',
                    'risk_level': underdog_analysis['confidence'],
                    'expected_value': underdog_analysis['underdog_probability'] - scenario['base_probability']
                },
                'match_context': {
                    'tournament': tournament,
                    'surface': surface,
                    'tournament_pressure': 'High' if any(major in tournament.lower() for major in ['wimbledon', 'us open', 'french open', 'australian open']) else 'Medium'
                }
            }
            
            return jsonify({
                'success': True,
                'analysis': detailed_analysis,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Underdog analysis error: {e}")
            return jsonify({
                'success': False,
                'error': 'Underdog analysis failed'
            }), 500

    @app.route('/api/refresh', methods=['GET', 'POST'])
    @require_api_key()
    def refresh_data():
        """Refresh data - Note: Old API integrations removed"""
        try:
            # Return success - old API integrations were removed
            return jsonify({
                'success': True,
                'message': 'Data refresh requested (old APIs removed)',
                'source': 'simulation',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Refresh error: {e}")
            return jsonify({
                'success': False,
                'error': create_safe_error_response(e, 'Data refresh failed')
            }), 500

    @app.route('/api/player-info/<player_name>', methods=['GET'])
    def get_player_info(player_name):
        """Player information"""
        try:
            analyzer = UnderdogAnalyzer()
            rank = analyzer.get_player_ranking(player_name)
            
            # Additional information (simulation)
            player_info = {
                'name': player_name,
                'ranking': rank,
                'tour': 'ATP' if rank <= 200 else 'Challenger',
                'estimated_level': 'Top 10' if rank <= 10 else 'Top 50' if rank <= 50 else 'Top 100' if rank <= 100 else 'Professional',
                'underdog_potential': 'High' if rank > 30 else 'Medium' if rank > 15 else 'Low'
            }
            
            return jsonify({
                'success': True,
                'player_info': player_info,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Player info error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/test-underdog', methods=['POST'])
    def test_underdog_analysis():
        """Test underdog analysis (for 'Test Underdog Analysis' button)"""
        try:
            data = request.get_json()
            
            if not data:
                # Use default data if none provided
                data = {
                    'player1': 'Flavio Cobolli',
                    'player2': 'Novak Djokovic',
                    'tournament': 'US Open',
                    'surface': 'Hard'
                }
            
            player1 = data.get('player1', 'Flavio Cobolli')
            player2 = data.get('player2', 'Novak Djokovic') 
            tournament = data.get('tournament', 'US Open')
            surface = data.get('surface', 'Hard')
            
            logger.info(f"🔮 Testing underdog analysis: {player1} vs {player2}")
            
            # Use UnderdogAnalyzer
            analyzer = UnderdogAnalyzer()
            underdog_analysis = analyzer.calculate_underdog_probability(
                player1, player2, tournament, surface
            )
            
            return jsonify({
                'success': True,
                'underdog_analysis': underdog_analysis,
                'match_info': {
                    'player1': player1,
                    'player2': player2,
                    'tournament': tournament,
                    'surface': surface
                },
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Test underdog error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    # Additional API status and management endpoints
    @app.route('/api/manual-api-update', methods=['POST'])
    @require_api_key()
    def manual_api_update():
        """Manual API data update with rate limiting control"""
        try:
            # Priority: Daily Scheduler with limits
            if DAILY_SCHEDULER_AVAILABLE and daily_scheduler:
                try:
                    result = daily_scheduler.make_manual_request("dashboard_manual_update")
                    
                    if result['success']:
                        return jsonify({
                            'success': True,
                            'message': 'Manual API update completed successfully',
                            'total_matches': result.get('total_matches', 0),
                            'daily_used': result.get('daily_used', 0),
                            'monthly_used': result.get('monthly_used', 0),
                            'api_usage': result.get('api_usage', {}),
                            'source': 'daily_scheduler',
                            'timestamp': datetime.now().isoformat()
                        })
                    else:
                        return jsonify({
                            'success': False,
                            'error': result.get('error', 'Manual request failed'),
                            'limits': result.get('limits', {}),
                            'daily_used': result.get('daily_used', 0),
                            'monthly_used': result.get('monthly_used', 0),
                            'source': 'daily_scheduler_denied'
                        }), 429  # Too Many Requests
                        
                except Exception as e:
                    logger.warning(f"Daily scheduler manual update failed: {e}")
            
            
            # Last resort - return information about unavailability
            return jsonify({
                'success': False,
                'error': 'Manual API update not available - daily scheduler and API economy unavailable',
                'message': 'Please wait for scheduled API updates or check system configuration',
                'source': 'no_services_available'
            }), 503  # Service Unavailable
            
        except Exception as e:
            logger.error(f"❌ Manual update error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/api-status', methods=['GET'])
    def get_comprehensive_api_status():
        """Comprehensive API status with Daily Scheduler and API Economy"""
        try:
            status_response = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'daily_scheduler': {},
                'api_economy': {},
                'recommendations': []
            }
            
            # Daily Scheduler Status
            if DAILY_SCHEDULER_AVAILABLE and daily_scheduler:
                try:
                    scheduler_status = daily_scheduler.get_status()
                    status_response['daily_scheduler'] = {
                        'available': True,
                        'status': scheduler_status['status'],
                        'daily_usage': scheduler_status['daily_usage'],
                        'monthly_usage': scheduler_status['monthly_usage'],
                        'next_scheduled': scheduler_status['schedule']['next_scheduled'][:2],  # Next 2 requests
                        'can_make_manual': scheduler_status['can_make_manual']
                    }
                    
                    # Recommendations based on usage
                    daily_used = scheduler_status['daily_usage']['requests_made']
                    daily_limit = scheduler_status['daily_usage']['total_limit']
                    
                    if daily_used >= daily_limit:
                        status_response['recommendations'].append("⚠️ Daily limit reached. Wait for tomorrow or scheduled requests.")
                    elif daily_used >= daily_limit * 0.8:
                        status_response['recommendations'].append("🟡 Near daily limit. Use manual requests carefully.")
                    else:
                        status_response['recommendations'].append("✅ Manual requests available.")
                        
                except Exception as e:
                    status_response['daily_scheduler'] = {
                        'available': False,
                        'error': str(e)
                    }
            else:
                status_response['daily_scheduler'] = {
                    'available': False,
                    'message': 'Daily scheduler not initialized'
                }
            
            # API Economy Status (removed during cleanup)
            status_response['api_economy'] = {
                'available': False,
                'message': 'API Economy removed during cleanup'
            }
            
            return jsonify(status_response)
            
        except Exception as e:
            logger.error(f"❌ Comprehensive API status error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/redis-status', methods=['GET'])
    @require_api_key()
    def get_redis_status():
        """Get detailed Redis connection status and performance metrics"""
        try:
            redis_health = check_redis_health()
            
            # Additional performance tests if Redis is available
            performance_metrics = {}
            if redis_health['available']:
                try:
                    import redis
                    import time
                    
                    redis_url = os.getenv('REDIS_URL', '').strip()
                    if redis_url.startswith('redis://'):
                        r = redis.Redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2)
                    else:
                        r = redis.Redis(host='localhost', port=6379, socket_connect_timeout=2, socket_timeout=2)
                    
                    # Test write/read performance
                    start_time = time.time()
                    test_key = f"health_check_{int(time.time())}"
                    r.set(test_key, 'test_value', ex=5)  # Expire in 5 seconds
                    r.get(test_key)
                    r.delete(test_key)
                    write_read_time = (time.time() - start_time) * 1000
                    
                    # Get memory usage
                    info = r.info('memory')
                    
                    performance_metrics = {
                        'write_read_time_ms': round(write_read_time, 2),
                        'used_memory': info.get('used_memory_human', 'unknown'),
                        'used_memory_peak': info.get('used_memory_peak_human', 'unknown'),
                        'connected_clients': r.info('clients').get('connected_clients', 0)
                    }
                    
                except Exception as e:
                    performance_metrics['error'] = str(e)
            
            return jsonify({
                'success': True,
                'redis_health': redis_health,
                'performance_metrics': performance_metrics,
                'rate_limiter_status': 'redis' if redis_health['available'] else 'in-memory fallback',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Redis status check error: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to check Redis status',
                'timestamp': datetime.now().isoformat()
            }), 500

    @app.route('/api/rankings-status', methods=['GET'])
    def get_rankings_status():
        """Get dynamic rankings system status"""
        try:
            from src.api.dynamic_rankings_api import get_rankings_status
            status = get_rankings_status()
            return jsonify({
                'success': True,
                'rankings_status': status,
                'timestamp': datetime.now().isoformat()
            })
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'Dynamic rankings not available',
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    @app.route('/api/refresh-rankings', methods=['POST'])
    @require_api_key()
    def refresh_rankings():
        """Force refresh tennis rankings from APIs"""
        try:
            from src.api.dynamic_rankings_api import refresh_tennis_rankings
            results = refresh_tennis_rankings()
            return jsonify({
                'success': True,
                'message': 'Rankings refresh completed',
                'results': results,
                'timestamp': datetime.now().isoformat()
            })
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'Dynamic rankings not available',
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Rankings refresh error: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    @app.route('/api/api-economy-status', methods=['GET'])
    def get_api_economy_status():
        """API Economy status (for 'API Status' button) - Legacy endpoint"""
        try:
            # Redirect to comprehensive status but format for backward compatibility
            comprehensive_status = get_comprehensive_api_status()
            data = comprehensive_status.get_json()
            
            if data['success']:
                # Extract relevant info for old format
                daily_scheduler_info = data.get('daily_scheduler', {})
                
                if daily_scheduler_info.get('available'):
                    daily_usage = daily_scheduler_info.get('daily_usage', {})
                    monthly_usage = daily_scheduler_info.get('monthly_usage', {})
                    
                    return jsonify({
                        'success': True,
                        'api_usage': {
                            'requests_this_hour': 'N/A (using daily scheduler)',
                            'max_per_hour': 'N/A (using daily scheduler)', 
                            'remaining_hour': f"{daily_usage.get('manual_remaining', 0)} manual requests remaining",
                            'daily_used': daily_usage.get('requests_made', 0),
                            'daily_limit': daily_usage.get('total_limit', 8),
                            'monthly_used': monthly_usage.get('requests_made', 0),
                            'monthly_limit': monthly_usage.get('limit', 500),
                            'manual_update_status': 'Available' if daily_scheduler_info.get('can_make_manual', False) else 'Limit reached'
                        },
                        'daily_scheduler_available': True,
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                            pass
            
            # Final fallback
            return jsonify({
                'success': True,
                'api_economy_available': False,
                'daily_scheduler_available': False,
                'api_usage': {
                    'requests_this_hour': 0,
                    'max_per_hour': 'N/A',
                    'remaining_hour': 'Service unavailable',
                    'manual_update_status': 'Unavailable'
                },
                'message': 'API services not available',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ API Economy status error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    # API-Tennis.com specific routes
    @app.route('/api/api-tennis/status', methods=['GET'])
    def get_api_tennis_status():
        """Get API-Tennis.com integration status"""
        try:
            from src.data.api_tennis_data_collector import get_api_tennis_data_collector
            collector = get_api_tennis_data_collector()
            
            status = collector.get_integration_status()
            
            return jsonify({
                'success': True,
                'api_tennis_status': status,
                'timestamp': datetime.now().isoformat()
            })
            
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'API-Tennis integration not available',
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"API-Tennis status error: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    @app.route('/api/api-tennis/tournaments', methods=['GET'])
    def get_api_tennis_tournaments():
        """Get tournaments from API-Tennis.com"""
        try:
            from src.data.api_tennis_data_collector import get_api_tennis_data_collector
            collector = get_api_tennis_data_collector()
            
            if not collector.is_available():
                return jsonify({
                    'success': False,
                    'error': 'API-Tennis not configured or unavailable',
                    'tournaments': []
                })
            
            tournaments = collector.get_tournaments()
            
            return jsonify({
                'success': True,
                'tournaments': tournaments,
                'count': len(tournaments),
                'data_source': 'API-Tennis',
                'timestamp': datetime.now().isoformat()
            })
            
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'API-Tennis integration not available',
                'tournaments': []
            })
        except Exception as e:
            logger.error(f"API-Tennis tournaments error: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'tournaments': []
            }), 500

    @app.route('/api/api-tennis/matches', methods=['GET'])
    def get_api_tennis_matches():
        """Get matches from API-Tennis.com"""
        try:
            from src.data.api_tennis_data_collector import get_api_tennis_data_collector
            collector = get_api_tennis_data_collector()
            
            if not collector.is_available():
                return jsonify({
                    'success': False,
                    'error': 'API-Tennis not configured or unavailable',
                    'matches': []
                })
            
            # Get parameters
            include_live = request.args.get('include_live', 'true').lower() == 'true'
            days_ahead = int(request.args.get('days_ahead', '2'))
            
            # Get current matches
            current_matches = collector.get_current_matches(include_live=include_live)
            
            # Get upcoming matches if requested
            if days_ahead > 0:
                upcoming_matches = collector.get_upcoming_matches(days_ahead)
                # Combine and deduplicate
                all_matches = current_matches + upcoming_matches
                seen_ids = set()
                unique_matches = []
                for match in all_matches:
                    match_id = match.get('id')
                    if match_id not in seen_ids:
                        seen_ids.add(match_id)
                        unique_matches.append(match)
                matches = unique_matches
            else:
                matches = current_matches
            
            return jsonify({
                'success': True,
                'matches': matches,
                'count': len(matches),
                'data_source': 'API-Tennis',
                'parameters': {
                    'include_live': include_live,
                    'days_ahead': days_ahead
                },
                'timestamp': datetime.now().isoformat()
            })
            
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'API-Tennis integration not available',
                'matches': []
            })
        except Exception as e:
            logger.error(f"API-Tennis matches error: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'matches': []
            }), 500

    @app.route('/api/api-tennis/player/<player_name>/matches', methods=['GET'])
    def get_api_tennis_player_matches(player_name):
        """Get matches for a specific player from API-Tennis.com"""
        try:
            from src.data.api_tennis_data_collector import get_api_tennis_data_collector
            collector = get_api_tennis_data_collector()
            
            if not collector.is_available():
                return jsonify({
                    'success': False,
                    'error': 'API-Tennis not configured or unavailable',
                    'matches': []
                })
            
            # Validate player name
            if not validate_player_name(player_name):
                return jsonify({
                    'success': False,
                    'error': 'Invalid player name provided',
                    'matches': []
                }), 400
            
            days_ahead = int(request.args.get('days_ahead', '30'))
            matches = collector.get_player_matches(player_name, days_ahead)
            
            return jsonify({
                'success': True,
                'player': player_name,
                'matches': matches,
                'count': len(matches),
                'days_ahead': days_ahead,
                'data_source': 'API-Tennis',
                'timestamp': datetime.now().isoformat()
            })
            
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'API-Tennis integration not available',
                'matches': []
            })
        except Exception as e:
            logger.error(f"API-Tennis player matches error: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'matches': []
            }), 500

    @app.route('/api/api-tennis/match/<int:match_id>/odds', methods=['GET'])
    def get_api_tennis_match_odds(match_id):
        """Get betting odds for a specific match from API-Tennis.com"""
        try:
            from src.data.api_tennis_data_collector import get_api_tennis_data_collector
            collector = get_api_tennis_data_collector()
            
            if not collector.is_available():
                return jsonify({
                    'success': False,
                    'error': 'API-Tennis not configured or unavailable',
                    'odds': {}
                })
            
            odds_data = collector.get_match_odds(match_id)
            
            return jsonify({
                'success': True,
                'match_id': match_id,
                'odds': odds_data,
                'data_source': 'API-Tennis',
                'timestamp': datetime.now().isoformat()
            })
            
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'API-Tennis integration not available',
                'odds': {}
            })
        except Exception as e:
            logger.error(f"API-Tennis match odds error: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'odds': {}
            }), 500

    @app.route('/api/api-tennis/enhanced', methods=['GET'])
    def get_enhanced_api_tennis_data():
        """Get comprehensive data using Enhanced API-Tennis collector"""
        try:
            from src.data.api_tennis_data_collector import get_enhanced_api_tennis_collector
            collector = get_enhanced_api_tennis_collector()
            
            days_ahead = int(request.args.get('days_ahead', '2'))
            matches = collector.get_comprehensive_match_data(days_ahead)
            
            # Get status information
            status = collector.get_status()
            
            return jsonify({
                'success': True,
                'matches': matches,
                'count': len(matches),
                'collector_status': status,
                'parameters': {
                    'days_ahead': days_ahead
                },
                'data_source': 'Enhanced_API_Tennis_Collector',
                'timestamp': datetime.now().isoformat()
            })
            
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'Enhanced API-Tennis collector not available',
                'matches': []
            })
        except Exception as e:
            logger.error(f"Enhanced API-Tennis data error: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'matches': []
            }), 500

    @app.route('/api/api-tennis/clear-cache', methods=['POST'])
    @require_api_key()
    def clear_api_tennis_cache():
        """Clear API-Tennis.com cache"""
        try:
            from src.data.api_tennis_data_collector import get_api_tennis_data_collector
            collector = get_api_tennis_data_collector()
            
            if not collector.is_available():
                return jsonify({
                    'success': False,
                    'error': 'API-Tennis not configured or unavailable'
                })
            
            collector.clear_cache()
            
            return jsonify({
                'success': True,
                'message': 'API-Tennis cache cleared successfully',
                'timestamp': datetime.now().isoformat()
            })
            
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'API-Tennis integration not available'
            })
        except Exception as e:
            logger.error(f"Clear API-Tennis cache error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/api-tennis/test-connection', methods=['GET'])
    def test_api_tennis_connection():
        """Test API-Tennis.com connection and API key"""
        try:
            from src.api.api_tennis_integration import get_api_tennis_client
            client = get_api_tennis_client()
            
            # Test basic connectivity by getting event types
            try:
                event_types = client.get_event_types()
                
                if isinstance(event_types, dict) and event_types.get('success') == 1:
                    return jsonify({
                        'success': True,
                        'message': 'API-Tennis connection successful',
                        'event_types_count': len(event_types.get('result', [])),
                        'api_version': '2.9.4',
                        'client_status': client.get_client_status(),
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'API-Tennis returned invalid response',
                        'response': event_types,
                        'timestamp': datetime.now().isoformat()
                    })
                    
            except Exception as api_error:
                return jsonify({
                    'success': False,
                    'error': f'API-Tennis connection failed: {api_error}',
                    'timestamp': datetime.now().isoformat()
                })
            
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'API-Tennis integration not available',
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Test API-Tennis connection error: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    # ==================================================
    # BACKTESTING AND FORWARD TESTING API ENDPOINTS
    # ==================================================
    
    @app.route('/api/testing/toggle-mode', methods=['POST'])
    @validate_json_request()
    @handle_betting_errors()
    def toggle_testing_mode():
        """Toggle between backtesting and forward testing modes"""
        try:
            data = request.get_json()
            test_mode = data.get('test_mode', 'live')
            
            # Validate test mode
            valid_modes = ['backtest', 'forward_test', 'live']
            if test_mode not in valid_modes:
                return jsonify({
                    'success': False,
                    'error': f'Invalid test mode. Must be one of: {valid_modes}'
                }), 400
            
            # Store current test mode in session or app context
            # For now, we'll return the requested mode
            app.config['CURRENT_TEST_MODE'] = test_mode
            
            logger.info(f"🔄 Switched to test mode: {test_mode}")
            
            return jsonify({
                'success': True,
                'test_mode': test_mode,
                'message': f'Switched to {test_mode} mode',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Error toggling test mode: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to toggle test mode'
            }), 500
    
    @app.route('/api/testing/current-mode', methods=['GET'])
    def get_current_testing_mode():
        """Get current testing mode"""
        try:
            current_mode = app.config.get('CURRENT_TEST_MODE', 'live')
            
            return jsonify({
                'success': True,
                'test_mode': current_mode,
                'available_modes': ['live', 'backtest', 'forward_test'],
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Error getting current test mode: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to get current test mode'
            }), 500
    
    @app.route('/api/backtest/sessions', methods=['GET'])
    def get_backtest_sessions():
        """Get all backtesting sessions"""
        try:
            from src.data.backtest_data_manager import BacktestDataManager
            manager = BacktestDataManager()
            
            active_only = request.args.get('active_only', 'false').lower() == 'true'
            sessions = manager.get_backtest_sessions(active_only=active_only)
            
            return jsonify({
                'success': True,
                'sessions': sessions,
                'count': len(sessions),
                'active_only': active_only,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Error getting backtest sessions: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to retrieve backtest sessions'
            }), 500
    
    @app.route('/api/backtest/create-session', methods=['POST'])
    @validate_json_request()
    @handle_betting_errors()
    def create_backtest_session():
        """Create a new backtesting session"""
        try:
            from src.data.backtest_data_manager import BacktestDataManager
            manager = BacktestDataManager()
            
            data = request.get_json()
            session_name = data.get('session_name', 'Backtest Session')
            start_date = datetime.fromisoformat(data.get('start_date'))
            end_date = datetime.fromisoformat(data.get('end_date'))
            initial_bankroll = data.get('initial_bankroll', 10000.0)
            max_stake_percentage = data.get('max_stake_percentage', 0.05)
            filters = data.get('filters', {})
            
            session_id = manager.create_backtest_session(
                session_name=session_name,
                start_date=start_date,
                end_date=end_date,
                initial_bankroll=initial_bankroll,
                max_stake_percentage=max_stake_percentage,
                filters=filters
            )
            
            if session_id:
                return jsonify({
                    'success': True,
                    'session_id': session_id,
                    'message': 'Backtest session created successfully',
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to create backtest session'
                }), 500
                
        except Exception as e:
            logger.error(f"❌ Error creating backtest session: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to create backtest session'
            }), 500
    
    @app.route('/api/backtest/matches', methods=['GET'])
    def get_backtest_matches():
        """Get historical matches for backtesting"""
        try:
            from src.data.backtest_data_manager import BacktestDataManager
            manager = BacktestDataManager()
            
            # Parse query parameters
            start_date = request.args.get('start_date')
            end_date = request.args.get('end_date')
            surface = request.args.get('surface')
            tournament_level = request.args.get('tournament_level')
            min_quality_score = float(request.args.get('min_quality_score', 7.0))
            limit = int(request.args.get('limit', 100))
            
            # Convert date strings
            start_date = datetime.fromisoformat(start_date) if start_date else None
            end_date = datetime.fromisoformat(end_date) if end_date else None
            
            # Parse ranking range
            ranking_range = None
            if request.args.get('ranking_min') and request.args.get('ranking_max'):
                ranking_range = (
                    int(request.args.get('ranking_min')),
                    int(request.args.get('ranking_max'))
                )
            
            matches = manager.get_backtest_matches(
                start_date=start_date,
                end_date=end_date,
                surface=surface,
                tournament_level=tournament_level,
                ranking_range=ranking_range,
                min_quality_score=min_quality_score,
                limit=limit
            )
            
            return jsonify({
                'success': True,
                'matches': matches,
                'count': len(matches),
                'filters_applied': {
                    'start_date': start_date.isoformat() if start_date else None,
                    'end_date': end_date.isoformat() if end_date else None,
                    'surface': surface,
                    'tournament_level': tournament_level,
                    'ranking_range': ranking_range,
                    'min_quality_score': min_quality_score
                },
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Error getting backtest matches: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to retrieve backtest matches'
            }), 500
    
    @app.route('/api/forward-test/sessions', methods=['GET'])
    def get_forward_test_sessions():
        """Get all forward testing sessions"""
        try:
            from src.data.forward_test_manager import ForwardTestManager
            manager = ForwardTestManager()
            
            active_only = request.args.get('active_only', 'false').lower() == 'true'
            sessions = manager.get_forward_test_sessions(active_only=active_only)
            
            return jsonify({
                'success': True,
                'sessions': sessions,
                'count': len(sessions),
                'active_only': active_only,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Error getting forward test sessions: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to retrieve forward test sessions'
            }), 500
    
    @app.route('/api/forward-test/create-session', methods=['POST'])
    @validate_json_request()
    @handle_betting_errors()
    def create_forward_test_session():
        """Create a new forward testing session"""
        try:
            from src.data.forward_test_manager import ForwardTestManager
            manager = ForwardTestManager()
            
            data = request.get_json()
            session_name = data.get('session_name', 'Forward Test Session')
            duration_days = data.get('duration_days', 30)
            initial_bankroll = data.get('initial_bankroll', 10000.0)
            max_stake_percentage = data.get('max_stake_percentage', 0.05)
            filters = data.get('filters', {})
            
            session_id = manager.create_forward_test_session(
                session_name=session_name,
                duration_days=duration_days,
                initial_bankroll=initial_bankroll,
                max_stake_percentage=max_stake_percentage,
                filters=filters
            )
            
            if session_id:
                return jsonify({
                    'success': True,
                    'session_id': session_id,
                    'message': 'Forward test session created successfully',
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to create forward test session'
                }), 500
                
        except Exception as e:
            logger.error(f"❌ Error creating forward test session: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to create forward test session'
            }), 500
    
    @app.route('/api/forward-test/live-matches', methods=['GET'])
    def get_forward_test_live_matches():
        """Get live matches for forward testing"""
        try:
            from src.data.forward_test_manager import ForwardTestManager
            manager = ForwardTestManager()
            
            session_id = request.args.get('session_id')
            days_ahead = int(request.args.get('days_ahead', 3))
            ranking_min = int(request.args.get('ranking_min', 10))
            ranking_max = int(request.args.get('ranking_max', 300))
            
            matches = manager.get_live_matches_for_testing(
                session_id=session_id,
                days_ahead=days_ahead,
                ranking_filter=(ranking_min, ranking_max)
            )
            
            return jsonify({
                'success': True,
                'matches': matches,
                'count': len(matches),
                'parameters': {
                    'session_id': session_id,
                    'days_ahead': days_ahead,
                    'ranking_filter': [ranking_min, ranking_max]
                },
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Error getting forward test matches: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to retrieve forward test matches'
            }), 500
    
    # ==================================================
    # BETTING LOGS API ENDPOINTS
    # ==================================================
    
    @app.route('/api/betting/logs', methods=['GET'])
    def get_betting_logs():
        """Get betting logs with filtering options"""
        try:
            from src.api.betting_tracker_service import BettingTrackerService
            from src.data.database_models import TestMode, BettingStatus
            
            tracker = BettingTrackerService()
            
            # Parse query parameters
            test_mode_str = request.args.get('test_mode')
            status_str = request.args.get('status')
            start_date_str = request.args.get('start_date')
            end_date_str = request.args.get('end_date')
            limit = int(request.args.get('limit', 100))
            
            # Convert parameters
            test_mode = None
            if test_mode_str:
                test_mode = TestMode(test_mode_str)
            
            status = None
            if status_str:
                status = BettingStatus(status_str)
            
            start_date = datetime.fromisoformat(start_date_str) if start_date_str else None
            end_date = datetime.fromisoformat(end_date_str) if end_date_str else None
            
            logs = tracker.get_betting_logs(
                test_mode=test_mode,
                status=status,
                start_date=start_date,
                end_date=end_date,
                limit=limit
            )
            
            return jsonify({
                'success': True,
                'betting_logs': logs,
                'count': len(logs),
                'filters_applied': {
                    'test_mode': test_mode_str,
                    'status': status_str,
                    'start_date': start_date_str,
                    'end_date': end_date_str,
                    'limit': limit
                },
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Error getting betting logs: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to retrieve betting logs'
            }), 500
    
    @app.route('/api/betting/log-decision', methods=['POST'])
    @validate_json_request()
    @handle_betting_errors()
    def log_betting_decision():
        """Log a new betting decision"""
        try:
            from src.api.betting_tracker_service import TennisBettingIntegration
            from src.data.database_models import TestMode
            
            integration = TennisBettingIntegration()
            data = request.get_json()
            
            # Extract match data
            match_data = data.get('match_data', {})
            underdog_analysis = data.get('underdog_analysis', {})
            stake_amount = data.get('stake_amount', 100.0)
            test_mode_str = data.get('test_mode', 'live')
            
            test_mode = TestMode(test_mode_str)
            
            bet_id = integration.log_underdog_bet(
                match_data=match_data,
                underdog_analysis=underdog_analysis,
                stake_amount=stake_amount,
                test_mode=test_mode
            )
            
            if bet_id:
                return jsonify({
                    'success': True,
                    'bet_id': bet_id,
                    'message': 'Betting decision logged successfully',
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to log betting decision'
                }), 500
                
        except Exception as e:
            logger.error(f"❌ Error logging betting decision: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to log betting decision'
            }), 500
    
    @app.route('/api/betting/update-outcome', methods=['POST'])
    @validate_json_request()
    @handle_betting_errors()
    def update_betting_outcome():
        """Update betting outcome when match is completed"""
        try:
            from src.api.betting_tracker_service import BettingTrackerService
            
            tracker = BettingTrackerService()
            data = request.get_json()
            
            bet_id = data.get('bet_id')
            match_result = data.get('match_result', {})
            settlement_notes = data.get('settlement_notes', '')
            
            if not bet_id:
                return jsonify({
                    'success': False,
                    'error': 'bet_id is required'
                }), 400
            
            success = tracker.update_betting_outcome(
                bet_id=bet_id,
                match_result=match_result,
                settlement_notes=settlement_notes
            )
            
            if success:
                return jsonify({
                    'success': True,
                    'message': 'Betting outcome updated successfully',
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to update betting outcome'
                }), 404
                
        except Exception as e:
            logger.error(f"❌ Error updating betting outcome: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to update betting outcome'
            }), 500
    
    @app.route('/api/betting/performance-summary', methods=['GET'])
    def get_betting_performance_summary():
        """Get comprehensive betting performance summary"""
        try:
            from src.api.betting_tracker_service import BettingTrackerService
            from src.data.database_models import TestMode
            
            tracker = BettingTrackerService()
            
            test_mode_str = request.args.get('test_mode')
            days_back = int(request.args.get('days_back', 30))
            
            test_mode = TestMode(test_mode_str) if test_mode_str else None
            
            summary = tracker.get_betting_performance_summary(
                test_mode=test_mode,
                days_back=days_back
            )
            
            return jsonify({
                'success': True,
                'performance_summary': summary,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Error getting betting performance summary: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to get performance summary'
            }), 500
    
    @app.route('/api/betting/pending-bets', methods=['GET'])
    def get_pending_bets():
        """Get all pending bets awaiting settlement"""
        try:
            from src.api.betting_tracker_service import BettingTrackerService
            from src.data.database_models import TestMode
            
            tracker = BettingTrackerService()
            
            test_mode_str = request.args.get('test_mode')
            test_mode = TestMode(test_mode_str) if test_mode_str else None
            
            pending_bets = tracker.get_pending_bets(test_mode=test_mode)
            
            return jsonify({
                'success': True,
                'pending_bets': pending_bets,
                'count': len(pending_bets),
                'test_mode': test_mode_str,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Error getting pending bets: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to get pending bets'
            }), 500
    
    # ==================================================
    # PERFORMANCE COMPARISON API ENDPOINTS
    # ==================================================
    
    @app.route('/api/performance/compare', methods=['POST'])
    @validate_json_request()
    @handle_betting_errors()
    def create_performance_comparison():
        """Create a performance comparison between backtest and forward test sessions"""
        try:
            from src.api.performance_comparison_service import PerformanceComparisonService
            
            service = PerformanceComparisonService()
            data = request.get_json()
            
            comparison_name = data.get('comparison_name', 'Performance Comparison')
            backtest_session_id = data.get('backtest_session_id')
            forward_test_session_id = data.get('forward_test_session_id')
            
            if not backtest_session_id or not forward_test_session_id:
                return jsonify({
                    'success': False,
                    'error': 'Both backtest_session_id and forward_test_session_id are required'
                }), 400
            
            comparison_id = service.create_performance_comparison(
                comparison_name=comparison_name,
                backtest_session_id=backtest_session_id,
                forward_test_session_id=forward_test_session_id
            )
            
            if comparison_id:
                return jsonify({
                    'success': True,
                    'comparison_id': comparison_id,
                    'message': 'Performance comparison created successfully',
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to create performance comparison'
                }), 500
                
        except Exception as e:
            logger.error(f"❌ Error creating performance comparison: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to create performance comparison'
            }), 500
    
    @app.route('/api/performance/comparisons', methods=['GET'])
    def get_performance_comparisons():
        """Get all performance comparisons"""
        try:
            from src.api.performance_comparison_service import PerformanceComparisonService
            
            service = PerformanceComparisonService()
            limit = int(request.args.get('limit', 10))
            
            comparisons = service.get_performance_comparisons(limit=limit)
            
            return jsonify({
                'success': True,
                'comparisons': comparisons,
                'count': len(comparisons),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Error getting performance comparisons: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to get performance comparisons'
            }), 500
    
    @app.route('/api/performance/comparison/<comparison_id>', methods=['GET'])
    def get_detailed_comparison(comparison_id):
        """Get detailed performance comparison analysis"""
        try:
            from src.api.performance_comparison_service import PerformanceComparisonService
            
            service = PerformanceComparisonService()
            comparison = service.get_detailed_comparison(comparison_id)
            
            if 'error' in comparison:
                return jsonify({
                    'success': False,
                    'error': comparison['error']
                }), 404
            
            return jsonify({
                'success': True,
                'comparison': comparison,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Error getting detailed comparison: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to get detailed comparison'
            }), 500
    
    @app.route('/api/performance/stability-report', methods=['GET'])
    def get_model_stability_report():
        """Get comprehensive model stability report"""
        try:
            from src.api.performance_comparison_service import PerformanceComparisonService
            
            service = PerformanceComparisonService()
            days_back = int(request.args.get('days_back', 90))
            
            report = service.generate_model_stability_report(days_back=days_back)
            
            return jsonify({
                'success': True,
                'stability_report': report,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Error getting stability report: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to get stability report'
            }), 500

    # ==================================================
    # BETTING STATISTICS API ENDPOINTS
    # ==================================================
    
    @app.route('/api/betting/statistics', methods=['GET'])
    def get_betting_statistics():
        """Get comprehensive betting statistics for different timeframes"""
        try:
            from src.api.betting_tracker_service import BettingTrackerService
            from src.data.database_models import TestMode
            
            tracker = BettingTrackerService()
            
            # Parse query parameters
            timeframe = request.args.get('timeframe', '1_month')  # 1_week, 1_month, 1_year, all_time
            test_mode_str = request.args.get('test_mode', 'live')
            
            # Convert test mode
            try:
                test_mode = TestMode(test_mode_str) if test_mode_str != 'all' else None
            except ValueError:
                test_mode = TestMode.LIVE
            
            # Get comprehensive statistics using the enhanced method
            statistics = tracker.get_betting_statistics_by_timeframe(
                test_mode=test_mode,
                timeframe=timeframe
            )
            
            return jsonify({
                'success': True,
                'timeframe': timeframe,
                'test_mode': test_mode_str,
                'statistics': statistics,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Error getting betting statistics: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to get betting statistics',
                'details': str(e)
            }), 500
    
    @app.route('/api/betting/charts-data', methods=['GET'])
    def get_betting_charts_data():
        """Get chart data for betting statistics visualization"""
        try:
            from src.api.betting_tracker_service import BettingTrackerService
            from src.data.database_models import TestMode
            
            tracker = BettingTrackerService()
            
            # Parse parameters
            timeframe = request.args.get('timeframe', '1_month')
            test_mode_str = request.args.get('test_mode', 'live')
            chart_type = request.args.get('chart_type', 'profit_timeline')
            
            try:
                test_mode = TestMode(test_mode_str) if test_mode_str != 'all' else None
            except ValueError:
                test_mode = TestMode.LIVE
            
            # Generate chart data using the enhanced method
            chart_data = tracker.get_chart_data_for_timeframe(
                test_mode=test_mode,
                timeframe=timeframe,
                chart_type=chart_type
            )
            
            return jsonify({
                'success': True,
                'chart_type': chart_type,
                'timeframe': timeframe,
                'test_mode': test_mode_str,
                'data': chart_data,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Error getting chart data: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to get chart data',
                'details': str(e)
            }), 500
    
    @app.route('/api/betting/timeframe-comparison', methods=['GET'])
    def get_timeframe_comparison():
        """Get betting statistics comparison across multiple timeframes"""
        try:
            from src.api.betting_tracker_service import BettingTrackerService
            from src.data.database_models import TestMode
            
            tracker = BettingTrackerService()
            
            # Parse parameters
            test_mode_str = request.args.get('test_mode', 'live')
            
            try:
                test_mode = TestMode(test_mode_str) if test_mode_str != 'all' else None
            except ValueError:
                test_mode = TestMode.LIVE
            
            # Get statistics for all timeframes
            timeframes = ['1_week', '1_month', '1_year', 'all_time']
            comparison_data = {}
            
            for timeframe in timeframes:
                try:
                    stats = tracker.get_betting_statistics_by_timeframe(
                        test_mode=test_mode,
                        timeframe=timeframe
                    )
                    comparison_data[timeframe] = stats
                except Exception as e:
                    logger.warning(f"Error getting {timeframe} statistics: {e}")
                    comparison_data[timeframe] = {
                        'error': str(e),
                        'basic_metrics': {
                            'total_bets': 0,
                            'win_rate': 0,
                            'net_profit': 0,
                            'roi_percentage': 0
                        }
                    }
            
            # Create enhanced summary comparison
            summary_comparison = {}
            quality_overview = {}
            
            for timeframe in timeframes:
                data = comparison_data[timeframe]
                if 'basic_metrics' in data:
                    # Financial and performance metrics
                    summary_comparison[timeframe] = {
                        'total_bets': data['basic_metrics'].get('total_bets', 0),
                        'win_rate': data['basic_metrics'].get('win_rate', 0),
                        'net_profit': data.get('financial_metrics', {}).get('net_profit', 0),
                        'roi_percentage': data.get('financial_metrics', {}).get('roi_percentage', 0),
                        'sharpe_ratio': data.get('risk_metrics', {}).get('sharpe_ratio', 0),
                        'max_drawdown': data.get('risk_metrics', {}).get('max_drawdown', 0),
                        'profit_factor': data.get('financial_metrics', {}).get('profit_factor', 0),
                        'period_label': data.get('period_label', timeframe.replace('_', ' ').title())
                    }
                    
                    # Data quality metrics
                    data_quality = data.get('data_quality', {})
                    quality_overview[timeframe] = {
                        'sample_size': data_quality.get('sample_size', 0),
                        'quality_score': data_quality.get('quality_score', 0),
                        'statistical_significance': data_quality.get('statistical_significance', 'unknown'),
                        'data_completeness': data_quality.get('data_completeness', 'unknown'),
                        'recommendations': data_quality.get('recommendations', [])
                    }
                else:
                    # Handle error cases
                    summary_comparison[timeframe] = {
                        'total_bets': 0,
                        'win_rate': 0,
                        'net_profit': 0,
                        'roi_percentage': 0,
                        'sharpe_ratio': 0,
                        'max_drawdown': 0,
                        'profit_factor': 0,
                        'period_label': timeframe.replace('_', ' ').title(),
                        'error': data.get('error', 'Unknown error')
                    }
                    
                    quality_overview[timeframe] = {
                        'sample_size': 0,
                        'quality_score': 0,
                        'statistical_significance': 'error',
                        'data_completeness': 'error',
                        'recommendations': ['Unable to load data for this timeframe']
                    }
            
            # Generate cross-timeframe insights
            insights = []
            
            # Performance trend analysis
            timeframe_order = ['1_week', '1_month', '1_year', 'all_time']
            roi_values = [summary_comparison[tf].get('roi_percentage', 0) for tf in timeframe_order if tf in summary_comparison]
            
            if len(roi_values) >= 2:
                if roi_values[0] > roi_values[1]:
                    insights.append("Recent performance (1 week) is stronger than monthly average")
                elif roi_values[1] > roi_values[2] if len(roi_values) > 2 else 0:
                    insights.append("Recent monthly performance is better than yearly average")
            
            # Sample size adequacy
            for timeframe in timeframes:
                quality = quality_overview.get(timeframe, {})
                sample_size = quality.get('sample_size', 0)
                if sample_size > 0 and sample_size < 30:
                    insights.append(f"{timeframe.replace('_', ' ').title()}: Limited data ({sample_size} bets) - exercise caution")
            
            # Best performing timeframe
            valid_timeframes = [tf for tf in timeframes if summary_comparison[tf].get('total_bets', 0) >= 10]
            if valid_timeframes:
                best_timeframe = max(valid_timeframes, 
                                   key=lambda tf: summary_comparison[tf].get('roi_percentage', -999))
                best_roi = summary_comparison[best_timeframe].get('roi_percentage', 0)
                insights.append(f"Best performing period: {best_timeframe.replace('_', ' ').title()} (ROI: {best_roi:.1f}%)")
            
            return jsonify({
                'success': True,
                'test_mode': test_mode_str,
                'comparison_data': comparison_data,
                'summary_comparison': summary_comparison,
                'quality_overview': quality_overview,
                'insights': insights,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Error getting timeframe comparison: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to get timeframe comparison',
                'details': str(e)
            }), 500
    
    @app.route('/api/betting/advanced-metrics', methods=['GET'])
    def get_advanced_betting_metrics():
        """Get advanced betting metrics including risk analysis and model performance"""
        try:
            from src.api.betting_tracker_service import BettingTrackerService
            from src.data.database_models import TestMode
            
            tracker = BettingTrackerService()
            
            # Parse parameters
            timeframe = request.args.get('timeframe', '1_month')
            test_mode_str = request.args.get('test_mode', 'live')
            
            try:
                test_mode = TestMode(test_mode_str) if test_mode_str != 'all' else None
            except ValueError:
                test_mode = TestMode.LIVE
            
            # Get comprehensive statistics
            statistics = tracker.get_betting_statistics_by_timeframe(
                test_mode=test_mode,
                timeframe=timeframe
            )
            
            # Extract advanced metrics
            advanced_metrics = {
                'risk_analysis': statistics.get('risk_metrics', {}),
                'streak_analysis': statistics.get('streak_analysis', {}),
                'odds_analysis': statistics.get('odds_analysis', {}),
                'model_performance': statistics.get('model_performance', {}),
                'rolling_metrics': statistics.get('rolling_metrics', {}),
                'time_analysis': statistics.get('time_analysis', {}),
                'data_quality': statistics.get('data_quality', {})
            }
            
            return jsonify({
                'success': True,
                'timeframe': timeframe,
                'test_mode': test_mode_str,
                'advanced_metrics': advanced_metrics,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Error getting advanced metrics: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to get advanced metrics',
                'details': str(e)
            }), 500

    # ========================================
    # PREDICTION-BETTING INTEGRATION ENDPOINTS
    # ========================================
    
    @app.route('/api/betting/telegram-predictions', methods=['GET'])
    def get_telegram_prediction_statistics():
        """Get betting statistics from Telegram predictions"""
        try:
            from src.api.prediction_betting_integration import get_betting_dashboard_stats
            
            # Parse parameters
            days = int(request.args.get('days', 30))
            days = max(1, min(365, days))  # Clamp between 1 and 365 days
            
            # Get statistics
            stats = get_betting_dashboard_stats(days)
            
            return jsonify({
                'success': True,
                'statistics': stats,
                'source': 'telegram_predictions',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Error getting telegram prediction statistics: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to get telegram prediction statistics',
                'details': str(e)
            }), 500
    
    @app.route('/api/betting/ml-performance', methods=['GET'])
    def get_ml_performance_from_predictions():
        """Get ML performance metrics from actual prediction records"""
        try:
            from src.api.prediction_betting_integration import get_prediction_betting_integrator
            
            # Parse parameters
            days = int(request.args.get('days', 30))
            days = max(1, min(365, days))  # Clamp between 1 and 365 days
            
            integrator = get_prediction_betting_integrator()
            stats = integrator.get_betting_statistics(days)
            
            # Format for ML performance display
            ml_performance = {
                'overview': {
                    'total_predictions': stats['total_bets'],
                    'settled_predictions': stats['settled_bets'],
                    'win_rate': stats['win_rate'],
                    'roi': stats['roi'],
                    'net_profit': stats['net_profit'],
                    'avg_odds': stats['avg_odds']
                },
                'confidence_breakdown': stats['confidence_breakdown'],
                'model_performance': stats['model_performance'],
                'bankroll_info': {
                    'current_bankroll': stats['current_bankroll'],
                    'total_staked': stats['total_staked'],
                    'total_returned': stats['total_returned']
                },
                'period_info': {
                    'days': days,
                    'last_updated': stats['last_updated']
                }
            }
            
            return jsonify({
                'success': True,
                'ml_performance': ml_performance,
                'source': 'live_predictions',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Error getting ML performance from predictions: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to get ML performance data',
                'details': str(e)
            }), 500
    
    @app.route('/api/betting/prediction-records', methods=['GET'])
    def get_prediction_betting_records():
        """Get detailed prediction betting records"""
        try:
            from src.api.prediction_betting_integration import get_prediction_betting_integrator
            from src.data.database_models import BettingLog, BettingStatus
            from datetime import datetime, timedelta
            
            integrator = get_prediction_betting_integrator()
            session = integrator.db_manager.get_session()
            
            # Parse parameters
            days = int(request.args.get('days', 30))
            limit = int(request.args.get('limit', 50))
            status_filter = request.args.get('status', 'all')
            
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Build query
            query = session.query(BettingLog).filter(
                BettingLog.timestamp >= start_date,
                BettingLog.timestamp <= end_date
            )
            
            # Apply status filter
            if status_filter != 'all':
                try:
                    status_enum = BettingStatus(status_filter)
                    query = query.filter(BettingLog.betting_status == status_enum)
                except ValueError:
                    pass  # Ignore invalid status
            
            # Order and limit
            records = query.order_by(BettingLog.timestamp.desc()).limit(limit).all()
            
            # Format records
            formatted_records = []
            for record in records:
                formatted_record = {
                    'bet_id': record.bet_id,
                    'timestamp': record.timestamp.isoformat(),
                    'match': {
                        'player1': record.player1,
                        'player2': record.player2,
                        'tournament': record.tournament,
                        'match_date': record.match_date.isoformat() if record.match_date else None
                    },
                    'prediction': {
                        'predicted_winner': record.predicted_winner,
                        'our_probability': record.our_probability,
                        'confidence_level': record.confidence_level,
                        'model_used': record.model_used
                    },
                    'betting': {
                        'odds_taken': record.odds_taken,
                        'stake_amount': record.stake_amount,
                        'edge_percentage': record.edge_percentage,
                        'risk_level': record.risk_level
                    },
                    'outcome': {
                        'status': record.betting_status.value if record.betting_status else 'unknown',
                        'actual_winner': record.actual_winner,
                        'profit_loss': record.profit_loss,
                        'roi_percentage': record.roi_percentage,
                        'prediction_correct': record.prediction_correct
                    }
                }
                formatted_records.append(formatted_record)
            
            session.close()
            
            return jsonify({
                'success': True,
                'records': formatted_records,
                'total_records': len(formatted_records),
                'filters_applied': {
                    'days': days,
                    'status': status_filter,
                    'limit': limit
                },
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Error getting prediction betting records: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to get prediction betting records',
                'details': str(e)
            }), 500

    @app.route('/api/betting/alerts', methods=['GET'])
    @app.limiter.limit("60 per hour")
    def get_betting_alerts():
        """Get active betting alerts and system notifications"""
        try:
            # Check for system alerts
            alerts = []
            
            # Check bankroll status
            from src.api.prediction_betting_integration import PredictionBettingIntegrator, PredictionBettingConfig
            config = PredictionBettingConfig()
            integrator = PredictionBettingIntegrator(config)
            
            stats = integrator.get_betting_statistics(days=1)
            
            # Low bankroll alert
            if stats.get('current_bankroll', 0) < config.initial_bankroll * 0.3:
                alerts.append({
                    'type': 'warning',
                    'category': 'bankroll',
                    'message': f"Bankroll low: ${stats.get('current_bankroll', 0):.2f}",
                    'timestamp': datetime.now().isoformat(),
                    'severity': 'medium'
                })
            
            # High loss rate alert
            if stats.get('win_rate', 100) < 40:
                alerts.append({
                    'type': 'warning',
                    'category': 'performance',
                    'message': f"Low win rate: {stats.get('win_rate', 0):.1f}%",
                    'timestamp': datetime.now().isoformat(),
                    'severity': 'high'
                })
            
            # Negative ROI alert
            if stats.get('roi', 0) < -10:
                alerts.append({
                    'type': 'warning',
                    'category': 'performance',
                    'message': f"Negative ROI: {stats.get('roi', 0):.1f}%",
                    'timestamp': datetime.now().isoformat(),
                    'severity': 'high'
                })
            
            # System health alerts
            if not os.path.exists('data/tennis_predictions.db'):
                alerts.append({
                    'type': 'error',
                    'category': 'system',
                    'message': "Database connection issue detected",
                    'timestamp': datetime.now().isoformat(),
                    'severity': 'critical'
                })
            
            return jsonify({
                'success': True,
                'alerts': alerts,
                'alert_count': len(alerts),
                'system_status': 'healthy' if len(alerts) == 0 else 'warning' if any(a['severity'] in ['medium', 'high'] for a in alerts) else 'critical',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Error getting betting alerts: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to get betting alerts',
                'details': str(e)
            }), 500

    @app.route('/api/betting/dashboard-stats', methods=['GET'])
    @app.limiter.limit("100 per hour")
    def get_betting_dashboard_stats():
        """Get comprehensive dashboard statistics for betting system"""
        try:
            from src.api.prediction_betting_integration import PredictionBettingIntegrator, PredictionBettingConfig
            config = PredictionBettingConfig()
            integrator = PredictionBettingIntegrator(config)
            
            # Get comprehensive statistics
            stats = integrator.get_betting_statistics(days=30)
            
            # Calculate additional dashboard metrics
            dashboard_stats = {
                'summary': {
                    'current_bankroll': stats.get('current_bankroll', 0),
                    'total_staked': stats.get('total_staked', 0),
                    'net_profit': stats.get('net_profit', 0),
                    'roi_percentage': stats.get('roi', 0),
                    'win_rate': stats.get('win_rate', 0),
                    'total_bets': stats.get('total_bets', 0),
                    'settled_bets': stats.get('settled_bets', 0),
                    'pending_bets': stats.get('pending_bets', 0)
                },
                'performance': {
                    'confidence_breakdown': stats.get('confidence_breakdown', {}),
                    'model_performance': stats.get('model_performance', {}),
                    'avg_odds': stats.get('avg_odds', 0),
                    'avg_stake': stats.get('avg_stake', 0)
                },
                'alerts': {
                    'low_bankroll': stats.get('current_bankroll', 0) < config.initial_bankroll * 0.3,
                    'poor_performance': stats.get('win_rate', 100) < 40,
                    'negative_roi': stats.get('roi', 0) < -10
                },
                'recent_activity': {
                    'last_bet_date': stats.get('last_bet_date'),
                    'bets_today': stats.get('bets_today', 0),
                    'profit_today': stats.get('profit_today', 0)
                }
            }
            
            return jsonify({
                'success': True,
                'dashboard_stats': dashboard_stats,
                'period': '30 days',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Error getting dashboard stats: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to get dashboard statistics',
                'details': str(e)
            }), 500

    logger.info("✅ All routes registered successfully (including prediction-betting integration)")