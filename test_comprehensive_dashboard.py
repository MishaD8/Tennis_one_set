#!/usr/bin/env python3
"""
Test script for the Comprehensive Betting Dashboard
This script starts a minimal Flask server to test the new dashboard
"""

import os
import sys
from flask import Flask, render_template, jsonify
from datetime import datetime

# Add the project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def create_test_app():
    """Create a minimal test app for the comprehensive dashboard"""
    
    template_dir = os.path.join(project_root, 'templates')
    static_dir = os.path.join(project_root, 'static')
    
    app = Flask(__name__, 
                template_folder=template_dir,
                static_folder=static_dir)
    
    # Test route for the comprehensive dashboard
    @app.route('/comprehensive-betting-stats')
    def comprehensive_betting_statistics():
        """Comprehensive betting statistics dashboard"""
        return render_template('comprehensive_betting_dashboard.html')
    
    @app.route('/')
    def index():
        """Home page redirect"""
        return '<h1>🎾 Tennis Dashboard Test</h1><p><a href="/comprehensive-betting-stats">📊 Comprehensive Betting Stats</a></p>'
    
    # Mock API endpoints for testing
    @app.route('/api/comprehensive-statistics')
    def get_comprehensive_statistics():
        """Mock comprehensive statistics API"""
        return jsonify({
            'success': True,
            'statistics': {
                'summary': {
                    'total_matches': 125,
                    'overall_win_rate': 67.2,
                    'average_odds': 2.34
                },
                'financial_summary': {
                    'total_profit': 1543.50,
                    'roi': 12.4
                },
                'current_streak': {
                    'type': 'win',
                    'count': 5
                }
            },
            'timestamp': datetime.now().isoformat()
        })
    
    @app.route('/api/match-statistics')
    def get_match_statistics():
        """Mock match statistics API"""
        return jsonify({
            'success': True,
            'matches': [
                {
                    'date': '2024-08-24',
                    'tournament': 'US Open',
                    'surface': 'hard',
                    'player1': {'name': 'Novak Djokovic', 'rank': 1, 'country': 'Serbia'},
                    'player2': {'name': 'Carlos Alcaraz', 'rank': 2, 'country': 'Spain'},
                    'score': '6-4, 6-7, 6-3',
                    'winner': 'Novak Djokovic',
                    'betting_ratios': {
                        'start_2nd_set': {'player1': 1.85, 'player2': 1.95},
                        'end_2nd_set': {'player1': 1.72, 'player2': 2.08}
                    }
                },
                {
                    'date': '2024-08-23',
                    'tournament': 'US Open',
                    'surface': 'hard',
                    'player1': {'name': 'Jannik Sinner', 'rank': 3, 'country': 'Italy'},
                    'player2': {'name': 'Daniil Medvedev', 'rank': 4, 'country': 'Russia'},
                    'score': '7-6, 6-4',
                    'winner': 'Jannik Sinner',
                    'betting_ratios': {
                        'start_2nd_set': {'player1': 2.10, 'player2': 1.75},
                        'end_2nd_set': {'player1': 1.95, 'player2': 1.85}
                    }
                }
            ],
            'pagination': {
                'page': 1,
                'per_page': 20,
                'total_matches': 2,
                'total_pages': 1
            },
            'summary': {
                'total_matches': 2
            }
        })
    
    @app.route('/api/player-statistics')
    def get_player_statistics():
        """Mock player statistics API"""
        return jsonify({
            'success': True,
            'statistics': {
                'top_performers': [
                    {
                        'name': 'Novak Djokovic',
                        'win_rate': 85.5,
                        'matches': 45,
                        'profit': 2340.50
                    },
                    {
                        'name': 'Carlos Alcaraz',
                        'win_rate': 78.2,
                        'matches': 38,
                        'profit': 1890.75
                    }
                ],
                'ranking_distribution': {
                    'top_10': 15,
                    'rank_11_50': 25,
                    'rank_51_100': 35,
                    'rank_100_plus': 50
                },
                'surface_performance': {
                    'hard': {'win_rate': 72.5, 'matches': 80, 'avg_odds': 2.15},
                    'clay': {'win_rate': 68.3, 'matches': 35, 'avg_odds': 2.25},
                    'grass': {'win_rate': 75.0, 'matches': 10, 'avg_odds': 2.05}
                }
            }
        })
    
    @app.route('/api/betting-ratio-analysis')
    def get_betting_analysis():
        """Mock betting ratio analysis API"""
        return jsonify({
            'success': True,
            'analysis': {
                'ratio_trends': {
                    'avg_change': 8.5,
                    'volatility': 12.3,
                    'upward_moves': 65,
                    'downward_moves': 35
                },
                'movement_analysis': {
                    'largest_positive': 25.8,
                    'largest_negative': -18.4,
                    'avg_positive': 12.5,
                    'avg_negative': -9.3
                },
                'insights': {
                    'success_rate': 67.2,
                    'best_timing': 'Start of 2nd set',
                    'recommended_strategy': 'Monitor early 2nd set movements'
                }
            }
        })
    
    @app.route('/api/clear-statistics', methods=['POST'])
    def clear_statistics():
        """Mock clear statistics API"""
        return jsonify({
            'success': True,
            'message': 'Statistics cleared successfully (mock mode)',
            'timestamp': datetime.now().isoformat()
        })
    
    return app

if __name__ == '__main__':
    app = create_test_app()
    print("🚀 Starting test server for Comprehensive Betting Dashboard...")
    print("🌐 Open: http://localhost:5001/comprehensive-betting-stats")
    print("📱 Or visit: http://localhost:5001/ for home page")
    
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True
    )