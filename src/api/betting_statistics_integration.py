#!/usr/bin/env python3
"""
🔗 Betting Statistics Integration Module
Integrates the comprehensive statistics system with existing betting predictions and match tracking
"""

import sys
import os
sys.path.append('/home/apps/Tennis_one_set')

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from src.api.comprehensive_statistics_service import ComprehensiveStatisticsService, integrate_with_existing_system
from src.data.database_models import DatabaseManager

logger = logging.getLogger(__name__)

class BettingStatisticsIntegrator:
    """
    Integration layer for connecting comprehensive statistics with existing betting system
    """
    
    def __init__(self):
        self.stats_service = ComprehensiveStatisticsService()
        self.integration_helper = integrate_with_existing_system()
        
    def record_prediction_with_statistics(self, prediction_data: Dict, match_result: Dict = None) -> str:
        """
        Record a prediction and automatically create comprehensive match statistics
        
        Args:
            prediction_data: Standard prediction data from tennis system
            match_result: Match result data (if available)
        
        Returns:
            match_id: ID of recorded match statistics
        """
        try:
            # Use the integration helper to convert formats
            match_id = self.integration_helper(prediction_data, match_result)
            
            logger.info(f"📊 Integrated prediction into comprehensive statistics: {match_id}")
            return match_id
            
        except Exception as e:
            logger.error(f"❌ Error integrating prediction with statistics: {e}")
            return ""
    
    def update_match_outcome(self, match_id: str, match_result: Dict) -> bool:
        """
        Update match outcome and recalculate statistics
        """
        try:
            # Find existing match statistics record
            session = self.stats_service.session
            from src.data.database_models import MatchStatistics
            
            match_stats = session.query(MatchStatistics).filter(
                MatchStatistics.match_id == match_id
            ).first()
            
            if not match_stats:
                logger.warning(f"⚠️ Match statistics not found for ID: {match_id}")
                return False
            
            # Update match outcome
            match_stats.winner = match_result.get('winner')
            match_stats.match_score = match_result.get('score', '')
            match_stats.sets_won_p1 = match_result.get('sets_p1', 0)
            match_stats.sets_won_p2 = match_result.get('sets_p2', 0)
            match_stats.match_completed = True
            
            # Update prediction correctness
            if match_stats.predicted_winner and match_stats.winner:
                match_stats.prediction_correct = (match_stats.predicted_winner == match_stats.winner)
                
                # Calculate probability error
                if match_stats.prediction_probability:
                    actual_prob = 1.0 if match_stats.prediction_correct else 0.0
                    match_stats.probability_error = abs(match_stats.prediction_probability - actual_prob)
            
            # Update upset status
            if (match_stats.player1_rank and match_stats.player2_rank and match_stats.winner):
                if match_stats.winner == match_stats.player1_name and match_stats.player1_rank > match_stats.player2_rank:
                    match_stats.upset_occurred = True
                elif match_stats.winner == match_stats.player2_name and match_stats.player2_rank > match_stats.player1_rank:
                    match_stats.upset_occurred = True
            
            # Update player statistics
            self.stats_service._update_player_statistics(
                match_stats.player1_name, 
                match_stats.player2_name, 
                match_stats
            )
            
            session.commit()
            
            logger.info(f"📊 Updated match outcome: {match_id}")
            logger.info(f"   Winner: {match_stats.winner}")
            logger.info(f"   Prediction: {'✅ Correct' if match_stats.prediction_correct else '❌ Incorrect'}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating match outcome: {e}")
            self.stats_service.session.rollback()
            return False
    
    def record_live_betting_ratios(self, match_id: str, stage: str, betting_data: Dict) -> bool:
        """
        Record live betting ratios during match (e.g., start/end of sets)
        """
        try:
            from src.data.database_models import MatchStatistics, BettingRatioSnapshot
            session = self.stats_service.session
            
            # Find match statistics record
            match_stats = session.query(MatchStatistics).filter(
                MatchStatistics.match_id == match_id
            ).first()
            
            if not match_stats:
                logger.warning(f"⚠️ Match not found for betting ratios: {match_id}")
                return False
            
            # Update main match statistics based on stage
            if stage == 'start_set2':
                match_stats.start_set2_odds_p1 = betting_data.get('odds_p1')
                match_stats.start_set2_odds_p2 = betting_data.get('odds_p2')
                match_stats.start_set2_ratio_p1 = betting_data.get('ratio_p1')
                match_stats.start_set2_ratio_p2 = betting_data.get('ratio_p2')
            elif stage == 'end_set2':
                match_stats.end_set2_odds_p1 = betting_data.get('odds_p1')
                match_stats.end_set2_odds_p2 = betting_data.get('odds_p2')
                match_stats.end_set2_ratio_p1 = betting_data.get('ratio_p1')
                match_stats.end_set2_ratio_p2 = betting_data.get('ratio_p2')
            
            # Create detailed snapshot
            snapshot = BettingRatioSnapshot(
                match_statistics_id=match_stats.id,
                snapshot_stage=stage,
                current_set=betting_data.get('current_set'),
                current_game=betting_data.get('current_game'),
                sets_score=betting_data.get('sets_score'),
                player1_odds=betting_data.get('odds_p1'),
                player2_odds=betting_data.get('odds_p2'),
                player1_ratio=betting_data.get('ratio_p1'),
                player2_ratio=betting_data.get('ratio_p2'),
                total_market_volume=betting_data.get('market_volume'),
                bookmaker_source=betting_data.get('bookmaker', 'live_feed'),
                odds_movement=betting_data.get('odds_movement'),
                market_sentiment=betting_data.get('market_sentiment')
            )
            
            session.add(snapshot)
            session.commit()
            
            logger.info(f"📊 Recorded live betting ratios: {match_id} - {stage}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error recording live betting ratios: {e}")
            self.stats_service.session.rollback()
            return False
    
    def get_dashboard_summary(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Get complete dashboard summary combining all statistics
        """
        try:
            # Get comprehensive statistics
            comprehensive_stats = self.stats_service.get_comprehensive_match_statistics(days_back=days_back)
            
            if 'error' in comprehensive_stats:
                return {'error': comprehensive_stats['error']}
            
            # Extract key metrics for dashboard
            summary = comprehensive_stats['summary']
            matches = comprehensive_stats['matches']
            player_stats = comprehensive_stats['player_stats']
            betting_analysis = comprehensive_stats['betting_analysis']
            
            # Calculate additional dashboard metrics
            recent_matches = matches[:10]  # Last 10 matches
            top_players = player_stats[:10]  # Top 10 players
            
            # Performance trends
            completed_matches = [m for m in matches if m['result']['completed']]
            if completed_matches:
                recent_predictions = [m for m in completed_matches[-20:] if m['prediction']['predicted_winner']]
                recent_accuracy = len([m for m in recent_predictions if m['prediction']['correct']]) / len(recent_predictions) * 100 if recent_predictions else 0
            else:
                recent_accuracy = 0
            
            dashboard_summary = {
                'overview': {
                    'total_matches_tracked': summary['total_matches'],
                    'completed_matches': summary['completed_matches'],
                    'prediction_accuracy': summary['prediction_accuracy'],
                    'recent_accuracy_20_matches': round(recent_accuracy, 1),
                    'upsets_detected': summary['upsets_occurred'],
                    'upset_rate': summary['upset_rate']
                },
                'recent_activity': {
                    'recent_matches': recent_matches,
                    'matches_today': len([m for m in matches if m['match_date'].startswith(datetime.now().strftime('%Y-%m-%d'))]),
                    'active_tournaments': list(summary['tournaments'].keys())[:5]
                },
                'player_insights': {
                    'top_performers': top_players,
                    'most_tracked_player': top_players[0]['name'] if top_players else None,
                    'players_with_upsets': len([p for p in player_stats if p.get('upset_wins', 0) > 0])
                },
                'betting_insights': {
                    'matches_with_ratios': betting_analysis.get('analysis_summary', {}).get('total_matches_with_ratios', 0),
                    'significant_swings': betting_analysis.get('analysis_summary', {}).get('matches_with_significant_swings', 0),
                    'prediction_ratio_agreement': betting_analysis.get('prediction_ratio_correlation', {}).get('agreement_rate', 0)
                },
                'system_health': {
                    'data_quality': 'excellent' if summary['total_matches'] >= 50 else 'good' if summary['total_matches'] >= 20 else 'limited',
                    'last_match_date': matches[0]['match_date'] if matches else None,
                    'statistical_significance': 'high' if summary['total_matches'] >= 50 else 'medium' if summary['total_matches'] >= 20 else 'low'
                },
                'period': f'Last {days_back} days',
                'last_updated': datetime.now().isoformat()
            }
            
            return dashboard_summary
            
        except Exception as e:
            logger.error(f"❌ Error generating dashboard summary: {e}")
            return {'error': str(e)}
    
    def export_statistics_report(self, days_back: int = 30, format: str = 'json') -> str:
        """
        Export comprehensive statistics report
        """
        try:
            import json
            import os
            
            # Get comprehensive data
            stats = self.stats_service.get_comprehensive_match_statistics(days_back=days_back)
            dashboard = self.get_dashboard_summary(days_back=days_back)
            
            export_data = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'period': f'Last {days_back} days',
                    'report_type': 'comprehensive_betting_statistics',
                    'version': '1.0'
                },
                'dashboard_summary': dashboard,
                'detailed_statistics': stats,
                'export_info': {
                    'total_matches': len(stats.get('matches', [])),
                    'total_players': len(stats.get('player_stats', [])),
                    'data_quality': dashboard.get('system_health', {}).get('data_quality', 'unknown')
                }
            }
            
            # Create export file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_dir = '/home/apps/Tennis_one_set/data/exports'
            os.makedirs(export_dir, exist_ok=True)
            
            if format.lower() == 'json':
                export_file = f"{export_dir}/comprehensive_betting_statistics_{timestamp}.json"
                with open(export_file, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            else:
                export_file = f"{export_dir}/comprehensive_betting_statistics_{timestamp}.txt"
                with open(export_file, 'w') as f:
                    f.write(f"COMPREHENSIVE BETTING STATISTICS REPORT\n")
                    f.write(f"Generated: {export_data['report_metadata']['generated_at']}\n")
                    f.write(f"Period: {export_data['report_metadata']['period']}\n\n")
                    
                    f.write(f"SUMMARY:\n")
                    f.write(f"- Total matches: {dashboard['overview']['total_matches_tracked']}\n")
                    f.write(f"- Prediction accuracy: {dashboard['overview']['prediction_accuracy']}%\n")
                    f.write(f"- Upsets detected: {dashboard['overview']['upsets_detected']}\n\n")
            
            logger.info(f"📊 Exported statistics report: {export_file}")
            return export_file
            
        except Exception as e:
            logger.error(f"❌ Error exporting statistics report: {e}")
            return ""


# Integration hooks for existing system
def hook_prediction_system():
    """
    Integration hook for existing prediction system
    """
    integrator = BettingStatisticsIntegrator()
    
    def record_prediction_hook(prediction_data: Dict, match_result: Dict = None) -> str:
        """Hook function to be called from existing prediction system"""
        return integrator.record_prediction_with_statistics(prediction_data, match_result)
    
    def update_result_hook(match_id: str, match_result: Dict) -> bool:
        """Hook function to be called when match results are available"""
        return integrator.update_match_outcome(match_id, match_result)
    
    def record_live_ratios_hook(match_id: str, stage: str, betting_data: Dict) -> bool:
        """Hook function for recording live betting ratios"""
        return integrator.record_live_betting_ratios(match_id, stage, betting_data)
    
    return {
        'record_prediction': record_prediction_hook,
        'update_result': update_result_hook,
        'record_live_ratios': record_live_ratios_hook,
        'get_dashboard': integrator.get_dashboard_summary,
        'export_report': integrator.export_statistics_report
    }


# Example usage
if __name__ == "__main__":
    print("🔗 BETTING STATISTICS INTEGRATION - TESTING")
    print("=" * 60)
    
    # Initialize integrator
    integrator = BettingStatisticsIntegrator()
    
    # Test prediction integration
    test_prediction = {
        'player1': 'Rafael Nadal',
        'player2': 'Roger Federer',
        'player1_rank': 2,
        'player2_rank': 8,
        'tournament': 'French Open 2025',
        'surface': 'Clay',
        'match_date': datetime.now().isoformat(),
        'predicted_winner': 'Rafael Nadal',
        'our_probability': 0.78,
        'confidence': 'High',
        'ml_system': 'clay_specialist_v2',
        'key_factors': ['Clay court mastery', 'Historical dominance'],
        'recommendation': 'STRONG_BET',
        'edge': 15.3
    }
    
    print("1️⃣ Testing prediction integration...")
    match_id = integrator.record_prediction_with_statistics(test_prediction)
    print(f"   📊 Recorded match: {match_id}")
    
    # Test live betting ratios
    print("\n2️⃣ Testing live betting ratio recording...")
    live_ratios = {
        'odds_p1': 1.8, 'odds_p2': 2.0,
        'ratio_p1': 0.55, 'ratio_p2': 0.45,
        'current_set': 2, 'sets_score': '1-0',
        'market_volume': 200000,
        'odds_movement': 'stable'
    }
    
    if match_id:
        recorded = integrator.record_live_betting_ratios(match_id, 'start_set2', live_ratios)
        print(f"   📊 Live ratios recorded: {recorded}")
    
    # Test dashboard summary
    print("\n3️⃣ Testing dashboard summary...")
    dashboard = integrator.get_dashboard_summary(days_back=7)
    
    if 'error' not in dashboard:
        overview = dashboard['overview']
        print(f"   📊 Total matches: {overview['total_matches_tracked']}")
        print(f"   🎯 Accuracy: {overview['prediction_accuracy']}%")
        print(f"   📈 Data quality: {dashboard['system_health']['data_quality']}")
    else:
        print(f"   ❌ Error: {dashboard['error']}")
    
    # Test export
    print("\n4️⃣ Testing statistics export...")
    export_file = integrator.export_statistics_report(days_back=7)
    if export_file:
        print(f"   📁 Export file: {export_file}")
    
    print("\n🔗 Integration testing completed!")