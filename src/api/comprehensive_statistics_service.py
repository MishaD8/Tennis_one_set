#!/usr/bin/env python3
"""
📊 Comprehensive Match Statistics Service for Tennis Betting System
Handles all match statistics collection, storage, and retrieval for dashboard display
"""

import os
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from sqlalchemy.orm import sessionmaker
from sqlalchemy import and_, or_, desc, asc, func, case

from src.data.database_models import (
    DatabaseManager, Base, MatchStatistics, PlayerStatistics, 
    BettingRatioSnapshot
)

logger = logging.getLogger(__name__)

class ComprehensiveStatisticsService:
    """
    Comprehensive statistics service for tracking all match data and betting performance
    """
    
    def __init__(self, database_manager: DatabaseManager = None):
        self.db_manager = database_manager or DatabaseManager()
        self.session = self.db_manager.get_session()
        
        # Ensure tables exist
        try:
            Base.metadata.create_all(bind=self.db_manager.engine)
            logger.info("✅ Statistics tables created/verified")
        except Exception as e:
            logger.error(f"❌ Error creating statistics tables: {e}")
    
    def clear_existing_statistics(self) -> bool:
        """
        Clear all existing statistics data to start fresh
        """
        try:
            # Clear all statistics tables
            tables_to_clear = [
                BettingRatioSnapshot,
                PlayerStatistics,
                MatchStatistics
            ]
            
            cleared_counts = {}
            for table in tables_to_clear:
                count = self.session.query(table).count()
                self.session.query(table).delete()
                cleared_counts[table.__tablename__] = count
            
            self.session.commit()
            
            logger.info("🧹 Cleared existing statistics data:")
            for table_name, count in cleared_counts.items():
                logger.info(f"   - {table_name}: {count} records")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error clearing statistics: {e}")
            self.session.rollback()
            return False
    
    def record_match_statistics(self, match_data: Dict) -> str:
        """
        Record comprehensive match statistics including betting ratios
        
        Args:
            match_data: Complete match information including:
                - Basic match info (players, tournament, date)
                - Player ranks
                - Betting odds/ratios at different stages
                - Prediction data
                - Outcome data
        
        Returns:
            match_id: Unique identifier for the match statistics record
        """
        try:
            # Generate unique match ID if not provided
            match_id = match_data.get('match_id', f"match_{uuid.uuid4().hex[:12]}")
            
            # Extract basic match information
            match_date_str = match_data.get('match_date', datetime.now().isoformat())
            if isinstance(match_date_str, str):
                match_date = datetime.fromisoformat(match_date_str.replace('Z', '+00:00'))
            else:
                match_date = match_date_str
            
            player1_name = match_data.get('player1_name', match_data.get('player1', ''))
            player2_name = match_data.get('player2_name', match_data.get('player2', ''))
            tournament = match_data.get('tournament', 'Unknown Tournament')
            
            # Extract player ranks
            player1_rank = match_data.get('player1_rank')
            player2_rank = match_data.get('player2_rank')
            
            # Extract outcome data
            winner = match_data.get('winner', match_data.get('actual_winner'))
            loser = player2_name if winner == player1_name else player1_name if winner == player2_name else None
            match_score = match_data.get('match_score', match_data.get('score', ''))
            
            # Calculate sets won
            sets_won_p1 = match_data.get('sets_won_p1', 0)
            sets_won_p2 = match_data.get('sets_won_p2', 0)
            
            # Determine if upset occurred
            upset_occurred = False
            if player1_rank and player2_rank and winner:
                # Upset if lower-ranked player (higher rank number) won
                if winner == player1_name and player1_rank > player2_rank:
                    upset_occurred = True
                elif winner == player2_name and player2_rank > player1_rank:
                    upset_occurred = True
            
            # Extract betting ratios
            betting_data = match_data.get('betting_ratios', {})
            start_set2 = betting_data.get('start_set2', {})
            end_set2 = betting_data.get('end_set2', {})
            
            # Extract prediction data
            prediction_data = match_data.get('prediction', {})
            predicted_winner = prediction_data.get('predicted_winner', match_data.get('predicted_winner'))
            prediction_probability = prediction_data.get('probability', match_data.get('our_probability'))
            prediction_confidence = prediction_data.get('confidence', match_data.get('confidence'))
            ml_model_used = prediction_data.get('model_used', match_data.get('ml_system', 'unknown'))
            key_factors = prediction_data.get('key_factors', match_data.get('key_factors', []))
            
            # Calculate prediction correctness
            prediction_correct = None
            if predicted_winner and winner:
                prediction_correct = (predicted_winner == winner)
            
            # Calculate probability error
            probability_error = None
            if prediction_probability and prediction_correct is not None:
                actual_prob = 1.0 if prediction_correct else 0.0
                probability_error = abs(prediction_probability - actual_prob)
            
            # Create match statistics record
            match_stats = MatchStatistics(
                match_id=match_id,
                match_date=match_date,
                tournament=tournament,
                surface=match_data.get('surface'),
                round_name=match_data.get('round_name'),
                player1_name=player1_name,
                player1_rank=player1_rank,
                player2_name=player2_name,
                player2_rank=player2_rank,
                winner=winner,
                loser=loser,
                match_score=match_score,
                sets_won_p1=sets_won_p1,
                sets_won_p2=sets_won_p2,
                match_completed=bool(winner),
                # Betting ratios at start of set 2
                start_set2_odds_p1=start_set2.get('odds_p1'),
                start_set2_odds_p2=start_set2.get('odds_p2'),
                start_set2_ratio_p1=start_set2.get('ratio_p1'),
                start_set2_ratio_p2=start_set2.get('ratio_p2'),
                # Betting ratios at end of set 2
                end_set2_odds_p1=end_set2.get('odds_p1'),
                end_set2_odds_p2=end_set2.get('odds_p2'),
                end_set2_ratio_p1=end_set2.get('ratio_p1'),
                end_set2_ratio_p2=end_set2.get('ratio_p2'),
                # Prediction data
                predicted_winner=predicted_winner,
                prediction_probability=prediction_probability,
                prediction_confidence=prediction_confidence,
                ml_model_used=ml_model_used,
                key_prediction_factors=json.dumps(key_factors) if key_factors else None,
                # Outcome analysis
                prediction_correct=prediction_correct,
                upset_occurred=upset_occurred,
                probability_error=probability_error,
                # System tracking
                caught_by_system=True,  # All matches processed by system
                betting_recommendation=match_data.get('betting_recommendation'),
                edge_percentage=match_data.get('edge_percentage'),
                # Metadata
                data_source=match_data.get('data_source', 'tennis_system'),
                notes=match_data.get('notes')
            )
            
            self.session.add(match_stats)
            self.session.flush()  # Get the ID
            
            # Record betting ratio snapshots if provided
            if betting_data:
                self._record_betting_snapshots(match_stats.id, betting_data)
            
            # Update player statistics
            self._update_player_statistics(player1_name, player2_name, match_stats)
            
            self.session.commit()
            
            logger.info(f"📊 Recorded match statistics: {match_id}")
            logger.info(f"   Match: {player1_name} vs {player2_name}")
            logger.info(f"   Tournament: {tournament}")
            if winner:
                logger.info(f"   Result: {winner} won")
                logger.info(f"   Prediction: {'✅ Correct' if prediction_correct else '❌ Incorrect'}")
            
            return match_id
            
        except Exception as e:
            logger.error(f"❌ Error recording match statistics: {e}")
            self.session.rollback()
            return ""
    
    def _record_betting_snapshots(self, match_stats_id: int, betting_data: Dict):
        """Record betting ratio snapshots for different match stages"""
        try:
            for stage, stage_data in betting_data.items():
                if not isinstance(stage_data, dict):
                    continue
                
                snapshot = BettingRatioSnapshot(
                    match_statistics_id=match_stats_id,
                    snapshot_stage=stage,
                    current_set=stage_data.get('current_set'),
                    current_game=stage_data.get('current_game'),
                    sets_score=stage_data.get('sets_score'),
                    player1_odds=stage_data.get('odds_p1'),
                    player2_odds=stage_data.get('odds_p2'),
                    player1_ratio=stage_data.get('ratio_p1'),
                    player2_ratio=stage_data.get('ratio_p2'),
                    total_market_volume=stage_data.get('market_volume'),
                    bookmaker_source=stage_data.get('bookmaker', 'simulation'),
                    odds_movement=stage_data.get('odds_movement'),
                    market_sentiment=stage_data.get('market_sentiment')
                )
                
                self.session.add(snapshot)
                
        except Exception as e:
            logger.error(f"❌ Error recording betting snapshots: {e}")
    
    def _update_player_statistics(self, player1_name: str, player2_name: str, match_stats: MatchStatistics):
        """Update aggregated player statistics"""
        try:
            for player_name in [player1_name, player2_name]:
                if not player_name:
                    continue
                
                # Get or create player statistics
                player_stats = self.session.query(PlayerStatistics).filter(
                    PlayerStatistics.player_name == player_name
                ).first()
                
                if not player_stats:
                    player_stats = PlayerStatistics(player_name=player_name)
                    self.session.add(player_stats)
                
                # Update basic stats
                player_stats.total_matches_tracked = (player_stats.total_matches_tracked or 0) + 1
                
                if match_stats.winner == player_name:
                    player_stats.matches_won = (player_stats.matches_won or 0) + 1
                elif match_stats.winner and match_stats.winner != player_name:
                    player_stats.matches_lost = (player_stats.matches_lost or 0) + 1
                
                # Calculate win percentage
                total_completed = (player_stats.matches_won or 0) + (player_stats.matches_lost or 0)
                if total_completed > 0:
                    player_stats.win_percentage = ((player_stats.matches_won or 0) / total_completed) * 100
                
                # Update prediction stats
                if match_stats.predicted_winner == player_name:
                    player_stats.times_predicted_to_win = (player_stats.times_predicted_to_win or 0) + 1
                    if match_stats.winner == player_name:
                        player_stats.times_won_when_predicted = (player_stats.times_won_when_predicted or 0) + 1
                    
                    # Calculate prediction accuracy
                    if player_stats.times_predicted_to_win > 0:
                        player_stats.prediction_accuracy_for_player = (
                            (player_stats.times_won_when_predicted or 0) / player_stats.times_predicted_to_win
                        ) * 100
                
                # Update rank information
                if player_name == player1_name and match_stats.player1_rank:
                    current_rank = match_stats.player1_rank
                elif player_name == player2_name and match_stats.player2_rank:
                    current_rank = match_stats.player2_rank
                else:
                    current_rank = None
                
                if current_rank:
                    player_stats.current_rank = current_rank
                    if not player_stats.highest_rank or current_rank < player_stats.highest_rank:
                        player_stats.highest_rank = current_rank
                    if not player_stats.lowest_rank or current_rank > player_stats.lowest_rank:
                        player_stats.lowest_rank = current_rank
                
                # Update surface stats
                if match_stats.surface and match_stats.winner:
                    is_winner = (match_stats.winner == player_name)
                    surface = match_stats.surface.lower()
                    
                    if 'hard' in surface:
                        if is_winner:
                            player_stats.hard_court_wins = (player_stats.hard_court_wins or 0) + 1
                        else:
                            player_stats.hard_court_losses = (player_stats.hard_court_losses or 0) + 1
                    elif 'clay' in surface:
                        if is_winner:
                            player_stats.clay_court_wins = (player_stats.clay_court_wins or 0) + 1
                        else:
                            player_stats.clay_court_losses = (player_stats.clay_court_losses or 0) + 1
                    elif 'grass' in surface:
                        if is_winner:
                            player_stats.grass_court_wins = (player_stats.grass_court_wins or 0) + 1
                        else:
                            player_stats.grass_court_losses = (player_stats.grass_court_losses or 0) + 1
                
        except Exception as e:
            logger.error(f"❌ Error updating player statistics: {e}")
    
    def get_comprehensive_match_statistics(self, 
                                         days_back: int = 30,
                                         tournament: str = None,
                                         surface: str = None) -> Dict[str, Any]:
        """
        Get comprehensive match statistics for dashboard display
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            query = self.session.query(MatchStatistics).filter(
                MatchStatistics.match_date >= cutoff_date
            )
            
            if tournament:
                query = query.filter(MatchStatistics.tournament.ilike(f"%{tournament}%"))
            
            if surface:
                query = query.filter(MatchStatistics.surface.ilike(f"%{surface}%"))
            
            matches = query.order_by(MatchStatistics.match_date.desc()).all()
            
            if not matches:
                return {
                    'summary': {'total_matches': 0, 'message': f'No matches found in last {days_back} days'},
                    'matches': [],
                    'player_stats': [],
                    'betting_analysis': {}
                }
            
            # Convert matches to dictionaries
            match_list = []
            for match in matches:
                match_dict = {
                    'match_id': match.match_id,
                    'match_date': match.match_date.isoformat(),
                    'tournament': match.tournament,
                    'surface': match.surface,
                    'round_name': match.round_name,
                    'player1': {
                        'name': match.player1_name,
                        'rank': match.player1_rank
                    },
                    'player2': {
                        'name': match.player2_name,
                        'rank': match.player2_rank
                    },
                    'result': {
                        'winner': match.winner,
                        'score': match.match_score,
                        'sets_p1': match.sets_won_p1,
                        'sets_p2': match.sets_won_p2,
                        'completed': match.match_completed
                    },
                    'betting_ratios': {
                        'start_set2': {
                            'odds_p1': match.start_set2_odds_p1,
                            'odds_p2': match.start_set2_odds_p2,
                            'ratio_p1': match.start_set2_ratio_p1,
                            'ratio_p2': match.start_set2_ratio_p2
                        },
                        'end_set2': {
                            'odds_p1': match.end_set2_odds_p1,
                            'odds_p2': match.end_set2_odds_p2,
                            'ratio_p1': match.end_set2_ratio_p1,
                            'ratio_p2': match.end_set2_ratio_p2
                        }
                    },
                    'prediction': {
                        'predicted_winner': match.predicted_winner,
                        'probability': match.prediction_probability,
                        'confidence': match.prediction_confidence,
                        'model_used': match.ml_model_used,
                        'correct': match.prediction_correct,
                        'key_factors': json.loads(match.key_prediction_factors) if match.key_prediction_factors else []
                    },
                    'analysis': {
                        'upset_occurred': match.upset_occurred,
                        'probability_error': match.probability_error,
                        'edge_percentage': match.edge_percentage,
                        'betting_recommendation': match.betting_recommendation
                    }
                }
                match_list.append(match_dict)
            
            # Calculate summary statistics
            summary = self._calculate_match_summary(matches)
            
            # Get player statistics
            player_stats = self._get_top_player_statistics(limit=20)
            
            # Get betting analysis
            betting_analysis = self._calculate_betting_analysis(matches)
            
            return {
                'summary': summary,
                'matches': match_list,
                'player_stats': player_stats,
                'betting_analysis': betting_analysis,
                'period': f'Last {days_back} days',
                'filters': {
                    'tournament': tournament,
                    'surface': surface
                },
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting comprehensive match statistics: {e}")
            return {'error': str(e)}
    
    def _calculate_match_summary(self, matches: List[MatchStatistics]) -> Dict[str, Any]:
        """Calculate summary statistics from matches"""
        try:
            total_matches = len(matches)
            completed_matches = len([m for m in matches if m.match_completed])
            
            # Prediction analysis
            matches_with_predictions = [m for m in matches if m.predicted_winner is not None]
            correct_predictions = len([m for m in matches_with_predictions if m.prediction_correct])
            
            prediction_accuracy = 0.0
            if matches_with_predictions:
                prediction_accuracy = (correct_predictions / len(matches_with_predictions)) * 100
            
            # Upset analysis
            upsets = len([m for m in matches if m.upset_occurred])
            upset_rate = (upsets / completed_matches) * 100 if completed_matches > 0 else 0
            
            # Tournament breakdown
            tournaments = {}
            surfaces = {}
            for match in matches:
                tournaments[match.tournament] = tournaments.get(match.tournament, 0) + 1
                if match.surface:
                    surfaces[match.surface] = surfaces.get(match.surface, 0) + 1
            
            # Model performance
            model_performance = {}
            for match in matches_with_predictions:
                if match.ml_model_used:
                    if match.ml_model_used not in model_performance:
                        model_performance[match.ml_model_used] = {'total': 0, 'correct': 0}
                    model_performance[match.ml_model_used]['total'] += 1
                    if match.prediction_correct:
                        model_performance[match.ml_model_used]['correct'] += 1
            
            # Calculate accuracy for each model
            for model, stats in model_performance.items():
                stats['accuracy'] = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
            
            return {
                'total_matches': total_matches,
                'completed_matches': completed_matches,
                'matches_with_predictions': len(matches_with_predictions),
                'prediction_accuracy': round(prediction_accuracy, 1),
                'correct_predictions': correct_predictions,
                'upsets_occurred': upsets,
                'upset_rate': round(upset_rate, 1),
                'tournaments': dict(sorted(tournaments.items(), key=lambda x: x[1], reverse=True)),
                'surfaces': dict(sorted(surfaces.items(), key=lambda x: x[1], reverse=True)),
                'model_performance': model_performance
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating match summary: {e}")
            return {'error': str(e)}
    
    def _get_top_player_statistics(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get top player statistics ordered by matches tracked"""
        try:
            players = self.session.query(PlayerStatistics).order_by(
                PlayerStatistics.total_matches_tracked.desc()
            ).limit(limit).all()
            
            player_list = []
            for player in players:
                player_dict = {
                    'name': player.player_name,
                    'total_matches': player.total_matches_tracked,
                    'wins': player.matches_won,
                    'losses': player.matches_lost,
                    'win_percentage': round(player.win_percentage, 1),
                    'current_rank': player.current_rank,
                    'highest_rank': player.highest_rank,
                    'lowest_rank': player.lowest_rank,
                    'prediction_stats': {
                        'times_predicted_to_win': player.times_predicted_to_win,
                        'times_won_when_predicted': player.times_won_when_predicted,
                        'prediction_accuracy': round(player.prediction_accuracy_for_player, 1)
                    },
                    'surface_performance': {
                        'hard_court': {
                            'wins': player.hard_court_wins,
                            'losses': player.hard_court_losses,
                            'win_rate': round((player.hard_court_wins / max(player.hard_court_wins + player.hard_court_losses, 1)) * 100, 1)
                        },
                        'clay_court': {
                            'wins': player.clay_court_wins,
                            'losses': player.clay_court_losses,
                            'win_rate': round((player.clay_court_wins / max(player.clay_court_wins + player.clay_court_losses, 1)) * 100, 1)
                        },
                        'grass_court': {
                            'wins': player.grass_court_wins,
                            'losses': player.grass_court_losses,
                            'win_rate': round((player.grass_court_wins / max(player.grass_court_wins + player.grass_court_losses, 1)) * 100, 1)
                        }
                    },
                    'last_updated': player.last_updated.isoformat()
                }
                player_list.append(player_dict)
            
            return player_list
            
        except Exception as e:
            logger.error(f"❌ Error getting player statistics: {e}")
            return []
    
    def _calculate_betting_analysis(self, matches: List[MatchStatistics]) -> Dict[str, Any]:
        """Calculate comprehensive betting ratio analysis"""
        try:
            # Analyze betting ratio changes
            ratio_changes = []
            significant_swings = []
            
            for match in matches:
                if (match.start_set2_ratio_p1 and match.end_set2_ratio_p1 and 
                    match.start_set2_ratio_p2 and match.end_set2_ratio_p2):
                    
                    # Calculate ratio changes
                    p1_change = match.end_set2_ratio_p1 - match.start_set2_ratio_p1
                    p2_change = match.end_set2_ratio_p2 - match.start_set2_ratio_p2
                    
                    change_data = {
                        'match_id': match.match_id,
                        'player1': match.player1_name,
                        'player2': match.player2_name,
                        'p1_ratio_change': round(p1_change, 3),
                        'p2_ratio_change': round(p2_change, 3),
                        'winner': match.winner,
                        'upset': match.upset_occurred
                    }
                    
                    ratio_changes.append(change_data)
                    
                    # Identify significant swings (>10% change)
                    if abs(p1_change) > 0.1 or abs(p2_change) > 0.1:
                        significant_swings.append(change_data)
            
            # Analyze prediction vs betting ratio correlation
            ratio_prediction_correlation = []
            for match in matches:
                if (match.predicted_winner and match.start_set2_ratio_p1 and 
                    match.start_set2_ratio_p2 and match.prediction_correct is not None):
                    
                    # Determine if betting ratios favored the predicted winner
                    if match.predicted_winner == match.player1_name:
                        predicted_ratio = match.start_set2_ratio_p1
                        other_ratio = match.start_set2_ratio_p2
                    else:
                        predicted_ratio = match.start_set2_ratio_p2
                        other_ratio = match.start_set2_ratio_p1
                    
                    ratios_favor_prediction = predicted_ratio > other_ratio
                    
                    correlation_data = {
                        'match_id': match.match_id,
                        'predicted_winner': match.predicted_winner,
                        'prediction_correct': match.prediction_correct,
                        'ratios_favor_prediction': ratios_favor_prediction,
                        'predicted_ratio': round(predicted_ratio, 3),
                        'other_ratio': round(other_ratio, 3)
                    }
                    
                    ratio_prediction_correlation.append(correlation_data)
            
            # Calculate correlation metrics
            correlation_metrics = {'error': 'No correlation data'}
            if ratio_prediction_correlation:
                both_correct = len([c for c in ratio_prediction_correlation 
                                  if c['prediction_correct'] and c['ratios_favor_prediction']])
                both_wrong = len([c for c in ratio_prediction_correlation 
                                if not c['prediction_correct'] and not c['ratios_favor_prediction']])
                prediction_right_ratio_wrong = len([c for c in ratio_prediction_correlation 
                                                  if c['prediction_correct'] and not c['ratios_favor_prediction']])
                prediction_wrong_ratio_right = len([c for c in ratio_prediction_correlation 
                                                  if not c['prediction_correct'] and c['ratios_favor_prediction']])
                
                total_corr = len(ratio_prediction_correlation)
                agreement_rate = (both_correct + both_wrong) / total_corr * 100 if total_corr > 0 else 0
                
                correlation_metrics = {
                    'total_comparisons': total_corr,
                    'both_correct': both_correct,
                    'both_wrong': both_wrong,
                    'prediction_right_ratio_wrong': prediction_right_ratio_wrong,
                    'prediction_wrong_ratio_right': prediction_wrong_ratio_right,
                    'agreement_rate': round(agreement_rate, 1)
                }
            
            return {
                'ratio_changes': ratio_changes[-10:],  # Last 10 for brevity
                'significant_swings': significant_swings[-5:],  # Last 5 significant swings
                'prediction_ratio_correlation': correlation_metrics,
                'analysis_summary': {
                    'total_matches_with_ratios': len(ratio_changes),
                    'matches_with_significant_swings': len(significant_swings),
                    'average_ratio_change': round(sum([abs(c['p1_ratio_change']) + abs(c['p2_ratio_change']) 
                                                     for c in ratio_changes]) / len(ratio_changes), 3) if ratio_changes else 0
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating betting analysis: {e}")
            return {'error': str(e)}
    
    def get_player_detailed_statistics(self, player_name: str) -> Dict[str, Any]:
        """Get detailed statistics for a specific player"""
        try:
            player_stats = self.session.query(PlayerStatistics).filter(
                PlayerStatistics.player_name == player_name
            ).first()
            
            if not player_stats:
                return {'error': f'Player {player_name} not found in statistics'}
            
            # Get recent matches for this player
            recent_matches = self.session.query(MatchStatistics).filter(
                or_(
                    MatchStatistics.player1_name == player_name,
                    MatchStatistics.player2_name == player_name
                )
            ).order_by(MatchStatistics.match_date.desc()).limit(10).all()
            
            # Convert recent matches to dict
            match_history = []
            for match in recent_matches:
                is_player1 = (match.player1_name == player_name)
                opponent = match.player2_name if is_player1 else match.player1_name
                won = (match.winner == player_name)
                
                match_dict = {
                    'match_date': match.match_date.isoformat(),
                    'tournament': match.tournament,
                    'opponent': opponent,
                    'surface': match.surface,
                    'result': 'Won' if won else 'Lost',
                    'score': match.match_score,
                    'was_predicted_to_win': (match.predicted_winner == player_name),
                    'prediction_correct': match.prediction_correct
                }
                match_history.append(match_dict)
            
            # Build comprehensive player profile
            player_profile = {
                'name': player_stats.player_name,
                'current_rank': player_stats.current_rank,
                'rank_history': {
                    'highest_rank': player_stats.highest_rank,
                    'lowest_rank': player_stats.lowest_rank
                },
                'match_statistics': {
                    'total_matches': player_stats.total_matches_tracked,
                    'wins': player_stats.matches_won,
                    'losses': player_stats.matches_lost,
                    'win_percentage': round(player_stats.win_percentage, 1)
                },
                'prediction_performance': {
                    'times_predicted_to_win': player_stats.times_predicted_to_win,
                    'times_won_when_predicted': player_stats.times_won_when_predicted,
                    'prediction_accuracy': round(player_stats.prediction_accuracy_for_player, 1)
                },
                'surface_breakdown': {
                    'hard_court': {
                        'wins': player_stats.hard_court_wins,
                        'losses': player_stats.hard_court_losses,
                        'win_rate': round((player_stats.hard_court_wins / 
                                        max(player_stats.hard_court_wins + player_stats.hard_court_losses, 1)) * 100, 1)
                    },
                    'clay_court': {
                        'wins': player_stats.clay_court_wins,
                        'losses': player_stats.clay_court_losses,
                        'win_rate': round((player_stats.clay_court_wins / 
                                        max(player_stats.clay_court_wins + player_stats.clay_court_losses, 1)) * 100, 1)
                    },
                    'grass_court': {
                        'wins': player_stats.grass_court_wins,
                        'losses': player_stats.grass_court_losses,
                        'win_rate': round((player_stats.grass_court_wins / 
                                        max(player_stats.grass_court_wins + player_stats.grass_court_losses, 1)) * 100, 1)
                    }
                },
                'recent_matches': match_history,
                'last_updated': player_stats.last_updated.isoformat()
            }
            
            return player_profile
            
        except Exception as e:
            logger.error(f"❌ Error getting detailed player statistics: {e}")
            return {'error': str(e)}
    
    def __del__(self):
        """Cleanup database session"""
        if hasattr(self, 'session'):
            self.session.close()


# Integration functions for existing betting system
def integrate_with_existing_system():
    """Integration helper for connecting with existing betting components"""
    
    def record_match_from_prediction(prediction_data: Dict, match_result: Dict = None) -> str:
        """
        Convert existing prediction data format to comprehensive match statistics
        """
        try:
            stats_service = ComprehensiveStatisticsService()
            
            # Transform prediction format to match statistics format
            match_data = {
                'match_id': prediction_data.get('match_id', f"match_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                'player1_name': prediction_data.get('player1', ''),
                'player2_name': prediction_data.get('player2', ''),
                'tournament': prediction_data.get('tournament', 'Unknown'),
                'match_date': prediction_data.get('match_date', datetime.now().isoformat()),
                'surface': prediction_data.get('surface'),
                'player1_rank': prediction_data.get('player1_rank'),
                'player2_rank': prediction_data.get('player2_rank'),
                'prediction': {
                    'predicted_winner': prediction_data.get('predicted_winner'),
                    'probability': prediction_data.get('our_probability'),
                    'confidence': prediction_data.get('confidence'),
                    'model_used': prediction_data.get('ml_system'),
                    'key_factors': prediction_data.get('key_factors', [])
                },
                'betting_recommendation': prediction_data.get('recommendation'),
                'edge_percentage': prediction_data.get('edge')
            }
            
            # Add match result if provided
            if match_result:
                match_data.update({
                    'winner': match_result.get('winner'),
                    'match_score': match_result.get('score'),
                    'sets_won_p1': match_result.get('sets_p1', 0),
                    'sets_won_p2': match_result.get('sets_p2', 0)
                })
            
            return stats_service.record_match_statistics(match_data)
            
        except Exception as e:
            logger.error(f"❌ Error integrating match from prediction: {e}")
            return ""
    
    return record_match_from_prediction


# Example usage and testing
if __name__ == "__main__":
    print("📊 COMPREHENSIVE STATISTICS SERVICE - TESTING")
    print("=" * 60)
    
    # Initialize service
    stats_service = ComprehensiveStatisticsService()
    
    # Clear existing data
    print("1️⃣ Clearing existing statistics...")
    cleared = stats_service.clear_existing_statistics()
    print(f"✅ Statistics cleared: {cleared}")
    
    # Test match data
    test_matches = [
        {
            'match_id': 'test_001',
            'player1_name': 'Carlos Alcaraz',
            'player1_rank': 2,
            'player2_name': 'Novak Djokovic',
            'player2_rank': 1,
            'tournament': 'US Open',
            'surface': 'Hard',
            'match_date': '2025-08-20',
            'winner': 'Carlos Alcaraz',
            'match_score': '6-3, 2-6, 7-6, 6-2',
            'sets_won_p1': 3,
            'sets_won_p2': 1,
            'betting_ratios': {
                'start_set2': {
                    'odds_p1': 2.8, 'odds_p2': 1.4,
                    'ratio_p1': 0.35, 'ratio_p2': 0.65
                },
                'end_set2': {
                    'odds_p1': 2.2, 'odds_p2': 1.6,
                    'ratio_p1': 0.45, 'ratio_p2': 0.55
                }
            },
            'prediction': {
                'predicted_winner': 'Carlos Alcaraz',
                'probability': 0.68,
                'confidence': 'Medium',
                'model_used': 'ensemble_ml',
                'key_factors': ['Hard court advantage', 'Recent form']
            }
        }
    ]
    
    # Record test matches
    print("\n2️⃣ Recording test matches...")
    for match in test_matches:
        match_id = stats_service.record_match_statistics(match)
        print(f"📊 Recorded: {match_id}")
    
    # Get comprehensive statistics
    print("\n3️⃣ Getting comprehensive statistics...")
    stats = stats_service.get_comprehensive_match_statistics(days_back=30)
    print(f"📈 Found {stats['summary']['total_matches']} matches")
    print(f"🎯 Prediction accuracy: {stats['summary']['prediction_accuracy']}%")
    
    # Get player details
    print("\n4️⃣ Getting player statistics...")
    player_stats = stats_service.get_player_detailed_statistics('Carlos Alcaraz')
    if 'error' not in player_stats:
        print(f"👤 {player_stats['name']}: {player_stats['match_statistics']['total_matches']} matches")
    
    print("\n📊 Comprehensive Statistics Service testing completed!")