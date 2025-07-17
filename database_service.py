#!/usr/bin/env python3
"""
🗄️ Database Service Layer for Tennis Prediction System
High-level operations for predictions, betting, and analytics
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, and_, or_
import pandas as pd

from database_models import DatabaseManager, Prediction, BettingRecord, ModelPerformance

class TennisDatabaseService:
    """High-level database service for tennis predictions"""
    
    def __init__(self, database_url=None):
        self.db_manager = DatabaseManager(database_url)
        self.db_manager.create_tables()
    
    def log_prediction(self, prediction_data: Dict) -> int:
        """
        Log a new prediction to database
        
        Args:
            prediction_data: Dictionary with prediction details
            
        Returns:
            prediction_id: ID of created prediction record
        """
        session = self.db_manager.get_session()
        try:
            prediction = Prediction(
                match_date=prediction_data.get('match_date'),
                player1=prediction_data.get('player1'),
                player2=prediction_data.get('player2'),
                tournament=prediction_data.get('tournament'),
                surface=prediction_data.get('surface'),
                round_name=prediction_data.get('round_name'),
                our_probability=prediction_data.get('our_probability'),
                confidence=prediction_data.get('confidence'),
                ml_system=prediction_data.get('ml_system'),
                prediction_type=prediction_data.get('prediction_type'),
                key_factors=prediction_data.get('key_factors'),
                bookmaker_odds=prediction_data.get('bookmaker_odds'),
                bookmaker_probability=prediction_data.get('bookmaker_probability'),
                edge=prediction_data.get('edge'),
                recommendation=prediction_data.get('recommendation')
            )
            
            session.add(prediction)
            session.commit()
            prediction_id = prediction.id
            
            print(f"✅ Prediction logged: {prediction.player1} vs {prediction.player2}")
            return prediction_id
            
        except Exception as e:
            session.rollback()
            print(f"❌ Error logging prediction: {e}")
            raise
        finally:
            session.close()
    
    def log_betting_record(self, betting_data: Dict, prediction_id: int = None) -> int:
        """
        Log a betting/value bet record
        
        Args:
            betting_data: Dictionary with betting details
            prediction_id: Optional linked prediction ID
            
        Returns:
            betting_record_id: ID of created betting record
        """
        session = self.db_manager.get_session()
        try:
            betting_record = BettingRecord(
                prediction_id=prediction_id,
                player1=betting_data.get('player1'),
                player2=betting_data.get('player2'),
                tournament=betting_data.get('tournament'),
                match_date=betting_data.get('match_date'),
                our_probability=betting_data.get('our_probability'),
                bookmaker_odds=betting_data.get('bookmaker_odds'),
                implied_probability=betting_data.get('implied_probability'),
                edge_percentage=betting_data.get('edge_percentage'),
                confidence_level=betting_data.get('confidence_level'),
                bet_recommendation=betting_data.get('bet_recommendation'),
                suggested_stake=betting_data.get('suggested_stake')
            )
            
            session.add(betting_record)
            session.commit()
            betting_id = betting_record.id
            
            print(f"✅ Betting record logged: {betting_record.player1} vs {betting_record.player2}")
            return betting_id
            
        except Exception as e:
            session.rollback()
            print(f"❌ Error logging betting record: {e}")
            raise
        finally:
            session.close()
    
    def update_match_result(self, prediction_id: int, result_data: Dict):
        """
        Update prediction with actual match result
        
        Args:
            prediction_id: ID of prediction to update
            result_data: Dictionary with match results
        """
        session = self.db_manager.get_session()
        try:
            prediction = session.query(Prediction).filter(Prediction.id == prediction_id).first()
            
            if prediction:
                prediction.actual_result = result_data.get('actual_result')
                prediction.actual_winner = result_data.get('actual_winner')
                prediction.sets_won_p1 = result_data.get('sets_won_p1')
                prediction.sets_won_p2 = result_data.get('sets_won_p2')
                prediction.match_score = result_data.get('match_score')
                
                # Calculate accuracy
                predicted_winner = prediction.player1 if prediction.our_probability > 0.5 else prediction.player2
                prediction.prediction_correct = (predicted_winner == prediction.actual_winner)
                
                # Calculate probability error
                if prediction.actual_winner == prediction.player1:
                    prediction.probability_error = abs(prediction.our_probability - 1.0)
                else:
                    prediction.probability_error = abs(prediction.our_probability - 0.0)
                
                session.commit()
                print(f"✅ Match result updated for prediction {prediction_id}")
            else:
                print(f"❌ Prediction {prediction_id} not found")
                
        except Exception as e:
            session.rollback()
            print(f"❌ Error updating match result: {e}")
            raise
        finally:
            session.close()
    
    def get_recent_predictions(self, days: int = 30) -> List[Dict]:
        """Get recent predictions as list of dictionaries"""
        session = self.db_manager.get_session()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            predictions = session.query(Prediction).filter(
                Prediction.timestamp >= cutoff_date
            ).order_by(desc(Prediction.timestamp)).all()
            
            return [self._prediction_to_dict(p) for p in predictions]
            
        finally:
            session.close()
    
    def get_model_accuracy(self, model_name: str = None, days: int = 30) -> Dict:
        """Calculate model accuracy statistics"""
        session = self.db_manager.get_session()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            query = session.query(Prediction).filter(
                and_(
                    Prediction.timestamp >= cutoff_date,
                    Prediction.actual_result.isnot(None)
                )
            )
            
            if model_name:
                query = query.filter(Prediction.ml_system == model_name)
            
            predictions = query.all()
            
            if not predictions:
                return {'total': 0, 'correct': 0, 'accuracy': 0.0}
            
            total = len(predictions)
            correct = sum(1 for p in predictions if p.prediction_correct)
            accuracy = (correct / total) * 100
            
            return {
                'total': total,
                'correct': correct,
                'accuracy': accuracy,
                'model': model_name or 'all',
                'period_days': days
            }
            
        finally:
            session.close()
    
    def get_value_bets(self, min_edge: float = 5.0, days: int = 7) -> List[Dict]:
        """Get recent value betting opportunities"""
        session = self.db_manager.get_session()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            betting_records = session.query(BettingRecord).filter(
                and_(
                    BettingRecord.timestamp >= cutoff_date,
                    BettingRecord.edge_percentage >= min_edge
                )
            ).order_by(desc(BettingRecord.edge_percentage)).all()
            
            return [self._betting_record_to_dict(b) for b in betting_records]
            
        finally:
            session.close()
    
    def export_predictions_to_dataframe(self, days: int = 30) -> pd.DataFrame:
        """Export predictions to pandas DataFrame"""
        session = self.db_manager.get_session()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            predictions = session.query(Prediction).filter(
                Prediction.timestamp >= cutoff_date
            ).order_by(desc(Prediction.timestamp)).all()
            
            data = [self._prediction_to_dict(p) for p in predictions]
            return pd.DataFrame(data)
            
        finally:
            session.close()
    
    def _prediction_to_dict(self, prediction: Prediction) -> Dict:
        """Convert Prediction object to dictionary"""
        return {
            'id': prediction.id,
            'timestamp': prediction.timestamp,
            'match_date': prediction.match_date,
            'player1': prediction.player1,
            'player2': prediction.player2,
            'tournament': prediction.tournament,
            'surface': prediction.surface,
            'round_name': prediction.round_name,
            'our_probability': prediction.our_probability,
            'confidence': prediction.confidence,
            'ml_system': prediction.ml_system,
            'prediction_type': prediction.prediction_type,
            'key_factors': prediction.key_factors,
            'bookmaker_odds': prediction.bookmaker_odds,
            'bookmaker_probability': prediction.bookmaker_probability,
            'edge': prediction.edge,
            'recommendation': prediction.recommendation,
            'actual_result': prediction.actual_result,
            'actual_winner': prediction.actual_winner,
            'sets_won_p1': prediction.sets_won_p1,
            'sets_won_p2': prediction.sets_won_p2,
            'match_score': prediction.match_score,
            'prediction_correct': prediction.prediction_correct,
            'probability_error': prediction.probability_error,
            'roi': prediction.roi
        }
    
    def _betting_record_to_dict(self, betting_record: BettingRecord) -> Dict:
        """Convert BettingRecord object to dictionary"""
        return {
            'id': betting_record.id,
            'prediction_id': betting_record.prediction_id,
            'timestamp': betting_record.timestamp,
            'player1': betting_record.player1,
            'player2': betting_record.player2,
            'tournament': betting_record.tournament,
            'match_date': betting_record.match_date,
            'our_probability': betting_record.our_probability,
            'bookmaker_odds': betting_record.bookmaker_odds,
            'implied_probability': betting_record.implied_probability,
            'edge_percentage': betting_record.edge_percentage,
            'confidence_level': betting_record.confidence_level,
            'bet_recommendation': betting_record.bet_recommendation,
            'suggested_stake': betting_record.suggested_stake,
            'bet_result': betting_record.bet_result,
            'actual_return': betting_record.actual_return,
            'roi_percentage': betting_record.roi_percentage
        }