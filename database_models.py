#!/usr/bin/env python3
"""
🗄️ PostgreSQL Database Models for Tennis Prediction System
SQLAlchemy models for predictions, matches, and betting data
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

Base = declarative_base()

class Prediction(Base):
    """Main prediction records table"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    match_date = Column(DateTime, nullable=False)
    player1 = Column(String(100), nullable=False)
    player2 = Column(String(100), nullable=False)
    tournament = Column(String(100))
    surface = Column(String(20))
    round_name = Column(String(50))
    
    # Prediction data
    our_probability = Column(Float, nullable=False)
    confidence = Column(String(20))
    ml_system = Column(String(50))
    prediction_type = Column(String(30))
    key_factors = Column(Text)
    
    # Bookmaker data
    bookmaker_odds = Column(Float)
    bookmaker_probability = Column(Float)
    edge = Column(Float)
    recommendation = Column(String(20))
    
    # Results (filled later)
    actual_result = Column(String(20))
    actual_winner = Column(String(100))
    sets_won_p1 = Column(Integer)
    sets_won_p2 = Column(Integer)
    match_score = Column(String(50))
    
    # Analysis
    prediction_correct = Column(Boolean)
    probability_error = Column(Float)
    roi = Column(Float)
    
    # Relationships
    betting_records = relationship("BettingRecord", back_populates="prediction")

class BettingRecord(Base):
    """Betting/value bet records table"""
    __tablename__ = 'betting_records'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey('predictions.id'), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Match details
    player1 = Column(String(100), nullable=False)
    player2 = Column(String(100), nullable=False)
    tournament = Column(String(100))
    match_date = Column(DateTime, nullable=False)
    
    # Betting analysis
    our_probability = Column(Float, nullable=False)
    bookmaker_odds = Column(Float, nullable=False)
    implied_probability = Column(Float, nullable=False)
    edge_percentage = Column(Float, nullable=False)
    confidence_level = Column(String(20))
    bet_recommendation = Column(String(20))
    suggested_stake = Column(Float)
    
    # Results
    bet_result = Column(String(20))  # 'win', 'loss', 'pending'
    actual_return = Column(Float)
    roi_percentage = Column(Float)
    
    # Relationship
    prediction = relationship("Prediction", back_populates="betting_records")

class ModelPerformance(Base):
    """Model performance tracking table"""
    __tablename__ = 'model_performance'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    model_name = Column(String(50), nullable=False)
    
    # Performance metrics
    total_predictions = Column(Integer, default=0)
    correct_predictions = Column(Integer, default=0)
    accuracy_percentage = Column(Float, default=0.0)
    average_confidence = Column(Float)
    
    # Time period
    period_start = Column(DateTime)
    period_end = Column(DateTime)
    
    # Context
    surface = Column(String(20))
    tournament_level = Column(String(30))

class DatabaseManager:
    """PostgreSQL database connection and operations manager"""
    
    def __init__(self, database_url=None):
        if database_url is None:
            # Get from environment or use default
            database_url = os.getenv('DATABASE_URL', 
                'postgresql://tennis_user:tennis_pass@localhost:5432/tennis_predictions')
        
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)
        print("✅ PostgreSQL tables created successfully")
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def test_connection(self):
        """Test database connection"""
        try:
            session = self.get_session()
            session.execute("SELECT 1")
            session.close()
            print("✅ PostgreSQL connection successful")
            return True
        except Exception as e:
            print(f"❌ PostgreSQL connection failed: {e}")
            return False