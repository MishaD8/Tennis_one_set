#!/usr/bin/env python3
"""
🧪 Database Integration Test
Test PostgreSQL database operations for tennis prediction system
"""

import os
import sys
from datetime import datetime, timedelta
from database_service import TennisDatabaseService

def test_basic_operations():
    """Test basic database operations"""
    print("🧪 Testing basic database operations...")
    
    try:
        # Initialize service
        db_service = TennisDatabaseService()
        
        # Test connection
        if not db_service.db_manager.test_connection():
            print("❌ Database connection failed")
            return False
        
        print("✅ Database connection successful")
        
        # Test prediction logging
        test_prediction = {
            'match_date': datetime.now() + timedelta(days=1),
            'player1': 'Novak Djokovic',
            'player2': 'Rafael Nadal',
            'tournament': 'Test Tournament',
            'surface': 'Clay',
            'round_name': 'Final',
            'our_probability': 0.65,
            'confidence': 'High',
            'ml_system': 'Neural Network',
            'prediction_type': 'Match Winner',
            'key_factors': 'Surface advantage, head-to-head record',
            'bookmaker_odds': 1.45,
            'bookmaker_probability': 0.69,
            'edge': 4.0,
            'recommendation': 'SKIP'
        }
        
        prediction_id = db_service.log_prediction(test_prediction)
        print(f"✅ Test prediction logged with ID: {prediction_id}")
        
        # Test betting record
        test_betting = {
            'player1': 'Novak Djokovic',
            'player2': 'Rafael Nadal',
            'tournament': 'Test Tournament',
            'match_date': datetime.now() + timedelta(days=1),
            'our_probability': 0.65,
            'bookmaker_odds': 1.45,
            'implied_probability': 0.69,
            'edge_percentage': 4.0,
            'confidence_level': 'High',
            'bet_recommendation': 'SKIP',
            'suggested_stake': 0.0
        }
        
        betting_id = db_service.log_betting_record(test_betting, prediction_id)
        print(f"✅ Test betting record logged with ID: {betting_id}")
        
        # Test result update
        test_result = {
            'actual_result': 'Player1_Win',
            'actual_winner': 'Novak Djokovic',
            'sets_won_p1': 3,
            'sets_won_p2': 1,
            'match_score': '6-4, 6-2, 6-3'
        }
        
        db_service.update_match_result(prediction_id, test_result)
        print("✅ Test result updated")
        
        # Test data retrieval
        recent_predictions = db_service.get_recent_predictions(days=1)
        print(f"✅ Retrieved {len(recent_predictions)} recent predictions")
        
        # Test accuracy calculation
        accuracy_stats = db_service.get_model_accuracy()
        print(f"✅ Accuracy stats: {accuracy_stats}")
        
        # Test DataFrame export
        df = db_service.export_predictions_to_dataframe(days=1)
        print(f"✅ Exported DataFrame with {len(df)} rows")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_migration_compatibility():
    """Test migration from SQLite format"""
    print("🔄 Testing migration compatibility...")
    
    try:
        from database_migration import DatabaseMigration
        
        migration = DatabaseMigration()
        print("✅ Migration class initialized")
        
        # Test PostgreSQL logger creation
        from database_migration import DatabaseCodeUpdater
        updater = DatabaseCodeUpdater()
        
        logger_path = updater.create_postgres_logger("test_logs")
        print(f"✅ PostgreSQL logger created: {logger_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration test failed: {e}")
        return False

def test_environment_setup():
    """Test environment configuration"""
    print("🔧 Testing environment setup...")
    
    try:
        # Check for .env file
        if os.path.exists('.env'):
            print("✅ .env file found")
        else:
            print("⚠️ .env file not found - using defaults")
        
        # Test environment variable reading
        database_url = os.getenv('DATABASE_URL', 
            'postgresql://tennis_user:tennis_pass@localhost:5432/tennis_predictions')
        print(f"✅ Database URL configured: {database_url.split('@')[0]}@***")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def cleanup_test_data():
    """Clean up test data"""
    print("🧹 Cleaning up test data...")
    
    try:
        db_service = TennisDatabaseService()
        session = db_service.db_manager.get_session()
        
        # Remove test predictions
        from database_models import Prediction, BettingRecord
        test_predictions = session.query(Prediction).filter(
            Prediction.tournament == 'Test Tournament'
        ).all()
        
        for prediction in test_predictions:
            session.delete(prediction)
        
        session.commit()
        session.close()
        
        print(f"✅ Cleaned up {len(test_predictions)} test records")
        return True
        
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        return False

def main():
    """Run all database integration tests"""
    print("🚀 Tennis Database Integration Tests")
    print("=" * 40)
    
    all_tests_passed = True
    
    # Test 1: Environment setup
    if not test_environment_setup():
        all_tests_passed = False
    
    print()
    
    # Test 2: Basic operations
    if not test_basic_operations():
        all_tests_passed = False
    
    print()
    
    # Test 3: Migration compatibility
    if not test_migration_compatibility():
        all_tests_passed = False
    
    print()
    
    # Cleanup
    if not cleanup_test_data():
        all_tests_passed = False
    
    print("\n" + "=" * 40)
    if all_tests_passed:
        print("🎉 All database integration tests passed!")
        print("\nDatabase integration is ready for production use.")
        print("\nTo complete the migration:")
        print("1. Run: python database_setup.py")
        print("2. Run: python database_migration.py")
        print("3. Update your main code to use PostgreSQLPredictionLogger")
    else:
        print("❌ Some tests failed")
        print("Please check the errors above and fix database setup")

if __name__ == "__main__":
    main()