#!/usr/bin/env python3
"""
🔄 Database Migration Utility
Migrate from SQLite/CSV to PostgreSQL and update existing code
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime
from typing import Dict, List
from database_service import TennisDatabaseService

class DatabaseMigration:
    """Utility to migrate existing data to PostgreSQL"""
    
    def __init__(self, sqlite_db_path="prediction_logs/predictions.db"):
        self.sqlite_path = sqlite_db_path
        self.postgres_service = TennisDatabaseService()
        
    def migrate_sqlite_to_postgres(self):
        """Migrate all data from SQLite to PostgreSQL"""
        if not os.path.exists(self.sqlite_path):
            print(f"⚠️ SQLite database not found: {self.sqlite_path}")
            return
        
        print("🔄 Starting SQLite to PostgreSQL migration...")
        
        # Connect to SQLite
        conn = sqlite3.connect(self.sqlite_path)
        
        try:
            # Migrate predictions
            predictions_df = pd.read_sql_query("SELECT * FROM predictions", conn)
            migrated_count = 0
            
            for _, row in predictions_df.iterrows():
                try:
                    # Convert SQLite row to PostgreSQL format
                    prediction_data = {
                        'match_date': datetime.fromisoformat(row['match_date']) if row['match_date'] else None,
                        'player1': row['player1'],
                        'player2': row['player2'],
                        'tournament': row['tournament'],
                        'surface': row['surface'],
                        'round_name': row['round_name'],
                        'our_probability': row['our_probability'],
                        'confidence': row['confidence'],
                        'ml_system': row['ml_system'],
                        'prediction_type': row['prediction_type'],
                        'key_factors': row['key_factors'],
                        'bookmaker_odds': row['bookmaker_odds'],
                        'bookmaker_probability': row['bookmaker_probability'],
                        'edge': row['edge'],
                        'recommendation': row['recommendation']
                    }
                    
                    # Log to PostgreSQL
                    prediction_id = self.postgres_service.log_prediction(prediction_data)
                    
                    # Update with results if available
                    if row['actual_result']:
                        result_data = {
                            'actual_result': row['actual_result'],
                            'actual_winner': row['actual_winner'],
                            'sets_won_p1': row['sets_won_p1'],
                            'sets_won_p2': row['sets_won_p2'],
                            'match_score': row['match_score']
                        }
                        self.postgres_service.update_match_result(prediction_id, result_data)
                    
                    migrated_count += 1
                    
                except Exception as e:
                    print(f"❌ Error migrating prediction {row.get('id', 'unknown')}: {e}")
            
            print(f"✅ Migrated {migrated_count} predictions from SQLite to PostgreSQL")
            
        except Exception as e:
            print(f"❌ Migration error: {e}")
        finally:
            conn.close()
    
    def migrate_csv_betting_data(self, csv_dir="betting_data"):
        """Migrate CSV betting files to PostgreSQL"""
        if not os.path.exists(csv_dir):
            print(f"⚠️ Betting CSV directory not found: {csv_dir}")
            return
        
        print("🔄 Migrating CSV betting data to PostgreSQL...")
        migrated_count = 0
        
        # Find all value_bets CSV files
        for filename in os.listdir(csv_dir):
            if filename.startswith('value_bets_') and filename.endswith('.csv'):
                csv_path = os.path.join(csv_dir, filename)
                
                try:
                    df = pd.read_csv(csv_path)
                    
                    for _, row in df.iterrows():
                        betting_data = {
                            'player1': row.get('player1', ''),
                            'player2': row.get('player2', ''),
                            'tournament': row.get('tournament', ''),
                            'match_date': datetime.fromisoformat(row['match_date']) if 'match_date' in row else datetime.now(),
                            'our_probability': row.get('our_probability', 0.0),
                            'bookmaker_odds': row.get('bookmaker_odds', 0.0),
                            'implied_probability': row.get('implied_probability', 0.0),
                            'edge_percentage': row.get('edge_percentage', 0.0),
                            'confidence_level': row.get('confidence_level', ''),
                            'bet_recommendation': row.get('bet_recommendation', ''),
                            'suggested_stake': row.get('suggested_stake', 0.0)
                        }
                        
                        self.postgres_service.log_betting_record(betting_data)
                        migrated_count += 1
                        
                except Exception as e:
                    print(f"❌ Error migrating CSV {filename}: {e}")
        
        print(f"✅ Migrated {migrated_count} betting records from CSV to PostgreSQL")

class DatabaseCodeUpdater:
    """Update existing code to use PostgreSQL instead of SQLite/CSV"""
    
    def __init__(self):
        self.postgres_service = TennisDatabaseService()
    
    def create_postgres_logger(self, data_dir="prediction_logs"):
        """Create PostgreSQL-based logger to replace SQLite logger"""
        
        postgres_logger_code = '''#!/usr/bin/env python3
"""
📊 POSTGRESQL PREDICTION LOGGER
Enhanced logger using PostgreSQL instead of SQLite
"""

import os
from datetime import datetime
from typing import Dict, List, Optional
from database_service import TennisDatabaseService

class PostgreSQLPredictionLogger:
    """PostgreSQL-based prediction logger"""
    
    def __init__(self, data_dir="prediction_logs"):
        self.data_dir = data_dir
        self.db_service = TennisDatabaseService()
        
        # Create directory for exports
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"✅ Created directory: {data_dir}")
    
    def log_prediction(self, prediction_data: Dict) -> int:
        """Log prediction to PostgreSQL"""
        return self.db_service.log_prediction(prediction_data)
    
    def log_betting_opportunity(self, betting_data: Dict, prediction_id: int = None) -> int:
        """Log betting opportunity to PostgreSQL"""
        return self.db_service.log_betting_record(betting_data, prediction_id)
    
    def update_result(self, prediction_id: int, result_data: Dict):
        """Update prediction with match result"""
        self.db_service.update_match_result(prediction_id, result_data)
    
    def get_accuracy_stats(self, model_name: str = None, days: int = 30) -> Dict:
        """Get model accuracy statistics"""
        return self.db_service.get_model_accuracy(model_name, days)
    
    def export_data(self, format: str = 'csv', days: int = 30) -> str:
        """Export predictions data"""
        df = self.db_service.export_predictions_to_dataframe(days)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'csv':
            export_file = os.path.join(self.data_dir, f'predictions_export_{timestamp}.csv')
            df.to_csv(export_file, index=False)
        elif format == 'excel':
            export_file = os.path.join(self.data_dir, f'predictions_export_{timestamp}.xlsx')
            df.to_excel(export_file, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"✅ Data exported: {export_file}")
        return export_file
    
    def get_recent_predictions(self, days: int = 7) -> List[Dict]:
        """Get recent predictions"""
        return self.db_service.get_recent_predictions(days)
    
    def get_value_bets(self, min_edge: float = 5.0, days: int = 7) -> List[Dict]:
        """Get recent value betting opportunities"""
        return self.db_service.get_value_bets(min_edge, days)
'''
        
        # Write the new PostgreSQL logger
        logger_path = os.path.join(data_dir, "postgresql_logger.py")
        with open(logger_path, 'w') as f:
            f.write(postgres_logger_code)
        
        print(f"✅ Created PostgreSQL logger: {logger_path}")
        return logger_path

def main():
    """Run database migration"""
    print("🚀 Tennis Database Migration to PostgreSQL")
    print("=" * 50)
    
    # Initialize migration
    migration = DatabaseMigration()
    
    # Test PostgreSQL connection
    if not migration.postgres_service.db_manager.test_connection():
        print("❌ PostgreSQL connection failed. Please check your database setup.")
        return
    
    # Migrate existing data
    migration.migrate_sqlite_to_postgres()
    migration.migrate_csv_betting_data()
    
    # Create new logger
    updater = DatabaseCodeUpdater()
    updater.create_postgres_logger()
    
    print("\n✅ Database migration completed!")
    print("\nNext steps:")
    print("1. Update your code to use PostgreSQLPredictionLogger")
    print("2. Replace CSV betting exports with database calls")
    print("3. Test the new database integration")

if __name__ == "__main__":
    main()