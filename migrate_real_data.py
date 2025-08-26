#!/usr/bin/env python3
"""
🔄 Migration Script: Transfer Real Betting Data to Comprehensive Statistics
Fixes the disconnect between real ML predictions and fake statistics dashboard data
"""

import sys
import os
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.api.comprehensive_statistics_service import ComprehensiveStatisticsService
from src.data.database_models import DatabaseManager

class RealDataMigrator:
    """Migrates real prediction data to comprehensive statistics tables"""
    
    def __init__(self):
        self.db_path = "/home/apps/Tennis_one_set/data/tennis_predictions.db"
        self.stats_service = ComprehensiveStatisticsService()
        
    def get_real_predictions(self) -> List[Dict[str, Any]]:
        """Get all real predictions from the predictions table"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM predictions 
                WHERE player1 NOT IN ('Carlos Alcaraz', 'Jannik Sinner', 'Test Player A', 'Test Underdog Player', 'Verification Player 1')
                AND player2 NOT IN ('Novak Djokovic', 'Daniil Medvedev', 'Test Player B', 'Test Favorite Player', 'Verification Player 2')
                ORDER BY timestamp DESC
            """)
            
            predictions = []
            for row in cursor.fetchall():
                predictions.append(dict(row))
            
            conn.close()
            print(f"📊 Found {len(predictions)} real predictions to migrate")
            return predictions
            
        except Exception as e:
            print(f"❌ Error getting real predictions: {e}")
            return []
    
    def get_real_betting_records(self) -> List[Dict[str, Any]]:
        """Get all real betting records from betting_records table"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM betting_records 
                WHERE player1 NOT IN ('Carlos Alcaraz', 'Jannik Sinner', 'Test Player A', 'Verification Player 1')
                AND player2 NOT IN ('Novak Djokovic', 'Daniil Medvedev', 'Test Player B', 'Verification Player 2')
                ORDER BY timestamp DESC
            """)
            
            betting_records = []
            for row in cursor.fetchall():
                betting_records.append(dict(row))
            
            conn.close()
            print(f"💰 Found {len(betting_records)} real betting records to migrate")
            return betting_records
            
        except Exception as e:
            print(f"❌ Error getting real betting records: {e}")
            return []
    
    def migrate_prediction_to_statistics(self, prediction: Dict[str, Any]) -> str:
        """Convert a prediction record to comprehensive match statistics format"""
        try:
            # Create match data structure
            match_data = {
                'match_id': f"real_match_{prediction.get('id', 'unknown')}",
                'player1_name': prediction.get('player1', ''),
                'player2_name': prediction.get('player2', ''),
                'tournament': prediction.get('tournament', 'Unknown'),
                'surface': prediction.get('surface', 'Hard'),
                'round_name': prediction.get('round_name', 'Unknown'),
                'match_date': prediction.get('match_date', prediction.get('timestamp', datetime.now().isoformat())),
                
                # Prediction data
                'predicted_winner': self._determine_predicted_winner(prediction),
                'our_probability': prediction.get('our_probability'),
                'confidence': prediction.get('confidence'),
                'ml_system': prediction.get('ml_system', 'SecondSetUnderdogML'),
                'key_factors': self._parse_key_factors(prediction.get('key_factors')),
                'betting_recommendation': prediction.get('recommendation'),
                'edge_percentage': prediction.get('edge'),
                
                # Results (if available)
                'winner': prediction.get('actual_winner'),
                'match_score': prediction.get('match_score'),
                'sets_won_p1': prediction.get('sets_won_p1'),
                'sets_won_p2': prediction.get('sets_won_p2'),
                'prediction_correct': prediction.get('prediction_correct'),
                'probability_error': prediction.get('probability_error'),
                'roi': prediction.get('roi'),
                
                # System metadata
                'data_source': 'real_tennis_system',
                'notes': f"Migrated from predictions table ID: {prediction.get('id')}"
            }
            
            # Record in comprehensive statistics
            match_id = self.stats_service.record_match_statistics(match_data)
            return match_id
            
        except Exception as e:
            print(f"❌ Error migrating prediction {prediction.get('id')}: {e}")
            return ""
    
    def _determine_predicted_winner(self, prediction: Dict[str, Any]) -> str:
        """Determine who was predicted to win based on probability and recommendation"""
        try:
            # Check for explicit predicted winner
            if 'predicted_winner' in prediction and prediction['predicted_winner']:
                return prediction['predicted_winner']
            
            # For second set underdog system, check if it's a BET recommendation
            recommendation = prediction.get('recommendation', '')
            our_probability = prediction.get('our_probability', 0.5)
            
            if recommendation in ['BET', 'STRONG_BET']:
                # In underdog system, if probability > 0.5, player1 is predicted winner
                if our_probability > 0.5:
                    return prediction.get('player1', '')
                else:
                    return prediction.get('player2', '')
            
            # Default: higher probability player
            if our_probability > 0.5:
                return prediction.get('player1', '')
            else:
                return prediction.get('player2', '')
                
        except Exception as e:
            print(f"⚠️ Could not determine predicted winner: {e}")
            return ""
    
    def _parse_key_factors(self, key_factors_str: str) -> List[str]:
        """Parse key factors from string format"""
        try:
            if not key_factors_str:
                return []
            
            # Try to parse as JSON first
            if key_factors_str.startswith('['):
                return json.loads(key_factors_str)
            
            # Otherwise split by common delimiters
            return [factor.strip() for factor in key_factors_str.split(',')]
            
        except Exception as e:
            print(f"⚠️ Could not parse key factors: {e}")
            return []
    
    def run_migration(self):
        """Run the complete migration process"""
        print("🔄 REAL DATA MIGRATION STARTING")
        print("=" * 50)
        
        # Get real data
        real_predictions = self.get_real_predictions()
        
        if not real_predictions:
            print("⚠️ No real predictions found to migrate")
            return
        
        # Migrate each prediction
        migrated_count = 0
        for prediction in real_predictions:
            match_id = self.migrate_prediction_to_statistics(prediction)
            if match_id:
                migrated_count += 1
                print(f"✅ Migrated: {prediction.get('player1')} vs {prediction.get('player2')} -> {match_id}")
        
        print(f"\n📊 Migration completed: {migrated_count}/{len(real_predictions)} matches migrated")
        
        # Verify migration
        print("\n🔍 Verification:")
        stats = self.stats_service.get_comprehensive_match_statistics(days_back=60)
        print(f"   - Total matches in statistics: {stats['summary']['total_matches']}")
        print(f"   - Prediction accuracy: {stats['summary']['prediction_accuracy']}%")
        print(f"   - Players tracked: {len(stats['player_stats'])}")
        
        # Show real player names found
        if stats['matches']:
            print("\n👤 Real players now in system:")
            players = set()
            for match in stats['matches'][:5]:  # Show first 5
                players.add(match['player1']['name'])
                players.add(match['player2']['name'])
            for player in sorted(players):
                if player:  # Only show non-empty names
                    print(f"   - {player}")
        
        print("\n🎉 Real data migration completed successfully!")

def main():
    """Main migration function"""
    migrator = RealDataMigrator()
    migrator.run_migration()

if __name__ == "__main__":
    main()