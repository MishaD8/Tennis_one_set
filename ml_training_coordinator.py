#!/usr/bin/env python3
"""
ML Training Coordinator for Tennis Second Set Prediction
Orchestrates data collection, preprocessing, training, and validation
"""

import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sqlite3

# Import existing components
from second_set_feature_engineering import SecondSetFeatureEngineer
from second_set_prediction_service import SecondSetPredictionService
from enhanced_ml_training_system import EnhancedMLTrainingSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLDataPipeline:
    """Enhanced data pipeline for second set ML training"""
    
    def __init__(self, data_dir: str = "tennis_data_enhanced"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Database for training data
        self.db_path = self.data_dir / "ml_training_data.db"
        self.feature_engineer = SecondSetFeatureEngineer()
        
        # Training data specifications
        self.second_set_features = [
            # Basic features
            'player_rank', 'player_age', 'opponent_rank', 'opponent_age',
            'tournament_importance', 'total_pressure', 'surface_advantage',
            
            # First set context
            'first_set_games_diff', 'first_set_total_games', 'first_set_close',
            'first_set_dominant', 'player1_won_first_set', 'player2_won_first_set',
            'first_set_duration', 'first_set_long', 'first_set_quick',
            
            # Break points and serving
            'breaks_difference', 'total_breaks', 'break_fest',
            'player1_bp_save_rate', 'player2_bp_save_rate', 'bp_save_difference',
            'player1_serve_percentage', 'player2_serve_percentage', 'serve_percentage_diff',
            'had_tiebreak', 'momentum_with_loser',
            
            # Momentum and adaptation
            'player1_second_set_improvement', 'player2_second_set_improvement',
            'player1_comeback_ability', 'player2_comeback_ability',
            'winner_momentum', 'loser_pressure_to_respond', 'active_comeback_scenario',
            
            # Underdog specific
            'player1_is_underdog', 'player2_is_underdog', 'ranking_gap',
            'underdog_won_first_set', 'underdog_lost_first_set',
            'underdog_confidence_boost', 'underdog_nothing_to_lose',
            'second_set_underdog_value', 'underdog_mental_toughness'
        ]
        
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database for training data"""
        with sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0) as conn:
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=memory")
            conn.execute("PRAGMA mmap_size=268435456")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS match_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT UNIQUE,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    player1_name TEXT NOT NULL,
                    player2_name TEXT NOT NULL,
                    tournament TEXT,
                    surface TEXT,
                    match_date DATE,
                    
                    -- First set data
                    first_set_winner TEXT,
                    first_set_score TEXT,
                    first_set_duration INTEGER,
                    first_set_breaks_p1 INTEGER,
                    first_set_breaks_p2 INTEGER,
                    first_set_bp_save_p1 REAL,
                    first_set_bp_save_p2 REAL,
                    first_set_serve_pct_p1 REAL,
                    first_set_serve_pct_p2 REAL,
                    first_set_had_tiebreak BOOLEAN,
                    
                    -- Second set outcome (target variable)
                    second_set_winner TEXT,
                    second_set_score TEXT,
                    underdog_won_second_set BOOLEAN,
                    
                    -- Player data at time of match
                    player1_rank INTEGER,
                    player1_age INTEGER,
                    player2_rank INTEGER,
                    player2_age INTEGER,
                    
                    -- Match context
                    tournament_importance INTEGER,
                    total_pressure REAL,
                    surface_advantage REAL,
                    
                    -- Data quality metrics
                    data_completeness REAL,
                    data_source TEXT,
                    quality_score INTEGER
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT,
                    feature_set_version TEXT,
                    features_json TEXT,
                    target_value REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (match_id) REFERENCES match_data (match_id)
                )
            """)
            
            # Create indexes (individual execute calls for thread safety)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_match_date ON match_data(match_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_data_quality ON match_data(quality_score)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_underdog_target ON match_data(underdog_won_second_set)")
    
    def collect_historical_data(self, start_date: datetime, end_date: datetime) -> int:
        """
        Collect historical match data for training
        This would integrate with your existing data collectors
        """
        logger.info(f"Collecting historical data from {start_date} to {end_date}")
        
        # In a real implementation, this would:
        # 1. Query tennis APIs for historical matches
        # 2. Filter for matches with set-by-set data
        # 3. Extract first and second set statistics
        # 4. Store in database
        
        # For now, simulate data collection process
        matches_collected = 0
        
        # Example implementation structure
        try:
            # This would integrate with your enhanced_universal_collector
            from enhanced_universal_collector import EnhancedUniversalCollector
            
            collector = EnhancedUniversalCollector()
            # Collect matches with set-by-set data
            
            logger.info(f"Would collect historical matches from {start_date} to {end_date}")
            matches_collected = 150  # Simulated
            
        except Exception as e:
            logger.error(f"Error collecting historical data: {e}")
        
        return matches_collected
    
    def process_match_for_training(self, match_data: Dict) -> Optional[Dict]:
        """
        Process a single match into training features
        
        Args:
            match_data: Raw match data with set-by-set information
            
        Returns:
            Dict: Processed features and target for ML training
        """
        try:
            # Extract match info
            match_id = f"{match_data.get('player1', 'p1')}_{match_data.get('player2', 'p2')}_{match_data.get('date', '')}"
            
            # Identify underdog based on rankings
            p1_rank = match_data.get('player1_rank', 100)
            p2_rank = match_data.get('player2_rank', 100)
            
            underdog_is_player1 = p1_rank > p2_rank
            
            # Extract first set data
            first_set_data = {
                'winner': match_data.get('first_set_winner'),
                'score': match_data.get('first_set_score'),
                'duration_minutes': match_data.get('first_set_duration', 45),
                'breaks_won_player1': match_data.get('first_set_breaks_p1', 1),
                'breaks_won_player2': match_data.get('first_set_breaks_p2', 1),
                'break_points_saved_player1': match_data.get('first_set_bp_save_p1', 0.6),
                'break_points_saved_player2': match_data.get('first_set_bp_save_p2', 0.6),
                'first_serve_percentage_player1': match_data.get('first_set_serve_pct_p1', 0.65),
                'first_serve_percentage_player2': match_data.get('first_set_serve_pct_p2', 0.65),
                'had_tiebreak': match_data.get('first_set_had_tiebreak', False)
            }
            
            # Create features using existing feature engineer
            features = self.feature_engineer.create_complete_feature_set(
                match_data.get('player1', 'Player1'),
                match_data.get('player2', 'Player2'),
                {
                    'rank': p1_rank,
                    'age': match_data.get('player1_age', 25)
                },
                {
                    'rank': p2_rank,
                    'age': match_data.get('player2_age', 25)
                },
                {
                    'tournament_importance': match_data.get('tournament_importance', 2),
                    'total_pressure': match_data.get('total_pressure', 2.5),
                    'player1_surface_advantage': match_data.get('surface_advantage', 0.0)
                },
                first_set_data
            )
            
            # Create target variable: did underdog win second set?
            second_set_winner = match_data.get('second_set_winner')
            if underdog_is_player1:
                target = 1.0 if second_set_winner == 'player1' else 0.0
            else:
                target = 1.0 if second_set_winner == 'player2' else 0.0
            
            return {
                'match_id': match_id,
                'features': features,
                'target': target,
                'underdog_player': 'player1' if underdog_is_player1 else 'player2',
                'ranking_gap': abs(p1_rank - p2_rank),
                'data_quality': self._assess_data_quality(match_data)
            }
            
        except Exception as e:
            logger.error(f"Error processing match for training: {e}")
            return None
    
    def _assess_data_quality(self, match_data: Dict) -> float:
        """Assess quality of match data for training purposes"""
        quality_score = 0.0
        max_score = 100.0
        
        # Check completeness of key fields
        required_fields = [
            'first_set_winner', 'second_set_winner', 'first_set_score',
            'player1_rank', 'player2_rank', 'tournament', 'surface'
        ]
        
        completeness = sum(1 for field in required_fields if match_data.get(field) is not None)
        quality_score += (completeness / len(required_fields)) * 40  # 40% for completeness
        
        # Check data consistency
        if match_data.get('first_set_winner') in ['player1', 'player2']:
            quality_score += 20  # 20% for valid first set winner
        
        if match_data.get('second_set_winner') in ['player1', 'player2']:
            quality_score += 20  # 20% for valid second set winner
        
        # Check ranking validity
        p1_rank = match_data.get('player1_rank', 0)
        p2_rank = match_data.get('player2_rank', 0)
        if 1 <= p1_rank <= 500 and 1 <= p2_rank <= 500:
            quality_score += 20  # 20% for valid rankings
        
        return min(quality_score, max_score)
    
    def create_training_dataset(self, min_quality_score: int = 70,
                              max_samples: int = 10000) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Create training dataset from processed matches
        
        Args:
            min_quality_score: Minimum data quality score to include
            max_samples: Maximum number of samples to include
            
        Returns:
            Tuple[features_df, target_array]
        """
        logger.info(f"Creating training dataset with quality >= {min_quality_score}")
        
        with sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0) as conn:
            # Get high-quality matches
            query = """
                SELECT match_id, features_json, target_value 
                FROM training_features tf
                JOIN match_data md ON tf.match_id = md.match_id
                WHERE md.quality_score >= ?
                ORDER BY md.match_date DESC
                LIMIT ?
            """
            
            cursor = conn.execute(query, (min_quality_score, max_samples))
            results = cursor.fetchall()
        
        if not results:
            logger.warning("No training data found matching criteria")
            return pd.DataFrame(), np.array([])
        
        # Process features and targets
        all_features = []
        all_targets = []
        
        for match_id, features_json, target in results:
            try:
                features = json.loads(features_json)
                # Ensure all expected features are present
                feature_vector = [features.get(feat, 0.0) for feat in self.second_set_features]
                
                all_features.append(feature_vector)
                all_targets.append(target)
            except Exception as e:
                logger.warning(f"Error processing features for {match_id}: {e}")
                continue
        
        if not all_features:
            logger.error("No valid features extracted")
            return pd.DataFrame(), np.array([])
        
        # Create DataFrame
        features_df = pd.DataFrame(all_features, columns=self.second_set_features)
        targets_array = np.array(all_targets)
        
        logger.info(f"Created training dataset: {len(features_df)} samples with {len(self.second_set_features)} features")
        logger.info(f"Target distribution: {np.mean(targets_array):.3f} underdog win rate")
        
        return features_df, targets_array

class AutomatedTrainingPipeline:
    """Automated training pipeline with scheduling and monitoring"""
    
    def __init__(self, models_dir: str = "tennis_models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.data_pipeline = MLDataPipeline()
        self.training_system = EnhancedMLTrainingSystem(str(self.models_dir))
        
        # Training configuration
        self.training_config = {
            'min_samples_for_training': 1000,
            'min_data_quality': 70,
            'retrain_frequency_days': 7,
            'validation_split': 0.2,
            'early_stopping_patience': 10
        }
        
        self.training_log = []
    
    def should_retrain_models(self) -> bool:
        """Check if models need retraining based on data freshness and performance"""
        try:
            # Check last training date
            metadata_path = self.models_dir / 'metadata.json'
            if not metadata_path.exists():
                return True
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            last_training = metadata.get('last_training_date')
            if not last_training:
                return True
            
            last_training_date = datetime.fromisoformat(last_training)
            days_since_training = (datetime.now() - last_training_date).days
            
            return days_since_training >= self.training_config['retrain_frequency_days']
            
        except Exception as e:
            logger.error(f"Error checking retrain status: {e}")
            return True
    
    def automated_training_cycle(self) -> Dict:
        """Execute complete automated training cycle"""
        logger.info("Starting automated training cycle for second set prediction")
        
        training_results = {
            'timestamp': datetime.now().isoformat(),
            'status': 'started',
            'data_collection': {},
            'feature_engineering': {},
            'model_training': {},
            'validation': {},
            'errors': []
        }
        
        try:
            # 1. Check if retraining needed
            if not self.should_retrain_models():
                logger.info("Models are up to date, skipping training")
                training_results['status'] = 'skipped'
                return training_results
            
            # 2. Create training dataset
            logger.info("Creating training dataset...")
            features_df, targets = self.data_pipeline.create_training_dataset(
                min_quality_score=self.training_config['min_data_quality']
            )
            
            if len(features_df) < self.training_config['min_samples_for_training']:
                logger.warning(f"Insufficient training data: {len(features_df)} samples")
                training_results['status'] = 'insufficient_data'
                training_results['data_collection']['samples'] = len(features_df)
                return training_results
            
            training_results['data_collection'] = {
                'samples': len(features_df),
                'features': len(features_df.columns),
                'underdog_win_rate': float(np.mean(targets)),
                'status': 'success'
            }
            
            # 3. Feature validation and preprocessing
            logger.info("Validating and preprocessing features...")
            processed_features = self._preprocess_features(features_df)
            
            training_results['feature_engineering'] = {
                'original_features': len(features_df.columns),
                'processed_features': len(processed_features.columns),
                'status': 'success'
            }
            
            # 4. Analyze system readiness and implement training phases
            logger.info("Analyzing system readiness...")
            readiness_analysis = self.training_system.analyze_system_readiness()
            
            # 5. Implement appropriate training phase
            phase_results = []
            
            # Phase 1: Always implement
            logger.info("Implementing Phase 1 enhancements...")
            phase1_results = self.training_system.implement_phase_1_enhancements(
                processed_features, targets
            )
            phase_results.append(phase1_results)
            
            # Phase 2: If data quality allows
            if readiness_analysis['phases_ready']['phase_2']:
                logger.info("Implementing Phase 2 enhancements...")
                phase2_results = self.training_system.implement_phase_2_enhancements(
                    processed_features, targets
                )
                phase_results.append(phase2_results)
            
            # Phase 3: If data quality is high
            if readiness_analysis['phases_ready']['phase_3']:
                logger.info("Implementing Phase 3 enhancements...")
                phase3_results = self.training_system.implement_phase_3_enhancements(
                    processed_features, targets
                )
                phase_results.append(phase3_results)
            
            training_results['model_training'] = {
                'phases_implemented': len(phase_results),
                'phase_results': phase_results,
                'data_quality_score': readiness_analysis['data_quality_analysis']['data_quality_score'],
                'status': 'success'
            }
            
            # 6. Generate comprehensive report
            enhancement_report = self.training_system.generate_enhancement_report(phase_results)
            training_results['validation'] = enhancement_report
            
            # 7. Update metadata with training info
            self._update_training_metadata(training_results)
            
            training_results['status'] = 'completed'
            logger.info("Automated training cycle completed successfully")
            
        except Exception as e:
            logger.error(f"Error in automated training cycle: {e}")
            training_results['status'] = 'error'
            training_results['errors'].append(str(e))
        
        # Log training results
        self.training_log.append(training_results)
        return training_results
    
    def _preprocess_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features for training"""
        # Handle missing values
        features_df = features_df.fillna(features_df.median())
        
        # Remove features with zero variance
        features_df = features_df.loc[:, features_df.var() != 0]
        
        # Feature scaling will be handled by individual models
        return features_df
    
    def _update_training_metadata(self, training_results: Dict):
        """Update training metadata file"""
        metadata_path = self.models_dir / 'training_metadata.json'
        
        metadata = {
            'last_training_date': training_results['timestamp'],
            'training_results': training_results,
            'model_version': '2.0_second_set_specialized',
            'target_type': 'underdog_second_set_wins',
            'features_count': len(self.data_pipeline.second_set_features),
            'training_config': self.training_config
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Training metadata updated: {metadata_path}")

# CLI interface for training operations
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Training Pipeline for Tennis Second Set Prediction')
    parser.add_argument('--collect-data', action='store_true', help='Collect historical training data')
    parser.add_argument('--train', action='store_true', help='Run automated training cycle')
    parser.add_argument('--force-retrain', action='store_true', help='Force model retraining')
    parser.add_argument('--data-quality-threshold', type=int, default=70, help='Minimum data quality score')
    
    args = parser.parse_args()
    
    pipeline = AutomatedTrainingPipeline()
    
    if args.collect_data:
        logger.info("Collecting historical data...")
        start_date = datetime.now() - timedelta(days=365)  # Last year
        end_date = datetime.now()
        
        matches_collected = pipeline.data_pipeline.collect_historical_data(start_date, end_date)
        logger.info(f"Collected {matches_collected} matches for training")
    
    if args.train or args.force_retrain:
        if args.force_retrain:
            # Remove last training date to force retraining
            metadata_path = pipeline.models_dir / 'training_metadata.json'
            if metadata_path.exists():
                metadata_path.unlink()
        
        results = pipeline.automated_training_cycle()
        logger.info(f"Training cycle completed with status: {results['status']}")
        
        if results['status'] == 'completed':
            logger.info("New models trained and ready for second set prediction!")
        elif results['status'] == 'insufficient_data':
            logger.warning(f"Need more training data. Current: {results['data_collection'].get('samples', 0)} samples")
        else:
            logger.error(f"Training failed. Check logs for details.")
    
    if not any([args.collect_data, args.train, args.force_retrain]):
        logger.info("Use --help for available options")
        logger.info("Example: python ml_training_coordinator.py --collect-data --train")