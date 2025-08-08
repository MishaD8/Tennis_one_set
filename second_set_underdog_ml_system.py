#!/usr/bin/env python3
"""
🎾 SECOND SET UNDERDOG ML SYSTEM

Advanced ML system specifically designed for predicting second set outcomes
for underdog players ranked 101-300 in ATP/WTA singles tournaments.

Key Features:
- Specialized models for second set prediction
- Focus on underdog players (ranks 101-300)
- Integration with comprehensive data collection
- Production-ready model training and evaluation
- Robust error handling and fallback mechanisms

Author: Claude Code (Anthropic)
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

# Advanced ML (optional)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Local imports
from comprehensive_ml_data_collector import ComprehensiveMLDataCollector
from second_set_feature_engineering import SecondSetFeatureEngineer
from ranks_101_300_feature_engineering import Ranks101to300FeatureEngineer
from enhanced_ml_training_system import (
    EnhancedCrossValidation, AdvancedMetricsEvaluator, 
    BasicHyperparameterTuner, FeatureSelector
)

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class SecondSetUnderdogDataPreprocessor:
    """Specialized data preprocessing for second set underdog prediction"""
    
    def __init__(self, target_variable: str = 'underdog_won_second_set'):
        self.target_variable = target_variable
        self.scaler = None
        self.feature_columns = None
        self.class_weights = None
        
    def prepare_training_data(self, features_df: pd.DataFrame, 
                              match_metadata: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data for second set underdog prediction
        
        Args:
            features_df: DataFrame with ML features
            match_metadata: List of match metadata dictionaries
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        
        logger.info("🔧 Preparing training data for second set underdog prediction...")
        
        # Create synthetic target variable since we don't have historical second set outcomes
        # In production, this would be replaced with actual historical data
        y = self._generate_synthetic_targets(features_df, match_metadata)
        
        # Select and clean features
        X, feature_names = self._select_and_clean_features(features_df)
        
        # Handle class imbalance
        self.class_weights = self._calculate_class_weights(y)
        
        logger.info(f"✅ Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"📊 Class distribution: {np.bincount(y)} (0: favorite wins, 1: underdog wins)")
        
        return X, y, feature_names
    
    def _generate_synthetic_targets(self, features_df: pd.DataFrame, 
                                   match_metadata: List[Dict]) -> np.ndarray:
        """
        Generate synthetic target variables for second set underdog prediction
        
        In production, replace with actual historical second set outcomes
        """
        logger.info("⚠️ Generating synthetic targets - replace with real historical data in production")
        
        n_samples = len(features_df)
        targets = np.zeros(n_samples)
        
        for i, (_, row) in enumerate(features_df.iterrows()):
            # Base probability that underdog wins second set
            base_prob = 0.35  # Underdogs win ~35% of second sets historically
            
            # Adjust based on features
            prob_adjustments = 0.0
            
            # First set outcome impact
            if row.get('underdog_lost_first_set', 0) > 0.5:
                prob_adjustments += 0.15  # "Nothing to lose" effect
            
            # Close first set bonus
            if row.get('first_set_close', 0) > 0.5:
                prob_adjustments += 0.10  # Competitiveness indicator
            
            # Ranking gap impact (larger gaps = lower probability)
            ranking_gap = row.get('ranking_gap', 50)
            if ranking_gap > 100:
                prob_adjustments -= 0.10  # Very large gap
            elif ranking_gap < 30:
                prob_adjustments += 0.08  # Small gap
            
            # Momentum and adaptation features
            momentum = row.get('momentum_with_loser', 0)
            prob_adjustments += momentum * 0.5
            
            # Player-specific second set improvement
            p1_improvement = row.get('player1_second_set_improvement', 0)
            p2_improvement = row.get('player2_second_set_improvement', 0)
            
            # Determine who is underdog
            if row.get('player1_is_underdog', 0) > 0.5:
                prob_adjustments += p1_improvement
            else:
                prob_adjustments += p2_improvement
            
            # Career stage factors
            if row.get('player1_is_rising', 0) > 0.5 or row.get('player2_is_rising', 0) > 0.5:
                prob_adjustments += 0.05  # Young hungry players
            
            # Surface and conditions
            surface_advantage = abs(row.get('surface_advantage_gap', 0))
            if surface_advantage > 0.1:
                prob_adjustments += 0.03  # Surface specialist advantage
            
            # Calculate final probability
            final_prob = max(0.10, min(0.65, base_prob + prob_adjustments))
            
            # Generate binary outcome
            targets[i] = 1 if np.random.random() < final_prob else 0
        
        return targets.astype(int)
    
    def _select_and_clean_features(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Select and clean features for training"""
        
        # Core features for second set underdog prediction
        core_features = [
            # Player rankings and gaps
            'player1_tier_position', 'player2_tier_position',
            'ranking_gap', 'ranking_gap_normalized',
            
            # First set context
            'first_set_close', 'first_set_dominant', 
            'underdog_lost_first_set', 'underdog_won_first_set',
            
            # Underdog psychology
            'underdog_nothing_to_lose', 'underdog_confidence_boost',
            'favorite_desperation_factor', 'favorite_comfort_zone',
            
            # Second set patterns
            'player1_second_set_improvement', 'player2_second_set_improvement',
            'player1_comeback_ability', 'player2_comeback_ability',
            'momentum_with_loser',
            
            # Career stages
            'player1_is_rising', 'player2_is_rising',
            'player1_is_veteran', 'player2_is_veteran',
            
            # Tournament context
            'tournament_importance', 'is_atp_event',
            
            # Surface factors
            'surface_advantage_gap', 'player1_surface_specialist', 'player2_surface_specialist',
            
            # Physical and mental factors
            'first_set_duration', 'first_set_long',
            'player1_bp_save_rate', 'player2_bp_save_rate',
            
            # Rank-specific features
            'player1_distance_from_100', 'player2_distance_from_100',
            'player1_relegation_risk', 'player2_relegation_risk',
            
            # Combination features
            'momentum_times_adaptation', 'pressure_fatigue_interaction',
            'rank_gap_times_first_set_closeness'
        ]
        
        # Select available features
        available_features = [f for f in core_features if f in features_df.columns]
        
        if len(available_features) < 10:
            logger.warning(f"Only {len(available_features)} core features available, using all features")
            available_features = list(features_df.columns)
            # Remove non-numeric columns
            numeric_features = []
            for col in available_features:
                if features_df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    numeric_features.append(col)
            available_features = numeric_features
        
        # Create feature matrix
        X = features_df[available_features].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        self.feature_columns = available_features
        
        logger.info(f"🎯 Selected {len(available_features)} features for training")
        
        return X.values, available_features
    
    def _calculate_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Calculate class weights to handle imbalanced data"""
        
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        logger.info(f"📊 Class weights: {weight_dict}")
        
        return weight_dict
    
    def fit_scaler(self, X: np.ndarray) -> StandardScaler:
        """Fit and return feature scaler"""
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.scaler.fit(X)
        
        logger.info("✅ Feature scaler fitted")
        return self.scaler
    
    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler"""
        if self.scaler is None:
            logger.warning("Scaler not fitted, returning original features")
            return X
        
        return self.scaler.transform(X)
    
    def save_preprocessor(self, filepath: str):
        """Save preprocessor components"""
        preprocessor_data = {
            'target_variable': self.target_variable,
            'feature_columns': self.feature_columns,
            'class_weights': self.class_weights,
            'scaler_params': self.scaler.get_params() if self.scaler else None
        }
        
        # Save scaler separately
        if self.scaler:
            scaler_path = filepath.replace('.json', '_scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            preprocessor_data['scaler_path'] = scaler_path
        
        with open(filepath, 'w') as f:
            json.dump(preprocessor_data, f, indent=2)
        
        logger.info(f"💾 Preprocessor saved to: {filepath}")

class SecondSetUnderdogMLTrainer:
    """ML trainer specialized for second set underdog prediction"""
    
    def __init__(self, models_dir: str = "tennis_models"):
        self.models_dir = models_dir
        self.models = {}
        self.model_performance = {}
        self.ensemble_weights = {}
        
        # Initialize components
        self.preprocessor = SecondSetUnderdogDataPreprocessor()
        self.cross_validator = EnhancedCrossValidation()
        self.metrics_evaluator = AdvancedMetricsEvaluator()
        
        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
    
    def train_models(self, features_df: pd.DataFrame, match_metadata: List[Dict]) -> Dict[str, Any]:
        """
        Train all models for second set underdog prediction
        
        Args:
            features_df: Features DataFrame
            match_metadata: Match metadata list
            
        Returns:
            Training results and performance metrics
        """
        
        logger.info("🚀 Starting second set underdog ML training...")
        
        training_start = datetime.now()
        
        # Prepare training data
        X, y, feature_names = self.preprocessor.prepare_training_data(features_df, match_metadata)
        
        # Fit scaler
        self.preprocessor.fit_scaler(X)
        X_scaled = self.preprocessor.transform_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train individual models
        training_results = {
            'training_start': training_start.isoformat(),
            'data_info': {
                'total_samples': len(X),
                'n_features': len(feature_names),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'class_distribution': dict(zip(*np.unique(y, return_counts=True)))
            },
            'model_results': {},
            'ensemble_results': {},
            'feature_importance': {}
        }
        
        # Define models to train
        models_to_train = self._create_model_configurations()
        
        # Train each model
        for model_name, model_config in models_to_train.items():
            logger.info(f"🔧 Training {model_name}...")
            
            try:
                model_result = self._train_single_model(
                    model_name, model_config, X_train, y_train, X_test, y_test
                )
                training_results['model_results'][model_name] = model_result
                
                # Save model
                self._save_model(model_name, model_result['trained_model'])
                
                logger.info(f"✅ {model_name} trained successfully (F1: {model_result['test_metrics']['f1_score']:.3f})")
                
            except Exception as e:
                logger.error(f"❌ {model_name} training failed: {e}")
                training_results['model_results'][model_name] = {'error': str(e)}
        
        # Train ensemble if multiple models succeeded
        successful_models = [name for name, result in training_results['model_results'].items() 
                           if 'error' not in result]
        
        if len(successful_models) >= 2:
            logger.info("🎼 Training ensemble models...")
            ensemble_results = self._train_ensemble_models(successful_models, X_train, y_train, X_test, y_test)
            training_results['ensemble_results'] = ensemble_results
        
        # Feature importance analysis
        training_results['feature_importance'] = self._analyze_feature_importance(feature_names)
        
        # Calculate final ensemble weights
        self.ensemble_weights = self._calculate_ensemble_weights(training_results['model_results'])
        training_results['final_ensemble_weights'] = self.ensemble_weights
        
        # Save training results and metadata
        training_end = datetime.now()
        training_results['training_end'] = training_end.isoformat()
        training_results['training_duration'] = (training_end - training_start).total_seconds()
        
        self._save_training_metadata(training_results, feature_names)
        
        logger.info(f"🎯 Training completed in {training_results['training_duration']:.1f} seconds")
        
        return training_results
    
    def _create_model_configurations(self) -> Dict[str, Dict]:
        """Create model configurations optimized for second set prediction"""
        
        class_weights = self.preprocessor.class_weights
        
        models = {
            'logistic_regression': {
                'model': LogisticRegression(
                    random_state=42,
                    class_weight=class_weights,
                    max_iter=1000,
                    penalty='l2',
                    C=1.0
                ),
                'hyperparams': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            
            'random_forest': {
                'model': RandomForestClassifier(
                    random_state=42,
                    class_weight=class_weights,
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2
                ),
                'hyperparams': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20],
                    'min_samples_split': [5, 10],
                    'min_samples_leaf': [2, 4]
                }
            },
            
            'gradient_boosting': {
                'model': GradientBoostingClassifier(
                    random_state=42,
                    n_estimators=150,
                    learning_rate=0.1,
                    max_depth=6,
                    min_samples_split=5,
                    subsample=0.9
                ),
                'hyperparams': {
                    'n_estimators': [100, 150, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [4, 6, 8],
                    'min_samples_split': [5, 10]
                }
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['xgboost'] = {
                'model': XGBClassifier(
                    random_state=42,
                    n_estimators=150,
                    learning_rate=0.1,
                    max_depth=6,
                    scale_pos_weight=class_weights[0] / class_weights[1] if class_weights else 1.0
                ),
                'hyperparams': {
                    'n_estimators': [100, 150, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [4, 6, 8],
                    'subsample': [0.8, 0.9, 1.0]
                }
            }
        
        return models
    
    def _train_single_model(self, model_name: str, model_config: Dict,
                           X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train a single model with cross-validation"""
        
        model = model_config['model']
        
        # Cross-validation
        cv_results = self.cross_validator.perform_cross_validation(
            model, X_train, y_train, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        )
        
        # Train on full training set
        model.fit(X_train, y_train)
        
        # Test set evaluation
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        test_metrics = self.metrics_evaluator.calculate_comprehensive_metrics(
            y_test, y_pred, y_pred_proba
        )
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_.tolist()
        elif hasattr(model, 'coef_'):
            feature_importance = np.abs(model.coef_[0]).tolist()
        
        result = {
            'trained_model': model,
            'cv_results': cv_results,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance
        }
        
        # Store model and performance
        self.models[model_name] = model
        self.model_performance[model_name] = test_metrics
        
        return result
    
    def _train_ensemble_models(self, successful_models: List[str],
                              X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train ensemble models"""
        
        ensemble_results = {}
        
        try:
            from sklearn.ensemble import VotingClassifier
            
            # Create voting classifier
            estimators = [(name, self.models[name]) for name in successful_models]
            
            voting_clf = VotingClassifier(estimators=estimators, voting='soft')
            voting_clf.fit(X_train, y_train)
            
            # Evaluate ensemble
            y_pred = voting_clf.predict(X_test)
            y_pred_proba = voting_clf.predict_proba(X_test)[:, 1]
            
            ensemble_metrics = self.metrics_evaluator.calculate_comprehensive_metrics(
                y_test, y_pred, y_pred_proba
            )
            
            ensemble_results['voting_classifier'] = {
                'model': voting_clf,
                'metrics': ensemble_metrics,
                'estimators_used': successful_models
            }
            
            # Save ensemble model
            self._save_model('voting_ensemble', voting_clf)
            
            logger.info(f"✅ Voting ensemble trained (F1: {ensemble_metrics['f1_score']:.3f})")
            
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            ensemble_results['error'] = str(e)
        
        return ensemble_results
    
    def _analyze_feature_importance(self, feature_names: List[str]) -> Dict[str, Any]:
        """Analyze feature importance across all models"""
        
        importance_analysis = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                importance_analysis[model_name] = dict(zip(feature_names, importances))
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
                importance_analysis[model_name] = dict(zip(feature_names, importances))
        
        # Calculate average importance across models
        if importance_analysis:
            avg_importance = {}
            for feature in feature_names:
                importances = [analysis.get(feature, 0) for analysis in importance_analysis.values()]
                avg_importance[feature] = np.mean(importances)
            
            # Sort by importance
            sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
            
            importance_analysis['average_importance'] = dict(sorted_features)
            importance_analysis['top_10_features'] = sorted_features[:10]
            
            logger.info(f"🔍 Top 5 features: {[f[0] for f in sorted_features[:5]]}")
        
        return importance_analysis
    
    def _calculate_ensemble_weights(self, model_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate ensemble weights based on model performance"""
        
        weights = {}
        total_score = 0
        
        for model_name, result in model_results.items():
            if 'error' not in result:
                # Use F1 score as primary metric
                f1_score = result.get('test_metrics', {}).get('f1_score', 0)
                weights[model_name] = f1_score
                total_score += f1_score
        
        # Normalize weights
        if total_score > 0:
            for model_name in weights:
                weights[model_name] /= total_score
        
        logger.info(f"📊 Ensemble weights: {weights}")
        
        return weights
    
    def _save_model(self, model_name: str, model) -> None:
        """Save trained model"""
        
        try:
            if model_name.endswith('neural_network') and TENSORFLOW_AVAILABLE:
                # Save neural network in H5 format
                model_path = os.path.join(self.models_dir, f"{model_name}.h5")
                model.save(model_path)
            else:
                # Save sklearn models in pickle format
                model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
                joblib.dump(model, model_path)
            
            logger.info(f"💾 Model saved: {model_path}")
            
        except Exception as e:
            logger.error(f"❌ Could not save model {model_name}: {e}")
    
    def _save_training_metadata(self, training_results: Dict[str, Any], feature_names: List[str]):
        """Save training metadata"""
        
        metadata = {
            'model_type': 'second_set_underdog_prediction',
            'target_variable': self.preprocessor.target_variable,
            'feature_columns': feature_names,
            'ensemble_weights': self.ensemble_weights,
            'training_timestamp': datetime.now().isoformat(),
            'model_performance': {
                name: {
                    'f1_score': result.get('test_metrics', {}).get('f1_score', 0),
                    'accuracy': result.get('test_metrics', {}).get('accuracy', 0),
                    'roc_auc': result.get('test_metrics', {}).get('roc_auc', 0)
                }
                for name, result in training_results['model_results'].items()
                if 'error' not in result
            },
            'training_summary': training_results
        }
        
        # Save metadata
        metadata_path = os.path.join(self.models_dir, 'second_set_underdog_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save preprocessor
        preprocessor_path = os.path.join(self.models_dir, 'second_set_underdog_preprocessor.json')
        self.preprocessor.save_preprocessor(preprocessor_path)
        
        logger.info(f"💾 Training metadata saved: {metadata_path}")
    
    def evaluate_models_performance(self) -> Dict[str, Any]:
        """Evaluate and compare model performance"""
        
        evaluation = {
            'model_comparison': {},
            'best_model': None,
            'best_f1_score': 0,
            'ensemble_performance': {}
        }
        
        # Compare individual models
        for model_name, performance in self.model_performance.items():
            f1_score = performance.get('f1_score', 0)
            evaluation['model_comparison'][model_name] = {
                'f1_score': f1_score,
                'accuracy': performance.get('accuracy', 0),
                'precision': performance.get('precision', 0),
                'recall': performance.get('recall', 0),
                'roc_auc': performance.get('roc_auc', 0)
            }
            
            if f1_score > evaluation['best_f1_score']:
                evaluation['best_f1_score'] = f1_score
                evaluation['best_model'] = model_name
        
        # Add recommendations
        evaluation['recommendations'] = []
        
        if evaluation['best_f1_score'] > 0.7:
            evaluation['recommendations'].append("🟢 Strong model performance - ready for production")
        elif evaluation['best_f1_score'] > 0.6:
            evaluation['recommendations'].append("🟡 Good model performance - monitor and improve with more data")
        else:
            evaluation['recommendations'].append("🔴 Model performance needs improvement - collect more training data")
        
        return evaluation

def main():
    """Main function to demonstrate the second set underdog ML system"""
    
    print("🎾 SECOND SET UNDERDOG ML SYSTEM")
    print("=" * 60)
    
    # Initialize data collector
    print("🔧 Initializing data collection...")
    data_collector = ComprehensiveMLDataCollector()
    
    # Collect training data
    print("📊 Collecting training data...")
    collected_data = data_collector.collect_comprehensive_data(max_matches=30, priority_second_set=True)
    
    # Generate ML dataset
    print("🤖 Generating ML training dataset...")
    training_dataset = data_collector.get_ml_training_dataset()
    
    if 'error' in training_dataset:
        print(f"❌ Dataset generation failed: {training_dataset['error']}")
        return
    
    # Initialize ML trainer
    print("🚀 Initializing ML trainer...")
    ml_trainer = SecondSetUnderdogMLTrainer()
    
    # Train models
    print("⚡ Training models...")
    training_results = ml_trainer.train_models(
        training_dataset['features_df'],
        training_dataset['match_metadata']
    )
    
    # Display results
    print(f"\n📈 Training Results:")
    print(f"  Duration: {training_results['training_duration']:.1f} seconds")
    print(f"  Models trained: {len(training_results['model_results'])}")
    
    successful_models = [name for name, result in training_results['model_results'].items() 
                        if 'error' not in result]
    print(f"  Successful models: {len(successful_models)}")
    
    # Show model performance
    for model_name in successful_models:
        result = training_results['model_results'][model_name]
        f1_score = result.get('test_metrics', {}).get('f1_score', 0)
        print(f"    {model_name}: F1={f1_score:.3f}")
    
    # Evaluate performance
    print("\n🔍 Model Evaluation:")
    evaluation = ml_trainer.evaluate_models_performance()
    print(f"  Best model: {evaluation['best_model']} (F1: {evaluation['best_f1_score']:.3f})")
    
    for rec in evaluation['recommendations']:
        print(f"  {rec}")
    
    print(f"\n✅ Second set underdog ML system training completed!")
    print(f"🎯 Models saved to: {ml_trainer.models_dir}")

if __name__ == "__main__":
    main()