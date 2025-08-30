#!/usr/bin/env python3
"""
Enhanced Tennis ML Pipeline Integration
======================================

Main pipeline that integrates all enhanced ML components for improved 
tennis second set prediction accuracy.

Features:
- Enhanced feature engineering
- Bayesian hyperparameter optimization  
- Dynamic ensemble with contextual weighting
- Real-time prediction capabilities
- Performance comparison and validation

Author: Tennis ML Enhancement System
Date: 2025-08-30
"""

import numpy as np
import pandas as pd
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

# Enhanced ML components
from src.ml.enhanced_feature_engineering import (
    EnhancedTennisFeatureEngineer, MatchContext, FirstSetStats
)
from src.ml.bayesian_hyperparameter_optimizer import TennisBayesianOptimizer
from src.ml.dynamic_ensemble import DynamicTennisEnsemble, ContextualWeightCalculator

logger = logging.getLogger(__name__)

class EnhancedTennisMLPipeline:
    """Enhanced ML pipeline for tennis second set prediction"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        
        # Core components
        self.feature_engineer = EnhancedTennisFeatureEngineer()
        self.optimizer = TennisBayesianOptimizer(
            n_calls=self.config['optimization_calls'],
            cv_folds=self.config['cv_folds']
        )
        self.ensemble = None
        
        # Performance tracking
        self.performance_history = []
        self.feature_importance_history = []
        
        logger.info("✅ Enhanced Tennis ML Pipeline initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default pipeline configuration"""
        return {
            'optimization_calls': 50,
            'cv_folds': 5,
            'test_size': 0.2,
            'random_state': 42,
            'models_to_optimize': ['random_forest', 'xgboost', 'lightgbm', 'logistic_regression'],
            'ensemble_weighting': 'dynamic',  # 'dynamic' or 'equal'
            'save_models': True,
            'models_directory': 'enhanced_tennis_models',
            'feature_selection': True,
            'min_feature_importance': 0.001
        }
    
    def prepare_enhanced_features(self, match_data_list: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, List]:
        """Prepare enhanced features from match data"""
        
        logger.info(f"🔧 Preparing enhanced features for {len(match_data_list)} matches...")
        
        features_list = []
        targets = []
        contexts = []
        
        for match_data in match_data_list:
            try:
                # Generate enhanced features
                enhanced_features = self.feature_engineer.create_all_enhanced_features(match_data)
                
                if not enhanced_features:
                    continue
                
                # Clean and validate features
                cleaned_features = self.feature_engineer.validate_features(enhanced_features)
                features_list.append(list(cleaned_features.values()))
                
                # Extract target (underdog wins second set)
                underdog_player = 2 if match_data['player1']['ranking'] < match_data['player2']['ranking'] else 1
                second_set_winner = match_data.get('second_set_winner')
                
                if second_set_winner is not None:
                    target = 1 if second_set_winner == underdog_player else 0
                    targets.append(target)
                    
                    # Create match context for dynamic ensemble
                    context = MatchContext(
                        surface=match_data['match_context'].surface,
                        tournament_tier=match_data['match_context'].tournament_tier,
                        round=match_data['match_context'].round,
                        is_indoor=match_data['match_context'].is_indoor,
                        player1_ranking=match_data['player1']['ranking'],
                        player2_ranking=match_data['player2']['ranking'],
                        ranking_gap=abs(match_data['player1']['ranking'] - match_data['player2']['ranking']),
                        is_upset_scenario=abs(match_data['player1']['ranking'] - match_data['player2']['ranking']) > 50,
                        surface_specialization={
                            'player1': match_data['player1'].get('surface_win_percentage', 0.5),
                            'player2': match_data['player2'].get('surface_win_percentage', 0.5)
                        },
                        h2h_history=match_data.get('h2h_data', {}).get('total_matches', 0),
                        tournament_importance=0.8  # Default importance
                    )
                    contexts.append(context)
            
            except Exception as e:
                logger.warning(f"Error processing match data: {e}")
                continue
        
        if not features_list:
            raise ValueError("No valid features generated from match data")
        
        X = np.array(features_list)
        y = np.array(targets)
        
        logger.info(f"✅ Generated {X.shape[1]} enhanced features for {X.shape[0]} matches")
        logger.info(f"   Target distribution: {np.mean(y):.3f} (underdog win rate)")
        
        return X, y, contexts
    
    def optimize_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize hyperparameters for all models"""
        
        logger.info("🚀 Starting Bayesian hyperparameter optimization...")
        
        optimization_results = self.optimizer.optimize_all_models(
            X, y, self.config['models_to_optimize']
        )
        
        logger.info("✅ Hyperparameter optimization completed")
        
        return optimization_results
    
    def create_optimized_models(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create models with optimized hyperparameters"""
        
        models = {}
        
        for model_type in self.config['models_to_optimize']:
            result = optimization_results.get(model_type, {})
            best_params = result.get('best_params', {})
            
            if model_type == 'random_forest':
                models[model_type] = RandomForestClassifier(
                    **best_params,
                    random_state=self.config['random_state'],
                    n_jobs=-1,
                    class_weight='balanced'
                )
            elif model_type == 'xgboost':
                models[model_type] = xgb.XGBClassifier(
                    **best_params,
                    random_state=self.config['random_state'],
                    n_jobs=-1,
                    eval_metric='logloss'
                )
            elif model_type == 'lightgbm':
                models[model_type] = lgb.LGBMClassifier(
                    **best_params,
                    random_state=self.config['random_state'],
                    n_jobs=-1,
                    class_weight='balanced',
                    verbose=-1
                )
            elif model_type == 'logistic_regression':
                models[model_type] = LogisticRegression(
                    **best_params,
                    random_state=self.config['random_state'],
                    class_weight='balanced'
                )
        
        logger.info(f"✅ Created {len(models)} optimized models")
        
        return models
    
    def train_enhanced_pipeline(self, match_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train the complete enhanced pipeline"""
        
        logger.info("🎾 Training Enhanced Tennis ML Pipeline...")
        start_time = datetime.now()
        
        # Step 1: Prepare enhanced features
        X, y, contexts = self.prepare_enhanced_features(match_data_list)
        
        # Step 2: Train-test split
        X_train, X_test, y_train, y_test, contexts_train, contexts_test = train_test_split(
            X, y, contexts, test_size=self.config['test_size'], 
            random_state=self.config['random_state'], stratify=y
        )
        
        # Step 3: Optimize hyperparameters
        optimization_results = self.optimize_models(X_train, y_train)
        
        # Step 4: Create optimized models
        optimized_models = self.create_optimized_models(optimization_results)
        
        # Step 5: Create and train dynamic ensemble
        weight_calculator = ContextualWeightCalculator()
        self.ensemble = DynamicTennisEnsemble(optimized_models, weight_calculator)
        self.ensemble.fit(X_train, y_train, contexts_train)
        
        # Step 6: Evaluate performance
        train_performance = self._evaluate_performance(X_train, y_train, contexts_train, "Training")
        test_performance = self._evaluate_performance(X_test, y_test, contexts_test, "Test")
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Create training summary
        training_summary = {
            'training_time_seconds': training_time,
            'dataset_size': len(match_data_list),
            'features_generated': X.shape[1],
            'optimization_results': optimization_results,
            'train_performance': train_performance,
            'test_performance': test_performance,
            'models_trained': list(optimized_models.keys()),
            'ensemble_type': 'dynamic_contextual',
            'timestamp': datetime.now().isoformat()
        }
        
        # Store performance history
        self.performance_history.append(training_summary)
        
        logger.info(f"✅ Enhanced pipeline training completed in {training_time:.1f}s")
        logger.info(f"   Test Accuracy: {test_performance['accuracy']:.4f}")
        logger.info(f"   Test Precision: {test_performance['precision']:.4f}")
        logger.info(f"   Test F1-Score: {test_performance['f1_score']:.4f}")
        
        return training_summary
    
    def _evaluate_performance(self, X: np.ndarray, y: np.ndarray, 
                            contexts: List, dataset_name: str) -> Dict[str, float]:
        """Evaluate ensemble performance"""
        
        predictions = self.ensemble.predict(X, contexts)
        probabilities = self.ensemble.predict_proba(X, contexts)
        
        performance = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions),
            'recall': recall_score(y, predictions),
            'f1_score': f1_score(y, predictions),
            'roc_auc': roc_auc_score(y, probabilities[:, 1])
        }
        
        logger.info(f"📊 {dataset_name} Performance:")
        for metric, value in performance.items():
            logger.info(f"   {metric.capitalize()}: {value:.4f}")
        
        return performance
    
    def predict_match(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make enhanced prediction for a single match"""
        
        if self.ensemble is None:
            raise ValueError("Pipeline must be trained before making predictions")
        
        # Generate enhanced features
        enhanced_features = self.feature_engineer.create_all_enhanced_features(match_data)
        cleaned_features = self.feature_engineer.validate_features(enhanced_features)
        
        # Prepare feature vector
        feature_vector = np.array(list(cleaned_features.values())).reshape(1, -1)
        
        # Create match context
        context = MatchContext(
            surface=match_data['match_context'].surface,
            tournament_tier=match_data['match_context'].tournament_tier,
            round=match_data['match_context'].round,
            is_indoor=match_data['match_context'].is_indoor,
            player1_ranking=match_data['player1']['ranking'],
            player2_ranking=match_data['player2']['ranking'],
            ranking_gap=abs(match_data['player1']['ranking'] - match_data['player2']['ranking']),
            is_upset_scenario=abs(match_data['player1']['ranking'] - match_data['player2']['ranking']) > 50,
            surface_specialization={
                'player1': match_data['player1'].get('surface_win_percentage', 0.5),
                'player2': match_data['player2'].get('surface_win_percentage', 0.5)
            },
            h2h_history=match_data.get('h2h_data', {}).get('total_matches', 0),
            tournament_importance=0.8
        )
        
        # Make prediction with explanation
        explanations = self.ensemble.predict_with_explanation(feature_vector, [context])
        explanation = explanations[0]
        
        # Determine underdog
        underdog_player = 2 if match_data['player1']['ranking'] < match_data['player2']['ranking'] else 1
        underdog_name = match_data['player2']['name'] if underdog_player == 2 else match_data['player1']['name']
        
        # Create prediction result
        prediction_result = {
            'underdog_player': underdog_player,
            'underdog_name': underdog_name,
            'underdog_win_probability': explanation['final_probability'],
            'favorite_win_probability': 1.0 - explanation['final_probability'],
            'confidence_level': self._get_confidence_level(explanation['final_probability']),
            'model_contributions': explanation['model_contributions'],
            'context_factors': explanation['weight_reasoning'],
            'features_used': len(cleaned_features),
            'prediction_time': datetime.now().isoformat(),
            'model_type': 'enhanced_dynamic_ensemble'
        }
        
        return prediction_result
    
    def _get_confidence_level(self, probability: float) -> str:
        """Get confidence level description"""
        if probability < 0.4 or probability > 0.6:
            return "High"
        elif probability < 0.45 or probability > 0.55:
            return "Medium"
        else:
            return "Low"
    
    def save_pipeline(self, filepath: str) -> None:
        """Save the enhanced pipeline"""
        
        if self.ensemble is None:
            raise ValueError("Pipeline must be trained before saving")
        
        # Save ensemble
        self.ensemble.save_ensemble(f"{filepath}_enhanced_ensemble")
        
        # Save pipeline configuration and history
        pipeline_data = {
            'config': self.config,
            'performance_history': self.performance_history,
            'feature_importance_history': self.feature_importance_history,
            'pipeline_version': '1.0.0'
        }
        
        with open(f"{filepath}_pipeline_config.json", 'w') as f:
            json.dump(pipeline_data, f, indent=2, default=str)
        
        logger.info(f"✅ Enhanced pipeline saved to {filepath}")

# Example usage
if __name__ == "__main__":
    # This would use real match data
    print("Enhanced Tennis ML Pipeline - Ready for Integration")
    print("Please provide match data to train and test the enhanced system.")