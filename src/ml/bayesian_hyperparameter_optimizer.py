#!/usr/bin/env python3
"""
Bayesian Hyperparameter Optimization for Tennis ML Models
=========================================================

Advanced hyperparameter optimization using Bayesian optimization for improved
model performance in tennis second set prediction.

Features:
- Bayesian optimization with Gaussian Process
- Tennis-specific objective functions
- Cross-validation with temporal awareness
- Model-specific parameter spaces
- Performance tracking and comparison

Author: Tennis ML Enhancement System
Date: 2025-08-30
"""

import numpy as np
import pandas as pd
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

# Bayesian optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt.acquisition import gaussian_ei
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    print("⚠️ scikit-optimize not available. Install with: pip install scikit-optimize")
    BAYESIAN_OPT_AVAILABLE = False

logger = logging.getLogger(__name__)

class TennisBayesianOptimizer:
    """Bayesian hyperparameter optimization for tennis prediction models"""
    
    def __init__(self, n_calls: int = 50, cv_folds: int = 5, random_state: int = 42):
        self.n_calls = n_calls
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.optimization_results = {}
        self.best_params = {}
        self.best_scores = {}
        
        # Tennis-specific scoring weights for multi-objective optimization
        self.scoring_weights = {
            'accuracy': 0.25,
            'precision': 0.35,  # Important for betting
            'recall': 0.20,
            'f1': 0.20
        }
    
    def define_parameter_spaces(self) -> Dict[str, List]:
        """Define parameter spaces for each model type"""
        
        parameter_spaces = {
            'random_forest': [
                Integer(50, 300, name='n_estimators'),
                Integer(3, 20, name='max_depth'),
                Integer(2, 10, name='min_samples_split'),
                Integer(1, 10, name='min_samples_leaf'),
                Real(0.1, 1.0, name='max_features'),
                Categorical(['gini', 'entropy'], name='criterion'),
                Categorical([True, False], name='bootstrap')
            ],
            
            'logistic_regression': [
                Real(1e-6, 1e2, prior='log-uniform', name='C'),
                Categorical(['l1', 'l2', 'elasticnet'], name='penalty'),
                Real(0.1, 0.9, name='l1_ratio'),  # For elasticnet
                Integer(100, 2000, name='max_iter'),
                Categorical(['liblinear', 'lbfgs', 'saga'], name='solver')
            ],
            
            'xgboost': [
                Integer(50, 300, name='n_estimators'),
                Integer(3, 10, name='max_depth'),
                Real(0.01, 0.3, name='learning_rate'),
                Real(0.5, 1.0, name='subsample'),
                Real(0.5, 1.0, name='colsample_bytree'),
                Real(1e-6, 10.0, prior='log-uniform', name='reg_alpha'),
                Real(1e-6, 10.0, prior='log-uniform', name='reg_lambda'),
                Integer(1, 10, name='min_child_weight')
            ],
            
            'lightgbm': [
                Integer(50, 300, name='n_estimators'),
                Integer(3, 15, name='max_depth'),
                Real(0.01, 0.3, name='learning_rate'),
                Real(0.5, 1.0, name='subsample'),
                Real(0.5, 1.0, name='colsample_bytree'),
                Real(1e-6, 10.0, prior='log-uniform', name='reg_alpha'),
                Real(1e-6, 10.0, prior='log-uniform', name='reg_lambda'),
                Integer(5, 100, name='num_leaves'),
                Integer(1, 20, name='min_child_samples')
            ]
        }
        
        return parameter_spaces
    
    def create_objective_function(self, model_type: str, X: np.ndarray, y: np.ndarray) -> Callable:
        """Create tennis-specific objective function for optimization"""
        
        parameter_space = self.define_parameter_spaces()[model_type]
        
        @use_named_args(parameter_space)
        def objective(**params):
            try:
                # Create model with current parameters
                model = self._create_model(model_type, params)
                
                # Use TimeSeriesSplit for temporal data
                cv = TimeSeriesSplit(n_splits=self.cv_folds)
                
                # Calculate multiple metrics
                scores = {}
                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
                    scores[metric] = np.mean(cv_scores)
                
                # Tennis-specific composite score (higher is better)
                composite_score = sum(scores[metric] * self.scoring_weights[metric] 
                                    for metric in scores.keys())
                
                # Return negative score for minimization
                return -composite_score
                
            except Exception as e:
                logger.warning(f"Error in objective function: {e}")
                return 1.0  # High value for failed optimization
        
        return objective
    
    def _create_model(self, model_type: str, params: Dict[str, Any]):
        """Create model instance with given parameters"""
        
        if model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', None),
                min_samples_split=params.get('min_samples_split', 2),
                min_samples_leaf=params.get('min_samples_leaf', 1),
                max_features=params.get('max_features', 'sqrt'),
                criterion=params.get('criterion', 'gini'),
                bootstrap=params.get('bootstrap', True),
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'
            )
        
        elif model_type == 'logistic_regression':
            # Handle solver compatibility
            penalty = params.get('penalty', 'l2')
            solver = params.get('solver', 'lbfgs')
            
            # Ensure solver-penalty compatibility
            if penalty == 'l1' and solver not in ['liblinear', 'saga']:
                solver = 'liblinear'
            elif penalty == 'elasticnet' and solver != 'saga':
                solver = 'saga'
            
            model_params = {
                'C': params.get('C', 1.0),
                'penalty': penalty,
                'max_iter': params.get('max_iter', 1000),
                'solver': solver,
                'random_state': self.random_state,
                'class_weight': 'balanced'
            }
            
            if penalty == 'elasticnet':
                model_params['l1_ratio'] = params.get('l1_ratio', 0.5)
            
            return LogisticRegression(**model_params)
        
        elif model_type == 'xgboost':
            return xgb.XGBClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.1),
                subsample=params.get('subsample', 1.0),
                colsample_bytree=params.get('colsample_bytree', 1.0),
                reg_alpha=params.get('reg_alpha', 0),
                reg_lambda=params.get('reg_lambda', 1),
                min_child_weight=params.get('min_child_weight', 1),
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='logloss'
            )
        
        elif model_type == 'lightgbm':
            return lgb.LGBMClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', -1),
                learning_rate=params.get('learning_rate', 0.1),
                subsample=params.get('subsample', 1.0),
                colsample_bytree=params.get('colsample_bytree', 1.0),
                reg_alpha=params.get('reg_alpha', 0),
                reg_lambda=params.get('reg_lambda', 0),
                num_leaves=params.get('num_leaves', 31),
                min_child_samples=params.get('min_child_samples', 20),
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced',
                verbose=-1
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def optimize_model(self, model_type: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize hyperparameters for a specific model"""
        
        if not BAYESIAN_OPT_AVAILABLE:
            logger.error("Bayesian optimization not available. Using default parameters.")
            return self._get_default_params(model_type)
        
        logger.info(f"Starting Bayesian optimization for {model_type}...")
        logger.info(f"Search space: {len(self.define_parameter_spaces()[model_type])} dimensions")
        logger.info(f"Optimization calls: {self.n_calls}")
        
        # Get parameter space and objective function
        parameter_space = self.define_parameter_spaces()[model_type]
        objective = self.create_objective_function(model_type, X, y)
        
        # Run Bayesian optimization
        start_time = datetime.now()
        
        try:
            result = gp_minimize(
                func=objective,
                dimensions=parameter_space,
                n_calls=self.n_calls,
                random_state=self.random_state,
                acq_func='EI',  # Expected Improvement
                n_initial_points=10,
                verbose=False
            )
            
            optimization_time = (datetime.now() - start_time).total_seconds()
            
            # Extract best parameters
            best_params = {}
            for i, param in enumerate(parameter_space):
                best_params[param.name] = result.x[i]
            
            # Evaluate best model
            best_model = self._create_model(model_type, best_params)
            cv = TimeSeriesSplit(n_splits=self.cv_folds)
            
            best_scores = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring=metric)
                best_scores[metric] = {
                    'mean': np.mean(cv_scores),
                    'std': np.std(cv_scores),
                    'scores': cv_scores.tolist()
                }
            
            # Store results
            optimization_result = {
                'model_type': model_type,
                'best_params': best_params,
                'best_scores': best_scores,
                'optimization_score': -result.fun,
                'n_calls': self.n_calls,
                'optimization_time_seconds': optimization_time,
                'convergence_history': result.func_vals.tolist(),
                'timestamp': datetime.now().isoformat()
            }
            
            self.optimization_results[model_type] = optimization_result
            self.best_params[model_type] = best_params
            self.best_scores[model_type] = best_scores
            
            logger.info(f"✅ {model_type} optimization completed in {optimization_time:.1f}s")
            logger.info(f"   Best score: {-result.fun:.4f}")
            logger.info(f"   Best params: {best_params}")
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"❌ Optimization failed for {model_type}: {e}")
            return self._get_default_params(model_type)
    
    def _get_default_params(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters if optimization fails"""
        
        default_params = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'criterion': 'gini',
                'bootstrap': True
            },
            'logistic_regression': {
                'C': 1.0,
                'penalty': 'l2',
                'max_iter': 1000,
                'solver': 'lbfgs'
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'min_child_weight': 1
            },
            'lightgbm': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'num_leaves': 31,
                'min_child_samples': 20
            }
        }
        
        return {
            'model_type': model_type,
            'best_params': default_params.get(model_type, {}),
            'optimization_score': 0.0,
            'is_default': True
        }
    
    def optimize_all_models(self, X: np.ndarray, y: np.ndarray, 
                           models: List[str] = None) -> Dict[str, Any]:
        """Optimize hyperparameters for all specified models"""
        
        if models is None:
            models = ['random_forest', 'logistic_regression', 'xgboost', 'lightgbm']
        
        logger.info("🚀 Starting comprehensive Bayesian optimization...")
        logger.info(f"   Models: {models}")
        logger.info(f"   Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"   Calls per model: {self.n_calls}")
        
        results = {}
        total_start_time = datetime.now()
        
        for model_type in models:
            logger.info(f"\n🔍 Optimizing {model_type}...")
            result = self.optimize_model(model_type, X, y)
            results[model_type] = result
        
        total_time = (datetime.now() - total_start_time).total_seconds()
        
        # Create summary
        summary = {
            'total_optimization_time_seconds': total_time,
            'models_optimized': len(models),
            'total_function_evaluations': len(models) * self.n_calls,
            'best_overall_model': self._find_best_model(),
            'optimization_timestamp': datetime.now().isoformat()
        }
        
        results['optimization_summary'] = summary
        
        logger.info(f"\n✅ Optimization completed in {total_time:.1f}s")
        logger.info(f"   Best model: {summary['best_overall_model']}")
        
        return results
    
    def _find_best_model(self) -> str:
        """Find the best performing model across all optimizations"""
        
        if not self.optimization_results:
            return "None"
        
        best_model = None
        best_score = -np.inf
        
        for model_type, result in self.optimization_results.items():
            score = result.get('optimization_score', 0)
            if score > best_score:
                best_score = score
                best_model = model_type
        
        return best_model or "None"
    
    def save_optimization_results(self, filepath: str) -> None:
        """Save optimization results to file"""
        
        try:
            results_data = {
                'optimization_results': self.optimization_results,
                'best_params': self.best_params,
                'best_scores': self.best_scores,
                'optimization_config': {
                    'n_calls': self.n_calls,
                    'cv_folds': self.cv_folds,
                    'random_state': self.random_state,
                    'scoring_weights': self.scoring_weights
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            logger.info(f"✅ Optimization results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
    
    def load_optimization_results(self, filepath: str) -> None:
        """Load optimization results from file"""
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.optimization_results = data.get('optimization_results', {})
            self.best_params = data.get('best_params', {})
            self.best_scores = data.get('best_scores', {})
            
            logger.info(f"✅ Optimization results loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load results: {e}")
    
    def generate_optimization_report(self) -> str:
        """Generate a comprehensive optimization report"""
        
        if not self.optimization_results:
            return "No optimization results available."
        
        report_lines = [
            "# Bayesian Hyperparameter Optimization Report",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Models Optimized:** {len(self.optimization_results)}",
            "",
            "## Optimization Summary",
            ""
        ]
        
        # Summary table
        report_lines.append("| Model | Best Score | Accuracy | Precision | Recall | F1 |")
        report_lines.append("|-------|------------|----------|-----------|--------|------|")
        
        for model_type, result in self.optimization_results.items():
            scores = result.get('best_scores', {})
            acc = scores.get('accuracy', {}).get('mean', 0)
            prec = scores.get('precision', {}).get('mean', 0)
            rec = scores.get('recall', {}).get('mean', 0)
            f1 = scores.get('f1', {}).get('mean', 0)
            opt_score = result.get('optimization_score', 0)
            
            report_lines.append(
                f"| {model_type} | {opt_score:.4f} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} |"
            )
        
        report_lines.extend(["", "## Best Parameters", ""])
        
        # Best parameters for each model
        for model_type, params in self.best_params.items():
            report_lines.append(f"### {model_type}")
            report_lines.append("```json")
            report_lines.append(json.dumps(params, indent=2))
            report_lines.append("```")
            report_lines.append("")
        
        return "\n".join(report_lines)

# Example usage and testing
if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = (X[:, 0] + X[:, 1] + np.random.randn(1000) * 0.5 > 0).astype(int)
    
    # Initialize optimizer
    optimizer = TennisBayesianOptimizer(n_calls=20, cv_folds=3)  # Reduced for testing
    
    # Optimize single model
    print("Testing single model optimization...")
    result = optimizer.optimize_model('random_forest', X, y)
    print(f"Best parameters: {result['best_params']}")
    
    # Save results
    optimizer.save_optimization_results('test_optimization_results.json')
    
    print("\n" + optimizer.generate_optimization_report())