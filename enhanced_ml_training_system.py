#!/usr/bin/env python3
"""
🚀 ENHANCED ML TRAINING SYSTEM FOR TENNIS PREDICTION
Strategic ML improvements with phased implementation approach

Phase 1: Immediate improvements with current data
Phase 2: Advanced techniques after data accumulation 
Phase 3: Mature system optimizations

Author: Claude Code (Anthropic)
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
import logging

# Scikit-learn imports
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, GridSearchCV, 
    RandomizedSearchCV, train_test_split, validation_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, RFE, SelectFromModel,
    mutual_info_classif, VarianceThreshold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# Advanced ML imports
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from tensorflow import keras
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataQualityAnalyzer:
    """Analyze API data quality and accumulation patterns"""
    
    def __init__(self, api_usage_file: str = "api_usage.json", cache_file: str = "api_cache.json"):
        self.api_usage_file = api_usage_file
        self.cache_file = cache_file
        
    def analyze_current_data_quality(self) -> Dict[str, Any]:
        """Analyze current data quality and volume"""
        analysis = {
            'api_usage_analysis': self._analyze_api_usage(),
            'cache_analysis': self._analyze_cache_data(),
            'data_quality_score': 0,
            'recommendations': []
        }
        
        # Calculate overall data quality score
        api_score = min(analysis['api_usage_analysis']['requests_today'] / 8, 1.0) * 30
        cache_score = min(analysis['cache_analysis']['total_matches'] / 100, 1.0) * 40
        diversity_score = analysis['cache_analysis']['source_diversity'] * 30
        
        analysis['data_quality_score'] = api_score + cache_score + diversity_score
        
        # Generate recommendations
        if analysis['data_quality_score'] < 50:
            analysis['recommendations'].append("❌ LOW DATA QUALITY: Focus on Phase 1 enhancements only")
        elif analysis['data_quality_score'] < 75:
            analysis['recommendations'].append("🟡 MODERATE DATA QUALITY: Phase 1 + basic Phase 2")
        else:
            analysis['recommendations'].append("✅ HIGH DATA QUALITY: Full enhancement roadmap applicable")
            
        return analysis
    
    def _analyze_api_usage(self) -> Dict[str, Any]:
        """Analyze API usage patterns"""
        try:
            if os.path.exists(self.api_usage_file):
                with open(self.api_usage_file, 'r') as f:
                    usage_data = json.load(f)
                
                hourly_requests = usage_data.get('hourly_requests', [])
                total_requests = usage_data.get('total_requests', 0)
                
                # Calculate requests in last 24 hours
                from datetime import datetime, timedelta
                now = datetime.now()
                recent_requests = []
                
                for req_time_str in hourly_requests:
                    try:
                        req_time = datetime.fromisoformat(req_time_str.replace('Z', '+00:00'))
                        if (now - req_time).total_seconds() < 86400:  # 24 hours
                            recent_requests.append(req_time_str)
                    except:
                        continue
                
                return {
                    'total_requests': total_requests,
                    'requests_today': len(recent_requests),
                    'api_available': True,
                    'last_request': hourly_requests[-1] if hourly_requests else None
                }
        except Exception as e:
            logger.warning(f"Could not analyze API usage: {e}")
        
        return {
            'total_requests': 0,
            'requests_today': 0,
            'api_available': False,
            'last_request': None
        }
    
    def _analyze_cache_data(self) -> Dict[str, Any]:
        """Analyze cached data quality"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                total_matches = len(cache_data.get('matches', []))
                sources = set()
                quality_scores = []
                
                for match in cache_data.get('matches', []):
                    if isinstance(match, dict):
                        sources.add(match.get('data_source', 'unknown'))
                        quality_scores.append(match.get('quality_score', 50))
                
                return {
                    'total_matches': total_matches,
                    'unique_sources': len(sources),
                    'source_diversity': min(len(sources) / 4, 1.0),  # Max 4 expected sources
                    'avg_quality_score': np.mean(quality_scores) if quality_scores else 50,
                    'cache_available': True
                }
        except Exception as e:
            logger.warning(f"Could not analyze cache data: {e}")
        
        return {
            'total_matches': 0,
            'unique_sources': 0,
            'source_diversity': 0,
            'avg_quality_score': 50,
            'cache_available': False
        }

class EnhancedCrossValidation:
    """Enhanced cross-validation with tennis-specific considerations"""
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        
    def perform_cross_validation(self, model, X: np.ndarray, y: np.ndarray, 
                                 scoring: List[str] = None) -> Dict[str, Any]:
        """Perform comprehensive cross-validation"""
        if scoring is None:
            scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        # Use StratifiedKFold to maintain class distribution
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        results = {}
        for metric in scoring:
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
                results[metric] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'scores': scores.tolist(),
                    'confidence_interval': [
                        scores.mean() - 1.96 * scores.std() / np.sqrt(len(scores)),
                        scores.mean() + 1.96 * scores.std() / np.sqrt(len(scores))
                    ]
                }
            except Exception as e:
                logger.warning(f"Could not calculate {metric}: {e}")
                results[metric] = {'mean': 0.0, 'std': 0.0, 'scores': [], 'error': str(e)}
        
        return results
    
    def compare_models(self, models: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Compare multiple models using cross-validation"""
        comparison_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Cross-validating {model_name}...")
            try:
                cv_results = self.perform_cross_validation(model, X, y)
                comparison_results[model_name] = cv_results
            except Exception as e:
                logger.error(f"Error in cross-validation for {model_name}: {e}")
                comparison_results[model_name] = {'error': str(e)}
        
        return comparison_results

class AdvancedMetricsEvaluator:
    """Advanced evaluation metrics for tennis prediction models"""
    
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                        y_pred_proba: np.ndarray = None) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics"""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary')
        metrics['recall'] = recall_score(y_true, y_pred, average='binary')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='binary')
        
        # Advanced metrics (if probabilities available)
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                
                # Calculate precision-recall curve metrics
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
                metrics['avg_precision'] = np.mean(precision_curve)
                
                # ROC curve data
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                metrics['roc_curve_data'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist()
                }
                
            except Exception as e:
                logger.warning(f"Could not calculate advanced metrics: {e}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = {
            'matrix': cm.tolist(),
            'true_negatives': int(cm[0, 0]),
            'false_positives': int(cm[0, 1]), 
            'false_negatives': int(cm[1, 0]),
            'true_positives': int(cm[1, 1])
        }
        
        # Classification report
        try:
            classification_rep = classification_report(y_true, y_pred, output_dict=True)
            metrics['classification_report'] = classification_rep
        except Exception as e:
            logger.warning(f"Could not generate classification report: {e}")
        
        # Tennis-specific metrics
        metrics['tennis_specific'] = self._calculate_tennis_metrics(y_true, y_pred, y_pred_proba)
        
        return metrics
    
    def _calculate_tennis_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_pred_proba: np.ndarray = None) -> Dict[str, Any]:
        """Calculate tennis-specific evaluation metrics"""
        tennis_metrics = {}
        
        # Underdog prediction accuracy
        underdog_mask = y_true == 1  # Assuming 1 = underdog wins
        if np.sum(underdog_mask) > 0:
            tennis_metrics['underdog_prediction_accuracy'] = accuracy_score(
                y_true[underdog_mask], y_pred[underdog_mask]
            )
        
        # Favorite prediction accuracy
        favorite_mask = y_true == 0  # Assuming 0 = favorite wins
        if np.sum(favorite_mask) > 0:
            tennis_metrics['favorite_prediction_accuracy'] = accuracy_score(
                y_true[favorite_mask], y_pred[favorite_mask]
            )
        
        # Confidence calibration (if probabilities available)
        if y_pred_proba is not None:
            # Calculate calibration bins
            bin_boundaries = np.linspace(0, 1, 11)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            accuracies = []
            confidences = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                    
                    accuracies.append(accuracy_in_bin)
                    confidences.append(avg_confidence_in_bin)
            
            if accuracies:
                tennis_metrics['calibration_error'] = np.mean(np.abs(
                    np.array(accuracies) - np.array(confidences)
                ))
        
        return tennis_metrics

class BasicHyperparameterTuner:
    """Basic hyperparameter tuning for Phase 1 implementation"""
    
    def __init__(self, cv_folds: int = 3, n_iter: int = 50, random_state: int = 42):
        self.cv_folds = cv_folds
        self.n_iter = n_iter
        self.random_state = random_state
        
    def tune_random_forest(self, X: np.ndarray, y: np.ndarray) -> Tuple[RandomForestClassifier, Dict]:
        """Tune Random Forest hyperparameters"""
        logger.info("Tuning Random Forest hyperparameters...")
        
        param_distributions = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.8]
        }
        
        rf = RandomForestClassifier(random_state=self.random_state)
        
        # Use RandomizedSearchCV for efficiency with limited data
        random_search = RandomizedSearchCV(
            rf, param_distributions,
            n_iter=self.n_iter,
            cv=self.cv_folds,
            scoring='f1',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1
        )
        
        random_search.fit(X, y)
        
        return random_search.best_estimator_, {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'cv_results': random_search.cv_results_
        }
    
    def tune_gradient_boosting(self, X: np.ndarray, y: np.ndarray) -> Tuple[GradientBoostingClassifier, Dict]:
        """Tune Gradient Boosting hyperparameters"""
        logger.info("Tuning Gradient Boosting hyperparameters...")
        
        param_distributions = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        gb = GradientBoostingClassifier(random_state=self.random_state)
        
        random_search = RandomizedSearchCV(
            gb, param_distributions,
            n_iter=self.n_iter,
            cv=self.cv_folds,
            scoring='f1',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1
        )
        
        random_search.fit(X, y)
        
        return random_search.best_estimator_, {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'cv_results': random_search.cv_results_
        }
    
    def tune_logistic_regression(self, X: np.ndarray, y: np.ndarray) -> Tuple[LogisticRegression, Dict]:
        """Tune Logistic Regression hyperparameters"""
        logger.info("Tuning Logistic Regression hyperparameters...")
        
        param_distributions = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [100, 200, 500, 1000]
        }
        
        lr = LogisticRegression(random_state=self.random_state)
        
        random_search = RandomizedSearchCV(
            lr, param_distributions,
            n_iter=self.n_iter,
            cv=self.cv_folds,
            scoring='f1',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1
        )
        
        random_search.fit(X, y)
        
        return random_search.best_estimator_, {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'cv_results': random_search.cv_results_
        }

class FeatureSelector:
    """Systematic feature selection methods"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
    def select_features_comprehensive(self, X: pd.DataFrame, y: np.ndarray, 
                                      feature_names: List[str]) -> Dict[str, Any]:
        """Comprehensive feature selection using multiple methods"""
        results = {}
        
        # Method 1: Variance Threshold
        results['variance_threshold'] = self._variance_threshold_selection(X, feature_names)
        
        # Method 2: Univariate Statistical Tests
        results['univariate_statistical'] = self._univariate_selection(X, y, feature_names)
        
        # Method 3: Recursive Feature Elimination
        results['rfe_selection'] = self._rfe_selection(X, y, feature_names)
        
        # Method 4: Model-based Selection
        results['model_based_selection'] = self._model_based_selection(X, y, feature_names)
        
        # Method 5: Mutual Information
        results['mutual_info_selection'] = self._mutual_info_selection(X, y, feature_names)
        
        # Combine results and provide recommendations
        results['recommendations'] = self._combine_feature_selections(results, feature_names)
        
        return results
    
    def _variance_threshold_selection(self, X: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
        """Remove features with low variance"""
        try:
            selector = VarianceThreshold(threshold=0.01)  # Remove very low variance features
            selector.fit(X)
            
            selected_features = np.array(feature_names)[selector.get_support()].tolist()
            removed_features = np.array(feature_names)[~selector.get_support()].tolist()
            
            return {
                'selected_features': selected_features,
                'removed_features': removed_features,
                'n_selected': len(selected_features),
                'selection_method': 'variance_threshold'
            }
        except Exception as e:
            logger.warning(f"Variance threshold selection failed: {e}")
            return {'error': str(e), 'selected_features': feature_names}
    
    def _univariate_selection(self, X: pd.DataFrame, y: np.ndarray, 
                              feature_names: List[str]) -> Dict[str, Any]:
        """Select features based on univariate statistical tests"""
        try:
            # Select top 50% of features
            k = max(1, len(feature_names) // 2)
            selector = SelectKBest(score_func=f_classif, k=k)
            selector.fit(X, y)
            
            selected_features = np.array(feature_names)[selector.get_support()].tolist()
            feature_scores = dict(zip(feature_names, selector.scores_))
            
            return {
                'selected_features': selected_features,
                'feature_scores': feature_scores,
                'n_selected': len(selected_features),
                'selection_method': 'univariate_statistical'
            }
        except Exception as e:
            logger.warning(f"Univariate selection failed: {e}")
            return {'error': str(e), 'selected_features': feature_names}
    
    def _rfe_selection(self, X: pd.DataFrame, y: np.ndarray, 
                       feature_names: List[str]) -> Dict[str, Any]:
        """Recursive Feature Elimination"""
        try:
            # Use Random Forest as base estimator
            estimator = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
            
            # Select top 70% of features
            n_features_to_select = max(1, int(len(feature_names) * 0.7))
            
            selector = RFE(estimator, n_features_to_select=n_features_to_select)
            selector.fit(X, y)
            
            selected_features = np.array(feature_names)[selector.get_support()].tolist()
            feature_rankings = dict(zip(feature_names, selector.ranking_))
            
            return {
                'selected_features': selected_features,
                'feature_rankings': feature_rankings,
                'n_selected': len(selected_features),
                'selection_method': 'rfe'
            }
        except Exception as e:
            logger.warning(f"RFE selection failed: {e}")
            return {'error': str(e), 'selected_features': feature_names}
    
    def _model_based_selection(self, X: pd.DataFrame, y: np.ndarray, 
                               feature_names: List[str]) -> Dict[str, Any]:
        """Model-based feature selection"""
        try:
            # Use Random Forest for feature importance
            estimator = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            selector = SelectFromModel(estimator, threshold='median')
            selector.fit(X, y)
            
            selected_features = np.array(feature_names)[selector.get_support()].tolist()
            
            # Get feature importances
            estimator.fit(X, y)
            feature_importances = dict(zip(feature_names, estimator.feature_importances_))
            
            return {
                'selected_features': selected_features,
                'feature_importances': feature_importances,
                'n_selected': len(selected_features),
                'selection_method': 'model_based'
            }
        except Exception as e:
            logger.warning(f"Model-based selection failed: {e}")
            return {'error': str(e), 'selected_features': feature_names}
    
    def _mutual_info_selection(self, X: pd.DataFrame, y: np.ndarray, 
                               feature_names: List[str]) -> Dict[str, Any]:
        """Mutual information based feature selection"""
        try:
            # Calculate mutual information scores
            mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
            feature_mi_scores = dict(zip(feature_names, mi_scores))
            
            # Select features with MI score above median
            threshold = np.median(mi_scores)
            selected_features = [name for name, score in feature_mi_scores.items() if score >= threshold]
            
            return {
                'selected_features': selected_features,
                'mutual_info_scores': feature_mi_scores,
                'n_selected': len(selected_features),
                'selection_method': 'mutual_information',
                'threshold': threshold
            }
        except Exception as e:
            logger.warning(f"Mutual information selection failed: {e}")
            return {'error': str(e), 'selected_features': feature_names}
    
    def _combine_feature_selections(self, results: Dict, feature_names: List[str]) -> Dict[str, Any]:
        """Combine results from different feature selection methods"""
        # Count how many times each feature was selected
        feature_votes = {name: 0 for name in feature_names}
        
        methods_used = 0
        for method_name, method_results in results.items():
            if 'selected_features' in method_results and 'error' not in method_results:
                methods_used += 1
                for feature in method_results['selected_features']:
                    feature_votes[feature] += 1
        
        if methods_used == 0:
            return {
                'recommended_features': feature_names,
                'feature_votes': feature_votes,
                'consensus_threshold': 0,
                'methods_used': 0
            }
        
        # Features selected by majority of methods
        consensus_threshold = max(1, methods_used // 2)
        recommended_features = [
            feature for feature, votes in feature_votes.items() 
            if votes >= consensus_threshold
        ]
        
        # If too few features selected, lower threshold
        if len(recommended_features) < 3:
            consensus_threshold = max(1, methods_used // 3)
            recommended_features = [
                feature for feature, votes in feature_votes.items() 
                if votes >= consensus_threshold
            ]
        
        return {
            'recommended_features': recommended_features,
            'feature_votes': feature_votes,
            'consensus_threshold': consensus_threshold,
            'methods_used': methods_used,
            'selection_strength': {
                feature: votes / methods_used for feature, votes in feature_votes.items()
            }
        }

class EnhancedMLTrainingSystem:
    """Main enhanced ML training system with phased approach"""
    
    def __init__(self, models_dir: str = "tennis_models", data_quality_threshold: int = 50):
        self.models_dir = models_dir
        self.data_quality_threshold = data_quality_threshold
        
        # Initialize components
        self.data_quality_analyzer = DataQualityAnalyzer()
        self.cross_validator = EnhancedCrossValidation()
        self.metrics_evaluator = AdvancedMetricsEvaluator()
        self.hyperparameter_tuner = BasicHyperparameterTuner()
        self.feature_selector = FeatureSelector()
        
        # Load existing metadata if available
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load existing model metadata"""
        metadata_path = os.path.join(self.models_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
        
        return {
            'feature_columns': [],
            'ensemble_weights': {},
            'validation_scores': {},
            'feature_importance': {}
        }
    
    def analyze_system_readiness(self) -> Dict[str, Any]:
        """Analyze system readiness for different enhancement phases"""
        logger.info("Analyzing system readiness for ML enhancements...")
        
        # Analyze data quality
        data_analysis = self.data_quality_analyzer.analyze_current_data_quality()
        
        # Determine which phases are recommended
        phases_ready = {
            'phase_1': True,  # Basic improvements always possible
            'phase_2': data_analysis['data_quality_score'] >= 60,
            'phase_3': data_analysis['data_quality_score'] >= 80
        }
        
        # Generate strategic recommendations
        recommendations = self._generate_strategic_recommendations(data_analysis, phases_ready)
        
        return {
            'data_quality_analysis': data_analysis,
            'phases_ready': phases_ready,
            'strategic_recommendations': recommendations,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _generate_strategic_recommendations(self, data_analysis: Dict, phases_ready: Dict) -> List[str]:
        """Generate strategic recommendations based on current system state"""
        recommendations = []
        
        quality_score = data_analysis['data_quality_score']
        
        if quality_score < 30:
            recommendations.extend([
                "🔴 CRITICAL: Very low data quality detected",
                "📊 Focus on data collection and API optimization first",
                "🎯 Implement only basic cross-validation and metrics",
                "⏳ Wait for more data before advanced techniques"
            ])
        elif quality_score < 60:
            recommendations.extend([
                "🟡 MODERATE: Decent data foundation available",
                "✅ Proceed with Phase 1 enhancements immediately",
                "📈 Begin collecting data for Phase 2 techniques",
                "🎯 Focus on hyperparameter tuning with current data"
            ])
        else:
            recommendations.extend([
                "🟢 GOOD: Strong data foundation for ML enhancements",
                "🚀 Full enhancement roadmap can be implemented",
                "📊 Advanced feature selection and regularization ready",
                "🎯 Consider Bayesian optimization and ensemble methods"
            ])
        
        # Add specific timing recommendations
        api_requests_today = data_analysis['api_usage_analysis']['requests_today']
        if api_requests_today < 3:
            recommendations.append("⚠️ TIMING: Low API activity today - consider scheduling enhancements during high-data periods")
        
        return recommendations
    
    def implement_phase_1_enhancements(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Implement Phase 1 enhancements: Cross-validation, metrics, basic hyperparameter tuning"""
        logger.info("🚀 Implementing Phase 1 ML Enhancements...")
        
        results = {
            'phase': 1,
            'implementation_timestamp': datetime.now().isoformat(),
            'enhancements_applied': []
        }
        
        # 1. Enhanced Cross-Validation
        logger.info("📊 Applying enhanced cross-validation...")
        try:
            # Load existing models for comparison
            existing_models = self._load_existing_models()
            
            if existing_models:
                cv_results = self.cross_validator.compare_models(existing_models, X.values, y)
                results['cross_validation_results'] = cv_results
                results['enhancements_applied'].append('enhanced_cross_validation')
                logger.info("✅ Enhanced cross-validation completed")
            else:
                logger.warning("⚠️ No existing models found for cross-validation")
                
        except Exception as e:
            logger.error(f"❌ Cross-validation failed: {e}")
            results['cross_validation_error'] = str(e)
        
        # 2. Advanced Metrics Evaluation
        logger.info("📈 Calculating advanced metrics...")
        try:
            # Use the best existing model for evaluation
            best_model = self._get_best_existing_model(existing_models)
            if best_model:
                # Split data for evaluation
                X_train, X_test, y_train, y_test = train_test_split(
                    X.values, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Train and evaluate
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)
                y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
                
                advanced_metrics = self.metrics_evaluator.calculate_comprehensive_metrics(
                    y_test, y_pred, y_pred_proba
                )
                results['advanced_metrics'] = advanced_metrics
                results['enhancements_applied'].append('advanced_metrics')
                logger.info("✅ Advanced metrics calculated")
                
        except Exception as e:
            logger.error(f"❌ Advanced metrics calculation failed: {e}")
            results['advanced_metrics_error'] = str(e)
        
        # 3. Basic Hyperparameter Tuning
        logger.info("🔧 Performing basic hyperparameter tuning...")
        try:
            tuning_results = {}
            
            # Tune Random Forest
            rf_tuned, rf_results = self.hyperparameter_tuner.tune_random_forest(X.values, y)
            tuning_results['random_forest'] = rf_results
            
            # Tune Gradient Boosting
            gb_tuned, gb_results = self.hyperparameter_tuner.tune_gradient_boosting(X.values, y)
            tuning_results['gradient_boosting'] = gb_results
            
            # Tune Logistic Regression
            lr_tuned, lr_results = self.hyperparameter_tuner.tune_logistic_regression(X.values, y)
            tuning_results['logistic_regression'] = lr_results
            
            results['hyperparameter_tuning'] = tuning_results
            results['enhancements_applied'].append('basic_hyperparameter_tuning')
            
            # Save tuned models
            tuned_models = {
                'random_forest_tuned': rf_tuned,
                'gradient_boosting_tuned': gb_tuned,
                'logistic_regression_tuned': lr_tuned
            }
            self._save_tuned_models(tuned_models)
            
            logger.info("✅ Basic hyperparameter tuning completed")
            
        except Exception as e:
            logger.error(f"❌ Hyperparameter tuning failed: {e}")
            results['hyperparameter_tuning_error'] = str(e)
        
        return results
    
    def implement_phase_2_enhancements(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Implement Phase 2 enhancements: Feature selection, advanced hyperparameter tuning"""
        logger.info("🚀 Implementing Phase 2 ML Enhancements...")
        
        results = {
            'phase': 2,
            'implementation_timestamp': datetime.now().isoformat(),
            'enhancements_applied': []
        }
        
        # 1. Comprehensive Feature Selection
        logger.info("🔍 Performing comprehensive feature selection...")
        try:
            feature_names = X.columns.tolist()
            feature_selection_results = self.feature_selector.select_features_comprehensive(
                X, y, feature_names
            )
            results['feature_selection'] = feature_selection_results
            results['enhancements_applied'].append('comprehensive_feature_selection')
            
            # Use recommended features for further training
            recommended_features = feature_selection_results['recommendations']['recommended_features']
            if len(recommended_features) > 3:  # Ensure minimum features
                X_selected = X[recommended_features]
                results['features_used'] = recommended_features
                logger.info(f"✅ Feature selection completed. Using {len(recommended_features)} features")
            else:
                X_selected = X  # Use all features if selection too aggressive
                results['features_used'] = feature_names
                logger.warning("⚠️ Feature selection too aggressive, using all features")
                
        except Exception as e:
            logger.error(f"❌ Feature selection failed: {e}")
            results['feature_selection_error'] = str(e)
            X_selected = X
        
        # 2. Advanced Hyperparameter Tuning (if Bayesian optimization available)
        if BAYESIAN_OPT_AVAILABLE:
            logger.info("🧠 Performing Bayesian optimization...")
            try:
                bayesian_results = self._perform_bayesian_optimization(X_selected, y)
                results['bayesian_optimization'] = bayesian_results
                results['enhancements_applied'].append('bayesian_optimization')
                logger.info("✅ Bayesian optimization completed")
            except Exception as e:
                logger.error(f"❌ Bayesian optimization failed: {e}")
                results['bayesian_optimization_error'] = str(e)
        else:
            logger.warning("⚠️ Bayesian optimization not available (scikit-optimize not installed)")
        
        return results
    
    def implement_phase_3_enhancements(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Implement Phase 3 enhancements: Advanced regularization, ensemble methods"""
        logger.info("🚀 Implementing Phase 3 ML Enhancements...")
        
        results = {
            'phase': 3,
            'implementation_timestamp': datetime.now().isoformat(),
            'enhancements_applied': []
        }
        
        # 1. Advanced Regularization Techniques
        logger.info("🎯 Implementing advanced regularization...")
        try:
            regularization_results = self._implement_advanced_regularization(X, y)
            results['advanced_regularization'] = regularization_results
            results['enhancements_applied'].append('advanced_regularization')
            logger.info("✅ Advanced regularization implemented")
        except Exception as e:
            logger.error(f"❌ Advanced regularization failed: {e}")
            results['advanced_regularization_error'] = str(e)
        
        # 2. Advanced Ensemble Methods
        logger.info("🎼 Implementing advanced ensemble methods...")
        try:
            ensemble_results = self._implement_advanced_ensembles(X, y)
            results['advanced_ensembles'] = ensemble_results
            results['enhancements_applied'].append('advanced_ensembles')
            logger.info("✅ Advanced ensemble methods implemented")
        except Exception as e:
            logger.error(f"❌ Advanced ensemble methods failed: {e}")
            results['advanced_ensembles_error'] = str(e)
        
        return results
    
    def _load_existing_models(self) -> Dict[str, Any]:
        """Load existing models from tennis_models directory"""
        models = {}
        
        model_files = {
            'random_forest': 'random_forest.pkl',
            'gradient_boosting': 'gradient_boosting.pkl',
            'logistic_regression': 'logistic_regression.pkl',
            'xgboost': 'xgboost.pkl'
        }
        
        for model_name, filename in model_files.items():
            filepath = os.path.join(self.models_dir, filename)
            if os.path.exists(filepath):
                try:
                    models[model_name] = joblib.load(filepath)
                    logger.info(f"✅ Loaded {model_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load {model_name}: {e}")
        
        # Load neural network if available
        nn_filepath = os.path.join(self.models_dir, 'neural_network.h5')
        if os.path.exists(nn_filepath) and TENSORFLOW_AVAILABLE:
            try:
                models['neural_network'] = keras.models.load_model(nn_filepath)
                logger.info("✅ Loaded neural_network")
            except Exception as e:
                logger.warning(f"⚠️ Could not load neural_network: {e}")
        
        return models
    
    def _get_best_existing_model(self, models: Dict[str, Any]) -> Any:
        """Get the best performing existing model"""
        if not models:
            return None
        
        # Use ensemble weights from metadata to determine best model
        ensemble_weights = self.metadata.get('ensemble_weights', {})
        
        if ensemble_weights:
            best_model_name = max(ensemble_weights, key=ensemble_weights.get)
            return models.get(best_model_name)
        
        # Fallback to first available model
        return next(iter(models.values()))
    
    def _save_tuned_models(self, tuned_models: Dict[str, Any]) -> None:
        """Save tuned models to disk"""
        for model_name, model in tuned_models.items():
            try:
                filepath = os.path.join(self.models_dir, f"{model_name}.pkl")
                joblib.dump(model, filepath)
                logger.info(f"✅ Saved tuned model: {model_name}")
            except Exception as e:
                logger.error(f"❌ Could not save {model_name}: {e}")
    
    def _perform_bayesian_optimization(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Perform Bayesian optimization for hyperparameter tuning"""
        if not BAYESIAN_OPT_AVAILABLE:
            return {'error': 'scikit-optimize not available'}
        
        results = {}
        
        # Define search spaces
        rf_space = {
            'n_estimators': Integer(50, 500),
            'max_depth': Integer(5, 20),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10),
        }
        
        try:
            # Bayesian optimization for Random Forest
            rf_bayes = BayesSearchCV(
                RandomForestClassifier(random_state=42),
                rf_space,
                n_iter=30,
                cv=3,
                n_jobs=-1,
                random_state=42
            )
            
            rf_bayes.fit(X.values, y)
            
            results['random_forest'] = {
                'best_params': rf_bayes.best_params_,
                'best_score': rf_bayes.best_score_,
                'optimization_type': 'bayesian'
            }
            
            # Save optimized model
            optimized_rf_path = os.path.join(self.models_dir, 'random_forest_bayesian_optimized.pkl')
            joblib.dump(rf_bayes.best_estimator_, optimized_rf_path)
            
        except Exception as e:
            results['random_forest_error'] = str(e)
        
        return results
    
    def _implement_advanced_regularization(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Implement advanced regularization techniques"""
        results = {}
        
        # L1/L2 regularization with different strengths
        regularization_strengths = [0.001, 0.01, 0.1, 1.0, 10.0]
        
        try:
            # Elastic Net with different l1_ratios
            from sklearn.linear_model import ElasticNet
            
            best_score = 0
            best_params = {}
            
            for C in regularization_strengths:
                for l1_ratio in [0.1, 0.5, 0.7, 0.9]:
                    try:
                        # Use LogisticRegression with elasticnet penalty
                        model = LogisticRegression(
                            penalty='elasticnet',
                            C=C,
                            l1_ratio=l1_ratio,
                            solver='saga',
                            max_iter=1000,
                            random_state=42
                        )
                        
                        # Cross-validation score
                        scores = cross_val_score(model, X.values, y, cv=3, scoring='f1')
                        avg_score = scores.mean()
                        
                        if avg_score > best_score:
                            best_score = avg_score
                            best_params = {'C': C, 'l1_ratio': l1_ratio}
                            
                    except Exception as e:
                        continue
            
            results['elastic_net_regularization'] = {
                'best_params': best_params,
                'best_score': best_score
            }
            
            # Train and save best regularized model
            if best_params:
                best_regularized_model = LogisticRegression(
                    penalty='elasticnet',
                    C=best_params['C'],
                    l1_ratio=best_params['l1_ratio'],
                    solver='saga',
                    max_iter=1000,
                    random_state=42
                )
                
                best_regularized_model.fit(X.values, y)
                
                # Save regularized model
                regularized_path = os.path.join(self.models_dir, 'logistic_regression_regularized.pkl')
                joblib.dump(best_regularized_model, regularized_path)
                
        except Exception as e:
            results['regularization_error'] = str(e)
        
        return results
    
    def _implement_advanced_ensembles(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Implement advanced ensemble methods"""
        results = {}
        
        try:
            from sklearn.ensemble import VotingClassifier, StackingClassifier
            from sklearn.model_selection import cross_val_score
            
            # Load existing models
            existing_models = self._load_existing_models()
            
            if len(existing_models) >= 3:
                # Create base estimators list (exclude neural network for voting)
                base_estimators = []
                for name, model in existing_models.items():
                    if name != 'neural_network':  # Exclude neural network for compatibility
                        base_estimators.append((name, model))
                
                if len(base_estimators) >= 2:
                    # 1. Voting Classifier
                    voting_clf = VotingClassifier(
                        estimators=base_estimators,
                        voting='soft'  # Use probability-based voting
                    )
                    
                    voting_scores = cross_val_score(voting_clf, X.values, y, cv=3, scoring='f1')
                    results['voting_ensemble'] = {
                        'cv_score_mean': voting_scores.mean(),
                        'cv_score_std': voting_scores.std(),
                        'estimators_used': [name for name, _ in base_estimators]
                    }
                    
                    # Fit and save voting ensemble
                    voting_clf.fit(X.values, y)
                    voting_path = os.path.join(self.models_dir, 'voting_ensemble.pkl')
                    joblib.dump(voting_clf, voting_path)
                    
                    # 2. Stacking Classifier (if we have enough models)
                    if len(base_estimators) >= 3:
                        stacking_clf = StackingClassifier(
                            estimators=base_estimators[:3],  # Use top 3 models
                            final_estimator=LogisticRegression(random_state=42),
                            cv=3
                        )
                        
                        stacking_scores = cross_val_score(stacking_clf, X.values, y, cv=3, scoring='f1')
                        results['stacking_ensemble'] = {
                            'cv_score_mean': stacking_scores.mean(),
                            'cv_score_std': stacking_scores.std(),
                            'base_estimators': [name for name, _ in base_estimators[:3]]
                        }
                        
                        # Fit and save stacking ensemble
                        stacking_clf.fit(X.values, y)
                        stacking_path = os.path.join(self.models_dir, 'stacking_ensemble.pkl')
                        joblib.dump(stacking_clf, stacking_path)
                
        except Exception as e:
            results['ensemble_error'] = str(e)
        
        return results
    
    def generate_enhancement_report(self, phase_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive enhancement report"""
        report = {
            'enhancement_summary': {
                'phases_implemented': len(phase_results),
                'total_enhancements': sum(len(result.get('enhancements_applied', [])) for result in phase_results),
                'implementation_date': datetime.now().isoformat()
            },
            'phase_results': phase_results,
            'performance_improvements': {},
            'recommendations': []
        }
        
        # Analyze performance improvements
        for result in phase_results:
            phase = result.get('phase', 0)
            
            if 'cross_validation_results' in result:
                cv_results = result['cross_validation_results']
                best_model = max(cv_results.keys(), key=lambda k: cv_results[k].get('f1', {}).get('mean', 0))
                best_f1 = cv_results[best_model].get('f1', {}).get('mean', 0)
                
                report['performance_improvements'][f'phase_{phase}_best_f1'] = best_f1
                report['performance_improvements'][f'phase_{phase}_best_model'] = best_model
        
        # Generate final recommendations
        if len(phase_results) == 1:  # Only Phase 1
            report['recommendations'].extend([
                "✅ Phase 1 enhancements completed successfully",
                "📈 Consider implementing Phase 2 when more data becomes available",
                "🔄 Monitor model performance and retrain with new data"
            ])
        elif len(phase_results) == 2:  # Phases 1 & 2
            report['recommendations'].extend([
                "🚀 Advanced ML enhancements successfully implemented",
                "🎯 Feature selection may have improved model efficiency",
                "📊 Consider Phase 3 for production-ready optimizations"
            ])
        elif len(phase_results) == 3:  # All phases
            report['recommendations'].extend([
                "🏆 Complete ML enhancement roadmap implemented",
                "⚡ System now has advanced regularization and ensemble methods",
                "🔄 Establish automated retraining pipeline",
                "📈 Monitor production performance and adjust as needed"
            ])
        
        # Save report
        report_path = os.path.join(self.models_dir, f'enhancement_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            report['report_saved_to'] = report_path
        except Exception as e:
            logger.error(f"Could not save enhancement report: {e}")
        
        return report

def main():
    """Main function to demonstrate enhanced ML training system"""
    print("🚀 ENHANCED ML TRAINING SYSTEM FOR TENNIS PREDICTION")
    print("=" * 70)
    
    # Initialize the enhanced training system
    enhanced_system = EnhancedMLTrainingSystem()
    
    # Analyze system readiness
    readiness_analysis = enhanced_system.analyze_system_readiness()
    
    print("\n📊 SYSTEM READINESS ANALYSIS:")
    print(f"Data Quality Score: {readiness_analysis['data_quality_analysis']['data_quality_score']:.1f}/100")
    
    for phase, ready in readiness_analysis['phases_ready'].items():
        status = "✅ READY" if ready else "⚠️ NOT READY"
        print(f"{phase.upper()}: {status}")
    
    print("\n🎯 STRATEGIC RECOMMENDATIONS:")
    for rec in readiness_analysis['strategic_recommendations']:
        print(f"  {rec}")
    
    print("\n" + "=" * 70)
    print("✅ Enhanced ML Training System initialized and analyzed!")
    print("🔧 Ready for phased implementation based on data quality assessment")

if __name__ == "__main__":
    main()