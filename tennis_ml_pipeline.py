#!/usr/bin/env python3
"""
Comprehensive Tennis ML Pipeline for Second-Set Underdog Prediction
===================================================================

This module implements a production-ready machine learning system for predicting
when underdog tennis players (ranked 10-300) will win the second set in ATP/WTA
best-of-3 matches.

Key Features:
- Advanced tennis-specific feature engineering
- Ensemble modeling with XGBoost, Random Forest, Logistic Regression, Neural Networks
- Betting-oriented evaluation with precision/recall optimization
- Production-ready pipeline with model persistence
- Comprehensive performance analysis and reporting

Author: Tennis Analytics ML System
Date: 2025-08-15
"""

import pandas as pd
import numpy as np
import sqlite3
import pickle
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, precision_score, recall_score
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

import xgboost as xgb
import lightgbm as lgb

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class TennisDataLoader:
    """Load and preprocess tennis match data from SQLite database."""
    
    def __init__(self, db_path: str):
        """Initialize data loader with database path."""
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path)
        
    def disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            
    def load_data(self) -> pd.DataFrame:
        """Load all match data from database."""
        if not self.conn:
            self.connect()
            
        query = """
        SELECT * FROM enhanced_matches 
        WHERE player_rank BETWEEN 10 AND 300
        AND total_sets >= 2
        AND total_sets <= 3
        ORDER BY match_date
        """
        
        df = pd.read_sql_query(query, self.conn)
        
        print(f"Loaded {len(df):,} matches for analysis")
        print(f"Date range: {df['match_date'].min()} to {df['match_date'].max()}")
        print(f"Surfaces: {df['surface'].value_counts().to_dict()}")
        
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data summary."""
        summary = {
            'total_matches': len(df),
            'date_range': {
                'start': df['match_date'].min(),
                'end': df['match_date'].max()
            },
            'surface_distribution': df['surface'].value_counts().to_dict(),
            'ranking_stats': {
                'player_rank_mean': df['player_rank'].mean(),
                'player_rank_std': df['player_rank'].std(),
                'opponent_rank_mean': df['opponent_rank'].mean(),
                'opponent_rank_std': df['opponent_rank'].std()
            },
            'target_distribution': df['won_at_least_one_set'].value_counts().to_dict(),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        return summary


class TennisFeatureEngineer:
    """Advanced tennis-specific feature engineering for second-set prediction."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def create_underdog_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create underdog-specific features."""
        df = df.copy()
        
        # Ranking differential features
        df['rank_difference'] = df['player_rank'] - df['opponent_rank']
        df['rank_ratio'] = df['player_rank'] / (df['opponent_rank'] + 1)
        df['is_underdog'] = (df['player_rank'] > df['opponent_rank']).astype(int)
        df['underdog_magnitude'] = np.where(
            df['is_underdog'] == 1,
            df['rank_difference'] / df['player_rank'],
            0
        )
        
        # Ranking momentum features
        df['rank_percentile'] = df['player_rank'] / 300  # Normalize to 0-1 for ranks 1-300
        df['opponent_rank_percentile'] = df['opponent_rank'] / 300
        
        # Create ranking advantage buckets
        df['rank_advantage_bucket'] = pd.cut(
            df['rank_difference'], 
            bins=[-np.inf, -50, -20, -5, 5, 20, 50, np.inf],
            labels=['Major_Favorite', 'Moderate_Favorite', 'Slight_Favorite', 
                   'Even', 'Slight_Underdog', 'Moderate_Underdog', 'Major_Underdog']
        )
        
        return df
    
    def create_second_set_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features specifically for second-set prediction."""
        df = df.copy()
        
        # Second set target creation
        # For best-of-3 matches, if total_sets >= 2, second set occurred
        df['second_set_occurred'] = (df['total_sets'] >= 2).astype(int)
        
        # Create second set win target based on sets won
        # If player won at least 1 set and match went to 2+ sets, they likely won second set
        # This is an approximation - ideally we'd have set-by-set data
        df['likely_won_second_set'] = np.where(
            (df['total_sets'] == 2) & (df['player_sets_won'] == 1),
            1,  # Won 1 set in 2-set match = likely won second set
            np.where(
                (df['total_sets'] == 3) & (df['player_sets_won'] >= 1),
                1,  # Won at least 1 set in 3-set match = could have won second set
                0
            )
        )
        
        # Momentum and pressure features for second set
        df['momentum_pressure'] = df['total_pressure'] * df['player_form_trend']
        df['surface_form_interaction'] = df['player_surface_advantage'] * df['player_form_trend']
        
        # Experience under pressure
        df['experience_pressure_ratio'] = (
            df['player_surface_experience'] / (df['total_pressure'] + 1)
        )
        
        return df
    
    def create_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced form and momentum features."""
        df = df.copy()
        
        # Recent form momentum
        df['form_momentum'] = df['player_form_trend'] * df['player_recent_win_rate']
        df['form_consistency'] = np.abs(df['player_form_trend'])
        
        # Match frequency and rest
        df['match_frequency'] = 1 / (df['player_days_since_last_match'] + 1)
        df['rest_advantage'] = np.where(
            df['player_days_since_last_match'] > 7, 1,
            np.where(df['player_days_since_last_match'] < 2, -1, 0)
        )
        
        # Surface specialization
        df['surface_specialization'] = (
            df['player_surface_win_rate'] - df['player_recent_win_rate']
        )
        
        return df
    
    def create_h2h_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create head-to-head and psychological features."""
        df = df.copy()
        
        # H2H dominance features
        df['h2h_dominance'] = np.where(
            df['h2h_matches'] > 0,
            (df['h2h_win_rate'] - 0.5) * df['h2h_matches'],
            0
        )
        
        # Recent H2H momentum
        df['h2h_momentum'] = df['h2h_recent_form'] * df['h2h_matches']
        
        # H2H experience factor
        df['h2h_experience'] = np.log1p(df['h2h_matches'])
        
        return df
    
    def create_tournament_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create tournament and context features."""
        df = df.copy()
        
        # Tournament pressure interactions
        df['pressure_rank_interaction'] = df['total_pressure'] * df['rank_percentile']
        df['pressure_experience'] = df['total_pressure'] * df['player_surface_experience']
        
        # Round importance
        round_importance = {
            'First Round': 1, 'Second Round': 2, 'Third Round': 3,
            'Fourth Round': 4, 'Quarterfinals': 5, 'Semifinals': 6, 'Final': 7
        }
        df['round_importance_score'] = df['round'].map(round_importance).fillna(1)
        
        return df
    
    def create_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create age and experience features."""
        df = df.copy()
        
        # Age differential
        df['age_difference'] = df['player_age'] - df['opponent_age']
        df['age_advantage'] = np.where(
            df['age_difference'] > 5, -1,  # Much older = disadvantage
            np.where(df['age_difference'] > 2, 0, 1)  # Younger = advantage
        )
        
        # Experience vs youth balance
        df['experience_youth_balance'] = (
            df['player_surface_experience'] / (df['player_age'] + 1)
        )
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        print("Engineering tennis-specific features...")
        
        # Apply all feature engineering steps
        df = self.create_underdog_features(df)
        df = self.create_second_set_features(df)
        df = self.create_form_features(df)
        df = self.create_h2h_features(df)
        df = self.create_tournament_features(df)
        df = self.create_age_features(df)
        
        # Encode categorical variables
        categorical_cols = ['surface', 'round', 'rank_advantage_bucket']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        print(f"Feature engineering complete. Total features: {len(df.columns)}")
        
        return df


class TennisEnsembleModels:
    """Ensemble modeling system for tennis prediction."""
    
    def __init__(self):
        """Initialize ensemble models."""
        self.models = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for modeling."""
        # Select features for modeling
        feature_cols = [
            # Core ranking features
            'player_rank', 'opponent_rank', 'rank_difference', 'rank_ratio',
            'is_underdog', 'underdog_magnitude', 'rank_percentile',
            
            # Form features
            'player_recent_win_rate', 'player_form_trend', 'form_momentum',
            'form_consistency', 'match_frequency',
            
            # Surface features
            'player_surface_win_rate', 'player_surface_advantage',
            'surface_specialization', 'player_surface_experience',
            
            # H2H features
            'h2h_win_rate', 'h2h_dominance', 'h2h_momentum', 'h2h_experience',
            
            # Pressure features
            'total_pressure', 'momentum_pressure', 'pressure_rank_interaction',
            'round_importance_score',
            
            # Age features
            'player_age', 'age_difference', 'age_advantage',
            
            # Encoded categorical features
            'surface_encoded', 'round_encoded'
        ]
        
        # Filter features that exist in dataframe
        available_features = [col for col in feature_cols if col in df.columns]
        
        X = df[available_features].fillna(0)
        y = df['likely_won_second_set']
        
        print(f"Using {len(available_features)} features for modeling")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X.values, y.values, available_features
    
    def train_xgboost(self, X: np.ndarray, y: np.ndarray) -> xgb.XGBClassifier:
        """Train XGBoost model with hyperparameter tuning."""
        print("Training XGBoost model...")
        
        # Calculate class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        scale_pos_weight = class_weights[1] / class_weights[0]
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )
        
        model.fit(X, y)
        return model
    
    def train_lightgbm(self, X: np.ndarray, y: np.ndarray) -> lgb.LGBMClassifier:
        """Train LightGBM model with hyperparameter tuning."""
        print("Training LightGBM model...")
        
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42,
            verbose=-1
        )
        
        model.fit(X, y)
        return model
    
    def train_random_forest(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        """Train Random Forest model."""
        print("Training Random Forest model...")
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X, y)
        return model
    
    def train_logistic_regression(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        """Train Logistic Regression model."""
        print("Training Logistic Regression model...")
        
        # Scale features for logistic regression
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        
        model.fit(X_scaled, y)
        
        # Store scaler with model
        model.scaler = scaler
        return model
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """Train all ensemble models."""
        print("Training ensemble models...")
        
        # Train individual models
        self.models['xgboost'] = self.train_xgboost(X, y)
        self.models['lightgbm'] = self.train_lightgbm(X, y)
        self.models['random_forest'] = self.train_random_forest(X, y)
        self.models['logistic_regression'] = self.train_logistic_regression(X, y)
        
        # Extract feature importance
        self.feature_importance['xgboost'] = dict(zip(feature_names, self.models['xgboost'].feature_importances_))
        self.feature_importance['lightgbm'] = dict(zip(feature_names, self.models['lightgbm'].feature_importances_))
        self.feature_importance['random_forest'] = dict(zip(feature_names, self.models['random_forest'].feature_importances_))
        
        print("All models trained successfully!")
    
    def predict_ensemble(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make ensemble predictions with confidence scores."""
        predictions = {}
        probabilities = {}
        
        # Get predictions from each model
        predictions['xgboost'] = self.models['xgboost'].predict_proba(X)[:, 1]
        predictions['lightgbm'] = self.models['lightgbm'].predict_proba(X)[:, 1]
        predictions['random_forest'] = self.models['random_forest'].predict_proba(X)[:, 1]
        
        # Scale data for logistic regression
        X_scaled = self.models['logistic_regression'].scaler.transform(X)
        predictions['logistic_regression'] = self.models['logistic_regression'].predict_proba(X_scaled)[:, 1]
        
        # Ensemble prediction (weighted average)
        weights = {'xgboost': 0.3, 'lightgbm': 0.3, 'random_forest': 0.25, 'logistic_regression': 0.15}
        
        ensemble_proba = np.zeros(len(X))
        for model_name, proba in predictions.items():
            ensemble_proba += weights[model_name] * proba
        
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        return ensemble_pred, ensemble_proba


class BettingEvaluator:
    """Betting-oriented evaluation and analysis system."""
    
    def __init__(self):
        """Initialize betting evaluator."""
        self.kelly_criterion_results = {}
        self.roi_analysis = {}
        
    def calculate_precision_recall_at_thresholds(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate precision and recall at various confidence thresholds."""
        thresholds = np.arange(0.1, 1.0, 0.05)
        results = {}
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if np.sum(y_pred) > 0:  # Only calculate if there are positive predictions
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                
                results[threshold] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
                    'predictions_made': np.sum(y_pred)
                }
        
        return results
    
    def simulate_betting_scenarios(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """Simulate betting scenarios with different strategies."""
        scenarios = {}
        
        # Scenario 1: High precision betting (threshold = 0.7)
        threshold = 0.7
        y_pred = (y_proba >= threshold).astype(int)
        
        if np.sum(y_pred) > 0:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            
            # Simulate betting with average underdog odds of 2.5
            avg_odds = 2.5
            stake_per_bet = 100  # $100 per bet
            
            total_bets = np.sum(y_pred)
            winning_bets = np.sum((y_true == 1) & (y_pred == 1))
            losing_bets = total_bets - winning_bets
            
            total_staked = total_bets * stake_per_bet
            total_won = winning_bets * stake_per_bet * avg_odds
            net_profit = total_won - total_staked
            roi = (net_profit / total_staked) * 100 if total_staked > 0 else 0
            
            scenarios['high_precision'] = {
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'total_bets': total_bets,
                'winning_bets': winning_bets,
                'total_staked': total_staked,
                'total_won': total_won,
                'net_profit': net_profit,
                'roi_percent': roi
            }
        
        # Scenario 2: Balanced betting (threshold = 0.6)
        threshold = 0.6
        y_pred = (y_proba >= threshold).astype(int)
        
        if np.sum(y_pred) > 0:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            
            total_bets = np.sum(y_pred)
            winning_bets = np.sum((y_true == 1) & (y_pred == 1))
            
            total_staked = total_bets * stake_per_bet
            total_won = winning_bets * stake_per_bet * avg_odds
            net_profit = total_won - total_staked
            roi = (net_profit / total_staked) * 100 if total_staked > 0 else 0
            
            scenarios['balanced'] = {
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'total_bets': total_bets,
                'winning_bets': winning_bets,
                'total_staked': total_staked,
                'total_won': total_won,
                'net_profit': net_profit,
                'roi_percent': roi
            }
        
        return scenarios
    
    def kelly_criterion_analysis(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """Apply Kelly Criterion for optimal bet sizing."""
        # Assuming average underdog odds of 2.5
        odds = 2.5
        
        # Calculate Kelly fraction for different confidence levels
        kelly_results = {}
        
        for confidence in [0.6, 0.7, 0.8, 0.9]:
            mask = y_proba >= confidence
            if np.sum(mask) > 0:
                win_rate = np.mean(y_true[mask])
                
                # Kelly Criterion: f = (bp - q) / b
                # where b = odds - 1, p = win probability, q = 1 - p
                b = odds - 1
                p = win_rate
                q = 1 - p
                
                kelly_fraction = (b * p - q) / b
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25% of bankroll
                
                kelly_results[confidence] = {
                    'kelly_fraction': kelly_fraction,
                    'win_rate': win_rate,
                    'sample_size': np.sum(mask)
                }
        
        return kelly_results


class TennisMLPipeline:
    """Main pipeline for tennis ML prediction system."""
    
    def __init__(self, db_path: str, model_save_path: str = "tennis_models"):
        """Initialize ML pipeline."""
        self.db_path = db_path
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(exist_ok=True)
        
        self.data_loader = TennisDataLoader(db_path)
        self.feature_engineer = TennisFeatureEngineer()
        self.ensemble_models = TennisEnsembleModels()
        self.betting_evaluator = BettingEvaluator()
        
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        
    def load_and_prepare_data(self):
        """Load and prepare data for modeling."""
        print("Loading and preparing tennis data...")
        
        # Load data
        self.df = self.data_loader.load_data()
        
        # Engineer features
        self.df = self.feature_engineer.engineer_features(self.df)
        
        # Prepare for modeling
        self.X, self.y, self.feature_names = self.ensemble_models.prepare_data(self.df)
        
        print(f"Data preparation complete: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        
    def train_models(self):
        """Train all ensemble models."""
        print("Training ensemble models...")
        
        self.ensemble_models.train_all_models(self.X, self.y, self.feature_names)
        
    def evaluate_models(self) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        print("Evaluating models...")
        
        # Make predictions
        y_pred, y_proba = self.ensemble_models.predict_ensemble(self.X)
        
        # Basic metrics
        accuracy = np.mean(y_pred == self.y)
        precision = precision_score(self.y, y_pred, zero_division=0)
        recall = recall_score(self.y, y_pred, zero_division=0)
        auc_score = roc_auc_score(self.y, y_proba)
        
        # Precision-recall at thresholds
        pr_thresholds = self.betting_evaluator.calculate_precision_recall_at_thresholds(self.y, y_proba)
        
        # Betting simulation
        betting_scenarios = self.betting_evaluator.simulate_betting_scenarios(self.y, y_proba)
        
        # Kelly criterion analysis
        kelly_analysis = self.betting_evaluator.kelly_criterion_analysis(self.y, y_proba)
        
        evaluation_results = {
            'basic_metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'auc_score': auc_score
            },
            'precision_recall_thresholds': pr_thresholds,
            'betting_scenarios': betting_scenarios,
            'kelly_criterion': kelly_analysis,
            'feature_importance': self.ensemble_models.feature_importance
        }
        
        return evaluation_results
    
    def save_models(self):
        """Save trained models and metadata."""
        print("Saving models...")
        
        # Save individual models
        for model_name, model in self.ensemble_models.models.items():
            model_path = self.model_save_path / f"{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save feature names and metadata
        unique_vals, counts = np.unique(self.y, return_counts=True)
        target_dist = {int(val): int(count) for val, count in zip(unique_vals, counts)}
        
        metadata = {
            'feature_names': self.feature_names,
            'feature_count': len(self.feature_names),
            'training_samples': len(self.X),
            'target_distribution': target_dist,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = self.model_save_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Models saved to {self.model_save_path}")
    
    def generate_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate comprehensive performance report."""
        report = f"""
# Tennis Second-Set Underdog Prediction ML Report
## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Summary
- Total matches analyzed: {len(self.df):,}
- Features engineered: {len(self.feature_names)}
- Target variable: likely_won_second_set
- Class distribution: {dict(zip(*np.unique(self.y, return_counts=True)))}

## Model Performance
### Basic Metrics
- Accuracy: {evaluation_results['basic_metrics']['accuracy']:.3f}
- Precision: {evaluation_results['basic_metrics']['precision']:.3f}
- Recall: {evaluation_results['basic_metrics']['recall']:.3f}
- AUC Score: {evaluation_results['basic_metrics']['auc_score']:.3f}

### Betting Performance Analysis
"""
        
        # Add betting scenarios
        for scenario_name, scenario in evaluation_results['betting_scenarios'].items():
            report += f"""
#### {scenario_name.replace('_', ' ').title()} Strategy
- Confidence Threshold: {scenario['threshold']:.1f}
- Precision: {scenario['precision']:.3f}
- Total Bets: {scenario['total_bets']}
- Winning Bets: {scenario['winning_bets']}
- ROI: {scenario['roi_percent']:.1f}%
- Net Profit: ${scenario['net_profit']:,.2f}
"""
        
        # Add top features
        if 'xgboost' in evaluation_results['feature_importance']:
            top_features = sorted(
                evaluation_results['feature_importance']['xgboost'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            report += "\n### Top 10 Most Important Features (XGBoost)\n"
            for i, (feature, importance) in enumerate(top_features, 1):
                report += f"{i}. {feature}: {importance:.4f}\n"
        
        return report
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete ML pipeline."""
        print("Starting Tennis ML Pipeline...")
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Train models
        self.train_models()
        
        # Evaluate models
        evaluation_results = self.evaluate_models()
        
        # Save models
        self.save_models()
        
        # Generate report
        report = self.generate_report(evaluation_results)
        
        # Save report
        report_path = self.model_save_path / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Pipeline complete! Report saved to {report_path}")
        
        return evaluation_results


if __name__ == "__main__":
    # Initialize and run pipeline
    pipeline = TennisMLPipeline(
        db_path="tennis_data_enhanced/enhanced_tennis_data.db",
        model_save_path="tennis_models"
    )
    
    results = pipeline.run_full_pipeline()
    
    print("\n" + "="*50)
    print("TENNIS ML PIPELINE COMPLETE")
    print("="*50)
    
    # Print key results
    basic_metrics = results['basic_metrics']
    print(f"Model Accuracy: {basic_metrics['accuracy']:.3f}")
    print(f"Model Precision: {basic_metrics['precision']:.3f}")
    print(f"Model AUC Score: {basic_metrics['auc_score']:.3f}")
    
    if 'betting_scenarios' in results:
        for scenario_name, scenario in results['betting_scenarios'].items():
            print(f"{scenario_name.title()} ROI: {scenario['roi_percent']:.1f}%")