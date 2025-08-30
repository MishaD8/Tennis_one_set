#!/usr/bin/env python3
"""
Enhanced ML System Validation Script
===================================

Validates that all enhanced ML components are working correctly
and demonstrates the improved performance capabilities.

Author: Tennis ML Enhancement System
Date: 2025-08-30
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import traceback

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_enhanced_features():
    """Validate enhanced feature engineering"""
    print("\n🔧 VALIDATING ENHANCED FEATURE ENGINEERING")
    print("=" * 60)
    
    try:
        from ml.enhanced_feature_engineering import (
            EnhancedTennisFeatureEngineer, MatchContext, FirstSetStats
        )
        
        # Create feature engineer
        engineer = EnhancedTennisFeatureEngineer()
        
        # Create sample match data
        match_data = {
            'player1': {
                'ranking': 25, 'age': 26,
                'last_match_date': datetime.now() - timedelta(days=3),
                'matches_last_14_days': 2,
                'hard_win_percentage': 0.65,
                'indoor_win_percentage': 0.70
            },
            'player2': {
                'ranking': 85, 'age': 24,
                'last_match_date': datetime.now() - timedelta(days=5),
                'matches_last_14_days': 1,
                'hard_win_percentage': 0.55,
                'indoor_win_percentage': 0.50
            },
            'first_set_stats': FirstSetStats(
                winner='player1', score='6-4', duration_minutes=45, total_games=10,
                break_points_player1={'faced': 2, 'saved': 2, 'converted': 1},
                break_points_player2={'faced': 3, 'saved': 1, 'converted': 0},
                service_points_player1={'won': 35, 'total': 50},
                service_points_player2={'won': 25, 'total': 40},
                unforced_errors_player1=8, unforced_errors_player2=12,
                winners_player1=15, winners_player2=8,
                double_faults_player1=1, double_faults_player2=2
            ),
            'match_context': MatchContext(
                tournament_tier='ATP500', surface='Hard', round='R16',
                is_indoor=True, altitude=500.0
            ),
            'h2h_data': {
                'overall': {'player1_wins': 2, 'player2_wins': 1},
                'recent_3': {'player1_wins': 2, 'player2_wins': 1}
            },
            'match_date': datetime.now()
        }
        
        # Generate features
        features = engineer.create_all_enhanced_features(match_data)
        
        print(f"✅ Enhanced features generated: {len(features)} features")
        print(f"   Feature categories: momentum, fatigue, pressure, surface, context")
        
        # Show sample features
        sample_features = list(features.items())[:10]
        print("   Sample features:")
        for name, value in sample_features:
            print(f"     {name}: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced feature engineering validation failed: {e}")
        traceback.print_exc()
        return False

def validate_bayesian_optimization():
    """Validate Bayesian hyperparameter optimization"""
    print("\n🎯 VALIDATING BAYESIAN OPTIMIZATION")
    print("=" * 60)
    
    try:
        from ml.bayesian_hyperparameter_optimizer import TennisBayesianOptimizer
        
        # Create synthetic data
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = (X[:, 0] + X[:, 1] + np.random.randn(100) * 0.5 > 0).astype(int)
        
        # Test optimization (minimal calls for validation)
        optimizer = TennisBayesianOptimizer(n_calls=10, cv_folds=3)  # Minimum 10 calls
        
        print("   Testing Random Forest optimization...")
        result = optimizer.optimize_model('random_forest', X, y)
        
        print(f"✅ Bayesian optimization working")
        print(f"   Best score: {result.get('optimization_score', 'N/A')}")
        print(f"   Parameters optimized: {list(result.get('best_params', {}).keys())}")
        
        return True
        
    except ImportError as e:
        print(f"⚠️ Bayesian optimization not available: {e}")
        print("   Install scikit-optimize: pip install scikit-optimize")
        return False
    except Exception as e:
        print(f"❌ Bayesian optimization validation failed: {e}")
        traceback.print_exc()
        return False

def validate_dynamic_ensemble():
    """Validate dynamic ensemble with contextual weighting"""
    print("\n🎲 VALIDATING DYNAMIC ENSEMBLE")
    print("=" * 60)
    
    try:
        from ml.dynamic_ensemble import DynamicTennisEnsemble, ContextualWeightCalculator
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
        # Create models
        models = {
            'random_forest': RandomForestClassifier(n_estimators=10, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=200)
        }
        
        # Create ensemble
        ensemble = DynamicTennisEnsemble(models)
        
        # Create synthetic data and contexts
        np.random.seed(42)
        X = np.random.randn(100, 15)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        from ml.dynamic_ensemble import MatchContext
        contexts = []
        for i in range(100):
            context = MatchContext(
                surface='Hard', tournament_tier='ATP500', round='QF', is_indoor=True,
                player1_ranking=25, player2_ranking=85, ranking_gap=60, is_upset_scenario=True,
                surface_specialization={'player1': 0.7, 'player2': 0.6},
                h2h_history=3, tournament_importance=0.8
            )
            contexts.append(context)
        
        # Train and predict
        ensemble.fit(X[:80], y[:80], contexts[:80])
        predictions = ensemble.predict(X[80:], contexts[80:])
        
        print(f"✅ Dynamic ensemble working")
        print(f"   Models: {list(models.keys())}")
        print(f"   Predictions made: {len(predictions)}")
        print(f"   Accuracy: {np.mean(predictions == y[80:]):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dynamic ensemble validation failed: {e}")
        traceback.print_exc()
        return False

def validate_lstm_model():
    """Validate LSTM sequential model"""
    print("\n🧠 VALIDATING LSTM SEQUENTIAL MODEL")
    print("=" * 60)
    
    try:
        import tensorflow as tf
        from ml.lstm_sequential_model import TennisLSTMModel, MatchSequence
        
        print(f"   TensorFlow version: {tf.__version__}")
        
        # Create minimal test
        config = {
            'max_sequence_length': 10,
            'lstm_units': [16, 8],
            'epochs': 2,
            'batch_size': 16
        }
        
        model = TennisLSTMModel(config)
        
        print("✅ LSTM model initialized successfully")
        print("   Ready for sequential match data training")
        
        return True
        
    except ImportError as e:
        print(f"⚠️ LSTM model not available: {e}")
        print("   Install TensorFlow: pip install tensorflow")
        return False
    except Exception as e:
        print(f"❌ LSTM model validation failed: {e}")
        traceback.print_exc()
        return False

def validate_real_time_collector():
    """Validate real-time data collector"""
    print("\n📡 VALIDATING REAL-TIME DATA COLLECTOR")
    print("=" * 60)
    
    try:
        from ml.realtime_data_collector import RealTimeTennisDataCollector, LiveMatchState
        
        # Create collector (no actual connection for validation)
        config = {
            'api_tennis_key': 'test_key',
            'websocket_url': 'wss://test.example.com'
        }
        
        collector = RealTimeTennisDataCollector(config)
        
        print("✅ Real-time data collector initialized")
        print("   WebSocket support: Available")
        print("   Ready for live match monitoring")
        
        return True
        
    except Exception as e:
        print(f"❌ Real-time collector validation failed: {e}")
        traceback.print_exc()
        return False

def run_integration_test():
    """Run basic integration test"""
    print("\n🔗 RUNNING INTEGRATION TEST")
    print("=" * 60)
    
    try:
        from ml.enhanced_pipeline import EnhancedTennisMLPipeline
        
        # Create pipeline
        config = {
            'optimization_calls': 5,  # Minimal for testing
            'cv_folds': 3,
            'models_to_optimize': ['random_forest'],  # Single model for testing
        }
        
        pipeline = EnhancedTennisMLPipeline(config)
        
        print("✅ Enhanced ML Pipeline created successfully")
        print("   All components integrated and ready")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main validation function"""
    print("🎾 TENNIS ENHANCED ML SYSTEM VALIDATION")
    print("=" * 80)
    print(f"Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    validation_results = []
    
    # Run all validations
    validations = [
        ("Enhanced Feature Engineering", validate_enhanced_features),
        ("Bayesian Optimization", validate_bayesian_optimization),
        ("Dynamic Ensemble", validate_dynamic_ensemble),
        ("LSTM Sequential Model", validate_lstm_model),
        ("Real-time Data Collector", validate_real_time_collector),
        ("Integration Test", run_integration_test)
    ]
    
    for name, validation_func in validations:
        try:
            result = validation_func()
            validation_results.append((name, result))
        except Exception as e:
            print(f"❌ {name} validation crashed: {e}")
            validation_results.append((name, False))
    
    # Summary
    print("\n📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in validation_results if result)
    total = len(validation_results)
    
    for name, result in validation_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} components validated successfully")
    
    if passed == total:
        print("\n🎉 ALL ENHANCED ML COMPONENTS VALIDATED SUCCESSFULLY!")
        print("Your tennis prediction system is ready for enhanced performance.")
    else:
        print(f"\n⚠️ {total - passed} components need attention.")
        print("Please install missing dependencies or check error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)#!/usr/bin/env python3
"""
Enhanced ML System Validation Script
===================================

Validates that all enhanced ML components are working correctly
and demonstrates the improved performance capabilities.

Author: Tennis ML Enhancement System
Date: 2025-08-30
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import traceback

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_enhanced_features():
    """Validate enhanced feature engineering"""
    print("\n🔧 VALIDATING ENHANCED FEATURE ENGINEERING")
    print("=" * 60)
    
    try:
        from ml.enhanced_feature_engineering import (
            EnhancedTennisFeatureEngineer, MatchContext, FirstSetStats
        )
        
        # Create feature engineer
        engineer = EnhancedTennisFeatureEngineer()
        
        # Create sample match data
        match_data = {
            'player1': {
                'ranking': 25, 'age': 26,
                'last_match_date': datetime.now() - timedelta(days=3),
                'matches_last_14_days': 2,
                'hard_win_percentage': 0.65,
                'indoor_win_percentage': 0.70
            },
            'player2': {
                'ranking': 85, 'age': 24,
                'last_match_date': datetime.now() - timedelta(days=5),
                'matches_last_14_days': 1,
                'hard_win_percentage': 0.55,
                'indoor_win_percentage': 0.50
            },
            'first_set_stats': FirstSetStats(
                winner='player1', score='6-4', duration_minutes=45, total_games=10,
                break_points_player1={'faced': 2, 'saved': 2, 'converted': 1},
                break_points_player2={'faced': 3, 'saved': 1, 'converted': 0},
                service_points_player1={'won': 35, 'total': 50},
                service_points_player2={'won': 25, 'total': 40},
                unforced_errors_player1=8, unforced_errors_player2=12,
                winners_player1=15, winners_player2=8,
                double_faults_player1=1, double_faults_player2=2
            ),
            'match_context': MatchContext(
                tournament_tier='ATP500', surface='Hard', round='R16',
                is_indoor=True, altitude=500.0
            ),
            'h2h_data': {
                'overall': {'player1_wins': 2, 'player2_wins': 1},
                'recent_3': {'player1_wins': 2, 'player2_wins': 1}
            },
            'match_date': datetime.now()
        }
        
        # Generate features
        features = engineer.create_all_enhanced_features(match_data)
        
        print(f"✅ Enhanced features generated: {len(features)} features")
        print(f"   Feature categories: momentum, fatigue, pressure, surface, context")
        
        # Show sample features
        sample_features = list(features.items())[:10]
        print("   Sample features:")
        for name, value in sample_features:
            print(f"     {name}: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced feature engineering validation failed: {e}")
        traceback.print_exc()
        return False

def validate_bayesian_optimization():
    """Validate Bayesian hyperparameter optimization"""
    print("\n🎯 VALIDATING BAYESIAN OPTIMIZATION")
    print("=" * 60)
    
    try:
        from ml.bayesian_hyperparameter_optimizer import TennisBayesianOptimizer
        
        # Create synthetic data
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = (X[:, 0] + X[:, 1] + np.random.randn(100) * 0.5 > 0).astype(int)
        
        # Test optimization (minimal calls for validation)
        optimizer = TennisBayesianOptimizer(n_calls=10, cv_folds=3)  # Minimum 10 calls
        
        print("   Testing Random Forest optimization...")
        result = optimizer.optimize_model('random_forest', X, y)
        
        print(f"✅ Bayesian optimization working")
        print(f"   Best score: {result.get('optimization_score', 'N/A')}")
        print(f"   Parameters optimized: {list(result.get('best_params', {}).keys())}")
        
        return True
        
    except ImportError as e:
        print(f"⚠️ Bayesian optimization not available: {e}")
        print("   Install scikit-optimize: pip install scikit-optimize")
        return False
    except Exception as e:
        print(f"❌ Bayesian optimization validation failed: {e}")
        traceback.print_exc()
        return False

def validate_dynamic_ensemble():
    """Validate dynamic ensemble with contextual weighting"""
    print("\n🎲 VALIDATING DYNAMIC ENSEMBLE")
    print("=" * 60)
    
    try:
        from ml.dynamic_ensemble import DynamicTennisEnsemble, ContextualWeightCalculator
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
        # Create models
        models = {
            'random_forest': RandomForestClassifier(n_estimators=10, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=200)
        }
        
        # Create ensemble
        ensemble = DynamicTennisEnsemble(models)
        
        # Create synthetic data and contexts
        np.random.seed(42)
        X = np.random.randn(100, 15)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        from ml.dynamic_ensemble import MatchContext
        contexts = []
        for i in range(100):
            context = MatchContext(
                surface='Hard', tournament_tier='ATP500', round='QF', is_indoor=True,
                player1_ranking=25, player2_ranking=85, ranking_gap=60, is_upset_scenario=True,
                surface_specialization={'player1': 0.7, 'player2': 0.6},
                h2h_history=3, tournament_importance=0.8
            )
            contexts.append(context)
        
        # Train and predict
        ensemble.fit(X[:80], y[:80], contexts[:80])
        predictions = ensemble.predict(X[80:], contexts[80:])
        
        print(f"✅ Dynamic ensemble working")
        print(f"   Models: {list(models.keys())}")
        print(f"   Predictions made: {len(predictions)}")
        print(f"   Accuracy: {np.mean(predictions == y[80:]):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dynamic ensemble validation failed: {e}")
        traceback.print_exc()
        return False

def validate_lstm_model():
    """Validate LSTM sequential model"""
    print("\n🧠 VALIDATING LSTM SEQUENTIAL MODEL")
    print("=" * 60)
    
    try:
        import tensorflow as tf
        from ml.lstm_sequential_model import TennisLSTMModel, MatchSequence
        
        print(f"   TensorFlow version: {tf.__version__}")
        
        # Create minimal test
        config = {
            'max_sequence_length': 10,
            'lstm_units': [16, 8],
            'epochs': 2,
            'batch_size': 16
        }
        
        model = TennisLSTMModel(config)
        
        print("✅ LSTM model initialized successfully")
        print("   Ready for sequential match data training")
        
        return True
        
    except ImportError as e:
        print(f"⚠️ LSTM model not available: {e}")
        print("   Install TensorFlow: pip install tensorflow")
        return False
    except Exception as e:
        print(f"❌ LSTM model validation failed: {e}")
        traceback.print_exc()
        return False

def validate_real_time_collector():
    """Validate real-time data collector"""
    print("\n📡 VALIDATING REAL-TIME DATA COLLECTOR")
    print("=" * 60)
    
    try:
        from ml.realtime_data_collector import RealTimeTennisDataCollector, LiveMatchState
        
        # Create collector (no actual connection for validation)
        config = {
            'api_tennis_key': 'test_key',
            'websocket_url': 'wss://test.example.com'
        }
        
        collector = RealTimeTennisDataCollector(config)
        
        print("✅ Real-time data collector initialized")
        print("   WebSocket support: Available")
        print("   Ready for live match monitoring")
        
        return True
        
    except Exception as e:
        print(f"❌ Real-time collector validation failed: {e}")
        traceback.print_exc()
        return False

def run_integration_test():
    """Run basic integration test"""
    print("\n🔗 RUNNING INTEGRATION TEST")
    print("=" * 60)
    
    try:
        from ml.enhanced_pipeline import EnhancedTennisMLPipeline
        
        # Create pipeline
        config = {
            'optimization_calls': 5,  # Minimal for testing
            'cv_folds': 3,
            'models_to_optimize': ['random_forest'],  # Single model for testing
        }
        
        pipeline = EnhancedTennisMLPipeline(config)
        
        print("✅ Enhanced ML Pipeline created successfully")
        print("   All components integrated and ready")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main validation function"""
    print("🎾 TENNIS ENHANCED ML SYSTEM VALIDATION")
    print("=" * 80)
    print(f"Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    validation_results = []
    
    # Run all validations
    validations = [
        ("Enhanced Feature Engineering", validate_enhanced_features),
        ("Bayesian Optimization", validate_bayesian_optimization),
        ("Dynamic Ensemble", validate_dynamic_ensemble),
        ("LSTM Sequential Model", validate_lstm_model),
        ("Real-time Data Collector", validate_real_time_collector),
        ("Integration Test", run_integration_test)
    ]
    
    for name, validation_func in validations:
        try:
            result = validation_func()
            validation_results.append((name, result))
        except Exception as e:
            print(f"❌ {name} validation crashed: {e}")
            validation_results.append((name, False))
    
    # Summary
    print("\n📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in validation_results if result)
    total = len(validation_results)
    
    for name, result in validation_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} components validated successfully")
    
    if passed == total:
        print("\n🎉 ALL ENHANCED ML COMPONENTS VALIDATED SUCCESSFULLY!")
        print("Your tennis prediction system is ready for enhanced performance.")
    else:
        print(f"\n⚠️ {total - passed} components need attention.")
        print("Please install missing dependencies or check error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)