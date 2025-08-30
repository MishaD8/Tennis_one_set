"""
Enhanced Machine Learning Module for Tennis Prediction
======================================================

Advanced ML components for improved tennis second set prediction:
- Enhanced feature engineering with momentum, fatigue, and pressure indicators
- Bayesian hyperparameter optimization
- Real-time data integration with WebSocket support
- Dynamic ensemble with contextual weighting
- LSTM-based sequential models for match progression

Author: Tennis ML Enhancement System
Date: 2025-08-30
"""

from .enhanced_feature_engineering import (
    EnhancedTennisFeatureEngineer,
    MatchContext,
    FirstSetStats
)

from .bayesian_hyperparameter_optimizer import (
    TennisBayesianOptimizer
)

from .realtime_data_collector import (
    RealTimeTennisDataCollector,
    LiveMatchState
)

from .dynamic_ensemble import (
    DynamicTennisEnsemble,
    ContextualWeightCalculator
)

try:
    from .lstm_sequential_model import (
        TennisLSTMModel,
        MatchSequence,
        SequencePreprocessor
    )
except ImportError:
    # TensorFlow not available
    pass

__version__ = "1.0.0"
__author__ = "Tennis ML Enhancement System"

__all__ = [
    'EnhancedTennisFeatureEngineer',
    'MatchContext',
    'FirstSetStats',
    'TennisBayesianOptimizer',
    'RealTimeTennisDataCollector',
    'LiveMatchState',
    'DynamicTennisEnsemble',
    'ContextualWeightCalculator',
    'TennisLSTMModel',
    'MatchSequence',
    'SequencePreprocessor'
]