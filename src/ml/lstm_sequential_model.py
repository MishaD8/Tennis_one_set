#!/usr/bin/env python3
"""
LSTM-based Sequential Model for Tennis Match Progression
=======================================================

Advanced neural network model that processes sequential tennis match data
to predict second set outcomes based on match progression patterns.

Features:
- LSTM architecture for sequential data processing
- Attention mechanism for key moment identification
- Multi-scale temporal features (point, game, set level)
- Momentum and pattern recognition
- Real-time prediction updates during matches

Author: Tennis ML Enhancement System
Date: 2025-08-30
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import json
from datetime import datetime

# Deep learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks, optimizers
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow not available. Install with: pip install tensorflow")
    TENSORFLOW_AVAILABLE = False

# Sklearn for preprocessing and metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)

@dataclass
class MatchSequence:
    """Sequential match data structure"""
    match_id: str
    player1_name: str
    player2_name: str
    player1_ranking: int
    player2_ranking: int
    
    # Sequential features (point-by-point or game-by-game)
    points_sequence: List[Dict[str, Any]]     # Point-level data
    games_sequence: List[Dict[str, Any]]      # Game-level data
    sets_sequence: List[Dict[str, Any]]       # Set-level data
    
    # Match context
    surface: str
    tournament: str
    round: str
    is_indoor: bool
    
    # Target (for training)
    second_set_winner: Optional[int] = None   # 1 or 2, None for live matches
    
    def get_sequence_length(self) -> int:
        """Get length of the longest sequence"""
        return max(len(self.points_sequence), len(self.games_sequence), len(self.sets_sequence))
    
    def get_first_set_data(self) -> Dict[str, List]:
        """Extract first set data for prediction"""
        first_set_points = [p for p in self.points_sequence if p.get('set_number') == 1]
        first_set_games = [g for g in self.games_sequence if g.get('set_number') == 1]
        
        return {
            'points': first_set_points,
            'games': first_set_games,
            'set_result': self.sets_sequence[0] if self.sets_sequence else None
        }

class SequencePreprocessor:
    """Preprocess tennis match sequences for LSTM input"""
    
    def __init__(self, max_sequence_length: int = 100):
        self.max_sequence_length = max_sequence_length
        self.feature_scalers = {}
        self.label_encoders = {}
        self.feature_names = []
        
        # Point-level features to extract
        self.point_features = [
            'serve_speed', 'serve_placement', 'rally_length', 'shot_type',
            'winner_type', 'error_type', 'pressure_situation', 'break_point',
            'set_point', 'match_point', 'server_id', 'winner_id'
        ]
        
        # Game-level features
        self.game_features = [
            'games_player1', 'games_player2', 'break_points_faced', 'break_points_saved',
            'aces', 'double_faults', 'first_serve_percentage', 'first_serve_points_won',
            'second_serve_points_won', 'return_points_won', 'total_points_won',
            'server_id', 'game_winner'
        ]
        
        # Set-level features
        self.set_features = [
            'set_score_player1', 'set_score_player2', 'tiebreak', 'set_duration',
            'total_points', 'total_games', 'breaks_of_serve', 'set_winner'
        ]
    
    def extract_point_features(self, point_data: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from point data"""
        features = []
        
        # Serve features
        features.append(point_data.get('serve_speed', 0) / 200.0)  # Normalize serve speed
        features.append(point_data.get('serve_placement', 0))      # Categorical: 0-8 (court zones)
        features.append(min(point_data.get('rally_length', 1) / 20.0, 1.0))  # Normalize rally length
        
        # Shot type (one-hot encoding)
        shot_types = ['serve', 'return', 'forehand', 'backhand', 'volley', 'smash', 'drop']
        shot_type = point_data.get('shot_type', 'serve')
        shot_features = [1.0 if shot_type == st else 0.0 for st in shot_types]
        features.extend(shot_features)
        
        # Outcome features
        features.append(1.0 if point_data.get('winner_type') else 0.0)    # Winner vs error
        features.append(1.0 if point_data.get('break_point', False) else 0.0)
        features.append(1.0 if point_data.get('set_point', False) else 0.0)
        features.append(1.0 if point_data.get('match_point', False) else 0.0)
        
        # Player who served/won
        features.append(point_data.get('server_id', 1))    # 1 or 2
        features.append(point_data.get('winner_id', 1))    # 1 or 2
        
        return np.array(features, dtype=np.float32)
    
    def extract_game_features(self, game_data: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from game data"""
        features = []
        
        # Score features
        features.append(game_data.get('games_player1', 0) / 7.0)    # Normalize to typical set length
        features.append(game_data.get('games_player2', 0) / 7.0)
        
        # Service statistics
        features.append(game_data.get('aces', 0) / 5.0)               # Normalize aces per game
        features.append(game_data.get('double_faults', 0) / 3.0)      # Normalize double faults
        features.append(game_data.get('first_serve_percentage', 60) / 100.0)
        features.append(game_data.get('first_serve_points_won', 60) / 100.0)
        features.append(game_data.get('second_serve_points_won', 40) / 100.0)
        
        # Return and pressure features
        features.append(game_data.get('return_points_won', 30) / 100.0)
        features.append(game_data.get('break_points_faced', 0) / 5.0)
        features.append(game_data.get('break_points_saved', 0) / 5.0)
        
        # Total points
        features.append(game_data.get('total_points_won', 4) / 10.0)
        
        # Game context
        features.append(game_data.get('server_id', 1))      # 1 or 2
        features.append(game_data.get('game_winner', 1))    # 1 or 2
        
        return np.array(features, dtype=np.float32)
    
    def create_sequence_matrix(self, match_sequence: MatchSequence, 
                              sequence_type: str = 'games') -> np.ndarray:
        """Create matrix representation of match sequence"""
        
        if sequence_type == 'points':
            sequence_data = match_sequence.points_sequence
            extract_func = self.extract_point_features
        elif sequence_type == 'games':
            sequence_data = match_sequence.games_sequence
            extract_func = self.extract_game_features
        else:
            raise ValueError(f"Unknown sequence type: {sequence_type}")
        
        # Extract features for each element in sequence
        sequence_features = []
        for element in sequence_data:
            features = extract_func(element)
            sequence_features.append(features)
        
        if not sequence_features:
            # Return empty sequence with proper dimensions
            feature_dim = len(self.extract_game_features({}))
            return np.zeros((self.max_sequence_length, feature_dim), dtype=np.float32)
        
        # Convert to numpy array
        sequence_matrix = np.array(sequence_features, dtype=np.float32)
        
        # Pad or truncate to max_sequence_length
        if len(sequence_matrix) > self.max_sequence_length:
            sequence_matrix = sequence_matrix[-self.max_sequence_length:]
        elif len(sequence_matrix) < self.max_sequence_length:
            # Pad with zeros at the beginning (older data)
            padding = np.zeros((self.max_sequence_length - len(sequence_matrix), 
                              sequence_matrix.shape[1]), dtype=np.float32)
            sequence_matrix = np.vstack([padding, sequence_matrix])
        
        return sequence_matrix
    
    def create_context_features(self, match_sequence: MatchSequence) -> np.ndarray:
        """Create static context features"""
        features = []
        
        # Player rankings (normalized)
        features.append(min(match_sequence.player1_ranking / 300.0, 1.0))
        features.append(min(match_sequence.player2_ranking / 300.0, 1.0))
        
        # Ranking gap
        ranking_gap = abs(match_sequence.player1_ranking - match_sequence.player2_ranking)
        features.append(min(ranking_gap / 200.0, 1.0))
        
        # Surface encoding
        surface_encoding = {
            'Hard': [1, 0, 0],
            'Clay': [0, 1, 0],
            'Grass': [0, 0, 1]
        }
        surface_features = surface_encoding.get(match_sequence.surface, [1, 0, 0])
        features.extend(surface_features)
        
        # Tournament round encoding
        round_encoding = {
            'R128': 0.1, 'R64': 0.2, 'R32': 0.3, 'R16': 0.5,
            'QF': 0.7, 'SF': 0.9, 'F': 1.0
        }
        round_weight = round_encoding.get(match_sequence.round, 0.3)
        features.append(round_weight)
        
        # Indoor/outdoor
        features.append(1.0 if match_sequence.is_indoor else 0.0)
        
        return np.array(features, dtype=np.float32)

class TennisLSTMModel:
    """LSTM-based model for tennis second set prediction"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.model = None
        self.preprocessor = SequencePreprocessor(self.config['max_sequence_length'])
        self.is_fitted = False
        self.training_history = None
        
        # Feature dimensions
        self.sequence_feature_dim = None
        self.context_feature_dim = None
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model")
        
        logger.info("✅ Tennis LSTM model initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default model configuration"""
        return {
            'max_sequence_length': 50,
            'lstm_units': [64, 32],
            'attention_units': 32,
            'dense_units': [128, 64],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 15,
            'sequence_type': 'games'  # 'points' or 'games'
        }
    
    def _build_model(self, sequence_input_shape: Tuple[int, int], 
                    context_input_shape: Tuple[int,]) -> keras.Model:
        """Build LSTM model architecture"""
        
        # Sequence input (LSTM branch)
        sequence_input = layers.Input(shape=sequence_input_shape, name='sequence_input')
        
        # LSTM layers
        lstm_out = sequence_input
        for i, units in enumerate(self.config['lstm_units']):
            return_sequences = i < len(self.config['lstm_units']) - 1
            lstm_out = layers.LSTM(
                units, 
                return_sequences=return_sequences,
                dropout=self.config['dropout_rate'],
                recurrent_dropout=self.config['dropout_rate'],
                name=f'lstm_{i+1}'
            )(lstm_out)
        
        # Attention mechanism (if return_sequences=True for last LSTM)
        if len(self.config['lstm_units']) > 1:
            # Add attention to the second-to-last LSTM output
            sequence_input_attention = layers.Input(shape=sequence_input_shape, name='sequence_attention')
            lstm_attention = sequence_input_attention
            
            for i, units in enumerate(self.config['lstm_units'][:-1]):
                lstm_attention = layers.LSTM(
                    units,
                    return_sequences=True,
                    dropout=self.config['dropout_rate'],
                    name=f'lstm_attention_{i+1}'
                )(lstm_attention)
            
            # Attention layer
            attention = layers.Dense(self.config['attention_units'], activation='tanh')(lstm_attention)
            attention = layers.Dense(1, activation='softmax')(attention)
            attention = layers.Flatten()(attention)
            
            # Apply attention weights
            attended_features = layers.Dot(axes=1)([attention, lstm_attention])
            attended_features = layers.GlobalAveragePooling1D()(attended_features)
            
            # Combine LSTM output with attended features
            lstm_combined = layers.Concatenate()([lstm_out, attended_features])
        else:
            lstm_combined = lstm_out
            sequence_input_attention = None
        
        # Context input (dense branch)
        context_input = layers.Input(shape=context_input_shape, name='context_input')
        context_dense = layers.Dense(32, activation='relu')(context_input)
        context_dense = layers.Dropout(self.config['dropout_rate'])(context_dense)
        
        # Combine LSTM and context features
        combined = layers.Concatenate()([lstm_combined, context_dense])
        
        # Dense layers
        dense_out = combined
        for i, units in enumerate(self.config['dense_units']):
            dense_out = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(dense_out)
            dense_out = layers.Dropout(self.config['dropout_rate'])(dense_out)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid', name='prediction')(dense_out)
        
        # Create model
        inputs = [sequence_input, context_input]
        if sequence_input_attention is not None:
            inputs.append(sequence_input_attention)
        
        model = keras.Model(inputs=inputs, outputs=output, name='tennis_lstm_model')
        
        return model
    
    def prepare_training_data(self, match_sequences: List[MatchSequence]) -> Tuple[List[np.ndarray], np.ndarray]:
        """Prepare training data from match sequences"""
        
        sequence_matrices = []
        context_matrices = []
        targets = []
        
        for match_seq in match_sequences:
            if match_seq.second_set_winner is None:
                continue  # Skip matches without target
            
            # Create sequence matrix
            seq_matrix = self.preprocessor.create_sequence_matrix(
                match_seq, self.config['sequence_type']
            )
            sequence_matrices.append(seq_matrix)
            
            # Create context features
            context_features = self.preprocessor.create_context_features(match_seq)
            context_matrices.append(context_features)
            
            # Target (underdog wins second set)
            underdog_player = 2 if match_seq.player1_ranking < match_seq.player2_ranking else 1
            target = 1.0 if match_seq.second_set_winner == underdog_player else 0.0
            targets.append(target)
        
        if not sequence_matrices:
            raise ValueError("No valid training sequences found")
        
        # Convert to numpy arrays
        X_sequence = np.array(sequence_matrices)
        X_context = np.array(context_matrices)
        y = np.array(targets)
        
        # Store dimensions
        self.sequence_feature_dim = X_sequence.shape[2]
        self.context_feature_dim = X_context.shape[1]
        
        logger.info(f"Prepared training data: {len(X_sequence)} sequences")
        logger.info(f"Sequence shape: {X_sequence.shape}")
        logger.info(f"Context shape: {X_context.shape}")
        logger.info(f"Target distribution: {np.mean(y):.3f}")
        
        return [X_sequence, X_context], y
    
    def fit(self, match_sequences: List[MatchSequence], 
           validation_split: float = 0.2) -> Dict[str, Any]:
        """Train the LSTM model"""
        
        logger.info("🔧 Training Tennis LSTM model...")
        
        # Prepare training data
        X, y = self.prepare_training_data(match_sequences)
        
        # Build model
        self.model = self._build_model(
            sequence_input_shape=(self.config['max_sequence_length'], self.sequence_feature_dim),
            context_input_shape=(self.context_feature_dim,)
        )
        
        # Compile model
        optimizer = optimizers.Adam(learning_rate=self.config['learning_rate'])
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Model summary
        logger.info("Model architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train model
        start_time = datetime.now()
        
        history = self.model.fit(
            X, y,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_split=validation_split,
            callbacks=callbacks_list,
            verbose=1
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        self.training_history = history.history
        self.is_fitted = True
        
        # Training summary
        final_accuracy = history.history['val_accuracy'][-1]
        final_loss = history.history['val_loss'][-1]
        
        logger.info(f"✅ LSTM training completed in {training_time:.1f}s")
        logger.info(f"   Final validation accuracy: {final_accuracy:.4f}")
        logger.info(f"   Final validation loss: {final_loss:.4f}")
        
        return {
            'training_time_seconds': training_time,
            'final_val_accuracy': final_accuracy,
            'final_val_loss': final_loss,
            'epochs_trained': len(history.history['loss']),
            'history': self.training_history
        }
    
    def predict(self, match_sequences: List[MatchSequence]) -> np.ndarray:
        """Make predictions for match sequences"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare input data
        sequence_matrices = []
        context_matrices = []
        
        for match_seq in match_sequences:
            seq_matrix = self.preprocessor.create_sequence_matrix(
                match_seq, self.config['sequence_type']
            )
            sequence_matrices.append(seq_matrix)
            
            context_features = self.preprocessor.create_context_features(match_seq)
            context_matrices.append(context_features)
        
        X_sequence = np.array(sequence_matrices)
        X_context = np.array(context_matrices)
        
        # Predict
        predictions = self.model.predict([X_sequence, X_context], verbose=0)
        
        return predictions.flatten()
    
    def predict_live_match(self, match_sequence: MatchSequence) -> Dict[str, Any]:
        """Make live prediction for ongoing match"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get prediction probability
        probability = self.predict([match_sequence])[0]
        
        # Determine underdog
        underdog_player = 2 if match_sequence.player1_ranking < match_sequence.player2_ranking else 1
        
        # Create prediction result
        prediction_result = {
            'underdog_player': underdog_player,
            'underdog_player_name': match_sequence.player2_name if underdog_player == 2 else match_sequence.player1_name,
            'underdog_win_probability': float(probability),
            'favorite_win_probability': float(1.0 - probability),
            'confidence_level': self._get_confidence_level(probability),
            'prediction_time': datetime.now().isoformat(),
            'sequence_length': match_sequence.get_sequence_length(),
            'model_type': 'LSTM'
        }
        
        return prediction_result
    
    def _get_confidence_level(self, probability: float) -> str:
        """Get confidence level description"""
        if probability < 0.45 or probability > 0.55:
            return "High"
        elif probability < 0.48 or probability > 0.52:
            return "Medium"
        else:
            return "Low"
    
    def evaluate(self, match_sequences: List[MatchSequence]) -> Dict[str, float]:
        """Evaluate model performance"""
        
        # Prepare test data
        X, y_true = self.prepare_training_data(match_sequences)
        
        # Predict
        y_pred_proba = self.model.predict(X, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # Save model architecture and weights
        self.model.save(f"{filepath}_lstm_model.h5")
        
        # Save configuration and training history
        model_data = {
            'config': self.config,
            'sequence_feature_dim': self.sequence_feature_dim,
            'context_feature_dim': self.context_feature_dim,
            'is_fitted': self.is_fitted,
            'training_history': self.training_history
        }
        
        with open(f"{filepath}_lstm_config.json", 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"✅ LSTM model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'TennisLSTMModel':
        """Load a saved model"""
        
        # Load configuration
        with open(f"{filepath}_lstm_config.json", 'r') as f:
            model_data = json.load(f)
        
        # Create model instance
        lstm_model = cls(model_data['config'])
        
        # Load trained model
        lstm_model.model = keras.models.load_model(f"{filepath}_lstm_model.h5")
        
        # Restore attributes
        lstm_model.sequence_feature_dim = model_data['sequence_feature_dim']
        lstm_model.context_feature_dim = model_data['context_feature_dim']
        lstm_model.is_fitted = model_data['is_fitted']
        lstm_model.training_history = model_data['training_history']
        
        logger.info(f"✅ LSTM model loaded from {filepath}")
        
        return lstm_model

# Example usage and testing
if __name__ == "__main__":
    # This would normally use real match data
    print("Tennis LSTM Model - Example Usage")
    
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow not available. Please install TensorFlow to use LSTM model.")
        exit(1)
    
    # Create sample match sequences for testing
    sample_sequences = []
    
    for i in range(100):  # Small dataset for testing
        # Create sample game sequence
        games_sequence = []
        for game in range(12):  # Typical first set length
            game_data = {
                'games_player1': min(game // 2, 6),
                'games_player2': min((game + 1) // 2, 6),
                'aces': np.random.poisson(0.5),
                'double_faults': np.random.poisson(0.2),
                'first_serve_percentage': np.random.normal(65, 10),
                'first_serve_points_won': np.random.normal(70, 10),
                'second_serve_points_won': np.random.normal(50, 15),
                'return_points_won': np.random.normal(35, 10),
                'total_points_won': np.random.randint(3, 8),
                'server_id': (game % 2) + 1,
                'game_winner': np.random.choice([1, 2], p=[0.6, 0.4])
            }
            games_sequence.append(game_data)
        
        # Create match sequence
        match_seq = MatchSequence(
            match_id=f"match_{i}",
            player1_name=f"Player1_{i}",
            player2_name=f"Player2_{i}",
            player1_ranking=np.random.randint(10, 100),
            player2_ranking=np.random.randint(50, 300),
            points_sequence=[],  # Empty for this example
            games_sequence=games_sequence,
            sets_sequence=[{'set_score_player1': 6, 'set_score_player2': 4}],
            surface=np.random.choice(['Hard', 'Clay', 'Grass']),
            tournament='ATP 250',
            round=np.random.choice(['R32', 'R16', 'QF']),
            is_indoor=np.random.choice([True, False]),
            second_set_winner=np.random.choice([1, 2])  # Random target for testing
        )
        
        sample_sequences.append(match_seq)
    
    # Create and train model
    config = {
        'max_sequence_length': 20,  # Shorter for testing
        'lstm_units': [32, 16],     # Smaller for testing
        'epochs': 10,               # Fewer epochs for testing
        'batch_size': 16
    }
    
    lstm_model = TennisLSTMModel(config)
    
    # Split data
    train_sequences = sample_sequences[:80]
    test_sequences = sample_sequences[80:]
    
    # Train
    print("Training LSTM model...")
    training_result = lstm_model.fit(train_sequences)
    print(f"Training completed: {training_result['final_val_accuracy']:.4f} accuracy")
    
    # Evaluate
    print("Evaluating model...")
    metrics = lstm_model.evaluate(test_sequences)
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Test live prediction
    live_match = test_sequences[0]
    live_match.second_set_winner = None  # Remove target for live prediction
    
    prediction = lstm_model.predict_live_match(live_match)
    print(f"\nLive prediction example:")
    print(f"  Underdog: {prediction['underdog_player_name']}")
    print(f"  Win probability: {prediction['underdog_win_probability']:.3f}")
    print(f"  Confidence: {prediction['confidence_level']}")