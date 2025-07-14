#!/usr/bin/env python3
"""
🎾 МОДУЛЬ ПРОГНОЗИРОВАНИЯ ДЛЯ ИНТЕГРАЦИИ
Готовый модуль для внедрения в ваш проект
"""

import pandas as pd
import numpy as np
import joblib
import os
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

try:
    from adaptive_ensemble_optimizer import (
        get_ensemble_optimizer, get_optimized_weights, 
        record_model_predictions, init_ensemble_optimizer
    )
    ADAPTIVE_OPTIMIZER_AVAILABLE = True
except ImportError:
    ADAPTIVE_OPTIMIZER_AVAILABLE = False

class TennisPredictionService:
    """
    Сервис прогнозирования теннисных матчей
    Использует ваши обученные модели
    """
    
    def __init__(self, models_dir="tennis_models", use_adaptive_weights=True):
        self.models_dir = models_dir
        self.models = {}
        self.scaler = None
        self.expected_features = []
        self.use_adaptive_weights = use_adaptive_weights and ADAPTIVE_OPTIMIZER_AVAILABLE
        
        # Load base weights from metadata if available
        self.base_weights = self._load_base_weights()
        
        # Current ensemble weights (will be updated by adaptive optimizer)
        self.ensemble_weights = self.base_weights.copy()
        
        self.is_loaded = False
        
        # Initialize adaptive optimizer if available
        if self.use_adaptive_weights:
            try:
                if not get_ensemble_optimizer():
                    init_ensemble_optimizer(self.base_weights)
                print("🧠 Adaptive ensemble optimization enabled")
            except Exception as e:
                print(f"⚠️ Could not initialize adaptive optimizer: {e}")
                self.use_adaptive_weights = False
    
    def _load_base_weights(self):
        """Load base ensemble weights from metadata file"""
        try:
            metadata_path = os.path.join(self.models_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    weights = metadata.get('ensemble_weights', {})
                    print(f"📊 Loaded base weights from metadata: {weights}")
                    return weights
        except Exception as e:
            print(f"⚠️ Could not load weights from metadata: {e}")
        
        # Fallback to hardcoded weights
        return {
            'neural_network': 0.2054,
            'xgboost': 0.2027,
            'random_forest': 0.1937,
            'gradient_boosting': 0.1916,
            'logistic_regression': 0.2065
        }
    
    def _get_current_weights(self, match_context=None):
        """Get current weights (adaptive or base)"""
        if self.use_adaptive_weights:
            try:
                adaptive_weights = get_optimized_weights(match_context)
                if adaptive_weights:
                    return adaptive_weights
            except Exception as e:
                print(f"⚠️ Error getting adaptive weights: {e}")
        
        return self.ensemble_weights
        
    def load_models(self):
        """Загрузка всех моделей"""
        try:
            print("📂 Загружаем модели прогнозирования...")
            
            # Загружаем скалер
            scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Скалер не найден: {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            
            # Загружаем модели
            model_files = {
                'neural_network': 'neural_network.h5',
                'xgboost': 'xgboost.pkl', 
                'random_forest': 'random_forest.pkl',
                'gradient_boosting': 'gradient_boosting.pkl',
                'logistic_regression': 'logistic_regression.pkl'
            }
            
            loaded_count = 0
            for model_name, filename in model_files.items():
                filepath = os.path.join(self.models_dir, filename)
                if os.path.exists(filepath):
                    try:
                        if model_name == 'neural_network':
                            self.models[model_name] = keras.models.load_model(filepath)
                        else:
                            self.models[model_name] = joblib.load(filepath)
                        loaded_count += 1
                    except Exception as e:
                        print(f"⚠️ Не удалось загрузить {model_name}: {e}")
                else:
                    print(f"⚠️ Файл не найден: {filepath}")
            
            if loaded_count == 0:
                raise Exception("Ни одна модель не загружена")
            
            # Определяем ожидаемые признаки
            if 'random_forest' in self.models:
                self.expected_features = list(self.models['random_forest'].feature_names_in_)
            elif 'xgboost' in self.models:
                self.expected_features = list(self.models['xgboost'].feature_names_in_)
            else:
                # Fallback к стандартному списку
                self.expected_features = [
                    'player_rank', 'player_age', 'opponent_rank', 'opponent_age',
                    'player_recent_win_rate', 'player_form_trend', 'player_surface_advantage',
                    'h2h_win_rate', 'total_pressure', 'rank_difference', 'rank_ratio',
                    'combined_form', 'surface_pressure_interaction'
                ]
            
            self.is_loaded = True
            print(f"✅ Загружено {loaded_count} моделей")
            print(f"🔧 Ожидается {len(self.expected_features)} признаков")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка загрузки моделей: {e}")
            return False
    
    def create_engineered_features(self, match_data):
        """Создание инженерных признаков"""
        enhanced_data = match_data.copy()
        
        # 1. Разность и соотношение рейтингов
        enhanced_data['rank_difference'] = enhanced_data['opponent_rank'] - enhanced_data['player_rank']
        enhanced_data['rank_ratio'] = enhanced_data['player_rank'] / (enhanced_data['opponent_rank'] + 1)
        
        # 2. Комбинированная форма
        enhanced_data['combined_form'] = (enhanced_data['player_recent_win_rate'] * 0.7 + 
                                        enhanced_data['h2h_win_rate'] * 0.3)
        
        # 3. Взаимодействие покрытия и давления
        enhanced_data['surface_pressure_interaction'] = (enhanced_data['player_surface_advantage'] * 
                                                       enhanced_data['total_pressure'])
        
        return enhanced_data
    
    def validate_input_data(self, match_data):
        """Валидация входных данных"""
        required_fields = [
            'player_rank', 'player_age', 'opponent_rank', 'opponent_age',
            'player_recent_win_rate', 'player_form_trend',
            'player_surface_advantage', 'h2h_win_rate', 'total_pressure'
        ]
        
        missing_fields = [field for field in required_fields if field not in match_data]
        if missing_fields:
            raise ValueError(f"Отсутствующие поля: {missing_fields}")
        
        # Проверка диапазонов
        if not (1 <= match_data['player_rank'] <= 1000):
            raise ValueError("player_rank должен быть от 1 до 1000")
        if not (1 <= match_data['opponent_rank'] <= 1000):
            raise ValueError("opponent_rank должен быть от 1 до 1000")
        if not (0 <= match_data['player_recent_win_rate'] <= 1):
            raise ValueError("player_recent_win_rate должен быть от 0 до 1")
        
        return True
    
    def predict_match(self, match_data, return_details=True):
        """
        Основная функция прогнозирования
        
        Args:
            match_data (dict): Данные матча с 9 обязательными полями
            return_details (bool): Возвращать ли детальную информацию
        
        Returns:
            dict: Результат прогнозирования
        """
        
        if not self.is_loaded:
            if not self.load_models():
                raise Exception("Модели не загружены")
        
        # Валидация
        self.validate_input_data(match_data)
        
        try:
            # Создание инженерных признаков
            enhanced_data = self.create_engineered_features(match_data)
            
            # Создание DataFrame с нужными признаками
            features_df = pd.DataFrame([enhanced_data])
            features_df = features_df[self.expected_features]
            
            # Заполнение пропусков
            features_df = features_df.fillna(features_df.median())
            
            # Нормализация для нейросети и логистической регрессии
            X_scaled = self.scaler.transform(features_df)
            
            # Prepare match context for adaptive weights
            match_context = {
                'surface': match_data.get('surface', ''),
                'tournament_importance': enhanced_data.get('total_pressure', 0),
                'rank_difference': enhanced_data.get('rank_difference', 0),
                'match_info': {
                    'player_rank': match_data.get('player_rank'),
                    'opponent_rank': match_data.get('opponent_rank'),
                    'h2h_win_rate': match_data.get('h2h_win_rate')
                }
            }
            
            # Get current weights (adaptive or base)
            current_weights = self._get_current_weights(match_context)
            
            # Прогнозы от каждой модели
            individual_predictions = {}
            
            for model_name, model in self.models.items():
                try:
                    if model_name == 'neural_network':
                        pred = float(model.predict(X_scaled, verbose=0)[0][0])
                    elif model_name == 'logistic_regression':
                        pred = float(model.predict_proba(X_scaled)[0, 1])
                    else:
                        pred = float(model.predict_proba(features_df)[0, 1])
                    
                    individual_predictions[model_name] = pred
                    
                except Exception as e:
                    print(f"⚠️ Ошибка модели {model_name}: {e}")
                    individual_predictions[model_name] = 0.5
            
            # Ансамблевый прогноз с адаптивными весами
            ensemble_pred = sum(pred * current_weights.get(name, 0.2) 
                               for name, pred in individual_predictions.items())
            total_weight = sum(current_weights.get(name, 0.2) 
                              for name in individual_predictions.keys())
            ensemble_pred /= total_weight if total_weight > 0 else 1
            
            # Определение уверенности
            if ensemble_pred >= 0.7:
                confidence = "High"
                confidence_ru = "Высокая"
            elif ensemble_pred >= 0.55:
                confidence = "Medium" 
                confidence_ru = "Средняя"
            else:
                confidence = "Low"
                confidence_ru = "Низкая"
            
            # Базовый результат
            result = {
                'probability': round(ensemble_pred, 4),
                'confidence': confidence,
                'confidence_ru': confidence_ru,
                'recommendation': self._get_recommendation(ensemble_pred, match_data)
            }
            
            # Детальная информация
            if return_details:
                result.update({
                    'individual_predictions': {
                        name: round(pred, 4) for name, pred in individual_predictions.items()
                    },
                    'key_factors': self._analyze_key_factors(match_data, ensemble_pred),
                    'model_weights': self.ensemble_weights,
                    'input_data': match_data
                })
            
            return result
            
        except Exception as e:
            raise Exception(f"Ошибка прогнозирования: {e}")
    
    def _get_recommendation(self, probability, match_data):
        """Генерация рекомендации"""
        if probability >= 0.7:
            return "Strong recommendation: High probability of winning at least one set"
        elif probability >= 0.55:
            return "Moderate recommendation: Consider additional factors"
        else:
            return "Caution: Low probability, possible underdog scenario"
    
    def _analyze_key_factors(self, match_data, prediction):
        """Анализ ключевых факторов"""
        factors = []
        
        if match_data['player_recent_win_rate'] > 0.8:
            factors.append("Excellent current form")
        
        rank_diff = match_data['opponent_rank'] - match_data['player_rank']
        if rank_diff > 20:
            factors.append(f"Significant ranking advantage (+{rank_diff})")
        elif rank_diff < -10:
            factors.append(f"Playing against higher ranked opponent ({rank_diff})")
        
        if match_data['player_surface_advantage'] > 0.1:
            factors.append("Strong surface advantage")
        elif match_data['player_surface_advantage'] < -0.05:
            factors.append("Surface disadvantage")
        
        if match_data['h2h_win_rate'] > 0.7:
            factors.append("Dominates head-to-head")
        elif match_data['h2h_win_rate'] < 0.4:
            factors.append("Poor head-to-head record")
        
        if match_data['player_form_trend'] > 0.1:
            factors.append("Rising form trend")
        elif match_data['player_form_trend'] < -0.1:
            factors.append("Declining form trend")
        
        return factors
    
    def predict_multiple_matches(self, matches_list):
        """Прогнозирование нескольких матчей"""
        results = []
        
        for i, match_data in enumerate(matches_list):
            try:
                prediction = self.predict_match(match_data, return_details=False)
                prediction['match_id'] = i
                results.append(prediction)
            except Exception as e:
                results.append({
                    'match_id': i,
                    'error': str(e),
                    'probability': None
                })
        
        return results
    
    def get_model_info(self):
        """Информация о загруженных моделях"""
        if not self.is_loaded:
            return {"status": "Models not loaded"}
        
        return {
            "status": "loaded",
            "models_count": len(self.models),
            "models": list(self.models.keys()),
            "expected_features": self.expected_features,
            "ensemble_weights": self.ensemble_weights,
            "adaptive_weights_enabled": self.use_adaptive_weights
        }
    
    def record_match_result(self, match_data, predictions_result, actual_result, match_info=None):
        """
        Record actual match result for adaptive weight optimization
        
        Args:
            match_data: Original match data used for prediction
            predictions_result: Result from predict_match() containing individual predictions
            actual_result: Actual match outcome (0 or 1)
            match_info: Additional match information
        """
        if not self.use_adaptive_weights:
            return False
        
        try:
            individual_predictions = predictions_result.get('individual_predictions', {})
            if individual_predictions and ADAPTIVE_OPTIMIZER_AVAILABLE:
                record_model_predictions(
                    individual_predictions, 
                    actual_result, 
                    match_info or match_data
                )
                print(f"📊 Recorded match result for adaptive optimization")
                return True
        except Exception as e:
            print(f"⚠️ Error recording match result: {e}")
        
        return False
    
    def get_ensemble_performance_report(self):
        """Get detailed ensemble performance report"""
        if not self.use_adaptive_weights or not ADAPTIVE_OPTIMIZER_AVAILABLE:
            return {
                'error': 'Adaptive ensemble optimization not available',
                'current_weights': self.ensemble_weights,
                'base_weights': self.base_weights
            }
        
        try:
            from adaptive_ensemble_optimizer import get_ensemble_performance_report
            return get_ensemble_performance_report()
        except Exception as e:
            return {'error': f'Could not get performance report: {e}'}
    
    def update_ensemble_weights(self, context=None):
        """Manually update ensemble weights based on current performance"""
        if not self.use_adaptive_weights:
            return False
        
        try:
            new_weights = self._get_current_weights(context)
            if new_weights:
                self.ensemble_weights = new_weights
                print(f"🔄 Updated ensemble weights: {new_weights}")
                return True
        except Exception as e:
            print(f"⚠️ Error updating weights: {e}")
        
        return False


# Функции для удобной интеграции

def create_match_data(player_rank, opponent_rank, player_age=25, opponent_age=25,
                     player_recent_win_rate=0.7, player_form_trend=0.0,
                     player_surface_advantage=0.0, h2h_win_rate=0.5, total_pressure=2.5):
    """
    Вспомогательная функция для создания данных матча
    
    Args:
        player_rank (int): Рейтинг игрока
        opponent_rank (int): Рейтинг соперника
        player_age (int): Возраст игрока
        opponent_age (int): Возраст соперника
        player_recent_win_rate (float): Винрейт в последних матчах (0-1)
        player_form_trend (float): Тренд формы (-0.5 до 0.5)
        player_surface_advantage (float): Преимущество на покрытии (-0.3 до 0.3)
        h2h_win_rate (float): Винрейт в очных встречах (0-1)
        total_pressure (float): Давление турнира (1-5)
    
    Returns:
        dict: Данные матча готовые для прогнозирования
    """
    return {
        'player_rank': float(player_rank),
        'opponent_rank': float(opponent_rank),
        'player_age': float(player_age),
        'opponent_age': float(opponent_age),
        'player_recent_win_rate': float(player_recent_win_rate),
        'player_form_trend': float(player_form_trend),
        'player_surface_advantage': float(player_surface_advantage),
        'h2h_win_rate': float(h2h_win_rate),
        'total_pressure': float(total_pressure)
    }

def quick_predict(player_rank, opponent_rank, models_dir="tennis_models", **kwargs):
    """
    Быстрый прогноз без создания объекта сервиса
    
    Args:
        player_rank (int): Рейтинг игрока
        opponent_rank (int): Рейтинг соперника
        models_dir (str): Папка с моделями
        **kwargs: Дополнительные параметры матча
    
    Returns:
        dict: Результат прогнозирования
    """
    service = TennisPredictionService(models_dir)
    match_data = create_match_data(player_rank, opponent_rank, **kwargs)
    return service.predict_match(match_data)


# Пример использования
if __name__ == "__main__":
    print("🎾 ТЕСТИРОВАНИЕ МОДУЛЯ ПРОГНОЗИРОВАНИЯ")
    print("=" * 50)
    
    # Пример 1: Быстрый прогноз
    print("📊 Быстрый прогноз:")
    result = quick_predict(
        player_rank=1, 
        opponent_rank=45,
        player_recent_win_rate=0.85,
        player_surface_advantage=0.12,
        h2h_win_rate=0.75
    )
    print(f"Вероятность: {result['probability']:.1%}")
    print(f"Уверенность: {result['confidence_ru']}")
    
    # Пример 2: Детальное использование
    print(f"\n📊 Детальный анализ:")
    service = TennisPredictionService()
    
    match_data = create_match_data(
        player_rank=5,
        opponent_rank=6,
        player_recent_win_rate=0.72,
        player_form_trend=0.02,
        h2h_win_rate=0.58,
        total_pressure=3.8
    )
    
    result = service.predict_match(match_data)
    
    print(f"Прогноз: {result['probability']:.1%}")
    print(f"Рекомендация: {result['recommendation']}")
    print(f"Ключевые факторы: {result['key_factors']}")
    
    # Пример 3: Несколько матчей
    print(f"\n📊 Прогноз нескольких матчей:")
    matches = [
        create_match_data(1, 45, player_recent_win_rate=0.85),
        create_match_data(5, 6, player_recent_win_rate=0.72),
        create_match_data(35, 8, player_recent_win_rate=0.88, player_surface_advantage=0.18)
    ]
    
    results = service.predict_multiple_matches(matches)
    for i, result in enumerate(results, 1):
        if 'probability' in result and result['probability'] is not None:
            print(f"Матч {i}: {result['probability']:.1%} ({result['confidence_ru']})")
        else:
            print(f"Матч {i}: Ошибка - {result.get('error', 'Unknown error')}")
    
    print("\n✅ Модуль готов к интеграции!")