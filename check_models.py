#!/usr/bin/env python3
"""
🔍 Проверка признаков, на которых были обучены модели
"""

import json
import os
from tennis_set_predictor import EnhancedTennisPredictor

def check_model_features():
    """Проверяем какие признаки использовались при обучении"""
    
    print("🔍 ПРОВЕРКА ПРИЗНАКОВ ОБУЧЕННЫХ МОДЕЛЕЙ")
    print("=" * 60)
    
    try:
        # Проверяем метаданные моделей
        metadata_path = os.path.join("tennis_models", "metadata.json")
        
        if os.path.exists(metadata_path):
            print("📄 Загружаем метаданные моделей...")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            feature_columns = metadata.get('feature_columns', [])
            ensemble_weights = metadata.get('ensemble_weights', {})
            validation_scores = metadata.get('validation_scores', {})
            
            print(f"✅ Найдено {len(feature_columns)} признаков:")
            print("-" * 40)
            
            for i, feature in enumerate(feature_columns, 1):
                print(f"{i:2d}. {feature}")
            
            print(f"\n📊 Веса моделей в ансамбле:")
            print("-" * 40)
            for model, weight in ensemble_weights.items():
                score = validation_scores.get(model, 'N/A')
                print(f"• {model:20}: {weight:.3f} (AUC: {score})")
            
            return feature_columns
            
        else:
            print("❌ Файл metadata.json не найден")
            
            # Попробуем загрузить модель и посмотреть на ее ожидания
            print("🔄 Пытаемся загрузить модель напрямую...")
            
            predictor = EnhancedTennisPredictor(model_dir="tennis_models")
            predictor.load_models()
            
            if hasattr(predictor, 'feature_columns'):
                print(f"✅ Признаки из загруженной модели:")
                for i, feature in enumerate(predictor.feature_columns, 1):
                    print(f"{i:2d}. {feature}")
                return predictor.feature_columns
            else:
                print("❌ Не удалось определить признаки")
                return []
                
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return []

def create_correct_demo_data(feature_columns):
    """Создаем демо данные с правильными признаками"""
    
    if not feature_columns:
        print("❌ Не удалось получить список признаков")
        return None
    
    print(f"\n🎯 СОЗДАНИЕ ДАННЫХ С ПРАВИЛЬНЫМИ ПРИЗНАКАМИ")
    print("=" * 60)
    
    # Базовый набор данных (на основе синтетических данных из обучения)
    base_data = {
        'player_rank': 5,
        'opponent_rank': 45,  
        'player_age': 25,
        'opponent_age': 28,
        'player_recent_win_rate': 0.85,
        'player_surface_advantage': 0.12,
        'h2h_win_rate': 0.75,
        'total_pressure': 3.2,
        'player_form_trend': 0.08
    }
    
    # Создаем полный набор данных с правильными признаками
    match_data = {}
    
    for feature in feature_columns:
        if feature in base_data:
            match_data[feature] = base_data[feature]
        else:
            # Генерируем реалистичные значения для отсутствующих признаков
            if 'rank' in feature:
                match_data[feature] = 25  # средний рейтинг
            elif 'age' in feature:
                match_data[feature] = 26  # средний возраст
            elif 'win_rate' in feature or 'advantage' in feature:
                match_data[feature] = 0.6  # средний винрейт
            elif 'pressure' in feature:
                match_data[feature] = 2.5  # среднее давление
            elif 'trend' in feature:
                match_data[feature] = 0.05  # небольшой тренд
            elif 'count' in feature:
                match_data[feature] = 10  # количество матчей
            elif 'days' in feature:
                match_data[feature] = 30  # дни
            else:
                match_data[feature] = 0.5  # значение по умолчанию
    
    print(f"✅ Создан набор данных с {len(match_data)} признаками")
    
    # Показываем какие данные мы создали
    print(f"\n📋 Данные для прогноза:")
    print("-" * 40)
    for feature, value in match_data.items():
        print(f"• {feature:30}: {value}")
    
    return match_data

def test_prediction_with_correct_data(feature_columns):
    """Тестируем прогнозирование с правильными данными"""
    
    print(f"\n🎾 ТЕСТИРОВАНИЕ ПРОГНОЗИРОВАНИЯ")
    print("=" * 60)
    
    # Создаем правильные данные
    match_data = create_correct_demo_data(feature_columns)
    
    if not match_data:
        return
    
    try:
        # Загружаем модель
        predictor = EnhancedTennisPredictor(model_dir="tennis_models")
        predictor.load_models()
        
        # Создаем DataFrame
        import pandas as pd
        match_df = pd.DataFrame([match_data])
        
        print(f"🔄 Подготовка признаков...")
        
        # Подготавливаем признаки
        match_features = predictor.prepare_features(match_df)
        print(f"✅ Признаки подготовлены: {match_features.shape}")
        
        # Делаем прогноз
        print(f"🔮 Делаем прогноз...")
        prediction = predictor.predict_probability(match_features)[0]
        
        print(f"\n🎯 РЕЗУЛЬТАТ ПРОГНОЗА:")
        print("=" * 40)
        print(f"Вероятность выиграть хотя бы один сет: {prediction:.1%}")
        
        if prediction >= 0.7:
            confidence = "Высокая 🔥"
        elif prediction >= 0.55:
            confidence = "Средняя ⚡"
        else:
            confidence = "Низкая 💭"
        
        print(f"Уровень уверенности: {confidence}")
        
        # Прогнозы отдельных моделей
        print(f"\n📊 ПРОГНОЗЫ ОТДЕЛЬНЫХ МОДЕЛЕЙ:")
        print("-" * 40)
        
        X_scaled = predictor.scaler.transform(match_features)
        
        for model_name, model in predictor.models.items():
            try:
                if model_name in ['neural_network', 'logistic_regression']:
                    pred = float(model.predict(X_scaled)[0])
                else:
                    pred = float(model.predict_proba(match_features)[0, 1])
                
                weight = predictor.ensemble_weights.get(model_name, 0)
                icons = "🔥" if pred > 0.7 else "⚡" if pred > 0.55 else "💭"
                print(f"{icons} {model_name:20}: {pred:.1%} (вес: {weight:.3f})")
                
            except Exception as e:
                print(f"❌ {model_name:20}: Ошибка - {e}")
        
        print(f"\n✅ ТЕСТ ПРОШЕЛ УСПЕШНО!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        return False

def main():
    print("🔍 ДИАГНОСТИКА МОДЕЛЕЙ И ПРИЗНАКОВ")
    print("=" * 70)
    
    # Проверяем признаки
    feature_columns = check_model_features()
    
    if feature_columns:
        # Тестируем прогнозирование
        success = test_prediction_with_correct_data(feature_columns)
        
        if success:
            print(f"\n🎉 ВСЕ РАБОТАЕТ!")
            print("=" * 30)
            print("💡 Теперь вы знаете правильный формат данных")
            print("🚀 Можете использовать эти признаки для реальных прогнозов")
        else:
            print(f"\n❌ ЕСТЬ ПРОБЛЕМЫ")
            print("💡 Проверьте целостность файлов моделей")
    else:
        print(f"\n❌ НЕ УДАЛОСЬ ОПРЕДЕЛИТЬ ПРИЗНАКИ")
        print("💡 Возможно, нужно переобучить модели")

if __name__ == "__main__":
    main()