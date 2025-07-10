#!/usr/bin/env python3
"""
🎾 БЫСТРАЯ ПРОВЕРКА ТЕННИСНОГО ПРОЕКТА
Проверяет ключевые компоненты без сложных зависимостей
"""

import os
import sys
import json
from datetime import datetime

def quick_test():
    print("🎾 БЫСТРАЯ ПРОВЕРКА ТЕННИСНОГО ПРОЕКТА")
    print("=" * 50)
    print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    results = {}
    
    # 1. Проверка файловой структуры
    print("\n📁 ПРОВЕРКА ФАЙЛОВ:")
    critical_files = {
        'tennis_prediction_module.py': 'ML модели',
        'real_tennis_predictor_integration.py': 'Реальные данные', 
        'tennis_backend.py': 'Underdog система',
        'api_economy_patch.py': 'API экономия',
        'config.json': 'Конфигурация'
    }
    
    files_found = 0
    for file_path, description in critical_files.items():
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {description}: {file_path} ({size} bytes)")
            files_found += 1
        else:
            print(f"❌ {description}: {file_path} - НЕ НАЙДЕН")
    
    results['files'] = {'found': files_found, 'total': len(critical_files)}
    
    # 2. Проверка директории моделей
    print(f"\n🤖 ПРОВЕРКА ML МОДЕЛЕЙ:")
    models_dir = "tennis_models"
    if os.path.exists(models_dir):
        model_files = os.listdir(models_dir)
        print(f"✅ Директория {models_dir}: {len(model_files)} файлов")
        
        expected_models = ['neural_network.h5', 'xgboost.pkl', 'random_forest.pkl', 
                          'gradient_boosting.pkl', 'logistic_regression.pkl', 'scaler.pkl']
        
        found_models = 0
        for model in expected_models:
            if model in model_files:
                print(f"  ✅ {model}")
                found_models += 1
            else:
                print(f"  ❌ {model}")
        
        results['models'] = {'found': found_models, 'total': len(expected_models)}
    else:
        print(f"❌ Директория {models_dir} не найдена")
        results['models'] = {'found': 0, 'total': 6}
    
    # 3. Тест импорта модулей
    print(f"\n🔧 ТЕСТ ИМПОРТА МОДУЛЕЙ:")
    modules = {
        'tennis_prediction_module': 'Основной ML сервис',
        'real_tennis_predictor_integration': 'Интеграция реальных данных',
        'api_economy_patch': 'API экономия'
    }
    
    imported_modules = 0
    for module_name, description in modules.items():
        try:
            exec(f"import {module_name}")
            print(f"✅ {description}: {module_name}")
            imported_modules += 1
        except ImportError as e:
            print(f"❌ {description}: {module_name} - ОШИБКА ИМПОРТА")
        except Exception as e:
            print(f"⚠️ {description}: {module_name} - {type(e).__name__}")
    
    results['imports'] = {'success': imported_modules, 'total': len(modules)}
    
    # 4. Проверка конфигурации
    print(f"\n⚙️ ПРОВЕРКА КОНФИГУРАЦИИ:")
    if os.path.exists('config.json'):
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            
            api_key = config.get('data_sources', {}).get('the_odds_api', {}).get('api_key', '')
            has_real_key = api_key and api_key != 'YOUR_API_KEY' and len(api_key) > 10
            
            print(f"✅ config.json загружен")
            print(f"🔑 API ключ: {'✅ Настроен' if has_real_key else '⚠️ Тестовый/пустой'}")
            
            results['config'] = {'loaded': True, 'has_api_key': has_real_key}
        except Exception as e:
            print(f"❌ Ошибка чтения config.json: {e}")
            results['config'] = {'loaded': False, 'has_api_key': False}
    else:
        print(f"❌ config.json не найден")
        results['config'] = {'loaded': False, 'has_api_key': False}
    
    # 5. Быстрый тест ML функциональности
    print(f"\n🧠 БЫСТРЫЙ ТЕСТ ML:")
    try:
        import tennis_prediction_module as tpm
        
        # Создаем сервис
        service = tpm.TennisPredictionService()
        print(f"✅ TennisPredictionService создан")
        
        # Пробуем загрузить модели
        models_loaded = service.load_models()
        print(f"📂 Загрузка моделей: {'✅ Успешно' if models_loaded else '⚠️ Демо режим'}")
        
        # Тестовый прогноз
        test_data = tpm.create_match_data(
            player_rank=1, opponent_rank=45, 
            player_recent_win_rate=0.85
        )
        
        prediction = service.predict_match(test_data)
        prob = prediction['probability']
        conf = prediction['confidence']
        
        print(f"🎯 Тестовый прогноз: {prob:.1%} ({conf})")
        print(f"✅ ML система работает!")
        
        results['ml_test'] = {
            'success': True, 
            'models_loaded': models_loaded,
            'prediction': {'probability': prob, 'confidence': conf}
        }
        
    except Exception as e:
        print(f"❌ ML тест неудачен: {e}")
        results['ml_test'] = {'success': False, 'error': str(e)}
    
    # 6. Тест underdog системы
    print(f"\n🎯 ТЕСТ UNDERDOG СИСТЕМЫ:")
    try:
        import tennis_backend as tb
        
        # Создаем анализатор
        analyzer = tb.UnderdogAnalyzer()
        print(f"✅ UnderdogAnalyzer создан")
        
        # Тест определения рейтингов
        alcaraz_rank = analyzer.get_player_ranking('Carlos Alcaraz')
        nakashima_rank = analyzer.get_player_ranking('Brandon Nakashima')
        
        print(f"📊 Рейтинги: Alcaraz #{alcaraz_rank}, Nakashima #{nakashima_rank}")
        
        # Тест underdog сценария
        scenario = analyzer.identify_underdog_scenario('Brandon Nakashima', 'Carlos Alcaraz')
        
        print(f"🎯 Underdog сценарий:")
        print(f"   Underdog: {scenario['underdog']} (#{scenario['underdog_rank']})")
        print(f"   Favorite: {scenario['favorite']} (#{scenario['favorite_rank']})")
        print(f"   Тип: {scenario['underdog_type']}")
        
        # Полный расчет
        analysis = analyzer.calculate_underdog_probability(
            'Brandon Nakashima', 'Carlos Alcaraz', 'Wimbledon', 'Grass'
        )
        
        prob = analysis['underdog_probability']
        quality = analysis['quality']
        ml_system = analysis['ml_system_used']
        
        print(f"🎲 Финальный анализ: {prob:.1%} ({quality})")
        print(f"🤖 ML система: {ml_system}")
        print(f"✅ Underdog система работает!")
        
        results['underdog_test'] = {
            'success': True,
            'uses_ml': ml_system != 'None',
            'analysis': {'probability': prob, 'quality': quality, 'ml_system': ml_system}
        }
        
    except Exception as e:
        print(f"❌ Underdog тест неудачен: {e}")
        results['underdog_test'] = {'success': False, 'error': str(e)}
    
    # 7. Итоговая оценка
    print(f"\n" + "=" * 50)
    print(f"📊 ИТОГОВАЯ ОЦЕНКА")
    print("=" * 50)
    
    # Подсчет успешности
    scores = []
    
    # Файлы (30%)
    file_score = (results['files']['found'] / results['files']['total']) * 30
    scores.append(file_score)
    print(f"📁 Файлы: {results['files']['found']}/{results['files']['total']} = {file_score:.1f}/30")
    
    # Модели (20%) 
    model_score = (results['models']['found'] / results['models']['total']) * 20
    scores.append(model_score)
    print(f"🤖 ML модели: {results['models']['found']}/{results['models']['total']} = {model_score:.1f}/20")
    
    # Импорты (20%)
    import_score = (results['imports']['success'] / results['imports']['total']) * 20
    scores.append(import_score)
    print(f"🔧 Импорты: {results['imports']['success']}/{results['imports']['total']} = {import_score:.1f}/20")
    
    # ML тест (15%)
    ml_score = 15 if results['ml_test']['success'] else 0
    scores.append(ml_score)
    print(f"🧠 ML тест: {'Успех' if results['ml_test']['success'] else 'Неудача'} = {ml_score}/15")
    
    # Underdog тест (15%)
    underdog_score = 15 if results['underdog_test']['success'] else 0
    scores.append(underdog_score)
    print(f"🎯 Underdog тест: {'Успех' if results['underdog_test']['success'] else 'Неудача'} = {underdog_score}/15")
    
    total_score = sum(scores)
    print(f"\n🎯 ОБЩИЙ СЧЕТ: {total_score:.1f}/100")
    
    # Выводы
    print(f"\n🔍 КЛЮЧЕВЫЕ ВЫВОДЫ:")
    
    if results['ml_test']['success'] and results['underdog_test']['success']:
        underdog_uses_ml = results['underdog_test']['analysis']['ml_system'] != 'None'
        if underdog_uses_ml:
            print(f"✅ ОТЛИЧНО! Underdog система использует РЕАЛЬНЫЕ ML модели")
            print(f"✅ Анализ базируется на комплексных алгоритмах, НЕ 'от балды'")
        else:
            print(f"⚠️ Underdog система работает, но использует базовую логику")
            print(f"💡 Нужна интеграция с обученными ML моделями")
    else:
        print(f"❌ Есть проблемы с ML или Underdog системой")
    
    if results['models']['found'] >= 5:
        print(f"✅ ML модели присутствуют и могут использоваться")
    else:
        print(f"⚠️ Мало ML моделей - нужно обучить и сохранить")
    
    if total_score >= 80:
        print(f"🎉 СИСТЕМА В ОТЛИЧНОМ СОСТОЯНИИ!")
    elif total_score >= 60:
        print(f"✅ Система работает хорошо, есть места для улучшения")
    else:
        print(f"⚠️ Требуется доработка ключевых компонентов")
    
    # Сохраняем результаты
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_score': total_score,
        'detailed_results': results,
        'conclusion': {
            'ml_working': results['ml_test']['success'],
            'underdog_working': results['underdog_test']['success'],
            'underdog_uses_ml': results['underdog_test']['success'] and results['underdog_test']['analysis']['ml_system'] != 'None',
            'models_available': results['models']['found'] >= 5
        }
    }
    
    filename = f"quick_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Отчет сохранен: {filename}")
    except Exception as e:
        print(f"\n⚠️ Не удалось сохранить отчет: {e}")
    
    return report

if __name__ == "__main__":
    report = quick_test()
    
    print(f"\n🎾 ПРОВЕРКА ЗАВЕРШЕНА")
    print(f"📊 Итоговый счет: {report['total_score']:.1f}/100")
    
    if report['conclusion']['underdog_uses_ml']:
        print(f"🎯 ✅ ГЛАВНЫЙ РЕЗУЛЬТАТ: Underdog система ДЕЙСТВИТЕЛЬНО использует ML!")
    else:
        print(f"🎯 ⚠️ ГЛАВНЫЙ РЕЗУЛЬТАТ: Underdog система нуждается в ML интеграции")