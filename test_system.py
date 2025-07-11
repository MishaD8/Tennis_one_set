#!/usr/bin/env python3
"""
🎾 ПОЛНЫЙ АНАЛИЗАТОР ТЕННИСНОЙ СИСТЕМЫ
Проверяет все компоненты и выдает детальный отчет о работе
Включает тестирование реальных данных, ML моделей и накопления опыта
"""

import os
import json
import sys
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import traceback
import importlib.util
import numpy as np

class ComprehensiveTennisSystemAnalyzer:
    """Полный анализатор теннисной системы"""
    
    def __init__(self):
        self.report = {
            'analysis_time': datetime.now().isoformat(),
            'system_components': {},
            'ml_performance': {},
            'data_integration': {},
            'learning_capability': {},
            'real_data_flow': {},
            'overall_assessment': {}
        }
        self.test_results = []
        
    def analyze_file_structure(self) -> Dict:
        """Анализ структуры файлов системы"""
        print("📁 АНАЛИЗ СТРУКТУРЫ ФАЙЛОВ")
        print("=" * 50)
        
        critical_files = {
            'main_backend': 'tennis_backend.py',
            'ml_predictor': 'real_tennis_predictor_integration.py', 
            'prediction_service': 'tennis_prediction_module.py',
            'api_economy': 'api_economy_patch.py',
            'odds_integration': 'correct_odds_api_integration.py',
            'universal_collector': 'universal_tennis_data_collector.py',
            'advanced_ml': 'advanced_tennis_ml_predictor_fixed.py',
            'set_predictor': 'tennis_set_predictor.py',
            'logging_system': 'prediction_logging_system.py',
            'config': 'config.json',
            'models_dir': 'tennis_models/',
            'data_dir': 'tennis_data_enhanced/',
            'logs_dir': 'prediction_logs/',
            'cache_file': 'api_cache.json',
            'usage_file': 'api_usage.json'
        }
        
        file_status = {}
        total_files = len(critical_files)
        found_files = 0
        
        for component, filename in critical_files.items():
            exists = os.path.exists(filename)
            if exists:
                found_files += 1
                if os.path.isfile(filename):
                    size = os.path.getsize(filename)
                    status = f"✅ {size:,} bytes"
                elif os.path.isdir(filename):
                    try:
                        items = len(os.listdir(filename))
                        status = f"✅ {items} items"
                    except:
                        status = "✅ Directory (access denied)"
                else:
                    status = "✅ Exists"
            else:
                status = "❌ Missing"
            
            file_status[component] = {
                'path': filename,
                'exists': exists,
                'status': status
            }
            
            print(f"{component:20}: {status}")
        
        print(f"\n📊 Файлов найдено: {found_files}/{total_files} ({found_files/total_files*100:.1f}%)")
        
        return {
            'files': file_status,
            'completeness': found_files / total_files,
            'critical_missing': [comp for comp, info in file_status.items() if not info['exists']]
        }
    
    def test_imports_and_modules(self) -> Dict:
        """Тестирование импортов и доступности модулей"""
        print("\n🔧 ТЕСТИРОВАНИЕ ИМПОРТОВ И МОДУЛЕЙ")
        print("=" * 50)
        
        modules_to_test = {
            'real_tennis_predictor_integration': 'RealTennisPredictor',
            'tennis_prediction_module': 'TennisPredictionService', 
            'api_economy_patch': 'economical_tennis_request',
            'correct_odds_api_integration': 'TennisOddsIntegrator',
            'universal_tennis_data_collector': 'UniversalTennisDataCollector',
            'prediction_logging_system': 'CompletePredictionLogger',
            'tennis_set_predictor': 'EnhancedTennisPredictor'
        }
        
        import_results = {}
        successful_imports = 0
        
        for module_name, class_name in modules_to_test.items():
            try:
                # Пытаемся импортировать модуль
                module = importlib.import_module(module_name)
                
                # Проверяем наличие класса
                if hasattr(module, class_name):
                    # Пытаемся создать экземпляр
                    cls = getattr(module, class_name)
                    if class_name == 'economical_tennis_request':
                        # Это функция, не класс
                        status = "✅ Function available"
                        working = True
                        error = None
                    else:
                        instance = cls()
                        status = "✅ Working"
                        working = True
                        error = None
                    successful_imports += 1
                else:
                    status = f"⚠️ Class {class_name} not found"
                    working = False
                    error = f"Class {class_name} missing"
                    
            except Exception as e:
                status = f"❌ Error: {str(e)[:50]}..."
                working = False
                error = str(e)
            
            import_results[module_name] = {
                'status': status,
                'working': working,
                'class_name': class_name,
                'error': error
            }
            
            print(f"{module_name:35}: {status}")
        
        print(f"\n📊 Успешных импортов: {successful_imports}/{len(modules_to_test)}")
        
        return {
            'results': import_results,
            'success_rate': successful_imports / len(modules_to_test),
            'working_modules': [name for name, info in import_results.items() if info['working']]
        }
    
    def test_ml_models_comprehensive(self) -> Dict:
        """Комплексное тестирование ML моделей"""
        print("\n🤖 КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ ML МОДЕЛЕЙ")
        print("=" * 50)
        
        ml_results = {}
        
        # 1. Тестируем TennisPredictionService
        try:
            from tennis_prediction_module import TennisPredictionService
            
            print("🔸 Тестируем TennisPredictionService...")
            service = TennisPredictionService()
            
            if service.load_models():
                print("  ✅ Модели загружены успешно")
                
                # Тестовые данные для прогноза
                test_data = {
                    'player_rank': 5.0,
                    'opponent_rank': 1.0,
                    'player_age': 27.0,
                    'opponent_age': 23.0,
                    'player_recent_win_rate': 0.75,
                    'player_form_trend': 0.05,
                    'player_surface_advantage': 0.02,
                    'h2h_win_rate': 0.6,
                    'total_pressure': 3.5
                }
                
                prediction = service.predict_match(test_data, return_details=True)
                
                print(f"  🎯 Тест прогноз: {prediction['probability']:.1%}")
                print(f"  🔧 Модели: {', '.join(prediction.get('individual_predictions', {}).keys())}")
                print(f"  📊 Признаков: {len(prediction.get('input_data', {}))}")
                
                ml_results['prediction_service'] = {
                    'available': True,
                    'models_loaded': True,
                    'models_count': len(prediction.get('individual_predictions', {})),
                    'test_probability': prediction['probability'],
                    'confidence': prediction['confidence']
                }
            else:
                print("  ❌ Модели не загружены")
                ml_results['prediction_service'] = {'available': False, 'error': 'Models not loaded'}
                
        except Exception as e:
            print(f"  ❌ Ошибка TennisPredictionService: {e}")
            ml_results['prediction_service'] = {'available': False, 'error': str(e)}
        
        # 2. Тестируем RealTennisPredictor
        try:
            from real_tennis_predictor_integration import RealTennisPredictor
            
            print("🔸 Тестируем RealTennisPredictor...")
            predictor = RealTennisPredictor()
            
            # Тестируем реальный прогноз
            result = predictor.predict_match(
                'Carlos Alcaraz', 'Novak Djokovic', 
                'US Open', 'Hard', 'SF'
            )
            
            print(f"  🎯 Тест прогноз: {result['probability']:.1%}")
            print(f"  🎪 Тип: {result['prediction_type']}")
            print(f"  🔍 Факторов: {len(result.get('key_factors', []))}")
            
            # Проверяем использует ли реальные ML модели
            uses_real_ml = hasattr(predictor, 'prediction_service') and predictor.prediction_service is not None
            
            ml_results['real_predictor'] = {
                'available': True,
                'uses_real_ml': uses_real_ml,
                'prediction_type': result['prediction_type'],
                'test_probability': result['probability'],
                'confidence': result['confidence']
            }
            
        except Exception as e:
            print(f"  ❌ Ошибка RealTennisPredictor: {e}")
            ml_results['real_predictor'] = {'available': False, 'error': str(e)}
        
        # 3. Тестируем EnhancedTennisPredictor
        try:
            from tennis_set_predictor import EnhancedTennisPredictor
            
            print("🔸 Тестируем EnhancedTennisPredictor...")
            enhanced = EnhancedTennisPredictor()
            
            # Генерируем тестовые данные
            test_data = pd.DataFrame({
                'player_rank': [10.0],
                'opponent_rank': [1.0],
                'player_age': [25.0],
                'opponent_age': [23.0],
                'player_recent_win_rate': [0.8],
                'player_form_trend': [0.1],
                'player_surface_advantage': [0.05],
                'h2h_win_rate': [0.4],
                'total_pressure': [4.0],
                'won_at_least_one_set': [1]
            })
            
            # Пытаемся подготовить признаки
            features = enhanced.prepare_features(test_data)
            print(f"  📊 Признаков подготовлено: {len(features.columns)}")
            
            ml_results['enhanced_predictor'] = {
                'available': True,
                'features_count': len(features.columns),
                'can_process_data': True
            }
            
        except Exception as e:
            print(f"  ❌ Ошибка EnhancedTennisPredictor: {e}")
            ml_results['enhanced_predictor'] = {'available': False, 'error': str(e)}
        
        # Подсчитываем общую оценку ML системы
        working_ml = sum(1 for result in ml_results.values() if result.get('available', False))
        total_ml = len(ml_results)
        
        print(f"\n📊 ML компонентов работает: {working_ml}/{total_ml}")
        
        return {
            'components': ml_results,
            'ml_score': working_ml / total_ml if total_ml > 0 else 0,
            'has_real_ml': ml_results.get('prediction_service', {}).get('available', False),
            'integration_quality': 'High' if working_ml >= 2 else 'Medium' if working_ml >= 1 else 'Low'
        }
    
    def test_real_data_integration(self) -> Dict:
        """Тестирование интеграции с реальными данными"""
        print("\n🌍 ТЕСТИРОВАНИЕ ИНТЕГРАЦИИ РЕАЛЬНЫХ ДАННЫХ")
        print("=" * 50)
        
        data_results = {}
        
        # 1. Тестируем Universal Data Collector
        try:
            from universal_tennis_data_collector import UniversalTennisDataCollector, UniversalOddsCollector
            
            print("🔸 Тестируем Universal Data Collector...")
            collector = UniversalTennisDataCollector()
            odds_collector = UniversalOddsCollector()
            
            # Получаем сводку
            summary = collector.get_summary()
            print(f"  📅 Дата: {summary['current_date']}")
            print(f"  🌍 Сезон: {summary['season_context']}")
            print(f"  🏆 Активных турниров: {summary['active_tournaments']}")
            print(f"  📋 Названия турниров: {summary['active_tournament_names']}")
            print(f"  🔜 Следующий Grand Slam: {summary['next_major']}")
            
            # Получаем матчи
            matches = collector.get_current_matches()
            print(f"  🎾 Доступно матчей: {len(matches)}")
            
            # Тестируем коэффициенты
            if matches:
                odds = odds_collector.generate_realistic_odds(matches[:3])
                print(f"  💰 Коэффициентов сгенерировано: {len(odds)}")
            else:
                odds = {}
            
            data_results['universal_collector'] = {
                'working': True,
                'season_context': summary['season_context'],
                'active_tournaments': summary['active_tournaments'],
                'matches_available': len(matches),
                'odds_generation': len(odds) > 0,
                'next_major': summary['next_major']
            }
            
        except Exception as e:
            print(f"  ❌ Ошибка Universal Collector: {e}")
            data_results['universal_collector'] = {'working': False, 'error': str(e)}
        
        # 2. Тестируем API Economy
        try:
            from api_economy_patch import init_api_economy, economical_tennis_request, get_api_usage
            
            print("🔸 Тестируем API Economy...")
            
            # Инициализируем с тестовым ключом
            init_api_economy("test_key", max_per_hour=30, cache_minutes=10)
            
            # Проверяем статистику
            usage = get_api_usage()
            print(f"  📊 Запросов в час: {usage['requests_this_hour']}/{usage['max_per_hour']}")
            print(f"  💾 В кеше: {usage['cache_items']} элементов")
            print(f"  🔄 Ручное обновление: {usage['manual_update_status']}")
            
            data_results['api_economy'] = {
                'working': True,
                'requests_available': usage['remaining_hour'],
                'cache_items': usage['cache_items'],
                'manual_update_support': 'manual_update_status' in usage
            }
            
        except Exception as e:
            print(f"  ❌ Ошибка API Economy: {e}")
            data_results['api_economy'] = {'working': False, 'error': str(e)}
        
        # 3. Тестируем реальные данные игроков
        try:
            from real_tennis_predictor_integration import RealPlayerDataCollector
            
            print("🔸 Тестируем данные игроков...")
            player_collector = RealPlayerDataCollector()
            
            # Тестируем известных игроков
            test_players = ['Carlos Alcaraz', 'Novak Djokovic', 'Iga Swiatek', 'Flavio Cobolli']
            known_players = 0
            
            for player in test_players:
                data = player_collector.get_player_data(player)
                if data['rank'] != 100:  # Не дефолтное значение
                    known_players += 1
                print(f"    {player}: Rank #{data['rank']}, Tour: {data['tour']}")
            
            data_results['player_data'] = {
                'working': True,
                'known_players': known_players,
                'total_tested': len(test_players),
                'coverage': known_players / len(test_players)
            }
            
        except Exception as e:
            print(f"  ❌ Ошибка данных игроков: {e}")
            data_results['player_data'] = {'working': False, 'error': str(e)}
        
        # 4. Проверяем API кеш
        try:
            if os.path.exists('api_cache.json'):
                with open('api_cache.json', 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                matches_in_cache = len(cache_data.get('data', []))
                cache_size = os.path.getsize('api_cache.json')
                
                print(f"🔸 API кеш: {matches_in_cache} матчей, {cache_size:,} bytes")
                
                data_results['api_cache'] = {
                    'exists': True,
                    'matches_count': matches_in_cache,
                    'size_bytes': cache_size,
                    'has_data': matches_in_cache > 0
                }
            else:
                print("🔸 API кеш: отсутствует")
                data_results['api_cache'] = {'exists': False}
                
        except Exception as e:
            print(f"  ❌ Ошибка чтения кеша: {e}")
            data_results['api_cache'] = {'exists': False, 'error': str(e)}
        
        # Оценка качества интеграции
        working_components = sum(1 for result in data_results.values() if result.get('working', False))
        total_components = len(data_results)
        
        print(f"\n📊 Компонентов данных работает: {working_components}/{total_components}")
        
        return {
            'components': data_results,
            'integration_score': working_components / total_components if total_components > 0 else 0,
            'real_data_available': data_results.get('universal_collector', {}).get('matches_available', 0) > 0,
            'player_data_quality': data_results.get('player_data', {}).get('coverage', 0)
        }
    
    def test_learning_and_accumulation(self) -> Dict:
        """Тестирование накопления данных и обучения"""
        print("\n🧠 ТЕСТИРОВАНИЕ НАКОПЛЕНИЯ ДАННЫХ И ОБУЧЕНИЯ")
        print("=" * 50)
        
        learning_results = {}
        
        # 1. Тестируем систему логирования
        try:
            from prediction_logging_system import CompletePredictionLogger, PredictionLoggerIntegration
            
            print("🔸 Тестируем систему логирования...")
            logger = CompletePredictionLogger()
            integration = PredictionLoggerIntegration()
            
            # Проверяем структуру БД
            db_exists = os.path.exists(logger.db_path)
            csv_exists = os.path.exists(logger.csv_path)
            
            print(f"  💾 База данных: {'✅' if db_exists else '❌'}")
            print(f"  📄 CSV файл: {'✅' if csv_exists else '❌'}")
            
            # Тестируем логирование
            test_prediction = {
                'player1': 'Test Player A',
                'player2': 'Test Player B',
                'tournament': 'Test Tournament',
                'surface': 'Hard',
                'match_date': datetime.now().date().isoformat(),
                'our_probability': 0.7,
                'confidence': 'High',
                'ml_system': 'TEST_SYSTEM',
                'prediction_type': 'TEST',
                'key_factors': ['Factor 1', 'Factor 2'],
                'bookmaker_odds': 1.8
            }
            
            pred_id = logger.log_prediction(test_prediction)
            logging_works = pred_id is not None and pred_id != ""
            
            print(f"  📝 Логирование: {'✅' if logging_works else '❌'}")
            
            # Получаем статистику
            stats = integration.get_system_performance()
            print(f"  📊 Всего прогнозов: {stats['total_predictions']}")
            print(f"  🎯 Точность: {stats['accuracy']:.1%}")
            
            learning_results['logging_system'] = {
                'working': True,
                'db_exists': db_exists,
                'csv_exists': csv_exists,
                'can_log': logging_works,
                'total_predictions': stats['total_predictions'],
                'accuracy': stats['accuracy']
            }
            
        except Exception as e:
            print(f"  ❌ Ошибка системы логирования: {e}")
            learning_results['logging_system'] = {'working': False, 'error': str(e)}
        
        # 2. Проверяем накопленные данные
        try:
            print("🔸 Анализируем накопленные данные...")
            
            data_dirs = ['prediction_logs', 'tennis_data_enhanced', 'tennis_models']
            accumulated_data = {}
            
            for dir_name in data_dirs:
                if os.path.exists(dir_name):
                    files = os.listdir(dir_name)
                    total_size = sum(os.path.getsize(os.path.join(dir_name, f)) 
                                   for f in files if os.path.isfile(os.path.join(dir_name, f)))
                    
                    accumulated_data[dir_name] = {
                        'files_count': len(files),
                        'total_size': total_size
                    }
                    
                    print(f"  📁 {dir_name}: {len(files)} файлов, {total_size:,} bytes")
                else:
                    accumulated_data[dir_name] = {'exists': False}
                    print(f"  📁 {dir_name}: ❌ не существует")
            
            learning_results['data_accumulation'] = {
                'working': any(info.get('files_count', 0) > 0 for info in accumulated_data.values()),
                'directories': accumulated_data
            }
            
        except Exception as e:
            print(f"  ❌ Ошибка анализа данных: {e}")
            learning_results['data_accumulation'] = {'working': False, 'error': str(e)}
        
        # 3. Тестируем возможность переобучения
        try:
            print("🔸 Тестируем возможность переобучения...")
            
            # Проверяем доступность EnhancedTennisPredictor для обучения
            from tennis_set_predictor import EnhancedTennisPredictor
            
            predictor = EnhancedTennisPredictor()
            
            # Генерируем небольшую тестовую выборку
            np.random.seed(42)
            n_samples = 100
            
            test_data = pd.DataFrame({
                'player_rank': np.random.randint(1, 100, n_samples),
                'opponent_rank': np.random.randint(1, 100, n_samples),
                'player_age': np.random.randint(18, 35, n_samples),
                'opponent_age': np.random.randint(18, 35, n_samples),
                'player_recent_win_rate': np.random.random(n_samples),
                'player_form_trend': np.random.normal(0, 0.1, n_samples),
                'player_surface_advantage': np.random.normal(0, 0.1, n_samples),
                'h2h_win_rate': np.random.random(n_samples),
                'total_pressure': np.random.uniform(1, 4, n_samples),
                'won_at_least_one_set': np.random.binomial(1, 0.6, n_samples)
            })
            
            # Подготавливаем данные
            X = predictor.prepare_features(test_data)
            y = test_data['won_at_least_one_set']
            
            # Разделяем на train/val
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Пытаемся обучить (только быстрые модели для теста)
            try:
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import roc_auc_score
                
                # Быстрое обучение RF
                rf = RandomForestClassifier(n_estimators=10, random_state=42)
                rf.fit(X_train, y_train)
                rf_pred = rf.predict_proba(X_val)[:, 1]
                rf_auc = roc_auc_score(y_val, rf_pred)
                
                # Быстрое обучение LR
                lr = LogisticRegression(random_state=42, max_iter=100)
                lr.fit(X_train, y_train)
                lr_pred = lr.predict_proba(X_val)[:, 1]
                lr_auc = roc_auc_score(y_val, lr_pred)
                
                can_retrain = True
                best_auc = max(rf_auc, lr_auc)
                
                print(f"  🏋️ Переобучение: ✅ (AUC: {best_auc:.3f})")
                
            except Exception as train_e:
                can_retrain = False
                best_auc = 0
                print(f"  🏋️ Переобучение: ❌ ({str(train_e)[:50]}...)")
            
            learning_results['retraining_capability'] = {
                'working': can_retrain,
                'test_auc': best_auc,
                'data_preparable': True
            }
            
        except Exception as e:
            print(f"  ❌ Ошибка тестирования переобучения: {e}")
            learning_results['retraining_capability'] = {'working': False, 'error': str(e)}
        
        # Оценка способности к обучению
        learning_score = sum(1 for result in learning_results.values() if result.get('working', False))
        total_learning = len(learning_results)
        
        print(f"\n📊 Компонентов обучения работает: {learning_score}/{total_learning}")
        
        return {
            'components': learning_results,
            'learning_score': learning_score / total_learning if total_learning > 0 else 0,
            'can_accumulate_data': learning_results.get('logging_system', {}).get('working', False),
            'can_retrain': learning_results.get('retraining_capability', {}).get('working', False)
        }
    
    def test_end_to_end_workflow(self) -> Dict:
        """Тестирование полного рабочего процесса"""
        print("\n🔄 ТЕСТИРОВАНИЕ ПОЛНОГО РАБОЧЕГО ПРОЦЕССА")
        print("=" * 50)
        
        workflow_results = {}
        
        try:
            print("🔸 Запуск полного цикла: данные → ML → логирование...")
            
            # 1. Получение данных
            print("  1️⃣ Получение данных о матчах...")
            from universal_tennis_data_collector import UniversalTennisDataCollector, UniversalOddsCollector
            
            collector = UniversalTennisDataCollector()
            odds_collector = UniversalOddsCollector()
            
            matches = collector.get_current_matches()
            if not matches:
                # Создаем тестовые данные
                matches = [{
                    'id': 'test_workflow_match',
                    'player1': 'Carlos Alcaraz',
                    'player2': 'Novak Djokovic',
                    'tournament': 'Test Tournament',
                    'surface': 'Hard',
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'time': '14:00',
                    'round': 'SF'
                }]
            
            print(f"    ✅ Получено матчей: {len(matches)}")
            
            # 2. Генерация коэффициентов
            print("  2️⃣ Генерация коэффициентов...")
            odds_data = odds_collector.generate_realistic_odds(matches)
            print(f"    ✅ Сгенерировано коэффициентов: {len(odds_data)}")
            
            # 3. ML анализ
            print("  3️⃣ ML анализ матчей...")
            from real_tennis_predictor_integration import RealTennisPredictor
            
            predictor = RealTennisPredictor()
            predictions = []
            
            for match in matches[:3]:  # Берем первые 3 матча
                try:
                    prediction = predictor.predict_match(
                        match['player1'], match['player2'],
                        match['tournament'], match.get('surface', 'Hard'), match.get('round', 'R32')
                    )
                    predictions.append({
                        'match_id': match['id'],
                        'probability': prediction['probability'],
                        'confidence': prediction['confidence'],
                        'prediction_type': prediction['prediction_type'],
                        'key_factors': prediction.get('key_factors', [])
                    })
                except Exception as e:
                    print(f"      ⚠️ Ошибка прогноза для {match['player1']}: {e}")
            
            print(f"    ✅ Проанализировано матчей: {len(predictions)}")
            
            # 4. Логирование результатов
            print("  4️⃣ Логирование результатов...")
            from prediction_logging_system import PredictionLoggerIntegration
            
            logger_integration = PredictionLoggerIntegration()
            
            logged_count = 0
            for i, prediction in enumerate(predictions):
                try:
                    match = matches[i]
                    match_result = {
                        'player1': match['player1'],
                        'player2': match['player2'],
                        'tournament': match['tournament'],
                        'surface': match.get('surface', 'Hard'),
                        'date': match['date'],
                        'prediction': prediction,
                        'underdog_analysis': {
                            'underdog_probability': prediction['probability'],
                            'confidence': prediction['confidence'],
                            'prediction_type': prediction['prediction_type'],
                            'key_factors': prediction['key_factors']
                        }
                    }
                    
                    logged_id = logger_integration.log_match_prediction(match_result)
                    if logged_id:
                        logged_count += 1
                        
                except Exception as e:
                    print(f"      ⚠️ Ошибка логирования: {e}")
            
            print(f"    ✅ Залогировано прогнозов: {logged_count}")
            
            # 5. Получение статистики системы
            print("  5️⃣ Получение статистики системы...")
            performance = logger_integration.get_system_performance()
            
            print(f"    📊 Всего прогнозов в системе: {performance['total_predictions']}")
            print(f"    🎯 Текущая точность: {performance['accuracy']:.1%}")
            
            workflow_results = {
                'working': True,
                'steps_completed': 5,
                'matches_processed': len(matches),
                'odds_generated': len(odds_data),
                'predictions_made': len(predictions),
                'predictions_logged': logged_count,
                'system_accuracy': performance['accuracy'],
                'total_predictions_in_system': performance['total_predictions']
            }
            
            print(f"  ✅ Полный рабочий процесс завершен успешно!")
            
        except Exception as e:
            print(f"  ❌ Ошибка в рабочем процессе: {e}")
            traceback.print_exc()
            workflow_results = {
                'working': False,
                'error': str(e),
                'error_details': traceback.format_exc()
            }
        
        return workflow_results
    
    def analyze_system_intelligence(self) -> Dict:
        """Анализ 'умности' системы - работает ли она грамотно"""
        print("\n🧠 АНАЛИЗ ИНТЕЛЛЕКТА СИСТЕМЫ")
        print("=" * 50)
        
        intelligence_results = {}
        
        # 1. Проверяем использует ли система реальные рейтинги
        try:
            print("🔸 Проверяем качество данных игроков...")
            from real_tennis_predictor_integration import RealPlayerDataCollector
            
            collector = RealPlayerDataCollector()
            
            # Тестируем известные различия в рейтингах
            test_cases = [
                ('Jannik Sinner', 1),    # Должен быть #1
                ('Carlos Alcaraz', 2),   # Должен быть #2  
                ('Flavio Cobolli', 32),  # КРИТИЧНО: должен быть #32, не #100
                ('Jacob Fearnley', 320), # Низкий рейтинг
            ]
            
            accurate_rankings = 0
            for player, expected_rank in test_cases:
                actual_data = collector.get_player_data(player)
                actual_rank = actual_data['rank']
                
                # Допускаем небольшую погрешность для топ игроков
                tolerance = 2 if expected_rank <= 10 else 10 if expected_rank <= 50 else 20
                
                if abs(actual_rank - expected_rank) <= tolerance:
                    accurate_rankings += 1
                    status = "✅"
                else:
                    status = "❌"
                
                print(f"    {status} {player}: ожидался #{expected_rank}, получен #{actual_rank}")
            
            ranking_accuracy = accurate_rankings / len(test_cases)
            print(f"    📊 Точность рейтингов: {ranking_accuracy:.1%}")
            
            intelligence_results['ranking_quality'] = {
                'accurate_count': accurate_rankings,
                'total_tested': len(test_cases),
                'accuracy': ranking_accuracy,
                'uses_real_rankings': ranking_accuracy > 0.5
            }
            
        except Exception as e:
            print(f"    ❌ Ошибка проверки рейтингов: {e}")
            intelligence_results['ranking_quality'] = {'working': False, 'error': str(e)}
        
        # 2. Проверяем логичность ML прогнозов
        try:
            print("🔸 Проверяем логичность ML прогнозов...")
            from real_tennis_predictor_integration import RealTennisPredictor
            
            predictor = RealTennisPredictor()
            
            # Тестовые сценарии для проверки логики
            test_scenarios = [
                {
                    'name': 'Топ против новичка',
                    'player1': 'Jannik Sinner',     # #1
                    'player2': 'Jacob Fearnley',    # #320
                    'expected_p1_favored': True
                },
                {
                    'name': 'Близкие рейтинги',
                    'player1': 'Carlos Alcaraz',    # #2
                    'player2': 'Alexander Zverev',  # #3
                    'expected_close': True
                },
                {
                    'name': 'Underdog сценарий',
                    'player1': 'Flavio Cobolli',   # #32
                    'player2': 'Novak Djokovic',   # #5
                    'expected_p1_favored': False
                }
            ]
            
            logical_predictions = 0
            for scenario in test_scenarios:
                try:
                    result = predictor.predict_match(
                        scenario['player1'], scenario['player2'],
                        'Test Tournament', 'Hard', 'R32'
                    )
                    
                    probability = result['probability']
                    
                    # Проверяем логику
                    if scenario.get('expected_p1_favored') == True:
                        logical = probability > 0.55  # Первый должен быть фаворитом
                    elif scenario.get('expected_p1_favored') == False:
                        logical = probability < 0.45  # Первый должен быть аутсайдером
                    elif scenario.get('expected_close') == True:
                        logical = 0.4 <= probability <= 0.6  # Должно быть близко к 50/50
                    else:
                        logical = True
                    
                    if logical:
                        logical_predictions += 1
                        status = "✅"
                    else:
                        status = "❌"
                    
                    print(f"    {status} {scenario['name']}: {probability:.1%}")
                    
                except Exception as e:
                    print(f"    ❌ Ошибка в сценарии '{scenario['name']}': {e}")
            
            logic_score = logical_predictions / len(test_scenarios)
            print(f"    📊 Логичность прогнозов: {logic_score:.1%}")
            
            intelligence_results['prediction_logic'] = {
                'logical_count': logical_predictions,
                'total_tested': len(test_scenarios),
                'logic_score': logic_score,
                'makes_sense': logic_score > 0.6
            }
            
        except Exception as e:
            print(f"    ❌ Ошибка проверки логики: {e}")
            intelligence_results['prediction_logic'] = {'working': False, 'error': str(e)}
        
        # 3. Проверяем адаптируется ли система к сезонам
        try:
            print("🔸 Проверяем сезонную адаптацию...")
            from universal_tennis_data_collector import UniversalTennisDataCollector
            
            collector = UniversalTennisDataCollector()
            summary = collector.get_summary()
            
            # Проверяем понимает ли система текущий сезон
            season_context = summary['season_context']
            current_month = datetime.now().month
            
            # Логика сезонов
            expected_seasons = {
                (1, 2): "Hard Court Season",
                (3, 4, 5): "Clay Court Season", 
                (6, 7): "Grass Court Season",
                (8, 9): "Hard Court Season",
                (10, 11): "Indoor Season",
                (12,): "Off Season"
            }
            
            season_correct = False
            for months, expected in expected_seasons.items():
                if current_month in months and expected in season_context:
                    season_correct = True
                    break
            
            print(f"    📅 Текущий сезон: {season_context}")
            print(f"    ✅ Сезон определен {'правильно' if season_correct else 'неправильно'}")
            
            # Проверяем адаптацию к турнирам
            active_tournaments = summary['active_tournaments']
            next_major = summary['next_major']
            
            print(f"    🏆 Активных турниров: {active_tournaments}")
            print(f"    🔜 Следующий major: {next_major}")
            
            intelligence_results['seasonal_adaptation'] = {
                'understands_season': season_correct,
                'season_context': season_context,
                'tracks_tournaments': active_tournaments >= 0,
                'knows_next_major': next_major != "Unknown"
            }
            
        except Exception as e:
            print(f"    ❌ Ошибка проверки сезонной адаптации: {e}")
            intelligence_results['seasonal_adaptation'] = {'working': False, 'error': str(e)}
        
        # 4. Проверяем накопление опыта
        try:
            print("🔸 Проверяем накопление опыта...")
            
            # Проверяем растет ли количество прогнозов со временем
            from prediction_logging_system import PredictionLoggerIntegration
            
            logger_integration = PredictionLoggerIntegration()
            performance = logger_integration.get_system_performance()
            
            total_predictions = performance['total_predictions']
            accuracy = performance['accuracy']
            
            print(f"    📊 Всего прогнозов в системе: {total_predictions}")
            print(f"    🎯 Текущая точность: {accuracy:.1%}")
            
            # Система считается обучающейся если:
            # 1. Есть накопленные прогнозы
            # 2. Есть система для отслеживания точности
            # 3. Точность разумная (не случайная)
            
            is_learning = (
                total_predictions > 0 and
                0.4 <= accuracy <= 0.9  # Разумный диапазон точности
            )
            
            print(f"    🧠 Система {'обучается' if is_learning else 'не обучается'}")
            
            intelligence_results['experience_accumulation'] = {
                'total_predictions': total_predictions,
                'accuracy': accuracy,
                'is_learning': is_learning,
                'has_memory': total_predictions > 0
            }
            
        except Exception as e:
            print(f"    ❌ Ошибка проверки накопления опыта: {e}")
            intelligence_results['experience_accumulation'] = {'working': False, 'error': str(e)}
        
        # Общая оценка интеллекта системы
        intelligence_score = 0
        max_score = 0
        
        for component, results in intelligence_results.items():
            if isinstance(results, dict) and 'working' not in results:
                max_score += 1
                if component == 'ranking_quality':
                    if results.get('uses_real_rankings', False):
                        intelligence_score += 1
                elif component == 'prediction_logic':
                    if results.get('makes_sense', False):
                        intelligence_score += 1
                elif component == 'seasonal_adaptation':
                    if results.get('understands_season', False):
                        intelligence_score += 1
                elif component == 'experience_accumulation':
                    if results.get('is_learning', False):
                        intelligence_score += 1
        
        overall_intelligence = intelligence_score / max_score if max_score > 0 else 0
        
        print(f"\n📊 Общий интеллект системы: {intelligence_score}/{max_score} ({overall_intelligence:.1%})")
        
        return {
            'components': intelligence_results,
            'intelligence_score': overall_intelligence,
            'uses_real_data': intelligence_results.get('ranking_quality', {}).get('uses_real_rankings', False),
            'logical_predictions': intelligence_results.get('prediction_logic', {}).get('makes_sense', False),
            'adapts_to_context': intelligence_results.get('seasonal_adaptation', {}).get('understands_season', False),
            'learns_from_experience': intelligence_results.get('experience_accumulation', {}).get('is_learning', False)
        }
    
    def generate_comprehensive_assessment(self) -> Dict:
        """Генерация итоговой оценки системы"""
        
        # Собираем все результаты тестов
        file_analysis = self.analyze_file_structure()
        import_analysis = self.test_imports_and_modules()
        ml_analysis = self.test_ml_models_comprehensive()
        data_analysis = self.test_real_data_integration()
        learning_analysis = self.test_learning_and_accumulation()
        workflow_analysis = self.test_end_to_end_workflow()
        intelligence_analysis = self.analyze_system_intelligence()
        
        # Сохраняем в отчет
        self.report.update({
            'file_structure': file_analysis,
            'imports_modules': import_analysis,
            'ml_performance': ml_analysis,
            'data_integration': data_analysis,
            'learning_capability': learning_analysis,
            'workflow_test': workflow_analysis,
            'system_intelligence': intelligence_analysis
        })
        
        # Рассчитываем общую оценку
        scores = {
            'file_completeness': file_analysis['completeness'],
            'imports_success': import_analysis['success_rate'],
            'ml_capability': ml_analysis['ml_score'],
            'data_integration': data_analysis['integration_score'],
            'learning_ability': learning_analysis['learning_score'],
            'workflow_success': 1.0 if workflow_analysis.get('working', False) else 0.0,
            'intelligence_level': intelligence_analysis['intelligence_score']
        }
        
        # Взвешенная оценка (некоторые аспекты важнее)
        weights = {
            'file_completeness': 0.1,
            'imports_success': 0.15,
            'ml_capability': 0.25,
            'data_integration': 0.2,
            'learning_ability': 0.15,
            'workflow_success': 0.1,
            'intelligence_level': 0.05
        }
        
        overall_score = sum(scores[key] * weights[key] for key in scores.keys())
        overall_score *= 100  # Переводим в проценты
        
        # Определяем grade
        if overall_score >= 90:
            grade = "🏆 ПРЕВОСХОДНО - Продакшн готов"
        elif overall_score >= 80:
            grade = "✅ ОТЛИЧНО - Готов к использованию"
        elif overall_score >= 70:
            grade = "👍 ХОРОШО - Работает с минимальными доработками"
        elif overall_score >= 60:
            grade = "⚠️ УДОВЛЕТВОРИТЕЛЬНО - Требует доработки"
        elif overall_score >= 40:
            grade = "🔧 ТРЕБУЕТ РАБОТЫ - Много проблем"
        else:
            grade = "❌ КРИТИЧНО - Система не функциональна"
        
        # Определяем готовность к реальному использованию
        ready_for_production = (
            overall_score >= 75 and
            ml_analysis['has_real_ml'] and
            data_analysis['real_data_available'] and
            workflow_analysis.get('working', False) and
            intelligence_analysis['uses_real_data']
        )
        
        assessment = {
            'overall_score': overall_score,
            'grade': grade,
            'component_scores': scores,
            'ready_for_production': ready_for_production,
            'strengths': self._identify_strengths(),
            'weaknesses': self._identify_weaknesses(),
            'recommendations': self._generate_recommendations(),
            'next_steps': self._suggest_next_steps()
        }
        
        self.report['overall_assessment'] = assessment
        
        return assessment
    
    def _identify_strengths(self) -> List[str]:
        """Выявление сильных сторон системы"""
        strengths = []
        
        # Анализируем каждый компонент
        if self.report.get('file_structure', {}).get('completeness', 0) > 0.8:
            strengths.append("✅ Полная структура файлов - все основные компоненты на месте")
        
        if self.report.get('imports_modules', {}).get('success_rate', 0) > 0.7:
            strengths.append("✅ Стабильные импорты - модули загружаются без ошибок")
        
        if self.report.get('ml_performance', {}).get('has_real_ml', False):
            strengths.append("✅ Реальные ML модели - использует обученные нейросети")
        
        if self.report.get('data_integration', {}).get('real_data_available', False):
            strengths.append("✅ Интеграция с реальными данными турниров")
        
        if self.report.get('learning_capability', {}).get('can_accumulate_data', False):
            strengths.append("✅ Накопление опыта - система записывает и анализирует результаты")
        
        if self.report.get('system_intelligence', {}).get('uses_real_data', False):
            strengths.append("✅ Использует актуальные рейтинги игроков")
        
        if self.report.get('system_intelligence', {}).get('logical_predictions', False):
            strengths.append("✅ Логичные прогнозы - система понимает силу игроков")
        
        if self.report.get('workflow_test', {}).get('working', False):
            strengths.append("✅ Полный рабочий процесс от данных до результатов")
        
        return strengths
    
    def _identify_weaknesses(self) -> List[str]:
        """Выявление слабых сторон системы"""
        weaknesses = []
        
        # Анализируем проблемы
        if self.report.get('file_structure', {}).get('completeness', 0) < 0.7:
            missing = self.report.get('file_structure', {}).get('critical_missing', [])
            weaknesses.append(f"❌ Отсутствуют критичные файлы: {', '.join(missing[:3])}")
        
        if not self.report.get('ml_performance', {}).get('has_real_ml', False):
            weaknesses.append("❌ Нет реальных ML моделей - работает на симуляции")
        
        if not self.report.get('data_integration', {}).get('real_data_available', False):
            weaknesses.append("❌ Нет доступа к реальным матчам - нужен API ключ")
        
        if not self.report.get('learning_capability', {}).get('can_retrain', False):
            weaknesses.append("❌ Система не может переобучаться на новых данных")
        
        if not self.report.get('system_intelligence', {}).get('uses_real_data', False):
            weaknesses.append("❌ Использует устаревшие или неточные рейтинги")
        
        # Проверяем накопленный опыт
        total_predictions = self.report.get('system_intelligence', {}).get('components', {}).get('experience_accumulation', {}).get('total_predictions', 0)
        if total_predictions < 10:
            weaknesses.append("❌ Мало накопленного опыта - система только начинает обучение")
        
        return weaknesses
    
    def _generate_recommendations(self) -> List[str]:
        """Генерация рекомендаций по улучшению"""
        recommendations = []
        
        # Рекомендации на основе анализа
        if not self.report.get('ml_performance', {}).get('has_real_ml', False):
            recommendations.append("🤖 Обучить и загрузить реальные ML модели в tennis_models/")
        
        if not self.report.get('data_integration', {}).get('real_data_available', False):
            recommendations.append("🔑 Настроить API ключ для The Odds API в config.json")
        
        overall_score = self.report.get('overall_assessment', {}).get('overall_score', 0)
        if overall_score < 70:
            recommendations.append("🔧 Исправить критические ошибки перед запуском")
        
        if not self.report.get('learning_capability', {}).get('can_retrain', False):
            recommendations.append("📚 Настроить автоматическое переобучение моделей")
        
        recommendations.extend([
            "📊 Начать накапливать реальные результаты матчей",
            "🎯 Запускать систему во время активных турниров",
            "📈 Отслеживать точность прогнозов и ROI",
            "🔄 Регулярно обновлять рейтинги игроков"
        ])
        
        return recommendations
    
    def _suggest_next_steps(self) -> List[str]:
        """Предложение следующих шагов"""
        next_steps = []
        
        overall_score = self.report.get('overall_assessment', {}).get('overall_score', 0)
        
        if overall_score >= 80:
            next_steps = [
                "🚀 Система готова к использованию!",
                "📅 Дождаться начала активных турниров",
                "💰 Начать с небольших тестовых ставок",
                "📊 Отслеживать результаты первые 2 недели",
                "📈 Постепенно увеличивать размеры ставок при хороших результатах"
            ]
        elif overall_score >= 60:
            next_steps = [
                "🔧 Исправить выявленные проблемы",
                "🧪 Протестировать на исторических данных",
                "📝 Настроить систему логирования",
                "🎯 Повторить анализ после исправлений"
            ]
        else:
            next_steps = [
                "🛠️ Критические исправления системы",
                "📚 Обучение ML моделей на больших данных",
                "🔑 Настройка всех API интеграций",
                "🏗️ Полная перестройка проблемных компонентов"
            ]
        
        return next_steps
    
    def save_full_report(self) -> str:
        """Сохранение полного отчета"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"tennis_system_comprehensive_analysis_{timestamp}.json"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(self.report, f, indent=2, default=str, ensure_ascii=False)
            
            print(f"\n💾 Полный отчет сохранен: {report_file}")
            return report_file
            
        except Exception as e:
            print(f"❌ Ошибка сохранения отчета: {e}")
            return ""
    
    def print_executive_summary(self, assessment: Dict):
        """Вывод краткого резюме для руководства"""
        print("\n" + "="*70)
        print("📋 EXECUTIVE SUMMARY - СИСТЕМА ГОТОВА К ПРОДАКШЕНУ?")
        print("="*70)
        
        score = assessment['overall_score']
        grade = assessment['grade']
        ready = assessment['ready_for_production']
        
        print(f"🎯 ИТОГОВАЯ ОЦЕНКА: {score:.1f}/100")
        print(f"🏆 РЕЙТИНГ: {grade}")
        print(f"🚀 ГОТОВНОСТЬ К ПРОДАКШЕНУ: {'✅ ДА' if ready else '❌ НЕТ'}")
        
        print(f"\n💪 СИЛЬНЫЕ СТОРОНЫ ({len(assessment['strengths'])}):")
        for strength in assessment['strengths']:
            print(f"  {strength}")
        
        print(f"\n⚠️ СЛАБЫЕ СТОРОНЫ ({len(assessment['weaknesses'])}):")
        for weakness in assessment['weaknesses']:
            print(f"  {weakness}")
        
        print(f"\n📋 СЛЕДУЮЩИЕ ШАГИ:")
        for step in assessment['next_steps']:
            print(f"  {step}")
        
        # Краткие метрики
        scores = assessment['component_scores']
        print(f"\n📊 ДЕТАЛЬНЫЕ ОЦЕНКИ:")
        print(f"  📁 Файловая структура: {scores['file_completeness']:.1%}")
        print(f"  🔧 Загрузка модулей: {scores['imports_success']:.1%}")
        print(f"  🤖 ML возможности: {scores['ml_capability']:.1%}")
        print(f"  🌍 Интеграция данных: {scores['data_integration']:.1%}")
        print(f"  🧠 Способность к обучению: {scores['learning_ability']:.1%}")
        print(f"  🔄 Полный рабочий процесс: {scores['workflow_success']:.1%}")
        print(f"  🎯 Интеллект системы: {scores['intelligence_level']:.1%}")
        
        if ready:
            print(f"\n🎉 ВЕРДИКТ: Система готова к использованию!")
            print(f"💡 Можно начинать с реальных ставок в следующем турнире.")
        else:
            print(f"\n🔧 ВЕРДИКТ: Система требует доработки.")
            print(f"💡 Исправьте критические проблемы перед запуском.")
        
        print("="*70)


def main():
    """Главная функция запуска полного анализа"""
    print("🎾 КОМПЛЕКСНЫЙ АНАЛИЗ ТЕННИСНОЙ СИСТЕМЫ")
    print("=" * 70)
    print("🔍 Проверяем ВСЕ компоненты системы:")
    print("• Файловую структуру и доступность модулей")
    print("• Реальную работу ML моделей") 
    print("• Интеграцию с источниками данных")
    print("• Способность к накоплению опыта и обучению")
    print("• Качество и логичность прогнозов")
    print("• Полный end-to-end рабочий процесс")
    print("• Готовность к реальному использованию")
    print("=" * 70)
    
    # Создаем анализатор
    analyzer = ComprehensiveTennisSystemAnalyzer()
    
    try:
        print(f"\n🕐 Начало анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Запускаем полный анализ
        assessment = analyzer.generate_comprehensive_assessment()
        
        # Выводим executive summary
        analyzer.print_executive_summary(assessment)
        
        # Сохраняем полный отчет
        report_file = analyzer.save_full_report()
        
        print(f"\n📄 Полный технический отчет сохранен: {report_file}")
        
        print(f"\n🎯 РЕЗЮМЕ АНАЛИЗА:")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # Показываем ключевые метрики
        if assessment['overall_score'] >= 80:
            print("🟢 СТАТУС: СИСТЕМА ГОТОВА К ИСПОЛЬЗОВАНИЮ")
            print("💡 Рекомендация: Можно начинать реальные ставки")
        elif assessment['overall_score'] >= 60:
            print("🟡 СТАТУС: СИСТЕМА РАБОТАЕТ, НО ТРЕБУЕТ УЛУЧШЕНИЙ")
            print("💡 Рекомендация: Исправить основные проблемы, затем тестировать")
        else:
            print("🔴 СТАТУС: СИСТЕМА НЕ ГОТОВА")
            print("💡 Рекомендация: Критические исправления перед использованием")
        
        print(f"📊 Общая оценка: {assessment['overall_score']:.1f}/100")
        print(f"🎯 Готовность к продакшену: {'ДА' if assessment['ready_for_production'] else 'НЕТ'}")
        
        # Топ-3 проблемы
        if assessment['weaknesses']:
            print(f"\n🚨 ПРИОРИТЕТНЫЕ ПРОБЛЕМЫ:")
            for i, weakness in enumerate(assessment['weaknesses'][:3], 1):
                print(f"  {i}. {weakness}")
        
        # Топ-3 рекомендации
        if assessment['recommendations']:
            print(f"\n💡 ПЕРВООЧЕРЕДНЫЕ ДЕЙСТВИЯ:")
            for i, rec in enumerate(assessment['recommendations'][:3], 1):
                print(f"  {i}. {rec}")
        
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"⏰ Анализ завершен: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🕐 Общее время: {(datetime.now() - datetime.fromisoformat(analyzer.report['analysis_time'])).total_seconds():.1f} секунд")
        
        return assessment
        
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА АНАЛИЗА: {e}")
        print("🔍 Детали ошибки:")
        traceback.print_exc()
        
        # Попытка сохранить частичный отчет
        try:
            analyzer.report['error'] = {
                'message': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat()
            }
            report_file = analyzer.save_full_report()
            print(f"💾 Частичный отчет сохранен: {report_file}")
        except:
            print("❌ Не удалось сохранить даже частичный отчет")
        
        return None


def run_quick_test():
    """Быстрый тест основных компонентов"""
    print("⚡ БЫСТРЫЙ ТЕСТ СИСТЕМЫ")
    print("=" * 40)
    
    quick_tests = [
        ("📁 Файлы", lambda: os.path.exists('tennis_backend.py')),
        ("🤖 ML модули", lambda: check_ml_imports()),
        ("🌍 Сборщики данных", lambda: check_data_collectors()),
        ("💾 Логирование", lambda: check_logging_system()),
    ]
    
    results = []
    for test_name, test_func in quick_tests:
        try:
            result = test_func()
            status = "✅" if result else "❌"
            results.append(result)
        except Exception as e:
            status = f"❌ ({str(e)[:20]}...)"
            results.append(False)
        
        print(f"{test_name}: {status}")
    
    success_rate = sum(results) / len(results)
    print(f"\n📊 Быстрый тест: {success_rate:.1%} компонентов работает")
    
    if success_rate > 0.7:
        print("✅ Система выглядит работоспособной - запускаем полный анализ")
        return True
    else:
        print("⚠️ Обнаружены проблемы - рекомендуется полный анализ")
        return False


def check_ml_imports():
    """Проверка ML импортов"""
    try:
        import tennis_prediction_module
        import real_tennis_predictor_integration
        return True
    except ImportError:
        return False


def check_data_collectors():
    """Проверка сборщиков данных"""
    try:
        import universal_tennis_data_collector
        import api_economy_patch
        return True
    except ImportError:
        return False


def check_logging_system():
    """Проверка системы логирования"""
    try:
        import prediction_logging_system
        return True
    except ImportError:
        return False


def print_usage_help():
    """Помощь по использованию"""
    print("""
🎾 АНАЛИЗАТОР ТЕННИСНОЙ СИСТЕМЫ - СПРАВКА
═══════════════════════════════════════════

ИСПОЛЬЗОВАНИЕ:
  python paste.py                    # Полный анализ
  python paste.py --quick            # Быстрый тест
  python paste.py --help             # Эта справка

ЧТО ПРОВЕРЯЕТСЯ:
  📁 Структура файлов              # Все ли файлы на месте
  🔧 Импорты модулей              # Загружаются ли компоненты
  🤖 ML модели                    # Работают ли нейросети
  🌍 Интеграция данных            # Доступ к турнирам и матчам
  🧠 Обучение системы             # Накопление опыта
  🔄 Рабочий процесс              # End-to-end тестирование
  🎯 Интеллект системы            # Логичность прогнозов

РЕЗУЛЬТАТ:
  • Оценка готовности к продакшену (0-100)
  • Детальный анализ каждого компонента
  • Список проблем и рекомендаций
  • JSON отчет с техническими деталями

ТРЕБОВАНИЯ:
  • Python 3.8+
  • Все файлы системы в текущей папке
  • Опционально: API ключи для реальных данных

ПРИМЕРЫ КОМАНД:
  python paste.py                    # Полный анализ (рекомендуется)
  python paste.py --quick            # Быстрая проверка за 30 сек
""")


if __name__ == "__main__":
    # Парсинг аргументов командной строки
    if len(sys.argv) > 1:
        if '--help' in sys.argv or '-h' in sys.argv:
            print_usage_help()
            sys.exit(0)
        elif '--quick' in sys.argv or '-q' in sys.argv:
            print("⚡ Запуск быстрого теста...")
            if run_quick_test():
                print("\n🎯 Хотите запустить полный анализ? (y/N): ", end="")
                try:
                    if input().lower().startswith('y'):
                        main()
                except KeyboardInterrupt:
                    print("\n👋 Анализ отменен")
            sys.exit(0)
    
    # Запуск полного анализа
    print("🚀 Запуск ПОЛНОГО комплексного анализа...")
    print("⏱️ Это займет 2-5 минут в зависимости от системы")
    print("🔍 Будут протестированы ВСЕ компоненты системы")
    
    try:
        assessment = main()
        
        if assessment:
            # Финальное сообщение пользователю
            print(f"\n🎉 АНАЛИЗ ЗАВЕРШЕН!")
            
            if assessment['ready_for_production']:
                print("🚀 Ваша теннисная система ГОТОВА к использованию!")
                print("💰 Можете начинать делать реальные ставки.")
                print("📊 Не забывайте отслеживать результаты.")
            elif assessment['overall_score'] >= 60:
                print("🔧 Система работает, но требует улучшений.")
                print("⚡ Исправьте основные проблемы и повторите анализ.")
            else:
                print("🛠️ Система требует серьезной доработки.")
                print("📚 Следуйте рекомендациям для исправления проблем.")
            
            print(f"\n📄 Детальный отчет сохранен для изучения.")
            print("💡 Используйте его для планирования улучшений.")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Анализ остановлен пользователем")
        print("💡 Запустите заново для получения результатов")
    except Exception as e:
        print(f"\n💥 Непредвиденная ошибка: {e}")
        print("🔍 Проверьте установку зависимостей и структуру файлов")
    
    print(f"\n👋 Спасибо за использование анализатора!")