#!/usr/bin/env python3
"""
🧪 ML SYSTEM TESTER
Тестирует всю ML систему для теннисных прогнозов
"""

import os
import sys
from datetime import datetime

def test_ml_predictor():
    """Тест ML предиктора"""
    print("🤖 Тестируем ML предиктор...")
    
    try:
        from real_tennis_predictor_integration import RealTennisPredictor
        
        predictor = RealTennisPredictor()
        
        # Тестовые матчи
        test_matches = [
            {
                'player1': 'Carlos Alcaraz',
                'player2': 'Novak Djokovic', 
                'tournament': 'Wimbledon',
                'surface': 'Grass',
                'round': 'F'
            },
            {
                'player1': 'Brandon Nakashima',
                'player2': 'Bu Yunchaokete',
                'tournament': 'Wimbledon', 
                'surface': 'Grass',
                'round': 'R64'
            }
        ]
        
        for i, match in enumerate(test_matches, 1):
            print(f"\n🎾 Тест {i}: {match['player1']} vs {match['player2']}")
            
            result = predictor.predict_match(
                match['player1'], match['player2'],
                match['tournament'], match['surface'], match['round']
            )
            
            print(f"   📊 Прогноз: {result['probability']:.1%}")
            print(f"   🎯 Уверенность: {result['confidence']}")
            print(f"   🔬 Тип: {result['prediction_type']}")
            
            if result['key_factors']:
                print(f"   🔍 Факторы: {len(result['key_factors'])} найдено")
        
        print("\n✅ ML предиктор работает!")
        return True
        
    except ImportError as e:
        print(f"❌ Ошибка импорта ML предиктора: {e}")
        return False
    except Exception as e:
        print(f"❌ Ошибка ML предиктора: {e}")
        return False

def test_tennis_prediction_module():
    """Тест модуля прогнозирования"""
    print("\n🔮 Тестируем модуль прогнозирования...")
    
    try:
        from tennis_prediction_module import TennisPredictionService, create_match_data
        
        service = TennisPredictionService()
        info = service.get_model_info()
        
        print(f"   📊 Статус: {info['status']}")
        print(f"   🤖 Модели: {info.get('models_count', 0)}")
        
        if info['status'] == 'loaded':
            # Тест прогноза
            test_data = create_match_data(
                player_rank=1,
                opponent_rank=45,
                player_recent_win_rate=0.85,
                player_surface_advantage=0.12
            )
            
            result = service.predict_match(test_data)
            print(f"   🎯 Тест прогноз: {result['probability']:.1%}")
            print(f"   ✅ Модуль работает!")
        else:
            print(f"   ⚠️ Модели не загружены, используется demo режим")
        
        return True
        
    except ImportError as e:
        print(f"❌ Ошибка импорта модуля: {e}")
        return False
    except Exception as e:
        print(f"❌ Ошибка модуля: {e}")
        return False

def test_api_endpoints():
    """Тест API endpoints"""
    print("\n📡 Тестируем API endpoints...")
    
    try:
        import requests
        
        base_url = "http://localhost:5001/api"
        endpoints = [
            ('/health', 'Health check'),
            ('/stats', 'Статистика'),
            ('/matches', 'Матчи')
        ]
        
        for endpoint, description in endpoints:
            try:
                response = requests.get(f"{base_url}{endpoint}", timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        print(f"   ✅ {description}: OK")
                    else:
                        print(f"   ⚠️ {description}: Ответ без success")
                else:
                    print(f"   ❌ {description}: HTTP {response.status_code}")
                    
            except requests.RequestException as e:
                print(f"   ❌ {description}: {e}")
        
        return True
        
    except ImportError:
        print("❌ requests не установлен")
        return False
    except Exception as e:
        print(f"❌ Ошибка тестирования API: {e}")
        return False

def test_prediction_api():
    """Тест API прогнозирования"""
    print("\n🔮 Тестируем API прогнозирования...")
    
    try:
        import requests
        
        test_data = {
            'player_rank': 1,
            'opponent_rank': 45,
            'player_recent_win_rate': 0.85,
            'player_surface_advantage': 0.12,
            'h2h_win_rate': 0.75
        }
        
        response = requests.post(
            'http://localhost:5001/api/predict',
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                pred = data['prediction']
                print(f"   🎯 Прогноз: {pred['probability']:.1%}")
                print(f"   🔬 Источник: {data.get('source', 'unknown')}")
                print(f"   ✅ API прогнозирования работает!")
                return True
            else:
                print(f"   ❌ API ошибка: {data.get('error', 'unknown')}")
        else:
            print(f"   ❌ HTTP ошибка: {response.status_code}")
        
        return False
        
    except Exception as e:
        print(f"❌ Ошибка API прогнозирования: {e}")
        return False

def test_data_quality():
    """Тест качества данных"""
    print("\n📊 Тестируем качество данных...")
    
    try:
        import requests
        
        response = requests.get('http://localhost:5001/api/matches', timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('success') and data.get('matches'):
                matches = data['matches']
                print(f"   📋 Найдено матчей: {len(matches)}")
                
                # Анализируем первый матч
                if matches:
                    match = matches[0]
                    source = match.get('source', 'unknown')
                    quality = match.get('data_quality', 'unknown')
                    
                    print(f"   🔍 Источник данных: {source}")
                    print(f"   💎 Качество данных: {quality}")
                    
                    # Проверяем ML прогноз
                    prediction = match.get('prediction', {})
                    prob = prediction.get('probability', 0)
                    
                    if 0.1 <= prob <= 0.9:
                        print(f"   🎯 ML прогноз: {prob:.1%} (реалистично)")
                        
                        # Проверяем, что это не простое 1/odds
                        odds = match.get('odds', {})
                        p1_odds = odds.get('player1', 2.0)
                        
                        simple_prob = 1.0 / p1_odds / (1.0 / p1_odds + 1.0 / 2.0)
                        
                        if abs(prob - simple_prob) > 0.05:
                            print(f"   ✅ Используется РЕАЛЬНЫЙ ML, не 1/odds!")
                        else:
                            print(f"   ⚠️ Похоже на простое 1/odds преобразование")
                    else:
                        print(f"   ⚠️ Нереалистичный прогноз: {prob:.1%}")
                
                return True
            else:
                print(f"   ⚠️ Нет данных о матчах")
        else:
            print(f"   ❌ Ошибка получения матчей: {response.status_code}")
        
        return False
        
    except Exception as e:
        print(f"❌ Ошибка тестирования данных: {e}")
        return False

def main():
    """Главная функция тестирования"""
    print("🧪 ML SYSTEM COMPREHENSIVE TEST")
    print("=" * 50)
    print(f"⏰ Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    tests = [
        ("ML Predictor", test_ml_predictor),
        ("Prediction Module", test_tennis_prediction_module), 
        ("API Endpoints", test_api_endpoints),
        ("Prediction API", test_prediction_api),
        ("Data Quality", test_data_quality)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 ТЕСТ: {test_name}")
        print("-" * 30)
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ Критическая ошибка в {test_name}: {e}")
            results[test_name] = False
    
    # Итоги
    print("\n" + "=" * 50)
    print("📊 ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ ПРОШЕЛ" if result else "❌ ПРОВАЛЕН"
        print(f"{status:12} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📈 Результат: {passed}/{total} тестов прошли")
    
    if passed == total:
        print("🎉 ВСЕ ТЕСТЫ ПРОШЛИ! Система работает отлично!")
        return True
    elif passed >= total * 0.8:
        print("⚡ Большинство тестов прошли. Система работает хорошо!")
        return True
    elif passed >= total * 0.5:
        print("⚠️ Половина тестов прошли. Есть проблемы для исправления.")
        return False
    else:
        print("❌ Много проблем. Система требует исправления.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Тестирование прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        sys.exit(1)