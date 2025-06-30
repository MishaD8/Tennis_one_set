#!/usr/bin/env python3
"""
🧪 Скрипт для тестирования API прогнозов
Проверяет работу всех новых endpoints
"""

import requests
import json
import time

API_BASE = 'http://localhost:5001'

def test_health_check():
    """Тест health check"""
    print("🏥 Testing health check...")
    try:
        response = requests.get(f"{API_BASE}/api/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check: {data['status']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_single_prediction():
    """Тест прогноза одного матча"""
    print("\n🎾 Testing single prediction...")
    
    test_data = {
        "player_rank": 1,
        "opponent_rank": 45,
        "player_age": 30,
        "opponent_age": 26,
        "player_recent_win_rate": 0.85,
        "player_form_trend": 0.08,
        "player_surface_advantage": 0.12,
        "h2h_win_rate": 0.75,
        "total_pressure": 3.2
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/api/predict",
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                prediction = data['prediction']
                print(f"✅ Single prediction successful!")
                print(f"   Probability: {prediction['probability']:.1%}")
                print(f"   Confidence: {prediction['confidence']}")
                print(f"   Recommendation: {prediction['recommendation']}")
                
                if 'individual_predictions' in prediction:
                    print(f"   Individual models:")
                    for model, prob in prediction['individual_predictions'].items():
                        print(f"     • {model}: {prob:.1%}")
                
                return True
            else:
                print(f"❌ Prediction failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Single prediction error: {e}")
    
    return False

def test_batch_predictions():
    """Тест множественных прогнозов"""
    print("\n🎾 Testing batch predictions...")
    
    test_data = {
        "matches": [
            {
                "player_rank": 1,
                "opponent_rank": 45,
                "player_recent_win_rate": 0.85,
                "player_surface_advantage": 0.12
            },
            {
                "player_rank": 5,
                "opponent_rank": 6,
                "player_recent_win_rate": 0.72,
                "h2h_win_rate": 0.58
            },
            {
                "player_rank": 35,
                "opponent_rank": 8,
                "player_recent_win_rate": 0.88,
                "player_surface_advantage": 0.18
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/api/predict/batch",
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                predictions = data['predictions']
                print(f"✅ Batch predictions successful!")
                print(f"   Processed {len(predictions)} matches:")
                
                for i, pred in enumerate(predictions, 1):
                    if 'probability' in pred and pred['probability'] is not None:
                        print(f"     {i}. Probability: {pred['probability']:.1%} ({pred['confidence']})")
                    else:
                        print(f"     {i}. Error: {pred.get('error', 'Unknown error')}")
                
                return True
            else:
                print(f"❌ Batch prediction failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
    
    return False

def test_existing_endpoints():
    """Тест существующих endpoints"""
    print("\n📊 Testing existing endpoints...")
    
    endpoints = [
        ('/api/stats', 'Stats'),
        ('/api/matches', 'Matches'),
        ('/api/refresh', 'Refresh')
    ]
    
    results = {}
    
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"{API_BASE}{endpoint}")
            if response.status_code == 200:
                data = response.json()
                if data.get('success', False):
                    print(f"✅ {name} endpoint working")
                    results[name] = True
                else:
                    print(f"⚠️ {name} endpoint returned success=false")
                    results[name] = False
            else:
                print(f"❌ {name} endpoint HTTP error: {response.status_code}")
                results[name] = False
        except Exception as e:
            print(f"❌ {name} endpoint error: {e}")
            results[name] = False
    
    return results

def test_dashboard():
    """Тест дашборда"""
    print("\n🌐 Testing dashboard...")
    try:
        response = requests.get(f"{API_BASE}/")
        if response.status_code == 200:
            content = response.text
            if 'Tennis Prediction Dashboard' in content:
                print("✅ Dashboard loading successfully")
                return True
            else:
                print("⚠️ Dashboard content unexpected")
        else:
            print(f"❌ Dashboard HTTP error: {response.status_code}")
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
    
    return False

def performance_test():
    """Тест производительности"""
    print("\n⚡ Performance test...")
    
    test_data = {
        "player_rank": 10,
        "opponent_rank": 15,
        "player_recent_win_rate": 0.7
    }
    
    times = []
    success_count = 0
    
    for i in range(5):
        try:
            start_time = time.time()
            response = requests.post(
                f"{API_BASE}/api/predict",
                json=test_data,
                headers={'Content-Type': 'application/json'}
            )
            end_time = time.time()
            
            if response.status_code == 200:
                times.append(end_time - start_time)
                success_count += 1
                
        except Exception as e:
            print(f"⚠️ Request {i+1} failed: {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"✅ Performance test results:")
        print(f"   Successful requests: {success_count}/5")
        print(f"   Average response time: {avg_time:.3f}s")
        print(f"   Min/Max: {min(times):.3f}s / {max(times):.3f}s")
        return True
    else:
        print("❌ All performance test requests failed")
        return False

def main():
    """Главная функция тестирования"""
    print("🧪 ТЕСТИРОВАНИЕ ИНТЕГРИРОВАННОЙ СИСТЕМЫ ПРОГНОЗИРОВАНИЯ")
    print("=" * 70)
    
    # Счетчик успешных тестов
    tests_passed = 0
    total_tests = 6
    
    # 1. Health check
    if test_health_check():
        tests_passed += 1
    
    # 2. Дашборд
    if test_dashboard():
        tests_passed += 1
    
    # 3. Существующие endpoints
    existing_results = test_existing_endpoints()
    if all(existing_results.values()):
        tests_passed += 1
    
    # 4. Прогноз одного матча
    if test_single_prediction():
        tests_passed += 1
    
    # 5. Множественные прогнозы
    if test_batch_predictions():
        tests_passed += 1
    
    # 6. Производительность
    if performance_test():
        tests_passed += 1
    
    # Итоги
    print(f"\n" + "=" * 70)
    print(f"🎾 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 70)
    print(f"✅ Пройдено тестов: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
        print("🚀 Система готова к использованию!")
    elif tests_passed >= total_tests * 0.8:
        print("⚡ Большинство тестов прошли!")
        print("💡 Незначительные проблемы требуют внимания")
    else:
        print("⚠️ Обнаружены проблемы!")
        print("🔧 Требуется диагностика системы")
    
    print(f"\n💡 СЛЕДУЮЩИЕ ШАГИ:")
    print("1. 🌐 Откройте http://localhost:5001 в браузере")
    print("2. 📊 Проверьте работу дашборда")
    print("3. 🎯 Используйте API для реальных прогнозов")
    print("4. 🔧 При необходимости настройте модели")

if __name__ == "__main__":
    main()