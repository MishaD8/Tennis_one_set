#!/usr/bin/env python3
"""
🧪 Быстрый тест Tennis API
Проверяем все основные endpoints вашего сервера
"""

import requests
import json
import time

def test_api():
    """Тестируем ваш Tennis API"""
    
    BASE_URL = "http://65.109.135.2:5001"
    
    print("🎾 ТЕСТИРОВАНИЕ TENNIS API")
    print("=" * 50)
    print(f"🌐 Базовый URL: {BASE_URL}")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 0
    
    # Тест 1: Health Check
    print("\n1️⃣ Тест Health Check")
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Check: {data.get('status', 'Unknown')}")
            print(f"   Prediction Service: {'✅' if data.get('prediction_service') else '❌'}")
            print(f"   Models Loaded: {'✅' if data.get('models_loaded') else '❌'}")
            tests_passed += 1
        else:
            print(f"❌ Health Check failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Health Check error: {e}")
    total_tests += 1
    
    # Тест 2: Stats
    print("\n2️⃣ Тест Stats API")
    try:
        response = requests.get(f"{BASE_URL}/api/stats", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                stats = data.get('stats', {})
                print(f"✅ Stats API работает")
                print(f"   Total matches: {stats.get('total_matches', 'N/A')}")
                print(f"   Accuracy: {stats.get('accuracy_rate', 'N/A')}")
                tests_passed += 1
            else:
                print(f"❌ Stats API: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ Stats API failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Stats API error: {e}")
    total_tests += 1
    
    # Тест 3: Matches
    print("\n3️⃣ Тест Matches API")
    try:
        response = requests.get(f"{BASE_URL}/api/matches", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                matches = data.get('matches', [])
                print(f"✅ Matches API работает")
                print(f"   Найдено матчей: {len(matches)}")
                if matches:
                    match = matches[0]
                    print(f"   Пример: {match.get('player1', 'N/A')} vs {match.get('player2', 'N/A')}")
                tests_passed += 1
            else:
                print(f"❌ Matches API: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ Matches API failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Matches API error: {e}")
    total_tests += 1
    
    # Тест 4: Prediction
    print("\n4️⃣ Тест Prediction API")
    try:
        test_data = {
            "player_rank": 1,
            "opponent_rank": 45,
            "player_recent_win_rate": 0.85,
            "player_surface_advantage": 0.12,
            "h2h_win_rate": 0.75
        }
        
        response = requests.post(
            f"{BASE_URL}/api/predict",
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                pred = data.get('prediction', {})
                print(f"✅ Prediction API работает")
                print(f"   Вероятность: {pred.get('probability', 'N/A')}")
                print(f"   Уверенность: {pred.get('confidence', 'N/A')}")
                print(f"   Источник: {data.get('source', 'N/A')}")
                tests_passed += 1
            else:
                print(f"❌ Prediction API: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ Prediction API failed: HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"❌ Prediction API error: {e}")
    total_tests += 1
    
    # Тест 5: Dashboard
    print("\n5️⃣ Тест Dashboard")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=10)
        if response.status_code == 200:
            content = response.text
            if "Tennis Analytics Dashboard" in content:
                print(f"✅ Dashboard загружается")
                print(f"   Размер контента: {len(content):,} символов")
                tests_passed += 1
            else:
                print(f"❌ Dashboard: неожиданный контент")
        else:
            print(f"❌ Dashboard failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
    total_tests += 1
    
    # Итоги
    print("\n" + "=" * 50)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 50)
    print(f"✅ Пройдено тестов: {tests_passed}/{total_tests}")
    print(f"📈 Процент успеха: {(tests_passed/total_tests)*100:.0f}%")
    
    if tests_passed == total_tests:
        print("🎉 ВСЕ ТЕСТЫ ПРОШЛИ! Система работает отлично!")
        print("✅ Можно смело удалять тестовые файлы")
    elif tests_passed >= 3:
        print("⚡ Основные функции работают!")
        print("💡 Незначительные проблемы не критичны")
    else:
        print("⚠️ Есть серьезные проблемы!")
        print("🔧 Лучше не удалять файлы до исправления")
    
    return tests_passed >= 3

if __name__ == "__main__":
    success = test_api()
    print(f"\n🏁 Тестирование завершено: {'SUCCESS' if success else 'FAILED'}")