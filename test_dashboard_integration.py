#!/usr/bin/env python3
"""
🧪 Тестирование интеграции Tennis Dashboard
Полное тестирование всех компонентов системы
"""

import requests
import json
import time
import os
from datetime import datetime

def test_backend_endpoints():
    """Тестирование всех backend endpoints"""
    print("🔍 Тестирование backend endpoints...")
    
    base_url = "http://localhost:5001"
    endpoints = [
        ("/api/health", "Health check"),
        ("/api/stats", "System statistics"),
        ("/api/matches", "Tennis matches"),
        ("/api/refresh", "Data refresh"),
        ("/api/check-sports", "Available sports")
    ]
    
    results = {}
    
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results[endpoint] = {
                    "status": "✅ Working",
                    "response_size": len(str(data))
                }
                print(f"✅ {description}: OK")
            else:
                results[endpoint] = {
                    "status": f"❌ HTTP {response.status_code}",
                    "error": response.text[:100]
                }
                print(f"❌ {description}: HTTP {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            results[endpoint] = {
                "status": "❌ Connection refused",
                "error": "Server not running"
            }
            print(f"❌ {description}: Server not running")
            
        except Exception as e:
            results[endpoint] = {
                "status": f"❌ Error",
                "error": str(e)
            }
            print(f"❌ {description}: {str(e)}")
    
    return results

def test_prediction_api():
    """Тестирование API прогнозирования"""
    print("\n🎾 Тестирование API прогнозирования...")
    
    base_url = "http://localhost:5001"
    
    # Тест одиночного прогноза
    single_match_data = {
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
            f"{base_url}/api/predict",
            json=single_match_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                pred = data['prediction']
                print(f"✅ Single prediction: {pred['probability']:.1%} ({pred.get('confidence', 'N/A')})")
                return True
            else:
                print(f"❌ Single prediction failed: {data.get('error')}")
        else:
            print(f"❌ Single prediction: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ Single prediction error: {e}")
    
    return False

def test_dashboard_html():
    """Тестирование HTML dashboard"""
    print("\n🌐 Тестирование HTML dashboard...")
    
    if not os.path.exists("web_dashboard.html"):
        print("❌ HTML файл не найден")
        return False
    
    try:
        with open("web_dashboard.html", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Проверяем ключевые элементы
        checks = {
            "API calls": "localhost:5001" in content,
            "Match cards": "match-card" in content,
            "Filters": "filter-group" in content,
            "Stats": "stats-grid" in content,
            "JavaScript": "fetchMatches" in content
        }
        
        all_ok = True
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"{status} {check}: {'Found' if passed else 'Missing'}")
            if not passed:
                all_ok = False
        
        return all_ok
        
    except Exception as e:
        print(f"❌ Ошибка анализа HTML: {e}")
        return False

def check_system_files():
    """Проверка системных файлов"""
    print("\n📁 Проверка системных файлов...")
    
    files_to_check = [
        ("web_backend.py", "Backend server"),
        ("web_dashboard.html", "HTML dashboard"),
        ("tennis_prediction_module.py", "Prediction module"),
        ("launch_dashboard.py", "Dashboard launcher"),
        ("quick_status.py", "Quick status check")
    ]
    
    all_present = True
    
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {description}: {size:,} bytes")
        else:
            print(f"❌ {description}: Missing")
            all_present = False
    
    return all_present

def performance_test():
    """Тест производительности"""
    print("\n⚡ Тест производительности...")
    
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
                "http://localhost:5001/api/predict",
                json=test_data,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            end_time = time.time()
            
            if response.status_code == 200:
                times.append(end_time - start_time)
                success_count += 1
                
        except Exception as e:
            print(f"⚠️ Request {i+1} failed: {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"✅ Performance results:")
        print(f"   Successful requests: {success_count}/5")
        print(f"   Average response time: {avg_time:.3f}s")
        print(f"   Min/Max: {min(times):.3f}s / {max(times):.3f}s")
        return True
    else:
        print("❌ All performance test requests failed")
        return False

def main():
    """Главная функция тестирования"""
    print("🧪 ПОЛНОЕ ТЕСТИРОВАНИЕ TENNIS DASHBOARD")
    print("="*70)
    print(f"⏰ Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Счетчик тестов
    tests_passed = 0
    total_tests = 5
    
    # 1. Проверка файлов
    if check_system_files():
        tests_passed += 1
    
    # 2. Тестирование HTML
    if test_dashboard_html():
        tests_passed += 1
    
    # 3. Тестирование backend endpoints
    backend_results = test_backend_endpoints()
    backend_ok = all("✅" in str(result.get("status", "")) for result in backend_results.values())
    if backend_ok:
        tests_passed += 1
    
    # 4. Тестирование API прогнозирования
    if test_prediction_api():
        tests_passed += 1
    
    # 5. Тест производительности
    if performance_test():
        tests_passed += 1
    
    # Итоги
    print(f"\n" + "="*70)
    print("📊 ИТОГИ ТЕСТИРОВАНИЯ:")
    print("="*70)
    print(f"✅ Пройдено тестов: {tests_passed}/{total_tests}")
    print(f"📈 Процент успеха: {(tests_passed/total_tests)*100:.0f}%")
    
    if tests_passed == total_tests:
        print("🎉 ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
        print("🚀 Система полностью готова к использованию!")
    elif tests_passed >= 3:
        print("⚡ Система работает хорошо!")
        print("💡 Некоторые компоненты требуют внимания")
    else:
        print("⚠️ Обнаружены проблемы!")
        print("🔧 Требуется диагностика")
    
    print(f"\n🌐 ДОСТУП К СИСТЕМЕ:")
    print("• Backend: http://localhost:5001")
    print("• Dashboard: web_dashboard.html")
    print("• Health check: http://localhost:5001/api/health")
    
    return tests_passed >= 3

if __name__ == "__main__":
    success = main()
