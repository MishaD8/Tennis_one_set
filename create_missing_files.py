#!/usr/bin/env python3
"""
📦 Создание недостающих файлов для Tennis Dashboard
Создает launch_dashboard.py и test_dashboard_integration.py
"""

import os

def create_launcher():
    """Создает launch_dashboard.py"""
    
    launcher_content = '''#!/usr/bin/env python3
"""
🚀 Tennis Dashboard Launcher
Автоматически запускает backend и открывает dashboard
"""

import subprocess
import webbrowser
import time
import os
import sys
import requests
from threading import Thread

def start_backend():
    """Запуск backend сервера"""
    try:
        print("🚀 Запуск backend сервера...")
        process = subprocess.Popen([
            sys.executable, "web_backend.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return process
    except Exception as e:
        print(f"❌ Ошибка запуска backend: {e}")
        return None

def wait_for_backend(max_attempts=30):
    """Ожидание запуска backend"""
    print("⏳ Ожидание запуска backend...")
    
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:5001/api/health", timeout=2)
            if response.status_code == 200:
                print("✅ Backend запущен успешно!")
                return True
        except:
            pass
        
        print(f"⏳ Попытка {attempt + 1}/{max_attempts}...")
        time.sleep(2)
    
    print("❌ Backend не отвечает")
    return False

def open_dashboard():
    """Открытие dashboard в браузере"""
    dashboard_path = os.path.abspath("web_dashboard.html")
    
    if os.path.exists(dashboard_path):
        print("🌐 Открытие dashboard...")
        webbrowser.open(f"file://{dashboard_path}")
        print(f"✅ Dashboard открыт: {dashboard_path}")
    else:
        print("❌ Файл web_dashboard.html не найден")

def main():
    print("🎾 TENNIS DASHBOARD LAUNCHER")
    print("=" * 50)
    
    # Проверяем наличие файлов
    required_files = ["web_backend.py", "web_dashboard.html"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Отсутствуют файлы: {', '.join(missing_files)}")
        return
    
    # Запускаем backend
    backend_process = start_backend()
    if not backend_process:
        return
    
    try:
        # Ждем запуска backend
        if wait_for_backend():
            # Открываем dashboard
            open_dashboard()
            
            print("\\n🎯 СИСТЕМА ЗАПУЩЕНА!")
            print("=" * 30)
            print("🌐 Dashboard: web_dashboard.html (открыт в браузере)")
            print("📡 Backend API: http://localhost:5001")
            print("🔍 Health check: http://localhost:5001/api/health")
            print("\\n⏹️ Нажмите Ctrl+C для остановки")
            
            # Ждем завершения
            backend_process.wait()
        else:
            print("❌ Не удалось запустить backend")
            backend_process.terminate()
            
    except KeyboardInterrupt:
        print("\\n⏹️ Остановка системы...")
        backend_process.terminate()
        backend_process.wait()
        print("✅ Система остановлена")

if __name__ == "__main__":
    main()
'''
    
    try:
        with open("launch_dashboard.py", "w", encoding="utf-8") as f:
            f.write(launcher_content)
        
        # Делаем файл исполняемым
        try:
            os.chmod("launch_dashboard.py", 0o755)
        except:
            pass
        
        print("✅ Создан launch_dashboard.py")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка создания launcher: {e}")
        return False

def create_test_integration():
    """Создает test_dashboard_integration.py"""
    
    test_content = '''#!/usr/bin/env python3
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
    print("\\n🎾 Тестирование API прогнозирования...")
    
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
    print("\\n🌐 Тестирование HTML dashboard...")
    
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
    print("\\n📁 Проверка системных файлов...")
    
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
    print("\\n⚡ Тест производительности...")
    
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
    print(f"\\n" + "="*70)
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
    
    print(f"\\n🌐 ДОСТУП К СИСТЕМЕ:")
    print("• Backend: http://localhost:5001")
    print("• Dashboard: web_dashboard.html")
    print("• Health check: http://localhost:5001/api/health")
    
    return tests_passed >= 3

if __name__ == "__main__":
    success = main()
'''
    
    try:
        with open("test_dashboard_integration.py", "w", encoding="utf-8") as f:
            f.write(test_content)
        
        # Делаем файл исполняемым
        try:
            os.chmod("test_dashboard_integration.py", 0o755)
        except:
            pass
        
        print("✅ Создан test_dashboard_integration.py")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка создания тестов: {e}")
        return False

def main():
    """Главная функция"""
    print("📦 СОЗДАНИЕ НЕДОСТАЮЩИХ ФАЙЛОВ")
    print("=" * 50)
    
    files_created = 0
    
    # 1. Создаем launcher
    print("1️⃣ Создание launcher...")
    if create_launcher():
        files_created += 1
    
    # 2. Создаем тесты
    print("\\n2️⃣ Создание тестов...")
    if create_test_integration():
        files_created += 1
    
    print(f"\\n✅ Создано файлов: {files_created}/2")
    
    if files_created == 2:
        print("\\n🎯 ГОТОВО! Теперь доступны:")
        print("🚀 python launch_dashboard.py      # Автозапуск системы")
        print("🧪 python test_dashboard_integration.py  # Полное тестирование")
        print("⚡ python quick_status.py          # Быстрая проверка")
        
        print("\\n💡 Рекомендации:")
        print("1. Запустите: python quick_status.py")
        print("2. Если все ОК: python launch_dashboard.py")
        print("3. Для детального тестирования: python test_dashboard_integration.py")
    else:
        print("\\n⚠️ Не все файлы созданы. Проверьте ошибки выше.")

if __name__ == "__main__":
    main()