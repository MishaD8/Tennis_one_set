#!/usr/bin/env python3
"""
⚡ Быстрая диагностика системы
Проверяет статус всех компонентов за несколько секунд
"""

import os
import sys
import requests
from datetime import datetime

def check_files():
    """Быстрая проверка файлов"""
    files = {
        "web_backend.py": "Backend сервер",
        "web_dashboard.html": "HTML Dashboard", 
        "tennis_prediction_module.py": "Модуль прогнозов",
        "launch_dashboard.py": "Launcher",
        "test_dashboard_integration.py": "Тестирование"
    }
    
    print("📁 ФАЙЛЫ:")
    all_present = True
    for file, desc in files.items():
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {desc}: {size:,} байт")
        else:
            print(f"❌ {desc}: ОТСУТСТВУЕТ")
            all_present = False
    
    return all_present

def check_backend():
    """Проверка backend сервера"""
    print("\n🖥️ BACKEND:")
    try:
        response = requests.get("http://localhost:5001/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Сервер работает: {data.get('status', 'OK')}")
            
            # Проверяем ключевые endpoints
            endpoints = ["/api/stats", "/api/matches", "/api/check-sports"]
            working_endpoints = 0
            
            for endpoint in endpoints:
                try:
                    resp = requests.get(f"http://localhost:5001{endpoint}", timeout=3)
                    if resp.status_code == 200:
                        working_endpoints += 1
                except:
                    pass
            
            print(f"📡 API endpoints: {working_endpoints}/{len(endpoints)} работают")
            return True
            
    except requests.exceptions.ConnectionError:
        print("❌ Сервер не запущен")
        print("   Запустите: python web_backend.py")
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
    
    return False

def check_prediction_module():
    """Проверка модуля прогнозов"""
    print("\n🧠 МОДУЛЬ ПРОГНОЗОВ:")
    try:
        from tennis_prediction_module import TennisPredictionService
        service = TennisPredictionService()
        
        if service.load_models():
            print("✅ Модели загружены")
            return True
        else:
            print("⚠️ Работает в demo режиме (модели не найдены)")
            return True
            
    except ImportError:
        print("❌ Модуль не найден")
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
    
    return False

def check_dashboard_config():
    """Проверка конфигурации dashboard"""
    print("\n🌐 DASHBOARD:")
    
    if not os.path.exists("web_dashboard.html"):
        print("❌ HTML файл не найден")
        return False
    
    try:
        with open("web_dashboard.html", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Проверяем порт API
        if "localhost:5001" in content:
            print("✅ Порт API настроен правильно (5001)")
        elif "localhost:5000" in content:
            print("⚠️ Неправильный порт API (5000 → нужен 5001)")
            print("   Исправьте: python fix_dashboard_ports.py")
            return False
        else:
            print("⚠️ API URL не найден")
            return False
        
        # Проверяем ключевые функции
        required_functions = ["fetchMatches", "fetchStats", "refreshData"]
        missing_functions = [f for f in required_functions if f not in content]
        
        if missing_functions:
            print(f"❌ Отсутствуют функции: {', '.join(missing_functions)}")
            return False
        else:
            print("✅ Все функции найдены")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка анализа: {e}")
        return False

def test_api_quickly():
    """Быстрый тест API"""
    print("\n🧪 БЫСТРЫЙ ТЕСТ API:")
    
    # Тест данных для прогноза
    test_data = {
        "player_rank": 1,
        "opponent_rank": 45,
        "player_recent_win_rate": 0.85,
        "player_surface_advantage": 0.12,
        "h2h_win_rate": 0.75
    }
    
    try:
        response = requests.post(
            "http://localhost:5001/api/predict",
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                prob = data['prediction']['probability']
                conf = data['prediction']['confidence']
                print(f"✅ Прогноз работает: {prob:.1%} ({conf})")
                return True
            else:
                print(f"❌ Ошибка прогноза: {data.get('error')}")
        else:
            print(f"❌ HTTP {response.status_code}: {response.text[:100]}")
            
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
    
    return False

def show_system_info():
    """Показывает информацию о системе"""
    print("\n💻 СИСТЕМА:")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"📁 Директория: {os.getcwd()}")
    print(f"⏰ Время: {datetime.now().strftime('%H:%M:%S')}")

def show_usage_instructions():
    """Показывает инструкции по использованию"""
    print(f"\n" + "="*60)
    print("🎯 ИНСТРУКЦИИ ПО ЗАПУСКУ:")
    print("="*60)
    
    print("\n🚀 БЫСТРЫЙ ЗАПУСК:")
    print("   python launch_dashboard.py")
    
    print("\n📋 РУЧНОЙ ЗАПУСК:")
    print("   1. python web_backend.py")
    print("   2. Откройте web_dashboard.html в браузере")
    
    print("\n🔧 ДИАГНОСТИКА:")
    print("   python quick_status.py       # Быстрая проверка")
    print("   python test_dashboard_integration.py  # Полное тестирование")
    print("   python fix_dashboard_ports.py         # Исправление портов")
    
    print("\n🌐 ДОСТУП:")
    print("   • Backend API: http://localhost:5001")
    print("   • Health check: http://localhost:5001/api/health")
    print("   • Dashboard: web_dashboard.html")

def main():
    """Главная функция быстрой диагностики"""
    print("⚡ БЫСТРАЯ ДИАГНОСТИКА TENNIS DASHBOARD")
    print("="*60)
    
    show_system_info()
    
    # Проверяем компоненты
    files_ok = check_files()
    backend_ok = check_backend()
    prediction_ok = check_prediction_module()
    dashboard_ok = check_dashboard_config()
    
    # Если backend работает, тестируем API
    api_ok = False
    if backend_ok:
        api_ok = test_api_quickly()
    
    # Подсчитываем результат
    total_checks = 5
    passed_checks = sum([files_ok, backend_ok, prediction_ok, dashboard_ok, api_ok])
    
    print(f"\n" + "="*60)
    print("📊 ИТОГ:")
    print(f"✅ Пройдено проверок: {passed_checks}/{total_checks}")
    print(f"📈 Готовность системы: {(passed_checks/total_checks)*100:.0f}%")
    
    # Определяем статус системы
    if passed_checks == total_checks:
        print("🎉 СИСТЕМА ПОЛНОСТЬЮ ГОТОВА К РАБОТЕ!")
    elif passed_checks >= 3:
        print("⚠️ СИСТЕМА ЧАСТИЧНО ГОТОВА")
        if not backend_ok:
            print("   🚀 Запустите backend: python web_backend.py")
        if not dashboard_ok:
            print("   🔧 Исправьте dashboard: python fix_dashboard_ports.py")
    else:
        print("❌ СИСТЕМА НЕ ГОТОВА")
        print("   📋 Запустите полную диагностику: python test_dashboard_integration.py")
    
    show_usage_instructions()
    
    return passed_checks >= 3

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)