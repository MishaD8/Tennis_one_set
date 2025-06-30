#!/usr/bin/env python3
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
            
            print("\n🎯 СИСТЕМА ЗАПУЩЕНА!")
            print("=" * 30)
            print("🌐 Dashboard: web_dashboard.html (открыт в браузере)")
            print("📡 Backend API: http://localhost:5001")
            print("🔍 Health check: http://localhost:5001/api/health")
            print("\n⏹️ Нажмите Ctrl+C для остановки")
            
            # Ждем завершения
            backend_process.wait()
        else:
            print("❌ Не удалось запустить backend")
            backend_process.terminate()
            
    except KeyboardInterrupt:
        print("\n⏹️ Остановка системы...")
        backend_process.terminate()
        backend_process.wait()
        print("✅ Система остановлена")

if __name__ == "__main__":
    main()
