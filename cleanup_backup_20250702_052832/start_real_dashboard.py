#!/usr/bin/env python3
"""
🚀 Быстрый запуск Tennis Dashboard с реальными данными
"""

import subprocess
import sys
import time
import webbrowser
import os

def start_backend():
    """Запуск backend сервера"""
    backend_files = [
        'web_backend_with_dashboard.py',
        'web_backend.py'
    ]
    
    for backend_file in backend_files:
        if os.path.exists(backend_file):
            print(f"🚀 Запускаем {backend_file}...")
            return subprocess.Popen([sys.executable, backend_file])
    
    print("❌ Backend файлы не найдены!")
    return None

def main():
    print("🎾 ЗАПУСК TENNIS DASHBOARD С РЕАЛЬНЫМИ ДАННЫМИ")
    print("=" * 60)
    
    # Запускаем backend
    process = start_backend()
    
    if process:
        print("⏰ Ждем запуска сервера...")
        time.sleep(5)
        
        print("🌐 Открываем браузер...")
        webbrowser.open("http://localhost:5001")
        
        print("✅ Dashboard запущен!")
        print("📱 URL: http://localhost:5001")
        print("🎾 Теперь вы видите РЕАЛЬНЫЕ матчи Wimbledon 2025!")
        print("⏹️ Нажмите Ctrl+C для остановки")
        
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n⏹️ Остановка сервера...")
            process.terminate()
            process.wait()
            print("✅ Сервер остановлен")
    else:
        print("❌ Не удалось запустить backend")

if __name__ == "__main__":
    main()
