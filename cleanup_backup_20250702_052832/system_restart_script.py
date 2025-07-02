#!/usr/bin/env python3
"""
🔄 RESTART TENNIS SYSTEM
Останавливает старый backend и запускает исправленную версию
"""

import subprocess
import sys
import time
import psutil
import signal
import os

def kill_processes_on_port(port):
    """Убивает процессы на указанном порту"""
    print(f"🔍 Ищем процессы на порту {port}...")
    
    killed = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Проверяем соединения процесса
            connections = proc.connections()
            for conn in connections:
                if conn.laddr.port == port:
                    print(f"⚠️ Найден процесс: PID {proc.info['pid']} - {proc.info['name']}")
                    
                    # Пытаемся завершить процесс gracefully
                    try:
                        proc.terminate()
                        proc.wait(timeout=5)
                        print(f"✅ Процесс PID {proc.info['pid']} завершен")
                        killed += 1
                    except psutil.TimeoutExpired:
                        # Принудительное завершение
                        proc.kill()
                        print(f"💀 Процесс PID {proc.info['pid']} принудительно завершен")
                        killed += 1
                    break
                        
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception as e:
            print(f"⚠️ Ошибка проверки процесса: {e}")
    
    if killed > 0:
        print(f"✅ Завершено {killed} процессов на порту {port}")
        time.sleep(2)  # Даем время на освобождение порта
    else:
        print(f"ℹ️ Процессы на порту {port} не найдены")
    
    return killed

def check_files():
    """Проверяем наличие нужных файлов"""
    print("📋 Проверяем файлы...")
    
    required = [
        'web_backend_fixed_ml.py',
        'test_ml_system.py', 
        'real_tennis_predictor_integration.py'
    ]
    
    missing = []
    for file in required:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing.append(file)
    
    if missing:
        print(f"\n⚠️ Отсутствуют файлы: {missing}")
        print("💡 Создайте их из артефактов перед перезапуском")
        return False
    
    return True

def start_fixed_backend():
    """Запускаем исправленный backend"""
    print("🚀 Запускаем исправленный backend...")
    
    try:
        process = subprocess.Popen(
            [sys.executable, 'web_backend_fixed_ml.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print(f"✅ Backend запущен (PID: {process.pid})")
        
        # Ждем несколько секунд и проверяем что процесс жив
        time.sleep(3)
        
        if process.poll() is None:
            print("✅ Backend работает стабильно")
            return process
        else:
            print("❌ Backend завершился неожиданно")
            stdout, stderr = process.communicate()
            if stderr:
                print(f"Ошибки: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Ошибка запуска: {e}")
        return None

def test_system():
    """Тестируем систему"""
    print("\n🧪 Тестируем исправленную систему...")
    
    if not os.path.exists('test_ml_system.py'):
        print("⚠️ test_ml_system.py не найден, пропускаем тестирование")
        return
    
    try:
        result = subprocess.run(
            [sys.executable, 'test_ml_system.py'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("✅ Все тесты прошли!")
        else:
            print("⚠️ Есть проблемы в тестах:")
            print(result.stdout)
            if result.stderr:
                print("Ошибки:", result.stderr)
    
    except subprocess.TimeoutExpired:
        print("❌ Тестирование прервано по таймауту")
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")

def main():
    """Главная функция перезапуска"""
    print("🔄 RESTART TENNIS SYSTEM")
    print("=" * 40)
    print("🎯 Цель: Заменить простые коэффициенты на РЕАЛЬНЫЙ ML")
    print("=" * 40)
    
    # 1. Проверяем файлы
    if not check_files():
        print("\n❌ Не хватает файлов для перезапуска!")
        return False
    
    # 2. Останавливаем старые процессы
    print(f"\n⏹️ Останавливаем процессы на порту 5001...")
    killed = kill_processes_on_port(5001)
    
    # 3. Запускаем исправленный backend
    print(f"\n🚀 Запускаем исправленный backend...")
    process = start_fixed_backend()
    
    if not process:
        print("❌ Не удалось запустить исправленный backend")
        return False
    
    # 4. Ждем готовности
    print("\n⏳ Ждем готовности системы...")
    time.sleep(5)
    
    # 5. Тестируем
    test_system()
    
    # 6. Показываем результат
    print("\n" + "=" * 50)
    print("🎉 СИСТЕМА ПЕРЕЗАПУЩЕНА С ИСПРАВЛЕНИЯМИ!")
    print("=" * 50)
    print("🌐 Dashboard: http://localhost:5001")
    print("🤖 Теперь используется РЕАЛЬНЫЙ ML вместо 1/odds")
    print("🎯 Проверьте прогнозы - они должны быть умнее!")
    print("=" * 50)
    
    # 7. Держим систему работающей
    try:
        print("\n⌨️ Нажмите Ctrl+C для остановки системы")
        while process.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⏹️ Останавливаем систему...")
        process.terminate()
        process.wait()
        print("✅ Система остановлена")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        sys.exit(1)