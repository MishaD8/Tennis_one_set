#!/usr/bin/env python3
"""
🚀 TENNIS SYSTEM LAUNCHER
Простой launcher для тестирования всей исправленной системы
"""

import subprocess
import sys
import time
import webbrowser
import os
from datetime import datetime
import signal

class TennisSystemLauncher:
    """Launcher для теннисной системы"""
    
    def __init__(self):
        self.backend_process = None
        self.current_backend = None
        
    def check_files(self):
        """Проверка наличия необходимых файлов"""
        print("🔍 Проверяем файлы системы...")
        
        required_files = {
            'Backend files': [
                'web_backend_fixed_ml.py',
                'backend_integration_fix.py', 
                'web_backend_with_dashboard.py'
            ],
            'ML Integration': [
                'real_tennis_predictor_integration.py',
                'tennis_prediction_module.py'
            ],
            'Test files': [
                'test_ml_system.py'
            ],
            'Config': [
                'config.json'
            ]
        }
        
        all_files_status = {}
        for category, files in required_files.items():
            print(f"\n📂 {category}:")
            category_status = {}
            
            for file in files:
                exists = os.path.exists(file)
                status = "✅" if exists else "❌"
                print(f"  {status} {file}")
                category_status[file] = exists
            
            all_files_status[category] = category_status
        
        return all_files_status
    
    def run_integration_fix(self):
        """Запуск исправления интеграции"""
        print("\n🔧 Запускаем исправление backend...")
        
        if not os.path.exists('backend_integration_fix.py'):
            print("❌ backend_integration_fix.py не найден!")
            return False
        
        try:
            result = subprocess.run(
                [sys.executable, 'backend_integration_fix.py'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("✅ Backend исправлен успешно!")
                if result.stdout:
                    print("📋 Результат:")
                    print(result.stdout)
                return True
            else:
                print(f"❌ Ошибка исправления (код {result.returncode})")
                if result.stderr:
                    print("❌ Ошибки:")
                    print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Превышено время ожидания исправления")
            return False
        except Exception as e:
            print(f"❌ Ошибка запуска исправления: {e}")
            return False
    
    def test_ml_system(self):
        """Тестирование ML системы"""
        print("\n🧪 Тестируем ML систему...")
        
        if not os.path.exists('test_ml_system.py'):
            print("❌ test_ml_system.py не найден!")
            return False
        
        try:
            result = subprocess.run(
                [sys.executable, 'test_ml_system.py'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print("✅ ML система работает!")
                print("📋 Результат теста:")
                print(result.stdout)
                return True
            else:
                print(f"⚠️ Тест ML системы выявил проблемы (код {result.returncode})")
                if result.stderr:
                    print("⚠️ Детали:")
                    print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Превышено время ожидания теста")
            return False
        except Exception as e:
            print(f"❌ Ошибка запуска теста: {e}")
            return False
    
    def find_best_backend(self):
        """Поиск лучшего доступного backend"""
        backends = [
            ('web_backend_fixed_ml.py', 'Исправленный ML backend'),
            ('web_backend_with_dashboard.py', 'Backend с dashboard'),
            ('web_backend_minimal.py', 'Минимальный backend'),
            ('web_backend.py', 'Базовый backend')
        ]
        
        for backend_file, description in backends:
            if os.path.exists(backend_file):
                print(f"🎯 Найден: {description}")
                return backend_file, description
        
        print("❌ Ни один backend не найден!")
        return None, None
    
    def start_backend(self, backend_file=None):
        """Запуск backend сервера"""
        if backend_file is None:
            backend_file, description = self.find_best_backend()
            if backend_file is None:
                return False
        else:
            description = backend_file
        
        print(f"\n🚀 Запускаем {description}...")
        
        try:
            self.backend_process = subprocess.Popen(
                [sys.executable, backend_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.current_backend = backend_file
            print(f"✅ Backend запущен (PID: {self.backend_process.pid})")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка запуска backend: {e}")
            return False
    
    def wait_for_backend(self, timeout=15):
        """Ожидание готовности backend"""
        print("⏳ Ожидаем готовности backend...")
        
        import requests
        
        for i in range(timeout):
            try:
                response = requests.get('http://localhost:5001/api/health', timeout=2)
                if response.status_code == 200:
                    print("✅ Backend готов!")
                    return True
            except:
                pass
            
            print(f"⏳ Ожидание... {i+1}/{timeout}")
            time.sleep(1)
        
        print("⚠️ Backend не отвечает, но продолжаем...")
        return False
    
    def open_dashboard(self):
        """Открытие dashboard в браузере"""
        print("\n🌐 Открываем dashboard...")
        try:
            webbrowser.open('http://localhost:5001')
            print("✅ Dashboard открыт в браузере")
            return True
        except Exception as e:
            print(f"❌ Ошибка открытия браузера: {e}")
            print("🌐 Откройте вручную: http://localhost:5001")
            return False
    
    def show_status(self):
        """Показ статуса системы"""
        print("\n📊 СТАТУС СИСТЕМЫ:")
        print("-" * 40)
        
        if self.backend_process:
            if self.backend_process.poll() is None:
                print(f"🟢 Backend: Работает ({self.current_backend})")
                print(f"🆔 PID: {self.backend_process.pid}")
            else:
                print(f"🔴 Backend: Остановлен (код {self.backend_process.returncode})")
        else:
            print("🔴 Backend: Не запущен")
        
        print(f"🌐 Dashboard: http://localhost:5001")
        print(f"📡 API: http://localhost:5001/api/*")
        print(f"⏰ Время: {datetime.now().strftime('%H:%M:%S')}")
    
    def stop_backend(self):
        """Остановка backend"""
        if self.backend_process:
            print("\n⏹️ Останавливаем backend...")
            try:
                self.backend_process.terminate()
                self.backend_process.wait(timeout=10)
                print("✅ Backend остановлен")
            except subprocess.TimeoutExpired:
                print("⚠️ Принудительная остановка...")
                self.backend_process.kill()
                self.backend_process.wait()
                print("✅ Backend принудительно остановлен")
            except Exception as e:
                print(f"❌ Ошибка остановки: {e}")
            finally:
                self.backend_process = None
                self.current_backend = None
    
    def interactive_menu(self):
        """Интерактивное меню"""
        while True:
            print("\n" + "="*50)
            print("🎾 TENNIS SYSTEM LAUNCHER - МЕНЮ")
            print("="*50)
            print("1. 📋 Проверить файлы")
            print("2. 🔧 Исправить backend")
            print("3. 🧪 Тестировать ML систему")
            print("4. 🚀 Запустить backend")
            print("5. 🌐 Открыть dashboard")
            print("6. 📊 Показать статус")
            print("7. ⏹️ Остановить backend")
            print("8. 🔄 Полный перезапуск")
            print("0. 👋 Выход")
            
            try:
                choice = input("\nВыберите действие (0-8): ").strip()
                
                if choice == '0':
                    print("👋 До свидания!")
                    break
                elif choice == '1':
                    self.check_files()
                elif choice == '2':
                    self.run_integration_fix()
                elif choice == '3':
                    self.test_ml_system()
                elif choice == '4':
                    self.start_backend()
                    self.wait_for_backend()
                elif choice == '5':
                    self.open_dashboard()
                elif choice == '6':
                    self.show_status()
                elif choice == '7':
                    self.stop_backend()
                elif choice == '8':
                    self.stop_backend()
                    time.sleep(2)
                    self.start_backend()
                    self.wait_for_backend()
                    self.open_dashboard()
                else:
                    print("❌ Неверный выбор")
                
                if choice != '0':
                    input("\nНажмите Enter для продолжения...")
                    
            except KeyboardInterrupt:
                print("\n\n⏹️ Прерывание...")
                break
            except Exception as e:
                print(f"\n❌ Ошибка: {e}")
                input("Нажмите Enter для продолжения...")
    
    def auto_launch(self):
        """Автоматический запуск системы"""
        print("🎾 АВТОМАТИЧЕСКИЙ ЗАПУСК TENNIS SYSTEM")
        print("=" * 50)
        
        # 1. Проверяем файлы
        files_status = self.check_files()
        
        # 2. Исправляем backend если нужно
        if os.path.exists('backend_integration_fix.py'):
            if self.run_integration_fix():
                print("✅ Backend исправлен")
            else:
                print("⚠️ Проблемы с исправлением, продолжаем...")
        
        # 3. Тестируем ML систему
        if os.path.exists('test_ml_system.py'):
            if self.test_ml_system():
                print("✅ ML система протестирована")
            else:
                print("⚠️ Проблемы с ML системой, продолжаем...")
        
        # 4. Запускаем backend
        if self.start_backend():
            print("✅ Backend запущен")
            
            # 5. Ждем готовности
            self.wait_for_backend()
            
            # 6. Открываем dashboard
            self.open_dashboard()
            
            # 7. Показываем статус
            self.show_status()
            
            print("\n🎉 СИСТЕМА ЗАПУЩЕНА УСПЕШНО!")
            print("\n💡 Что доступно:")
            print("  • 🌐 Dashboard: http://localhost:5001")
            print("  • 📡 API: http://localhost:5001/api/health")
            print("  • 🎾 ML прогнозы с реальными данными")
            print("  • 🎯 Value betting анализ")
            
            print("\n⌨️ Нажмите Ctrl+C для остановки или Enter для меню")
            
            try:
                input()
                self.interactive_menu()
            except KeyboardInterrupt:
                print("\n⏹️ Останавливаем систему...")
            finally:
                self.stop_backend()
        else:
            print("❌ Не удалось запустить backend")
            print("💡 Попробуйте интерактивное меню")
            self.interactive_menu()
    
    def __del__(self):
        """Cleanup при завершении"""
        self.stop_backend()


def signal_handler(signum, frame):
    """Обработчик сигналов для graceful shutdown"""
    print("\n⏹️ Получен сигнал остановки...")
    sys.exit(0)


def main():
    """Главная функция"""
    # Настраиваем обработчик сигналов
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    launcher = TennisSystemLauncher()
    
    print("🎾 TENNIS SYSTEM LAUNCHER")
    print("=" * 50)
    print("🚀 Запуск и тестирование исправленной системы")
    print("🤖 ML прогнозы вместо простых коэффициентов")
    print("🎯 Реальные данные игроков и турниров")
    print("=" * 50)
    
    # Проверяем аргументы командной строки
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ['auto', 'start', 'run']:
            launcher.auto_launch()
        elif arg == 'menu':
            launcher.interactive_menu()
        elif arg == 'test':
            launcher.check_files()
            launcher.test_ml_system()
        elif arg == 'fix':
            launcher.run_integration_fix()
        else:
            print(f"❌ Неизвестный аргумент: {arg}")
            print("💡 Используйте: auto, menu, test, fix")
    else:
        # Интерактивный выбор режима
        print("\n🔧 Выберите режим запуска:")
        print("1. 🚀 Автоматический (рекомендуется)")
        print("2. 📋 Интерактивное меню") 
        print("3. 🧪 Только тестирование")
        
        try:
            choice = input("\nВыберите (1-3): ").strip()
            
            if choice == '1':
                launcher.auto_launch()
            elif choice == '2':
                launcher.interactive_menu()
            elif choice == '3':
                launcher.check_files()
                launcher.test_ml_system()
            else:
                print("❌ Неверный выбор, запускаем автоматический режим")
                launcher.auto_launch()
                
        except KeyboardInterrupt:
            print("\n👋 Отменено пользователем")
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")


if __name__ == "__main__":
    main()