#!/usr/bin/env python3
"""
🚀 Universal Tennis Dashboard Launcher
Запускает систему, которая работает КРУГЛЫЙ ГОД!
"""

import subprocess
import sys
import time
import webbrowser
import os
from datetime import datetime

def show_system_info():
    """Показывает информацию о системе"""
    try:
        from universal_tennis_data_collector import UniversalTennisDataCollector
        
        collector = UniversalTennisDataCollector()
        summary = collector.get_summary()
        
        print("🌍 UNIVERSAL TENNIS SYSTEM - INFORMATION")
        print("=" * 50)
        print(f"📅 Current Date: {summary['current_date']}")
        print(f"🏟️ Season Context: {summary['season_context']}")
        print(f"🏆 Active Tournaments: {summary['active_tournaments']}")
        
        if summary['active_tournament_names']:
            print(f"📋 Current Tournaments: {', '.join(summary['active_tournament_names'])}")
        
        print(f"🔜 Next Major: {summary['next_major']}")
        print(f"🎾 Available Matches: {summary['matches_available']}")
        print("=" * 50)
        
        return True
        
    except ImportError:
        print("⚠️ Universal system not yet installed")
        return False
    except Exception as e:
        print(f"⚠️ Error getting system info: {e}")
        return False

def start_backend():
    """Запуск backend сервера с приоритетом"""
    backend_files = [
        'web_backend_with_dashboard.py',
        'web_backend.py',
        'web_backend_minimal.py'
    ]
    
    for backend_file in backend_files:
        if os.path.exists(backend_file):
            print(f"🚀 Starting {backend_file}...")
            return subprocess.Popen([sys.executable, backend_file])
    
    print("❌ No backend files found!")
    return None

def main():
    print("🌍 UNIVERSAL TENNIS DASHBOARD - YEAR-ROUND SYSTEM")
    print("=" * 60)
    print("🎾 Works with ANY tournament, ANY time of year!")
    print("🚀 No more code rewrites after tournaments end!")
    print("=" * 60)
    
    # Показываем информацию о системе
    show_system_info()
    
    # Запускаем backend
    process = start_backend()
    
    if process:
        print("\n⏰ Starting server...")
        time.sleep(5)
        
        print("🌐 Opening browser...")
        webbrowser.open("http://localhost:5001")
        
        print("\n✅ UNIVERSAL DASHBOARD LAUNCHED!")
        print("📱 URL: http://localhost:5001")
        print("🌍 Showing current tennis matches worldwide!")
        print("🔄 System automatically updates with new tournaments!")
        print("⏹️ Press Ctrl+C to stop")
        
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n⏹️ Stopping server...")
            process.terminate()
            process.wait()
            print("✅ Server stopped")
    else:
        print("❌ Failed to start backend")

if __name__ == "__main__":
    main()
