#!/usr/bin/env python3
"""
🚀 Tennis System Starter - The Odds API Integration
Запускает систему с реальными коэффициентами
"""

import subprocess
import sys
import time
import webbrowser
import os

def check_dependencies():
    """Проверка зависимостей"""
    print("🔍 Checking dependencies...")
    
    required_files = [
        'correct_odds_api_integration.py',
        'web_backend_with_live_odds.py',
        'config.json'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True

def start_backend():
    """Запуск backend с The Odds API"""
    backend_files = [
        'web_backend_with_live_odds.py',
        'web_backend_with_dashboard.py',
        'web_backend.py'
    ]
    
    for backend_file in backend_files:
        if os.path.exists(backend_file):
            print(f"🚀 Starting {backend_file}...")
            return subprocess.Popen([sys.executable, backend_file])
    
    print("❌ No backend files found!")
    return None

def main():
    print("🎾 TENNIS SYSTEM WITH THE ODDS API")
    print("=" * 50)
    print("🎯 Real-time bookmaker odds")
    print("📊 Professional tennis matches")
    print("🤖 AI predictions")
    print("=" * 50)
    
    # Проверяем зависимости
    if not check_dependencies():
        print("\n💡 Run the integration script first:")
        print("python final_integration_script.py")
        return
    
    # Запускаем backend
    process = start_backend()
    
    if process:
        print("\n⏰ Starting server...")
        time.sleep(5)
        
        print("🌐 Opening dashboard...")
        webbrowser.open("http://localhost:5001")
        
        print("\n✅ TENNIS SYSTEM LAUNCHED!")
        print("📱 Dashboard: http://localhost:5001")
        print("🎯 Features:")
        print("  • Live tennis matches with real odds")
        print("  • Multiple bookmakers (Betfair, Bet365, etc.)")
        print("  • AI predictions and analysis")
        print("  • Auto-refresh every 60 seconds")
        print("  • API usage monitoring")
        print("\n⏹️ Press Ctrl+C to stop")
        
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
