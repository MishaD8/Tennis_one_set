#!/usr/bin/env python3
"""
🧹 AUTOMATED PROJECT CLEANUP SCRIPT
Generated on 2025-07-02 05:20:57
"""
import datetime
import os
import shutil
from pathlib import Path

def backup_before_cleanup():
    """Создает бэкап перед очисткой"""
    backup_dir = Path(f"cleanup_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    backup_dir.mkdir(exist_ok=True)
    return backup_dir

def safe_delete(file_path, backup_dir):
    """Безопасное удаление с бэкапом"""
    file_path = Path(file_path)
    if file_path.exists():
        # Копируем в бэкап
        shutil.copy2(file_path, backup_dir / file_path.name)
        # Удаляем оригинал
        file_path.unlink()
        print(f"✅ Deleted: {file_path}")
    else:
        print(f"⚠️ Not found: {file_path}")

def main():
    print("🧹 STARTING PROJECT CLEANUP")
    print("=" * 50)
    
    # Создаем бэкап
    backup_dir = backup_before_cleanup()
    print(f"💾 Backup directory: {backup_dir}")
    
    # Файлы для удаления
    files_to_delete = [
        "backend_integration_fix.py (duplicate of web_backend_fixed_ml.py)",
        "working_tennis_predictor.py",
        "api_integretion.py",
        "final_integration_script.py",
        "tennis_odds_tester.py",
        "test_real_data_integration.py",
        "launch_system.py",
        "system_restart_script.py",
        "start_real_dashboard.py",
        "universal_tennis_integration.py",
        "gunicorn.conf.py",
        "project_cleanup_analyzer.py",
        "tennis_betting_pipeline.py",
    ]
    
    # Удаляем файлы
    deleted_count = 0
    for file_path in files_to_delete:
        try:
            safe_delete(file_path, backup_dir)
            deleted_count += 1
        except Exception as e:
            print(f"❌ Error deleting {file_path}: {e}")
    
    print(f"\n📊 CLEANUP SUMMARY:")
    print(f"🗑️ Files deleted: {deleted_count}")
    print(f"💾 Backup created: {backup_dir}")
    print(f"\n✅ Cleanup completed!")

if __name__ == "__main__":
    import datetime
    main()
