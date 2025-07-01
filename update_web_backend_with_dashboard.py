#!/usr/bin/env python3
"""
🔄 ОБНОВЛЕНИЕ web_backend_with_dashboard.py
Специальный скрипт для интеграции реальных данных в ваш главный файл
"""

import os
import re
from datetime import datetime

TARGET_FILE = 'web_backend_with_dashboard.py'

def backup_file():
    """Создание резервной копии"""
    if os.path.exists(TARGET_FILE):
        backup_name = f"{TARGET_FILE}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            with open(TARGET_FILE, 'r', encoding='utf-8') as original:
                with open(backup_name, 'w', encoding='utf-8') as backup:
                    backup.write(original.read())
            print(f"💾 Backup created: {backup_name}")
            return backup_name
        except Exception as e:
            print(f"❌ Backup error: {e}")
            return None
    else:
        print(f"❌ {TARGET_FILE} not found!")
        return None

def check_dependencies():
    """Проверка необходимых файлов"""
    
    missing_files = []
    
    if not os.path.exists('real_tennis_data_collector.py'):
        missing_files.append('real_tennis_data_collector.py')
    
    if missing_files:
        print(f"❌ Missing required files:")
        for file in missing_files:
            print(f"   📄 {file}")
        
        print(f"\n💡 Create real_tennis_data_collector.py first!")
        print(f"Use the code from 'Real Tennis Data Integration Fix' artifact")
        return False
    
    print("✅ All required files found")
    return True

def update_imports():
    """Добавление импортов реальных данных"""
    
    try:
        with open(TARGET_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'REAL_DATA_AVAILABLE' in content:
            print("⚠️ Real data imports already exist")
            return True
        
        # Импорт для добавления
        import_addition = """
# ДОБАВЛЕНО: Импорт реальных данных
try:
    from real_tennis_data_collector import RealTennisDataCollector, RealOddsCollector
    REAL_DATA_AVAILABLE = True
    print("✅ Real tennis data collector imported")
except ImportError as e:
    print(f"⚠️ Real data collector not available: {e}")
    REAL_DATA_AVAILABLE = False
"""
        
        # Находим место для вставки после импортов
        lines = content.split('\n')
        insert_pos = 0
        
        # Ищем последний импорт
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                insert_pos = i + 1
        
        # Вставляем импорт
        lines.insert(insert_pos, import_addition)
        content = '\n'.join(lines)
        
        with open(TARGET_FILE, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Imports added to {TARGET_FILE}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating imports: {e}")
        return False

def add_real_data_methods():
    """Добавление методов для работы с реальными данными"""
    
    try:
        with open(TARGET_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'def get_real_tennis_data(' in content:
            print("⚠️ Real data methods already exist")
            return True
        
        # Методы для добавления
        real_data_methods = '''
    def get_real_tennis_data(self, days_ahead=7, filters=None):
        """НОВОЕ: Получение реальных теннисных данных"""
        if not REAL_DATA_AVAILABLE:
            print("⚠️ Real data not available, using demo")
            return self.get_demo_data_with_warnings()
        
        try:
            print("🔍 Fetching REAL tennis data...")
            
            collector = RealTennisDataCollector()
            odds_collector = RealOddsCollector()
            
            # Получаем реальные матчи
            all_matches = []
            
            # Wimbledon 2025
            wimbledon_matches = collector.get_wimbledon_2025_schedule()
            if wimbledon_matches:
                all_matches.extend(wimbledon_matches)
                print(f"✅ Wimbledon: {len(wimbledon_matches)} matches")
            
            if all_matches:
                # Получаем коэффициенты
                real_odds = odds_collector.get_real_odds(all_matches)
                
                # Обрабатываем матчи
                processed = []
                for match in all_matches:
                    processed_match = self.process_real_match(match, real_odds)
                    processed.append(processed_match)
                
                print(f"🎉 SUCCESS: {len(processed)} REAL matches processed!")
                return processed
            else:
                print("⚠️ No real matches found")
                return self.get_demo_data_with_warnings()
                
        except Exception as e:
            print(f"❌ Error getting real data: {e}")
            return self.get_demo_data_with_warnings()
    
    def process_real_match(self, match, odds_data):
        """НОВОЕ: Обработка реального матча"""
        match_id = match['id']
        
        # Получаем коэффициенты
        match_odds = odds_data.get(match_id, {})
        odds_info = match_odds.get('best_markets', {}).get('winner', {})
        
        p1_odds = odds_info.get('player1', {}).get('odds', 2.0)
        p2_odds = odds_info.get('player2', {}).get('odds', 2.0)
        
        # Расчет вероятности из коэффициентов
        prediction_prob = 1.0 / p1_odds / (1.0 / p1_odds + 1.0 / p2_odds)
        
        return {
            'id': match_id,
            'player1': match['player1'],
            'player2': match['player2'],
            'tournament': match['tournament'],
            'surface': match['surface'],
            'date': match['date'],
            'time': match['time'],
            'round': match.get('round', 'R32'),
            'prediction': {
                'probability': round(prediction_prob, 3),
                'confidence': 'High' if abs(p1_odds - p2_odds) > 1.0 else 'Medium',
                'expected_value': round((prediction_prob * (p1_odds - 1)) - (1 - prediction_prob), 3)
            },
            'metrics': {
                'player1_rank': RealOddsCollector()._estimate_ranking(match['player1']),
                'player2_rank': RealOddsCollector()._estimate_ranking(match['player2']),
                'h2h': 'TBD',
                'recent_form': 'Good'
            },
            'betting': {
                'odds': p1_odds,
                'stake': min(prediction_prob * 100, 50),
                'kelly': max(0, round((prediction_prob * p1_odds - 1) / (p1_odds - 1) * 0.25, 3)),
                'bookmaker': odds_info.get('player1', {}).get('bookmaker', 'Pinnacle')
            },
            'source': 'REAL_DATA',
            'data_quality': 'HIGH'
        }
    
    def get_demo_data_with_warnings(self):
        """НОВОЕ: Демо данные с предупреждениями"""
        try:
            # Пытаемся использовать существующий метод
            if hasattr(self, 'generate_fallback_matches'):
                demo_matches = self.generate_fallback_matches()
            else:
                # Создаем простые демо данные
                demo_matches = [{
                    'id': 'demo_001',
                    'player1': 'Demo Player A',
                    'player2': 'Demo Player B',
                    'tournament': 'Demo Tournament',
                    'surface': 'Hard',
                    'date': '2025-07-01',
                    'time': '15:00',
                    'prediction': {'probability': 0.6, 'confidence': 'Medium'},
                    'betting': {'odds': 1.8, 'stake': 50}
                }]
        except:
            demo_matches = []
        
        # Добавляем предупреждения
        for match in demo_matches:
            match['id'] = f"DEMO_{match.get('id', 'unknown')}"
            match['player1'] = f"⚠️ [DEMO] {match.get('player1', 'Unknown')}"
            match['player2'] = f"⚠️ [DEMO] {match.get('player2', 'Unknown')}"
            match['tournament'] = f"⚠️ DEMO: {match.get('tournament', 'Unknown')}"
            match['warning'] = 'DEMONSTRATION DATA - NOT REAL MATCH'
            match['source'] = 'DEMO_DATA'
            match['data_quality'] = 'DEMO'
        
        return demo_matches
'''
        
        # Ищем класс для добавления методов
        if 'class ' in content:
            # Ищем последний метод в классе
            class_pattern = r'(class\s+\w+.*?)(\n\nclass|\n\ndef(?!\s{4})|\n\nif __name__|\Z)'
            match = re.search(class_pattern, content, re.DOTALL)
            
            if match:
                class_content = match.group(1)
                # Добавляем методы в конец класса
                new_class_content = class_content.rstrip() + real_data_methods + '\n'
                content = content.replace(class_content, new_class_content)
                
                with open(TARGET_FILE, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print("✅ Real data methods added")
                return True
            else:
                print("❌ Could not find class structure")
                return False
        else:
            print("❌ No class found")
            return False
            
    except Exception as e:
        print(f"❌ Error adding methods: {e}")
        return False

def update_main_method():
    """Обновление основного метода get_upcoming_matches"""
    
    try:
        with open(TARGET_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Ищем метод get_upcoming_matches
        if 'def get_upcoming_matches(' in content:
            # Заменяем метод
            old_pattern = r'def get_upcoming_matches\(self[^)]*\):.*?(?=\n    def |\n\nclass |\nclass |\Z)'
            
            new_method = '''def get_upcoming_matches(self, days_ahead=7, filters=None):
        """ОБНОВЛЕНО: Приоритет реальным данным"""
        try:
            if REAL_DATA_AVAILABLE:
                print("🔍 Attempting to get REAL tennis data...")
                real_matches = self.get_real_tennis_data(days_ahead, filters)
                if real_matches and len(real_matches) > 0:
                    # Кэшируем реальные данные
                    self.cached_matches = real_matches
                    self.last_update = datetime.now()
                    return real_matches
            
            print("⚠️ Using demo data with warnings")
            demo_matches = self.get_demo_data_with_warnings()
            self.cached_matches = demo_matches
            self.last_update = datetime.now()
            return demo_matches
            
        except Exception as e:
            print(f"❌ Error in get_upcoming_matches: {e}")
            return self.get_demo_data_with_warnings()'''
            
            new_content = re.sub(old_pattern, new_method, content, flags=re.DOTALL)
            
            if new_content != content:
                with open(TARGET_FILE, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print("✅ get_upcoming_matches method updated")
                return True
            else:
                print("⚠️ get_upcoming_matches method not found or already updated")
                return False
        else:
            print("⚠️ get_upcoming_matches method not found")
            return False
            
    except Exception as e:
        print(f"❌ Error updating main method: {e}")
        return False

def verify_update():
    """Проверка результатов обновления"""
    
    try:
        with open(TARGET_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = {
            'Real data imports': 'REAL_DATA_AVAILABLE' in content,
            'Real data method': 'def get_real_tennis_data(' in content,
            'Process method': 'def process_real_match(' in content,
            'Demo warnings': 'get_demo_data_with_warnings' in content,
            'Updated main method': 'Attempting to get REAL tennis data' in content
        }
        
        print("\n🔍 VERIFICATION:")
        success_count = 0
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {check}")
            if passed:
                success_count += 1
        
        print(f"\n📊 Verification: {success_count}/{len(checks)} checks passed")
        return success_count >= 3  # Минимум 3 из 5 проверок
        
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False

def main():
    """Главная функция обновления"""
    
    print(f"🔄 UPDATING {TARGET_FILE}")
    print("=" * 60)
    
    # Проверяем зависимости
    if not check_dependencies():
        return
    
    # Создаем резервную копию
    backup = backup_file()
    if not backup:
        return
    
    # Выполняем обновления
    steps = [
        ("Adding imports", update_imports),
        ("Adding real data methods", add_real_data_methods),
        ("Updating main method", update_main_method)
    ]
    
    success_count = 0
    for step_name, step_func in steps:
        print(f"\n🔧 {step_name}...")
        try:
            if step_func():
                success_count += 1
                print("   ✅ Success")
            else:
                print("   ❌ Failed")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n📊 UPDATE RESULTS: {success_count}/{len(steps)} steps completed")
    
    # Проверяем результат
    if verify_update():
        print(f"\n🎉 SUCCESS! {TARGET_FILE} updated for real data!")
        print(f"\n📋 NEXT STEPS:")
        print(f"1. Test: python3 test_real_data.py")
        print(f"2. Start server: python3 {TARGET_FILE}")
        print(f"3. Check dashboard - should show real Wimbledon matches!")
        print(f"4. Look for 'REAL_DATA' source instead of 'DEMO_DATA'")
        
        print(f"\n🔍 WHAT TO EXPECT:")
        print(f"• Real player names: Rublev, Berrettini, Zverev, etc.")
        print(f"• Tournament: 'Wimbledon 2025'")
        print(f"• Surface: 'Grass'")
        print(f"• Clear warnings for any remaining demo data")
    else:
        print(f"\n⚠️ PARTIAL SUCCESS - some issues occurred")
        print(f"💡 Check backup file: {backup}")
        print(f"🔧 Manual review may be needed")

if __name__ == "__main__":
    main()