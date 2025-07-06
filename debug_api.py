#!/usr/bin/env python3
"""
🔍 ДИАГНОСТИКА API ECONOMY
Проверяем почему не работает получение матчей
"""

import requests
import json
import os
from datetime import datetime
from api_economy_patch import init_api_economy, economical_tennis_request, get_api_usage

def test_api_economy_step_by_step():
    """Пошаговая диагностика API Economy"""
    
    print("🔍 ДИАГНОСТИКА API ECONOMY СИСТЕМЫ")
    print("=" * 60)
    print(f"🕐 Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Шаг 1: Инициализация
    print("\n1️⃣ ИНИЦИАЛИЗАЦИЯ API ECONOMY")
    try:
        init_api_economy(
            api_key="a1b20d709d4bacb2d95ddab880f91009",
            max_per_hour=30,
            cache_minutes=20
        )
        print("✅ API Economy инициализирован")
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        return False
    
    # Шаг 2: Проверка статистики
    print("\n2️⃣ ПРОВЕРКА ТЕКУЩЕЙ СТАТИСТИКИ")
    try:
        usage = get_api_usage()
        print(f"📊 Запросов в час: {usage.get('requests_this_hour', 0)}/{usage.get('max_per_hour', 30)}")
        print(f"📊 Остается запросов: {usage.get('remaining_hour', 0)}")
        print(f"📊 Элементов в кеше: {usage.get('cache_items', 0)}")
        print(f"📊 Статус ручного обновления: {usage.get('manual_update_status', 'не требуется')}")
    except Exception as e:
        print(f"❌ Ошибка получения статистики: {e}")
    
    # Шаг 3: Тест прямого API запроса
    print("\n3️⃣ ТЕСТ ПРЯМОГО API ЗАПРОСА")
    try:
        api_key = "a1b20d709d4bacb2d95ddab880f91009"
        url = "https://api.the-odds-api.com/v4/sports/tennis/odds"
        params = {
            'apiKey': api_key,
            'regions': 'us,uk,eu',
            'markets': 'h2h',
            'oddsFormat': 'decimal',
            'dateFormat': 'iso'
        }
        
        print(f"🌐 Запрос к: {url}")
        print(f"🔑 API ключ: {api_key[:10]}...{api_key[-5:]}")
        
        response = requests.get(url, params=params, timeout=10)
        
        print(f"📊 HTTP статус: {response.status_code}")
        print(f"📊 Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API отвечает! Получено {len(data)} матчей")
            
            # Показываем первые несколько матчей
            for i, match in enumerate(data[:3], 1):
                print(f"   {i}. {match.get('home_team', 'N/A')} vs {match.get('away_team', 'N/A')}")
                print(f"      Начало: {match.get('commence_time', 'N/A')}")
                print(f"      Букмекеров: {len(match.get('bookmakers', []))}")
        
        elif response.status_code == 401:
            print("❌ Неверный API ключ!")
            return False
        elif response.status_code == 422:
            print("⚠️ Нет данных для tennis (матчей нет)")
        else:
            print(f"❌ API ошибка: {response.status_code}")
            print(f"📄 Ответ: {response.text}")
            
    except Exception as e:
        print(f"❌ Ошибка прямого запроса: {e}")
    
    # Шаг 4: Тест через API Economy
    print("\n4️⃣ ТЕСТ ЧЕРЕЗ API ECONOMY")
    try:
        result = economical_tennis_request('tennis', force_fresh=True)
        
        print(f"📊 Успех: {result.get('success', False)}")
        print(f"📊 Источник: {result.get('source', 'unknown')}")
        print(f"📊 Статус: {result.get('status', 'unknown')}")
        print(f"📊 Эмодзи: {result.get('emoji', 'none')}")
        
        if result.get('success'):
            data = result.get('data', [])
            print(f"✅ API Economy работает! Получено {len(data)} матчей")
            
            if data:
                # Анализируем первый матч
                first_match = data[0]
                print(f"\n📋 АНАЛИЗ ПЕРВОГО МАТЧА:")
                print(f"   ID: {first_match.get('id', 'N/A')}")
                print(f"   Игроки: {first_match.get('home_team', 'N/A')} vs {first_match.get('away_team', 'N/A')}")
                print(f"   Спорт: {first_match.get('sport_title', 'N/A')}")
                print(f"   Букмекеров: {len(first_match.get('bookmakers', []))}")
                
                # Анализируем коэффициенты
                bookmakers = first_match.get('bookmakers', [])
                if bookmakers:
                    first_bookmaker = bookmakers[0]
                    print(f"   Первый букмекер: {first_bookmaker.get('title', 'N/A')}")
                    
                    markets = first_bookmaker.get('markets', [])
                    if markets:
                        h2h_market = None
                        for market in markets:
                            if market.get('key') == 'h2h':
                                h2h_market = market
                                break
                        
                        if h2h_market:
                            outcomes = h2h_market.get('outcomes', [])
                            print(f"   Исходов в H2H: {len(outcomes)}")
                            for outcome in outcomes:
                                print(f"     {outcome.get('name', 'N/A')}: {outcome.get('price', 'N/A')}")
                        else:
                            print("   ❌ Нет H2H рынка")
                    else:
                        print("   ❌ Нет рынков")
                else:
                    print("   ❌ Нет букмекеров")
            else:
                print("⚠️ Данные пустые")
        else:
            error = result.get('error', 'Unknown error')
            print(f"❌ API Economy ошибка: {error}")
            
    except Exception as e:
        print(f"❌ Ошибка API Economy теста: {e}")
    
    # Шаг 5: Проверка альтернативных спортов
    print("\n5️⃣ ПРОВЕРКА АЛЬТЕРНАТИВНЫХ ТЕННИСНЫХ КЛЮЧЕЙ")
    tennis_keys = ['tennis_atp', 'tennis_wta', 'tennis_atp_wimbledon', 'tennis_wta_wimbledon']
    
    for sport_key in tennis_keys:
        try:
            print(f"\n🎾 Тест {sport_key}:")
            result = economical_tennis_request(sport_key)
            
            if result.get('success'):
                data = result.get('data', [])
                print(f"   ✅ {len(data)} матчей")
            else:
                print(f"   ❌ {result.get('error', 'No data')}")
                
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
    
    # Шаг 6: Проверка файлов
    print("\n6️⃣ ПРОВЕРКА СОЗДАННЫХ ФАЙЛОВ")
    files_to_check = ['api_usage.json', 'api_cache.json', 'manual_update_trigger.json']
    
    for filename in files_to_check:
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                print(f"✅ {filename}: {len(str(data))} символов")
            except Exception as e:
                print(f"❌ {filename}: ошибка чтения - {e}")
        else:
            print(f"⚪ {filename}: не существует")
    
    print("\n" + "=" * 60)
    print("🎯 ВЫВОДЫ И РЕКОМЕНДАЦИИ:")
    print("=" * 60)
    
    return True

def suggest_fixes():
    """Предложения по исправлению"""
    print("\n💡 ВОЗМОЖНЫЕ РЕШЕНИЯ:")
    print("1. Если API ключ неверный - получите новый на the-odds-api.com")
    print("2. Если нет матчей - попробуйте другое время дня")
    print("3. Если превышен лимит - подождите час или смените ключ")
    print("4. Если ошибки API - проверьте интернет соединение")
    print("5. Если матчи есть но не показываются - проблема в фильтрах")
    
    print("\n🔧 ДЕЙСТВИЯ ДЛЯ ИСПРАВЛЕНИЯ:")
    print("• Запустите: python manual_update.py full")
    print("• Или в браузере нажмите 'Force API Update'")
    print("• Проверьте логи backend'а на ошибки")
    print("• Попробуйте разные tennis ключи (tennis_atp, tennis_wta)")

if __name__ == "__main__":
    try:
        test_api_economy_step_by_step()
        suggest_fixes()
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()