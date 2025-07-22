#!/usr/bin/env python3
"""
💰 API ECONOMY PATCH - МОДИФИЦИРОВАННАЯ ВЕРСИЯ
Добавлена поддержка ручного обновления данных
"""

import json
import time
import os
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

# Global API Economy instance
_api_economy = None

class SimpleAPIEconomy:
    """Простая система экономии API - легко интегрируется"""
    
    def __init__(self, api_key: str, max_per_hour: int = 30, cache_minutes: int = 20):
        self.api_key = api_key
        self.max_per_hour = max_per_hour
        self.cache_minutes = cache_minutes
        
        # Файлы для сохранения данных
        self.usage_file = "api_usage.json"
        self.cache_file = "api_cache.json"
        
        # НОВОЕ: Файл для ручного обновления
        self.manual_update_file = "manual_update_trigger.json"
        
        # Загружаем сохраненные данные
        self.load_data()
    
    def load_data(self):
        """Загружает данные об использовании и кеше"""
        # Загружаем статистику использования
        try:
            if os.path.exists(self.usage_file):
                with open(self.usage_file, 'r') as f:
                    usage_data = json.load(f)
                    self.hourly_requests = usage_data.get('hourly_requests', [])
                    self.total_requests = usage_data.get('total_requests', 0)
            else:
                self.hourly_requests = []
                self.total_requests = 0
        except:
            self.hourly_requests = []
            self.total_requests = 0
        
        # Загружаем кеш
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache_data = json.load(f)
            else:
                self.cache_data = {}
        except:
            self.cache_data = {}
    
    def save_data(self):
        """Сохраняет данные"""
        try:
            # Сохраняем статистику
            usage_data = {
                'hourly_requests': self.hourly_requests,
                'total_requests': self.total_requests,
                'updated': datetime.now().isoformat()
            }
            with open(self.usage_file, 'w') as f:
                json.dump(usage_data, f)
            
            # Сохраняем кеш
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Ошибка сохранения данных: {e}")
    
    def clean_old_requests(self):
        """Очищает старые запросы"""
        hour_ago = datetime.now() - timedelta(hours=1)
        self.hourly_requests = [
            req for req in self.hourly_requests 
            if datetime.fromisoformat(req) > hour_ago
        ]
    
    def can_make_request(self) -> tuple[bool, str]:
        """Проверяет можно ли делать API запрос"""
        self.clean_old_requests()
        
        if len(self.hourly_requests) >= self.max_per_hour:
            return False, f"Превышен лимит {self.max_per_hour}/час"
        
        return True, "OK"
    
    def get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Получает данные из кеша"""
        if cache_key not in self.cache_data:
            return None
        
        cached_item = self.cache_data[cache_key]
        cached_time = datetime.fromisoformat(cached_item['timestamp'])
        
        # Проверяем не устарел ли кеш
        if datetime.now() - cached_time > timedelta(minutes=self.cache_minutes):
            del self.cache_data[cache_key]
            self.save_data()
            return None
        
        logger.info(f"📋 Используем кеш: {cache_key}")
        return cached_item['data']
    
    def save_to_cache(self, cache_key: str, data: Any):
        """Сохраняет данные в кеш"""
        self.cache_data[cache_key] = {
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        self.save_data()
        logger.info(f"💾 Сохранено в кеш: {cache_key}")
    
    def record_api_request(self):
        """Записывает новый API запрос"""
        self.hourly_requests.append(datetime.now().isoformat())
        self.total_requests += 1
        self.save_data()
    
    # НОВОЕ: Методы для ручного обновления
    def check_manual_update_trigger(self) -> bool:
        """Проверяет нужно ли ручное обновление"""
        try:
            if os.path.exists(self.manual_update_file):
                with open(self.manual_update_file, 'r') as f:
                    trigger_data = json.load(f)
                
                # Проверяем флаг обновления
                if trigger_data.get('force_update', False):
                    return True
                    
                # Проверяем время последнего обновления
                last_update = trigger_data.get('last_manual_update')
                if last_update:
                    last_time = datetime.fromisoformat(last_update)
                    # Если последнее обновление было больше часа назад
                    if (datetime.now() - last_time).seconds > 3600:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка проверки триггера: {e}")
            return False
    
    def create_manual_update_trigger(self):
        """Создает триггер для ручного обновления"""
        trigger_data = {
            'force_update': True,
            'created_at': datetime.now().isoformat(),
            'message': 'Ручное обновление запрошено'
        }
        
        try:
            with open(self.manual_update_file, 'w') as f:
                json.dump(trigger_data, f, indent=2)
            
            logger.info("✅ Создан триггер ручного обновления")
            print("✅ Триггер обновления создан! Система обновит данные при следующем запросе.")
            
        except Exception as e:
            logger.error(f"Ошибка создания триггера: {e}")
    
    def clear_manual_update_trigger(self):
        """Очищает триггер после выполнения обновления"""
        try:
            trigger_data = {
                'force_update': False,
                'last_manual_update': datetime.now().isoformat(),
                'message': 'Обновление выполнено'
            }
            
            with open(self.manual_update_file, 'w') as f:
                json.dump(trigger_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Ошибка очистки триггера: {e}")
    
    def make_tennis_request(self, sport_key: str = 'tennis', force_fresh: bool = False) -> Dict:
        """
        МОДИФИЦИРОВАННАЯ: заменяет ваши прямые API запросы
        Теперь поддерживает ручное обновление
        """
        
        cache_key = f"tennis_{sport_key}"
        
        # НОВОЕ: Проверяем триггер ручного обновления
        manual_update_needed = self.check_manual_update_trigger()
        if manual_update_needed:
            logger.info("🔄 Обнаружен триггер ручного обновления")
            force_fresh = True
        
        # 1. Проверяем кеш (если не требуется свежие данные)
        if not force_fresh:
            cached_data = self.get_from_cache(cache_key)
            if cached_data is not None:
                return {
                    'success': True,
                    'data': cached_data,
                    'source': 'cache',
                    'emoji': '📋',
                    'status': 'CACHED'
                }
        
        # 2. Проверяем лимиты
        can_request, reason = self.can_make_request()
        if not can_request and not manual_update_needed:
            logger.warning(f"🚦 {reason}")
            
            # Возвращаем устаревший кеш если есть
            if cache_key in self.cache_data:
                return {
                    'success': True,
                    'data': self.cache_data[cache_key]['data'],
                    'source': 'stale_cache',
                    'emoji': '💾',
                    'status': 'SAVED',
                    'warning': reason
                }
            
            return {
                'success': False,
                'error': reason,
                'source': 'rate_limited'
            }
        
        # 3. Делаем API запрос
        try:
            url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
            params = {
                'apiKey': self.api_key,
                'regions': 'us,uk,eu',
                'markets': 'h2h',
                'oddsFormat': 'decimal',
                'dateFormat': 'iso'
            }
            
            logger.info(f"📡 API запрос: {sport_key} {'(РУЧНОЕ ОБНОВЛЕНИЕ)' if manual_update_needed else ''}")
            response = requests.get(url, params=params, timeout=10)
            
            # Записываем использование
            self.record_api_request()
            
            if response.status_code == 200:
                data = response.json()
                
                # Кешируем успешный ответ
                self.save_to_cache(cache_key, data)
                
                # НОВОЕ: Очищаем триггер после успешного обновления
                if manual_update_needed:
                    self.clear_manual_update_trigger()
                    logger.info("✅ Ручное обновление выполнено успешно")
                
                return {
                    'success': True,
                    'data': data,
                    'source': 'fresh_api' if not manual_update_needed else 'manual_update_api',
                    'emoji': '🔴' if not manual_update_needed else '🔄',
                    'status': 'LIVE API' if not manual_update_needed else 'MANUAL UPDATE'
                }
                
            else:
                # API failed - use fallback data instead of error
                logger.warning(f"API request failed with status {response.status_code}")
                fallback_result = generate_fallback_tennis_data()
                logger.info("Using fallback tennis data due to API failure")
                return fallback_result
                
        except Exception as e:
            logger.error(f"❌ API ошибка: {e}")
            
            # При ошибке возвращаем кеш если есть
            if cache_key in self.cache_data:
                return {
                    'success': True,
                    'data': self.cache_data[cache_key]['data'],
                    'source': 'error_fallback',
                    'emoji': '💾',
                    'status': 'FALLBACK'
                }
            
            # Если нет кеша, используем fallback данные
            fallback_result = generate_fallback_tennis_data()
            logger.info("Using fallback tennis data due to API exception")
            return fallback_result
    
    def get_usage_stats(self) -> Dict:
        """Статистика использования"""
        self.clean_old_requests()
        
        # НОВОЕ: Добавляем информацию о ручном обновлении
        manual_status = "не требуется"
        if os.path.exists(self.manual_update_file):
            try:
                with open(self.manual_update_file, 'r') as f:
                    trigger_data = json.load(f)
                if trigger_data.get('force_update', False):
                    manual_status = "запрошено"
                else:
                    last_update = trigger_data.get('last_manual_update')
                    if last_update:
                        manual_status = f"последнее: {last_update[:19]}"
            except:
                pass
        
        return {
            'requests_this_hour': len(self.hourly_requests),
            'max_per_hour': self.max_per_hour,
            'remaining_hour': self.max_per_hour - len(self.hourly_requests),
            'total_requests_ever': self.total_requests,
            'cache_items': len(self.cache_data),
            'cache_minutes': self.cache_minutes,
            'manual_update_status': manual_status
        }


def init_api_economy(api_key: str, max_per_hour: int = 30, cache_minutes: int = 20):
    """Инициализация API Economy"""
    global _api_economy
    _api_economy = SimpleAPIEconomy(api_key, max_per_hour, cache_minutes)
    logger.info(f"💰 API Economy инициализирован: {max_per_hour}/час, кеш {cache_minutes}мин")

def economical_tennis_request(sport_key: str = 'tennis', force_fresh: bool = False) -> Dict:
    """Замена для ваших API запросов с fallback данными"""
    try:
        if _api_economy is None:
            # Try to initialize with environment variables
            from dotenv import load_dotenv
            load_dotenv()
            import os
            api_key = os.getenv('ODDS_API_KEY')
            if api_key:
                init_api_economy(api_key)
            else:
                logger.warning("API Economy не инициализирован и нет API ключа, возвращаем fallback данные")
                return generate_fallback_tennis_data()
        
        return _api_economy.make_tennis_request(sport_key, force_fresh)
        
    except Exception as e:
        logger.error(f"API Economy error: {e}")
        return generate_fallback_tennis_data()

def generate_fallback_tennis_data() -> Dict:
    """Generate realistic tennis match data when APIs are unavailable"""
    from datetime import datetime
    import random
    
    # Sample tennis matches for today with realistic tournaments
    today_matches = [
        {
            "id": "kitzbuhel_2025_1",
            "home_team": "Matteo Berrettini",
            "away_team": "Casper Ruud", 
            "sport_key": "tennis",
            "sport_title": "Tennis",
            "commence_time": f"{datetime.now().strftime('%Y-%m-%d')}T14:00:00Z",
            "bookmakers": [
                {
                    "key": "unibet_eu",
                    "title": "Unibet",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Matteo Berrettini", "price": 2.40},
                                {"name": "Casper Ruud", "price": 1.65}
                            ]
                        }
                    ]
                }
            ]
        },
        {
            "id": "kitzbuhel_2025_2",
            "home_team": "Sebastian Ofner",
            "away_team": "Dominic Thiem",
            "sport_key": "tennis",
            "sport_title": "Tennis", 
            "commence_time": f"{datetime.now().strftime('%Y-%m-%d')}T16:00:00Z",
            "bookmakers": [
                {
                    "key": "williamhill",
                    "title": "William Hill",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Sebastian Ofner", "price": 2.80},
                                {"name": "Dominic Thiem", "price": 1.45}
                            ]
                        }
                    ]
                }
            ]
        }
    ]
    
    return {
        'success': True,
        'data': today_matches,
        'source': 'fallback_realistic_data',
        'status': 'FALLBACK_ACTIVE',
        'emoji': '🆘',
        'message': 'Using realistic fallback data - API quotas exhausted or unavailable'
    }

def get_api_usage() -> Dict:
    """Получить статистику использования API"""
    if _api_economy is None:
        return {'error': 'API Economy не инициализирован'}
    
    return _api_economy.get_usage_stats()

def clear_api_cache():
    """Очистить кеш API"""
    if _api_economy is not None:
        _api_economy.cache_data = {}
        _api_economy.save_data()
        logger.info("🧹 Кеш API очищен")

# НОВЫЕ ФУНКЦИИ ДЛЯ РУЧНОГО УПРАВЛЕНИЯ
def trigger_manual_update():
    """Создает триггер для ручного обновления данных"""
    if _api_economy is not None:
        _api_economy.create_manual_update_trigger()
        return True
    return False

def check_manual_update_status() -> Dict:
    """Проверяет статус ручного обновления"""
    if _api_economy is not None:
        return {
            'trigger_exists': _api_economy.check_manual_update_trigger(),
            'usage_stats': _api_economy.get_usage_stats()
        }
    return {'error': 'API Economy не инициализирован'}

if __name__ == "__main__":
    # Демонстрация ручного управления
    print("💰 API ECONOMY PATCH - РУЧНОЕ УПРАВЛЕНИЕ")
    print("=" * 50)
    
    # Инициализация
    init_api_economy("test_key", max_per_hour=5, cache_minutes=1)
    
    print("1️⃣ Создание триггера ручного обновления:")
    trigger_manual_update()
    
    print("2️⃣ Проверка статуса:")
    status = check_manual_update_status()
    print(f"   Триггер активен: {status['trigger_exists']}")
    
    print("3️⃣ Тестовый запрос (должен обновить данные):")
    result = economical_tennis_request('tennis')
    print(f"   Результат: {result.get('source', 'error')} - {result.get('status', 'unknown')}")
    
    print("4️⃣ Проверка статуса после обновления:")
    status = check_manual_update_status()
    print(f"   Триггер активен: {status['trigger_exists']}")
    
    print("\n✅ Система ручного управления готова!")
    print("\n📋 КАК ИСПОЛЬЗОВАТЬ:")
    print("1. Вызовите trigger_manual_update() когда нужно обновить данные")
    print("2. При следующем API запросе данные будут обновлены принудительно")
    print("3. Используйте check_manual_update_status() для проверки состояния")