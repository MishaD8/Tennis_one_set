#!/usr/bin/env python3
"""
💰 API ECONOMY PATCH
Добавляется к существующему tennis backend без переписывания
Просто импортируйте этот модуль в ваш backend!
"""

import json
import time
import os
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

class SimpleAPIEconomy:
    """Простая система экономии API - легко интегрируется"""
    
    def __init__(self, api_key: str, max_per_hour: int = 30, cache_minutes: int = 20):
        self.api_key = api_key
        self.max_per_hour = max_per_hour
        self.cache_minutes = cache_minutes
        
        # Файлы для сохранения данных
        self.usage_file = "api_usage.json"
        self.cache_file = "api_cache.json"
        
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
    
    def make_tennis_request(self, sport_key: str = 'tennis', force_fresh: bool = False) -> Dict:
        """
        ГЛАВНАЯ ФУНКЦИЯ: заменяет ваши прямые API запросы
        Просто замените requests.get на этот метод!
        """
        
        cache_key = f"tennis_{sport_key}"
        
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
        if not can_request:
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
            
            logger.info(f"📡 API запрос: {sport_key}")
            response = requests.get(url, params=params, timeout=10)
            
            # Записываем использование
            self.record_api_request()
            
            if response.status_code == 200:
                data = response.json()
                
                # Кешируем успешный ответ
                self.save_to_cache(cache_key, data)
                
                return {
                    'success': True,
                    'data': data,
                    'source': 'fresh_api',
                    'emoji': '🔴',
                    'status': 'LIVE API'
                }
                
            elif response.status_code == 401:
                return {'success': False, 'error': 'Неверный API ключ'}
            elif response.status_code == 422:
                return {'success': False, 'error': 'Нет данных'}
            else:
                return {'success': False, 'error': f'HTTP {response.status_code}'}
                
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
            
            return {'success': False, 'error': str(e)}
    
    def get_usage_stats(self) -> Dict:
        """Статистика использования"""
        self.clean_old_requests()
        
        return {
            'requests_this_hour': len(self.hourly_requests),
            'max_per_hour': self.max_per_hour,
            'remaining_hour': self.max_per_hour - len(self.hourly_requests),
            'total_requests_ever': self.total_requests,
            'cache_items': len(self.cache_data),
            'cache_minutes': self.cache_minutes
        }

# ГЛОБАЛЬНАЯ ПЕРЕМЕННАЯ - инициализируется один раз
_api_economy = None

def init_api_economy(api_key: str, max_per_hour: int = 30, cache_minutes: int = 20):
    """
    ИНИЦИАЛИЗАЦИЯ: вызовите один раз в начале вашего backend
    """
    global _api_economy
    _api_economy = SimpleAPIEconomy(api_key, max_per_hour, cache_minutes)
    logger.info(f"💰 API Economy инициализирован: {max_per_hour}/час, кеш {cache_minutes}мин")

def economical_tennis_request(sport_key: str = 'tennis', force_fresh: bool = False) -> Dict:
    """
    ЗАМЕНА ДЛЯ ВАШИХ API ЗАПРОСОВ
    
    ВМЕСТО:
        response = requests.get("https://api.the-odds-api.com/v4/sports/tennis/odds", ...)
        
    ИСПОЛЬЗУЙТЕ:
        result = economical_tennis_request('tennis')
        if result['success']:
            matches = result['data']
    """
    if _api_economy is None:
        raise Exception("API Economy не инициализирован! Вызовите init_api_economy() первым")
    
    return _api_economy.make_tennis_request(sport_key, force_fresh)

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

# Пример интеграции в ваш существующий код
def example_integration():
    """
    ПРИМЕР: как интегрировать в ваш существующий backend
    """
    
    # 1. В начале вашего backend файла добавьте:
    """
    from api_economy_patch import init_api_economy, economical_tennis_request, get_api_usage
    
    # Инициализируйте экономию API
    init_api_economy(
        api_key="your_api_key_here",
        max_per_hour=30,    # ваш лимит
        cache_minutes=20    # время кеша
    )
    """
    
    # 2. В вашей функции получения данных замените:
    """
    # СТАРЫЙ КОД:
    def get_tennis_odds():
        response = requests.get("https://api.the-odds-api.com/v4/sports/tennis/odds", 
                               params={'apiKey': API_KEY, ...})
        if response.status_code == 200:
            return response.json()
        return None
    
    # НОВЫЙ КОД:
    def get_tennis_odds():
        result = economical_tennis_request('tennis')
        if result['success']:
            return result['data']
        return None
    """
    
    # 3. Добавьте новые endpoints (опционально):
    """
    @app.route('/api/usage')
    def api_usage():
        return jsonify(get_api_usage())
    """

if __name__ == "__main__":
    # Тест системы
    print("💰 ТЕСТИРОВАНИЕ API ECONOMY PATCH")
    print("=" * 50)
    
    # Инициализация
    init_api_economy("test_key", max_per_hour=5, cache_minutes=1)
    
    # Тест запросов
    print("1️⃣ Первый запрос:")
    result1 = economical_tennis_request('tennis')
    print(f"   Результат: {result1.get('source', 'error')}")
    
    print("2️⃣ Второй запрос (должен быть из кеша):")
    result2 = economical_tennis_request('tennis')
    print(f"   Результат: {result2.get('source', 'error')}")
    
    print("3️⃣ Статистика:")
    stats = get_api_usage()
    print(f"   Запросов за час: {stats['requests_this_hour']}")
    print(f"   Элементов в кеше: {stats['cache_items']}")
    
    print("\n✅ Патч готов к интеграции!")
    