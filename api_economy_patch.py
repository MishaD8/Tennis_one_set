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
    
    def __init__(self, api_key: str = None, max_per_hour: int = 30, cache_minutes: int = 20):
        # Load API key from config if not provided
        if not api_key:
            try:
                from config_loader import load_secure_config
                config = load_secure_config()
                api_key = (config.get('data_sources', {})
                          .get('the_odds_api', {})
                          .get('api_key'))
                if not api_key:
                    api_key = (config.get('betting_apis', {})
                              .get('the_odds_api', {})
                              .get('api_key'))
            except Exception as e:
                logger.warning(f"Could not load API key from config: {e}")
        
        # Secure API key handling - never store in plain text in logs
        self.api_key = api_key
        self._api_key_hash = None  # For verification without storing key
        if api_key:
            import hashlib
            self._api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:8]
        
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
    
    def make_tennis_request(self, sport_key: str = 'tennis_atp_canadian_open', force_fresh: bool = False) -> Dict:
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
        
        # 3. Делаем API запрос - пробуем несколько sport keys
        sport_keys_to_try = [
            'tennis_atp_canadian_open',
            'tennis_wta_canadian_open'
        ]
        
        # Start with requested sport key if it's not in our list
        if sport_key not in sport_keys_to_try:
            sport_keys_to_try.insert(0, sport_key)
        
        for try_sport_key in sport_keys_to_try:
            try:
                url = f"https://api.the-odds-api.com/v4/sports/{try_sport_key}/odds"
                params = {
                    'apiKey': self.api_key,
                    'regions': 'us,uk,eu',
                    'markets': 'h2h',
                    'oddsFormat': 'decimal',
                    'dateFormat': 'iso'
                }
                
                # Secure logging - never log API keys
                masked_params = params.copy()
                if 'apiKey' in masked_params:
                    masked_params['apiKey'] = f"***{self._api_key_hash or 'MASKED'}***"
                
                logger.info(f"📡 API запрос: {try_sport_key} {'(РУЧНОЕ ОБНОВЛЕНИЕ)' if manual_update_needed else ''}")
                logger.debug(f"Request params (API key masked): {masked_params}")
                
                response = requests.get(url, params=params, timeout=10)
                
                # Записываем использование
                self.record_api_request()
                
                if response.status_code == 200:
                    data = response.json()
                    if data:  # If we actually got matches
                        # Process and return the data
                        tennis_data = self.convert_to_tennis_format(data, try_sport_key)
                        
                        # Сохраняем в кеш
                        self.save_to_cache(cache_key, tennis_data)
                        
                        # НОВОЕ: Очищаем триггер после успешного обновления
                        if manual_update_needed:
                            self.clear_manual_update_trigger()
                        
                        return {
                            'success': True,
                            'data': tennis_data,
                            'source': 'fresh_api' if not manual_update_needed else 'manual_update',
                            'emoji': '🔴' if not manual_update_needed else '🔄',
                            'status': 'LIVE_API' if not manual_update_needed else 'MANUAL_UPDATE',
                            'matches_count': len(tennis_data),
                            'sport_key_used': try_sport_key
                        }
                    else:
                        logger.info(f"No matches found for {try_sport_key}, trying next...")
                        continue
                        
                elif response.status_code == 404:
                    logger.info(f"Sport key {try_sport_key} not found (404), trying next...")
                    continue
                else:
                    logger.warning(f"API request failed with status {response.status_code} for {try_sport_key}")
                    continue
                    
            except Exception as e:
                logger.error(f"API request failed for {try_sport_key}: {e}")
                continue
        
        # If all sport keys failed
        logger.warning("All sport keys failed")
        # Return cached data if available
        if cache_key in self.cache_data:
            return {
                'success': True,
                'data': self.cache_data[cache_key]['data'],
                'source': 'error_fallback',
                'emoji': '💾',
                'status': 'FALLBACK'
            }
        
        return {
            'success': False,
            'error': 'All API requests failed and no cached data available',
            'source': 'api_error'
        }
    
    def convert_to_tennis_format(self, api_data: list, sport_key: str) -> Dict:
        """Convert Odds API format to tennis system format"""
        converted_matches = {}
        
        for match in api_data:
            try:
                match_id = match.get('id', f"odds_{datetime.now().timestamp()}")
                player1 = match.get('home_team', 'Player 1')
                player2 = match.get('away_team', 'Player 2')
                
                # Find best odds
                best_p1_odds = None
                best_p2_odds = None
                best_p1_bookmaker = None
                best_p2_bookmaker = None
                
                for bookmaker in match.get('bookmakers', []):
                    bookmaker_name = bookmaker.get('title', bookmaker.get('key', 'Unknown'))
                    
                    for market in bookmaker.get('markets', []):
                        if market.get('key') == 'h2h':
                            for outcome in market.get('outcomes', []):
                                player_name = outcome.get('name')
                                odds = outcome.get('price', 0)
                                
                                if not odds:
                                    continue
                                
                                decimal_odds = float(odds)
                                
                                # Match to players and keep best odds
                                if player_name == player1:
                                    if best_p1_odds is None or decimal_odds > best_p1_odds:
                                        best_p1_odds = decimal_odds
                                        best_p1_bookmaker = bookmaker_name
                                elif player_name == player2:
                                    if best_p2_odds is None or decimal_odds > best_p2_odds:
                                        best_p2_odds = decimal_odds
                                        best_p2_bookmaker = bookmaker_name
                
                # Create tennis format result
                if best_p1_odds and best_p2_odds:
                    converted_matches[match_id] = {
                        'match_info': {
                            'player1': player1,
                            'player2': player2,
                            'tournament': match.get('sport_title', 'Tennis Tournament'),
                            'surface': 'Unknown',
                            'date': match.get('commence_time', datetime.now().isoformat())[:10],
                            'time': match.get('commence_time', datetime.now().isoformat())[11:16],
                            'source': 'api_economy_patch'
                        },
                        'best_markets': {
                            'winner': {
                                'player1': {
                                    'odds': round(best_p1_odds, 2),
                                    'bookmaker': best_p1_bookmaker
                                },
                                'player2': {
                                    'odds': round(best_p2_odds, 2),
                                    'bookmaker': best_p2_bookmaker
                                }
                            }
                        }
                    }
                    
            except Exception as e:
                logger.error(f"Error processing match {match.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"🔄 Converted {len(converted_matches)} matches from {sport_key}")
        return converted_matches
    
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
    """Return empty data when APIs are unavailable - only show real tournaments"""
    return {
        'success': False,
        'data': [],
        'source': 'no_fallback_data',
        'status': 'NO_REAL_DATA',
        'emoji': '🚫',
        'message': 'No real tournament data available - only ATP/WTA tournaments shown'
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