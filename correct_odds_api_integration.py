#!/usr/bin/env python3
"""
🎾 ПРАВИЛЬНАЯ ИНТЕГРАЦИЯ THE ODDS API
Использует реальную структуру данных от The Odds API
"""

import requests
import json
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class TheOddsAPICorrect:
    """
    Правильная интеграция с The Odds API
    Основана на реальной структуре их данных
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4"
        self.requests_used = 0
        self.requests_remaining = None
        
    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Базовый метод для запросов с обработкой ошибок"""
        try:
            params['apiKey'] = self.api_key
            
            response = requests.get(f"{self.base_url}/{endpoint}", params=params, timeout=10)
            
            # Сохраняем информацию об использовании
            headers = response.headers
            self.requests_used = headers.get('x-requests-used', 'Unknown')
            self.requests_remaining = headers.get('x-requests-remaining', 'Unknown')
            
            logger.info(f"API Usage: {self.requests_used} used, {self.requests_remaining} remaining")
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                logger.error("❌ Invalid API key")
                return None
            elif response.status_code == 422:
                logger.warning("⚠️ Invalid parameters or no data available")
                return None
            else:
                logger.error(f"❌ API error {response.status_code}: {response.text}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"❌ Request failed: {e}")
            return None
    
    def get_available_sports(self) -> List[Dict]:
        """Получить доступные виды спорта"""
        logger.info("🏃 Getting available sports...")
        
        data = self._make_request("sports", {})
        if data:
            # Фильтруем только теннис
            tennis_sports = [sport for sport in data if 'tennis' in sport.get('key', '').lower()]
            logger.info(f"🎾 Found {len(tennis_sports)} tennis sports: {[s['key'] for s in tennis_sports]}")
            return tennis_sports
        return []
    
    def get_tennis_odds(self, sport_key: str = "tennis", regions: str = "us,uk,eu,au") -> List[Dict]:
        """
        Получить коэффициенты на теннис
        
        Args:
            sport_key: Ключ спорта (tennis, tennis_atp, tennis_wta)
            regions: Регионы букмекеров (us, uk, eu, au)
        """
        logger.info(f"🎾 Getting tennis odds for {sport_key}...")
        
        params = {
            'regions': regions,
            'markets': 'h2h',  # Head-to-head (winner market)
            'oddsFormat': 'decimal',
            'dateFormat': 'iso'
        }
        
        data = self._make_request(f"sports/{sport_key}/odds", params)
        
        if data:
            logger.info(f"✅ Found {len(data)} tennis matches with odds")
            return data
        return []
    
    def convert_to_tennis_format(self, odds_data: List[Dict]) -> Dict[str, Dict]:
        """
        Конвертирует данные The Odds API в формат твоей системы
        
        The Odds API возвращает список матчей с bookmakers
        Твоя система ожидает словарь с match_id как ключом
        """
        converted_odds = {}
        
        for match in odds_data:
            try:
                # Создаем уникальный ID матча
                match_id = match.get('id', f"odds_{datetime.now().timestamp()}")
                
                # Определяем игроков (в теннисе home_team и away_team - это игроки)
                player1 = match.get('home_team', 'Player 1')
                player2 = match.get('away_team', 'Player 2')
                
                # Ищем лучшие коэффициенты среди букмекеров
                best_p1_odds = None
                best_p2_odds = None
                best_p1_bookmaker = None
                best_p2_bookmaker = None
                
                for bookmaker in match.get('bookmakers', []):
                    bookmaker_name = bookmaker.get('title', bookmaker.get('key', 'Unknown'))
                    
                    for market in bookmaker.get('markets', []):
                        if market.get('key') == 'h2h':  # Head-to-head market
                            
                            for outcome in market.get('outcomes', []):
                                player_name = outcome.get('name')
                                odds = outcome.get('price')
                                
                                if not odds:
                                    continue
                                
                                # The Odds API может давать американские коэффициенты
                                # Конвертируем в десятичные если нужно
                                if isinstance(odds, (int, float)):
                                    if odds > 0 and odds < 1:
                                        # Уже десятичные
                                        decimal_odds = odds
                                    elif odds > 100 or odds < -100:
                                        # Американские - конвертируем в десятичные
                                        if odds > 0:
                                            decimal_odds = (odds / 100) + 1
                                        else:
                                            decimal_odds = (100 / abs(odds)) + 1
                                    else:
                                        decimal_odds = odds
                                else:
                                    decimal_odds = float(odds) if odds else 2.0
                                
                                # Определяем какой это игрок и сохраняем лучшие коэффициенты
                                if player_name == player1 or player_name.lower() in player1.lower():
                                    if best_p1_odds is None or decimal_odds > best_p1_odds:
                                        best_p1_odds = decimal_odds
                                        best_p1_bookmaker = bookmaker_name
                                        
                                elif player_name == player2 or player_name.lower() in player2.lower():
                                    if best_p2_odds is None or decimal_odds > best_p2_odds:
                                        best_p2_odds = decimal_odds
                                        best_p2_bookmaker = bookmaker_name
                
                # Формируем результат в формате твоей системы
                if best_p1_odds and best_p2_odds:
                    converted_odds[match_id] = {
                        'match_info': {
                            'player1': player1,
                            'player2': player2,
                            'tournament': f"Tennis Match ({match.get('sport_title', 'Tennis')})",
                            'surface': 'Unknown',  # The Odds API не предоставляет покрытие
                            'date': match.get('commence_time', datetime.now().isoformat())[:10],
                            'time': match.get('commence_time', datetime.now().isoformat())[11:16],
                            'source': 'the_odds_api'
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
                        },
                        'raw_data': match  # Сохраняем исходные данные для отладки
                    }
                
            except Exception as e:
                logger.error(f"❌ Error processing match {match.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"🔄 Converted {len(converted_odds)} matches to tennis format")
        return converted_odds
    
    def test_tennis_sports(self) -> Dict[str, int]:
        """Тестирует разные теннисные виды спорта"""
        logger.info("🔍 Testing different tennis sports...")
        
        # Возможные ключи теннисных видов спорта в The Odds API
        tennis_sport_keys = [
            'tennis',
            'tennis_atp', 
            'tennis_wta',
            'tennis_atp_french_open',
            'tennis_wta_french_open',
            'tennis_atp_wimbledon',
            'tennis_wta_wimbledon',
            'tennis_atp_us_open',
            'tennis_wta_us_open'
        ]
        
        results = {}
        
        for sport_key in tennis_sport_keys:
            try:
                odds_data = self.get_tennis_odds(sport_key, regions="us,uk")
                results[sport_key] = len(odds_data)
                logger.info(f"  🎾 {sport_key}: {len(odds_data)} matches")
                
                if odds_data:
                    # Показываем пример первого матча
                    first_match = odds_data[0]
                    logger.info(f"    Example: {first_match.get('home_team')} vs {first_match.get('away_team')}")
                    
            except Exception as e:
                logger.error(f"  ❌ {sport_key}: {e}")
                results[sport_key] = 0
        
        return results
    
    def get_usage_stats(self) -> Dict:
        """Получить статистику использования API"""
        return {
            'requests_used': self.requests_used,
            'requests_remaining': self.requests_remaining,
            'last_update': datetime.now().isoformat()
        }


class TennisOddsIntegrator:
    """
    Интегратор для подключения The Odds API к твоей системе
    """
    
    def __init__(self, api_key: str):
        self.odds_api = TheOddsAPICorrect(api_key)
        self.cache = {}
        self.last_update = None
        
    def get_live_tennis_odds(self, force_refresh: bool = False) -> Dict[str, Dict]:
        """
        Получить актуальные коэффициенты на теннис
        С кешированием для экономии API запросов
        """
        now = datetime.now()
        
        # Используем кеш если данные свежие (младше 10 минут)
        if (not force_refresh and self.cache and self.last_update and 
            (now - self.last_update).total_seconds() < 600):
            logger.info("📋 Using cached odds data")
            return self.cache
        
        logger.info("🔄 Fetching fresh odds data...")
        
        # Пробуем разные теннисные виды спорта
        all_odds = {}
        
        # Сначала пробуем основной теннис
        tennis_odds = self.odds_api.get_tennis_odds('tennis')
        if tennis_odds:
            converted = self.odds_api.convert_to_tennis_format(tennis_odds)
            all_odds.update(converted)
        
        # Пробуем ATP и WTA отдельно
        for sport_key in ['tennis_atp', 'tennis_wta']:
            try:
                sport_odds = self.odds_api.get_tennis_odds(sport_key)
                if sport_odds:
                    converted = self.odds_api.convert_to_tennis_format(sport_odds)
                    all_odds.update(converted)
            except Exception as e:
                logger.warning(f"⚠️ Could not get {sport_key}: {e}")
        
        # Обновляем кеш
        self.cache = all_odds
        self.last_update = now
        
        logger.info(f"✅ Got {len(all_odds)} tennis matches with real odds")
        return all_odds
    
    def get_integration_status(self) -> Dict:
        """Статус интеграции"""
        try:
            # Тест подключения
            sports = self.odds_api.get_available_sports()
            
            # Тест получения коэффициентов
            odds = self.get_live_tennis_odds()
            
            usage = self.odds_api.get_usage_stats()
            
            return {
                'status': 'connected',
                'tennis_sports_available': len(sports),
                'matches_with_odds': len(odds),
                'api_usage': usage,
                'last_check': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }


def test_integration(api_key: str):
    """Тестирование интеграции"""
    print("🎾 TESTING THE ODDS API INTEGRATION")
    print("=" * 50)
    
    if not api_key or api_key == "YOUR_API_KEY":
        print("❌ Please provide a real API key!")
        return
    
    try:
        # Создаем интегратор
        integrator = TennisOddsIntegrator(api_key)
        
        # Проверяем статус
        status = integrator.get_integration_status()
        
        print(f"🔌 Status: {status['status']}")
        print(f"🎾 Tennis sports: {status.get('tennis_sports_available', 0)}")
        print(f"⚽ Matches with odds: {status.get('matches_with_odds', 0)}")
        
        if 'api_usage' in status:
            usage = status['api_usage']
            print(f"📊 API Usage: {usage['requests_used']}/{usage['requests_remaining']} remaining")
        
        # Получаем реальные коэффициенты
        print("\n🔍 Getting live tennis odds...")
        odds = integrator.get_live_tennis_odds(force_refresh=True)
        
        if odds:
            print(f"\n✅ SUCCESS! Got {len(odds)} matches with real odds:")
            
            for i, (match_id, match_data) in enumerate(list(odds.items())[:3], 1):
                match_info = match_data['match_info']
                winner_odds = match_data['best_markets']['winner']
                
                print(f"\n🎾 Match {i}: {match_info['player1']} vs {match_info['player2']}")
                print(f"   📅 Date: {match_info['date']} {match_info['time']}")
                print(f"   💰 Odds: {winner_odds['player1']['odds']} ({winner_odds['player1']['bookmaker']}) vs")
                print(f"           {winner_odds['player2']['odds']} ({winner_odds['player2']['bookmaker']})")
        
        else:
            print("⚠️ No tennis matches found")
            print("💡 Tennis might be out of season or check sport keys")
        
        print(f"\n🎯 Ready for integration!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")


if __name__ == "__main__":
    # ЗАМЕНИ НА СВОЙ РЕАЛЬНЫЙ API КЛЮЧ
    API_KEY = "a1b20d709d4bacb2d95ddab880f91009"
    
    test_integration(API_KEY)