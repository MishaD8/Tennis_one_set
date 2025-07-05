#!/usr/bin/env python3
"""
🌍 ИСПРАВЛЕННАЯ УНИВЕРСАЛЬНАЯ ИНТЕГРАЦИЯ ТЕННИСНЫХ ДАННЫХ
Работает круглый год с любыми активными турнирами
"""

import requests
import json
from datetime import datetime
import os
from typing import List, Dict, Optional

class UniversalTennisDataFix:
    """Универсальная система получения теннисных данных круглый год"""
    
    def __init__(self):
        self.api_key = os.getenv('ODDS_API_KEY', 'a1b20d709d4bacb2d95ddab880f91009')
        self.base_url = "https://api.the-odds-api.com/v4"
        
        # ИСПРАВЛЕНО: Все возможные теннисные ключи
        self.tennis_sport_keys = [
            # Grand Slams (активные ключи)
            'tennis_atp_wimbledon',
            'tennis_wta_wimbledon', 
            'tennis_atp_us_open',
            'tennis_wta_us_open',
            'tennis_atp_french_open',
            'tennis_wta_french_open',
            'tennis_atp_australian_open',
            'tennis_wta_australian_open',
            
            # Основные теннисные ключи (всегда пробуем)
            'tennis',
            'tennis_atp',
            'tennis_wta',
            
            # Masters и крупные турниры
            'tennis_atp_indian_wells',
            'tennis_wta_indian_wells',
            'tennis_atp_miami',
            'tennis_wta_miami',
            'tennis_atp_madrid',
            'tennis_wta_madrid',
            'tennis_atp_rome',
            'tennis_wta_rome',
            'tennis_atp_cincinnati',
            'tennis_wta_cincinnati',
            'tennis_atp_shanghai',
            'tennis_wta_beijing',
            'tennis_atp_paris',
            'tennis_wta_finals',
            'tennis_atp_finals'
        ]
    
    def discover_active_tennis_sports(self) -> List[Dict]:
        """Автоматическое обнаружение активных теннисных турниров"""
        print("🔍 ПОИСК АКТИВНЫХ ТЕННИСНЫХ ТУРНИРОВ")
        print("=" * 50)
        
        try:
            # Получаем все доступные виды спорта
            response = requests.get(
                f"{self.base_url}/sports",
                params={'apiKey': self.api_key},
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"❌ Ошибка API: HTTP {response.status_code}")
                return []
            
            all_sports = response.json()
            
            # Фильтруем только теннисные и активные
            tennis_sports = []
            for sport in all_sports:
                if (sport.get('active', False) and 
                    ('tennis' in sport.get('key', '').lower() or
                     'tennis' in sport.get('title', '').lower() or
                     'tennis' in sport.get('group', '').lower())):
                    tennis_sports.append(sport)
            
            print(f"✅ Найдено активных теннисных турниров: {len(tennis_sports)}")
            
            # Показываем что нашли
            for sport in tennis_sports:
                status = "🔥 ACTIVE" if sport.get('active') else "💤 inactive"
                print(f"   {status} {sport['key']}: {sport['title']}")
            
            return tennis_sports
            
        except Exception as e:
            print(f"❌ Ошибка поиска турниров: {e}")
            return []
    
    def get_universal_tennis_matches(self) -> List[Dict]:
        """Получение матчей из всех активных турниров"""
        print(f"\n🎾 ПОЛУЧЕНИЕ МАТЧЕЙ ИЗ ВСЕХ АКТИВНЫХ ТУРНИРОВ")
        print("=" * 50)
        
        # ИСПРАВЛЕНО: Сначала пробуем все основные ключи
        priority_keys = ['tennis', 'tennis_atp', 'tennis_wta']
        all_matches = []
        successful_tournaments = 0
        
        # 1. Сначала пробуем основные ключи
        for sport_key in priority_keys:
            print(f"\n🎾 Проверяем основной ключ: {sport_key}...")
            
            matches = self._get_tournament_matches(sport_key, f"Tennis ({sport_key})")
            
            if matches:
                all_matches.extend(matches)
                successful_tournaments += 1
                print(f"   ✅ Получено {len(matches)} матчей")
            else:
                print(f"   ⚪ Нет матчей")
        
        # 2. Если основные не сработали, пробуем все остальные
        if not all_matches:
            print(f"\n🔍 Основные ключи не дали результата, пробуем специфичные турниры...")
            
            # Находим активные турниры
            active_sports = self.discover_active_tennis_sports()
            
            for sport in active_sports:
                sport_key = sport['key']
                sport_title = sport['title']
                
                print(f"\n🎾 Проверяем {sport_title} ({sport_key})...")
                
                matches = self._get_tournament_matches(sport_key, sport_title)
                
                if matches:
                    all_matches.extend(matches)
                    successful_tournaments += 1
                    print(f"   ✅ Получено {len(matches)} матчей")
                else:
                    print(f"   ⚪ Нет матчей")
        
        print(f"\n📊 ИТОГО:")
        print(f"   🏆 Успешных турниров: {successful_tournaments}")
        print(f"   🎾 Всего матчей: {len(all_matches)}")
        
        return all_matches
    
    def _get_tournament_matches(self, sport_key: str, sport_title: str) -> List[Dict]:
        """Получение матчей из конкретного турнира"""
        try:
            url = f"{self.base_url}/sports/{sport_key}/odds"
            params = {
                'apiKey': self.api_key,
                'regions': 'us,uk,eu,au',
                'markets': 'h2h',
                'oddsFormat': 'decimal',
                'dateFormat': 'iso'
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                matches = response.json()
                
                # Добавляем метаданные
                for match in matches:
                    match['sport_key'] = sport_key
                    match['tournament_title'] = sport_title
                    match['surface'] = self._detect_surface(sport_key, sport_title)
                    match['tournament_level'] = self._detect_tournament_level(sport_key, sport_title)
                
                return matches
            elif response.status_code == 422:
                # Нет данных для этого спорта
                return []
            elif response.status_code == 401:
                print(f"❌ Неверный API ключ!")
                return []
            else:
                print(f"❌ API ошибка {response.status_code} для {sport_key}")
                return []
                
        except Exception as e:
            print(f"❌ Ошибка запроса для {sport_key}: {e}")
            return []
    
    def _detect_surface(self, sport_key: str, sport_title: str) -> str:
        """Определение покрытия турнира"""
        key_lower = sport_key.lower()
        title_lower = sport_title.lower()
        
        # Определяем по ключевым словам
        if any(x in key_lower for x in ['wimbledon']):
            return 'Grass'
        elif any(x in key_lower for x in ['french', 'roland', 'garros', 'madrid', 'rome']):
            return 'Clay'
        elif any(x in title_lower for x in ['grass']):
            return 'Grass'
        elif any(x in title_lower for x in ['clay']):
            return 'Clay'
        else:
            return 'Hard'  # По умолчанию
    
    def _detect_tournament_level(self, sport_key: str, sport_title: str) -> str:
        """Определение уровня турнира"""
        key_lower = sport_key.lower()
        title_lower = sport_title.lower()
        
        # Grand Slams
        if any(x in key_lower for x in ['australian_open', 'french_open', 'wimbledon', 'us_open']):
            return 'Grand Slam'
        
        # Masters/WTA 1000
        elif any(x in key_lower for x in ['indian_wells', 'miami', 'madrid', 'rome', 'cincinnati', 'shanghai', 'paris']):
            return 'Masters 1000'
        
        # Finals
        elif 'finals' in key_lower:
            return 'Finals'
        
        # ATP/WTA общие
        elif 'atp' in key_lower:
            return 'ATP Tour'
        elif 'wta' in key_lower:
            return 'WTA Tour'
        
        else:
            return 'Professional'
    
    def adapt_matches_for_underdog_system(self, raw_matches: List[Dict]) -> List[Dict]:
        """Адаптация матчей для underdog системы"""
        print(f"\n🎯 АДАПТАЦИЯ ДЛЯ UNDERDOG СИСТЕМЫ")
        print("=" * 50)
        
        underdog_matches = []
        
        for match in raw_matches:
            adapted_match = self._adapt_single_match(match)
            if adapted_match:
                underdog_matches.append(adapted_match)
        
        print(f"✅ Адаптировано для underdog анализа: {len(underdog_matches)}")
        
        # Сортируем по качеству (лучшие первыми)
        underdog_matches.sort(key=lambda x: x['underdog_analysis']['prediction']['probability'], reverse=True)
        
        return underdog_matches
    
    def _adapt_single_match(self, api_match: Dict) -> Optional[Dict]:
        """Адаптация одного матча"""
        try:
            # Извлекаем базовую информацию
            player1 = api_match.get('home_team', 'Player 1')
            player2 = api_match.get('away_team', 'Player 2')
            tournament_title = api_match.get('tournament_title', 'Tennis Tournament')
            surface = api_match.get('surface', 'Hard')
            level = api_match.get('tournament_level', 'Professional')
            
            # Извлекаем лучшие коэффициенты
            odds1, odds2, bookmaker = self._extract_best_odds_with_bookmaker(api_match.get('bookmakers', []))
            
            if not odds1 or not odds2:
                return None
            
            # Определяем андердога
            if odds1 > odds2:
                underdog = player1
                favorite = player2
                underdog_odds = odds1
                favorite_odds = odds2
            else:
                underdog = player2
                favorite = player1
                underdog_odds = odds2
                favorite_odds = odds1
            
            # Фильтр качества: коэффициенты в разумном диапазоне
            if not (1.8 <= underdog_odds <= 8.0):
                return None
            
            # Рассчитываем вероятность взять сет
            match_prob = 1.0 / underdog_odds
            
            # Корректировки на основе турнира и покрытия
            surface_bonus = 0.05 if surface == 'Grass' else 0.02  # Трава более непредсказуема
            level_bonus = 0.03 if 'Grand Slam' in level else 0.01
            
            set_probability = min(0.88, match_prob + 0.25 + surface_bonus + level_bonus)
            
            # Фильтр: попадание в целевой диапазон
            if not (0.45 <= set_probability <= 0.88):
                return None
            
            # Определяем уверенность
            confidence = 'Very High' if set_probability > 0.8 else \
                        'High' if set_probability > 0.7 else \
                        'Medium'
            
            # Качественные факторы
            key_factors = [
                f"🏆 {level} - высокий уровень турнира",
                f"🏟️ {surface} покрытие - дополнительная непредсказуемость" if surface != 'Hard' else f"🏟️ Хард - сбалансированное покрытие",
                f"💰 Коэффициент {underdog_odds:.1f} - хорошая ценность",
                f"📊 {set_probability:.0%} шанс взять хотя бы один сет"
            ]
            
            # Формируем результат
            return {
                'id': f"universal_{api_match.get('id', f'match_{datetime.now().timestamp()}')}",
                'player1': f"🎾 {player1}",
                'player2': f"🎾 {player2}",
                'tournament': f"🏆 {tournament_title} - {level}",
                'surface': surface,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'time': api_match.get('commence_time', '14:00')[:5] if api_match.get('commence_time') else '14:00',
                'round': 'Live',
                'court': f"{tournament_title} Court",
                'status': 'upcoming',
                'odds': {
                    'player1': odds1,
                    'player2': odds2
                },
                'underdog_analysis': {
                    'underdog': underdog,
                    'favorite': favorite,
                    'underdog_odds': underdog_odds,
                    'favorite_odds': favorite_odds,
                    'prediction': {
                        'probability': round(set_probability, 3),
                        'confidence': confidence,
                        'key_factors': key_factors
                    },
                    'quality_rating': 'HIGH' if set_probability > 0.75 else 'MEDIUM'
                },
                'source': f'LIVE_{api_match.get("sport_key", "TENNIS").upper()}',
                'tournament_metadata': {
                    'sport_key': api_match.get('sport_key'),
                    'level': level,
                    'surface': surface,
                    'bookmaker': bookmaker
                }
            }
            
        except Exception as e:
            print(f"❌ Ошибка адаптации матча: {e}")
            return None
    
    def _extract_best_odds_with_bookmaker(self, bookmakers: List[Dict]) -> tuple:
        """Извлечение лучших коэффициентов с информацией о букмекере"""
        best_odds1 = None
        best_odds2 = None
        best_bookmaker = "Unknown"
        
        for bookmaker in bookmakers:
            bookmaker_name = bookmaker.get('title', bookmaker.get('key', 'Unknown'))
            
            for market in bookmaker.get('markets', []):
                if market.get('key') == 'h2h':
                    outcomes = market.get('outcomes', [])
                    if len(outcomes) >= 2:
                        odds1 = outcomes[0].get('price')
                        odds2 = outcomes[1].get('price')
                        
                        if odds1 and odds2:
                            # Берем максимальные коэффициенты (лучшие для игрока)
                            if not best_odds1 or (odds1 > best_odds1):
                                best_odds1 = odds1
                                best_bookmaker = bookmaker_name
                            if not best_odds2 or (odds2 > best_odds2):
                                best_odds2 = odds2
        
        return best_odds1, best_odds2, best_bookmaker
    
    def get_season_context(self) -> str:
        """Определение контекста текущего сезона"""
        month = datetime.now().month
        
        if month in [1, 2]:
            return "Hard Court Season - Australian Open & Middle East"
        elif month in [3, 4, 5]:
            return "Clay Court Season - European Spring"
        elif month in [6, 7]:
            return "Grass Court Season - Wimbledon Period"
        elif month in [8, 9]:
            return "Hard Court Season - US Open Series"
        elif month in [10, 11]:
            return "Indoor Season - Masters & Finals"
        else:  # December
            return "Off Season - Exhibition Matches"
    
    def run_universal_integration(self) -> bool:
        """Запуск универсальной интеграции"""
        print("🌍 УНИВЕРСАЛЬНАЯ ТЕННИСНАЯ ИНТЕГРАЦИЯ")
        print("=" * 60)
        print(f"🕐 Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎾 Сезон: {self.get_season_context()}")
        print(f"🎯 Цель: найти underdog возможности в активных турнирах")
        print("=" * 60)
        
        # 1. Получаем все матчи
        all_matches = self.get_universal_tennis_matches()
        
        if not all_matches:
            print("\n❌ НЕТ АКТИВНЫХ МАТЧЕЙ")
            print("💡 Возможные причины:")
            print("   • Между турнирами")
            print("   • Проблемы с API ключом")
            print("   • Нет данных в The Odds API")
            print(f"   • API ключ: {self.api_key[:10]}...{self.api_key[-5:]}")
            return False
        
        # 2. Адаптируем для underdog системы
        underdog_matches = self.adapt_matches_for_underdog_system(all_matches)
        
        if not underdog_matches:
            print("\n⚠️ НЕТ ПОДХОДЯЩИХ UNDERDOG ВОЗМОЖНОСТЕЙ")
            print("💡 Возможные причины:")
            print("   • Все коэффициенты вне диапазона 1.8-8.0")
            print("   • Вероятности вне целевого диапазона 45-88%")
            print("   • Слишком очевидные фавориты")
            print(f"   • Всего матчей было: {len(all_matches)}")
            return False
        
        # 3. Демонстрация результатов
        print(f"\n🎉 НАЙДЕНЫ КАЧЕСТВЕННЫЕ UNDERDOG ВОЗМОЖНОСТИ!")
        print("=" * 50)
        
        for i, match in enumerate(underdog_matches[:5], 1):  # Показываем топ-5
            analysis = match['underdog_analysis']
            prediction = analysis['prediction']
            metadata = match.get('tournament_metadata', {})
            
            print(f"\n{i}. {match['player1']} vs {match['player2']}")
            print(f"   🏆 {match['tournament']}")
            print(f"   🏟️ {match['surface']} • {metadata.get('level', 'Professional')}")
            print(f"   🎯 Андердог: {analysis['underdog']} (коэф. {analysis['underdog_odds']})")
            print(f"   📊 Вероятность взять сет: {prediction['probability']:.1%}")
            print(f"   💎 Качество: {analysis['quality_rating']}")
            print(f"   📈 Букмекер: {metadata.get('bookmaker', 'N/A')}")
        
        # 4. Сохранение данных
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'universal_tennis_data_{timestamp}.json'
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'season_context': self.get_season_context(),
                    'total_raw_matches': len(all_matches),
                    'underdog_matches_count': len(underdog_matches),
                    'matches': underdog_matches
                }, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n💾 Данные сохранены: {filename}")
        except Exception as e:
            print(f"\n⚠️ Не удалось сохранить файл: {e}")
        
        print(f"📊 Статистика:")
        print(f"   • Всего матчей найдено: {len(all_matches)}")
        print(f"   • Подходящих для underdog: {len(underdog_matches)}")
        print(f"   • Качественных возможностей: {len([m for m in underdog_matches if m['underdog_analysis']['quality_rating'] == 'HIGH'])}")
        
        return True

def main():
    """Главная функция"""
    integrator = UniversalTennisDataFix()
    
    try:
        success = integrator.run_universal_integration()
        
        if success:
            print(f"\n🚀 УСПЕХ! Универсальная интеграция работает!")
            print(f"\n📋 ЭТА СИСТЕМА РАБОТАЕТ КРУГЛЫЙ ГОД:")
            print("✅ Автоматически находит активные турниры")
            print("✅ Адаптируется к любому сезону")
            print("✅ Работает с Grand Slam, Masters, ATP/WTA")
            print("✅ Определяет покрытие и уровень турнира")
            print("✅ Фильтрует качественные underdog возможности")
        else:
            print(f"\n⚠️ Сейчас нет подходящих матчей")
            print(f"💡 Проверьте API ключ и соединение")
            
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()