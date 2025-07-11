#!/usr/bin/env python3
"""
🌍 UNIVERSAL Tennis Data Collector - Works Year-Round
Автоматически определяет текущие турниры и показывает актуальные матчи
БЕЗ привязки к конкретному турниру!
"""

from datetime import datetime, timedelta, date
from typing import Dict, List, Optional
import random

class UniversalTennisDataCollector:
    """Универсальный сборщик данных - работает круглый год"""
    
    def __init__(self):
        self.current_date = datetime.now()
        self.tournament_calendar = self._load_tournament_calendar()
        
    def _load_tournament_calendar(self) -> Dict:
        """Календарь турниров на весь год"""
        return {
            # Январь
            "2025-01-12": {"name": "Australian Open", "location": "Melbourne", "surface": "Hard", "level": "Grand Slam", "status": "major"},
            "2025-01-27": {"name": "Linz Open", "location": "Linz", "surface": "Hard", "level": "WTA 500", "status": "wta"},
            
            # Февраль  
            "2025-02-09": {"name": "Qatar Open", "location": "Doha", "surface": "Hard", "level": "WTA 1000", "status": "wta"},
            "2025-02-17": {"name": "Dallas Open", "location": "Dallas", "surface": "Hard", "level": "ATP 500", "status": "atp"},
            "2025-02-24": {"name": "Rio Open", "location": "Rio de Janeiro", "surface": "Clay", "level": "ATP 500", "status": "atp"},
            
            # Март
            "2025-03-05": {"name": "Indian Wells Masters", "location": "Indian Wells", "surface": "Hard", "level": "ATP 1000", "status": "masters"},
            "2025-03-19": {"name": "Miami Open", "location": "Miami", "surface": "Hard", "level": "ATP 1000", "status": "masters"},
            
            # Апрель
            "2025-04-07": {"name": "Charleston Open", "location": "Charleston", "surface": "Clay", "level": "WTA 500", "status": "wta"},
            "2025-04-14": {"name": "Monte Carlo Masters", "location": "Monaco", "surface": "Clay", "level": "ATP 1000", "status": "masters"},
            "2025-04-21": {"name": "Barcelona Open", "location": "Barcelona", "surface": "Clay", "level": "ATP 500", "status": "atp"},
            
            # Май
            "2025-05-05": {"name": "Madrid Open", "location": "Madrid", "surface": "Clay", "level": "ATP 1000", "status": "masters"},
            "2025-05-12": {"name": "Italian Open", "location": "Rome", "surface": "Clay", "level": "ATP 1000", "status": "masters"},
            "2025-05-19": {"name": "French Open", "location": "Paris", "surface": "Clay", "level": "Grand Slam", "status": "major"},
            
            # Июнь
            "2025-06-09": {"name": "Stuttgart Open", "location": "Stuttgart", "surface": "Grass", "level": "ATP 250", "status": "atp"},
            "2025-06-16": {"name": "Queen's Club", "location": "London", "surface": "Grass", "level": "ATP 500", "status": "atp"},
            "2025-06-23": {"name": "Eastbourne International", "location": "Eastbourne", "surface": "Grass", "level": "WTA 500", "status": "wta"},
            
            # Июль
            "2025-06-30": {"name": "Wimbledon", "location": "London", "surface": "Grass", "level": "Grand Slam", "status": "major"},
            "2025-07-14": {"name": "Hamburg Open", "location": "Hamburg", "surface": "Clay", "level": "ATP 500", "status": "atp"},
            "2025-07-21": {"name": "Los Cabos Open", "location": "Los Cabos", "surface": "Hard", "level": "ATP 250", "status": "atp"},
            
            # Август
            "2025-08-04": {"name": "Montreal Masters", "location": "Montreal", "surface": "Hard", "level": "ATP 1000", "status": "masters"},
            "2025-08-11": {"name": "Cincinnati Masters", "location": "Cincinnati", "surface": "Hard", "level": "ATP 1000", "status": "masters"},
            "2025-08-25": {"name": "US Open", "location": "New York", "surface": "Hard", "level": "Grand Slam", "status": "major"},
            
            # Сентябрь
            "2025-09-08": {"name": "Davis Cup", "location": "Various", "surface": "Various", "level": "Team Event", "status": "team"},
            "2025-09-15": {"name": "Laver Cup", "location": "San Francisco", "surface": "Hard", "level": "Exhibition", "status": "exhibition"},
            
            # Октябрь
            "2025-10-06": {"name": "China Open", "location": "Beijing", "surface": "Hard", "level": "ATP 500", "status": "atp"},
            "2025-10-13": {"name": "Shanghai Masters", "location": "Shanghai", "surface": "Hard", "level": "ATP 1000", "status": "masters"},
            "2025-10-20": {"name": "Vienna Open", "location": "Vienna", "surface": "Hard", "level": "ATP 500", "status": "atp"},
            
            # Ноябрь
            "2025-11-03": {"name": "Paris Masters", "location": "Paris", "surface": "Hard", "level": "ATP 1000", "status": "masters"},
            "2025-11-10": {"name": "ATP Finals", "location": "Turin", "surface": "Hard", "level": "Finals", "status": "finals"},
            "2025-11-17": {"name": "WTA Finals", "location": "Riyadh", "surface": "Hard", "level": "Finals", "status": "finals"},
        }
    
    def get_current_active_tournaments(self) -> List[Dict]:
        """Определяет какие турниры сейчас идут"""
        active_tournaments = []
        current_date = self.current_date.date()
        
        for start_date_str, tournament in self.tournament_calendar.items():
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
            
            # Определяем длительность турнира
            if tournament["level"] == "Grand Slam":
                duration = 15  # 2+ недели
            elif "1000" in tournament["level"] or "Masters" in tournament["level"]:
                duration = 12  # ~2 недели
            elif "500" in tournament["level"]:
                duration = 7   # неделя
            elif "Finals" in tournament["level"]:
                duration = 8   # неделя+
            else:
                duration = 6   # ATP 250, WTA 250
            
            end_date = start_date + timedelta(days=duration)
            
            # Проверяем попадает ли текущая дата в период турнира
            if start_date <= current_date <= end_date:
                tournament_copy = tournament.copy()
                tournament_copy["start_date"] = start_date_str
                tournament_copy["end_date"] = end_date.strftime("%Y-%m-%d")
                tournament_copy["days_running"] = (current_date - start_date).days + 1
                tournament_copy["days_remaining"] = (end_date - current_date).days
                active_tournaments.append(tournament_copy)
        
        return active_tournaments
    
    def get_upcoming_tournaments(self, days_ahead: int = 14) -> List[Dict]:
        """Турниры которые начнутся в ближайшее время"""
        upcoming = []
        current_date = self.current_date.date()
        
        for start_date_str, tournament in self.tournament_calendar.items():
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
            days_until = (start_date - current_date).days
            
            if 0 < days_until <= days_ahead:
                tournament_copy = tournament.copy()
                tournament_copy["start_date"] = start_date_str
                tournament_copy["days_until"] = days_until
                upcoming.append(tournament_copy)
        
        return sorted(upcoming, key=lambda x: x["days_until"])
    
    def generate_realistic_matches(self, tournaments: List[Dict]) -> List[Dict]:
        """Генерирует реалистичные матчи для активных турниров"""
        
        # Современные топ игроки
        top_players = {
            "atp": [
                {"name": "Jannik Sinner", "rank": 1, "country": "Italy"},
                {"name": "Carlos Alcaraz", "rank": 2, "country": "Spain"}, 
                {"name": "Alexander Zverev", "rank": 3, "country": "Germany"},
                {"name": "Daniil Medvedev", "rank": 4, "country": "Russia"},
                {"name": "Novak Djokovic", "rank": 5, "country": "Serbia"},
                {"name": "Andrey Rublev", "rank": 6, "country": "Russia"},
                {"name": "Casper Ruud", "rank": 7, "country": "Norway"},
                {"name": "Holger Rune", "rank": 8, "country": "Denmark"},
                {"name": "Grigor Dimitrov", "rank": 9, "country": "Bulgaria"},
                {"name": "Stefanos Tsitsipas", "rank": 10, "country": "Greece"},
                {"name": "Taylor Fritz", "rank": 11, "country": "USA"},
                {"name": "Tommy Paul", "rank": 12, "country": "USA"},
                {"name": "Ben Shelton", "rank": 15, "country": "USA"},
                {"name": "Frances Tiafoe", "rank": 18, "country": "USA"},
                {"name": "Felix Auger-Aliassime", "rank": 20, "country": "Canada"},
            ],
            "wta": [
                {"name": "Aryna Sabalenka", "rank": 1, "country": "Belarus"},
                {"name": "Iga Swiatek", "rank": 2, "country": "Poland"},
                {"name": "Coco Gauff", "rank": 3, "country": "USA"},
                {"name": "Jessica Pegula", "rank": 4, "country": "USA"},
                {"name": "Elena Rybakina", "rank": 5, "country": "Kazakhstan"},
                {"name": "Qinwen Zheng", "rank": 6, "country": "China"},
                {"name": "Jasmine Paolini", "rank": 7, "country": "Italy"},
                {"name": "Emma Navarro", "rank": 8, "country": "USA"},
                {"name": "Daria Kasatkina", "rank": 9, "country": "Russia"},
                {"name": "Barbora Krejcikova", "rank": 10, "country": "Czech Republic"},
                {"name": "Paula Badosa", "rank": 11, "country": "Spain"},
                {"name": "Danielle Collins", "rank": 12, "country": "USA"},
                {"name": "Jelena Ostapenko", "rank": 15, "country": "Latvia"},
                {"name": "Madison Keys", "rank": 18, "country": "USA"},
                {"name": "Emma Raducanu", "rank": 25, "country": "UK"},
            ]
        }
        
        # Добавляем других игроков
        other_players = []
        for rank in range(26, 100):
            other_players.append({
                "name": f"Player #{rank}",
                "rank": rank,
                "country": random.choice(["USA", "Spain", "France", "Italy", "Germany", "Australia"])
            })
        
        all_matches = []
        
        for tournament in tournaments:
            # Определяем тип турнира для выбора игроков
            if tournament["status"] in ["major", "masters", "finals"]:
                player_pool = top_players["atp"] + top_players["wta"]
                match_count = random.randint(8, 16)  # Больше матчей для важных турниров
            elif tournament["status"] in ["atp", "wta"]:
                if "1000" in tournament["level"]:
                    player_pool = top_players["atp"] + top_players["wta"] + other_players[:30]
                    match_count = random.randint(6, 12)
                elif "500" in tournament["level"]:
                    player_pool = top_players["atp"] + top_players["wta"] + other_players[:20]
                    match_count = random.randint(4, 8)
                else:  # 250
                    player_pool = top_players["atp"][5:] + top_players["wta"][5:] + other_players[:15]
                    match_count = random.randint(3, 6)
            else:
                player_pool = top_players["atp"] + top_players["wta"]
                match_count = random.randint(2, 5)
            
            # Генерируем матчи для турнира
            for i in range(match_count):
                player1 = random.choice(player_pool)
                player2 = random.choice([p for p in player_pool if p["name"] != player1["name"]])
                
                # Определяем раунд в зависимости от дней турнира
                days_running = tournament.get("days_running", 1)
                if days_running <= 2:
                    round_name = "R64" if tournament["status"] == "major" else "R32"
                elif days_running <= 5:
                    round_name = "R32" if tournament["status"] == "major" else "R16"
                elif days_running <= 8:
                    round_name = "R16" if tournament["status"] == "major" else "QF"
                elif days_running <= 12:
                    round_name = "QF" if tournament["status"] == "major" else "SF"
                else:
                    round_name = "SF" if tournament["status"] == "major" else "F"
                
                # Определяем статус матча
                status_options = ["upcoming", "upcoming", "live", "upcoming"]  # больше upcoming
                if days_running >= 3:
                    status_options.append("live")  # больше live матчей в середине турнира
                
                # Определяем время матча
                current_hour = self.current_date.hour
                if current_hour < 12:
                    time_options = ["12:00", "14:00", "16:00", "18:00"]
                elif current_hour < 18:
                    time_options = ["now", "16:00", "18:00", "20:00"]
                else:
                    time_options = ["tomorrow 12:00", "tomorrow 14:00"]
                
                match = {
                    "id": f"{tournament['name'].lower().replace(' ', '_')}_{i+1}",
                    "player1": player1["name"],
                    "player2": player2["name"],
                    "tournament": tournament["name"],
                    "location": tournament["location"],
                    "surface": tournament["surface"],
                    "level": tournament["level"],
                    "date": self.current_date.strftime("%Y-%m-%d"),
                    "time": random.choice(time_options),
                    "round": round_name,
                    "court": self._get_court_name(tournament, i),
                    "status": random.choice(status_options),
                    "source": "universal_collector",
                    "tournament_info": tournament
                }
                
                all_matches.append(match)
        
        return all_matches
    
    def _get_court_name(self, tournament: Dict, match_index: int) -> str:
        """Генерирует реалистичные названия кортов"""
        if tournament["name"] == "Wimbledon":
            courts = ["Centre Court", "Court 1", "Court 2", "Court 3", "Court 12", "Court 18"]
        elif tournament["name"] == "US Open":
            courts = ["Arthur Ashe Stadium", "Louis Armstrong Stadium", "Grandstand", "Court 17", "Court 5"]
        elif tournament["name"] == "French Open":
            courts = ["Philippe Chatrier", "Suzanne Lenglen", "Simonne Mathieu", "Court 14", "Court 7"]
        elif tournament["name"] == "Australian Open":
            courts = ["Rod Laver Arena", "John Cain Arena", "Margaret Court Arena", "Court 3", "Court 7"]
        elif tournament["status"] == "masters":
            courts = ["Stadium Court", "Court 1", "Court 2", "Practice Court 1"]
        else:
            courts = ["Center Court", "Court 1", "Court 2", "Court 3"]
        
        return random.choice(courts)
    
    def get_current_matches(self) -> List[Dict]:
        """Возвращает пустой список если нет реальных данных"""
        
        # Проверяем активные турниры
        active_tournaments = self.get_current_active_tournaments()
        
        if not active_tournaments:
            return []
        
        # Честно сообщаем что нет реальных расписаний
        print(f"🏆 Активный турнир: {active_tournaments[0]['name']}")
        print("💡 Для получения реальных матчей используйте ручное обновление API")
        
        return []  # Не генерируем выдуманные матчи
    
    def _generate_preparation_matches(self, upcoming_tournaments: List[Dict]) -> List[Dict]:
        """Генерирует подготовительные матчи перед турниром"""
        matches = []
        for tournament in upcoming_tournaments:
            match = {
                "id": f"prep_{tournament['name'].lower().replace(' ', '_')}",
                "player1": "Carlos Alcaraz", 
                "player2": "Practice Partner",
                "tournament": f"Preparing for {tournament['name']}",
                "location": tournament["location"],
                "surface": tournament["surface"],
                "level": "Practice",
                "date": self.current_date.strftime("%Y-%m-%d"),
                "time": "TBD",
                "round": "Practice",
                "court": "Practice Court",
                "status": "preparation",
                "source": "universal_collector",
                "days_until": tournament["days_until"]
            }
            matches.append(match)
        
        return matches
    
    def _generate_training_matches(self) -> List[Dict]:
        """Генерирует тренировочные матчи в межсезонье"""
        return [{
            "id": "training_session",
            "player1": "Jannik Sinner",
            "player2": "Sparring Partner", 
            "tournament": "Off-Season Training",
            "location": "Training Center",
            "surface": "Hard",
            "level": "Training",
            "date": self.current_date.strftime("%Y-%m-%d"),
            "time": "TBD",
            "round": "Training",
            "court": "Training Court",
            "status": "training",
            "source": "universal_collector"
        }]
    
    def _get_season_context(self) -> str:
        """Контекст текущего сезона"""
        month = self.current_date.month
        
        if month in [1, 2]:
            return "Hard Court Season - Australian Summer"
        elif month in [3, 4, 5]:
            return "Clay Court Season - European Spring"  
        elif month in [6, 7]:
            return "Grass Court Season - Wimbledon Period"
        elif month in [8, 9]:
            return "Hard Court Season - US Open Series"
        elif month in [10, 11]:
            return "Indoor Season - Year End Finals"
        else:  # December
            return "Off Season - Preparation Period"
    
    def _get_tournament_status(self, tournament: Dict) -> str:
        """Статус турнира"""
        days_running = tournament.get("days_running", 0)
        days_remaining = tournament.get("days_remaining", 0)
        
        if days_running == 1:
            return "🚀 Just Started"
        elif days_remaining <= 2:
            return "🏁 Final Stages"
        elif days_running <= 3:
            return "🔥 Early Rounds"
        else:
            return "⚡ Main Draw"
    
    def get_summary(self) -> Dict:
        """Сводка по текущему состоянию тенниса"""
        active = self.get_current_active_tournaments()
        upcoming = self.get_upcoming_tournaments(14)
        
        return {
            "current_date": self.current_date.strftime("%Y-%m-%d"),
            "season_context": self._get_season_context(),
            "active_tournaments": len(active),
            "active_tournament_names": [t["name"] for t in active],
            "upcoming_tournaments": len(upcoming),
            "next_major": self._get_next_major(),
            "matches_available": len(self.get_current_matches())
        }
    
    def _get_next_major(self) -> Optional[str]:
        """Следующий Grand Slam"""
        current_date = self.current_date.date()
        
        majors = [
            ("2025-01-12", "Australian Open"),
            ("2025-05-19", "French Open"), 
            ("2025-06-30", "Wimbledon"),
            ("2025-08-25", "US Open")
        ]
        
        for date_str, name in majors:
            major_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if major_date >= current_date:
                days_until = (major_date - current_date).days
                return f"{name} (in {days_until} days)"
        
        return "Australian Open 2026 (next year)"


class UniversalOddsCollector:
    """Универсальный сборщик коэффициентов"""
    
    def __init__(self):
        # Более актуальные рейтинги (июль 2025)
        self.player_rankings = {
            # ATP Top 20
            "jannik sinner": 1, "carlos alcaraz": 2, "alexander zverev": 3,
            "daniil medvedev": 4, "novak djokovic": 5, "andrey rublev": 6,
            "casper ruud": 7, "holger rune": 8, "grigor dimitrov": 9,
            "stefanos tsitsipas": 10, "taylor fritz": 11, "tommy paul": 12,
            "ben shelton": 15, "frances tiafoe": 18, "felix auger-aliassime": 20,
            
            # WTA Top 20
            "aryna sabalenka": 1, "iga swiatek": 2, "coco gauff": 3,
            "jessica pegula": 4, "elena rybakina": 5, "qinwen zheng": 6,
            "jasmine paolini": 7, "emma navarro": 8, "daria kasatkina": 9,
            "barbora krejcikova": 10, "paula badosa": 11, "danielle collins": 12,
            "jelena ostapenko": 15, "madison keys": 18, "emma raducanu": 25,
            
            # Другие известные игроки
            "fabio fognini": 85, "arthur rinderknech": 45, "yannick hanfmann": 95,
            "jacob fearnley": 320, "joao fonseca": 145, "katie boulter": 28,
            "renata zarazua": 180, "caroline dolehide": 85, "carson branstine": 125
        }
    
    def get_player_ranking(self, player_name: str) -> int:
        """Получить рейтинг игрока"""
        name_lower = player_name.lower()
        
        # Прямое совпадение
        if name_lower in self.player_rankings:
            return self.player_rankings[name_lower]
        
        # Поиск по частям имени
        for known_player, rank in self.player_rankings.items():
            if any(part in known_player for part in name_lower.split()):
                return rank
        
        # Если игрок неизвестен, генерируем рейтинг на основе контекста
        if "practice" in name_lower or "sparring" in name_lower:
            return random.randint(100, 300)
        
        return random.randint(30, 80)  # Средний рейтинг для неизвестных
    
    def generate_realistic_odds(self, matches: List[Dict]) -> Dict[str, Dict]:
        """Генерирует реалистичные коэффициенты"""
        odds_data = {}
        
        for match in matches:
            match_id = match["id"]
            
            # Получаем рейтинги
            p1_rank = self.get_player_ranking(match["player1"])
            p2_rank = self.get_player_ranking(match["player2"])
            
            # Корректировки на основе покрытия
            surface = match.get("surface", "Hard")
            surface_adjustments = {
                "Clay": {"specialists": ["rafael nadal", "carlos alcaraz"], "bonus": 0.1},
                "Grass": {"specialists": ["novak djokovic"], "bonus": 0.15},
                "Hard": {"specialists": ["daniil medvedev", "jannik sinner"], "bonus": 0.05}
            }
            
            # Базовые коэффициенты на основе рейтингов
            rank_diff = p2_rank - p1_rank
            
            if rank_diff > 30:  # P1 намного сильнее
                base_p1_odds = 1.3 + (rank_diff * 0.01)
                base_p2_odds = 4.0 - (rank_diff * 0.02)
            elif rank_diff < -30:  # P2 намного сильнее  
                base_p1_odds = 4.0 + (abs(rank_diff) * 0.02)
                base_p2_odds = 1.3 + (abs(rank_diff) * 0.01)
            else:  # Примерно равные
                base_p1_odds = 1.8 + (rank_diff * 0.01)
                base_p2_odds = 2.2 - (rank_diff * 0.01)
            
            # Применяем корректировки покрытия
            p1_name_lower = match["player1"].lower()
            p2_name_lower = match["player2"].lower()
            
            if surface in surface_adjustments:
                specialists = surface_adjustments[surface]["specialists"]
                bonus = surface_adjustments[surface]["bonus"]
                
                if any(spec in p1_name_lower for spec in specialists):
                    base_p1_odds *= (1 - bonus)
                    base_p2_odds *= (1 + bonus)
                elif any(spec in p2_name_lower for spec in specialists):
                    base_p1_odds *= (1 + bonus)
                    base_p2_odds *= (1 - bonus)
            
            # Ограничиваем диапазон
            p1_odds = max(1.1, min(base_p1_odds, 10.0))
            p2_odds = max(1.1, min(base_p2_odds, 10.0))
            
            # Выбираем букмекера
            bookmakers = ["Bet365", "Pinnacle", "William Hill", "Unibet", "888Sport"]
            
            odds_data[match_id] = {
                "match_info": match,
                "best_markets": {
                    "winner": {
                        "player1": {
                            "odds": round(p1_odds, 2),
                            "bookmaker": random.choice(bookmakers)
                        },
                        "player2": {
                            "odds": round(p2_odds, 2),
                            "bookmaker": random.choice(bookmakers)
                        }
                    }
                },
                "market_info": {
                    "surface_factor": surface,
                    "ranking_difference": rank_diff,
                    "tournament_level": match.get("level", "Regular")
                }
            }
        
        return odds_data


# Пример использования
if __name__ == "__main__":
    print("🌍 UNIVERSAL TENNIS DATA COLLECTOR - ТЕСТИРОВАНИЕ")
    print("=" * 60)
    
    collector = UniversalTennisDataCollector()
    odds_collector = UniversalOddsCollector()
    
    # Показываем сводку
    summary = collector.get_summary()
    print(f"📅 Дата: {summary['current_date']}")
    print(f"🏟️ Сезон: {summary['season_context']}")
    print(f"🏆 Активных турниров: {summary['active_tournaments']}")
    if summary['active_tournament_names']:
        print(f"📋 Текущие турниры: {', '.join(summary['active_tournament_names'])}")
    print(f"🔜 Следующий Grand Slam: {summary['next_major']}")
    print(f"🎾 Доступно матчей: {summary['matches_available']}")
    
    print("\n🎾 ТЕКУЩИЕ МАТЧИ:")
    print("-" * 50)
    
    # Получаем текущие матчи
    current_matches = collector.get_current_matches()
    
    for i, match in enumerate(current_matches[:8], 1):
        status_emoji = {
            "live": "🔴 LIVE",
            "upcoming": "⏰ Upcoming", 
            "preparation": "🏋️ Prep",
            "training": "💪 Training"
        }
        
        status = status_emoji.get(match["status"], "📅 Scheduled")
        print(f"{i}. {match['player1']} vs {match['player2']}")
        print(f"   🏆 {match['tournament']} ({match['level']})")
        print(f"   📍 {match['location']} • {match['surface']} • {match['round']}")
        print(f"   {status} • {match['date']} {match['time']}")
        
        if "season_context" in match:
            print(f"   🌍 Context: {match['season_context']}")
        print()
    
    # Тестируем коэффициенты
    print("💰 ТЕСТ КОЭФФИЦИЕНТОВ:")
    print("-" * 30)
    
    odds = odds_collector.generate_realistic_odds(current_matches[:3])
    
    for match_id, odds_info in odds.items():
        match_info = odds_info["match_info"]
        winner_odds = odds_info["best_markets"]["winner"]
        
        print(f"🎾 {match_info['player1']} vs {match_info['player2']}")
        print(f"   💰 {winner_odds['player1']['odds']} vs {winner_odds['player2']['odds']}")
        print(f"   📊 Ranking diff: {odds_info['market_info']['ranking_difference']}")
        print()
    
    print("✅ Универсальная система работает круглый год!")
    print("🌍 Автоматически подстраивается под текущие турниры!")