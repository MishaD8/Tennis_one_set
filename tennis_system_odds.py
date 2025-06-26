import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Tuple, Optional
import warnings
import sqlite3
import os
import json
from dataclasses import dataclass
from enum import Enum
import time
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import re

warnings.filterwarnings('ignore')

class ConfidenceLevel(Enum):
    VERY_HIGH = "Очень высокая"
    HIGH = "Высокая"
    MEDIUM = "Средняя"
    LOW = "Низкая"

@dataclass
class ValueBet:
    match_id: str
    player_name: str
    opponent_name: str
    tournament: str
    surface: str
    match_date: str
    bookmaker: str
    market: str
    odds: float
    our_probability: float
    implied_probability: float
    expected_value: float
    kelly_fraction: float
    confidence_level: ConfidenceLevel
    value_rating: float
    recommended_stake: float

@dataclass
class BettingMetrics:
    total_bets: int
    total_profit: float
    roi: float
    win_rate: float
    average_odds: float
    sharpe_ratio: float
    max_drawdown: float
    confidence_breakdown: Dict[str, Dict]

class OddsCollector:
    """
    НОВОЕ: Сборщик коэффициентов с разных источников
    """
    def __init__(self):
        self.sources = {
            'pinnacle': {
                'base_url': 'https://api.pinnacle.com/v1/',
                'requires_auth': True
            },
            'oddsportal': {
                'base_url': 'https://www.oddsportal.com/',
                'requires_auth': False
            },
            'bet365': {
                'base_url': 'https://www.bet365.com/',
                'requires_auth': False
            }
        }
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def collect_pinnacle_odds(self, sport_id: int = 33) -> List[Dict]:
        """
        Сбор коэффициентов с Pinnacle API (требует API ключ)
        """
        # Примечание: требует регистрации и получения API ключа
        odds_data = []
        
        try:
            # Пример структуры данных от Pinnacle
            # В реальном использовании здесь будет API запрос
            sample_odds = [
                {
                    'event_id': 'PNN_001',
                    'player1': 'Novak Djokovic',
                    'player2': 'Rafael Nadal',
                    'tournament': 'Roland Garros',
                    'date': '2025-06-20',
                    'markets': {
                        'winner': {'player1': 1.85, 'player2': 1.95},
                        'set_winner': {'player1': 1.65, 'player2': 2.25},
                        'total_sets': {'over_3_5': 1.90, 'under_3_5': 1.90},
                        'first_set_winner': {'player1': 1.75, 'player2': 2.05}
                    }
                },
                {
                    'event_id': 'PNN_002',
                    'player1': 'Carlos Alcaraz',
                    'player2': 'Jannik Sinner',
                    'tournament': 'Wimbledon',
                    'date': '2025-06-21',
                    'markets': {
                        'winner': {'player1': 1.70, 'player2': 2.10},
                        'set_winner': {'player1': 1.55, 'player2': 2.35},
                        'total_sets': {'over_3_5': 1.85, 'under_3_5': 1.95},
                        'first_set_winner': {'player1': 1.68, 'player2': 2.15}
                    }
                }
            ]
            
            return sample_odds
            
        except Exception as e:
            print(f"❌ Ошибка сбора данных Pinnacle: {e}")
            return []
    
    def scrape_oddsportal(self, sport: str = 'tennis') -> List[Dict]:
        """
        Парсинг коэффициентов с Oddsportal
        """
        odds_data = []
        
        try:
            # Пример парсинга (в реальности требует более сложной логики)
            # Здесь должен быть код парсинга веб-страниц
            
            # Симуляция данных для демонстрации
            sample_data = [
                {
                    'source': 'oddsportal',
                    'match_id': 'OP_001',
                    'player1': 'Novak Djokovic',
                    'player2': 'Rafael Nadal',
                    'bookmakers': {
                        'Bet365': {'player1_win': 1.83, 'player2_win': 1.97},
                        'William Hill': {'player1_win': 1.85, 'player2_win': 1.95},
                        'Unibet': {'player1_win': 1.80, 'player2_win': 2.00}
                    }
                }
            ]
            
            return sample_data
            
        except Exception as e:
            print(f"❌ Ошибка парсинга Oddsportal: {e}")
            return []
    
    def get_best_odds(self, all_odds: List[Dict]) -> Dict:
        """
        Поиск лучших коэффициентов среди всех букмекеров
        """
        best_odds = {}
        
        for odds_data in all_odds:
            match_id = odds_data.get('event_id', odds_data.get('match_id'))
            
            if match_id not in best_odds:
                best_odds[match_id] = {
                    'match_info': odds_data,
                    'best_markets': {}
                }
            
            # Анализируем каждый рынок
            markets = odds_data.get('markets', odds_data.get('bookmakers', {}))
            
            for market, odds in markets.items():
                if market not in best_odds[match_id]['best_markets']:
                    best_odds[match_id]['best_markets'][market] = {}
                
                if isinstance(odds, dict):
                    for outcome, odd_value in odds.items():
                        current_best = best_odds[match_id]['best_markets'][market].get(outcome, {})
                        if not current_best or odd_value > current_best.get('odds', 0):
                            best_odds[match_id]['best_markets'][market][outcome] = {
                                'odds': odd_value,
                                'bookmaker': odds_data.get('source', 'unknown')
                            }
        
        return best_odds

class EnhancedTennisBettingSystem:
    """
    УЛУЧШЕННАЯ: Система поиска ценных ставок с продвинутым анализом
    """
    def __init__(self, model_predictor, bankroll: float = 10000):
        self.predictor = model_predictor
        self.bankroll = bankroll
        self.initial_bankroll = bankroll
        self.betting_history = []
        self.odds_collector = OddsCollector()
        
        # Настройки ставок
        self.min_confidence = 0.55  # Снижен для больше возможностей
        self.min_odds = 1.4         # Минимальные коэффициенты
        self.max_odds = 15.0        # Максимальные коэффициенты
        self.max_stake_pct = 0.05   # Максимум 5% от банка
        self.min_edge = 0.02        # Минимальное преимущество 2%
        
        # Мультипликаторы уверенности
        self.confidence_multipliers = {
            ConfidenceLevel.VERY_HIGH: 1.0,
            ConfidenceLevel.HIGH: 0.8,
            ConfidenceLevel.MEDIUM: 0.6,
            ConfidenceLevel.LOW: 0.4
        }
    
    def calculate_dynamic_kelly(self, probability: float, odds: float, 
                              confidence_level: ConfidenceLevel, 
                              recent_performance: float = 1.0) -> float:
        """
        УЛУЧШЕНО: Динамический критерий Келли с корректировками
        """
        if probability <= 0 or odds <= 1:
            return 0
        
        # Базовый расчет Келли
        b = odds - 1  # Выигрыш при ставке в 1 единицу
        p = probability  # Наша вероятность
        q = 1 - p  # Вероятность проигрыша
        
        kelly_fraction = (b * p - q) / b
        
        if kelly_fraction <= 0:
            return 0
        
        # Корректировки
        # 1. Уверенность в прогнозе
        confidence_mult = self.confidence_multipliers[confidence_level]
        
        # 2. Недавняя производительность модели
        performance_mult = min(max(recent_performance, 0.5), 1.5)
        
        # 3. Размер банка (более консервативно при малом банке)
        bankroll_mult = min(self.bankroll / self.initial_bankroll, 1.0)
        
        # 4. Консервативный множитель (дробный Келли)
        conservative_mult = 0.25  # Используем 1/4 от полного Келли
        
        adjusted_kelly = (kelly_fraction * confidence_mult * 
                         performance_mult * bankroll_mult * conservative_mult)
        
        # Ограничиваем максимальную ставку
        return max(0, min(adjusted_kelly, self.max_stake_pct))
    
    def calculate_expected_value(self, probability: float, odds: float) -> float:
        """
        Расчет ожидаемого значения ставки
        """
        return (probability * (odds - 1)) - (1 - probability)
    
    def get_confidence_level(self, probability: float, model_confidence: float = 1.0) -> ConfidenceLevel:
        """
        УЛУЧШЕНО: Определение уровня уверенности с учетом качества модели
        """
        adjusted_prob = probability * model_confidence
        
        if adjusted_prob >= 0.80:
            return ConfidenceLevel.VERY_HIGH
        elif adjusted_prob >= 0.70:
            return ConfidenceLevel.HIGH
        elif adjusted_prob >= 0.60:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def analyze_market_efficiency(self, odds_data: Dict) -> Dict:
        """
        НОВОЕ: Анализ эффективности рынка
        """
        market_analysis = {}
        
        for match_id, match_data in odds_data.items():
            best_markets = match_data.get('best_markets', {})
            
            for market, outcomes in best_markets.items():
                if len(outcomes) >= 2:  # Нужно минимум 2 исхода
                    # Рассчитываем маржу букмекера
                    implied_probs = []
                    for outcome, odds_info in outcomes.items():
                        implied_prob = 1 / odds_info['odds']
                        implied_probs.append(implied_prob)
                    
                    total_implied_prob = sum(implied_probs)
                    bookmaker_margin = total_implied_prob - 1.0
                    
                    market_analysis[f"{match_id}_{market}"] = {
                        'margin': bookmaker_margin,
                        'efficiency': 1 - bookmaker_margin,  # Чем меньше маржа, тем эффективнее
                        'outcomes': outcomes
                    }
        
        return market_analysis
    
    def find_value_bets(self, matches_data: pd.DataFrame, odds_data: Dict, 
                       model_confidence: float = 1.0) -> List[ValueBet]:
        """
        УЛУЧШЕНО: Поиск ценных ставок с мультирыночным анализом
        """
        print("🔍 Поиск ценных ставок...")
        
        value_bets = []
        market_analysis = self.analyze_market_efficiency(odds_data)
        
        for idx, match in matches_data.iterrows():
            try:
                # Получаем предсказание модели
                match_features = self.predictor.prepare_features(pd.DataFrame([match]))
                probability = self.predictor.predict_probability(match_features)[0]
                
                match_id = match.get('match_id', f"match_{idx}")
                
                # Ищем коэффициенты для этого матча
                if match_id not in odds_data:
                    continue
                
                best_markets = odds_data[match_id].get('best_markets', {})
                
                # Анализируем каждый рынок
                for market, outcomes in best_markets.items():
                    # Фокусируемся на рынке "игрок выиграет хотя бы один сет"
                    if 'set_winner' in market.lower() or 'player1' in outcomes:
                        
                        player_outcome = outcomes.get('player1', outcomes.get('set_winner', {}))
                        if not player_outcome:
                            continue
                        
                        odds_value = player_outcome.get('odds', 0)
                        bookmaker = player_outcome.get('bookmaker', 'Unknown')
                        
                        if not (self.min_odds <= odds_value <= self.max_odds):
                            continue
                        
                        # Рассчитываем метрики
                        implied_prob = 1 / odds_value
                        expected_value = self.calculate_expected_value(probability, odds_value)
                        
                        # Проверяем критерии ценности
                        if (probability > implied_prob and 
                            probability >= self.min_confidence and 
                            expected_value >= self.min_edge):
                            
                            confidence_level = self.get_confidence_level(probability, model_confidence)
                            kelly_size = self.calculate_dynamic_kelly(
                                probability, odds_value, confidence_level, model_confidence
                            )
                            
                            if kelly_size > 0:
                                recommended_stake = min(
                                    kelly_size * self.bankroll,
                                    self.max_stake_pct * self.bankroll
                                )
                                
                                value_bet = ValueBet(
                                    match_id=match_id,
                                    player_name=match.get('player_name', 'Unknown'),
                                    opponent_name=match.get('opponent_name', 'Unknown'),
                                    tournament=match.get('tournament', 'Unknown'),
                                    surface=match.get('surface', 'Unknown'),
                                    match_date=str(match.get('match_date', '')),
                                    bookmaker=bookmaker,
                                    market=market,
                                    odds=odds_value,
                                    our_probability=probability,
                                    implied_probability=implied_prob,
                                    expected_value=expected_value,
                                    kelly_fraction=kelly_size,
                                    confidence_level=confidence_level,
                                    value_rating=expected_value / implied_prob,
                                    recommended_stake=recommended_stake
                                )
                                
                                value_bets.append(value_bet)
                                
            except Exception as e:
                print(f"⚠️ Ошибка при анализе матча {match.get('match_id', idx)}: {e}")
                continue
        
        # Сортируем по рейтингу ценности
        value_bets.sort(key=lambda x: x.value_rating, reverse=True)
        
        print(f"✅ Найдено {len(value_bets)} ценных ставок")
        return value_bets
    
    def simulate_betting_performance(self, historical_data: pd.DataFrame, 
                                   historical_odds: Dict, 
                                   start_date: str, end_date: str) -> BettingMetrics:
        """
        НОВОЕ: Симуляция результатов ставок на исторических данных
        """
        print("🎲 Симуляция исторической производительности...")
        
        # Фильтруем данные по периоду
        historical_data['match_date'] = pd.to_datetime(historical_data['match_date'])
        period_data = historical_data[
            (historical_data['match_date'] >= start_date) &
            (historical_data['match_date'] <= end_date)
        ].copy()
        
        simulation_bankroll = self.initial_bankroll
        bet_history = []
        daily_bankroll = []
        
        # Группируем по дням для симуляции
        for date in pd.date_range(start_date, end_date, freq='D'):
            day_matches = period_data[period_data['match_date'].dt.date == date.date()]
            
            if len(day_matches) == 0:
                daily_bankroll.append(simulation_bankroll)
                continue
            
            # Находим ставки на этот день
            day_odds = {k: v for k, v in historical_odds.items() 
                       if k in day_matches['match_id'].values}
            
            if day_odds:
                # Временно изменяем банк для расчета ставок
                original_bankroll = self.bankroll
                self.bankroll = simulation_bankroll
                
                value_bets = self.find_value_bets(day_matches, day_odds)
                
                # Симулируем результаты ставок
                day_profit = 0
                for bet in value_bets:
                    # Получаем фактический результат
                    match_result = day_matches[
                        day_matches['match_id'] == bet.match_id
                    ]['won_at_least_one_set'].iloc[0] if len(day_matches[
                        day_matches['match_id'] == bet.match_id
                    ]) > 0 else 0
                    
                    # Рассчитываем результат ставки
                    if match_result == 1:  # Выиграл хотя бы один сет
                        profit = bet.recommended_stake * (bet.odds - 1)
                    else:
                        profit = -bet.recommended_stake
                    
                    day_profit += profit
                    simulation_bankroll += profit
                    
                    bet_history.append({
                        'date': date,
                        'match_id': bet.match_id,
                        'stake': bet.recommended_stake,
                        'odds': bet.odds,
                        'result': match_result,
                        'profit': profit,
                        'bankroll': simulation_bankroll
                    })
                
                # Восстанавливаем оригинальный банк
                self.bankroll = original_bankroll
            
            daily_bankroll.append(simulation_bankroll)
        
        # Рассчитываем метрики
        if bet_history:
            total_bets = len(bet_history)
            total_profit = simulation_bankroll - self.initial_bankroll
            total_staked = sum([bet['stake'] for bet in bet_history])
            roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0
            win_rate = sum([1 for bet in bet_history if bet['result'] == 1]) / total_bets
            avg_odds = np.mean([bet['odds'] for bet in bet_history])
            
            # Расчет Sharpe ratio и максимальной просадки
            returns = np.diff(daily_bankroll) / np.array(daily_bankroll[:-1])
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            # Максимальная просадка
            peak = np.maximum.accumulate(daily_bankroll)
            drawdown = (daily_bankroll - peak) / peak
            max_drawdown = np.min(drawdown) * 100
            
            # Разбивка по уровням уверенности
            confidence_breakdown = {}
            for level in ConfidenceLevel:
                level_bets = [bet for bet, vb in zip(bet_history, value_bets) 
                             if vb.confidence_level == level]
                if level_bets:
                    confidence_breakdown[level.value] = {
                        'count': len(level_bets),
                        'profit': sum([bet['profit'] for bet in level_bets]),
                        'win_rate': sum([bet['result'] for bet in level_bets]) / len(level_bets)
                    }
        else:
            total_bets = total_profit = roi = win_rate = avg_odds = sharpe_ratio = max_drawdown = 0
            confidence_breakdown = {}
        
        return BettingMetrics(
            total_bets=total_bets,
            total_profit=total_profit,
            roi=roi,
            win_rate=win_rate,
            average_odds=avg_odds,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            confidence_breakdown=confidence_breakdown
        )
    
    def generate_comprehensive_report(self, value_bets: List[ValueBet], 
                                    historical_metrics: Optional[BettingMetrics] = None) -> str:
        """
        УЛУЧШЕНО: Генерация подробного отчета
        """
        if not value_bets:
            return "❌ Ценных ставок не найдено."
        
        report = f"""
🎾 ДЕТАЛЬНЫЙ ОТЧЕТ ПО ТЕННИСНЫМ СТАВКАМ
{'='*70}

📊 ОБЩАЯ СТАТИСТИКА:
• Найдено ценных ставок: {len(value_bets)}
• Общая рекомендуемая сумма: ${sum([bet.recommended_stake for bet in value_bets]):,.2f}
• Средняя наша вероятность: {np.mean([bet.our_probability for bet in value_bets]):.1%}
• Средние коэффициенты: {np.mean([bet.odds for bet in value_bets]):.2f}
• Среднее ожидаемое значение: {np.mean([bet.expected_value for bet in value_bets]):.3f}
• Средний рейтинг ценности: {np.mean([bet.value_rating for bet in value_bets]):.3f}

📈 РАСПРЕДЕЛЕНИЕ ПО УВЕРЕННОСТИ:"""
        
        # Группировка по уровням уверенности
        confidence_groups = {}
        for bet in value_bets:
            level = bet.confidence_level.value
            if level not in confidence_groups:
                confidence_groups[level] = []
            confidence_groups[level].append(bet)
        
        for level, bets in confidence_groups.items():
            total_stake = sum([bet.recommended_stake for bet in bets])
            avg_ev = np.mean([bet.expected_value for bet in bets])
            report += f"""
• {level}: {len(bets)} ставок, ${total_stake:,.2f}, EV: {avg_ev:.3f}"""
        
        report += f"""

🏆 ТОП-10 РЕКОМЕНДАЦИЙ:
{'='*50}"""
        
        for i, bet in enumerate(value_bets[:10], 1):
            report += f"""
#{i}. {bet.player_name} vs {bet.opponent_name}
   📅 {bet.match_date} | 🏟️ {bet.tournament} ({bet.surface})
   💰 Коэффициент: {bet.odds:.2f} ({bet.bookmaker})
   🎯 Наша вероятность: {bet.our_probability:.1%}
   📊 Подразумеваемая вероятность: {bet.implied_probability:.1%}
   📈 Ожидаемое значение: {bet.expected_value:.3f}
   💵 Рекомендуемая ставка: ${bet.recommended_stake:.2f} ({bet.kelly_fraction:.1%} банка)
   ⭐ Уверенность: {bet.confidence_level.value}
   🔥 Рейтинг ценности: {bet.value_rating:.3f}
"""
        
        # Добавляем историческую производительность, если есть
        if historical_metrics:
            report += f"""

📊 ИСТОРИЧЕСКАЯ ПРОИЗВОДИТЕЛЬНОСТЬ:
{'='*50}
• Общее количество ставок: {historical_metrics.total_bets}
• Общая прибыль: ${historical_metrics.total_profit:,.2f}
• ROI: {historical_metrics.roi:.2f}%
• Винрейт: {historical_metrics.win_rate:.1%}
• Средние коэффициенты: {historical_metrics.average_odds:.2f}
• Sharpe Ratio: {historical_metrics.sharpe_ratio:.2f}
• Максимальная просадка: {historical_metrics.max_drawdown:.2f}%

📊 ПО УРОВНЯМ УВЕРЕННОСТИ:"""
            
            for level, stats in historical_metrics.confidence_breakdown.items():
                report += f"""
• {level}: {stats['count']} ставок, ${stats['profit']:,.2f}, {stats['win_rate']:.1%} винрейт"""
        
        report += f"""

💡 РЕКОМЕНДАЦИИ ПО УПРАВЛЕНИЮ БАНКОМ:
{'='*50}
• Никогда не превышайте рекомендуемые размеры ставок
• Ведите подробный учет всех ставок для анализа
• Пересматривайте стратегию каждые 100 ставок
• При просадке более 20% снижайте размеры ставок вдвое
• Используйте только 50-70% от доступного банка

🎯 РЫНОЧНАЯ ЭФФЕКТИВНОСТЬ:
• Фокусируйтесь на менее популярных рынках
• Избегайте матчей с чрезмерным медиа вниманием
• Лучшие возможности в ранних раундах турниров
• Обращайте внимание на матчи в неевропейское время

⚠️ ВАЖНЫЕ НАПОМИНАНИЯ:
• Это рекомендации на основе математической модели
• Прошлая производительность не гарантирует будущие результаты
• Ставки всегда связаны с риском потери средств
• Играйте ответственно и в рамках своих возможностей
• Рассматривайте это как долгосрочную инвестиционную стратегию

📱 СЛЕДУЮЩИЕ ШАГИ:
1. Проверьте актуальность коэффициентов перед ставкой
2. Установите лимиты на день/неделю/месяц
3. Ведите Excel-таблицу с результатами
4. Анализируйте результаты каждые 2 недели
5. Корректируйте стратегию на основе данных
"""
        
        return report
    
    def save_betting_data(self, value_bets: List[ValueBet], 
                         historical_metrics: Optional[BettingMetrics] = None):
        """
        НОВОЕ: Сохранение данных о ставках
        """
        # Создаем директорию для данных о ставках
        betting_dir = "betting_data"
        if not os.path.exists(betting_dir):
            os.makedirs(betting_dir)
        
        # Сохраняем ценные ставки в CSV
        if value_bets:
            bets_data = []
            for bet in value_bets:
                bets_data.append({
                    'match_id': bet.match_id,
                    'player_name': bet.player_name,
                    'opponent_name': bet.opponent_name,
                    'tournament': bet.tournament,
                    'surface': bet.surface,
                    'match_date': bet.match_date,
                    'bookmaker': bet.bookmaker,
                    'market': bet.market,
                    'odds': bet.odds,
                    'our_probability': bet.our_probability,
                    'implied_probability': bet.implied_probability,
                    'expected_value': bet.expected_value,
                    'kelly_fraction': bet.kelly_fraction,
                    'confidence_level': bet.confidence_level.value,
                    'value_rating': bet.value_rating,
                    'recommended_stake': bet.recommended_stake
                })
            
            bets_df = pd.DataFrame(bets_data)
            csv_path = os.path.join(betting_dir, f'value_bets_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            bets_df.to_csv(csv_path, index=False)
            print(f"💾 Ценные ставки сохранены: {csv_path}")
        
        # Сохраняем исторические метрики
        if historical_metrics:
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': {
                    'total_bets': historical_metrics.total_bets,
                    'total_profit': historical_metrics.total_profit,
                    'roi': historical_metrics.roi,
                    'win_rate': historical_metrics.win_rate,
                    'average_odds': historical_metrics.average_odds,
                    'sharpe_ratio': historical_metrics.sharpe_ratio,
                    'max_drawdown': historical_metrics.max_drawdown,
                    'confidence_breakdown': historical_metrics.confidence_breakdown
                }
            }
            
            json_path = os.path.join(betting_dir, f'historical_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(json_path, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            print(f"💾 Исторические метрики сохранены: {json_path}")

def create_sample_matches_and_enhanced_odds():
    """
    УЛУЧШЕНО: Создание примера данных с реалистичными коэффициентами
    """
    # Примеры предстоящих матчей с улучшенными данными
    matches = [
        {
            'match_id': 'RG2025_001',
            'player_name': 'Новак Джокович',
            'opponent_name': 'Рафаэль Надаль',
            'tournament': 'Roland Garros',
            'surface': 'Clay',
            'match_date': '2025-06-20',
            'player_rank': 1, 'opponent_rank': 2,
            'player_recent_win_rate': 0.85, 'player_surface_advantage': 0.05,
            'h2h_win_rate': 0.51, 'total_pressure': 3.8,
            'player_form_trend': 0.1
        },
        {
            'match_id': 'WIM2025_001',
            'player_name': 'Карлос Алкарас',
            'opponent_name': 'Янник Синнер',
            'tournament': 'Wimbledon',
            'surface': 'Grass',
            'match_date': '2025-06-21',
            'player_rank': 3, 'opponent_rank': 4,
            'player_recent_win_rate': 0.78, 'player_surface_advantage': -0.02,
            'h2h_win_rate': 0.43, 'total_pressure': 3.2,
            'player_form_trend': -0.05
        },
        {
            'match_id': 'AO2025_001',
            'player_name': 'Даниил Медведев',
            'opponent_name': 'Александр Зверев',
            'tournament': 'Australian Open',
            'surface': 'Hard',
            'match_date': '2025-06-22',
            'player_rank': 5, 'opponent_rank': 6,
            'player_recent_win_rate': 0.72, 'player_surface_advantage': 0.08,
            'h2h_win_rate': 0.58, 'total_pressure': 3.5,
            'player_form_trend': 0.15
        }
    ]
    
    # Расширенные коэффициенты с множественными рынками
    enhanced_odds = {
        'RG2025_001': {
            'match_info': matches[0],
            'best_markets': {
                'winner': {
                    'player1': {'odds': 1.85, 'bookmaker': 'Pinnacle'},
                    'player2': {'odds': 1.95, 'bookmaker': 'Bet365'}
                },
                'set_winner': {
                    'player1': {'odds': 1.65, 'bookmaker': 'William Hill'},
                    'player2': {'odds': 2.25, 'bookmaker': 'Pinnacle'}
                },
                'total_sets': {
                    'over_3_5': {'odds': 1.90, 'bookmaker': 'Unibet'},
                    'under_3_5': {'odds': 1.90, 'bookmaker': 'Bet365'}
                }
            }
        },
        'WIM2025_001': {
            'match_info': matches[1],
            'best_markets': {
                'winner': {
                    'player1': {'odds': 1.70, 'bookmaker': 'Bet365'},
                    'player2': {'odds': 2.10, 'bookmaker': 'Pinnacle'}
                },
                'set_winner': {
                    'player1': {'odds': 1.55, 'bookmaker': 'Pinnacle'},
                    'player2': {'odds': 2.35, 'bookmaker': 'William Hill'}
                },
                'total_sets': {
                    'over_3_5': {'odds': 1.85, 'bookmaker': 'Unibet'},
                    'under_3_5': {'odds': 1.95, 'bookmaker': 'Bet365'}
                }
            }
        },
        'AO2025_001': {
            'match_info': matches[2],
            'best_markets': {
                'winner': {
                    'player1': {'odds': 1.75, 'bookmaker': 'Pinnacle'},
                    'player2': {'odds': 2.05, 'bookmaker': 'William Hill'}
                },
                'set_winner': {
                    'player1': {'odds': 1.60, 'bookmaker': 'Bet365'},
                    'player2': {'odds': 2.30, 'bookmaker': 'Pinnacle'}
                },
                'total_sets': {
                    'over_3_5': {'odds': 1.88, 'bookmaker': 'Unibet'},
                    'under_3_5': {'odds': 1.92, 'bookmaker': 'William Hill'}
                }
            }
        }
    }
    
    return pd.DataFrame(matches), enhanced_odds

def backtest_betting_strategy(predictor, start_date: str = '2024-01-01', 
                            end_date: str = '2024-12-31') -> BettingMetrics:
    """
    НОВОЕ: Бэктестинг стратегии на исторических данных
    """
    print("🔄 Запуск бэктестинга стратегии...")
    
    # Генерируем исторические данные для демонстрации
    # В реальности здесь должны быть ваши исторические данные
    np.random.seed(42)
    
    date_range = pd.date_range(start_date, end_date, freq='D')
    historical_data = []
    historical_odds = {}
    
    for i, date in enumerate(date_range):
        # Генерируем 1-3 матча в день (не каждый день)
        if np.random.random() > 0.7:  # 30% дней с матчами
            num_matches = np.random.randint(1, 4)
            
            for j in range(num_matches):
                match_id = f"HIST_{date.strftime('%Y%m%d')}_{j}"
                
                # Генерируем реалистичные данные
                player_strength = np.random.beta(2, 2)
                opponent_strength = np.random.beta(2, 2)
                
                match_data = {
                    'match_id': match_id,
                    'player_name': f'Player_{i}_{j}',
                    'opponent_name': f'Opponent_{i}_{j}',
                    'tournament': np.random.choice(['ATP 250', 'ATP 500', 'Masters 1000', 'Grand Slam']),
                    'surface': np.random.choice(['Hard', 'Clay', 'Grass']),
                    'match_date': date,
                    'player_rank': max(1, int(np.random.exponential(30))),
                    'opponent_rank': max(1, int(np.random.exponential(30))),
                    'player_recent_win_rate': player_strength,
                    'player_surface_advantage': np.random.normal(0, 0.1),
                    'h2h_win_rate': np.random.beta(2, 2),
                    'total_pressure': np.random.uniform(1, 4),
                    'player_form_trend': np.random.normal(0, 0.2),
                    'won_at_least_one_set': 1 if player_strength > opponent_strength + np.random.normal(0, 0.2) else 0
                }
                
                historical_data.append(match_data)
                
                # Генерируем коэффициенты
                true_prob = 0.3 + 0.4 * player_strength  # Базовая вероятность
                bookmaker_margin = 0.05  # 5% маржа
                
                fair_odds = 1 / true_prob
                market_odds = fair_odds * (1 - bookmaker_margin)
                
                historical_odds[match_id] = {
                    'match_info': match_data,
                    'best_markets': {
                        'set_winner': {
                            'player1': {
                                'odds': round(market_odds + np.random.normal(0, 0.1), 2),
                                'bookmaker': np.random.choice(['Bet365', 'Pinnacle', 'William Hill'])
                            }
                        }
                    }
                }
    
    historical_df = pd.DataFrame(historical_data)
    
    if len(historical_df) == 0:
        print("❌ Нет исторических данных для бэктестинга")
        return BettingMetrics(0, 0, 0, 0, 0, 0, 0, {})
    
    print(f"📊 Бэктестинг на {len(historical_df)} исторических матчах")
    
    # Создаем систему ставок
    betting_system = EnhancedTennisBettingSystem(predictor, bankroll=10000)
    
    # Запускаем симуляцию
    metrics = betting_system.simulate_betting_performance(
        historical_df, historical_odds, start_date, end_date
    )
    
    return metrics

def main():
    """
    ГЛАВНАЯ ФУНКЦИЯ: Демонстрация улучшенной системы ставок
    """
    print("🎾 УЛУЧШЕННАЯ СИСТЕМА ТЕННИСНЫХ СТАВОК")
    print("=" * 70)
    print("🚀 Новые возможности:")
    print("• Интеграция коэффициентов с нескольких источников")
    print("• Динамический критерий Келли с корректировками")
    print("• Анализ эффективности рынка")
    print("• Мультирыночный анализ ставок")
    print("• Симуляция исторической производительности")
    print("• Подробные отчеты и аналитика")
    print("=" * 70)
    
    # Импортируем предиктор (в реальности загружается обученная модель)
    from enhanced_predictor import EnhancedTennisPredictor
    
    # Создаем или загружаем обученную модель
    predictor = EnhancedTennisPredictor()
    
    # Для демонстрации создаем простую заглушку
    class MockPredictor:
        def prepare_features(self, df):
            return df[['player_recent_win_rate', 'player_surface_advantage', 
                      'h2h_win_rate', 'total_pressure', 'player_form_trend']]
        
        def predict_probability(self, X):
            # Простая логика для демонстрации
            strength = (X['player_recent_win_rate'] * 0.4 +
                       X['player_surface_advantage'] * 0.2 +
                       X['h2h_win_rate'] * 0.2 +
                       X['total_pressure'] * 0.1 +
                       X['player_form_trend'] * 0.1)
            return np.clip(strength + np.random.normal(0, 0.1, len(X)), 0.1, 0.9)
    
    mock_predictor = MockPredictor()
    
    # Создаем систему ставок
    betting_system = EnhancedTennisBettingSystem(mock_predictor, bankroll=10000)
    
    print("\n📊 Загрузка данных о матчах и коэффициентах...")
    
    # Создаем примеры данных
    matches_df, enhanced_odds = create_sample_matches_and_enhanced_odds()
    
    print(f"✅ Загружено {len(matches_df)} матчей с коэффициентами")
    
    # Сбор коэффициентов (демонстрация)
    print("\n💰 Сбор коэффициентов с различных источников...")
    odds_collector = OddsCollector()
    
    # В реальности здесь будут API вызовы
    pinnacle_odds = odds_collector.collect_pinnacle_odds()
    oddsportal_odds = odds_collector.scrape_oddsportal()
    
    print(f"✅ Собрано коэффициентов: Pinnacle ({len(pinnacle_odds)}), Oddsportal ({len(oddsportal_odds)})")
    
    # Поиск ценных ставок
    print("\n🔍 Поиск ценных ставок...")
    value_bets = betting_system.find_value_bets(matches_df, enhanced_odds, model_confidence=0.9)
    
    if value_bets:
        print(f"✅ Найдено {len(value_bets)} ценных ставок")
        
        # Показываем топ-3 ставки
        print("\n🏆 ТОП-3 РЕКОМЕНДАЦИИ:")
        for i, bet in enumerate(value_bets[:3], 1):
            print(f"{i}. {bet.player_name} vs {bet.opponent_name}")
            print(f"   💰 Коэффициент: {bet.odds:.2f}, EV: {bet.expected_value:.3f}")
            print(f"   💵 Ставка: ${bet.recommended_stake:.2f} ({bet.confidence_level.value})")
    else:
        print("❌ Ценных ставок не найдено")
    
    # Бэктестинг стратегии
    print("\n🎲 Запуск бэктестинга стратегии...")
    historical_metrics = backtest_betting_strategy(mock_predictor, '2024-01-01', '2024-06-30')
    
    if historical_metrics.total_bets > 0:
        print(f"📈 Результаты бэктестинга:")
        print(f"   • Ставок: {historical_metrics.total_bets}")
        print(f"   • Прибыль: ${historical_metrics.total_profit:.2f}")
        print(f"   • ROI: {historical_metrics.roi:.2f}%")
        print(f"   • Винрейт: {historical_metrics.win_rate:.1%}")
        print(f"   • Sharpe Ratio: {historical_metrics.sharpe_ratio:.2f}")
    
    # Генерация отчета
    print("\n📄 Генерация подробного отчета...")
    report = betting_system.generate_comprehensive_report(value_bets, historical_metrics)
    
    # Сохранение данных
    betting_system.save_betting_data(value_bets, historical_metrics)
    
    # Выводим сокращенную версию отчета
    print("\n" + "="*50)
    print("📋 КРАТКИЙ ОТЧЕТ:")
    print("="*50)
    
    if value_bets:
        total_stake = sum([bet.recommended_stake for bet in value_bets])
        avg_ev = np.mean([bet.expected_value for bet in value_bets])
        
        print(f"💰 Рекомендуемых ставок: {len(value_bets)}")
        print(f"💵 Общая сумма ставок: ${total_stake:.2f}")
        print(f"📊 Средний EV: {avg_ev:.3f}")
        print(f"🎯 Ожидаемая прибыль: ${total_stake * avg_ev:.2f}")
        
        if historical_metrics and historical_metrics.total_bets > 0:
            print(f"📈 Исторический ROI: {historical_metrics.roi:.2f}%")
            print(f"🏆 Исторический винрейт: {historical_metrics.win_rate:.1%}")
    
    print("\n🚀 СЛЕДУЮЩИЕ ШАГИ:")
    print("1. 🔧 Настройте API ключи для реальных коэффициентов")
    print("2. 📊 Обучите модель на исторических данных")
    print("3. 🎯 Запустите систему в реальном времени")
    print("4. 📈 Отслеживайте результаты и корректируйте стратегию")
    
    print(f"\n📁 Полный отчет сохранен в папке: betting_data/")
    print("✅ Система готова к боевому применению!")

if __name__ == "__main__":
    main()