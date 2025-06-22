import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class TennisBettingSystem:
    def __init__(self, model_predictor):
        self.predictor = model_predictor
        self.min_confidence = 0.65  # Минимальная уверенность для ставки
        self.min_odds = 1.5  # Минимальные коэффициенты
        self.max_odds = 10.0  # Максимальные коэффициенты
        
    def calculate_kelly_criterion(self, probability: float, odds: float) -> float:
        """
        Расчет оптимального размера ставки по критерию Келли
        """
        if probability <= 0 or odds <= 1:
            return 0
        
        # Преобразуем десятичные коэффициенты в вероятность букмекера
        bookmaker_prob = 1 / odds
        
        # Критерий Келли: f = (bp - q) / b
        # где b = odds - 1, p = наша вероятность, q = 1 - p
        b = odds - 1
        p = probability
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Ограничиваем размер ставки (консервативный подход)
        return max(0, min(kelly_fraction * 0.25, 0.05))  # Максимум 5% банка
    
    def find_value_bets(self, matches_data: pd.DataFrame, odds_data: pd.DataFrame) -> pd.DataFrame:
        """
        Поиск ценных ставок
        """
        value_bets = []
        
        for idx, match in matches_data.iterrows():
            try:
                # Получаем предсказание модели
                match_features = self.predictor.prepare_features(pd.DataFrame([match]))
                probability = self.predictor.predict_probability(match_features)[0][0]
                
                # Находим коэффициенты для этого матча
                match_odds = odds_data[odds_data['match_id'] == match['match_id']]
                
                if match_odds.empty:
                    continue
                
                # Анализируем коэффициенты на "игрок возьмет хотя бы один сет"
                for _, odds_row in match_odds.iterrows():
                    bookmaker = odds_row['bookmaker']
                    odds_value = odds_row['player_wins_set_odds']
                    
                    if self.min_odds <= odds_value <= self.max_odds:
                        # Рассчитываем ожидаемую ценность
                        implied_prob = 1 / odds_value
                        expected_value = (probability * (odds_value - 1)) - (1 - probability)
                        
                        # Проверяем, есть ли ценность
                        if (probability > implied_prob and 
                            probability >= self.min_confidence and 
                            expected_value > 0):
                            
                            kelly_size = self.calculate_kelly_criterion(probability, odds_value)
                            
                            if kelly_size > 0:
                                value_bets.append({
                                    'match_id': match['match_id'],
                                    'player_name': match['player_name'],
                                    'opponent_name': match['opponent_name'],
                                    'tournament': match['tournament'],
                                    'surface': match['surface'],
                                    'match_date': match['match_date'],
                                    'bookmaker': bookmaker,
                                    'odds': odds_value,
                                    'our_probability': probability,
                                    'implied_probability': implied_prob,
                                    'expected_value': expected_value,
                                    'kelly_fraction': kelly_size,
                                    'confidence_level': self._get_confidence_level(probability),
                                    'value_rating': expected_value / implied_prob  # Рейтинг ценности
                                })
                                
            except Exception as e:
                print(f"Ошибка при анализе матча {match.get('match_id', 'unknown')}: {e}")
                continue
        
        if value_bets:
            df_bets = pd.DataFrame(value_bets)
            # Сортируем по рейтингу ценности
            df_bets = df_bets.sort_values('value_rating', ascending=False)
            return df_bets
        else:
            return pd.DataFrame()
    
    def _get_confidence_level(self, probability: float) -> str:
        """
        Определение уровня уверенности
        """
        if probability >= 0.8:
            return "Очень высокая"
        elif probability >= 0.7:
            return "Высокая"
        elif probability >= 0.6:
            return "Средняя"
        else:
            return "Низкая"
    
    def generate_betting_report(self, value_bets: pd.DataFrame) -> str:
        """
        Генерация отчета по ставкам
        """
        if value_bets.empty:
            return "Ценных ставок не найдено."
        
        report = f"""
🎾 ОТЧЕТ ПО ТЕННИСНЫМ СТАВКАМ
{'='*50}

📊 ОБЩАЯ СТАТИСТИКА:
• Найдено ценных ставок: {len(value_bets)}
• Средняя наша вероятность: {value_bets['our_probability'].mean():.1%}
• Средние коэффициенты: {value_bets['odds'].mean():.2f}
• Среднее ожидаемое значение: {value_bets['expected_value'].mean():.3f}

🏆 ТОП-5 РЕКОМЕНДАЦИЙ:
"""
        
        for idx, bet in value_bets.head(5).iterrows():
            report += f"""
#{idx+1}. {bet['player_name']} vs {bet['opponent_name']}
   📅 {bet['match_date']} | 🏟️ {bet['tournament']} ({bet['surface']})
   💰 Коэффициент: {bet['odds']:.2f} ({bet['bookmaker']})
   🎯 Наша вероятность: {bet['our_probability']:.1%}
   📈 Ожидаемое значение: {bet['expected_value']:.3f}
   💵 Рекомендуемая доля банка: {bet['kelly_fraction']:.1%}
   ⭐ Уверенность: {bet['confidence_level']}
"""
        
        report += f"""
💡 РЕКОМЕНДАЦИИ:
• Используйте консервативный подход к управлению банком
• Не превышайте рекомендуемые размеры ставок
• Отслеживайте результаты для корректировки модели
• Учитывайте изменения линий перед матчем

⚠️ ВАЖНО: Это рекомендации основанные на математической модели.
Ставки всегда связаны с риском. Играйте ответственно!
"""
        return report

def create_sample_matches_and_odds():
    """
    Создание примера данных матчей и коэффициентов
    """
    # Примеры предстоящих матчей
    matches = [
        {
            'match_id': 'M001',
            'player_name': 'Новак Джокович',
            'opponent_name': 'Рафаэль Надаль',
            'tournament': 'Roland Garros',
            'surface': 'Грунт',
            'match_date': '2025-06-20',
            'player_age': 38, 'player_rank': 1, 'player_points': 8350,
            'recent_matches_count': 12, 'wins_last_10': 8, 'losses_last_10': 2,
            'sets_won_ratio': 0.75, 'sets_lost_ratio': 0.25, 'tiebreak_win_ratio': 0.65,
            'sets_won_on_surface': 45, 'sets_lost_on_surface': 18,
            'first_serve_pct': 0.68, 'first_serve_won_pct': 0.73, 
            'second_serve_won_pct': 0.58, 'break_points_saved_pct': 0.67,
            'first_serve_return_won_pct': 0.32, 'second_serve_return_won_pct': 0.52,
            'break_points_converted_pct': 0.41,
            'opponent_age': 39, 'opponent_rank': 2, 'opponent_points': 7980,
            'opponent_recent_form': 0.8, 'h2h_wins': 30, 'h2h_losses': 29,
            'tournament_level': 4, 'surface_type': 1, 'match_round': 4,
            'best_of_sets': 5, 'is_home': 0
        },
        {
            'match_id': 'M002',
            'player_name': 'Карлос Алкарас',
            'opponent_name': 'Янник Синнер',
            'tournament': 'Wimbledon',
            'surface': 'Трава',
            'match_date': '2025-06-21',
            'player_age': 22, 'player_rank': 3, 'player_points': 7200,
            'recent_matches_count': 10, 'wins_last_10': 7, 'losses_last_10': 3,
            'sets_won_ratio': 0.72, 'sets_lost_ratio': 0.28, 'tiebreak_win_ratio': 0.58,
            'sets_won_on_surface': 25, 'sets_lost_on_surface': 12,
            'first_serve_pct': 0.65, 'first_serve_won_pct': 0.71, 
            'second_serve_won_pct': 0.55, 'break_points_saved_pct': 0.63,
            'first_serve_return_won_pct': 0.35, 'second_serve_return_won_pct': 0.48,
            'break_points_converted_pct': 0.38,
            'opponent_age': 23, 'opponent_rank': 4, 'opponent_points': 6800,
            'opponent_recent_form': 0.75, 'h2h_wins': 3, 'h2h_losses': 4,
            'tournament_level': 4, 'surface_type': 2, 'match_round': 3,
            'best_of_sets': 5, 'is_home': 0
        }
    ]
    
    # Коэффициенты от разных букмекеров
    odds = [
        {'match_id': 'M001', 'bookmaker': 'Bet365', 'player_wins_set_odds': 1.85},
        {'match_id': 'M001', 'bookmaker': 'William Hill', 'player_wins_set_odds': 1.90},
        {'match_id': 'M001', 'bookmaker': 'Pinnacle', 'player_wins_set_odds': 1.88},
        {'match_id': 'M002', 'bookmaker': 'Bet365', 'player_wins_set_odds': 1.75},
        {'match_id': 'M002', 'bookmaker': 'William Hill', 'player_wins_set_odds': 1.80},
        {'match_id': 'M002', 'bookmaker': 'Pinnacle', 'player_wins_set_odds': 1.78},
    ]
    
    return pd.DataFrame(matches), pd.DataFrame(odds)

# Функция для анализа исторических результатов
def analyze_historical_performance(predictions: pd.DataFrame, actual_results: pd.DataFrame) -> Dict:
    """
    Анализ исторической эффективности модели
    """
    merged = predictions.merge(actual_results, on='match_id')
    
    # Группируем по уровням уверенности
    confidence_groups = {
        'Очень высокая': merged[merged['confidence_level'] == 'Очень высокая'],
        'Высокая': merged[merged['confidence_level'] == 'Высокая'],
        'Средняя': merged[merged['confidence_level'] == 'Средняя']
    }
    
    results = {}
    for level, group in confidence_groups.items():
        if len(group) > 0:
            accuracy = (group['predicted_outcome'] == group['actual_outcome']).mean()
            profit = group['profit'].sum()
            roi = profit / group['stake'].sum() if group['stake'].sum() > 0 else 0
            
            results[level] = {
                'matches': len(group),
                'accuracy': accuracy,
                'total_profit': profit,
                'roi': roi
            }
    
    return results

# Пример использования системы
if __name__ == "__main__":
    print("🎾 Система поиска выгодных ставок на теннис")
    print("=" * 50)
    
    # Создаем тестовые данные
    matches_df, odds_df = create_sample_matches_and_odds()
    
    print(f"📊 Анализируем {len(matches_df)} матчей с коэффициентами от {len(odds_df)} букмекеров")
    
    # Здесь должна быть инициализация обученной модели
    # betting_system = TennisBettingSystem(trained_predictor)
    # value_bets = betting_system.find_value_bets(matches_df, odds_df)
    # report = betting_system.generate_betting_report(value_bets)
    # print(report)
    
    print("\n💡 Для полноценной работы системы необходимо:")
    print("1. Обучить модель на исторических данных")
    print("2. Настроить парсинг актуальных коэффициентов")
    print("3. Создать базу данных матчей и результатов")
    print("4. Реализовать систему уведомлений о ценных ставках")