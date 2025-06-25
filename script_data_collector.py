import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
import time
import sqlite3
from typing import Dict, List, Tuple, Optional
import re
import warnings
warnings.filterwarnings('ignore')

class EnhancedTennisDataCollector:
    def __init__(self, data_dir="tennis_data_enhanced"):
        self.data_dir = data_dir
        self.base_urls = {
            'atp_matches': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/',
            'wta_matches': 'https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/',
            'match_charting': 'https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master/'
        }
        self.ensure_data_directory()
        
    def ensure_data_directory(self):
        """Создание директории для данных"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"✅ Создана директория: {self.data_dir}")

    def parse_set_score(self, score_string: str) -> Dict:
        """
        ИСПРАВЛЕНО: Правильный парсинг счета сетов
        """
        if pd.isna(score_string) or not score_string:
            return {'player_sets_won': None, 'opponent_sets_won': None, 'total_sets': None}
        
        # Очищаем счет от лишних символов
        clean_score = re.sub(r'[^\d\-\s\(\)]', '', str(score_string))
        
        # Разбиваем на сеты
        sets = clean_score.split()
        player_sets = 0
        opponent_sets = 0
        valid_sets = 0
        
        for set_score in sets:
            if '-' in set_score:
                # Убираем тайбрейки в скобках
                set_score = re.sub(r'\([^)]*\)', '', set_score)
                scores = set_score.split('-')
                
                if len(scores) == 2:
                    try:
                        p_score = int(scores[0])
                        o_score = int(scores[1])
                        
                        # Определяем победителя сета
                        if p_score > o_score:
                            player_sets += 1
                        else:
                            opponent_sets += 1
                        valid_sets += 1
                    except ValueError:
                        continue
        
        return {
            'player_sets_won': player_sets,
            'opponent_sets_won': opponent_sets,
            'total_sets': valid_sets,
            'won_at_least_one_set': 1 if player_sets >= 1 else 0
        }

    def calculate_recent_form(self, player_matches: pd.DataFrame, reference_date: str, days_back: int = 90) -> Dict:
        """
        НОВОЕ: Расчет формы игрока за последний период
        """
        ref_date = pd.to_datetime(reference_date)
        start_date = ref_date - timedelta(days=days_back)
        
        # Фильтруем матчи за период
        recent_matches = player_matches[
            (pd.to_datetime(player_matches['tourney_date']) >= start_date) &
            (pd.to_datetime(player_matches['tourney_date']) < ref_date)
        ].copy()
        
        if len(recent_matches) == 0:
            return {
                'recent_matches_count': 0,
                'recent_win_rate': 0.5,
                'recent_sets_win_rate': 0.5,
                'form_trend': 0,
                'days_since_last_match': 365
            }
        
        # Сортируем по дате
        recent_matches = recent_matches.sort_values('tourney_date')
        
        # Рассчитываем веса по давности (более свежие матчи важнее)
        days_ago = (ref_date - pd.to_datetime(recent_matches['tourney_date'])).dt.days
        weights = np.exp(-0.05 * days_ago)  # Экспоненциальное затухание
        
        # Результаты матчей (1 - победа, 0 - поражение)
        wins = recent_matches['won'].values
        
        # Взвешенная форма
        if len(wins) > 0:
            weighted_form = np.average(wins, weights=weights)
            
            # Тренд формы (последние 5 матчей vs предыдущие)
            if len(wins) >= 6:
                recent_5 = wins[-5:].mean()
                previous_5 = wins[-10:-5].mean() if len(wins) >= 10 else wins[:-5].mean()
                form_trend = recent_5 - previous_5
            else:
                form_trend = 0
        else:
            weighted_form = 0.5
            form_trend = 0
        
        # Статистика по сетам
        set_results = []
        for _, match in recent_matches.iterrows():
            score_data = self.parse_set_score(match.get('score', ''))
            if score_data['player_sets_won'] is not None:
                total_sets = score_data['player_sets_won'] + score_data['opponent_sets_won']
                if total_sets > 0:
                    set_results.append(score_data['player_sets_won'] / total_sets)
        
        sets_win_rate = np.mean(set_results) if set_results else 0.5
        
        # Дни с последнего матча
        days_since_last = (ref_date - pd.to_datetime(recent_matches['tourney_date'].iloc[-1])).days
        
        return {
            'recent_matches_count': len(recent_matches),
            'recent_win_rate': weighted_form,
            'recent_sets_win_rate': sets_win_rate,
            'form_trend': form_trend,
            'days_since_last_match': min(days_since_last, 365)
        }

    def calculate_surface_advantage(self, player_matches: pd.DataFrame, target_surface: str) -> Dict:
        """
        НОВОЕ: Расчет преимущества игрока на покрытии
        """
        # Статистика на целевом покрытии
        surface_matches = player_matches[player_matches['surface'] == target_surface]
        other_matches = player_matches[player_matches['surface'] != target_surface]
        
        # Винрейт на покрытии
        surface_winrate = surface_matches['won'].mean() if len(surface_matches) > 0 else 0.5
        other_winrate = other_matches['won'].mean() if len(other_matches) > 0 else 0.5
        
        # Относительное преимущество
        surface_advantage = surface_winrate - other_winrate
        
        # Статистика по сетам на покрытии
        surface_sets_data = []
        for _, match in surface_matches.iterrows():
            score_data = self.parse_set_score(match.get('score', ''))
            if score_data['total_sets'] is not None and score_data['total_sets'] > 0:
                surface_sets_data.append(score_data['player_sets_won'] / score_data['total_sets'])
        
        surface_sets_rate = np.mean(surface_sets_data) if surface_sets_data else 0.5
        
        return {
            'surface_matches_count': len(surface_matches),
            'surface_win_rate': surface_winrate,
            'surface_advantage': surface_advantage,
            'surface_sets_rate': surface_sets_rate,
            'surface_experience': min(len(surface_matches) / 50, 1.0)  # Нормализованный опыт
        }

    def calculate_h2h_advanced(self, player_matches: pd.DataFrame, opponent_id: int, reference_date: str) -> Dict:
        """
        НОВОЕ: Продвинутая статистика очных встреч
        """
        # Находим матчи против конкретного соперника
        h2h_matches = player_matches[
            (player_matches['opponent_id'] == opponent_id) &
            (pd.to_datetime(player_matches['tourney_date']) < pd.to_datetime(reference_date))
        ].copy()
        
        if len(h2h_matches) == 0:
            return {
                'h2h_matches': 0,
                'h2h_win_rate': 0.5,
                'h2h_recent_form': 0.5,
                'h2h_sets_advantage': 0,
                'days_since_last_h2h': 365
            }
        
        # Сортируем по дате
        h2h_matches = h2h_matches.sort_values('tourney_date')
        
        # Общая статистика
        h2h_winrate = h2h_matches['won'].mean()
        
        # Форма в последних встречах (последние 3 матча получают больший вес)
        if len(h2h_matches) >= 3:
            weights = [0.2] * (len(h2h_matches) - 3) + [0.3, 0.3, 0.4]
            h2h_recent_form = np.average(h2h_matches['won'], weights=weights)
        else:
            h2h_recent_form = h2h_winrate
        
        # Статистика по сетам
        h2h_sets_data = []
        for _, match in h2h_matches.iterrows():
            score_data = self.parse_set_score(match.get('score', ''))
            if score_data['total_sets'] is not None and score_data['total_sets'] > 0:
                h2h_sets_data.append(score_data['player_sets_won'] - score_data['opponent_sets_won'])
        
        h2h_sets_advantage = np.mean(h2h_sets_data) if h2h_sets_data else 0
        
        # Время с последней встречи
        last_h2h_date = pd.to_datetime(h2h_matches['tourney_date'].iloc[-1])
        days_since_h2h = (pd.to_datetime(reference_date) - last_h2h_date).days
        
        return {
            'h2h_matches': len(h2h_matches),
            'h2h_win_rate': h2h_winrate,
            'h2h_recent_form': h2h_recent_form,
            'h2h_sets_advantage': h2h_sets_advantage,
            'days_since_last_h2h': min(days_since_h2h, 365)
        }

    def calculate_tournament_pressure(self, tournament_name: str, tournament_level: str, round_num: int) -> Dict:
        """
        НОВОЕ: Расчет давления турнира и важности матча
        """
        # Важность турнира
        tournament_weights = {
            'G': 4,  # Grand Slam
            'M': 3,  # Masters 1000
            'A': 2,  # ATP 500
            'D': 1,  # ATP 250
            'F': 1   # Futures
        }
        
        tournament_importance = tournament_weights.get(tournament_level, 1)
        
        # Давление раунда (финалы и полуфиналы = высокое давление)
        round_pressure_map = {
            1: 0.1,  # Первый раунд
            2: 0.2,  # Второй раунд
            3: 0.4,  # Третий раунд
            4: 0.6,  # Четвертьфинал
            5: 0.8,  # Полуфинал
            6: 1.0,  # Финал
            7: 1.0   # Финал (на случай разных систем нумерации)
        }
        
        round_pressure = round_pressure_map.get(round_num, 0.1)
        
        # Особые турниры с повышенным давлением
        high_pressure_tournaments = [
            'Wimbledon', 'Roland Garros', 'US Open', 'Australian Open',
            'ATP Finals', 'WTA Finals'
        ]
        
        is_high_pressure = any(tournament in tournament_name for tournament in high_pressure_tournaments)
        pressure_multiplier = 1.5 if is_high_pressure else 1.0
        
        total_pressure = tournament_importance * round_pressure * pressure_multiplier
        
        return {
            'tournament_importance': tournament_importance,
            'round_pressure': round_pressure,
            'total_pressure': total_pressure,
            'is_high_pressure_tournament': int(is_high_pressure)
        }

    def extract_enhanced_features(self, match_row: pd.Series, all_matches: pd.DataFrame) -> Dict:
        """
        НОВОЕ: Извлечение всех улучшенных признаков для матча
        """
        player_id = match_row.get('winner_id') if match_row.get('result') == 'W' else match_row.get('loser_id')
        opponent_id = match_row.get('loser_id') if match_row.get('result') == 'W' else match_row.get('winner_id')
        match_date = match_row['tourney_date']
        surface = match_row['surface']
        
        # Получаем исторические матчи игрока до этой даты
        player_history = all_matches[
            (all_matches['player_id'] == player_id) &
            (pd.to_datetime(all_matches['tourney_date']) < pd.to_datetime(match_date))
        ].copy()
        
        enhanced_features = {}
        
        # Базовая информация о матче
        enhanced_features.update({
            'match_id': f"{match_row['tourney_id']}_{match_row.get('match_num', 0)}",
            'player_id': player_id,
            'opponent_id': opponent_id,
            'tournament': match_row['tourney_name'],
            'surface': surface,
            'match_date': match_date,
            'round': match_row.get('round', 1)
        })
        
        # Парсим результат матча
        score_data = self.parse_set_score(match_row.get('score', ''))
        enhanced_features.update(score_data)
        
        # Форма игрока
        form_data = self.calculate_recent_form(player_history, match_date)
        enhanced_features.update({f'player_{k}': v for k, v in form_data.items()})
        
        # Преимущество на покрытии
        surface_data = self.calculate_surface_advantage(player_history, surface)
        enhanced_features.update({f'player_{k}': v for k, v in surface_data.items()})
        
        # Очные встречи
        h2h_data = self.calculate_h2h_advanced(player_history, opponent_id, match_date)
        enhanced_features.update(h2h_data)
        
        # Давление турнира
        pressure_data = self.calculate_tournament_pressure(
            match_row['tourney_name'], 
            match_row.get('tourney_level', 'D'), 
            match_row.get('round', 1)
        )
        enhanced_features.update(pressure_data)
        
        # Базовые характеристики игрока и соперника
        enhanced_features.update({
            'player_rank': match_row.get('winner_rank' if match_row.get('result') == 'W' else 'loser_rank'),
            'opponent_rank': match_row.get('loser_rank' if match_row.get('result') == 'W' else 'winner_rank'),
            'player_age': match_row.get('winner_age' if match_row.get('result') == 'W' else 'loser_age'),
            'opponent_age': match_row.get('loser_age' if match_row.get('result') == 'W' else 'winner_age'),
        })
        
        return enhanced_features

    def download_and_process_data(self) -> pd.DataFrame:
        """
        ОБНОВЛЕНО: Основной метод для загрузки и обработки данных
        """
        print("🎾 Загрузка и обработка теннисных данных...")
        print("=" * 60)
        
        all_matches = []
        
        # Загружаем данные за последние годы
        years = ['2024', '2023', '2022', '2021', '2020']
        tours = ['atp', 'wta']
        
        for tour in tours:
            print(f"\n📊 Обрабатываем {tour.upper()} матчи...")
            
            for year in years:
                try:
                    url = f"{self.base_urls[f'{tour}_matches']}{tour}_matches_{year}.csv"
                    print(f"📥 Загружаем {tour}_{year}...")
                    
                    df = pd.read_csv(url)
                    
                    # Подготавливаем данные для обработки
                    df['tour'] = tour.upper()
                    df['year'] = year
                    
                    # Создаем записи для каждого игрока
                    winner_records = df.copy()
                    winner_records['player_id'] = df['winner_id']
                    winner_records['opponent_id'] = df['loser_id'] 
                    winner_records['result'] = 'W'
                    winner_records['won'] = 1
                    
                    loser_records = df.copy()
                    loser_records['player_id'] = df['loser_id']
                    loser_records['opponent_id'] = df['winner_id']
                    loser_records['result'] = 'L'
                    loser_records['won'] = 0
                    
                    year_matches = pd.concat([winner_records, loser_records], ignore_index=True)
                    all_matches.append(year_matches)
                    
                    print(f"✅ {tour}_{year}: {len(df)} матчей → {len(year_matches)} записей")
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"❌ Ошибка загрузки {tour}_{year}: {e}")
        
        if not all_matches:
            raise Exception("Не удалось загрузить данные!")
        
        # Объединяем все данные
        combined_matches = pd.concat(all_matches, ignore_index=True)
        print(f"\n📈 Всего загружено: {len(combined_matches)} записей матчей")
        
        # Обрабатываем каждый матч для извлечения признаков
        print("\n🔍 Извлечение улучшенных признаков...")
        enhanced_data = []
        
        # Обрабатываем по батчам для экономии памяти
        batch_size = 1000
        total_batches = len(combined_matches) // batch_size + 1
        
        for i in range(0, len(combined_matches), batch_size):
            batch = combined_matches.iloc[i:i+batch_size]
            print(f"📊 Обрабатываем батч {i//batch_size + 1}/{total_batches}")
            
            for idx, match_row in batch.iterrows():
                try:
                    enhanced_features = self.extract_enhanced_features(match_row, combined_matches)
                    enhanced_data.append(enhanced_features)
                except Exception as e:
                    print(f"⚠️ Ошибка обработки матча {idx}: {e}")
                    continue
        
        enhanced_df = pd.DataFrame(enhanced_data)
        
        # Очистка данных
        enhanced_df = self.clean_data(enhanced_df)
        
        print(f"\n✅ Создан улучшенный датасет: {len(enhanced_df)} записей")
        print(f"📊 Количество признаков: {len(enhanced_df.columns)}")
        
        return enhanced_df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        НОВОЕ: Очистка и валидация данных
        """
        print("🧹 Очистка данных...")
        
        initial_size = len(df)
        
        # Удаляем записи без целевой переменной
        df = df.dropna(subset=['won_at_least_one_set'])
        
        # Заполняем пропуски в числовых колонках медианными значениями
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Ограничиваем выбросы
        for col in ['player_rank', 'opponent_rank']:
            if col in df.columns:
                df[col] = df[col].clip(upper=1000)  # Максимальный рейтинг 1000
        
        # Удаляем дубликаты
        df = df.drop_duplicates(subset=['match_id', 'player_id'])
        
        final_size = len(df)
        print(f"✅ Очистка завершена: {initial_size} → {final_size} записей")
        
        return df

    def save_enhanced_data(self, df: pd.DataFrame):
        """
        НОВОЕ: Сохранение улучшенных данных
        """
        # Сохраняем в CSV
        csv_path = os.path.join(self.data_dir, 'enhanced_tennis_dataset.csv')
        df.to_csv(csv_path, index=False)
        print(f"💾 Данные сохранены в CSV: {csv_path}")
        
        # Сохраняем в SQLite
        db_path = os.path.join(self.data_dir, 'enhanced_tennis_data.db')
        with sqlite3.connect(db_path) as conn:
            df.to_sql('enhanced_matches', conn, if_exists='replace', index=False)
            print(f"💾 Данные сохранены в БД: {db_path}")
        
        # Сохраняем описание признаков
        feature_description = {
            'Target Variable': 'won_at_least_one_set',
            'Features Count': len(df.columns),
            'Samples Count': len(df),
            'Date Range': f"{df['match_date'].min()} to {df['match_date'].max()}",
            'Key Features': [
                'player_recent_win_rate', 'player_surface_advantage', 
                'h2h_win_rate', 'total_pressure', 'form_trend'
            ]
        }
        
        import json
        desc_path = os.path.join(self.data_dir, 'dataset_description.json')
        with open(desc_path, 'w') as f:
            json.dump(feature_description, f, indent=2, default=str)
        
        print(f"📝 Описание сохранено: {desc_path}")

def main():
    """
    Основная функция для создания улучшенного датасета
    """
    print("🎾 УЛУЧШЕННЫЙ СБОРЩИК ТЕННИСНЫХ ДАННЫХ")
    print("=" * 60)
    print("🚀 Новые возможности:")
    print("• Правильный парсинг счета сетов")
    print("• Расчет формы с учетом давности матчей")
    print("• Преимущество на покрытиях")
    print("• Продвинутая статистика H2H")
    print("• Анализ давления турниров")
    print("=" * 60)
    
    collector = EnhancedTennisDataCollector()
    
    try:
        # Создаем улучшенный датасет
        enhanced_df = collector.download_and_process_data()
        
        # Сохраняем результаты
        collector.save_enhanced_data(enhanced_df)
        
        # Показываем статистику
        print("\n📊 ФИНАЛЬНАЯ СТАТИСТИКА:")
        print("=" * 50)
        print(f"📈 Общее количество записей: {len(enhanced_df):,}")
        print(f"🎯 Матчей с выигранным сетом: {enhanced_df['won_at_least_one_set'].sum():,}")
        print(f"📊 Процент положительных исходов: {enhanced_df['won_at_least_one_set'].mean():.1%}")
        print(f"📅 Период данных: {enhanced_df['match_date'].min()} - {enhanced_df['match_date'].max()}")
        
        # Топ признаки по корреляции с целевой переменной
        numeric_features = enhanced_df.select_dtypes(include=[np.number]).columns
        correlations = enhanced_df[numeric_features].corrwith(enhanced_df['won_at_least_one_set']).abs().sort_values(ascending=False)
        
        print(f"\n🔍 ТОП-10 НАИБОЛЕЕ ВАЖНЫХ ПРИЗНАКОВ:")
        for feature, corr in correlations.head(10).items():
            if feature != 'won_at_least_one_set':
                print(f"• {feature}: {corr:.3f}")
        
        print(f"\n✅ Готово! Данные сохранены в: {collector.data_dir}")
        print("🚀 Следующий шаг: Обучение улучшенной модели")
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()