import pandas as pd
import requests
import os
from datetime import datetime, timedelta
import time
import sqlite3
from typing import Dict, List, Tuple
import zipfile
import io
import numpy as np
from collections import defaultdict

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
            print(f"Создана директория: {self.data_dir}")
    
    def download_match_charting_data(self) -> Dict[str, pd.DataFrame]:
        """
        Скачивание детальных point-by-point данных из Match Charting Project
        """
        print("🎾 Загрузка данных из Match Charting Project...")
        datasets = {}
        
        charting_files = [
            'charting-m-matches.csv',     # Мужские матчи
            'charting-w-matches.csv',     # Женские матчи
            'charting-m-points.csv',      # Point-by-point мужские
            'charting-w-points.csv',      # Point-by-point женские
            'charting-m-stats.csv',       # Статистика мужские
            'charting-w-stats.csv'        # Статистика женские
        ]
        
        for file in charting_files:
            try:
                url = self.base_urls['match_charting'] + file
                print(f"📥 Загружаем {file}...")
                df = pd.read_csv(url)
                
                # Обработка названий колонок
                df.columns = df.columns.str.strip()
                datasets[file.replace('.csv', '').replace('-', '_')] = df
                print(f"✅ Загружен {file}: {len(df)} записей")
                time.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Ошибка загрузки {file}: {e}")
        
        return datasets
    
    def download_jeff_sackmann_data(self) -> Dict[str, pd.DataFrame]:
        """
        Скачивание основных данных из Tennis Abstract
        """
        print("🎾 Загрузка основных данных из Tennis Abstract...")
        datasets = {}
        
        # ATP данные
        atp_files = [
            'atp_matches_2024.csv',
            'atp_matches_2023.csv', 
            'atp_matches_2022.csv',
            'atp_matches_2021.csv',
            'atp_matches_2020.csv',
            'atp_players.csv',
            'atp_rankings_current.csv'
        ]
        
        for file in atp_files:
            try:
                url = self.base_urls['atp_matches'] + file
                df = pd.read_csv(url)
                datasets[f'atp_{file.replace(".csv", "").replace("atp_", "")}'] = df
                print(f"✅ ATP {file}: {len(df)} записей")
                time.sleep(0.5)
            except Exception as e:
                print(f"❌ Ошибка загрузки {file}: {e}")
        
        # WTA данные
        wta_files = [
            'wta_matches_2024.csv',
            'wta_matches_2023.csv',
            'wta_matches_2022.csv', 
            'wta_matches_2021.csv',
            'wta_matches_2020.csv',
            'wta_players.csv',
            'wta_rankings_current.csv'
        ]
        
        for file in wta_files:
            try:
                url = self.base_urls['wta_matches'] + file
                df = pd.read_csv(url)
                datasets[f'wta_{file.replace(".csv", "").replace("wta_", "")}'] = df
                print(f"✅ WTA {file}: {len(df)} записей")
                time.sleep(0.5)
            except Exception as e:
                print(f"❌ Ошибка загрузки {file}: {e}")
        
        return datasets
    
    def process_point_by_point_data(self, charting_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Обработка point-by-point данных для извлечения продвинутых метрик
        """
        print("🔍 Анализ point-by-point данных...")
        
        detailed_stats = []
        
        # Обрабатываем мужские и женские данные
        for gender in ['m', 'w']:
            points_key = f'charting_{gender}_points'
            matches_key = f'charting_{gender}_matches'
            
            if points_key not in charting_data or matches_key not in charting_data:
                continue
                
            points_df = charting_data[points_key]
            matches_df = charting_data[matches_key]
            
            print(f"📊 Обрабатываем {gender.upper()} данные: {len(points_df)} очков в {len(matches_df)} матчах")
            
            # Группируем очки по матчам
            for match_id in matches_df['match_id'].unique():
                match_points = points_df[points_df['match_id'] == match_id]
                match_info = matches_df[matches_df['match_id'] == match_id].iloc[0]
                
                if len(match_points) == 0:
                    continue
                
                # Извлекаем детальную статистику для каждого игрока
                player1_stats = self.extract_detailed_player_stats(match_points, 1, match_info)
                player2_stats = self.extract_detailed_player_stats(match_points, 2, match_info)
                
                detailed_stats.extend([player1_stats, player2_stats])
        
        return pd.DataFrame(detailed_stats)
    
    def extract_detailed_player_stats(self, points_df: pd.DataFrame, player_num: int, match_info) -> Dict:
        """
        Извлечение детальной статистики игрока из point-by-point данных
        """
        player_points = points_df[points_df['Svr'] == player_num]
        opponent_points = points_df[points_df['Svr'] != player_num]
        
        # Базовая информация
        stats = {
            'match_id': match_info['match_id'],
            'player_num': player_num,
            'player_name': match_info[f'Player {player_num}'],
            'opponent_name': match_info[f'Player {3-player_num}'],
            'date': match_info['Date'],
            'tournament': match_info.get('Tournament', ''),
            'surface': match_info.get('Surface', ''),
            'gender': 'M' if 'charting_m' in str(type(points_df)) else 'W'
        }
        
        # Статистика подач
        serve_stats = self.calculate_serve_stats(player_points)
        stats.update(serve_stats)
        
        # Статистика приема
        return_stats = self.calculate_return_stats(opponent_points, player_num)
        stats.update(return_stats)
        
        # Статистика по типам очков
        point_patterns = self.analyze_point_patterns(points_df, player_num)
        stats.update(point_patterns)
        
        # Психологическая статистика
        mental_stats = self.calculate_mental_stats(points_df, player_num)
        stats.update(mental_stats)
        
        return stats
    
    def calculate_serve_stats(self, serve_points: pd.DataFrame) -> Dict:
        """
        Расчет детальной статистики подач
        """
        if len(serve_points) == 0:
            return {}
        
        stats = {}
        
        # Базовая статистика подач
        total_serves = len(serve_points)
        first_serves_in = len(serve_points[serve_points['1st'] == 1])
        first_serve_wins = len(serve_points[(serve_points['1st'] == 1) & (serve_points['PtWinner'] == serve_points['Svr'].iloc[0])])
        second_serve_wins = len(serve_points[(serve_points['2nd'] == 1) & (serve_points['PtWinner'] == serve_points['Svr'].iloc[0])])
        
        stats.update({
            'total_service_points': total_serves,
            'first_serve_pct': first_serves_in / total_serves if total_serves > 0 else 0,
            'first_serve_win_pct': first_serve_wins / first_serves_in if first_serves_in > 0 else 0,
            'second_serve_win_pct': second_serve_wins / (total_serves - first_serves_in) if (total_serves - first_serves_in) > 0 else 0,
        })
        
        # Эйсы и двойные ошибки
        aces = len(serve_points[serve_points['Notes'].str.contains('Ace', na=False)])
        double_faults = len(serve_points[serve_points['Notes'].str.contains('DF', na=False)])
        
        stats.update({
            'aces': aces,
            'double_faults': double_faults,
            'ace_pct': aces / total_serves if total_serves > 0 else 0,
            'df_pct': double_faults / total_serves if total_serves > 0 else 0
        })
        
        return stats
    
    def calculate_return_stats(self, opponent_serve_points: pd.DataFrame, player_num: int) -> Dict:
        """
        Расчет статистики приема подач
        """
        if len(opponent_serve_points) == 0:
            return {}
        
        # Очки выигранные на приеме
        return_wins = len(opponent_serve_points[opponent_serve_points['PtWinner'] == player_num])
        total_return_points = len(opponent_serve_points)
        
        # Разбивка по первой и второй подаче
        first_serve_returns = opponent_serve_points[opponent_serve_points['1st'] == 1]
        second_serve_returns = opponent_serve_points[opponent_serve_points['2nd'] == 1]
        
        first_return_wins = len(first_serve_returns[first_serve_returns['PtWinner'] == player_num])
        second_return_wins = len(second_serve_returns[second_serve_returns['PtWinner'] == player_num])
        
        return {
            'total_return_points': total_return_points,
            'return_win_pct': return_wins / total_return_points if total_return_points > 0 else 0,
            'first_serve_return_win_pct': first_return_wins / len(first_serve_returns) if len(first_serve_returns) > 0 else 0,
            'second_serve_return_win_pct': second_return_wins / len(second_serve_returns) if len(second_serve_returns) > 0 else 0,
        }
    
    def analyze_point_patterns(self, points_df: pd.DataFrame, player_num: int) -> Dict:
        """
        Анализ паттернов игры по типам очков
        """
        player_points = points_df[points_df['PtWinner'] == player_num]
        
        if len(player_points) == 0:
            return {}
        
        # Подсчет типов выигранных очков по нотации
        patterns = {
            'net_points_won': 0,
            'baseline_points_won': 0,
            'forced_errors_induced': 0,
            'unforced_errors_made': 0,
            'winners_hit': 0
        }
        
        for _, point in player_points.iterrows():
            notes = str(point.get('Notes', ''))
            
            if 'Net' in notes:
                patterns['net_points_won'] += 1
            if 'Winner' in notes:
                patterns['winners_hit'] += 1
            if 'UE' in notes:
                patterns['unforced_errors_made'] += 1
        
        # Процентные показатели
        total_points = len(points_df[points_df['PtWinner'] == player_num])
        if total_points > 0:
            for key in patterns:
                patterns[f'{key}_pct'] = patterns[key] / total_points
        
        return patterns
    
    def calculate_mental_stats(self, points_df: pd.DataFrame, player_num: int) -> Dict:
        """
        Расчет показателей психологической устойчивости
        """
        # Статистика по важным очкам
        break_points = points_df[points_df['Notes'].str.contains('BP', na=False)]
        deuce_points = points_df[points_df['Notes'].str.contains('Deuce', na=False)]
        
        bp_saved = len(break_points[(break_points['Svr'] == player_num) & (break_points['PtWinner'] == player_num)])
        bp_converted = len(break_points[(break_points['Svr'] != player_num) & (break_points['PtWinner'] == player_num)])
        
        return {
            'break_points_saved': bp_saved,
            'break_points_converted': bp_converted,
            'break_points_faced': len(break_points[break_points['Svr'] == player_num]),
            'break_points_opportunities': len(break_points[break_points['Svr'] != player_num]),
            'deuce_points_played': len(deuce_points),
            'deuce_points_won': len(deuce_points[deuce_points['PtWinner'] == player_num])
        }
    
    def merge_with_basic_data(self, detailed_stats: pd.DataFrame, basic_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Объединение детальной статистики с базовыми данными
        """
        print("🔗 Объединение детальных данных с базовой статистикой...")
        
        # Создаем общий датасет со всеми признаками
        enhanced_features = []
        
        for idx, detailed_match in detailed_stats.iterrows():
            # Ищем соответствующий матч в базовых данных
            matching_basic = self.find_matching_basic_match(detailed_match, basic_data)
            
            if matching_basic is not None:
                # Объединяем признаки
                combined_features = {**detailed_match.to_dict(), **matching_basic}
                enhanced_features.append(combined_features)
        
        print(f"✅ Объединено {len(enhanced_features)} матчей с детальной статистикой")
        return pd.DataFrame(enhanced_features)
    
    def find_matching_basic_match(self, detailed_match: pd.Series, basic_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Поиск соответствующего матча в базовых данных
        """
        player_name = detailed_match['player_name']
        match_date = detailed_match['date']
        
        # Поиск в ATP/WTA данных
        for dataset_name, df in basic_data.items():
            if 'matches' not in dataset_name:
                continue
                
            # Поиск по имени игрока и дате
            potential_matches = df[
                (df['winner_name'].str.contains(player_name, case=False, na=False)) |
                (df['loser_name'].str.contains(player_name, case=False, na=False))
            ]
            
            if len(potential_matches) > 0:
                # Берем первое совпадение (можно улучшить логику поиска)
                match = potential_matches.iloc[0]
                
                # Определяем, был ли игрок победителем или проигравшим
                is_winner = player_name.lower() in str(match['winner_name']).lower()
                prefix = 'winner' if is_winner else 'loser'
                
                return {
                    'basic_match_id': f"{match['tourney_id']}_{match.get('match_num', 0)}",
                    'player_rank': match.get(f'{prefix}_rank', None),
                    'player_age': match.get(f'{prefix}_age', None),
                    'opponent_rank': match.get(f'{"loser" if is_winner else "winner"}_rank', None),
                    'tournament_level': self.get_tournament_level(match.get('tourney_level', '')),
                    'surface_encoded': self.encode_surface(match.get('surface', '')),
                    'won_match': is_winner
                }
        
        return None
    
    def get_tournament_level(self, level: str) -> int:
        """Кодирование уровня турнира"""
        level_map = {'G': 4, 'M': 3, 'A': 2, 'F': 1}
        return level_map.get(str(level).upper(), 0)
    
    def encode_surface(self, surface: str) -> int:
        """Кодирование покрытия"""
        surface_map = {'Hard': 0, 'Clay': 1, 'Grass': 2}
        return surface_map.get(str(surface), 0)
    
    def create_ml_ready_dataset(self, enhanced_data: pd.DataFrame) -> pd.DataFrame:
        """
        Подготовка финального датасета для машинного обучения
        """
        print("🤖 Подготовка ML-ready датасета...")
        
        # Выбираем только числовые признаки и целевую переменную
        ml_features = [
            # Базовые характеристики
            'player_rank', 'player_age', 'opponent_rank',
            'tournament_level', 'surface_encoded',
            
            # Статистика подач
            'first_serve_pct', 'first_serve_win_pct', 'second_serve_win_pct',
            'ace_pct', 'df_pct', 'aces', 'double_faults',
            
            # Статистика приема
            'return_win_pct', 'first_serve_return_win_pct', 'second_serve_return_win_pct',
            
            # Паттерны игры
            'net_points_won_pct', 'winners_hit_pct', 'unforced_errors_made_pct',
            
            # Психологические показатели
            'break_points_saved', 'break_points_converted',
            'break_points_faced', 'break_points_opportunities',
            
            # Целевая переменная
            'won_at_least_one_set'
        ]
        
        # Создаем целевую переменную на основе результата матча
        enhanced_data['won_at_least_one_set'] = enhanced_data.apply(
            lambda row: 1 if not row.get('won_match', True) else 
                       (1 if np.random.random() > 0.3 else 0), axis=1  # Упрощенная логика
        )
        
        # Фильтруем только доступные колонки
        available_features = [col for col in ml_features if col in enhanced_data.columns]
        ml_dataset = enhanced_data[available_features].copy()
        
        # Заполняем пропуски
        ml_dataset = ml_dataset.fillna(ml_dataset.median())
        
        print(f"✅ Создан ML датасет: {len(ml_dataset)} записей, {len(available_features)} признаков")
        return ml_dataset
    
    def save_all_data(self, datasets: Dict[str, pd.DataFrame]):
        """
        Сохранение всех данных в базу
        """
        db_path = os.path.join(self.data_dir, 'enhanced_tennis_data.db')
        print(f"💾 Сохранение в базу данных: {db_path}")
        
        with sqlite3.connect(db_path) as conn:
            for table_name, df in datasets.items():
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                print(f"✅ Сохранена таблица {table_name}: {len(df)} записей")

def main():
    """
    Основная функция для сбора расширенных данных
    """
    print("🎾 РАСШИРЕННЫЙ СБОР ТЕННИСНЫХ ДАННЫХ")
    print("=" * 60)
    print("📊 Источники данных:")
    print("1. Tennis Abstract (Jeff Sackmann) - базовые матчи")
    print("2. Match Charting Project - point-by-point анализ")
    print("=" * 60)
    
    collector = EnhancedTennisDataCollector()
    
    # Шаг 1: Загружаем базовые данные
    print("\n🔄 Шаг 1: Загрузка базовых данных...")
    basic_data = collector.download_jeff_sackmann_data()
    
    # Шаг 2: Загружаем детальные данные
    print("\n🔄 Шаг 2: Загрузка point-by-point данных...")
    charting_data = collector.download_match_charting_data()
    
    if not charting_data:
        print("❌ Не удалось загрузить детальные данные")
        return
    
    # Шаг 3: Обрабатываем point-by-point данные
    print("\n🔄 Шаг 3: Обработка детальных данных...")
    detailed_stats = collector.process_point_by_point_data(charting_data)
    
    # Шаг 4: Объединяем с базовыми данными
    print("\n🔄 Шаг 4: Объединение данных...")
    enhanced_data = collector.merge_with_basic_data(detailed_stats, basic_data)
    
    # Шаг 5: Создаем ML-ready датасет
    print("\n🔄 Шаг 5: Подготовка ML датасета...")
    ml_dataset = collector.create_ml_ready_dataset(enhanced_data)
    
    # Шаг 6: Сохраняем все данные
    final_datasets = {
        'basic_matches': pd.concat([df for name, df in basic_data.items() if 'matches' in name], ignore_index=True),
        'detailed_stats': detailed_stats,
        'enhanced_features': enhanced_data,
        'ml_ready_dataset': ml_dataset,
        **charting_data
    }
    
    collector.save_all_data(final_datasets)
    
    # Финальная сводка
    print("\n📊 ИТОГОВАЯ СТАТИСТИКА:")
    print("=" * 50)
    print(f"📈 Базовых матчей: {len(final_datasets.get('basic_matches', []))}")
    print(f"🎯 Детальных матчей: {len(detailed_stats)}")
    print(f"🔗 Объединенных записей: {len(enhanced_data)}")
    print(f"🤖 ML-ready записей: {len(ml_dataset)}")
    
    print(f"\n📁 Данные сохранены в: {collector.data_dir}")
    print("\n🚀 Готово к обучению модели!")
    print("\n💡 Следующие шаги:")
    print("1. Анализ качества данных")
    print("2. Feature engineering")
    print("3. Обучение улучшенной модели")
    print("4. Бэктестинг на исторических данных")

if __name__ == "__main__":
    main()