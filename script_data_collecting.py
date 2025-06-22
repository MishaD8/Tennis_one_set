import pandas as pd
import requests
import os
from datetime import datetime, timedelta
import time
import sqlite3
from typing import Dict, List
import zipfile
import io

class TennisDataCollector:
    def __init__(self, data_dir="tennis_data"):
        self.data_dir = data_dir
        self.base_urls = {
            'atp_matches': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/',
            'wta_matches': 'https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/',
            'atp_rankings': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/',
            'wta_rankings': 'https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/'
        }
        self.ensure_data_directory()
        
    def ensure_data_directory(self):
        """Создание директории для данных"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Создана директория: {self.data_dir}")
    
    def download_jeff_sackmann_data(self) -> Dict[str, pd.DataFrame]:
        """
        Скачивание данных из Tennis Abstract (Jeff Sackmann)
        """
        print("🎾 Загрузка данных из Tennis Abstract...")
        datasets = {}
        
        # ATP данные
        print("📥 Загружаем ATP данные...")
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
                print(f"✅ Загружен {file}: {len(df)} записей")
                time.sleep(0.5)  # Вежливость к серверу
            except Exception as e:
                print(f"❌ Ошибка загрузки {file}: {e}")
        
        # WTA данные
        print("📥 Загружаем WTA данные...")
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
                print(f"✅ Загружен {file}: {len(df)} записей")
                time.sleep(0.5)
            except Exception as e:
                print(f"❌ Ошибка загрузки {file}: {e}")
        
        return datasets
    
    def process_match_data(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Обработка и очистка данных матчей
        """
        print("🔄 Обработка данных матчей...")
        
        all_matches = []
        
        for key, df in raw_data.items():
            if 'matches_' in key:
                # Добавляем мета-информацию
                df['tour'] = 'ATP' if key.startswith('atp') else 'WTA'
                df['year'] = key.split('_')[-1]
                
                # Обработка результатов матчей
                df = self.extract_set_statistics(df)
                
                all_matches.append(df)
        
        combined_df = pd.concat(all_matches, ignore_index=True)
        print(f"✅ Обработано {len(combined_df)} матчей")
        
        return combined_df
    
    def extract_set_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечение статистики по сетам из результатов
        """
        # Целевая переменная: взял ли игрок хотя бы один сет
        def player_won_at_least_one_set(score):
            if pd.isna(score):
                return None
            
            try:
                # Парсим счет типа "6-4 6-3" или "6-4 3-6 6-2"
                sets = score.split(' ')
                player_sets = 0
                
                for set_score in sets:
                    if '-' in set_score:
                        games = set_score.split('-')
                        if len(games) == 2:
                            player_games = int(games[0])
                            opponent_games = int(games[1])
                            
                            # Игрок выиграл сет
                            if player_games > opponent_games:
                                player_sets += 1
                
                return player_sets >= 1
            except:
                return None
        
        # Применяем функцию к результатам
        if 'score' in df.columns:
            df['winner_won_at_least_one_set'] = df['score'].apply(player_won_at_least_one_set)
            # Для проигравшего - инвертируем логику
            df['loser_won_at_least_one_set'] = df.apply(
                lambda row: self.loser_sets_from_score(row['score']), axis=1
            )
        
        return df
    
    def loser_sets_from_score(self, score):
        """Определяем, взял ли проигравший хотя бы один сет"""
        if pd.isna(score):
            return None
        
        try:
            sets = score.split(' ')
            loser_sets = 0
            
            for set_score in sets:
                if '-' in set_score:
                    games = set_score.split('-')
                    if len(games) == 2:
                        winner_games = int(games[0])
                        loser_games = int(games[1])
                        
                        # Проигравший выиграл сет (в реальности этого не может быть в финальном счете,
                        # но мы ищем сеты где у проигравшего больше геймов чем у победителя в промежуточных сетах)
                        if loser_games >= 6 or (loser_games >= winner_games - 2 and winner_games >= 6):
                            loser_sets += 1
            
            return loser_sets >= 1
        except:
            return None
    
    def create_features_dataset(self, matches_df: pd.DataFrame, 
                               players_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Создание датасета с признаками для ML
        """
        print("🔧 Создание признаков для машинного обучения...")
        
        features_list = []
        
        for idx, match in matches_df.iterrows():
            if idx % 1000 == 0:
                print(f"Обработано матчей: {idx}")
            
            # Создаем две записи: для победителя и проигравшего
            winner_features = self.extract_player_features(
                match, 'winner', players_data, matches_df
            )
            loser_features = self.extract_player_features(
                match, 'loser', players_data, matches_df  
            )
            
            if winner_features and loser_features:
                features_list.extend([winner_features, loser_features])
        
        features_df = pd.DataFrame(features_list)
        print(f"✅ Создано {len(features_df)} записей с признаками")
        
        return features_df
    
    def extract_player_features(self, match, player_type, players_data, all_matches):
        """
        Извлечение признаков для конкретного игрока в матче
        """
        try:
            player_id = match[f'{player_type}_id']
            opponent_id = match['winner_id'] if player_type == 'loser' else match['loser_id']
            
            # Базовая информация
            features = {
                'match_id': f"{match['tourney_id']}_{match['match_num']}",
                'player_id': player_id,
                'opponent_id': opponent_id,
                'player_name': match[f'{player_type}_name'],
                'opponent_name': match['winner_name'] if player_type == 'loser' else match['loser_name'],
                'tournament': match['tourney_name'],
                'surface': match['surface'],
                'match_date': match['tourney_date'],
                'round': match['round'],
                'tour': match['tour'],
                
                # Базовые статистики игрока
                'player_age': match[f'{player_type}_age'],
                'player_rank': match[f'{player_type}_rank'],
                'opponent_rank': match['winner_rank'] if player_type == 'loser' else match['loser_rank'],
                
                # Целевая переменная
                'won_at_least_one_set': match[f'{player_type}_won_at_least_one_set']
            }
            
            # Добавляем статистику матча если доступна
            match_stats = ['ace', 'df', '1stIn', '1stWon', '2ndWon', 'SvGms', 'bpSaved', 'bpFaced']
            for stat in match_stats:
                if f'{player_type}_{stat}' in match.index:
                    features[f'player_{stat}'] = match[f'{player_type}_{stat}']
                if f'{"loser" if player_type == "winner" else "winner"}_{stat}' in match.index:
                    features[f'opponent_{stat}'] = match[f'{"loser" if player_type == "winner" else "winner"}_{stat}']
            
            # Рассчитываем производные метрики
            if features.get('player_1stIn') and features.get('player_SvGms'):
                features['first_serve_pct'] = features['player_1stIn'] / features['player_SvGms']
            
            return features
            
        except Exception as e:
            return None
    
    def save_to_database(self, datasets: Dict[str, pd.DataFrame]):
        """
        Сохранение данных в SQLite базу
        """
        db_path = os.path.join(self.data_dir, 'tennis_data.db')
        print(f"💾 Сохранение в базу данных: {db_path}")
        
        with sqlite3.connect(db_path) as conn:
            for table_name, df in datasets.items():
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                print(f"✅ Сохранена таблица {table_name}: {len(df)} записей")
    
    def get_recent_data_summary(self, datasets: Dict[str, pd.DataFrame]):
        """
        Сводка по загруженным данным
        """
        print("\n📊 СВОДКА ПО ДАННЫМ:")
        print("=" * 50)
        
        for name, df in datasets.items():
            print(f"{name}: {len(df)} записей")
            if 'tourney_date' in df.columns:
                min_date = df['tourney_date'].min()
                max_date = df['tourney_date'].max()
                print(f"  Период: {min_date} - {max_date}")
            print()

def main():
    """
    Основная функция для сбора данных
    """
    print("🎾 СБОР ИСТОРИЧЕСКИХ ДАННЫХ ДЛЯ ТЕННИСА")
    print("=" * 50)
    
    collector = TennisDataCollector()
    
    # Шаг 1: Скачиваем сырые данные
    raw_datasets = collector.download_jeff_sackmann_data()
    
    if not raw_datasets:
        print("❌ Не удалось загрузить данные")
        return
    
    # Шаг 2: Обрабатываем данные матчей
    matches_df = collector.process_match_data(raw_datasets)
    
    # Шаг 3: Создаем признаки для ML
    players_data = {k: v for k, v in raw_datasets.items() if 'players' in k}
    features_df = collector.create_features_dataset(matches_df, players_data)
    
    # Шаг 4: Сохраняем все в базу
    final_datasets = {
        'matches': matches_df,
        'features': features_df,
        **raw_datasets
    }
    
    collector.save_to_database(final_datasets)
    collector.get_recent_data_summary(final_datasets)
    
    print("\n✅ Сбор данных завершен!")
    print(f"📁 Данные сохранены в: {collector.data_dir}")
    print("\n💡 Следующие шаги:")
    print("1. Проанализируйте качество данных")
    print("2. Создайте дополнительные признаки")
    print("3. Обучите модель на собранных данных")

if __name__ == "__main__":
    main()