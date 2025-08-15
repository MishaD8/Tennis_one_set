#!/usr/bin/env python3
"""
📊 ЗАВЕРШЕННАЯ СИСТЕМА ЛОГИРОВАНИЯ РЕЗУЛЬТАТОВ
Полная система для накопления данных и анализа точности
"""

import os
import json
import csv
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

class CompletePredictionLogger:
    """Полная система логирования прогнозов и результатов"""
    
    def __init__(self, data_dir="prediction_logs"):
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, "predictions.db")
        self.csv_path = os.path.join(data_dir, "predictions.csv")
        self.stats_file = os.path.join(data_dir, "accuracy_stats.json")
        
        # Создаем директорию
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"✅ Создана директория: {data_dir}")
        
        # Инициализируем базу данных
        self._init_database()
        
        # Инициализируем CSV
        self._init_csv()
        
    def _init_database(self):
        """Инициализация SQLite базы данных"""
        try:
                # ВАЖНО: создаем подключение к базе
            self.conn = sqlite3.connect(self.db_path)
            
            self.conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                match_date TEXT NOT NULL,
                player1 TEXT NOT NULL,
                player2 TEXT NOT NULL,
                tournament TEXT,
                surface TEXT,
                round_name TEXT,
                our_probability REAL NOT NULL,
                confidence TEXT,
                ml_system TEXT,
                prediction_type TEXT,
                key_factors TEXT,
                
                -- Букмекерские данные
                bookmaker_odds REAL,
                bookmaker_probability REAL,
                edge REAL,
                recommendation TEXT,
                
                -- Результаты (заполняются позже)
                actual_result TEXT,
                actual_winner TEXT,
                sets_won_p1 INTEGER,
                sets_won_p2 INTEGER,
                match_score TEXT,
                
                -- Анализ результата
                correct_prediction INTEGER,
                profit_loss REAL,
                logged_result INTEGER DEFAULT 0,
                
                -- Метаданные
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Индексы для быстрого поиска
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_match_date ON predictions(match_date)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_players ON predictions(player1, player2)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_logged ON predictions(logged_result)')
            
            # Сохраняем изменения
            self.conn.commit()
            
            print(f"✅ База данных инициализирована: {self.db_path}")
        except Exception as e:
            print(f"❌ Ошибка инициализации БД: {e}")

    # ТАКЖЕ убедитесь что в __init__ вызывается _init_database:
    def __init__(self, data_dir="prediction_logs"):
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, "predictions.db")
        self.csv_path = os.path.join(data_dir, "predictions.csv")
        self.stats_file = os.path.join(data_dir, "accuracy_stats.json")
        
        # Создаем директорию
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"✅ Создана директория: {data_dir}")
        
        # ВАЖНО: Инициализируем базу данных (создает self.conn)
        self._init_database()
        
        # Инициализируем CSV
        self._init_csv()
    
    def _init_csv(self):
        """Инициализация CSV файла"""
        if not os.path.exists(self.csv_path):
            headers = [
                'id', 'timestamp', 'match_date', 'player1', 'player2',
                'tournament', 'surface', 'round_name', 
                'our_probability', 'confidence', 'ml_system', 'prediction_type',
                'bookmaker_odds', 'edge', 'recommendation',
                'actual_result', 'actual_winner', 'sets_won_p1', 'sets_won_p2',
                'correct_prediction', 'profit_loss', 'logged_result'
            ]
            
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            
            print(f"✅ CSV файл создан: {self.csv_path}")
    
    def log_prediction(self, prediction_data):
        """Логирует прогноз на матч в БД и CSV"""
        timestamp = datetime.now().isoformat()
        prediction_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Извлекаем данные
        player1 = prediction_data.get('player1', '')
        player2 = prediction_data.get('player2', '')
        tournament = prediction_data.get('tournament', '')
        surface = prediction_data.get('surface', '')
        round_name = prediction_data.get('round', 'R64')
        
        our_prob = prediction_data.get('our_probability', 0.5)
        confidence = prediction_data.get('confidence', 'Medium')
        ml_system = prediction_data.get('ml_system', 'Unknown')
        prediction_type = prediction_data.get('prediction_type', 'ML_PREDICTION')
        key_factors = json.dumps(prediction_data.get('key_factors', []))
        
        # Букмекерские данные
        bookmaker_odds = prediction_data.get('bookmaker_odds', 2.0)
        bookmaker_prob = 1 / bookmaker_odds if bookmaker_odds else 0.5
        edge = our_prob - bookmaker_prob
        recommendation = 'BET' if edge > 0.05 else 'PASS'
        
        try:
            # ИСПРАВЛЕНО: Используем cursor для получения lastrowid
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO predictions (
                    timestamp, match_date, player1, player2, tournament, surface, round_name,
                    our_probability, confidence, ml_system, prediction_type, key_factors,
                    bookmaker_odds, bookmaker_probability, edge, recommendation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp, prediction_data.get('match_date', datetime.now().date().isoformat()),
                player1, player2, tournament, surface, round_name,
                our_prob, confidence, ml_system, prediction_type, key_factors,
                bookmaker_odds, bookmaker_prob, edge, recommendation
            ))
            
            # ИСПРАВЛЕНО: cursor.lastrowid вместо conn.lastrowid
            row_id = cursor.lastrowid
            self.conn.commit()
            
            # Записываем в CSV
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    row_id, timestamp, prediction_data.get('match_date', datetime.now().date().isoformat()),
                    player1, player2, tournament, surface, round_name,
                    our_prob, confidence, ml_system, prediction_type,
                    bookmaker_odds, edge, recommendation,
                    '', '', '', '', '', '', 0  # Пустые поля для результатов
                ])
            
            print(f"📝 Прогноз залогирован: {player1} vs {player2}")
            print(f"   Наш прогноз: {our_prob:.1%} vs Букмекеры: {bookmaker_prob:.1%}")
            print(f"   Преимущество: {edge:+.1%} → {recommendation}")
            print(f"   ID в БД: {row_id}")
            
            return str(row_id)
            
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Ошибка логирования: {e}")
            return None
    
    def update_result(self, player1: str, player2: str, match_date: str,
                     actual_winner: str, match_score: str = "") -> bool:
        """Обновляет результат матча"""
        try:
            # Парсим счет
            sets_p1, sets_p2 = self._parse_score(match_score)
            
            # Обновляем в базе данных
            with sqlite3.connect(self.db_path) as conn:
                # Находим запись
                cursor = conn.execute('''
                SELECT id, our_probability, bookmaker_odds, recommendation 
                FROM predictions 
                WHERE player1 = ? AND player2 = ? AND match_date = ? AND logged_result = 0
                LIMIT 1
                ''', (player1, player2, match_date))
                
                row = cursor.fetchone()
                if not row:
                    print(f"❌ Матч не найден: {player1} vs {player2} на {match_date}")
                    return False
                
                db_id, our_prob, bookmaker_odds, recommendation = row
                
                # Определяем правильность прогноза
                predicted_winner = player1 if our_prob > 0.5 else player2
                correct = 1 if predicted_winner == actual_winner else 0
                
                # Рассчитываем прибыль/убыток
                if recommendation == 'BET':
                    stake = 100  # Условная ставка $100
                    if actual_winner == player1:  # Выиграли
                        profit = stake * (bookmaker_odds - 1)
                    else:  # Проиграли
                        profit = -stake
                else:
                    profit = 0.0  # Не ставили
                
                # Обновляем запись
                conn.execute('''
                UPDATE predictions SET 
                    actual_result = 'COMPLETED',
                    actual_winner = ?,
                    sets_won_p1 = ?,
                    sets_won_p2 = ?,
                    match_score = ?,
                    correct_prediction = ?,
                    profit_loss = ?,
                    logged_result = 1
                WHERE id = ?
                ''', (actual_winner, sets_p1, sets_p2, match_score, correct, profit, db_id))
            
            print(f"✅ Результат обновлен: {actual_winner} победил")
            print(f"   Прогноз {'ВЕРНЫЙ' if correct else 'НЕВЕРНЫЙ'}")
            if recommendation == 'BET':
                print(f"   Прибыль/убыток: ${profit:.2f}")
            
            # Обновляем статистику
            self._update_accuracy_stats()
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка обновления результата: {e}")
            return False
    
    def _parse_score(self, score: str) -> tuple:
        """Парсит счет матча по сетам"""
        try:
            if not score:
                return (0, 0)
            
            # Удаляем лишние символы и разбиваем на сеты
            clean_score = score.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
            sets = clean_score.split(',') if ',' in clean_score else clean_score.split()
            
            p1_sets = 0
            p2_sets = 0
            
            for set_score in sets:
                set_score = set_score.strip()
                if '-' in set_score:
                    games = set_score.split('-')
                    if len(games) == 2:
                        try:
                            g1, g2 = int(games[0]), int(games[1])
                            if g1 > g2:
                                p1_sets += 1
                            else:
                                p2_sets += 1
                        except ValueError:
                            continue
            
            return (p1_sets, p2_sets)
            
        except Exception:
            return (0, 0)
    
    def _update_accuracy_stats(self):
        """Обновляет статистику точности"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Читаем завершенные матчи
                df = pd.read_sql_query('''
                SELECT * FROM predictions WHERE logged_result = 1
                ''', conn)
            
            if len(df) == 0:
                return
            
            # Рассчитываем общую статистику
            total_predictions = len(df)
            correct_predictions = df['correct_prediction'].sum()
            accuracy = correct_predictions / total_predictions
            
            # Статистика по рекомендациям
            bet_predictions = df[df['recommendation'] == 'BET']
            bet_count = len(bet_predictions)
            
            if bet_count > 0:
                bet_correct = bet_predictions['correct_prediction'].sum()
                bet_accuracy = bet_correct / bet_count
                total_profit = bet_predictions['profit_loss'].sum()
                avg_profit = total_profit / bet_count
                roi = (total_profit / (bet_count * 100)) * 100  # ROI в процентах
            else:
                bet_accuracy = 0
                total_profit = 0
                avg_profit = 0
                roi = 0
            
            # Статистика по уверенности
            confidence_stats = {}
            for conf in ['High', 'Medium', 'Low']:
                conf_df = df[df['confidence'] == conf]
                if len(conf_df) > 0:
                    confidence_stats[conf] = {
                        'count': len(conf_df),
                        'accuracy': conf_df['correct_prediction'].mean(),
                        'avg_probability': conf_df['our_probability'].mean()
                    }
            
            # Статистика по периодам
            df['match_date'] = pd.to_datetime(df['match_date'])
            last_30_days = df[df['match_date'] >= datetime.now() - timedelta(days=30)]
            last_7_days = df[df['match_date'] >= datetime.now() - timedelta(days=7)]
            
            # Сохраняем статистику
            stats = {
                'updated': datetime.now().isoformat(),
                'total_predictions': int(total_predictions),
                'correct_predictions': int(correct_predictions),
                'overall_accuracy': float(accuracy),
                
                'bet_recommendations': int(bet_count),
                'bet_accuracy': float(bet_accuracy),
                'total_profit': float(total_profit),
                'average_profit_per_bet': float(avg_profit),
                'roi_percentage': float(roi),
                
                'confidence_breakdown': confidence_stats,
                
                'recent_performance': {
                    'last_30_days': {
                        'count': len(last_30_days),
                        'accuracy': float(last_30_days['correct_prediction'].mean()) if len(last_30_days) > 0 else 0
                    },
                    'last_7_days': {
                        'count': len(last_7_days),
                        'accuracy': float(last_7_days['correct_prediction'].mean()) if len(last_7_days) > 0 else 0
                    }
                }
            }
            
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
            
            print(f"📊 Статистика обновлена:")
            print(f"   Общая точность: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
            if bet_count > 0:
                print(f"   Точность ставок: {bet_accuracy:.1%} ({bet_count} ставок)")
                print(f"   ROI: {roi:.1f}%")
            
        except Exception as e:
            print(f"⚠️ Ошибка обновления статистики: {e}")
    
    def get_accuracy_report(self) -> Dict:
        """Получает подробный отчет о точности"""
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                
                # Добавляем тренды
                with sqlite3.connect(self.db_path) as conn:
                    df = pd.read_sql_query('''
                    SELECT match_date, correct_prediction, our_probability, confidence
                    FROM predictions WHERE logged_result = 1
                    ORDER BY match_date
                    ''', conn)
                
                if len(df) > 0:
                    df['match_date'] = pd.to_datetime(df['match_date'])
                    
                    # Тренд точности (скользящее среднее по 10 матчам)
                    df['accuracy_trend'] = df['correct_prediction'].rolling(window=10, min_periods=1).mean()
                    
                    # Калибровка (насколько наши проценты соответствуют реальности)
                    prob_bins = pd.cut(df['our_probability'], bins=[0, 0.5, 0.7, 0.9, 1.0], 
                                     labels=['Low', 'Medium', 'High', 'Very High'])
                    calibration = df.groupby(prob_bins)['correct_prediction'].agg(['count', 'mean']).to_dict()
                    
                    stats['trends'] = {
                        'latest_accuracy_trend': float(df['accuracy_trend'].iloc[-1]) if len(df) > 0 else 0,
                        'calibration': calibration
                    }
                
                return stats
            else:
                return {"message": "Нет данных для анализа"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def export_data(self, format='csv') -> str:
        """Экспорт данных в различных форматах"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query('SELECT * FROM predictions', conn)
            
            if format == 'csv':
                export_file = os.path.join(self.data_dir, f'predictions_export_{timestamp}.csv')
                df.to_csv(export_file, index=False)
            elif format == 'excel':
                export_file = os.path.join(self.data_dir, f'predictions_export_{timestamp}.xlsx')
                df.to_excel(export_file, index=False)
            else:
                export_file = os.path.join(self.data_dir, f'predictions_export_{timestamp}.json')
                df.to_json(export_file, orient='records', indent=2)
            
            print(f"📁 Данные экспортированы: {export_file}")
            return export_file
            
        except Exception as e:
            print(f"❌ Ошибка экспорта: {e}")
            return ""
    
    def get_pending_predictions(self) -> List[Dict]:
        """Получает прогнозы ожидающие результата"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                SELECT id, player1, player2, match_date, tournament, our_probability, confidence
                FROM predictions 
                WHERE logged_result = 0 
                ORDER BY match_date DESC
                ''')
                
                pending = []
                for row in cursor.fetchall():
                    pending.append({
                        'id': row[0],
                        'player1': row[1],
                        'player2': row[2],
                        'match_date': row[3],
                        'tournament': row[4],
                        'our_probability': row[5],
                        'confidence': row[6]
                    })
                
                return pending
                
        except Exception as e:
            print(f"❌ Ошибка получения ожидающих прогнозов: {e}")
            return []


# Пример использования и тестирования
if __name__ == "__main__":
    print("📊 ТЕСТИРОВАНИЕ СИСТЕМЫ ЛОГИРОВАНИЯ")
    print("=" * 50)
    
    # Создаем логгер
    logger = CompletePredictionLogger()
    
    # Тестовые данные прогноза
    test_prediction = {
        'player1': 'Flavio Cobolli',
        'player2': 'Novak Djokovic',
        'tournament': 'Wimbledon',
        'surface': 'Grass',
        'round': 'R64',
        'match_date': '2025-07-10',
        'our_probability': 0.25,
        'confidence': 'Medium',
        'ml_system': 'REAL_ML_MODEL',
        'prediction_type': 'UNDERDOG_ANALYSIS',
        'key_factors': ['Grass advantage', 'Underdog potential', 'Recent form'],
        'bookmaker_odds': 4.5
    }
    
    # Логируем прогноз
    print("1️⃣ Логирование тестового прогноза...")
    prediction_id = logger.log_prediction(test_prediction)
    
    # Имитируем результат матча
    print("\n2️⃣ Обновление результата...")
    logger.update_result(
        player1='Flavio Cobolli',
        player2='Novak Djokovic', 
        match_date='2025-07-10',
        actual_winner='Novak Djokovic',
        match_score='6-4, 6-2, 6-3'
    )
    
    # Получаем отчет
    print("\n3️⃣ Получение отчета о точности...")
    report = logger.get_accuracy_report()
    print(f"📊 Отчет: {json.dumps(report, indent=2, default=str)}")
    
    # Проверяем ожидающие прогнозы
    print("\n4️⃣ Ожидающие прогнозы...")
    pending = logger.get_pending_predictions()
    print(f"📋 Ожидающих результата: {len(pending)}")
    
    # Экспорт данных
    print("\n5️⃣ Экспорт данных...")
    export_file = logger.export_data('csv')
    
    print("\n✅ Система логирования готова к использованию!")
    print(f"📁 База данных: {logger.db_path}")
    print(f"📄 CSV файл: {logger.csv_path}")
    print(f"📊 Статистика: {logger.stats_file}")


class PredictionLoggerIntegration:
    """Интеграция логгера с вашей существующей системой"""
    
    def __init__(self):
        self.logger = CompletePredictionLogger()
    
    def log_match_prediction(self, match_result: Dict) -> str:
        """Интегрируется с вашим underdog analyzer"""
        
        # Адаптируем данные из underdog_analysis к формату логгера
        if 'underdog_analysis' in match_result:
            analysis = match_result['underdog_analysis']
            scenario = analysis.get('underdog_scenario', {})
            
            match_data = {
                'player1': match_result.get('player1', '').replace('🎾 ', ''),
                'player2': match_result.get('player2', '').replace('🎾 ', ''),
                'tournament': match_result.get('tournament', '').replace('🏆 ', ''),
                'surface': match_result.get('surface', 'Hard'),
                'round': match_result.get('round', 'R64'),
                'match_date': match_result.get('date', datetime.now().date().isoformat()),
                'our_probability': analysis.get('underdog_probability', 0.5),
                'confidence': analysis.get('confidence', 'Medium'),
                'ml_system': analysis.get('ml_system_used', 'Unknown'),
                'prediction_type': analysis.get('prediction_type', 'UNDERDOG_ANALYSIS'),
                'key_factors': analysis.get('key_factors', []),
                'bookmaker_odds': match_result.get('odds', {}).get('player1', 2.0)
            }
            
            return self.logger.log_prediction(match_data)
        
        return ""
    
    def bulk_log_matches(self, matches_list: List[Dict]) -> List[str]:
        """Логирует несколько матчей сразу"""
        logged_ids = []
        
        for match in matches_list:
            try:
                prediction_id = self.log_match_prediction(match)
                if prediction_id:
                    logged_ids.append(prediction_id)
            except Exception as e:
                print(f"⚠️ Ошибка логирования матча: {e}")
                continue
        
        print(f"📝 Залогировано {len(logged_ids)} из {len(matches_list)} матчей")
        return logged_ids
    
    def update_match_results_from_api(self, api_results: List[Dict]) -> int:
        """Обновляет результаты матчей из внешнего API"""
        updated_count = 0
        
        for result in api_results:
            try:
                success = self.logger.update_result(
                    player1=result.get('player1'),
                    player2=result.get('player2'),
                    match_date=result.get('match_date'),
                    actual_winner=result.get('winner'),
                    match_score=result.get('score', '')
                )
                
                if success:
                    updated_count += 1
                    
            except Exception as e:
                print(f"⚠️ Ошибка обновления результата: {e}")
                continue
        
        print(f"✅ Обновлено {updated_count} результатов")
        return updated_count
    
    def get_system_performance(self) -> Dict:
        """Получает производительность системы"""
        report = self.logger.get_accuracy_report()
        
        if 'error' in report:
            return {'status': 'error', 'message': report['error']}
        
        # Адаптируем для вашего интерфейса
        performance = {
            'status': 'active',
            'total_predictions': report.get('total_predictions', 0),
            'accuracy': report.get('overall_accuracy', 0),
            'bet_performance': {
                'total_bets': report.get('bet_recommendations', 0),
                'bet_accuracy': report.get('bet_accuracy', 0),
                'roi': report.get('roi_percentage', 0),
                'profit': report.get('total_profit', 0)
            },
            'recent_performance': report.get('recent_performance', {}),
            'confidence_breakdown': report.get('confidence_breakdown', {}),
            'last_updated': report.get('updated', datetime.now().isoformat())
        }
        
        return performance


# Интеграция с tennis_backend.py
def integrate_with_tennis_backend():
    """Показывает как интегрировать с вашим tennis_backend.py"""
    
    integration_code = '''
# Добавьте в tennis_backend.py в начало файла:

from complete_logging_system import PredictionLoggerIntegration

# Создайте глобальный логгер
prediction_logger = PredictionLoggerIntegration()

# В функции get_live_matches_with_underdog_focus() добавьте после генерации матчей:

def get_live_matches_with_underdog_focus():
    # ... ваш существующий код ...
    
    # В конце функции, перед return:
    if processed_matches:
        # Логируем все прогнозы
        logged_ids = prediction_logger.bulk_log_matches(processed_matches)
        print(f"📝 Залогировано прогнозов: {len(logged_ids)}")
    
    return {
        'matches': processed_matches,
        'source': f"LIVE_API_{api_result['status']}",
        'success': True,
        'logged_predictions': len(logged_ids) if processed_matches else 0
    }

# Добавьте новый API endpoint:

@app.route('/api/prediction-stats', methods=['GET'])
def get_prediction_stats():
    """Статистика прогнозов"""
    try:
        stats = prediction_logger.get_system_performance()
        return jsonify({
            'success': True,
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/update-result', methods=['POST'])
def update_match_result():
    """Обновление результата матча"""
    try:
        data = request.get_json()
        
        success = prediction_logger.logger.update_result(
            player1=data.get('player1'),
            player2=data.get('player2'),
            match_date=data.get('match_date'),
            actual_winner=data.get('actual_winner'),
            match_score=data.get('match_score', '')
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Result updated successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Match not found or already updated'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
'''
    
    print("🔗 КОД ДЛЯ ИНТЕГРАЦИИ С tennis_backend.py:")
    print("=" * 50)
    print(integration_code)


if __name__ == "__main__":
    integrate_with_tennis_backend()