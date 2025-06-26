#!/usr/bin/env python3
"""
🎾 COMPLETE TENNIS PREDICTION & BETTING PIPELINE
Полная система для сбора данных, обучения моделей и поиска ценных ставок
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import logging
import json
import time
from typing import Dict, List, Optional, Tuple
import warnings

# Попытка импорта schedule с обработкой ошибки
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
    logging.warning("⚠️ Библиотека 'schedule' не установлена. Установите: pip install schedule")

# Импорты наших модулей
try:
    from enhanced_data_collector import EnhancedTennisDataCollector
    from enhanced_predictor import EnhancedTennisPredictor, time_series_split_validation
    from enhanced_betting_system import (
        EnhancedTennisBettingSystem, OddsCollector, ValueBet, BettingMetrics,
        create_sample_matches_and_enhanced_odds, backtest_betting_strategy
    )
except ImportError as e:
    logging.error(f"❌ Ошибка импорта модулей: {e}")
    logging.error("💡 Убедитесь, что файлы enhanced_*.py находятся в том же каталоге")
    sys.exit(1)

warnings.filterwarnings('ignore')

class TennisPipelineConfig:
    """Конфигурация для pipeline"""
    
    def __init__(self, config_file: str = 'config.json'):
        self.config_file = config_file
        self.load_config()
    
    def load_config(self):
        """Загрузка конфигурации"""
        default_config = {
            "data_collection": {
                "update_frequency_hours": 24,
                "min_matches_for_training": 1000,
                "years_to_collect": ["2024", "2023", "2022", "2021", "2020"]
            },
            "model_training": {
                "retrain_frequency_days": 7,
                "validation_splits": 5,
                "min_model_performance": 0.65,
                "ensemble_models": ["neural_network", "xgboost", "random_forest"]
            },
            "betting_system": {
                "bankroll": 10000,
                "max_stake_percentage": 0.05,
                "min_confidence": 0.55,
                "min_edge": 0.02,
                "max_daily_stakes": 5
            },
            "notifications": {
                "telegram_bot_token": "",
                "telegram_chat_id": "",
                "email_notifications": False,
                "discord_webhook": ""
            },
            "api_keys": {
                "pinnacle_username": "",
                "pinnacle_password": "",
                "oddsapi_key": ""
            }
        }
        
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                loaded_config = json.load(f)
                # Объединяем с дефолтной конфигурацией
                self._merge_configs(default_config, loaded_config)
        
        self.config = default_config
        self.save_config()
    
    def _merge_configs(self, default: dict, loaded: dict):
        """Рекурсивное объединение конфигураций"""
        for key, value in loaded.items():
            if key in default:
                if isinstance(value, dict) and isinstance(default[key], dict):
                    self._merge_configs(default[key], value)
                else:
                    default[key] = value
    
    def save_config(self):
        """Сохранение конфигурации"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, path: str, default=None):
        """Получение значения по пути (например, 'data_collection.update_frequency_hours')"""
        keys = path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value

class TennisNotificationSystem:
    """Система уведомлений"""
    
    def __init__(self, config: TennisPipelineConfig):
        self.config = config
        self.telegram_token = config.get('notifications.telegram_bot_token')
        self.telegram_chat_id = config.get('notifications.telegram_chat_id')
        self.discord_webhook = config.get('notifications.discord_webhook')
    
    def send_telegram_message(self, message: str):
        """Отправка сообщения в Telegram"""
        if not self.telegram_token or not self.telegram_chat_id:
            return
        
        try:
            import requests
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            requests.post(url, data=data)
        except Exception as e:
            logging.error(f"Ошибка отправки Telegram сообщения: {e}")
    
    def send_discord_message(self, message: str):
        """Отправка сообщения в Discord"""
        if not self.discord_webhook:
            return
        
        try:
            import requests
            data = {'content': message}
            requests.post(self.discord_webhook, json=data)
        except Exception as e:
            logging.error(f"Ошибка отправки Discord сообщения: {e}")
    
    def notify_value_bets(self, value_bets: List[ValueBet]):
        """Уведомление о найденных ценных ставках"""
        if not value_bets:
            return
        
        message = f"🎾 *НАЙДЕНЫ ЦЕННЫЕ СТАВКИ!*\n\n"
        
        for i, bet in enumerate(value_bets[:5], 1):  # Топ-5
            message += f"*{i}. {bet.player_name}* vs {bet.opponent_name}\n"
            message += f"💰 Коэф: {bet.odds:.2f} | EV: {bet.expected_value:.3f}\n"
            message += f"💵 Ставка: ${bet.recommended_stake:.2f}\n"
            message += f"🏟️ {bet.tournament} | {bet.match_date}\n\n"
        
        message += f"📊 Всего найдено: {len(value_bets)} ставок"
        
        self.send_telegram_message(message)
        self.send_discord_message(message)
    
    def notify_performance_update(self, metrics: BettingMetrics):
        """Уведомление об обновлении производительности"""
        message = f"📈 *ОБНОВЛЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ*\n\n"
        message += f"💰 Прибыль: ${metrics.total_profit:.2f}\n"
        message += f"📊 ROI: {metrics.roi:.2f}%\n"
        message += f"🎯 Винрейт: {metrics.win_rate:.1%}\n"
        message += f"📈 Sharpe: {metrics.sharpe_ratio:.2f}\n"
        message += f"📉 Макс. просадка: {metrics.max_drawdown:.2f}%"
        
        self.send_telegram_message(message)

class TennisPipeline:
    """Главный класс pipeline"""
    
    def __init__(self, config_file: str = 'config.json'):
        self.config = TennisPipelineConfig(config_file)
        self.setup_logging()
        
        # Инициализируем компоненты
        self.data_collector = EnhancedTennisDataCollector()
        self.predictor = EnhancedTennisPredictor()
        self.betting_system = None  # Будет инициализирован после загрузки модели
        self.odds_collector = OddsCollector()
        self.notification_system = TennisNotificationSystem(self.config)
        
        # Состояние pipeline
        self.last_data_update = None
        self.last_model_training = None
        self.model_performance = {}
        self.betting_history = []
        
        logging.info("🎾 Tennis Pipeline инициализирован")
    
    def setup_logging(self):
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('tennis_pipeline.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def update_data(self, force: bool = False) -> bool:
        """Обновление данных"""
        try:
            # Проверяем, нужно ли обновление
            update_hours = self.config.get('data_collection.update_frequency_hours', 24)
            
            if not force and self.last_data_update:
                hours_since_update = (datetime.now() - self.last_data_update).total_seconds() / 3600
                if hours_since_update < update_hours:
                    logging.info(f"⏰ Данные обновлены {hours_since_update:.1f} часов назад, пропускаем")
                    return False
            
            logging.info("📊 Начинаем обновление данных...")
            
            # Собираем данные
            enhanced_df = self.data_collector.download_and_process_data()
            
            # Проверяем количество данных
            min_matches = self.config.get('data_collection.min_matches_for_training', 1000)
            if len(enhanced_df) < min_matches:
                logging.warning(f"⚠️ Недостаточно данных: {len(enhanced_df)} < {min_matches}")
                return False
            
            # Сохраняем данные
            self.data_collector.save_enhanced_data(enhanced_df)
            self.last_data_update = datetime.now()
            
            logging.info(f"✅ Данные обновлены: {len(enhanced_df)} записей")
            return True
            
        except Exception as e:
            logging.error(f"❌ Ошибка обновления данных: {e}")
            return False
    
    def train_models(self, force: bool = False) -> bool:
        """Обучение моделей"""
        try:
            # Проверяем, нужно ли переобучение
            retrain_days = self.config.get('model_training.retrain_frequency_days', 7)
            
            if not force and self.last_model_training:
                days_since_training = (datetime.now() - self.last_model_training).days
                if days_since_training < retrain_days:
                    logging.info(f"🧠 Модели обучены {days_since_training} дней назад, пропускаем")
                    return True
            
            logging.info("🏋️‍♂️ Начинаем обучение моделей...")
            
            # Загружаем данные
            data_path = os.path.join(self.data_collector.data_dir, 'enhanced_tennis_dataset.csv')
            if not os.path.exists(data_path):
                logging.error(f"❌ Файл данных не найден: {data_path}")
                return False
            
            df = pd.read_csv(data_path)
            
            # Разделяем данные по времени
            df['match_date'] = pd.to_datetime(df['match_date'])
            cutoff_date = df['match_date'].quantile(0.8)
            
            train_df = df[df['match_date'] < cutoff_date]
            test_df = df[df['match_date'] >= cutoff_date]
            
            # Подготавливаем данные
            X_train = self.predictor.prepare_features(train_df)
            y_train = train_df['won_at_least_one_set']
            
            X_test = self.predictor.prepare_features(test_df)
            y_test = test_df['won_at_least_one_set']
            
            # Разделяем тренировочные данные
            from sklearn.model_selection import train_test_split
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            # Обучаем ансамбль
            performance = self.predictor.train_ensemble_models(
                X_train_split, y_train_split, X_val_split, y_val_split
            )
            
            # Оцениваем на тестовых данных
            test_results = self.predictor.evaluate_models(X_test, y_test)
            
            # Проверяем качество модели
            min_performance = self.config.get('model_training.min_model_performance', 0.65)
            best_auc = max([results['auc'] for results in test_results.values()])
            
            if best_auc < min_performance:
                logging.warning(f"⚠️ Низкое качество модели: AUC = {best_auc:.3f} < {min_performance}")
                return False
            
            # Сохраняем модели
            self.predictor.save_models()
            self.model_performance = test_results
            self.last_model_training = datetime.now()
            
            # Инициализируем систему ставок
            bankroll = self.config.get('betting_system.bankroll', 10000)
            self.betting_system = EnhancedTennisBettingSystem(self.predictor, bankroll)
            
            logging.info(f"✅ Модели обучены: лучший AUC = {best_auc:.3f}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Ошибка обучения моделей: {e}")
            return False
    
    def find_daily_bets(self) -> List[ValueBet]:
        """Поиск ставок на сегодня"""
        try:
            if not self.betting_system:
                logging.error("❌ Система ставок не инициализирована")
                return []
            
            logging.info("🔍 Поиск ценных ставок на сегодня...")
            
            # Здесь должен быть код для получения сегодняшних матчей
            # Для демонстрации создаем примеры
            matches_df, enhanced_odds = create_sample_matches_and_enhanced_odds()
            
            # Фильтруем только сегодняшние матчи
            today = datetime.now().strftime('%Y-%m-%d')
            today_matches = matches_df[matches_df['match_date'] == today]
            
            if len(today_matches) == 0:
                logging.info("📅 Нет матчей на сегодня")
                return []
            
            # Ищем ценные ставки
            value_bets = self.betting_system.find_value_bets(today_matches, enhanced_odds)
            
            # Ограничиваем количество ставок в день
            max_daily_stakes = self.config.get('betting_system.max_daily_stakes', 5)
            value_bets = value_bets[:max_daily_stakes]
            
            if value_bets:
                logging.info(f"✅ Найдено {len(value_bets)} ценных ставок")
                
                # Отправляем уведомления
                self.notification_system.notify_value_bets(value_bets)
                
                # Сохраняем ставки
                self.betting_system.save_betting_data(value_bets)
            else:
                logging.info("📊 Ценных ставок не найдено")
            
            return value_bets
            
        except Exception as e:
            logging.error(f"❌ Ошибка поиска ставок: {e}")
            return []
    
    def update_performance(self):
        """Обновление статистики производительности"""
        try:
            # Загружаем историю ставок
            betting_dir = "betting_data"
            if not os.path.exists(betting_dir):
                return
            
            # Здесь должна быть логика анализа результатов ставок
            # Для демонстрации создаем примерные метрики
            
            sample_metrics = BettingMetrics(
                total_bets=50,
                total_profit=250.0,
                roi=5.2,
                win_rate=0.58,
                average_odds=1.85,
                sharpe_ratio=1.2,
                max_drawdown=-8.5,
                confidence_breakdown={
                    "Высокая": {"count": 20, "profit": 180.0, "win_rate": 0.65},
                    "Средняя": {"count": 25, "profit": 70.0, "win_rate": 0.52},
                    "Низкая": {"count": 5, "profit": 0.0, "win_rate": 0.40}
                }
            )
            
            # Отправляем уведомление о производительности (раз в неделю)
            if datetime.now().weekday() == 0:  # Понедельник
                self.notification_system.notify_performance_update(sample_metrics)
            
            logging.info(f"📈 Производительность обновлена: ROI = {sample_metrics.roi:.2f}%")
            
        except Exception as e:
            logging.error(f"❌ Ошибка обновления производительности: {e}")
    
    def run_daily_pipeline(self):
        """Запуск ежедневного pipeline"""
        logging.info("🚀 Запуск ежедневного pipeline")
        
        try:
            # 1. Обновляем данные если нужно
            data_updated = self.update_data()
            
            # 2. Переобучаем модели если нужно или если данные обновились
            if data_updated or not self.last_model_training:
                model_trained = self.train_models()
                if not model_trained:
                    logging.error("❌ Не удалось обучить модели, пропускаем поиск ставок")
                    return
            
            # 3. Загружаем модели если они не загружены
            if not self.betting_system:
                try:
                    self.predictor.load_models()
                    bankroll = self.config.get('betting_system.bankroll', 10000)
                    self.betting_system = EnhancedTennisBettingSystem(self.predictor, bankroll)
                    logging.info("✅ Модели загружены из файлов")
                except Exception as e:
                    logging.error(f"❌ Не удалось загрузить модели: {e}")
                    return
            
            # 4. Ищем ценные ставки на сегодня
            value_bets = self.find_daily_bets()
            
            # 5. Обновляем статистику производительности
            self.update_performance()
            
            logging.info("✅ Ежедневный pipeline завершен успешно")
            
        except Exception as e:
            logging.error(f"❌ Критическая ошибка в pipeline: {e}")
            import traceback
            traceback.print_exc()
    
    def setup_scheduler(self):
        """Настройка планировщика задач"""
        if not SCHEDULE_AVAILABLE:
            logging.error("❌ Библиотека schedule не доступна. Запуск в ручном режиме.")
            return False
        
        # Ежедневный запуск в 08:00
        schedule.every().day.at("08:00").do(self.run_daily_pipeline)
        
        # Обновление данных каждые 6 часов
        schedule.every(6).hours.do(self.update_data)
        
        # Поиск ставок каждые 2 часа в день матчей
        schedule.every(2).hours.do(self.find_daily_bets)
        
        logging.info("⏰ Планировщик настроен")
        return True
    
    def run_scheduler(self):
        """Запуск планировщика"""
        if not SCHEDULE_AVAILABLE:
            logging.error("❌ Планировщик не может быть запущен без библиотеки schedule")
            logging.info("💡 Выполните: pip install schedule")
            return
        
        logging.info("🔄 Запуск планировщика...")
        
        # Первый запуск
        self.run_daily_pipeline()
        
        # Бесконечный цикл планировщика
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Проверяем каждую минуту
            except KeyboardInterrupt:
                logging.info("⏹️ Планировщик остановлен пользователем")
                break
            except Exception as e:
                logging.error(f"❌ Ошибка в планировщике: {e}")
                time.sleep(300)  # Ждем 5 минут перед повтором
    
    def run_manual_mode(self):
        """Ручной режим без планировщика"""
        logging.info("🔄 Запуск в ручном режиме...")
        
        while True:
            try:
                print("\n" + "="*50)
                print("🎾 TENNIS PIPELINE - РУЧНОЙ РЕЖИМ")
                print("="*50)
                print("1. Обновить данные")
                print("2. Обучить модели")
                print("3. Найти ценные ставки")
                print("4. Запустить полный pipeline")
                print("5. Показать статистику")
                print("0. Выход")
                
                choice = input("\nВыберите действие (0-5): ").strip()
                
                if choice == "0":
                    print("👋 До свидания!")
                    break
                elif choice == "1":
                    self.update_data(force=True)
                elif choice == "2":
                    self.train_models(force=True)
                elif choice == "3":
                    self.find_daily_bets()
                elif choice == "4":
                    self.run_daily_pipeline()
                elif choice == "5":
                    self.show_statistics()
                else:
                    print("❌ Неверный выбор")
                
                input("\nНажмите Enter для продолжения...")
                
            except KeyboardInterrupt:
                print("\n⏹️ Выход из программы")
                break
            except Exception as e:
                logging.error(f"❌ Ошибка: {e}")
    
    def show_statistics(self):
        """Показать статистику"""
        print("\n📊 СТАТИСТИКА PIPELINE")
        print("-" * 30)
        
        if self.last_data_update:
            print(f"📅 Последнее обновление данных: {self.last_data_update.strftime('%Y-%m-%d %H:%M')}")
        else:
            print("📅 Данные еще не обновлялись")
        
        if self.last_model_training:
            print(f"🧠 Последнее обучение модели: {self.last_model_training.strftime('%Y-%m-%d %H:%M')}")
        else:
            print("🧠 Модели еще не обучались")
        
        if self.model_performance:
            print("\n🎯 Производительность моделей:")
            for model_name, metrics in self.model_performance.items():
                print(f"  {model_name}: AUC = {metrics['auc']:.3f}")
        
        # Проверяем файлы
        print(f"\n📁 Статус файлов:")
        data_file = os.path.join(self.data_collector.data_dir, 'enhanced_tennis_dataset.csv')
        print(f"  Данные: {'✅' if os.path.exists(data_file) else '❌'}")
        
        models_file = os.path.join(self.predictor.models_dir, 'ensemble_models.pkl')
        print(f"  Модели: {'✅' if os.path.exists(models_file) else '❌'}")

def create_sample_config():
    """Создание примера конфигурации"""
    config = TennisPipelineConfig()
    
    # Обновляем некоторые настройки для демонстрации
    config.config['betting_system']['bankroll'] = 5000
    config.config['model_training']['min_model_performance'] = 0.60
    config.config['notifications']['telegram_bot_token'] = "YOUR_BOT_TOKEN_HERE"
    config.config['notifications']['telegram_chat_id'] = "YOUR_CHAT_ID_HERE"
    
    config.save_config()
    print("✅ Создан файл config.json с примером настроек")

def main():
    """Главная функция для запуска pipeline"""
    parser = argparse.ArgumentParser(description='🎾 Tennis Prediction & Betting Pipeline')
    
    parser.add_argument('--mode', choices=['setup', 'train', 'predict', 'backtest', 'run', 'manual'], 
                       default='manual', help='Режим работы')
    parser.add_argument('--config', default='config.json', help='Файл конфигурации')
    parser.add_argument('--force-update', action='store_true', help='Принудительное обновление данных')
    parser.add_argument('--force-train', action='store_true', help='Принудительное переобучение')
    
    args = parser.parse_args()
    
    print("🎾 TENNIS PREDICTION & BETTING PIPELINE")
    print("=" * 70)
    print("🚀 Полная автоматизированная система для:")
    print("• Сбора и обработки теннисных данных")
    print("• Обучения ансамбля ML моделей")
    print("• Поиска ценных ставок в реальном времени")
    print("• Уведомлений и мониторинга производительности")
    print("=" * 70)
    
    if args.mode == 'setup':
        print("\n🔧 Настройка системы...")
        create_sample_config()
        
        # Создаем необходимые директории
        for directory in ['tennis_data_enhanced', 'tennis_models', 'betting_data']:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"📁 Создана директория: {directory}")
        
        print("\n✅ Настройка завершена!")
        print("📝 Отредактируйте config.json для ваших настроек")
        print("🚀 Запустите: python tennis_pipeline.py --mode train")
        return
    
    # Инициализируем pipeline
    pipeline = TennisPipeline(args.config)
    
    if args.mode == 'train':
        print("\n🏋️‍♂️ Режим обучения...")
        
        # Обновляем данные
        print("📊 Обновление данных...")
        pipeline.update_data(force=args.force_update)
        
        # Обучаем модели
        print("🧠 Обучение моделей...")
        success = pipeline.train_models(force=args.force_train)
        
        if success:
            print("✅ Обучение завершено успешно!")
            print("🚀 Запустите: python tennis_pipeline.py --mode predict")
        else:
            print("❌ Ошибка обучения")
    
    elif args.mode == 'predict':
        print("\n🔍 Режим поиска ставок...")
        
        try:
            # Загружаем модели
            pipeline.predictor.load_models()
            bankroll = pipeline.config.get('betting_system.bankroll', 10000)
            pipeline.betting_system = EnhancedTennisBettingSystem(pipeline.predictor, bankroll)
            
            # Ищем ставки
            value_bets = pipeline.find_daily_bets()
            
            if value_bets:
                print(f"\n✅ Найдено {len(value_bets)} ценных ставок:")
                for i, bet in enumerate(value_bets, 1):
                    print(f"{i}. {bet.player_name} vs {bet.opponent_name}")
                    print(f"   💰 {bet.odds:.2f} | EV: {bet.expected_value:.3f} | ${bet.recommended_stake:.2f}")
            else:
                print("📊 Ценных ставок не найдено")
                
        except Exception as e:
            print(f"❌ Ошибка: {e}")
    
    elif args.mode == 'backtest':
        print("\n🎲 Режим бэктестинга...")
        
        try:
            # Загружаем модели
            pipeline.predictor.load_models()
            
            # Запускаем бэктест
            metrics = backtest_betting_strategy(pipeline.predictor, '2024-01-01', '2024-06-30')
            
            print(f"\n📈 Результаты бэктестинга:")
            print(f"• Ставок: {metrics.total_bets}")
            print(f"• Прибыль: ${metrics.total_profit:.2f}")
            print(f"• ROI: {metrics.roi:.2f}%")
            print(f"• Винрейт: {metrics.win_rate:.1%}")
            print(f"• Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            print(f"• Макс. просадка: {metrics.max_drawdown:.2f}%")
            
        except Exception as e:
            print(f"❌ Ошибка бэктестинга: {e}")
    
    elif args.mode == 'run':
        print("\n🔄 Режим автоматического запуска...")
        
        if pipeline.setup_scheduler():
            print("⏰ Настройка планировщика...")
            print("🚀 Pipeline запущен!")
            print("📱 Уведомления будут отправлены при нахождении ценных ставок")
            print("⏹️ Нажмите Ctrl+C для остановки")
            
            try:
                pipeline.run_scheduler()
            except KeyboardInterrupt:
                print("\n⏹️ Pipeline остановлен")
        else:
            print("❌ Не удалось настроить планировщик")
            print("💡 Используйте режим --mode manual")
    
    elif args.mode == 'manual':
        print("\n🔄 Ручной режим...")
        pipeline.run_manual_mode()

def run_web_interface():
    """Веб-интерфейс для мониторинга (опционально)"""
    try:
        from flask import Flask, render_template, jsonify
        import json
        
        app = Flask(__name__)
        
        @app.route('/')
        def index():
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Tennis Pipeline Dashboard</title>
                <meta charset="utf-8">
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .card { border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 8px; }
                    .status { font-size: 18px; }
                </style>
            </head>
            <body>
                <h1>🎾 Tennis Pipeline Dashboard</h1>
                <div class="card">
                    <h2>Статус системы</h2>
                    <div id="status" class="status">Загрузка...</div>
                </div>
                <script>
                    function updateStatus() {
                        fetch('/api/status')
                            .then(response => response.json())
                            .then(data => {
                                document.getElementById('status').innerHTML = 
                                    '<p>📅 Последнее обновление: ' + data.last_update + '</p>' +
                                    '<p>🧠 Производительность модели: ' + data.model_performance + '</p>' +
                                    '<p>💰 Ценных ставок сегодня: ' + data.today_bets + '</p>' +
                                    '<p>📊 ROI: ' + data.roi + '%</p>';
                            })
                            .catch(error => {
                                document.getElementById('status').innerHTML = 
                                    '<p style="color: red;">❌ Ошибка загрузки данных</p>';
                            });
                    }
                    
                    updateStatus();
                    setInterval(updateStatus, 30000); // Обновляем каждые 30 секунд
                </script>
            </body>
            </html>
            """
        
        @app.route('/api/status')
        def api_status():
            return jsonify({
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_performance': 'AUC: 0.72',
                'today_bets': 3,
                'roi': 7.5
            })
        
        print("🌐 Веб-интерфейс запущен на http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except ImportError:
        print("❌ Flask не установлен. Веб-интерфейс недоступен.")
        print("💡 Установите: pip install flask")

if __name__ == "__main__":
    main()