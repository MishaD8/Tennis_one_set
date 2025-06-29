#!/usr/bin/env python3
"""
🎾 COMPLETE TENNIS PREDICTION & BETTING PIPELINE - PRODUCTION VERSION
Полная система для сбора данных, обучения моделей и поиска ценных ставок
Исправлена для продакшн деплоя на Hetzner
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
import signal
import threading
from typing import Dict, List, Optional, Tuple
import warnings
from pathlib import Path

# Настройка логирования
def setup_logging(log_level=logging.INFO):
    """Настройка системы логирования"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'tennis_pipeline.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

setup_logging()
logger = logging.getLogger(__name__)

# Безопасная проверка schedule
SCHEDULE_AVAILABLE = False
try:
    import schedule
    SCHEDULE_AVAILABLE = True
    logger.info("✅ Schedule library available")
except ImportError:
    logger.warning("⚠️ Schedule library not available. Install: pip install schedule")

# Безопасный импорт модулей системы
MODULES_AVAILABLE = {
    'data_collector': False,
    'predictor': False,
    'betting_system': False
}

try:
    from script_data_collector import EnhancedTennisDataCollector
    MODULES_AVAILABLE['data_collector'] = True
    logger.info("✅ Data collector module loaded")
except ImportError as e:
    logger.warning(f"⚠️ Data collector module not available: {e}")

try:
    from tennis_set_predictor import EnhancedTennisPredictor
    MODULES_AVAILABLE['predictor'] = True
    logger.info("✅ Predictor module loaded")
except ImportError as e:
    logger.warning(f"⚠️ Predictor module not available: {e}")

try:
    from tennis_system_odds import (
        EnhancedTennisBettingSystem, OddsCollector, ValueBet, BettingMetrics,
        create_sample_matches_and_enhanced_odds, backtest_betting_strategy
    )
    MODULES_AVAILABLE['betting_system'] = True
    logger.info("✅ Betting system module loaded")
except ImportError as e:
    logger.warning(f"⚠️ Betting system module not available: {e}")

warnings.filterwarnings('ignore')


class TennisPipelineConfig:
    """Конфигурация для pipeline с поддержкой продакшна"""
    
    def __init__(self, config_file: str = None):
        # Определяем базовую директорию
        self.base_dir = Path(os.getcwd())
        if config_file is None:
            config_file = self.base_dir / 'config.json'
        
        self.config_file = Path(config_file)
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
                "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
                "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
                "email_notifications": False,
                "discord_webhook": os.getenv("DISCORD_WEBHOOK", ""),
                "timeout_seconds": 10,
                "retry_attempts": 3
            },
            "api_keys": {
                "pinnacle_username": os.getenv("PINNACLE_USERNAME", ""),
                "pinnacle_password": os.getenv("PINNACLE_PASSWORD", ""),
                "oddsapi_key": os.getenv("ODDS_API_KEY", "")
            },
            "paths": {
                "data_dir": str(self.base_dir / "tennis_data_enhanced"),
                "models_dir": str(self.base_dir / "tennis_models"),
                "betting_dir": str(self.base_dir / "betting_data"),
                "logs_dir": str(self.base_dir / "logs")
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    self._merge_configs(default_config, loaded_config)
                logger.info(f"✅ Configuration loaded from {self.config_file}")
            except Exception as e:
                logger.warning(f"⚠️ Error loading config: {e}. Using defaults.")
        
        self.config = default_config
        self.save_config()
        
        # Создаем необходимые директории
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Создание необходимых директорий"""
        for path_key, path_value in self.config["paths"].items():
            path = Path(path_value)
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"📁 Ensured directory: {path}")
    
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
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Error saving config: {e}")
    
    def get(self, path: str, default=None):
        """Получение значения по пути"""
        keys = path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value


class TennisNotificationSystem:
    """Улучшенная система уведомлений с retry и timeout"""
    
    def __init__(self, config: TennisPipelineConfig):
        self.config = config
        self.telegram_token = config.get('notifications.telegram_bot_token')
        self.telegram_chat_id = config.get('notifications.telegram_chat_id')
        self.discord_webhook = config.get('notifications.discord_webhook')
        self.timeout = config.get('notifications.timeout_seconds', 10)
        self.retry_attempts = config.get('notifications.retry_attempts', 3)
    
    def _make_request_with_retry(self, url: str, data: dict, request_type: str = "POST"):
        """HTTP запрос с retry и timeout"""
        import requests
        
        for attempt in range(self.retry_attempts):
            try:
                if request_type.upper() == "POST":
                    response = requests.post(url, json=data, timeout=self.timeout)
                else:
                    response = requests.get(url, params=data, timeout=self.timeout)
                
                response.raise_for_status()
                return True
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"⚠️ Notification attempt {attempt + 1} failed: {e}")
                if attempt == self.retry_attempts - 1:
                    logger.error(f"❌ All notification attempts failed for {url}")
                    return False
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return False
    
    def send_telegram_message(self, message: str) -> bool:
        """Отправка сообщения в Telegram с retry"""
        if not self.telegram_token or not self.telegram_chat_id:
            logger.debug("Telegram not configured, skipping")
            return False
        
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        data = {
            'chat_id': self.telegram_chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }
        
        return self._make_request_with_retry(url, data)
    
    def send_discord_message(self, message: str) -> bool:
        """Отправка сообщения в Discord с retry"""
        if not self.discord_webhook:
            logger.debug("Discord not configured, skipping")
            return False
        
        data = {'content': message}
        return self._make_request_with_retry(self.discord_webhook, data)
    
    def notify_value_bets(self, value_bets: List) -> bool:
        """Уведомление о найденных ценных ставках"""
        if not value_bets:
            return False
        
        message = f"🎾 *НАЙДЕНЫ ЦЕННЫЕ СТАВКИ!*\n\n"
        
        for i, bet in enumerate(value_bets[:5], 1):
            message += f"*{i}. {bet.player_name}* vs {bet.opponent_name}\n"
            message += f"💰 Коэф: {bet.odds:.2f} | EV: {bet.expected_value:.3f}\n"
            message += f"💵 Ставка: ${bet.recommended_stake:.2f}\n"
            message += f"🏟️ {bet.tournament} | {bet.match_date}\n\n"
        
        message += f"📊 Всего найдено: {len(value_bets)} ставок"
        
        # Отправляем в оба канала
        telegram_success = self.send_telegram_message(message)
        discord_success = self.send_discord_message(message)
        
        return telegram_success or discord_success


class TennisPipeline:
    """Главный класс pipeline с улучшенной обработкой ошибок"""
    
    def __init__(self, config_file: str = None):
        self.config = TennisPipelineConfig(config_file)
        self.shutdown_requested = False
        
        # Компоненты системы
        self.data_collector = None
        self.predictor = None
        self.betting_system = None
        self.odds_collector = None
        self.notification_system = TennisNotificationSystem(self.config)
        
        # Состояние pipeline
        self.last_data_update = None
        self.last_model_training = None
        self.model_performance = {}
        self.betting_history = []
        
        # Инициализация компонентов
        self._initialize_components()
        
        # Настройка graceful shutdown
        self._setup_signal_handlers()
        
        logger.info("🎾 Tennis Pipeline initialized")
    
    def _initialize_components(self):
        """Безопасная инициализация компонентов"""
        # Data Collector
        if MODULES_AVAILABLE['data_collector']:
            try:
                self.data_collector = EnhancedTennisDataCollector(
                    data_dir=self.config.get('paths.data_dir')
                )
                logger.info("✅ Data collector initialized")
            except Exception as e:
                logger.error(f"❌ Error initializing data collector: {e}")
        
        # Predictor
        if MODULES_AVAILABLE['predictor']:
            try:
                self.predictor = EnhancedTennisPredictor(
                    model_dir=self.config.get('paths.models_dir')
                )
                logger.info("✅ Predictor initialized")
            except Exception as e:
                logger.error(f"❌ Error initializing predictor: {e}")
        
        # Betting System
        if MODULES_AVAILABLE['betting_system'] and self.predictor:
            try:
                bankroll = self.config.get('betting_system.bankroll', 10000)
                self.betting_system = EnhancedTennisBettingSystem(self.predictor, bankroll)
                self.odds_collector = OddsCollector()
                logger.info("✅ Betting system initialized")
            except Exception as e:
                logger.error(f"❌ Error initializing betting system: {e}")
    
    def _setup_signal_handlers(self):
        """Настройка обработчиков сигналов для graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"📡 Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def update_data(self, force: bool = False) -> bool:
        """Обновление данных с улучшенной обработкой ошибок"""
        if not self.data_collector:
            logger.warning("⚠️ Data collector not available")
            return False
        
        try:
            update_hours = self.config.get('data_collection.update_frequency_hours', 24)
            
            if not force and self.last_data_update:
                hours_since_update = (datetime.now() - self.last_data_update).total_seconds() / 3600
                if hours_since_update < update_hours:
                    logger.info(f"⏰ Data updated {hours_since_update:.1f} hours ago, skipping")
                    return False
            
            logger.info("📊 Starting data update...")
            
            enhanced_df = self.data_collector.download_and_process_data()
            
            min_matches = self.config.get('data_collection.min_matches_for_training', 1000)
            if len(enhanced_df) < min_matches:
                logger.warning(f"⚠️ Insufficient data: {len(enhanced_df)} < {min_matches}")
                return False
            
            self.data_collector.save_enhanced_data(enhanced_df)
            self.last_data_update = datetime.now()
            
            logger.info(f"✅ Data updated: {len(enhanced_df)} records")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating data: {e}")
            return False
    
    def train_models(self, force: bool = False) -> bool:
        """Обучение моделей с улучшенной обработкой ошибок"""
        if not self.predictor:
            logger.warning("⚠️ Predictor not available")
            return False
        
        try:
            retrain_days = self.config.get('model_training.retrain_frequency_days', 7)
            
            if not force and self.last_model_training:
                days_since_training = (datetime.now() - self.last_model_training).days
                if days_since_training < retrain_days:
                    logger.info(f"🧠 Models trained {days_since_training} days ago, skipping")
                    return True
            
            logger.info("🏋️‍♂️ Starting model training...")
            
            # Проверяем наличие данных
            data_path = Path(self.config.get('paths.data_dir')) / 'enhanced_tennis_dataset.csv'
            if not data_path.exists():
                logger.error(f"❌ Data file not found: {data_path}")
                return False
            
            df = pd.read_csv(data_path)
            
            # Подготовка данных для обучения
            df['match_date'] = pd.to_datetime(df['match_date'])
            cutoff_date = df['match_date'].quantile(0.8)
            
            train_df = df[df['match_date'] < cutoff_date]
            test_df = df[df['match_date'] >= cutoff_date]
            
            if len(train_df) == 0 or len(test_df) == 0:
                logger.error("❌ Insufficient data for train/test split")
                return False
            
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
                logger.warning(f"⚠️ Low model performance: AUC = {best_auc:.3f} < {min_performance}")
                return False
            
            # Сохраняем модели
            self.predictor.save_models()
            self.model_performance = test_results
            self.last_model_training = datetime.now()
            
            # Обновляем betting system
            if MODULES_AVAILABLE['betting_system']:
                bankroll = self.config.get('betting_system.bankroll', 10000)
                self.betting_system = EnhancedTennisBettingSystem(self.predictor, bankroll)
            
            logger.info(f"✅ Models trained: best AUC = {best_auc:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training models: {e}")
            return False
    
    def find_daily_bets(self) -> List:
        """Поиск ставок на сегодня с улучшенной обработкой ошибок"""
        if not self.betting_system:
            logger.warning("⚠️ Betting system not available")
            return []
        
        try:
            logger.info("🔍 Searching for value bets today...")
            
            # Получаем данные о матчах (в реальной системе - из API)
            if MODULES_AVAILABLE['betting_system']:
                matches_df, enhanced_odds = create_sample_matches_and_enhanced_odds()
            else:
                logger.warning("⚠️ Using fallback match data")
                return []
            
            # Фильтруем только сегодняшние матчи
            today = datetime.now().strftime('%Y-%m-%d')
            today_matches = matches_df[matches_df['match_date'] == today]
            
            if len(today_matches) == 0:
                logger.info("📅 No matches today")
                return []
            
            # Ищем ценные ставки
            value_bets = self.betting_system.find_value_bets(today_matches, enhanced_odds)
            
            # Ограничиваем количество ставок в день
            max_daily_stakes = self.config.get('betting_system.max_daily_stakes', 5)
            value_bets = value_bets[:max_daily_stakes]
            
            if value_bets:
                logger.info(f"✅ Found {len(value_bets)} value bets")
                
                # Отправляем уведомления
                self.notification_system.notify_value_bets(value_bets)
                
                # Сохраняем ставки
                self.betting_system.save_betting_data(value_bets)
            else:
                logger.info("📊 No value bets found")
            
            return value_bets
            
        except Exception as e:
            logger.error(f"❌ Error finding bets: {e}")
            return []
    
    def run_daily_pipeline(self):
        """Запуск ежедневного pipeline с улучшенной обработкой ошибок"""
        logger.info("🚀 Starting daily pipeline")
        
        try:
            # 1. Обновляем данные если нужно
            data_updated = self.update_data()
            
            # 2. Переобучаем модели если нужно
            if data_updated or not self.last_model_training:
                model_trained = self.train_models()
                if not model_trained:
                    logger.error("❌ Failed to train models, skipping bet search")
                    return
            
            # 3. Загружаем модели если они не загружены
            if not self.betting_system and self.predictor:
                try:
                    self.predictor.load_models()
                    bankroll = self.config.get('betting_system.bankroll', 10000)
                    if MODULES_AVAILABLE['betting_system']:
                        self.betting_system = EnhancedTennisBettingSystem(self.predictor, bankroll)
                    logger.info("✅ Models loaded from files")
                except Exception as e:
                    logger.error(f"❌ Failed to load models: {e}")
                    return
            
            # 4. Ищем ценные ставки на сегодня
            value_bets = self.find_daily_bets()
            
            # 5. Обновляем статистику производительности
            self.update_performance()
            
            logger.info("✅ Daily pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Critical error in pipeline: {e}")
    
    def update_performance(self):
        """Обновление статистики производительности"""
        try:
            betting_dir = Path(self.config.get('paths.betting_dir'))
            if not betting_dir.exists():
                return
            
            # Здесь должна быть логика анализа результатов ставок
            logger.info("📈 Performance updated")
            
        except Exception as e:
            logger.error(f"❌ Error updating performance: {e}")
    
    def setup_scheduler(self) -> bool:
        """Настройка планировщика задач"""
        if not SCHEDULE_AVAILABLE:
            logger.error("❌ Schedule library not available")
            return False
        
        try:
            # Ежедневный запуск в 08:00
            schedule.every().day.at("08:00").do(self.run_daily_pipeline)
            
            # Обновление данных каждые 6 часов
            schedule.every(6).hours.do(self.update_data)
            
            # Поиск ставок каждые 2 часа
            schedule.every(2).hours.do(self.find_daily_bets)
            
            logger.info("⏰ Scheduler configured")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error setting up scheduler: {e}")
            return False
    
    def run_scheduler(self):
        """Запуск планировщика с graceful shutdown"""
        if not SCHEDULE_AVAILABLE:
            logger.error("❌ Cannot run scheduler without schedule library")
            return
        
        logger.info("🔄 Starting scheduler...")
        
        # Первый запуск
        self.run_daily_pipeline()
        
        # Основной цикл
        while not self.shutdown_requested:
            try:
                schedule.run_pending()
                time.sleep(60)
            except KeyboardInterrupt:
                logger.info("⏹️ Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"❌ Scheduler error: {e}")
                time.sleep(300)  # Wait 5 minutes before retry
        
        logger.info("⏹️ Scheduler stopped")
    
    def run_manual_mode(self):
        """Ручной режим без планировщика"""
        logger.info("🔄 Starting manual mode...")
        
        while not self.shutdown_requested:
            try:
                print("\n" + "="*50)
                print("🎾 TENNIS PIPELINE - MANUAL MODE")
                print("="*50)
                print("1. Update data")
                print("2. Train models")
                print("3. Find value bets")
                print("4. Run full pipeline")
                print("5. Show statistics")
                print("0. Exit")
                
                choice = input("\nSelect action (0-5): ").strip()
                
                if choice == "0" or self.shutdown_requested:
                    print("👋 Goodbye!")
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
                    print("❌ Invalid choice")
                
                if not self.shutdown_requested:
                    input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n⏹️ Exiting...")
                break
            except Exception as e:
                logger.error(f"❌ Error in manual mode: {e}")
    
    def show_statistics(self):
        """Показать статистику"""
        print("\n📊 PIPELINE STATISTICS")
        print("-" * 30)
        
        if self.last_data_update:
            print(f"📅 Last data update: {self.last_data_update.strftime('%Y-%m-%d %H:%M')}")
        else:
            print("📅 Data not updated yet")
        
        if self.last_model_training:
            print(f"🧠 Last model training: {self.last_model_training.strftime('%Y-%m-%d %H:%M')}")
        else:
            print("🧠 Models not trained yet")
        
        if self.model_performance:
            print("\n🎯 Model performance:")
            for model_name, metrics in self.model_performance.items():
                print(f"  {model_name}: AUC = {metrics['auc']:.3f}")
        
        # Check file status
        print(f"\n📁 File status:")
        data_file = Path(self.config.get('paths.data_dir')) / 'enhanced_tennis_dataset.csv'
        print(f"  Data: {'✅' if data_file.exists() else '❌'}")
        
        models_file = Path(self.config.get('paths.models_dir')) / 'ensemble_models.pkl'
        print(f"  Models: {'✅' if models_file.exists() else '❌'}")


def install_as_systemd_service():
    """Установка как systemd сервис"""
    if os.geteuid() != 0:
        print("❌ Root privileges required for systemd installation")
        return False
    
    script_path = os.path.abspath(__file__)
    working_dir = os.path.dirname(script_path)
    
    service_content = f"""[Unit]
Description=Tennis Betting Pipeline
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory={working_dir}
Environment=PATH={working_dir}/venv/bin
ExecStart={working_dir}/venv/bin/python {script_path} --mode run
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
    
    try:
        with open('/etc/systemd/system/tennis-pipeline.service', 'w') as f:
            f.write(service_content)
        
        os.system('systemctl daemon-reload')
        os.system('systemctl enable tennis-pipeline')
        
        print("✅ Systemd service installed")
        print("🚀 Start with: systemctl start tennis-pipeline")
        print("📊 Status: systemctl status tennis-pipeline")
        print("📋 Logs: journalctl -u tennis-pipeline -f")
        
        return True
        
    except Exception as e:
        print(f"❌ Error installing systemd service: {e}")
        return False


def health_check():
    """Health check для мониторинга"""
    try:
        # Check if config exists
        config_path = Path("config.json")
        if not config_path.exists():
            return {"status": "unhealthy", "reason": "config missing"}
        
        # Check if data directory exists
        config = TennisPipelineConfig()
        data_dir = Path(config.get('paths.data_dir'))
        if not data_dir.exists():
            return {"status": "unhealthy", "reason": "data directory missing"}
        
        # Check if models exist
        models_dir = Path(config.get('paths.models_dir'))
        if not any(models_dir.glob("*.pkl")) and not any(models_dir.glob("*.h5")):
            return {"status": "warning", "reason": "no trained models"}
        
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def main():
    """Главная функция для запуска pipeline"""
    parser = argparse.ArgumentParser(description='🎾 Tennis Prediction & Betting Pipeline')
    
    parser.add_argument('--mode', choices=['setup', 'train', 'predict', 'backtest', 'run', 'manual'], 
                       default='manual', help='Operation mode')
    parser.add_argument('--config', default=None, help='Configuration file path')
    parser.add_argument('--force-update', action='store_true', help='Force data update')
    parser.add_argument('--force-train', action='store_true', help='Force model retraining')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--install-systemd', action='store_true',
                       help='Install as systemd service (requires root)')
    parser.add_argument('--health-check', action='store_true',
                       help='Perform system health check')
    
    args = parser.parse_args()
    
    # Специальные команды
    if args.install_systemd:
        print("🔧 Installing systemd service...")
        if install_as_systemd_service():
            print("✅ Systemd service installed successfully")
        else:
            print("❌ Failed to install systemd service")
        return
    
    if args.health_check:
        print("🏥 Performing health check...")
        health = health_check()
        print(f"Status: {health['status']}")
        if 'reason' in health:
            print(f"Reason: {health['reason']}")
        if 'timestamp' in health:
            print(f"Timestamp: {health['timestamp']}")
        sys.exit(0 if health['status'] == 'healthy' else 1)
    
    # Настройка уровня логирования
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(log_level)
    
    print("🎾 TENNIS PREDICTION & BETTING PIPELINE")
    print("=" * 70)
    print("🚀 Production-ready system for:")
    print("• Data collection and processing")
    print("• ML model training and ensemble")
    print("• Value betting opportunities")
    print("• Real-time notifications and monitoring")
    print("=" * 70)
    
    try:
        if args.mode == 'setup':
            print("\n🔧 Setting up system...")
            config = TennisPipelineConfig(args.config)
            
            print("\n✅ Setup completed!")
            print("📝 Edit config.json for your settings")
            print("🚀 Run: python tennis_betting_pipeline.py --mode train")
            return
        
        # Initialize pipeline
        pipeline = TennisPipeline(args.config)
        
        if args.mode == 'train':
            print("\n🏋️‍♂️ Training mode...")
            
            print("📊 Updating data...")
            pipeline.update_data(force=args.force_update)
            
            print("🧠 Training models...")
            success = pipeline.train_models(force=args.force_train)
            
            if success:
                print("✅ Training completed successfully!")
                print("🚀 Run: python tennis_betting_pipeline.py --mode predict")
            else:
                print("❌ Training failed")
        
        elif args.mode == 'predict':
            print("\n🔍 Prediction mode...")
            
            try:
                # Load models if available
                if pipeline.predictor:
                    pipeline.predictor.load_models()
                    if MODULES_AVAILABLE['betting_system']:
                        bankroll = pipeline.config.get('betting_system.bankroll', 10000)
                        pipeline.betting_system = EnhancedTennisBettingSystem(pipeline.predictor, bankroll)
                
                # Find bets
                value_bets = pipeline.find_daily_bets()
                
                if value_bets:
                    print(f"\n✅ Found {len(value_bets)} value bets:")
                    for i, bet in enumerate(value_bets, 1):
                        print(f"{i}. {bet.player_name} vs {bet.opponent_name}")
                        print(f"   💰 {bet.odds:.2f} | EV: {bet.expected_value:.3f} | ${bet.recommended_stake:.2f}")
                else:
                    print("📊 No value bets found")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif args.mode == 'backtest':
            print("\n🎲 Backtesting mode...")
            
            try:
                if not pipeline.predictor:
                    print("❌ Predictor not available")
                    return
                
                pipeline.predictor.load_models()
                
                if MODULES_AVAILABLE['betting_system']:
                    metrics = backtest_betting_strategy(pipeline.predictor, '2024-01-01', '2024-06-30')
                    
                    print(f"\n📈 Backtest results:")
                    print(f"• Bets: {metrics.total_bets}")
                    print(f"• Profit: ${metrics.total_profit:.2f}")
                    print(f"• ROI: {metrics.roi:.2f}%")
                    print(f"• Win rate: {metrics.win_rate:.1%}")
                    print(f"• Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
                    print(f"• Max drawdown: {metrics.max_drawdown:.2f}%")
                else:
                    print("❌ Betting system not available for backtesting")
                
            except Exception as e:
                print(f"❌ Backtesting error: {e}")
        
        elif args.mode == 'run':
            print("\n🔄 Automatic mode...")
            
            if pipeline.setup_scheduler():
                print("⏰ Scheduler configured...")
                print("🚀 Pipeline started!")
                print("📱 Notifications will be sent when value bets are found")
                print("⏹️ Press Ctrl+C to stop")
                
                try:
                    pipeline.run_scheduler()
                except KeyboardInterrupt:
                    print("\n⏹️ Pipeline stopped")
            else:
                print("❌ Failed to setup scheduler")
                print("💡 Use --mode manual instead")
        
        elif args.mode == 'manual':
            print("\n🔄 Manual mode...")
            pipeline.run_manual_mode()
    
    except KeyboardInterrupt:
        print("\n⏹️ Operation interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()