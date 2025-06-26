import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class EnhancedTennisPredictor:
    def __init__(self, model_dir="tennis_models"):
        self.model_dir = model_dir
        self.scaler = RobustScaler()  # Более устойчив к выбросам
        self.models = {}
        self.feature_importance = {}
        self.feature_columns = []
        self.ensemble_weights = {}
        self.validation_scores = {}
        
        # Создаем директорию для моделей
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        УЛУЧШЕНО: Подготовка признаков с feature engineering
        """
        features = df.copy()
        
        # Базовые признаки игрока
        player_basic = [
            'player_rank', 'player_age', 'opponent_rank', 'opponent_age'
        ]
        
        # Форма и статистика
        form_features = [
            'player_recent_matches_count', 'player_recent_win_rate', 
            'player_recent_sets_win_rate', 'player_form_trend',
            'player_days_since_last_match'
        ]
        
        # Покрытие
        surface_features = [
            'player_surface_matches_count', 'player_surface_win_rate',
            'player_surface_advantage', 'player_surface_sets_rate', 
            'player_surface_experience'
        ]
        
        # Очные встречи
        h2h_features = [
            'h2h_matches', 'h2h_win_rate', 'h2h_recent_form',
            'h2h_sets_advantage', 'days_since_last_h2h'
        ]
        
        # Давление турнира
        pressure_features = [
            'tournament_importance', 'round_pressure', 'total_pressure',
            'is_high_pressure_tournament'
        ]
        
        self.feature_columns = (player_basic + form_features + surface_features + 
                               h2h_features + pressure_features)
        
        # Проверяем наличие признаков
        available_features = [col for col in self.feature_columns if col in features.columns]
        missing_features = [col for col in self.feature_columns if col not in features.columns]
        
        if missing_features:
            print(f"⚠️ Отсутствующие признаки: {missing_features}")
        
        # Создаем дополнительные признаки
        features_enhanced = features[available_features].copy()
        
        # Feature engineering - создаем новые признаки
        if 'player_rank' in features_enhanced.columns and 'opponent_rank' in features_enhanced.columns:
            # Относительная сила игроков
            features_enhanced['rank_difference'] = features_enhanced['opponent_rank'] - features_enhanced['player_rank']
            features_enhanced['rank_ratio'] = features_enhanced['player_rank'] / (features_enhanced['opponent_rank'] + 1)
            
        if 'player_recent_win_rate' in features_enhanced.columns and 'h2h_win_rate' in features_enhanced.columns:
            # Комбинированная форма
            features_enhanced['combined_form'] = (features_enhanced['player_recent_win_rate'] * 0.7 + 
                                                features_enhanced['h2h_win_rate'] * 0.3)
        
        if 'player_surface_advantage' in features_enhanced.columns and 'total_pressure' in features_enhanced.columns:
            # Адаптация к давлению на покрытии
            features_enhanced['surface_pressure_interaction'] = (features_enhanced['player_surface_advantage'] * 
                                                               features_enhanced['total_pressure'])
        
        # Заполняем пропуски медианными значениями
        features_enhanced = features_enhanced.fillna(features_enhanced.median())
        
        return features_enhanced
    
    def create_neural_network(self, input_dim: int) -> keras.Model:
        """
        УЛУЧШЕНО: Создание более сложной нейронной сети
        """
        model = Sequential([
            # Входной слой с нормализацией
            Dense(256, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            
            # Скрытые слои
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.25),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.15),
            
            Dense(16, activation='relu'),
            Dropout(0.1),
            
            # Выходной слой
            Dense(1, activation='sigmoid')
        ])
        
        # Оптимизатор с адаптивным learning rate
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'AUC']
        )
        
        return model
    
    def train_ensemble_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                        X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """
        НОВОЕ: Обучение ансамбля моделей
        """
        print("🧠 Обучение ансамбля моделей...")
        
        # Нормализация данных
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        models_performance = {}
        
        # 1. Нейронная сеть
        print("🔸 Обучение нейронной сети...")
        nn_model = self.create_neural_network(X_train_scaled.shape[1])
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
        ]
        
        history = nn_model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=150,
            batch_size=64,
            verbose=0,
            callbacks=callbacks
        )
        
        nn_pred = nn_model.predict(X_val_scaled).flatten()
        nn_auc = roc_auc_score(y_val, nn_pred)
        models_performance['neural_network'] = nn_auc
        self.models['neural_network'] = nn_model
        
        print(f"✅ Нейронная сеть: AUC = {nn_auc:.4f}")
        
        # 2. XGBoost
        print("🔸 Обучение XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='auc'
        )
        
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
        xgb_auc = roc_auc_score(y_val, xgb_pred)
        models_performance['xgboost'] = xgb_auc
        self.models['xgboost'] = xgb_model
        
        # Сохраняем важность признаков
        self.feature_importance['xgboost'] = dict(zip(X_train.columns, xgb_model.feature_importances_))
        
        print(f"✅ XGBoost: AUC = {xgb_auc:.4f}")
        
        # 3. Random Forest
        print("🔸 Обучение Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict_proba(X_val)[:, 1]
        rf_auc = roc_auc_score(y_val, rf_pred)
        models_performance['random_forest'] = rf_auc
        self.models['random_forest'] = rf_model
        
        # Сохраняем важность признаков
        self.feature_importance['random_forest'] = dict(zip(X_train.columns, rf_model.feature_importances_))
        
        print(f"✅ Random Forest: AUC = {rf_auc:.4f}")
        
        # 4. Градиентный бустинг
        print("🔸 Обучение Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
        
        gb_model.fit(X_train, y_train)
        gb_pred = gb_model.predict_proba(X_val)[:, 1]
        gb_auc = roc_auc_score(y_val, gb_pred)
        models_performance['gradient_boosting'] = gb_auc
        self.models['gradient_boosting'] = gb_model
        
        print(f"✅ Gradient Boosting: AUC = {gb_auc:.4f}")
        
        # 5. Логистическая регрессия (как baseline)
        print("🔸 Обучение Logistic Regression...")
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict_proba(X_val_scaled)[:, 1]
        lr_auc = roc_auc_score(y_val, lr_pred)
        models_performance['logistic_regression'] = lr_auc
        self.models['logistic_regression'] = lr_model
        
        print(f"✅ Logistic Regression: AUC = {lr_auc:.4f}")
        
        # Рассчитываем веса для ансамбля на основе производительности
        total_performance = sum(models_performance.values())
        self.ensemble_weights = {
            model: performance / total_performance 
            for model, performance in models_performance.items()
        }
        
        print(f"\n📊 Веса ансамбля:")
        for model, weight in self.ensemble_weights.items():
            print(f"• {model}: {weight:.3f}")
        
        self.validation_scores = models_performance
        
        return models_performance
    
    def predict_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        НОВОЕ: Предсказание с использованием ансамбля
        """
        X_features = self.prepare_features(X)
        X_scaled = self.scaler.transform(X_features)
        
        predictions = []
        weights = []
        
        # Получаем предсказания от каждой модели
        for model_name, model in self.models.items():
            if model_name == 'neural_network' or model_name == 'logistic_regression':
                pred = model.predict(X_scaled).flatten()
            else:
                pred = model.predict_proba(X_features)[:, 1]
            
            predictions.append(pred)
            weights.append(self.ensemble_weights[model_name])
        
        # Взвешенное усреднение
        ensemble_prediction = np.average(predictions, weights=weights, axis=0)
        
        return ensemble_prediction
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        НОВОЕ: Подробная оценка всех моделей
        """
        print("📊 Оценка моделей на тестовых данных...")
        
        X_test_features = self.prepare_features(X_test)
        X_test_scaled = self.scaler.transform(X_test_features)
        
        results = {}
        
        # Оценка каждой модели
        for model_name, model in self.models.items():
            if model_name == 'neural_network' or model_name == 'logistic_regression':
                y_pred_proba = model.predict(X_test_scaled).flatten()
            else:
                y_pred_proba = model.predict_proba(X_test_features)[:, 1]
            
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            auc_score = roc_auc_score(y_test, y_pred_proba)
            accuracy = (y_pred == y_test).mean()
            
            results[model_name] = {
                'auc': auc_score,
                'accuracy': accuracy,
                'predictions': y_pred_proba
            }
        
        # Оценка ансамбля
        ensemble_pred = self.predict_probability(X_test)
        ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)
        
        ensemble_auc = roc_auc_score(y_test, ensemble_pred)
        ensemble_accuracy = (ensemble_pred_binary == y_test).mean()
        
        results['ensemble'] = {
            'auc': ensemble_auc,
            'accuracy': ensemble_accuracy,
            'predictions': ensemble_pred
        }
        
        # Выводим результаты
        print(f"\n📈 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
        print("=" * 50)
        for model_name, metrics in results.items():
            print(f"{model_name:20}: AUC = {metrics['auc']:.4f}, Accuracy = {metrics['accuracy']:.4f}")
        
        return results
    
    def plot_feature_importance(self, top_n: int = 20):
        """
        НОВОЕ: Визуализация важности признаков
        """
        if not self.feature_importance:
            print("❌ Нет данных о важности признаков")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        axes = axes.ravel()
        
        plot_idx = 0
        for model_name, importance in self.feature_importance.items():
            if plot_idx >= 4:
                break
                
            # Сортируем по важности
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
            features, values = zip(*sorted_features)
            
            axes[plot_idx].barh(range(len(features)), values)
            axes[plot_idx].set_yticks(range(len(features)))
            axes[plot_idx].set_yticklabels(features)
            axes[plot_idx].set_title(f'Feature Importance - {model_name}')
            axes[plot_idx].set_xlabel('Importance')
            
            plot_idx += 1
        
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison(self, results: Dict):
        """
        НОВОЕ: Сравнение производительности моделей
        """
        models = list(results.keys())
        auc_scores = [results[model]['auc'] for model in models]
        accuracy_scores = [results[model]['accuracy'] for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # AUC scores
        bars1 = ax1.bar(models, auc_scores, color='skyblue', alpha=0.7)
        ax1.set_title('Model Comparison - AUC Score')
        ax1.set_ylabel('AUC Score')
        ax1.set_ylim(0.5, 1.0)
        ax1.tick_params(axis='x', rotation=45)
        
        # Добавляем значения на столбцы
        for bar, score in zip(bars1, auc_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Accuracy scores
        bars2 = ax2.bar(models, accuracy_scores, color='lightcoral', alpha=0.7)
        ax2.set_title('Model Comparison - Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0.5, 1.0)
        ax2.tick_params(axis='x', rotation=45)
        
        # Добавляем значения на столбцы
        for bar, score in zip(bars2, accuracy_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def save_models(self):
        """
        НОВОЕ: Сохранение всех моделей
        """
        print("💾 Сохранение моделей...")
        
        # Сохраняем скалер
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.pkl'))
        
        # Сохраняем каждую модель
        for model_name, model in self.models.items():
            if model_name == 'neural_network':
                model.save(os.path.join(self.model_dir, f'{model_name}.h5'))
            else:
                joblib.dump(model, os.path.join(self.model_dir, f'{model_name}.pkl'))
        
        # Сохраняем метаданные
        metadata = {
            'feature_columns': self.feature_columns,
            'ensemble_weights': self.ensemble_weights,
            'validation_scores': self.validation_scores,
            'feature_importance': self.feature_importance
        }
        
        import json
        with open(os.path.join(self.model_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"✅ Все модели сохранены в: {self.model_dir}")
    
    def load_models(self):
        """
        НОВОЕ: Загрузка сохраненных моделей
        """
        print("📂 Загрузка моделей...")
        
        # Загружаем скалер
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
        
        # Загружаем метаданные
        import json
        with open(os.path.join(self.model_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        self.feature_columns = metadata['feature_columns']
        self.ensemble_weights = metadata['ensemble_weights']
        self.validation_scores = metadata['validation_scores']
        self.feature_importance = metadata['feature_importance']
        
        # Загружаем модели
        for model_name in self.ensemble_weights.keys():
            if model_name == 'neural_network':
                self.models[model_name] = keras.models.load_model(
                    os.path.join(self.model_dir, f'{model_name}.h5')
                )
            else:
                self.models[model_name] = joblib.load(
                    os.path.join(self.model_dir, f'{model_name}.pkl')
                )
        
        print(f"✅ Модели загружены из: {self.model_dir}")

def time_series_split_validation(df: pd.DataFrame, predictor: EnhancedTennisPredictor, 
                                n_splits: int = 5) -> Dict:
    """
    НОВОЕ: Валидация с учетом временной структуры данных
    """
    print("⏰ Временная валидация моделей...")
    
    # Сортируем по дате
    df_sorted = df.sort_values('match_date').copy()
    
    # Подготавливаем данные
    X = predictor.prepare_features(df_sorted)
    y = df_sorted['won_at_least_one_set']
    
    # Временное разбиение
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"📊 Фолд {fold + 1}/{n_splits}")
        
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        
        # Создаем новый предиктор для каждого фолда
        fold_predictor = EnhancedTennisPredictor()
        
        # Обучаем на тренировочных данных
        fold_predictor.train_ensemble_models(X_train_cv, y_train_cv, X_val_cv, y_val_cv)
        
        # Оцениваем на валидационных данных
        ensemble_pred = fold_predictor.predict_probability(X_val_cv)
        fold_auc = roc_auc_score(y_val_cv, ensemble_pred)
        
        cv_scores.append(fold_auc)
        print(f"✅ Фолд {fold + 1} AUC: {fold_auc:.4f}")
    
    print(f"\n📈 Средний AUC по фолдам: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    return {
        'mean_auc': np.mean(cv_scores),
        'std_auc': np.std(cv_scores),
        'fold_scores': cv_scores
    }

def main():
    """
    ПРИМЕР ИСПОЛЬЗОВАНИЯ: Обучение улучшенной модели
    """
    print("🎾 УЛУЧШЕННАЯ СИСТЕМА ПРОГНОЗИРОВАНИЯ ТЕННИСНЫХ МАТЧЕЙ")
    print("=" * 70)
    print("🚀 Возможности:")
    print("• Ансамбль из 5 моделей (NN, XGBoost, RF, GB, LR)")
    print("• Улучшенная feature engineering")
    print("• Временная валидация")
    print("• Анализ важности признаков")
    print("=" * 70)
    
    # Пример с синтетическими данными (замените на ваши реальные данные)
    print("\n📊 Загрузка данных...")
    
    # Здесь должна быть загрузка ваших реальных данных
    # df = pd.read_csv('enhanced_tennis_dataset.csv')
    
    # Создаем пример данных для демонстрации
    np.random.seed(42)
    n_samples = 5000
    
    # Генерируем синтетические данные с улучшенными признаками
    data = {
        'player_rank': np.random.exponential(50, n_samples),
        'opponent_rank': np.random.exponential(50, n_samples),
        'player_age': np.random.normal(26, 4, n_samples),
        'opponent_age': np.random.normal(26, 4, n_samples),
        'player_recent_win_rate': np.random.beta(2, 2, n_samples),
        'player_surface_advantage': np.random.normal(0, 0.1, n_samples),
        'h2h_win_rate': np.random.beta(2, 2, n_samples),
        'total_pressure': np.random.uniform(0, 4, n_samples),
        'player_form_trend': np.random.normal(0, 0.2, n_samples),
        'match_date': pd.date_range('2020-01-01', '2024-12-31', periods=n_samples)
    }
    
    # Создаем целевую переменную с более реалистичной логикой
    strength_factor = (
        (1 / (data['player_rank'] + 1)) * 0.3 +
        data['player_recent_win_rate'] * 0.25 +
        data['player_surface_advantage'] * 0.15 +
        data['h2h_win_rate'] * 0.2 +
        data['player_form_trend'] * 0.1
    )
    
    data['won_at_least_one_set'] = np.random.binomial(1, np.clip(strength_factor, 0.1, 0.9), n_samples)
    
    df = pd.DataFrame(data)
    
    print(f"✅ Загружен датасет: {len(df)} записей")
    print(f"📊 Баланс классов: {df['won_at_least_one_set'].value_counts().to_dict()}")
    
    # Разделение данных с учетом времени
    cutoff_date = df['match_date'].quantile(0.8)
    train_df = df[df['match_date'] < cutoff_date]
    test_df = df[df['match_date'] >= cutoff_date]
    
    print(f"📅 Обучающая выборка: {len(train_df)} записей")
    print(f"📅 Тестовая выборка: {len(test_df)} записей")
    
    # Инициализация предиктора
    predictor = EnhancedTennisPredictor()
    
    # Подготовка данных для обучения
    X_train = predictor.prepare_features(train_df)
    y_train = train_df['won_at_least_one_set']
    
    X_test = predictor.prepare_features(test_df)
    y_test = test_df['won_at_least_one_set']
    
    # Разделение тренировочных данных на train/validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Обучение ансамбля моделей
    print("\n🏋️‍♂️ Обучение ансамбля моделей...")
    performance = predictor.train_ensemble_models(X_train_split, y_train_split, X_val_split, y_val_split)
    
    # Оценка на тестовых данных
    print("\n📊 Оценка на тестовых данных...")
    test_results = predictor.evaluate_models(X_test, y_test)
    
    # Визуализация результатов
    print("\n📈 Создание визуализаций...")
    predictor.plot_model_comparison(test_results)
    predictor.plot_feature_importance()
    
    # Временная валидация
    print("\n⏰ Временная кросс-валидация...")
    cv_results = time_series_split_validation(train_df, predictor, n_splits=3)
    
    # Сохранение моделей
    predictor.save_models()
    
    print("\n🎯 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print("=" * 50)
    best_model = max(test_results.keys(), key=lambda x: test_results[x]['auc'])
    print(f"🏆 Лучшая модель: {best_model}")
    print(f"📊 AUC: {test_results[best_model]['auc']:.4f}")
    print(f"🎯 Точность: {test_results[best_model]['accuracy']:.4f}")
    print(f"⏰ CV AUC: {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")
    
    print(f"\n💾 Модели сохранены в: {predictor.model_dir}")
    print("🚀 Готово к использованию в боевых условиях!")

if __name__ == "__main__":
    main()