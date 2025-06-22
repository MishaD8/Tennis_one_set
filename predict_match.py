import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class TennisSetPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = []
        
    def prepare_features(self, df):
        """
        Подготовка признаков для модели
        """
        features = []
        
        # Базовые статистики игрока
        player_stats = [
            'player_age', 'player_rank', 'player_points',
            'recent_matches_count', 'wins_last_10', 'losses_last_10'
        ]
        
        # Статистика по сетам
        set_stats = [
            'sets_won_ratio', 'sets_lost_ratio', 'tiebreak_win_ratio',
            'sets_won_on_surface', 'sets_lost_on_surface'
        ]
        
        # Статистика подач
        serve_stats = [
            'first_serve_pct', 'first_serve_won_pct', 
            'second_serve_won_pct', 'break_points_saved_pct'
        ]
        
        # Статистика приема
        return_stats = [
            'first_serve_return_won_pct', 'second_serve_return_won_pct',
            'break_points_converted_pct'
        ]
        
        # Противник
        opponent_stats = [
            'opponent_age', 'opponent_rank', 'opponent_points',
            'opponent_recent_form', 'h2h_wins', 'h2h_losses'
        ]
        
        # Контекст матча
        match_context = [
            'tournament_level', 'surface_type', 'match_round',
            'best_of_sets', 'is_home'
        ]
        
        self.feature_columns = (player_stats + set_stats + serve_stats + 
                               return_stats + opponent_stats + match_context)
        
        return df[self.feature_columns]
    
    def create_neural_network(self, input_dim):
        """
        Создание нейронной сети
        """
        model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            Dropout(0.2),
            
            Dense(1, activation='sigmoid')  # Вероятность взять хотя бы 1 сет
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """
        Обучение модели
        """
        # Нормализация данных
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Создание модели
        self.model = self.create_neural_network(X_train_scaled.shape[1])
        
        # Обучение
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=100,
            batch_size=32,
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=15, 
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', 
                    factor=0.5, 
                    patience=10
                )
            ]
        )
        
        return history
    
    def predict_probability(self, X):
        """
        Предсказание вероятности взять хотя бы один сет
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate_model(self, X_test, y_test):
        """
        Оценка модели
        """
        X_test_scaled = self.scaler.transform(X_test)
        
        # Предсказания
        y_pred_proba = self.model.predict(X_test_scaled)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Метрики
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return y_pred_proba, y_pred

def create_sample_data():
    """
    Создание примера данных для демонстрации
    """
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        # Базовые статистики игрока
        'player_age': np.random.normal(26, 4, n_samples),
        'player_rank': np.random.exponential(50, n_samples),
        'player_points': np.random.normal(1000, 800, n_samples),
        'recent_matches_count': np.random.poisson(8, n_samples),
        'wins_last_10': np.random.binomial(10, 0.6, n_samples),
        'losses_last_10': 10 - np.random.binomial(10, 0.6, n_samples),
        
        # Статистика по сетам
        'sets_won_ratio': np.random.beta(2, 2, n_samples),
        'sets_lost_ratio': np.random.beta(2, 2, n_samples),
        'tiebreak_win_ratio': np.random.beta(1.5, 1.5, n_samples),
        'sets_won_on_surface': np.random.poisson(15, n_samples),
        'sets_lost_on_surface': np.random.poisson(12, n_samples),
        
        # Статистика подач
        'first_serve_pct': np.random.beta(6, 4, n_samples),
        'first_serve_won_pct': np.random.beta(7, 3, n_samples),
        'second_serve_won_pct': np.random.beta(4, 6, n_samples),
        'break_points_saved_pct': np.random.beta(3, 3, n_samples),
        
        # Статистика приема
        'first_serve_return_won_pct': np.random.beta(3, 7, n_samples),
        'second_serve_return_won_pct': np.random.beta(5, 5, n_samples),
        'break_points_converted_pct': np.random.beta(2, 3, n_samples),
        
        # Противник
        'opponent_age': np.random.normal(26, 4, n_samples),
        'opponent_rank': np.random.exponential(50, n_samples),
        'opponent_points': np.random.normal(1000, 800, n_samples),
        'opponent_recent_form': np.random.beta(2, 2, n_samples),
        'h2h_wins': np.random.poisson(2, n_samples),
        'h2h_losses': np.random.poisson(2, n_samples),
        
        # Контекст матча
        'tournament_level': np.random.choice([1, 2, 3, 4], n_samples),
        'surface_type': np.random.choice([0, 1, 2], n_samples),  # 0-хард, 1-грунт, 2-трава
        'match_round': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'best_of_sets': np.random.choice([3, 5], n_samples),
        'is_home': np.random.choice([0, 1], n_samples)
    }
    
    # Создание целевой переменной (взял ли хотя бы один сет)
    # Логика: более сильные игроки чаще берут сеты
    strength_factor = (
        (1 / (data['player_rank'] + 1)) * 0.3 +
        data['sets_won_ratio'] * 0.25 +
        data['first_serve_won_pct'] * 0.2 +
        data['wins_last_10'] / 10 * 0.15 +
        (1 / (data['opponent_rank'] + 1)) * 0.1
    )
    
    # Добавляем случайность
    noise = np.random.normal(0, 0.1, n_samples)
    probability = np.clip(strength_factor + noise, 0, 1)
    
    data['won_at_least_one_set'] = np.random.binomial(1, probability, n_samples)
    
    return pd.DataFrame(data)

# Пример использования
if __name__ == "__main__":
    print("Создание тестовых данных...")
    df = create_sample_data()
    
    print(f"Размер датасета: {df.shape}")
    print(f"Баланс классов: {df['won_at_least_one_set'].value_counts()}")
    
    # Инициализация предиктора
    predictor = TennisSetPredictor()
    
    # Подготовка данных
    X = predictor.prepare_features(df)
    y = df['won_at_least_one_set']
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print("\nОбучение модели...")
    history = predictor.train_model(X_train, y_train, X_val, y_val)
    
    print("\nОценка модели на тестовых данных:")
    y_pred_proba, y_pred = predictor.evaluate_model(X_test, y_test)
    
    print(f"\nПримеры предсказаний (вероятность взять хотя бы один сет):")
    for i in range(5):
        print(f"Игрок {i+1}: {y_pred_proba[i][0]:.3f} (факт: {y_test.iloc[i]})")