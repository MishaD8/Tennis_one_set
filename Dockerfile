FROM python:3.9-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Установка Python зависимостей
COPY requirements_production.txt .
RUN pip install --no-cache-dir -r requirements_production.txt

# Копирование приложения
COPY . .

# Создание пользователя
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Настройка переменных окружения
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# Открытие порта
EXPOSE 5001

# Здоровье контейнера
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/api/health || exit 1

# Запуск приложения
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "2", "--timeout", "120", "web_backend:app"]
