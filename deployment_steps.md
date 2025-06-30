# 🎾 Tennis Prediction System - Deployment Steps

## Подготовка к деплою (выполняется локально)

### 1. Создание production файлов

```bash
# Сохраните артефакты в папку проекта
cd tennis-prediction/

# Сохраните deployment_preparation.sh и production_fixes.py
# Затем запустите их:

chmod +x deployment_preparation.sh
bash deployment_preparation.sh

python3 production_fixes.py
```

Это создаст файлы:
- `config_production.json` - production конфиг
- `gunicorn.conf.py` - WSGI сервер конфиг
- `health_monitor.py` - скрипт мониторинга
- `performance_optimizer.py` - оптимизатор
- `requirements_fixes.txt` - дополнительные зависимости

### 2. Обновление web_backend.py

Добавьте в начало `web_backend.py` (после импортов):

```python
import os
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache

# Load environment variables
load_dotenv()

# Update Config class (замените существующий)
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'tennis-prediction-secret-key-2024'
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///tennis_data.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Ports and hosts  
    HOST = os.environ.get('FLASK_HOST') or '0.0.0.0'
    PORT = int(os.environ.get('FLASK_PORT') or 5001)
    DEBUG = os.environ.get('FLASK_DEBUG') == 'True'
    
    # Security
    WTF_CSRF_ENABLED = True
    
    # Caching
    CACHE_TYPE = os.environ.get('CACHE_TYPE') or 'simple'
    CACHE_DEFAULT_TIMEOUT = int(os.environ.get('CACHE_TIMEOUT') or 300)
    
    # Paths (замените существующие)
    DATA_DIR = os.path.join(os.getcwd(), 'tennis_data_enhanced')
    MODELS_DIR = os.path.join(os.getcwd(), 'tennis_models')
    BETTING_DIR = os.path.join(os.getcwd(), 'betting_data')

# Добавьте после создания app
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per minute"]
)

cache = Cache(app)

# Добавьте rate limiting к API endpoints (замените существующие декораторы)
@app.route('/api/matches')
@limiter.limit("10 per minute")
def get_matches():
    # existing code...

# Обновите error handler (замените существующий)
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}", exc_info=True)
    
    if app.config['DEBUG']:
        return jsonify({'success': False, 'error': str(e)}), 500
    else:
        return jsonify({'success': False, 'error': 'Internal server error'}), 500
```

### 3. Обновление requirements

Объедините все requirements:

```bash
cat requirements.txt requirements_production.txt requirements_fixes.txt > requirements_final.txt
```

### 4. Структура файлов проекта

После подготовки у вас должна быть такая структура:

```
tennis-prediction/
├── api_integretion.py
├── config.json
├── requirements.txt
├── script_data_collector.py
├── startup.py
├── tennis_betting_pipeline.py
├── tennis_set_predictor.py
├── tennis_system_odds.py
├── web_backend.py (обновленный!)
├── web_dashboard.html
│
├── Production files (созданные скриптами):
├── config_production.json
├── .env.production
├── gunicorn.conf.py
├── health_monitor.py
├── performance_optimizer.py
├── tennis-prediction.service
├── tennis_nginx.conf
├── deploy.sh
├── backup.sh
├── monitor.sh
├── Dockerfile
├── docker-compose.yml
└── requirements_final.txt
```

## Deployment на Hetzner VPS

### 1. Создание VPS
- Зайдите в Hetzner Cloud Console
- Создайте VPS CPX21 (или CPX31)
- Выберите Ubuntu 22.04 LTS
- Добавьте SSH ключ

### 2. Загрузка файлов на сервер

```bash
# Локально - упакуйте проект
tar -czf tennis-prediction.tar.gz tennis-prediction/

# Загрузите на сервер
scp tennis-prediction.tar.gz root@YOUR_SERVER_IP:/tmp/

# Подключитесь к серверу
ssh root@YOUR_SERVER_IP

# Распакуйте
cd /tmp
tar -xzf tennis-prediction.tar.gz
```

### 3. Настройка сервера

```bash
# На сервере
cd /tmp/tennis-prediction

# Сделайте deploy.sh исполняемым
chmod +x deploy.sh

# Запустите deployment
./deploy.sh
```

### 4. Настройка environment

```bash
# Отредактируйте production config
cd /opt/tennis_prediction
nano .env.production

# Добавьте ваши API ключи:
PINNACLE_USERNAME=your_username
PINNACLE_PASSWORD=your_password
ODDS_API_KEY=your_api_key
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
SECRET_KEY=your-very-secret-key-change-this

# Сохраните и выйдите (Ctrl+X, Y, Enter)
```

### 5. Настройка домена

```bash
# Отредактируйте nginx конфиг
nano /etc/nginx/sites-available/tennis_prediction

# Замените "your-domain.com" на ваш реальный домен
# Сохраните и перезапустите nginx
systemctl restart nginx
```

### 6. SSL сертификат

```bash
# Установите Let's Encrypt
apt install certbot python3-certbot-nginx

# Получите SSL сертификат
certbot --nginx -d your-domain.com
```

### 7. Проверка работы

```bash
# Проверьте статус сервиса
systemctl status tennis-prediction

# Проверьте health check
curl http://localhost:5001/api/health

# Проверьте через nginx
curl https://your-domain.com/api/health
```

## Финальная проверка

После deployment система должна быть доступна по адресу:
- **Dashboard:** https://your-domain.com
- **API:** https://your-domain.com/api/matches
- **Health:** https://your-domain.com/api/health

## Мониторинг

Добавьте мониторинг в crontab:

```bash
crontab -e

# Добавьте строки:
*/5 * * * * /opt/tennis_prediction/monitor.sh
0 2 * * * /opt/tennis_prediction/backup.sh
```

## Логи

```bash
# Логи приложения
journalctl -u tennis-prediction -f

# Логи nginx
tail -f /var/log/nginx/tennis_access.log

# Логи системы
tail -f /opt/tennis_prediction/logs/tennis_system.log
```
