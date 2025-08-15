# Tennis Prediction System - Production Deployment Guide

> **Status**: Production Ready ✅  
> **Supported Platforms**: Linux, Docker, Cloud  
> **Last Updated**: August 2025

## Overview

This guide provides comprehensive instructions for deploying the Tennis Prediction System to production environments, including local deployment, cloud deployment, and container orchestration.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Local Production Deployment](#local-production-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Deployment (AWS/GCP/Azure)](#cloud-deployment)
6. [Database Setup](#database-setup)
7. [Security Configuration](#security-configuration)
8. [Monitoring & Logging](#monitoring--logging)
9. [CI/CD Pipeline](#cicd-pipeline)
10. [Troubleshooting](#troubleshooting)
11. [Maintenance](#maintenance)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 2 cores
- RAM: 4GB
- Storage: 20GB SSD
- OS: Ubuntu 20.04+ / CentOS 8+ / Docker-compatible environment

**Recommended Requirements:**
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 50GB+ SSD
- OS: Ubuntu 22.04 LTS

### Software Dependencies

- **Python**: 3.8+
- **PostgreSQL**: 13+
- **Redis**: 6+
- **Node.js**: 16+ (for frontend, if applicable)
- **Docker**: 20.10+ (for containerized deployment)
- **Git**: 2.25+

## Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-org/tennis-prediction-system.git
cd tennis-prediction-system
```

### 2. Create Environment Configuration

Create production environment file:

```bash
cp env_example.txt .env.production
```

Edit `.env.production` with production values:

```bash
# Application Configuration
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=your-super-secret-key-here
PORT=5000
HOST=0.0.0.0

# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/tennis_production
REDIS_URL=redis://localhost:6379/0

# API Keys (obtain from respective services)
ODDS_API_KEY=your-odds-api-key
TENNIS_API_KEY=your-tennis-api-key

# Monitoring & Alerting
ALERT_EMAIL_ENABLED=true
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=alerts@yourcompany.com
SMTP_PASSWORD=your-app-password
ALERT_RECIPIENTS=admin@yourcompany.com,ops@yourcompany.com
ALERT_WEBHOOK_URL=https://hooks.slack.com/your-webhook-url

# Security
SSL_ENABLED=true
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
RATE_LIMIT_ENABLED=true

# Performance
CACHE_TTL=3600
MAX_WORKERS=4
REQUEST_TIMEOUT=30
```

### 3. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y python3.8 python3.8-venv python3.8-dev
sudo apt install -y postgresql postgresql-contrib redis-server
sudo apt install -y nginx supervisor
sudo apt install -y build-essential libpq-dev
```

**CentOS/RHEL:**
```bash
sudo yum update -y
sudo yum install -y python38 python38-venv python38-devel
sudo yum install -y postgresql-server postgresql-contrib redis
sudo yum install -y nginx supervisor
sudo yum install -y gcc postgresql-devel
```

## Local Production Deployment

### 1. Database Setup

#### PostgreSQL Installation & Configuration

```bash
# Initialize PostgreSQL (CentOS only)
sudo postgresql-setup initdb
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql
```

```sql
CREATE USER tennis_user WITH ENCRYPTED PASSWORD 'secure_password_here';
CREATE DATABASE tennis_production OWNER tennis_user;
GRANT ALL PRIVILEGES ON DATABASE tennis_production TO tennis_user;
\q
```

#### Redis Setup

```bash
sudo systemctl start redis
sudo systemctl enable redis

# Configure Redis for production
sudo nano /etc/redis/redis.conf
```

Update Redis configuration:
```
maxmemory 1gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
```

### 2. Application Setup

#### Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Install Python Dependencies

```bash
# Install production requirements
pip install --upgrade pip
pip install -r requirements_production.txt

# Install additional production packages
pip install gunicorn psycopg2-binary redis
```

#### Database Migration

```bash
# Set environment variables
export $(cat .env.production | xargs)

# Run database setup
python database_setup.py

# Run migrations (if you have existing data)
python database_migration.py
```

#### Install ML Models

```bash
# Create models directory
mkdir -p tennis_models

# Download or copy your trained models
# These should include:
# - neural_network.h5
# - xgboost.pkl
# - random_forest.pkl
# - gradient_boosting.pkl
# - logistic_regression.pkl
# - scaler.pkl
# - metadata.json
```

### 3. Web Server Configuration

#### Nginx Configuration

Create `/etc/nginx/sites-available/tennis-prediction`:

```nginx
upstream tennis_app {
    server 127.0.0.1:5000;
    server 127.0.0.1:5001;  # If running multiple workers
    server 127.0.0.1:5002;
    server 127.0.0.1:5003;
}

server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    # SSL Configuration
    ssl_certificate /etc/ssl/certs/tennis-prediction.crt;
    ssl_certificate_key /etc/ssl/private/tennis-prediction.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";

    # Gzip compression
    gzip on;
    gzip_types text/plain application/json application/javascript text/css application/xml;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=ml:10m rate=2r/s;

    # Main application
    location / {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://tennis_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }

    # ML prediction endpoints with stricter rate limiting
    location ~ ^/api/(test-ml|underdog-analysis) {
        limit_req zone=ml burst=5 nodelay;
        
        proxy_pass http://tennis_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Static files (if any)
    location /static/ {
        alias /path/to/tennis-prediction-system/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://tennis_app/api/health;
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/tennis-prediction /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 4. Process Management with Supervisor

Create `/etc/supervisor/conf.d/tennis-prediction.conf`:

```ini
[group:tennis-prediction]
programs=tennis-app-5000,tennis-app-5001,tennis-app-5002,tennis-app-5003

[program:tennis-app-5000]
command=/path/to/tennis-prediction-system/venv/bin/gunicorn -w 1 -b 127.0.0.1:5000 --timeout 120 --max-requests 1000 main:app
directory=/path/to/tennis-prediction-system
user=tennis
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/tennis-prediction/app-5000.log
environment=ENVIRONMENT="production"

[program:tennis-app-5001]
command=/path/to/tennis-prediction-system/venv/bin/gunicorn -w 1 -b 127.0.0.1:5001 --timeout 120 --max-requests 1000 main:app
directory=/path/to/tennis-prediction-system
user=tennis
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/tennis-prediction/app-5001.log
environment=ENVIRONMENT="production"

[program:tennis-app-5002]
command=/path/to/tennis-prediction-system/venv/bin/gunicorn -w 1 -b 127.0.0.1:5002 --timeout 120 --max-requests 1000 main:app
directory=/path/to/tennis-prediction-system
user=tennis
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/tennis-prediction/app-5002.log
environment=ENVIRONMENT="production"

[program:tennis-app-5003]
command=/path/to/tennis-prediction-system/venv/bin/gunicorn -w 1 -b 127.0.0.1:5003 --timeout 120 --max-requests 1000 main:app
directory=/path/to/tennis-prediction-system
user=tennis
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/tennis-prediction/app-5003.log
environment=ENVIRONMENT="production"
```

Start services:
```bash
sudo mkdir -p /var/log/tennis-prediction
sudo chown tennis:tennis /var/log/tennis-prediction
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start tennis-prediction:*
```

## Docker Deployment

### 1. Build Production Image

Create `Dockerfile.production`:

```dockerfile
FROM python:3.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        postgresql-client \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash tennis

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements_production.txt .
RUN pip install --no-cache-dir -r requirements_production.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs tennis_models

# Set permissions
RUN chown -R tennis:tennis /app

# Switch to app user
USER tennis

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Start command
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "main:app"]
```

### 2. Docker Compose Configuration

Create `docker-compose.production.yml`:

```yaml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.production
    restart: unless-stopped
    ports:
      - "5000:5000"
    environment:
      - DATABASE_URL=postgresql://tennis_user:${DB_PASSWORD}@db:5432/tennis_production
      - REDIS_URL=redis://redis:6379/0
      - ODDS_API_KEY=${ODDS_API_KEY}
      - ENVIRONMENT=production
    depends_on:
      - db
      - redis
    volumes:
      - ./tennis_models:/app/tennis_models:ro
      - ./logs:/app/logs
    networks:
      - tennis-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:13
    restart: unless-stopped
    environment:
      - POSTGRES_DB=tennis_production
      - POSTGRES_USER=tennis_user
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    ports:
      - "5432:5432"
    networks:
      - tennis-network

  redis:
    image: redis:6-alpine
    restart: unless-stopped
    command: redis-server --appendonly yes --maxmemory 1gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    networks:
      - tennis-network

  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl:ro
    depends_on:
      - app
    networks:
      - tennis-network

volumes:
  postgres_data:
  redis_data:

networks:
  tennis-network:
    driver: bridge
```

### 3. Deploy with Docker Compose

```bash
# Create environment file for Docker
cp .env.production .env

# Build and start services
docker-compose -f docker-compose.production.yml up -d

# Check status
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f app
```

## Cloud Deployment

### AWS Deployment

#### Using AWS ECS

1. **Create ECR Repository:**
```bash
aws ecr create-repository --repository-name tennis-prediction-system
```

2. **Build and Push Image:**
```bash
# Build image
docker build -f Dockerfile.production -t tennis-prediction-system .

# Tag for ECR
docker tag tennis-prediction-system:latest <account-id>.dkr.ecr.<region>.amazonaws.com/tennis-prediction-system:latest

# Push to ECR
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/tennis-prediction-system:latest
```

3. **Create ECS Task Definition:**
```json
{
  "family": "tennis-prediction-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::<account-id>:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "tennis-app",
      "image": "<account-id>.dkr.ecr.<region>.amazonaws.com/tennis-prediction-system:latest",
      "portMappings": [
        {
          "containerPort": 5000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:<region>:<account-id>:secret:tennis-db-credentials"
        },
        {
          "name": "ODDS_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:<region>:<account-id>:secret:tennis-api-keys"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/tennis-prediction",
          "awslogs-region": "<region>",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Using AWS Lambda (Serverless)

Create `serverless.yml`:
```yaml
service: tennis-prediction-api

provider:
  name: aws
  runtime: python3.8
  region: us-east-1
  environment:
    DATABASE_URL: ${ssm:/tennis/database_url}
    ODDS_API_KEY: ${ssm:/tennis/odds_api_key~true}
    REDIS_URL: ${ssm:/tennis/redis_url}

functions:
  app:
    handler: lambda_handler.handler
    events:
      - http:
          path: /{proxy+}
          method: ANY
    timeout: 30
    memorySize: 1024

plugins:
  - serverless-python-requirements
  - serverless-wsgi

custom:
  wsgi:
    app: tennis_backend.app
```

### Google Cloud Platform

#### Using Cloud Run

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/tennis-prediction-system

# Deploy to Cloud Run
gcloud run deploy tennis-prediction-api \
  --image gcr.io/PROJECT_ID/tennis-prediction-system \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --set-env-vars ENVIRONMENT=production \
  --set-env-vars DATABASE_URL="postgresql://..." \
  --set-env-vars REDIS_URL="redis://..."
```

### Azure Deployment

#### Using Container Instances

```bash
# Create resource group
az group create --name tennis-prediction-rg --location eastus

# Create container instance
az container create \
  --resource-group tennis-prediction-rg \
  --name tennis-prediction-app \
  --image tennis-prediction-system:latest \
  --dns-name-label tennis-prediction \
  --ports 5000 \
  --environment-variables ENVIRONMENT=production \
  --secure-environment-variables DATABASE_URL="postgresql://..." ODDS_API_KEY="..."
```

## Security Configuration

### SSL/TLS Setup

#### Let's Encrypt (Free SSL)

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### Firewall Configuration

```bash
# Configure UFW (Ubuntu)
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
sudo ufw enable

# For production servers, also allow only specific IPs for SSH
sudo ufw allow from YOUR_OFFICE_IP to any port 22
```

### Environment Security

```bash
# Set secure file permissions
chmod 600 .env.production
chown tennis:tennis .env.production

# Secure log files
chmod 640 /var/log/tennis-prediction/*.log
chown tennis:adm /var/log/tennis-prediction/*.log
```

## Monitoring & Logging

### Application Monitoring

1. **Health Checks:**
   - Configure load balancer health checks to `/api/health`
   - Set up monitoring alerts for health check failures

2. **Application Metrics:**
   - Monitor prediction success rates
   - Track API response times
   - Monitor memory and CPU usage

3. **Log Management:**
   - Use centralized logging (ELK stack, Splunk, etc.)
   - Configure log rotation
   - Set up log-based alerting

### Example Monitoring Setup

```bash
# Install monitoring tools
sudo apt install prometheus node_exporter grafana

# Configure Prometheus
cat > /etc/prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'tennis-prediction'
    static_configs:
      - targets: ['localhost:5000']
    metrics_path: '/metrics'

  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
EOF

# Start services
sudo systemctl start prometheus node_exporter grafana-server
sudo systemctl enable prometheus node_exporter grafana-server
```

## CI/CD Pipeline

### GitHub Actions Example

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=. --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Deploy to production
      uses: appleboy/ssh-action@v0.1.4
      with:
        host: ${{ secrets.PRODUCTION_HOST }}
        username: ${{ secrets.PRODUCTION_USER }}
        key: ${{ secrets.PRODUCTION_SSH_KEY }}
        script: |
          cd /path/to/tennis-prediction-system
          git pull origin main
          source venv/bin/activate
          pip install -r requirements_production.txt
          sudo supervisorctl restart tennis-prediction:*
          
          # Run health check
          sleep 10
          curl -f http://localhost/api/health || exit 1
```

## Troubleshooting

### Common Issues

1. **Models Not Loading:**
```bash
# Check model files exist and have correct permissions
ls -la tennis_models/
chmod 644 tennis_models/*.pkl tennis_models/*.h5

# Check TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

2. **Database Connection Issues:**
```bash
# Test database connection
python -c "
import psycopg2
conn = psycopg2.connect('postgresql://tennis_user:password@localhost/tennis_production')
print('Database connection successful')
"
```

3. **High Memory Usage:**
```bash
# Monitor memory usage
htop
# or
ps aux --sort=-%mem | head

# Check for memory leaks in application logs
grep -i "memory\|oom" /var/log/tennis-prediction/*.log
```

4. **API Rate Limiting Issues:**
```bash
# Check API usage
curl -s http://localhost:5000/api/api-status | jq .

# Reset rate limits if needed (be careful)
redis-cli FLUSHDB
```

### Log Analysis

```bash
# Application logs
tail -f /var/log/tennis-prediction/app-*.log

# System logs
journalctl -u tennis-prediction -f

# Nginx logs
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log

# Database logs
sudo tail -f /var/log/postgresql/postgresql-13-main.log
```

## Maintenance

### Regular Maintenance Tasks

1. **Daily:**
   - Monitor system health
   - Check error logs
   - Verify API rate limits

2. **Weekly:**
   - Update ML model weights
   - Review prediction accuracy
   - Database maintenance

3. **Monthly:**
   - Security updates
   - Performance optimization
   - Backup verification

### Backup Procedures

```bash
# Database backup
pg_dump -h localhost -U tennis_user tennis_production > backup_$(date +%Y%m%d_%H%M%S).sql

# Application backup
tar -czf tennis_app_backup_$(date +%Y%m%d).tar.gz \
  --exclude='venv' \
  --exclude='logs' \
  --exclude='__pycache__' \
  /path/to/tennis-prediction-system

# Automated backup script
cat > /usr/local/bin/tennis_backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backups/tennis"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Database backup
pg_dump -h localhost -U tennis_user tennis_production | gzip > $BACKUP_DIR/db_$DATE.sql.gz

# Keep only last 30 days of backups
find $BACKUP_DIR -name "db_*.sql.gz" -mtime +30 -delete

echo "Backup completed: $DATE"
EOF

chmod +x /usr/local/bin/tennis_backup.sh

# Add to crontab for daily backups
echo "0 2 * * * /usr/local/bin/tennis_backup.sh" | crontab -
```

### Performance Optimization

1. **Database Optimization:**
```sql
-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_player ON predictions(player_name);

-- Update table statistics
ANALYZE predictions;
ANALYZE betting_records;
```

2. **Application Optimization:**
```bash
# Profile application performance
pip install py-spy
py-spy record -o profile.svg -d 60 -p $(pgrep -f gunicorn)

# Optimize gunicorn configuration
# Edit supervisor config to tune workers
workers = (CPU_cores * 2) + 1
```

### Scaling Considerations

1. **Horizontal Scaling:**
   - Use load balancer (HAProxy, AWS ALB)
   - Deploy multiple application instances
   - Consider microservices architecture

2. **Database Scaling:**
   - Implement read replicas
   - Consider database sharding for large datasets
   - Use connection pooling (PgBouncer)

3. **Caching Strategy:**
   - Implement Redis cluster for high availability
   - Use CDN for static content
   - Cache ML predictions for repeated requests

## Support

For additional support:

1. Check application logs first
2. Review this deployment guide
3. Consult API documentation
4. Contact system administrators

---

**Note:** Always test deployments in a staging environment before deploying to production. Keep backups and have a rollback plan ready.