#!/bin/bash

# 📦 Tennis Prediction System - Backup Script

BACKUP_DIR="/opt/backups/tennis_prediction"
APP_DIR="/opt/tennis_prediction"
DATE=$(date +%Y%m%d_%H%M%S)

echo "🗄️ Creating backup: tennis_backup_$DATE"

# Create backup directory
mkdir -p $BACKUP_DIR

# Create backup
tar -czf $BACKUP_DIR/tennis_backup_$DATE.tar.gz \
    -C $APP_DIR \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='logs/*.log' \
    .

# Keep only last 7 backups
find $BACKUP_DIR -name "tennis_backup_*.tar.gz" -mtime +7 -delete

echo "✅ Backup completed: $BACKUP_DIR/tennis_backup_$DATE.tar.gz"
