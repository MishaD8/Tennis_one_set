# Tennis_one_set - Quick Start Guide
**Last Updated:** October 1, 2025

---

## ✅ Current Status

- **Container:** Healthy and running
- **Telegram:** Working (tested)
- **API Endpoint:** http://localhost:5001
- **Data Collection:** ❌ Not working (API subscription needed)

---

## Quick Health Check

```bash
# Check container status
docker ps | grep tennis

# Test health endpoint
curl http://localhost:5001/api/health | jq '.status'

# Test Telegram notifications
python3 test_telegram_manual.py
```

Expected Output:
```
STATUS: healthy
TELEGRAM: ✅ Success
```

---

## Common Commands

### Start/Stop System
```bash
# Start everything
docker-compose up -d

# Stop everything
docker-compose down

# Restart just the app
docker-compose restart tennis-app

# View logs (live)
docker logs -f tennis_prediction

# View last 100 lines
docker logs tennis_prediction --tail 100
```

### Check Statistics
```bash
# Get system stats
curl http://localhost:5001/api/stats | jq

# Check service status
curl http://localhost:5001/api/service-status | jq
```

### Redis Management
```bash
# Check Redis connection
docker exec tennis_redis redis-cli ping

# View cache keys
docker exec tennis_redis redis-cli KEYS "*"

# Clear cache (if needed)
docker exec tennis_redis redis-cli FLUSHALL
```

---

## Project Structure

```
Tennis_one_set/
├── src/
│   ├── api/          # Flask routes and API logic
│   ├── data/         # Data collection modules
│   ├── ml/           # Machine learning models
│   ├── models/       # Prediction services
│   ├── utils/        # Utilities (Telegram, etc.)
│   └── config/       # Configuration
├── tennis_models/    # Trained ML models (.pkl files)
├── cache/           # API response cache
├── logs/            # Application logs
├── main.py          # Application entry point
├── docker-compose.yml
└── Dockerfile
```

---

## How the System Works

### Normal Flow (When API Active)

1. **Data Collection** (every 30 minutes)
   - `main.py` → `_get_current_matches()`
   - Reads from `cache/api_tennis/`
   - Filters for ATP/WTA singles, ranks 10-300

2. **Prediction**
   - Loads ML models from `tennis_models/`
   - Generates underdog probability
   - Calculates confidence score

3. **Notification**
   - If probability > 0.55 AND confidence = High/Medium
   - Sends Telegram message via `TelegramNotificationSystem`
   - Marks match as processed (no duplicates)

### Current Flow (No API Subscription)

1. **Data Collection** → ❌ No new data
2. **Prediction** → ⏸️ Skipped (no matches)
3. **Notification** → ⏸️ Never triggered

---

## Fixing Data Collection

### Option 1: Renew API-Tennis Subscription

```bash
# 1. Get new API key from api-tennis.com
# 2. Update environment variable
nano .env.production

# 3. Change this line:
API_TENNIS_KEY=your_new_key_here

# 4. Restart container
docker-compose restart tennis-app

# 5. Verify
docker logs tennis_prediction | grep "Analyzing.*matches"
```

### Option 2: Use Test Data (Development)

Create `src/data/test_data_provider.py`:
```python
def get_test_matches():
    return [
        {
            'player1': 'A. Rublev',
            'player2': 'J. Sinner',
            'player1_rank': 8,
            'player2_rank': 4,
            'tournament': 'ATP Vienna',
            'surface': 'Hard'
        }
    ]
```

Modify `main.py:337` to use test data:
```python
def _get_current_matches(self):
    if os.getenv('USE_TEST_DATA') == 'true':
        from src.data.test_data_provider import get_test_matches
        return get_test_matches()
    # ... existing code
```

---

## Monitoring & Troubleshooting

### Container Unhealthy
```bash
# Check logs for errors
docker logs tennis_prediction --tail 50

# Common issues:
# - Port 5001 already in use → change port in docker-compose.yml
# - Redis not connected → check tennis_redis container
# - Python errors → check dependencies
```

### No Notifications
```bash
# 1. Check Telegram config
docker exec tennis_prediction env | grep TELEGRAM

# 2. Test manually
docker exec tennis_prediction python test_telegram_manual.py

# 3. Check if matches are being processed
docker logs tennis_prediction | grep "matches processed"
```

### High Memory Usage
```bash
# Check resource usage
docker stats tennis_prediction

# If high:
# - ML models loaded in memory (~1-2 GB normal)
# - TensorFlow can use a lot
# - Consider switching to tensorflow-lite
```

### API Rate Limiting
```bash
# Check Redis rate limit status
docker exec tennis_redis redis-cli GET "rate-limit:127.0.0.1"

# Clear rate limits (if needed)
docker exec tennis_redis redis-cli FLUSHDB
```

---

## Environment Variables

### Required
```bash
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_IDS=your_chat_id
```

### Optional (for data collection)
```bash
API_TENNIS_KEY=your_api_key
RAPIDAPI_KEY=your_rapidapi_key
```

### System Settings
```bash
FLASK_ENV=production
REDIS_URL=redis://tennis_redis:6379/0
LOG_LEVEL=INFO
```

---

## Performance Optimization

### Reduce Docker Image Size

Current: **4.26 GB**
Target: **~1.5 GB**

```dockerfile
# Use multi-stage build
FROM python:3.9-slim as builder
RUN pip install --user -r requirements.txt

FROM python:3.9-slim
COPY --from=builder /root/.local /root/.local
```

### Implement Log Rotation

Add to `main.py`:
```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    'tennis_integrated_system.log',
    maxBytes=10*1024*1024,  # 10 MB
    backupCount=5
)
logging.basicConfig(handlers=[handler])
```

### Cache Player Rankings

Add to `main.py`:
```python
@lru_cache(maxsize=1)
def _load_player_rankings_cached(self):
    # Cache rankings for 1 hour
    ...
```

---

## Security Checklist

- [x] `.env.production` in `.gitignore`
- [x] Secrets never committed to git
- [x] Rate limiting enabled
- [x] CORS restricted to localhost
- [ ] TODO: SSL certificates for production
- [ ] TODO: API key rotation schedule

---

## Backup & Recovery

### Backup Important Data
```bash
# Backup ML models
tar -czf tennis_models_backup.tar.gz tennis_models/

# Backup cache
tar -czf cache_backup.tar.gz cache/

# Backup configuration
cp .env.production .env.production.backup
```

### Restore from Backup
```bash
# Restore models
tar -xzf tennis_models_backup.tar.gz

# Rebuild container
docker-compose down
docker-compose up -d --build
```

---

## Development Workflow

### Making Changes

```bash
# 1. Edit files locally
nano src/api/routes.py

# 2. Restart container (code is mounted)
docker-compose restart tennis-app

# 3. Check logs
docker logs -f tennis_prediction

# 4. Test changes
curl http://localhost:5001/api/health
```

### Adding New Dependencies

```bash
# 1. Add to requirements.txt
echo "new-package>=1.0.0" >> requirements.txt

# 2. Rebuild image
docker-compose down
docker-compose up -d --build

# 3. Verify
docker exec tennis_prediction pip list | grep new-package
```

---

## API Endpoints

### Health & Status
- `GET /api/health` - Comprehensive health check
- `GET /api/stats` - System statistics
- `GET /api/service-status` - Background service status

### Matches & Predictions
- `GET /api/matches` - Current matches (cached)
- `GET /api/underdog-analysis` - Underdog predictions
- `GET /api/value-bets` - High-value betting opportunities

### Testing
- `GET /api/test-ml` - Test ML model prediction
- `POST /api/refresh` - Manually trigger data refresh

---

## Support & Resources

### Documentation
- `README.md` - Project overview
- `PROJECT_FIX_SUMMARY.md` - Detailed problem analysis
- `docs/` - Additional documentation

### Logs Location
- Inside container: `/app/logs/`
- On host: `./logs/` (mounted volume)

### Configuration
- Container: `.env.production`
- Docker: `docker-compose.yml`
- App: `src/config/config.py`

---

## What to Do When API is Restored

1. ✅ Update API key in `.env.production`
2. ✅ Restart container: `docker-compose restart tennis-app`
3. ✅ Monitor logs: `docker logs -f tennis_prediction`
4. ✅ Wait for first match: Check every 30 minutes
5. ✅ Verify Telegram notification arrives
6. ✅ Check `/api/stats` for match count

Expected first notification within **30-60 minutes** of API activation.

---

## Emergency Contacts

- **Telegram Errors:** Check bot token and chat ID
- **Container Won't Start:** Check `docker logs tennis_prediction`
- **Out of Memory:** Restart container or reduce model count
- **Redis Issues:** Restart with `docker-compose restart redis`

---

**Remember:** The system is working correctly. It just needs live match data to analyze!
