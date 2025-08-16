# Tennis Backend Security Deployment Guide

## 🛡️ Security Fixes Applied

### Critical Issues Resolved

1. **Redis Connection Fallback** ✅
   - **Issue**: Flask-Limiter was failing with "Connection refused" errors when Redis was unavailable
   - **Solution**: Implemented automatic fallback to secure in-memory storage
   - **Implementation**: 
     - Added connection testing before Redis initialization
     - Enabled `in_memory_fallback_enabled=True` in Flask-Limiter
     - Graceful degradation with appropriate logging

2. **SSL/TLS Handling** ✅
   - **Issue**: Malformed HTTPS requests showing garbled SSL handshake data
   - **Solution**: Enhanced request monitoring and proper HTTPS enforcement
   - **Implementation**:
     - Added SSL handshake detection in request monitor
     - Implemented HTTPS enforcement for production
     - Custom 426 Upgrade Required exception handling
     - Proper redirect for GET requests, error for POST requests

3. **Enhanced Security Headers** ✅
   - **Added**: Comprehensive security headers for financial applications
   - **Features**: 
     - Strict CSP policies for API vs web endpoints
     - HSTS with preload
     - Cross-Origin policies
     - Permissions Policy blocking unnecessary features
     - Cache control for sensitive API data

4. **Redis Monitoring** ✅
   - **Added**: Health check endpoint with Redis monitoring
   - **Features**: Connection time monitoring, performance metrics, error reporting
   - **Endpoint**: `/api/redis-status` (requires API key)

## 🚀 Production Deployment

### Environment Variables

Set these environment variables for production:

```bash
# Security
export FLASK_ENV=production
export FLASK_SECRET_KEY="your-secret-key-here"
export TENNIS_API_KEY="your-secure-api-key"

# Redis (optional)
export REDIS_URL="redis://localhost:6379"
# or for external Redis:
# export REDIS_URL="redis://username:password@host:port/db"

# SSL/TLS (optional - if not using reverse proxy)
export SSL_CERT_PATH="/path/to/certificate.crt"
export SSL_KEY_PATH="/path/to/private.key"

# Trusted proxies (if behind load balancer)
export TRUSTED_PROXIES="10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"

# CORS
export ALLOWED_ORIGINS="https://your-domain.com,https://www.your-domain.com"
```

### Nginx Configuration

Use the updated `tennis-nginx.conf` with enhanced security:

```bash
# Install the configuration
sudo cp tennis-nginx.conf /etc/nginx/sites-available/tennis-backend
sudo ln -s /etc/nginx/sites-available/tennis-backend /etc/nginx/sites-enabled/

# Update the server_name in the config
sudo sed -i 's/your-domain.com/your-actual-domain.com/g' /etc/nginx/sites-enabled/tennis-backend

# Get SSL certificates (Let's Encrypt)
sudo certbot --nginx -d your-actual-domain.com

# Test and reload nginx
sudo nginx -t
sudo systemctl reload nginx
```

### Redis Setup (Optional but Recommended)

```bash
# Install Redis
sudo apt update
sudo apt install redis-server

# Configure Redis for security
sudo vi /etc/redis/redis.conf

# Add these settings:
# bind 127.0.0.1 ::1
# requirepass your-redis-password
# maxmemory 256mb
# maxmemory-policy allkeys-lru

# Restart Redis
sudo systemctl restart redis-server
sudo systemctl enable redis-server

# Update Redis URL with password
export REDIS_URL="redis://:your-redis-password@localhost:6379"
```

### System Service

Create a systemd service:

```bash
sudo cp tennis-prediction.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable tennis-prediction
sudo systemctl start tennis-prediction
```

## 🔒 Security Features

### Rate Limiting
- **Production**: Uses Redis for distributed rate limiting
- **Fallback**: Automatic in-memory storage when Redis unavailable
- **Limits**: 100/day, 20/hour, 5/minute (configurable)
- **Monitoring**: Real-time connection health checks

### HTTPS Enforcement
- **Production**: Automatic HTTPS redirection
- **Development**: Optional HTTPS
- **Headers**: HSTS with preload, secure cookies
- **SSL/TLS**: Modern protocols only (TLS 1.2+)

### Security Headers
- **CSP**: Strict Content Security Policy
- **HSTS**: HTTP Strict Transport Security
- **Cross-Origin**: Comprehensive CORS and cross-origin policies
- **Permissions**: Disabled unnecessary browser features

### API Security
- **Authentication**: API key validation with timing attack protection
- **Input Validation**: Comprehensive sanitization and validation
- **Error Handling**: Safe error responses without information leakage
- **Monitoring**: Request pattern analysis and threat detection

## 📊 Monitoring Endpoints

### Health Check
```bash
curl https://your-domain.com/api/health
```

Response includes:
- Component status
- Infrastructure health (Redis, SSL, rate limiting)
- Security configuration
- Performance warnings

### Redis Status (Requires API Key)
```bash
curl -H "X-API-Key: your-api-key" https://your-domain.com/api/redis-status
```

Response includes:
- Connection health and performance
- Memory usage
- Client connections
- Performance metrics

## 🛠️ Troubleshooting

### Redis Connection Issues
- **Symptom**: "Connection refused" errors
- **Solution**: Application automatically falls back to in-memory storage
- **Check**: `curl https://your-domain.com/api/health` shows `rate_limiting: memory`
- **Fix**: Restart Redis service or check Redis configuration

### SSL/TLS Issues
- **Symptom**: "Bad request version" errors in logs
- **Solution**: Ensure proper SSL termination at nginx level
- **Check**: Verify certificate installation and nginx SSL configuration
- **Debug**: Check nginx error logs for SSL-related issues

### Rate Limiting Not Working
- **Check**: Health endpoint for rate limiting status
- **Verify**: Redis connection in health check
- **Test**: Make multiple requests to trigger rate limiting

## 🔧 Configuration Options

### Rate Limiting Customization
```python
# In tennis_backend.py
limiter = Limiter(
    app=app,
    key_func=secure_rate_limit_key_func,
    default_limits=["200 per day", "50 per hour", "10 per minute"],  # Adjust as needed
    # ... other settings
)
```

### Security Headers Customization
```python
# Modify CSP in set_security_headers function
csp = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline'; "  # Add trusted CDNs if needed
    # ... other policies
)
```

## 📈 Performance Considerations

### Redis Configuration
- **Memory**: Allocate appropriate memory (256MB recommended for small deployments)
- **Persistence**: Configure based on needs (RDB for snapshots, AOF for durability)
- **Network**: Use local Redis for best performance

### Security vs Performance
- **Headers**: Security headers add minimal overhead
- **Rate Limiting**: In-memory fallback is faster but not distributed
- **SSL/TLS**: Modern ciphers balance security and performance

## ✅ Verification Checklist

After deployment, verify:

- [ ] Health endpoint returns `status: "healthy"` or `status: "degraded"`
- [ ] Redis status shows connection health
- [ ] HTTPS redirection works (HTTP → HTTPS)
- [ ] Security headers present in responses
- [ ] Rate limiting functions (test with multiple requests)
- [ ] API authentication works
- [ ] SSL certificate is valid and not expired
- [ ] Nginx configuration passes `nginx -t`
- [ ] Application logs show no critical errors

## 🆘 Emergency Procedures

### Service Issues
1. Check service status: `sudo systemctl status tennis-prediction`
2. View logs: `journalctl -u tennis-prediction -f`
3. Restart service: `sudo systemctl restart tennis-prediction`

### High Load/DDoS
1. Check rate limiting is active
2. Monitor nginx access logs
3. Implement additional nginx rate limiting if needed

### SSL Certificate Issues
1. Renew certificate: `sudo certbot renew`
2. Check nginx configuration
3. Restart nginx: `sudo systemctl restart nginx`

---

**Security Contact**: Monitor logs regularly and keep system updated with security patches.