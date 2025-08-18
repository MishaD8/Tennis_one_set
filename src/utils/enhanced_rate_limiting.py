#!/usr/bin/env python3
"""
Enhanced Rate Limiting with Redis Fallback
Comprehensive rate limiting system with in-memory fallback
"""

import os
import time
import logging
import threading
from typing import Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class RateLimitInfo:
    """Rate limit information"""
    limit: int
    remaining: int
    reset_time: float
    window_size: int

class InMemoryRateLimiter:
    """In-memory rate limiter with sliding window"""
    
    def __init__(self):
        self.requests = defaultdict(deque)
        self.locks = defaultdict(threading.Lock)
    
    def is_allowed(self, key: str, limit: int, window: int) -> Tuple[bool, RateLimitInfo]:
        """Check if request is allowed under rate limit"""
        now = time.time()
        
        with self.locks[key]:
            # Clean old requests outside the window
            request_times = self.requests[key]
            while request_times and request_times[0] <= now - window:
                request_times.popleft()
            
            # Check if under limit
            current_count = len(request_times)
            allowed = current_count < limit
            
            if allowed:
                request_times.append(now)
            
            # Calculate reset time (when oldest request will expire)
            reset_time = now + window
            if request_times:
                reset_time = request_times[0] + window
            
            rate_limit_info = RateLimitInfo(
                limit=limit,
                remaining=max(0, limit - current_count - (1 if allowed else 0)),
                reset_time=reset_time,
                window_size=window
            )
            
            return allowed, rate_limit_info

class RedisRateLimiter:
    """Redis-based rate limiter"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def is_allowed(self, key: str, limit: int, window: int) -> Tuple[bool, RateLimitInfo]:
        """Check if request is allowed using Redis sliding window"""
        now = time.time()
        window_start = now - window
        
        try:
            # Use Redis pipeline for atomic operations
            pipe = self.redis.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests
            pipe.zcard(key)
            
            # Add current request with score as timestamp
            pipe.zadd(key, {str(now): now})
            
            # Set expiration for cleanup
            pipe.expire(key, window)
            
            results = pipe.execute()
            
            current_count = results[1]  # Count after cleanup
            
            # Check if allowed (current_count is before adding this request)
            allowed = current_count < limit
            
            if not allowed:
                # Remove the request we just added since it's not allowed
                self.redis.zrem(key, str(now))
            
            # Calculate remaining and reset time
            remaining = max(0, limit - current_count - (1 if allowed else 0))
            
            # Get oldest request to calculate reset time
            oldest = self.redis.zrange(key, 0, 0, withscores=True)
            reset_time = now + window
            if oldest:
                reset_time = oldest[0][1] + window
            
            rate_limit_info = RateLimitInfo(
                limit=limit,
                remaining=remaining,
                reset_time=reset_time,
                window_size=window
            )
            
            return allowed, rate_limit_info
            
        except Exception as e:
            logger.error(f"Redis rate limiting error: {e}")
            # Fall back to allowing the request
            return True, RateLimitInfo(limit, limit-1, now + window, window)

class EnhancedRateLimitManager:
    """Enhanced rate limiting manager with Redis fallback"""
    
    def __init__(self):
        self.redis_client = None
        self.redis_available = False
        self.in_memory_limiter = InMemoryRateLimiter()
        self.redis_limiter = None
        
        # Default rate limits
        self.default_limits = {
            'api_requests': {'limit': 20, 'window': 3600},  # 20 per hour
            'predictions': {'limit': 100, 'window': 86400},  # 100 per day
            'betting_actions': {'limit': 50, 'window': 3600},  # 50 per hour
            'notifications': {'limit': 10, 'window': 600},   # 10 per 10 minutes
            'data_collection': {'limit': 8, 'window': 86400}  # 8 per day
        }
        
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection with fallback"""
        redis_url = os.getenv('REDIS_URL', '').strip()
        
        if redis_url and redis_url != 'memory://':
            try:
                import redis
                
                if redis_url.startswith('redis://'):
                    self.redis_client = redis.Redis.from_url(
                        redis_url,
                        socket_connect_timeout=2,
                        socket_timeout=2,
                        retry_on_timeout=True,
                        health_check_interval=30
                    )
                else:
                    # Default local Redis
                    self.redis_client = redis.Redis(
                        host='localhost',
                        port=6379,
                        socket_connect_timeout=2,
                        socket_timeout=2,
                        retry_on_timeout=True
                    )
                
                # Test connection
                self.redis_client.ping()
                self.redis_available = True
                self.redis_limiter = RedisRateLimiter(self.redis_client)
                logger.info("✅ Redis rate limiting enabled")
                return
                
            except ImportError:
                logger.warning("⚠️ Redis package not installed")
            except Exception as e:
                logger.warning(f"⚠️ Redis connection failed: {e}")
        
        # Fall back to in-memory
        self.redis_available = False
        logger.info("✅ In-memory rate limiting enabled (Redis fallback)")
    
    def check_rate_limit(self, category: str, identifier: str, 
                        limit: Optional[int] = None, window: Optional[int] = None) -> Tuple[bool, RateLimitInfo]:
        """Check if request is within rate limits"""
        
        # Get limits for category
        if category in self.default_limits:
            default_limit = self.default_limits[category]['limit']
            default_window = self.default_limits[category]['window']
        else:
            default_limit = 10
            default_window = 3600
        
        final_limit = limit or default_limit
        final_window = window or default_window
        
        # Create rate limit key
        key = f"rate_limit:{category}:{identifier}"
        
        # Use appropriate rate limiter
        if self.redis_available and self.redis_limiter:
            try:
                return self.redis_limiter.is_allowed(key, final_limit, final_window)
            except Exception as e:
                logger.warning(f"Redis rate limiter failed, using in-memory: {e}")
                # Fall back to in-memory
                return self.in_memory_limiter.is_allowed(key, final_limit, final_window)
        else:
            return self.in_memory_limiter.is_allowed(key, final_limit, final_window)
    
    def get_rate_limit_status(self, category: str, identifier: str) -> Optional[RateLimitInfo]:
        """Get current rate limit status without consuming a request"""
        if category in self.default_limits:
            limit = self.default_limits[category]['limit']
            window = self.default_limits[category]['window']
        else:
            return None
        
        key = f"rate_limit:{category}:{identifier}"
        now = time.time()
        
        if self.redis_available and self.redis_client:
            try:
                # Get current count without modifying
                window_start = now - window
                self.redis_client.zremrangebyscore(key, 0, window_start)
                current_count = self.redis_client.zcard(key)
                
                # Get reset time
                oldest = self.redis_client.zrange(key, 0, 0, withscores=True)
                reset_time = now + window
                if oldest:
                    reset_time = oldest[0][1] + window
                
                return RateLimitInfo(
                    limit=limit,
                    remaining=max(0, limit - current_count),
                    reset_time=reset_time,
                    window_size=window
                )
                
            except Exception as e:
                logger.warning(f"Failed to get Redis rate limit status: {e}")
        
        # In-memory fallback
        with self.in_memory_limiter.locks[key]:
            request_times = self.in_memory_limiter.requests[key]
            
            # Clean old requests
            while request_times and request_times[0] <= now - window:
                request_times.popleft()
            
            current_count = len(request_times)
            reset_time = now + window
            if request_times:
                reset_time = request_times[0] + window
            
            return RateLimitInfo(
                limit=limit,
                remaining=max(0, limit - current_count),
                reset_time=reset_time,
                window_size=window
            )
    
    def reset_rate_limit(self, category: str, identifier: str) -> bool:
        """Reset rate limit for a specific key"""
        key = f"rate_limit:{category}:{identifier}"
        
        try:
            if self.redis_available and self.redis_client:
                self.redis_client.delete(key)
                logger.info(f"✅ Redis rate limit reset for {key}")
            
            # Also reset in-memory
            if key in self.in_memory_limiter.requests:
                with self.in_memory_limiter.locks[key]:
                    self.in_memory_limiter.requests[key].clear()
                logger.info(f"✅ In-memory rate limit reset for {key}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to reset rate limit for {key}: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get rate limiting system status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'redis_available': self.redis_available,
            'rate_limiter': 'redis' if self.redis_available else 'in-memory',
            'default_limits': self.default_limits,
            'active_limits': {}
        }
        
        # Add Redis-specific info
        if self.redis_available and self.redis_client:
            try:
                redis_info = self.redis_client.info('memory')
                status['redis_info'] = {
                    'used_memory': redis_info.get('used_memory_human'),
                    'connected_clients': redis_info.get('connected_clients'),
                    'uptime': redis_info.get('uptime_in_seconds')
                }
            except Exception as e:
                status['redis_error'] = str(e)
        
        # Add in-memory info
        status['in_memory_info'] = {
            'active_keys': len(self.in_memory_limiter.requests),
            'total_locks': len(self.in_memory_limiter.locks)
        }
        
        return status
    
    def cleanup_expired_limits(self):
        """Clean up expired rate limit entries (for in-memory storage)"""
        now = time.time()
        cleaned_count = 0
        
        # Clean in-memory entries
        for key in list(self.in_memory_limiter.requests.keys()):
            with self.in_memory_limiter.locks[key]:
                request_times = self.in_memory_limiter.requests[key]
                original_length = len(request_times)
                
                # Remove entries older than maximum window (1 day)
                while request_times and request_times[0] <= now - 86400:
                    request_times.popleft()
                
                cleaned_count += original_length - len(request_times)
                
                # Remove empty deques
                if not request_times:
                    del self.in_memory_limiter.requests[key]
                    if key in self.in_memory_limiter.locks:
                        del self.in_memory_limiter.locks[key]
        
        if cleaned_count > 0:
            logger.info(f"✅ Cleaned {cleaned_count} expired rate limit entries")
        
        return cleaned_count

# Global rate limit manager instance
rate_limit_manager = None

def get_rate_limit_manager() -> EnhancedRateLimitManager:
    """Get singleton rate limit manager instance"""
    global rate_limit_manager
    if rate_limit_manager is None:
        rate_limit_manager = EnhancedRateLimitManager()
    return rate_limit_manager

def rate_limit_decorator(category: str, limit: Optional[int] = None, window: Optional[int] = None):
    """Decorator for rate limiting functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = get_rate_limit_manager()
            
            # Use function name as identifier
            identifier = func.__name__
            
            # Check rate limit
            allowed, rate_info = manager.check_rate_limit(category, identifier, limit, window)
            
            if not allowed:
                logger.warning(f"Rate limit exceeded for {category}:{identifier}")
                raise Exception(f"Rate limit exceeded. Try again at {datetime.fromtimestamp(rate_info.reset_time)}")
            
            # Add rate limit info to response if function returns dict
            result = func(*args, **kwargs)
            
            if isinstance(result, dict):
                result['rate_limit'] = {
                    'limit': rate_info.limit,
                    'remaining': rate_info.remaining,
                    'reset_time': rate_info.reset_time
                }
            
            return result
        
        return wrapper
    return decorator

# Convenience decorators for common rate limits
def api_rate_limit(func):
    """Rate limit for API requests (20 per hour)"""
    return rate_limit_decorator('api_requests')(func)

def prediction_rate_limit(func):
    """Rate limit for predictions (100 per day)"""
    return rate_limit_decorator('predictions')(func)

def betting_rate_limit(func):
    """Rate limit for betting actions (50 per hour)"""
    return rate_limit_decorator('betting_actions')(func)

def notification_rate_limit(func):
    """Rate limit for notifications (10 per 10 minutes)"""
    return rate_limit_decorator('notifications')(func)