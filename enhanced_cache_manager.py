#!/usr/bin/env python3
"""
🚀 Enhanced Cache Manager - Smart Redis + Disk Caching
Implements intelligent caching with Redis primary and disk fallback
"""

import json
import pickle
import os
import time
import hashlib

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Union
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration for cache manager"""
    redis_host: str = 'localhost'
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # TTL settings (in seconds)
    default_ttl: int = 1200  # 20 minutes
    odds_ttl: int = 300      # 5 minutes for live odds
    rankings_ttl: int = 86400  # 24 hours for rankings
    tournament_ttl: int = 3600  # 1 hour for tournament data
    
    # Disk cache settings
    disk_cache_dir: str = "cache"
    max_disk_cache_size: int = 100 * 1024 * 1024  # 100MB
    
    # Compression settings
    enable_compression: bool = True
    compression_threshold: int = 1024  # Compress data larger than 1KB

class SmartCacheManager:
    """
    Enhanced cache manager with Redis primary and disk fallback
    Features:
    - Redis for fast access and distributed caching
    - Disk fallback when Redis unavailable
    - Intelligent TTL based on data type
    - Data compression for large objects
    - Cache statistics and monitoring
    """
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.redis_client = None
        self.redis_available = False
        
        # Ensure disk cache directory exists
        os.makedirs(self.config.disk_cache_dir, exist_ok=True)
        
        # Statistics
        self.stats = {
            'redis_hits': 0,
            'redis_misses': 0,
            'disk_hits': 0,
            'disk_misses': 0,
            'sets': 0,
            'errors': 0
        }
        
        self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis connection"""
        if not REDIS_AVAILABLE:
            logger.info("⚠️ Redis module not available. Using disk cache only.")
            self.redis_available = False
            return
            
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=False,  # We handle encoding ourselves
                socket_timeout=2,
                socket_connect_timeout=2
            )
            
            # Test connection
            self.redis_client.ping()
            self.redis_available = True
            logger.info(f"✅ Redis connected: {self.config.redis_host}:{self.config.redis_port}")
            
        except Exception as e:
            logger.warning(f"⚠️ Redis unavailable: {e}. Using disk cache only.")
            self.redis_available = False
    
    def _get_cache_key(self, namespace: str, key: str) -> str:
        """Generate cache key with namespace"""
        return f"tennis_cache:{namespace}:{key}"
    
    def _get_ttl(self, data_type: str, context: Dict = None) -> int:
        """Calculate intelligent TTL based on data type and context"""
        context = context or {}
        
        # Dynamic TTL based on data type
        ttl_map = {
            'odds': self.config.odds_ttl,
            'rankings': self.config.rankings_ttl,
            'tournament': self.config.tournament_ttl,
            'api_response': self.config.default_ttl
        }
        
        base_ttl = ttl_map.get(data_type, self.config.default_ttl)
        
        # Adjust TTL based on context
        if data_type == 'odds' and context:
            match_start = context.get('match_start_time')
            if match_start:
                try:
                    start_time = datetime.fromisoformat(match_start.replace('Z', '+00:00'))
                    time_to_match = (start_time - datetime.now()).total_seconds()
                    
                    # Shorter TTL closer to match time
                    if time_to_match < 3600:  # Less than 1 hour
                        base_ttl = min(base_ttl, 120)  # 2 minutes
                    elif time_to_match < 7200:  # Less than 2 hours
                        base_ttl = min(base_ttl, 300)  # 5 minutes
                except Exception:
                    pass
        
        return base_ttl
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress data if enabled and beneficial"""
        serialized = pickle.dumps(data)
        
        if (self.config.enable_compression and 
            len(serialized) > self.config.compression_threshold):
            try:
                import gzip
                compressed = gzip.compress(serialized)
                if len(compressed) < len(serialized):
                    return b'compressed:' + compressed
            except ImportError:
                logger.warning("gzip not available, skipping compression")
        
        return b'raw:' + serialized
    
    def _decompress_data(self, data: bytes) -> Any:
        """Decompress data if needed"""
        if data.startswith(b'compressed:'):
            try:
                import gzip
                decompressed = gzip.decompress(data[11:])  # Remove 'compressed:' prefix
                return pickle.loads(decompressed)
            except ImportError:
                logger.error("gzip not available for decompression")
                raise
        elif data.startswith(b'raw:'):
            return pickle.loads(data[4:])  # Remove 'raw:' prefix
        else:
            # Legacy format
            return pickle.loads(data)
    
    def _get_disk_path(self, cache_key: str) -> str:
        """Get disk cache file path"""
        # Create hash of key for filename
        key_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return os.path.join(self.config.disk_cache_dir, f"{key_hash}.cache")
    
    def get(self, namespace: str, key: str, data_type: str = "default") -> Optional[Any]:
        """
        Get value from cache (Redis first, then disk)
        
        Args:
            namespace: Cache namespace (e.g., 'api', 'odds', 'rankings')
            key: Cache key
            data_type: Type of data for TTL calculation
        """
        cache_key = self._get_cache_key(namespace, key)
        
        # Try Redis first
        if self.redis_available:
            try:
                redis_data = self.redis_client.get(cache_key)
                if redis_data:
                    result = self._decompress_data(redis_data)
                    self.stats['redis_hits'] += 1
                    logger.debug(f"📋 Redis cache hit: {namespace}:{key}")
                    return result
                else:
                    self.stats['redis_misses'] += 1
            except Exception as e:
                logger.warning(f"Redis error: {e}")
                self.stats['errors'] += 1
                self.redis_available = False
        
        # Try disk cache
        disk_path = self._get_disk_path(cache_key)
        try:
            if os.path.exists(disk_path):
                # Check if file is not expired
                file_age = time.time() - os.path.getmtime(disk_path)
                ttl = self._get_ttl(data_type)
                
                if file_age < ttl:
                    with open(disk_path, 'rb') as f:
                        result = self._decompress_data(f.read())
                    self.stats['disk_hits'] += 1
                    logger.debug(f"💾 Disk cache hit: {namespace}:{key}")
                    
                    # Populate Redis cache if available
                    if self.redis_available:
                        try:
                            self._set_redis(cache_key, result, ttl - int(file_age))
                        except Exception:
                            pass
                    
                    return result
                else:
                    # Remove expired file
                    os.remove(disk_path)
        except Exception as e:
            logger.warning(f"Disk cache error: {e}")
            self.stats['errors'] += 1
        
        self.stats['disk_misses'] += 1
        return None
    
    def _set_redis(self, cache_key: str, data: Any, ttl: int):
        """Set data in Redis with TTL"""
        compressed_data = self._compress_data(data)
        self.redis_client.setex(cache_key, ttl, compressed_data)
    
    def set(self, namespace: str, key: str, data: Any, data_type: str = "default", 
            context: Dict = None) -> bool:
        """
        Set value in cache (both Redis and disk)
        
        Args:
            namespace: Cache namespace
            key: Cache key
            data: Data to cache
            data_type: Type of data for TTL calculation
            context: Additional context for TTL calculation
        """
        cache_key = self._get_cache_key(namespace, key)
        ttl = self._get_ttl(data_type, context)
        success = False
        
        # Set in Redis
        if self.redis_available:
            try:
                self._set_redis(cache_key, data, ttl)
                success = True
                logger.debug(f"📋 Redis cache set: {namespace}:{key} (TTL: {ttl}s)")
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
                self.stats['errors'] += 1
                self.redis_available = False
        
        # Set in disk cache
        try:
            disk_path = self._get_disk_path(cache_key)
            compressed_data = self._compress_data(data)
            
            with open(disk_path, 'wb') as f:
                f.write(compressed_data)
            
            success = True
            logger.debug(f"💾 Disk cache set: {namespace}:{key}")
            
            # Cleanup old disk cache files if needed
            self._cleanup_disk_cache()
            
        except Exception as e:
            logger.warning(f"Disk cache set error: {e}")
            self.stats['errors'] += 1
        
        if success:
            self.stats['sets'] += 1
        
        return success
    
    def delete(self, namespace: str, key: str) -> bool:
        """Delete value from cache"""
        cache_key = self._get_cache_key(namespace, key)
        success = False
        
        # Delete from Redis
        if self.redis_available:
            try:
                self.redis_client.delete(cache_key)
                success = True
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")
                self.stats['errors'] += 1
        
        # Delete from disk
        try:
            disk_path = self._get_disk_path(cache_key)
            if os.path.exists(disk_path):
                os.remove(disk_path)
                success = True
        except Exception as e:
            logger.warning(f"Disk delete error: {e}")
            self.stats['errors'] += 1
        
        return success
    
    def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in a namespace"""
        pattern = self._get_cache_key(namespace, "*")
        cleared = 0
        
        # Clear from Redis
        if self.redis_available:
            try:
                keys = self.redis_client.keys(pattern)
                if keys:
                    cleared += self.redis_client.delete(*keys)
            except Exception as e:
                logger.warning(f"Redis clear error: {e}")
                self.stats['errors'] += 1
        
        # Clear from disk (more complex due to hashing)
        try:
            for filename in os.listdir(self.config.disk_cache_dir):
                if filename.endswith('.cache'):
                    filepath = os.path.join(self.config.disk_cache_dir, filename)
                    # We can't easily match namespace in disk cache due to hashing
                    # This is a limitation of the current design
                    # For now, we'll skip disk cleanup for namespace operations
        except Exception as e:
            logger.warning(f"Disk clear error: {e}")
            self.stats['errors'] += 1
        
        logger.info(f"🧹 Cleared {cleared} keys from namespace '{namespace}'")
        return cleared
    
    def _cleanup_disk_cache(self):
        """Clean up old disk cache files to maintain size limit"""
        try:
            cache_files = []
            total_size = 0
            
            for filename in os.listdir(self.config.disk_cache_dir):
                if filename.endswith('.cache'):
                    filepath = os.path.join(self.config.disk_cache_dir, filename)
                    size = os.path.getsize(filepath)
                    mtime = os.path.getmtime(filepath)
                    cache_files.append((filepath, size, mtime))
                    total_size += size
            
            if total_size > self.config.max_disk_cache_size:
                # Sort by modification time (oldest first)
                cache_files.sort(key=lambda x: x[2])
                
                for filepath, size, _ in cache_files:
                    os.remove(filepath)
                    total_size -= size
                    logger.debug(f"🗑️ Removed old cache file: {filepath}")
                    
                    if total_size <= self.config.max_disk_cache_size * 0.8:
                        break
                        
        except Exception as e:
            logger.warning(f"Cache cleanup error: {e}")
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        redis_info = {}
        if self.redis_available:
            try:
                info = self.redis_client.info()
                redis_info = {
                    'connected_clients': info.get('connected_clients', 0),
                    'used_memory_human': info.get('used_memory_human', 'Unknown'),
                    'keyspace_hits': info.get('keyspace_hits', 0),
                    'keyspace_misses': info.get('keyspace_misses', 0)
                }
            except Exception:
                redis_info = {'error': 'Failed to get Redis info'}
        
        # Disk cache stats
        disk_stats = {'size': 0, 'files': 0}
        try:
            for filename in os.listdir(self.config.disk_cache_dir):
                if filename.endswith('.cache'):
                    filepath = os.path.join(self.config.disk_cache_dir, filename)
                    disk_stats['size'] += os.path.getsize(filepath)
                    disk_stats['files'] += 1
        except Exception:
            disk_stats = {'error': 'Failed to get disk stats'}
        
        return {
            'redis_available': self.redis_available,
            'redis_info': redis_info,
            'disk_cache': disk_stats,
            'hit_rates': {
                'redis_hit_rate': self._calculate_hit_rate('redis'),
                'disk_hit_rate': self._calculate_hit_rate('disk'),
                'total_hit_rate': self._calculate_hit_rate('total')
            },
            'counters': self.stats
        }
    
    def _calculate_hit_rate(self, cache_type: str) -> float:
        """Calculate hit rate for cache type"""
        if cache_type == 'redis':
            total = self.stats['redis_hits'] + self.stats['redis_misses']
            hits = self.stats['redis_hits']
        elif cache_type == 'disk':
            total = self.stats['disk_hits'] + self.stats['disk_misses']
            hits = self.stats['disk_hits']
        elif cache_type == 'total':
            total = (self.stats['redis_hits'] + self.stats['redis_misses'] + 
                    self.stats['disk_hits'] + self.stats['disk_misses'])
            hits = self.stats['redis_hits'] + self.stats['disk_hits']
        else:
            return 0.0
        
        return (hits / total * 100) if total > 0 else 0.0

# Global cache manager instance
_cache_manager = None

def init_cache_manager(config: CacheConfig = None) -> SmartCacheManager:
    """Initialize global cache manager"""
    global _cache_manager
    _cache_manager = SmartCacheManager(config)
    logger.info("🚀 Enhanced cache manager initialized")
    return _cache_manager

def get_cache_manager() -> Optional[SmartCacheManager]:
    """Get global cache manager instance"""
    return _cache_manager

# Convenience functions
def cache_get(namespace: str, key: str, data_type: str = "default") -> Optional[Any]:
    """Get from cache using global manager"""
    if _cache_manager:
        return _cache_manager.get(namespace, key, data_type)
    return None

def cache_set(namespace: str, key: str, data: Any, data_type: str = "default", 
              context: Dict = None) -> bool:
    """Set in cache using global manager"""
    if _cache_manager:
        return _cache_manager.set(namespace, key, data, data_type, context)
    return False

def cache_delete(namespace: str, key: str) -> bool:
    """Delete from cache using global manager"""
    if _cache_manager:
        return _cache_manager.delete(namespace, key)
    return False

def cache_stats() -> Dict:
    """Get cache statistics"""
    if _cache_manager:
        return _cache_manager.get_stats()
    return {'error': 'Cache manager not initialized'}

if __name__ == "__main__":
    # Test the enhanced cache manager
    print("🚀 Testing Enhanced Cache Manager")
    print("=" * 50)
    
    # Initialize with default config
    cache_manager = init_cache_manager()
    
    # Test basic operations
    print("1. Testing basic cache operations...")
    
    test_data = {
        'player1': 'Novak Djokovic',
        'player2': 'Rafael Nadal',
        'odds': {'player1': 1.85, 'player2': 1.95}
    }
    
    # Set data
    success = cache_set('odds', 'match_123', test_data, 'odds')
    print(f"   Set result: {success}")
    
    # Get data
    retrieved = cache_get('odds', 'match_123', 'odds')
    print(f"   Get result: {retrieved is not None}")
    print(f"   Data matches: {retrieved == test_data}")
    
    # Test with context for dynamic TTL
    context = {'match_start_time': (datetime.now() + timedelta(minutes=30)).isoformat()}
    cache_set('odds', 'urgent_match', test_data, 'odds', context)
    
    # Test different data types
    cache_set('rankings', 'atp_top_100', {'ranking': 'data'}, 'rankings')
    cache_set('api', 'tournament_list', ['tournament1', 'tournament2'], 'tournament')
    
    # Get statistics
    stats = cache_stats()
    print(f"\n2. Cache Statistics:")
    print(f"   Redis available: {stats['redis_available']}")
    print(f"   Total hit rate: {stats['hit_rates']['total_hit_rate']:.1f}%")
    print(f"   Operations: {stats['counters']['sets']} sets")
    
    print("\n✅ Enhanced cache manager test completed!")