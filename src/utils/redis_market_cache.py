#!/usr/bin/env python3
"""
Redis Market Data Caching System
High-performance caching layer for real-time tennis betting market data and ML predictions
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
import redis
from redis.sentinel import Sentinel
import hashlib
import pickle
import threading
from contextlib import contextmanager
import asyncio
import aioredis

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Redis cache configuration"""
    host: str = 'localhost'
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    socket_timeout: int = 5
    connection_pool_max_connections: int = 50
    
    # Sentinel configuration for high availability
    use_sentinel: bool = False
    sentinel_hosts: List[Tuple[str, int]] = None
    sentinel_service_name: str = 'tennis-redis'
    
    # Cache TTL settings (seconds)
    default_ttl: int = 300  # 5 minutes
    odds_ttl: int = 30      # 30 seconds for live odds
    prediction_ttl: int = 60  # 1 minute for predictions
    match_data_ttl: int = 120  # 2 minutes for match data
    position_ttl: int = 60    # 1 minute for positions
    
    # Performance settings
    enable_compression: bool = True
    enable_async: bool = True
    key_prefix: str = 'tennis_betting'
    
    # Monitoring
    enable_metrics: bool = True
    metrics_ttl: int = 3600  # 1 hour


class RedisMarketCache:
    """
    High-performance Redis caching system for tennis betting market data
    
    Features:
    - Real-time odds caching with optimized TTL
    - ML prediction caching with versioning
    - Position and risk data caching
    - Pub/Sub for real-time updates
    - High availability with Redis Sentinel
    - Compression for large objects
    - Async support for non-blocking operations
    """
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        
        # Connection pools
        self.redis_client = None
        self.async_redis_client = None
        self.pubsub_client = None
        
        # Metrics
        self.metrics = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0,
            'start_time': datetime.now()
        }
        
        # Thread lock for metrics
        self.metrics_lock = threading.Lock()
        
        # Initialize connections
        self._initialize_connections()
        
        # Start background tasks
        if self.config.enable_metrics:
            self._start_metrics_collection()
    
    def _initialize_connections(self):
        """Initialize Redis connections with failover support"""
        try:
            if self.config.use_sentinel and self.config.sentinel_hosts:
                # High availability setup with Sentinel
                sentinel = Sentinel(
                    self.config.sentinel_hosts,
                    socket_timeout=self.config.socket_timeout
                )
                
                self.redis_client = sentinel.master_for(
                    self.config.sentinel_service_name,
                    socket_timeout=self.config.socket_timeout,
                    password=self.config.password,
                    db=self.config.db
                )
                
                logger.info("✅ Redis Sentinel connection established")
            else:
                # Standard Redis connection
                connection_pool = redis.ConnectionPool(
                    host=self.config.host,
                    port=self.config.port,
                    db=self.config.db,
                    password=self.config.password,
                    max_connections=self.config.connection_pool_max_connections,
                    socket_timeout=self.config.socket_timeout
                )
                
                self.redis_client = redis.Redis(connection_pool=connection_pool)
                
                logger.info(f"✅ Redis connection established: {self.config.host}:{self.config.port}")
            
            # Test connection
            self.redis_client.ping()
            
            # Initialize async client if enabled
            if self.config.enable_async:
                self._initialize_async_client()
            
            # Initialize pub/sub client
            self.pubsub_client = self.redis_client.pubsub()
            
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            raise
    
    async def _initialize_async_client(self):
        """Initialize async Redis client"""
        try:
            if self.config.use_sentinel:
                # Async sentinel support would require aioredis-sentinel
                redis_url = f"redis://{self.config.host}:{self.config.port}/{self.config.db}"
            else:
                redis_url = f"redis://{self.config.host}:{self.config.port}/{self.config.db}"
            
            self.async_redis_client = await aioredis.from_url(
                redis_url,
                password=self.config.password,
                socket_timeout=self.config.socket_timeout
            )
            
            logger.info("✅ Async Redis client initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ Async Redis client initialization failed: {e}")
    
    def _make_key(self, key_type: str, identifier: str, suffix: str = None) -> str:
        """Create standardized cache key"""
        key_parts = [self.config.key_prefix, key_type, identifier]
        if suffix:
            key_parts.append(suffix)
        return ':'.join(key_parts)
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data with optional compression"""
        try:
            if isinstance(data, (dict, list)):
                serialized = json.dumps(data, default=str).encode('utf-8')
            else:
                serialized = pickle.dumps(data)
            
            if self.config.enable_compression and len(serialized) > 1024:  # Compress if > 1KB
                import zlib
                serialized = zlib.compress(serialized)
                # Add compression marker
                serialized = b'COMPRESSED:' + serialized
            
            return serialized
            
        except Exception as e:
            logger.error(f"❌ Data serialization failed: {e}")
            raise
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data with decompression"""
        try:
            # Check for compression marker
            if data.startswith(b'COMPRESSED:'):
                import zlib
                data = zlib.decompress(data[11:])  # Remove marker
            
            # Try JSON first (most common)
            try:
                return json.loads(data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Fall back to pickle
                return pickle.loads(data)
                
        except Exception as e:
            logger.error(f"❌ Data deserialization failed: {e}")
            raise
    
    def _update_metrics(self, operation: str):
        """Update operation metrics"""
        if not self.config.enable_metrics:
            return
        
        with self.metrics_lock:
            self.metrics[operation] = self.metrics.get(operation, 0) + 1
    
    # Odds Caching Methods
    def cache_live_odds(self, match_id: str, odds_data: Dict[str, Any], 
                       source: str = 'betfair') -> bool:
        """
        Cache live odds data with optimized TTL
        
        Args:
            match_id: Unique match identifier
            odds_data: Odds data dictionary
            source: Data source (betfair, api_tennis, etc.)
            
        Returns:
            bool: Success status
        """
        try:
            key = self._make_key('odds', match_id, source)
            
            # Add metadata
            cache_data = {
                'odds': odds_data,
                'source': source,
                'cached_at': datetime.now().isoformat(),
                'match_id': match_id
            }
            
            # Serialize and store
            serialized_data = self._serialize_data(cache_data)
            result = self.redis_client.setex(
                key, 
                self.config.odds_ttl, 
                serialized_data
            )
            
            if result:
                self._update_metrics('sets')
                
                # Publish update notification
                self._publish_update('odds_update', {
                    'match_id': match_id,
                    'source': source,
                    'timestamp': cache_data['cached_at']
                })
                
                logger.debug(f"📊 Odds cached: {match_id} ({source})")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to cache odds: {e}")
            self._update_metrics('errors')
            return False
    
    def get_live_odds(self, match_id: str, source: str = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve live odds from cache
        
        Args:
            match_id: Unique match identifier
            source: Specific source to retrieve (optional)
            
        Returns:
            Dict containing odds data or None if not found
        """
        try:
            if source:
                # Get odds from specific source
                key = self._make_key('odds', match_id, source)
                data = self.redis_client.get(key)
                
                if data:
                    self._update_metrics('hits')
                    return self._deserialize_data(data)
                else:
                    self._update_metrics('misses')
                    return None
            else:
                # Get odds from all sources
                pattern = self._make_key('odds', match_id, '*')
                keys = self.redis_client.keys(pattern)
                
                if not keys:
                    self._update_metrics('misses')
                    return None
                
                all_odds = {}
                for key in keys:
                    data = self.redis_client.get(key)
                    if data:
                        odds_data = self._deserialize_data(data)
                        source_name = key.decode('utf-8').split(':')[-1]
                        all_odds[source_name] = odds_data
                
                if all_odds:
                    self._update_metrics('hits')
                    return all_odds
                else:
                    self._update_metrics('misses')
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Failed to get odds: {e}")
            self._update_metrics('errors')
            return None
    
    def get_best_odds(self, match_id: str) -> Optional[Dict[str, Any]]:
        """
        Get best available odds across all sources
        
        Args:
            match_id: Unique match identifier
            
        Returns:
            Dict containing best odds data
        """
        try:
            all_odds = self.get_live_odds(match_id)
            if not all_odds:
                return None
            
            best_odds = {}
            
            for source, odds_data in all_odds.items():
                odds = odds_data.get('odds', {})
                
                # Compare player 1 odds
                if 'player1_odds' in odds:
                    if 'player1_odds' not in best_odds or odds['player1_odds'] > best_odds['player1_odds']:
                        best_odds['player1_odds'] = odds['player1_odds']
                        best_odds['player1_source'] = source
                
                # Compare player 2 odds
                if 'player2_odds' in odds:
                    if 'player2_odds' not in best_odds or odds['player2_odds'] > best_odds['player2_odds']:
                        best_odds['player2_odds'] = odds['player2_odds']
                        best_odds['player2_source'] = source
            
            return best_odds if best_odds else None
            
        except Exception as e:
            logger.error(f"❌ Failed to get best odds: {e}")
            return None
    
    # ML Prediction Caching
    def cache_prediction(self, match_id: str, prediction_data: Dict[str, Any], 
                        model_version: str = 'v1.0') -> bool:
        """
        Cache ML prediction with versioning
        
        Args:
            match_id: Unique match identifier
            prediction_data: Prediction data dictionary
            model_version: Model version identifier
            
        Returns:
            bool: Success status
        """
        try:
            key = self._make_key('prediction', match_id, model_version)
            
            # Add metadata
            cache_data = {
                'prediction': prediction_data,
                'model_version': model_version,
                'cached_at': datetime.now().isoformat(),
                'match_id': match_id,
                'ttl': self.config.prediction_ttl
            }
            
            # Serialize and store
            serialized_data = self._serialize_data(cache_data)
            result = self.redis_client.setex(
                key,
                self.config.prediction_ttl,
                serialized_data
            )
            
            if result:
                self._update_metrics('sets')
                
                # Store latest prediction reference
                latest_key = self._make_key('prediction', match_id, 'latest')
                self.redis_client.setex(latest_key, self.config.prediction_ttl, model_version.encode())
                
                # Publish update notification
                self._publish_update('prediction_update', {
                    'match_id': match_id,
                    'model_version': model_version,
                    'confidence': prediction_data.get('confidence', 0),
                    'timestamp': cache_data['cached_at']
                })
                
                logger.debug(f"🧠 Prediction cached: {match_id} ({model_version})")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to cache prediction: {e}")
            self._update_metrics('errors')
            return False
    
    def get_prediction(self, match_id: str, model_version: str = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve ML prediction from cache
        
        Args:
            match_id: Unique match identifier
            model_version: Specific model version (optional, gets latest if None)
            
        Returns:
            Dict containing prediction data or None if not found
        """
        try:
            if not model_version:
                # Get latest model version
                latest_key = self._make_key('prediction', match_id, 'latest')
                model_version_bytes = self.redis_client.get(latest_key)
                
                if not model_version_bytes:
                    self._update_metrics('misses')
                    return None
                
                model_version = model_version_bytes.decode('utf-8')
            
            key = self._make_key('prediction', match_id, model_version)
            data = self.redis_client.get(key)
            
            if data:
                self._update_metrics('hits')
                return self._deserialize_data(data)
            else:
                self._update_metrics('misses')
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get prediction: {e}")
            self._update_metrics('errors')
            return None
    
    # Match Data Caching
    def cache_match_data(self, match_id: str, match_data: Dict[str, Any]) -> bool:
        """
        Cache live match data
        
        Args:
            match_id: Unique match identifier
            match_data: Match data dictionary
            
        Returns:
            bool: Success status
        """
        try:
            key = self._make_key('match', match_id)
            
            # Add metadata
            cache_data = {
                'match_data': match_data,
                'cached_at': datetime.now().isoformat(),
                'match_id': match_id
            }
            
            # Serialize and store
            serialized_data = self._serialize_data(cache_data)
            result = self.redis_client.setex(
                key,
                self.config.match_data_ttl,
                serialized_data
            )
            
            if result:
                self._update_metrics('sets')
                
                # Store in sorted set for time-based queries
                score = time.time()
                self.redis_client.zadd('active_matches', {match_id: score})
                
                # Set expiry for sorted set entry
                self.redis_client.expire('active_matches', self.config.match_data_ttl)
                
                logger.debug(f"🎾 Match data cached: {match_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to cache match data: {e}")
            self._update_metrics('errors')
            return False
    
    def get_match_data(self, match_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve match data from cache
        
        Args:
            match_id: Unique match identifier
            
        Returns:
            Dict containing match data or None if not found
        """
        try:
            key = self._make_key('match', match_id)
            data = self.redis_client.get(key)
            
            if data:
                self._update_metrics('hits')
                return self._deserialize_data(data)
            else:
                self._update_metrics('misses')
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get match data: {e}")
            self._update_metrics('errors')
            return None
    
    def get_active_matches(self, limit: int = 100) -> List[str]:
        """
        Get list of active match IDs
        
        Args:
            limit: Maximum number of matches to return
            
        Returns:
            List of active match IDs
        """
        try:
            # Get from sorted set (most recent first)
            matches = self.redis_client.zrevrange('active_matches', 0, limit - 1)
            return [match.decode('utf-8') for match in matches]
            
        except Exception as e:
            logger.error(f"❌ Failed to get active matches: {e}")
            return []
    
    # Position and Risk Caching
    def cache_position_data(self, user_id: str, position_data: Dict[str, Any]) -> bool:
        """
        Cache position and risk data
        
        Args:
            user_id: User/session identifier
            position_data: Position data dictionary
            
        Returns:
            bool: Success status
        """
        try:
            key = self._make_key('position', user_id)
            
            # Add metadata
            cache_data = {
                'position_data': position_data,
                'cached_at': datetime.now().isoformat(),
                'user_id': user_id
            }
            
            # Serialize and store
            serialized_data = self._serialize_data(cache_data)
            result = self.redis_client.setex(
                key,
                self.config.position_ttl,
                serialized_data
            )
            
            if result:
                self._update_metrics('sets')
                logger.debug(f"💰 Position data cached: {user_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to cache position data: {e}")
            self._update_metrics('errors')
            return False
    
    def get_position_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve position data from cache
        
        Args:
            user_id: User/session identifier
            
        Returns:
            Dict containing position data or None if not found
        """
        try:
            key = self._make_key('position', user_id)
            data = self.redis_client.get(key)
            
            if data:
                self._update_metrics('hits')
                return self._deserialize_data(data)
            else:
                self._update_metrics('misses')
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get position data: {e}")
            self._update_metrics('errors')
            return None
    
    # Pub/Sub for Real-time Updates
    def _publish_update(self, channel: str, data: Dict[str, Any]):
        """Publish update notification"""
        try:
            message = json.dumps(data, default=str)
            self.redis_client.publish(f"{self.config.key_prefix}:{channel}", message)
        except Exception as e:
            logger.warning(f"⚠️ Failed to publish update: {e}")
    
    def subscribe_to_updates(self, channels: List[str], callback: callable):
        """
        Subscribe to real-time updates
        
        Args:
            channels: List of channels to subscribe to
            callback: Function to call when update received
        """
        try:
            # Subscribe to channels
            channel_names = [f"{self.config.key_prefix}:{channel}" for channel in channels]
            self.pubsub_client.subscribe(*channel_names)
            
            logger.info(f"📡 Subscribed to channels: {channels}")
            
            # Listen for messages
            for message in self.pubsub_client.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        callback(message['channel'].decode('utf-8'), data)
                    except Exception as e:
                        logger.warning(f"⚠️ Message processing error: {e}")
                        
        except Exception as e:
            logger.error(f"❌ Subscription failed: {e}")
    
    # Async Methods
    async def async_cache_odds(self, match_id: str, odds_data: Dict[str, Any], 
                              source: str = 'betfair') -> bool:
        """Async version of cache_live_odds"""
        if not self.async_redis_client:
            return False
        
        try:
            key = self._make_key('odds', match_id, source)
            
            cache_data = {
                'odds': odds_data,
                'source': source,
                'cached_at': datetime.now().isoformat(),
                'match_id': match_id
            }
            
            serialized_data = self._serialize_data(cache_data)
            result = await self.async_redis_client.setex(
                key, 
                self.config.odds_ttl, 
                serialized_data
            )
            
            if result:
                self._update_metrics('sets')
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Async odds caching failed: {e}")
            self._update_metrics('errors')
            return False
    
    async def async_get_odds(self, match_id: str, source: str = None) -> Optional[Dict[str, Any]]:
        """Async version of get_live_odds"""
        if not self.async_redis_client:
            return None
        
        try:
            if source:
                key = self._make_key('odds', match_id, source)
                data = await self.async_redis_client.get(key)
                
                if data:
                    self._update_metrics('hits')
                    return self._deserialize_data(data)
                else:
                    self._update_metrics('misses')
                    return None
            else:
                # Get all odds sources
                pattern = self._make_key('odds', match_id, '*')
                keys = await self.async_redis_client.keys(pattern)
                
                if not keys:
                    self._update_metrics('misses')
                    return None
                
                all_odds = {}
                for key in keys:
                    data = await self.async_redis_client.get(key)
                    if data:
                        odds_data = self._deserialize_data(data)
                        source_name = key.decode('utf-8').split(':')[-1]
                        all_odds[source_name] = odds_data
                
                if all_odds:
                    self._update_metrics('hits')
                    return all_odds
                else:
                    self._update_metrics('misses')
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Async odds retrieval failed: {e}")
            self._update_metrics('errors')
            return None
    
    # Cache Management
    def clear_cache(self, pattern: str = None) -> int:
        """
        Clear cache entries matching pattern
        
        Args:
            pattern: Redis key pattern (clears all if None)
            
        Returns:
            Number of keys deleted
        """
        try:
            if pattern:
                keys = self.redis_client.keys(f"{self.config.key_prefix}:{pattern}")
            else:
                keys = self.redis_client.keys(f"{self.config.key_prefix}:*")
            
            if keys:
                deleted = self.redis_client.delete(*keys)
                self._update_metrics('deletes')
                logger.info(f"🧹 Cleared {deleted} cache entries")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Cache clearing failed: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            # Redis info
            redis_info = self.redis_client.info()
            
            # Calculate hit rate
            total_operations = self.metrics['hits'] + self.metrics['misses']
            hit_rate = (self.metrics['hits'] / total_operations * 100) if total_operations > 0 else 0
            
            # Uptime
            uptime = datetime.now() - self.metrics['start_time']
            
            return {
                'cache_metrics': {
                    **self.metrics,
                    'hit_rate_percent': round(hit_rate, 2),
                    'uptime_seconds': uptime.total_seconds()
                },
                'redis_info': {
                    'connected_clients': redis_info.get('connected_clients', 0),
                    'used_memory_human': redis_info.get('used_memory_human', 'Unknown'),
                    'total_commands_processed': redis_info.get('total_commands_processed', 0),
                    'keyspace_hits': redis_info.get('keyspace_hits', 0),
                    'keyspace_misses': redis_info.get('keyspace_misses', 0),
                    'expired_keys': redis_info.get('expired_keys', 0)
                },
                'config': {
                    'default_ttl': self.config.default_ttl,
                    'odds_ttl': self.config.odds_ttl,
                    'prediction_ttl': self.config.prediction_ttl,
                    'compression_enabled': self.config.enable_compression,
                    'async_enabled': self.config.enable_async
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get cache stats: {e}")
            return {'error': str(e)}
    
    def _start_metrics_collection(self):
        """Start background metrics collection"""
        def collect_metrics():
            while True:
                try:
                    # Store metrics in Redis
                    metrics_key = self._make_key('metrics', 'system')
                    metrics_data = {
                        'timestamp': datetime.now().isoformat(),
                        **self.metrics
                    }
                    
                    serialized_metrics = self._serialize_data(metrics_data)
                    self.redis_client.setex(metrics_key, self.config.metrics_ttl, serialized_metrics)
                    
                    # Sleep for 1 minute
                    time.sleep(60)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Metrics collection error: {e}")
                    time.sleep(60)
        
        # Start metrics thread
        metrics_thread = threading.Thread(target=collect_metrics, daemon=True)
        metrics_thread.start()
        logger.info("📊 Metrics collection started")
    
    @contextmanager
    def pipeline(self):
        """Context manager for Redis pipeline operations"""
        pipe = self.redis_client.pipeline()
        try:
            yield pipe
            pipe.execute()
        except Exception as e:
            logger.error(f"❌ Pipeline operation failed: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on Redis cache"""
        try:
            # Test basic operations
            test_key = f"{self.config.key_prefix}:health_check"
            test_value = str(time.time())
            
            # Set and get test
            self.redis_client.setex(test_key, 10, test_value)
            retrieved_value = self.redis_client.get(test_key)
            
            if retrieved_value and retrieved_value.decode('utf-8') == test_value:
                # Clean up test key
                self.redis_client.delete(test_key)
                
                return {
                    'status': 'healthy',
                    'response_time_ms': self._measure_response_time(),
                    'connection_pool_available': True,
                    'async_client_available': self.async_redis_client is not None
                }
            else:
                return {
                    'status': 'unhealthy',
                    'error': 'Read/write test failed'
                }
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def _measure_response_time(self) -> float:
        """Measure Redis response time"""
        start_time = time.time()
        self.redis_client.ping()
        end_time = time.time()
        return round((end_time - start_time) * 1000, 2)  # Convert to milliseconds
    
    def close(self):
        """Close Redis connections"""
        try:
            if self.pubsub_client:
                self.pubsub_client.close()
            
            if self.redis_client:
                self.redis_client.close()
            
            if self.async_redis_client:
                asyncio.create_task(self.async_redis_client.close())
            
            logger.info("✅ Redis connections closed")
            
        except Exception as e:
            logger.warning(f"⚠️ Error closing Redis connections: {e}")


# Global cache instance
_cache_instance = None

def get_cache(config: CacheConfig = None) -> RedisMarketCache:
    """Get singleton cache instance"""
    global _cache_instance
    
    if _cache_instance is None:
        _cache_instance = RedisMarketCache(config)
    
    return _cache_instance


if __name__ == '__main__':
    # Example usage
    import os
    
    # Configuration
    config = CacheConfig(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        password=os.getenv('REDIS_PASSWORD'),
        enable_compression=True,
        enable_async=True
    )
    
    # Initialize cache
    cache = RedisMarketCache(config)
    
    # Test operations
    print("🔧 Testing Redis Market Cache...")
    
    # Test odds caching
    odds_data = {
        'player1_odds': 1.85,
        'player2_odds': 2.10,
        'market_id': 'test_market_123',
        'last_updated': datetime.now().isoformat()
    }
    
    success = cache.cache_live_odds('match_123', odds_data, 'betfair')
    print(f"Odds cached: {success}")
    
    # Test odds retrieval
    retrieved_odds = cache.get_live_odds('match_123', 'betfair')
    print(f"Odds retrieved: {retrieved_odds is not None}")
    
    # Test prediction caching
    prediction_data = {
        'predicted_winner': 1,
        'confidence': 0.75,
        'probability_1': 0.65,
        'probability_2': 0.35
    }
    
    success = cache.cache_prediction('match_123', prediction_data, 'v1.0')
    print(f"Prediction cached: {success}")
    
    # Test prediction retrieval
    retrieved_prediction = cache.get_prediction('match_123')
    print(f"Prediction retrieved: {retrieved_prediction is not None}")
    
    # Get cache stats
    stats = cache.get_cache_stats()
    print(f"Cache stats: {stats}")
    
    # Health check
    health = cache.health_check()
    print(f"Health check: {health}")
    
    print("✅ Redis Market Cache test completed")