#!/usr/bin/env python3
"""
🚀 Enhanced API Integration - Uses Smart Cache Manager
Replaces file-based caching with Redis + intelligent disk caching
"""

import requests
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from enhanced_cache_manager import (
    init_cache_manager, get_cache_manager, 
    CacheConfig, cache_get, cache_set, cache_delete, cache_stats
)

logger = logging.getLogger(__name__)

class EnhancedAPIIntegration:
    """
    Enhanced API integration with smart caching
    Replaces both api_economy_patch.py and correct_odds_api_integration.py
    """
    
    def __init__(self, api_key: str = None, config: Dict = None):
        self.api_key = api_key
        self.config = config or {}
        
        # Initialize cache manager if not already done
        if get_cache_manager() is None:
            cache_config = CacheConfig(
                odds_ttl=self.config.get('cache_minutes', 20) * 60,
                rankings_ttl=86400,  # 24 hours
                tournament_ttl=3600,  # 1 hour
                enable_compression=True
            )
            init_cache_manager(cache_config)
        
        self.cache_manager = get_cache_manager()
        
        # API usage tracking
        self.requests_used = 0
        self.requests_remaining = None
        self.max_per_hour = self.config.get('max_per_hour', 30)
        
        # Load API key from config if not provided
        if not self.api_key:
            self._load_api_key()
    
    def _load_api_key(self):
        """Load API key from config.json"""
        try:
            if os.path.exists('config.json'):
                with open('config.json', 'r') as f:
                    config = json.load(f)
                    # Try both locations in config
                    api_key = (config.get('data_sources', {})
                              .get('the_odds_api', {})
                              .get('api_key'))
                    
                    if not api_key:
                        api_key = (config.get('betting_apis', {})
                                  .get('the_odds_api', {})
                                  .get('api_key'))
                    
                    # Handle environment variable format
                    if api_key and api_key.startswith('${') and api_key.endswith('}'):
                        env_var = api_key[2:-1]
                        api_key = os.getenv(env_var)
                    
                    self.api_key = api_key
                    
        except Exception as e:
            logger.warning(f"Could not load API key from config: {e}")
            self.api_key = None
    
    def _can_make_request(self) -> tuple[bool, str]:
        """Check if we can make an API request based on rate limits"""
        # Get recent request count from cache
        hourly_key = f"api_requests_hour_{datetime.now().strftime('%Y%m%d_%H')}"
        request_count = cache_get('rate_limit', hourly_key, 'api_response') or 0
        
        if request_count >= self.max_per_hour:
            return False, f"Rate limit exceeded: {request_count}/{self.max_per_hour} per hour"
        
        return True, "OK"
    
    def _record_request(self):
        """Record an API request for rate limiting"""
        hourly_key = f"api_requests_hour_{datetime.now().strftime('%Y%m%d_%H')}"
        request_count = cache_get('rate_limit', hourly_key, 'api_response') or 0
        
        # Cache for 1 hour with 1 hour TTL
        cache_set('rate_limit', hourly_key, request_count + 1, 'api_response',
                 context={'ttl_override': 3600})
    
    def _make_api_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Make API request with enhanced error handling"""
        if not self.api_key:
            logger.error("No API key available")
            return None
        
        try:
            params['apiKey'] = self.api_key
            url = f"https://api.the-odds-api.com/v4/{endpoint}"
            
            logger.info(f"📡 Making API request: {endpoint}")
            response = requests.get(url, params=params, timeout=15)
            
            # Update usage tracking from headers
            headers = response.headers
            self.requests_used = headers.get('x-requests-used', 'Unknown')
            self.requests_remaining = headers.get('x-requests-remaining', 'Unknown')
            
            # Record request for rate limiting
            self._record_request()
            
            logger.info(f"API Usage: {self.requests_used} used, {self.requests_remaining} remaining")
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                logger.error("❌ Invalid API key")
                return None
            elif response.status_code == 422:
                logger.warning("⚠️ No data available or invalid parameters")
                return None
            elif response.status_code == 429:
                logger.warning("⚠️ Rate limit exceeded")
                return None
            else:
                logger.error(f"❌ API error {response.status_code}: {response.text}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"❌ Request failed: {e}")
            return None
    
    def get_tennis_odds(self, sport_key: str = "tennis", force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get tennis odds with smart caching
        
        Returns:
            Dict with success status, data, source, and metadata
        """
        cache_key = f"odds_{sport_key}"
        
        # Check manual update trigger (compatibility with old system)
        manual_update_needed = self._check_manual_update_trigger()
        if manual_update_needed:
            force_refresh = True
            logger.info("🔄 Manual update triggered")
        
        # Try cache first unless force refresh
        if not force_refresh:
            cached_data = cache_get('odds', cache_key, 'odds')
            if cached_data:
                return {
                    'success': True,
                    'data': cached_data,
                    'source': 'cache',
                    'status': 'CACHED',
                    'emoji': '📋'
                }
        
        # Check rate limits
        can_request, reason = self._can_make_request()
        if not can_request and not manual_update_needed:
            logger.warning(f"🚦 {reason}")
            
            # Return stale cache if available
            stale_data = cache_get('odds', cache_key, 'odds')
            if stale_data:
                return {
                    'success': True,
                    'data': stale_data,
                    'source': 'stale_cache',
                    'status': 'RATE_LIMITED',
                    'emoji': '💾',
                    'warning': reason
                }
            
            return {
                'success': False,
                'error': reason,
                'source': 'rate_limited'
            }
        
        # Make API request
        params = {
            'regions': 'us,uk,eu,au',
            'markets': 'h2h',
            'oddsFormat': 'decimal',
            'dateFormat': 'iso'
        }
        
        data = self._make_api_request(f"sports/{sport_key}/odds", params)
        
        if data:
            # Convert to tennis format
            converted_data = self._convert_odds_format(data)
            
            # Cache the result with context for smart TTL
            context = {
                'sport_key': sport_key,
                'timestamp': datetime.now().isoformat(),
                'matches_count': len(converted_data)
            }
            
            cache_set('odds', cache_key, converted_data, 'odds', context)
            
            # Clear manual update trigger if it was set
            if manual_update_needed:
                self._clear_manual_update_trigger()
                logger.info("✅ Manual update completed")
            
            return {
                'success': True,
                'data': converted_data,
                'source': 'fresh_api' if not manual_update_needed else 'manual_update',
                'status': 'LIVE_API' if not manual_update_needed else 'MANUAL_UPDATE',
                'emoji': '🔴' if not manual_update_needed else '🔄',
                'matches_count': len(converted_data),
                'api_usage': {
                    'used': self.requests_used,
                    'remaining': self.requests_remaining
                }
            }
        else:
            # API failed, try to return cached data
            fallback_data = cache_get('odds', cache_key, 'odds')
            if fallback_data:
                return {
                    'success': True,
                    'data': fallback_data,
                    'source': 'error_fallback',
                    'status': 'API_ERROR_FALLBACK',
                    'emoji': '💾'
                }
            
            return {
                'success': False,
                'error': 'API request failed and no cached data available',
                'source': 'api_error'
            }
    
    def _convert_odds_format(self, api_data: List[Dict]) -> Dict[str, Dict]:
        """Convert The Odds API format to tennis system format"""
        converted_odds = {}
        
        for match in api_data:
            try:
                match_id = match.get('id', f"odds_{datetime.now().timestamp()}")
                player1 = match.get('home_team', 'Player 1')
                player2 = match.get('away_team', 'Player 2')
                
                # Find best odds
                best_p1_odds = None
                best_p2_odds = None
                best_p1_bookmaker = None
                best_p2_bookmaker = None
                
                for bookmaker in match.get('bookmakers', []):
                    bookmaker_name = bookmaker.get('title', bookmaker.get('key', 'Unknown'))
                    
                    for market in bookmaker.get('markets', []):
                        if market.get('key') == 'h2h':
                            for outcome in market.get('outcomes', []):
                                player_name = outcome.get('name')
                                odds = outcome.get('price', 0)
                                
                                if not odds:
                                    continue
                                
                                # Convert to decimal odds if needed
                                decimal_odds = float(odds)
                                
                                # Match to players and keep best odds
                                if self._player_name_match(player_name, player1):
                                    if best_p1_odds is None or decimal_odds > best_p1_odds:
                                        best_p1_odds = decimal_odds
                                        best_p1_bookmaker = bookmaker_name
                                elif self._player_name_match(player_name, player2):
                                    if best_p2_odds is None or decimal_odds > best_p2_odds:
                                        best_p2_odds = decimal_odds
                                        best_p2_bookmaker = bookmaker_name
                
                # Create tennis format result
                if best_p1_odds and best_p2_odds:
                    converted_odds[match_id] = {
                        'match_info': {
                            'player1': player1,
                            'player2': player2,
                            'tournament': f"Tennis Match ({match.get('sport_title', 'Tennis')})",
                            'surface': 'Unknown',
                            'date': match.get('commence_time', datetime.now().isoformat())[:10],
                            'time': match.get('commence_time', datetime.now().isoformat())[11:16],
                            'source': 'enhanced_api'
                        },
                        'best_markets': {
                            'winner': {
                                'player1': {
                                    'odds': round(best_p1_odds, 2),
                                    'bookmaker': best_p1_bookmaker
                                },
                                'player2': {
                                    'odds': round(best_p2_odds, 2),
                                    'bookmaker': best_p2_bookmaker
                                }
                            }
                        },
                        'cache_info': {
                            'cached_at': datetime.now().isoformat(),
                            'source': 'enhanced_api_integration'
                        }
                    }
                    
            except Exception as e:
                logger.error(f"Error processing match {match.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"🔄 Converted {len(converted_odds)} matches to tennis format")
        return converted_odds
    
    def _player_name_match(self, api_name: str, system_name: str) -> bool:
        """Check if player names match (fuzzy matching)"""
        if not api_name or not system_name:
            return False
        
        # Simple fuzzy matching
        api_lower = api_name.lower().strip()
        system_lower = system_name.lower().strip()
        
        return (api_lower == system_lower or 
                api_lower in system_lower or 
                system_lower in api_lower)
    
    def get_available_sports(self) -> List[Dict]:
        """Get available sports with caching"""
        cache_key = "available_sports"
        
        # Try cache first
        cached_data = cache_get('sports', cache_key, 'api_response')
        if cached_data:
            logger.info("📋 Using cached sports data")
            return cached_data
        
        # Make API request
        data = self._make_api_request("sports", {})
        
        if data:
            # Filter tennis sports
            tennis_sports = [sport for sport in data if 'tennis' in sport.get('key', '').lower()]
            
            # Cache for 24 hours
            cache_set('sports', cache_key, tennis_sports, 'api_response',
                     context={'ttl_override': 86400})
            
            logger.info(f"🎾 Found {len(tennis_sports)} tennis sports")
            return tennis_sports
        
        return []
    
    def get_multiple_sports_odds(self, sport_keys: List[str] = None, force_refresh: bool = False) -> Dict:
        """Get odds for multiple tennis sports"""
        if sport_keys is None:
            sport_keys = ['tennis', 'tennis_atp', 'tennis_wta']
        
        all_odds = {}
        results = {
            'success': True,
            'total_matches': 0,
            'sport_results': {},
            'api_usage': None
        }
        
        for sport_key in sport_keys:
            try:
                sport_result = self.get_tennis_odds(sport_key, force_refresh)
                results['sport_results'][sport_key] = {
                    'success': sport_result['success'],
                    'matches_count': len(sport_result.get('data', {})),
                    'source': sport_result.get('source'),
                    'status': sport_result.get('status')
                }
                
                if sport_result['success']:
                    all_odds.update(sport_result['data'])
                    results['total_matches'] += len(sport_result['data'])
                
                # Update API usage info
                if 'api_usage' in sport_result:
                    results['api_usage'] = sport_result['api_usage']
                    
            except Exception as e:
                logger.error(f"Error getting odds for {sport_key}: {e}")
                results['sport_results'][sport_key] = {
                    'success': False,
                    'error': str(e)
                }
        
        results['data'] = all_odds
        return results
    
    def _check_manual_update_trigger(self) -> bool:
        """Check for manual update trigger (compatibility)"""
        trigger_file = "manual_update_trigger.json"
        try:
            if os.path.exists(trigger_file):
                with open(trigger_file, 'r') as f:
                    trigger_data = json.load(f)
                return trigger_data.get('force_update', False)
        except Exception:
            pass
        return False
    
    def _clear_manual_update_trigger(self):
        """Clear manual update trigger (compatibility)"""
        trigger_file = "manual_update_trigger.json"
        try:
            trigger_data = {
                'force_update': False,
                'last_manual_update': datetime.now().isoformat(),
                'message': 'Update completed by enhanced API integration'
            }
            with open(trigger_file, 'w') as f:
                json.dump(trigger_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not clear manual trigger: {e}")
    
    def create_manual_update_trigger(self):
        """Create manual update trigger for compatibility"""
        trigger_file = "manual_update_trigger.json"
        try:
            trigger_data = {
                'force_update': True,
                'created_at': datetime.now().isoformat(),
                'message': 'Manual update requested via enhanced API'
            }
            with open(trigger_file, 'w') as f:
                json.dump(trigger_data, f, indent=2)
            logger.info("✅ Manual update trigger created")
            return True
        except Exception as e:
            logger.error(f"Could not create manual trigger: {e}")
            return False
    
    def get_cache_stats(self) -> Dict:
        """Get comprehensive cache statistics"""
        cache_stats_data = cache_stats()
        
        return {
            'cache_manager': cache_stats_data,
            'api_integration': {
                'requests_used': self.requests_used,
                'requests_remaining': self.requests_remaining,
                'max_per_hour': self.max_per_hour,
                'api_key_configured': bool(self.api_key)
            }
        }
    
    def clear_all_cache(self):
        """Clear all cached data"""
        if self.cache_manager:
            self.cache_manager.clear_namespace('odds')
            self.cache_manager.clear_namespace('sports')
            self.cache_manager.clear_namespace('api_response')
            self.cache_manager.clear_namespace('rate_limit')
            logger.info("🧹 All cache cleared")

# Global instance for compatibility
_enhanced_api = None

def init_enhanced_api(api_key: str = None, config: Dict = None) -> EnhancedAPIIntegration:
    """Initialize global enhanced API instance"""
    global _enhanced_api
    _enhanced_api = EnhancedAPIIntegration(api_key, config)
    logger.info("🚀 Enhanced API integration initialized")
    return _enhanced_api

def get_enhanced_api() -> Optional[EnhancedAPIIntegration]:
    """Get global enhanced API instance"""
    return _enhanced_api

# Compatibility functions for existing code
def economical_tennis_request(sport_key: str = 'tennis', force_fresh: bool = False) -> Dict:
    """Compatibility function for existing code"""
    if _enhanced_api:
        return _enhanced_api.get_tennis_odds(sport_key, force_fresh)
    else:
        return {'success': False, 'error': 'Enhanced API not initialized'}

def get_api_usage() -> Dict:
    """Compatibility function for existing code"""
    if _enhanced_api:
        return _enhanced_api.get_cache_stats()
    return {'error': 'Enhanced API not initialized'}

def trigger_manual_update() -> bool:
    """Compatibility function for existing code"""
    if _enhanced_api:
        return _enhanced_api.create_manual_update_trigger()
    return False

if __name__ == "__main__":
    # Test the enhanced API integration
    print("🚀 Testing Enhanced API Integration")
    print("=" * 50)
    
    # Initialize
    enhanced_api = init_enhanced_api()
    
    # Test getting odds
    print("1. Getting tennis odds...")
    result = enhanced_api.get_tennis_odds('tennis')
    print(f"   Success: {result['success']}")
    print(f"   Source: {result.get('source')}")
    print(f"   Status: {result.get('status')}")
    
    if result['success']:
        print(f"   Matches found: {len(result['data'])}")
    
    # Test cache hit
    print("\n2. Testing cache hit...")
    result2 = enhanced_api.get_tennis_odds('tennis')
    print(f"   Source: {result2.get('source')} (should be 'cache')")
    
    # Test multiple sports
    print("\n3. Getting multiple sports...")
    multi_result = enhanced_api.get_multiple_sports_odds(['tennis', 'tennis_atp'])
    print(f"   Total matches: {multi_result['total_matches']}")
    
    # Get statistics
    stats = enhanced_api.get_cache_stats()
    print(f"\n4. Cache Statistics:")
    print(f"   Cache hit rate: {stats['cache_manager']['hit_rates']['total_hit_rate']:.1f}%")
    
    print("\n✅ Enhanced API integration test completed!")