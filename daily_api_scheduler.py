#!/usr/bin/env python3
"""
📅 Daily API Scheduler - 3 Requests Per Day Rate Limiter
Manages daily API quota of 500 requests/month = ~16 requests/day
Uses only 3 requests/day to stay under monthly limit with margin
"""

import json
import os
import schedule
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

class DailyAPIScheduler:
    """
    Daily API rate limiter that makes exactly 3 API requests per day:
    - Morning: 08:00
    - Lunch: 12:00 
    - Evening: 18:00
    
    Also tracks monthly usage and provides manual override functionality.
    """
    
    def __init__(self, config_file: str = "api_scheduler_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        self.running = False
        self.scheduler_thread = None
        
        # API usage tracking
        self.daily_requests_made = 0
        self.monthly_requests_made = 0
        self.last_request_date = None
        
        # Load state
        self._load_state()
        
        # Setup schedule
        self._setup_schedule()
        
    def _load_config(self) -> Dict:
        """Load scheduler configuration"""
        default_config = {
            "daily_limit": 3,
            "monthly_limit": 500,
            "schedule": {
                "morning": "08:00",
                "lunch": "12:00", 
                "evening": "18:00"
            },
            "timezone": "UTC",
            "emergency_override_limit": 5,  # Extra requests per day in emergency
            "data_sources": ["tennis", "tennis_atp", "tennis_wta"]
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            else:
                self._save_config(default_config)
                return default_config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return default_config
    
    def _save_config(self, config: Dict):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def _load_state(self):
        """Load API usage state"""
        state_file = "api_scheduler_state.json"
        try:
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                self.daily_requests_made = state.get('daily_requests_made', 0)
                self.monthly_requests_made = state.get('monthly_requests_made', 0)
                self.last_request_date = state.get('last_request_date')
                
                # Reset daily count if new day
                if self.last_request_date:
                    last_date = datetime.fromisoformat(self.last_request_date).date()
                    today = datetime.now().date()
                    if last_date < today:
                        self.daily_requests_made = 0
                        logger.info("🌅 New day detected, resetting daily counter")
                
                # Reset monthly count if new month
                if self.last_request_date:
                    last_month = datetime.fromisoformat(self.last_request_date).month
                    current_month = datetime.now().month
                    if last_month != current_month:
                        self.monthly_requests_made = 0
                        logger.info("📅 New month detected, resetting monthly counter")
                        
        except Exception as e:
            logger.error(f"Error loading state: {e}")
    
    def _save_state(self):
        """Save current API usage state"""
        state_file = "api_scheduler_state.json"
        try:
            state = {
                'daily_requests_made': self.daily_requests_made,
                'monthly_requests_made': self.monthly_requests_made,
                'last_request_date': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def _setup_schedule(self):
        """Setup the daily schedule"""
        schedule_config = self.config.get("schedule", {})
        
        # Clear any existing jobs
        schedule.clear()
        
        # Schedule morning request
        morning_time = schedule_config.get("morning", "08:00")
        schedule.every().day.at(morning_time).do(self._scheduled_api_request, "morning")
        
        # Schedule lunch request  
        lunch_time = schedule_config.get("lunch", "12:00")
        schedule.every().day.at(lunch_time).do(self._scheduled_api_request, "lunch")
        
        # Schedule evening request
        evening_time = schedule_config.get("evening", "18:00")
        schedule.every().day.at(evening_time).do(self._scheduled_api_request, "evening")
        
        logger.info(f"📅 Scheduled API requests: {morning_time}, {lunch_time}, {evening_time}")
    
    def _scheduled_api_request(self, period: str):
        """Execute a scheduled API request"""
        if not self._can_make_scheduled_request():
            logger.warning(f"🚫 Skipping {period} request - daily limit reached")
            return
        
        try:
            logger.info(f"⏰ Making scheduled {period} API request...")
            
            # Import API modules (avoid circular imports)
            from enhanced_api_integration import get_enhanced_api
            
            api = get_enhanced_api()
            if api:
                # Make request with multiple sports for efficiency
                sports_to_fetch = self.config.get("data_sources", ["tennis"])
                result = api.get_multiple_sports_odds(sports_to_fetch, force_refresh=True)
                
                if result['success']:
                    self._record_api_request(f"scheduled_{period}")
                    logger.info(f"✅ {period.title()} API request completed - {result['total_matches']} matches")
                    
                    # Log to daily summary
                    self._log_request_summary(period, result)
                else:
                    logger.error(f"❌ {period.title()} API request failed")
            else:
                logger.error("❌ Enhanced API not available")
                
        except Exception as e:
            logger.error(f"❌ Scheduled {period} request failed: {e}")
    
    def _can_make_scheduled_request(self) -> bool:
        """Check if we can make a scheduled request"""
        return (self.daily_requests_made < self.config.get("daily_limit", 3) and
                self.monthly_requests_made < self.config.get("monthly_limit", 500))
    
    def _can_make_manual_request(self) -> bool:
        """Check if manual override request is allowed"""
        emergency_limit = self.config.get("emergency_override_limit", 5)
        total_daily_limit = self.config.get("daily_limit", 3) + emergency_limit
        
        return (self.daily_requests_made < total_daily_limit and
                self.monthly_requests_made < self.config.get("monthly_limit", 500))
    
    def _record_api_request(self, source: str):
        """Record that an API request was made"""
        self.daily_requests_made += 1
        self.monthly_requests_made += 1
        self.last_request_date = datetime.now().isoformat()
        self._save_state()
        
        logger.info(f"📊 API Request recorded: {source} (Daily: {self.daily_requests_made}/{self.config['daily_limit']}, Monthly: {self.monthly_requests_made}/{self.config['monthly_limit']})")
    
    def _log_request_summary(self, period: str, result: Dict):
        """Log detailed summary of API request"""
        summary_file = f"api_requests_log_{datetime.now().strftime('%Y_%m')}.json"
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'period': period,
            'type': 'scheduled',
            'total_matches': result.get('total_matches', 0),
            'sport_results': result.get('sport_results', {}),
            'api_usage': result.get('api_usage', {}),
            'daily_count': self.daily_requests_made,
            'monthly_count': self.monthly_requests_made
        }
        
        try:
            # Load existing log
            logs = []
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    logs = json.load(f)
            
            # Add new entry
            logs.append(log_entry)
            
            # Save back
            with open(summary_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error logging request summary: {e}")
    
    def make_manual_request(self, reason: str = "manual_override") -> Dict:
        """Make a manual API request (emergency override)"""
        if not self._can_make_manual_request():
            return {
                'success': False,
                'error': 'Manual request denied - daily or monthly limit exceeded',
                'daily_used': self.daily_requests_made,
                'monthly_used': self.monthly_requests_made,
                'limits': {
                    'daily_scheduled': self.config.get("daily_limit", 3),
                    'daily_total': self.config.get("daily_limit", 3) + self.config.get("emergency_override_limit", 5),
                    'monthly': self.config.get("monthly_limit", 500)
                }
            }
        
        try:
            logger.info(f"🔧 Making manual API request: {reason}")
            
            # Import API modules
            from enhanced_api_integration import get_enhanced_api
            
            api = get_enhanced_api()
            if api:
                # Force refresh for manual requests
                sports_to_fetch = self.config.get("data_sources", ["tennis"])
                result = api.get_multiple_sports_odds(sports_to_fetch, force_refresh=True)
                
                if result['success']:
                    self._record_api_request(f"manual_{reason}")
                    
                    # Log manual request
                    self._log_manual_request(reason, result)
                    
                    return {
                        'success': True,
                        'total_matches': result['total_matches'],
                        'source': 'manual_override',
                        'reason': reason,
                        'daily_used': self.daily_requests_made,
                        'monthly_used': self.monthly_requests_made,
                        'api_usage': result.get('api_usage', {}),
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    return {
                        'success': False,
                        'error': 'API request failed',
                        'details': result
                    }
            else:
                return {
                    'success': False,
                    'error': 'Enhanced API not available'
                }
                
        except Exception as e:
            logger.error(f"❌ Manual request failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _log_manual_request(self, reason: str, result: Dict):
        """Log manual API request"""
        manual_log_file = f"manual_api_requests_{datetime.now().strftime('%Y_%m')}.json"
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'type': 'manual_override',
            'total_matches': result.get('total_matches', 0),
            'sport_results': result.get('sport_results', {}),
            'api_usage': result.get('api_usage', {}),
            'daily_count': self.daily_requests_made,
            'monthly_count': self.monthly_requests_made
        }
        
        try:
            # Load existing log
            logs = []
            if os.path.exists(manual_log_file):
                with open(manual_log_file, 'r') as f:
                    logs = json.load(f)
            
            # Add new entry
            logs.append(log_entry)
            
            # Save back
            with open(manual_log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error logging manual request: {e}")
    
    def get_status(self) -> Dict:
        """Get current scheduler status"""
        today = datetime.now().date()
        current_month = datetime.now().month
        
        # Calculate remaining requests
        daily_scheduled_remaining = max(0, self.config.get("daily_limit", 3) - self.daily_requests_made)
        daily_manual_remaining = max(0, self.config.get("emergency_override_limit", 5) - max(0, self.daily_requests_made - self.config.get("daily_limit", 3)))
        monthly_remaining = max(0, self.config.get("monthly_limit", 500) - self.monthly_requests_made)
        
        # Get next scheduled times
        next_scheduled = self._get_next_scheduled_times()
        
        return {
            'status': 'active' if self.running else 'stopped',
            'daily_usage': {
                'requests_made': self.daily_requests_made,
                'scheduled_limit': self.config.get("daily_limit", 3),
                'manual_limit': self.config.get("emergency_override_limit", 5),
                'total_limit': self.config.get("daily_limit", 3) + self.config.get("emergency_override_limit", 5),
                'scheduled_remaining': daily_scheduled_remaining,
                'manual_remaining': daily_manual_remaining,
                'date': today.isoformat()
            },
            'monthly_usage': {
                'requests_made': self.monthly_requests_made,
                'limit': self.config.get("monthly_limit", 500),
                'remaining': monthly_remaining,
                'month': current_month
            },
            'schedule': {
                'times': self.config.get("schedule", {}),
                'next_scheduled': next_scheduled,
                'timezone': self.config.get("timezone", "UTC")
            },
            'last_request': self.last_request_date,
            'can_make_scheduled': self._can_make_scheduled_request(),
            'can_make_manual': self._can_make_manual_request()
        }
    
    def _get_next_scheduled_times(self) -> List[str]:
        """Get next scheduled request times"""
        now = datetime.now()
        today = now.date()
        schedule_times = self.config.get("schedule", {})
        
        next_times = []
        for period, time_str in schedule_times.items():
            try:
                # Parse time
                hour, minute = map(int, time_str.split(':'))
                scheduled_time = datetime.combine(today, datetime.min.time().replace(hour=hour, minute=minute))
                
                # If time has passed today, schedule for tomorrow
                if scheduled_time <= now:
                    scheduled_time += timedelta(days=1)
                
                next_times.append({
                    'period': period,
                    'time': scheduled_time.strftime('%Y-%m-%d %H:%M'),
                    'in_hours': round((scheduled_time - now).total_seconds() / 3600, 1)
                })
            except Exception as e:
                logger.error(f"Error parsing schedule time {time_str}: {e}")
        
        return sorted(next_times, key=lambda x: x['in_hours'])
    
    def start(self):
        """Start the scheduler"""
        if self.running:
            logger.warning("⚠️ Scheduler already running")
            return
        
        self.running = True
        
        def run_scheduler():
            logger.info("🚀 Daily API Scheduler started")
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            logger.info("🛑 Daily API Scheduler stopped")
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("✅ Daily API Scheduler initialized and running")
    
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("🛑 Daily API Scheduler stopped")
    
    def update_schedule(self, new_schedule: Dict):
        """Update schedule times"""
        self.config["schedule"] = new_schedule
        self._save_config(self.config)
        self._setup_schedule()
        logger.info(f"📅 Schedule updated: {new_schedule}")

# Global scheduler instance
_daily_scheduler = None

def get_daily_scheduler() -> Optional[DailyAPIScheduler]:
    """Get global scheduler instance"""
    return _daily_scheduler

def init_daily_scheduler(config_file: str = "api_scheduler_config.json") -> DailyAPIScheduler:
    """Initialize global scheduler"""
    global _daily_scheduler
    _daily_scheduler = DailyAPIScheduler(config_file)
    return _daily_scheduler

def start_daily_scheduler():
    """Start the global scheduler"""
    if _daily_scheduler:
        _daily_scheduler.start()
    else:
        logger.error("❌ Daily scheduler not initialized")

def stop_daily_scheduler():
    """Stop the global scheduler"""
    if _daily_scheduler:
        _daily_scheduler.stop()

if __name__ == "__main__":
    # Test the scheduler
    print("📅 Testing Daily API Scheduler")
    print("=" * 50)
    
    # Initialize
    scheduler = init_daily_scheduler()
    
    # Show status
    status = scheduler.get_status()
    print(f"Daily usage: {status['daily_usage']['requests_made']}/{status['daily_usage']['total_limit']}")
    print(f"Monthly usage: {status['monthly_usage']['requests_made']}/{status['monthly_usage']['limit']}")
    print(f"Can make scheduled: {status['can_make_scheduled']}")
    print(f"Can make manual: {status['can_make_manual']}")
    
    print("\nNext scheduled requests:")
    for req in status['schedule']['next_scheduled']:
        print(f"  {req['period']}: {req['time']} (in {req['in_hours']} hours)")
    
    # Test manual request
    print("\n🔧 Testing manual request...")
    result = scheduler.make_manual_request("test_manual")
    print(f"Manual request result: {result['success']}")
    
    # Start scheduler for testing (will run in background)
    print("\n🚀 Starting scheduler...")
    start_daily_scheduler()
    
    print("✅ Daily API Scheduler test completed!")
    print("Press Ctrl+C to stop...")
    
    try:
        # Keep running for testing
        while True:
            time.sleep(10)
            status = scheduler.get_status()
            print(f"\rStatus: Daily {status['daily_usage']['requests_made']}/{status['daily_usage']['total_limit']} | Monthly {status['monthly_usage']['requests_made']}/{status['monthly_usage']['limit']}", end="")
    except KeyboardInterrupt:
        stop_daily_scheduler()
        print("\n🛑 Scheduler stopped")