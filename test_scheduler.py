#!/usr/bin/env python3
"""
Test script for daily API scheduler
"""

import sys
import os
sys.path.append(os.getcwd())

from daily_api_scheduler import init_daily_scheduler, get_daily_scheduler

def test_scheduler():
    print("📅 Testing Daily API Scheduler")
    print("=" * 50)
    
    try:
        # Initialize
        scheduler = init_daily_scheduler()
        print("✅ Scheduler initialized")
        
        # Show status
        status = scheduler.get_status()
        print(f"📊 Status: {status['status']}")
        print(f"📈 Daily usage: {status['daily_usage']['requests_made']}/{status['daily_usage']['total_limit']}")
        print(f"📅 Monthly usage: {status['monthly_usage']['requests_made']}/{status['monthly_usage']['limit']}")
        print(f"🔧 Can make manual: {status['can_make_manual']}")
        print(f"⏰ Can make scheduled: {status['can_make_scheduled']}")
        
        print("\n⏰ Next scheduled requests:")
        for req in status['schedule']['next_scheduled'][:3]:
            print(f"  {req['period']}: {req['time']} (in {req['in_hours']} hours)")
        
        # Test configuration
        config = scheduler.config
        print(f"\n⚙️ Configuration:")
        print(f"  Daily limit: {config.get('daily_limit', 3)}")
        print(f"  Monthly limit: {config.get('monthly_limit', 500)}")
        print(f"  Emergency override: {config.get('emergency_override_limit', 5)}")
        print(f"  Schedule times: {config.get('schedule', {})}")
        
        # Show recommendations
        if status.get('recommendations'):
            print(f"\n💡 Recommendations:")
            for rec in status['recommendations']:
                print(f"  {rec}")
        
        print("\n✅ Daily API Scheduler test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_scheduler()
    sys.exit(0 if success else 1)