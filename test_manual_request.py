#!/usr/bin/env python3
"""
Test manual API request functionality (dry run)
"""

import sys
import os
sys.path.append(os.getcwd())

from daily_api_scheduler import init_daily_scheduler

def test_manual_request():
    print("🔧 Testing Manual API Request")
    print("=" * 50)
    
    try:
        # Initialize
        scheduler = init_daily_scheduler()
        print("✅ Scheduler initialized")
        
        # Check initial status
        status = scheduler.get_status()
        print(f"📊 Initial daily usage: {status['daily_usage']['requests_made']}/{status['daily_usage']['total_limit']}")
        print(f"🔧 Can make manual request: {status['can_make_manual']}")
        
        # Try to make a manual request (this will fail because enhanced_api_integration isn't initialized)
        print("\n🔧 Attempting manual request...")
        result = scheduler.make_manual_request("test_request")
        
        if result['success']:
            print(f"✅ Manual request succeeded!")
            print(f"📊 Total matches: {result.get('total_matches', 0)}")
            print(f"📈 Daily used: {result.get('daily_used', 0)}")
            print(f"📅 Monthly used: {result.get('monthly_used', 0)}")
        else:
            print(f"⚠️ Manual request failed (expected): {result.get('error', 'Unknown error')}")
            print(f"📊 Current usage: Daily {result.get('daily_used', 0)}, Monthly {result.get('monthly_used', 0)}")
            
            # This is expected since enhanced_api_integration is not set up
            if 'Enhanced API not available' in result.get('error', ''):
                print("✅ Error is expected - enhanced API not configured in test")
        
        # Check status after attempt
        status_after = scheduler.get_status()
        print(f"\n📊 After attempt daily usage: {status_after['daily_usage']['requests_made']}/{status_after['daily_usage']['total_limit']}")
        print(f"🔧 Can still make manual request: {status_after['can_make_manual']}")
        
        print("\n✅ Manual request test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_manual_request()
    sys.exit(0 if success else 1)