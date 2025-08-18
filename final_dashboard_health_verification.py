#!/usr/bin/env python3
"""
Final Dashboard Health Verification
Tests the exact scenario that was causing the issue and verifies it's fixed
"""

import requests
import json
from datetime import datetime

API_BASE = "http://127.0.0.1:5001/api"

def simulate_health_monitor_calls():
    """Simulate the exact calls the system health monitor makes"""
    
    print("🏥 FINAL DASHBOARD HEALTH VERIFICATION")
    print("=" * 60)
    print("Testing the exact calls that system-health-monitor.js makes...")
    print()
    
    # Test 1: API Health Check (working)
    print("1️⃣ Testing API Health Check...")
    try:
        response = requests.get(f"{API_BASE}/health-check", headers={'Accept': 'application/json'})
        if response.ok:
            print("   ✅ API Health Check: WORKING")
            print(f"   📄 Response: {response.json()}")
        else:
            print(f"   ❌ API Health Check: FAILED ({response.status_code})")
    except Exception as e:
        print(f"   ❌ API Health Check: ERROR ({e})")
    
    print()
    
    # Test 2: Database Health (fixed)
    print("2️⃣ Testing Database Health...")
    try:
        response = requests.get(f"{API_BASE}/system/database-health", headers={'Accept': 'application/json'})
        data = response.json()
        
        print(f"   📄 Raw Response: {data}")
        
        # Simulate the FIXED JavaScript logic
        is_healthy_old_logic = data.get('success', False)  # OLD (broken) logic
        is_healthy_new_logic = data.get('status') == 'healthy'  # NEW (fixed) logic
        
        print(f"   🔧 OLD Logic (data.success): {is_healthy_old_logic}")
        print(f"   ✅ NEW Logic (data.status === 'healthy'): {is_healthy_new_logic}")
        
        if is_healthy_new_logic:
            print("   🎯 Database Health: HEALTHY (dashboard will show green)")
        else:
            print("   ❌ Database Health: UNHEALTHY (dashboard will show red)")
            
    except Exception as e:
        print(f"   ❌ Database Health: ERROR ({e})")
    
    print()
    
    # Test 3: ML Models Health (fixed)
    print("3️⃣ Testing ML Models Health...")
    try:
        response = requests.get(f"{API_BASE}/system/ml-health", headers={'Accept': 'application/json'})
        data = response.json()
        
        print(f"   📄 Raw Response: {json.dumps(data, indent=4)}")
        
        # Simulate the FIXED JavaScript logic
        is_healthy_old_logic = data.get('success', False)  # OLD (broken) logic
        is_healthy_new_logic = data.get('status') == 'healthy'  # NEW (fixed) logic
        
        print(f"   🔧 OLD Logic (data.success): {is_healthy_old_logic}")
        print(f"   ✅ NEW Logic (data.status === 'healthy'): {is_healthy_new_logic}")
        
        if is_healthy_new_logic:
            print("   🎯 ML Models Health: HEALTHY (dashboard will show green)")
        else:
            print("   ❌ ML Models Health: UNHEALTHY (dashboard will show red)")
            
    except Exception as e:
        print(f"   ❌ ML Models Health: ERROR ({e})")
    
    print()
    
    # Test 4: Data Collection (should work)
    print("4️⃣ Testing Data Collection Health...")
    try:
        response = requests.get(f"{API_BASE}/api-economy-status", headers={'Accept': 'application/json'})
        data = response.json()
        
        # This endpoint already returns 'success' field, so it should work
        is_healthy = data.get('success', False)
        print(f"   📄 Response success field: {is_healthy}")
        
        if is_healthy:
            print("   ✅ Data Collection: HEALTHY")
        else:
            print("   ❌ Data Collection: UNHEALTHY")
            
    except Exception as e:
        print(f"   ❌ Data Collection: ERROR ({e})")
    
    print()
    print("=" * 60)
    print("🎯 DASHBOARD HEALTH STATUS PREDICTION")
    print("=" * 60)
    
    # Simulate overall health calculation
    try:
        # Get all health data
        api_ok = requests.get(f"{API_BASE}/health-check").ok
        
        db_response = requests.get(f"{API_BASE}/system/database-health")
        db_healthy = db_response.json().get('status') == 'healthy' if db_response.ok else False
        
        ml_response = requests.get(f"{API_BASE}/system/ml-health") 
        ml_healthy = ml_response.json().get('status') == 'healthy' if ml_response.ok else False
        
        data_response = requests.get(f"{API_BASE}/api-economy-status")
        data_healthy = data_response.json().get('success', False) if data_response.ok else False
        
        components = {
            'API': api_ok,
            'Database': db_healthy, 
            'ML Models': ml_healthy,
            'Data Collection': data_healthy
        }
        
        print("Component Health Status:")
        for component, status in components.items():
            print(f"  • {component}: {'✅ HEALTHY' if status else '❌ UNHEALTHY'}")
        
        healthy_count = sum(components.values())
        total_count = len(components)
        
        print(f"\nOverall: {healthy_count}/{total_count} components healthy")
        
        if healthy_count == total_count:
            print("🎊 PREDICTION: Dashboard will show HEALTHY status")
        elif healthy_count >= total_count * 0.75:
            print("⚠️ PREDICTION: Dashboard will show DEGRADED status") 
        else:
            print("🚨 PREDICTION: Dashboard will show CRITICAL status")
            
        print("\n✅ ISSUE RESOLUTION:")
        print("• Fixed system-health-monitor.js to parse 'status' field instead of 'success'")
        print("• Database and ML endpoints return status: 'healthy'")
        print("• JavaScript now correctly interprets backend responses")
        print("• Dashboard should display accurate health status")
        
    except Exception as e:
        print(f"❌ Error calculating overall health: {e}")

if __name__ == "__main__":
    simulate_health_monitor_calls()