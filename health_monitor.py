#!/usr/bin/env python3
import requests
import time
from datetime import datetime

def check_health():
    try:
        response = requests.get("http://localhost:5001/api/health", timeout=10)
        if response.status_code == 200:
            print(f"✅ {datetime.now()}: Service healthy")
            return True
        else:
            print(f"❌ {datetime.now()}: Service unhealthy - HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {datetime.now()}: Service error - {e}")
        return False

if __name__ == "__main__":
    check_health()