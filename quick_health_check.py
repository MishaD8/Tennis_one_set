#!/usr/bin/env python3
"""
Quick health check for the tennis backend server
"""

import subprocess
import socket
import sys
import time

def check_port_open(host='127.0.0.1', port=5001, timeout=3):
    """Check if a port is open and accepting connections"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False

def check_process_running():
    """Check if the tennis backend process is running"""
    try:
        result = subprocess.run(['pgrep', '-f', 'tennis_backend.py'], 
                              capture_output=True, text=True)
        return len(result.stdout.strip()) > 0
    except Exception:
        return False

def main():
    print("🎾 Quick Health Check")
    print("=" * 30)
    
    # Check if process is running
    process_running = check_process_running()
    print(f"Process running: {'✅' if process_running else '❌'}")
    
    # Check if port is open
    port_open = check_port_open()
    print(f"Port 5001 open: {'✅' if port_open else '❌'}")
    
    # Overall status
    if process_running and port_open:
        print("\n✅ Server appears to be running correctly!")
        return 0
    elif port_open:
        print("\n⚠️ Port is open but process not detected. May be running differently.")
        return 1
    else:
        print("\n❌ Server does not appear to be running.")
        print("Try starting with: python3 tennis_backend.py")
        return 2

if __name__ == "__main__":
    sys.exit(main())