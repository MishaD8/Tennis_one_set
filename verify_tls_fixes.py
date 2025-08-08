#!/usr/bin/env python3
"""
Verify that TLS/HTTPS fixes are working correctly
"""

import requests
import socket
import sys
import json

def test_http_endpoints():
    """Test that HTTP endpoints work correctly"""
    print("🔍 Testing HTTP Endpoints")
    print("-" * 30)
    
    base_url = "http://localhost:5001"
    endpoints = [
        ("/", "Dashboard"),
        ("/api/health", "Health Check"),
        ("/api/stats", "Statistics")
    ]
    
    results = []
    for path, name in endpoints:
        try:
            response = requests.get(f"{base_url}{path}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {name}: Working (200 OK)")
                results.append(True)
            else:
                print(f"❌ {name}: Failed ({response.status_code})")
                results.append(False)
        except Exception as e:
            print(f"❌ {name}: Error - {str(e)}")
            results.append(False)
    
    return all(results)

def test_security_headers():
    """Test that security headers are properly configured for HTTP"""
    print("\n🔒 Testing Security Headers")
    print("-" * 30)
    
    try:
        response = requests.get("http://localhost:5001/", timeout=5)
        headers = response.headers
        
        # Check for proper HTTP configuration
        tests = [
            ("X-Content-Type-Options", "nosniff", "Content type sniffing protection"),
            ("X-Frame-Options", "DENY", "Clickjacking protection"),
            ("Content-Security-Policy", lambda x: "upgrade-insecure-requests" not in x, "No HTTPS upgrade for HTTP mode"),
            ("Referrer-Policy", "same-origin", "Development-friendly referrer policy"),
        ]
        
        results = []
        for header_name, expected_value, description in tests:
            if header_name in headers:
                if callable(expected_value):
                    if expected_value(headers[header_name]):
                        print(f"✅ {description}: OK")
                        results.append(True)
                    else:
                        print(f"❌ {description}: Failed")
                        results.append(False)
                elif expected_value in headers[header_name]:
                    print(f"✅ {description}: OK")
                    results.append(True)
                else:
                    print(f"❌ {description}: Unexpected value")
                    results.append(False)
            else:
                print(f"❌ {description}: Header missing")
                results.append(False)
        
        # Check that HSTS is NOT present (good for HTTP development)
        if "Strict-Transport-Security" not in headers:
            print("✅ HSTS correctly omitted for HTTP development")
            results.append(True)
        else:
            print("❌ HSTS present in HTTP mode (should be omitted)")
            results.append(False)
        
        return all(results)
        
    except Exception as e:
        print(f"❌ Security headers test failed: {e}")
        return False

def test_tls_handshake_handling():
    """Test that TLS handshake attempts are handled gracefully"""
    print("\n🤝 Testing TLS Handshake Error Handling")
    print("-" * 30)
    
    try:
        # Create a raw socket connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(('localhost', 5001))
        
        # Send TLS handshake bytes (what was causing the original error)
        tls_handshake = b'\x16\x03\x01\x00\x10' + b'test_handshake'
        sock.send(tls_handshake)
        
        # Receive response
        response = sock.recv(1024).decode('utf-8', errors='ignore')
        sock.close()
        
        # Check if server responded with HTTP error instead of crashing
        if "HTTP" in response and ("400" in response or "Protocol Mismatch" in response):
            print("✅ TLS handshake gracefully handled with HTTP error response")
            return True
        else:
            print("❌ TLS handshake not properly handled")
            return False
            
    except Exception as e:
        print(f"❌ TLS handshake test failed: {e}")
        return False

def test_cors_configuration():
    """Test CORS configuration"""
    print("\n🌐 Testing CORS Configuration")
    print("-" * 30)
    
    try:
        response = requests.get("http://localhost:5001/api/health", 
                               headers={"Origin": "http://localhost:5001"}, 
                               timeout=5)
        
        if "Access-Control-Allow-Origin" in response.headers:
            print("✅ CORS headers present")
            return True
        else:
            print("❌ CORS headers missing")
            return False
            
    except Exception as e:
        print(f"❌ CORS test failed: {e}")
        return False

def main():
    """Run all verification tests"""
    print("🎾 TLS/HTTPS FIXES VERIFICATION")
    print("=" * 50)
    
    tests = [
        ("HTTP Endpoints", test_http_endpoints),
        ("Security Headers", test_security_headers), 
        ("TLS Handshake Handling", test_tls_handshake_handling),
        ("CORS Configuration", test_cors_configuration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All fixes verified! The TLS/HTTPS issues have been resolved.")
        print("✅ Dashboard should now work correctly without TLS handshake errors.")
        return 0
    else:
        print(f"❌ Some tests failed. {total - passed} issues remaining.")
        return 1

if __name__ == "__main__":
    sys.exit(main())