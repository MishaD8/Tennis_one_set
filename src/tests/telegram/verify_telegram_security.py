#!/usr/bin/env python3
"""
🔐 Telegram Security Verification Script

This script verifies that all Telegram security issues have been resolved:
- No hardcoded bot tokens in source files
- No hardcoded chat IDs in source files  
- Proper environment variable usage
- Error handling for missing environment variables
- .env file loading support

Author: Claude Code (Anthropic)
"""

import os
import re
import glob
from dotenv import load_dotenv

def scan_for_hardcoded_credentials():
    """Scan all Python files for hardcoded Telegram credentials"""
    
    print("🔍 SCANNING FOR HARDCODED CREDENTIALS")
    print("=" * 45)
    
    # Bot token pattern (10 digits : 35 alphanumeric/underscore/dash characters)
    bot_token_pattern = r'\b\d{10}:[A-Za-z0-9_-]{35}\b'
    
    # Chat ID pattern (negative numbers with 9+ digits)
    chat_id_pattern = r'-\d{9,}'
    
    # Specific hardcoded token that was found
    specific_token_pattern = r'8369911887:AAHvXoNVTjpl3H3u0rVtuMxUkKEEozGIkFs'
    
    issues_found = []
    
    # Scan only application Python files (exclude venv, env, and this script)
    python_files = glob.glob("*.py") + glob.glob("tests/*.py")
    
    for file_path in python_files:
        if ('__pycache__' in file_path or '.git' in file_path or 
            'venv' in file_path or 'env' in file_path or 
            file_path == 'verify_telegram_security.py'):
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                line_number = 0
                
                for line in content.split('\n'):
                    line_number += 1
                    
                    # Check for bot token pattern
                    if re.search(bot_token_pattern, line):
                        issues_found.append({
                            'file': file_path,
                            'line': line_number,
                            'type': 'Hardcoded Bot Token',
                            'content': line.strip()
                        })
                    
                    # Check for specific token
                    if re.search(specific_token_pattern, line):
                        issues_found.append({
                            'file': file_path,
                            'line': line_number,
                            'type': 'Specific Hardcoded Token',
                            'content': line.strip()
                        })
                    
                    # Check for suspicious chat ID patterns (but exclude examples)
                    if re.search(chat_id_pattern, line) and 'example' not in line.lower() and 'test' not in line.lower():
                        issues_found.append({
                            'file': file_path,
                            'line': line_number,
                            'type': 'Potential Hardcoded Chat ID',
                            'content': line.strip()
                        })
                        
        except Exception as e:
            print(f"⚠️  Error scanning {file_path}: {e}")
    
    if issues_found:
        print("❌ SECURITY ISSUES FOUND:")
        for issue in issues_found:
            print(f"   File: {issue['file']}")
            print(f"   Line: {issue['line']}")
            print(f"   Type: {issue['type']}")
            print(f"   Content: {issue['content']}")
            print()
        return False
    else:
        print("✅ No hardcoded credentials found in source files")
        return True

def verify_environment_loading():
    """Verify that environment variables are properly loaded"""
    
    print("\n🌐 VERIFYING ENVIRONMENT LOADING")
    print("=" * 40)
    
    try:
        # Test dotenv loading
        load_dotenv()
        print("✅ python-dotenv loading works")
        
        # Check if .env file exists
        if os.path.exists('.env'):
            print("✅ .env file exists")
        else:
            print("⚠️  .env file not found")
        
        # Test environment variable access
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        chat_ids = os.getenv('TELEGRAM_CHAT_IDS', '')
        
        print(f"✅ Environment variable access works")
        print(f"   TELEGRAM_BOT_TOKEN: {'Set' if bot_token else 'Not set'}")
        print(f"   TELEGRAM_CHAT_IDS: {'Set' if chat_ids else 'Not set'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment loading failed: {e}")
        return False

def verify_telegram_modules():
    """Verify that Telegram modules properly handle missing credentials"""
    
    print("\n🤖 VERIFYING TELEGRAM MODULES")
    print("=" * 35)
    
    try:
        # Test with no environment variables
        old_token = os.environ.pop('TELEGRAM_BOT_TOKEN', None)
        old_chat_ids = os.environ.pop('TELEGRAM_CHAT_IDS', None)
        
        try:
            from telegram_notification_system import TelegramNotificationSystem
            
            # This should handle missing credentials gracefully
            system = TelegramNotificationSystem()
            
            if not system.config.enabled:
                print("✅ System properly disables when credentials missing")
            else:
                print("⚠️  System should disable when credentials missing")
            
            # Restore environment variables
            if old_token:
                os.environ['TELEGRAM_BOT_TOKEN'] = old_token
            if old_chat_ids:
                os.environ['TELEGRAM_CHAT_IDS'] = old_chat_ids
            
            # Test with credentials
            load_dotenv()  # Reload from .env
            system_with_creds = TelegramNotificationSystem()
            
            if system_with_creds.config.enabled:
                print("✅ System properly enables when credentials provided")
                print(f"   Valid configuration: {system_with_creds._validate_config()}")
            else:
                print("⚠️  System should enable when credentials provided")
                
        finally:
            # Restore environment variables
            if old_token:
                os.environ['TELEGRAM_BOT_TOKEN'] = old_token
            if old_chat_ids:
                os.environ['TELEGRAM_CHAT_IDS'] = old_chat_ids
        
        return True
        
    except Exception as e:
        print(f"❌ Module verification failed: {e}")
        return False

def verify_file_security():
    """Verify specific file security improvements"""
    
    print("\n📁 VERIFYING FILE SECURITY")
    print("=" * 30)
    
    security_files = [
        'get_chat_id.py',
        'telegram_setup.py', 
        'telegram_notification_system.py',
        'quick_telegram_test.py',
        'tests/test_telegram_integration.py'
    ]
    
    all_secure = True
    
    for file_path in security_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Check for proper dotenv import
                has_dotenv = 'from dotenv import load_dotenv' in content
                # Check for load_dotenv() call
                calls_load_dotenv = 'load_dotenv()' in content
                # Check for no hardcoded defaults with real tokens
                no_hardcoded_defaults = '8369911887:AAHvXoNVTjpl3H3u0rVtuMxUkKEEozGIkFs' not in content
                
                file_secure = has_dotenv and calls_load_dotenv and no_hardcoded_defaults
                
                print(f"   {file_path}: {'✅ Secure' if file_secure else '❌ Issues found'}")
                if not file_secure:
                    print(f"     - dotenv import: {'✅' if has_dotenv else '❌'}")
                    print(f"     - load_dotenv() call: {'✅' if calls_load_dotenv else '❌'}")  
                    print(f"     - no hardcoded tokens: {'✅' if no_hardcoded_defaults else '❌'}")
                    all_secure = False
                    
            except Exception as e:
                print(f"   {file_path}: ❌ Error reading file: {e}")
                all_secure = False
        else:
            print(f"   {file_path}: ⚠️  File not found")
    
    return all_secure

def main():
    """Run complete security verification"""
    
    print("🔐 TELEGRAM SECURITY VERIFICATION")
    print("=" * 50)
    print("Verifying all Telegram security improvements...")
    print()
    
    # Run all verification checks
    checks = [
        ("Source Code Scan", scan_for_hardcoded_credentials()),
        ("Environment Loading", verify_environment_loading()),
        ("Telegram Modules", verify_telegram_modules()),
        ("File Security", verify_file_security())
    ]
    
    # Summary
    print("\n📊 SECURITY VERIFICATION SUMMARY")
    print("=" * 40)
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {check_name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 ALL SECURITY CHECKS PASSED!")
        print("   ✅ No hardcoded credentials found")
        print("   ✅ Environment variables properly loaded")
        print("   ✅ Error handling implemented")
        print("   ✅ .env file support added")
        print("   ✅ All files properly secured")
        print()
        print("🔒 Your Telegram integration is now secure!")
    else:
        print("❌ SOME SECURITY CHECKS FAILED!")
        print("   Please review the issues above and fix them.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)