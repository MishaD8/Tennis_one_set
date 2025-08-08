#!/usr/bin/env python3
"""
🧪 TEST SCRIPT FOR SECURE TOURNAMENT FILTERING INTEGRATION

This script tests the integration of secure tournament filtering with the
existing tennis data collection system, specifically verifying CLAUDE.md
requirement #3 implementation.

Test Coverage:
- SecureTournamentFilter functionality
- TournamentFilterIntegration with data collectors
- ComprehensiveMLDataCollector integration
- Grand Slam exclusion validation
- Best-of-3 format enforcement

Author: Claude Code (Anthropic) - Backend Security Specialist
"""

import sys
import os
import logging
from datetime import datetime
from typing import Dict, List, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modules for testing
from secure_tournament_filter import SecureTournamentFilter, SecurityLevel, create_strict_filter
from tournament_filter_integration import TournamentFilterIntegration, create_secure_integration
from comprehensive_ml_data_collector import ComprehensiveMLDataCollector


def test_secure_tournament_filter():
    """Test core secure tournament filter functionality"""
    print("🔒 Testing SecureTournamentFilter...")
    
    filter_system = create_strict_filter()
    
    # Test tournaments - mix of approved and Grand Slams
    test_tournaments = [
        # Should be APPROVED (best-of-3 format)
        {"tournament": "Miami Open", "level": "ATP 1000", "category": "Masters"},
        {"tournament": "Rome Masters", "level": "ATP 1000", "category": "Masters"}, 
        {"tournament": "Cincinnati Open", "level": "ATP 1000", "category": "Masters"},
        {"tournament": "Barcelona Open", "level": "ATP 500", "category": "ATP"},
        {"tournament": "Queen's Club", "level": "ATP 250", "category": "ATP"},
        {"tournament": "Miami Open", "level": "WTA 1000", "category": "WTA"},
        
        # Should be REJECTED (best-of-5 format - Grand Slams)
        {"tournament": "Australian Open", "level": "Grand Slam", "category": "Major"},
        {"tournament": "French Open", "level": "Grand Slam", "category": "Major"},
        {"tournament": "Wimbledon", "level": "Grand Slam", "category": "Major"},
        {"tournament": "US Open", "level": "Grand Slam", "category": "Major"},
    ]
    
    approved_count = 0
    rejected_count = 0
    grand_slam_rejected = 0
    
    for tournament in test_tournaments:
        is_valid, result = filter_system.is_best_of_3_tournament(tournament)
        tournament_name = tournament.get('tournament', 'Unknown')
        
        if is_valid:
            approved_count += 1
            print(f"  ✅ {tournament_name} - APPROVED ({result.get('reason', 'N/A')})")
        else:
            rejected_count += 1
            reason = result.get('reason', 'N/A')
            if reason == 'GRAND_SLAM_EXCLUDED':
                grand_slam_rejected += 1
            print(f"  ❌ {tournament_name} - REJECTED ({reason})")
    
    print(f"\n📊 Filter Test Results:")
    print(f"  Total tournaments: {len(test_tournaments)}")
    print(f"  Approved (best-of-3): {approved_count}")
    print(f"  Rejected total: {rejected_count}")
    print(f"  Grand Slams rejected: {grand_slam_rejected}")
    
    # Verify all Grand Slams were rejected
    expected_grand_slams = 4
    if grand_slam_rejected == expected_grand_slams:
        print("  ✅ All Grand Slams properly excluded")
        return True
    else:
        print(f"  ❌ Grand Slam filtering failed: {grand_slam_rejected}/{expected_grand_slams}")
        return False


def test_integration_functionality():
    """Test tournament filter integration"""
    print("\n🔗 Testing TournamentFilterIntegration...")
    
    integration = create_secure_integration(SecurityLevel.STRICT)
    
    test_matches = [
        {
            "id": "match_1",
            "tournament": "Indian Wells",
            "level": "ATP 1000",
            "player1": "Player A",
            "player2": "Player B"
        },
        {
            "id": "match_2", 
            "tournament": "Australian Open",
            "level": "Grand Slam",
            "player1": "Player C",
            "player2": "Player D"
        },
        {
            "id": "match_3",
            "tournament": "Madrid Open",
            "level": "ATP 1000", 
            "player1": "Player E",
            "player2": "Player F"
        },
        {
            "id": "match_4",
            "tournament": "Wimbledon",
            "level": "Grand Slam",
            "player1": "Player G",
            "player2": "Player H"
        }
    ]
    
    # Test batch filtering
    filter_result = integration.filter_tournaments_secure(test_matches)
    
    if filter_result['filter_status'] == 'SUCCESS':
        approved_tournaments = filter_result['tournaments']
        print(f"  ✅ Batch filtering successful")
        print(f"  📊 Results: {len(approved_tournaments)}/{len(test_matches)} approved")
        
        # Verify no Grand Slams in approved list
        grand_slams_found = []
        for tournament in approved_tournaments:
            tournament_name = tournament.get('tournament', '').lower()
            if any(gs in tournament_name for gs in ['australian open', 'french open', 'wimbledon', 'us open']):
                grand_slams_found.append(tournament_name)
        
        if not grand_slams_found:
            print("  ✅ No Grand Slams in approved tournaments")
            return True
        else:
            print(f"  ❌ Grand Slams found in approved list: {grand_slams_found}")
            return False
    else:
        print(f"  ❌ Batch filtering failed: {filter_result.get('error', 'Unknown error')}")
        return False


def test_data_collector_integration():
    """Test integration with ComprehensiveMLDataCollector"""
    print("\n🗃️ Testing ComprehensiveMLDataCollector integration...")
    
    try:
        # Create collector instance (this should have secure filtering integrated)
        collector = ComprehensiveMLDataCollector()
        
        # Verify secure filtering components are initialized
        if hasattr(collector, 'tournament_filter_integration'):
            print("  ✅ Tournament filter integration initialized")
        else:
            print("  ❌ Tournament filter integration not found")
            return False
        
        if hasattr(collector, 'secure_tournament_validator'):
            print("  ✅ Secure tournament validator available")
        else:
            print("  ❌ Secure tournament validator not found")
            return False
        
        # Test validator function
        validator = collector.secure_tournament_validator
        
        # Test with approved tournament
        approved_tournament = {
            "tournament": "Rome Masters",
            "level": "ATP 1000",
            "category": "Masters"
        }
        
        if validator(approved_tournament):
            print("  ✅ Validator approves best-of-3 tournament")
        else:
            print("  ❌ Validator incorrectly rejects best-of-3 tournament")
            return False
        
        # Test with Grand Slam
        grand_slam_tournament = {
            "tournament": "French Open", 
            "level": "Grand Slam",
            "category": "Major"
        }
        
        if not validator(grand_slam_tournament):
            print("  ✅ Validator correctly rejects Grand Slam")
            return True
        else:
            print("  ❌ Validator incorrectly approves Grand Slam")
            return False
            
    except Exception as e:
        print(f"  ❌ Data collector integration test failed: {e}")
        return False


def test_security_edge_cases():
    """Test security edge cases and potential bypasses"""
    print("\n🛡️ Testing security edge cases...")
    
    filter_system = create_strict_filter()
    
    # Test potential bypass attempts
    edge_cases = [
        # Variation attempts of Grand Slam names
        {"tournament": "AUSTRALIAN OPEN", "level": "tournament", "category": "tennis"},
        {"tournament": "australian-open", "level": "atp", "category": "professional"},
        {"tournament": "AO 2025", "level": "major", "category": "grand slam"},
        {"tournament": "French_Open", "level": "rg", "category": "clay"},
        {"tournament": "WIMBLEDON CHAMPIONSHIPS", "level": "grass", "category": "major"},
        {"tournament": "us-open-2025", "level": "hard", "category": "usopen"},
        {"tournament": "Roland Garros 2025", "level": "clay", "category": "major"},
        
        # Suspicious input patterns (should be sanitized)
        {"tournament": "Test<script>alert('xss')</script>", "level": "test", "category": "test"},
        {"tournament": "Test\x00Tournament", "level": "test", "category": "test"},
        
        # Valid tournaments that should pass
        {"tournament": "Miami Open", "level": "ATP 1000", "category": "Masters"},
        {"tournament": "Barcelona Open", "level": "ATP 500", "category": "ATP"}
    ]
    
    grand_slams_blocked = 0
    suspicious_blocked = 0
    valid_approved = 0
    
    for test_case in edge_cases:
        is_valid, result = filter_system.is_best_of_3_tournament(test_case)
        tournament_name = test_case.get('tournament', 'Unknown')
        reason = result.get('reason', 'N/A')
        
        # Check if Grand Slam variants are blocked
        if any(gs in tournament_name.lower().replace('-', ' ').replace('_', ' ') 
               for gs in ['australian open', 'french open', 'wimbledon', 'us open', 'roland garros']):
            if not is_valid and reason in ['GRAND_SLAM_EXCLUDED', 'INPUT_VALIDATION_FAILED']:
                grand_slams_blocked += 1
                print(f"  ✅ Grand Slam variant blocked: {tournament_name}")
            else:
                print(f"  ❌ Grand Slam variant not blocked: {tournament_name}")
        
        # Check if suspicious input is blocked
        elif any(char in tournament_name for char in ['<', '>', '\x00', 'script']):
            if not is_valid and reason == 'INPUT_VALIDATION_FAILED':
                suspicious_blocked += 1
                print(f"  ✅ Suspicious input blocked: {tournament_name[:20]}...")
            else:
                print(f"  ❌ Suspicious input not blocked: {tournament_name[:20]}...")
        
        # Check if valid tournaments are approved
        elif tournament_name in ['Miami Open', 'Barcelona Open']:
            if is_valid:
                valid_approved += 1
                print(f"  ✅ Valid tournament approved: {tournament_name}")
            else:
                print(f"  ❌ Valid tournament rejected: {tournament_name}")
    
    print(f"\n📊 Security Test Results:")
    print(f"  Grand Slam variants blocked: {grand_slams_blocked}")
    print(f"  Suspicious inputs blocked: {suspicious_blocked}")
    print(f"  Valid tournaments approved: {valid_approved}")
    
    return grand_slams_blocked >= 5 and suspicious_blocked >= 2 and valid_approved >= 2


def main():
    """Run all integration tests"""
    print("🧪 SECURE TOURNAMENT FILTERING - INTEGRATION TESTS")
    print("=" * 70)
    print("Testing CLAUDE.md requirement #3 implementation:")
    print("'Use only best-of-3 sets format, exclude Grand Slams (best-of-5)'")
    print("=" * 70)
    
    test_results = []
    
    # Run all tests
    test_functions = [
        ("Core Filter Functionality", test_secure_tournament_filter),
        ("Integration Functionality", test_integration_functionality), 
        ("Data Collector Integration", test_data_collector_integration),
        ("Security Edge Cases", test_security_edge_cases)
    ]
    
    for test_name, test_func in test_functions:
        print(f"\n🔬 Running: {test_name}")
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            test_results.append((test_name, False))
    
    # Summary
    print(f"\n" + "=" * 70)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, passed in test_results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status} - {test_name}")
        if passed:
            passed_tests += 1
    
    print(f"\n📊 Overall Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("✅ CLAUDE.md requirement #3 successfully implemented")
        print("✅ Grand Slam tournaments (best-of-5) are excluded") 
        print("✅ Only best-of-3 format tournaments are approved")
        print("✅ Security filtering and validation working correctly")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        print("⚠️ Review implementation and fix failing components")
        return 1


if __name__ == "__main__":
    sys.exit(main())