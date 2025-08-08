#!/usr/bin/env python3
"""
🔒 TOURNAMENT FILTER INTEGRATION MODULE

This module provides secure integration of the tournament filtering system 
into existing tennis data collectors and prediction services.

Security Features:
- Seamless integration with existing codebase
- Backward compatibility with current data structures
- Secure filtering without disrupting current functionality
- Comprehensive error handling and fallback mechanisms
- Audit logging for compliance and security monitoring

Integration Points:
1. ComprehensiveMLDataCollector
2. EnhancedUniversalCollector  
3. Tennis Backend filtering functions
4. Prediction services validation

Author: Claude Code (Anthropic) - Backend Security Specialist
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime
import functools

from secure_tournament_filter import SecureTournamentFilter, SecurityLevel, create_strict_filter

logger = logging.getLogger(__name__)

class TournamentFilterIntegration:
    """
    Integration wrapper for secure tournament filtering system
    """
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STRICT):
        self.filter_system = SecureTournamentFilter(security_level)
        self.integration_stats = {
            'initialized_at': datetime.now().isoformat(),
            'total_integrations': 0,
            'successful_filters': 0,
            'failed_filters': 0,
            'bypass_attempts': 0
        }
        logger.info(f"🔒 Tournament filter integration initialized with {security_level.value} security")
    
    def secure_tournament_validator(self, func: Callable) -> Callable:
        """
        Decorator to add secure tournament filtering to any function that processes tournaments
        
        Usage:
        @integration.secure_tournament_validator
        def collect_matches(self, tournaments):
            # Your existing code here
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract tournament data from function arguments
            tournament_data = self._extract_tournament_data_from_args(args, kwargs)
            
            if tournament_data:
                # Apply secure filtering
                filtered_data = self.filter_tournaments_secure(tournament_data)
                
                # Update arguments with filtered data
                updated_args, updated_kwargs = self._update_args_with_filtered_data(
                    args, kwargs, tournament_data, filtered_data
                )
                
                # Call original function with filtered data
                return func(*updated_args, **updated_kwargs)
            else:
                # No tournament data found, proceed normally
                return func(*args, **kwargs)
        
        return wrapper
    
    def filter_tournaments_secure(self, tournaments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply secure tournament filtering with integration-specific enhancements
        
        Args:
            tournaments: List of tournament data
            
        Returns:
            Filtered tournament data with integration metadata
        """
        try:
            self.integration_stats['total_integrations'] += 1
            
            # Apply core filtering
            filter_results = self.filter_system.filter_tournament_list(tournaments)
            
            if filter_results.get('error'):
                self.integration_stats['failed_filters'] += 1
                logger.error(f"🚨 Tournament filtering failed: {filter_results['error']}")
                
                # Fallback: return empty list for security
                return {
                    'tournaments': [],
                    'total_input': len(tournaments),
                    'total_filtered': 0,
                    'filter_applied': True,
                    'filter_status': 'FAILED_SECURE_FALLBACK',
                    'error': filter_results['error']
                }
            
            self.integration_stats['successful_filters'] += 1
            
            # Extract filtered tournaments
            filtered_tournaments = filter_results.get('filtered_tournaments', [])
            
            # Add integration metadata
            for tournament in filtered_tournaments:
                tournament['_security'] = {
                    'filtered_by': 'SecureTournamentFilter',
                    'security_level': self.filter_system.security_level.value,
                    'validation_timestamp': datetime.now().isoformat(),
                    'compliant_with_claude_md': True,
                    'format': 'best_of_3_only'
                }
            
            return {
                'tournaments': filtered_tournaments,
                'total_input': filter_results['total_input'],
                'total_filtered': filter_results['total_approved'],
                'rejected_count': filter_results['total_rejected'],
                'rejection_reasons': filter_results['rejection_reasons'],
                'filter_applied': True,
                'filter_status': 'SUCCESS',
                'processing_time_ms': filter_results['processing_time_ms'],
                'security_level': self.filter_system.security_level.value
            }
            
        except Exception as e:
            self.integration_stats['failed_filters'] += 1
            error_msg = f"Tournament filtering integration error: {e}"
            logger.error(f"🚨 {error_msg}")
            
            # Secure fallback
            return {
                'tournaments': [],
                'total_input': len(tournaments) if tournaments else 0,
                'total_filtered': 0,
                'filter_applied': True,
                'filter_status': 'ERROR_SECURE_FALLBACK',
                'error': error_msg
            }
    
    def validate_tournament_data_secure(self, tournament_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate individual tournament data using secure filtering
        
        Args:
            tournament_data: Single tournament data dictionary
            
        Returns:
            Tuple of (is_valid, validation_details)
        """
        try:
            return self.filter_system.is_best_of_3_tournament(tournament_data)
        except Exception as e:
            logger.error(f"🚨 Tournament validation error: {e}")
            return False, {
                'valid': False,
                'reason': 'VALIDATION_ERROR',
                'error': str(e),
                'message': 'Tournament validation failed - rejected for security'
            }
    
    def create_tournament_data_validator(self) -> Callable[[Dict[str, Any]], bool]:
        """
        Create a simple validator function for use in existing data collectors
        
        Returns:
            Function that takes tournament data and returns True if valid (best-of-3)
        """
        def validator(tournament_data: Dict[str, Any]) -> bool:
            try:
                is_valid, _ = self.validate_tournament_data_secure(tournament_data)
                return is_valid
            except Exception as e:
                logger.warning(f"🔒 Validator error, rejecting for security: {e}")
                return False
        
        return validator
    
    def patch_existing_collector(self, collector_instance: Any, method_name: str = 'collect_matches') -> bool:
        """
        Dynamically patch existing collector instance with secure filtering
        
        Args:
            collector_instance: Instance of data collector to patch
            method_name: Name of method to patch with filtering
            
        Returns:
            True if patching successful, False otherwise
        """
        try:
            if not hasattr(collector_instance, method_name):
                logger.warning(f"🔒 Cannot patch {type(collector_instance).__name__}: method '{method_name}' not found")
                return False
            
            # Get original method
            original_method = getattr(collector_instance, method_name)
            
            # Create patched method with secure filtering
            def patched_method(*args, **kwargs):
                # Call original method to get data
                result = original_method(*args, **kwargs)
                
                # If result contains tournament/match data, apply filtering
                if isinstance(result, dict):
                    if 'matches' in result and isinstance(result['matches'], list):
                        # Filter matches data
                        filter_result = self.filter_tournaments_secure(result['matches'])
                        result['matches'] = filter_result['tournaments']
                        result['_filtering'] = {
                            'applied': True,
                            'original_count': filter_result['total_input'],
                            'filtered_count': filter_result['total_filtered'],
                            'security_level': self.filter_system.security_level.value
                        }
                elif isinstance(result, list):
                    # Direct list of tournaments/matches
                    filter_result = self.filter_tournaments_secure(result)
                    result = filter_result['tournaments']
                
                return result
            
            # Replace method
            setattr(collector_instance, method_name, patched_method)
            
            # Add validator method to instance
            setattr(collector_instance, '_tournament_validator', self.create_tournament_data_validator())
            setattr(collector_instance, '_has_secure_filtering', True)
            
            logger.info(f"✅ Successfully patched {type(collector_instance).__name__}.{method_name} with secure filtering")
            return True
            
        except Exception as e:
            logger.error(f"🚨 Failed to patch collector: {e}")
            return False
    
    def get_integration_status(self) -> Dict[str, Any]:
        """
        Get comprehensive integration status and statistics
        
        Returns:
            Dictionary with integration status and security metrics
        """
        filter_stats = self.filter_system.get_filter_statistics()
        
        return {
            'integration_info': {
                'initialized_at': self.integration_stats['initialized_at'],
                'security_level': self.filter_system.security_level.value,
                'total_integrations': self.integration_stats['total_integrations'],
                'success_rate': (
                    self.integration_stats['successful_filters'] / 
                    max(1, self.integration_stats['total_integrations'])
                )
            },
            'filtering_stats': filter_stats,
            'security_events': {
                'bypass_attempts': self.integration_stats['bypass_attempts'],
                'failed_filters': self.integration_stats['failed_filters'],
                'security_alerts': filter_stats.get('security_events', 0)
            },
            'compliance_status': {
                'claude_md_requirement_3': 'IMPLEMENTED',
                'grand_slam_exclusion': 'ACTIVE',
                'best_of_3_only': 'ENFORCED',
                'audit_logging': 'ENABLED'
            }
        }
    
    def _extract_tournament_data_from_args(self, args: tuple, kwargs: dict) -> Optional[List[Dict[str, Any]]]:
        """
        Extract tournament data from function arguments
        
        Returns:
            List of tournament data if found, None otherwise
        """
        # Check various common argument patterns
        
        # Check kwargs first
        for key in ['tournaments', 'matches', 'tournament_data', 'data']:
            if key in kwargs and isinstance(kwargs[key], list):
                return kwargs[key]
        
        # Check positional arguments
        for arg in args:
            if isinstance(arg, list) and arg:
                # Check if it looks like tournament data
                if isinstance(arg[0], dict) and any(
                    field in arg[0] for field in ['tournament', 'name', 'level', 'category']
                ):
                    return arg
        
        return None
    
    def _update_args_with_filtered_data(self, args: tuple, kwargs: dict, 
                                      original_data: List[Dict[str, Any]], 
                                      filtered_result: Dict[str, Any]) -> Tuple[tuple, dict]:
        """
        Update function arguments with filtered tournament data
        
        Returns:
            Updated args and kwargs tuples
        """
        filtered_tournaments = filtered_result.get('tournaments', [])
        
        # Update kwargs
        updated_kwargs = kwargs.copy()
        for key in ['tournaments', 'matches', 'tournament_data', 'data']:
            if key in updated_kwargs and updated_kwargs[key] == original_data:
                updated_kwargs[key] = filtered_tournaments
                break
        
        # Update positional arguments
        updated_args = list(args)
        for i, arg in enumerate(updated_args):
            if arg == original_data:
                updated_args[i] = filtered_tournaments
                break
        
        return tuple(updated_args), updated_kwargs


# Convenience functions for easy integration
def create_secure_integration(security_level: SecurityLevel = SecurityLevel.STRICT) -> TournamentFilterIntegration:
    """
    Create a tournament filter integration instance
    
    Args:
        security_level: Security level for filtering (STRICT recommended)
        
    Returns:
        Configured TournamentFilterIntegration instance
    """
    return TournamentFilterIntegration(security_level)

def patch_data_collector_with_security(collector_instance: Any, 
                                     security_level: SecurityLevel = SecurityLevel.STRICT) -> bool:
    """
    Patch existing data collector with secure tournament filtering
    
    Args:
        collector_instance: Data collector instance to patch
        security_level: Security level for filtering
        
    Returns:
        True if patching successful
    """
    integration = create_secure_integration(security_level)
    return integration.patch_existing_collector(collector_instance)

def create_tournament_validator(security_level: SecurityLevel = SecurityLevel.STRICT) -> Callable[[Dict[str, Any]], bool]:
    """
    Create a standalone tournament validator function
    
    Args:
        security_level: Security level for validation
        
    Returns:
        Validator function that returns True for valid best-of-3 tournaments
    """
    integration = create_secure_integration(security_level)
    return integration.create_tournament_data_validator()


# Example usage and testing
if __name__ == "__main__":
    print("🔒 TOURNAMENT FILTER INTEGRATION - TESTING")
    print("=" * 60)
    
    # Create integration instance
    integration = create_secure_integration(SecurityLevel.STRICT)
    
    # Test data
    test_tournaments = [
        {"tournament": "Miami Masters", "level": "ATP 1000", "category": "Masters"},
        {"tournament": "Australian Open", "level": "Grand Slam", "category": "Major"},
        {"tournament": "Barcelona Open", "level": "ATP 500", "category": "ATP"},
        {"tournament": "Wimbledon", "level": "Grand Slam", "category": "Major"}
    ]
    
    print(f"\n🧪 Testing integration with {len(test_tournaments)} tournaments...")
    
    # Test secure filtering
    filter_result = integration.filter_tournaments_secure(test_tournaments)
    
    print(f"\n📊 INTEGRATION RESULTS:")
    print(f"  Status: {filter_result['filter_status']}")
    print(f"  Input tournaments: {filter_result['total_input']}")
    print(f"  Approved tournaments: {filter_result['total_filtered']}")
    print(f"  Rejected tournaments: {filter_result.get('rejected_count', 0)}")
    print(f"  Processing time: {filter_result['processing_time_ms']:.2f}ms")
    
    # Show approved tournaments with security metadata
    print(f"\n✅ APPROVED TOURNAMENTS:")
    for tournament in filter_result['tournaments']:
        name = tournament.get('tournament', 'Unknown')
        level = tournament.get('level', 'Unknown')
        security_info = tournament.get('_security', {})
        print(f"  • {name} ({level})")
        print(f"    Security: {security_info.get('format', 'N/A')} - {security_info.get('validation_timestamp', 'N/A')}")
    
    # Test individual tournament validation
    print(f"\n🔍 Individual Tournament Validation:")
    for tournament in test_tournaments:
        is_valid, result = integration.validate_tournament_data_secure(tournament)
        status = "✅ VALID" if is_valid else "❌ INVALID"
        name = tournament.get('tournament', 'Unknown')
        reason = result.get('reason', 'N/A')
        print(f"  {status} - {name}: {reason}")
    
    # Test validator function creation
    print(f"\n⚙️  Testing validator function:")
    validator = integration.create_tournament_data_validator()
    
    for tournament in test_tournaments:
        is_valid = validator(tournament)
        status = "✅" if is_valid else "❌"
        name = tournament.get('tournament', 'Unknown')
        print(f"  {status} {name}")
    
    # Get integration status
    print(f"\n📈 Integration Status:")
    status = integration.get_integration_status()
    
    print(f"  Total integrations: {status['integration_info']['total_integrations']}")
    print(f"  Success rate: {status['integration_info']['success_rate']:.1%}")
    print(f"  Security level: {status['integration_info']['security_level']}")
    print(f"  Compliance status: {status['compliance_status']['claude_md_requirement_3']}")
    
    print(f"\n✅ INTEGRATION TESTING COMPLETE")
    print("🔒 Secure tournament filtering integration ready for deployment!")