#!/usr/bin/env python3
"""
API Key Validator and Setup Helper
Validates API keys and provides clear setup instructions when keys are missing
"""

import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, using system environment variables

logger = logging.getLogger(__name__)


class APIKeyStatus(Enum):
    """API key status enumeration"""
    CONFIGURED = "configured"
    MISSING = "missing"
    INVALID = "invalid"
    UNTESTED = "untested"


@dataclass
class APIKeyValidation:
    """API key validation result"""
    key_name: str
    status: APIKeyStatus
    message: str
    is_critical: bool = False
    setup_instructions: str = ""


class APIKeyValidator:
    """Validates API keys and provides setup guidance"""
    
    def __init__(self):
        self.required_keys = {
            'API_TENNIS_KEY': {
                'description': 'API-Tennis.com key for live tennis data',
                'critical': True,
                'setup_url': 'https://api-tennis.com/',
                'test_method': self._test_api_tennis_key
            },
            'RAPIDAPI_KEY': {
                'description': 'RapidAPI key for backup tennis data',
                'critical': False,
                'setup_url': 'https://rapidapi.com/',
                'test_method': None  # No test method available
            },
            'TENNIS_API_KEY': {
                'description': 'Alternative tennis API key for redundancy',
                'critical': False,
                'setup_url': 'https://tennisdata.net/',
                'test_method': None  # No test method available
            },
            'BETFAIR_APP_KEY': {
                'description': 'Betfair API key for live betting',
                'critical': False,
                'setup_url': 'https://developer.betfair.com/',
                'test_method': None  # Requires complex setup
            }
        }
    
    def validate_all_keys(self) -> Dict[str, APIKeyValidation]:
        """Validate all configured API keys"""
        validations = {}
        
        for key_name, config in self.required_keys.items():
            validation = self._validate_single_key(key_name, config)
            validations[key_name] = validation
            
        return validations
    
    def _validate_single_key(self, key_name: str, config: Dict) -> APIKeyValidation:
        """Validate a single API key"""
        key_value = os.getenv(key_name, '').strip()
        
        if not key_value:
            return APIKeyValidation(
                key_name=key_name,
                status=APIKeyStatus.MISSING,
                message=f"{key_name} not configured",
                is_critical=config['critical'],
                setup_instructions=self._generate_setup_instructions(key_name, config)
            )
        
        # Test the key if test method available
        if config.get('test_method'):
            try:
                if config['test_method'](key_value):
                    return APIKeyValidation(
                        key_name=key_name,
                        status=APIKeyStatus.CONFIGURED,
                        message=f"{key_name} configured and tested successfully"
                    )
                else:
                    return APIKeyValidation(
                        key_name=key_name,
                        status=APIKeyStatus.INVALID,
                        message=f"{key_name} configured but failed validation test",
                        is_critical=config['critical']
                    )
            except Exception as e:
                logger.error(f"Error testing {key_name}: {e}")
                return APIKeyValidation(
                    key_name=key_name,
                    status=APIKeyStatus.CONFIGURED,
                    message=f"{key_name} configured but test failed: {str(e)}"
                )
        else:
            # Key is configured but we can't test it
            return APIKeyValidation(
                key_name=key_name,
                status=APIKeyStatus.CONFIGURED,
                message=f"{key_name} configured (not tested)"
            )
    
    def _test_api_tennis_key(self, api_key: str) -> bool:
        """Test API Tennis key with a simple request"""
        try:
            import requests
            
            url = "https://api.api-tennis.com/tennis/"
            params = {
                'method': 'get_events',
                'APIkey': api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                # API Tennis returns success: 1 for valid requests
                return isinstance(data, list) or (isinstance(data, dict) and data.get('success') == 1)
            
            return False
            
        except Exception as e:
            logger.warning(f"API Tennis key test failed: {e}")
            return False
    
    def _generate_setup_instructions(self, key_name: str, config: Dict) -> str:
        """Generate setup instructions for missing API key"""
        instructions = f"""
To configure {key_name}:

1. Visit: {config['setup_url']}
2. Create an account and get your API key
3. Add the key to your .env file:
   {key_name}=your_api_key_here
4. Restart the application

Description: {config['description']}
Critical: {'Yes' if config['critical'] else 'No (optional)'}
"""
        return instructions.strip()
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of API key validation results"""
        validations = self.validate_all_keys()
        
        summary = {
            'total_keys': len(validations),
            'configured': 0,
            'missing': 0,
            'invalid': 0,
            'critical_missing': 0,
            'system_ready': True,
            'warnings': [],
            'errors': []
        }
        
        for validation in validations.values():
            if validation.status == APIKeyStatus.CONFIGURED:
                summary['configured'] += 1
            elif validation.status == APIKeyStatus.MISSING:
                summary['missing'] += 1
                if validation.is_critical:
                    summary['critical_missing'] += 1
                    summary['system_ready'] = False
                    summary['errors'].append(validation.message)
                else:
                    summary['warnings'].append(validation.message)
            elif validation.status == APIKeyStatus.INVALID:
                summary['invalid'] += 1
                if validation.is_critical:
                    summary['system_ready'] = False
                    summary['errors'].append(validation.message)
                else:
                    summary['warnings'].append(validation.message)
        
        return summary
    
    def print_validation_report(self):
        """Print a comprehensive validation report"""
        validations = self.validate_all_keys()
        summary = self.get_validation_summary()
        
        print("\n🔑 API Key Validation Report")
        print("=" * 50)
        
        # Overall status
        status_icon = "✅" if summary['system_ready'] else "⚠️"
        status_text = "READY" if summary['system_ready'] else "NEEDS ATTENTION"
        print(f"{status_icon} System Status: {status_text}")
        
        # Summary stats
        print(f"\n📊 Summary:")
        print(f"   Configured: {summary['configured']}/{summary['total_keys']}")
        print(f"   Missing: {summary['missing']} (Critical: {summary['critical_missing']})")
        print(f"   Invalid: {summary['invalid']}")
        
        # Detailed results
        print(f"\n📋 Detailed Results:")
        for key_name, validation in validations.items():
            status_icon = {
                APIKeyStatus.CONFIGURED: "✅",
                APIKeyStatus.MISSING: "❌" if validation.is_critical else "⚠️",
                APIKeyStatus.INVALID: "❌",
                APIKeyStatus.UNTESTED: "🔶"
            }.get(validation.status, "❓")
            
            criticality = " (CRITICAL)" if validation.is_critical else ""
            print(f"   {status_icon} {key_name}{criticality}: {validation.message}")
        
        # Show setup instructions for missing critical keys
        missing_critical = [v for v in validations.values() 
                          if v.status == APIKeyStatus.MISSING and v.is_critical]
        
        if missing_critical:
            print(f"\n🛠️ Setup Instructions for Critical Keys:")
            for validation in missing_critical:
                print(f"\n{validation.setup_instructions}")
        
        # Show warnings
        if summary['warnings']:
            print(f"\n⚠️ Warnings:")
            for warning in summary['warnings']:
                print(f"   • {warning}")
        
        # Show errors
        if summary['errors']:
            print(f"\n❌ Errors:")
            for error in summary['errors']:
                print(f"   • {error}")
    
    def ensure_system_ready(self) -> bool:
        """Ensure system is ready with proper API keys"""
        summary = self.get_validation_summary()
        
        if not summary['system_ready']:
            self.print_validation_report()
            return False
        
        # Log successful validation
        logger.info(f"API key validation successful: {summary['configured']}/{summary['total_keys']} keys configured")
        
        return True


# Global validator instance
api_key_validator = APIKeyValidator()


def validate_api_keys() -> Dict[str, APIKeyValidation]:
    """Convenience function to validate API keys"""
    return api_key_validator.validate_all_keys()


def ensure_api_keys_ready() -> bool:
    """Convenience function to ensure system is ready"""
    return api_key_validator.ensure_system_ready()


def print_api_key_report():
    """Convenience function to print validation report"""
    api_key_validator.print_validation_report()


if __name__ == "__main__":
    # Run validation when called directly
    print_api_key_report()