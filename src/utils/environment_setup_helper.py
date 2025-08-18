#!/usr/bin/env python3
"""
Environment Setup Helper for Tennis Betting System
Checks and helps configure missing API keys and environment variables
"""

import os
import sys
import logging
from typing import Dict, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class EnvironmentSetupHelper:
    """Helper class to validate and setup environment variables"""
    
    REQUIRED_VARS = {
        # Critical API Keys for betting functionality
        'BETFAIR_APP_KEY': 'Betfair Exchange API key for automated betting',
        'BETFAIR_USERNAME': 'Betfair username for authentication',
        'BETFAIR_PASSWORD': 'Betfair password for authentication',
        
        # Optional but recommended API keys
        'TENNIS_API_KEY': 'Tennis-specific API key (optional)',
        'RAPIDAPI_KEY': 'RapidAPI key for tennis data sources',
        'API_TENNIS_KEY': 'API-Tennis.com key for enhanced tennis data',
        
        # Database configuration
        'DATABASE_URL': 'Database connection string',
        
        # Redis configuration (optional - has fallback)
        'REDIS_URL': 'Redis connection URL for caching',
        
        # Flask configuration
        'FLASK_SECRET_KEY': 'Flask application secret key',
        'FLASK_ENV': 'Flask environment (development/production)',
        
        # Security configuration
        'ALLOWED_ORIGINS': 'CORS allowed origins',
        'TRUSTED_PROXIES': 'Trusted proxy IPs',
    }
    
    OPTIONAL_VARS = {
        'ML_MODEL_PATH': ('./tennis_models/', 'Path to ML model files'),
        'DEFAULT_STAKE': ('10.0', 'Default betting stake amount'),
        'MAX_STAKE': ('100.0', 'Maximum betting stake amount'),
        'PREDICTION_CONFIDENCE_THRESHOLD': ('0.6', 'ML prediction confidence threshold'),
        'DAILY_API_LIMIT': ('8', 'Daily API request limit'),
        'MONTHLY_API_LIMIT': ('500', 'Monthly API request limit'),
        'RISK_MANAGEMENT_ENABLED': ('true', 'Enable risk management'),
    }
    
    def __init__(self):
        self.missing_vars = []
        self.present_vars = []
        self.warnings = []
    
    def check_environment(self) -> Dict[str, any]:
        """Check all environment variables and return status"""
        self.missing_vars = []
        self.present_vars = []
        self.warnings = []
        
        # Check required variables
        for var_name, description in self.REQUIRED_VARS.items():
            value = os.getenv(var_name)
            
            if not value or value.strip() == '':
                self.missing_vars.append({
                    'name': var_name,
                    'description': description,
                    'critical': var_name.startswith('BETFAIR_')
                })
            else:
                self.present_vars.append({
                    'name': var_name,
                    'description': description,
                    'value_length': len(value) if value else 0,
                    'masked_value': self._mask_sensitive_value(var_name, value)
                })
        
        # Check optional variables and set defaults
        for var_name, (default_value, description) in self.OPTIONAL_VARS.items():
            value = os.getenv(var_name)
            if not value:
                self.warnings.append({
                    'name': var_name,
                    'description': description,
                    'default_value': default_value,
                    'severity': 'info'
                })
        
        # Special checks
        self._perform_special_checks()
        
        return {
            'missing_vars': self.missing_vars,
            'present_vars': self.present_vars,
            'warnings': self.warnings,
            'total_missing': len(self.missing_vars),
            'critical_missing': len([v for v in self.missing_vars if v.get('critical', False)]),
            'environment_ready': len([v for v in self.missing_vars if v.get('critical', False)]) == 0
        }
    
    def _mask_sensitive_value(self, var_name: str, value: str) -> str:
        """Mask sensitive values for display"""
        sensitive_vars = ['PASSWORD', 'SECRET', 'KEY', 'TOKEN']
        
        if any(sensitive in var_name for sensitive in sensitive_vars):
            if len(value) <= 4:
                return '*' * len(value)
            else:
                return value[:2] + '*' * (len(value) - 4) + value[-2:]
        else:
            return value[:50] + '...' if len(value) > 50 else value
    
    def _perform_special_checks(self):
        """Perform special environment checks"""
        
        # Check Redis connectivity
        redis_url = os.getenv('REDIS_URL', '').strip()
        if redis_url and redis_url != 'memory://':
            try:
                import redis
                if redis_url.startswith('redis://'):
                    r = redis.Redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2)
                else:
                    r = redis.Redis(host='localhost', port=6379, socket_connect_timeout=2, socket_timeout=2)
                r.ping()
            except Exception as e:
                self.warnings.append({
                    'name': 'REDIS_CONNECTIVITY',
                    'description': f'Redis connection test failed: {str(e)}',
                    'severity': 'warning'
                })
        
        # Check ML model path
        ml_path = os.getenv('ML_MODEL_PATH', './tennis_models/')
        if not os.path.exists(ml_path):
            self.warnings.append({
                'name': 'ML_MODEL_PATH',
                'description': f'ML model directory does not exist: {ml_path}',
                'severity': 'error'
            })
        else:
            # Check for model files
            model_files = [f for f in os.listdir(ml_path) if f.endswith(('.pkl', '.h5', '.json'))]
            if len(model_files) == 0:
                self.warnings.append({
                    'name': 'ML_MODELS',
                    'description': f'No ML model files found in {ml_path}',
                    'severity': 'warning'
                })
        
        # Check Flask environment
        flask_env = os.getenv('FLASK_ENV', 'development')
        if flask_env == 'production':
            # Production-specific checks
            if not os.getenv('FLASK_SECRET_KEY'):
                self.warnings.append({
                    'name': 'FLASK_SECRET_KEY',
                    'description': 'Flask secret key should be set in production',
                    'severity': 'error'
                })
    
    def generate_env_template(self, include_comments: bool = True) -> str:
        """Generate .env template file content"""
        template_lines = []
        
        if include_comments:
            template_lines.extend([
                "# Tennis Betting System Environment Variables",
                "# Generated on: " + datetime.now().isoformat(),
                "",
                "# ===== CRITICAL BETFAIR CONFIGURATION =====",
                "# Required for automated betting functionality",
            ])
        
        # Add required variables
        for var_name, description in self.REQUIRED_VARS.items():
            current_value = os.getenv(var_name, '')
            
            if include_comments:
                template_lines.append(f"# {description}")
            
            if var_name.startswith('BETFAIR_'):
                template_lines.append(f"{var_name}=# REQUIRED: {description}")
            else:
                template_lines.append(f"{var_name}={current_value}")
            
            if include_comments:
                template_lines.append("")
        
        if include_comments:
            template_lines.extend([
                "# ===== OPTIONAL CONFIGURATION =====",
                "# These have sensible defaults but can be customized",
            ])
        
        # Add optional variables with defaults
        for var_name, (default_value, description) in self.OPTIONAL_VARS.items():
            current_value = os.getenv(var_name, default_value)
            
            if include_comments:
                template_lines.append(f"# {description} (default: {default_value})")
            
            template_lines.append(f"{var_name}={current_value}")
            
            if include_comments:
                template_lines.append("")
        
        return "\n".join(template_lines)
    
    def save_env_template(self, filepath: str = '.env.template') -> bool:
        """Save environment template to file"""
        try:
            template_content = self.generate_env_template(include_comments=True)
            
            with open(filepath, 'w') as f:
                f.write(template_content)
            
            print(f"✅ Environment template saved to: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save environment template: {e}")
            return False
    
    def print_status_report(self):
        """Print comprehensive environment status report"""
        status = self.check_environment()
        
        print("\n" + "="*60)
        print("🔧 TENNIS BETTING SYSTEM - ENVIRONMENT STATUS")
        print("="*60)
        
        # Summary
        print(f"\n📊 SUMMARY:")
        print(f"   Environment Ready: {'✅ YES' if status['environment_ready'] else '❌ NO'}")
        print(f"   Variables Present: {len(status['present_vars'])}")
        print(f"   Variables Missing: {len(status['missing_vars'])}")
        print(f"   Critical Missing: {status['critical_missing']}")
        print(f"   Warnings: {len(status['warnings'])}")
        
        # Missing critical variables
        critical_missing = [v for v in status['missing_vars'] if v.get('critical', False)]
        if critical_missing:
            print(f"\n🚨 CRITICAL MISSING VARIABLES:")
            for var in critical_missing:
                print(f"   ❌ {var['name']}: {var['description']}")
        
        # Missing non-critical variables
        non_critical_missing = [v for v in status['missing_vars'] if not v.get('critical', False)]
        if non_critical_missing:
            print(f"\n⚠️  OPTIONAL MISSING VARIABLES:")
            for var in non_critical_missing:
                print(f"   ⚠️  {var['name']}: {var['description']}")
        
        # Present variables
        if status['present_vars']:
            print(f"\n✅ CONFIGURED VARIABLES:")
            for var in status['present_vars']:
                print(f"   ✅ {var['name']}: {var['masked_value']}")
        
        # Warnings
        if status['warnings']:
            print(f"\n⚠️  WARNINGS:")
            for warning in status['warnings']:
                severity_icon = {'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}.get(warning.get('severity', 'info'), 'ℹ️')
                print(f"   {severity_icon} {warning['name']}: {warning['description']}")
        
        print("\n" + "="*60)
        
        # Setup instructions
        if not status['environment_ready']:
            print("🔧 SETUP INSTRUCTIONS:")
            print("   1. Copy the missing variables to your .env file")
            print("   2. Fill in your Betfair API credentials")
            print("   3. Restart the application")
            print("   4. Run this check again to verify")
        
        print()


def main():
    """Main function for standalone execution"""
    helper = EnvironmentSetupHelper()
    
    # Print status report
    helper.print_status_report()
    
    # Ask if user wants to generate template
    try:
        response = input("Generate .env template file? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            helper.save_env_template('.env.template')
            print("\n📝 Next steps:")
            print("   1. Copy .env.template to .env")
            print("   2. Edit .env and fill in your API keys")
            print("   3. Restart the application")
    except KeyboardInterrupt:
        print("\n\nExiting...")
        return
    
    # Return status for scripting
    status = helper.check_environment()
    sys.exit(0 if status['environment_ready'] else 1)


if __name__ == '__main__':
    main()