#!/usr/bin/env python3
"""
🔒 Secure Configuration Loader
Loads configuration with environment variable substitution
"""

import os
import json
import re
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SecureConfigLoader:
    """Secure configuration loader with environment variable support"""
    
    def __init__(self, config_file: str = "config.json"):
        # Secure file path validation to prevent path traversal
        if not self._is_safe_file_path(config_file):
            raise ValueError("Invalid configuration file path")
        self.config_file = config_file
        self.config = None
    
    def _is_safe_file_path(self, file_path: str) -> bool:
        """Validate file path to prevent path traversal attacks"""
        if not isinstance(file_path, str) or not file_path.strip():
            return False
        
        # Normalize path and check for suspicious patterns
        import os.path
        normalized_path = os.path.normpath(file_path)
        
        # Prevent path traversal attempts
        dangerous_patterns = ['../', '..\\', '../', '..\\\\', '/etc/', '/proc/', '/root/', '~/', '$HOME']
        for pattern in dangerous_patterns:
            if pattern in normalized_path or pattern in file_path:
                return False
        
        # Only allow files in current directory or subdirectories
        if normalized_path.startswith('/') or ':' in normalized_path:
            return False
        
        # Allow only specific file extensions
        allowed_extensions = ['.json', '.yaml', '.yml', '.toml']
        if not any(normalized_path.endswith(ext) for ext in allowed_extensions):
            return False
        
        return True
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration with environment variable substitution"""
        try:
            with open(self.config_file, 'r') as f:
                config_text = f.read()
            
            # Replace environment variables in format ${VAR_NAME}
            config_text = self._substitute_env_vars(config_text)
            
            # Parse JSON
            self.config = json.loads(config_text)
            
            # Validate critical configurations
            self._validate_config()
            
            return self.config
            
        except FileNotFoundError:
            print(f"❌ Configuration file {self.config_file} not found")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in {self.config_file}: {e}")
            return self._get_default_config()
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return self._get_default_config()
    
    def _substitute_env_vars(self, text: str) -> str:
        """Replace ${VAR_NAME} with environment variables securely"""
        def replace_var(match):
            var_name = match.group(1)
            
            # Validate environment variable name to prevent injection
            if not self._is_safe_env_var_name(var_name):
                print(f"⚠️ Invalid environment variable name: {var_name}")
                return f"INVALID_{var_name}"
            
            env_value = os.getenv(var_name)
            
            if env_value is None:
                print(f"⚠️ Environment variable {var_name} not set, using placeholder")
                return f"MISSING_{var_name}"
            
            # Validate environment variable value
            if not self._is_safe_env_var_value(env_value, var_name):
                print(f"⚠️ Environment variable {var_name} contains unsafe content")
                return f"UNSAFE_{var_name}"
            
            return env_value
        
        # Pattern to match ${VAR_NAME} - more restrictive
        pattern = r'\$\{([A-Z_][A-Z0-9_]*)\}'
        return re.sub(pattern, replace_var, text)
    
    def _is_safe_env_var_name(self, var_name: str) -> bool:
        """Validate environment variable name"""
        if not isinstance(var_name, str) or not var_name:
            return False
        
        # Only allow uppercase letters, numbers, and underscores
        # Must start with letter or underscore
        if not re.match(r'^[A-Z_][A-Z0-9_]*$', var_name):
            return False
        
        # Reasonable length limit
        if len(var_name) > 100:
            return False
        
        return True
    
    def _is_safe_env_var_value(self, value: str, var_name: str) -> bool:
        """Validate environment variable value"""
        if not isinstance(value, str):
            return False
        
        # Length check to prevent DoS
        if len(value) > 10000:
            return False
        
        # For API keys and secrets, ensure minimum security requirements
        if 'key' in var_name.lower() or 'secret' in var_name.lower() or 'token' in var_name.lower():
            if len(value) < 8:  # Minimum length for security tokens
                return False
        
        # Prevent injection attacks in values
        dangerous_patterns = ['$(', '`', '#{', '${']
        for pattern in dangerous_patterns:
            if pattern in value:
                return False
        
        return True
    
    def _validate_config(self):
        """Validate critical configuration values"""
        if not self.config:
            return
            
        # Check if API keys are properly set
        data_sources = self.config.get('data_sources', {})
        odds_api = data_sources.get('the_odds_api', {})
        
        if odds_api.get('enabled') and odds_api.get('api_key', '').startswith('MISSING_'):
            print("⚠️ Warning: Odds API key not configured. Set ODDS_API_KEY environment variable.")
        
        # Check betting APIs
        betting_apis = self.config.get('betting_apis', {})
        for api_name, api_config in betting_apis.items():
            if api_config.get('enabled'):
                # Check API key configuration
                if api_config.get('api_key', '').startswith('MISSING_'):
                    print(f"⚠️ Warning: {api_name} API key not configured.")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return minimal default configuration"""
        return {
            "web_interface": {
                "host": "127.0.0.1",
                "port": 5001,
                "debug": False
            },
            "data_sources": {
                "the_odds_api": {
                    "enabled": False,
                    "api_key": "",
                    "base_url": "https://api.the-odds-api.com/v4"
                }
            },
            "model_settings": {
                "min_confidence_threshold": 0.55,
                "ensemble_weights": {
                    "neural_network": 0.25,
                    "xgboost": 0.25,
                    "random_forest": 0.2,
                    "gradient_boosting": 0.2,
                    "logistic_regression": 0.1
                }
            },
            "logging": {
                "level": "INFO",
                "file": "tennis_system.log"
            }
        }
    
    def get_api_key(self, service: str) -> str:
        """Safely get API key for a service"""
        if not self.config:
            return ""
            
        # Map service names to config paths
        service_map = {
            'odds_api': ['data_sources', 'the_odds_api', 'api_key'],
            'pinnacle': ['betting_apis', 'pinnacle', 'api_key'],
            'rapidapi': ['data_sources', 'rapidapi_tennis', 'api_key']
        }
        
        if service not in service_map:
            return ""
        
        # Navigate to the key
        current = self.config
        for i, key in enumerate(service_map[service]):
            if i == len(service_map[service]) - 1:  # Last key (the actual value)
                return str(current.get(key, "")) if current.get(key) else ""
            else:  # Intermediate keys (should be dicts)
                current = current.get(key, {})
                if not isinstance(current, dict):
                    return ""
        
        return ""

# Global config loader instance
config_loader = SecureConfigLoader()

def load_secure_config(config_file: str = "config.json") -> Dict[str, Any]:
    """Load configuration securely with environment variables"""
    loader = SecureConfigLoader(config_file)
    return loader.load_config()

def get_api_key(service: str) -> str:
    """Get API key for a service safely"""
    return config_loader.get_api_key(service)

