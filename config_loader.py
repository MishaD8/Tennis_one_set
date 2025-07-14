#!/usr/bin/env python3
"""
🔒 Secure Configuration Loader
Loads configuration with environment variable substitution
"""

import os
import json
import re
from typing import Dict, Any

class SecureConfigLoader:
    """Secure configuration loader with environment variable support"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = None
        
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
        """Replace ${VAR_NAME} with environment variables"""
        def replace_var(match):
            var_name = match.group(1)
            env_value = os.getenv(var_name)
            
            if env_value is None:
                print(f"⚠️ Environment variable {var_name} not set, using placeholder")
                return f"MISSING_{var_name}"
            
            return env_value
        
        # Pattern to match ${VAR_NAME}
        pattern = r'\$\{([^}]+)\}'
        return re.sub(pattern, replace_var, text)
    
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
            if api_config.get('enabled') and api_config.get('api_key', '').startswith('MISSING_'):
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
            'betfair': ['betting_apis', 'betfair', 'app_key']
        }
        
        if service not in service_map:
            return ""
        
        # Navigate to the key
        current = self.config
        for key in service_map[service]:
            current = current.get(key, {})
            if not isinstance(current, dict) and key != service_map[service][-1]:
                return ""
        
        return str(current) if current else ""

# Global config loader instance
config_loader = SecureConfigLoader()

def load_secure_config(config_file: str = "config.json") -> Dict[str, Any]:
    """Load configuration securely with environment variables"""
    loader = SecureConfigLoader(config_file)
    return loader.load_config()

def get_api_key(service: str) -> str:
    """Get API key for a service safely"""
    return config_loader.get_api_key(service)