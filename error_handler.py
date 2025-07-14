#!/usr/bin/env python3
"""
⚠️ Comprehensive Error Handling Module
Provides robust error handling for API calls, ML predictions, and system operations
"""

import logging
import traceback
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError, HTTPError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TennisSystemError(Exception):
    """Base exception for tennis system errors"""
    pass

class APIError(TennisSystemError):
    """API-related errors"""
    pass

class MLError(TennisSystemError):
    """Machine Learning related errors"""
    pass

class DataError(TennisSystemError):
    """Data processing errors"""
    pass

def safe_api_call(max_retries: int = 3, delay: float = 1.0, backoff_multiplier: float = 2.0):
    """
    Decorator for safe API calls with retry logic
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff_multiplier: Multiplier for exponential backoff
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Tuple[bool, Any]:
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"✅ API call succeeded on attempt {attempt + 1}")
                    return True, result
                    
                except (ConnectionError, Timeout) as e:
                    last_exception = e
                    logger.warning(f"🔄 Network error on attempt {attempt + 1}: {e}")
                    
                except HTTPError as e:
                    last_exception = e
                    if e.response.status_code in [429, 502, 503, 504]:  # Retry on these codes
                        logger.warning(f"🔄 HTTP error {e.response.status_code} on attempt {attempt + 1}")
                    else:
                        logger.error(f"❌ Non-retryable HTTP error: {e}")
                        return False, {"error": f"HTTP {e.response.status_code}: {str(e)}"}
                        
                except RequestException as e:
                    last_exception = e
                    logger.warning(f"🔄 Request error on attempt {attempt + 1}: {e}")
                    
                except Exception as e:
                    last_exception = e
                    logger.error(f"❌ Unexpected error in API call: {e}")
                    logger.error(traceback.format_exc())
                    return False, {"error": f"Unexpected error: {str(e)}"}
                
                # Don't sleep after the last attempt
                if attempt < max_retries:
                    logger.info(f"⏳ Waiting {current_delay:.1f}s before retry...")
                    time.sleep(current_delay)
                    current_delay *= backoff_multiplier
            
            # All attempts failed
            error_msg = f"API call failed after {max_retries + 1} attempts. Last error: {last_exception}"
            logger.error(f"❌ {error_msg}")
            return False, {"error": error_msg}
            
        return wrapper
    return decorator

def safe_ml_prediction(func: Callable) -> Callable:
    """
    Decorator for safe ML predictions with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Tuple[bool, Any]:
        try:
            result = func(*args, **kwargs)
            
            # Validate prediction result
            if isinstance(result, dict):
                if 'probability' in result:
                    prob = result['probability']
                    if not (0 <= prob <= 1):
                        logger.warning(f"⚠️ Invalid probability: {prob}, clamping to [0,1]")
                        result['probability'] = max(0, min(1, prob))
                        
            return True, result
            
        except Exception as e:
            logger.error(f"❌ ML prediction failed: {e}")
            logger.error(traceback.format_exc())
            
            # Return safe fallback prediction
            fallback = {
                "probability": 0.5,  # Neutral prediction
                "confidence": "Low",
                "prediction_type": "FALLBACK",
                "error": str(e),
                "timestamp": time.time()
            }
            return False, fallback
            
    return wrapper

def safe_data_processing(func: Callable) -> Callable:
    """
    Decorator for safe data processing operations
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Tuple[bool, Any]:
        try:
            result = func(*args, **kwargs)
            
            # Basic validation
            if result is None:
                logger.warning("⚠️ Data processing returned None")
                return False, {"error": "No data processed"}
                
            return True, result
            
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"❌ Data processing error: {e}")
            return False, {"error": f"Data error: {str(e)}"}
            
        except Exception as e:
            logger.error(f"❌ Unexpected data processing error: {e}")
            logger.error(traceback.format_exc())
            return False, {"error": f"Processing failed: {str(e)}"}
            
    return wrapper

def validate_player_data(player_data: Dict) -> bool:
    """Validate player data structure"""
    required_fields = ['rank', 'points', 'age']
    
    if not isinstance(player_data, dict):
        return False
        
    for field in required_fields:
        if field not in player_data:
            logger.warning(f"⚠️ Missing required field: {field}")
            return False
            
        value = player_data[field]
        if not isinstance(value, (int, float)) or value < 0:
            logger.warning(f"⚠️ Invalid value for {field}: {value}")
            return False
            
    return True

def validate_match_data(match_data: Dict) -> bool:
    """Validate match data structure"""
    required_fields = ['player1', 'player2']
    
    if not isinstance(match_data, dict):
        return False
        
    for field in required_fields:
        if field not in match_data:
            logger.warning(f"⚠️ Missing required field: {field}")
            return False
            
        if not isinstance(match_data[field], str) or not match_data[field].strip():
            logger.warning(f"⚠️ Invalid player name: {match_data[field]}")
            return False
            
    return True

class ErrorHandler:
    """Centralized error handling and logging"""
    
    def __init__(self, log_file: str = "tennis_errors.log"):
        self.logger = logging.getLogger("TennisSystem")
        self.log_file = log_file
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.setLevel(logging.INFO)
    
    def log_error(self, component: str, error: Exception, context: Dict = None):
        """Log error with context"""
        error_info = {
            "component": component,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }
        
        self.logger.error(f"Error in {component}: {error_info}")
    
    def log_warning(self, component: str, message: str, context: Dict = None):
        """Log warning with context"""
        self.logger.warning(f"{component}: {message} | Context: {context or {}}")
    
    def log_info(self, component: str, message: str):
        """Log info message"""
        self.logger.info(f"{component}: {message}")

# Global error handler instance
error_handler = ErrorHandler()

def get_error_handler() -> ErrorHandler:
    """Get global error handler instance"""
    return error_handler