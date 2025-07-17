#!/usr/bin/env python3
"""
🚨 Monitoring and Alerting System for Tennis Predictions
Production-ready monitoring with alerts for failed predictions and system issues
"""

import logging
import json
import smtplib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import time

logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    title: str
    message: str
    timestamp: datetime
    component: str
    additional_data: Dict = None
    
    def to_dict(self):
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class MetricSnapshot:
    """System metrics snapshot"""
    timestamp: datetime
    prediction_success_rate: float
    api_response_time: float
    active_models: int
    memory_usage_mb: float
    error_count_last_hour: int
    
class MonitoringSystem:
    """Production monitoring system for tennis prediction system"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._load_default_config()
        self.alerts = deque(maxlen=1000)  # Keep last 1000 alerts
        self.metrics = deque(maxlen=100)   # Keep last 100 metric snapshots
        self.error_counts = defaultdict(int)
        self.prediction_failures = deque(maxlen=50)
        self.api_failures = deque(maxlen=50)
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Setup logging
        self._setup_logging()
        
    def _load_default_config(self) -> Dict:
        """Load default monitoring configuration"""
        return {
            'alerting': {
                'email_enabled': os.getenv('ALERT_EMAIL_ENABLED', 'false').lower() == 'true',
                'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
                'smtp_port': int(os.getenv('SMTP_PORT', '587')),
                'smtp_username': os.getenv('SMTP_USERNAME', ''),
                'smtp_password': os.getenv('SMTP_PASSWORD', ''),
                'alert_recipients': os.getenv('ALERT_RECIPIENTS', '').split(','),
                'webhook_url': os.getenv('ALERT_WEBHOOK_URL', '')
            },
            'thresholds': {
                'prediction_failure_rate': 0.3,  # Alert if >30% predictions fail
                'api_failure_rate': 0.2,          # Alert if >20% API calls fail
                'response_time_ms': 5000,         # Alert if response time >5s
                'memory_usage_mb': 1000,          # Alert if memory usage >1GB
                'error_count_per_hour': 10        # Alert if >10 errors per hour
            },
            'monitoring': {
                'check_interval_seconds': 60,     # Check metrics every minute
                'alert_cooldown_minutes': 30      # Don't repeat same alert for 30 min
            }
        }
    
    def _setup_logging(self):
        """Setup structured logging for monitoring"""
        # Create monitoring-specific logger
        self.monitor_logger = logging.getLogger('tennis_monitoring')
        
        # Create file handler for monitoring logs
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(f'{log_dir}/monitoring.log')
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.monitor_logger.addHandler(file_handler)
        self.monitor_logger.setLevel(logging.INFO)
    
    def start_monitoring(self):
        """Start background monitoring thread"""
        if self.is_monitoring:
            logger.warning("Monitoring already running")
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("🎯 Monitoring system started")
        self.monitor_logger.info("Monitoring system started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("🛑 Monitoring system stopped")
        self.monitor_logger.info("Monitoring system stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        check_interval = self.config['monitoring']['check_interval_seconds']
        
        while self.is_monitoring:
            try:
                self._collect_metrics()
                self._check_thresholds()
                time.sleep(check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(check_interval)
    
    def _collect_metrics(self):
        """Collect system metrics"""
        try:
            # Calculate prediction success rate
            recent_predictions = list(self.prediction_failures)[-20:]  # Last 20 predictions
            if recent_predictions:
                failures = sum(1 for p in recent_predictions if not p.get('success', True))
                success_rate = 1.0 - (failures / len(recent_predictions))
            else:
                success_rate = 1.0
            
            # Calculate API response time (mock - would be real measurement)
            api_response_time = self._get_average_api_response_time()
            
            # Get memory usage (mock - would be real measurement)
            memory_usage = self._get_memory_usage()
            
            # Count recent errors
            one_hour_ago = datetime.now() - timedelta(hours=1)
            error_count = sum(1 for alert in self.alerts 
                            if alert.timestamp > one_hour_ago and alert.severity in ['critical', 'high'])
            
            # Create metrics snapshot
            snapshot = MetricSnapshot(
                timestamp=datetime.now(),
                prediction_success_rate=success_rate,
                api_response_time=api_response_time,
                active_models=5,  # Based on your 5 ML models
                memory_usage_mb=memory_usage,
                error_count_last_hour=error_count
            )
            
            self.metrics.append(snapshot)
            
            # Log metrics
            self.monitor_logger.info(f"Metrics collected: success_rate={success_rate:.2f}, "
                                   f"api_time={api_response_time:.1f}ms, "
                                   f"memory={memory_usage:.1f}MB, "
                                   f"errors_1h={error_count}")
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    def _get_average_api_response_time(self) -> float:
        """Get average API response time (mock implementation)"""
        # In real implementation, this would track actual API response times
        return 850.0  # Mock value in milliseconds
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage (mock implementation)"""
        # In real implementation, this would use psutil or similar
        import random
        return random.uniform(400, 800)  # Mock value in MB
    
    def _check_thresholds(self):
        """Check if any thresholds are exceeded"""
        if not self.metrics:
            return
            
        latest_metrics = self.metrics[-1]
        thresholds = self.config['thresholds']
        
        # Check prediction failure rate
        if latest_metrics.prediction_success_rate < (1 - thresholds['prediction_failure_rate']):
            self._create_alert(
                'high',
                'High Prediction Failure Rate',
                f"Prediction success rate dropped to {latest_metrics.prediction_success_rate:.1%}",
                'ml_predictions'
            )
        
        # Check API response time
        if latest_metrics.api_response_time > thresholds['response_time_ms']:
            self._create_alert(
                'medium',
                'High API Response Time',
                f"API response time is {latest_metrics.api_response_time:.0f}ms",
                'api_performance'
            )
        
        # Check memory usage
        if latest_metrics.memory_usage_mb > thresholds['memory_usage_mb']:
            self._create_alert(
                'medium',
                'High Memory Usage',
                f"Memory usage is {latest_metrics.memory_usage_mb:.0f}MB",
                'system_resources'
            )
        
        # Check error count
        if latest_metrics.error_count_last_hour > thresholds['error_count_per_hour']:
            self._create_alert(
                'high',
                'High Error Rate',
                f"Detected {latest_metrics.error_count_last_hour} errors in the last hour",
                'system_errors'
            )
    
    def record_prediction_failure(self, prediction_data: Dict, error: str):
        """Record a prediction failure for monitoring"""
        failure_record = {
            'timestamp': datetime.now(),
            'prediction_data': prediction_data,
            'error': error,
            'success': False
        }
        
        self.prediction_failures.append(failure_record)
        
        # Create alert for critical prediction failures
        if 'model' in error.lower() or 'load' in error.lower():
            self._create_alert(
                'critical',
                'ML Model Failure',
                f"Prediction failed: {error}",
                'ml_predictions',
                {'prediction_data': prediction_data}
            )
        
        self.monitor_logger.error(f"Prediction failure: {error}")
    
    def record_prediction_success(self, prediction_data: Dict, result: Dict):
        """Record a successful prediction"""
        success_record = {
            'timestamp': datetime.now(),
            'prediction_data': prediction_data,
            'result': result,
            'success': True
        }
        
        self.prediction_failures.append(success_record)
    
    def record_api_failure(self, api_endpoint: str, error: str, response_code: int = None):
        """Record an API failure"""
        failure_record = {
            'timestamp': datetime.now(),
            'endpoint': api_endpoint,
            'error': error,
            'response_code': response_code
        }
        
        self.api_failures.append(failure_record)
        
        # Create alert for critical API failures
        if response_code and response_code >= 500:
            self._create_alert(
                'high',
                'API Server Error',
                f"API endpoint {api_endpoint} returned {response_code}: {error}",
                'api_failures'
            )
        
        self.monitor_logger.error(f"API failure: {api_endpoint} - {error}")
    
    def _create_alert(self, severity: str, title: str, message: str, component: str, additional_data: Dict = None):
        """Create and process an alert"""
        alert_id = f"{component}_{severity}_{int(time.time())}"
        
        # Check if similar alert exists recently (alert cooldown)
        cooldown_minutes = self.config['monitoring']['alert_cooldown_minutes']
        cutoff_time = datetime.now() - timedelta(minutes=cooldown_minutes)
        
        similar_alerts = [
            a for a in self.alerts 
            if a.component == component and a.severity == severity and a.timestamp > cutoff_time
        ]
        
        if similar_alerts:
            logger.debug(f"Skipping duplicate alert for {component} (cooldown active)")
            return
        
        alert = Alert(
            alert_id=alert_id,
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.now(),
            component=component,
            additional_data=additional_data or {}
        )
        
        self.alerts.append(alert)
        
        # Log the alert
        self.monitor_logger.warning(f"ALERT [{severity.upper()}] {title}: {message}")
        
        # Send notifications
        self._send_alert_notifications(alert)
    
    def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications via configured channels"""
        try:
            # Send email notification
            if self.config['alerting']['email_enabled']:
                self._send_email_alert(alert)
            
            # Send webhook notification
            if self.config['alerting']['webhook_url']:
                self._send_webhook_alert(alert)
                
        except Exception as e:
            logger.error(f"Failed to send alert notifications: {e}")
    
    def _send_email_alert(self, alert: Alert):
        """Send email alert"""
        try:
            smtp_config = self.config['alerting']
            
            if not smtp_config['smtp_username'] or not smtp_config['alert_recipients']:
                return
            
            msg = MimeMultipart()
            msg['From'] = smtp_config['smtp_username']
            msg['To'] = ', '.join(smtp_config['alert_recipients'])
            msg['Subject'] = f"🚨 Tennis System Alert: {alert.title}"
            
            body = f"""
Tennis Prediction System Alert

Severity: {alert.severity.upper()}
Component: {alert.component}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Message:
{alert.message}

Additional Data:
{json.dumps(alert.additional_data, indent=2)}

Alert ID: {alert.alert_id}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port'])
            server.starttls()
            server.login(smtp_config['smtp_username'], smtp_config['smtp_password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent for {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _send_webhook_alert(self, alert: Alert):
        """Send webhook alert"""
        try:
            import requests
            
            webhook_url = self.config['alerting']['webhook_url']
            if not webhook_url:
                return
            
            payload = {
                'alert': alert.to_dict(),
                'system': 'tennis_prediction',
                'environment': os.getenv('ENVIRONMENT', 'production')
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Webhook alert sent for {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def get_system_health(self) -> Dict:
        """Get current system health status"""
        if not self.metrics:
            return {'status': 'unknown', 'message': 'No metrics available'}
        
        latest_metrics = self.metrics[-1]
        
        # Determine overall health
        issues = []
        
        if latest_metrics.prediction_success_rate < 0.8:
            issues.append(f"Low prediction success rate: {latest_metrics.prediction_success_rate:.1%}")
        
        if latest_metrics.api_response_time > 3000:
            issues.append(f"High API response time: {latest_metrics.api_response_time:.0f}ms")
        
        if latest_metrics.error_count_last_hour > 5:
            issues.append(f"High error count: {latest_metrics.error_count_last_hour} errors/hour")
        
        if issues:
            status = 'degraded' if len(issues) < 3 else 'unhealthy'
        else:
            status = 'healthy'
        
        return {
            'status': status,
            'timestamp': latest_metrics.timestamp.isoformat(),
            'metrics': {
                'prediction_success_rate': latest_metrics.prediction_success_rate,
                'api_response_time_ms': latest_metrics.api_response_time,
                'memory_usage_mb': latest_metrics.memory_usage_mb,
                'active_models': latest_metrics.active_models,
                'error_count_last_hour': latest_metrics.error_count_last_hour
            },
            'issues': issues,
            'recent_alerts': len([a for a in self.alerts if a.timestamp > datetime.now() - timedelta(hours=24)])
        }
    
    def get_alerts(self, limit: int = 20, severity: str = None) -> List[Dict]:
        """Get recent alerts"""
        alerts = list(self.alerts)
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        # Sort by timestamp (most recent first)
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        return [alert.to_dict() for alert in alerts[:limit]]
    
    def get_metrics_summary(self, hours: int = 24) -> Dict:
        """Get metrics summary for the specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {'error': 'No metrics available for the specified period'}
        
        # Calculate averages
        avg_success_rate = sum(m.prediction_success_rate for m in recent_metrics) / len(recent_metrics)
        avg_response_time = sum(m.api_response_time for m in recent_metrics) / len(recent_metrics)
        avg_memory_usage = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
        
        return {
            'period_hours': hours,
            'metrics_count': len(recent_metrics),
            'averages': {
                'prediction_success_rate': avg_success_rate,
                'api_response_time_ms': avg_response_time,
                'memory_usage_mb': avg_memory_usage
            },
            'latest': {
                'prediction_success_rate': recent_metrics[-1].prediction_success_rate,
                'api_response_time_ms': recent_metrics[-1].api_response_time,
                'memory_usage_mb': recent_metrics[-1].memory_usage_mb
            }
        }


# Global monitoring instance
_monitoring_system = None

def get_monitoring_system() -> MonitoringSystem:
    """Get global monitoring system instance"""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = MonitoringSystem()
    return _monitoring_system

def init_monitoring(config: Dict = None) -> MonitoringSystem:
    """Initialize monitoring system"""
    global _monitoring_system
    _monitoring_system = MonitoringSystem(config)
    return _monitoring_system

# Convenience functions
def record_prediction_failure(prediction_data: Dict, error: str):
    """Record a prediction failure"""
    monitoring = get_monitoring_system()
    monitoring.record_prediction_failure(prediction_data, error)

def record_prediction_success(prediction_data: Dict, result: Dict):
    """Record a prediction success"""
    monitoring = get_monitoring_system()
    monitoring.record_prediction_success(prediction_data, result)

def record_api_failure(api_endpoint: str, error: str, response_code: int = None):
    """Record an API failure"""
    monitoring = get_monitoring_system()
    monitoring.record_api_failure(api_endpoint, error, response_code)

def start_monitoring():
    """Start the monitoring system"""
    monitoring = get_monitoring_system()
    monitoring.start_monitoring()

def get_system_health() -> Dict:
    """Get current system health"""
    monitoring = get_monitoring_system()
    return monitoring.get_system_health()