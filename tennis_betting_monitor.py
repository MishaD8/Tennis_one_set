#!/usr/bin/env python3
"""
Tennis Betting System Monitor
Production monitoring and alerting for the automated tennis betting system
"""

import os
import json
import time
import logging
import threading
import smtplib
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import psutil

from config import get_config
from automated_betting_engine import AutomatedBettingEngine, BetOrder, BetStatus
from betfair_api_client import BetfairAPIClient

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MonitoringMetric(Enum):
    """System monitoring metrics"""
    SYSTEM_HEALTH = "system_health"
    BETTING_PERFORMANCE = "betting_performance"
    API_CONNECTIVITY = "api_connectivity"
    BALANCE_THRESHOLD = "balance_threshold"
    ERROR_RATE = "error_rate"
    BET_VOLUME = "bet_volume"
    PROFIT_LOSS = "profit_loss"


@dataclass
class Alert:
    """System alert"""
    severity: AlertSeverity
    metric: MonitoringMetric
    message: str
    value: Any
    threshold: Any
    timestamp: datetime
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['severity'] = self.severity.value
        data['metric'] = self.metric.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_connections: int
    uptime_seconds: float
    active_threads: int
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class BettingMetrics:
    """Betting system metrics"""
    total_bets_placed: int
    bets_won: int
    bets_lost: int
    total_profit_loss: float
    current_balance: float
    win_rate: float
    average_stake: float
    largest_win: float
    largest_loss: float
    active_bets: int
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class TelegramNotifier:
    """Telegram notification system for alerts"""
    
    def __init__(self, bot_token: str = None, chat_id: str = None):
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID', '')
        self.enabled = bool(self.bot_token and self.chat_id)
        
        if not self.enabled:
            logger.warning("Telegram notifications disabled - bot token or chat ID not configured")
    
    def send_alert(self, alert: Alert) -> bool:
        """Send alert via Telegram"""
        if not self.enabled:
            return False
        
        try:
            emoji_map = {
                AlertSeverity.INFO: "ℹ️",
                AlertSeverity.WARNING: "⚠️",
                AlertSeverity.ERROR: "❌",
                AlertSeverity.CRITICAL: "🚨"
            }
            
            emoji = emoji_map.get(alert.severity, "📊")
            
            message = f"{emoji} *{alert.severity.value.upper()}*\n\n"
            message += f"🎾 *Tennis Betting Alert*\n"
            message += f"📊 Metric: {alert.metric.value.replace('_', ' ').title()}\n"
            message += f"💬 Message: {alert.message}\n"
            message += f"📈 Value: {alert.value}\n"
            
            if alert.threshold:
                message += f"🎯 Threshold: {alert.threshold}\n"
            
            message += f"🕐 Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=data, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Telegram alert sent: {alert.severity.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False


class EmailNotifier:
    """Email notification system for alerts"""
    
    def __init__(self, smtp_host: str = None, smtp_port: int = None, 
                 smtp_user: str = None, smtp_password: str = None, 
                 recipients: List[str] = None):
        self.smtp_host = smtp_host or os.getenv('SMTP_HOST', '')
        self.smtp_port = smtp_port or int(os.getenv('SMTP_PORT', '587'))
        self.smtp_user = smtp_user or os.getenv('SMTP_USER', '')
        self.smtp_password = smtp_password or os.getenv('SMTP_PASSWORD', '')
        self.recipients = recipients or os.getenv('ALERT_RECIPIENTS', '').split(',')
        
        self.enabled = bool(self.smtp_host and self.smtp_user and self.recipients)
        
        if not self.enabled:
            logger.warning("Email notifications disabled - SMTP settings not configured")
    
    def send_alert(self, alert: Alert) -> bool:
        """Send alert via email"""
        if not self.enabled:
            return False
        
        try:
            subject = f"Tennis Betting Alert - {alert.severity.value.upper()}"
            
            body = f"""
Tennis Betting System Alert

Severity: {alert.severity.value.upper()}
Metric: {alert.metric.value.replace('_', ' ').title()}
Message: {alert.message}
Value: {alert.value}
Threshold: {alert.threshold}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

This is an automated alert from the Tennis Betting System.
Please check the system status and take appropriate action if necessary.
"""
            
            msg = MIMEMultipart()
            msg['From'] = self.smtp_user
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.starttls()
            server.login(self.smtp_user, self.smtp_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent: {alert.severity.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False


class TennisBettingMonitor:
    """Production monitoring system for tennis betting"""
    
    def __init__(self, betting_engine: AutomatedBettingEngine = None):
        self.config = get_config()
        self.betting_engine = betting_engine
        self.betfair_client = BetfairAPIClient()
        
        # Notification systems
        self.telegram_notifier = TelegramNotifier()
        self.email_notifier = EmailNotifier()
        
        # Monitoring state
        self.is_running = False
        self.stop_event = threading.Event()
        
        # Alert history and thresholds
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable] = []
        
        # Monitoring thresholds
        self.thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'min_balance': 100.0,
            'max_daily_loss': 500.0,
            'error_rate': 10.0,  # per hour
            'max_concurrent_bets': 20
        }
        
        # Metrics collection
        self.system_metrics_history: List[SystemMetrics] = []
        self.betting_metrics_history: List[BettingMetrics] = []
        
        # Error tracking
        self.error_count = 0
        self.last_error_reset = datetime.now()
        
        logger.info("Tennis Betting Monitor initialized")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for alert events"""
        self.alert_callbacks.append(callback)
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network_connections = len(psutil.net_connections())
            uptime = time.time() - psutil.boot_time()
            active_threads = threading.active_count()
            
            metrics = SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                network_connections=network_connections,
                uptime_seconds=uptime,
                active_threads=active_threads,
                timestamp=datetime.now()
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return None
    
    def collect_betting_metrics(self) -> BettingMetrics:
        """Collect current betting metrics"""
        try:
            if not self.betting_engine:
                return None
            
            stats = self.betting_engine.get_stats()
            risk_summary = stats.get('risk_summary', {})
            
            # Calculate additional metrics
            total_bets = stats.get('bets_won', 0) + stats.get('bets_lost', 0)
            win_rate = (stats.get('bets_won', 0) / total_bets * 100) if total_bets > 0 else 0
            
            bet_history = self.betting_engine.get_bet_history()
            stakes = [bet.stake for bet in bet_history] if bet_history else [0]
            payouts = [bet.potential_payout for bet in bet_history if bet.status == BetStatus.SETTLED_WON]
            losses = [bet.stake for bet in bet_history if bet.status == BetStatus.SETTLED_LOST]
            
            metrics = BettingMetrics(
                total_bets_placed=stats.get('bets_placed', 0),
                bets_won=stats.get('bets_won', 0),
                bets_lost=stats.get('bets_lost', 0),
                total_profit_loss=stats.get('total_profit_loss', 0.0),
                current_balance=risk_summary.get('current_bankroll', 0.0),
                win_rate=win_rate,
                average_stake=sum(stakes) / len(stakes) if stakes else 0.0,
                largest_win=max(payouts) if payouts else 0.0,
                largest_loss=max(losses) if losses else 0.0,
                active_bets=len(self.betting_engine.get_active_bets()),
                timestamp=datetime.now()
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect betting metrics: {e}")
            return None
    
    def check_system_health(self, metrics: SystemMetrics):
        """Check system health and trigger alerts"""
        if not metrics:
            return
        
        # CPU usage alert
        if metrics.cpu_usage > self.thresholds['cpu_usage']:
            self._trigger_alert(
                AlertSeverity.WARNING,
                MonitoringMetric.SYSTEM_HEALTH,
                f"High CPU usage detected",
                f"{metrics.cpu_usage:.1f}%",
                f"{self.thresholds['cpu_usage']}%"
            )
        
        # Memory usage alert
        if metrics.memory_usage > self.thresholds['memory_usage']:
            self._trigger_alert(
                AlertSeverity.WARNING,
                MonitoringMetric.SYSTEM_HEALTH,
                f"High memory usage detected",
                f"{metrics.memory_usage:.1f}%",
                f"{self.thresholds['memory_usage']}%"
            )
        
        # Disk usage alert
        if metrics.disk_usage > self.thresholds['disk_usage']:
            self._trigger_alert(
                AlertSeverity.ERROR,
                MonitoringMetric.SYSTEM_HEALTH,
                f"High disk usage detected",
                f"{metrics.disk_usage:.1f}%",
                f"{self.thresholds['disk_usage']}%"
            )
    
    def check_betting_health(self, metrics: BettingMetrics):
        """Check betting system health and trigger alerts"""
        if not metrics:
            return
        
        # Balance threshold alert
        if metrics.current_balance < self.thresholds['min_balance']:
            self._trigger_alert(
                AlertSeverity.CRITICAL,
                MonitoringMetric.BALANCE_THRESHOLD,
                f"Account balance below minimum threshold",
                f"€{metrics.current_balance:.2f}",
                f"€{self.thresholds['min_balance']:.2f}"
            )
        
        # Daily loss alert
        if abs(metrics.total_profit_loss) > self.thresholds['max_daily_loss'] and metrics.total_profit_loss < 0:
            self._trigger_alert(
                AlertSeverity.ERROR,
                MonitoringMetric.PROFIT_LOSS,
                f"Daily loss limit exceeded",
                f"€{metrics.total_profit_loss:.2f}",
                f"€{self.thresholds['max_daily_loss']:.2f}"
            )
        
        # Active bets alert
        if metrics.active_bets > self.thresholds['max_concurrent_bets']:
            self._trigger_alert(
                AlertSeverity.WARNING,
                MonitoringMetric.BET_VOLUME,
                f"High number of active bets",
                metrics.active_bets,
                self.thresholds['max_concurrent_bets']
            )
        
        # Low win rate alert (only if sufficient data)
        if metrics.total_bets_placed > 20 and metrics.win_rate < 30:
            self._trigger_alert(
                AlertSeverity.WARNING,
                MonitoringMetric.BETTING_PERFORMANCE,
                f"Low win rate detected",
                f"{metrics.win_rate:.1f}%",
                "30%"
            )
    
    def check_api_connectivity(self):
        """Check API connectivity and trigger alerts"""
        try:
            # Check Betfair API
            health = self.betfair_client.health_check()
            
            if health['status'] != 'healthy':
                self._trigger_alert(
                    AlertSeverity.ERROR,
                    MonitoringMetric.API_CONNECTIVITY,
                    f"Betfair API connection issue",
                    health.get('message', 'Unknown error'),
                    "Healthy connection required"
                )
            
        except Exception as e:
            self._trigger_alert(
                AlertSeverity.CRITICAL,
                MonitoringMetric.API_CONNECTIVITY,
                f"API connectivity check failed",
                str(e),
                "All APIs should be accessible"
            )
    
    def check_error_rate(self):
        """Check error rate and trigger alerts"""
        now = datetime.now()
        
        # Reset error count every hour
        if now - self.last_error_reset > timedelta(hours=1):
            self.error_count = 0
            self.last_error_reset = now
        
        # Check error rate
        if self.error_count > self.thresholds['error_rate']:
            self._trigger_alert(
                AlertSeverity.WARNING,
                MonitoringMetric.ERROR_RATE,
                f"High error rate detected",
                f"{self.error_count} errors/hour",
                f"{self.thresholds['error_rate']} errors/hour"
            )
    
    def _trigger_alert(self, severity: AlertSeverity, metric: MonitoringMetric, 
                      message: str, value: Any, threshold: Any):
        """Trigger system alert"""
        alert = Alert(
            severity=severity,
            metric=metric,
            message=message,
            value=value,
            threshold=threshold,
            timestamp=datetime.now()
        )
        
        # Add to alert history
        self.alerts.append(alert)
        
        # Keep only last 1000 alerts
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
        
        # Send notifications
        if severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
            self.telegram_notifier.send_alert(alert)
            self.email_notifier.send_alert(alert)
        elif severity == AlertSeverity.WARNING:
            self.telegram_notifier.send_alert(alert)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.warning(f"ALERT [{severity.value.upper()}]: {message} (Value: {value}, Threshold: {threshold})")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error and increment error count"""
        self.error_count += 1
        logger.error(f"Error in {context}: {error}")
    
    def _monitoring_worker(self):
        """Main monitoring worker thread"""
        while not self.stop_event.is_set():
            try:
                # Collect metrics
                system_metrics = self.collect_system_metrics()
                betting_metrics = self.collect_betting_metrics()
                
                # Store metrics
                if system_metrics:
                    self.system_metrics_history.append(system_metrics)
                    if len(self.system_metrics_history) > 1440:  # Keep 24 hours (1 min intervals)
                        self.system_metrics_history = self.system_metrics_history[-1440:]
                
                if betting_metrics:
                    self.betting_metrics_history.append(betting_metrics)
                    if len(self.betting_metrics_history) > 1440:
                        self.betting_metrics_history = self.betting_metrics_history[-1440:]
                
                # Run health checks
                self.check_system_health(system_metrics)
                self.check_betting_health(betting_metrics)
                self.check_api_connectivity()
                self.check_error_rate()
                
                # Wait before next check
                self.stop_event.wait(60)  # Check every minute
                
            except Exception as e:
                self.log_error(e, "monitoring worker")
                self.stop_event.wait(60)
    
    def start(self):
        """Start monitoring system"""
        logger.info("Starting Tennis Betting Monitor...")
        
        self.is_running = True
        self.stop_event.clear()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Tennis Betting Monitor started successfully")
    
    def stop(self):
        """Stop monitoring system"""
        logger.info("Stopping Tennis Betting Monitor...")
        
        self.is_running = False
        self.stop_event.set()
        
        logger.info("Tennis Betting Monitor stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        latest_system = self.system_metrics_history[-1] if self.system_metrics_history else None
        latest_betting = self.betting_metrics_history[-1] if self.betting_metrics_history else None
        
        recent_alerts = [alert for alert in self.alerts if alert.timestamp > datetime.now() - timedelta(hours=24)]
        
        return {
            'monitor_running': self.is_running,
            'latest_system_metrics': latest_system.to_dict() if latest_system else None,
            'latest_betting_metrics': latest_betting.to_dict() if latest_betting else None,
            'alerts_24h': len(recent_alerts),
            'critical_alerts_24h': len([a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL]),
            'error_count_current_hour': self.error_count,
            'notification_systems': {
                'telegram_enabled': self.telegram_notifier.enabled,
                'email_enabled': self.email_notifier.enabled
            },
            'thresholds': self.thresholds
        }
    
    def get_metrics_history(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics history for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        system_history = [m for m in self.system_metrics_history if m.timestamp > cutoff_time]
        betting_history = [m for m in self.betting_metrics_history if m.timestamp > cutoff_time]
        
        return {
            'system_metrics': [m.to_dict() for m in system_history],
            'betting_metrics': [m.to_dict() for m in betting_history],
            'time_range_hours': hours
        }
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_time]
        
        return [alert.to_dict() for alert in recent_alerts]


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create monitor
    monitor = TennisBettingMonitor()
    
    # Add alert callback
    def handle_alert(alert: Alert):
        print(f"ALERT: {alert.severity.value} - {alert.message}")
    
    monitor.add_alert_callback(handle_alert)
    
    try:
        # Start monitoring
        monitor.start()
        
        # Run for a while
        time.sleep(120)
        
        # Get status
        status = monitor.get_status()
        print(f"Monitor Status: {status}")
        
    except KeyboardInterrupt:
        monitor.stop()