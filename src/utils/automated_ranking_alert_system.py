#!/usr/bin/env python3
"""
Automated Tennis Ranking Alert System
Continuously monitors ranking data accuracy and sends automated alerts for discrepancies
"""

import os
import sys
import json
import logging
import asyncio
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ranking_accuracy_validator import RankingAccuracyValidator
from utils.realtime_ranking_monitor import RealtimeRankingMonitor
from utils.telegram_notification_system import TelegramNotificationSystem

logger = logging.getLogger(__name__)

@dataclass
class AlertRule:
    """Configuration for ranking alert rules"""
    name: str
    discrepancy_threshold: int
    priority: str  # LOW, MEDIUM, HIGH, CRITICAL
    alert_frequency_minutes: int
    players: List[str]  # Empty list means all players
    enabled: bool = True

class AutomatedRankingAlertSystem:
    """Automated system for monitoring and alerting on ranking discrepancies"""
    
    def __init__(self):
        self.is_running = False
        self.alert_thread = None
        self.last_alert_times = {}
        
        # Initialize components
        self.validator = RankingAccuracyValidator()
        self.monitor = RealtimeRankingMonitor()
        self.telegram_system = TelegramNotificationSystem()
        
        # Alert rules configuration
        self.alert_rules = [
            AlertRule(
                name="Critical_Financial_Risk",
                discrepancy_threshold=50,
                priority="CRITICAL",
                alert_frequency_minutes=15,  # Alert every 15 minutes for critical issues
                players=[]  # All players
            ),
            AlertRule(
                name="High_Risk_TODO_Players",
                discrepancy_threshold=20,
                priority="HIGH",
                alert_frequency_minutes=30,
                players=["linda noskova", "l. noskova", "ajla tomljanovic", "a. tomljanovic"]
            ),
            AlertRule(
                name="Medium_Risk_Monitoring",
                discrepancy_threshold=10,
                priority="MEDIUM",
                alert_frequency_minutes=60,
                players=["ekaterina alexandrova", "e. alexandrova"]
            ),
            AlertRule(
                name="General_Accuracy_Monitor",
                discrepancy_threshold=5,
                priority="LOW",
                alert_frequency_minutes=240,  # Every 4 hours
                players=[]
            )
        ]
        
        # System configuration
        self.monitoring_interval = 180  # 3 minutes between checks
        self.alert_log_file = "/home/apps/Tennis_one_set/logs/automated_ranking_alerts.log"
        self.alert_history_file = "/home/apps/Tennis_one_set/data/alert_history.json"
        
        self.setup_logging()
        self.load_alert_history()
    
    def setup_logging(self):
        """Setup logging for the alert system"""
        log_dir = os.path.dirname(self.alert_log_file)
        os.makedirs(log_dir, exist_ok=True)
        
        # Create alert-specific logger
        self.alert_logger = logging.getLogger('automated_ranking_alerts')
        self.alert_logger.setLevel(logging.INFO)
        
        # Add file handler if not already present
        if not any(isinstance(h, logging.FileHandler) for h in self.alert_logger.handlers):
            file_handler = logging.FileHandler(self.alert_log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            self.alert_logger.addHandler(file_handler)
    
    def load_alert_history(self):
        """Load alert history from file"""
        try:
            if os.path.exists(self.alert_history_file):
                with open(self.alert_history_file, 'r') as f:
                    data = json.load(f)
                    self.last_alert_times = data.get('last_alert_times', {})
                    self.alert_logger.info(f"📄 Loaded alert history: {len(self.last_alert_times)} entries")
            else:
                self.last_alert_times = {}
                self.alert_logger.info("🆕 No existing alert history found, starting fresh")
        except Exception as e:
            self.alert_logger.error(f"❌ Error loading alert history: {e}")
            self.last_alert_times = {}
    
    def save_alert_history(self):
        """Save alert history to file"""
        try:
            history_dir = os.path.dirname(self.alert_history_file)
            os.makedirs(history_dir, exist_ok=True)
            
            data = {
                'last_updated': datetime.now().isoformat(),
                'last_alert_times': self.last_alert_times
            }
            
            with open(self.alert_history_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.alert_logger.debug("💾 Alert history saved")
            
        except Exception as e:
            self.alert_logger.error(f"❌ Error saving alert history: {e}")
    
    def check_alert_rules(self, validation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check validation results against alert rules"""
        alerts_to_send = []
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
            
            # Check each validation source
            for source_name, source_data in validation_results.get("validation_sources", {}).items():
                if "players_tested" not in source_data:
                    continue
                
                # Check players
                for player_name, player_result in source_data["players_tested"].items():
                    if "discrepancy" not in player_result:
                        continue
                    
                    discrepancy = player_result["discrepancy"]
                    
                    # Check if this rule applies to this player
                    if rule.players and player_name not in rule.players:
                        continue
                    
                    # Check if discrepancy exceeds threshold
                    if discrepancy >= rule.discrepancy_threshold:
                        alert_key = f"{rule.name}_{player_name}_{source_name}"
                        
                        # Check if enough time has passed since last alert
                        if self._should_send_alert(alert_key, rule.alert_frequency_minutes):
                            alerts_to_send.append({
                                "rule": rule,
                                "player_name": player_name,
                                "player_result": player_result,
                                "source_name": source_name,
                                "alert_key": alert_key
                            })
        
        return alerts_to_send
    
    def _should_send_alert(self, alert_key: str, frequency_minutes: int) -> bool:
        """Check if enough time has passed to send this alert again"""
        if alert_key not in self.last_alert_times:
            return True
        
        last_alert_str = self.last_alert_times[alert_key]
        last_alert = datetime.fromisoformat(last_alert_str)
        time_since_alert = (datetime.now() - last_alert).total_seconds() / 60
        
        return time_since_alert >= frequency_minutes
    
    async def send_alerts(self, alerts: List[Dict[str, Any]]):
        """Send alerts via Telegram"""
        if not alerts:
            return
        
        self.alert_logger.info(f"📤 Sending {len(alerts)} ranking alerts")
        
        for alert in alerts:
            try:
                # Format alert message
                message = self._format_alert_message(alert)
                
                # Send alert
                if alert["rule"].priority == "CRITICAL":
                    # Use emergency alert for critical issues
                    await self.telegram_system._send_emergency_alert(message)
                else:
                    # Use regular message for other priorities
                    for chat_id in self.telegram_system.config.chat_ids:
                        await self.telegram_system._send_message(chat_id, message)
                
                # Record that alert was sent
                self.last_alert_times[alert["alert_key"]] = datetime.now().isoformat()
                
                self.alert_logger.info(f"✅ Alert sent: {alert['rule'].name} for {alert['player_name']}")
                
            except Exception as e:
                self.alert_logger.error(f"❌ Error sending alert for {alert['player_name']}: {e}")
        
        # Save updated alert history
        self.save_alert_history()
    
    def _format_alert_message(self, alert: Dict[str, Any]) -> str:
        """Format alert message for Telegram"""
        rule = alert["rule"]
        player_name = alert["player_name"]
        player_result = alert["player_result"]
        source_name = alert["source_name"]
        
        # Choose emoji and urgency based on priority
        priority_config = {
            "CRITICAL": {"emoji": "🔴", "urgency": "CRITICAL ALERT"},
            "HIGH": {"emoji": "🚨", "urgency": "HIGH ALERT"},
            "MEDIUM": {"emoji": "⚠️", "urgency": "WARNING"},
            "LOW": {"emoji": "ℹ️", "urgency": "INFO"}
        }
        
        config = priority_config.get(rule.priority, priority_config["MEDIUM"])
        
        message = f"{config['emoji']} {config['urgency']}: RANKING DISCREPANCY\n\n"
        message += f"<b>Alert Rule:</b> {rule.name}\n"
        message += f"<b>Player:</b> {player_name.title()}\n"
        message += f"<b>Data Source:</b> {source_name}\n"
        message += f"<b>Expected Rank:</b> #{player_result.get('expected_rank', 'N/A')}\n"
        message += f"<b>System Rank:</b> #{player_result.get('api_rank', player_result.get('hardcoded_rank', 'N/A'))}\n"
        message += f"<b>Discrepancy:</b> {player_result['discrepancy']} positions\n\n"
        
        # Add priority-specific guidance
        if rule.priority == "CRITICAL":
            message += "🚨 <b>IMMEDIATE ACTION REQUIRED</b>\n"
            message += "This ranking error poses significant financial risk!\n"
            message += "• Stop automated betting until resolved\n"
            message += "• Manually verify player rankings\n"
            message += "• Update ranking data sources\n"
        elif rule.priority == "HIGH":
            message += "⚡ <b>Urgent attention needed</b>\n"
            message += "Review ranking data for this player immediately\n"
        elif rule.priority == "MEDIUM":
            message += "📊 Monitor this discrepancy closely\n"
        else:
            message += "📝 For your information and tracking\n"
        
        message += f"\n⏰ <i>Alert generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</i>"
        
        return message
    
    def run_monitoring_cycle(self):
        """Run one complete monitoring and alerting cycle"""
        try:
            self.alert_logger.info("🔄 Starting automated ranking alert cycle")
            
            # Run comprehensive validation
            validation_results = self.validator.run_comprehensive_validation()
            
            # Check alert rules
            alerts = self.check_alert_rules(validation_results)
            
            # Send alerts if any
            if alerts:
                asyncio.run(self.send_alerts(alerts))
                self.alert_logger.warning(f"🚨 Sent {len(alerts)} ranking alerts")
            else:
                self.alert_logger.info("✅ No ranking alerts needed this cycle")
            
            # Log cycle summary
            total_issues = sum(
                source.get("summary", {}).get("critical_errors", 0) + 
                source.get("summary", {}).get("warnings", 0)
                for source in validation_results.get("validation_sources", {}).values()
            )
            
            self.alert_logger.info(f"📊 Monitoring cycle complete: {total_issues} total issues detected")
            
        except Exception as e:
            self.alert_logger.error(f"❌ Error in monitoring cycle: {e}")
    
    def start_automated_monitoring(self):
        """Start automated monitoring and alerting"""
        if self.is_running:
            self.alert_logger.warning("⚠️ Automated ranking alerts already running")
            return
        
        self.is_running = True
        self.alert_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.alert_thread.start()
        
        self.alert_logger.info(f"🚀 Automated ranking alert system started (checking every {self.monitoring_interval}s)")
    
    def stop_automated_monitoring(self):
        """Stop automated monitoring"""
        self.is_running = False
        if self.alert_thread:
            self.alert_logger.info("🛑 Stopping automated ranking alert system...")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Run monitoring cycle
                self.run_monitoring_cycle()
                
                # Wait for next cycle
                time.sleep(self.monitoring_interval)
                
            except KeyboardInterrupt:
                self.alert_logger.info("⏹️ Automated alerts stopped by user")
                break
            except Exception as e:
                self.alert_logger.error(f"❌ Error in alert monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
        
        self.is_running = False
        self.alert_logger.info("🏁 Automated ranking alert system stopped")
    
    def get_alert_system_status(self) -> Dict[str, Any]:
        """Get status of the alert system"""
        return {
            "is_running": self.is_running,
            "monitoring_interval_seconds": self.monitoring_interval,
            "active_alert_rules": len([r for r in self.alert_rules if r.enabled]),
            "total_alert_rules": len(self.alert_rules),
            "last_alert_count": len(self.last_alert_times),
            "telegram_enabled": self.telegram_system.config.enabled,
            "alert_rules": [
                {
                    "name": rule.name,
                    "priority": rule.priority,
                    "threshold": rule.discrepancy_threshold,
                    "frequency_minutes": rule.alert_frequency_minutes,
                    "enabled": rule.enabled
                }
                for rule in self.alert_rules
            ]
        }

def main():
    """Main function for testing the automated alert system"""
    print("🤖 Automated Tennis Ranking Alert System")
    print("=" * 50)
    
    alert_system = AutomatedRankingAlertSystem()
    
    print(f"\n📊 Alert System Status:")
    status = alert_system.get_alert_system_status()
    print(f"   Active Rules: {status['active_alert_rules']}/{status['total_alert_rules']}")
    print(f"   Monitoring Interval: {status['monitoring_interval_seconds']}s")
    print(f"   Telegram Enabled: {status['telegram_enabled']}")
    
    print(f"\n🔧 Alert Rules:")
    for rule_info in status["alert_rules"]:
        enabled_status = "✅" if rule_info["enabled"] else "❌"
        print(f"   {enabled_status} {rule_info['name']} ({rule_info['priority']}) - {rule_info['threshold']}+ positions")
    
    print(f"\n🔍 Running single monitoring cycle...")
    alert_system.run_monitoring_cycle()
    
    print(f"\n💡 To start continuous monitoring:")
    print(f"   alert_system.start_automated_monitoring()")
    
    print(f"\n✅ Automated alert system test complete!")

if __name__ == "__main__":
    main()