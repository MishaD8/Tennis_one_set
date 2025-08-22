#!/usr/bin/env python3
"""
Real-time Tennis Ranking Monitor
Specific monitoring for players mentioned in TODO.md with critical ranking issues
"""

import os
import sys
import json
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.dynamic_rankings_api import dynamic_rankings
from utils.ranking_accuracy_validator import RankingAccuracyValidator

logger = logging.getLogger(__name__)

class RealtimeRankingMonitor:
    """Real-time monitor for critical tennis ranking accuracy"""
    
    def __init__(self):
        self.critical_players = {
            # Players from TODO.md with known ranking issues
            "linda noskova": {
                "actual_rank": 23,
                "incorrect_rank_seen": 150,
                "tour": "wta",
                "priority": "HIGH",
                "issue": "127 positions off - significant financial risk"
            },
            "l. noskova": {
                "actual_rank": 23,
                "incorrect_rank_seen": 150,
                "tour": "wta",
                "priority": "HIGH",
                "issue": "Alternative name format"
            },
            "ekaterina alexandrova": {
                "actual_rank": 14,
                "incorrect_rank_seen": 13,
                "tour": "wta",
                "priority": "MEDIUM",
                "issue": "1 position off - minor but needs monitoring"
            },
            "e. alexandrova": {
                "actual_rank": 14,
                "incorrect_rank_seen": 13,
                "tour": "wta",
                "priority": "MEDIUM",
                "issue": "Alternative name format"
            },
            "ajla tomljanovic": {
                "actual_rank": 84,
                "incorrect_rank_seen": 250,
                "tour": "wta",
                "priority": "HIGH",
                "issue": "166 positions off - massive discrepancy"
            },
            "a. tomljanovic": {
                "actual_rank": 84,
                "incorrect_rank_seen": 250,
                "tour": "wta",
                "priority": "HIGH",
                "issue": "Alternative name format"
            }
        }
        
        self.monitor_interval = 300  # 5 minutes
        self.alert_threshold_high = 20    # positions
        self.alert_threshold_critical = 50  # positions
        
        self.is_running = False
        self.monitor_thread = None
        self.validator = RankingAccuracyValidator()
        
        # Alert tracking to avoid spam
        self.last_alert_times = {}
        self.alert_cooldown = 3600  # 1 hour between same alerts
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for real-time monitor"""
        log_file = "/home/apps/Tennis_one_set/logs/realtime_ranking_monitor.log"
        log_dir = os.path.dirname(log_file)
        os.makedirs(log_dir, exist_ok=True)
        
        # Create monitor-specific logger
        self.monitor_logger = logging.getLogger('realtime_ranking_monitor')
        self.monitor_logger.setLevel(logging.INFO)
        
        # Add file handler if not already present
        if not any(isinstance(h, logging.FileHandler) for h in self.monitor_logger.handlers):
            file_handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            self.monitor_logger.addHandler(file_handler)
    
    def check_player_ranking(self, player_name: str, expected_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check a specific player's ranking accuracy"""
        try:
            # Get current ranking from dynamic API
            ranking_data = dynamic_rankings.get_player_ranking(player_name, expected_data["tour"])
            current_rank = ranking_data.get("rank", 999)
            expected_rank = expected_data["actual_rank"]
            
            # Calculate discrepancy
            discrepancy = abs(current_rank - expected_rank)
            
            result = {
                "player": player_name,
                "expected_rank": expected_rank,
                "current_rank": current_rank,
                "discrepancy": discrepancy,
                "tour": expected_data["tour"],
                "check_time": datetime.now().isoformat(),
                "status": "accurate",
                "alert_level": "none"
            }
            
            # Determine alert level
            if discrepancy == 0:
                result["status"] = "accurate"
            elif discrepancy <= 5:
                result["status"] = "minor_variance"
                result["alert_level"] = "info"
            elif discrepancy <= self.alert_threshold_high:
                result["status"] = "warning"
                result["alert_level"] = "warning"
            elif discrepancy <= self.alert_threshold_critical:
                result["status"] = "high_alert"
                result["alert_level"] = "high"
            else:
                result["status"] = "critical_alert"
                result["alert_level"] = "critical"
            
            # Log result
            if result["alert_level"] == "none":
                self.monitor_logger.info(f"✅ {player_name}: Rank #{current_rank} (Expected #{expected_rank})")
            elif result["alert_level"] == "info":
                self.monitor_logger.info(f"ℹ️ {player_name}: Minor variance - Rank #{current_rank} vs #{expected_rank} (diff: {discrepancy})")
            elif result["alert_level"] == "warning":
                self.monitor_logger.warning(f"⚠️ {player_name}: Ranking discrepancy - Rank #{current_rank} vs #{expected_rank} (diff: {discrepancy})")
            elif result["alert_level"] == "high":
                self.monitor_logger.error(f"🚨 {player_name}: HIGH ALERT - Rank #{current_rank} vs #{expected_rank} (diff: {discrepancy})")
            else:
                self.monitor_logger.critical(f"🔴 {player_name}: CRITICAL ALERT - Rank #{current_rank} vs #{expected_rank} (diff: {discrepancy})")
            
            return result
            
        except Exception as e:
            self.monitor_logger.error(f"❌ Error checking {player_name}: {e}")
            return {
                "player": player_name,
                "error": str(e),
                "check_time": datetime.now().isoformat(),
                "status": "error",
                "alert_level": "error"
            }
    
    def run_monitoring_cycle(self) -> List[Dict[str, Any]]:
        """Run one complete monitoring cycle for all critical players"""
        self.monitor_logger.info("🔄 Starting monitoring cycle for critical players")
        
        results = []
        alerts_to_send = []
        
        for player_name, expected_data in self.critical_players.items():
            result = self.check_player_ranking(player_name, expected_data)
            results.append(result)
            
            # Check if we need to send an alert
            if result["alert_level"] in ["high", "critical"]:
                alert_key = f"{player_name}_{result['alert_level']}"
                
                # Check cooldown
                if self._should_send_alert(alert_key):
                    alerts_to_send.append({
                        "player": player_name,
                        "result": result,
                        "expected_data": expected_data
                    })
                    self.last_alert_times[alert_key] = datetime.now()
        
        # Send alerts if any
        if alerts_to_send:
            self._send_alerts(alerts_to_send)
        
        # Save monitoring report
        self._save_monitoring_report(results)
        
        return results
    
    def _should_send_alert(self, alert_key: str) -> bool:
        """Check if enough time has passed to send the same alert again"""
        if alert_key not in self.last_alert_times:
            return True
        
        last_alert = self.last_alert_times[alert_key]
        time_since_alert = (datetime.now() - last_alert).total_seconds()
        
        return time_since_alert >= self.alert_cooldown
    
    def _send_alerts(self, alerts: List[Dict[str, Any]]):
        """Send alerts for critical ranking issues"""
        try:
            # Import telegram system
            from telegram_notification_system import TelegramNotificationSystem
            
            telegram_system = TelegramNotificationSystem()
            
            if not telegram_system.config.enabled:
                self.monitor_logger.warning("⚠️ Telegram alerts disabled - logging only")
                return
            
            for alert in alerts:
                player = alert["player"]
                result = alert["result"]
                expected_data = alert["expected_data"]
                
                # Create alert message
                alert_message = self._format_alert_message(player, result, expected_data)
                
                # Send alert
                asyncio.run(self._send_telegram_alert(telegram_system, alert_message))
                
        except Exception as e:
            self.monitor_logger.error(f"❌ Failed to send alerts: {e}")
    
    def _format_alert_message(self, player: str, result: Dict[str, Any], expected_data: Dict[str, Any]) -> str:
        """Format alert message for critical ranking issues"""
        alert_level = result["alert_level"]
        discrepancy = result["discrepancy"]
        current_rank = result["current_rank"]
        expected_rank = result["expected_rank"]
        
        # Choose emoji based on alert level
        if alert_level == "critical":
            emoji = "🔴"
            urgency = "CRITICAL"
        else:
            emoji = "🚨"
            urgency = "HIGH ALERT"
        
        message = f"{emoji} {urgency}: RANKING DATA ISSUE\n\n"
        message += f"Player: {player.title()}\n"
        message += f"Expected Rank: #{expected_rank}\n"
        message += f"Current System Rank: #{current_rank}\n"
        message += f"Discrepancy: {discrepancy} positions\n"
        message += f"Tour: {expected_data['tour'].upper()}\n\n"
        message += f"Issue: {expected_data['issue']}\n\n"
        message += "⚠️ IMMEDIATE ACTION REQUIRED\n"
        message += "This ranking error could lead to incorrect betting decisions and financial losses!"
        
        return message
    
    async def _send_telegram_alert(self, telegram_system, message: str):
        """Send telegram alert"""
        try:
            # Send to all configured chat IDs
            for chat_id in telegram_system.config.chat_ids:
                await telegram_system._send_message(chat_id, message)
            
            self.monitor_logger.info("📱 Critical alert sent via Telegram")
            
        except Exception as e:
            self.monitor_logger.error(f"❌ Failed to send telegram alert: {e}")
    
    def _save_monitoring_report(self, results: List[Dict[str, Any]]):
        """Save monitoring results to file"""
        try:
            report_file = "/home/apps/Tennis_one_set/data/realtime_monitoring_report.json"
            report_dir = os.path.dirname(report_file)
            os.makedirs(report_dir, exist_ok=True)
            
            report = {
                "monitoring_cycle_time": datetime.now().isoformat(),
                "total_players_monitored": len(results),
                "results": results,
                "summary": {
                    "accurate": len([r for r in results if r.get("alert_level") == "none"]),
                    "warnings": len([r for r in results if r.get("alert_level") == "warning"]),
                    "high_alerts": len([r for r in results if r.get("alert_level") == "high"]),
                    "critical_alerts": len([r for r in results if r.get("alert_level") == "critical"]),
                    "errors": len([r for r in results if r.get("alert_level") == "error"])
                }
            }
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.monitor_logger.info(f"📄 Monitoring report saved to {report_file}")
            
        except Exception as e:
            self.monitor_logger.error(f"❌ Failed to save monitoring report: {e}")
    
    def start_monitoring(self):
        """Start continuous real-time monitoring"""
        if self.is_running:
            self.monitor_logger.warning("⚠️ Real-time monitoring already running")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.monitor_logger.info(f"🚀 Real-time ranking monitor started (checking every {self.monitor_interval}s)")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_logger.info("🛑 Stopping real-time ranking monitor...")
            # Thread will stop on next cycle
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Run monitoring cycle
                results = self.run_monitoring_cycle()
                
                # Log cycle summary
                summary_msg = f"📊 Cycle complete: "
                summary_msg += f"{len([r for r in results if r.get('alert_level') == 'none'])} accurate, "
                summary_msg += f"{len([r for r in results if r.get('alert_level') in ['warning', 'high', 'critical']])} alerts"
                
                self.monitor_logger.info(summary_msg)
                
                # Wait for next cycle
                time.sleep(self.monitor_interval)
                
            except KeyboardInterrupt:
                self.monitor_logger.info("⏹️ Monitoring stopped by user")
                break
            except Exception as e:
                self.monitor_logger.error(f"❌ Error in monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
        
        self.is_running = False
        self.monitor_logger.info("🏁 Real-time ranking monitor stopped")
    
    def run_single_check(self) -> Dict[str, Any]:
        """Run a single monitoring check (for testing/manual execution)"""
        self.monitor_logger.info("🔍 Running single ranking check for critical players")
        
        results = self.run_monitoring_cycle()
        
        # Print results to console
        print(f"\n🔍 Critical Player Ranking Check Results")
        print(f"=" * 50)
        print(f"Check Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        for result in results:
            if "error" in result:
                print(f"\n❌ {result['player']}: ERROR - {result['error']}")
            else:
                status_emoji = {
                    "accurate": "✅",
                    "minor_variance": "ℹ️",
                    "warning": "⚠️",
                    "high_alert": "🚨",
                    "critical_alert": "🔴"
                }.get(result["status"], "❓")
                
                print(f"\n{status_emoji} {result['player'].title()}")
                print(f"   Expected: #{result['expected_rank']}")
                print(f"   Current:  #{result['current_rank']}")
                if result['discrepancy'] > 0:
                    print(f"   Discrepancy: {result['discrepancy']} positions")
        
        print(f"\n📄 Full report saved to monitoring files")
        return {"results": results, "check_time": datetime.now().isoformat()}

def main():
    """Main function for testing the real-time monitor"""
    print("🔄 Real-time Tennis Ranking Monitor")
    print("=" * 50)
    
    monitor = RealtimeRankingMonitor()
    
    print("\n🔍 Running single check for critical players...")
    results = monitor.run_single_check()
    
    print("\n💡 Options:")
    print("1. Start continuous monitoring (run monitor.start_monitoring())")
    print("2. Check specific players as needed")
    print("3. Integration with main tennis prediction system")
    
    print("\n✅ Real-time monitor test complete!")

if __name__ == "__main__":
    main()