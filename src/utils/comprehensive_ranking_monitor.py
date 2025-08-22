#!/usr/bin/env python3
"""
Comprehensive Tennis Ranking Data Accuracy Monitoring System
Integrates all ranking monitoring components into a unified system
"""

import os
import sys
import json
import logging
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ranking_accuracy_validator import RankingAccuracyValidator
from utils.realtime_ranking_monitor import RealtimeRankingMonitor  
from utils.automated_ranking_alert_system import AutomatedRankingAlertSystem
from utils.telegram_notification_system import TelegramNotificationSystem

logger = logging.getLogger(__name__)

@dataclass
class MonitoringReport:
    """Comprehensive monitoring report"""
    timestamp: datetime
    validation_results: Dict[str, Any]
    critical_issues: List[Dict[str, Any]]
    alerts_sent: int
    accuracy_metrics: Dict[str, float]
    recommendations: List[str]
    system_health: str  # HEALTHY, WARNING, CRITICAL

class ComprehensiveRankingMonitor:
    """Unified system for comprehensive tennis ranking data accuracy monitoring"""
    
    def __init__(self):
        # Core monitoring components
        self.validator = RankingAccuracyValidator()
        self.realtime_monitor = RealtimeRankingMonitor()
        self.alert_system = AutomatedRankingAlertSystem()
        self.telegram_system = TelegramNotificationSystem()
        
        # System configuration
        self.comprehensive_check_interval = 900  # 15 minutes
        self.is_running = False
        self.monitor_thread = None
        
        # Performance tracking
        self.monitoring_history = []
        self.max_history_length = 100
        
        # File paths
        self.comprehensive_log_file = "/home/apps/Tennis_one_set/logs/comprehensive_ranking_monitor.log"
        self.dashboard_data_file = "/home/apps/Tennis_one_set/data/ranking_monitor_dashboard.json"
        self.performance_metrics_file = "/home/apps/Tennis_one_set/data/ranking_performance_metrics.json"
        
        self.setup_logging()
        self.load_monitoring_history()
    
    def setup_logging(self):
        """Setup comprehensive monitoring logging"""
        log_dir = os.path.dirname(self.comprehensive_log_file)
        os.makedirs(log_dir, exist_ok=True)
        
        # Create comprehensive monitor logger
        self.monitor_logger = logging.getLogger('comprehensive_ranking_monitor')
        self.monitor_logger.setLevel(logging.INFO)
        
        # Add file handler if not already present
        if not any(isinstance(h, logging.FileHandler) for h in self.monitor_logger.handlers):
            file_handler = logging.FileHandler(self.comprehensive_log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            self.monitor_logger.addHandler(file_handler)
    
    def load_monitoring_history(self):
        """Load previous monitoring history"""
        try:
            if os.path.exists(self.performance_metrics_file):
                with open(self.performance_metrics_file, 'r') as f:
                    data = json.load(f)
                    # Convert string timestamps back to datetime objects for recent history
                    self.monitoring_history = data.get('recent_history', [])[-50:]  # Keep last 50
                    self.monitor_logger.info(f"📄 Loaded {len(self.monitoring_history)} monitoring history entries")
            else:
                self.monitoring_history = []
                self.monitor_logger.info("🆕 Starting fresh monitoring history")
        except Exception as e:
            self.monitor_logger.error(f"❌ Error loading monitoring history: {e}")
            self.monitoring_history = []
    
    def run_comprehensive_check(self) -> MonitoringReport:
        """Run comprehensive ranking accuracy check"""
        self.monitor_logger.info("🔍 Starting comprehensive ranking accuracy check")
        
        start_time = datetime.now()
        
        try:
            # 1. Run validation across all sources
            validation_results = self.validator.run_comprehensive_validation()
            
            # 2. Run real-time monitoring for critical players
            realtime_results = self.realtime_monitor.run_monitoring_cycle()
            
            # 3. Calculate accuracy metrics
            accuracy_metrics = self._calculate_accuracy_metrics(validation_results, realtime_results)
            
            # 4. Identify critical issues
            critical_issues = self._identify_critical_issues(validation_results, realtime_results)
            
            # 5. Check if alerts need to be sent
            alerts_sent = 0
            if critical_issues:
                alerts = self.alert_system.check_alert_rules(validation_results)
                if alerts:
                    asyncio.run(self.alert_system.send_alerts(alerts))
                    alerts_sent = len(alerts)
            
            # 6. Generate recommendations
            recommendations = self._generate_recommendations(validation_results, critical_issues, accuracy_metrics)
            
            # 7. Determine system health status
            system_health = self._determine_system_health(accuracy_metrics, critical_issues)
            
            # Create comprehensive report
            report = MonitoringReport(
                timestamp=start_time,
                validation_results=validation_results,
                critical_issues=critical_issues,
                alerts_sent=alerts_sent,
                accuracy_metrics=accuracy_metrics,
                recommendations=recommendations,
                system_health=system_health
            )
            
            # Save report and update history
            self._save_monitoring_report(report)
            self._update_monitoring_history(report)
            
            # Log summary
            duration = (datetime.now() - start_time).total_seconds()
            self.monitor_logger.info(
                f"✅ Comprehensive check complete in {duration:.1f}s: "
                f"{system_health} status, {len(critical_issues)} critical issues, {alerts_sent} alerts sent"
            )
            
            return report
            
        except Exception as e:
            self.monitor_logger.error(f"❌ Error in comprehensive check: {e}")
            raise
    
    def _calculate_accuracy_metrics(self, validation_results: Dict[str, Any], realtime_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate comprehensive accuracy metrics"""
        metrics = {
            "overall_accuracy_rate": 0.0,
            "dynamic_api_accuracy": 0.0,
            "enhanced_integration_accuracy": 0.0,
            "critical_player_accuracy": 0.0,
            "average_discrepancy": 0.0,
            "max_discrepancy": 0.0
        }
        
        try:
            # Calculate overall accuracy from validation results
            total_tested = 0
            total_accurate = 0
            all_discrepancies = []
            
            for source_name, source_data in validation_results.get("validation_sources", {}).items():
                if "summary" in source_data:
                    summary = source_data["summary"]
                    source_total = summary.get("total_tested", 0)
                    source_accurate = summary.get("accurate", 0)
                    
                    total_tested += source_total
                    total_accurate += source_accurate
                    
                    # Calculate source-specific accuracy
                    source_accuracy = source_accurate / source_total if source_total > 0 else 0
                    if source_name == "dynamic_api":
                        metrics["dynamic_api_accuracy"] = source_accuracy
                    elif source_name == "enhanced_integration":
                        metrics["enhanced_integration_accuracy"] = source_accuracy
                
                # Collect discrepancies
                if "players_tested" in source_data:
                    for player_result in source_data["players_tested"].values():
                        if "discrepancy" in player_result:
                            all_discrepancies.append(player_result["discrepancy"])
            
            # Overall accuracy
            metrics["overall_accuracy_rate"] = total_accurate / total_tested if total_tested > 0 else 0
            
            # Critical player accuracy from realtime results
            critical_accurate = len([r for r in realtime_results if r.get("alert_level") == "none"])
            critical_total = len(realtime_results)
            metrics["critical_player_accuracy"] = critical_accurate / critical_total if critical_total > 0 else 0
            
            # Discrepancy metrics
            if all_discrepancies:
                metrics["average_discrepancy"] = sum(all_discrepancies) / len(all_discrepancies)
                metrics["max_discrepancy"] = max(all_discrepancies)
            
        except Exception as e:
            self.monitor_logger.error(f"❌ Error calculating accuracy metrics: {e}")
        
        return metrics
    
    def _identify_critical_issues(self, validation_results: Dict[str, Any], realtime_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify critical ranking issues that need immediate attention"""
        critical_issues = []
        
        try:
            # From validation results
            for issue in validation_results.get("critical_issues", []):
                critical_issues.append({
                    "type": "validation_critical",
                    "source": issue.get("source", "unknown"),
                    "issue": issue.get("issue", ""),
                    "impact": issue.get("impact", ""),
                    "priority": "HIGH"
                })
            
            # From realtime monitoring
            for result in realtime_results:
                if result.get("alert_level") in ["critical", "high"]:
                    critical_issues.append({
                        "type": "realtime_critical",
                        "player": result.get("player", "unknown"),
                        "issue": f"Ranking discrepancy: {result.get('discrepancy', 'unknown')} positions",
                        "expected_rank": result.get("expected_rank"),
                        "current_rank": result.get("current_rank"),
                        "priority": "CRITICAL" if result.get("alert_level") == "critical" else "HIGH"
                    })
            
        except Exception as e:
            self.monitor_logger.error(f"❌ Error identifying critical issues: {e}")
        
        return critical_issues
    
    def _generate_recommendations(self, validation_results: Dict[str, Any], critical_issues: List[Dict[str, Any]], accuracy_metrics: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations based on monitoring results"""
        recommendations = []
        
        try:
            # Based on accuracy metrics
            if accuracy_metrics["overall_accuracy_rate"] < 0.8:
                recommendations.append("URGENT: Overall ranking accuracy below 80% - immediate system review required")
            elif accuracy_metrics["overall_accuracy_rate"] < 0.95:
                recommendations.append("Moderate: Ranking accuracy could be improved - review data sources")
            
            # Based on critical issues
            critical_count = len([i for i in critical_issues if i.get("priority") == "CRITICAL"])
            if critical_count > 0:
                recommendations.append(f"CRITICAL: {critical_count} critical ranking issues require immediate attention")
            
            high_count = len([i for i in critical_issues if i.get("priority") == "HIGH"])
            if high_count > 0:
                recommendations.append(f"HIGH: {high_count} high-priority ranking issues need review")
            
            # Specific API recommendations
            if accuracy_metrics["dynamic_api_accuracy"] < 0.9:
                recommendations.append("Check dynamic rankings API configuration and data sources")
            
            if accuracy_metrics["enhanced_integration_accuracy"] < 0.9:
                recommendations.append("Review enhanced ranking integration implementation")
            
            # Based on discrepancies
            if accuracy_metrics["max_discrepancy"] > 100:
                recommendations.append("URGENT: Maximum ranking discrepancy exceeds 100 positions - data integrity issue")
            elif accuracy_metrics["max_discrepancy"] > 50:
                recommendations.append("WARNING: Large ranking discrepancies detected - verify data sources")
            
            # Operational recommendations
            if not recommendations:  # System is healthy
                recommendations.append("System operating within normal parameters - continue monitoring")
            
            recommendations.append("Ensure automated alerts are configured and functioning")
            recommendations.append("Review betting decisions for affected players")
            
        except Exception as e:
            self.monitor_logger.error(f"❌ Error generating recommendations: {e}")
            recommendations.append("Error generating recommendations - manual system review recommended")
        
        return recommendations
    
    def _determine_system_health(self, accuracy_metrics: Dict[str, float], critical_issues: List[Dict[str, Any]]) -> str:
        """Determine overall system health status"""
        try:
            critical_count = len([i for i in critical_issues if i.get("priority") == "CRITICAL"])
            overall_accuracy = accuracy_metrics.get("overall_accuracy_rate", 0)
            max_discrepancy = accuracy_metrics.get("max_discrepancy", 0)
            
            # Critical conditions
            if critical_count > 0 or overall_accuracy < 0.7 or max_discrepancy > 100:
                return "CRITICAL"
            
            # Warning conditions
            high_count = len([i for i in critical_issues if i.get("priority") == "HIGH"])
            if high_count > 0 or overall_accuracy < 0.9 or max_discrepancy > 50:
                return "WARNING"
            
            # Healthy
            return "HEALTHY"
            
        except Exception as e:
            self.monitor_logger.error(f"❌ Error determining system health: {e}")
            return "UNKNOWN"
    
    def _save_monitoring_report(self, report: MonitoringReport):
        """Save comprehensive monitoring report"""
        try:
            data_dir = os.path.dirname(self.dashboard_data_file)
            os.makedirs(data_dir, exist_ok=True)
            
            # Convert report to dict for JSON serialization
            report_dict = asdict(report)
            report_dict["timestamp"] = report.timestamp.isoformat()
            
            # Add summary for dashboard
            dashboard_data = {
                "last_updated": report.timestamp.isoformat(),
                "system_health": report.system_health,
                "accuracy_metrics": report.accuracy_metrics,
                "critical_issues_count": len(report.critical_issues),
                "alerts_sent": report.alerts_sent,
                "top_recommendations": report.recommendations[:3],
                "detailed_report": report_dict
            }
            
            with open(self.dashboard_data_file, 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
            
            self.monitor_logger.info(f"📄 Monitoring report saved to {self.dashboard_data_file}")
            
        except Exception as e:
            self.monitor_logger.error(f"❌ Error saving monitoring report: {e}")
    
    def _update_monitoring_history(self, report: MonitoringReport):
        """Update monitoring history for trend analysis"""
        try:
            # Add to history
            history_entry = {
                "timestamp": report.timestamp.isoformat(),
                "system_health": report.system_health,
                "overall_accuracy": report.accuracy_metrics.get("overall_accuracy_rate", 0),
                "critical_issues_count": len(report.critical_issues),
                "alerts_sent": report.alerts_sent
            }
            
            self.monitoring_history.append(history_entry)
            
            # Keep only recent history
            if len(self.monitoring_history) > self.max_history_length:
                self.monitoring_history = self.monitoring_history[-self.max_history_length:]
            
            # Save updated history
            history_data = {
                "last_updated": datetime.now().isoformat(),
                "total_entries": len(self.monitoring_history),
                "recent_history": self.monitoring_history
            }
            
            metrics_dir = os.path.dirname(self.performance_metrics_file)
            os.makedirs(metrics_dir, exist_ok=True)
            
            with open(self.performance_metrics_file, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)
            
        except Exception as e:
            self.monitor_logger.error(f"❌ Error updating monitoring history: {e}")
    
    def start_comprehensive_monitoring(self):
        """Start comprehensive monitoring system"""
        if self.is_running:
            self.monitor_logger.warning("⚠️ Comprehensive monitoring already running")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        # Also start component monitoring systems
        self.realtime_monitor.start_monitoring()
        self.alert_system.start_automated_monitoring()
        
        self.monitor_logger.info(f"🚀 Comprehensive ranking monitoring system started (checking every {self.comprehensive_check_interval}s)")
    
    def stop_comprehensive_monitoring(self):
        """Stop comprehensive monitoring system"""
        self.is_running = False
        
        # Stop component systems
        self.realtime_monitor.stop_monitoring()
        self.alert_system.stop_automated_monitoring()
        
        if self.monitor_thread:
            self.monitor_logger.info("🛑 Stopping comprehensive ranking monitoring system...")
    
    def _monitoring_loop(self):
        """Main comprehensive monitoring loop"""
        while self.is_running:
            try:
                # Run comprehensive check
                report = self.run_comprehensive_check()
                
                # Send critical system alerts if needed
                if report.system_health == "CRITICAL":
                    asyncio.run(self._send_system_critical_alert(report))
                
                # Wait for next cycle
                time.sleep(self.comprehensive_check_interval)
                
            except KeyboardInterrupt:
                self.monitor_logger.info("⏹️ Comprehensive monitoring stopped by user")
                break
            except Exception as e:
                self.monitor_logger.error(f"❌ Error in comprehensive monitoring loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
        
        self.is_running = False
        self.monitor_logger.info("🏁 Comprehensive ranking monitoring system stopped")
    
    async def _send_system_critical_alert(self, report: MonitoringReport):
        """Send system-wide critical alert"""
        try:
            message = f"🔴 SYSTEM CRITICAL: RANKING MONITOR ALERT\n\n"
            message += f"System Health: {report.system_health}\n"
            message += f"Overall Accuracy: {report.accuracy_metrics.get('overall_accuracy_rate', 0):.1%}\n"
            message += f"Critical Issues: {len(report.critical_issues)}\n"
            message += f"Max Discrepancy: {report.accuracy_metrics.get('max_discrepancy', 0)} positions\n\n"
            message += "🚨 IMMEDIATE ACTION REQUIRED\n"
            message += "• Review ranking data sources\n"
            message += "• Suspend automated betting\n"
            message += "• Manual verification needed\n\n"
            message += f"Check monitoring dashboard for details"
            
            await self.telegram_system._send_emergency_alert(message)
            
        except Exception as e:
            self.monitor_logger.error(f"❌ Error sending system critical alert: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Load latest dashboard data if available
            latest_report = None
            if os.path.exists(self.dashboard_data_file):
                with open(self.dashboard_data_file, 'r') as f:
                    latest_report = json.load(f)
            
            return {
                "monitoring_active": self.is_running,
                "check_interval_minutes": self.comprehensive_check_interval / 60,
                "components": {
                    "validator": "active",
                    "realtime_monitor": "active" if self.realtime_monitor.is_running else "inactive",
                    "alert_system": "active" if self.alert_system.is_running else "inactive",
                    "telegram_system": "enabled" if self.telegram_system.config.enabled else "disabled"
                },
                "latest_report": latest_report,
                "history_entries": len(self.monitoring_history)
            }
            
        except Exception as e:
            self.monitor_logger.error(f"❌ Error getting system status: {e}")
            return {"error": str(e)}

def main():
    """Main function for testing the comprehensive monitoring system"""
    print("🎯 Comprehensive Tennis Ranking Monitoring System")
    print("=" * 60)
    
    monitor = ComprehensiveRankingMonitor()
    
    print(f"\n📊 System Status:")
    status = monitor.get_system_status()
    print(f"   Monitoring Active: {status['monitoring_active']}")
    print(f"   Check Interval: {status['check_interval_minutes']} minutes")
    print(f"   Components:")
    for component, state in status["components"].items():
        status_emoji = "✅" if state == "active" or state == "enabled" else "❌"
        print(f"     {status_emoji} {component}: {state}")
    
    print(f"\n🔍 Running comprehensive check...")
    report = monitor.run_comprehensive_check()
    
    print(f"\n📋 Check Results:")
    print(f"   System Health: {report.system_health}")
    print(f"   Overall Accuracy: {report.accuracy_metrics.get('overall_accuracy_rate', 0):.1%}")
    print(f"   Critical Issues: {len(report.critical_issues)}")
    print(f"   Alerts Sent: {report.alerts_sent}")
    
    if report.recommendations:
        print(f"\n💡 Top Recommendations:")
        for i, rec in enumerate(report.recommendations[:3], 1):
            print(f"   {i}. {rec}")
    
    print(f"\n📄 Reports saved to:")
    print(f"   Dashboard: {monitor.dashboard_data_file}")
    print(f"   Logs: {monitor.comprehensive_log_file}")
    
    print(f"\n🚀 To start continuous monitoring:")
    print(f"   monitor.start_comprehensive_monitoring()")
    
    print(f"\n✅ Comprehensive monitoring system test complete!")

if __name__ == "__main__":
    main()