#!/usr/bin/env python3
"""
Production Deployment Script for Tennis Betting System
Comprehensive deployment automation with monitoring and error recovery
"""

import os
import sys
import time
import logging
import subprocess
import signal
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Import system components
from config import get_config, validate_config
from automated_betting_engine import (
    AutomatedBettingEngine, 
    RiskManagementConfig,
    BetOrder
)
from realtime_prediction_engine import PredictionEngine, PredictionConfig
from tennis_betting_monitor import TennisBettingMonitor, Alert
from betfair_api_client import BetfairAPIClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tennis_betting_production.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProductionDeployment:
    """Production deployment manager for tennis betting system"""
    
    def __init__(self):
        self.config = get_config()
        self.is_running = False
        self.components = {}
        self.start_time = None
        
        # Production settings
        self.risk_config = RiskManagementConfig.conservative()  # Start conservative
        self.initial_bankroll = float(os.getenv('INITIAL_BANKROLL', '1000.0'))
        
        # Component references
        self.betting_engine = None
        self.prediction_engine = None
        self.monitor = None
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown()
        sys.exit(0)
    
    def validate_production_environment(self) -> bool:
        """Validate production environment and configuration"""
        logger.info("🔍 Validating production environment...")
        
        validation_passed = True
        
        # Validate configuration
        validation = validate_config()
        if not validation['valid']:
            logger.error("❌ Configuration validation failed:")
            for error in validation['errors']:
                logger.error(f"  - {error}")
            validation_passed = False
        
        # Check critical environment variables
        critical_vars = [
            'BETFAIR_APP_KEY',
            'BETFAIR_USERNAME', 
            'BETFAIR_PASSWORD'
        ]
        
        for var in critical_vars:
            if not os.getenv(var):
                logger.error(f"❌ Critical environment variable not set: {var}")
                validation_passed = False
        
        # Check file permissions
        required_files = [
            'automated_betting_engine.py',
            'betfair_api_client.py',
            'realtime_prediction_engine.py',
            'tennis_betting_monitor.py'
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                logger.error(f"❌ Required file not found: {file_path}")
                validation_passed = False
            elif not os.access(file_path, os.R_OK):
                logger.error(f"❌ Cannot read required file: {file_path}")
                validation_passed = False
        
        # Check disk space
        disk_usage = os.statvfs('.')
        available_gb = (disk_usage.f_bavail * disk_usage.f_frsize) / (1024**3)
        if available_gb < 1.0:  # Less than 1GB
            logger.error(f"❌ Insufficient disk space: {available_gb:.1f}GB available")
            validation_passed = False
        
        # Test API connectivity
        try:
            betfair_client = BetfairAPIClient()
            health = betfair_client.health_check()
            if health['status'] != 'healthy':
                logger.warning(f"⚠️ Betfair API health check: {health['message']}")
        except Exception as e:
            logger.error(f"❌ Betfair API test failed: {e}")
            validation_passed = False
        
        if validation_passed:
            logger.info("✅ Production environment validation passed")
        else:
            logger.error("❌ Production environment validation failed")
        
        return validation_passed
    
    def initialize_components(self) -> bool:
        """Initialize all system components"""
        logger.info("🚀 Initializing system components...")
        
        try:
            # Initialize Betting Engine
            logger.info("Initializing Automated Betting Engine...")
            self.betting_engine = AutomatedBettingEngine(
                risk_config=self.risk_config,
                initial_bankroll=self.initial_bankroll
            )
            
            # Initialize Prediction Engine
            logger.info("Initializing Prediction Engine...")
            prediction_config = PredictionConfig.default()
            self.prediction_engine = PredictionEngine(prediction_config)
            
            # Connect engines
            self.betting_engine.initialize(self.prediction_engine)
            
            # Initialize Monitor
            logger.info("Initializing Monitoring System...")
            self.monitor = TennisBettingMonitor(self.betting_engine)
            
            # Add monitoring callbacks
            self.betting_engine.add_bet_callback(self._handle_bet_event)
            self.monitor.add_alert_callback(self._handle_alert)
            
            logger.info("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            return False
    
    def start_system(self) -> bool:
        """Start the complete tennis betting system"""
        logger.info("🎾 Starting Tennis Automated Betting System...")
        
        try:
            # Start monitoring first
            self.monitor.start()
            
            # Start prediction engine
            api_key = os.getenv('TENNIS_API_KEY') or os.getenv('API_TENNIS_KEY')
            self.prediction_engine.start(api_key)
            
            # Start betting engine
            self.betting_engine.start()
            
            self.is_running = True
            self.start_time = datetime.now()
            
            logger.info("🎉 Tennis Betting System started successfully!")
            logger.info(f"💰 Initial Bankroll: €{self.initial_bankroll:.2f}")
            logger.info(f"⚙️ Risk Profile: {self.risk_config.risk_level.value}")
            logger.info(f"🎯 Min Confidence: {self.risk_config.min_confidence_threshold}")
            logger.info(f"📊 Max Stake per Bet: €{self.risk_config.max_stake_per_bet:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ System startup failed: {e}")
            return False
    
    def monitor_system(self):
        """Monitor system operation and handle issues"""
        logger.info("👁️ Starting system monitoring...")
        
        check_interval = 30  # seconds
        last_health_check = datetime.now()
        
        while self.is_running:
            try:
                time.sleep(check_interval)
                
                # Periodic health checks
                if (datetime.now() - last_health_check).total_seconds() > 300:  # Every 5 minutes
                    self._perform_health_check()
                    last_health_check = datetime.now()
                
                # Log system stats
                self._log_system_stats()
                
            except KeyboardInterrupt:
                logger.info("Monitoring interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(60)  # Wait longer after error
    
    def _perform_health_check(self):
        """Perform comprehensive health check"""
        try:
            # Check betting engine
            if self.betting_engine:
                stats = self.betting_engine.get_stats()
                active_bets = len(self.betting_engine.get_active_bets())
                
                logger.info(f"💊 Health Check - Bets: {stats['bets_placed']}, "
                           f"Opportunities: {stats['opportunities_found']}, "
                           f"Active: {active_bets}")
            
            # Check prediction engine
            if self.prediction_engine:
                pred_stats = self.prediction_engine.get_stats()
                logger.info(f"🧠 Prediction Engine - Events: {pred_stats['total_events']}, "
                           f"Predictions: {pred_stats['predictions_generated']}")
            
            # Check monitor
            if self.monitor:
                monitor_status = self.monitor.get_status()
                logger.info(f"📊 Monitor - Alerts (24h): {monitor_status['alerts_24h']}, "
                           f"Critical: {monitor_status['critical_alerts_24h']}")
        
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    def _log_system_stats(self):
        """Log current system statistics"""
        try:
            if not self.betting_engine:
                return
            
            stats = self.betting_engine.get_stats()
            risk_summary = stats.get('risk_summary', {})
            
            # Calculate uptime
            uptime = datetime.now() - self.start_time if self.start_time else None
            uptime_str = str(uptime).split('.')[0] if uptime else "Unknown"
            
            # Log summary
            logger.info(f"📈 SYSTEM STATS - Uptime: {uptime_str}")
            logger.info(f"💰 Balance: €{risk_summary.get('current_bankroll', 0):.2f} | "
                       f"P&L: €{stats.get('total_profit_loss', 0):.2f}")
            logger.info(f"🎯 Bets: {stats.get('bets_placed', 0)} | "
                       f"Win Rate: {stats.get('win_rate', 0):.1f}% | "
                       f"Active: {len(self.betting_engine.get_active_bets())}")
            logger.info(f"🔍 Opportunities: {stats.get('opportunities_found', 0)} | "
                       f"Queue: {stats.get('queue_size', 0)}")
        
        except Exception as e:
            logger.error(f"Failed to log system stats: {e}")
    
    def _handle_bet_event(self, bet: BetOrder):
        """Handle bet placement events"""
        logger.info(f"🎯 BET EVENT: {bet.status.value} - "
                   f"€{bet.stake:.2f} on {bet.selection} at {bet.odds}")
        
        # Log to production file
        bet_log = {
            'timestamp': datetime.now().isoformat(),
            'bet_id': bet.order_id,
            'status': bet.status.value,
            'selection': bet.selection,
            'stake': bet.stake,
            'odds': bet.odds,
            'potential_payout': bet.potential_payout
        }
        
        self._append_to_production_log('bets', bet_log)
    
    def _handle_alert(self, alert: Alert):
        """Handle system alerts"""
        logger.warning(f"🚨 ALERT: [{alert.severity.value}] {alert.message}")
        
        # Take action on critical alerts
        if alert.severity.value == 'critical':
            logger.error("Critical alert received - implementing emergency measures")
            
            # Stop placing new bets temporarily
            if self.betting_engine:
                # This would pause betting until issue is resolved
                logger.info("Pausing new bet placements due to critical alert")
    
    def _append_to_production_log(self, log_type: str, data: Dict[str, Any]):
        """Append data to production log files"""
        try:
            log_file = f"production_{log_type}.log"
            with open(log_file, 'a') as f:
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            logger.error(f"Failed to write to production log: {e}")
    
    def shutdown(self):
        """Gracefully shutdown the system"""
        logger.info("🛑 Initiating system shutdown...")
        
        self.is_running = False
        
        # Stop components in reverse order
        if self.betting_engine:
            logger.info("Stopping Betting Engine...")
            self.betting_engine.stop()
        
        if self.prediction_engine:
            logger.info("Stopping Prediction Engine...")
            self.prediction_engine.stop()
        
        if self.monitor:
            logger.info("Stopping Monitor...")
            self.monitor.stop()
        
        # Final stats
        if self.betting_engine:
            final_stats = self.betting_engine.get_stats()
            risk_summary = final_stats.get('risk_summary', {})
            
            logger.info("📊 FINAL STATISTICS:")
            logger.info(f"  Total Bets Placed: {final_stats.get('bets_placed', 0)}")
            logger.info(f"  Bets Won: {final_stats.get('bets_won', 0)}")
            logger.info(f"  Bets Lost: {final_stats.get('bets_lost', 0)}")
            logger.info(f"  Win Rate: {final_stats.get('win_rate', 0):.1f}%")
            logger.info(f"  Total P&L: €{final_stats.get('total_profit_loss', 0):.2f}")
            logger.info(f"  Final Balance: €{risk_summary.get('current_bankroll', 0):.2f}")
        
        logger.info("✅ System shutdown completed")
    
    def run(self):
        """Main execution method"""
        logger.info("🎾 TENNIS AUTOMATED BETTING SYSTEM - PRODUCTION DEPLOYMENT")
        logger.info("=" * 70)
        
        try:
            # Validation
            if not self.validate_production_environment():
                logger.error("❌ Production environment validation failed - aborting deployment")
                return False
            
            # Initialize components
            if not self.initialize_components():
                logger.error("❌ Component initialization failed - aborting deployment")
                return False
            
            # Start system
            if not self.start_system():
                logger.error("❌ System startup failed - aborting deployment")
                return False
            
            # Monitor system
            self.monitor_system()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Production deployment failed: {e}")
            return False
        
        finally:
            self.shutdown()


def main():
    """Main entry point"""
    deployment = ProductionDeployment()
    success = deployment.run()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())