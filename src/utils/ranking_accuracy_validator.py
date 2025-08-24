#!/usr/bin/env python3
"""
Tennis Ranking Accuracy Validator
Validates ranking data accuracy against known current rankings and implements real-time monitoring
"""

import os
import sys

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, using system environment variables
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import time

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.dynamic_rankings_api import dynamic_rankings
from api.enhanced_ranking_integration import EnhancedRankingClient

logger = logging.getLogger(__name__)

class RankingAccuracyValidator:
    """Validates and monitors tennis ranking data accuracy"""
    
    def __init__(self):
        self.validation_log_file = "/home/apps/Tennis_one_set/logs/ranking_validation.log"
        self.accuracy_report_file = "/home/apps/Tennis_one_set/data/ranking_accuracy_report.json"
        
        # Known current rankings for validation (as of TODO.md)
        self.known_current_rankings = {
            "linda noskova": {"actual_rank": 23, "tour": "wta"},
            "l. noskova": {"actual_rank": 23, "tour": "wta"},
            "noskova": {"actual_rank": 23, "tour": "wta"},
            "e. alexandrova": {"actual_rank": 14, "tour": "wta"},
            "ekaterina alexandrova": {"actual_rank": 14, "tour": "wta"},
            "alexandrova": {"actual_rank": 14, "tour": "wta"},
            "a. tomljanovic": {"actual_rank": 84, "tour": "wta"},  # Corrected from 250 in notification
            "ajla tomljanovic": {"actual_rank": 84, "tour": "wta"},
            "tomljanovic": {"actual_rank": 84, "tour": "wta"}
        }
        
        # Tolerance for ranking discrepancies
        self.warning_threshold = 5    # positions
        self.critical_threshold = 20  # positions
        
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for ranking validation"""
        log_dir = os.path.dirname(self.validation_log_file)
        os.makedirs(log_dir, exist_ok=True)
        
        # Create validator-specific logger
        self.validator_logger = logging.getLogger('ranking_validator')
        self.validator_logger.setLevel(logging.INFO)
        
        # Add file handler if not already present
        if not any(isinstance(h, logging.FileHandler) for h in self.validator_logger.handlers):
            file_handler = logging.FileHandler(self.validation_log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            self.validator_logger.addHandler(file_handler)
    
    def validate_dynamic_rankings_api(self) -> Dict[str, Any]:
        """Test the dynamic rankings API accuracy"""
        self.validator_logger.info("🔍 Starting dynamic rankings API validation")
        
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "api_source": "dynamic_rankings_api",
            "players_tested": {},
            "summary": {
                "total_tested": 0,
                "accurate": 0,
                "warnings": 0,
                "critical_errors": 0
            }
        }
        
        for player_name, expected_data in self.known_current_rankings.items():
            try:
                # Get ranking from dynamic API
                api_result = dynamic_rankings.get_player_ranking(player_name, expected_data["tour"])
                api_rank = api_result.get("rank", 999)
                expected_rank = expected_data["actual_rank"]
                
                # Calculate discrepancy
                discrepancy = abs(api_rank - expected_rank)
                
                result = {
                    "expected_rank": expected_rank,
                    "api_rank": api_rank,
                    "discrepancy": discrepancy,
                    "tour": expected_data["tour"],
                    "status": "accurate"
                }
                
                # Determine status based on discrepancy
                if discrepancy == 0:
                    validation_results["summary"]["accurate"] += 1
                elif discrepancy <= self.warning_threshold:
                    result["status"] = "minor_warning"
                    validation_results["summary"]["warnings"] += 1
                    self.validator_logger.warning(
                        f"⚠️ Minor ranking discrepancy for {player_name}: "
                        f"Expected #{expected_rank}, got #{api_rank} (diff: {discrepancy})"
                    )
                elif discrepancy <= self.critical_threshold:
                    result["status"] = "warning"
                    validation_results["summary"]["warnings"] += 1
                    self.validator_logger.warning(
                        f"⚠️ Ranking discrepancy for {player_name}: "
                        f"Expected #{expected_rank}, got #{api_rank} (diff: {discrepancy})"
                    )
                else:
                    result["status"] = "critical_error"
                    validation_results["summary"]["critical_errors"] += 1
                    self.validator_logger.error(
                        f"❌ CRITICAL ranking error for {player_name}: "
                        f"Expected #{expected_rank}, got #{api_rank} (diff: {discrepancy})"
                    )
                
                validation_results["players_tested"][player_name] = result
                validation_results["summary"]["total_tested"] += 1
                
            except Exception as e:
                self.validator_logger.error(f"❌ Error validating {player_name}: {e}")
                validation_results["players_tested"][player_name] = {
                    "error": str(e),
                    "status": "error"
                }
        
        # Log summary
        summary = validation_results["summary"]
        self.validator_logger.info(
            f"📊 Validation complete: {summary['accurate']} accurate, "
            f"{summary['warnings']} warnings, {summary['critical_errors']} critical errors"
        )
        
        return validation_results
    
    def validate_enhanced_ranking_integration(self) -> Dict[str, Any]:
        """Test the enhanced ranking integration accuracy"""
        self.validator_logger.info("🔍 Starting enhanced ranking integration validation")
        
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "api_source": "enhanced_ranking_integration",
            "players_tested": {},
            "summary": {
                "total_tested": 0,
                "accurate": 0,
                "warnings": 0,
                "critical_errors": 0
            }
        }
        
        try:
            enhanced_client = EnhancedRankingClient()
            
            for player_name, expected_data in self.known_current_rankings.items():
                try:
                    # Get WTA standings since our test players are WTA
                    if expected_data["tour"] == "wta":
                        standings = enhanced_client.get_standings_corrected("WTA")
                        
                        # Search for player in standings with flexible matching
                        api_rank = None
                        for standing in standings:
                            player_name_in_standings = standing.get('player', '').lower()
                            
                            # Try exact match first
                            if player_name_in_standings == player_name:
                                api_rank = int(standing.get('place', 999))
                                break
                            
                            # Try partial matching - if all parts of search name are in the standing name
                            search_parts = player_name.split()
                            if all(part in player_name_in_standings for part in search_parts):
                                api_rank = int(standing.get('place', 999))
                                break
                                
                            # Try last name matching for abbreviated names like "l. noskova"
                            if '.' in player_name:
                                last_name = player_name.split()[-1]  # Get last part after dot
                                if last_name in player_name_in_standings:
                                    api_rank = int(standing.get('place', 999))
                                    break
                        
                        if api_rank is None:
                            api_rank = 999  # Not found
                        
                        expected_rank = expected_data["actual_rank"]
                        discrepancy = abs(api_rank - expected_rank)
                        
                        result = {
                            "expected_rank": expected_rank,
                            "api_rank": api_rank,
                            "discrepancy": discrepancy,
                            "tour": expected_data["tour"],
                            "status": "accurate"
                        }
                        
                        # Determine status
                        if discrepancy == 0:
                            validation_results["summary"]["accurate"] += 1
                        elif discrepancy <= self.warning_threshold:
                            result["status"] = "minor_warning"
                            validation_results["summary"]["warnings"] += 1
                        elif discrepancy <= self.critical_threshold:
                            result["status"] = "warning"
                            validation_results["summary"]["warnings"] += 1
                        else:
                            result["status"] = "critical_error"
                            validation_results["summary"]["critical_errors"] += 1
                        
                        validation_results["players_tested"][player_name] = result
                        validation_results["summary"]["total_tested"] += 1
                        
                except Exception as e:
                    self.validator_logger.error(f"❌ Error validating {player_name} in enhanced API: {e}")
                    validation_results["players_tested"][player_name] = {
                        "error": str(e),
                        "status": "error"
                    }
        
        except Exception as e:
            self.validator_logger.error(f"❌ Error initializing enhanced ranking client: {e}")
        
        return validation_results
    
    def validate_hardcoded_rankings(self, hardcoded_rankings: Dict[str, int]) -> Dict[str, Any]:
        """Validate hardcoded rankings against known current values"""
        self.validator_logger.info("🔍 Validating hardcoded rankings")
        
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "api_source": "hardcoded_rankings",
            "players_tested": {},
            "summary": {
                "total_tested": 0,
                "accurate": 0,
                "warnings": 0,
                "critical_errors": 0
            }
        }
        
        for player_name, expected_data in self.known_current_rankings.items():
            hardcoded_rank = hardcoded_rankings.get(player_name, 999)
            expected_rank = expected_data["actual_rank"]
            discrepancy = abs(hardcoded_rank - expected_rank)
            
            result = {
                "expected_rank": expected_rank,
                "hardcoded_rank": hardcoded_rank,
                "discrepancy": discrepancy,
                "tour": expected_data["tour"],
                "status": "accurate"
            }
            
            # Determine status
            if discrepancy == 0:
                validation_results["summary"]["accurate"] += 1
            elif discrepancy <= self.warning_threshold:
                result["status"] = "minor_warning"
                validation_results["summary"]["warnings"] += 1
            elif discrepancy <= self.critical_threshold:
                result["status"] = "warning"
                validation_results["summary"]["warnings"] += 1
                self.validator_logger.warning(
                    f"⚠️ Hardcoded ranking discrepancy for {player_name}: "
                    f"Expected #{expected_rank}, hardcoded #{hardcoded_rank} (diff: {discrepancy})"
                )
            else:
                result["status"] = "critical_error"
                validation_results["summary"]["critical_errors"] += 1
                self.validator_logger.error(
                    f"❌ CRITICAL hardcoded ranking error for {player_name}: "
                    f"Expected #{expected_rank}, hardcoded #{hardcoded_rank} (diff: {discrepancy})"
                )
            
            validation_results["players_tested"][player_name] = result
            validation_results["summary"]["total_tested"] += 1
        
        return validation_results
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation across all ranking sources"""
        self.validator_logger.info("🚀 Starting comprehensive ranking validation")
        
        comprehensive_results = {
            "validation_timestamp": datetime.now().isoformat(),
            "validation_sources": {},
            "critical_issues": [],
            "recommendations": []
        }
        
        # Test dynamic rankings API
        try:
            dynamic_results = self.validate_dynamic_rankings_api()
            comprehensive_results["validation_sources"]["dynamic_api"] = dynamic_results
            
            # Check for critical issues
            if dynamic_results["summary"]["critical_errors"] > 0:
                comprehensive_results["critical_issues"].append({
                    "source": "dynamic_api",
                    "issue": f"{dynamic_results['summary']['critical_errors']} critical ranking errors detected",
                    "impact": "HIGH - May cause incorrect betting decisions"
                })
        except Exception as e:
            self.validator_logger.error(f"❌ Dynamic API validation failed: {e}")
            comprehensive_results["critical_issues"].append({
                "source": "dynamic_api",
                "issue": f"Validation failed: {e}",
                "impact": "HIGH - API unavailable"
            })
        
        # Test enhanced ranking integration
        try:
            enhanced_results = self.validate_enhanced_ranking_integration()
            comprehensive_results["validation_sources"]["enhanced_integration"] = enhanced_results
            
            if enhanced_results["summary"]["critical_errors"] > 0:
                comprehensive_results["critical_issues"].append({
                    "source": "enhanced_integration",
                    "issue": f"{enhanced_results['summary']['critical_errors']} critical ranking errors detected",
                    "impact": "HIGH - Backup ranking source unreliable"
                })
        except Exception as e:
            self.validator_logger.error(f"❌ Enhanced integration validation failed: {e}")
            comprehensive_results["critical_issues"].append({
                "source": "enhanced_integration",
                "issue": f"Validation failed: {e}",
                "impact": "MEDIUM - Backup API unavailable"
            })
        
        # Generate recommendations
        comprehensive_results["recommendations"] = self._generate_recommendations(comprehensive_results)
        
        # Save results
        self._save_validation_report(comprehensive_results)
        
        return comprehensive_results
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Check if any critical issues exist
        if results["critical_issues"]:
            recommendations.append({
                "priority": "HIGH",
                "action": "Immediately fix critical ranking data errors",
                "description": "Critical ranking discrepancies detected that could lead to financial losses"
            })
        
        # Check accuracy of each source
        for source_name, source_data in results["validation_sources"].items():
            if "summary" in source_data:
                summary = source_data["summary"]
                accuracy_rate = summary["accurate"] / summary["total_tested"] if summary["total_tested"] > 0 else 0
                
                if accuracy_rate < 0.8:  # Less than 80% accuracy
                    recommendations.append({
                        "priority": "HIGH",
                        "action": f"Improve accuracy of {source_name}",
                        "description": f"Only {accuracy_rate:.1%} accuracy rate for {source_name}"
                    })
                elif accuracy_rate < 0.95:  # Less than 95% accuracy
                    recommendations.append({
                        "priority": "MEDIUM",
                        "action": f"Fine-tune {source_name}",
                        "description": f"{accuracy_rate:.1%} accuracy rate - room for improvement"
                    })
        
        # Always recommend implementing real-time monitoring
        recommendations.append({
            "priority": "MEDIUM",
            "action": "Implement continuous ranking monitoring",
            "description": "Set up automated alerts for ranking discrepancies"
        })
        
        return recommendations
    
    def _save_validation_report(self, results: Dict[str, Any]):
        """Save validation report to file"""
        try:
            report_dir = os.path.dirname(self.accuracy_report_file)
            os.makedirs(report_dir, exist_ok=True)
            
            with open(self.accuracy_report_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.validator_logger.info(f"📄 Validation report saved to {self.accuracy_report_file}")
            
        except Exception as e:
            self.validator_logger.error(f"❌ Failed to save validation report: {e}")
    
    def create_real_time_monitor(self, check_interval_minutes: int = 60) -> None:
        """Create a real-time ranking accuracy monitor"""
        self.validator_logger.info(f"🔄 Starting real-time ranking monitor (check every {check_interval_minutes} minutes)")
        
        while True:
            try:
                # Run validation
                results = self.run_comprehensive_validation()
                
                # Check for critical issues and alert
                if results["critical_issues"]:
                    self._send_critical_alert(results["critical_issues"])
                
                # Wait for next check
                time.sleep(check_interval_minutes * 60)
                
            except KeyboardInterrupt:
                self.validator_logger.info("⏹️ Real-time monitor stopped by user")
                break
            except Exception as e:
                self.validator_logger.error(f"❌ Error in real-time monitor: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _send_critical_alert(self, critical_issues: List[Dict[str, Any]]):
        """Send alert for critical ranking issues"""
        alert_message = "🚨 CRITICAL RANKING DATA ISSUES DETECTED:\n\n"
        
        for issue in critical_issues:
            alert_message += f"• Source: {issue['source']}\n"
            alert_message += f"  Issue: {issue['issue']}\n"
            alert_message += f"  Impact: {issue['impact']}\n\n"
        
        alert_message += "⚠️ IMMEDIATE ACTION REQUIRED - Potential financial risk from incorrect rankings!"
        
        self.validator_logger.critical(alert_message)
        
        # TODO: Integrate with telegram notification system for immediate alerts
        try:
            # Import telegram system if available
            sys.path.append('/home/apps/Tennis_one_set/src/utils')
            from telegram_notification_system import TelegramNotificationSystem
            
            telegram_system = TelegramNotificationSystem()
            if telegram_system.config.enabled:
                # Send emergency alert
                asyncio.run(telegram_system._send_emergency_alert(alert_message))
        except Exception as e:
            self.validator_logger.warning(f"Could not send telegram alert: {e}")

def main():
    """Main function for testing the ranking validator"""
    print("🔍 Tennis Ranking Accuracy Validator")
    print("=" * 50)
    
    validator = RankingAccuracyValidator()
    
    # Run comprehensive validation
    print("\n🚀 Running comprehensive ranking validation...")
    results = validator.run_comprehensive_validation()
    
    # Display results
    print(f"\n📊 Validation Results:")
    print(f"   Validation completed at: {results['validation_timestamp']}")
    
    for source_name, source_data in results["validation_sources"].items():
        if "summary" in source_data:
            summary = source_data["summary"]
            accuracy_rate = summary["accurate"] / summary["total_tested"] if summary["total_tested"] > 0 else 0
            print(f"\n   {source_name.upper()}:")
            print(f"     Tested: {summary['total_tested']} players")
            print(f"     Accurate: {summary['accurate']} ({accuracy_rate:.1%})")
            print(f"     Warnings: {summary['warnings']}")
            print(f"     Critical Errors: {summary['critical_errors']}")
    
    # Display critical issues
    if results["critical_issues"]:
        print(f"\n🚨 CRITICAL ISSUES ({len(results['critical_issues'])}):")
        for issue in results["critical_issues"]:
            print(f"   • {issue['source']}: {issue['issue']}")
            print(f"     Impact: {issue['impact']}")
    else:
        print(f"\n✅ No critical issues detected")
    
    # Display recommendations
    if results["recommendations"]:
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in results["recommendations"]:
            print(f"   [{rec['priority']}] {rec['action']}")
            print(f"      {rec['description']}")
    
    print(f"\n📄 Full report saved to: {validator.accuracy_report_file}")
    print("✅ Validation complete!")

if __name__ == "__main__":
    main()