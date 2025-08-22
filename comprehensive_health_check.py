#!/usr/bin/env python3
"""
🩺 Comprehensive Tennis Betting System Health Check
Performs detailed diagnosis and health assessment of all system components
"""

import os
import sys
import json
import sqlite3
import requests
import subprocess
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class TennisSystemHealthCheck:
    """Comprehensive health checker for tennis betting system"""
    
    def __init__(self):
        self.health_report = {
            'check_time': datetime.now().isoformat(),
            'overall_status': 'unknown',
            'components': {},
            'issues_found': [],
            'fixes_applied': [],
            'recommendations': []
        }
        
        # System endpoints to check
        self.api_endpoints = [
            '/api/health-check',
            '/api/system/database-health',
            '/api/system/ml-health',
            '/api/api-economy-status',
            '/api/betting/dashboard-stats',
            '/api/betting/performance-summary'
        ]
        
        self.base_url = 'http://localhost:5001'
    
    def check_process_status(self) -> Dict[str, Any]:
        """Check if main system processes are running"""
        print("🔍 Checking process status...")
        
        try:
            # Check for main python process
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            processes = result.stdout
            
            main_process_running = 'main.py' in processes or 'python' in processes
            
            # Check specific tennis processes
            tennis_processes = []
            for line in processes.split('\n'):
                if any(term in line.lower() for term in ['tennis', 'betting', 'main.py']):
                    if 'grep' not in line and 'ps aux' not in line:
                        tennis_processes.append(line.strip())
            
            status = {
                'status': 'healthy' if main_process_running else 'unhealthy',
                'main_process_running': main_process_running,
                'tennis_processes': tennis_processes,
                'total_processes': len(tennis_processes)
            }
            
            if not main_process_running:
                self.health_report['issues_found'].append('Main tennis system process not running')
                
            print(f"✅ Process status: {status['status']}")
            return status
            
        except Exception as e:
            error_status = {
                'status': 'error',
                'error': str(e),
                'main_process_running': False
            }
            self.health_report['issues_found'].append(f'Process check failed: {e}')
            print(f"❌ Process check failed: {e}")
            return error_status
    
    def check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and integrity"""
        print("🗄️ Checking database health...")
        
        try:
            db_paths = [
                'data/tennis_predictions.db',
                'tennis_data_enhanced/enhanced_tennis_data.db',
                'prediction_logs/predictions.db'
            ]
            
            db_status = {}
            
            for db_path in db_paths:
                if os.path.exists(db_path):
                    try:
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()
                        
                        # Check table count
                        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                        table_count = cursor.fetchone()[0]
                        
                        # Check if we can write (simple test)
                        cursor.execute("SELECT 1")
                        result = cursor.fetchone()
                        
                        db_status[db_path] = {
                            'status': 'healthy',
                            'accessible': True,
                            'table_count': table_count,
                            'readable': True,
                            'writable': True
                        }
                        
                        conn.close()
                        print(f"✅ Database {db_path}: {table_count} tables")
                        
                    except Exception as e:
                        db_status[db_path] = {
                            'status': 'error',
                            'accessible': False,
                            'error': str(e)
                        }
                        self.health_report['issues_found'].append(f'Database error {db_path}: {e}')
                        print(f"❌ Database {db_path} error: {e}")
                else:
                    db_status[db_path] = {
                        'status': 'missing',
                        'accessible': False,
                        'exists': False
                    }
                    self.health_report['issues_found'].append(f'Database missing: {db_path}')
                    print(f"⚠️ Database missing: {db_path}")
            
            overall_db_health = 'healthy' if any(db['status'] == 'healthy' for db in db_status.values()) else 'unhealthy'
            
            return {
                'status': overall_db_health,
                'databases': db_status,
                'total_databases': len([db for db in db_status.values() if db['status'] == 'healthy'])
            }
            
        except Exception as e:
            error_status = {
                'status': 'error',
                'error': str(e)
            }
            self.health_report['issues_found'].append(f'Database health check failed: {e}')
            print(f"❌ Database health check failed: {e}")
            return error_status
    
    def check_redis_connectivity(self) -> Dict[str, Any]:
        """Check Redis connectivity for caching and rate limiting"""
        print("📦 Checking Redis connectivity...")
        
        try:
            import redis
            
            # Try connecting to Redis
            r = redis.Redis(host='localhost', port=6379, socket_connect_timeout=2, socket_timeout=2)
            r.ping()
            
            info = r.info()
            status = {
                'status': 'healthy',
                'connected': True,
                'redis_version': info.get('redis_version', 'unknown'),
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory_human', 'unknown'),
                'uptime': info.get('uptime_in_seconds', 0)
            }
            
            print(f"✅ Redis connected: version {status['redis_version']}")
            return status
            
        except ImportError:
            status = {
                'status': 'redis_not_installed',
                'connected': False,
                'error': 'Redis package not installed'
            }
            self.health_report['issues_found'].append('Redis package not installed')
            print("⚠️ Redis package not installed")
            return status
            
        except Exception as e:
            status = {
                'status': 'connection_failed',
                'connected': False,
                'error': str(e)
            }
            self.health_report['issues_found'].append(f'Redis connection failed: {e}')
            print(f"❌ Redis connection failed: {e}")
            return status
    
    def check_api_endpoints(self) -> Dict[str, Any]:
        """Check API endpoint health and responsiveness"""
        print("🌐 Checking API endpoints...")
        
        endpoint_status = {}
        
        for endpoint in self.api_endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                response = requests.get(url, timeout=10)
                
                endpoint_status[endpoint] = {
                    'status': 'healthy' if response.status_code == 200 else 'error',
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds(),
                    'accessible': True
                }
                
                if response.status_code == 429:
                    endpoint_status[endpoint]['error'] = 'Rate limited'
                    self.health_report['issues_found'].append(f'Rate limiting on {endpoint}')
                elif response.status_code == 404:
                    endpoint_status[endpoint]['error'] = 'Endpoint not found'
                    self.health_report['issues_found'].append(f'Missing endpoint: {endpoint}')
                elif response.status_code != 200:
                    endpoint_status[endpoint]['error'] = f'HTTP {response.status_code}'
                    self.health_report['issues_found'].append(f'HTTP error {response.status_code} on {endpoint}')
                
                status_icon = "✅" if response.status_code == 200 else "❌" if response.status_code == 429 else "⚠️"
                print(f"{status_icon} {endpoint}: {response.status_code}")
                
            except requests.exceptions.ConnectionError:
                endpoint_status[endpoint] = {
                    'status': 'connection_refused',
                    'accessible': False,
                    'error': 'Connection refused'
                }
                self.health_report['issues_found'].append(f'Server not running or connection refused for {endpoint}')
                print(f"❌ {endpoint}: Connection refused")
                
            except Exception as e:
                endpoint_status[endpoint] = {
                    'status': 'error',
                    'accessible': False,
                    'error': str(e)
                }
                self.health_report['issues_found'].append(f'Error accessing {endpoint}: {e}')
                print(f"❌ {endpoint}: {e}")
        
        healthy_endpoints = len([ep for ep in endpoint_status.values() if ep['status'] == 'healthy'])
        total_endpoints = len(endpoint_status)
        
        return {
            'status': 'healthy' if healthy_endpoints > total_endpoints * 0.5 else 'unhealthy',
            'endpoints': endpoint_status,
            'healthy_count': healthy_endpoints,
            'total_count': total_endpoints
        }
    
    def check_ml_models(self) -> Dict[str, Any]:
        """Check ML model availability and health"""
        print("🤖 Checking ML models...")
        
        model_dirs = [
            'tennis_models',
            'tennis_models_enhanced',
            'tennis_models_corrected'
        ]
        
        model_status = {}
        
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                try:
                    model_files = os.listdir(model_dir)
                    pkl_files = [f for f in model_files if f.endswith('.pkl')]
                    h5_files = [f for f in model_files if f.endswith('.h5')]
                    
                    has_metadata = 'metadata.json' in model_files
                    
                    model_status[model_dir] = {
                        'status': 'healthy' if pkl_files or h5_files else 'no_models',
                        'exists': True,
                        'model_count': len(pkl_files) + len(h5_files),
                        'pkl_models': pkl_files,
                        'h5_models': h5_files,
                        'has_metadata': has_metadata
                    }
                    
                    print(f"✅ {model_dir}: {len(pkl_files + h5_files)} models")
                    
                except Exception as e:
                    model_status[model_dir] = {
                        'status': 'error',
                        'exists': True,
                        'error': str(e)
                    }
                    print(f"❌ {model_dir} error: {e}")
            else:
                model_status[model_dir] = {
                    'status': 'missing',
                    'exists': False
                }
                print(f"⚠️ {model_dir}: Directory missing")
        
        healthy_models = sum(1 for ms in model_status.values() if ms['status'] == 'healthy')
        
        return {
            'status': 'healthy' if healthy_models > 0 else 'unhealthy',
            'model_directories': model_status,
            'total_model_count': sum(ms.get('model_count', 0) for ms in model_status.values() if ms['status'] == 'healthy')
        }
    
    def check_log_files(self) -> Dict[str, Any]:
        """Check log files for recent errors"""
        print("📋 Checking log files...")
        
        log_dirs = ['logs', 'logs/archived_logs']
        recent_errors = []
        log_status = {}
        
        for log_dir in log_dirs:
            if os.path.exists(log_dir):
                try:
                    log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
                    
                    for log_file in log_files[-5:]:  # Check last 5 log files
                        log_path = os.path.join(log_dir, log_file)
                        
                        try:
                            # Check last 50 lines for errors
                            with open(log_path, 'r') as f:
                                lines = f.readlines()
                                recent_lines = lines[-50:] if len(lines) > 50 else lines
                                
                                for line in recent_lines:
                                    if any(term in line.lower() for term in ['error', 'failed', 'exception', 'critical']):
                                        if 'rate limit' not in line.lower():  # Ignore rate limit errors for now
                                            recent_errors.append({
                                                'file': log_file,
                                                'line': line.strip()
                                            })
                                
                        except Exception as e:
                            print(f"⚠️ Could not read {log_file}: {e}")
                    
                    log_status[log_dir] = {
                        'status': 'healthy',
                        'file_count': len(log_files),
                        'recent_errors': len([e for e in recent_errors if log_dir in e['file']])
                    }
                    
                    print(f"✅ {log_dir}: {len(log_files)} log files")
                    
                except Exception as e:
                    log_status[log_dir] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    print(f"❌ {log_dir} error: {e}")
            else:
                log_status[log_dir] = {
                    'status': 'missing',
                    'exists': False
                }
        
        return {
            'status': 'healthy' if len(recent_errors) < 10 else 'unhealthy',
            'log_directories': log_status,
            'recent_errors': recent_errors[:10],  # Return top 10 errors
            'total_errors': len(recent_errors)
        }
    
    def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run all health checks and compile report"""
        print("🩺 COMPREHENSIVE TENNIS SYSTEM HEALTH CHECK")
        print("=" * 60)
        print(f"📅 Check Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Run all checks
        checks = {
            'processes': self.check_process_status(),
            'database': self.check_database_health(),
            'redis': self.check_redis_connectivity(),
            'api_endpoints': self.check_api_endpoints(),
            'ml_models': self.check_ml_models(),
            'log_files': self.check_log_files()
        }
        
        # Store in health report
        self.health_report['components'] = checks
        
        # Determine overall status
        component_statuses = [check['status'] for check in checks.values()]
        healthy_components = len([s for s in component_statuses if s == 'healthy'])
        total_components = len(component_statuses)
        
        if healthy_components == total_components:
            self.health_report['overall_status'] = 'healthy'
        elif healthy_components >= total_components * 0.7:
            self.health_report['overall_status'] = 'warning'
        else:
            self.health_report['overall_status'] = 'unhealthy'
        
        # Generate recommendations
        self._generate_recommendations()
        
        return self.health_report
    
    def _generate_recommendations(self):
        """Generate recommendations based on health check results"""
        
        # Check for specific issues and add recommendations
        issues = self.health_report['issues_found']
        
        if any('Rate limiting' in issue for issue in issues):
            self.health_report['recommendations'].append(
                'Fix rate limiting configuration - increase limits for health endpoints'
            )
        
        if any('Missing endpoint' in issue for issue in issues):
            self.health_report['recommendations'].append(
                'Implement missing API endpoints (e.g., /api/betting/alerts)'
            )
        
        if any('Redis' in issue for issue in issues):
            self.health_report['recommendations'].append(
                'Install and configure Redis for improved caching and rate limiting'
            )
        
        if any('Database' in issue for issue in issues):
            self.health_report['recommendations'].append(
                'Check database permissions and integrity'
            )
        
        if self.health_report['components']['processes']['status'] != 'healthy':
            self.health_report['recommendations'].append(
                'Restart main tennis system process (python main.py)'
            )
    
    def print_health_summary(self):
        """Print formatted health summary"""
        
        print("\n🏥 HEALTH CHECK SUMMARY")
        print("=" * 60)
        
        # Overall status
        status_icon = "✅" if self.health_report['overall_status'] == 'healthy' else "⚠️" if self.health_report['overall_status'] == 'warning' else "❌"
        print(f"Overall Status: {status_icon} {self.health_report['overall_status'].upper()}")
        print()
        
        # Component status
        print("Component Status:")
        for component, data in self.health_report['components'].items():
            status_icon = "✅" if data['status'] == 'healthy' else "❌" if data['status'] in ['error', 'unhealthy'] else "⚠️"
            print(f"  {status_icon} {component.title()}: {data['status']}")
        print()
        
        # Issues found
        if self.health_report['issues_found']:
            print(f"Issues Found ({len(self.health_report['issues_found'])}):")
            for i, issue in enumerate(self.health_report['issues_found'][:10], 1):
                print(f"  {i}. {issue}")
            if len(self.health_report['issues_found']) > 10:
                print(f"  ... and {len(self.health_report['issues_found']) - 10} more issues")
            print()
        
        # Recommendations
        if self.health_report['recommendations']:
            print("Recommendations:")
            for i, rec in enumerate(self.health_report['recommendations'], 1):
                print(f"  {i}. {rec}")
            print()
        
        print("=" * 60)

def main():
    """Run comprehensive health check"""
    
    try:
        checker = TennisSystemHealthCheck()
        health_report = checker.run_comprehensive_check()
        
        # Print summary
        checker.print_health_summary()
        
        # Save detailed report
        report_filename = f"system_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(health_report, f, indent=2)
        
        print(f"💾 Detailed health report saved to: {report_filename}")
        
        # Return appropriate exit code
        if health_report['overall_status'] == 'healthy':
            print("\n🎉 System is healthy!")
            return 0
        elif health_report['overall_status'] == 'warning':
            print("\n⚠️ System has some issues but is functional")
            return 1
        else:
            print("\n❌ System is unhealthy and needs attention")
            return 2
            
    except Exception as e:
        print(f"\n💥 Health check failed: {e}")
        traceback.print_exc()
        return 3

if __name__ == "__main__":
    exit(main())