#!/usr/bin/env python3
"""
ML Training Monitor and Performance Tracker
Real-time monitoring of ML training progress and model performance
"""

import pandas as pd
import numpy as np
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from collections import defaultdict
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLTrainingMonitor:
    """
    Comprehensive monitoring system for ML training pipeline
    Tracks data quality, training progress, and model performance
    """
    
    def __init__(self, models_dir: str = "tennis_models", 
                 data_dir: str = "tennis_data_enhanced"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        
        # Monitoring database
        self.monitor_db = self.data_dir / "training_monitor.db"
        
        # Performance thresholds
        self.performance_thresholds = {
            'accuracy_min': 0.55,
            'f1_min': 0.50,
            'auc_min': 0.60,
            'precision_min': 0.45,
            'recall_min': 0.45,
            'underdog_prediction_min': 0.35  # Minimum underdog prediction accuracy
        }
        
        # Training progress tracking
        self.training_stages = [
            'data_collection', 'feature_engineering', 'preprocessing',
            'model_training', 'validation', 'ensemble_optimization'
        ]
        
        self._initialize_monitoring_db()
    
    def _initialize_monitoring_db(self):
        """Initialize monitoring database"""
        self.data_dir.mkdir(exist_ok=True)
        
        with sqlite3.connect(self.monitor_db, check_same_thread=False, timeout=30.0) as conn:
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=memory")
            conn.execute("PRAGMA mmap_size=268435456")
            # Training sessions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    end_time DATETIME,
                    status TEXT, -- 'running', 'completed', 'failed', 'interrupted'
                    
                    -- Data characteristics
                    total_samples INTEGER,
                    training_samples INTEGER,
                    validation_samples INTEGER,
                    features_count INTEGER,
                    target_balance REAL,
                    data_quality_score REAL,
                    
                    -- Training configuration
                    models_trained TEXT, -- JSON array of model names
                    hyperparameter_tuning BOOLEAN,
                    feature_selection BOOLEAN,
                    ensemble_optimization BOOLEAN,
                    
                    -- Results
                    best_model TEXT,
                    best_accuracy REAL,
                    best_f1_score REAL,
                    best_auc REAL,
                    training_duration_minutes INTEGER,
                    
                    -- Notes and errors
                    notes TEXT,
                    errors TEXT -- JSON array of errors
                )
            """)
            
            # Model performance tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    model_name TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    
                    -- Performance metrics
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    auc_roc REAL,
                    
                    -- Tennis-specific metrics
                    underdog_prediction_accuracy REAL,
                    favorite_prediction_accuracy REAL,
                    calibration_error REAL,
                    
                    -- Cross-validation results
                    cv_mean_accuracy REAL,
                    cv_std_accuracy REAL,
                    cv_mean_f1 REAL,
                    cv_std_f1 REAL,
                    
                    -- Training details
                    training_time_seconds REAL,
                    hyperparameters TEXT, -- JSON
                    feature_importance TEXT, -- JSON
                    validation_curve_data TEXT, -- JSON
                    
                    FOREIGN KEY (session_id) REFERENCES training_sessions (session_id)
                )
            """)
            
            # Training progress log
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    stage TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    status TEXT, -- 'started', 'completed', 'failed'
                    progress_percentage INTEGER,
                    message TEXT,
                    details TEXT, -- JSON
                    
                    FOREIGN KEY (session_id) REFERENCES training_sessions (session_id)
                )
            """)
            
            # Data quality monitoring
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_quality_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    data_source TEXT,
                    
                    -- Quality metrics
                    total_samples INTEGER,
                    complete_samples INTEGER,
                    missing_data_percentage REAL,
                    duplicate_percentage REAL,
                    outlier_percentage REAL,
                    
                    -- Feature quality
                    features_with_zero_variance INTEGER,
                    highly_correlated_features INTEGER,
                    feature_importance_entropy REAL,
                    
                    -- Target variable quality
                    target_balance REAL,
                    target_missing_count INTEGER,
                    
                    quality_score INTEGER
                )
            """)
            
            # Performance alerts
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    alert_type TEXT, -- 'performance_drop', 'data_quality', 'training_failure'
                    severity TEXT, -- 'low', 'medium', 'high', 'critical'
                    message TEXT,
                    details TEXT, -- JSON
                    resolved BOOLEAN DEFAULT 0,
                    resolution_notes TEXT
                )
            """)
            
            # Create indexes (individual execute calls for thread safety)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON training_sessions(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model_performance ON model_performance(session_id, model_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_training_progress ON training_progress(session_id, stage)")
            
            logger.info(f"Initialized training monitor database: {self.monitor_db}")
    
    def start_training_session(self, config: Dict) -> str:
        """
        Start a new training session monitoring
        
        Args:
            config: Training configuration dictionary
            
        Returns:
            str: Session ID for tracking
        """
        session_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with sqlite3.connect(self.monitor_db, check_same_thread=False, timeout=30.0) as conn:
            conn.execute("""
                INSERT INTO training_sessions (
                    session_id, status, models_trained, hyperparameter_tuning,
                    feature_selection, ensemble_optimization, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, 'running', 
                json.dumps(config.get('models', [])),
                config.get('hyperparameter_tuning', False),
                config.get('feature_selection', False),
                config.get('ensemble_optimization', False),
                json.dumps(config)
            ))
        
        logger.info(f"Started training session: {session_id}")
        return session_id
    
    def log_training_progress(self, session_id: str, stage: str, 
                            status: str, progress: int = 0, 
                            message: str = "", details: Dict = None):
        """Log training progress for a specific stage"""
        with sqlite3.connect(self.monitor_db, check_same_thread=False, timeout=30.0) as conn:
            conn.execute("""
                INSERT INTO training_progress (
                    session_id, stage, status, progress_percentage, message, details
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id, stage, status, progress, message,
                json.dumps(details) if details else None
            ))
        
        status_emoji = {'started': '🔄', 'completed': '✅', 'failed': '❌'}.get(status, '📝')
        logger.info(f"[{session_id}] {status_emoji} {stage}: {message} ({progress}%)")
    
    def log_model_performance(self, session_id: str, model_name: str, 
                            performance_metrics: Dict, training_details: Dict = None):
        """Log detailed model performance metrics"""
        with sqlite3.connect(self.monitor_db, check_same_thread=False, timeout=30.0) as conn:
            conn.execute("""
                INSERT INTO model_performance (
                    session_id, model_name, accuracy, precision, recall, f1_score, auc_roc,
                    underdog_prediction_accuracy, favorite_prediction_accuracy, calibration_error,
                    cv_mean_accuracy, cv_std_accuracy, cv_mean_f1, cv_std_f1,
                    training_time_seconds, hyperparameters, feature_importance, validation_curve_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, model_name,
                performance_metrics.get('accuracy'),
                performance_metrics.get('precision'),
                performance_metrics.get('recall'),
                performance_metrics.get('f1_score'),
                performance_metrics.get('auc_roc'),
                performance_metrics.get('underdog_prediction_accuracy'),
                performance_metrics.get('favorite_prediction_accuracy'),
                performance_metrics.get('calibration_error'),
                performance_metrics.get('cv_mean_accuracy'),
                performance_metrics.get('cv_std_accuracy'),
                performance_metrics.get('cv_mean_f1'),
                performance_metrics.get('cv_std_f1'),
                training_details.get('training_time_seconds') if training_details else None,
                json.dumps(training_details.get('hyperparameters', {})) if training_details else None,
                json.dumps(training_details.get('feature_importance', {})) if training_details else None,
                json.dumps(training_details.get('validation_curve_data', {})) if training_details else None
            ))
        
        # Check for performance alerts
        self._check_performance_alerts(session_id, model_name, performance_metrics)
    
    def _check_performance_alerts(self, session_id: str, model_name: str, metrics: Dict):
        """Check if performance metrics trigger any alerts"""
        alerts = []
        
        # Check critical performance thresholds
        for metric, threshold in self.performance_thresholds.items():
            metric_value = metrics.get(metric.replace('_min', ''), 0)
            if metric_value < threshold:
                alerts.append({
                    'type': 'performance_drop',
                    'severity': 'high' if metric_value < threshold * 0.8 else 'medium',
                    'message': f"{model_name} {metric.replace('_min', '')} below threshold: {metric_value:.3f} < {threshold}",
                    'details': {'session_id': session_id, 'model': model_name, 'metric': metric, 'value': metric_value, 'threshold': threshold}
                })
        
        # Log alerts
        for alert in alerts:
            self._log_alert(alert)
    
    def _log_alert(self, alert: Dict):
        """Log a performance alert"""
        with sqlite3.connect(self.monitor_db, check_same_thread=False, timeout=30.0) as conn:
            conn.execute("""
                INSERT INTO performance_alerts (
                    alert_type, severity, message, details
                ) VALUES (?, ?, ?, ?)
            """, (
                alert['type'], alert['severity'], 
                alert['message'], json.dumps(alert['details'])
            ))
        
        severity_emoji = {'low': '💡', 'medium': '⚠️', 'high': '🚨', 'critical': '🔥'}
        logger.warning(f"{severity_emoji.get(alert['severity'], '📢')} ALERT: {alert['message']}")
    
    def log_data_quality(self, data_quality_metrics: Dict, data_source: str = "training"):
        """Log data quality metrics"""
        with sqlite3.connect(self.monitor_db, check_same_thread=False, timeout=30.0) as conn:
            conn.execute("""
                INSERT INTO data_quality_log (
                    data_source, total_samples, complete_samples, missing_data_percentage,
                    duplicate_percentage, outlier_percentage, features_with_zero_variance,
                    highly_correlated_features, feature_importance_entropy,
                    target_balance, target_missing_count, quality_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data_source,
                data_quality_metrics.get('total_samples', 0),
                data_quality_metrics.get('complete_samples', 0),
                data_quality_metrics.get('missing_data_percentage', 0),
                data_quality_metrics.get('duplicate_percentage', 0),
                data_quality_metrics.get('outlier_percentage', 0),
                data_quality_metrics.get('features_with_zero_variance', 0),
                data_quality_metrics.get('highly_correlated_features', 0),
                data_quality_metrics.get('feature_importance_entropy', 0),
                data_quality_metrics.get('target_balance', 0.5),
                data_quality_metrics.get('target_missing_count', 0),
                data_quality_metrics.get('quality_score', 50)
            ))
        
        quality_score = data_quality_metrics.get('quality_score', 50)
        if quality_score < 60:
            self._log_alert({
                'type': 'data_quality',
                'severity': 'high' if quality_score < 40 else 'medium',
                'message': f"Low data quality detected: score {quality_score}/100",
                'details': data_quality_metrics
            })
    
    def complete_training_session(self, session_id: str, final_results: Dict):
        """Mark training session as completed and log final results"""
        duration_minutes = final_results.get('training_duration_minutes', 0)
        
        with sqlite3.connect(self.monitor_db, check_same_thread=False, timeout=30.0) as conn:
            conn.execute("""
                UPDATE training_sessions SET 
                    end_time = CURRENT_TIMESTAMP,
                    status = ?,
                    total_samples = ?,
                    training_samples = ?,
                    validation_samples = ?,
                    features_count = ?,
                    target_balance = ?,
                    data_quality_score = ?,
                    best_model = ?,
                    best_accuracy = ?,
                    best_f1_score = ?,
                    best_auc = ?,
                    training_duration_minutes = ?
                WHERE session_id = ?
            """, (
                final_results.get('status', 'completed'),
                final_results.get('total_samples', 0),
                final_results.get('training_samples', 0),
                final_results.get('validation_samples', 0),
                final_results.get('features_count', 0),
                final_results.get('target_balance', 0.5),
                final_results.get('data_quality_score', 50),
                final_results.get('best_model', ''),
                final_results.get('best_accuracy', 0),
                final_results.get('best_f1_score', 0),
                final_results.get('best_auc', 0),
                duration_minutes,
                session_id
            ))
        
        logger.info(f"Completed training session {session_id}: {final_results.get('status', 'completed')} in {duration_minutes} minutes")
    
    def get_training_summary(self, session_id: Optional[str] = None) -> Dict:
        """Get comprehensive training summary"""
        with sqlite3.connect(self.monitor_db, check_same_thread=False, timeout=30.0) as conn:
            if session_id:
                # Specific session summary
                session_query = "SELECT * FROM training_sessions WHERE session_id = ?"
                session_data = pd.read_sql_query(session_query, conn, params=(session_id,))
                
                if session_data.empty:
                    return {'error': f'Session {session_id} not found'}
                
                # Get model performance for this session
                perf_query = "SELECT * FROM model_performance WHERE session_id = ?"
                performance_data = pd.read_sql_query(perf_query, conn, params=(session_id,))
                
                # Get training progress
                progress_query = "SELECT * FROM training_progress WHERE session_id = ? ORDER BY timestamp"
                progress_data = pd.read_sql_query(progress_query, conn, params=(session_id,))
                
                return {
                    'session_info': session_data.iloc[0].to_dict(),
                    'model_performance': performance_data.to_dict('records'),
                    'training_progress': progress_data.to_dict('records')
                }
            else:
                # Overall training summary
                sessions_query = """
                    SELECT 
                        COUNT(*) as total_sessions,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_sessions,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_sessions,
                        AVG(training_duration_minutes) as avg_duration_minutes,
                        AVG(best_accuracy) as avg_best_accuracy,
                        MAX(best_accuracy) as max_accuracy_achieved
                    FROM training_sessions
                """
                
                summary = pd.read_sql_query(sessions_query, conn).iloc[0].to_dict()
                
                # Recent performance trends
                recent_query = """
                    SELECT model_name, AVG(accuracy) as avg_accuracy, AVG(f1_score) as avg_f1
                    FROM model_performance 
                    WHERE timestamp >= date('now', '-30 days')
                    GROUP BY model_name
                    ORDER BY avg_f1 DESC
                """
                recent_performance = pd.read_sql_query(recent_query, conn).to_dict('records')
                
                # Active alerts
                alerts_query = """
                    SELECT alert_type, severity, COUNT(*) as count
                    FROM performance_alerts 
                    WHERE resolved = 0 AND timestamp >= date('now', '-7 days')
                    GROUP BY alert_type, severity
                """
                active_alerts = pd.read_sql_query(alerts_query, conn).to_dict('records')
                
                return {
                    'overall_summary': summary,
                    'recent_performance': recent_performance,
                    'active_alerts': active_alerts
                }
    
    def get_model_performance_comparison(self, days_back: int = 30) -> Dict:
        """Compare model performance over time"""
        with sqlite3.connect(self.monitor_db, check_same_thread=False, timeout=30.0) as conn:
            query = f"""
                SELECT 
                    model_name,
                    COUNT(*) as training_sessions,
                    AVG(accuracy) as avg_accuracy,
                    AVG(f1_score) as avg_f1,
                    AVG(auc_roc) as avg_auc,
                    AVG(underdog_prediction_accuracy) as avg_underdog_accuracy,
                    MAX(accuracy) as best_accuracy,
                    MIN(accuracy) as worst_accuracy,
                    AVG(training_time_seconds) as avg_training_time
                FROM model_performance 
                WHERE timestamp >= date('now', '-{days_back} days')
                GROUP BY model_name
                ORDER BY avg_f1 DESC
            """
            
            performance_comparison = pd.read_sql_query(query, conn)
            
            # Performance trends over time
            trends_query = f"""
                SELECT 
                    DATE(timestamp) as date,
                    model_name,
                    AVG(accuracy) as daily_accuracy,
                    AVG(f1_score) as daily_f1
                FROM model_performance 
                WHERE timestamp >= date('now', '-{days_back} days')
                GROUP BY DATE(timestamp), model_name
                ORDER BY date DESC
            """
            
            performance_trends = pd.read_sql_query(trends_query, conn)
            
            return {
                'model_comparison': performance_comparison.to_dict('records'),
                'performance_trends': performance_trends.to_dict('records'),
                'best_overall_model': performance_comparison.iloc[0]['model_name'] if not performance_comparison.empty else None
            }
    
    def generate_training_report(self, session_id: str, save_path: Optional[str] = None) -> Dict:
        """Generate comprehensive training report"""
        summary = self.get_training_summary(session_id)
        
        if 'error' in summary:
            return summary
        
        session_info = summary['session_info']
        model_performance = summary['model_performance']
        training_progress = summary['training_progress']
        
        # Create comprehensive report
        report = {
            'session_id': session_id,
            'generation_time': datetime.now().isoformat(),
            'training_overview': {
                'start_time': session_info['start_time'],
                'end_time': session_info['end_time'],
                'duration_minutes': session_info['training_duration_minutes'],
                'status': session_info['status'],
                'models_trained': json.loads(session_info['models_trained']) if session_info['models_trained'] else []
            },
            'data_summary': {
                'total_samples': session_info['total_samples'],
                'training_samples': session_info['training_samples'],
                'validation_samples': session_info['validation_samples'],
                'features_count': session_info['features_count'],
                'target_balance': session_info['target_balance'],
                'data_quality_score': session_info['data_quality_score']
            },
            'model_performance_summary': {},
            'best_model_analysis': {},
            'training_timeline': training_progress,
            'recommendations': []
        }
        
        # Analyze model performance
        if model_performance:
            best_model = max(model_performance, key=lambda x: x['f1_score'] if x['f1_score'] else 0)
            
            report['best_model_analysis'] = {
                'model_name': best_model['model_name'],
                'accuracy': best_model['accuracy'],
                'f1_score': best_model['f1_score'],
                'auc_roc': best_model['auc_roc'],
                'underdog_prediction_accuracy': best_model['underdog_prediction_accuracy'],
                'cross_validation': {
                    'mean_accuracy': best_model['cv_mean_accuracy'],
                    'std_accuracy': best_model['cv_std_accuracy'],
                    'mean_f1': best_model['cv_mean_f1'],
                    'std_f1': best_model['cv_std_f1']
                }
            }
            
            # Performance comparison
            report['model_performance_summary'] = {
                model['model_name']: {
                    'accuracy': model['accuracy'],
                    'f1_score': model['f1_score'],
                    'auc_roc': model['auc_roc'],
                    'training_time': model['training_time_seconds']
                }
                for model in model_performance
            }
        
        # Generate recommendations
        report['recommendations'] = self._generate_training_recommendations(report)
        
        # Save report if requested
        if save_path:
            report_path = Path(save_path) / f"training_report_{session_id}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Training report saved: {report_path}")
        
        return report
    
    def _generate_training_recommendations(self, report: Dict) -> List[str]:
        """Generate actionable recommendations based on training results"""
        recommendations = []
        
        # Data quality recommendations
        data_quality = report['data_summary']['data_quality_score']
        if data_quality < 70:
            recommendations.append("🔧 Improve data quality by collecting more complete match statistics")
        
        # Target balance recommendations
        target_balance = report['data_summary']['target_balance']
        if target_balance < 0.3:
            recommendations.append("⚖️ Address class imbalance by collecting more underdog wins or using sampling techniques")
        
        # Model performance recommendations
        best_model = report['best_model_analysis']
        if best_model:
            best_f1 = best_model['f1_score']
            if best_f1 < 0.55:
                recommendations.append("📈 Consider feature engineering or hyperparameter optimization to improve model performance")
            
            underdog_acc = best_model['underdog_prediction_accuracy']
            if underdog_acc and underdog_acc < 0.4:
                recommendations.append("🎯 Focus on improving underdog prediction accuracy with specialized features")
        
        # Cross-validation recommendations
        if best_model and best_model['cross_validation']['std_f1'] and best_model['cross_validation']['std_f1'] > 0.1:
            recommendations.append("🔄 High CV variance suggests overfitting - consider regularization or more data")
        
        # Sample size recommendations
        total_samples = report['data_summary']['total_samples']
        if total_samples < 2000:
            recommendations.append("📊 Collect more training samples for better model generalization")
        
        return recommendations

# CLI interface for monitoring operations
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Training Monitor and Performance Tracker')
    parser.add_argument('--summary', action='store_true', help='Show overall training summary')
    parser.add_argument('--session', type=str, help='Show specific session details')
    parser.add_argument('--compare-models', action='store_true', help='Compare model performance')
    parser.add_argument('--alerts', action='store_true', help='Show active alerts')
    parser.add_argument('--report', type=str, help='Generate detailed report for session')
    parser.add_argument('--days-back', type=int, default=30, help='Days back for analysis')
    
    args = parser.parse_args()
    
    monitor = MLTrainingMonitor()
    
    if args.summary:
        summary = monitor.get_training_summary()
        print("\n=== OVERALL TRAINING SUMMARY ===")
        overall = summary['overall_summary']
        print(f"Total sessions: {overall['total_sessions']}")
        print(f"Completed: {overall['completed_sessions']}, Failed: {overall['failed_sessions']}")
        print(f"Average duration: {overall['avg_duration_minutes']:.1f} minutes")
        print(f"Best accuracy achieved: {overall['max_accuracy_achieved']:.3f}")
        
        if summary['recent_performance']:
            print("\n📊 Recent Model Performance (30 days):")
            for model in summary['recent_performance']:
                print(f"  {model['model_name']}: ACC={model['avg_accuracy']:.3f}, F1={model['avg_f1']:.3f}")
        
        if summary['active_alerts']:
            print("\n🚨 Active Alerts:")
            for alert in summary['active_alerts']:
                print(f"  {alert['alert_type']} ({alert['severity']}): {alert['count']} alerts")
    
    if args.session:
        summary = monitor.get_training_summary(args.session)
        if 'error' in summary:
            print(f"❌ {summary['error']}")
        else:
            session = summary['session_info']
            print(f"\n=== SESSION {args.session} ===")
            print(f"Status: {session['status']}")
            print(f"Duration: {session['training_duration_minutes']} minutes")
            print(f"Best model: {session['best_model']} (F1: {session['best_f1_score']:.3f})")
            print(f"Data quality: {session['data_quality_score']}")
    
    if args.compare_models:
        comparison = monitor.get_model_performance_comparison(args.days_back)
        print(f"\n=== MODEL PERFORMANCE COMPARISON ({args.days_back} days) ===")
        print(f"Best overall model: {comparison['best_overall_model']}")
        
        for model in comparison['model_comparison']:
            print(f"\n{model['model_name']}:")
            print(f"  Sessions: {model['training_sessions']}")
            print(f"  Avg F1: {model['avg_f1']:.3f}")
            print(f"  Avg Accuracy: {model['avg_accuracy']:.3f}")
            print(f"  Best: {model['best_accuracy']:.3f}")
    
    if args.report:
        report = monitor.generate_training_report(args.report, "reports")
        if 'error' in report:
            print(f"❌ {report['error']}")
        else:
            print(f"\n=== TRAINING REPORT FOR {args.report} ===")
            print(f"Best model: {report['best_model_analysis']['model_name']}")
            print(f"F1 Score: {report['best_model_analysis']['f1_score']:.3f}")
            print(f"Underdog accuracy: {report['best_model_analysis']['underdog_prediction_accuracy']:.3f}")
            
            if report['recommendations']:
                print("\n💡 Recommendations:")
                for rec in report['recommendations']:
                    print(f"  • {rec}")
    
    if not any([args.summary, args.session, args.compare_models, args.alerts, args.report]):
        print("Use --help for available options")
        print("Example: python ml_training_monitor.py --summary --compare-models")