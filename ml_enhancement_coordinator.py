#!/usr/bin/env python3
"""
🎯 ML ENHANCEMENT COORDINATOR
Coordinates the strategic implementation of ML enhancements based on data quality and timing

Integrates with existing Tennis_one_set system and provides:
- Real-time data quality assessment
- Strategic timing recommendations  
- Automated enhancement deployment
- Performance monitoring

Author: Claude Code (Anthropic)
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

# Import existing system components
try:
    from enhanced_ml_training_system import EnhancedMLTrainingSystem
    from real_tennis_predictor_integration import RealPlayerDataCollector
    from tennis_prediction_module import TennisPredictionService
    ENHANCED_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Enhanced system not fully available: {e}")
    ENHANCED_SYSTEM_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIDataQualityMonitor:
    """Monitor API data quality and accumulation patterns"""
    
    def __init__(self):
        self.api_usage_file = "api_usage.json"
        self.cache_file = "api_cache.json"
        self.rapidapi_requests_file = "rapidapi_requests.json"
        
    def get_comprehensive_data_assessment(self) -> Dict[str, Any]:
        """Get comprehensive assessment of current data quality and volume"""
        
        assessment = {
            'timestamp': datetime.now().isoformat(),
            'api_sources': {
                'the_odds_api': self._assess_odds_api(),
                'rapidapi': self._assess_rapidapi(),
                'tennisexplorer': self._assess_tennisexplorer(),
                'universal_collector': self._assess_universal_collector()
            },
            'overall_score': 0,
            'readiness_levels': {},
            'timing_recommendations': []
        }
        
        # Calculate overall data quality score
        source_scores = []
        for source, data in assessment['api_sources'].items():
            if data.get('available', False):
                source_scores.append(data.get('quality_score', 0))
        
        if source_scores:
            assessment['overall_score'] = np.mean(source_scores)
        
        # Determine readiness levels for each phase
        assessment['readiness_levels'] = self._calculate_readiness_levels(assessment['overall_score'])
        
        # Generate timing recommendations
        assessment['timing_recommendations'] = self._generate_timing_recommendations(assessment)
        
        return assessment
    
    def _assess_odds_api(self) -> Dict[str, Any]:
        """Assess The Odds API data quality and usage"""
        try:
            if os.path.exists(self.api_usage_file):
                with open(self.api_usage_file, 'r') as f:
                    usage_data = json.load(f)
                
                total_requests = usage_data.get('total_requests', 0)
                hourly_requests = usage_data.get('hourly_requests', [])
                
                # Calculate recent activity
                now = datetime.now()
                recent_requests = 0
                for req_time_str in hourly_requests[-24:]:  # Last 24 entries
                    try:
                        req_time = datetime.fromisoformat(req_time_str.replace('Z', '+00:00'))
                        if (now - req_time).total_seconds() < 86400:  # Last 24 hours
                            recent_requests += 1
                    except:
                        continue
                
                # Quality assessment
                quality_score = min(100, (total_requests / 100) * 50 + (recent_requests / 10) * 50)
                
                return {
                    'available': True,
                    'total_requests': total_requests,
                    'recent_requests_24h': recent_requests,
                    'quality_score': quality_score,
                    'last_request': hourly_requests[-1] if hourly_requests else None,
                    'status': 'active' if recent_requests > 0 else 'inactive'
                }
                
        except Exception as e:
            logger.warning(f"Could not assess Odds API: {e}")
        
        return {
            'available': False,
            'quality_score': 0,
            'status': 'unavailable',
            'error': 'No usage data found'
        }
    
    def _assess_rapidapi(self) -> Dict[str, Any]:
        """Assess RapidAPI data quality and availability"""
        try:
            if os.path.exists(self.rapidapi_requests_file):
                with open(self.rapidapi_requests_file, 'r') as f:
                    rapidapi_data = json.load(f)
                
                total_matches = len(rapidapi_data.get('matches', []))
                tournaments = len(set(match.get('tournament', {}).get('name', 'Unknown') 
                                    for match in rapidapi_data.get('matches', [])))
                
                # Quality based on data volume and diversity
                volume_score = min(50, (total_matches / 50) * 50)
                diversity_score = min(50, (tournaments / 10) * 50)
                quality_score = volume_score + diversity_score
                
                return {
                    'available': True,
                    'total_matches': total_matches,
                    'unique_tournaments': tournaments,
                    'quality_score': quality_score,
                    'status': 'active' if total_matches > 0 else 'inactive'
                }
                
        except Exception as e:
            logger.warning(f"Could not assess RapidAPI: {e}")
        
        return {
            'available': False,
            'quality_score': 0,
            'status': 'unavailable',
            'error': 'No RapidAPI data found'
        }
    
    def _assess_tennisexplorer(self) -> Dict[str, Any]:
        """Assess TennisExplorer integration quality"""
        try:
            # Check if TennisExplorer integration files exist
            te_files = [
                'tennisexplorer_scraper.py',
                'tennisexplorer_integration.py'
            ]
            
            files_present = sum(1 for f in te_files if os.path.exists(f))
            
            if files_present >= 2:
                # Simulate quality assessment based on file presence and integration
                return {
                    'available': True,
                    'integration_files': files_present,
                    'quality_score': 75,  # High quality for real tournament data
                    'status': 'integrated',
                    'data_type': 'real_tournament_schedules'
                }
        except Exception as e:
            logger.warning(f"Could not assess TennisExplorer: {e}")
        
        return {
            'available': False,
            'quality_score': 0,
            'status': 'not_integrated'
        }
    
    def _assess_universal_collector(self) -> Dict[str, Any]:
        """Assess Universal Collector data quality"""
        try:
            # Check for Universal Collector files
            uc_files = [
                'universal_tennis_data_collector.py',
                'enhanced_universal_collector.py'
            ]
            
            files_present = sum(1 for f in uc_files if os.path.exists(f))
            
            if files_present >= 1:
                return {
                    'available': True,
                    'collector_files': files_present,
                    'quality_score': 60,  # Medium quality for generated data
                    'status': 'available',
                    'data_type': 'generated_tournaments'
                }
        except Exception as e:
            logger.warning(f"Could not assess Universal Collector: {e}")
        
        return {
            'available': False,
            'quality_score': 0,
            'status': 'unavailable'
        }
    
    def _calculate_readiness_levels(self, overall_score: float) -> Dict[str, Any]:
        """Calculate readiness levels for different enhancement phases"""
        
        readiness = {
            'phase_1_ready': overall_score >= 30,  # Basic enhancements
            'phase_2_ready': overall_score >= 60,  # Advanced techniques
            'phase_3_ready': overall_score >= 80,  # Production optimization
            'confidence_level': 'high' if overall_score >= 70 else 'medium' if overall_score >= 40 else 'low'
        }
        
        # Add specific feature readiness
        readiness['features_ready'] = {
            'cross_validation': True,  # Always ready
            'advanced_metrics': True,  # Always ready
            'basic_hyperparameter_tuning': overall_score >= 25,
            'feature_selection': overall_score >= 50,
            'bayesian_optimization': overall_score >= 70,
            'advanced_regularization': overall_score >= 75,
            'ensemble_methods': overall_score >= 80
        }
        
        return readiness
    
    def _generate_timing_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate strategic timing recommendations"""
        recommendations = []
        overall_score = assessment['overall_score']
        
        # Current time analysis
        current_hour = datetime.now().hour
        is_peak_hours = 8 <= current_hour <= 22  # Reasonable working hours
        
        # Data quality recommendations
        if overall_score < 30:
            recommendations.extend([
                "🔴 CRITICAL: Data quality too low for meaningful ML improvements",
                "📊 Priority 1: Focus on API data collection and integration",
                "⏳ Recommendation: Wait 3-7 days for data accumulation before ML enhancements",
                "🎯 Implement basic logging and monitoring first"
            ])
        elif overall_score < 60:
            recommendations.extend([
                "🟡 CAUTION: Moderate data quality - proceed with Phase 1 only",
                "✅ Safe to implement: Cross-validation and basic metrics",
                "⚠️ Avoid: Advanced hyperparameter tuning until more data available",
                "📈 Target: Collect data for 5-10 days before Phase 2"
            ])
        else:
            recommendations.extend([
                "🟢 GOOD: Data quality supports comprehensive ML enhancements",
                "🚀 Recommended: Implement all phases progressively",
                "⚡ Optimal timing: During low-traffic periods for system updates",
                "📊 Advanced techniques ready: Bayesian optimization, ensemble methods"
            ])
        
        # Timing-specific recommendations
        if not is_peak_hours:
            recommendations.append("🕐 TIMING: Off-peak hours - ideal for resource-intensive enhancements")
        else:
            recommendations.append("⏰ TIMING: Peak hours - consider lightweight enhancements only")
        
        # API usage recommendations
        api_sources = assessment['api_sources']
        active_sources = sum(1 for source in api_sources.values() if source.get('status') == 'active')
        
        if active_sources == 0:
            recommendations.append("📡 API: No active data sources - implement data collection first")
        elif active_sources == 1:
            recommendations.append("📡 API: Single data source active - consider diversifying data collection")
        else:
            recommendations.append("📡 API: Multiple data sources active - good foundation for ML enhancements")
        
        return recommendations

class EnhancementScheduler:
    """Schedule ML enhancements based on data quality and system load"""
    
    def __init__(self):
        self.data_monitor = APIDataQualityMonitor()
        self.enhancement_history = []
        self.last_assessment = None
        
    def should_trigger_enhancement(self, phase: int) -> Dict[str, Any]:
        """Determine if an enhancement phase should be triggered"""
        
        # Get current data assessment
        current_assessment = self.data_monitor.get_comprehensive_data_assessment()
        self.last_assessment = current_assessment
        
        overall_score = current_assessment['overall_score']
        readiness = current_assessment['readiness_levels']
        
        decision = {
            'should_trigger': False,
            'phase': phase,
            'reason': '',
            'confidence': 'low',
            'recommended_delay_hours': 0,
            'prerequisites': []
        }
        
        # Phase-specific logic
        if phase == 1:
            if overall_score >= 30 or len(self.enhancement_history) == 0:
                decision.update({
                    'should_trigger': True,
                    'reason': 'Basic enhancements safe with current data quality',
                    'confidence': 'high' if overall_score >= 50 else 'medium'
                })
            else:
                decision.update({
                    'reason': 'Data quality too low for reliable results',
                    'recommended_delay_hours': 24,
                    'prerequisites': ['Improve API data collection', 'Wait for more historical data']
                })
        
        elif phase == 2:
            if readiness['phase_2_ready']:
                # Check if enough time has passed since Phase 1
                last_phase_1 = self._get_last_phase_completion(1)
                if last_phase_1:
                    hours_since = (datetime.now() - last_phase_1).total_seconds() / 3600
                    if hours_since >= 12:  # At least 12 hours for data accumulation
                        decision.update({
                            'should_trigger': True,
                            'reason': 'Sufficient data quality and time since Phase 1',
                            'confidence': readiness['confidence_level']
                        })
                    else:
                        decision.update({
                            'reason': 'Need more time for data accumulation after Phase 1',
                            'recommended_delay_hours': int(12 - hours_since),
                            'prerequisites': [f'Wait {int(12 - hours_since)} more hours after Phase 1']
                        })
                else:
                    decision.update({
                        'reason': 'Phase 1 must be completed first',
                        'prerequisites': ['Complete Phase 1 enhancements']
                    })
            else:
                decision.update({
                    'reason': 'Data quality insufficient for advanced techniques',
                    'recommended_delay_hours': 48,
                    'prerequisites': ['Accumulate more training data', 'Improve API coverage']
                })
        
        elif phase == 3:
            if readiness['phase_3_ready']:
                last_phase_2 = self._get_last_phase_completion(2)
                if last_phase_2:
                    hours_since = (datetime.now() - last_phase_2).total_seconds() / 3600
                    if hours_since >= 24:  # More time needed for production-ready optimizations
                        decision.update({
                            'should_trigger': True,
                            'reason': 'High data quality and sufficient time for production optimization',
                            'confidence': 'high'
                        })
                    else:
                        decision.update({
                            'reason': 'Need more data and validation time after Phase 2',
                            'recommended_delay_hours': int(24 - hours_since),
                            'prerequisites': [f'Wait {int(24 - hours_since)} more hours after Phase 2']
                        })
                else:
                    decision.update({
                        'reason': 'Phases 1 and 2 must be completed first',
                        'prerequisites': ['Complete Phase 1 and Phase 2 enhancements']
                    })
            else:
                decision.update({
                    'reason': 'Data quality not ready for production optimizations',
                    'recommended_delay_hours': 72,
                    'prerequisites': ['Build larger training dataset', 'Validate Phase 2 improvements']
                })
        
        return decision
    
    def _get_last_phase_completion(self, phase: int) -> Optional[datetime]:
        """Get timestamp of last phase completion"""
        for record in reversed(self.enhancement_history):
            if record.get('phase') == phase and record.get('status') == 'completed':
                return datetime.fromisoformat(record['timestamp'])
        return None
    
    def record_enhancement_attempt(self, phase: int, status: str, details: Dict[str, Any] = None):
        """Record an enhancement attempt"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'phase': phase,
            'status': status,  # 'started', 'completed', 'failed'
            'data_quality_score': self.last_assessment['overall_score'] if self.last_assessment else 0,
            'details': details or {}
        }
        
        self.enhancement_history.append(record)
        
        # Save history to file
        try:
            with open('ml_enhancement_history.json', 'w') as f:
                json.dump(self.enhancement_history, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save enhancement history: {e}")

class StrategicMLCoordinator:
    """Main coordinator for strategic ML enhancement implementation"""
    
    def __init__(self):
        self.scheduler = EnhancementScheduler()
        self.data_monitor = APIDataQualityMonitor()
        
        if ENHANCED_SYSTEM_AVAILABLE:
            self.enhanced_system = EnhancedMLTrainingSystem()
        else:
            self.enhanced_system = None
            
    def execute_strategic_enhancement_plan(self) -> Dict[str, Any]:
        """Execute strategic enhancement plan based on current system state"""
        
        logger.info("🚀 Starting Strategic ML Enhancement Plan Execution...")
        
        plan_results = {
            'execution_timestamp': datetime.now().isoformat(),
            'phases_executed': [],
            'overall_success': False,
            'performance_improvements': {},
            'next_recommendations': []
        }
        
        # Get comprehensive data assessment
        data_assessment = self.data_monitor.get_comprehensive_data_assessment()
        plan_results['data_assessment'] = data_assessment
        
        logger.info(f"📊 Data Quality Score: {data_assessment['overall_score']:.1f}/100")
        
        # Execute phases based on readiness and timing
        for phase in [1, 2, 3]:
            logger.info(f"\n🔍 Evaluating Phase {phase} readiness...")
            
            decision = self.scheduler.should_trigger_enhancement(phase)
            
            if decision['should_trigger']:
                logger.info(f"✅ Phase {phase} approved: {decision['reason']}")
                
                # Record start of enhancement
                self.scheduler.record_enhancement_attempt(phase, 'started')
                
                try:
                    # Execute the enhancement phase
                    phase_result = self._execute_enhancement_phase(phase, data_assessment)
                    
                    if phase_result.get('success', False):
                        plan_results['phases_executed'].append(phase)
                        plan_results['performance_improvements'][f'phase_{phase}'] = phase_result
                        
                        # Record successful completion
                        self.scheduler.record_enhancement_attempt(phase, 'completed', phase_result)
                        logger.info(f"🎉 Phase {phase} completed successfully!")
                        
                    else:
                        # Record failure
                        self.scheduler.record_enhancement_attempt(phase, 'failed', phase_result)
                        logger.error(f"❌ Phase {phase} failed: {phase_result.get('error', 'Unknown error')}")
                        break  # Stop on failure
                        
                except Exception as e:
                    error_details = {'error': str(e)}
                    self.scheduler.record_enhancement_attempt(phase, 'failed', error_details)
                    logger.error(f"❌ Phase {phase} exception: {e}")
                    break
                    
            else:
                logger.info(f"⏳ Phase {phase} delayed: {decision['reason']}")
                if decision['recommended_delay_hours'] > 0:
                    logger.info(f"⏰ Recommended retry in {decision['recommended_delay_hours']} hours")
                
                plan_results['next_recommendations'].append({
                    'phase': phase,
                    'reason': decision['reason'],
                    'delay_hours': decision['recommended_delay_hours'],
                    'prerequisites': decision['prerequisites']
                })
                
                # Stop at first non-ready phase
                break
        
        # Determine overall success
        plan_results['overall_success'] = len(plan_results['phases_executed']) > 0
        
        # Generate final recommendations
        plan_results['final_recommendations'] = self._generate_final_recommendations(plan_results)
        
        return plan_results
    
    def _execute_enhancement_phase(self, phase: int, data_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific enhancement phase"""
        
        if not self.enhanced_system:
            return {
                'success': False,
                'error': 'Enhanced ML training system not available',
                'phase': phase
            }
        
        try:
            # Create synthetic training data for demonstration
            # In production, this would load real training data
            X, y = self._prepare_training_data()
            
            if phase == 1:
                result = self.enhanced_system.implement_phase_1_enhancements(X, y)
            elif phase == 2:
                result = self.enhanced_system.implement_phase_2_enhancements(X, y)
            elif phase == 3:
                result = self.enhanced_system.implement_phase_3_enhancements(X, y)
            else:
                return {'success': False, 'error': f'Unknown phase: {phase}'}
            
            # Add success indicator
            result['success'] = len(result.get('enhancements_applied', [])) > 0
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'phase': phase
            }
    
    def _prepare_training_data(self) -> tuple:
        """Prepare training data for enhancements (synthetic for demo)"""
        
        # In production, this would load real match data and features
        # For now, create synthetic data matching the expected feature structure
        
        np.random.seed(42)
        
        # Feature names from metadata.json
        feature_names = [
            "player_rank", "player_age", "opponent_rank", "opponent_age",
            "player_recent_matches_count", "player_recent_win_rate",
            "player_recent_sets_win_rate", "player_form_trend",
            "player_days_since_last_match", "player_surface_matches_count",
            "player_surface_win_rate", "player_surface_advantage",
            "player_surface_sets_rate", "player_surface_experience",
            "h2h_matches", "h2h_win_rate", "h2h_recent_form",
            "h2h_sets_advantage", "days_since_last_h2h",
            "tournament_importance", "round_pressure", "total_pressure",
            "is_high_pressure_tournament"
        ]
        
        # Generate synthetic but realistic tennis data
        n_samples = 1000
        
        data = {}
        
        # Player rankings (1-500)
        data['player_rank'] = np.random.uniform(1, 500, n_samples)
        data['opponent_rank'] = np.random.uniform(1, 500, n_samples)
        
        # Ages (18-38)
        data['player_age'] = np.random.uniform(18, 38, n_samples)
        data['opponent_age'] = np.random.uniform(18, 38, n_samples)
        
        # Match counts
        data['player_recent_matches_count'] = np.random.uniform(5, 25, n_samples)
        data['player_surface_matches_count'] = np.random.uniform(10, 100, n_samples)
        data['h2h_matches'] = np.random.uniform(0, 15, n_samples)
        
        # Win rates and probabilities (0-1)
        data['player_recent_win_rate'] = np.random.beta(2, 2, n_samples)  # Bell curve around 0.5
        data['player_recent_sets_win_rate'] = data['player_recent_win_rate'] * np.random.uniform(0.8, 1.2, n_samples)
        data['player_surface_win_rate'] = np.random.beta(2, 2, n_samples)
        data['h2h_win_rate'] = np.random.beta(2, 2, n_samples)
        data['h2h_recent_form'] = data['h2h_win_rate'] + np.random.normal(0, 0.1, n_samples)
        data['player_surface_experience'] = np.random.beta(3, 2, n_samples)  # Skewed toward experienced
        
        # Trends and advantages (-0.3 to 0.3)
        data['player_form_trend'] = np.random.normal(0, 0.1, n_samples)
        data['player_surface_advantage'] = np.random.normal(0, 0.1, n_samples)
        
        # Time-based features
        data['player_days_since_last_match'] = np.random.uniform(1, 30, n_samples)
        data['days_since_last_h2h'] = np.random.uniform(30, 1000, n_samples)
        data['h2h_sets_advantage'] = np.random.normal(0, 1, n_samples)
        
        # Tournament features
        data['tournament_importance'] = np.random.choice([1, 2, 3, 4], n_samples, p=[0.4, 0.3, 0.2, 0.1])
        data['round_pressure'] = np.random.uniform(0.1, 1.0, n_samples)
        data['total_pressure'] = data['tournament_importance'] * (1 + data['round_pressure'])
        data['is_high_pressure_tournament'] = (data['tournament_importance'] >= 3).astype(int)
        
        # Clip values to reasonable ranges
        for key in data:
            if key.endswith('_win_rate') or key.endswith('_rate'):
                data[key] = np.clip(data[key], 0, 1)
        
        # Create DataFrame
        X = pd.DataFrame(data)
        
        # Create target variable (player wins at least one set)
        # Base probability on ranking difference and form
        rank_advantage = (data['opponent_rank'] - data['player_rank']) / 100
        form_advantage = data['player_recent_win_rate'] - 0.5
        
        base_prob = 0.5 + rank_advantage * 0.3 + form_advantage * 0.2
        base_prob = np.clip(base_prob, 0.1, 0.9)
        
        y = np.random.binomial(1, base_prob)
        
        return X, y
    
    def _generate_final_recommendations(self, plan_results: Dict[str, Any]) -> List[str]:
        """Generate final strategic recommendations"""
        recommendations = []
        
        phases_executed = plan_results['phases_executed']
        
        if not phases_executed:
            recommendations.extend([
                "🔴 No enhancements executed - focus on data collection first",
                "📊 Improve API data quality and volume",
                "⏳ Retry enhancement plan in 24-48 hours",
                "🎯 Consider manual data collection if APIs unavailable"
            ])
        elif len(phases_executed) == 1:
            recommendations.extend([
                "🟡 Phase 1 completed - good foundation established",
                "📈 Monitor model performance with new cross-validation metrics",
                "🔄 Continue data collection for Phase 2 readiness",
                "⏰ Plan Phase 2 implementation in 12-24 hours"
            ])
        elif len(phases_executed) == 2:
            recommendations.extend([
                "🟢 Phases 1-2 completed - advanced ML system ready",
                "🚀 Feature selection may have optimized model efficiency",
                "📊 Hyperparameter tuning should improve prediction accuracy",
                "🎯 Consider Phase 3 for production deployment"
            ])
        elif len(phases_executed) == 3:
            recommendations.extend([
                "🏆 Complete ML enhancement roadmap implemented!",
                "⚡ Production-ready system with advanced optimizations",
                "🔄 Establish automated monitoring and retraining pipeline",
                "📈 Track performance improvements in live predictions"
            ])
        
        # Add data-specific recommendations
        data_score = plan_results.get('data_assessment', {}).get('overall_score', 0)
        
        if data_score < 40:
            recommendations.append("🔧 PRIORITY: Focus on improving data collection systems")
        elif data_score < 70:
            recommendations.append("📊 GOOD: Current data supports implemented enhancements")
        else:
            recommendations.append("🎉 EXCELLENT: Data quality supports all advanced features")
        
        return recommendations
    
    def get_system_status_report(self) -> Dict[str, Any]:
        """Get comprehensive system status report"""
        
        data_assessment = self.data_monitor.get_comprehensive_data_assessment()
        
        # Check enhancement history
        enhancement_history = getattr(self.scheduler, 'enhancement_history', [])
        
        # Get model status
        model_status = self._assess_model_status()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_quality': {
                'overall_score': data_assessment['overall_score'],
                'api_sources': data_assessment['api_sources'],
                'readiness_levels': data_assessment['readiness_levels']
            },
            'enhancement_status': {
                'phases_completed': len([h for h in enhancement_history if h.get('status') == 'completed']),
                'last_enhancement': enhancement_history[-1] if enhancement_history else None,
                'system_ready': self.enhanced_system is not None
            },
            'model_status': model_status,
            'recommendations': data_assessment['timing_recommendations']
        }
        
        return report
    
    def _assess_model_status(self) -> Dict[str, Any]:
        """Assess current model status"""
        models_dir = "tennis_models"
        
        status = {
            'models_available': {},
            'metadata_present': False,
            'last_training': None
        }
        
        # Check for model files
        model_files = [
            'random_forest.pkl', 'gradient_boosting.pkl',
            'logistic_regression.pkl', 'xgboost.pkl',
            'neural_network.h5'
        ]
        
        for model_file in model_files:
            filepath = os.path.join(models_dir, model_file)
            if os.path.exists(filepath):
                try:
                    stat = os.stat(filepath)
                    status['models_available'][model_file] = {
                        'present': True,
                        'size_mb': round(stat.st_size / 1024 / 1024, 2),
                        'last_modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    }
                except:
                    status['models_available'][model_file] = {'present': True, 'error': 'Could not read file info'}
            else:
                status['models_available'][model_file] = {'present': False}
        
        # Check metadata
        metadata_path = os.path.join(models_dir, 'metadata.json')
        status['metadata_present'] = os.path.exists(metadata_path)
        
        return status

def main():
    """Main demonstration function"""
    print("🎯 ML ENHANCEMENT COORDINATOR")
    print("=" * 60)
    
    coordinator = StrategicMLCoordinator()
    
    # Get system status report
    print("\n📊 SYSTEM STATUS REPORT:")
    status_report = coordinator.get_system_status_report()
    
    print(f"Data Quality Score: {status_report['data_quality']['overall_score']:.1f}/100")
    print(f"Phases Completed: {status_report['enhancement_status']['phases_completed']}")
    print(f"Models Available: {sum(1 for m in status_report['model_status']['models_available'].values() if m.get('present', False))}")
    
    # Execute strategic enhancement plan
    print(f"\n🚀 EXECUTING STRATEGIC ENHANCEMENT PLAN:")
    print("-" * 60)
    
    plan_results = coordinator.execute_strategic_enhancement_plan()
    
    print(f"\nPhases Executed: {len(plan_results['phases_executed'])}")
    print(f"Overall Success: {plan_results['overall_success']}")
    
    print(f"\n🎯 FINAL RECOMMENDATIONS:")
    for rec in plan_results['final_recommendations']:
        print(f"  {rec}")
    
    print(f"\n" + "=" * 60)
    print("✅ ML Enhancement Coordination Complete!")

if __name__ == "__main__":
    main()