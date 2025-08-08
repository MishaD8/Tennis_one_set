#!/usr/bin/env python3
"""
🚀 IMPLEMENT ML ENHANCEMENTS FOR TENNIS_ONE_SET
Integration script that works with the existing tennis_backend.py system

This script provides:
1. Seamless integration with existing TennisPredictionService
2. Backward-compatible enhancements
3. Progressive implementation based on data quality
4. Safe rollback mechanisms

Usage:
  python implement_ml_enhancements.py --phase 1
  python implement_ml_enhancements.py --assess-only
  python implement_ml_enhancements.py --full-implementation

Author: Claude Code (Anthropic)
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ml_enhancement_coordinator import StrategicMLCoordinator
    from enhanced_ml_training_system import EnhancedMLTrainingSystem
    from tennis_prediction_module import TennisPredictionService
    ENHANCEMENT_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Enhancement system components not available: {e}")
    ENHANCEMENT_SYSTEM_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLEnhancementImplementor:
    """Main implementation coordinator for ML enhancements"""
    
    def __init__(self):
        self.models_dir = "tennis_models"
        self.backup_dir = "tennis_models_backup"
        self.implementation_log = []
        
        if ENHANCEMENT_SYSTEM_AVAILABLE:
            self.coordinator = StrategicMLCoordinator()
            self.enhanced_system = EnhancedMLTrainingSystem()
        else:
            self.coordinator = None
            self.enhanced_system = None
    
    def assess_system_readiness(self) -> Dict[str, Any]:
        """Comprehensive system readiness assessment"""
        
        logger.info("🔍 Assessing Tennis_one_set system readiness for ML enhancements...")
        
        assessment = {
            'timestamp': datetime.now().isoformat(),
            'system_status': {},
            'data_quality': {},
            'enhancement_readiness': {},
            'recommendations': []
        }
        
        # 1. Check existing system components
        assessment['system_status'] = self._assess_system_components()
        
        # 2. Analyze data quality
        if self.coordinator:
            data_analysis = self.coordinator.data_monitor.get_comprehensive_data_assessment()
            assessment['data_quality'] = data_analysis
        else:
            assessment['data_quality'] = self._basic_data_assessment()
        
        # 3. Determine enhancement readiness
        assessment['enhancement_readiness'] = self._calculate_enhancement_readiness(
            assessment['system_status'], 
            assessment['data_quality']
        )
        
        # 4. Generate strategic recommendations
        assessment['recommendations'] = self._generate_implementation_recommendations(assessment)
        
        return assessment
    
    def _assess_system_components(self) -> Dict[str, Any]:
        """Assess existing Tennis_one_set system components"""
        
        components = {
            'models_directory': os.path.exists(self.models_dir),
            'existing_models': {},
            'prediction_service': False,
            'backend_integration': os.path.exists('tennis_backend.py'),
            'metadata': os.path.exists(os.path.join(self.models_dir, 'metadata.json'))
        }
        
        # Check existing model files
        if components['models_directory']:
            model_files = [
                'random_forest.pkl',
                'gradient_boosting.pkl', 
                'logistic_regression.pkl',
                'xgboost.pkl',
                'neural_network.h5',
                'scaler.pkl'
            ]
            
            for model_file in model_files:
                path = os.path.join(self.models_dir, model_file)
                components['existing_models'][model_file] = {
                    'exists': os.path.exists(path),
                    'size_kb': round(os.path.getsize(path) / 1024, 2) if os.path.exists(path) else 0
                }
        
        # Test TennisPredictionService
        try:
            service = TennisPredictionService()
            service.load_models()
            components['prediction_service'] = service.is_loaded
        except Exception as e:
            logger.warning(f"Could not test TennisPredictionService: {e}")
            components['prediction_service'] = False
        
        return components
    
    def _basic_data_assessment(self) -> Dict[str, Any]:
        """Basic data quality assessment without full coordinator"""
        
        assessment = {
            'overall_score': 50,  # Default moderate score
            'api_sources': {},
            'data_files_present': {}
        }
        
        # Check for data files
        data_files = [
            'api_usage.json',
            'api_cache.json', 
            'rapidapi_requests.json'
        ]
        
        for file_name in data_files:
            if os.path.exists(file_name):
                try:
                    with open(file_name, 'r') as f:
                        data = json.load(f)
                    
                    assessment['data_files_present'][file_name] = {
                        'present': True,
                        'size': len(str(data)),
                        'keys': list(data.keys()) if isinstance(data, dict) else []
                    }
                    
                    # Adjust quality score based on data presence
                    if file_name == 'api_usage.json' and data.get('total_requests', 0) > 50:
                        assessment['overall_score'] += 20
                    elif file_name == 'rapidapi_requests.json' and data.get('matches'):
                        assessment['overall_score'] += 15
                        
                except Exception as e:
                    assessment['data_files_present'][file_name] = {
                        'present': True,
                        'error': str(e)
                    }
            else:
                assessment['data_files_present'][file_name] = {'present': False}
        
        assessment['overall_score'] = min(100, assessment['overall_score'])
        
        return assessment
    
    def _calculate_enhancement_readiness(self, system_status: Dict, data_quality: Dict) -> Dict[str, Any]:
        """Calculate readiness for different enhancement phases"""
        
        readiness = {
            'phase_1': {'ready': False, 'confidence': 'low', 'blockers': []},
            'phase_2': {'ready': False, 'confidence': 'low', 'blockers': []},
            'phase_3': {'ready': False, 'confidence': 'low', 'blockers': []}
        }
        
        data_score = data_quality.get('overall_score', 0)
        models_available = sum(1 for m in system_status.get('existing_models', {}).values() if m.get('exists', False))
        
        # Phase 1 Assessment
        if (system_status.get('models_directory', False) and 
            models_available >= 3 and 
            system_status.get('prediction_service', False)):
            readiness['phase_1']['ready'] = True
            readiness['phase_1']['confidence'] = 'high' if data_score >= 50 else 'medium'
        else:
            if not system_status.get('models_directory', False):
                readiness['phase_1']['blockers'].append('Models directory missing')
            if models_available < 3:
                readiness['phase_1']['blockers'].append(f'Only {models_available}/6 models available')
            if not system_status.get('prediction_service', False):
                readiness['phase_1']['blockers'].append('TennisPredictionService not functional')
        
        # Phase 2 Assessment
        if readiness['phase_1']['ready'] and data_score >= 60:
            readiness['phase_2']['ready'] = True
            readiness['phase_2']['confidence'] = 'high' if data_score >= 75 else 'medium'
        else:
            if not readiness['phase_1']['ready']:
                readiness['phase_2']['blockers'].append('Phase 1 not ready')
            if data_score < 60:
                readiness['phase_2']['blockers'].append(f'Data quality too low: {data_score}/100 (need 60+)')
        
        # Phase 3 Assessment
        if readiness['phase_2']['ready'] and data_score >= 80:
            readiness['phase_3']['ready'] = True
            readiness['phase_3']['confidence'] = 'high'
        else:
            if not readiness['phase_2']['ready']:
                readiness['phase_3']['blockers'].append('Phase 2 not ready')
            if data_score < 80:
                readiness['phase_3']['blockers'].append(f'Data quality insufficient: {data_score}/100 (need 80+)')
        
        return readiness
    
    def _generate_implementation_recommendations(self, assessment: Dict) -> List[str]:
        """Generate specific implementation recommendations"""
        
        recommendations = []
        system_status = assessment['system_status']
        data_quality = assessment['data_quality']
        readiness = assessment['enhancement_readiness']
        
        # System-level recommendations
        if not system_status.get('models_directory', False):
            recommendations.append("🔴 CRITICAL: Create tennis_models directory and train initial models")
        elif sum(1 for m in system_status.get('existing_models', {}).values() if m.get('exists', False)) < 3:
            recommendations.append("🔴 CRITICAL: Train missing ML models before enhancements")
        
        # Data quality recommendations
        data_score = data_quality.get('overall_score', 0)
        
        if data_score < 30:
            recommendations.append("📊 URGENT: Improve API data collection - quality too low for enhancements")
        elif data_score < 60:
            recommendations.append("📈 Focus on Phase 1 only - accumulate more data for advanced features")
        else:
            recommendations.append("🚀 Data quality supports comprehensive enhancements")
        
        # Phase-specific recommendations
        for phase_name, phase_data in readiness.items():
            if phase_data['ready']:
                phase_num = phase_name.split('_')[1]
                recommendations.append(f"✅ {phase_name.upper()} ready for implementation (confidence: {phase_data['confidence']})")
            else:
                blockers = ', '.join(phase_data['blockers'])
                recommendations.append(f"⚠️ {phase_name.upper()} blocked: {blockers}")
        
        # Timing recommendations
        if readiness['phase_1']['ready']:
            recommendations.append("⏰ TIMING: Implement Phase 1 during low-traffic hours (2-6 AM)")
        
        if readiness['phase_2']['ready']:
            recommendations.append("📅 TIMING: Phase 2 can be implemented 12+ hours after Phase 1")
        
        return recommendations
    
    def implement_phase(self, phase: int, force: bool = False) -> Dict[str, Any]:
        """Implement specific enhancement phase"""
        
        logger.info(f"🚀 Starting Phase {phase} implementation...")
        
        result = {
            'phase': phase,
            'start_time': datetime.now().isoformat(),
            'success': False,
            'enhancements_applied': [],
            'performance_improvements': {},
            'errors': [],
            'backup_created': False
        }
        
        try:
            # 1. Pre-implementation checks
            if not force:
                assessment = self.assess_system_readiness()
                phase_key = f'phase_{phase}'
                
                if not assessment['enhancement_readiness'].get(phase_key, {}).get('ready', False):
                    blockers = assessment['enhancement_readiness'][phase_key].get('blockers', [])
                    result['errors'].append(f"Phase {phase} not ready: {', '.join(blockers)}")
                    return result
            
            # 2. Create backup
            result['backup_created'] = self._create_model_backup()
            
            # 3. Load training data
            training_data = self._prepare_training_data()
            if not training_data:
                result['errors'].append("Could not prepare training data")
                return result
            
            X, y = training_data
            
            # 4. Execute phase-specific enhancements
            if self.enhanced_system:
                if phase == 1:
                    phase_result = self.enhanced_system.implement_phase_1_enhancements(X, y)
                elif phase == 2:
                    phase_result = self.enhanced_system.implement_phase_2_enhancements(X, y)
                elif phase == 3:
                    phase_result = self.enhanced_system.implement_phase_3_enhancements(X, y)
                else:
                    result['errors'].append(f"Unknown phase: {phase}")
                    return result
                
                # Merge results
                result['enhancements_applied'] = phase_result.get('enhancements_applied', [])
                result['performance_improvements'] = phase_result.get('cross_validation_results', {})
                
                if phase_result.get('enhancements_applied'):
                    result['success'] = True
                    logger.info(f"✅ Phase {phase} completed successfully!")
                else:
                    result['errors'].append("No enhancements were applied")
            else:
                result['errors'].append("Enhanced ML training system not available")
            
        except Exception as e:
            logger.error(f"❌ Phase {phase} implementation failed: {e}")
            result['errors'].append(str(e))
            
            # Attempt rollback if backup exists
            if result['backup_created']:
                self._rollback_from_backup()
                result['rollback_performed'] = True
        
        result['end_time'] = datetime.now().isoformat()
        
        # Log implementation attempt
        self.implementation_log.append(result)
        self._save_implementation_log()
        
        return result
    
    def _create_model_backup(self) -> bool:
        """Create backup of existing models"""
        try:
            if not os.path.exists(self.models_dir):
                return False
            
            if os.path.exists(self.backup_dir):
                import shutil
                shutil.rmtree(self.backup_dir)
            
            import shutil
            shutil.copytree(self.models_dir, self.backup_dir)
            logger.info("✅ Model backup created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create model backup: {e}")
            return False
    
    def _rollback_from_backup(self) -> bool:
        """Rollback to backup models"""
        try:
            if not os.path.exists(self.backup_dir):
                return False
            
            if os.path.exists(self.models_dir):
                import shutil
                shutil.rmtree(self.models_dir)
            
            import shutil
            shutil.copytree(self.backup_dir, self.models_dir)
            logger.info("✅ Successfully rolled back to backup models")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to rollback from backup: {e}")
            return False
    
    def _prepare_training_data(self):
        """Prepare training data from existing system"""
        try:
            # Try to load real player data using existing collectors
            from real_tennis_predictor_integration import RealPlayerDataCollector
            
            collector = RealPlayerDataCollector()
            
            # Generate training data based on real player rankings and stats
            return self._generate_realistic_training_data(collector)
            
        except Exception as e:
            logger.warning(f"Could not load real data collector: {e}")
            return None
    
    def _generate_realistic_training_data(self, collector):
        """Generate realistic training data using real player data"""
        try:
            import pandas as pd
            import numpy as np
            
            # Get real player data
            sample_players = [
                'jannik sinner', 'carlos alcaraz', 'alexander zverev',
                'daniil medvedev', 'novak djokovic', 'brandon nakashima',
                'bu yunchaokete', 'flavio cobolli', 'aryna sabalenka',
                'iga swiatek', 'coco gauff'
            ]
            
            # Create realistic match scenarios
            n_matches = 800
            features_data = []
            outcomes = []
            
            np.random.seed(42)
            
            for i in range(n_matches):
                # Randomly select two players
                player1 = np.random.choice(sample_players)
                player2 = np.random.choice([p for p in sample_players if p != player1])
                
                # Get real player data
                p1_data = collector.get_player_data(player1)
                p2_data = collector.get_player_data(player2)
                
                # Calculate realistic match features
                p1_form = collector.calculate_recent_form(player1)
                p2_form = collector.calculate_recent_form(player2)
                
                # Surface advantage
                surface = np.random.choice(['Hard', 'Clay', 'Grass'])
                p1_surface_adv = collector.get_surface_advantage(player1, surface)
                
                # H2H data
                h2h_data = collector.get_head_to_head(player1, player2)
                
                # Tournament pressure
                tournament = np.random.choice(['US Open', 'Wimbledon', 'ATP Masters', 'ATP 500'])
                from real_tennis_predictor_integration import TournamentPressureCalculator
                pressure_calc = TournamentPressureCalculator()
                pressure_data = pressure_calc.calculate_pressure(tournament)
                
                # Create feature vector matching expected format
                match_features = {
                    'player_rank': float(p1_data['rank']),
                    'player_age': float(p1_data['age']),
                    'opponent_rank': float(p2_data['rank']),
                    'opponent_age': float(p2_data['age']),
                    'player_recent_matches_count': float(p1_form['recent_matches_count']),
                    'player_recent_win_rate': p1_form['recent_win_rate'],
                    'player_recent_sets_win_rate': p1_form['recent_sets_win_rate'],
                    'player_form_trend': p1_form['form_trend'],
                    'player_days_since_last_match': float(p1_form['days_since_last_match']),
                    'player_surface_matches_count': float(max(10, 50 - p1_data['rank'] // 4)),
                    'player_surface_win_rate': max(0.3, p1_form['recent_win_rate'] + p1_surface_adv),
                    'player_surface_advantage': p1_surface_adv,
                    'player_surface_sets_rate': max(0.3, p1_form['recent_sets_win_rate'] + p1_surface_adv * 0.5),
                    'player_surface_experience': min(1.0, max(0.1, 1.0 - p1_data['rank'] / 200)),
                    'h2h_matches': float(h2h_data['h2h_matches']),
                    'h2h_win_rate': h2h_data['h2h_win_rate'],
                    'h2h_recent_form': h2h_data['h2h_recent_form'],
                    'h2h_sets_advantage': h2h_data['h2h_sets_advantage'],
                    'days_since_last_h2h': float(h2h_data['days_since_last_h2h']),
                    'tournament_importance': float(pressure_data['tournament_importance']),
                    'round_pressure': pressure_data['round_pressure'],
                    'total_pressure': pressure_data['total_pressure'],
                    'is_high_pressure_tournament': float(pressure_data['is_high_pressure_tournament'])
                }
                
                features_data.append(match_features)
                
                # Calculate realistic outcome based on ranking difference and form
                rank_advantage = (p2_data['rank'] - p1_data['rank']) / 100
                form_advantage = p1_form['recent_win_rate'] - 0.5
                
                win_prob = 0.5 + rank_advantage * 0.3 + form_advantage * 0.2
                win_prob = max(0.1, min(0.9, win_prob))
                
                outcome = np.random.binomial(1, win_prob)
                outcomes.append(outcome)
            
            # Convert to DataFrame and arrays
            X = pd.DataFrame(features_data)
            y = np.array(outcomes)
            
            logger.info(f"✅ Generated {len(X)} realistic training examples")
            return X, y
            
        except Exception as e:
            logger.error(f"Failed to generate realistic training data: {e}")
            return None
    
    def _save_implementation_log(self):
        """Save implementation log to file"""
        try:
            log_file = "ml_enhancement_implementation_log.json"
            with open(log_file, 'w') as f:
                json.dump(self.implementation_log, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save implementation log: {e}")
    
    def run_comprehensive_assessment(self):
        """Run comprehensive system assessment and display results"""
        print("\n" + "=" * 80)
        print("🎾 TENNIS_ONE_SET ML ENHANCEMENT ASSESSMENT")
        print("=" * 80)
        
        assessment = self.assess_system_readiness()
        
        # System Status
        print(f"\n📊 SYSTEM STATUS:")
        system_status = assessment['system_status']
        print(f"  Models Directory: {'✅' if system_status['models_directory'] else '❌'}")
        print(f"  Prediction Service: {'✅' if system_status['prediction_service'] else '❌'}")
        print(f"  Backend Integration: {'✅' if system_status['backend_integration'] else '❌'}")
        
        models_count = sum(1 for m in system_status['existing_models'].values() if m.get('exists', False))
        print(f"  Available Models: {models_count}/6")
        
        # Data Quality
        print(f"\n📈 DATA QUALITY:")
        data_quality = assessment['data_quality']
        print(f"  Overall Score: {data_quality['overall_score']:.1f}/100")
        
        if 'api_sources' in data_quality:
            active_sources = sum(1 for source in data_quality['api_sources'].values() 
                                if source.get('status') == 'active')
            print(f"  Active API Sources: {active_sources}")
        
        # Enhancement Readiness
        print(f"\n🚀 ENHANCEMENT READINESS:")
        readiness = assessment['enhancement_readiness']
        
        for phase_name, phase_data in readiness.items():
            status = "✅ READY" if phase_data['ready'] else "⚠️ NOT READY"
            confidence = phase_data['confidence']
            print(f"  {phase_name.upper()}: {status} (confidence: {confidence})")
            
            if phase_data['blockers']:
                for blocker in phase_data['blockers']:
                    print(f"    - {blocker}")
        
        # Recommendations
        print(f"\n💡 STRATEGIC RECOMMENDATIONS:")
        for i, rec in enumerate(assessment['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print(f"\n" + "=" * 80)
        
        return assessment

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='Implement ML enhancements for Tennis_one_set')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3], help='Implement specific phase')
    parser.add_argument('--assess-only', action='store_true', help='Only run assessment')
    parser.add_argument('--full-implementation', action='store_true', help='Implement all ready phases')
    parser.add_argument('--force', action='store_true', help='Force implementation ignoring readiness checks')
    
    args = parser.parse_args()
    
    implementor = MLEnhancementImplementor()
    
    if args.assess_only:
        implementor.run_comprehensive_assessment()
        return
    
    if args.phase:
        print(f"\n🚀 Implementing Phase {args.phase}...")
        result = implementor.implement_phase(args.phase, force=args.force)
        
        if result['success']:
            print(f"✅ Phase {args.phase} implemented successfully!")
            print(f"Enhancements applied: {', '.join(result['enhancements_applied'])}")
        else:
            print(f"❌ Phase {args.phase} implementation failed:")
            for error in result['errors']:
                print(f"  - {error}")
        
        return
    
    if args.full_implementation:
        print(f"\n🚀 Running full implementation plan...")
        
        assessment = implementor.assess_system_readiness()
        readiness = assessment['enhancement_readiness']
        
        implemented_phases = []
        
        for phase_num in [1, 2, 3]:
            phase_key = f'phase_{phase_num}'
            
            if readiness.get(phase_key, {}).get('ready', False):
                print(f"\n📋 Implementing Phase {phase_num}...")
                result = implementor.implement_phase(phase_num, force=args.force)
                
                if result['success']:
                    implemented_phases.append(phase_num)
                    print(f"✅ Phase {phase_num} completed successfully!")
                else:
                    print(f"❌ Phase {phase_num} failed, stopping implementation")
                    break
            else:
                print(f"⏭️ Skipping Phase {phase_num} - not ready")
        
        print(f"\n🎯 IMPLEMENTATION SUMMARY:")
        print(f"Phases completed: {implemented_phases}")
        
        if implemented_phases:
            print(f"✅ {len(implemented_phases)} phase(s) implemented successfully!")
        else:
            print(f"⚠️ No phases were implemented - check system readiness")
        
        return
    
    # Default: run assessment
    implementor.run_comprehensive_assessment()

if __name__ == "__main__":
    main()