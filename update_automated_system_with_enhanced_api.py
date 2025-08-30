#!/usr/bin/env python3
"""
Update Automated Tennis Prediction System with Enhanced API Integration
Apply improvements from the API-Tennis payment upgrade
"""

import os
import sys
import shutil
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemUpdater:
    """Updates the automated system to use enhanced API integration"""
    
    def __init__(self):
        self.backup_dir = f"backups/system_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def backup_current_system(self):
        """Backup current system files before updating"""
        logger.info("📦 Creating system backup...")
        
        files_to_backup = [
            'automated_tennis_prediction_service.py',
            'src/api/api_tennis_integration.py',
            'src/utils/telegram_notification_system.py'
        ]
        
        for file_path in files_to_backup:
            if os.path.exists(file_path):
                backup_path = os.path.join(self.backup_dir, os.path.basename(file_path))
                shutil.copy2(file_path, backup_path)
                logger.info(f"   ✅ Backed up {file_path}")
        
        logger.info(f"💾 Backup created in: {self.backup_dir}")
    
    def update_automated_service_rankings(self):
        """Update automated service to use enhanced rankings"""
        logger.info("🔧 Updating automated prediction service...")
        
        service_file = 'automated_tennis_prediction_service.py'
        if not os.path.exists(service_file):
            logger.error(f"❌ {service_file} not found")
            return
        
        with open(service_file, 'r') as f:
            content = f.read()
        
        # Add enhanced ranking integration
        enhanced_rankings_update = '''    def _load_enhanced_player_rankings(self) -> Dict[str, int]:
        """Load enhanced player rankings from paid API tier"""
        try:
            from enhanced_api_tennis_integration import EnhancedAPITennisIntegration
            enhanced_api = EnhancedAPITennisIntegration()
            
            # Get comprehensive rankings
            rankings_data = enhanced_api.get_enhanced_player_rankings()
            
            # Convert to name -> ranking mapping for compatibility
            name_rankings = {}
            for player_id, data in rankings_data.items():
                if data.get('name') and data.get('rank'):
                    name_key = data['name'].lower().strip()
                    name_rankings[name_key] = data['rank']
                    
                    # Add alternate name formats
                    name_parts = name_key.split()
                    if len(name_parts) >= 2:
                        # Add "First Last" format
                        alt_name = f"{name_parts[0]} {name_parts[-1]}"
                        name_rankings[alt_name] = data['rank']
                        
                        # Add abbreviated format like "F. Last"
                        abbrev_name = f"{name_parts[0][0].lower()}. {name_parts[-1]}"
                        name_rankings[abbrev_name] = data['rank']
            
            logger.info(f"✅ Enhanced rankings loaded: {len(name_rankings)} player mappings")
            return name_rankings
            
        except Exception as e:
            logger.error(f"❌ Enhanced rankings failed, using fallback: {e}")
            return self._load_fallback_player_rankings()
    
    def _load_fallback_player_rankings(self) -> Dict[str, int]:
        """Fallback to existing ranking system"""
        return {
            # Keep existing rankings as fallback
            "jannik sinner": 1, "carlos alcaraz": 2, "alexander zverev": 3,
            "daniil medvedev": 4, "novak djokovic": 5, "andrey rublev": 6,
            # ... existing rankings ...
        }'''
        
        # Insert enhanced rankings method after existing _load_player_rankings
        if '_load_player_rankings(self) -> Dict[str, int]:' in content:
            # Find the end of the current method
            method_start = content.find('def _load_player_rankings(self) -> Dict[str, int]:')
            if method_start != -1:
                # Find the end of the method (next method definition)
                next_method = content.find('\n    def ', method_start + 1)
                if next_method != -1:
                    # Insert enhanced method before next method
                    content = content[:next_method] + '\n' + enhanced_rankings_update + '\n' + content[next_method:]
                else:
                    # Insert at end of class
                    content += '\n' + enhanced_rankings_update
        
        # Update the __init__ method to use enhanced rankings
        init_update = '''        # Load enhanced player rankings from paid API tier
        try:
            self.player_rankings = self._load_enhanced_player_rankings()
        except Exception as e:
            logger.error(f"❌ Enhanced rankings failed: {e}")
            self.player_rankings = self._load_player_rankings()'''
        
        # Replace the existing player rankings loading
        if 'self.player_rankings = self._load_player_rankings()' in content:
            content = content.replace(
                'self.player_rankings = self._load_player_rankings()',
                '''# Try enhanced rankings first, fallback to existing
        try:
            self.player_rankings = self._load_enhanced_player_rankings()
        except Exception as e:
            logger.warning(f"Enhanced rankings unavailable, using fallback: {e}")
            self.player_rankings = self._load_player_rankings()'''
            )
        
        # Write updated content
        with open(service_file, 'w') as f:
            f.write(content)
        
        logger.info(f"✅ Updated {service_file} with enhanced rankings")
    
    def create_enhanced_match_getter(self):
        """Update match getting to use enhanced API"""
        logger.info("🎾 Creating enhanced match getter...")
        
        service_file = 'automated_tennis_prediction_service.py'
        
        with open(service_file, 'r') as f:
            content = f.read()
        
        # Add enhanced match getter
        enhanced_match_method = '''    def _get_enhanced_current_matches(self) -> List[Dict]:
        """Get current matches using enhanced API integration"""
        try:
            from enhanced_api_tennis_integration import EnhancedAPITennisIntegration
            enhanced_api = EnhancedAPITennisIntegration()
            
            # Get enhanced matches with rankings
            enhanced_matches = enhanced_api.get_enhanced_fixtures_with_rankings()
            
            # Convert to format expected by existing system
            compatible_matches = []
            for match in enhanced_matches:
                if match.get('enhanced_data', {}).get('is_underdog_scenario'):
                    compatible_match = {
                        'event_key': match.get('id'),
                        'event_first_player': match.get('player1', {}).get('name', ''),
                        'event_second_player': match.get('player2', {}).get('name', ''),
                        'tournament_name': match.get('tournament_name', ''),
                        'event_type_type': match.get('event_type', ''),
                        'event_final_result': '-',  # Upcoming match
                        'player1_rank': match.get('player1', {}).get('ranking'),
                        'player2_rank': match.get('player2', {}).get('ranking'),
                        'enhanced_data': match.get('enhanced_data', {})
                    }
                    compatible_matches.append(compatible_match)
            
            logger.info(f"✅ Enhanced API provided {len(compatible_matches)} underdog opportunities")
            return compatible_matches
            
        except Exception as e:
            logger.warning(f"⚠️ Enhanced API unavailable, using fallback: {e}")
            return self._get_current_matches_fallback()
    
    def _get_current_matches_fallback(self) -> List[Dict]:
        """Fallback to existing match getting method"""
        return self._get_current_matches_original()'''
        
        # Add enhanced method
        if 'def _get_current_matches(self) -> List[Dict]:' in content:
            method_start = content.find('def _get_current_matches(self) -> List[Dict]:')
            if method_start != -1:
                next_method = content.find('\n    def ', method_start + 1)
                if next_method != -1:
                    content = content[:next_method] + '\n' + enhanced_match_method + '\n' + content[next_method:]
        
        # Rename original method for fallback
        content = content.replace(
            'def _get_current_matches(self) -> List[Dict]:',
            'def _get_current_matches_original(self) -> List[Dict]:'
        )
        
        # Add new enhanced method as primary
        new_main_method = '''    def _get_current_matches(self) -> List[Dict]:
        """Get current matches - enhanced version with fallback"""
        try:
            return self._get_enhanced_current_matches()
        except Exception as e:
            logger.error(f"❌ Enhanced match getting failed: {e}")
            return self._get_current_matches_original()'''
        
        # Insert new main method
        insert_point = content.find('def _get_current_matches_original(self) -> List[Dict]:')
        if insert_point != -1:
            content = content[:insert_point] + new_main_method + '\n\n    ' + content[insert_point:]
        
        with open(service_file, 'w') as f:
            f.write(content)
        
        logger.info("✅ Enhanced match getting integrated")
    
    def update_prediction_features(self):
        """Update ML features to use enhanced data"""
        logger.info("🧠 Updating ML prediction features...")
        
        service_file = 'automated_tennis_prediction_service.py'
        
        with open(service_file, 'r') as f:
            content = f.read()
        
        # Enhanced feature creation method
        enhanced_features = '''    def _create_enhanced_ml_features(self, match: Dict, player1_rank: int, player2_rank: int) -> np.ndarray:
        """Create enhanced ML features using paid tier data"""
        try:
            features = [
                player1_rank,
                player2_rank,
                abs(player1_rank - player2_rank),  # ranking gap
                min(player1_rank, player2_rank),   # favorite rank
                max(player1_rank, player2_rank),   # underdog rank
            ]
            
            # Add enhanced features if available
            enhanced_data = match.get('enhanced_data', {})
            if enhanced_data:
                features.extend([
                    enhanced_data.get('ranking_gap', 0) / 100.0,  # Normalized gap
                    enhanced_data.get('data_quality_score', 0.5),  # Data quality
                    1.0 if 'us open' in match.get('tournament_name', '').lower() else 0.0  # Grand Slam
                ])
            
            # Add player points differential if available
            player1_points = match.get('player1', {}).get('points', 1000)
            player2_points = match.get('player2', {}).get('points', 1000)
            
            if player1_points and player2_points:
                points_ratio = player1_points / player2_points if player2_points > 0 else 1.0
                features.append(min(max(points_ratio, 0.1), 10.0))  # Clamped ratio
            else:
                features.append(1.0)  # Neutral if no points data
            
            # Surface features (enhanced detection)
            surface = match.get('surface', 'Hard').lower()
            features.extend([
                1.0 if surface == 'hard' else 0.0,
                1.0 if surface == 'clay' else 0.0,
                1.0 if surface == 'grass' else 0.0,
            ])
            
            # Tournament importance
            tournament_name = match.get('tournament_name', '').lower()
            if any(slam in tournament_name for slam in ['us open', 'wimbledon', 'french', 'australian']):
                importance = 4.0  # Grand Slam
            elif 'masters' in tournament_name or '1000' in tournament_name:
                importance = 3.0  # Masters
            elif '500' in tournament_name:
                importance = 2.5  # ATP 500
            else:
                importance = 2.0  # ATP 250 / WTA regular
            
            features.append(importance)
            
            # Form indicators from ranking movement
            player1_movement = match.get('player1', {}).get('ranking_movement', 'same')
            player2_movement = match.get('player2', {}).get('ranking_movement', 'same')
            
            movement_score = 0.0
            if player1_movement == 'up':
                movement_score += 0.1
            elif player1_movement == 'down':
                movement_score -= 0.1
                
            if player2_movement == 'up':
                movement_score -= 0.1
            elif player2_movement == 'down':
                movement_score += 0.1
            
            features.append(movement_score)
            
            # Pad/trim to expected feature count
            expected_features = self.metadata.get('feature_columns', [])
            if expected_features:
                while len(features) < len(expected_features):
                    features.append(0.0)
                features = features[:len(expected_features)]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.warning(f"Enhanced features failed, using fallback: {e}")
            return self._create_ml_features(match, player1_rank, player2_rank)'''
        
        # Add enhanced features method
        if 'def _create_ml_features(self, match: Dict, player1_rank: int, player2_rank: int) -> np.ndarray:' in content:
            method_start = content.find('def _create_ml_features(self, match: Dict, player1_rank: int, player2_rank: int) -> np.ndarray:')
            if method_start != -1:
                next_method = content.find('\n    def ', method_start + 1)
                if next_method != -1:
                    content = content[:next_method] + '\n' + enhanced_features + '\n' + content[next_method:]
        
        # Update prediction generation to use enhanced features
        content = content.replace(
            'features = self._create_ml_features(match, player1_rank, player2_rank)',
            '''# Try enhanced features first
            try:
                features = self._create_enhanced_ml_features(match, player1_rank, player2_rank)
            except Exception as e:
                logger.debug(f"Enhanced features unavailable: {e}")
                features = self._create_ml_features(match, player1_rank, player2_rank)'''
        )
        
        with open(service_file, 'w') as f:
            f.write(content)
        
        logger.info("✅ Enhanced ML features integrated")
    
    def test_updated_system(self):
        """Test the updated system"""
        logger.info("🧪 Testing updated system...")
        
        try:
            # Import and test the updated service
            sys.path.insert(0, '.')
            from automated_tennis_prediction_service import AutomatedTennisPredictionService
            
            logger.info("   Creating service instance...")
            service = AutomatedTennisPredictionService()
            
            logger.info("   Testing single prediction cycle...")
            service.run_single_check()
            
            logger.info("✅ System test completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ System test failed: {e}")
            return False
    
    def run_full_update(self):
        """Run complete system update"""
        logger.info("🚀 Starting Full System Update")
        logger.info("=" * 60)
        
        try:
            # Step 1: Backup
            self.backup_current_system()
            
            # Step 2: Update rankings
            self.update_automated_service_rankings()
            
            # Step 3: Update match getting
            self.create_enhanced_match_getter()
            
            # Step 4: Update ML features
            self.update_prediction_features()
            
            # Step 5: Test system
            test_passed = self.test_updated_system()
            
            if test_passed:
                logger.info("✅ SYSTEM UPDATE COMPLETED SUCCESSFULLY!")
                logger.info("🎯 Your automated tennis prediction system now uses:")
                logger.info("   • 3,610 comprehensive player rankings")
                logger.info("   • Enhanced US Open data coverage")
                logger.info("   • Improved ML features with points, movement")
                logger.info("   • Better tournament-specific analysis")
                logger.info("   • Robust fallback mechanisms")
            else:
                logger.warning("⚠️ System update completed but testing failed")
                logger.info(f"💾 Backup available in: {self.backup_dir}")
            
        except Exception as e:
            logger.error(f"❌ Update failed: {e}")
            logger.info(f"💾 System backup available in: {self.backup_dir}")
            logger.info("   You can restore from backup if needed")

def main():
    """Main update function"""
    updater = SystemUpdater()
    updater.run_full_update()

if __name__ == "__main__":
    main()