#!/usr/bin/env python3
"""
🎾 SECOND SET INTEGRATION
Integration layer to modify existing tennis backend for second set prediction
"""

import os
import sys
from typing import Dict, List, Optional
from datetime import datetime

# Import existing components
from second_set_prediction_service import SecondSetPredictionService
from second_set_feature_engineering import SecondSetFeatureEngineer

class SecondSetIntegrator:
    """
    Integration layer to modify existing tennis system for second set predictions
    """
    
    def __init__(self):
        self.second_set_service = SecondSetPredictionService()
        self.feature_engineer = SecondSetFeatureEngineer()
        
        # Try to load existing models for second set adaptation
        self.second_set_service.load_models(retrain_for_second_set=True)
        
    def create_modified_underdog_analyzer(self):
        """
        Create modified UnderdogAnalyzer class for second set predictions
        This replaces the existing UnderdogAnalyzer in tennis_backend.py
        """
        
        class SecondSetUnderdogAnalyzer:
            """
            MODIFIED UnderdogAnalyzer for SECOND SET prediction focus
            """
            
            def __init__(self):
                self.second_set_service = SecondSetPredictionService()
                self.feature_engineer = SecondSetFeatureEngineer()
                
                # Keep existing player rankings for compatibility
                self.player_rankings = {
                    # ATP rankings (from original)
                    "jannik sinner": 1, "carlos alcaraz": 2, "alexander zverev": 3,
                    "daniil medvedev": 4, "novak djokovic": 5, "andrey rublev": 6,
                    "casper ruud": 7, "holger rune": 8, "grigor dimitrov": 9,
                    "stefanos tsitsipas": 10, "taylor fritz": 11, "tommy paul": 12,
                    # ... (keep all existing rankings)
                    "flavio cobolli": 32, "brandon nakashima": 45, "bu yunchaokete": 85,
                    # WTA rankings (from original)  
                    "aryna sabalenka": 1, "iga swiatek": 2, "coco gauff": 3,
                    # ... (keep all existing rankings)
                }
            
            def get_player_ranking(self, player_name: str) -> int:
                """Keep existing ranking logic"""
                name_lower = player_name.replace('🎾 ', '').lower().strip()
                
                if name_lower in self.player_rankings:
                    return self.player_rankings[name_lower]
                
                for known_player, rank in self.player_rankings.items():
                    if any(part in known_player for part in name_lower.split()):
                        return rank
                
                return 50
            
            def simulate_first_set_data(self, player1: str, player2: str, 
                                      tournament: str, surface: str) -> Dict:
                """
                Simulate realistic first set data for testing second set prediction
                In production, this would come from real live match data
                """
                p1_rank = self.get_player_ranking(player1)
                p2_rank = self.get_player_ranking(player2)
                
                # Simulate first set outcome based on rankings
                import random
                random.seed(hash(player1 + player2 + tournament) % 1000)
                
                # Stronger player (lower rank number) more likely to win first set
                if p1_rank < p2_rank:
                    first_set_winner = "player1" if random.random() < 0.7 else "player2"
                elif p1_rank > p2_rank:
                    first_set_winner = "player2" if random.random() < 0.7 else "player1"
                else:
                    first_set_winner = "player1" if random.random() < 0.5 else "player2"
                
                # Simulate score based on competitiveness
                rank_gap = abs(p1_rank - p2_rank)
                if rank_gap > 100:
                    # Big gap - likely dominant set
                    scores = ["6-2", "6-3", "6-1"]
                elif rank_gap > 50:
                    # Medium gap
                    scores = ["6-4", "6-3", "7-5"]
                else:
                    # Close match
                    scores = ["7-6", "7-5", "6-4"]
                
                score = random.choice(scores)
                had_tiebreak = "7-6" in score
                
                # Simulate other match statistics
                first_set_data = {
                    "winner": first_set_winner,
                    "score": score,
                    "duration_minutes": random.randint(35, 70),
                    "breaks_won_player1": random.randint(0, 3),
                    "breaks_won_player2": random.randint(0, 3),
                    "break_points_saved_player1": random.uniform(0.3, 0.9),
                    "break_points_saved_player2": random.uniform(0.3, 0.9),
                    "first_serve_percentage_player1": random.uniform(0.55, 0.85),
                    "first_serve_percentage_player2": random.uniform(0.55, 0.85),
                    "had_tiebreak": had_tiebreak
                }
                
                return first_set_data
            
            def calculate_second_set_underdog_probability(self, player1: str, player2: str,
                                                        tournament: str, surface: str) -> Dict:
                """
                MAIN METHOD: Calculate underdog probability for SECOND SET
                This replaces the original calculate_underdog_probability method
                """
                
                # Get player data
                p1_rank = self.get_player_ranking(player1)
                p2_rank = self.get_player_ranking(player2)
                
                player1_data = {"rank": p1_rank, "age": 25}  # Default age
                player2_data = {"rank": p2_rank, "age": 25}
                
                # Create match context
                match_context = {
                    "tournament_importance": 4 if any(slam in tournament.lower() 
                                                   for slam in ["wimbledon", "us open", "french", "australian"]) else 2,
                    "total_pressure": 3.8 if "slam" in tournament.lower() else 2.5,
                    "player1_surface_advantage": 0.0  # Simplified for now
                }
                
                # Simulate first set data (in production, get from live match)
                first_set_data = self.simulate_first_set_data(player1, player2, tournament, surface)
                
                # Use second set prediction service
                try:
                    result = self.second_set_service.predict_second_set(
                        player1.replace('🎾 ', ''), player2.replace('🎾 ', ''),
                        player1_data, player2_data, match_context, first_set_data,
                        return_details=True
                    )
                    
                    # Determine which player is underdog and extract their probability
                    if result['underdog_player'] == 'player1':
                        underdog = player1
                        favorite = player2
                        underdog_probability = result['underdog_second_set_probability']
                    else:
                        underdog = player2  
                        favorite = player1
                        underdog_probability = result['underdog_second_set_probability']
                    
                    # Create response in original format for compatibility
                    return {
                        'underdog_probability': underdog_probability,
                        'quality': 'Excellent' if result['confidence'] == 'High' else 'Good',
                        'confidence': result['confidence'],
                        'ml_system_used': 'SECOND_SET_ML_MODEL',
                        'prediction_type': 'SECOND_SET_UNDERDOG_PREDICTION',
                        'key_factors': result['key_factors'],
                        'underdog_scenario': {
                            'underdog': underdog,
                            'favorite': favorite,
                            'underdog_rank': p1_rank if result['underdog_player'] == 'player1' else p2_rank,
                            'favorite_rank': p2_rank if result['underdog_player'] == 'player1' else p1_rank,
                            'rank_gap': abs(p1_rank - p2_rank),
                            'underdog_type': self._classify_underdog_type(abs(p1_rank - p2_rank)),
                            'base_probability': 0.35  # Base for second set
                        },
                        'first_set_context': {
                            'winner': first_set_data['winner'],
                            'score': first_set_data['score'],
                            'duration': first_set_data['duration_minutes']
                        },
                        'second_set_specifics': {
                            'focus': 'Second set win probability for underdog',
                            'scenario': 'Underdog needs to win SET 2 specifically',
                            'market_advantage': 'More specific than general match betting'
                        }
                    }
                    
                except Exception as e:
                    print(f"⚠️ Second set prediction error: {e}")
                    # Fallback to enhanced simulation
                    return self._fallback_second_set_simulation(player1, player2, tournament, surface)
            
            def _classify_underdog_type(self, rank_gap: int) -> str:
                """Classify underdog based on ranking gap"""
                if rank_gap >= 200:
                    return "Extreme Underdog"
                elif rank_gap >= 100:
                    return "Major Underdog" 
                elif rank_gap >= 50:
                    return "Significant Underdog"
                elif rank_gap >= 20:
                    return "Moderate Underdog"
                else:
                    return "Minor Underdog"
            
            def _fallback_second_set_simulation(self, player1: str, player2: str,
                                              tournament: str, surface: str) -> Dict:
                """Fallback simulation for second set when ML models fail"""
                
                p1_rank = self.get_player_ranking(player1)
                p2_rank = self.get_player_ranking(player2)
                rank_gap = abs(p1_rank - p2_rank)
                
                # Determine underdog
                if p1_rank > p2_rank:
                    underdog = player1
                    favorite = player2
                else:
                    underdog = player2
                    favorite = player1
                
                # Second set specific probability (higher than match probability)
                base_prob = 0.35  # Base chance for underdog to win a set
                
                # Adjustments for second set dynamics
                if rank_gap > 200:
                    second_set_prob = base_prob + 0.05  # Nothing to lose effect
                elif rank_gap > 100:
                    second_set_prob = base_prob + 0.08
                elif rank_gap > 50:
                    second_set_prob = base_prob + 0.10
                else:
                    second_set_prob = base_prob + 0.15
                
                # Tournament pressure can help underdogs in second set
                if any(slam in tournament.lower() for slam in ["wimbledon", "us open", "french", "australian"]):
                    second_set_prob += 0.05
                
                second_set_prob = max(0.20, min(second_set_prob, 0.65))
                
                return {
                    'underdog_probability': second_set_prob,
                    'quality': 'Good',
                    'confidence': 'Medium',
                    'ml_system_used': 'SECOND_SET_SIMULATION',
                    'prediction_type': 'SECOND_SET_UNDERDOG_SIMULATION',
                    'key_factors': [
                        f"🎯 Focus on second set specifically",
                        f"📊 Ranking gap: {rank_gap} positions",
                        f"⚡ Second set dynamics favor adaptation",
                        f"🎲 'Nothing to lose' mentality for trailing underdog"
                    ],
                    'underdog_scenario': {
                        'underdog': underdog,
                        'favorite': favorite,
                        'underdog_rank': p1_rank if underdog == player1 else p2_rank,
                        'favorite_rank': p2_rank if underdog == player1 else p1_rank,
                        'rank_gap': rank_gap,
                        'underdog_type': self._classify_underdog_type(rank_gap),
                        'base_probability': base_prob
                    }
                }
        
        return SecondSetUnderdogAnalyzer
    
    def get_backend_modifications(self) -> Dict[str, str]:
        """
        Get code modifications needed for tennis_backend.py
        """
        
        modifications = {
            "import_additions": """
# ADD TO IMPORTS SECTION (around line 40)
try:
    from second_set_integration import SecondSetIntegrator
    SECOND_SET_AVAILABLE = True
    print("✅ Second set prediction integration loaded")
except ImportError as e:
    print(f"⚠️ Second set integration not available: {e}")
    SECOND_SET_AVAILABLE = False
""",

            "global_variable_additions": """
# ADD TO GLOBAL VARIABLES (around line 130)
second_set_integrator = None
""",

            "initialization_additions": """
# ADD TO initialize_services() function (around line 360)
    # Initialize Second Set Integrator
    if SECOND_SET_AVAILABLE:
        try:
            global second_set_integrator
            second_set_integrator = SecondSetIntegrator()
            logger.info("✅ Second Set Integrator initialized")
        except Exception as e:
            logger.error(f"❌ Second set integration failed: {e}")
""",

            "underdog_analyzer_replacement": """
# REPLACE UnderdogAnalyzer class (around line 370) with:
class UnderdogAnalyzer:
    '''Modified UnderdogAnalyzer for SECOND SET prediction'''
    
    def __init__(self):
        global second_set_integrator
        
        if SECOND_SET_AVAILABLE and second_set_integrator:
            # Use second set specialized analyzer
            SecondSetAnalyzer = second_set_integrator.create_modified_underdog_analyzer()
            self.analyzer = SecondSetAnalyzer()
            self.second_set_mode = True
            logger.info("🎯 Using SECOND SET specialized prediction")
        else:
            # Fallback to original logic
            self.second_set_mode = False
            logger.info("⚠️ Using original underdog logic")
            # Keep original player_rankings dict here for fallback
    
    def get_player_ranking(self, player_name: str) -> int:
        '''Get player ranking'''
        if self.second_set_mode:
            return self.analyzer.get_player_ranking(player_name)
        else:
            # Original logic fallback
            # ... (keep existing logic)
            pass
    
    def identify_underdog_scenario(self, player1: str, player2: str) -> Dict:
        '''Identify underdog scenario - MODIFIED for second set'''
        if self.second_set_mode:
            # Delegate to second set analyzer
            result = self.analyzer.calculate_second_set_underdog_probability(
                player1, player2, "Tournament", "Hard"  
            )
            return result.get('underdog_scenario', {})
        else:
            # Original logic fallback
            # ... (keep existing logic)
            pass
    
    def calculate_underdog_probability(self, player1: str, player2: str, 
                                    tournament: str, surface: str) -> Dict:
        '''MAIN METHOD - Calculate underdog probability for SECOND SET'''
        
        if self.second_set_mode:
            # Use second set specialized prediction
            return self.analyzer.calculate_second_set_underdog_probability(
                player1, player2, tournament, surface
            )
        else:
            # Original logic fallback
            # ... (keep existing original logic)
            pass
""",

            "route_modifications": """
# MODIFY /api/matches route response (around line 1130) to add second set context:
        return jsonify({
            'success': True,
            'matches': formatted_matches,
            'count': len(formatted_matches),
            'source': matches_result.get('source', 'unknown'),
            'prediction_type': formatted_matches[0]['prediction_type'] if formatted_matches else 'UNKNOWN',
            'prediction_focus': 'SECOND_SET_UNDERDOG_WINS',  # NEW
            'market_advantage': 'More specific than general match betting',  # NEW
            'timestamp': datetime.now().isoformat()
        })
"""
        }
        
        return modifications
    
    def generate_integration_instructions(self) -> str:
        """
        Generate step-by-step integration instructions
        """
        
        instructions = """
🎾 SECOND SET PREDICTION - INTEGRATION INSTRUCTIONS
================================================================

STEP 1: FILE ADDITIONS
----------------------
✅ Copy these new files to your project:
- second_set_feature_engineering.py
- second_set_prediction_service.py  
- second_set_integration.py

STEP 2: BACKEND MODIFICATIONS
-----------------------------
Modify tennis_backend.py with the following changes:

A) ADD IMPORTS (around line 40):
```python
try:
    from second_set_integration import SecondSetIntegrator
    SECOND_SET_AVAILABLE = True
    print("✅ Second set prediction integration loaded")
except ImportError as e:
    print(f"⚠️ Second set integration not available: {e}")
    SECOND_SET_AVAILABLE = False
```

B) ADD GLOBAL VARIABLE (around line 130):
```python
second_set_integrator = None
```

C) MODIFY initialize_services() (around line 360):
```python
# Initialize Second Set Integrator
if SECOND_SET_AVAILABLE:
    try:
        global second_set_integrator
        second_set_integrator = SecondSetIntegrator()
        logger.info("✅ Second Set Integrator initialized")
    except Exception as e:
        logger.error(f"❌ Second set integration failed: {e}")
```

D) REPLACE UnderdogAnalyzer class (around line 370):
- Replace the entire UnderdogAnalyzer class with the modified version
- The new version uses second set specialized prediction when available
- Falls back to original logic if second set integration fails

STEP 3: MODEL RETRAINING (CRITICAL)
----------------------------------
🚨 IMPORTANT: Current models predict general match outcomes, not second set outcomes!

Required actions:
1. Collect historical tennis data with SET-BY-SET results
2. Create training dataset with target: "underdog_won_second_set" 
3. Use SecondSetFeatureEngineer to generate features
4. Retrain all 5 models with second set target
5. Save new models in tennis_models/ directory

Minimum data requirements:
- 15,000+ matches with set-by-set results
- Various tournament levels and surfaces  
- Focus on matches where underdog lost first set
- Balance underdog wins/losses in second set

STEP 4: TESTING
---------------
1. Start the server: python tennis_backend.py
2. Test endpoint: GET /api/matches
3. Look for prediction_focus: "SECOND_SET_UNDERDOG_WINS"
4. Verify key factors mention second set dynamics

STEP 5: LIVE DATA INTEGRATION
-----------------------------
For production, replace simulate_first_set_data() with real live match data:
- Connect to live tennis data feed
- Extract first set results as they complete
- Feed real first set stats to second set predictor

EXPECTED OUTCOMES:
=================
✅ Predictions focus on second set specifically
✅ Better accuracy for underdog set wins vs match wins  
✅ Key factors highlight second set dynamics
✅ Market advantage over general match betting
✅ "Nothing to lose" scenarios identified
✅ Momentum and adaptation factors considered

STRATEGIC ADVANTAGES:
====================
🎯 More specific than "any set" predictions
📊 Captures tennis-specific momentum shifts  
⚡ Identifies underdog opportunities in second set
🧠 Considers psychological factors and adaptation
💰 Potential higher accuracy for specialized betting markets
🔄 Better handles comeback scenarios and player patterns

NEXT STEPS AFTER INTEGRATION:
=============================
1. Monitor prediction accuracy on live matches
2. Collect real second set outcome data for validation
3. Fine-tune model weights for second set dynamics
4. Consider third set prediction as next enhancement
5. Analyze profitability vs general match predictions
"""
        
        return instructions


# Generate integration guide
if __name__ == "__main__":
    print("🎾 SECOND SET INTEGRATION SETUP")
    print("=" * 60)
    
    integrator = SecondSetIntegrator()
    
    # Test the modified analyzer
    SecondSetAnalyzer = integrator.create_modified_underdog_analyzer()
    analyzer = SecondSetAnalyzer()
    
    print("🧪 Testing second set analyzer...")
    
    result = analyzer.calculate_second_set_underdog_probability(
        "Flavio Cobolli", "Novak Djokovic", "US Open", "Hard"
    )
    
    print(f"\n🎯 Test Result:")
    print(f"Underdog: {result['underdog_scenario']['underdog']}")  
    print(f"Second set probability: {result['underdog_probability']:.1%}")
    print(f"Prediction type: {result['prediction_type']}")
    print(f"Focus: {result.get('second_set_specifics', {}).get('focus', 'N/A')}")
    
    # Generate integration instructions
    instructions = integrator.generate_integration_instructions()
    
    # Save instructions to file
    with open('/home/apps/Tennis_one_set/SECOND_SET_INTEGRATION_GUIDE.md', 'w') as f:
        f.write(instructions)
    
    print(f"\n✅ Integration ready!")
    print(f"📚 Complete guide saved: SECOND_SET_INTEGRATION_GUIDE.md")
    print(f"🔧 Modifications ready for tennis_backend.py")
    print(f"⚠️ Remember: Models need retraining for second set target!")