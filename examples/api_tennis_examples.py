#!/usr/bin/env python3
"""
API-Tennis.com Integration Examples
Comprehensive examples showing how to use the API-Tennis integration
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_tennis_integration import (
    APITennisClient, TennisMatch, Tournament, TennisPlayer,
    get_api_tennis_client, initialize_api_tennis_client
)
from api_tennis_data_collector import (
    APITennisDataCollector, EnhancedAPITennisCollector,
    get_api_tennis_data_collector, get_enhanced_api_tennis_collector
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def example_basic_client_usage():
    """Example 1: Basic API-Tennis client usage"""
    print("=" * 60)
    print("EXAMPLE 1: Basic API-Tennis Client Usage")
    print("=" * 60)
    
    # Initialize client (will use API_TENNIS_KEY environment variable)
    client = APITennisClient()
    
    print(f"Client Status: {client.get_client_status()}")
    
    if not client.api_key:
        print("⚠️  No API key configured. Set API_TENNIS_KEY environment variable.")
        print("Example: export API_TENNIS_KEY='your_api_key_here'")
        return
    
    try:
        # Get event types
        print("\n🎾 Getting Event Types...")
        event_types = client.get_event_types()
        print(f"Found {len(event_types)} event types")
        if event_types:
            for i, event in enumerate(event_types[:3], 1):
                print(f"  {i}. {event}")
        
        # Get tournaments
        print("\n🏆 Getting Tournaments...")
        tournaments = client.get_tournaments()
        print(f"Found {len(tournaments)} tournaments")
        if tournaments:
            for i, tournament in enumerate(tournaments[:5], 1):
                print(f"  {i}. {tournament.name} ({tournament.location}) - {tournament.category}")
        
        # Get today's matches
        print("\n📅 Getting Today's Matches...")
        today_matches = client.get_today_matches()
        print(f"Found {len(today_matches)} matches today")
        if today_matches:
            for i, match in enumerate(today_matches[:3], 1):
                player1_name = match.player1.name if match.player1 else "Unknown"
                player2_name = match.player2.name if match.player2 else "Unknown"
                print(f"  {i}. {player1_name} vs {player2_name} - {match.tournament_name}")
        
        # Get live matches
        print("\n🔴 Getting Live Matches...")
        live_matches = client.get_live_matches()
        print(f"Found {len(live_matches)} live matches")
        if live_matches:
            for i, match in enumerate(live_matches[:3], 1):
                player1_name = match.player1.name if match.player1 else "Unknown"
                player2_name = match.player2.name if match.player2 else "Unknown"
                print(f"  {i}. {player1_name} vs {player2_name} - {match.score}")
        
    except Exception as e:
        print(f"❌ Error in basic client usage: {e}")


def example_advanced_client_features():
    """Example 2: Advanced client features"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Advanced Client Features")
    print("=" * 60)
    
    client = get_api_tennis_client()
    
    if not client.api_key:
        print("⚠️  No API key configured. Skipping advanced examples.")
        return
    
    try:
        # Get upcoming matches with date range
        print("\n📆 Getting Upcoming Matches (Next 3 Days)...")
        upcoming_matches = client.get_upcoming_matches(days_ahead=3)
        print(f"Found {len(upcoming_matches)} upcoming matches")
        
        if upcoming_matches:
            print("\n🎾 Sample upcoming matches:")
            for i, match in enumerate(upcoming_matches[:5], 1):
                player1_name = match.player1.name if match.player1 else "Unknown"
                player2_name = match.player2.name if match.player2 else "Unknown"
                start_time = match.start_time.strftime('%Y-%m-%d %H:%M') if match.start_time else 'TBD'
                print(f"  {i}. {player1_name} vs {player2_name}")
                print(f"     🏆 {match.tournament_name} | ⏰ {start_time} | 🎯 {match.round}")
        
        # Search for specific player matches
        print("\n🔍 Searching for Player Matches...")
        test_players = ["Sinner", "Alcaraz", "Djokovic", "Swiatek"]
        
        for player in test_players:
            player_matches = client.search_matches_by_player(player, days_ahead=14)
            if player_matches:
                print(f"  📍 Found {len(player_matches)} matches for '{player}'")
                match = player_matches[0]
                player1_name = match.player1.name if match.player1 else "Unknown"
                player2_name = match.player2.name if match.player2 else "Unknown"
                print(f"     Next: {player1_name} vs {player2_name} ({match.tournament_name})")
                break
        else:
            print("  📍 No matches found for test players")
        
        # Demonstrate caching
        print("\n💾 Testing Cache Performance...")
        import time
        
        # First request (will hit API)
        start_time = time.time()
        tournaments1 = client.get_tournaments()
        first_request_time = time.time() - start_time
        
        # Second request (should use cache)
        start_time = time.time()
        tournaments2 = client.get_tournaments()
        second_request_time = time.time() - start_time
        
        print(f"  First request: {first_request_time:.3f}s ({len(tournaments1)} tournaments)")
        print(f"  Cached request: {second_request_time:.3f}s ({len(tournaments2)} tournaments)")
        print(f"  Cache speedup: {first_request_time/second_request_time:.1f}x faster")
        
    except Exception as e:
        print(f"❌ Error in advanced features: {e}")


def example_data_collector_usage():
    """Example 3: Using the API-Tennis data collector"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: API-Tennis Data Collector Usage")
    print("=" * 60)
    
    # Get the data collector
    collector = get_api_tennis_data_collector()
    
    print(f"Data Collector Available: {collector.is_available()}")
    
    if not collector.is_available():
        print("⚠️  Data collector not available (missing API key)")
        return
    
    try:
        # Get integration status
        print("\n📊 Integration Status:")
        status = collector.get_integration_status()
        print(f"  API Key Configured: {status['api_key_configured']}")
        print(f"  Available: {status['available']}")
        if 'client_status' in status and status['client_status']:
            client_status = status['client_status']
            print(f"  Caching: {client_status.get('caching_enabled', False)}")
            print(f"  Timeout: {client_status.get('timeout', 'N/A')}s")
        
        # Get current matches in Universal Collector format
        print("\n🎾 Getting Current Matches (Universal Format):")
        current_matches = collector.get_current_matches()
        print(f"Found {len(current_matches)} current matches")
        
        if current_matches:
            print("\n📋 Sample matches in Universal format:")
            for i, match in enumerate(current_matches[:3], 1):
                print(f"  {i}. {match['player1']} vs {match['player2']}")
                print(f"     🏆 {match['tournament']} | 📍 {match['location']}")
                print(f"     🎯 {match['level']} | 🏟️ {match['surface']} | ⭐ Quality: {match['quality_score']}")
                print(f"     📊 Source: {match['data_source']}")
        
        # Get upcoming matches
        print("\n📅 Getting Upcoming Matches:")
        upcoming_matches = collector.get_upcoming_matches(days_ahead=5)
        print(f"Found {len(upcoming_matches)} upcoming matches")
        
        # Get tournaments
        print("\n🏆 Getting Professional Tournaments:")
        tournaments = collector.get_tournaments()
        print(f"Found {len(tournaments)} professional tournaments")
        
        if tournaments:
            print("\n📋 Sample tournaments:")
            for i, tournament in enumerate(tournaments[:3], 1):
                print(f"  {i}. {tournament['name']} ({tournament['location']})")
                print(f"     Level: {tournament['level']} | Surface: {tournament['surface']}")
        
    except Exception as e:
        print(f"❌ Error in data collector usage: {e}")


def example_enhanced_collector_integration():
    """Example 4: Enhanced collector with multiple data sources"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Enhanced Collector Integration")
    print("=" * 60)
    
    # Get enhanced collector
    enhanced_collector = get_enhanced_api_tennis_collector()
    
    # Check status of all data sources
    print("📊 Data Sources Status:")
    status = enhanced_collector.get_status()
    print(f"  API-Tennis Available: {status['api_tennis']['available']}")
    print(f"  Universal Collector Available: {status['universal_collector']}")
    print(f"  Total Sources Available: {status['total_sources_available']}")
    
    try:
        # Get comprehensive match data from all sources
        print("\n🌍 Getting Comprehensive Match Data from All Sources:")
        comprehensive_matches = enhanced_collector.get_comprehensive_match_data(days_ahead=3)
        print(f"Found {len(comprehensive_matches)} total matches after deduplication")
        
        if comprehensive_matches:
            # Analyze data sources
            source_counts = {}
            quality_scores = []
            
            for match in comprehensive_matches:
                source = match.get('data_source', 'Unknown')
                source_counts[source] = source_counts.get(source, 0) + 1
                quality_scores.append(match.get('quality_score', 0))
            
            print("\n📈 Data Source Distribution:")
            for source, count in source_counts.items():
                print(f"  {source}: {count} matches")
            
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
                print(f"\n⭐ Average Quality Score: {avg_quality:.1f}")
            
            # Show sample high-quality matches
            high_quality_matches = [m for m in comprehensive_matches if m.get('quality_score', 0) >= 90]
            print(f"\n🔥 High Quality Matches (≥90 quality score): {len(high_quality_matches)}")
            
            for i, match in enumerate(high_quality_matches[:3], 1):
                print(f"  {i}. {match['player1']} vs {match['player2']}")
                print(f"     🏆 {match['tournament']} | ⭐ {match['quality_score']}")
                print(f"     📊 {match['data_source']} | 🎯 {match['level']}")
        
    except Exception as e:
        print(f"❌ Error in enhanced collector: {e}")


def example_odds_and_betting_integration():
    """Example 5: Odds and betting data integration"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Odds and Betting Integration")
    print("=" * 60)
    
    client = get_api_tennis_client()
    collector = get_api_tennis_data_collector()
    
    if not client.api_key:
        print("⚠️  No API key configured. Skipping odds examples.")
        return
    
    try:
        # Get matches with odds data
        print("💰 Getting Matches with Odds Data:")
        matches = collector.get_current_matches()
        
        matches_with_odds = [m for m in matches if m.get('player1_odds') or m.get('player2_odds')]
        print(f"Found {len(matches_with_odds)} matches with odds data")
        
        if matches_with_odds:
            print("\n💸 Sample matches with odds:")
            for i, match in enumerate(matches_with_odds[:3], 1):
                print(f"  {i}. {match['player1']} vs {match['player2']}")
                if match.get('player1_odds') and match.get('player2_odds'):
                    p1_odds = match['player1_odds']
                    p2_odds = match['player2_odds']
                    print(f"     💰 Odds: {p1_odds} vs {p2_odds}")
                    
                    # Calculate implied probabilities
                    p1_prob = 1 / p1_odds if p1_odds > 0 else 0
                    p2_prob = 1 / p2_odds if p2_odds > 0 else 0
                    total_prob = p1_prob + p2_prob
                    
                    if total_prob > 0:
                        p1_pct = (p1_prob / total_prob) * 100
                        p2_pct = (p2_prob / total_prob) * 100
                        print(f"     📊 Implied: {p1_pct:.1f}% vs {p2_pct:.1f}%")
        
        # Try to get odds for specific matches
        print("\n🎯 Getting Detailed Odds for Specific Matches:")
        today_matches = client.get_today_matches()
        
        for match in today_matches[:2]:  # Try first 2 matches
            if match.id:
                try:
                    odds_data = collector.get_match_odds(match.id)
                    if odds_data.get('bookmakers'):
                        player1_name = match.player1.name if match.player1 else "Unknown"
                        player2_name = match.player2.name if match.player2 else "Unknown"
                        print(f"  📈 {player1_name} vs {player2_name}:")
                        print(f"     Bookmakers: {len(odds_data['bookmakers'])}")
                        for bookmaker in odds_data['bookmakers'][:3]:  # Show first 3
                            print(f"     {bookmaker['bookmaker']}: {bookmaker.get('player1_odds', 'N/A')} vs {bookmaker.get('player2_odds', 'N/A')}")
                except Exception as e:
                    print(f"  ❌ Failed to get odds for match {match.id}: {e}")
        
    except Exception as e:
        print(f"❌ Error in odds integration: {e}")


def example_ml_integration_preparation():
    """Example 6: Preparing data for ML integration"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: ML Integration Preparation")
    print("=" * 60)
    
    enhanced_collector = get_enhanced_api_tennis_collector()
    
    try:
        # Get ML-ready matches from enhanced collector
        print("🤖 Getting ML-Ready Matches:")
        if hasattr(enhanced_collector.universal_collector, 'get_ml_ready_matches'):
            ml_matches = enhanced_collector.universal_collector.get_ml_ready_matches(min_quality_score=85)
            print(f"Found {len(ml_matches)} ML-ready matches")
            
            if ml_matches:
                print("\n📊 Sample ML Features:")
                sample_match = ml_matches[0]
                ml_features = sample_match.get('ml_features', {})
                
                if ml_features:
                    feature_categories = {
                        'Player Rankings': ['player1_ranking', 'player2_ranking', 'ranking_difference'],
                        'Odds Data': ['player1_odds', 'player2_odds', 'implied_probability_p1', 'implied_probability_p2'],
                        'Tournament Info': ['surface_encoded', 'tournament_level_encoded', 'is_grand_slam'],
                        'Temporal': ['day_of_year', 'month', 'is_weekend']
                    }
                    
                    for category, features in feature_categories.items():
                        print(f"\n  {category}:")
                        for feature in features:
                            if feature in ml_features:
                                print(f"    {feature}: {ml_features[feature]}")
        else:
            print("ML-ready matches method not available in this configuration")
        
        # Analyze data quality and coverage
        print("\n📈 Data Quality Analysis:")
        matches = enhanced_collector.get_comprehensive_match_data(days_ahead=7)
        
        if matches:
            # Quality score distribution
            quality_scores = [m.get('quality_score', 0) for m in matches]
            avg_quality = sum(quality_scores) / len(quality_scores)
            high_quality = len([q for q in quality_scores if q >= 90])
            
            print(f"  Total Matches: {len(matches)}")
            print(f"  Average Quality: {avg_quality:.1f}")
            print(f"  High Quality (≥90): {high_quality} ({high_quality/len(matches)*100:.1f}%)")
            
            # Data completeness
            with_rankings = len([m for m in matches if m.get('player1_ranking') and m.get('player2_ranking')])
            with_odds = len([m for m in matches if m.get('player1_odds') and m.get('player2_odds')])
            
            print(f"  With Rankings: {with_rankings} ({with_rankings/len(matches)*100:.1f}%)")
            print(f"  With Odds: {with_odds} ({with_odds/len(matches)*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error in ML preparation: {e}")


def run_all_examples():
    """Run all examples"""
    print("🎾 API-Tennis.com Integration Examples")
    print("Starting comprehensive demonstration...")
    
    try:
        example_basic_client_usage()
        example_advanced_client_features()
        example_data_collector_usage()
        example_enhanced_collector_integration()
        example_odds_and_betting_integration()
        example_ml_integration_preparation()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        
        # Final recommendations
        print("\n📋 Next Steps:")
        print("1. Set API_TENNIS_KEY environment variable with your API key")
        print("2. Test the integration with your real data")
        print("3. Configure caching and rate limiting as needed")
        print("4. Integrate with your existing ML pipeline")
        print("5. Set up monitoring and error handling")
        
    except KeyboardInterrupt:
        print("\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set up environment for examples
    print("🏁 Setting up API-Tennis.com Integration Examples")
    
    # Check for API key
    api_key = os.getenv('API_TENNIS_KEY')
    if not api_key:
        print("⚠️  API_TENNIS_KEY environment variable not set")
        print("Examples will run but API calls will fail")
        print("Set your API key with: export API_TENNIS_KEY='your_key_here'")
        print()
    else:
        print(f"✅ API key configured (length: {len(api_key)} characters)")
        print()
    
    # Run all examples
    run_all_examples()