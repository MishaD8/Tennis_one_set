#!/usr/bin/env python3
"""
Test the improved underdog notification format
Shows how the notification will clearly indicate the underdog
"""

import sys
import os
sys.path.append('/home/apps/Tennis_one_set/src/utils')

from telegram_notification_system import TelegramNotificationSystem

def test_underdog_notification_format():
    """Test the new underdog notification format"""
    
    print("🎾 Testing Improved Underdog Notification Format")
    print("=" * 60)
    
    # Create test prediction data
    test_prediction = {
        'match_context': {
            'player1': 'David Goffin',      # Underdog
            'player2': 'Novak Djokovic',   # Favorite  
            'tournament': 'ATP Masters Monte Carlo',
            'surface': 'Clay'
        },
        'underdog_second_set_probability': 0.67,  # 67% chance for underdog
        'confidence': 'High',
        'underdog_player': 'player1',  # Goffin is the underdog
        'strategic_insights': [
            'Goffin has won 85% of clay court second sets this season',
            'Djokovic struggling with endurance in long clay matches',
            'Weather conditions favor defensive baseline play'
        ],
        'success': True
    }
    
    # Initialize notification system
    notification_system = TelegramNotificationSystem()
    
    # Format the message
    message = notification_system._format_underdog_message(test_prediction)
    
    print("📱 NEW IMPROVED NOTIFICATION FORMAT:")
    print("=" * 60)
    print(message.replace('<b>', '**').replace('</b>', '**').replace('<i>', '*').replace('</i>', '*'))
    print("=" * 60)
    
    print("\n✅ IMPROVEMENTS MADE:")
    print("🎯 CLEAR underdog identification with 'UNDERDOG:' label")
    print("🆚 Visual separation between underdog and favorite")
    print("📊 Explicit 'UNDERDOG Second Set Win Probability'")
    print("🎯 Clear prediction statement: 'PREDICTION: [Player] to win the 2nd set'")
    print("💰 Betting recommendation: 'Bet on: [Player] 2nd set winner'")
    
    print("\n📋 WHAT YOU'LL SEE NOW:")
    print("• 'UNDERDOG: David Goffin (#15) 🆚 Novak Djokovic (#1)'")
    print("• 'UNDERDOG Second Set Win Probability: 67.0%'")
    print("• 'PREDICTION: David Goffin to win the 2nd set'")
    print("• 'Bet on: David Goffin 2nd set winner'")
    
    print("\n🎾 No more confusion about who the underdog is!")

if __name__ == "__main__":
    test_underdog_notification_format()#!/usr/bin/env python3
"""
Test the improved underdog notification format
Shows how the notification will clearly indicate the underdog
"""

import sys
import os
sys.path.append('/home/apps/Tennis_one_set/src/utils')

from telegram_notification_system import TelegramNotificationSystem

def test_underdog_notification_format():
    """Test the new underdog notification format"""
    
    print("🎾 Testing Improved Underdog Notification Format")
    print("=" * 60)
    
    # Create test prediction data
    test_prediction = {
        'match_context': {
            'player1': 'David Goffin',      # Underdog
            'player2': 'Novak Djokovic',   # Favorite  
            'tournament': 'ATP Masters Monte Carlo',
            'surface': 'Clay'
        },
        'underdog_second_set_probability': 0.67,  # 67% chance for underdog
        'confidence': 'High',
        'underdog_player': 'player1',  # Goffin is the underdog
        'strategic_insights': [
            'Goffin has won 85% of clay court second sets this season',
            'Djokovic struggling with endurance in long clay matches',
            'Weather conditions favor defensive baseline play'
        ],
        'success': True
    }
    
    # Initialize notification system
    notification_system = TelegramNotificationSystem()
    
    # Format the message
    message = notification_system._format_underdog_message(test_prediction)
    
    print("📱 NEW IMPROVED NOTIFICATION FORMAT:")
    print("=" * 60)
    print(message.replace('<b>', '**').replace('</b>', '**').replace('<i>', '*').replace('</i>', '*'))
    print("=" * 60)
    
    print("\n✅ IMPROVEMENTS MADE:")
    print("🎯 CLEAR underdog identification with 'UNDERDOG:' label")
    print("🆚 Visual separation between underdog and favorite")
    print("📊 Explicit 'UNDERDOG Second Set Win Probability'")
    print("🎯 Clear prediction statement: 'PREDICTION: [Player] to win the 2nd set'")
    print("💰 Betting recommendation: 'Bet on: [Player] 2nd set winner'")
    
    print("\n📋 WHAT YOU'LL SEE NOW:")
    print("• 'UNDERDOG: David Goffin (#15) 🆚 Novak Djokovic (#1)'")
    print("• 'UNDERDOG Second Set Win Probability: 67.0%'")
    print("• 'PREDICTION: David Goffin to win the 2nd set'")
    print("• 'Bet on: David Goffin 2nd set winner'")
    
    print("\n🎾 No more confusion about who the underdog is!")

if __name__ == "__main__":
    test_underdog_notification_format()