#!/usr/bin/env python3
"""
Test the new concise Telegram notification format
"""

import sys
import os
sys.path.append('src')

def test_concise_format():
    """Test the new concise Telegram format"""
    
    print("🎾 NEW CONCISE TELEGRAM NOTIFICATION FORMAT TEST")
    print("=" * 60)
    
    # Simulate the exact format that will be sent
    test_scenarios = [
        {
            'name': 'Strong Underdog Scenario',
            'tournament': 'ATP Masters Monte Carlo',
            'underdog': 'Ben Shelton',
            'underdog_rank': 14,
            'favorite': 'Carlos Alcaraz', 
            'favorite_rank': 2,
            'surface': 'Clay',
            'probability': 73.2,
            'confidence': 'High'
        },
        {
            'name': 'Medium Confidence Scenario',
            'tournament': 'WTA 1000 Miami Open',
            'underdog': 'Emma Raducanu',
            'underdog_rank': 45,
            'favorite': 'Iga Swiatek',
            'favorite_rank': 1,
            'surface': 'Hard',
            'probability': 61.8,
            'confidence': 'Medium'
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📱 SCENARIO {i}: {scenario['name']}")
        print("─" * 40)
        
        # Format message exactly as it will appear in Telegram
        message = f"""🏆 <b>{scenario['tournament']}</b>
    <b>{scenario['underdog']}</b> (#{scenario['underdog_rank']}) vs <b>{scenario['favorite']}</b> (#{scenario['favorite_rank']})
🏟️ Surface: {scenario['surface']}
📊 UNDERDOG Second Set Win Probability: <b>{scenario['probability']:.1f}%</b>
🔮 Confidence: <b>{scenario['confidence']}</b>
🎯 PREDICTION: <b>{scenario['underdog']} to win 2nd set</b>"""
        
        # Show both raw (with HTML tags) and clean versions
        print("Raw message (with HTML formatting):")
        print(message)
        print("\nClean message (as it appears in Telegram):")
        clean_message = message.replace('<b>', '**').replace('</b>', '**').replace('<i>', '*').replace('</i>', '*')
        print(clean_message)
        
        # Count characters and lines
        clean_lines = clean_message.split('\n')
        print(f"\n📊 Format Analysis:")
        print(f"   Total lines: {len(clean_lines)}")
        print(f"   Total characters: {len(clean_message)}")
        print(f"   Max line length: {max(len(line) for line in clean_lines)}")
    
    print(f"\n✅ FORMAT BENEFITS:")
    print(f"   🎯 Ultra-concise: 6 lines only")
    print(f"   📱 Mobile-friendly: <80 characters per line")  
    print(f"   ⚡ Quick to read: Essential info only")
    print(f"   🔍 Clear underdog identification")
    print(f"   📊 Key metrics highlighted")
    print(f"   🎲 Direct betting recommendation")

if __name__ == "__main__":
    test_concise_format()