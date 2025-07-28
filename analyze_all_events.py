#!/usr/bin/env python3
"""
Analyze all events to understand what data is available
"""

import sys
import json
from rapidapi_tennis_client import RapidAPITennisClient
from collections import Counter

def analyze_events():
    """Analyze all events data structure"""
    try:
        print("🔍 Analyzing all RapidAPI events...")
        client = RapidAPITennisClient()
        
        # Get all events
        all_events = client.get_all_events()
        if not all_events:
            print("❌ No events found")
            return
        
        print(f"📊 Total events: {len(all_events)}")
        
        # Analyze status types
        statuses = [event.get('status', {}).get('type', 'unknown') for event in all_events]
        status_counts = Counter(statuses)
        print(f"\n📈 Status distribution:")
        for status, count in status_counts.most_common():
            print(f"  {status}: {count}")
        
        # Look for upcoming matches
        upcoming_statuses = ['scheduled', 'notstarted', 'postponed', 'delayed', 'upcoming']
        upcoming_matches = []
        
        for event in all_events:
            status_type = event.get('status', {}).get('type', '')
            if status_type in upcoming_statuses:
                upcoming_matches.append(event)
        
        print(f"\n🔮 Potential upcoming matches: {len(upcoming_matches)}")
        
        # Show all unique field keys
        all_keys = set()
        for event in all_events[:10]:  # Sample first 10
            def extract_keys(obj, prefix=''):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        full_key = f"{prefix}.{key}" if prefix else key
                        all_keys.add(full_key)
                        if isinstance(value, (dict, list)) and len(str(value)) < 200:
                            extract_keys(value, full_key)
            extract_keys(event)
        
        print(f"\n🔑 Available data fields:")
        for key in sorted(all_keys):
            print(f"  {key}")
        
        # Show sample event structure
        if all_events:
            print(f"\n📝 Sample event structure:")
            sample = all_events[0]
            print(json.dumps(sample, indent=2, default=str)[:1000] + "...")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_events()