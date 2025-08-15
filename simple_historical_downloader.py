#!/usr/bin/env python3
"""
Simple Historical Data Downloader
Downloads 2 years of tennis match data for ranks 10-300
"""

import sys
import os
import requests
import json
import sqlite3
from datetime import datetime, timedelta, date
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_database():
    """Create simple SQLite database for historical data"""
    conn = sqlite3.connect('tennis_historical_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS historical_matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            player1 TEXT,
            player2 TEXT,
            player1_rank INTEGER,
            player2_rank INTEGER,
            tournament TEXT,
            surface TEXT,
            round TEXT,
            score TEXT,
            winner TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database setup complete")

def download_historical_data():
    """Download historical data using simple API calls"""
    
    # Check if API key is available
    api_key = os.getenv('API_TENNIS_KEY')
    if not api_key:
        logger.error("API_TENNIS_KEY not found in environment variables")
        return
    
    base_url = "https://api-tennis.com/v2"
    headers = {"X-RapidAPI-Key": api_key}
    
    # Setup database
    setup_database()
    conn = sqlite3.connect('tennis_historical_data.db')
    cursor = conn.cursor()
    
    # Calculate date range (2 years back)
    end_date = date.today()
    start_date = end_date - timedelta(days=730)  # 2 years
    
    logger.info(f"Downloading data from {start_date} to {end_date}")
    
    total_matches = 0
    current_date = start_date
    
    while current_date <= end_date:
        try:
            # Format date for API
            date_str = current_date.strftime("%Y-%m-%d")
            
            # API call to get matches for this date
            url = f"{base_url}/matches"
            params = {
                "date": date_str,
                "live": "false"
            }
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                matches = data.get('data', [])
                
                for match in matches:
                    try:
                        # Extract match data
                        player1 = match.get('home_player', {}).get('full_name', 'Unknown')
                        player2 = match.get('away_player', {}).get('full_name', 'Unknown')
                        player1_rank = match.get('home_player', {}).get('current_rank', 999)
                        player2_rank = match.get('away_player', {}).get('current_rank', 999)
                        
                        # Filter for ranks 10-300
                        if (10 <= player1_rank <= 300) or (10 <= player2_rank <= 300):
                            tournament = match.get('tournament', {}).get('name', 'Unknown')
                            surface = match.get('surface', 'Unknown')
                            round_name = match.get('round', 'Unknown')
                            score = match.get('score', '')
                            winner = match.get('winner', '')
                            
                            # Insert into database
                            cursor.execute('''
                                INSERT INTO historical_matches 
                                (date, player1, player2, player1_rank, player2_rank, 
                                 tournament, surface, round, score, winner)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (date_str, player1, player2, player1_rank, player2_rank,
                                  tournament, surface, round_name, score, winner))
                            
                            total_matches += 1
                    
                    except Exception as e:
                        logger.warning(f"Error processing match: {e}")
                        continue
                
                logger.info(f"Processed {date_str}: {len(matches)} matches, {total_matches} total saved")
                
            else:
                logger.warning(f"API request failed for {date_str}: {response.status_code}")
            
            # Rate limiting - wait between requests
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error processing {current_date}: {e}")
        
        # Move to next day
        current_date += timedelta(days=1)
    
    # Commit and close
    conn.commit()
    conn.close()
    
    logger.info(f"Historical data download complete! Total matches saved: {total_matches}")
    return total_matches

if __name__ == "__main__":
    # Check if API key is available
    if not os.getenv('API_TENNIS_KEY'):
        print("❌ API_TENNIS_KEY environment variable not set")
        print("Please set your API-Tennis.com API key:")
        print("export API_TENNIS_KEY='your_api_key_here'")
        sys.exit(1)
    
    print("🎾 Starting Historical Tennis Data Download")
    print("This will download 2 years of match data for players ranked 10-300")
    print("Estimated time: 30-60 minutes depending on API rate limits")
    
    total = download_historical_data()
    print(f"✅ Download complete! {total} matches saved to tennis_historical_data.db")