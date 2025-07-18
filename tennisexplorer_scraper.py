#!/usr/bin/env python3
"""
🎾 TennisExplorer.com Web Scraper
Collects tennis data from tennisexplorer.com with login functionality
"""

import requests
from bs4 import BeautifulSoup
import time
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
from typing import Dict, List, Optional, Tuple
import re
import logging
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TennisMatch:
    """Tennis match data structure"""
    player1: str
    player2: str
    tournament: str
    surface: str
    round_info: str
    start_time: Optional[str]
    odds_player1: Optional[float] = None
    odds_player2: Optional[float] = None
    result: Optional[str] = None
    score: Optional[str] = None

class TennisExplorerScraper:
    """Web scraper for tennisexplorer.com"""
    
    def __init__(self):
        self.base_url = "https://www.tennisexplorer.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        self.username = os.getenv('TENNISEXPLORER_USERNAME')
        self.password = os.getenv('TENNISEXPLORER_PASSWORD')
        self.logged_in = False
        
    def login(self) -> bool:
        """Login to tennisexplorer.com"""
        if not self.username or not self.password:
            logger.warning("TennisExplorer credentials not found in environment variables")
            return False
            
        try:
            # Check main page first to find login form or links
            main_page = self.session.get(self.base_url, timeout=10)
            if main_page.status_code != 200:
                logger.error(f"Failed to access main page: {main_page.status_code}")
                return False
            
            main_soup = BeautifulSoup(main_page.content, 'html.parser')
            
            # Look for login form on main page first
            login_form = main_soup.find('form', {'action': re.compile(r'.*login.*', re.I)})
            if not login_form:
                # Look for any form with username/password fields
                forms = main_soup.find_all('form')
                for form in forms:
                    if (form.find('input', {'name': re.compile(r'user|email|login', re.I)}) and 
                        form.find('input', {'type': 'password'})):
                        login_form = form
                        break
            
            if login_form:
                # Use main page for login
                login_page_url = self.base_url
                login_page = main_page
                soup = main_soup
            else:
                # Try dedicated login page
                login_page_url = f"{self.base_url}/login/"
                login_page = self.session.get(login_page_url, timeout=10)
            
                if login_page.status_code != 200:
                    logger.error(f"Failed to access login page: {login_page.status_code}")
                    return False
                    
                soup = BeautifulSoup(login_page.content, 'html.parser')
                
                # Find login form and any hidden fields
                login_form = soup.find('form', {'action': re.compile(r'.*login.*', re.I)})
                if not login_form:
                    login_form = soup.find('form')  # Fallback to first form
                
            if not login_form:
                logger.error("Could not find login form")
                return False
                
            # Prepare login data
            login_data = {
                'username': self.username,
                'password': self.password
            }
            
            # Add any hidden fields (CSRF tokens, etc.)
            for hidden_input in login_form.find_all('input', {'type': 'hidden'}):
                name = hidden_input.get('name')
                value = hidden_input.get('value', '')
                if name:
                    login_data[name] = value
                    
            # Find the actual form action URL
            form_action = login_form.get('action', '/login/')
            login_url = urljoin(self.base_url, form_action)
            
            # Perform login
            logger.info("Attempting to login to TennisExplorer...")
            login_response = self.session.post(login_url, data=login_data, timeout=10)
            
            # Check if login was successful
            if login_response.status_code == 200:
                # Look for indicators of successful login
                content = login_response.text.lower()
                if any(indicator in content for indicator in ['logout', 'profile', 'account', 'dashboard']):
                    logger.info("✅ Successfully logged in to TennisExplorer")
                    self.logged_in = True
                    return True
                else:
                    logger.error("Login appeared to fail - no success indicators found")
                    return False
            else:
                logger.error(f"Login request failed with status: {login_response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Login failed with exception: {e}")
            return False
            
    def get_current_matches(self, days_ahead: int = 2) -> List[TennisMatch]:
        """Get current and upcoming matches"""
        matches = []
        
        try:
            # Get today's matches
            today_url = f"{self.base_url}/next/"
            response = self.session.get(today_url, timeout=10)
            
            if response.status_code == 200:
                matches.extend(self._parse_matches_page(response.content, "Today"))
                
            # Get upcoming matches for the next few days
            for day in range(1, days_ahead + 1):
                date = datetime.now() + timedelta(days=day)
                date_str = date.strftime("%Y-%m-%d")
                
                matches_url = f"{self.base_url}/next/{date_str}/"
                response = self.session.get(matches_url, timeout=10)
                
                if response.status_code == 200:
                    day_matches = self._parse_matches_page(response.content, date_str)
                    matches.extend(day_matches)
                    
                # Rate limiting
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error getting current matches: {e}")
            
        logger.info(f"Found {len(matches)} matches")
        return matches
        
    def _parse_matches_page(self, html_content: bytes, date_info: str) -> List[TennisMatch]:
        """Parse matches from a page"""
        matches = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for match containers (this may need adjustment based on actual site structure)
            match_containers = soup.find_all(['tr', 'div'], class_=re.compile(r'.*match.*|.*game.*|.*fixture.*', re.I))
            
            if not match_containers:
                # Fallback: look for tables with tennis data
                tables = soup.find_all('table')
                for table in tables:
                    rows = table.find_all('tr')
                    match_containers.extend(rows)
                    
            for container in match_containers:
                match = self._extract_match_data(container, date_info)
                if match:
                    matches.append(match)
                    
        except Exception as e:
            logger.error(f"Error parsing matches page: {e}")
            
        return matches
        
    def _extract_match_data(self, container, date_info: str) -> Optional[TennisMatch]:
        """Extract match data from a container element"""
        try:
            # Extract text content
            text_content = container.get_text(strip=True)
            
            # Look for player names (common patterns)
            player_links = container.find_all('a', href=re.compile(r'.*player.*', re.I))
            players = [link.get_text(strip=True) for link in player_links if link.get_text(strip=True)]
            
            # If we don't have exactly 2 players, try alternative extraction
            if len(players) != 2:
                # Look for patterns like "Player1 - Player2" or "Player1 vs Player2"
                vs_pattern = re.search(r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s*(?:-|vs\.?)\s*([A-Z][a-z]+\s+[A-Z][a-z]+)', text_content)
                if vs_pattern:
                    players = [vs_pattern.group(1).strip(), vs_pattern.group(2).strip()]
                    
            if len(players) != 2:
                return None
                
            # Extract tournament information
            tournament_link = container.find('a', href=re.compile(r'.*tournament.*', re.I))
            tournament = tournament_link.get_text(strip=True) if tournament_link else "Unknown Tournament"
            
            # Extract time information
            time_element = container.find(['td', 'span', 'div'], class_=re.compile(r'.*time.*|.*hour.*', re.I))
            start_time = time_element.get_text(strip=True) if time_element else None
            
            # Extract odds if available
            odds_elements = container.find_all(['td', 'span'], class_=re.compile(r'.*odds.*|.*bet.*', re.I))
            odds_player1, odds_player2 = None, None
            
            if len(odds_elements) >= 2:
                try:
                    odds_player1 = float(odds_elements[0].get_text(strip=True))
                    odds_player2 = float(odds_elements[1].get_text(strip=True))
                except (ValueError, IndexError):
                    pass
                    
            # Try to determine surface (look for surface indicators)
            surface = "Unknown"
            surface_indicators = {
                'hard': ['hard', 'hardcourt'],
                'clay': ['clay', 'terre'],
                'grass': ['grass', 'lawn']
            }
            
            text_lower = text_content.lower()
            for surface_type, indicators in surface_indicators.items():
                if any(indicator in text_lower for indicator in indicators):
                    surface = surface_type.title()
                    break
                    
            return TennisMatch(
                player1=players[0],
                player2=players[1],
                tournament=tournament,
                surface=surface,
                round_info="Unknown",
                start_time=start_time,
                odds_player1=odds_player1,
                odds_player2=odds_player2
            )
            
        except Exception as e:
            logger.debug(f"Error extracting match data: {e}")
            return None
            
    def get_player_stats(self, player_name: str) -> Dict:
        """Get player statistics and information"""
        try:
            # Search for player
            search_url = f"{self.base_url}/search/"
            search_data = {'q': player_name}
            
            response = self.session.get(search_url, params=search_data, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find player link
                player_link = soup.find('a', href=re.compile(r'.*player.*', re.I))
                if player_link:
                    player_url = urljoin(self.base_url, player_link['href'])
                    return self._parse_player_page(player_url)
                    
        except Exception as e:
            logger.error(f"Error getting player stats for {player_name}: {e}")
            
        return {}
        
    def _parse_player_page(self, player_url: str) -> Dict:
        """Parse player information from player page"""
        try:
            response = self.session.get(player_url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                stats = {}
                
                # Extract basic info
                stats['name'] = soup.find('h1').get_text(strip=True) if soup.find('h1') else "Unknown"
                
                # Look for ranking
                ranking_element = soup.find(text=re.compile(r'.*ranking.*', re.I))
                if ranking_element:
                    ranking_parent = ranking_element.parent
                    if ranking_parent:
                        stats['ranking'] = ranking_parent.get_text(strip=True)
                        
                # Look for win-loss record
                record_element = soup.find(text=re.compile(r'\d+-\d+'))
                if record_element:
                    stats['record'] = record_element.strip()
                    
                # Extract recent matches
                recent_matches = []
                match_rows = soup.find_all('tr', class_=re.compile(r'.*match.*', re.I))
                
                for row in match_rows[:5]:  # Last 5 matches
                    match_info = row.get_text(strip=True)
                    if match_info:
                        recent_matches.append(match_info)
                        
                stats['recent_matches'] = recent_matches
                
                return stats
                
        except Exception as e:
            logger.error(f"Error parsing player page: {e}")
            
        return {}
        
    def test_connection(self) -> bool:
        """Test connection to tennisexplorer.com"""
        try:
            response = self.session.get(self.base_url, timeout=10)
            if response.status_code == 200:
                logger.info("✅ Successfully connected to TennisExplorer")
                return True
            else:
                logger.error(f"Failed to connect: HTTP {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

def main():
    """Test the scraper"""
    scraper = TennisExplorerScraper()
    
    print("🎾 TennisExplorer Scraper Test")
    print("=" * 40)
    
    # Test connection
    if not scraper.test_connection():
        print("❌ Failed to connect to TennisExplorer")
        return
        
    # Test login
    if scraper.login():
        print("✅ Login successful")
    else:
        print("⚠️ Login failed, continuing without authentication")
        
    # Get current matches
    print("\n🔍 Getting current matches...")
    matches = scraper.get_current_matches(days_ahead=3)
    
    print(f"Found {len(matches)} matches:")
    for i, match in enumerate(matches[:5]):  # Show first 5
        print(f"\n{i+1}. {match.player1} vs {match.player2}")
        print(f"   Tournament: {match.tournament}")
        print(f"   Surface: {match.surface}")
        if match.start_time:
            print(f"   Time: {match.start_time}")
        if match.odds_player1 and match.odds_player2:
            print(f"   Odds: {match.odds_player1} - {match.odds_player2}")

if __name__ == "__main__":
    main()