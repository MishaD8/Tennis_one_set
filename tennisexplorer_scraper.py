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
            
    def get_current_matches(self, days_ahead: int = 7) -> List[TennisMatch]:
        """Get current and upcoming matches from multiple tournament sections - ENHANCED for scheduled matches"""
        matches = []
        
        try:
            # 1. PRIORITY: Get matches from general /next/ section (more days for scheduled matches)
            matches.extend(self._get_matches_from_next_section(days_ahead))
            
            # 2. Get scheduled matches from current ATP tournaments
            atp_matches = self._get_scheduled_matches_from_atp_tournaments()
            matches.extend(atp_matches)
            
            # 3. Get scheduled matches from current WTA tournaments  
            wta_matches = self._get_scheduled_matches_from_wta_tournaments()
            matches.extend(wta_matches)
            
            # 4. Get tournament draws and upcoming rounds
            draw_matches = self._get_tournament_draws()
            matches.extend(draw_matches)
            
            # 5. Get live/ongoing matches (lower priority - already started)
            live_matches = self._get_live_matches()
            matches.extend(live_matches)
            
        except Exception as e:
            logger.error(f"Error getting current matches: {e}")
            
        # Remove duplicates based on player names and tournament
        unique_matches = self._remove_duplicate_matches(matches)
        logger.info(f"Found {len(unique_matches)} unique matches")
        return unique_matches
    
    def _get_matches_from_next_section(self, days_ahead: int = 2) -> List[TennisMatch]:
        """Get matches from the general /next/ section"""
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
            logger.error(f"Error getting matches from /next/ section: {e}")
            
        return matches
    
    def _get_scheduled_matches_from_atp_tournaments(self) -> List[TennisMatch]:
        """Get SCHEDULED matches from current ATP tournaments - focus on tomorrow's matches"""
        matches = []
        
        # Current ATP tournaments (summer hard court season)
        atp_tournaments = [
            "kitzbuhel",  # ATP Kitzbuhel (Austria)
            "croatia-open-umag",  # ATP Umag (Croatia)
            "atp-washington",  # ATP Washington (USA)
            "los-cabos-open",  # Los Cabos Open (Mexico)
            "atlanta-open",  # Atlanta Open (USA)
            "winston-salem-open",  # Winston-Salem Open
            "cincinnati-masters"  # Cincinnati Masters
        ]
        
        for tournament in atp_tournaments:
            try:
                # Try different URLs for tournament schedule/draw
                tournament_urls = [
                    f"{self.base_url}/atp/{tournament}/draws/",
                    f"{self.base_url}/atp/{tournament}/schedule/",
                    f"{self.base_url}/atp/{tournament}/",
                    f"{self.base_url}/next/{tournament}/"
                ]
                
                for url in tournament_urls:
                    try:
                        response = self.session.get(url, timeout=10)
                        if response.status_code == 200:
                            tournament_matches = self._parse_scheduled_tournament_matches(response.content, tournament)
                            if tournament_matches:
                                matches.extend(tournament_matches)
                                logger.info(f"✅ Found {len(tournament_matches)} scheduled matches from ATP {tournament}")
                                break  # Found matches, no need to try other URLs
                    except:
                        continue
                        
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.warning(f"Could not get scheduled matches from ATP {tournament}: {e}")
                
        return matches
    
    def _get_scheduled_matches_from_wta_tournaments(self) -> List[TennisMatch]:
        """Get SCHEDULED matches from current WTA tournaments - focus on tomorrow's matches"""
        matches = []
        
        # Current WTA tournaments (summer hard court season)
        wta_tournaments = [
            "livesport-prague-open",  # WTA Prague (Czech Republic)
            "wta-washington",  # WTA Washington (USA)
            "budapest-open",  # Budapest Open (Hungary)
            "palermo-open",  # Palermo Open (Italy)
            "montreal-masters-wta",  # Montreal Masters WTA
            "cincinnati-masters-wta"  # Cincinnati Masters WTA
        ]
        
        for tournament in wta_tournaments:
            try:
                # Try different URLs for tournament schedule/draw
                tournament_urls = [
                    f"{self.base_url}/wta/{tournament}/draws/",
                    f"{self.base_url}/wta/{tournament}/schedule/",
                    f"{self.base_url}/wta/{tournament}/",
                    f"{self.base_url}/next/{tournament}/"
                ]
                
                for url in tournament_urls:
                    try:
                        response = self.session.get(url, timeout=10)
                        if response.status_code == 200:
                            tournament_matches = self._parse_scheduled_tournament_matches(response.content, tournament)
                            if tournament_matches:
                                matches.extend(tournament_matches)
                                logger.info(f"✅ Found {len(tournament_matches)} scheduled matches from WTA {tournament}")
                                break  # Found matches, no need to try other URLs
                    except:
                        continue
                        
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.warning(f"Could not get scheduled matches from WTA {tournament}: {e}")
                
        return matches
    
    def _get_tournament_draws(self) -> List[TennisMatch]:
        """Get matches from tournament draws - focusing on upcoming rounds"""
        matches = []
        
        try:
            # Try main draws page
            draws_urls = [
                f"{self.base_url}/draws/",
                f"{self.base_url}/schedule/",
                f"{self.base_url}/calendar/"
            ]
            
            for url in draws_urls:
                try:
                    response = self.session.get(url, timeout=10)
                    if response.status_code == 200:
                        draw_matches = self._parse_draws_page(response.content)
                        if draw_matches:
                            matches.extend(draw_matches)
                            logger.info(f"✅ Found {len(draw_matches)} matches from draws")
                            break
                except:
                    continue
                    
        except Exception as e:
            logger.warning(f"Could not get tournament draws: {e}")
            
        return matches
    
    def _parse_scheduled_tournament_matches(self, html_content: bytes, tournament: str) -> List[TennisMatch]:
        """Parse SCHEDULED matches from tournament pages - enhanced to find tomorrow's matches"""
        matches = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for schedule tables, draw tables, or upcoming match containers
            match_containers = soup.find_all(['tr', 'div', 'td'], class_=re.compile(r'.*(?:match|game|fixture|draw|schedule|upcoming).*', re.I))
            
            # Also look for time-based containers (matches with specific times)
            time_containers = soup.find_all(string=re.compile(r'\d{1,2}:\d{2}|\d{1,2}\.\d{2}|tomorrow|schedule', re.I))
            for time_container in time_containers:
                if time_container.parent:
                    match_containers.append(time_container.parent)
            
            # Look for tables that might contain future matches
            tables = soup.find_all('table')
            for table in tables:
                table_text = table.get_text().lower()
                if any(keyword in table_text for keyword in ['tomorrow', 'schedule', 'draw', 'upcoming', ':', 'round']):
                    rows = table.find_all('tr')
                    match_containers.extend(rows)
                    
            for container in match_containers:
                match = self._extract_scheduled_match_data(container, tournament)
                if match:
                    matches.append(match)
                    
        except Exception as e:
            logger.error(f"Error parsing scheduled tournament matches: {e}")
            
        return matches
    
    def _parse_draws_page(self, html_content: bytes) -> List[TennisMatch]:
        """Parse tournament draws page for upcoming matches"""
        matches = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for draw brackets or upcoming match listings
            draw_containers = soup.find_all(['div', 'table', 'tr'], class_=re.compile(r'.*(?:draw|bracket|round|upcoming).*', re.I))
            
            for container in draw_containers:
                match = self._extract_draw_match_data(container)
                if match:
                    matches.append(match)
                    
        except Exception as e:
            logger.error(f"Error parsing draws page: {e}")
            
        return matches
    
    def _extract_scheduled_match_data(self, container, tournament: str) -> Optional[TennisMatch]:
        """Extract scheduled match data - enhanced to catch tomorrow's matches"""
        try:
            text = container.get_text(strip=True)
            if not text or len(text) < 10:
                return None
                
            # Look for time indicators (scheduled matches have times)
            time_patterns = [
                r'(\d{1,2}:\d{2})',  # 14:30
                r'(\d{1,2}\.\d{2})',  # 14.30
                r'(tomorrow)',  # tomorrow
                r'(scheduled)',  # scheduled
            ]
            
            has_time = any(re.search(pattern, text.lower()) for pattern in time_patterns)
            
            # Look for player names
            player_patterns = [
                r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:vs?|x|-|–)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
                r'([A-Z]\.\s*[A-Z][a-z]+)\s+(?:vs?|x|-|–)\s+([A-Z]\.\s*[A-Z][a-z]+)',
                r'([A-Z][a-z]+)\s+(?:vs?|x|-|–)\s+([A-Z][a-z]+)'
            ]
            
            player1, player2 = None, None
            for pattern in player_patterns:
                match = re.search(pattern, text)
                if match:
                    player1, player2 = match.groups()
                    player1, player2 = player1.strip(), player2.strip()
                    break
                    
            if not player1 or not player2:
                return None
            
            # Extract time if present
            time_match = re.search(r'(\d{1,2}[:\.]\d{2})', text)
            start_time = time_match.group(1) if time_match else None
            
            # If no specific time but has scheduling keywords, it's probably tomorrow
            if not start_time and any(keyword in text.lower() for keyword in ['tomorrow', 'schedule', 'next']):
                start_time = "Tomorrow"
            
            # Extract round info
            round_info = "Scheduled"
            round_keywords = ['r128', 'r64', 'r32', 'r16', 'r8', 'r4', 'r2', 'final', 'semifinal', 'quarter', 'round']
            for keyword in round_keywords:
                if keyword.lower() in text.lower():
                    round_info = keyword.title()
                    break
                    
            return TennisMatch(
                player1=player1,
                player2=player2,
                tournament=tournament.replace('-', ' ').title(),
                surface="Hard",  # Default for current season
                round_info=round_info,
                start_time=start_time
            )
            
        except Exception as e:
            logger.debug(f"Could not extract scheduled match data: {e}")
            return None
    
    def _extract_draw_match_data(self, container) -> Optional[TennisMatch]:
        """Extract match data from tournament draws"""
        try:
            text = container.get_text(strip=True)
            if not text or len(text) < 10:
                return None
                
            # Similar extraction logic but for draws
            player_patterns = [
                r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:vs?|x|-|–)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            ]
            
            for pattern in player_patterns:
                match = re.search(pattern, text)
                if match:
                    player1, player2 = match.groups()
                    return TennisMatch(
                        player1=player1.strip(),
                        player2=player2.strip(),
                        tournament="Tournament Draw",
                        surface="Hard",
                        round_info="Draw",
                        start_time="Scheduled"
                    )
                    
        except Exception as e:
            logger.debug(f"Could not extract draw match data: {e}")
            
        return None
    
    def _get_matches_from_atp_tournaments(self) -> List[TennisMatch]:
        """Get matches from current ATP tournaments"""
        matches = []
        
        # Current ATP tournaments (real tournaments only)
        atp_tournaments = [
            "croatia-open-umag",  # ATP Umag
            "atp-washington",  # ATP Washington
            "los-cabos-open",  # Los Cabos Open
            "atlanta-open"  # Atlanta Open
        ]
        
        for tournament in atp_tournaments:
            try:
                tournament_url = f"{self.base_url}/atp/{tournament}/"
                response = self.session.get(tournament_url, timeout=10)
                
                if response.status_code == 200:
                    tournament_matches = self._parse_tournament_matches(response.content, tournament)
                    matches.extend(tournament_matches)
                    
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.warning(f"Could not get matches from ATP {tournament}: {e}")
                
        return matches
    
    def _get_matches_from_wta_tournaments(self) -> List[TennisMatch]:
        """Get matches from current WTA tournaments"""  
        matches = []
        
        # Current WTA tournaments
        wta_tournaments = [
            "livesport-prague-open",  # WTA Prague
            "wta-washington",  # WTA Washington
            "budapest-open",  # Budapest Open
            "palermo-open"  # Palermo Open
        ]
        
        for tournament in wta_tournaments:
            try:
                tournament_url = f"{self.base_url}/wta/{tournament}/"
                response = self.session.get(tournament_url, timeout=10)
                
                if response.status_code == 200:
                    tournament_matches = self._parse_tournament_matches(response.content, tournament)
                    matches.extend(tournament_matches)
                    
                # Rate limiting  
                time.sleep(1)
                
            except Exception as e:
                logger.warning(f"Could not get matches from WTA {tournament}: {e}")
                
        return matches
    
    def _get_live_matches(self) -> List[TennisMatch]:
        """Get live/ongoing matches"""
        matches = []
        
        try:
            live_url = f"{self.base_url}/live/"
            response = self.session.get(live_url, timeout=10)
            
            if response.status_code == 200:
                live_matches = self._parse_matches_page(response.content, "Live")
                matches.extend(live_matches)
                
        except Exception as e:
            logger.warning(f"Could not get live matches: {e}")
            
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
    
    def _parse_tournament_matches(self, html_content: bytes, tournament: str) -> List[TennisMatch]:
        """Parse matches from a tournament-specific page"""
        matches = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for tournament match tables or containers
            match_containers = soup.find_all(['tr', 'div'], class_=re.compile(r'.*match.*|.*game.*|.*fixture.*|.*draw.*', re.I))
            
            # Also look for table rows that might contain match data
            if not match_containers:
                tables = soup.find_all('table')
                for table in tables:
                    # Look for tables that might contain match schedules or results
                    if any(keyword in str(table).lower() for keyword in ['match', 'draw', 'schedule', 'fixture']):
                        rows = table.find_all('tr')
                        match_containers.extend(rows)
                        
            for container in match_containers:
                match = self._extract_tournament_match_data(container, tournament)
                if match:
                    matches.append(match)
                    
        except Exception as e:
            logger.error(f"Error parsing tournament matches: {e}")
            
        return matches
    
    def _extract_tournament_match_data(self, container, tournament: str) -> Optional[TennisMatch]:
        """Extract match data from a tournament page container"""
        try:
            # Get all text content from the container
            text = container.get_text(strip=True)
            if not text:
                return None
                
            # Look for player names (typically separated by "vs", "-", "x", etc.)
            player_patterns = [
                r'([A-Za-z\s\.]+)\s+vs\s+([A-Za-z\s\.]+)',
                r'([A-Za-z\s\.]+)\s+x\s+([A-Za-z\s\.]+)', 
                r'([A-Za-z\s\.]+)\s+-\s+([A-Za-z\s\.]+)',
                r'([A-Za-z\s\.]+)\s+–\s+([A-Za-z\s\.]+)'
            ]
            
            player1, player2 = None, None
            for pattern in player_patterns:
                match = re.search(pattern, text, re.I)
                if match:
                    player1, player2 = match.groups()
                    player1, player2 = player1.strip(), player2.strip()
                    break
                    
            if not player1 or not player2:
                return None
                
            # Extract time if present
            time_match = re.search(r'(\d{1,2}:\d{2})', text)
            start_time = time_match.group(1) if time_match else None
            
            # Extract round info
            round_info = "Unknown Round"
            round_keywords = ['final', 'semifinal', 'quarter', 'round', 'r32', 'r16', 'r8', 'r4', 'r2', 'r1']
            for keyword in round_keywords:
                if keyword.lower() in text.lower():
                    round_info = keyword.title()
                    break
                    
            # Extract surface (common surfaces)
            surface = "Hard"  # Default
            if any(surf in text.lower() for surf in ['clay', 'grass', 'carpet', 'indoor']):
                if 'clay' in text.lower():
                    surface = "Clay"
                elif 'grass' in text.lower():
                    surface = "Grass"
                elif 'indoor' in text.lower():
                    surface = "Indoor"
                    
            return TennisMatch(
                player1=player1,
                player2=player2,
                tournament=tournament.replace('-', ' ').title(),
                surface=surface,
                round_info=round_info,
                start_time=start_time
            )
            
        except Exception as e:
            logger.debug(f"Could not extract match data: {e}")
            return None
    
    def _remove_duplicate_matches(self, matches: List[TennisMatch]) -> List[TennisMatch]:
        """Remove duplicate matches based on player names and tournament"""
        seen = set()
        unique_matches = []
        
        for match in matches:
            # Create a key for deduplication
            key = (
                tuple(sorted([match.player1.lower().strip(), match.player2.lower().strip()])),
                match.tournament.lower().strip()
            )
            
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)
                
        return unique_matches

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