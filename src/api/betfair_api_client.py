#!/usr/bin/env python3
"""
Betfair Exchange API Client
Production-ready Betfair API integration for automated tennis betting
"""

import os
import json
import time
import logging
import requests
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import urllib.parse

from config import get_config

logger = logging.getLogger(__name__)


class BetfairException(Exception):
    """Base exception for Betfair API errors"""
    pass


class BetfairAuthenticationError(BetfairException):
    """Authentication related errors"""
    pass


class BetfairAPIError(BetfairException):
    """General API errors"""
    pass


class MarketStatus(Enum):
    """Betfair market status"""
    INACTIVE = "INACTIVE"
    OPEN = "OPEN"
    SUSPENDED = "SUSPENDED"
    CLOSED = "CLOSED"


class OrderStatus(Enum):
    """Betfair order status"""
    PENDING = "PENDING"
    EXECUTABLE = "EXECUTABLE"
    EXECUTION_COMPLETE = "EXECUTION_COMPLETE"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class BetSide(Enum):
    """Bet side (back or lay)"""
    BACK = "B"
    LAY = "L"


@dataclass
class BetfairMarket:
    """Betfair market data"""
    market_id: str
    market_name: str
    event_name: str
    event_id: str
    start_time: datetime
    status: MarketStatus
    total_matched: float
    runners: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        data['status'] = self.status.value
        return data


@dataclass
class BetfairOdds:
    """Betfair selection odds"""
    selection_id: str
    selection_name: str
    status: str
    adjustment_factor: float
    last_price_traded: Optional[float]
    total_matched: float
    back_prices: List[Tuple[float, float]]  # (price, size)
    lay_prices: List[Tuple[float, float]]   # (price, size)
    
    def get_best_back_price(self) -> Optional[float]:
        """Get best available back price"""
        if self.back_prices:
            return self.back_prices[0][0]
        return None
    
    def get_best_lay_price(self) -> Optional[float]:
        """Get best available lay price"""
        if self.lay_prices:
            return self.lay_prices[0][0]
        return None


@dataclass
class BetfairBet:
    """Betfair bet order"""
    bet_id: str
    market_id: str
    selection_id: str
    side: BetSide
    status: OrderStatus
    persistence_type: str
    order_type: str
    placed_date: datetime
    matched_date: Optional[datetime]
    price_requested: float
    size_requested: float
    size_matched: float
    size_remaining: float
    size_cancelled: float
    size_lapsed: float
    price_matched: Optional[float]
    average_price_matched: Optional[float]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['side'] = self.side.value
        data['status'] = self.status.value
        data['placed_date'] = self.placed_date.isoformat()
        data['matched_date'] = self.matched_date.isoformat() if self.matched_date else None
        return data


class BetfairAPIClient:
    """Production Betfair Exchange API client"""
    
    def __init__(self, app_key: str = None, username: str = None, password: str = None, cert_file: str = None, key_file: str = None):
        self.config = get_config()
        
        # API Configuration
        self.app_key = app_key or self.config.BETFAIR_APP_KEY
        self.username = username or self.config.BETFAIR_USERNAME
        self.password = password or self.config.BETFAIR_PASSWORD
        self.cert_file = cert_file or os.getenv('BETFAIR_CERT_FILE', '')
        self.key_file = key_file or os.getenv('BETFAIR_KEY_FILE', '')
        
        # API URLs
        self.login_url = "https://identitysso-cert.betfair.com/api/certlogin"
        self.api_url = "https://api.betfair.com/exchange"
        self.betting_url = f"{self.api_url}/betting/rest/v1.0"
        self.account_url = f"{self.api_url}/account/rest/v1.0"
        
        # Session management
        self.session_token = None
        self.session_expires = None
        self.session = requests.Session()
        
        # Rate limiting
        self.request_count = 0
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Configure SSL certificates for authentication
        if self.cert_file and self.key_file:
            if os.path.exists(self.cert_file) and os.path.exists(self.key_file):
                self.session.cert = (self.cert_file, self.key_file)
            else:
                logger.warning("Betfair SSL certificates not found - will attempt username/password authentication")
        
        # Simulation mode if credentials not configured
        self.simulation_mode = not all([self.app_key, self.username])
        
        if self.simulation_mode:
            logger.info("Betfair client running in simulation mode")
    
    def _rate_limit(self):
        """Implement rate limiting"""
        now = time.time()
        if now - self.last_request_time < self.min_request_interval:
            sleep_time = self.min_request_interval - (now - self.last_request_time)
            time.sleep(sleep_time)
        self.last_request_time = time.time()
        self.request_count += 1
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None, method: str = "POST") -> Dict[str, Any]:
        """Make authenticated API request with rate limiting"""
        if self.simulation_mode:
            return self._simulate_api_response(endpoint, params)
        
        self._rate_limit()
        
        if not self.session_token or self._session_expired():
            self.authenticate()
        
        headers = {
            'X-Application': self.app_key,
            'X-Authentication': self.session_token,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        try:
            if method == "POST":
                response = self.session.post(endpoint, json=params, headers=headers, timeout=30)
            else:
                response = self.session.get(endpoint, params=params, headers=headers, timeout=30)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Betfair API request failed: {e}")
            raise BetfairAPIError(f"API request failed: {e}")
    
    def _session_expired(self) -> bool:
        """Check if session token has expired"""
        if not self.session_expires:
            return True
        return datetime.now() >= self.session_expires
    
    def authenticate(self) -> bool:
        """Authenticate with Betfair API"""
        if self.simulation_mode:
            logger.info("Authentication simulated - running in demo mode")
            self.session_token = "simulated_session_token"
            self.session_expires = datetime.now() + timedelta(hours=12)
            return True
        
        try:
            # Try certificate authentication first
            if self.cert_file and self.key_file:
                auth_data = {
                    'username': self.username,
                    'password': self.password
                }
                
                response = self.session.post(self.login_url, data=auth_data, timeout=30)
                response.raise_for_status()
                
                auth_result = response.json()
                
                if auth_result.get('status') == 'SUCCESS':
                    self.session_token = auth_result.get('token')
                    self.session_expires = datetime.now() + timedelta(hours=12)
                    logger.info("Betfair authentication successful")
                    return True
                else:
                    raise BetfairAuthenticationError(f"Authentication failed: {auth_result.get('error', 'Unknown error')}")
            
            else:
                # Fallback to interactive login if certificates not available
                logger.warning("Certificate authentication not available - betting will be simulated")
                self.simulation_mode = True
                return self.authenticate()
        
        except Exception as e:
            logger.error(f"Betfair authentication failed: {e}")
            raise BetfairAuthenticationError(f"Authentication failed: {e}")
    
    def get_tennis_events(self, date_from: datetime = None, date_to: datetime = None) -> List[Dict[str, Any]]:
        """Get tennis events from Betfair"""
        if not date_from:
            date_from = datetime.now()
        if not date_to:
            date_to = date_from + timedelta(days=7)
        
        params = {
            'filter': {
                'eventTypeIds': ['2'],  # Tennis event type ID
                'marketStartTime': {
                    'from': date_from.isoformat(),
                    'to': date_to.isoformat()
                }
            }
        }
        
        endpoint = f"{self.betting_url}/listEvents/"
        return self._make_request(endpoint, params)
    
    def get_tennis_markets(self, event_id: str = None, market_types: List[str] = None) -> List[BetfairMarket]:
        """Get tennis markets for events"""
        if not market_types:
            market_types = ['MATCH_ODDS', 'SET_WINNER', 'TOTAL_GAMES', 'HANDICAP']
        
        filter_params = {
            'eventTypeIds': ['2'],  # Tennis
            'marketTypeCodes': market_types
        }
        
        if event_id:
            filter_params['eventIds'] = [event_id]
        
        params = {
            'filter': filter_params,
            'marketProjection': ['COMPETITION', 'EVENT', 'EVENT_TYPE', 'MARKET_START_TIME', 'RUNNER_DESCRIPTION'],
            'maxResults': 100
        }
        
        endpoint = f"{self.betting_url}/listMarketCatalogue/"
        response = self._make_request(endpoint, params)
        
        markets = []
        for market_data in response:
            try:
                market = BetfairMarket(
                    market_id=market_data['marketId'],
                    market_name=market_data['marketName'],
                    event_name=market_data['event']['name'],
                    event_id=market_data['event']['id'],
                    start_time=datetime.fromisoformat(market_data['marketStartTime'].replace('Z', '+00:00')),
                    status=MarketStatus.OPEN,  # Default status
                    total_matched=0.0,
                    runners=market_data.get('runners', [])
                )
                markets.append(market)
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse market data: {e}")
        
        return markets
    
    def get_market_book(self, market_ids: List[str]) -> Dict[str, List[BetfairOdds]]:
        """Get market book (current odds) for markets"""
        params = {
            'marketIds': market_ids,
            'priceProjection': {
                'priceData': ['EX_BEST_OFFERS'],
                'exBestOffersOverrides': {
                    'bestPricesDepth': 3,
                    'rollupModel': 'STAKE',
                    'rollupLimit': 20
                }
            }
        }
        
        endpoint = f"{self.betting_url}/listMarketBook/"
        response = self._make_request(endpoint, params)
        
        market_odds = {}
        for market_data in response:
            market_id = market_data['marketId']
            odds_list = []
            
            for runner in market_data.get('runners', []):
                try:
                    # Parse back prices
                    back_prices = []
                    for price_data in runner.get('ex', {}).get('availableToBack', []):
                        back_prices.append((price_data['price'], price_data['size']))
                    
                    # Parse lay prices
                    lay_prices = []
                    for price_data in runner.get('ex', {}).get('availableToLay', []):
                        lay_prices.append((price_data['price'], price_data['size']))
                    
                    odds = BetfairOdds(
                        selection_id=str(runner['selectionId']),
                        selection_name=runner.get('name', 'Unknown'),
                        status=runner['status'],
                        adjustment_factor=runner.get('adjustmentFactor', 1.0),
                        last_price_traded=runner.get('lastPriceTraded'),
                        total_matched=runner.get('totalMatched', 0.0),
                        back_prices=back_prices,
                        lay_prices=lay_prices
                    )
                    odds_list.append(odds)
                    
                except (KeyError, ValueError) as e:
                    logger.warning(f"Failed to parse runner odds: {e}")
            
            market_odds[market_id] = odds_list
        
        return market_odds
    
    def place_bet(self, market_id: str, selection_id: str, side: BetSide, price: float, size: float, persistence_type: str = "LAPSE") -> Dict[str, Any]:
        """Place a bet on Betfair"""
        params = {
            'marketId': market_id,
            'instructions': [
                {
                    'selectionId': int(selection_id),
                    'handicap': 0,
                    'side': side.value,
                    'orderType': 'LIMIT',
                    'limitOrder': {
                        'size': size,
                        'price': price,
                        'persistenceType': persistence_type
                    }
                }
            ]
        }
        
        endpoint = f"{self.betting_url}/placeOrders/"
        response = self._make_request(endpoint, params)
        
        if response.get('status') == 'SUCCESS':
            instruction_reports = response.get('instructionReports', [])
            if instruction_reports and instruction_reports[0].get('status') == 'SUCCESS':
                bet_id = instruction_reports[0].get('betId')
                logger.info(f"Bet placed successfully: {bet_id}")
                return {
                    'status': 'success',
                    'bet_id': bet_id,
                    'market_id': market_id,
                    'selection_id': selection_id,
                    'price': price,
                    'size': size
                }
            else:
                error_code = instruction_reports[0].get('errorCode', 'Unknown error')
                logger.error(f"Bet placement failed: {error_code}")
                return {'status': 'error', 'error': error_code}
        else:
            error_code = response.get('errorCode', 'Unknown error')
            logger.error(f"Bet placement failed: {error_code}")
            return {'status': 'error', 'error': error_code}
    
    def cancel_bet(self, market_id: str, bet_id: str, size_reduction: float = None) -> Dict[str, Any]:
        """Cancel a bet or reduce its size"""
        instruction = {
            'betId': bet_id
        }
        
        if size_reduction:
            instruction['sizeReduction'] = size_reduction
        
        params = {
            'marketId': market_id,
            'instructions': [instruction]
        }
        
        endpoint = f"{self.betting_url}/cancelOrders/"
        response = self._make_request(endpoint, params)
        
        if response.get('status') == 'SUCCESS':
            return {'status': 'success', 'bet_id': bet_id}
        else:
            error_code = response.get('errorCode', 'Unknown error')
            return {'status': 'error', 'error': error_code}
    
    def get_current_orders(self, market_id: str = None) -> List[BetfairBet]:
        """Get current unmatched orders"""
        params = {}
        if market_id:
            params['marketIds'] = [market_id]
        
        endpoint = f"{self.betting_url}/listCurrentOrders/"
        response = self._make_request(endpoint, params)
        
        orders = []
        for order_data in response.get('currentOrders', []):
            try:
                bet = BetfairBet(
                    bet_id=order_data['betId'],
                    market_id=order_data['marketId'],
                    selection_id=str(order_data['selectionId']),
                    side=BetSide(order_data['side']),
                    status=OrderStatus(order_data['status']),
                    persistence_type=order_data['persistenceType'],
                    order_type=order_data['orderType'],
                    placed_date=datetime.fromisoformat(order_data['placedDate'].replace('Z', '+00:00')),
                    matched_date=datetime.fromisoformat(order_data['matchedDate'].replace('Z', '+00:00')) if order_data.get('matchedDate') else None,
                    price_requested=order_data['priceSize']['price'],
                    size_requested=order_data['priceSize']['size'],
                    size_matched=order_data['sizeMatched'],
                    size_remaining=order_data['sizeRemaining'],
                    size_cancelled=order_data['sizeCancelled'],
                    size_lapsed=order_data['sizeLapsed'],
                    price_matched=order_data.get('averagePriceMatched'),
                    average_price_matched=order_data.get('averagePriceMatched')
                )
                orders.append(bet)
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse order data: {e}")
        
        return orders
    
    def get_cleared_orders(self, bet_status: str = "SETTLED", from_record: int = 0, record_count: int = 100) -> List[BetfairBet]:
        """Get settled/cleared orders"""
        params = {
            'betStatus': bet_status,
            'fromRecord': from_record,
            'recordCount': record_count
        }
        
        endpoint = f"{self.betting_url}/listClearedOrders/"
        response = self._make_request(endpoint, params)
        
        # Similar parsing as get_current_orders but for settled bets
        orders = []
        for order_data in response.get('clearedOrders', []):
            try:
                # Parse cleared order data (structure might be different)
                # Implementation would depend on Betfair's cleared orders response format
                pass
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse cleared order data: {e}")
        
        return orders
    
    def get_account_funds(self) -> Dict[str, Any]:
        """Get account balance and available funds"""
        endpoint = f"{self.account_url}/getAccountFunds/"
        response = self._make_request(endpoint, method="POST")
        
        return {
            'available_balance': response.get('availableToBetBalance', 0.0),
            'exposure': response.get('exposure', 0.0),
            'retained_commission': response.get('retainedCommission', 0.0),
            'exposure_limit': response.get('exposureLimit', 0.0),
            'discount_rate': response.get('discountRate', 0.0),
            'points_balance': response.get('pointsBalance', 0)
        }
    
    def _simulate_api_response(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Simulate API responses for testing"""
        logger.debug(f"Simulating API call to {endpoint}")
        
        if "listEvents" in endpoint:
            return [
                {
                    'event': {
                        'id': '12345',
                        'name': 'Nadal vs Djokovic',
                        'timezone': 'UTC'
                    }
                }
            ]
        
        elif "listMarketCatalogue" in endpoint:
            return [
                {
                    'marketId': 'tennis_match_12345',
                    'marketName': 'Match Winner',
                    'marketStartTime': (datetime.now() + timedelta(hours=2)).isoformat(),
                    'event': {
                        'id': '12345',
                        'name': 'Nadal vs Djokovic'
                    },
                    'runners': [
                        {'selectionId': 12345, 'runnerName': 'Nadal'},
                        {'selectionId': 12346, 'runnerName': 'Djokovic'}
                    ]
                }
            ]
        
        elif "listMarketBook" in endpoint:
            return [
                {
                    'marketId': 'tennis_match_12345',
                    'status': 'OPEN',
                    'runners': [
                        {
                            'selectionId': 12345,
                            'status': 'ACTIVE',
                            'lastPriceTraded': 1.85,
                            'totalMatched': 1000.0,
                            'ex': {
                                'availableToBack': [
                                    {'price': 1.85, 'size': 100.0}
                                ],
                                'availableToLay': [
                                    {'price': 1.90, 'size': 150.0}
                                ]
                            }
                        }
                    ]
                }
            ]
        
        elif "placeOrders" in endpoint:
            import uuid
            return {
                'status': 'SUCCESS',
                'instructionReports': [
                    {
                        'status': 'SUCCESS',
                        'instruction': params['instructions'][0],
                        'betId': str(uuid.uuid4()),
                        'placedDate': datetime.now().isoformat(),
                        'averagePriceMatched': 0.0,
                        'sizeMatched': 0.0
                    }
                ]
            }
        
        elif "getAccountFunds" in endpoint:
            return {
                'availableToBetBalance': 1000.0,
                'exposure': 0.0,
                'retainedCommission': 0.0,
                'exposureLimit': -5000.0,
                'discountRate': 2.0,
                'pointsBalance': 0
            }
        
        else:
            return {'status': 'SUCCESS', 'result': 'simulated'}
    
    def health_check(self) -> Dict[str, Any]:
        """Check API connectivity and authentication status"""
        try:
            if self.simulation_mode:
                return {
                    'status': 'healthy',
                    'mode': 'simulation',
                    'authenticated': True,
                    'message': 'Running in simulation mode'
                }
            
            # Try to get account funds as a health check
            funds = self.get_account_funds()
            return {
                'status': 'healthy',
                'mode': 'live',
                'authenticated': bool(self.session_token),
                'balance': funds.get('available_balance', 0.0),
                'message': 'Connected to Betfair API'
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'mode': 'simulation' if self.simulation_mode else 'live',
                'authenticated': False,
                'error': str(e),
                'message': 'Betfair API connection failed'
            }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    client = BetfairAPIClient()
    
    # Health check
    health = client.health_check()
    print(f"Health Check: {health}")
    
    # Get tennis events
    events = client.get_tennis_events()
    print(f"Found {len(events)} tennis events")
    
    # Get tennis markets
    markets = client.get_tennis_markets()
    print(f"Found {len(markets)} tennis markets")
    
    if markets:
        # Get odds for first market
        market_odds = client.get_market_book([markets[0].market_id])
        print(f"Market odds: {market_odds}")
    
    # Get account funds
    try:
        funds = client.get_account_funds()
        print(f"Account funds: {funds}")
    except Exception as e:
        print(f"Could not get account funds: {e}")