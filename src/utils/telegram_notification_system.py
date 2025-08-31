#!/usr/bin/env python3
"""
🤖 Telegram Notification System for Tennis Predictions

Sends Telegram notifications when the system finds strong underdog matches
that meet the criteria for second set betting opportunities.

Integrates with the comprehensive tennis prediction service to automatically
notify users when profitable underdog opportunities are identified.

Author: Claude Code (Anthropic)
"""

import logging
import os
import sys
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Import dynamic rankings API
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.dynamic_rankings_api import dynamic_rankings

# Import prediction-betting integration will be done lazily to avoid circular imports
process_telegram_prediction_as_bet = None

logger = logging.getLogger(__name__)

@dataclass
class TelegramConfig:
    """Telegram bot configuration"""
    bot_token: str
    chat_ids: List[str]  # Support multiple recipients
    enabled: bool = True
    min_probability: float = 0.55  # Minimum underdog probability to notify
    max_notifications_per_hour: int = 10
    notification_cooldown_minutes: int = 30

class TelegramNotificationSystem:
    """
    Telegram notification system for tennis underdog predictions
    """
    
    def __init__(self, config: TelegramConfig = None):
        self.config = config or self._load_config_from_env()
        self.notification_history = []  # Track sent notifications
        self.rate_limiter = {}  # Rate limiting per chat
        
        # Validate configuration
        if not self._validate_config():
            logger.error("❌ Invalid Telegram configuration")
            self.config.enabled = False
        
        self._setup_logging()
        
    def _load_config_from_env(self) -> TelegramConfig:
        """Load configuration from environment variables"""
        # Load environment variables from .env file
        load_dotenv()
        
        # Get environment variables with validation
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '').strip()
        chat_ids_str = os.getenv('TELEGRAM_CHAT_IDS', '').strip()
        enabled_str = os.getenv('TELEGRAM_NOTIFICATIONS_ENABLED', 'true').lower()
        
        # Parse chat IDs safely
        chat_ids = []
        if chat_ids_str:
            chat_ids = [chat_id.strip() for chat_id in chat_ids_str.split(',') if chat_id.strip()]
        
        # Parse numeric values safely
        try:
            min_probability = float(os.getenv('TELEGRAM_MIN_PROBABILITY', '0.55'))
            if not (0.0 <= min_probability <= 1.0):
                logger.warning(f"Invalid TELEGRAM_MIN_PROBABILITY value: {min_probability}, using 0.55")
                min_probability = 0.55
        except (ValueError, TypeError):
            logger.warning("Invalid TELEGRAM_MIN_PROBABILITY value, using default 0.55")
            min_probability = 0.55
            
        try:
            max_notifications_per_hour = int(os.getenv('TELEGRAM_MAX_NOTIFICATIONS_PER_HOUR', '10'))
            if max_notifications_per_hour <= 0:
                logger.warning(f"Invalid TELEGRAM_MAX_NOTIFICATIONS_PER_HOUR value: {max_notifications_per_hour}, using 10")
                max_notifications_per_hour = 10
        except (ValueError, TypeError):
            logger.warning("Invalid TELEGRAM_MAX_NOTIFICATIONS_PER_HOUR value, using default 10")
            max_notifications_per_hour = 10
            
        try:
            notification_cooldown_minutes = int(os.getenv('TELEGRAM_COOLDOWN_MINUTES', '30'))
            if notification_cooldown_minutes < 0:
                logger.warning(f"Invalid TELEGRAM_COOLDOWN_MINUTES value: {notification_cooldown_minutes}, using 30")
                notification_cooldown_minutes = 30
        except (ValueError, TypeError):
            logger.warning("Invalid TELEGRAM_COOLDOWN_MINUTES value, using default 30")
            notification_cooldown_minutes = 30
        
        return TelegramConfig(
            bot_token=bot_token,
            chat_ids=chat_ids,
            enabled=enabled_str in ['true', 'yes', '1', 'on'],
            min_probability=min_probability,
            max_notifications_per_hour=max_notifications_per_hour,
            notification_cooldown_minutes=notification_cooldown_minutes
        )
    
    def _validate_config(self) -> bool:
        """Validate Telegram configuration"""
        if not self.config.bot_token:
            logger.error("❌ TELEGRAM_BOT_TOKEN not provided in environment variables or .env file")
            logger.error("Please set your bot token:")
            logger.error("  Option 1: export TELEGRAM_BOT_TOKEN='your_bot_token_here'")
            logger.error("  Option 2: Add TELEGRAM_BOT_TOKEN=your_bot_token_here to .env file")
            return False
        
        # Validate bot token format (basic check)
        if not self.config.bot_token.count(':') == 1:
            logger.error("❌ Invalid TELEGRAM_BOT_TOKEN format. Expected format: 123456789:ABCdefGhIJKlmNoPQRstUVwxyz")
            return False
        
        if not self.config.chat_ids:
            logger.error("❌ TELEGRAM_CHAT_IDS not provided in environment variables or .env file")
            logger.error("Please set your chat IDs:")
            logger.error("  Option 1: export TELEGRAM_CHAT_IDS='your_chat_id_here'")
            logger.error("  Option 2: Add TELEGRAM_CHAT_IDS=your_chat_id_here to .env file")
            logger.error("Use 'python get_chat_id.py' to find your chat ID")
            return False
        
        # Clean up and validate chat IDs
        self.config.chat_ids = [chat_id.strip() for chat_id in self.config.chat_ids if chat_id.strip()]
        
        if not self.config.chat_ids:
            logger.error("❌ No valid chat IDs found after filtering")
            return False
        
        logger.info(f"✅ Telegram config valid: {len(self.config.chat_ids)} chat(s), min_prob={self.config.min_probability}")
        return True
    
    def _setup_logging(self):
        """Setup logging for telegram notifications"""
        try:
            log_dir = 'logs'
            os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(f'{log_dir}/telegram_notifications.log')
            file_handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            self.telegram_logger = logging.getLogger('telegram_notifications')
            self.telegram_logger.addHandler(file_handler)
            self.telegram_logger.setLevel(logging.INFO)
        except PermissionError as e:
            # If we can't write to the log file, use console logging only
            logger.warning(f"⚠️ Cannot create Telegram log file: {e}. Using console logging only.")
            self.telegram_logger = logging.getLogger('telegram_notifications')
            self.telegram_logger.setLevel(logging.INFO)
    
    def should_notify(self, prediction_result: Dict) -> bool:
        """Check if this prediction should trigger a notification"""
        
        if not self.config.enabled:
            return False
        
        # Check if prediction was successful
        if not prediction_result.get('success', False):
            return False
        
        # Check minimum probability threshold
        underdog_prob = prediction_result.get('underdog_second_set_probability', 0)
        if underdog_prob < self.config.min_probability:
            logger.debug(f"Underdog probability {underdog_prob:.1%} below threshold {self.config.min_probability:.1%}")
            return False
        
        # Check confidence level
        confidence = prediction_result.get('confidence', '').lower()
        if confidence not in ['medium', 'high']:
            logger.debug(f"Confidence level '{confidence}' not high enough for notification")
            return False
        
        # Check rate limiting
        if self._is_rate_limited():
            logger.debug("Rate limited - too many notifications recently")
            return False
        
        # Check for duplicate/similar matches (cooldown)
        if self._is_duplicate_match(prediction_result):
            logger.debug("Similar match recently notified (cooldown active)")
            return False
        
        return True
    
    def _is_rate_limited(self) -> bool:
        """Check if we're rate limited"""
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        
        recent_notifications = [
            n for n in self.notification_history 
            if n.get('timestamp', now) > one_hour_ago
        ]
        
        return len(recent_notifications) >= self.config.max_notifications_per_hour
    
    def _is_duplicate_match(self, prediction_result: Dict) -> bool:
        """Check if this is a duplicate/similar match within cooldown period"""
        now = datetime.now()
        cooldown_cutoff = now - timedelta(minutes=self.config.notification_cooldown_minutes)
        
        match_context = prediction_result.get('match_context', {})
        current_players = {match_context.get('player1', ''), match_context.get('player2', '')}
        
        for notification in self.notification_history:
            if notification.get('timestamp', now) <= cooldown_cutoff:
                continue
            
            # Check for same players
            notified_context = notification.get('prediction', {}).get('match_context', {})
            notified_players = {notified_context.get('player1', ''), notified_context.get('player2', '')}
            
            if current_players == notified_players:
                return True
        
        return False
    
    async def send_underdog_notification(self, prediction_result: Dict) -> bool:
        """Send Telegram notification for underdog prediction"""
        
        if not self.should_notify(prediction_result):
            return False
        
        try:
            message = self._format_underdog_message(prediction_result)
            
            # Send to all configured chats
            success_count = 0
            for chat_id in self.config.chat_ids:
                if await self._send_message(chat_id, message):
                    success_count += 1
            
            # Record successful notification
            if success_count > 0:
                self._record_notification(prediction_result, success_count)
                
                # NEW: Create betting record for this prediction
                self._create_betting_record_for_prediction(prediction_result)
                
                self.telegram_logger.info(
                    f"📤 Sent underdog notification to {success_count}/{len(self.config.chat_ids)} chats: "
                    f"{prediction_result.get('match_context', {}).get('player1', 'Unknown')} vs "
                    f"{prediction_result.get('match_context', {}).get('player2', 'Unknown')}"
                )
                
                return True
            else:
                logger.error("❌ Failed to send notification to any chat")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error sending underdog notification: {e}")
            return False
    
    def _get_live_player_ranking(self, player_name: str) -> int:
        """Get live player ranking with fallback to avoid incorrect rankings"""
        try:
            # Clean and standardize player name
            clean_name = player_name.lower().strip()
            
            # Try dynamic rankings API first
            ranking_data = dynamic_rankings.get_player_ranking(clean_name)
            
            if ranking_data and ranking_data.get('rank', 999) != 999:
                live_rank = ranking_data.get('rank')
                logger.debug(f"🎯 Live ranking for {player_name}: #{live_rank}")
                return live_rank
            else:
                logger.warning(f"⚠️ No live ranking found for {player_name}, using fallback")
                
                # Fallback rankings with CORRECTED values from TODO.md
                corrected_fallback = {
                    # CORRECTED WTA rankings based on TODO.md
                    "linda noskova": 23, "l. noskova": 23, "noskova": 23,
                    "ekaterina alexandrova": 14, "e. alexandrova": 14, "alexandrova": 14,
                    "ajla tomljanovic": 84, "a. tomljanovic": 84, "tomljanovic": 84,
                    
                    # Fixed ranking for M. Bouzkova (was showing #100, actual rank #53)
                    "marie bouzkova": 53, "m. bouzkova": 53, "bouzkova": 53,
                    "m.bouzkova": 53,
                    
                    # Common player name variations
                    "l.noskova": 23, "e.alexandrova": 14, "a.tomljanovic": 84,
                }
                
                # Check corrected fallback first
                if clean_name in corrected_fallback:
                    fallback_rank = corrected_fallback[clean_name]
                    logger.info(f"✅ Using corrected fallback ranking for {player_name}: #{fallback_rank}")
                    return fallback_rank
                
                # Default fallback
                default_rank = 150
                logger.warning(f"⚠️ Using default ranking for {player_name}: #{default_rank}")
                return default_rank
                
        except Exception as e:
            logger.error(f"❌ Error getting live ranking for {player_name}: {e}")
            return 150  # Safe default
    
    def _extract_enhanced_insights(self, prediction_result: Dict) -> List[str]:
        """Extract enhanced insights from API data for Telegram notifications"""
        enhanced_insights = []
        
        try:
            # Get prediction metadata to check if we have enhanced data
            prediction_metadata = prediction_result.get('prediction_metadata', {})
            match_context = prediction_result.get('match_context', {})
            
            # Check for enhanced prediction type
            if prediction_metadata.get('service_type') == 'automated_ml_prediction':
                models_used = prediction_metadata.get('models_used', [])
                if models_used:
                    enhanced_insights.append(f"🤖 ML Models: {len(models_used)} models used ({', '.join(models_used[:2])}...)")
            
            # Extract ranking-based insights
            player1_rank = match_context.get('player1_rank', 150)
            player2_rank = match_context.get('player2_rank', 150)
            underdog_rank = max(player1_rank, player2_rank)
            
            # Tier-specific insights for our 10-300 strategy
            if 10 <= underdog_rank <= 50:
                enhanced_insights.append("⭐ Quality underdog (top 50) - higher success potential")
            elif 51 <= underdog_rank <= 100:
                enhanced_insights.append("💪 Solid underdog (51-100) - good value opportunity")
            elif 101 <= underdog_rank <= 200:
                enhanced_insights.append("📊 Standard underdog (101-200) - moderate risk/reward")
            elif 201 <= underdog_rank <= 300:
                enhanced_insights.append("🎯 Deep underdog (201-300) - high risk/high reward")
            
            # Extract surface-specific insights if available
            surface = match_context.get('surface', '').lower()
            if surface in ['clay', 'grass', 'hard']:
                enhanced_insights.append(f"🎾 Surface: {surface.title()} court specialist advantage possible")
            
            # Extract tournament importance
            tournament = match_context.get('tournament', '').lower()
            if any(slam in tournament for slam in ['us open', 'wimbledon', 'french', 'australian']):
                enhanced_insights.append("🏆 Grand Slam match - higher unpredictability factor")
            elif 'masters' in tournament or '1000' in tournament:
                enhanced_insights.append("🔥 Masters level tournament - premium competition")
            
            # Extract data quality insights if available from strategic insights
            strategic_insights = prediction_result.get('strategic_insights', [])
            for insight in strategic_insights:
                if 'data quality' in insight.lower():
                    enhanced_insights.append("✅ High-quality data analysis available")
                    break
                elif 'form' in insight.lower() and ('rising' in insight.lower() or 'declining' in insight.lower()):
                    enhanced_insights.append("📈 Recent form trends favor underdog scenario")
                    break
            
            # Confidence-based insights
            confidence = prediction_result.get('confidence', 'Medium')
            underdog_prob = prediction_result.get('underdog_second_set_probability', 0)
            
            if confidence == 'High' and underdog_prob > 0.6:
                enhanced_insights.append("🔥 High-confidence prediction with strong probability")
            elif confidence == 'Medium' and underdog_prob > 0.5:
                enhanced_insights.append("⚡ Solid prediction with competitive probability")
            
            return enhanced_insights[:4]  # Return up to 4 insights
            
        except Exception as e:
            logger.debug(f"Error extracting enhanced insights: {e}")
            return []
    
    def _format_underdog_message(self, prediction_result: Dict) -> str:
        """Format the underdog prediction message for Telegram - Concise version"""
        
        match_context = prediction_result.get('match_context', {})
        underdog_prob = prediction_result.get('underdog_second_set_probability', 0)
        confidence = prediction_result.get('confidence', 'Medium')
        underdog_player = prediction_result.get('underdog_player', 'unknown')
        
        # Determine which player is the underdog
        player1 = match_context.get('player1', 'Player 1')
        player2 = match_context.get('player2', 'Player 2')
        
        # Get live rankings instead of using potentially incorrect cached values
        player1_rank = self._get_live_player_ranking(player1)
        player2_rank = self._get_live_player_ranking(player2)
        
        # Also log the ranking source for transparency
        logger.info(f"📊 Live rankings: {player1} = #{player1_rank}, {player2} = #{player2_rank}")
        
        if underdog_player == 'player1':
            underdog_name = player1
            underdog_rank = player1_rank
            favorite_name = player2
            favorite_rank = player2_rank
        else:
            underdog_name = player2
            underdog_rank = player2_rank
            favorite_name = player1
            favorite_rank = player1_rank
        
        # Calculate ranking gap for clarity (ensure positive value)
        ranking_gap = abs(underdog_rank - favorite_rank)
        
        # Tournament and surface info
        tournament = match_context.get('tournament', 'Unknown Tournament')
        surface = match_context.get('surface', 'Hard')
        
        # Build concise message
        message_lines = [
            "🎾 <b>TENNIS UNDERDOG ALERT</b> 🚨",
            "",
            f"🏆 <b>{tournament}</b>",
            f"🎯 <b>{underdog_name}</b> (#{underdog_rank}) vs <b>{favorite_name}</b> (#{favorite_rank})",
            f"🏟️ Surface: {surface}",
            "",
            f"📊 <b>Second Set Win Probability: {underdog_prob:.1%}</b>",
            f"🔮 Confidence: {confidence}",
            f"📈 Ranking Gap: {ranking_gap} positions",
        ]
        
        # Add top strategic insights (max 2 for brevity)
        insights = prediction_result.get('strategic_insights', [])
        if insights:
            message_lines.extend(["", "💡 <b>Key Insights:</b>"])
            for insight in insights[:2]:  # Only show top 2 insights
                # Clean insight and make it concise
                clean_insight = insight.strip()
                # Remove excessive emojis
                for emoji in ['🔥', '⚡', '🛡️', '🏆', '📊', '⚖️']:
                    clean_insight = clean_insight.replace(emoji, '').strip()
                # Shorten if too long
                if len(clean_insight) > 60:
                    clean_insight = clean_insight[:57] + "..."
                message_lines.append(f"• {clean_insight}")
        
        # Add timestamp
        from datetime import datetime
        current_time = datetime.now().strftime('%H:%M')
        message_lines.extend([
            "",
            f"⏰ <i>Alert sent at {current_time}</i>"
        ])
        
        return "\n".join(message_lines)
    
    async def _send_message(self, chat_id: str, message: str) -> bool:
        """Send message to specific Telegram chat"""
        
        url = f"https://api.telegram.org/bot{self.config.bot_token}/sendMessage"
        
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'HTML',
            'disable_web_page_preview': True
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as response:
                    if response.status == 200:
                        logger.debug(f"✅ Message sent to chat {chat_id}")
                        return True
                    else:
                        response_text = await response.text()
                        logger.error(f"❌ Failed to send message to chat {chat_id}: {response.status} - {response_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Error sending message to chat {chat_id}: {e}")
            return False
    
    def _record_notification(self, prediction_result: Dict, success_count: int):
        """Record sent notification for rate limiting and deduplication"""
        
        notification_record = {
            'timestamp': datetime.now(),
            'prediction': prediction_result,
            'chats_notified': success_count,
            'underdog_probability': prediction_result.get('underdog_second_set_probability', 0)
        }
        
        self.notification_history.append(notification_record)
        
        # Keep only recent history (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.notification_history = [
            n for n in self.notification_history 
            if n.get('timestamp', datetime.now()) > cutoff_time
        ]
    
    def _create_betting_record_for_prediction(self, prediction_result: Dict):
        """Create a betting record for this prediction notification"""
        try:
            # Lazy import to avoid circular imports
            global process_telegram_prediction_as_bet
            if process_telegram_prediction_as_bet is None:
                try:
                    from api.prediction_betting_integration import process_telegram_prediction_as_bet
                except ImportError:
                    self.telegram_logger.warning("⚠️ Betting integration not available, skipping betting record creation")
                    return
                
            # Create betting record asynchronously
            bet_id = process_telegram_prediction_as_bet(prediction_result)
            
            if bet_id:
                self.telegram_logger.info(f"✅ Created betting record {bet_id} for notification")
                
                # Add betting record ID to notification history for tracking
                if self.notification_history:
                    self.notification_history[-1]['betting_record_id'] = bet_id
            else:
                self.telegram_logger.warning("⚠️ Failed to create betting record for notification")
                
        except Exception as e:
            self.telegram_logger.error(f"❌ Error creating betting record for prediction: {e}")
    
    def send_notification_sync(self, prediction_result: Dict) -> bool:
        """Synchronous wrapper for sending notifications"""
        try:
            # Create event loop if it doesn't exist
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is already running, create a task
                    asyncio.create_task(self.send_underdog_notification(prediction_result))
                    return True  # Return True optimistically
                else:
                    return loop.run_until_complete(self.send_underdog_notification(prediction_result))
            except RuntimeError:
                # No event loop in current thread, create new one
                return asyncio.run(self.send_underdog_notification(prediction_result))
                
        except Exception as e:
            logger.error(f"❌ Error in sync notification wrapper: {e}")
            return False
    
    async def send_test_message(self, test_message: str = None) -> bool:
        """Send test message to verify Telegram integration"""
        
        if not self.config.enabled:
            logger.error("❌ Telegram notifications disabled")
            return False
        
        test_msg = test_message or (
            "🤖 <b>Tennis Prediction Bot - Test Message</b>\n\n"
            "✅ Telegram integration is working!\n"
            f"📅 Test sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
            "🎾 Ready to receive underdog tennis predictions!"
        )
        
        success_count = 0
        for chat_id in self.config.chat_ids:
            if await self._send_message(chat_id, test_msg):
                success_count += 1
        
        logger.info(f"📤 Test message sent to {success_count}/{len(self.config.chat_ids)} chats")
        return success_count > 0
    
    def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification statistics"""
        
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        one_day_ago = now - timedelta(days=1)
        
        recent_notifications = [
            n for n in self.notification_history 
            if n.get('timestamp', now) > one_day_ago
        ]
        
        hourly_notifications = [
            n for n in recent_notifications
            if n.get('timestamp', now) > one_hour_ago
        ]
        
        return {
            'enabled': self.config.enabled,
            'chat_count': len(self.config.chat_ids),
            'min_probability_threshold': self.config.min_probability,
            'notifications_last_hour': len(hourly_notifications),
            'notifications_last_24h': len(recent_notifications),
            'rate_limit_remaining': max(0, self.config.max_notifications_per_hour - len(hourly_notifications)),
            'average_underdog_probability': (
                sum(n.get('underdog_probability', 0) for n in recent_notifications) / 
                max(1, len(recent_notifications))
            ) if recent_notifications else 0
        }
    
    async def _send_emergency_alert(self, alert_message: str) -> bool:
        """Send emergency alert for critical issues (bypasses rate limiting)"""
        try:
            logger.warning("🚨 SENDING EMERGENCY ALERT - bypassing rate limits")
            
            success_count = 0
            for chat_id in self.config.chat_ids:
                try:
                    # Send emergency alert without rate limiting
                    url = f"https://api.telegram.org/bot{self.config.bot_token}/sendMessage"
                    
                    payload = {
                        'chat_id': chat_id,
                        'text': alert_message,
                        'parse_mode': 'HTML',
                        'disable_web_page_preview': True
                    }
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.post(url, json=payload, timeout=10) as response:
                            if response.status == 200:
                                success_count += 1
                                logger.info(f"✅ Emergency alert sent to chat {chat_id}")
                            else:
                                logger.error(f"❌ Failed to send emergency alert to chat {chat_id}: {response.status}")
                                
                except Exception as e:
                    logger.error(f"❌ Error sending emergency alert to {chat_id}: {e}")
            
            logger.warning(f"🚨 Emergency alert sent to {success_count}/{len(self.config.chat_ids)} chats")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"❌ Critical error sending emergency alert: {e}")
            return False


# Global instance for easy access
_telegram_system = None

def get_telegram_system() -> TelegramNotificationSystem:
    """Get global Telegram notification system instance"""
    global _telegram_system
    if _telegram_system is None:
        _telegram_system = TelegramNotificationSystem()
    return _telegram_system

def init_telegram_system(config: TelegramConfig = None) -> TelegramNotificationSystem:
    """Initialize Telegram notification system"""
    global _telegram_system
    _telegram_system = TelegramNotificationSystem(config)
    return _telegram_system

def send_underdog_alert(prediction_result: Dict) -> bool:
    """Convenience function to send underdog alert"""
    telegram_system = get_telegram_system()
    return telegram_system.send_notification_sync(prediction_result)

def send_test_notification(message: str = None) -> bool:
    """Convenience function to send test notification"""
    telegram_system = get_telegram_system()
    try:
        return asyncio.run(telegram_system.send_test_message(message))
    except Exception as e:
        logger.error(f"❌ Error sending test notification: {e}")
        return False

# Test the system if run directly
if __name__ == "__main__":
    print("🤖 TELEGRAM NOTIFICATION SYSTEM TEST")
    print("=" * 50)
    
    # Test configuration
    config = TelegramConfig(
        bot_token=os.getenv('TELEGRAM_BOT_TOKEN', 'test_token'),
        chat_ids=[os.getenv('TELEGRAM_CHAT_ID', 'test_chat_id')],
        min_probability=0.55
    )
    
    system = TelegramNotificationSystem(config)
    
    print(f"📊 Configuration:")
    print(f"  Enabled: {system.config.enabled}")
    print(f"  Chat IDs: {len(system.config.chat_ids)}")
    print(f"  Min Probability: {system.config.min_probability:.1%}")
    
    # Test notification stats
    stats = system.get_notification_stats()
    print(f"\n📈 Current Stats:")
    print(f"  Notifications last hour: {stats['notifications_last_hour']}")
    print(f"  Rate limit remaining: {stats['rate_limit_remaining']}")
    
    # Create sample prediction for testing
    sample_prediction = {
        'success': True,
        'underdog_second_set_probability': 0.62,
        'confidence': 'High',
        'underdog_player': 'player1',
        'match_context': {
            'player1': 'Test Underdog',
            'player2': 'Test Favorite',
            'player1_rank': 150,
            'player2_rank': 75,
            'tournament': 'ATP 250 Test Tournament',
            'surface': 'Hard'
        },
        'strategic_insights': [
            '🔥 Strong underdog opportunity detected',
            'Ranking Gap: 55 positions'
        ],
        'prediction_metadata': {
            'prediction_time': datetime.now().isoformat()
        }
    }
    
    print(f"\n🧪 Testing notification logic:")
    should_notify = system.should_notify(sample_prediction)
    print(f"  Should notify: {should_notify}")
    
    if system.config.enabled and should_notify:
        print(f"\n📤 Sending test notification...")
        # Don't actually send in test mode unless explicitly configured
        print(f"  (Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_IDS to test actual sending)")
    
    print(f"\n✅ Telegram system test completed!")