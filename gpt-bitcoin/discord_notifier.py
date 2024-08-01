import requests
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def send_discord_message(message: str) -> None:
    """Send a message to Discord using the webhook URL."""
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        logger.error("Discord webhook URL is not set in the environment variables.")
        return

    payload = {
        "content": message[:2000]  # Discord has a 2000 character limit
    }

    try:
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()
        logger.info("Message sent to Discord successfully.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send message to Discord: {e}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"Discord API response: {e.response.text}")

def create_roll_summary(decision: Dict[str, Any], portfolio_state: Dict[str, float], current_price: float) -> str:
    """Create a summary message for each roll."""
    summary = f"🔄 Roll Summary ({decision['decision'].upper()})\n\n"
    summary += f"Decision: {decision['decision'].upper()}\n"
    summary += f"Percentage: {decision['percentage']:.2f}%\n"
    summary += f"Target Price: {f'₩{decision["target_price"]:,}' if decision['target_price'] else 'N/A'}\n"
    summary += f"Current BTC Price: ₩{current_price:,}\n"
    summary += f"Reason: {decision['reason']}\n\n"
    summary += f"Portfolio State:\n"
    summary += f"BTC Balance: {portfolio_state['btc_balance']:.8f} BTC\n"
    summary += f"KRW Balance: ₩{portfolio_state['krw_balance']:,.2f}\n"
    summary += f"Total Balance (KRW): ₩{portfolio_state['total_balance_krw']:,.2f}\n"

    return summary

def send_roll_summary(decision: Dict[str, Any], portfolio_state: Dict[str, float], current_price: float) -> None:
    """Create and send a roll summary to Discord."""
    summary = create_roll_summary(decision, portfolio_state, current_price)
    send_discord_message(summary)