import logging
import os

import requests

logger = logging.getLogger(__name__)


def send_discord_message(message: str) -> None:
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        logger.error("Discord webhook URL is not set in the environment variables.")
        return

    payload = {"content": message[:2000]}  # Discord has a 2000-character limit

    try:
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()
        logger.info("Message sent to Discord successfully.")
    except Exception as e:
        logger.error(f"Failed to send message to Discord: {e}")
