import os
import time
from datetime import datetime
import logging
import schedule
from dotenv import load_dotenv
from database import initialize_db, log_decision
from config import load_config
from data_preparation import fetch_and_prepare_data
from trading_logic import analyze_data_with_gpt4, execute_buy, execute_sell
from api_client import UptbitClient, OpenAIClient
from discord_notifier import send_discord_message

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Load configuration
config = load_config()

# Initialize clients
upbit_client = UptbitClient(os.getenv("UPBIT_ACCESS_KEY"), os.getenv("UPBIT_SECRET_KEY"))
openai_client = OpenAIClient(os.getenv("OPENAI_API_KEY"))



def make_decision_and_execute() -> None:
    """Make a trading decision and execute it based on dynamic parameters."""
    logger.info("Making decision and executing...")
    upbit_client.cancel_existing_orders()

    data = fetch_and_prepare_data(upbit_client)
    if data.empty:
        error_message = "No data available for analysis. Skipping this iteration."
        logger.error(error_message)
        send_discord_message(error_message)
        return

    try:
        decision = analyze_data_with_gpt4(data, openai_client, config, upbit_client)
        logger.info(f"Decision: {decision}")

        btc_balance = upbit_client.get_balance("BTC")
        krw_balance = upbit_client.get_balance("KRW")
        btc_current_price = upbit_client.get_current_price("KRW-BTC")
        total_balance_krw = (btc_balance * btc_current_price) + krw_balance

        # Prepare the Discord message
        discord_message = (
            f"Trading Decision:\n"
            f"Action: {decision['decision'].upper()}\n"
            f"Current BTC Price: ₩{btc_current_price:,}\n"
            f"Target Price: {f'₩{decision["target_price"]:,}' if decision['target_price'] else 'N/A'}\n"
            f"Percentage of Total Portfolio: {decision['percentage']:.2f}%\n"
            f"Reason: {decision['reason']}\n"
            f"Total Portfolio Value: ₩{total_balance_krw:,.2f}\n"
            f"BTC Balance: {btc_balance:.8f} BTC\n"
            f"KRW Balance: ₩{krw_balance:,.2f}"
        )

        if decision['decision'] == "buy":
            if krw_balance < config['min_krw_balance']:
                discord_message += f"\nBuy order not placed: Insufficient KRW balance (Minimum: ₩{config['min_krw_balance']:,})"
                logger.info(f"Skipping buy decision: KRW balance ({krw_balance}) is less than {config['min_krw_balance']} KRW")
            else:
                execute_buy(upbit_client, decision['percentage'], decision['target_price'], config)
                discord_message += "\nBuy order placed."
        elif decision['decision'] == "sell":
            execute_sell(upbit_client, decision['percentage'], decision['target_price'], config)
            discord_message += "\nSell order placed."
        else:
            discord_message += "\nHolding position."

        # Send the Discord message
        send_discord_message(discord_message)

        log_decision(decision, upbit_client)
    except Exception as e:
        error_message = f"Error in make_decision_and_execute: {e}"
        logger.error(error_message)
        logger.exception("Traceback:")
        send_discord_message(error_message)


def main() -> None:
    """Main function to run the trading bot."""
    if not upbit_client.check_connection():
        error_message = "Failed to connect to Upbit. Please check your API keys and internet connection."
        logger.error(error_message)
        send_discord_message(error_message)
        return

    initialize_db()

    # Run immediately upon starting
    logger.info("Initial execution of trading logic.")
    make_decision_and_execute()

    # Schedule the task to run every hour at 1 minute past the hour
    # for hour in range(24):
    #     schedule_time_01 = f"{hour:02d}:01"
    #     schedule_time_31 = f"{hour:02d}:31"
    #     schedule.every().day.at(schedule_time_01).do(make_decision_and_execute)
    #     schedule.every().day.at(schedule_time_31).do(make_decision_and_execute)

    schedule.every(10).minutes.do(make_decision_and_execute)

    logger.info("Trading bot started. Running on hourly schedule.")
    send_discord_message("Trading bot started. Running on hourly schedule.")

    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    main()
