import json
import logging
from discord_notifier import send_discord_message, send_roll_summary
from database import get_previous_decision, update_decision_accuracy, get_recent_decisions, get_accuracy_over_time, initialize_db


logger = logging.getLogger(__name__)
initialize_db()


def get_instructions(file_path: str) -> str:
    """Read instructions from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        logger.error(f"Instructions file {file_path} not found.")
    except Exception as e:
        logger.error(f"An error occurred while reading the instructions file: {e}")
    return ""


def calculate_accuracy(action, target_price, current_price, btc_krw_price_at_decision):
    """Calculate the accuracy of a decision based on price movement."""
    price_change_percentage = (current_price - btc_krw_price_at_decision) / btc_krw_price_at_decision * 100

    if action == 'buy':
        # For buy decisions, positive price change is good
        return max(0, min(100, 50 + price_change_percentage))
    elif action == 'sell':
        # For sell decisions, negative price change is good
        return max(0, min(100, 50 - price_change_percentage))
    else:  # hold
        # For hold decisions, price stability is good
        # Define a threshold for "significant" price movement (e.g., 1%)
        threshold = 1.0
        if abs(price_change_percentage) <= threshold:
            # Price didn't move significantly, hold was a good decision
            return 100 - (abs(price_change_percentage) / threshold) * 50
        else:
            # Price moved significantly, hold might not have been the best decision
            return max(0, 50 - (abs(price_change_percentage) - threshold))


def evaluate_decisions(upbit_client, openai_client):
    """Evaluate the accuracy of recent decisions and provide insights."""
    try:
        recent_decisions = get_recent_decisions(days=7)
        if not recent_decisions:
            logger.info("No recent decisions found for evaluation.")
            return None

        current_price = upbit_client.get_current_price("KRW-BTC")

        evaluations = []
        last_decision = None
        for decision in recent_decisions:
            try:
                decision_id, timestamp, action, percentage, target_price, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, prev_accuracy = decision

                # Handle potential None values
                if target_price is None:
                    target_price = btc_krw_price

                if target_price is None or btc_krw_price is None:
                    logger.warning(f"Skipping decision {decision_id} due to missing price data")
                    continue

                # Calculate accuracy
                accuracy = calculate_accuracy(action, target_price, current_price, btc_krw_price)

                update_decision_accuracy(decision_id, accuracy)

                evaluation = {
                    "timestamp": str(timestamp),
                    "action": action,
                    "target_price": target_price,
                    "btc_krw_price_at_decision": btc_krw_price,
                    "current_price": current_price,
                    "price_change_percentage": ((current_price - btc_krw_price) / btc_krw_price) * 100,
                    "accuracy": accuracy
                }
                evaluations.append(evaluation)

                # Store the most recent decision
                if last_decision is None or timestamp > datetime.strptime(last_decision["timestamp"],
                                                                          "%Y-%m-%d %H:%M:%S.%f"):
                    last_decision = evaluation

            except Exception as e:
                logger.error(f"Error processing decision {decision_id}: {e}")
                continue

        accuracies = get_accuracy_over_time()

        # Prepare data for GPT-4 analysis
        evaluation_data = {
            "recent_decisions": evaluations,
            "current_price": current_price,
            "accuracies": accuracies
        }

        # Use GPT-4 to analyze the decisions and provide insights
        prompt = f"""
        Analyze the recent trading decisions and their outcomes:

        Recent Decisions: {json.dumps(evaluation_data['recent_decisions'], indent=2)}
        Current Price: {evaluation_data['current_price']}
        Accuracies: {json.dumps(evaluation_data['accuracies'], indent=2)}

        Please provide:
        1. An assessment of the overall performance based on these recent decisions.
        2. Identify any patterns or trends in the decision-making process.
        3. Explain possible reasons for any consistent inaccuracies or successes.
        4. Suggest how to apply these insights to improve future estimations.
        5. Considering the accuracy over different time periods, what adjustments should be made to the decision-making process?

        Format your response as a JSON object with keys: "overall_assessment", "patterns_identified", "reasons_for_performance", "future_improvements", "adjustments_needed".
        """

        response = openai_client.chat_completion(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI assistant analyzing Bitcoin trading decisions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            response_format={"type": "json_object"}
        )

        # Log the raw response for debugging
        logger.debug(f"Raw GPT-4 response: {response.choices[0].message.content}")

        try:
            analysis = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from GPT-4 response: {e}")
            logger.error(f"Problematic JSON string: {response.choices[0].message.content}")
            return None

        # Prepare and send Discord message
        last_decision_assessment = "No recent decisions found."
        if last_decision:
            accuracy_threshold = 60  # Consider a decision "correct" if accuracy is above 60%
            if last_decision['accuracy'] >= accuracy_threshold:
                last_decision_assessment = f"The last {last_decision['action']} decision was CORRECT with an accuracy of {last_decision['accuracy']:.2f}%."
            else:
                last_decision_assessment = f"The last {last_decision['action']} decision was INCORRECT with an accuracy of {last_decision['accuracy']:.2f}%."
            last_decision_assessment += f" Price changed by {last_decision['price_change_percentage']:.2f}% since the decision."

        discord_message = f"""
        Recent Decisions Evaluation:
        1-day Accuracy: {accuracies.get('1_day', 'N/A'):.2f}%
        7-day Accuracy: {accuracies.get('7_day', 'N/A'):.2f}%
        30-day Accuracy: {accuracies.get('30_day', 'N/A'):.2f}%

        Last Decision Assessment:
        {last_decision_assessment}

        Overall Assessment: {analysis.get('overall_assessment', 'N/A')}

        Patterns Identified: {analysis.get('patterns_identified', 'N/A')}

        Reasons for Performance: {analysis.get('reasons_for_performance', 'N/A')}

        Future Improvements: {analysis.get('future_improvements', 'N/A')}

        Adjustments Needed: {analysis.get('adjustments_needed', 'N/A')}
        """
        send_discord_message(discord_message)

        return analysis

    except Exception as e:
        logger.error(f"Error in evaluating decisions: {e}")
        logger.exception("Traceback:")
        return None


def analyze_data_with_gpt4(data, openai_client, config, upbit_client):
    """Analyze data using GPT-4 and return a decision."""
    instructions = """
    Analyze the provided market data and make a trading decision for Bitcoin (BTC).
    Your response should be a JSON object with the following structure:
    {
        "decision": "buy" or "sell" or "hold",
        "percentage": a number between 0 and 100 representing the percentage of the total portfolio value (BTC+KRW in KRW terms) to use for the trade,
        "target_price": the target price for the trade (use null for hold decisions),
        "reason": a brief explanation of your decision
    }
    Base your decision on the provided market data, including the Stochastic Oscillator, Market Sentiment, Price Divergence, and the current portfolio state.
    Consider the full portfolio value (BTC+KRW in KRW terms) when determining the percentage for buy/sell decisions.
    Only suggest "buy" if there is available KRW, and only suggest "sell" if there is BTC available.
    Apply aggressive risk management principles, prioritizing higher potential returns while acknowledging increased risks.
    """
    data_json = data.to_json(orient='split')

    latest_data = data.iloc[-1]
    btc_balance = upbit_client.get_balance("BTC")
    krw_balance = upbit_client.get_balance("KRW")
    btc_current_price = upbit_client.get_current_price("KRW-BTC")
    total_balance_krw = (btc_balance * btc_current_price) + krw_balance

    portfolio_state = {
        "btc_balance": btc_balance,
        "krw_balance": krw_balance,
        "btc_current_price": btc_current_price,
        "total_balance_krw": total_balance_krw
    }

    logger.info(f"Latest data point: Close: {latest_data['close']}, RSI: {latest_data['RSI_14']}, "
                f"Stochastic %K: {latest_data['Stochastic_%K']}, Stochastic %D: {latest_data['Stochastic_%D']}, "
                f"MACD: {latest_data['MACD']}, MACD Signal: {latest_data['MACD_Signal']}")
    logger.info(f"Portfolio state: {portfolio_state}")

    try:
        # Evaluate recent decisions before making a new one
        decisions_evaluation = evaluate_decisions(upbit_client, openai_client)

        # Include decisions evaluation in the prompt if available
        if decisions_evaluation:
            instructions += f"\n\nConsider the following evaluation of recent decisions when making your recommendation:\n{json.dumps(decisions_evaluation, indent=2)}"

        response = openai_client.chat_completion(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": json.dumps({"market_data": data_json, "portfolio_state": portfolio_state, "decisions_evaluation": decisions_evaluation})}
            ],
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        advice = response.choices[0].message.content
        logger.info(f"Raw GPT-4 response: {advice}")
        parsed_decision = parse_decision(advice, portfolio_state)
        logger.info(f"Parsed decision: {parsed_decision}")

        # Send roll summary to Discord
        send_roll_summary(parsed_decision, portfolio_state, btc_current_price)

        return parsed_decision
    except Exception as e:
        error_message = f"Error in gpt-4o-mini analysis: {e}"
        logger.error(error_message)
        logger.exception("Traceback:")
        send_discord_message(error_message)
        raise



def parse_decision(advice: str, portfolio_state: dict) -> dict:
    """Parse the decision from the GPT-4 advice."""
    try:
        decision = json.loads(advice)

        if "decision" in decision:
            parsed_decision = {
                "decision": decision.get("decision", "hold"),
                "percentage": float(decision.get("percentage", 0) or 0),  # Use 0 if percentage is None
                "reason": decision.get("reason", "No specific reason provided"),
                "target_price": float(decision.get("target_price", 0) or 0) if decision.get(
                    "target_price") is not None else None
            }

            # Adjust the percentage based on the total portfolio value
            if parsed_decision["decision"] == "buy":
                max_buy_amount = portfolio_state["krw_balance"]
                adjusted_percentage = (parsed_decision["percentage"] / 100) * portfolio_state[
                    "total_balance_krw"] / max_buy_amount * 100
                parsed_decision["percentage"] = min(adjusted_percentage, 100)
            elif parsed_decision["decision"] == "sell":
                max_sell_amount = portfolio_state["btc_balance"] * portfolio_state["btc_current_price"]
                adjusted_percentage = (parsed_decision["percentage"] / 100) * portfolio_state[
                    "total_balance_krw"] / max_sell_amount * 100
                parsed_decision["percentage"] = min(adjusted_percentage, 100)

            return parsed_decision

        # If the format is not recognized
        logger.warning("GPT-4 response doesn't contain expected decision format.")
        return {
            "decision": "hold",
            "percentage": 0,
            "reason": "Unexpected response format from GPT-4",
            "target_price": None
        }
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error parsing GPT-4 advice: {e}. Using default 'hold' decision.")
        return {
            "decision": "hold",
            "percentage": 0,
            "reason": f"Error parsing advice: {str(e)}",
            "target_price": None
        }


def execute_buy(upbit_client, percentage: float, target_price: float, config: dict) -> None:
    """Execute a buy limit order at a specified target price."""
    try:
        krw_balance = upbit_client.get_balance("KRW")
        btc_balance = upbit_client.get_balance("BTC")
        btc_current_price = upbit_client.get_current_price("KRW-BTC")
        total_balance_krw = (btc_balance * btc_current_price) + krw_balance

        if krw_balance < config['min_krw_balance']:
            logger.info(f"Buy order not placed: KRW balance ({krw_balance}) is less than {config['min_krw_balance']} KRW")
            return

        if target_price is None:
            logger.info("Buy order not placed: No target price specified")
            return

        amount_to_invest = total_balance_krw * (percentage / 100)
        amount_to_invest = min(amount_to_invest, krw_balance * 0.9995)  # Adjust for potential fees
        amount_to_buy = amount_to_invest / target_price

        logger.info(f"Buy order details: Total balance: {total_balance_krw} KRW, Amount to invest: {amount_to_invest} KRW, "
                    f"Target price: {target_price} KRW, Amount to buy: {amount_to_buy} BTC")

        if amount_to_buy < config['min_order_size']:
            logger.info(f"Buy order not placed: Amount to buy ({amount_to_buy} BTC) is below the minimum order size ({config['min_order_size']} BTC)")
            return

        if amount_to_invest > config['min_transaction_amount']:
            result = upbit_client.buy_limit_order("KRW-BTC", target_price, amount_to_buy)
            if result is None:
                logger.warning("Buy limit order placement returned None. This might indicate an issue with the order.")
            else:
                logger.info(f"Buy limit order placed successfully: {result}")
        else:
            logger.info(f"Buy order not placed: Amount to invest ({amount_to_invest} KRW) is below the minimum transaction amount ({config['min_transaction_amount']} KRW)")
    except Exception as e:
        if "InsufficientFundsBid" in str(e):
            logger.warning(f"InsufficientFundsBid error: Not enough funds to place the buy order. Available KRW: {krw_balance}, Attempted to use: {amount_to_invest}")
            # Optionally, you could try to place a smaller order here
        else:
            logger.error(f"Failed to execute buy limit order: {e}")
        logger.exception("Traceback:")


def execute_sell(upbit_client, percentage: float, target_price: float, config: dict) -> None:
    """Execute a sell limit order at a specified target price."""
    try:
        if target_price is None:
            logger.info("Sell order not placed: No target price specified")
            return

        btc_balance = upbit_client.get_balance("BTC")
        krw_balance = upbit_client.get_balance("KRW")
        btc_current_price = upbit_client.get_current_price("KRW-BTC")
        total_balance_krw = (btc_balance * btc_current_price) + krw_balance

        amount_to_sell_krw = total_balance_krw * (percentage / 100)
        amount_to_sell_btc = min(amount_to_sell_krw / target_price, btc_balance)  # Ensure we don't exceed available BTC

        logger.info(f"Sell order details: Total balance: {total_balance_krw} KRW, Amount to sell: {amount_to_sell_krw} KRW, "
                    f"Target price: {target_price} KRW, Amount to sell: {amount_to_sell_btc} BTC")

        if amount_to_sell_btc * target_price > config['min_transaction_amount']:
            result = upbit_client.sell_limit_order("KRW-BTC", target_price, amount_to_sell_btc)
            logger.info(f"Sell limit order placed successfully: {result}")
        else:
            logger.info(f"Sell order not placed: Amount too small. BTC balance: {btc_balance}, Amount to sell: {amount_to_sell_btc}, Target price: {target_price}")
    except Exception as e:
        logger.error(f"Failed to execute sell limit order: {e}")
        logger.exception("Traceback:")