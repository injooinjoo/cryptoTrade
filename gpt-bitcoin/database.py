import sqlite3
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

DB_PATH = 'trading_decisions.sqlite'


def initialize_db(db_path: str = DB_PATH) -> None:
    """Initialize the SQLite database and create necessary tables."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                decision TEXT,
                percentage REAL,
                target_price REAL,
                btc_balance REAL,
                krw_balance REAL,
                btc_avg_buy_price REAL,
                btc_krw_price REAL,
                accuracy REAL
            );
        ''')
        conn.commit()
    logger.info("Database initialized successfully.")


def log_decision(decision: dict, upbit_client) -> None:
    """Log the trading decision to the database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO decisions (timestamp, decision, percentage, target_price, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, accuracy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                decision['decision'],
                decision['percentage'],
                decision.get('target_price'),
                upbit_client.get_balance("BTC"),
                upbit_client.get_balance("KRW"),
                upbit_client.get_avg_buy_price("BTC"),
                upbit_client.get_current_price("KRW-BTC"),
                None  # Accuracy will be updated later
            ))
        except sqlite3.OperationalError as e:
            logger.error(f"Error logging decision: {e}")
            raise
        conn.commit()
    logger.info("Decision logged to database.")


def get_previous_decision() -> dict:
    """Retrieve the most recent decision from the database."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row  # This allows accessing columns by name
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM decisions
            ORDER BY timestamp DESC
            LIMIT 1
        ''')
        row = cursor.fetchone()
        if row:
            result = dict(row)
            logger.debug(f"Previous decision: {result}")
            return result
        logger.debug("No previous decision found")
        return None  # Return None if no decision found


def update_decision_accuracy(decision_id: int, accuracy: float) -> None:
    """Update the accuracy of a previous decision."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE decisions
            SET accuracy = ?
            WHERE id = ?
        ''', (accuracy, decision_id))
        conn.commit()
    logger.info(f"Updated accuracy for decision {decision_id}: {accuracy}")


def get_average_accuracy(days: int = 7) -> float:
    """Calculate the average accuracy of decisions over the past specified number of days."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT AVG(accuracy)
            FROM decisions
            WHERE timestamp >= ? AND accuracy IS NOT NULL
        ''', (datetime.now() - timedelta(days=days),))
        return cursor.fetchone()[0] or 0.0


def get_recent_decisions(days: int = 7) -> list:
    """Retrieve decisions from the past specified number of days."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, timestamp, decision, percentage, 
                   COALESCE(target_price, btc_krw_price) as target_price, 
                   btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, accuracy
            FROM decisions
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        ''', (datetime.now() - timedelta(days=days),))
        rows = cursor.fetchall()
        decisions = [dict(row) for row in rows]
        logger.debug(f"Recent decisions: {decisions}")
        return decisions


def get_accuracy_over_time() -> dict:
    """Calculate accuracy over different time periods."""
    periods = [1, 7, 30]  # days
    accuracies = {}
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        for period in periods:
            cursor.execute('''
                SELECT AVG(accuracy)
                FROM decisions
                WHERE timestamp >= ? AND accuracy IS NOT NULL
            ''', (datetime.now() - timedelta(days=period),))
            accuracies[f'{period}_day'] = cursor.fetchone()[0] or 0.0
    return accuracies


