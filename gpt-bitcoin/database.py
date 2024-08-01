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
                btc_krw_price REAL
            );
        ''')

        # Check if accuracy column exists, if not, add it
        cursor.execute("PRAGMA table_info(decisions)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'accuracy' not in columns:
            cursor.execute('ALTER TABLE decisions ADD COLUMN accuracy REAL;')

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
            if "no column named accuracy" in str(e):
                # If the accuracy column doesn't exist, add it and retry
                cursor.execute('ALTER TABLE decisions ADD COLUMN accuracy REAL;')
                conn.commit()
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
            else:
                raise
        conn.commit()
    logger.info("Decision logged to database.")

def get_previous_decision():
    """Retrieve the most recent decision from the database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM decisions
            ORDER BY timestamp DESC
            LIMIT 1
        ''')
        return cursor.fetchone()


def update_decision_accuracy(decision_id: int, accuracy: float):
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


def get_average_accuracy(days: int = 7):
    """Calculate the average accuracy of decisions over the past specified number of days."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT AVG(accuracy)
            FROM decisions
            WHERE timestamp >= ? AND accuracy IS NOT NULL
        ''', (datetime.now() - timedelta(days=days),))
        return cursor.fetchone()[0] or 0


def get_recent_decisions(days: int = 7):
    """Retrieve decisions from the past specified number of days."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, timestamp, decision, percentage, 
                   COALESCE(target_price, btc_krw_price) as target_price, 
                   btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, accuracy
            FROM decisions
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        ''', (datetime.now() - timedelta(days=days),))
        return cursor.fetchall()



def get_accuracy_over_time():
    """Calculate accuracy over different time periods."""
    periods = [1, 7, 30]  # days
    accuracies = {}
    for period in periods:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT AVG(accuracy)
                FROM decisions
                WHERE timestamp >= ? AND accuracy IS NOT NULL
            ''', (datetime.now() - timedelta(days=period),))
            accuracies[f'{period}_day'] = cursor.fetchone()[0] or 0
    return accuracies


