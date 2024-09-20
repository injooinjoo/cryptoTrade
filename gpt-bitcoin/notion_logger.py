import logging
import os
from datetime import datetime
from notion_client import Client

logger = logging.getLogger(__name__)


class NotionLogger:
    def __init__(self):
        self.notion = Client(auth="secret_ljqYjsypmavB0RY1eYTS1vfHJHNVnBYqe7jlbidnsHS")
        self.database_id = "2eb0445c443d4031b20ffbf880f7a158"

    def log_prediction(self, timestamp, model_name, current_btc_price, predicted_price, actual_price=None):
        # try:
            if isinstance(timestamp, float):
                timestamp = datetime.fromtimestamp(timestamp)

            properties = {
                "예측시각": {"date": {"start": timestamp.isoformat()}},
                "모델명": {"select": {"name": model_name}},
                "해당시점 BTC가격": {"number": float(current_btc_price)},
                "예측한 가격": {"number": float(predicted_price)},
            }

            if actual_price is not None:
                properties["실제 (10분후) 가격"] = {"number": float(actual_price)}
                prediction_success = "맞춤" if (
                                                         predicted_price > current_btc_price and actual_price > current_btc_price) or \
                                             (predicted_price < current_btc_price and actual_price < current_btc_price) \
                    else "틀림"
                properties["최종 예측 성공"] = {"select": {"name": prediction_success}}

            self.notion.pages.create(
                parent={"database_id": self.database_id},
                properties=properties
            )
            # logger.info(f"Logged prediction for {model_name} at {timestamp}")
        # except Exception as e:
        #     logger.error(f"Error logging prediction: {e}")

    def update_actual_price(self, timestamp, model_name, actual_price):
        # try:
            if isinstance(timestamp, float):
                timestamp = datetime.fromtimestamp(timestamp)

            # 해당 예측 기록 찾기
            query = self.notion.databases.query(
                database_id=self.database_id,
                filter={
                    "and": [
                        {"property": "예측시각", "date": {"equals": timestamp.isoformat()}},
                        {"property": "모델명", "select": {"equals": model_name}}
                    ]
                }
            )

            if query["results"]:
                page_id = query["results"][0]["id"]
                current_btc_price = query["results"][0]["properties"]["해당시점 BTC가격"]["number"]
                predicted_price = query["results"][0]["properties"]["예측한 가격"]["number"]

                prediction_success = "맞춤" if (
                                                         predicted_price > current_btc_price and actual_price > current_btc_price) or \
                                             (predicted_price < current_btc_price and actual_price < current_btc_price) \
                    else "틀림"

                self.notion.pages.update(
                    page_id=page_id,
                    properties={
                        "실제 (10분후) 가격": {"number": float(actual_price)},
                        "최종 예측 성공": {"select": {"name": prediction_success}}
                    }
                )
                # logger.info(f"Updated actual price for {model_name} prediction at {timestamp}")
            # else:
            #     logger.warning(f"No matching prediction found for {model_name} at {timestamp}")
        # except Exception as e:
        #     logger.error(f"Error updating actual price: {e}")

    def get_recent_predictions(self, limit=10):
        try:
            query = self.notion.databases.query(
                database_id=self.database_id,
                sorts=[{"property": "예측시각", "direction": "descending"}],
                page_size=limit
            )

            predictions = []
            for result in query["results"]:
                prediction = {
                    "timestamp": result["properties"]["예측시각"]["date"]["start"],
                    "model_name": result["properties"]["모델명"]["select"]["name"],
                    "current_btc_price": result["properties"]["해당시점 BTC가격"]["number"],
                    "predicted_price": result["properties"]["예측한 가격"]["number"],
                }
                if "실제 (10분후) 가격" in result["properties"]:
                    prediction["actual_price"] = result["properties"]["실제 (10분후) 가격"]["number"]
                if "최종 예측 성공" in result["properties"]:
                    prediction["prediction_success"] = result["properties"]["최종 예측 성공"]["select"]["name"]
                predictions.append(prediction)

            return predictions
        except Exception as e:
            logger.error(f"Error getting recent predictions: {e}")
            return []


# 사용 예시
if __name__ == "__main__":
    logger = NotionLogger()

    # 예시 데이터
    timestamp = datetime.now()
    model_name = "GPT"
    current_btc_price = 50000
    predicted_price = 51000

    logger.log_prediction(timestamp, model_name, current_btc_price, predicted_price)

    # 10분 후
    actual_price = 51500
    logger.update_actual_price(timestamp, model_name, actual_price)