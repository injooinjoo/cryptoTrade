import logging
from typing import Dict, Any

import requests
import tweepy
from textblob import TextBlob

logger = logging.getLogger(__name__)


class AdditionalDataFetcher:
    def __init__(self, twitter_api_key: str, twitter_api_secret: str, twitter_access_token: str, twitter_access_token_secret: str):
        self.fear_greed_url = "https://api.alternative.me/fng/"
        self.blockchain_info_url = "https://api.blockchain.info/stats"

        # Twitter API 설정
        auth = tweepy.OAuthHandler(twitter_api_key, twitter_api_secret)
        auth.set_access_token(twitter_access_token, twitter_access_token_secret)
        self.twitter_api = tweepy.API(auth)

    def get_fear_greed_index(self) -> Dict[str, Any]:
        try:
            response = requests.get(self.fear_greed_url, timeout=10)
            response.raise_for_status()
            data = response.json()['data'][0]
            return {
                'value': int(data['value']),
                'classification': data['value_classification']
            }
        except requests.RequestException as e:
            logger.error(f"Network error in fetching Fear & Greed Index: {e}")
        except (KeyError, ValueError) as e:
            logger.error(f"Data parsing error in Fear & Greed Index: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in fetching Fear & Greed Index: {e}")
        return {'value': None, 'classification': None}

    def get_blockchain_data(self) -> Dict[str, Any]:
        try:
            response = requests.get(self.blockchain_info_url)
            response.raise_for_status()
            data = response.json()
            return {
                'transactions_per_second': data['transactions_per_second'],
                'mempool_size': data['mempool_size'],
                'difficulty': data['difficulty'],
                'hash_rate': data['hash_rate']
            }
        except Exception as e:
            logger.error(f"Error fetching blockchain data: {e}")
            return {
                'transactions_per_second': None,
                'mempool_size': None,
                'difficulty': None,
                'hash_rate': None
            }

    def get_twitter_sentiment(self) -> Dict[str, float]:
        try:
            tweets = self.twitter_api.search_tweets(q="bitcoin", count=100, lang="en")
            sentiments = [TextBlob(tweet.text).sentiment.polarity for tweet in tweets]
            positive_tweets = sum(1 for s in sentiments if s > 0) / len(sentiments)
            negative_tweets = sum(1 for s in sentiments if s < 0) / len(sentiments)
            return {
                'average_sentiment': sum(sentiments) / len(sentiments),
                'positive_tweets': positive_tweets,
                'negative_tweets': negative_tweets
            }
        except Exception as e:
            logger.error(f"Error fetching Twitter sentiment: {e}")
            return {
                'average_sentiment': 0,
                'positive_tweets': 0,
                'negative_tweets': 0
            }

    def fetch_all_data(self) -> Dict[str, Any]:
        return {
            'fear_greed_index': self.get_fear_greed_index(),
            'blockchain_data': self.get_blockchain_data(),
            'twitter_sentiment': self.get_twitter_sentiment()
        }
