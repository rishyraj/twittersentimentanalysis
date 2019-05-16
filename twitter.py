from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy import Cursor
from tweepy import API
from tweepy import Status

import json
import pandas as pd
import numpy as np
#create a twitter_credentials file with api keys
import twitter_credentials
 
 # # # # TWITTER STREAMER # # # #
class TwitterClient():
    def __init__(self):
        auth = OAuthHandler(twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_KEY_SECRET)
        auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)
        self.twitter_client = API(auth)

    def get_twitter_client_api(self):
        return self.twitter_client

    def get_tweets(self, num_tweets):
        tweets = []
        for tweet in Cursor(self.twitter_client.search, q='Donald Trump', tweet_mode='extended').items(num_tweets):
            tweets.append(tweet)
        return tweets
# # # # TWITTER STREAMER # # # #
class TwitterStreamer():
    """
    Class for streaming and processing live tweets.
    """
    def __init__(self):
        pass

    def stream_tweets(self, fetched_tweets_filename, hash_tag_list):
        # This handles Twitter authentication and the connection to Twitter Streaming API
        listener = StdOutListener(fetched_tweets_filename)
        auth = OAuthHandler(twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_KEY_SECRET)
        auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)
        stream = Stream(auth, listener)

        # This line filter Twitter Streams to capture data by the keywords: 
        stream.filter(languages=["en"],track=hash_tag_list)


# # # # TWITTER STREAM LISTENER # # # #
class StdOutListener(StreamListener):
    """
    This is a basic listener that just prints received tweets to stdout.
    """
    def __init__(self, fetched_tweets_filename):
        self.fetched_tweets_filename = fetched_tweets_filename

    def on_data(self, data):
        try:
            print(data)
            with open(self.fetched_tweets_filename, 'a') as tf:
                tf.write(data)
            return True
        except BaseException as e:
            print("Error on_data %s" % str(e))
        return True
          

    def on_error(self, status):
        print(status)

 
class TweetAnalyzer():
    #Functionality for analyzing and categorizing content from tweets.
    
    def tweets_to_data_frame(self, tweets):
        df = pd.DataFrame(data=[tweet.full_text for tweet in tweets],columns=['Tweets'])
        df['id'] = np.array([tweet.id for tweet in tweets])
        df['date'] = np.array([tweet.created_at for tweet in tweets])
        return df


if __name__ == '__main__':
 
    # Authenticate using config.py and connect to Twitter Streaming API.
    hash_tag_list = ["United States","America"]
    fetched_tweets_filename = "data/tweets.json"

    twitter_streamer = TwitterStreamer()
    twitter_streamer.stream_tweets(fetched_tweets_filename, hash_tag_list)
    twitter_client = TwitterClient()
    # tweets = twitter_client.get_tweets(10)
    # api = twitter_client.get_twitter_client_api()

    # tweets = api.search(q=u"\U0001F602", count=10)
    # tweet_analyzer = TweetAnalyzer()
    # df = tweet_analyzer.tweets_to_data_frame(tweets)
    # print(df.head(10))
    # print(dir(tweets[0]))
    # for tweet in tweets:
    #     print(tweet.full_text)
    # tweets_fetched = twitter_client.get_tweets(20)
