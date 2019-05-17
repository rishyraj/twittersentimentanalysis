# twittersentimentanalysis

Tweet training data for keras model : https://www.kaggle.com/kazanova/sentiment140


Preprocessing text code based off this tutorial: https://towardsdatascience.com/word-embeddings-for-sentiment-analysis-65f42ea5d26e

Purpose of each python program:
twitter_classification_sentiment.py: This file generates the Keras Deep Learning Model and saves it in the model directory
twitter.py: This file streams live tweets based on a tag and stores the tweets in the data directory
twitter_trend_analysis: This file gives the average sentiment of the tweets from the twitter.py stream, using the keras model made earlier.