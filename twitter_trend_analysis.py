import numpy as np
import pandas as pd
import re
import collections
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import json


vocab_count = 10000
val_size = 1000
epochs = 6
batch_size = 512
max_tweet_len = 25

def predict(model,tweet):
    tweet = tweet.replace(":)", "happy")
    tweet = tweet.replace(":(", "sad")
    tweet = remove_stopwords(tweet)
    tweet = remove_mentions(tweet)
    tweet = [tweet.split(" ")]
    # print(tweet)
    tk = Tokenizer(num_words=vocab_count, filters='!"#$%&()*+,-./:;<=>?@[]^_`{"}~\r\t\n',lower=True, split=" ")
    tk.fit_on_texts(tweet)
    tweet = tk.texts_to_sequences(tweet)
    tweet = pad_sequences(tweet, maxlen=max_tweet_len)
    prediction = model.predict(tweet)
    # print(prediction)
    neg_prob = prediction[0][0]
    pos_prob = prediction[0][1]
    if (neg_prob > pos_prob):
        # print("Negative Sentiment")
        return -1
    else:
        # print("Positive Sentiment")
        return 1

def remove_stopwords(input_text):
    '''
    Function to remove English stopwords from a Pandas Series.
    
    Parameters:
        input_text : text to clean
    Output:
        cleaned Pandas Series 
    '''
    stopwords_list = stopwords.words('english')
    # Some words which might indicate a certain sentiment are kept via a whitelist
    whitelist = ["n't", "not", "no"]
    words = input_text.split() 
    clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
    return " ".join(clean_words) 
    
def remove_mentions(input_text):
    '''
    Function to remove mentions, preceded by @, in a Pandas Series
    
    Parameters:
        input_text : text to clean
    Output:
        cleaned Pandas Series 
    '''
    return re.sub(r'@\w+', '', input_text)


json_file = open('model/emb_model_bin.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model/model_bin.h5")
print("Loaded model from disk")

tweets_json = []
with open('data/tweets.json', 'r',encoding='utf-8') as f:
    for line in f:
        if ("{" in line):
            tweets_json.append(line)
extracted_tweets = []

for tweet in tweets_json:
    try:
        tweet_json_loaded = json.loads(tweet)
        extracted_tweets.append(tweet_json_loaded["quoted_status"]["extended_tweet"]["full_text"])
    except KeyError:
        extracted_tweets.append(tweet_json_loaded["text"])
average = 0
for t in extracted_tweets:
    average+=predict(loaded_model,t)
average = average/len(extracted_tweets)
#Above zero leans towards positive sentiment vice versa
print("Average Sentiment is:",average)
