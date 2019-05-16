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

import sys

vocab_count = 10000
val_size = 1000
epochs = 15
batch_size = 512
max_tweet_len = 25

def deep_model(model, x_train, y_train, x_valid, y_valid):
    '''
    Function to train a multi-class model. The number of epochs and 
    batch_size are set by the constants at the top of the
    notebook. 
    
    Parameters:
        model : model with the chosen architecture
        X_train : training features
        y_train : training target
        X_valid : validation features
        Y_valid : validation target
    Output:
        model training history
    '''
    model.compile(optimizer='rmsprop'
                  , loss='categorical_crossentropy'
                  , metrics=['accuracy'])
    
    history = model.fit(x_train
                       , y_train
                       , epochs=epochs
                       , batch_size=batch_size
                       , validation_data=(x_valid, y_valid)
                       , verbose=1)
    return history
def eval_metric(history, metric_name):
    '''
    Function to evaluate a trained model on a chosen metric. 
    Training and validation metric are plotted in a
    line chart for each epoch.
    
    Parameters:
        history : model training history
        metric_name : loss or accuracy
    Output:
        line chart with epochs of x-axis and metric on
        y-axis
    '''
    metric = history.history[metric_name]
    val_metric = history.history['val_' + metric_name]

    e = range(1, epochs + 1)

    plt.plot(e, metric, 'bo', label='Train ' + metric_name)
    plt.plot(e, val_metric, 'b', label='Validation ' + metric_name)
    plt.legend()
    plt.show()

def test_model(model, x_train, y_train, x_test, y_test, epoch_stop):
    '''
    Function to test the model on new data after training it
    on the full training data with the optimal number of epochs.
    
    Parameters:
        model : trained model
        X_train : training features
        y_train : training target
        X_test : test features
        y_test : test target
        epochs : optimal number of epochs
    Output:
        test accuracy and test loss
    '''
    model.fit(x_train
              , y_train
              , epochs=epoch_stop
              , batch_size=batch_size
              , verbose=0)
    results = model.evaluate(x_test, y_test)
    
    return results

def predict(model,tweet):
    tweet = remove_stopwords(tweet)
    tweet = remove_mentions(tweet)
    tweet = [tweet.split(" ")]
    # print(tweet)
    tk = Tokenizer(num_words=vocab_count, filters='!"#$%&()*+,-./:;<=>?@[]^_`{"}~\r\t\n',lower=True, split=" ")
    tk.fit_on_texts(tweet)
    tweet = tk.texts_to_sequences(tweet)
    tweet = pad_sequences(tweet, maxlen=max_tweet_len)
    prediction = model.predict(tweet)
    neg_prob = prediction[0][0]
    pos_prob = prediction[0][4]
    if (neg_prob > pos_prob):
        print("Negative Sentiment")
    else:
        print("Positive Sentiment")

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


# df = pd.read_csv('data/tweets_dataset.csv',names=["sentiment","id","date","flag","user","text"],encoding='ISO-8859-1')
# df = df.reindex(np.random.permutation(df.index))
# df = df.drop(df.index[1000:])
# df = df[['text', 'sentiment']]
# df.text = df.text.apply(remove_stopwords).apply(remove_mentions)
# df['sentiment'].replace(to_replace=4,value=1)
# x_train, x_test, y_train, y_test = train_test_split(df.text, df.sentiment, test_size=0.1,random_state=22)

# tk = Tokenizer(num_words=vocab_count, filters='!"#$%&()*+,-./:;<=>?@[]^_`{"}~\r\t\n',lower=True, split=" ")
# tk.fit_on_texts(x_train)
# Creating "vectors" for the tweets
# x_train_seq = tk.texts_to_sequences(x_train)
# x_test_seq = tk.texts_to_sequences(x_test)

# print(list(x_train_seq)[0])

# x_train_seq_trunc = pad_sequences(x_train_seq, maxlen=max_tweet_len)
# x_test_seq_trunc = pad_sequences(x_test_seq, maxlen=max_tweet_len)

# y_train_conv = to_categorical(y_train)
# y_test_conv = to_categorical(y_test)

# x_train_emb, x_valid_emb, y_train_emb, y_valid_emb = train_test_split(x_train_seq_trunc, y_train_conv, test_size=0.1, random_state=22)

# emb_model = models.Sequential()
# emb_model.add(layers.Embedding(vocab_count,8,input_length=max_tweet_len))
# emb_model.add(layers.Flatten())
# emb_model.add(layers.Dense(5, activation='softmax'))
# emb_history = deep_model(emb_model, x_train_emb, y_train_emb, x_valid_emb, y_valid_emb)
# eval_metric(emb_history,'loss')
# emb_results = test_model(emb_model, x_train_seq_trunc, y_train_conv, x_test_seq_trunc, y_test_conv, 6)
# print('/n')
# print('Test accuracy of word embeddings model: {0:.2f}%'.format(emb_results[1]*100))

# serialize model to JSON
# model_json = emb_model.to_json()
# with open("model/emb_model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# emb_model.save_weights("model/model.h5")
# print("Saved model to disk")

# later...

json_file = open('model/emb_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model/model.h5")
# print("Loaded model from disk")

# predict(loaded_model,"Recall DT for being an immature and unfit leader.")
# predict(loaded_model, "What a great day outside, so nice and warm.")