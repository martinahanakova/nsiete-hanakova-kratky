from __future__ import print_function
import datetime
import os
import numpy as np
import pandas as pd
import keras
import re
import json

import tensorflow.keras as keras
from tensorboard.plugins import projector

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, LSTM, Embedding, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def load_vocabulary(X_train, X_test):
    words = set()
    train = X_train
    test = X_test

    for dataset in [train, test]:
        for sample in dataset:
            words = words.union(sample)

    #words = [re.sub(r'\"', '', str(item)) for item in words]
    #words = [re.sub(r'^\'', '', str(item)) for item in words]
    #words = [re.sub(r'\'&', '', str(item)) for item in words]

    vocabulary = {'<pad>': 0}  # Zero is reserved for padding in keras
    for i, word in enumerate(words):
        vocabulary[word] = i+1
    return vocabulary

data = pd.read_json('/content/drive/My Drive/app/dataset/Office_Products.json', lines=True)

data = data[['reviewText', 'helpful']]

data['helpful'] = data['helpful'].astype(str)
data['helpful'] = pd.DataFrame(data['helpful'].str.replace('[', ''))
data['helpful'] = pd.DataFrame(data['helpful'].str.replace(']', ''))
data['helpful'] = pd.DataFrame(data['helpful'].str.replace(' ', ''))
data['helpful_positive'], data['helpful_negative'] = data['helpful'].str.split(",").str
del data['helpful']
data['helpful'] = (data['helpful_positive'].astype(int) - data['helpful_negative'].astype(int)) / (data['helpful_positive'].astype(int) + data['helpful_negative'].astype(int))

X_train, X_test, y_train, y_test = train_test_split(data['reviewText'], data['helpful'], test_size=0.2, random_state=0)

X_train = [keras.preprocessing.text.text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ') for text in X_train]
X_test = [keras.preprocessing.text.text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ') for text in X_test]

vocab = load_vocabulary(X_train, X_test)

max_sample_length = 0
for sample in X_train:
    max_sample_length = len(sample) if max_sample_length < len(sample) else max_sample_length
for sample in X_test:
    max_sample_length = len(sample) if max_sample_length < len(sample) else max_sample_length

x_train = [np.pad([vocab[word] for word in sample], (0, max_sample_length), 'constant').tolist()[:max_sample_length]
           for sample in X_train]

x_test = [np.pad([vocab[word] for word in sample], (0, max_sample_length), 'constant').tolist()[:max_sample_length]
          for sample in X_test]

batch_size = 128
num_classes = 10
epochs = 12

class ReviewTagger(keras.Model):
    def __init__(self):
        super(ReviewTagger, self).__init__()

        self.emb = Embedding(
            input_dim=len(vocab),
            output_dim=300,
            trainable=True,
            mask_zero=True)

        self.lstm = Bidirectional(LSTM(
            units=300,
            dropout=0.2))

        self.conv = Conv2D(
            filters=100,
            kernel_size=3,
            padding='same',
            activation='relu')

        self.maxpooling = MaxPooling2D(
            pool_size=(2, 2),
            strides=(1,1),
            padding='same')

        self.dense = Dense(
            units=2,
            activation='softmax')

    def call(self, x):
        x = self.emb(x)
        x = self.lstm(x)
        x = self.conv(x)
        x = self.maxpooling(x)
        x = self.dense(x)
        return x

model = ReviewTagger()

model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])

callbacks = [
    keras.callbacks.TensorBoard(
        log_dir="/content/drive/My Drive/app/logs/" + timestamp(),
        histogram_freq=1)
]

model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test),
            callbacks=callbacks)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
