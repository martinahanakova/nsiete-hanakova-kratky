import load_data
import keras

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, LSTM, Embedding, Bidirectional


class ReviewTagger(keras.Model):
    def __init__(self):
        super(ReviewTagger, self).__init__()

        self.emb = Embedding(
            dataset.num_words,
            embedding_dim,
            input_length = dataset.max_input_length,
            output_dim=300)

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

model = keras.Sequential()


model.add(keras.layers.Embedding(dataset.num_words, embedding_dim, input_length = dataset.max_input_length))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))
