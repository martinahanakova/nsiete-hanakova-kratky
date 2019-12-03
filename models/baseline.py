import load_data
import keras

from keras.layers import Dense, Flatten, Embedding


class BaselineModel(keras.Model):
    def __init__(self, max_input_length, num_words, embedding_dim, regularization=0.0001, dropout=0.2):
        super(BaselineModel, self).__init__()

        self.emb = Embedding(
            input_dim=num_words,
            output_dim=embedding_dim,
            input_length = max_input_length)
        self.flatten = Flatten()

        self.dense1 = Dense(
            units=64,
            activation='relu')

        self.dense2 = Dense(
            units=64,
            activation='relu')

        self.dense3 = Dense(
            units=3,
            activation='softmax')


        self.model = keras.Sequential()

        self.model.add(self.emb)
        self.model.add(self.flatten)
        self.model.add(self.dense1)
        self.model.add(self.dense2)
        self.model.add(self.dense3)
