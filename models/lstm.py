import load_data
import keras

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, LSTM, Embedding, Bidirectional


class LSTMModel(keras.Model):
    def __init__(self, max_input_length, num_words, embedding_dim, regularization=0.0001, dropout=0.2):
        super(LSTMModel, self).__init__()

        self.emb = Embedding(
            input_dim=num_words,
            output_dim=embedding_dim,
            input_length = max_input_length,
            trainable=True)

        self.dropout1 = Dropout(dropout)

        self.lstm1 = Bidirectional(LSTM(
            input_shape=(max_input_length, embedding_dim),
            units=64,
            return_sequences=True,
            kernel_regularizer=keras.regularizers.l1_l2(l1=regularization, l2=regularization)))

        self.lstm2 = Bidirectional(LSTM(
            input_shape=(max_input_length, embedding_dim),
            units=64,
            return_sequences=True,
            kernel_regularizer=keras.regularizers.l1_l2(l1=regularization, l2=regularization)))

        self.flatten = Flatten()

        self.dropout2 = Dropout(dropout)

        self.dense = Dense(
            units=3,
            activation='softmax',
            kernel_regularizer=keras.regularizers.l1_l2(l1=regularization, l2=regularization))


        self.model = keras.Sequential()

        self.model.add(self.emb)
        self.model.add(self.dropout1)
        self.model.add(self.lstm1)
        self.model.add(self.lstm2)
        self.model.add(self.flatten)
        self.model.add(self.dropout2)
        self.model.add(self.dense)
