import load_data
import keras

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, LSTM, Embedding, Bidirectional
from keras.initializers import Constant


class ReviewTagger(keras.Model):
    def __init__(self, embedding, max_input_length, num_words, embedding_dim, regularization=0.0001, dropout=0.2):
        super(ReviewTagger, self).__init__()

        self.emb = Embedding(
            input_dim=num_words,
            output_dim=embedding_dim,
            input_length=max_input_length,
            embeddings_initializer=Constant(embedding),
            trainable=False)

        self.dropout1 = Dropout(dropout)

        self.lstm = Bidirectional(LSTM(
            input_shape=(max_input_length, embedding_dim),
            units=64,
            return_sequences=True,
            kernel_regularizer=keras.regularizers.l1_l2(l1=regularization, l2=regularization)))

        self.conv1 = Conv1D(
            filters=128,
            kernel_size=3,
            padding='same',
            activation='relu',
            kernel_regularizer=keras.regularizers.l1_l2(l1=regularization, l2=regularization))

        self.maxpooling1 = MaxPooling1D(
            pool_size=2,
            strides=2,
            padding='same')

        self.conv2 = Conv1D(
            filters=128,
            kernel_size=3,
            padding='same',
            activation='relu',
            kernel_regularizer=keras.regularizers.l1_l2(l1=regularization, l2=regularization))

        self.maxpooling2 = MaxPooling1D(
            pool_size=2,
            strides=2,
            padding='same')

        self.flatten = Flatten()

        self.dropout2 = Dropout(dropout)

        self.dense = Dense(
            units=3,
            activation='softmax',
            kernel_regularizer=keras.regularizers.l1_l2(l1=regularization, l2=regularization))


        self.model = keras.Sequential()

        self.model.add(self.emb)
        self.model.add(self.dropout1)
        self.model.add(self.lstm)
        self.model.add(self.conv1)
        self.model.add(self.maxpooling1)
        self.model.add(self.conv2)
        self.model.add(self.maxpooling2)
        self.model.add(self.flatten)
        self.model.add(self.dropout2)
        self.model.add(self.dense)
