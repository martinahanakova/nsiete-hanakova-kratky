import load_data
import keras

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, LSTM, Embedding, Bidirectional


class ConvModel(keras.Model):
    def __init__(self, max_input_length, num_words, embedding_dim, regularization=0.0001, dropout=0.2):
        super(ConvModel, self).__init__()

        self.emb = Embedding(
            input_dim=num_words,
            output_dim=embedding_dim,
            input_length = max_input_length,
            trainable=True)

        self.dropout1 = Dropout(dropout)

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

        self.conv3 = Conv1D(
            filters=128,
            kernel_size=3,
            padding='same',
            activation='relu',
            kernel_regularizer=keras.regularizers.l1_l2(l1=regularization, l2=regularization))

        self.maxpooling3 = MaxPooling1D(
            pool_size=2,
            strides=2,
            padding='same')

        self.conv4 = Conv1D(
            filters=128,
            kernel_size=3,
            padding='same',
            activation='relu',
            kernel_regularizer=keras.regularizers.l1_l2(l1=regularization, l2=regularization))

        self.maxpooling4 = MaxPooling1D(
            pool_size=2,
            strides=2,
            padding='same')

        self.conv5 = Conv1D(
            filters=128,
            kernel_size=3,
            padding='same',
            activation='relu',
            kernel_regularizer=keras.regularizers.l1_l2(l1=regularization, l2=regularization))

        self.maxpooling5 = MaxPooling1D(
            pool_size=2,
            strides=2,
            padding='same')

        self.conv6 = Conv1D(
            filters=128,
            kernel_size=3,
            padding='same',
            activation='relu',
            kernel_regularizer=keras.regularizers.l1_l2(l1=regularization, l2=regularization))

        self.maxpooling6 = MaxPooling1D(
            pool_size=2,
            strides=2,
            padding='same')

        self.conv7 = Conv1D(
            filters=128,
            kernel_size=3,
            padding='same',
            activation='relu',
            kernel_regularizer=keras.regularizers.l1_l2(l1=regularization, l2=regularization))

        self.maxpooling7 = MaxPooling1D(
            pool_size=2,
            strides=2,
            padding='same')

        self.conv8 = Conv1D(
            filters=128,
            kernel_size=3,
            padding='same',
            activation='relu',
            kernel_regularizer=keras.regularizers.l1_l2(l1=regularization, l2=regularization))

        self.maxpooling8 = MaxPooling1D(
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
        self.model.add(self.conv1)
        self.model.add(self.maxpooling1)
        self.model.add(self.conv2)
        self.model.add(self.maxpooling2)
        self.model.add(self.conv3)
        self.model.add(self.maxpooling3)
        self.model.add(self.conv4)
        self.model.add(self.maxpooling4)
        self.model.add(self.conv5)
        self.model.add(self.maxpooling5)
        self.model.add(self.conv6)
        self.model.add(self.maxpooling6)
        self.model.add(self.conv7)
        self.model.add(self.maxpooling7)
        self.model.add(self.conv8)
        self.model.add(self.maxpooling8)
        self.model.add(self.flatten)
        self.model.add(self.dropout2)
        self.model.add(self.dense)
