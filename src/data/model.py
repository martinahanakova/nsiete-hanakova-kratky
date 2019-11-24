import load_data
import keras
import datetime


dataset = load_data.AmazonReviewDataset('Office_Products_5.json')


epochs          =   10
batch_size      =   32
embedding_dim   =   32

#todo create class Model
print("creating model")

model = keras.Sequential()


model.add(keras.layers.Embedding(dataset.num_words, embedding_dim, input_length = dataset.max_input_length))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))

model.summary()


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


callbacks = [
    keras.callbacks.TensorBoard(
        log_dir="../../logs/" + timestamp(),
        histogram_freq=1)
]


model.fit(  dataset.x_train,
            dataset.y_train,
            batch_size      = batch_size,
            epochs          = epochs,
            verbose         = 1,
            validation_data = (dataset.x_test, dataset.y_test),
            callbacks=callbacks)
