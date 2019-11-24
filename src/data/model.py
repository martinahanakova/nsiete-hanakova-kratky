import dataset
import keras


dataset = dataset.AmazonReviewDataset('Office_Products_5.json')




epochs          =   10
batch_size      =   32
embedding_dim   =   256

#todo create class Model
print("creating model")

model = keras.Sequential()


model.add(keras.layers.Input(shape=(256, )))
model.add(keras.layers.Embedding(1000, embedding_dim))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(3))

model.summary()


model.compile(  loss        = keras.losses.mean_squared_error,
                optimizer   = keras.optimizers.Adam(),
                metrics     = ['mae'])

'''
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir="/content/drive/My Drive/app/logs/" + timestamp(),
        histogram_freq=1)
]
'''


model.fit(  dataset.x_train,
            dataset.y_train,
            batch_size      = batch_size,
            epochs          = epochs,
            verbose         = 1,
            validation_data = (dataset.x_test, dataset.y_test))
