import sys
sys.path.append("../data")
sys.path.append("../../models")

import model
import keras
import load_data
import baseline
import lstm
import conv
import datetime


dataset = load_data.AmazonReviewDataset('Office_Products_5.json')


epochs          =   10
batch_size      =   32
learning_rate   =   0.001

print("creating model")
#model = model.ReviewTagger(max_input_length=dataset.max_input_length, num_words=dataset.num_words, embedding_dim=64)

#model = baseline.BaselineModel(max_input_length=dataset.max_input_length, num_words=dataset.num_words, embedding_dim=64)
model = lstm.LSTMModel(max_input_length=dataset.max_input_length, num_words=dataset.num_words, embedding_dim=64)
#model = conv.ConvModel(max_input_length=dataset.max_input_length, num_words=dataset.num_words, embedding_dim=64)

model.model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.model.summary()

def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

callbacks = [
    keras.callbacks.TensorBoard(
        log_dir="../../logs/" + timestamp(),
        histogram_freq=1)
]


model.model.fit(  dataset.x_train,
            dataset.y_train,
            batch_size      = batch_size,
            epochs          = epochs,
            verbose         = 1,
            validation_data = (dataset.x_test, dataset.y_test),
            callbacks=callbacks)


score = model.model.evaluate(dataset.x_test, dataset.y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
