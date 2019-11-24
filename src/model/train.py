import model
import keras


dataset = load_data.AmazonReviewDataset('Office_Products_5.json')


epochs          =   10
batch_size      =   32
embedding_dim   =   32

print("creating model")
model = model.ReviewTagger()

model.summary()


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


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
