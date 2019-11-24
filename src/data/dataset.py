import pandas as pd
import keras
import numpy as np

from sklearn.model_selection        import train_test_split
from keras.preprocessing.sequence   import pad_sequences


import re
import keras

class AmazonReviewDataset:

    def __init__(self, dataset_file_name, test_ratio = 0.2, num_words = 1000, max_input_length = 256):

        print("loading from ", dataset_file_name)
        data = pd.read_json(dataset_file_name, lines=True)

        #with open(dataset_file_name) as json_file:
        #    data = json.load(json_file, lines=True)

        data = data[['reviewText', 'helpful']]

        data['helpful'] = data['helpful'].astype(str)
        data['helpful'] = pd.DataFrame(data['helpful'].str.replace('[', ''))
        data['helpful'] = pd.DataFrame(data['helpful'].str.replace(']', ''))
        data['helpful'] = pd.DataFrame(data['helpful'].str.replace(' ', ''))
        data['helpful_positive'], data['helpful_negative'] = data['helpful'].str.split(",").str
        del data['helpful']
        data['helpful'] = (data['helpful_positive'].astype(int) - data['helpful_negative'].astype(int)) / (data['helpful_positive'].astype(int) + data['helpful_negative'].astype(int))

        print("preprocessing")
        x_sen = []
        for raw_sen in data["reviewText"]:
            x_sen.append(self.preprocess_text(raw_sen))

        print("creating target")
        y_target = []

        for i in range(len(data["reviewText"])):
            np.zeros(3)
            #print(">>> ", int(data['helpful_positive'][i]), int(data['helpful_negative'][i]))
            if int(data['helpful_positive'][i]) == int(data['helpful_negative'][i]):
                v = [1.0, 0.0, 0.0]
            else:
                if int(data['helpful_positive'][i]) > int(data['helpful_negative'][i]):
                    v = [0.0, 1.0, 0.0]
                else:
                    v = [0.0, 0.0, 1.0]

            y_target.append(v)


        tokenizer = keras.preprocessing.text.Tokenizer(num_words)
        tokenizer.fit_on_texts(x_sen)

        x = tokenizer.texts_to_sequences(x_sen)
        x = pad_sequences(x, padding = 'post', maxlen = max_input_length)

        x           = np.asarray(x, dtype=np.float32)
        y_target    = np.asarray(y_target, dtype=np.float32)

        x = x.reshape(-1,)
        y_target = y_target.reshape(-1,)



        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y_target, test_size=test_ratio, random_state=0)
        self.num_words = num_words

        print("shape = ", self.x_train.shape, self.y_train.shape)

    def remove_tags(self, text):
        TAG_RE = re.compile(r'<[^>]+>')
        return TAG_RE.sub('', text)

    def preprocess_text(self, raw_sen):
        # Removing html tags
        sentence = self.remove_tags(raw_sen)

        # Remove punctuations and numbers
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)

        # Single character removal
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

        # Removing multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)

        return sentence


#dataset = AmazonReviewDataset('Office_Products_5.json')
