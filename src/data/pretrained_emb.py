import os
import load_data
import numpy as np

BASE_DIR = '../..'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove')
EMBEDDING_DIM = 100

class PretrainedEmbedding():

    def __init__(self, max_num_words, word_index):

        print('Indexing word vectors.')
        embeddings_index = {}
        with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, 'f', sep=' ')
                embeddings_index[word] = coefs


        print('Preparing embedding matrix.')
        # prepare embedding matrix
        num_words = max_num_words
        embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
        for word, i in word_index.items():
            if i >= max_num_words:
                continue

            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector


        self.embedding = embedding_matrix
