from __future__ import print_function
import sys
import cPickle as pickle
import pandas as pd
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.models import Graph
from data_util import load_csvs

"""
following https://gist.github.com/xccds/8f0e5b0fe4eb6193261d to do 1d-CNN sentiment detection on the mr data.

Group syntax from https://github.com/fchollet/keras/issues/233 has several issues.

Will follow Kai Xiao's verbose method

- test syntax of using multiple inputs
- switch from MR to ASAP2 (one set).

1/8/2016 try to follow Kim's method
- w/o using hidden state
- dropout on the penultimate layer
- trying identical params

- only using 100 filter_size={3,4,5} filters, already obtain 0.75 after 6 epoch


"""
np.random.seed(75513)  # for reproducibility

print
"loading data..."

train_df = pd.read_csv('data/asap2/train1.csv')
test_df = pd.read_csv('data/asap2/test1.csv')

nb_words = 2900
maxlen = 75
embd_dim = 50

X_train, Y_train, X_test, Y_test, nb_classes = load_csvs('data/asap2/train1.csv',
                                                         'data/asap2/test1.csv',
                                                         nb_words, maxlen, 'self', w2v=None)

nb_filter = 100
nb_epoch = 20
batch_size = 50

print('Build model...')
ngram_filters = [3, 4, 5]
nd_convs = ['conv_'+str(n) for n in ngram_filters]
nd_pools = ['pool_'+str(n) for n in ngram_filters]
nd_flats = ['flat_'+str(n) for n in ngram_filters]

model = Graph()
model.add_input(name='input', input_shape=(maxlen,), dtype=int)

model.add_node(Embedding(nb_words, embd_dim, input_length=maxlen),
               name='embedding', input='input')
# three CNNs
for i, n_gram in enumerate(ngram_filters):
    pool_length = maxlen - n_gram + 1
    model.add_node(Convolution1D(nb_filter=nb_filter,
                                 filter_length=n_gram,
                                 border_mode="valid",
                                 activation="relu"),
                   name=nd_convs[i], input='embedding')
    model.add_node(MaxPooling1D(pool_length=pool_length),
                   name=nd_pools[i], input=nd_convs[i])
    model.add_node(Flatten(), name=nd_flats[i], input=nd_pools[i])
model.add_node(Dropout(0.5), name='dropout', inputs=nd_flats, merge_mode='concat')
model.add_node(Dense(nb_classes, activation='softmax'), name='softmax', input='dropout')
model.add_output(name='output', input='softmax')
model.compile('rmsprop', loss={'output': 'categorical_crossentropy'})  # note Graph()'s diff syntax

# early stopping
earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
model.fit({'input': X_train, 'output': Y_train},
          nb_epoch=nb_epoch, batch_size=batch_size,
          validation_split=0.1, callbacks=[earlystop])
# Graph doesn't have several arg/func existing in Sequential()
# - fit no show-accuracy
# - no predict_classes
classes = model.predict({'input': X_test}, batch_size=batch_size)['output'].argmax(axis=1)
acc = np_utils.accuracy(classes, np_utils.categorical_probas_to_classes(Y_test))  # accuracy only supports classes
print('Test accuracy:', acc)

