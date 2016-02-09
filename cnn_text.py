"""
This example demonstrates the use of Convolution1D for text classification.

Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_cnn.py

Get to 0.835 test accuracy after 2 epochs. 100s/epoch on K520 GPU.
==============
cnn_text.py

- test the idea of jointly using two groups of features (word embedding + #char)
- got an error
  TypeError: ('Bad input argument to theano function with name "python2.7/site-packages/keras/backend/theano_backend.py:362"
  at index 1(0-based)', 'Wrong number of dimensions: expected 2, got 1 with shape (32,).')

"""

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb

# set parameters:
max_features = 5000
maxlen = 100
batch_size = 32
embedding_dims = 100
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 2

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features,
                                                      test_split=0.2)
len_train = np.array([len(sent) for sent in X_train], dtype='float32')
len_test = np.array([len(sent) for sent in X_test], dtype='float32')

len_train = len_train.reshape(len_train.shape[0], 1)
len_test = len_test.reshape(len_test.shape[0], 1)
print('len_train:', len_train.shape)


print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
model.add(Dropout(0.25))

# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
# we use standard max pooling (halving the output of the previous layer):
model.add(MaxPooling1D(pool_length=2))

# We flatten the output of the conv layer,
# so that we can add a vanilla dense layer:
model.add(Flatten())
first = model

# single #char feature.
second = Sequential()
second.add(Dense(1, input_dim=1))
second.add(Activation('linear'))

# using Merge
model = Sequential()
model.add(Merge([first, second], mode='concat'))

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.25))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              class_mode='binary')

model.fit([X_train, len_train], y_train, batch_size=batch_size,
          nb_epoch=nb_epoch, show_accuracy=True,
          validation_data=([X_test, len_test], y_test))
