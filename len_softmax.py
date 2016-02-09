"""
This example demonstrates the use of Convolution1D for text classification.

Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_cnn.py

Get to 0.835 test accuracy after 2 epochs. 100s/epoch on K520 GPU.
==============
len_softmax.py

- only using #char softmax to figure out how to feed features

"""

from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.datasets import imdb

# set parameters:
max_features = 5000
batch_size = 64
nb_epoch = 2

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features,
                                                      test_split=0.2)

len_train = np.array([len(sent) for sent in X_train], dtype='float32')
len_test = np.array([len(sent) for sent in X_test], dtype='float32')

len_train = len_train.reshape(len_train.shape[0], 1)
len_test = len_test.reshape(len_test.shape[0], 1)
print('len_train:', len_train.shape)


print('Build model...')
second = Sequential()
second.add(Dense(output_dim=1, activation='sigmoid', input_dim=1))

second.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              class_mode='binary')

second.fit(len_train, y_train, batch_size=batch_size,
          nb_epoch=nb_epoch, show_accuracy=True,
          validation_data=(len_test, y_test))
