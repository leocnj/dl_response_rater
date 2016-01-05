from __future__ import print_function
import sys

import pandas as pd
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.callbacks import EarlyStopping

from data_util import load_pd, df_2_embd, df_2_3dtensor, load_asap

"""
following https://gist.github.com/xccds/8f0e5b0fe4eb6193261d
to do 1d-CNN sentiment detection on the mr data.

- depending on type
  w2vembd using word2vec pre-trained embd (dim=300)
  selfembd training an embd directl from data, vocab_size (5000) and embd_dim (100)
- 1d-CNN and then Max-Over-Time (MOT)

"""

def cnn1d_w2vembd(X_train, Y_train, X_test, Y_test, nb_classes,
                  maxlen,
                  nb_filter, filter_length, hidden_dims, batch_size, nb_epoch, optm):
    """
    - CNN-1d on 3d sensor which uses word2vec embedding
    - MOT
    - fully-connected model

    :param <X, Y> train and test sets
    :param nb_classes # of classes
    :param maxlen max of n char in a sentence
    :param nb_filter
    :param filter_length
    :param hidden_dims
    :param batch_size
    :param nb_epoch
    :param optm
    :return:
    """
    pool_length = maxlen - filter_length + 1

    model = Sequential()

    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode="valid",
                            activation="relu", input_shape=(maxlen, 300)))
    model.add(MaxPooling1D(pool_length=pool_length))

    model.add(Flatten())
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optm)

    earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_split=0.1, show_accuracy=True, callbacks=[earlystop])

    score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1, show_accuracy=True)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def cnn1d_selfembd(X_train, Y_train, X_test, Y_test, nb_classes,
                   maxlen, vocab_size, embd_dim,
                   nb_filter, filter_length, hidden_dims, batch_size, nb_epoch, optm):
    """
    - CNN-1d on text input (represented in int)
    - MOT
    - fully-connected model

    :param <X, Y> train and test sets
    :param nb_classes # of classes
    :param maxlen max of n char in a sentence
    :param vocab_size
    :param embd_dim
    :param nb_filter
    :param filter_length
    :param hidden_dims
    :param batch_size
    :param nb_epoch
    :param optm optimizer options, e.g., adam, rmsprop, etc.
    :return:
    """
    pool_length = maxlen - filter_length + 1

    model = Sequential()
    model.add(Embedding(vocab_size, embd_dim, input_length=maxlen))
    model.add(Dropout(0.25))

    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode="valid",
                            activation="relu"))
    model.add(MaxPooling1D(pool_length=pool_length))

    model.add(Flatten())
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optm)

    earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_split=0.1, show_accuracy=True, callbacks=[earlystop])

    score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1, show_accuracy=True)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def test_mr_embd():
    maxlen = 100
    vocab_size = 20000
    embd_dim = 100
    X_train, Y_train, X_test, Y_test, nb_classes = df_2_embd(load_pd('data/mr.p'), maxlen, vocab_size)
    cnn1d_selfembd(X_train, Y_train, X_test, Y_test, nb_classes,
                   maxlen, vocab_size, embd_dim,
                   100, 5, 100, 32, 20, 'adadelta')

def test_mr_w2v():
    maxlen = 100
    X_train, Y_train, X_test, Y_test, nb_classes = df_2_3dtensor(load_pd('data/mr.p'), maxlen)
    cnn1d_w2vembd(X_train, Y_train, X_test, Y_test, nb_classes,
                   maxlen,
                   100, 5, 100, 32, 20, 'adam') # only adam can move ACC to about 70%.


if __name__ == "__main__":
    # test_mr_selfembd()
    #
    # five epoch, each 5 sec (CPU version will be 450 sec), ACC is 75.75%
    #
    print('='*50)
    # test_mr_embd()
    test_mr_w2v()
