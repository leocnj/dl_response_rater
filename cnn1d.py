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
from keras.utils import np_utils
from keras.layers.recurrent  import SimpleRNN, GRU, LSTM
from keras.regularizers import l2

from data_util import load_asap, load_sg15, load_mr
import cPickle as pickle

"""
   following https://gist.github.com/xccds/8f0e5b0fe4eb6193261d
   to do 1d-CNN on different text classification tasks

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

    # score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1, show_accuracy=True)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])
    score = earlystop.model.evaluate(X_test, Y_test, batch_size=batch_size)
    # earlystop only returns one score val. 1/6/2016
    print('Test score:', score)
    # print('Test accuracy:', score[1])
    classes = earlystop.model.predict_classes(X_test, batch_size=batch_size)
    acc = np_utils.accuracy(classes, np_utils.categorical_probas_to_classes(Y_test)) # accuracy only supports classes
    print('Test accuracy:', acc)


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
    #pool_length = maxlen - filter_length + 1
    pool_length = 10

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
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    # model.add(Dense(nb_classes, W_regularizer=l2(3)))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optm)

    earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_split=0.1, show_accuracy=True, callbacks=[earlystop])

    score = earlystop.model.evaluate(X_test, Y_test, batch_size=batch_size)
    # earlystop only returns one score val. 1/6/2016
    # score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1, show_accuracy=True)
    print('Test score:', score)
    # print('Test accuracy:', score[1])
    classes = earlystop.model.predict_classes(X_test, batch_size=batch_size)
    acc = np_utils.accuracy(classes, np_utils.categorical_probas_to_classes(Y_test)) # accuracy only supports classes
    print('Test accuracy:', acc)


def lstm_selfembd(X_train, Y_train, X_test, Y_test, nb_classes,
                   maxlen, vocab_size, embd_dim,
                   hidden_dims, batch_size, nb_epoch, optm):
    """
    - LSTM  on text input (represented in int)
    - fully-connected model

    :param <X, Y> train and test sets
    :param nb_classes # of classes
    :param maxlen max of n char in a sentence
    :param vocab_size
    :param embd_dim
    :param hidden_dims
    :param batch_size
    :param nb_epoch
    :param optm optimizer options, e.g., adam, rmsprop, etc.
    :return:
    """

    model = Sequential()
    model.add(Embedding(vocab_size, embd_dim, input_length=maxlen))
    model.add(Dropout(0.25))

    model.add(LSTM(100))

    model.add(Flatten())
    #model.add(Dense(hidden_dims))
    model.add(Dropout(0.5))
    #model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optm)

    earlystop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_split=0.1, show_accuracy=True, callbacks=[earlystop])

    score = earlystop.model.evaluate(X_test, Y_test, batch_size=batch_size)
    # earlystop only returns one score val. 1/6/2016
    # score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1, show_accuracy=True)
    print('Test score:', score)
    # print('Test accuracy:', score[1])
    classes = earlystop.model.predict_classes(X_test, batch_size=batch_size)
    acc = np_utils.accuracy(classes, np_utils.categorical_probas_to_classes(Y_test)) # accuracy only supports classes
    print('Test accuracy:', acc)



def test_asap():
    nb_words = 5000
    maxlen = 150
    embd_dim = 100
    X_train, Y_train, X_test, Y_test, nb_classes = load_asap(nb_words, maxlen, 'self')
    cnn1d_selfembd(X_train, Y_train, X_test, Y_test, nb_classes,
                   maxlen, nb_words, embd_dim,
                   100, 5, 100, 32, 20, 'rmsprop')

def test_sg15_lstm():
    nb_words = 10000 # for NNS speakers, should be sufficient
    maxlen = 200
    embd_dim = 100
    X_train, Y_train, X_test, Y_test, nb_classes = load_sg15(nb_words, maxlen, 'self')
    lstm_selfembd(X_train, Y_train, X_test, Y_test, nb_classes,
                   maxlen, nb_words, embd_dim,
                   50, 32, 20, 'rmsprop')


def test_sg15():
    nb_words = 10000
    maxlen = 200
    embd_dim = 100
    X_train, Y_train, X_test, Y_test, nb_classes = load_sg15(nb_words, maxlen, 'self')
    cnn1d_selfembd(X_train, Y_train, X_test, Y_test, nb_classes,
                   maxlen, nb_words, embd_dim,
                   100, 15, 50, 32, 20, 'rmsprop')

def test_sg15_w2v():
    maxlen = 200
    # embd_dim = 100
    X_train, Y_train, X_test, Y_test, nb_classes = load_sg15(0, maxlen, 'w2v')
    cnn1d_w2vembd(X_train, Y_train, X_test, Y_test, nb_classes,
                   maxlen,
                   100, 15, 50, 32, 20, 'rmsprop')


def test_mr_embd():
    nb_words = 50000
    maxlen = 100
    embd_dim = 100
    X_train, Y_train, X_test, Y_test, nb_classes = load_mr(nb_words, maxlen, 'self')
    cnn1d_selfembd(X_train, Y_train, X_test, Y_test, nb_classes,
                   maxlen, nb_words, embd_dim,
                   100, 5, 100, 32, 20, 'rmsprop')


def test_mr_w2v():
    maxlen = 64
    X_train, Y_train, X_test, Y_test, nb_classes = load_mr(0, maxlen, 'w2v')
    cnn1d_w2vembd(X_train, Y_train, X_test, Y_test, nb_classes,
                   maxlen,
                   100, 5, 100, 32, 20, 'adam') # only adam can move ACC to about 70%.


if __name__ == "__main__":

    np.random.seed(1337)  # for reproducibility

    print('='*50)
    print('sg15 self CNN')
    test_sg15()

    print('='*50)
    print('sg15 self LSTM')
    test_sg15_lstm()

    print('='*50)
    print('mr self')
    #test_mr_embd()
