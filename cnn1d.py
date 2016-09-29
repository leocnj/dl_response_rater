from __future__ import print_function
import sys

import pandas as pd
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, merge
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import SimpleRNN, GRU, LSTM
from keras.regularizers import l2

from data_util import load_mr
import cPickle as pickle

import ml_metrics as metrics

"""
   following https://gist.github.com/xccds/8f0e5b0fe4eb6193261d
   to do 1d-CNN on different text classification tasks

"""


def cnn1d_w2vembd(X_train, Y_train, X_test, Y_test, nb_classes,
                  maxlen,
                  nb_filter, filter_length, batch_size, nb_epoch, optm):
    """
    - CNN-1d on 3d sensor which uses word2vec embedding
    - MOT

    :param <X, Y> train and test sets
    :param nb_classes # of classes
    :param maxlen max of n char in a sentence
    :param nb_filter
    :param filter_length
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
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optm)

    earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_split=0.1, show_accuracy=True, callbacks=[earlystop])

    classes = earlystop.model.predict_classes(X_test, batch_size=batch_size)
    acc = np_utils.accuracy(classes, np_utils.categorical_probas_to_classes(Y_test))  # accuracy only supports classes
    print('Test accuracy:', acc)
    # return(acc)
    kappa = metrics.quadratic_weighted_kappa(classes, np_utils.categorical_probas_to_classes(Y_test))
    print('Test Kappa:', kappa)
    return (kappa)


def qw_kappa_loss(y_true, y_pred):
    return metrics.quadratic_weighted_kappa(y_true, y_pred)


def cnn1d_selfembd(X_train, Y_train, X_test, Y_test, nb_classes,
                   maxlen, vocab_size, embd_dim,
                   nb_filter, filter_length, batch_size, nb_epoch, optm):
    """
    - CNN-1d on text input (represented in int)
    - MOT
    - dropout + L2 softmax

    :param <X, Y> train and test sets
    :param nb_classes # of classes
    :param maxlen max of n char in a sentence
    :param vocab_size
    :param embd_dim
    :param nb_filter
    :param filter_length
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
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optm)

    earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_split=0.1, show_accuracy=True, callbacks=[earlystop])

    classes = earlystop.model.predict_classes(X_test, batch_size=batch_size)
    acc = np_utils.accuracy(classes, np_utils.categorical_probas_to_classes(Y_test))
    print('Test accuracy:', acc)
    # return(acc)
    kappa = metrics.quadratic_weighted_kappa(classes, np_utils.categorical_probas_to_classes(Y_test))
    print('Test Kappa:', kappa)
    return (kappa)


def cnn_var_selfembd(X_train, Y_train, X_test, Y_test, nb_classes,
                     maxlen, vocab_size, embd_size,
                     nb_filter, batch_size, nb_epoches, optm):
    ngram_filters = [2, 5, 8]

    input = Input(shape=(maxlen,), name='input', dtype='int32')
    embedded = Embedding(input_dim=vocab_size, output_dim=embd_size, input_length=maxlen)(input)

    convs = [None, None, None]
    # three CNNs
    for i, n_gram in enumerate(ngram_filters):
        pool_length = maxlen - n_gram + 1
        convs[i] = Convolution1D(nb_filter=nb_filter,
                                 filter_length=n_gram,
                                 border_mode="valid",
                                 activation="relu")(embedded)
        convs[i] = MaxPooling1D(pool_length=pool_length)(convs[i])
        convs[i] = Flatten()(convs[i])

    merged = merge([convs[0], convs[1], convs[2]], mode='concat', concat_axis=1)
    merged = Dropout(0.5)(merged)
    output = Dense(nb_classes, activation='softmax', name='output')(merged)

    model = Model(input, output)
    model.compile(optm, loss={'output': 'categorical_crossentropy'})
    earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
    model.fit(X_train, Y_train,
              nb_epoch=nb_epoches, batch_size=batch_size,
              validation_split=0.1, callbacks=[earlystop])

    probs = earlystop.model.predict(X_test, batch_size=batch_size)
    classes = np_utils.categorical_probas_to_classes(probs)

    acc = np_utils.accuracy(classes,
                            np_utils.categorical_probas_to_classes(Y_test))
    print('Test accuracy:', acc)
    kappa = metrics.quadratic_weighted_kappa(classes,
                                             np_utils.categorical_probas_to_classes(Y_test))
    print('Test Kappa:', kappa)
    return acc


def cnn_var_selfembd_other(X_train, Y_train, X_test, Y_test, nb_classes,
                           Other_train, Other_test, k,
                           maxlen, vocab_size, embd_size,
                           nb_filter, batch_size, nb_epoches, optm):
    """
    cnn1d using varying filter lengths
    note need using Graph
    :param X_train:
    :param Y_Train:
    :param X_test:
    :param Y_test:
    :param nb_classes:
    :param maxlen:
    :param vocab_size:
    :param embd_size:
    :param batch_size:
    :param nb_epoches:
    :param optm:
    :return:
    """
    ngram_filters = [2, 5, 8]
    nd_convs = ['conv_' + str(n) for n in ngram_filters]
    nd_pools = ['pool_' + str(n) for n in ngram_filters]
    nd_flats = ['flat_' + str(n) for n in ngram_filters]

    model = Graph()
    model.add_input(name='input', input_shape=(maxlen,), dtype=int)

    model.add_node(Embedding(vocab_size, embd_size, input_length=maxlen),
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

    # CNN for other
    pos_f_len = 10
    pos_pool_len = maxlen - pos_f_len + 1
    model.add_input(name='other_input', input_shape=(maxlen, k), dtype='float')
    model.add_node(Convolution1D(nb_filter=nb_filter,
                                 filter_length=pos_f_len,
                                 border_mode='valid',
                                 activation='relu',
                                 input_shape=(maxlen, k)),
                   name='poscnn', input='other_input')
    model.add_node(MaxPooling1D(pool_length=pos_pool_len),
                   name='pospool', input='poscnn')
    model.add_node(Flatten(), name='posflat', input='pospool')
    model.add_node(Dropout(0.5), name='posdropout', input='posflat')

    model.add_node(Dense(nb_classes, activation='softmax'), name='softmax',
                   inputs=['dropout', 'posdropout'],
                   merge_mode='concat')

    model.add_output(name='output', input='softmax')
    model.compile(optm, loss={'output': 'categorical_crossentropy'})  # note Graph()'s diff syntax

    # early stopping
    earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
    model.fit({'input': X_train, 'other_input': Other_train, 'output': Y_train},
              nb_epoch=nb_epoches, batch_size=batch_size,
              validation_split=0.1, callbacks=[earlystop])
    # Graph doesn't have several arg/func existing in Sequential()
    # - fit no show-accuracy
    # - no predict_classes
    classes = model.predict({'input': X_test, 'other_input': Other_test},
                            batch_size=batch_size)['output'].argmax(axis=1)
    acc = np_utils.accuracy(classes, np_utils.categorical_probas_to_classes(Y_test))  # accuracy only supports classes
    print('Test accuracy:', acc)
    kappa = metrics.quadratic_weighted_kappa(classes, np_utils.categorical_probas_to_classes(Y_test))
    print('Test Kappa:', kappa)
    # return kappa
    return acc


def cnn_other(Y_train, Y_test, nb_classes,
              Other_train, Other_test, k,
              maxlen,
              nb_filter, filter_size, batch_size, nb_epoches, optm):
    """
    cnn1d using varying filter lengths
    note need using Graph
    :param Y_Train:
    :param Y_test:
    :param nb_classes:
    :param maxlen:
    :param vocab_size:
    :param embd_size:
    :param batch_size:
    :param nb_epoches:
    :param optm:
    :return:
    """
    model = Graph()

    # CNN for other
    pos_pool_len = maxlen / 2 - filter_size + 1
    model.add_input(name='other_input', input_shape=(maxlen, k), dtype='float')

    model.add_node(Convolution1D(nb_filter=nb_filter,
                                 filter_length=filter_size,
                                 border_mode='valid',
                                 activation='relu',
                                 input_shape=(maxlen, k)),
                   name='poscnn', input='other_input')
    model.add_node(MaxPooling1D(pool_length=5),
                   name='pospool', input='poscnn')

    # 2nd CNN
    model.add_node(Convolution1D(nb_filter=nb_filter * 2,
                                 filter_length=filter_size,
                                 border_mode='valid',
                                 activation='relu'),
                   name='cnn2', input='pospool')
    model.add_node(MaxPooling1D(pool_length=10),
                   name='cnn2_pool', input='cnn2')

    model.add_node(Flatten(), name='posflat', input='cnn2_pool')
    model.add_node(Dropout(0.5), name='posdropout', input='posflat')

    model.add_node(Dense(nb_classes, activation='softmax'), name='softmax',
                   input='posdropout')
    model.add_output(name='output', input='softmax')
    model.compile(optm, loss={'output': 'categorical_crossentropy'})  # note Graph()'s diff syntax

    # early stopping
    earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
    model.fit({'other_input': Other_train, 'output': Y_train},
              nb_epoch=nb_epoches, batch_size=batch_size,
              validation_split=0.1, callbacks=[earlystop])
    # Graph doesn't have several arg/func existing in Sequential()
    # - fit no show-accuracy
    # - no predict_classes
    classes = model.predict({'other_input': Other_test},
                            batch_size=batch_size)['output'].argmax(axis=1)
    acc = np_utils.accuracy(classes, np_utils.categorical_probas_to_classes(Y_test))  # accuracy only supports classes
    print('Test accuracy:', acc)
    kappa = metrics.quadratic_weighted_kappa(classes, np_utils.categorical_probas_to_classes(Y_test))
    print('Test Kappa:', kappa)
    return acc


def cnn_var_w2vembd(X_train, Y_train, X_test, Y_test, nb_classes,
                    maxlen,
                    nb_filter, batch_size, nb_epoches, optm):
    """
    - CNN-1d on 3d sensor which uses word2vec embedding
    - MOT

    :param <X, Y> train and test sets
    :param nb_classes # of classes
    :param maxlen max of n char in a sentence
    :param nb_filter
    :param filter_length
    :param batch_size
    :param nb_epoch
    :param optm
    :return:
    """
    ngram_filters = [2, 5, 8]
    nd_convs = ['conv_' + str(n) for n in ngram_filters]
    nd_pools = ['pool_' + str(n) for n in ngram_filters]
    nd_flats = ['flat_' + str(n) for n in ngram_filters]

    model = Graph()
    model.add_input(name='input', input_shape=(maxlen, 300), dtype='float')

    # three CNNs
    for i, n_gram in enumerate(ngram_filters):
        pool_length = maxlen - n_gram + 1
        model.add_node(Convolution1D(nb_filter=nb_filter,
                                     filter_length=n_gram,
                                     border_mode="valid",
                                     activation="relu", input_shape=(maxlen, 300)),
                       name=nd_convs[i], input='input')
        model.add_node(MaxPooling1D(pool_length=pool_length),
                       name=nd_pools[i], input=nd_convs[i])
        model.add_node(Flatten(), name=nd_flats[i], input=nd_pools[i])

    model.add_node(Dropout(0.5), name='dropout', inputs=nd_flats, merge_mode='concat')
    model.add_node(Dense(nb_classes, activation='softmax'), name='softmax', input='dropout')

    model.add_output(name='output', input='softmax')
    model.compile(optm, loss={'output': 'categorical_crossentropy'})  # note Graph()'s diff syntax

    # early stopping
    earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
    model.fit({'input': X_train, 'output': Y_train},
              nb_epoch=nb_epoches, batch_size=batch_size,
              validation_split=0.1, callbacks=[earlystop])
    # Graph doesn't have several arg/func existing in Sequential()
    # - fit no show-accuracy
    # - no predict_classes
    classes = model.predict({'input': X_test}, batch_size=batch_size)['output'].argmax(axis=1)
    acc = np_utils.accuracy(classes, np_utils.categorical_probas_to_classes(Y_test))  # accuracy only supports cla
    print('Test Acc: ', acc)
    return acc


def cnn_multi_selfembd(X_train, Y_train, X_test, Y_test, nb_classes,
                       maxlen, vocab_size, embd_size,
                       pos_train, pos_test, pos_embd_dim,
                       dp_train, dp_test, dp_embd_dim,
                       nb_filter, batch_size, nb_epoches, optm):
    """
    cnn1d using multi-inputs, i.e., word, POS, DP
    word using varying filter lengths
    note need using Graph
    :param X_train:
    :param Y_Train:
    :param X_test:
    :param Y_test:
    :param nb_classes:
    :param maxlen:
    :param vocab_size:
    :param embd_size:
    :param batch_size:
    :param nb_epoches:
    :param optm:
    :return:
    """
    ngram_filters = [2, 5, 8]
    nd_convs = ['conv_' + str(n) for n in ngram_filters]
    nd_pools = ['pool_' + str(n) for n in ngram_filters]
    nd_flats = ['flat_' + str(n) for n in ngram_filters]

    model = Graph()
    model.add_input(name='input', input_shape=(maxlen,), dtype=int)

    model.add_node(Embedding(vocab_size, embd_size, input_length=maxlen),
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

    # POS CNN
    nb_pos = 15
    pos_f_len = 3
    pos_pool_len = maxlen - pos_f_len + 1
    model.add_input(name='posinput', input_shape=(maxlen,), dtype=int)
    model.add_node(Embedding(nb_pos, pos_embd_dim, input_length=maxlen),
                   name='posembd', input='posinput')
    model.add_node(Convolution1D(nb_filter=nb_filter,
                                 filter_length=pos_f_len,
                                 border_mode='valid',
                                 activation='relu'),
                   name='poscnn', input='posembd')
    model.add_node(MaxPooling1D(pool_length=pos_pool_len),
                   name='pospool', input='poscnn')
    model.add_node(Flatten(), name='posflat', input='pospool')
    model.add_node(Dropout(0.5), name='posdropout', input='posflat')

    # DP CNN
    nb_dp = vocab_size
    dp_f_len = 3
    dp_pool_len = maxlen - dp_f_len + 1
    model.add_input(name='dpinput', input_shape=(maxlen,), dtype=int)
    model.add_node(Embedding(nb_dp, dp_embd_dim, input_length=maxlen),
                   name='dpembd', input='dpinput')
    model.add_node(Convolution1D(nb_filter=nb_filter,
                                 filter_length=dp_f_len,
                                 border_mode='valid',
                                 activation='relu'),
                   name='dpcnn', input='dpembd')
    model.add_node(MaxPooling1D(pool_length=dp_pool_len),
                   name='dppool', input='dpcnn')
    model.add_node(Flatten(), name='dpflat', input='dppool')
    model.add_node(Dropout(0.5), name='dpdropout', input='dpflat')

    model.add_node(Dense(nb_classes, activation='softmax'), name='softmax',
                   inputs=['dropout', 'posdropout', 'dpdropout'],
                   merge_mode='concat')

    model.add_output(name='output', input='softmax')
    model.compile(optm, loss={'output': 'categorical_crossentropy'})  # note Graph()'s diff syntax

    # early stopping
    earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
    model.fit({'input': X_train, 'posinput': pos_train, 'dpinput': dp_train,
               'output': Y_train},
              nb_epoch=nb_epoches, batch_size=batch_size,
              validation_split=0.1, callbacks=[earlystop])
    # Graph doesn't have several arg/func existing in Sequential()
    # - fit no show-accuracy
    # - no predict_classes
    classes = model.predict({'input': X_test, 'posinput': pos_test, 'dpinput': dp_test}
                            , batch_size=batch_size)['output'].argmax(axis=1)
    acc = np_utils.accuracy(classes, np_utils.categorical_probas_to_classes(Y_test))  # accuracy only supports classes
    print('Test accuracy:', acc)
    kappa = metrics.quadratic_weighted_kappa(classes, np_utils.categorical_probas_to_classes(Y_test))
    print('Test Kappa:', kappa)
    return kappa
    # return acc


def lstm_selfembd(X_train, Y_train, X_test, Y_test, nb_classes,
                  maxlen, vocab_size, embd_dim,
                  batch_size, nb_epoch, optm):
    """
    - LSTM  on text input (represented in int)
    - fully-connected model

    :param <X, Y> train and test sets
    :param nb_classes # of classes
    :param maxlen max of n char in a sentence
    :param vocab_size
    :param embd_dim
    :param batch_size
    :param nb_epoch
    :param optm optimizer options, e.g., adam, rmsprop, etc.
    :return:
    """

    model = Sequential()
    model.add(Embedding(vocab_size, embd_dim, input_length=maxlen))
    model.add(Dropout(0.25))

    # model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(50))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optm)

    earlystop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_split=0.1, show_accuracy=True, callbacks=[earlystop])

    classes = earlystop.model.predict_classes(X_test, batch_size=batch_size)
    acc = np_utils.accuracy(classes, np_utils.categorical_probas_to_classes(Y_test))  # accuracy only supports classes
    print('Test accuracy:', acc)
    kappa = metrics.quadratic_weighted_kappa(classes, np_utils.categorical_probas_to_classes(Y_test))
    print('Test Kappa:', kappa)
    return (kappa)


def test_mr_embd():
    nb_words = 18000
    maxlen = 64
    embd_dim = 100
    X_train, Y_train, X_test, Y_test, nb_classes = load_mr(nb_words, maxlen, 'self')
    cnn1d_selfembd(X_train, Y_train, X_test, Y_test, nb_classes,
                   maxlen, nb_words, embd_dim,
                   100, 5, 32, 20, 'rmsprop')


def test_mr_w2v():
    maxlen = 64
    X_train, Y_train, X_test, Y_test, nb_classes = load_mr(0, maxlen, 'w2v')
    cnn1d_w2vembd(X_train, Y_train, X_test, Y_test, nb_classes,
                  maxlen,
                  100, 5, 32, 20, 'rmsprop')


if __name__ == "__main__":
    np.random.seed(1337)  # for reproducibility

    print('=' * 50)
    print("mr selfembd CNN")
    test_mr_embd()
