from __future__ import print_function
import sys
import cPickle
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

from sklearn.cross_validation import train_test_split

from sent_w2v import load_w2v, sents_3dtensor

"""
following https://gist.github.com/xccds/8f0e5b0fe4eb6193261d to do 1d-CNN sentiment detection on the mr data.

- using wd2vec (k=300)
- 1d-CNN and then max-over-time
-

"""

def load_pd(pickfile):
    """
    read picked file and create a DF with two cols, label and text
    """
    print("loading %s" % pickfile)
    x = cPickle.load(open(pickfile, "rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print("data loaded!")

    # focusing on revs.
    texts, labels = [], []
    for rev in revs:
        texts.append(rev["text"])
        labels.append(rev["y"])

    df = pd.DataFrame({'label': labels, 'text': texts})
    return df


def df_to_nn_inputs(df, maxlen):
    """
    load a DF and output to NN input, 3d tensor (#samples, max_len, 300)
    :param df: DF holding train+test data; using a random split
    :param maxlen: max length of each sentence
    :return: X_train (3d), Y_train (1d), X_test, Y_test
    """
    print(df.head())
    # x
    text_raw = df.text.values.tolist()
    text_raw = [line.encode('utf-8') for line in text_raw]  # keras needs str
    # y
    y = df.label.values
    nb_classes = len(np.unique(y))
    # simple 80/20 split on the entire data.
    train_X, test_X, train_y, test_y = train_test_split(text_raw, y, train_size=0.8, random_state=1)

    w2v = load_w2v('data/Google_w2v.bin')
    print("loaded Google word2vec")

    X_train = sents_3dtensor(train_X, maxlen, w2v)
    X_test = sents_3dtensor(test_X, maxlen, w2v)
    Y_train = np_utils.to_categorical(train_y, nb_classes)
    Y_test = np_utils.to_categorical(test_y, nb_classes)

    print('tensor shape: ',X_train.shape)
    return (X_train, Y_train, X_test, Y_test, nb_classes)


def cnn1d_mot(df, maxlen):
    """
    - CNN-1d on 3d sensor
    - MOT
    - fully-connected model

    :param df DF containing entire data set
    :param maxlen sentence max char
    :return:
    """
    nb_filter = 100
    filter_length = 5
    hidden_dims = 100
    nb_epoch = 20
    batch_size = 32
    pool_length = maxlen - filter_length + 1

    X_train, Y_train, X_test, Y_test, nb_classes = df_to_nn_inputs(df, maxlen)

    model = Sequential()
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode="valid",
                            activation="relu", input_shape=(100, 300)))
    model.add(Dropout(0.25))

    # max-over-time (mot)
    model.add(MaxPooling1D(pool_length=pool_length))

    model.add(Flatten())
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)

    result = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                       validation_split=0.1, show_accuracy=True, callbacks=[earlystop])

    score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1, show_accuracy=True)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def main():
    np.random.seed(1337)  # for reproducibility
    maxlen = 100

    pd = load_pd('data/mr.p')
    cnn1d_mot(pd, maxlen)

if __name__=="__main__":
    main()
