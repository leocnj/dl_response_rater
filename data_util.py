from __future__ import print_function

import sys
import cPickle
import pandas as pd
import numpy as np

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils

from sklearn.cross_validation import train_test_split
from sent_op import load_w2v, sents_3dtensor


def xcol_nninput_embd(xseq, nb_words, maxlen):
    """
    load textlist, which is corresponding to the text col in a DF
    :param textlist:
    :param nb_words:
    :param maxlen:
    :return:
    """
    # padding
    xseq_padded = sequence.pad_sequences(xseq, maxlen=maxlen, padding='post', truncating='post')
    return(xseq_padded)


def pickled2df(pickfile):
    """
    read a pickled file and create a DF with two cols, label and text
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


def load_csvs(traincsv, testcsv, nb_words, maxlen, embd_type):

    train_df = pd.read_csv(traincsv)
    test_df = pd.read_csv(testcsv)
    print(train_df.head())

    train_X = train_df.text.values.tolist()
    test_X = test_df.text.values.tolist()

    # save for w2v embd
    train_X_wds = train_X
    test_X_wds = test_X

    train_y = train_df.label.values
    test_y  = test_df.label.values
    nb_classes = len(np.unique(train_y))
    Y_train = np_utils.to_categorical(train_y, nb_classes)
    Y_test = np_utils.to_categorical(test_y, nb_classes)

    # tokenrize should be applied on train+test jointly
    n_ta = len(train_X)
    n_ts = len(test_X)
    print('train len vs. test len', n_ta, n_ts)

    textraw = [line.encode('utf-8') for line in train_X+test_X]  # keras needs str
    # keras deals with tokens
    token = Tokenizer(nb_words=nb_words)
    token.fit_on_texts(textraw)
    textseq = token.texts_to_sequences(textraw)

    # stat about textlist
    print('nb_words: ', len(token.word_counts))
    print('mean len: ', np.mean([len(x) for x in textseq]))

    train_X = textseq[0:n_ta]
    test_X = textseq[n_ta:]

    if(embd_type == 'self'):
        X_train = sequence.pad_sequences(train_X, maxlen=maxlen, padding='post', truncating='post')
        X_test = sequence.pad_sequences(test_X, maxlen=maxlen, padding='post', truncating='post')
    elif(embd_type == 'w2v'):
        w2v = load_w2v('data/Google_w2v.bin')
        print("loaded Google word2vec")
        X_train = sents_3dtensor(train_X_wds, maxlen, w2v)
        X_test = sents_3dtensor(test_X_wds, maxlen, w2v)
    else:
        print('wrong embd_type')

    print('X tensor shape: ', X_train.shape)
    print('Y tensor shape: ', Y_train.shape)
    return(X_train, Y_train, X_test, Y_test, nb_classes)


def load_mr(nb_words=20000, maxlen=64, embd_type='self'):
    """
    :param embd_type: self vs. w2v
    :return:
    """

    train_size = 0.8

    df = pickled2df('data/mr.p')
    print(df.head())

    train_X, test_X, train_y, test_y = train_test_split(df.text.values.tolist(),
                                                        df.label.values,
                                                        train_size=train_size, random_state=1)
    train_X_wds = train_X
    test_X_wds = test_X

    nb_classes = len(np.unique(train_y))
    Y_train = np_utils.to_categorical(train_y, nb_classes)
    Y_test  = np_utils.to_categorical(test_y, nb_classes)

    # tokenrize should be applied on train+test jointly
    n_ta = len(train_X)
    n_ts = len(test_X)
    print('train len vs. test len', n_ta, n_ts)

    textraw = [line.encode('utf-8') for line in train_X+test_X]  # keras needs str
    # keras deals with tokens
    token = Tokenizer(nb_words=nb_words)
    token.fit_on_texts(textraw)
    textseq = token.texts_to_sequences(textraw)

    # stat about textlist
    print('nb_words: ',len(token.word_counts))
    print('mean len: ',np.mean([len(x) for x in textseq]))

    train_X = textseq[0:n_ta]
    test_X = textseq[n_ta:]

    if(embd_type == 'self'):
        X_train = xcol_nninput_embd(train_X, nb_words, maxlen)
        X_test  = xcol_nninput_embd(test_X,  nb_words, maxlen)
    elif(embd_type == 'w2v'):
        w2v = load_w2v('data/Google_w2v.bin')
        print("loaded Google word2vec")
        X_train = sents_3dtensor(train_X_wds, maxlen, w2v)
        X_test  = sents_3dtensor(test_X_wds, maxlen, w2v)
    else:
        print('wrong embd_type')

    print('X tensor shape: ', X_train.shape)
    print('Y tensor shape: ', Y_train.shape)
    return (X_train, Y_train, X_test, Y_test, nb_classes)


def main():
    load_mr('self')

if __name__=="__main__":
    main()
