from __future__ import print_function

import pandas as pd
import numpy as np

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils

def df2seq(df, nb_words):

    textraw = df.txt.values.tolist()
    textraw = [line.encode('utf-8') for line in textraw]  # keras needs str

    # keras deals with tokens
    token = Tokenizer(nb_words=nb_words)
    token.fit_on_texts(textraw)
    text_seq = token.texts_to_sequences(textraw)
    return(text_seq, df.score.values)

def csv_2_embd(csvFile, nb_words, maxlen):
    """
    load a csv file into a pd and then prepare embd input
    :param csvFile:
    :param nb_words:
    :param maxlen:
    :return:
    """

    # load csv to PD
    csv_df = pd.read_csv(csvFile)
    print(csv_df.head())

    # force colname contains label and txt
    csv_df.columns = ['id', 'score', 'txt']

    # tokenrize
    df_X, df_y = df2seq(csv_df, nb_words)
    nb_classes = len(np.unique(df_y))
    
    # padding
    df_X_padded = sequence.pad_sequences(df_X, maxlen=maxlen, padding='post', truncating='post')
    df_y_cat = np_utils.to_categorical(df_y, nb_classes)

    return(df_X_padded, df_y_cat, nb_classes)

def load_asap():
    # load csv to PD
    trainFile = '../asap_sas/set1_train.csv'
    testFile = '../asap_sas/set1_test.csv'

    nb_words = 20000
    maxlen = 200
    X_train, Y_train, nb_classes = csv_2_embd(trainFile, nb_words, maxlen)
    X_test,  Y_test,  nb_2 = csv_2_embd(testFile, nb_words, maxlen)
    # make sure nb_classes == nb_2
    print('nb2', nb_2)
    return(X_train, Y_train, X_test, Y_test, nb_classes)


def main():
    X_ta, Y_ta, X_ts, Y_ts, nbc = load_asap()
    print('train tensor shape', X_ta.shape)
    print('#classes', nbc)

if __name__=="__main__":
    main()
