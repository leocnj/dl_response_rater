from __future__ import print_function

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

"""
following https://gist.github.com/xccds/8f0e5b0fe4eb6193261d to do 1d-CNN sentiment detection on the mr data.

- use data/train|test.csv to load txt|score two-col DF

"""
def df2seq(df, nb_words):

    textraw = df.txt.values.tolist()
    textraw = [line.encode('utf-8') for line in textraw]  # keras needs str

    # keras deals with tokens
    token = Tokenizer(nb_words=nb_words)
    token.fit_on_texts(textraw)
    text_seq = token.texts_to_sequences(textraw)
    return(text_seq, df.score.values)


np.random.seed(75513)  # for reproducibility

# load csv to PD
train_csv = pd.read_csv('data/train.csv')
test_csv = pd.read_csv('data/test.csv')

print(train_csv.head())

nb_words = 5000
train_X, train_y = df2seq(train_csv, nb_words)
test_X, test_y   = df2seq(test_csv, nb_words)
nb_classes = len(np.unique(train_y))

# set parameters:

maxlen = 200 #
batch_size = 32
embedding_dims = 300
nb_filter = 100
filter_length = 5
hidden_dims = 100
nb_epoch = 20
pool_length = maxlen - filter_length + 1 # max over time

# padding

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(train_X, maxlen=maxlen, padding='post', truncating='post')
X_test  = sequence.pad_sequences(test_X, maxlen=maxlen, padding='post', truncating='post')
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

Y_train = np_utils.to_categorical(train_y, nb_classes)
Y_test  = np_utils.to_categorical(test_y, nb_classes)

print('Build model...')
model = Sequential()

model.add(Embedding(nb_words, embedding_dims, input_length=maxlen))
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

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
#model.compile(loss='mean_squared_error', optimizer='sgd')

earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
result = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
            validation_split=0.2, show_accuracy=True, callbacks=[earlystop])

score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1, show_accuracy=True)

print('Test score:', score[0])
print('Test accuracy:', score[1])