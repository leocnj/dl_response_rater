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

"""
following https://gist.github.com/xccds/8f0e5b0fe4eb6193261d to do 1d-CNN sentiment detection on the mr data.

1/8/2016 try to follow Kim's method
- w/o using hidden state
- dropout on the penultimate layer
- trying identical params

- only using 100 filter_size=5 filters, already obtain 0.75 after 6 epoch


"""
np.random.seed(75513)  # for reproducibility

print
"loading data..."

pf = str(sys.argv[1])
print(pf)

x = pickle.load(open(pf, "rb"))
revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
print("data loaded!")

# focusing on revs.
texts, labels = [], []
for rev in revs:
    texts.append(rev["text"])
    labels.append(rev["y"])

df = pd.DataFrame({'label': labels, 'text': texts})
print(df.head())
df.to_csv(path_or_buf="df.csv")

textraw = df.text.values.tolist()
textraw = [line.encode('utf-8') for line in textraw]  # keras needs str

# keras handels tokens
maxfeatures = 18000  # mr has 18K words

token = Tokenizer(nb_words=maxfeatures)
token.fit_on_texts(textraw)
text_seq = token.texts_to_sequences(textraw)
print(np.median([len(x) for x in text_seq]))

# y
y = df.label.values
nb_classes = len(np.unique(y))
print(nb_classes)

# simple 80/20 split on the entire data.
from sklearn.cross_validation import train_test_split
train_X, test_X, train_y, test_y = train_test_split(text_seq, y, train_size=0.8, random_state=1)

# set parameters:
maxlen = 64
batch_size = 50
embedding_dims = 100
nb_filter = 100
filter_length = 4
nb_epoch = 20
pool_length = maxlen - filter_length + 1

# padding

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(train_X, maxlen=maxlen, padding='post', truncating='post')
X_test = sequence.pad_sequences(test_X, maxlen=maxlen, padding='post', truncating='post')
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

Y_train = np_utils.to_categorical(train_y, nb_classes)
Y_test = np_utils.to_categorical(test_y, nb_classes)

print('Build model...')
model = Sequential()

model.add(Embedding(maxfeatures, embedding_dims, input_length=maxlen))
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
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# early stopping
earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          validation_split=0.1, show_accuracy=True,callbacks=[earlystop])

classes = earlystop.model.predict_classes(X_test, batch_size=batch_size)
acc = np_utils.accuracy(classes, np_utils.categorical_probas_to_classes(Y_test)) # accuracy only supports classes
print('Test accuracy:', acc)
