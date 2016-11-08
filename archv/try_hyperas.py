from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform


def keras_model():

    import pandas as pd
    import numpy as np

    from keras.preprocessing import sequence
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.convolutional import Convolution1D, MaxPooling1D
    from keras.callbacks import EarlyStopping
    from keras.utils import np_utils

    from data_util import load_csvs, load_other
    import ml_metrics as metrics

    nb_words = 6500
    maxlen = 175
    filter_length = 10
    other_col_dim = 4

    X_train, Y_train, X_test, Y_test, nb_classes = load_csvs('data/tpov4/train_1.csv',
                                                             'data/tpov4/test_1.csv',
                                                              nb_words, maxlen, 'self', w2v=None)

    # read _other.csv
    other_train = load_other('data/tpov4/train_1_other.csv', maxlen, other_col_dim)
    other_test = load_other('data/tpov4/test_1_other.csv', maxlen, other_col_dim)

    print('other tensor:', other_train.shape)

    pool_length = maxlen - filter_length + 1

    model = Sequential()
    model.add(Convolution1D(nb_filter=50,
                            filter_length=filter_length,
                            border_mode="valid", activation="relu",
                            input_shape=(maxlen, other_col_dim)))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(Flatten())
    model.add(Dropout(0.05))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer={{choice(['rmsprop', 'adam', 'adadelta', 'adagrad'])}})

    earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)

    model.fit(other_train, Y_train, batch_size=32, nb_epoch=25,
              validation_split=0.1, show_accuracy=True, callbacks=[earlystop])

    classes = earlystop.model.predict_classes(other_test, batch_size=32)
    org_classes = np_utils.categorical_probas_to_classes(Y_test)

    acc = np_utils.accuracy(classes, org_classes)  # accuracy only supports classes
    print('Test accuracy:', acc)
    kappa = metrics.quadratic_weighted_kappa(classes, org_classes)
    print('Test Kappa:', kappa)
    return {'loss': -acc, 'status': STATUS_OK}


if __name__ == '__main__':
    best_run = optim.minimize(keras_model,
                              algo=tpe.suggest,
                              max_evals=10,
                              trials=Trials())
    print(best_run)





