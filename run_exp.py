from data_util import load_csvs
from cnn1d import cnn1d_selfembd, cnn1d_w2vembd, lstm_selfembd
import numpy as np

def load_asap(nb_words=10000, maxlen=200, embd_type='self'):
    X_train, Y_train, X_test, Y_test, nb_classes = load_csvs('../asap_sas/set1_train.csv',
                                                             '../asap_sas/set1_test.csv',
                                                             nb_words, maxlen, embd_type)
    return(X_train, Y_train, X_test, Y_test, nb_classes)


def load_sg15(nb_words=8000, maxlen=150, embd_type='self'):
    X_train, Y_train, X_test, Y_test, nb_classes = load_csvs('data/sg15_train.csv',
                                                             'data/test.csv',
                                                             nb_words, maxlen, embd_type)
    return(X_train, Y_train, X_test, Y_test, nb_classes)



def load_ted(nb_words=8000, maxlen=150, embd_type='self'):
    X_train, Y_train, X_test, Y_test, nb_classes = load_csvs('data/TED/train1.csv',
                                                             'data/TED/test1.csv',
                                                             nb_words, maxlen, embd_type)
    return(X_train, Y_train, X_test, Y_test, nb_classes)


def test_asap():
    nb_words = 2900
    maxlen = 75
    embd_dim = 50
    X_train, Y_train, X_test, Y_test, nb_classes = load_asap(nb_words, maxlen, 'self')
    cnn1d_selfembd(X_train, Y_train, X_test, Y_test, nb_classes,
                   maxlen, nb_words, embd_dim,
                   100, 5, 50, 20, 'rmsprop')

def test_asap_w2v():
    maxlen = 75
    X_train, Y_train, X_test, Y_test, nb_classes = load_asap(0, maxlen, 'w2v')
    cnn1d_w2vembd(X_train, Y_train, X_test, Y_test, nb_classes,
                   maxlen,
                   100, 5, 50, 20, 'rmsprop')


def test_sg15_lstm():
    nb_words = 8000 # for NNS speakers, should be sufficient
    maxlen = 120
    embd_dim = 100
    X_train, Y_train, X_test, Y_test, nb_classes = load_sg15(nb_words, maxlen, 'self')
    lstm_selfembd(X_train, Y_train, X_test, Y_test, nb_classes,
                   maxlen, nb_words, embd_dim,
                   50, 20, 'rmsprop')

def test_sg15():
    nb_words = 8000
    maxlen = 120
    embd_dim = 100
    X_train, Y_train, X_test, Y_test, nb_classes = load_sg15(nb_words, maxlen, 'self')
    cnn1d_selfembd(X_train, Y_train, X_test, Y_test, nb_classes,
                   maxlen, nb_words, embd_dim,
                   100, 5, 50, 20, 'rmsprop')

def test_sg15_w2v():
    maxlen = 120
    X_train, Y_train, X_test, Y_test, nb_classes = load_sg15(0, maxlen, 'w2v')
    cnn1d_w2vembd(X_train, Y_train, X_test, Y_test, nb_classes,
                   maxlen,
                   100, 10, 64, 20, 'rmsprop')

def test_ted():
    nb_words = 7500
    maxlen = 20
    embd_dim = 100
    X_train, Y_train, X_test, Y_test, nb_classes = load_ted(nb_words, maxlen, 'self')
    cnn1d_selfembd(X_train, Y_train, X_test, Y_test, nb_classes,
                   maxlen, nb_words, embd_dim,
                   100, 5, 50, 20, 'rmsprop')


def test_ted_w2v():
    maxlen = 20
    X_train, Y_train, X_test, Y_test, nb_classes = load_ted(0, maxlen, 'w2v')
    cnn1d_w2vembd(X_train, Y_train, X_test, Y_test, nb_classes,
                   maxlen,
                   100, 5, 50, 20, 'rmsprop')


def pun_cv():
    maxlen = 20
    nb_words = 8000
    embd_dim = 100

    folds = range(1,11)
    trains = ['data/pun_of_day/train'+str(fold)+'.csv' for fold in folds]
    tests = ['data/pun_of_day/test'+str(fold)+'.csv' for fold in folds]
    pairs = zip(trains, tests)

    accs = []
    for (train, test) in pairs:
        print(train + '=>' + test)
        X_train, Y_train, X_test, Y_test, nb_classes = load_csvs(train, test,
                                                             nb_words, maxlen, embd_type='self')

        acc = cnn1d_selfembd(X_train, Y_train, X_test, Y_test, nb_classes,
                             maxlen, nb_words, embd_dim,
                             100, 5, 50, 20, 'rmsprop')
        accs.append(acc)
    acc_cv = np.mean(accs)
    print('after 10-fold cv:' + str(acc_cv))

def pun_cv_w2v():
    maxlen = 20

    folds = range(1,11)
    trains = ['data/pun_of_day/train'+str(fold)+'.csv' for fold in folds]
    tests = ['data/pun_of_day/test'+str(fold)+'.csv' for fold in folds]
    pairs = zip(trains, tests)

    accs = []
    for (train, test) in pairs:
        print(train + '=>' + test)
        X_train, Y_train, X_test, Y_test, nb_classes = load_csvs(train, test,
                                                             0, maxlen, embd_type='w2v')

        acc = cnn1d_w2vembd(X_train, Y_train, X_test, Y_test, nb_classes,
                             maxlen,
                             100, 5, 50, 20, 'rmsprop')
        accs.append(acc)
    acc_cv = np.mean(accs)
    print('after 10-fold cv:' + str(acc_cv))


def ted_cv():
    maxlen = 20
    nb_words = 14000
    embd_dim = 100

    folds = range(1,11)
    trains = ['data/TED/train'+str(fold)+'.csv' for fold in folds]
    tests = ['data/TED/test'+str(fold)+'.csv' for fold in folds]
    pairs = zip(trains, tests)

    accs = []
    for (train, test) in pairs:
        print(train + '=>' + test)
        X_train, Y_train, X_test, Y_test, nb_classes = load_csvs(train, test,
                                                             nb_words, maxlen, embd_type='self')

        acc = cnn1d_selfembd(X_train, Y_train, X_test, Y_test, nb_classes,
                             maxlen, nb_words, embd_dim,
                             100, 5, 50, 20, 'rmsprop')
        accs.append(acc)
    acc_cv = np.mean(accs)
    print('after 10-fold cv:' + str(acc_cv))


def ted_cv_w2v():
    maxlen = 20

    folds = range(1,11)
    trains = ['data/TED/train'+str(fold)+'.csv' for fold in folds]
    tests = ['data/TED/test'+str(fold)+'.csv' for fold in folds]
    pairs = zip(trains, tests)

    accs = []
    for (train, test) in pairs:
        print(train + '=>' + test)
        X_train, Y_train, X_test, Y_test, nb_classes = load_csvs(train, test,
                                                             0, maxlen, embd_type='w2v')

        acc = cnn1d_w2vembd(X_train, Y_train, X_test, Y_test, nb_classes,
                             maxlen,
                             100, 5, 50, 20, 'rmsprop')
        accs.append(acc)
    acc_cv = np.mean(accs)
    print('after 10-fold cv:' + str(acc_cv))



if __name__=="__main__":
    # pun_cv_w2v()
    # ted_cv_w2v()
    ted_cv()
    






