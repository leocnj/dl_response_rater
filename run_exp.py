from data_util import load_csvs
from cnn1d import cnn1d_selfembd, cnn1d_w2vembd, lstm_selfembd, cnn_var_selfembd, cnn_var_w2vembd
import numpy as np
import ml_metrics as metrics
from sent_op import load_w2v

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


def pun_cv_cnnvar():
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
                                                             nb_words, maxlen, embd_type='self', w2v=None)

        acc = cnn_var_selfembd(X_train, Y_train, X_test, Y_test, nb_classes,
                             maxlen, nb_words, embd_dim,
                             100, 50, 20, 'rmsprop')
        accs.append(acc)
    acc_cv = np.mean(accs)
    print('after 10-fold cv:' + str(acc_cv))


def pun_cv_w2v_cnnvar():
    maxlen = 20

    folds = range(1,11)
    trains = ['data/pun_of_day/train'+str(fold)+'.csv' for fold in folds]
    tests = ['data/pun_of_day/test'+str(fold)+'.csv' for fold in folds]
    pairs = zip(trains, tests)

    w2v = load_w2v('data/Google_w2v.bin')
    print("loaded Google word2vec")

    accs = []
    for (train, test) in pairs:
        print(train + '=>' + test)
        X_train, Y_train, X_test, Y_test, nb_classes = load_csvs(train, test,
                                                             0, maxlen, embd_type='w2v', w2v=w2v)

        acc = cnn_var_w2vembd(X_train, Y_train, X_test, Y_test, nb_classes,
                             maxlen,
                             100, 50, 20, 'rmsprop')
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


def ted_cv_cnnvar():
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
                                                             nb_words, maxlen, embd_type='self', w2v=None)

        acc = cnn_var_selfembd(X_train, Y_train, X_test, Y_test, nb_classes,
                             maxlen, nb_words, embd_dim,
                             100, 50, 20, 'rmsprop')
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

# TODO convert to func

def asap_cv():
    maxlen = 75
    nb_words = 4500
    embd_dim = 50

    folds = (1,2,3,4,5,6,7,8,9,10)
    trains = ['data/asap2/train'+str(fold)+'.csv' for fold in folds]
    tests = ['data/asap2/test'+str(fold)+'.csv' for fold in folds]
    pairs = zip(trains, tests)

    kappas = []
    for (train, test) in pairs:
        print(train + '=>' + test)
        X_train, Y_train, X_test, Y_test, nb_classes = load_csvs(train, test,
                                                             nb_words, maxlen, embd_type='self', w2v=None)

        kappa = cnn1d_selfembd(X_train, Y_train, X_test, Y_test, nb_classes,
                             maxlen, nb_words, embd_dim,
                             100, 5, 50, 20, 'rmsprop')
        kappas.append(kappa)
    kappa_cv = metrics.mean_quadratic_weighted_kappa(kappas)
    # TODO add other metrics.
    print('after 10-fold cv:' + str(kappa_cv))


def asap_cv_w2v():
    maxlen = 40

    folds = range(1,11)
    trains = ['data/asap2/train'+str(fold)+'.csv' for fold in folds]
    tests = ['data/asap2/test'+str(fold)+'.csv' for fold in folds]
    pairs = zip(trains, tests)

    w2v = load_w2v('data/Google_w2v.bin')
    print("loaded Google word2vec")

    kappas = []
    for (train, test) in pairs:
        print(train + '=>' + test)
        X_train, Y_train, X_test, Y_test, nb_classes = load_csvs(train, test,
                                                             0, maxlen, embd_type='w2v', w2v=w2v)

        kappa = cnn1d_w2vembd(X_train, Y_train, X_test, Y_test, nb_classes,
                             maxlen,
                             100, 3, 50, 20, 'rmsprop')
        kappas.append(kappa)
    kappa_cv = np.mean(kappas)
    print('after 10-fold cv:' + str(kappa_cv))


def asap_cv_cnnvar():
    maxlen = 75
    nb_words = 4500
    embd_dim = 50

    folds = (1,2,3,4,5,6,7,8,9,10)
    trains = ['data/asap2/train'+str(fold)+'.csv' for fold in folds]
    tests = ['data/asap2/test'+str(fold)+'.csv' for fold in folds]
    pairs = zip(trains, tests)

    kappas = []
    for (train, test) in pairs:
        print(train + '=>' + test)
        X_train, Y_train, X_test, Y_test, nb_classes = load_csvs(train, test,
                                                             nb_words, maxlen, embd_type='self', w2v=None)

        kappa = cnn_var_selfembd(X_train, Y_train, X_test, Y_test, nb_classes,
                             maxlen, nb_words, embd_dim,
                             100, 50, 20, 'rmsprop')
        kappas.append(kappa)
    kappa_cv = metrics.mean_quadratic_weighted_kappa(kappas)

    print('after 10-fold cv:' + str(kappa_cv))

if __name__=="__main__":
    # pun_cv_cnnvar()
    # pun_cv_w2v_cnnvar()
    # ted_cv_cnnvar()
    # pun_cv_w2v()
    # ted_cv_w2v()
    # ted_cv()
    # asap_cv()
    # asap_cv_w2v()
    asap_cv_cnnvar()

    






