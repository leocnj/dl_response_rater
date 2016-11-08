from data_util import load_csvs, load_other
from cnn1d import cnn1d_selfembd, cnn1d_w2vembd, lstm_selfembd, \
    cnn_var_selfembd, cnn_var_w2vembd, cnn_multi_selfembd, \
    cnn_var_selfembd_other, cnn_other

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
        X_train, Y_train, X_test, Y_test, nb_classes = load_csvs(train, test, nb_words, maxlen,
                                                                 embd_type='self', w2v=None)

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

    w2v = load_w2v('data/Google_w2v.bin')
    print("loaded Google word2vec")

    accs = []
    for (train, test) in pairs:
        print(train + '=>' + test)
        X_train, Y_train, X_test, Y_test, nb_classes = load_csvs(train, test,
                                                             0, maxlen, embd_type='w2v', w2v=w2v)

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



def ted_cv_w2v_cnnvar():
    maxlen = 20

    folds = range(1,11)
    trains = ['data/TED/train'+str(fold)+'.csv' for fold in folds]
    tests = ['data/TED/test'+str(fold)+'.csv' for fold in folds]
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
                             100, 50, 25, 'adagrad')
        accs.append(acc)
    acc_cv = np.mean(accs)
    print('after 10-fold cv:' + str(acc_cv))


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
                             50, 32, 30, 'rmsprop')
        kappas.append(kappa)
    kappa_cv = metrics.mean_quadratic_weighted_kappa(kappas)

    print('after 10-fold cv:' + str(kappa_cv))


def asap_cv_cnn_multi():
    maxlen = 75
    nb_words = 4500
    embd_dim = 50
    nb_pos = 15

    folds = (1,2,3,4,5,6,7,8,9,10)
    trains = ['data/asap2/train'+str(fold)+'.csv' for fold in folds]
    tests = ['data/asap2/test'+str(fold)+'.csv' for fold in folds]
    pos_tas = ['data/asap2/pos/train'+str(fold)+'_pos.csv' for fold in folds]
    pos_tss = ['data/asap2/pos/test'+str(fold)+'_pos.csv' for fold in folds]
    dp_tas = ['data/asap2/dp/train'+str(fold)+'_dp.csv' for fold in folds]
    dp_tss = ['data/asap2/dp/test'+str(fold)+'_dp.csv' for fold in folds]

    pairs = zip(trains, tests, pos_tas, pos_tss, dp_tas, dp_tss)

    kappas = []
    for (train, test, pos_ta, pos_ts, dp_ta, dp_ts) in pairs:
        print(train + '=>' + test)
        X_train, Y_train, X_test, Y_test, nb_classes = load_csvs(train, test,
                                                                nb_words, maxlen, embd_type='self', w2v=None)
        pos_train, foo1, pos_test, foo2, foo3 = load_csvs(pos_ta, pos_ts,
                                                          nb_pos, maxlen, embd_type='self', w2v=None)
        dp_train,  foo1, dp_test, foo2, foo3 = load_csvs(dp_ta, dp_ts,
                                                         nb_words, maxlen, embd_type='self', w2v=None)

        kappa = cnn_multi_selfembd(X_train, Y_train, X_test, Y_test, nb_classes,
                             maxlen, nb_words, embd_dim,
                             pos_train, pos_test, 10,
                             dp_train, dp_test, 40,
                             50, 32, 30, 'rmsprop')
        kappas.append(kappa)
    kappa_cv = metrics.mean_quadratic_weighted_kappa(kappas)

    print('after 10-fold cv:' + str(kappa_cv))


def tpo_cv_cnnvar():
    maxlen = 200
    nb_words = 6500
    embd_dim = 100

    folds = range(1, 11)
    trains = ['data/tpov4/train_'+str(fold)+'.csv' for fold in folds]
    tests = ['data/tpov4/test_'+str(fold)+'.csv' for fold in folds]
    pairs = zip(trains, tests)

    accs = []
    for (train, test) in pairs:
        print(train + '=>' + test)
        X_train, Y_train, X_test, Y_test, nb_classes = load_csvs(train, test,
                                                             nb_words, maxlen, embd_type='self', w2v=None)

        acc = cnn_var_selfembd(X_train, Y_train, X_test, Y_test, nb_classes,
                             maxlen, nb_words, embd_dim,
                             50, 32, 25, 'rmsprop')
        accs.append(acc)
    acc_cv = np.mean(accs)
    print('after 10-fold cv:' + str(acc_cv))


def tpo_cv_w2v_cnnvar():
    maxlen = 175

    folds = range(1,11)
    trains = ['data/tpov4/train_'+str(fold)+'.csv' for fold in folds]
    tests = ['data/tpov4/test_'+str(fold)+'.csv' for fold in folds]
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
                             100, 50, 25, 'rmsprop')
        accs.append(acc)
    acc_cv = np.mean(accs)
    print('after 10-fold cv:' + str(acc_cv))


def tpo_cv_cnnvar_other():

    maxlen = 200
    nb_words = 6500
    embd_dim = 100
    k = 4

    folds = range(1, 11)
    trains = ['data/tpov4/train_'+str(fold)+'.csv' for fold in folds]
    tests = ['data/tpov4/test_'+str(fold)+'.csv' for fold in folds]
    tas_other = ['data/tpov4/train_'+str(fold)+'_other.csv' for fold in folds]
    tss_other = ['data/tpov4/test_'+str(fold)+'_other.csv' for fold in folds]
    pairs = zip(trains, tests, tas_other, tss_other)

    accs = []
    for (train, test, ta_other, ts_other) in pairs:
        print(train + '=>' + test)
        X_train, Y_train, X_test, Y_test, nb_classes = load_csvs(train, test,
                                                             nb_words, maxlen, embd_type='self', w2v=None)
        Other_train = load_other(ta_other, maxlen, k)
        Other_test = load_other(ts_other, maxlen, k)

        acc = cnn_var_selfembd_other(X_train, Y_train, X_test, Y_test, nb_classes,
                               Other_train, Other_test, k,
                               maxlen, nb_words, embd_dim,
                               50, 32, 25, 'rmsprop')
        accs.append(acc)
    acc_cv = np.mean(accs)
    print('after 10-fold cv:' + str(acc_cv))


def tpo_cv_cnn_other():

    maxlen = 200
    nb_words = 6500
    filter_size = 20
    k = 4

    folds = range(1, 11)
    trains = ['data/tpov4/train_'+str(fold)+'.csv' for fold in folds]
    tests = ['data/tpov4/test_'+str(fold)+'.csv' for fold in folds]
    tas_other = ['data/tpov4/train_'+str(fold)+'_other.csv' for fold in folds]
    tss_other = ['data/tpov4/test_'+str(fold)+'_other.csv' for fold in folds]
    pairs = zip(trains, tests, tas_other, tss_other)

    accs = []
    for (train, test, ta_other, ts_other) in pairs:
        print(train + '=>' + test)
        X_train, Y_train, X_test, Y_test, nb_classes = load_csvs(train, test,
                                                             nb_words, maxlen, embd_type='self', w2v=None)
        Other_train = load_other(ta_other, maxlen, k)
        Other_test = load_other(ts_other, maxlen, k)

        acc = cnn_other(Y_train, Y_test, nb_classes,
                        Other_train, Other_test, k,
                        maxlen,
                        50, filter_size, 32, 25, 'rmsprop')
        accs.append(acc)
    acc_cv = np.mean(accs)
    print('after 10-fold cv:' + str(acc_cv))


def argu_cv():
    maxlen = 25
    nb_words = 10000
    embd_dim = 50

    folds = ['VC048263',
             'VC048408',
             'VC084849',
             'VC084851',
             'VC084853',
             'VC101537',
             'VC101541',
             'VC140094',
             'VC207640',
             'VC248479']

    trains = ['data/Argu/csv/generic_' + str(fold) + '_training.csv' for fold in folds]
    tests  = ['data/Argu/csv/generic_' + str(fold) + '_testing.csv' for fold in folds]
    pairs = zip(trains, tests)

    accs = []
    for (train, test) in pairs:
        print(train + '=>' + test)
        X_train, Y_train, X_test, Y_test, nb_classes = load_csvs(train, test, nb_words, maxlen,
                                                                 embd_type='self', w2v=None)

        acc = cnn1d_selfembd(X_train, Y_train, X_test, Y_test, nb_classes,
                             maxlen, nb_words, embd_dim,
                             100, 5, 50, 20, 'rmsprop')
        accs.append(acc)
    acc_cv = np.mean(accs)
    print('after 10-fold cv:' + str(acc_cv))



if __name__=="__main__":
    # pun_cv()
    # pun_cv_cnnvar()
    # pun_cv_w2v_cnnvar()
    # ted_cv_cnnvar()
    # pun_cv_w2v()
    # ted_cv_w2v()
    # ted_cv_w2v_cnnvar()
    # ted_cv()
    # asap_cv_cnn_multi()
    # asap_cv_w2v()
    # asap_cv_cnnvar()
    # tpo_cv_cnnvar()      #  ACC 0.5464
    # tpo_cv_w2v_cnnvar()  0.43 acc just chance.
    # tpo_cv_cnnvar_other()  # ACC 0.5456
    # tpo_cv_cnn_other()

    argu_cv()



# tpo_cv_cnnvar  max_len 175 filter = 100 ACC 0.5464
#                max_len 200 filter = 50  ACC 0.5364
# tpo cv_cnnvar_other max_len 200 filter = 50 ACC is 0.5520
# tpo cv_cnn_other max_len 200 filter 50 f_size 10 ACC  0.5015



