"""
load score-level asroutput files into a pandas df

"""
import re
import pandas as pd

def mr_csvs():
    """
    load data/mr.p and generate two csv files.
    :return:
    """
    x = pickle.load(open('data/mr.p', "rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print("mr.p has been loaded!")

    # focusing on revs.
    texts, labels = [], []
        for rev in revs:
        texts.append(rev["text"])
        labels.append(rev["y"])

    df = pd.DataFrame({'label': labels, 'text': texts})
    print(df.head())


def read_asrout(data_folder, csvFile, clean_string=True):
    """
    read per-score level asrout files
    """
    txts = []
    scores = []
    for score in [0, 1, 2, 3]:
        afile = data_folder[score]
        print "score:" + str(score) + data_folder[score]
        with open(afile, "rb") as f:
            for line in f:
                rev = []
                rev.append(line.strip())
                if clean_string:
                    orig_rev = clean_str(" ".join(rev))
                else:
                    orig_rev = " ".join(rev).lower()
                scores.append(score)
                txts.append(orig_rev)
    df = pd.DataFrame({'text': txts, 'score': scores})
    df.to_csv(open(csvFile, 'wb'))


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

if __name__=="__main__":
    data_folder = ["data/train1.txt", "data/train2.txt", "data/train3.txt", "data/train4.txt"]
    read_asrout(data_folder, csvFile = "data/train.csv")

    data_folder = ["data/test1.txt", "data/test2.txt", "data/test3.txt", "data/test4.txt"]
    read_asrout(data_folder, csvFile = "data/test.csv")
