"""
sent_op.py

contains sentence level op

Regarding using word2vec embedding

- loading pre-trained |dim|=300 Google word2vec
- for a sentence, get its w2v representation [max_len x 300]

Regarding training an embedding directly from data

- tokenrize a char-string to int-string

"""
from gensim.models.word2vec import Word2Vec
import numpy as np


def load_w2v(w2vfile):
    w2v = Word2Vec.load_word2vec_format(w2vfile, binary=True)  # C binary format
    return(w2v)


def sent_2dvec(sent, max_len, w2v):
    k = 300
    blank_vec = np.random.uniform(-0.25, 0.25, k)
    tensor_2d = np.zeros((max_len, k), dtype='float32')
    i = 0
    for wd in sent.split(' '):
        if wd in w2v.vocab:
            tensor_2d[i] = w2v[wd]
        else:
            tensor_2d[i] = blank_vec
        i += 1
    if i >= max_len:
        pass
    else:
        # padding
        while i < max_len:
            tensor_2d[i] = blank_vec
            i += 1
    return tensor_2d[0:max_len] ## need to figure out


def sents_3dtensor(sents, max_len, w2v):
    k = 300
    tensor_3d = np.zeros((len(sents), max_len, k), dtype='float32')
    print tensor_3d.shape

    i = 0
    for sent in sents:
        tensor_3d[i] = sent_2dvec(sent, max_len, w2v)
        i += 1
    return(tensor_3d)


def main():
    w2v = load_w2v('data/Google_w2v.bin')

    #out = sent_2dvec('this is a book', 10, w2v)
    #print(out)

    out2 = sents_3dtensor(('this is cool', 'that is bad'), 10, w2v)
    print(out2)

if __name__=="__main__":
    main()







