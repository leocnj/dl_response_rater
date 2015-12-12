"""
sent_w2v.py

- loading pre-trained |dim|=300 Google word2vec
- for a sentence, get its w2v representation [max_len x 300]

"""
from gensim.models.word2vec import Word2Vec
import numpy as np


def sent_2dvec(sent, max_len, w2v):
    k = 300
    blank_vec = np.random.uniform(-0.25,0.25,k)
    tensor_2d = np.zeros((max_len, k), dtype='float32')
    i = 0
    for wd in sent.split(' '):
        if wd in w2v.vocab:
            tensor_2d[i] = w2v[wd]
        else:
            tensor_2d[i] = blank_vec
        i = i + 1
    if i >= max_len:
        pass
    else:
        # padding
        while i < max_len:
            tensor_2d[i] = blank_vec
            i = i + 1
    return(tensor_2d[0:max_len-1])

w2v = Word2Vec.load_word2vec_format('data/Google_w2v.bin', binary=True)  # C binary format

out = sent_2dvec('this is a book', 10, w2v)
print(out)






