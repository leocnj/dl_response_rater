import sys
import re
from os import listdir
from os.path import isfile, join
import pandas as pd

def get_label(x):
    if len(x) == 0:
        out = 0
    else:
        out = 1
    return(out)

def one_tsv(tsv):
    df = pd.read_csv(join('tsv/', tsv), sep='\t', na_filter=False)
    labels = df['Annotations']
    texts = [get_label(t) for t in labels]
    sents = df['Sentence']
    df2 = pd.DataFrame({'label':texts,
                        'text':sents})
    csv = re.sub('tsv', 'csv', tsv)
    print(tsv + '=>' + csv)
    df2.to_csv(join('csv/', csv), index=False)
    
[one_tsv(f) for f in listdir('tsv/') if isfile(join('tsv/', f))]
