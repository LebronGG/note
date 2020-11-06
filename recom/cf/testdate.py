import numpy as np
import pandas as pd
load='./dataset/ml-100k/u.data'
save='./dataset/ml-100k/ratings.dat'
with open(load,'r',encoding='utf-8') as f:
    with open(save,'w',encoding='utf-8') as fw:
        for line in f:
            tmp=line.strip().split('\t')
            fw.write('::'.join(tmp)+'\n')