import pandas as pd
import numpy as np
import sklearn
from sklearn.utils import shuffle
a={'a':['a','b','c','a','a'],'b':[1,2,3,1,5]}
df=pd.DataFrame.from_dict(a)

for i in df.values:
    print(i)
# df=shuffle(pd.DataFrame.from_dict(a))
# print(df['a']['b'])
# user = df['a'].unique().tolist()
# id = df['b'].unique().tolist()
# df['a'] = df['a'].apply(lambda x : user.index(x))
# df['b'] = df['b'].apply(lambda x : id.index(x))
# print(user)
# print(id)
# for i in