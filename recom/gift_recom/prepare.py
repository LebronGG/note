import pandas as pd
import re
with open('./dataset/out.csv','r') as fr:
    with open('./dataset/ratings.dat','w') as fw:
        for line in fr:
            if re.search(r'\d+',line):
                a=line.strip().split('|')

                uid = int(a[1].strip())
                item_id = int(a[2].strip())
                count = int(a[3].strip())
                price = float(a[4].strip())

                fw.write('::'.join(map(str,[uid,item_id,price]))+'\n')
            else:
                continue
