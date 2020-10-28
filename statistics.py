import context
import pandas as pd
import numpy as np
import os

# data = pd.read_csv('./train/type-two-10-150000-samples.csv')
# count = data.loc[:,'Label'].value_counts()
# data = pd.read_csv('./train/type-two-8-150000-samples.csv')
# count2 = data.loc[:,'Label'].value_counts()
# print(count)
# print(count2)
# # table = pd.DataFrame(count)
# print(pd.concat([count,count2],axis=1))

table = pd.DataFrame()
for root, dirs, fnames in os.walk('./train'):
    for fname in fnames:
        print('start to process: ' + fname)
        data = pd.read_csv(os.path.join(root, fname))
        count = data.loc[:,'Label'].value_counts()
        table = pd.concat([table,count.astype(int)], axis=1)
        print('table is:')
        print(table)
    table.columns = fnames
print(table)
print('*****end*****')
table.fillna(None)
table.T.to_csv('statistics.csv')
