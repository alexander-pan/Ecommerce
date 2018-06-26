import numpy as np
import pandas as pd



DF1 = pd.read_pickle('../data/numericFts_may2017_apr2018.pkl')
DF2a = pd.read_pickle('../data/categorical_fts_may2017_nov2017.pkl')
DF2b = pd.read_pickle('../data/categorical_fts_dec2017_apr2018.pkl')
DF2 = pd.concat([DF2a,DF2b])
DF1.columns = map(str.upper, DF1.columns)
DF2.columns = map(str.upper, DF2.columns)
DF2 = DF2.sort_values(['ILINK','MONTH','YEAR'])
DF2.reset_index(drop=True,inplace=True)

DF = DF1.merge(DF2,on=['ILINK','DEPARTMENT_NAME','MONTH','YEAR'],how='outer')
DF['BOUGHT_PANTS'] = [1 if x.DEPARTMENT_NAME == 'Pants' else 0 for i,x in DF.iterrows()]
DF['BOUGHT_WOVENSHIRTS'] = [1 if x.DEPARTMENT_NAME == 'Woven Shirts' else 0 for i,x in DF.iterrows()]
DF['BOUGHT_DRESSES'] = [1 if x.DEPARTMENT_NAME == 'Dresses' else 0 for i,x in DF.iterrows()]
DF['BOUGHT_KNITTOPS'] = [1 if x.DEPARTMENT_NAME == 'Knit Tops' else 0 for i,x in DF.iterrows()]

print 'Current size: ', DF.shape[0]
#print 'Filling Null results with means'
#DF.fillna(DF.mean(),inplace=True)

print 'Dropping NaN results'
DF.dropna(axis=0,how='any',inplace=True)
print 'New Size:', DF.shape[0]

print 'Saving new DataFrame'
DF.to_pickle('../data/CustomerTime_May2017_Apri2018_Dataset.pkl')
