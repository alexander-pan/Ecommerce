import numpy as np
import pandas as pd



DF1 = pd.read_pickle('../data/numericalFts_15kUsers_BaselineV1.pkl')
DF2 = pd.read_pickle('../data/categoricalFts_15kUsers_BaselineV1.pkl')
DF1.drop('MOST_RECENT_PAST_ORDER_DATE',axis='columns',inplace=True)

DF = DF1.merge(DF2,on=['ILINK','REF_DATE_MONTH','REF_DATE_DAY','REF_DATE_YEAR'])
DF.columns = [x.replace(' ','_') for x in DF.columns]

print 'Current size: ', DF.shape[0]
#print 'Filling Null results with means'
#DF.fillna(DF.mean(),inplace=True)

#print 'Dropping NaN results'
#DF.dropna(axis=0,how='any',inplace=True)
#print 'New Size:', DF.shape[0]

print 'Saving new DataFrame'
DF.to_pickle('../data/Customer_Dataset_10k_Baseline.pkl')
