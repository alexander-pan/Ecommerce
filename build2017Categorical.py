import numpy as np
import pandas as pd
import math
from datetime import datetime as dt, timedelta
import functions as f

#modules for specific application
import random
import dill

#accessing aws data
import sys
sys.path.append('./utils')
from db_utils import DBUtil

#function to merge dictionaries
def mergeDict(x,y):
    z = x.copy()
    z.update(y)
    return z

#connect to aws
dbu = DBUtil("jjill_redshift","/home/jjill/.databases.conf")

print 'Getting Query'
query = """
select *
from jjill.jjill_keyed_data
where is_emailable_ind='Y' and
department_name in ('Woven Shirts','Knit Tops','Pants','Dresses')
and order_date between '2017-01-01' and '2017-12-31';
"""
DF = dbu.get_df_from_query(query)
DF.columns = map(str.upper,DF.columns)

print 'Number of unique customers:', len(DF.ILINK.unique().tolist())
ilinks = DF.ILINK.unique().tolist()

print 'Getting Column Headers'
features = ['END_USE_DESC','MASTER_CHANNEL','PAY_TYPE_CD','FABRIC_CATEGORY_DESC']
columns = ['ILINK','DEPARTMENT_NAME']
HasFts = {}
PerFts = {}
for ft in features:
    HasFts[ft] = f.BuildHasColumn(DF,ft)
    columns = columns + f.BuildHasColumn(DF,ft)

for ft in features:
    PerFts[ft] = f.BuildPercentColumn(DF,ft)
    columns = columns + f.BuildPercentColumn(DF,ft)

print 'Building Table Data Now...'
features = ['END_USE_DESC','MASTER_CHANNEL','PAY_TYPE_CD','FABRIC_CATEGORY_DESC']
depts = ['Woven Shirts','Knit Tops','Dresses','Pants']
info = {}
table = []
count = 1
for ilink in ilinks:
    if count % 10000 == 0:
        print 'On Customer %d' % count
    count+=1
    for dept in depts:
        #temp = DF[(DF.ILINK==ilink)]
        temp = DF[(DF.ILINK==ilink) & (DF.DEPARTMENT_NAME==dept)]
        info[ilink] = {}
        x = {}
        for ft in features:
            y = f.HasPurchased(temp,ft,HasFts[ft])
            x = mergeDict(x,y)
            z = f.PercentPurchased(temp,ft,PerFts[ft])
            x = mergeDict(x,z)

        #create row for customer
        row = [ilink,dept]
        for col in columns[2:]:
            row.append(x[col])
        table.append(tuple(row))

print 'Building DataFrame'
categorical_table = pd.DataFrame(table,columns = columns)

print 'Saving DataFrame'
categorical_table.to_pickle('./data/categorical_fts_2017.pkl')
