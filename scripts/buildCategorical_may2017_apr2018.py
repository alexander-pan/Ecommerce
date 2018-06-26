import numpy as np
import pandas as pd
import math
from datetime import datetime as dt, timedelta
import functionsV2 as f

#modules for specific application
import random
import dill

#accessing aws data
import sys
sys.path.append('../utils')
from db_utils import DBUtil

#function to merge dictionaries
def mergeDict(x,y):
    z = x.copy()
    z.update(y)
    return z

#connect to aws
#dbu = DBUtil("jjill_redshift","/home/jjill/.databases.conf")
dbu = DBUtil("jjill_redshift","../../databases/database.conf")

print 'Getting Query'
query = """
select *,
date_part(mon,order_date) as month,
date_part(yr,order_date) as year
from jjill.jjill_keyed_data
where is_emailable_ind='Y' and
department_name in ('Woven Shirts','Knit Tops','Pants','Dresses')
and order_date between '2017-05-01' and '2018-04-30'
order by ilink,month,year
"""
DF = dbu.get_df_from_query(query)
DF.columns = map(str.upper,DF.columns)

print 'Number of unique customers:', len(DF.ILINK.unique().tolist())
ilinks = DF.ILINK.unique().tolist()

print 'Getting Column Headers'
#Builds the necessary columns/headers for the categorical Features

features = ['END_USE_DESC','MASTER_CHANNEL','PAY_TYPE_CD','FABRIC_CATEGORY_DESC','PRICE_CD']
columns = ['ILINK','DEPARTMENT_NAME']
HasFts = {}
PerFts = {}
for ft in features:
    HasFts[ft] = f.BuildHasColumn(DF,ft)
    columns = columns + f.BuildHasColumn(DF,ft)

for ft in features:
    PerFts[ft] = f.BuildPercentColumn(DF,ft)
    columns = columns + f.BuildPercentColumn(DF,ft)

print 'Getting Dummy Variables Now...'
#Creates Dummy variables for the features and their sub_categories/features
df_dummies = pd.get_dummies(DF[features])
df_new = pd.concat([DF[['ILINK','DEPARTMENT_NAME','MONTH','YEAR']],df_dummies],axis=1)

print 'Building Table Data Now...'
#Groups and aggregates dummy features/variables by ILINK,DEPT NAME, MONTH, YEAR
#Also removes various descriptions like "DESC" and "CD" from Labels
N = df_new.groupby(['ILINK','DEPARTMENT_NAME','MONTH','YEAR']).size()
df_new = df_new.groupby(['ILINK','DEPARTMENT_NAME','MONTH','YEAR']).sum()
df_new.insert(0,'Total_Purchased',N)

#replacing certain substrings from column headers
df_new.columns = [x.replace(' ','') for x in df_new.columns]
df_new.columns = [x.replace('_DESC','') for x in df_new.columns]
df_new.columns = [x.replace('_CD','') for x in df_new.columns]

print 'number of unique customers: ',len(df_new.index.levels[0])

#Final Column Headers and Features
#Categorical Features have specific labels
#UPPERCASENOSPACES_Has_Ft or UPPERCASENOSPACE_%_Ft
columns = ['ILINK','DEPARTMENT_NAME','MONTH','YEAR',
 'END_USE_Has_Core','END_USE_Has_Wearever','END_USE_Has_PureJill',
 'MASTER_CHANNEL_Has_D','MASTER_CHANNEL_Has_R',
 'PRICE_Has_FP','PRICE_Has_SP',
 'PAY_TYPE_Has_JJC','PAY_TYPE_Has_VISA','PAY_TYPE_Has_MC','PAY_TYPE_Has_AMEX',
 'PAY_TYPE_Has_DISC','PAY_TYPE_Has_DEBIT','PAY_TYPE_Has_CASH','PAY_TYPE_Has_CK','PAY_TYPE_Has_OTHER',
 'FABRIC_CATEGORY_Has_Cotton/CottonBl','FABRIC_CATEGORY_Has_Synthetic/SynBlend','FABRIC_CATEGORY_Has_Linen/LinenBl',
 'END_USE_%_Core','END_USE_%_Wearever','END_USE_%_PureJill',
 'MASTER_CHANNEL_%_D','MASTER_CHANNEL_%_R',
 'PRICE_%_FP','PRICE_%_SP',
 'PAY_TYPE_%_JJC','PAY_TYPE_%_VISA','PAY_TYPE_%_MC','PAY_TYPE_%_AMEX','PAY_TYPE_%_DISC',
 'PAY_TYPE_%_DEBIT','PAY_TYPE_%_CASH','PAY_TYPE_%_CK','PAY_TYPE_%_OTHER',
 'FABRIC_CATEGORY_%_Cotton/CottonBl','FABRIC_CATEGORY_%_Synthetic/SynBlend','FABRIC_CATEGORY_%_Linen/LinenBl']

#Algorithm for building categorical features
#Goes through each row
#each row it gets the customers info "ilink, dept, month, year"
#Then it builds the categorical features
#If "Has_X" category, it will check if purchases are greater than 0
#If "%_X" category, it will calculate the %_sold
table = []
for i,row in df_new.iterrows():
    ilink = i[0]
    dept = i[1]
    month = i[2]
    year = i[3]
    info = [ilink,dept,month,year]
    cols = row.index.tolist()
    count = 1
    for col in columns[4:]:

        #Follow this section if category is "Has_X"
        if 'Has' in col:
            ft = col.replace('_Has_','_')
            if 'PAY_TYPE' in ft:

                #split substring, so last element is the sub-ft
                X = col.split('_')
                if X[-1] != 'OTHER':
                    if '%s'%ft in cols:
                        if row['%s'%ft] > 0:
                            info.append(1)
                        else:
                            info.append(0)
                    else:
                        info.append(0)
                else:
                    ifOther = False
                    other = ['GCRD','AGC','','MALL', 'EXCH', 'TVCK', 'IHA', 'STCK']
                    for i in other:
                        if 'PAY_TYPE_%s'%i in cols:
                            if row['%s'%('PAY_TYPE_%s'%i)] > 0:
                                ifOther = True
                                break
                    if ifOther:
                        info.append(1)
                    else:
                        info.append(0)
            else:
                if '%s'%ft in cols:
                    if row['%s'%ft] > 0:
                        info.append(1)
                    else:
                        info.append(0)
                else:
                    info.append(0)
        #If feature is "%_X" follow this
        elif '%' in col:
            ft = col.replace('_%_','_')
            if 'PAY_TYPE' in ft:
                X = col.split('_')
                if X[-1] != 'OTHER':
                    if '%s'%ft in cols:
                        num = row['%s'%ft]*1.0
                        den = row['Total_Purchased']
                        value = round(num/den,3)
                        info.append(value)
                    else:
                        info.append(0.0)
                else:
                    other = ['GCRD','AGC','','MALL', 'EXCH', 'TVCK', 'IHA', 'STCK']
                    num = 0.0
                    for i in other:
                        if 'PAY_TYPE_%s'%i in cols:
                            num = num + row['%s'%('PAY_TYPE_%s'%i)]
                    den = row['Total_Purchased']
                    value = round(num/den,3)
                    info.append(value)

            else:
                if '%s'%ft in cols:
                    num = row['%s'%ft]*1.0
                    den = row['Total_Purchased']
                    value = round(num/den,3)
                    info.append(value)
                else:
                    info.append(0.0)
    #print len(info)
    table.append(tuple(info))

print 'Building DataFrame'
#print len(info)
categorical_table = pd.DataFrame(table,columns=columns)

print 'Saving DataFrame'
categorical_table.to_pickle('../data/categorical_fts_may2017_apr2017.pkl')

print 'Getting '
