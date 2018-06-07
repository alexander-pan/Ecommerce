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

#connect to aws
dbu = DBUtil("jjill_redshift","/home/jjill/.databases.conf")
print 'get first query'
query = """
select * from jjill.jjill_master_data where is_emailable_ind='Y' limit 1000000;
"""
df = dbu.get_df_from_query(query)

print 'second query'
query = """
select * from jjill.jjill_master_data
where is_emailable_ind='Y' and
department_name in ('Woven Shirts','Knit Tops','Pants','Dresses') and
order_date between '2017-01-01' and '2017-05-01'
limit 1000000;
"""
df1 = dbu.get_df_from_query(query)

df1.columns = map(str.upper, df1.columns)
df.columns = map(str.upper, df.columns)

print 'Number of unique customers:', len(df1.ILINK.unique().tolist())
ilinks = df1.ILINK.unique().tolist()

customers=ilinks

print 'Getting Thetas'
#Theta Value based on start-end timeframe
UserTheta = f.getCustTheta(customers,df1)

DFEXCUST = df.loc[df.ILINK.isin(customers)]
DFEXCUST.reset_index(drop=True,inplace=True)

#get indiv month datasets
aprildf = f.getDF(DFEXCUST,'2017-4-1','2017-5-1')
maydf = f.getDF(DFEXCUST,'2017-5-1','2017-6-1')
junedf = f.getDF(DFEXCUST,'2017-6-1','2017-7-1')
julydf = f.getDF(DFEXCUST,'2017-7-1','2017-8-1')
augdf = f.getDF(DFEXCUST,'2017-8-1','2017-9-1')
#septdf = f.getDF(DFEXCUST,'2017-9-1','2017-10-1')

def checkBuy(N):
    if N > 0:
        return 1.0
    else:
        return 0.0

columns = ['Ilink','N_ws','S_ws','D_ws','R_ws',
           'N_kt','S_kt','D_kt','R_kt',
           'N_d','S_d','D_d','R_d',
           'N_p','S_p','D_p','R_p']
cols = ['Ilink','month','year',
           'N_ws','S_ws','D_ws','R_ws','B_ws',
           'N_kt','S_kt','D_kt','R_kt', 'B_kt',
           'N_d','S_d','D_d','R_d','B_d',
           'N_p','S_p','D_p','R_p','B_p']

print 'Building Tables'
table = pd.DataFrame([],columns=cols)
for ilink in customers:
    try:
        temp = f.getTableRating([ilink],aprildf,columns,UserTheta)
	Nws = f.getTotalFreq(maydf[maydf.ILINK==ilink],'Woven Shirts')
	Nkt = f.getTotalFreq(maydf[maydf.ILINK==ilink],'Knit Tops')
	Nd = f.getTotalFreq(maydf[maydf.ILINK==ilink],'Dresses')
	Np = f.getTotalFreq(maydf[maydf.ILINK==ilink],'Pants')
        temp.insert(1,'month',value=5)
        temp.insert(2,'year',value=2017)
        temp.insert(7,'B_ws',value=checkBuy(Nws))
        temp.insert(12,'B_kt',value=checkBuy(Nkt))
        temp.insert(17,'B_d',value=checkBuy(Nd))
        temp.insert(22,'B_p',value=checkBuy(Np))
        table = pd.concat([table,temp])
    except:
        pass

    try:
        temp = f.getTableRating([ilink],maydf,columns,UserTheta)
        Nws = f.getTotalFreq(junedf[junedf.ILINK==ilink],'Woven Shirts')
        Nkt = f.getTotalFreq(junedf[junedf.ILINK==ilink],'Knit Tops')
        Nd = f.getTotalFreq(junedf[junedf.ILINK==ilink],'Dresses')
        Np = f.getTotalFreq(junedf[junedf.ILINK==ilink],'Pants')
        temp.insert(1,'month',value=6)
        temp.insert(2,'year',value=2017)
        temp.insert(7,'B_ws',value=checkBuy(Nws))
        temp.insert(12,'B_kt',value=checkBuy(Nkt))
        temp.insert(17,'B_d',value=checkBuy(Nd))
        temp.insert(22,'B_p',value=checkBuy(Np))
        table = pd.concat([table,temp])
    except:
        pass
    try:
        temp = f.getTableRating([ilink],junedf,columns,UserTheta)
	Nws = f.getTotalFreq(julydf[julydf.ILINK==ilink],'Woven Shirts')
        Nkt = f.getTotalFreq(julydf[julydf.ILINK==ilink],'Knit Tops')
        Nd = f.getTotalFreq(julydf[julydf.ILINK==ilink],'Dresses')
        Np = f.getTotalFreq(julydf[julydf.ILINK==ilink],'Pants')
        temp.insert(1,'month',value=7)
        temp.insert(2,'year',value=2017)
        temp.insert(7,'B_ws',value=checkBuy(Nws))
        temp.insert(12,'B_kt',value=checkBuy(Nkt))
        temp.insert(17,'B_d',value=checkBuy(Nd))
        temp.insert(22,'B_p',value=checkBuy(Np))
        table = pd.concat([table,temp])
    except:
        pass

    try:
        temp = f.getTableRating([ilink],julydf,columns,UserTheta)
        Nws = f.getTotalFreq(augdf[augdf.ILINK==ilink],'Woven Shirts')
        Nkt = f.getTotalFreq(augdf[augdf.ILINK==ilink],'Knit Tops')
        Nd = f.getTotalFreq(augdf[augdf.ILINK==ilink],'Dresses')
        Np = f.getTotalFreq(augdf[augdf.ILINK==ilink],'Pants')
        temp.insert(1,'month',value=8)
        temp.insert(2,'year',value=2017)
        temp.insert(7,'B_ws',value=checkBuy(Nws))
        temp.insert(12,'B_kt',value=checkBuy(Nkt))
        temp.insert(17,'B_d',value=checkBuy(Nd))
        temp.insert(22,'B_p',value=checkBuy(Np))
        table = pd.concat([table,temp])
    except:
        pass

#    try:
#        temp = f.getTableRating([ilink],augdf,columns,UserTheta)
#        temp.insert(1,'month',value=9)
#        temp.insert(2,'year',value=2017)
#        temp.insert(7,'B_ws',value=checkBuy(temp.N_ws.values[0]))
#        temp.insert(12,'B_kt',value=checkBuy(temp.N_kt.values[0]))
#        temp.insert(17,'B_d',value=checkBuy(temp.N_d.values[0]))
#        temp.insert(22,'B_p',value=checkBuy(temp.N_p.values[0]))
#        table = pd.concat([table,temp])
#    except:
#        pass

table = table[(table[cols[3:]].T != 0).any()]

print 'Saving Table/Data'
table.to_pickle('./dataset_AllCustomers.pkl')
