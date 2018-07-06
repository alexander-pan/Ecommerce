import numpy as np
import pandas as pd

#modules for specific application
import random
import dill

#accessing aws data
import sys
sys.path.append('../../utils')
from db_utils import DBUtil

#connect to aws
dbu = DBUtil("jjill_redshift","/home/jjill/.databases.conf")
#dbu = DBUtil("jjill_redshift","../../databases/database.conf")

#Queries reflec the features you choose, so make sure to change query based on
#this
print 'Getting First Query'
query = """
with all_users as (
  select ilink
  from jjill.jjill_keyed_data
  where department_name in ('Knit Tops','Woven Shirts','Dresses','Pants')
  and is_emailable_ind='Y'
  and order_date::date > '2018-01-01'
  and order_date::date < '2018-06-01'
  group by ilink
  order by ilink
  limit 15000
),

all_departments as (
  select
(
  case
       when department_name not in('Woven Shirts','Dresses','Knit Tops','Pants') then 'Other_Dept'
       else department_name
  end
)
  from jjill.jjill_keyed_data
  where order_date::date > '2018-01-01'
  and order_date::date < '2018-06-01'
  group by (
  case
       when department_name not in('Woven Shirts','Dresses','Knit Tops','Pants') then 'Other_Dept'
       else department_name
  end
)
),

all_dates as (
  select order_date as ref_date
  from jjill.jjill_keyed_data
  where order_date::date > '2018-01-01'
  and order_date::date < '2018-06-01'
  group by order_date
),

reference_data as (
  select distinct ilink, department_name, ref_date
  from all_users
  join all_departments on (1 = 1)
  join all_dates on (1 = 1)
)

select
t3.ilink, date_part(mon,t3.ref_date) as ref_date_month, date_part(day,t3.ref_date) as ref_date_day,
date_part(yr,t3.ref_date) as ref_date_year,
t3.end_use_desc,t3.price_cd,t3.pay_type_cd,t3.master_channel,t3.fabric_category_desc
from (
  select
      t1.ilink, t1.ref_date as ref_date,(
      case
           when t1.department_name not in('Woven Shirts','Dresses','Knit Tops','Pants') then 'Other_Dept'
           else t1.department_name
      end
    ),
    t2.end_use_desc,t2.price_cd,
    (
      case
        when t2.pay_type_cd not in ('VISA','MC','DISC','CASH','DEBIT','JJC','CK') then 'Other_Paytype'
        else t2.pay_type_cd
      end
    ),t2.master_channel,
    (
      case
        when t2.fabric_category_desc not in ('Cotton/Cotton Bl','Synthetic/Syn Blend','Linen/Linen Bl') then 'Other_Fabric'
        else t2.fabric_category_desc
      end
    )
  from reference_data as t1
  join jjill.jjill_keyed_data as t2
  on (
    t1.ilink = t2.ilink
    and t1.department_name = t2.department_name
    and t1.ref_date > t2.order_date
    and t1.ref_date::date - 30 < t2.order_date
  )
  group by t1.ilink,t1.ref_date,(
      case
           when t1.department_name not in('Woven Shirts','Dresses','Knit Tops','Pants') then 'Other_Dept'
           else t1.department_name
      end
    ),t2.end_use_desc,t2.price_cd,(
      case
        when t2.pay_type_cd not in ('VISA','MC','DISC','CASH','DEBIT','JJC','CK') then 'Other_Paytype'
        else t2.pay_type_cd
      end
    ),t2.master_channel,
    (
      case
        when t2.fabric_category_desc not in ('Cotton/Cotton Bl','Synthetic/Syn Blend','Linen/Linen Bl') then 'Other_Fabric'
        else t2.fabric_category_desc
      end
    )
  )as t3
group by t3.ilink,ref_date_month,ref_date_day,ref_date_year,t3.end_use_desc,t3.price_cd,t3.pay_type_cd,t3.master_channel,t3.fabric_category_desc
order by t3.ilink, ref_date_month,ref_date_day,ref_date_year;
"""
df = dbu.get_df_from_query(query)

print "Build 'Has' Categorical Fts"

#Include the feature labels to get
fts = ['end_use_desc','price_cd','pay_type_cd','master_channel','fabric_category_desc']
df_new = df[['ilink','ref_date_month','ref_date_day','ref_date_year']].copy()
cols = []

#Create Dummies and concat onto existing copy
for ft in fts:
    df_dummies = pd.get_dummies(df[ft])
    df_dummies.columns = ['Has_%s'%x for x in df_dummies.columns]
    cols.append(df_dummies.columns)
    df_new = pd.concat([df_new,df_dummies],axis=1)

#groupby ilink,month,day,year
df_new = df_new.groupby(['ilink','ref_date_month','ref_date_day','ref_date_year']).sum().reset_index()

#Change all values that are greater than 0 to 1 in all new cols
for col in cols:
    for x in col:
        df_new[x] = np.where(df_new[x]>0,1,0)
DF = df_new.copy()

#create master feature list
ft_cols = [item for sublist in cols for item in sublist]

print "Getting 2nd Query"

#this query doesn't groupby like the first, this way I can groupby and sum to
#get total values of said specific dummy/sub-categories to calc. percentages
query = """
with all_users as (
  select ilink
  from jjill.jjill_keyed_data
  where department_name in ('Knit Tops','Woven Shirts','Dresses','Pants')
  and is_emailable_ind='Y'
  and order_date::date > '2018-01-01'
  and order_date::date < '2018-06-01'
  group by ilink
  order by ilink
  limit 15000
),

all_departments as (
  select
(
  case
       when department_name not in('Woven Shirts','Dresses','Knit Tops','Pants') then 'Other_Dept'
       else department_name
  end
)
  from jjill.jjill_keyed_data
  where order_date::date > '2018-01-01'
  and order_date::date < '2018-06-01'
  group by (
  case
       when department_name not in('Woven Shirts','Dresses','Knit Tops','Pants') then 'Other_Dept'
       else department_name
  end
)
),

all_dates as (
  select order_date as ref_date
  from jjill.jjill_keyed_data
  where order_date::date > '2018-01-01'
  and order_date::date < '2018-06-01'
  group by order_date
),

reference_data as (
  select distinct ilink, department_name, ref_date
  from all_users
  join all_departments on (1 = 1)
  join all_dates on (1 = 1)
)

select
t3.ilink, date_part(mon,t3.ref_date) as ref_date_month, date_part(day,t3.ref_date) as ref_date_day,
date_part(yr,t3.ref_date) as ref_date_year,
t3.end_use_desc,t3.price_cd,t3.pay_type_cd,t3.master_channel,t3.fabric_category_desc
from (
  select
      t1.ilink, t1.ref_date as ref_date,(
      case
           when t1.department_name not in('Woven Shirts','Dresses','Knit Tops','Pants') then 'Other_Dept'
           else t1.department_name
      end
    ),
    t2.end_use_desc,t2.price_cd,
    (
      case
        when t2.pay_type_cd not in ('VISA','MC','DISC','CASH','DEBIT','JJC','CK') then 'Other_Paytype'
        else t2.pay_type_cd
      end
    ),t2.master_channel,
    (
      case
        when t2.fabric_category_desc not in ('Cotton/Cotton Bl','Synthetic/Syn Blend','Linen/Linen Bl') then 'Other_Fabric'
        else t2.fabric_category_desc
      end
    )
  from reference_data as t1
  join jjill.jjill_keyed_data as t2
  on (
    t1.ilink = t2.ilink
    and t1.department_name = t2.department_name
    and t1.ref_date > t2.order_date
    and t1.ref_date::date - 30 < t2.order_date
  )
 )as t3
 order by t3.ilink, ref_date_month,ref_date_day,ref_date_year;
"""
df = dbu.get_df_from_query(query)

print "Building '%' Bought in Fts"
temp = df[['ilink','ref_date_month','ref_date_day','ref_date_year']].copy()

#Getting Dummies
cols = []
for ft in fts:
    df_dummies = pd.get_dummies(df[ft])
    df_dummies.columns = ['Num_%s'%x for x in df_dummies.columns]
    cols.append(df_dummies.columns)
    temp = pd.concat([temp,df_dummies],axis=1)
N = temp.groupby(['ilink','ref_date_month','ref_date_day','ref_date_year']).size()
df_new = temp.groupby(['ilink','ref_date_month','ref_date_day','ref_date_year']).sum()
df_new.insert(0,'Total_Purchased',N)
df_new.reset_index(inplace=True)

print "Calculating '%'"
new_cols = []
for col in cols:
    for x in col:
        a,b = x.split('Num_')
        new_col = '%%_in_%s'%b
        new_cols.append(new_col)
        df_new[new_col] = df_new[x]/df_new['Total_Purchased']

features = ['ilink','ref_date_month','ref_date_day','ref_date_year'] +ft_cols + new_cols
df = DF.merge(df_new,on=['ilink','ref_date_month','ref_date_day','ref_date_year'])
df = df[features]
df.columns = [str.upper(x.replace(' ','_')) for x in df.columns]

print "Saving DataFrame"
df.to_pickle('../../data/Baseline/categoricalFts_15kUsers_BaselineV1.pkl')
