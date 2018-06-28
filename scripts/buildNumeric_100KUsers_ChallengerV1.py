import pandas as pd

#accessing aws data
import sys
sys.path.append('../utils')
from db_utils import DBUtil

#connect to aws
#dbu = DBUtil("jjill_redshift","/home/jjill/.databases.conf")
dbu = DBUtil("jjill_redshift","../../databases/database.conf")

print 'get first query'
query = """
with all_users as (
  select ilink
  from jjill.jjill_keyed_data
  where department_name in ('Knit Tops','Woven Shirts','Dresses','Pants')
  and order_date::date > '2018-01-01'
  and order_date::date < '2018-06-01'
  group by ilink
  limit 100
),

all_departments as (
  select department_name
  from jjill.jjill_keyed_data
  where order_date::date > '2018-01-01'
  and order_date::date < '2018-06-01'
  group by department_name
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
  t3.ilink, date_part(mon,t3.ref_date) as ref_date_month, date_part(yr,t3.ref_date) as ref_date_year, t3.department_name,
  max(t3.most_recent_past_order_date) as most_recent_past_order_date,
  max(t3.num_past_orders) as num_past_orders,
  max(t3.sum_past_shipped_sold_amt) as sum_past_shipped_sold_amt,
  max(t3.avg_past_shipped_sold_amt) as avg_past_shipped_sold_amt,
  max(t3.stddev_past_shipped_sold_amt) as stddev_past_shipped_sold_amt,
  max(t3.var_past_shipped_sold_amt) as var_past_shipped_sold_amt,
  max(t3.sum_past_discount) as sum_past_discount,
  max(t3.avg_past_discount) as avg_past_discount,
  max(t3.stddev_past_discount) as stddev_past_discount,
  max(t3.var_past_discount) as var_past_discount

from (
  select
      t1.ilink, t1.ref_date as ref_date,--date_part(mon,t1.ref_date::date) as ref_date_month, date_part(yr,t1.ref_date::date) as ref_date_year,
  (
    case
         when t1.department_name not in('Woven Shirts','Dresses','Knit Tops','Pants') then 'Other'
         else t1.department_name
    end
  ),
      max(t2.order_date) as most_recent_past_order_date,
      count(t2.order_date) as num_past_orders,
      --sum(case when t2.shipped_cost_amt = '' then 0.0 else t2.shipped_cost_amt::float end) as sum_past_shipped_cost_amt,
      sum(case when t2.shipped_sold_amt = '' then 0.0 else t2.shipped_sold_amt::float end) as sum_past_shipped_sold_amt,
      avg(case when t2.shipped_sold_amt = '' then 0.0 else t2.shipped_sold_amt::float end) as avg_past_shipped_sold_amt,
      stddev_samp(case when t2.shipped_sold_amt = '' then 0.0 else t2.shipped_sold_amt::float end) as stddev_past_shipped_sold_amt,
      var_samp(case when t2.shipped_sold_amt = '' then 0.0 else t2.shipped_sold_amt::float end) as var_past_shipped_sold_amt,
      sum(case when t2.discount = '' then 0.0 else t2.discount::float end) as sum_past_discount,
      avg(case when t2.discount = '' then 0.0 else t2.discount::float end) as avg_past_discount,
      stddev_samp(case when t2.discount = '' then 0.0 else t2.discount::float end) as stddev_past_discount,
      var_samp(case when t2.discount = '' then 0.0 else t2.discount::float end) as var_past_discount

  from reference_data as t1
  join jjill.jjill_keyed_data as t2
  on (
    t1.ilink = t2.ilink
    and t1.department_name = t2.department_name
    and t1.ref_date > t2.order_date
    and t1.ref_date::date - 30 < t2.order_date
  )
  group by t1.ilink,(
    case
         when t1.department_name not in('Woven Shirts','Dresses','Knit Tops','Pants') then 'Other'
         else t1.department_name
    end
  ),t1.ref_date
) as t3
group by t3.ilink, t3.department_name, ref_date_month,ref_date_year
order by t3.ilink, ref_date_month,ref_date_year,t3.department_name;
"""
df = dbu.get_df_from_query(query)

print 'Getting Dummies,Formatting, and Saving'
df_dummies = pd.get_dummies(df['department_name'])
df_dummies.columns = ['BOUGHT_%s' % x for x in df_dummies.columns]
df_new = pd.concat([df,df_dummies],axis=1)
df_new.columns = map(str.upper,df_new.columns)
df_new = df_new.groupby('DEPARTMENT_NAME').transform(lambda x: x.fillna(x.mean()))
df_new.to_pickle('../data/numericalFts_100Users_ChallengerV1.pkl')