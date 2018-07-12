import pandas as pd

#accessing aws data
import sys
sys.path.append('../../utils')
from db_utils import DBUtil

#connect to aws
dbu = DBUtil("jjill_redshift","/home/jjill/.databases.conf")
#dbu = DBUtil("jjill_redshift","../../databases/database.conf")

print 'get first query'
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
  limit 3000
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
  date_part(yr,t3.ref_date) as ref_date_year, t3.department_name
from
(
  select
    t1.ilink, t1.ref_date as ref_date,
    (
      case
           when t1.department_name not in('Woven Shirts','Dresses','Knit Tops','Pants') then 'Other_Dept'
           else t1.department_name
      end
    )
  from reference_data as t1
  join jjill.jjill_keyed_data as t2 on (
                                    t1.ilink = t2.ilink
                                    and t1.department_name = (
                                                                case
                                                                     when t2.department_name not in('Woven Shirts','Dresses','Knit Tops','Pants') then 'Other_Dept'
                                                                     else t2.department_name
                                                                end
                                                              )
                                    and t1.ref_date > t2.order_date
                                    and t1.ref_date::date + 30 > t2.order_date
                                  )
  where t2.order_date between '2018-01-01' and '2018-06-01'
  group by t1.ilink, t1.ref_date,
  (
    case
         when t1.department_name not in('Woven Shirts','Dresses','Knit Tops','Pants') then 'Other_Dept'
         else t1.department_name
    end
  )
) as t3
group by t3.ilink, ref_date_month, ref_date_day, ref_date_year, t3.department_name
order by t3.ilink, ref_date_month, ref_date_day, ref_date_year, t3.department_name;
"""

df = dbu.get_df_from_query(query)

#print 'Getting Dummies'
df_dummies = pd.get_dummies(df['department_name'])
df_dummies.columns = ['BOUGHT_%s' % x for x in df_dummies.columns]
df_new = pd.concat([df[['ilink','ref_date_month','ref_date_day','ref_date_year']],df_dummies],axis=1)
df_new = df_new.groupby(['ilink','ref_date_month','ref_date_day','ref_date_year']).sum().reset_index()
DF = df_new.copy()

print "Getting 2nd Query"
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
  limit 3000
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
  t3.ilink, date_part(mon,t3.ref_date) as ref_date_month,
  date_part(day,t3.ref_date) as ref_date_day,
  date_part(yr,t3.ref_date) as ref_date_year,
  t3.department_name,
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
      t1.ilink, t1.ref_date as ref_date,
      max(t2.order_date) as most_recent_past_order_date,
      count(t2.order_date) as num_past_orders,
      (
      case
           when t1.department_name not in('Woven Shirts','Dresses','Knit Tops','Pants') then 'Other_Dept'
           else t1.department_name
      end
      ),
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
    and t1.department_name = (
                                case
                                     when t2.department_name not in('Woven Shirts','Dresses','Knit Tops','Pants') then 'Other_Dept'
                                     else t2.department_name
                                end
                              )
    and t1.ref_date > t2.order_date
    and t1.ref_date::date - 30 < t2.order_date
  )
  group by t1.ilink,t1.ref_date,(
      case
           when t1.department_name not in('Woven Shirts','Dresses','Knit Tops','Pants') then 'Other_Dept'
           else t1.department_name
      end
    )
) as t3
 group by t3.ilink, ref_date_month, ref_date_day, ref_date_year,t3.department_name
 order by t3.ilink,ref_date_month, ref_date_day, ref_date_year;
"""
df = dbu.get_df_from_query(query)
columns = ['ilink','ref_date_month','ref_date_day','ref_date_year','num_past_orders',
	   'sum_past_shipped_sold_amt','avg_past_shipped_sold_amt','stddev_past_shipped_sold_amt',
	   'var_past_shipped_sold_amt','sum_past_discount','avg_past_discount','stddev_past_discount',
	   'var_past_discount']
#print 'Getting Dummies'
df_dummies = pd.get_dummies(df['department_name'])
df_dummies.columns = ['BOUGHT_PAST_%s' % x for x in df_dummies.columns]
df_new = pd.concat([df[columns],df_dummies],axis=1)
df_new = df_new.groupby(['ilink','ref_date_month','ref_date_day','ref_date_year']).sum().reset_index()

#print 'Joining Tables and Saving...'
df_new = df_new.merge(DF,on=['ilink','ref_date_month','ref_date_day','ref_date_year'])
df_new.columns = map(str.upper,df_new.columns)
df_new.fillna(df_new.mean(),inplace=True)
#df_new.dropna(inplace=True)
#df_new.to_pickle('../data/numericalFts_15kUsers_BaselineV1.pkl')
df_new.to_pickle('../../data/Baseline/numericalFts_3kUsers_Baseline.pkl')
