#accessing aws data
import sys
sys.path.append('../utils')
from db_utils import DBUtil

#connect to aws
#dbu = DBUtil("jjill_redshift","/home/jjill/.databases.conf")
dbu = DBUtil("jjill_redshift","../../databases/database.conf")

print 'get first query'
query = """
select ilink,
department_name,
date_part(mon,order_date) as month,
date_part(yr,order_date) as year,
count(*) as Total_Bought,
sum(shipped_sold_amt) as shipped_sold_amt, avg(shipped_sold_amt) as avg_shipped_sold_amt,
stddev_samp(shipped_sold_amt) as std_shipped_sold_amt, var_samp(shipped_sold_amt) as var_shipped_sold_amt,
sum(discount) as discount, avg(discount) as avg_discount,
stddev_samp(discount) as std_discount, var_samp(discount) as var_discount
from jjill.jjill_keyed_data
where is_emailable_ind='Y' and
department_name in ('Woven Shirts','Knit Tops','Pants','Dresses')
and order_date between '2017-05-01' and '2018-04-30'
group by 1,2,3,4
order by ilink;
"""
df = dbu.get_df_from_query(query)
df.to_pickle('../data/numericFts_may2017_apr2018.pkl')
