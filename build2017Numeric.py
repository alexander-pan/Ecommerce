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
dbu = DBUtil("komodo_redshift","/home/jjill/.databases.conf")
print 'get first query'
query = """
select ilink,department_name, count(*) as Total_Bought,
sum(shipped_sold_amt) as shipped_sold_amt, avg(shipped_sold_amt) as avg_shipped_sold_amt,
stddev_samp(shipped_sold_amt) as std_shipped_sold_amt, var_samp(shipped_sold_amt) as var_shipped_sold_amt,
sum(discount) as discount, avg(discount) as avg_discount,
stddev_samp(discount) as std_discount, var_samp(discount) as var_discount
from jjill.jjill_keyed_data
where is_emailable_ind='Y' and
department_name in ('Woven Shirts','Knit Tops','Pants','Dresses')
and order_date between '2017-01-01' and '2017-12-31'
group by ilink,department_name order by ilink;
"""
df = dbu.get_df_from_query(query)
df.to_pickle("./data/numeric_fts_2017.pkl")
