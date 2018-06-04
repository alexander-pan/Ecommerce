import numpy as np
import pandas as pd
import math
from datetime import datetime as dt, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV,ShuffleSplit
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
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

print 'getting first query'
query = """
select * from jjill.jjill_master_data where is_emailable_ind='Y' limit 100;
"""
df = dbu.get_df_from_query(query)

