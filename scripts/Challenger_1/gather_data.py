###############
##  IMPORTS  ##
###############

from datetime import datetime, timedelta
from random import random
from pprint import pprint
import pandas as pd
import numpy as np
import ast
import sys

sys.path.append('../../utils')
from db_utils import DBUtil

#################
##  FUNCTIONS  ##
#################

# Defining Continuous Variables On The Sample Reference Level
def define_continuous_ref_level_vars(params):
    max_date_diff = str((datetime.strptime(params['max_date'], '%Y-%m-%d') - datetime.strptime(params['min_date'], '%Y-%m-%d')).days)
    numerical_vars = ['original_retail_price_amt', 'shipped_cost_amt', 'shipped_sold_amt', 'margin', 'discount', 'markdown']
    numerical_vars_fields = [
        {
            'sum_past_' + x: {'definition': "sum(CASE WHEN t2." + x + " = '' THEN 0.0 ELSE t2." + x + "::float END)::float",
                              'default_value': 0.0},
            'sum_past_' + x + '_in_department': {'definition': """sum(CASE WHEN t1.department_name != t2.department_name THEN 0.0 
                                                                                  WHEN t2.""" + x + """ = '' THEN 0.0 
                                                                                  ELSE t2.""" + x + """::float END)::float""",
                                                 'default_value': 0.0},
            'avg_past_' + x: {'definition': "avg(CASE WHEN t2." + x + " = '' THEN 0.0 ELSE t2." + x + "::float END)::float",
                              'default_value': 0.0},
            'avg_past_' + x + '_in_department': {'definition': """avg(CASE WHEN t1.department_name != t2.department_name THEN 0.0 
                                                                                  WHEN t2.""" + x + """ = '' THEN 0.0 
                                                                                  ELSE t2.""" + x + """::float END)::float""",
                                                 'default_value': 0.0}
        }
        for x in numerical_vars
    ]
    numerical_vars_fields.append(
        {
            'most_recent_past_order_days_ago': {'definition': 'min(t1.ref_date - t2.order_date)::float',
                                                'default_value': max_date_diff},
            'most_recent_past_order_days_ago_in_department': {'definition': """min(CASE WHEN t1.department_name != t2.department_name THEN """ + max_date_diff + """
                                                                                        ELSE t1.ref_date - t2.order_date END)::float""",
                                                              'default_value': max_date_diff},

            'num_past_orders': {'definition': 'count(t2.order_date)::float',
                                'default_value': 0.0},
            'num_past_orders_in_department': {'definition': 'sum(CASE WHEN t1.department_name != t2.department_name THEN 0 ELSE 1 END)::float',
                                              'default_value': 0.0},
        }
    )
    return numerical_vars, numerical_vars_fields

# Defining Categorical Variables On The Sample Reference Level
def define_categorical_ref_level_vars():
    categorical_vars = {
        # 'day_of_the_wk': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        # 'color_family_desc': ['Blue', 'Black', 'White/Ivory', 'Grey', 'Pink', 'Green'],
        # 'fabric_category_desc': ['Cotton/Cotton Bl', 'Synthetic/Syn Blend' ,'Linen/Linen Bl'],
        # 'pay_type_cd': ['VISA', 'MC', 'DISC', 'CASH', 'DEBIT', 'JJC', 'CK'],
        # 'end_use_desc': ['Core', 'Wearever', 'Pure Jill'],
        # 'price_cd': ['SP', 'FP'],
        # 'master_channel': ['D', 'R']
        # 'is_jjch_ind': ['Y'],
        # 'is_callable_ind': ['Y']
    }
    categorical_vars_fields = [
        {
            # 'has_' + x + '_' + y.split(' ')[0].split('/')[0].lower(): {'definition': "max(CASE WHEN t2." + x + " IN ('" + y + "') THEN 1.0 ELSE 0.0 END)::float",
            #                                                            'default_value': 0.0},
            'prop_' + x + '_' + y.split(' ')[0].split('/')[0].lower(): {'definition': """(CASE WHEN count(t2.ilink)::int = 0 THEN 0.0
                                                                                               ELSE sum(CASE WHEN t2.""" + x + """ IN ('""" + y + """') THEN 1.0 
                                                                                                             ELSE 0.0 END) / count(t2.ilink)::float
                                                                                          END)::float""",
                                                                        'default_value': 0.0}
        }
        for x in categorical_vars for y in categorical_vars[x]
    ]
    return categorical_vars, categorical_vars_fields

# Defining Other Binary Variables On The Sample Reference Level
def define_other_binary_ref_level_vars():
    other_binary_vars = ['first_order_date', 'first_catalog_order_date', 'first_retail_order_date', 'first_web_order_date', 'prior_order_date']
    other_binary_vars_fields = [
        {
            'past_' + x + '_exists': {'definition': "max(CASE WHEN (t2." + x + " IS NOT NULL) AND (t2." + x + "::text != '') THEN 1.0 ELSE 0.0 END)::float",
                                      'default_value': 0.0},
            'past_' + x + '_exists_in_department': {'definition': """max(CASE WHEN t1.department_name != t2.department_name THEN 0.0
                                                                          WHEN (t2.""" + x + """ IS NOT NULL) AND (t2.""" + x + """::text != '') THEN 1.0 
                                                                          ELSE 0.0 END)::float""",
                                                    'default_value': 0.0}
        }
        for x in other_binary_vars
    ]
    return other_binary_vars, other_binary_vars_fields

# Defining All Variables On The Department Level
def define_department_level_vars(numerical_vars):
    department_vars = {}
    for x in numerical_vars:
        department_vars['department_avg_' + x] = 'avg(' + x + ')'
    department_categorical_vars = {
        #'day_of_the_wk': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        #'color_family_desc': ['Blue', 'Black', 'White/Ivory', 'Grey', 'Pink', 'Green'],
        'fabric_category_desc': ['Cotton/Cotton Bl', 'Synthetic/Syn Blend' ,'Linen/Linen Bl'],
        'pay_type_cd': ['VISA', 'MC', 'DISC', 'CASH', 'DEBIT', 'JJC', 'CK'],
        'end_use_desc': ['Core', 'Wearever', 'Pure Jill'],
        'price_cd': ['SP', 'FP'],
        'master_channel': ['D', 'R']
    }
    for x in department_categorical_vars:
        for y in department_categorical_vars[x]:
            val = "sum(CASE WHEN " + x + " IN ('" + y + "') THEN 1.0 ELSE 0.0 END) / count(ilink)::float"
            department_vars['department_' + x + '_' + y.split(' ')[0].split('/')[0].lower() + '_pct'] = val
    return department_vars

# Defining Any Other Variables Not Fitting In The Above Categories
def define_other_vars(params):
    other_vars = {

        'ref_season_winter': {'definition': "max(CASE WHEN split_part(t1.ref_date::date, '-', 2)::int IN (12,1,2) THEN 1.0 ELSE 0.0 END)::float",
                              'default_value': 0.0},
        'ref_season_spring': {'definition': "max(CASE WHEN split_part(t1.ref_date::date, '-', 2)::int IN (3,4,5) THEN 1.0 ELSE 0.0 END)::float",
                              'default_value': 0.0},
        'ref_season_summer': {'definition': "max(CASE WHEN split_part(t1.ref_date::date, '-', 2)::int IN (6,7,8) THEN 1.0 ELSE 0.0 END)::float",
                              'default_value': 0.0},
        'ref_season_fall': {'definition': "max(CASE WHEN split_part(t1.ref_date::date, '-', 2)::int IN (9,10,11) THEN 1.0 ELSE 0.0 END)::float",
                            'default_value': 0.0},

        'user_age': {'definition': "(max(t1.ref_date - t2.birth_date) / 365.0)::float",
                     'default_value': 0.0},

        'past_order_dates': {'definition': "listagg(distinct t2.order_date, ' ')",
                             'default_value': ''},
        'past_order_dates_in_department': {'definition': "listagg(distinct (CASE WHEN t1.department_name != t2.department_name THEN NULL ELSE t2.order_date END), ' ')",
                                           'default_value': ''}
    }
    # valid_departments = ast.literal_eval(params['valid_departments'])
    # for x in valid_departments:
    #     key = 'ref_department_' + x.replace(' ','_').lower()
    #     other_vars[key] = {'definition': "max(CASE WHEN t1.department_name = '" + x + "' THEN 1.0 ELSE 0.0 END)::float",
    #                        'default_value': 0.0}
    return other_vars

# Combining All Variables
def combine_vars(numerical_vars_fields, categorical_vars_fields, other_binary_vars_fields, other_vars):
    output_vars = {}
    for i in range(0,len(numerical_vars_fields)): 
        for y in numerical_vars_fields[i]:
            output_vars[y] = numerical_vars_fields[i][y]
    for i in range(0,len(categorical_vars_fields)): 
        for y in categorical_vars_fields[i]:
            output_vars[y] = categorical_vars_fields[i][y]
    for i in range(0,len(other_binary_vars_fields)): 
        for y in other_binary_vars_fields[i]:
            output_vars[y] = other_binary_vars_fields[i][y]
    for x in other_vars:
        output_vars[x] = other_vars[x]
    return output_vars

# Running SQL To Gather Training / Prediction Data
def run_data_query(output_vars, department_vars, params, dbu):
    min_reference_date = str(datetime.strptime(params['min_date'], '%Y-%m-%d') + timedelta(days=int(params['lookback_window']))).split(' ')[0]
    sql = """
        WITH all_users AS (
            SELECT ilink
            FROM jjill.jjill_keyed_data
            WHERE department_name in ('Knit Tops','Woven Shirts','Dresses','Pants')
            AND is_emailable_ind='Y'
            AND order_date::date > '""" + params['min_date'] + """'
            AND order_date::date < '""" + params['max_date'] + """'
            GROUP BY ilink
            ORDER BY ilink
            LIMIT """ + params['num_users'] + """
        ),

        all_departments AS (
            SELECT
                (CASE WHEN department_name NOT IN """ + params['valid_departments'] + """ THEN 'Other_Dept' ELSE department_name END)
            FROM jjill.jjill_keyed_data
            WHERE order_date::date > '""" + params['min_date'] + """'
            AND order_date::date < '""" + params['max_date'] + """'
            GROUP BY 1
        ),

        all_dates AS (
            SELECT order_date AS ref_date
            FROM jjill.jjill_keyed_data
            WHERE order_date::date > '""" + min_reference_date + """'
            AND order_date::date < '""" + params['max_date'] + """'
            GROUP BY order_date
        ),

        all_department_date_pairs_current_view AS (
            SELECT 
                order_date::date AS order_date, 
                department_name,
                """ + ',\n\t\t'.join(sorted([ department_vars[x] + ' AS ' + x for x in department_vars ])) + """
            FROM jjill.jjill_keyed_data
            WHERE order_date::date > '""" + params['min_date'] + """'
            AND order_date::date < '""" + params['max_date'] + """'
            GROUP BY order_date::date, department_name
        ),

        all_user_department_date_tuples AS (
            SELECT distinct ilink, department_name, ref_date
            FROM all_users
            JOIN all_departments on (1 = 1)
            JOIN all_dates on (1 = 1)
        ),

        reference_data AS (
            SELECT *
            FROM
            (
                SELECT 
                    ta.ilink, ta.department_name, ta.ref_date,
                    """ + ',\n\t\t'.join(sorted([ 'tb.' + x for x in department_vars ])) + """,
                    row_number() over(partition by ta.ilink, ta.department_name, ta.ref_date order by tb.order_date desc) AS date_rank
                FROM all_user_department_date_tuples AS ta
                JOIN all_department_date_pairs_current_view AS tb ON (
                    ta.department_name = tb.department_name
                    AND ta.ref_date > tb.order_date
                )
            ) AS tc
            WHERE tc.date_rank = 1
        )

        --- main query ---
        SELECT
            t3.ilink, t3.ref_date, t3.department_name,
            max(CASE WHEN t4.order_date IS NULL THEN 0 ELSE 1 END) AS outcome,
            """ + ',\n\t\t'.join(sorted([ 'max(t3.' + x + ') AS ' + x for x in department_vars ])) + """,
            """ + ',\n\t\t'.join(sorted([ 'max(t3.' + x + ') AS ' + x for x in output_vars ])) + """
        FROM
        (
            SELECT 
                t1.ilink, t1.ref_date::date as ref_date, t1.department_name,
                """ + ',\n\t\t'.join(sorted([ 'max(t1.' + x + ') AS ' + x for x in department_vars ])) + """,
                """ + ',\n\t\t'.join([ output_vars[x]['definition'] + ' AS ' + x for x in output_vars ]) + """
            FROM reference_data AS t1
            LEFT JOIN jjill.jjill_keyed_data AS t2 ON (
                t1.ilink = t2.ilink 
                AND t1.ref_date > t2.order_date
                AND t1.ref_date::date - """ + params['lookback_window'] + """ < t2.order_date
            )
            WHERE ((t2.order_date BETWEEN '""" + params['min_date'] + """' AND '""" + params['max_date'] + """') OR (t2.order_date IS NULL))
            GROUP BY t1.ilink, t1.ref_date::date, t1.department_name
        ) AS t3
        LEFT JOIN jjill.jjill_keyed_data AS t4 ON (
            t3.ilink = t4.ilink 
            AND t3.department_name = t4.department_name
            AND t3.ref_date <= t4.order_date
            AND t3.ref_date + 30 > t4.order_date
        )
        WHERE ((t4.order_date BETWEEN '""" + params['min_date'] + """' AND '""" + params['max_date'] + """') OR (t4.order_date is null))
        GROUP BY t3.ilink, t3.ref_date, t3.department_name
        ORDER BY random()
    """

    print('running queries...'); #print(sql)
    queried_data = dbu.get_df_from_query(sql, server_cur=True, itersize=1000)
    columns = sql.split('- main query -')[1].split('SELECT')[1].split('FROM')[0].split(',')
    columns = [ columns[i].strip().lower().replace('t3.','') for i in range(0,len(columns)) ]
    columns = [ columns[i].split(' as ')[-1] for i in range(0,len(columns)) ]
    return queried_data, columns

# Creating Variables Based On All Distinct Order Dates
def distinct_order_date_based_variables(row, output_vars):
    suffixes = ['', '_in_department']
    for suffix in suffixes:
        row['avg_days_between_orders' + suffix] = float(output_vars['most_recent_past_order_days_ago' + suffix]['default_value'])
        past_order_dates = row['past_order_dates' + suffix]
        if (past_order_dates != '') and (past_order_dates != None):
            past_order_dates = past_order_dates.split(' ')
            if len(past_order_dates) > 1:
                past_order_dates = sorted(past_order_dates, reverse=True)
                past_order_date_diffs = [ float((datetime.strptime(past_order_dates[i], '%Y-%m-%d') - datetime.strptime(past_order_dates[i+1], '%Y-%m-%d')).days) 
                                          for i in range(0,len(past_order_dates)-1) ]
                row['avg_days_between_orders' + suffix] = np.mean(past_order_date_diffs)
        del row['past_order_dates' + suffix]
    return row

# Outputting Training / Prediction Data To A File
def save_data(queried_data, columns, output_vars):
    f = open('train_data.txt','w')
    class_sizes = {0:0, 1:0}; c = 0; n_estimate = 20000
    for item in queried_data:
        row = dict(zip(columns, item))

        ###
        row = distinct_order_date_based_variables(row, output_vars)
        ###

        class_sizes[row['outcome']] += 1
        if c == n_estimate:
            majority_class = max(class_sizes, key=class_sizes.get)
            balance_ratio = min(class_sizes.values()) / float(max(class_sizes.values()))
            print('\tmajority_class: ' + str(majority_class))
            print('\tbalance_ratio: ' + str(balance_ratio))
        if c > n_estimate:
            if row['outcome'] == majority_class:
                if random() > balance_ratio: continue 
        for x in row: row[x] = output_vars[x]['default_value'] if row[x] == None else row[x]
        row['ref_date'] = str(row['ref_date'])
        f.write(str(row).replace("'", '"') + '\n')
        c += 1
        if c % 10000 == 0: print('stored ' + str(c) + ' rows')
    f.close()

############
##  MAIN  ##
############

params = {
    'num_users': '3000',
    'min_date': '2018-01-01',
    'max_date': '2018-06-01',
    'valid_departments': "('Knit Tops','Woven Shirts','Dresses','Pants')",
    'lookback_window': '90' #30
}
print('\nstart time: ' + str(datetime.now()))
dbu = DBUtil("jjill_redshift","C:\Users\Terry\Desktop\KT_GitHub\databases\databases.conf")

numerical_vars, numerical_vars_fields = define_continuous_ref_level_vars(params)
categorical_vars, categorical_vars_fields = define_categorical_ref_level_vars()
other_binary_vars, other_binary_vars_fields = define_other_binary_ref_level_vars()
department_vars = define_department_level_vars(numerical_vars)
other_vars = define_other_vars(params)
output_vars = combine_vars(numerical_vars_fields, categorical_vars_fields, other_binary_vars_fields, other_vars)

queried_data, columns = run_data_query(output_vars, department_vars, params, dbu)
save_data(queried_data, columns, output_vars)

print('end time: ' + str(datetime.now()) + '\n')
