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
            f + '_past_' + x: {'definition': f + "(CASE WHEN t2." + x + " = '' THEN 0.0 ELSE t2." + x + "::float END)::float",
                               'default_value': 0.0},
            f + '_past_' + x + '_in_department': {'definition': f + """(CASE WHEN t1.department_name != t2.department_name THEN 0.0 
                                                                                  WHEN t2.""" + x + """ = '' THEN 0.0 
                                                                                  ELSE t2.""" + x + """::float END)::float""",
                                                  'default_value': 0.0}
        }
        for x in numerical_vars for f in ['sum', 'avg']
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
            'user_age': {'definition': "(max(t1.ref_date - t2.birth_date) / 365.0)::float",
                         'default_value': 0.0},
        }
    )
    return numerical_vars, numerical_vars_fields

# Defining Binary Variables On The Sample Reference Level
def define_binary_ref_level_vars():
    binary_vars = ['first_order_date', 'first_catalog_order_date', 'first_retail_order_date', 'first_web_order_date', 'prior_order_date']
    binary_vars_fields = [
        {
            'past_' + x + '_exists': {'definition': "max(CASE WHEN (t2." + x + " IS NOT NULL) AND (t2." + x + "::text != '') THEN 1.0 ELSE 0.0 END)::float",
                                      'default_value': 0.0},
            'past_' + x + '_exists_in_department': {'definition': """max(CASE WHEN t1.department_name != t2.department_name THEN 0.0
                                                                          WHEN (t2.""" + x + """ IS NOT NULL) AND (t2.""" + x + """::text != '') THEN 1.0 
                                                                          ELSE 0.0 END)::float""",
                                                    'default_value': 0.0}
        }
        for x in binary_vars
    ]
    return binary_vars, binary_vars_fields

# Defining All Variables On The Department Level
def define_department_level_vars(numerical_vars):
    department_vars = {}
    for x in numerical_vars:
        department_vars['department_avg_' + x] = 'avg(' + x + ')'
    department_categorical_vars = {
        'fabric_category_desc': ['Cotton/Cotton Bl', 'Synthetic/Syn Blend' ,'Linen/Linen Bl'],
        'pay_type_cd': ['VISA', 'MC', 'DISC', 'CASH', 'DEBIT', 'JJC', 'CK'],
        'end_use_desc': ['Core', 'Wearever', 'Pure Jill'],
        'price_cd': ['SP', 'FP'],
        'master_channel': ['D', 'R']
    }
    for x in department_categorical_vars:
        for y in department_categorical_vars[x]:
            val = "sum(CASE WHEN " + x + " IN ('" + y + "') THEN 1.0 ELSE 0.0 END) / count(ilink)::float"
            department_vars['department_' + x + '__' + y.split(' ')[0].split('/')[0].lower() + '_pct'] = val
    return department_vars

# Defining Any Other Variables Not Fitting In The Above Categories
def define_other_vars(params):
    other_vars = {
        'past_order_dates': {'definition': "listagg(distinct t2.order_date, ' ')",
                             'default_value': ''},
        'past_order_dates_in_department': {'definition': "listagg(distinct (CASE WHEN t1.department_name != t2.department_name THEN NULL ELSE t2.order_date END), ' ')",
                                           'default_value': ''}
    }
    return other_vars

# Combining All Variables
def combine_vars(numerical_vars_fields, binary_vars_fields, other_vars):
    output_vars = {}
    fields_list = [numerical_vars_fields, binary_vars_fields]
    for fields in fields_list:
        for i in range(0,len(fields)): 
            for y in fields[i]:
                output_vars[y] = fields[i][y]
    for x in other_vars: output_vars[x] = other_vars[x]
    return output_vars

# Adjusting Data For Query
def adjust_dates(dbu, params):
    dates = {}
    df = dbu.get_df_from_query("""
        SELECT min(order_date) AS min_date, max(order_date) AS max_date
        FROM jjill.jjill_keyed_data
        WHERE order_date::date >= '""" + params['min_date'] + """'
        AND order_date::date <= '""" + params['max_date'] + """'
    """)
    for i, row in df.iterrows(): date_extremas = { x:str(dict(row)[x]) for x in dict(row) }

    # Ensuring only samples from most current date possible created for production mode
    if params['mode'] == 'production':
        dates['min_data_date'] = str(datetime.strptime(date_extremas['max_date'], '%Y-%m-%d') - timedelta(days=int(params['lookback_window']))).split(' ')[0]
        dates['max_data_date'] = date_extremas['max_date']
        dates['min_reference_date'] = date_extremas['max_date']
        dates['max_reference_date'] = date_extremas['max_date']

    # Handling train and eval modes
    else:
        # Ensuring minimum (earliest) reference date has a full lookback window & maximum (latest) reference date has a full lookfront window
        dates['min_reference_date'] = str(datetime.strptime(date_extremas['min_date'], '%Y-%m-%d') + timedelta(days=int(params['lookback_window']))).split(' ')[0]
        dates['max_reference_date'] = str(datetime.strptime(date_extremas['max_date'], '%Y-%m-%d') - timedelta(days=int(params['lookfront_window']))).split(' ')[0]
        # Ensuring earliest date where data is collect at is at least a full lookback window away from the minimum reference date
        max_possible_min_data_date = str(datetime.strptime(dates['max_reference_date'], '%Y-%m-%d') - timedelta(days=int(params['lookback_window']))).split(' ')[0]
        dates['min_data_date'] = date_extremas['min_date'] if date_extremas['min_date'] < max_possible_min_data_date else max_possible_min_data_date
        dates['max_data_date'] = date_extremas['max_date']
        # Ensuring we are only creating samples for maximum reference date during eval mode
        if params['mode'] == 'eval':
            dates['min_data_date'] = max_possible_min_data_date
            dates['min_reference_date'] = dates['max_reference_date']

    pprint(dates)
    if dates['max_reference_date'] < dates['min_reference_date']: print('invalid date range'); sys.exit()
    return dates

# Associating Reference Fields With Data Types
def associating_ref_fields_with_data_types(dbu, columns):
    df = dbu.get_df_from_query("""
        SET search_path TO 'jjill';
        SELECT *
        FROM pg_table_def
        WHERE tablename = 'jjill_keyed_data'
    """)
    table_cols = {}
    for i, row in df.iterrows(): table_cols[row['column']] = row['type']
    column_types = {'ref_date':'date'}
    for col in columns:
        field = col.replace('department_avg_', '')
        if field in table_cols: column_types[col] = table_cols[field]
        if col[-4:] == '_pct': column_types[col] = 'double precision'
    return column_types

# Create Temporary Reference Table
def create_reference_table(dbu, dates, department_vars, params):
    print('\nrunning reference data queries...')
    user_limit = params['num_users'] if params['mode'] == 'train' else str(1000000)
    columns = ['ilink', 'department_name', 'ref_date'] + sorted([ x for x in department_vars ])
    column_types = associating_ref_fields_with_data_types(dbu, columns)
    dbu.update_db("""
        DROP TABLE IF EXISTS jjill.jjill_reference_data;
        CREATE TABLE jjill.jjill_reference_data
         (
             """ + ',\n\t\t'.join([ columns[i] + ' ' + column_types[columns[i]] for i in range(0,len(columns)) ]) + """
         ) distkey(ilink) sortkey(ref_date)
    """)
    dbu.update_db("""
        INSERT INTO jjill.jjill_reference_data (""" + ', '.join(columns) + """)

        WITH all_users AS (
            SELECT ilink
            FROM jjill.jjill_keyed_data
            WHERE department_name in """ + params['valid_departments'] + """
            AND is_emailable_ind='Y'
            AND order_date::date >= '""" + dates['min_data_date'] + """'
            AND order_date::date <= '""" + dates['max_reference_date'] + """'
            GROUP BY ilink
            ORDER BY ilink
            LIMIT """ + user_limit + """
        ),

        all_departments AS (
            SELECT
                (CASE WHEN department_name NOT IN """ + params['valid_departments'] + """ THEN 'Other_Dept' ELSE department_name END)
            FROM jjill.jjill_keyed_data
            WHERE order_date::date >= '""" + dates['min_data_date'] + """'
            AND order_date::date <= '""" + dates['max_reference_date'] + """'
            GROUP BY 1
        ),

        all_dates AS (
            SELECT order_date AS ref_date
            FROM jjill.jjill_keyed_data
            WHERE order_date::date >= '""" + dates['min_reference_date'] + """'
            AND order_date::date <= '""" + dates['max_reference_date'] + """'
            GROUP BY order_date
        ),

        all_department_date_pairs_current_view AS (
            SELECT 
                order_date::date AS order_date, 
                department_name,
                """ + ',\n\t\t'.join(sorted([ department_vars[x] + ' AS ' + x for x in department_vars ])) + """
            FROM jjill.jjill_keyed_data
            WHERE order_date::date >= '""" + dates['min_data_date'] + """'
            AND order_date::date <= '""" + dates['max_reference_date'] + """'
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

        SELECT """ + ', '.join(columns) + """
        FROM reference_data
        
    """)
    return

# Get Data Columns From SQL
def columns_from_sql(sql):
    columns = sql.split('SELECT')[1].split('FROM')[0].split(',')
    columns = [ columns[i].strip().lower().replace('t3.','') for i in range(0,len(columns)) ]
    columns = [ columns[i].split(' as ')[-1] for i in range(0,len(columns)) ]
    return columns

# Running SQL To Gather Training / Prediction Data
def run_data_query(output_vars, department_vars, params, dates, dbu):

    # Create Temporary Reference Table
    create_reference_table(dbu, dates, department_vars, params)

    # Main Query
    sql = """
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
            FROM jjill.jjill_reference_data AS t1
            LEFT JOIN jjill.jjill_keyed_data AS t2 ON (
                t1.ilink = t2.ilink 
                AND t1.ref_date > t2.order_date
                AND t1.ref_date::date - """ + params['lookback_window'] + """ < t2.order_date
            )
            WHERE (
                (t2.order_date >= '""" + dates['min_data_date'] + """' AND t2.order_date < '""" + dates['max_reference_date'] + """') 
                OR (t2.order_date IS NULL)
            )
            GROUP BY t1.ilink, t1.ref_date::date, t1.department_name
        ) AS t3
        LEFT JOIN jjill.jjill_keyed_data AS t4 ON (
            t3.ilink = t4.ilink 
            AND t3.department_name = t4.department_name
            AND t3.ref_date <= t4.order_date
            AND t3.ref_date + """ + str(params['lookfront_window']) + """ > t4.order_date
        )
        WHERE (
            (t4.order_date >= '""" + dates['min_reference_date'] + """' AND t4.order_date <= '""" + dates['max_data_date'] + """') 
            OR (t4.order_date is null)
        )
        GROUP BY t3.ilink, t3.ref_date, t3.department_name
        ORDER BY random()
    """

    print('running data query'); print('\n' + sql) #; sys.exit()
    queried_data = dbu.get_df_from_query(sql, server_cur=True, itersize=1000)
    columns = columns_from_sql(sql)
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
def save_data(queried_data, columns, output_vars, params):
    f = open(params['mode'] + '_data.txt','w')
    class_sizes = {0:0, 1:0}; c = 0; n_estimate = 20000
    for item in queried_data:
        row = dict(zip(columns, item))
        row = distinct_order_date_based_variables(row, output_vars)
        class_sizes[row['outcome']] += 1
        if c == n_estimate:
            majority_class = max(class_sizes, key=class_sizes.get)
            balance_ratio = min(class_sizes.values()) / float(max(class_sizes.values()))
            print('\tmajority_class: ' + str(majority_class))
            print('\tbalance_ratio: ' + str(balance_ratio))
        if params['mode'] == 'train':
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

print('\nstart time: ' + str(datetime.now()))
arg_options = ['train', 'eval', 'production']
error_msg = 'script requires 1 argument: {' + '|'.join(arg_options) + '}'
if len(sys.argv) != 2: print(error_msg); sys.exit()
if sys.argv[1] not in arg_options: print(error_msg); sys.exit()

dbu = DBUtil("jjill_redshift","C:\Users\Terry\Desktop\KT_GitHub\databases\databases.conf")
params = {
    'num_users': '3000', # only used during training
    'min_date': '2017-10-01',
    'max_date': str(datetime.now().date()), # '2018-02-10' for train; '2018-04-17' for eval; str(datetime.now().date()) for production 
    'valid_departments': "('Knit Tops', 'Woven Shirts', 'Dresses', 'Pants')",
    'lookback_window': '60',
    'lookfront_window': '30',
    'mode': sys.argv[1]
}
dates = adjust_dates(dbu, params)

numerical_vars, numerical_vars_fields = define_continuous_ref_level_vars(params)
binary_vars, binary_vars_fields = define_binary_ref_level_vars()
department_vars = define_department_level_vars(numerical_vars)
other_vars = define_other_vars(params)
output_vars = combine_vars(numerical_vars_fields, binary_vars_fields, other_vars)

queried_data, columns = run_data_query(output_vars, department_vars, params, dates, dbu)
save_data(queried_data, columns, output_vars, params)

print('end time: ' + str(datetime.now()) + '\n')
