###############
##  IMPORTS  ##
###############

from datetime import datetime, timedelta
from random import random, shuffle
from pandasql import sqldf
from pprint import pprint
import pandas as pd
import numpy as np
import shutil
import math
import json
import ast
import sys
import os

sys.path.append('../../utils')
#from db_utils import DBUtil
pysqldf = lambda q: sqldf(q, globals())

#################
##  FUNCTIONS  ##
#################

# Defines Whether Script Is Running In Train Or Production Mode
def handle_initial_arguments(args):
    print('\nstart time: ' + str(datetime.now()))
    arg_options = ['train', 'eval', 'production']
    error_msg = 'script requires 1 argument: {' + '|'.join(arg_options) + '}'
    if len(args) != 2: print(error_msg); sys.exit()
    if args[1] not in arg_options: print(error_msg); sys.exit()
    return args[1]

# Defining Continuous Variables On The Sample Reference Level
def define_continuous_ref_level_vars(params):
    max_date_diff = str((datetime.strptime(params['max_date'], '%Y-%m-%d') - datetime.strptime(params['min_date'], '%Y-%m-%d')).days)
    numerical_vars = [
        'original_retail_price_amt', 
        'shipped_cost_amt', 'shipped_sold_amt', 'shipped_qty',
        'returned_cost_amt', 'returned_sold_amt', 'returned_qty',
        'ordered_cost_amt', 'ordered_sold_amt', 'ordered_qty'
    ]
    numerical_vars_fields = [
        {
            f + '_past_' + x: {'definition': f + "(CAST((CASE WHEN t2." + x + " = '' THEN 0.0 ELSE t2." + x + " END) AS float))",
                               'default_value': 0.0},
            f + '_past_' + x + '_in_department': {'definition': f + """(CAST((CASE WHEN t1.department_name != t2.department_name THEN 0.0 
                                                                                  WHEN t2.""" + x + """ = '' THEN 0.0 
                                                                                  ELSE t2.""" + x + """ END) AS float))""",
                                                  'default_value': 0.0}
        }
        for x in numerical_vars for f in ['sum', 'avg']
    ]
    numerical_vars_fields.append(
        {
            'most_recent_past_order_days_ago': {'definition': 'CAST(min(t1.ref_date - t2.order_date) AS float)',
                                                'default_value': max_date_diff},
            'most_recent_past_order_days_ago_in_department': {'definition': """CAST(min(CASE WHEN t1.department_name != t2.department_name THEN """ + max_date_diff + """
                                                                                        ELSE t1.ref_date - t2.order_date END) AS float)""",
                                                              'default_value': max_date_diff},
            'num_past_orders': {'definition': 'CAST(count(t2.order_date) AS float)',
                                'default_value': 0.0},
            'num_past_orders_in_department': {'definition': 'CAST(sum(CASE WHEN t1.department_name != t2.department_name THEN 0 ELSE 1 END) AS float)',
                                              'default_value': 0.0},
            'user_age': {'definition': "CAST((max(t1.ref_date - t2.birth_date) / 365.0) AS float)",
                         'default_value': 0.0},
        }
    )
    return numerical_vars, numerical_vars_fields

# Defining Binary Variables On The Sample Reference Level
def define_binary_ref_level_vars():
    binary_vars = ['first_order_date', 'first_catalog_order_date', 'first_retail_order_date', 'first_web_order_date', 'prior_order_date']
    binary_vars_fields = [
        {
            'past_' + x + '_exists': {'definition': "CAST(max(CASE WHEN (t2." + x + " IS NOT NULL) AND (CAST(t2." + x + " AS text) != '') THEN 1.0 ELSE 0.0 END) AS float)",
                                      'default_value': 0.0},
            'past_' + x + '_exists_in_department': {'definition': """CAST(max(CASE WHEN t1.department_name != t2.department_name THEN 0.0
                                                                          WHEN (t2.""" + x + """ IS NOT NULL) AND (CAST(t2.""" + x + """ AS text) != '') THEN 1.0 
                                                                          ELSE 0.0 END) AS float)""",
                                                    'default_value': 0.0}
        }
        for x in binary_vars
    ]
    return binary_vars, binary_vars_fields

# Defining All Variables On The Department Level
def define_department_level_vars(numerical_vars):
    department_vars_fields = {}
    # for x in numerical_vars: department_vars_fields['department_avg_' + x] = 'avg(' + x + ')'
    department_categorical_vars = {
        'department_name': ['Knit Tops', 'Woven Shirts', 'Dresses', 'Pants']
    }
    for x in department_categorical_vars:
        for y in department_categorical_vars[x]:
            val = "sum(CASE WHEN " + x + " IN ('" + y + "') THEN 1.0 ELSE 0.0 END) / CAST(count(ilink) AS float)"
            department_vars_fields['department_' + x + '__' + y.split(' ')[0].split('/')[0].lower() + '_pct'] = val
    return department_categorical_vars, department_vars_fields

# Defining Any Other Variables Not Fitting In The Above Categories
def define_other_vars(params):
    other_vars_fields = {
        'past_order_dates': {'definition': "listagg(DISTINCT t2.order_date)",
                             'default_value': ''},
        'past_order_dates_in_department': {'definition': """listagg(DISTINCT (CASE 
                                                                                  WHEN t1.department_name != t2.department_name THEN NULL 
                                                                                  ELSE t2.order_date 
                                                                              END))""",
                                           'default_value': ''}
    }
    return other_vars_fields

# Combining All Variables Into Single Dictionary For SQL Query
def combine_vars(numerical_vars_fields, binary_vars_fields, other_vars_fields, numerical_vars, binary_vars, department_categorical_vars):
    output_vars = {}
    fields_list = [numerical_vars_fields, binary_vars_fields]
    for fields in fields_list:
        for i in range(0,len(fields)): 
            for y in fields[i]:
                output_vars[y] = fields[i][y]
    for x in other_vars_fields: output_vars[x] = other_vars_fields[x]
    output_base_vars = numerical_vars + binary_vars + [ x for x in department_categorical_vars ] 
    output_base_vars = output_base_vars + ['ilink', 'order_date', 'department_name', 'is_emailable_ind', 'birth_date']
    output_base_vars = { x:1 for x in output_base_vars }
    return output_vars, output_base_vars

# Specifying How To Handle The Header Row In Input Flat File
def handle_input_data_file_header(row, fs, header, m, output_base_vars):
    header = { i:row[i].lower() for i in range(0,len(row)) if row[i].lower() in output_base_vars }
    header_fields = sorted([header[key] for key in header], reverse=False)
    if m == 1: 
        for key in fs: fs[key].write(str(header_fields) + '\n')
    pprint(header)
    # pprint(sorted(row)); sys.exit()
    return header, header_fields

# Specifying How To Handle Each Non-Header Row In Input Flat File
def handle_input_data_file_row(row, fs, header, header_fields, m, cs, users, params, max_date):
    row = { header[key]:row[key] for key in header }
    if len(row) != len(header_fields): return cs, max_date, True
    if row['ilink'] == '': return cs, max_date, True
    if (row['order_date'] >= params['min_date']) and (row['order_date'] <= params['max_date']):
        if m == 1: 
            if row['ilink'] not in users: return cs, max_date, True
            fs[users[row['ilink']]].write(str([ row[header_fields[i]] for i in range(0,len(header_fields)) ]) + '\n')
            cs += 1
        if m == 0: 
            if len(users) < 2000000: users[row['ilink']] = 1
            max_date = max(row['order_date'], max_date)
            cs += 1
    return cs, max_date, False

# After First Flat File Scan-Through, Defining User & Date-Range Limits
def handle_pre_scanned_file_info(users, params, f_name_sampled, f_name_sampled_ilinks, max_date):
    print('number of users seen: ' + str(len(users)))
    users = [ x for x in users ]
    shuffle(users)
    users = users[0:int(params['num_users'])]
    users = { x:1 for x in users }
    users = [ x for x in users ]
    shuffle(users)
    batch_size = 100000
    num_user_files = int(math.ceil(len(users) / float(batch_size)))
    users_new = {}
    for i in range(0,num_user_files):
        users_batch = users[i*batch_size:(i+1)*batch_size]
        for user in users_batch: users_new[user] = str(i)
        fsu = open(f_name_sampled_ilinks + str(i) + '.csv', 'w')
        fsu.write(str(users_batch))
        fsu.close()
    users = users_new
    print('number of users stored: ' + str(len(users)))
    params['num_user_files'] = num_user_files
    params['data_flat_file_path'] = f_name_sampled
    if params['mode'] == 'production': 
        max_date_ts = datetime.strptime(max_date, '%Y-%m-%d')
        params['max_date'] = max_date
        params['min_date'] = str(max_date_ts - timedelta(days=int(params['lookback_window']))).split(' ')[0]
    return users, params

# Perparing Input Data Files From Flat File If Necessary
def prepare_input_files(params, output_base_vars):
    f_name = params['data_flat_file_path']
    f_name_sampled = f_name.split('.csv')[0] + '_sampled_'
    f_name_sampled_ilinks = f_name_sampled + 'ilinks_'
    users = {}; max_date = '2000-01-01'
    # Iterating Over Flat File Twice; Once For Determining User & Date-Range Constraints And Once For Data Collection
    for m in range(0,2):
        print('\nstarting loop ' + str('pre-scan' if m == 0 else 'store input data')); pprint(params)
        f = open(f_name, 'r')
        fs = {}
        if m == 1:
            for i in range(0,params['num_user_files']): fs[str(i)] = open(f_name_sampled + str(i) + '.csv', 'w')
        header = 0; c = 0; cs = 0
        for line in f:
            row = line.replace('\n', '').split(params['data_flat_file_delimiter'])
            # Specifying How To Handle The Header Row In Input Flat File
            if header == 0: header, header_fields = handle_input_data_file_header(row, fs, header, m, output_base_vars)
            else: 
                # Specifying How To Handle Each Non-Header Row In Input Flat File
                cs, max_date, continue_bool = handle_input_data_file_row(row, fs, header, header_fields, m, cs, users, params, max_date)
                if continue_bool == True: continue
            c += 1
            if c % 1000000 == 0: print('rows iterated over: ' + str(c) + '; rows stored: ' + str(cs))
        print('rows iterated over: ' + str(c) + '; rows stored: ' + str(cs))
        f.close()
        for key in fs: fs[key].close()
        # After First Flat File Scan-Through, Defining User & Date-Range Limits
        if m == 0: users, params = handle_pre_scanned_file_info(users, params, f_name_sampled, f_name_sampled_ilinks, max_date)
    return params

# Converting File Of Sampled Data To Dataframe
def flat_file_to_dataframe(file_name):
    df = []; print('\ncreating source data df')
    f = open(file_name, 'r'); c = 0
    for line in f:
        row = json.loads(line.replace("'", '"'))
        if c == 0: header = row
        else:
            if len(row) != len(header): continue
            row = { header[i]:row[i] for i in range(0,len(header)) }
            df.append(row)
            if c == 1000000: print('rows iterated over: ' + str(c))
        c += 1
    df = pd.DataFrame(df)
    return df

# Adjusting Date-Ranges For Query
def adjust_dates(dbu, params):
    dates = {}
    sql = """
        SELECT min(order_date) AS min_date, max(order_date) AS max_date
        FROM jjill.jjill_keyed_data
        WHERE order_date >= '""" + params['min_date'] + """'
        AND order_date <= '""" + params['max_date'] + """'
    """
    if params['data_format'] == 'sql': df = dbu.get_df_from_query(sql)
    if params['data_format'] == 'flat': df = pysqldf(sql.replace('jjill.jjill_keyed_data', 'dbu'))
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

# If Gathering Data From Database, Associating Reference Fields With Data Types
def associating_ref_fields_with_data_types(dbu, columns):
    sql = """
        SET search_path TO 'jjill';
        SELECT *
        FROM pg_table_def
        WHERE tablename = 'jjill_keyed_data'
    """
    df = dbu.get_df_from_query(sql)
    table_cols = {}
    for i, row in df.iterrows(): table_cols[row['column']] = row['type']
    column_types = {'ref_date':'date'}
    for col in columns:
        field = col.replace('department_avg_', '')
        if field in table_cols: column_types[col] = table_cols[field]
        if col[-4:] == '_pct': column_types[col] = 'double precision'
    return column_types

# Creating Temporary Reference Table (Contains Reference Data Rows)
def create_reference_table(dbu, dates, department_vars_fields, params):
    print('\nrunning reference data queries...')
    columns = ['ilink', 'department_name', 'ref_date'] + sorted([ x for x in department_vars_fields ])
    date_range = []; cur_date = datetime.strptime(dates['max_reference_date'], '%Y-%m-%d')
    while cur_date >= datetime.strptime(dates['min_reference_date'], '%Y-%m-%d'):
        date_range.append("'" + str(cur_date).split(' ')[0] + "'")
        cur_date = cur_date - timedelta(days=1)
    date_range = sorted(date_range, reverse=False)
    reference_sql = """
        WITH all_users AS (
            SELECT ilink
            FROM jjill.jjill_keyed_data
            WHERE department_name IN """ + params['valid_departments'] + """
            AND is_emailable_ind = 'Y'
            AND order_date >= '""" + dates['min_data_date'] + """'
            AND order_date <= '""" + dates['max_reference_date'] + """'
            GROUP BY ilink
            ORDER BY RANDOM()
            LIMIT """ + params['num_users'] + """
        ),

        all_departments AS (
            SELECT
                (CASE WHEN department_name NOT IN """ + params['valid_departments'] + """ THEN 'Other_Dept' ELSE department_name END) AS department_name
            FROM jjill.jjill_keyed_data
            WHERE order_date >= '""" + dates['min_data_date'] + """'
            AND order_date <= '""" + dates['max_reference_date'] + """'
            GROUP BY 1
        ),

        all_dates AS (
            SELECT CAST(ref_date AS text) AS ref_date
            FROM
            (
                SELECT """ + ' AS ref_date \n UNION ALL SELECT '.join(date_range) + """ AS ref_date
            )
        ),

        all_department_date_pairs_current_view AS (
            SELECT 
                order_date AS order_date, 
                department_name,
                """ + ',\n\t\t'.join(sorted([ department_vars_fields[x] + ' AS ' + x for x in department_vars_fields ])) + """
            FROM jjill.jjill_keyed_data
            WHERE order_date >= '""" + dates['min_data_date'] + """'
            AND order_date <= '""" + dates['max_reference_date'] + """'
            GROUP BY order_date, department_name
        ),

        all_user_department_date_tuples AS (
            SELECT distinct ilink, department_name, ref_date
            FROM all_users
            JOIN all_departments on (1 = 1)
            JOIN all_dates on (1 = 1)
        ),

        reference_data_pre AS (
            SELECT 
                ta.ilink, ta.department_name, ta.ref_date, tb.order_date,
                """ + ',\n\t\t'.join(sorted([ 'tb.' + x for x in department_vars_fields ])) + """
            FROM all_user_department_date_tuples AS ta
            JOIN all_department_date_pairs_current_view AS tb ON 
            (
                ta.department_name = tb.department_name
                AND ta.ref_date > tb.order_date
            )
        ),

        reference_data AS (
            SELECT t1.*
            FROM reference_data_pre AS t1
            JOIN 
            (
                SELECT 
                    ilink, department_name, ref_date,
                    max(order_date) AS max_order_date
                FROM reference_data_pre
                GROUP BY ilink, department_name, ref_date
            ) AS t2 ON 
            (
                t1.ilink = t2.ilink
                AND t1.department_name = t2.department_name
                AND t1.ref_date = t2.ref_date
                AND t1.order_date = t2.max_order_date
            )
        )

        SELECT """ + ', '.join(columns) + """
        FROM reference_data
    """
    if params['data_format'] == 'sql':
        column_types = associating_ref_fields_with_data_types(dbu, columns)
        dbu.update_db("""
            DROP TABLE IF EXISTS jjill.jjill_reference_data;
            CREATE TABLE jjill.jjill_reference_data
             (
                 """ + ',\n\t\t'.join([ columns[i] + ' ' + column_types[columns[i]] for i in range(0,len(columns)) ]) + """
             ) distkey(ilink) sortkey(ref_date)
        """)
        dbu.update_db("INSERT INTO jjill.jjill_reference_data (" + ', '.join(columns) + ")\n" + reference_sql)
        ref_df = ''
    if params['data_format'] == 'flat': 
        ref_df = pysqldf(reference_sql.replace('jjill.jjill_keyed_data', 'dbu'))
        print('reference data shape: ' + str(ref_df.shape))
    return ref_df

# Running SQL / PandaSQL To Gather Train / Production Data
def run_data_query(output_vars, department_vars_fields, params, dates, dbu, ref_df):
    sql = """
        SELECT
            t3.ilink, t3.ref_date, t3.department_name,
            max(CASE WHEN t4.order_date IS NULL THEN 0 ELSE 1 END) AS outcome,
            """ + ',\n\t\t'.join(sorted([ 'max(t3.' + x + ') AS ' + x for x in department_vars_fields ])) + """,
            """ + ',\n\t\t'.join(sorted([ 'max(t3.' + x + ') AS ' + x for x in output_vars ])) + """
        FROM
        (
            SELECT 
                t1.ilink, t1.ref_date, t1.department_name,
                """ + ',\n\t\t'.join(sorted([ 'max(t1.' + x + ') AS ' + x for x in department_vars_fields ])) + """,
                """ + ',\n\t\t'.join([ output_vars[x]['definition'] + ' AS ' + x for x in output_vars ]) + """
            FROM jjill.jjill_reference_data AS t1
            LEFT JOIN jjill.jjill_keyed_data AS t2 ON (
                t1.ilink = t2.ilink 
                AND t1.ref_date > t2.order_date
                AND date(t1.ref_date, '-""" + params['lookback_window'] + """ day') < t2.order_date                
            )
            WHERE (
                (t2.order_date >= '""" + dates['min_data_date'] + """' AND t2.order_date < '""" + dates['max_reference_date'] + """') 
                OR (t2.order_date IS NULL)
            )
            GROUP BY t1.ilink, t1.ref_date, t1.department_name
        ) AS t3
        LEFT JOIN jjill.jjill_keyed_data AS t4 ON (
            t3.ilink = t4.ilink 
            AND t3.department_name = t4.department_name
            AND t3.ref_date <= t4.order_date
            AND date(t3.ref_date, '+""" + str(params['lookfront_window']) + """ day') > t4.order_date
        )
        WHERE (
            (t4.order_date >= '""" + dates['min_reference_date'] + """' AND t4.order_date <= '""" + dates['max_data_date'] + """') 
            OR (t4.order_date is null)
        )
        GROUP BY t3.ilink, t3.ref_date, t3.department_name
        ORDER BY random()
    """

    print('running data query'); #print('\n' + sql) ; sys.exit()
    if params['data_format'] == 'sql': queried_data = dbu.get_df_from_query(sql, server_cur=True, itersize=1000)
    if params['data_format'] == 'flat': 
        sql = sql.replace('jjill.jjill_keyed_data', 'dbu').replace('jjill.jjill_reference_data', 'ref_df')
        sql = sql.replace('listagg', 'group_concat')
        queried_data = pysqldf(sql)
        print('queried data shape: ' + str(queried_data.shape))
    columns = sql.split('SELECT')[1].split('FROM')[0].split(',')
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
            past_order_dates = past_order_dates.split(',')
            if len(past_order_dates) > 1:
                past_order_dates = sorted(past_order_dates, reverse=True)
                past_order_date_diffs = [ float((datetime.strptime(past_order_dates[i], '%Y-%m-%d') - datetime.strptime(past_order_dates[i+1], '%Y-%m-%d')).days) 
                                          for i in range(0,len(past_order_dates)-1) ]
                row['avg_days_between_orders' + suffix] = np.mean(past_order_date_diffs)
        del row['past_order_dates' + suffix]
    return row

# Specifying How Output Data Should Be Formatted
def format_output(row, f, class_sizes, c, n_estimate, output_vars, params):
    row = distinct_order_date_based_variables(row, output_vars)
    class_sizes[row['outcome']] += 1
    if c == n_estimate:
        cls_sz = {0:class_sizes[0], 1:class_sizes[1]}
        class_sizes['majority_class'] = max(cls_sz, key=cls_sz.get)
        class_sizes['balance_ratio'] = min(cls_sz.values()) / float(max(cls_sz.values()))
        print('\tmajority_class: ' + str(class_sizes['majority_class']))
        print('\tbalance_ratio: ' + str(class_sizes['balance_ratio']))
    if params['mode'] == 'train':
        if c > n_estimate:
            if row['outcome'] == class_sizes['majority_class']:
                if random() > class_sizes['balance_ratio']: return c, class_sizes, True
    for x in row: row[x] = output_vars[x]['default_value'] if (row[x] == None) or (str(row[x]) == 'nan') else row[x]
    row['ref_date'] = str(row['ref_date'])
    f.write(str(row).replace("'", '"') + '\n')
    c += 1
    if c % 10000 == 0: print('stored ' + str(c) + ' rows')
    return c, class_sizes, False

# Outputting Training / Production Data To A File
def save_data(queried_data, f_out, columns, output_vars, params):
    class_sizes = {0:0, 1:0, 'majority_class':'', 'balance_ratio':''}; 
    c = 0; n_estimate = 20000
    if params['data_format'] == 'sql':
        for item in queried_data:
            row = dict(zip(columns, item))
            c, class_sizes, continue_bool = format_output(row, f_out, class_sizes, c, n_estimate, output_vars, params)
            if continue_bool == True: continue
    if params['data_format'] == 'flat': 
        for i, row in queried_data.iterrows():
            row = dict(row)
            c, class_sizes, continue_bool = format_output(row, f_out, class_sizes, c, n_estimate, output_vars, params)
            if continue_bool == True: continue
    print('stored ' + str(c) + ' rows')   

############
##  MAIN  ##
############

# Defines Whether Script Is Running In Train Or Production Mode
mode = handle_initial_arguments(sys.argv)

# Specifying Parameters Of Script
if mode == 'train':
    params = {
        'data_format': 'flat', # 'sql' once gathering from database
        'data_flat_file_path': 'master26_2017_2018.csv',
        'data_flat_file_delimiter': ',',
        'num_users': '3000',
        'min_date': '2018-03-01',
        'max_date': '2018-07-01', 
        'valid_departments': "('Knit Tops', 'Woven Shirts', 'Dresses', 'Pants')",
        'lookback_window': '60',
        'lookfront_window': '30',
        'mode': mode
    }
if mode == 'production':
    params = {
        'data_format': 'flat', # 'sql' once gathering from database
        'data_flat_file_path': 'master26_2017_2018.csv',
        'data_flat_file_delimiter': ',',
        'num_users': '1000000',
        'min_date': '2018-06-01',
        'max_date': str(datetime.now().date()),
        'valid_departments': "('Knit Tops', 'Woven Shirts', 'Dresses', 'Pants')",
        'lookback_window': '60',
        'lookfront_window': '30',
        'mode': mode
    }

# Defining Continuous Variables On The Sample Reference Level
numerical_vars, numerical_vars_fields = define_continuous_ref_level_vars(params)

# Defining Binary Variables On The Sample Reference Level
binary_vars, binary_vars_fields = define_binary_ref_level_vars()

# Defining All Variables On The Department Level
department_categorical_vars, department_vars_fields = define_department_level_vars(numerical_vars)

# Defining Any Other Variables Not Fitting In The Above Categories
other_vars_fields = define_other_vars(params)

# Combining All Variables Into Single Dictionary For SQL Query
output_vars, output_base_vars = combine_vars(numerical_vars_fields, binary_vars_fields, other_vars_fields, numerical_vars, binary_vars, department_categorical_vars)

# Perparing Input Data Files From Flat File If Necessary (Multiple Input Files Created For Efficient Memory Handling)
if params['data_format'] == 'sql': params['num_user_files'] = 1
if params['data_format'] == 'flat': params = prepare_input_files(params, output_base_vars)

# Defining Output File
f_out = open(params['mode'] + '_data.txt','w')

# Iterating Over All Created Input Files
for i in range(0,params['num_user_files']):
    print('\nprocessing file set: ' + str(i))

    # If Flat File Original Input, Converting Input File To Dataframe
    if params['data_format'] == 'sql': dbu = DBUtil("jjill_redshift", "C:\\Users\\Terry\\Desktop\\KT_GitHub\\databases\\databases.conf")
    if params['data_format'] == 'flat': dbu = flat_file_to_dataframe(params['data_flat_file_path'] + str(i) + '.csv')

    # Adjusting Date-Ranges For Query
    dates = adjust_dates(dbu, params)

    # Creating Temporary Reference Table (Contains Reference Data Rows)
    ref_df = create_reference_table(dbu, dates, department_vars_fields, params)

    # Running SQL / PandaSQL To Gather Train / Production Data
    queried_data, columns = run_data_query(output_vars, department_vars_fields, params, dates, dbu, ref_df)

    # Outputting Training / Production Data To A File
    save_data(queried_data, f_out, columns, output_vars, params)

    # Clearing Stored Variable For Better Memory Management
    dbu = ''; ref_df = ''; dbu = pysqldf("SELECT 1"); ref_df = pysqldf("SELECT 1"); queried_data = ''

# Ending Script    
f_out.close()
print('end time: ' + str(datetime.now()) + '\n')
