import psycopg2
import psycopg2.extras 
import configparser
from psycopg2.pool import SimpleConnectionPool
import numpy as np
import pandas as pd 

class DBUtil(): 
    def __init__(self, db_name, config_file): 
        self.db_name = db_name
        self.config_file = config_file 

    def connect_to_db(self): 
        db_name = self.db_name 
        cp = configparser.ConfigParser()
        cp.read(self.config_file)
        password = cp.get(db_name, "password")
        user = cp.get(db_name, "user")
        database = cp.get(db_name, "database")
        host = cp.get(db_name, "host") 
        port = cp.get(db_name, "port") 


        kwargs = {"host":host,"password":password, 
            "user":user,"dbname":database, "port":port}

        self.conn_pool = SimpleConnectionPool(1, 3, **kwargs)
        
    def get_conn(self): 
        try: 
            conn = self.conn_pool.getconn() 
        except: 
            self.connect_to_db()
            conn = self.conn_pool.getconn()   
        return conn 
    
    def get_df_from_query(self, query, params=None, pprint=False, to_df=True, server_cur=False, itersize=20000):
        try:
            conn = self.conn_pool.getconn()
        except:
            self.connect_to_db()
            conn = self.conn_pool.getconn()
        
        if pprint==True:
            print(self.format_sql(query))

        if server_cur == True:
            cur = conn.cursor('server_side_cursor')
            cur.itersize = itersize
            cur.execute(query, params)            
            return cur
        else:
            with conn.cursor() as cur:
                cur.execute(query, params)
                data = cur.fetchall()
                columns = [desc[0] for desc in cur.description]
            
        self.conn_pool.putconn(conn)
        
        if to_df == True: 
            df = pd.DataFrame(data, columns=columns)
            return df
        else:
            return data, columns 
    
    def get_arr_from_query(self, query, params=None): 
        results_arr = [] 
        conn = self.get_conn()       
            
        with conn.cursor() as cur: 
            cur.execute(query, params)

            data = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            results_arr.append(columns)

        self.conn_pool.putconn(conn)
        for row in data: 
            results_arr.append(list(row)) 
        return results_arr

    def update_db(self, query, params=None): 

        conn = self.get_conn()
        
        with conn.cursor() as cur: 
            try: 
                cur.execute(query, params)
            except Exception as e: 
                print(e)
                self.conn_pool.putconn(conn)
                raise e

        conn.commit()
        self.conn_pool.putconn(conn)
        return 0  
    
    def write_df_to_table(self, df, tablename): 
        
        arr = [] 
        columns = df.columns 
        for index, row in df.iterrows(): 
            row = [ str(i)[:255] for i in row.tolist()]
            arr.append(row) 
        self.write_arr_to_table(arr, tablename, columns)
    
    def write_arr_to_table(self, arr, tablename, columns, new_table=True): 
        
        conn = self.get_conn()       

        column_str = "({0})".format( ",".join(columns))
        column_def = "({0} varchar(256) )".format( " varchar(256),".join(columns))
        value_str = "({0})".format( ",".join(["%s" for c in columns]))
        
        sql = "insert into {0} {1} values {2};".format(tablename, column_str, value_str)
        
        try: 
            print(sql, arr[0])
        except IndexError as e: 
            print(e, len(arr))
         
        with conn.cursor() as cur: 
            if new_table==True: 
                cur.execute("DROP TABLE IF EXISTS {0}".format(tablename))
                cur.execute("CREATE TABLE {0} {1}".format(tablename, column_def))
            try: 
                for row in arr: 
                    cur.execute(sql, row )

            except Exception as e: 
                print(e)
                self.conn_pool.putconn(conn)
                raise e 
                

        conn.commit()
        self.conn_pool.putconn(conn)
        return 0 
        
def sort_features(unsorted_list): 
    sorted_list = [] 

    for entry in unsorted_list: 
        # base case, just starting out
        # entry = (abs(entry[0]), entry[1], entry[0] > 0 )
        if len(sorted_list) == 0: 
            sorted_list.append(entry)
        # scan sorted list and insert 
        else: 
            # check each end 
            if entry < sorted_list[0]: 
                sorted_list.insert(0, entry)
                continue 
            if entry > sorted_list[len(sorted_list)-1]: 
                sorted_list.append(entry)
                continue 
            # scan whole list 
            si = 0
            for this in sorted_list[:len(sorted_list)-2]: 
                si +=1 
                that = sorted_list[si]
                if this < entry and entry < that: 
                    sorted_list.insert(si, entry)
            
    return sorted_list