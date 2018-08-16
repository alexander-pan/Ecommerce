# Imports
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from random import random
from pprint import pprint
import pandas as pd
import numpy as np
import json
import math
import ast
import sys

# Functions
def load_data():
    f = open('production_predictions.txt', 'r')
    d = {}; c = 0
    for line in f:
        ilink, department, score = line.replace('\n', '').split(',')
        if department not in d: d[department] = {}
        d[department][ilink] = score
        c += 1
        if c % 100000 == 0: print('processed rows: ' + str(c))
    f.close()
    return d

def determine_department_percentiles(d):
    u = {}
    for department in d:
        d[department] = sorted(d[department].items(), key=lambda x: x[1], reverse=True)
        num_users = float(len(d[department]))
        d[department] = { d[department][i][0]:{
                            'score': d[department][i][1], 
                            'rank': i, 
                            'percentile': (num_users-i) / num_users} 
                            for i in range(0,len(d[department])) 
                        }
        for user in d[department]:
            if user not in u: u[user] = {}
            u[user][department] = d[department][user]
    d = {}
    return u, d

def form_test_groups(u):
    u_test = {}
    for user in u:
        c_avg = 0; c_low = 0
        for department in u[user]:
            percentile = u[user][department]['percentile']
            if percentile >= 0.5 and percentile <= 1.00: c_avg += 1
            if percentile <= 0.2: c_low += 1
            if c_avg == 3 and c_low == 1: u_test[user] = u[user]
    return u_test

# Main
d = load_data()
u, d = determine_department_percentiles(d)
u_test = form_test_groups(u)

print('\n' + str(len(u_test))); c = 0
for user in u_test:
    print('\n' + user)
    pprint(u_test[user])
    c += 1
    if c == 3: break

