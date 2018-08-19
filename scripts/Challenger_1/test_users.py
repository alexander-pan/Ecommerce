###############
##  IMPORTS  ##
###############

from datetime import datetime, timedelta
from random import random, shuffle
import matplotlib.pyplot as plt
from pprint import pprint
import pandas as pd
import numpy as np
import json
import math
import ast
import sys

#################
##  FUNCTIONS  ##
#################

# Loading Data From Input File
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

# Calculating Each User's Score Percentile Within Each Department
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

# Forming Testing & Control Groups For A-B Test
def form_test_groups(u):
    test_low_bound = 0.0; test_high_bound = 0.25; f_test = open('test_users_test_group.csv', 'w')
    ctrl_low_bound = 0.5; ctrl_high_bound = 1.00; f_ctrl = open('test_users_ctrl_group.csv', 'w')
    u_test = {}; c = 0
    for user in u:
        avgs = {}; lows = {}
        for department in u[user]:
            percentile = u[user][department]['percentile']
            if percentile >= ctrl_low_bound and percentile <= ctrl_high_bound: avgs[department] = 1
            if percentile >= test_low_bound and percentile <= test_high_bound: lows[department] = 1
            if len(avgs) == 3 and len(lows) == 1: 
                u_test[user] = u[user]
                c += 1
                if c % 2 == 0:
                    test_department = [ x for x in lows ][0]
                    f_test.write(','.join([ user, test_department, str(u_test[user][test_department]['score']) ]) + '\n')
                else:
                    ctrl_department = [ x for x in avgs ]
                    shuffle(ctrl_department)
                    ctrl_department = ctrl_department[0]
                    f_ctrl.write(','.join([ user, ctrl_department, str(u_test[user][ctrl_department]['score']) ]) + '\n')
    print('\nnumber of original users: ' + str(len(u)))
    print('number of valid A-B testing users: ' + str(len(u_test)))
    f_test.close()
    f_ctrl.close()

############
##  MAIN  ##
############

# Loading Data From Input File
d = load_data()

# Calculating Each User's Score Percentile Within Each Department
u, d = determine_department_percentiles(d)

# Forming Testing & Control Groups For A-B Test
u_test = form_test_groups(u)

