###############
##  IMPORTS  ##
###############

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from random import random
from pprint import pprint
import pandas as pd
import numpy as np
import ujson
import math
import dill
import sys

sys.path.append('../../utils')
from db_utils import DBUtil

#################
##  FUNCTIONS  ##
#################

# Loading Data From Text File
def load_data():
    df = []
    f = open('train_data.txt', 'r')
    for line in f:
        row = ujson.loads(line.replace('\n', ''))
        df.append(row)
    f.close()
    df = pd.DataFrame(df)
    return df

# Randomizing Data
def randomize_df(df):
    df = df.sample(frac=1).reset_index(drop=True)
    return df

# Splitting Data Into Train & Eval Sets
def split_to_train_eval(df, train_fraction=0.7):
    df_len = df.shape[0]
    split_index = int(math.floor(df_len * train_fraction))
    split_val = sorted(df['ilink'].tolist(), reverse=False)[split_index]
    df_train = []; df_eval = []
    for i, row in df.iterrows():
        #if random() < train_fraction: df_train.append(row)
        if row['ilink'] < split_val: df_train.append(row)
        else: df_eval.append(row)
    df_train = pd.DataFrame(df_train)
    df_eval = pd.DataFrame(df_eval)
    return df_train, df_eval

# Balancing Classes In Dataset
def class_balance_df(df):
    min_class_size = int(min(list(df.groupby(['outcome']).size())))
    df_0 = df[df['outcome'] == 0].sample(n=min_class_size).reset_index(drop=True)
    df_1 = df[df['outcome'] == 1].sample(n=min_class_size).reset_index(drop=True)
    df = pd.concat([df_0, df_1], axis=0)
    return df

# Splitting Dataset Into Inputs, Outputs, & Context
def split_to_inputs_outputs(df):
    context = ['ilink', 'ref_date', 'department_name']
    outcome = ['outcome']
    df_x = df.drop(outcome + context, axis=1)
    df_y = df[outcome]
    df_context = df[context]
    return df_x, df_y, df_context

# Running Classifier
def run_classifier(df_train_x, df_train_y):
    print('running classifier...')
    print('train sample size: ' + str(len(df_train_y)))
    #clf = RandomForestClassifier(n_estimators=500, random_state=123)
    clf = ExtraTreesClassifier(n_estimators=1000, random_state=123)
    clf.fit(df_train_x, df_train_y)
    columns = list(df_train_x.columns)
    importances = clf.feature_importances_
    importances = { str(columns[i]):importances[i] for i in range(0,len(importances)) }
    importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    print('\nimportances:' ); pprint(importances)
    return clf

# Evaluate Classifier Performance
def evaluate_model(clf, df_eval_x, df_eval_y):
    accuracy = clf.score(df_eval_x, df_eval_y)
    class_sizes = list(df_eval_y.groupby(['outcome']).size())
    print('\neval sample size: ' + str(len(df_eval_y)))
    print('accuracy: ' + str(accuracy))
    print('accuracy if random guess: ' + str(min(class_sizes) / float(sum(class_sizes))))
    return 

############
##  MAIN  ##
############

print('\nstart time: ' + str(datetime.now()))
dbu = DBUtil("jjill_redshift","C:\Users\Terry\Desktop\KT_GitHub\databases\databases.conf")

df = load_data()
df = randomize_df(df)
df_train, df_eval = split_to_train_eval(df)
df_train = class_balance_df(df_train)
df_eval = class_balance_df(df_eval)
df_train_x, df_train_y, df_train_context = split_to_inputs_outputs(df_train)
df_eval_x, df_eval_y, df_eval_context = split_to_inputs_outputs(df_eval)
clf = run_classifier(df_train_x, df_train_y)
evaluate_model(clf, df_eval_x, df_eval_y)

print('end time: ' + str(datetime.now()) + '\n')
