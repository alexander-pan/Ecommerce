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
import pickle
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
def load_data(f, batch_size):
    df = []
    for line in f:
        row = ujson.loads(line.replace('\n', ''))
        df.append(row)
        if len(df) == batch_size: break
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

# Sorting DataFrame Columns
def sort_dataframe_columns(df):
    sorted_columns = sorted(list(df.columns), reverse=False)
    df = df[sorted_columns]
    return df

# Running Classifier
def run_classifier(df_train_x, df_train_y):
    print('running classifier...')
    print('train sample size: ' + str(len(df_train_y)))
    #clf = RandomForestClassifier(n_estimators=500, random_state=123)
    clf = ExtraTreesClassifier(n_estimators=50, random_state=123)
    clf.fit(df_train_x, df_train_y)
    pickle.dump(clf, open('model.p', 'wb'))
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

# Loading Classifier Model
def load_classifier():
    clf = pickle.load(open("model.p", "rb" ))
    return clf

# Generating Predictions With Classifier
def generate_predictions(clf, df_x, df_y):
    predictions = []
    ys = df_y['outcome'].tolist()
    pred_probs = clf.predict_proba(df_x)
    for i in range(0,len(pred_probs)):
        score = max(pred_probs[i])
        pred_class = list(pred_probs[i]).index(score)
        correct = 1 if int(ys[i]) == pred_class else 0
        predictions.append({'score':score, 'correct':correct, 'true_class':int(ys[i]), 'pred_class':pred_class})
    return predictions

# Isolate Top Predictions
def isolate_top_predictions(predictions, top_preds_batch, prediction_meta, top_n=5000):
    for x in predictions:
        prediction_meta['total_class_' + str(x['true_class'])] += 1
        if x['pred_class'] == 1: top_preds_batch.append(x)
    top_preds_batch = sorted(top_preds_batch, key=lambda k: k['score'], reverse=True)
    top_preds_batch = top_preds_batch[0:top_n]
    return top_preds_batch

# Evaluate Predictions
def evaluate_predictions(top_preds_batch, prediction_meta):
    total_number_samples_scored = prediction_meta['total_class_0'] + prediction_meta['total_class_1']
    random_accuracy = float(prediction_meta['total_class_1']) / float(total_number_samples_scored)
    min_threshold = 1000; num_points = 50
    increment = int((len(top_preds_batch) - min_threshold) / float(num_points))
    xs = []; ys = []
    for i in range(0,num_points+1):
        index = min_threshold + increment * i
        top_batch = top_preds_batch[0:index]
        num_correct_top_preds_batch = float(sum([ x['correct'] for x in top_batch ]))
        num_samples_top_preds_batch = float(len(top_batch))
        accuracy_top_preds_batch = num_correct_top_preds_batch / num_samples_top_preds_batch
        xs.append(index)
        ys.append(accuracy_top_preds_batch)
        if i == num_points:
            print('Total number scored samples: ' + str(total_number_samples_scored))
            print('Sample size top_preds_batch: ' + str(num_samples_top_preds_batch))
            print('Accuracy top_preds_batch: ' + str(accuracy_top_preds_batch))
            print('Random class 1 accuracy: ' + str(random_accuracy))
    f, ax = plt.subplots(1)
    plt.scatter(xs, ys)
    plt.plot(xs, ys, '-')
    plt.plot([min(xs),max(xs)], [random_accuracy, random_accuracy], '-', c='r')
    ax.set_ylim(ymin=0)
    plt.xlabel("Top scored set size")
    plt.ylabel('Class 1 accuracy')
    plt.savefig('top_scored_accuracy.png')
    plt.show()
    return 

############
##  MAIN  ##
############

print('\nstart time: ' + str(datetime.now()))
#dbu = DBUtil("jjill_redshift","C:\Users\Terry\Desktop\KT_GitHub\databases\databases.conf")

arg_options = ['train', 'eval']
error_msg = 'script requires 1 argument: {' + '|'.join(arg_options) + '}'
if len(sys.argv) != 2: print(error_msg); sys.exit()
if sys.argv[1] not in arg_options: print(error_msg); sys.exit()
mode = sys.argv[1]; batch_size = 50000 #100000

prediction_meta = {'total_class_0':0, 'total_class_1':0}
if mode == 'eval': clf = load_classifier()

f = open(mode + '_data.txt', 'r')
c = 0; top_preds_batch = []
while True:
    df = load_data(f, batch_size)
    c += len(df)

    if mode == 'train':
        df = randomize_df(df)
        df_train, df_eval = split_to_train_eval(df)
        df_train = class_balance_df(df_train)
        df_eval = class_balance_df(df_eval)
        df_train_x, df_train_y, df_train_context = split_to_inputs_outputs(df_train)
        df_eval_x, df_eval_y, df_eval_context = split_to_inputs_outputs(df_eval)
        df_train_x = sort_dataframe_columns(df_train_x)
        df_eval_x = sort_dataframe_columns(df_eval_x)
        clf = run_classifier(df_train_x, df_train_y)
        evaluate_model(clf, df_eval_x, df_eval_y)

    if mode == 'eval':
        df_x, df_y, df_context = split_to_inputs_outputs(df)
        df_x = sort_dataframe_columns(df_x)
        predictions = generate_predictions(clf, df_x, df_y)
        top_preds_batch = isolate_top_predictions(predictions, top_preds_batch, prediction_meta, top_n=50000)

    if (len(df) != batch_size) or (mode == 'train'): break
    else: print('processed rows: ' + str(c))

f.close()
if mode == 'eval': evaluate_predictions(top_preds_batch, prediction_meta)
print('end time: ' + str(datetime.now()) + '\n')
