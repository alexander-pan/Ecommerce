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
import json
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
        row = json.loads(line.replace('\n', ''))
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
def generate_predictions(clf, df_x, df_y, df_context):
    predictions = {}
    ys = df_y['outcome'].tolist()
    cs_ilink = df_context['ilink'].tolist()
    cs_department = df_context['department_name'].tolist()
    pred_probs = clf.predict_proba(df_x)
    for i in range(0,len(pred_probs)):
        score = max(pred_probs[i])
        pred_class = list(pred_probs[i]).index(score)
        correct = 1 if int(ys[i]) == pred_class else 0
        value = {
            'score': score, 
            'correct': correct, 
            'true_class': int(ys[i]), 
            'pred_class': pred_class,
            'ilink': cs_ilink[i],
            'department_name': cs_department[i]
        }
        key = cs_ilink[i] + ':' + cs_department[i]
        predictions[key] = value
    return predictions

# Isolate Top Predictions
def isolate_top_predictions(predictions, top_preds_batch, top_preds_1p_user_batch, prediction_meta, top_n=5000):
    for x in predictions:
        ilink, department_name = x.split(':')
        pred_dict = predictions[x]
        prediction_meta['total_class_' + str(pred_dict['true_class']) + '_all'] += 1
        prediction_meta['total_class_' + str(pred_dict['true_class']) + '_' + pred_dict['department_name']] += 1
        prediction_meta['users'][ilink] = 1
        if pred_dict['pred_class'] == 1: 
            top_preds_batch['all'].append(pred_dict)
            top_preds_batch[pred_dict['department_name']].append(pred_dict)
            if ilink in top_preds_1p_user_batch:
                if pred_dict['score'] > top_preds_1p_user_batch[ilink]['score']:
                    top_preds_1p_user_batch[ilink] = pred_dict
            else:
                top_preds_1p_user_batch[ilink] = pred_dict
    for key in top_preds_batch:
        top_preds_batch[key] = sorted(top_preds_batch[key], key=lambda k: k['score'], reverse=True)
        top_preds_batch[key] = top_preds_batch[key][0:top_n]
    top_preds_1p_user_batch = sorted(top_preds_1p_user_batch.items(), key=lambda x: x[1]['score'], reverse=True)
    top_preds_1p_user_batch = top_preds_1p_user_batch[0:top_n]
    top_preds_1p_user_batch = { top_preds_1p_user_batch[i][0]:top_preds_1p_user_batch[i][1] for i in range(0,len(top_preds_1p_user_batch)) }
    return top_preds_batch, top_preds_1p_user_batch

# Evaluate Predictions
def evaluate_predictions(top_preds_batch, top_preds_1p_user_batch, prediction_meta, top_n):
    top_preds_1p_user_batch = sorted(top_preds_1p_user_batch.items(), key=lambda x: x[1]['score'], reverse=True)
    top_preds_1p_user_batch = [ top_preds_1p_user_batch[i][1] for i in range(0,len(top_preds_1p_user_batch)) ]
    total_number_samples_scored = {}; random_accuracy = {}
    for key in top_preds_batch:
        key_alt = key.lower().replace(' ','_')
        total_number_samples_scored[key_alt] = prediction_meta['total_class_0_' + key] + prediction_meta['total_class_1_' + key]
        random_accuracy[key_alt] = float(prediction_meta['total_class_1_' + key]) / float(total_number_samples_scored[key_alt])
    batches = { ('top_preds_' + key).lower().replace(' ','_'):top_preds_batch[key] for key in top_preds_batch }
    #batches['top_preds_1p_user'] = top_preds_1p_user_batch
    for batch_name in batches:
        key_alt_name = batch_name.replace('top_preds_', '')
        batch = batches[batch_name]
        min_threshold = 0; num_points = 50
        increment = int((len(batch) - min_threshold) / float(num_points))
        samples_scored = total_number_samples_scored[key_alt_name] if batch_name != 'top_preds_1p_user' else len(prediction_meta['users'])
        rand_accuracy = random_accuracy[key_alt_name] if batch_name != 'top_preds_1p_user' else random_accuracy
        xs = []; ys = []
        for i in range(0,num_points+1):
            index = min_threshold + increment * i
            top_batch = batch[0:index]
            num_correct_batch = float(sum([ x['correct'] for x in top_batch ]))
            num_samples_batch = float(len(top_batch))
            accuracy_batch = num_correct_batch / num_samples_batch if num_samples_batch > 0 else 0.0
            xs.append(index)
            ys.append(accuracy_batch)
            if i == num_points:
                print('\nBatch name: ' + batch_name)
                print('Total number scored samples: ' + str(samples_scored))
                print('Sample size top_preds_batch: ' + str(num_samples_batch))
                print('Accuracy top_preds_batch: ' + str(accuracy_batch))
                print('Random class 1 accuracy: ' + str(rand_accuracy))
        pcts = [ '%.1f'%(100*(xs[i] / float(samples_scored))) + '%' for i in range(0,len(xs)) ]
        f, ax = plt.subplots(1)
        plt.scatter(xs[1:], ys[1:])
        plt.plot(xs[1:], ys[1:], '-')
        plt.plot([min(xs),max(xs)], [rand_accuracy, rand_accuracy], '-', c='r')
        ax.set_ylim(ymin=0)
        x_text_inds = [ xs[i] for i in range(0,len(xs)) if i % 10 == 0 ]
        x_text_vals = [ pcts[i] for i in range(0,len(pcts)) if i % 10 == 0 ]
        plt.xticks(x_text_inds, x_text_inds)
        for i in range(0,len(x_text_inds)): ax.text(x_text_inds[i], -0.079, x_text_vals[i], size=6, ha='center')
        plt.xlabel("Top scored set size", labelpad=15)
        plt.ylabel('Class 1 accuracy')
        title = 'top_scored_accuracy_' + batch_name
        plt.title(title)
        f.subplots_adjust(bottom=0.22)
        plt.savefig(title + '.png')
        plt.show()

        if batch_name == 'top_preds_1p_user':
            for i in range(0,top_n):
                if i == 0: pairs_file = open('top_scored_pairs_' + batch_name + '.csv', 'w')
                ilink = str(batch[i]['ilink'])
                ilink = ''.join((10 - len(ilink)) * ['0']) + ilink
                pairs_file.write(ilink + ',' + str(batch[i]['department_name']) + ',' + str(batch[i]['score']) + '\n')
            pairs_file.close()

    return

# Storing Predictions
def store_predictions(predictions, f_out):
    for x in predictions:
        d = predictions[x]
        output = ','.join([d['ilink'], d['department_name'], str(d['score'])])
        f_out.write(output + '\n')
    return 

############
##  MAIN  ##
############

print('\nstart time: ' + str(datetime.now()))
arg_options = ['train', 'eval', 'production']
error_msg = 'script requires 1 argument: {' + '|'.join(arg_options) + '}'
if len(sys.argv) != 2: print(error_msg); sys.exit()
if sys.argv[1] not in arg_options: print(error_msg); sys.exit()
mode = sys.argv[1]; batch_size = 50000; top_n = 50000
if mode != 'train': clf = load_classifier()
if mode == 'production': f_out = open('production_predictions.txt', 'w')

top_preds_batch = {'all':[], 'Pants':[], 'Dresses':[], 'Woven Shirts':[], 'Knit Tops':[]}; 
top_preds_1p_user_batch = {}
prediction_meta = ['total_class_0', 'total_class_1']
prediction_meta = { (x + '_' + y):0 for x in prediction_meta for y in top_preds_batch }
prediction_meta['users'] = {} 
f = open(mode + '_data.txt', 'r')
c = 0
while True:
    df = load_data(f, batch_size)
    c += len(df)

    # ###
    # temp_df_sample = df.sample(n=100)
    # temp_df_sample.to_csv('eval_data_sample.csv')
    # sys.exit()
    # ###

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
        predictions = generate_predictions(clf, df_x, df_y, df_context)
        top_preds_batch, top_preds_1p_user_batch = isolate_top_predictions(predictions, top_preds_batch, top_preds_1p_user_batch, prediction_meta, top_n=top_n)

    if mode == 'production':
        df_x, df_y, df_context = split_to_inputs_outputs(df)
        df_x = sort_dataframe_columns(df_x)
        predictions = generate_predictions(clf, df_x, df_y, df_context)
        store_predictions(predictions, f_out)

    print('processed rows: ' + str(c))
    if (len(df) != batch_size) or (mode == 'train'): break

f.close()
if mode == 'eval': evaluate_predictions(top_preds_batch, top_preds_1p_user_batch, prediction_meta, top_n)
print('end time: ' + str(datetime.now()) + '\n')
