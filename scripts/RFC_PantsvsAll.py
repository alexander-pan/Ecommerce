import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from datetime import datetime as dt, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV,ShuffleSplit
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler, Normalizer
from sklearn.metrics import confusion_matrix, roc_curve, auc

#modules for specific application
import random
import dill
import ModelEvaluation as me

DF = pd.read_pickle('../data/CustomerTime_May2017_Apri2018_Dataset.pkl')
print 'Total Size of Dataset: ', DF.shape[0]

print 'Getting Fts Columns'
cols = DF.columns
cols_X = cols[2:4].tolist() + cols[4:-1].tolist()
cols_y = 'BOUGHT_PANTS'

print 'Building Train and Test'
print 'Training data is May 2017-Mar 2018'
print 'Test Data is April 2018'
TrainDF = DF[DF.MONTH!=4]
TestDF = DF[DF.MONTH==4]

X_train = TrainDF[cols_X]
y_train = TrainDF[cols_y]

X_test = TestDF[cols_X]
y_test = TestDF[cols_y]
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=42)

#Using Training data, to make validation set. 60-20-20 (training, valid, test)
cv = ShuffleSplit(n_splits=50,test_size=.20,random_state=42)

print 'Random Forest Classifier Fit %s' % cols_y
pipe = Pipeline([('rfc',RandomForestClassifier(n_estimators=100,
                                               max_features='auto'))])

params = [{'rfc__max_depth': [7]}]
grid = GridSearchCV(estimator=pipe,
                    param_grid=params,
                    scoring='roc_auc',
                    cv = cv,
                    n_jobs=-1)

grid.fit(X_train,y_train)
print 'Best estimator score:',grid.best_score_
print 'Best estimator params:',grid.best_params_
print ''
dill.settings['recurse']=True
with open('../models/RFC_PantsVsAll_%s.pkl' % cols_y,'wb') as outfile:
    dill.dump(grid,outfile)

me.ModelEvalClassifier(grid,X_train,X_test,y_train,y_test)
print ''
