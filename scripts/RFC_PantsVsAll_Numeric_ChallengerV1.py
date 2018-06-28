import numpy as np
import pandas as pd
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

DF = pd.read_pickle('../data/numericalFts_100Users_ChallengerV1.pkl')

print 'Total Size of Dataset: %d' % DF.shape[0]
print 'Total Unique Users: %d\n\n' % len(DF.ILINK.unique().tolist())

X_features = ['REF_DATE_MONTH',
             'REF_DATE_YEAR',
             'NUM_PAST_ORDERS',
             'SUM_PAST_SHIPPED_SOLD_AMT',
             'AVG_PAST_SHIPPED_SOLD_AMT',
             'STDDEV_PAST_SHIPPED_SOLD_AMT',
             'VAR_PAST_SHIPPED_SOLD_AMT',
             'SUM_PAST_DISCOUNT',
             'AVG_PAST_DISCOUNT',
             'STDDEV_PAST_DISCOUNT',
             'VAR_PAST_DISCOUNT']
dependent = 'BOUGHT_PANTS'

print 'Input features: \n%s\n' % X_features
print 'Dependent feature: %s\n\n' % dependent
X = DF[X_features]
y = DF[dependent]

print 'Building Train and Test...'
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=42)

#Using Training data, to make validation set. 60-20-20 (training, valid, test)
cv = ShuffleSplit(n_splits=50,test_size=.20,random_state=42)

print 'Random Forest Classifier Fitting...'
pipe = Pipeline([('rfc',RandomForestClassifier(n_estimators=100,
                                               max_features='auto'))])

params = [{'rfc__max_depth': [1,2,3,4,5,6,7,8,9,10,None]}]
grid = GridSearchCV(estimator=pipe,
                    param_grid=params,
                    scoring='roc_auc',
                    cv = cv,
                    n_jobs=-1)

grid.fit(X_train,y_train)
print 'Best estimator score:',grid.best_score_
print 'Best estimator params:',grid.best_params_
print ''

name = 'ChallengerV1_Numeric'
dill.settings['recurse']=True
with open('../models/RFC_PantsVsAll_%s.pkl' % name,'wb') as outfile:
    dill.dump(grid,outfile)

#me.ModelEvalClassifier(grid,X_train,X_test,y_train,y_test)
#print ''
