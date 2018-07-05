import numpy as np
import pandas as pd
import math
from datetime import datetime as dt, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV,ShuffleSplit
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler, Normalizer
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, auc

#modules for specific application
import random
import dill
#import ModelEvaluation as me

#DF = pd.read_pickle('../data/numericalFts_10kUsers_BaselineV1.pkl')
DF = pd.read_pickle('../data/Customer_Dataset_15k_Baseline.pkl')
#DF = pd.read_pickle('../data/trial.pkl')
print 'Total Size of Dataset: %d' % DF.shape[0]
print 'Total Unique Users: %d\n\n' % len(DF.ILINK.unique().tolist())

X_features = ['REF_DATE_MONTH', 'REF_DATE_DAY', 'REF_DATE_YEAR', 'NUM_PAST_ORDERS',
 'SUM_PAST_SHIPPED_SOLD_AMT', 'AVG_PAST_SHIPPED_SOLD_AMT', 'STDDEV_PAST_SHIPPED_SOLD_AMT',
 'VAR_PAST_SHIPPED_SOLD_AMT', 'SUM_PAST_DISCOUNT', 'AVG_PAST_DISCOUNT',
 'STDDEV_PAST_DISCOUNT', 'VAR_PAST_DISCOUNT', 'HAS_CORE', 'HAS_PURE_JILL',
 'HAS_WEAREVER', 'HAS_FP', 'HAS_SP', 'HAS_CASH', 'HAS_CK', 'HAS_DEBIT',
 'HAS_DISC', 'HAS_JJC', 'HAS_MC', 'HAS_OTHER_PAYTYPE', 'HAS_VISA',
 'HAS_D', 'HAS_R', 'HAS_COTTON/COTTON_BL', 'HAS_LINEN/LINEN_BL', 'HAS_OTHER_FABRIC',
 'HAS_SYNTHETIC/SYN_BLEND', '%_IN_CORE', '%_IN_PURE_JILL', '%_IN_WEAREVER',
 '%_IN_FP', '%_IN_SP', '%_IN_CASH', '%_IN_CK', '%_IN_DEBIT', '%_IN_DISC', '%_IN_JJC',
 '%_IN_MC', '%_IN_OTHER_PAYTYPE', '%_IN_VISA', '%_IN_D', '%_IN_R', '%_IN_COTTON/COTTON_BL',
 '%_IN_LINEN/LINEN_BL', '%_IN_OTHER_FABRIC', '%_IN_SYNTHETIC/SYN_BLEND']
dependent = 'BOUGHT_PANTS'

print 'Input features: \n%s\n' % X_features
print 'Dependent feature: %s\n\n' % dependent
#X = DF[X_features]
#y = DF[dependent]

#getting 80% of customers for training and 20% customers left for testing
print 'Building Train and Test...'
train = DF.ILINK.unique().tolist()[:11953]
test = DF.ILINK.unique().tolist()[11953:]
DFTrain = DF[DF.ILINK.isin(train)]
DFTest = DF[DF.ILINK.isin(test)]

X_train = DFTrain[X_features]
y_train = DFTrain[dependent]

X_test = DFTest[X_features]
y_test = DFTest[dependent]
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=42)

#Using Training data, to make validation set. 60-20-20 (training, valid, test)
cv = ShuffleSplit(n_splits=50,test_size=.25,random_state=42)

print 'For test set,'
print 'Number of negatives (0): %d '  % y_test.value_counts()[0]
print 'Number of positives (1): %d '  % y_test.value_counts()[1]

print 'Random Forest Classifier Fitting...'
pipe = Pipeline([('rfc',RandomForestClassifier(n_estimators=100,
                                               max_features='auto',
                                               class_weight='balanced'))])

params = [{'rfc__max_depth': [None]}]
grid = GridSearchCV(estimator=pipe,
                    param_grid=params,
                    scoring='roc_auc',
                    cv = cv,
                    n_jobs=-1)

grid.fit(X_train,y_train)
print 'Best estimator score:',grid.best_score_
print 'Best estimator params:',grid.best_params_
print ''

name = 'BaselineV1_15k'
dill.settings['recurse']=True
with open('../models/RFC_PantsVsAll_%s.pkl' % name,'wb') as outfile:
    dill.dump(grid,outfile)

pred = grid.predict(X_test)
print(classification_report(y_test,pred))
