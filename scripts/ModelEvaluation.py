import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from datetime import datetime as dt, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV,ShuffleSplit
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler, Normalizer
from sklearn.metrics import confusion_matrix, roc_curve, auc

def ModelEvalClassifier(grid,X_train,X_test,y_train,y_test):
    #X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=42)

    #Using Confusion matrix to test accuracy
    i = grid.best_index_
    test_mean = grid.cv_results_['mean_test_score'][i] * 100
    std_test = grid.cv_results_['std_test_score'][i] * 100
    train_mean = grid.cv_results_['mean_train_score'][i] * 100
    std_train = grid.cv_results_['std_train_score'][i] * 100
    print("Validation Score Accuracy: %.2f%% +/- %.2f%%" % (test_mean,std_test))
    print("Train Score Accuracy: %.2f%% +/- %.2f%%" % (train_mean,std_train))

    y_pred = grid.predict(X_test)
    confmat = confusion_matrix(y_test,y_pred,labels=[1,0])

    fig,ax = plt.subplots(figsize=(2.5,2.5))
    ax.matshow(confmat,cmap=plt.cm.Blues,alpha=.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j,y=i,
                    s=confmat[i,j],
                    va='center',ha='center')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()

    total = float(X_test.shape[0])
    TN = float(confmat[0][0])
    TP = float(confmat[1][1])
    FP = float(confmat[0][1])
    FN = float(confmat[1][0])
    try:
        print("Accuracy: %.2f" % ((TP+TN)/total))
        print("Missclassification Rate: %.2f" % ((FN+FP)/total))
        print("(1) Positive Rate(Recall): %.2f" % (TP/(TP+FN)))
        print("(0) Positive Rate(Sensitivity): %.2f" % (TN/(TN+FP)))
        print("Precision(proportion of predicted (1)): %.2f" % (TP/(TP+FP)))
        print("Specificity(proportion of predicted (0): %.2f" % (TN/(TN+FN)))
    except:
        pass

    #ROC Curve
    probas = grid.predict_proba(X_test)
    fpr,tpr, thresholds = roc_curve(y_test, probas[:,1])
    plt.plot(fpr,tpr)
    plt.show()
    print'AUC(Area Under Curve) score: %f' % auc(fpr,tpr)
