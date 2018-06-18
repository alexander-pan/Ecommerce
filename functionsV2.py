import numpy as np
import pandas as pd
import math
from datetime import datetime as dt, timedelta
from sklearn import preprocessing

#function to get dataframes/datasets based on start,end date
#dates are str type in format example: '2017-5-1' or 'year-month-day'
def getIlinkDF(dataframe,ilink,start,end):
    start_date = dt.strptime(start,'%Y-%m-%d').date()
    end_date = dt.strptime(end,'%Y-%m-%d').date()
    return dataframe[(dataframe.ILINK==ilink) & (dataframe.ORDER_DATE>=start) & (dataframe.ORDER_DATE < end)]

#Get Avg. Cost per Order
#Use specific table that is aggregated by ilink,order_date
def getAvgCostOrder(dataframe,ilink):
    value = dataframe[dataframe.ILINK==ilink].TOTAL_SSA.mean()
    return round(value,2)

#Builds column headers for categorical features "Has_x"
def BuildHasColumn(df,category):
    feature = category.replace('_','')
    if 'DESC' in category:
        feature = feature.replace('DESC','')

    if 'CD' in category:
        feature = feature.replace('CD','')

    if category == 'END_USE_DESC':
        cats = df[category].unique().tolist()
        if 'Does Not Apply' in cats:
            cats.remove('Does Not Apply')
    elif category == 'PAY_TYPE_CD':
        cats = ['JJC','VISA','MC','AMEX','DISC','DEBIT','CASH','CK','OTHER']
    else:
        cats = df[category].unique().tolist()
    fts = []
    for cat in cats:
        fts.append('%s_Has_%s' % (feature,cat.replace(' ','')))
    return fts

#Builds column headers for categorical feature "%_x"
def BuildPercentColumn(df,category):
    feature = category.replace('_','')
    if 'DESC' in category:
        feature = feature.replace('DESC','')

    if 'CD' in category:
        feature = feature.replace('CD','')

    if category == 'END_USE_DESC':
        cats = df[category].unique().tolist()
        if 'Does Not Apply' in cats:
            cats.remove('Does Not Apply')
    elif category == 'PAY_TYPE_CD':
        cats = ['JJC','VISA','MC','AMEX','DISC','DEBIT','CASH','CK','OTHER']
    else:
        cats = df[category].unique().tolist()
    fts = []
    for cat in cats:
        fts.append('%s_%%_%s'%(feature,cat.replace(' ','')))
    return fts

#Gets the values for "Has_x"
def HasPurchased(df,category,fts):
    if category != 'PAY_TYPE_CD':
        cats = df[category].unique().tolist()
        #print cats
        cats = [x.replace(' ','') for x in cats]
        cust = {}
        for ft in fts:
            ftcat,suf,cat = ft.split('_')
            if cat in cats:
                cust[ft] = 1
            else:
                cust[ft] = 0
        return cust
    else:
        cats = df[category].unique().tolist()
        cats2 = ['JJC','VISA','MC','AMEX','DISC','DEBIT','CASH','CK']
        cats2 = list(set(cats) & set(cats2))
        other = ['GCRD','AGC','','MALL', 'EXCH', 'TVCK', 'IHA', 'STCK']
        other = list(set(cats) & set(other))
        #print cats
        cust = {}
        for ft in fts:
            ftcat,suf,cat = ft.split('_')
            if cat != 'OTHER':
                if cat in cats2:
                    cust[ft] = 1
                else:
                    cust[ft] = 0
            else:
                if len(other) > 0:
                    cust[ft] = 1
                else:
                    cust[ft] = 0
        return cust

#Calc and aggregates valeus for "%_x"
def PercentPurchased(dataframe,category,fts):
    if category != 'PAY_TYPE_CD':
        cust = {}
        for ft in fts:
            ftcat,suf,cat = ft.split('_')
            X = dataframe.groupby(category).size().to_dict()
            total = sum(X.values())
            for key,value in X.iteritems():
                x = key.replace(' ','')
                X[x] = X.pop(key)
            if cat in X:
                cust[ft] = X[cat]*1.0/total
            else:
                cust[ft] = 0.0
        return cust
    else:
        cats = dataframe[category].unique().tolist()
        cats2 = ['JJC','VISA','MC','AMEX','DISC','DEBIT','CASH','CK']
        cats2 = list(set(cats) & set(cats2))
        other = ['GCRD','AGC','','MALL', 'EXCH', 'TVCK', 'IHA', 'STCK']
        other = list(set(cats) & set(other))
        cust = {}
        for ft in fts:
            ftcat,suf,cat = ft.split('_')
            X = dataframe.groupby(category).size().to_dict()
            total = sum(X.values())
            for key,value in X.iteritems():
                x = key.replace(' ','')
                X[x] = X.pop(key)
            if cat != 'OTHER':
                if cat in cats2:
                    for key,value in X.iteritems():
                        x = key.replace(' ','')
                        X[x] = X.pop(key)
                    cust[ft] = X[cat]*1.0/total
                else:
                    cust[ft] = 0.0
            else:
                if len(other) > 0:
                    other_total = 0.0
                    for paytype in other:
                        other_total += X[paytype]
                    cust[ft] = other_total/total
                else:
                    cust[ft] = 0.0
        return cust
