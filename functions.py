import numpy as np

#convert Ilink
def convertIlink(ilink):
    s = str(ilink)
    r = 10-len(s)
    return '0'*r + s

#get frequency totals for specific departments: woven shirts, dresses, knit tops, and pants
def getTotalFreq(dataframe,dept):
    x = dataframe.groupby('DEPARTMENT_NAME').size().to_dict()
    if dept in x:
        return x[dept]
    else:
        return 0
    
#get avg. freq per order
def getAvgFreqOrder(dataframe,dept):
    orders = dataframe.ORDER_KEY.unique().tolist()
    items =[]
    for order in orders:
        temp = dataframe[dataframe['ORDER_KEY']==order]
        x = temp.groupby('DEPARTMENT_NAME').size().to_dict()
        if dept in x:
            items.append(x[dept])
        else:
            items.append(0)
    return np.mean(items).round()

#Get Total Costs for specific depts
def getTotalAmts(dataframe,dept):
    temp = dataframe[dataframe.DEPARTMENT_NAME==dept]
    orig_retail = temp.ORIGINAL_RETAIL_PRICE_AMT.sum().round(2)
    goods = temp.SHIPPED_COST_AMT.sum().round(2)
    gross = temp.SHIPPED_GROSS_AMT.sum().round(2)
    sold = temp.SHIPPED_SOLD_AMT.sum().round(2)
    discount = temp.discount.sum().round(2)
    return orig_retail,goods,gross,sold,discount

#Get Avg Costs per order for depts
def getAvgAmtsOrder(dataframe,dept):
    orders = dataframe.ORDER_KEY.unique().tolist()
    orig=[]
    goods=[]
    gross=[]
    sold=[]
    discount = []
    for order in orders:
        temp = dataframe[(dataframe.DEPARTMENT_NAME==dept) & (dataframe.ORDER_KEY==order)]
        orig.append(temp.ORIGINAL_RETAIL_PRICE_AMT.sum().round(2))
        goods.append(temp.SHIPPED_COST_AMT.sum().round(2))
        gross.append(temp.SHIPPED_GROSS_AMT.sum().round(2))
        sold.append(temp.SHIPPED_SOLD_AMT.sum().round(2))
        discount.append(temp.discount.sum().round(2))
    return np.mean(orig).round(2), np.mean(goods).round(2), np.mean(gross).round(2), np.mean(sold).round(2),np.mean(discount).round(2)

#Get Avg. Amount for Dept Item that this customer bought
def getAvgAmtItem(dataframe,dept):
    temp = dataframe[dataframe.DEPARTMENT_NAME==dept]
    orig_retail = round(temp.ORIGINAL_RETAIL_PRICE_AMT.mean(),2)
    goods = round(temp.SHIPPED_COST_AMT.mean(),2)
    gross = round(temp.SHIPPED_GROSS_AMT.mean(),2)
    sold = round(temp.SHIPPED_SOLD_AMT.mean(),2)
    margin = round(temp.margin.mean(),2)
    discount = round(temp.discount.mean(),2)
    markdown = round(temp.markdown.mean(),2)
    discount_plus = round(temp.discount_plus.mean(),2)
    return orig_retail,goods,gross,sold,discount#,margin,discount,markdown,discount_plus

#getAge
def getAge(dataframe):
    birth = dataframe.BIRTH_DATE.unique()[0]
    return (dt.now().date() - birth).days/365

#get Most/Least key,value,percent of items of a category
#MASTER_CHANNEL,PAY_TYPE_CD,EXTENDED_SIZE_DESC,FABRIC_CATEGORY_DESC,COLOR_FAMILY_DESC,
#STYLETYPEDESC,CLASS_NAME
def getMostCategory(dataframe,category):
    temp = dataframe.groupby(category).size().to_dict()
    max_key = max(temp,key=temp.get)
    percent = temp[max_key]*1.0/sum(temp.values())
    return max_key,temp[max_key],percent

def getLeastCategory(dataframe,category):
    temp = dataframe.groupby(category).size().to_dict()
    max_key = min(temp,key=temp.get)
    percent = temp[max_key]*1.0/sum(temp.values())
    return max_key,temp[max_key],percent

#get jj private membership length if member returns (years,status)
def getLengthMembership(dataframe):
    if dataframe.JJCH_OPEN_DATE.isnull().all():
        return None,'No'
    elif (~dataframe.JJCH_OPEN_DATE.isnull().all()) and (~dataframe.JJCH_CLOSE_DATE.isnull().any()):
        openDate = dt.strptime(dataframe.JJCH_OPEN_DATE.unique()[0],'%Y-%m-%d').date()
        closeDate = dt.strptime(dataframe.JJCH_CLOSE_DATE.unique()[0],'%Y-%m-%d').date()
        length = (closeDate-openDate).days/365
        return length,'Closed'
    elif(~dataframe.JJCH_OPEN_DATE.isnull().all()) and (dataframe.JJCH_CLOSE_DATE.isnull().all()):
        openDate = dt.strptime(dataframe.JJCH_OPEN_DATE.unique()[0],'%Y-%m-%d').date()
        length = (dt.now().date()-openDate).days/365
        return length,'Current'
    
#get END_USE stats
#get frequency totals 
def getTotalFreqEndUse(dataframe):
    x = dataframe.groupby('END_USE_DESC').size().to_dict()
    x = x.to_dict()
    core = 0
    pure = 0
    wearever = 0
    if 'Core' in x:
        core = x['Core']
    if 'Knit Tops' in x:
        pure = x['Pure Jill']
    if 'Wearever' in x:
        wearever = x['Wearever']
    return core,pure,wearever

#get avg. freq per order
def getAvgFreqOrderEndUse(dataframe):
    orders = dataframe.ORDER_KEY.unique().tolist()
    core = []
    pure = []
    wearever = []
    for order in orders:
        temp = dataframe[dataframe['ORDER_KEY']==order]
        x = temp.groupby('DEPARTMENT_NAME').size().to_dict()
        if 'Core' in x:
            core.append(x['Core'])
        else:
            core.append(0)
        if 'Pure Jill' in x:
            pure.append(x['Pure Jill'])
        else:
            pure.append(0)
        if 'Wearever' in x:
            wearever.append(x['Wearever'])
        else:
            wearever.append(0)
    return np.mean(core).round(),np.mean(pure).round(),np.mean(wearever).round()

#Get Total Costs for specific end_use
def getTotalAmtsEndUse(dataframe,end):
    temp = dataframe[dataframe.END_USE_DESC==end]
    orig_retail = temp.ORIGINAL_RETAIL_PRICE_AMT.sum().round(2)
    goods = temp.SHIPPED_COST_AMT.sum().round(2)
    gross = temp.SHIPPED_GROSS_AMT.sum().round(2)
    sold = temp.SHIPPED_SOLD_AMT.sum().round(2)
    return orig_retail,goods,gross,sold

#Get Avg Costs per order for enduse
def getAvgAmtsOrderEndUse(dataframe,end):
    orders = dataframe.ORDER_KEY.unique().tolist()
    orig=[]
    goods=[]
    gross=[]
    sold=[]
    for order in orders:
        temp = dataframe[(dataframe.END_USE_DESC==end) & (dataframe.ORDER_KEY==order)]
        orig.append(temp.ORIGINAL_RETAIL_PRICE_AMT.sum().round(2))
        goods.append(temp.SHIPPED_COST_AMT.sum().round(2))
        gross.append(temp.SHIPPED_GROSS_AMT.sum().round(2))
        sold = temp.SHIPPED_SOLD_AMT.sum().round(2)
    return np.mean(orig).round(2), np.mean(goods).round(2), np.mean(gross).round(2), np.mean(sold).round(2)

#Get Avg. Amount for Dept Item that this customer bought
def getAvgAmtEndUse(dataframe,end):
    temp = dataframe[dataframe.END_USE_DESC==end]
    orig_retail = round(temp.ORIGINAL_RETAIL_PRICE_AMT.mean(),2)
    goods = round(temp.SHIPPED_COST_AMT.mean(),2)
    gross = round(temp.SHIPPED_GROSS_AMT.mean(),2)
    sold = round(temp.SHIPPED_SOLD_AMT.mean(),2)
    margin = round(temp.margin.mean(),2)
    discount = round(temp.discount.mean(),2)
    markdown = round(temp.markdown.mean(),2)
    discount_plus = round(temp.discount_plus.mean(),2)
    return orig_retail,goods,gross,sold,margin,discount,markdown,discount_plus

#customer modified timeline value
def AvgMonthlyTrans(dataframe):
    dates = zip(dataframe.ORDER_DATE.dt.month,dataframe.ORDER_DATE.dt.year)
    monthly = {}
    for i,row in dataframe.iterrows():
        month = row.ORDER_DATE.month
        year = row.ORDER_DATE.year
        if (month,year) in monthly:
            monthly[(month,year)] += row.SHIPPED_SOLD_AMT
        else:
            monthly[(month,year)] = row.SHIPPED_SOLD_AMT
    return np.mean(monthly.values()).round(2)

def AvgOrderValue(dataframe):
    orders = dataframe.ORDER_KEY.unique().tolist()
    order_values = []
    for order in orders:
        temp = dataframe.loc[dataframe.ORDER_KEY==order]
        order_values.append(temp.SHIPPED_SOLD_AMT.sum().round(2))
    return np.mean(order_values).round(2)

def AvgGrossMargin(dataframe):
    sold = dataframe.SHIPPED_SOLD_AMT
    costs = dataframe.SHIPPED_COST_AMT
    gross_margin = (sold-costs)/sold
    return gross_margin.mean()
"""
def AvgLifeSpan(dataframe):
    years = [2014,2015,2016,2017]
    n = 1
    alt = 0
    for year in years:
        temp = dataframe[dataframe.ORDER_DATE.dt.year == year]
        N = len(temp.ORDER_DATE.dt.month.unique())
        alt += N#*n
        #print N,n,alt
        if N > 0 & year != 2014:
            n += 1
        else:
            n = 1
    return alt

def CLV(dataframe):
    T = AvgMonthlyTrans(dataframe)
    aov = AvgOrderValue(dataframe)
    alt = AvgLifeSpan(dataframe)
    agm = AvgGrossMargin(dataframe)
    return (T*aov)*agm*alt
"""

def getTableRating(ilinks,dataframe,cols):
    """
    cols = ['Ilink','N_ws','n_ws','S_ws','s_ws','si_ws','D_ws','d_ws','di_ws','R_ws',
        'N_kt','n_kt','S_kt','s_kt','si_kt','D_kt','d_kt','di_kt','R_kt',
        'N_d','n_d','S_d','s_d','si_d','D_d','d_d','di_d','R_d',
        'N_p','n_p','S_p','s_p','si_p','D_p','d_p','di_p','R_p']
    """
    tab = pd.DataFrame([],columns=cols)
    #need to create rows that describe a customers freq and sales of items in specific departments
    depts = ['Woven Shirts','Knit Tops','Dresses','Pants']
    K = 4 #Number of Depts/topics
    for ilink in ilinks:
        dfCust = dataframe.loc[(dataframe.ILINK==ilink)]
        if ~dfCust.empty:
            N_u = dfCust.shape[0]#sum([f.getTotalFreq(dfCust,dept) for dept in depts])
            #print ilink,N_u
            fts = []
            for dept in depts:
                N = f.getTotalFreq(dfCust,dept)
                n = f.getAvgFreqOrder(dfCust,dept)
                O,Q,G,S,D = f.getTotalAmts(dfCust,dept)
                o,q,g,s,d = f.getAvgAmtsOrder(dfCust,dept)
                oi,qi,gi,si,di = f.getAvgAmtItem(dfCust,dept)
                alpha = [N,n,S,s,si,D,d,di]
                #alpha = [S,s,si,D,d,di]
                #alpha = [N,S,s,si]
                alpha = [0 if math.isnan(x) else x for x in alpha]
                alpha1 = np.array(normalize(alpha))
                sigma = UserSigma[ilink][dept]
                #sigma = (N + alpha1)/(N_u + K*alpha1)
                pref = np.dot(sigma,alpha1).round(2)
                #print ilink,alpha
                #print sigma
                #print ''
                fts.append(alpha+[pref])
                #fts2.append(alpha)
        else:
            pass
        row = [i for sub in fts for i in sub]
        row.insert(0,ilink)
        temp = pd.DataFrame([tuple(row)],columns=cols)
        tab = pd.concat([tab,temp])
    return tab

#This function will get the customers highest Rating pref
#It checks the rating first
#if ratings are equal it checks for freq, total sold/sales as tiebreakers in that order
#if that is not a tie-breaker it will return as many dept as "preferred" and consider that user as 
#having multiple preferences
def getHigherPref(row):
    ws = (row.R_ws,row.N_ws,row.S_ws)
    kt = (row.R_kt,row.N_kt,row.S_kt)
    d = (row.R_d,row.N_d,row.S_d)
    p = (row.R_p,row.N_p,row.S_p)
    R = {'Woven Shirts': ws, 'Knit Tops': kt, 'Dresses': d, 'Pants': p}
    maxNum = 0.0
    maxKey = []
    #get those with highest rating first
    for key,value in R.iteritems():
        if value[0] > maxNum:
            maxNum = value[0]
            maxKey.append(key)
        elif value[0] == maxNum and value[0] != 0:
            maxKey.append(key)
    
    #if no purchases were made
    if len(maxKey) == 0:
        return 0
    #if rating are equal, then look at freq purchased for pref
    i = 1
    while len(maxKey) != 1:
        RTop = {x:R[x] for x in maxKey}
        n = 0
        maxKey = []
        for key,value in RTop.iteritems():
            #print value[i]
            if value[i] > n:
                n = value[i]
                if len(maxKey)==0:
                    maxKey.append(key)
                else:
                    maxKey.pop()
                    maxKey.append(key)
            elif value[i] == n and value[i] !=0:
                maxKey.append(key)
        #print maxKey,n
        if len(maxKey) > 1:
            i += 1
            if i > 2:
                return maxKey
    return maxKey[0]
